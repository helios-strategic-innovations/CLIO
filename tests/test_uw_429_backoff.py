"""UW-2 (S211, ADR-005): a 429 must recover inside DPOF's 8 s per-call timeout.

Old behaviour: sleep(60 s) taken from a constant `x-uw-req-per-minute-reset: 60000` header, then
one retry -- but DPOF cancels each call at 8 s, so the retry never ran (0 retry rows in the ledger).
New: short jittered backoff, exactly one paced retry, pacer slows down. No network, no real sleep,
and the live quota ledger is never touched (the quota coordinator is faked).
"""

import asyncio

import pytest

from agent.integrations import uw_flow_intel as uw
from agent.integrations import uw_pacer
from agent.integrations.uw_quota_client import QuotaDecision

RATE_LIMITED_HEADERS = {
    "x-uw-req-per-minute-reset": "60000",          # constant, untruthful (the old code slept on it)
    "x-uw-req-per-minute-remaining": "1000000",
}


class FakeResponse:
    def __init__(self, status_code, data=None, headers=None):
        self.status_code = status_code
        self._data = data or {}
        self.headers = headers if headers is not None else {
            "x-uw-daily-req-count": "100", "x-uw-token-req-limit": "40000",
        }

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def get(self, *_a, **_k):
        self.calls += 1
        return self.responses.pop(0)


class FakeQuota:
    def __init__(self):
        self.reservations = []

    def reserve(self, **kw):
        self.reservations.append(kw)
        return QuotaDecision(allowed=True, request_id=f"00000000-0000-0000-0000-{len(self.reservations):012d}",
                             reason_code="P0_RESERVED", source="supabase", remaining=39000)

    def observe_response(self, decision, **kw):
        return kw

    def release(self, decision, reason):
        return True


@pytest.fixture()
def harness(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(uw, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(uw, "_get_quota_coordinator", lambda: FakeQuota())
    monkeypatch.setattr(uw.asyncio, "sleep", fake_sleep)
    uw_pacer.reset_uw_pacer_for_tests()
    holder = SimpleHolder(sleeps)

    def install(responses):
        client = FakeClient(responses)

        async def get_client():
            return client

        monkeypatch.setattr(uw, "_get_client", get_client)
        holder.client = client
        return client

    holder.install = install
    return holder


class SimpleHolder:
    def __init__(self, sleeps):
        self.sleeps = sleeps
        self.client = None
        self.install = None


def _call():
    return asyncio.run(uw._uw_get("/stock/SPY/nope", skip_moltbot_budget=True))


def test_429_then_200_recovers_with_a_short_backoff_and_one_retry(harness):
    client = harness.install([
        FakeResponse(429, headers=dict(RATE_LIMITED_HEADERS)),
        FakeResponse(200, {"data": [1]}),
    ])
    assert _call() == {"data": [1]}
    assert client.calls == 2
    assert harness.sleeps, "a backoff sleep must have happened"
    assert max(harness.sleeps) < 12.0, f"never the old 60 s header wait: {harness.sleeps}"
    assert 0.5 <= harness.sleeps[0] <= 2.1  # default 1.5 s + <=0.5 s jitter
    # the retry takes a fresh pacer slot, and the 429 just halved the pacer to 1.5 req/s, so the slot
    # after the first request's is 1/1.5 s away (this pins both the re-pace and the AIMD slow-down)
    assert len(harness.sleeps) == 2 and harness.sleeps[1] == pytest.approx(1 / 1.5, abs=0.05)
    assert uw_pacer.get_uw_pacer().n_429 == 1


def test_429_then_429_returns_empty_after_exactly_one_retry(harness):
    client = harness.install([
        FakeResponse(429, headers=dict(RATE_LIMITED_HEADERS)),
        FakeResponse(429, headers=dict(RATE_LIMITED_HEADERS)),
    ])
    assert _call() == {}
    assert client.calls == 2  # no retry storm
    assert uw_pacer.get_uw_pacer().n_429 == 2  # the retry's 429 also informs the pacer
    assert uw_pacer.get_uw_pacer().effective_rate() < 3.0


def test_retry_after_header_is_honored_but_clamped(harness):
    harness.install([
        FakeResponse(429, headers={"Retry-After": "4"}),
        FakeResponse(200, {"ok": 1}),
    ])
    assert _call() == {"ok": 1}
    assert 4.0 <= harness.sleeps[0] <= 4.51


def test_backoff_seconds_clamping_and_garbage():
    b = uw_pacer.backoff_seconds_429
    zero = lambda: 0.0  # noqa: E731
    assert b({"x-uw-req-per-minute-reset": "60000"}, zero) == pytest.approx(1.5)  # ignored
    assert b({"Retry-After": "3"}, zero) == pytest.approx(3.0)
    assert b({"retry-after": "999"}, zero) == pytest.approx(10.0)
    assert b({"Retry-After": "0.01"}, zero) == pytest.approx(0.5)
    assert b({"Retry-After": "soon"}, zero) == pytest.approx(1.5)
    assert b({"Retry-After": "nan"}, zero) == pytest.approx(1.5)
    assert b(None, lambda: 1.0) == pytest.approx(2.0)


def test_single_200_is_not_delayed_by_the_pacer(harness):
    harness.install([FakeResponse(200, {"data": []})])
    assert _call() == {"data": []}
    assert harness.sleeps == []  # first request takes the free slot


def test_second_back_to_back_call_is_spaced(harness):
    harness.install([FakeResponse(200, {"a": 1}), FakeResponse(200, {"b": 2})])

    async def two():
        first = await uw._uw_get("/stock/SPY/nope", skip_moltbot_budget=True)
        second = await uw._uw_get("/stock/QQQ/nope", skip_moltbot_budget=True)
        return first, second

    assert asyncio.run(two()) == ({"a": 1}, {"b": 2})
    assert len(harness.sleeps) == 1 and harness.sleeps[0] == pytest.approx(1 / 3, abs=0.02)


def test_pacer_disabled_restores_the_legacy_fixed_sleep(harness, monkeypatch):
    monkeypatch.setenv("CLIO_UW_PACER_ENABLED", "0")
    uw_pacer.reset_uw_pacer_for_tests()
    harness.install([FakeResponse(200, {"data": [2]})])
    assert _call() == {"data": [2]}
    assert harness.sleeps == [uw._REQUEST_DELAY]


def test_rate_limit_warning_is_deduplicated(harness, caplog):
    import logging

    uw_pacer._last_429_log = -1e9
    uw_pacer._suppressed_429 = 0
    harness.install([FakeResponse(429, headers=dict(RATE_LIMITED_HEADERS)) for _ in range(6)])
    with caplog.at_level(logging.DEBUG, logger="openclaw.uw_pacer"):
        for _ in range(3):
            asyncio.run(uw._uw_get("/stock/SPY/nope", skip_moltbot_budget=True))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "UW rate limited" in r.getMessage()]
    assert len(warnings) == 1  # 6 x 429 within 30 s -> one WARNING, the rest at DEBUG
