"""UW auth-failure backoff in uw_flow_intel._uw_get (2026-09-25).

With a revoked/invalid key, every UW caller would hit the API and log a 401
ERROR. After a 401 the client returns {} without a request (and without a
budget charge) for _AUTH_BACKOFF_SECONDS, then probes once; a 200 clears it.
"""

import asyncio
import logging

import pytest

from agent.integrations import uw_flow_intel as uw


class _Resp:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or {}
        self.headers = {}

    def json(self):
        return self._data


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def get(self, *_a, **_kw):
        self.calls += 1
        return self.responses.pop(0)


@pytest.fixture
def harness(monkeypatch):
    clock = {"t": 1000.0}
    budget_calls = []

    monkeypatch.setattr(uw, "_auth_blocked_until", 0.0)
    monkeypatch.setattr(uw, "_auth_fail_count", 0)
    monkeypatch.setattr(uw, "_auth_probe_in_flight", False)
    monkeypatch.setattr(uw, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(uw, "_get_quota_coordinator", lambda: (_ for _ in ()).throw(RuntimeError("no quota in tests")))
    monkeypatch.setattr(uw._time, "monotonic", lambda: clock["t"])

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(uw.asyncio, "sleep", _no_sleep)

    async def _budget(name):
        budget_calls.append(name)

    import agent.rate_limiter as rl
    monkeypatch.setattr(rl, "check_and_record", _budget)

    def _install(responses):
        client = _Client(responses)

        async def _get_client():
            return client

        monkeypatch.setattr(uw, "_get_client", _get_client)
        return client

    return clock, budget_calls, _install


def test_401_pauses_all_calls_without_request_or_budget(harness, caplog):
    clock, budget_calls, install = harness
    client = install([_Resp(401)])
    with caplog.at_level(logging.WARNING, logger=uw.logger.name):
        assert asyncio.run(uw._uw_get("/market/market-tide")) == {}
        # inside the backoff window: no HTTP call, no budget charge, no new log line
        for _ in range(5):
            assert asyncio.run(uw._uw_get("/market/market-tide")) == {}
    assert client.calls == 1
    assert budget_calls == ["unusual_whales"]
    assert sum("UW AUTH FAILED (401)" in r.message for r in caplog.records) == 1


def test_probe_after_backoff_and_recovery(harness, caplog):
    clock, budget_calls, install = harness
    client = install([_Resp(401), _Resp(401), _Resp(200, {"data": [1]}), _Resp(200, {"data": [2]})])
    asyncio.run(uw._uw_get("/x"))
    clock["t"] += uw._AUTH_BACKOFF_SECONDS + 1
    with caplog.at_level(logging.WARNING, logger=uw.logger.name):
        assert asyncio.run(uw._uw_get("/x")) == {}          # probe #2 still 401
    assert any("still failing" in r.message for r in caplog.records)
    assert asyncio.run(uw._uw_get("/x")) == {}              # blocked again, no request
    assert client.calls == 2
    clock["t"] += uw._AUTH_BACKOFF_SECONDS + 1
    assert asyncio.run(uw._uw_get("/x")) == {"data": [1]}   # recovered
    assert uw._auth_blocked_until == 0.0 and uw._auth_fail_count == 0
    assert asyncio.run(uw._uw_get("/x")) == {"data": [2]}   # normal service
    assert client.calls == 4


def test_no_key_still_short_circuits(harness, monkeypatch):
    _clock, budget_calls, install = harness
    client = install([])
    monkeypatch.setattr(uw, "_get_api_key", lambda: "")
    assert asyncio.run(uw._uw_get("/x")) == {}
    assert client.calls == 0 and budget_calls == []


def test_only_one_probe_when_many_callers_race(harness):
    """DPOF gathers ~14 UW calls at once; after the pause only ONE may probe."""
    clock, _budget, install = harness
    client = install([_Resp(401)] + [_Resp(401)] * 20)
    asyncio.run(uw._uw_get("/x"))                       # first 401 -> paused
    clock["t"] += uw._AUTH_BACKOFF_SECONDS + 1

    async def _race():
        return await asyncio.gather(*(uw._uw_get(f"/x/{i}") for i in range(14)))

    results = asyncio.run(_race())
    assert all(r == {} for r in results)
    assert client.calls == 2                            # the original + exactly one probe
    assert uw._auth_fail_count == 2
    assert uw.uw_auth_backoff_active() is True


def test_429_then_401_on_retry_engages_backoff(harness):
    clock, _budget, install = harness
    client = install([_Resp(429), _Resp(401)])
    assert asyncio.run(uw._uw_get("/x")) == {}
    assert uw._auth_fail_count == 1 and uw.uw_auth_backoff_active() is True
    assert asyncio.run(uw._uw_get("/x")) == {}
    assert client.calls == 2


def test_non_401_error_lifts_the_pause(harness):
    clock, _budget, install = harness
    client = install([_Resp(401), _Resp(500), _Resp(200, {"data": [1]})])
    asyncio.run(uw._uw_get("/x"))
    clock["t"] += uw._AUTH_BACKOFF_SECONDS + 1
    assert asyncio.run(uw._uw_get("/x")) == {}          # probe answered 500: key authenticated
    assert uw.uw_auth_backoff_active() is False
    assert asyncio.run(uw._uw_get("/x")) == {"data": [1]}
    assert client.calls == 3


def test_first_discovery_burst_logs_once(harness, caplog):
    """Calls already in flight when the first 401 lands (DPOF gathers ~14) all
    get 401 too; only the first one may set the pause and log."""
    clock, _budget, install = harness

    class _SlowClient:
        calls = 0

        async def get(self, *_a, **_kw):
            type(self).calls += 1
            # Yield so every call is in flight together. call_soon, not
            # asyncio.sleep: the harness freezes time.monotonic (asyncio timers).
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            loop.call_soon(fut.set_result, None)
            await fut
            return _Resp(401)

    client = _SlowClient()

    async def _get_client():
        return client

    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(uw, "_get_client", _get_client)
    try:
        async def _race():
            return await asyncio.gather(*(uw._uw_get(f"/x/{i}") for i in range(14)))

        with caplog.at_level(logging.WARNING, logger=uw.logger.name):
            asyncio.run(_race())
    finally:
        mp.undo()
    assert _SlowClient.calls == 14            # first discovery: all were already in flight
    assert uw._auth_fail_count == 1
    assert sum("401" in r.message for r in caplog.records) == 1
