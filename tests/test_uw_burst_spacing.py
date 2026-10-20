"""UW-1 (S211, ADR-005): DPOF's 12-14 simultaneous UW requests must reach the client spaced out.

End to end through the real ``_uw_get``/``_uw_request`` with real timers, at a scaled-up pacer rate
(40 req/s instead of 3) so the test stays fast. Before the fix all 14 sends landed within a few ms
of each other (the old per-call sleep(0.3) ran in parallel). The quota coordinator is faked so the
live ledger is never touched.
"""

import asyncio
import time

from agent.integrations import uw_flow_intel as uw
from agent.integrations import uw_pacer
from agent.integrations.uw_quota_client import QuotaDecision

N = 14
RATE = 40.0


class _Resp:
    status_code = 200
    headers = {"x-uw-daily-req-count": "1", "x-uw-token-req-limit": "40000"}

    def json(self):
        return {"data": []}


class _TimestampingClient:
    def __init__(self):
        self.sent = []

    async def get(self, *_a, **_k):
        self.sent.append(time.monotonic())
        return _Resp()


class _Quota:
    def reserve(self, **kw):
        return QuotaDecision(allowed=True, request_id="00000000-0000-0000-0000-000000000001",
                             reason_code="P0_RESERVED", source="supabase", remaining=39000)

    def observe_response(self, decision, **kw):
        return kw

    def release(self, decision, reason):
        return True


def _burst(monkeypatch):
    client = _TimestampingClient()

    async def get_client():
        return client

    monkeypatch.setattr(uw, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(uw, "_get_quota_coordinator", lambda: _Quota())
    monkeypatch.setattr(uw, "_get_client", get_client)

    async def fan_out():
        # the same shape as DPOF._maybe_refresh_uw: everything at once through asyncio.gather
        return await asyncio.gather(*(uw._uw_get(f"/stock/T{i}/nope", skip_moltbot_budget=True) for i in range(N)))

    results = asyncio.run(fan_out())
    return client, results


def test_gathered_burst_is_spaced_by_the_pacer(monkeypatch):
    monkeypatch.setenv("CLIO_UW_RATE_PER_SEC", str(RATE))
    monkeypatch.setenv("CLIO_UW_BURST", "1")
    uw_pacer.reset_uw_pacer_for_tests()
    client, results = _burst(monkeypatch)

    assert results == [{"data": []}] * N
    sends = sorted(client.sent)
    assert len(sends) == N
    gaps = [b - a for a, b in zip(sends, sends[1:])]
    assert min(gaps) >= 0.5 / RATE, f"sends bunched up: min gap {min(gaps) * 1000:.1f} ms"
    assert sends[-1] - sends[0] >= 0.75 * (N - 1) / RATE  # ~13/40 s = 325 ms, not ~0


def test_with_the_pacer_disabled_the_burst_is_unspaced_legacy_behaviour(monkeypatch):
    """Rollback flag sanity: CLIO_UW_PACER_ENABLED=0 really restores the old (bunched) behaviour."""
    monkeypatch.setenv("CLIO_UW_PACER_ENABLED", "0")
    monkeypatch.setattr(uw, "_REQUEST_DELAY", 0.0)  # legacy sleep, made instant for the test
    uw_pacer.reset_uw_pacer_for_tests()
    client, _ = _burst(monkeypatch)
    sends = sorted(client.sent)
    assert sends[-1] - sends[0] < 0.1  # all 14 within 100 ms: exactly the bug the pacer removes
