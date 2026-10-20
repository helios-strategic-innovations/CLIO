"""UW-5 (S211): a stale quota ledger may heal itself through ONE bounded probe.

Before: with no header-bearing response for `stale_after_s`, only P0 requests were admitted. Overnight no P0
call runs, so every P1/P2 call (e.g. /news/headlines each minute) was denied ("DEGRADED_STALE_CACHE", 136
WARNINGs in 13 h) until the first P0 call at the open -- the ledger could never refresh itself.
Now: one P1/P2 request at a time (single-flight, bounded per interval) is admitted as `STALE_PROBE`; its
response headers refresh the ledger and normal priority rules resume. Tmp ledgers only, fake clock.
"""

from datetime import datetime, timedelta, timezone

import pytest

from agent.integrations.uw_quota_client import QuotaPriority, SupabaseUWQuotaClient

T0 = datetime(2026, 9, 30, 6, 0, 0, tzinfo=timezone.utc)   # 02:00 ET: overnight, nothing fresh


class Box:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def _client(tmp_path, box, **kw):
    return SupabaseUWQuotaClient(caller="clio-test", db_path=str(tmp_path / "q.sqlite3"), enabled=True,
                                 clock=box, start_worker=False, stale_after_s=300.0, **kw)


def _reserve(client, priority, endpoint="/news/headlines"):
    return client.reserve(consumer="t", endpoint=endpoint, priority=int(priority))


def _observe_with_headers(client, decision, count=10):
    return client.observe_response(
        decision, consumer="t", endpoint="/news/headlines", purpose="unit", status_code=200,
        headers={"x-uw-daily-req-count": str(count), "x-uw-token-req-limit": "40000"}, latency_ms=1.0,
    )


def _observe_without_headers(client, decision):
    return client.observe_response(
        decision, consumer="t", endpoint="/news/headlines", purpose="unit", status_code=200,
        headers={}, latency_ms=1.0,
    )


@pytest.fixture()
def box():
    return Box()


def test_stale_ledger_admits_exactly_one_p1_probe_at_a_time(tmp_path, box):
    client = _client(tmp_path, box)
    first = _reserve(client, QuotaPriority.P1)
    assert first.allowed and first.reason_code == "STALE_PROBE"
    second = _reserve(client, QuotaPriority.P1)            # probe still in flight
    third = _reserve(client, QuotaPriority.P2, "/congress/recent-trades")
    assert not second.allowed and second.reason_code == "DEGRADED_STALE_CACHE"
    assert not third.allowed and third.reason_code == "DEGRADED_STALE_CACHE"


def test_probe_response_with_headers_refreshes_the_ledger_and_normal_rules_resume(tmp_path, box):
    client = _client(tmp_path, box)
    probe = _reserve(client, QuotaPriority.P1)
    assert probe.reason_code == "STALE_PROBE"
    _observe_with_headers(client, probe)
    after = _reserve(client, QuotaPriority.P1)
    assert after.allowed and after.reason_code == "P1_RESERVED"       # no longer stale
    p2 = _reserve(client, QuotaPriority.P2, "/congress/recent-trades")
    assert p2.allowed and p2.reason_code == "P2_RESERVED"


def test_a_headerless_probe_does_not_turn_stale_into_allow_everything(tmp_path, box):
    """The failure mode the bound exists for: responses that never refresh observed_at."""
    client = _client(tmp_path, box)
    probe = _reserve(client, QuotaPriority.P1)
    _observe_without_headers(client, probe)                # completes, but the ledger is still stale
    box.advance(10)
    assert not _reserve(client, QuotaPriority.P1).allowed              # < 60 s since the last probe
    box.advance(60)
    again = _reserve(client, QuotaPriority.P1)
    assert again.allowed and again.reason_code == "STALE_PROBE"        # next probe after the interval


def test_a_crashed_probe_frees_after_the_lease_ttl(tmp_path, box):
    client = _client(tmp_path, box, lease_ttl_s=120)
    assert _reserve(client, QuotaPriority.P1).reason_code == "STALE_PROBE"    # never observed (crash)
    box.advance(30)
    assert not _reserve(client, QuotaPriority.P1).allowed              # still in flight
    box.advance(100)                                                   # 130 s > lease 120 s
    assert _reserve(client, QuotaPriority.P1).reason_code == "STALE_PROBE"


def test_a_slow_probe_still_in_flight_blocks_the_next_one(tmp_path, box):
    """Pending longer than the 60 s interval but shorter than the 120 s lease: only the single-flight
    check (not the per-interval bound) can deny a second probe here."""
    client = _client(tmp_path, box, lease_ttl_s=120)
    assert _reserve(client, QuotaPriority.P1).reason_code == "STALE_PROBE"
    box.advance(90)
    blocked = _reserve(client, QuotaPriority.P1)
    assert not blocked.allowed and blocked.reason_code == "DEGRADED_STALE_CACHE"


def test_p0_behaviour_is_unchanged(tmp_path, box):
    client = _client(tmp_path, box)
    p0 = _reserve(client, QuotaPriority.P0, "/market/market-tide")
    assert p0.allowed and p0.reason_code == "DEGRADED_P0_LOCAL"
    p0b = _reserve(client, QuotaPriority.P0, "/market/market-tide")
    assert p0b.allowed and p0b.reason_code == "DEGRADED_P0_LOCAL"


def test_no_probe_when_the_daily_quota_is_exhausted(tmp_path, box, monkeypatch):
    monkeypatch.setenv("UW_RATE_BUDGET_DAILY_LIMIT", "1")
    client = _client(tmp_path, box)
    assert _reserve(client, QuotaPriority.P0, "/market/market-tide").allowed      # uses the only request
    denied = _reserve(client, QuotaPriority.P1)
    assert not denied.allowed and denied.reason_code == "DEGRADED_STALE_CACHE"


def test_probe_flag_off_restores_the_old_behaviour(tmp_path, box, monkeypatch):
    monkeypatch.setenv("UW_QUOTA_STALE_PROBE", "0")
    client = _client(tmp_path, box)
    denied = _reserve(client, QuotaPriority.P1)
    assert not denied.allowed and denied.reason_code == "DEGRADED_STALE_CACHE"


def test_interval_is_configurable(tmp_path, box, monkeypatch):
    monkeypatch.setenv("UW_QUOTA_STALE_PROBE_INTERVAL_S", "5")
    client = _client(tmp_path, box)
    probe = _reserve(client, QuotaPriority.P1)
    _observe_without_headers(client, probe)
    box.advance(6)
    assert _reserve(client, QuotaPriority.P1).reason_code == "STALE_PROBE"


def test_fresh_ledger_never_probes(tmp_path, box):
    client = _client(tmp_path, box)
    _observe_with_headers(client, _reserve(client, QuotaPriority.P0, "/market/market-tide"))
    box.advance(60)                                        # well inside stale_after_s (300)
    d = _reserve(client, QuotaPriority.P1)
    assert d.allowed and d.reason_code == "P1_RESERVED"


def test_probe_reserved_rows_are_recorded_with_the_probe_reason(tmp_path, box):
    import sqlite3

    client = _client(tmp_path, box)
    _reserve(client, QuotaPriority.P1)
    con = sqlite3.connect(str(tmp_path / "q.sqlite3"))
    rows = con.execute("SELECT reason_code, state FROM quota_reservations").fetchall()
    con.close()
    assert rows == [("STALE_PROBE", "pending")]


# ---------------------------------------------------------------- R-13: minute-only 200s refresh the ledger
MINUTE_ONLY_HEADERS = {"x-uw-req-per-minute-remaining": "95", "x-uw-req-per-minute-reset": "1790000000000"}


def _observe_minute_only(client, decision, status=200, endpoint="/news/headlines"):
    return client.observe_response(
        decision, consumer="t", endpoint=endpoint, purpose="unit", status_code=status,
        headers=dict(MINUTE_ONLY_HEADERS), latency_ms=1.0,
    )


def test_minute_only_200_refreshes_the_ledger_and_p1_is_admitted(tmp_path, box):
    client = _client(tmp_path, box)
    probe = _reserve(client, QuotaPriority.P1)
    assert probe.reason_code == "STALE_PROBE"
    payload = _observe_minute_only(client, probe)
    assert payload["header_class"] == "minute_only"
    after = _reserve(client, QuotaPriority.P1)
    assert after.allowed and after.reason_code == "P1_RESERVED" and after.stale is False


def test_minute_only_200_refreshes_even_without_a_probe_decision(tmp_path, box):
    client = _client(tmp_path, box)
    assert client.status()["stale"] is True
    _observe_minute_only(client, None)
    assert client.status()["stale"] is False


@pytest.mark.parametrize("status", [429, 500, 503])
def test_minute_only_429_and_5xx_never_refresh(tmp_path, box, status):
    client = _client(tmp_path, box)
    probe = _reserve(client, QuotaPriority.P1)
    _observe_minute_only(client, probe, status=status)
    assert client.status()["stale"] is True
    box.advance(61)                                                    # the next probe is due again
    again = _reserve(client, QuotaPriority.P1)
    assert again.reason_code == "STALE_PROBE"                          # still stale: only the bounded probe admits
    assert not _reserve(client, QuotaPriority.P1).allowed


def test_headerless_200_still_does_not_refresh(tmp_path, box):
    client = _client(tmp_path, box)
    probe = _reserve(client, QuotaPriority.P1)
    _observe_without_headers(client, probe)
    assert client.status()["stale"] is True


def test_flag_off_reproduces_the_stale_denial(tmp_path, box, monkeypatch):
    monkeypatch.setenv("UW_QUOTA_MINUTE_ONLY_FRESH", "0")
    client = _client(tmp_path, box)
    probe = _reserve(client, QuotaPriority.P1)
    payload = _observe_minute_only(client, probe)
    assert payload["header_class"] == "minute_only"                    # classification itself is unchanged
    denied = _reserve(client, QuotaPriority.P1)
    assert not denied.allowed and denied.reason_code == "DEGRADED_STALE_CACHE"
    assert client.status()["stale"] is True


def test_refresh_does_not_touch_observed_count_or_entitlement(tmp_path, box):
    client = _client(tmp_path, box)
    before = client.status()
    _observe_minute_only(client, _reserve(client, QuotaPriority.P1))
    after = client.status()
    assert after["observed_count"] == before["observed_count"]
    assert after["entitlement"] == before["entitlement"]


def test_local_usage_still_counts_and_real_exhaustion_still_denies(tmp_path, box, monkeypatch):
    """After a minute-only refresh, local calls keep counting: the account limit and the reserves bind."""
    monkeypatch.setenv("UW_RATE_BUDGET_DAILY_LIMIT", "3")
    client = _client(tmp_path, box)
    for _ in range(3):
        decision = _reserve(client, QuotaPriority.P0, "/market/market-tide")
        assert decision.allowed
        _observe_minute_only(client, decision, endpoint="/market/market-tide")
    assert client.status()["stale"] is False                           # refreshed, yet usage is 3 of 3
    exhausted = _reserve(client, QuotaPriority.P0, "/market/market-tide")
    assert not exhausted.allowed and exhausted.reason_code == "ACCOUNT_LIMIT"
    p1 = _reserve(client, QuotaPriority.P1)
    assert not p1.allowed and p1.reason_code == "P0_RESERVE"


def test_p1_reserve_binds_after_a_refresh(tmp_path, box, monkeypatch):
    """Remaining just above the P0 reserve (5000): after a refresh P1 is still denied, P0 still admitted."""
    monkeypatch.setenv("UW_RATE_BUDGET_DAILY_LIMIT", "5001")
    client = _client(tmp_path, box)
    _observe_minute_only(client, _reserve(client, QuotaPriority.P1))   # 1 local call, ledger refreshed
    p1 = _reserve(client, QuotaPriority.P1)
    assert not p1.allowed and p1.reason_code == "P0_RESERVE"
    p0 = _reserve(client, QuotaPriority.P0, "/market/market-tide")
    assert p0.allowed and p0.reason_code == "P0_RESERVED"
