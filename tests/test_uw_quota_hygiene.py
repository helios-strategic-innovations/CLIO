"""UW-3 / UW-4 / UW-5(log) / UW-6 (S211): quota-ledger and telemetry hygiene after the UW burst fix.

* UW-3: concurrent responses arrive out of order, so the reported daily count can dip 1-2 below the
  observed max. That is noise, not an anomaly (the count itself is protected by max()).
* UW-4: the Supabase outbox POST must not reuse the 0.9 s RPC timeout.
* UW-5: the off-hours "quota denied" WARNING is deduplicated (a suppressed counter keeps the volume).
* UW-6: new cycles inherit the last header-confirmed entitlement; the code default is UW Basic (40,000).

All ledgers live under tmp_path -- the shared production ledger is never touched.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pytest

from agent.integrations import uw_flow_intel as uw
from agent.integrations import uw_quota_client as quota_module
from agent.integrations.uw_quota_client import QuotaDecision, SupabaseUWQuotaClient


def _client(tmp_path, *, clock=None, **kwargs):
    return SupabaseUWQuotaClient(
        caller="clio-test",
        db_path=str(tmp_path / "quota.sqlite3"),
        enabled=True,
        clock=clock,
        start_worker=False,
        **kwargs,
    )


def _observe(client, decision, count, limit="40000", endpoint="/stock/SPY/nope"):
    return client.observe_response(
        decision,
        consumer="test",
        endpoint=endpoint,
        purpose="unit",
        status_code=200,
        headers={
            "x-uw-daily-req-count": str(count),
            "x-uw-token-req-limit": limit,
            "x-uw-req-per-minute-remaining": "1000000",
            "x-uw-req-per-minute-reset": "60000",
        },
        latency_ms=4.2,
    )


def _reserve(client, name="c"):
    return client.reserve(consumer=name, endpoint="/market/market-tide", priority=0)


# ---------------------------------------------------------------- UW-3
def test_small_out_of_order_decrease_is_not_an_anomaly(tmp_path, caplog):
    client = _client(tmp_path)
    _observe(client, _reserve(client, "a"), 1261)
    with caplog.at_level(logging.DEBUG):
        _observe(client, _reserve(client, "b"), 1259)  # the shape seen live: previous=1261 reported=1259
    status = client.status()
    assert status["observed_count"] == 1261  # never lowered
    assert status["anomaly_count"] == 0
    assert "UW_HEADER_COUNT_DECREASE" not in caplog.text
    assert "UW_HEADER_COUNT_REORDER" in caplog.text  # still visible at DEBUG


def test_decrease_beyond_tolerance_is_still_an_anomaly(tmp_path, caplog):
    client = _client(tmp_path)
    _observe(client, _reserve(client, "a"), 1261)
    with caplog.at_level(logging.ERROR):
        _observe(client, _reserve(client, "b"), 1261 - 6)  # tolerance is 5
    assert client.status()["anomaly_count"] == 1
    assert "UW_HEADER_COUNT_DECREASE" in caplog.text


def test_tolerance_zero_restores_the_old_strict_behaviour(tmp_path, monkeypatch):
    monkeypatch.setenv("UW_QUOTA_REORDER_TOLERANCE", "0")
    client = _client(tmp_path)
    _observe(client, _reserve(client, "a"), 1261)
    _observe(client, _reserve(client, "b"), 1260)
    assert client.status()["anomaly_count"] == 1


def test_garbage_tolerance_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("UW_QUOTA_REORDER_TOLERANCE", "many")
    assert quota_module._reorder_tolerance() == 5
    monkeypatch.setenv("UW_QUOTA_REORDER_TOLERANCE", "-3")
    assert quota_module._reorder_tolerance() == 0


# ---------------------------------------------------------------- UW-4
class _FakeHttpResponse:
    def raise_for_status(self):
        return None


def _flush_timeout(tmp_path, monkeypatch, env=None):
    monkeypatch.setenv("SUPABASE_URL", "https://supabase.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-role-key")
    if env is not None:
        monkeypatch.setenv("UW_QUOTA_SUPABASE_POST_TIMEOUT_S", env)
    seen = {}

    def fake_post(url, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        seen["url"] = url
        return _FakeHttpResponse()

    monkeypatch.setattr(quota_module.requests, "post", fake_post)
    client = _client(tmp_path)
    assert client.timeout_s == pytest.approx(0.9)  # the RPC timeout is unchanged
    _observe(client, _reserve(client), 10)  # creates one outbox row
    assert client.flush_outbox_once() == 1
    return seen


def test_outbox_post_uses_a_longer_timeout_than_the_rpc(tmp_path, monkeypatch):
    seen = _flush_timeout(tmp_path, monkeypatch)
    assert seen["timeout"] == pytest.approx(5.0)
    assert seen["url"].endswith("/rest/v1/rpc/uw_quota_observe_batch")


def test_outbox_post_timeout_is_configurable_but_never_below_the_rpc_timeout(tmp_path, monkeypatch):
    assert _flush_timeout(tmp_path, monkeypatch, env="7.5")["timeout"] == pytest.approx(7.5)
    assert _flush_timeout(tmp_path / "b", monkeypatch, env="0.1")["timeout"] == pytest.approx(0.9)


# ---------------------------------------------------------------- UW-5 (log part)
class _DeniedQuota:
    def __init__(self, reason="ACCOUNT_LIMIT"):
        self.calls = 0
        self.reason = reason

    def reserve(self, **kw):
        self.calls += 1
        return QuotaDecision(allowed=False, request_id="00000000-0000-0000-0000-000000000009",
                             reason_code=self.reason, source="sqlite_wal", remaining=40000)

    def observe_response(self, decision, **kw):
        return kw

    def release(self, decision, reason):
        return True


def _denied_call(monkeypatch, endpoint="/news/headlines", reason="ACCOUNT_LIMIT"):
    monkeypatch.setattr(uw, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(uw, "_get_quota_coordinator", lambda: _DeniedQuota(reason))

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(uw.asyncio, "sleep", no_sleep)
    return asyncio.run(uw._uw_get(endpoint, skip_moltbot_budget=True))


def _denied_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "UW_ACCOUNT_QUOTA_DENIED" in r.getMessage()]


def test_quota_denied_warning_is_deduplicated_with_a_suppressed_counter(monkeypatch, caplog):
    uw._quota_denied_log_at.clear()
    uw._quota_denied_suppressed.clear()
    with caplog.at_level(logging.DEBUG, logger=uw.logger.name):
        for _ in range(5):
            assert _denied_call(monkeypatch) == {}
    assert len(_denied_warnings(caplog)) == 1
    assert "suppressed=0" in _denied_warnings(caplog)[0].getMessage()

    # 10 minutes later the next denial logs again and reports what was suppressed
    key = ("/news/headlines", "ACCOUNT_LIMIT")
    uw._quota_denied_log_at[key] -= uw._QUOTA_DENIED_LOG_INTERVAL_S + 1
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=uw.logger.name):
        _denied_call(monkeypatch)
    warns = _denied_warnings(caplog)
    assert len(warns) == 1 and "suppressed=4" in warns[0].getMessage()


def test_quota_denied_dedup_is_per_endpoint(monkeypatch, caplog):
    uw._quota_denied_log_at.clear()
    uw._quota_denied_suppressed.clear()
    with caplog.at_level(logging.DEBUG, logger=uw.logger.name):
        _denied_call(monkeypatch, "/news/headlines")
        _denied_call(monkeypatch, "/stock/SPY/nope")
    assert len(_denied_warnings(caplog)) == 2


# ---------------------------------------------------------------- R-13 (log level)
def _denied_records(caplog):
    return [r for r in caplog.records if "UW_ACCOUNT_QUOTA_DENIED" in r.getMessage() and r.levelno >= logging.INFO]


def test_stale_cache_denial_logs_info_once_per_episode(monkeypatch, caplog):
    """DEGRADED_STALE_CACHE is an expected state until a 200 refreshes the ledger: INFO, not WARNING."""
    uw._quota_denied_log_at.clear()
    uw._quota_denied_suppressed.clear()
    with caplog.at_level(logging.DEBUG, logger=uw.logger.name):
        for _ in range(5):
            _denied_call(monkeypatch, reason="DEGRADED_STALE_CACHE")
    recs = _denied_records(caplog)
    assert len(recs) == 1
    assert recs[0].levelno == logging.INFO
    assert not _denied_warnings(caplog)
    assert "reason=DEGRADED_STALE_CACHE" in recs[0].getMessage()
    assert "suppressed=0" in recs[0].getMessage()

    # 10 minutes later: still inside the 30-minute stale-cache episode -> nothing logged
    key = ("/news/headlines", "DEGRADED_STALE_CACHE")
    uw._quota_denied_log_at[key] -= uw._QUOTA_DENIED_LOG_INTERVAL_S + 1
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=uw.logger.name):
        _denied_call(monkeypatch, reason="DEGRADED_STALE_CACHE")
    assert not _denied_records(caplog)

    # 30 minutes after the first line: logged again at INFO with the suppressed count
    uw._quota_denied_log_at[key] -= uw._QUOTA_DENIED_STALE_LOG_INTERVAL_S
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=uw.logger.name):
        _denied_call(monkeypatch, reason="DEGRADED_STALE_CACHE")
    recs = _denied_records(caplog)
    assert len(recs) == 1 and recs[0].levelno == logging.INFO and "suppressed=5" in recs[0].getMessage()


def test_real_denials_stay_warning(monkeypatch, caplog):
    uw._quota_denied_log_at.clear()
    uw._quota_denied_suppressed.clear()
    with caplog.at_level(logging.DEBUG, logger=uw.logger.name):
        for reason in ("ACCOUNT_LIMIT", "P0_RESERVE", "P1_P0_RESERVE"):
            _denied_call(monkeypatch, reason=reason)
    recs = _denied_records(caplog)
    assert len(recs) == 3
    assert all(r.levelno == logging.WARNING for r in recs)


# ---------------------------------------------------------------- UW-6
def test_new_cycle_inherits_the_last_header_confirmed_entitlement(tmp_path):
    day1 = datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc)   # 11:00 ET, cycle 2026-09-28
    now = [day1]
    client = _client(tmp_path, clock=lambda: now[0])
    # 50,000 (an upgraded plan) differs from the 40,000 code default, so inheritance is observable
    _observe(client, _reserve(client), 100, limit="50000")
    assert client.status()["entitlement"] == 50000

    now[0] = day1 + timedelta(hours=14)  # 01:00 UTC = 21:00 ET -> after the 8 PM ET reset: a new cycle
    status = client.status()
    assert status["cycle_key"] != quota_module.quota_cycle_key(day1)
    assert status["entitlement"] == 50000          # inherited from the confirmed cycle...
    assert status["observed_count"] == 0           # ...while usage starts from zero


def test_new_cycle_ignores_unconfirmed_cycles_and_uses_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("UW_RATE_BUDGET_DAILY_LIMIT", "12345")
    day1 = datetime(2026, 9, 29, 15, 0, tzinfo=timezone.utc)
    now = [day1]
    client = _client(tmp_path, clock=lambda: now[0])
    assert client.status()["entitlement"] == 12345   # no header seen yet -> configured default
    client._ledger.default_entitlement = 999          # the default changes; the unconfirmed row must NOT be inherited
    now[0] = day1 + timedelta(hours=14)
    assert client.status()["entitlement"] == 999     # nothing confirmed -> the (new) default, not the old row


def test_code_default_entitlement_is_the_basic_plan(tmp_path, monkeypatch):
    monkeypatch.delenv("UW_RATE_BUDGET_DAILY_LIMIT", raising=False)
    assert _client(tmp_path).default_entitlement == 40000
