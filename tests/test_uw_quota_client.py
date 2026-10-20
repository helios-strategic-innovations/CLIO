from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from agent.integrations import uw_quota_client as quota_module
from agent.integrations.uw_quota_client import (
    QuotaPriority,
    SupabaseUWQuotaClient,
    extract_uw_headers,
    quota_cycle_key,
)


def _client(tmp_path, *, clock=None, **kwargs):
    return SupabaseUWQuotaClient(
        caller="hydrinia-test",
        db_path=str(tmp_path / "quota.sqlite3"),
        enabled=True,
        clock=clock,
        start_worker=False,
        **kwargs,
    )


def _observe(client, decision, count=36490, status=200, endpoint="/market/market-tide"):
    return client.observe_response(
        decision,
        consumer="test",
        endpoint=endpoint,
        purpose="unit",
        status_code=status,
        headers={
            "x-uw-daily-req-count": str(count),
            "x-uw-token-req-limit": "160000",
            "x-uw-minute-req-counter": "12",
            "x-uw-req-per-minute-remaining": "999988",
            "x-uw-req-per-minute-reset": "60000",
        },
        latency_ms=4.2,
    )


def test_extracts_documented_uw_headers_case_insensitively():
    parsed = extract_uw_headers(
        {
            "X-UW-Daily-Req-Count": "36490",
            "x-uw-token-req-limit": "160000",
            "x-uw-minute-req-counter": "12",
            "x-uw-req-per-minute-remaining": "1000000",
            "x-uw-req-per-minute-reset": "60000",
            "Retry-After": "5",
            "Authorization": "must-not-leak",
        }
    )
    assert parsed["daily_count"] == 36490
    assert parsed["entitlement"] == 160000
    assert parsed["minute_remaining"] == 1000000
    assert parsed["retry_after"] == "5"
    assert "authorization" not in parsed["raw"]


def test_8pm_et_cycle_boundary_handles_standard_and_daylight_time():
    assert quota_cycle_key(datetime(2026, 1, 15, 0, 59, tzinfo=timezone.utc)) == "2026-01-13"
    assert quota_cycle_key(datetime(2026, 1, 15, 1, 0, tzinfo=timezone.utc)) == "2026-01-14"
    assert quota_cycle_key(datetime(2026, 7, 9, 23, 59, tzinfo=timezone.utc)) == "2026-07-08"
    assert quota_cycle_key(datetime(2026, 7, 10, 0, 0, tzinfo=timezone.utc)) == "2026-07-09"


def test_reserve_and_observe_never_call_supabase_on_hot_path(tmp_path, monkeypatch):
    client = _client(
        tmp_path,
        base_url="https://example.supabase.co",
        service_role_key="test-only-secret",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("network entered provider hot path")

    monkeypatch.setattr(quota_module.requests, "post", forbidden)
    decision = client.reserve(
        consumer="uw_service",
        endpoint="/market/market-tide",
        purpose="oracle",
        priority=QuotaPriority.P0,
    )
    assert decision.allowed is True
    assert decision.source == "sqlite_wal"
    payload = _observe(client, decision)
    assert payload["daily_count"] == 36490
    assert client.status()["observed_count"] == 36490


def test_stale_state_allows_only_bounded_p0(tmp_path, monkeypatch):
    # The original contract, now the UW_QUOTA_STALE_PROBE=0 rollback mode. With the probe on (default, UW-5 S211)
    # ONE bounded P1/P2 probe may also go out to refresh a stale ledger -- see tests/test_uw_stale_probe.py.
    monkeypatch.setenv("UW_QUOTA_STALE_PROBE", "0")
    now = [datetime(2026, 7, 9, 15, 0, tzinfo=timezone.utc)]
    client = _client(tmp_path, clock=lambda: now[0], stale_after_s=60)
    seed = client.reserve(consumer="seed", endpoint="/market/market-tide", priority=0)
    _observe(client, seed, count=36000)
    now[0] += timedelta(hours=1)

    p0 = client.reserve(consumer="p0", endpoint="/volatility", priority=0)
    p1 = client.reserve(consumer="p1", endpoint="/alerts", priority=1)
    p2 = client.reserve(consumer="p2", endpoint="/congress", priority=2)

    assert p0.allowed is True
    assert p0.reason_code == "DEGRADED_P0_LOCAL"
    assert p1.allowed is False
    assert p2.allowed is False
    assert p1.stale is True


def test_fresh_state_sheds_p2_then_p1_and_preserves_p0(tmp_path):
    client = _client(tmp_path)
    seed = client.reserve(consumer="seed", endpoint="/market/market-tide", priority=0)
    _observe(client, seed, count=151000)

    assert client.reserve(consumer="p0", endpoint="/volatility", priority=0).allowed
    assert client.reserve(consumer="p1", endpoint="/alerts", priority=1).allowed
    assert not client.reserve(consumer="p2", endpoint="/congress", priority=2).allowed


def test_multi_writer_reservations_are_atomic(tmp_path):
    db_path = str(tmp_path / "shared.sqlite3")

    def reserve(index):
        client = SupabaseUWQuotaClient(
            caller=f"writer-{index}",
            db_path=db_path,
            enabled=True,
            start_worker=False,
        )
        return client.reserve(consumer="race", endpoint="/market/market-tide", priority=0)

    with ThreadPoolExecutor(max_workers=12) as pool:
        decisions = list(pool.map(reserve, range(48)))

    assert all(decision.allowed for decision in decisions)
    assert len({decision.request_id for decision in decisions}) == 48
    inspector = SupabaseUWQuotaClient(
        caller="inspector", db_path=db_path, enabled=True, start_worker=False
    )
    assert inspector.status()["pending_count"] == 48


def test_lease_expiry_reclaims_crashed_reservation(tmp_path):
    now = [datetime(2026, 7, 9, 15, 0, tzinfo=timezone.utc)]
    client = _client(tmp_path, clock=lambda: now[0], lease_ttl_s=5)
    assert client.reserve(consumer="crashed", endpoint="/market/market-tide", priority=0).allowed
    assert client.status()["pending_count"] == 1
    now[0] += timedelta(seconds=6)
    assert client.status()["pending_count"] == 0


def test_header_count_decrease_is_flagged_and_never_reduces_usage(tmp_path):
    client = _client(tmp_path)
    first = client.reserve(consumer="first", endpoint="/market/market-tide", priority=0)
    _observe(client, first, count=36501)
    second = client.reserve(consumer="second", endpoint="/market/market-tide", priority=0)
    _observe(client, second, count=36000)
    status = client.status()
    assert status["observed_count"] == 36501
    assert status["anomaly_count"] == 1


def test_missing_headers_keep_a_conservative_local_unconfirmed_count(tmp_path):
    client = _client(tmp_path)
    decision = client.reserve(consumer="missing", endpoint="/market/market-tide", priority=0)
    client.observe_response(
        decision,
        consumer="missing",
        endpoint="/market/market-tide",
        purpose="unit",
        status_code=200,
        headers={},
    )
    status = client.status()
    assert status["local_unconfirmed"] == 1
    assert status["header_schema_ok"] == 0


def test_429_context_is_durable_and_redacts_non_uw_headers(tmp_path):
    client = _client(tmp_path)
    decision = client.reserve(consumer="flow", endpoint="/option-trades/flow-alerts", priority=0)
    payload = client.observe_response(
        decision,
        consumer="flow",
        endpoint="/option-trades/flow-alerts",
        purpose="dpof",
        status_code=429,
        headers={
            "x-uw-daily-req-count": "36501",
            "x-uw-token-req-limit": "160000",
            "x-uw-req-per-minute-remaining": "0",
            "Retry-After": "12",
            "Authorization": "must-not-leak",
        },
        latency_ms=321.5,
    )
    assert payload["endpoint"] == "/option-trades/flow-alerts"
    assert payload["retry_after"] == "12"
    assert "authorization" not in payload["uw_headers"]
    assert client.status()["outbox_pending"] == 1


def test_supabase_outbox_flush_is_explicitly_off_hot_path(tmp_path, monkeypatch):
    client = _client(
        tmp_path,
        base_url="https://example.supabase.co",
        service_role_key="test-only-secret",
    )
    decision = client.reserve(consumer="flow", endpoint="/market/market-tide", priority=0)
    _observe(client, decision)
    calls = []

    class Response:
        def raise_for_status(self):
            return None

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(quota_module.requests, "post", post)
    assert client.flush_outbox_once() == 1
    assert len(calls) == 1
    assert calls[0][0].endswith("/rpc/uw_quota_observe_batch")
    assert len(calls[0][1]["json"]["p_events"]) == 1
    assert client.status()["outbox_pending"] == 0


def test_async_writer_is_daemon_and_has_bounded_close(tmp_path):
    client = SupabaseUWQuotaClient(
        caller="writer",
        db_path=str(tmp_path / "writer.sqlite3"),
        base_url="https://example.supabase.co",
        service_role_key="test-only-secret",
        enabled=True,
        start_worker=True,
        timeout_s=0.01,
    )
    assert client._worker is not None
    assert client._worker._thread is not None
    assert client._worker._thread.daemon is True
    assert client.close(timeout_s=0.5) is True
    assert client._worker._thread.is_alive() is False


def test_new_reset_cycle_does_not_inherit_prior_count(tmp_path):
    now = [datetime(2026, 7, 10, 0, 1, tzinfo=timezone.utc)]
    client = _client(tmp_path, clock=lambda: now[0])
    seed = client.reserve(consumer="seed", endpoint="/market/market-tide", priority=0)
    _observe(client, seed, count=159900)
    assert client.status()["observed_count"] == 159900
    now[0] += timedelta(days=1)
    status = client.status()
    assert status["observed_count"] == 0
    assert status["stale"] is True


def test_no_supabase_credentials_still_uses_local_sqlite_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://environment.example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-only-service-role")
    client = SupabaseUWQuotaClient(
        caller="test",
        base_url="",
        service_role_key="",
        db_path=str(tmp_path / "quota.sqlite3"),
        enabled=True,
        start_worker=False,
    )
    decision = client.reserve(consumer="p0", endpoint="/market/market-tide", priority=0)
    assert decision.allowed is True
    assert decision.source == "sqlite_wal"
    assert client.status()["supabase_sync_enabled"] is False


def test_explicitly_disabled_client_preserves_existing_runtime_behavior(tmp_path):
    client = SupabaseUWQuotaClient(
        caller="test",
        db_path=str(tmp_path / "quota.sqlite3"),
        enabled=False,
    )
    decision = client.reserve(consumer="bots", endpoint="/anything", priority=2)
    assert decision.allowed is True
    assert decision.reason_code == "COORDINATOR_DISABLED"
