"""OP-13 (S214): minute-only UW quota headers are logged at INFO once per endpoint per hour.

Live HTTP 200 responses from some UW endpoints carry only the per-minute headers and no daily-usage
headers. That is a WARNING flood (~94/h) with no accounting impact, so the minute-only shape is
demoted to a deduplicated INFO. Ledger rows and the payload's existing keys are unchanged.

All ledgers live under tmp_path -- the shared production ledger is never touched.
"""

import logging

from agent.integrations.uw_quota_client import SupabaseUWQuotaClient

MINUTE_ONLY = {
    "x-uw-req-per-minute-remaining": "1000000",
    "x-uw-req-per-minute-reset": "60000",
}
FULL = {
    "x-uw-daily-req-count": "1261",
    "x-uw-token-req-limit": "40000",
    "x-uw-req-per-minute-remaining": "1000000",
    "x-uw-req-per-minute-reset": "60000",
}


class _Mono:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _client(tmp_path, mono=None):
    return SupabaseUWQuotaClient(
        caller="clio-test",
        db_path=str(tmp_path / "quota.sqlite3"),
        enabled=True,
        start_worker=False,
        monotonic=mono or _Mono(),
    )


def _call(client, endpoint="/news/headlines", status=200, headers=None):
    decision = client.reserve(consumer="test", endpoint=endpoint, priority=0)
    return client.observe_response(
        decision,
        consumer="test",
        endpoint=endpoint,
        purpose="unit",
        status_code=status,
        headers=MINUTE_ONLY if headers is None else headers,
        latency_ms=3.0,
    )


def _records(caplog, name):
    return [r for r in caplog.records if r.getMessage().startswith(name)]


def test_minute_only_200_logs_info_once(tmp_path, caplog):
    client = _client(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        _call(client)
    info = _records(caplog, "UW_HEADER_MINUTE_ONLY")
    assert len(info) == 1
    assert info[0].levelno == logging.INFO
    assert _records(caplog, "UW_HEADER_SCHEMA_MISSING") == []


def test_second_call_same_endpoint_within_hour_is_silent(tmp_path, caplog):
    mono = _Mono()
    client = _client(tmp_path, mono)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        _call(client)
        mono.now += 3599.0  # one second short of the hour: still silent
        _call(client)
    assert len(_records(caplog, "UW_HEADER_MINUTE_ONLY")) == 1
    assert _records(caplog, "UW_HEADER_SCHEMA_MISSING") == []


def test_different_endpoint_logs_separately(tmp_path, caplog):
    client = _client(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        _call(client, endpoint="/news/headlines")
        _call(client, endpoint="/market/market-tide")
    assert len(_records(caplog, "UW_HEADER_MINUTE_ONLY")) == 2


def test_logs_again_after_interval(tmp_path, caplog):
    mono = _Mono()
    client = _client(tmp_path, mono)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        _call(client)
        mono.now += 3600.0  # exactly one hour: logs again
        _call(client)
    assert len(_records(caplog, "UW_HEADER_MINUTE_ONLY")) == 2


def test_single_minute_reset_header_is_minute_only(tmp_path, caplog):
    client = _client(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        payload = _call(client, headers={"x-uw-req-per-minute-reset": "60000"})
    assert len(_records(caplog, "UW_HEADER_MINUTE_ONLY")) == 1
    assert _records(caplog, "UW_HEADER_SCHEMA_MISSING") == []
    assert payload["header_class"] == "minute_only"


def test_single_minute_remaining_header_is_minute_only(tmp_path, caplog):
    client = _client(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        payload = _call(client, headers={"x-uw-req-per-minute-remaining": "1000000"})
    assert len(_records(caplog, "UW_HEADER_MINUTE_ONLY")) == 1
    assert _records(caplog, "UW_HEADER_SCHEMA_MISSING") == []
    assert payload["header_class"] == "minute_only"


def test_201_and_204_minute_only_stay_warning(tmp_path, caplog):
    client = _client(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        p201 = _call(client, endpoint="/e201", status=201)
        p204 = _call(client, endpoint="/e204", status=204)
    assert _records(caplog, "UW_HEADER_MINUTE_ONLY") == []
    assert len(_records(caplog, "UW_HEADER_SCHEMA_MISSING")) == 2
    assert "header_class" not in p201
    assert "header_class" not in p204


def test_no_uw_headers_at_all_stays_warning(tmp_path, caplog):
    client = _client(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        _call(client, headers={})
    warns = _records(caplog, "UW_HEADER_SCHEMA_MISSING")
    assert len(warns) == 1
    assert warns[0].levelno == logging.WARNING
    assert _records(caplog, "UW_HEADER_MINUTE_ONLY") == []


def test_full_daily_headers_take_normal_path(tmp_path, caplog):
    client = _client(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        payload = _call(client, headers=FULL)
    assert _records(caplog, "UW_HEADER_MINUTE_ONLY") == []
    assert _records(caplog, "UW_HEADER_SCHEMA_MISSING") == []
    assert "header_class" not in payload


def test_only_one_daily_header_stays_warning(tmp_path, caplog):
    client = _client(tmp_path)
    only_count = {"x-uw-daily-req-count": "1261", **MINUTE_ONLY}
    only_limit = {"x-uw-token-req-limit": "40000", **MINUTE_ONLY}
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        p1 = _call(client, endpoint="/a", headers=only_count)
        p2 = _call(client, endpoint="/b", headers=only_limit)
    assert len(_records(caplog, "UW_HEADER_SCHEMA_MISSING")) == 2
    assert _records(caplog, "UW_HEADER_MINUTE_ONLY") == []
    assert "header_class" not in p1
    assert "header_class" not in p2


def test_429_with_minute_only_headers_stays_provider_error(tmp_path, caplog):
    client = _client(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        payload = _call(client, status=429)
    provider = _records(caplog, "UW_PROVIDER_ERROR")
    assert len(provider) == 1
    assert provider[0].levelno == logging.WARNING
    assert _records(caplog, "UW_HEADER_MINUTE_ONLY") == []
    assert "header_class" not in payload


def test_non_200_minute_only_stays_warning(tmp_path, caplog):
    client = _client(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="UWQuota"):
        payload = _call(client, status=404)
    assert _records(caplog, "UW_HEADER_MINUTE_ONLY") == []
    assert len(_records(caplog, "UW_HEADER_SCHEMA_MISSING")) == 1
    assert "header_class" not in payload


def test_minute_only_payload_has_header_class_and_all_previous_keys(tmp_path):
    client = _client(tmp_path)
    minute = _call(client, endpoint="/news/headlines")
    baseline = _call(client, endpoint="/no/headers", headers={})
    assert minute["header_class"] == "minute_only"
    assert set(minute) - {"header_class"} == set(baseline)
    assert minute["header_schema_ok"] is False
    assert minute["uw_headers"] == MINUTE_ONLY


def test_minute_only_200_refreshes_the_ledger_staleness_but_429_does_not(tmp_path):
    """R-13: a minute-only 200 is a fresh observation (observed_at); 429/5xx never are."""
    client = _client(tmp_path)
    assert client.status()["stale"] is True
    _call(client, status=429)
    assert client.status()["stale"] is True
    _call(client, status=503)
    assert client.status()["stale"] is True
    _call(client, status=200)
    assert client.status()["stale"] is False


def test_minute_only_refresh_kill_switch_env_is_read_by_the_client(tmp_path, monkeypatch):
    assert _client(tmp_path).minute_only_fresh is True
    monkeypatch.setenv("UW_QUOTA_MINUTE_ONLY_FRESH", "0")
    assert _client(tmp_path).minute_only_fresh is False
