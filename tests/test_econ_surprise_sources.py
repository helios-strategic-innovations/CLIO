"""Tests for agent/integrations/econ_surprise.py — _fetch_nasdaq_raw /
_fetch_tradingview_raw / _fetch_all_sources: HTTP happy path, shape-drift
detection (WARNING-once-per-day + source marked unhealthy, never crash,
never fabricate), budget exhaustion, and source fallback.

No network: httpx.AsyncClient is monkeypatched wholesale (same pattern as
tests/test_econ_calendar_fred_fallback.py). Budgets are redirected to a
per-session temp file by tests/conftest.py, so check_and_record() here never
touches real rate-limit state.
"""

import asyncio
from datetime import date

import pytest

import agent.integrations.econ_surprise as es


def _run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════
#  Fakes (mirrors tests/test_econ_calendar_fred_fallback.py's pattern)
# ═══════════════════════════════════════════════════════════════════════════

class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class _FakeAsyncClient:
    def __init__(self, outcome_holder, calls, **_kwargs):
        self._outcome_holder = outcome_holder
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, url, headers=None, params=None):
        self._calls.append((url, params))
        outcome = self._outcome_holder[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _install_fake_httpx(monkeypatch, outcome):
    import httpx as real_httpx
    calls: list = []
    holder = [outcome]

    def _factory(**kwargs):
        return _FakeAsyncClient(holder, calls, **kwargs)

    monkeypatch.setattr(real_httpx, "AsyncClient", _factory)
    return calls, holder


def _nasdaq_payload(rows):
    return {"data": {"asOf": "2026-09-25", "headers": {}, "rows": rows}}


def _tv_payload(result, status="ok"):
    return {"status": status, "result": result}


# ═══════════════════════════════════════════════════════════════════════════
#  Nasdaq fetch
# ═══════════════════════════════════════════════════════════════════════════

def test_fetch_nasdaq_raw_happy_path(monkeypatch):
    state = es._default_state()
    rows = [
        {"gmt": "08:30", "country": "United States", "eventName": "Initial Jobless Claims",
         "actual": "197K", "consensus": "201K", "previous": "198K"},
        {"gmt": "24H", "country": "United Kingdom", "eventName": "Car Registration",
         "actual": "-39.8%", "consensus": " ", "previous": "32.7%"},  # non-US, must be filtered
    ]
    calls, _ = _install_fake_httpx(monkeypatch, _FakeResponse(200, _nasdaq_payload(rows)))
    result = _run(es._fetch_nasdaq_raw(date(2026, 9, 25), state))
    assert result is not None
    assert len(result) == 1
    assert result[0]["event"] == "Initial Jobless Claims"
    assert len(calls) == 1
    url, params = calls[0]
    assert params == {"date": "2026-09-25"}
    assert state["source_health"]["nasdaq"]["healthy"] is True


def test_fetch_nasdaq_raw_http_error_returns_none(monkeypatch):
    state = es._default_state()
    _install_fake_httpx(monkeypatch, _FakeResponse(503, {}))
    result = _run(es._fetch_nasdaq_raw(date(2026, 9, 25), state))
    assert result is None


def test_fetch_nasdaq_raw_network_exception_returns_none(monkeypatch):
    state = es._default_state()
    _install_fake_httpx(monkeypatch, ConnectionError("boom"))
    result = _run(es._fetch_nasdaq_raw(date(2026, 9, 25), state))
    assert result is None


def test_fetch_nasdaq_raw_shape_drift_marks_unhealthy_and_warns_once_per_day(monkeypatch, caplog):
    state = es._default_state()
    # Missing "data" key entirely -- classic shape drift
    _install_fake_httpx(monkeypatch, _FakeResponse(200, {"unexpected": "shape"}))
    with caplog.at_level("WARNING"):
        result = _run(es._fetch_nasdaq_raw(date(2026, 9, 25), state))
    assert result is None
    assert state["source_health"]["nasdaq"]["healthy"] is False
    warn_count = sum(1 for r in caplog.records if r.levelname == "WARNING" and "nasdaq" in r.message.lower())
    assert warn_count == 1

    # A second failure the SAME day must not warn again (already recorded today)
    caplog.clear()
    with caplog.at_level("WARNING"):
        _run(es._fetch_nasdaq_raw(date(2026, 9, 25), state))
    warn_count2 = sum(1 for r in caplog.records if r.levelname == "WARNING")
    assert warn_count2 == 0


def test_fetch_nasdaq_raw_rows_not_list_is_shape_drift(monkeypatch):
    state = es._default_state()
    _install_fake_httpx(monkeypatch, _FakeResponse(200, {"data": {"rows": "not-a-list"}}))
    result = _run(es._fetch_nasdaq_raw(date(2026, 9, 25), state))
    assert result is None
    assert state["source_health"]["nasdaq"]["healthy"] is False


def test_fetch_nasdaq_raw_row_missing_expected_keys_is_shape_drift(monkeypatch):
    state = es._default_state()
    rows = [{"gmt": "08:30", "country": "United States"}]  # missing eventName/actual/consensus/previous
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _nasdaq_payload(rows)))
    result = _run(es._fetch_nasdaq_raw(date(2026, 9, 25), state))
    assert result is None


def test_fetch_nasdaq_raw_genuine_empty_week_is_ok_not_drift(monkeypatch):
    state = es._default_state()
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _nasdaq_payload([])))
    result = _run(es._fetch_nasdaq_raw(date(2026, 9, 25), state))
    assert result == []
    assert state["source_health"]["nasdaq"]["healthy"] is True


def test_fetch_nasdaq_raw_budget_exhausted_returns_none(monkeypatch):
    """Budget enforcement itself is rate_limiter's own concern (tested there);
    here we only need econ_surprise to treat BudgetExhaustedError as "this
    source didn't answer this cycle" -- never crash, never fabricate."""
    state = es._default_state()

    async def _raise(*_a, **_kw):
        raise es.BudgetExhaustedError("econ_surprise", "budget exhausted")

    monkeypatch.setattr(es, "check_and_record", _raise)
    result = _run(es._fetch_nasdaq_raw(date(2026, 9, 25), state))
    assert result is None


def test_fetch_tradingview_raw_budget_exhausted_returns_none(monkeypatch):
    state = es._default_state()

    async def _raise(*_a, **_kw):
        raise es.BudgetExhaustedError("econ_surprise", "budget exhausted")

    monkeypatch.setattr(es, "check_and_record", _raise)
    result = _run(es._fetch_tradingview_raw(date(2026, 9, 25), date(2026, 9, 25), state))
    assert result is None


# ═══════════════════════════════════════════════════════════════════════════
#  TradingView fetch
# ═══════════════════════════════════════════════════════════════════════════

def test_fetch_tradingview_raw_happy_path(monkeypatch):
    state = es._default_state()
    result_rows = [
        {"title": "Non Farm Payrolls", "date": "2026-09-04T12:30:00.000Z",
         "actual": 162, "forecast": 56, "previous": 21, "unit": None, "importance": 1},
    ]
    calls, _ = _install_fake_httpx(monkeypatch, _FakeResponse(200, _tv_payload(result_rows)))
    result = _run(es._fetch_tradingview_raw(date(2026, 9, 4), date(2026, 9, 4), state))
    assert result is not None
    assert len(result) == 1
    assert result[0]["event"] == "Non Farm Payrolls"
    assert len(calls) == 1
    _url, params = calls[0]
    assert params["countries"] == "US"
    assert "from" in params and "to" in params
    assert state["source_health"]["tradingview"]["healthy"] is True


def test_fetch_tradingview_raw_status_not_ok_is_shape_drift(monkeypatch):
    state = es._default_state()
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _tv_payload([], status="error")))
    result = _run(es._fetch_tradingview_raw(date(2026, 9, 25), date(2026, 9, 25), state))
    assert result is None
    assert state["source_health"]["tradingview"]["healthy"] is False


def test_fetch_tradingview_raw_result_not_list_is_shape_drift(monkeypatch):
    state = es._default_state()
    _install_fake_httpx(monkeypatch, _FakeResponse(200, {"status": "ok", "result": "nope"}))
    result = _run(es._fetch_tradingview_raw(date(2026, 9, 25), date(2026, 9, 25), state))
    assert result is None


def test_fetch_tradingview_raw_http_error_returns_none(monkeypatch):
    state = es._default_state()
    _install_fake_httpx(monkeypatch, _FakeResponse(500, {}))
    result = _run(es._fetch_tradingview_raw(date(2026, 9, 25), date(2026, 9, 25), state))
    assert result is None


def test_fetch_tradingview_raw_network_exception_returns_none(monkeypatch):
    state = es._default_state()
    _install_fake_httpx(monkeypatch, TimeoutError("timeout"))
    result = _run(es._fetch_tradingview_raw(date(2026, 9, 25), date(2026, 9, 25), state))
    assert result is None


def test_fetch_tradingview_raw_bad_json_is_shape_drift(monkeypatch):
    state = es._default_state()
    _install_fake_httpx(monkeypatch, _FakeResponse(200, ValueError("bad json")))
    result = _run(es._fetch_tradingview_raw(date(2026, 9, 25), date(2026, 9, 25), state))
    assert result is None
    assert state["source_health"]["tradingview"]["healthy"] is False


# ═══════════════════════════════════════════════════════════════════════════
#  _fetch_all_sources: Nasdaq date-at-a-time loop, TV range, fallback shape
# ═══════════════════════════════════════════════════════════════════════════

def test_fetch_all_sources_both_ok(monkeypatch):
    """Both sources succeed -- dict has both non-None."""
    calls_seen = {"nasdaq": 0, "tv": 0}

    async def _fake_nasdaq(d, state):
        calls_seen["nasdaq"] += 1
        return [{"event": "X", "release_time_et": "2026-09-25T08:30:00-04:00",
                  "actual": 1.0, "actual_unit": "level", "consensus": 0.5, "consensus_unit": "level",
                  "previous": None, "previous_unit": None, "importance": 1, "source": "nasdaq"}]

    async def _fake_tv(start, end, state):
        calls_seen["tv"] += 1
        return [{"event": "X", "release_time_et": "2026-09-25T08:30:00-04:00",
                  "actual": 1.1, "actual_unit": "level", "consensus": 0.5, "consensus_unit": "level",
                  "previous": None, "previous_unit": None, "importance": 1, "source": "tradingview"}]

    monkeypatch.setattr(es, "_fetch_nasdaq_raw", _fake_nasdaq)
    monkeypatch.setattr(es, "_fetch_tradingview_raw", _fake_tv)

    state = es._default_state()
    result = _run(es._fetch_all_sources(date(2026, 9, 25), date(2026, 9, 25), state))
    assert result["nasdaq"] is not None and len(result["nasdaq"]) == 1
    assert result["tradingview"] is not None and len(result["tradingview"]) == 1
    assert calls_seen == {"nasdaq": 1, "tv": 1}


def test_fetch_all_sources_nasdaq_called_once_per_day_in_range(monkeypatch):
    calls = []

    async def _fake_nasdaq(d, state):
        calls.append(d)
        return []

    async def _fake_tv(start, end, state):
        return []

    monkeypatch.setattr(es, "_fetch_nasdaq_raw", _fake_nasdaq)
    monkeypatch.setattr(es, "_fetch_tradingview_raw", _fake_tv)

    state = es._default_state()
    _run(es._fetch_all_sources(date(2026, 9, 25), date(2026, 9, 27), state))
    assert calls == [date(2026, 9, 25), date(2026, 9, 26), date(2026, 9, 27)]


def test_fetch_all_sources_nasdaq_down_tv_up(monkeypatch):
    async def _fake_nasdaq(d, state):
        return None

    async def _fake_tv(start, end, state):
        return [{"event": "Y", "release_time_et": "2026-09-25T08:30:00-04:00",
                  "actual": 1.0, "actual_unit": "level", "consensus": None, "consensus_unit": None,
                  "previous": None, "previous_unit": None, "importance": 1, "source": "tradingview"}]

    monkeypatch.setattr(es, "_fetch_nasdaq_raw", _fake_nasdaq)
    monkeypatch.setattr(es, "_fetch_tradingview_raw", _fake_tv)

    state = es._default_state()
    result = _run(es._fetch_all_sources(date(2026, 9, 25), date(2026, 9, 25), state))
    assert result["nasdaq"] is None
    assert result["tradingview"] is not None


def test_fetch_all_sources_both_down(monkeypatch):
    async def _fake_nasdaq(d, state):
        return None

    async def _fake_tv(start, end, state):
        return None

    monkeypatch.setattr(es, "_fetch_nasdaq_raw", _fake_nasdaq)
    monkeypatch.setattr(es, "_fetch_tradingview_raw", _fake_tv)

    state = es._default_state()
    result = _run(es._fetch_all_sources(date(2026, 9, 25), date(2026, 9, 25), state))
    assert result == {"nasdaq": None, "tradingview": None}


# ═══════════════════════════════════════════════════════════════════════════
#  Nasdaq empty day (no scheduled releases) — verified live 2026-09-27 (Sunday)
# ═══════════════════════════════════════════════════════════════════════════

_NASDAQ_NO_RECORDS = {
    "data": None, "message": None,
    "status": {"rCode": 200, "developerMessage": None,
               "bCodeMessage": [{"code": 1002, "errorMessage": "Economic Events Calendar: No record found."}]},
}


def test_fetch_nasdaq_raw_empty_day_is_not_drift(monkeypatch, caplog):
    state = es._default_state()
    state["source_health"]["nasdaq"]["healthy"] = False  # e.g. left over from a real drift
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _NASDAQ_NO_RECORDS))
    with caplog.at_level("WARNING", logger=es.logger.name):
        result = _run(es._fetch_nasdaq_raw(date(2026, 9, 27), state))
    assert result == []
    assert state["source_health"]["nasdaq"]["healthy"] is True
    assert not any("drift" in r.message or "unhealthy" in r.message for r in caplog.records)


def test_null_data_without_the_no_record_code_is_still_drift(monkeypatch):
    state = es._default_state()
    payload = {"data": None, "message": None, "status": {"rCode": 200, "bCodeMessage": None}}
    _install_fake_httpx(monkeypatch, _FakeResponse(200, payload))
    assert _run(es._fetch_nasdaq_raw(date(2026, 9, 27), state)) is None
    assert state["source_health"]["nasdaq"]["healthy"] is False
