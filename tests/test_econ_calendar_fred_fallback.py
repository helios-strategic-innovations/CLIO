"""Tests for the FRED + hardcoded-FOMC fallback added to
agent/integrations/econ_calendar.py (2026-09-25).

Goal under test: DPOF's economic-event entry blackout must survive a UW
outage. UW's economic-calendar endpoint failing must no longer be the only
thing standing between DPOF and a healthy calendar cache -- see
agent/dpof/data.py's _econ_calendar_read_sync() ("missing is never zero") and
the companion DPOF-side tests in tests/dpof/test_data.py
(test_econ_blackout_fires_around_fred_sourced_cpi_event and friends).

No network. httpx.AsyncClient is monkeypatched wholesale; uw_flow_intel._uw_get
is monkeypatched per-test (it is imported lazily inside econ_calendar.py, so
patching the attribute on the real uw_flow_intel module is sufficient and
never touches the network itself). agent.config.config's fred_api_key /
uw_api_key are monkeypatched per-test, never read from a real .env.
"""

import asyncio
from datetime import date, datetime, timedelta

import pytest

import agent.integrations.econ_calendar as ec
from agent.config import config


def _run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════
#  Fakes
# ═══════════════════════════════════════════════════════════════════════════

class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        return self._json_data


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient as an async context manager. `outcome`
    is a mutable single-slot holder (a list) so each test can configure what
    the next `.get()` call returns/raises without needing a real client."""

    def __init__(self, outcome, calls, **_kwargs):
        self._outcome = outcome
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, url, params=None):
        self._calls.append((url, params))
        outcome = self._outcome[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _install_fake_httpx(monkeypatch, outcome):
    """Patches the real httpx module's AsyncClient (econ_calendar.py does
    `import httpx` lazily inside fetch_fred_calendar, resolving to the same
    sys.modules['httpx'] this patches). Returns the calls list for assertions.
    """
    import httpx as real_httpx
    calls: list = []
    holder = [outcome]

    def _factory(**kwargs):
        return _FakeAsyncClient(holder, calls, **kwargs)

    monkeypatch.setattr(real_httpx, "AsyncClient", _factory)
    return calls, holder


def _fred_release_dates_payload(items):
    return {"release_dates": items}


async def _fake_uw_get_factory(payload):
    async def _fake_uw_get(endpoint, params=None, *, skip_moltbot_budget=False):
        return payload
    return _fake_uw_get


def _patch_uw(monkeypatch, payload):
    """payload=None simulates every _uw_get failure mode (bare {} — see
    econ_calendar._fetch_uw_calendar_raw's docstring on why that's the
    signal); payload=dict simulates a genuine 200 response."""
    from agent.integrations import uw_flow_intel

    async def _fake_uw_get(endpoint, params=None, *, skip_moltbot_budget=False):
        return {} if payload is None else payload

    monkeypatch.setattr(uw_flow_intel, "_uw_get", _fake_uw_get)


@pytest.fixture(autouse=True)
def _keys_and_budgets(monkeypatch):
    """Both keys configured by default; individual tests clear one/both.
    Budgets are already redirected to a per-session temp file by the repo-wide
    tests/conftest.py, so check_and_record() here never touches data/."""
    monkeypatch.setattr(config, "uw_api_key", "uw-test-key")
    monkeypatch.setattr(config, "fred_api_key", "fred-test-key")


# ═══════════════════════════════════════════════════════════════════════════
#  FRED response parsing / release-time + importance mapping
# ═══════════════════════════════════════════════════════════════════════════

def test_normalize_fred_release_known_id_maps_time_and_importance():
    ev = ec._normalize_fred_release({"release_id": 10, "release_name": "Consumer Price Index", "date": "2026-10-13"})
    assert ev is not None
    assert ev["name"] == "CPI (Consumer Price Index)"
    assert ev["importance"] == 3
    assert ev["source"] == "fred"
    assert ev["datetime_et"].startswith("2026-10-13T08:30:00")


def test_normalize_fred_release_unknown_id_skipped():
    """release_ids outside the curated table are dropped, not defaulted to
    importance 1 in the output -- they simply never become a row, so
    macro_research/fed_analysis/news_intel aren't flooded with the ~300+
    irrelevant releases /releases/dates also returns."""
    assert ec._normalize_fred_release({"release_id": 999999, "release_name": "Some Regional Fed Survey", "date": "2026-10-13"}) is None


def test_normalize_fred_release_bad_date_skipped():
    assert ec._normalize_fred_release({"release_id": 10, "date": "not-a-date"}) is None
    assert ec._normalize_fred_release({"release_id": 10, "date": None}) is None
    assert ec._normalize_fred_release({"release_id": 10}) is None


def test_normalize_fred_release_bad_release_id_skipped():
    assert ec._normalize_fred_release({"release_id": "not-an-int", "date": "2026-10-13"}) is None
    assert ec._normalize_fred_release("not-a-dict") is None


def test_fred_release_table_importance_matches_classifier_semantics():
    """Every curated release's canonical `name` independently classifies to
    the SAME importance _classify_importance() would assign it (defense in
    depth — see _FRED_RELEASE_TABLE's module comment on why importance is
    still set explicitly from the table, never derived from the name)."""
    for release_id, meta in ec._FRED_RELEASE_TABLE.items():
        classified = ec._classify_importance(meta["name"])
        assert classified == meta["importance"], (
            f"release_id={release_id} name={meta['name']!r}: table says importance="
            f"{meta['importance']} but _classify_importance() says {classified}"
        )


def test_fred_release_table_high_importance_set_matches_docstring_contract():
    """HIGH=3 set should be FOMC/CPI/PPI/NFP/GDP/PCE/Retail Sales per
    _classify_importance()'s own docstring contract — no more, no less,
    among the curated releases."""
    high = {meta["name"] for meta in ec._FRED_RELEASE_TABLE.values() if meta["importance"] == 3}
    expected_fragments = ["FOMC", "CPI", "PPI", "Non-Farm Payrolls", "GDP", "PCE", "Retail Sales"]
    for frag in expected_fragments:
        assert any(frag in name for name in high), f"expected a HIGH-importance release containing {frag!r}"
    assert len(high) == len(expected_fragments)


# ═══════════════════════════════════════════════════════════════════════════
#  fetch_fred_calendar: unavailable vs genuine-empty vs populated
# ═══════════════════════════════════════════════════════════════════════════

def test_fetch_fred_calendar_no_key_returns_none(monkeypatch):
    monkeypatch.setattr(config, "fred_api_key", "")
    result = _run(ec.fetch_fred_calendar(days_ahead=7))
    assert result is None


def test_fetch_fred_calendar_http_error_returns_none(monkeypatch):
    _install_fake_httpx(monkeypatch, _FakeResponse(status_code=503, json_data={}))
    result = _run(ec.fetch_fred_calendar(days_ahead=7))
    assert result is None


def test_fetch_fred_calendar_network_exception_returns_none(monkeypatch):
    _install_fake_httpx(monkeypatch, ConnectionError("boom"))
    result = _run(ec.fetch_fred_calendar(days_ahead=7))
    assert result is None


def test_fetch_fred_calendar_bad_shape_returns_none(monkeypatch):
    _install_fake_httpx(monkeypatch, _FakeResponse(200, {"release_dates": "not-a-list"}))
    result = _run(ec.fetch_fred_calendar(days_ahead=7))
    assert result is None


def test_fetch_fred_calendar_genuine_empty_returns_empty_list(monkeypatch):
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload([])))
    result = _run(ec.fetch_fred_calendar(days_ahead=7))
    assert result == []


def test_fetch_fred_calendar_populated_filters_to_curated_and_window(monkeypatch):
    today = datetime.now(ec._NY).date()
    within = (today + timedelta(days=2)).isoformat()
    outside = (today + timedelta(days=30)).isoformat()  # past days_ahead=7 cutoff
    items = [
        {"release_id": 10, "release_name": "Consumer Price Index", "date": within},   # curated, in window
        {"release_id": 999999, "release_name": "Some Regional Survey", "date": within},  # not curated
        {"release_id": 46, "release_name": "Producer Price Index", "date": outside},  # curated but out of window
    ]
    calls, _holder = _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload(items)))
    result = _run(ec.fetch_fred_calendar(days_ahead=7))
    assert result is not None
    names = [ev["name"] for ev in result]
    assert names == ["CPI (Consumer Price Index)"]
    # sent the params the FRED docs/STEP2 spec names
    assert len(calls) == 1
    _url, params = calls[0]
    for key in ("api_key", "file_type", "realtime_start", "realtime_end", "include_release_dates_with_no_data", "limit"):
        assert key in params


# ═══════════════════════════════════════════════════════════════════════════
#  fetch_fomc_static
# ═══════════════════════════════════════════════════════════════════════════

def test_fomc_static_dates_within_2026_2027(monkeypatch):
    _freeze_clock(monkeypatch, 2026, 1, 1, 12, 0)  # before every 2026 date, so the lower bound drops none
    events = ec.fetch_fomc_static(days_ahead=3650)  # ~10y window: return everything hardcoded
    assert len(events) == 16  # 8 in 2026 + 8 in 2027
    for ev in events:
        assert ev["source"] == "fomc_static"
        assert ev["importance"] == 3
        dt = datetime.fromisoformat(ev["datetime_et"])
        assert dt.year in (2026, 2027)
        assert dt.hour == 14 and dt.minute == 0
    years = sorted({datetime.fromisoformat(ev["datetime_et"]).year for ev in events})
    assert years == [2026, 2027]


def test_fomc_static_respects_days_ahead_window():
    events_wide = ec.fetch_fomc_static(days_ahead=3650)
    events_narrow = ec.fetch_fomc_static(days_ahead=0)
    assert len(events_narrow) <= len(events_wide)


# ═══════════════════════════════════════════════════════════════════════════
#  _merge_events / _dedup_key
# ═══════════════════════════════════════════════════════════════════════════

def test_merge_events_prefers_primary_on_same_day_same_importance_bucket():
    primary = [{"name": "CPI (UW)", "importance": 3, "datetime_et": "2026-10-13T08:30:00-04:00", "source": "uw"}]
    secondary = [{"name": "CPI (Consumer Price Index)", "importance": 3, "datetime_et": "2026-10-13T08:30:00-04:00", "source": "fred"}]
    merged = ec._merge_events(primary, secondary)
    assert len(merged) == 1
    assert merged[0]["source"] == "uw"


def test_merge_events_keeps_distinct_days():
    primary = [{"name": "CPI (UW)", "importance": 3, "datetime_et": "2026-10-13T08:30:00-04:00", "source": "uw"}]
    secondary = [{"name": "FOMC", "importance": 3, "datetime_et": "2026-10-28T14:00:00-04:00", "source": "fomc_static"}]
    merged = ec._merge_events(primary, secondary)
    assert len(merged) == 2


def test_merge_events_keeps_distinct_importance_same_day():
    primary = [{"name": "CPI (UW)", "importance": 3, "datetime_et": "2026-10-13T08:30:00-04:00", "source": "uw"}]
    secondary = [{"name": "Fed speaker", "importance": 2, "datetime_et": "2026-10-13T12:00:00-04:00", "source": "fred"}]
    merged = ec._merge_events(primary, secondary)
    assert len(merged) == 2


# ═══════════════════════════════════════════════════════════════════════════
#  fetch_calendar_with_fallback orchestration
# ═══════════════════════════════════════════════════════════════════════════

def test_uw_fail_falls_back_to_fred(monkeypatch):
    """UW unavailable (bare {} — no key/budget/HTTP/exception, all
    indistinguishable per _uw_get's contract) -> FRED rows used, source_ok True."""
    _patch_uw(monkeypatch, payload=None)
    today = datetime.now(ec._NY).date()
    items = [{"release_id": 10, "release_name": "Consumer Price Index", "date": (today + timedelta(days=1)).isoformat()}]
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload(items)))

    events, source_ok = _run(ec.fetch_calendar_with_fallback(days_ahead=7))
    assert source_ok is True
    fred_rows = [ev for ev in events if ev["source"] == "fred"]
    assert len(fred_rows) == 1
    assert fred_rows[0]["name"] == "CPI (Consumer Price Index)"


def test_uw_empty_falls_back_to_fred(monkeypatch):
    """UW succeeds with a genuine {"data": []} (a real empty week, not a
    failure) -> fallback to FRED still triggers per STEP2 ('fails OR returns
    nothing'), source_ok True."""
    _patch_uw(monkeypatch, payload={"data": []})
    today = datetime.now(ec._NY).date()
    items = [{"release_id": 46, "release_name": "Producer Price Index", "date": (today + timedelta(days=1)).isoformat()}]
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload(items)))

    events, source_ok = _run(ec.fetch_calendar_with_fallback(days_ahead=7))
    assert source_ok is True
    assert any(ev["source"] == "fred" and ev["name"].startswith("PPI") for ev in events)


def test_uw_success_with_events_skips_fred_entirely(monkeypatch):
    """UW returns real events -> FRED is never called (no fallback trigger)."""
    now_et = datetime.now(ec._NY)
    uw_payload = {"data": [{
        "event": "Initial Jobless Claims", "type": "report",
        "time": now_et.strftime("%Y-%m-%dT%H:%M:%SZ"), "forecast": None, "prev": None,
    }]}
    _patch_uw(monkeypatch, payload=uw_payload)
    calls, _holder = _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload([])))

    events, source_ok = _run(ec.fetch_calendar_with_fallback(days_ahead=7))
    assert source_ok is True
    assert calls == []  # FRED never called
    assert any(ev["source"] == "uw" for ev in events)


def test_both_uw_and_fred_present_dedupes_and_prefers_uw(monkeypatch):
    """When UW is down and BOTH FRED and the static FOMC list would produce
    an entry for the same FOMC decision day, FRED's row wins over the static
    row (FRED is queried/merged before the static list in
    fetch_calendar_with_fallback -- 'prefer UW rows when fresh' generalizes to
    'prefer the earlier, more-authoritative source')."""
    _patch_uw(monkeypatch, payload=None)  # UW down
    fomc_day = "2026-10-28"  # one of the hardcoded FOMC dates
    items = [{"release_id": 101, "release_name": "FOMC Press Release", "date": fomc_day}]
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload(items)))

    events, source_ok = _run(ec.fetch_calendar_with_fallback(days_ahead=60))
    assert source_ok is True
    fomc_bucket = [ev for ev in events if ev["datetime_et"].startswith(fomc_day)]
    assert len(fomc_bucket) == 1  # deduped, not two rows for the same day
    assert fomc_bucket[0]["source"] == "fred"


def test_no_key_at_all_is_unavailable(monkeypatch):
    """Neither UW nor FRED is configured -> source_ok False (the static FOMC
    list alone must not flip this — see fetch_calendar_with_fallback's
    docstring)."""
    monkeypatch.setattr(config, "uw_api_key", "")
    monkeypatch.setattr(config, "fred_api_key", "")
    events, source_ok = _run(ec.fetch_calendar_with_fallback(days_ahead=365))
    assert source_ok is False
    # the static FOMC list may still be present in `events` (informational —
    # a year-wide window is guaranteed to catch at least one hardcoded FOMC
    # date), but it must never be mistaken for "a live source worked":
    assert len(events) > 0
    assert all(ev["source"] == "fomc_static" for ev in events)


def test_both_uw_and_fred_fail_is_unavailable(monkeypatch):
    _patch_uw(monkeypatch, payload=None)
    _install_fake_httpx(monkeypatch, ConnectionError("fred is down too"))
    events, source_ok = _run(ec.fetch_calendar_with_fallback(days_ahead=7))
    assert source_ok is False


# ═══════════════════════════════════════════════════════════════════════════
#  run_calendar_refresh: only saves (refreshes cache mtime) when source_ok
# ═══════════════════════════════════════════════════════════════════════════

class _FakeBot:
    async def send_message(self, *args, **kwargs):
        return None


class _FakeContext:
    def __init__(self):
        self.bot = _FakeBot()


def _within_time_guard_now(monkeypatch):
    """run_calendar_refresh() no-ops outside 5AM-10PM ET — pin `now` inside
    the window so tests are time-of-day independent."""
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 25, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(ec, "datetime", _FixedDatetime)


def test_run_calendar_refresh_saves_when_fred_only_available(monkeypatch, tmp_path):
    _within_time_guard_now(monkeypatch)
    cache_path = tmp_path / "econ_calendar.json"
    monkeypatch.setattr(ec, "CALENDAR_CACHE_PATH", str(cache_path))

    _patch_uw(monkeypatch, payload=None)  # UW down
    today = date(2026, 9, 25)
    items = [{"release_id": 10, "release_name": "Consumer Price Index", "date": (today + timedelta(days=1)).isoformat()}]
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload(items)))

    saved = {}
    monkeypatch.setattr(ec, "save_calendar", lambda events: saved.setdefault("events", events))

    _run(ec.run_calendar_refresh(_FakeContext()))
    assert "events" in saved  # save_calendar WAS called -> cache mtime would refresh
    assert any(ev["source"] == "fred" for ev in saved["events"])


def test_run_calendar_refresh_does_not_save_when_neither_source_available(monkeypatch, tmp_path):
    _within_time_guard_now(monkeypatch)
    cache_path = tmp_path / "econ_calendar.json"
    monkeypatch.setattr(ec, "CALENDAR_CACHE_PATH", str(cache_path))

    _patch_uw(monkeypatch, payload=None)  # UW down
    _install_fake_httpx(monkeypatch, ConnectionError("fred is down too"))  # FRED down

    save_calls = []
    monkeypatch.setattr(ec, "save_calendar", lambda events: save_calls.append(events))

    _run(ec.run_calendar_refresh(_FakeContext()))
    assert save_calls == []  # save_calendar NOT called -> stale cache keeps aging toward DPOF-unhealthy


def test_merge_keeps_distinct_times_same_day_and_dedupes_same_time():
    """Regression: a same-day CPI 08:30 and FOMC 14:00 must both survive the
    merge (the DPOF blackout is a window around each event's time)."""
    from agent.integrations.econ_calendar import _merge_events

    cpi_uw = {"datetime_et": "2026-10-28T08:30:00-04:00", "importance": 3, "name": "CPI", "source": "uw"}
    cpi_fred = {"datetime_et": "2026-10-28T08:30:00-04:00", "importance": 3, "name": "Consumer Price Index", "source": "fred"}
    fomc = {"datetime_et": "2026-10-28T14:00:00-04:00", "importance": 3, "name": "FOMC Rate Decision", "source": "fomc_static"}
    merged = _merge_events([cpi_uw], [cpi_fred, fomc])
    assert [e["source"] for e in merged] == ["uw", "fomc_static"]
    # UW down: FRED CPI then static FOMC on the same day -> both kept.
    merged2 = _merge_events(_merge_events([], [cpi_fred]), [fomc])
    assert {e["source"] for e in merged2} == {"fred", "fomc_static"}


# ═══════════════════════════════════════════════════════════════════════════
#  R-14 (2026-10-12): phantom FRED FOMC rows for release 101
# ═══════════════════════════════════════════════════════════════════════════

def _freeze_clock(monkeypatch, y, m, d, hh, mm):
    """Pin econ_calendar's `datetime.now()` (ET-aware) to a fixed instant."""
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(y, m, d, hh, mm, 0, tzinfo=tz)

    monkeypatch.setattr(ec, "datetime", _FixedDatetime)


def _week(start: date, n: int = 7) -> list[str]:
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def _fred_items(release_id: int, days: list[str]) -> list[dict]:
    return [{"release_id": release_id, "release_name": "x", "date": d} for d in days]


def _redirect_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "econ_calendar.json"
    monkeypatch.setattr(ec, "CALENDAR_CACHE_PATH", str(cache_path))
    return cache_path


def test_fred_release_101_phantom_week_is_dropped(monkeypatch):
    """Observed artefact: release 101 on seven consecutive days (10-12..10-18),
    none of them the real 10-28 FOMC decision -> zero FOMC rows."""
    _freeze_clock(monkeypatch, 2026, 10, 12, 12, 0)
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload(
        _fred_items(101, _week(date(2026, 10, 12))))))
    result = _run(ec.fetch_fred_calendar(days_ahead=30))
    assert result == []


def test_fred_release_101_keeps_only_static_decision_day(monkeypatch):
    """Seven consecutive days 10-25..10-31 contain the real 10-28 decision:
    only that date survives."""
    _freeze_clock(monkeypatch, 2026, 10, 12, 12, 0)
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload(
        _fred_items(101, _week(date(2026, 10, 25))))))
    result = _run(ec.fetch_fred_calendar(days_ahead=30))
    assert [ev["datetime_et"][:10] for ev in result] == ["2026-10-28"]
    assert result[0]["datetime_et"].startswith("2026-10-28T14:00:00")


def test_fred_guard_does_not_touch_other_releases(monkeypatch):
    """Release 10 (CPI) on seven consecutive days all survive: the static
    decision-day gate is FOMC-only."""
    _freeze_clock(monkeypatch, 2026, 10, 12, 12, 0)
    days = _week(date(2026, 10, 12))
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload(
        _fred_items(10, days))))
    result = _run(ec.fetch_fred_calendar(days_ahead=30))
    assert sorted(ev["datetime_et"][:10] for ev in result) == days
    assert all(ev["name"] == "CPI (Consumer Price Index)" for ev in result)


def test_is_static_fomc_day_uses_et_calendar_date():
    """Mutant guard: comparing the UTC date. A 14:00 ET row never differs
    between ET and UTC dates, so this case uses 20:30 ET (= 00:30 UTC next
    day): its ET date 2026-10-28 is a decision day, its UTC date 10-29 is not."""
    late_et = datetime(2026, 10, 28, 20, 30, tzinfo=ec._NY)
    assert late_et.astimezone(ec._UTC).date().isoformat() == "2026-10-29"
    assert ec._is_static_fomc_day(late_et) is True
    assert ec._is_static_fomc_day(datetime(2026, 10, 12, 14, 0, tzinfo=ec._NY)) is False


def test_consecutive_day_run_logs_info_without_dropping(monkeypatch, caplog):
    """A release with >3 consecutive days is logged at INFO (visible in
    production) and the other curated release's rows are not dropped."""
    _freeze_clock(monkeypatch, 2026, 10, 12, 12, 0)
    items = _fred_items(101, _week(date(2026, 10, 12))) + _fred_items(10, _week(date(2026, 10, 12), 2))
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload(items)))
    with caplog.at_level("INFO", logger="openclaw.econ_calendar"):
        result = _run(ec.fetch_fred_calendar(days_ahead=30))
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert any("FRED release 101" in m and "7 consecutive" in m for m in msgs)
    assert not any("FRED release 10 " in m for m in msgs)  # 2 days: below threshold
    assert len(result) == 2  # the two CPI rows survive


def test_fomc_static_drops_dates_older_than_one_day(monkeypatch):
    _freeze_clock(monkeypatch, 2026, 10, 12, 12, 0)
    dates = [datetime.fromisoformat(ev["datetime_et"]).date().isoformat()
             for ev in ec.fetch_fomc_static(days_ahead=3650)]
    assert "2026-09-16" not in dates  # > 1 day in the past
    assert "2026-10-28" in dates      # future decision date kept
    assert all(d >= "2026-10-11" for d in dates)


def test_no_fomc_row_in_48h_window_on_2026_10_12(monkeypatch, tmp_path):
    """End to end: UW down, FRED returns the phantom week -> saved cache ->
    get_upcoming_events(48) at 2026-10-12 12:00 ET has no FOMC event."""
    _freeze_clock(monkeypatch, 2026, 10, 12, 12, 0)
    cache_path = _redirect_cache(monkeypatch, tmp_path)
    _patch_uw(monkeypatch, payload=None)
    _install_fake_httpx(monkeypatch, _FakeResponse(200, _fred_release_dates_payload(
        _fred_items(101, _week(date(2026, 10, 12))))))
    events, _ok = _run(ec.fetch_calendar_with_fallback(days_ahead=7))
    ec.save_calendar(events)
    assert cache_path.exists()
    upcoming = ec.get_upcoming_events(48)
    assert not any("FOMC" in ev["name"] or ev["source"] == "fomc_static" for ev in upcoming)


def test_real_fomc_row_in_48h_window_on_2026_10_27(monkeypatch, tmp_path):
    """The real 2026-10-28 14:00 ET decision is returned from a 10-27 noon clock."""
    _freeze_clock(monkeypatch, 2026, 10, 27, 12, 0)
    _redirect_cache(monkeypatch, tmp_path)
    _patch_uw(monkeypatch, payload=None)
    _install_fake_httpx(monkeypatch, ConnectionError("fred down"))
    events, _ok = _run(ec.fetch_calendar_with_fallback(days_ahead=7))
    ec.save_calendar(events)
    upcoming = ec.get_upcoming_events(48)
    fomc = [ev for ev in upcoming if ev["source"] == "fomc_static"]
    assert len(fomc) == 1
    assert fomc[0]["datetime_et"].startswith("2026-10-28T14:00:00")
