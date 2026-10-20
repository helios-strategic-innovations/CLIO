"""Tests for agent/integrations/econ_surprise.py — refresh_econ_surprise()
cadence orchestration (fast-poll vs slow-refresh vs idle), the
overlapping-tick guard, alert dedup, the sync/mtime-cached readers
(get_recent_surprises / get_upcoming_releases), and bootstrap_history().

No network: _fetch_all_sources / _fetch_tradingview_raw are monkeypatched
directly (their own HTTP/shape-validation behavior is covered by
tests/test_econ_surprise_sources.py). econ_calendar's get_todays_events()/
get_upcoming_events() are monkeypatched so this module's own cadence logic
is tested in isolation from econ_calendar's cache state.
"""

import asyncio
import os
import threading
from datetime import datetime, timedelta

import pytest

import agent.integrations.econ_surprise as es
from agent.integrations import econ_calendar as ec


def _run(coro):
    return asyncio.run(coro)


class _FakeBot:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, chat_id, text, **_kwargs):
        self.sent.append((chat_id, text))


class _FakeContext:
    def __init__(self):
        self.bot = _FakeBot()


@pytest.fixture(autouse=True)
def _isolated_state_file(monkeypatch, tmp_path):
    path = str(tmp_path / "econ_surprise.json")
    monkeypatch.setenv("CLIO_ECON_SURPRISE_FILE", path)
    yield path


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Fast-poll loop tests would otherwise take real wall-clock seconds.
    Patches the actual stdlib asyncio module's sleep (es.asyncio IS that
    module) -- reverted automatically at test teardown."""
    async def _fast_sleep(_seconds):
        return None
    monkeypatch.setattr(es.asyncio, "sleep", _fast_sleep)


@pytest.fixture(autouse=True)
def _fake_chat_ids(monkeypatch):
    from agent.config import config as _cfg
    monkeypatch.setattr(_cfg, "telegram_notify_chat_ids", ["12345"])


def _seed_state(**overrides) -> dict:
    state = es._default_state()
    state.update(overrides)
    es._write_state_blocking(state)
    return state


# Deterministic weekday-active-window / off-window reference times (item 5,
# 2026-09-25 pass-2 fix) -- 2026-09-23 is a Wednesday, 2026-09-26 a Saturday
# (verified via datetime(...).strftime("%A")). Freezing `es.datetime.now()`
# to one of these makes slow-refresh-cadence tests deterministic regardless
# of the real wall-clock time the suite happens to run at.
_WEEKDAY_ACTIVE_NOW = datetime(2026, 9, 23, 10, 0, 0, tzinfo=es._NY)   # Wed 10:00 ET -> 30min cadence
_WEEKEND_NOW = datetime(2026, 9, 26, 10, 0, 0, tzinfo=es._NY)          # Sat 10:00 ET -> 3h cadence


def _freeze_time(monkeypatch, fixed_dt: datetime):
    """Patches es.datetime so datetime.now(tz) always returns `fixed_dt`
    (converted to `tz` if given)."""
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_dt.astimezone(tz) if tz is not None else fixed_dt
    monkeypatch.setattr(es, "datetime", _FixedDatetime)
    return fixed_dt


# ═══════════════════════════════════════════════════════════════════════════
#  refresh_econ_surprise: mode selection
# ═══════════════════════════════════════════════════════════════════════════

def test_refresh_idle_when_no_window_and_recently_slow_refreshed(monkeypatch):
    now = datetime.now(es._NY)
    _seed_state(last_slow_refresh_at=now.isoformat())
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    fetch_called = {"n": 0}
    async def _fake_fetch_all(start, end, state):
        fetch_called["n"] += 1
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    result = _run(es.refresh_econ_surprise(None))
    assert result == {"mode": "idle", "active_windows": 0, "new_surprises": 0, "events": []}
    assert fetch_called["n"] == 0  # no network work at all when idle


def test_refresh_slow_path_first_time_calls_fetch_and_persists(monkeypatch):
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    fetch_calls = []
    async def _fake_fetch_all(start, end, state):
        fetch_calls.append((start, end))
        return {"nasdaq": [{
            "event": "CPI MoM", "release_time_et": "2026-10-13T08:30:00-04:00",
            "actual": 0.4, "actual_unit": "%", "consensus": 0.3, "consensus_unit": "%",
            "previous": 0.2, "previous_unit": "%", "importance": 3, "source": "nasdaq",
        }], "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    ctx = _FakeContext()
    result = _run(es.refresh_econ_surprise(ctx))
    assert result["mode"] == "slow_refresh"
    assert result["new_surprises"] == 1
    assert result["events"] == ["CPI MoM"]
    assert len(fetch_calls) == 1

    assert len(ctx.bot.sent) == 1
    assert "CPI MoM" in ctx.bot.sent[0][1]
    assert "0.4" in ctx.bot.sent[0][1]

    saved = es._read_state_cached()
    assert len(saved["surprises"]) == 1
    assert saved["last_slow_refresh_at"] is not None


def test_refresh_slow_path_skipped_when_recent(monkeypatch):
    now = datetime.now(es._NY)
    _seed_state(last_slow_refresh_at=(now - timedelta(minutes=5)).isoformat())
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    called = {"n": 0}
    async def _fake_fetch_all(start, end, state):
        called["n"] += 1
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    result = _run(es.refresh_econ_surprise(None))
    assert result["mode"] == "idle"
    assert called["n"] == 0


def test_refresh_slow_path_runs_again_after_interval_elapsed(monkeypatch):
    """31 min only clears the 30min ACTIVE-window threshold -- freeze `now`
    to a weekday-active reference time so this is deterministic regardless
    of the real wall-clock time the suite runs at (item 5, pass-2 fix)."""
    now = _freeze_time(monkeypatch, _WEEKDAY_ACTIVE_NOW)
    _seed_state(last_slow_refresh_at=(now - timedelta(minutes=31)).isoformat())
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    called = {"n": 0}
    async def _fake_fetch_all(start, end, state):
        called["n"] += 1
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    result = _run(es.refresh_econ_surprise(None))
    assert result["mode"] == "slow_refresh"
    assert called["n"] == 1


def test_refresh_fast_poll_is_one_tick_per_invocation(monkeypatch):
    """CRITICAL fix (2026-09-25): a single refresh_econ_surprise() call must
    do AT MOST one fetch and return immediately — no internal sleep loop.
    Simulates the orchestrator's 15s-cadence ticks as two SEPARATE calls:
    tick 1 (no actual yet) must NOT write state or alert; tick 2 (actual
    appears) must write + alert IMMEDIATELY on that same call, not after
    some later window-close."""
    now = datetime.now(es._NY)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [{
        "name": "Initial Jobless Claims", "importance": 2, "datetime_et": now.isoformat(),
    }])

    call_count = {"n": 0}
    async def _fake_fetch_all(start, end, state):
        call_count["n"] += 1
        if call_count["n"] < 2:
            actual, actual_unit = None, None
        else:
            actual, actual_unit = 197.0, "K"
        return {"nasdaq": [{
            "event": "Initial Jobless Claims", "release_time_et": now.isoformat(),
            "actual": actual, "actual_unit": actual_unit, "consensus": 201.0, "consensus_unit": "K",
            "previous": 198.0, "previous_unit": "K", "importance": 2, "source": "nasdaq",
        }], "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    ctx = _FakeContext()

    # Tick 1: no actual yet -- exactly ONE fetch call, no surprise, no alert.
    result1 = _run(es.refresh_econ_surprise(ctx))
    assert result1["mode"] == "fast_poll"
    assert result1["active_windows"] == 1
    assert result1["new_surprises"] == 0
    assert call_count["n"] == 1  # exactly one fetch this invocation -- not a multi-tick loop
    assert len(ctx.bot.sent) == 0

    # Tick 2 (a separate invocation, as the orchestrator's next 15s job fire
    # would be): actual now present -- must be recorded + alerted on THIS
    # call, immediately, not deferred to some later "window close".
    result2 = _run(es.refresh_econ_surprise(ctx))
    assert result2["mode"] == "fast_poll"
    assert result2["new_surprises"] == 1
    assert call_count["n"] == 2
    assert len(ctx.bot.sent) == 1

    # And it's actually on disk right now (not buffered in memory for later).
    saved = es._read_state_cached()
    assert len(saved["surprises"]) == 1
    assert saved["surprises"][0]["event"] == "Initial Jobless Claims"


def test_refresh_idle_tick_never_loads_or_writes_state(monkeypatch):
    """Idle path must be cheap: no full state load, no write."""
    now = datetime.now(es._NY)
    _seed_state(last_slow_refresh_at=now.isoformat())
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    load_calls = {"n": 0}
    write_calls = {"n": 0}
    real_load = es._load_state_for_write
    real_write = es._write_state_blocking

    def _spy_load():
        load_calls["n"] += 1
        return real_load()

    def _spy_write(state):
        write_calls["n"] += 1
        return real_write(state)

    monkeypatch.setattr(es, "_load_state_for_write", _spy_load)
    monkeypatch.setattr(es, "_write_state_blocking", _spy_write)

    result = _run(es.refresh_econ_surprise(None))
    assert result["mode"] == "idle"
    assert load_calls["n"] == 0
    assert write_calls["n"] == 0


def test_refresh_fast_tick_skips_write_when_nothing_changed(monkeypatch):
    """A re-poll that returns byte-identical data (e.g. a Fed-speaker row
    that never gets a consensus) must not touch disk on the second tick."""
    now = datetime.now(es._NY)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [{
        "name": "FOMC Member Speaks", "importance": 2, "datetime_et": now.isoformat(),
    }])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": [{
            "event": "FOMC Member Speaks", "release_time_et": now.isoformat(),
            "actual": None, "actual_unit": None, "consensus": None, "consensus_unit": None,
            "previous": None, "previous_unit": None, "importance": 2, "source": "nasdaq",
        }], "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    write_calls = {"n": 0}
    real_write = es._write_state_blocking

    def _spy_write(state):
        write_calls["n"] += 1
        return real_write(state)
    monkeypatch.setattr(es, "_write_state_blocking", _spy_write)

    result1 = _run(es.refresh_econ_surprise(None))
    assert result1["new_surprises"] == 0
    assert write_calls["n"] == 1  # first tick DOES write (consensus_table gained a fresh entry)

    result2 = _run(es.refresh_econ_surprise(None))
    assert result2["new_surprises"] == 0
    assert write_calls["n"] == 1  # second tick: identical data -> no write


def test_refresh_low_importance_event_does_not_trigger_fast_poll(monkeypatch):
    now = datetime.now(es._NY)
    # Seed a recent slow refresh so the ONLY thing under test is "does this
    # importance=1 event trigger fast-poll mode" (isolates from the
    # first-run-always-slow-refreshes behavior covered elsewhere).
    _seed_state(last_slow_refresh_at=now.isoformat())
    monkeypatch.setattr(ec, "get_todays_events", lambda: [{
        "name": "Baker Hughes Oil Rig Count", "importance": 1, "datetime_et": now.isoformat(),
    }])
    called = {"n": 0}
    async def _fake_fetch_all(start, end, state):
        called["n"] += 1
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    result = _run(es.refresh_econ_surprise(None))
    assert result["mode"] == "idle"
    assert called["n"] == 0


def test_refresh_event_outside_window_does_not_trigger_fast_poll(monkeypatch):
    now = datetime.now(es._NY)
    _seed_state(last_slow_refresh_at=now.isoformat())
    far_event = now + timedelta(hours=3)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [{
        "name": "CPI MoM", "importance": 3, "datetime_et": far_event.isoformat(),
    }])
    called = {"n": 0}
    async def _fake_fetch_all(start, end, state):
        called["n"] += 1
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    result = _run(es.refresh_econ_surprise(None))
    assert result["mode"] == "idle"
    assert called["n"] == 0


def test_refresh_econ_calendar_failure_does_not_crash(monkeypatch):
    def _boom():
        raise RuntimeError("econ_calendar cache unreadable")
    monkeypatch.setattr(ec, "get_todays_events", _boom)

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    result = _run(es.refresh_econ_surprise(None))
    assert result["mode"] == "slow_refresh"  # falls through to slow path (no active windows found)


# ═══════════════════════════════════════════════════════════════════════════
#  _slow_refresh_interval_minutes: weekday-active vs off-window/weekend
#  cadence (2026-09-25 pass-2 fix, item 5 — was a flat 30min/24-7 cadence,
#  5 budget calls/cycle burned needlessly overnight and on weekends).
# ═══════════════════════════════════════════════════════════════════════════

def test_slow_refresh_interval_weekday_active_window():
    # Wed 10:00 ET, inside 04:00-20:00
    assert es._slow_refresh_interval_minutes(_WEEKDAY_ACTIVE_NOW) == 30


def test_slow_refresh_interval_weekday_active_window_boundaries():
    wed = _WEEKDAY_ACTIVE_NOW.date()
    start_edge = datetime(wed.year, wed.month, wed.day, 4, 0, 0, tzinfo=es._NY)
    end_edge = datetime(wed.year, wed.month, wed.day, 20, 0, 0, tzinfo=es._NY)
    just_before_start = datetime(wed.year, wed.month, wed.day, 3, 59, 0, tzinfo=es._NY)
    assert es._slow_refresh_interval_minutes(start_edge) == 30      # 04:00 itself is inside (>=)
    assert es._slow_refresh_interval_minutes(end_edge) == 180       # 20:00 itself is OUTSIDE (< end)
    assert es._slow_refresh_interval_minutes(just_before_start) == 180


def test_slow_refresh_interval_weekday_overnight_off_window():
    wed = _WEEKDAY_ACTIVE_NOW.date()
    overnight = datetime(wed.year, wed.month, wed.day, 2, 0, 0, tzinfo=es._NY)
    evening = datetime(wed.year, wed.month, wed.day, 22, 0, 0, tzinfo=es._NY)
    assert es._slow_refresh_interval_minutes(overnight) == 180
    assert es._slow_refresh_interval_minutes(evening) == 180


def test_slow_refresh_interval_weekend_always_off_window():
    # Sat 10:00 ET -- would be "active hours" on a weekday, but weekend wins
    assert es._slow_refresh_interval_minutes(_WEEKEND_NOW) == 180
    sat = _WEEKEND_NOW.date()
    sat_evening = datetime(sat.year, sat.month, sat.day, 18, 0, 0, tzinfo=es._NY)
    assert es._slow_refresh_interval_minutes(sat_evening) == 180


def test_refresh_slow_path_active_window_30min_not_yet_due(monkeypatch):
    """29 min < 30min ACTIVE threshold -> not due yet."""
    now = _freeze_time(monkeypatch, _WEEKDAY_ACTIVE_NOW)
    _seed_state(last_slow_refresh_at=(now - timedelta(minutes=29)).isoformat())
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    called = {"n": 0}
    async def _fake_fetch_all(start, end, state):
        called["n"] += 1
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    result = _run(es.refresh_econ_surprise(None))
    assert result["mode"] == "idle"
    assert called["n"] == 0


def test_refresh_slow_path_weekend_31min_not_due_needs_3h(monkeypatch):
    """31 min clears the ACTIVE (30min) threshold but NOT the weekend/
    off-window (180min) threshold -- on a Saturday this must stay idle."""
    now = _freeze_time(monkeypatch, _WEEKEND_NOW)
    _seed_state(last_slow_refresh_at=(now - timedelta(minutes=31)).isoformat())
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    called = {"n": 0}
    async def _fake_fetch_all(start, end, state):
        called["n"] += 1
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    result = _run(es.refresh_econ_surprise(None))
    assert result["mode"] == "idle"
    assert called["n"] == 0


def test_refresh_slow_path_weekend_181min_is_due(monkeypatch):
    now = _freeze_time(monkeypatch, _WEEKEND_NOW)
    _seed_state(last_slow_refresh_at=(now - timedelta(minutes=181)).isoformat())
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    result = _run(es.refresh_econ_surprise(None))
    assert result["mode"] == "slow_refresh"


# ═══════════════════════════════════════════════════════════════════════════
#  Off-loop I/O on every tick (2026-09-25 pass-2 fix, item 2):
#  _active_release_windows() and _get_last_slow_refresh_at() both do
#  synchronous file I/O (econ_calendar's load_calendar() open()+json.load(),
#  and the mtime-cache's fallback read respectively) and must never run on
#  the event-loop thread.
# ═══════════════════════════════════════════════════════════════════════════

def test_active_release_windows_check_dispatched_off_loop(monkeypatch):
    main_thread = threading.current_thread()
    seen_threads = []

    def _spy(now):
        seen_threads.append(threading.current_thread())
        return []
    monkeypatch.setattr(es, "_active_release_windows", _spy)

    # No active windows -> falls through to the slow-refresh path; mock
    # _fetch_all_sources so this stays hermetic (no real network attempt).
    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    _run(es.refresh_econ_surprise(None))

    assert len(seen_threads) == 1
    assert seen_threads[0] is not main_thread


def test_get_last_slow_refresh_at_check_dispatched_off_loop(monkeypatch):
    main_thread = threading.current_thread()
    seen_threads = []
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])  # no active windows -> reach the slow-refresh check

    def _spy():
        seen_threads.append(threading.current_thread())
        return None
    monkeypatch.setattr(es, "_get_last_slow_refresh_at", _spy)

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    _run(es.refresh_econ_surprise(None))

    assert len(seen_threads) == 1
    assert seen_threads[0] is not main_thread


def test_active_release_windows_itself_does_blocking_file_io_when_uncached(monkeypatch, tmp_path):
    """Sanity check that the thing being dispatched off-loop is REAL blocking
    I/O, not a no-op — econ_calendar.load_calendar() does open()+json.load()
    synchronously when its own mtime cache misses."""
    import json as _json
    cal_path = tmp_path / "econ_calendar.json"
    cal_path.write_text(_json.dumps({"fetched_at": datetime.now(es._NY).isoformat(), "events": []}))
    monkeypatch.setattr(ec, "CALENDAR_CACHE_PATH", str(cal_path))

    # Just confirms the real (unmocked) call path resolves without raising —
    # the off-loop dispatch itself is proven by the two tests above.
    result = es._active_release_windows(datetime.now(es._NY))
    assert result == []


def test_idle_tick_still_no_full_state_load_even_with_off_loop_checks(monkeypatch):
    """The off-loop dispatch of the two cheap checks (item 2) must not
    regress the "idle tick never loads/writes full state" behavior (already
    covered by test_refresh_idle_tick_never_loads_or_writes_state) — belt
    and braces given _do_refresh's structure changed for this fix."""
    now = datetime.now(es._NY)
    _seed_state(last_slow_refresh_at=now.isoformat())
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    load_calls = {"n": 0}
    real_load = es._load_state_for_write

    def _spy_load():
        load_calls["n"] += 1
        return real_load()
    monkeypatch.setattr(es, "_load_state_for_write", _spy_load)

    result = _run(es.refresh_econ_surprise(None))
    assert result["mode"] == "idle"
    assert load_calls["n"] == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Alert dedup + no-context handling
# ═══════════════════════════════════════════════════════════════════════════

def test_alert_dedup_across_separate_refresh_calls(monkeypatch):
    now = datetime.now(es._NY)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [{
        "name": "CPI MoM", "importance": 3, "datetime_et": now.isoformat(),
    }])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": [{
            "event": "CPI MoM", "release_time_et": "2026-10-13T08:30:00-04:00",
            "actual": 0.4, "actual_unit": "%", "consensus": 0.3, "consensus_unit": "%",
            "previous": 0.2, "previous_unit": "%", "importance": 3, "source": "nasdaq",
        }], "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    ctx = _FakeContext()
    result1 = _run(es.refresh_econ_surprise(ctx))
    assert result1["new_surprises"] == 1
    assert len(ctx.bot.sent) == 1

    result2 = _run(es.refresh_econ_surprise(ctx))
    assert result2["new_surprises"] == 0
    assert len(ctx.bot.sent) == 1  # no second alert for the same release


def test_refresh_without_bot_context_logs_only_no_crash(monkeypatch, caplog):
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": [{
            "event": "GDP QoQ", "release_time_et": "2026-10-29T08:30:00-04:00",
            "actual": 3.0, "actual_unit": "level", "consensus": 2.5, "consensus_unit": "level",
            "previous": 2.0, "previous_unit": "level", "importance": 3, "source": "nasdaq",
        }], "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    with caplog.at_level("INFO"):
        result = _run(es.refresh_econ_surprise(None))
    assert result["new_surprises"] == 1
    assert any("GDP QoQ" in r.message for r in caplog.records)


def test_refresh_skips_when_already_in_progress():
    """The asyncio.Lock guard: a refresh already holding the lock means a
    concurrent tick must no-op immediately, never pile up a second fetch."""
    async def _run_guard():
        async with es._refresh_lock:
            return await es.refresh_econ_surprise(None)
    result = _run(_run_guard())
    assert result is None


def test_refresh_wallclock_cap_aborts_never_ending_cycle(monkeypatch):
    monkeypatch.setattr(es, "_WALLCLOCK_CAP_SECONDS", 0.05)

    async def _never_ending(_context):
        await asyncio.Event().wait()  # never resolves; real (short) wait_for timeout below bounds it

    monkeypatch.setattr(es, "_do_refresh", _never_ending)
    result = _run(es.refresh_econ_surprise(None))
    assert result is None


# ═══════════════════════════════════════════════════════════════════════════
#  Importance-gated alerts + digest for multi-entry ticks (2026-09-25
#  pass-2 fix, item 3): only importance>=_IMPORTANCE_MIN_FOR_FAST_POLL
#  surprises alert (all are still recorded); >1 alertable entry from one
#  tick collapses into ONE digest message.
# ═══════════════════════════════════════════════════════════════════════════

def _importance_row(event, importance, actual=0.4, consensus=0.3):
    return {
        "event": event, "release_time_et": datetime.now(es._NY).isoformat(),
        "actual": actual, "actual_unit": "%", "consensus": consensus, "consensus_unit": "%",
        "previous": 0.2, "previous_unit": "%", "importance": importance, "source": "nasdaq",
    }


def test_dispatch_alerts_skips_low_importance_but_still_records():
    """Direct unit test of _dispatch_alerts: importance=1 entries are
    recorded (already happened via _process_rows before this is called) but
    must NOT generate a Telegram send."""
    ctx = _FakeContext()
    low = {"event": "Current Account", "importance": 1, "actual": 0.1, "consensus": 0.1, "unit": "%",
           "surprise_z": None, "market_read": None, "confirmed_by": ["nasdaq"], "source": "nasdaq"}
    _run(es._dispatch_alerts(ctx, [low]))
    assert ctx.bot.sent == []


def test_refresh_low_importance_surprise_recorded_but_not_alerted(monkeypatch):
    """Integration: importance=1 surprises from a real tick land in
    state["surprises"] (DPOF context may still use them) but generate NO
    Telegram alert."""
    now = datetime.now(es._NY)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": [{
            "event": "Current Account", "release_time_et": now.isoformat(),
            "actual": -246.0, "actual_unit": "B", "consensus": -258.0, "consensus_unit": "B",
            "previous": -212.6, "previous_unit": "B", "importance": 1, "source": "nasdaq",
        }], "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    ctx = _FakeContext()
    result = _run(es.refresh_econ_surprise(ctx))
    assert result["new_surprises"] == 1  # recorded

    saved = es._read_state_cached()
    assert len(saved["surprises"]) == 1
    assert saved["surprises"][0]["event"] == "Current Account"
    assert ctx.bot.sent == []  # never alerted


def test_dispatch_alerts_single_alertable_entry_sends_single_message():
    ctx = _FakeContext()
    high = {"event": "CPI MoM", "importance": 3, "actual": 0.4, "consensus": 0.3, "unit": "%",
            "surprise_z": None, "market_read": "hawkish", "confirmed_by": ["nasdaq"], "source": "nasdaq"}
    _run(es._dispatch_alerts(ctx, [high]))
    assert len(ctx.bot.sent) == 1
    assert "CPI MoM" in ctx.bot.sent[0][1]
    assert ctx.bot.sent[0][1].count("\n") == 0  # single-entry format, no digest header


def test_dispatch_alerts_multiple_alertable_entries_send_one_digest():
    """A tick that yields MULTIPLE alertable surprises (e.g. the first slow
    refresh after a restart catching several already-released prints at
    once) must send exactly ONE combined message, not a burst."""
    ctx = _FakeContext()
    entries = [
        {"event": "CPI MoM", "importance": 3, "actual": 0.4, "consensus": 0.3, "unit": "%",
         "surprise_z": None, "market_read": "hawkish", "confirmed_by": ["nasdaq"], "source": "nasdaq"},
        {"event": "Initial Jobless Claims", "importance": 2, "actual": 197.0, "consensus": 201.0, "unit": "K",
         "surprise_z": None, "market_read": "growth_up", "confirmed_by": ["nasdaq"], "source": "nasdaq"},
        {"event": "Current Account", "importance": 1, "actual": -246.0, "consensus": -258.0, "unit": "B",
         "surprise_z": None, "market_read": None, "confirmed_by": ["nasdaq"], "source": "nasdaq"},  # not alertable
    ]
    _run(es._dispatch_alerts(ctx, entries))
    assert len(ctx.bot.sent) == 1  # ONE message total, not 2 or 3
    text = ctx.bot.sent[0][1]
    assert "CPI MoM" in text
    assert "Initial Jobless Claims" in text
    assert "Current Account" not in text  # the low-importance one is excluded from the digest too
    assert text.startswith("2 economic surprises:")


def test_dispatch_alerts_zero_alertable_entries_sends_nothing():
    ctx = _FakeContext()
    _run(es._dispatch_alerts(ctx, [
        {"event": "Current Account", "importance": 1, "actual": -246.0, "consensus": -258.0, "unit": "B",
         "surprise_z": None, "market_read": None, "confirmed_by": ["nasdaq"], "source": "nasdaq"},
    ]))
    assert ctx.bot.sent == []


def test_dispatch_alerts_empty_new_entries_sends_nothing():
    ctx = _FakeContext()
    _run(es._dispatch_alerts(ctx, []))
    assert ctx.bot.sent == []


def test_refresh_fixture_shaped_multi_entry_slow_refresh_sends_one_digest(monkeypatch):
    """End-to-end shape close to the real 2026-09-25 fixture day: several
    surprises land in one slow-refresh tick, mixed importance -- exactly one
    Telegram digest, containing only the importance>=2 ones."""
    now = datetime.now(es._NY)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": [
            {"event": "Building Permits", "release_time_et": now.isoformat(),
             "actual": -2.1, "actual_unit": "%", "consensus": -2.7, "consensus_unit": "%",
             "previous": 4.3, "previous_unit": "%", "importance": 2, "source": "nasdaq"},
            {"event": "Initial Jobless Claims", "release_time_et": now.isoformat(),
             "actual": 197.0, "actual_unit": "K", "consensus": 201.0, "consensus_unit": "K",
             "previous": 198.0, "previous_unit": "K", "importance": 2, "source": "nasdaq"},
            {"event": "Continuing Jobless Claims", "release_time_et": now.isoformat(),
             "actual": 1719.0, "actual_unit": "K", "consensus": 1750.0, "consensus_unit": "K",
             "previous": 1717.0, "previous_unit": "K", "importance": 1, "source": "nasdaq"},
            {"event": "Current Account", "release_time_et": now.isoformat(),
             "actual": -246.0, "actual_unit": "B", "consensus": -258.0, "consensus_unit": "B",
             "previous": -212.6, "previous_unit": "B", "importance": 1, "source": "nasdaq"},
            {"event": "Natural Gas Storage", "release_time_et": now.isoformat(),
             "actual": 53.0, "actual_unit": "B", "consensus": 50.0, "consensus_unit": "B",
             "previous": 44.0, "previous_unit": "B", "importance": 1, "source": "nasdaq"},
        ], "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    ctx = _FakeContext()
    result = _run(es.refresh_econ_surprise(ctx))
    assert result["new_surprises"] == 5  # ALL recorded

    saved = es._read_state_cached()
    assert len(saved["surprises"]) == 5  # ALL persisted regardless of importance

    assert len(ctx.bot.sent) == 1  # exactly one Telegram send
    text = ctx.bot.sent[0][1]
    assert "Building Permits" in text
    assert "Initial Jobless Claims" in text
    assert "Continuing Jobless Claims" not in text
    assert "Current Account" not in text
    assert "Natural Gas Storage" not in text


# ═══════════════════════════════════════════════════════════════════════════
#  _write_state_blocking: FD-leak-safe atomic write (2026-09-25 pass-2 fix,
#  item 6).
# ═══════════════════════════════════════════════════════════════════════════

def test_write_state_blocking_happy_path_still_works(monkeypatch, tmp_path):
    state_path = tmp_path / "econ_surprise.json"
    monkeypatch.setenv("CLIO_ECON_SURPRISE_FILE", str(state_path))
    ok = es._write_state_blocking(es._default_state())
    assert ok is True
    assert state_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []  # no leftover tmp file on success


def test_write_state_blocking_closes_fd_when_fdopen_raises(monkeypatch, tmp_path):
    """If os.fdopen() itself raises AFTER tempfile.mkstemp() has already
    handed out a raw fd, that fd must still be closed via os.close() in the
    finally block — never leaked — and the orphaned tmp file must be
    unlinked, not left behind."""
    state_path = tmp_path / "econ_surprise.json"
    monkeypatch.setenv("CLIO_ECON_SURPRISE_FILE", str(state_path))

    closed_fds = []
    real_close = os.close

    def _spy_close(fd):
        closed_fds.append(fd)
        return real_close(fd)
    monkeypatch.setattr(os, "close", _spy_close)

    def _raising_fdopen(_fd, *_a, **_kw):
        raise OSError("simulated fdopen failure")
    monkeypatch.setattr(os, "fdopen", _raising_fdopen)

    result = es._write_state_blocking(es._default_state())

    assert result is False  # write failed...
    assert len(closed_fds) == 1  # ...but the raw fd was closed, not leaked
    assert not state_path.exists()  # target file never touched
    assert list(tmp_path.glob("*.tmp")) == []  # orphaned tmp file cleaned up


def test_write_state_blocking_no_double_close_on_success(monkeypatch, tmp_path):
    """On the success path, os.fdopen()'s own file-object close (via the
    `with` block) must be the ONLY close — the explicit `if fd >= 0:
    os.close(fd)` guard must not fire a second time (fd was set to -1)."""
    state_path = tmp_path / "econ_surprise.json"
    monkeypatch.setenv("CLIO_ECON_SURPRISE_FILE", str(state_path))

    explicit_close_calls = []
    real_close = os.close

    def _spy_close(fd):
        explicit_close_calls.append(fd)
        return real_close(fd)
    monkeypatch.setattr(os, "close", _spy_close)

    ok = es._write_state_blocking(es._default_state())
    assert ok is True
    # os.fdopen's own `with`-block close doesn't go through our os.close spy
    # (file objects close their fd via a different C-level path) -- what we
    # ARE asserting is that _write_state_blocking's own explicit
    # `if fd >= 0: os.close(fd)` guard never fires on the success path.
    assert explicit_close_calls == []


# ═══════════════════════════════════════════════════════════════════════════
#  _prune_state / _release_time_from_key: bounds alerted/consensus_table/
#  bls_checked growth (2026-09-25 orchestrator fix)
# ═══════════════════════════════════════════════════════════════════════════

def test_release_time_from_key_parses_and_rejects():
    now = datetime.now(es._NY)
    key = f"CPI MoM|{now.isoformat()}"
    dt = es._release_time_from_key(key)
    assert dt is not None
    assert dt.isoformat() == now.isoformat()

    assert es._release_time_from_key("no-pipe-in-here") is None
    assert es._release_time_from_key("CPI MoM|not-a-timestamp") is None
    assert es._release_time_from_key(None) is None
    assert es._release_time_from_key(123) is None


def test_prune_state_drops_old_alerted_and_consensus_table_entries():
    now = datetime.now(es._NY)
    old = now - timedelta(days=15)
    recent = now - timedelta(days=1)
    state = es._default_state()
    state["alerted"] = {
        f"CPI MoM|{old.isoformat()}": True,
        f"NFP|{recent.isoformat()}": True,
        "unparsable-key": True,
    }
    state["consensus_table"] = {
        f"CPI MoM|{old.isoformat()}": {"event": "CPI MoM", "consensus": 0.3, "previous": 0.2, "importance": 3},
        f"NFP|{recent.isoformat()}": {"event": "NFP", "consensus": 150.0, "previous": 120.0, "importance": 3},
    }

    es._prune_state(state, now)

    assert set(state["alerted"].keys()) == {f"NFP|{recent.isoformat()}"}
    assert set(state["consensus_table"].keys()) == {f"NFP|{recent.isoformat()}"}


def test_prune_state_keeps_entries_inside_ttl():
    now = datetime.now(es._NY)
    just_inside = now - timedelta(days=13)
    state = es._default_state()
    key = f"CPI MoM|{just_inside.isoformat()}"
    state["alerted"] = {key: True}
    state["consensus_table"] = {key: {"event": "CPI MoM", "consensus": 0.3, "previous": 0.2, "importance": 3}}

    es._prune_state(state, now)

    assert key in state["alerted"]
    assert key in state["consensus_table"]


def test_prune_state_drops_stale_bls_checked_dates():
    now = datetime.now(es._NY)
    old_date = (now - timedelta(days=20)).date().isoformat()
    recent_date = (now - timedelta(days=1)).date().isoformat()
    state = es._default_state()
    state["bls_checked"] = {"CPI MoM": old_date, "NFP": recent_date, "PPI MoM": "not-a-date"}

    es._prune_state(state, now)

    assert set(state["bls_checked"].keys()) == {"NFP"}


def test_prune_state_never_raises_on_malformed_input():
    state = {"alerted": "not-a-dict", "consensus_table": None, "bls_checked": 123}
    es._prune_state(state, datetime.now(es._NY))  # must not raise


def test_prune_state_runs_as_part_of_a_real_write_cycle(monkeypatch):
    """Integration: a slow refresh that finds a fresh event must leave old
    (>14d) alerted/consensus_table keys behind after the write. Frozen to a
    weekday-active reference time so the 31min elapsed is guaranteed to
    clear the (variable, item 5) slow-refresh threshold regardless of when
    the suite actually runs."""
    now = _freeze_time(monkeypatch, _WEEKDAY_ACTIVE_NOW)
    old = now - timedelta(days=30)
    _seed_state(
        last_slow_refresh_at=(now - timedelta(minutes=31)).isoformat(),
        alerted={f"Old Series|{old.isoformat()}": True},
        consensus_table={f"Old Series|{old.isoformat()}": {"event": "Old Series", "consensus": 1.0,
                                                             "previous": 1.0, "importance": 1}},
    )
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": [{
            "event": "CPI MoM", "release_time_et": now.isoformat(),
            "actual": 0.4, "actual_unit": "%", "consensus": 0.3, "consensus_unit": "%",
            "previous": 0.2, "previous_unit": "%", "importance": 3, "source": "nasdaq",
        }], "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    _run(es.refresh_econ_surprise(None))

    saved = es._read_state_cached()
    assert not any(k.startswith("Old Series|") for k in saved["alerted"])
    assert not any(k.startswith("Old Series|") for k in saved["consensus_table"])
    assert any(k.startswith("CPI MoM|") for k in saved["consensus_table"])


# ═══════════════════════════════════════════════════════════════════════════
#  Late detection: entries first recorded >15min after release_time_et
# ═══════════════════════════════════════════════════════════════════════════

def test_refresh_marks_late_when_first_seen_well_after_release(monkeypatch):
    """Simulates the restart scenario: the first tick after a restart catches
    an already-released print (e.g. 08:30 CPI first seen at 10:00)."""
    now = datetime.now(es._NY)
    released_long_ago = now - timedelta(hours=1, minutes=30)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])  # no active window -> slow path

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": [{
            "event": "CPI MoM", "release_time_et": released_long_ago.isoformat(),
            "actual": 0.4, "actual_unit": "%", "consensus": 0.3, "consensus_unit": "%",
            "previous": 0.2, "previous_unit": "%", "importance": 3, "source": "nasdaq",
        }], "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    ctx = _FakeContext()
    result = _run(es.refresh_econ_surprise(ctx))
    assert result["new_surprises"] == 1

    saved = es._read_state_cached()
    entry = saved["surprises"][0]
    assert entry["late"] is True
    assert len(ctx.bot.sent) == 1
    assert "detected late" in ctx.bot.sent[0][1]
    assert released_long_ago.strftime("%H:%M") in ctx.bot.sent[0][1]


def test_refresh_does_not_mark_late_for_timely_detection(monkeypatch):
    now = datetime.now(es._NY)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": [{
            "event": "PPI MoM", "release_time_et": now.isoformat(),
            "actual": 0.5, "actual_unit": "%", "consensus": 0.4, "consensus_unit": "%",
            "previous": 0.3, "previous_unit": "%", "importance": 3, "source": "nasdaq",
        }], "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    ctx = _FakeContext()
    _run(es.refresh_econ_surprise(ctx))

    saved = es._read_state_cached()
    entry = saved["surprises"][0]
    assert entry["late"] is False
    assert "detected late" not in ctx.bot.sent[0][1]


def test_send_alert_late_text_falls_back_gracefully_on_bad_release_time():
    entry = {
        "event": "CPI MoM", "release_time_et": "not-a-timestamp", "actual": 0.4,
        "consensus": 0.3, "previous": 0.2, "unit": "%", "surprise_raw": 0.1,
        "surprise_z": None, "direction": "above", "market_read": "hawkish",
        "importance": 3, "source": "nasdaq", "confirmed_by": ["nasdaq"],
        "as_of": datetime.now(es._NY).isoformat(), "late": True,
    }
    ctx = _FakeContext()
    _run(es._send_alert(ctx, entry))
    assert len(ctx.bot.sent) == 1
    assert "detected late" in ctx.bot.sent[0][1]  # degrades to the no-timestamp variant, never raises


# ═══════════════════════════════════════════════════════════════════════════
#  bootstrap_history: never auto-invoked from refresh
# ═══════════════════════════════════════════════════════════════════════════

def test_bootstrap_history_never_called_by_refresh(monkeypatch):
    called = {"n": 0}
    async def _spy(*_a, **_kw):
        called["n"] += 1
        return {"seeded": {}, "source": "tradingview"}
    monkeypatch.setattr(es, "bootstrap_history", _spy)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    _run(es.refresh_econ_surprise(None))
    assert called["n"] == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Optional BLS confirm: off by default, gated by env, slow-path only
# ═══════════════════════════════════════════════════════════════════════════

def test_bls_confirm_off_by_default(monkeypatch):
    monkeypatch.delenv("CLIO_ECON_SURPRISE_BLS_CONFIRM", raising=False)
    called = {"n": 0}
    async def _spy(_state, _now):
        called["n"] += 1
    monkeypatch.setattr(es, "_bls_confirm_recent", _spy)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    _run(es.refresh_econ_surprise(None))
    assert called["n"] == 0


def test_bls_confirm_runs_when_env_enabled(monkeypatch):
    monkeypatch.setenv("CLIO_ECON_SURPRISE_BLS_CONFIRM", "1")
    called = {"n": 0}
    async def _spy(_state, _now):
        called["n"] += 1
    monkeypatch.setattr(es, "_bls_confirm_recent", _spy)
    monkeypatch.setattr(ec, "get_todays_events", lambda: [])

    async def _fake_fetch_all(start, end, state):
        return {"nasdaq": None, "tradingview": None}
    monkeypatch.setattr(es, "_fetch_all_sources", _fake_fetch_all)

    _run(es.refresh_econ_surprise(None))
    assert called["n"] == 1


def test_bls_confirm_recent_appends_bls_when_value_matches(monkeypatch):
    now = datetime.now(es._NY)
    state = es._default_state()
    state["surprises"] = [{
        "event": "CPI MoM", "release_time_et": now.isoformat(), "actual": 334.98,
        "consensus": 334.5, "previous": 333.9, "unit": "level", "surprise_raw": 0.48,
        "surprise_z": None, "direction": "above", "market_read": "hawkish",
        "importance": 3, "source": "nasdaq", "confirmed_by": ["nasdaq"],
        "as_of": now.isoformat(),
    }]

    class _FakeBLSResp:
        status_code = 200
        def json(self):
            return {"Results": {"series": [{"data": [{"value": "334.980"}]}]}}

    class _FakeBLSClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_a):
            return False
        async def get(self, _url):
            return _FakeBLSResp()

    import httpx as real_httpx
    monkeypatch.setattr(real_httpx, "AsyncClient", lambda **_kw: _FakeBLSClient())

    _run(es._bls_confirm_recent(state, now))
    assert "bls" in state["surprises"][0]["confirmed_by"]
    assert state["bls_checked"]["CPI MoM"] == now.date().isoformat()


def test_bls_confirm_recent_skips_non_coverable_and_stale_entries(monkeypatch):
    now = datetime.now(es._NY)
    old = now - timedelta(days=10)
    state = es._default_state()
    state["surprises"] = [
        {"event": "Building Permits", "release_time_et": now.isoformat(), "actual": 1.4,
         "confirmed_by": ["nasdaq"], "as_of": now.isoformat()},          # not BLS-coverable
        {"event": "CPI MoM", "release_time_et": old.isoformat(), "actual": 300.0,
         "confirmed_by": ["nasdaq"], "as_of": old.isoformat()},          # too old (>3 days)
    ]

    import httpx as real_httpx

    class _Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_a):
            return False
        async def get(self, _url):
            raise AssertionError("BLS should never be called for out-of-scope entries")

    monkeypatch.setattr(real_httpx, "AsyncClient", lambda **_kw: _Client())
    _run(es._bls_confirm_recent(state, now))  # must not raise


# ═══════════════════════════════════════════════════════════════════════════
#  get_recent_surprises (sync, mtime-cached, never raises)
# ═══════════════════════════════════════════════════════════════════════════

def test_get_recent_surprises_missing_file_returns_empty():
    assert es.get_recent_surprises() == []


def test_get_recent_surprises_filters_stale_and_sorts():
    now = datetime.now(es._NY)
    fresh = {
        "event": "CPI MoM", "release_time_et": "2026-10-13T08:30:00-04:00",
        "actual": 0.4, "consensus": 0.3, "previous": 0.2, "unit": "%",
        "surprise_raw": 0.1, "surprise_z": None, "direction": "above",
        "market_read": "hawkish", "importance": 3, "source": "nasdaq",
        "confirmed_by": ["nasdaq"], "as_of": now.isoformat(),
    }
    stale = dict(fresh, event="Old Event", release_time_et="2026-09-01T08:30:00-04:00",
                 as_of=(now - timedelta(hours=6)).isoformat())
    state = es._default_state()
    state["surprises"] = [stale, fresh]
    es._write_state_blocking(state)

    recent = es.get_recent_surprises(max_age_minutes=240)
    events = [e["event"] for e in recent]
    assert "CPI MoM" in events
    assert "Old Event" not in events
    # contract keys present
    for k in ("event", "release_time_et", "actual", "consensus", "previous", "unit",
              "surprise_raw", "surprise_z", "direction", "market_read", "importance",
              "source", "confirmed_by", "as_of"):
        assert k in recent[0]


def test_get_recent_surprises_corrupt_file_returns_empty(_isolated_state_file):
    with open(_isolated_state_file, "w") as f:
        f.write("not json{{{")
    assert es.get_recent_surprises() == []


def test_get_recent_surprises_never_raises_on_malformed_entries():
    state = es._default_state()
    state["surprises"] = [
        {"event": "Bad", "as_of": "not-a-timestamp"},
        {"event": "AlsoBad"},  # missing as_of entirely
        "not-even-a-dict",
    ]
    es._write_state_blocking(state)
    assert es.get_recent_surprises() == []


# ═══════════════════════════════════════════════════════════════════════════
#  get_upcoming_releases (sync, mtime-cached, never raises)
# ═══════════════════════════════════════════════════════════════════════════

def test_get_upcoming_releases_merges_consensus_table(monkeypatch):
    now = datetime.now(es._NY)
    event_dt = now + timedelta(hours=2)
    monkeypatch.setattr(ec, "get_upcoming_events", lambda hours=24: [{
        "name": "CPI MoM", "datetime_et": event_dt.isoformat(), "importance": 3,
        "forecast": "", "previous": "",
    }])
    state = es._default_state()
    key = f"CPI MoM|{event_dt.isoformat()}"
    state["consensus_table"][key] = {"event": "CPI MoM", "consensus": 0.3, "previous": 0.2, "importance": 3}
    es._write_state_blocking(state)

    releases = es.get_upcoming_releases(hours=6.0)
    assert len(releases) == 1
    r = releases[0]
    assert r["event"] == "CPI MoM"
    assert r["consensus"] == 0.3
    assert r["previous"] == 0.2
    assert r["importance"] == 3
    assert 110 <= r["minutes_until"] <= 130


def test_get_upcoming_releases_falls_back_to_calendar_forecast(monkeypatch):
    now = datetime.now(es._NY)
    event_dt = now + timedelta(hours=1)
    monkeypatch.setattr(ec, "get_upcoming_events", lambda hours=24: [{
        "name": "Building Permits", "datetime_et": event_dt.isoformat(), "importance": 2,
        "forecast": "1.394M", "previous": "1.433M",
    }])
    releases = es.get_upcoming_releases(hours=6.0)
    assert len(releases) == 1
    assert releases[0]["consensus"] == 1.394
    assert releases[0]["previous"] == 1.433


def test_get_upcoming_releases_econ_calendar_failure_returns_empty(monkeypatch):
    def _boom(hours=24):
        raise RuntimeError("econ_calendar down")
    monkeypatch.setattr(ec, "get_upcoming_events", _boom)
    assert es.get_upcoming_releases(hours=6.0) == []


def test_get_upcoming_releases_disambiguated_lookup_prefers_percent(monkeypatch):
    """econ_calendar's event names carry no unit -- when BOTH a "%" and a
    "level" consensus_table entry exist for a fallback event, the "%"
    variant is preferred deterministically (item 1, 2026-09-25 pass-2 fix)."""
    now = datetime.now(es._NY)
    event_dt = now + timedelta(hours=2)
    monkeypatch.setattr(ec, "get_upcoming_events", lambda hours=24: [{
        "name": "Building Permits", "datetime_et": event_dt.isoformat(), "importance": 2,
        "forecast": "", "previous": "",
    }])
    state = es._default_state()
    pct_key = f"Building Permits (%)|{event_dt.isoformat()}"
    level_key = f"Building Permits (level)|{event_dt.isoformat()}"
    state["consensus_table"][pct_key] = {"event": "Building Permits", "consensus": -2.7,
                                          "previous": 4.3, "importance": 2}
    state["consensus_table"][level_key] = {"event": "Building Permits", "consensus": 1.394,
                                            "previous": 1.433, "importance": 2}
    es._write_state_blocking(state)

    releases = es.get_upcoming_releases(hours=6.0)
    assert len(releases) == 1
    assert releases[0]["consensus"] == -2.7  # the "%" variant wins


def test_get_upcoming_releases_disambiguated_lookup_falls_back_to_level(monkeypatch):
    """Only the "level" variant exists -> that's used (deterministic
    preference: "%" first, else "level")."""
    now = datetime.now(es._NY)
    event_dt = now + timedelta(hours=2)
    monkeypatch.setattr(ec, "get_upcoming_events", lambda hours=24: [{
        "name": "New Home Sales", "datetime_et": event_dt.isoformat(), "importance": 2,
        "forecast": "", "previous": "",
    }])
    state = es._default_state()
    level_key = f"New Home Sales (level)|{event_dt.isoformat()}"
    state["consensus_table"][level_key] = {"event": "New Home Sales", "consensus": 615.0,
                                            "previous": 643.0, "importance": 2}
    es._write_state_blocking(state)

    releases = es.get_upcoming_releases(hours=6.0)
    assert len(releases) == 1
    assert releases[0]["consensus"] == 615.0


def test_get_upcoming_releases_curated_event_lookup_unaffected_by_disambiguation(monkeypatch):
    """A curated/mapped event's consensus_table key is never suffixed, so
    the plain-name lookup path (unchanged from before item 1) still works."""
    now = datetime.now(es._NY)
    event_dt = now + timedelta(hours=2)
    monkeypatch.setattr(ec, "get_upcoming_events", lambda hours=24: [{
        "name": "CPI (Consumer Price Index)", "datetime_et": event_dt.isoformat(), "importance": 3,
        "forecast": "", "previous": "",
    }])
    state = es._default_state()
    key = f"CPI MoM|{event_dt.isoformat()}"
    state["consensus_table"][key] = {"event": "CPI MoM", "consensus": 0.3, "previous": 0.2, "importance": 3}
    es._write_state_blocking(state)

    releases = es.get_upcoming_releases(hours=6.0)
    assert len(releases) == 1
    assert releases[0]["consensus"] == 0.3


def test_get_upcoming_releases_sorted_chronologically(monkeypatch):
    now = datetime.now(es._NY)
    later = now + timedelta(hours=4)
    sooner = now + timedelta(hours=1)
    monkeypatch.setattr(ec, "get_upcoming_events", lambda hours=24: [
        {"name": "Later Event", "datetime_et": later.isoformat(), "importance": 2, "forecast": "", "previous": ""},
        {"name": "Sooner Event", "datetime_et": sooner.isoformat(), "importance": 2, "forecast": "", "previous": ""},
    ])
    releases = es.get_upcoming_releases(hours=6.0)
    assert [r["event"] for r in releases] == ["Sooner Event", "Later Event"]


# ═══════════════════════════════════════════════════════════════════════════
#  bootstrap_history
# ═══════════════════════════════════════════════════════════════════════════

def test_bootstrap_history_seeds_from_tradingview(monkeypatch):
    async def _fake_tv(_start, _end, _state):
        return [
            {"event": "CPI MoM", "release_time_et": "2026-01-13T08:30:00-05:00",
             "actual": 0.3, "actual_unit": "%", "consensus": 0.2, "consensus_unit": "%",
             "previous": 0.1, "previous_unit": "%", "importance": 3, "source": "tradingview"},
            {"event": "CPI MoM", "release_time_et": "2026-02-12T08:30:00-05:00",
             "actual": 0.1, "actual_unit": "%", "consensus": 0.2, "consensus_unit": "%",
             "previous": 0.3, "previous_unit": "%", "importance": 3, "source": "tradingview"},
            {"event": "PPI MoM", "release_time_et": "2026-01-14T08:30:00-05:00",
             "actual": 0.5, "actual_unit": "%", "consensus": 0.4, "consensus_unit": "%",
             "previous": 0.3, "previous_unit": "%", "importance": 3, "source": "tradingview"},
        ]
    monkeypatch.setattr(es, "_fetch_tradingview_raw", _fake_tv)

    result = _run(es.bootstrap_history())
    assert result["source"] == "tradingview"
    assert result["seeded"] == {"CPI MoM": 2, "PPI MoM": 1}

    saved = es._read_state_cached()
    assert saved["history"]["CPI MoM"] == [pytest.approx(0.1), pytest.approx(-0.1)]
    assert saved["history"]["PPI MoM"] == [pytest.approx(0.1)]


def test_bootstrap_history_scoped_to_single_key(monkeypatch):
    async def _fake_tv(_start, _end, _state):
        return [
            {"event": "CPI MoM", "release_time_et": "2026-01-13T08:30:00-05:00",
             "actual": 0.3, "actual_unit": "%", "consensus": 0.2, "consensus_unit": "%",
             "previous": 0.1, "previous_unit": "%", "importance": 3, "source": "tradingview"},
            {"event": "PPI MoM", "release_time_et": "2026-01-14T08:30:00-05:00",
             "actual": 0.5, "actual_unit": "%", "consensus": 0.4, "consensus_unit": "%",
             "previous": 0.3, "previous_unit": "%", "importance": 3, "source": "tradingview"},
        ]
    monkeypatch.setattr(es, "_fetch_tradingview_raw", _fake_tv)
    result = _run(es.bootstrap_history(canonical_key="CPI MoM"))
    assert result["seeded"] == {"CPI MoM": 1}


def test_bootstrap_history_respects_max_points_cap(monkeypatch):
    rows = [
        {"event": "CPI MoM", "release_time_et": f"2020-01-{i:02d}T08:30:00-05:00",
         "actual": float(i), "actual_unit": "level", "consensus": 0.0, "consensus_unit": "level",
         "previous": None, "previous_unit": None, "importance": 3, "source": "tradingview"}
        for i in range(1, 15)
    ]
    async def _fake_tv(_start, _end, _state):
        return rows
    monkeypatch.setattr(es, "_fetch_tradingview_raw", _fake_tv)
    result = _run(es.bootstrap_history(max_points=5))
    assert result["seeded"] == {"CPI MoM": 5}


def test_bootstrap_history_tradingview_unavailable_seeds_nothing(monkeypatch):
    async def _fake_tv(_start, _end, _state):
        return None
    monkeypatch.setattr(es, "_fetch_tradingview_raw", _fake_tv)
    result = _run(es.bootstrap_history())
    assert result == {"seeded": {}, "source": "tradingview"}
