"""S215 / R-16: the staleness sentinel SLA table must match the real job schedules.

Every expected value below is derived from the job registration in
agent/integrations/scheduled_tasks.py and the module's own gate, not from the
production SLA_MAP. Frozen clocks are passed straight into _scan_one(now_utc=...);
the intelligence dir is redirected to tmp_path, so no live state is read or written.
"""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from agent.integrations import staleness_sentinel as ss

NY = ZoneInfo("America/New_York")

# Expected SLA in hours, derived from cadence + gate dead time + 12 h slack.
# (cadence / dead time / slack breakdown is in the comment on each row.)
EXPECTED_SLA_HOURS = {
    "news_intel_state.json": 14,              # 12 h overnight gap + 0.5 h interval + 1.5 h slack
    "fed_analysis_state.json": 36,            # 24 h daily + 12 h slack (+48 weekend allowance)
    "commodity_research_state.json": 36,      # 24 + 12 (+48 weekend allowance)
    "bond_intel_state.json": 108,             # Thu 08:00 -> Tue 08:00 = 96 h + 12 h slack
    "geopolitical_research_state.json": 336,  # event-driven (escalate() only): 14 days without any escalation
    "econ_calendar.json": 36,                 # 24 + 12, runs weekends too
    "macro_research_state.json": 36,          # 24 + 12 (+48 weekend allowance)
    "trade_policy_state.json": 186,           # 168 h weekly trigger + 12 h overnight + 6 h slack
    "earnings_calendar.json": 84,             # 72 h (3-day refresh rule) + 12 h (+48 weekend allowance)
    "auction_calendar.json": 180,             # 168 h weekly Monday refresh + 12 h
    "analytics_fun_facts.json": 36,           # 24 + 12 (+48 weekend allowance)
}

WEEKEND_FILES = {
    "fed_analysis_state.json",
    "commodity_research_state.json",
    "macro_research_state.json",
    "earnings_calendar.json",
    "analytics_fun_facts.json",
}


def _et(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=NY).astimezone(timezone.utc)


# Monday 2026-10-12 is "today" for most cases; Friday 2026-10-09 and Thursday 2026-10-08 are the prior writes.
MON_1157 = _et(2026, 10, 12, 11, 57)
MON_0800 = _et(2026, 10, 12, 8, 0)


@pytest.fixture
def intel_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "_INTEL_DIR", str(tmp_path))
    return tmp_path


def _write(intel_dir, filename, ts_utc):
    (intel_dir / filename).write_text(json.dumps({"last_updated": ts_utc.isoformat()}))


def _status(filename, intel_dir, last_write_utc, now_utc):
    _write(intel_dir, filename, last_write_utc)
    return ss._scan_one(filename, now_utc=now_utc)


# ---------------------------------------------------------------- table itself

def test_sla_table_matches_expected_cadence_exactly():
    assert set(ss.SLA_MAP) == set(EXPECTED_SLA_HOURS)
    actual = {k: v.total_seconds() / 3600 for k, v in ss.SLA_MAP.items()}
    assert actual == {k: float(v) for k, v in EXPECTED_SLA_HOURS.items()}


def test_weekend_allowance_set_is_exactly_the_weekday_only_jobs():
    assert set(ss.WEEKEND_ALLOWANCE_FILES) == WEEKEND_FILES
    assert ss.WEEKEND_ALLOWANCE == timedelta(hours=48)


def test_bond_and_auction_do_not_get_weekend_allowance():
    assert "bond_intel_state.json" not in ss.WEEKEND_ALLOWANCE_FILES
    assert "auction_calendar.json" not in ss.WEEKEND_ALLOWANCE_FILES
    assert "news_intel_state.json" not in ss.WEEKEND_ALLOWANCE_FILES
    assert "econ_calendar.json" not in ss.WEEKEND_ALLOWANCE_FILES


# ---------------------------------------------------------------- bond_intel (Tue-Thu only)

def test_bond_100h_old_on_monday_1157_is_not_stale(intel_dir):
    # Thu 2026-10-08 08:01 ET -> Mon 2026-10-12 11:57 ET = 99.9 h
    r = _status("bond_intel_state.json", intel_dir, _et(2026, 10, 8, 8, 1), MON_1157)
    assert r["status"] == "FRESH"
    assert r["sla_hours"] == 108.0


def test_bond_10_days_old_is_stale(intel_dir):
    r = _status("bond_intel_state.json", intel_dir, _et(2026, 10, 2, 8, 0), MON_1157)
    assert r["status"] == "STALE"
    assert r["sla_hours"] == 108.0


# ---------------------------------------------------------------- weekday jobs: Friday allowance

def test_fed_written_friday_is_fresh_on_monday_0800(intel_dir):
    # Fri 07:30 -> Mon 08:00 = 72.5 h; base 36 h + 48 h weekend allowance = 84 h
    r = _status("fed_analysis_state.json", intel_dir, _et(2026, 10, 9, 7, 30), MON_0800)
    assert r["status"] == "FRESH"
    assert r["sla_hours"] == 84.0


def test_fed_edge_just_inside_and_outside_84h(intel_dir):
    inside = _status("fed_analysis_state.json", intel_dir, _et(2026, 10, 9, 7, 30), _et(2026, 10, 12, 19, 29))
    outside = _status("fed_analysis_state.json", intel_dir, _et(2026, 10, 9, 7, 30), _et(2026, 10, 12, 19, 31))
    assert inside["status"] == "FRESH"   # 83.98 h
    assert outside["status"] == "STALE"  # 84.02 h


def test_fed_written_thursday_and_missed_friday_is_stale_on_monday(intel_dir):
    # Thu 07:30 -> Mon 08:00 = 96.5 h: the Friday run did not write, so the allowance must not hide it
    r = _status("fed_analysis_state.json", intel_dir, _et(2026, 10, 8, 7, 30), MON_0800)
    assert r["status"] == "STALE"


def test_analytics_friday_2005_is_fresh_on_monday_0800(intel_dir):
    # Fri 20:05 -> Mon 08:00 = 59.9 h; without the allowance this would be STALE at 36 h
    r = _status("analytics_fun_facts.json", intel_dir, _et(2026, 10, 9, 20, 5), MON_0800)
    assert r["status"] == "FRESH"


def test_allowance_not_applied_to_file_that_runs_weekends(intel_dir):
    # econ_calendar runs every day incl. weekends: Fri 06:00 -> Mon 08:00 (50 h) is STALE at 36 h
    r = _status("econ_calendar.json", intel_dir, _et(2026, 10, 9, 6, 0), MON_0800)
    assert r["status"] == "STALE"
    assert r["sla_hours"] == 36.0


def test_weekend_allowance_anchor_is_the_most_recent_saturday_midnight(intel_dir):
    # A weekday-only file last written on SATURDAY 15:33 ET (as fed_analysis_state.json was on 2026-10-10) is NOT before
    # the most recent Saturday 00:00 ET, so it gets no allowance: 44 h old on Monday 11:57 > 36 h => STALE. An
    # anchor shifted to Sunday would wrongly grant the 48 h allowance and call it FRESH.
    r = _status("fed_analysis_state.json", intel_dir, _et(2026, 10, 10, 15, 33), _et(2026, 10, 12, 11, 57))
    assert r["status"] == "STALE"
    # ... while a Friday write on the same Monday is covered by the allowance.
    ok = _status("fed_analysis_state.json", intel_dir, _et(2026, 10, 9, 7, 30), _et(2026, 10, 12, 11, 57))
    assert ok["status"] == "FRESH"


def test_news_intel_overnight_gap_is_covered_but_not_more(intel_dir):
    # last topic-monitor write Mon 19:30 ET, checked Tue 07:50 ET (12.3 h) -> fresh
    fresh = _status("news_intel_state.json", intel_dir, _et(2026, 10, 12, 19, 30), _et(2026, 10, 13, 7, 50))
    assert fresh["status"] == "FRESH"
    # a 20 h-old news file is stale even on a weekend-spanning window (no allowance for news)
    stale = _status("news_intel_state.json", intel_dir, _et(2026, 10, 9, 12, 0), MON_0800)
    assert stale["status"] == "STALE"


# ---------------------------------------------------------------- earnings 3-day refresh

def test_earnings_refreshed_wed_is_fresh_on_monday_0730(intel_dir):
    # Wed 07:15 -> Mon 07:30 = 120.25 h: worst case of the 3-day rule across a weekend
    r = _status("earnings_calendar.json", intel_dir, _et(2026, 10, 7, 7, 15), _et(2026, 10, 12, 7, 30))
    assert r["status"] == "FRESH"
    assert r["sla_hours"] == 132.0


def test_earnings_refreshed_monday_thursday_0730_is_fresh(intel_dir):
    # Mon 07:15 -> Thu 07:30 = 72.25 h, inside the 84 h base SLA with no weekend involved
    r = _status("earnings_calendar.json", intel_dir, _et(2026, 10, 5, 7, 15), _et(2026, 10, 8, 7, 30))
    assert r["status"] == "FRESH"
    assert r["sla_hours"] == 84.0


def test_earnings_refreshed_monday_not_refreshed_by_friday_is_stale(intel_dir):
    # Mon 07:15 -> Fri 07:30 = 96.25 h, no weekend in the window: the Friday refresh did not happen
    r = _status("earnings_calendar.json", intel_dir, _et(2026, 10, 5, 7, 15), _et(2026, 10, 9, 7, 30))
    assert r["status"] == "STALE"


# ---------------------------------------------------------------- weekly files

def test_auction_calendar_last_monday_is_fresh_on_monday_0800(intel_dir):
    # Mon 2026-10-05 08:00 -> Mon 2026-10-12 08:00 = 168 h, SLA 180 h
    r = _status("auction_calendar.json", intel_dir, _et(2026, 10, 5, 8, 0), MON_0800)
    assert r["status"] == "FRESH"
    assert r["sla_hours"] == 180.0


def test_trade_policy_just_over_seven_days_is_fresh_and_past_slack_is_stale(intel_dir):
    fresh = _status("trade_policy_state.json", intel_dir, _et(2026, 10, 5, 8, 0), _et(2026, 10, 12, 18, 0))
    assert fresh["status"] == "FRESH"    # 178 h
    stale = _status("trade_policy_state.json", intel_dir, _et(2026, 10, 5, 8, 0), _et(2026, 10, 13, 18, 0))
    assert stale["status"] == "STALE"    # 202 h


# ---------------------------------------------------------------- unscheduled file is untouched

def test_geopolitical_is_event_driven_with_a_14_day_sla():
    assert ss.SLA_MAP["geopolitical_research_state.json"] == timedelta(hours=336)
    assert "geopolitical_research_state.json" not in ss.WEEKEND_ALLOWANCE_FILES
