"""Tests for agent/dpof/risk.py (pure).

Operator directive (2026-09-24): no mechanical breakers, no loss-streak
cooldown, no re-entry cooldown, and (no-hard-caps amendment) no
max_concurrent_positions/max_entries_per_day/max_entries_per_symbol_day
count caps either. What's left: P&L accounting (roll_state, record_entry,
record_close), the entry gate (operator halt/HALT file + entry window
only), and operator halt/resume.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from agent.dpof.models import RiskState
from agent.dpof.risk import (
    check_entry_allowed,
    et_day,
    et_week,
    operator_halt,
    operator_resume,
    parse_hhmm,
    record_close,
    record_entry,
    roll_state,
)

ET = ZoneInfo("America/New_York")


def dt(y, m, d, hh=10, mm=0, ss=0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=ET)


# ─────────────────────────────────────────────────────────────────────────
# et_day / et_week / parse_hhmm
# ─────────────────────────────────────────────────────────────────────────


class TestDateHelpers:
    def test_et_day_format(self):
        assert et_day(dt(2026, 9, 24, 14, 30)) == "2026-09-24"

    def test_et_week_format(self):
        # 2026-09-24 is a Thursday in ISO week 39.
        assert et_week(dt(2026, 9, 24)) == "2026-W39"

    def test_et_week_monday_boundary(self):
        # Monday 2026-09-21 starts week 39; Sunday 2026-09-20 is still week 38.
        assert et_week(dt(2026, 9, 21)) == "2026-W39"
        assert et_week(dt(2026, 9, 20)) == "2026-W38"

    def test_parse_hhmm(self):
        assert parse_hhmm("09:45") == time(9, 45)
        assert parse_hhmm("15:15") == time(15, 15)

    def test_et_day_across_dst_spring_forward(self):
        # 2026-03-08 is the US spring-forward date.
        before = dt(2026, 3, 8, 1, 30)
        after = dt(2026, 3, 8, 3, 30)
        assert et_day(before) == "2026-03-08"
        assert et_day(after) == "2026-03-08"


# ─────────────────────────────────────────────────────────────────────────
# roll_state
# ─────────────────────────────────────────────────────────────────────────


class TestRollState:
    def test_cold_start_initializes_day_and_week(self):
        state = RiskState()
        now = dt(2026, 9, 24, 9, 31)
        out = roll_state(state, now, 20_000.0)
        assert out.day == "2026-09-24"
        assert out.week == "2026-W39"
        assert out.start_of_day_equity == 20_000.0
        assert out.start_of_week_equity == 20_000.0
        assert out.hwm_equity == 20_000.0

    def test_same_day_no_reset(self):
        state = RiskState(day="2026-09-24", week="2026-W39", realized_pnl_today=50.0, entries_today=2)
        out = roll_state(state, dt(2026, 9, 24, 14, 0), 20_500.0)
        assert out.realized_pnl_today == 50.0
        assert out.entries_today == 2

    def test_new_day_resets_daily_counters_not_weekly(self):
        state = RiskState(
            day="2026-09-23", week="2026-W39", realized_pnl_today=100.0, entries_today=3,
            entries_by_symbol={"SPY": 3}, realized_pnl_week=200.0,
        )
        out = roll_state(state, dt(2026, 9, 24, 9, 31), 21_000.0)
        assert out.day == "2026-09-24"
        assert out.realized_pnl_today == 0.0
        assert out.entries_today == 0
        assert out.entries_by_symbol == {}
        assert out.realized_pnl_week == 200.0  # unchanged -- same week

    def test_new_week_resets_weekly_not_daily(self):
        state = RiskState(day="2026-09-24", week="2026-W38", realized_pnl_today=50.0, realized_pnl_week=200.0)
        out = roll_state(state, dt(2026, 9, 24, 9, 31), 20_000.0)
        assert out.week == "2026-W39"
        assert out.realized_pnl_week == 0.0
        assert out.realized_pnl_today == 50.0  # same day (day field unchanged) -- untouched

    def test_hwm_tracks_max_never_decreases(self):
        state = RiskState(day="2026-09-24", week="2026-W39", hwm_equity=25_000.0)
        out = roll_state(state, dt(2026, 9, 24, 14, 0), 20_000.0)  # equity dropped
        assert out.hwm_equity == 25_000.0  # HWM unchanged
        out2 = roll_state(state, dt(2026, 9, 24, 14, 0), 30_000.0)  # new high
        assert out2.hwm_equity == 30_000.0

    def test_none_equity_does_not_touch_hwm(self):
        state = RiskState(day="2026-09-24", week="2026-W39", hwm_equity=25_000.0)
        out = roll_state(state, dt(2026, 9, 24, 14, 0), None)
        assert out.hwm_equity == 25_000.0

    def test_does_not_mutate_input(self):
        state = RiskState(day="2026-09-23", week="2026-W38")
        before = copy.deepcopy(state)
        roll_state(state, dt(2026, 9, 24, 9, 31), 20_000.0)
        assert state == before

    def test_day_rollover_resets_stale_data_alarm_state(self):
        st = RiskState(day="2026-09-23", week="2026-W39", last_valid_snapshot=1.0, stale_alarm_sent=True)
        out = roll_state(st, dt(2026, 9, 24, 9, 31), 20_000.0)
        assert out.last_valid_snapshot is None and out.stale_alarm_sent is False
        same_day = roll_state(replace(out, last_valid_snapshot=5.0), dt(2026, 9, 24, 10, 0), 20_000.0)
        assert same_day.last_valid_snapshot == 5.0


# ─────────────────────────────────────────────────────────────────────────
# check_entry_allowed
# ─────────────────────────────────────────────────────────────────────────


class TestCheckEntryAllowed:
    def test_all_clear_allowed(self, cfg):
        state = RiskState()
        result = check_entry_allowed(state, dt(2026, 9, 24, 10, 0), "SPY", cfg=cfg)
        assert result.allowed is True
        assert result.reasons == []

    def test_hard_halt_reason_included(self, cfg):
        state = RiskState(hard_halted=True, halt_reason="manual")
        result = check_entry_allowed(state, dt(2026, 9, 24, 10, 0), "SPY", cfg=cfg)
        assert result.allowed is False
        assert "hard_halt:manual" in result.reasons

    def test_hard_halt_without_reason_string(self, cfg):
        state = RiskState(hard_halted=True, halt_reason="")
        result = check_entry_allowed(state, dt(2026, 9, 24, 10, 0), "SPY", cfg=cfg)
        assert "hard_halt:unknown" in result.reasons

    def test_halt_file_present(self, cfg):
        state = RiskState()
        result = check_entry_allowed(state, dt(2026, 9, 24, 10, 0), "SPY", cfg=cfg, halt_file_present=True)
        assert "halt_file" in result.reasons

    def test_entry_window_start_inclusive(self, cfg):
        cfg2 = replace(cfg, entry_start_et="09:45", entry_end_et="15:15")
        state = RiskState()
        result = check_entry_allowed(state, dt(2026, 9, 24, 9, 45, 0), "SPY", cfg=cfg2)
        assert "outside_entry_window" not in result.reasons

    def test_entry_window_one_minute_before_start_blocked(self, cfg):
        cfg2 = replace(cfg, entry_start_et="09:45", entry_end_et="15:15")
        state = RiskState()
        result = check_entry_allowed(state, dt(2026, 9, 24, 9, 44, 59), "SPY", cfg=cfg2)
        assert "outside_entry_window" in result.reasons

    def test_entry_window_end_inclusive(self, cfg):
        cfg2 = replace(cfg, entry_start_et="09:45", entry_end_et="15:15")
        state = RiskState()
        result = check_entry_allowed(state, dt(2026, 9, 24, 15, 15, 0), "SPY", cfg=cfg2)
        assert "outside_entry_window" not in result.reasons

    def test_entry_window_one_second_after_end_blocked(self, cfg):
        cfg2 = replace(cfg, entry_start_et="09:45", entry_end_et="15:15")
        state = RiskState()
        result = check_entry_allowed(state, dt(2026, 9, 24, 15, 15, 1), "SPY", cfg=cfg2)
        assert "outside_entry_window" in result.reasons

    def test_no_entry_count_caps_left(self, cfg):
        """Operator directive (no-hard-caps amendment): a huge entries_today/
        entries_by_symbol count no longer blocks anything -- only the
        operator halt/HALT-file and the entry window do."""
        state = RiskState(entries_today=10_000, entries_by_symbol={"SPY": 10_000})
        result = check_entry_allowed(state, dt(2026, 9, 24, 10, 0), "SPY", cfg=cfg)
        assert result.allowed is True
        assert result.reasons == []

    def test_multiple_failing_reasons_all_collected(self, cfg):
        cfg2 = replace(cfg, entry_start_et="09:45", entry_end_et="15:15")
        state = RiskState(hard_halted=True, halt_reason="manual")
        result = check_entry_allowed(state, dt(2026, 9, 24, 16, 0), "SPY", cfg=cfg2, halt_file_present=True)
        assert result.allowed is False
        assert "hard_halt:manual" in result.reasons
        assert "halt_file" in result.reasons
        assert "outside_entry_window" in result.reasons


# ─────────────────────────────────────────────────────────────────────────
# record_entry / record_close
# ─────────────────────────────────────────────────────────────────────────


class TestRecordEntry:
    def test_increments_counts(self):
        state = RiskState(entries_today=1, entries_by_symbol={"SPY": 1})
        new_state = record_entry(state, "SPY", now_epoch=1_700_000_000.0)
        assert new_state.entries_today == 2
        assert new_state.entries_by_symbol == {"SPY": 2}

    def test_new_symbol_starts_at_one(self):
        state = RiskState(entries_today=1, entries_by_symbol={"SPY": 1})
        new_state = record_entry(state, "QQQ", now_epoch=1_700_000_000.0)
        assert new_state.entries_by_symbol == {"SPY": 1, "QQQ": 1}

    def test_does_not_mutate_input(self):
        state = RiskState(entries_today=1, entries_by_symbol={"SPY": 1})
        before = copy.deepcopy(state)
        record_entry(state, "SPY", now_epoch=1_700_000_000.0)
        assert state == before


class TestRecordClose:
    def test_pnl_accumulates_today_week_and_total(self):
        state = RiskState(realized_pnl_today=10.0, realized_pnl_week=50.0, realized_pnl_total=100.0)
        new_state = record_close(state, "SPY", realized_pnl=25.0, now_epoch=1000.0)
        assert new_state.realized_pnl_today == 35.0
        assert new_state.realized_pnl_week == 75.0
        assert new_state.realized_pnl_total == 125.0

    def test_negative_pnl_also_accumulates(self):
        state = RiskState(realized_pnl_today=10.0, realized_pnl_week=50.0, realized_pnl_total=100.0)
        new_state = record_close(state, "SPY", realized_pnl=-40.0, now_epoch=1000.0)
        assert new_state.realized_pnl_today == -30.0
        assert new_state.realized_pnl_week == 10.0
        assert new_state.realized_pnl_total == 60.0

    def test_realized_pnl_total_never_reset_by_rollover(self):
        """Unlike _today/_week, realized_pnl_total is cumulative since
        inception (or the last operator resume re-baseline) -- day/week
        rollover (roll_state) must never touch it."""
        state = record_close(RiskState(realized_pnl_total=100.0), "SPY", realized_pnl=25.0, now_epoch=1000.0)
        assert state.realized_pnl_total == 125.0
        state = record_close(state, "QQQ", realized_pnl=-40.0, now_epoch=2000.0)
        assert state.realized_pnl_total == 85.0
        rolled = roll_state(state, dt(2099, 1, 1, 9, 31), 20_000.0)
        assert rolled.realized_pnl_total == 85.0

    def test_no_streak_or_cooldown_bookkeeping_any_more(self):
        """Operator directive: no loss-streak counter, no cooldown, no
        per-symbol last-stop bookkeeping -- record_close is P&L accounting
        only now."""
        state = RiskState(consecutive_losses=2, cooldown_until=None, last_stop_by_symbol={})
        new_state = record_close(state, "SPY", realized_pnl=-10.0, now_epoch=1000.0)
        assert new_state.consecutive_losses == 2  # untouched
        assert new_state.cooldown_until is None    # untouched
        assert new_state.last_stop_by_symbol == {}  # untouched

    def test_does_not_mutate_input(self):
        state = RiskState(realized_pnl_today=5.0)
        before = copy.deepcopy(state)
        record_close(state, "SPY", realized_pnl=-10.0, now_epoch=1000.0)
        assert state == before


# ─────────────────────────────────────────────────────────────────────────
# operator_halt / operator_resume
# ─────────────────────────────────────────────────────────────────────────


class TestOperatorHaltResume:
    def test_operator_halt_sets_fields(self):
        state = RiskState()
        new_state = operator_halt(state, "manual_stop")
        assert new_state.hard_halted is True
        assert new_state.halt_reason == "manual_stop"

    def test_operator_resume_clears_fields(self):
        state = RiskState(
            hard_halted=True,
            daily_halted=True,
            cooldown_until=12345.0,
            halt_reason="manual",
            consecutive_losses=3,
            entries_today=2,
        )
        new_state = operator_resume(state)
        assert new_state.hard_halted is False
        assert new_state.daily_halted is False
        assert new_state.cooldown_until is None
        assert new_state.halt_reason == ""
        # Untouched fields survive.
        assert new_state.consecutive_losses == 3
        assert new_state.entries_today == 2

    def test_operator_resume_with_equity_resets_realized_pnl_total(self):
        """A re-baseline is an explicit operator decision that DPOF's virtual
        capital starts fresh -- realized_pnl_total resets to 0.0 exactly like
        realized_pnl_today/_week do."""
        state = RiskState(realized_pnl_total=250.0, realized_pnl_today=10.0, realized_pnl_week=30.0)
        new_state = operator_resume(state, equity=20_000.0)
        assert new_state.realized_pnl_total == 0.0
        assert new_state.realized_pnl_today == 0.0
        assert new_state.realized_pnl_week == 0.0

    def test_operator_resume_without_equity_keeps_realized_pnl_total(self):
        state = RiskState(realized_pnl_total=250.0)
        new_state = operator_resume(state, equity=None)
        assert new_state.realized_pnl_total == 250.0

    def test_operator_resume_without_equity_keeps_baselines(self):
        st = RiskState(hard_halted=True, halt_reason="x", hwm_equity=20_000.0, start_of_week_equity=19_000.0)
        out = operator_resume(st)
        assert not out.hard_halted and out.hwm_equity == 20_000.0 and out.start_of_week_equity == 19_000.0

    def test_operator_resume_does_not_mutate_input(self):
        state = RiskState(hard_halted=True, daily_halted=True, cooldown_until=1.0, halt_reason="x")
        before = copy.deepcopy(state)
        operator_resume(state)
        assert state == before

    def test_operator_halt_does_not_mutate_input(self):
        state = RiskState()
        before = copy.deepcopy(state)
        operator_halt(state, "manual_stop")
        assert state == before
