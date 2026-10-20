"""S215 / R-16: bond_intel weekday and hour gates are visible, and logged once per day.

run_bond_intel is a daily job that returns silently on Fri/Sat/Sun/Mon. These tests pin
the INFO skip line (once per ET day) and confirm Tuesday still proceeds past the gates.
The clock is frozen through a datetime subclass; nothing touches the network, the
bridge dir or the live intelligence state (the state, calendar and research calls are faked).
"""

import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from agent.integrations import bond_intel

NY = ZoneInfo("America/New_York")
SKIP_WEEKDAY = "bond_intel skipped: weekday gate (Tue-Thu only)"
SKIP_HOUR = "bond_intel skipped: outside 06:00-10:00 ET window"
PROCEED = "Starting scheduled bond auction intelligence..."


class _FrozenDT(datetime):
    frozen: datetime = datetime(2026, 10, 9, 8, 0, tzinfo=NY)

    @classmethod
    def now(cls, tz=None):
        return _FrozenDT.frozen.astimezone(tz) if tz else _FrozenDT.frozen.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def _gate_env(monkeypatch):
    monkeypatch.setattr(bond_intel, "datetime", _FrozenDT)
    monkeypatch.setattr(bond_intel.config, "news_intel_enabled", True, raising=False)
    bond_intel._GATE_SKIP_LOGGED.clear()
    refresh_calls = []

    async def fake_refresh():
        refresh_calls.append(1)
        return []

    monkeypatch.setattr(bond_intel, "_refresh_auction_calendar", fake_refresh)
    monkeypatch.setattr(bond_intel, "_load_state", lambda: {})
    monkeypatch.setattr(bond_intel, "_already_analyzed_today", lambda state: True)
    return refresh_calls


def _freeze(monkeypatch, y, mo, d, h):
    _FrozenDT.frozen = datetime(y, mo, d, h, 0, tzinfo=NY)


def _skip_lines(caplog, text):
    return [r for r in caplog.records if r.getMessage() == text]


@pytest.mark.parametrize(
    "day, label",
    [((2026, 10, 9), "friday"), ((2026, 10, 10), "saturday"), ((2026, 10, 11), "sunday"), ((2026, 10, 12), "monday")],
)
def test_weekday_gate_logs_skip_exactly_once_per_day(monkeypatch, caplog, _gate_env, day, label):
    caplog.set_level(logging.INFO, logger="openclaw.bond_intel")
    _freeze(monkeypatch, *day, 8)
    asyncio.run(bond_intel.run_bond_intel(None))
    asyncio.run(bond_intel.run_bond_intel(None))  # second callback the same day: no new line
    assert len(_skip_lines(caplog, SKIP_WEEKDAY)) == 1
    assert _skip_lines(caplog, SKIP_WEEKDAY)[0].levelno == logging.INFO
    assert not any(r.getMessage() == PROCEED for r in caplog.records)


def test_weekday_gate_skip_logs_again_on_the_next_day(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="openclaw.bond_intel")
    _freeze(monkeypatch, 2026, 10, 9, 8)
    asyncio.run(bond_intel.run_bond_intel(None))
    _freeze(monkeypatch, 2026, 10, 10, 8)
    asyncio.run(bond_intel.run_bond_intel(None))
    assert len(_skip_lines(caplog, SKIP_WEEKDAY)) == 2


def test_monday_still_refreshes_calendar_and_skips_analysis(monkeypatch, caplog, _gate_env):
    caplog.set_level(logging.INFO, logger="openclaw.bond_intel")
    _freeze(monkeypatch, 2026, 10, 12, 8)
    asyncio.run(bond_intel.run_bond_intel(None))
    assert _gate_env == [1]
    assert len(_skip_lines(caplog, SKIP_WEEKDAY)) == 1


def test_tuesday_proceeds_past_gates_without_skip_line(monkeypatch, caplog, _gate_env):
    caplog.set_level(logging.INFO, logger="openclaw.bond_intel")
    _freeze(monkeypatch, 2026, 10, 6, 8)  # Tuesday 08:00 ET
    asyncio.run(bond_intel.run_bond_intel(None))
    assert any(r.getMessage() == PROCEED for r in caplog.records)
    assert _skip_lines(caplog, SKIP_WEEKDAY) == []
    assert _skip_lines(caplog, SKIP_HOUR) == []
    assert _gate_env == []  # analysis path short-circuits on the dedup check, no calendar refresh


@pytest.mark.parametrize("hour", [5, 10])
def test_hour_gate_is_logged_once_on_a_tuesday(monkeypatch, caplog, hour):
    caplog.set_level(logging.INFO, logger="openclaw.bond_intel")
    _freeze(monkeypatch, 2026, 10, 6, hour)
    asyncio.run(bond_intel.run_bond_intel(None))
    asyncio.run(bond_intel.run_bond_intel(None))
    assert len(_skip_lines(caplog, SKIP_HOUR)) == 1
    assert not any(r.getMessage() == PROCEED for r in caplog.records)


def test_gate_memo_is_pruned_across_days(monkeypatch):
    _freeze(monkeypatch, 2026, 10, 9, 8)
    asyncio.run(bond_intel.run_bond_intel(None))
    assert len(bond_intel._GATE_SKIP_LOGGED) == 1
    _freeze(monkeypatch, 2026, 10, 10, 8)
    asyncio.run(bond_intel.run_bond_intel(None))
    assert {k[0] for k in bond_intel._GATE_SKIP_LOGGED} == {"2026-10-10"}
