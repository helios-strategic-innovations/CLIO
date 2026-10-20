"""R-31 (S215): a daily task that returns without advancing its state file is escalated.

Two consecutive misses on a day the task is EXPECTED to advance -> one ERROR log per escalated miss and
one plain-text Telegram alert per task per ET day. A success resets the counter. Weekdays the task is not
expected to run on never count. Nothing here touches live state: the daily-run marker, the state files,
the chat list and the Telegram bot are all faked; the clock is frozen.
"""

import asyncio
import logging
import os
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from agent.integrations import scheduled_tasks as st

NY = ZoneInfo("America/New_York")
# 2026-10-10 is a Saturday; 10-12 Monday, 10-13 Tuesday ... 10-17 Saturday.
MON, TUE, WED, THU, FRI, SAT, SUN = (
    datetime(2026, 10, d, 9, 0, tzinfo=NY) for d in (12, 13, 14, 15, 16, 17, 18)
)


class FakeBot:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_message(self, chat_id, **kwargs):
        self.sent.append({"chat_id": chat_id, **kwargs})


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "_DAILY_TASK_RAN_FILE", str(tmp_path / "daily_task_runs.json"))
    monkeypatch.setattr(st, "_DAILY_TASK_LOCK_FILE", str(tmp_path / "daily_task_runs.json.lock"))
    monkeypatch.setattr(st, "_get_chat_ids", lambda: ["111"])
    monkeypatch.setattr(st, "_daily_task_consecutive_misses", {})
    monkeypatch.setattr(st, "_daily_task_alert_day", {})
    clock = {"now": MON}
    monkeypatch.setattr(st, "_daily_task_et_now", lambda: clock["now"])
    return clock


@pytest.fixture
def clock(_isolated):
    return _isolated


@pytest.fixture
def state_file(tmp_path):
    return str(tmp_path / "state.json")


def _run(events, state_file, bot, task="macro_research"):
    """events: list of (et_datetime, advanced). Runs them in order inside one event loop."""
    results = []

    async def _drive():
        for i, (when, advanced) in enumerate(events):
            st._daily_task_et_now = lambda when=when: when  # frozen clock per event
            baseline = os.path.getmtime(state_file) if os.path.exists(state_file) else 0.0
            if advanced:
                stamp = 1_800_000_000.0 + 100.0 * (i + 1)
                with open(state_file, "a"):
                    pass
                os.utime(state_file, (stamp, stamp))
            results.append(st._mark_daily_task_if_state_progressed(task, state_file, baseline, bot=bot))
            for _ in range(3):
                await asyncio.sleep(0)

    asyncio.run(_drive())
    return results


# ------------------------------------------------------------------ R-31 behaviour

def test_expected_weekdays_are_derived_from_module_gates():
    assert st._DAILY_TASK_EXPECTED_WEEKDAYS["bond_intel"] == frozenset({1, 2, 3})
    for t in ("macro_research", "fed_analysis", "earnings_intel", "commodity_review", "analytics_fun_facts"):
        assert st._DAILY_TASK_EXPECTED_WEEKDAYS[t] == frozenset(range(5)), t
    assert st._DAILY_TASK_EXPECTED_WEEKDAYS["calendar_refresh"] == frozenset(range(7))


def test_single_miss_warns_but_does_not_escalate(state_file, caplog):
    bot = FakeBot()
    with caplog.at_level(logging.WARNING, logger=st.logger.name):
        assert _run([(MON, False)], state_file, bot) == [False]
    assert bot.sent == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("NOT marking as ran" in r.getMessage() for r in caplog.records)
    assert st._daily_task_consecutive_misses["macro_research"] == 1


def test_two_consecutive_misses_log_error_and_send_one_plain_text_alert(state_file, caplog):
    bot = FakeBot()
    with caplog.at_level(logging.ERROR, logger=st.logger.name):
        _run([(MON, False), (TUE, False)], state_file, bot)
    assert len(bot.sent) == 1
    msg = bot.sent[0]
    assert msg["chat_id"] == "111"
    assert "parse_mode" not in msg
    assert "macro_research" in msg["text"] and "state.json" in msg["text"]
    assert any("consecutive expected runs" in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)


def test_repeated_misses_same_day_send_only_one_alert(state_file):
    bot = FakeBot()
    _run([(MON, False), (MON, False), (MON, False)], state_file, bot)
    assert len(bot.sent) == 1
    assert st._daily_task_consecutive_misses["macro_research"] == 3


def test_stuck_task_alerts_again_on_the_next_et_day(state_file):
    bot = FakeBot()
    _run([(MON, False), (TUE, False), (WED, False)], state_file, bot)
    assert len(bot.sent) == 2


def test_success_resets_the_counter(state_file):
    bot = FakeBot()
    _run([(MON, False), (TUE, True), (WED, False)], state_file, bot)
    assert bot.sent == []
    assert st._daily_task_consecutive_misses["macro_research"] == 1


def test_weekend_misses_never_count_for_a_market_task(state_file):
    bot = FakeBot()
    _run([(SAT, False), (SUN, False), (MON, False)], state_file, bot)
    assert bot.sent == []
    assert st._daily_task_consecutive_misses["macro_research"] == 1


def test_bond_intel_ignores_monday_and_weekend_misses(state_file):
    bot = FakeBot()
    _run([(SAT, False), (SUN, False), (MON, False)], state_file, bot, task="bond_intel")
    assert bot.sent == []
    assert st._daily_task_consecutive_misses.get("bond_intel", 0) == 0


def test_bond_intel_tue_wed_misses_escalate(state_file):
    bot = FakeBot()
    _run([(TUE, False), (WED, False)], state_file, bot, task="bond_intel")
    assert len(bot.sent) == 1


def test_no_bot_means_error_logged_without_alert_and_no_crash(state_file, caplog):
    with caplog.at_level(logging.ERROR, logger=st.logger.name):
        _run([(MON, False), (TUE, False)], state_file, None)
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert st._daily_task_alert_day.get("macro_research") is None


def test_commodity_noop_counts_as_success(monkeypatch, tmp_path, clock):
    state = str(tmp_path / "commodity_state.json")
    monkeypatch.setitem(st._STATE_FILE_FOR_TASK, "commodity_review", state)
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot)

    async def _noop(_ctx):
        return {"reviewed": 0, "failed": 0, "skipped_recent": 3}

    async def _miss(_ctx):
        return {"reviewed": 0, "failed": 2, "skipped_recent": 0}

    async def _drive():
        clock["now"] = MON
        await st._run_commodity_review_guarded(ctx, _miss)
        clock["now"] = TUE
        await st._run_commodity_review_guarded(ctx, _noop)
        assert st._daily_task_consecutive_misses["commodity_review"] == 0
        clock["now"] = WED
        await st._run_commodity_review_guarded(ctx, _miss)
        for _ in range(3):
            await asyncio.sleep(0)

    asyncio.run(_drive())
    assert bot.sent == []


def test_success_path_writes_only_to_the_redirected_marker(state_file, tmp_path):
    # The fixture redirects the daily-run marker; a success must land in tmp_path, never the live file.
    bot = FakeBot()
    _run([(MON, True)], state_file, bot)
    assert st._DAILY_TASK_RAN_FILE.startswith(str(tmp_path))
    assert os.path.exists(st._DAILY_TASK_RAN_FILE)
