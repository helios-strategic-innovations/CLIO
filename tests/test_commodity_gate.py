"""CM-1 (S211): a commodity review with nothing due is a no-op, not a silent failure.

run_weekly_commodity_review skips every commodity analysed within its 6-day window and returns without
touching commodity_research_state.json; the S185/S205 state-advance gate warned about it every day
("returned but ... did not advance"). The review now returns {"reviewed", "skipped_recent", "failed"} and the
guarded runner treats only "nothing due, nothing failed" as a legitimate no-op. Real failures still warn.

Nothing here touches live state: the daily-task marker file, the commodity state file and the network
are all faked.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from agent.integrations import commodity_research as cr
from agent.integrations import scheduled_tasks as st

NY = ZoneInfo("America/New_York")
N = len(cr.COMMODITY_RESEARCH_MAP)


# ---------------------------------------------------------------- classifier
@pytest.mark.parametrize(
    "result, expected",
    [
        ({"reviewed": 0, "skipped_recent": 4, "failed": 0}, True),             # everything recently analysed
        ({"reviewed": 1, "skipped_recent": 3, "failed": 0}, False),            # real work happened
        ({"reviewed": 0, "skipped_recent": 3, "failed": 1}, False),            # a commodity failed
        ({"reviewed": 0, "skipped_recent": 4, "failed": 0, "error": True}, False),
        ({"reviewed": 0, "skipped_recent": 0, "failed": 0}, False),            # nothing skipped, nothing done: unknown
        ({"reviewed": 0, "failed": 0}, False),                                 # no skipped_recent key
        (None, False),                                                         # time guard / unknown -> gate as before
        ("ok", False),
        ([], False),
    ],
)
def test_noop_classifier_truth_table(result, expected):
    assert st._commodity_review_was_noop(result) is expected


# ---------------------------------------------------------------- guarded runner
@pytest.fixture()
def gate(monkeypatch):
    calls = {"ran": [], "progressed": [], "captured": []}
    monkeypatch.setattr(st, "_daily_task_already_ran", lambda name: calls.get("already", False))
    monkeypatch.setattr(st, "_mark_daily_task_ran", lambda name: calls["ran"].append(name))
    monkeypatch.setattr(st, "_capture_mtime", lambda path: calls["captured"].append(path) or 123.0)
    monkeypatch.setattr(
        st, "_mark_daily_task_if_state_progressed",
        lambda name, path, baseline, bot=None: calls["progressed"].append((name, path, baseline)) or True,
    )
    return calls


def _run_guarded(result):
    async def real(_context):
        return result

    asyncio.run(st._run_commodity_review_guarded(object(), real))


def test_noop_marks_ran_and_skips_the_state_advance_warning(gate):
    _run_guarded({"reviewed": 0, "skipped_recent": N, "failed": 0})
    assert gate["ran"] == ["commodity_review"]
    assert gate["progressed"] == []          # the gate (and its WARNING) is not consulted


@pytest.mark.parametrize(
    "result",
    [
        {"reviewed": 2, "skipped_recent": 2, "failed": 0},
        {"reviewed": 0, "skipped_recent": 3, "failed": 1},
        {"reviewed": 0, "skipped_recent": 0, "failed": 1, "error": True},
        None,
    ],
)
def test_real_work_or_failure_still_goes_through_the_gate(gate, result):
    _run_guarded(result)
    assert gate["ran"] == []                 # only the gate may stamp the marker
    assert len(gate["progressed"]) == 1
    name, path, baseline = gate["progressed"][0]
    assert name == "commodity_review" and path.endswith("commodity_research_state.json") and baseline == 123.0


def test_already_ran_today_skips_everything(gate):
    gate["already"] = True
    _run_guarded({"reviewed": 0, "skipped_recent": N, "failed": 0})
    assert gate["ran"] == [] and gate["progressed"] == [] and gate["captured"] == []


# ---------------------------------------------------------------- the review's own result
class _FixedDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 9, 29, 7, 0, tzinfo=tz)   # Tuesday 07:00 ET: inside the 5-10 AM guard


@pytest.fixture()
def review(monkeypatch):
    saved, sleeps = [], []
    state = {"analyses": []}
    monkeypatch.setattr(cr, "config", SimpleNamespace(news_intel_enabled=True))
    monkeypatch.setattr(cr, "datetime", _FixedDT)
    monkeypatch.setattr(cr, "_load_state", lambda: state)
    monkeypatch.setattr(cr, "_save_state", lambda s: saved.append(True))   # never the live state file
    monkeypatch.setattr(cr, "_record_analysis", lambda ctype, n, s: s["analyses"].append(
        {"commodity_type": ctype, "analyzed_at": _FixedDT.now(NY).isoformat()}))

    async def no_sleep(_s):
        sleeps.append(_s)

    monkeypatch.setattr(cr.asyncio, "sleep", no_sleep)
    return SimpleNamespace(state=state, saved=saved)


def _recent(review):
    review.state["analyses"] = [
        {"commodity_type": c, "analyzed_at": (_FixedDT.now(NY) - timedelta(days=1)).isoformat()}
        for c in cr.COMMODITY_RESEARCH_MAP
    ]


def _run_review():
    return asyncio.run(cr.run_weekly_commodity_review(object()))


def test_everything_recent_reports_skipped_not_failed(review):
    _recent(review)
    assert _run_review() == {"reviewed": 0, "skipped_recent": N, "failed": 0}
    assert review.saved == []                # state untouched, as before


def test_empty_research_counts_as_a_failure(review, monkeypatch):
    async def none_found(_ctype):
        return []

    monkeypatch.setattr(cr, "_gather_commodity_research", none_found)
    assert _run_review() == {"reviewed": 0, "skipped_recent": 0, "failed": N}


def test_missing_analysis_counts_as_a_failure(review, monkeypatch):
    async def found(_ctype):
        return [{"url": "https://example.test/a"}]

    async def no_analysis(**_kw):
        return None

    monkeypatch.setattr(cr, "_gather_commodity_research", found)
    monkeypatch.setattr(cr, "_recall_supply_balance_history", lambda ctype: "")
    monkeypatch.setattr(cr, "_validate_data_sources", lambda ctype, results: {})
    monkeypatch.setattr(cr, "_generate_supply_balance_analysis", no_analysis)
    assert _run_review() == {"reviewed": 0, "skipped_recent": 0, "failed": N}


def test_successful_review_counts_and_saves(review, monkeypatch):
    async def found(_ctype):
        return [{"url": "https://example.test/a"}]

    async def analysis(**_kw):
        return "analysis text"

    async def deliver(*_a, **_k):
        return None

    monkeypatch.setattr(cr, "_gather_commodity_research", found)
    monkeypatch.setattr(cr, "_recall_supply_balance_history", lambda ctype: "")
    monkeypatch.setattr(cr, "_validate_data_sources", lambda ctype, results: {})
    monkeypatch.setattr(cr, "_generate_supply_balance_analysis", analysis)
    monkeypatch.setattr(cr, "_deliver_commodity_intel", deliver)
    assert _run_review() == {"reviewed": N, "skipped_recent": 0, "failed": 0}
    assert len(review.saved) == N


def test_exception_returns_an_error_result_and_logs(review, monkeypatch, caplog):
    def boom():
        raise RuntimeError("state unreadable")

    monkeypatch.setattr(cr, "_load_state", boom)
    with caplog.at_level(logging.ERROR):
        result = _run_review()
    assert result == {"reviewed": 0, "skipped_recent": 0, "failed": 1, "error": True}
    assert "Weekly commodity review failed" in caplog.text
    assert st._commodity_review_was_noop(result) is False


def test_time_guard_still_returns_none(review, monkeypatch):
    class Weekend(_FixedDT):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 27, 7, 0, tzinfo=tz)   # Sunday

    monkeypatch.setattr(cr, "datetime", Weekend)
    assert _run_review() is None


def test_disabled_news_intel_still_returns_none(review, monkeypatch):
    monkeypatch.setattr(cr, "config", SimpleNamespace(news_intel_enabled=False))
    assert _run_review() is None


# ---------------------------------------------------------------- ERN-1
def test_earnings_exit_message_names_the_real_source():
    import inspect

    from agent.integrations import earnings_intel

    src = inspect.getsource(earnings_intel)
    assert "UW/FD upstream" not in src          # the stale S186 guess (source is DuckDuckGo)
    assert "DDG search returned none" in src
