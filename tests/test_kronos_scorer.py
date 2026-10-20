"""agent.kronos.scorer -- status-driven resolution (CLAUDE.md #36: a late
run still scores old pending entries), metrics computation, null-when-empty,
and the history-file cap."""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from agent.kronos.bars import Bar
from agent.kronos.scorer import record_forecast, resolve_and_score

_ET = ZoneInfo("America/New_York")


def _et(y, m, d, hh, mm) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=_ET)


def _bar(t_open_et: datetime, c: float, minutes: int = 5) -> Bar:
    t_open = t_open_et.timestamp()
    return Bar(t_open=t_open, t_close=t_open + minutes * 60, o=c, h=c, l=c, c=c, v=100.0, amount=100.0 * c)


@pytest.fixture
def history_file(tmp_path):
    return str(tmp_path / "kronos_history.jsonl")


def test_record_forecast_noop_on_empty_horizons(history_file):
    record_forecast(history_file, 1000, symbol="SPY", anchor_epoch=1.0, anchor_close=100.0, horizons={})
    import os
    assert not os.path.exists(history_file)


def test_record_forecast_appends_one_row_per_horizon(history_file):
    horizons = {
        "15": {"minutes": 15, "p_up": 0.7, "expected_abs_move_pct": 0.2},
        "30": {"minutes": 30, "p_up": 0.4, "expected_abs_move_pct": 0.3},
    }
    record_forecast(history_file, 1000, symbol="SPY", anchor_epoch=1_700_000_000.0, anchor_close=500.0, horizons=horizons)
    with open(history_file) as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == 2
    labels = {r["horizon_minutes"] for r in rows}
    assert labels == {15, 30}
    for r in rows:
        assert r["resolved"] is False
        assert r["realized_return_pct"] is None
        assert r["symbol"] == "SPY"
        assert r["anchor_close"] == 500.0


def test_resolve_and_score_resolves_when_target_bar_present(history_file):
    anchor_et = _et(2026, 9, 22, 11, 0)
    anchor_epoch = anchor_et.timestamp()
    record_forecast(
        history_file, 1000, symbol="SPY", anchor_epoch=anchor_epoch, anchor_close=100.0,
        horizons={"15": {"minutes": 15, "p_up": 0.7, "expected_abs_move_pct": 0.5}},
    )
    # Target bar: closes at anchor + 15 min = 11:15, realized close 101.0 -> +1.0% return.
    target_bar = _bar(_et(2026, 9, 22, 11, 10), c=101.0)  # opens 11:10, closes 11:15
    now_epoch = _et(2026, 9, 22, 11, 20).timestamp()
    scores = resolve_and_score(history_file, 1000, bars_by_symbol={"SPY": [target_bar]}, now_epoch=now_epoch)
    m = scores["symbols"]["SPY"]["15"]["all"]
    assert m["n"] == 1
    assert m["hit_rate"] == 1.0  # predicted up (p_up=0.7 >= 0.5), realized up (+1.0%) -> hit
    assert m["n_confident"] == 1  # p_up=0.7 >= 0.6
    assert m["brier"] == pytest.approx((0.7 - 1.0) ** 2)
    assert m["mae_abs_move_pct"] == pytest.approx(abs(0.5 - 1.0))

    with open(history_file) as fh:
        rows = [json.loads(line) for line in fh]
    assert rows[0]["resolved"] is True
    assert rows[0]["realized_return_pct"] == pytest.approx(1.0)


def test_resolve_and_score_leaves_unresolved_when_no_target_bar_yet(history_file):
    anchor_et = _et(2026, 9, 22, 11, 0)
    record_forecast(
        history_file, 1000, symbol="SPY", anchor_epoch=anchor_et.timestamp(), anchor_close=100.0,
        horizons={"15": {"minutes": 15, "p_up": 0.7, "expected_abs_move_pct": 0.5}},
    )
    # No bars at all -- can't resolve yet.
    now_epoch = _et(2026, 9, 22, 11, 5).timestamp()
    scores = resolve_and_score(history_file, 1000, bars_by_symbol={"SPY": []}, now_epoch=now_epoch)
    m = scores["symbols"].get("SPY", {}).get("15", {}).get("all")
    # No resolved rows at all yet -> symbol/horizon key absent entirely (nothing to report).
    assert "SPY" not in scores["symbols"] or m is None or m["n"] == 0
    with open(history_file) as fh:
        rows = [json.loads(line) for line in fh]
    assert rows[0]["resolved"] is False


def test_resolve_and_score_is_status_driven_a_late_run_resolves_old_entries(history_file):
    """CLAUDE.md #36: discovery is by status (resolved=false), never a
    fixed recency window -- an entry written long ago, and never resolved
    on time (e.g. the service was down), still resolves the moment a
    matching bar shows up in ANY later run."""
    old_anchor_et = _et(2026, 9, 20, 11, 0)  # days before "now"
    record_forecast(
        history_file, 1000, symbol="SPY", anchor_epoch=old_anchor_et.timestamp(), anchor_close=100.0,
        horizons={"15": {"minutes": 15, "p_up": 0.3, "expected_abs_move_pct": 0.4}},
    )
    # First run: still no target bar -> stays pending.
    resolve_and_score(history_file, 1000, bars_by_symbol={"SPY": []}, now_epoch=old_anchor_et.timestamp() + 60)

    # A much later run finally has the target bar available.
    target_bar = _bar(old_anchor_et + timedelta(minutes=10), c=99.0)  # -1.0% realized
    later_epoch = _et(2026, 9, 22, 9, 0).timestamp()
    scores = resolve_and_score(history_file, 1000, bars_by_symbol={"SPY": [target_bar]}, now_epoch=later_epoch)
    m = scores["symbols"]["SPY"]["15"]["all"]
    assert m["n"] == 1
    assert m["hit_rate"] == 1.0  # predicted down (p_up=0.3), realized down (-1.0%) -> hit


def test_metrics_are_null_when_no_resolved_entries(history_file):
    record_forecast(
        history_file, 1000, symbol="QQQ", anchor_epoch=1_700_000_000.0, anchor_close=400.0,
        horizons={"30": {"minutes": 30, "p_up": 0.5, "expected_abs_move_pct": 0.1}},
    )
    scores = resolve_and_score(history_file, 1000, bars_by_symbol={}, now_epoch=1_700_010_000.0)
    # Nothing resolved -> no symbol/horizon entry should carry a non-null count.
    assert scores["symbols"] == {} or all(
        h["all"]["n"] == 0 and h["all"]["hit_rate"] is None
        for sym in scores["symbols"].values() for h in sym.values()
    )


def test_history_file_capped_at_max_lines(history_file):
    for i in range(10):
        record_forecast(
            history_file, 5, symbol="SPY", anchor_epoch=float(1_700_000_000 + i), anchor_close=100.0,
            horizons={"15": {"minutes": 15, "p_up": 0.5, "expected_abs_move_pct": 0.1}},
        )
    with open(history_file) as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == 5
    # Newest entries kept (highest anchor_epoch survives).
    assert max(r["anchor_epoch"] for r in rows) == 1_700_000_000 + 9


def test_resolve_and_score_recent_window_excludes_old_sessions(history_file):
    """'recent' = current session + 4 prior sessions; an entry from well
    before that cutoff appears in 'all' but not 'recent'."""
    old_et = _et(2026, 9, 1, 11, 0)  # ~3 weeks earlier -- far outside 5 sessions
    record_forecast(
        history_file, 1000, symbol="SPY", anchor_epoch=old_et.timestamp(), anchor_close=100.0,
        horizons={"15": {"minutes": 15, "p_up": 0.9, "expected_abs_move_pct": 1.0}},
    )
    target_bar = _bar(old_et + timedelta(minutes=10), c=102.0)
    now_epoch = _et(2026, 9, 22, 9, 0).timestamp()
    scores = resolve_and_score(history_file, 1000, bars_by_symbol={"SPY": [target_bar]}, now_epoch=now_epoch)
    m_all = scores["symbols"]["SPY"]["15"]["all"]
    m_recent = scores["symbols"]["SPY"]["15"]["recent"]
    assert m_all["n"] == 1
    assert m_recent["n"] == 0
    assert m_recent["hit_rate"] is None


def test_resolve_and_score_recent_window_includes_current_session(history_file):
    anchor_et = _et(2026, 9, 22, 11, 0)
    record_forecast(
        history_file, 1000, symbol="SPY", anchor_epoch=anchor_et.timestamp(), anchor_close=100.0,
        horizons={"15": {"minutes": 15, "p_up": 0.9, "expected_abs_move_pct": 1.0}},
    )
    target_bar = _bar(anchor_et + timedelta(minutes=10), c=102.0)
    now_epoch = _et(2026, 9, 22, 11, 20).timestamp()
    scores = resolve_and_score(history_file, 1000, bars_by_symbol={"SPY": [target_bar]}, now_epoch=now_epoch)
    assert scores["symbols"]["SPY"]["15"]["recent"]["n"] == 1
