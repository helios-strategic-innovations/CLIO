"""Tests for agent.dpof.trade_stats (Task ORCH-7Q4K-DPOF builder B2,
docs/design/clio_dpof_followups_2026-09-26.md §4, 2026-09-26): the pure
formulas (`is_real_trade`, `pnl_incomplete`, `compute_trade_stats`,
`render_trade_stats`), parity with agent.dpof.reflection's exclusion
constants, the new readmodel.py readers/windows (`load_archived_positions`,
`_all_history_positions`, `trade_stats_text`), the EOD digest's "Trades"
section, and `/dpof stats` dispatch.

Plugin-free (no pytest-asyncio), matching the rest of tests/dpof/.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from agent.dpof import readmodel, trade_stats
from agent.dpof.models import PositionRecord

ET = ZoneInfo("America/New_York")


def _pos(
    id="p1",
    status="closed",
    qty_original=1,
    entry_price=1.0,
    closed_time=1000.0,
    realized_pnl=0.0,
    exit_reason="decider_exit",
    order_fills=None,
    underlying="SPY",
) -> PositionRecord:
    return PositionRecord(
        id=id,
        underlying=underlying,
        direction="bullish",
        structure="single",
        status=status,
        qty_original=qty_original,
        qty_open=0,
        entry_price=entry_price,
        closed_time=closed_time,
        realized_pnl=realized_pnl,
        exit_reason=exit_reason,
        order_fills=order_fills or {},
    )


# ─────────────────────────────────────────────────────────────────────────
# is_real_trade / pnl_incomplete
# ─────────────────────────────────────────────────────────────────────────

def test_is_real_trade_true_for_normal_closed_position():
    assert trade_stats.is_real_trade(_pos()) is True


def test_is_real_trade_false_when_not_closed():
    assert trade_stats.is_real_trade(_pos(status="open")) is False


def test_is_real_trade_false_when_qty_original_not_positive():
    assert trade_stats.is_real_trade(_pos(qty_original=0)) is False


def test_is_real_trade_excludes_unfilled_entry():
    assert trade_stats.is_real_trade(_pos(entry_price=None)) is False


def test_is_real_trade_false_when_closed_time_missing():
    assert trade_stats.is_real_trade(_pos(closed_time=None)) is False


@pytest.mark.parametrize("reason", ["log_only", "submit_unconfirmed", "reconciled_absent"])
def test_is_real_trade_excludes_every_named_exit_reason(reason):
    assert trade_stats.is_real_trade(_pos(exit_reason=reason)) is False


@pytest.mark.parametrize("reason", ["submit_failed:timeout", "submit_failed:", "submit_failed:rejected"])
def test_is_real_trade_excludes_submit_failed_prefix(reason):
    assert trade_stats.is_real_trade(_pos(exit_reason=reason)) is False


def test_pnl_incomplete_true_for_reconcile_fill():
    pos = _pos(order_fills={"reconcile:1:SPY": {"purpose": "reconcile_add", "avg": None}})
    assert trade_stats.pnl_incomplete(pos) is True


def test_pnl_incomplete_true_for_reconcile_remove():
    pos = _pos(order_fills={"reconcile:1:SPY": {"purpose": "reconcile_remove", "avg": None}})
    assert trade_stats.pnl_incomplete(pos) is True


def test_pnl_incomplete_false_for_normal_fills():
    pos = _pos(order_fills={"cid1": {"purpose": "entry", "avg": 1.23}, "cid2": {"purpose": "exit", "avg": 1.5}})
    assert trade_stats.pnl_incomplete(pos) is False


def test_pnl_incomplete_false_when_no_fills():
    assert trade_stats.pnl_incomplete(_pos(order_fills={})) is False


# ─────────────────────────────────────────────────────────────────────────
# compute_trade_stats -- formulas
# ─────────────────────────────────────────────────────────────────────────

def test_compute_trade_stats_formulas_hand_computed():
    # Chronological order: +100 (t=1000), -40 (t=2000), +60 (t=3000)
    positions = [
        _pos(id="a", realized_pnl=100.0, closed_time=1000.0),
        _pos(id="b", realized_pnl=-40.0, closed_time=2000.0),
        _pos(id="c", realized_pnl=60.0, closed_time=3000.0),
    ]
    stats = trade_stats.compute_trade_stats(positions)
    assert stats["n"] == 3
    assert stats["n_complete"] == 3
    assert stats["incomplete"] == 0
    assert stats["wins"] == 2
    assert stats["losses"] == 1
    assert stats["scratches"] == 0
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["net_pnl"] == pytest.approx(120.0)
    assert stats["avg_win"] == pytest.approx(80.0)
    assert stats["avg_loss"] == pytest.approx(-40.0)
    assert stats["profit_factor"] == pytest.approx(4.0)
    assert stats["expectancy"] == pytest.approx(40.0)
    assert stats["best"] == pytest.approx(100.0)
    assert stats["worst"] == pytest.approx(-40.0)
    # cum after +100 -> 100 (peak 100); after -40 -> 60 (fall 40); after +60 -> 120 (new peak)
    assert stats["max_drawdown"] == pytest.approx(-40.0)


def test_compute_trade_stats_no_losses_pf_is_none_never_inf():
    positions = [
        _pos(id="a", realized_pnl=100.0, closed_time=1000.0),
        _pos(id="b", realized_pnl=50.0, closed_time=2000.0),
    ]
    stats = trade_stats.compute_trade_stats(positions)
    assert stats["losses"] == 0
    assert stats["profit_factor"] is None
    assert stats["avg_loss"] is None
    assert stats["win_rate"] == pytest.approx(1.0)
    assert stats["expectancy"] == pytest.approx(75.0)


def test_compute_trade_stats_empty_input_all_none():
    stats = trade_stats.compute_trade_stats([])
    assert stats["n"] == 0
    assert stats["wins"] == 0
    assert stats["losses"] == 0
    assert stats["scratches"] == 0
    assert stats["incomplete"] == 0
    assert stats["n_complete"] == 0
    for key in (
        "win_rate", "net_pnl", "avg_win", "avg_loss", "profit_factor",
        "expectancy", "best", "worst", "max_drawdown",
    ):
        assert stats[key] is None, key


def test_compute_trade_stats_scratches_excluded_from_win_rate_denominator():
    positions = [
        _pos(id="w", realized_pnl=50.0, closed_time=1000.0),
        _pos(id="s", realized_pnl=0.0, closed_time=2000.0),
        _pos(id="l", realized_pnl=-30.0, closed_time=3000.0),
    ]
    stats = trade_stats.compute_trade_stats(positions)
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["scratches"] == 1
    assert stats["win_rate"] == pytest.approx(0.5)  # 1 / (1 + 1), scratch excluded
    assert stats["net_pnl"] == pytest.approx(20.0)


def test_compute_trade_stats_drawdown_independent_of_input_order():
    a = _pos(id="a", realized_pnl=100.0, closed_time=1000.0)
    b = _pos(id="b", realized_pnl=-40.0, closed_time=2000.0)
    c = _pos(id="c", realized_pnl=60.0, closed_time=3000.0)
    stats_forward = trade_stats.compute_trade_stats([a, b, c])
    stats_shuffled = trade_stats.compute_trade_stats([c, a, b])
    assert stats_forward["max_drawdown"] == stats_shuffled["max_drawdown"] == pytest.approx(-40.0)


def test_compute_trade_stats_no_fall_drawdown_is_zero_not_none():
    positions = [
        _pos(id="a", realized_pnl=10.0, closed_time=1000.0),
        _pos(id="b", realized_pnl=10.0, closed_time=2000.0),
    ]
    stats = trade_stats.compute_trade_stats(positions)
    assert stats["max_drawdown"] == 0.0
    assert stats["max_drawdown"] is not None


def test_compute_trade_stats_incomplete_positions_excluded_from_dollar_stats():
    complete = _pos(id="a", realized_pnl=100.0, closed_time=1000.0)
    incomplete = _pos(
        id="b", realized_pnl=999.0, closed_time=2000.0,
        order_fills={"reconcile:1:SPY": {"purpose": "reconcile_add", "avg": None}},
    )
    stats = trade_stats.compute_trade_stats([complete, incomplete])
    assert stats["n"] == 2
    assert stats["incomplete"] == 1
    assert stats["n_complete"] == 1
    assert stats["wins"] == 1
    assert stats["net_pnl"] == pytest.approx(100.0)  # 999 never enters a $ stat
    assert stats["best"] == pytest.approx(100.0)


def test_compute_trade_stats_ignores_non_real_trades():
    real = _pos(id="a", realized_pnl=50.0, closed_time=1000.0)
    fake = _pos(id="b", realized_pnl=-9999.0, closed_time=1500.0, exit_reason="log_only")
    stats = trade_stats.compute_trade_stats([real, fake])
    assert stats["n"] == 1
    assert stats["net_pnl"] == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────
# render_trade_stats
# ─────────────────────────────────────────────────────────────────────────

def test_render_trade_stats_matches_expected_line():
    positions = [
        _pos(id="a", realized_pnl=100.0, closed_time=1000.0),
        _pos(id="b", realized_pnl=-40.0, closed_time=2000.0),
        _pos(id="c", realized_pnl=60.0, closed_time=3000.0),
    ]
    stats = trade_stats.compute_trade_stats(positions)
    text = trade_stats.render_trade_stats("today", stats)
    assert text == (
        "Trades today: 3 (2W 1L) | win 67% | net +$120 | PF 4.0 | "
        "exp +$40/trade | avg W +$80 L -$40 | maxDD -$40"
    )


def test_render_trade_stats_na_for_empty():
    stats = trade_stats.compute_trade_stats([])
    text = trade_stats.render_trade_stats("today", stats)
    assert text == (
        "Trades today: 0 (0W 0L) | win n/a | net n/a | PF n/a | "
        "exp n/a | avg W n/a L n/a | maxDD n/a"
    )


def test_render_trade_stats_zero_dollars_has_no_sign():
    stats = trade_stats.compute_trade_stats([_pos(id="s", realized_pnl=0.0)])
    text = trade_stats.render_trade_stats("today", stats)
    assert "net $0" in text and "+$0" not in text and "-$0" not in text


def test_render_trade_stats_shows_scratches_and_incomplete_note():
    positions = [
        _pos(id="w", realized_pnl=50.0, closed_time=1000.0),
        _pos(id="s", realized_pnl=0.0, closed_time=2000.0),
        _pos(
            id="i", realized_pnl=999.0, closed_time=3000.0,
            order_fills={"reconcile:1:SPY": {"purpose": "reconcile_add", "avg": None}},
        ),
    ]
    stats = trade_stats.compute_trade_stats(positions)
    text = trade_stats.render_trade_stats("today", stats)
    lines = text.split("\n")
    assert "0S" not in lines[0] and "1S" in lines[0]
    assert lines[1] == "(1 excluded: P&L incomplete after reconcile)"


# ─────────────────────────────────────────────────────────────────────────
# Parity with reflection.py's exclusion constants
# ─────────────────────────────────────────────────────────────────────────

def test_exclusion_constants_match_reflection_module():
    from agent.dpof import reflection

    assert trade_stats._EXCLUDED_EXIT_REASONS == reflection._EXCLUDED_EXIT_REASONS
    assert trade_stats._EXCLUDED_EXIT_REASON_PREFIXES == reflection._EXCLUDED_EXIT_REASON_PREFIXES


# ─────────────────────────────────────────────────────────────────────────
# readmodel.load_archived_positions
# ─────────────────────────────────────────────────────────────────────────

def _write_archive_jsonl(ddir: str, records: list[dict], trailing_junk: bool = True) -> None:
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "positions_archive.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        if trailing_junk:
            f.write("not valid json\n")


def _write_positions_json(ddir: str, positions: list[PositionRecord]) -> None:
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "positions.json"), "w") as f:
        json.dump({"version": 1, "positions": [p.to_dict() for p in positions]}, f)


def test_load_archived_positions_missing_file_returns_empty(tmp_path):
    assert readmodel.load_archived_positions(str(tmp_path)) == []


def test_load_archived_positions_skips_unparsable_and_idless_lines(tmp_path):
    d = str(tmp_path)
    good = _pos(id="x1", realized_pnl=10.0, closed_time=500.0)
    _write_archive_jsonl(d, [good.to_dict(), {"underlying": "SPY"}])  # 2nd has no id
    out = readmodel.load_archived_positions(d)
    assert len(out) == 1
    assert out[0].id == "x1"
    assert out[0].realized_pnl == 10.0


# ─────────────────────────────────────────────────────────────────────────
# readmodel._all_history_positions -- dedupe (positions.json wins, R4)
# ─────────────────────────────────────────────────────────────────────────

def test_all_history_dedup_positions_json_wins(tmp_path):
    d = str(tmp_path)
    archived = _pos(id="dup", realized_pnl=5.0, closed_time=1000.0)
    _write_archive_jsonl(d, [archived.to_dict()], trailing_junk=False)
    live = _pos(id="dup", realized_pnl=999.0, closed_time=1000.0)
    _write_positions_json(d, [live])

    merged = readmodel._all_history_positions(d)

    assert len(merged) == 1
    assert merged[0].realized_pnl == 999.0


def test_all_history_includes_both_archive_only_and_live_only(tmp_path):
    d = str(tmp_path)
    archived_only = _pos(id="old", realized_pnl=1.0, closed_time=500.0)
    _write_archive_jsonl(d, [archived_only.to_dict()], trailing_junk=False)
    live_only = _pos(id="new", realized_pnl=2.0, closed_time=1500.0)
    _write_positions_json(d, [live_only])

    merged = readmodel._all_history_positions(d)

    ids = {p.id for p in merged}
    assert ids == {"old", "new"}


# ─────────────────────────────────────────────────────────────────────────
# readmodel.trade_stats_text -- windows
# ─────────────────────────────────────────────────────────────────────────

def test_trade_stats_text_today_vs_all_history(tmp_path):
    d = str(tmp_path)
    today_epoch = datetime(2026, 9, 25, 12, 0, tzinfo=ET).timestamp()
    yesterday_epoch = datetime(2026, 9, 24, 12, 0, tzinfo=ET).timestamp()
    positions = [
        _pos(id="t1", realized_pnl=50.0, closed_time=today_epoch),
        _pos(id="y1", realized_pnl=-20.0, closed_time=yesterday_epoch),
    ]
    _write_positions_json(d, positions)

    text = readmodel.trade_stats_text(d, now_epoch=today_epoch)
    lines = text.split("\n")

    assert lines[0].startswith("Trades today: 1")
    assert lines[1].startswith("Trades since 2026-09-24: 2")


def test_trade_stats_text_empty_uses_all_time_label(tmp_path):
    text = readmodel.trade_stats_text(str(tmp_path))
    lines = text.split("\n")
    assert lines[0].startswith("Trades today: 0")
    assert lines[1].startswith("Trades all-time: 0")


def test_trade_stats_text_never_raises_on_corrupt_positions_json(tmp_path):
    d = str(tmp_path)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "positions.json"), "w") as f:
        f.write("{not valid json")
    text = readmodel.trade_stats_text(d)  # must not raise
    assert isinstance(text, str) and "Trades" in text


# ─────────────────────────────────────────────────────────────────────────
# eod_digest_text -- "Trades" section
# ─────────────────────────────────────────────────────────────────────────

def _write_day_stats(ddir: str, stats: dict) -> None:
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "signal_day_stats.json"), "w") as f:
        json.dump(stats, f)


def test_eod_digest_contains_trades_section(tmp_path):
    d = str(tmp_path)
    _write_day_stats(d, {"day": "2026-09-25", "entries": 0, "nominations": 0, "cycles_evaluated": 10})
    text = readmodel.eod_digest_text(ddir=d, today="2026-09-25")
    assert "Entries: 0" in text
    assert "Trades today:" in text
    assert "Trades all-time:" in text


def test_eod_digest_trades_section_reflects_real_data(tmp_path):
    # trade_stats_text() (called by eod_digest_text with no now_epoch override)
    # windows "today" off the REAL clock -- unlike the signal-pipeline stats
    # section, which compares against the `today` argument. Anchor the fixture
    # to the actual current ET date so this test is independent of when it runs.
    d = str(tmp_path)
    today = readmodel._today_et()
    _write_day_stats(d, {"day": today, "entries": 1, "nominations": 1, "cycles_evaluated": 10})
    closed_epoch = datetime.strptime(today, "%Y-%m-%d").replace(hour=12, minute=0, tzinfo=ET).timestamp()
    _write_positions_json(d, [_pos(id="a", realized_pnl=42.0, closed_time=closed_epoch)])
    text = readmodel.eod_digest_text(ddir=d, today=today)
    assert "Trades today: 1" in text
    assert "+$42" in text


def test_eod_digest_trades_today_follows_the_today_argument(tmp_path):
    """An explicit `today` windows the Trades section too, not only the
    signal-pipeline section (pass-2 review NIT-1)."""
    d = str(tmp_path)
    _write_day_stats(d, {"day": "2026-09-25", "entries": 1, "nominations": 1, "cycles_evaluated": 10})
    closed_epoch = datetime(2026, 9, 25, 12, 0, tzinfo=ET).timestamp()
    _write_positions_json(d, [_pos(id="a", realized_pnl=42.0, closed_time=closed_epoch)])
    text = readmodel.eod_digest_text(ddir=d, today="2026-09-25")
    assert "Trades today: 1" in text


def test_eod_digest_survives_corrupt_positions_json(tmp_path):
    d = str(tmp_path)
    _write_day_stats(d, {"day": "2026-09-25", "entries": 0, "nominations": 0, "cycles_evaluated": 10})
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "positions.json"), "w") as f:
        f.write("{not valid json")
    text = readmodel.eod_digest_text(ddir=d, today="2026-09-25")  # must not raise
    assert isinstance(text, str) and text
    assert "Trades" in text


def test_eod_digest_no_signal_cycles_early_return_unaffected(tmp_path):
    """The digest's existing 'no signal cycles today' early return must
    stay -- it must NOT gain a Trades section (F2 never fires on that path)."""
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")
    assert "no signal cycles were recorded today" in text
    assert "Trades" not in text


# ─────────────────────────────────────────────────────────────────────────
# /dpof stats dispatch
# ─────────────────────────────────────────────────────────────────────────

class _FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class _FakeUpdate:
    def __init__(self, user_id):
        self.effective_user = types.SimpleNamespace(id=user_id) if user_id is not None else None
        self.message = _FakeMessage()


class _FakeContext:
    def __init__(self, args=None):
        self.args = args or []


def _patch_telegram_bot_module(monkeypatch, authorized_ids):
    fake_module = types.ModuleType("agent.integrations.telegram_bot")
    fake_module.is_authorized = lambda uid: uid in authorized_ids
    monkeypatch.setitem(sys.modules, "agent.integrations.telegram_bot", fake_module)


def _patch_get_engine(monkeypatch):
    fake_module = types.ModuleType("agent.dpof.engine")
    fake_module.get_engine = lambda: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "agent.dpof.engine", fake_module)


def test_dpof_stats_dispatch_calls_readmodel_off_loop(monkeypatch):
    import threading

    import agent.dpof.telegram_commands as tc_mod

    _patch_telegram_bot_module(monkeypatch, {1})
    _patch_get_engine(monkeypatch)
    seen = {}

    def _stats():
        seen["thread"] = threading.current_thread() is threading.main_thread()
        return "STATS TEXT"

    monkeypatch.setattr(readmodel, "trade_stats_text", _stats)

    update = _FakeUpdate(1)
    asyncio.run(tc_mod.dpof_command(update, _FakeContext(["stats"])))

    assert update.message.replies == ["STATS TEXT"]
    assert seen["thread"] is False


def test_usage_string_mentions_stats(monkeypatch):
    import agent.dpof.telegram_commands as tc_mod

    _patch_telegram_bot_module(monkeypatch, {1})
    _patch_get_engine(monkeypatch)

    update = _FakeUpdate(1)
    asyncio.run(tc_mod.dpof_command(update, _FakeContext(["bogus"])))

    assert len(update.message.replies) == 1
    assert "stats" in update.message.replies[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
