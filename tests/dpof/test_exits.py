"""Tests for agent/dpof/exits.py.

Operator directive (2026-09-24): no mechanical trading. Stop/breakeven/
trail/TP-tiers/max-hold/thesis-vote/expiry-day-guard are GONE — the ONLY
exit rule left is the 15:50 ET forced flatten. These tests cover exactly
that, plus the surviving pure helpers (position_pnl_pct, parse_occ_expiry,
position_expiry).
"""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

from agent.dpof.exits import evaluate_exit, parse_occ_expiry, position_expiry, position_pnl_pct
from agent.dpof.models import PositionRecord

ET = ZoneInfo("America/New_York")


def occ_symbol(underlying: str, yymmdd: str, right: str, strike: float) -> str:
    letter = "C" if right == "call" else "P"
    return f"{underlying}{yymmdd}{letter}{int(round(strike * 1000)):08d}"


def mk_pos(now_et: datetime, **overrides) -> PositionRecord:
    defaults = dict(
        id="pos1",
        underlying="SPY",
        direction="bullish",
        structure="single",
        legs=[
            {
                "symbol": occ_symbol("SPY", "261015", "call", 575.0),  # ~3 weeks out by default
                "side": "buy",
                "ratio_qty": 1,
                "position_intent": "buy_to_open",
            }
        ],
        status="open",
        qty_original=10,
        qty_open=10,
        entry_price=5.00,
        entry_time=now_et.timestamp() - 30 * 60,  # 30 min ago
    )
    defaults.update(overrides)
    return PositionRecord(**defaults)


def mid_day(hour=11, minute=0, day=24):
    return datetime(2026, 9, day, hour, minute, 0, tzinfo=ET)


# ── position_pnl_pct ─────────────────────────────────────────────────────────


def test_position_pnl_pct_basic():
    assert position_pnl_pct(5.00, 5.50) == 0.10
    assert position_pnl_pct(5.00, 3.25) == -0.35


def test_position_pnl_pct_missing_inputs():
    assert position_pnl_pct(None, 5.00) is None
    assert position_pnl_pct(5.00, None) is None
    assert position_pnl_pct(0.0, 5.00) is None
    assert position_pnl_pct(-1.0, 5.00) is None


# ── parse_occ_expiry / position_expiry ──────────────────────────────────────


def test_parse_occ_expiry_valid():
    assert parse_occ_expiry(occ_symbol("SPY", "261015", "call", 575.0)) == date(2026, 10, 15)


def test_parse_occ_expiry_invalid_returns_none():
    assert parse_occ_expiry("not-an-occ-symbol") is None
    assert parse_occ_expiry("") is None


def test_position_expiry_reads_first_leg():
    now_et = mid_day()
    pos = mk_pos(now_et)
    assert position_expiry(pos) == date(2026, 10, 15)


def test_position_expiry_no_legs_returns_none():
    now_et = mid_day()
    pos = mk_pos(now_et, legs=[])
    assert position_expiry(pos) is None


# ── evaluate_exit: EOD flatten (the ONE kept rail) ──────────────────────────


def test_evaluate_exit_eod_flatten_exact_boundary(cfg):
    now_et = datetime(2026, 9, 24, 15, 50, 0, tzinfo=ET)  # cfg.flatten_et default "15:50"
    pos = mk_pos(now_et)
    decision = evaluate_exit(pos, 5.00, now_et, cfg)
    assert decision.action == "close"
    assert decision.reason == "eod_flatten"
    assert decision.urgent is True
    assert decision.qty == pos.qty_open


def test_evaluate_exit_eod_flatten_just_before_boundary_does_not_fire(cfg):
    now_et = datetime(2026, 9, 24, 15, 49, 59, tzinfo=ET)
    pos = mk_pos(now_et)
    decision = evaluate_exit(pos, 5.10, now_et, cfg)
    assert decision.action == "hold"
    assert decision.reason != "eod_flatten"


def test_evaluate_exit_eod_flatten_fires_regardless_of_pnl(cfg):
    """No mechanical stop any more -- a huge unrealized loss changes nothing
    about whether/why the position closes at flatten time."""
    now_et = datetime(2026, 9, 24, 15, 55, 0, tzinfo=ET)
    pos = mk_pos(now_et, entry_price=5.00)
    mark = 3.00  # a -40% mark; there is no stop_loss any more to race against
    decision = evaluate_exit(pos, mark, now_et, cfg)
    assert decision.action == "close"
    assert decision.reason == "eod_flatten"


def test_evaluate_exit_eod_flatten_fires_even_with_no_mark(cfg):
    """The flatten is pure wall-clock arithmetic -- it doesn't need (or read) a
    mark at all, unlike the old ladder's 'no_mark' hold."""
    now_et = datetime(2026, 9, 24, 15, 55, 0, tzinfo=ET)
    pos = mk_pos(now_et)
    decision = evaluate_exit(pos, None, now_et, cfg)
    assert decision.action == "close"
    assert decision.reason == "eod_flatten"


def test_evaluate_exit_eod_flatten_disabled(cfg):
    c = replace(cfg, flatten_eod=False)
    now_et = datetime(2026, 9, 24, 15, 55, 0, tzinfo=ET)
    pos = mk_pos(now_et)
    decision = evaluate_exit(pos, 5.10, now_et, c)
    assert decision.action == "hold"
    assert decision.reason != "eod_flatten"


# ── evaluate_exit: no mechanical exit outside the flatten window ───────────


def test_evaluate_exit_holds_regardless_of_mark_or_time_before_flatten(cfg):
    """No stop-loss, breakeven, trail, TP tiers, max-hold, or thesis-vote any
    more -- ANY mark/entry_time combination before flatten_et holds."""
    now_et = mid_day()
    for mark, entry_time_ago_min in [(5.00, 30), (0.50, 400), (50.0, 5), (None, 30)]:
        pos = mk_pos(now_et, entry_price=5.00, entry_time=now_et.timestamp() - entry_time_ago_min * 60)
        decision = evaluate_exit(pos, mark, now_et, cfg)
        assert decision.action == "hold"
        assert decision.qty == 0
        assert decision.reason == "no_exit_condition_met"


def test_evaluate_exit_never_produces_a_partial_action(cfg):
    """Take-profit tiering (the only source of a 'partial' action in the old
    ladder) is gone -- evaluate_exit only ever returns hold/close now."""
    now_et = mid_day()
    pos = mk_pos(now_et, entry_price=5.00, qty_original=10, qty_open=10)
    for mark in (5.05, 6.50, 10.0, 0.5, None):
        decision = evaluate_exit(pos, mark, now_et, cfg)
        assert decision.action in ("hold", "close")


# ── evaluate_exit: inputs not mutated ────────────────────────────────────────


def test_evaluate_exit_does_not_mutate_position_record(cfg):
    now_et = mid_day()
    pos = mk_pos(now_et, entry_price=5.00, qty_original=10, qty_open=10)
    snapshot = deepcopy(pos)
    _decision = evaluate_exit(pos, 5.05, now_et, cfg)
    assert pos == snapshot
    assert pos.qty_open == 10
