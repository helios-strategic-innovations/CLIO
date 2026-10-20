"""Tests for the DPOF entry-outcome observability wiring (Task ORCH-7Q4K-DPOF,
2026-09-25): engine.py's _signal_block_outcome/_gate_outcome/
_classify_harvest_outcome/_fallback_entry_outcome/_note_entry_outcome, and
_log_cycle_signals end-to-end via real run_cycle()s.

Per the build task: drive real cycles where the fakes allow (the
fresh-nomination taxonomy — outside_entry_window/halted/data_unhealthy/
composite_unavailable/below_threshold/regime_requires_strong/dp_veto/
price_not_confirmed/position_open/deliberation_in_flight/cooldown/
insufficient_capital/decider_unavailable/no_chain/no_candidates/deliberating);
unit-test the classifier otherwise (the harvest taxonomy —
_classify_harvest_outcome reads a decision_log dict directly, exactly as
_apply_entry_outcome builds one, without needing 15 separate multi-cycle
integration tests for every branch).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace

import pytest

from agent.dpof import engine as engine_module
from agent.dpof import risk as risk_module
from agent.dpof.config import DpofMode, JevMode
from agent.dpof.engine import DPOFEngine
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import Direction, JevVerdict, OptionQuote, PillarReading, Regime, RiskState, Signal, UnderlyingSnapshot

from tests.dpof.test_engine import (
    FakeBroker,
    FakeDecider,
    FakeEntryDecision,
    FakeFeed,
    FakeOrderManager,
    _base_dt,
    _bullish_snapshot,
    _bullish_underlying,
    _dedupe_alerts,
    _fake_candidate_fn,
    _make_chain,
    _make_engine,
    _neutral_snapshot,
    _patch_entry_candidate_cls,
    _patch_jev_entry_check,
    _position,
)


# ─────────────────────────────────────────────────────────────────────────
# Pure classifiers: _signal_block_outcome / _gate_outcome /
# _classify_harvest_outcome
# ─────────────────────────────────────────────────────────────────────────

def _sig(**overrides) -> Signal:
    base = dict(
        symbol="SPY", direction=Direction.NEUTRAL, composite=None, conviction=0.0,
        strength="none", regime=Regime.NORMAL, pillars_used={}, price_confirmed=False,
        reasons=[], vetoes=[],
    )
    base.update(overrides)
    return Signal(**base)


def test_signal_block_outcome_data_unhealthy_takes_priority():
    sig = _sig(direction=Direction.BULLISH, strength="strong", vetoes=["data_unhealthy", "price_not_confirmed"])
    assert DPOFEngine._signal_block_outcome(sig) == "data_unhealthy"


def test_signal_block_outcome_composite_unavailable():
    sig = _sig(direction=Direction.NEUTRAL, reasons=["composite_unavailable"])
    assert DPOFEngine._signal_block_outcome(sig) == "composite_unavailable"


def test_signal_block_outcome_below_threshold():
    sig = _sig(direction=Direction.NEUTRAL, reasons=["composite_below_normal_threshold"])
    assert DPOFEngine._signal_block_outcome(sig) == "below_threshold"


def test_signal_block_outcome_regime_requires_strong():
    sig = _sig(direction=Direction.NEUTRAL, reasons=["regime_requires_strong"])
    assert DPOFEngine._signal_block_outcome(sig) == "regime_requires_strong"


def test_signal_block_outcome_dp_veto():
    sig = _sig(direction=Direction.BULLISH, strength="strong", vetoes=["dp_veto"])
    assert DPOFEngine._signal_block_outcome(sig) == "dp_veto"


def test_signal_block_outcome_price_not_confirmed():
    sig = _sig(direction=Direction.BEARISH, strength="normal", vetoes=["price_not_confirmed"])
    assert DPOFEngine._signal_block_outcome(sig) == "price_not_confirmed"


def test_signal_block_outcome_unknown_veto_defensive_fallback():
    sig = _sig(direction=Direction.BULLISH, strength="strong", vetoes=["some_future_veto"])
    assert DPOFEngine._signal_block_outcome(sig) == "vetoed"


def test_gate_outcome_hard_halt():
    assert DPOFEngine._gate_outcome(["hard_halt:operator"]) == "halted"


def test_gate_outcome_halt_file():
    assert DPOFEngine._gate_outcome(["halt_file"]) == "halted"


def test_gate_outcome_outside_entry_window():
    assert DPOFEngine._gate_outcome(["outside_entry_window"]) == "outside_entry_window"


def test_gate_outcome_halt_takes_priority_over_window():
    assert DPOFEngine._gate_outcome(["hard_halt:operator", "outside_entry_window"]) == "halted"


def test_gate_outcome_defensive_fallback():
    assert DPOFEngine._gate_outcome(["some_future_reason"]) == "gate_blocked"


@pytest.mark.parametrize("reasons,expected", [
    (["position_gone"], "position_gone"),
    (["stale_decision"], "stale_decision"),
    (["pass"], "decider_pass"),
    (["unknown_action:frobnicate"], "decider_invalid_action"),
    (["invalid_candidate_key"], "decider_invalid_candidate"),
    (["symbol_position_exists"], "position_open"),
    (["hard_halt:operator"], "halted"),
    (["halt_file"], "halted"),
    (["outside_entry_window"], "outside_entry_window"),
    (["signal_or_snapshot_unavailable"], "signal_unavailable"),
    (["requote_failed"], "requote_failed"),
    (["requote_unusable"], "requote_unusable"),
    (["slippage:0.0700"], "slippage_exceeded"),
    (["no_size"], "no_size"),
    (["jev_veto"], "jev_veto"),
    (["open_position_failed"], "open_position_failed"),
    (["open_position_rejected:broker_error"], "open_position_rejected"),
])
def test_classify_harvest_outcome_reasons(reasons, expected):
    harvest = {"reasons": reasons, "action": None, "outcome": None}
    assert DPOFEngine._classify_harvest_outcome(harvest) == expected


def test_classify_harvest_outcome_entered_via_entry_submitted():
    harvest = {"reasons": [], "outcome": "entry_submitted:P1"}
    assert DPOFEngine._classify_harvest_outcome(harvest) == "entered"


def test_classify_harvest_outcome_entered_via_would_enter():
    harvest = {"reasons": [], "outcome": "would_enter"}
    assert DPOFEngine._classify_harvest_outcome(harvest) == "entered"


def test_classify_harvest_outcome_decider_error_unavailable():
    harvest = {"reasons": ["decider_error:DecisionUnavailable: no providers"], "unavailable": True}
    assert DPOFEngine._classify_harvest_outcome(harvest) == "decider_unavailable"


def test_classify_harvest_outcome_decider_error_not_unavailable():
    harvest = {"reasons": ["decider_error:ValueError: bad json"], "unavailable": False}
    assert DPOFEngine._classify_harvest_outcome(harvest) == "decider_error"


def test_classify_harvest_outcome_action_enter_no_reason_defensive_fallback():
    harvest = {"reasons": [], "action": "enter", "outcome": None}
    assert DPOFEngine._classify_harvest_outcome(harvest) == "decider_enter"


def test_classify_harvest_outcome_action_pass_no_reason_defensive_fallback():
    harvest = {"reasons": [], "action": "pass", "outcome": None}
    assert DPOFEngine._classify_harvest_outcome(harvest) == "decider_pass"


def test_classify_harvest_outcome_totally_unknown_defensive_fallback():
    harvest = {"reasons": [], "action": None, "outcome": None}
    assert DPOFEngine._classify_harvest_outcome(harvest) == "decider_result_unknown"


def test_note_entry_outcome_writes_into_summary():
    summary: dict = {}
    DPOFEngine._note_entry_outcome(summary, "SPY", "cooldown")
    assert summary["entry_outcomes"]["SPY"] == {"outcome": "cooldown", "detail": []}
    DPOFEngine._note_entry_outcome(summary, "QQQ", "halted", detail=["hard_halt:operator"])
    assert summary["entry_outcomes"]["QQQ"] == {"outcome": "halted", "detail": ["hard_halt:operator"]}


# ─────────────────────────────────────────────────────────────────────────
# _fallback_entry_outcome (instance method — needs a real ledger)
# ─────────────────────────────────────────────────────────────────────────

def test_fallback_entry_outcome_position_open(cfg):
    async def _t():
        ledger = DPOFLedger(cfg.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "QQQ", "QQQTESTC00400000"))
        eng = _make_engine(cfg=cfg, ledger=ledger)
        out = eng._fallback_entry_outcome("QQQ", _base_dt(), cfg, RiskState(), halt_present=False)
        assert out == "position_open"

    asyncio.run(_t())


def test_fallback_entry_outcome_deliberation_in_flight(cfg):
    async def _t():
        ledger = DPOFLedger(cfg.data_dir)
        await ledger.load()
        eng = _make_engine(cfg=cfg, ledger=ledger)
        eng._entry_tasks["QQQ"] = asyncio.ensure_future(asyncio.sleep(999))
        try:
            out = eng._fallback_entry_outcome("QQQ", _base_dt(), cfg, RiskState(), halt_present=False)
            assert out == "deliberation_in_flight"
        finally:
            eng._entry_tasks["QQQ"].cancel()

    asyncio.run(_t())


def test_fallback_entry_outcome_halted(cfg):
    async def _t():
        ledger = DPOFLedger(cfg.data_dir)
        await ledger.load()
        eng = _make_engine(cfg=cfg, ledger=ledger)
        out = eng._fallback_entry_outcome("QQQ", _base_dt(), cfg, RiskState(hard_halted=True, halt_reason="op"), halt_present=False)
        assert out == "halted"

    asyncio.run(_t())


def test_fallback_entry_outcome_not_selected_instrument(cfg):
    async def _t():
        ledger = DPOFLedger(cfg.data_dir)
        await ledger.load()
        eng = _make_engine(cfg=cfg, ledger=ledger)
        out = eng._fallback_entry_outcome("QQQ", _base_dt(), cfg, RiskState(), halt_present=False)
        assert out == "not_selected_instrument"

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# Real cycles: fresh-nomination outcomes end-to-end via signals.jsonl
# ─────────────────────────────────────────────────────────────────────────

def _read_signals(data_dir: str) -> list[dict]:
    path = os.path.join(data_dir, "signals.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _outcome_for(data_dir: str, underlying: str) -> str:
    recs = [r for r in _read_signals(data_dir) if r["underlying"] == underlying]
    assert recs, f"no signals.jsonl record for {underlying}"
    return recs[-1]["outcome"]


def _base_paper_cfg(cfg, **overrides):
    return replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",),
        alpaca_key_id="k", alpaca_secret_key="s", jev_mode=JevMode.OFF,
        **overrides,
    )


def test_outcome_data_unhealthy(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY", data_healthy=False, health_issues=["stale_bars"]))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "data_unhealthy"

    asyncio.run(_t())


def test_outcome_composite_unavailable(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "composite_unavailable"

    asyncio.run(_t())


def test_outcome_below_threshold(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)

    def _weak_snapshot(now):
        from agent.dpof.models import MarketSnapshot

        u = UnderlyingSnapshot(
            symbol="SPY",
            pillars={
                "nofa": PillarReading(name="nofa", value=0.05, available=True, source="test"),
                "premium": PillarReading(name="premium", value=0.04, available=True, source="test"),
                "darkpool": PillarReading(name="darkpool", value=0.03, available=True, source="test"),
            },
            last_price=100.5, vwap=100.0, ret_5m=0.001, ret_15m=0.001, bars_count=20,
        )
        return MarketSnapshot(asof=now, underlyings={"SPY": u}, vix=16.0, vix_age_seconds=30.0, data_healthy=True)

    feed = FakeFeed(_weak_snapshot)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "below_threshold"

    asyncio.run(_t())


def test_outcome_regime_requires_strong(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)

    def _choppy_normal_snapshot(now):
        from agent.dpof.models import MarketSnapshot

        # composite lands in the "normal" band (>= threshold_normal, <
        # threshold_strong) while VIX is CHOPPY -> regime_requires_strong.
        u = UnderlyingSnapshot(
            symbol="SPY",
            pillars={
                "nofa": PillarReading(name="nofa", value=0.20, available=True, source="test"),
                "premium": PillarReading(name="premium", value=0.20, available=True, source="test"),
                "darkpool": PillarReading(name="darkpool", value=0.20, available=True, source="test"),
            },
            last_price=100.5, vwap=100.0, ret_5m=0.005, ret_15m=0.01, bars_count=20,
        )
        return MarketSnapshot(asof=now, underlyings={"SPY": u}, vix=28.0, vix_age_seconds=30.0, data_healthy=True)

    feed = FakeFeed(_choppy_normal_snapshot)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "regime_requires_strong"

    asyncio.run(_t())


def test_outcome_price_not_confirmed(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)

    def _unconfirmed_snapshot(now):
        from agent.dpof.models import MarketSnapshot

        # Strong bullish composite, but price below VWAP -> price_not_confirmed veto.
        u = UnderlyingSnapshot(
            symbol="SPY",
            pillars={
                "nofa": PillarReading(name="nofa", value=0.6, available=True, source="test"),
                "premium": PillarReading(name="premium", value=0.5, available=True, source="test"),
                "darkpool": PillarReading(name="darkpool", value=0.5, available=True, source="test"),
            },
            last_price=99.0, vwap=100.0, ret_5m=0.005, ret_15m=0.01, bars_count=20,
        )
        return MarketSnapshot(asof=now, underlyings={"SPY": u}, vix=16.0, vix_age_seconds=30.0, data_healthy=True)

    feed = FakeFeed(_unconfirmed_snapshot)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "price_not_confirmed"

    asyncio.run(_t())


def test_outcome_dp_veto(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)

    def _dp_veto_snapshot(now):
        from agent.dpof.models import MarketSnapshot

        # Bullish composite, darkpool strongly against it -> dp_veto.
        u = UnderlyingSnapshot(
            symbol="SPY",
            pillars={
                "nofa": PillarReading(name="nofa", value=0.6, available=True, source="test"),
                "premium": PillarReading(name="premium", value=0.5, available=True, source="test"),
                "darkpool": PillarReading(name="darkpool", value=-0.9, available=True, source="test"),
            },
            last_price=100.5, vwap=100.0, ret_5m=0.005, ret_15m=0.01, bars_count=20,
        )
        return MarketSnapshot(asof=now, underlyings={"SPY": u}, vix=16.0, vix_age_seconds=30.0, data_healthy=True)

    feed = FakeFeed(_dp_veto_snapshot)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "dp_veto"

    asyncio.run(_t())


def test_outcome_outside_entry_window(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    early = _base_dt().replace(hour=8, minute=0)  # before entry_start_et=09:45

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger,
                            clock=lambda: early.timestamp(), market_open_fn=lambda: True)
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "outside_entry_window"

    asyncio.run(_t())


def test_outcome_halted(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.save_risk(risk_module.operator_halt(ledger.risk, "operator test"))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "halted"

    asyncio.run(_t())


def test_outcome_position_open(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", "SPYTESTC00500000"))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "position_open"

    asyncio.run(_t())


def test_outcome_deliberation_in_flight(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        eng._entry_tasks["SPY"] = asyncio.ensure_future(asyncio.sleep(999))
        try:
            await eng.run_cycle(bot=None)
            assert _outcome_for(c.data_dir, "SPY") == "deliberation_in_flight"
        finally:
            eng._entry_tasks["SPY"].cancel()

    asyncio.run(_t())


def test_outcome_cooldown(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    now_epoch = _base_dt().timestamp()

    # The real composite _bullish_snapshot produces (evaluate_signal, same as
    # the engine computes) -- last_composite must be WITHIN
    # entry_redeliberate_delta of it, or the "composite strengthened" bypass
    # fires and the cooldown never actually blocks anything.
    from agent.dpof.signals import evaluate_signal

    real_signal = evaluate_signal(_bullish_snapshot(now_epoch, "SPY"), c)
    assert real_signal.direction == Direction.BULLISH

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: now_epoch)
        eng._entry_cooldowns["SPY:bullish"] = {
            "next_allowed_at": now_epoch + 10_000, "recheck_at": now_epoch + 10_000,
            "last_composite": real_signal.composite, "set_at": now_epoch,  # unchanged -> never bypassed
        }
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "cooldown"

    asyncio.run(_t())


def test_outcome_insufficient_capital(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, account={"equity": 0.0, "options_trading_level": 3})
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "insufficient_capital"

    asyncio.run(_t())


def test_outcome_decider_unavailable(cfg, monkeypatch):
    import agent.dpof.decider as decider_mod

    def _boom(cfg):
        raise RuntimeError("decider construction unavailable for test")

    monkeypatch.setattr(decider_mod, "DPOFDecider", _boom)

    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        assert eng._decider is None
        await eng.run_cycle(bot=object())
        assert _outcome_for(c.data_dir, "SPY") == "decider_unavailable"

    asyncio.run(_t())


def test_outcome_no_chain(cfg, monkeypatch):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=None)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    decider = FakeDecider()

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "no_chain"

    asyncio.run(_t())


def test_outcome_no_candidates(cfg, monkeypatch):
    today = _base_dt().date()
    chain, _leg = _make_chain("SPY", today)
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    decider = FakeDecider()

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            clock=lambda: _base_dt().timestamp(), candidate_fn=lambda *a, **kw: [],
        )
        await eng.run_cycle(bot=None)
        assert _outcome_for(c.data_dir, "SPY") == "no_candidates"

    asyncio.run(_t())


def test_outcome_deliberating_then_harvest_entered(cfg, monkeypatch):
    """End-to-end wiring check: launch cycle -> 'deliberating'; harvest cycle
    (decider says enter, Jev allows, order opens) -> 'entered', proving
    summary['entry_harvests'][sym] (the SAME decision_log _apply_entry_outcome
    appends to decisions.jsonl) correctly reaches _log_cycle_signals."""
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = _base_paper_cfg(cfg, allow_verticals=False)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="off", called=False, allow=True))
    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=0.8, qty=2))

    async def open_position_paper(mgr, choice, qty, signal, jev, now_epoch, judgment):
        from agent.dpof.models import PositionRecord

        pos = PositionRecord(
            id="P1", underlying=choice.underlying, direction=choice.direction.value, structure=choice.structure,
            legs=[{"symbol": l.symbol, "side": l.side, "ratio_qty": l.ratio_qty, "position_intent": l.position_intent} for l in choice.legs],
            status="pending", qty_original=qty, qty_open=0, mode="paper",
            thesis=judgment["thesis"], invalidation=judgment["invalidation"],
            entry_decision=judgment["entry_decision"], next_review_at=judgment["next_review_at"],
        )
        await mgr.ledger.upsert(pos)
        return pos

    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        orders.open_position_return = open_position_paper
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=object())
        assert _outcome_for(c.data_dir, "SPY") == "deliberating"

        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=object())

        recs = [r for r in _read_signals(c.data_dir) if r["underlying"] == "SPY"]
        last = recs[-1]
        assert last["outcome"] == "entered"
        assert last["decider"]["action"] == "enter"
        assert last["decider"]["conviction"] == pytest.approx(0.8)

        stats = json.load(open(os.path.join(c.data_dir, "signal_day_stats.json")))
        assert stats["entries"] == 1
        assert stats["decider_enter"] == 1

    asyncio.run(_t())


def test_outcome_decider_pass_then_cooldown(cfg, monkeypatch):
    today = _base_dt().date()
    chain, _leg = _make_chain("SPY", today)
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    decider = FakeDecider(entry_result=FakeEntryDecision(action="pass", recheck_minutes=10))
    clock_box = {"t": _base_dt().timestamp()}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        recs = [r for r in _read_signals(c.data_dir) if r["underlying"] == "SPY"]
        assert recs[-1]["outcome"] == "decider_pass"
        assert recs[-1]["decider"]["action"] == "pass"

    asyncio.run(_t())


def test_signal_record_shape_has_all_pillars_and_flow_source(cfg):
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        recs = _read_signals(c.data_dir)
        assert len(recs) == 1
        rec = recs[0]
        for key in ("ts", "underlying", "composite", "strength", "direction", "pillars", "flow_source",
                    "regime", "price_confirm", "vetoes", "reasons", "is_signal_symbol", "nominated", "outcome"):
            assert key in rec
        assert set(rec["pillars"].keys()) == {"nofa", "premium", "darkpool", "gex"}
        assert rec["flow_source"] == "uw"  # test="uw:..."-less source in _bullish_underlying -> not selfflow-prefixed
        assert rec["is_signal_symbol"] is True

    asyncio.run(_t())


def test_two_underlyings_only_signal_symbol_gets_full_detail(cfg):
    """cfg.underlyings=(SPY, QQQ): only ONE is ever evaluate_signal's chosen
    instrument per cycle (select_instrument) -- the other gets a
    not_selected_instrument record with its own (real) composite/strength
    context, never a synthesized entry-pipeline verdict."""
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY", "QQQ"), alpaca_key_id="k", alpaca_secret_key="s")
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)

    def _two_symbol_snapshot(now):
        from agent.dpof.models import MarketSnapshot

        return MarketSnapshot(
            asof=now, underlyings={"SPY": _bullish_underlying("SPY"), "QQQ": _bullish_underlying("QQQ")},
            vix=16.0, vix_age_seconds=30.0, data_healthy=True,
        )

    feed = FakeFeed(_two_symbol_snapshot)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        await eng.run_cycle(bot=None)
        recs = {r["underlying"]: r for r in _read_signals(c.data_dir)}
        assert set(recs.keys()) == {"SPY", "QQQ"}
        signal_syms = [sym for sym, r in recs.items() if r["is_signal_symbol"]]
        assert len(signal_syms) == 1
        other = "QQQ" if signal_syms[0] == "SPY" else "SPY"
        assert recs[other]["outcome"] == "not_selected_instrument"
        assert recs[other]["is_signal_symbol"] is False
        # Context (composite/pillars) is still real, not blanked out.
        assert recs[other]["composite"] is not None

    asyncio.run(_t())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


def test_cycle_timeout_in_observability_never_skips_save_risk(cfg):
    """The observability step runs after the risk persist: a cycle-timeout
    cancellation landing while it is in flight (production wraps every cycle
    in asyncio.wait_for) must not skip the state-of-record write."""
    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: _base_dt().timestamp())
        calls = {"save_risk": 0}
        real_save = ledger.save_risk

        async def _spy_save(state):
            calls["save_risk"] += 1
            return await real_save(state)

        ledger.save_risk = _spy_save
        entered = asyncio.Event()

        async def _hang(*_a, **_kw):
            entered.set()
            await asyncio.sleep(3600)

        eng._log_cycle_signals = _hang
        task = asyncio.create_task(eng.run_cycle(bot=None))
        await asyncio.wait_for(entered.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls["save_risk"] == 1

    asyncio.run(_t())
