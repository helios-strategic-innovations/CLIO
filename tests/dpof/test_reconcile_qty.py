"""Tests for the reconcile() quantity check + single-leg synthetic-fill
correction (docs/design/clio_dpof_followups_2026-09-26.md §3).

Reuses the FakeBroker/_open_pos/_paper_cfg/_fresh_ledger helpers from
test_orders.py (same fake-broker patterns, same conftest.py fixtures --
hermetic, no network, no data/ writes).
"""

import asyncio
from dataclasses import replace

import pytest

from agent.dpof.config import DpofMode
from agent.dpof.models import PositionRecord
from agent.dpof.orders import OrderManager, _batch_realized_pnl, recompute_from_fills
from tests.dpof.test_orders import FakeBroker, _fresh_ledger, _open_pos, _paper_cfg


SYMBOL = "SPY261002C00575000"
SYMBOL2 = "SPY261002C00580000"


def _single_leg_open(qty=1, entry_price=1.00, **overrides) -> PositionRecord:
    return _open_pos(qty=qty, entry_price=entry_price, **overrides)


# ---------------------------------------------------------------------------
# recompute_from_fills: synthetic fills widen qty_open, never entry_price/pnl
# ---------------------------------------------------------------------------


def test_recompute_counts_reconcile_add_toward_qty_open_but_not_entry_price():
    pos = _single_leg_open(qty=1, entry_price=1.00)
    pos.order_fills["reconcile:1000:SPY"] = {
        "purpose": "reconcile_add", "filled_qty": 2, "avg": None, "status": "reconciled",
    }
    got = recompute_from_fills(pos)
    assert got.qty_open == 3  # 1 real + 2 synthetic
    assert got.entry_price == pytest.approx(1.00)  # unaffected by the synthetic fill
    assert got.realized_pnl == 0.0


def test_recompute_counts_reconcile_remove_toward_exited_but_not_realized_pnl():
    pos = _single_leg_open(qty=3, entry_price=1.00)
    pos.order_fills["reconcile:1000:SPY"] = {
        "purpose": "reconcile_remove", "filled_qty": 1, "avg": None, "status": "reconciled",
    }
    got = recompute_from_fills(pos)
    assert got.qty_open == 2  # 3 - 1 synthetic removal
    assert got.entry_price == pytest.approx(1.00)
    assert got.realized_pnl == 0.0  # synthetic removal never contributes P&L


def test_recompute_keeps_entry_price_from_real_fills_alongside_synthetic_add():
    pos = _single_leg_open(qty=0, entry_price=None, order_fills={})
    pos.order_fills["c1"] = {"purpose": "entry", "filled_qty": 2, "avg": 1.20, "status": "filled"}
    pos.order_fills["reconcile:1000:SPY"] = {
        "purpose": "reconcile_add", "filled_qty": 1, "avg": None, "status": "reconciled",
    }
    got = recompute_from_fills(pos)
    assert got.qty_open == 3
    assert got.entry_price == pytest.approx(1.20)  # weighted avg over REAL entry fills only


def test_recompute_after_reconcile_add_leaves_exit_batch_math_unchanged():
    """A synthetic reconcile_add fill widens qty_open, but a subsequent
    partial exit's batch accounting (_batch_realized_pnl /
    _exit_fills_in_current_batch) must be exactly as if the synthetic fill
    never happened -- both stay purpose=='exit'-only."""
    pos = _single_leg_open(qty=1, entry_price=1.00)
    pos.order_fills["reconcile:1000:SPY"] = {
        "purpose": "reconcile_add", "filled_qty": 2, "avg": None, "status": "reconciled",
    }
    pos = recompute_from_fills(pos)
    assert pos.qty_open == 3

    pos.exit_client_order_ids = ["x1"]
    pos.exit_filled_baseline = 0
    pos.order_fills["x1"] = {"purpose": "exit", "filled_qty": 3, "avg": 1.30, "status": "filled"}
    pos = recompute_from_fills(pos)
    assert pos.qty_open == 0
    batch_pnl = _batch_realized_pnl(pos)
    assert batch_pnl == pytest.approx((1.30 - 1.00) * 100 * 3)  # only the real exit fill counted
    assert pos.realized_pnl == pytest.approx((1.30 - 1.00) * 100 * 3)


# ---------------------------------------------------------------------------
# reconcile(): eligibility, confirm timing, surplus/deficit adoption
# ---------------------------------------------------------------------------


def test_first_sighting_of_mismatch_changes_nothing_but_warns(cfg):
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]

        warnings = await om.reconcile(now_epoch=1000.0)
        assert any(w.startswith(f"qty_mismatch:{SYMBOL}:ledger=1:broker=3") for w in warnings)
        assert not any(w.startswith("qty_adopted:") for w in warnings)
        got = ledger.get(pos.id)
        assert got.qty_open == 1  # unchanged -- first sighting only starts the clock
        assert om.has_pending_qty_mismatch() is True

    asyncio.run(_t())


def test_surplus_adopted_after_confirm_window_elapses(cfg):
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]

        await om.reconcile(now_epoch=1000.0)  # first sighting
        assert ledger.get(pos.id).qty_open == 1

        # Same (ledger,broker) pair, but not yet 30s later -- must still wait.
        warnings_early = await om.reconcile(now_epoch=1015.0)
        assert not any(w.startswith("qty_adopted:") for w in warnings_early)
        assert ledger.get(pos.id).qty_open == 1

        # Now >= 30s since first_seen -- adopts.
        warnings = await om.reconcile(now_epoch=1031.0)
        assert f"qty_adopted:{pos.id}:{SYMBOL}:1->3" in warnings
        got = ledger.get(pos.id)
        assert got.qty_open == 3
        assert got.entry_price == pytest.approx(1.00)  # untouched -- synthetic fill has avg=None
        assert got.qty_original == 1  # never touched
        assert got.status == "open"  # never touched
        assert got.exit_reason == ""  # never touched
        assert broker.submitted == [] and broker.cancelled == []  # never places/cancels an order
        assert om.has_pending_qty_mismatch() is False

        synth = [f for f in got.order_fills.values() if f.get("purpose") == "reconcile_add"]
        assert len(synth) == 1 and synth[0]["filled_qty"] == 2 and synth[0]["avg"] is None

    asyncio.run(_t())


def test_deficit_adopted_after_confirm_window_elapses(cfg):
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=3, entry_price=1.00)
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 1, "avg_entry_price": 1.0, "market_value": 100.0, "unrealized_pl": 0.0}
        ]

        await om.reconcile(now_epoch=1000.0)
        warnings = await om.reconcile(now_epoch=1031.0)
        assert f"qty_adopted:{pos.id}:{SYMBOL}:3->1" in warnings
        got = ledger.get(pos.id)
        assert got.qty_open == 1
        synth = [f for f in got.order_fills.values() if f.get("purpose") == "reconcile_remove"]
        assert len(synth) == 1 and synth[0]["filled_qty"] == 2 and synth[0]["avg"] is None
        assert got.realized_pnl == 0.0  # synthetic removal never invents P&L

    asyncio.run(_t())


def test_surplus_adopt_lets_close_position_use_new_qty_open(cfg):
    """After a surplus adopt, close_position(qty=new qty_open) is accepted."""
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)
        broker.set_quote(SYMBOL, bid=1.10, ask=1.20)

        pos = _single_leg_open(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]
        await om.reconcile(now_epoch=1000.0)
        await om.reconcile(now_epoch=1031.0)
        pos_now = ledger.get(pos.id)
        assert pos_now.qty_open == 3

        ok = await om.close_position(pos_now, qty=3, reason="take_profit", urgent=False, now_epoch=1040.0)
        assert ok is True
        assert ledger.get(pos.id).status == "closing"

    asyncio.run(_t())


def test_changed_quantities_reset_the_confirm_clock(cfg):
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]
        await om.reconcile(now_epoch=1000.0)  # first_seen=1000, (l=1,b=3)

        # Broker reading changes before the window elapses -- clock restarts.
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 4, "avg_entry_price": 1.0, "market_value": 400.0, "unrealized_pl": 0.0}
        ]
        await om.reconcile(now_epoch=1020.0)  # (l=1,b=4) now -- restarts at 1020

        # 30s past the FIRST sighting (1000) but only 11s past the restarted one.
        warnings = await om.reconcile(now_epoch=1031.0)
        assert not any(w.startswith("qty_adopted:") for w in warnings)
        assert ledger.get(pos.id).qty_open == 1

        # 30s past the restarted clock (1020 + 30 = 1050) -- now adopts to 4.
        warnings2 = await om.reconcile(now_epoch=1051.0)
        assert f"qty_adopted:{pos.id}:{SYMBOL}:1->4" in warnings2
        assert ledger.get(pos.id).qty_open == 4

    asyncio.run(_t())


def test_open_position_with_live_recorded_order_warns_but_never_adopts(cfg):
    """An "open" position always gets the warning (never silent); a chain
    order whose RECORDED status is not terminal blocks only the correction."""
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=1, entry_price=1.00)
        pos.entry_client_order_ids = list(pos.entry_client_order_ids) + ["DPOF-EN-live"]
        pos.order_fills["DPOF-EN-live"] = {"purpose": "entry", "filled_qty": 0, "avg": None, "status": "new"}
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]

        w1 = await om.reconcile(now_epoch=1000.0)
        assert w1 == [f"qty_mismatch:{SYMBOL}:ledger=1:broker=3"]
        w2 = await om.reconcile(now_epoch=1100.0)
        assert w2 == [f"qty_mismatch:{SYMBOL}:ledger=1:broker=3"]
        assert ledger.get(pos.id).qty_open == 1
        assert om.has_pending_qty_mismatch() is False

    asyncio.run(_t())


def test_abandoned_chain_order_never_blocks_the_check(cfg):
    """A chain cid with NO order_fills entry never reached the broker (the
    manager resends only after a 404) -- it must not silence or block the
    quantity check forever (pass-2 review SHOULD-FIX #1)."""
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=1, entry_price=1.00)
        pos.entry_client_order_ids = ["DPOF-EN-nofill"] + list(pos.entry_client_order_ids) + ["DPOF-EN-nofill-2"]
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]

        w1 = await om.reconcile(now_epoch=1000.0)
        assert w1 == [f"qty_mismatch:{SYMBOL}:ledger=1:broker=3"]
        w2 = await om.reconcile(now_epoch=1031.0)
        assert f"qty_adopted:{pos.id}:{SYMBOL}:1->3" in w2
        assert ledger.get(pos.id).qty_open == 3

    asyncio.run(_t())


def test_no_adopt_when_status_closing_no_warning(cfg):
    """§3.2 point 2: status != "open" (e.g. "closing") is also a SKIP (no
    warning) condition -- distinct from the adopt-eligibility-only gates
    below (two positions/debit vertical/sell leg/negative qty/knob off),
    which DO still emit the qty_mismatch warning."""
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(
            qty=1, entry_price=1.00, status="closing", exit_target_qty=1, exit_filled_baseline=0,
            exit_client_order_ids=["x1"],
            order_fills={
                "DPOF-EN-seed": {"purpose": "entry", "filled_qty": 1, "avg": 1.00, "status": "filled"},
                "x1": {"purpose": "exit", "filled_qty": 0, "avg": None, "status": "filled"},
            },
        )
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]

        w1 = await om.reconcile(now_epoch=1000.0)
        assert w1 == []  # skipped -- status != "open"
        w2 = await om.reconcile(now_epoch=1100.0)
        assert w2 == []
        assert ledger.get(pos.id).qty_open == 1

    asyncio.run(_t())


def test_no_adopt_when_cancel_requested(cfg):
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=1, entry_price=1.00, cancel_requested=True)
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]

        w1 = await om.reconcile(now_epoch=1000.0)
        assert w1 == []  # cancel_requested -> ineligible -> no warning either
        w2 = await om.reconcile(now_epoch=1100.0)
        assert w2 == []

    asyncio.run(_t())


def test_no_adopt_with_two_positions_on_one_symbol_warning_still_emitted(cfg):
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos1 = _single_leg_open(qty=1, entry_price=1.00, id="p1")
        pos2 = _single_leg_open(qty=1, entry_price=1.10, id="p2", entry_client_order_id="DPOF-EN-seed2",
                                 entry_client_order_ids=["DPOF-EN-seed2"],
                                 order_fills={"DPOF-EN-seed2": {"purpose": "entry", "filled_qty": 1, "avg": 1.10, "status": "filled"}})
        await ledger.upsert(pos1)
        await ledger.upsert(pos2)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 5, "avg_entry_price": 1.0, "market_value": 500.0, "unrealized_pl": 0.0}
        ]

        w1 = await om.reconcile(now_epoch=1000.0)
        assert any(w.startswith(f"qty_mismatch:{SYMBOL}:ledger=2:broker=5") for w in w1)
        w2 = await om.reconcile(now_epoch=1100.0)
        assert not any(w.startswith("qty_adopted:") for w in w2)  # two positions hold the symbol -- never adopted
        assert ledger.get("p1").qty_open == 1 and ledger.get("p2").qty_open == 1

    asyncio.run(_t())


def test_no_adopt_for_debit_vertical_warning_still_emitted(cfg):
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        legs = [
            {"symbol": SYMBOL, "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"},
            {"symbol": SYMBOL2, "side": "sell", "ratio_qty": 1, "position_intent": "sell_to_open"},
        ]
        pos = _single_leg_open(qty=1, entry_price=1.50, structure="debit_vertical", legs=legs)
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]

        w1 = await om.reconcile(now_epoch=1000.0)
        assert any(w.startswith(f"qty_mismatch:{SYMBOL}:ledger=1:broker=3") for w in w1)
        w2 = await om.reconcile(now_epoch=1100.0)
        assert not any(w.startswith("qty_adopted:") for w in w2)  # debit_vertical -- never adopted

    asyncio.run(_t())


def test_no_adopt_for_sell_leg_warning_still_emitted(cfg):
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        legs = [{"symbol": SYMBOL, "side": "sell", "ratio_qty": 1, "position_intent": "sell_to_open"}]
        pos = _single_leg_open(qty=1, entry_price=1.00, legs=legs)
        # qty_open positive with a sell leg -> ledger_qty is NEGATIVE (-1); broker
        # reports a positive qty -- definitely a mismatch on a sell-side single leg.
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]

        w1 = await om.reconcile(now_epoch=1000.0)
        assert any(w.startswith(f"qty_mismatch:{SYMBOL}:ledger=-1:broker=3") for w in w1)
        w2 = await om.reconcile(now_epoch=1100.0)
        assert not any(w.startswith("qty_adopted:") for w in w2)  # side=="sell" -- never adopted

    asyncio.run(_t())


def test_no_adopt_for_negative_broker_qty_warning_still_emitted(cfg):
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        # Broker reports a NEGATIVE qty for this symbol (a short position at
        # the broker) -- our single-leg long ledger position disagrees.
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": -2, "avg_entry_price": 1.0, "market_value": -200.0, "unrealized_pl": 0.0}
        ]

        w1 = await om.reconcile(now_epoch=1000.0)
        assert any(w.startswith(f"qty_mismatch:{SYMBOL}:ledger=1:broker=-2") for w in w1)
        w2 = await om.reconcile(now_epoch=1100.0)
        assert not any(w.startswith("qty_adopted:") for w in w2)  # broker_qty <= 0 -- never adopted

    asyncio.run(_t())


def test_no_adopt_when_knob_off_warning_still_emitted(cfg):
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0, reconcile_adopt_qty=False)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]

        w1 = await om.reconcile(now_epoch=1000.0)
        assert any(w.startswith(f"qty_mismatch:{SYMBOL}:ledger=1:broker=3") for w in w1)
        w2 = await om.reconcile(now_epoch=1100.0)
        assert not any(w.startswith("qty_adopted:") for w in w2)  # knob off -- never adopted
        assert ledger.get(pos.id).qty_open == 1

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# reconciled_absent / orphan / broker-unavailable behavior stays unchanged
# ---------------------------------------------------------------------------


def test_reconciled_absent_still_works_with_now_epoch_param(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker._option_positions = []

        warnings = await om.reconcile(now_epoch=1000.0)
        assert any(w.startswith("reconciled_absent:") for w in warnings)
        got = ledger.get(pos.id)
        assert got.status == "closed"
        assert got.exit_reason == "reconciled_absent"
        # New quantity-check logic never fires for a symbol with no broker presence.
        assert not any(w.startswith("qty_mismatch:") or w.startswith("qty_adopted:") for w in warnings)

    asyncio.run(_t())


def test_orphan_still_reported_without_quantity_check_interference(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 1, "avg_entry_price": 1.0, "market_value": 100.0, "unrealized_pl": 0.0}
        ]

        warnings = await om.reconcile(now_epoch=1000.0)
        assert "orphan:SPY261002C00575000" in warnings
        assert not any(w.startswith("qty_mismatch:") or w.startswith("qty_adopted:") for w in warnings)
        assert broker.submitted == [] and broker.cancelled == []

    asyncio.run(_t())


def test_broker_unavailable_still_short_circuits_before_quantity_check(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        class UnavailableBroker(FakeBroker):
            async def list_option_positions(self):
                return None

        broker = UnavailableBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _single_leg_open(qty=1, entry_price=1.00)
        await ledger.upsert(pos)

        warnings = await om.reconcile(now_epoch=1000.0)
        assert warnings == ["reconcile_broker_unavailable"]
        assert om.has_pending_qty_mismatch() is False

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Engine wiring: reconcile runs on a non-tenth cycle while a mismatch pends
# ---------------------------------------------------------------------------


def test_engine_runs_reconcile_off_cycle_while_mismatch_pending(cfg):
    """Directly exercises OrderManager.has_pending_qty_mismatch()'s intended
    consumer condition (engine.py step 8c: `is_tenth_cycle or
    self._orders.has_pending_qty_mismatch()`) without standing up the full
    engine -- confirms the flag flips on first sighting and clears once
    adopted, which is exactly what unlocks the off-cycle reconcile call."""
    paper_cfg = _paper_cfg(cfg, reconcile_confirm_seconds=30.0)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)
        assert om.has_pending_qty_mismatch() is False

        pos = _single_leg_open(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker._option_positions = [
            {"symbol": SYMBOL, "qty": 3, "avg_entry_price": 1.0, "market_value": 300.0, "unrealized_pl": 0.0}
        ]

        await om.reconcile(now_epoch=1000.0)
        is_tenth_cycle = False
        assert is_tenth_cycle or om.has_pending_qty_mismatch()  # step 8c's condition would fire off-cycle

        await om.reconcile(now_epoch=1031.0)  # adopts
        assert om.has_pending_qty_mismatch() is False
        assert not (is_tenth_cycle or om.has_pending_qty_mismatch())  # condition now false off-cycle

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Config (§3.4): dataclass default == load_dpof_config default; validate()
# ---------------------------------------------------------------------------


def test_config_dataclass_default_agrees_with_from_env_default(monkeypatch):
    from agent.dpof.config import DPOFConfig, load_dpof_config

    monkeypatch.delenv("CLIO_DPOF_RECONCILE_ADOPT", raising=False)
    monkeypatch.delenv("CLIO_DPOF_RECONCILE_CONFIRM_SECONDS", raising=False)
    d = DPOFConfig()
    env_cfg = load_dpof_config()
    assert env_cfg.reconcile_adopt_qty == d.reconcile_adopt_qty is True
    assert env_cfg.reconcile_confirm_seconds == d.reconcile_confirm_seconds == 30.0


def test_config_env_overrides_reconcile_knobs(monkeypatch):
    from agent.dpof.config import load_dpof_config

    monkeypatch.setenv("CLIO_DPOF_RECONCILE_ADOPT", "false")
    monkeypatch.setenv("CLIO_DPOF_RECONCILE_CONFIRM_SECONDS", "45")
    c = load_dpof_config()
    assert c.reconcile_adopt_qty is False
    assert c.reconcile_confirm_seconds == pytest.approx(45.0)


def test_config_validate_rejects_negative_confirm_seconds(cfg):
    bad = replace(cfg, reconcile_confirm_seconds=-1.0)
    problems = bad.validate()
    assert any("reconcile_confirm_seconds" in p for p in problems)


def test_config_validate_accepts_zero_confirm_seconds(cfg):
    ok = replace(cfg, reconcile_confirm_seconds=0.0)
    problems = ok.validate()
    assert not any("reconcile_confirm_seconds" in p for p in problems)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
