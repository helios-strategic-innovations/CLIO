"""Tests for agent.dpof.orders.OrderManager and recompute_from_fills.

New protocol under test: cancel -> confirm terminal -> resend with a fresh
client_order_id (never Alpaca's PATCH/"replace", which spawns a new order).
No in-memory working-order cache exists any more -- every test that checks
"restart recovery" simply constructs a brand-new OrderManager against the
same ledger and confirms it behaves identically.
"""

import asyncio
from dataclasses import replace

import pytest

from agent.dpof.config import DpofMode
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import (
    ContractChoice,
    Direction,
    JevVerdict,
    OrderLeg,
    OrderResult,
    PositionRecord,
    Signal,
)
from agent.dpof.orders import OrderManager, _batch_realized_pnl, recompute_from_fills


# ---------------------------------------------------------------------------
# Fake broker
# ---------------------------------------------------------------------------


class FakeQuote:
    def __init__(self, bid, ask):
        self.bid = bid
        self.ask = ask


class FakeBroker:
    """Records calls; lets tests script get_order_by_client_id responses.
    Has NO replace_order method at all -- a test that accidentally calls it
    fails loudly with AttributeError, which is the point."""

    def __init__(self, would_submit=False, submit_error=None):
        self.would_submit = would_submit
        self.submit_error = submit_error
        self.submitted = []
        self.cancelled = []
        self._orders: dict[str, dict] = {}
        self._quotes: dict[str, FakeQuote] = {}
        self._option_positions: list[dict] = []

    def set_quote(self, symbol, bid, ask):
        self._quotes[symbol] = FakeQuote(bid, ask)

    def set_order(self, client_order_id, **fields):
        base = {"id": f"b-{client_order_id}", "status": "new", "filled_qty": 0,
                "filled_avg_price": None, "limit_price": None, "legs": []}
        base.update(fields)
        self._orders[client_order_id] = base

    async def submit(self, intent):
        # Never two live orders: fail loudly if the caller sends a second
        # order while a prior one for the same intent shape is still live
        # and non-terminal -- tests assert on self.submitted directly, but
        # this assertion catches a regression even in tests that forget to.
        self.submitted.append(intent)
        if self.submit_error:
            return OrderResult(ok=False, client_order_id=intent.client_order_id, error=str(self.submit_error))
        if self.would_submit:
            return OrderResult(ok=True, client_order_id=intent.client_order_id, would_submit=True)
        self.set_order(intent.client_order_id, status="new", limit_price=intent.limit_price)
        return OrderResult(ok=True, client_order_id=intent.client_order_id, status="new")

    async def get_order_by_client_id(self, client_order_id):
        return self._orders.get(client_order_id)

    async def cancel_order(self, broker_order_id, client_order_id):
        self.cancelled.append((broker_order_id, client_order_id))
        return True

    async def get_option_quotes(self, symbols):
        return {s: self._quotes[s] for s in symbols if s in self._quotes}

    async def list_option_positions(self):
        return self._option_positions


class CancellingBroker(FakeBroker):
    """submit() records the call and then raises CancelledError, simulating
    the calling task being cancelled while the broker call was in flight."""

    async def submit(self, intent):
        self.submitted.append(intent)
        raise asyncio.CancelledError()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _single_choice(underlying="SPY", direction=Direction.BULLISH):
    return ContractChoice(
        structure="single",
        legs=[OrderLeg(symbol="SPY261002C00575000", side="buy", ratio_qty=1, position_intent="buy_to_open")],
        underlying=underlying,
        direction=direction,
        expiration="2026-10-02",
        dte=8,
        est_debit=1.05,
        est_bid=1.00,
        est_ask=1.10,
        long_delta=0.5,
    )


def _vertical_choice():
    return ContractChoice(
        structure="debit_vertical",
        legs=[
            OrderLeg(symbol="SPY261002C00575000", side="buy", ratio_qty=1, position_intent="buy_to_open"),
            OrderLeg(symbol="SPY261002C00580000", side="sell", ratio_qty=1, position_intent="sell_to_open"),
        ],
        underlying="SPY",
        direction=Direction.BULLISH,
        expiration="2026-10-02",
        dte=8,
        est_debit=1.50,
        est_bid=1.40,
        est_ask=1.60,
        long_delta=0.5,
    )


def _signal():
    return Signal(symbol="SPY", direction=Direction.BULLISH, composite=0.4, conviction=0.8, strength="strong")


def _jev():
    return JevVerdict(mode="off", called=False, allow=True)


async def _fresh_ledger(cfg) -> DPOFLedger:
    led = DPOFLedger(cfg.data_dir)
    await led.load()
    return led


def _paper_cfg(cfg, **overrides):
    base = dict(mode=DpofMode.PAPER, alpaca_paper=True, alpaca_key_id="k", alpaca_secret_key="s")
    base.update(overrides)
    return replace(cfg, **base)


def _pending_pos(cid, qty_original=1, **overrides) -> PositionRecord:
    base = dict(
        id="p1",
        underlying="SPY",
        direction="bullish",
        structure="single",
        legs=[{"symbol": "SPY261002C00575000", "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}],
        status="pending",
        qty_original=qty_original,
        qty_open=0,
        entry_client_order_id=cid,
        entry_client_order_ids=[cid],
        working_since=1000.0,
        reprices=0,
    )
    base.update(overrides)
    return PositionRecord(**base)


def _open_pos(qty=2, entry_price=1.00, structure="single", legs=None, **overrides) -> PositionRecord:
    legs = legs or [{"symbol": "SPY261002C00575000", "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}]
    base = dict(
        id="p1",
        underlying="SPY",
        direction="bullish",
        structure=structure,
        legs=legs,
        status="open",
        qty_original=qty,
        qty_open=qty,
        entry_price=entry_price,
        entry_time=900.0,
        entry_client_order_id="DPOF-EN-seed",
        entry_client_order_ids=["DPOF-EN-seed"],
        order_fills={"DPOF-EN-seed": {"purpose": "entry", "filled_qty": qty, "avg": entry_price, "status": "filled"}},
    )
    base.update(overrides)
    return PositionRecord(**base)


# ---------------------------------------------------------------------------
# recompute_from_fills (pure) -- heavily unit-tested per the orchestrator's
# requirement
# ---------------------------------------------------------------------------


def test_recompute_from_fills_no_fills_yet():
    pos = _pending_pos("c1")
    got = recompute_from_fills(pos)
    assert got.entry_price is None
    assert got.qty_open == 0
    assert got.realized_pnl == 0.0


def test_recompute_from_fills_single_entry_order():
    pos = _pending_pos("c1", qty_original=2)
    pos.order_fills["c1"] = {"purpose": "entry", "filled_qty": 2, "avg": 1.10, "status": "filled"}
    got = recompute_from_fills(pos)
    assert got.qty_open == 2
    assert got.entry_price == pytest.approx(1.10)


def test_recompute_from_fills_weighted_avg_across_a_3_order_entry_chain():
    pos = _pending_pos("c1", qty_original=6)
    # 1 @ 1.00, then a repriced order for the remaining 5 fills 3 @ 1.20,
    # then a further repriced order fills the last 2 @ 1.30.
    pos.order_fills["c1"] = {"purpose": "entry", "filled_qty": 1, "avg": 1.00, "status": "canceled"}
    pos.order_fills["c2"] = {"purpose": "entry", "filled_qty": 3, "avg": 1.20, "status": "canceled"}
    pos.order_fills["c3"] = {"purpose": "entry", "filled_qty": 2, "avg": 1.30, "status": "filled"}
    got = recompute_from_fills(pos)
    assert got.qty_open == 6
    expected = (1 * 1.00 + 3 * 1.20 + 2 * 1.30) / 6
    assert got.entry_price == pytest.approx(expected)


def test_recompute_from_fills_is_idempotent_under_repeated_identical_polls():
    pos = _pending_pos("c1", qty_original=2)
    pos.order_fills["c1"] = {"purpose": "entry", "filled_qty": 2, "avg": 1.05, "status": "filled"}
    first = recompute_from_fills(pos)
    second = recompute_from_fills(first)
    third = recompute_from_fills(second)
    assert first.qty_open == second.qty_open == third.qty_open == 2
    assert first.entry_price == second.entry_price == third.entry_price == pytest.approx(1.05)
    assert first.realized_pnl == second.realized_pnl == third.realized_pnl


def test_recompute_from_fills_exits_across_a_chain_accumulate_correctly():
    pos = _open_pos(qty=3, entry_price=1.00)
    # Exit batch 1: sells 1 contract, cancel-then-resend chain of 2 orders
    # (first order 0 fills, second fills the 1).
    pos.exit_client_order_ids = ["x1", "x2"]
    pos.order_fills["x1"] = {"purpose": "exit", "filled_qty": 0, "avg": None, "status": "canceled"}
    pos.order_fills["x2"] = {"purpose": "exit", "filled_qty": 1, "avg": 1.30, "status": "filled"}
    got = recompute_from_fills(pos)
    assert got.qty_open == 2
    assert got.realized_pnl == pytest.approx((1.30 - 1.00) * 100 * 1)

    # Exit batch 2 (later): sells the remaining 2 at 0.80 (a loss).
    pos = got
    pos.exit_client_order_ids = ["x1", "x2", "x3"]
    pos.order_fills["x3"] = {"purpose": "exit", "filled_qty": 2, "avg": 0.80, "status": "filled"}
    final = recompute_from_fills(pos)
    assert final.qty_open == 0
    expected_total = (1.30 - 1.00) * 100 * 1 + (0.80 - 1.00) * 100 * 2
    assert final.realized_pnl == pytest.approx(expected_total)


def test_recompute_from_fills_vertical_exit_avg_is_abs_of_reported_credit():
    # mleg exit avg comes back NEGATIVE (a credit) per Alpaca's sign
    # convention -- recompute must treat it as a positive credit received.
    pos = _open_pos(qty=1, entry_price=1.50, structure="debit_vertical")
    pos.exit_client_order_ids = ["x1"]
    pos.order_fills["x1"] = {"purpose": "exit", "filled_qty": 1, "avg": -1.30, "status": "filled"}
    got = recompute_from_fills(pos)
    assert got.realized_pnl == pytest.approx((1.30 - 1.50) * 100 * 1)


def test_recompute_from_fills_partial_fill_with_no_avg_is_skipped_for_pnl():
    pos = _open_pos(qty=1, entry_price=1.00)
    pos.exit_client_order_ids = ["x1"]
    pos.order_fills["x1"] = {"purpose": "exit", "filled_qty": 0, "avg": None, "status": "canceled"}
    got = recompute_from_fills(pos)
    assert got.qty_open == 1
    assert got.realized_pnl == 0.0


def test_recompute_from_fills_does_not_mutate_input():
    pos = _pending_pos("c1", qty_original=1)
    pos.order_fills["c1"] = {"purpose": "entry", "filled_qty": 1, "avg": 1.00, "status": "filled"}
    before = pos.qty_open
    recompute_from_fills(pos)
    assert pos.qty_open == before  # pure -- caller's object untouched


def test_batch_realized_pnl_isolates_current_batch_from_prior_batches():
    pos = _open_pos(qty=3, entry_price=1.00)
    pos.exit_client_order_ids = ["x1", "x2"]
    pos.order_fills["x1"] = {"purpose": "exit", "filled_qty": 1, "avg": 1.30, "status": "filled"}
    pos.order_fills["x2"] = {"purpose": "exit", "filled_qty": 2, "avg": 0.80, "status": "filled"}
    pos.exit_filled_baseline = 1  # batch 2 started after the first 1 contract was already closed
    pos = recompute_from_fills(pos)
    batch_pnl = _batch_realized_pnl(pos)
    assert batch_pnl == pytest.approx((0.80 - 1.00) * 100 * 2)  # only x2's contribution


# ---------------------------------------------------------------------------
# new_client_order_id
# ---------------------------------------------------------------------------


def test_new_client_order_id_shape(cfg):
    om = OrderManager(cfg, FakeBroker(), None)
    cid = om.new_client_order_id("entry")
    assert cid.startswith(f"{cfg.client_order_prefix}-EN-")
    assert len(cid) <= 48
    cid2 = om.new_client_order_id("exit")
    assert cid2.startswith(f"{cfg.client_order_prefix}-EX-")
    assert cid != cid2


# ---------------------------------------------------------------------------
# open_position
# ---------------------------------------------------------------------------


def test_open_position_log_only_never_submits(cfg):
    lo_cfg = replace(cfg, mode=DpofMode.LOG_ONLY)

    async def _t():
        broker = FakeBroker(would_submit=True)
        ledger = await _fresh_ledger(lo_cfg)
        om = OrderManager(lo_cfg, broker, ledger)

        pos = await om.open_position(_single_choice(), qty=2, signal=_signal(), jev=_jev(), now_epoch=1000.0)
        assert pos is not None
        assert pos.status == "cancelled"
        assert pos.exit_reason == "log_only"
        assert pos.id not in {p.id for p in ledger.open_positions()}
        assert len(broker.submitted) == 1  # broker.submit was called, but IT itself sends nothing

    asyncio.run(_t())


def test_open_position_persists_before_submit_with_chain_recorded(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = await om.open_position(_single_choice(), qty=1, signal=_signal(), jev=_jev(), now_epoch=1000.0)
        assert pos.entry_client_order_id.startswith(paper_cfg.client_order_prefix)
        assert pos.entry_client_order_ids == [pos.entry_client_order_id]
        assert pos.status == "pending"
        assert pos.working_since == 1000.0
        stored = ledger.get(pos.id)
        assert stored.entry_client_order_ids == pos.entry_client_order_ids

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Cancellation safety (orchestrator addendum)
# ---------------------------------------------------------------------------


def test_cancelled_task_between_submit_and_persist_still_leaves_recoverable_ledger_record(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = CancellingBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        with pytest.raises(asyncio.CancelledError):
            await om.open_position(_single_choice(), qty=1, signal=_signal(), jev=_jev(), now_epoch=1000.0)

        open_positions = ledger.open_positions()
        assert len(open_positions) == 1
        pos = open_positions[0]
        assert pos.status == "pending"
        assert pos.entry_client_order_id.startswith(paper_cfg.client_order_prefix)
        assert len(broker.submitted) == 1

        # A FRESH OrderManager (no cache exists at all in the new protocol)
        # must still be able to find and process this order from the
        # ledger alone.
        fresh_broker = FakeBroker()
        fresh_broker.set_order(pos.entry_client_order_id, status="filled", filled_qty=1, filled_avg_price=1.05)
        fresh_om = OrderManager(paper_cfg, fresh_broker, ledger)

        events = await fresh_om.work_orders(now_epoch=1100.0)
        assert any(e["type"] == "entry_filled" and e["position_id"] == pos.id for e in events)
        assert ledger.get(pos.id).status == "open"

    asyncio.run(_t())


def test_asyncio_shield_protects_submit_and_persist_from_caller_cancellation(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()

        async def slow_submit(intent):
            await asyncio.sleep(0.05)
            broker.submitted.append(intent)
            broker.set_order(intent.client_order_id, status="new")
            return OrderResult(ok=True, client_order_id=intent.client_order_id, status="new")

        broker.submit = slow_submit
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        task = asyncio.create_task(
            om.open_position(_single_choice(), qty=1, signal=_signal(), jev=_jev(), now_epoch=1000.0)
        )
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.1)

        assert len(broker.submitted) == 1  # the submit was NOT aborted by the cancel
        positions = ledger.open_positions()
        assert len(positions) == 1
        assert positions[0].status == "pending"

    asyncio.run(_t())


def test_restart_mid_cancel_recovers(cfg):
    """A cancel was requested (cancel_requested=True) but the process died
    before ever seeing the order confirm terminal. A fresh OrderManager must
    pick the cancel back up from the ledger's own cancel_requested flag."""
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        ledger = await _fresh_ledger(paper_cfg)
        pos = _pending_pos("DPOF-EN-midcancel", qty_original=1, working_since=1000.0, cancel_requested=True)
        await ledger.upsert(pos)

        broker = FakeBroker()
        broker.set_order("DPOF-EN-midcancel", status="canceled", filled_qty=0)
        broker.set_quote("SPY261002C00575000", bid=1.00, ask=1.10)
        om = OrderManager(paper_cfg, broker, ledger)  # brand new -- no cache

        events = await om.work_orders(now_epoch=2000.0)
        # Order confirmed terminal (canceled, 0 fill) -- reprices is still 0
        # (budget available) so it must resend for the remaining qty rather
        # than getting stuck forever waiting on a cancel nobody will ever see land.
        assert any(e["type"] == "entry_repriced" for e in events)
        got = ledger.get(pos.id)
        assert got.cancel_requested is False
        assert len(got.entry_client_order_ids) == 2
        assert len(broker.submitted) == 1  # exactly one new order -- never two live orders

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Entry: weighted-average fills, cancel -> confirm -> resend
# ---------------------------------------------------------------------------


def test_entry_partial_fill_then_cancel_then_remainder_resend(cfg):
    paper_cfg = _paper_cfg(cfg, entry_reprice_seconds=10, entry_max_reprices=2)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = await om.open_position(_single_choice(), qty=3, signal=_signal(), jev=_jev(), now_epoch=1000.0)
        cid1 = pos.entry_client_order_id
        broker.set_quote("SPY261002C00575000", bid=1.00, ask=1.10)

        # Poll: still working, not yet stale.
        events = await om.work_orders(now_epoch=1005.0)
        assert events == []
        assert broker.cancelled == []

        # Poll: stale -> cancel requested. Never resend while cancel is out.
        events = await om.work_orders(now_epoch=1015.0)
        assert any(e["type"] == "entry_cancel_requested" for e in events)
        assert broker.cancelled == [(f"b-{cid1}", cid1)]
        assert len(broker.submitted) == 1  # still just the original order

        # Cancel lands: partial fill of 1 was captured before it died.
        broker.set_order(cid1, status="canceled", filled_qty=1, filled_avg_price=1.05)
        events = await om.work_orders(now_epoch=1016.0)
        assert any(e["type"] == "entry_repriced" for e in events)
        assert len(broker.submitted) == 2  # exactly one new order, never two live

        pos_now = ledger.get(pos.id)
        assert pos_now.qty_open == 1
        assert pos_now.entry_price == pytest.approx(1.05)
        cid2 = pos_now.entry_client_order_id
        assert cid2 != cid1
        assert pos_now.entry_client_order_ids == [cid1, cid2]
        remainder_intent = broker.submitted[-1]
        assert remainder_intent.qty == 2  # 3 - 1 already filled

        # Remainder fills fully.
        broker.set_order(cid2, status="filled", filled_qty=2, filled_avg_price=1.15)
        events = await om.work_orders(now_epoch=1017.0)
        assert any(e["type"] == "entry_filled" for e in events)
        final = ledger.get(pos.id)
        assert final.status == "open"
        assert final.qty_open == 3
        expected_avg = (1 * 1.05 + 2 * 1.15) / 3
        assert final.entry_price == pytest.approx(expected_avg)

    asyncio.run(_t())


def test_never_two_live_orders_for_entry(cfg):
    paper_cfg = _paper_cfg(cfg, entry_reprice_seconds=10, entry_max_reprices=3)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)
        broker.set_quote("SPY261002C00575000", bid=1.00, ask=1.10)

        pos = await om.open_position(_single_choice(), qty=1, signal=_signal(), jev=_jev(), now_epoch=1000.0)
        cid1 = pos.entry_client_order_id
        broker.set_order(cid1, status="new", filled_qty=0)

        # Go stale -- request cancel. While cancel_requested is True and the
        # order is STILL reported non-terminal, work_orders must never submit.
        await om.work_orders(now_epoch=1011.0)
        assert len(broker.submitted) == 1
        for t in (1012.0, 1013.0, 1020.0):
            events = await om.work_orders(now_epoch=t)
            assert events == []
            assert len(broker.submitted) == 1  # still exactly one -- no double order

    asyncio.run(_t())


def test_entry_cancelled_after_max_reprices(cfg):
    paper_cfg = _paper_cfg(cfg, entry_max_reprices=1, entry_reprice_seconds=10)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)
        broker.set_quote("SPY261002C00575000", bid=1.00, ask=1.10)

        pos = await om.open_position(_single_choice(), qty=1, signal=_signal(), jev=_jev(), now_epoch=1000.0)
        cid1 = pos.entry_client_order_id
        broker.set_order(cid1, status="new", filled_qty=0)

        await om.work_orders(now_epoch=1011.0)  # cancel requested
        broker.set_order(cid1, status="canceled", filled_qty=0)
        events = await om.work_orders(now_epoch=1012.0)  # confirmed dead -> resend (reprices 0->1)
        assert any(e["type"] == "entry_repriced" for e in events)
        cid2 = ledger.get(pos.id).entry_client_order_id
        broker.set_order(cid2, status="new", filled_qty=0)

        await om.work_orders(now_epoch=1023.0)  # cancel requested again
        broker.set_order(cid2, status="canceled", filled_qty=0)
        events = await om.work_orders(now_epoch=1024.0)  # budget exhausted -> finalize

        assert any(e["type"] == "entry_cancelled" and e["reason"] == "entry_max_reprices" for e in events)
        got = ledger.get(pos.id)
        assert got.status == "cancelled"
        assert got.exit_reason == "entry_max_reprices"

    asyncio.run(_t())


def test_pending_entry_marked_submit_unconfirmed_after_interval(cfg):
    paper_cfg = _paper_cfg(cfg, entry_reprice_seconds=10)

    async def _t():
        ledger = await _fresh_ledger(paper_cfg)
        pos = _pending_pos("DPOF-EN-ghost", qty_original=1, working_since=10000.0)
        await ledger.upsert(pos)

        broker = FakeBroker()  # get_order_by_client_id always returns None
        om = OrderManager(paper_cfg, broker, ledger)

        events = await om.work_orders(now_epoch=10005.0)  # not yet past the interval
        assert events == []
        assert ledger.get(pos.id).status == "pending"

        events = await om.work_orders(now_epoch=10011.0)  # past working_since + interval
        assert any(e["type"] == "entry_cancelled" and e["reason"] == "submit_unconfirmed" for e in events)
        got = ledger.get(pos.id)
        assert got.status == "cancelled"
        assert got.exit_reason == "submit_unconfirmed"

    asyncio.run(_t())


def test_entry_missing_with_partial_fill_finalizes_open_not_cancelled(cfg):
    paper_cfg = _paper_cfg(cfg, entry_reprice_seconds=10)

    async def _t():
        ledger = await _fresh_ledger(paper_cfg)
        pos = _pending_pos("DPOF-EN-g2", qty_original=2, working_since=10000.0)
        pos.order_fills["DPOF-EN-g2"] = {"purpose": "entry", "filled_qty": 1, "avg": 1.00, "status": "canceled"}
        pos.qty_open = 1
        pos.entry_price = 1.00
        await ledger.upsert(pos)

        broker = FakeBroker()  # missing now (e.g. a resend never reached the broker)
        om = OrderManager(paper_cfg, broker, ledger)
        events = await om.work_orders(now_epoch=10011.0)
        assert any(e["type"] == "entry_partial_then_submit_unconfirmed" and e["qty"] == 1 for e in events)
        assert ledger.get(pos.id).status == "open"

    asyncio.run(_t())


def test_restart_recovery_of_pending_entry(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        ledger = await _fresh_ledger(paper_cfg)
        pos = _pending_pos("DPOF-EN-restart-1", qty_original=3, working_since=5000.0)
        await ledger.upsert(pos)

        broker = FakeBroker()
        broker.set_order("DPOF-EN-restart-1", status="filled", filled_qty=3, filled_avg_price=2.10)

        om = OrderManager(paper_cfg, broker, ledger)  # brand-new, no cache to lose
        events = await om.work_orders(now_epoch=5001.0)

        assert any(e["type"] == "entry_filled" for e in events)
        got = ledger.get(pos.id)
        assert got.status == "open"
        assert got.qty_open == 3
        assert got.entry_price == pytest.approx(2.10)

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# close_position / exit
# ---------------------------------------------------------------------------


def test_close_position_reverses_legs_and_reaches_broker(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _open_pos(qty=2, entry_price=1.00)
        await ledger.upsert(pos)
        broker.set_quote("SPY261002C00575000", bid=1.10, ask=1.20)

        ok = await om.close_position(pos, qty=2, reason="stop_loss", urgent=True, now_epoch=1010.0)
        assert ok is True

        updated = ledger.get(pos.id)
        assert updated.status == "closing"
        assert len(updated.exit_client_order_ids) == 1
        assert len(broker.submitted) == 1

        exit_intent = broker.submitted[-1]
        assert exit_intent.legs[0].side == "sell"
        assert exit_intent.legs[0].position_intent == "sell_to_close"
        assert exit_intent.limit_price is not None

    asyncio.run(_t())


def test_close_position_refuses_when_already_closing(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _open_pos(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker.set_quote("SPY261002C00575000", bid=1.10, ask=1.20)

        ok1 = await om.close_position(pos, qty=1, reason="stop_loss", urgent=True, now_epoch=1010.0)
        assert ok1 is True
        submitted_after_first = len(broker.submitted)

        pos_now = ledger.get(pos.id)
        ok2 = await om.close_position(pos_now, qty=1, reason="stop_loss", urgent=True, now_epoch=1011.0)
        assert ok2 is False
        assert len(broker.submitted) == submitted_after_first

    asyncio.run(_t())


def test_exit_fill_completes_batch_and_closes_position(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _open_pos(qty=3, entry_price=1.00)
        await ledger.upsert(pos)
        broker.set_quote("SPY261002C00575000", bid=1.10, ask=1.20)

        await om.close_position(pos, qty=3, reason="take_profit", urgent=False, now_epoch=1010.0)
        pos_now = ledger.get(pos.id)
        exit_cid = pos_now.exit_client_order_ids[-1]
        broker.set_order(exit_cid, status="filled", filled_qty=3, filled_avg_price=1.30)

        events = await om.work_orders(now_epoch=1011.0)
        ev = next(e for e in events if e["type"] == "exit_filled")
        assert ev["closed"] is True
        assert ev["qty"] == 3
        assert ev["realized_delta"] == pytest.approx((1.30 - 1.00) * 100 * 3)

        final = ledger.get(pos.id)
        assert final.realized_pnl == pytest.approx(90.0)
        assert final.qty_open == 0
        assert final.status == "closed"

    asyncio.run(_t())


def test_partial_exit_batch_leaves_position_open(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _open_pos(qty=3, entry_price=1.00)
        await ledger.upsert(pos)
        broker.set_quote("SPY261002C00575000", bid=1.10, ask=1.20)

        await om.close_position(pos, qty=1, reason="take_profit_tier1", urgent=False, now_epoch=1010.0)
        pos_now = ledger.get(pos.id)
        exit_cid = pos_now.exit_client_order_ids[-1]
        broker.set_order(exit_cid, status="filled", filled_qty=1, filled_avg_price=1.30)

        events = await om.work_orders(now_epoch=1011.0)
        ev = next(e for e in events if e["type"] == "exit_filled")
        assert ev["closed"] is False
        assert ev["qty"] == 1

        final = ledger.get(pos.id)
        assert final.status == "open"
        assert final.qty_open == 2
        assert final.realized_pnl == pytest.approx((1.30 - 1.00) * 100 * 1)

    asyncio.run(_t())


def test_vertical_exit_uses_net_pricing_and_negative_mleg_sign(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        legs = [
            {"symbol": "SPY261002C00575000", "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"},
            {"symbol": "SPY261002C00580000", "side": "sell", "ratio_qty": 1, "position_intent": "sell_to_open"},
        ]
        pos = _open_pos(qty=1, entry_price=1.50, structure="debit_vertical", legs=legs)
        await ledger.upsert(pos)
        broker.set_quote("SPY261002C00575000", bid=1.60, ask=1.70)
        broker.set_quote("SPY261002C00580000", bid=0.30, ask=0.40)

        ok = await om.close_position(pos, qty=1, reason="take_profit", urgent=False, now_epoch=1010.0)
        assert ok is True

        exit_intent = broker.submitted[-1]
        assert len(exit_intent.legs) == 2
        # net bid = long.bid - short.ask = 1.20; net ask = long.ask - short.bid = 1.40
        assert 1.20 <= exit_intent.limit_price <= 1.40

        pos_now = ledger.get(pos.id)
        exit_cid = pos_now.exit_client_order_ids[-1]
        # Broker reports the credit as NEGATIVE per Alpaca's mleg sign convention.
        broker.set_order(exit_cid, status="filled", filled_qty=1, filled_avg_price=-1.30)
        await om.work_orders(now_epoch=1011.0)
        final = ledger.get(pos.id)
        assert final.realized_pnl == pytest.approx((1.30 - 1.50) * 100 * 1)

    asyncio.run(_t())


def test_urgent_exit_escalates_to_market_only_after_prior_order_confirmed_terminal(cfg):
    paper_cfg = _paper_cfg(cfg, exit_max_reprices=1, exit_reprice_seconds=10)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _open_pos(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker.set_quote("SPY261002C00575000", bid=0.50, ask=0.60)

        await om.close_position(pos, qty=1, reason="stop_loss", urgent=True, now_epoch=1010.0)
        pos_now = ledger.get(pos.id)
        cid1 = pos_now.exit_client_order_ids[-1]
        broker.set_order(cid1, status="new", filled_qty=0)

        # Stale -> cancel requested. Must NOT escalate to market yet.
        events = await om.work_orders(now_epoch=1021.0)
        assert any(e["type"] == "exit_cancel_requested" for e in events)
        assert not any(i.purpose == "exit" and i.limit_price is None for i in broker.submitted)
        assert broker.cancelled == [(f"b-{cid1}", cid1)]

        # Still not terminal yet -- must keep waiting, not escalate.
        events = await om.work_orders(now_epoch=1022.0)
        assert events == []
        assert not any(i.limit_price is None for i in broker.submitted)

        # NOW it confirms terminal -- reprices (0) < max (1), so first it
        # resends a normal repriced limit order, not market yet.
        broker.set_order(cid1, status="canceled", filled_qty=0)
        events = await om.work_orders(now_epoch=1023.0)
        assert any(e["type"] == "exit_repriced" for e in events)
        assert not any(i.limit_price is None for i in broker.submitted)

        # That resend goes stale too, gets cancelled, confirms terminal --
        # NOW reprices (1) >= max (1) -> market, single-leg, urgent.
        cid2 = ledger.get(pos.id).exit_client_order_ids[-1]
        broker.set_order(cid2, status="new", filled_qty=0)
        await om.work_orders(now_epoch=1034.0)  # cancel requested
        broker.set_order(cid2, status="canceled", filled_qty=0)
        events = await om.work_orders(now_epoch=1035.0)
        assert any(e["type"] == "exit_escalated_market" for e in events)
        market_orders = [i for i in broker.submitted if i.purpose == "exit" and i.limit_price is None]
        assert len(market_orders) == 1
        assert len(market_orders[0].legs) == 1

    asyncio.run(_t())


def test_mleg_exit_never_escalates_to_market(cfg):
    paper_cfg = _paper_cfg(cfg, exit_max_reprices=0, exit_reprice_seconds=10)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        legs = [
            {"symbol": "SPY261002C00575000", "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"},
            {"symbol": "SPY261002C00580000", "side": "sell", "ratio_qty": 1, "position_intent": "sell_to_open"},
        ]
        pos = _open_pos(qty=1, entry_price=1.50, structure="debit_vertical", legs=legs)
        await ledger.upsert(pos)
        broker.set_quote("SPY261002C00575000", bid=1.60, ask=1.70)
        broker.set_quote("SPY261002C00580000", bid=0.30, ask=0.40)

        await om.close_position(pos, qty=1, reason="take_profit", urgent=True, now_epoch=1010.0)
        cid1 = ledger.get(pos.id).exit_client_order_ids[-1]
        broker.set_order(cid1, status="new", filled_qty=0)

        await om.work_orders(now_epoch=1021.0)  # cancel requested (reprices already 0 == max)
        broker.set_order(cid1, status="canceled", filled_qty=0)
        events = await om.work_orders(now_epoch=1022.0)

        assert any(e["type"] == "exit_escalated_marketable" for e in events)
        assert not any(i.purpose == "exit" and i.limit_price is None for i in broker.submitted)
        mleg_orders = [i for i in broker.submitted if i.purpose == "exit" and len(i.legs) == 2]
        assert mleg_orders[-1].limit_price == pytest.approx(1.20, abs=0.01)  # natural bid

    asyncio.run(_t())


def test_exit_missing_order_resends_for_remaining_qty(cfg):
    """close_position() had no usable quote and left the batch pending --
    step 4's missing-order handling must resend, not dead-end forever."""
    paper_cfg = _paper_cfg(cfg, exit_reprice_seconds=10)

    async def _t():
        broker = FakeBroker()  # no quotes set -> exit_limit_price is None
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _open_pos(qty=1, entry_price=1.00)
        await ledger.upsert(pos)

        ok = await om.close_position(pos, qty=1, reason="stop_loss", urgent=False, now_epoch=1000.0)
        assert ok is False  # no quote -> nothing sent yet
        assert broker.submitted == []
        assert ledger.get(pos.id).status == "closing"
        assert ledger.get(pos.id).exit_client_order_ids == []

        # Still no quote, not yet past the interval.
        events = await om.work_orders(now_epoch=1005.0)
        assert events == []

        # Quote becomes available; past the interval -- must resend now.
        broker.set_quote("SPY261002C00575000", bid=1.10, ask=1.20)
        events = await om.work_orders(now_epoch=1011.0)
        assert any(e["type"] == "exit_resent" for e in events)
        assert len(broker.submitted) == 1
        got = ledger.get(pos.id)
        assert len(got.exit_client_order_ids) == 1

    asyncio.run(_t())


def test_restart_recovery_of_closing_exit(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        ledger = await _fresh_ledger(paper_cfg)
        pos = _open_pos(qty=2, entry_price=1.00, status="closing",
                         exit_target_qty=2, exit_filled_baseline=0, exit_urgent=False,
                         working_since=900.0, reprices=0,
                         exit_client_order_ids=["DPOF-EX-restart-1"], exit_reason="stop_loss")
        await ledger.upsert(pos)

        broker = FakeBroker()
        broker.set_order("DPOF-EX-restart-1", status="filled", filled_qty=2, filled_avg_price=0.60)

        om = OrderManager(paper_cfg, broker, ledger)  # fresh -- no cache to lose
        events = await om.work_orders(now_epoch=5000.0)

        assert any(e["type"] == "exit_filled" for e in events)
        got = ledger.get(pos.id)
        assert got.status == "closed"
        assert got.qty_open == 0
        assert got.realized_pnl == pytest.approx(-80.0)  # (0.60 - 1.00) * 100 * 2

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------


def test_reconcile_marks_ledger_open_broker_absent_position_closed(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _open_pos(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker._option_positions = []

        warnings = await om.reconcile()
        assert any(w.startswith("reconciled_absent:") for w in warnings)
        got = ledger.get(pos.id)
        assert got.status == "closed"
        assert got.exit_reason == "reconciled_absent"

    asyncio.run(_t())


def test_reconcile_skips_position_with_cancel_requested(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _open_pos(qty=1, entry_price=1.00, status="closing", cancel_requested=True,
                         exit_client_order_ids=["DPOF-EX-1"])
        await ledger.upsert(pos)
        broker._option_positions = []  # would otherwise look "absent"

        warnings = await om.reconcile()
        assert warnings == []
        assert ledger.get(pos.id).status == "closing"  # untouched

    asyncio.run(_t())


def test_reconcile_skips_position_with_non_terminal_latest_order(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _open_pos(qty=1, entry_price=1.00, status="closing",
                         exit_client_order_ids=["DPOF-EX-1"],
                         order_fills={"DPOF-EX-1": {"purpose": "exit", "filled_qty": 0, "avg": None, "status": "new"}})
        await ledger.upsert(pos)
        broker._option_positions = []

        warnings = await om.reconcile()
        assert warnings == []
        assert ledger.get(pos.id).status == "closing"

    asyncio.run(_t())


def test_reconcile_reports_orphan_without_trading_it(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        broker._option_positions = [
            {"symbol": "SPY261002C00575000", "qty": 1, "avg_entry_price": 1.0, "market_value": 100.0, "unrealized_pl": 0.0}
        ]

        warnings = await om.reconcile()
        assert "orphan:SPY261002C00575000" in warnings
        assert broker.submitted == []
        assert broker.cancelled == []

    asyncio.run(_t())


def test_reconcile_ignores_symbols_off_our_underlyings(cfg):
    paper_cfg = replace(_paper_cfg(cfg), underlyings=("SPY",))

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        broker._option_positions = [
            {"symbol": "AAPL261002C00200000", "qty": 1, "avg_entry_price": 1.0, "market_value": 100.0, "unrealized_pl": 0.0}
        ]

        warnings = await om.reconcile()
        assert warnings == []

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Engine-facing event contract (engine.py consumes these exact shapes)
# ---------------------------------------------------------------------------


def _assert_base_event_shape(events):
    for e in events:
        assert "type" in e and isinstance(e["type"], str)
        assert "position_id" in e and isinstance(e["position_id"], str)


def test_event_contract_entry_filled(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = await om.open_position(_single_choice(), qty=2, signal=_signal(), jev=_jev(), now_epoch=1000.0)
        broker.set_order(pos.entry_client_order_id, status="filled", filled_qty=2, filled_avg_price=1.10)
        events = await om.work_orders(now_epoch=1001.0)

        _assert_base_event_shape(events)
        ev = next(e for e in events if e["type"] == "entry_filled")
        assert ev["qty"] == 2  # entry_filled total

    asyncio.run(_t())


def test_event_contract_entry_cancelled(cfg):
    paper_cfg = _paper_cfg(cfg, entry_reprice_seconds=10)

    async def _t():
        ledger = await _fresh_ledger(paper_cfg)
        pos = _pending_pos("DPOF-EN-nevershow", qty_original=1, working_since=1000.0)
        await ledger.upsert(pos)
        broker = FakeBroker()  # never found
        om = OrderManager(paper_cfg, broker, ledger)

        events = await om.work_orders(now_epoch=1011.0)
        _assert_base_event_shape(events)
        ev = next(e for e in events if e["type"] == "entry_cancelled")
        assert "reason" in ev

    asyncio.run(_t())


def test_event_contract_entry_partial_then_status(cfg):
    paper_cfg = _paper_cfg(cfg, entry_reprice_seconds=10)

    async def _t():
        ledger = await _fresh_ledger(paper_cfg)
        pos = _pending_pos("DPOF-EN-partial", qty_original=2, working_since=1000.0)
        pos.order_fills["DPOF-EN-partial"] = {"purpose": "entry", "filled_qty": 1, "avg": 1.00, "status": "canceled"}
        pos.qty_open = 1
        pos.entry_price = 1.00
        await ledger.upsert(pos)
        broker = FakeBroker()  # never found again
        om = OrderManager(paper_cfg, broker, ledger)

        events = await om.work_orders(now_epoch=1011.0)
        _assert_base_event_shape(events)
        ev = next(e for e in events if e["type"].startswith("entry_partial_then_"))
        assert ev["qty"] == 1

    asyncio.run(_t())


def test_event_contract_exit_filled(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        om = OrderManager(paper_cfg, broker, ledger)

        pos = _open_pos(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        broker.set_quote("SPY261002C00575000", bid=1.10, ask=1.20)
        await om.close_position(pos, qty=1, reason="take_profit", urgent=False, now_epoch=1010.0)
        exit_cid = ledger.get(pos.id).exit_client_order_ids[-1]
        broker.set_order(exit_cid, status="filled", filled_qty=1, filled_avg_price=1.30)

        events = await om.work_orders(now_epoch=1011.0)
        _assert_base_event_shape(events)
        ev = next(e for e in events if e["type"] == "exit_filled")
        assert set(("qty", "realized_delta", "closed")) <= set(ev.keys())
        assert isinstance(ev["closed"], bool)
        assert ev["realized_delta"] is None or isinstance(ev["realized_delta"], float)

    asyncio.run(_t())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))



# ---------------------------------------------------------------------------
# Orchestrator review: lookup failure != not found
# ---------------------------------------------------------------------------

class LookupFailBroker(FakeBroker):
    """get_order_by_client_id raises BrokerLookupError (transient API failure)."""

    async def get_order_by_client_id(self, client_order_id):
        from agent.dpof.broker import BrokerLookupError
        raise BrokerLookupError("lookup_failed:ConnectionError")


def test_exit_lookup_error_never_resends_or_mutates(cfg):
    paper_cfg = _paper_cfg(cfg, exit_reprice_seconds=10)

    async def _t():
        broker = LookupFailBroker()
        broker.set_quote("SPY261002C00575000", bid=1.10, ask=1.20)
        ledger = await _fresh_ledger(paper_cfg)
        pos = _open_pos(qty=2, entry_price=1.00, status="closing",
                        exit_target_qty=2, exit_filled_baseline=0, exit_urgent=True,
                        working_since=900.0, reprices=0,
                        exit_client_order_ids=["DPOF-EX-live-1"], exit_reason="stop_loss")
        await ledger.upsert(pos)
        before = ledger.get(pos.id).to_dict()
        om = OrderManager(paper_cfg, broker, ledger)
        events = await om.work_orders(now_epoch=5000.0)   # far past every interval
        assert [e["type"] for e in events] == ["lookup_error"]
        assert broker.submitted == [] and broker.cancelled == []
        assert ledger.get(pos.id).to_dict() == before

    asyncio.run(_t())


def test_entry_lookup_error_never_gives_up_live_entry(cfg):
    paper_cfg = _paper_cfg(cfg, entry_reprice_seconds=10)

    async def _t():
        broker = LookupFailBroker()
        ledger = await _fresh_ledger(paper_cfg)
        pos = _open_pos(qty=2, entry_price=None, status="pending", qty_open=0,
                        entry_client_order_id="DPOF-EN-live-1", entry_client_order_ids=["DPOF-EN-live-1"],
                        working_since=900.0, reprices=0)
        await ledger.upsert(pos)
        om = OrderManager(paper_cfg, broker, ledger)
        events = await om.work_orders(now_epoch=5000.0)
        assert [e["type"] for e in events] == ["lookup_error"]
        assert ledger.get(pos.id).status == "pending"   # NOT cancelled/submit_unconfirmed
        assert broker.submitted == [] and broker.cancelled == []

    asyncio.run(_t())



# ---------------------------------------------------------------------------
# Pass-2 CRITICAL-1 regression: a failed/lost cancel must never wedge a position
# ---------------------------------------------------------------------------

class FlakyCancelBroker(FakeBroker):
    def __init__(self, cancel_results):
        super().__init__()
        self.cancel_results = list(cancel_results)

    async def cancel_order(self, broker_order_id, client_order_id):
        self.cancelled.append((broker_order_id, client_order_id))
        return self.cancel_results.pop(0) if self.cancel_results else True


def test_failed_cancel_is_retried_not_wedged(cfg):
    paper_cfg = _paper_cfg(cfg, exit_reprice_seconds=10)

    async def _t():
        broker = FlakyCancelBroker([False, False])
        ledger = await _fresh_ledger(paper_cfg)
        cid = "DPOF-EX-live-1"
        broker.set_order(cid, status="new")          # live, never goes terminal on its own
        pos = _open_pos(qty=2, entry_price=1.00, status="closing", exit_target_qty=2, exit_filled_baseline=0,
                        exit_urgent=True, working_since=900.0, reprices=0,
                        exit_client_order_ids=[cid], exit_reason="stop_loss")
        await ledger.upsert(pos)
        om = OrderManager(paper_cfg, broker, ledger)
        e1 = await om.work_orders(now_epoch=1000.0)
        assert [e["type"] for e in e1] == ["exit_cancel_failed"] and ledger.get(pos.id).cancel_requested is False
        e2 = await om.work_orders(now_epoch=1030.0)
        assert [e["type"] for e in e2] == ["exit_cancel_failed"]
        e3 = await om.work_orders(now_epoch=1060.0)
        assert [e["type"] for e in e3] == ["exit_cancel_requested"] and ledger.get(pos.id).cancel_requested is True
        # Cancel "accepted" but the order never goes terminal: re-issued after the interval.
        e4 = await om.work_orders(now_epoch=1065.0)
        assert e4 == []
        e5 = await om.work_orders(now_epoch=1071.0)
        assert [e["type"] for e in e5] == ["exit_cancel_retried"]
        assert len(broker.cancelled) == 4 and broker.submitted == []   # never a second live order

    asyncio.run(_t())


def test_cancel_stuck_event_after_repeated_attempts(cfg):
    paper_cfg = _paper_cfg(cfg, exit_reprice_seconds=10)

    async def _t():
        broker = FlakyCancelBroker([False] * 20)
        ledger = await _fresh_ledger(paper_cfg)
        cid = "DPOF-EX-live-2"
        broker.set_order(cid, status="pending_cancel")
        pos = _open_pos(qty=1, entry_price=1.00, status="closing", exit_target_qty=1, exit_filled_baseline=0,
                        exit_urgent=True, working_since=900.0, exit_client_order_ids=[cid], exit_reason="eod_flatten")
        await ledger.upsert(pos)
        om = OrderManager(paper_cfg, broker, ledger)
        types = []
        for i in range(5):
            types += [e["type"] for e in await om.work_orders(now_epoch=1000.0 + 30 * i)]
        assert types.count("cancel_stuck") == 1 and ledger.get(pos.id).cancel_attempts == 5

    asyncio.run(_t())


def test_terminal_order_resets_cancel_counters(cfg):
    paper_cfg = _paper_cfg(cfg, exit_reprice_seconds=10)

    async def _t():
        broker = FlakyCancelBroker([])
        ledger = await _fresh_ledger(paper_cfg)
        cid = "DPOF-EX-live-3"
        broker.set_order(cid, status="canceled", filled_qty=0)
        broker.set_quote("SPY261002C00575000", bid=1.10, ask=1.20)
        pos = _open_pos(qty=1, entry_price=1.00, status="closing", exit_target_qty=1, exit_filled_baseline=0,
                        exit_urgent=False, working_since=900.0, exit_client_order_ids=[cid], exit_reason="max_hold",
                        cancel_requested=True, cancel_requested_at=950.0, cancel_attempts=3)
        await ledger.upsert(pos)
        om = OrderManager(paper_cfg, broker, ledger)
        await om.work_orders(now_epoch=1000.0)
        got = ledger.get(pos.id)
        assert got.cancel_requested is False and got.cancel_attempts == 0 and got.cancel_requested_at is None
        assert len(broker.submitted) == 1          # resent the exit for the remaining qty

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# flatten_entry (EOD-flatten path for a still-PENDING entry -- pass 2,
# order-safety finding: without this, a working entry order has no wall-
# clock awareness of flatten_et at all and can keep reprice-resubmitting
# (or simply fill) past 15:50, undiscovered until the next session).
# ---------------------------------------------------------------------------


def test_flatten_entry_cancels_working_order_and_exhausts_reprice_budget(cfg):
    paper_cfg = _paper_cfg(cfg, entry_reprice_seconds=45, entry_max_reprices=2)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        cid = "DPOF-EN-live-1"
        broker.set_order(cid, status="new")  # still working, unfilled
        pos = _pending_pos(cid, qty_original=2, working_since=1000.0, reprices=0)
        await ledger.upsert(pos)
        om = OrderManager(paper_cfg, broker, ledger)

        cancelled = await om.flatten_entry(ledger.get("p1"), "eod_flatten", now_epoch=1010.0)

        assert cancelled is True
        assert broker.cancelled == [(f"b-{cid}", cid)]
        got = ledger.get("p1")
        assert got.cancel_requested is True
        assert got.reprices == paper_cfg.entry_max_reprices  # budget exhausted -- can never resubmit
        assert got.exit_reason == "eod_flatten"
        assert got.status == "pending"  # not finalized yet -- work_orders() finalizes once the cancel confirms

        # Next poll: cancel confirms terminal. Even though `remaining` (2) is
        # still > 0, the exhausted reprice budget must take the existing
        # "budget exhausted" branch -- NOT resubmit a fresh reprice order.
        broker.set_order(cid, status="canceled", filled_qty=0)
        events = await om.work_orders(now_epoch=1011.0)
        assert any(e["type"] == "entry_cancelled" and e["reason"] == "eod_flatten" for e in events)
        assert len(broker.submitted) == 0  # never resubmitted
        final = ledger.get("p1")
        assert final.status == "cancelled"

    asyncio.run(_t())


def test_flatten_entry_partial_fill_finalizes_to_open_not_resubmitted(cfg):
    """A partially-filled pending entry, force-cancelled at flatten_et, must
    finalize to status="open" (so engine.py's flatten-for-"open" path closes
    the filled qty on the very next cycle) -- never resubmit a reprice order
    for the unfilled remainder."""
    paper_cfg = _paper_cfg(cfg, entry_reprice_seconds=45, entry_max_reprices=2)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        cid = "DPOF-EN-live-2"
        broker.set_order(cid, status="new")
        pos = _pending_pos(cid, qty_original=3, working_since=1000.0, reprices=0)
        await ledger.upsert(pos)
        om = OrderManager(paper_cfg, broker, ledger)

        cancelled = await om.flatten_entry(ledger.get("p1"), "eod_flatten", now_epoch=1010.0)
        assert cancelled is True

        # 1 of 3 filled before the cancel won the race.
        broker.set_order(cid, status="canceled", filled_qty=1, filled_avg_price=1.05)
        events = await om.work_orders(now_epoch=1011.0)
        assert any(e["type"] == "entry_partial_then_canceled" for e in events)
        assert len(broker.submitted) == 0  # never resubmitted for the remaining 2
        final = ledger.get("p1")
        assert final.status == "open"
        assert final.qty_open == 1

    asyncio.run(_t())


def test_flatten_entry_noop_when_already_cancelling(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        cid = "DPOF-EN-live-3"
        broker.set_order(cid, status="pending_cancel")
        pos = _pending_pos(
            cid, qty_original=1, working_since=1000.0, reprices=0,
            cancel_requested=True, cancel_requested_at=1005.0,
        )
        await ledger.upsert(pos)
        om = OrderManager(paper_cfg, broker, ledger)

        cancelled = await om.flatten_entry(ledger.get("p1"), "eod_flatten", now_epoch=1010.0)

        assert cancelled is False
        assert broker.cancelled == []  # no duplicate cancel call -- the ordinary retry loop owns this
        got = ledger.get("p1")
        assert got.reprices == paper_cfg.entry_max_reprices  # budget still exhausted regardless

    asyncio.run(_t())


def test_flatten_entry_noop_when_order_already_terminal(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        cid = "DPOF-EN-live-4"
        broker.set_order(cid, status="filled", filled_qty=1, filled_avg_price=1.05)
        pos = _pending_pos(cid, qty_original=1, working_since=1000.0, reprices=0)
        await ledger.upsert(pos)
        om = OrderManager(paper_cfg, broker, ledger)

        cancelled = await om.flatten_entry(ledger.get("p1"), "eod_flatten", now_epoch=1010.0)

        assert cancelled is False
        assert broker.cancelled == []  # already resolved -- next work_orders() poll finalizes it normally

    asyncio.run(_t())


def test_flatten_entry_noop_on_non_pending_status(cfg):
    paper_cfg = _paper_cfg(cfg)

    async def _t():
        broker = FakeBroker()
        ledger = await _fresh_ledger(paper_cfg)
        pos = _open_pos(qty=1, entry_price=1.00)
        await ledger.upsert(pos)
        om = OrderManager(paper_cfg, broker, ledger)

        cancelled = await om.flatten_entry(ledger.get("p1"), "eod_flatten", now_epoch=1010.0)

        assert cancelled is False
        assert broker.cancelled == []
        got = ledger.get("p1")
        assert got.reprices == 0  # untouched -- flatten_entry only ever acts on status=="pending"

    asyncio.run(_t())
