"""
CLIO DPOF — order lifecycle (I/O): entry/exit submission, cancel-confirm-resend
reprice/escalate, derived-fill bookkeeping, reconciliation.

THE BUG THIS FILE USED TO HAVE: Alpaca's PATCH /v2/orders/{id} does NOT edit
an order in place. It creates a NEW order (new id, new auto-generated
client_order_id unless one is sent) and the OLD order transitions to status
"replaced" with replaced_by=<new id>. A prior version of this module called
``DPOFBroker.replace_order`` to reprice, which would have silently lost track
of the live order on every reprice -> orphaned working orders / possible
double fills. That method no longer exists (see the comment in its place in
broker.py); nothing in this module calls anything named "replace" any more.

Protocol (restart-safe, no in-memory working-order cache at all -- every
piece of state work_orders() needs is a field on PositionRecord, per
models.py's entry_client_order_id(s)/exit_client_order_ids/order_fills/
exit_target_qty/exit_filled_baseline/exit_urgent/working_since/reprices/
cancel_requested):

  1. FILLS ARE DERIVED, NEVER INCREMENTED. Every poll of an order overwrites
     ``pos.order_fills[cid]`` with the broker's CUMULATIVE
     {"purpose","filled_qty","avg","status"}. :func:`recompute_from_fills` is
     a pure function that re-derives qty_open/entry_price/realized_pnl from
     that dict every time -- safe to call repeatedly, and safe after a
     restart with zero in-memory history (nothing to lose).
  2. Order chain: every order ever sent for a position appends a NEW
     DPOF-prefixed client id to entry_client_order_ids / exit_client_order_ids
     (entry_client_order_id tracks the latest), persisted (shielded) BEFORE
     submit. work_orders() polls only the latest order of each chain, plus
     any earlier order whose last-known fill status was non-terminal.
  3. A stale working order (now - working_since >= reprice interval) is
     CANCELLED, never repriced-in-place. cancel_requested=True gates out any
     further send for that position until the cancel is confirmed terminal.
     Once terminal, fills are recorded, then either the next order in the
     chain is sent for whatever qty remains, or the batch/position finalizes.
  4. A missing order (get_order_by_client_id -> None) is handled the same
     way once the grace interval (anchored on the persisted working_since,
     so it survives a restart) has elapsed: entries give up
     ("submit_unconfirmed"); exits are risk-reducing and always resend.
  5. Persist-before-submit + asyncio.shield around submit (+ any immediate
     ledger write) still holds for cancellation safety (CLAUDE.md
     anti-pattern #35 -- CancelledError bypasses ``except Exception``).
  6. A position is NEVER allowed to have two live orders in flight: nothing
     is ever sent while cancel_requested is True, or while the chain's
     latest order is confirmed non-terminal.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from agent.dpof.broker import BrokerLookupError, DPOFBroker
from agent.dpof.config import DPOFConfig
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import (
    ContractChoice,
    JevVerdict,
    OrderIntent,
    OrderLeg,
    PositionRecord,
    RECONCILE_FILL_PURPOSES,
    Signal,
)
from agent.dpof.pricing import entry_limit_price, exit_limit_price, reprice_entry, reprice_exit

# RECONCILE_FILL_PURPOSES = ("reconcile_add", "reconcile_remove") -- see models.py.
# "reconcile_add" behaves like an "entry" fill for qty_open purposes; "reconcile_remove"
# behaves like an "exit" fill. Both are ALWAYS avg=None (price unknown) and never enter
# entry_price/realized_pnl -- only recompute_from_fills (below) and reconcile() (the
# only writer) ever look at these two purpose strings.
_RECONCILE_ADD, _RECONCILE_REMOVE = RECONCILE_FILL_PURPOSES

logger = logging.getLogger("clio.dpof.orders")

_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

# "replaced" is defensive: nothing in this module calls replace_order any
# more, but a "replaced" order (from any external source) must still be
# treated as terminal-dead rather than polled forever.
_TERMINAL_FILLED = "filled"
_TERMINAL_DEAD = ("canceled", "expired", "rejected", "replaced")
_TERMINAL_STATUSES = (_TERMINAL_FILLED,) + _TERMINAL_DEAD
_CANCEL_STUCK_ATTEMPTS = 5   # alert (and re-alert every N) when a cancel will not take


def _is_terminal(status: str) -> bool:
    return status in _TERMINAL_STATUSES


def _underlying_of(symbol: str) -> str:
    m = _OCC_RE.match(symbol)
    return m.group(1) if m else ""


def _has_live_recorded_order(pos: PositionRecord) -> bool:
    """True if any order in the entry or exit chain has a RECORDED status
    that isn't terminal (a fill could still land). A chain cid with no
    order_fills entry at all never reached the broker -- the manager only
    moves past one after a 404 -- so it is ignored rather than treated as
    live forever. Used only by reconcile()'s quantity-check adoption gate
    (docs/design/clio_dpof_followups_2026-09-26.md §3.2)."""
    for cid in list(pos.entry_client_order_ids) + list(pos.exit_client_order_ids):
        fill = pos.order_fills.get(cid)
        if fill is not None and not _is_terminal(fill.get("status", "")):
            return True
    return False


def _leg_field(leg: Any, key: str):
    """Read a field off either an OrderLeg dataclass or its dict form."""
    return leg.get(key) if isinstance(leg, dict) else getattr(leg, key)


# ---------------------------------------------------------------------------
# Pure: derived-fill accounting
# ---------------------------------------------------------------------------


def recompute_from_fills(pos: PositionRecord) -> PositionRecord:
    """PURE. Re-derives entry_price / qty_open / realized_pnl from
    ``pos.order_fills`` (the broker's last-reported CUMULATIVE fill per
    order). Never increments anything -- idempotent under repeated identical
    polls, and correct immediately after a restart with no history at all.

    qty_open: entry_filled minus exited, floored at 0, WHERE entry_filled
    sums purpose "entry" AND "reconcile_add" fills, and exited sums purpose
    "exit" AND "reconcile_remove" fills (docs/design/
    clio_dpof_followups_2026-09-26.md §3.1) -- the two RECONCILE_FILL_PURPOSES
    synthetic entries reconcile() may write let the ledger's notion of "how
    many contracts do we hold" track the broker's, without a real order.

    entry_price: qty-weighted average over purpose "entry" fills ONLY with a
    known avg (None if nothing has filled yet). Synthetic "reconcile_add"
    fills are ALWAYS avg=None (price genuinely unknown -- R2) and never enter
    this average, even though their quantity counts toward qty_open above.

    realized_pnl: sum over purpose "exit" fills ONLY of (avg - entry_price) *
    100 * qty, skipped per-order when either side is unknown. Synthetic
    "reconcile_remove" fills never contribute (also always avg=None) --
    exactly like every other order_fills consumer in this module, this stays
    purpose-"exit"-only; adopting a reconcile_remove quantity flags the
    position's P&L as incomplete (readmodel/trade_stats territory, not here)
    rather than inventing a number. For a debit vertical the exit avg is the
    mleg NET CREDIT reported as a negative number (broker.py sends the
    closing mleg limit negative per Alpaca's sign convention) -- take abs()
    so realized_pnl reads as (credit_received - debit_paid).

    Returns a new PositionRecord (does not mutate the input).
    """
    new = copy.deepcopy(pos)

    entry_qty_fills = [
        f for f in new.order_fills.values()
        if f.get("purpose") in ("entry", _RECONCILE_ADD) and (f.get("filled_qty") or 0) > 0
    ]
    entry_filled = sum(f["filled_qty"] for f in entry_qty_fills)
    if entry_filled > 0:
        priced = [f for f in entry_qty_fills if f.get("purpose") == "entry" and f.get("avg") is not None]
        priced_qty = sum(f["filled_qty"] for f in priced)
        new.entry_price = (sum(f["avg"] * f["filled_qty"] for f in priced) / priced_qty) if priced_qty > 0 else None
    else:
        new.entry_price = None

    exit_qty_fills = [
        f for f in new.order_fills.values()
        if f.get("purpose") in ("exit", _RECONCILE_REMOVE) and (f.get("filled_qty") or 0) > 0
    ]
    exited = sum(f["filled_qty"] for f in exit_qty_fills)

    new.qty_open = max(0, entry_filled - exited)

    realized = 0.0
    if new.entry_price is not None:
        for f in exit_qty_fills:
            if f.get("purpose") != "exit":
                continue  # reconcile_remove is always avg=None anyway; explicit for clarity/§3.1
            avg = f.get("avg")
            if avg is None:
                continue
            avg_signed = abs(avg) if new.structure == "debit_vertical" else avg
            realized += (avg_signed - new.entry_price) * 100.0 * f["filled_qty"]
    new.realized_pnl = realized

    return new


def _exit_fills_in_current_batch(pos: PositionRecord) -> list[dict]:
    """The order_fills belonging to the batch that started at
    exit_filled_baseline, in chain order. Batches never split a single order
    (a new batch always starts a brand-new order), so walking the chain and
    keeping everything from the point the running total first reaches the
    baseline is exact."""
    running_before = 0
    result = []
    for cid in pos.exit_client_order_ids:
        f = pos.order_fills.get(cid)
        qty = (f.get("filled_qty") or 0) if f else 0
        if f is not None and f.get("purpose") == "exit" and running_before >= pos.exit_filled_baseline:
            result.append(f)
        running_before += qty
    return result


def _batch_realized_pnl(pos: PositionRecord) -> Optional[float]:
    """PURE. realized_pnl contributed by just the current/most-recently-
    finished exit batch (not the position's lifetime total). None when it
    genuinely can't be computed (no entry price known yet) -- distinct from
    a real, computed zero."""
    if pos.entry_price is None:
        return None
    total = 0.0
    for f in _exit_fills_in_current_batch(pos):
        avg = f.get("avg")
        if avg is None:
            continue
        avg_signed = abs(avg) if pos.structure == "debit_vertical" else avg
        total += (avg_signed - pos.entry_price) * 100.0 * (f.get("filled_qty") or 0)
    return total


class OrderManager:
    def __init__(self, cfg: DPOFConfig, broker: DPOFBroker, ledger: DPOFLedger):
        self._cfg = cfg
        self._broker = broker
        self._ledger = ledger
        # Reconcile quantity-check confirm state (docs/design/
        # clio_dpof_followups_2026-09-26.md §3.2 point 4): symbol -> {"ledger":
        # int, "broker": int, "first_seen": epoch}. IN-MEMORY ONLY -- a restart
        # loses it, which only delays an adoption (the mismatch is re-observed
        # and re-confirmed from scratch), never corrupts anything.
        self._pending_qty_mismatch: dict[str, dict] = {}

    def update_config(self, cfg: DPOFConfig) -> None:
        self._cfg = cfg

    def has_pending_qty_mismatch(self) -> bool:
        """True while at least one symbol is mid-confirmation for the
        quantity-check adoption (§3.2 point 4) -- engine.py's step 8c uses
        this to run reconcile() every cycle (not just every Nth) while a
        mismatch is pending, so the confirm window doesn't have to wait out
        a full 10-cycle gap on top of reconcile_confirm_seconds."""
        return bool(self._pending_qty_mismatch)

    def new_client_order_id(self, purpose: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        tag = (purpose[:2] or "xx").upper()
        cid = f"{self._cfg.client_order_prefix}-{tag}-{ts}-{suffix}"
        return cid[:48]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _leg_to_dict(leg: OrderLeg) -> dict:
        return {
            "symbol": leg.symbol,
            "side": leg.side,
            "ratio_qty": leg.ratio_qty,
            "position_intent": leg.position_intent,
        }

    @staticmethod
    def _legs_from_dicts(legs: list[dict]) -> list[OrderLeg]:
        return [
            OrderLeg(symbol=l["symbol"], side=l["side"], ratio_qty=l.get("ratio_qty", 1), position_intent=l.get("position_intent", ""))
            for l in legs
        ]

    @staticmethod
    def _reverse_leg(leg: Any) -> OrderLeg:
        symbol = _leg_field(leg, "symbol")
        ratio_qty = _leg_field(leg, "ratio_qty") or 1
        intent = _leg_field(leg, "position_intent")
        side = _leg_field(leg, "side")
        if intent == "buy_to_open" or (not intent and side == "buy"):
            return OrderLeg(symbol=symbol, side="sell", ratio_qty=ratio_qty, position_intent="sell_to_close")
        return OrderLeg(symbol=symbol, side="buy", ratio_qty=ratio_qty, position_intent="buy_to_close")

    @staticmethod
    def _net_bid_ask(legs: list, quotes: dict) -> tuple[Optional[float], Optional[float]]:
        """Net bid/ask for the position (single leg, or long-minus-short for
        a vertical). Works for BOTH entry legs (buy_to_open/sell_to_open) and
        the reversed exit legs (sell_to_close/buy_to_close) -- "long"/"short"
        here means which side of the spread we're long, not which order side
        we're about to submit."""
        if not legs:
            return None, None
        if len(legs) == 1:
            q = quotes.get(_leg_field(legs[0], "symbol"))
            if q is None:
                return None, None
            return q.bid, q.ask

        def _is_long(l):
            return _leg_field(l, "position_intent") in ("buy_to_open", "sell_to_close")

        def _is_short(l):
            return _leg_field(l, "position_intent") in ("sell_to_open", "buy_to_close")

        long_leg = next((l for l in legs if _is_long(l)), None)
        short_leg = next((l for l in legs if _is_short(l)), None)
        if long_leg is None or short_leg is None:
            return None, None
        ql = quotes.get(_leg_field(long_leg, "symbol"))
        qs = quotes.get(_leg_field(short_leg, "symbol"))
        if ql is None or qs is None or ql.bid is None or ql.ask is None or qs.bid is None or qs.ask is None:
            return None, None
        return (ql.bid - qs.ask), (ql.ask - qs.bid)

    @staticmethod
    def _fill_from_order(order: dict, purpose: str) -> dict:
        return {
            "purpose": purpose,
            "filled_qty": order.get("filled_qty") or 0,
            "avg": order.get("filled_avg_price"),
            "status": order.get("status", ""),
        }

    async def _poll_chain(self, pos: PositionRecord, chain: list[str]) -> dict[str, dict]:
        """Poll the latest cid in the chain, plus any earlier one whose
        last-known status wasn't terminal (a cancel that may still be in
        flight). Returns {cid: broker_order_dict} for everything that came
        back non-None."""
        if not chain:
            return {}
        to_poll = {chain[-1]}
        for cid in chain[:-1]:
            prior = pos.order_fills.get(cid)
            if prior is None or not _is_terminal(prior.get("status", "")):
                to_poll.add(cid)
        polled: dict[str, dict] = {}
        for cid in to_poll:
            order = await self._broker.get_order_by_client_id(cid)
            if order is not None:
                polled[cid] = order
        return polled

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def open_position(
        self,
        choice: ContractChoice,
        qty: int,
        signal: Signal,
        jev: JevVerdict,
        now_epoch: float,
        *,
        judgment: Optional[dict] = None,
    ) -> Optional[PositionRecord]:
        """`judgment` (judgment layer, optional): {"thesis": str, "invalidation": str,
        "entry_decision": dict, "next_review_at": float|None} from the decision
        model's EntryDecision, set on the PositionRecord from its very first
        (pending) write. Omitted/None leaves those fields at their dataclass
        defaults ("" / "" / {} / None) -- existing callers are unaffected."""
        if qty <= 0:
            logger.warning("DPOF orders: open_position called with qty<=0; refusing")
            return None
        if not choice.legs:
            logger.error("DPOF orders: open_position called with a ContractChoice that has no legs; refusing")
            return None

        position_id = str(uuid.uuid4())
        client_order_id = self.new_client_order_id("entry")
        direction_val = choice.direction.value if hasattr(choice.direction, "value") else str(choice.direction)
        j = judgment or {}

        pos = PositionRecord(
            id=position_id,
            underlying=choice.underlying,
            direction=direction_val,
            structure=choice.structure,
            legs=[self._leg_to_dict(l) for l in choice.legs],
            status="pending",
            qty_original=qty,
            qty_open=0,
            entry_signal=signal.to_dict(),
            entry_jev=jev.to_dict(),
            entry_client_order_id=client_order_id,
            entry_client_order_ids=[client_order_id],
            working_since=now_epoch,
            reprices=0,
            cancel_requested=False,
            mode=self._cfg.mode.value,
            thesis=j.get("thesis", ""),
            invalidation=j.get("invalidation", ""),
            entry_decision=j.get("entry_decision") or {},
            next_review_at=j.get("next_review_at"),
        )

        # Persist BEFORE touching the broker. If the process (or just this
        # coroutine) dies right after, the ledger already has enough
        # (status="pending" + entry_client_order_ids) for a fresh
        # OrderManager to rediscover the order via work_orders() -- no
        # in-memory cache is needed or used anywhere in this module.
        await self._ledger.upsert(pos)

        limit_price = entry_limit_price(choice.est_bid, choice.est_ask, self._cfg)
        if limit_price is None:
            logger.warning(f"DPOF orders: no usable entry quote for {position_id}; left pending, resolved by the missing-order timeout")
            return pos
        pos.entry_limit_price = limit_price  # prices this PENDING position against the open-premium cap

        intent = OrderIntent(
            client_order_id=client_order_id,
            legs=choice.legs,
            qty=qty,
            limit_price=limit_price,
            purpose="entry",
            reason="signal_entry",
            position_id=position_id,
        )

        async def _do_submit_and_record():
            result = await self._broker.submit(intent)
            if result.would_submit:
                pos.status = "cancelled"
                pos.exit_reason = "log_only"
                pos.closed_time = now_epoch
                pos.working_since = None
                await self._ledger.upsert(pos)
            elif not result.ok:
                pos.status = "cancelled"
                pos.exit_reason = f"submit_failed:{result.error}"[:200]
                pos.closed_time = now_epoch
                pos.working_since = None
                await self._ledger.upsert(pos)
            # A genuine success needs no further write here -- fills are
            # only ever recorded through the poll path (work_orders), never
            # from the submit() result itself.
            return result

        await asyncio.shield(_do_submit_and_record())
        return pos

    async def _work_entry(self, pos: PositionRecord, now_epoch: float) -> list[dict]:
        chain = pos.entry_client_order_ids
        if not chain:
            return []

        polled = await self._poll_chain(pos, chain)
        latest_cid = chain[-1]
        latest_order = polled.get(latest_cid)

        dirty = False
        for cid, order in polled.items():
            fill = self._fill_from_order(order, "entry")
            if pos.order_fills.get(cid) != fill:
                pos.order_fills[cid] = fill
                dirty = True

        if dirty:
            self._apply_recompute(pos, now_epoch)

        if latest_order is None:
            if dirty:
                await asyncio.shield(self._ledger.upsert(pos))
            return await self._entry_missing_or_wait(pos, now_epoch)

        status = latest_order.get("status", "")

        if _is_terminal(status):
            pos.cancel_requested = False
            pos.cancel_requested_at = None
            pos.cancel_attempts = 0
            remaining = max(0, pos.qty_original - pos.qty_open)
            return await self._advance_entry(pos, remaining, now_epoch, order_status=status)

        # Still working.
        if dirty:
            await asyncio.shield(self._ledger.upsert(pos))

        return await self._cancel_step(pos, latest_order, latest_cid, now_epoch, self._cfg.entry_reprice_seconds, "entry")

    async def _cancel_step(
        self, pos: PositionRecord, latest_order: dict, latest_cid: str, now_epoch: float, interval: float, purpose: str
    ) -> list[dict]:
        """Stale working order -> request its cancel; a requested cancel that has not
        turned the order terminal within `interval` is RE-ISSUED (cancels are
        idempotent). A failed cancel call never sets cancel_requested, so it is simply
        retried next cycle. Review finding (pass 2): the old code set cancel_requested
        unconditionally and then waited forever — one failed/lost cancel wedged the
        position in "closing", invisible to every exit rule, even across restarts."""
        if pos.cancel_requested:
            since = now_epoch - (pos.cancel_requested_at if pos.cancel_requested_at is not None else now_epoch)
            if since < interval:
                return []
        else:
            elapsed = now_epoch - (pos.working_since if pos.working_since is not None else now_epoch)
            if elapsed < interval:
                return []

        ok = await self._broker.cancel_order(latest_order.get("id", ""), latest_cid)
        pos.cancel_attempts += 1
        events: list[dict] = []
        if ok:
            kind = "cancel_retried" if pos.cancel_requested else "cancel_requested"
            pos.cancel_requested = True
            pos.cancel_requested_at = now_epoch
            events.append({"type": f"{purpose}_{kind}", "position_id": pos.id, "attempts": pos.cancel_attempts})
        else:
            logger.warning(
                f"DPOF orders: cancel of {latest_cid} for {pos.id} failed (attempt {pos.cancel_attempts}); retrying next cycle"
            )
            events.append({"type": f"{purpose}_cancel_failed", "position_id": pos.id, "attempts": pos.cancel_attempts})
        if pos.cancel_attempts >= _CANCEL_STUCK_ATTEMPTS and pos.cancel_attempts % _CANCEL_STUCK_ATTEMPTS == 0:
            logger.error(f"DPOF orders: {purpose} order {latest_cid} for {pos.id} still not cancelled after {pos.cancel_attempts} attempts")
            events.append({"type": "cancel_stuck", "position_id": pos.id, "purpose": purpose, "attempts": pos.cancel_attempts})
        await asyncio.shield(self._ledger.upsert(pos))
        return events

    @staticmethod
    def _apply_recompute(pos: PositionRecord, now_epoch: float) -> None:
        recomputed = recompute_from_fills(pos)
        pos.entry_price = recomputed.entry_price
        pos.qty_open = recomputed.qty_open
        pos.realized_pnl = recomputed.realized_pnl
        if pos.qty_open > 0 and pos.entry_time is None:
            pos.entry_time = now_epoch

    async def _entry_missing_or_wait(self, pos: PositionRecord, now_epoch: float) -> list[dict]:
        anchor = pos.working_since if pos.working_since is not None else now_epoch
        elapsed = now_epoch - anchor
        if elapsed < self._cfg.entry_reprice_seconds:
            return []
        pos.cancel_requested = False
        pos.working_since = None
        if pos.qty_open > 0:
            pos.status = "open"
            await asyncio.shield(self._ledger.upsert(pos))
            return [{
                "type": "entry_partial_then_submit_unconfirmed",
                "position_id": pos.id,
                "qty": pos.qty_open,
                "reason": "submit_unconfirmed",
            }]
        pos.status = "cancelled"
        pos.exit_reason = "submit_unconfirmed"
        pos.closed_time = now_epoch
        await asyncio.shield(self._ledger.upsert(pos))
        return [{"type": "entry_cancelled", "position_id": pos.id, "reason": "submit_unconfirmed"}]

    async def _advance_entry(self, pos: PositionRecord, remaining: int, now_epoch: float, order_status: str = "unknown") -> list[dict]:
        pos.cancel_requested = False

        if remaining <= 0:
            pos.status = "open"
            pos.working_since = None
            pos.reprices = 0
            await asyncio.shield(self._ledger.upsert(pos))
            return [{"type": "entry_filled", "position_id": pos.id, "qty": pos.qty_open}]

        if pos.reprices >= self._cfg.entry_max_reprices:
            pos.working_since = None
            # A pre-existing exit_reason (set by flatten_entry() when it
            # force-exhausted the budget at EOD, before this branch ever
            # ran) is preserved rather than overwritten -- so the ledger/
            # alert/decision-log trail reads "eod_flatten", not the
            # generic/misleading "entry_max_reprices", for a flatten-forced
            # finalize. Organic reprice exhaustion (the only case before
            # flatten_entry existed) always reaches here with exit_reason
            # still "" and is unaffected.
            reason = pos.exit_reason or "entry_max_reprices"
            if pos.qty_open > 0:
                pos.status = "open"
                await asyncio.shield(self._ledger.upsert(pos))
                return [{
                    "type": f"entry_partial_then_{order_status}",
                    "position_id": pos.id,
                    "qty": pos.qty_open,
                    "reason": reason,
                }]
            pos.status = "cancelled"
            pos.exit_reason = reason
            pos.closed_time = now_epoch
            await asyncio.shield(self._ledger.upsert(pos))
            return [{"type": "entry_cancelled", "position_id": pos.id, "reason": reason}]

        symbols = [l["symbol"] for l in pos.legs]
        quotes = await self._broker.get_option_quotes(symbols) or {}
        net_bid, net_ask = self._net_bid_ask(pos.legs, quotes)
        price = reprice_entry(net_bid, net_ask, pos.reprices + 1, self._cfg)
        if price is None:
            # Can't reprice right now (no usable quote) -- try again next
            # cycle. cancel_requested is already cleared above and persisted
            # fill state (if any) was already saved by the caller.
            return []

        new_cid = self.new_client_order_id("entry")
        pos.entry_client_order_ids = list(pos.entry_client_order_ids) + [new_cid]
        pos.entry_client_order_id = new_cid
        pos.reprices += 1
        pos.working_since = now_epoch
        pos.entry_limit_price = price
        await self._ledger.upsert(pos)  # persist BEFORE submit

        intent = OrderIntent(
            client_order_id=new_cid,
            legs=self._legs_from_dicts(pos.legs),
            qty=remaining,
            limit_price=price,
            purpose="entry",
            reason="reprice",
            position_id=pos.id,
        )

        async def _do_submit():
            await self._broker.submit(intent)

        await asyncio.shield(_do_submit())
        return [{"type": "entry_repriced", "position_id": pos.id, "price": price}]

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    async def close_position(self, pos: PositionRecord, qty: int, reason: str, urgent: bool, now_epoch: float) -> bool:
        if pos.status == "closing":
            logger.info(f"DPOF orders: position {pos.id} already has a working exit batch; refusing duplicate close")
            return False
        if pos.status != "open":
            logger.warning(f"DPOF orders: close_position called on {pos.id} with status={pos.status}; refusing")
            return False
        if qty <= 0 or qty > pos.qty_open:
            logger.warning(f"DPOF orders: close_position invalid qty={qty} for {pos.id} (qty_open={pos.qty_open})")
            return False

        exited_before = sum((f.get("filled_qty") or 0) for f in pos.order_fills.values() if f.get("purpose") == "exit")

        pos.status = "closing"
        pos.exit_target_qty = qty
        pos.exit_filled_baseline = exited_before
        pos.exit_urgent = urgent
        pos.reprices = 0
        pos.working_since = now_epoch
        pos.cancel_requested = False
        pos.exit_reason = reason

        legs = [self._reverse_leg(l) for l in pos.legs]
        symbols = [l.symbol for l in legs]
        quotes = await self._broker.get_option_quotes(symbols) or {}
        net_bid, net_ask = self._net_bid_ask(legs, quotes)
        limit_price = exit_limit_price(net_bid, net_ask, self._cfg)

        if limit_price is None:
            if urgent and len(legs) == 1:
                # No usable quote but this is urgent and single-leg: go
                # straight to market rather than wait a full reprice cycle.
                new_cid = self.new_client_order_id("exit")
                pos.exit_client_order_ids = list(pos.exit_client_order_ids) + [new_cid]
                await self._ledger.upsert(pos)  # persist BEFORE submit
                intent = OrderIntent(
                    client_order_id=new_cid, legs=legs, qty=qty, limit_price=None,
                    purpose="exit", reason=reason, position_id=pos.id, position_mode=pos.mode,
                )

                async def _do_submit_market():
                    await self._broker.submit(intent)

                await asyncio.shield(_do_submit_market())
                return True

            # Leave the batch pending with no order sent -- no dead end:
            # work_orders()'s missing-order path resends once
            # exit_reprice_seconds has elapsed (working_since is already
            # persisted above).
            await self._ledger.upsert(pos)
            logger.warning(f"DPOF orders: no usable exit quote for {pos.id}; batch pending, retried next cycle")
            return False

        new_cid = self.new_client_order_id("exit")
        pos.exit_client_order_ids = list(pos.exit_client_order_ids) + [new_cid]
        await self._ledger.upsert(pos)  # persist BEFORE submit

        intent = OrderIntent(
            client_order_id=new_cid, legs=legs, qty=qty, limit_price=limit_price,
            purpose="exit", reason=reason, position_id=pos.id, position_mode=pos.mode,
        )

        async def _do_submit():
            await self._broker.submit(intent)

        await asyncio.shield(_do_submit())
        return True

    async def _work_exit(self, pos: PositionRecord, now_epoch: float) -> list[dict]:
        chain = pos.exit_client_order_ids

        if not chain:
            return await self._exit_missing_or_wait(pos, now_epoch)

        polled = await self._poll_chain(pos, chain)
        latest_cid = chain[-1]
        latest_order = polled.get(latest_cid)

        dirty = False
        for cid, order in polled.items():
            fill = self._fill_from_order(order, "exit")
            if pos.order_fills.get(cid) != fill:
                pos.order_fills[cid] = fill
                dirty = True

        if dirty:
            self._apply_recompute(pos, now_epoch)

        if latest_order is None:
            if dirty:
                await asyncio.shield(self._ledger.upsert(pos))
            return await self._exit_missing_or_wait(pos, now_epoch)

        status = latest_order.get("status", "")

        if _is_terminal(status):
            pos.cancel_requested = False
            pos.cancel_requested_at = None
            pos.cancel_attempts = 0
            exited_now = sum((f.get("filled_qty") or 0) for f in pos.order_fills.values() if f.get("purpose") == "exit")
            batch_closed = exited_now - pos.exit_filled_baseline
            remaining = max(0, pos.exit_target_qty - batch_closed)
            return await self._advance_exit(pos, remaining, now_epoch, missing=False)

        # Still working.
        if dirty:
            await asyncio.shield(self._ledger.upsert(pos))

        return await self._cancel_step(pos, latest_order, latest_cid, now_epoch, self._cfg.exit_reprice_seconds, "exit")

    async def _exit_missing_or_wait(self, pos: PositionRecord, now_epoch: float) -> list[dict]:
        anchor = pos.working_since if pos.working_since is not None else now_epoch
        elapsed = now_epoch - anchor
        if elapsed < self._cfg.exit_reprice_seconds:
            return []
        # Never reached the broker (or the ONLY earlier order in the chain
        # is already terminal and unaccounted-for qty remains) -- exits are
        # risk-reducing, so always resend rather than give up.
        exited_now = sum((f.get("filled_qty") or 0) for f in pos.order_fills.values() if f.get("purpose") == "exit")
        batch_closed = exited_now - pos.exit_filled_baseline
        remaining = max(0, pos.exit_target_qty - batch_closed)
        return await self._advance_exit(pos, remaining, now_epoch, missing=True)

    async def _advance_exit(self, pos: PositionRecord, remaining: int, now_epoch: float, missing: bool) -> list[dict]:
        pos.cancel_requested = False

        if remaining <= 0:
            batch_pnl = _batch_realized_pnl(pos)
            exited_now = sum((f.get("filled_qty") or 0) for f in pos.order_fills.values() if f.get("purpose") == "exit")
            batch_qty = max(0, exited_now - pos.exit_filled_baseline)
            pos.exit_target_qty = 0
            pos.exit_filled_baseline = 0
            pos.exit_urgent = False
            pos.working_since = None
            pos.reprices = 0
            closed = pos.qty_open <= 0
            pos.status = "closed" if closed else "open"
            if closed:
                pos.closed_time = now_epoch
            await asyncio.shield(self._ledger.upsert(pos))
            return [{
                "type": "exit_filled",
                "position_id": pos.id,
                "qty": batch_qty,
                "realized_delta": batch_pnl,
                "closed": closed,
            }]

        legs = [self._reverse_leg(l) for l in pos.legs]
        is_single = len(legs) == 1

        # Missing-order resends never spend a reprice (the order literally
        # never existed at the broker -- it isn't "another attempt at the
        # same price ladder", it's "the first attempt never landed").
        if not missing and pos.reprices >= self._cfg.exit_max_reprices:
            symbols = [l.symbol for l in legs]
            quotes = await self._broker.get_option_quotes(symbols) or {}
            net_bid, net_ask = self._net_bid_ask(legs, quotes)

            if pos.exit_urgent and is_single:
                new_cid = self.new_client_order_id("exit")
                pos.exit_client_order_ids = list(pos.exit_client_order_ids) + [new_cid]
                pos.working_since = now_epoch
                await self._ledger.upsert(pos)
                intent = OrderIntent(
                    client_order_id=new_cid, legs=legs, qty=remaining, limit_price=None,
                    purpose="exit", reason=pos.exit_reason, position_id=pos.id, position_mode=pos.mode,
                )

                async def _do_submit_market():
                    await self._broker.submit(intent)

                await asyncio.shield(_do_submit_market())
                return [{"type": "exit_escalated_market", "position_id": pos.id}]

            # mleg (or non-urgent single-leg) NEVER gets a bare market order
            # -- marketable limit at the natural bid instead.
            if net_bid is None:
                await asyncio.shield(self._ledger.upsert(pos))
                return []
            price = max(0.01, round(net_bid, 2))
            new_cid = self.new_client_order_id("exit")
            pos.exit_client_order_ids = list(pos.exit_client_order_ids) + [new_cid]
            pos.working_since = now_epoch
            await self._ledger.upsert(pos)
            intent = OrderIntent(
                client_order_id=new_cid, legs=legs, qty=remaining, limit_price=price,
                purpose="exit", reason=pos.exit_reason, position_id=pos.id, position_mode=pos.mode,
            )

            async def _do_submit_marketable():
                await self._broker.submit(intent)

            await asyncio.shield(_do_submit_marketable())
            return [{"type": "exit_escalated_marketable", "position_id": pos.id, "price": price}]

        # Normal resend at the next repriced limit -- covers both "budget
        # still available" and "missing order -> resend" (which passes
        # reprices+1 too, but doesn't consume the budget; see below).
        symbols = [l.symbol for l in legs]
        quotes = await self._broker.get_option_quotes(symbols) or {}
        net_bid, net_ask = self._net_bid_ask(legs, quotes)
        price = reprice_exit(net_bid, net_ask, pos.reprices + 1, self._cfg)
        if price is None:
            await asyncio.shield(self._ledger.upsert(pos))
            return []

        new_cid = self.new_client_order_id("exit")
        pos.exit_client_order_ids = list(pos.exit_client_order_ids) + [new_cid]
        if not missing:
            pos.reprices += 1
        pos.working_since = now_epoch
        await self._ledger.upsert(pos)  # persist BEFORE submit

        intent = OrderIntent(
            client_order_id=new_cid, legs=legs, qty=remaining, limit_price=price,
            purpose="exit", reason=pos.exit_reason, position_id=pos.id, position_mode=pos.mode,
        )

        async def _do_submit():
            await self._broker.submit(intent)

        await asyncio.shield(_do_submit())
        return [{"type": "exit_resent" if missing else "exit_repriced", "position_id": pos.id, "price": price}]

    # ------------------------------------------------------------------
    # EOD flatten (engine.py, position-review pass) — the one kept rail
    # ------------------------------------------------------------------

    async def flatten_entry(self, pos: PositionRecord, reason: str, now_epoch: float) -> bool:
        """EOD-flatten path ONLY: permanently abandons a still-PENDING entry
        order instead of leaving it to the ordinary reprice ladder.

        Review finding (pass 2, order-safety lens): engine.py's position-
        review loop only ever flattens ``status == "open"`` positions -- a
        PENDING entry (a working, unfilled limit order) has no fill yet to
        sell, so it was never covered by the flatten rail at all. Left
        alone, the ordinary reprice ladder (``_work_entry``/``_advance_entry``)
        has NO wall-clock awareness of ``flatten_et`` and will happily submit
        *another* reprice order once the current one resolves, as long as
        ``entry_max_reprices`` isn't exhausted yet -- so a pending entry
        could still be working, or even fill, after 15:50 ET. Once the
        market closes, engine.py stops calling ``work_orders()`` at all (see
        its step-4 market-hours gate), so a late fill would sit undiscovered
        in the ledger as ``status="pending"`` until the next session --
        effectively a 1DTE contract held overnight with nobody managing it.

        Fix: (1) exhaust the reprice budget right now, so the NEXT poll's
        ``_advance_entry`` takes its existing "budget exhausted" branch
        (finalize to ``open`` if partially filled, ``cancelled`` otherwise)
        instead of resubmitting -- this reuses tested, already-epoch-only
        logic rather than teaching ``orders.py`` any wall-clock ET awareness
        of its own; (2) force-cancel the current working order immediately
        rather than waiting out the normal ``entry_reprice_seconds``
        staleness window, narrowing the fill race.

        The reprice budget is exhausted unconditionally (first line below),
        so the ladder can never resubmit for this position again today no
        matter what happens next. The return value only reports whether a
        NEW cancel was issued THIS call -- it is False (a no-op past that
        point) when a cancel is already in flight (the ordinary retry loop
        in ``_cancel_step`` owns that cadence) or the latest order is
        already terminal (the next ``work_orders()`` poll finalizes it as
        usual).
        """
        if pos.status != "pending":
            return False
        # Exhaust the reprice budget FIRST and unconditionally -- even if
        # the cancel below fails or a lookup errors, the position can never
        # be resubmitted into a fresh working order again today.
        pos.reprices = max(pos.reprices, self._cfg.entry_max_reprices)
        if not pos.exit_reason:
            pos.exit_reason = reason
        chain = pos.entry_client_order_ids
        if not chain or pos.cancel_requested:
            await asyncio.shield(self._ledger.upsert(pos))
            return False
        latest_cid = chain[-1]
        try:
            order = await self._broker.get_order_by_client_id(latest_cid)
        except BrokerLookupError as e:
            logger.warning(
                f"DPOF orders: eod-flatten order lookup failed for {pos.id} ({e}); "
                f"reprice budget already exhausted, will finalize once resolved"
            )
            await asyncio.shield(self._ledger.upsert(pos))
            return False
        if order is None or _is_terminal(order.get("status", "")):
            await asyncio.shield(self._ledger.upsert(pos))
            return False
        ok = await self._broker.cancel_order(order.get("id", ""), latest_cid)
        pos.cancel_attempts += 1
        if ok:
            pos.cancel_requested = True
            pos.cancel_requested_at = now_epoch
        else:
            logger.warning(
                f"DPOF orders: eod-flatten cancel of pending entry {latest_cid} for {pos.id} "
                f"failed (attempt {pos.cancel_attempts}); reprice budget is exhausted regardless, "
                f"so the position will finalize (not resubmit) once the order resolves"
            )
        await asyncio.shield(self._ledger.upsert(pos))
        return ok

    # ------------------------------------------------------------------
    # Cycle entrypoint
    # ------------------------------------------------------------------

    async def work_orders(self, now_epoch: float) -> list[dict]:
        events: list[dict] = []
        for pos in list(self._ledger.open_positions()):
            try:
                if pos.status == "pending":
                    events.extend(await self._work_entry(pos, now_epoch))
                elif pos.status == "closing":
                    events.extend(await self._work_exit(pos, now_epoch))
                # "open" positions have no working order -- nothing to poll.
            except BrokerLookupError as e:
                # Couldn't learn the order's state: change NOTHING this cycle (no
                # resend, no cancel, no give-up). Resending on a lookup failure could
                # double an exit or orphan a live entry.
                logger.warning(f"DPOF orders: order state unknown for {pos.id} ({e}); retrying next cycle")
                events.append({"type": "lookup_error", "position_id": pos.id, "reason": str(e)})
        return events

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def reconcile(self, now_epoch: Optional[float] = None) -> list[str]:
        warnings: list[str] = []
        now = now_epoch if now_epoch is not None else time.time()

        broker_positions = await self._broker.list_option_positions()
        if broker_positions is None:
            warnings.append("reconcile_broker_unavailable")
            return warnings

        broker_symbols = {p["symbol"] for p in broker_positions if abs(p.get("qty") or 0) > 0}

        ledger_open_symbols: set[str] = set()
        for pos in self._ledger.open_positions():
            if pos.status not in ("open", "closing"):
                continue
            pos_symbols = {l["symbol"] for l in pos.legs}
            ledger_open_symbols |= pos_symbols

            if pos.cancel_requested:
                continue  # a cancel is in flight; let it resolve first

            chain = pos.exit_client_order_ids if pos.status == "closing" else pos.entry_client_order_ids
            if chain:
                latest_fill = pos.order_fills.get(chain[-1])
                if latest_fill is not None and not _is_terminal(latest_fill.get("status", "")):
                    continue  # an order is still live; not an anomaly yet

            if pos_symbols & broker_symbols:
                continue

            logger.warning(
                f"DPOF orders: position {pos.id} ({pos.underlying}) open in ledger but absent at broker; "
                f"marking closed (reconciled_absent)"
            )
            pos.status = "closed"
            if not pos.exit_reason:
                pos.exit_reason = "reconciled_absent"
            if pos.closed_time is None:
                pos.closed_time = time.time()
            await asyncio.shield(self._ledger.upsert(pos))
            warnings.append(f"reconciled_absent:{pos.id}")

        for symbol in sorted(broker_symbols - ledger_open_symbols):
            underlying = _underlying_of(symbol)
            if underlying and underlying in self._cfg.underlyings:
                logger.warning(f"DPOF orders: orphan broker position on our underlyings, not in ledger: {symbol}")
                warnings.append(f"orphan:{symbol}")

        # ------------------------------------------------------------------
        # Quantity check + single-leg synthetic-fill correction (docs/design/
        # clio_dpof_followups_2026-09-26.md §3.2 points 1-6). Runs on the
        # ledger AS IT STANDS after the presence-only checks above -- a
        # position just marked reconciled_absent is status="closed" now and
        # drops out of the open/closing universe below, so it can never both
        # get closed above AND quantity-corrected here in the same pass.
        # ------------------------------------------------------------------
        broker_qty: dict[str, int] = {
            p["symbol"]: int(p.get("qty") or 0) for p in broker_positions if abs(p.get("qty") or 0) > 0
        }

        ledger_qty: dict[str, int] = {}
        symbol_positions: dict[str, dict[str, PositionRecord]] = {}
        for pos in self._ledger.open_positions():
            if pos.status not in ("open", "closing"):
                continue
            for leg in pos.legs:
                sym = _leg_field(leg, "symbol")
                if not sym:
                    continue
                ratio = _leg_field(leg, "ratio_qty") or 1
                sign = 1 if _leg_field(leg, "side") == "buy" else -1
                ledger_qty[sym] = ledger_qty.get(sym, 0) + pos.qty_open * ratio * sign
                symbol_positions.setdefault(sym, {})[pos.id] = pos

        mismatched = sorted(s for s in ledger_qty if s in broker_qty and ledger_qty[s] != broker_qty[s])
        seen_this_cycle: set[str] = set()

        for symbol in mismatched:
            seen_this_cycle.add(symbol)
            holding = list(symbol_positions.get(symbol, {}).values())
            l = ledger_qty[symbol]
            b = broker_qty[symbol]

            in_flight = any(p.status != "open" or p.cancel_requested for p in holding)
            if in_flight:
                # The manager is still working an entry/exit or a cancel on
                # this contract (states that end on their own) -- that
                # plausibly explains the mismatch for now: no warning, and
                # drop any pending confirm clock (it would otherwise resume
                # against a stale (ledger,broker) reading).
                self._pending_qty_mismatch.pop(symbol, None)
                continue

            warnings.append(f"qty_mismatch:{symbol}:ledger={l}:broker={b}")

            leg0 = holding[0].legs[0] if (len(holding) == 1 and holding[0].legs) else None
            adopt_eligible = bool(
                self._cfg.reconcile_adopt_qty
                and len(holding) == 1
                and holding[0].structure == "single"
                and len(holding[0].legs) == 1
                and leg0 is not None
                and _leg_field(leg0, "side") == "buy"
                and (_leg_field(leg0, "ratio_qty") or 1) == 1
                and b > 0
                and not _has_live_recorded_order(holding[0])
            )
            if not adopt_eligible:
                self._pending_qty_mismatch.pop(symbol, None)
                continue

            prior = self._pending_qty_mismatch.get(symbol)
            if prior is not None and prior.get("ledger") == l and prior.get("broker") == b:
                if (now - prior.get("first_seen", now)) >= self._cfg.reconcile_confirm_seconds:
                    pos = holding[0]
                    delta = b - l
                    base_key = f"reconcile:{int(now)}:{symbol}"
                    key = base_key
                    n = 0
                    while key in pos.order_fills:
                        n += 1
                        key = f"{base_key}:{n}"
                    pos.order_fills[key] = {
                        "purpose": _RECONCILE_ADD if delta > 0 else _RECONCILE_REMOVE,
                        "filled_qty": abs(delta),
                        "avg": None,
                        "status": "reconciled",
                        "broker_qty": b,
                        "ledger_qty": l,
                        "at": now,
                    }
                    self._apply_recompute(pos, now)
                    await asyncio.shield(self._ledger.upsert(pos))
                    logger.warning(
                        f"DPOF orders: reconcile adopted quantity for {pos.id} ({symbol}): "
                        f"ledger {l} -> broker {b} (delta {delta:+d})"
                    )
                    warnings.append(f"qty_adopted:{pos.id}:{symbol}:{l}->{b}")
                    self._pending_qty_mismatch.pop(symbol, None)
                # else: same (ledger,broker) pair seen again, but not yet
                # reconcile_confirm_seconds apart -- keep waiting.
            else:
                # First sighting of this symbol's mismatch, or the
                # (ledger,broker) reading changed since the last sighting --
                # (re)start the confirm clock.
                self._pending_qty_mismatch[symbol] = {"ledger": l, "broker": b, "first_seen": now}

        for stale_symbol in [s for s in self._pending_qty_mismatch if s not in seen_this_cycle]:
            self._pending_qty_mismatch.pop(stale_symbol, None)

        return warnings
