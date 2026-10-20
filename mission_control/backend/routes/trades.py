"""Trade log routes — chronological trade history."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from mission_control.backend.auth import require_auth
from mission_control.backend.deps import get_trading_client, run_in_executor

router = APIRouter(prefix="/api", tags=["trades"])
logger = logging.getLogger("mission_control.trades")


def _dpof_client_order_index() -> dict:
    """Map every entry/exit client_order_id -> its DPOF PositionRecord.

    Never raises — readmodel itself never raises, but this stays defensive
    since it also runs outside a request context in tests.
    """
    index: dict = {}
    try:
        from agent.dpof import readmodel

        for pos in readmodel.load_positions():
            for coid in (pos.entry_client_order_ids or []):
                if coid:
                    index[str(coid)] = pos
            for coid in (pos.exit_client_order_ids or []):
                if coid:
                    index[str(coid)] = pos
    except Exception as e:
        logger.debug(f"DPOF client-order index unavailable: {e}")
    return index


@router.get("/trades")
async def get_trades(
    limit: int = Query(50, ge=1, le=500),
    symbol: Optional[str] = None,
    _=Depends(require_auth),
):
    """Chronological trade log — Alpaca fills enriched with DPOF context."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = get_trading_client()

        req_params = {"status": QueryOrderStatus.CLOSED, "limit": limit}
        if symbol:
            req_params["symbols"] = [symbol.upper()]
        orders = await run_in_executor(
            lambda: client.get_orders(GetOrdersRequest(**req_params))
        )

        # DPOF is CLIO's own book — map order client_order_id -> PositionRecord.
        coid_index = await run_in_executor(_dpof_client_order_index)

        trades = []
        for o in orders:
            if not o.filled_at:
                continue
            trade = {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": o.side.value if o.side else "",
                "qty": str(o.qty),
                "filled_qty": str(o.filled_qty or 0),
                "filled_avg_price": str(o.filled_avg_price) if o.filled_avg_price else None,
                "order_type": o.order_type.value if o.order_type else "",
                "filled_at": str(o.filled_at),
                "submitted_at": str(o.submitted_at) if o.submitted_at else None,
                "client_order_id": str(o.client_order_id) if o.client_order_id else None,
                "extended_hours": bool(o.extended_hours),
            }
            # Enrich from DPOF's own ledger if this order belongs to a DPOF position.
            coid = str(o.client_order_id) if o.client_order_id else None
            pos = coid_index.get(coid) if coid else None
            if pos is not None:
                trade["dpof_id"] = pos.id
                trade["underlying"] = pos.underlying
                trade["direction"] = pos.direction
                trade["structure"] = pos.structure
                trade["is_exit"] = coid in (pos.exit_client_order_ids or [])
                trade["realized_pnl"] = pos.realized_pnl if pos.status == "closed" else None
            trades.append(trade)

        return {"trades": trades, "count": len(trades)}
    except Exception as e:
        logger.error(f"Trade log failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
