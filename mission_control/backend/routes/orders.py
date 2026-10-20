"""Order routes — submit, list, cancel orders."""

import logging
import re

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from typing import Optional

from mission_control.backend.auth import require_auth, require_trade_execution_enabled
from mission_control.backend.deps import get_trading_client, run_in_executor
from agent.integrations.alpaca_values import normalize_order_side

router = APIRouter(prefix="/api", tags=["orders"])
logger = logging.getLogger("mission_control.orders")

_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
_MAX_QTY = 10000


class OrderRequest(BaseModel):
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    order_type: str  # "market", "limit", "stop", "stop_limit"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    extended_hours: bool = False
    trade_id: Optional[str] = None


@router.get("/orders")
async def list_orders(
    status: str = Query("all", pattern="^(open|closed|all)$"),
    limit: int = Query(50, ge=1, le=200),
    _=Depends(require_auth),
):
    """List orders from Alpaca."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        status_map = {"open": QueryOrderStatus.OPEN, "closed": QueryOrderStatus.CLOSED, "all": QueryOrderStatus.ALL}
        client = get_trading_client()
        orders = await run_in_executor(
            lambda: client.get_orders(GetOrdersRequest(status=status_map[status], limit=limit))
        )
        return {"orders": [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": normalize_order_side(o.side),
                "qty": str(o.qty),
                "filled_qty": str(o.filled_qty or 0),
                "order_type": o.order_type.value if o.order_type else "",
                "status": o.status.value if o.status else "",
                "limit_price": str(o.limit_price) if o.limit_price else None,
                "stop_price": str(o.stop_price) if o.stop_price else None,
                "filled_avg_price": str(o.filled_avg_price) if o.filled_avg_price else None,
                "submitted_at": str(o.submitted_at) if o.submitted_at else None,
                "filled_at": str(o.filled_at) if o.filled_at else None,
                "time_in_force": o.time_in_force.value if o.time_in_force else "",
                "extended_hours": bool(o.extended_hours),
                "client_order_id": str(o.client_order_id) if o.client_order_id else None,
            }
            for o in orders
        ]}
    except Exception as e:
        logger.error(f"List orders failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/orders")
async def submit_order(
    req: OrderRequest,
    _=Depends(require_auth),
    _gate=Depends(require_trade_execution_enabled),
):
    """Submit a new order via Alpaca."""
    # Validation
    if not _TICKER_RE.match(req.symbol):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {req.symbol}")
    if req.side not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail=f"Invalid side: {req.side}")
    if not (0 < req.qty <= _MAX_QTY):
        raise HTTPException(status_code=400, detail=f"Invalid qty: {req.qty} (max {_MAX_QTY})")

    try:
        from alpaca.trading.requests import (
            MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce

        side = OrderSide.BUY if req.side == "buy" else OrderSide.SELL
        tif_map = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK}
        tif = TimeInForce.DAY if req.extended_hours else tif_map.get(req.time_in_force, TimeInForce.DAY)

        if req.order_type == "market":
            request = MarketOrderRequest(symbol=req.symbol, qty=req.qty, side=side, time_in_force=tif, client_order_id=req.trade_id)
        elif req.order_type == "limit":
            if req.limit_price is None:
                raise HTTPException(status_code=400, detail="limit_price required for limit orders")
            request = LimitOrderRequest(symbol=req.symbol, qty=req.qty, side=side, time_in_force=tif, limit_price=req.limit_price, extended_hours=req.extended_hours, client_order_id=req.trade_id)
        elif req.order_type == "stop":
            if req.stop_price is None:
                raise HTTPException(status_code=400, detail="stop_price required for stop orders")
            request = StopOrderRequest(symbol=req.symbol, qty=req.qty, side=side, time_in_force=tif, stop_price=req.stop_price, client_order_id=req.trade_id)
        elif req.order_type == "stop_limit":
            if req.limit_price is None or req.stop_price is None:
                raise HTTPException(status_code=400, detail="limit_price and stop_price required")
            request = StopLimitOrderRequest(symbol=req.symbol, qty=req.qty, side=side, time_in_force=tif, limit_price=req.limit_price, stop_price=req.stop_price, client_order_id=req.trade_id)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid order_type: {req.order_type}")

        client = get_trading_client()
        order = await run_in_executor(lambda: client.submit_order(request))

        logger.info(f"Order submitted: {order.id} {req.side} {req.qty} {req.symbol}")
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": normalize_order_side(order.side),
            "qty": str(order.qty),
            "order_type": order.order_type.value,
            "status": order.status.value,
            "submitted_at": str(order.submitted_at),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Order submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str, _=Depends(require_auth)):
    """Cancel a pending order."""
    try:
        client = get_trading_client()
        await run_in_executor(lambda: client.cancel_order_by_id(order_id))
        logger.info(f"Order cancelled: {order_id}")
        return {"cancelled": order_id}
    except Exception as e:
        logger.error(f"Cancel order failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
