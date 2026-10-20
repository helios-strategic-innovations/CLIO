"""Options routes — chain, quotes, orders."""

import logging
import math
import re

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from typing import Optional

from mission_control.backend.auth import require_auth, require_trade_execution_enabled
from mission_control.backend.deps import get_trading_client, run_in_executor
from agent.integrations.alpaca_values import normalize_order_side

router = APIRouter(prefix="/api/options", tags=["options"])
logger = logging.getLogger("mission_control.options")

_OCC_RE = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _timestamp_text(value) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _option_snapshot_fields(snapshot) -> dict:
    """Normalize a snapshot without turning absent market data into neutral zeroes."""
    quote = getattr(snapshot, "latest_quote", None) if snapshot else None
    trade = getattr(snapshot, "latest_trade", None) if snapshot else None
    greeks = getattr(snapshot, "greeks", None) if snapshot else None

    bid = _optional_float(getattr(quote, "bid_price", None))
    ask = _optional_float(getattr(quote, "ask_price", None))
    last = _optional_float(getattr(trade, "price", None))
    mid = round((bid + ask) / 2, 4) if bid is not None and ask is not None and bid > 0 and ask > 0 else last
    spread = round(ask - bid, 4) if bid is not None and ask is not None and bid > 0 and ask > 0 else None

    greek_fields = {
        name: _optional_float(getattr(greeks, name, None))
        for name in ("implied_volatility", "delta", "gamma", "theta", "vega", "rho")
    }
    reasons = []
    if snapshot is None:
        reasons.append("NO_SNAPSHOT")
    if quote is None:
        reasons.append("MISSING_QUOTE")
    if greeks is None:
        reasons.append("MISSING_GREEKS")
    elif any(greek_fields[name] is None for name in ("implied_volatility", "delta", "gamma", "theta", "vega")):
        reasons.append("PARTIAL_GREEKS")

    as_of = _timestamp_text(getattr(quote, "timestamp", None)) or _timestamp_text(
        getattr(trade, "timestamp", None)
    )
    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "last": last,
        "trade_size": _optional_int(getattr(trade, "size", None)),
        **greek_fields,
        "source": "alpaca_option_snapshot",
        "as_of": as_of,
        "stale": bool(reasons),
        "reason_code": reasons[0] if reasons else "FRESH",
    }


# ---------------------------------------------------------------------------
# Option chain + quotes
# ---------------------------------------------------------------------------

@router.get("/chain/{underlying}")
async def get_option_chain(
    underlying: str,
    expiration: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    option_type: str = Query("both", pattern="^(call|put|both)$"),
    strike_min: Optional[float] = None,
    strike_max: Optional[float] = None,
    limit: int = Query(30, ge=1, le=100),
    _=Depends(require_auth),
):
    """Fetch options chain for an underlying symbol."""
    underlying = underlying.upper()
    if not re.match(r"^[A-Z]{1,6}$", underlying):
        raise HTTPException(400, f"Invalid underlying: {underlying}")

    try:
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import AssetStatus
        from datetime import date

        kwargs: dict = {
            "underlying_symbols": [underlying],
            "status": AssetStatus.ACTIVE,
            "limit": limit,
        }
        if expiration:
            kwargs["expiration_date"] = expiration
        if option_type == "call":
            kwargs["type"] = "call"
        elif option_type == "put":
            kwargs["type"] = "put"
        if strike_min is not None:
            kwargs["strike_price_gte"] = str(strike_min)
        if strike_max is not None:
            kwargs["strike_price_lte"] = str(strike_max)

        client = get_trading_client()
        response = await run_in_executor(
            lambda: client.get_option_contracts(GetOptionContractsRequest(**kwargs))
        )
        # get_option_contracts returns response object — extract .option_contracts list
        contracts = getattr(response, "option_contracts", None) or []

        # Get snapshots for Greeks/bid/ask
        occ_symbols = [str(c.symbol) for c in contracts]
        snapshots = {}
        if occ_symbols:
            try:
                from alpaca.data.historical.option import OptionHistoricalDataClient
                from alpaca.data.requests import OptionSnapshotRequest
                from agent.config import config
                opt_client = OptionHistoricalDataClient(
                    api_key=config.alpaca_api_key, secret_key=config.alpaca_secret_key
                )
                snap_data = await run_in_executor(
                    lambda: opt_client.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=occ_symbols))
                )
                snapshots = snap_data or {}
            except Exception as e:
                logger.debug(f"Snapshot fetch failed: {e}")

        result = []
        for c in contracts:
            sym = str(c.symbol)
            snap = snapshots.get(sym)
            market = _option_snapshot_fields(snap)
            daily_bar = getattr(snap, "daily_bar", None) if snap else None

            entry = {
                "occ_symbol": sym,
                "underlying": underlying,
                "expiration": str(c.expiration_date) if c.expiration_date else None,
                "strike": _optional_float(getattr(c, "strike_price", None)),
                "option_type": str(c.type) if c.type else "",
                "volume": _optional_int(getattr(daily_bar, "volume", None)),
                "open_interest": _optional_int(getattr(c, "open_interest", None)),
            }
            market.pop("trade_size", None)
            market.pop("rho", None)
            entry.update(market)
            result.append(entry)

        return {
            "underlying": underlying,
            "contracts": result,
            "count": len(result),
            "source": "alpaca_option_snapshot",
            "stale": any(item["stale"] for item in result),
            "reason_code": "PARTIAL_MARKET_DATA" if any(item["stale"] for item in result) else "FRESH",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Option chain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/quote/{occ_symbol}")
async def get_option_quote(occ_symbol: str, _=Depends(require_auth)):
    """Get live quote + Greeks for a single option contract."""
    occ_symbol = occ_symbol.upper()
    if not _OCC_RE.match(occ_symbol):
        raise HTTPException(400, f"Invalid OCC symbol: {occ_symbol}")

    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionSnapshotRequest
        from agent.config import config

        opt_client = OptionHistoricalDataClient(
            api_key=config.alpaca_api_key, secret_key=config.alpaca_secret_key
        )
        snap_data = await run_in_executor(
            lambda: opt_client.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=[occ_symbol]))
        )
        snap = (snap_data or {}).get(occ_symbol)
        if not snap:
            missing = _option_snapshot_fields(None)
            missing["volume"] = missing.pop("trade_size")
            return {"error": "No snapshot data", "occ_symbol": occ_symbol, **missing}

        fields = _option_snapshot_fields(snap)
        fields["volume"] = fields.pop("trade_size")
        return {"occ_symbol": occ_symbol, **fields}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Option quote failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Options order submission
# ---------------------------------------------------------------------------

class OptionsOrderRequest(BaseModel):
    occ_symbol: str
    side: str  # "buy" or "sell"
    qty: int
    position_intent: str = "buy_to_open"
    limit_price: Optional[float] = None  # If None, compute mid-point
    spread_adjustment: float = 0.02
    time_in_force: str = "day"
    trade_id: Optional[str] = None


@router.post("/orders")
async def submit_options_order(
    req: OptionsOrderRequest,
    _=Depends(require_auth),
    _gate=Depends(require_trade_execution_enabled),
):
    """Submit an options order via Alpaca with mid-point limit pricing."""
    occ = req.occ_symbol.upper()
    if not _OCC_RE.match(occ):
        raise HTTPException(400, f"Invalid OCC symbol: {occ}")
    if req.side not in ("buy", "sell"):
        raise HTTPException(400, f"Invalid side: {req.side}")
    if not (0 < req.qty <= 100):
        raise HTTPException(400, f"Invalid qty: {req.qty} (max 100)")

    try:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        # Compute limit price from mid-point if not overridden
        limit_price = req.limit_price
        if limit_price is None:
            # Fetch live quote for mid-point
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.requests import OptionSnapshotRequest
            from agent.config import config

            opt_client = OptionHistoricalDataClient(
                api_key=config.alpaca_api_key, secret_key=config.alpaca_secret_key
            )
            snap_data = await run_in_executor(
                lambda: opt_client.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=[occ]))
            )
            snap = (snap_data or {}).get(occ)
            if not snap:
                raise HTTPException(400, "No quote data available for mid-point pricing")

            quote = getattr(snap, "latest_quote", None)
            bid = float(getattr(quote, "bid_price", 0) or 0)
            ask = float(getattr(quote, "ask_price", 0) or 0)
            if bid <= 0 or ask <= 0:
                raise HTTPException(400, f"Invalid bid/ask: {bid}/{ask}")
            spread = ask - bid
            mid = (bid + ask) / 2
            # Sanity: reject if spread > 50% of mid
            if mid > 0 and spread / mid > 0.50:
                raise HTTPException(400, f"Spread too wide: ${spread:.2f} ({spread / mid * 100:.0f}% of mid)")
            # Adjust toward taker side
            adj = spread * req.spread_adjustment
            if req.side == "buy":
                limit_price = round(mid + adj, 2)
            else:
                limit_price = round(mid - adj, 2)
            limit_price = max(0.01, limit_price)

        side = OrderSide.BUY if req.side == "buy" else OrderSide.SELL
        tif_map = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK}
        tif = tif_map.get(req.time_in_force, TimeInForce.DAY)

        # Map position intent
        _INTENT_MAP = {
            "buy_to_open": "buy_to_open", "bto": "buy_to_open",
            "buy_to_close": "buy_to_close", "btc": "buy_to_close",
            "sell_to_open": "sell_to_open", "sto": "sell_to_open",
            "sell_to_close": "sell_to_close", "stc": "sell_to_close",
        }
        intent = _INTENT_MAP.get(req.position_intent, "buy_to_open")

        order_kwargs = {
            "symbol": occ,
            "qty": req.qty,
            "side": side,
            "time_in_force": tif,
            "limit_price": limit_price,
        }
        if req.trade_id:
            order_kwargs["client_order_id"] = req.trade_id
        # Position intent — use enum if available, fall back to string
        try:
            from alpaca.trading.enums import PositionIntent
            order_kwargs["position_intent"] = PositionIntent(intent)
        except (ImportError, ValueError):
            order_kwargs["position_intent"] = intent

        request = LimitOrderRequest(**order_kwargs)
        client = get_trading_client()
        order = await run_in_executor(lambda: client.submit_order(request))

        logger.info(f"Options order: {order.id} {req.side} {req.qty}x {occ} @ ${limit_price}")
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": normalize_order_side(order.side),
            "qty": str(order.qty),
            "limit_price": str(limit_price),
            "position_intent": intent,
            "status": str(order.status),
            "submitted_at": str(order.submitted_at),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Options order failed: {e}", exc_info=True)
        raise HTTPException(500, "Internal server error")
