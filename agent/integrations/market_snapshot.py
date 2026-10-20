"""Market price snapshots sourced from Alpaca Elite (POLY-1 / SUB-1, S211).

Polygon is no longer a paid subscription (operator, 2026-09-29): its real-time
``/v2/snapshot/...`` endpoint answers 403 NOT_AUTHORIZED, so every ``market_events`` tick and
learning-cycle "market snapshot" failed. Alpaca (Elite) is the market-data source of record.

``get_snapshot_raw`` returns the SAME dict shape as ``polygon.get_snapshot_raw`` so callers only
change an import::

    {"last_price", "day_close", "prev_close", "change", "change_pct", "day_high", "day_low", "volume"}

Index symbols (VIX, DXY, TNX, ...) are not equities on Alpaca -- they keep polygon's free Yahoo
fallback. ``CLIO_MARKET_SNAPSHOT_SOURCE=polygon`` restores the legacy passthrough (rollback flag).

Alpaca SDK calls are blocking: they go through ``alpaca._retry_alpaca_call`` (executor + retry),
never directly on the event loop.
"""

from __future__ import annotations

import logging
import math
import os

logger = logging.getLogger("openclaw.integrations.market_snapshot")


def _source() -> str:
    value = os.getenv("CLIO_MARKET_SNAPSHOT_SOURCE", "alpaca").strip().lower()
    return value if value in ("alpaca", "polygon") else "alpaca"


def _num(value) -> float | None:
    """float(value) or None; NaN/inf and non-numerics are 'unknown', 0.0 stays a real value."""
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


async def _alpaca_snapshot(symbol: str) -> dict:
    from alpaca.data.requests import StockSnapshotRequest

    from agent.integrations import alpaca as _alp

    client = _alp._get_data_client()
    request = StockSnapshotRequest(symbol_or_symbols=symbol)
    snapshots = await _alp._retry_alpaca_call(client.get_stock_snapshot, request)
    snap = snapshots.get(symbol) if isinstance(snapshots, dict) else None
    if snap is None:
        raise LookupError(f"no Alpaca snapshot returned for {symbol}")

    trade = getattr(snap, "latest_trade", None)
    daily = getattr(snap, "daily_bar", None)
    prev = getattr(snap, "previous_daily_bar", None)

    last = _num(getattr(trade, "price", None))
    day_close = _num(getattr(daily, "close", None))
    if day_close is None:
        day_close = last
    prev_close = _num(getattr(prev, "close", None))
    if prev_close is None:
        prev_close = 0.0

    change = day_close - prev_close if (day_close is not None and prev_close) else 0
    pct = (change / prev_close) * 100 if prev_close else 0
    return {
        "last_price": last if last is not None else (day_close if day_close is not None else 0.0),
        "day_close": day_close,
        "prev_close": prev_close,
        "change": change,
        "change_pct": pct,
        "day_high": _num(getattr(daily, "high", None)) or 0.0,
        "day_low": _num(getattr(daily, "low", None)) or 0.0,
        "volume": _num(getattr(daily, "volume", None)) or 0.0,
    }


async def get_snapshot_raw(symbol: str) -> dict:
    """Snapshot dict for one symbol (see module docstring). Raises on provider errors,
    exactly like ``polygon.get_snapshot_raw`` -- callers already handle exceptions."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return {}
    from agent.integrations import polygon as _pg

    if _source() == "polygon":
        return await _pg.get_snapshot_raw(sym)
    if sym in _pg._INDEX_SYMBOLS:
        return await _pg._yahoo_index_snapshot(sym)
    return await _alpaca_snapshot(sym)


async def get_price_change(symbol: str) -> dict | None:
    """``{"price", "change_pct"}`` for one symbol, or None when no data (learning-cycle shape)."""
    snap = await get_snapshot_raw(symbol)
    if not snap:
        return None
    price = snap.get("last_price")
    if price is None:
        price = snap.get("day_close", 0)
    return {"price": price, "change_pct": snap.get("change_pct", 0)}
