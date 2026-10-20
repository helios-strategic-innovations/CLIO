"""Quote routes — live pricing via Polygon and Alpaca."""

import copy
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query

from mission_control.backend.auth import require_auth
from mission_control.backend.deps import run_in_executor

router = APIRouter(prefix="/api", tags=["quotes"])
logger = logging.getLogger("mission_control.quotes")
_NY = ZoneInfo("America/New_York")
_QUOTE_HISTORY_PERIODS = {
    "1D": {"days": 7, "limit": 120, "timeframe": "5Min", "ttl": 30},
    "5D": {"days": 12, "limit": 5, "timeframe": "1D", "ttl": 300},
    "1M": {"days": 45, "limit": 30, "timeframe": "1D", "ttl": 900},
    "3M": {"days": 110, "limit": 66, "timeframe": "1D", "ttl": 900},
    "6M": {"days": 220, "limit": 132, "timeframe": "1D", "ttl": 1800},
    "1Y": {"days": 390, "limit": 260, "timeframe": "1D", "ttl": 1800},
}
_QUOTE_SNAPSHOT_TTL_SECONDS = 12
_QUOTE_DETAILS_TTL_SECONDS = 3600
_QUOTE_CACHE_MAX_ENTRIES = 512
_QUOTE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _cached_fetch(
    key: tuple[Any, ...],
    ttl_seconds: int,
    loader: Callable[[], Awaitable[Any]],
    *,
    usable: Callable[[Any], bool] = bool,
) -> tuple[Any, dict[str, Any]]:
    """Return fresh cached data, refresh it, or use an explicit stale fallback."""
    now = time.monotonic()
    entry = _QUOTE_CACHE.get(key)
    if entry is not None:
        age = max(0.0, now - float(entry["stored_monotonic"]))
        if age <= ttl_seconds:
            return copy.deepcopy(entry["value"]), {
                "hit": True,
                "age_seconds": round(age, 3),
                "ttl_seconds": ttl_seconds,
                "stored_at": entry["stored_at"],
                "stale": False,
                "reason_code": "OK",
            }

    try:
        loaded = await loader()
    except Exception as exc:
        logger.debug("Quote cache refresh failed for %s: %s", key, exc)
        loaded = None

    if usable(loaded):
        stored_at = _utc_now_iso()
        if key not in _QUOTE_CACHE and len(_QUOTE_CACHE) >= _QUOTE_CACHE_MAX_ENTRIES:
            oldest_key = min(
                _QUOTE_CACHE,
                key=lambda candidate: float(_QUOTE_CACHE[candidate]["stored_monotonic"]),
            )
            _QUOTE_CACHE.pop(oldest_key, None)
        _QUOTE_CACHE[key] = {
            "stored_monotonic": now,
            "stored_at": stored_at,
            "value": copy.deepcopy(loaded),
        }
        return copy.deepcopy(loaded), {
            "hit": False,
            "age_seconds": 0.0,
            "ttl_seconds": ttl_seconds,
            "stored_at": stored_at,
            "stale": False,
            "reason_code": "OK",
        }

    if entry is not None and usable(entry.get("value")):
        age = max(0.0, now - float(entry["stored_monotonic"]))
        return copy.deepcopy(entry["value"]), {
            "hit": True,
            "age_seconds": round(age, 3),
            "ttl_seconds": ttl_seconds,
            "stored_at": entry["stored_at"],
            "stale": True,
            "reason_code": "STALE_CACHE_FALLBACK",
        }

    return loaded, {
        "hit": False,
        "age_seconds": None,
        "ttl_seconds": ttl_seconds,
        "stored_at": _utc_now_iso(),
        "stale": True,
        "reason_code": "NO_DATA",
    }


def _positive_float(value) -> float:
    try:
        parsed = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0 else 0.0


def _quote_mark(snap: dict, alpaca_q: dict) -> tuple[float, str]:
    """Choose the safest display/chain anchor price without treating raw 0 as real."""
    last = _positive_float(snap.get("last_price"))
    day_close = _positive_float(snap.get("day_close"))
    bid = _positive_float(alpaca_q.get("bid_price"))
    ask = _positive_float(alpaca_q.get("ask_price"))
    prev_close = _positive_float(snap.get("prev_close"))

    if last:
        return last, "last_trade"
    if day_close:
        return day_close, "day_close"
    if bid and ask:
        return round((bid + ask) / 2, 4), "bid_ask_mid"
    if prev_close:
        return prev_close, "prev_close"
    return 0.0, "unavailable"


async def _polygon_snapshot(symbol: str) -> dict:
    """Get raw snapshot from Polygon."""
    try:
        from agent.integrations.polygon import get_snapshot_raw
        return await get_snapshot_raw(symbol)
    except Exception as e:
        logger.debug(f"Polygon snapshot failed for {symbol}: {e}")
        return {}


async def _alpaca_quote(symbol: str) -> dict:
    """Get latest quote from Alpaca."""
    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        from mission_control.backend.deps import get_data_client
        client = get_data_client()
        quotes = await run_in_executor(
            lambda: client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        )
        q = quotes.get(symbol)
        if q:
            return {
                "ask_price": float(q.ask_price or 0),
                "ask_size": float(q.ask_size or 0),
                "bid_price": float(q.bid_price or 0),
                "bid_size": float(q.bid_size or 0),
                "timestamp": str(q.timestamp) if q.timestamp else None,
            }
    except Exception as e:
        logger.debug(f"Alpaca quote failed for {symbol}: {e}")
    return {}


async def _fetch_equity_market_snapshot(symbol: str) -> dict:
    """Fetch a quote snapshot shared by quote routes and option-chain selection."""
    symbol = symbol.upper().strip()
    snap = await _polygon_snapshot(symbol)
    alpaca_q = await _alpaca_quote(symbol)

    # get_snapshot_raw returns a FLAT dict. A zero last/day price means the snapshot
    # has no usable trade price, not that the equity traded at zero.
    price = _positive_float(snap.get("last_price")) or _positive_float(snap.get("day_close"))
    prev_close = _positive_float(snap.get("prev_close"))
    mark_price, mark_source = _quote_mark(snap, alpaca_q)
    change = snap.get("change") if price else (mark_price - prev_close if mark_price and prev_close else None)
    change_pct = snap.get("change_pct") if price else (
        (change / prev_close) * 100 if change is not None and prev_close else None
    )

    bid = _positive_float(alpaca_q.get("bid_price"))
    ask = _positive_float(alpaca_q.get("ask_price"))
    if price:
        source_status = "ok"
    elif mark_source == "bid_ask_mid":
        source_status = "quote_only"
    elif mark_source == "prev_close":
        source_status = "prev_close"
    else:
        source_status = "unavailable"

    available_sources = []
    if snap:
        available_sources.append("polygon_snapshot")
    if alpaca_q:
        available_sources.append("alpaca_quote")
    if source_status == "prev_close":
        reason_code = "PREVIOUS_CLOSE_ONLY"
    elif len(available_sources) == 2:
        reason_code = "OK"
    elif available_sources:
        reason_code = "PARTIAL_QUOTE_SOURCES"
    else:
        reason_code = "QUOTE_DATA_UNAVAILABLE"

    return {
        "symbol": symbol,
        "price": price or None,
        "mark_price": mark_price or None,
        "display_price": mark_price or None,
        "price_source": mark_source,
        "change": round(float(change), 2) if change is not None else None,
        "change_pct": round(float(change_pct), 2) if change_pct is not None else None,
        "open": None,
        "high": _positive_float(snap.get("day_high")) or None,
        "low": _positive_float(snap.get("day_low")) or None,
        "close": _positive_float(snap.get("day_close")) or None,
        "volume": int(_positive_float(snap.get("volume"))),
        "prev_close": prev_close or None,
        "bid": bid or None,
        "ask": ask or None,
        "spread": round(ask - bid, 4) if ask and bid else None,
        "quote_timestamp": alpaca_q.get("timestamp"),
        "source_status": source_status,
        "source": "+".join(available_sources) if available_sources else "unavailable",
        "sources": {
            "polygon_snapshot": {"available": bool(snap)},
            "alpaca_quote": {"available": bool(alpaca_q)},
        },
        "as_of": alpaca_q.get("timestamp") or _utc_now_iso(),
        "stale": source_status in {"prev_close", "unavailable"},
        "reason_code": reason_code,
    }


async def get_equity_market_snapshot(symbol: str) -> dict:
    """Cached quote snapshot with additive provenance and freshness metadata."""
    normalized_symbol = symbol.upper().strip()
    snapshot, cache = await _cached_fetch(
        ("snapshot", normalized_symbol),
        _QUOTE_SNAPSHOT_TTL_SECONDS,
        lambda: _fetch_equity_market_snapshot(normalized_symbol),
        usable=lambda value: bool(value) and value.get("source_status") != "unavailable",
    )
    if not isinstance(snapshot, dict):
        snapshot = await _fetch_equity_market_snapshot(normalized_symbol)
    snapshot["cache"] = cache
    if cache["reason_code"] == "STALE_CACHE_FALLBACK":
        snapshot["stale"] = True
        snapshot["reason_code"] = "STALE_CACHE_FALLBACK"
    return snapshot


async def _polygon_details(symbol: str) -> dict:
    try:
        from agent.integrations.polygon import _get
        data = await _get(f"/v3/reference/tickers/{symbol}")
        result = data.get("results", {}) if isinstance(data, dict) else {}
        if not isinstance(result, dict):
            return {}
        return {
            "name": result.get("name"),
            "primary_exchange": result.get("primary_exchange"),
            "market_cap": result.get("market_cap"),
            "shares_outstanding": result.get("share_class_shares_outstanding"),
            "sic_description": result.get("sic_description"),
        }
    except Exception as e:
        logger.debug(f"Polygon details failed for {symbol}: {e}")
        return {}


async def _polygon_history(symbol: str, days: int, limit: int) -> list[dict]:
    try:
        from agent.integrations.polygon import _get
        end = datetime.now(_NY)
        start = end - timedelta(days=days)
        data = await _get(
            f"/v2/aggs/ticker/{symbol}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
            {"adjusted": "true", "sort": "desc", "limit": limit},
        )
        rows = data.get("results", []) if isinstance(data, dict) else []
        # Polygon applies limit before returning rows; request newest first, then
        # restore chronological order for charts and beta calculations.
        rows = list(reversed(rows[:limit])) if isinstance(rows, list) else []
        bars = []
        for row in rows:
            close = _positive_float(row.get("c"))
            if not close:
                continue
            bars.append({
                "timestamp": datetime.fromtimestamp(row.get("t", 0) / 1000, tz=_NY).isoformat(),
                "open": _positive_float(row.get("o")),
                "high": _positive_float(row.get("h")),
                "low": _positive_float(row.get("l")),
                "close": close,
                "volume": int(_positive_float(row.get("v"))),
            })
        return bars
    except Exception as e:
        logger.debug(f"Polygon history failed for {symbol}: {e}")
        return []


async def _polygon_intraday_history(symbol: str, days: int, limit: int) -> list[dict]:
    try:
        from agent.integrations.polygon import _get
        end = datetime.now(_NY)
        start = end - timedelta(days=days)
        data = await _get(
            f"/v2/aggs/ticker/{symbol}/range/5/minute/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
            {"adjusted": "true", "sort": "desc", "limit": limit},
        )
        rows = data.get("results", []) if isinstance(data, dict) else []
        rows = list(reversed(rows[:limit])) if isinstance(rows, list) else []
        bars = []
        for row in rows:
            close = _positive_float(row.get("c"))
            if not close:
                continue
            bars.append({
                "timestamp": datetime.fromtimestamp(row.get("t", 0) / 1000, tz=_NY).isoformat(),
                "open": _positive_float(row.get("o")),
                "high": _positive_float(row.get("h")),
                "low": _positive_float(row.get("l")),
                "close": close,
                "volume": int(_positive_float(row.get("v"))),
            })
        return bars
    except Exception as e:
        logger.debug(f"Polygon intraday history failed for {symbol}: {e}")
        return []


async def _cached_polygon_details(symbol: str) -> tuple[dict, dict[str, Any]]:
    details, cache = await _cached_fetch(
        ("reference_details", symbol),
        _QUOTE_DETAILS_TTL_SECONDS,
        lambda: _polygon_details(symbol),
    )
    return details if isinstance(details, dict) else {}, cache


async def _cached_polygon_bars(
    symbol: str,
    *,
    timeframe: str,
    days: int,
    limit: int,
    ttl_seconds: int,
) -> tuple[list[dict], dict[str, Any]]:
    loader = (
        (lambda: _polygon_intraday_history(symbol, days=days, limit=limit))
        if timeframe == "5Min"
        else (lambda: _polygon_history(symbol, days=days, limit=limit))
    )
    bars, cache = await _cached_fetch(
        ("aggregate_bars", symbol, timeframe, days, limit),
        ttl_seconds,
        loader,
    )
    return bars if isinstance(bars, list) else [], cache


def _structured_bars_response(
    *,
    symbol: str,
    period: str,
    timeframe: str,
    bars: list[dict],
    cache: dict[str, Any],
) -> dict[str, Any]:
    points = [
        {
            "t": bar.get("timestamp"),
            "o": bar.get("open"),
            "h": bar.get("high"),
            "l": bar.get("low"),
            "c": bar.get("close"),
            "v": bar.get("volume"),
        }
        for bar in bars
    ]
    as_of = next(
        (bar.get("timestamp") for bar in reversed(bars) if bar.get("timestamp")),
        cache.get("stored_at") or _utc_now_iso(),
    )
    stale = bool(cache.get("stale")) or not bars
    if cache.get("reason_code") == "STALE_CACHE_FALLBACK":
        reason_code = "STALE_CACHE_FALLBACK"
    elif not bars:
        reason_code = "POLYGON_CANDLES_UNAVAILABLE"
    else:
        reason_code = "OK"
    return {
        "symbol": symbol,
        "period": period,
        "timeframe": timeframe,
        "points": points,
        "source": "polygon_aggregates",
        "as_of": as_of,
        "stale": stale,
        "reason_code": reason_code,
        "count": len(points),
        "freshness": {
            "as_of": as_of,
            "stale": stale,
            "cache_hit": bool(cache.get("hit")),
            "cache_age_seconds": cache.get("age_seconds"),
            "ttl_seconds": cache.get("ttl_seconds"),
            "reason_code": reason_code,
        },
        # Existing history consumers use `bars`; candles clients use `points`.
        "bars": bars,
    }


def _daily_returns(bars: list[dict]) -> dict[str, float]:
    returns = {}
    for prev, cur in zip(bars, bars[1:]):
        prev_close = _positive_float(prev.get("close"))
        cur_close = _positive_float(cur.get("close"))
        if prev_close and cur_close:
            returns[str(cur.get("timestamp", ""))[:10]] = (cur_close / prev_close) - 1
    return returns


async def _beta_1y(symbol: str, bars: list[dict]) -> float | None:
    if symbol == "SPY":
        return 1.0
    spy_bars, _ = await _cached_polygon_bars(
        "SPY",
        timeframe="1D",
        days=390,
        limit=260,
        ttl_seconds=1800,
    )
    symbol_returns = _daily_returns(bars)
    spy_returns = _daily_returns(spy_bars)
    common = sorted(set(symbol_returns) & set(spy_returns))
    if len(common) < 30:
        return None
    xs = [spy_returns[d] for d in common]
    ys = [symbol_returns[d] for d in common]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    variance = sum((x - x_mean) ** 2 for x in xs)
    if variance <= 0:
        return None
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return round(covariance / variance, 2)


@router.get("/quote/{symbol}")
async def get_quote(symbol: str, _=Depends(require_auth)):
    """Live quote for a symbol — Polygon snapshot + Alpaca bid/ask."""
    return await get_equity_market_snapshot(symbol)


@router.get("/quote/{symbol}/details")
async def get_quote_details(symbol: str, _=Depends(require_auth)):
    """Order-page market card: quote, 52-week range, beta, and basic identity fields."""
    symbol = symbol.upper().strip()
    snapshot = await get_equity_market_snapshot(symbol)
    details, details_cache = await _cached_polygon_details(symbol)
    bars, bars_cache = await _cached_polygon_bars(
        symbol,
        timeframe="1D",
        days=390,
        limit=260,
        ttl_seconds=1800,
    )

    highs = [_positive_float(b.get("high")) for b in bars if _positive_float(b.get("high"))]
    lows = [_positive_float(b.get("low")) for b in bars if _positive_float(b.get("low"))]
    week52_high = round(max(highs), 2) if highs else None
    week52_low = round(min(lows), 2) if lows else None
    beta_1y = await _beta_1y(symbol, bars) if bars else None
    history_as_of = next(
        (bar.get("timestamp") for bar in reversed(bars) if bar.get("timestamp")),
        bars_cache.get("stored_at"),
    )
    week52_reason = (
        "STALE_CACHE_FALLBACK"
        if bars_cache.get("stale") and bars
        else "OK" if week52_high is not None and week52_low is not None
        else "INSUFFICIENT_52W_HISTORY"
    )
    beta_reason = (
        "STALE_CACHE_FALLBACK"
        if bars_cache.get("stale") and beta_1y is not None
        else "OK" if beta_1y is not None
        else "INSUFFICIENT_BETA_HISTORY"
    )
    reference_reason = (
        "STALE_CACHE_FALLBACK"
        if details_cache.get("reason_code") == "STALE_CACHE_FALLBACK"
        else "OK" if details
        else "POLYGON_REFERENCE_UNAVAILABLE"
    )

    snapshot.update(details)
    snapshot.update({
        "week52_high": week52_high,
        "week52_low": week52_low,
        "beta_1y": beta_1y,
        "beta": beta_1y,
        "history_points": len(bars),
        "week52": {
            "high": week52_high,
            "low": week52_low,
            "source": "polygon_aggregates",
            "as_of": history_as_of,
            "stale": bool(bars_cache.get("stale")) or not bars,
            "reason_code": week52_reason,
        },
        "beta_details": {
            "value": beta_1y,
            "benchmark": "SPY",
            "lookback": "1Y",
            "source": "polygon_aggregates",
            "as_of": history_as_of,
            "stale": bool(bars_cache.get("stale")) or beta_1y is None,
            "reason_code": beta_reason,
        },
        "details_provenance": {
            "reference": {
                "source": "polygon_reference",
                "stale": bool(details_cache.get("stale")),
                "as_of": details_cache.get("stored_at"),
                "reason_code": reference_reason,
            },
            "history": {
                "source": "polygon_aggregates",
                "stale": bool(bars_cache.get("stale")) or not bars,
                "as_of": history_as_of,
                "reason_code": week52_reason,
            },
        },
    })
    if not snapshot.get("stale"):
        if not bars:
            snapshot["stale"] = True
            snapshot["reason_code"] = "QUOTE_DETAILS_HISTORY_UNAVAILABLE"
        elif (
            bars_cache.get("reason_code") == "STALE_CACHE_FALLBACK"
            or details_cache.get("reason_code") == "STALE_CACHE_FALLBACK"
        ):
            snapshot["stale"] = True
            snapshot["reason_code"] = "STALE_CACHE_FALLBACK"
        elif beta_1y is None:
            snapshot["reason_code"] = "QUOTE_DETAILS_BETA_UNAVAILABLE"
        elif not details:
            snapshot["reason_code"] = "QUOTE_REFERENCE_DETAILS_UNAVAILABLE"
        else:
            snapshot["reason_code"] = "OK"
    return snapshot


@router.get("/quote/{symbol}/history")
async def get_quote_history(
    symbol: str,
    period: str = Query("3M", pattern="^(1D|5D|1M|3M|6M|1Y)$"),
    range_: Optional[str] = Query(None, alias="range", pattern="^(1D|5D|1M|3M|6M|1Y)$"),
    limit: Optional[int] = Query(None, ge=5, le=5000),
    _=Depends(require_auth),
):
    """Structured bars for order-page charts."""
    symbol = symbol.upper().strip()
    selected_period = range_ or period
    cfg = _QUOTE_HISTORY_PERIODS[selected_period]
    resolved_limit = limit or int(cfg["limit"])
    bars, cache = await _cached_polygon_bars(
        symbol,
        timeframe=str(cfg["timeframe"]),
        days=int(cfg["days"]),
        limit=resolved_limit,
        ttl_seconds=int(cfg["ttl"]),
    )
    return _structured_bars_response(
        symbol=symbol,
        period=selected_period,
        timeframe=str(cfg["timeframe"]),
        bars=bars,
        cache=cache,
    )


@router.get("/quote/{symbol}/candles")
async def get_quote_candles(
    symbol: str,
    period: str = Query("3M", pattern="^(1D|5D|1M|3M|6M|1Y)$"),
    range_: Optional[str] = Query(None, alias="range", pattern="^(1D|5D|1M|3M|6M|1Y)$"),
    limit: Optional[int] = Query(None, ge=5, le=5000),
    _=Depends(require_auth),
):
    """Structured OHLCV candles with bounded caching and explicit provenance."""
    symbol = symbol.upper().strip()
    selected_period = range_ or period
    cfg = _QUOTE_HISTORY_PERIODS[selected_period]
    resolved_limit = limit or int(cfg["limit"])
    bars, cache = await _cached_polygon_bars(
        symbol,
        timeframe=str(cfg["timeframe"]),
        days=int(cfg["days"]),
        limit=resolved_limit,
        ttl_seconds=int(cfg["ttl"]),
    )
    return _structured_bars_response(
        symbol=symbol,
        period=selected_period,
        timeframe=str(cfg["timeframe"]),
        bars=bars,
        cache=cache,
    )


@router.get("/quote/{symbol}/bars")
async def get_bars(
    symbol: str,
    timeframe: str = Query("1D", pattern="^(1Min|5Min|15Min|1H|1D|1W)$"),
    limit: int = Query(30, ge=1, le=200),
    _=Depends(require_auth),
):
    """Historical bars for mini-chart."""
    try:
        from agent.integrations.polygon import _get_bars
        result = await _get_bars(symbol.upper(), timeframe, limit)
        return {"symbol": symbol.upper(), "bars_text": result}
    except Exception as e:
        logger.error(f"Bars fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
