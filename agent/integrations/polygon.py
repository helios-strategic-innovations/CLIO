"""
OpenClaw/Moltbot — Polygon.io Market Data Integration
Primary source for real-time quotes, historical bars, snapshots, and news.
Polygon Massive tier provides: quotes, bars, snapshots, Benzinga news,
ticker details, earnings, and analyst ratings.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import asyncio

import httpx
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from agent.config import config

# Singleton httpx client for connection reuse
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client(timeout: float = 15.0) -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        async with _client_lock:
            if _client is None or _client.is_closed:
                _client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    return _client


async def close_client():
    global _client
    async with _client_lock:
        if _client is not None:
            await _client.aclose()
            _client = None

_NY = ZoneInfo("America/New_York")
from agent.rate_limiter import retry_on_transient
from agent.security import sanitize_ingested_content

# Index symbols not available on Polygon stocks endpoints — use Yahoo fallback
_INDEX_SYMBOLS = {"VIX", "DXY", "TNX", "TYX", "IRX"}

logger = logging.getLogger("openclaw.integrations.polygon")


def _safe_float(val, default=None):
    """Coerce API value to float — Polygon can return numeric fields as strings."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

BASE_URL = None
API_KEY = None


def _strip_api_key(url: str) -> str:
    """Remove apiKey query parameter from a URL before logging."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params.pop("apiKey", None)
    cleaned = parsed._replace(query=urlencode(params, doseq=True))
    return urlunparse(cleaned)


def _init():
    global BASE_URL, API_KEY
    if BASE_URL is None:
        BASE_URL = config.polygon_base_url.rstrip("/")
        API_KEY = config.polygon_api_key


# POLY-1 (S211): the plan is not entitled to real-time stock snapshots -- the endpoint answers
# 403 NOT_AUTHORIZED on every call (aggregates/news are fine). Stop asking for a window instead
# of sending ~100 doomed requests a day. Only the snapshot path is affected.
_SNAPSHOT_PATH_PREFIX = "/v2/snapshot/locale/us/markets/stocks/"
_snapshot_denied_until = 0.0  # time.monotonic() deadline; 0.0 = breaker closed


class PolygonNotEntitled(Exception):
    """Raised without any HTTP call while the snapshot 403 breaker is open."""


def _snapshot_breaker_seconds() -> float:
    try:
        return max(60.0, float(os.getenv("CLIO_POLYGON_SNAPSHOT_BREAKER_S", "21600")))
    except ValueError:
        return 21600.0


@retry_on_transient(max_attempts=3, base_delay=1.0)
async def _get(endpoint: str, params: dict | None = None) -> dict:
    """Make authenticated GET request to Polygon API with retry on transient errors."""
    global _snapshot_denied_until
    _init()
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY not configured")

    is_snapshot = endpoint.startswith(_SNAPSHOT_PATH_PREFIX)
    if is_snapshot and time.monotonic() < _snapshot_denied_until:
        raise PolygonNotEntitled("Polygon snapshot endpoint not entitled (breaker open)")

    url = f"{BASE_URL}{endpoint}"
    # S118 W-025: Copy params dict to avoid mutating caller's dict
    request_params = dict(params) if params else {}
    request_params["apiKey"] = API_KEY

    client = await _get_client()
    resp = await client.get(url, params=request_params)
    if is_snapshot and resp.status_code == 403:
        window = _snapshot_breaker_seconds()
        _snapshot_denied_until = time.monotonic() + window
        logger.warning(
            "Polygon snapshot endpoint returned 403 (plan not entitled); "
            "suppressing /v2/snapshot calls for %.0fs", window,
        )
    resp.raise_for_status()
    return resp.json()


async def get_market_data(params: dict[str, Any]) -> str:
    """Fetch market data from Polygon — quotes, bars, snapshots, news, earnings, ratings, guidance, or details."""
    symbol = params["symbol"].upper()
    data_type = params.get("data_type", "quote")

    try:
        if data_type == "quote":
            return await _get_quote(symbol)
        elif data_type == "bars":
            tf = params.get("timeframe", "1d")
            limit = params.get("limit", 10)
            return await _get_bars(symbol, tf, limit)
        elif data_type == "snapshot":
            return await _get_snapshot(symbol)
        elif data_type == "news":
            limit = params.get("limit", 10)
            return await _get_news(symbol, limit)
        elif data_type == "details":
            return await _get_ticker_details(symbol)
        elif data_type == "earnings":
            return await _get_earnings(symbol)
        elif data_type == "ratings":
            return await _get_analyst_ratings(symbol)
        elif data_type == "guidance":
            return await _get_guidance(symbol)
        elif data_type == "calendar":
            return await _get_earnings_calendar(symbol)
        else:
            return (
                f"Unknown data_type: {data_type}. Valid: quote, bars, snapshot, news, "
                f"details, earnings, ratings, guidance, calendar"
            )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            if data_type == "quote":
                try:
                    return await _get_snapshot(symbol)
                except Exception as e:
                    logger.debug(f"Fire-and-forget snapshot fallback failed: {e}")
            return f"ERROR: Polygon API access denied for {symbol} ({data_type}). Check plan tier."
        safe_url = _strip_api_key(str(e.request.url)) if e.request else "unknown"
        logger.error(f"Polygon HTTP {e.response.status_code} for {safe_url}")
        return f"ERROR: Polygon API error — {e.response.status_code}"
    except Exception as e:
        safe_msg = _strip_api_key(str(e)) if "apiKey" in str(e) else str(e)
        logger.error(f"Polygon error for {symbol}: {safe_msg}")
        return f"ERROR: Polygon failed for {symbol} — {safe_msg}"


async def _get_quote(symbol: str) -> str:
    """Get latest quote using snapshot endpoint (reliable across all tiers)."""
    if symbol in _INDEX_SYMBOLS:
        return await _format_yahoo_index(symbol)
    # Use snapshot endpoint — includes lastQuote, lastTrade, day, prevDay
    # This is more reliable than /v3/quotes which can return 403
    data = await _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")

    ticker = data.get("ticker", {})
    if not ticker:
        return f"No quote data for {symbol}"

    last_quote = ticker.get("lastQuote", {})
    last_trade = ticker.get("lastTrade", {})
    day = ticker.get("day", {})
    prev = ticker.get("prevDay", {})

    ask = _safe_float(last_quote.get("P"), 0)
    bid = _safe_float(last_quote.get("p"), 0)
    spread = ask - bid if ask and bid else 0
    last_price = _safe_float(last_trade.get("p"), 0)
    last_size = int(_safe_float(last_trade.get("s"), 0))

    prev_close = _safe_float(prev.get("c"), 0)
    change = last_price - prev_close if last_price and prev_close else 0
    pct = (change / prev_close) * 100 if prev_close else 0

    day_o = _safe_float(day.get("o"), 0)
    day_h = _safe_float(day.get("h"), 0)
    day_l = _safe_float(day.get("l"), 0)
    day_c = _safe_float(day.get("c"), 0)
    day_v = int(_safe_float(day.get("v"), 0))

    result = (
        f"Quote for {symbol} (Polygon):\n"
        f"  Last: ${last_price:.2f} ({last_size:,} shares)\n"
        f"  Ask: ${ask:.2f} x {int(_safe_float(last_quote.get('S'), 0))}  |  Bid: ${bid:.2f} x {int(_safe_float(last_quote.get('s'), 0))}\n"
        f"  Spread: ${spread:.4f}\n"
        f"  Day: O ${day_o:.2f} H ${day_h:.2f} "
        f"L ${day_l:.2f} C ${day_c:.2f} V {day_v:,}\n"
        f"  Change: ${change:+,.2f} ({pct:+.2f}%) from prev close ${prev_close:.2f}"
    )
    return sanitize_ingested_content(result, "polygon_api")


async def _get_bars(symbol: str, timeframe: str, limit: int) -> str:
    """Get historical bars (OHLCV)."""
    tf_map = {
        "1m": ("1", "minute"),
        "5m": ("5", "minute"),
        "15m": ("15", "minute"),
        "1h": ("1", "hour"),
        "1d": ("1", "day"),
        "1w": ("1", "week"),
    }
    multiplier, timespan = tf_map.get(timeframe, ("1", "day"))

    lookback = {"1m": 1, "5m": 3, "15m": 7, "1h": 14, "1d": 60, "1w": 365}
    days = lookback.get(timeframe, 60)

    end = datetime.now(_NY)
    start = end - timedelta(days=days)

    data = await _get(
        f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
        {"adjusted": "true", "sort": "asc", "limit": limit},
    )

    results = data.get("results", [])
    if not results:
        return f"No bar data for {symbol} ({timeframe})"

    lines = [f"Bars for {symbol} ({timeframe}, {len(results)} bars) via Polygon:"]
    for bar in results:
        ts = datetime.fromtimestamp(bar["t"] / 1000, tz=_NY).strftime("%Y-%m-%d %H:%M")
        o, h, l, c, v = (_safe_float(bar.get("o"), 0), _safe_float(bar.get("h"), 0),
                         _safe_float(bar.get("l"), 0), _safe_float(bar.get("c"), 0),
                         int(_safe_float(bar.get("v"), 0)))
        lines.append(
            f"  {ts} | O: ${o:.2f} H: ${h:.2f} "
            f"L: ${l:.2f} C: ${c:.2f} V: {v:,}"
        )

    if len(results) >= 2:
        first_close = _safe_float(results[0].get("c"), 0)
        last_close = _safe_float(results[-1].get("c"), 0)
        change = last_close - first_close
        pct = (change / first_close) * 100 if first_close else 0
        total_vol = sum(int(_safe_float(b.get("v"), 0)) for b in results)
        lines.append(f"\n  Period Change: ${change:+,.2f} ({pct:+.2f}%) | Total Volume: {total_vol:,}")

    return sanitize_ingested_content("\n".join(lines), "polygon_api")


async def _get_snapshot(symbol: str) -> str:
    """Get full snapshot — price, quote, prev day, minute bar."""
    if symbol in _INDEX_SYMBOLS:
        return await _format_yahoo_index(symbol)
    data = await _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")

    ticker = data.get("ticker", {})
    if not ticker:
        return f"No snapshot for {symbol}"

    day = ticker.get("day", {})
    prev = ticker.get("prevDay", {})
    last_quote = ticker.get("lastQuote", {})
    last_trade = ticker.get("lastTrade", {})
    minute = ticker.get("min", {})

    _dc = _safe_float(day.get("c"))
    day_close = _dc if _dc is not None else _safe_float(last_trade.get("p"), 0)  # S153: avoid falsy on 0.0
    prev_close = _safe_float(prev.get("c"), 0)
    change = day_close - prev_close if day_close is not None and prev_close else 0
    pct = (change / prev_close) * 100 if prev_close else 0

    lines = [
        f"Snapshot for {symbol} (Polygon):",
        f"  Last Trade: ${_safe_float(last_trade.get('p', 0), 0):.2f} ({int(_safe_float(last_trade.get('s', 0), 0))} shares)",
        f"  Quote: Ask ${_safe_float(last_quote.get('P', 0), 0):.2f} / Bid ${_safe_float(last_quote.get('p', 0), 0):.2f}",
        f"  Today: O ${_safe_float(day.get('o', 0), 0):.2f} H ${_safe_float(day.get('h', 0), 0):.2f} "
        f"L ${_safe_float(day.get('l', 0), 0):.2f} C ${_safe_float(day.get('c', 0), 0):.2f} V {int(_safe_float(day.get('v', 0), 0)):,}",
        f"  Change: ${change:+,.2f} ({pct:+.2f}%)",
        f"  Prev Close: ${prev_close:.2f} | Prev Volume: {int(_safe_float(prev.get('v', 0), 0)):,}",
    ]
    if minute:
        lines.append(
            f"  Last Minute: O ${_safe_float(minute.get('o', 0), 0):.2f} H ${_safe_float(minute.get('h', 0), 0):.2f} "
            f"L ${_safe_float(minute.get('l', 0), 0):.2f} C ${_safe_float(minute.get('c', 0), 0):.2f} V {int(_safe_float(minute.get('v', 0), 0)):,}"
        )
    return sanitize_ingested_content("\n".join(lines), "polygon_api")


async def get_snapshot_raw(symbol: str) -> dict:
    """Get raw snapshot data as dict (for programmatic use, not formatted text)."""
    # Session 68: Index symbols not available on Polygon stocks endpoint — Yahoo fallback
    if symbol.upper() in _INDEX_SYMBOLS:
        return await _yahoo_index_snapshot(symbol)

    data = await _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
    ticker = data.get("ticker", {})
    if not ticker:
        return {}
    day = ticker.get("day", {})
    prev = ticker.get("prevDay", {})
    last_trade = ticker.get("lastTrade", {})
    _dc2 = _safe_float(day.get("c"))
    day_close = _dc2 if _dc2 is not None else _safe_float(last_trade.get("p"), 0)  # S153: avoid falsy on 0.0
    prev_close = _safe_float(prev.get("c"), 0)
    change = day_close - prev_close if day_close is not None and prev_close else 0
    pct = (change / prev_close) * 100 if prev_close else 0
    return {
        "last_price": _safe_float(last_trade.get("p"), 0),
        "day_close": day_close,
        "prev_close": prev_close,
        "change": change,
        "change_pct": pct,
        "day_high": _safe_float(day.get("h"), 0),
        "day_low": _safe_float(day.get("l"), 0),
        "volume": _safe_float(day.get("v"), 0),
    }


async def _yahoo_index_snapshot(symbol: str) -> dict:
    """Session 68: Fetch index data from Yahoo Finance (free, no API key required)."""
    yahoo_symbol = f"^{symbol.upper()}"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    try:
        client = await _get_client()
        resp = await client.get(url, params={"range": "1d", "interval": "1d"},
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=8.0)
        if resp.status_code != 200:
            logger.debug(f"Yahoo index snapshot {resp.status_code} for {symbol}")
            return {}
        data = resp.json()
        chart_result = data.get("chart", {}).get("result", [])
        if not chart_result:
            return {}
        result = chart_result[0]
        meta = result.get("meta", {})
        # S162 M-11: Validate Yahoo response fields instead of silently defaulting to 0
        price = meta.get("regularMarketPrice") or meta.get("regularMarketPreviousClose")
        if price is None:
            logger.warning(f"Yahoo index snapshot missing price for {symbol}: {list(meta.keys())[:10]}")
            return {}
        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
        if prev_close is None:
            prev_close = price  # 0% change fallback
        change = price - prev_close if price and prev_close else 0
        pct = (change / prev_close) * 100 if prev_close else 0
        return {
            "last_price": price,
            "day_close": price,
            "prev_close": prev_close,
            "change": change,
            "change_pct": pct,
            "day_high": meta.get("regularMarketDayHigh", 0),
            "day_low": meta.get("regularMarketDayLow", 0),
            "volume": meta.get("regularMarketVolume", 0),
        }
    except Exception as e:
        logger.debug(f"Yahoo index snapshot error for {symbol}: {e}")
        return {}


async def _format_yahoo_index(symbol: str) -> str:
    """Format Yahoo index data as user-facing text for quote/snapshot handlers."""
    data = await _yahoo_index_snapshot(symbol)
    if not data or not data.get("last_price"):
        return f"No data available for index {symbol}."
    formatted = (
        f"Index Snapshot for {symbol} (Yahoo Finance):\n"
        f"  Last: ${data['last_price']:.2f}\n"
        f"  Day High: ${data['day_high']:.2f} | Day Low: ${data['day_low']:.2f}\n"
        f"  Prev Close: ${data['prev_close']:.2f}\n"
        f"  Change: ${data['change']:+,.2f} ({data['change_pct']:+.2f}%)\n"
        f"  Volume: {data['volume']:,}"
    )
    return sanitize_ingested_content(formatted, source="yahoo_finance")


async def _get_news(symbol: str, limit: int = 10) -> str:
    """Get Benzinga news for a symbol via Polygon's reference/news endpoint."""
    data = await _get("/v2/reference/news", {
        "ticker": symbol,
        "limit": min(limit, 20),
        "order": "desc",
        "sort": "published_utc",
    })

    results = data.get("results", [])
    if not results:
        return f"No news found for {symbol}."

    lines = [f"News for {symbol} ({len(results)} articles via Polygon/Benzinga):"]
    for i, article in enumerate(results[:limit], 1):
        title = article.get("title", "No title")
        publisher = article.get("publisher", {}).get("name", "Unknown")
        published = article.get("published_utc", "")[:16].replace("T", " ")
        description = article.get("description", "")

        safe_title = sanitize_ingested_content(title, f"polygon_news:{publisher}")
        safe_desc = sanitize_ingested_content(description, f"polygon_news:{publisher}") if description else ""

        lines.append(f"\n{i}. [{publisher}] {published}")
        lines.append(f"   {safe_title}")
        if safe_desc:
            lines.append(f"   {safe_desc[:200]}")

    return "\n".join(lines)


async def _get_ticker_details(symbol: str) -> str:
    """Get ticker details — company info, market cap, shares outstanding."""
    data = await _get(f"/v3/reference/tickers/{symbol}")

    result = data.get("results", {})
    if not result:
        return f"No details found for {symbol}."

    name = result.get("name", "Unknown")
    market_cap = result.get("market_cap", 0)
    shares = result.get("share_class_shares_outstanding", 0)
    sic_desc = result.get("sic_description", "N/A")
    exchange = result.get("primary_exchange", "N/A")
    locale = result.get("locale", "N/A")
    description = result.get("description", "")

    safe_desc = sanitize_ingested_content(description, f"polygon_details:{symbol}") if description else ""

    lines = [
        f"Ticker Details — {symbol} ({name}):",
        f"  Exchange: {exchange} | Locale: {locale}",
        f"  Sector: {sic_desc}",
        f"  Market Cap: ${market_cap:,.0f}" if market_cap else "  Market Cap: N/A",
        f"  Shares Outstanding: {shares:,}" if shares else "  Shares Outstanding: N/A",
    ]
    if safe_desc:
        lines.append(f"  Description: {safe_desc[:300]}")

    return sanitize_ingested_content("\n".join(lines), "polygon_api")


async def _get_earnings(symbol: str) -> str:
    """Get Benzinga earnings data — EPS/revenue estimates, actuals, surprises."""
    data = await _get("/benzinga/v1/earnings", {
        "ticker": symbol,
        "limit": 8,
        "sort": "date.desc",
    })

    results = data.get("results", [])
    if not results:
        return f"No earnings data for {symbol}."

    lines = [f"Earnings History — {symbol} (Benzinga via Polygon):"]
    for e in results:
        date = e.get("date", "?")
        period = e.get("fiscal_period", "?")
        year = e.get("fiscal_year", "?")
        status = e.get("date_status", "?")

        eps_est = _safe_float(e.get("estimated_eps"))
        eps_act = _safe_float(e.get("actual_eps"))
        eps_surp = _safe_float(e.get("eps_surprise_percent"))
        rev_est = _safe_float(e.get("estimated_revenue"))
        rev_act = _safe_float(e.get("actual_revenue"))
        rev_surp = _safe_float(e.get("revenue_surprise_percent"))

        eps_line = ""
        if eps_act is not None:
            eps_line = f"EPS: ${eps_act}"
            if eps_est is not None:
                eps_line += f" vs est ${eps_est}"
            if eps_surp is not None:
                eps_line += f" ({eps_surp:+.1f}%)"
        elif eps_est is not None:
            eps_line = f"EPS est: ${eps_est}"

        def _fmt_rev_val(v):
            return f"${v / 1e9:.2f}B" if v > 1e9 else f"${v / 1e6:.0f}M"

        rev_line = ""
        if rev_act is not None:
            rev_line = f"Rev: {_fmt_rev_val(rev_act)}"
            if rev_est is not None:
                rev_line += f" vs est {_fmt_rev_val(rev_est)}"
            if rev_surp is not None:
                rev_line += f" ({rev_surp:+.1f}%)"
        elif rev_est is not None:
            rev_line = f"Rev est: {_fmt_rev_val(rev_est)}"

        parts = [f"  {date} | {period} {year} ({status})"]
        if eps_line:
            parts.append(f"    {eps_line}")
        if rev_line:
            parts.append(f"    {rev_line}")
        lines.append("\n".join(parts))

    return sanitize_ingested_content("\n".join(lines), "polygon:earnings")


async def _get_analyst_ratings(symbol: str) -> str:
    """Get Benzinga analyst ratings — upgrades, downgrades, price targets."""
    data = await _get("/benzinga/v1/ratings", {
        "ticker": symbol,
        "limit": 15,
        "sort": "last_updated.desc",
    })

    results = data.get("results", [])
    if not results:
        return f"No analyst ratings for {symbol}."

    lines = [f"Analyst Ratings — {symbol} (Benzinga via Polygon):"]
    for r in results:
        date = r.get("date", r.get("last_updated", "?"))[:10]
        firm = r.get("firm", r.get("analyst_firm", "?"))
        analyst = r.get("analyst", r.get("analyst_name", ""))
        action = r.get("rating_action", "?")
        old_rating = r.get("previous_rating", "")
        new_rating = r.get("rating", "?")
        old_pt = r.get("previous_adjusted_price_target", r.get("previous_price_target"))
        new_pt = r.get("adjusted_price_target", r.get("price_target"))
        pt_action = r.get("price_target_action", "")

        rating_str = new_rating
        if old_rating and old_rating != new_rating:
            rating_str = f"{old_rating} → {new_rating}"

        pt_str = ""
        if new_pt:
            pt_str = f"PT: ${new_pt}"
            if old_pt and old_pt != new_pt:
                pt_str = f"PT: ${old_pt} → ${new_pt}"
            if pt_action:
                pt_str += f" ({pt_action})"

        analyst_str = f" ({analyst})" if analyst else ""
        lines.append(f"  {date} | {firm}{analyst_str} | {action}: {rating_str} | {pt_str}")

    return sanitize_ingested_content("\n".join(lines), "polygon:ratings")


async def _get_guidance(symbol: str) -> str:
    """Get Benzinga corporate guidance — company outlook, revenue/EPS guidance."""
    data = await _get("/benzinga/v1/guidance", {
        "ticker": symbol,
        "limit": 10,
        "sort": "date.desc",
    })

    results = data.get("results", [])
    if not results:
        return f"No guidance data for {symbol}."

    lines = [f"Corporate Guidance — {symbol} (Benzinga via Polygon):"]
    for g in results:
        date = g.get("date", "?")
        period = g.get("fiscal_period", "?")
        year = g.get("fiscal_year", "?")
        positioning = g.get("positioning", "?")
        company = g.get("company_name", "")

        eps_low = _safe_float(g.get("min_eps_guidance"))
        eps_high = _safe_float(g.get("max_eps_guidance"))
        eps_est = _safe_float(g.get("estimated_eps_guidance"))
        rev_low = _safe_float(g.get("min_revenue_guidance"))
        rev_high = _safe_float(g.get("max_revenue_guidance"))
        rev_est = _safe_float(g.get("estimated_revenue_guidance"))

        parts = [f"  {date} | {period} {year} | Positioning: {positioning}"]

        if eps_low is not None and eps_high is not None:
            eps_str = f"    EPS Guidance: ${eps_low} - ${eps_high}"
            if eps_est is not None:
                eps_str += f" (consensus est: ${eps_est})"
            parts.append(eps_str)

        if rev_low is not None and rev_high is not None:
            def _fmt_rev(v):
                return f"${v / 1e9:.2f}B" if v > 1e9 else f"${v / 1e6:.0f}M"
            rev_str = f"    Rev Guidance: {_fmt_rev(rev_low)} - {_fmt_rev(rev_high)}"
            if rev_est is not None:
                rev_str += f" (consensus est: {_fmt_rev(rev_est)})"
            parts.append(rev_str)

        lines.append("\n".join(parts))

    return sanitize_ingested_content("\n".join(lines), "polygon:guidance")


async def _get_earnings_calendar(symbol: str) -> str:
    """Get TMX/Wall Street Horizon earnings calendar — upcoming dates and events."""
    today = datetime.now(_NY).strftime("%Y-%m-%d")
    future = (datetime.now(_NY) + timedelta(days=90)).strftime("%Y-%m-%d")

    data = await _get("/tmx/v1/corporate-events", {
        "ticker": symbol,
        "type": "earnings_announcement_date",
        "date.gte": today,
        "date.lte": future,
        "limit": 10,
        "sort": "date.asc",
    })

    results = data.get("results", [])
    if not results:
        return f"No upcoming earnings calendar events for {symbol}."

    lines = [f"Earnings Calendar — {symbol} (TMX/Wall Street Horizon via Polygon):"]
    for ev in results:
        date = ev.get("date", "?")
        event_type = ev.get("type", ev.get("event_type", "?"))
        status = ev.get("status", "?")
        timing = ev.get("timing", "")
        notes = ev.get("notes", ev.get("description", ""))

        line = f"  {date} | {event_type} | Status: {status}"
        if timing:
            line += f" | {timing}"
        lines.append(line)
        if notes:
            safe_notes = sanitize_ingested_content(str(notes)[:200], f"tmx:{symbol}")
            lines.append(f"    {safe_notes}")

    return sanitize_ingested_content("\n".join(lines), "polygon_api")


async def get_previous_close(symbol: str) -> dict | None:
    """Get previous day's close — useful for other integrations."""
    try:
        data = await _get(f"/v2/aggs/ticker/{symbol}/prev")
        results = data.get("results", [])
        return results[0] if results else None
    except Exception as e:
        safe_msg = _strip_api_key(str(e)) if "apiKey" in str(e) else str(e)
        logger.error(f"Previous close failed for {symbol}: {safe_msg}")
        return None
