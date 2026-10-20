"""
OpenClaw/Moltbot — Unusual Whales Integration
Handles: get_options_flow, get_darkpool_data, get_market_tide, get_congress_trades
API Docs: https://api.unusualwhales.com/docs

Endpoint paths and auth format matched to Hydrinia v7.0.0 production system
(dpof_machine.py and lazarus_core.py).
"""

import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import asyncio

import httpx

_NY = ZoneInfo("America/New_York")

from agent.config import config
from agent.integrations.uw_pacer import get_uw_pacer
from agent.rate_limiter import budget_and_retry, BudgetExhaustedError

# Singleton httpx client for connection reuse
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        async with _client_lock:
            if _client is None or _client.is_closed:
                _client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)  # S233 CR-4
    return _client


async def close_client():
    global _client
    async with _client_lock:
        if _client is not None:
            await _client.aclose()
            _client = None
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.integrations.unusual_whales")

BASE_URL = None
API_KEY = None


def _init():
    global BASE_URL, API_KEY
    if BASE_URL is None:
        BASE_URL = config.uw_base_url.rstrip("/")
        API_KEY = config.uw_api_key


@budget_and_retry("unusual_whales", max_attempts=3, base_delay=1.0)
async def _get(endpoint: str, params: dict | None = None) -> dict:
    """Authenticated GET to Unusual Whales API with budget tracking and retry.

    Auth: API key in Authorization header WITH 'Bearer' prefix + UW-CLIENT-API-ID.
    This matches the format used by uw_flow_intel.py.
    Budget: 16,000 requests/day default CLIO runaway guard (~10% of UW Startup 160K/day).
    """
    _init()
    if not API_KEY:
        raise ValueError("UW_API_KEY not configured")

    url = f"{BASE_URL}{endpoint}"
    # S118 W-022: Match uw_flow_intel.py auth pattern (Bearer prefix + client ID)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "UW-CLIENT-API-ID": os.getenv("UW_CLIENT_API_ID", "100001"),
        "Accept": "application/json",
    }

    client = await _get_client()
    # S211 (ADR-005): share the process-wide UW pacer so a chat-tool call cannot land inside a DPOF burst.
    # (This module has its own client and does not go through uw_flow_intel._uw_request.)
    await get_uw_pacer().acquire()
    resp = await client.get(url, headers=headers, params=params or {}, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


async def get_options_flow(params: dict[str, Any]) -> str:
    """Monitor unusual options activity — sweeps, blocks, unusual volume.

    Endpoint: /api/option-trades/flow-alerts (matched to Hydrinia dpof_machine.py)
    """
    symbol = params.get("symbol")
    flow_type = params.get("flow_type", "all")
    limit = params.get("limit", 20)

    try:
        request_params = {"limit": limit}
        if symbol:
            request_params["ticker"] = symbol.upper()

        data = await _get("/api/option-trades/flow-alerts", request_params)
        flows = data.get("data", [])

        if not flows:
            target = f" for {symbol}" if symbol else ""
            return f"No unusual options flow found{target}."

        # Filter by flow type
        if flow_type != "all":
            type_map = {
                "sweeps": "SWEEP",
                "blocks": "BLOCK",
                "unusual_volume": "UNUSUAL",
            }
            filter_val = type_map.get(flow_type, "")
            if filter_val:
                flows = [f for f in flows if filter_val in (f.get("type", "") or "").upper()]

        if not flows:
            return f"No {flow_type} options flow found{f' for {symbol}' if symbol else ''}."

        flows = flows[:15]
        lines = [f"Unusual Options Flow{f' — {symbol}' if symbol else ''} ({len(flows)} alerts):"]

        for f in flows:
            ticker = f.get("ticker", f.get("underlying_symbol", "?"))
            strike = f.get("strike", f.get("strike_price", "?"))
            exp = f.get("expiry", f.get("expires_date", "?"))
            side = f.get("put_call", f.get("option_type", "?"))
            premium = f.get("premium", f.get("cost_basis", 0))
            vol = f.get("volume", 0)
            oi = f.get("open_interest", 0)
            sentiment = f.get("sentiment", f.get("trade_side", "?"))
            ftype = f.get("type", f.get("alert_rule", "?"))

            premium_fmt = f"${premium:,.0f}" if isinstance(premium, (int, float)) else str(premium)
            vol_oi = f"V/OI: {vol}/{oi}" if vol and oi else f"Vol: {vol}"

            lines.append(
                f"  {ftype} | {ticker} ${strike} {side} {exp} | "
                f"Premium: {premium_fmt} | {vol_oi} | {sentiment}"
            )

        return sanitize_ingested_content("\n".join(lines), "unusual_whales:options_flow")

    except BudgetExhaustedError as e:
        return f"BUDGET EXHAUSTED: {e}"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "ERROR: Unusual Whales API key invalid or expired."
        if e.response.status_code == 429:
            return "ERROR: Unusual Whales rate limit exceeded. Try again shortly."
        logger.error(f"UW HTTP error: status={e.response.status_code}")
        return f"ERROR: Unusual Whales API error — {e.response.status_code}"
    except Exception as e:
        logger.error(f"Options flow error: {type(e).__name__}")
        return f"ERROR: Options flow failed — {e}"


def _safe_int(val, default=0, label=""):
    """S121 W-26: Added label param for debug tracing on conversion failure."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        if label:
            logger.debug(f"_safe_int conversion failed for {label}: {val!r}")
        return default

def _safe_float(val, default=0.0, label=""):
    """S121 W-26: Added label param for debug tracing on conversion failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        if label:
            logger.debug(f"_safe_float conversion failed for {label}: {val!r}")
        return default

async def get_net_flow_timeline(params: dict[str, Any]) -> str:
    """Get per-ticker net options flow in 5-minute buckets.

    Shows call vs put premium flow over the trading day — like a text version
    of the bigshort.com net flow chart. Useful for spotting large put sells
    (bullish reversal signal) or call dumps (bearish signal).
    """
    symbol = params.get("symbol", "SPY").upper()
    bucket_minutes = params.get("bucket_minutes", 5)
    date = params.get("date", "")

    try:
        from agent.integrations.uw_flow_intel import get_ticker_flow_timeline
        buckets = await get_ticker_flow_timeline(symbol, bucket_minutes, date)

        if not buckets:
            return f"No net flow data available for {symbol}."

        def _fmt_prem(val):
            """Format premium as $X.XM or $XXK."""
            av = abs(val)
            if av >= 1_000_000:
                return f"{'+'if val >= 0 else '-'}${av/1e6:.1f}M"
            if av >= 1_000:
                return f"{'+'if val >= 0 else '-'}${av/1e3:.0f}K"
            return f"{'+'if val >= 0 else '-'}${av:.0f}"

        def _bar(val, max_abs):
            """Mini ASCII bar for net flow direction."""
            if max_abs == 0:
                return ""
            length = int(abs(val) / max_abs * 8)
            if val > 0:
                return "+" * max(1, length)
            elif val < 0:
                return "-" * max(1, length)
            return "."

        # Find max for bar scaling
        max_abs = max((abs(b["net_flow"]) for b in buckets), default=1)

        # Header
        date_str = date or datetime.now(_NY).strftime("%Y-%m-%d")
        lines = [f"{symbol} Net Options Flow ({bucket_minutes}-min) — {date_str}"]
        lines.append("(times ET; Net Put > 0 = puts net bought = bearish; Net Flow = Net Call - Net Put)\n")
        lines.append(f"{'Time':<7}{'Net Call':>10}{'Net Put':>10}{'Net Flow':>10}  {'Direction'}")
        lines.append("-" * 58)

        # Accumulate totals
        total_call = 0.0
        total_put = 0.0
        peak_bull = ("", 0.0)
        peak_bear = ("", 0.0)

        for b in buckets:
            total_call += b["net_call"]
            total_put += b["net_put"]
            if b["net_flow"] > peak_bull[1]:
                peak_bull = (b["time"], b["net_flow"])
            if b["net_flow"] < peak_bear[1]:
                peak_bear = (b["time"], b["net_flow"])

            lines.append(
                f"{b['time']:<7}"
                f"{_fmt_prem(b['net_call']):>10}"
                f"{_fmt_prem(b['net_put']):>10}"
                f"{_fmt_prem(b['net_flow']):>10}  "
                f"{_bar(b['net_flow'], max_abs)}"
            )

        # Summary
        total_flow = total_call - total_put  # net_put > 0 = puts bought (bearish)
        bias = "CALL BIAS" if total_flow > 0 else "PUT BIAS" if total_flow < 0 else "NEUTRAL"
        lines.append("-" * 58)
        lines.append(f"Total: {_fmt_prem(total_flow)} {bias}")
        lines.append(f"  Calls: {_fmt_prem(total_call)} | Puts: {_fmt_prem(total_put)}")
        if peak_bull[0]:
            lines.append(f"  Peak bull: {peak_bull[0]} ({_fmt_prem(peak_bull[1])})")
        if peak_bear[0]:
            lines.append(f"  Peak bear: {peak_bear[0]} ({_fmt_prem(peak_bear[1])})")

        return sanitize_ingested_content("\n".join(lines), "unusual_whales:net_flow_timeline")

    except BudgetExhaustedError as e:
        return f"BUDGET EXHAUSTED: {e}"
    except Exception as e:
        logger.error(f"Net flow timeline error: {type(e).__name__}: {e}")
        return f"ERROR: Net flow timeline failed — {e}"


async def get_darkpool_data(params: dict[str, Any]) -> str:
    """Fetch dark pool trading data for a symbol.

    Endpoint: /api/darkpool/{ticker} (matched to Hydrinia dpof_machine.py)
    """
    symbol = params["symbol"].upper()

    try:
        data = await _get(f"/api/darkpool/{symbol}", {"limit": 50})
        prints = data.get("data", [])

        if not prints:
            return f"No dark pool data for {symbol}."

        total_volume = 0
        total_notional = 0.0
        buys = 0
        sells = 0
        neutral = 0

        for p in prints:
            vol = _safe_int(p.get("volume", p.get("size", 0)))
            price = _safe_float(p.get("price", p.get("average_price", 0)))
            total_volume += vol
            total_notional += vol * price

            # Classify direction based on available fields
            tracking = (p.get("tracking", "") or "").upper()
            sentiment = (p.get("sentiment", "") or "").upper()
            if "BUY" in tracking or "BULLISH" in sentiment:
                buys += 1
            elif "SELL" in tracking or "BEARISH" in sentiment:
                sells += 1
            else:
                neutral += 1

        # Determine overall sentiment
        if buys > sells * 1.5:
            overall = "ACCUMULATION (bullish)"
        elif sells > buys * 1.5:
            overall = "DISTRIBUTION (bearish)"
        else:
            overall = "NEUTRAL"

        lines = [
            f"Dark Pool Activity — {symbol}:",
            f"  Total Prints: {len(prints)}",
            f"  Total Volume: {total_volume:,}",
            f"  Total Notional: ${total_notional:,.0f}",
            f"  Buy/Sell/Neutral: {buys}/{sells}/{neutral}",
            f"  Overall Signal: {overall}",
            "",
            "  Largest Prints:",
        ]

        sorted_prints = sorted(
            prints,
            key=lambda x: _safe_int(x.get("volume", x.get("size", 0))) * _safe_float(x.get("price", x.get("average_price", 0))),
            reverse=True,
        )
        for p in sorted_prints[:10]:
            vol = _safe_int(p.get("volume", p.get("size", 0)))
            price = _safe_float(p.get("price", p.get("average_price", 0)))
            notional = vol * price if vol and price else 0
            ts = p.get("executed_at", p.get("date", p.get("timestamp", "?")))
            tracking = p.get("tracking", p.get("sentiment", "?"))

            lines.append(
                f"    ${price:.2f} x {vol:,} = ${notional:,.0f} | {tracking} | {ts}"
            )

        return sanitize_ingested_content("\n".join(lines), "unusual_whales:darkpool")

    except BudgetExhaustedError as e:
        return f"BUDGET EXHAUSTED: {e}"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "ERROR: Unusual Whales API key invalid or expired."
        if e.response.status_code == 429:
            return "ERROR: Rate limit exceeded. Try again shortly."
        logger.error(f"UW dark pool error: status={e.response.status_code}")
        return f"ERROR: Dark pool API error — {e.response.status_code}"
    except Exception as e:
        logger.error(f"Dark pool error: {type(e).__name__}")
        return f"ERROR: Dark pool failed for {symbol} — {e}"


async def get_market_tide(params: dict[str, Any]) -> str:
    """Get market-wide bullish/bearish sentiment from options flow.

    Endpoint: /api/market/tide (matched to Hydrinia dpof_machine.py)
    """
    try:
        data = await _get("/api/market/tide")
        tide = data.get("data", data)

        if not tide:
            return "No market tide data available."

        bullish = tide.get("bullish_premium", tide.get("bullish", 0))
        bearish = tide.get("bearish_premium", tide.get("bearish", 0))
        neutral = tide.get("neutral", 0)

        lines = [
            "Market Tide (Unusual Whales):",
            f"  Bullish: {bullish}",
            f"  Bearish: {bearish}",
        ]
        if neutral:
            lines.append(f"  Neutral: {neutral}")

        return sanitize_ingested_content("\n".join(lines), "unusual_whales:market_tide")

    except BudgetExhaustedError as e:
        return f"BUDGET EXHAUSTED: {e}"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.debug("Market tide endpoint returned 404 (intermittent UW API issue)")
            return "Market tide data temporarily unavailable."
        if e.response.status_code == 401:
            return "ERROR: UW API authentication failed."
        if e.response.status_code == 429:
            return "ERROR: UW API rate limited. Try again shortly."
        logger.error(f"Market tide error: {e}")
        return f"ERROR: Market tide failed — {e.response.status_code}"
    except Exception as e:
        logger.error(f"Market tide error: {e}")
        return f"ERROR: Market tide failed — {e}"


async def get_congress_trades(params: dict[str, Any]) -> str:
    """Get recent congressional trading activity.

    Endpoint: /api/congress/recent-trades (matched to Hydrinia lazarus_core.py)
    """
    symbol = params.get("symbol")

    try:
        request_params = {}
        if symbol:
            request_params["ticker"] = symbol.upper()

        data = await _get("/api/congress/recent-trades", request_params)
        trades = data.get("data", [])

        if not trades:
            target = f" for {symbol}" if symbol else ""
            return f"No congressional trades found{target}."

        trades = trades[:15]
        lines = [f"Congressional Trades{f' — {symbol}' if symbol else ''} ({len(trades)} recent):"]

        for t in trades:
            politician = t.get("politician", t.get("representative", "?"))
            ticker = t.get("ticker", "?")
            txn_type = t.get("transaction_type", t.get("type", "?"))
            amount = t.get("amount", "?")
            date = t.get("transaction_date", t.get("date", "?"))

            lines.append(f"  {politician} | {ticker} | {txn_type} | {amount} | {date}")

        return sanitize_ingested_content("\n".join(lines), "unusual_whales:congress_trades")

    except BudgetExhaustedError as e:
        return f"BUDGET EXHAUSTED: {e}"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "ERROR: UW API authentication failed."
        if e.response.status_code == 429:
            return "ERROR: UW API rate limited. Try again shortly."
        logger.error(f"Congress trades error: {e}")
        return f"ERROR: Congress trades failed — {e.response.status_code}"
    except Exception as e:
        logger.error(f"Congress trades error: {e}")
        return f"ERROR: Congress trades failed — {e}"
