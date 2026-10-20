"""
OpenClaw/Moltbot — Kalshi Prediction Markets Integration
Handles: get_prediction_consensus, get_prediction_markets, get_prediction_event
API Docs: https://docs.kalshi.com

Provides market-implied probability distributions for macro events
(Fed rate decisions, CPI, GDP, etc.) as consensus signals.

Auth: RSA-PSS signed requests (no SDK dependency — direct httpx).
"""

import asyncio
import base64
import logging
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from agent.config import config
from agent.rate_limiter import retry_on_transient
from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.integrations.kalshi")

# S159 M-18: Config-derived path instead of hardcoded /home/agent/openclaw
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = None
API_KEY = None
_private_key = None

# In-memory cache: {cache_key: {"data": ..., "ts": float}}
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()  # S121 W-27: Thread-safe cache access
CACHE_TTL = 3600  # 1 hour — prediction market odds don't change rapidly

# Topic → Kalshi series tickers (ordered by priority)
TOPIC_SERIES_MAP = {
    "fed": ["KXFEDDECISION", "KXFED"],
    "cpi": ["KXCPI", "KXCPIYOY"],
    "gdp": ["KXGDP"],
    "payrolls": ["KXPAYROLLS"],
    "jobs": ["KXPAYROLLS"],
    "recession": ["KXRECSSNBER"],
    "rates": ["KXRATECUTCOUNT", "KXFED"],
    "rate_cuts": ["KXRATECUTCOUNT", "KXFED"],
}


_init_lock = threading.Lock()  # S153: Thread-safe init


def _init():
    global BASE_URL, API_KEY, _private_key
    if BASE_URL is not None:
        return
    with _init_lock:
        if BASE_URL is not None:
            return  # Double-checked locking
        BASE_URL = config.kalshi_base_url.rstrip("/")
        API_KEY = config.kalshi_api_key

        pem_path = config.kalshi_private_key_path
        if pem_path and not os.path.isabs(pem_path):
            pem_path = os.path.join(_PROJECT_ROOT, pem_path)
        if pem_path and os.path.exists(pem_path):
            with open(pem_path, "rb") as f:
                _private_key = load_pem_private_key(f.read(), password=None)
            logger.info(f"Kalshi private key loaded from {pem_path}")
        else:
            logger.warning(f"Kalshi private key not found at {pem_path}")


def _resolve_pem_path() -> str | None:
    """Resolve the configured pem to an absolute path IF it exists on disk, else None.
    Keep in sync with _init()'s pem resolution (L69-77)."""
    pem = config.kalshi_private_key_path
    if pem and not os.path.isabs(pem):
        pem = os.path.join(_PROJECT_ROOT, pem)
    return pem if (pem and os.path.exists(pem)) else None


def is_available() -> bool:
    """Kalshi is usable only if BOTH the API key AND a resolvable private-key pem exist,
    so tool EXPOSURE matches SIGNABILITY (prevents offering a tool that fails at signing
    with 'Kalshi API key or private key not configured')."""
    return bool(config.kalshi_api_key) and _resolve_pem_path() is not None


def _sign_headers(method: str, path: str) -> dict[str, str]:
    """Generate RSA-PSS signed auth headers for Kalshi API."""
    if _private_key is None:
        raise ValueError("Kalshi private key not loaded — cannot sign request")
    ts = str(int(time.time() * 1000))
    message = ts + method.upper() + path
    signature = _private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": API_KEY,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


def _cache_get(key: str) -> Any | None:
    with _cache_lock:
        cached = _cache.get(key)
        if cached and time.time() - cached["ts"] < CACHE_TTL:
            return cached["data"]
        return None


_MAX_CACHE_ENTRIES = 200


def _cache_set(key: str, data: Any) -> None:
    with _cache_lock:
        # Evict expired entries if cache exceeds max size
        if len(_cache) > _MAX_CACHE_ENTRIES:
            now = time.time()
            expired = [k for k, v in _cache.items() if now - v["ts"] > CACHE_TTL]
            for k in expired:
                del _cache[k]
            # S156 M-A: Hard evict oldest 10% if still over cap after expiry cleanup
            if len(_cache) > _MAX_CACHE_ENTRIES:
                sorted_keys = sorted(_cache.keys(), key=lambda k: _cache[k]["ts"])
                for k in sorted_keys[:len(sorted_keys) // 10 + 1]:
                    del _cache[k]
        _cache[key] = {"data": data, "ts": time.time()}


# Singleton httpx client for connection reuse (matches polygon.py pattern — P1-10)
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


@retry_on_transient(max_attempts=3, base_delay=1.0)
async def _get(endpoint: str, params: dict | None = None) -> dict:
    """Authenticated GET to Kalshi API with retry on transient errors."""
    _init()
    if not API_KEY or not _private_key:
        raise ValueError("Kalshi API key or private key not configured")

    url = f"{BASE_URL}{endpoint}"
    # Sign with path only (no query params)
    parsed = urlparse(url)
    sign_path = parsed.path
    headers = _sign_headers("GET", sign_path)
    headers["Accept"] = "application/json"

    client = await _get_client()
    resp = await client.get(url, headers=headers, params=params or {})
    resp.raise_for_status()
    return resp.json()


def _price_to_pct(price_str: str | None) -> float:
    """Convert FixedPointDollars string (e.g., '0.5600') to percentage (56.0)."""
    if not price_str:
        return 0.0
    try:
        return float(price_str) * 100
    except (ValueError, TypeError):
        return 0.0


def _midpoint_pct(yes_bid: str | None, yes_ask: str | None) -> float:
    """Calculate midpoint probability from bid/ask as percentage."""
    bid = _price_to_pct(yes_bid)
    ask = _price_to_pct(yes_ask)
    # S136 M1: Check raw inputs (not parsed floats — _price_to_pct always returns float, never None)
    if yes_bid is not None and yes_ask is not None:
        return (bid + ask) / 2
    if yes_bid is not None:
        return bid
    if yes_ask is not None:
        return ask
    return 0.0


# ============================================================
# Public Functions
# ============================================================

async def get_prediction_consensus(params: dict[str, Any]) -> str:
    """Get market-implied probability distribution for a macro topic.

    Maps topics (fed, cpi, gdp, etc.) to Kalshi series tickers,
    finds the nearest open event, and builds a probability distribution.
    """
    # S118 W-023: Use .get() to avoid KeyError on missing param
    topic = params.get("topic", "")
    if not topic:
        return "Error: 'topic' parameter is required"
    topic = topic.lower()
    series_list = TOPIC_SERIES_MAP.get(topic)
    if not series_list:
        valid = ", ".join(sorted(TOPIC_SERIES_MAP.keys()))
        return f"Unknown topic: {topic}. Valid topics: {valid}"

    try:
        all_sections = []

        for series_ticker in series_list:
            # Check cache
            cache_key = f"consensus:{series_ticker}"
            cached = _cache_get(cache_key)
            if cached:
                all_sections.append(cached)
                continue

            # Find the nearest open event for this series
            events_data = await _get("/events", {
                "series_ticker": series_ticker,
                "status": "open",
                "limit": 3,
            })
            events = events_data.get("events", [])
            if not events:
                continue

            # Take the nearest event (first in list, sorted by strike_date)
            event = events[0]
            event_ticker = event["event_ticker"]
            event_title = event.get("title", event_ticker)

            # Fetch all markets for this event
            markets_data = await _get("/markets", {
                "event_ticker": event_ticker,
                "limit": 50,
            })
            markets = markets_data.get("markets", [])
            if not markets:
                continue

            # Build probability distribution
            entries = []
            total_volume = 0
            for m in markets:
                prob = _midpoint_pct(
                    m.get("yes_bid_dollars"),
                    m.get("yes_ask_dollars"),
                )
                last = _price_to_pct(m.get("last_price_dollars"))
                try:
                    vol = float(m.get("volume_fp", "0") or "0")
                except (ValueError, TypeError):
                    vol = 0
                try:
                    oi = float(m.get("open_interest_fp", "0") or "0")
                except (ValueError, TypeError):
                    oi = 0
                total_volume += vol

                # Extract human-readable outcome from subtitle or ticker
                subtitle = m.get("subtitle", m.get("title", m.get("ticker", "?")))

                entries.append({
                    "label": subtitle,
                    "prob": prob,
                    "last": last,
                    "vol": vol,
                    "oi": oi,
                    "ticker": m.get("ticker", "?"),
                    "strike": m.get("floor_strike", m.get("functional_strike")),
                })

            # Sort: by strike if available, else by probability descending
            if entries and entries[0].get("strike") is not None:
                entries.sort(key=lambda e: float(e["strike"] or 0), reverse=True)
            else:
                entries.sort(key=lambda e: e["prob"], reverse=True)

            # Format output
            lines = [f"Kalshi Consensus — {event_title}:"]
            for e in entries:
                if e["prob"] < 1.0 and e["last"] < 1.0:
                    continue  # Skip near-zero markets
                prob_str = f"{e['prob']:5.1f}%"
                last_str = f"(last: {e['last']:.0f}%)" if e["last"] else ""
                vol_str = f"Vol: {e['vol']:,.0f}" if e["vol"] else ""
                lines.append(f"  {e['label'][:60]:<60} {prob_str} {last_str} {vol_str}")

            lines.append(f"\n  Source: {event_ticker} | Total Vol: {total_volume:,.0f}")

            section = "\n".join(lines)
            _cache_set(cache_key, section)
            all_sections.append(section)

        if not all_sections:
            return f"No open Kalshi markets found for topic '{topic}'."

        return sanitize_ingested_content(
            "\n\n".join(all_sections),
            f"kalshi:consensus:{topic}",
        )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "ERROR: Kalshi API authentication failed. Check API key and private key."
        # S121 W-32: No exc_info=True — stack trace may expose auth headers
        logger.error(f"Kalshi HTTP error: {e.response.status_code}")
        return f"ERROR: Kalshi API error — {e.response.status_code}"
    except Exception as e:
        # S121 W-32: No exc_info=True — stack trace may expose auth headers
        logger.error(f"Kalshi consensus error: {type(e).__name__}: {e}")
        return f"ERROR: Kalshi consensus failed — {e}"


async def get_prediction_markets(params: dict[str, Any]) -> str:
    """Search/browse Kalshi prediction markets by keywords or series ticker."""
    keywords = params.get("keywords", "")
    series = params.get("series", "")
    limit = min(params.get("limit", 10), 30)

    try:
        markets = []

        if series:
            # Direct series query
            cache_key = f"markets:{series}"
            cached = _cache_get(cache_key)
            if cached:
                markets = cached
            else:
                data = await _get("/markets", {
                    "series_ticker": series,
                    "status": "open",
                    "limit": limit,
                })
                markets = data.get("markets", [])
                _cache_set(cache_key, markets)

        elif keywords:
            # Search events by keyword, then get their markets
            cache_key = f"search:{keywords.lower()}"
            cached = _cache_get(cache_key)
            if cached:
                markets = cached
            else:
                events_data = await _get("/events", {
                    "status": "open",
                    "limit": 100,
                })
                events = events_data.get("events", [])

                # Filter by keyword
                kw_lower = keywords.lower()
                matching_events = [
                    e for e in events
                    if kw_lower in (e.get("title", "") or "").lower()
                    or kw_lower in (e.get("category", "") or "").lower()
                ]

                # Get markets for matching events (limit API calls)
                for evt in matching_events[:5]:
                    evt_data = await _get("/markets", {
                        "event_ticker": evt["event_ticker"],
                        "limit": 10,
                    })
                    markets.extend(evt_data.get("markets", []))

                _cache_set(cache_key, markets)
        else:
            return "Please provide either 'keywords' or 'series' to search."

        if not markets:
            target = f" for series '{series}'" if series else f" matching '{keywords}'"
            return f"No open Kalshi markets found{target}."

        # Format output
        lines = [f"Kalshi Markets ({len(markets[:limit])} results):"]
        for m in markets[:limit]:
            ticker = m.get("ticker", "?")
            title = m.get("subtitle", m.get("title", ticker))
            prob = _midpoint_pct(m.get("yes_bid_dollars"), m.get("yes_ask_dollars"))
            last = _price_to_pct(m.get("last_price_dollars"))
            try:
                vol = float(m.get("volume_fp", "0") or "0")
            except (TypeError, ValueError):
                vol = 0.0

            prob_str = f"{prob:.0f}%" if prob else "N/A"
            last_str = f"last {last:.0f}%" if last else ""
            vol_str = f"vol {vol:,.0f}" if vol else ""

            lines.append(f"  {ticker}: {title[:50]} | {prob_str} {last_str} {vol_str}")

        return sanitize_ingested_content(
            "\n".join(lines),
            f"kalshi:markets:{series or keywords}",
        )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "ERROR: Kalshi API authentication failed."
        # S121 W-32: No exc_info=True — stack trace may expose auth headers
        logger.error(f"Kalshi markets error: {e.response.status_code}")
        return f"ERROR: Kalshi API error — {e.response.status_code}"
    except Exception as e:
        # S121 W-32: No exc_info=True — stack trace may expose auth headers
        logger.error(f"Kalshi markets error: {type(e).__name__}: {e}")
        return f"ERROR: Kalshi markets search failed — {e}"


async def get_prediction_event(params: dict[str, Any]) -> str:
    """Get detailed probability distribution for a specific Kalshi event."""
    # S118 W-023: Use .get() to avoid KeyError on missing param
    event_ticker = params.get("event_ticker", "")
    if not event_ticker:
        return "Error: 'event_ticker' parameter is required"
    event_ticker = event_ticker.upper()

    try:
        # Check cache
        cache_key = f"event:{event_ticker}"
        cached = _cache_get(cache_key)

        if cached:
            event_data = cached
        else:
            event_data = await _get(
                f"/events/{event_ticker}",
                {"with_nested_markets": "true"},
            )
            _cache_set(cache_key, event_data)

        event = event_data.get("event", event_data)
        title = event.get("title", event_ticker)
        category = event.get("category", "")
        markets = event.get("markets", [])

        if not markets:
            # If no nested markets, fetch separately
            markets_data = await _get("/markets", {
                "event_ticker": event_ticker,
                "limit": 50,
            })
            markets = markets_data.get("markets", [])

        if not markets:
            return f"No markets found for event {event_ticker}."

        # Build detailed distribution
        entries = []
        total_vol = 0
        total_oi = 0

        for m in markets:
            yes_bid = _price_to_pct(m.get("yes_bid_dollars"))
            yes_ask = _price_to_pct(m.get("yes_ask_dollars"))
            mid = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else yes_bid or yes_ask
            last = _price_to_pct(m.get("last_price_dollars"))
            try:
                vol = float(m.get("volume_fp", "0") or "0")
            except (TypeError, ValueError):
                vol = 0.0
            try:
                oi = float(m.get("open_interest_fp", "0") or "0")
            except (TypeError, ValueError):
                oi = 0.0
            spread = yes_ask - yes_bid if yes_bid and yes_ask else 0
            total_vol += vol
            total_oi += oi

            subtitle = m.get("subtitle", m.get("title", m.get("ticker", "?")))
            strike = m.get("floor_strike", m.get("functional_strike"))

            entries.append({
                "label": subtitle,
                "mid": mid,
                "bid": yes_bid,
                "ask": yes_ask,
                "spread": spread,
                "last": last,
                "vol": vol,
                "oi": oi,
                "strike": strike,
                "ticker": m.get("ticker", "?"),
            })

        # Sort by strike if available
        if entries and entries[0].get("strike") is not None:
            entries.sort(key=lambda e: float(e["strike"] or 0), reverse=True)
        else:
            entries.sort(key=lambda e: e["mid"], reverse=True)

        # Format
        lines = [
            f"Kalshi Event — {title}",
            f"  Category: {category} | Event: {event_ticker}",
            f"  Total Volume: {total_vol:,.0f} | Total OI: {total_oi:,.0f}",
            "",
            f"  {'Outcome':<50} {'Mid':>6} {'Bid':>6} {'Ask':>6} {'Spread':>7} {'Last':>6} {'Vol':>8} {'OI':>8}",
            f"  {'-'*50} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*8} {'-'*8}",
        ]

        for e in entries:
            if e["mid"] < 0.5 and e["last"] < 0.5:
                continue  # Skip near-zero markets
            label = e["label"][:50]
            lines.append(
                f"  {label:<50} {e['mid']:5.1f}% {e['bid']:5.1f}% {e['ask']:5.1f}% "
                f"{e['spread']:5.1f}pp {e['last']:5.1f}% {e['vol']:>7,.0f} {e['oi']:>7,.0f}"
            )

        return sanitize_ingested_content(
            "\n".join(lines),
            f"kalshi:event:{event_ticker}",
        )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "ERROR: Kalshi API authentication failed."
        if e.response.status_code == 404:
            return f"ERROR: Event '{event_ticker}' not found on Kalshi."
        # S121 W-32: No exc_info=True — stack trace may expose auth headers
        logger.error(f"Kalshi event error: {e.response.status_code}")
        return f"ERROR: Kalshi API error — {e.response.status_code}"
    except Exception as e:
        # S121 W-32: No exc_info=True — stack trace may expose auth headers
        logger.error(f"Kalshi event error: {type(e).__name__}: {e}")
        return f"ERROR: Kalshi event detail failed — {e}"
