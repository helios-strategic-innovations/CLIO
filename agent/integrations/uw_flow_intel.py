"""
Unusual Whales institutional flow intelligence for Moltbot.

5-pillar signal engine: NOFA, Net Premium Flow, Dark Pool, GEX, Congress/Insider.
Provides cached composite scoring for downstream consumers (premarket scan, trade ideas,
alignment checker, system prompt, hourly debrief, Hydrinia bridge).

Feature-gated by the UW_API_KEY env var, read once at startup through agent.config
(empty = silent no-op everywhere; a new key needs a moltbot restart).
"""

import asyncio
import json
import logging
import math
import os
import threading
import time as _time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from agent.integrations.uw_pacer import backoff_seconds_429, get_uw_pacer, note_429_and_log

logger = logging.getLogger(__name__)
_NY = ZoneInfo("America/New_York")

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

_API_BASE = "https://api.unusualwhales.com/api"
_REQUEST_DELAY = 0.3  # seconds between API calls (avoids rate limiting)
_TIMEOUT = 15  # seconds per HTTP request
_SCAN_TIMEOUT = 180  # S162: max wall-clock for entire 5-pillar scan

# Cache TTLs
_CACHE_TTL_MARKET = 900     # 15 min during market hours
_CACHE_TTL_OFFHOURS = 3600  # 60 min outside market hours
_TIDE_TTL = 300             # 5 min for market tide (more time-sensitive)

# Composite weights
_WEIGHTS = {
    "nofa": 0.35,
    "premium": 0.25,
    "darkpool": 0.20,
    "gex": 0.10,
    "congress": 0.10,
}

# ─── TICKER UNIVERSE ─────────────────────────────────────────────────────────

TIER1 = [
    "AVAV", "LMT", "RTX", "NOC", "GD", "PLTR", "HII", "LHX",
    "NVDA", "AMD", "TSM", "AVGO", "MRVL", "QCOM", "MU",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    "XOM", "CVX", "COP", "HAL", "SLB", "OXY",
    "NEM", "GOLD", "AEM",
    "MRNA", "LLY", "UNH", "REGN", "VRTX",
    "ZIM", "DAC", "GOGL", "RKLB",
    "BABA", "PDD", "JD",
]

ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLE", "XLF", "XLK", "XBI",
    "DFEN", "ITA", "XAR", "GLD", "GDX", "SLV", "USO", "UNG",
    "TLT", "HYG", "FXI", "KWEB",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "UVIX", "SVIX", "SPXL",
    "LABU", "LABD", "BOIL", "KOLD",
]

ALL_TICKERS = TIER1 + ETFS

# Focused subsets for expensive per-ticker endpoints
_PREMIUM_TICKERS = TIER1[:20]
_DARKPOOL_TICKERS = TIER1[:15]
_GEX_TICKERS = [
    "SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMD",
    "AMZN", "META", "GOOGL", "AVAV", "PLTR", "XLE", "GLD", "TLT",
]

# ─── MODULE STATE ────────────────────────────────────────────────────────────

_cache: dict = {}
_cache_time: float = 0.0
_tide_cache: dict = {}
_tide_cache_time: float = 0.0
_cache_lock = threading.Lock()
_client_lock = threading.Lock()
_client = None  # httpx.AsyncClient singleton
_req_count: int = 0
_usage: dict = {"limit": None, "used": None, "remaining": None}
_quota_coordinator = None

# Auth-failure backoff: with a revoked/invalid key every caller (DPOF each cycle,
# NewsFlash, vol surface, econ calendar, scans) would hit UW and log a 401 ERROR —
# thousands a day. After a 401, every UW call returns {} (no request, no budget
# charge) for _AUTH_BACKOFF_SECONDS; then one probe request decides again.
_AUTH_BACKOFF_SECONDS = 900
_auth_blocked_until = 0.0     # time.monotonic() deadline; 0.0 = not blocked
_auth_fail_count = 0          # consecutive 401 blocks (logging only)
_auth_probe_in_flight = False # only ONE request probes once the pause elapses (callers gather ~14 at once)


def uw_auth_backoff_active() -> bool:
    """True while UW calls are being short-circuited by the 401 backoff (paused,
    or the single probe request is already in flight). Callers that meter their
    own UW attempts (e.g. DPOF's daily cap) use this to skip no-op calls."""
    if not _auth_blocked_until:
        return False
    return _auth_probe_in_flight or _time.monotonic() < _auth_blocked_until


def _note_auth_failure(endpoint: str) -> None:
    global _auth_blocked_until, _auth_fail_count
    if _auth_blocked_until and _time.monotonic() < _auth_blocked_until:
        # A request already in flight when the pause began (e.g. the first
        # discovery of a dead key while ~14 DPOF calls were outstanding) —
        # the pause is already set; don't extend it, count it or log it again.
        return
    _auth_fail_count += 1
    _auth_blocked_until = _time.monotonic() + _AUTH_BACKOFF_SECONDS
    if _auth_fail_count == 1:
        logger.error(
            f"UW AUTH FAILED (401) on {endpoint}. Check UW_API_KEY in .env (read at startup: restart after changing it). "
            f"Pausing all UW calls for {_AUTH_BACKOFF_SECONDS // 60} min between probes."
        )
    else:
        logger.warning(
            f"UW auth still failing (401, probe #{_auth_fail_count}); UW calls paused another "
            f"{_AUTH_BACKOFF_SECONDS // 60} min"
        )


def _note_auth_ok() -> None:
    """Any non-401 answer means the key authenticated: lift the pause."""
    global _auth_blocked_until, _auth_fail_count
    if _auth_fail_count:
        logger.info(f"UW auth recovered after {_auth_fail_count} failed probe(s) — UW calls resumed")
    _auth_fail_count = 0
    _auth_blocked_until = 0.0


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Get UW API key from config. Returns empty string if not configured."""
    try:
        from agent.config import config
        return config.uw_api_key or ""
    except Exception as e:
        logger.debug(f"UW API key retrieval failed: {e}")
        return ""


def _is_market_hours() -> bool:
    """Check if within US market hours (9:30 AM – 4:00 PM ET weekdays)."""
    now = datetime.now(_NY)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def _cache_ttl() -> float:
    """Get appropriate cache TTL based on market hours."""
    return _CACHE_TTL_MARKET if _is_market_hours() else _CACHE_TTL_OFFHOURS


async def _get_client():
    """Get or create httpx.AsyncClient singleton."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        import httpx
        _client = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True)
        return _client


def _get_quota_coordinator():
    global _quota_coordinator
    if _quota_coordinator is not None:
        return _quota_coordinator
    with _client_lock:
        if _quota_coordinator is None:
            from agent.integrations.uw_quota_client import SupabaseUWQuotaClient
            _quota_coordinator = SupabaseUWQuotaClient(caller="clio-uw-flow-intel")
    return _quota_coordinator


def _priority_for_endpoint(endpoint: str) -> int:
    normalized = (endpoint or "").split("?", 1)[0].lower()
    if any(token in normalized for token in (
        "market-tide", "net-prem-ticks", "flow-alerts", "darkpool",
        "volatility", "greeks", "spot-exposures", "interpolated-iv",
    )):
        return 0
    if any(token in normalized for token in ("congress", "insider", "institution", "screener")):
        return 2
    return 1


async def close_client():
    """Close and nil the httpx.AsyncClient singleton for graceful shutdown."""
    global _client
    with _client_lock:
        if _client is not None:
            try:
                await _client.aclose()
            except Exception as e:
                logger.debug("UW httpx client close error: %s", e)
            _client = None
            logger.info("UW httpx client closed")


# ─── API LAYER ───────────────────────────────────────────────────────────────

# UW-5 (S211): when the ledger's last header observation is stale (overnight: no P0 call can refresh
# it) every P1/P2 request is denied by design -- 136 identical WARNINGs in 13 h. Keep the signal,
# lose the flood: one WARNING per (endpoint, reason) per 10 min carrying a suppressed count.
_QUOTA_DENIED_LOG_INTERVAL_S = 600.0
# R-13: DEGRADED_STALE_CACHE is an expected state until a 200 refreshes the ledger -> INFO, once per episode
# (then at most every 30 min). Real denials (budget / priority reserves) keep the WARNING.
_QUOTA_DENIED_STALE_LOG_INTERVAL_S = 1800.0
_quota_denied_log_at: dict = {}
_quota_denied_suppressed: dict = {}


def _log_quota_denied(endpoint: str, decision) -> None:
    key = (endpoint, str(decision.reason_code))
    now = _time.monotonic()
    last = _quota_denied_log_at.get(key)
    stale = str(decision.reason_code) == "DEGRADED_STALE_CACHE"
    interval = _QUOTA_DENIED_STALE_LOG_INTERVAL_S if stale else _QUOTA_DENIED_LOG_INTERVAL_S
    if last is None or now - last >= interval:
        suppressed = _quota_denied_suppressed.pop(key, 0)
        _quota_denied_log_at[key] = now
        logger.log(
            logging.INFO if stale else logging.WARNING,
            "UW_ACCOUNT_QUOTA_DENIED caller=clio endpoint=%s reason=%s source=%s remaining=%s suppressed=%d",
            endpoint,
            decision.reason_code,
            decision.source,
            decision.remaining,
            suppressed,
        )
    else:
        _quota_denied_suppressed[key] = _quota_denied_suppressed.get(key, 0) + 1
        logger.debug("UW_ACCOUNT_QUOTA_DENIED endpoint=%s reason=%s (deduplicated)", endpoint, decision.reason_code)


async def _pace_request() -> None:
    """Space UW requests process-wide (UW-1, S211; ADR-005).

    UW Basic 429s bursts (12-14 same-second requests -> ~31% rejected; <=3 -> none). The shared
    pacer hands out slots ~1/rate apart. CLIO_UW_PACER_ENABLED=0 restores the legacy per-call sleep.
    """
    pacer = get_uw_pacer()
    if pacer.enabled:
        await pacer.acquire()
    else:
        await asyncio.sleep(_REQUEST_DELAY)


async def _uw_get(endpoint: str, params: dict | None = None, *, skip_moltbot_budget: bool = False) -> dict:
    """Make authenticated GET request to Unusual Whales API.

    Handles rate limiting (429), auth errors (401), tier limits (403).
    Records budget usage per call unless skip_moltbot_budget=True (CLIO path).
    After a 401 all calls return {} for _AUTH_BACKOFF_SECONDS; then exactly one
    call probes while any concurrent callers keep getting {}.
    """
    global _auth_probe_in_flight

    api_key = _get_api_key()
    if not api_key:
        return {}
    is_probe = False
    if _auth_blocked_until:
        if _auth_probe_in_flight or _time.monotonic() < _auth_blocked_until:
            return {}
        _auth_probe_in_flight = True
        is_probe = True
    try:
        return await _uw_request(endpoint, params, api_key, skip_moltbot_budget=skip_moltbot_budget)
    finally:
        if is_probe:
            _auth_probe_in_flight = False


async def _uw_request(endpoint: str, params: dict | None, api_key: str, *, skip_moltbot_budget: bool) -> dict:
    global _req_count, _usage

    # Budget check — skip when called from CLIO bridge (has its own budget)
    if not skip_moltbot_budget:
        try:
            from agent.rate_limiter import check_and_record
            await check_and_record("unusual_whales")
        except Exception as e:
            if "BudgetExhausted" in type(e).__name__:
                return {}
            logger.warning(f"UW budget check failed (proceeding without enforcement): {e}")
            # Fall through to make the API call — fail-open for availability

    url = f"{_API_BASE}{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "UW-CLIENT-API-ID": "100001",
        "Accept": "application/json, text/plain",
    }

    await _pace_request()
    _req_count += 1

    quota_decision = None
    quota = None
    try:
        try:
            quota = _get_quota_coordinator()
            quota_decision = await asyncio.to_thread(
                quota.reserve,
                consumer="clio",
                endpoint=endpoint,
                purpose="clio_flow_intel",
                priority=_priority_for_endpoint(endpoint),
            )
            if not quota_decision.allowed:
                _log_quota_denied(endpoint, quota_decision)
                return {}
        except Exception as quota_exc:
            quota = None
            quota_decision = None
            logger.warning(
                "UW quota coordinator unavailable; CLIO 16K and HTTP guards remain active: %s",
                type(quota_exc).__name__,
            )
        client = await _get_client()
        request_started = _time.perf_counter()
        resp = await client.get(url, headers=headers, params=params or {})
        if quota is not None:
            await asyncio.to_thread(
                quota.observe_response,
                quota_decision,
                consumer="clio",
                endpoint=endpoint,
                purpose="clio_flow_intel",
                status_code=resp.status_code,
                headers=resp.headers,
                latency_ms=(_time.perf_counter() - request_started) * 1000.0,
            )

        # Track API usage from response headers
        dl = resp.headers.get("x-uw-token-req-limit")
        du = resp.headers.get("x-uw-daily-req-count")
        if dl:
            _usage["limit"] = int(dl)
        if du:
            _usage["used"] = int(du)
            if _usage["limit"]:
                _usage["remaining"] = _usage["limit"] - int(du)

        if _req_count % 20 == 0:
            logger.info(
                f"UW API: {_usage.get('used', '?')}/{_usage.get('limit', '?')} daily"
            )

        if resp.status_code not in (401, 429):
            _note_auth_ok()
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            # UW-2 (S211): short jittered backoff. The old wait came from a constant 60000 ms header
            # and outlived DPOF's 8 s per-call timeout, so the retry never ran. The pacer also
            # slows down briefly (AIMD) and the WARNING is deduplicated.
            wait = backoff_seconds_429(resp.headers)
            note_429_and_log(endpoint, wait)
            await asyncio.sleep(wait)
            await _pace_request()
            # Retry once (no recursive budget charge — already recorded above)
            try:
                retry_decision = None
                if quota is not None:
                    retry_decision = await asyncio.to_thread(
                        quota.reserve,
                        consumer="clio",
                        endpoint=endpoint,
                        purpose="clio_flow_intel_retry",
                        priority=_priority_for_endpoint(endpoint),
                    )
                    if not retry_decision.allowed:
                        logger.warning(
                            "UW_ACCOUNT_QUOTA_DENIED caller=clio endpoint=%s reason=%s retry=1",
                            endpoint,
                            retry_decision.reason_code,
                        )
                        return {}
                retry_started = _time.perf_counter()
                resp2 = await client.get(url, headers=headers, params=params or {})
                if quota is not None:
                    await asyncio.to_thread(
                        quota.observe_response,
                        retry_decision,
                        consumer="clio",
                        endpoint=endpoint,
                        purpose="clio_flow_intel_retry",
                        status_code=resp2.status_code,
                        headers=resp2.headers,
                        latency_ms=(_time.perf_counter() - retry_started) * 1000.0,
                    )
                if resp2.status_code == 429:
                    get_uw_pacer().note_429()
                if resp2.status_code == 401:
                    _note_auth_failure(endpoint)
                    return {}
                if resp2.status_code == 200:
                    _note_auth_ok()
                    return resp2.json()
            except Exception as e:
                if quota is not None:
                    await asyncio.to_thread(quota.release, locals().get("retry_decision"), "HTTP_RETRY_EXCEPTION")
                logger.debug(f"UW 429 retry failed: {e}")
            return {}
        elif resp.status_code == 401:
            _note_auth_failure(endpoint)
            return {}
        elif resp.status_code == 403:
            logger.warning(f"UW 403 on {endpoint}. May need higher API tier.")
            return {}
        else:
            logger.warning(f"UW HTTP {resp.status_code} on {endpoint}")
            return {}
    except Exception as e:
        if quota is not None:
            await asyncio.to_thread(quota.release, quota_decision, "HTTP_EXCEPTION")
        logger.debug(f"UW API error on {endpoint}: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  PILLAR 1: NOFA — Net Options Flow Activity
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_flow_alerts() -> list:
    """Fetch institutional flow alerts (3 API calls)."""
    alerts = []

    # High-premium OTM openers
    d = await _uw_get("/option-trades/flow-alerts", {
        "limit": 200, "min_premium": 50000,
        "size_greater_oi": True, "is_otm": True,
    })
    alerts.extend(d.get("data", []))

    # Calls $25K+
    d = await _uw_get("/option-trades/flow-alerts", {
        "limit": 100, "min_premium": 25000,
        "is_call": True, "size_greater_oi": True,
    })
    alerts.extend(d.get("data", []))

    # Puts $25K+
    d = await _uw_get("/option-trades/flow-alerts", {
        "limit": 100, "min_premium": 25000,
        "is_put": True, "size_greater_oi": True,
    })
    alerts.extend(d.get("data", []))

    return alerts


def _compute_nofa(alerts: list) -> dict:
    """NOFA = (Call Premium - Put Premium) / Total Premium per ticker. Range [-1, +1]."""
    td = defaultdict(lambda: {
        "call_prem": 0, "put_prem": 0, "total_prem": 0,
        "call_n": 0, "put_n": 0, "total_n": 0,
        "call_vol": 0, "put_vol": 0,
        "openers": 0, "otm_n": 0, "max_trade": 0, "trades": [],
    })
    seen = set()

    for a in alerts:
        tk = a.get("ticker_symbol") or a.get("ticker", "")
        if not tk:
            continue
        key = f"{tk}_{a.get('option_symbol', '')}_{a.get('total_premium', '')}"
        if key in seen:
            continue
        seen.add(key)

        prem = float(a.get("total_premium", 0) or 0)
        vol = int(float(a.get("total_size", 0) or 0))
        is_call = str(a.get("type", "")).upper() == "CALL" or a.get("is_call")
        is_put = str(a.get("type", "")).upper() == "PUT" or a.get("is_put")

        t = td[tk]
        t["total_prem"] += prem
        t["total_n"] += 1
        if prem > t["max_trade"]:
            t["max_trade"] = prem
        if is_call:
            t["call_prem"] += prem
            t["call_n"] += 1
            t["call_vol"] += vol
        elif is_put:
            t["put_prem"] += prem
            t["put_n"] += 1
            t["put_vol"] += vol
        if a.get("size_greater_oi"):
            t["openers"] += 1
        if a.get("is_otm"):
            t["otm_n"] += 1
        t["trades"].append({
            "type": "CALL" if is_call else "PUT",
            "premium": prem,
            "volume": vol,
            "strike": a.get("strike"),
            "expiry": a.get("expires"),
        })

    nofa = {}
    for tk, t in td.items():
        raw = (t["call_prem"] - t["put_prem"]) / max(t["total_prem"], 1)
        nofa[tk] = {
            "nofa": round(raw, 4),
            "dir": "BULLISH" if raw > 0.15 else ("BEARISH" if raw < -0.15 else "NEUTRAL"),
            "total_prem": t["total_prem"],
            "call_prem": t["call_prem"],
            "put_prem": t["put_prem"],
            "call_n": t["call_n"],
            "put_n": t["put_n"],
            "total_n": t["total_n"],
            "cp_ratio": round(t["call_n"] / max(t["put_n"], 1), 2),
            "openers_pct": round(t["openers"] / max(t["total_n"], 1), 2),
            "otm_pct": round(t["otm_n"] / max(t["total_n"], 1), 2),
            "max_trade": t["max_trade"],
        }
    return nofa


# ═══════════════════════════════════════════════════════════════════════════════
#  PILLAR 2: NET PREMIUM FLOW (Market Tide)
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_market_tide() -> dict:
    """Fetch market-wide premium flow (2 API calls)."""
    full = (await _uw_get("/market/market-tide", {"interval_5m": False})).get("data", [])
    intra = (await _uw_get("/market/market-tide", {"interval_5m": True})).get("data", [])
    return {"full_day": full, "intraday": intra}


async def _fetch_ticker_premiums(tickers: list) -> dict:
    """Fetch per-ticker net premium ticks for today (ET)."""
    date = datetime.now(_NY).strftime("%Y-%m-%d")  # S148 H-2: always pass date param
    out = {}
    for tk in tickers:
        d = (await _uw_get(f"/stock/{tk}/net-prem-ticks", {"date": date})).get("data", [])
        if d:
            out[tk] = d
    return out


def _premium_num(v) -> float:
    """float() that tolerates UW's STRING numerics; None/garbage/NaN/inf -> 0.0."""
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError):  # OverflowError: an int of 309+ digits
        return 0.0
    return x if math.isfinite(x) else 0.0


def _compute_premium_signals(tide: dict, tk_prems: dict) -> dict:
    """Compute premium flow signals from market tide + per-ticker data."""
    sig: dict = {"market": {}, "tickers": {}}

    full = tide.get("full_day", [])
    if full:
        latest = full[-1] if isinstance(full, list) else full
        nc = float(latest.get("net_call_premium", 0) or 0)
        np_ = float(latest.get("net_put_premium", 0) or 0)
        sig["market"] = {
            "net_call": nc,
            "net_put": np_,
            "net_flow": nc + np_,
            # NOTE: tide sign convention UNVERIFIED (2026-10-10 probe: |corr| <= 0.03 vs 5-min SPY/QQQ returns, inconclusive).
            "sentiment": "BULLISH" if nc > abs(np_) else "BEARISH",
            "ratio": round(nc / max(abs(np_), 1), 2),
        }

    intra = tide.get("intraday", [])
    if len(intra) >= 6:
        rc = sum(float(r.get("net_call_premium", 0) or 0) for r in intra[-6:])
        rp = sum(float(r.get("net_put_premium", 0) or 0) for r in intra[-6:])
        sig["market"]["last_30m"] = "BULLISH" if rc > abs(rp) else "BEARISH"
        sig["market"]["last_30m_net"] = round(rc + rp, 2)

    for tk, ticks in tk_prems.items():
        if not ticks:
            continue
        net = cp = pp = 0.0
        for t in ticks[-20:]:
            if not isinstance(t, dict):
                continue
            # Verified 2026-10-09 on live UW net-prem-ticks: net_call_premium > 0 = calls bought
            # (bullish); net_put_premium > 0 = puts bought (bearish). Net flow = nc - np.
            # Evidence: per-tick corr with net_delta 0.76 for (nc - np) vs 0.32 for (nc + np).
            # The legacy net_premium/call_premium/put_premium branch only serves old fixtures.
            if "net_call_premium" in t or "net_put_premium" in t:
                nc = _premium_num(t.get("net_call_premium"))
                npr = _premium_num(t.get("net_put_premium"))
                net += nc - npr
                cp += nc
                pp += npr
            else:
                net += _premium_num(t.get("net_premium"))
                cp += _premium_num(t.get("call_premium"))
                pp += _premium_num(t.get("put_premium"))
        sig["tickers"][tk] = {
            "net": net,
            "call": cp,
            "put": pp,
            "dir": "BULLISH" if net > 0 else "BEARISH",
            "score": min(max(net / max(abs(cp) + abs(pp), 1), -1), 1),
        }

    return sig


# ═══════════════════════════════════════════════════════════════════════════════
#  NET FLOW TIMELINE (on-demand, per-ticker)
# ═══════════════════════════════════════════════════════════════════════════════

async def get_ticker_flow_timeline(ticker: str, bucket_minutes: int = 5, date: str = "") -> list[dict]:
    """Fetch per-ticker net premium ticks and bucket into intervals.

    Uses /api/stock/{ticker}/net-prem-ticks endpoint.
    net_call / net_put are raw signed sums of net_call_premium / net_put_premium.
    net_put > 0 = puts bought (bearish); net_flow = net_call - net_put (bullish positive).
    Returns list of dicts sorted by time:
      [{"time": "09:30", "net_call": 1200000, "net_put": 800000,
        "net_flow": 400000, "call_vol": 500, "put_vol": 300}, ...]
    """
    if bucket_minutes < 1:
        bucket_minutes = 5
    # S146: Always pass a date to the API — empty date returns previous trading day's data
    if not date:
        date = datetime.now(_NY).strftime("%Y-%m-%d")
    params = {"date": date}

    data = await _uw_get(f"/stock/{ticker.upper()}/net-prem-ticks", params)
    ticks = data.get("data", [])
    if not ticks:
        return []

    # Parse ticks into buckets
    buckets: dict[str, dict] = {}
    for t in ticks:
        tape_time = t.get("tape_time", "")
        if not tape_time:
            continue
        try:
            # tape_time is ISO 8601. UW now returns UTC ("2026-10-09T13:31:00.000000Z"); older
            # rows carried an ET offset ("2026-03-20T09:31:00-04:00"). Bucket in ET either way.
            dt = datetime.fromisoformat(tape_time)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_NY)
            dt = dt.astimezone(_NY)
        except (ValueError, TypeError):
            continue

        # S146: Skip ticks from wrong day (defense-in-depth)
        if dt.strftime("%Y-%m-%d") != date:
            continue

        # Bucket to N-minute intervals
        minute = dt.minute - (dt.minute % bucket_minutes)
        bucket_key = dt.replace(minute=minute, second=0, microsecond=0).strftime("%H:%M")

        if bucket_key not in buckets:
            buckets[bucket_key] = {
                "time": bucket_key,
                "net_call": 0.0,
                "net_put": 0.0,
                "net_flow": 0.0,
                "call_vol": 0,
                "put_vol": 0,
                "call_vol_ask": 0,
                "put_vol_ask": 0,
                "ticks": 0,
            }

        b = buckets[bucket_key]
        # VERIFIED 2026-10-10: net_call_premium > 0 = calls net bought = BULLISH; net_put_premium > 0 = puts net bought = BEARISH.
        # Bullish net flow = nc - np. Evidence: 10 sessions, same-5-min SPY return corr +0.565 with (nc-np) vs +0.146 with (nc+np)
        # (10/10 days favour the difference; QQQ +0.416 vs +0.116); per-tick corr with net_delta 0.761 vs 0.316.
        nc = _premium_num(t.get("net_call_premium"))
        np_ = _premium_num(t.get("net_put_premium"))
        # If neither net_call nor net_put present, fall back to net_premium (positive = call side, negative = put side)
        if nc == 0 and np_ == 0:
            net_p = _premium_num(t.get("net_premium"))
            nc = max(net_p, 0.0)
            np_ = max(-net_p, 0.0)  # so that nc - np_ == net_p
        b["net_call"] += nc
        b["net_put"] += np_
        b["net_flow"] += nc - np_
        b["call_vol"] += int(t.get("call_volume", 0) or 0)
        b["put_vol"] += int(t.get("put_volume", 0) or 0)
        b["call_vol_ask"] += int(t.get("call_volume_ask_side", 0) or 0)
        b["put_vol_ask"] += int(t.get("put_volume_ask_side", 0) or 0)
        b["ticks"] += 1

    # Sort by time and return
    return sorted(buckets.values(), key=lambda x: x["time"])


# ═══════════════════════════════════════════════════════════════════════════════
#  PILLAR 3: DARK POOL
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_darkpool() -> tuple:
    """Fetch dark pool data (1 market + N ticker calls)."""
    mkt = (await _uw_get("/darkpool/recent")).get("data", [])
    tk_dp = {}
    for tk in _DARKPOOL_TICKERS:
        d = (await _uw_get(f"/darkpool/{tk}")).get("data", [])
        if d:
            tk_dp[tk] = d
    return mkt, tk_dp


def _compute_dp_signals(mkt_prints: list, tk_prints: dict) -> dict:
    """Compute dark pool accumulation/distribution signals."""
    sig: dict = {"market_top": {}, "tickers": {}}

    by_tk = defaultdict(lambda: {"notional": 0, "shares": 0, "n": 0, "prices": []})
    for p in mkt_prints:
        tk = p.get("ticker", "")
        px = float(p.get("price", 0) or 0)
        sz = int(float(p.get("size", 0) or 0))
        if tk and px > 0 and sz > 0:
            b = by_tk[tk]
            b["notional"] += px * sz
            b["shares"] += sz
            b["n"] += 1
            b["prices"].append(px)

    top = sorted(by_tk.items(), key=lambda x: x[1]["notional"], reverse=True)[:30]
    sig["market_top"] = {
        tk: {
            "notional": round(d["notional"], 2),
            "shares": d["shares"],
            "prints": d["n"],
            "avg_size": round(d["shares"] / max(d["n"], 1)),
            "vwap": round(d["notional"] / max(d["shares"], 1), 2),
        }
        for tk, d in top
    }

    for tk, prints in tk_prints.items():
        if not prints:
            continue
        notional = 0
        shares = 0
        prices = []
        for p in prints:
            px = float(p.get("price", 0) or 0)
            sz = int(float(p.get("size", 0) or 0))
            if px > 0 and sz > 0:
                notional += px * sz
                shares += sz
                prices.append(px)
        if shares == 0:
            continue
        vwap = notional / shares
        if prices:
            med = sorted(prices)[len(prices) // 2]
            bias = "ACCUMULATION" if vwap >= med else "DISTRIBUTION"
            bs = (vwap - med) / max(med, 0.01)
        else:
            bias = "NEUTRAL"
            bs = 0
        sig["tickers"][tk] = {
            "notional": round(notional, 2),
            "shares": shares,
            "prints": len(prints),
            "avg_size": round(shares / len(prints)),
            "vwap": round(vwap, 2),
            "bias": bias,
            "bias_score": round(min(max(bs, -1), 1), 4),
        }

    return sig


# ═══════════════════════════════════════════════════════════════════════════════
#  PILLAR 4: GAMMA EXPOSURE (GEX)
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_gex() -> dict:
    """Fetch gamma exposure data for key tickers."""
    out = {}
    for tk in _GEX_TICKERS:
        d = (await _uw_get(f"/stock/{tk}/spot-exposures/strike")).get("data", [])
        if d:
            out[tk] = d
    return out


def _compute_gex(gex_data: dict) -> dict:
    """Compute GEX regime (STABILIZING vs ACCELERATING) per ticker."""
    sig = {}
    for tk, strikes in gex_data.items():
        if not strikes:
            continue
        tc = 0
        tp = 0
        mx_strike = None
        mx_val = 0
        for s in strikes:
            st = float(s.get("strike", 0) or 0)
            cg = float(s.get("call_gamma_oi", 0) or s.get("call_gex", 0) or 0)
            pg = float(s.get("put_gamma_oi", 0) or s.get("put_gex", 0) or 0)
            tc += cg
            tp += pg
            net = cg + pg
            if abs(net) > abs(mx_val):
                mx_val = net
                mx_strike = st
        ng = tc + tp
        sig[tk] = {
            "net_gex": round(ng, 2),
            "call_gex": round(tc, 2),
            "put_gex": round(tp, 2),
            "regime": "STABILIZING" if ng > 0 else "ACCELERATING",
            "pin_strike": mx_strike,
            "pin_val": round(mx_val, 2),
        }
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
#  PILLAR 5: CONGRESSIONAL / INSIDER
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_congress_insider() -> tuple:
    """Fetch congressional + insider trading data (2 API calls)."""
    cong = (await _uw_get("/congress/recent-trades")).get("data", [])
    ins = (await _uw_get("/insider/transactions")).get("data", [])
    return cong, ins


def _compute_congress(cong: list, ins: list) -> dict:
    """Compute congressional/insider directional signal per ticker."""
    ta = defaultdict(lambda: {"cb": 0, "cs": 0, "ib": 0, "is_": 0})
    for t in cong:
        tk = t.get("ticker", "")
        tp = str(t.get("type", "")).lower()
        if not tk:
            continue
        a = ta[tk]
        if "purchase" in tp or "buy" in tp:
            a["cb"] += 1
        elif "sale" in tp or "sell" in tp:
            a["cs"] += 1

    for t in ins:
        tk = t.get("ticker", "")
        tp = str(t.get("transaction_type", "") or t.get("type", "")).lower()
        if not tk:
            continue
        a = ta[tk]
        if "purchase" in tp or "buy" in tp:
            a["ib"] += 1
        elif "sale" in tp or "sell" in tp:
            a["is_"] += 1

    sig = {}
    for tk, a in ta.items():
        tot = a["cb"] + a["cs"] + a["ib"] + a["is_"]
        if tot == 0:
            continue
        sc = (a["cb"] * 2 - a["cs"] * 2 + a["ib"] - a["is_"]) / max(tot, 1)
        sig[tk] = {
            "score": round(min(max(sc, -1), 1), 4),
            "dir": "BULLISH" if sc > 0.1 else ("BEARISH" if sc < -0.1 else "NEUTRAL"),
            "congress_buys": a["cb"],
            "congress_sells": a["cs"],
            "insider_buys": a["ib"],
            "insider_sells": a["is_"],
        }
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPOSITE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_composite(nofa: dict, prem: dict, dp: dict, gex: dict, cong: dict) -> list:
    """Compute weighted composite score per ticker across all 5 pillars."""
    all_tk = set()
    all_tk.update(nofa.keys())
    all_tk.update(prem.get("tickers", {}).keys())
    all_tk.update(dp.get("tickers", {}).keys())
    all_tk.update(gex.keys())
    all_tk.update(cong.keys())

    out = []
    for tk in all_tk:
        n = nofa.get(tk, {})
        nr = n.get("nofa", 0)
        p = prem.get("tickers", {}).get(tk, {})
        pr = p.get("score", 0)
        d = dp.get("tickers", {}).get(tk, {})
        dr = d.get("bias_score", 0)
        g = gex.get(tk, {})
        gr = 0.5 if g.get("regime") == "ACCELERATING" and nr > 0 else (
            -0.5 if g.get("regime") == "ACCELERATING" and nr < 0 else 0)
        c = cong.get(tk, {})
        cr = c.get("score", 0)

        comp = (
            _WEIGHTS["nofa"] * nr
            + _WEIGHTS["premium"] * pr
            + _WEIGHTS["darkpool"] * dr
            + _WEIGHTS["gex"] * gr
            + _WEIGHTS["congress"] * cr
        )
        conf = sum(1 for x in [n, p, d, g, c] if x)
        dirn = (
            "STRONG BUY" if comp > 0.4 else
            "BUY" if comp > 0.15 else
            "LEAN BULL" if comp > 0.05 else
            "NEUTRAL" if comp > -0.05 else
            "LEAN BEAR" if comp > -0.15 else
            "SELL" if comp > -0.4 else
            "STRONG SELL"
        )

        out.append({
            "ticker": tk,
            "score": round(comp, 4),
            "dir": dirn,
            "conf": conf,
            "nofa": round(nr, 4),
            "prem": round(pr, 4),
            "dp": round(dr, 4),
            "gex": round(gr, 4),
            "cong": round(cr, 4),
            "flow_prem": n.get("total_prem", 0),
            "dp_notional": d.get("notional", 0),
            "gex_regime": g.get("regime", "N/A"),
            "gex_pin": g.get("pin_strike"),
            "dp_bias": d.get("bias", "N/A"),
        })

    out.sort(key=lambda x: abs(x["score"]), reverse=True)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL REFRESH
# ═══════════════════════════════════════════════════════════════════════════════

async def _refresh_all() -> dict:
    """Run full 5-pillar scan. Returns signals dict for caching."""
    global _req_count
    logger.info("UW flow scan starting...")
    t0 = _time.time()
    _req_count = 0  # Reset per-scan counter

    # Pillar 1: NOFA
    alerts = await _fetch_flow_alerts()
    nofa = _compute_nofa(alerts)
    logger.info(f"UW P1 NOFA: {len(nofa)} tickers")

    # Pillar 2: Premium Flow
    tide = await _fetch_market_tide()
    tk_prems = await _fetch_ticker_premiums(_PREMIUM_TICKERS)
    prem_sig = _compute_premium_signals(tide, tk_prems)

    # Pillar 3: Dark Pool
    mkt_dp, tk_dp = await _fetch_darkpool()
    dp_sig = _compute_dp_signals(mkt_dp, tk_dp)
    logger.info(f"UW P3 DP: {len(dp_sig.get('tickers', {}))} tickers")

    # Pillar 4: GEX
    gex_data = await _fetch_gex()
    gex_sig = _compute_gex(gex_data)
    logger.info(f"UW P4 GEX: {len(gex_sig)} tickers")

    # Pillar 5: Congress/Insider
    cong_raw, ins_raw = await _fetch_congress_insider()
    cong_sig = _compute_congress(cong_raw, ins_raw)

    # Composite
    signals = _compute_composite(nofa, prem_sig, dp_sig, gex_sig, cong_sig)

    elapsed = _time.time() - t0
    logger.info(
        f"UW flow scan complete: {len(signals)} tickers, "
        f"{_req_count} API calls, {elapsed:.1f}s"
    )

    result = {
        "signals": signals,
        "market": prem_sig.get("market", {}),
        "gex": gex_sig,
        "dp_top": dp_sig.get("market_top", {}),
        "congress": cong_sig,
        "timestamp": datetime.now(_NY).isoformat(),
        "stale": False,
        "api_calls": _req_count,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

async def get_uw_signals() -> dict:
    """Get cached composite signals. Refreshes if cache expired.

    Returns dict with keys: signals, market, gex, dp_top, congress, timestamp, stale.
    Returns empty dict if no API key configured.
    """
    global _cache, _cache_time

    if not _get_api_key():
        return {}

    now = _time.time()
    ttl = _cache_ttl()

    with _cache_lock:
        if _cache and (now - _cache_time) < ttl:
            return _cache

    # Cache expired — refresh (S162: timeout protection against event loop freeze)
    try:
        result = await asyncio.wait_for(_refresh_all(), timeout=_SCAN_TIMEOUT)
        with _cache_lock:
            _cache = result
            _cache_time = _time.time()
        return result
    except asyncio.TimeoutError:
        logger.warning(f"UW scan timed out after {_SCAN_TIMEOUT}s — returning stale cache")
    except Exception as e:
        logger.debug(f"UW signal refresh failed: {e}")
    # Fallback: return stale cache or empty
    with _cache_lock:
        if _cache:
            _cache["stale"] = True
            return _cache
    return {}


async def get_market_tide() -> dict:
    """Get market-wide sentiment (shorter cache TTL than full signals).

    Returns dict with keys: net_call, net_put, net_flow, sentiment, ratio, last_30m.
    Returns empty dict if no API key configured.
    """
    global _tide_cache, _tide_cache_time

    if not _get_api_key():
        return {}

    now = _time.time()
    with _cache_lock:
        if _tide_cache and (now - _tide_cache_time) < _TIDE_TTL:
            return _tide_cache

    # Refresh tide only (2 API calls, much cheaper than full scan)
    try:
        tide = await _fetch_market_tide()
        prem_sig = _compute_premium_signals(tide, {})
        market = prem_sig.get("market", {})
        with _cache_lock:
            _tide_cache = market
            _tide_cache_time = _time.time()
        return market
    except Exception as e:
        logger.debug(f"UW tide refresh failed: {e}")
        with _cache_lock:
            return _tide_cache if _tide_cache else {}


def get_ticker_flow(ticker: str) -> dict | None:
    """Get flow signals for a specific ticker from cache (sync).

    Returns dict with keys: ticker, score, dir, nofa, prem, dp, gex, cong, conf,
    flow_prem, dp_notional, gex_regime, gex_pin, dp_bias.
    Returns None if not found or no cache.
    """
    with _cache_lock:
        if not _cache or not _cache.get("signals"):
            return None
        signals = list(_cache.get("signals", []))
    for s in signals:
        if s.get("ticker") == ticker:
            return s
    return None


def get_cached_tide() -> dict:
    """Get cached market tide (sync). Returns empty dict if no cache."""
    with _cache_lock:
        return dict(_tide_cache) if _tide_cache else {}


def get_flow_summary(max_chars: int = 800) -> str:
    """Get formatted text summary for system prompt injection (sync).

    Returns compact string with market tide + top movers + GEX.
    Returns "" if no data.
    """
    with _cache_lock:
        if not _cache:
            return ""
        data = _cache.copy()

    market = data.get("market", {})
    signals = data.get("signals", [])
    gex = data.get("gex", {})

    if not market and not signals:
        return ""

    parts = []

    # Market tide
    if market:
        sentiment = market.get("sentiment", "N/A")
        net_flow = market.get("net_flow", 0)
        ratio = market.get("ratio", "N/A")
        last_30m = market.get("last_30m", "N/A")
        parts.append(
            f"Market Tide: {sentiment} | Net Flow: ${net_flow:,.0f} | "
            f"C/P: {ratio} | 30m: {last_30m}"
        )

    # GEX regime for SPY/QQQ
    for idx_tk in ("SPY", "QQQ"):
        g = gex.get(idx_tk)
        if g and g.get("regime"):
            pin = g.get("pin_strike")
            if pin:
                try:
                    parts.append(f"{idx_tk} GEX: {g['regime']} (pin: ${float(pin):.0f})")
                except (ValueError, TypeError):
                    parts.append(f"{idx_tk} GEX: {g['regime']}")
            else:
                parts.append(f"{idx_tk} GEX: {g['regime']}")

    # Top bullish/bearish
    bulls = [s for s in signals if s["score"] > 0.1][:3]
    bears = [s for s in signals if s["score"] < -0.1][:3]

    if bulls:
        bull_str = ", ".join(f"{s['ticker']}({s['score']:+.2f})" for s in bulls)
        parts.append(f"Top Bullish Flow: {bull_str}")
    if bears:
        bear_str = ", ".join(f"{s['ticker']}({s['score']:+.2f})" for s in bears)
        parts.append(f"Top Bearish Flow: {bear_str}")

    result = "\n".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars - 3] + "..."
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  HYDRINIA BRIDGE WRITE (F6)
# ═══════════════════════════════════════════════════════════════════════════════

def _write_bridge_signals(signals: dict) -> None:
    """Write UW signals to Hydrinia bridge for agent consumption."""
    try:
        from agent.config import config

        if not signals or not signals.get("signals"):
            return

        bridge_dir = Path(config.tradingbot_dir) / "moltbot_knowledge" / "intelligence"
        bridge_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now(_NY).strftime("%Y%m%d")
        bridge_data = {
            "id": f"uw_flow_{today}",
            "type": "institutional_flow",
            "source": "unusual_whales",
            "created": datetime.now(_NY).isoformat(),
            "market_tide": signals.get("market", {}),
            "top_signals": signals.get("signals", [])[:20],
            "gex": signals.get("gex", {}),
            "dark_pool_top": signals.get("dp_top", {}),
            "congress": signals.get("congress", {}),
            "pillars_available": max(
                (s.get("conf", 0) for s in signals.get("signals", [{}])),
                default=0,
            ),
        }

        fp = bridge_dir / f"uw_flow_{today}.json"
        tmp = fp.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(bridge_data, f, indent=2, default=str)
        os.replace(tmp, fp)
        logger.info(f"UW bridge write: {fp.name} ({len(bridge_data['top_signals'])} signals)")
    except Exception as e:
        logger.debug(f"UW bridge write failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SCHEDULED TASK ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def run_uw_scan(context) -> None:
    """Scheduled task callback. Refreshes all UW signal pillars and updates cache.

    Registered in scheduled_tasks.py as repeating every 30 min.
    Skips if no API key configured.
    """
    if not _get_api_key():
        return
    # Skip outside extended market hours to conserve UW API budget
    _now_uw = datetime.now(_NY)
    if _now_uw.weekday() >= 5 or _now_uw.hour < 6 or _now_uw.hour >= 20:
        return

    try:
        signals = await get_uw_signals()
        if signals:
            _write_bridge_signals(signals)

            # Log summary
            market = signals.get("market", {})
            n_signals = len(signals.get("signals", []))
            sentiment = market.get("sentiment", "N/A")
            logger.info(
                f"UW scan complete: {sentiment} tide, {n_signals} ticker signals, "
                f"API usage: {_usage.get('used', '?')}/{_usage.get('limit', '?')}"
            )
    except Exception as e:
        logger.debug(f"UW scheduled scan failed: {e}")
