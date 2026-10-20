"""
CLIO-native market regime classifier.

Hydrinia (/home/agent/tradingbot) is permanently off (operator direction,
2026-09-24). Its `state/daily_metrics.json["market_regime"]` field is frozen
at whatever value the process last wrote before it stopped, and every CLIO
consumer that read it was silently anchoring on the same stale regime read
forever. This module REPLACES that field for all of CLIO with a small,
self-contained, CLIO-native classifier.

Data sources (CLIO-native only, no Hydrinia files):
  - SPY daily bars (~60 sessions) via agent/integrations/alpaca.py's Alpaca
    market-data client (feed = config.alpaca_stock_feed).
  - VIX via agent/integrations/polygon.py:get_snapshot_raw("VIX") -- the same
    source agent/dpof/data.py and social_intel_divergence.py already use for
    VIX (Polygon has no VIX ticker; that function Yahoo-falls-back for index
    symbols). If VIX cannot be read, the classifier still runs on the SPY
    trend signal alone, at reduced confidence -- it never blocks entirely on
    a missing VIX read.

Classification mirrors Hydrinia's documented heuristic ("VIX>25->CHOPPY,
VIX>30->BEAR", CLAUDE.md Regime Detection row) plus an SPY 20/50-day SMA
trend read. See `classify_regime()` for the exact, deterministic rule and
the confidence formula.

Output file: <repo>/data/intelligence/market_regime.json (path overridable
via env CLIO_MARKET_REGIME_FILE, mainly for tests), written atomically
(tmp file + os.fsync + os.replace, off the event loop).

Public contract (also documented for other CLIO builders in NATIVE_CONTEXT):
  get_market_regime(max_age_seconds=6h) -> dict | None      sync, never raises
  get_market_regime_label(default="CHOPPY", max_age_seconds=6h) -> str
  async refresh_market_regime(context=None) -> dict | None   PTB job callback

On any refresh failure (bars unavailable, write failure, etc.) the previous
file is left untouched and refresh_market_regime() logs a warning and
returns None -- it never raises, so a scheduled-job failure never crashes
the scheduler.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("openclaw.market_regime")

# ---------------------------------------------------------------------------
# Paths / knobs
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_FILE = os.path.join(_REPO_ROOT, "data", "intelligence", "market_regime.json")

_DEFAULT_MAX_AGE_SECONDS = 6 * 3600     # matches CLAUDE.md's other "fresh" thresholds
_SESSIONS_REQUESTED = 60                # SPY daily bars to request
_MAX_BAR_AGE_DAYS = 4                   # newest SPY bar older than this (ET calendar days) = stale (R-15)
_NY = ZoneInfo("America/New_York")
_MIN_SESSIONS_FOR_SMA50 = 50            # minimum bars needed to compute SMA50
_VIX_CALL_TIMEOUT_SECONDS = 8.0         # mirrors agent/dpof/data.py's _VIX_CALL_TIMEOUT_SECONDS

_REGIMES = ("BULL", "BEAR", "CHOPPY", "CRISIS")


def _file_path() -> str:
    """Runtime-evaluated so tests can monkeypatch the env var per-test."""
    return os.environ.get("CLIO_MARKET_REGIME_FILE", _DEFAULT_FILE)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _bucket_for_vix(vix: Optional[float]) -> Optional[str]:
    """VIX>30 -> risk_off, VIX>25 -> choppy, else bull. None if VIX unavailable."""
    if vix is None:
        return None
    if vix > 30:
        return "risk_off"
    if vix > 25:
        return "choppy"
    return "bull"


def _bucket_for_sma(sma20: float, sma50: float) -> str:
    if sma20 > sma50:
        return "bull"
    if sma20 < sma50:
        return "risk_off"
    return "choppy"


def _bucket_for_price(close: float, sma50: float) -> str:
    if close > sma50:
        return "bull"
    if close < sma50:
        return "risk_off"
    return "choppy"


def _final_bucket(regime: str) -> str:
    if regime == "BULL":
        return "bull"
    if regime in ("BEAR", "CRISIS"):
        return "risk_off"
    return "choppy"


def classify_regime(
    spy_close: float, spy_sma20: float, spy_sma50: float, vix: Optional[float]
) -> tuple[str, float]:
    """Deterministic BULL/BEAR/CHOPPY/CRISIS classification + confidence in [0, 1].

    Rule (mirrors Hydrinia's documented VIX heuristic + an SPY 20/50-SMA trend read):
      CRISIS  VIX >= 35
      BEAR    VIX > 30, OR (close < SMA50 AND SMA20 < SMA50)
      CHOPPY  VIX > 25, OR the trend is mixed (neither clearly bull nor clearly bear)
      BULL    close > SMA50 AND SMA20 > SMA50 AND (VIX is None OR VIX <= 25)
    A missing VIX never blocks a classification -- the "VIX > N" legs simply don't
    fire, and BULL's "VIX <= 25" leg treats a missing reading as satisfied (so a
    confirmed price uptrend with no VIX read still classifies BULL, just at the
    confidence penalty below).

    Confidence: three independent "risk bucket" reads (VIX, SMA20-vs-SMA50, and
    close-vs-SMA50, each bucketed into bull/choppy/risk_off) are compared against
    the FINAL classification's bucket (BULL->bull, {BEAR,CRISIS}->risk_off,
    CHOPPY->choppy). confidence = (# of the 3 that agree) / 3 -- NOT divided by
    only the available signals, so a missing VIX read can never contribute an
    "agree" and mechanically caps confidence at 2/3 even when both trend reads
    agree with the classification. This is the "lower confidence when VIX is
    unavailable" behavior called for in the module contract.
    """
    vix_ge_35 = vix is not None and vix >= 35
    vix_gt_30 = vix is not None and vix > 30
    vix_gt_25 = vix is not None and vix > 25
    vix_le_25 = vix is None or vix <= 25

    bull_trend = spy_close > spy_sma50 and spy_sma20 > spy_sma50
    bear_trend = spy_close < spy_sma50 and spy_sma20 < spy_sma50
    mixed_trend = not bull_trend and not bear_trend

    if vix_ge_35:
        regime = "CRISIS"
    elif vix_gt_30 or bear_trend:
        regime = "BEAR"
    elif vix_gt_25 or mixed_trend:
        regime = "CHOPPY"
    elif bull_trend and vix_le_25:
        regime = "BULL"
    else:
        regime = "CHOPPY"  # unreachable given the partition above; safe deterministic default

    final = _final_bucket(regime)
    buckets = (
        _bucket_for_vix(vix),
        _bucket_for_sma(spy_sma20, spy_sma50),
        _bucket_for_price(spy_close, spy_sma50),
    )
    agreeing = sum(1 for b in buckets if b == final)
    confidence = round(agreeing / 3.0, 4)
    return regime, confidence


# ---------------------------------------------------------------------------
# Data fetch (blocking SDK calls -- callers MUST run these in an executor)
# ---------------------------------------------------------------------------


def _resolve_stock_feed():
    """Map config.alpaca_stock_feed -> alpaca.data.enums.DataFeed (defaults to IEX)."""
    from alpaca.data.enums import DataFeed
    from agent.config import config

    mapping = {
        "iex": DataFeed.IEX,
        "sip": DataFeed.SIP,
        "delayed_sip": DataFeed.DELAYED_SIP,
        "otc": DataFeed.OTC,
    }
    raw = str(getattr(config, "alpaca_stock_feed", "iex") or "iex").lower()
    return mapping.get(raw, DataFeed.IEX)


def _now_utc() -> datetime:
    """Indirection so tests can pin 'now' (weekend / stale-bar cases)."""
    return datetime.now(timezone.utc)


def _bar_session_date(ts) -> Optional[date]:
    """ET session date of a bar timestamp (naive timestamps are treated as UTC)."""
    if not isinstance(ts, datetime):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_NY).date()


def _bars_stale(bars_day: date, now: datetime) -> bool:
    """True when the newest bar's ET session date is more than _MAX_BAR_AGE_DAYS calendar
    days before today (ET). Long weekends stay inside the window (Fri bar, Mon morning = 3)."""
    today_et = now.astimezone(_NY).date()
    return (today_et - bars_day).days > _MAX_BAR_AGE_DAYS


def _fetch_spy_daily_bars(symbol: str = "SPY", sessions: int = _SESSIONS_REQUESTED) -> list[tuple[date, float]]:
    """BLOCKING. Fetch the NEWEST ~`sessions` daily bars for `symbol` via CLIO's Alpaca data
    client as (session_date, close) pairs, oldest first. Returns [] on any failure (never
    raises) so the executor-wrapped caller can treat "no data" uniformly.

    The request asks for sort=DESC so the provider's `limit` truncates the OLD tail of the
    window (an ascending request with a limit returns the oldest bars, leaving the regime
    ~16 sessions stale -- R-15). The result is re-sorted ascending locally."""
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.common.enums import Sort
        from agent.integrations.alpaca import _get_data_client, TIMEFRAME_MAP

        client = _get_data_client()
        # Padding for weekends/holidays: ~1.6 calendar days per trading session, plus slack.
        lookback_days = int(sessions * 1.6) + 15
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TIMEFRAME_MAP["1d"],
            start=_now_utc() - timedelta(days=lookback_days),
            limit=sessions,
            sort=Sort.DESC,
            feed=_resolve_stock_feed(),
        )
        barset = client.get_stock_bars(request)
        bars = barset[symbol]
    except Exception as e:
        logger.warning(f"market_regime: SPY bars fetch failed ({type(e).__name__}: {e})")
        return []
    dated: list[tuple[date, float]] = []
    for b in bars or []:
        c = getattr(b, "close", None)
        d = _bar_session_date(getattr(b, "timestamp", None))
        if c is None or d is None:
            continue
        try:
            dated.append((d, float(c)))
        except (TypeError, ValueError):
            continue
    dated.sort(key=lambda t: t[0])
    return dated


async def _fetch_vix() -> Optional[float]:
    """Reuses agent.integrations.polygon.get_snapshot_raw("VIX") -- the same CLIO-native VIX
    source agent/dpof/data.py already relies on. Returns None on any failure/timeout."""
    try:
        from agent.integrations.polygon import get_snapshot_raw

        res = await asyncio.wait_for(get_snapshot_raw("VIX"), timeout=_VIX_CALL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.debug(f"market_regime: VIX fetch timed out after {_VIX_CALL_TIMEOUT_SECONDS}s")
        return None
    except Exception as e:
        logger.debug(f"market_regime: VIX fetch failed ({type(e).__name__}: {e})")
        return None
    if not isinstance(res, dict) or not res:
        return None
    price = res.get("last_price")
    if price is None:
        return None
    try:
        return float(price)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _write_atomic(payload: dict) -> bool:
    """BLOCKING. tmp + fsync + os.replace. Returns False (and leaves the previous
    file untouched) on any failure -- never raises."""
    path = _file_path()
    d = os.path.dirname(path)
    tmp_path = None
    try:
        os.makedirs(d, exist_ok=True)
        data_str = json.dumps(payload, indent=2, default=str)  # serialize BEFORE mkstemp
        fd, tmp_path = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data_str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return True
    except OSError as e:
        logger.warning(f"market_regime: atomic write failed, keeping previous file ({type(e).__name__}: {e})")
        return False


# ---------------------------------------------------------------------------
# Refresh (PTB JobQueue-compatible)
# ---------------------------------------------------------------------------


async def refresh_market_regime(context=None) -> Optional[dict]:
    """Scheduled job entry point. Computes a fresh regime read from CLIO-native
    data and atomically writes it to `_file_path()`. On ANY failure (insufficient
    bars, fetch error, write error) the previous file is left as-is, a warning is
    logged, and None is returned -- this function never raises."""
    try:
        loop = asyncio.get_running_loop()
        bars = await loop.run_in_executor(None, _fetch_spy_daily_bars)
        if len(bars) < _MIN_SESSIONS_FOR_SMA50:
            logger.warning(
                f"market_regime: insufficient SPY closes ({len(bars)} < {_MIN_SESSIONS_FOR_SMA50}), "
                "keeping previous file"
            )
            return None

        bars_day, _ = bars[-1]
        now = _now_utc()
        if _bars_stale(bars_day, now):
            logger.warning(
                f"market_regime: last bar {bars_day.isoformat()} is stale; keeping previous regime"
            )
            return None
        closes = [c for _, c in bars]

        spy_close = closes[-1]
        spy_sma20 = sum(closes[-20:]) / 20.0
        spy_sma50 = sum(closes[-50:]) / 50.0

        vix = await _fetch_vix()

        regime, confidence = classify_regime(spy_close, spy_sma20, spy_sma50, vix)
        payload = {
            "regime": regime,
            "confidence": confidence,
            "vix": vix,
            "spy_close": round(spy_close, 4),
            "spy_sma20": round(spy_sma20, 4),
            "spy_sma50": round(spy_sma50, 4),
            "as_of": now.isoformat(),
            "bars_as_of": bars_day.isoformat(),
            "source": "clio_native",
        }

        ok = await loop.run_in_executor(None, _write_atomic, payload)
        if not ok:
            return None
        logger.info(f"market_regime: refreshed -> {regime} (confidence={confidence}, vix={vix})")
        return payload
    except Exception as e:  # noqa: BLE001 — scheduled job: must never raise into the scheduler
        logger.warning(f"market_regime: refresh failed, keeping previous file ({type(e).__name__}: {e})")
        return None


# ---------------------------------------------------------------------------
# Read (sync, cached, never raises)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: dict[str, tuple[tuple[int, int, int], Any]] = {}


def _read_cached(path: str) -> Optional[dict]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    # Inode too: writers replace the file atomically (new inode per write), and two writes of the
    # same size can share an mtime tick -- (mtime, size) alone then serves the stale content.
    key = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _cache_lock:
        hit = _cache.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.debug(f"market_regime: unreadable {path} ({type(e).__name__}: {e})")
        return None
    if not isinstance(data, dict):
        return None
    with _cache_lock:
        _cache[path] = (key, data)
    return data


def get_market_regime(max_age_seconds: int = _DEFAULT_MAX_AGE_SECONDS) -> Optional[dict]:
    """Sync, mtime-cached, NEVER raises. None when the file is missing, malformed,
    or older than `max_age_seconds`."""
    try:
        data = _read_cached(_file_path())
        if not data:
            return None
        regime = data.get("regime")
        if regime not in _REGIMES:
            return None
        as_of = data.get("as_of")
        if not isinstance(as_of, str):
            return None
        try:
            dt = datetime.fromisoformat(as_of)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = _now_utc()
        age = (now - dt).total_seconds()
        if age < 0 or age > max_age_seconds:
            return None
        # Payloads written before R-15 carry no bars_as_of: they keep the as_of-only check.
        bars_as_of = data.get("bars_as_of")
        if bars_as_of is not None:
            if not isinstance(bars_as_of, str):
                return None
            try:
                bars_day = date.fromisoformat(bars_as_of)
            except ValueError:
                return None
            if _bars_stale(bars_day, now):
                return None
        return data
    except Exception as e:  # noqa: BLE001 — public read API: must never raise
        logger.debug(f"market_regime: get_market_regime failed ({type(e).__name__}: {e})")
        return None


def get_market_regime_label(default: str = "CHOPPY", max_age_seconds: int = _DEFAULT_MAX_AGE_SECONDS) -> str:
    """Sync, NEVER raises. `default` when the regime is missing/stale/malformed."""
    try:
        data = get_market_regime(max_age_seconds=max_age_seconds)
        if data:
            label = data.get("regime")
            if label in _REGIMES:
                return label
        return default
    except Exception as e:  # noqa: BLE001 — public read API: must never raise
        logger.debug(f"market_regime: get_market_regime_label failed ({type(e).__name__}: {e})")
        return default
