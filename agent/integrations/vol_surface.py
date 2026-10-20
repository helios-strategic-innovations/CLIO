"""
CLIO Vol Surface Intelligence ("V", INTEL_V3_SPEC.md) — SPY/QQQ term
structure, 25-delta skew, and IV-vs-realized-vol richness.

Adapts the *signal math* of Hydrinia's (/home/user/hydrinia, operator-owned —
credited here) "Atlas Oracle" module, vol_surface_intelligence.py: term
structure slope/state, put-call skew, and the IV-RV spread. The code below is
independently written for CLIO's own UW HTTP primitive and CLIO's
advisory-only, stateless-per-refresh contract (no Hydrinia code imported,
no IVHistoryStore/z-score rolling-window state carried over), but the S177
fix is carried over deliberately, since it documents a real UW data quirk:

  S177 (Hydrinia uw_market_analyzer.get_raw_realized_vol, credited): UW's
  /volatility/realized is forward-shifted — a row's `date` is the IV
  observation date and its realized_volatility covers the window ENDING at
  `unshifted_rv_date`, so the newest rows stay null until their window
  closes. Fix: walk back to the latest row with a value (= the most recently
  completed trailing window), accept it only if that window ended recently,
  else fall back to /volatility/stats's "rv". See _extract_realized_vol().

Operator directive (INTEL_V3_SPEC.md): this module is ADVISORY ONLY. It never
imports agent.dpof.broker/orders/engine and never places, sizes, cancels or
modifies an order — pure UW-data math, no LLM calls, no DDG research.

Endpoints (verified 2026-09-25 against docs/api/uw.txt's operation-name list
AND the UW OpenAPI spec's `paths`/`components.schemas`,
scratchpad/uw_openapi.json):

  GET /stock/{ticker}/volatility/term-structure
      -> {"data": [{date, dte, expiry, implied_move, implied_move_perc,
          volatility}, ...]}   (LIST; "volatility" is the IV field — matches
          Hydrinia's own S150-verified field name, NOT "iv"/"implied_volatility")

  GET /stock/{ticker}/volatility/stats
      -> {"data": {date, iv, iv_high, iv_low, iv_rank, rv, rv_high, rv_low,
          ticker}}             (SINGLE OBJECT, not a list — verified from the
          OpenAPI schema's own example; every other endpoint here returns a list)

  GET /stock/{ticker}/volatility/realized?days=21&timeframe=1M
      -> {"data": [{date, implied_volatility, implied_volatility_days, price,
          realized_volatility, realized_volatility_days, unshifted_rv_date}, ...]}
          (LIST; `days` is a UW enum {1,3,5,10,21,42,63,126,251} — no literal
          "20", so 21 is the closest bucket to this module's nominal "rv_20d"
          window; see the constant below)

  GET /stock/{ticker}/historical-risk-reversal-skew?delta=25&expiry=<YYYY-MM-DD>
      -> {"data": [{date, delta, risk_reversal, ticker}, ...]}   (LIST;
          `risk_reversal` is documented by UW itself as "the difference
          between the iv of a put and a call with similar absolute deltas"
          — i.e. put_iv - call_iv, exactly this module's skew_25d definition.
          `expiry` is a REQUIRED param on this endpoint — mirrors Hydrinia's
          own "expiry is REQUIRED — skip rather than get 422" comment; see
          _pick_skew_expiry())

Up to 4 UW calls per symbol per refresh (skew is skipped, dropping to 3,
when no usable expiry can be derived from term-structure) — with the default
2-symbol DPOF universe that is ≤8 calls/refresh. All calls go through
agent.integrations.uw_flow_intel._uw_get, which already owns auth, the
"unusual_whales" rate_limiter budget, and quota accounting — no new budget
is registered here.

Units: iv_front / iv_30d / rv_20d / term_slope / iv_rv_spread / skew_25d are
ALL reported in "vol points" (percentage-point scale, e.g. 23.19 for a 23.19%
IV reading). UW returns these as raw decimal fractions ("0.2319"); this
module multiplies by 100 uniformly so every absolute vol-value field and
every field derived from a simple subtraction of two such fields (term_slope,
iv_rv_spread, skew_25d) shares one consistent scale. iv_rank is reported
0-100 (UW's raw iv_rank/iv_rank_1y is a 0-1 fraction, same *100 treatment).
Any value CLIO could not compute is None, never 0 — CLAUDE.md "missing is
never zero".

State file: <repo>/data/intelligence/vol_surface.json (env
CLIO_VOL_SURFACE_FILE overrides it, mainly for tests). Written atomically
(tmp file + fsync + os.replace) off the event loop. Per-symbol entries are
merged onto whatever was already on disk: a symbol that got zero successful
UW calls this cycle keeps its previous entry untouched rather than being
overwritten with an all-None one (mirrors econ_calendar.py's "at least one
live source worked this cycle" merge discipline).

Public contract (INTEL_V3_SPEC.md):
  async def refresh_vol_surface(context=None) -> dict | None        PTB job callback
  def get_vol_surface(symbol, max_age_seconds=3600) -> dict | None  sync, mtime-cached, never raises

Per-symbol dict shape (also documented for G, the DPOF-integration builder):
  {symbol, iv_front, iv_30d, term_slope, term_state ("contango"|"flat"|
   "backwardation"), rv_20d, iv_rv_spread, iv_rank, skew_25d, richness
   ("rich"|"fair"|"cheap"|None), as_of, sources, quality_notes}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
from datetime import date, datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("openclaw.vol_surface")
_NY = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Paths / knobs
# ---------------------------------------------------------------------------

# agent/integrations/vol_surface.py -> 3x dirname = repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_FILE = os.path.join(_REPO_ROOT, "data", "intelligence", "vol_surface.json")

_DEFAULT_MAX_AGE_SECONDS = 3600
_DEFAULT_SYMBOLS = ("SPY", "QQQ")

# /volatility/realized `days` enum is {1,3,5,10,21,42,63,126,251} — no
# literal 20; 21 is the closest bucket to this module's nominal "rv_20d".
_REALIZED_VOL_DAYS = 21
# Must reach well past the 21-trading-day horizon: every row whose window has
# not closed yet is null, so a 1M response is (almost) all nulls.
_REALIZED_VOL_TIMEFRAME = "3M"
# A realized window that ended more than this many calendar days ago is too old
# to present as "current" realized vol (covers weekends + a holiday).
_RV_MAX_WINDOW_END_AGE_DAYS = 5

# 25-delta risk reversal; historical-risk-reversal-skew requires a concrete expiry.
_SKEW_DELTA = 25
_SKEW_MIN_DTE = 7          # skip near-0DTE expiries when picking the skew expiry (mirrors Hydrinia)
_TARGET_30D_DTE = 30

# term_state neutral band, in vol points. Mirrors the *spirit* of Hydrinia's
# ratio-based INVERSION_BUFFER (slope ratio > 1.05, i.e. ~5% relative) but
# expressed as an absolute vol-point band since this contract defines
# term_slope as a DIFFERENCE (front minus 30d), not a ratio. At typical
# SPY/QQQ IV levels (~12-20 vol points) a 5% relative move is roughly
# 0.6-1.0 vol points, so +-0.5 is a comparable, slightly tighter band.
_TERM_STATE_NEUTRAL_BAND = 0.5

# Richness thresholds. Documented design choice — NOT z-scored against a
# persisted rolling window like Hydrinia's IvRvSpreadSignal (which needs 30
# days of IV history this contract doesn't call for); a simple, stateless
# spread+rank read instead. The asymmetric spread thresholds mirror
# Hydrinia's own asymmetry (EXPENSIVE at z>1.5, the tighter CHEAP at z<-1.0).
_RICH_SPREAD_VOL_POINTS = 3.0
_CHEAP_SPREAD_VOL_POINTS = -2.0
_RICH_IV_RANK_MIN = 50.0
_CHEAP_IV_RANK_MAX = 50.0

# Defensive refresh window (weekdays, extended hours). The orchestrator owns
# the actual cadence (~30min during market hours + one premarket run,
# scheduled_tasks.py); this guard is defense-in-depth only, mirroring
# uw_flow_intel.run_uw_scan's own weekday/hour window.
_REFRESH_START_HOUR_ET = 4
_REFRESH_END_HOUR_ET = 20


def _file_path() -> str:
    """Runtime-evaluated so tests can monkeypatch the env var per-test."""
    return os.environ.get("CLIO_VOL_SURFACE_FILE", _DEFAULT_FILE)


def _now_et() -> datetime:
    return datetime.now(_NY)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> Optional[float]:
    """Parse a UW numeric field (frequently a string) -> float, or None.
    Never returns 0 for missing/garbage input — CLAUDE.md "missing is never
    zero"."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf guard
        return None
    return f


def _to_vol_points(fraction: Optional[float]) -> Optional[float]:
    """UW volatility fields are raw decimal fractions (0.2319 = 23.19%).
    Convert to vol points (percentage-point scale) — see module docstring."""
    if fraction is None:
        return None
    return round(fraction * 100.0, 4)


def _to_iv_rank_scale(fraction: Optional[float]) -> Optional[float]:
    """UW iv_rank / iv_rank_1y are 0-1 fractions; this contract wants 0-100."""
    if fraction is None:
        return None
    return round(max(0.0, min(100.0, fraction * 100.0)), 2)


# ---------------------------------------------------------------------------
# Signal 1: term structure
# ---------------------------------------------------------------------------


def _compute_term_structure(entries: Optional[list]) -> dict:
    """entries: the raw `data` list from /volatility/term-structure (maybe
    None/[]/malformed). Returns {iv_front, iv_30d, term_slope, term_state},
    all vol points / None on insufficient data."""
    parsed = []
    for row in entries or []:
        if not isinstance(row, dict):
            continue
        iv = _safe_float(row.get("volatility"))
        if iv is None or iv <= 0:
            continue
        dte_raw = row.get("dte")
        try:
            dte = int(dte_raw)
        except (TypeError, ValueError):
            continue
        if dte < 0:
            continue
        parsed.append({"iv": iv, "dte": dte})

    if not parsed:
        return {"iv_front": None, "iv_30d": None, "term_slope": None, "term_state": None}

    parsed.sort(key=lambda e: e["dte"])
    front = parsed[0]
    nearest_30 = min(parsed, key=lambda e: abs(e["dte"] - _TARGET_30D_DTE))

    iv_front = _to_vol_points(front["iv"])
    iv_30d = _to_vol_points(nearest_30["iv"])
    term_slope = round(iv_front - iv_30d, 4)

    if term_slope > _TERM_STATE_NEUTRAL_BAND:
        term_state = "backwardation"   # near-dated IV richer than 30d -> inverted curve
    elif term_slope < -_TERM_STATE_NEUTRAL_BAND:
        term_state = "contango"        # normal upward-sloping curve
    else:
        term_state = "flat"

    return {"iv_front": iv_front, "iv_30d": iv_30d, "term_slope": term_slope, "term_state": term_state}


def _pick_skew_expiry(term_structure_entries: Optional[list]) -> Optional[str]:
    """Choose an expiry (YYYY-MM-DD) for the historical-risk-reversal-skew
    call: the one closest to ~30 DTE among entries with dte >= _SKEW_MIN_DTE
    (mirrors Hydrinia's _find_best_expiry — avoids handing UW a near-0DTE
    expiry for what's meant to be a ~monthly skew read).

    Returns None if no usable expiry exists (term-structure fetch failed, or
    every entry is too near). Callers MUST skip the skew call in that case:
    `expiry` is a REQUIRED UW param and an empty/omitted value 422s."""
    candidates = []
    for row in term_structure_entries or []:
        if not isinstance(row, dict):
            continue
        expiry = row.get("expiry")
        try:
            dte = int(row.get("dte"))
        except (TypeError, ValueError):
            continue
        if not expiry or dte < _SKEW_MIN_DTE:
            continue
        candidates.append((abs(dte - _TARGET_30D_DTE), str(expiry)[:10]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Signal 2: skew
# ---------------------------------------------------------------------------


def _compute_skew(rows: Optional[list]) -> Optional[float]:
    """rows: the raw `data` list from /historical-risk-reversal-skew.
    Returns the most recent usable `risk_reversal` reading, in vol points, or
    None. UW documents risk_reversal as "the difference between the iv of a
    put and a call with similar absolute deltas" — i.e. put_iv - call_iv,
    exactly this module's skew_25d definition, no further arithmetic needed."""
    for row in reversed(rows or []):
        if not isinstance(row, dict):
            continue
        val = _safe_float(row.get("risk_reversal"))
        if val is not None:
            return _to_vol_points(val)
    return None


# ---------------------------------------------------------------------------
# Signal 3: realized vol / IV-RV spread (S177 fix)
# ---------------------------------------------------------------------------


def _extract_stats(stats_payload: Any) -> dict:
    """stats_payload: the raw `data` OBJECT from /volatility/stats (a single
    dict, NOT a list — see module docstring)."""
    if not isinstance(stats_payload, dict):
        stats_payload = {}
    return {
        "iv": _safe_float(stats_payload.get("iv")),
        "rv": _safe_float(stats_payload.get("rv")),
        "iv_rank": _safe_float(stats_payload.get("iv_rank")),
    }


def _extract_realized_vol(
    rows: Optional[list], stats: dict, today: Optional[date] = None,
) -> tuple[Optional[float], bool]:
    """rows: the raw `data` list from /volatility/realized (ascending by date).
    Returns (rv_20d_in_vol_points, used_stats_fallback).

    S177 semantics (Hydrinia uw_market_analyzer.get_raw_realized_vol, credited):
    walk BACKWARD to the latest row with a positive realized_volatility. Because
    the series is forward-shifted, that row's value is the realized vol over the
    window ending at its `unshifted_rv_date` — i.e. the most recently completed
    trailing window, which IS the current trailing realized vol. It is accepted
    only if that window ended within _RV_MAX_WINDOW_END_AGE_DAYS of `today`
    (falls back to the row's `date` when unshifted_rv_date is absent — stricter,
    since `date` precedes the window end); otherwise /volatility/stats "rv"."""
    today = today or _now_et().date()
    if rows:
        for row in reversed(rows):
            if not isinstance(row, dict):
                continue
            val = _safe_float(row.get("realized_volatility"))
            if val is None or val <= 0:
                continue
            end_raw = row.get("unshifted_rv_date") or row.get("date")
            try:
                end = date.fromisoformat(str(end_raw)[:10])
            except (TypeError, ValueError):
                break  # undatable -> can't prove freshness; use the stats fallback
            if 0 <= (today - end).days <= _RV_MAX_WINDOW_END_AGE_DAYS:
                return _to_vol_points(val), False
            break  # latest completed window is too old; don't dig further back

    stats_rv = stats.get("rv")
    if stats_rv is not None and stats_rv > 0:
        return _to_vol_points(stats_rv), True

    return None, False


def _classify_richness(iv_rv_spread: Optional[float], iv_rank: Optional[float]) -> Optional[str]:
    """rich/fair/cheap from iv_rv_spread (vol points) + iv_rank (0-100).
    See the threshold constants' docstring above for the design rationale.
    Requires iv_rv_spread; a missing iv_rank never blocks a call (treated as
    non-disconfirming), matching "missing is never zero" — it just can't
    reinforce it either way."""
    if iv_rv_spread is None:
        return None
    if iv_rv_spread >= _RICH_SPREAD_VOL_POINTS and (iv_rank is None or iv_rank >= _RICH_IV_RANK_MIN):
        return "rich"
    if iv_rv_spread <= _CHEAP_SPREAD_VOL_POINTS and (iv_rank is None or iv_rank <= _CHEAP_IV_RANK_MAX):
        return "cheap"
    return "fair"


# ---------------------------------------------------------------------------
# Per-symbol fetch (UW HTTP layer)
# ---------------------------------------------------------------------------


async def _fetch_symbol_bundle(symbol: str) -> dict:
    """Sequential (not concurrent) 3-4 call UW fetch for one symbol — mirrors
    Hydrinia's S163 sequential-not-concurrent fix, avoiding a UW budget
    burst. Every call goes through uw_flow_intel._uw_get, which already
    returns {} on ANY failure (no key, budget/quota denied, HTTP error,
    exception, ...) and never raises under normal operation — this function
    adds no additional error handling of its own beyond treating {} as "no
    data" for that source.

    Returns {ts_rows, stats_data, realized_rows, skew_rows, skew_expiry,
    calls, sources}."""
    from agent.integrations.uw_flow_intel import _uw_get

    calls = 0
    sources: list[str] = []

    ts_payload = await _uw_get(f"/stock/{symbol}/volatility/term-structure")
    calls += 1
    ts_rows = ts_payload.get("data") if isinstance(ts_payload, dict) else None
    ts_rows = ts_rows if isinstance(ts_rows, list) else []
    if ts_rows:
        sources.append("term_structure")

    stats_payload = await _uw_get(f"/stock/{symbol}/volatility/stats")
    calls += 1
    stats_data = stats_payload.get("data") if isinstance(stats_payload, dict) else None
    stats_data = stats_data if isinstance(stats_data, dict) else {}
    if stats_data:
        sources.append("vol_stats")

    realized_payload = await _uw_get(
        f"/stock/{symbol}/volatility/realized",
        {"days": _REALIZED_VOL_DAYS, "timeframe": _REALIZED_VOL_TIMEFRAME},
    )
    calls += 1
    realized_rows = realized_payload.get("data") if isinstance(realized_payload, dict) else None
    realized_rows = realized_rows if isinstance(realized_rows, list) else []
    # "realized" is credited in _build_entry only when a row was actually used.

    skew_expiry = _pick_skew_expiry(ts_rows)
    skew_rows: list = []
    if skew_expiry:
        skew_payload = await _uw_get(
            f"/stock/{symbol}/historical-risk-reversal-skew",
            {"delta": _SKEW_DELTA, "expiry": skew_expiry},
        )
        calls += 1
        skew_rows = skew_payload.get("data") if isinstance(skew_payload, dict) else None
        skew_rows = skew_rows if isinstance(skew_rows, list) else []
        if skew_rows:
            sources.append("skew")

    return {
        "ts_rows": ts_rows,
        "stats_data": stats_data,
        "realized_rows": realized_rows,
        "skew_rows": skew_rows,
        "skew_expiry": skew_expiry,
        "calls": calls,
        "sources": sources,
    }


def _compose_symbol_result(symbol: str, bundle: dict, as_of_iso: str) -> dict:
    """Pure function: bundle (raw UW rows) -> the contract dict for one symbol."""
    quality_notes: list[str] = []

    ts = _compute_term_structure(bundle["ts_rows"])
    if not bundle["ts_rows"]:
        quality_notes.append("term-structure unavailable this cycle")

    stats = _extract_stats(bundle["stats_data"])
    if not bundle["stats_data"]:
        quality_notes.append("volatility/stats unavailable this cycle")

    rv_20d, used_fallback = _extract_realized_vol(bundle["realized_rows"], stats)
    sources = list(bundle["sources"])
    if rv_20d is not None and not used_fallback:
        sources.append("realized")
    if used_fallback:
        quality_notes.append("realized_vol: no recent completed window in volatility/realized (S177); used volatility/stats.rv")
    elif not bundle["realized_rows"]:
        quality_notes.append("volatility/realized unavailable this cycle")

    if bundle["skew_expiry"] is None:
        quality_notes.append("skew skipped: no usable expiry from term-structure (required UW param)")
    skew_25d = _compute_skew(bundle["skew_rows"])
    if bundle["skew_expiry"] is not None and skew_25d is None:
        quality_notes.append("skew unavailable (empty or denied)")

    # iv_rv_spread = iv_30d - rv_20d. If term-structure has no iv_30d, fall
    # back to volatility/stats' own "iv" field so one endpoint's outage
    # doesn't blank the richness read entirely.
    iv_30d_for_spread = ts["iv_30d"]
    if iv_30d_for_spread is None:
        iv_30d_for_spread = _to_vol_points(stats.get("iv"))
        if iv_30d_for_spread is not None:
            quality_notes.append("iv_30d unavailable from term-structure; used volatility/stats.iv for iv_rv_spread")

    if iv_30d_for_spread is not None and rv_20d is not None:
        iv_rv_spread = round(iv_30d_for_spread - rv_20d, 4)
    else:
        iv_rv_spread = None

    iv_rank = _to_iv_rank_scale(stats.get("iv_rank"))
    richness = _classify_richness(iv_rv_spread, iv_rank)

    return {
        "symbol": symbol,
        "iv_front": ts["iv_front"],
        "iv_30d": ts["iv_30d"],
        "term_slope": ts["term_slope"],
        "term_state": ts["term_state"],
        "rv_20d": rv_20d,
        "iv_rv_spread": iv_rv_spread,
        "iv_rank": iv_rank,
        "skew_25d": skew_25d,
        "richness": richness,
        "as_of": as_of_iso,
        "sources": sources,
        "quality_notes": quality_notes,
    }


# ---------------------------------------------------------------------------
# Symbol universe (lazy DPOF import — never blocks this module's own tests)
# ---------------------------------------------------------------------------


def _resolve_symbols() -> list[str]:
    """DPOF's configured underlyings (first = primary), uppercased and
    deduped, order preserved. Falls back to ("SPY", "QQQ") on ANY failure —
    missing agent.dpof, a bad config, or an empty underlyings tuple."""
    try:
        from agent.dpof.config import load_dpof_config

        cfg = load_dpof_config()
        symbols = [str(s).strip().upper() for s in (cfg.underlyings or ()) if str(s).strip()]
        if symbols:
            seen: set[str] = set()
            out: list[str] = []
            for s in symbols:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out
    except Exception as e:
        logger.debug(f"vol_surface: DPOF config unavailable, using default symbols ({type(e).__name__}: {e})")
    return list(_DEFAULT_SYMBOLS)


def _within_refresh_window() -> bool:
    """Weekdays, extended hours (04:00-20:00 ET). Defense-in-depth only —
    the orchestrator owns the real schedule."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    return _REFRESH_START_HOUR_ET <= now.hour < _REFRESH_END_HOUR_ET


# ---------------------------------------------------------------------------
# Atomic write / merge-read
# ---------------------------------------------------------------------------


def _load_existing() -> dict:
    """BLOCKING plain read (no mtime cache — used once per refresh to merge).
    Returns {"symbols": {}} on any failure/absence."""
    try:
        with open(_file_path(), "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("symbols"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"symbols": {}}


def _write_atomic(payload: dict) -> bool:
    """BLOCKING. tmp + fsync + os.replace. Returns False (and leaves the
    previous file untouched) on any failure — never raises."""
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
        logger.warning(f"vol_surface: atomic write failed, keeping previous file ({type(e).__name__}: {e})")
        return False


# ---------------------------------------------------------------------------
# Refresh (PTB JobQueue-compatible)
# ---------------------------------------------------------------------------


_refresh_lock = asyncio.Lock()   # single-flight (same pattern as econ_surprise/newsflash_lite)


async def refresh_vol_surface(context=None) -> Optional[dict]:
    """PTB scheduled-job callback. Fetches term-structure/stats/realized/skew
    for each DPOF underlying (default SPY, QQQ) and atomically writes
    vol_surface.json. NEVER raises — any failure is logged and the previous
    file (or an individual symbol's previous entry) is left untouched.

    Advisory-only: makes zero agent.dpof.broker/orders/engine calls; the
    flow signal here only informs, never trades.

    Returns a status dict {refreshed, errors, api_calls, as_of} on success,
    None if skipped (outside window / no UW key / already running) or on failure."""
    if _refresh_lock.locked():
        logger.debug("vol_surface: refresh already in progress, skipping this call")
        return None
    async with _refresh_lock:
        return await _refresh_vol_surface_locked(context)


async def _refresh_vol_surface_locked(context=None) -> Optional[dict]:
    try:
        if not _within_refresh_window():
            return None

        try:
            from agent.config import config

            if not (config.uw_api_key or ""):
                logger.debug("vol_surface: no UW_API_KEY configured, skipping refresh")
                return None
        except Exception as e:
            logger.debug(f"vol_surface: config import failed ({type(e).__name__}: {e})")
            return None

        symbols = _resolve_symbols()
        as_of_iso = datetime.now(timezone.utc).isoformat()

        loop = asyncio.get_running_loop()
        existing = await loop.run_in_executor(None, _load_existing)
        symbols_out: dict = dict(existing.get("symbols", {}))

        total_calls = 0
        errors: list[str] = []
        refreshed: list[str] = []

        for symbol in symbols:
            try:
                bundle = await _fetch_symbol_bundle(symbol)
            except Exception as e:
                logger.warning(f"vol_surface: fetch failed for {symbol} ({type(e).__name__}: {e})")
                errors.append(symbol)
                continue

            total_calls += bundle["calls"]

            if not bundle["sources"]:
                # Zero successful endpoints this cycle. Keep whatever good
                # data was already on disk for this symbol rather than
                # overwriting it with an all-None entry; if there was never
                # an entry, write an explicit all-None one so get_vol_surface
                # sees "we tried, nothing came back" instead of a KeyError.
                if symbol not in symbols_out:
                    entry = _compose_symbol_result(symbol, bundle, as_of_iso)
                    entry["quality_notes"].append("all UW endpoints unavailable this cycle")
                    symbols_out[symbol] = entry
                else:
                    logger.debug(f"vol_surface: {symbol} — all endpoints failed, keeping previous entry")
                errors.append(symbol)
                continue

            symbols_out[symbol] = _compose_symbol_result(symbol, bundle, as_of_iso)
            refreshed.append(symbol)

        payload = {"generated_at": as_of_iso, "symbols": symbols_out}

        ok = await loop.run_in_executor(None, _write_atomic, payload)
        if not ok:
            return None

        logger.info(
            f"vol_surface: refreshed {refreshed} ({total_calls} UW calls, {len(errors)} symbol errors)"
        )
        return {"refreshed": refreshed, "errors": errors, "api_calls": total_calls, "as_of": as_of_iso}
    except Exception as e:  # noqa: BLE001 — scheduled job: must never raise into the scheduler
        logger.warning(f"vol_surface: refresh failed ({type(e).__name__}: {e})")
        return None


# ---------------------------------------------------------------------------
# Read (sync, mtime-cached, never raises)
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
        logger.debug(f"vol_surface: unreadable {path} ({type(e).__name__}: {e})")
        return None
    if not isinstance(data, dict):
        return None
    with _cache_lock:
        _cache[path] = (key, data)
    return data


def get_vol_surface(symbol: str, max_age_seconds: int = _DEFAULT_MAX_AGE_SECONDS) -> Optional[dict]:
    """Sync, mtime-cached, NEVER raises. None when the file is missing, the
    symbol has no entry, the entry is malformed, or it is older than
    max_age_seconds."""
    try:
        if not symbol:
            return None
        data = _read_cached(_file_path())
        if not data:
            return None
        symbols = data.get("symbols")
        if not isinstance(symbols, dict):
            return None
        entry = symbols.get(str(symbol).strip().upper())
        if not isinstance(entry, dict):
            return None
        as_of = entry.get("as_of")
        if not isinstance(as_of, str):
            return None
        try:
            dt = datetime.fromisoformat(as_of)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        if age < 0 or age > max_age_seconds:
            return None
        return entry
    except Exception as e:  # noqa: BLE001 — public read API: must never raise
        logger.debug(f"vol_surface: get_vol_surface failed ({type(e).__name__}: {e})")
        return None
