"""
CLIO DPOF — live market data feed (I/O).

DPOFDataFeed.snapshot() is the single place DPOF reads the outside world:
UW options-flow pillars (NOFA/premium/darkpool/GEX), 1-min bars -> session
VWAP/momentum, VIX, and the econ calendar / breaking news / vol regime
readings. Everything else in agent/dpof/ is pure and only ever sees the
MarketSnapshot this module builds.

Operator directive (Task 1, 2026-09-25): econ blackout, breaking news, VIX
HALT and blocked vol regimes are no longer entry BLOCKS anywhere in this
module -- they are DECISION CONTEXT for the decider now (see
docs/design/clio_dpof.md §4/Task1). `snapshot()`'s `risk_flags`/
`econ_events_soon`/`data_healthy` outputs are unchanged in shape (still
consumed by signals.py/engine.py/jev.py as before) but `data_healthy` no
longer depends on econ-calendar/learning-state gate-SOURCE availability --
only on the flow pillars/price data actually feeding the composite.
`DPOFDataFeed.get_intel_ctx()` is a separate, best-effort ADVISORY method
(Task 2) that bundles upcoming econ releases + recent surprises (Builder E),
per-underlying vol surface (Builder V), breaking-news state (Builder N, with
a Lighthouse/learning_state fallback), and the econ calendar's own freshness
(`calendar_status`) for engine.py to merge into every decider ctx.

MISSING IS NEVER ZERO (see agent/dpof/models.py docstring and
docs/design/clio_dpof.md §1). Hydrinia's DPOF ran a full trading day on flat
FPS=0.00 because every pillar parser fail-opened an API failure to 0.0
("expected dict, got list"). Every fetch in this module distinguishes three
outcomes and never confuses them:
  1. FAILURE (timeout, exception, {} with no "data" key, or the wrong shape
     entirely -- e.g. a list where a dict was expected) -> PillarReading.missing()
     / the previous cached reading is kept and simply ages toward staleness.
  2. Genuine empty result (a real, well-shaped `{"data": []}`) -> a legitimate
     value (0.0 for a pillar), tagged detail["empty"] = True.
  3. A real value.
No exception from a CLIO integration call is ever silently turned into 0.0.

Time-budget discipline (uw_flow_intel._uw_get can sleep up to 5 min inside a
UW 429 backoff, and DPOF's cycle is ~30s under a ~25s asyncio.wait_for):
  - Every individual UW HTTP call is wrapped in asyncio.wait_for(_UW_CALL_TIMEOUT_SECONDS).
  - All UW/bars/VIX/gate fetches for a cycle run CONCURRENTLY (asyncio.gather),
    so a full refresh costs about one per-call timeout at worst, not N of them.
  - The whole snapshot() refresh phase is additionally wrapped in an outer
    asyncio.wait_for(_SNAPSHOT_BUDGET_SECONDS) as defense-in-depth.
  - asyncio.CancelledError is never caught by an `except Exception` clause
    anywhere in this module (CancelledError is a BaseException, not an
    Exception) -- real cancellation always propagates out of snapshot().

All CLIO integration imports (agent.config, agent.integrations.*, httpx) are
LAZY -- inside the functions/methods that need them -- so importing this
module alone never imports agent.config or httpx. Only sibling agent.dpof.*
modules (config, models, signals) are imported at module level; none of them
import agent.config or httpx (they're pure -- stdlib + agent.dpof.config/
models only -- see their own docstrings/CLAUDE.md dataclass rules).

Blocking file reads used by the CLIO integrations this module reads
(econ_calendar's JSON cache, learning_engine's learning_state.json, and
uw_signal_bridge's snapshot file) are always run via
loop.run_in_executor(None, ...) -- never flock/open() directly on the
asyncio event loop thread (CLAUDE.md anti-pattern #19).
"""

from __future__ import annotations

import asyncio
import hashlib
import contextlib
import logging
import math
import os
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from agent.dpof.config import DPOFConfig, JevMode
from agent.dpof.dynamic_weights import DynamicWeightsManager
from agent.dpof.models import MarketSnapshot, PILLAR_NAMES, PillarReading, UnderlyingSnapshot
from agent.dpof.signals import KRONOS_PILLAR, gex_factor, kronos_pillar_from_forecast

logger = logging.getLogger("clio.dpof.data")

# Schema-drift canary: rows that carry none of the recognized fields are logged
# at most once per (pillar, ticker) per window. Field NAMES only, never values.
_SCHEMA_DRIFT_WARN_SECONDS = 3600.0
_schema_drift_last_warn: dict[tuple[str, str], float] = {}


def _warn_schema_drift(name: str, ticker: str, field_names: list[str]) -> None:
    key = (name, ticker)
    now_mono = time.monotonic()
    last = _schema_drift_last_warn.get(key)
    if last is not None and (now_mono - last) < _SCHEMA_DRIFT_WARN_SECONDS:
        return
    _schema_drift_last_warn[key] = now_mono
    logger.warning(
        "DPOF %s schema drift for %s: rows carry no recognized fields; row field names=%s",
        name, ticker, field_names,
    )


# Dynamic pillar weights (agent/dpof/dynamic_weights.py): every hook is fail-static. A failure
# degrades to the static weights and logs at most one WARNING per hook per hour.
_DW_WARN_SECONDS = 3600.0
_dw_last_warn: dict[str, float] = {}


def _warn_dw(name: str, exc: BaseException) -> None:
    now_mono = time.monotonic()
    last = _dw_last_warn.get(name)
    if last is not None and (now_mono - last) < _DW_WARN_SECONDS:
        return
    _dw_last_warn[name] = now_mono
    logger.warning(
        "DPOF dynamic weights %s failed (%s: %s); static weights for this cycle",
        name, type(exc).__name__, exc,
    )


# Premium shadow alternatives (R-37, SHADOW ONLY): computed beside the live premium reading from the same
# tick list and handed to the manager. No decision path reads them. The live formula is unchanged and
# lives in agent.integrations.uw_flow_intel._compute_premium_signals (last 20 ticks).
_PREM_ALT_W60 = 60
_PREM_ALT_EWMA_ALPHA = 0.1
_PREM_ALT_NEW_KEYS = ("net_call_premium", "net_put_premium")
_PREM_ALT_LEGACY_KEYS = ("net_premium", "call_premium", "put_premium")


def _prem_strict(v) -> float | None:
    """v as a finite float, or None (garbage, NaN, inf, overflow)."""
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _prem_tick(t) -> tuple[float, float, float] | None:
    """(net, call, put) of one tick with the live field handling, or None when the tick is unusable
    (not a dict, or no recognised premium field holds a finite number)."""
    from agent.integrations.uw_flow_intel import _premium_num

    if not isinstance(t, dict):
        return None
    if "net_call_premium" in t or "net_put_premium" in t:
        if all(_prem_strict(t.get(k)) is None for k in _PREM_ALT_NEW_KEYS):
            return None
        nc = _premium_num(t.get("net_call_premium"))
        npr = _premium_num(t.get("net_put_premium"))
        return nc - npr, nc, npr
    if all(_prem_strict(t.get(k)) is None for k in _PREM_ALT_LEGACY_KEYS):
        return None
    return (_premium_num(t.get("net_premium")), _premium_num(t.get("call_premium")),
            _premium_num(t.get("put_premium")))


def _prem_clamp(x: float) -> float:
    return round(min(1.0, max(-1.0, x)), 4)


def _premium_alternatives(ticks) -> dict:
    """{"w60", "ewma", "day"}: live-formula score over the last 60 ticks, EWMA of the per-tick share
    (alpha 0.1, first tick initialises) and the live-formula score over all ticks. Each is None when
    no usable tick exists. Rounded to 4 decimals. Never raises for list input; the caller also guards."""
    out: dict = {"w60": None, "ewma": None, "day": None}
    if not isinstance(ticks, list) or not ticks:
        return out
    tail_rows = [r for r in (_prem_tick(t) for t in ticks[-_PREM_ALT_W60:]) if r is not None]
    if tail_rows:
        net = sum(r[0] for r in tail_rows)
        cp = sum(r[1] for r in tail_rows)
        pp = sum(r[2] for r in tail_rows)
        out["w60"] = _prem_clamp(net / max(abs(cp) + abs(pp), 1))
    all_rows = [r for r in (_prem_tick(t) for t in ticks) if r is not None]
    if all_rows:
        net = sum(r[0] for r in all_rows)
        cp = sum(r[1] for r in all_rows)
        pp = sum(r[2] for r in all_rows)
        out["day"] = _prem_clamp(net / max(abs(cp) + abs(pp), 1))
        ewma = None
        for net_i, cp_i, pp_i in all_rows:
            share = net_i / max(abs(cp_i) + abs(pp_i), 1)
            ewma = share if ewma is None else _PREM_ALT_EWMA_ALPHA * share + (1 - _PREM_ALT_EWMA_ALPHA) * ewma
        out["ewma"] = _prem_clamp(ewma)
    return out

ET = ZoneInfo("America/New_York")

# ── Time-budget constants (module-level so tests can monkeypatch them down
#    to make timeout/hang tests fast without waiting out the real defaults) ──
_UW_CALL_TIMEOUT_SECONDS = 8.0
_BARS_CALL_TIMEOUT_SECONDS = 8.0
_VIX_CALL_TIMEOUT_SECONDS = 8.0
_FILE_GATE_TIMEOUT_SECONDS = 5.0
_ECON_CACHE_MAX_AGE_SECONDS = 36 * 3600.0   # econ_calendar refreshes daily ~06:00 ET
_SNAPSHOT_BUDGET_SECONDS = 15.0

# ── NOFA per-ticker windowing ─────────────────────────────────────────────
_NOFA_WINDOW_SECONDS = 1800.0   # only alerts from the last 30 min feed the pillar
_NOFA_MIN_ALERTS = 3            # fewer than this in-window -> "thin", not a real signal

# ── Headlines (Jev advisory input) ─────────────────────────────────────────
_HEADLINE_REFRESH_SECONDS = 600.0        # cadence for kicking off a background rescan
_HEADLINE_SCAN_TIMEOUT_SECONDS = 45.0    # bound on the executor-run scan_current_window()
_HEADLINE_MAX_AGE_SECONDS = 900.0        # cache older than this is never served to a snapshot
_HEADLINE_LOOKBACK_SECONDS = 3600.0      # only headlines published in the last 60 min qualify

# ── Self-computed flow fallback (agent.flowcollector) ──────────────────────
# Selfflow's own gex_regime vocabulary ("positive"/"negative"/"neutral" --
# agent.flowcollector.aggregator.compute_gex's relative-band dealer-gamma
# sign) is DIFFERENT from UW's own regime labels ("ACCELERATING"/
# "STABILIZING", uw_flow_intel._compute_gex) that signals.gex_factor()/
# jev.py/_gex_regime_cache already understand -- passing selfflow's raw
# strings into gex_factor() unmapped would silently land BOTH "positive" and
# "negative" on its neutral else-branch (factor 0.0), losing all
# directionality. Mapped explicitly (never passed through raw): selfflow
# "negative" (dealers net short gamma, the same SqueezeMetrics/SpotGamma-
# replica assumption both aggregator.py's and UW's "ACCELERATING" share --
# amplifies moves) -> UW's "ACCELERATING" (gex_factor +0.5); selfflow
# "positive" (dealers net long gamma, dampens moves) -> UW's "STABILIZING"
# (gex_factor -0.3); "neutral" already lands on gex_factor's else-branch
# (0.0) without translation -- mapped to "NEUTRAL" here anyway for a uniform
# vocabulary in UnderlyingSnapshot.gex_regime regardless of source.
_SELFFLOW_GEX_REGIME_TO_UW = {"negative": "ACCELERATING", "positive": "STABILIZING", "neutral": "NEUTRAL"}

# ── UW endpoints reused/mirrored from agent/integrations/uw_flow_intel.py ──
_EP_FLOW_ALERTS = "/option-trades/flow-alerts"      # per-ticker (ticker_symbol param), NOFA
_EP_MARKET_TIDE = "/market/market-tide"             # uw_flow_intel.py:331-335 (premium, market-wide)
_EP_NET_PREM_TICKS = "/stock/{ticker}/net-prem-ticks"   # uw_flow_intel.py:338-346, per ticker
_EP_DARKPOOL_TICKER = "/darkpool/{ticker}"          # uw_flow_intel.py:476-484 loop body, per ticker
_EP_GEX_STRIKE = "/stock/{ticker}/spot-exposures/strike"  # uw_flow_intel.py:555-562 loop body, per ticker
# Decision CONTEXT only (models.UnderlyingSnapshot.nope/.max_pain docstring:
# "never a mechanical gate") -- not mirrored from uw_flow_intel.py (it has no
# nope/max-pain consumer); endpoints verified directly against
# docs/api/uw.txt + the scratchpad's uw_openapi.json
# (PublicApi.TickerController.nope / .max_pain operations).
_EP_NOPE_TICKER = "/stock/{ticker}/nope"            # per-ticker, per-minute-of-session readings
_EP_MAX_PAIN_TICKER = "/stock/{ticker}/max-pain"    # per-ticker, per-expiry readings


# ═══════════════════════════════════════════════════════════════════════════
#  PURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _session_start_utc(now_epoch: float) -> datetime:
    """Today's (ET calendar date) 09:30 session open, as a tz-aware UTC datetime.

    Used as the `start_utc` argument to broker.get_minute_bars(). If `now_epoch`
    falls before today's 9:30 ET (pre-market), this is in the future relative to
    "now" -- that's fine, the broker is expected to return an empty bars list
    for a session that hasn't started, which compute_vwap_and_returns() treats
    as a legitimate empty result, not a failure.
    """
    now_et = datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(ET)
    session_open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    return session_open_et.astimezone(timezone.utc)


def compute_vwap_and_returns(bars: list[dict], now_epoch: float) -> dict:
    """PURE. Session VWAP + short-horizon momentum from ascending 1-min bars.

    Design choices (per BUILD_SPEC instruction to choose and document):
      - Unsorted input is SORTED (stable, by "t") rather than rejected. The
        broker contract promises ascending bars, but a single out-of-order or
        duplicated bar from a flaky feed should degrade to "sorted, cheap,
        deterministic" rather than either crash the cycle or silently corrupt
        VWAP/return math -- consistent with this layer's MISSING-IS-NEVER-ZERO
        discipline (a plausible-looking wrong number is worse than a defensive
        sort).
      - Malformed bar entries (not a dict, or missing/non-numeric "t"/"c") are
        skipped rather than raising. "h"/"l" missing or non-numeric fall back
        to "c" for that bar's typical-price contribution (better than dropping
        an otherwise-usable bar's volume from the VWAP entirely). "v"
        missing/non-numeric is treated as 0 (excluded from VWAP, per spec:
        "only v>0 bars").
      - Zero-volume bars are excluded from the VWAP sum but still count for
        last_price / bars_count / price_age_seconds / ret_5m / ret_15m (those
        just need a close price at a point in time, not volume weighting). If
        every bar has v<=0, vwap is None (never 0.0 -- there is no such thing
        as a legitimate zero VWAP; division-by-zero is avoided explicitly,
        never via float('inf')/nan, per CLAUDE.md anti-pattern #1).
      - ret_5m / ret_15m look for the latest bar at or before now-5m/now-15m.
        If no such bar exists (insufficient history / a gap at the start of
        the session), the return is None, never 0.0 -- "no data" and "flat
        0% return" are different facts and must never be confused.
    """
    out: dict = {
        "last_price": None,
        "vwap": None,
        "ret_5m": None,
        "ret_15m": None,
        "bars_count": 0,
        "price_age_seconds": None,
    }

    clean: list[dict] = []
    for b in bars or []:
        if not isinstance(b, dict):
            continue
        t_raw = b.get("t")
        c_raw = b.get("c")
        if t_raw is None or c_raw is None:
            continue
        try:
            t = float(t_raw)
            c = float(c_raw)
        except (TypeError, ValueError):
            continue
        v_raw = b.get("v")
        try:
            v = float(v_raw) if v_raw is not None else 0.0
        except (TypeError, ValueError):
            v = 0.0
        h_raw = b.get("h")
        l_raw = b.get("l")
        try:
            h = float(h_raw) if h_raw is not None else c
        except (TypeError, ValueError):
            h = c
        try:
            l = float(l_raw) if l_raw is not None else c
        except (TypeError, ValueError):
            l = c
        clean.append({"t": t, "c": c, "v": v, "h": h, "l": l})

    if not clean:
        return out

    clean.sort(key=lambda b: b["t"])
    out["bars_count"] = len(clean)

    last = clean[-1]
    out["last_price"] = last["c"]
    out["price_age_seconds"] = max(0.0, now_epoch - last["t"])

    num = 0.0
    den = 0.0
    for b in clean:
        if b["v"] > 0:
            tp = (b["h"] + b["l"] + b["c"]) / 3.0
            num += tp * b["v"]
            den += b["v"]
    if den > 0:
        out["vwap"] = num / den

    def _ret_vs(seconds_ago: float) -> float | None:
        cutoff = now_epoch - seconds_ago
        candidate = None
        for b in clean:
            if b["t"] <= cutoff:
                candidate = b
            else:
                break
        if candidate is None or candidate["c"] == 0:
            return None
        return (last["c"] - candidate["c"]) / candidate["c"]

    out["ret_5m"] = _ret_vs(300.0)
    out["ret_15m"] = _ret_vs(900.0)
    return out


def _parse_alert_timestamp(alert: dict) -> datetime | None:
    """Best-effort parse of a UW flow-alert's timestamp, trying created_at
    first then falling back to start_time/executed_at/date (field-name
    uncertainty across UW alert shapes -- try each, use whichever parses).
    Returns a tz-aware UTC datetime, or None if nothing on the alert parses.
    """
    for key in ("created_at", "start_time", "executed_at", "date"):
        raw = alert.get(key)
        if not raw:
            continue
        try:
            s = str(raw)
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
        except (ValueError, TypeError):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _filter_nofa_alerts(alerts: list, ticker: str, now_epoch: float) -> tuple[list, int]:
    """PURE. Keeps only alerts for `ticker` published within the last
    _NOFA_WINDOW_SECONDS of now_epoch. An alert whose ticker_symbol/ticker
    field is present and disagrees with `ticker` is dropped defensively (the
    API call is already scoped by ticker_symbol=<ticker>, but a response
    should never be trusted blindly). An alert whose timestamp can't be
    parsed at all is dropped and counted separately (never silently kept nor
    silently treated as "in window") -- returns (kept_alerts, unparsed_count).
    """
    cutoff = datetime.fromtimestamp(now_epoch, tz=timezone.utc) - timedelta(seconds=_NOFA_WINDOW_SECONDS)
    kept: list = []
    unparsed = 0
    for a in alerts:
        if not isinstance(a, dict):
            continue
        tk = a.get("ticker_symbol") or a.get("ticker")
        if tk and tk != ticker:
            continue
        ts = _parse_alert_timestamp(a)
        if ts is None:
            unparsed += 1
            continue
        if ts < cutoff:
            continue
        kept.append(a)
    return kept, unparsed


def _parse_latest_nope(entries: list) -> float | None:
    """PURE. `entries` is the UW GET /stock/{ticker}/nope response's "data"
    list -- per-minute NOPE readings for the session, in whatever order the
    API returns them (not documented as guaranteed-ascending, so this never
    assumes it). Returns the `nope` field (NOT `nope_fill` -- the "original
    NOPE calculation", per the endpoint's own description, is the canonical
    one this uses) of the latest-timestamped entry. None if `entries` is
    empty or nothing in it parses -- never a stand-in 0.0 (this module's own
    MISSING IS NEVER ZERO discipline; a real NOPE of exactly 0.0 is a
    legitimate reading, distinguishable from "couldn't compute" by returning
    None only when parsing genuinely fails).
    """
    best_ts: datetime | None = None
    best_val: float | None = None
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        raw_ts = entry.get("timestamp")
        if not raw_ts:
            continue
        try:
            s = str(raw_ts)
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            ts = datetime.fromisoformat(s)
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        raw_val = entry.get("nope")
        if raw_val is None:
            continue
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            continue
        if best_ts is None or ts > best_ts:
            best_ts = ts
            best_val = val
    return best_val


def _parse_nearest_max_pain(entries: list, today: date) -> float | None:
    """PURE. `entries` is the UW GET /stock/{ticker}/max-pain response's
    "data" list -- one {expiry, max_pain} pair per upcoming expiration as of
    the response's snapshot date. Returns the max_pain strike of the nearest
    expiry that is on or after `today`. The 1DTE target session (see
    agent.dpof.contracts.target_expiries) is almost always exactly this
    nearest expiry; matching by "nearest >= today" rather than an exact ISO-
    date string equality keeps this robust to UW's own weekend/holiday
    calendar not lining up byte-for-byte with contracts.target_expiries'
    (e.g. a 0DTE row UW still lists for an already-past cutoff today). None
    if `entries` is empty, nothing parses, or every listed expiry is already
    in the past relative to `today` -- never a fabricated 0.0.
    """
    best_expiry: date | None = None
    best_val: float | None = None
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        raw_exp = entry.get("expiry")
        if not raw_exp:
            continue
        try:
            exp = date.fromisoformat(str(raw_exp)[:10])
        except (ValueError, TypeError):
            continue
        if exp < today:
            continue
        raw_val = entry.get("max_pain")
        if raw_val is None:
            continue
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            continue
        if best_expiry is None or exp < best_expiry:
            best_expiry = exp
            best_val = val
    return best_val


def _shape_check_uw_response(res, what: str) -> tuple[list | None, str]:
    """Never trust the shape uw_flow_intel._uw_get hands back.

    {} (no "data" key) means the underlying HTTP call failed, was budget-
    rejected, or hit an auth/rate-limit error inside _uw_get -- FAILURE, never
    silently treated as zero flow (this is the exact Hydrinia Sep-7 failure
    mode this module exists to prevent). A response that isn't a dict at all,
    or whose "data" isn't a list -- e.g. "expected dict, got list" -- is also
    FAILURE, with a precise, mode-specific error string. Only a dict WITH a
    "data" key holding an actual (possibly empty) list is legitimate; an empty
    list there is genuine "no results this call", not a failure.
    """
    if not isinstance(res, dict):
        return None, f"{what}: expected dict, got {type(res).__name__}"
    if "data" not in res:
        return None, f"{what}: no 'data' key in response"
    data = res["data"]
    if not isinstance(data, list):
        return None, f"{what}: expected list for 'data', got {type(data).__name__}"
    return data, ""


# ── Module-level sync helpers for blocking file reads (run via executor) ──

def _econ_calendar_read_sync() -> tuple[bool, list]:
    """Returns (cache_file_present, events). Lazy-imports econ_calendar and
    reads its cache file -- both inside the executor thread, off the loop.

    econ_calendar.load_calendar() itself cannot distinguish "cache file never
    written" from "cache file exists but genuinely has 0 events this week"
    from "cache file corrupt" -- all three paths return [] inside that module
    (econ_calendar.py:258-285), and we must not edit that file. File-presence
    (os.path.isfile) is the only failure signal this module can observe
    without editing econ_calendar.py, so it is what data.py treats as
    "required gate source unavailable".
    """
    from agent.integrations import econ_calendar
    path = econ_calendar.CALENDAR_CACHE_PATH
    present = os.path.isfile(path)
    if present:
        # A present-but-stale cache (daily refresh silently failing) must not pass:
        # a week-old calendar would let an FOMC/CPI day through the blackout gate.
        age = time.time() - os.path.getmtime(path)
        if age > _ECON_CACHE_MAX_AGE_SECONDS:
            logger.warning("DPOF econ calendar cache is %.1fh old (> %.0fh); treating gate source as unavailable",
                           age / 3600.0, _ECON_CACHE_MAX_AGE_SECONDS / 3600.0)
            return False, []
    events = econ_calendar.load_calendar() if present else []
    if not isinstance(events, list):
        events = []
    return present, events


def _learning_state_read_sync() -> tuple[bool, dict]:
    """Returns (state_file_present, state). Same file-presence-as-failure-
    signal reasoning as _econ_calendar_read_sync: learning_engine._load_
    learning_state() returns a default dict on both "missing" and "corrupt",
    so file-presence is the distinguishable failure signal available without
    editing learning_engine.py.
    """
    from agent.integrations import learning_engine
    present = os.path.isfile(learning_engine.LEARNING_STATE_FILE)
    state = learning_engine._load_learning_state() if present else {}
    if not isinstance(state, dict):
        state = {}
    return present, state


def _vol_regime_read_sync() -> tuple[bool, dict]:
    """Returns (bridge_data_fresh, vol_regime_dict). uw_signal_bridge.get_uw_
    bridge().is_fresh()/.get_vol_regime() both do their own synchronous file
    reads (uw_signal_bridge.py's _read_snapshot), so both must run off the
    event loop; is_fresh() is checked first and get_vol_regime() is skipped
    entirely when stale (per BUILD_SPEC: "ONLY if bridge data fresh").
    """
    from agent.integrations.uw_signal_bridge import get_uw_bridge
    bridge = get_uw_bridge()
    fresh = bool(bridge.is_fresh())
    regime = bridge.get_vol_regime() if fresh else {}
    if not isinstance(regime, dict):
        regime = {}
    return fresh, regime


def _econ_calendar_status_sync() -> tuple[bool, float]:
    """(cache_file_present, age_hours). Pure file-presence/mtime check used
    ONLY for `calendar_status` decision context (Task 1, 2026-09-25: calendar
    staleness is advisory now, never a health/veto signal) -- kept separate
    from _econ_calendar_read_sync, which still collapses "stale" and "missing"
    into one failure signal for the (unrelated) `gate_source_unavailable:
    econ_calendar` health_issues note.
    """
    from agent.integrations import econ_calendar
    path = econ_calendar.CALENDAR_CACHE_PATH
    if not os.path.isfile(path):
        return False, 0.0
    age = time.time() - os.path.getmtime(path)
    return True, age / 3600.0


_KRONOS_TRACK_METRIC_KEYS = ("n", "hit_rate", "hit_rate_confident", "brier")


def _kronos_ctx_read_sync(symbols: tuple[str, ...], max_age_seconds: float, now: Optional[float] = None) -> dict:
    """Kronos advisory context (Task K2, agent/kronos -- a separate CLIO
    service; docs/design/clio_kronos.md §4). Pure file I/O -- call via
    run_in_executor, never on the event loop (CLAUDE.md anti-pattern #19).

    Returns {symbol: "unavailable" | {"asof_bar", "horizons": {...},
    "track_record": {...}}} for every configured symbol. "unavailable" (the
    literal string, not a zeroed/faked reading -- CLAUDE.md "no fake zeros")
    covers: agent.kronos not installed yet, the forecast file missing/stale
    (state.read_forecast's own max_age_seconds gate), or that symbol's status
    isn't "ok". `track_record` is trimmed to the 4 fields the decider/log
    actually use (n, hit_rate, hit_rate_confident, brier) per horizon, each
    under "all"/"recent" -- omitted (not a horizon key at all) when the
    scorer has nothing for that horizon yet, never zeroed.
    """
    out: dict[str, object] = {sym: "unavailable" for sym in symbols}
    try:
        from agent.kronos import state as kronos_state
    except Exception as e:
        logger.debug(f"DPOF intel: agent.kronos.state unavailable: {type(e).__name__}: {e}")
        return out

    try:
        forecast = kronos_state.read_forecast(now=now, max_age_seconds=max_age_seconds)
        scores = kronos_state.read_scores()
    except Exception as e:
        logger.debug(f"DPOF intel: kronos state read failed: {type(e).__name__}: {e}")
        return out

    scores_by_symbol = scores.get("symbols") if isinstance(scores, dict) else None
    if not isinstance(scores_by_symbol, dict):
        scores_by_symbol = {}

    for sym in symbols:
        entry = kronos_state.symbol_forecast(forecast, sym)
        if entry is None:
            continue  # stays "unavailable" -- missing/stale/non-"ok" status

        horizons_out: dict[str, dict] = {}
        raw_horizons = entry.get("horizons")
        if isinstance(raw_horizons, dict):
            for h in kronos_state.HORIZONS:
                hz = raw_horizons.get(h)
                if isinstance(hz, dict):
                    horizons_out[h] = {
                        "p_up": hz.get("p_up"),
                        "mean_return_pct": hz.get("mean_return_pct"),
                        "q10_return_pct": hz.get("q10_return_pct"),
                        "q90_return_pct": hz.get("q90_return_pct"),
                        "expected_abs_move_pct": hz.get("expected_abs_move_pct"),
                    }

        track_out: dict[str, dict] = {}
        sym_scores = scores_by_symbol.get(sym)
        if isinstance(sym_scores, dict):
            for h in kronos_state.HORIZONS:
                hz_scores = sym_scores.get(h)
                if not isinstance(hz_scores, dict):
                    continue
                track_entry: dict[str, dict] = {}
                for bucket in ("all", "recent"):
                    metrics = hz_scores.get(bucket)
                    if isinstance(metrics, dict):
                        track_entry[bucket] = {k: metrics.get(k) for k in _KRONOS_TRACK_METRIC_KEYS}
                if track_entry:
                    track_out[h] = track_entry

        if not horizons_out and not track_out:
            continue  # nothing usable this cycle -- stays "unavailable"
        out[sym] = {"asof_bar": entry.get("asof_bar"), "horizons": horizons_out, "track_record": track_out}

    return out


def _kronos_pillar_read_sync(
    symbols: tuple[str, ...], max_age_seconds: float, horizon: int, now: Optional[float] = None,
) -> dict:
    """Kronos LIVE pillar readings (docs/design/clio_kronos.md §4b): {symbol: PillarReading} for every
    symbol with a fresh, usable forecast -- absent otherwise (never a 0.0 stand-in). File I/O: call via
    run_in_executor (CLAUDE.md anti-pattern #19)."""
    out: dict = {}
    try:
        from agent.kronos import state as kronos_state
    except Exception as e:
        logger.debug(f"DPOF data: agent.kronos.state unavailable: {type(e).__name__}: {e}")
        return out
    forecast = kronos_state.read_forecast(now=now, max_age_seconds=max_age_seconds)
    for sym in symbols:
        got = kronos_pillar_from_forecast(kronos_state.symbol_forecast(forecast, sym), horizon)
        if got is None:
            continue
        value, detail = got
        out[sym] = PillarReading(name=KRONOS_PILLAR, value=value, available=True, source="kronos", detail=detail)
    return out


# R-18 (S215): severity-3 newsflash events that name no market link (e.g. a casualty headline) were read by the
# decider as "severity-3 risk-off". CLIO_DPOF_NEWSFLASH_REQUIRE_LINK: "log" (default: INFO count only, context text
# unchanged) | "1" (show such events at severity 2, DPOF text only) | "0" (off). Unset or bad values mean "log".
_NEWSFLASH_REQUIRE_LINK_ENV = "CLIO_DPOF_NEWSFLASH_REQUIRE_LINK"


def _newsflash_require_link_mode() -> str:
    raw = (os.environ.get(_NEWSFLASH_REQUIRE_LINK_ENV) or "").strip().lower()
    return raw if raw in ("log", "1", "0") else "log"


def _apply_newsflash_link_mode(state: dict) -> dict:
    """The breaking-state dict the decider context is built from. The market_linked key (R-18) is stripped so the
    context text stays byte-identical to the pre-R-18 shape in every mode; mode "1" additionally shows an unlinked
    severity-3 event at severity 2 (this copy only -- newsflash_lite's own state is never touched)."""
    events = state.get("events")
    if not isinstance(events, list):
        return state
    mode = _newsflash_require_link_mode()
    shown_events: list = []
    would_cap = 0
    for ev in events:
        if not isinstance(ev, dict):
            shown_events.append(ev)
            continue
        shown = {k: v for k, v in ev.items() if k != "market_linked"}
        if ev.get("market_linked") is False and ev.get("severity") == 3:
            would_cap += 1
            if mode == "1":
                shown["severity"] = 2
        shown_events.append(shown)
    out = {**state, "events": shown_events}
    if would_cap and mode == "1":
        levels = [
            e.get("severity") for e in shown_events
            if isinstance(e, dict) and isinstance(e.get("severity"), int) and not isinstance(e.get("severity"), bool)
        ]
        out["max_severity"] = max(levels, default=0)
        logger.info("DPOF intel: capped %d sev3 newsflash event(s) with no market link to severity 2 (%s=1)",
                    would_cap, _NEWSFLASH_REQUIRE_LINK_ENV)
    elif would_cap and mode == "log":
        logger.info("DPOF intel: would cap sev3->2 (no market link) on %d newsflash event(s); context unchanged (%s=log)",
                    would_cap, _NEWSFLASH_REQUIRE_LINK_ENV)
    return out


def _normalize_lighthouse_breaking(breaking: dict, present: bool) -> dict:
    """Normalizes the existing Lighthouse/learning_state breaking-news dict
    (`state["breaking_news"]`, written by agent/integrations/market_events.py
    and consumed by learning_engine.py) into Builder N's
    newsflash_lite.get_breaking_state() contract shape (INTEL_V3_SPEC.md) --
    used ONLY as a fallback when N's own module is unavailable (Task 2).

    This source has no graded severity of its own, so a single synthetic
    event's severity is derived from the size of the move that triggered it
    (spy_pct/vix_change) -- a documented, simple heuristic (never tuned
    against live data), not a claim of parity with N's own analysis.
    """
    active = present and bool(breaking.get("active"))
    if not active:
        return {
            "active": False, "max_severity": 0, "events": [], "as_of": None,
            "stale": not present, "source": "lighthouse_fallback",
        }
    try:
        spy_pct = float(breaking.get("spy_pct")) if breaking.get("spy_pct") is not None else None
    except (TypeError, ValueError):
        spy_pct = None
    try:
        vix_change = float(breaking.get("vix_change")) if breaking.get("vix_change") is not None else None
    except (TypeError, ValueError):
        vix_change = None
    severity = 3 if ((spy_pct is not None and abs(spy_pct) >= 2.0) or (vix_change is not None and abs(vix_change) >= 8.0)) else 2
    triggered_at = breaking.get("triggered_at")
    event_type = breaking.get("event_type") or "breaking_news"
    headline = f"{event_type} (SPY {spy_pct}%, VIX chg {vix_change})" if spy_pct is not None or vix_change is not None else str(event_type)
    event = {
        # No triggered_at -> a content-derived id, so two distinct untimestamped
        # events never collapse onto one id (engine dedupes review triggers by id).
        "id": f"lighthouse:{triggered_at}" if triggered_at else (
            "lighthouse:" + hashlib.sha1(f"{event_type}|{spy_pct}|{vix_change}".encode()).hexdigest()[:12]
        ),
        "headline": headline,
        "category": event_type,
        "tier": None,
        "severity": severity,
        "confidence": None,
        "source": "lighthouse",
        "ts": triggered_at,
        "tickers": [],           # Lighthouse's breaking-news state is market-wide, not per-ticker
        "direction_hint": None,
    }
    return {"active": True, "max_severity": severity, "events": [event], "as_of": triggered_at, "stale": False, "source": "lighthouse_fallback"}


# Advisory intel (Task 2, INTEL_V3_SPEC.md): bounds the whole get_intel_ctx()
# gather as defense-in-depth (mirrors _SNAPSHOT_BUDGET_SECONDS) even though
# every E/V/N reader it calls is contracted "sync, never raise, mtime-cached".
_INTEL_BUDGET_SECONDS = 8.0


# ═══════════════════════════════════════════════════════════════════════════
#  DPOFDataFeed
# ═══════════════════════════════════════════════════════════════════════════

class DPOFDataFeed:
    """Builds one MarketSnapshot per cycle from cached, independently-aged
    per-pillar/per-ticker readings. Nothing here is ever overwritten to a
    "safe-looking" default on failure -- a failed refresh just leaves the
    previous cached reading in place, aging toward the 3x-uw_refresh
    staleness cutoff, at which point it becomes PillarReading.missing().
    """

    def __init__(self, cfg: DPOFConfig, broker) -> None:
        self._cfg = cfg
        self._broker = broker  # duck-typed: awaitable .get_minute_bars(symbol, start_utc) -> list[dict] | None

        # UW pillars: keyed (ticker, pillar_name) -> reading / epoch of last successful fetch.
        self._pillar_cache: dict[tuple[str, str], PillarReading] = {}
        self._pillar_cache_time: dict[tuple[str, str], float] = {}
        self._pillar_last_error: dict[tuple[str, str], str] = {}
        self._gex_regime_cache: dict[str, str | None] = {}
        # Decision-CONTEXT-only NOPE / max-pain (never a gate -- see
        # UnderlyingSnapshot.nope/.max_pain docstring). Separate caches, not
        # folded into _pillar_cache/_pillar_cache_time: these aren't
        # PillarReading composite inputs, just extra context fields, and
        # keeping their own (ticker) -> value/time/error maps mirrors the
        # gex-regime side-cache pattern (self._gex_regime_cache) already used
        # here for the same "extra field alongside a pillar reading" shape.
        self._nope_cache: dict[str, float | None] = {}
        self._nope_cache_time: dict[str, float] = {}
        self._nope_last_error: dict[str, str] = {}
        self._max_pain_cache: dict[str, float | None] = {}
        self._max_pain_cache_time: dict[str, float] = {}
        self._max_pain_last_error: dict[str, str] = {}
        self._uw_last_refresh_attempt: float = 0.0

        # Self-computed flow fallback (agent.flowcollector.state.get_selfflow),
        # keyed by symbol -> the raw entry dict (or None). Refreshed once per
        # snapshot() cycle, and ONLY for underlyings where at least one UW-
        # derived pillar/NOPE/max-pain reading is not available this cycle
        # (_needs_selfflow) -- when UW is fully healthy for a symbol,
        # get_selfflow() is never called for it at all (grep-proven isolation;
        # see tests/dpof/test_data.py). Never blended with UW: UW wins
        # whenever it's available for a given pillar (see
        # _build_underlying_snapshot).
        self._selfflow_cache: dict[str, dict | None] = {}
        self._selfflow_cache_time: dict[str, float] = {}
        # One-shot-per-transition log dedup (CLAUDE.md log-level discipline:
        # log a UW<->selfflow flip once, not every ~30s cycle). None = no
        # snapshot built yet for that symbol; True/False thereafter mirrors
        # the UnderlyingSnapshot.flow_selfflow this module last returned.
        self._selfflow_active_last: dict[str, bool] = {}

        # DPOF-side daily UW call cap (separate from and inside CLIO's shared budget).
        self._uw_call_day: str = ""
        self._uw_call_count: int = 0

        # Bars / VWAP.
        self._bars_cache: dict[str, list[dict]] = {}
        self._bars_cache_time: dict[str, float] = {}
        self._bars_last_attempt: dict[str, float] = {}

        # VIX.
        self._vix_value: float | None = None
        self._vix_cache_time: float | None = None
        self._vix_last_attempt: float = 0.0

        # Headlines (Jev advisory input): a detached background refresher,
        # never awaited from snapshot() -- see _maybe_start_headline_refresh.
        self._headline_cache: list[str] = []
        self._headline_cache_time: float | None = None
        self._headline_task: asyncio.Task | None = None
        self._headline_fail_streak: int = 0

        # Dynamic pillar weights (docs/design/clio_dpof_dynamic_weights.md). None = disabled
        # (mode off, or the manager failed to build): every hook below is then a no-op.
        self._dw: DynamicWeightsManager | None = None
        try:
            self._dw = DynamicWeightsManager(cfg, cfg.data_dir)
        except Exception as e:
            logger.warning("DPOF dynamic weights disabled: manager init failed (%s: %s)", type(e).__name__, e)

    def update_config(self, cfg: DPOFConfig) -> None:
        self._cfg = cfg

    # ── Dynamic pillar weights hooks (all fail-static; no-ops when self._dw is None) ──

    def _dw_observe(self, ticker: str, pillar: str, value, now_epoch: float) -> None:
        if self._dw is None:
            return
        try:
            self._dw.observe_uw(ticker, pillar, value, now_epoch)
        except Exception as e:
            _warn_dw("observe", e)

    def _dw_observe_premium_alt(self, ticker: str, live_value, ticks, now_epoch: float) -> None:
        if self._dw is None:
            return
        try:
            self._dw.observe_premium_alt(ticker, live_value, _premium_alternatives(ticks), now_epoch)
        except Exception as e:
            _warn_dw("premium_alt", e)

    def _dw_observe_gex_alt(self, ticker: str, net_gex, now_epoch: float) -> None:
        if self._dw is None:
            return
        try:
            self._dw.observe_gex(ticker, net_gex, now_epoch)
        except Exception as e:
            _warn_dw("gex_alt", e)

    def _dw_masked(self, symbol: str, name: str, now_epoch: float) -> bool:
        if self._dw is None:
            return False
        try:
            return bool(self._dw.is_masked(symbol, name, now_epoch))
        except Exception as e:
            _warn_dw("mask", e)
            return False

    async def _dw_ensure_loaded(self) -> None:
        if self._dw is None:
            return
        try:
            await self._dw.ensure_loaded()
        except Exception as e:  # CancelledError is a BaseException and propagates unchanged
            _warn_dw("load", e)

    def _dw_finish(self, now_epoch: float, underlyings: dict) -> None:
        if self._dw is None:
            return
        try:
            self._dw.finish_snapshot(now_epoch, underlyings)
        except Exception as e:
            _warn_dw("finish", e)

    async def close(self) -> None:
        """UW (agent.integrations.uw_flow_intel) and Polygon (agent.
        integrations.polygon) each own a shared, CLIO-wide httpx.AsyncClient
        singleton, already closed by agent.integrations.scheduled_tasks.
        shutdown_background_tasks() -- this module never opens one of its own
        and must not close those here (would race that shutdown path).

        The one resource this module DOES own is the detached headline
        refresh task (see _maybe_start_headline_refresh) -- cancel it and
        wait for the cancellation to land so no task is left running after
        the engine believes the data feed is closed. This is the one place
        in this module that legitimately catches asyncio.CancelledError: WE
        are the ones who just cancelled it, so absorbing that specific,
        self-inflicted CancelledError here is correct, not a swallow of real
        external cancellation (which still propagates everywhere else).
        """
        if self._headline_task is not None and not self._headline_task.done():
            self._headline_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._headline_task
        return None

    # ── UW daily call cap bookkeeping ────────────────────────────────────

    @staticmethod
    def _uw_today_str(now_epoch: float) -> str:
        return datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(ET).strftime("%Y-%m-%d")

    def _roll_uw_day(self, now_epoch: float) -> None:
        today = self._uw_today_str(now_epoch)
        if today != self._uw_call_day:
            self._uw_call_day = today
            self._uw_call_count = 0

    def _uw_cap_reached(self, now_epoch: float) -> bool:
        self._roll_uw_day(now_epoch)
        return self._uw_call_count >= self._cfg.uw_daily_call_cap

    def _record_uw_call(self, now_epoch: float) -> None:
        self._roll_uw_day(now_epoch)
        self._uw_call_count += 1

    # ── UW: the one primitive every pillar fetch goes through ───────────

    async def _uw_fetch(self, endpoint: str, params: dict | None, now_epoch: float, what: str) -> tuple[list | None, str]:
        """Records the call against the daily cap, bounds it to
        _UW_CALL_TIMEOUT_SECONDS (uw_flow_intel._uw_get can otherwise sleep up
        to 5 min inside its own 429 backoff), and shape-checks the response.
        Never raises except a genuine asyncio.CancelledError (propagates
        untouched -- only Exception/TimeoutError are caught here).
        """
        # During the UW 401 backoff _uw_get returns {} without a request; don't
        # charge DPOF's daily UW cap for those no-ops (the cap would otherwise run
        # out mid-session and keep UW off for the rest of the day after recovery).
        try:
            from agent.integrations.uw_flow_intel import uw_auth_backoff_active
            if uw_auth_backoff_active():
                return None, f"{what}: UW auth backoff active (not attempted)"
        except Exception as e:
            logger.debug("DPOF: uw_auth_backoff_active check failed: %s: %s", type(e).__name__, e)
        self._record_uw_call(now_epoch)
        try:
            from agent.integrations.uw_flow_intel import _uw_get
            res = await asyncio.wait_for(_uw_get(endpoint, params), timeout=_UW_CALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return None, f"{what}: timeout after {_UW_CALL_TIMEOUT_SECONDS}s"
        except Exception as e:
            return None, f"{what}: {type(e).__name__}: {e}"
        return _shape_check_uw_response(res, what)

    # ── NOFA (per-ticker fetch + compute) ────────────────────────────────
    #
    # Verified production precedent: Hydrinia's UWSignalService.get_raw_
    # options_flow (hydrinia/uw_market_analyzer.py:978-995) calls
    # GET /option-trades/flow-alerts with a `ticker_symbol` param -- ONE call
    # per underlying, not a market-wide sample. Mirrored exactly here.

    async def _refresh_nofa(self, now_epoch: float) -> None:
        results = await asyncio.gather(*(
            self._uw_fetch(
                _EP_FLOW_ALERTS,
                {"limit": 200, "min_premium": 25000, "size_greater_oi": True, "ticker_symbol": t},
                now_epoch, f"nofa:{t}",
            )
            for t in self._cfg.underlyings
        ))
        from agent.integrations.uw_flow_intel import _compute_nofa
        source_tpl = f"uw:{_EP_FLOW_ALERTS}?ticker_symbol="
        for ticker, (alerts, err) in zip(self._cfg.underlyings, results):
            if alerts is None:
                logger.debug("DPOF NOFA refresh for %s failed: %s", ticker, err)
                self._pillar_last_error[(ticker, "nofa")] = err
                continue

            kept, unparsed_ts = _filter_nofa_alerts(alerts, ticker, now_epoch)
            source = source_tpl + ticker

            if not kept:
                # Zero alerts in the last _NOFA_WINDOW_SECONDS -- genuine "no
                # flow", a legitimate 0.0, never confused with a fetch failure.
                reading = PillarReading(
                    name="nofa", value=0.0, available=True, source=source,
                    detail={"empty": True, "unparsed_ts": unparsed_ts},
                )
            elif len(kept) < _NOFA_MIN_ALERTS:
                # A handful of alerts is low activity, not a directional signal --
                # still a real, computed 0.0 (not "missing"), flagged as "thin" so
                # a consumer can tell it apart from a genuinely empty window.
                reading = PillarReading(
                    name="nofa", value=0.0, available=True, source=source,
                    detail={"thin": True, "n": len(kept), "unparsed_ts": unparsed_ts},
                )
            else:
                nofa_by_ticker = _compute_nofa(kept)
                entry = nofa_by_ticker.get(ticker)
                if entry is None and len(nofa_by_ticker) == 1:
                    # Defensive fallback: _compute_nofa buckets by
                    # ticker_symbol/ticker, defaulting to "" when an alert
                    # carries neither field. Every alert here was already
                    # fetched scoped to one ticker (or matches it / has no
                    # ticker field per _filter_nofa_alerts), so a single
                    # resulting bucket IS this ticker's aggregate even if its
                    # key isn't literally `ticker`.
                    entry = next(iter(nofa_by_ticker.values()))
                if entry is None:
                    reading = PillarReading(
                        name="nofa", value=0.0, available=True, source=source,
                        detail={"empty": True, "unparsed_ts": unparsed_ts},
                    )
                else:
                    val = max(-1.0, min(1.0, float(entry.get("nofa", 0.0) or 0.0)))
                    reading = PillarReading(
                        name="nofa", value=val, available=True, source=source,
                        detail={
                            "n": entry.get("total_n"),
                            "total_prem": entry.get("total_prem"),
                            "call_prem": entry.get("call_prem"),
                            "put_prem": entry.get("put_prem"),
                            "unparsed_ts": unparsed_ts,
                        },
                    )
            self._pillar_cache[(ticker, "nofa")] = reading
            self._pillar_cache_time[(ticker, "nofa")] = now_epoch

    # ── Premium flow (shared market tide + per-ticker net-prem-ticks) ────

    async def _fetch_tide(self, now_epoch: float) -> tuple[dict | None, str]:
        """Mirrors uw_flow_intel._fetch_market_tide()'s 2 calls exactly."""
        full_r, intra_r = await asyncio.gather(
            self._uw_fetch(_EP_MARKET_TIDE, {"interval_5m": False}, now_epoch, "premium:tide_full"),
            self._uw_fetch(_EP_MARKET_TIDE, {"interval_5m": True}, now_epoch, "premium:tide_intraday"),
        )
        full, err1 = full_r
        if full is None:
            return None, err1
        intra, err2 = intra_r
        if intra is None:
            return None, err2
        return {"full_day": full, "intraday": intra}, ""

    def _handle_empty_pillar(self, ticker: str, name: str, source: str, reason: str, now_epoch: float) -> bool:
        """Flag-gated empty/drifted pillar handling. Returns True when the caller
        must `continue` WITHOUT writing _pillar_cache/_pillar_cache_time (flag on:
        the pillar is treated as a failed fetch, so the self-flow fallback can take
        over after 3x uw_refresh_seconds). Returns False when the flag is off and
        the caller keeps the legacy stored 0.0 reading. `now_epoch` is unused and
        kept only for call-site symmetry with the cache writers.
        """
        if not self._cfg.empty_pillar_unavailable:
            return False
        self._pillar_last_error[(ticker, name)] = f"empty:{reason}"
        logger.info("DPOF %s pillar for %s empty (%s): treated as unavailable (flag on)", name, ticker, reason)
        return True

    async def _refresh_premium(self, now_epoch: float) -> None:
        tide, err = await self._fetch_tide(now_epoch)
        if tide is None:
            logger.debug("DPOF premium refresh: tide failed (%s) -- all tickers left stale", err)
            for ticker in self._cfg.underlyings:
                self._pillar_last_error[(ticker, "premium")] = err
            return

        date_str = datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(ET).strftime("%Y-%m-%d")
        results = await asyncio.gather(*(
            self._uw_fetch(f"/stock/{t}/net-prem-ticks", {"date": date_str}, now_epoch, f"premium:{t}:net_prem_ticks")
            for t in self._cfg.underlyings
        ))

        from agent.integrations.uw_flow_intel import _compute_premium_signals
        for ticker, (ticks, terr) in zip(self._cfg.underlyings, results):
            if ticks is None:
                logger.debug("DPOF premium refresh for %s failed: %s", ticker, terr)
                self._pillar_last_error[(ticker, "premium")] = terr
                continue
            source = f"uw:{_EP_MARKET_TIDE}+/stock/{ticker}/net-prem-ticks"
            # Schema canary: non-empty rows carrying none of the recognized premium
            # fields would otherwise score a silent 0.0 (the Wave-1 defect class).
            drift = bool(ticks) and not any(
                isinstance(t, dict) and (set(t) & {"net_call_premium", "net_put_premium", "net_premium",
                                                   "call_premium", "put_premium"})
                for t in ticks[-20:]
            )
            if drift:
                last_dict = next((t for t in reversed(ticks[-20:]) if isinstance(t, dict)), None)
                _warn_schema_drift("premium", ticker, sorted(last_dict.keys()) if last_dict else [])
                if self._handle_empty_pillar(ticker, "premium", source, "schema_drift", now_epoch):
                    continue
            sig = _compute_premium_signals(tide, {ticker: ticks})
            entry = sig.get("tickers", {}).get(ticker)
            if entry is None:
                if self._handle_empty_pillar(ticker, "premium", source, "empty_ticks", now_epoch):
                    continue
                reading = PillarReading(name="premium", value=0.0, available=True, source=source, detail={"empty": True})
            else:
                val = max(-1.0, min(1.0, float(entry.get("score", 0.0) or 0.0)))
                reading = PillarReading(
                    name="premium", value=val, available=True, source=source,
                    detail={"net": entry.get("net"), "call": entry.get("call"), "put": entry.get("put")},
                )
            self._pillar_cache[(ticker, "premium")] = reading
            self._pillar_cache_time[(ticker, "premium")] = now_epoch
            self._dw_observe(ticker, "premium", reading.value, now_epoch)
            self._dw_observe_premium_alt(ticker, reading.value, ticks, now_epoch)

    # ── Dark pool (per-ticker only; skips the market-wide /darkpool/recent
    #    call since only the per-ticker bias score is needed here) ───────

    async def _refresh_darkpool(self, now_epoch: float) -> None:
        results = await asyncio.gather(*(
            self._uw_fetch(f"/darkpool/{t}", None, now_epoch, f"darkpool:{t}")
            for t in self._cfg.underlyings
        ))
        from agent.integrations.uw_flow_intel import _compute_dp_signals
        from agent.dpof.darkpool_nbbo import nbbo_signed_ratio
        for ticker, (prints, err) in zip(self._cfg.underlyings, results):
            if prints is None:
                logger.debug("DPOF darkpool refresh for %s failed: %s", ticker, err)
                self._pillar_last_error[(ticker, "darkpool")] = err
                continue
            try:
                nb = nbbo_signed_ratio(prints)
            except Exception as exc:
                logger.info("DPOF darkpool nbbo compute failed for %s: %s", ticker, type(exc).__name__)
                nb = None
            sig = _compute_dp_signals([], {ticker: prints})
            entry = sig.get("tickers", {}).get(ticker)
            source = f"uw:/darkpool/{ticker}"
            if self._cfg.darkpool_nbbo and nb is not None:
                val = max(-1.0, min(1.0, float(nb["ratio"])))
                reading = PillarReading(
                    name="darkpool", value=val, available=True, source=f"{source}#nbbo",
                    detail={"ratio": nb["ratio"], "classified": nb["classified"], "at_mid": nb["at_mid"],
                            "no_nbbo": nb["no_nbbo"], "gross_notional": nb["gross_notional"]},
                )
            elif self._cfg.darkpool_nbbo:
                # Flag on but under-sampled: never a fabricated NBBO value.
                if self._handle_empty_pillar(ticker, "darkpool", source, "nbbo_undersampled", now_epoch):
                    continue
                reading = PillarReading(name="darkpool", value=0.0, available=True, source=source, detail={"empty": True})
            elif entry is None:
                if self._handle_empty_pillar(ticker, "darkpool", source, "empty_prints", now_epoch):
                    continue
                reading = PillarReading(name="darkpool", value=0.0, available=True, source=source, detail={"empty": True})
            else:
                val = max(-1.0, min(1.0, float(entry.get("bias_score", 0.0) or 0.0)))
                reading = PillarReading(
                    name="darkpool", value=val, available=True, source=source,
                    detail={"notional": entry.get("notional"), "shares": entry.get("shares"),
                            "prints": entry.get("prints"), "bias": entry.get("bias")},
                )
            self._pillar_cache[(ticker, "darkpool")] = reading
            self._pillar_cache_time[(ticker, "darkpool")] = now_epoch
            self._dw_observe(ticker, "darkpool", reading.value, now_epoch)
            # Shadow readout (state of record written first): legacy vs NBBO, INFO,
            # one line per ticker per refresh. Logging failures must never break refresh.
            if not self._cfg.darkpool_nbbo and entry is not None:
                try:
                    legacy_bias = float(entry.get("bias_score", 0.0) or 0.0)
                    nb_txt = "n/a" if nb is None else f"{nb['ratio']:+.4f}"
                    logger.info(
                        "DPOF darkpool shadow %s: legacy=%+.4f nbbo=%s classified=%s at_mid=%s",
                        ticker, legacy_bias, nb_txt,
                        "n/a" if nb is None else nb["classified"],
                        "n/a" if nb is None else nb["at_mid"],
                    )
                except Exception as exc:
                    logger.debug("DPOF darkpool shadow log failed for %s: %s", ticker, type(exc).__name__)

    # ── GEX (per-ticker; also feeds UnderlyingSnapshot.gex_regime) ──────

    async def _refresh_gex(self, now_epoch: float) -> None:
        results = await asyncio.gather(*(
            self._uw_fetch(f"/stock/{t}/spot-exposures/strike", None, now_epoch, f"gex:{t}")
            for t in self._cfg.underlyings
        ))
        from agent.integrations.uw_flow_intel import _compute_gex
        for ticker, (strikes, err) in zip(self._cfg.underlyings, results):
            if strikes is None:
                logger.debug("DPOF gex refresh for %s failed: %s", ticker, err)
                self._pillar_last_error[(ticker, "gex")] = err
                continue
            gex_transition = None
            net_gex_obs = None  # shadow only: raw net_gex, set only when the strike data produced one
            source = f"uw:{_EP_GEX_STRIKE.format(ticker=ticker)}"
            if not strikes:
                # Genuine empty data list -> legitimate neutral 0.0, detail["empty"]=True.
                # No regime is known in this case (see PillarReading.detail, not
                # gex_factor's "None regime" branch -- the empty-list rule wins).
                reading = PillarReading(name="gex", value=0.0, available=True, source=source, detail={"empty": True})
                self._gex_regime_cache[ticker] = None
            else:
                sig = _compute_gex({ticker: strikes})
                entry = sig.get(ticker)
                if entry is None:
                    reading = PillarReading(name="gex", value=0.0, available=True, source=source, detail={"empty": True})
                    self._gex_regime_cache[ticker] = None
                else:
                    regime = entry.get("regime")
                    net_gex_obs = entry.get("net_gex")
                    # uw_flow_intel._compute_gex emits "ACCELERATING"/"STABILIZING"
                    # (uppercase); signals.gex_factor lowercases internally, so
                    # these map correctly without any translation here.
                    factor = gex_factor(regime, None)  # pin_risk unknown -- no CLIO primitive produces it
                    reading = PillarReading(
                        name="gex", value=factor, available=True, source=source,
                        detail={"regime": regime, "net_gex": entry.get("net_gex"), "pin_strike": entry.get("pin_strike"),
                                "pin_risk": None},
                    )
                    prev_regime = self._gex_regime_cache.get(ticker)
                    self._gex_regime_cache[ticker] = regime
                    if prev_regime != regime:
                        gex_transition = (prev_regime, regime, factor)
            self._pillar_cache[(ticker, "gex")] = reading
            self._pillar_cache_time[(ticker, "gex")] = now_epoch
            self._dw_observe_gex_alt(ticker, net_gex_obs, now_epoch)
            if gex_transition is not None:
                logger.info("DPOF gex regime %s: %s -> %s (factor %+.2f)", ticker, *gex_transition)

    # ── NOPE / max-pain (per-ticker; decision CONTEXT only -- never a gate,
    #    see UnderlyingSnapshot.nope/.max_pain and models.py's module
    #    docstring). Same cadence/cap/timeout/shape-check discipline as the
    #    4 composite pillars above via the shared _uw_fetch primitive; on
    #    failure the previous cached reading is left in place to age toward
    #    the same 3x-uw_refresh staleness cutoff _build_underlying_snapshot
    #    applies to pillars (see there) -- never overwritten with a fail-open
    #    default. ──────────────────────────────────────────────────────────

    async def _refresh_nope(self, now_epoch: float) -> None:
        results = await asyncio.gather(*(
            self._uw_fetch(_EP_NOPE_TICKER.format(ticker=t), None, now_epoch, f"nope:{t}")
            for t in self._cfg.underlyings
        ))
        for ticker, (entries, err) in zip(self._cfg.underlyings, results):
            if entries is None:
                logger.debug("DPOF nope refresh for %s failed: %s", ticker, err)
                self._nope_last_error[ticker] = err
                continue
            self._nope_cache[ticker] = _parse_latest_nope(entries)
            self._nope_cache_time[ticker] = now_epoch

    async def _refresh_max_pain(self, now_epoch: float) -> None:
        today = datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(ET).date()
        results = await asyncio.gather(*(
            self._uw_fetch(_EP_MAX_PAIN_TICKER.format(ticker=t), None, now_epoch, f"max_pain:{t}")
            for t in self._cfg.underlyings
        ))
        for ticker, (entries, err) in zip(self._cfg.underlyings, results):
            if entries is None:
                logger.debug("DPOF max_pain refresh for %s failed: %s", ticker, err)
                self._max_pain_last_error[ticker] = err
                continue
            self._max_pain_cache[ticker] = _parse_nearest_max_pain(entries, today)
            self._max_pain_cache_time[ticker] = now_epoch

    # ── Cadence orchestration for the whole UW pillar batch ──────────────

    async def _maybe_refresh_uw(self, now_epoch: float) -> list[str]:
        due = self._uw_last_refresh_attempt == 0.0 or (now_epoch - self._uw_last_refresh_attempt) >= self._cfg.uw_refresh_seconds
        if not due:
            return []
        self._uw_last_refresh_attempt = now_epoch
        if self._uw_cap_reached(now_epoch):
            logger.info("DPOF UW daily call cap (%d) reached for %s -- skipping UW refresh this cycle",
                        self._cfg.uw_daily_call_cap, self._uw_call_day)
            return ["uw_dpof_cap_reached"]
        await asyncio.gather(
            self._refresh_nofa(now_epoch),
            self._refresh_premium(now_epoch),
            self._refresh_darkpool(now_epoch),
            self._refresh_gex(now_epoch),
            self._refresh_nope(now_epoch),
            self._refresh_max_pain(now_epoch),
        )
        return []

    # ── Self-computed flow fallback (agent.flowcollector) ────────────────
    #
    # Runs AFTER _maybe_refresh_uw above has had its chance this cycle (see
    # snapshot()) so _needs_selfflow sees this cycle's actual UW outcome, not
    # a stale one from before the UW refresh ran. Only reads get_selfflow()
    # for underlyings that actually need it -- a symbol whose UW pillars/
    # NOPE/max-pain are all fresh this cycle is skipped entirely, so a
    # healthy UW connection never triggers a single flowcollector state-file
    # read (see tests/dpof/test_data.py's "UW available -> selfflow never
    # read" case).

    def _needs_selfflow(self, symbol: str, now_epoch: float) -> bool:
        max_age = 3 * self._cfg.uw_refresh_seconds
        for name in PILLAR_NAMES:
            if self._dw_masked(symbol, name, now_epoch):
                return True  # a masked (dead) UW pillar must get the self-flow substitution even though its cache is fresh
            ts = self._pillar_cache_time.get((symbol, name))
            if ts is None or (now_epoch - ts) > max_age:
                return True
        for cache_time in (self._nope_cache_time, self._max_pain_cache_time):
            ts = cache_time.get(symbol)
            if ts is None or (now_epoch - ts) > max_age:
                return True
        return False

    async def _maybe_refresh_selfflow(self, now_epoch: float) -> None:
        if not self._cfg.selfflow_fallback:
            return
        needed = [t for t in self._cfg.underlyings if self._needs_selfflow(t, now_epoch)]
        if not needed:
            return
        try:
            from agent.flowcollector.state import get_selfflow
        except Exception as e:
            logger.debug("DPOF selfflow fallback: agent.flowcollector.state unavailable: %s: %s", type(e).__name__, e)
            return
        loop = asyncio.get_running_loop()
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(
                    loop.run_in_executor(None, get_selfflow, t, self._cfg.selfflow_max_age_seconds)
                    for t in needed
                )),
                timeout=_FILE_GATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            # get_selfflow() itself is contracted "sync, never raises" -- this
            # only catches a genuine timeout or an executor-pool failure.
            # asyncio.CancelledError is a BaseException, never caught here.
            logger.debug("DPOF selfflow fallback refresh failed/timed out: %s: %s", type(e).__name__, e)
            return
        for ticker, entry in zip(needed, results):
            self._selfflow_cache[ticker] = entry if isinstance(entry, dict) else None
            self._selfflow_cache_time[ticker] = now_epoch

    def _selfflow_pillar_reading(self, symbol: str, name: str, entry: dict | None) -> PillarReading | None:
        """Maps one field of a get_selfflow() entry to a PillarReading for
        `name`, or None if the fallback isn't usable for this pillar this
        cycle (no entry at all, or the specific field is itself missing --
        MISSING IS NEVER ZERO applies here exactly as it does to the UW path
        above: a present-but-None field yields None, never a fabricated 0.0).

        Per-pillar mapping (see agent.flowcollector.aggregator's module
        docstring for the self-computed math backing each field):
          - nofa/premium/darkpool: taken from selfflow's SCALE-FREE ratio
            fields (`nofa_ratio`, `premium_ratio`, `darkpool_ratio` = signed
            classified notional / gross classified notional, in [-1, 1]) --
            the same form as UW's NOFA ((call - put) / total) and premium score
            (net / (|call| + |put|)), bullish-positive. selfflow's plain
            `nofa`/`premium`/`darkpool` fields are tanh(dollars / an
            uncalibrated $ scale) and are NOT used: nomination thresholds were
            tuned on UW ratios. Darkpool's methodology still differs (UW:
            print price vs the window median; selfflow: each dark print's side
            vs its NBBO midpoint at ingest) but both read "net buying (+) /
            selling (-)". An empty window (nothing classified) maps to 0.0,
            matching the UW path's empty-alerts reading; a missing ratio key
            (older collector) is unavailable, never a guess.
          - gex: unlike the three above, this pillar's .value is NEVER the
            raw signal (selfflow's raw net_gex is a dollar magnitude, not a
            [-1,1] score) -- signals.compute_composite expects PillarReading
            ("gex").value to already BE the gex_factor (its own docstring:
            "its .value IS the gex_factor already"), exactly mirroring what
            _refresh_gex computes for the UW path. Selfflow's gex_regime
            vocabulary is translated via _SELFFLOW_GEX_REGIME_TO_UW first
            (see that constant's docstring), then run through the SAME
            signals.gex_factor() the UW path calls.
        """
        if not isinstance(entry, dict):
            return None
        source = f"selfflow:{symbol}"
        if name in ("nofa", "premium", "darkpool"):
            ratio_key = f"{name}_ratio"
            if ratio_key not in entry:
                return None  # collector predates the ratio fields -- never fall back to the tanh value
            raw = entry.get(ratio_key)
            notes = entry.get("quality_notes") if isinstance(entry.get("quality_notes"), dict) else {}
            count_key = "dark_classified_5m" if name == "darkpool" else "option_trades_5m"
            count = notes.get(count_key)
            count_known = isinstance(count, (int, float)) and not isinstance(count, bool)
            if count_known and count <= 0:
                # Zero prints in the window: for SPY/QQQ in session that is a
                # stalled stream, not a quiet tape -> unavailable, never 0.0.
                return None
            if raw is None:
                if notes.get(f"{name}_empty") is True:
                    return PillarReading(
                        name=name, value=0.0, available=True, source=source,
                        detail={"selfflow": True, "empty": True, "quality_notes": notes},
                    )
                return None
            # Minimum-sample guard, mirroring the UW path's _NOFA_MIN_ALERTS "thin"
            # reading: a ratio of 1-2 trades is always ~+/-1.0 and would read as a
            # strong signal (e.g. right after the open or a collector restart).
            if not count_known:
                return None  # sample size unknown -> can't tell a signal from noise
            if count < self._cfg.selfflow_min_trades:
                return PillarReading(
                    name=name, value=0.0, available=True, source=source,
                    detail={"selfflow": True, "thin": True, "n": count, "quality_notes": notes},
                )
            try:
                val = max(-1.0, min(1.0, float(raw)))
            except (TypeError, ValueError):
                return None
            return PillarReading(
                name=name, value=val, available=True, source=source,
                detail={"selfflow": True, "raw": raw, "quality_notes": entry.get("quality_notes")},
            )
        if name == "gex":
            raw_regime = entry.get("gex_regime")
            if raw_regime is None:
                return None
            mapped_regime = _SELFFLOW_GEX_REGIME_TO_UW.get(str(raw_regime).strip().lower())
            if mapped_regime is None:
                return None
            factor = gex_factor(mapped_regime, None)  # pin_risk unknown, same as the UW path
            if factor is None:
                return None
            gex_detail = entry.get("gex_detail") if isinstance(entry.get("gex_detail"), dict) else {}
            return PillarReading(
                name="gex", value=factor, available=True, source=source,
                detail={
                    "selfflow": True, "regime": mapped_regime, "selfflow_regime": raw_regime,
                    "net_gex": entry.get("gex"), "pin_strike": gex_detail.get("pin_strike"), "pin_risk": None,
                },
            )
        return None

    # ── Bars / VWAP ───────────────────────────────────────────────────────

    async def _fetch_bars_raw(self, ticker: str, now_epoch: float) -> tuple[list | None, str]:
        start_utc = _session_start_utc(now_epoch)
        try:
            result = await asyncio.wait_for(self._broker.get_minute_bars(ticker, start_utc), timeout=_BARS_CALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return None, f"bars:{ticker}: timeout after {_BARS_CALL_TIMEOUT_SECONDS}s"
        except Exception as e:
            return None, f"bars:{ticker}: {type(e).__name__}: {e}"
        if result is None:
            return None, f"bars:{ticker}: broker returned None"
        if not isinstance(result, list):
            return None, f"bars:{ticker}: expected list, got {type(result).__name__}"
        return result, ""

    async def _maybe_refresh_bars(self, now_epoch: float) -> None:
        due_tickers = [
            t for t in self._cfg.underlyings
            if self._bars_last_attempt.get(t, 0.0) == 0.0
            or (now_epoch - self._bars_last_attempt[t]) >= self._cfg.bars_refresh_seconds
        ]
        if not due_tickers:
            return
        results = await asyncio.gather(*(self._fetch_bars_raw(t, now_epoch) for t in due_tickers))
        for ticker, (bars, err) in zip(due_tickers, results):
            self._bars_last_attempt[ticker] = now_epoch
            if bars is None:
                logger.debug("DPOF bars refresh for %s failed: %s", ticker, err)
                continue
            self._bars_cache[ticker] = bars
            self._bars_cache_time[ticker] = now_epoch

    # ── VIX (cadence shares bars_refresh_seconds: it's price data, same as
    #    bars, and comfortably under vix_max_age_seconds=900s at any sane
    #    bars_refresh_seconds default) ────────────────────────────────────

    async def _fetch_vix_raw(self) -> tuple[float | None, str]:
        try:
            from agent.integrations.polygon import get_snapshot_raw
            res = await asyncio.wait_for(get_snapshot_raw("VIX"), timeout=_VIX_CALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return None, f"vix: timeout after {_VIX_CALL_TIMEOUT_SECONDS}s"
        except Exception as e:
            return None, f"vix: {type(e).__name__}: {e}"
        if not isinstance(res, dict):
            return None, f"vix: expected dict, got {type(res).__name__}"
        if not res:
            return None, "vix: empty snapshot (polygon/yahoo fetch failed)"
        price = res.get("last_price")
        if price is None:
            return None, "vix: no last_price in snapshot"
        try:
            price = float(price)
        except (TypeError, ValueError):
            return None, f"vix: non-numeric last_price {price!r}"
        return price, ""

    async def _maybe_refresh_vix(self, now_epoch: float) -> None:
        due = self._vix_last_attempt == 0.0 or (now_epoch - self._vix_last_attempt) >= self._cfg.bars_refresh_seconds
        if not due:
            return
        self._vix_last_attempt = now_epoch
        price, err = await self._fetch_vix_raw()
        if price is None:
            logger.debug("DPOF VIX refresh failed: %s", err)
            return
        self._vix_value = price
        self._vix_cache_time = now_epoch

    # ── Gates ─────────────────────────────────────────────────────────────

    async def _check_econ_gate(self, now_epoch: float) -> tuple[list[dict], bool]:
        """Returns (importance==3 events within +/- event_context_window_minutes,
        source_ok). source_ok False is a REQUIRED-gate-SOURCE-availability
        signal used only for the `gate_source_unavailable:econ_calendar`
        health_issues note and `calendar_status` (get_intel_ctx) -- Task 1
        (2026-09-25): it no longer forces data_healthy False (see snapshot()
        below) and the events it returns no longer block anything, they are
        decision CONTEXT (`risk_flags`/ctx["events"]) only. See
        _econ_calendar_read_sync's docstring for why file-presence is the
        distinguishable failure signal.
        """
        loop = asyncio.get_running_loop()
        try:
            present, events = await asyncio.wait_for(
                loop.run_in_executor(None, _econ_calendar_read_sync), timeout=_FILE_GATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.debug("DPOF econ calendar gate read failed: %s: %s", type(e).__name__, e)
            return [], False
        if not present:
            return [], False

        now_et = datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(ET)
        window_s = self._cfg.event_context_window_minutes * 60.0
        soon = []
        for ev in events:
            if not isinstance(ev, dict) or ev.get("importance") != 3:
                continue
            dt_str = ev.get("datetime_et")
            if not dt_str:
                continue
            try:
                ev_dt = datetime.fromisoformat(dt_str)
            except (ValueError, TypeError):
                continue
            if ev_dt.tzinfo is None:
                ev_dt = ev_dt.replace(tzinfo=ET)
            if abs((ev_dt - now_et).total_seconds()) <= window_s:
                soon.append(ev)
        return soon, True

    async def _check_breaking_news(self, now_epoch: float) -> tuple[bool, bool]:
        """Returns (breaking_news_active, source_ok). source_ok False is a
        REQUIRED-gate-source failure (forces data_healthy False).
        """
        loop = asyncio.get_running_loop()
        try:
            present, state = await asyncio.wait_for(
                loop.run_in_executor(None, _learning_state_read_sync), timeout=_FILE_GATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.debug("DPOF learning_state gate read failed: %s: %s", type(e).__name__, e)
            return False, False
        if not present:
            return False, False
        breaking = state.get("breaking_news", {})
        if not isinstance(breaking, dict):
            breaking = {}
        return bool(breaking.get("active")), True

    async def _check_vol_regime(self, now_epoch: float) -> tuple[str | None, str | None]:
        """Returns (risk_flag_or_None, health_note_or_None). This is NOT a
        required gate source -- a failure/staleness here never forces
        data_healthy False, it only adds an informational health_issues note
        (per BUILD_SPEC: "stale -> ignored with health note, not a flag").

        Task 1 (2026-09-25): `block_on_vol_regimes` is retired -- there is no
        allow/block-list any more, ANY fresh vol regime reading surfaces as an
        informational risk_flag (decision context, never a veto: see
        evaluate_signal()/signals.py).
        """
        loop = asyncio.get_running_loop()
        try:
            fresh, regime = await asyncio.wait_for(
                loop.run_in_executor(None, _vol_regime_read_sync), timeout=_FILE_GATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.debug("DPOF vol regime gate read failed: %s: %s", type(e).__name__, e)
            return None, "vol_regime_unavailable"
        if not fresh:
            return None, "vol_regime_stale"
        reg = regime.get("regime")
        if reg:
            return f"vol_regime:{reg}", None
        return None, None

    # ── Advisory intel for the decider (Task 2, INTEL_V3_SPEC.md) ────────
    #
    # get_intel_ctx() is a SEPARATE async method from snapshot() -- it is
    # awaited once per cycle by engine.py (never inside snapshot()'s own
    # budget-constrained gather), so a slow/broken/not-yet-landed E/V/N
    # sibling module can never affect MarketSnapshot construction. Every
    # sub-fetch below degrades to an empty/None shape on ANY failure
    # (ImportError because the sibling module hasn't landed yet, a contract
    # violation, a timeout, ...) and NEVER raises -- this is advisory context
    # for the decision model, never a gate.

    async def _fallback_upcoming_events(self, now_epoch: float, hours: float) -> list[dict]:
        """Fallback for econ_surprise.get_upcoming_releases() (Builder E) when
        that module is unavailable -- reuses the econ calendar cache DPOF
        already reads for _check_econ_gate, normalized to E's contract shape
        (event, release_time_et, importance, consensus, previous,
        minutes_until). Unlike _check_econ_gate (importance==3 only, +/- a
        symmetric window), this includes EVERY upcoming (future-only) release
        in the window so the model can weigh lower-importance events too.
        """
        loop = asyncio.get_running_loop()
        try:
            present, events = await asyncio.wait_for(
                loop.run_in_executor(None, _econ_calendar_read_sync), timeout=_FILE_GATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.debug("DPOF intel: econ calendar fallback (upcoming events) read failed: %s: %s", type(e).__name__, e)
            return []
        if not present:
            return []
        now_et = datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(ET)
        window_minutes = max(0.0, hours) * 60.0
        out: list[dict] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            dt_str = ev.get("datetime_et")
            if not dt_str:
                continue
            try:
                ev_dt = datetime.fromisoformat(dt_str)
            except (ValueError, TypeError):
                continue
            if ev_dt.tzinfo is None:
                ev_dt = ev_dt.replace(tzinfo=ET)
            minutes_until = (ev_dt - now_et).total_seconds() / 60.0
            if not (0.0 <= minutes_until <= window_minutes):
                continue
            out.append({
                "event": ev.get("name") or "unknown_event",
                "release_time_et": ev_dt.isoformat(),
                "importance": ev.get("importance"),
                "consensus": ev.get("forecast") or None,
                "previous": ev.get("previous") or None,
                "minutes_until": round(minutes_until, 1),
            })
        out.sort(key=lambda e: e["minutes_until"])
        return out

    async def _fetch_events(self, now_epoch: float) -> dict:
        """{"upcoming": [...], "recent_surprises": [...], "source": ...}.
        `upcoming` from Builder E's econ_surprise.get_upcoming_releases(6.0)
        when importable, else the econ-calendar fallback above.
        `recent_surprises` from econ_surprise.get_recent_surprises(240) --
        always [] when E's module is absent (DPOF has no native surprise-z
        source to fall back to for that half). Contract: both readers are
        SYNC, mtime-cached, never raise (INTEL_V3_SPEC.md) -- still run via
        run_in_executor (never blocking file I/O on the event loop) and
        wrapped in try/except as defense in depth against a contract
        violation in an in-progress sibling module.
        """
        loop = asyncio.get_running_loop()
        source = "econ_surprise"
        upcoming: list = []
        recent_surprises: list = []
        try:
            from agent.integrations import econ_surprise
            upcoming = await loop.run_in_executor(None, econ_surprise.get_upcoming_releases, 6.0)
            recent_surprises = await loop.run_in_executor(None, econ_surprise.get_recent_surprises, 240)
            if not isinstance(upcoming, list):
                upcoming = []
            if not isinstance(recent_surprises, list):
                recent_surprises = []
        except Exception as e:
            logger.debug(
                "DPOF intel: econ_surprise unavailable (%s: %s); falling back to econ calendar for "
                "upcoming events, no recent_surprises source", type(e).__name__, e,
            )
            source = "econ_calendar_fallback"
            upcoming = await self._fallback_upcoming_events(now_epoch, 6.0)
            recent_surprises = []
        return {"upcoming": upcoming, "recent_surprises": recent_surprises, "source": source}

    async def _fetch_vol_surface(self) -> dict:
        """{SYM: vol_surface.get_vol_surface(SYM) or None} for every
        configured underlying (Builder V). None per-symbol, or for the whole
        dict when V's module isn't importable yet -- never a fabricated
        reading.
        """
        out: dict[str, dict | None] = {sym: None for sym in self._cfg.underlyings}
        try:
            from agent.integrations import vol_surface
        except Exception as e:
            logger.debug("DPOF intel: vol_surface unavailable: %s: %s", type(e).__name__, e)
            return out
        loop = asyncio.get_running_loop()
        for sym in self._cfg.underlyings:
            try:
                val = await loop.run_in_executor(None, vol_surface.get_vol_surface, sym)
            except Exception as e:
                logger.debug("DPOF intel: vol_surface.get_vol_surface(%s) failed: %s: %s", sym, type(e).__name__, e)
                val = None
            out[sym] = val if isinstance(val, dict) else None
        return out

    async def _attach_kronos_pillar(self, underlyings: dict, now_epoch: float) -> None:
        """Kronos LIVE pillar (docs/design/clio_kronos.md §4b): adds a "kronos" PillarReading to each
        underlying that has a fresh forecast. Off, stale, missing, slow or broken -> nothing is added and
        the composite is exactly what the flow pillars alone give. Never raises, never blocks longer
        than _FILE_GATE_TIMEOUT_SECONDS. Kronos is deliberately NOT in PILLAR_NAMES (the flow pillars:
        UW caches, the self-flow fallback trigger and data_healthy all iterate that tuple)."""
        if self._cfg.kronos_pillar != "live":
            return
        try:
            loop = asyncio.get_running_loop()
            readings = await asyncio.wait_for(
                loop.run_in_executor(
                    None, _kronos_pillar_read_sync, tuple(underlyings), self._cfg.kronos_max_age_seconds,
                    self._cfg.kronos_horizon, now_epoch,
                ),
                timeout=_FILE_GATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.debug(f"DPOF data: kronos pillar read failed: {type(e).__name__}: {e}")
            return
        for sym, reading in readings.items():
            u = underlyings.get(sym)
            if u is not None:
                u.pillars[KRONOS_PILLAR] = reading

    async def _fetch_kronos(self, now_epoch: float) -> Optional[dict]:
        """Kronos advisory context (Task K2). None when CLIO_DPOF_KRONOS_
        CONTEXT is off (get_intel_ctx then omits the "kronos" key entirely --
        the decider ctx must not carry a "disabled" marker, per docs/design/
        clio_kronos.md §4). Otherwise {symbol: "unavailable" | {...}} -- the
        file read runs in an executor (never blocking I/O on the loop).
        `now_epoch` is threaded through to state.read_forecast's own
        freshness gate so this respects the SAME simulated clock every other
        _fetch_* method in this class uses (real time.time() in production,
        a fixed test clock under asyncio.run(..., now_epoch=...))."""
        if not self._cfg.kronos_context:
            return None
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, _kronos_ctx_read_sync, tuple(self._cfg.underlyings), self._cfg.kronos_max_age_seconds, now_epoch,
            )
        except Exception as e:
            logger.debug(f"DPOF intel: kronos ctx read failed: {type(e).__name__}: {e}")
            return {sym: "unavailable" for sym in self._cfg.underlyings}

    async def _fetch_breaking(self, now_epoch: float) -> dict:
        """newsflash_lite.get_breaking_state(900) (Builder N) when importable
        and shaped right, else the existing Lighthouse/learning_state read
        this module already performs for `_check_breaking_news`, normalized
        to N's contract shape via _normalize_lighthouse_breaking().
        """
        try:
            from agent.integrations import newsflash_lite
            loop = asyncio.get_running_loop()
            state = await loop.run_in_executor(None, newsflash_lite.get_breaking_state, 900)
            if isinstance(state, dict):
                return _apply_newsflash_link_mode(state)
            logger.debug("DPOF intel: newsflash_lite.get_breaking_state returned non-dict; falling back")
        except Exception as e:
            logger.debug(
                "DPOF intel: newsflash_lite unavailable (%s: %s); falling back to Lighthouse/learning_state",
                type(e).__name__, e,
            )
        loop = asyncio.get_running_loop()
        try:
            present, state = await asyncio.wait_for(
                loop.run_in_executor(None, _learning_state_read_sync), timeout=_FILE_GATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.debug("DPOF intel: breaking-news fallback read failed: %s: %s", type(e).__name__, e)
            return _normalize_lighthouse_breaking({}, False)
        breaking = state.get("breaking_news", {}) if present else {}
        if not isinstance(breaking, dict):
            breaking = {}
        return _normalize_lighthouse_breaking(breaking, present)

    async def _fetch_calendar_status(self, now_epoch: float) -> dict:
        """{"status": "ok"|"stale"|"missing", "age_hours": float|None,
        "sources": [...]}. Task 1: this is the econ calendar's ADVISORY
        freshness now (never a data_healthy input -- see snapshot() above)."""
        loop = asyncio.get_running_loop()
        try:
            present, age_hours = await asyncio.wait_for(
                loop.run_in_executor(None, _econ_calendar_status_sync), timeout=_FILE_GATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.debug("DPOF intel: calendar_status read failed: %s: %s", type(e).__name__, e)
            return {"status": "missing", "age_hours": None, "sources": ["econ_calendar"]}
        if not present:
            return {"status": "missing", "age_hours": None, "sources": ["econ_calendar"]}
        max_age_hours = _ECON_CACHE_MAX_AGE_SECONDS / 3600.0
        status = "ok" if age_hours <= max_age_hours else "stale"
        return {"status": status, "age_hours": round(age_hours, 2), "sources": ["econ_calendar"]}

    async def get_intel_ctx(self, now_epoch: float | None = None) -> dict:
        """Best-effort ADVISORY intel for the decider (Task 2, G prompt /
        INTEL_V3_SPEC.md): upcoming econ releases + recent surprises (Builder
        E), per-underlying vol surface (Builder V), breaking-news state
        (Builder N, falling back to Lighthouse/learning_state), the econ
        calendar's own freshness as `calendar_status`, and (Task K2, agent/
        kronos) a per-underlying Kronos forecast/track-record block under
        "kronos" -- omitted entirely when CLIO_DPOF_KRONOS_CONTEXT is off (see
        _fetch_kronos). NEVER raises and NEVER blocks a cycle -- this is
        context for the decision model, not a gate; every sub-fetch above
        already degrades to an empty/None shape internally, and the whole
        call is additionally bounded by an outer timeout as defense in depth
        (mirrors snapshot()'s _SNAPSHOT_BUDGET_SECONDS pattern). The VIX
        regime/label is NOT computed here -- that is pure (no I/O) and lives
        in engine.py alongside signals.classify_regime.
        """
        now_epoch = time.time() if now_epoch is None else now_epoch
        try:
            events, vol_surface_ctx, breaking, calendar_status, kronos = await asyncio.wait_for(
                asyncio.gather(
                    self._fetch_events(now_epoch),
                    self._fetch_vol_surface(),
                    self._fetch_breaking(now_epoch),
                    self._fetch_calendar_status(now_epoch),
                    self._fetch_kronos(now_epoch),
                ),
                timeout=_INTEL_BUDGET_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "DPOF intel ctx build exceeded the %.0fs budget -- serving an empty intel ctx this cycle",
                _INTEL_BUDGET_SECONDS,
            )
            out = {
                "events": {"upcoming": [], "recent_surprises": [], "source": "timeout"},
                "vol_surface": {sym: None for sym in self._cfg.underlyings},
                "breaking": {"active": False, "max_severity": 0, "events": [], "as_of": None, "stale": True, "source": "timeout"},
                "calendar_status": {"status": "missing", "age_hours": None, "sources": ["econ_calendar"]},
            }
            if self._cfg.kronos_context:
                out["kronos"] = {sym: "unavailable" for sym in self._cfg.underlyings}
            return out
        out = {"events": events, "vol_surface": vol_surface_ctx, "breaking": breaking, "calendar_status": calendar_status}
        if kronos is not None:
            out["kronos"] = kronos
        return out

    # ── Headlines: detached background refresher ─────────────────────────
    #
    # agent.integrations.global_news_feeds.scan_current_window() is the only
    # headline source in CLIO. It is fully SYNCHRONOUS and performs LIVE
    # network RSS fetches across every feed in the active scan window on
    # every call -- there is no cache-only "give me what's already been
    # scanned" accessor. That is exactly the kind of blocking, unbounded-
    # latency call this module must never let stall a snapshot(): it is run
    # off the event loop via run_in_executor, bounded by
    # _HEADLINE_SCAN_TIMEOUT_SECONDS, and kicked off as a detached
    # asyncio.Task that snapshot() never awaits -- snapshot() only ever reads
    # whatever the LAST completed refresh left in the cache (or [] if that
    # cache is empty or older than _HEADLINE_MAX_AGE_SECONDS). A failed or
    # slow scan is never a health issue: Jev's headline questions are purely
    # advisory (see docs/design/clio_dpof.md §8) and jev.py already handles
    # an empty headlines list gracefully. Strings here are RAW, untrusted,
    # NOT sanitized -- jev.py is the single sanitization point (it runs
    # sanitize_ingested_content on each headline itself); sanitizing here too
    # would double-wrap the <ingested_data> envelope.

    def _maybe_start_headline_refresh(self, now_epoch: float) -> None:
        if self._cfg.jev_mode == JevMode.OFF:
            return  # Jev disabled -- no reason to ever hit the network for headlines
        if self._headline_task is not None and not self._headline_task.done():
            return  # one in flight at a time
        stale = (
            self._headline_cache_time is None
            or (now_epoch - self._headline_cache_time) >= _HEADLINE_REFRESH_SECONDS
        )
        if not stale:
            return
        self._headline_task = asyncio.create_task(self._refresh_headlines(now_epoch))

    async def _refresh_headlines(self, now_epoch: float) -> None:
        loop = asyncio.get_running_loop()
        try:
            from agent.integrations import global_news_feeds

            def _scan_then_read_buffer():
                # scan_current_window() CONSUMES headlines process-wide (each is
                # returned to one caller only). DPOF drives the non-consuming
                # refresh_recent_headlines() instead -- leaving learning_engine as
                # the scan's only consumer -- and reads the shared buffer.
                refresh = getattr(global_news_feeds, "refresh_recent_headlines", None)
                reader = getattr(global_news_feeds, "get_recent_headlines", None)
                if refresh is None or reader is None:
                    return global_news_feeds.scan_current_window()
                refresh()
                buffered = reader(int(_HEADLINE_LOOKBACK_SECONDS))
                return buffered if isinstance(buffered, list) else []

            raw_headlines = await asyncio.wait_for(
                loop.run_in_executor(None, _scan_then_read_buffer),
                timeout=_HEADLINE_SCAN_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._headline_fail_streak += 1
            if self._headline_fail_streak == 1 or self._headline_fail_streak % 10 == 0:
                logger.info(
                    "DPOF headline refresh failed (streak=%d, never a health issue -- advisory only): %s: %s",
                    self._headline_fail_streak, type(e).__name__, e,
                )
            return
        self._headline_fail_streak = 0
        self._headline_cache = self._process_headlines(raw_headlines, now_epoch)
        self._headline_cache_time = now_epoch

    def _process_headlines(self, raw_headlines, now_epoch: float) -> list[str]:
        """Filter to the last _HEADLINE_LOOKBACK_SECONDS, prefer T1_VERIFIED
        then T2 (ties broken newest-first), cap at cfg.jev_max_headlines.

        Returns RAW "<source_name>: <title>" strings, NOT sanitized here --
        jev.py (per BUILD_SPEC) is the single sanitization point: it runs
        sanitize_ingested_content() on each headline itself before sending it
        anywhere. Sanitizing here too would double-wrap the <ingested_data>
        envelope.
        """
        if not isinstance(raw_headlines, list):
            return []
        cutoff = datetime.fromtimestamp(now_epoch, tz=timezone.utc) - timedelta(seconds=_HEADLINE_LOOKBACK_SECONDS)
        recent = []
        for hl in raw_headlines:
            pub = getattr(hl, "published_utc", None)
            if not isinstance(pub, datetime):
                continue
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub < cutoff:
                continue
            recent.append((hl, pub))

        tier_rank = {"T1_VERIFIED": 0, "T2_SEMI_INDEPENDENT": 1, "T3_STATE_AFFILIATED": 2}
        recent.sort(key=lambda pair: (
            tier_rank.get(getattr(pair[0], "reliability_tier", ""), 9),
            -pair[1].timestamp(),
        ))

        out: list[str] = []
        for hl, _pub in recent[: self._cfg.jev_max_headlines]:
            source_name = getattr(hl, "source_name", "") or ""
            title = getattr(hl, "title", "") or ""
            text = f"{source_name}: {title}"
            if len(text) > 200:
                text = text[:200]
            out.append(text)
        return out

    def _get_headlines(self, now_epoch: float) -> list[str]:
        if self._headline_cache_time is None:
            return []
        if (now_epoch - self._headline_cache_time) > _HEADLINE_MAX_AGE_SECONDS:
            return []
        return list(self._headline_cache)

    # ── Snapshot assembly ────────────────────────────────────────────────

    def _composite_computable(self, u: UnderlyingSnapshot | None) -> bool:
        """Light mirror of signals.compute_composite's availability gate,
        used ONLY to decide data_healthy here -- signals.py (built separately)
        owns the authoritative composite math; this only answers "is there
        enough live data to even attempt it".
        """
        if u is None:
            return False
        available = [name for name in PILLAR_NAMES if u.pillars.get(name) and u.pillars[name].available]
        if "nofa" not in available:
            return False
        return len(available) >= self._cfg.min_pillars

    def _build_underlying_snapshot(self, symbol: str, now_epoch: float) -> UnderlyingSnapshot:
        pillars: dict[str, PillarReading] = {}
        max_age = 3 * self._cfg.uw_refresh_seconds
        selfflow_entry = self._selfflow_cache.get(symbol) if self._cfg.selfflow_fallback else None
        any_selfflow = False
        for name in PILLAR_NAMES:
            cached = self._pillar_cache.get((symbol, name))
            ts = self._pillar_cache_time.get((symbol, name))
            masked = self._dw_masked(symbol, name, now_epoch)
            if cached is not None and ts is not None and (now_epoch - ts) <= max_age and not masked:
                age = now_epoch - ts
                pillars[name] = replace(cached, age_seconds=age, available=True)
                continue

            # UW missing/stale for this pillar this cycle -- try the selfflow
            # fallback before giving up on it entirely. UW always wins above
            # (this branch is only reached when it did NOT); the two sources
            # are never blended for the same pillar.
            fallback = self._selfflow_pillar_reading(symbol, name, selfflow_entry) if self._cfg.selfflow_fallback else None
            if fallback is not None:
                pillars[name] = fallback
                any_selfflow = True
                if name == "gex":
                    # Mirror _refresh_gex's own "write the regime alongside
                    # the pillar value" pattern (using the SAME translated
                    # vocabulary gex_factor() was just called with) so any
                    # downstream reader of gex_regime (signals.
                    # compute_composite's own-regime fallback branch, jev.py,
                    # engine.py's underlyings ctx) sees a value consistent
                    # with this selfflow-derived pillar.
                    self._gex_regime_cache[symbol] = fallback.detail.get("regime")
            else:
                if masked:
                    error = "dead: liveness gate (dynamic weights)"
                else:
                    error = (
                        f"stale: age={(now_epoch - ts):.0f}s > {max_age}s" if (cached is not None and ts is not None)
                        else self._pillar_last_error.get((symbol, name), "never fetched")
                    )
                pillars[name] = PillarReading.missing(name, error=error, source=(cached.source if cached is not None else ""))

        bars = self._bars_cache.get(symbol, [])
        vwap_info = compute_vwap_and_returns(bars, now_epoch)

        # Same staleness discipline as the pillars above: a reading older than
        # 3x uw_refresh_seconds is treated as unavailable (None), never served
        # indefinitely stale -- never a fail-open default either (no 0.0
        # stand-in exists for NOPE/max-pain; both genuinely have none). The
        # selfflow fallback (Task F, 2026-09-25) only ever fills a value that
        # is STILL None after the UW attempt above -- UW wins whenever it
        # produced a real reading, including a genuine 0.0.
        nope_val = None
        nope_ts = self._nope_cache_time.get(symbol)
        if nope_ts is not None and (now_epoch - nope_ts) <= max_age:
            nope_val = self._nope_cache.get(symbol)
        if nope_val is None and selfflow_entry is not None:
            raw_nope = selfflow_entry.get("nope")
            if isinstance(raw_nope, (int, float)) and not isinstance(raw_nope, bool):
                nope_val = float(raw_nope)
                any_selfflow = True

        max_pain_val = None
        max_pain_ts = self._max_pain_cache_time.get(symbol)
        if max_pain_ts is not None and (now_epoch - max_pain_ts) <= max_age:
            max_pain_val = self._max_pain_cache.get(symbol)
        if max_pain_val is None and selfflow_entry is not None:
            raw_max_pain = selfflow_entry.get("max_pain")
            if isinstance(raw_max_pain, (int, float)) and not isinstance(raw_max_pain, bool):
                max_pain_val = float(raw_max_pain)
                any_selfflow = True

        self._log_selfflow_transition(symbol, any_selfflow)

        return UnderlyingSnapshot(
            symbol=symbol,
            pillars=pillars,
            gex_regime=self._gex_regime_cache.get(symbol),
            pin_risk=None,  # never computed -- see final report; no primitive in CLIO produces this
            last_price=vwap_info["last_price"],
            vwap=vwap_info["vwap"],
            ret_5m=vwap_info["ret_5m"],
            ret_15m=vwap_info["ret_15m"],
            bars_count=vwap_info["bars_count"],
            price_age_seconds=vwap_info["price_age_seconds"],
            nope=nope_val,
            max_pain=max_pain_val,
            flow_selfflow=any_selfflow,
        )

    def _log_selfflow_transition(self, symbol: str, active_now: bool) -> None:
        """INFO once per UW<->selfflow transition per underlying, per process
        -- never every cycle (CLAUDE.md log-level discipline: UW being down
        for hours would otherwise spam the journal every ~cycle_seconds).
        The very first snapshot built for a symbol never logs (prev is None)
        -- there is no "transition" from nothing."""
        prev = self._selfflow_active_last.get(symbol)
        self._selfflow_active_last[symbol] = active_now
        if prev is None or prev == active_now:
            return
        if active_now:
            logger.info("DPOF %s: flow source switched to self-computed fallback (UW unavailable/stale)", symbol)
        else:
            logger.info("DPOF %s: flow source switched back to UW", symbol)

    async def snapshot(self, now_epoch: float | None = None) -> MarketSnapshot:
        now_epoch = time.time() if now_epoch is None else now_epoch
        await self._dw_ensure_loaded()
        health_issues: list[str] = []
        risk_flags: list[str] = []

        # Fire-and-forget: never awaited, never blocks the cycle below.
        self._maybe_start_headline_refresh(now_epoch)

        try:
            uw_issues, _bars_done, _vix_done, econ_result, learning_result, vol_result = await asyncio.wait_for(
                asyncio.gather(
                    self._maybe_refresh_uw(now_epoch),
                    self._maybe_refresh_bars(now_epoch),
                    self._maybe_refresh_vix(now_epoch),
                    self._check_econ_gate(now_epoch),
                    self._check_breaking_news(now_epoch),
                    self._check_vol_regime(now_epoch),
                ),
                timeout=_SNAPSHOT_BUDGET_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "DPOF snapshot refresh exceeded the %.0fs budget -- serving cached/stale data this cycle",
                _SNAPSHOT_BUDGET_SECONDS,
            )
            health_issues.append("snapshot_refresh_timeout")
            uw_issues = []
            econ_result = ([], False)
            learning_result = (False, False)
            vol_result = (None, "vol_regime_unavailable")

        health_issues.extend(uw_issues)

        econ_events_soon, econ_ok = econ_result
        breaking_active, learning_ok = learning_result
        vol_flag, vol_note = vol_result

        if not econ_ok:
            health_issues.append("gate_source_unavailable:econ_calendar")
        if not learning_ok:
            health_issues.append("gate_source_unavailable:learning_state")
        if vol_note:
            health_issues.append(vol_note)

        for ev in econ_events_soon:
            risk_flags.append(f"econ_blackout:{ev.get('name') or 'unknown_event'}")
        # Task 1 (2026-09-25): `block_on_breaking_news` is retired -- an
        # active breaking-news state always surfaces as an informational
        # risk_flag now (decision context, never a veto).
        if breaking_active:
            risk_flags.append("breaking_news")
        if vol_flag:
            risk_flags.append(vol_flag)

        # Self-computed flow fallback (Task F, 2026-09-25): runs AFTER the UW
        # refresh above so _needs_selfflow sees THIS cycle's actual UW
        # outcome. Own internal timeout/exception handling (never raises,
        # never blocks longer than _FILE_GATE_TIMEOUT_SECONDS) -- a slow/
        # broken flowcollector state file can never stall a snapshot.
        await self._maybe_refresh_selfflow(now_epoch)

        underlyings = {sym: self._build_underlying_snapshot(sym, now_epoch) for sym in self._cfg.underlyings}
        await self._attach_kronos_pillar(underlyings, now_epoch)
        self._dw_finish(now_epoch, underlyings)

        primary = self._cfg.primary
        primary_u = underlyings.get(primary)
        primary_composite_ok = self._composite_computable(primary_u)
        primary_price_ok = (
            primary_u is not None
            and primary_u.price_age_seconds is not None
            and primary_u.price_age_seconds <= 3 * self._cfg.bars_refresh_seconds
        )
        if not primary_composite_ok:
            health_issues.append("primary_composite_unavailable")
        if not primary_price_ok:
            health_issues.append("primary_price_stale_or_missing")

        # Task 1 (2026-09-25, operator directive): econ-calendar/learning-state
        # GATE-SOURCE availability (econ_ok/learning_ok) no longer forces
        # data_healthy False -- data_healthy is strictly about the flow
        # pillars/price data DPOF actually trades on (missing is never zero
        # there, unchanged: primary_composite_ok/primary_price_ok above).
        # econ_ok/learning_ok are still recorded into health_issues above
        # (diagnostic only) and econ_ok/its staleness now separately drive
        # `calendar_status` in get_intel_ctx() below, shown to the decider as
        # context rather than gating anything.
        data_healthy = bool(primary_composite_ok and primary_price_ok)

        return MarketSnapshot(
            asof=now_epoch,
            underlyings=underlyings,
            vix=self._vix_value,
            vix_age_seconds=(now_epoch - self._vix_cache_time) if self._vix_cache_time is not None else None,
            risk_flags=risk_flags,
            headlines=self._get_headlines(now_epoch),
            econ_events_soon=econ_events_soon,
            data_healthy=data_healthy,
            health_issues=health_issues,
        )
