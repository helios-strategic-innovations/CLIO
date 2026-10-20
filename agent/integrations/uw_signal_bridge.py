"""
UW Signal Bridge v2.0 — CLIO's single point of contact for Hydrinia-computed UW signals.

Reads Hydrinia's exported snapshot.json (15 sections post-Trident v2.0) from the
bridge directory, validates freshness, applies reasonableness filters, and provides
a clean Python API for CLIO consumers.

Also makes supplementary UW API calls using CLIO's own 10K/day budget for
deep-dive intelligence (fundamentals, earnings, technicals, shorts, politicians).

CC-CLIO-UW-1 + CC-CLIO-UW-7 | Atlas Strategies — March 2026
Supersedes: CC-MB-UW-1 (v1.0, 423 LOC)
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, date
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("openclaw.uw_signal_bridge")

_NY = ZoneInfo("America/New_York")

# ============================================================
# Feature Flags — CLIO_UW_* namespace (independent of Hydrinia)
# ============================================================

# ============================================================
# Paths & Constants
# ============================================================

_SNAPSHOT_DIR = "/home/agent/tradingbot/moltbot_knowledge/uw_signals"
_SNAPSHOT_FILE = os.path.join(_SNAPSHOT_DIR, "snapshot.json")
_DEFAULT_TTL_SECONDS = 600  # 10 minutes
_OFFHOURS_TTL_MULTIPLIER = 6  # Allow 60-min staleness off-hours
# UW-7 (S211): the snapshot CLIO builds from its own uw_flow_intel file is refreshed by a 15-minute job
# (file `created` observed 12:49:52 vs the last logged scan 12:35:13), so a 600 s TTL made it "stale"
# for a third of every market-hours cycle -- 130 WARNINGs/day, and DPOF skipped the vol-regime context
# while stale. Job cadence 900 s + scan time + margin. The Hydrinia-native snapshot.json keeps its own
# (unchanged) freshness_ttl_seconds contract.
_FALLBACK_TTL_SECONDS = 1200

# Regime levels for reasonableness filter
_REGIME_LEVELS = {"CALM": 0, "ELEVATED": 1, "HIGH": 2, "CRISIS": 3}
_MAX_REGIME_JUMP = 1  # Max allowed single-cycle regime jump

# Reasonableness bounds for Z-scored and continuous fields
_REASONABLENESS_BOUNDS = {
    "oracle.iv_rv_spread_z": (-5.0, 5.0),
    "oracle.skew_z": (-5.0, 5.0),
    "oracle.vol_of_vol": (0.0, 200.0),
    "oracle.iv_rank": (0.0, 100.0),
    "regime_enhanced.pc_ratio_5d": (0.1, 5.0),
}

# Cache TTLs for CLIO supplementary API calls
_CLIO_CACHE_TTLS = {
    "fundamentals": 86400,          # 24h
    "earnings_detail": 86400,       # 24h
    "politician_portfolio": 604800, # 7 days
    "ticker_technicals": 7200,      # 2h
    "short_interest": 14400,        # 4h
    "fda_calendar": 86400,          # 24h
    "insider_detail": 86400,        # 24h
    "cash_flows": 604800,           # 7 days
    "balance_sheets": 604800,       # 7 days
}

# ============================================================
# Utilities
# ============================================================

def _safe_float(val, default=0.0):
    """Parse UW string-number fields safely. Nearly ALL UW numeric fields are strings."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _clamp(value: float, bounds_key: str) -> float:
    """Clamp value within reasonableness bounds. Clamps to [lo, hi] if out of range."""
    bounds = _REASONABLENESS_BOUNDS.get(bounds_key)
    if bounds is None:
        return value
    lo, hi = bounds
    if value < lo or value > hi:
        logger.warning("CLIO-UW-1 REASONABLENESS: %s=%.3f out of bounds (%.1f, %.1f) → clamped",
                       bounds_key, value, lo, hi)
        return max(lo, min(hi, value))
    return value


# ============================================================
# CLIO API Rate Limiter (separate from Hydrinia's 40K)
# ============================================================

class CLIOApiRateLimiter:
    """Track CLIO's own UW API call budget (10K/day, separate from Hydrinia)."""
    DAILY_CEILING = 10000
    SOFT_WARNING = 8000

    def __init__(self):
        self._call_count = 0
        self._reset_date: Optional[date] = None
        self._lock = threading.Lock()

    def _maybe_reset(self):
        today = date.today()
        if self._reset_date != today:
            self._call_count = 0
            self._reset_date = today

    def can_call(self) -> bool:
        with self._lock:
            self._maybe_reset()
            return self._call_count < self.DAILY_CEILING

    def record_call(self, endpoint: str = ""):
        with self._lock:
            self._maybe_reset()
            self._call_count += 1
            if self._call_count >= self.SOFT_WARNING:
                logger.warning("CLIO UW API budget: %d/%d (%.0f%%) [%s]",
                               self._call_count, self.DAILY_CEILING,
                               self._call_count / self.DAILY_CEILING * 100, endpoint)

    @property
    def calls_today(self) -> int:
        with self._lock:
            self._maybe_reset()
            return self._call_count


# ============================================================
# State
# ============================================================

_prev_vol_regime: Optional[str] = None
_prev_oracle_regime: Optional[str] = None
# S159 M-17: Repeated-rejection counters — accept regime after 3 consecutive rejections
_vol_regime_reject_count: int = 0
_oracle_regime_reject_count: int = 0
_prev_gex_regimes: dict[str, str] = {}
_regime_lock = threading.Lock()
_supplementary_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()
_rate_limiter = CLIOApiRateLimiter()

# Singleton
_instance: Optional["UWSignalBridge"] = None
_instance_lock = threading.Lock()


def get_uw_bridge() -> "UWSignalBridge":
    """Get or create singleton UWSignalBridge instance."""
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is not None:
            return _instance
        _instance = UWSignalBridge()
        return _instance


# ============================================================
# Bridge Client v2.0
# ============================================================

class UWSignalBridge:
    """Reads Hydrinia-exported UW signals (15 sections post-Trident) and provides
    clean API for CLIO consumers. Also makes supplementary API calls using CLIO's
    own 10K/day budget."""

    # --- Core ---

    def _read_snapshot(self) -> dict:
        """Read snapshot.json from bridge directory. Falls back to uw_flow_intel if stale."""
        if not (os.environ.get("CLIO_UW_BRIDGE_ENABLED", os.environ.get("MOLTBOT_UW_BRIDGE_ENABLED", "1")) == "1"):
            return {}
        try:
            if not os.path.isfile(_SNAPSHOT_FILE):
                return self._read_flow_intel_fallback()
            with open(_SNAPSHOT_FILE, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return self._read_flow_intel_fallback()
            # S160-7: If snapshot is very stale (>4h), try uw_flow_intel fallback
            ts_str = data.get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=_NY)
                    age = (datetime.now(_NY) - ts).total_seconds()
                    if age > _DEFAULT_TTL_SECONDS * _OFFHOURS_TTL_MULTIPLIER * 4:  # ~4 hours
                        fallback = self._read_flow_intel_fallback()
                        if fallback:
                            return fallback
                except (ValueError, TypeError):
                    pass
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("CLIO-UW-1 snapshot read failed: %s", e)
            return self._read_flow_intel_fallback()

    def _read_flow_intel_fallback(self) -> dict:
        """S160-7: Read latest uw_flow_intel daily file as snapshot fallback."""
        import glob as _glob
        _intel_dir = os.path.join(os.path.dirname(_SNAPSHOT_DIR), "intelligence")
        try:
            _files = sorted(
                _glob.glob(os.path.join(_intel_dir, "uw_flow_*.json")),
                reverse=True,
            )
            if not _files:
                return {}
            with open(_files[0], "r") as f:
                flow_data = json.load(f)
            if not isinstance(flow_data, dict):
                return {}
            # Map uw_flow_intel → snapshot structure
            _tide = flow_data.get("market_tide", {})
            _tide_dir = _tide.get("sentiment", "NEUTRAL") if isinstance(_tide, dict) else "NEUTRAL"
            _signals = flow_data.get("top_signals", [])

            # S168: Map GEX data — derive gex_regime from net_gex sign
            # net_gex < 0 → dealers short gamma → NEGATIVE_GEX (accelerates moves)
            # net_gex > 0 → dealers long gamma → POSITIVE_GEX (dampens moves)
            # CRITICAL: enum values MUST contain "NEGATIVE" / "POSITIVE" substrings —
            # get_conviction_factors() uses `"NEGATIVE" in gex_regime` matching.
            # Do not shorten to "NEG_GEX" / "POS_GEX" — would silently zero the boost.
            _raw_gex = flow_data.get("gex", {})
            _mapped_gex = {}
            if isinstance(_raw_gex, dict):
                for _gex_ticker, _gex_data in _raw_gex.items():
                    if not isinstance(_gex_data, dict):
                        continue
                    _net = _safe_float(_gex_data.get("net_gex"))
                    if _net < 0:
                        _regime = "NEGATIVE_GEX"
                    elif _net > 0:
                        _regime = "POSITIVE_GEX"
                    else:
                        _regime = "NEUTRAL"
                    _mapped_gex[_gex_ticker] = {"gex_regime": _regime, "net_gex": _net}

            # S168: Map top_signals → flow_confirm for the strongest signal
            _flow_confirm = {}
            if isinstance(_signals, list) and _signals:
                _top = _signals[0]
                if isinstance(_top, dict) and _top.get("ticker"):
                    _nofa_val = _safe_float(_top.get("nofa"))
                    _nofa_dir = "BULLISH" if _nofa_val > 0 else ("BEARISH" if _nofa_val < 0 else "NEUTRAL")
                    _flow_confirm = {
                        "last_check": {
                            "ticker": _top["ticker"],
                            "nofa_direction": _nofa_dir,
                            "oi_position_type": "NEW_POSITIONS" if abs(_nofa_val) > 0.5 else "NEUTRAL",
                        }
                    }

            _snapshot = {
                "timestamp": flow_data.get("created", ""),
                "freshness_ttl_seconds": _FALLBACK_TTL_SECONDS,
                "source": "uw_flow_intel_fallback",
                "market_context": {"market_tide_direction": _tide_dir},
                "sector_rotation": {"top_inflows": [], "top_outflows": [], "unusual_tickers": []},
                "flow_confirm": _flow_confirm,
                "vol_regime": {},
                "gex": _mapped_gex,
                "congressional": {},
                "lazarus_signals": {},
                "_flow_signals": _signals,
                "_flow_tide": _tide if isinstance(_tide, dict) else {},
            }
            logger.debug("CLIO-UW-1 using uw_flow_intel fallback (%s)", os.path.basename(_files[0]))
            return _snapshot
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.debug("uw_flow_intel fallback failed: %s", e)
            return {}

    def _is_market_hours(self) -> bool:
        """Check if within US market hours (9:30-16:00 ET, weekdays)."""
        now = datetime.now(_NY)
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close

    def _read_fresh_snapshot(self, caller: str = "") -> dict:
        """Read snapshot with staleness logging (log-only, never blocks).

        Calls _read_snapshot(), checks timestamp freshness using same TTL logic
        as is_fresh(), logs a warning if stale, but always returns the data.

        Design note (S156 H-C): Individual getters intentionally serve stale data
        for analysis context — learning engine benefits from slightly-stale UW data
        over no data. Real-time consumers should use get_snapshot() which rejects
        stale data and returns {}.
        """
        snapshot = self._read_snapshot()
        if not snapshot:
            return {}
        ts_str = snapshot.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=_NY)
                age = (datetime.now(_NY) - ts).total_seconds()
                ttl = snapshot.get("freshness_ttl_seconds", _DEFAULT_TTL_SECONDS)
                if not self._is_market_hours():
                    ttl *= _OFFHOURS_TTL_MULTIPLIER
                if age > ttl:
                    logger.info("CLIO-UW stale data served to %s: age=%ds > ttl=%ds",
                                caller, int(age), ttl)
            except (ValueError, TypeError):
                pass
        return snapshot

    def is_fresh(self) -> bool:
        """Check if snapshot is within acceptable freshness window."""
        snapshot = self._read_snapshot()
        if not snapshot:
            return False
        ts_str = snapshot.get("timestamp")
        if not ts_str:
            return False
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_NY)
            age = (datetime.now(_NY) - ts).total_seconds()
            ttl = snapshot.get("freshness_ttl_seconds", _DEFAULT_TTL_SECONDS)
            if not self._is_market_hours():
                ttl *= _OFFHOURS_TTL_MULTIPLIER
            if age > ttl:
                # S159-B L-1: Demote to DEBUG during off-hours (expected — UW doesn't refresh)
                if self._is_market_hours():
                    logger.warning("CLIO-UW-1 STALE: snapshot age=%ds > ttl=%ds", int(age), ttl)
                else:
                    logger.debug("CLIO-UW-1 STALE (off-hours, expected): snapshot age=%ds > ttl=%ds", int(age), ttl)
                return False
            return True
        except (ValueError, TypeError) as e:
            logger.debug("CLIO-UW-1 freshness check failed: %s", e)
            return False

    # --- Regime Validation ---

    def _validate_vol_regime(self, new_regime: str) -> str:
        global _prev_vol_regime, _vol_regime_reject_count
        with _regime_lock:
            if _prev_vol_regime is None:
                _prev_vol_regime = new_regime
                _vol_regime_reject_count = 0
                return new_regime
            new_level = _REGIME_LEVELS.get(new_regime)
            prev_level = _REGIME_LEVELS.get(_prev_vol_regime)
            if new_level is not None and prev_level is not None:
                delta = abs(new_level - prev_level)
                if delta > _MAX_REGIME_JUMP:
                    _vol_regime_reject_count += 1
                    # S159 M-17: Accept after 3 consecutive rejections (persistent shift)
                    if _vol_regime_reject_count >= 3:
                        logger.warning(
                            "S159 M-17: Vol regime jump accepted after %d rejections: %s -> %s",
                            _vol_regime_reject_count, _prev_vol_regime, new_regime,
                        )
                        _prev_vol_regime = new_regime
                        _vol_regime_reject_count = 0
                        return new_regime
                    logger.info(
                        "CLIO-UW-1 REGIME_GUARD: vol %s → %s (delta=%d), holding %s (reject #%d)",
                        _prev_vol_regime, new_regime, delta, _prev_vol_regime, _vol_regime_reject_count,
                    )
                    return _prev_vol_regime
            _prev_vol_regime = new_regime
            _vol_regime_reject_count = 0
            return new_regime

    def _validate_gex_regime(self, ticker: str, new_regime: str) -> str:
        global _prev_gex_regimes
        with _regime_lock:
            prev = _prev_gex_regimes.get(ticker)
            if prev is None:
                _prev_gex_regimes[ticker] = new_regime
                return new_regime
            if prev != new_regime:
                logger.info("CLIO-UW-1 GEX regime change: %s %s → %s", ticker, prev, new_regime)
            _prev_gex_regimes[ticker] = new_regime
            return new_regime

    # ── Original 7 Section Getters (backward-compatible) ──

    def get_snapshot(self) -> dict:
        """Full snapshot with freshness check. Returns empty dict if stale/missing."""
        if not self.is_fresh():
            return {}
        snapshot = self._read_snapshot()
        if not snapshot:
            return {}
        vol = snapshot.get("vol_regime", {})
        if isinstance(vol, dict) and vol.get("regime"):
            vol["regime"] = self._validate_vol_regime(vol["regime"])
        return snapshot

    def get_gex_regime(self, ticker: str = "SPY") -> dict:
        snapshot = self._read_fresh_snapshot(caller="get_gex_regime")
        gex_data = snapshot.get("gex", {})
        if not isinstance(gex_data, dict):
            return {}
        ticker_gex = gex_data.get(ticker, {})
        if isinstance(ticker_gex, dict) and ticker_gex.get("gex_regime"):
            ticker_gex["gex_regime"] = self._validate_gex_regime(ticker, ticker_gex["gex_regime"])
        return ticker_gex

    def get_flow_confirm_status(self) -> dict:
        snapshot = self._read_fresh_snapshot(caller="get_flow_confirm_status")
        return snapshot.get("flow_confirm", {})

    def get_vol_regime(self) -> dict:
        snapshot = self._read_fresh_snapshot(caller="get_vol_regime")
        vol = snapshot.get("vol_regime", {})
        if isinstance(vol, dict) and vol.get("regime"):
            vol["regime"] = self._validate_vol_regime(vol["regime"])
        return vol

    def get_sector_rotation(self) -> dict:
        snapshot = self._read_fresh_snapshot(caller="get_sector_rotation")
        return snapshot.get("sector_rotation", {})

    def get_unusual_tickers(self) -> list[str]:
        snapshot = self._read_fresh_snapshot(caller="get_unusual_tickers")
        sr = snapshot.get("sector_rotation", {})
        return sr.get("unusual_tickers", []) if isinstance(sr, dict) else []

    def get_lazarus_signals(self) -> dict:
        snapshot = self._read_fresh_snapshot(caller="get_lazarus_signals")
        return snapshot.get("lazarus_signals", {})

    def get_congressional_intel(self) -> dict:
        snapshot = self._read_fresh_snapshot(caller="get_congressional_intel")
        return snapshot.get("congressional", {})

    def get_market_context(self) -> dict:
        snapshot = self._read_fresh_snapshot(caller="get_market_context")
        return snapshot.get("market_context", {})

    # ── Trident v2.0 Section Getters (NEW) ──

    def get_oracle(self) -> dict:
        """Vol surface intelligence: IV-RV spread Z, term structure slope, skew Z, VoV, IV rank."""
        snapshot = self._read_fresh_snapshot(caller="get_oracle")
        oracle = snapshot.get("oracle", {})
        if not isinstance(oracle, dict):
            return {}
        # Apply reasonableness on Z-scored fields
        if "iv_rv_spread_z" in oracle:
            oracle["iv_rv_spread_z"] = _clamp(_safe_float(oracle["iv_rv_spread_z"]),
                                               "oracle.iv_rv_spread_z")
        if "skew_z" in oracle:
            oracle["skew_z"] = _clamp(_safe_float(oracle["skew_z"]), "oracle.skew_z")
        if "vol_of_vol" in oracle:
            oracle["vol_of_vol"] = _clamp(_safe_float(oracle["vol_of_vol"]), "oracle.vol_of_vol")
        if "iv_rank" in oracle:
            oracle["iv_rank"] = _clamp(_safe_float(oracle["iv_rank"]), "oracle.iv_rank")
        # Validate oracle regime if present
        regime = oracle.get("regime")
        if regime and regime in _REGIME_LEVELS:
            global _prev_oracle_regime, _oracle_regime_reject_count
            with _regime_lock:
                if _prev_oracle_regime is None:
                    _prev_oracle_regime = regime
                    _oracle_regime_reject_count = 0
                else:
                    new_lev = _REGIME_LEVELS.get(regime, 0)
                    prev_lev = _REGIME_LEVELS.get(_prev_oracle_regime, 0)
                    if abs(new_lev - prev_lev) > _MAX_REGIME_JUMP:
                        _oracle_regime_reject_count += 1
                        # S159 M-17: Accept after 3 consecutive rejections (persistent shift)
                        if _oracle_regime_reject_count >= 3:
                            logger.warning(
                                "S159 M-17: Oracle regime jump accepted after %d rejections: %s -> %s",
                                _oracle_regime_reject_count, _prev_oracle_regime, regime,
                            )
                            _prev_oracle_regime = regime
                            _oracle_regime_reject_count = 0
                        else:
                            logger.info("CLIO-UW-1 REGIME_GUARD: oracle %s → %s, holding %s (reject #%d)",
                                        _prev_oracle_regime, regime, _prev_oracle_regime, _oracle_regime_reject_count)
                            oracle["regime"] = _prev_oracle_regime
                    else:
                        _prev_oracle_regime = regime
                        _oracle_regime_reject_count = 0
        return oracle

    def get_dpof_enhanced(self) -> dict:
        """DPOF enhanced: NOPE Z, max pain, OI S/R, net flow expiry, flow concentration."""
        snapshot = self._read_fresh_snapshot(caller="get_dpof_enhanced")
        dpof = snapshot.get("dpof_enhanced", {})
        if not isinstance(dpof, dict):
            return {}
        # Validate NOPE Z-scores within ±4.0
        nope_z = dpof.get("nope_z", {})
        if isinstance(nope_z, dict):
            for ticker, z_val in nope_z.items():
                z = _safe_float(z_val)
                if abs(z) > 4.0:
                    logger.warning("CLIO-UW-1 REASONABLENESS: nope_z.%s=%.2f clamped", ticker, z)
                    nope_z[ticker] = max(-4.0, min(4.0, z))
            dpof["nope_z"] = nope_z
        return dpof

    def get_earnings(self) -> dict:
        """Earnings intelligence: upcoming, straddle accuracy, recent surprises."""
        snapshot = self._read_fresh_snapshot(caller="get_earnings")
        return snapshot.get("earnings", {}) if isinstance(snapshot.get("earnings"), dict) else {}

    def get_regime_enhanced(self) -> dict:
        """Regime enhanced: P/C ratio, correlations, insider aggregate, market OI changes."""
        snapshot = self._read_fresh_snapshot(caller="get_regime_enhanced")
        regime = snapshot.get("regime_enhanced", {})
        if not isinstance(regime, dict):
            return {}
        # Validate P/C ratio
        if "pc_ratio_5d" in regime:
            regime["pc_ratio_5d"] = _clamp(_safe_float(regime["pc_ratio_5d"]),
                                            "regime_enhanced.pc_ratio_5d")
        # Validate correlation changes (|delta| should be < 0.8)
        corr_changes = regime.get("correlation_changes", {})
        if isinstance(corr_changes, dict):
            for pair, data in corr_changes.items():
                if isinstance(data, dict) and "delta" in data:
                    d = _safe_float(data["delta"])
                    if abs(d) > 0.8:
                        logger.warning("CLIO-UW-1 REASONABLENESS: corr_change.%s.delta=%.2f clamped",
                                       pair, d)
                        data["delta"] = max(-0.8, min(0.8, d))
        return regime

    def get_sector_enhanced(self) -> dict:
        """Sector rotation enhanced: ETF overlap, group greek flow, ETF tide."""
        snapshot = self._read_fresh_snapshot(caller="get_sector_enhanced")
        return snapshot.get("sector_rotation_enhanced", {}) if isinstance(
            snapshot.get("sector_rotation_enhanced"), dict) else {}

    def get_lazarus_enhanced(self) -> dict:
        """Lazarus enhanced: short screener hits, short volume capitulation, FTD spikes."""
        snapshot = self._read_fresh_snapshot(caller="get_lazarus_enhanced")
        return snapshot.get("lazarus_enhanced", {}) if isinstance(
            snapshot.get("lazarus_enhanced"), dict) else {}

    def get_predictions(self) -> dict:
        """Prediction market: unusual finance, smart money, whale concentration, insider alerts."""
        snapshot = self._read_fresh_snapshot(caller="get_predictions")
        return snapshot.get("predictions", {}) if isinstance(
            snapshot.get("predictions"), dict) else {}

    def get_technicals(self, ticker: str = None) -> dict:
        """Technical indicators. If ticker provided, returns that ticker's data only."""
        snapshot = self._read_fresh_snapshot(caller="get_technicals")
        tech = snapshot.get("technicals", {})
        if not isinstance(tech, dict):
            return {}
        if ticker:
            return tech.get(ticker, {})
        return tech

    def get_active_trident_flags(self) -> list:
        """Which Hydrinia Trident feature flags are active (informational)."""
        snapshot = self._read_fresh_snapshot(caller="get_active_trident_flags")
        return snapshot.get("trident_flags_active", [])

    # ── Composite Helpers ──

    def get_conviction_factors(self, ticker: str, direction: str, *, allow_stale: bool = False) -> dict:
        """All conviction inputs for a trade idea evaluation.

        Returns dict with all 8 factor inputs pre-computed.
        Direction: 'LONG' or 'SHORT'.
        S133 HI-2: Read snapshot ONCE (was 9 file reads per call).
        S159-B M-1: allow_stale=True skips freshness gate (for off-hours
        trade_idea_feedback — stale data is better than all-zeros).
        """
        factors = {
            "gex_boost": 0.0, "flow_boost": 0.0, "vol_sizing": 1.0,
            "nope_boost": 0.0, "max_pain_penalty": 0.0, "oi_sr_boost": 0.0,
            "predictions_boost": 0.0, "technical_boost": 0.0,
        }
        if not allow_stale and not self.is_fresh():
            return factors

        # Single snapshot read for all factors (was 9 reads per call)
        snapshot = self._read_snapshot()
        if not snapshot:
            return factors

        is_long = direction.upper() in ("LONG", "BUY")

        # 1. GEX alignment (from snapshot directly)
        gex_data = snapshot.get("gex", {})
        if isinstance(gex_data, dict):
            ticker_gex = gex_data.get(ticker, {})
            if isinstance(ticker_gex, dict):
                gex_regime = ticker_gex.get("gex_regime", "NEUTRAL")
                if gex_regime:
                    gex_regime = self._validate_gex_regime(ticker, gex_regime)
                if is_long:
                    factors["gex_boost"] = 0.20 if "NEGATIVE" in gex_regime else (
                        -0.10 if "POSITIVE" in gex_regime else 0.0)
                else:
                    factors["gex_boost"] = 0.20 if "POSITIVE" in gex_regime else (
                        -0.10 if "NEGATIVE" in gex_regime else 0.0)

        # 2. Flow confirmation (from snapshot directly)
        fc = snapshot.get("flow_confirm", {})
        last_check = fc.get("last_check", {})
        if last_check.get("ticker") == ticker:
            nofa = last_check.get("nofa_direction", "")
            oi_type = last_check.get("oi_position_type", "")
            dir_match = (nofa == "BULLISH" and is_long) or (nofa == "BEARISH" and not is_long)
            if dir_match and oi_type == "NEW_POSITIONS":
                factors["flow_boost"] = 0.15
            elif not dir_match and oi_type == "CLOSING":
                factors["flow_boost"] = -0.15

        # 3. Vol regime sizing (from snapshot directly)
        vol = snapshot.get("vol_regime", {})
        if isinstance(vol, dict) and vol.get("regime"):
            vol["regime"] = self._validate_vol_regime(vol["regime"])
        factors["vol_sizing"] = {"CALM": 1.0, "ELEVATED": 0.90,
                                  "HIGH": 0.75, "CRISIS": 0.60}.get(vol.get("regime", "CALM"), 1.0)

        # 4. NOPE timing (from snapshot directly)
        dpof = snapshot.get("dpof_enhanced", {})
        if isinstance(dpof, dict):
            nope_z_dict = dpof.get("nope_z", {})
            if isinstance(nope_z_dict, dict):
                nope_key = ticker if ticker in nope_z_dict else "SPY"
                nope_z = _safe_float(nope_z_dict.get(nope_key))
                nope_z = max(-4.0, min(4.0, nope_z))  # Reasonableness clamp
                if abs(nope_z) > 2.0:
                    nope_opposing = (nope_z > 0 and not is_long) or (nope_z < 0 and is_long)
                    factors["nope_boost"] = 0.10 if nope_opposing else -0.10
        else:
            dpof = {}

        # 5. Max pain proximity
        mp = dpof.get("max_pain", {}).get(ticker, {})
        if mp:
            dist_pct = _safe_float(mp.get("distance_pct"))
            dte = mp.get("dte", 999)
            if isinstance(dte, (int, float)) and dte < 3 and abs(dist_pct) < 2.0:
                factors["max_pain_penalty"] = -0.10  # Pinning risk

        # 6. OI S/R alignment
        oi_sr = dpof.get("oi_sr", {}).get(ticker, {})
        if oi_sr:
            support = oi_sr.get("support", [])
            resistance = oi_sr.get("resistance", [])
            if support and resistance:
                factors["oi_sr_boost"] = 0.05  # Presence of structured S/R is mildly positive

        # 7. Predictions smart money (from snapshot directly)
        preds = snapshot.get("predictions", {})
        if isinstance(preds, dict):
            sm_consensus = preds.get("smart_money_consensus", "NEUTRAL")
            if sm_consensus != "NEUTRAL":
                sm_match = (sm_consensus == "BULLISH" and is_long) or (
                    sm_consensus == "BEARISH" and not is_long)
                factors["predictions_boost"] = 0.10 if sm_match else -0.10

        # 8. Technical signals (from snapshot directly)
        tech_all = snapshot.get("technicals", {})
        tech = tech_all.get(ticker, {}) if isinstance(tech_all, dict) and ticker else {}
        if tech:
            signals = tech.get("active_signals", [])
            rsi = _safe_float(tech.get("rsi_14")) if tech.get("rsi_14") is not None else None
            if is_long:
                if rsi is not None and rsi < 30:
                    factors["technical_boost"] = 0.10
                elif rsi is not None and rsi > 70:
                    factors["technical_boost"] = -0.10
                elif "TREND_STRONG" in signals:
                    factors["technical_boost"] = 0.05
            else:
                if rsi is not None and rsi > 70:
                    factors["technical_boost"] = 0.10
                elif rsi is not None and rsi < 30:
                    factors["technical_boost"] = -0.10

        # S151-B H-1: Diagnostic logging for zeroed conviction factors
        _zeroed = [k for k, v in factors.items() if v == 0.0 and k != "vol_sizing"]
        if _zeroed:
            _snap_keys = list(snapshot.keys()) if snapshot else []
            _missing_sections = [s for s in ("dpof_enhanced", "predictions", "technicals")
                                 if s not in _snap_keys or not snapshot.get(s)]
            logger.debug(
                f"UW conviction for {ticker}: {len(_zeroed)}/7 factors zeroed "
                f"({', '.join(_zeroed[:4])}{'...' if len(_zeroed) > 4 else ''})"
                f"{' | missing snapshot sections: ' + ', '.join(_missing_sections) if _missing_sections else ''}"
            )

        # S159 L-1: Warn when majority of conviction factors are zeroed
        _zero_count = sum(1 for v in factors.values() if v == 0.0)
        if len(factors) > 0 and _zero_count > len(factors) / 2:
            logger.warning(f"S159 L-1: {_zero_count}/{len(factors)} conviction factors zeroed for {ticker}")

        return factors

    def get_analysis_context_block(self, sections: list[str] = None) -> str:
        """Pre-formatted text block for injection into Claude analysis prompts.

        Args:
            sections: Optional list of section names to include. If None, include all available.

        Returns empty string if snapshot is stale/missing or bridge is LOG_ONLY.
        """
        if os.environ.get("CLIO_UW_BRIDGE_LOG_ONLY", os.environ.get("MOLTBOT_UW_BRIDGE_LOG_ONLY", "1")) == "1":
            snapshot = self._read_snapshot()
            if snapshot:
                logger.info("CLIO-UW-1 [LOG_ONLY] would provide context block (%d keys)", len(snapshot))
            return ""

        snapshot = self.get_snapshot()
        if not snapshot:
            return ""

        try:
            ts_str = snapshot.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=_NY)
                time_label = ts.astimezone(_NY).strftime("%H:%M ET")
            except (ValueError, TypeError):
                time_label = "?"

            lines = [f"=== UW SIGNAL CONTEXT ({time_label}) ==="]

            # Helper: should include this section?
            def _include(name: str) -> bool:
                return sections is None or name in sections

            # GEX
            if _include("gex"):
                gex = snapshot.get("gex", {})
                parts = []
                for t in ("SPY", "QQQ", "IWM"):
                    tg = gex.get(t, {})
                    r = tg.get("gex_regime", "")
                    if r:
                        label = "dealers amplifying" if "NEGATIVE" in r else (
                            "dealers suppressing" if "POSITIVE" in r else r.lower())
                        parts.append(f"{t}={r} ({label})")
                lines.append(f"GEX: {', '.join(parts) if parts else 'unavailable'}")

            # Vol regime
            if _include("vol_regime"):
                vol = snapshot.get("vol_regime", {})
                lines.append(
                    f"VOL: {vol.get('regime', '?')} (IV rank {vol.get('iv_rank_spy', '?')}th pct), "
                    f"term structure {vol.get('term_structure', '?')}, "
                    f"event risk {'FLAGGED' if vol.get('event_risk_flag') else 'CLEAR'}")

            # Oracle
            if _include("oracle"):
                oracle = snapshot.get("oracle", {})
                if oracle:
                    lines.append(
                        f"ORACLE: IV-RV spread Z={oracle.get('iv_rv_spread_z', '?')}, "
                        f"slope={oracle.get('term_structure_slope', '?')}, "
                        f"skew Z={oracle.get('skew_z', '?')}, "
                        f"VoV={oracle.get('vol_of_vol', '?')}")

            # Flow
            if _include("flow_confirm"):
                fc = snapshot.get("flow_confirm", {})
                lc = fc.get("last_check", {})
                mc = snapshot.get("market_context", {})
                flow_detail = ""
                if lc:
                    flow_detail = (f", NOFA {lc.get('nofa_direction', '?')} on "
                                   f"{lc.get('ticker', '?')} (genuineness {lc.get('genuineness', '?')})")
                lines.append(f"FLOW: Tide {mc.get('market_tide_direction', '?')}{flow_detail}")

            # DPOF enhanced
            if _include("dpof_enhanced"):
                dpof = snapshot.get("dpof_enhanced", {})
                if dpof:
                    nope_parts = [f"{t}={z}" for t, z in (dpof.get("nope_z") or {}).items()]
                    mp_parts = []
                    for t, mp in (dpof.get("max_pain") or {}).items():
                        if isinstance(mp, dict):
                            mp_parts.append(f"{t}@{mp.get('strike','?')} ({mp.get('distance_pct','?')}%)")
                    if nope_parts or mp_parts:
                        lines.append(f"DPOF: NOPE Z [{', '.join(nope_parts[:3])}] | "
                                     f"MaxPain [{', '.join(mp_parts[:3])}]")

            # Sectors
            if _include("sector_rotation"):
                sr = snapshot.get("sector_rotation", {})
                top_in = sr.get("top_inflows", [])[:3]
                top_out = sr.get("top_outflows", [])[:3]
                inflows = ", ".join(
                    f"{s.get('sector','?')} +${_safe_float(s.get('net_premium'))/1e6:.0f}M"
                    for s in top_in) if top_in else "none"
                outflows = ", ".join(
                    f"{s.get('sector','?')} -${abs(_safe_float(s.get('net_premium')))/1e6:.0f}M"
                    for s in top_out) if top_out else "none"
                unusual = sr.get("unusual_tickers", [])
                lines.append(f"SECTORS: In: {inflows} | Out: {outflows}")
                if unusual:
                    lines.append(f"UNUSUAL: {', '.join(unusual[:5])}")

            # Predictions
            if _include("predictions"):
                preds = snapshot.get("predictions", {})
                if preds:
                    sm = preds.get("smart_money_consensus", "NEUTRAL")
                    unusual_fin = preds.get("unusual_finance", [])
                    unusual_str = ", ".join(
                        f"{u.get('ticker','?')} ({u.get('direction','?')})"
                        for u in unusual_fin[:3]) if unusual_fin else "none"
                    lines.append(f"PREDICTIONS: Smart money={sm} | Unusual: {unusual_str}")

            # Regime enhanced
            if _include("regime_enhanced"):
                re_data = snapshot.get("regime_enhanced", {})
                if re_data:
                    pc = re_data.get("pc_ratio_5d", "?")
                    insider_agg = re_data.get("insider_aggregate", {})
                    ins_signal = insider_agg.get("signal", "?") if isinstance(insider_agg, dict) else "?"
                    corr_alerts = [p for p, d in (re_data.get("correlation_changes") or {}).items()
                                   if isinstance(d, dict) and d.get("alert")]
                    corr_str = ", ".join(corr_alerts) if corr_alerts else "none"
                    lines.append(f"REGIME: P/C={pc}, insider aggregate={ins_signal}, corr alerts: {corr_str}")

            # Earnings
            if _include("earnings"):
                earn = snapshot.get("earnings", {})
                upcoming = earn.get("upcoming", [])
                if upcoming:
                    lines.append("EARNINGS: " + ", ".join(
                        f"{e.get('ticker','?')} {e.get('date','?')} (±{e.get('expected_move_pct','?')}%)"
                        for e in upcoming[:3]))

            # Technicals (summary only)
            if _include("technicals"):
                tech = snapshot.get("technicals", {})
                if tech:
                    signal_summary = []
                    for t, td in list(tech.items())[:5]:
                        if isinstance(td, dict):
                            sigs = td.get("active_signals", [])
                            if sigs:
                                signal_summary.append(f"{t}: {','.join(sigs[:2])}")
                    if signal_summary:
                        lines.append(f"TECHNICALS: {' | '.join(signal_summary)}")

            # Lazarus
            if _include("lazarus_signals"):
                laz = snapshot.get("lazarus_signals", {})
                kills = laz.get("top_kill_scores", [])
                if kills:
                    lines.append("LAZARUS: " + ", ".join(
                        f"{k.get('ticker','?')} kill={k.get('score','?')}"
                        + (" (insider)" if k.get("insider_cluster") else "")
                        for k in kills[:3]))

            # Econ
            if _include("market_context"):
                mc = snapshot.get("market_context", {})
                econ = mc.get("economic_calendar_next", {})
                if econ and econ.get("event"):
                    lines.append(f"ECON: {econ.get('event')} ({econ.get('date','')}, {econ.get('impact','?')})")

            # Congressional
            if _include("congressional"):
                cong = snapshot.get("congressional", {})
                trades = cong.get("recent_trades", [])
                if trades:
                    cong_lines = []
                    for ct in trades[:3]:
                        ct_name = ct.get("politician", "?") if isinstance(ct, dict) else "?"
                        ct_ticker = ct.get("ticker", "?") if isinstance(ct, dict) else "?"
                        ct_type = ct.get("type", "?") if isinstance(ct, dict) else "?"
                        cong_lines.append(f"{ct_name} {ct_type} {ct_ticker}")
                    lines.append("CONGRESS: " + "; ".join(cong_lines))

            return "\n".join(lines)

        except Exception as e:
            logger.debug("CLIO-UW-1 context block generation failed: %s", e)
            return ""

    # ── Supplementary API (CLIO's own 10K budget) ──

    def _get_cached(self, key: str, ttl: float) -> Any:
        with _cache_lock:
            entry = _supplementary_cache.get(key)
            if entry is None:
                return None
            cached_at, value = entry
            if time.monotonic() - cached_at > ttl:
                return None
            return value

    def _set_cached(self, key: str, value: Any) -> None:
        with _cache_lock:
            if len(_supplementary_cache) > 500:
                oldest = sorted(_supplementary_cache, key=lambda k: _supplementary_cache[k][0])
                for k in oldest[:100]:
                    del _supplementary_cache[k]
            _supplementary_cache[key] = (time.monotonic(), value)

    async def _clio_uw_get(self, endpoint: str, params: dict = None) -> dict:
        """Make a UW call behind both CLIO's 10K and persistent 16K guards."""
        if not (os.environ.get("CLIO_UW_SUPPLEMENTARY_API", os.environ.get("MOLTBOT_UW_SUPPLEMENTARY_API", "1")) == "1"):
            return {}
        if not _rate_limiter.can_call():
            logger.warning("CLIO-UW-7 rate limit exceeded for %s", endpoint)
            return {}
        if os.environ.get("CLIO_UW_SUPPLEMENTARY_LOG_ONLY", "1") == "1":
            logger.info("CLIO-UW-7 [LOG_ONLY] would call %s", endpoint)
            return {}
        try:
            from agent.integrations.uw_flow_intel import _uw_get
            _rate_limiter.record_call(endpoint)
            data = await _uw_get(endpoint, params=params)
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return {"data": data}  # Wrap list APIs for uniform dict access
            return {}
        except Exception as e:
            logger.debug("CLIO-UW-7 API call failed %s: %s", endpoint, e)
            return {}

    async def fetch_fundamentals(self, ticker: str, statement: str = "financials") -> dict:
        """Fetch fundamental data for a ticker. statement: financials|balance-sheets|cash-flows|income-statements"""
        cache_key = f"fundamentals_{ticker}_{statement}"
        cached = self._get_cached(cache_key, _CLIO_CACHE_TTLS.get("fundamentals", 86400))
        if cached is not None:
            return cached
        data = await self._clio_uw_get(f"/api/fundamentals/{ticker}/{statement}")
        if data:
            self._set_cached(cache_key, data)
        return data

    async def fetch_earnings_detail(self, ticker: str) -> dict:
        cache_key = f"earnings_detail_{ticker}"
        cached = self._get_cached(cache_key, _CLIO_CACHE_TTLS.get("earnings_detail", 86400))
        if cached is not None:
            return cached
        data = await self._clio_uw_get(f"/api/earnings/{ticker}")
        if data:
            self._set_cached(cache_key, data)
        return data

    async def fetch_ticker_technicals(self, ticker: str) -> dict:
        cache_key = f"technicals_{ticker}"
        cached = self._get_cached(cache_key, _CLIO_CACHE_TTLS.get("ticker_technicals", 7200))
        if cached is not None:
            return cached
        data = await self._clio_uw_get(f"/api/technical-indicator/{ticker}")
        if data:
            self._set_cached(cache_key, data)
        return data

    async def fetch_short_interest(self, ticker: str) -> dict:
        cache_key = f"short_{ticker}"
        cached = self._get_cached(cache_key, _CLIO_CACHE_TTLS.get("short_interest", 14400))
        if cached is not None:
            return cached
        data = await self._clio_uw_get(f"/api/shorts/{ticker}/volume-and-ratio")
        if data:
            self._set_cached(cache_key, data)
        return data

    async def fetch_politician_portfolio(self, politician_id: str) -> dict:
        cache_key = f"politician_{politician_id}"
        cached = self._get_cached(cache_key, _CLIO_CACHE_TTLS.get("politician_portfolio", 604800))
        if cached is not None:
            return cached
        data = await self._clio_uw_get(f"/api/politician/{politician_id}/portfolio")
        if data:
            self._set_cached(cache_key, data)
        return data

    async def fetch_insider_detail(self, ticker: str) -> list[dict]:
        cache_key = f"insider_{ticker}"
        cached = self._get_cached(cache_key, _CLIO_CACHE_TTLS.get("insider_detail", 86400))
        if cached is not None:
            return cached
        data = await self._clio_uw_get(f"/api/insider-trades/{ticker}")
        result = data.get("data", []) if isinstance(data, dict) else []
        if result:
            self._set_cached(cache_key, result)
        return result

    async def fetch_fda_calendar_detail(self, ticker: str) -> list[dict]:
        cache_key = f"fda_{ticker}"
        cached = self._get_cached(cache_key, _CLIO_CACHE_TTLS.get("fda_calendar", 86400))
        if cached is not None:
            return cached
        data = await self._clio_uw_get(f"/api/fda-calendar", params={"ticker": ticker})
        result = data.get("data", []) if isinstance(data, dict) else []
        if result:
            self._set_cached(cache_key, result)
        return result

    @property
    def rate_limiter(self) -> CLIOApiRateLimiter:
        return _rate_limiter
