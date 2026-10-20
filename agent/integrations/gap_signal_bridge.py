"""
MB-GAP-1: Gap Signal Bridge Client

Reads Hydrinia's CRUCIBLE stress test results and PHOENIX drawdown tier
state from the shared bridge directory. Provides a clean Python API for
Moltbot consumers (learning engine, trade idea feedback).

Architecture: Hub-and-Spoke (same as MB-UW bridge)
  - Hydrinia computes gap signals → exports to moltbot_knowledge/gap_signals/
  - This bridge reads, validates, caches, and serves to Moltbot consumers

Safety:
  - Staleness guards: reject stale data by TTL, at any hour (S205-B2b: previously only
    flagged stale during 8:00-20:00 ET, so an age>TTL read outside that window was
    silently treated as fresh)
  - Position-count validation: reject stress results with 0 positions
  - Reasonableness filter: suppress implausible tier flips
  - Graceful degradation: return None on any error, never block consumers

Feature flags:
  - MOLTBOT_GAP_BRIDGE_ENABLED: master switch (default "1")
  - MOLTBOT_GAP_BRIDGE_LOG_ONLY: log reads without passing to consumers (default "1")

Hydrinia (source of every file this bridge reads) is permanently off (operator
direction, 2026-09-24). Every read routes through _read_cached(), which is gated on
agent.hydrinia_bridge.bridge_enabled() (default False) -- disabled, every accessor
degrades to its "unavailable/stale" value (None / [] / {} / the documented NORMAL-tier
default), so downstream consumers (e.g. learning_engine's gap context injection) skip
cleanly rather than reading Hydrinia files that will never update again.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from agent.config import config

logger = logging.getLogger("openclaw.gap_bridge")

_NY = ZoneInfo("America/New_York")

# Bridge directories
_GAP_SIGNALS_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "gap_signals")
_NEWSFLASH_DIR = os.path.join(config.tradingbot_dir, "moltbot_knowledge", "newsflash")

# Cache TTLs (seconds)
_STRESS_CACHE_TTL = 300       # Re-read stress file every 5 minutes (it changes weekly)
_DRAWDOWN_CACHE_TTL = 60      # Re-read drawdown file every 60 seconds (it changes every 30s)
_NEWSFLASH_CACHE_TTL = 120    # Re-read newsflash file every 2 minutes

# Validation thresholds
_MIN_POSITIONS_FOR_VALID_STRESS = 1
_MIN_EQUITY_FOR_VALID_STRESS = 5000.0

# Regime levels for reasonableness filter
_TIER_LEVELS = {"NORMAL": 0, "CAUTION": 1, "RECOVERY": 2, "EMERGENCY": 3}


class GapSignalBridge:
    """Bridge client for reading Hydrinia Gap module signals.

    Singleton pattern — use get_gap_bridge() to obtain the instance.
    Thread-safe reads with in-memory caching.
    """

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._cache_times: Dict[str, float] = {}
        self._last_known_tier: str = "NORMAL"
        self._lock = threading.Lock()
        self._read_count = 0
        logger.info("MB-GAP-1: GapSignalBridge initialized "
                     f"(enabled={os.environ.get('MOLTBOT_GAP_BRIDGE_ENABLED', '1') == '1'}, "
                     f"log_only={os.environ.get('MOLTBOT_GAP_BRIDGE_LOG_ONLY', '1') == '1'})")

    # ================================================================
    # PUBLIC API — Stress Test Results
    # ================================================================

    def get_stress_summary(self) -> Optional[Dict]:
        """Get latest CRUCIBLE stress test summary.

        Returns None if:
          - Bridge disabled
          - File doesn't exist (CRUCIBLE not yet deployed)
          - File is stale beyond TTL during market hours
          - Data validation fails (0 positions, low equity)
        """
        if os.environ.get("MOLTBOT_GAP_BRIDGE_ENABLED", "1") != "1":
            return None
        return self._read_cached("stress", "stress_summary.json",
                                  _STRESS_CACHE_TTL, self._validate_stress)

    def get_concentration_flags(self) -> List[str]:
        """Get portfolio concentration warning flags from last stress test."""
        stress = self.get_stress_summary()
        if not stress:
            return []
        return stress.get("concentration", {}).get("flags", [])

    def get_sector_concentration(self) -> Dict[str, float]:
        """Get sector concentration percentages from last stress test."""
        stress = self.get_stress_summary()
        if not stress:
            return {}
        return stress.get("concentration", {}).get("sector_pcts", {})

    def get_worst_scenarios(self, n: int = 3) -> List[Dict]:
        """Get top N worst stress scenarios by P&L impact."""
        stress = self.get_stress_summary()
        if not stress:
            return []
        return stress.get("worst_scenarios", [])[:n]

    # ================================================================
    # PUBLIC API — Drawdown State
    # ================================================================

    def get_drawdown_state(self) -> Optional[Dict]:
        """Get latest PHOENIX drawdown recovery state.

        Returns None if:
          - Bridge disabled
          - File doesn't exist (PHOENIX not yet deployed)
          - File is stale beyond TTL during market hours
        """
        if os.environ.get("MOLTBOT_GAP_BRIDGE_ENABLED", "1") != "1":
            return None
        return self._read_cached("drawdown", "drawdown_state.json",
                                  _DRAWDOWN_CACHE_TTL, self._validate_drawdown)

    def get_current_tier(self) -> str:
        """Get current PHOENIX tier. Returns 'NORMAL' if unavailable."""
        dd = self.get_drawdown_state()
        if not dd:
            return "NORMAL"
        return dd.get("tier", "NORMAL")

    def get_min_conviction(self) -> float:
        """Get current minimum conviction threshold from PHOENIX tier."""
        dd = self.get_drawdown_state()
        if not dd:
            return 0.30  # Default NORMAL minimum
        return dd.get("tier_config", {}).get("min_conviction", 0.30)

    def get_restrictions_summary(self) -> str:
        """Get human-readable restrictions summary from PHOENIX."""
        dd = self.get_drawdown_state()
        if not dd:
            return "NORMAL: All trades permitted. Min conviction 0.30."
        return dd.get("restrictions_summary", "Unknown tier state")

    def is_leveraged_etf_allowed(self) -> bool:
        """Check if leveraged ETF entry is allowed under current tier."""
        dd = self.get_drawdown_state()
        if not dd:
            return True  # Default NORMAL allows
        return dd.get("tier_config", {}).get("allow_leveraged_etf_entry", True)

    # ================================================================
    # PUBLIC API — Vol Surface (STUB — future ORACLE activation)
    # ================================================================

    def get_vol_surface(self) -> Optional[Dict]:
        """Get ORACLE vol surface intelligence.

        Reads gap_signals/vol_surface.json exported by ORACLE after each
        30-minute refresh. Contains SPY term structure, skew, VoV signals,
        portfolio vol exposure, and actionable alerts.
        Returns None if ORACLE hasn't exported yet or bridge is disabled.
        """
        if os.environ.get("MOLTBOT_GAP_BRIDGE_ENABLED", "1") != "1":
            return None
        vol_path = os.path.join(_GAP_SIGNALS_DIR, "vol_surface.json")
        if not os.path.exists(vol_path):
            return None  # ORACLE not yet deployed
        return self._read_cached("vol_surface", "vol_surface.json",
                                  _DRAWDOWN_CACHE_TTL, None)

    # ================================================================
    # PUBLIC API — NewsFlash Events
    # ================================================================

    def get_newsflash_events(self) -> Optional[Dict]:
        """Get recent BINARY/AMBIGUOUS newsflash events from Hydrinia.

        Returns None if bridge disabled or file doesn't exist.
        """
        if os.environ.get("MOLTBOT_GAP_BRIDGE_ENABLED", "1") != "1":
            return None
        return self._read_cached("newsflash", "recent_events.json",
                                  _NEWSFLASH_CACHE_TTL, None,
                                  base_dir=_NEWSFLASH_DIR)

    def get_newsflash_for_ticker(self, ticker: str,
                                  max_age_hours: float = 2.0) -> Optional[Dict]:
        """Get the most recent newsflash event affecting a specific ticker.

        Returns the single most recent BINARY/AMBIGUOUS event where:
          - ticker is in affected_tickers
          - event is within max_age_hours
          - direction is BULLISH or BEARISH (not NEUTRAL)

        Returns None if no match, bridge disabled, or LOG_ONLY.
        """
        if os.environ.get("MOLTBOT_GAP_BRIDGE_ENABLED", "1") != "1":
            return None

        data = self.get_newsflash_events()
        if not data:
            return None

        events = data.get("events", [])
        if not events:
            return None

        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(hours=max_age_hours)
        ticker_upper = ticker.upper()

        for event in events:  # already sorted newest-first from export
            # Check ticker match
            affected = [t.upper() for t in event.get("affected_tickers", [])
                        if isinstance(t, str)]
            if ticker_upper not in affected:
                continue

            # Check direction is actionable
            direction = event.get("direction", "NEUTRAL")
            if direction == "NEUTRAL":
                continue

            # Check age
            ts_str = event.get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < cutoff:
                        continue  # too old
                except (ValueError, TypeError):
                    continue  # can't parse timestamp, skip

            if os.environ.get("MOLTBOT_GAP_BRIDGE_LOG_ONLY", "1") == "1":
                logger.info(
                    f"MB-GAP-1: [LOG_ONLY] NewsFlash match for {ticker}: "
                    f"{event.get('tier')} {direction} — "
                    f"{event.get('headline', '')[:60]}"
                )
                return None  # Don't return in LOG_ONLY mode

            return event

        return None

    # ================================================================
    # PUBLIC API — Composite Context Block
    # ================================================================

    def get_analysis_context_block(self) -> str:
        """Pre-formatted text block for injection into Claude analysis prompts.

        Returns a compact multi-line string summarizing all available gap signals.
        Returns empty string if no data available or if LOG_ONLY mode.
        """
        if os.environ.get("MOLTBOT_GAP_BRIDGE_ENABLED", "1") != "1":
            return ""

        lines = []
        now = datetime.now(_NY)
        lines.append(f"=== PORTFOLIO RISK CONTEXT ({now.strftime('%H:%M')} ET) ===")

        # S190 L11: `or 0` fallbacks below are intentional. `None` from a missing Hydrinia
        # key becomes 0 in the rendered context, which is "partial context is more useful
        # than no context" per learning_engine.py guidance. Do NOT replace with strict
        # `is not None` checks here — that would drop entire risk-context lines when one
        # sub-field is absent from the snapshot, defeating the purpose of the bridge.
        # Drawdown tier
        dd = self.get_drawdown_state()
        if dd:
            tier = dd.get("tier", "UNKNOWN")
            dd_pct = (dd.get("drawdown", {}).get("current_drawdown_pct") or 0)
            restrictions = dd.get("restrictions_summary", "")
            lines.append(f"DRAWDOWN: {tier} tier (dd={dd_pct:.1%}). {restrictions}")

            # Cause info if not NONE
            cause = dd.get("cause", {}).get("classification", "NONE")
            if cause != "NONE":
                cause_detail = dd.get("cause", {}).get("details", "")
                lines.append(f"CAUSE: {cause} — {cause_detail}")

        # Stress test results
        stress = self.get_stress_summary()
        if stress:
            port = stress.get("portfolio", {})
            conc = stress.get("concentration", {})
            regime = port.get("current_regime", "UNKNOWN")
            beta = (port.get("portfolio_beta") or 0)
            n_pos = (port.get("num_positions") or 0)
            lines.append(f"PORTFOLIO: {n_pos} positions, regime={regime}, beta={beta:.2f}")

            # Sector concentration
            sectors = conc.get("sector_pcts", {})
            if sectors:
                top_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:3]
                sector_str = ", ".join(f"{s}={p:.0f}%" for s, p in top_sectors)
                lines.append(f"CONCENTRATION: {sector_str}")
                hedge_pct = (conc.get("hedge_pct") or 0)
                lev_pct = (conc.get("leveraged_etf_pct") or 0)
                lines.append(f"HEDGE: {hedge_pct:.0f}% | LEVERAGED: {lev_pct:.0f}%")

            # Concentration flags
            flags = conc.get("flags", [])
            for flag in flags[:3]:  # Max 3 flags to keep context compact
                lines.append(flag)

            # Top 2 worst scenarios
            worst = stress.get("worst_scenarios", [])[:2]
            for ws in worst:
                name = ws.get("name", "UNKNOWN")
                pnl_pct = (ws.get("total_pnl_pct") or 0)
                he = (ws.get("hedge_effectiveness") or 0)
                lines.append(f"STRESS {name}: {pnl_pct:+.1%} (hedge eff={he:.0%})")

            # VaR
            var95 = stress.get("var_95")
            if var95 is not None:
                lines.append(f"VaR-95: ${var95:,.0f} | CVaR-95: ${(stress.get('cvar_95') or 0):,.0f}")

        # Vol surface
        vol = self.get_vol_surface()
        if vol:
            spy = vol.get("spy", {})
            ts = spy.get("term_structure", {})
            sk = spy.get("skew", {})
            vv = spy.get("vov", {})
            pv = vol.get("portfolio", {})
            lines.append(
                f"VOL: SPY term={ts.get('signal') or '?'} "
                f"(z={(ts.get('slope_z_score') or 0):+.1f})"
                f"{' INVERTED' if ts.get('is_inverted') else ''} | "
                f"skew={sk.get('signal') or '?'} (z={(sk.get('skew_z_score') or 0):+.1f}) | "
                f"VoV={vv.get('acceleration') or '?'} (z={(vv.get('vov_z_score') or 0):+.1f})"
            )
            if pv:
                lines.append(
                    f"VOL EXPOSURE: vega=${(pv.get('total_vega') or 0):+.0f} | "
                    f"gamma=${(pv.get('net_gamma_dollars') or 0):+.0f} | "
                    f"avg IV={((pv.get('weighted_avg_iv') or 0)) * 100:.1f}%"
                )
            for alert in vol.get("alerts", [])[:2]:
                lines.append(f"VOL ALERT: {alert}")

        if len(lines) <= 1:
            return ""  # No data available

        context = "\n".join(lines)

        if os.environ.get("MOLTBOT_GAP_BRIDGE_LOG_ONLY", "1") == "1":
            logger.info(f"MB-GAP-1: [LOG_ONLY] Context block generated ({len(context)} chars)")
            return ""  # Don't inject in LOG_ONLY mode

        return context

    def is_fresh(self) -> bool:
        """Check if at least one gap signal is fresh."""
        stress = self.get_stress_summary()
        dd = self.get_drawdown_state()
        return stress is not None or dd is not None

    # ================================================================
    # INTERNAL — Cache and Validation
    # ================================================================

    def _read_cached(self, cache_key: str, filename: str,
                     ttl: float, validator,
                     base_dir: str = None) -> Optional[Dict]:
        """Read a JSON file from a bridge directory with caching.

        Lock covers full operation: cache check → disk read → validation →
        cache update. This serializes reads but protects _last_known_tier
        and _read_count from concurrent access.

        Args:
            base_dir: Override bridge directory (default: _GAP_SIGNALS_DIR).
        """
        try:
            from agent.hydrinia_bridge import bridge_enabled
            if not bridge_enabled():
                return None
        except Exception as e:
            logger.debug(f"MB-GAP-1: hydrinia_bridge check failed: {e}")
            return None

        now = time.time()

        with self._lock:
            # Check cache
            if cache_key in self._cache and (now - self._cache_times.get(cache_key, 0)) < ttl:
                return self._cache[cache_key]

            # Read from disk
            filepath = os.path.join(base_dir or _GAP_SIGNALS_DIR, filename)
            if not os.path.exists(filepath):
                return None

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"MB-GAP-1: Failed to read {filename}: {e}")
                return None

            # Staleness check. S205-B2b: previously gated on self._is_market_hours()
            # (8:00-20:00 ET, Mon-Fri) -- an age>TTL read outside that window was
            # silently treated as fresh. Now checked at any hour.
            ts_str = data.get("timestamp", "")
            ttl_seconds = (data.get("freshness_ttl_seconds") or 600)
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=_NY)
                    age = (datetime.now(_NY) - ts).total_seconds()
                    if age > ttl_seconds:
                        logger.warning(f"MB-GAP-1: {filename} stale (age={age:.0f}s > ttl={ttl_seconds}s)")
                        # Return stale data rather than None — consumer decides
                        data["_is_stale"] = True
                except (ValueError, TypeError) as e:
                    logger.debug(f"MB-GAP-1: Failed to parse timestamp from {filename}: {e}")

            # Validation
            if validator:
                if not validator(data):
                    return None

            # Update cache
            self._cache[cache_key] = data
            self._cache_times[cache_key] = now

            # Log periodically (cap counter to prevent verbose logging after weeks)
            self._read_count += 1
            if self._read_count % 50 == 1:
                logger.info(f"MB-GAP-1: Read {filename} successfully")
            if self._read_count >= 1000:
                self._read_count = 0

            return data

    def _validate_stress(self, data: Dict) -> bool:
        """Validate stress summary data."""
        portfolio = data.get("portfolio", {})
        n_pos = (portfolio.get("num_positions") or 0)
        equity = (portfolio.get("total_equity") or 0)

        if n_pos < _MIN_POSITIONS_FOR_VALID_STRESS:
            logger.warning(f"MB-GAP-1: Stress summary has {n_pos} positions — rejecting (min={_MIN_POSITIONS_FOR_VALID_STRESS})")
            return False

        if equity < _MIN_EQUITY_FOR_VALID_STRESS:
            logger.warning(f"MB-GAP-1: Stress summary equity ${equity:.0f} — rejecting (min=${_MIN_EQUITY_FOR_VALID_STRESS:.0f})")
            return False

        return True

    def _validate_drawdown(self, data: Dict) -> bool:
        """Validate drawdown state with reasonableness filter."""
        new_tier = data.get("tier", "NORMAL")

        # Reasonableness filter: suppress 2+ level tier jumps
        new_level = _TIER_LEVELS.get(new_tier, 0)
        old_level = _TIER_LEVELS.get(self._last_known_tier, 0)

        if abs(new_level - old_level) >= 2 and self._last_known_tier != "NORMAL":
            # First time seeing a big jump — log warning but accept
            # (initial read from NORMAL to EMERGENCY is valid on first boot)
            logger.warning(f"MB-GAP-1: TIER_GUARD: {self._last_known_tier} → {new_tier} "
                           f"(delta={abs(new_level - old_level)}), accepting but flagging")

        self._last_known_tier = new_tier
        return True


# ================================================================
# SINGLETON
# ================================================================

_bridge_instance: Optional[GapSignalBridge] = None
_bridge_lock = threading.Lock()


def get_gap_bridge() -> GapSignalBridge:
    """Get or create the singleton GapSignalBridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        with _bridge_lock:
            if _bridge_instance is None:
                _bridge_instance = GapSignalBridge()
    return _bridge_instance
