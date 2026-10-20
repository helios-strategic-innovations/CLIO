"""
AthenaBridge: Vault-reading bridge for Hydrinia's AthenaEngine forecasts.

Hub-and-spoke pattern (same as gap_signal_bridge.py):
  - Hydrinia runs TimesFM forecasts -> exports to vault JSONL
  - This bridge reads, caches, and serves forecast data to CLIO consumers

Data sources:
  - /home/agent/.hydrinia/vault/athena/forecasts.jsonl (ticker + macro forecasts)

Feature flags:
  - CLIO_ATHENA_ENABLED: master switch (default "0" = OFF)
  - CLIO_ATHENA_LOG_ONLY: log forecasts but don't inject into reports (default "1")

Returns AthenaForecast-compatible dicts derived from vault data.
Graceful degradation: returns None on any error, never blocks consumers.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openclaw.athena_bridge")

# Vault paths (Hydrinia writes, CLIO reads)
_VAULT_DIR = os.path.join(os.path.expanduser("~"), ".hydrinia", "vault", "athena")
_FORECASTS_JSONL = os.path.join(_VAULT_DIR, "forecasts.jsonl")

# Cache TTLs (seconds)
_TICKER_CACHE_TTL = 900       # 15 min
_MACRO_CACHE_TTL = 3600       # 60 min
_VAULT_RELOAD_TTL = 300       # Re-read vault file every 5 min

# Staleness threshold: reject forecasts older than this (seconds)
_FORECAST_STALENESS_THRESHOLD = 86400  # 24 hours

# Bridge log file (CLIO-side)
_BRIDGE_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "athena_bridge_log.jsonl",
)


def _is_enabled() -> bool:
    """Check if Athena bridge is enabled (runtime-evaluated)."""
    return os.environ.get("CLIO_ATHENA_ENABLED", "0") == "1"


def _is_log_only() -> bool:
    """Check if Athena bridge is in log-only mode (runtime-evaluated)."""
    return os.environ.get("CLIO_ATHENA_LOG_ONLY", "1") == "1"


def _directional_label(alignment: float) -> str:
    """Convert directional_alignment float (-1.0 to +1.0) to label.
    S170B P2-5: NaN/inf/non-numeric explicitly fall through to NEUTRAL."""
    import math as _m
    if not isinstance(alignment, (int, float)) or isinstance(alignment, bool):
        return "NEUTRAL"
    if _m.isnan(alignment) or _m.isinf(alignment):
        return "NEUTRAL"
    if alignment > 0.3:
        return "BULLISH"
    elif alignment < -0.3:
        return "BEARISH"
    return "NEUTRAL"


class AthenaBridge:
    """Bridge client for reading Hydrinia Athena vault forecasts.

    Singleton pattern -- use get_athena_bridge() to obtain the instance.
    Thread-safe reads with in-memory caching.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, float] = {}
        self._vault_data: List[Dict] = []
        self._vault_mtime: float = 0.0
        self._vault_loaded_at: float = 0.0
        self._lock = threading.Lock()
        self._load_failed = False
        self._malformed_drop_count = 0
        logger.info(
            "AthenaBridge initialized "
            f"(enabled={_is_enabled()}, log_only={_is_log_only()})"
        )

    # ================================================================
    # VAULT READING
    # ================================================================

    def _reload_vault_if_stale(self) -> None:
        """Reload vault JSONL if file has been modified since last read."""
        now = time.time()
        if now - self._vault_loaded_at < _VAULT_RELOAD_TTL:
            return  # Cache still fresh

        try:
            if not os.path.isfile(_FORECASTS_JSONL):
                self._vault_data = []
                self._vault_loaded_at = now
                self._malformed_drop_count = 0
                self._load_failed = False
                return

            mtime = os.path.getmtime(_FORECASTS_JSONL)
            if mtime == self._vault_mtime and self._vault_data:
                self._vault_loaded_at = now
                return  # File unchanged

            entries = []
            dropped = 0
            raw_line_count = 0
            with open(_FORECASTS_JSONL, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw_line_count += 1
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        dropped += 1
                        continue

            self._vault_data = entries
            self._vault_mtime = mtime
            self._vault_loaded_at = now
            self._malformed_drop_count = dropped
            # A non-empty file that yields zero valid entries is a load failure,
            # not a quiet zero -- distinct from "file absent" above.
            self._load_failed = raw_line_count > 0 and not entries
            if dropped:
                logger.warning(
                    "AthenaBridge: dropped %d malformed JSONL line(s) while loading vault (%d valid)",
                    dropped, len(entries),
                )
            logger.debug(
                "AthenaBridge: vault reloaded (%d entries, mtime=%s)",
                len(entries), datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
            )

        except Exception as e:
            logger.error("AthenaBridge: vault reload failed: %s", e)
            self._vault_loaded_at = now  # Avoid retry storm
            self._load_failed = True

    def _find_latest_forecast(
        self, ticker: str, event_type: str = "forecast_ticker"
    ) -> Optional[Dict]:
        """Find the most recent vault entry for a ticker and event type."""
        self._reload_vault_if_stale()

        best = None
        best_ts = ""
        for entry in self._vault_data:
            if entry.get("event_type") != event_type:
                continue
            data = entry.get("data", {})
            if data.get("ticker") != ticker:
                continue
            ts = entry.get("timestamp", "")
            if ts > best_ts:
                best_ts = ts
                best = data

        if best is None:
            return None

        # Staleness check.
        # S170B P2-A: prefer the entry-ROOT timestamp (best_ts) — canonical.
        # Vault writers populate `data.timestamp` redundantly today, but that's
        # not contractual. Fall back to data.timestamp only if root is empty.
        try:
            forecast_ts = best_ts or best.get("timestamp", "")
            if forecast_ts:
                dt = datetime.fromisoformat(forecast_ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - dt).total_seconds()
                if age > _FORECAST_STALENESS_THRESHOLD:
                    logger.debug(
                        "AthenaBridge: stale forecast for %s (age=%.0fh)",
                        ticker, age / 3600,
                    )
                    return None
        except (ValueError, TypeError):
            pass  # If we can't parse timestamp, still use the data

        return best

    # ================================================================
    # CACHE
    # ================================================================

    def _get_cached(self, key: str, ttl: float) -> Optional[Any]:
        """Retrieve cached value if not expired.
        S170B P2-4: serialize cache reads/writes under self._lock."""
        with self._lock:
            if key not in self._cache:
                return None
            age = time.time() - self._cache_times.get(key, 0)
            if age > ttl:
                self._cache.pop(key, None)
                self._cache_times.pop(key, None)
                return None
            return self._cache[key]

    def _set_cached(self, key: str, value: Any) -> None:
        """Store value in cache.
        S170B P2-4: serialize cache reads/writes under self._lock."""
        with self._lock:
            self._cache[key] = value
            self._cache_times[key] = time.time()

    # ================================================================
    # LOGGING
    # ================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Append event to bridge JSONL log (fire-and-forget)."""
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "data": data,
            }
            # Atomic append (single write, no need for flock on append-only JSONL)
            with open(_BRIDGE_LOG_PATH, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.debug("AthenaBridge: log write failed: %s", e)

    # ================================================================
    # PUBLIC API
    # ================================================================

    def is_available(self) -> bool:
        """Check if vault data is accessible."""
        if not _is_enabled():
            return False
        if self._load_failed:
            return False
        return os.path.isfile(_FORECASTS_JSONL)

    def forecast_ticker(
        self, ticker: str, horizon: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest vault forecast for a ticker.

        Returns dict with:
          - point_forecast: list of forecast prices
          - quantiles: dict of {p10, p20, ..., p90} price paths
          - scenarios: dict of {worst, bad, base, good, best} endpoint % changes
          - current_price: float
          - implied_vol: float (annualized)
          - directional_alignment: float (-1.0 to +1.0)
          - directional_label: str (BULLISH/BEARISH/NEUTRAL)
          - horizon_days: int
        Or None if unavailable/stale.
        """
        if not _is_enabled():
            return None

        try:
            cache_key = f"forecast_{ticker}"
            cached = self._get_cached(cache_key, _TICKER_CACHE_TTL)
            if cached is not None:
                return cached

            raw = self._find_latest_forecast(ticker, "forecast_ticker")
            if raw is None:
                self._log_event("forecast_ticker_miss", {"ticker": ticker})
                return None

            # Shallow copy to avoid mutating vault data in-place
            data = dict(raw)

            # Enrich with derived label
            alignment = data.get("directional_alignment", 0.0)
            if isinstance(alignment, (int, float)):
                data["directional_label"] = _directional_label(alignment)
            else:
                data["directional_label"] = "NEUTRAL"

            self._set_cached(cache_key, data)
            self._log_event("forecast_ticker_hit", {
                "ticker": ticker,
                "horizon": data.get("horizon_days"),
                "alignment": data.get("directional_label"),
            })
            return data

        except Exception as e:
            logger.error("AthenaBridge: forecast_ticker(%s) failed: %s", ticker, e)
            return None

    def forecast_macro(
        self, indicators: Optional[List[str]] = None
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Get latest vault macro forecasts.

        Args:
            indicators: List of macro indicator names (e.g., ["DFF", "T10Y2Y"]).
                        If None, returns all available macro forecasts.

        Returns dict of {indicator_name: forecast_dict} or None.
        """
        if not _is_enabled():
            return None

        try:
            cache_key = "forecast_macro_all"
            cached = self._get_cached(cache_key, _MACRO_CACHE_TTL)
            if cached is not None:
                if indicators is not None:
                    return {k: v for k, v in cached.items() if k in indicators}
                return cached

            self._reload_vault_if_stale()

            # S170B P2-A: track per-indicator entry-root timestamp + apply
            # staleness filter (was missing, would inject month-old forecasts
            # if vault writer ever stops updating).
            macro_forecasts = {}
            macro_ts: Dict[str, str] = {}
            _now_utc = datetime.now(timezone.utc)
            for entry in self._vault_data:
                if entry.get("event_type") != "forecast_macro":
                    continue
                raw_data = entry.get("data", {})
                indicator = raw_data.get("ticker", "")
                if not indicator:
                    continue
                entry_ts = entry.get("timestamp", "") or raw_data.get("timestamp", "")
                if entry_ts <= macro_ts.get(indicator, ""):
                    continue
                # Staleness filter — same 24h gate as ticker forecasts
                try:
                    _dt = datetime.fromisoformat(entry_ts)
                    if _dt.tzinfo is None:
                        _dt = _dt.replace(tzinfo=timezone.utc)
                    if (_now_utc - _dt).total_seconds() > _FORECAST_STALENESS_THRESHOLD:
                        continue
                except (ValueError, TypeError):
                    pass  # unparseable ts: tolerate, don't reject
                data = dict(raw_data)
                alignment = data.get("directional_alignment", 0.0)
                if isinstance(alignment, (int, float)):
                    data["directional_label"] = _directional_label(alignment)
                else:
                    data["directional_label"] = "NEUTRAL"
                macro_ts[indicator] = entry_ts
                macro_forecasts[indicator] = data

            if not macro_forecasts:
                self._log_event("forecast_macro_miss", {})
                return None

            self._set_cached(cache_key, macro_forecasts)
            self._log_event("forecast_macro_hit", {
                "indicators": list(macro_forecasts.keys()),
                "count": len(macro_forecasts),
            })

            if indicators is not None:
                return {k: v for k, v in macro_forecasts.items() if k in indicators}
            return macro_forecasts

        except Exception as e:
            logger.error("AthenaBridge: forecast_macro() failed: %s", e)
            return None

    def check_exit_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Check if the latest forecast suggests an exit signal.

        Reads from vault entries with event_type "forecast_ticker" and
        derives exit signal from directional_alignment and quantile spread.

        Returns dict with:
          - signal: str ("EXIT" / "HOLD_MONITOR" / "HOLD")
          - directional_alignment: float
          - implied_vol: float
          - quantile_spread_pct: float (p90 - p10 endpoint % of current price)
        Or None if unavailable.
        """
        if not _is_enabled():
            return None

        try:
            forecast = self.forecast_ticker(ticker)
            if forecast is None:
                return None

            alignment = forecast.get("directional_alignment", 0.0)
            if not isinstance(alignment, (int, float)):
                alignment = 0.0

            implied_vol = forecast.get("implied_vol", 0.0)
            if not isinstance(implied_vol, (int, float)):
                implied_vol = 0.0

            # Compute quantile spread at endpoint
            quantiles = forecast.get("quantiles", {})
            p10 = quantiles.get("p10", [])
            p90 = quantiles.get("p90", [])
            current_price = forecast.get("current_price", 1.0)
            if not isinstance(current_price, (int, float)) or current_price == 0:
                current_price = 1.0

            spread_pct = 0.0
            if p10 and p90:
                spread_pct = (p90[-1] - p10[-1]) / current_price

            # Derive signal from alignment extremes
            if abs(alignment) > 0.8 and alignment < 0:
                signal = "EXIT"
            elif abs(alignment) > 0.5 and alignment < 0:
                signal = "HOLD_MONITOR"
            else:
                signal = "HOLD"

            result = {
                "signal": signal,
                "directional_alignment": alignment,
                "implied_vol": implied_vol,
                "quantile_spread_pct": spread_pct,
            }

            self._log_event("exit_signal_check", {
                "ticker": ticker,
                "signal": signal,
                "alignment": alignment,
            })
            return result

        except Exception as e:
            logger.error("AthenaBridge: check_exit_signal(%s) failed: %s", ticker, e)
            return None

    def get_scenario_paths(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get scenario paths (worst/bad/base/good/best) from vault.

        Returns dict with:
          - scenarios: dict of {worst, bad, base, good, best} endpoint % changes
          - quantiles: dict of full quantile paths
          - current_price: float
        Or None if unavailable.
        """
        if not _is_enabled():
            return None

        try:
            forecast = self.forecast_ticker(ticker)
            if forecast is None:
                return None

            scenarios = forecast.get("scenarios", {})
            quantiles = forecast.get("quantiles", {})
            if not scenarios and not quantiles:
                return None

            return {
                "scenarios": scenarios,
                "quantiles": quantiles,
                "current_price": forecast.get("current_price", 0.0),
                "horizon_days": forecast.get("horizon_days", 0),
            }

        except Exception as e:
            logger.error("AthenaBridge: get_scenario_paths(%s) failed: %s", ticker, e)
            return None

    def get_status(self) -> Dict[str, Any]:
        """Health check: vault accessibility, cache size, data freshness."""
        # S-P2-5: reload first, so a recent all-malformed vault is reflected here
        # instead of reporting whatever (possibly empty, pre-load) state happens
        # to be cached on this instance.
        self._reload_vault_if_stale()

        vault_exists = os.path.isfile(_FORECASTS_JSONL)
        vault_age_hours = None
        if vault_exists:
            try:
                mtime = os.path.getmtime(_FORECASTS_JSONL)
                vault_age_hours = (time.time() - mtime) / 3600
            except OSError:
                pass

        return {
            "available": _is_enabled() and vault_exists and not self._load_failed,
            "enabled": _is_enabled(),
            "log_only": _is_log_only(),
            "vault_exists": vault_exists,
            "vault_path": _FORECASTS_JSONL,
            "vault_entries": len(self._vault_data),
            "vault_age_hours": round(vault_age_hours, 1) if vault_age_hours is not None else None,
            "cached_forecasts": len(self._cache),
            "load_failed": self._load_failed,
            "malformed_drops": self._malformed_drop_count,
            "issues": self._get_issues(vault_exists, vault_age_hours),
        }

    def _get_issues(
        self, vault_exists: bool, vault_age_hours: Optional[float]
    ) -> List[str]:
        """Identify any health issues."""
        issues = []
        if not vault_exists:
            issues.append("Vault file not found")
        elif vault_age_hours is not None and vault_age_hours > 24:
            issues.append(f"Vault data stale ({vault_age_hours:.1f}h old)")
        if self._load_failed:
            issues.append("Vault load failed (all entries malformed or unreadable)")
        elif self._malformed_drop_count:
            issues.append(f"{self._malformed_drop_count} malformed JSONL line(s) dropped on last load")
        if not _is_enabled():
            issues.append("Bridge disabled (CLIO_ATHENA_ENABLED=0)")
        return issues


# ================================================================
# SINGLETON
# ================================================================

_bridge_instance: Optional[AthenaBridge] = None
_bridge_lock = threading.Lock()


def get_athena_bridge() -> AthenaBridge:
    """Get or create the singleton AthenaBridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        with _bridge_lock:
            if _bridge_instance is None:
                _bridge_instance = AthenaBridge()
    return _bridge_instance
