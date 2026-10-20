"""
CLIO flowcollector — configuration (env-driven, dataclass defaults MUST match
from_env() defaults -- CLAUDE.md coding standard).

Deliberately independent of `agent.dpof.config`: this package is a separate
long-running process with its own systemd unit (see
`systemd/clio-flowcollector.service`), not part of the DPOF engine's process
or config surface (INTEL_V3_SPEC.md: "SHADOW ONLY this phase ... NOT
consumed by DPOF yet"). Every env var is read fresh in `from_env()` --
runtime-evaluated, matching the CLIO_* flag family convention.

Alpaca credentials: reuses CLIO's existing shared PAPER data keys
(`ALPACA_PAPER_KEY_ID`/`ALPACA_PAPER_SECRET_KEY`) -- same keys `agent.dpof`
falls back to for a paper account (agent/dpof/broker.py's module docstring),
and the same account the operator confirmed carries the Algo Trader Plus
market-data entitlement this collector needs for OPRA/SIP. This module never
imports `agent.config` (kept import-light/independent, like
`agent.dpof.data`'s own module docstring requires of itself) -- it reads the
env vars directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("clio.flowcollector.config")

# agent/flowcollector/config.py -> 3x dirname = repo root (same depth as
# agent/integrations/*.py -- CLAUDE.md "dirname level counting").
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val.strip() == "" else val.strip()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(f"flowcollector config: invalid float for {name}={raw!r}; using default {default}")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"flowcollector config: invalid int for {name}={raw!r}; using default {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    items = tuple(s.strip().upper() for s in raw.split(",") if s.strip())
    return items or default


@dataclass(frozen=True)
class FlowCollectorConfig:
    # Universe
    underlyings: tuple[str, ...] = ("SPY", "QQQ")
    strike_band_pct: float = 0.04          # +/- 4% of spot
    max_dte: int = 2                        # 0/1/2 DTE in TRADING SESSIONS (Fri -> Fri/Mon/Tue)
    universe_refresh_seconds: float = 900.0     # 15 min periodic refresh
    universe_spot_drift_pct: float = 0.005      # 0.5% spot move also forces a refresh
    # Short retry cadence used INSTEAD OF universe_refresh_seconds while the
    # universe is empty (never fetched yet, or every underlying's fetch came
    # back with nothing) -- an empty universe means the option stream can't
    # start at all (see collector.py's _ensure_option_stream_started), so
    # waiting the full 15-min cadence to try again is far too slow.
    universe_empty_retry_seconds: float = 60.0
    symbol_cap_per_connection: int = 1000       # Algo Trader Plus documented cap (recon doc §1; UNVERIFIED whether quotes-only or combined trades+quotes)

    # Feeds (Algo Trader Plus entitles OPRA/SIP; DPOFBroker's sticky
    # downgrade-to-indicative/IEX pattern is mirrored, not imported, in
    # collector.py -- see its module docstring). stock_feed is "sip" or
    # "iex" ONLY -- alpaca-py's live StockDataStream raises ValueError for
    # anything else (it does not support "delayed_sip", a REST-only
    # concept); validate() flags any other value and
    # AlpacaFlowBroker._feed_for_stock normalizes defensively at the point
    # of use so a misconfigured env var can never repeatedly crash stream
    # construction.
    options_feed: str = "opra"
    stock_feed: str = "sip"

    # Alpaca credentials -- CLIO's shared PAPER data keys (see module docstring)
    alpaca_key_id: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True

    # Classification
    quote_epsilon: float = 0.005
    sweep_window_seconds: float = 0.5
    sweep_min_exchanges: int = 2

    nofa_scale_usd: float = 2_000_000.0
    premium_scale_usd: float = 4_000_000.0
    darkpool_scale_usd: float = 20_000_000.0
    gex_neutral_band: float = 0.05

    # Self-computed (Black-Scholes-Merton) greeks fallback -- used whenever
    # Alpaca's OPRA/indicative snapshot greeks are missing or fail
    # greeks.sanitize_greeks (2026-09-25 incident: Alpaca returned NO
    # greeks at all, all day, for every symbol -- see greeks.py's module
    # docstring and collector.py's maybe_refresh_strikes/
    # maybe_recompute_greeks). risk_free_rate/dividend_yield are
    # continuously-compounded ANNUAL rates; dividend_yield defaults to a
    # single flat value (not per-underlying) -- SPY/QQQ's trailing 12m
    # dividend yields both round to roughly this figure as of this pass
    # (UNVERIFIED against a live feed -- chosen, not measured, like every
    # other scale constant in this package; override per CLAUDE.md's
    # BUDGET_*-style env-override convention if it drifts).
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.012
    # Cadence (seconds) for the LIGHTER, network-free greeks recompute pass
    # (collector.py's maybe_recompute_greeks) that keeps self-computed
    # deltas/gammas tracking live NBBO moves between the much slower
    # strikes_refresh_seconds (OI + Alpaca-greeks REST) pulls -- see that
    # method's docstring.
    greeks_recompute_seconds: float = 60.0
    # Short retry cadence used INSTEAD OF strikes_refresh_seconds when a
    # strikes refresh is due but this underlying has no spot yet (startup,
    # before the first stock trade/quote arrives) -- see collector.py's
    # maybe_refresh_strikes docstring for why the expensive OI/Alpaca-greeks
    # REST pull is skipped entirely rather than burned on a cycle that can't
    # self-compute/interpolate anything without a spot. Same "empty universe
    # retries fast" shape as universe_empty_retry_seconds above.
    strikes_spot_retry_seconds: float = 30.0
    # "No fake zeros" coverage gates (aggregator.compute_gex /
    # Aggregator.snapshot's NOPE gating): below these fractions, gex/nope
    # report None (unavailable) rather than a computation dominated by
    # unknown-gamma/unknown-delta contracts silently treated as zero.
    gex_min_coverage: float = 0.5
    nope_min_coverage: float = 0.5
    # Relative bid/ask spread ((ask-bid)/mid) above which a quote is too
    # wide to trust for an implied-vol solve (greeks.compute_contract_greeks).
    max_spread_pct_for_iv: float = 0.5

    # Cadence
    aggregation_interval_seconds: float = 15.0     # state file write cadence
    comparison_log_interval_seconds: float = 300.0  # 5 min
    strikes_refresh_seconds: float = 900.0          # OI/greeks re-pull cadence (still EOD-stale OI -- see recon doc §1)
    tick_seconds: float = 2.0                       # main loop poll interval

    # Market hours idling (agent.integrations.market_events holidays/early
    # close are reused -- see collector.py)
    pre_open_minutes: float = 5.0   # active from 09:25 ET
    post_close_minutes: float = 5.0  # active until 16:05 ET

    # Backpressure / resource guards
    inbound_queue_maxsize: int = 20_000
    reconnect_min_backoff_seconds: float = 1.0
    reconnect_max_backoff_seconds: float = 30.0
    reconnect_max_consecutive_failures: int = 0  # 0 = unbounded retries (never gives up)

    # State + comparison log paths (env-overridable per builder contract)
    state_file: str = field(default_factory=lambda: os.path.join(_REPO_ROOT, "data", "intelligence", "selfflow_state.json"))
    compare_file: str = field(default_factory=lambda: os.path.join(_REPO_ROOT, "data", "intelligence", "selfflow_compare.jsonl"))

    # Where CLIO's own UW-flow bridge export lives (agent.integrations.
    # uw_flow_intel._write_bridge_signals) -- read-only comparison source,
    # see state.py's docstring for why this is the only UW pillar snapshot
    # ever found on disk (grep-verified against agent/dpof/*.py).
    tradingbot_dir: str = "/home/agent/tradingbot"

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.underlyings:
            issues.append("underlyings is empty")
        if not (0.0 < self.strike_band_pct < 1.0):
            issues.append(f"strike_band_pct out of range: {self.strike_band_pct}")
        if self.max_dte < 0:
            issues.append(f"max_dte negative: {self.max_dte}")
        if self.symbol_cap_per_connection <= 0:
            issues.append(f"symbol_cap_per_connection must be positive: {self.symbol_cap_per_connection}")
        if self.options_feed not in ("opra", "indicative"):
            issues.append(f"unknown options_feed: {self.options_feed!r}")
        if self.stock_feed not in ("sip", "iex"):
            # "delayed_sip" is a REST-only DataFeed value; alpaca-py's live
            # StockDataStream.__init__ raises ValueError for anything other
            # than IEX/SIP (probed: DataFeed.DELAYED_SIP -> "only IEX and
            # SIP feeds ar supported") -- never a valid stock_feed here.
            issues.append(f"unsupported stock_feed for the live stream (must be 'sip' or 'iex'): {self.stock_feed!r}")
        if self.universe_empty_retry_seconds <= 0:
            issues.append(f"universe_empty_retry_seconds must be positive: {self.universe_empty_retry_seconds}")
        if self.greeks_recompute_seconds <= 0:
            issues.append(f"greeks_recompute_seconds must be positive: {self.greeks_recompute_seconds}")
        if self.strikes_spot_retry_seconds <= 0:
            issues.append(f"strikes_spot_retry_seconds must be positive: {self.strikes_spot_retry_seconds}")
        if not (0.0 <= self.gex_min_coverage <= 1.0):
            issues.append(f"gex_min_coverage out of range: {self.gex_min_coverage}")
        if not (0.0 <= self.nope_min_coverage <= 1.0):
            issues.append(f"nope_min_coverage out of range: {self.nope_min_coverage}")
        if self.max_spread_pct_for_iv <= 0:
            issues.append(f"max_spread_pct_for_iv must be positive: {self.max_spread_pct_for_iv}")
        return issues


def load_config() -> FlowCollectorConfig:
    default = FlowCollectorConfig()
    return FlowCollectorConfig(
        underlyings=_env_list("CLIO_FLOWCOLLECTOR_UNDERLYINGS", default.underlyings),
        strike_band_pct=_env_float("CLIO_FLOWCOLLECTOR_STRIKE_BAND_PCT", default.strike_band_pct),
        max_dte=_env_int("CLIO_FLOWCOLLECTOR_MAX_DTE", default.max_dte),
        universe_refresh_seconds=_env_float("CLIO_FLOWCOLLECTOR_UNIVERSE_REFRESH_SECONDS", default.universe_refresh_seconds),
        universe_spot_drift_pct=_env_float("CLIO_FLOWCOLLECTOR_UNIVERSE_DRIFT_PCT", default.universe_spot_drift_pct),
        universe_empty_retry_seconds=_env_float("CLIO_FLOWCOLLECTOR_UNIVERSE_EMPTY_RETRY_SECONDS", default.universe_empty_retry_seconds),
        symbol_cap_per_connection=_env_int("CLIO_FLOWCOLLECTOR_SYMBOL_CAP", default.symbol_cap_per_connection),
        options_feed=_env_str("CLIO_FLOWCOLLECTOR_OPTIONS_FEED", default.options_feed),
        stock_feed=_env_str("CLIO_FLOWCOLLECTOR_STOCK_FEED", default.stock_feed),
        alpaca_key_id=_env_str("ALPACA_PAPER_KEY_ID", default.alpaca_key_id),
        alpaca_secret_key=_env_str("ALPACA_PAPER_SECRET_KEY", default.alpaca_secret_key),
        alpaca_paper=_env_bool("ALPACA_PAPER", default.alpaca_paper),
        quote_epsilon=_env_float("CLIO_FLOWCOLLECTOR_QUOTE_EPSILON", default.quote_epsilon),
        sweep_window_seconds=_env_float("CLIO_FLOWCOLLECTOR_SWEEP_WINDOW_SECONDS", default.sweep_window_seconds),
        sweep_min_exchanges=_env_int("CLIO_FLOWCOLLECTOR_SWEEP_MIN_EXCHANGES", default.sweep_min_exchanges),
        nofa_scale_usd=_env_float("CLIO_FLOWCOLLECTOR_NOFA_SCALE_USD", default.nofa_scale_usd),
        premium_scale_usd=_env_float("CLIO_FLOWCOLLECTOR_PREMIUM_SCALE_USD", default.premium_scale_usd),
        darkpool_scale_usd=_env_float("CLIO_FLOWCOLLECTOR_DARKPOOL_SCALE_USD", default.darkpool_scale_usd),
        gex_neutral_band=_env_float("CLIO_FLOWCOLLECTOR_GEX_NEUTRAL_BAND", default.gex_neutral_band),
        risk_free_rate=_env_float("CLIO_FLOWCOLLECTOR_RISK_FREE_RATE", default.risk_free_rate),
        dividend_yield=_env_float("CLIO_FLOWCOLLECTOR_DIVIDEND_YIELD", default.dividend_yield),
        greeks_recompute_seconds=_env_float("CLIO_FLOWCOLLECTOR_GREEKS_RECOMPUTE_SECONDS", default.greeks_recompute_seconds),
        strikes_spot_retry_seconds=_env_float("CLIO_FLOWCOLLECTOR_STRIKES_SPOT_RETRY_SECONDS", default.strikes_spot_retry_seconds),
        gex_min_coverage=_env_float("CLIO_FLOWCOLLECTOR_GEX_MIN_COVERAGE", default.gex_min_coverage),
        nope_min_coverage=_env_float("CLIO_FLOWCOLLECTOR_NOPE_MIN_COVERAGE", default.nope_min_coverage),
        max_spread_pct_for_iv=_env_float("CLIO_FLOWCOLLECTOR_MAX_SPREAD_PCT_FOR_IV", default.max_spread_pct_for_iv),
        aggregation_interval_seconds=_env_float("CLIO_FLOWCOLLECTOR_AGGREGATION_INTERVAL_SECONDS", default.aggregation_interval_seconds),
        comparison_log_interval_seconds=_env_float("CLIO_FLOWCOLLECTOR_COMPARE_INTERVAL_SECONDS", default.comparison_log_interval_seconds),
        strikes_refresh_seconds=_env_float("CLIO_FLOWCOLLECTOR_STRIKES_REFRESH_SECONDS", default.strikes_refresh_seconds),
        tick_seconds=_env_float("CLIO_FLOWCOLLECTOR_TICK_SECONDS", default.tick_seconds),
        pre_open_minutes=_env_float("CLIO_FLOWCOLLECTOR_PRE_OPEN_MINUTES", default.pre_open_minutes),
        post_close_minutes=_env_float("CLIO_FLOWCOLLECTOR_POST_CLOSE_MINUTES", default.post_close_minutes),
        inbound_queue_maxsize=_env_int("CLIO_FLOWCOLLECTOR_QUEUE_MAXSIZE", default.inbound_queue_maxsize),
        reconnect_min_backoff_seconds=_env_float("CLIO_FLOWCOLLECTOR_RECONNECT_MIN_BACKOFF", default.reconnect_min_backoff_seconds),
        reconnect_max_backoff_seconds=_env_float("CLIO_FLOWCOLLECTOR_RECONNECT_MAX_BACKOFF", default.reconnect_max_backoff_seconds),
        reconnect_max_consecutive_failures=_env_int("CLIO_FLOWCOLLECTOR_RECONNECT_MAX_FAILURES", default.reconnect_max_consecutive_failures),
        state_file=_env_str("CLIO_SELFFLOW_FILE", default.state_file),
        compare_file=_env_str("CLIO_SELFFLOW_COMPARE_FILE", default.compare_file),
        tradingbot_dir=_env_str("TRADINGBOT_DIR", default.tradingbot_dir),
    )
