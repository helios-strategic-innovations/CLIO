"""
CLIO DPOF — single source of truth for every tunable.

Hydrinia's DPOF split its config across two files and they drifted
(DPOF_MAX_POSITIONS=2 vs dpof_max_concurrent_positions=3, a dead
options_target_delta, two sizing formulas). Here there is exactly ONE
dataclass; `from_env()` is the only constructor used at runtime, and the
dataclass defaults are the values `from_env()` falls back to (both sites
must agree — CLAUDE.md coding standard).

Env is re-read on every `load_dpof_config()` call (runtime-evaluated, like
the CLIO_* flag family) so a `.env` edit + restart is the only deploy step.

Defaults are sized for a ~$20K single-purpose Alpaca account trading long
SPY/QQQ options. Operator directive (2026-09-24): "paper mode IS the soak" —
there is no separate log-only staging period, and "everything is built
active, not shadow" — every default is the ACTIVE mode. `mode` therefore
defaults to PAPER (the engine trades a paper account from the first cycle
unless CLIO_DPOF_MODE is explicitly set to off/log_only) and `jev_mode`
defaults to VETO (inert until a TypeSafe key is configured — see
`jev_effective_mode`). LIVE is unaffected: every gate in broker.py §7 still
applies regardless of this default, and nothing here weakens them.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field, fields
from datetime import time as _time
from enum import Enum
from typing import Optional

logger = logging.getLogger("clio.dpof.config")


def _parse_hhmm(s: str) -> Optional[_time]:
    """Parse "HH:MM" into a naive time, or None if it isn't one. Mirrors
    agent.dpof.risk.parse_hhmm's exact parsing (split(":") + int()) — kept as
    a separate, dependency-free copy here (not imported) because risk.py
    imports DPOFConfig, so config.py importing risk.py back would be
    circular. Used only by validate() below; never raises."""
    try:
        hh, mm = s.split(":")
        return _time(hour=int(hh), minute=int(mm))
    except (ValueError, AttributeError, TypeError):
        return None

# agent/dpof/config.py -> 3x dirname = repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "data", "dpof")

# Exact phrase the operator must set to arm LIVE mode. Deliberately long and
# unambiguous so it can't be satisfied by a stray "1"/"true".
LIVE_ACK_PHRASE = "I_ACCEPT_REAL_MONEY_RISK"


class DpofMode(str, Enum):
    OFF = "off"            # engine does nothing (default)
    LOG_ONLY = "log_only"  # full pipeline, decisions + would-be orders logged, NO orders
    PAPER = "paper"        # orders to a PAPER Alpaca account
    LIVE = "live"          # orders to a LIVE account — requires every live gate


class JevMode(str, Enum):
    OFF = "off"        # Jev never called
    SHADOW = "shadow"  # Jev called + logged, verdict ignored
    VETO = "veto"      # Jev may BLOCK an entry; can never initiate, upsize, or gate an exit


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
        logger.warning(f"DPOF config: invalid float for {name}={raw!r}; using default {default}")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"DPOF config: invalid int for {name}={raw!r}; using default {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int_clamped(name: str, default: int, lo: int, hi: int) -> int:
    """Same as _env_int, then clamps the result into [lo, hi] (warning on
    out-of-range so a bad .env value degrades to the nearest valid bound
    instead of silently being used as-is)."""
    val = _env_int(name, default)
    if val < lo or val > hi:
        logger.warning(f"DPOF config: {name}={val} out of range [{lo},{hi}]; clamping")
        val = max(lo, min(hi, val))
    return val


def _claim_check_mode(default: str) -> str:
    """CLIO_DPOF_CLAIM_CHECK: "log" or "off". Anything else degrades to "log" (never an enforce mode)."""
    raw = _env_str("CLIO_DPOF_CLAIM_CHECK", default).lower()
    if raw in ("log", "off"):
        return raw
    logger.warning(f"DPOF config: CLIO_DPOF_CLAIM_CHECK={raw!r} not in (log, off); using log")
    return "log"


def _env_kronos_pillar(default: str) -> str:
    raw = _env_str("CLIO_DPOF_KRONOS_PILLAR", default).lower()
    if raw not in ("live", "off"):
        logger.warning(f"DPOF config: CLIO_DPOF_KRONOS_PILLAR={raw!r} is not live/off; using off")
        return "off"
    return raw


def _env_kronos_horizon(default: int) -> int:
    val = _env_int("CLIO_DPOF_KRONOS_HORIZON", default)
    if val not in (15, 30, 60):
        logger.warning(f"DPOF config: CLIO_DPOF_KRONOS_HORIZON={val} is not 15/30/60; using {default}")
        return default
    return val


_DW_MODES = ("off", "shadow", "live")


def _norm_dw_mode(raw) -> str:
    """Normalised dynamic-weights mode; anything unknown (or non-string) is "off"."""
    if not isinstance(raw, str):
        return "off"
    mode = raw.strip().lower()
    return mode if mode in _DW_MODES else "off"


def dynamic_weights_mode(cfg) -> str:
    """Normalised dynamic-weights mode for ANY DPOFConfig instance (bad value => "off")."""
    return _norm_dw_mode(getattr(cfg, "dynamic_weights", "off"))


def _env_dw_mode(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":  # blank == unset, like the sibling helpers
        return _norm_dw_mode(default)
    mode = _norm_dw_mode(raw)
    if mode != raw.strip().lower():
        logger.warning(f"DPOF config: {name}={raw!r} is not off/shadow/live; using off")
    return mode


def _env_dw_eps(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning(f"DPOF config: invalid float for {name}={raw!r}; using default {default}")
        return default
    if not math.isfinite(val):
        logger.warning(f"DPOF config: non-finite {name}={raw!r}; using default {default}")
        return default
    return max(0.0005, min(0.05, val))


def _env_anchor_share(name: str, default: float) -> float:
    """A share of equity in (0, 0.5]. A DPOF config validate() problem stops the whole cycle, so a typo is normalised
    here (back to the default), never left for validate() to catch."""
    val = _env_float(name, default)
    if not (0.0 < val <= 0.5):   # NaN fails every comparison, so it lands here too
        logger.warning(f"DPOF config: {name}={val} is outside (0, 0.5]; using {default}")
        return default
    return val


def _env_anchor_shares(d: "DPOFConfig") -> tuple[float, float]:
    c50 = _env_anchor_share("CLIO_DPOF_ANCHOR_SHARE_C50", d.size_anchor_share_c50)
    c80 = _env_anchor_share("CLIO_DPOF_ANCHOR_SHARE_C80", d.size_anchor_share_c80)
    if c80 < c50:
        logger.warning(f"DPOF config: anchor share at 0.80 ({c80}) is below the share at 0.50 ({c50}); using the defaults")
        return d.size_anchor_share_c50, d.size_anchor_share_c80
    return c50, c80


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    items = tuple(s.strip().upper() for s in raw.split(",") if s.strip())
    return items or default


def _env_model_or_off(name: str, default: str) -> str:
    """Model id from env; "off"/"none"/"0" disables it (returns "")."""
    val = _env_str(name, default)
    return "" if val.lower() in ("off", "none", "0") else val


def _env_enum(name: str, enum_cls, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return enum_cls(raw.strip().lower())
    except ValueError:
        logger.warning(f"DPOF config: invalid {name}={raw!r}; using default {default.value}")
        return default


@dataclass(frozen=True)
class DPOFConfig:
    # ── Mode / identity ────────────────────────────────────────────────
    mode: DpofMode = DpofMode.PAPER
    underlyings: tuple[str, ...] = ("SPY", "QQQ")   # tradeable; first = primary
    data_dir: str = DEFAULT_DATA_DIR
    client_order_prefix: str = "DPOF"               # every DPOF order id starts with this

    # ── Alpaca account (dedicated preferred; falls back to CLIO's shared ──
    # paper keys when unset and alpaca_paper=True — see load_dpof_config()) ─
    alpaca_key_id: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True                       # CLIO_DPOF_ALPACA_PAPER
    # "dedicated" | "clio_paper" | "none" — which source alpaca_key_id/_secret_key
    # actually resolved from (set by load_dpof_config(); not a secret itself).
    alpaca_key_source: str = "none"
    # "opra" (default — operator confirmed Algo Trader Plus / real-time OPRA on
    # the account) | "auto" (try OPRA, fall back to indicative for the rest of
    # the session on a 401/403) | "indicative" (explicit, no OPRA attempt).
    # Feed selection/fallback itself lives in broker.py/data.py, not here.
    options_feed: str = "opra"
    stock_feed: str = "sip"                          # Algo Trader Plus includes full SIP
    # Virtual capital for DPOF's slice of a SHARED paper account (which may hold
    # other, non-DPOF positions/balance). > 0: sizing and risk breakers use
    # min(real account equity, capital_base + DPOF's own realized+unrealized P&L)
    # instead of the whole account's equity — see engine._dpof_equity(). 0 disables
    # this and uses the real account equity unchanged (correct for a dedicated
    # account). CLIO_DPOF_CAPITAL.
    capital_base: float = 20_000.0

    # ── Live gates (ALL must pass or the engine degrades to LOG_ONLY) ───
    live_ack: str = ""                              # must equal LIVE_ACK_PHRASE
    live_account_number: str = ""                   # must equal broker account_number
    min_options_level: int = 2                      # 2 = long calls/puts; verticals need 3
    min_account_equity: float = 2_000.0

    # ── Cadence / windows (ET) ──────────────────────────────────────────
    cycle_seconds: int = 30
    uw_refresh_seconds: int = 180
    # Dynamic pillar weights (docs/design/clio_dpof_dynamic_weights.md). shadow =
    # computed + logged beside static, decisions use static; live = decisions use
    # dynamic; off = today's path. Invalid values normalise to off (see loader).
    dynamic_weights: str = "shadow"  # CLIO_DPOF_DYNAMIC_WEIGHTS: off | shadow | live
    dw_ring_size: int = 40  # CLIO_DPOF_DW_RING, clamped [10, 200]
    dw_dead_eps: float = 0.005  # CLIO_DPOF_DW_DEAD_EPS, clamped [0.0005, 0.05]
    dw_quality_apply: bool = False  # CLIO_DPOF_DW_QUALITY_APPLY
    # Darkpool pillar: NBBO-midpoint signed ratio (agent/dpof/darkpool_nbbo.py)
    # instead of the legacy VWAP-vs-median bias (~+-0.002, effectively dead).
    # Default OFF: uncalibrated on regular-session prints; flag-off logs a
    # shadow INFO line comparing both readings.
    darkpool_nbbo: bool = False  # CLIO_DPOF_DARKPOOL_NBBO
    # Legacy (non-NBBO) darkpool treated as UNAVAILABLE when flat (|v| < 1e-3)
    # and excluding it keeps count_available >= min_pillars (signals.py guard).
    # Default OFF = legacy composite, byte-identical. Non-truthy values -> off.
    darkpool_flat_unavailable: bool = False  # CLIO_DPOF_DARKPOOL_FLAT_UNAVAILABLE
    # Empty / schema-drifted UW pillar responses become UNAVAILABLE (follow the
    # failed-fetch path, so the self-flow fallback can take over) instead of a
    # stored fake 0.0 reading. Default OFF = legacy behaviour.
    empty_pillar_unavailable: bool = False  # CLIO_DPOF_EMPTY_PILLAR_UNAVAILABLE
    bars_refresh_seconds: int = 60
    entry_start_et: str = "09:45"
    entry_end_et: str = "15:15"
    flatten_et: str = "15:50"                       # all DPOF positions closed by here
    flatten_eod: bool = True

    # ── Signal (port of Hydrinia NOFA 4-pillar composite) ──────────────
    w_nofa: float = 0.40
    w_premium: float = 0.25
    w_darkpool: float = 0.20
    w_gex: float = 0.15
    min_pillars: int = 3                            # NOFA must be one of them
    threshold_normal: float = 0.15
    threshold_strong: float = 0.30
    dp_veto_threshold: float = 0.30                 # DP contradicting composite beyond this -> veto
    require_price_confirmation: bool = True         # VWAP side + 15m momentum must agree
    qqq_divergence_ratio: float = 1.5               # pick QQQ if |NOFA_qqq| >= ratio*|NOFA_spy| and agrees
    exit_threshold: float = 0.05                    # thesis-invalidation: composite crossed this vs position

    # ── Regime (VIX) ────────────────────────────────────────────────────
    vix_trending_below: float = 18.0
    vix_choppy_at: float = 25.0                     # CHOPPY requires STRONG composite
    # Former "no new entries at/above" HALT threshold. Operator directive
    # (Task 1, 2026-09-25): the 4 former hard entry blocks (econ blackout,
    # breaking news, VIX halt, blocked vol regimes) became DECISION CONTEXT,
    # never a veto -- see docs/design/clio_dpof.md §4/Task1. This threshold
    # now only decides the REGIME LABEL shown to the decider ("EXTREME" at/
    # above it); classify_regime()/signals.py still requires strength=="strong"
    # to nominate at/above it (same rule CHOPPY already had), it just never
    # vetoes outright any more.
    vix_extreme_label_at: float = 35.0
    vix_max_age_seconds: int = 900

    # ── Event / regime advisory labels (Task 1, 2026-09-25): these NEVER
    # block a nomination or entry any more -- see docs/design/clio_dpof.md
    # §4/Task1. `block_on_breaking_news` and `block_on_vol_regimes` are
    # RETIRED (nothing reads them any more: breaking-news and vol-regime
    # info always surfaces as decision context now, unconditionally). DPOFConfig
    # is built fresh from env every load_dpof_config() call (never deserialized
    # from an old persisted blob), so there is no old-config migration concern.
    event_context_window_minutes: int = 30          # importance-3 events within +/- N min surface as context
    data_stale_alarm_minutes: int = 30              # alarm if no valid snapshot for this long in session

    # ── Contract selection (1DTE standard; see expiry_sessions_ahead below) ──
    # The v1 5-14 DTE window (min_dte/max_dte) and the theta/day filter
    # (max_theta_pct_per_day) are RETIRED (operator directive 2026-09-24):
    # theta is decision CONTEXT for a 1DTE contract, never a mechanical gate.
    target_delta: float = 0.50
    min_delta: float = 0.35
    max_delta: float = 0.65
    max_spread_pct: float = 0.06                    # (ask-bid)/mid
    min_open_interest: int = 500

    # ── Verticals (debit only) ──────────────────────────────────────────
    # Default True (operator confirmed options level 3 on the account,
    # 2026-09-24). Gated at RUNTIME by the broker-reported account options
    # level (>= 3), not by config validation -- see engine.py, which offers
    # verticals to choose_candidates only when the account actually supports
    # them (level < 3 -> singles only, no error; see min_options_level below).
    allow_verticals: bool = True
    vertical_when_vix_at: float = 22.0              # prefer a debit vertical at/above this VIX
    vertical_width_pct: float = 0.015               # short strike ~1.5% of spot beyond long

    # Sizing / risk envelope / exits: operator directive (no-hard-caps
    # amendment, 2026-09-24) removed risk_pct_min/max, max_trade_premium_pct,
    # max_open_premium_pct, max_contracts, max_concurrent_positions,
    # max_entries_per_day, max_entries_per_symbol_day, reentry_cooldown_minutes,
    # loss_streak_limit/_cooldown_minutes, daily/weekly_loss_limit_pct,
    # max_drawdown_pct, stop_loss_pct, tp_tiers, breakeven_after_first_tier,
    # trail_giveback, max_hold_minutes, thesis_exit_votes — entries AND exits
    # are judged by CLIO's decision model now; the only code-side size bound
    # left is SOLVENCY (sizing.py's max_affordable_qty/clamp_to_solvency) and
    # the only exit rail left is the 15:50 ET forced flatten (exits.py).

    # ── Order handling ─────────────────────────────────────────────────
    entry_lean: float = 0.25                        # limit = mid + lean*spread
    entry_reprice_seconds: int = 45
    entry_max_reprices: int = 2
    entry_max_slippage_pct: float = 0.05            # never pay more than mid*(1+this)
    exit_lean: float = 0.25                         # limit = mid - lean*spread
    exit_reprice_seconds: int = 20
    exit_max_reprices: int = 3                      # then marketable (bid) / market for urgent

    # ── Reconciliation quantity check (Task ORCH-7Q4K-DPOF builder B1,
    # 2026-09-26 -- see agent/dpof/orders.py's reconcile(), docs/design/
    # clio_dpof_followups_2026-09-26.md §3). Purely a ledger-bookkeeping
    # correction (single-leg long positions only, via a synthetic order_fills
    # entry, avg=None) -- never places, sizes or cancels an order, and never
    # widens/narrows the presence-only reconciled_absent/orphan checks above.
    reconcile_adopt_qty: bool = True                # CLIO_DPOF_RECONCILE_ADOPT -- False: warn on a mismatch, never correct the ledger
    reconcile_confirm_seconds: float = 30.0         # CLIO_DPOF_RECONCILE_CONFIRM_SECONDS -- same (ledger,broker) pair must persist this long before adopting

    # ── UW budget share ────────────────────────────────────────────────
    uw_daily_call_cap: int = 1500                   # DPOF's own cap inside CLIO's shared UW budget

    # ── Self-computed flow fallback (agent/flowcollector, operator directive
    # 2026-09-25: UW subscription/key reissue in progress, ~Tuesday) ───────
    # When a UW-derived flow pillar (or NOPE/max-pain context) is unavailable/
    # stale THIS CYCLE, data.py falls back to agent.flowcollector.state.
    # get_selfflow() (Alpaca OPRA/SIP self-computed pillars) instead of
    # leaving it missing. UW always wins whenever it IS available -- this
    # never blends the two sources for the same pillar. See
    # docs/design/clio_dpof.md's "Flow fallback" section for the exact
    # per-pillar mapping. Paper-only in practice (DPOF is paper by default;
    # this is part of the soak) -- the flag itself doesn't gate mode.
    selfflow_fallback: bool = True                  # CLIO_DPOF_SELFFLOW_FALLBACK
    selfflow_max_age_seconds: int = 120              # CLIO_DPOF_SELFFLOW_MAX_AGE
    # Fewer classified trades than this in selfflow's 5-min window (options for
    # nofa/premium, D-tape prints for darkpool) is "thin": a 0.0 reading, not a
    # ratio — one or two trades would otherwise read as a full +/-1.0.
    selfflow_min_trades: int = 20                    # CLIO_DPOF_SELFFLOW_MIN_TRADES

    # ── Kronos advisory context (Task K2, agent/kronos — a separate CLIO
    # service; see docs/design/clio_kronos.md §4). Advisory-only technical
    # input for the decider (entries and reviews) plus the signal log; the
    # composite, Jev state, and every order path are unchanged (K2 ruling).
    # kronos_context off -> the "kronos" ctx key is absent entirely from the
    # decider prompt (never shown as "disabled"); the signal-log "kronos"
    # field is still always present, null when unavailable.
    kronos_context: bool = True                      # CLIO_DPOF_KRONOS_CONTEXT
    kronos_max_age_seconds: int = 600                 # CLIO_DPOF_KRONOS_MAX_AGE, clamped [120, 1800]
    # ── Kronos LIVE pillar (Dom, 2026-09-26: "We build Kronos for live use";
    # docs/design/clio_kronos.md §4b). A directional composite pillar =
    # 2*p_up - 1 at `kronos_horizon` minutes from a FRESH forecast only;
    # weighed beside nofa/premium/darkpool (compute_composite is a weighted
    # average over the available directional pillars, so w_kronos is NOT part
    # of the four-weight sum). Never counts toward min_pillars. Bad env values
    # are normalized in load_dpof_config (off / 30 / clamped) -- a config
    # problem would stop the whole cycle.
    kronos_pillar: str = "live"                       # CLIO_DPOF_KRONOS_PILLAR: live | off
    w_kronos: float = 0.15                            # CLIO_DPOF_W_KRONOS, clamped [0, 1]
    kronos_horizon: int = 30                          # CLIO_DPOF_KRONOS_HORIZON: 15 | 30 | 60

    # ── Jev (TypeSafe) fast decision layer ──────────────────────────────
    jev_mode: JevMode = JevMode.VETO
    jev_api_key: str = ""                           # TYPESAFE_API_KEY (SDK-standard name)
    # Optional alternative to TYPESAFE_API_KEY in .env: path to an owner-only (0600, no symlink)
    # file holding CLIO's OWN TypeSafe key (CLIO does not share a key with other consumers).
    # Takes precedence when set. moltbot.service has ProtectHome=tmpfs, so the file must live
    # under a bound path (e.g. /home/agent/openclaw/data/secrets/, gitignored).
    jev_key_file: str = ""
    jev_base_url: str = "https://api.typesafe.ai"   # TYPESAFE_BASE_URL
    jev_model: str = "jev-1.13.0"                   # pinned: thresholds are tuned per version
    jev_timeout_seconds: float = 2.5
    jev_headline_conflict_threshold: float = 0.70
    jev_event_risk_threshold: float = 0.70
    jev_setup_veto_confidence: float = 0.60
    jev_fail_closed: bool = True                    # VETO mode: Jev error/timeout blocks the ENTRY (never exits)
    jev_circuit_failures: int = 3
    jev_circuit_cooldown_seconds: int = 600
    jev_max_headlines: int = 12
    # Optional wire-compatible fallback (e.g. a self-hosted server, which serves
    # the same POST /v1/systemone contract). Tried once, inside the same deadline, only
    # when the primary errors/times out. Empty = no fallback.
    jev_fallback_base_url: str = ""
    jev_fallback_model: str = ""
    jev_fallback_api_key: str = ""

    # ── Judgment layer (operator 2026-09-24: NO mechanical SL/TP) ────────
    # Entries AND exits are decided by CLIO's decision model; the flow signal
    # only nominates a candidate and code only enforces hard size caps and the
    # 15:50 ET flatten. Primary: Anthropic Opus 5.5; fallback: OpenAI GPT-6 Sol.
    # Both down -> no entries, open positions HELD, operator alerted.
    decision_model: str = "claude-opus-5-5"         # CLIO_DPOF_DECISION_MODEL
    decision_effort: str = "medium"                 # Opus 5.5 output_config.effort (thinking is always on)
    decision_timeout_seconds: float = 60.0
    decision_max_tokens: int = 16000                # thinking + JSON answer
    fallback_model: str = "gpt-6-sol"               # CLIO_DPOF_FALLBACK_MODEL ("" disables the fallback)
    fallback_effort: str = "medium"                 # OpenAI reasoning.effort
    fallback_timeout_seconds: float = 60.0
    openai_api_key: str = ""                        # OPENAI_API_KEY (shared with CLIO's Whisper)
    openai_base_url: str = "https://api.openai.com/v1"
    decider_circuit_failures: int = 3               # consecutive failures before a provider is skipped
    decider_circuit_cooldown_seconds: int = 300
    decider_sdk_max_retries: int = 0                # CLIO_DPOF_DECISION_SDK_RETRIES: Anthropic SDK-internal retries (0..3)
    attempt_log_enabled: bool = True                # CLIO_DPOF_ATTEMPT_LOG: per-attempt decider ledger (decider_attempts.jsonl)
    attempt_log_max_lines: int = 20000              # CLIO_DPOF_ATTEMPT_LOG_MAX_LINES (clamped 100..200000)
    claim_check_mode: str = "log"                   # CLIO_DPOF_CLAIM_CHECK: "log" (violations logged + ledgered) | "off"; no enforce mode
    max_decisions_per_day: int = 600                # cost guard: entry deliberations + reviews
    decision_max_age_seconds: int = 120             # a finished decision older than this is discarded
    # Position reviews
    review_interval_minutes: int = 5                # default cadence per open position
    review_min_minutes: int = 1                     # decider may ask for sooner...
    review_max_minutes: int = 15                    # ...or later, within these bounds
    review_trigger_move_pct: float = 0.15           # |mark move| since last review that pulls a review forward
    model_down_alert_minutes: int = 15              # re-alert cadence while no decider is reachable
    # Entry deliberations
    entry_candidates: int = 3                       # contracts offered to the decider per deliberation
    entry_deliberation_cooldown_seconds: int = 300  # per underlying+direction, unless the signal strengthens
    entry_redeliberate_delta: float = 0.10          # composite strengthening that bypasses the cooldown
    # Expiry policy (Hydrinia DPOF design intent: 1DTE standard)
    expiry_sessions_ahead: int = 1                  # 1 = next trading session's expiry (Fri -> Mon)
    max_expiry_sessions_ahead: int = 2              # fallback when the target expiry has no liquid contracts
    # Jev as the fast exit SENTINEL: it can only pull a decider review forward
    jev_escalate_threshold: float = 0.65
    # Jev entry-veto sizing question (operator directive, no-hard-caps
    # amendment): a validated `size_reckless` noul >= this threshold adds
    # "size_reckless" to the entry veto reasons (VETO blocks; SHADOW logs
    # would_veto). jev.py reads this via getattr(cfg, "jev_size_veto_threshold",
    # 0.70) so it degrades safely even before this field existed.
    jev_size_veto_threshold: float = 0.70

    # ── Observability (Task ORCH-7Q4K-DPOF, 2026-09-25): signals.jsonl +
    # /dpof why + the EOD digest. Purely additive -- never read by any gate/
    # decision path. See agent/dpof/signal_log.py and docs/design/clio_dpof.md.
    signal_log_max_lines: int = 20_000       # CLIO_DPOF_SIGNAL_LOG_MAX_LINES
    near_miss_log_minutes: int = 10          # CLIO_DPOF_NEAR_MISS_LOG_MINUTES -- INFO re-log cadence per (underlying, outcome)
    eod_digest_enabled: bool = True          # CLIO_DPOF_EOD_DIGEST -- scheduled_tasks.py job gate

    # ── Learning loop F1: post-trade lessons (reflection), Task ORCH-7Q4K-DPOF
    # builder B2, 2026-09-26 -- see agent/dpof/reflection.py, docs/design/
    # clio_dpof_learning_loop.md §4/§7. Context only: lessons/tallies are never
    # a rule, a gate, or a vote -- they only enter the decider's own prompt. ──
    reflection_enabled: bool = True           # CLIO_DPOF_REFLECTION -- master gate for the whole post-trade reflection loop
    lessons_in_prompt: bool = True            # CLIO_DPOF_LESSONS_IN_PROMPT -- inject stored lessons into entry/review prompts

    # ── Advisory size anchor (operator 2026-09-29; agent/dpof/sizing.py anchor_*). A REFERENCE scale shown
    # beside each entry candidate, never a cap or a rule: the no-hard-caps directive stands. Premium as a share
    # of dpof_equity for a full entry at conviction 0.50 / 0.80 (linear between, clamped 2%..30%). Off -> the
    # "size_anchor" ctx key is absent and the decider message is byte-identical to before.
    size_anchor_enabled: bool = True          # CLIO_DPOF_SIZE_ANCHOR
    size_anchor_share_c50: float = 0.10       # CLIO_DPOF_ANCHOR_SHARE_C50, normalised into (0, 0.5]
    size_anchor_share_c80: float = 0.20       # CLIO_DPOF_ANCHOR_SHARE_C80, normalised into (0, 0.5], >= the c50 share
    reflection_model: str = ""                # CLIO_DPOF_REFLECTION_MODEL -- "" -> falls back to decision_model at call time
    reflection_fallback_model: str = ""       # CLIO_DPOF_REFLECTION_FALLBACK_MODEL -- "" -> falls back to fallback_model
    reflection_timeout_seconds: float = 60.0  # CLIO_DPOF_REFLECTION_TIMEOUT
    reflection_lookback_days: float = 7.0     # CLIO_DPOF_REFLECTION_LOOKBACK_DAYS -- a close older than this is never reflected on
    reflection_max_attempts: int = 3          # CLIO_DPOF_REFLECTION_MAX_ATTEMPTS -- per position; then permanently skipped (no lesson, ever)
    reflection_max_per_run: int = 2           # CLIO_DPOF_REFLECTION_MAX_PER_RUN -- eligible positions reflected on per cycle
    lessons_same_underlying: int = 5          # CLIO_DPOF_LESSONS_SAME_UNDERLYING -- max recent lessons shown for the nominated/position underlying
    lessons_cross: int = 3                    # CLIO_DPOF_LESSONS_CROSS -- max recent lessons shown for OTHER underlyings
    lessons_prompt_max_chars: int = 2500       # CLIO_DPOF_LESSONS_PROMPT_MAX_CHARS -- cap on the rendered lessons section
    lesson_alerts: bool = True                 # CLIO_DPOF_LESSON_ALERTS -- one plain-text Telegram line per new lesson

    # ── Learning loop F2: borderline debate context (bull/bear from a faster
    # model), same task/design-doc reference as F1 above. Advisory only --
    # never a vote, never blocks/accelerates an exit on its own. ────────────
    debate_enabled: bool = True                       # CLIO_DPOF_DEBATE -- master gate (entries)
    debate_on_reviews: bool = True                     # CLIO_DPOF_DEBATE_REVIEWS -- gate for the review-side debate (alignment_against only)
    debate_model: str = "claude-sonnet-5"              # CLIO_DPOF_DEBATE_MODEL
    debate_fallback_model: str = "gpt-6-sol"           # CLIO_DPOF_DEBATE_FALLBACK_MODEL
    debate_margin: float = 0.05                        # CLIO_DPOF_DEBATE_MARGIN -- entries: |composite| - threshold_normal <= this triggers a debate
    debate_timeout_seconds: float = 20.0               # CLIO_DPOF_DEBATE_TIMEOUT -- entry-side total deadline (bull+bear in parallel)
    debate_review_timeout_seconds: float = 12.0        # CLIO_DPOF_DEBATE_REVIEW_TIMEOUT -- review-side total deadline
    debate_review_min_minutes_to_flatten: float = 30.0 # CLIO_DPOF_DEBATE_REVIEW_MIN_MINUTES -- never debate a review this close to the flatten

    # ── Derived helpers ─────────────────────────────────────────────────
    @property
    def primary(self) -> str:
        return self.underlyings[0]

    @property
    def places_orders(self) -> bool:
        """True only for modes that are allowed to reach the broker at all."""
        return self.mode in (DpofMode.PAPER, DpofMode.LIVE)

    def live_ack_ok(self) -> bool:
        return self.live_ack == LIVE_ACK_PHRASE

    @property
    def jev_effective_mode(self) -> JevMode:
        """`jev_mode` degrades to OFF when Jev is completely UNCONFIGURED (no
        TYPESAFE_API_KEY, key file, or fallback URL) -- inert by construction
        rather than blocking every entry the moment VETO becomes the default.
        A CONFIGURED key/key_file that fails to resolve at RUNTIME (e.g. an
        unreadable key file) is a real failure, not an unconfigured state --
        this stays whatever `jev_mode` is (fail-closed in VETO, per
        `jev_fail_closed`; see jev.py's key-file resolution)."""
        if self.jev_mode != JevMode.OFF and not (
            self.jev_api_key or self.jev_key_file or self.jev_fallback_base_url
        ):
            return JevMode.OFF
        return self.jev_mode

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty = OK). Never raises."""
        problems: list[str] = []
        if not self.underlyings:
            problems.append("no underlyings configured")
        wsum = self.w_nofa + self.w_premium + self.w_darkpool + self.w_gex
        if abs(wsum - 1.0) > 1e-6:
            problems.append(f"pillar weights sum to {wsum:.3f}, expected 1.0")
        if not (0 < self.threshold_normal <= self.threshold_strong):
            problems.append("need 0 < threshold_normal <= threshold_strong")
        if not (1 <= self.min_pillars <= 4):
            problems.append("min_pillars must be 1..4")
        if not (0 < self.min_delta <= self.target_delta <= self.max_delta < 1):
            problems.append("need 0 < min_delta <= target_delta <= max_delta < 1")
        if self.mode == DpofMode.LIVE and self.alpaca_paper:
            problems.append("mode=live but CLIO_DPOF_ALPACA_PAPER is true")
        if self.mode == DpofMode.PAPER and not self.alpaca_paper:
            problems.append("mode=paper but CLIO_DPOF_ALPACA_PAPER is false")
        if self.places_orders and not (self.alpaca_key_id and self.alpaca_secret_key):
            problems.append("orders enabled but CLIO_DPOF_ALPACA_KEY_ID / _SECRET_KEY not set")
        # No "allow_verticals requires min_options_level >= 3" rule any more:
        # verticals are gated at RUNTIME against the broker-reported account
        # options level (engine.py), never a config-time hard error -- an
        # account below level 3 simply gets singles only.
        # Jev is never a problem: an unconfigured key just makes jev_effective_mode
        # degrade to OFF (see the property) -- it must not block the whole engine.
        if not math.isfinite(self.capital_base) or self.capital_base < 0:
            problems.append("capital_base must be >= 0 and finite")
        elif 0 < self.capital_base < self.min_account_equity:
            problems.append(
                f"capital_base ({self.capital_base}) is below min_account_equity ({self.min_account_equity})"
            )
        if self.decision_effort not in ("low", "medium", "high", "xhigh", "max"):
            problems.append("decision_effort must be low|medium|high|xhigh|max")
        if not (1 <= self.review_min_minutes <= self.review_interval_minutes <= self.review_max_minutes):
            problems.append("need 1 <= review_min_minutes <= review_interval_minutes <= review_max_minutes")
        if not (1 <= self.expiry_sessions_ahead <= self.max_expiry_sessions_ahead):
            problems.append("need 1 <= expiry_sessions_ahead <= max_expiry_sessions_ahead")
        if not (1 <= self.entry_candidates <= 6):
            problems.append("entry_candidates must be 1..6")
        if self.decision_timeout_seconds <= 0 or self.fallback_timeout_seconds <= 0:
            problems.append("decision/fallback timeouts must be positive")
        if not (0 < self.jev_escalate_threshold < 1):
            problems.append("jev_escalate_threshold must be in (0,1)")
        if not (0 < self.jev_size_veto_threshold < 1):
            problems.append("jev_size_veto_threshold must be in (0,1)")
        if self.reconcile_confirm_seconds < 0:
            problems.append("reconcile_confirm_seconds must be >= 0")
        if not (120 <= self.kronos_max_age_seconds <= 1800):
            problems.append("kronos_max_age_seconds must be in [120, 1800]")
        if self.kronos_pillar not in ("live", "off"):
            problems.append("kronos_pillar must be 'live' or 'off'")
        if not (0.0 <= self.w_kronos <= 1.0):
            problems.append("w_kronos must be in [0, 1]")
        if self.kronos_horizon not in (15, 30, 60):
            problems.append("kronos_horizon must be 15, 30 or 60")

        # Learning loop (F1 reflection / F2 debate) -- see docs/design/
        # clio_dpof_learning_loop.md §7. Never touches a threshold/gate/size
        # knob above; these only bound the two advisory LLM legs.
        if self.reflection_timeout_seconds <= 0:
            problems.append("reflection_timeout_seconds must be positive")
        if self.reflection_lookback_days <= 0:
            problems.append("reflection_lookback_days must be positive")
        if self.reflection_max_attempts < 1:
            problems.append("reflection_max_attempts must be >= 1")
        if self.reflection_max_per_run < 1:
            problems.append("reflection_max_per_run must be >= 1")
        if self.lessons_same_underlying < 0:
            problems.append("lessons_same_underlying must be >= 0")
        if self.lessons_cross < 0:
            problems.append("lessons_cross must be >= 0")
        if self.lessons_prompt_max_chars < 1:
            problems.append("lessons_prompt_max_chars must be >= 1")
        if not (0 <= self.debate_margin < 1):
            problems.append("debate_margin must be in [0,1)")
        if self.debate_timeout_seconds <= 0:
            problems.append("debate_timeout_seconds must be positive")
        if self.debate_review_timeout_seconds <= 0:
            problems.append("debate_review_timeout_seconds must be positive")
        if self.debate_review_min_minutes_to_flatten < 0:
            problems.append("debate_review_min_minutes_to_flatten must be >= 0")

        # Session time ordering (ET). Parsed the same way engine.py/risk.py's
        # parse_hhmm() do (split(":") + int()) -- an unparseable string is its
        # own problem, reported once, and skips the ordering checks that
        # would otherwise need it (never a crash from comparing None).
        t_start = _parse_hhmm(self.entry_start_et)
        t_end = _parse_hhmm(self.entry_end_et)
        t_flatten = _parse_hhmm(self.flatten_et)
        if t_start is None:
            problems.append(f"entry_start_et is not a valid HH:MM string: {self.entry_start_et!r}")
        if t_end is None:
            problems.append(f"entry_end_et is not a valid HH:MM string: {self.entry_end_et!r}")
        if t_flatten is None:
            problems.append(f"flatten_et is not a valid HH:MM string: {self.flatten_et!r}")
        if t_start is not None and t_end is not None and t_start >= t_end:
            problems.append(
                f"need entry_start_et < entry_end_et (got entry_start_et={self.entry_start_et!r} >= "
                f"entry_end_et={self.entry_end_et!r})"
            )
        if t_end is not None and t_flatten is not None and t_end >= t_flatten:
            problems.append(
                f"need entry_end_et < flatten_et (got entry_end_et={self.entry_end_et!r} >= "
                f"flatten_et={self.flatten_et!r})"
            )
        return problems


def _resolve_alpaca_keys(alpaca_paper: bool, d: DPOFConfig) -> tuple[str, str, str]:
    """Returns (key_id, secret_key, source); source is "dedicated" | "clio_paper" | "none".

    Dedicated DPOF keys (CLIO_DPOF_ALPACA_KEY_ID/_SECRET_KEY) win whenever BOTH
    halves are set. Otherwise, for a PAPER account only, falls back to CLIO's
    existing shared paper keys (ALPACA_PAPER_KEY_ID/_SECRET_KEY -- the same ones
    agent/config.py uses) -- but only as a COMPLETE pair from that one source,
    never mixed with a partial dedicated id/secret (a lone dedicated id paired
    with the shared secret, or vice versa, is worse than either alone). LIVE
    (alpaca_paper False) never falls back -- it always requires its own
    dedicated keys; validate()'s "orders enabled but keys not set" still
    catches a live config missing them.
    """
    dedicated_id = _env_str("CLIO_DPOF_ALPACA_KEY_ID", d.alpaca_key_id)
    dedicated_secret = _env_str("CLIO_DPOF_ALPACA_SECRET_KEY", d.alpaca_secret_key)
    if dedicated_id and dedicated_secret:
        return dedicated_id, dedicated_secret, "dedicated"
    if not alpaca_paper:
        return dedicated_id, dedicated_secret, "none"
    clio_id = _env_str("ALPACA_PAPER_KEY_ID", "")
    clio_secret = _env_str("ALPACA_PAPER_SECRET_KEY", "")
    if clio_id and clio_secret:
        return clio_id, clio_secret, "clio_paper"
    return dedicated_id, dedicated_secret, "none"


def load_dpof_config() -> DPOFConfig:
    """Build config from the current environment. Re-reads env every call."""
    d = DPOFConfig()  # defaults — single source for fallbacks
    _anchor_c50, _anchor_c80 = _env_anchor_shares(d)
    _resolved_paper = _env_bool("CLIO_DPOF_ALPACA_PAPER", d.alpaca_paper)
    _key_id, _secret_key, _key_source = _resolve_alpaca_keys(_resolved_paper, d)
    return DPOFConfig(
        mode=_env_enum("CLIO_DPOF_MODE", DpofMode, d.mode),
        underlyings=_env_list("CLIO_DPOF_UNDERLYINGS", d.underlyings),
        data_dir=_env_str("CLIO_DPOF_DATA_DIR", d.data_dir),
        client_order_prefix=_env_str("CLIO_DPOF_ORDER_PREFIX", d.client_order_prefix),
        alpaca_key_id=_key_id,
        alpaca_secret_key=_secret_key,
        alpaca_paper=_resolved_paper,
        alpaca_key_source=_key_source,
        options_feed=_env_str("CLIO_DPOF_OPTIONS_FEED", _env_str("ALPACA_OPTIONS_FEED", d.options_feed)).lower(),
        stock_feed=_env_str("CLIO_DPOF_STOCK_FEED", _env_str("ALPACA_STOCK_FEED", d.stock_feed)).lower(),
        capital_base=_env_float("CLIO_DPOF_CAPITAL", d.capital_base),
        live_ack=_env_str("CLIO_DPOF_LIVE_ACK", d.live_ack),
        live_account_number=_env_str("CLIO_DPOF_LIVE_ACCOUNT", d.live_account_number),
        min_options_level=_env_int("CLIO_DPOF_MIN_OPTIONS_LEVEL", d.min_options_level),
        min_account_equity=_env_float("CLIO_DPOF_MIN_EQUITY", d.min_account_equity),
        cycle_seconds=_env_int("CLIO_DPOF_CYCLE_SECONDS", d.cycle_seconds),
        uw_refresh_seconds=_env_int("CLIO_DPOF_UW_REFRESH_SECONDS", d.uw_refresh_seconds),
        darkpool_nbbo=_env_bool("CLIO_DPOF_DARKPOOL_NBBO", d.darkpool_nbbo),
        darkpool_flat_unavailable=_env_bool("CLIO_DPOF_DARKPOOL_FLAT_UNAVAILABLE", d.darkpool_flat_unavailable),
        empty_pillar_unavailable=_env_bool("CLIO_DPOF_EMPTY_PILLAR_UNAVAILABLE", d.empty_pillar_unavailable),
        dynamic_weights=_env_dw_mode("CLIO_DPOF_DYNAMIC_WEIGHTS", d.dynamic_weights),
        dw_ring_size=_env_int_clamped("CLIO_DPOF_DW_RING", d.dw_ring_size, 10, 200),
        dw_dead_eps=_env_dw_eps("CLIO_DPOF_DW_DEAD_EPS", d.dw_dead_eps),
        dw_quality_apply=_env_bool("CLIO_DPOF_DW_QUALITY_APPLY", d.dw_quality_apply),
        bars_refresh_seconds=_env_int("CLIO_DPOF_BARS_REFRESH_SECONDS", d.bars_refresh_seconds),
        entry_start_et=_env_str("CLIO_DPOF_ENTRY_START_ET", d.entry_start_et),
        entry_end_et=_env_str("CLIO_DPOF_ENTRY_END_ET", d.entry_end_et),
        flatten_et=_env_str("CLIO_DPOF_FLATTEN_ET", d.flatten_et),
        flatten_eod=_env_bool("CLIO_DPOF_FLATTEN_EOD", d.flatten_eod),
        w_nofa=_env_float("CLIO_DPOF_W_NOFA", d.w_nofa),
        w_premium=_env_float("CLIO_DPOF_W_PREMIUM", d.w_premium),
        w_darkpool=_env_float("CLIO_DPOF_W_DARKPOOL", d.w_darkpool),
        w_gex=_env_float("CLIO_DPOF_W_GEX", d.w_gex),
        min_pillars=_env_int("CLIO_DPOF_MIN_PILLARS", d.min_pillars),
        threshold_normal=_env_float("CLIO_DPOF_THRESHOLD_NORMAL", d.threshold_normal),
        threshold_strong=_env_float("CLIO_DPOF_THRESHOLD_STRONG", d.threshold_strong),
        dp_veto_threshold=_env_float("CLIO_DPOF_DP_VETO", d.dp_veto_threshold),
        require_price_confirmation=_env_bool("CLIO_DPOF_PRICE_CONFIRM", d.require_price_confirmation),
        qqq_divergence_ratio=_env_float("CLIO_DPOF_QQQ_DIVERGENCE", d.qqq_divergence_ratio),
        exit_threshold=_env_float("CLIO_DPOF_EXIT_THRESHOLD", d.exit_threshold),
        vix_trending_below=_env_float("CLIO_DPOF_VIX_TRENDING", d.vix_trending_below),
        vix_choppy_at=_env_float("CLIO_DPOF_VIX_CHOPPY", d.vix_choppy_at),
        vix_extreme_label_at=_env_float("CLIO_DPOF_VIX_EXTREME_LABEL", d.vix_extreme_label_at),
        vix_max_age_seconds=_env_int("CLIO_DPOF_VIX_MAX_AGE", d.vix_max_age_seconds),
        event_context_window_minutes=_env_int("CLIO_DPOF_EVENT_CONTEXT_WINDOW_MIN", d.event_context_window_minutes),
        data_stale_alarm_minutes=_env_int("CLIO_DPOF_STALE_ALARM_MIN", d.data_stale_alarm_minutes),
        target_delta=_env_float("CLIO_DPOF_TARGET_DELTA", d.target_delta),
        min_delta=_env_float("CLIO_DPOF_MIN_DELTA", d.min_delta),
        max_delta=_env_float("CLIO_DPOF_MAX_DELTA", d.max_delta),
        max_spread_pct=_env_float("CLIO_DPOF_MAX_SPREAD_PCT", d.max_spread_pct),
        min_open_interest=_env_int("CLIO_DPOF_MIN_OI", d.min_open_interest),
        allow_verticals=_env_bool("CLIO_DPOF_ALLOW_VERTICALS", d.allow_verticals),
        vertical_when_vix_at=_env_float("CLIO_DPOF_VERTICAL_VIX", d.vertical_when_vix_at),
        vertical_width_pct=_env_float("CLIO_DPOF_VERTICAL_WIDTH_PCT", d.vertical_width_pct),
        entry_lean=_env_float("CLIO_DPOF_ENTRY_LEAN", d.entry_lean),
        entry_reprice_seconds=_env_int("CLIO_DPOF_ENTRY_REPRICE_S", d.entry_reprice_seconds),
        entry_max_reprices=_env_int("CLIO_DPOF_ENTRY_MAX_REPRICES", d.entry_max_reprices),
        entry_max_slippage_pct=_env_float("CLIO_DPOF_ENTRY_MAX_SLIPPAGE", d.entry_max_slippage_pct),
        exit_lean=_env_float("CLIO_DPOF_EXIT_LEAN", d.exit_lean),
        exit_reprice_seconds=_env_int("CLIO_DPOF_EXIT_REPRICE_S", d.exit_reprice_seconds),
        exit_max_reprices=_env_int("CLIO_DPOF_EXIT_MAX_REPRICES", d.exit_max_reprices),
        reconcile_adopt_qty=_env_bool("CLIO_DPOF_RECONCILE_ADOPT", d.reconcile_adopt_qty),
        reconcile_confirm_seconds=_env_float("CLIO_DPOF_RECONCILE_CONFIRM_SECONDS", d.reconcile_confirm_seconds),
        uw_daily_call_cap=_env_int("CLIO_DPOF_UW_DAILY_CAP", d.uw_daily_call_cap),
        selfflow_fallback=_env_bool("CLIO_DPOF_SELFFLOW_FALLBACK", d.selfflow_fallback),
        selfflow_max_age_seconds=_env_int("CLIO_DPOF_SELFFLOW_MAX_AGE", d.selfflow_max_age_seconds),
        selfflow_min_trades=_env_int("CLIO_DPOF_SELFFLOW_MIN_TRADES", d.selfflow_min_trades),
        kronos_context=_env_bool("CLIO_DPOF_KRONOS_CONTEXT", d.kronos_context),
        kronos_max_age_seconds=_env_int_clamped("CLIO_DPOF_KRONOS_MAX_AGE", d.kronos_max_age_seconds, 120, 1800),
        kronos_pillar=_env_kronos_pillar(d.kronos_pillar),
        w_kronos=max(0.0, min(1.0, _env_float("CLIO_DPOF_W_KRONOS", d.w_kronos))),
        kronos_horizon=_env_kronos_horizon(d.kronos_horizon),
        jev_mode=_env_enum("CLIO_DPOF_JEV_MODE", JevMode, d.jev_mode),
        jev_api_key=_env_str("TYPESAFE_API_KEY", d.jev_api_key),
        jev_key_file=_env_str("CLIO_DPOF_JEV_KEY_FILE", d.jev_key_file),
        jev_base_url=_env_str("TYPESAFE_BASE_URL", d.jev_base_url).rstrip("/"),
        jev_model=_env_str("CLIO_DPOF_JEV_MODEL", d.jev_model),
        jev_timeout_seconds=_env_float("CLIO_DPOF_JEV_TIMEOUT", d.jev_timeout_seconds),
        jev_headline_conflict_threshold=_env_float("CLIO_DPOF_JEV_CONFLICT_T", d.jev_headline_conflict_threshold),
        jev_event_risk_threshold=_env_float("CLIO_DPOF_JEV_EVENT_T", d.jev_event_risk_threshold),
        jev_setup_veto_confidence=_env_float("CLIO_DPOF_JEV_SETUP_CONF", d.jev_setup_veto_confidence),
        jev_fail_closed=_env_bool("CLIO_DPOF_JEV_FAIL_CLOSED", d.jev_fail_closed),
        jev_circuit_failures=_env_int("CLIO_DPOF_JEV_CIRCUIT_FAILURES", d.jev_circuit_failures),
        jev_circuit_cooldown_seconds=_env_int("CLIO_DPOF_JEV_CIRCUIT_COOLDOWN", d.jev_circuit_cooldown_seconds),
        jev_max_headlines=_env_int("CLIO_DPOF_JEV_MAX_HEADLINES", d.jev_max_headlines),
        jev_fallback_base_url=_env_str("CLIO_DPOF_JEV_FALLBACK_URL", d.jev_fallback_base_url).rstrip("/"),
        jev_fallback_model=_env_str("CLIO_DPOF_JEV_FALLBACK_MODEL", d.jev_fallback_model),
        jev_fallback_api_key=_env_str("CLIO_DPOF_JEV_FALLBACK_KEY", d.jev_fallback_api_key),
        decision_model=_env_str("CLIO_DPOF_DECISION_MODEL", d.decision_model),
        decision_effort=_env_str("CLIO_DPOF_DECISION_EFFORT", d.decision_effort).lower(),
        decision_timeout_seconds=_env_float("CLIO_DPOF_DECISION_TIMEOUT", d.decision_timeout_seconds),
        decision_max_tokens=_env_int("CLIO_DPOF_DECISION_MAX_TOKENS", d.decision_max_tokens),
        fallback_model=_env_model_or_off("CLIO_DPOF_FALLBACK_MODEL", d.fallback_model),
        fallback_effort=_env_str("CLIO_DPOF_FALLBACK_EFFORT", d.fallback_effort).lower(),
        fallback_timeout_seconds=_env_float("CLIO_DPOF_FALLBACK_TIMEOUT", d.fallback_timeout_seconds),
        openai_api_key=_env_str("OPENAI_API_KEY", d.openai_api_key),
        openai_base_url=_env_str("CLIO_DPOF_OPENAI_BASE_URL", d.openai_base_url).rstrip("/"),
        decider_circuit_failures=_env_int("CLIO_DPOF_DECIDER_CIRCUIT_FAILURES", d.decider_circuit_failures),
        decider_sdk_max_retries=_env_int_clamped("CLIO_DPOF_DECISION_SDK_RETRIES", d.decider_sdk_max_retries, 0, 3),
        attempt_log_enabled=_env_bool("CLIO_DPOF_ATTEMPT_LOG", d.attempt_log_enabled),
        attempt_log_max_lines=_env_int_clamped("CLIO_DPOF_ATTEMPT_LOG_MAX_LINES", d.attempt_log_max_lines, 100, 200000),
        claim_check_mode=_claim_check_mode(d.claim_check_mode),
        decider_circuit_cooldown_seconds=_env_int("CLIO_DPOF_DECIDER_CIRCUIT_COOLDOWN", d.decider_circuit_cooldown_seconds),
        max_decisions_per_day=_env_int("CLIO_DPOF_MAX_DECISIONS_DAY", d.max_decisions_per_day),
        decision_max_age_seconds=_env_int("CLIO_DPOF_DECISION_MAX_AGE", d.decision_max_age_seconds),
        review_interval_minutes=_env_int("CLIO_DPOF_REVIEW_MINUTES", d.review_interval_minutes),
        review_min_minutes=_env_int("CLIO_DPOF_REVIEW_MIN_MINUTES", d.review_min_minutes),
        review_max_minutes=_env_int("CLIO_DPOF_REVIEW_MAX_MINUTES", d.review_max_minutes),
        review_trigger_move_pct=_env_float("CLIO_DPOF_REVIEW_TRIGGER_MOVE", d.review_trigger_move_pct),
        model_down_alert_minutes=_env_int("CLIO_DPOF_MODEL_DOWN_ALERT_MIN", d.model_down_alert_minutes),
        entry_candidates=_env_int("CLIO_DPOF_ENTRY_CANDIDATES", d.entry_candidates),
        entry_deliberation_cooldown_seconds=_env_int("CLIO_DPOF_ENTRY_DELIB_COOLDOWN", d.entry_deliberation_cooldown_seconds),
        entry_redeliberate_delta=_env_float("CLIO_DPOF_ENTRY_REDELIB_DELTA", d.entry_redeliberate_delta),
        expiry_sessions_ahead=_env_int("CLIO_DPOF_EXPIRY_SESSIONS", d.expiry_sessions_ahead),
        max_expiry_sessions_ahead=_env_int("CLIO_DPOF_MAX_EXPIRY_SESSIONS", d.max_expiry_sessions_ahead),
        jev_escalate_threshold=_env_float("CLIO_DPOF_JEV_ESCALATE_T", d.jev_escalate_threshold),
        jev_size_veto_threshold=_env_float("CLIO_DPOF_JEV_SIZE_T", d.jev_size_veto_threshold),
        signal_log_max_lines=_env_int("CLIO_DPOF_SIGNAL_LOG_MAX_LINES", d.signal_log_max_lines),
        near_miss_log_minutes=_env_int("CLIO_DPOF_NEAR_MISS_LOG_MINUTES", d.near_miss_log_minutes),
        eod_digest_enabled=_env_bool("CLIO_DPOF_EOD_DIGEST", d.eod_digest_enabled),
        reflection_enabled=_env_bool("CLIO_DPOF_REFLECTION", d.reflection_enabled),
        lessons_in_prompt=_env_bool("CLIO_DPOF_LESSONS_IN_PROMPT", d.lessons_in_prompt),
        size_anchor_enabled=_env_bool("CLIO_DPOF_SIZE_ANCHOR", d.size_anchor_enabled),
        size_anchor_share_c50=_anchor_c50,
        size_anchor_share_c80=_anchor_c80,
        reflection_model=_env_str("CLIO_DPOF_REFLECTION_MODEL", d.reflection_model),
        reflection_fallback_model=_env_str("CLIO_DPOF_REFLECTION_FALLBACK_MODEL", d.reflection_fallback_model),
        reflection_timeout_seconds=_env_float("CLIO_DPOF_REFLECTION_TIMEOUT", d.reflection_timeout_seconds),
        reflection_lookback_days=_env_float("CLIO_DPOF_REFLECTION_LOOKBACK_DAYS", d.reflection_lookback_days),
        reflection_max_attempts=_env_int("CLIO_DPOF_REFLECTION_MAX_ATTEMPTS", d.reflection_max_attempts),
        reflection_max_per_run=_env_int("CLIO_DPOF_REFLECTION_MAX_PER_RUN", d.reflection_max_per_run),
        lessons_same_underlying=_env_int("CLIO_DPOF_LESSONS_SAME_UNDERLYING", d.lessons_same_underlying),
        lessons_cross=_env_int("CLIO_DPOF_LESSONS_CROSS", d.lessons_cross),
        lessons_prompt_max_chars=_env_int("CLIO_DPOF_LESSONS_PROMPT_MAX_CHARS", d.lessons_prompt_max_chars),
        lesson_alerts=_env_bool("CLIO_DPOF_LESSON_ALERTS", d.lesson_alerts),
        debate_enabled=_env_bool("CLIO_DPOF_DEBATE", d.debate_enabled),
        debate_on_reviews=_env_bool("CLIO_DPOF_DEBATE_REVIEWS", d.debate_on_reviews),
        debate_model=_env_str("CLIO_DPOF_DEBATE_MODEL", d.debate_model),
        debate_fallback_model=_env_model_or_off("CLIO_DPOF_DEBATE_FALLBACK_MODEL", d.debate_fallback_model),
        debate_margin=_env_float("CLIO_DPOF_DEBATE_MARGIN", d.debate_margin),
        debate_timeout_seconds=_env_float("CLIO_DPOF_DEBATE_TIMEOUT", d.debate_timeout_seconds),
        debate_review_timeout_seconds=_env_float("CLIO_DPOF_DEBATE_REVIEW_TIMEOUT", d.debate_review_timeout_seconds),
        debate_review_min_minutes_to_flatten=_env_float(
            "CLIO_DPOF_DEBATE_REVIEW_MIN_MINUTES", d.debate_review_min_minutes_to_flatten
        ),
    )


def config_summary(cfg: DPOFConfig) -> dict:
    """Loggable view of the config with secrets redacted."""
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if f.name in ("alpaca_key_id", "alpaca_secret_key", "jev_api_key", "jev_fallback_api_key", "live_ack", "openai_api_key"):
            v = "set" if v else "unset"
        elif isinstance(v, Enum):
            v = v.value
        out[f.name] = v
    return out
