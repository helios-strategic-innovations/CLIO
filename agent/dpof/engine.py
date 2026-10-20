"""
CLIO DPOF — cycle orchestration (I/O).

Judgment layer (operator directive, 2026-09-24): NO mechanical trading.
Entries and exits are judged by CLIO's decision model
(agent.dpof.decider.DPOFDecider — Opus 5.5, fallback GPT-6 Sol); code only
enforces hard size caps and the ONE kept rail, the 15:50 ET forced flatten
(agent.dpof.exits.evaluate_exit). Jev (TypeSafe) is a veto-only entry gate
(unchanged) and, for open positions, a SENTINEL that may only pull a
position's next review forward — it never trades.

DPOFEngine.run_cycle() stays a ~30s cycle under `_cycle_lock` with a ~25s job
timeout, but decider calls (10-60s) never run inside it: decide_entry() /
review_position() are launched as background asyncio Tasks (at most one
in-flight per underlying for entries, one per position for reviews) and
harvested — applied only if still fresh (`decision_max_age_seconds`) and
still applicable — on a LATER cycle. See docs/design/clio_dpof.md and
DPOF_V2_SPEC.md for the full flow this follows.

CANCELLATION (CLAUDE.md #35): run_dpof_cycle runs under scheduled_tasks'
asyncio.wait_for timeout wrapper, so this coroutine can be cancelled at any
await. asyncio.CancelledError is a BaseException and is never caught by any
`except Exception` in this file. Every ledger/risk write that records a
state change (decision/jev log appends, position updates, the final risk-
state save) is wrapped in asyncio.shield(...) so a cancellation lands only
ever AFTER that specific write has completed, never mid-write.
open_position()/close_position()/work_orders() already shield their own
writes internally (agent/dpof/orders.py) — this file only shields the
writes it performs directly. Background decision Tasks are cancelled
explicitly on shutdown_dpof() and whenever mode turns OFF (see
_cancel_all_decision_tasks) — CancelledError raised into a task's own
wrapper (_run_decision_task) is re-raised, never swallowed into a result.

The exit-management path (position reviews, step 8c below) calls Jev only
as the SENTINEL (jev_position_check) — it can only pull a review forward,
never trade. Jev's VETO role (jev_entry_check) is consulted only for an
already-gated, decider-approved ENTRY, per docs/design/clio_dpof.md §8's
hard boundary.

ORDER-SAFETY RULE (in-flight entry vs EOD; trim during a working exit;
review during a working exit — see the final build report for the risk
analysis): a harvested review decision (trim/exit) is applied ONLY when the
position is still `status == "open"` AND its qty_open is unchanged since the
review was launched — a position that has moved to "closing" (another exit
batch already working, e.g. the EOD flatten fired first this cycle) discards
the harvested decision ("position_changed_or_not_open") rather than
submitting a second, conflicting close. `urgency == "urgent"` on an
applicable decision instead lets OrderManager escalate THAT close to
marketable/market sooner (see orders.py); it is not a bypass of the
"already-closing" rule for a batch that's already in flight.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from agent.dpof import risk as risk_module
from agent.dpof.alerts import (
    DPOFAlerts,
    fmt_entry,
    fmt_exit,
    fmt_gates,
    fmt_model_down,
    fmt_model_recovered,
    fmt_stale,
)
from agent.dpof.config import DPOFConfig, DpofMode, JevMode, config_summary, load_dpof_config
from agent.dpof.contracts import choose_candidates, expiry_fetch_window, option_right
from agent.dpof.debate import (
    build_debate_input,
    render_debate_section,
    run_debate,
    should_debate_entry,
    should_debate_review,
)
from agent.dpof.exits import evaluate_exit, position_pnl_pct
from agent.dpof.jev import (
    JevClient,
    _gamma_regime_words,
    _momentum_words,
    _price_vs_vwap_words,
    bucket_vix,
    build_position_request,
    jev_entry_check,
    jev_position_check,
    prepare_entry_request,
)
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import (
    ContractChoice,
    Direction,
    JevVerdict,
    MarketSnapshot,
    OrderLeg,
    PILLAR_NAMES,
    PositionRecord,
    Regime,
    RiskState,
    Signal,
)
from agent.dpof.prompts import _format_vol_surface_line
from agent.dpof.signal_log import SignalLog, flow_source_label, format_outcome_log_line
from agent.dpof.signals import KRONOS_PILLAR, classify_regime, compute_composite, evaluate_signal, position_alignment
from agent.dpof.sizing import clamp_to_solvency, max_affordable_qty, size_telemetry

logger = logging.getLogger("clio.dpof.engine")

ET = ZoneInfo("America/New_York")

# ── Scheduler cadence (read once at import — restart to change; BUILD_SPEC) ──
_startup_cfg = load_dpof_config()
DPOF_JOB_INTERVAL: int = max(10, min(300, _startup_cfg.cycle_seconds))
DPOF_JOB_TIMEOUT: int = max(8, DPOF_JOB_INTERVAL - 5)
del _startup_cfg

_RECONCILE_EVERY_N_CYCLES = 10

# A missing/unavailable decider retries this soon on a REVIEW (spec: "retry
# after ~60s (not the full interval)") rather than waiting a full
# review_interval_minutes.
_REVIEW_RETRY_AFTER_UNAVAILABLE_SECONDS = 60.0
# Kronos live pillar: alert once a session day after this long with no fresh forecast in the entry window.
_KRONOS_STALE_ALERT_SECONDS = 900.0

_OCC_STRIKE_RE = re.compile(r"^[A-Z.]{1,6}\d{6}([CP])(\d{8})$")

# Task 1 (2026-09-25): the label shown to the decider for signals.Regime.HALT
# is "EXTREME" -- the enum member itself keeps its original name/value
# ("halt", persisted/logged history) since it is not this file's/G's to
# rename (agent/dpof/models.py); only its PRESENTATION to the decider changes,
# from an automatic entry veto to an advisory regime label.
_REGIME_LABELS = {
    Regime.TRENDING: "TRENDING",
    Regime.NORMAL: "NORMAL",
    Regime.CHOPPY: "CHOPPY",
    Regime.HALT: "EXTREME",
    Regime.UNKNOWN: "UNKNOWN",
}


def _parse_iso_epoch(s: Optional[str]) -> Optional[float]:
    """Best-effort ISO-8601 -> epoch seconds for the timestamp-ish fields
    the E/V/N advisory intel contracts carry (release_time_et, ts, as_of).
    Naive datetimes are assumed ET (matches the rest of agent/dpof's own
    "no manual UTC-offset math" convention). Never raises -- None on any
    parse failure or falsy input."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return dt.timestamp()


def _clamp01(x: Optional[float]) -> float:
    if x is None or not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, x))


# ─────────────────────────────────────────────────────────────────────────
# Market-hours default (lazy, holiday-aware, with a logged-once fallback)
# ─────────────────────────────────────────────────────────────────────────

_market_fallback_warned = False


def _local_market_open_fallback() -> bool:
    """Weekday 09:30-16:00 ET, no holiday awareness. Last-resort only."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


def _default_market_open_fn() -> bool:
    global _market_fallback_warned
    try:
        from agent.integrations.market_events import _is_market_hours  # lazy

        return _is_market_hours()
    except Exception as e:
        if not _market_fallback_warned:
            logger.warning(
                "DPOF engine: agent.integrations.market_events unavailable (%s: %s); "
                "falling back to a local weekday 09:30-16:00 ET check (not holiday-aware)",
                type(e).__name__, e,
            )
            _market_fallback_warned = True
        return _local_market_open_fallback()


def _early_close_dates() -> frozenset:
    try:
        from agent.integrations.market_events import MARKET_EARLY_CLOSE  # lazy

        return frozenset(MARKET_EARLY_CLOSE)
    except Exception as e:
        logger.debug(f"DPOF engine: MARKET_EARLY_CLOSE unavailable: {type(e).__name__}: {e}")
        return frozenset()


def _market_holidays() -> frozenset:
    try:
        from agent.integrations.market_events import MARKET_HOLIDAYS  # lazy

        return frozenset(MARKET_HOLIDAYS)
    except Exception as e:
        logger.debug(f"DPOF engine: MARKET_HOLIDAYS unavailable: {type(e).__name__}: {e}")
        return frozenset()


def _session_cfg(cfg: DPOFConfig, now_et: datetime) -> DPOFConfig:
    """Clamp the day's flatten/entry-end times to the actual session close.

    Review finding (pass 2): on NYSE half-days (13:00 close) the market-hours gate
    stops the cycle at 13:00, so a 15:50 flatten never fires and positions carry
    over unmanaged (Dec 24 -> across Christmas). On early-close days flatten is
    moved to close-10min and the entry window ends at close-45min."""
    if now_et.date() not in _early_close_dates():
        return cfg
    close_min = 13 * 60
    def _hhmm(total: int) -> str:
        return f"{total // 60:02d}:{total % 60:02d}"
    def _mins(hhmm: str) -> int:
        t = risk_module.parse_hhmm(hhmm)
        return t.hour * 60 + t.minute
    flatten = min(_mins(cfg.flatten_et), close_min - 10)
    entry_end = min(_mins(cfg.entry_end_et), close_min - 45)
    return replace(cfg, flatten_et=_hhmm(flatten), entry_end_et=_hhmm(entry_end))


def _in_entry_window(now_et: datetime, cfg: DPOFConfig) -> bool:
    t = now_et.time()
    return risk_module.parse_hhmm(cfg.entry_start_et) <= t <= risk_module.parse_hhmm(cfg.entry_end_et)


def _sessions_to_expiry(expiration_iso: str, today: date) -> int:
    """Trading sessions from `today` (exclusive) to the option's expiration
    (inclusive), skipping weekends + agent.integrations.market_events.
    MARKET_HOLIDAYS. Small, deliberate duplication of contracts.py's
    target_expiries() session-skip logic (that function generates N future
    dates; this one counts sessions to a single KNOWN date) — used only to
    populate EntryCandidate.sessions_to_expiry for the decision model's
    context, never as a gate."""
    try:
        expiry = date.fromisoformat(expiration_iso)
    except (TypeError, ValueError):
        return 0
    holidays = _market_holidays()
    count = 0
    cursor = today
    while cursor < expiry:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5 and cursor not in holidays:
            count += 1
    return count


def _strike_desc(legs: list[OrderLeg]) -> str:
    """OCC legs -> a short human strike description, e.g. "580C" (single) or
    "580/585C" (vertical, same right)."""
    parts: list[str] = []
    rights: list[str] = []
    for leg in legs:
        m = _OCC_STRIKE_RE.match(leg.symbol or "")
        if not m:
            parts.append("?")
            rights.append("")
            continue
        right, strike_raw = m.groups()
        k = int(strike_raw) / 1000.0
        parts.append(f"{k:g}")
        rights.append(right)
    if len(parts) == 1:
        return f"{parts[0]}{rights[0]}"
    if len(parts) == 2 and rights[0] == rights[1] and rights[0]:
        return f"{parts[0]}/{parts[1]}{rights[1]}"
    return "/".join(f"{p}{r}" for p, r in zip(parts, rights))


def _net_bid_ask_from_legs(legs: list[dict], quotes: dict) -> tuple[Optional[float], Optional[float]]:
    """Net bid/ask for a position's legs (dict form: {"symbol", "side",
    "ratio_qty", "position_intent"}) from a {symbol: OptionQuote} map.
    Single leg -> its own bid/ask. Vertical -> long-minus-short. Small,
    deliberate duplication of orders.py's OrderManager._net_bid_ask (a
    private staticmethod on a class tests fake out entirely) so engine.py's
    entry re-quote never depends on the real/fake OrderManager's internals.
    """
    if not legs:
        return None, None
    if len(legs) == 1:
        q = quotes.get(legs[0].get("symbol"))
        if q is None:
            return None, None
        return q.bid, q.ask

    def _is_long(l: dict) -> bool:
        return l.get("position_intent") == "buy_to_open"

    def _is_short(l: dict) -> bool:
        return l.get("position_intent") == "sell_to_open"

    long_leg = next((l for l in legs if _is_long(l)), None)
    short_leg = next((l for l in legs if _is_short(l)), None)
    if long_leg is None or short_leg is None:
        return None, None
    ql = quotes.get(long_leg.get("symbol"))
    qs = quotes.get(short_leg.get("symbol"))
    if ql is None or qs is None or ql.bid is None or ql.ask is None or qs.bid is None or qs.ask is None:
        return None, None
    return (ql.bid - qs.ask), (ql.ask - qs.bid)


# ─────────────────────────────────────────────────────────────────────────
# Virtual capital for a SHARED Alpaca account (cfg.capital_base)
# ─────────────────────────────────────────────────────────────────────────

def _dpof_equity(
    account_equity: Optional[float], risk: RiskState, unrealized: float, cfg: DPOFConfig,
) -> Optional[float]:
    """The equity DPOF actually uses for sizing and status.

    CLIO's shared paper account may hold other, non-DPOF positions and a much
    larger balance than the ~$20K this engine is sized for. Sizing by the
    full account equity oversizes. This scopes DPOF to a virtual slice of the
    account instead.

    cfg.capital_base <= 0 (or non-finite) disables virtual capital entirely --
    returns `account_equity` unchanged, which is correct for a dedicated
    account (capital_base=0 means "trust the real account equity").

    cfg.capital_base > 0: DPOF's own equity is `capital_base +
    risk.realized_pnl_total + unrealized` (a virtual ledger seeded at
    capital_base and tracked forward purely by DPOF's own trades), capped at
    the REAL account equity so DPOF can never claim more capital exists than
    the account actually has. Unknown real equity must stay unknown -- this
    NEVER invents a number when `account_equity` is None/non-finite.
    """
    if cfg.capital_base is None or not math.isfinite(cfg.capital_base) or cfg.capital_base <= 0:
        return account_equity
    if account_equity is None or not math.isfinite(account_equity):
        return None
    unrealized = unrealized if (unrealized is not None and math.isfinite(unrealized)) else 0.0
    virtual = cfg.capital_base + risk.realized_pnl_total + unrealized
    if not math.isfinite(virtual):
        return account_equity
    return min(account_equity, virtual)


def _jev_inactive(cfg: DPOFConfig) -> bool:
    """True when jev_mode is configured (not OFF) but jev_effective_mode
    degraded it to OFF because no TypeSafe key/key_file/fallback URL is set."""
    return cfg.jev_mode != JevMode.OFF and cfg.jev_effective_mode == JevMode.OFF


def _jev_status_text(cfg: DPOFConfig) -> str:
    """'<mode> (inactive — no TYPESAFE_API_KEY)' when unconfigured; else '<mode>'."""
    if _jev_inactive(cfg):
        return f"{cfg.jev_mode.value} (inactive — no TYPESAFE_API_KEY)"
    return cfg.jev_mode.value


# ─────────────────────────────────────────────────────────────────────────
# Memory hooks (fire-and-forget; blocking calls run off the event loop)
# ─────────────────────────────────────────────────────────────────────────

class _RealMemoryHooks:
    """Default memory collaborator: the actual CLIO memory engine."""

    def remember(self, **kwargs: Any) -> None:
        from agent.memory.memory_engine import get_memory_engine

        get_memory_engine().remember(**kwargs)

    def learn_from_outcome(self, **kwargs: Any) -> None:
        from agent.memory.memory_engine import get_memory_engine

        get_memory_engine().learn_from_outcome(**kwargs)


# ─────────────────────────────────────────────────────────────────────────
# Decider unavailability — engine's own stand-in until agent.dpof.decider
# (D1) lands. Once `from agent.dpof.decider import DPOFDecider,
# DecisionUnavailable` succeeds, `self._decision_unavailable_exc` is
# repointed at the REAL DecisionUnavailable and this class is never raised.
# ─────────────────────────────────────────────────────────────────────────

class _NoDeciderAvailable(Exception):
    """Engine-local sentinel: agent.dpof.decider has not landed / failed to
    import. Never raised outside this module; used only so `self.
    _decision_unavailable_exc` always names SOME exception type, letting
    `_run_decision_task` classify a failure as "unavailable" uniformly."""


async def _default_jev_sentinel(cfg: DPOFConfig, client, position_ctx: dict):
    """Default `jev_sentinel=` collaborator: agent.dpof.jev.jev_position_check
    (D2). Injectable so engine tests fake it out without touching the real
    Jev transport."""
    return await jev_position_check(cfg, client, position_ctx)


def _default_candidate_fn(chain, *, underlying, direction, spot, today, cfg, now_et=None):
    """Default `candidate_fn=` collaborator: agent.dpof.contracts.
    choose_candidates (D4)."""
    return choose_candidates(chain, underlying=underlying, direction=direction, spot=spot, today=today, cfg=cfg, now_et=now_et)


def _default_expiry_fetch_window_fn(today, cfg):
    """Default `expiry_fetch_window_fn=` collaborator: agent.dpof.contracts.
    expiry_fetch_window (D4) — the 1DTE-aware chain-fetch window spanning
    every session choose_candidates might fall back to."""
    return expiry_fetch_window(today, cfg)


def _default_position_alignment_fn(snapshot_for_underlying, direction):
    """Default `position_alignment_fn=` collaborator: agent.dpof.signals.
    position_alignment (D4)."""
    return position_alignment(snapshot_for_underlying, direction)


# ─────────────────────────────────────────────────────────────────────────
# DPOFEngine
# ─────────────────────────────────────────────────────────────────────────

class DPOFEngine:
    def __init__(
        self,
        *,
        broker: Any = None,
        feed: Any = None,
        ledger: Optional[DPOFLedger] = None,
        orders: Any = None,
        jev: Optional[JevClient] = None,
        alerts: Optional[DPOFAlerts] = None,
        memory: Any = None,
        decider: Any = None,
        signal_log: Optional[SignalLog] = None,
        # Learning loop (Task ORCH-7Q4K-DPOF builder B3, 2026-09-26): F1 post-
        # trade lessons (reflection.py) + F2 borderline debate (debate.py) --
        # see docs/design/clio_dpof_learning_loop.md. Injectable so tests fake
        # them out; mirrors how decider/signal_log are injected above.
        advisory_reflection: Any = None,
        advisory_debate: Any = None,
        reflection_service: Any = None,
        lesson_store: Any = None,
        # decision-record producer: records every entry-veto/exit-sentinel
        # question + Jev's answer (or lack of one) into the shared `decisions`
        # schema, fans the same question out to configured local-model shadow
        # endpoints, and links a closed position back to its entry decision.
        # Injectable so tests fake it out; built lazily next to JevClient
        # otherwise (see _ensure_collaborators).
        decisions: Any = None,
        candidate_fn: Callable = _default_candidate_fn,
        expiry_fetch_window_fn: Callable = _default_expiry_fetch_window_fn,
        jev_sentinel: Callable = _default_jev_sentinel,
        position_alignment_fn: Callable = _default_position_alignment_fn,
        clock: Callable[[], float] = time.time,
        market_open_fn: Optional[Callable[[], bool]] = None,
        config_loader: Callable[[], DPOFConfig] = load_dpof_config,
    ) -> None:
        self._broker = broker
        self._feed = feed
        self._ledger = ledger
        self._orders = orders
        self._jev = jev
        self._alerts = alerts
        self._memory = memory or _RealMemoryHooks()
        self._decider = decider
        self._signal_log = signal_log
        self._advisory_reflection = advisory_reflection
        self._advisory_debate = advisory_debate
        self._reflection = reflection_service
        self._lesson_store = lesson_store
        self._decisions = decisions
        self._decision_unavailable_exc: type = _NoDeciderAvailable
        self._decider_import_warned = decider is not None
        self._candidate_fn = candidate_fn
        self._expiry_fetch_window_fn = expiry_fetch_window_fn
        self._jev_sentinel = jev_sentinel
        self._position_alignment_fn = position_alignment_fn
        self._clock = clock
        self._market_open_fn = market_open_fn or _default_market_open_fn
        self._config_loader = config_loader

        self._cfg: Optional[DPOFConfig] = None
        self._ledger_loaded = False
        self._build_lock = asyncio.Lock()

        # One-shot, process-lifetime flags (the engine singleton lives for the
        # whole process, so a plain bool is "once per process" by construction;
        # _safe_alert's own dedupe_key is defense-in-depth on top of it).
        self._activation_alert_sent = False
        self._jev_inactive_logged = False

        self._cycle_count = 0
        self._last_snapshot: Optional[MarketSnapshot] = None
        self._last_signal: Optional[Signal] = None
        self._last_jev_verdict: Optional[JevVerdict] = None
        self._last_marks: dict[str, Optional[float]] = {}
        self._last_unrealized: Optional[float] = None
        self._model_down = False
        # Advisory intel (Task 2, INTEL_V3_SPEC.md): events/vol_surface/
        # breaking/regime/calendar_status, computed ONCE per cycle
        # (_build_intel_ctx) so _build_ctx/_review_triggers/_cooldown_allows
        # all see the SAME snapshot of intel within one cycle. Empty by
        # default (before any cycle has run, or on a build failure) -- every
        # reader below treats that as "no intel available", never a crash.
        self._last_intel_ctx: dict = {}

        # In-flight decision tasks (spec F: >=1 per key; cancelled on
        # shutdown_dpof()/mode OFF; never awaited inside a cycle).
        self._entry_tasks: dict[str, asyncio.Task] = {}       # key: underlying
        self._entry_task_meta: dict[str, dict] = {}
        self._review_tasks: dict[str, asyncio.Task] = {}      # key: position id
        self._review_task_meta: dict[str, dict] = {}
        # Per underlying+direction re-deliberation cooldown (spec: bypassed by
        # composite strengthening or the decider's own recheck_minutes).
        self._entry_cooldowns: dict[str, dict] = {}
        # Task 3 review trigger (b): breaking-news event ids already seen FOR
        # THIS POSITION (per-position set; never persisted -- a process
        # restart just means the next review relies on the regular "due"/
        # other triggers instead, which is acceptable for an advisory
        # trigger). Popped on exit_filled closes and swept every review pass
        # for any position no longer open (all close paths), so this never
        # grows unbounded.
        self._seen_breaking_event_ids: dict[str, set] = {}

        self._background_tasks: set[asyncio.Task] = set()   # memory hooks only
        # Serializes a cycle with operator /dpof halt|resume. Without it a halt
        # landing mid-cycle is overwritten by the cycle's final save_risk() (the
        # cycle read risk at step 7). A halt waits at most one cycle (<= job
        # timeout); the data/dpof/HALT file remains the instant entry stop.
        self._cycle_lock = asyncio.Lock()

        # Learning loop (Task ORCH-7Q4K-DPOF builder B3): F1 reflection
        # scheduling state + F2's data-capture inputs. `_reflection_task` is
        # the ENGINE's own single-flight guard for the fire-and-forget
        # run_pending() Task -- separate from ReflectionService's own
        # internal `_running` flag, so a cycle never even spawns a redundant
        # Task while one is in flight. `_pnl_extremes[position_id] = [max,
        # min]` tracks observed mark-based P&L% extremes in memory each
        # cycle (step 8b) while a position is open; popped at close (never
        # persisted -- reset by a restart, which is acceptable per the design
        # frame's §4.1 "best-effort" note). `_last_bot` is the most recent
        # cycle's Telegram bot handle, so a background reflection alert (which
        # never runs inside the cycle itself) can still reach _safe_alert.
        self._reflection_task: Optional[asyncio.Task] = None
        self._pnl_extremes: dict[str, list[float]] = {}
        # Kronos live pillar: when the forecast was first seen missing inside the entry window.
        self._kronos_missing_since: Optional[float] = None
        self._last_bot: Any = None

        # decision-record producer -- outcome linkage (step 13c,
        # next to _maybe_schedule_reflection): a single-flight background
        # scan of the ledger's closed positions, joining each one that
        # carries a decision_call_id (judgment.entry_decision) to a
        # decisions.outcomes row. `_decision_outcome_ids` is lazy-loaded
        # (off-loop) from a small persisted JSON set living next to the
        # decisions outbox -- None means "not loaded yet", not "empty".
        self._decision_outcomes_task: Optional[asyncio.Task] = None
        self._decision_outcome_ids: Optional[set] = None

    # ------------------------------------------------------------------
    # Collaborator lifecycle
    # ------------------------------------------------------------------

    async def _ensure_collaborators(self, cfg: DPOFConfig) -> None:
        async with self._build_lock:
            if self._broker is None:
                from agent.dpof.broker import DPOFBroker

                self._broker = DPOFBroker(cfg)
            if self._feed is None:
                from agent.dpof.data import DPOFDataFeed

                self._feed = DPOFDataFeed(cfg, self._broker)
            if self._ledger is None:
                self._ledger = DPOFLedger(cfg.data_dir)
            if not self._ledger_loaded:
                await self._ledger.load()
                self._ledger_loaded = True
            if self._signal_log is None:
                self._signal_log = SignalLog(cfg.data_dir)
            if self._orders is None:
                from agent.dpof.orders import OrderManager

                self._orders = OrderManager(cfg, self._broker, self._ledger)
            if self._jev is None:
                self._jev = JevClient(cfg)
            if self._alerts is None:
                self._alerts = DPOFAlerts()
            if self._decisions is None:
                try:
                    from agent.decisions import DecisionRecorder

                    self._decisions = DecisionRecorder()
                except Exception as e:
                    logger.warning(
                        "DPOF engine: DecisionRecorder unavailable (%s: %s) -- decisions recording is OFF",
                        type(e).__name__, e,
                    )
            if self._decider is None:
                try:
                    from agent.dpof.decider import DecisionUnavailable, DPOFDecider

                    self._decider = DPOFDecider(cfg)
                    self._decision_unavailable_exc = DecisionUnavailable
                    self._decider_import_warned = False
                    logger.info("DPOF engine: decider ready (agent.dpof.decider landed)")
                except Exception as e:
                    if not self._decider_import_warned:
                        logger.warning(
                            "DPOF engine: decider unavailable (%s: %s) -- no entries will be "
                            "attempted; open positions are held with model-down alerts until "
                            "agent.dpof.decider is importable",
                            type(e).__name__, e,
                        )
                        self._decider_import_warned = True

            # Learning loop (Task ORCH-7Q4K-DPOF builder B3): each collaborator
            # is independent and failure-isolated -- an outage/import failure
            # in one never blocks the others or the rest of _ensure_collaborators.
            # TWO separate AdvisoryLLM instances (independent circuit breakers:
            # a reflection outage must not trip the debate breaker, and vice
            # versa).
            if self._advisory_reflection is None:
                try:
                    from agent.dpof.advisory_llm import AdvisoryLLM

                    self._advisory_reflection = AdvisoryLLM()
                except Exception as e:
                    logger.warning(
                        "DPOF engine: AdvisoryLLM (reflection) unavailable (%s: %s) -- "
                        "post-trade lessons (F1) are OFF",
                        type(e).__name__, e,
                    )
            if self._advisory_debate is None:
                try:
                    from agent.dpof.advisory_llm import AdvisoryLLM

                    self._advisory_debate = AdvisoryLLM()
                except Exception as e:
                    logger.warning(
                        "DPOF engine: AdvisoryLLM (debate) unavailable (%s: %s) -- "
                        "borderline debate context (F2) is OFF",
                        type(e).__name__, e,
                    )
            if self._lesson_store is None:
                try:
                    from agent.dpof.reflection import LessonStore

                    store = LessonStore(cfg.data_dir)
                    await store.load()
                    self._lesson_store = store
                except Exception as e:
                    logger.warning(
                        "DPOF engine: LessonStore unavailable (%s: %s) -- post-trade lessons "
                        "(F1) are OFF",
                        type(e).__name__, e,
                    )
            if self._reflection is None and self._lesson_store is not None and self._advisory_reflection is not None:
                try:
                    from agent.dpof.reflection import ReflectionService

                    self._reflection = ReflectionService(
                        self._lesson_store, self._advisory_reflection,
                        cfg_getter=lambda: self._cfg or cfg,
                        decisions_reader=self._read_position_reviews,
                        alert_fn=self._reflection_alert_fn,
                    )
                except Exception as e:
                    logger.warning(
                        "DPOF engine: ReflectionService unavailable (%s: %s) -- post-trade "
                        "lessons (F1) are OFF",
                        type(e).__name__, e,
                    )

    async def _ensure_ledger_only(self, cfg: DPOFConfig) -> None:
        """Constructs (and loads) ONLY the ledger + alerts -- never the broker,
        feed, orders, jev, or decider. Used by the OFF path so it can check for
        and alert on stranded open positions without ever touching the broker
        (mode=off must remain fully network-inert; see
        test_off_does_nothing_and_constructs_nothing)."""
        async with self._build_lock:
            if self._ledger is None:
                self._ledger = DPOFLedger(cfg.data_dir)
            if not self._ledger_loaded:
                await self._ledger.load()
                self._ledger_loaded = True
            if self._alerts is None:
                self._alerts = DPOFAlerts()

    def _push_config(self, cfg: DPOFConfig) -> None:
        for collaborator in (self._broker, self._feed, self._orders, self._jev, self._decider):
            if collaborator is None:
                continue
            try:
                collaborator.update_config(cfg)
            except Exception as e:
                logger.error(f"DPOF engine: update_config failed on {type(collaborator).__name__}: {e}")

    # ------------------------------------------------------------------
    # Fire-and-forget plumbing (memory hooks only — decision tasks use their
    # own tracked dicts, never this set)
    # ------------------------------------------------------------------

    def _fire_and_forget(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _memory_remember_safe(self, position_id: str, thesis: str) -> None:
        try:
            self._memory.remember(
                key=f"dpof_entry_{position_id}",
                content=thesis,
                category="trading",
                priority=0.6,
                domain="trading",
                source="dpof_engine",
            )
        except Exception as e:
            logger.debug(f"DPOF engine: memory.remember failed: {type(e).__name__}: {e}")

    def _memory_learn_safe(self, underlying: str, direction: str, outcome: str, pnl: float, details: str) -> None:
        try:
            self._memory.learn_from_outcome(
                action=f"dpof_{direction}_{underlying}",
                outcome=outcome,
                domain="trading",
                was_correct=(pnl > 0),
                details=details,
            )
        except Exception as e:
            logger.debug(f"DPOF engine: memory.learn_from_outcome failed: {type(e).__name__}: {e}")

    async def _remember_entry(self, pos: PositionRecord) -> None:
        try:
            loop = asyncio.get_running_loop()
            thesis = pos.thesis or (
                f"DPOF entry {pos.underlying} {pos.direction} {pos.structure} "
                f"qty={pos.qty_original} entry={pos.entry_price}"
            )
            await loop.run_in_executor(None, self._memory_remember_safe, pos.id, thesis)
        except Exception as e:
            logger.debug(f"DPOF engine: remember(entry) failed: {type(e).__name__}: {e}")

    async def _learn_from_exit(self, pos: PositionRecord) -> None:
        try:
            loop = asyncio.get_running_loop()
            pnl = pos.realized_pnl
            outcome = f"{pnl:+.2f} USD ({pos.exit_reason or 'unknown'})"
            details = f"{pos.underlying} {pos.direction} {pos.structure} qty={pos.qty_original}"
            await loop.run_in_executor(
                None, self._memory_learn_safe, pos.underlying, pos.direction, outcome, pnl, details,
            )
        except Exception as e:
            logger.debug(f"DPOF engine: learn_from_outcome failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Learning loop (Task ORCH-7Q4K-DPOF builder B3) -- collaborator
    # callbacks + shared helpers used by both F1 (reflection) and F2 (debate).
    # ------------------------------------------------------------------

    async def _reflection_alert_fn(self, text: str) -> None:
        """ReflectionService's `alert_fn` -- plain-text Telegram via the same
        `_safe_alert` every other DPOF alert uses. `cfg.lesson_alerts` is
        already checked by ReflectionService itself before calling this."""
        await self._safe_alert(self._last_bot, text)

    async def _read_position_reviews(self, position_id: str) -> list[dict]:
        """ReflectionService's `decisions_reader` -- every decisions.jsonl
        `kind="position_review"` row for `position_id`, oldest-first (the
        file's own append order; `readmodel.recent_decisions` never
        reorders). Off-loop (the file read is sync I/O). Never raises --
        an unreadable/missing decisions.jsonl degrades to an empty list, same
        as every other readmodel reader's own contract."""
        try:
            cfg = self._cfg or self._config_loader()
            from agent.dpof import readmodel  # lazy -- module-boundary discipline

            loop = asyncio.get_running_loop()
            rows = await loop.run_in_executor(
                None, lambda: readmodel.position_reviews(position_id, ddir=cfg.data_dir)
            )
        except Exception as e:
            logger.debug(f"DPOF engine: _read_position_reviews failed for {position_id}: {type(e).__name__}: {e}")
            return []
        if not isinstance(rows, list):
            return []
        return [
            r for r in rows
            if isinstance(r, dict) and r.get("kind") == "position_review" and r.get("position_id") == position_id
        ]

    def _maybe_add_lessons_ctx(self, ctx: dict, underlying: str, cfg: DPOFConfig) -> None:
        """§4.6: sets `ctx["lessons"]` from the in-memory lesson cache (no
        I/O -- `_build_ctx` itself stays sync and I/O-free) ONLY when there
        is at least one same/cross lesson or a non-empty tally to show;
        otherwise leaves `ctx` untouched so an entry/review message with no
        stored lessons yet renders byte-identical to before this feature
        existed (`prompts._format_lessons` also independently guards against
        an empty section, but this avoids even setting the key)."""
        if not cfg.lessons_in_prompt or self._reflection is None:
            return
        try:
            lessons = self._reflection.lessons_ctx(
                underlying, same_n=cfg.lessons_same_underlying, cross_n=cfg.lessons_cross,
            )
        except Exception as e:
            logger.debug(f"DPOF engine: lessons_ctx failed for {underlying}: {type(e).__name__}: {e}")
            return
        if not isinstance(lessons, dict):
            return
        if lessons.get("same") or lessons.get("cross") or lessons.get("tally"):
            ctx["lessons"] = {**lessons, "max_chars": cfg.lessons_prompt_max_chars}

    def _maybe_schedule_reflection(self, now: float, cfg: DPOFConfig) -> None:
        """§4.3 scheduling: fires `ReflectionService.run_pending` as
        fire-and-forget background work -- NEVER awaited by the cycle. The
        engine-level single-flight guard (`_reflection_task`) is separate
        from ReflectionService's own internal `_running` flag: this one stops
        the cycle from even SPAWNING a redundant Task while one is still in
        flight (ReflectionService's own flag is what makes a same-process
        double-call safe regardless)."""
        if not getattr(cfg, "reflection_enabled", True):
            return
        if self._reflection is None:
            return
        if self._reflection_task is not None and not self._reflection_task.done():
            return
        task = self._fire_and_forget(self._reflection.run_pending(list(self._ledger.positions), now))
        self._reflection_task = task
        task.add_done_callback(self._on_reflection_task_done)

    def _on_reflection_task_done(self, task: "asyncio.Task") -> None:
        if self._reflection_task is task:
            self._reflection_task = None

    # ------------------------------------------------------------------
    # decision-record producer -- outcome linkage (status-driven,
    # scheduled next to _maybe_schedule_reflection at step 13c). Joins a
    # closed position carrying a decision_call_id (its entry-veto call) to a
    # decisions.outcomes row. Never re-records a position once its id is in
    # `_decision_outcome_ids` -- the position-outcome row's own uuid5 id
    # (records.position_outcome_id) makes a LOST/reset id-set harmless (a
    # resend is a DB-side no-op via ignore-duplicates, never a duplicate row).
    # ------------------------------------------------------------------

    def _maybe_schedule_decision_outcomes(self, now: float, cfg: DPOFConfig) -> None:
        if not self._decisions_enabled():
            return
        if self._decision_outcomes_task is not None and not self._decision_outcomes_task.done():
            return
        task = self._fire_and_forget(self._scan_decision_outcomes(now))
        self._decision_outcomes_task = task
        task.add_done_callback(self._on_decision_outcomes_task_done)

    def _on_decision_outcomes_task_done(self, task: "asyncio.Task") -> None:
        if self._decision_outcomes_task is task:
            self._decision_outcomes_task = None

    async def _scan_decision_outcomes(self, now: float) -> None:
        try:
            await self._scan_decision_outcomes_impl(now)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"DPOF engine: decision-outcome scan failed: {type(e).__name__}: {e}", exc_info=True)

    async def _scan_decision_outcomes_impl(self, now: float) -> None:
        if self._decisions is None or self._ledger is None:
            return

        ids = self._decision_outcome_ids
        # The DecisionsConfig backing `self._decisions` -- a fake injected in
        # tests may not carry a real one, which degrades persistence to
        # in-memory-only tracking for the lifetime of this engine instance
        # (fine for tests; the real DecisionRecorder always has one).
        d_cfg = getattr(self._decisions, "cfg", None)
        if ids is None:
            ids = set()
            if d_cfg is not None:
                try:
                    from agent.decisions.store import load_recorded_outcome_ids_sync

                    loop = asyncio.get_running_loop()
                    ids = await loop.run_in_executor(None, load_recorded_outcome_ids_sync, d_cfg)
                except Exception as e:
                    logger.debug(f"DPOF engine: loading recorded_outcome_ids failed: {type(e).__name__}: {e}")
                    ids = set()
            self._decision_outcome_ids = ids

        newly_recorded = False
        for pos in list(self._ledger.positions):
            if pos.status != "closed" or not pos.id or pos.id in ids:
                continue
            entry_decision = pos.entry_decision if isinstance(pos.entry_decision, dict) else {}
            # None for a position entered before this feature, or while recording was off: the row is
            # still written, keyed by position_id, so its exit-sentinel questions (recorded with the
            # same position_id) can still be joined to an outcome.
            call_id = entry_decision.get("decision_call_id") or None
            try:
                self._decisions.record_position_outcome(pos, outcome_source="live", call_id=call_id)
            except Exception as e:
                logger.error(f"DPOF engine: record_position_outcome failed for {pos.id}: {type(e).__name__}: {e}")
                continue
            ids.add(pos.id)
            newly_recorded = True

        if newly_recorded and d_cfg is not None:
            try:
                from agent.decisions.store import save_recorded_outcome_ids_sync

                loop = asyncio.get_running_loop()
                await asyncio.shield(loop.run_in_executor(None, save_recorded_outcome_ids_sync, d_cfg, set(ids)))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"DPOF engine: persisting recorded_outcome_ids failed: {type(e).__name__}: {e}")

    @staticmethod
    def _debate_summary(debate_result: Any) -> dict:
        """§5.2/§5.3: the compact record stored in a decision Task's meta and
        copied into that deliberation's/review's decision_log as `debate:
        {ran, available, latency_ms, bull_strength, bear_strength, reason}`.
        Only ever built when the debate gate fired and the wrapper actually
        ran (`ran` is always True here) -- see _run_entry_debate_then_decide /
        _run_review_debate_then_decide, the only callers."""
        if not isinstance(debate_result, dict):
            return {"ran": True, "available": False, "latency_ms": None, "bull_strength": None, "bear_strength": None, "reason": "malformed_result"}
        bull = debate_result.get("bull")
        bear = debate_result.get("bear")
        return {
            "ran": True,
            "available": bool(debate_result.get("available")),
            "latency_ms": debate_result.get("latency_ms"),
            "bull_strength": bull.get("strength") if isinstance(bull, dict) else None,
            "bear_strength": bear.get("strength") if isinstance(bear, dict) else None,
            "reason": debate_result.get("reason", ""),
        }

    @staticmethod
    def _build_debate_event_summary(ctx: dict) -> str:
        """A short, human-readable event-risk line for the debate input
        (§5.2): the next high-importance release (or, absent one, the
        soonest upcoming release) plus whether breaking news is active right
        now. Never raises -- degrades to "" on any malformed/missing ctx
        shape (this is observability-adjacent context for the debate, never
        a gate)."""
        try:
            if not isinstance(ctx, dict):
                return ""
            events = ctx.get("events")
            breaking = ctx.get("breaking")
            parts: list[str] = []
            upcoming = (events or {}).get("upcoming") if isinstance(events, dict) else None
            if isinstance(upcoming, list) and upcoming:
                top = next((e for e in upcoming if isinstance(e, dict) and e.get("importance") == 3), None)
                if top is None:
                    top = next((e for e in upcoming if isinstance(e, dict)), None)
                if isinstance(top, dict):
                    mins = top.get("minutes_until")
                    name = top.get("event") or "event"
                    importance = top.get("importance")
                    if isinstance(mins, (int, float)) and not isinstance(mins, bool):
                        parts.append(
                            f"next release: {name} in {mins:.0f}min "
                            f"(importance={importance if importance is not None else 'n/a'})"
                        )
            if isinstance(breaking, dict) and breaking.get("active"):
                parts.append(f"breaking news active, max_severity={breaking.get('max_severity', 0)}")
            return "; ".join(parts)
        except Exception:
            return ""

    async def _run_entry_debate_then_decide(
        self, underlying: str, ctx: dict, candidates: list, signal: Signal, underlyings_ctx: dict, cfg: DPOFConfig,
    ):
        """§5.2 entry wrapper: runs the bull/bear debate INSIDE this decision
        Task (never inline in the cycle), sets `ctx["debate"]` when
        available, stashes a compact summary in `_entry_task_meta[underlying]`
        for `_apply_entry_outcome` to copy into decisions.jsonl, then makes
        the SAME `decide_entry` call as the no-debate path. Any failure in
        the debate step degrades to "no debate" and still calls decide_entry
        -- this function must never raise except CancelledError."""
        debate_result: dict = {"available": False, "reason": "not_run"}
        try:
            u_ctx = underlyings_ctx.get(underlying) or {}
            candidates_summary = [
                {
                    "key": getattr(c, "key", None), "contract": getattr(c, "strike_desc", None),
                    "unit_premium": getattr(c, "unit_premium", None), "delta": getattr(c, "delta", None),
                    "max_affordable_qty": getattr(c, "max_affordable_qty", None),
                }
                for c in (candidates or [])
            ]
            lessons_ctx = ctx.get("lessons")
            lessons_tally = lessons_ctx.get("tally") if isinstance(lessons_ctx, dict) else None
            price_vs_vwap = {"above_vwap": u_ctx.get("above_vwap"), "ret_15m": u_ctx.get("ret_15m")}
            vs_ctx = ctx.get("vol_surface")
            vol_line = _format_vol_surface_line(underlying, (vs_ctx or {}).get(underlying) if isinstance(vs_ctx, dict) else None)
            event_summary = self._build_debate_event_summary(ctx)
            debate_input = build_debate_input(
                underlying=underlying, direction=signal.direction.value, composite=signal.composite,
                strength=signal.strength, pillars=u_ctx.get("pillars"), price_vs_vwap=price_vs_vwap,
                regime=(ctx.get("regime") or {}).get("label", ""), vol_surface_line=vol_line,
                event_risk_summary=event_summary, candidates=candidates_summary, lessons_tally=lessons_tally,
            )
            debate_result = await run_debate(
                self._advisory_debate, kind="entry", debate_input=debate_input,
                model=cfg.debate_model, fallback_model=cfg.debate_fallback_model,
                timeout_seconds=cfg.debate_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.info(f"DPOF engine: entry debate failed for {underlying}: {type(e).__name__}: {e}")
            debate_result = {"available": False, "reason": f"error:{type(e).__name__}"}

        self._entry_task_meta.setdefault(underlying, {})["debate_summary"] = self._debate_summary(debate_result)
        if isinstance(debate_result, dict) and debate_result.get("available"):
            ctx["debate"] = {**debate_result, "rendered": render_debate_section(debate_result)}

        return await self._decider.decide_entry(ctx, candidates)

    async def _run_review_debate_then_decide(
        self, pos: PositionRecord, ctx: dict, position_dict: dict, underlyings_ctx: dict, cfg: DPOFConfig,
    ):
        """§5.2 review wrapper -- same shape as the entry wrapper above, but
        against the one open position under review instead of a candidate
        menu."""
        debate_result: dict = {"available": False, "reason": "not_run"}
        try:
            u_ctx = underlyings_ctx.get(pos.underlying) or {}
            lessons_ctx = ctx.get("lessons")
            lessons_tally = lessons_ctx.get("tally") if isinstance(lessons_ctx, dict) else None
            price_vs_vwap = {"above_vwap": u_ctx.get("above_vwap"), "ret_15m": u_ctx.get("ret_15m")}
            vs_ctx = ctx.get("vol_surface")
            vol_line = _format_vol_surface_line(pos.underlying, (vs_ctx or {}).get(pos.underlying) if isinstance(vs_ctx, dict) else None)
            event_summary = self._build_debate_event_summary(ctx)
            position_summary = {
                "underlying": pos.underlying, "direction": pos.direction, "structure": pos.structure,
                "qty_open": pos.qty_open, "mark": position_dict.get("mark"), "pnl_pct": position_dict.get("pnl_pct"),
                "thesis": pos.thesis, "invalidation": pos.invalidation,
            }
            debate_input = build_debate_input(
                underlying=pos.underlying, direction=pos.direction, composite=u_ctx.get("composite"),
                strength=u_ctx.get("strength", "none"), pillars=u_ctx.get("pillars"), price_vs_vwap=price_vs_vwap,
                regime=(ctx.get("regime") or {}).get("label", ""), vol_surface_line=vol_line,
                event_risk_summary=event_summary, position=position_summary, lessons_tally=lessons_tally,
            )
            debate_result = await run_debate(
                self._advisory_debate, kind="review", debate_input=debate_input,
                model=cfg.debate_model, fallback_model=cfg.debate_fallback_model,
                timeout_seconds=cfg.debate_review_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.info(f"DPOF engine: review debate failed for {pos.id}: {type(e).__name__}: {e}")
            debate_result = {"available": False, "reason": f"error:{type(e).__name__}"}

        self._review_task_meta.setdefault(pos.id, {})["debate_summary"] = self._debate_summary(debate_result)
        if isinstance(debate_result, dict) and debate_result.get("available"):
            ctx["debate"] = {**debate_result, "rendered": render_debate_section(debate_result)}

        return await self._decider.review_position(ctx, position_dict)

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    async def _safe_alert(
        self, bot, text: Optional[str], dedupe_key: Optional[str] = None, dedupe_seconds: int = 86400,
    ) -> None:
        if text is None or self._alerts is None:
            return
        try:
            await self._alerts.send(bot, text, dedupe_key=dedupe_key, dedupe_seconds=dedupe_seconds)
        except Exception as e:
            logger.error(f"DPOF engine: alert send failed: {type(e).__name__}: {e}")

    async def _handle_model_down(self, bot, cfg: DPOFConfig) -> None:
        self._model_down = True
        n_open = len(self._ledger.open_positions()) if self._ledger is not None else 0
        await self._safe_alert(
            bot, fmt_model_down(n_open, cfg.flatten_et),
            dedupe_key="dpof_model_down", dedupe_seconds=max(60, cfg.model_down_alert_minutes * 60),
        )

    async def _handle_model_recovered_if_needed(self, bot) -> None:
        if self._model_down:
            self._model_down = False
            await self._safe_alert(bot, fmt_model_recovered(), dedupe_key="dpof_model_recovered", dedupe_seconds=60)

    async def _handle_review_unavailable(self, pos: PositionRecord, now: float, bot, cfg: DPOFConfig, summary: dict) -> None:
        """DecisionUnavailable on a review -> HOLD, review_failures += 1, retry
        after ~60s (not the full review_interval_minutes)."""
        updated = replace(pos, review_failures=pos.review_failures + 1, next_review_at=now + _REVIEW_RETRY_AFTER_UNAVAILABLE_SECONDS)
        try:
            await asyncio.shield(self._ledger.upsert(updated))
        except Exception as e:
            logger.error(f"DPOF engine: failed to persist review_failures for {pos.id}: {type(e).__name__}: {e}")
            summary["errors"].append(f"persist_review_failure:{pos.id}")
        await self._handle_model_down(bot, cfg)

    # ------------------------------------------------------------------
    # Pure-ish helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _position_premium(pos: PositionRecord) -> float:
        """Dollar premium a position contributes to open_premium. A PENDING
        position is priced at qty_original x (fill avg if any, else the working
        entry limit recorded by OrderManager) so an unfilled entry still counts
        against the open-premium cap."""
        if pos.status == "pending":
            qty = pos.qty_original
            price = pos.entry_price if pos.entry_price is not None else pos.entry_limit_price
            if price is None:
                return 0.0
            try:
                return float(price) * 100.0 * qty
            except (TypeError, ValueError):
                return 0.0
        price = pos.entry_price
        if price is None:
            return 0.0
        return float(price) * 100.0 * pos.qty_open

    @staticmethod
    def _mark_for(pos: PositionRecord, quotes: dict) -> Optional[float]:
        legs = pos.legs
        if not legs:
            return None
        if pos.structure == "debit_vertical":
            long_leg = next((l for l in legs if l.get("position_intent") == "buy_to_open"), None)
            short_leg = next((l for l in legs if l.get("position_intent") == "sell_to_open"), None)
            if long_leg is None or short_leg is None:
                return None
            ql = quotes.get(long_leg.get("symbol"))
            qs = quotes.get(short_leg.get("symbol"))
            if ql is None or qs is None or ql.mid is None or qs.mid is None:
                return None
            return ql.mid - qs.mid
        q = quotes.get(legs[0].get("symbol"))
        return q.mid if q is not None else None

    def _safe_market_open(self) -> bool:
        try:
            return bool(self._market_open_fn())
        except Exception as e:
            logger.error(f"DPOF engine: market_open_fn raised ({type(e).__name__}: {e}); treating as closed")
            return False

    def _minutes_to_flatten(self, now_et: datetime, cfg: DPOFConfig) -> float:
        t = risk_module.parse_hhmm(cfg.flatten_et)
        flatten_dt = now_et.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        return max(0.0, (flatten_dt - now_et).total_seconds() / 60.0)

    def _market_regime_label(self) -> Optional[str]:
        try:
            from agent.integrations.market_regime import get_market_regime_label

            return get_market_regime_label()
        except Exception as e:
            logger.debug(f"DPOF engine: market_regime unavailable: {type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------------
    # Decision task plumbing (shared by entries + reviews)
    # ------------------------------------------------------------------

    async def _run_decision_task(self, coro) -> dict:
        """Wraps a decider call so its Task NEVER raises: any exception
        (including the real/local DecisionUnavailable) is captured into the
        result dict and classified via `unavailable`. CancelledError is
        re-raised (CLAUDE.md #35) — asyncio.gather(return_exceptions=True) in
        _cancel_all_decision_tasks handles that at the call site."""
        started = self._clock()
        try:
            result = await coro
            return {"ok": True, "result": result, "error": None, "unavailable": False, "started": started, "finished": self._clock()}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            unavailable = isinstance(e, self._decision_unavailable_exc)
            return {
                "ok": False, "result": None, "error": f"{type(e).__name__}: {e}",
                "unavailable": unavailable, "started": started, "finished": self._clock(),
            }

    async def _cancel_all_decision_tasks(self) -> None:
        tasks = [t for t in list(self._entry_tasks.values()) + list(self._review_tasks.values()) if not t.done()]
        self._entry_tasks.clear()
        self._entry_task_meta.clear()
        self._review_tasks.clear()
        self._review_task_meta.clear()
        # Learning loop: the in-flight reflection run_pending() Task (if any)
        # is a background job of the same shape -- cancel it alongside the
        # entry/review decision tasks (shutdown() and mode->OFF both need
        # this). Its own done-callback (_on_reflection_task_done) still fires
        # on cancellation, clearing _reflection_task.
        if self._reflection_task is not None and not self._reflection_task.done():
            tasks.append(self._reflection_task)
        # decision-record producer: the outcome-linkage scan is the
        # same shape of background job -- cancel it alongside reflection.
        if self._decision_outcomes_task is not None and not self._decision_outcomes_task.done():
            tasks.append(self._decision_outcomes_task)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _maybe_alert_kronos_stale(
        self, snap: Optional[MarketSnapshot], now: float, now_et: datetime, cfg: DPOFConfig, bot,
    ) -> None:
        """Kronos live pillar (docs/design/clio_kronos.md §4b): one Telegram alert per session day when no
        underlying has had a fresh Kronos forecast for 15+ minutes inside the entry window -- the pillar
        silently drops out of the composite while the service is down, so say so. Never raises."""
        try:
            if cfg.kronos_pillar != "live" or snap is None or not _in_entry_window(now_et, cfg):
                self._kronos_missing_since = None
                return
            present = any(KRONOS_PILLAR in (u.pillars or {}) for u in snap.underlyings.values())
            if present:
                self._kronos_missing_since = None
                return
            if self._kronos_missing_since is None:
                self._kronos_missing_since = now
                return
            if now - self._kronos_missing_since >= _KRONOS_STALE_ALERT_SECONDS:
                await self._safe_alert(
                    bot,
                    f"[{cfg.mode.value.upper()}] DPOF: no fresh Kronos forecast for "
                    f"{int((now - self._kronos_missing_since) // 60)}+ min -- the kronos pillar is out of the "
                    f"composite until it returns (check clio-kronos.service).",
                    dedupe_key=f"dpof_kronos_stale_{now_et.date().isoformat()}",
                    dedupe_seconds=86400,
                )
        except Exception as e:
            logger.debug(f"DPOF engine: kronos stale check failed: {type(e).__name__}: {e}")

    def _decider_available(self) -> bool:
        return self._decider is not None

    def _decisions_enabled(self) -> bool:
        """decision-record producer: cheap enabled-check used to
        skip building a request (and any side-effecting collaborator calls
        that goes through, e.g. position_alignment_fn) entirely when
        recording is disabled -- not just a no-op INSIDE record_call. A fake
        recorder without a `.cfg` (as tests may inject) is treated as
        enabled, matching DecisionsConfig's own default."""
        if self._decisions is None:
            return False
        d_cfg = getattr(self._decisions, "cfg", None)
        if d_cfg is None:
            return True
        return bool(getattr(d_cfg, "enabled", True))

    # ------------------------------------------------------------------
    # Context building for the decision model
    # ------------------------------------------------------------------

    def _regime_ctx(self, snap: Optional[MarketSnapshot], cfg: DPOFConfig) -> dict:
        """{"vix": ..., "vix_age_seconds": ..., "label": ...} -- the VIX
        value/regime label for the decider ctx (Task 2's `regime` key),
        including the former HALT level presented as "EXTREME" (Task 1;
        see _REGIME_LABELS). Pure -- no I/O, mirrors _build_underlyings_ctx's
        own classify_regime() call (kept separate: that one is per-underlying/
        embedded for back-compat, this is the new top-level `ctx["regime"]`)."""
        if snap is None:
            return {"vix": None, "vix_age_seconds": None, "label": _REGIME_LABELS[Regime.UNKNOWN]}
        regime = classify_regime(snap.vix, snap.vix_age_seconds, cfg)
        return {
            "vix": snap.vix,
            "vix_age_seconds": snap.vix_age_seconds,
            "label": _REGIME_LABELS.get(regime, regime.value.upper()),
        }

    async def _build_intel_ctx(
        self, cfg: DPOFConfig, snap: Optional[MarketSnapshot], now_epoch: float,
        intel_task: Optional["asyncio.Task"] = None,
    ) -> dict:
        """Task 2 (INTEL_V3_SPEC.md / G prompt): advisory intel merged into
        every decider ctx (see _build_ctx) -- events (upcoming releases +
        recent surprises), vol_surface, breaking, regime, calendar_status,
        and (Task K2) kronos. Computed ONCE per cycle (_run_cycle_locked, not
        per entry-deliberation/review launch) and cached on self._last_intel_ctx
        so _build_ctx/_review_triggers/_cooldown_allows all see the SAME
        snapshot of intel within one cycle. Never raises, never blocks the
        cycle -- self._feed.get_intel_ctx() already degrades internally; this
        wraps it in try/except as defense in depth (e.g. self._feed is None
        very early in a cycle, or a future refactor of get_intel_ctx breaks
        its own never-raises contract) and always returns a COMPLETE shape
        (every key present, defaulting to an empty/None value) so downstream
        readers never need extra None-guards beyond `dict.get`. "kronos" is
        the one exception to "every key present, non-None default": it
        defaults to None (feed omitted it, meaning CLIO_DPOF_KRONOS_CONTEXT
        is off, or self._feed/get_intel_ctx is unavailable) -- _build_ctx
        reads that None to decide whether the decider ctx carries a "kronos"
        key at all (see its own docstring).
        """
        intel: dict = {}
        if intel_task is not None:
            # Started by _run_cycle_locked concurrently with feed.snapshot() so the
            # two budgets (snapshot 15 s, intel 8 s) overlap instead of stacking
            # inside DPOF_JOB_TIMEOUT.
            try:
                intel = await intel_task
            except Exception as e:
                logger.error(f"DPOF engine: get_intel_ctx failed: {type(e).__name__}: {e}", exc_info=True)
                intel = {}
        elif self._feed is not None:
            try:
                intel = await self._feed.get_intel_ctx(now_epoch)
            except Exception as e:
                logger.error(f"DPOF engine: get_intel_ctx failed: {type(e).__name__}: {e}", exc_info=True)
                intel = {}
        if not isinstance(intel, dict):
            intel = {}
        return {
            "events": intel.get("events") or {"upcoming": [], "recent_surprises": [], "source": "unavailable"},
            "vol_surface": intel.get("vol_surface") or {sym: None for sym in cfg.underlyings},
            "breaking": intel.get("breaking") or {
                "active": False, "max_severity": 0, "events": [], "as_of": None, "stale": True, "source": "unavailable",
            },
            "calendar_status": intel.get("calendar_status") or {"status": "missing", "age_hours": None, "sources": []},
            "regime": self._regime_ctx(snap, cfg),
            "kronos": intel.get("kronos"),
        }

    def _build_underlyings_ctx(self, snap: MarketSnapshot, cfg: DPOFConfig, signal: Optional[Signal]) -> dict:
        regime = classify_regime(snap.vix, snap.vix_age_seconds, cfg)
        out: dict[str, dict] = {}
        for sym, u in snap.underlyings.items():
            composite, pillars_used, _reasons = compute_composite(u, cfg)
            above_vwap = None
            if u.last_price is not None and u.vwap is not None:
                above_vwap = u.last_price > u.vwap
            strength = "none"
            if composite is not None:
                ac = abs(composite)
                if ac >= cfg.threshold_strong:
                    strength = "strong"
                elif ac >= cfg.threshold_normal:
                    strength = "normal"
            dp_veto = False
            dp = pillars_used.get("darkpool")
            if composite is not None and dp is not None:
                csign = 1 if composite > 0 else (-1 if composite < 0 else 0)
                if dp * csign <= -cfg.dp_veto_threshold:
                    dp_veto = True
            direction_nominated = None
            if signal is not None and signal.symbol == sym and signal.direction != Direction.NEUTRAL:
                direction_nominated = signal.direction.value
            out[sym] = {
                "composite": composite,
                "pillars": pillars_used,
                "direction": direction_nominated,
                "strength": strength,
                "dp_veto": dp_veto,
                "vwap": u.vwap,
                "last": u.last_price,
                "above_vwap": above_vwap,
                "ret_5m": u.ret_5m,
                "ret_15m": u.ret_15m,
                "gex_regime": u.gex_regime,
                "nope": u.nope,
                "max_pain": u.max_pain,
                "regime": regime.value,
                "vix": snap.vix,
            }
            # Self-computed flow fallback (Task F, 2026-09-25, operator
            # directive -- UW subscription/key reissue in progress): plain
            # words for the decider/Jev whenever data.py filled >=1 pillar/
            # nope/max_pain for this underlying from agent.flowcollector
            # instead of UW this cycle (data.UnderlyingSnapshot.flow_selfflow
            # -- see docs/design/clio_dpof.md's "Flow fallback" section).
            # Static CHARTER text stays prompt-cached; this note is per-
            # underlying, per-cycle context only -- omitted entirely (never a
            # blank/false value) when the fallback isn't in use this cycle.
            if getattr(u, "flow_selfflow", False):
                out[sym]["flow_note"] = (
                    "flow pillars self-computed from Alpaca OPRA/SIP trades (quote-rule "
                    "classification; not validated against Unusual Whales)"
                )
        return out

    def _build_ctx(
        self, now_et: datetime, cfg: DPOFConfig, snap: Optional[MarketSnapshot], underlyings_ctx: dict,
        market_regime: Optional[str], risk: RiskState, open_positions: list[PositionRecord],
        dpof_equity: Optional[float], open_premium: float,
    ) -> dict:
        open_lines = []
        for p in open_positions:
            mark = self._last_marks.get(p.id)
            pnl_pct = position_pnl_pct(p.entry_price, mark)
            open_lines.append({
                "id": p.id, "underlying": p.underlying, "direction": p.direction, "structure": p.structure,
                "qty_open": p.qty_open, "qty_original": p.qty_original, "entry_price": p.entry_price,
                "mark": mark, "pnl_pct": pnl_pct, "status": p.status, "thesis": p.thesis,
            })

        recent_closed: list[dict] = []
        try:
            from agent.dpof import readmodel  # lazy -- module-boundary discipline

            closed = sorted(
                (p for p in self._ledger.positions if p.status == "closed" and p.closed_time is not None),
                key=lambda p: p.closed_time, reverse=True,
            )[:5]
            for p in closed:
                held_min = None
                if p.entry_time is not None and p.closed_time is not None:
                    held_min = (p.closed_time - p.entry_time) / 60.0
                recent_closed.append({
                    "label": readmodel.position_label(p), "pnl": p.realized_pnl,
                    "exit_reason": p.exit_reason, "held_minutes": held_min,
                })
        except Exception as e:
            logger.debug(f"DPOF engine: recent_closed build failed: {type(e).__name__}: {e}")

        # Raw strings (uncapped only by count) -- the decider (D1) sanitizes
        # for the LLM via sanitize_ingested_content(..., source="dpof_headlines")
        # per DPOF_V2_SPEC.md's Interfaces section; this layer never sanitizes.
        raw_headlines = list(snap.headlines) if snap is not None else []
        headlines = [str(h) for h in raw_headlines[: max(0, cfg.jev_max_headlines)]]
        econ_events = list(snap.econ_events_soon) if snap is not None else []

        # Task 2 (INTEL_V3_SPEC.md / G prompt): advisory intel computed ONCE
        # per cycle by _build_intel_ctx, merged in here unchanged. Defaults
        # match _build_intel_ctx's own "complete shape" contract in case a
        # cycle hasn't run yet (e.g. a direct unit-test call to _build_ctx).
        intel = self._last_intel_ctx or {}

        out = {
            "now_et": now_et.isoformat(),
            "minutes_to_flatten": self._minutes_to_flatten(now_et, cfg),
            "mode": cfg.mode.value,
            "underlyings": underlyings_ctx,
            "market_regime": market_regime,
            "open_positions": open_lines,
            # Operator directive (no-hard-caps amendment): no per-trade/open-
            # premium/count caps any more -- the decision model sizes freely;
            # `capital_available` is the ONE number that actually bounds it
            # (solvency, enforced fresh again at apply time).
            "capital": {
                "dpof_equity": dpof_equity,
                "capital_available": (None if dpof_equity is None else dpof_equity - open_premium),
                "open_premium": open_premium,
                "open_positions_count": len(open_positions),
                "entries_today": risk.entries_today,
                "realized_today": risk.realized_pnl_today,
            },
            "recent_closed": recent_closed,
            "headlines": headlines,
            "econ_events": econ_events,
            "jev": (self._last_jev_verdict.to_dict() if self._last_jev_verdict is not None else None),
            "events": intel.get("events") or {"upcoming": [], "recent_surprises": [], "source": "unavailable"},
            "vol_surface": intel.get("vol_surface") or {},
            "breaking": intel.get("breaking") or {
                "active": False, "max_severity": 0, "events": [], "as_of": None, "stale": True, "source": "unavailable",
            },
            "regime": intel.get("regime") or {"vix": None, "vix_age_seconds": None, "label": _REGIME_LABELS[Regime.UNKNOWN]},
            "calendar_status": intel.get("calendar_status") or {"status": "missing", "age_hours": None, "sources": []},
        }
        # Task K2 (agent/kronos): the "kronos" key is present ONLY when intel
        # actually carries one (CLIO_DPOF_KRONOS_CONTEXT is on and the read
        # succeeded/degraded to per-symbol "unavailable") -- absent entirely
        # when the flag is off, so a disabled config renders a decider
        # message byte-identical to before this feature existed (see
        # agent.dpof.prompts._format_kronos).
        kronos_ctx = intel.get("kronos")
        if kronos_ctx is not None:
            out["kronos"] = kronos_ctx
        # Advisory size anchor (operator 2026-09-29): present ONLY when enabled, so a disabled config renders
        # a decider message byte-identical to before (agent.dpof.prompts._format_size_anchor).
        if cfg.size_anchor_enabled:
            out["size_anchor"] = {
                "share_c50": cfg.size_anchor_share_c50,
                "share_c80": cfg.size_anchor_share_c80,
            }
        return out

    def _build_jev_position_ctx(
        self, pos: PositionRecord, pnl_pct: Optional[float], now_et: datetime, cfg: DPOFConfig,
        snap: Optional[MarketSnapshot], underlyings_ctx: dict,
    ) -> dict:
        alignment: dict = {}
        u_ctx = underlyings_ctx.get(pos.underlying)
        if u_ctx is not None:
            try:
                alignment = self._position_alignment_fn(u_ctx, Direction(pos.direction))
            except Exception as e:
                logger.debug(f"DPOF engine: position_alignment_fn failed for {pos.id}: {type(e).__name__}: {e}")
                alignment = {}
        headlines = list(snap.headlines) if snap is not None else []
        return {
            "direction": pos.direction,
            "structure": pos.structure,
            "pnl_pct": pnl_pct,
            "minutes_to_flatten": self._minutes_to_flatten(now_et, cfg),
            "flow_vs_position": alignment.get("flow", "neutral"),
            "momentum_vs_position": alignment.get("momentum", "neutral"),
            "vwap_vs_position": alignment.get("vwap", "neutral"),
            "headlines": headlines,
        }

    def _build_review_position_dict(
        self, pos: PositionRecord, mark: Optional[float], pnl_pct: Optional[float], now: float, triggers: list[str],
    ) -> dict:
        d = pos.to_dict()
        pnl_usd = None
        if mark is not None and pos.entry_price is not None:
            pnl_usd = (mark - pos.entry_price) * 100.0 * pos.qty_open
        minutes_held = None
        if pos.entry_time is not None:
            minutes_held = (now - pos.entry_time) / 60.0
        d.update({
            "mark": mark, "pnl_pct": pnl_pct, "pnl_usd": pnl_usd, "minutes_held": minutes_held,
            "trigger": triggers, "thesis": pos.thesis, "invalidation": pos.invalidation, "last_review": pos.last_review,
        })
        return d

    # ------------------------------------------------------------------
    # Position reviews (step 8c) — Jev SENTINEL + review triggers + the
    # decider's ReviewDecision. NEVER Jev-vetoed; NEVER a mechanical exit
    # except the EOD flatten (evaluate_exit), checked first every cycle.
    # ------------------------------------------------------------------

    async def _jev_sentinel_screen(
        self, pos: PositionRecord, pnl_pct: Optional[float], now_et: datetime, cfg: DPOFConfig,
        snap: Optional[MarketSnapshot], underlyings_ctx: dict,
    ) -> bool:
        """Returns True iff Jev recommends escalating (pulling the next review
        forward to now). Never raises (jev_position_check's own contract);
        logs the escalation record to the ledger like an entry verdict.

        decision-record producer: recorded even when Jev is OFF --
        the method must still return False exactly as before in that case
        (no behavior change to the sentinel itself), but the position request
        is still built and recorded (when decisions are enabled) so the
        production eval set doesn't depend on Jev being keyed."""
        if cfg.jev_effective_mode == JevMode.OFF:
            self._record_sentinel_decision_off(pos, pnl_pct, now_et, cfg, snap, underlyings_ctx)
            return False
        position_ctx = self._build_jev_position_ctx(pos, pnl_pct, now_et, cfg, snap, underlyings_ctx)
        try:
            esc = await self._jev_sentinel(cfg, self._jev, position_ctx)
        except Exception as e:
            logger.debug(f"DPOF engine: jev_sentinel raised unexpectedly for {pos.id}: {type(e).__name__}: {e}")
            return False
        if esc is None:
            return False
        skipped = bool(getattr(esc, "skipped", False))
        if not skipped and hasattr(esc, "to_dict"):
            try:
                await asyncio.shield(self._ledger.append_jev(esc.to_dict()))
            except Exception as e:
                logger.error(f"DPOF engine: append_jev (sentinel) failed for {pos.id}: {type(e).__name__}: {e}")
        self._record_sentinel_decision(esc, pos, position_ctx, cfg, now_et)
        return bool(getattr(esc, "escalate", False))

    def _record_sentinel_decision(
        self, esc: Any, pos: PositionRecord, position_ctx: dict, cfg: DPOFConfig, now_et: datetime,
    ) -> None:
        """decision-record producer -- fire-and-forget (record_call
        is SYNC and never raises); never affects the sentinel's own result."""
        if not self._decisions_enabled():
            return
        try:
            request = getattr(esc, "request", None)
            if request is None:
                request = build_position_request(cfg, position_ctx)
            state, questions = request["state"], request["questions"]
            served = None
            if not bool(getattr(esc, "skipped", False)):
                probs = getattr(esc, "probabilities", None) or {}
                answers = {qid: {"type": "noul", "noul": v} for qid, v in probs.items()}
                served = (
                    answers, getattr(esc, "model", "") or "jev-unknown", getattr(esc, "endpoint", ""),
                    getattr(esc, "latency_ms", None), getattr(esc, "error", ""),
                )
            producer_ref = {"cycle_ts": now_et.isoformat(), "underlying": pos.underlying}
            self._decisions.record_call(
                "dpof_exit_sentinel", producer_ref, state, questions, served=served, position_id=pos.id,
            )
        except Exception as e:
            logger.debug(f"DPOF engine: decisions record_call (sentinel) failed for {pos.id}: {type(e).__name__}: {e}")

    def _record_sentinel_decision_off(
        self, pos: PositionRecord, pnl_pct: Optional[float], now_et: datetime, cfg: DPOFConfig,
        snap: Optional[MarketSnapshot], underlyings_ctx: dict,
    ) -> None:
        """decision-record producer -- Jev is OFF, so build the
        position request and record it WITHOUT ever calling Jev (mirrors
        `prepare_entry_request`'s OFF-path role on the entry side). Gated on
        `_decisions_enabled()` (not just `self._decisions is not None`) so
        building the position context -- which calls the injected
        `position_alignment_fn` -- never happens at all when recording is
        disabled (CLIO_DECISIONS_ENABLED=0 is the test-suite default)."""
        if not self._decisions_enabled():
            return
        try:
            position_ctx = self._build_jev_position_ctx(pos, pnl_pct, now_et, cfg, snap, underlyings_ctx)
            request = build_position_request(cfg, position_ctx)
            producer_ref = {"cycle_ts": now_et.isoformat(), "underlying": pos.underlying}
            self._decisions.record_call(
                "dpof_exit_sentinel", producer_ref, request["state"], request["questions"],
                served=None, position_id=pos.id,
            )
        except Exception as e:
            logger.debug(f"DPOF engine: decisions record_call (sentinel, OFF) failed for {pos.id}: {type(e).__name__}: {e}")

    def _review_triggers(
        self, pos: PositionRecord, mark: Optional[float], now: float, jev_escalated: bool,
        snap: Optional[MarketSnapshot], underlyings_ctx: dict, cfg: DPOFConfig,
    ) -> list[str]:
        triggers: list[str] = []
        if pos.next_review_at is None or now >= pos.next_review_at:
            triggers.append("due")
        if jev_escalated:
            triggers.append("jev_escalation")
        if pos.last_review_mark is not None and pos.last_review_mark > 0 and mark is not None:
            move = abs(mark - pos.last_review_mark) / pos.last_review_mark
            if move >= cfg.review_trigger_move_pct:
                triggers.append("mark_move")
        u_ctx = underlyings_ctx.get(pos.underlying)
        if u_ctx is not None:
            try:
                alignment = self._position_alignment_fn(u_ctx, Direction(pos.direction))
                if alignment.get("flow") == "against" or alignment.get("vwap") == "against":
                    triggers.append("alignment_against")
            except Exception as e:
                logger.debug(f"DPOF engine: position_alignment_fn (trigger) failed for {pos.id}: {type(e).__name__}: {e}")
        if snap is not None and "breaking_news" in snap.risk_flags:
            triggers.append("breaking_news")

        # Task 3(a): a NEW recent surprise (|surprise_z| >= 1.5, or -- when z
        # is unavailable -- importance-3 with a non-inline direction) released
        # after this position's last review pulls a review forward. Baseline
        # falls back to entry_time when there has been no review yet (never
        # to "the epoch", which would spuriously fire on every historical
        # surprise the first time a fresh position is checked).
        intel = self._last_intel_ctx or {}
        baseline = pos.last_review_at if pos.last_review_at is not None else pos.entry_time
        if baseline is not None:
            for s in (intel.get("events") or {}).get("recent_surprises") or []:
                if not isinstance(s, dict):
                    continue
                released = _parse_iso_epoch(s.get("release_time_et"))
                if released is None or released <= baseline:
                    continue
                z = s.get("surprise_z")
                if isinstance(z, (int, float)) and not isinstance(z, bool):
                    qualifies = abs(z) >= 1.5
                else:
                    qualifies = s.get("importance") == 3 and s.get("direction") not in (None, "inline")
                if qualifies:
                    triggers.append("new_surprise")
                    break

        # Task 3(b): a NEW breaking event (id not seen before FOR THIS
        # POSITION) with severity >= 2 affecting this underlying or macro-wide
        # (no tickers listed on the event at all) pulls a review forward.
        # Every id seen this cycle is remembered so it never re-fires for this
        # same position, regardless of whether it qualified on severity/scope.
        breaking = intel.get("breaking") or {}
        seen = self._seen_breaking_event_ids.setdefault(pos.id, set())
        new_breaking = False
        for ev in breaking.get("events") or []:
            if not isinstance(ev, dict):
                continue
            ev_id = ev.get("id")
            if not ev_id or ev_id in seen:
                continue
            seen.add(ev_id)
            severity = ev.get("severity")
            tickers = ev.get("tickers") or []
            affects = (not tickers) or (pos.underlying in tickers)
            if isinstance(severity, (int, float)) and not isinstance(severity, bool) and severity >= 2 and affects:
                new_breaking = True
        if new_breaking:
            triggers.append("new_breaking_event")

        return triggers

    async def _handle_position_reviews(
        self, now: float, now_et: datetime, cfg: DPOFConfig, bot, marks: dict, snap: Optional[MarketSnapshot],
        underlyings_ctx: dict, market_regime: Optional[str], risk: RiskState, dpof_equity: Optional[float],
        summary: dict, exits_log: list[dict],
    ) -> None:
        # (i) Harvest completed reviews first, so an applied trim/exit is
        # reflected before this cycle's trigger-detection pass runs.
        for pos_id, task in list(self._review_tasks.items()):
            if not task.done():
                continue
            del self._review_tasks[pos_id]
            outcome = task.result()
            await self._apply_review_outcome(pos_id, outcome, now, now_et, cfg, bot, summary, exits_log)

        open_positions = self._ledger.open_positions()
        open_premium = sum(self._position_premium(p) for p in open_positions)

        # Drop per-position "seen breaking event" sets for positions that are no
        # longer open -- covers every close path (exit fill, cancelled entry,
        # reconcile()'s reconciled_absent), not just the exit_filled branch.
        _open_ids = {p.id for p in open_positions}
        for _pid in [k for k in self._seen_breaking_event_ids if k not in _open_ids]:
            del self._seen_breaking_event_ids[_pid]

        for pos in open_positions:
            mark = marks.get(pos.id)
            pnl_pct = position_pnl_pct(pos.entry_price, mark)

            # (ii) The one kept rail, checked BEFORE any judgment trigger --
            # for EVERY status open_positions() can return (pending/open/
            # closing), not just "open". Review finding (pass 2,
            # order-safety lens): evaluate_exit() itself is status-agnostic
            # (pure wall-clock arithmetic), but this loop used to gate the
            # whole flatten check on status=="open" first -- a still-PENDING
            # entry order (working, unfilled) was therefore never flattened
            # at all, and the ordinary reprice ladder in orders.py has no
            # wall-clock awareness of flatten_et, so it could keep working
            # (and could still fill) past 15:50 -- a 1DTE contract stranded
            # overnight, undiscovered until the next session (market_closed
            # stops work_orders() from being polled again once the exchange
            # shuts -- see step 4 above). See OrderManager.flatten_entry's
            # docstring for the full mechanism.
            flatten = evaluate_exit(pos, mark, now_et, cfg)
            if flatten.action == "close":
                stale = self._review_tasks.pop(pos.id, None)
                self._review_task_meta.pop(pos.id, None)
                if stale is not None and not stale.done():
                    stale.cancel()
                if pos.status == "open":
                    try:
                        ok = await self._orders.close_position(pos, flatten.qty, flatten.reason, flatten.urgent, now)
                        exits_log.append({"position_id": pos.id, "action": "close", "qty": flatten.qty, "reason": flatten.reason, "submitted": ok})
                    except Exception as e:
                        logger.error(f"DPOF engine: eod flatten close_position failed for {pos.id}: {type(e).__name__}: {e}", exc_info=True)
                        summary["errors"].append(f"close_position:{pos.id}")
                elif pos.status == "pending":
                    try:
                        cancelled = await self._orders.flatten_entry(pos, flatten.reason, now)
                        exits_log.append({"position_id": pos.id, "action": "flatten_entry", "reason": flatten.reason, "submitted": cancelled})
                    except Exception as e:
                        logger.error(f"DPOF engine: eod flatten flatten_entry failed for {pos.id}: {type(e).__name__}: {e}", exc_info=True)
                        summary["errors"].append(f"flatten_entry:{pos.id}")
                elif pos.status == "closing" and not pos.exit_urgent:
                    # An exit batch is already working (e.g. a review-launched
                    # trim/exit) -- never send a second, conflicting close.
                    # Past flatten_et, escalate it to urgent so orders.py's
                    # existing reprice-exhaustion ladder (exits.py owns no
                    # escalation logic itself) is allowed to go marketable/
                    # market once its budget runs out, same as a flatten-
                    # originated close would have been from the start.
                    try:
                        await asyncio.shield(self._ledger.upsert(replace(pos, exit_urgent=True)))
                        exits_log.append({"position_id": pos.id, "action": "flatten_escalate_closing", "reason": flatten.reason})
                    except Exception as e:
                        logger.error(f"DPOF engine: eod flatten escalate-closing failed for {pos.id}: {type(e).__name__}: {e}", exc_info=True)
                        summary["errors"].append(f"flatten_escalate_closing:{pos.id}")
                continue

            if pos.status != "open":
                continue

            if pos.id in self._review_tasks:
                continue  # already reviewing

            jev_escalated = await self._jev_sentinel_screen(pos, pnl_pct, now_et, cfg, snap, underlyings_ctx)
            triggers = self._review_triggers(pos, mark, now, jev_escalated, snap, underlyings_ctx, cfg)
            if not triggers:
                continue

            if not self._decider_available():
                await self._handle_review_unavailable(pos, now, bot, cfg, summary)
                continue

            ctx = self._build_ctx(now_et, cfg, snap, underlyings_ctx, market_regime, risk, open_positions, dpof_equity, open_premium)
            self._maybe_add_lessons_ctx(ctx, pos.underlying, cfg)
            position_dict = self._build_review_position_dict(pos, mark, pnl_pct, now, triggers)
            self._review_task_meta[pos.id] = {"launched_at": now, "underlying": pos.underlying, "qty_open_at_launch": pos.qty_open}

            # Learning loop F2 (§5.2/R3): a review debates only on an
            # alignment-flip trigger, never within debate_review_min_minutes_
            # to_flatten of the 15:50 flatten, and never alongside a
            # jev_escalation. When the gate is false the coroutine below is
            # exactly today's (byte-identical, no wrapper).
            should_debate = False
            try:
                should_debate = should_debate_review(
                    triggers=triggers, minutes_to_flatten=self._minutes_to_flatten(now_et, cfg),
                    min_minutes=cfg.debate_review_min_minutes_to_flatten, enabled=cfg.debate_enabled,
                    on_reviews=cfg.debate_on_reviews,
                )
            except Exception as e:
                logger.debug(f"DPOF engine: should_debate_review failed for {pos.id}: {type(e).__name__}: {e}")

            if should_debate and self._advisory_debate is not None:
                coro = self._run_review_debate_then_decide(pos, ctx, position_dict, underlyings_ctx, cfg)
            else:
                coro = self._decider.review_position(ctx, position_dict)
            self._review_tasks[pos.id] = asyncio.ensure_future(self._run_decision_task(coro))

    async def _apply_review_outcome(
        self, pos_id: str, outcome: dict, now: float, now_et: datetime, cfg: DPOFConfig, bot, summary: dict, exits_log: list[dict],
    ) -> None:
        meta = self._review_task_meta.pop(pos_id, {})
        pos = self._ledger.get(pos_id)
        decision_log: dict = {
            "ts": now, "mode": cfg.mode.value, "kind": "position_review", "position_id": pos_id,
            "underlying": meta.get("underlying"), "launched_at": meta.get("launched_at"),
            "finished_at": outcome.get("finished"), "reasons": [],
        }
        # Learning loop F2 (§5.3): only present when this review's debate gate
        # fired (_run_review_debate_then_decide set it in the task meta) --
        # every early return below already appends decision_log unchanged.
        debate_summary = meta.get("debate_summary")
        if debate_summary is not None:
            decision_log["debate"] = debate_summary
        if pos is None:
            decision_log["reasons"].append("position_gone")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        if not outcome["ok"]:
            decision_log["reasons"].append(f"decider_error:{outcome['error']}")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            if outcome.get("unavailable"):
                await self._handle_review_unavailable(pos, now, bot, cfg, summary)
            return

        await self._handle_model_recovered_if_needed(bot)
        decision = outcome["result"]  # decider.ReviewDecision
        decision_log["provider"] = getattr(decision, "provider", "")
        decision_log["model"] = getattr(decision, "model", "")
        decision_log["action"] = getattr(decision, "action", None)
        # Learning loop (Task ORCH-7Q4K-DPOF builder B2/reflection.py): richer
        # review trail -- reflection.build_trade_record's _cap_reviews reads
        # these two fields directly off each decisions.jsonl position_review
        # row (additive; nothing existing reads/relies on their absence).
        decision_log["reason"] = getattr(decision, "reason", "")
        decision_log["thesis_status"] = getattr(decision, "thesis_status", "")

        finished = outcome.get("finished", now)
        if (now - finished) > cfg.decision_max_age_seconds:
            decision_log["reasons"].append("stale_decision")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        # ORDER-SAFETY: only apply against the SAME open position, unchanged
        # since launch (see module docstring's rule).
        if pos.status != "open" or pos.qty_open != meta.get("qty_open_at_launch"):
            decision_log["reasons"].append("position_changed_or_not_open")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        action = getattr(decision, "action", "hold")
        urgency = getattr(decision, "urgency", "normal")
        next_review_minutes = getattr(decision, "next_review_minutes", None) or cfg.review_interval_minutes
        next_review_minutes = max(cfg.review_min_minutes, min(cfg.review_max_minutes, next_review_minutes))

        mark = self._last_marks.get(pos.id)
        last_review_summary = {
            "action": action, "reason": getattr(decision, "reason", ""), "thesis_status": getattr(decision, "thesis_status", ""),
            "urgency": urgency, "provider": getattr(decision, "provider", ""), "model": getattr(decision, "model", ""),
        }
        base_updates = dict(
            last_review=last_review_summary, last_review_at=now, last_review_mark=mark,
            review_count=pos.review_count + 1, review_failures=0,
            next_review_at=now + next_review_minutes * 60,
        )

        if action not in ("hold", "trim", "exit"):
            decision_log["reasons"].append(f"unknown_action:{action}")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        updated = replace(pos, **base_updates)
        try:
            await asyncio.shield(self._ledger.upsert(updated))
        except Exception as e:
            logger.error(f"DPOF engine: failed to persist review for {pos.id}: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append(f"persist_review:{pos.id}")

        if action == "hold":
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        urgent = urgency == "urgent"
        if action == "exit":
            qty = updated.qty_open
            reason = "decider_exit"
        else:
            trim_fraction = max(0.0, min(1.0, getattr(decision, "trim_fraction", 0.0) or 0.0))
            qty = math.ceil(updated.qty_open * trim_fraction) if updated.qty_open > 0 else 0
            qty = max(0, qty)
            if qty >= updated.qty_open:
                qty = updated.qty_open
                reason = "decider_exit"
            elif qty > 0:
                qty = max(1, qty)
                reason = "decider_trim"
            else:
                reason = "decider_trim"

        if qty <= 0:
            decision_log["reasons"].append("trim_zero_qty")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        try:
            ok = await self._orders.close_position(updated, qty, reason, urgent, now)
            exits_log.append({"position_id": updated.id, "action": action, "qty": qty, "reason": reason, "submitted": ok})
        except Exception as e:
            logger.error(f"DPOF engine: close_position (review) failed for {updated.id}: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append(f"close_position:{updated.id}")

        await asyncio.shield(self._ledger.append_decision(decision_log))

    # ------------------------------------------------------------------
    # Entry deliberation (step 11)
    # ------------------------------------------------------------------

    def _cooldown_allows(self, underlying: str, signal: Signal, now: float, cfg: DPOFConfig) -> bool:
        key = f"{underlying}:{signal.direction.value}"
        rec = self._entry_cooldowns.get(key)
        if rec is None:
            return True
        if now >= rec.get("next_allowed_at", 0.0):
            return True
        recheck_at = rec.get("recheck_at")
        if recheck_at is not None and now >= recheck_at:
            return True
        last_composite = rec.get("last_composite")
        if (
            signal.composite is not None and last_composite is not None
            and abs(abs(signal.composite) - abs(last_composite)) >= cfg.entry_redeliberate_delta
        ):
            return True
        # Task 3 (light entry re-deliberation nudge): a new |surprise_z| >=
        # 1.5 released AFTER this cooldown record was set bypasses it for the
        # next cycle -- the decider still decides (enter/pass); this only
        # gets it deliberating sooner instead of waiting out the full cooldown.
        set_at = rec.get("set_at")
        if set_at is not None:
            intel = self._last_intel_ctx or {}
            for s in (intel.get("events") or {}).get("recent_surprises") or []:
                if not isinstance(s, dict):
                    continue
                z = s.get("surprise_z")
                if not isinstance(z, (int, float)) or isinstance(z, bool) or abs(z) < 1.5:
                    continue
                released = _parse_iso_epoch(s.get("release_time_et"))
                if released is not None and released > set_at:
                    return True
        return False

    # ------------------------------------------------------------------
    # Entry-outcome classification (Task ORCH-7Q4K-DPOF, 2026-09-25) — pure,
    # OBSERVABILITY ONLY. These never gate/change anything: they read state
    # _handle_entry_pipeline/_launch_entry_deliberation/_apply_entry_outcome
    # ALREADY computed (or, for _signal_block_outcome/_gate_outcome, the
    # SAME real Signal/GateResult those functions already branch on) and
    # turn it into a taxonomy string for signals.jsonl/`/dpof why`/the EOD
    # digest. See docs/design/clio_dpof.md's Observability section for the
    # full taxonomy and where each value comes from.
    # ------------------------------------------------------------------

    @staticmethod
    def _note_entry_outcome(summary: dict, underlying: str, outcome: str, detail: Optional[list] = None) -> None:
        """Records THIS cycle's fresh-nomination-pipeline outcome for
        `underlying` into summary['entry_outcomes'] — called at every real
        return point of _handle_entry_pipeline/_launch_entry_deliberation,
        additively, without touching any of those functions' actual
        conditionals/order."""
        summary.setdefault("entry_outcomes", {})[underlying] = {"outcome": outcome, "detail": list(detail or [])}

    @staticmethod
    def _gate_outcome(reasons: list[str]) -> str:
        """risk_module.check_entry_allowed()'s `reasons` -> a taxonomy
        outcome. Both `hard_halt:<reason>` and `halt_file` mean the SAME
        thing to an operator reading `/dpof why` (a manual halt is active),
        so both map to "halted"; `outside_entry_window` is the only other
        reason that function ever produces (risk.py's own docstring)."""
        if any(r == "halt_file" or r.startswith("hard_halt") for r in reasons):
            return "halted"
        if "outside_entry_window" in reasons:
            return "outside_entry_window"
        return "gate_blocked"  # defensive fallback; risk.py names every real reason above

    @staticmethod
    def _signal_block_outcome(signal: Signal) -> str:
        """Classifies WHY `_handle_entry_pipeline`'s own
        `if signal is None or signal.direction == Direction.NEUTRAL or
        signal.vetoes: return` fired, from the SAME Signal fields that
        condition reads — never a second/different computation of the
        signal itself (signals.py's evaluate_signal already ran; this only
        reads its output). Priority mirrors evaluate_signal()'s own order:
        data_unhealthy is appended before composite/strength/regime are even
        computed, so it's checked first; a NEUTRAL direction (composite
        unavailable / below threshold / regime-downgraded) is why `return`
        fired whenever direction is NEUTRAL, regardless of any veto that
        happens to also be set; only when direction is NOT neutral can a
        veto (dp_veto / price_not_confirmed) be the actual reason."""
        if "data_unhealthy" in signal.vetoes:
            return "data_unhealthy"
        if signal.direction == Direction.NEUTRAL:
            if "composite_unavailable" in signal.reasons:
                return "composite_unavailable"
            if "regime_requires_strong" in signal.reasons:
                return "regime_requires_strong"
            if "composite_below_normal_threshold" in signal.reasons:
                return "below_threshold"
            return "no_signal"  # defensive fallback; evaluate_signal always sets one of the above when NEUTRAL
        if "dp_veto" in signal.vetoes:
            return "dp_veto"
        if "price_not_confirmed" in signal.vetoes:
            return "price_not_confirmed"
        if signal.vetoes:
            return "vetoed"  # defensive fallback for a future veto this classifier doesn't name yet
        return "blocked"  # unreachable given _handle_entry_pipeline's own condition; defensive only

    @staticmethod
    def _classify_harvest_outcome(harvest: dict) -> str:
        """Classifies a HARVESTED entry deliberation (a decision_log dict
        _apply_entry_outcome already built and appended to decisions.jsonl
        unchanged) into the same taxonomy — reads decision_log['reasons']/
        ['outcome']/['action'] in the EXACT order _apply_entry_outcome's own
        branches append them, so the first (and only ever one, since every
        branch there returns immediately after appending) reason present is
        the real one."""
        outcome_field = harvest.get("outcome")
        if isinstance(outcome_field, str) and (outcome_field.startswith("entry_submitted") or outcome_field == "would_enter"):
            return "entered"
        for r in harvest.get("reasons") or []:
            if not isinstance(r, str):
                continue
            if r == "position_gone":
                return "position_gone"
            if r.startswith("decider_error"):
                return "decider_unavailable" if harvest.get("unavailable") else "decider_error"
            if r == "stale_decision":
                return "stale_decision"
            if r == "pass":
                return "decider_pass"
            if r.startswith("unknown_action"):
                return "decider_invalid_action"
            if r == "invalid_candidate_key":
                return "decider_invalid_candidate"
            if r == "symbol_position_exists":
                return "position_open"
            if r == "halt_file" or r.startswith("hard_halt"):
                return "halted"
            if r == "outside_entry_window":
                return "outside_entry_window"
            if r == "signal_or_snapshot_unavailable":
                return "signal_unavailable"
            if r == "requote_failed":
                return "requote_failed"
            if r == "requote_unusable":
                return "requote_unusable"
            if r.startswith("slippage:"):
                return "slippage_exceeded"
            if r == "no_size":
                return "no_size"
            if r == "jev_veto":
                return "jev_veto"
            if r == "open_position_failed":
                return "open_position_failed"
            if r.startswith("open_position_rejected"):
                return "open_position_rejected"
        if harvest.get("action") == "enter":
            return "decider_enter"  # decider said ENTER but no specific reason matched above (defensive)
        if harvest.get("action") == "pass":
            return "decider_pass"
        return "decider_result_unknown"  # defensive fallback; every real _apply_entry_outcome exit is named above

    def _fallback_entry_outcome(self, sym: str, now_et: datetime, cfg: DPOFConfig, risk: RiskState, halt_present: bool) -> str:
        """For an underlying that was neither this cycle's fresh nomination
        (signal.symbol) nor a harvested deliberation this cycle — the real
        pipeline never evaluated it at all this cycle, so this only reports
        real, symbol-agnostic-or-per-symbol FACTS (ledger open positions /
        in-flight tasks are per-symbol; check_entry_allowed's gates are
        symbol-agnostic by construction — see risk.py's own docstring: its
        `symbol` param is accepted for call-site symmetry only and never
        changes the result), never a synthesized signal/composite verdict
        for a symbol the entry pipeline didn't touch this cycle."""
        if any(p.underlying == sym for p in self._ledger.open_positions()):
            return "position_open"
        if sym in self._entry_tasks:
            return "deliberation_in_flight"
        gate = risk_module.check_entry_allowed(risk, now_et, sym, cfg, halt_present)
        if not gate.allowed:
            return self._gate_outcome(gate.reasons)
        return "not_selected_instrument"

    def _build_entry_candidate(
        self, key: str, choice: ContractChoice, quote_by_symbol: dict, today: date,
        capital_available: Optional[float], entry_candidate_cls: type,
    ):
        """`capital_available` = dpof_equity - open_premium_committed (incl.
        pending entries), the SAME solvency slice `_apply_entry_outcome` will
        re-check freshly at actual-sizing time. `max_affordable_qty` here is
        informational context for the decision model, computed at LAUNCH
        time — operator directive (no-hard-caps amendment): this is NOT a
        risk-pct/premium-pct cap, just "how many of this contract could DPOF
        possibly afford right now"."""
        leg_dicts = [
            {"symbol": l.symbol, "side": l.side, "ratio_qty": l.ratio_qty, "position_intent": l.position_intent}
            for l in choice.legs
        ]
        long_leg = next((l for l in choice.legs if l.position_intent == "buy_to_open"), choice.legs[0] if choice.legs else None)
        long_quote = quote_by_symbol.get(long_leg.symbol) if long_leg is not None else None
        unit_premium = choice.est_debit * 100.0 if choice.est_debit else None
        spread_pct = None
        if choice.est_debit and choice.est_debit > 0 and choice.est_ask is not None and choice.est_bid is not None:
            spread_pct = (choice.est_ask - choice.est_bid) / choice.est_debit
        max_qty = max_affordable_qty(capital_available, choice.est_debit)
        direction_val = choice.direction.value if hasattr(choice.direction, "value") else str(choice.direction)
        return entry_candidate_cls(
            key=key, underlying=choice.underlying, direction=direction_val, structure=choice.structure,
            legs=leg_dicts, expiry=choice.expiration, sessions_to_expiry=_sessions_to_expiry(choice.expiration, today),
            strike_desc=_strike_desc(choice.legs), bid=choice.est_bid, ask=choice.est_ask, mid=choice.est_debit,
            spread_pct=spread_pct, delta=choice.long_delta,
            theta=(long_quote.theta if long_quote is not None else None),
            iv=(long_quote.iv if long_quote is not None else None),
            open_interest=(long_quote.open_interest if long_quote is not None else None),
            unit_premium=unit_premium, max_affordable_qty=max_qty,
        )

    async def _launch_entry_deliberation(
        self, signal: Signal, now: float, now_et: datetime, cfg: DPOFConfig, snap: Optional[MarketSnapshot],
        underlyings_ctx: dict, market_regime: Optional[str], risk: RiskState, dpof_equity: Optional[float],
        options_level: Optional[int], summary: dict,
    ) -> None:
        underlying = signal.symbol
        today = now_et.date()

        try:
            from agent.dpof.decider import EntryCandidate
        except Exception as e:
            logger.debug(f"DPOF engine: EntryCandidate unavailable ({type(e).__name__}: {e}); cannot build candidate menu yet")
            self._note_entry_outcome(summary, underlying, "candidate_schema_unavailable")
            return

        # Verticals gated at RUNTIME by the broker-reported account options
        # level (operator directive) -- an account below level 3 gets
        # singles only, no error, regardless of cfg.allow_verticals.
        candidate_cfg = cfg
        if cfg.allow_verticals and not (options_level is not None and options_level >= 3):
            candidate_cfg = replace(cfg, allow_verticals=False)

        try:
            lo, hi = self._expiry_fetch_window_fn(today, candidate_cfg)
        except Exception as e:
            logger.error(f"DPOF engine: expiry_fetch_window_fn failed: {type(e).__name__}: {e}")
            self._note_entry_outcome(summary, underlying, "expiry_window_error")
            return
        right = option_right(signal.direction)
        chain_fetch_error = False
        try:
            chain = await self._broker.get_option_chain(underlying, lo, hi, right)
        except Exception as e:
            logger.error(f"DPOF engine: get_option_chain failed for {underlying}: {type(e).__name__}: {e}")
            chain = None
            chain_fetch_error = True
        if not chain:
            logger.debug(f"DPOF engine: no chain for {underlying} {right} [{lo}..{hi}]; entry launch skipped this cycle")
            self._note_entry_outcome(summary, underlying, "chain_fetch_error" if chain_fetch_error else "no_chain")
            return

        u = snap.underlyings.get(underlying) if snap is not None else None
        spot = u.last_price if u is not None else None
        candidate_build_error = False
        try:
            choices = self._candidate_fn(chain, underlying=underlying, direction=signal.direction, spot=spot, today=today, cfg=candidate_cfg, now_et=now_et)
        except Exception as e:
            logger.error(f"DPOF engine: candidate_fn failed for {underlying}: {type(e).__name__}: {e}")
            choices = []
            candidate_build_error = True
        if not choices:
            logger.debug(f"DPOF engine: candidate_fn returned no candidates for {underlying}; entry launch skipped this cycle")
            self._note_entry_outcome(summary, underlying, "candidate_build_error" if candidate_build_error else "no_candidates")
            return

        quote_by_symbol = {q.symbol: q for q in chain}
        open_positions = self._ledger.open_positions()
        open_premium = sum(self._position_premium(p) for p in open_positions)
        capital_available = None if dpof_equity is None else (dpof_equity - open_premium)
        candidates = [
            self._build_entry_candidate(f"C{i}", choice, quote_by_symbol, today, capital_available, EntryCandidate)
            for i, choice in enumerate(choices, start=1)
        ]

        ctx = self._build_ctx(now_et, cfg, snap, underlyings_ctx, market_regime, risk, open_positions, dpof_equity, open_premium)
        self._maybe_add_lessons_ctx(ctx, underlying, cfg)
        self._entry_task_meta[underlying] = {
            "launched_at": now, "direction": signal.direction.value, "signal": signal,
            "candidates": {c.key: c for c in candidates},
        }

        # Learning loop F2 (§5.1/R3): only a genuinely borderline "normal"
        # nomination debates -- a "strong" signal, or one with no composite,
        # never does. When the gate is false the coroutine below is exactly
        # today's (byte-identical, no wrapper).
        should_debate = False
        try:
            should_debate = should_debate_entry(
                strength=signal.strength, composite=signal.composite,
                threshold_normal=cfg.threshold_normal, margin=cfg.debate_margin, enabled=cfg.debate_enabled,
            )
        except Exception as e:
            logger.debug(f"DPOF engine: should_debate_entry failed for {underlying}: {type(e).__name__}: {e}")

        if should_debate and self._advisory_debate is not None:
            coro = self._run_entry_debate_then_decide(underlying, ctx, candidates, signal, underlyings_ctx, cfg)
        else:
            coro = self._decider.decide_entry(ctx, candidates)
        self._entry_tasks[underlying] = asyncio.ensure_future(self._run_decision_task(coro))
        self._note_entry_outcome(summary, underlying, "deliberating")

    @staticmethod
    def _contract_choice_from_candidate(candidate, bid: Optional[float], ask: Optional[float], mid: Optional[float]) -> ContractChoice:
        legs = [
            OrderLeg(symbol=l["symbol"], side=l["side"], ratio_qty=l.get("ratio_qty", 1), position_intent=l.get("position_intent", ""))
            for l in candidate.legs
        ]
        direction = candidate.direction if isinstance(candidate.direction, Direction) else Direction(candidate.direction)
        return ContractChoice(
            structure=candidate.structure, legs=legs, underlying=candidate.underlying, direction=direction,
            expiration=candidate.expiry, dte=candidate.sessions_to_expiry, est_debit=mid or 0.0,
            est_bid=bid or 0.0, est_ask=ask or 0.0, long_delta=candidate.delta, reasons=[],
        )

    def _record_entry_decision(
        self, verdict: JevVerdict, signal: Signal, snap: Optional[MarketSnapshot], now_et: datetime,
        cfg: DPOFConfig, sizing_ctx: dict, underlying: str, candidate_key: Optional[str],
    ) -> Optional[str]:
        """decision-record producer -- SYNC, never raises, never
        blocks (record_call's own contract). Returns the call_id (or None
        when decisions are disabled/unavailable) so the caller can persist
        it onto the position's judgment.entry_decision.decision_call_id."""
        if not self._decisions_enabled():
            return None
        try:
            if verdict.request is not None:
                state, questions = verdict.request["state"], verdict.request["questions"]
            elif snap is not None:
                state, questions = prepare_entry_request(signal, snap, now_et, cfg, sizing_ctx)
            else:
                return None
            served = None
            if verdict.called:
                served = (
                    verdict.answers or None, verdict.model or "jev-unknown", verdict.endpoint,
                    verdict.latency_ms, verdict.error,
                )
            producer_ref = {
                "cycle_ts": now_et.isoformat(), "candidate_key": candidate_key,
                "symbol": underlying, "jev_mode": verdict.mode,
            }
            return self._decisions.record_call("dpof_entry_veto", producer_ref, state, questions, served=served)
        except Exception as e:
            logger.debug(f"DPOF engine: decisions record_call (entry) failed for {underlying}: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _set_size_telemetry(
        decision_log: dict, decision: object, qty: int, unit_premium: Optional[float], dpof_equity: Optional[float],
        capital_available: Optional[float], conviction: float, cfg: DPOFConfig, verdict: object = None,
    ) -> None:
        """S212 SZ-1: record the sizing picture on the decision row. Fire-and-forget: it only WRITES a new
        top-level `sizing` key (never inside `jev`, which the signal log copies whole) and can never change
        the qty, the veto or raise into the entry path."""
        try:
            decision_log["sizing"] = size_telemetry(
                qty_requested=getattr(decision, "qty", None), qty=qty, unit_premium=unit_premium,
                dpof_equity=dpof_equity, capital_available=capital_available, conviction=conviction,
                anchor_shares=(
                    (cfg.size_anchor_share_c50, cfg.size_anchor_share_c80) if cfg.size_anchor_enabled else None
                ),
                jev_answers=getattr(verdict, "answers", None),
                jev_size_veto_threshold=getattr(cfg, "jev_size_veto_threshold", 0.70) if verdict is not None else None,
            )
        except Exception as e:
            logger.debug(f"DPOF engine: size telemetry failed: {type(e).__name__}: {e}")

    async def _apply_entry_outcome(
        self, underlying: str, outcome: dict, now: float, now_et: datetime, cfg: DPOFConfig, bot, summary: dict, dpof_equity: Optional[float],
    ) -> None:
        meta = self._entry_task_meta.pop(underlying, {})
        decision_log: dict = {
            "ts": now, "mode": cfg.mode.value, "kind": "entry_deliberation", "underlying": underlying,
            "direction": meta.get("direction"), "launched_at": meta.get("launched_at"),
            "finished_at": outcome.get("finished"), "reasons": [],
        }
        # Learning loop F2 (§5.3): only present when this deliberation's
        # debate gate fired (_run_entry_debate_then_decide set it in the task
        # meta) -- every early return below already appends decision_log
        # unchanged.
        debate_summary = meta.get("debate_summary")
        if debate_summary is not None:
            decision_log["debate"] = debate_summary
        # Observability (Task ORCH-7Q4K-DPOF): `decision_log` is mutated (never
        # reassigned) for the rest of this function, so storing the reference
        # here lets _log_cycle_signals read its FINAL state after this whole
        # await completes -- one line, no change to any branch below.
        summary.setdefault("entry_harvests", {})[underlying] = decision_log
        if not outcome["ok"]:
            decision_log["reasons"].append(f"decider_error:{outcome['error']}")
            decision_log["unavailable"] = bool(outcome.get("unavailable"))
            await asyncio.shield(self._ledger.append_decision(decision_log))
            if outcome.get("unavailable"):
                await self._handle_model_down(bot, cfg)
            return

        await self._handle_model_recovered_if_needed(bot)
        decision = outcome["result"]  # decider.EntryDecision
        decision_log["provider"] = getattr(decision, "provider", "")
        decision_log["model"] = getattr(decision, "model", "")
        decision_log["action"] = getattr(decision, "action", None)

        finished = outcome.get("finished", now)
        if (now - finished) > cfg.decision_max_age_seconds:
            decision_log["reasons"].append("stale_decision")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        action = getattr(decision, "action", "pass")
        signal: Optional[Signal] = meta.get("signal")

        if action == "pass":
            recheck_minutes = getattr(decision, "recheck_minutes", None) or max(1, cfg.entry_deliberation_cooldown_seconds // 60)
            key = f"{underlying}:{meta.get('direction')}"
            self._entry_cooldowns[key] = {
                "next_allowed_at": now + cfg.entry_deliberation_cooldown_seconds,
                "recheck_at": now + max(1, recheck_minutes) * 60,
                "last_composite": signal.composite if signal is not None else None,
                "set_at": now,  # Task 3: baseline for the new-surprise cooldown bypass in _cooldown_allows
            }
            decision_log["reasons"].append("pass")
            thesis_line = getattr(decision, "thesis_line", "")  # decider-sanitized one-liner; absent on other decision types
            if isinstance(thesis_line, str) and thesis_line:
                decision_log["thesis"] = thesis_line
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        if action != "enter":
            decision_log["reasons"].append(f"unknown_action:{action}")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        candidates = meta.get("candidates", {})
        candidate_key = getattr(decision, "candidate_key", None)
        candidate = candidates.get(candidate_key) if candidate_key else None
        if candidate is None:
            decision_log["reasons"].append("invalid_candidate_key")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        # Applicability: no position materialized for this underlying since
        # launch, and every hard cap still allows an entry.
        existing = [p for p in self._ledger.open_positions() if p.underlying == underlying]
        if existing:
            decision_log["reasons"].append("symbol_position_exists")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return
        risk = self._ledger.risk
        gate = risk_module.check_entry_allowed(risk, now_et, underlying, cfg, self._ledger.halt_file_present())
        if not gate.allowed:
            decision_log["reasons"].extend(gate.reasons)
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        snap = self._last_snapshot
        if signal is None or snap is None:
            decision_log["reasons"].append("signal_or_snapshot_unavailable")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        # Re-quote the chosen candidate; discard on excess slippage vs the
        # decided mid. Fresh quotes also feed the solvency clamp below.
        symbols = [l["symbol"] for l in candidate.legs]
        try:
            quotes = await self._broker.get_option_quotes(symbols) or {}
        except Exception as e:
            logger.error(f"DPOF engine: re-quote failed for {underlying}: {type(e).__name__}: {e}")
            decision_log["reasons"].append("requote_failed")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return
        net_bid, net_ask = _net_bid_ask_from_legs(candidate.legs, quotes)
        if net_bid is None or net_ask is None or net_ask < net_bid:
            decision_log["reasons"].append("requote_unusable")
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return
        fresh_mid = (net_bid + net_ask) / 2.0
        decided_mid = candidate.mid
        if decided_mid is not None and decided_mid > 0:
            slip = abs(fresh_mid - decided_mid) / decided_mid
            if slip > cfg.entry_max_slippage_pct:
                decision_log["reasons"].append(f"slippage:{slip:.4f}")
                await asyncio.shield(self._ledger.append_decision(decision_log))
                return

        # Operator directive (no-hard-caps amendment): sizing is the decision
        # model's `qty` directly, clamped ONLY by a fresh solvency check
        # (equity/open-premium may have moved during the model's latency).
        conviction = _clamp01(getattr(decision, "conviction", 0.0))
        decision_log["conviction"] = conviction
        open_positions_now = self._ledger.open_positions()
        open_premium_now = sum(self._position_premium(p) for p in open_positions_now)
        capital_available_now = None if dpof_equity is None else (dpof_equity - open_premium_now)
        qty, size_reasons = clamp_to_solvency(getattr(decision, "qty", None), capital_available_now, fresh_mid)
        self._set_size_telemetry(
            decision_log, decision, qty, (fresh_mid * 100.0) if fresh_mid else None, dpof_equity,
            capital_available_now, conviction, cfg,
        )
        if qty <= 0:
            decision_log["reasons"].append("no_size")
            decision_log["size_reasons"] = size_reasons
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return
        if size_reasons:
            decision_log["size_reasons"] = size_reasons  # e.g. clamped_by_solvency:N->M (still entered at the clamp)

        # Jev entry veto (D2: gains a `sizing_ctx` size question once `qty`
        # is known) — the nominated Signal from launch time.
        trade_premium = qty * fresh_mid * 100.0
        sizing_ctx = {
            "premium_pct_of_capital": (trade_premium / dpof_equity) if dpof_equity else None,
            "open_exposure_pct": ((open_premium_now + trade_premium) / dpof_equity) if dpof_equity else None,
            "positions_open": len(open_positions_now),
            "entries_today": risk.entries_today,
            "minutes_to_flatten": self._minutes_to_flatten(now_et, cfg),
            "realized_today_pct": (risk.realized_pnl_today / dpof_equity) if dpof_equity else None,
            "conviction": conviction,
        }
        verdict = await jev_entry_check(signal, snap, now_et, cfg, self._jev, sizing_ctx=sizing_ctx)
        self._set_size_telemetry(
            decision_log, decision, qty, (fresh_mid * 100.0) if fresh_mid else None, dpof_equity,
            capital_available_now, conviction, cfg, verdict=verdict,
        )
        if verdict.called:
            self._last_jev_verdict = verdict
            decision_log["jev"] = {
                "allow": verdict.allow, "veto_reasons": list(verdict.veto_reasons),
                "endpoint": verdict.endpoint, "error": verdict.error,
            }
            try:
                await asyncio.shield(self._ledger.append_jev(verdict.to_dict()))
            except Exception as e:
                logger.error(f"DPOF engine: append_jev failed: {type(e).__name__}: {e}")

        # decision-record producer: recorded regardless of veto outcome
        # and regardless of whether Jev was actually called (OFF/unkeyed) --
        # fire-and-forget, after the verdict is computed, so latency and the
        # veto itself are untouched.
        decision_call_id = self._record_entry_decision(
            verdict, signal, snap, now_et, cfg, sizing_ctx, underlying, candidate_key,
        )

        if not verdict.allow:
            decision_log["reasons"].append("jev_veto")
            decision_log["jev_veto_reasons"] = verdict.veto_reasons
            await asyncio.shield(self._ledger.append_decision(decision_log))
            return

        choice = self._contract_choice_from_candidate(candidate, net_bid, net_ask, fresh_mid)
        judgment = {
            "thesis": getattr(decision, "thesis", ""),
            "invalidation": getattr(decision, "invalidation", ""),
            "entry_decision": {
                "provider": getattr(decision, "provider", ""), "model": getattr(decision, "model", ""),
                "conviction": conviction, "qty": qty,
                "latency_ms": getattr(decision, "latency_ms", None), "candidate_key": candidate_key,
                "decision_call_id": decision_call_id,
            },
            "next_review_at": now + cfg.review_interval_minutes * 60,
        }
        pos = await self._orders.open_position(choice, qty, signal, verdict, now, judgment=judgment)
        if pos is not None:
            # Learning loop data capture (§4.1): the underlying's last_price
            # from the snapshot at entry-apply time. Least-invasive placement
            # (Jev-advised, confidence 0.98): a shielded upsert right here,
            # never a change to orders.py's own PositionRecord construction --
            # `pos` is reassigned so every branch below (alerts, decision_log,
            # summary) sees the enriched record.
            u = snap.underlyings.get(underlying) if snap is not None else None
            entry_u = u.last_price if u is not None else None
            if entry_u is not None:
                try:
                    pos = replace(pos, entry_underlying_price=entry_u)
                    await asyncio.shield(self._ledger.upsert(pos))
                except Exception as e:
                    logger.error(
                        f"DPOF engine: failed to record entry_underlying_price for {pos.id}: "
                        f"{type(e).__name__}: {e}",
                    )
        if pos is None:
            decision_log["reasons"].append("open_position_failed")
        elif pos.status in ("pending", "open"):
            risk = risk_module.record_entry(risk, underlying, now)
            try:
                await asyncio.shield(self._ledger.save_risk(risk))
            except Exception as e:
                logger.error(f"DPOF engine: save_risk (post-entry) failed: {type(e).__name__}: {e}")
            decision_log["outcome"] = f"entry_submitted:{pos.id}"
            summary["entries"].append({"position_id": pos.id, "qty": qty})
            await self._safe_alert(bot, fmt_entry(pos, signal, verdict, cfg.mode.value))
        elif pos.exit_reason == "log_only":
            risk = risk_module.record_entry(risk, underlying, now)
            try:
                await asyncio.shield(self._ledger.save_risk(risk))
            except Exception as e:
                logger.error(f"DPOF engine: save_risk (post-entry, log_only) failed: {type(e).__name__}: {e}")
            decision_log["outcome"] = "would_enter"
            summary["entries"].append({"position_id": pos.id, "qty": qty, "log_only": True})
            await self._safe_alert(
                bot, fmt_entry(pos, signal, verdict, cfg.mode.value),
                dedupe_key=f"dpof_would_enter_{underlying}_{signal.direction.value}", dedupe_seconds=1800,
            )
        else:
            decision_log["reasons"].append(f"open_position_rejected:{pos.exit_reason}")

        await asyncio.shield(self._ledger.append_decision(decision_log))

    async def _handle_entry_pipeline(
        self, now: float, now_et: datetime, cfg: DPOFConfig, bot, snap: Optional[MarketSnapshot], underlyings_ctx: dict,
        market_regime: Optional[str], risk: RiskState, dpof_equity: Optional[float], options_level: Optional[int], summary: dict,
    ) -> None:
        for underlying, task in list(self._entry_tasks.items()):
            if not task.done():
                continue
            del self._entry_tasks[underlying]
            outcome = task.result()
            await self._apply_entry_outcome(underlying, outcome, now, now_et, cfg, bot, summary, dpof_equity)

        signal = self._last_signal
        if signal is None or signal.direction == Direction.NEUTRAL or signal.vetoes:
            if signal is not None:
                self._note_entry_outcome(summary, signal.symbol, self._signal_block_outcome(signal))
            return
        underlying = signal.symbol
        summary["signal"] = signal.to_dict()

        if underlying in self._entry_tasks:
            self._note_entry_outcome(summary, underlying, "deliberation_in_flight")
            return  # already deliberating this underlying (concurrency control, not a cap)
        existing = [p for p in self._ledger.open_positions() if p.underlying == underlying]
        if existing:
            self._note_entry_outcome(summary, underlying, "position_open")
            return
        gate = risk_module.check_entry_allowed(risk, now_et, underlying, cfg, self._ledger.halt_file_present())
        if not gate.allowed:
            self._note_entry_outcome(summary, underlying, self._gate_outcome(gate.reasons), detail=gate.reasons)
            return
        if not self._cooldown_allows(underlying, signal, now, cfg):
            self._note_entry_outcome(summary, underlying, "cooldown")
            return

        # Solvency pre-check (operator directive: the ONLY code-side size
        # bound left) -- no point deliberating a trade DPOF can't afford even
        # one contract of.
        open_premium = sum(self._position_premium(p) for p in self._ledger.open_positions())
        capital_available = None if dpof_equity is None else (dpof_equity - open_premium)
        if capital_available is None or capital_available <= 0:
            self._note_entry_outcome(summary, underlying, "insufficient_capital")
            return

        if not self._decider_available():
            await self._handle_model_down(bot, cfg)
            self._note_entry_outcome(summary, underlying, "decider_unavailable")
            return

        await self._launch_entry_deliberation(
            signal, now, now_et, cfg, snap, underlyings_ctx, market_regime, risk, dpof_equity, options_level, summary,
        )

    # ------------------------------------------------------------------
    # Observability: signals.jsonl / day stats / rate-limited INFO logging
    # (Task ORCH-7Q4K-DPOF, 2026-09-25). OBSERVABILITY ONLY -- see the
    # _note_entry_outcome/_classify_harvest_outcome/_fallback_entry_outcome
    # block above and docs/design/clio_dpof.md's Observability section.
    # ------------------------------------------------------------------

    async def _log_cycle_signals(
        self, now: float, now_et: datetime, cfg: DPOFConfig, snap: Optional[MarketSnapshot],
        signal: Optional[Signal], underlyings_ctx: dict, summary: dict, risk: RiskState,
    ) -> None:
        """One signals.jsonl record per underlying in cfg.underlyings, built
        ENTIRELY from state _handle_entry_pipeline/_launch_entry_deliberation/
        _apply_entry_outcome already computed this cycle
        (summary['entry_outcomes'] for a fresh nomination attempt this cycle,
        summary['entry_harvests'] for a deliberation harvested this cycle) --
        never a second/parallel evaluation of the signal or the gates. An
        underlying touched by neither this cycle falls back to
        _fallback_entry_outcome's real, symbol-scoped ledger/gate facts.

        No snapshot this cycle (snap is None, e.g. feed.snapshot() failed) or
        no SignalLog collaborator yet (mode was OFF and _ensure_collaborators
        never ran) -> no-op; there's nothing real to attribute a record to.
        """
        if self._signal_log is None or snap is None:
            return

        entry_outcomes = summary.get("entry_outcomes") or {}
        entry_harvests = summary.get("entry_harvests") or {}
        halt_present = self._ledger.halt_file_present() if self._ledger is not None else False
        regime_label = self._regime_ctx(snap, cfg).get("label")
        # Task K2 (agent/kronos): this cycle's kronos ctx, same source
        # _build_ctx reads (self._last_intel_ctx["kronos"]) -- None whenever
        # CLIO_DPOF_KRONOS_CONTEXT is off or the read degraded, in which case
        # every record's "kronos" field below is null (never zeros).
        kronos_ctx_all = (self._last_intel_ctx or {}).get("kronos")

        records: list[dict] = []
        for sym in cfg.underlyings:
            u = snap.underlyings.get(sym)
            ctx = underlyings_ctx.get(sym) or {}
            is_signal_symbol = signal is not None and signal.symbol == sym

            fresh = entry_outcomes.get(sym)
            harvest = entry_harvests.get(sym)
            decider_info: Optional[dict] = None
            jev_info: Optional[dict] = None
            # Harvest wins when BOTH fired this cycle for the same underlying
            # (e.g. a harvested "pass" immediately sets the cooldown the SAME
            # cycle's fresh nomination check then reads as blocked, or a
            # harvested "enter" opens the position the fresh check then reads
            # as "position_open") -- the decider's judgment (the rare/
            # expensive event) is always more diagnostically useful than the
            # routine gate-check consequence that mechanically follows it a
            # few lines later in the SAME _handle_entry_pipeline call.
            if harvest is not None:
                outcome = self._classify_harvest_outcome(harvest)
                decider_info = {
                    "provider": harvest.get("provider", ""), "model": harvest.get("model", ""),
                    "action": harvest.get("action"), "conviction": harvest.get("conviction"),
                }
                jev_info = harvest.get("jev")
            elif fresh is not None:
                outcome = fresh["outcome"]
            else:
                outcome = self._fallback_entry_outcome(sym, now_et, cfg, risk, halt_present)

            pillars: dict = {}
            available_sources: list[str] = []
            for name in PILLAR_NAMES:
                pr = u.pillars.get(name) if u is not None else None
                if pr is not None:
                    pillars[name] = {"value": pr.value, "available": pr.available, "source": pr.source}
                    if pr.available and pr.source:
                        available_sources.append(pr.source)
                else:
                    pillars[name] = {"value": None, "available": False, "source": ""}

            direction = ctx.get("direction")
            strength = ctx.get("strength", "none")
            composite = ctx.get("composite")
            vetoes: list[str] = []
            reasons: list[str] = []
            price_confirm: dict = {"ok": None, "reasons": ["not_signal_symbol"]}
            if is_signal_symbol and signal is not None:
                direction = signal.direction.value
                strength = signal.strength
                composite = signal.composite
                vetoes = list(signal.vetoes)
                reasons = list(signal.reasons)
                price_confirm = {
                    "ok": signal.price_confirmed if cfg.require_price_confirmation else None,
                    "reasons": [
                        r for r in signal.reasons
                        if ("price" in r or "momentum" in r or "vwap" in r)
                    ],
                }

            # Task K2 (agent/kronos): compact per-underlying Kronos read for
            # replay fidelity -- null (never zeroed) whenever kronos is
            # disabled/unavailable, or this symbol has no fresh forecast.
            sym_kronos = kronos_ctx_all.get(sym) if isinstance(kronos_ctx_all, dict) else None
            kronos_log: Optional[dict] = None
            if isinstance(sym_kronos, dict):
                h30 = sym_kronos.get("horizons")
                h30 = h30.get("30") if isinstance(h30, dict) else None
                kronos_log = {
                    "asof": sym_kronos.get("asof_bar"),
                    "p_up_30": h30.get("p_up") if isinstance(h30, dict) else None,
                    "mean_return_30": h30.get("mean_return_pct") if isinstance(h30, dict) else None,
                }

            # Task K2 follow-up: the Jev bucket words the log never kept, so a
            # decision-record replay can reconstruct exactly what build_entry_
            # request would have shown Jev this cycle -- computed with jev.py's
            # OWN bucketing helpers (never duplicated) from the same snapshot,
            # never fed back into any gate/decision here.
            jev_words = {
                "gamma_regime": _gamma_regime_words(u),
                "price_vs_vwap": _price_vs_vwap_words(u),
                "momentum_15m": _momentum_words(u),
                "volatility": bucket_vix(snap.vix),
            }

            # Kronos LIVE pillar (§4b): the value the composite actually weighed this cycle, if any.
            kp = u.pillars.get(KRONOS_PILLAR) if u is not None else None
            if kp is not None and kp.available and kp.value is not None:
                kronos_log = dict(kronos_log or {})
                kronos_log["pillar"] = kp.value
                kronos_log["pillar_horizon"] = (kp.detail or {}).get("horizon_min")

            record = {
                "ts": now_et.isoformat(),
                "underlying": sym,
                "composite": composite,
                "strength": strength,
                "direction": direction,
                "pillars": pillars,
                "flow_source": flow_source_label(available_sources),
                "regime": regime_label,
                "price_confirm": price_confirm,
                "vetoes": vetoes,
                "reasons": reasons,
                "is_signal_symbol": is_signal_symbol,
                # evaluate_signal computes composite/strength/vetoes on the
                # PRIMARY underlying; select_instrument may still pick the
                # secondary (divergence) as the symbol to trade.
                "signal_basis": cfg.underlyings[0] if is_signal_symbol and cfg.underlyings else None,
                "nominated": bool(is_signal_symbol and signal is not None and signal.actionable),
                "outcome": outcome,
                "decider": decider_info,
                "jev": jev_info,
                "kronos": kronos_log,
                "jev_words": jev_words,
            }
            records.append(record)

            if self._signal_log.near_miss.should_log(sym, outcome, now, cfg.near_miss_log_minutes * 60):
                logger.info(format_outcome_log_line(record))

        await self._signal_log.write_cycle(records, now_et, cfg.signal_log_max_lines)

    # ------------------------------------------------------------------
    # The cycle
    # ------------------------------------------------------------------

    async def run_cycle(self, bot=None) -> dict:
        async with self._cycle_lock:
            return await self._run_cycle_locked(bot)

    async def _run_cycle_locked(self, bot=None) -> dict:
        # Learning loop (Task ORCH-7Q4K-DPOF builder B3): the most recent
        # cycle's Telegram bot handle, so a background reflection alert
        # (fire-and-forget, never awaited by the cycle) can still reach
        # _safe_alert even though the cycle that scheduled it has long since
        # returned.
        self._last_bot = bot
        cfg = self._config_loader()

        # 1. OFF -> construct ONLY the ledger + alerts (never the broker/feed/
        # orders/jev/decider) and cancel any decision tasks left over from a
        # prior paper/live run, so a stranded open position is still visible
        # and alerted on, not silently unmanaged.
        if cfg.mode == DpofMode.OFF:
            self._cfg = cfg
            await self._cancel_all_decision_tasks()
            await self._ensure_ledger_only(cfg)
            unmanaged = [
                p for p in self._ledger.open_positions() if p.mode in (DpofMode.PAPER.value, DpofMode.LIVE.value)
            ]
            if unmanaged:
                await self._safe_alert(
                    bot,
                    f"DPOF is OFF with {len(unmanaged)} open position(s) — they are NOT being managed; "
                    f"set CLIO_DPOF_MODE=paper (or close manually)",
                    dedupe_key="dpof_off_with_positions",
                    dedupe_seconds=86400,
                )
            return {
                "skipped": "off", "mode": cfg.mode.value, "effective_mode": cfg.mode.value,
                "unmanaged_positions": len(unmanaged),
            }

        await self._ensure_collaborators(cfg)
        self._cfg = cfg
        self._push_config(cfg)

        # 2. Config validity.
        problems = cfg.validate()
        if problems:
            unmanaged = [
                p for p in self._ledger.open_positions() if p.mode in (DpofMode.PAPER.value, DpofMode.LIVE.value)
            ]
            extra = f" — {len(unmanaged)} open position(s) NOT being managed" if unmanaged else ""
            await self._safe_alert(
                bot,
                "[CONFIG] DPOF config invalid: " + "; ".join(problems) + extra,
                dedupe_key="dpof_config_problems",
                dedupe_seconds=86400,
            )
            return {
                "skipped": "config_invalid", "problems": problems, "mode": cfg.mode.value,
                "unmanaged_positions": len(unmanaged),
            }

        if cfg.mode == DpofMode.LOG_ONLY and not (cfg.alpaca_key_id and cfg.alpaca_secret_key):
            return {"skipped": "no_alpaca_keys", "mode": cfg.mode.value}

        # 4. Market hours.
        now = self._clock()
        now_et = datetime.fromtimestamp(now, tz=timezone.utc).astimezone(ET)
        if not self._safe_market_open():
            stranded = [p for p in self._ledger.open_positions() if p.mode in (DpofMode.PAPER.value, DpofMode.LIVE.value)]
            if stranded and cfg.flatten_eod:
                await self._safe_alert(
                    bot,
                    f"[{cfg.mode.value.upper()}] DPOF: market closed with {len(stranded)} position(s) still open "
                    f"({', '.join(sorted({p.underlying for p in stranded}))}) — flatten did not complete; check manually.",
                    dedupe_key=f"dpof_stranded_{now_et.date().isoformat()}",
                    dedupe_seconds=86400,
                )
            # Learning loop F1 (§4.3): a lesson still lands after the close --
            # a close that happened right at/after 15:50 shouldn't have to
            # wait for tomorrow's first market-hours cycle to be reflected on.
            self._maybe_schedule_reflection(now, cfg)
            # decision-record producer: link same-day closes to their
            # entry decision tonight, not at tomorrow's first session cycle.
            self._maybe_schedule_decision_outcomes(now, cfg)
            return {"skipped": "market_closed", "mode": cfg.mode.value, "stranded": len(stranded)}

        # Clamp flatten / entry-end to the real session close (half-days).
        cfg = _session_cfg(cfg, now_et)
        self._cfg = cfg
        self._push_config(cfg)

        self._cycle_count += 1
        is_tenth_cycle = (self._cycle_count % _RECONCILE_EVERY_N_CYCLES == 0)

        summary: dict[str, Any] = {
            "mode": cfg.mode.value,
            "effective_mode": cfg.mode.value,
            "signal": None,
            "entries": [],
            "exits": [],
            "events": [],
            "data_healthy": None,
            "risk": None,
            "errors": [],
        }

        # 5. Gates.
        effective = cfg.mode
        try:
            gates = await self._broker.verify_gates(now)
            effective = self._broker.effective_mode()
            if cfg.mode in (DpofMode.PAPER, DpofMode.LIVE) and not gates.allowed:
                await self._safe_alert(
                    bot, fmt_gates(gates.reasons), dedupe_key="dpof_gates_failed", dedupe_seconds=86400,
                )
        except Exception as e:
            logger.error(f"DPOF engine: verify_gates failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("verify_gates")
        summary["effective_mode"] = effective.value

        # 6. Account / equity / options level (the last gates verticals at
        # runtime — operator directive; no config-time hard error).
        equity: Optional[float] = None
        options_level: Optional[int] = None
        try:
            account = await self._broker.get_account()
            equity = account.get("equity") if account else None
            options_level = account.get("options_trading_level") if account else None
        except Exception as e:
            logger.error(f"DPOF engine: get_account failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("get_account")

        # 6a. Operator visibility: once per process, on the first cycle mode is
        # PAPER/LIVE, announce that DPOF is actively trading.
        if not self._activation_alert_sent and cfg.mode in (DpofMode.PAPER, DpofMode.LIVE):
            if cfg.capital_base > 0:
                equity_txt = f"${equity:,.0f}" if equity is not None else "unknown"
                cap_txt = f"capital ${cfg.capital_base:,.0f} (account equity {equity_txt})"
            else:
                cap_txt = f"account equity ${equity:,.0f}" if equity is not None else "account equity unknown"
            jev_txt = "inactive (no key)" if _jev_inactive(cfg) else cfg.jev_mode.value
            await self._safe_alert(
                bot,
                f"DPOF active: {cfg.mode.value.upper()} | keys: {cfg.alpaca_key_source} | {cap_txt} | "
                f"underlyings {','.join(cfg.underlyings)} | Jev: {jev_txt} | decider: {cfg.decision_model}"
                f"{'/' + cfg.fallback_model if cfg.fallback_model else ''}",
                dedupe_key="dpof_activation",
                dedupe_seconds=30 * 86400,
            )
            self._activation_alert_sent = True

        # 6b. Jev-inactive-due-to-no-key is logged once at INFO.
        if not self._jev_inactive_logged and _jev_inactive(cfg):
            logger.info(
                "DPOF: jev_mode=%s but no TYPESAFE_API_KEY/key_file/fallback URL configured — "
                "Jev is INACTIVE (never called, never blocks/escalates anything) until one is set.",
                cfg.jev_mode.value,
            )
            self._jev_inactive_logged = True

        # 7. Risk rollover (P&L accounting only — no mechanical breakers).
        dpof_equity = _dpof_equity(equity, self._ledger.risk, self._last_unrealized or 0.0, cfg)
        try:
            risk = risk_module.roll_state(self._ledger.risk, now_et, dpof_equity)
        except Exception as e:
            logger.error(f"DPOF engine: roll_state failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("roll_state")
            risk = self._ledger.risk

        exits_log: list[dict] = []

        # 8a. Work orders (fills/cancels) + P&L accounting on a full close.
        events: list[dict] = []
        try:
            events = await self._orders.work_orders(now)
        except Exception as e:
            logger.error(f"DPOF engine: work_orders failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("work_orders")

        for ev in events:
            etype = ev.get("type")
            pid = ev.get("position_id")
            pos = self._ledger.get(pid) if pid else None

            if etype == "entry_filled":
                if pos is not None:
                    summary["entries"].append({"position_id": pid, "qty": ev.get("qty"), "event": "filled"})
                    self._fire_and_forget(self._remember_entry(pos))
                    tag = f"[{(pos.mode or cfg.mode.value).upper()}]"
                    await self._safe_alert(
                        bot,
                        f"{tag} DPOF entry filled\n"
                        f"{pos.underlying} {pos.direction} {pos.structure} qty={ev.get('qty')} "
                        f"entry_price={pos.entry_price if pos.entry_price is not None else 'n/a'}",
                    )

            elif etype == "exit_filled":
                closed = bool(ev.get("closed"))
                exits_log.append(
                    {
                        "position_id": pid,
                        "qty": ev.get("qty"),
                        "realized_pnl_delta": ev.get("realized_pnl_delta"),
                        "closed": closed,
                    }
                )
                if closed and pos is not None:
                    try:
                        risk = risk_module.record_close(risk, pos.underlying, pos.realized_pnl, now)
                    except Exception as e:
                        logger.error(f"DPOF engine: record_close failed for {pid}: {type(e).__name__}: {e}", exc_info=True)
                        summary["errors"].append(f"record_close:{pid}")
                    # Learning loop data capture (§4.1): exit_underlying_price
                    # (from the most recent snapshot -- this cycle's fresh
                    # snapshot isn't fetched until step 9, so `self.
                    # _last_snapshot` is the best "current" reading available
                    # here, same as entry_underlying_price's own capture site)
                    # + the observed P&L% extremes tracked in memory each
                    # cycle by the marks loop (8b) while this position was
                    # open. Popped unconditionally so the in-memory map never
                    # grows unbounded, even if the upsert itself fails.
                    try:
                        extremes = self._pnl_extremes.pop(pid, None)
                        u = self._last_snapshot.underlyings.get(pos.underlying) if self._last_snapshot is not None else None
                        exit_u = u.last_price if u is not None else None
                        obs_max = extremes[0] if extremes is not None else None
                        obs_min = extremes[1] if extremes is not None else None
                        pos = replace(pos, exit_underlying_price=exit_u, obs_max_pnl_pct=obs_max, obs_min_pnl_pct=obs_min)
                        await asyncio.shield(self._ledger.upsert(pos))
                    except Exception as e:
                        logger.error(
                            f"DPOF engine: failed to record close-time observations for {pid}: "
                            f"{type(e).__name__}: {e}",
                        )
                    await self._safe_alert(bot, fmt_exit(pos, pos.exit_reason or "unknown", pos.realized_pnl))
                    self._fire_and_forget(self._learn_from_exit(pos))
                    # Task 3(b) bookkeeping: this position is done reviewing, so
                    # drop its per-position "seen breaking event ids" set.
                    self._seen_breaking_event_ids.pop(pid, None)
                elif not closed:
                    logger.info(f"DPOF partial exit fill: {pid} qty={ev.get('qty')}")

            elif etype == "entry_cancelled":
                reason = ev.get("reason", "")
                is_log_only = pos is not None and pos.exit_reason == "log_only"
                logger.info(f"DPOF entry cancelled: {pid} reason={reason}")
                if not is_log_only:
                    tag = f"[{(pos.mode if pos else cfg.mode.value).upper()}]"
                    await self._safe_alert(
                        bot,
                        f"{tag} DPOF entry cancelled ({reason}): {pos.underlying if pos else pid}",
                        dedupe_key=f"dpof_entry_cancelled_{pid}",
                        dedupe_seconds=3600,
                    )

            elif etype == "cancel_stuck":
                await self._safe_alert(
                    bot,
                    f"[{cfg.mode.value.upper()}] DPOF: {ev.get('purpose', 'order')} order for "
                    f"{pos.underlying if pos else pid} still not cancelled after {ev.get('attempts')} attempts — check the broker.",
                    dedupe_key=f"dpof_cancel_stuck_{pid}",
                    dedupe_seconds=3600,
                )

            else:
                logger.debug(f"DPOF engine: unhandled work_orders event type={etype!r} position_id={pid}")

        # 8b. Marks / unrealized P&L over every not-yet-closed position.
        open_positions = self._ledger.open_positions()
        all_symbols = sorted({leg.get("symbol") for p in open_positions for leg in p.legs if leg.get("symbol")})
        quotes: dict = {}
        try:
            if all_symbols:
                quotes = await self._broker.get_option_quotes(all_symbols) or {}
        except Exception as e:
            logger.error(f"DPOF engine: get_option_quotes (marks) failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("get_option_quotes_marks")

        marks: dict[str, Optional[float]] = {}
        unrealized = 0.0
        for p in open_positions:
            m = self._mark_for(p, quotes)
            marks[p.id] = m
            if m is not None and p.entry_price is not None:
                unrealized += (m - p.entry_price) * 100.0 * p.qty_open
            # Learning loop data capture (§4.1): in-memory observed P&L%
            # extremes while the position is open. Written onto the position
            # ONCE, at close (see the exit_filled handler above) -- never a
            # gate/exit rule (obs_max_pnl_pct/obs_min_pnl_pct are read only by
            # reflection.py's post-trade lesson).
            pnl_pct = position_pnl_pct(p.entry_price, m)
            if pnl_pct is not None:
                extremes = self._pnl_extremes.get(p.id)
                if extremes is None:
                    self._pnl_extremes[p.id] = [pnl_pct, pnl_pct]
                else:
                    extremes[0] = max(extremes[0], pnl_pct)
                    extremes[1] = min(extremes[1], pnl_pct)
        # Drop extremes for positions no longer open (a close that bypassed the
        # exit_filled handler, e.g. reconcile's reconciled_absent).
        open_ids = {p.id for p in open_positions}
        for stale_id in [pid for pid in self._pnl_extremes if pid not in open_ids]:
            self._pnl_extremes.pop(stale_id, None)
        self._last_marks = marks
        self._last_unrealized = unrealized

        # 8c. Reconciliation every Nth cycle, or every cycle while a quantity
        # mismatch is pending confirmation (docs/design/
        # clio_dpof_followups_2026-09-26.md §3.3) -- lets the SAME
        # (ledger,broker) pair be re-observed reconcile_confirm_seconds apart
        # without waiting out a full 10-cycle gap on top of that.
        if is_tenth_cycle or self._orders.has_pending_qty_mismatch():
            try:
                warnings = await self._orders.reconcile(now_epoch=now)
                for w in warnings:
                    await self._safe_alert(
                        bot, f"[RECONCILE] DPOF {w}", dedupe_key=f"dpof_reconcile_{w}", dedupe_seconds=86400,
                    )
            except Exception as e:
                logger.error(f"DPOF engine: reconcile failed: {type(e).__name__}: {e}", exc_info=True)
                summary["errors"].append("reconcile")

        # 9. Fresh snapshot + stale-data alarm. The advisory intel read (step 10b)
        # starts first and runs concurrently with the snapshot.
        intel_task: Optional[asyncio.Task] = None
        try:
            intel_task = asyncio.create_task(self._feed.get_intel_ctx(now))
            # get_intel_ctx never raises; this only keeps an orphaned task (cycle
            # cancelled before step 10b awaits it) from logging "exception never retrieved".
            intel_task.add_done_callback(lambda t: t.cancelled() or t.exception())
        except Exception as e:
            logger.error(f"DPOF engine: could not start get_intel_ctx: {type(e).__name__}: {e}", exc_info=True)
            intel_task = None
        snap: Optional[MarketSnapshot] = None
        try:
            snap = await self._feed.snapshot(now)
            self._last_snapshot = snap
            if snap.data_healthy:
                risk = replace(risk, last_valid_snapshot=now, stale_alarm_sent=False)
            else:
                if risk.last_valid_snapshot is None:
                    risk = replace(risk, last_valid_snapshot=now)
                if (
                    _in_entry_window(now_et, cfg)
                    and risk.last_valid_snapshot is not None
                    and not risk.stale_alarm_sent
                ):
                    elapsed_min = (now - risk.last_valid_snapshot) / 60.0
                    if elapsed_min >= cfg.data_stale_alarm_minutes:
                        await self._safe_alert(
                            bot,
                            fmt_stale(elapsed_min, snap.health_issues),
                            dedupe_key="dpof_data_stale",
                            dedupe_seconds=max(60, cfg.data_stale_alarm_minutes * 60),
                        )
                        risk = replace(risk, stale_alarm_sent=True)
        except Exception as e:
            logger.error(f"DPOF engine: feed.snapshot failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("snapshot")
        summary["data_healthy"] = snap.data_healthy if snap is not None else None
        await self._maybe_alert_kronos_stale(snap, now, now_et, cfg, bot)

        # 10. Signal + shared per-underlying context (used by both position
        # reviews and the entry pipeline below).
        signal: Optional[Signal] = None
        underlyings_ctx: dict = {}
        market_regime = self._market_regime_label()
        try:
            if snap is not None:
                signal = evaluate_signal(snap, cfg)
                self._last_signal = signal
                underlyings_ctx = self._build_underlyings_ctx(snap, cfg, signal)
        except Exception as e:
            logger.error(f"DPOF engine: evaluate_signal failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("evaluate_signal")

        # 10b. Advisory intel (Task 2, INTEL_V3_SPEC.md) — events/vol_surface/
        # breaking/regime/calendar_status, computed ONCE per cycle so position
        # reviews, review triggers, entry cooldown and the entry deliberation
        # ctx all see the same snapshot of it. Never blocks/fails the cycle.
        try:
            self._last_intel_ctx = await self._build_intel_ctx(cfg, snap, now, intel_task=intel_task)
        except Exception as e:
            logger.error(f"DPOF engine: _build_intel_ctx failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("build_intel_ctx")
            self._last_intel_ctx = {}

        # 11. Position reviews — EOD flatten (the one kept rail) first, then
        # Jev-sentinel + trigger detection + decider ReviewDecision harvest.
        try:
            await self._handle_position_reviews(
                now, now_et, cfg, bot, marks, snap, underlyings_ctx, market_regime, risk, dpof_equity, summary, exits_log,
            )
        except Exception as e:
            logger.error(f"DPOF engine: position review pipeline failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("position_reviews")

        # 12. Entry pipeline — harvest any finished deliberation, then launch
        # a new one if the signal, gates, and cooldown all allow it. NEVER
        # awaits the decider inline (decide_entry runs as a background Task).
        try:
            await self._handle_entry_pipeline(now, now_et, cfg, bot, snap, underlyings_ctx, market_regime, risk, dpof_equity, options_level, summary)
        except Exception as e:
            logger.error(f"DPOF engine: entry pipeline failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("entry_pipeline")

        # DPOF-1 (S211): step 12's _apply_entry_outcome recorded the entry on the LEDGER's risk and
        # saved it; this cycle's `risk` was rolled at step 7, before that. Saving it as-is would write
        # the pre-entry entries_today / entries_by_symbol back. Carry the ledger's counters over first.
        risk = risk_module.merge_entry_counters(risk, self._ledger.risk)

        # 13. Persist risk (shielded — this IS the state-of-record write).
        summary["risk"] = risk.to_dict()
        try:
            await asyncio.shield(self._ledger.save_risk(risk))
        except Exception as e:
            logger.error(f"DPOF engine: save_risk failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("save_risk")

        # 13b. Observability (Task ORCH-7Q4K-DPOF): signals.jsonl + day stats +
        # rate-limited INFO logging. Runs LAST, after the real pipeline and the
        # state-of-record write, so it only ever RECORDS what already happened
        # and neither an exception nor a cycle-timeout cancellation landing
        # here can skip any real step.
        try:
            await self._log_cycle_signals(now, now_et, cfg, snap, signal, underlyings_ctx, summary, risk)
        except Exception as e:
            logger.error(f"DPOF engine: log_cycle_signals failed: {type(e).__name__}: {e}", exc_info=True)
            summary["errors"].append("log_cycle_signals")

        # 13c. Learning loop F1 (§4.3): schedule a reflection scan (fire-and-
        # forget; never awaited here) AFTER the real pipeline and the
        # state-of-record write, same placement rule as 13b's observability.
        self._maybe_schedule_reflection(now, cfg)

        # 13d. decision-record producer: schedule the outcome-
        # linkage scan alongside reflection -- same fire-and-forget,
        # after-the-state-of-record-write placement.
        self._maybe_schedule_decision_outcomes(now, cfg)

        # 14. Summary.
        summary["exits"] = exits_log
        summary["events"] = events
        return summary

    # ------------------------------------------------------------------
    # Telegram-facing text
    # ------------------------------------------------------------------

    async def status_text(self) -> str:
        cfg = self._cfg or self._config_loader()
        if cfg.mode == DpofMode.OFF or self._ledger is None:
            return f"DPOF mode={cfg.mode.value}\nengine idle (no cycle has run yet)"

        effective = cfg.mode
        gates_line = "gates: unknown"
        if self._broker is not None:
            try:
                gates = await self._broker.verify_gates(self._clock())
                effective = self._broker.effective_mode()
                gates_line = "gates: OK" if gates.allowed else f"gates: FAILED ({', '.join(gates.reasons)})"
            except Exception as e:
                gates_line = f"gates: error ({type(e).__name__})"

        equity_line = "account equity: unknown"
        dpof_equity_line = ""
        if self._broker is not None:
            try:
                account = await self._broker.get_account()
                acct_equity = account.get("equity") if account else None
                if acct_equity is not None:
                    equity_line = f"account equity: ${acct_equity:,.2f}"
                    dpof_eq = _dpof_equity(acct_equity, self._ledger.risk, self._last_unrealized or 0.0, cfg)
                    if cfg.capital_base > 0:
                        dpof_equity_line = (
                            f"dpof equity: ${dpof_eq:,.2f} (capital_base=${cfg.capital_base:,.2f})"
                            if dpof_eq is not None else "dpof equity: unknown"
                        )
            except Exception as e:
                equity_line = f"account equity: error ({type(e).__name__})"

        risk = self._ledger.risk
        unrealized = self._last_unrealized
        unrealized_str = f"${unrealized:+.2f}" if unrealized is not None else "n/a"

        halt_bits = []
        if risk.hard_halted:
            halt_bits.append(f"HARD_HALT({risk.halt_reason or 'unknown'})")
        if self._ledger.halt_file_present():
            halt_bits.append("HALT_FILE")

        positions = self._ledger.open_positions()
        pos_lines = [
            f"  - {p.underlying} {p.direction} {p.status} qty={p.qty_open}/{p.qty_original} "
            f"reviews={p.review_count} next_review={'in-flight' if p.id in self._review_tasks else 'due' if p.next_review_at is None else f'{max(0, int((p.next_review_at - self._clock())))}s'}"
            for p in positions
        ]

        signal_line = "last signal: none yet"
        if self._last_signal is not None:
            s = self._last_signal
            composite = f"{s.composite:+.2f}" if s.composite is not None else "n/a"
            signal_line = (
                f"last signal: {s.symbol} {s.direction.value} composite={composite} "
                f"strength={s.strength} vetoes={s.vetoes or 'none'}"
            )

        data_line = "data health: no snapshot yet"
        if self._last_snapshot is not None:
            data_line = (
                "data health: OK"
                if self._last_snapshot.data_healthy
                else "data health: ISSUES: " + ", ".join(self._last_snapshot.health_issues)
            )
            selfflow_syms = [
                sym for sym, u in self._last_snapshot.underlyings.items() if getattr(u, "flow_selfflow", False)
            ]
            if selfflow_syms:
                data_line += f" | flow: self-computed fallback active for {', '.join(selfflow_syms)}"

        jev_lines = [f"jev: {_jev_status_text(cfg)}"]
        if self._last_jev_verdict is not None:
            v = self._last_jev_verdict
            jev_lines.append(
                f"last jev verdict: allow={v.allow} reasons={v.veto_reasons or 'none'} "
                f"endpoint={v.endpoint or 'n/a'} error={v.error or 'none'}"
            )

        decider_line = "decider: unavailable (agent.dpof.decider not landed)"
        if self._decider is not None:
            try:
                st = self._decider.status()
                decider_line = f"decider: {st}"
            except Exception as e:
                decider_line = f"decider: status error ({type(e).__name__})"
        model_down_line = f"model_down={self._model_down}"

        lines = [
            f"DPOF mode={cfg.mode.value} effective={effective.value}",
            gates_line,
            equity_line,
            *([dpof_equity_line] if dpof_equity_line else []),
            f"day P&L: realized=${risk.realized_pnl_today:+.2f} unrealized={unrealized_str}",
            f"entries today: {risk.entries_today} by_symbol={dict(risk.entries_by_symbol)}",
            "halts: " + (", ".join(halt_bits) if halt_bits else "none"),
            f"open positions: {len(positions)}",
            *pos_lines,
            signal_line,
            data_line,
            *jev_lines,
            decider_line,
            model_down_line,
            f"in-flight: {len(self._entry_tasks)} entry deliberation(s), {len(self._review_tasks)} review(s)",
        ]
        return "\n".join(lines)

    async def positions_text(self) -> str:
        if self._ledger is None:
            return "DPOF: no cycle has run yet (mode=off or engine idle)."
        positions = self._ledger.open_positions()
        if not positions:
            return "DPOF: no open positions."
        lines = [f"DPOF open positions ({len(positions)}):"]
        for p in positions:
            mark = self._last_marks.get(p.id)
            pnl_pct = position_pnl_pct(p.entry_price, mark)
            pnl_str = f"{pnl_pct:+.1%}" if pnl_pct is not None else "n/a"
            entry_str = f"{p.entry_price:.2f}" if p.entry_price is not None else "n/a"
            thesis = (p.thesis or "n/a")
            if len(thesis) > 140:
                thesis = thesis[:137] + "..."
            last_review_action = (p.last_review or {}).get("action", "none") if p.last_review else "none"
            lines.append(
                f"- {p.id[:8]} {p.underlying} {p.direction} {p.structure} status={p.status} "
                f"qty={p.qty_open}/{p.qty_original} entry={entry_str} pnl={pnl_str} "
                f"reviews={p.review_count} last_review={last_review_action}\n"
                f"  thesis: {thesis}"
            )
        return "\n".join(lines)

    async def config_text(self) -> str:
        cfg = self._cfg or self._config_loader()
        summary = config_summary(cfg)
        lines = [f"{k}={v}" for k, v in summary.items()]
        # Synthetic (not a dataclass field): makes an unconfigured-Jev inactive
        # state visible next to the raw jev_mode= line above, per operator ask.
        lines.append(f"jev={_jev_status_text(cfg)}")
        return "DPOF config:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Operator controls
    # ------------------------------------------------------------------

    async def operator_halt(self, reason: str) -> str:
        async with self._cycle_lock:
            cfg = self._cfg or self._config_loader()
            await self._ensure_collaborators(cfg)
            new_risk = risk_module.operator_halt(self._ledger.risk, reason)
            await asyncio.shield(self._ledger.save_risk(new_risk))
            return f"DPOF halted: {reason}"

    async def operator_resume(self) -> str:
        async with self._cycle_lock:
            return await self._operator_resume_locked()

    async def _operator_resume_locked(self) -> str:
        cfg = self._cfg or self._config_loader()
        await self._ensure_collaborators(cfg)
        equity: Optional[float] = None
        try:
            account = await self._broker.get_account()
            equity = account.get("equity") if account else None
        except Exception as e:
            logger.error(f"DPOF engine: operator_resume account fetch failed: {type(e).__name__}: {e}")
        dpof_equity = _dpof_equity(equity, self._ledger.risk, self._last_unrealized or 0.0, cfg)
        new_risk = risk_module.operator_resume(self._ledger.risk, equity=dpof_equity)
        await asyncio.shield(self._ledger.save_risk(new_risk))
        if equity is None or dpof_equity is None:
            return (
                "DPOF resumed (WARNING: could not fetch account equity — breakers were NOT "
                "re-baselined; re-run /dpof resume once equity is available)"
            )
        if cfg.capital_base > 0:
            return (
                f"DPOF resumed (account equity ${equity:,.2f}, dpof equity ${dpof_equity:,.2f} "
                f"— breakers re-baselined)"
            )
        return f"DPOF resumed (equity ${equity:,.2f} — breakers re-baselined)"

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Cancels every in-flight decision task, then closes the decider +
        jev client + data feed. Idempotent (close() methods are themselves
        idempotent — see jev.py/data.py; decider.close() is expected to be
        too, per the D1 interface)."""
        await self._cancel_all_decision_tasks()
        if self._decider is not None:
            try:
                await self._decider.close()
            except Exception as e:
                logger.debug(f"DPOF engine shutdown: decider close failed: {type(e).__name__}: {e}")
        if self._jev is not None:
            try:
                await self._jev.close()
            except Exception as e:
                logger.debug(f"DPOF engine shutdown: jev close failed: {type(e).__name__}: {e}")
        if self._feed is not None:
            try:
                await self._feed.close()
            except Exception as e:
                logger.debug(f"DPOF engine shutdown: feed close failed: {type(e).__name__}: {e}")
        for name, advisory in (("reflection", self._advisory_reflection), ("debate", self._advisory_debate)):
            if advisory is not None:
                try:
                    await advisory.close()
                except Exception as e:
                    logger.debug(f"DPOF engine shutdown: advisory_llm ({name}) close failed: {type(e).__name__}: {e}")
        if self._decisions is not None:
            try:
                await self._decisions.aclose()
            except Exception as e:
                logger.debug(f"DPOF engine shutdown: decisions aclose failed: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────
# Singleton + scheduler entrypoints
# ─────────────────────────────────────────────────────────────────────────

_engine_singleton: Optional[DPOFEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> DPOFEngine:
    global _engine_singleton
    if _engine_singleton is None:
        with _engine_lock:
            if _engine_singleton is None:
                _engine_singleton = DPOFEngine()
    return _engine_singleton


async def run_dpof_cycle(context) -> None:
    """python-telegram-bot JobQueue entrypoint."""
    bot = getattr(context, "bot", None)
    await get_engine().run_cycle(bot=bot)


async def shutdown_dpof() -> None:
    """Only acts if the singleton was ever created (mirrors the neighbouring
    shutdown hooks in agent/integrations/scheduled_tasks.py)."""
    global _engine_singleton
    if _engine_singleton is not None:
        await _engine_singleton.shutdown()
