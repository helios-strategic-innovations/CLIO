"""
CLIO DPOF — shared data contracts.

Every module in agent/dpof/ exchanges these types. Design rules:

1. MISSING IS NOT ZERO. A pillar/price/VIX that could not be read is
   `None` with `available=False` — never silently coerced to 0.0. Hydrinia's
   Sep-7-2026 outage was a fail-open-to-zero parser that ran a full trading
   day on flat FPS=0.00 without anyone noticing; this layer makes that
   structurally impossible (signals require `min_pillars` real readings).
2. Pure data. No I/O, no imports of broker/HTTP SDKs, so every decision
   module is unit-testable without network or credentials.
3. JSON round-trip via to_dict()/from_dict(); from_dict() filters unknown
   fields (CLAUDE.md dataclass-deserialization rule) so old ledger files
   survive schema additions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Optional


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.BULLISH else (-1 if self is Direction.BEARISH else 0)


class Regime(str, Enum):
    TRENDING = "trending"
    NORMAL = "normal"
    CHOPPY = "choppy"
    HALT = "halt"
    UNKNOWN = "unknown"   # VIX unavailable/stale -> treated like CHOPPY (strong-only)


def _filter_kwargs(cls, d: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (d or {}).items() if k in names}


# ─────────────────────────────────────────────────────────────────────────
# Market data
# ─────────────────────────────────────────────────────────────────────────

PILLAR_NAMES = ("nofa", "premium", "darkpool", "gex")


@dataclass
class PillarReading:
    """One signal pillar for one underlying. value in [-1, +1] (bullish > 0)."""
    name: str                          # one of PILLAR_NAMES
    value: Optional[float] = None      # None = unavailable (NEVER 0.0 as a stand-in)
    available: bool = False
    source: str = ""                   # e.g. "uw:/option-trades/flow-alerts"
    age_seconds: Optional[float] = None
    detail: dict = field(default_factory=dict)   # raw-ish context (regime, pin_risk, counts)
    error: str = ""                    # why unavailable, if not available

    @classmethod
    def missing(cls, name: str, error: str, source: str = "") -> "PillarReading":
        return cls(name=name, value=None, available=False, source=source, error=error)


@dataclass
class UnderlyingSnapshot:
    symbol: str
    pillars: dict[str, PillarReading] = field(default_factory=dict)
    gex_regime: Optional[str] = None           # e.g. "negative_gamma", "positive_gamma", "neutral"
    pin_risk: Optional[bool] = None
    last_price: Optional[float] = None
    vwap: Optional[float] = None               # session VWAP from 1-min bars
    ret_5m: Optional[float] = None             # fractional return over last 5 min
    ret_15m: Optional[float] = None            # fractional return over last 15 min
    bars_count: int = 0
    price_age_seconds: Optional[float] = None
    # Decision CONTEXT only (never a mechanical gate): UW NOPE and max-pain
    # strike for the nearest (1DTE) expiry, when available.
    nope: Optional[float] = None
    max_pain: Optional[float] = None
    # True when >=1 of {pillars, nope, max_pain} for this underlying came from
    # the self-computed Alpaca-OPRA/SIP fallback (agent.flowcollector) rather
    # than UW THIS CYCLE (see agent.dpof.data's selfflow-fallback code and
    # docs/design/clio_dpof.md's "Flow fallback" section). Advisory-only
    # summary marker for the decider ctx / `/dpof status` -- never affects
    # data_healthy or any gate. Per-pillar provenance is still on each
    # PillarReading.source ("selfflow:<symbol>" vs "uw:..."); this field just
    # spares callers from re-deriving "should I mention this" from 4+2
    # separate fields every time. UnderlyingSnapshot has no to_dict()/
    # from_dict() and is never persisted (unlike PositionRecord/RiskState),
    # so adding this field with a default is safe -- nothing to migrate.
    flow_selfflow: bool = False
    # Per-pillar directional weights computed by the dynamic-weights layer
    # (live mode only). None => compute_composite uses the static config
    # weights. Never persisted (same reasoning as flow_selfflow above).
    pillar_weights: Optional[dict] = None

    def available_pillars(self) -> dict[str, float]:
        return {k: p.value for k, p in self.pillars.items() if p.available and p.value is not None}


@dataclass
class MarketSnapshot:
    asof: float                                 # epoch seconds (UTC)
    underlyings: dict[str, UnderlyingSnapshot] = field(default_factory=dict)
    vix: Optional[float] = None
    vix_age_seconds: Optional[float] = None
    risk_flags: list[str] = field(default_factory=list)       # e.g. "econ_blackout:FOMC", "breaking_news", "vol_regime:CRISIS"
    headlines: list[str] = field(default_factory=list)        # recent RAW "<source>: <title>" (untrusted; jev.py sanitizes), best-first
    econ_events_soon: list[dict] = field(default_factory=list)
    data_healthy: bool = False
    health_issues: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Signals
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    symbol: str                         # the instrument to trade (SPY or QQQ)
    direction: Direction = Direction.NEUTRAL
    composite: Optional[float] = None   # weighted pillar composite in [-1, +1]; None = not computable (never a 0.0 stand-in)
    conviction: float = 0.0             # [0, 1]
    strength: str = "none"              # "none" | "normal" | "strong"
    regime: Regime = Regime.UNKNOWN
    pillars_used: dict[str, float] = field(default_factory=dict)
    price_confirmed: bool = False
    reasons: list[str] = field(default_factory=list)          # why it fired / why not
    vetoes: list[str] = field(default_factory=list)           # deterministic vetoes (dp_veto, regime, gates)

    @property
    def actionable(self) -> bool:
        return self.direction is not Direction.NEUTRAL and not self.vetoes and self.strength != "none"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["regime"] = self.regime.value
        return d


# ─────────────────────────────────────────────────────────────────────────
# Options / orders
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class OptionQuote:
    symbol: str                          # OCC, e.g. SPY261002C00575000
    underlying: str
    expiration: str                      # ISO date YYYY-MM-DD
    strike: float
    right: str                           # "call" | "put"
    bid: Optional[float] = None
    ask: Optional[float] = None
    delta: Optional[float] = None        # calls > 0, puts < 0 (as returned by broker)
    theta: Optional[float] = None        # per day, usually negative
    iv: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None or self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            return None
        return round((self.bid + self.ask) / 2.0, 4)

    @property
    def spread_pct(self) -> Optional[float]:
        m = self.mid
        if m is None or m <= 0:
            return None
        return (self.ask - self.bid) / m


@dataclass
class OrderLeg:
    symbol: str                          # OCC
    side: str                            # "buy" | "sell"
    ratio_qty: int = 1
    position_intent: str = ""            # "buy_to_open" | "sell_to_open" | "sell_to_close" | "buy_to_close"


@dataclass
class ContractChoice:
    structure: str                       # "single" | "debit_vertical"
    legs: list[OrderLeg] = field(default_factory=list)
    underlying: str = ""
    direction: Direction = Direction.NEUTRAL
    expiration: str = ""
    dte: int = 0
    est_debit: float = 0.0               # per 1 unit (per share, x100 for contract $), > 0
    est_bid: float = 0.0                 # per unit, natural credit if sold right now
    est_ask: float = 0.0                 # per unit, natural debit if bought right now
    long_delta: Optional[float] = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class OrderIntent:
    client_order_id: str                 # always starts with config.client_order_prefix
    legs: list[OrderLeg]
    qty: int
    limit_price: Optional[float]         # per unit; None only for an urgent market exit
    purpose: str                         # "entry" | "exit"
    reason: str = ""
    tif: str = "day"
    position_id: str = ""
    # DpofMode value the POSITION was opened under. Exits for a position opened in
    # paper/live are sent even while entry gates fail (a transient account-fetch or
    # kill-switch state must never trap an open position) — see DPOFBroker.submit.
    position_mode: str = ""


@dataclass
class OrderResult:
    ok: bool
    broker_order_id: str = ""
    client_order_id: str = ""
    status: str = ""                     # broker status string
    filled_qty: int = 0
    filled_avg_price: Optional[float] = None
    error: str = ""
    would_submit: bool = False           # True in LOG_ONLY (nothing sent)


# ─────────────────────────────────────────────────────────────────────────
# Jev
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class JevVerdict:
    mode: str                            # JevMode value at time of call
    called: bool = False
    allow: bool = True                   # False only ever blocks an ENTRY
    veto_reasons: list[str] = field(default_factory=list)
    answers: dict = field(default_factory=dict)   # raw validated answers keyed by question id
    model: str = ""                      # versioned model id that answered
    endpoint: str = ""                   # "primary" | "fallback"
    latency_ms: Optional[float] = None
    error: str = ""
    # decision-record producer: the exact {"state":..., "questions":...} this
    # verdict was built from -- None only when jev_effective_mode was OFF
    # (nothing was built). EXCLUDED from to_dict() (see below) so the
    # data/dpof/jev.jsonl ledger line this feeds stays byte-identical to
    # before this field existed. repr=False so an accidental %r/f-string of
    # a JevVerdict never dumps raw state/headlines into a log line.
    request: Optional[dict] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("request", None)
        return d


# ─────────────────────────────────────────────────────────────────────────
# Ledger / risk
# ─────────────────────────────────────────────────────────────────────────

# Purposes of SYNTHETIC order_fills entries written by OrderManager.reconcile()
# when the broker's quantity of a single-leg contract differs from the ledger's
# (docs/design/clio_dpof_followups_2026-09-26.md §3). "reconcile_add" counts as
# entry quantity, "reconcile_remove" as exited quantity; both carry avg=None
# (price unknown), so a position holding either has incomplete realized P&L.
RECONCILE_FILL_PURPOSES = ("reconcile_add", "reconcile_remove")


@dataclass
class PositionRecord:
    id: str
    underlying: str
    direction: str                       # Direction value
    structure: str                       # "single" | "debit_vertical"
    legs: list[dict] = field(default_factory=list)            # OrderLeg as dicts
    status: str = "pending"              # pending -> open -> closing -> closed | cancelled
    qty_original: int = 0
    qty_open: int = 0
    entry_price: Optional[float] = None  # avg debit per unit actually filled
    entry_limit_price: Optional[float] = None  # limit of the latest entry order (prices a PENDING position's premium)
    entry_time: Optional[float] = None   # epoch of first fill
    entry_signal: dict = field(default_factory=dict)
    entry_jev: dict = field(default_factory=dict)
    entry_client_order_id: str = ""                            # latest entry order in the chain
    entry_client_order_ids: list[str] = field(default_factory=list)   # every entry order ever sent (chain)
    exit_client_order_ids: list[str] = field(default_factory=list)    # every exit order ever sent (chain)
    # Per-order CUMULATIVE fills as last reported by the broker, keyed by client_order_id:
    # {"purpose": "entry"|"exit", "filled_qty": int, "avg": float|None, "status": str}.
    # qty_open / entry_price / realized_pnl are DERIVED from this (never incremented),
    # so re-polling an order or recovering after a restart can't double-count a fill.
    # Reconciliation may also add SYNTHETIC entries (key "reconcile:...", purpose in
    # RECONCILE_FILL_PURPOSES, avg None) that move qty_open to the broker's quantity;
    # exit-batch arithmetic counts purpose "exit" only, so it ignores them.
    order_fills: dict[str, dict] = field(default_factory=dict)
    exit_target_qty: int = 0              # contracts the CURRENT exit batch must sell
    exit_filled_baseline: int = 0         # total exited qty when this exit batch started
    exit_urgent: bool = False             # current exit batch may escalate to market (single-leg)
    working_since: Optional[float] = None # epoch the latest working order was sent (reprice clock)
    reprices: int = 0                     # reprices used by the current entry/exit batch
    cancel_requested: bool = False        # latest order has a CONFIRMED-sent cancel in flight; never resend until terminal
    cancel_requested_at: Optional[float] = None  # epoch the cancel was (re)issued — re-issued if not terminal in time
    cancel_attempts: int = 0              # cancel calls for the current working order (alert when stuck)
    # tiers_hit/peak_pnl_pct/breakeven_armed REMOVED (operator directive, no
    # mechanical trading): the take-profit-tier/trailing-stop/breakeven ladder
    # they backed is gone. from_dict()'s unknown-field filter (_filter_kwargs)
    # means an OLD persisted positions.json carrying these keys still loads
    # fine -- the values are just dropped, never crash.
    realized_pnl: float = 0.0            # dollars, accumulated across partial exits
    exit_reason: str = ""
    closed_time: Optional[float] = None
    mode: str = ""                       # DpofMode value when opened
    # ── Judgment layer (Opus 5.5 / GPT-6 Sol decide; Jev escalates) ──────
    thesis: str = ""                      # decider's entry thesis
    invalidation: str = ""                # what the decider said would break the thesis
    entry_decision: dict = field(default_factory=dict)   # provider/model/conviction/qty/clamped/latency
    last_review: dict = field(default_factory=dict)      # latest applied review decision (summary)
    next_review_at: Optional[float] = None               # epoch the next scheduled review is due
    last_review_at: Optional[float] = None
    last_review_mark: Optional[float] = None             # mark at last review (move-trigger baseline)
    review_count: int = 0
    review_failures: int = 0                             # consecutive decider-unavailable reviews
    # ── Reflection-loop observations (Task ORCH-7Q4K-DPOF builder B2,
    # 2026-09-26 -- see agent/dpof/reflection.py, docs/design/
    # clio_dpof_learning_loop.md §4.1). All four default to None so an OLD
    # persisted positions.json (predating this field set) loads unchanged via
    # from_dict()'s unknown-field filter. entry_underlying_price/
    # exit_underlying_price are the underlying's `last_price` from the
    # snapshot at entry-apply time / at the cycle that records the full
    # close (engine.py sets these -- B2 only adds the fields). obs_max_pnl_pct/
    # obs_min_pnl_pct are OBSERVATIONS ONLY: extremes of the position's
    # mark-based P&L % tracked in memory each cycle while open and written
    # onto the position once, at close (best-effort -- reset by a restart).
    # They are named obs_* on purpose: the REMOVED peak_pnl_pct backed the
    # retired mechanical take-profit/trailing-stop ladder -- nothing in
    # exits.py/the review-trigger path may ever read obs_max_pnl_pct/
    # obs_min_pnl_pct (grep-tested); they exist ONLY for
    # reflection.build_trade_record's post-trade lesson, never a gate.
    entry_underlying_price: Optional[float] = None
    exit_underlying_price: Optional[float] = None
    obs_max_pnl_pct: Optional[float] = None
    obs_min_pnl_pct: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PositionRecord":
        return cls(**_filter_kwargs(cls, d))


@dataclass
class RiskState:
    """Persisted risk counters. Day/week keys are ET dates/ISO weeks."""
    day: str = ""                        # YYYY-MM-DD (ET)
    start_of_day_equity: Optional[float] = None
    realized_pnl_today: float = 0.0
    entries_today: int = 0
    entries_by_symbol: dict[str, int] = field(default_factory=dict)
    last_stop_by_symbol: dict[str, float] = field(default_factory=dict)   # epoch
    consecutive_losses: int = 0
    cooldown_until: Optional[float] = None                                 # epoch
    daily_halted: bool = False
    week: str = ""                       # e.g. 2026-W39
    realized_pnl_week: float = 0.0
    start_of_week_equity: Optional[float] = None
    hwm_equity: Optional[float] = None
    # Cumulative DPOF realized P&L since inception (or the last operator
    # /dpof resume re-baseline, which resets it to 0.0) -- NEVER reset by
    # day/week rollover, unlike realized_pnl_today/_week. Feeds
    # engine._dpof_equity()'s virtual-capital tracking for a shared account.
    # Old persisted risk_state.json files predating this field load it as
    # 0.0 via from_dict()'s unknown-field filter (never crashes on load).
    realized_pnl_total: float = 0.0
    hard_halted: bool = False            # weekly/drawdown/manual — only operator /dpof resume clears
    halt_reason: str = ""
    last_valid_snapshot: Optional[float] = None
    stale_alarm_sent: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskState":
        return cls(**_filter_kwargs(cls, d))


@dataclass
class ExitDecision:
    action: str = "hold"                 # "hold" | "partial" | "close"
    qty: int = 0
    reason: str = ""
    urgent: bool = False                 # True -> may escalate to marketable/market after reprices
    new_breakeven_armed: Optional[bool] = None
    tier_index: Optional[int] = None


@dataclass
class GateResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)


def as_jsonable(obj: Any) -> Any:
    """Best-effort conversion of dataclasses/enums for JSONL logging."""
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: as_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    return obj
