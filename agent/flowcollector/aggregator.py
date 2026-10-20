"""
CLIO flowcollector — rolling-window pillar aggregation (PURE math).

Turns a stream of `classifier.TradeEvent`s (plus a periodic EOD
open-interest/greeks snapshot) into the UW-style pillar shape
`agent.flowcollector.state.get_selfflow` serves: nofa, premium, darkpool,
gex (+ gex_regime), nope, max_pain. Every formula here is standard,
uncopyrightable options-flow/market-microstructure math (see recon doc §3
and §5) implemented from scratch against Alpaca's own field names -- no
third-party (especially AGPL) code is used.

Design choices this module makes and documents (per the "choose and
document" precedent in agent.dpof.data.compute_vwap_and_returns):

  - `nofa` vs `premium` are two DIFFERENT projections of the same trade
    stream, not the same number twice: `nofa` is DIRECTIONAL positioning
    skew (buying calls / selling puts is bullish, buying puts / selling
    calls is bearish -- the `right` sign flips into the sum); `premium` is
    the raw AGGRESSOR imbalance regardless of call/put (bought dollars minus
    sold dollars). A market that is net buying both calls AND puts heavily
    (a long-gamma/hedging environment) can show strong `premium` activity
    with a near-zero `nofa` -- that is intentional, not a bug, and mirrors
    UW's own NOFA/premium-tide split being two distinct signals (recon doc
    §5, `uw_flow_intel._compute_nofa` vs `_compute_premium_signals`).
  - Both are normalized with `tanh(raw / scale)` into [-1, 1], never a hard
    clip -- `scale` is a config-provided constant (dollars of net premium
    that saturate the pillar near +-1), deliberately NOT calibrated against
    live volume this pass (UNVERIFIED -- see FlowCollectorConfig's
    docstring); a raw sum of exactly 0 (genuinely balanced flow, or no
    trades at all) legitimately normalizes to 0.0, which is why every
    pillar here additionally carries a `n`/`empty` quality_notes flag so a
    consumer can distinguish "computed, balanced" from "no data this
    window" without needing a MISSING/None sentinel for a pure-math layer
    that (unlike agent.dpof.data) never itself observes fetch failures.
  - `darkpool` bias classifies EACH dark (exchange="D") print against the
    SIP NBBO MIDPOINT CAPTURED AT INGEST TIME (`TradeEvent.mid_at_ingest`,
    set once by collector.py's `_process_stock_trade` from the live NBBO
    cache at the moment the trade was processed), not the ask/bid quote
    rule (classifier.classify_dark_trade_side) and not a midpoint re-read
    later at snapshot time -- a dark print can sit anywhere in a
    5/15/30-min rolling window, and re-reading "whatever the midpoint
    happens to be right now" at snapshot time would misclassify trades near
    a window's edge during a moving market (see TradeEvent's docstring).
  - `gex` sign convention: dealers are assumed net SHORT calls / net LONG
    puts (the standard SqueezeMetrics/SpotGamma-replica assumption, recon
    doc §3 -- an assumption, not an observed fact): call open interest
    contributes +gamma*OI*100*spot^2, put open interest contributes the
    same magnitude with a NEGATIVE sign. Multiple expiries (0/1/2DTE) often
    share the same strike, each with its OWN gamma (0DTE gamma is typically
    far larger than 2DTE gamma at the same strike) -- `StrikeExposure`
    therefore accumulates a PRE-WEIGHTED `Σ gamma_i * OI_i` per strike/right
    (`call_gamma_oi`/`put_gamma_oi`, summed once per contract by the
    caller — collector.py's `maybe_refresh_strikes`) rather than a single
    "last contract's gamma" value multiplied by total OI, which would
    silently misweight every strike with more than one live expiry.
    `gex_regime` buckets the signed
    total into "positive"/"negative"/"neutral" via a RELATIVE band (a
    fraction of total |contribution|, not a fixed dollar threshold --
    because the dollar magnitude scales with spot^2*100 and is therefore
    wildly different for a $50 vs $600 underlying); the band fraction is a
    chosen constant, UNVERIFIED against real GEX magnitude distributions.
  - `pin_strike` is a best-effort "largest total OI" magnet, explicitly NOT
    UW's own (undisclosed) pin-risk methodology -- do not treat it as
    equivalent.
  - `max_pain` is the textbook formula (sum of ITM intrinsic payout to
    option holders at each candidate spot, minimized over the strike grid)
    -- this needs only EOD open interest + strikes, both natively available
    from Alpaca (recon doc §3), so it has no approximation gap beyond OI
    itself being EOD-stale.
  - `nope` uses the Lily Francus definition (recon doc §3): (call
    delta-weighted volume + put delta-weighted volume) / underlying share
    volume, where put deltas are already negative (Alpaca's own sign
    convention -- agent.dpof.models.OptionQuote docstring: "puts < 0") so
    a plain sum nets the two sides correctly without a separate right-based
    sign flip (unlike nofa/premium, which classify by TRADE SIDE, not
    option greek sign).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from agent.flowcollector.classifier import (
    DARK_POOL_EXCHANGE,
    SIDE_BUY,
    SIDE_SELL,
    TradeEvent,
    classify_dark_trade_side,
    find_sweeps,
)

# ── Normalization scales (tanh saturation points; UNVERIFIED -- see module
#    docstring). Net premium in USD that pushes a pillar to ~tanh(1)=0.76. ──
DEFAULT_NOFA_SCALE_USD = 2_000_000.0
DEFAULT_PREMIUM_SCALE_USD = 4_000_000.0
DEFAULT_DARKPOOL_SCALE_USD = 20_000_000.0

# Relative GEX neutral band: |net_gex| below this fraction of total |contrib|
# is "neutral" rather than weakly positive/negative. Chosen, not measured.
DEFAULT_GEX_NEUTRAL_BAND = 0.05

# "No fake zeros" coverage gates (2026-09-25 fix): below these fractions,
# gex/nope report None (unavailable) instead of a value computed mostly off
# unknown-gamma/unknown-delta contracts silently treated as zero -- see
# compute_gex's and Aggregator.snapshot's docstrings for the incident this
# fixes (Alpaca returning no greeks at all, all day, made GEX read as an
# indistinguishable-from-genuine "neutral" 0.0).
DEFAULT_GEX_MIN_COVERAGE = 0.5
DEFAULT_NOPE_MIN_COVERAGE = 0.5

# Rolling windows this collector maintains per underlying (seconds).
WINDOW_SHORT_SECONDS = 300.0   # 5 min
WINDOW_MEDIUM_SECONDS = 900.0  # 15 min
WINDOW_LONG_SECONDS = 1800.0   # 30 min
WINDOWS = (("5m", WINDOW_SHORT_SECONDS), ("15m", WINDOW_MEDIUM_SECONDS), ("30m", WINDOW_LONG_SECONDS))


def _side_sign(side: str) -> int:
    if side == SIDE_BUY:
        return 1
    if side == SIDE_SELL:
        return -1
    return 0


def _right_sign(right: str) -> int:
    if right == "call":
        return 1
    if right == "put":
        return -1
    return 0


def compute_nofa(option_trades: list[TradeEvent], scale: float = DEFAULT_NOFA_SCALE_USD) -> dict:
    """PURE. Directional (call-vs-put) net premium skew, tanh-normalized to
    [-1, 1]. See module docstring for the nofa-vs-premium distinction."""
    raw = 0.0
    gross = 0.0
    n = 0
    for t in option_trades:
        ss = _side_sign(t.side)
        rs = _right_sign(t.right)
        if ss == 0 or rs == 0:
            continue
        notional = t.price * t.size * 100.0
        raw += notional * ss * rs
        gross += abs(notional)
        n += 1
    value = math.tanh(raw / scale) if scale else 0.0
    # Scale-free share of classified premium, in [-1, 1] — same form as UW's
    # NOFA ((call - put) / total), so DPOF's fallback can compare like with like.
    ratio = raw / gross if gross > 0 else None
    return {"value": value, "ratio": ratio, "raw": raw, "n": n, "empty": n == 0}


def compute_premium(option_trades: list[TradeEvent], scale: float = DEFAULT_PREMIUM_SCALE_USD) -> dict:
    """PURE. Raw aggressor premium imbalance (buys minus sells, any right),
    tanh-normalized to [-1, 1]. See module docstring for the nofa-vs-premium
    distinction."""
    raw = 0.0
    gross = 0.0
    n = 0
    for t in option_trades:
        ss = _side_sign(t.side)
        if ss == 0:
            continue
        notional = t.price * t.size * 100.0
        raw += notional * ss
        gross += abs(notional)
        n += 1
    value = math.tanh(raw / scale) if scale else 0.0
    # Scale-free (buys - sells) / (buys + sells), in [-1, 1] — same form as UW's
    # premium score (net / (|call| + |put|)).
    ratio = raw / gross if gross > 0 else None
    return {"value": value, "ratio": ratio, "raw": raw, "n": n, "empty": n == 0}


def compute_darkpool(
    equity_trades: list[TradeEvent],
    total_equity_volume: Optional[float],
    scale: float = DEFAULT_DARKPOOL_SCALE_USD,
) -> dict:
    """PURE. Isolates exchange="D" (FINRA ADF) equity trades from
    `equity_trades`, classifies each against its OWN `mid_at_ingest` (the
    NBBO midpoint captured when the trade was first processed -- see
    TradeEvent's docstring for why this is ingest-time, not a live re-read
    at call time), and reports a signed tanh-normalized bias plus the
    dark-volume share of total consolidated volume. `total_equity_volume`
    is the caller's own tally of ALL (lit+dark) trade volume in the same
    window -- this function does not recompute it, so it cannot silently
    disagree with whatever window the caller used.
    """
    dark = [t for t in equity_trades if t.exchange == DARK_POOL_EXCHANGE]
    dark_volume = sum(t.size for t in dark)
    signed = 0.0
    gross = 0.0
    classified_n = 0
    for t in dark:
        side = classify_dark_trade_side(t.price, t.mid_at_ingest)
        ss = _side_sign(side)
        if ss == 0:
            continue
        signed += t.price * t.size * ss
        gross += abs(t.price * t.size)
        classified_n += 1
    bias = math.tanh(signed / scale) if scale else 0.0
    ratio = signed / gross if gross > 0 else None   # scale-free, in [-1, 1]
    volume_share: Optional[float] = None
    if total_equity_volume is not None and total_equity_volume > 0:
        volume_share = dark_volume / total_equity_volume
    return {
        "bias": bias,
        "ratio": ratio,
        "raw_signed_notional": signed,
        "dark_volume": dark_volume,
        "dark_trade_count": len(dark),
        "classified_count": classified_n,
        "volume_share": volume_share,
        "empty": len(dark) == 0,
    }


def compute_delta_volume(option_trades: list[TradeEvent], deltas_by_symbol: dict[str, float]) -> float:
    """PURE. Sum of delta(contract)*size over `option_trades`, where each
    contract's delta comes from the latest known greeks snapshot
    (`deltas_by_symbol`, OCC symbol -> delta; calls positive, puts negative
    per Alpaca's own convention -- agent.dpof.models.OptionQuote docstring:
    "puts < 0"). A trade for a contract with NO known delta is excluded
    entirely, never treated as delta=0 (which would silently bias the NOPE
    numerator toward whichever side happens to have fresher greeks)."""
    total = 0.0
    for t in option_trades:
        d = deltas_by_symbol.get(t.symbol)
        if d is None:
            continue
        total += d * t.size
    return total


def option_volume_coverage(option_trades: list[TradeEvent], deltas_by_symbol: dict[str, float]) -> dict:
    """PURE. NOPE's own coverage metric -- distinct from compute_gex's
    OI-weighted gamma coverage, which is about STANDING open interest, not
    TRADED volume. Reports what fraction of `option_trades`' total size (in
    this window) belongs to a contract with a KNOWN delta.
    `compute_delta_volume` already silently excludes unknown-delta trades
    from its sum (by design, see that function's docstring); this function
    tells the caller HOW MUCH was excluded, so NOPE can be gated to None
    (see DEFAULT_NOPE_MIN_COVERAGE) rather than silently reporting a value
    computed off a small, unrepresentative slice of the window's volume --
    the same "no fake zeros" concern compute_gex's `coverage` param
    addresses for gamma/OI. `coverage` is None (not 0.0/1.0) when
    `option_trades` is empty -- there is no legitimate coverage fraction of
    zero trades."""
    known = 0.0
    total = 0.0
    for t in option_trades:
        total += t.size
        if deltas_by_symbol.get(t.symbol) is not None:  # same test as compute_delta_volume
            known += t.size
    coverage = (known / total) if total > 0 else None
    return {"known_volume": known, "total_volume": total, "coverage": coverage}


def compute_nope(net_delta_volume: float, share_volume: Optional[float]) -> Optional[float]:
    """PURE. Lily Francus NOPE definition (recon doc §3): net delta-weighted
    option volume / underlying share volume. `net_delta_volume` is expected
    to be `compute_delta_volume`'s output -- a plain signed sum across BOTH
    calls and puts, since Alpaca's own negative-delta-for-puts convention
    already nets the two sides correctly (see that function's docstring; no
    extra call/put sign flip belongs here, unlike nofa/premium which key off
    TRADE SIDE, not option greek sign). Returns None (never 0.0) when
    share_volume is missing/non-positive -- there is no legitimate
    zero-denominator NOPE."""
    if share_volume is None or share_volume <= 0:
        return None
    return net_delta_volume / share_volume


@dataclass
class StrikeExposure:
    """One strike's aggregated open-interest/gamma inputs for compute_gex.

    `call_oi`/`put_oi` are plain summed open interest (used only for
    `pin_strike`'s magnet ranking below -- max_pain uses its own separate
    `oi_by_strike` dict, not this class). `call_gamma_oi`/`put_gamma_oi` are
    the PRE-WEIGHTED sum Σ gamma_i * OI_i across every contract at this
    strike+right (collector.py's `maybe_refresh_strikes` accumulates these
    per contract as it walks the universe) -- this is NOT the same as
    `call_oi * (some single gamma)`: when multiple expiries (0/1/2DTE) share
    a strike, each has its own gamma (0DTE gamma is typically far larger
    than 2DTE gamma at the same strike), so only a per-contract weighted sum
    is correct. Multiplying a SUMMED oi by a single "last contract seen"
    gamma -- the bug this dataclass shape exists to make impossible --
    silently misweights every multi-expiry strike."""

    strike: float
    call_oi: float = 0.0
    call_gamma_oi: float = 0.0
    put_oi: float = 0.0
    put_gamma_oi: float = 0.0


def compute_gex(
    strikes: list[StrikeExposure],
    spot: Optional[float],
    neutral_band: float = DEFAULT_GEX_NEUTRAL_BAND,
    coverage: Optional[float] = None,
    min_coverage: float = DEFAULT_GEX_MIN_COVERAGE,
) -> dict:
    """PURE. net_gex = sum over strikes of
    (call_gamma_oi - put_gamma_oi) * 100 * spot^2, where call_gamma_oi/
    put_gamma_oi are ALREADY the per-contract Σ gamma_i*OI_i sum (see
    StrikeExposure's docstring -- never re-multiplied by OI here, that
    would double-count it) -- dealer short-calls/long-puts sign convention
    (see module docstring). `gex_regime` is "positive"/"negative"/"neutral",
    using a RELATIVE band (neutral_band * total |contribution|) rather than
    a fixed dollar threshold. `pin_strike` is the strike with the largest
    total (call_oi + put_oi) -- an approximate open-interest magnet, NOT
    UW's pin-risk methodology (see module docstring). Returns
    {"net_gex": None, "regime": None, "pin_strike": None, "empty": True}
    when `strikes` is empty or `spot` is missing/non-positive -- there is no
    legitimate GEX without a spot to square.

    `coverage` (OI-weighted fraction of open interest whose contract has a
    KNOWN gamma -- collector.py's maybe_refresh_strikes computes this; see
    UnderlyingFlowState.gamma_oi_coverage) gates the WHOLE result to
    {"net_gex": None, "regime": None, "pin_strike": None, "empty": True,
    "reason": "low_gamma_coverage"} when below `min_coverage`. `coverage is
    None` (the default, and every pre-2026-09-25 caller/test) means
    "coverage unknown" and does NOT gate -- this is the "no fake zeros" fix
    for the 2026-09-25 incident: a StrikeExposure whose gamma silently
    defaulted to 0.0 for EVERY contract (Alpaca returned no greeks all day)
    made net_gex compute to exactly 0.0 (`regime` "neutral"), indistinguishable
    from a genuinely balanced GEX regime. See greeks.py and collector.py's
    maybe_refresh_strikes/_rebuild_strikes_from_greeks.
    """
    if coverage is not None and coverage < min_coverage:
        return {"net_gex": None, "regime": None, "pin_strike": None, "empty": True, "reason": "low_gamma_coverage"}
    if not strikes or spot is None or spot <= 0:
        return {"net_gex": None, "regime": None, "pin_strike": None, "empty": True}

    net = 0.0
    total_abs = 0.0
    best_strike: Optional[float] = None
    best_oi = -1.0
    for s in strikes:
        call_contrib = s.call_gamma_oi * 100.0 * spot * spot
        put_contrib = s.put_gamma_oi * 100.0 * spot * spot
        net += call_contrib - put_contrib
        total_abs += abs(call_contrib) + abs(put_contrib)
        total_oi = s.call_oi + s.put_oi
        if total_oi > best_oi:
            best_oi = total_oi
            best_strike = s.strike

    if total_abs <= 0:
        regime = "neutral"
    elif abs(net) < neutral_band * total_abs:
        regime = "neutral"
    elif net > 0:
        regime = "positive"
    else:
        regime = "negative"

    return {
        "net_gex": net,
        "regime": regime,
        "pin_strike": best_strike,
        "empty": False,
    }


def compute_max_pain(oi_by_strike: dict[float, dict]) -> Optional[float]:
    """PURE. Textbook max-pain: at each candidate spot in the strike grid,
    sum(call_oi * max(0, candidate - strike) * 100) +
    sum(put_oi * max(0, strike - candidate) * 100) -- the total intrinsic
    payout option HOLDERS would collect if the underlying settled there.
    Returns the candidate strike that MINIMIZES that payout (where writers
    collectively profit most / holders lose most). None (never a fabricated
    value) if `oi_by_strike` is empty.
    """
    if not oi_by_strike:
        return None
    strikes = sorted(oi_by_strike.keys())
    best_strike: Optional[float] = None
    best_pain: Optional[float] = None
    for candidate in strikes:
        pain = 0.0
        for k, d in oi_by_strike.items():
            call_oi = d.get("call_oi") or 0.0
            put_oi = d.get("put_oi") or 0.0
            if candidate > k:
                pain += call_oi * (candidate - k) * 100.0
            if k > candidate:
                pain += put_oi * (k - candidate) * 100.0
        if best_pain is None or pain < best_pain:
            best_pain = pain
            best_strike = candidate
    return best_strike


# ─────────────────────────────────────────────────────────────────────────
# Rolling window store
# ─────────────────────────────────────────────────────────────────────────


class RollingTradeStore:
    """Append-only, time-pruned buffer of TradeEvents for ONE underlying
    (options + equity trades share one store, distinguished by `right`).
    Bounded by `max_age_seconds` (the longest window this collector ever
    needs, WINDOW_LONG_SECONDS by default) so memory never grows unbounded
    across a session -- see also collector.BoundedDropOldestQueue for the
    separate ingest-side backpressure bound.
    """

    def __init__(self, max_age_seconds: float = WINDOW_LONG_SECONDS) -> None:
        self._max_age = max_age_seconds
        self._events: list[TradeEvent] = []

    def add(self, event: TradeEvent) -> None:
        self._events.append(event)

    def prune(self, now_epoch: float) -> int:
        """Drops events older than max_age_seconds; returns count dropped."""
        cutoff = now_epoch - self._max_age
        kept = [e for e in self._events if e.ts >= cutoff]
        dropped = len(self._events) - len(kept)
        self._events = kept
        return dropped

    def window(self, now_epoch: float, window_seconds: float) -> list[TradeEvent]:
        cutoff = now_epoch - window_seconds
        return [e for e in self._events if e.ts >= cutoff]

    def __len__(self) -> int:
        return len(self._events)


@dataclass
class UnderlyingFlowState:
    """Per-underlying rolling state the Aggregator owns. `options` holds
    both calls and puts (right != ""); `equity` holds SPY/QQQ share trades
    (right == "")."""

    symbol: str
    options: RollingTradeStore = field(default_factory=RollingTradeStore)
    equity: RollingTradeStore = field(default_factory=RollingTradeStore)
    # Latest OI/greeks snapshot (collector.py's maybe_refresh_strikes
    # refreshes OI on the slower strikes_refresh_seconds REST cadence --
    # still EOD-stale OI, see recon doc §1's "Open interest" section --
    # while maybe_recompute_greeks refreshes ONLY the self-computed
    # deltas/gammas, network-free, on the faster greeks_recompute_seconds
    # cadence; see both methods' docstrings).
    strikes: list[StrikeExposure] = field(default_factory=list)
    oi_by_strike: dict[float, dict] = field(default_factory=dict)
    deltas_by_symbol: dict[str, float] = field(default_factory=dict)  # OCC symbol -> delta (calls>0, puts<0)
    # OCC symbol -> gamma (always >= 0, either source -- see greeks.py's
    # sanitize_greeks). Parallel to deltas_by_symbol; NOT itself consumed by
    # compute_gex (which reads the already-per-contract-weighted
    # StrikeExposure.call_gamma_oi/put_gamma_oi instead) -- this dict exists
    # so collector.py's maybe_recompute_greeks can rebuild that weighted sum
    # from scratch on its own faster cadence, and so quality_notes can
    # report a "gammas_loaded" count independent of "deltas_loaded".
    gammas_by_symbol: dict[str, float] = field(default_factory=dict)
    # OCC symbol -> "alpaca" | "computed" | "interpolated" | "missing" --
    # the source precedence outcome collector.py's maybe_refresh_strikes
    # recorded for each contract this last OI/greeks refresh.
    # maybe_recompute_greeks only ever touches symbols NOT marked "alpaca"
    # here (see that method's docstring); the "alpaca"-sourced ones are left
    # alone until the next strikes_refresh_seconds pull. "interpolated"
    # (2026-09-25 follow-up) means the contract's own quote/solve failed but
    # its IV was filled from the same-expiry smile of nearby contracts that
    # DID resolve -- see greeks.fill_iv_from_smile.
    greeks_source_by_symbol: dict[str, str] = field(default_factory=dict)
    # OCC symbol -> a TRUSTED iv (Alpaca-sane, or solved by
    # compute_contract_greeks from a usable quote) -- i.e. "alpaca"/
    # "computed"-sourced contracts only, NEVER "interpolated" ones (an
    # interpolated IV is itself an estimate; treating it as an anchor for
    # further interpolation would compound uncertainty rather than pin it to
    # real market data). This is the ANCHOR set greeks.fill_iv_from_smile
    # draws from in both maybe_refresh_strikes and maybe_recompute_greeks --
    # kept on this state object (not recomputed from scratch each cycle) so
    # maybe_recompute_greeks's live-quote-driven anchors accumulate across
    # calls within the same strikes_refresh_seconds window.
    iv_by_symbol: dict[str, float] = field(default_factory=dict)
    # {"alpaca": n, "computed": m, "interpolated": p, "missing": k} --
    # descriptive counts of greeks_source_by_symbol, surfaced in
    # quality_notes. Kept as its own field (not derived on every snapshot()
    # call) so collector.py can update it incrementally without re-scanning
    # every symbol each tick.
    greeks_source_counts: dict = field(default_factory=lambda: {"alpaca": 0, "computed": 0, "interpolated": 0, "missing": 0})
    # OI-weighted fraction of open interest whose contract has a KNOWN
    # gamma (either source) -- None until the first successful strikes
    # refresh. Feeds compute_gex's `coverage` gate (see that function's
    # docstring); NOT the same metric as NOPE's option_volume_coverage
    # (OI-weighted vs TRADED-volume-weighted -- see that function's
    # docstring for why they're kept separate).
    gamma_oi_coverage: Optional[float] = None
    spot: Optional[float] = None
    strikes_as_of: Optional[float] = None


class Aggregator:
    """Owns one UnderlyingFlowState per underlying and assembles the
    `get_selfflow`-shaped snapshot dict. All ingestion/pruning is called
    from collector.py's single-writer drain loop (never concurrently from
    multiple threads -- see collector.py's module docstring), so this class
    itself takes no locks.
    """

    def __init__(
        self,
        underlyings: list[str],
        nofa_scale: float = DEFAULT_NOFA_SCALE_USD,
        premium_scale: float = DEFAULT_PREMIUM_SCALE_USD,
        darkpool_scale: float = DEFAULT_DARKPOOL_SCALE_USD,
        gex_neutral_band: float = DEFAULT_GEX_NEUTRAL_BAND,
        sweep_window_seconds: float = 0.5,
        sweep_min_exchanges: int = 2,
        gex_min_coverage: float = DEFAULT_GEX_MIN_COVERAGE,
        nope_min_coverage: float = DEFAULT_NOPE_MIN_COVERAGE,
    ) -> None:
        self._nofa_scale = nofa_scale
        self._premium_scale = premium_scale
        self._darkpool_scale = darkpool_scale
        self._gex_neutral_band = gex_neutral_band
        self._sweep_window_seconds = sweep_window_seconds
        self._sweep_min_exchanges = sweep_min_exchanges
        self._gex_min_coverage = gex_min_coverage
        self._nope_min_coverage = nope_min_coverage
        self.state: dict[str, UnderlyingFlowState] = {u: UnderlyingFlowState(symbol=u) for u in underlyings}

    def add_option_trade(self, underlying: str, event: TradeEvent) -> None:
        st = self.state.get(underlying)
        if st is None:
            return
        st.options.add(event)

    def add_equity_trade(self, underlying: str, event: TradeEvent) -> None:
        st = self.state.get(underlying)
        if st is None:
            return
        st.equity.add(event)

    def prune(self, now_epoch: float) -> None:
        for st in self.state.values():
            st.options.prune(now_epoch)
            st.equity.prune(now_epoch)

    def snapshot(self, underlying: str, now_epoch: float) -> dict:
        """Builds the `get_selfflow`-shaped dict for one underlying. Top-
        level nofa/premium reflect the freshest (5m) window; `windows`
        additionally carries all three per-window aggregates ("Aggregates
        per underlying (rolling 5/15/30 min)" -- INTEL_V3_SPEC.md) for
        observability. `quality_notes` summarizes counts/emptiness so a
        consumer can tell "balanced flow" apart from "no data" without a
        None sentinel on the pillar values themselves (see module
        docstring). Darkpool classification uses each trade's own
        `mid_at_ingest` (see TradeEvent's docstring) -- no live NBBO
        callback is needed here any more."""
        st = self.state.get(underlying)
        if st is None:
            return {
                "nofa": 0.0, "premium": 0.0, "darkpool": 0.0, "gex": None,
                "nofa_ratio": None, "premium_ratio": None, "darkpool_ratio": None,
                "nope": None, "max_pain": None, "gex_regime": None,
                "as_of": now_epoch, "quality_notes": {"error": f"unknown_underlying:{underlying}"},
            }

        windows: dict[str, dict] = {}
        for label, seconds in WINDOWS:
            opt_w = st.options.window(now_epoch, seconds)
            windows[label] = {
                "nofa": compute_nofa(opt_w, self._nofa_scale),
                "premium": compute_premium(opt_w, self._premium_scale),
            }

        short_opt = st.options.window(now_epoch, WINDOW_SHORT_SECONDS)
        short_eq = st.equity.window(now_epoch, WINDOW_SHORT_SECONDS)
        share_volume_5m = sum(e.size for e in short_eq)

        nofa_5m = windows["5m"]["nofa"]
        premium_5m = windows["5m"]["premium"]
        dp = compute_darkpool(short_eq, share_volume_5m, self._darkpool_scale)
        gex = compute_gex(st.strikes, st.spot, self._gex_neutral_band, coverage=st.gamma_oi_coverage, min_coverage=self._gex_min_coverage)
        delta_vol_5m = compute_delta_volume(short_opt, st.deltas_by_symbol)
        vol_coverage = option_volume_coverage(short_opt, st.deltas_by_symbol)
        nope = compute_nope(delta_vol_5m, share_volume_5m)
        # "No fake zeros" for NOPE too (see option_volume_coverage's
        # docstring): a NOPE computed off a small, unrepresentative slice of
        # this window's option volume (most of it unclassified) is gated to
        # None rather than reported at face value.
        # coverage None = zero option prints in the window: for SPY/QQQ that is a
        # stalled stream, not a balanced tape -> unavailable, never NOPE 0.0.
        if nope is not None and (vol_coverage["coverage"] is None or vol_coverage["coverage"] < self._nope_min_coverage):
            nope = None
        max_pain = compute_max_pain(st.oi_by_strike)
        sweep_idxs = find_sweeps(short_opt, self._sweep_window_seconds, self._sweep_min_exchanges)

        quality_notes = {
            "option_trades_5m": nofa_5m["n"],
            "equity_trades_5m": len(short_eq),
            "share_volume_5m": share_volume_5m,
            "dark_trades_5m": dp["dark_trade_count"],
            "dark_classified_5m": dp["classified_count"],
            "sweep_prints_5m": len(sweep_idxs),
            "strikes_loaded": len(st.strikes),
            "strikes_as_of": st.strikes_as_of,
            "deltas_loaded": len(st.deltas_by_symbol),
            "gammas_loaded": len(st.gammas_by_symbol),
            "greeks_source": dict(st.greeks_source_counts or {}),
            "gamma_oi_coverage": st.gamma_oi_coverage,
            "delta_volume_coverage": vol_coverage["coverage"],
            "nofa_empty": nofa_5m["empty"],
            "premium_empty": premium_5m["empty"],
            "darkpool_empty": dp["empty"],
            "gex_empty": gex["empty"],
            "nope_available": nope is not None,
            "max_pain_available": max_pain is not None,
        }

        return {
            "nofa": nofa_5m["value"],
            "premium": premium_5m["value"],
            "darkpool": dp["bias"],
            # Scale-free [-1, 1] ratios (None when nothing classified) — what DPOF's
            # UW fallback consumes; the tanh values above depend on $ scale constants.
            "nofa_ratio": nofa_5m["ratio"],
            "premium_ratio": premium_5m["ratio"],
            "darkpool_ratio": dp["ratio"],
            "gex": gex["net_gex"],
            "gex_regime": gex["regime"],
            "nope": nope,
            "max_pain": max_pain,
            "as_of": now_epoch,
            "quality_notes": quality_notes,
            "windows": windows,
            "darkpool_detail": {"volume_share": dp["volume_share"], "dark_volume": dp["dark_volume"]},
            "gex_detail": {"pin_strike": gex["pin_strike"]},
        }
