"""
CLIO DPOF — contract selection (PURE).

1DTE-standard candidate menu (choose_candidates): delta-banded single-leg
picks plus an optional debit-vertical structure, offered to the decision
model rather than auto-picked. Never guesses: a quote missing greeks, OI, or
a usable bid/ask is simply excluded from the liquid pool for its expiry — an
empty result is diagnosable via the caller's own logging (see
choose_candidates' docstring), never a fabricated candidate.

Puts carry a negative delta from the broker; every delta comparison in this
module uses |delta| so calls and puts share one band (config.min_delta /
config.max_delta).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from agent.dpof.config import DPOFConfig
from agent.dpof.models import ContractChoice, Direction, OptionQuote, OrderLeg

logger = logging.getLogger("clio.dpof.contracts")

# Hydrinia's ATM-fallback OTM tolerance for the "no greeks" strike-distance
# selector (`_dpof_select_spy_option`'s primary candidate pass: `|strike-
# underlying|/underlying <= 3%` — see LATEST_DPOF_SPEC.md §3). Only used by
# choose_candidates' fallback path, never by the delta-banded path.
_STRIKE_DISTANCE_TOLERANCE_PCT = 0.03

# How many delta-neighbor singles choose_candidates offers alongside the
# closest-to-target pick (one step lower |delta|, one step higher).
_DELTA_NEIGHBOR_COUNT = 1


def dte(expiration_iso: str, today: date) -> int:
    """Calendar days from `today` to the option's expiration (ET dates)."""
    exp = date.fromisoformat(expiration_iso)
    return (exp - today).days


def target_expiries(today: date, cfg: DPOFConfig) -> list[date]:
    """1DTE-standard target expiries (operator directive 2026-09-24 / Hydrinia
    design intent — LATEST_DPOF_SPEC.md §2): the N-th next trading session's
    calendar date, for every N in [cfg.expiry_sessions_ahead,
    cfg.max_expiry_sessions_ahead] (both inclusive; DPOFConfig.validate()
    guarantees expiry_sessions_ahead <= max_expiry_sessions_ahead). Index 0 of
    the result is the PRIMARY 1DTE target; the rest are fallback sessions
    choose_candidates walks through only if the primary yields no liquid
    candidates.

    Session counting skips weekends AND agent.integrations.market_events.
    MARKET_HOLIDAYS (Fri -> Mon; around a holiday, etc.) — never anything
    ticker-specific (0DTE/weekly-vs-monthly expiry existence is a chain-
    fetch/liquidity question for choose_candidates, not a calendar question
    this function answers).

    MARKET_HOLIDAYS is imported LAZILY: agent.integrations.market_events is a
    CLIO integrations module, and agent.dpof.contracts stays pure/stdlib-only
    at import time (CLAUDE.md dataclass/module-purity rule — matches how
    agent.dpof.data lazy-imports every agent.integrations.* dependency). If
    the import fails for any reason this degrades to weekends-only skipping
    rather than raising — a degraded calendar is safer than crashing the
    whole snapshot/candidate cycle over a holiday list.
    """
    try:
        from agent.integrations.market_events import MARKET_HOLIDAYS
    except Exception as e:
        logger.debug("DPOF target_expiries: MARKET_HOLIDAYS unavailable (%s); falling back to weekends-only", e)
        MARKET_HOLIDAYS: frozenset = frozenset()

    def _is_session(d: date) -> bool:
        return d.weekday() < 5 and d not in MARKET_HOLIDAYS

    max_n = max(cfg.expiry_sessions_ahead, cfg.max_expiry_sessions_ahead)
    out: list[date] = []
    cursor = today
    session_count = 0
    while session_count < max_n:
        cursor = cursor + timedelta(days=1)
        if not _is_session(cursor):
            continue
        session_count += 1
        if session_count >= cfg.expiry_sessions_ahead:
            out.append(cursor)
    return out


def expiry_fetch_window(today: date, cfg: DPOFConfig) -> tuple[date, date]:
    """[min(target_expiries), max(target_expiries)] — the 1DTE-aware window to
    request from the option chain, spanning every session choose_candidates
    might fall back to (the caller, engine.py, not this module).
    """
    expiries = target_expiries(today, cfg)
    if not expiries:
        # Defensive only: DPOFConfig.validate() guarantees
        # 1 <= expiry_sessions_ahead <= max_expiry_sessions_ahead, so
        # target_expiries() always returns at least one date in practice.
        # [today, today] is a safe empty-ish window (no calendar days skipped,
        # no reliance on the removed v1 min_dte/max_dte fields).
        return today, today
    return expiries[0], expiries[-1]


def option_right(direction: Direction) -> str:
    """"call" for bullish, "put" otherwise.

    Only ever called downstream for an actionable BULLISH/BEARISH signal —
    NEUTRAL is never passed into contract selection by the engine. Mapped
    defensively to "put" rather than raising, so this stays a total function.
    """
    return "call" if direction == Direction.BULLISH else "put"


def _short_vertical_leg(
    chain: list[OptionQuote], long_quote: OptionQuote, direction: Direction, spot: float, cfg: DPOFConfig
) -> tuple[Optional[OptionQuote], list[str]]:
    """Short leg for a debit vertical whose long leg is `long_quote`: further
    OTM near long_strike +/- vertical_width_pct*spot, same expiration/right,
    valid two-sided quote. Liquidity: OI floor, and a spread allowance of 2x
    max_spread_pct (further-OTM strikes carry wider %-spreads on the same $
    width). Used by choose_candidates against a long leg it picked itself
    (the delta-banded/strike-distance 1DTE pick).
    """
    right = option_right(direction)
    is_call = right == "call"
    target_short_strike = (
        long_quote.strike + cfg.vertical_width_pct * spot
        if is_call
        else long_quote.strike - cfg.vertical_width_pct * spot
    )

    def _further_otm(q: OptionQuote) -> bool:
        return q.strike > long_quote.strike if is_call else q.strike < long_quote.strike

    short_candidates = [
        q
        for q in chain
        if q.expiration == long_quote.expiration
        and q.right == right
        and q.symbol != long_quote.symbol
        and _further_otm(q)
        and q.mid is not None
        # Short leg must be liquid too: OI floor, and a spread allowance of 2x the long
        # leg's (further-OTM strikes carry wider %-spreads on the same $ width).
        and q.open_interest is not None
        and q.open_interest >= cfg.min_open_interest
        and q.spread_pct is not None
        and q.spread_pct <= 2 * cfg.max_spread_pct
    ]
    if not short_candidates:
        return None, ["no_short_leg_available"]

    return min(short_candidates, key=lambda q: abs(q.strike - target_short_strike)), []


def _vertical_from_legs(
    long_quote: OptionQuote, short_quote: OptionQuote, direction: Direction, today: date
) -> tuple[Optional[ContractChoice], list[str]]:
    """Common debit/width validation + ContractChoice assembly for a
    (long_quote, short_quote) pair already selected by the caller. Requires
    0 < natural_debit < width. Used by choose_candidates's vertical-candidate
    path.
    """
    width = abs(short_quote.strike - long_quote.strike)
    natural_debit = long_quote.ask - short_quote.bid  # both guaranteed non-None: mid-not-None filters upstream
    if natural_debit <= 0:
        return None, ["vertical_debit_non_positive"]
    if natural_debit >= width:
        return None, ["vertical_debit_exceeds_width"]

    long_leg = OrderLeg(symbol=long_quote.symbol, side="buy", ratio_qty=1, position_intent="buy_to_open")
    short_leg = OrderLeg(symbol=short_quote.symbol, side="sell", ratio_qty=1, position_intent="sell_to_open")
    choice = ContractChoice(
        structure="debit_vertical",
        legs=[long_leg, short_leg],
        underlying=long_quote.underlying,
        direction=direction,
        expiration=long_quote.expiration,
        dte=dte(long_quote.expiration, today),
        est_debit=long_quote.mid - short_quote.mid,
        est_bid=long_quote.bid - short_quote.ask,
        est_ask=natural_debit,
        long_delta=long_quote.delta,
        reasons=[],
    )
    return choice, []


# ─────────────────────────────────────────────────────────────────────────
# 1DTE candidate menu (choose_candidates) — offers the decider a MENU, does
# not auto-pick a single structure the way choose_structure does. No theta
# gate (theta is decision CONTEXT for 1DTE per the operator directive, never
# a mechanical filter) and no min_dte/max_dte/eligible() DTE check (the
# expiry is already pinned to one of target_expiries()'s sessions before
# liquidity filtering even starts).
# ─────────────────────────────────────────────────────────────────────────


def _liquid_for_expiry(
    chain: list[OptionQuote], right: str, expiration_iso: str, cfg: DPOFConfig
) -> list[OptionQuote]:
    """Liquidity-only filter for one fixed (right, expiration) slice of the
    chain: valid two-sided quote, spread/mid <= max_spread_pct, OI >=
    min_open_interest. Deliberately NOT eligible() — no DTE-window check
    (expiration is already pinned by the caller) and no theta gate.
    """
    out: list[OptionQuote] = []
    for q in chain:
        if q.right != right or q.expiration != expiration_iso:
            continue
        if q.mid is None:
            continue
        if q.spread_pct is None or q.spread_pct > cfg.max_spread_pct:
            continue
        if q.open_interest is None or q.open_interest < cfg.min_open_interest:
            continue
        out.append(q)
    return out


def _select_singles(
    liquid: list[OptionQuote], cfg: DPOFConfig, spot: Optional[float]
) -> tuple[list[OptionQuote], str]:
    """Up to 1 + 2*_DELTA_NEIGHBOR_COUNT singles for one expiry's liquid pool:
    the quote closest to cfg.target_delta, then one delta-step lower and one
    delta-step higher (nearest-|delta|-neighbor, not a fixed increment — the
    chain's own strike spacing sets the step), all within [min_delta,
    max_delta]. Returns (picks, mode) where mode is "delta_band" or
    "strike_distance" (Hydrinia's ATM-fallback, LATEST_DPOF_SPEC.md §3: OTM
    tolerance <= _STRIKE_DISTANCE_TOLERANCE_PCT of spot) for diagnosability.

    The strike-distance fallback fires ONLY when every liquid quote lacks
    delta entirely (a feed limitation — "greeks missing", per the D4 spec
    text) — NOT when deltas are present but genuinely outside
    [min_delta, max_delta] for every quote (that is a real "no candidates in
    band" outcome, not a greeks-missing outcome, and is left for the caller
    to treat as "this expiry has nothing" / try the next expiry).
    """
    with_delta = [q for q in liquid if q.delta is not None]
    if with_delta:
        banded = [q for q in with_delta if cfg.min_delta <= abs(q.delta) <= cfg.max_delta]
        if not banded:
            return [], "delta_band"

        ordered = sorted(
            banded,
            key=lambda q: (abs(abs(q.delta) - cfg.target_delta), q.spread_pct, -(q.open_interest or 0), q.symbol),
        )
        closest = ordered[0]
        picks = [closest]

        remaining = [q for q in banded if q.symbol != closest.symbol]
        chosen_abs_delta = abs(closest.delta)
        lower = sorted(
            (q for q in remaining if abs(q.delta) < chosen_abs_delta),
            key=lambda q: (chosen_abs_delta - abs(q.delta), q.spread_pct, -(q.open_interest or 0), q.symbol),
        )
        higher = sorted(
            (q for q in remaining if abs(q.delta) > chosen_abs_delta),
            key=lambda q: (abs(q.delta) - chosen_abs_delta, q.spread_pct, -(q.open_interest or 0), q.symbol),
        )
        picks.extend(lower[:_DELTA_NEIGHBOR_COUNT])
        picks.extend(higher[:_DELTA_NEIGHBOR_COUNT])
        return picks, "delta_band"

    # No quote in this liquid pool has delta at all -> greeks-missing fallback.
    if spot is None:
        return [], "strike_distance"
    near = [q for q in liquid if abs(q.strike - spot) / spot <= _STRIKE_DISTANCE_TOLERANCE_PCT]
    ordered = sorted(near, key=lambda q: (abs(q.strike - spot), q.spread_pct, -(q.open_interest or 0), q.symbol))
    return ordered, "strike_distance"


def _single_from_quote(q: OptionQuote, direction: Direction, today: date) -> ContractChoice:
    leg = OrderLeg(symbol=q.symbol, side="buy", ratio_qty=1, position_intent="buy_to_open")
    return ContractChoice(
        structure="single",
        legs=[leg],
        underlying=q.underlying,
        direction=direction,
        expiration=q.expiration,
        dte=dte(q.expiration, today),
        est_debit=q.mid,
        est_bid=q.bid,
        est_ask=q.ask,
        long_delta=q.delta,
        reasons=[],
    )


def choose_candidates(
    chain: list[OptionQuote],
    *,
    underlying: str,
    direction: Direction,
    spot: Optional[float],
    today: date,
    cfg: DPOFConfig,
    now_et: Optional[object] = None,
) -> list[ContractChoice]:
    """1DTE candidate MENU for the decider (up to cfg.entry_candidates
    ContractChoice objects) — replaces auto-picking one structure
    (choose_structure) with offering several for CLIO's decision model to
    judge. Never a mechanical entry decision by itself: this only narrows an
    option chain down to a liquid, sensibly-spread set of choices; "enter" /
    "pass" / which key is the decider's call (agent/dpof/decider.py).

    Order: target_expiries(today, cfg)'s sessions in order (index 0 = the
    primary 1DTE target); the FIRST expiry that yields at least one candidate
    wins — later (fallback) sessions are tried only when an earlier one
    yields nothing at all (per-expiry liquidity, not a partial mix across
    expiries). Within the winning expiry: up to 1 + 2*_DELTA_NEIGHBOR_COUNT
    singles from _select_singles (closest-to-target-delta pick plus its
    nearest lower/higher-delta neighbors, or the ATM-ish strike-distance
    fallback when greeks are absent from the whole pool), plus one debit
    vertical when cfg.allow_verticals and a valid one can be built from the
    closest single as its long leg (reusing choose_vertical's exact short-leg
    liquidity logic via _short_vertical_leg/_vertical_from_legs). The
    vertical, when built, always gets a reserved slot — it is not dropped in
    favor of an extra single when cfg.entry_candidates is tight. Total
    length is always <= cfg.entry_candidates. [] when nothing liquid at any
    session in range (never raises, never guesses a candidate).

    Unlike choose_structure, this is direction-agnostic w.r.t. VIX (no
    vertical_when_vix_at gate) — the vertical is offered as one more MENU
    OPTION whenever allow_verticals is on and one is buildable, letting the
    decider weigh it rather than code auto-selecting it by VIX level.

    `now_et` is accepted for interface parity with decider.py's calling
    convention (agent/dpof/decider.py, D1) and left for a future time-of-day-
    sensitive refinement (e.g. late-session liquidity tightening); it is not
    read by anything in this function today.

    Stable candidate `key` labels (e.g. "C1") are assigned by the engine, not
    here (DPOF_V2_SPEC.md's EntryCandidate.key) — ContractChoice carries no
    key field.
    """
    right = option_right(direction)
    scoped_chain = [q for q in chain if q.underlying == underlying]

    expiries = target_expiries(today, cfg)
    if not expiries:
        logger.debug("DPOF choose_candidates(%s): target_expiries() returned no sessions", underlying)
        return []

    for expiry in expiries:
        expiration_iso = expiry.isoformat()
        liquid = _liquid_for_expiry(scoped_chain, right, expiration_iso, cfg)
        if not liquid:
            continue

        picks, mode = _select_singles(liquid, cfg, spot)
        if not picks:
            continue

        vertical_choice: Optional[ContractChoice] = None
        if cfg.allow_verticals and spot is not None:
            short, short_reasons = _short_vertical_leg(scoped_chain, picks[0], direction, spot, cfg)
            if short is not None:
                vertical_choice, _v_reasons = _vertical_from_legs(picks[0], short, direction, today)
            else:
                logger.debug(
                    "DPOF choose_candidates(%s): vertical not built for %s expiry: %s",
                    underlying, expiration_iso, short_reasons,
                )

        slot_budget = max(0, cfg.entry_candidates - (1 if vertical_choice is not None else 0))
        out: list[ContractChoice] = [_single_from_quote(q, direction, today) for q in picks[:slot_budget]]
        if vertical_choice is not None:
            out.append(vertical_choice)

        if out:
            return out
        logger.debug(
            "DPOF choose_candidates(%s): %s expiry had %d liquid quote(s) (mode=%s) but produced no candidates",
            underlying, expiration_iso, len(liquid), mode,
        )

    return []
