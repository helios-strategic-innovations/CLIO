"""
CLIO flowcollector — self-computed Black-Scholes-Merton greeks (PURE, stdlib
only: `math` + `datetime`/`zoneinfo`, nothing else).

Why this module exists (2026-09-25 live incident): `collector.py`'s
`_get_option_greeks_chunk` reads `OptionsSnapshot.greeks` from Alpaca's
`OptionHistoricalDataClient.get_option_snapshot` (OPRA feed, Algo Trader
Plus). The installed alpaca-py (`alpaca/data/models/snapshots.py`) confirms
that parsing is correct: `greeks` is `Optional[OptionsGreeks]`, populated
only `if mapped_snapshot.get("greeks", None) is not None` -- i.e. Alpaca's
OWN API response simply omitted the `"greeks"` key (or sent it null) for
every symbol all day on 2026-09-25, no errors, nothing to parse wrong. This
module exists so the collector never depends on Alpaca actually returning
greeks: whenever Alpaca's own delta/gamma are missing or fail a sanity
check, `collector.py` falls back to computing them itself from a live
bid/ask quote via textbook Black-Scholes-Merton (continuous dividend yield
q) plus a bisection implied-vol solve. No third-party/copyrighted code is
used -- BSM and bisection root-finding are decades-old, textbook,
uncopyrightable methodology (same "choose and document" precedent as
aggregator.py/classifier.py's module docstrings).

Conventions:
  - All prices/rates are per-share (not per-contract-times-100) -- callers
    multiply by the 100 contract multiplier themselves where needed (see
    aggregator.py's StrikeExposure/compute_gex, which already does this for
    gamma*OI).
  - `T` (time to expiry) is in YEARS, calendar-year convention (365 days --
    chosen, not measured, like every other scale constant in this package;
    see aggregator.py's module docstring for the "choose and document"
    pattern this repeats).
  - `r`/`q` are continuously-compounded ANNUAL rates (e.g. 0.04 = 4%).
  - `right` is the string "call" or "put" (matches classifier.py/
    universe.py's convention -- never "C"/"P").
  - Every public function returns `None` (never a fabricated number) when
    its inputs are missing, non-finite, or out of a sane/no-arbitrage range
    -- the same MISSING-IS-NEVER-ZERO discipline as the rest of this
    package (see aggregator.py/state.py's module docstrings).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────────────────────────────────
# Numeric guards
# ─────────────────────────────────────────────────────────────────────────


def is_finite(x: object) -> bool:
    """PURE. True only for a real, finite int/float -- excludes bool
    (subclass of int; CLAUDE.md numeric type-guard discipline), NaN, +-inf,
    and non-numeric types."""
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    return math.isfinite(x)


def sanitize_greeks(delta: object, gamma: object) -> Optional[dict]:
    """PURE. The one sanity bar applied to a (delta, gamma) pair from ANY
    source -- Alpaca's own snapshot greeks OR this module's BSM output:
    both finite, |delta| <= 1 (+ a tiny float-noise tolerance), gamma >= 0
    (a small negative gamma from float noise is clamped to 0.0 rather than
    rejected outright; a genuinely negative gamma is not a legitimate
    vanilla-option value and IS rejected). Returns None -- never a
    partially-sane pair -- if either leg fails; collector.py treats that
    contract's greeks as unusable from this source and tries the next one
    in the precedence chain (see collector.py's module docstring, "source
    precedence")."""
    if not is_finite(delta) or not is_finite(gamma):
        return None
    if abs(delta) > 1.0 + 1e-6:
        return None
    if gamma < -1e-6:
        return None
    return {"delta": float(delta), "gamma": max(0.0, float(gamma))}


# ─────────────────────────────────────────────────────────────────────────
# Time to expiry
# ─────────────────────────────────────────────────────────────────────────

_SECONDS_PER_YEAR = 365.0 * 86400.0  # calendar-year convention -- UNVERIFIED/chosen, see module docstring
_MIN_T_SECONDS = 5.0 * 60.0  # floor: never let T represent less than 5 minutes


def _early_close_dates() -> frozenset:
    """Lazy import of agent.integrations.market_events's early-close date
    set, mirroring collector.py's own `_market_calendar()` pattern exactly
    -- never required (an unavailable/failed import degrades to "no early
    closes known", not an exception), and that module's own docstring notes
    its date sets are 2026/2027-only (extend there annually, not here)."""
    try:
        from agent.integrations.market_events import MARKET_EARLY_CLOSE

        return MARKET_EARLY_CLOSE
    except Exception:
        return frozenset()


def time_to_expiry_years(
    now: datetime, expiry_date: date, early_close_dates: Optional[frozenset] = None
) -> Optional[float]:
    """PURE-ish (one lazy, optional import -- see `_early_close_dates`).
    Time to expiry, in YEARS, to the 16:00 ET close of `expiry_date` (13:00
    ET on an early-close half-day per `agent.integrations.market_events.
    MARKET_EARLY_CLOSE`). `now` is coerced to America/New_York; a naive
    `now` is treated as already being UTC first (matching this codebase's
    `datetime.now(timezone.utc)` convention -- CLAUDE.md anti-pattern #9),
    then converted.

    Returns None if `now` is already at/past that close instant -- there is
    no legitimate (positive) time-to-expiry past it, and BSM is undefined
    for T <= 0. Otherwise the result is floored at the 5-minute-equivalent
    number of years (`_MIN_T_SECONDS`) so a contract expiring in, say, 90
    real seconds still yields a solvable (if crude) BSM input rather than
    dividing by a near-zero T.
    """
    if now is None or expiry_date is None:
        return None
    et = ZoneInfo("America/New_York")
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_et = now.astimezone(et)
    if early_close_dates is None:
        early_close_dates = _early_close_dates()
    close_hour, close_minute = (13, 0) if expiry_date in early_close_dates else (16, 0)
    close_dt = datetime(
        expiry_date.year, expiry_date.month, expiry_date.day,
        close_hour, close_minute, 0, tzinfo=et,
    )
    delta_seconds = (close_dt - now_et).total_seconds()
    if delta_seconds <= 0:
        return None
    delta_seconds = max(delta_seconds, _MIN_T_SECONDS)
    return delta_seconds / _SECONDS_PER_YEAR


# ─────────────────────────────────────────────────────────────────────────
# Black-Scholes-Merton (continuous dividend yield q)
# ─────────────────────────────────────────────────────────────────────────

_SQRT_2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """PURE. Standard normal CDF via math.erf -- no scipy/numpy dependency
    (this package is deliberately stdlib-only, see module docstring)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def norm_pdf(x: float) -> float:
    """PURE. Standard normal PDF."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _d1_d2(
    spot: Optional[float], strike: Optional[float], T: Optional[float], r: float, q: float, sigma: Optional[float]
) -> Optional[tuple]:
    """PURE. Returns (d1, d2), or None if any input is missing/degenerate
    (spot/strike/T/sigma must all be > 0 -- BSM's d1/d2 are undefined
    otherwise, e.g. log(spot/strike) blows up at strike<=0, division by
    sigma*sqrt(T) blows up at either <= 0)."""
    if spot is None or strike is None or T is None or sigma is None:
        return None
    if not (is_finite(spot) and is_finite(strike) and is_finite(T) and is_finite(sigma)):
        return None
    if spot <= 0 or strike <= 0 or T <= 0 or sigma <= 0:
        return None
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bsm_price(spot: Optional[float], strike: Optional[float], T: Optional[float], r: float, q: float, sigma: Optional[float], right: str) -> Optional[float]:
    """PURE. Black-Scholes-Merton price with continuous dividend yield q.
    None for degenerate inputs (see `_d1_d2`) or an unrecognized `right`."""
    d = _d1_d2(spot, strike, T, r, q, sigma)
    if d is None or right not in ("call", "put"):
        return None
    d1, d2 = d
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if right == "call":
        return spot * disc_q * norm_cdf(d1) - strike * disc_r * norm_cdf(d2)
    return strike * disc_r * norm_cdf(-d2) - spot * disc_q * norm_cdf(-d1)


def bsm_delta(spot: Optional[float], strike: Optional[float], T: Optional[float], r: float, q: float, sigma: Optional[float], right: str) -> Optional[float]:
    """PURE. BSM delta (dividend-adjusted: e^{-qT} * N(d1) for a call,
    e^{-qT} * (N(d1) - 1) for a put -- equivalently -e^{-qT}*N(-d1))."""
    d = _d1_d2(spot, strike, T, r, q, sigma)
    if d is None or right not in ("call", "put"):
        return None
    d1, _d2 = d
    disc_q = math.exp(-q * T)
    if right == "call":
        return disc_q * norm_cdf(d1)
    return disc_q * (norm_cdf(d1) - 1.0)


def bsm_gamma(spot: Optional[float], strike: Optional[float], T: Optional[float], r: float, q: float, sigma: Optional[float]) -> Optional[float]:
    """PURE. BSM gamma (identical formula for calls and puts -- no `right`
    argument needed, same as every other options-pricing library)."""
    d = _d1_d2(spot, strike, T, r, q, sigma)
    if d is None:
        return None
    d1, _d2 = d
    disc_q = math.exp(-q * T)
    return disc_q * norm_pdf(d1) / (spot * sigma * math.sqrt(T))


def bsm_vega(spot: Optional[float], strike: Optional[float], T: Optional[float], r: float, q: float, sigma: Optional[float]) -> Optional[float]:
    """PURE. BSM vega (dPrice/dSigma, identical formula for calls and puts).
    Used only by `implied_volatility` as a solvability sanity check (see
    that function's docstring) -- this package otherwise has no use for
    vega itself (no vega-weighted anything downstream)."""
    d = _d1_d2(spot, strike, T, r, q, sigma)
    if d is None:
        return None
    d1, _d2 = d
    disc_q = math.exp(-q * T)
    return spot * disc_q * math.sqrt(T) * norm_pdf(d1)


# ─────────────────────────────────────────────────────────────────────────
# Implied volatility (bracketed bisection)
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_IV_LO = 0.005
DEFAULT_IV_HI = 5.0
DEFAULT_IV_TOL = 1e-6  # price tolerance, not sigma tolerance
DEFAULT_IV_MAX_ITER = 100

# Minimum |vega| (dPrice/dSigma) a solved sigma must have to be trusted, in
# the same $ units as `price`. A PRICE-tolerance bisection (as specified)
# is mathematically sound but numerically ill-conditioned wherever vega is
# near zero -- deep ITM/OTM and/or very-short-T contracts, where a wide
# RANGE of sigma values all reproduce ~the same price to within `tol`, so
# the "converged" sigma is effectively arbitrary, not a meaningful implied
# vol. Empirically probed (not just theorized) against this module's own
# bsm_price/bsm_vega: a realistic 0DTE SPY/QQQ contract inside this
# collector's +/-4% strike band has vega in the ~0.2-12 range (ATM, hours to
# close); a deep-ITM-at-minutes-to-close pathological case measured ~1e-38 --
# 1e-6 sits many orders of magnitude below every realistic case and many
# orders above every pathological one, so it cleanly separates them without
# needing a moneyness/DTE special case. Chosen from that probe, not a
# textbook constant -- see this module's docstring's "choose and document"
# precedent.
DEFAULT_IV_MIN_VEGA = 1e-6


def _no_arbitrage_bounds(spot: float, strike: float, T: float, r: float, q: float, right: str) -> tuple:
    """PURE. Textbook no-arbitrage price bounds for a European option:
    call in [max(0, S*e^-qT - K*e^-rT), S*e^-qT]; put in
    [max(0, K*e^-rT - S*e^-qT), K*e^-rT]. A quoted mid outside this band
    cannot be reproduced by ANY sigma (BSM price is monotonic increasing in
    sigma and asymptotes to the upper bound as sigma -> infinity, to the
    lower bound as sigma -> 0) -- implied_volatility uses this to return
    None rather than extrapolating a meaningless sigma."""
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if right == "call":
        return max(0.0, spot * disc_q - strike * disc_r), spot * disc_q
    return max(0.0, strike * disc_r - spot * disc_q), strike * disc_r


def implied_volatility(
    price: Optional[float], spot: Optional[float], strike: Optional[float], T: Optional[float],
    r: float, q: float, right: str,
    lo: float = DEFAULT_IV_LO, hi: float = DEFAULT_IV_HI,
    tol: float = DEFAULT_IV_TOL, max_iter: int = DEFAULT_IV_MAX_ITER,
    min_vega: float = DEFAULT_IV_MIN_VEGA,
) -> Optional[float]:
    """PURE. Solves for sigma such that bsm_price(..., sigma, right) ==
    `price`, via bracketed bisection on sigma in [lo, hi] (BSM price is
    strictly monotonic increasing in sigma, so bisection is robust -- no
    Newton/Brent derivative or secant instability to worry about; simplicity
    over marginal speed, since this runs at most a few hundred times per
    refresh cycle -- see collector.py's `_compute_greeks_batch_sync`).

    Returns None (never an extrapolated/guessed sigma) when:
      - any input is missing/non-finite, or spot/strike/T <= 0, or `price`
        is negative;
      - `price` sits outside the no-arbitrage band for this contract
        (`_no_arbitrage_bounds`) by more than a tiny float-noise epsilon;
      - `price` is achievable only at exactly `lo`/`hi` beyond `tol` (i.e.
        even the widest sigma in the bracket can't reach it -- a bracket
        [0.005, 5.0] covers effectively every real-world liquid-option IV,
        so this should be rare and usually indicates a bad quote, not a
        genuinely extreme IV);
      - the candidate sigma's vega (`bsm_vega`) is below `min_vega` -- a
        PRICE-tolerance bisection is ill-conditioned wherever vega is near
        zero (deep ITM/OTM and/or very-short-T contracts: a wide RANGE of
        sigma all reproduce ~the same price within `tol`, so the
        "converged" value is effectively arbitrary rather than a meaningful
        implied vol). See `DEFAULT_IV_MIN_VEGA`'s docstring for how that
        threshold was picked (empirically probed, not theorized).
    """
    if price is None or spot is None or strike is None or T is None:
        return None
    if not (is_finite(price) and is_finite(spot) and is_finite(strike) and is_finite(T)):
        return None
    if spot <= 0 or strike <= 0 or T <= 0 or price < 0:
        return None
    if right not in ("call", "put"):
        return None

    lower, upper = _no_arbitrage_bounds(spot, strike, T, r, q, right)
    eps = max(1e-9, 1e-9 * max(1.0, upper))
    if price < lower - eps or price > upper + eps:
        return None

    def _accept(sigma: float) -> Optional[float]:
        v = bsm_vega(spot, strike, T, r, q, sigma)
        if v is None or v < min_vega:
            return None
        return sigma

    price_lo = bsm_price(spot, strike, T, r, q, lo, right)
    price_hi = bsm_price(spot, strike, T, r, q, hi, right)
    if price_lo is None or price_hi is None:
        return None
    if price <= price_lo:
        return _accept(lo) if (price_lo - price) <= tol else None
    if price >= price_hi:
        return _accept(hi) if (price - price_hi) <= tol else None

    a, b = lo, hi
    for _ in range(max_iter):
        mid = (a + b) / 2.0
        p_mid = bsm_price(spot, strike, T, r, q, mid, right)
        if p_mid is None:
            return None
        if abs(p_mid - price) <= tol:
            return _accept(mid)
        if p_mid < price:
            a = mid
        else:
            b = mid
    return _accept((a + b) / 2.0)


# ─────────────────────────────────────────────────────────────────────────
# Public entry point: one contract's quote -> {iv, delta, gamma}
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_MAX_SPREAD_PCT_FOR_IV = 0.5


def compute_contract_greeks(
    bid: Optional[float], ask: Optional[float], spot: Optional[float], strike: Optional[float],
    T: Optional[float], r: float, q: float, right: str,
    iv_hint: Optional[float] = None,
    max_spread_pct_for_iv: float = DEFAULT_MAX_SPREAD_PCT_FOR_IV,
) -> Optional[dict]:
    """PURE. Self-computed {"iv", "delta", "gamma"} for one contract from a
    bid/ask quote, or None if the quote/inputs aren't usable. This is
    collector.py's fallback whenever Alpaca's own snapshot greeks are
    missing or fail `sanitize_greeks` (see module docstring).

    Guards, in order:
      1. `right` must be "call"/"put"; spot/strike/T must be present,
         finite, and > 0.
      2. `bid` must be present, finite, and > 0 (a zero/missing bid is not
         a usable two-sided quote).
      3. `ask` must be present, finite, and > bid.
      4. relative spread (ask-bid)/mid must be <= `max_spread_pct_for_iv`
         -- a wide/crossed-looking quote produces an unreliable IV solve
         (this is the SAME discipline as `agent.dpof`'s own mid-price-only
         execution guard, applied here to inputs rather than orders).
      5. `iv_hint` (typically Alpaca's own `implied_volatility` field, when
         present and in (0, 5]) is used AS-IS instead of re-solving --
         Alpaca's IV can be sane even when its greeks/delta/gamma aren't
         (they come from separate response fields); solving from scratch
         only when no usable hint is given.
      6. the resulting (delta, gamma) must pass `sanitize_greeks` -- a
         degenerate near-zero-T or extreme-strike input can otherwise
         produce a numerically absurd gamma.
    """
    if right not in ("call", "put"):
        return None
    if spot is None or strike is None or T is None:
        return None
    if not (is_finite(spot) and is_finite(strike) and is_finite(T)):
        return None
    if spot <= 0 or strike <= 0 or T <= 0:
        return None
    if bid is None or ask is None or not is_finite(bid) or not is_finite(ask):
        return None
    if bid <= 0 or ask <= bid:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    spread_pct = (ask - bid) / mid
    if spread_pct > max_spread_pct_for_iv:
        return None

    if iv_hint is not None and is_finite(iv_hint) and 0.0 < iv_hint <= 5.0:
        iv = float(iv_hint)
    else:
        iv = implied_volatility(mid, spot, strike, T, r, q, right)
    if iv is None or not is_finite(iv) or iv <= 0:
        return None

    delta = bsm_delta(spot, strike, T, r, q, iv, right)
    gamma = bsm_gamma(spot, strike, T, r, q, iv)
    sane = sanitize_greeks(delta, gamma)
    if sane is None:
        return None
    return {"iv": iv, "delta": sane["delta"], "gamma": sane["gamma"]}


# ─────────────────────────────────────────────────────────────────────────
# IV smile fill (2026-09-25 follow-up -- coverage-risk fix)
#
# The +/-4% / 0-2DTE SPY/QQQ universe has a real gap `compute_contract_
# greeks` cannot close on its own: beyond roughly +/-1.5% (0DTE) or +/-3%
# (1-2DTE) of spot, these contracts routinely quote bid=0 or a few cents
# with a huge relative spread -- both fail that function's bid>0 /
# max_spread_pct_for_iv guards, by design (a bad quote should never be
# solved into a fabricated-precision IV). Those strikes often carry LARGE
# open interest (round numbers draw OI), so leaving them all "missing"
# drags gamma_oi_coverage below DEFAULT_GEX_MIN_COVERAGE for large chunks of
# the day -- exactly defeating the point of the BSM fallback. Their true
# gamma is SMALL (far OTM), not unknown: this section fills their IV from
# the same-expiry smile of contracts that DID resolve (a "trusted anchor"),
# rather than leaving them uncomputed.
# ─────────────────────────────────────────────────────────────────────────


def _interp_iv(strike: float, strikes_sorted: list, ivs_sorted: list) -> float:
    """PURE. Linear-in-strike interpolation of `ivs_sorted` (parallel,
    strike-sorted) at `strike`; flat (nearest-anchor) extrapolation outside
    [strikes_sorted[0], strikes_sorted[-1]]. Caller (`fill_iv_from_smile`)
    guarantees `strikes_sorted` has >= 2 distinct-or-equal entries and is
    sorted ascending."""
    if strike <= strikes_sorted[0]:
        return ivs_sorted[0]
    if strike >= strikes_sorted[-1]:
        return ivs_sorted[-1]
    n = len(strikes_sorted)
    for i in range(1, n):
        if strikes_sorted[i] >= strike:
            k_lo, k_hi = strikes_sorted[i - 1], strikes_sorted[i]
            iv_lo, iv_hi = ivs_sorted[i - 1], ivs_sorted[i]
            if k_hi <= k_lo:
                return iv_lo
            w = (strike - k_lo) / (k_hi - k_lo)
            return iv_lo + w * (iv_hi - iv_lo)
    return ivs_sorted[-1]  # defensive/unreachable given the two guards above


def fill_iv_from_smile(
    anchors: list,
    targets: list,
    spot: Optional[float] = None,
    strike_lo: Optional[float] = None,
    strike_hi: Optional[float] = None,
) -> dict:
    """PURE. Per-expiry, linear-in-strike IV smile fill for contracts whose
    own quote/solve failed.

    `anchors`: contracts with a TRUSTED iv already (Alpaca-sane, or solved
    by `compute_contract_greeks` from a usable quote) -- each a dict with
    "symbol", "strike", "expiry" (any hashable/comparable key -- collector.py
    passes the contract's `date`), "right" ("call"/"put"), "iv" (float > 0).
    `targets`: contracts needing a fill -- same shape, no "iv" key required.

    Returns {symbol: iv} for every target that got a fill (never a
    fabricated entry for one that didn't). A target's expiry group needs
    >= 2 DISTINCT anchor strikes to interpolate FROM at all -- a single
    anchor strike (even from two contracts, e.g. a call and a put at that
    one strike) carries no SLOPE information, only a level, so per this
    function's "never guess a slope from one point" rule such a target is
    left unfilled (absent from the result) rather than flatly copying that
    one anchor's IV everywhere in the expiry.

    `strike_lo`/`strike_hi` (the caller's own configured strike band edges,
    e.g. spot*(1 +/- strike_band_pct)) bound how far FLAT extrapolation may
    reach beyond the outermost anchor: a target strike outside
    [strike_lo, strike_hi] is left unfilled even if its expiry has >= 2
    anchors. Omit either (None, the default) to not bound that side.

    Same-strike anchor dedup: a strike almost always has BOTH a call and a
    put anchor. When both are present at one strike (same expiry), this
    function picks ONE representative IV rather than interpolating a
    3-legged curve: with `spot` given, the OTM leg is preferred (OTM options
    are the standard smile-fitting convention here -- more liquid, cleaner
    aggressor-side price discovery than their ITM counterpart at the exact
    same strike, which is thinner-traded and more early-exercise/discount-
    affected); without `spot`, or when both legs are on the same side (e.g.
    exactly ATM, or `spot` unavailable), the two IVs are averaged --
    deterministic, no directional guess. Documented design choice, not the
    only valid one.
    """
    by_expiry: dict = {}
    for a in anchors or []:
        strike = a.get("strike")
        iv = a.get("iv")
        expiry = a.get("expiry")
        right = a.get("right")
        if strike is None or iv is None or expiry is None:
            continue
        if not is_finite(strike) or not is_finite(iv) or iv <= 0:
            continue
        group = by_expiry.setdefault(expiry, {})
        existing = group.get(strike)
        if existing is None:
            group[strike] = {"iv": float(iv), "right": right}
            continue
        if right not in ("call", "put") or existing.get("right") not in ("call", "put") or right == existing.get("right"):
            # Same right seen twice at one strike (shouldn't happen for a
            # real option chain, but defensive), or an unrecognized right --
            # average rather than guess.
            group[strike] = {"iv": (existing["iv"] + float(iv)) / 2.0, "right": right}
            continue
        if spot is not None and spot > 0:
            existing_otm = (existing["right"] == "call" and strike > spot) or (existing["right"] == "put" and strike < spot)
            new_otm = (right == "call" and strike > spot) or (right == "put" and strike < spot)
            if new_otm and not existing_otm:
                group[strike] = {"iv": float(iv), "right": right}
            elif new_otm == existing_otm:
                group[strike] = {"iv": (existing["iv"] + float(iv)) / 2.0, "right": right}
            # else: new leg is ITM while existing is already OTM -- keep existing
        else:
            group[strike] = {"iv": (existing["iv"] + float(iv)) / 2.0, "right": right}

    targets_by_expiry: dict = {}
    for t in targets or []:
        expiry = t.get("expiry")
        targets_by_expiry.setdefault(expiry, []).append(t)

    out: dict = {}
    for expiry, group in by_expiry.items():
        strikes_sorted = sorted(group.keys())
        if len(strikes_sorted) < 2:
            continue  # nothing to interpolate FROM in this expiry
        ivs_sorted = [group[k]["iv"] for k in strikes_sorted]
        for t in targets_by_expiry.get(expiry, []):
            sym = t.get("symbol")
            strike = t.get("strike")
            if not sym or strike is None or not is_finite(strike):
                continue
            if strike_lo is not None and strike < strike_lo:
                continue
            if strike_hi is not None and strike > strike_hi:
                continue
            iv = _interp_iv(strike, strikes_sorted, ivs_sorted)
            if iv is not None and is_finite(iv) and iv > 0:
                out[sym] = iv
    return out
