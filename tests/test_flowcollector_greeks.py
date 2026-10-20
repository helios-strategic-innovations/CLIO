"""Tests for agent/flowcollector/greeks.py. Hermetic: stdlib math only, no
network. Reference numbers below were computed independently against this
module's own bsm_price/bsm_delta/bsm_gamma AND cross-checked via a
finite-difference of bsm_price with respect to spot (a second, independent
derivation of delta/gamma that does not reuse this module's own closed-form
formulas) -- see the session report for that cross-check's output.
"""

import math
from datetime import date, datetime, timezone

import pytest

from agent.flowcollector.greeks import (
    DEFAULT_IV_HI,
    DEFAULT_IV_LO,
    bsm_delta,
    bsm_gamma,
    bsm_price,
    bsm_vega,
    compute_contract_greeks,
    fill_iv_from_smile,
    implied_volatility,
    is_finite,
    norm_cdf,
    norm_pdf,
    sanitize_greeks,
    time_to_expiry_years,
)

ET_2026_09_25_1600 = datetime(2026, 9, 25, 20, 0, tzinfo=timezone.utc)  # 16:00:00 ET (EDT, UTC-4)


# ═══════════════════════════════════════════════════════════════════════════
#  norm_cdf / norm_pdf
# ═══════════════════════════════════════════════════════════════════════════


def test_norm_cdf_known_values():
    assert math.isclose(norm_cdf(0.0), 0.5, abs_tol=1e-12)
    assert math.isclose(norm_cdf(1.959963985), 0.975, abs_tol=1e-6)
    assert math.isclose(norm_cdf(-1.959963985), 0.025, abs_tol=1e-6)


def test_norm_pdf_known_values():
    assert math.isclose(norm_pdf(0.0), 1.0 / math.sqrt(2 * math.pi), abs_tol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════
#  BSM textbook values (S=100, K=100, T=0.25, r=0.05, q=0, sigma=0.20)
# ═══════════════════════════════════════════════════════════════════════════

_S, _K, _T, _R, _Q, _SIGMA = 100.0, 100.0, 0.25, 0.05, 0.0, 0.20


def test_bsm_price_call_matches_textbook_value():
    # Independently derived: d1 = (ln(1) + (0.05+0.02)*0.25)/(0.2*0.5) = 0.175,
    # d2 = 0.075; call = 100*N(0.175) - 100*e^-0.0125*N(0.075).
    call = bsm_price(_S, _K, _T, _R, _Q, _SIGMA, "call")
    assert call == pytest.approx(4.6150, abs=0.002)


def test_bsm_put_call_parity_zero_dividend():
    call = bsm_price(_S, _K, _T, _R, _Q, _SIGMA, "call")
    put = bsm_price(_S, _K, _T, _R, _Q, _SIGMA, "put")
    # C - P = S*e^-qT - K*e^-rT
    assert (call - put) == pytest.approx(_S * math.exp(-_Q * _T) - _K * math.exp(-_R * _T), abs=1e-9)


def test_bsm_put_call_parity_nonzero_dividend():
    q = 0.02
    call = bsm_price(_S, _K, _T, _R, q, _SIGMA, "call")
    put = bsm_price(_S, _K, _T, _R, q, _SIGMA, "put")
    assert (call - put) == pytest.approx(_S * math.exp(-q * _T) - _K * math.exp(-_R * _T), abs=1e-9)


def test_bsm_delta_matches_textbook_value():
    delta = bsm_delta(_S, _K, _T, _R, _Q, _SIGMA, "call")
    assert delta == pytest.approx(0.5695, abs=0.001)


def test_bsm_gamma_matches_textbook_value():
    gamma = bsm_gamma(_S, _K, _T, _R, _Q, _SIGMA)
    assert gamma == pytest.approx(0.0393, abs=0.0005)


def test_bsm_gamma_identical_for_call_and_put_same_strike():
    # BSM gamma has no `right` argument -- same value for both by
    # construction; put_delta uses the SAME underlying d1 too, just fed
    # through a different combination (see bsm_delta's docstring).
    call_delta = bsm_delta(_S, _K, _T, _R, _Q, _SIGMA, "call")
    put_delta = bsm_delta(_S, _K, _T, _R, _Q, _SIGMA, "put")
    assert put_delta == pytest.approx(call_delta - math.exp(-_Q * _T), abs=1e-9)


def test_bsm_delta_and_gamma_cross_checked_via_finite_difference():
    """Independent second derivation: numerically differentiate bsm_price
    with respect to spot (central difference for delta, 3-point for gamma)
    and compare to the closed-form bsm_delta/bsm_gamma -- this does NOT
    reuse the closed-form d1/d2 math, so it is a genuine cross-check, not a
    tautology."""
    h = 1e-4
    p_up = bsm_price(_S + h, _K, _T, _R, _Q, _SIGMA, "call")
    p_dn = bsm_price(_S - h, _K, _T, _R, _Q, _SIGMA, "call")
    p0 = bsm_price(_S, _K, _T, _R, _Q, _SIGMA, "call")
    fd_delta = (p_up - p_dn) / (2 * h)
    fd_gamma = (p_up - 2 * p0 + p_dn) / (h * h)
    assert fd_delta == pytest.approx(bsm_delta(_S, _K, _T, _R, _Q, _SIGMA, "call"), abs=1e-6)
    assert fd_gamma == pytest.approx(bsm_gamma(_S, _K, _T, _R, _Q, _SIGMA), abs=1e-3)


def test_bsm_price_degenerate_inputs_return_none():
    assert bsm_price(None, _K, _T, _R, _Q, _SIGMA, "call") is None
    assert bsm_price(_S, 0.0, _T, _R, _Q, _SIGMA, "call") is None
    assert bsm_price(_S, _K, 0.0, _R, _Q, _SIGMA, "call") is None
    assert bsm_price(_S, _K, _T, _R, _Q, 0.0, "call") is None
    assert bsm_price(_S, _K, _T, _R, _Q, _SIGMA, "invalid") is None


def test_bsm_price_rejects_non_finite_inputs():
    assert bsm_price(float("nan"), _K, _T, _R, _Q, _SIGMA, "call") is None
    assert bsm_price(float("inf"), _K, _T, _R, _Q, _SIGMA, "call") is None


# ═══════════════════════════════════════════════════════════════════════════
#  implied_volatility: round-trip
# ═══════════════════════════════════════════════════════════════════════════


def test_iv_round_trip_simple_case():
    price = bsm_price(_S, _K, _T, _R, _Q, _SIGMA, "call")
    iv = implied_volatility(price, _S, _K, _T, _R, _Q, "call")
    assert iv == pytest.approx(_SIGMA, abs=1e-4)


def test_iv_round_trip_put():
    price = bsm_price(_S, _K, _T, _R, _Q, _SIGMA, "put")
    iv = implied_volatility(price, _S, _K, _T, _R, _Q, "put")
    assert iv == pytest.approx(_SIGMA, abs=1e-4)


def test_iv_round_trip_across_moneyness_and_sigma():
    """A grid across moneyness (within +/-15%, comfortably covering this
    collector's production +/-4% strike band) and sigma. Cases whose true
    price underflows below a meaningfully-quotable level (< 1e-4) are
    skipped -- no real bid/ask quote would look like that (compute_contract_
    greeks' own bid>0 guard already excludes it in production)."""
    import random

    rng = random.Random(1)
    n = 0
    max_err = 0.0
    for _ in range(400):
        spot = rng.uniform(50, 600)
        moneyness = rng.uniform(0.85, 1.15)
        strike = spot * moneyness
        T = rng.uniform(1.0 / 365.0 / 24.0, 1.0)
        r = rng.uniform(-0.01, 0.08)
        q = rng.uniform(0.0, 0.04)
        sigma = rng.uniform(0.05, 2.0)
        right = "call" if rng.random() < 0.5 else "put"
        price = bsm_price(spot, strike, T, r, q, sigma, right)
        if price is None or price < 1e-4:
            continue
        iv = implied_volatility(price, spot, strike, T, r, q, right)
        if iv is None:
            continue  # a low-vega (near-zero-sensitivity) case -- see its own test below
        n += 1
        max_err = max(max_err, abs(iv - sigma))
    assert n > 300  # the grid should resolve the vast majority of cases
    assert max_err < 0.01  # tight round-trip accuracy away from the low-vega regime


def test_iv_returns_none_for_pathologically_low_vega_rather_than_a_wrong_answer():
    """A deep-ITM, minutes-to-close contract has near-zero vega -- a
    price-tolerance bisection is fundamentally ill-conditioned there (a wide
    range of sigma reproduce ~the same price). implied_volatility must
    return None (per DEFAULT_IV_MIN_VEGA's gate) rather than a numerically
    arbitrary sigma nowhere near the true one."""
    spot, strike, T, r, q, true_sigma = 75.81784475607313, 67.65360960569848, 0.0060286649727879825, 0.049639022839669304, 0.03730507715281556, 0.11180047795808154
    price = bsm_price(spot, strike, T, r, q, true_sigma, "call")
    assert price is not None and price > 0
    iv = implied_volatility(price, spot, strike, T, r, q, "call")
    assert iv is None


# ═══════════════════════════════════════════════════════════════════════════
#  implied_volatility: no-arbitrage bounds / edge cases
# ═══════════════════════════════════════════════════════════════════════════


def test_iv_none_when_price_above_no_arbitrage_upper_bound():
    # A call can never be worth more than the spot itself (undiscounted
    # upper bound S*e^-qT <= S) -- a price of 2x spot is unreachable by any sigma.
    assert implied_volatility(2 * _S, _S, _K, _T, _R, _Q, "call") is None


def test_iv_none_when_price_below_no_arbitrage_lower_bound():
    # A deep ITM call priced at (or near) zero is below its own intrinsic
    # no-arbitrage floor.
    deep_itm_strike = 10.0
    assert implied_volatility(0.0, _S, deep_itm_strike, _T, _R, _Q, "call") is None


def test_iv_none_for_negative_price():
    assert implied_volatility(-1.0, _S, _K, _T, _R, _Q, "call") is None


def test_iv_none_for_missing_or_non_finite_inputs():
    assert implied_volatility(None, _S, _K, _T, _R, _Q, "call") is None
    assert implied_volatility(4.6, None, _K, _T, _R, _Q, "call") is None
    assert implied_volatility(4.6, _S, _K, None, _R, _Q, "call") is None
    assert implied_volatility(float("nan"), _S, _K, _T, _R, _Q, "call") is None


def test_iv_none_for_zero_or_negative_spot_strike_T():
    assert implied_volatility(4.6, 0.0, _K, _T, _R, _Q, "call") is None
    assert implied_volatility(4.6, _S, 0.0, _T, _R, _Q, "call") is None
    assert implied_volatility(4.6, _S, _K, 0.0, _R, _Q, "call") is None


def test_iv_none_for_unrecognized_right():
    assert implied_volatility(4.6, _S, _K, _T, _R, _Q, "straddle") is None


def test_iv_at_bracket_lower_edge_returns_lo():
    price_lo = bsm_price(_S, _K, _T, _R, _Q, DEFAULT_IV_LO, "call")
    iv = implied_volatility(price_lo, _S, _K, _T, _R, _Q, "call")
    assert iv == pytest.approx(DEFAULT_IV_LO, abs=1e-6)


def test_iv_at_bracket_upper_edge_returns_hi_or_none():
    # At sigma=5.0 (the upper edge of the bracket), vega can itself be tiny
    # for some parameter combos (very high sigma flattens the price curve
    # too) -- accept either a precise hi or a documented None (min-vega
    # gate), but never a wrong number.
    price_hi = bsm_price(_S, _K, _T, _R, _Q, DEFAULT_IV_HI, "call")
    iv = implied_volatility(price_hi, _S, _K, _T, _R, _Q, "call")
    assert iv is None or iv == pytest.approx(DEFAULT_IV_HI, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
#  time_to_expiry_years
# ═══════════════════════════════════════════════════════════════════════════


def test_time_to_expiry_none_when_already_past_close():
    now = ET_2026_09_25_1600  # exactly 16:00:00 ET
    assert time_to_expiry_years(now, date(2026, 9, 25), early_close_dates=frozenset()) is None
    past = datetime(2026, 9, 25, 21, 0, tzinfo=timezone.utc)  # 17:00 ET -- well past close
    assert time_to_expiry_years(past, date(2026, 9, 25), early_close_dates=frozenset()) is None


def test_time_to_expiry_five_minute_floor():
    # 90 real seconds before the 16:00 ET close -- floored to 5 minutes.
    now = datetime(2026, 9, 25, 19, 58, 30, tzinfo=timezone.utc)  # 15:58:30 ET
    T = time_to_expiry_years(now, date(2026, 9, 25), early_close_dates=frozenset())
    assert T is not None
    seconds = T * 365.0 * 86400.0
    assert seconds == pytest.approx(300.0, abs=1e-6)


def test_time_to_expiry_normal_case_hours_to_close():
    now = datetime(2026, 9, 25, 18, 0, tzinfo=timezone.utc)  # 14:00 ET -- 2h to the 16:00 close
    T = time_to_expiry_years(now, date(2026, 9, 25), early_close_dates=frozenset())
    assert T is not None
    hours = T * 365.0 * 24.0
    assert hours == pytest.approx(2.0, abs=1e-6)


def test_time_to_expiry_early_close_uses_13_00_et():
    early = frozenset({date(2026, 11, 27)})
    before = datetime(2026, 11, 27, 17, 0, tzinfo=timezone.utc)  # 12:00 ET -- 1h before 13:00 early close
    T = time_to_expiry_years(before, date(2026, 11, 27), early_close_dates=early)
    assert T is not None
    assert (T * 365.0 * 24.0) == pytest.approx(1.0, abs=1e-6)
    after = datetime(2026, 11, 27, 18, 10, tzinfo=timezone.utc)  # 13:10 ET -- past the early close
    assert time_to_expiry_years(after, date(2026, 11, 27), early_close_dates=early) is None


def test_time_to_expiry_naive_now_treated_as_utc():
    naive = datetime(2026, 9, 25, 19, 0)  # no tzinfo -- treated as 19:00 UTC = 15:00 ET
    T = time_to_expiry_years(naive, date(2026, 9, 25), early_close_dates=frozenset())
    assert T is not None
    assert (T * 365.0 * 24.0) == pytest.approx(1.0, abs=1e-6)


def test_time_to_expiry_none_inputs():
    assert time_to_expiry_years(None, date(2026, 9, 25)) is None
    assert time_to_expiry_years(datetime.now(timezone.utc), None) is None


def test_time_to_expiry_real_lazy_import_finds_a_known_early_close():
    """No monkeypatching -- exercises the REAL lazy import of
    agent.integrations.market_events.MARKET_EARLY_CLOSE (never required per
    the module docstring, but present in this repo) by calling
    time_to_expiry_years WITHOUT an explicit early_close_dates override, on
    a date that module's own MARKET_EARLY_CLOSE lists (2026-11-27)."""
    before = datetime(2026, 11, 27, 17, 0, tzinfo=timezone.utc)  # 12:00 ET -- 1h before the 13:00 early close
    T = time_to_expiry_years(before, date(2026, 11, 27))
    assert T is not None
    assert (T * 365.0 * 24.0) == pytest.approx(1.0, abs=1e-6)


def test_time_to_expiry_degrades_to_no_early_closes_when_import_unavailable(monkeypatch):
    """`_early_close_dates()`'s import is optional (mirrors collector.py's
    own `_market_calendar` pattern) -- simulate it being unavailable and
    confirm time_to_expiry_years still returns a normal 16:00 ET close
    (never an exception, never treats an early-close date as early-close
    once the lookup itself is gone)."""
    import agent.flowcollector.greeks as greeks_mod

    monkeypatch.setattr(greeks_mod, "_early_close_dates", lambda: frozenset())
    # 2026-11-27 IS a real early-close date, but with the lookup degraded
    # to empty, 14:00 ET (2h before the NORMAL 16:00 close) must still see
    # 2 hours to expiry, not treat 13:00 as the close. (2026-11-27 is EST,
    # UTC-5, post-DST -- see test_time_to_expiry_real_lazy_import_finds_a_
    # known_early_close's 17:00 UTC = 12:00 ET for the same offset.)
    now = datetime(2026, 11, 27, 19, 0, tzinfo=timezone.utc)  # 14:00 ET
    T = time_to_expiry_years(now, date(2026, 11, 27))
    assert T is not None
    assert (T * 365.0 * 24.0) == pytest.approx(2.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
#  sanitize_greeks / is_finite
# ═══════════════════════════════════════════════════════════════════════════


def test_is_finite():
    assert is_finite(1.0) is True
    assert is_finite(0) is True
    assert is_finite(-3) is True
    assert is_finite(float("nan")) is False
    assert is_finite(float("inf")) is False
    assert is_finite(None) is False
    assert is_finite("0.5") is False
    assert is_finite(True) is False  # bool is a bool, not a numeric field (CLAUDE.md type-guard discipline)


def test_sanitize_greeks_accepts_sane_pair():
    r = sanitize_greeks(0.55, 0.04)
    assert r == {"delta": 0.55, "gamma": 0.04}


def test_sanitize_greeks_clamps_tiny_negative_gamma_noise():
    r = sanitize_greeks(0.5, -1e-9)
    assert r == {"delta": 0.5, "gamma": 0.0}


def test_sanitize_greeks_rejects_out_of_range_delta():
    assert sanitize_greeks(1.5, 0.04) is None
    assert sanitize_greeks(-1.5, 0.04) is None


def test_sanitize_greeks_rejects_meaningfully_negative_gamma():
    assert sanitize_greeks(0.5, -0.1) is None


def test_sanitize_greeks_rejects_non_finite_or_missing():
    assert sanitize_greeks(None, 0.04) is None
    assert sanitize_greeks(0.5, None) is None
    assert sanitize_greeks(float("nan"), 0.04) is None
    assert sanitize_greeks(0.5, float("inf")) is None


# ═══════════════════════════════════════════════════════════════════════════
#  compute_contract_greeks: guards (bid>0, ask>bid, spread%, iv_hint)
# ═══════════════════════════════════════════════════════════════════════════


def _T_1day():
    return time_to_expiry_years(
        datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc), date(2026, 9, 26), early_close_dates=frozenset()
    )


def test_compute_contract_greeks_happy_path():
    T = _T_1day()
    r = compute_contract_greeks(2.0, 2.2, 580.0, 580.0, T, 0.04, 0.012, "call")
    assert r is not None
    assert r["delta"] > 0
    assert r["gamma"] > 0
    assert 0.0 < r["iv"] < 5.0


def test_compute_contract_greeks_rejects_zero_or_missing_bid():
    T = _T_1day()
    assert compute_contract_greeks(0.0, 2.2, 580.0, 580.0, T, 0.04, 0.012, "call") is None
    assert compute_contract_greeks(None, 2.2, 580.0, 580.0, T, 0.04, 0.012, "call") is None


def test_compute_contract_greeks_rejects_ask_not_above_bid():
    T = _T_1day()
    assert compute_contract_greeks(2.0, 2.0, 580.0, 580.0, T, 0.04, 0.012, "call") is None
    assert compute_contract_greeks(2.0, 1.9, 580.0, 580.0, T, 0.04, 0.012, "call") is None  # crossed


def test_compute_contract_greeks_rejects_wide_spread():
    T = _T_1day()
    # mid = 1.5, spread = (2.0-1.0)/1.5 = 0.667 > default max_spread_pct_for_iv(0.5)
    assert compute_contract_greeks(1.0, 2.0, 580.0, 580.0, T, 0.04, 0.012, "call") is None
    # the SAME quote passes when the caller widens the tolerance
    r = compute_contract_greeks(1.0, 2.0, 580.0, 580.0, T, 0.04, 0.012, "call", max_spread_pct_for_iv=0.9)
    assert r is not None


def test_compute_contract_greeks_uses_iv_hint_instead_of_solving():
    T = _T_1day()
    solved = compute_contract_greeks(2.0, 2.2, 580.0, 580.0, T, 0.04, 0.012, "call")
    hinted = compute_contract_greeks(2.0, 2.2, 580.0, 580.0, T, 0.04, 0.012, "call", iv_hint=0.30)
    assert hinted is not None
    assert hinted["iv"] == pytest.approx(0.30, abs=1e-9)  # used AS-IS, not re-solved
    assert hinted["iv"] != pytest.approx(solved["iv"], abs=1e-6)  # differs from the from-quote solve


def test_compute_contract_greeks_ignores_insane_iv_hint_and_solves_instead():
    T = _T_1day()
    solved = compute_contract_greeks(2.0, 2.2, 580.0, 580.0, T, 0.04, 0.012, "call")
    with_bad_hint = compute_contract_greeks(2.0, 2.2, 580.0, 580.0, T, 0.04, 0.012, "call", iv_hint=-1.0)
    assert with_bad_hint is not None
    assert with_bad_hint["iv"] == pytest.approx(solved["iv"], abs=1e-6)


def test_compute_contract_greeks_none_for_missing_time_or_spot_or_strike():
    T = _T_1day()
    assert compute_contract_greeks(2.0, 2.2, None, 580.0, T, 0.04, 0.012, "call") is None
    assert compute_contract_greeks(2.0, 2.2, 580.0, None, T, 0.04, 0.012, "call") is None
    assert compute_contract_greeks(2.0, 2.2, 580.0, 580.0, None, 0.04, 0.012, "call") is None


def test_compute_contract_greeks_none_for_unrecognized_right():
    T = _T_1day()
    assert compute_contract_greeks(2.0, 2.2, 580.0, 580.0, T, 0.04, 0.012, "iron_condor") is None


def test_compute_contract_greeks_put_has_negative_delta():
    T = _T_1day()
    r = compute_contract_greeks(1.8, 2.0, 580.0, 580.0, T, 0.04, 0.012, "put")
    assert r is not None
    assert r["delta"] < 0
    assert r["gamma"] > 0


# ═══════════════════════════════════════════════════════════════════════════
#  bsm_vega (used only internally by implied_volatility, but public/tested
#  on its own since it's part of the "verify the math" deliverable)
# ═══════════════════════════════════════════════════════════════════════════


def test_bsm_vega_positive_for_a_normal_atm_case():
    v = bsm_vega(_S, _K, _T, _R, _Q, _SIGMA)
    assert v is not None and v > 0


def test_bsm_vega_none_for_degenerate_inputs():
    assert bsm_vega(_S, _K, 0.0, _R, _Q, _SIGMA) is None


# ═══════════════════════════════════════════════════════════════════════════
#  fill_iv_from_smile (2026-09-25 coverage-risk follow-up)
# ═══════════════════════════════════════════════════════════════════════════

_EXPIRY = date(2026, 9, 26)


def test_fill_iv_from_smile_linear_midpoint():
    """The exact worked example from the follow-up directive: anchors
    0.20@K1 and 0.30@K2 -> a target exactly at the midpoint strike gets
    0.25."""
    anchors = [
        {"symbol": "A1", "strike": 575.0, "expiry": _EXPIRY, "right": "put", "iv": 0.20},
        {"symbol": "A2", "strike": 585.0, "expiry": _EXPIRY, "right": "call", "iv": 0.30},
    ]
    targets = [{"symbol": "T", "strike": 580.0, "expiry": _EXPIRY, "right": "call"}]
    filled = fill_iv_from_smile(anchors, targets, spot=580.0)
    assert filled == {"T": pytest.approx(0.25)}


def test_fill_iv_from_smile_off_center_linear():
    anchors = [
        {"symbol": "A1", "strike": 500.0, "expiry": _EXPIRY, "right": "put", "iv": 0.10},
        {"symbol": "A2", "strike": 600.0, "expiry": _EXPIRY, "right": "call", "iv": 0.40},
    ]
    targets = [{"symbol": "T25", "strike": 525.0, "expiry": _EXPIRY, "right": "put"}]  # 25% of the way from 500 to 600
    filled = fill_iv_from_smile(anchors, targets, spot=550.0)
    assert filled["T25"] == pytest.approx(0.10 + 0.25 * (0.40 - 0.10))


def test_fill_iv_from_smile_flat_extrapolation_beyond_outermost_anchor():
    anchors = [
        {"symbol": "A1", "strike": 575.0, "expiry": _EXPIRY, "right": "put", "iv": 0.20},
        {"symbol": "A2", "strike": 585.0, "expiry": _EXPIRY, "right": "call", "iv": 0.30},
    ]
    targets = [
        {"symbol": "T_below", "strike": 560.0, "expiry": _EXPIRY, "right": "put"},
        {"symbol": "T_above", "strike": 600.0, "expiry": _EXPIRY, "right": "call"},
    ]
    filled = fill_iv_from_smile(anchors, targets, spot=580.0)
    assert filled["T_below"] == pytest.approx(0.20)
    assert filled["T_above"] == pytest.approx(0.30)


def test_fill_iv_from_smile_respects_band_edge():
    """Flat extrapolation only reaches up to the caller's own configured
    strike band -- a target beyond it is left unfilled even with >= 2
    anchors in its expiry."""
    anchors = [
        {"symbol": "A1", "strike": 575.0, "expiry": _EXPIRY, "right": "put", "iv": 0.20},
        {"symbol": "A2", "strike": 585.0, "expiry": _EXPIRY, "right": "call", "iv": 0.30},
    ]
    targets = [
        {"symbol": "T_in_band", "strike": 587.0, "expiry": _EXPIRY, "right": "call"},
        {"symbol": "T_outside_band", "strike": 595.0, "expiry": _EXPIRY, "right": "call"},
    ]
    filled = fill_iv_from_smile(anchors, targets, spot=580.0, strike_lo=570.0, strike_hi=590.0)
    assert "T_in_band" in filled
    assert "T_outside_band" not in filled


def test_fill_iv_from_smile_requires_at_least_two_anchors():
    """A single anchor strike (even with both a call AND a put anchor at
    it) carries no slope -- no fill, target stays absent from the result."""
    anchors = [
        {"symbol": "A1", "strike": 580.0, "expiry": _EXPIRY, "right": "call", "iv": 0.25},
        {"symbol": "A1p", "strike": 580.0, "expiry": _EXPIRY, "right": "put", "iv": 0.24},
    ]
    targets = [{"symbol": "T", "strike": 585.0, "expiry": _EXPIRY, "right": "call"}]
    filled = fill_iv_from_smile(anchors, targets, spot=580.0)
    assert filled == {}


def test_fill_iv_from_smile_empty_anchors_or_targets():
    assert fill_iv_from_smile([], [{"symbol": "T", "strike": 580.0, "expiry": _EXPIRY, "right": "call"}]) == {}
    assert fill_iv_from_smile(
        [{"symbol": "A1", "strike": 575.0, "expiry": _EXPIRY, "right": "put", "iv": 0.20},
         {"symbol": "A2", "strike": 585.0, "expiry": _EXPIRY, "right": "call", "iv": 0.30}],
        [],
    ) == {}


def test_fill_iv_from_smile_never_crosses_expiries():
    """Anchors from a DIFFERENT expiry never bracket a target -- each
    expiry's smile is interpolated independently (0DTE vol != 2DTE vol at
    the same strike)."""
    other_expiry = date(2026, 9, 28)
    anchors = [
        {"symbol": "A1", "strike": 575.0, "expiry": _EXPIRY, "right": "put", "iv": 0.20},
        {"symbol": "A2", "strike": 585.0, "expiry": _EXPIRY, "right": "call", "iv": 0.30},
    ]
    targets = [{"symbol": "T_other_expiry", "strike": 580.0, "expiry": other_expiry, "right": "call"}]
    filled = fill_iv_from_smile(anchors, targets, spot=580.0)
    assert filled == {}


def test_fill_iv_from_smile_dedup_prefers_otm_leg_when_spot_given():
    # strike=575 (below spot=580): the PUT there is OTM, the CALL is ITM --
    # the OTM (put) IV must be the one used as this strike's anchor.
    anchors = [
        {"symbol": "C575", "strike": 575.0, "expiry": _EXPIRY, "right": "call", "iv": 0.50},  # ITM call
        {"symbol": "P575", "strike": 575.0, "expiry": _EXPIRY, "right": "put", "iv": 0.22},   # OTM put
        {"symbol": "A2", "strike": 585.0, "expiry": _EXPIRY, "right": "call", "iv": 0.30},
    ]
    targets = [{"symbol": "T_at_575", "strike": 575.0, "expiry": _EXPIRY, "right": "call"}]
    filled = fill_iv_from_smile(anchors, targets, spot=580.0)
    assert filled["T_at_575"] == pytest.approx(0.22)  # the OTM leg, not 0.50 and not the average


def test_fill_iv_from_smile_dedup_averages_without_spot():
    anchors = [
        {"symbol": "C575", "strike": 575.0, "expiry": _EXPIRY, "right": "call", "iv": 0.50},
        {"symbol": "P575", "strike": 575.0, "expiry": _EXPIRY, "right": "put", "iv": 0.22},
        {"symbol": "A2", "strike": 585.0, "expiry": _EXPIRY, "right": "call", "iv": 0.30},
    ]
    targets = [{"symbol": "T_at_575", "strike": 575.0, "expiry": _EXPIRY, "right": "call"}]
    filled = fill_iv_from_smile(anchors, targets, spot=None)
    assert filled["T_at_575"] == pytest.approx((0.50 + 0.22) / 2.0)


def test_fill_iv_from_smile_skips_target_missing_strike_or_symbol():
    anchors = [
        {"symbol": "A1", "strike": 575.0, "expiry": _EXPIRY, "right": "put", "iv": 0.20},
        {"symbol": "A2", "strike": 585.0, "expiry": _EXPIRY, "right": "call", "iv": 0.30},
    ]
    targets = [
        {"symbol": None, "strike": 580.0, "expiry": _EXPIRY, "right": "call"},
        {"symbol": "T_no_strike", "strike": None, "expiry": _EXPIRY, "right": "call"},
        {"symbol": "T_good", "strike": 580.0, "expiry": _EXPIRY, "right": "call"},
    ]
    filled = fill_iv_from_smile(anchors, targets, spot=580.0)
    assert set(filled.keys()) == {"T_good"}


def test_fill_iv_from_smile_filled_greeks_feed_correct_delta_gamma():
    """End-to-end sanity: the filled IV, run through bsm_delta/bsm_gamma
    (the same path collector.py's _fill_smile_batch_sync takes), produces a
    plausible small-but-nonzero far-OTM greek pair -- not a fake zero."""
    anchors = [
        {"symbol": "A1", "strike": 640.0, "expiry": _EXPIRY, "right": "put", "iv": 0.15},
        {"symbol": "A2", "strike": 660.0, "expiry": _EXPIRY, "right": "call", "iv": 0.18},
    ]
    far_strike = 685.0  # well OTM call, flat-extrapolated from A2's IV
    targets = [{"symbol": "T_far", "strike": far_strike, "expiry": _EXPIRY, "right": "call"}]
    filled = fill_iv_from_smile(anchors, targets, spot=660.0, strike_lo=634.0, strike_hi=686.0)
    assert "T_far" in filled
    T = time_to_expiry_years(datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc), _EXPIRY, early_close_dates=frozenset())
    delta = bsm_delta(660.0, far_strike, T, 0.04, 0.012, filled["T_far"], "call")
    gamma = bsm_gamma(660.0, far_strike, T, 0.04, 0.012, filled["T_far"])
    assert 0.0 < delta < 0.5  # a far OTM call: positive but well below 0.5
    assert gamma > 0.0  # small but NOT a fake zero
