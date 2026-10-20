"""Tests for agent/dpof/contracts.py."""

from dataclasses import replace
from datetime import date, timedelta

from agent.dpof.contracts import (
    choose_candidates,
    dte,
    expiry_fetch_window,
    option_right,
    target_expiries,
)
from agent.dpof.models import Direction, OptionQuote

TODAY = date(2026, 9, 24)  # a Thursday


def occ_symbol(underlying: str, expiration: date, right: str, strike: float) -> str:
    letter = "C" if right == "call" else "P"
    return f"{underlying}{expiration.strftime('%y%m%d')}{letter}{int(round(strike * 1000)):08d}"


def mk_quote(
    underlying="SPY",
    expiration=None,
    strike=575.0,
    right="call",
    bid=5.00,
    ask=5.20,
    delta=0.50,
    theta=-0.20,
    open_interest=1000,
    iv=0.18,
    volume=100,
) -> OptionQuote:
    exp = expiration or (TODAY + timedelta(days=10))
    return OptionQuote(
        symbol=occ_symbol(underlying, exp, right, strike),
        underlying=underlying,
        expiration=exp.isoformat(),
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        delta=delta,
        theta=theta,
        iv=iv,
        open_interest=open_interest,
        volume=volume,
    )


# ── dte / option_right ──────────────────────────────────────────────────────


def test_dte_basic():
    exp = TODAY + timedelta(days=10)
    assert dte(exp.isoformat(), TODAY) == 10


def test_dte_zero_and_negative():
    assert dte(TODAY.isoformat(), TODAY) == 0
    past = TODAY - timedelta(days=3)
    assert dte(past.isoformat(), TODAY) == -3


def test_option_right_bullish_bearish():
    assert option_right(Direction.BULLISH) == "call"
    assert option_right(Direction.BEARISH) == "put"


def test_option_right_neutral_defaults_to_put():
    # NEUTRAL is never passed into contract selection by the engine; documented
    # defensive mapping so the function stays total.
    assert option_right(Direction.NEUTRAL) == "put"


# ── target_expiries ──────────────────────────────────────────────────────


def test_target_expiries_mon_thu_is_next_calendar_day(cfg):
    monday = date(2026, 9, 21)
    c = replace(cfg, expiry_sessions_ahead=1, max_expiry_sessions_ahead=1)
    assert target_expiries(monday, c) == [date(2026, 9, 22)]

    thursday = date(2026, 9, 24)
    assert target_expiries(thursday, c) == [date(2026, 9, 25)]


def test_target_expiries_friday_skips_weekend_to_monday(cfg):
    friday = date(2026, 9, 25)
    c = replace(cfg, expiry_sessions_ahead=1, max_expiry_sessions_ahead=1)
    assert target_expiries(friday, c) == [date(2026, 9, 28)]


def test_target_expiries_skips_holiday():
    # Labor Day 2026 is Monday 2026-09-07 (MARKET_HOLIDAYS). Friday before it
    # (2026-09-04) must skip Sat/Sun AND the Monday holiday -> Tuesday.
    friday_before_labor_day = date(2026, 9, 4)
    c = replace(_DPOFConfig(), expiry_sessions_ahead=1, max_expiry_sessions_ahead=1)
    assert target_expiries(friday_before_labor_day, c) == [date(2026, 9, 8)]


def test_target_expiries_day_before_holiday():
    # Thanksgiving 2026 is Thursday 2026-11-26 -- a mid-week holiday, so the
    # day immediately before it (Wednesday) isolates pure holiday-skipping
    # from any weekend-skipping (unlike a Friday-before-a-Monday-holiday case).
    wednesday_before_thanksgiving = date(2026, 11, 25)
    c = replace(_DPOFConfig(), expiry_sessions_ahead=1, max_expiry_sessions_ahead=1)
    assert target_expiries(wednesday_before_thanksgiving, c) == [date(2026, 11, 27)]


def test_target_expiries_max_sessions_multiple_entries(cfg):
    monday = date(2026, 9, 21)
    c = replace(cfg, expiry_sessions_ahead=1, max_expiry_sessions_ahead=3)
    assert target_expiries(monday, c) == [date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]


def test_target_expiries_starts_ahead_of_session_1():
    # expiry_sessions_ahead=2 skips the immediate next session entirely.
    monday = date(2026, 9, 21)
    c = replace(_DPOFConfig(), expiry_sessions_ahead=2, max_expiry_sessions_ahead=2)
    assert target_expiries(monday, c) == [date(2026, 9, 23)]


def test_target_expiries_falls_back_to_weekends_only_if_market_events_unimportable(cfg, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "agent.integrations.market_events":
            raise ImportError("simulated unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    friday_before_labor_day = date(2026, 9, 4)
    c = replace(cfg, expiry_sessions_ahead=1, max_expiry_sessions_ahead=1)
    # Without MARKET_HOLIDAYS, the Monday holiday is NOT skipped -> next
    # session is simply the next weekday (Monday), not Tuesday.
    assert target_expiries(friday_before_labor_day, c) == [date(2026, 9, 7)]


def test_expiry_fetch_window_spans_first_and_last_target_expiry(cfg):
    monday = date(2026, 9, 21)
    c = replace(cfg, expiry_sessions_ahead=1, max_expiry_sessions_ahead=3)
    lo, hi = expiry_fetch_window(monday, c)
    assert lo == date(2026, 9, 22)
    assert hi == date(2026, 9, 24)


# ── choose_candidates ───────────────────────────────────────────────────


def _DPOFConfig():
    from agent.dpof.config import DPOFConfig
    return DPOFConfig()


TARGET_EXP = date(2026, 9, 25)     # TODAY (Thu 2026-09-24) + 1 session
FALLBACK_EXP = date(2026, 9, 28)   # + 2 sessions (skips the weekend)


def cq_quote(
    underlying="SPY",
    expiration=TARGET_EXP,
    strike=575.0,
    right="call",
    bid=5.00,
    ask=5.20,
    delta=0.50,
    open_interest=1000,
) -> OptionQuote:
    return OptionQuote(
        symbol=occ_symbol(underlying, expiration, right, strike),
        underlying=underlying,
        expiration=expiration.isoformat(),
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        delta=delta,
        theta=-0.20,
        open_interest=open_interest,
        iv=0.18,
        volume=100,
    )


def test_choose_candidates_delta_spread_picks_closest_and_neighbors(cfg):
    c = replace(cfg, entry_candidates=3, target_delta=0.50, min_delta=0.20, max_delta=0.80, allow_verticals=False)
    far_low = cq_quote(strike=560.0, delta=0.30)
    near = cq_quote(strike=575.0, delta=0.50)   # exactly target_delta
    far_high = cq_quote(strike=590.0, delta=0.70)
    chain = [far_low, near, far_high]
    out = choose_candidates(chain, underlying="SPY", direction=Direction.BULLISH, spot=575.0, today=TODAY, cfg=c)
    assert len(out) == 3
    assert all(o.structure == "single" for o in out)
    picked_symbols = {leg.symbol for o in out for leg in o.legs}
    assert picked_symbols == {far_low.symbol, near.symbol, far_high.symbol}
    # closest-to-target is always first.
    assert out[0].legs[0].symbol == near.symbol
    assert out[0].expiration == TARGET_EXP.isoformat()


def test_choose_candidates_fallback_expiry_when_target_has_nothing(cfg):
    c = replace(cfg, entry_candidates=3)
    # No candidates at all for TARGET_EXP; one liquid quote at FALLBACK_EXP.
    q = cq_quote(expiration=FALLBACK_EXP, strike=575.0, delta=0.50)
    out = choose_candidates([q], underlying="SPY", direction=Direction.BULLISH, spot=575.0, today=TODAY, cfg=c)
    assert len(out) == 1
    assert out[0].expiration == FALLBACK_EXP.isoformat()
    assert out[0].legs[0].symbol == q.symbol


def test_choose_candidates_no_greeks_falls_back_to_strike_distance(cfg):
    c = replace(cfg, entry_candidates=3)
    spot = 575.0
    atm = cq_quote(strike=575.0, delta=None)               # 0% from spot
    near = cq_quote(strike=580.0, delta=None, open_interest=900)  # ~0.87% from spot
    far = cq_quote(strike=650.0, delta=None, open_interest=900)   # >3% from spot -> excluded
    out = choose_candidates([atm, near, far], underlying="SPY", direction=Direction.BULLISH,
                             spot=spot, today=TODAY, cfg=c)
    picked_symbols = {leg.symbol for o in out for leg in o.legs}
    assert picked_symbols == {atm.symbol, near.symbol}
    assert far.symbol not in picked_symbols
    # ATM (closest to spot) ranks first.
    assert out[0].legs[0].symbol == atm.symbol


def test_choose_candidates_liquidity_filters_spread_and_oi(cfg):
    c = replace(cfg, entry_candidates=3, max_spread_pct=0.06, min_open_interest=500)
    good = cq_quote(strike=575.0, delta=0.50, bid=4.90, ask=5.10, open_interest=1000)
    wide_spread = cq_quote(strike=576.0, delta=0.49, bid=4.00, ask=6.00, open_interest=1000)  # too wide
    thin_oi = cq_quote(strike=574.0, delta=0.51, open_interest=100)  # below min_open_interest
    out = choose_candidates([good, wide_spread, thin_oi], underlying="SPY", direction=Direction.BULLISH,
                             spot=575.0, today=TODAY, cfg=c)
    picked_symbols = {leg.symbol for o in out for leg in o.legs}
    assert picked_symbols == {good.symbol}


def test_choose_candidates_includes_vertical_when_allowed(cfg):
    c = replace(cfg, entry_candidates=2, allow_verticals=True)
    long_q = cq_quote(strike=575.0, delta=0.50, open_interest=1000)
    short_q = cq_quote(strike=580.0, delta=0.15, bid=1.45, ask=1.55, open_interest=1000)
    out = choose_candidates([long_q, short_q], underlying="SPY", direction=Direction.BULLISH,
                             spot=575.0, today=TODAY, cfg=c)
    structures = [o.structure for o in out]
    assert "debit_vertical" in structures
    assert len(out) <= c.entry_candidates
    vertical = next(o for o in out if o.structure == "debit_vertical")
    assert vertical.legs[0].symbol == long_q.symbol
    assert vertical.legs[1].symbol == short_q.symbol


def test_choose_candidates_vertical_reserves_a_slot_over_extra_single(cfg):
    # entry_candidates=1 with 2 good singles + a buildable vertical -> the
    # vertical wins the only slot (never dropped for an extra single).
    c = replace(cfg, entry_candidates=1, allow_verticals=True, target_delta=0.50,
                min_delta=0.10, max_delta=0.90)
    long_q = cq_quote(strike=575.0, delta=0.50, open_interest=1000)
    other_single = cq_quote(strike=560.0, delta=0.30, open_interest=1000)
    short_q = cq_quote(strike=580.0, delta=0.15, bid=1.45, ask=1.55, open_interest=1000)
    out = choose_candidates([long_q, other_single, short_q], underlying="SPY", direction=Direction.BULLISH,
                             spot=575.0, today=TODAY, cfg=c)
    assert len(out) == 1
    assert out[0].structure == "debit_vertical"


def test_choose_candidates_empty_chain_returns_empty_list(cfg):
    out = choose_candidates([], underlying="SPY", direction=Direction.BULLISH, spot=575.0, today=TODAY, cfg=cfg)
    assert out == []


def test_choose_candidates_nothing_liquid_at_any_session_returns_empty(cfg):
    # Liquid quotes exist, but only at an expiry beyond max_expiry_sessions_ahead.
    beyond = date(2026, 10, 5)
    q = cq_quote(expiration=beyond, strike=575.0, delta=0.50)
    out = choose_candidates([q], underlying="SPY", direction=Direction.BULLISH, spot=575.0, today=TODAY, cfg=cfg)
    assert out == []


def test_choose_candidates_scopes_to_underlying(cfg):
    c = replace(cfg, entry_candidates=3)
    spy_q = cq_quote(underlying="SPY", strike=575.0, delta=0.50)
    qqq_q = cq_quote(underlying="QQQ", strike=480.0, delta=0.50)
    out = choose_candidates([spy_q, qqq_q], underlying="SPY", direction=Direction.BULLISH,
                             spot=575.0, today=TODAY, cfg=c)
    picked_symbols = {leg.symbol for o in out for leg in o.legs}
    assert picked_symbols == {spy_q.symbol}


def test_choose_candidates_deterministic_across_runs(cfg):
    c = replace(cfg, entry_candidates=3)
    chain = [
        cq_quote(strike=560.0, delta=0.30),
        cq_quote(strike=575.0, delta=0.50),
        cq_quote(strike=590.0, delta=0.70),
    ]
    out1 = choose_candidates(chain, underlying="SPY", direction=Direction.BULLISH, spot=575.0, today=TODAY, cfg=c)
    out2 = choose_candidates(list(reversed(chain)), underlying="SPY", direction=Direction.BULLISH,
                              spot=575.0, today=TODAY, cfg=c)
    assert [o.legs[0].symbol for o in out1] == [o.legs[0].symbol for o in out2]


def test_choose_candidates_puts_for_bearish(cfg):
    c = replace(cfg, entry_candidates=1)
    q = cq_quote(right="put", strike=575.0, delta=-0.50)
    out = choose_candidates([q], underlying="SPY", direction=Direction.BEARISH, spot=575.0, today=TODAY, cfg=c)
    assert len(out) == 1
    assert out[0].direction == Direction.BEARISH
    assert out[0].legs[0].symbol == q.symbol
