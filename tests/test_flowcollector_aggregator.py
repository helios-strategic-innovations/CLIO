"""Tests for agent/flowcollector/aggregator.py. Hermetic: no network, no I/O."""

import math

from agent.flowcollector.aggregator import (
    Aggregator,
    RollingTradeStore,
    StrikeExposure,
    UnderlyingFlowState,
    WINDOW_SHORT_SECONDS,
    compute_darkpool,
    compute_delta_volume,
    compute_gex,
    compute_max_pain,
    compute_nofa,
    compute_nope,
    compute_premium,
)
from agent.flowcollector.classifier import SIDE_BUY, SIDE_MID, SIDE_SELL, TradeEvent


def _opt(ts, side, right, price=1.0, size=10, symbol="SPY261002C00580000", exch="A"):
    return TradeEvent(ts=ts, symbol=symbol, underlying="SPY", right=right, strike=580.0, exchange=exch, price=price, size=size, side=side)


def _eq(ts, side, price=580.0, size=100, exch="P", mid_at_ingest=None):
    return TradeEvent(
        ts=ts, symbol="SPY", underlying="SPY", right="", strike=None, exchange=exch, price=price, size=size,
        side=side, mid_at_ingest=mid_at_ingest,
    )


# ── compute_nofa / compute_premium ─────────────────────────────────────────


def test_compute_nofa_empty_is_zero_and_flagged():
    r = compute_nofa([])
    assert r["value"] == 0.0
    assert r["empty"] is True
    assert r["n"] == 0


def test_compute_nofa_call_buying_is_bullish_put_buying_is_bearish():
    call_buy = [_opt(1.0, SIDE_BUY, "call", price=2.0, size=10)]
    r_call = compute_nofa(call_buy, scale=1_000_000.0)
    assert r_call["value"] > 0
    put_buy = [_opt(1.0, SIDE_BUY, "put", price=2.0, size=10)]
    r_put = compute_nofa(put_buy, scale=1_000_000.0)
    assert r_put["value"] < 0
    # symmetric magnitude
    assert math.isclose(r_call["value"], -r_put["value"], rel_tol=1e-9)


def test_compute_nofa_call_selling_is_bearish():
    call_sell = [_opt(1.0, SIDE_SELL, "call", price=2.0, size=10)]
    r = compute_nofa(call_sell, scale=1_000_000.0)
    assert r["value"] < 0


def test_compute_nofa_excludes_mid_unknown_and_saturates():
    events = [_opt(1.0, SIDE_MID, "call"), _opt(1.0, "unknown", "put")]
    r = compute_nofa(events)
    assert r["n"] == 0 and r["empty"] is True
    huge_buy = [_opt(1.0, SIDE_BUY, "call", price=1000.0, size=100000)]
    r2 = compute_nofa(huge_buy, scale=1000.0)
    assert 0.99 < r2["value"] <= 1.0


def test_compute_premium_ignores_call_put_distinction():
    # Buying a put and buying a call are BOTH aggressor buys for `premium`
    # (unlike nofa, which flips sign by right).
    events = [_opt(1.0, SIDE_BUY, "call", price=1.0, size=10), _opt(1.0, SIDE_BUY, "put", price=1.0, size=10)]
    r = compute_premium(events, scale=1_000_000.0)
    assert r["value"] > 0
    assert r["n"] == 2


def test_compute_premium_buy_sell_balance_is_zero():
    events = [_opt(1.0, SIDE_BUY, "call", price=1.0, size=10), _opt(1.0, SIDE_SELL, "put", price=1.0, size=10)]
    r = compute_premium(events)
    assert r["value"] == 0.0


# ── compute_darkpool ─────────────────────────────────────────────────────


def test_compute_darkpool_classifies_vs_midpoint_and_reports_volume_share():
    # side field is unused by darkpool classify -- classification is purely
    # price vs. mid_at_ingest (captured once, at ingest -- see TradeEvent's
    # docstring).
    dark_buy = _eq(1.0, side="unknown", price=101.0, size=1000, exch="D", mid_at_ingest=100.0)
    dark_sell = _eq(1.0, side="unknown", price=99.0, size=500, exch="D", mid_at_ingest=100.0)
    lit = _eq(1.0, side=SIDE_BUY, price=100.0, size=2000, exch="P")

    r = compute_darkpool([dark_buy, dark_sell, lit], total_equity_volume=1000 + 500 + 2000, scale=1_000_000.0)
    assert r["dark_volume"] == 1500
    assert r["dark_trade_count"] == 2
    assert r["classified_count"] == 2
    assert r["bias"] > 0  # 1000@101 buy outweighs 500@99 sell
    assert math.isclose(r["volume_share"], 1500 / 3500)
    assert r["empty"] is False


def test_compute_darkpool_ignores_a_live_mid_re_read_and_uses_ingest_time_mid_only():
    # A dark print whose PRICE (101) is above the CURRENT/live mid (105) but
    # above its own mid_at_ingest (100) must still classify as a buy -- this
    # is the item-9 staleness fix: darkpool classification is frozen at
    # ingest time, never re-read against whatever the NBBO cache holds now.
    dark = _eq(1.0, side="unknown", price=101.0, size=100, exch="D", mid_at_ingest=100.0)
    r = compute_darkpool([dark], total_equity_volume=100)
    assert r["bias"] > 0


def test_compute_darkpool_empty_when_no_dark_trades():
    lit = _eq(1.0, side=SIDE_BUY, price=100.0, size=100, exch="P")
    r = compute_darkpool([lit], total_equity_volume=100)
    assert r["empty"] is True
    assert r["dark_volume"] == 0
    assert r["bias"] == 0.0


def test_compute_darkpool_volume_share_none_without_total():
    dark = _eq(1.0, side="unknown", price=100.0, size=10, exch="D", mid_at_ingest=100.0)
    r = compute_darkpool([dark], total_equity_volume=None)
    assert r["volume_share"] is None


def test_compute_darkpool_no_mid_at_ingest_is_unclassified_not_dropped():
    # A dark print with no mid_at_ingest (e.g. no NBBO had arrived yet)
    # still counts toward dark_volume/dark_trade_count but contributes 0 to
    # the signed bias -- "no signal" is not "no dark activity".
    dark = _eq(1.0, side="unknown", price=101.0, size=100, exch="D", mid_at_ingest=None)
    r = compute_darkpool([dark], total_equity_volume=100)
    assert r["dark_volume"] == 100
    assert r["dark_trade_count"] == 1
    assert r["classified_count"] == 0
    assert r["bias"] == 0.0


# ── compute_delta_volume / compute_nope ────────────────────────────────────


def test_compute_delta_volume_sums_known_deltas_excludes_unknown():
    events = [
        _opt(1.0, SIDE_BUY, "call", symbol="C1", size=10),
        _opt(1.0, SIDE_BUY, "put", symbol="P1", size=5),
        _opt(1.0, SIDE_BUY, "call", symbol="UNKNOWN", size=999),
    ]
    deltas = {"C1": 0.5, "P1": -0.3}
    total = compute_delta_volume(events, deltas)
    assert math.isclose(total, 0.5 * 10 + (-0.3) * 5)


def test_compute_nope_none_without_share_volume():
    assert compute_nope(100.0, None) is None
    assert compute_nope(100.0, 0.0) is None
    assert compute_nope(100.0, -5.0) is None


def test_compute_nope_divides_correctly():
    assert math.isclose(compute_nope(500.0, 1000.0), 0.5)
    assert math.isclose(compute_nope(-500.0, 1000.0), -0.5)


# ── compute_gex ──────────────────────────────────────────────────────────


def test_compute_gex_empty_or_no_spot():
    assert compute_gex([], 580.0)["empty"] is True
    assert compute_gex([StrikeExposure(strike=580.0, call_oi=100, call_gamma_oi=5.0)], None)["empty"] is True
    assert compute_gex([StrikeExposure(strike=580.0, call_oi=100, call_gamma_oi=5.0)], 0.0)["empty"] is True


def test_compute_gex_positive_regime_dominated_by_call_oi():
    strikes = [StrikeExposure(strike=580.0, call_oi=10000, call_gamma_oi=10000 * 0.05, put_oi=100, put_gamma_oi=100 * 0.05)]
    r = compute_gex(strikes, spot=580.0)
    assert r["net_gex"] > 0
    assert r["regime"] == "positive"
    assert r["pin_strike"] == 580.0


def test_compute_gex_negative_regime_dominated_by_put_oi():
    strikes = [StrikeExposure(strike=580.0, call_oi=100, call_gamma_oi=100 * 0.05, put_oi=10000, put_gamma_oi=10000 * 0.05)]
    r = compute_gex(strikes, spot=580.0)
    assert r["net_gex"] < 0
    assert r["regime"] == "negative"


def test_compute_gex_neutral_when_balanced_within_band():
    strikes = [StrikeExposure(strike=580.0, call_oi=1000, call_gamma_oi=1000 * 0.05, put_oi=1000, put_gamma_oi=1000 * 0.05)]
    r = compute_gex(strikes, spot=580.0, neutral_band=0.05)
    assert r["regime"] == "neutral"


def test_compute_gex_pin_strike_is_largest_total_oi():
    strikes = [
        StrikeExposure(strike=575.0, call_oi=100, call_gamma_oi=100 * 0.01, put_oi=100, put_gamma_oi=100 * 0.01),
        StrikeExposure(strike=580.0, call_oi=5000, call_gamma_oi=5000 * 0.01, put_oi=5000, put_gamma_oi=5000 * 0.01),
        StrikeExposure(strike=585.0, call_oi=200, call_gamma_oi=200 * 0.01, put_oi=200, put_gamma_oi=200 * 0.01),
    ]
    r = compute_gex(strikes, spot=580.0)
    assert r["pin_strike"] == 580.0


def test_compute_gex_multi_expiry_same_strike_uses_pre_weighted_sum_not_last_gamma():
    # The item-2 bug: if a caller (incorrectly) overwrote a single
    # "call_gamma" with the LAST contract's gamma and multiplied by TOTAL
    # oi, a 0DTE contract's much larger gamma would either be lost (if
    # processed first) or wrongly applied to the 2DTE contract's OI too (if
    # processed last). The correct pre-weighted sum is
    # Σ gamma_i*OI_i = (0DTE: 500*0.08) + (2DTE: 300*0.01) = 40 + 3 = 43,
    # which this test pins exactly -- StrikeExposure only ever stores the
    # already-summed call_gamma_oi/put_gamma_oi (see its docstring), so
    # THIS test is really pinning compute_gex's contract; the accumulation
    # itself is exercised end-to-end in
    # test_flowcollector_collector.py::test_maybe_refresh_strikes_weights_gamma_per_contract_not_last_seen.
    strikes = [StrikeExposure(strike=580.0, call_oi=500 + 300, call_gamma_oi=(500 * 0.08) + (300 * 0.01))]
    r = compute_gex(strikes, spot=580.0, neutral_band=0.0)
    expected_net = 43.0 * 100.0 * 580.0 * 580.0
    assert math.isclose(r["net_gex"], expected_net)
    assert r["regime"] == "positive"


# ── compute_max_pain ─────────────────────────────────────────────────────


def test_compute_max_pain_none_when_empty():
    assert compute_max_pain({}) is None


def test_compute_max_pain_simple_symmetric_chain():
    # Three strikes, symmetric OI -> max pain should sit at the middle strike
    # (any move away from it makes one side's ITM payout grow).
    oi = {
        95.0: {"call_oi": 100, "put_oi": 100},
        100.0: {"call_oi": 100, "put_oi": 100},
        105.0: {"call_oi": 100, "put_oi": 100},
    }
    assert compute_max_pain(oi) == 100.0


def test_compute_max_pain_skewed_chain_favors_heavy_side():
    # A single dominant position (huge call OI at the LOWEST strike) makes
    # every candidate ABOVE it pay a large ITM payout on that position, but
    # a candidate AT or BELOW it pays nothing on it (candidate > strike is
    # required to owe anything on a call) -- so max pain should land right
    # at that heavy strike, dominating the smaller balanced OI elsewhere.
    oi = {
        100.0: {"call_oi": 100000, "put_oi": 0},
        105.0: {"call_oi": 100, "put_oi": 100},
        110.0: {"call_oi": 100, "put_oi": 100},
    }
    assert compute_max_pain(oi) == 100.0


# ── RollingTradeStore ────────────────────────────────────────────────────


def test_rolling_trade_store_add_window_prune():
    store = RollingTradeStore(max_age_seconds=100.0)
    store.add(_opt(1000.0, SIDE_BUY, "call"))
    store.add(_opt(1050.0, SIDE_BUY, "call"))
    store.add(_opt(1099.0, SIDE_BUY, "call"))
    assert len(store) == 3
    assert len(store.window(1099.0, 50.0)) == 2  # 1050 and 1099 within last 50s
    dropped = store.prune(1099.0)  # cutoff = 999.0, nothing older
    assert dropped == 0
    dropped2 = store.prune(1200.0)  # cutoff = 1100.0 -- all 3 events are older
    assert dropped2 == 3
    assert len(store) == 0


# ── Aggregator.snapshot() end-to-end ──────────────────────────────────────


def test_aggregator_snapshot_unknown_underlying():
    agg = Aggregator(["SPY"])
    snap = agg.snapshot("QQQ", 1000.0)
    assert snap["quality_notes"]["error"] == "unknown_underlying:QQQ"
    assert snap["gex"] is None


def test_aggregator_snapshot_shape_and_contract_keys():
    agg = Aggregator(["SPY"])
    now = 1_700_000_000.0
    agg.add_option_trade("SPY", _opt(now - 10, SIDE_BUY, "call", price=2.0, size=50, symbol="SPY261002C00580000"))
    agg.add_option_trade("SPY", _opt(now - 5, SIDE_SELL, "put", price=1.0, size=20, symbol="SPY261002P00580000"))
    agg.add_equity_trade("SPY", _eq(now - 8, side=SIDE_BUY, price=580.0, size=1000, exch="D", mid_at_ingest=580.0))
    agg.add_equity_trade("SPY", _eq(now - 3, side=SIDE_BUY, price=580.0, size=500, exch="P"))
    st = agg.state["SPY"]
    st.spot = 580.0
    st.strikes = [StrikeExposure(strike=580.0, call_oi=1000, call_gamma_oi=1000 * 0.05, put_oi=800, put_gamma_oi=800 * 0.04)]
    st.oi_by_strike = {580.0: {"call_oi": 1000, "put_oi": 800}}
    st.deltas_by_symbol = {"SPY261002C00580000": 0.5}

    snap = agg.snapshot("SPY", now)

    for key in ("nofa", "premium", "darkpool", "gex", "nope", "max_pain", "gex_regime", "as_of", "quality_notes"):
        assert key in snap
    assert snap["as_of"] == now
    assert -1.0 <= snap["nofa"] <= 1.0
    assert -1.0 <= snap["premium"] <= 1.0
    assert -1.0 <= snap["darkpool"] <= 1.0
    assert snap["gex"] is not None
    assert snap["gex_regime"] in ("positive", "negative", "neutral")
    assert snap["max_pain"] == 580.0
    # nope: only the call trade's contract has a known delta (0.5*50=25),
    # the put trade's contract delta was never supplied -- excluded per
    # compute_delta_volume's documented behavior. share volume 5m = 1500.
    assert snap["nope"] is not None
    assert math.isclose(snap["nope"], (0.5 * 50) / 1500.0)
    assert "windows" in snap
    assert set(snap["windows"].keys()) == {"5m", "15m", "30m"}
    assert snap["quality_notes"]["option_trades_5m"] == 2
    assert snap["quality_notes"]["equity_trades_5m"] == 2
    assert snap["quality_notes"]["dark_trades_5m"] == 1


def test_aggregator_prune_removes_old_events_from_all_underlyings():
    agg = Aggregator(["SPY", "QQQ"])
    agg.add_option_trade("SPY", _opt(1000.0, SIDE_BUY, "call"))
    agg.add_equity_trade("QQQ", _eq(1000.0, side=SIDE_BUY))
    agg.prune(1000.0 + WINDOW_SHORT_SECONDS * 100)  # way past the long window
    assert len(agg.state["SPY"].options) == 0
    assert len(agg.state["QQQ"].equity) == 0


def test_aggregator_add_trade_for_unconfigured_underlying_is_noop():
    agg = Aggregator(["SPY"])
    agg.add_option_trade("QQQ", _opt(1.0, SIDE_BUY, "call"))
    agg.add_equity_trade("QQQ", _eq(1.0, side=SIDE_BUY))
    assert "QQQ" not in agg.state


# ── Scale-free ratios consumed by DPOF's UW fallback (2026-09-25) ─────────────

def test_nofa_premium_ratios_are_scale_free():
    import pytest as _pytest
    from agent.flowcollector.aggregator import compute_nofa, compute_premium
    from agent.flowcollector.classifier import SIDE_BUY, SIDE_SELL, TradeEvent

    def _t(right, side, price, size):
        return TradeEvent(ts=1.0, symbol="SPY261002C00500000", underlying="SPY", right=right,
                          strike=500.0, exchange="X", price=price, size=size, side=side)

    trades = [_t("call", SIDE_BUY, 2.0, 10), _t("put", SIDE_BUY, 1.0, 10), _t("call", SIDE_SELL, 1.0, 5)]
    n1 = compute_nofa(trades, scale=1e3)
    n2 = compute_nofa(trades, scale=1e9)
    assert n1["ratio"] == n2["ratio"]      # independent of the $ scale
    assert n1["value"] != n2["value"]      # the tanh value is not
    # call buy +2000, put buy -1000, call sell -500 -> raw 500 / gross 3500
    assert n1["ratio"] == _pytest.approx(500 / 3500)
    p = compute_premium(trades, scale=1e6)
    # buys 3000, sells 500 -> (3000 - 500) / 3500
    assert p["ratio"] == _pytest.approx(2500 / 3500)
    assert compute_nofa([], scale=1e6)["ratio"] is None


def test_nope_unavailable_when_no_option_prints():
    """Zero option trades in the window (stalled option stream) must read
    unavailable, never a 'balanced' NOPE 0.0 (pass-2 review R1 #1)."""
    agg = Aggregator(["SPY"])
    now = 1_700_000_000.0
    agg.add_equity_trade("SPY", _eq(now - 3, side=SIDE_BUY, price=580.0, size=500_000, exch="P"))
    st = agg.state["SPY"]
    st.spot = 580.0
    st.deltas_by_symbol = {"SPY261002C00580000": 0.5}
    snap = agg.snapshot("SPY", now)
    assert snap["nope"] is None
    assert snap["quality_notes"]["delta_volume_coverage"] is None


def test_volume_coverage_ignores_none_deltas():
    """A symbol mapped to a None delta is not 'known' (same test as
    compute_delta_volume) -- pass-2 review R1 #3."""
    from agent.flowcollector.aggregator import option_volume_coverage

    trades = [
        _opt(1.0, SIDE_BUY, "call", size=10, symbol="A"),
        _opt(2.0, SIDE_BUY, "call", size=30, symbol="B"),
    ]
    cov = option_volume_coverage(trades, {"A": 0.5, "B": None})
    assert cov["coverage"] == 10 / 40
    assert compute_delta_volume(trades, {"A": 0.5, "B": None}) == 0.5 * 10
