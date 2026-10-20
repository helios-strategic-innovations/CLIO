"""Tests for agent/flowcollector/classifier.py. Hermetic: no network, no I/O."""

from agent.flowcollector.classifier import (
    DARK_POOL_EXCHANGE,
    NBBO,
    SIDE_BUY,
    SIDE_MID,
    SIDE_SELL,
    SIDE_UNKNOWN,
    LastPriceTracker,
    NBBOCache,
    TradeEvent,
    classify_dark_trade_side,
    classify_trade_side,
    find_sweeps,
    is_reliable_trade,
    tick_test,
)


# ── NBBO ─────────────────────────────────────────────────────────────────


def test_nbbo_usable_requires_positive_crossed_free_bid_ask():
    assert NBBO(bid=1.0, ask=1.2).usable is True
    assert NBBO(bid=None, ask=1.2).usable is False
    assert NBBO(bid=1.0, ask=None).usable is False
    assert NBBO(bid=0.0, ask=1.2).usable is False
    assert NBBO(bid=1.5, ask=1.2).usable is False  # crossed/inverted


def test_nbbo_mid():
    assert NBBO(bid=1.0, ask=1.2).mid == 1.1
    assert NBBO(bid=None, ask=1.2).mid is None


# ── tick_test ────────────────────────────────────────────────────────────


def test_tick_test_uptick_downtick_zero_tick_and_unknown():
    assert tick_test(101.0, 100.0, None) == SIDE_BUY
    assert tick_test(99.0, 100.0, None) == SIDE_SELL
    assert tick_test(100.0, 100.0, SIDE_BUY) == SIDE_BUY  # zero-tick carries forward
    assert tick_test(100.0, 100.0, SIDE_SELL) == SIDE_SELL
    assert tick_test(100.0, None, None) == SIDE_UNKNOWN  # nothing to compare
    assert tick_test(100.0, None, None, mid=99.0) == SIDE_BUY
    assert tick_test(100.0, None, None, mid=101.0) == SIDE_SELL
    assert tick_test(100.0, None, None, mid=100.0) == SIDE_MID


# ── classify_trade_side (quote rule + Lee-Ready fallback) ─────────────────


def test_classify_at_or_above_ask_is_buy():
    nbbo = NBBO(bid=1.00, ask=1.10)
    assert classify_trade_side(1.10, nbbo) == SIDE_BUY
    assert classify_trade_side(1.15, nbbo) == SIDE_BUY


def test_classify_at_or_below_bid_is_sell():
    nbbo = NBBO(bid=1.00, ask=1.10)
    assert classify_trade_side(1.00, nbbo) == SIDE_SELL
    assert classify_trade_side(0.90, nbbo) == SIDE_SELL


def test_classify_epsilon_tolerance_absorbs_subcent_noise():
    nbbo = NBBO(bid=1.00, ask=1.10)
    # 1.098 is inside the raw spread but within epsilon of the ask.
    assert classify_trade_side(1.098, nbbo, epsilon=0.005) == SIDE_BUY
    assert classify_trade_side(1.002, nbbo, epsilon=0.005) == SIDE_SELL


def test_classify_inside_spread_falls_back_to_tick_test():
    nbbo = NBBO(bid=1.00, ask=1.10)
    # Strictly inside the spread, epsilon=0 -> quote rule doesn't resolve it.
    assert classify_trade_side(1.05, nbbo, prev_price=1.03, epsilon=0.0) == SIDE_BUY
    assert classify_trade_side(1.05, nbbo, prev_price=1.07, epsilon=0.0) == SIDE_SELL
    # No prev price either -> midpoint split via tick_test's mid fallback.
    assert classify_trade_side(1.06, nbbo, epsilon=0.0) == SIDE_BUY  # 1.06 > mid 1.05
    assert classify_trade_side(1.04, nbbo, epsilon=0.0) == SIDE_SELL


def test_classify_no_usable_nbbo_uses_tick_test_only():
    assert classify_trade_side(100.0, None, prev_price=99.0) == SIDE_BUY
    assert classify_trade_side(100.0, NBBO(bid=None, ask=None), prev_price=101.0) == SIDE_SELL
    assert classify_trade_side(100.0, None, prev_price=None) == SIDE_UNKNOWN


# ── classify_dark_trade_side ───────────────────────────────────────────────


def test_classify_dark_trade_side_vs_midpoint():
    assert classify_dark_trade_side(101.0, 100.0) == SIDE_BUY
    assert classify_dark_trade_side(99.0, 100.0) == SIDE_SELL
    assert classify_dark_trade_side(100.0, 100.0) == SIDE_MID
    assert classify_dark_trade_side(100.0, None) == SIDE_UNKNOWN


# ── is_reliable_trade ──────────────────────────────────────────────────────


def test_is_reliable_trade_default_excludes_only_odd_lot():
    assert is_reliable_trade(None) is True
    assert is_reliable_trade([]) is True
    assert is_reliable_trade(["@"]) is True
    assert is_reliable_trade(["I"]) is False
    assert is_reliable_trade(["@", "I"]) is False


def test_is_reliable_trade_custom_exclude_set():
    assert is_reliable_trade(["Z"], exclude=frozenset({"Z"})) is False
    assert is_reliable_trade(["Z"], exclude=frozenset({"I"})) is True


# ── NBBOCache / LastPriceTracker ───────────────────────────────────────────


def test_nbbo_cache_update_get_snapshot():
    cache = NBBOCache()
    assert cache.get("SPY") is None
    n = NBBO(bid=1.0, ask=1.1)
    cache.update("SPY", n)
    assert cache.get("SPY") == n
    assert len(cache) == 1
    snap = cache.snapshot()
    assert snap == {"SPY": n}
    # snapshot() is a copy, not a live view
    cache.update("QQQ", NBBO(bid=2.0, ask=2.1))
    assert "QQQ" not in snap


def test_last_price_tracker_get_update():
    tr = LastPriceTracker()
    assert tr.get("SPY") == (None, None)
    tr.update("SPY", 100.0, SIDE_BUY)
    assert tr.get("SPY") == (100.0, SIDE_BUY)
    tr.update("SPY", 101.0, SIDE_SELL)
    assert tr.get("SPY") == (101.0, SIDE_SELL)


# ── find_sweeps ─────────────────────────────────────────────────────────────


def _trade(ts, exch, side="buy", underlying="SPY", strike=580.0, right="call"):
    return TradeEvent(
        ts=ts, symbol=f"SPY261002C00580000", underlying=underlying, right=right, strike=strike,
        exchange=exch, price=1.5, size=10, side=side,
    )


def test_find_sweeps_flags_multi_exchange_cluster_within_window():
    events = [
        _trade(1000.00, "A"),
        _trade(1000.10, "B"),
        _trade(1000.20, "C"),
    ]
    swept = find_sweeps(events, window_seconds=0.5, min_exchanges=2)
    assert swept == {0, 1, 2}


def test_find_sweeps_single_exchange_not_flagged():
    events = [_trade(1000.0, "A"), _trade(1000.1, "A"), _trade(1000.2, "A")]
    assert find_sweeps(events, window_seconds=0.5, min_exchanges=2) == set()


def test_find_sweeps_outside_window_not_clustered():
    events = [_trade(1000.0, "A"), _trade(1002.0, "B")]  # 2s apart, window 0.5s
    assert find_sweeps(events, window_seconds=0.5, min_exchanges=2) == set()


def test_find_sweeps_mixed_sides_do_not_merge():
    events = [
        _trade(1000.0, "A", side="buy"),
        _trade(1000.1, "B", side="sell"),  # different side -> different group
    ]
    assert find_sweeps(events, window_seconds=0.5, min_exchanges=2) == set()


def test_find_sweeps_excludes_undirected_sides():
    events = [_trade(1000.0, "A", side=SIDE_MID), _trade(1000.1, "B", side=SIDE_MID)]
    assert find_sweeps(events, window_seconds=0.5, min_exchanges=2) == set()


def test_find_sweeps_different_strikes_do_not_merge():
    events = [
        _trade(1000.0, "A", strike=580.0),
        _trade(1000.1, "B", strike=585.0),
    ]
    assert find_sweeps(events, window_seconds=0.5, min_exchanges=2) == set()


def test_dark_pool_exchange_constant():
    assert DARK_POOL_EXCHANGE == "D"
