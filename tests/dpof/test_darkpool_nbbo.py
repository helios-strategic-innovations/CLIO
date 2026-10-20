"""Tests for agent.dpof.darkpool_nbbo.nbbo_signed_ratio (NBBO-midpoint dark-print ratio)."""

import random

import pytest

from agent.dpof.darkpool_nbbo import nbbo_signed_ratio
from agent.flowcollector.classifier import (
    SIDE_BUY,
    SIDE_MID,
    SIDE_SELL,
    classify_dark_trade_side,
)


def mk(price, size, bid, ask, **kw):
    """Real-shaped UW dark print: price/bid/ask given as strings (raw values pass through)."""
    p = {
        "canceled": False,
        "price": price,
        "size": size,
        "nbbo_bid": bid,
        "nbbo_ask": ask,
        "ticker": "SPY",
    }
    p.update(kw)
    return p


QUOTE_BID, QUOTE_ASK = "10.00", "10.02"  # mid 10.01


def test_buy_above_mid_and_sell_below_mid_give_signed_ratio():
    """A buy above mid and a sell below mid net to (buy - sell) / (buy + sell)."""
    prints = [
        mk("10.02", 100, QUOTE_BID, QUOTE_ASK),  # buy, notional 1002.0
        mk("10.00", 50, QUOTE_BID, QUOTE_ASK),  # sell, notional 500.0
    ]
    r = nbbo_signed_ratio(prints, min_classified=1)
    assert r["ratio"] == pytest.approx(502 / 1502)
    assert r["signed_notional"] == pytest.approx(502.0)
    assert r["gross_notional"] == pytest.approx(1502.0)
    assert r["classified"] == 2


def test_all_buys_ratio_one_and_all_sells_ratio_minus_one():
    """Only buys give +1.0; only sells give -1.0."""
    buys = [mk("10.02", 100, QUOTE_BID, QUOTE_ASK) for _ in range(3)]
    sells = [mk("10.00", 100, QUOTE_BID, QUOTE_ASK) for _ in range(3)]
    assert nbbo_signed_ratio(buys, min_classified=1)["ratio"] == pytest.approx(1.0)
    assert nbbo_signed_ratio(sells, min_classified=1)["ratio"] == pytest.approx(-1.0)


def test_exact_midpoint_print_is_skipped_as_at_mid():
    """A print at the midpoint (float-noise case 779.01 in 779.00/779.02) is counted in at_mid and not classified."""
    at_mid_only = [mk("779.01", 100, "779.00", "779.02")]
    assert nbbo_signed_ratio(at_mid_only, min_classified=1) is None
    mixed = at_mid_only + [mk("779.02", 10, "779.00", "779.02")]  # buy at the ask
    r = nbbo_signed_ratio(mixed, min_classified=1)
    assert r["at_mid"] == 1
    assert r["classified"] == 1
    assert r["ratio"] == pytest.approx(1.0)


def test_mixed_buys_sells_and_at_mid():
    """2 buys (1000 each), 1 sell (500), 3 at-mid -> ratio 0.6, at_mid 3, classified 3."""
    bid, ask = "10.00", "12.00"  # mid 11.0
    prints = [
        mk("12.5", 80, bid, ask),  # buy, 1000
        mk("12.5", 80, bid, ask),  # buy, 1000
        mk("10.0", 50, bid, ask),  # sell, 500
        mk("11.0", 10, bid, ask),
        mk("11.0", 20, bid, ask),
        mk("11.0", 30, bid, ask),
    ]
    r = nbbo_signed_ratio(prints, min_classified=1)
    assert r["ratio"] == pytest.approx(0.6)
    assert r["at_mid"] == 3
    assert r["classified"] == 3


def test_canceled_prints_ignored_bool_and_string():
    """canceled=True (bool) and canceled='true' (string) are skipped and counted in canceled."""
    prints = [
        mk("10.02", 100, QUOTE_BID, QUOTE_ASK, canceled=True),
        mk("10.00", 100, QUOTE_BID, QUOTE_ASK, canceled="true"),
        mk("10.02", 100, QUOTE_BID, QUOTE_ASK),
    ]
    r = nbbo_signed_ratio(prints, min_classified=1)
    assert r["canceled"] == 2
    assert r["classified"] == 1
    assert r["ratio"] == pytest.approx(1.0)


def test_invalid_nbbo_skipped_and_counted():
    """bid '0', missing nbbo keys, and ask < bid are counted in no_nbbo and skipped."""
    bad_zero_bid = mk("10.02", 100, "0", "10.02")
    missing = mk("10.02", 100, QUOTE_BID, QUOTE_ASK)
    del missing["nbbo_bid"]
    del missing["nbbo_ask"]
    inverted = mk("10.02", 100, "10.05", "10.00")
    good = mk("10.02", 100, QUOTE_BID, QUOTE_ASK)
    r = nbbo_signed_ratio([bad_zero_bid, missing, inverted, good], min_classified=1)
    assert r["no_nbbo"] == 3
    assert r["classified"] == 1


def test_bad_price_size_and_non_dict_counted():
    """Bad price/size values count in bad_px; non-dict elements count in non_dict."""
    prints = [
        mk("0", 100, QUOTE_BID, QUOTE_ASK),
        mk("-1", 100, QUOTE_BID, QUOTE_ASK),
        mk("x", 100, QUOTE_BID, QUOTE_ASK),
        mk(None, 100, QUOTE_BID, QUOTE_ASK),
        mk("10.02", 0, QUOTE_BID, QUOTE_ASK),
        mk("10.02", None, QUOTE_BID, QUOTE_ASK),
        "junk",
        42,
        mk("10.02", 100, QUOTE_BID, QUOTE_ASK),
    ]
    r = nbbo_signed_ratio(prints, min_classified=1)
    assert r["bad_px"] == 6
    assert r["non_dict"] == 2
    assert r["classified"] == 1


def test_minimum_sample_guard():
    """19 classified prints under the default min_classified=20 -> None; 20 -> a dict."""
    nineteen = [mk("10.02", 100, QUOTE_BID, QUOTE_ASK) for _ in range(19)]
    twenty = nineteen + [mk("10.02", 100, QUOTE_BID, QUOTE_ASK)]
    assert nbbo_signed_ratio(nineteen) is None
    r = nbbo_signed_ratio(twenty)
    assert isinstance(r, dict)
    assert r["classified"] == 20


def test_ratio_bounded_and_non_sequence_inputs_return_none():
    """Ratio stays in [-1, 1] over 200 seeded random prints; non-list inputs return None."""
    rng = random.Random(7)
    prints = []
    for _ in range(200):
        bid = round(rng.uniform(9.0, 11.0), 2)
        ask = round(bid + rng.uniform(0.01, 0.5), 2)
        price = round(rng.uniform(8.5, 11.5), 2)
        size = rng.randint(1, 1000)
        prints.append(mk(f"{price:.2f}", size, f"{bid:.2f}", f"{ask:.2f}"))
    r = nbbo_signed_ratio(prints, min_classified=1)
    assert r is not None
    assert -1.0 <= r["ratio"] <= 1.0
    for bad in (None, [], "x"):
        assert nbbo_signed_ratio(bad) is None


@pytest.mark.parametrize("price", [10.0, 10.25, 10.5])
@pytest.mark.parametrize("mid", [10.0, 10.25, 10.5])
def test_parity_with_flowcollector_classifier(price, mid):
    """Sign derived from nbbo_signed_ratio matches the flowcollector dark-trade classifier."""
    # Exactly representable quote: bid = mid - 0.25, ask = mid + 0.25 -> mid exactly.
    bid, ask = mid - 0.25, mid + 0.25
    expected = {SIDE_BUY: 1.0, SIDE_SELL: -1.0, SIDE_MID: None}[
        classify_dark_trade_side(price, mid)
    ]
    r = nbbo_signed_ratio([mk(price, 100, bid, ask)], min_classified=1)
    if expected is None:
        assert r is None
    else:
        assert r["ratio"] == pytest.approx(expected)


@pytest.mark.parametrize("bid,ask,price", [("779.06", "779.08", "779.07"), ("779.17", "779.19", "779.18")])
def test_float_noise_midpoint_stays_at_mid(bid, ask, price):
    """Real quotes whose float midpoint differs from the decimal midpoint by ~1e-13 must still count as at-mid."""
    assert (float(bid) + float(ask)) / 2.0 != float(price)  # proves this case really has float noise
    prints = [mk(price, 100, bid, ask), mk(ask, 10, bid, ask)]  # one at-mid print + one clear buy at the ask
    r = nbbo_signed_ratio(prints, min_classified=1)
    assert r["at_mid"] == 1
    assert r["classified"] == 1
    assert r["ratio"] == pytest.approx(1.0)


def test_min_classified_zero_with_nothing_classified_is_none_not_zerodivision():
    """With min_classified=0 and nothing classified the result is None, never a ZeroDivisionError."""
    assert nbbo_signed_ratio([], min_classified=0) is None
    assert nbbo_signed_ratio([mk("779.01", 10, "779.00", "779.02")], min_classified=0) is None


def test_bool_price_or_size_is_rejected():
    """JSON booleans are not numbers: float(True) == 1.0 must not pass as a price or size."""
    prints = [
        mk(True, 100, "10.00", "10.02"),
        mk("10.02", True, "10.00", "10.02"),
        mk("10.02", 100, "10.00", "10.02"),
    ]
    r = nbbo_signed_ratio(prints, min_classified=1)
    assert r["bad_px"] == 2
    assert r["classified"] == 1
