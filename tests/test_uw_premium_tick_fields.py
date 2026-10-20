"""Tests for the UW net-prem-ticks premium signal in _compute_premium_signals.

Guards the live row shape (string numerics, net_call_premium / net_put_premium),
the verified sign convention (bullish net = nc - np), the 20-row window,
robustness to garbage values, and the legacy fixture shape.
"""

import pytest

from agent.integrations.uw_flow_intel import _compute_premium_signals


def _row(nc, np_, **extra):
    """Build a row with the REAL UW net-prem-ticks shape (numerics as strings)."""
    row = {
        "call_volume": 6317,
        "call_volume_ask_side": 4000,
        "call_volume_bid_side": 2000,
        "date": "2026-10-09",
        "net_call_premium": nc,
        "net_call_volume": 100,
        "net_delta": "131356.8578756113476",
        "net_put_premium": np_,
        "net_put_volume": -50,
        "put_volume": 7721,
        "put_volume_ask_side": 4000,
        "put_volume_bid_side": 3721,
        "tape_time": "2026-10-09T19:02:00.000000Z",
    }
    row.update(extra)
    return row


def _legacy(net, cp, pp):
    """Build a legacy-shape row (net_premium / call_premium / put_premium)."""
    return {"net_premium": net, "call_premium": cp, "put_premium": pp}


def _run(rows):
    """Run the premium signal for SPY over the given rows."""
    return _compute_premium_signals({"full_day": [], "intraday": []}, {"SPY": rows})


def test_t1_bullish_calls_saturate_score():
    """Net call buying with small put selling gives cp/pp/net sums and score 1.0."""
    res = _run([_row("1000.00", "-200.00")] * 20)["tickers"]["SPY"]
    assert res["call"] == pytest.approx(20000)
    assert res["put"] == pytest.approx(-4000)
    assert res["net"] == pytest.approx(24000)
    assert res["score"] == pytest.approx(1.0)
    assert res["dir"] == "BULLISH"


def test_t2_bearish_flow_saturates_negative_score():
    """Net put buying dominates, giving net -26000 and score -1.0."""
    res = _run([_row("-500.00", "800.00")] * 20)["tickers"]["SPY"]
    assert res["net"] == pytest.approx(-26000)
    assert res["score"] == pytest.approx(-1.0)
    assert res["dir"] == "BEARISH"


def test_t3_partial_score_half():
    """Moderate call and put flows give net 4000 and score 0.5."""
    res = _run([_row("300.00", "100.00")] * 20)["tickers"]["SPY"]
    assert res["call"] == pytest.approx(6000)
    assert res["put"] == pytest.approx(2000)
    assert res["net"] == pytest.approx(4000)
    assert res["score"] == pytest.approx(0.5)


def test_t4_convention_pin_nc_minus_np_not_plus():
    """Equal call and put buying nets to zero; an nc+np formula would give +1.0."""
    res = _run([_row("100.00", "100.00")] * 20)["tickers"]["SPY"]
    assert res["net"] == pytest.approx(0.0)
    assert res["score"] == pytest.approx(0.0)


def test_t5_exact_live_row_regression():
    """The exact live row no longer yields a dead 0.0 score (regression for the bug)."""
    row = _row("339008.00", "-45246.00")
    res = _run([row] * 20)["tickers"]["SPY"]
    assert res["score"] > 0
    assert res["score"] != 0.0
    assert res["dir"] == "BULLISH"


def test_t6_legacy_shape_still_summed():
    """Legacy-shape rows still aggregate net, call and put as before (score 0.2)."""
    res = _run([_legacy("100", "300", "200")] * 20)["tickers"]["SPY"]
    assert res["net"] == pytest.approx(2000)
    assert res["call"] == pytest.approx(6000)
    assert res["put"] == pytest.approx(4000)
    assert res["score"] == pytest.approx(0.2)


def test_t7_only_last_20_rows_count():
    """Rows beyond the last 20 are ignored, so the result matches T1 exactly."""
    old = [_row("-9999999", "9999999")] * 5
    recent = [_row("1000.00", "-200.00")] * 20
    res = _run(old + recent)["tickers"]["SPY"]
    base = _run(recent)["tickers"]["SPY"]
    assert res == base
    assert res["net"] == pytest.approx(24000)
    assert res["score"] == pytest.approx(1.0)


def test_t8_garbage_values_count_as_zero():
    """None, empty, nan, inf, garbage and non-dict rows are tolerated and count as 0."""
    garbage = [
        _row(None, None),
        _row("", ""),
        _row("nan", "nan"),
        _row("inf", "-inf"),
        _row("garbage", "garbage"),
        "x",
    ]
    rows = garbage + [_row("1000.00", "-200.00")] * 10
    res = _run(rows)["tickers"]["SPY"]
    assert res["net"] == pytest.approx(12000)
    assert res["score"] == pytest.approx(1.0)


def test_t8b_all_garbage_rows_give_zero_entry():
    """A window of only garbage yields an entry with net 0.0 and score 0.0."""
    rows = [_row("nan", "inf"), _row(None, ""), "x"]
    res = _run(rows)["tickers"]["SPY"]
    assert res["net"] == pytest.approx(0.0)
    assert res["score"] == pytest.approx(0.0)


def test_t9_empty_list_has_no_entry():
    """An empty ticker list produces no entry for that ticker."""
    result = _run([])
    assert "SPY" not in result["tickers"]


def test_t10_mixed_shapes_sum_individually():
    """A mixed window equals the sum of the new-shape and legacy-shape results."""
    new_rows = [_row("1000.00", "-200.00")] * 10
    legacy_rows = [_legacy("100", "300", "200")] * 10
    mixed = _run(new_rows + legacy_rows)["tickers"]["SPY"]
    new_only = _run(new_rows)["tickers"]["SPY"]
    legacy_only = _run(legacy_rows)["tickers"]["SPY"]
    assert mixed["net"] == pytest.approx(new_only["net"] + legacy_only["net"])
    assert mixed["call"] == pytest.approx(new_only["call"] + legacy_only["call"])
    assert mixed["put"] == pytest.approx(new_only["put"] + legacy_only["put"])


def test_t11_nonfinite_values_contribute_exactly_zero():
    """nan/inf/-inf add exactly 0 to net/call/put (not 1.0, not propagated): result equals the finite rows alone."""
    finite = [_row("1000.00", "-200.00")] * 10
    noisy = [_row("nan", "inf"), _row("-inf", "nan"), _row("inf", "inf")]
    res = _run(noisy + finite)["tickers"]["SPY"]
    base = _run(finite)["tickers"]["SPY"]
    assert res["net"] == pytest.approx(base["net"])
    assert res["call"] == pytest.approx(base["call"])
    assert res["put"] == pytest.approx(base["put"])
    assert res["score"] == pytest.approx(base["score"])


def test_t12_put_only_row_uses_new_branch_and_is_bearish():
    """A row with only net_put_premium (no call key) must take the new branch: puts bought = bearish, not a legacy zero."""
    rows = [{"net_put_premium": "100.00", "tape_time": "2026-10-09T19:02:00.000000Z"}] * 20
    res = _run(rows)["tickers"]["SPY"]
    assert res["net"] == pytest.approx(-2000)
    assert res["put"] == pytest.approx(2000)
    assert res["call"] == pytest.approx(0)
    assert res["score"] == pytest.approx(-1.0)
    assert res["dir"] == "BEARISH"


def test_t13_oversized_int_nan_and_inf_never_raise_and_contribute_zero():
    """A 309+-digit int (float() raises OverflowError), 'nan' and 'inf' are 0.0 and never raise out of the scorer."""
    from agent.integrations.uw_flow_intel import _premium_num
    assert _premium_num(10 ** 400) == 0.0 and _premium_num(-(10 ** 400)) == 0.0
    assert _premium_num("nan") == 0.0 and _premium_num("inf") == 0.0 and _premium_num(True if False else None) == 0.0
    rows = [_row(10 ** 400, "inf")] + [_row("1000.00", "-200.00")] * 19
    res = _run(rows)["tickers"]["SPY"]
    assert res["net"] == pytest.approx(19 * 1200.0)
    assert res["score"] == pytest.approx(1.0)
