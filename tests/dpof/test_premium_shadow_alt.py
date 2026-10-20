"""R-37 shadow evidence: premium alternatives (w60 / ewma / day), premium saturation run and GEX continuous
normalisation. SHADOW ONLY: every test here also pins that the live reading is byte-identical.

Plugin-free (asyncio.run). Only the UW fetch is faked (tests/dpof/test_data.Fakes); everything under tmp_path.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

import agent.dpof.data as data
import agent.flowcollector.state as flowcollector_state
from agent.dpof.config import DPOFConfig
from agent.dpof.dynamic_weights import DynamicWeightsManager
from agent.dpof.models import PillarReading, UnderlyingSnapshot
from agent.dpof.pillar_weights_io import ShadowLog
from tests.dpof.test_data import FakeBroker, Fakes

ET = ZoneInfo("America/New_York")
BASE = datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc).timestamp()
DAY = "2026-10-05"
STEP = 120.0

OLD_ROW_KEYS = {
    "ts", "u", "mode", "static", "dyn", "composite_with_flat", "composite_without_flat",
    "w", "alive", "m", "ma", "n_eff", "avail", "dead",
}
NEW_ROW_KEYS = {"premium_alt", "premium_sat_run", "gex_cont"}


# ───────────────────────── helpers ─────────────────────────

def _pos(nc=100, np_=0):
    return {"net_call_premium": str(nc), "net_put_premium": str(np_)}


def _neg(nc=0, np_=100):
    return {"net_call_premium": str(nc), "net_put_premium": str(np_)}


def _mixed_then_one_sided():
    return [_pos(100, 100)] * 10 + [_pos(100, 0)] * 20


def _neg50_pos20():
    # 50 negative then 20 positive: last 60 = 40 neg + 20 pos; last 20 = all pos; all = 50 neg + 20 pos
    return [_neg()] * 50 + [_pos()] * 20


CASES = {
    "mixed_then_one_sided": (_mixed_then_one_sided(), list(reversed(_mixed_then_one_sided()))),
    "long_negative_then_positive": (_neg50_pos20(), [_pos()] * 3 + [_neg()] * 5),
    "legacy_fields": ([{"net_premium": 50, "call_premium": 100, "put_premium": 50}] * 5, []),
    "garbage_only": (["x", None, {"foo": 1}, {"net_call_premium": "abc"}], [{"net_put_premium": None}]),
    "empty": ([], []),
}


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    return Fakes(monkeypatch, tmp_path)


@pytest.fixture(autouse=True)
def _fast_timeouts(monkeypatch):
    monkeypatch.setattr(data, "_UW_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_BARS_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_VIX_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_FILE_GATE_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_SNAPSHOT_BUDGET_SECONDS", 5.0)


@pytest.fixture(autouse=True)
def _isolate_selfflow_file(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(tmp_path / "no_selfflow_state.json"))


def _make_feed(tmp_path, name, mode):
    data_dir = tmp_path / name
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(
        DPOFConfig(), data_dir=str(data_dir), min_pillars=2, uw_refresh_seconds=100,
        bars_refresh_seconds=50, underlyings=("SPY", "QQQ"), dynamic_weights=mode, dw_ring_size=10,
        selfflow_fallback=False, kronos_pillar="off", empty_pillar_unavailable=False,
    )
    return data.DPOFDataFeed(cfg, FakeBroker()), cfg


def _snap(sym="SPY", premium=0.2, gex=None):
    pillars = {}
    if premium is None:
        pillars["premium"] = PillarReading.missing("premium", "absent", "uw:/flow")
    else:
        pillars["premium"] = PillarReading(name="premium", value=premium, available=True, source="uw:/flow")
    pillars["nofa"] = PillarReading(name="nofa", value=0.3, available=True, source="uw:/nofa")
    pillars["darkpool"] = PillarReading(name="darkpool", value=0.1, available=True, source="uw:/darkpool")
    return UnderlyingSnapshot(symbol=sym, pillars=pillars, last_price=100.0, price_age_seconds=5.0,
                              ret_15m=0.002, gex_regime=gex, pin_risk=None)


def _mgr(tmp_path, mode="shadow"):
    cfg = dataclasses.replace(DPOFConfig(), data_dir=str(tmp_path), dynamic_weights=mode,
                              dw_ring_size=10, kronos_pillar="live")
    return DynamicWeightsManager(cfg, str(tmp_path), session_fn=lambda now: (DAY, True), sync_io=True)


def _et(h, m):
    return datetime(2026, 10, 5, h, m, tzinfo=ET).timestamp()


def _last_row(tmp_path):
    return ShadowLog(str(tmp_path)).read_tail(1)[-1]


# ───────────────────────── A. formulas ─────────────────────────

def test_one_sided_recent_ticks_read_live_one_but_alternatives_stay_below_one():
    ticks = [_pos(100, 100)] * 10 + [_pos(100, 0)] * 20
    alt = data._premium_alternatives(ticks)
    # live = last 20 ticks only: net 2000 / (2000 + 0) = 1.0
    from agent.integrations.uw_flow_intel import _compute_premium_signals
    live = _compute_premium_signals({"full_day": [], "intraday": []}, {"SPY": ticks})["tickers"]["SPY"]["score"]
    assert live == pytest.approx(1.0)
    # w60 and day over all 30 ticks: net 2000 / (3000 + 1000) = 0.5
    assert alt["w60"] == pytest.approx(0.5)
    assert alt["day"] == pytest.approx(0.5)
    # ewma: first 10 shares are 0, then 20 shares of 1 with alpha 0.1 -> 1 - 0.9**20 = 0.8784
    assert alt["ewma"] == 0.8784
    assert alt["ewma"] < 1.0 and alt["day"] < 1.0


def test_w60_is_the_last_sixty_ticks_and_day_is_all_ticks():
    ticks = [_neg()] * 50 + [_pos()] * 20  # 70 ticks (the 10 oldest negative ticks fall outside w60)
    alt = data._premium_alternatives(ticks)
    # last 60 = 40 negative + 20 positive: net -2000 / (2000 + 4000) = -1/3
    assert alt["w60"] == pytest.approx(-1 / 3, abs=1e-4)
    # all 70: net -3000 / (2000 + 5000) = -3/7
    assert alt["day"] == pytest.approx(-3 / 7, abs=1e-4)


def test_ewma_alpha_is_point_one():
    ticks = [_pos(100, 100)] * 10 + [_pos(100, 0)] * 20
    # With alpha 0.5 the same series gives 1 - 0.5**20 ~ 1.0; alpha 0.1 must stay at 0.8784.
    assert data._premium_alternatives(ticks)["ewma"] == 0.8784


def test_ewma_first_tick_initialises_and_is_clamped():
    assert data._premium_alternatives([_pos(500, 0)])["ewma"] == 1.0
    assert data._premium_alternatives([_neg(0, 500)])["ewma"] == -1.0


def test_legacy_fields_branch_matches_the_live_formula():
    ticks = [{"net_premium": 50, "call_premium": 100, "put_premium": 50}] * 5
    alt = data._premium_alternatives(ticks)
    # net 250 / (500 + 250) = 1/3
    assert alt["w60"] == pytest.approx(1 / 3, abs=1e-4)
    assert alt["day"] == pytest.approx(1 / 3, abs=1e-4)
    assert alt["ewma"] == pytest.approx(1 / 3, abs=1e-4)


@pytest.mark.parametrize("ticks", [[], None, "not-a-list", ["x", None, 5, {"foo": 1}], [{"net_call_premium": "abc"}]])
def test_no_usable_ticks_gives_none_everywhere(ticks):
    assert data._premium_alternatives(ticks) == {"w60": None, "ewma": None, "day": None}


def test_non_dict_ticks_are_skipped_not_fatal():
    ticks = [None, _pos(100, 0), "junk", 7]
    alt = data._premium_alternatives(ticks)
    assert alt == {"w60": 1.0, "ewma": 1.0, "day": 1.0}


def test_results_are_rounded_to_four_decimals():
    ticks = [_pos(1, 0), _neg(0, 3), _pos(2, 1)]  # shares 1, -1, 1/3: the EWMA is not a short decimal
    alt = data._premium_alternatives(ticks)
    for v in alt.values():
        assert v == round(v, 4)
    assert alt["ewma"] == round(alt["ewma"], 4)


# ───────────────────────── B. golden: live reading unchanged ─────────────────────────

def _state(feed):
    return {
        "cache": dict(feed._pillar_cache),
        "time": dict(feed._pillar_cache_time),
        "err": dict(feed._pillar_last_error),
        "gex_regime": dict(feed._gex_regime_cache),
    }


def _refresh_both(feed, fakes, spy_ticks, qqq_ticks, strikes):
    fakes.uw_responses[data._EP_MARKET_TIDE] = {"data": []}
    fakes.uw_responses["/stock/SPY/net-prem-ticks"] = {"data": spy_ticks}
    fakes.uw_responses["/stock/QQQ/net-prem-ticks"] = {"data": qqq_ticks}
    fakes.uw_responses["/stock/SPY/spot-exposures/strike"] = {"data": strikes}
    fakes.uw_responses["/stock/QQQ/spot-exposures/strike"] = {"data": []}

    async def go():
        await feed._refresh_premium(BASE)
        await feed._refresh_gex(BASE)

    asyncio.run(go())
    return _state(feed)


STRIKES = [{"strike": 580, "call_gamma_oi": 10, "put_gamma_oi": -50}]


@pytest.mark.parametrize("case", sorted(CASES))
def test_live_premium_and_gex_identical_with_hook_raising_or_absent_or_off(case, tmp_path, fakes, monkeypatch):
    spy, qqq = CASES[case]

    normal_feed, _ = _make_feed(tmp_path, "normal", "shadow")
    ref = _refresh_both(normal_feed, fakes, spy, qqq, STRIKES)

    absent_feed, _ = _make_feed(tmp_path, "absent", "shadow")
    absent_feed._dw = None
    spy_calls = []
    real = data._premium_alternatives
    monkeypatch.setattr(data, "_premium_alternatives", lambda t: (spy_calls.append(1), real(t))[1])
    assert _refresh_both(absent_feed, fakes, spy, qqq, STRIKES) == ref
    assert spy_calls == []  # no manager: the alternatives are never even computed
    monkeypatch.setattr(data, "_premium_alternatives", real)

    off_feed, _ = _make_feed(tmp_path, "off", "off")
    assert _refresh_both(off_feed, fakes, spy, qqq, STRIKES) == ref

    raising_feed, _ = _make_feed(tmp_path, "raising", "shadow")

    def boom(*_a, **_k):
        raise RuntimeError("injected hook failure")

    monkeypatch.setattr(raising_feed._dw, "observe_premium_alt", boom)
    monkeypatch.setattr(raising_feed._dw, "observe_gex", boom)
    assert _refresh_both(raising_feed, fakes, spy, qqq, STRIKES) == ref


def test_alternatives_computation_failure_never_changes_or_raises(tmp_path, fakes, monkeypatch):
    spy, qqq = CASES["mixed_then_one_sided"]
    feed_ok, _ = _make_feed(tmp_path, "ok", "shadow")
    ref = _refresh_both(feed_ok, fakes, spy, qqq, STRIKES)

    def boom(_t):
        raise ValueError("injected formula failure")

    feed_bad, _ = _make_feed(tmp_path, "bad", "shadow")
    monkeypatch.setattr(data, "_premium_alternatives", boom)
    assert _refresh_both(feed_bad, fakes, spy, qqq, STRIKES) == ref
    assert feed_bad._dw._premium_alt == {}


def test_refresh_records_alternatives_and_gex_observations_in_memory(tmp_path, fakes):
    spy, qqq = CASES["mixed_then_one_sided"]
    feed, _ = _make_feed(tmp_path, "rec", "shadow")
    _refresh_both(feed, fakes, spy, qqq, STRIKES)
    rec = feed._dw._premium_alt["SPY"]
    assert rec == {"live": 1.0, "w60": 0.5, "ewma": 0.8784, "day": 0.5}
    # QQQ had an empty strike list: no GEX observation may be created for it
    assert "QQQ" not in feed._dw._gex_obs
    # SPY net_gex = 10 + (-50) = -40
    assert list(feed._dw._gex_obs["SPY"]) == [-40.0]


# ───────────────────────── C. manager: saturation run ─────────────────────────

def test_saturation_run_counts_consecutive_cycles_and_resets(tmp_path):
    m = _mgr(tmp_path)
    seq = [1.0, -1.0, 0.9999, 0.5, 0.999, 0.998, None, 1.0]
    expected = [1, 2, 3, 0, 1, 0, 0, 1]
    got = []
    for i, p in enumerate(seq):
        m.finish_snapshot(_et(10, 0) + i * 60, {"SPY": _snap(premium=p)})
        got.append(_last_row(tmp_path)["premium_sat_run"])
    assert got == expected


def test_saturation_run_is_tracked_per_underlying(tmp_path):
    m = _mgr(tmp_path)
    t = _et(10, 0)
    m.finish_snapshot(t, {"SPY": _snap("SPY", premium=1.0), "QQQ": _snap("QQQ", premium=0.1)})
    m.finish_snapshot(t + 60, {"SPY": _snap("SPY", premium=1.0), "QQQ": _snap("QQQ", premium=0.1)})
    rows = ShadowLog(str(tmp_path)).read_tail(4)
    runs = {(r["u"], r["premium_sat_run"]) for r in rows[-2:]}
    assert runs == {("SPY", 2), ("QQQ", 0)}


# ───────────────────────── D. manager: gex_cont ─────────────────────────

def _gex_row(m, tmp_path, t):
    m.finish_snapshot(t, {"SPY": _snap("SPY")})
    return _last_row(tmp_path)


def test_gex_cont_is_none_before_twenty_observations_then_normalised(tmp_path):
    m = _mgr(tmp_path)
    for i in range(19):
        m.observe_gex("SPY", -10.0, _et(10, 0) + i)
    assert _gex_row(m, tmp_path, _et(10, 5))["gex_cont"] is None
    m.observe_gex("SPY", 30.0, _et(10, 30))  # 20 observations: |values| median = 10
    assert _gex_row(m, tmp_path, _et(10, 35))["gex_cont"] == 3.0


def test_gex_cont_median_uses_absolute_values_not_signed(tmp_path):
    m = _mgr(tmp_path)
    for i in range(11):
        m.observe_gex("SPY", -100.0, _et(10, 0) + i)
    for i in range(9):
        m.observe_gex("SPY", 1.0, _et(10, 0) + 20 + i)
    # latest = 1.0 ; |x| median = 100 -> 0.01 (a signed median would give -0.01)
    assert _gex_row(m, tmp_path, _et(10, 40))["gex_cont"] == 0.01


def test_gex_cont_is_none_when_the_median_is_zero(tmp_path):
    m = _mgr(tmp_path)
    for i in range(20):
        m.observe_gex("SPY", 0.0, _et(10, 0) + i)
    assert _gex_row(m, tmp_path, _et(10, 30))["gex_cont"] is None


def test_gex_window_is_bounded_at_two_hundred(tmp_path):
    m = _mgr(tmp_path)
    for i in range(250):
        m.observe_gex("SPY", float(i + 1), _et(10, 0) + i)
    dq = m._gex_obs["SPY"]
    assert len(dq) == 200
    assert dq[0] == 51.0 and dq[-1] == 250.0


@pytest.mark.parametrize("bad", [None, True, False, float("nan"), float("inf"), "12", [1]])
def test_non_finite_or_non_numeric_gex_is_never_recorded(tmp_path, bad):
    m = _mgr(tmp_path)
    m.observe_gex("SPY", bad, _et(10, 0))
    assert "SPY" not in m._gex_obs


def test_gex_observations_are_per_underlying(tmp_path):
    m = _mgr(tmp_path)
    m.observe_gex("SPY", 5.0, _et(10, 0))
    assert "QQQ" not in m._gex_obs
    assert list(m._gex_obs["SPY"]) == [5.0]


# ───────────────────────── E. manager: shadow row ─────────────────────────

def test_row_has_old_keys_and_new_keys_and_none_before_first_observation(tmp_path):
    m = _mgr(tmp_path)
    m.finish_snapshot(_et(10, 0), {"SPY": _snap()})
    row = _last_row(tmp_path)
    assert OLD_ROW_KEYS <= set(row)
    assert NEW_ROW_KEYS <= set(row)
    assert row["premium_alt"] is None
    assert row["gex_cont"] is None
    assert row["premium_sat_run"] == 0  # premium 0.2 is not saturated


def test_premium_alt_row_carries_live_and_three_alternatives(tmp_path):
    m = _mgr(tmp_path)
    m.observe_premium_alt("SPY", 1.0, {"w60": 0.5, "ewma": 0.8784, "day": 0.5}, _et(10, 0))
    m.finish_snapshot(_et(10, 0), {"SPY": _snap(premium=1.0)})
    row = _last_row(tmp_path)
    assert row["premium_alt"] == {"live": 1.0, "w60": 0.5, "ewma": 0.8784, "day": 0.5}
    assert row["premium_sat_run"] == 1


def test_premium_alt_non_finite_values_become_none(tmp_path):
    m = _mgr(tmp_path)
    m.observe_premium_alt("SPY", float("nan"), {"w60": None, "ewma": float("inf"), "day": 0.25}, _et(10, 0))
    m.finish_snapshot(_et(10, 0), {"SPY": _snap()})
    assert _last_row(tmp_path)["premium_alt"] == {"live": None, "w60": None, "ewma": None, "day": 0.25}


def test_unavailable_premium_resets_the_run_and_the_row_still_has_the_key(tmp_path):
    m = _mgr(tmp_path)
    m.finish_snapshot(_et(10, 0), {"SPY": _snap(premium=1.0)})
    m.finish_snapshot(_et(10, 1), {"SPY": _snap(premium=None)})
    row = _last_row(tmp_path)
    assert row["premium_sat_run"] == 0
    assert "premium_sat_run" in row


def test_live_weights_and_static_composite_are_unchanged_by_the_shadow_alternatives(tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "alt").mkdir()
    plain = _mgr(tmp_path / "plain")
    with_alt = _mgr(tmp_path / "alt")
    with_alt.observe_premium_alt("SPY", 1.0, {"w60": 0.5, "ewma": 0.1, "day": 0.2}, _et(10, 0))
    with_alt.observe_gex("SPY", 3.0, _et(10, 0))
    u_plain, u_alt = _snap(premium=1.0), _snap(premium=1.0)
    plain.finish_snapshot(_et(10, 0), {"SPY": u_plain})
    with_alt.finish_snapshot(_et(10, 0), {"SPY": u_alt})
    a = _last_row(tmp_path / "plain")
    b = _last_row(tmp_path / "alt")
    for k in OLD_ROW_KEYS - {"ts"}:
        assert a[k] == b[k], k
    assert u_plain.pillar_weights == u_alt.pillar_weights
    assert u_plain.pillars["premium"].value == u_alt.pillars["premium"].value == 1.0


def test_alt_helper_failure_drops_only_the_new_fields(tmp_path, caplog):
    m = _mgr(tmp_path)
    m._gex_obs = None  # breaks only the gex_cont step
    m.finish_snapshot(_et(10, 0), {"SPY": _snap()})
    row = _last_row(tmp_path)
    assert row["gex_cont"] is None
    assert OLD_ROW_KEYS <= set(row)  # the row was still written with all its old keys


def test_shadow_row_is_json_strict(tmp_path):
    m = _mgr(tmp_path)
    m.observe_premium_alt("SPY", 1.0, {"w60": 0.5, "ewma": 0.1, "day": 0.2}, _et(10, 0))
    m.finish_snapshot(_et(10, 0), {"SPY": _snap(premium=1.0)})
    line = (tmp_path / "pillar_weights_shadow.jsonl").read_text().splitlines()[-1]

    def _reject(const):
        raise ValueError(f"non-strict JSON constant {const}")

    json.loads(line, parse_constant=_reject)  # NaN / Infinity must never reach the file


def test_old_rows_without_new_keys_still_parse_with_a_plain_dict_reader(tmp_path):
    """An old-format row (no R-37 keys) and a new row side by side: the common keys read the same way."""
    old = {"ts": "2026-10-05T10:00:00-04:00", "u": "SPY", "mode": "shadow", "static": 0.1, "dyn": 0.1,
           "composite_with_flat": 0.1, "composite_without_flat": 0.1, "w": {}, "alive": {}, "m": {},
           "ma": {}, "n_eff": {}, "avail": [], "dead": []}
    path = tmp_path / "pillar_weights_shadow.jsonl"
    path.write_text(json.dumps(old) + "\n")
    m = _mgr(tmp_path)
    m.finish_snapshot(_et(10, 5), {"SPY": _snap()})
    rows = ShadowLog(str(tmp_path)).read_tail(5)
    assert len(rows) == 2
    assert all("static" in r and "dyn" in r for r in rows)
    assert "premium_alt" not in rows[0] and "premium_alt" in rows[1]


# ───────────────────────── F. orchestrator mutants (R-37 fix-up) ─────────────────────────

def _live_score(ticks):
    """The LIVE premium score for one ticker, straight from the production function (last 20 ticks)."""
    from agent.integrations.uw_flow_intel import _compute_premium_signals
    return _compute_premium_signals({}, {"SPY": ticks})["tickers"]["SPY"]["score"]


def _parity_cases():
    import random
    rng = random.Random(20261012)
    cases = {
        "tiny_sub_dollar_premiums": [{"net_call_premium": "0.2", "net_put_premium": "0"}] * 3,
        "legacy_net_exceeds_gross": [{"net_premium": 100, "call_premium": 1, "put_premium": 1}] * 2,
        "legacy_negative_net_exceeds_gross": [{"net_premium": -250, "call_premium": 2, "put_premium": 1}],
        "offsetting_call_ticks": [_pos(100, 0), _pos(-100, 0), _pos(50, 20)],
        "put_heavy": [_neg(10, 900)] * 4 + [_pos(30, 0)],
    }
    for i in range(12):
        n = rng.randint(1, 20)
        cases[f"random_{i}"] = [
            {"net_call_premium": str(rng.choice([0, 0.3, 40, -25, 1200, 5e5])),
             "net_put_premium": str(rng.choice([0, 0.4, 70, -10, 900, 4e5]))}
            for _ in range(n)
        ]
    return cases


@pytest.mark.parametrize("name,ticks", list(_parity_cases().items()))
def test_w60_and_day_equal_the_live_formula_when_the_window_covers_every_tick(name, ticks):
    # With <= 20 usable ticks the live window, the 60-tick window and the whole day are the same set of ticks,
    # so the shadow alternatives must reproduce the live score exactly (same clamp, same denominator floor).
    alt = data._premium_alternatives(ticks)
    expected = round(_live_score(ticks), 4)
    assert alt["w60"] == expected, name
    assert alt["day"] == expected, name


def test_ewma_runs_over_every_tick_of_the_day_not_just_the_last_sixty():
    ticks = [_neg()] + [_pos()] * 69          # first tick share -1, then 69 ticks of +1
    share = [-1.0] + [1.0] * 69
    ewma = share[0]
    for s in share[1:]:
        ewma = 0.1 * s + 0.9 * ewma
    got = data._premium_alternatives(ticks)["ewma"]
    assert got == round(ewma, 4)
    assert got < 1.0                            # the day-opening tick still leaves a visible trace at four decimals


def test_saturation_run_does_not_borrow_another_underlyings_count(tmp_path):
    m = _mgr(tmp_path)
    t = _et(10, 0)
    for i, qqq in enumerate((0.1, 0.1, 1.0)):
        m.finish_snapshot(t + 60 * i, {"SPY": _snap("SPY", premium=1.0), "QQQ": _snap("QQQ", premium=qqq)})
    rows = {r["u"]: r["premium_sat_run"] for r in ShadowLog(str(tmp_path)).read_tail(2)}
    assert rows == {"SPY": 3, "QQQ": 1}


def test_observe_premium_alt_rounds_every_value_to_four_decimals(tmp_path):
    m = _mgr(tmp_path)
    m.observe_premium_alt("SPY", 0.123456, {"w60": 0.987654, "ewma": -0.5, "day": 0.0001234}, _et(10, 0))
    m.finish_snapshot(_et(10, 0), {"SPY": _snap("SPY")})
    assert _last_row(tmp_path)["premium_alt"] == {"live": 0.1235, "w60": 0.9877, "ewma": -0.5, "day": 0.0001}


def test_off_mode_records_no_shadow_observations_at_all(tmp_path):
    m = _mgr(tmp_path, mode="off")
    m.observe_gex("SPY", 5.0, _et(10, 0))
    m.observe_premium_alt("SPY", 0.5, {"w60": 0.5, "ewma": 0.5, "day": 0.5}, _et(10, 0))
    assert m._gex_obs == {} and m._premium_alt == {}


def test_a_broken_gex_window_drops_only_gex_cont_whatever_the_error_type(tmp_path):
    from collections import deque
    m = _mgr(tmp_path)
    m._gex_obs["SPY"] = deque(["x"] * 25, maxlen=200)      # abs("x") raises TypeError, not ZeroDivisionError
    m.finish_snapshot(_et(10, 0), {"SPY": _snap("SPY")})
    row = _last_row(tmp_path)
    assert row["gex_cont"] is None and OLD_ROW_KEYS <= set(row)


def test_a_broken_premium_alt_record_drops_only_premium_alt(tmp_path):
    m = _mgr(tmp_path)
    m._premium_alt["SPY"] = 5                                # dict(5) raises TypeError
    m.finish_snapshot(_et(10, 0), {"SPY": _snap("SPY", premium=1.0)})
    row = _last_row(tmp_path)
    assert row["premium_alt"] is None and row["premium_sat_run"] == 1 and OLD_ROW_KEYS <= set(row)


def test_a_broken_saturation_counter_drops_only_the_run_field(tmp_path):
    m = _mgr(tmp_path)
    m._premium_sat_run = None                                # None.get raises AttributeError inside the sat-run step
    m.finish_snapshot(_et(10, 0), {"SPY": _snap("SPY", premium=1.0)})
    row = _last_row(tmp_path)
    assert row["premium_sat_run"] == 0 and OLD_ROW_KEYS <= set(row)
    assert row["premium_alt"] is None                        # untouched steps still produce their normal (empty) value
