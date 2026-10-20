"""agent.kronos.forecaster -- pure metrics (hand-computed), horizon
truncation at the session close, seed determinism, and the torch-lazy
KronosPredictor wrapper exercised only through its torch-free short-circuit
path (empty input -- torch must never be imported by this test suite)."""

import math
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from agent.kronos.config import KronosConfig
from agent.kronos.forecaster import (
    HORIZONS_MINUTES,
    HorizonMetrics,
    KronosPredictor,
    build_forecast_horizons,
    compute_horizon_metrics,
    p_up_from_returns,
    resolve_pred_plan,
    seed_from_anchor,
)

_ET = ZoneInfo("America/New_York")


def _et_dt(y, m, d, hh, mm) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=_ET)


# ── seed_from_anchor ──────────────────────────────────────────────────────

def test_seed_from_anchor_is_deterministic():
    assert seed_from_anchor(1_700_000_000.0) == seed_from_anchor(1_700_000_000.0)


def test_seed_from_anchor_differs_across_anchors():
    assert seed_from_anchor(1_700_000_000.0) != seed_from_anchor(1_700_000_300.0)


def test_seed_from_anchor_is_within_valid_torch_range():
    s = seed_from_anchor(1_700_000_000.123)
    assert 0 <= s <= 0x7FFFFFFF


# ── resolve_pred_plan (horizon truncation at the session close) ─────────

def test_resolve_pred_plan_all_horizons_fit_mid_session():
    anchor = _et_dt(2026, 9, 22, 11, 0)
    close = _et_dt(2026, 9, 22, 16, 0)
    pred_len, horizons = resolve_pred_plan(anchor, close)
    assert pred_len == 12
    assert [label for label, _, _ in horizons] == ["15", "30", "60"]
    assert horizons == [("15", 15, 3), ("30", 30, 6), ("60", 60, 12)]


def test_resolve_pred_plan_only_15_and_30_fit():
    # 40 minutes to close: 15/30 fit, 60 does not.
    anchor = _et_dt(2026, 9, 22, 15, 20)
    close = _et_dt(2026, 9, 22, 16, 0)
    pred_len, horizons = resolve_pred_plan(anchor, close)
    assert [label for label, _, _ in horizons] == ["15", "30"]
    assert pred_len == 6


def test_resolve_pred_plan_none_fit_at_the_close():
    anchor = _et_dt(2026, 9, 22, 15, 55)
    close = _et_dt(2026, 9, 22, 16, 0)
    pred_len, horizons = resolve_pred_plan(anchor, close)
    assert horizons == []
    assert pred_len == 0


def test_resolve_pred_plan_exact_boundary_is_included():
    # Exactly 60 minutes to close -- target == close, must be included
    # (spec: "would cross the close" excludes it, landing exactly ON it does not).
    anchor = _et_dt(2026, 9, 22, 15, 0)
    close = _et_dt(2026, 9, 22, 16, 0)
    _, horizons = resolve_pred_plan(anchor, close)
    assert [label for label, _, _ in horizons] == ["15", "30", "60"]


def test_resolve_pred_plan_early_close_day_truncates_more():
    anchor = _et_dt(2026, 11, 27, 12, 20)
    close = _et_dt(2026, 11, 27, 13, 0)  # early-close day
    _, horizons = resolve_pred_plan(anchor, close)
    assert [label for label, _, _ in horizons] == ["15", "30"]


# ── p_up_from_returns ─────────────────────────────────────────────────────

def test_p_up_all_positive():
    assert p_up_from_returns([1.0, 2.0, 3.0]) == 1.0


def test_p_up_all_negative():
    assert p_up_from_returns([-1.0, -2.0]) == 0.0


def test_p_up_ties_count_half():
    # 1 up, 1 flat, 1 down -> (1 + 0.5) / 3
    assert p_up_from_returns([1.0, 0.0, -1.0]) == pytest.approx(0.5)


# ── compute_horizon_metrics (hand-computed) ──────────────────────────────

def test_compute_horizon_metrics_hand_computed():
    returns = [1.0, 2.0, 3.0, -1.0, -2.0]  # 3 up, 2 down -> p_up = 3/5 = 0.6
    m = compute_horizon_metrics(15, returns)
    assert isinstance(m, HorizonMetrics)
    assert m.minutes == 15
    assert m.p_up == pytest.approx(0.6)
    assert m.mean_return_pct == pytest.approx(sum(returns) / 5, abs=1e-6)
    assert m.median_return_pct == pytest.approx(1.0)  # sorted: -2,-1,1,2,3 -> median 1
    assert m.expected_abs_move_pct == pytest.approx(sum(abs(r) for r in returns) / 5, abs=1e-6)
    # q10/q90 are numpy linear-interpolation percentiles of the sorted data.
    assert m.q10_return_pct < m.median_return_pct < m.q90_return_pct


def test_compute_horizon_metrics_drops_non_finite_samples():
    m = compute_horizon_metrics(15, [1.0, float("nan"), float("inf"), 2.0])
    assert m is not None
    assert m.mean_return_pct == pytest.approx(1.5)


def test_compute_horizon_metrics_none_with_fewer_than_2_finite():
    assert compute_horizon_metrics(15, [1.0]) is None
    assert compute_horizon_metrics(15, []) is None
    assert compute_horizon_metrics(15, [float("nan"), 1.0]) is None


def test_horizon_metrics_as_dict_roundtrip():
    m = compute_horizon_metrics(30, [1.0, -1.0, 0.5])
    d = m.as_dict()
    assert d["minutes"] == 30
    assert set(d.keys()) == {"minutes", "p_up", "mean_return_pct", "median_return_pct", "q10_return_pct", "q90_return_pct", "expected_abs_move_pct"}


# ── build_forecast_horizons (integration of the two pieces above) ────────

def test_build_forecast_horizons_hand_computed():
    anchor_close = 100.0
    anchor = _et_dt(2026, 9, 22, 11, 0)
    close = _et_dt(2026, 9, 22, 16, 0)
    # 3 paths, pred_len=12 bars. bar index 2 (0-based) = 15-min horizon (3rd bar).
    paths = [
        [100 + i * 0.1 for i in range(1, 13)],   # steadily up
        [100 - i * 0.1 for i in range(1, 13)],   # steadily down
        [100.0] * 12,                              # flat
    ]
    horizons = build_forecast_horizons(anchor_close, anchor, close, paths)
    assert set(horizons.keys()) == {"15", "30", "60"}
    h15 = horizons["15"]
    # path1 close at bar3 = 100.3 -> +0.3%; path2 = 99.7 -> -0.3%; path3 = 100.0 -> 0%
    assert h15["p_up"] == pytest.approx(0.5)  # 1 up, 1 flat(half), 1 down
    assert h15["mean_return_pct"] == pytest.approx(0.0, abs=1e-9)


def test_build_forecast_horizons_empty_when_no_horizons_fit():
    anchor_close = 100.0
    anchor = _et_dt(2026, 9, 22, 15, 55)
    close = _et_dt(2026, 9, 22, 16, 0)
    horizons = build_forecast_horizons(anchor_close, anchor, close, [[100.0] * 12, [101.0] * 12])
    assert horizons == {}


def test_build_forecast_horizons_handles_non_positive_anchor_close():
    horizons = build_forecast_horizons(0.0, _et_dt(2026, 9, 22, 11, 0), _et_dt(2026, 9, 22, 16, 0), [[1.0] * 12])
    assert horizons == {}


def test_build_forecast_horizons_omits_horizon_with_insufficient_paths():
    # Only 1 usable path -> compute_horizon_metrics returns None -> horizon omitted, never a fabricated entry.
    anchor_close = 100.0
    anchor = _et_dt(2026, 9, 22, 11, 0)
    close = _et_dt(2026, 9, 22, 16, 0)
    horizons = build_forecast_horizons(anchor_close, anchor, close, [[101.0] * 12])
    assert horizons == {}


# ── KronosPredictor: torch-free short-circuit paths only ────────────────

def test_kronos_predictor_construction_never_touches_torch():
    KronosPredictor(KronosConfig())
    assert "torch" not in sys.modules


def test_kronos_predictor_predict_batch_empty_bars_short_circuits_without_torch():
    predictor = KronosPredictor(KronosConfig())
    out = predictor.predict_batch({}, pred_len=12, seed=1)
    assert out == {}
    assert "torch" not in sys.modules


def test_kronos_predictor_predict_batch_zero_pred_len_short_circuits_without_torch():
    predictor = KronosPredictor(KronosConfig())
    out = predictor.predict_batch({"SPY": [object()]}, pred_len=0, seed=1)
    assert out == {}
    assert "torch" not in sys.modules


def test_kronos_predictor_predict_batch_all_empty_symbol_lists_short_circuits():
    predictor = KronosPredictor(KronosConfig())
    out = predictor.predict_batch({"SPY": [], "QQQ": []}, pred_len=12, seed=1)
    assert out == {}
    assert "torch" not in sys.modules
