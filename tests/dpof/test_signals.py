"""Tests for agent/dpof/signals.py (pure)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent.dpof.models import (
    Direction,
    MarketSnapshot,
    PillarReading,
    Regime,
    UnderlyingSnapshot,
)
from agent.dpof.signals import (
    classify_regime,
    compute_composite,
    evaluate_signal,
    gex_factor,
    position_alignment,
    price_confirms,
    select_instrument,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def pillar(name: str, value: float | None, available: bool = True) -> PillarReading:
    if value is None:
        return PillarReading.missing(name, error="test_missing")
    return PillarReading(name=name, value=value, available=available)


def make_underlying(
    symbol: str = "SPY",
    nofa: float | None = None,
    premium: float | None = None,
    darkpool: float | None = None,
    gex_pillar: float | None = None,
    gex_regime: str | None = None,
    pin_risk: bool | None = None,
    last_price: float | None = None,
    vwap: float | None = None,
    ret_15m: float | None = None,
) -> UnderlyingSnapshot:
    pillars = {}
    if nofa is not None:
        pillars["nofa"] = pillar("nofa", nofa)
    if premium is not None:
        pillars["premium"] = pillar("premium", premium)
    if darkpool is not None:
        pillars["darkpool"] = pillar("darkpool", darkpool)
    if gex_pillar is not None:
        pillars["gex"] = pillar("gex", gex_pillar)
    return UnderlyingSnapshot(
        symbol=symbol,
        pillars=pillars,
        gex_regime=gex_regime,
        pin_risk=pin_risk,
        last_price=last_price,
        vwap=vwap,
        ret_15m=ret_15m,
    )


def make_snapshot(
    cfg,
    spy: UnderlyingSnapshot | None = None,
    qqq: UnderlyingSnapshot | None = None,
    vix: float | None = 16.0,
    vix_age_seconds: float | None = 30.0,
    data_healthy: bool = True,
    health_issues=None,
    risk_flags=None,
) -> MarketSnapshot:
    underlyings = {}
    if spy is not None:
        underlyings["SPY"] = spy
    if qqq is not None:
        underlyings["QQQ"] = qqq
    return MarketSnapshot(
        asof=1_700_000_000.0,
        underlyings=underlyings,
        vix=vix,
        vix_age_seconds=vix_age_seconds,
        risk_flags=list(risk_flags or []),
        data_healthy=data_healthy,
        health_issues=list(health_issues or []),
    )


# ─────────────────────────────────────────────────────────────────────────
# classify_regime
# ─────────────────────────────────────────────────────────────────────────


class TestClassifyRegime:
    def test_none_vix_is_unknown(self, cfg):
        assert classify_regime(None, 10.0, cfg) is Regime.UNKNOWN

    def test_none_age_is_unknown(self, cfg):
        assert classify_regime(20.0, None, cfg) is Regime.UNKNOWN

    def test_age_over_max_is_unknown(self, cfg):
        assert classify_regime(20.0, cfg.vix_max_age_seconds + 0.01, cfg) is Regime.UNKNOWN

    def test_age_exactly_at_max_is_not_unknown(self, cfg):
        # Spec: "age > max -> UNKNOWN" — exactly at max is still fresh.
        assert classify_regime(20.0, float(cfg.vix_max_age_seconds), cfg) is Regime.NORMAL

    def test_halt_exactly_at_threshold(self, cfg):
        assert classify_regime(cfg.vix_extreme_label_at, 10.0, cfg) is Regime.HALT

    def test_halt_above_threshold(self, cfg):
        assert classify_regime(cfg.vix_extreme_label_at + 5, 10.0, cfg) is Regime.HALT

    def test_choppy_just_below_halt(self, cfg):
        assert classify_regime(cfg.vix_extreme_label_at - 0.01, 10.0, cfg) is Regime.CHOPPY

    def test_choppy_exactly_at_threshold(self, cfg):
        assert classify_regime(cfg.vix_choppy_at, 10.0, cfg) is Regime.CHOPPY

    def test_normal_just_below_choppy(self, cfg):
        assert classify_regime(cfg.vix_choppy_at - 0.01, 10.0, cfg) is Regime.NORMAL

    def test_normal_exactly_at_trending_boundary(self, cfg):
        # trending is "< trending_below" so exactly at the boundary is NOT trending.
        assert classify_regime(cfg.vix_trending_below, 10.0, cfg) is Regime.NORMAL

    def test_trending_just_below_boundary(self, cfg):
        assert classify_regime(cfg.vix_trending_below - 0.01, 10.0, cfg) is Regime.TRENDING


# ─────────────────────────────────────────────────────────────────────────
# gex_factor
# ─────────────────────────────────────────────────────────────────────────


class TestGexFactor:
    def test_none_regime_is_none(self):
        assert gex_factor(None, None) is None

    @pytest.mark.parametrize("regime", ["negative_gamma", "NEGATIVE_GAMMA", "accelerating", "Accelerating"])
    def test_negative_gamma_family(self, regime):
        assert gex_factor(regime, None) == 0.5

    @pytest.mark.parametrize("regime", ["positive_gamma", "POSITIVE_GAMMA", "stabilizing", "Stabilizing"])
    def test_positive_gamma_family(self, regime):
        assert gex_factor(regime, None) == -0.3

    @pytest.mark.parametrize("regime", ["neutral", "unknown", "sideways", ""])
    def test_other_regimes_are_zero(self, regime):
        assert gex_factor(regime, None) == 0.0

    def test_pin_risk_true_halves_negative_gamma(self):
        assert gex_factor("negative_gamma", True) == pytest.approx(0.25)

    def test_pin_risk_true_halves_positive_gamma(self):
        assert gex_factor("positive_gamma", True) == pytest.approx(-0.15)

    def test_pin_risk_false_does_not_halve(self):
        assert gex_factor("negative_gamma", False) == 0.5

    def test_pin_risk_none_does_not_halve(self):
        assert gex_factor("negative_gamma", None) == 0.5


# ─────────────────────────────────────────────────────────────────────────
# compute_composite
# ─────────────────────────────────────────────────────────────────────────


class TestComputeComposite:
    def test_nofa_missing_returns_none(self, cfg):
        u = make_underlying(premium=0.5, darkpool=0.5, gex_regime="negative_gamma")
        composite, used, reasons = compute_composite(u, cfg)
        assert composite is None
        assert "nofa_unavailable" in reasons

    def test_min_pillars_boundary_below(self, cfg):
        # default min_pillars=3; nofa+premium only (gex unavailable, darkpool
        # unavailable) => count=2 < 3 -> None.
        cfg2 = replace(cfg, min_pillars=3)
        u = make_underlying(nofa=0.5, premium=0.5)  # no darkpool, no gex_regime
        composite, used, reasons = compute_composite(u, cfg2)
        assert composite is None
        assert any(r.startswith("insufficient_pillars:2<3") for r in reasons)

    def test_min_pillars_boundary_at_threshold(self, cfg):
        cfg2 = replace(cfg, min_pillars=3)
        u = make_underlying(nofa=0.5, premium=0.5, darkpool=0.5)  # 3 available, no gex
        composite, used, reasons = compute_composite(u, cfg2)
        assert composite is not None
        assert reasons == []

    def test_all_three_directional_no_gex(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        u = make_underlying(nofa=0.6, premium=0.2, darkpool=0.2)
        composite, used, reasons = compute_composite(u, cfg2)
        expected = (
            cfg2.w_nofa * 0.6 + cfg2.w_premium * 0.2 + cfg2.w_darkpool * 0.2
        ) / (cfg2.w_nofa + cfg2.w_premium + cfg2.w_darkpool)
        assert composite == pytest.approx(expected)
        assert "gex" not in used
        assert used == {"nofa": 0.6, "premium": 0.2, "darkpool": 0.2}

    def test_gex_amplifies_bullish_lean_via_negative_gamma(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        u_no_gex = make_underlying(nofa=0.6, premium=0.4, darkpool=0.4)
        u_with_gex = replace(u_no_gex, gex_regime="negative_gamma")  # factor +0.5, pre>0 -> amplifies
        base, _, _ = compute_composite(u_no_gex, cfg2)
        amplified, used, _ = compute_composite(u_with_gex, cfg2)
        assert used["gex"] == 0.5
        assert amplified > base

    def test_gex_dampens_bullish_lean_via_positive_gamma(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        u_no_gex = make_underlying(nofa=0.6, premium=0.4, darkpool=0.4)
        u_with_gex = replace(u_no_gex, gex_regime="positive_gamma")  # factor -0.3, pre>0 -> dampens
        base, _, _ = compute_composite(u_no_gex, cfg2)
        dampened, used, _ = compute_composite(u_with_gex, cfg2)
        assert used["gex"] == -0.3
        assert dampened < base

    def test_gex_amplifies_bearish_lean_via_negative_gamma(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        u_no_gex = make_underlying(nofa=-0.6, premium=-0.4, darkpool=-0.4)
        u_with_gex = replace(u_no_gex, gex_regime="negative_gamma")  # contrib = 0.5*sign(-)= -0.5
        base, _, _ = compute_composite(u_no_gex, cfg2)
        amplified, _, _ = compute_composite(u_with_gex, cfg2)
        assert amplified < base  # more negative = amplified bearish lean

    def test_gex_dampens_bearish_lean_via_positive_gamma(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        u_no_gex = make_underlying(nofa=-0.6, premium=-0.4, darkpool=-0.4)
        u_with_gex = replace(u_no_gex, gex_regime="positive_gamma")  # contrib = -0.3*sign(-) = +0.3
        base, _, _ = compute_composite(u_no_gex, cfg2)
        dampened, _, _ = compute_composite(u_with_gex, cfg2)
        assert dampened > base  # pulled toward zero = dampened bearish lean

    def test_gex_never_initiates_direction_when_directional_pillars_are_flat(self, cfg):
        # GEX is direction-agnostic (design doc §4): with pre == 0 (no directional
        # lean at all — sign(pre) == 0), GEX's contribution is exactly 0 regardless
        # of regime, so it can never be the sole source of a directional composite.
        cfg2 = replace(cfg, min_pillars=1)
        u = make_underlying(nofa=0.0, premium=0.0, darkpool=0.0, gex_regime="negative_gamma")
        composite, used, _ = compute_composite(u, cfg2)
        assert composite == 0.0
        assert used["gex"] == 0.5  # the factor is available/nonzero...
        # ...but contributes nothing because sign(pre) == 0 (gex never picks a side).

    def test_gex_with_default_weight_only_dampens_does_not_dominate(self, cfg):
        # Under the DEFAULT config weight (w_gex=0.15, well below the combined
        # 0.85 directional weight), a comfortably-positive pre stays positive
        # after the strongest possible dampening (positive_gamma, factor -0.3).
        u_no_gex = make_underlying(nofa=0.5, premium=0.5, darkpool=0.5)
        u_with_gex = replace(u_no_gex, gex_regime="positive_gamma")
        base, _, _ = compute_composite(u_no_gex, cfg)
        dampened, _, _ = compute_composite(u_with_gex, cfg)
        assert 0 < dampened < base

    def test_gex_pillar_overrides_regime_calc(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        u = make_underlying(nofa=0.6, gex_pillar=0.9, gex_regime="positive_gamma")
        composite, used, _ = compute_composite(u, cfg2)
        assert used["gex"] == 0.9  # pillar value used verbatim, not gex_factor("positive_gamma")

    def test_gex_pillar_unavailable_falls_back_to_regime(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        u = make_underlying(nofa=0.6, gex_regime="negative_gamma")
        composite, used, _ = compute_composite(u, cfg2)
        assert used["gex"] == 0.5

    def test_clamped_to_range(self, cfg):
        cfg2 = replace(cfg, min_pillars=1, w_nofa=1.0, w_premium=0.0, w_darkpool=0.0, w_gex=0.0)
        u = make_underlying(nofa=5.0)  # malformed out-of-domain input
        composite, _, _ = compute_composite(u, cfg2)
        assert composite == 1.0

    def test_gex_only_with_nofa_counts_toward_min_pillars(self, cfg):
        cfg2 = replace(cfg, min_pillars=2)
        u = make_underlying(nofa=0.5, gex_regime="negative_gamma")  # nofa + gex = 2 available
        composite, used, reasons = compute_composite(u, cfg2)
        assert composite is not None
        assert reasons == []


# ─────────────────────────────────────────────────────────────────────────
# price_confirms
# ─────────────────────────────────────────────────────────────────────────


class TestPriceConfirms:
    def test_bullish_confirmed(self):
        u = make_underlying(last_price=101.0, vwap=100.0, ret_15m=0.01)
        ok, reasons = price_confirms(u, Direction.BULLISH)
        assert ok is True
        assert reasons == []

    def test_bullish_below_vwap_not_confirmed(self):
        u = make_underlying(last_price=99.0, vwap=100.0, ret_15m=0.01)
        ok, reasons = price_confirms(u, Direction.BULLISH)
        assert ok is False
        assert "price_below_vwap" in reasons

    def test_bullish_non_positive_momentum_not_confirmed(self):
        u = make_underlying(last_price=101.0, vwap=100.0, ret_15m=0.0)
        ok, reasons = price_confirms(u, Direction.BULLISH)
        assert ok is False
        assert "momentum_not_positive" in reasons

    def test_bearish_confirmed(self):
        u = make_underlying(last_price=99.0, vwap=100.0, ret_15m=-0.01)
        ok, reasons = price_confirms(u, Direction.BEARISH)
        assert ok is True
        assert reasons == []

    def test_bearish_above_vwap_not_confirmed(self):
        u = make_underlying(last_price=101.0, vwap=100.0, ret_15m=-0.01)
        ok, reasons = price_confirms(u, Direction.BEARISH)
        assert ok is False
        assert "price_above_vwap" in reasons

    def test_bearish_non_negative_momentum_not_confirmed(self):
        u = make_underlying(last_price=99.0, vwap=100.0, ret_15m=0.0)
        ok, reasons = price_confirms(u, Direction.BEARISH)
        assert ok is False
        assert "momentum_not_negative" in reasons

    @pytest.mark.parametrize("field", ["last_price", "vwap", "ret_15m"])
    def test_missing_data(self, field):
        kwargs = dict(last_price=101.0, vwap=100.0, ret_15m=0.01)
        kwargs[field] = None
        u = make_underlying(**kwargs)
        ok, reasons = price_confirms(u, Direction.BULLISH)
        assert ok is False
        assert reasons == ["price_data_missing"]

    def test_neutral_direction(self):
        u = make_underlying(last_price=101.0, vwap=100.0, ret_15m=0.01)
        ok, reasons = price_confirms(u, Direction.NEUTRAL)
        assert ok is False
        assert reasons == ["direction_neutral"]


# ─────────────────────────────────────────────────────────────────────────
# select_instrument
# ─────────────────────────────────────────────────────────────────────────


class TestSelectInstrument:
    def test_single_underlying_configured(self, cfg):
        cfg2 = replace(cfg, underlyings=("SPY",))
        snap = make_snapshot(cfg2, spy=make_underlying(nofa=0.5))
        symbol, reasons = select_instrument(snap, 1, cfg2)
        assert symbol == "SPY"
        assert reasons == ["single_underlying_configured"]

    def test_primary_nofa_unavailable(self, cfg):
        snap = make_snapshot(cfg, spy=make_underlying(), qqq=make_underlying(nofa=0.9))
        symbol, reasons = select_instrument(snap, 1, cfg)
        assert symbol == "SPY"
        assert "SPY_nofa_unavailable" in reasons

    def test_secondary_nofa_unavailable(self, cfg):
        snap = make_snapshot(cfg, spy=make_underlying(nofa=0.5), qqq=make_underlying())
        symbol, reasons = select_instrument(snap, 1, cfg)
        assert symbol == "SPY"
        assert "QQQ_nofa_unavailable" in reasons

    def test_divergence_selects_secondary(self, cfg):
        # ratio default 1.5; |nofa_qqq|=0.9 >= 1.5*0.5=0.75, same sign as composite (1)
        snap = make_snapshot(cfg, spy=make_underlying(nofa=0.5), qqq=make_underlying(nofa=0.9))
        symbol, reasons = select_instrument(snap, 1, cfg)
        assert symbol == "QQQ"
        assert reasons == ["divergence_selected:QQQ"]

    def test_divergence_ratio_boundary_exact(self, cfg):
        # exactly ratio*|nofa1| -> still selects (>=)
        snap = make_snapshot(cfg, spy=make_underlying(nofa=0.5), qqq=make_underlying(nofa=0.75))
        symbol, _ = select_instrument(snap, 1, cfg)
        assert symbol == "QQQ"

    def test_divergence_just_below_ratio_stays_primary(self, cfg):
        snap = make_snapshot(cfg, spy=make_underlying(nofa=0.5), qqq=make_underlying(nofa=0.749))
        symbol, reasons = select_instrument(snap, 1, cfg)
        assert symbol == "SPY"
        assert reasons == ["primary_selected:SPY"]

    def test_divergence_disagreeing_sign_stays_primary(self, cfg):
        # magnitude qualifies but sign of nofa2 disagrees with composite_sign
        snap = make_snapshot(cfg, spy=make_underlying(nofa=0.5), qqq=make_underlying(nofa=-0.9))
        symbol, reasons = select_instrument(snap, 1, cfg)
        assert symbol == "SPY"
        assert reasons == ["primary_selected:SPY"]


# ─────────────────────────────────────────────────────────────────────────
# evaluate_signal
# ─────────────────────────────────────────────────────────────────────────


class TestEvaluateSignal:
    def test_data_unhealthy_vetoes(self, cfg):
        u = make_underlying(nofa=0.6, premium=0.5, darkpool=0.5, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg, spy=u, data_healthy=False, health_issues=["gate_source_unavailable:econ_calendar"])
        sig = evaluate_signal(snap, cfg)
        assert "data_unhealthy" in sig.vetoes
        assert "gate_source_unavailable:econ_calendar" in sig.vetoes
        assert not sig.actionable

    def test_risk_flags_no_longer_vetoed_but_present_as_context(self, cfg):
        """Task 1 (2026-09-25, operator directive): econ blackout / breaking
        news / vol regime are DECISION CONTEXT now, never a veto. The
        underlying (strong composite, price confirmed, no DP veto) still
        nominates -- the risk_flags are surfaced as diagnosable `reasons`
        instead, never in `vetoes`."""
        u = make_underlying(nofa=0.6, premium=0.5, darkpool=0.5, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg, spy=u, risk_flags=["breaking_news", "econ_blackout:FOMC", "vol_regime:CRISIS"])
        sig = evaluate_signal(snap, cfg)
        assert "risk_flag:breaking_news" not in sig.vetoes
        assert "risk_flag:econ_blackout:FOMC" not in sig.vetoes
        assert "risk_flag:vol_regime:CRISIS" not in sig.vetoes
        assert sig.vetoes == []
        assert "risk_flag:breaking_news" in sig.reasons
        assert "risk_flag:econ_blackout:FOMC" in sig.reasons
        assert "risk_flag:vol_regime:CRISIS" in sig.reasons
        assert sig.actionable

    def test_composite_none_is_neutral(self, cfg):
        u = make_underlying(premium=0.5, darkpool=0.5)  # no nofa
        snap = make_snapshot(cfg, spy=u)
        sig = evaluate_signal(snap, cfg)
        assert sig.direction is Direction.NEUTRAL
        assert sig.strength == "none"
        assert sig.composite is None
        assert "nofa_unavailable" in sig.reasons
        assert not sig.actionable

    def test_missing_primary_underlying(self, cfg):
        snap = make_snapshot(cfg)  # no SPY entry at all
        sig = evaluate_signal(snap, cfg)
        assert sig.composite is None
        assert "underlying_snapshot_missing" in sig.reasons
        assert sig.symbol == "SPY"

    def test_regime_extreme_with_strong_composite_nominates(self, cfg):
        """Task 1 (2026-09-25, operator directive): VIX HALT/"EXTREME" no
        longer vetoes outright. With a STRONG composite it nominates exactly
        like CHOPPY already did -- there is no "regime_halt" veto any more."""
        u = make_underlying(nofa=0.6, premium=0.5, darkpool=0.5, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg, spy=u, vix=cfg.vix_extreme_label_at)
        sig = evaluate_signal(snap, cfg)
        assert "regime_halt" not in sig.vetoes
        assert sig.vetoes == []
        assert sig.regime is Regime.HALT
        assert sig.strength == "strong"
        assert sig.actionable

    def test_regime_extreme_with_only_normal_composite_requires_strong(self, cfg):
        """The strong-only rule that already applied to CHOPPY/UNKNOWN now
        applies to EXTREME too (Task 1: "apply the same rule to EXTREME")."""
        u = make_underlying(nofa=0.2, premium=0.2, darkpool=0.2, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg, spy=u, vix=cfg.vix_extreme_label_at)
        sig = evaluate_signal(snap, cfg)
        assert sig.regime is Regime.HALT
        assert sig.strength == "none"
        assert "regime_requires_strong" in sig.reasons
        assert not sig.actionable
        assert sig.vetoes == []  # never a veto -- just fails the strength gate, same as CHOPPY

    def test_strength_strong_at_exact_threshold(self, cfg):
        # nofa alone, weight 1.0, so composite == nofa exactly.
        cfg2 = replace(cfg, min_pillars=1, w_nofa=1.0, w_premium=0.0, w_darkpool=0.0, w_gex=0.0)
        u = make_underlying(nofa=cfg2.threshold_strong, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert sig.strength == "strong"

    def test_strength_normal_at_exact_threshold(self, cfg):
        cfg2 = replace(cfg, min_pillars=1, w_nofa=1.0, w_premium=0.0, w_darkpool=0.0, w_gex=0.0)
        u = make_underlying(nofa=cfg2.threshold_normal, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert sig.strength == "normal"

    def test_strength_none_just_below_normal(self, cfg):
        cfg2 = replace(cfg, min_pillars=1, w_nofa=1.0, w_premium=0.0, w_darkpool=0.0, w_gex=0.0)
        u = make_underlying(nofa=cfg2.threshold_normal - 0.001)
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert sig.strength == "none"
        assert sig.direction is Direction.NEUTRAL
        assert "composite_below_normal_threshold" in sig.reasons

    def test_choppy_regime_downgrades_normal_to_none(self, cfg):
        cfg2 = replace(cfg, min_pillars=1, w_nofa=1.0, w_premium=0.0, w_darkpool=0.0, w_gex=0.0)
        u = make_underlying(nofa=cfg2.threshold_normal, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg2, spy=u, vix=cfg2.vix_choppy_at)
        sig = evaluate_signal(snap, cfg2)
        assert sig.regime is Regime.CHOPPY
        assert sig.strength == "none"
        assert "regime_requires_strong" in sig.reasons
        assert sig.direction is Direction.NEUTRAL

    def test_choppy_regime_keeps_strong(self, cfg):
        cfg2 = replace(cfg, min_pillars=1, w_nofa=1.0, w_premium=0.0, w_darkpool=0.0, w_gex=0.0)
        u = make_underlying(nofa=cfg2.threshold_strong, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg2, spy=u, vix=cfg2.vix_choppy_at)
        sig = evaluate_signal(snap, cfg2)
        assert sig.strength == "strong"
        assert sig.direction is Direction.BULLISH
        assert sig.actionable

    def test_unknown_regime_requires_strong_too(self, cfg):
        cfg2 = replace(cfg, min_pillars=1, w_nofa=1.0, w_premium=0.0, w_darkpool=0.0, w_gex=0.0)
        u = make_underlying(nofa=cfg2.threshold_normal, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg2, spy=u, vix=None, vix_age_seconds=None)
        sig = evaluate_signal(snap, cfg2)
        assert sig.regime is Regime.UNKNOWN
        assert sig.strength == "none"
        assert "regime_requires_strong" in sig.reasons

    def test_dp_veto_boundary_exact(self, cfg):
        # composite positive (bullish); darkpool*sign(c) <= -0.30 triggers.
        cfg2 = replace(cfg, min_pillars=1, dp_veto_threshold=0.30)
        u = make_underlying(nofa=0.6, darkpool=-0.30, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert "dp_veto" in sig.vetoes

    def test_dp_veto_just_inside_not_triggered(self, cfg):
        cfg2 = replace(cfg, min_pillars=1, dp_veto_threshold=0.30)
        u = make_underlying(nofa=0.6, darkpool=-0.2999, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert "dp_veto" not in sig.vetoes

    def test_dp_veto_skipped_when_darkpool_unavailable(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        u = make_underlying(nofa=0.6, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert "dp_veto" not in sig.vetoes

    def test_price_confirmation_missing_data_vetoes(self, cfg):
        cfg2 = replace(cfg, min_pillars=1, require_price_confirmation=True)
        u = make_underlying(nofa=0.6)  # no price data
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert "price_not_confirmed" in sig.vetoes
        assert "price_data_missing" in sig.reasons
        assert sig.price_confirmed is False

    def test_price_confirmation_disabled_no_veto(self, cfg):
        cfg2 = replace(cfg, min_pillars=1, require_price_confirmation=False)
        u = make_underlying(nofa=0.6)  # no price data — would fail if checked
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert "price_not_confirmed" not in sig.vetoes

    def test_fully_actionable_bullish_signal(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        u = make_underlying(
            nofa=0.6, premium=0.5, darkpool=0.5, last_price=101, vwap=100, ret_15m=0.01
        )
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert sig.actionable
        assert sig.direction is Direction.BULLISH
        assert sig.symbol == "SPY"
        assert sig.price_confirmed is True
        assert 0.0 < sig.conviction <= 1.0

    def test_fully_actionable_bearish_signal(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        u = make_underlying(
            nofa=-0.6, premium=-0.5, darkpool=-0.5, last_price=99, vwap=100, ret_15m=-0.01
        )
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert sig.actionable
        assert sig.direction is Direction.BEARISH

    def test_conviction_capped_at_one(self, cfg):
        cfg2 = replace(cfg, min_pillars=1, w_nofa=1.0, w_premium=0.0, w_darkpool=0.0, w_gex=0.0)
        u = make_underlying(nofa=1.0, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg2, spy=u)
        sig = evaluate_signal(snap, cfg2)
        assert sig.conviction == 1.0

    def test_qqq_selected_when_diverging(self, cfg):
        cfg2 = replace(cfg, min_pillars=1)
        spy = make_underlying(nofa=0.5, premium=0.5, darkpool=0.5)
        qqq = make_underlying(nofa=0.9, last_price=101, vwap=100, ret_15m=0.01)
        snap = make_snapshot(cfg2, spy=spy, qqq=qqq)
        sig = evaluate_signal(snap, cfg2)
        assert sig.symbol == "QQQ"
        # price confirmation runs on the CHOSEN instrument (QQQ), which is confirmed.
        assert sig.price_confirmed is True


# ─────────────────────────────────────────────────────────────────────────
# position_alignment
# ─────────────────────────────────────────────────────────────────────────


class TestPositionAlignment:
    def test_neutral_direction_is_all_neutral(self):
        ctx = {"composite": 0.8, "ret_15m": 0.02, "last": 105.0, "vwap": 100.0}
        out = position_alignment(ctx, Direction.NEUTRAL)
        assert out == {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"}

    def test_empty_mapping_is_all_neutral(self):
        out = position_alignment({}, Direction.BULLISH)
        assert out == {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"}

    def test_flow_aligned_and_against(self):
        aligned = position_alignment({"composite": 0.5}, Direction.BULLISH)
        assert aligned["flow"] == "aligned"
        against = position_alignment({"composite": 0.5}, Direction.BEARISH)
        assert against["flow"] == "against"
        against2 = position_alignment({"composite": -0.5}, Direction.BULLISH)
        assert against2["flow"] == "against"

    def test_flow_neutral_band(self):
        # Just inside the neutral band -> neutral, not aligned/against.
        just_inside = position_alignment({"composite": 0.049}, Direction.BULLISH)
        assert just_inside["flow"] == "neutral"
        # At/over the band edge -> a real read.
        at_edge = position_alignment({"composite": 0.05}, Direction.BULLISH)
        assert at_edge["flow"] == "aligned"

    def test_momentum_aligned_against_and_neutral_band(self):
        aligned = position_alignment({"ret_15m": 0.01}, Direction.BULLISH)
        assert aligned["momentum"] == "aligned"
        against = position_alignment({"ret_15m": -0.01}, Direction.BULLISH)
        assert against["momentum"] == "against"
        flat = position_alignment({"ret_15m": 0.0001}, Direction.BULLISH)
        assert flat["momentum"] == "neutral"

    def test_vwap_aligned_against_and_neutral_band_using_last(self):
        above = position_alignment({"last": 101.0, "vwap": 100.0}, Direction.BULLISH)
        assert above["vwap"] == "aligned"
        below = position_alignment({"last": 99.0, "vwap": 100.0}, Direction.BULLISH)
        assert below["vwap"] == "against"
        flat = position_alignment({"last": 100.02, "vwap": 100.0}, Direction.BULLISH)  # 0.02% move
        assert flat["vwap"] == "neutral"

    def test_vwap_falls_back_to_last_price_field(self):
        # UnderlyingSnapshot-shaped input uses "last_price", not "last".
        out = position_alignment({"last_price": 101.0, "vwap": 100.0}, Direction.BULLISH)
        assert out["vwap"] == "aligned"

    def test_bearish_direction_flips_alignment_sense(self):
        out = position_alignment({"composite": -0.5, "ret_15m": -0.01, "last": 99.0, "vwap": 100.0}, Direction.BEARISH)
        assert out == {"flow": "aligned", "momentum": "aligned", "vwap": "aligned"}

    def test_missing_fields_are_neutral_not_raising(self):
        out = position_alignment({"composite": None, "ret_15m": None, "vwap": None}, Direction.BULLISH)
        assert out == {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"}

    def test_non_numeric_values_never_raise(self):
        out = position_alignment(
            {"composite": "not_a_number", "ret_15m": [], "last": "x", "vwap": {}}, Direction.BULLISH
        )
        assert out == {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"}

    def test_vwap_zero_never_zero_divides(self):
        out = position_alignment({"last": 5.0, "vwap": 0.0}, Direction.BULLISH)
        assert out["vwap"] == "neutral"

    def test_accepts_underlyingsnapshot_object_via_attribute_access(self):
        u = make_underlying(last_price=101.0, vwap=100.0, ret_15m=0.01)
        out = position_alignment(u, Direction.BULLISH)
        # UnderlyingSnapshot has no "composite" attribute -> flow stays neutral;
        # momentum/vwap read fine via getattr().
        assert out["flow"] == "neutral"
        assert out["momentum"] == "aligned"
        assert out["vwap"] == "aligned"
