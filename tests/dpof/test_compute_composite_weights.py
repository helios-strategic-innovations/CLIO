"""Optional per-pillar weights in compute_composite: regression, overrides, fallbacks, reasons."""
import inspect
import random

import pytest

from agent.dpof.config import DPOFConfig
from agent.dpof.models import PillarReading, UnderlyingSnapshot
from agent.dpof.signals import KRONOS_PILLAR, _DIRECTIONAL_PILLARS, _sign, compute_composite, gex_factor

STATIC = {"nofa": 0.40, "premium": 0.25, "darkpool": 0.20, KRONOS_PILLAR: 0.15}


def _old_compute_composite(u, cfg):
    """Transcribed verbatim from HEAD agent/dpof/signals.py compute_composite (pre-B2a)."""
    avail = u.available_pillars()
    directional = {k: v for k, v in avail.items() if k in _DIRECTIONAL_PILLARS}
    if getattr(cfg, "kronos_pillar", "off") != "live":
        directional.pop(KRONOS_PILLAR, None)

    gex_available = False
    gex_val = 0.0
    if "gex" in avail:
        gex_available = True
        gex_val = avail["gex"]
    else:
        factor = gex_factor(u.gex_regime, u.pin_risk)
        if factor is not None:
            gex_available = True
            gex_val = factor

    used_values = dict(directional)
    if gex_available:
        used_values["gex"] = gex_val

    reasons = []

    if "nofa" not in directional:
        reasons.append("nofa_unavailable")
        return None, used_values, reasons

    count_available = sum(1 for k in directional if k in ("nofa", "premium", "darkpool")) + (1 if gex_available else 0)
    if count_available < cfg.min_pillars:
        reasons.append(f"insufficient_pillars:{count_available}<{cfg.min_pillars}")
        return None, used_values, reasons

    weights = {"nofa": cfg.w_nofa, "premium": cfg.w_premium, "darkpool": cfg.w_darkpool,
               KRONOS_PILLAR: getattr(cfg, "w_kronos", 0.0)}
    sum_w = sum(weights[k] for k in directional)
    sum_wp = sum(weights[k] * directional[k] for k in directional)
    if sum_w == 0:
        reasons.append("zero_weight_sum")
        return None, used_values, reasons

    pre = sum_wp / sum_w
    contrib = gex_val * _sign(pre) if gex_available else 0.0

    numerator = sum_wp + (cfg.w_gex * contrib if gex_available else 0.0)
    denominator = sum_w + (cfg.w_gex if gex_available else 0.0)

    composite = numerator / denominator
    composite = max(-1.0, min(1.0, composite))
    return composite, used_values, reasons


def _snap(values, gex_regime=None, pin_risk=None):
    """Build a snapshot; a None value means the pillar is unavailable."""
    pillars = {}
    for name, v in values.items():
        if v is None:
            pillars[name] = PillarReading.missing(name, "test")
        else:
            pillars[name] = PillarReading(name=name, value=v, available=True)
    return UnderlyingSnapshot(symbol="SPY", pillars=pillars, gex_regime=gex_regime, pin_risk=pin_risk)


def _hand_case(pillar_weights=None):
    """nofa=0.5, premium=0.2, darkpool=-0.4, negative_gamma (gex factor 0.5), no kronos."""
    return _snap({"nofa": 0.5, "premium": 0.2, "darkpool": -0.4}, gex_regime="negative_gamma", pin_risk=False)


def test_regression_none_weights_matches_old_formula_over_random_cases():
    """With no override the new compute_composite equals the transcribed HEAD formula on 80 seeded cases."""
    rng = random.Random(20261010)
    regimes = [None, "negative_gamma", "positive_gamma", "neutral", "accelerating", "stabilizing"]
    for _ in range(80):
        vals = {}
        for name in ("nofa", "premium", "darkpool", KRONOS_PILLAR):
            r = rng.random()
            if r < 0.25:
                vals[name] = None
            elif r < 0.35:
                vals[name] = 0.0
            else:
                vals[name] = rng.uniform(-1.0, 1.0)
        u = _snap(vals, gex_regime=rng.choice(regimes), pin_risk=rng.choice([None, True, False]))
        cfg = DPOFConfig(kronos_pillar=rng.choice(["live", "off"]), min_pillars=rng.choice([1, 2, 3, 4]))
        assert compute_composite(u, cfg) == _old_compute_composite(u, cfg)
        assert compute_composite(u, cfg, weights=None) == _old_compute_composite(u, cfg)


def test_override_equal_to_static_gives_same_composite_and_adds_reason():
    """An override equal to the static weights reproduces the composite and adds weights:dynamic."""
    cfg = DPOFConfig()
    u = _hand_case()
    base_c, base_used, base_r = _old_compute_composite(u, cfg)
    c, used, reasons = compute_composite(u, cfg, weights=dict(STATIC))
    assert c == pytest.approx(base_c, abs=1e-12)
    assert used == base_used
    assert reasons == ["weights:dynamic"]


def test_premium_zeroed_and_others_scaled_matches_hand_computed_value():
    """Premium 0.0 with nofa/darkpool x1/0.75 gives 47/190 for the hand case."""
    # nofa 8/15, darkpool 4/15 -> sum_w 0.8, sum_wp 0.16; gex 0.5*sign(0.2)*0.15 -> (0.235)/(0.95)
    cfg = DPOFConfig()
    u = _hand_case()
    override = {"nofa": 0.40 / 0.75, "premium": 0.0, "darkpool": 0.20 / 0.75}
    c, _used, reasons = compute_composite(u, cfg, weights=override)
    assert c == pytest.approx(47.0 / 190.0, abs=1e-12)
    assert reasons == ["weights:dynamic"]


def test_missing_key_uses_static_weight_for_that_key():
    """A Mapping without darkpool uses the static darkpool weight."""
    cfg = DPOFConfig()
    u = _hand_case()
    static_c = _old_compute_composite(u, cfg)[0]
    c, _used, _r = compute_composite(u, cfg, weights={"nofa": 0.40, "premium": 0.25})
    assert c == pytest.approx(static_c, abs=1e-12)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), float("-inf"), True, False,
                                 "0.3", None, 10 ** 400])
def test_bad_darkpool_value_falls_back_to_static(bad):
    """Negative, NaN, inf, bool, string, None and oversized ints fall back to the static darkpool weight."""
    cfg = DPOFConfig()
    u = _hand_case()
    static_c = _old_compute_composite(u, cfg)[0]
    c, _used, _r = compute_composite(u, cfg, weights={"nofa": 0.40, "premium": 0.25, "darkpool": bad})
    assert c == pytest.approx(static_c, abs=1e-12)


@pytest.mark.parametrize("bad_override", [[0.4, 0.25, 0.2], "abc", 5])
def test_non_mapping_override_is_ignored(bad_override):
    """A list, string or int override is ignored: static composite and no weights:dynamic reason."""
    cfg = DPOFConfig()
    u = _hand_case()
    base = _old_compute_composite(u, cfg)
    assert compute_composite(u, cfg, weights=bad_override) == base
    u.pillar_weights = bad_override
    assert compute_composite(u, cfg) == base


def test_snapshot_field_used_when_weights_none_and_explicit_arg_wins():
    """u.pillar_weights applies when weights is None; an explicit weights argument overrides it."""
    cfg = DPOFConfig()
    u = _hand_case()
    u.pillar_weights = {"nofa": 0.40 / 0.75, "premium": 0.0, "darkpool": 0.20 / 0.75}
    c, _used, reasons = compute_composite(u, cfg)
    assert c == pytest.approx(47.0 / 190.0, abs=1e-12)
    assert reasons == ["weights:dynamic"]
    c2, _used2, reasons2 = compute_composite(u, cfg, weights=dict(STATIC))
    assert c2 == pytest.approx(_old_compute_composite(u, cfg)[0], abs=1e-12)
    assert reasons2 == ["weights:dynamic"]


def test_all_zero_available_weights_hit_zero_weight_sum():
    """Zero weights on every available directional pillar return None via zero_weight_sum."""
    cfg = DPOFConfig()
    u = _hand_case()
    c, _used, reasons = compute_composite(u, cfg, weights={"nofa": 0.0, "premium": 0.0, "darkpool": 0.0})
    assert c is None
    assert "zero_weight_sum" in reasons
    assert "weights:dynamic" not in reasons


def test_no_dynamic_reason_on_early_returns():
    """nofa_unavailable and insufficient_pillars early returns never carry weights:dynamic."""
    cfg = DPOFConfig()
    no_nofa = _snap({"premium": 0.3, "darkpool": 0.2}, gex_regime="negative_gamma", pin_risk=False)
    c, _used, reasons = compute_composite(no_nofa, cfg, weights=dict(STATIC))
    assert c is None
    assert reasons == ["nofa_unavailable"]
    thin = _snap({"nofa": 0.3})
    c2, _used2, reasons2 = compute_composite(thin, DPOFConfig(min_pillars=3), weights=dict(STATIC))
    assert c2 is None
    assert all(r != "weights:dynamic" for r in reasons2)
    assert any(r.startswith("insufficient_pillars") for r in reasons2)


def test_weights_parameter_is_last_and_keyword_capable():
    """compute_composite(u, cfg, weights=None): weights is the last parameter with default None."""
    params = list(inspect.signature(compute_composite).parameters.values())
    assert [p.name for p in params] == ["u", "cfg", "weights"]
    assert params[-1].default is None
    assert params[-1].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert compute_composite(_hand_case(), DPOFConfig())[0] is not None


@pytest.mark.parametrize("override", [{}, {"nofa": None, "premium": -1.0}, {"x": 1.0}, {"nofa": float("nan")}])
def test_override_with_no_usable_key_is_not_dynamic_and_equals_static(override):
    """A Mapping override with no usable weight uses the static weights and adds NO 'weights:dynamic' reason."""
    cfg = DPOFConfig()
    u = _hand_case()
    c, used, reasons = compute_composite(u, cfg, weights=override)
    old_c, old_used, old_reasons = _old_compute_composite(u, cfg)
    assert c == pytest.approx(old_c, abs=1e-12)
    assert used == old_used
    assert reasons == old_reasons
    assert "weights:dynamic" not in reasons


def test_any_mapping_type_is_accepted_as_an_override():
    """A read-only Mapping (not a dict) is honoured exactly like a dict."""
    import types
    cfg = DPOFConfig()
    u = _hand_case()
    raw = {"nofa": 0.40 / 0.75, "premium": 0.0, "darkpool": 0.20 / 0.75}
    c_dict, _u1, r_dict = compute_composite(u, cfg, weights=raw)
    c_proxy, _u2, r_proxy = compute_composite(u, cfg, weights=types.MappingProxyType(raw))
    assert c_proxy == pytest.approx(c_dict, abs=1e-12)
    assert r_proxy == r_dict == ["weights:dynamic"]
