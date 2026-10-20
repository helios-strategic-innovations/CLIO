"""Legacy darkpool 'flat = unavailable' flag (CLIO_DPOF_DARKPOOL_FLAT_UNAVAILABLE, S215 W7A / R-36).

Default OFF: every pillar value, availability flag and composite must stay
byte-identical to the pre-flag code (pinned by the golden digest below).
Flag ON: a legacy (non-NBBO) darkpool reading with |v| < 1e-3 is excluded from
the composite (renormalised over the remaining pillars) unless excluding it
would drop count_available below min_pillars.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random

import pytest

from agent.dpof.config import DPOFConfig
from agent.dpof.models import MarketSnapshot, PillarReading, UnderlyingSnapshot
from agent.dpof.signals import compute_composite, evaluate_signal

# sha256 over the repr of 360 deterministic synthetic cycles, recorded on the
# base commit 2dc0dca (before the flag existed). Flag-off code must reproduce it.
GOLDEN_DIGEST_FLAG_OFF = "4fbc8f4a4d7973af464a16bbe67723b1add0122bae38458460c0f0bd43db16e6"

_DP_VALUES = [0.0, 0.0005, -0.0005, 0.002, -0.002, 0.00099, 0.001, -0.3, 0.5, 0.9, -0.7, None]
_NOFA_VALUES = [0.4, -0.35, 0.05, -0.02, 0.0, None]
_PREM_VALUES = [0.3, -0.25, 0.0, None]
_GEX_REGIMES = ["negative_gamma", "positive_gamma", "neutral", None]


def _pillar(name, value):
    if value is None:
        return PillarReading.missing(name, error="synthetic_missing")
    return PillarReading(name=name, value=value, available=True,
                         source=f"uw:/{name}/SPY")


def _case(rng: random.Random):
    pillars = {}
    for name, choices in (("nofa", _NOFA_VALUES), ("premium", _PREM_VALUES), ("darkpool", _DP_VALUES)):
        v = rng.choice(choices)
        if v is not None or rng.random() < 0.5:
            pillars[name] = _pillar(name, v)
    if rng.random() < 0.3:
        pillars["gex"] = _pillar("gex", rng.choice([0.5, -0.3, 0.0, None]))
    weights = None
    if rng.random() < 0.3:
        weights = {k: rng.choice([0.1, 0.3, 0.0, None]) for k in ("nofa", "premium", "darkpool", "kronos")}
    u = UnderlyingSnapshot(
        symbol="SPY",
        pillars=pillars,
        gex_regime=rng.choice(_GEX_REGIMES),
        pin_risk=rng.choice([True, False, None]),
        last_price=rng.choice([500.0, 501.0, 499.0, None]),
        vwap=rng.choice([500.0, 500.5, None]),
        ret_15m=rng.choice([0.001, -0.001, 0.0, None]),
        pillar_weights=weights,
    )
    return u


def _cfgs():
    base = DPOFConfig()
    out = []
    for min_p in (2, 3, 4):
        for nbbo in (False, True):
            out.append(dataclasses.replace(base, min_pillars=min_p, darkpool_nbbo=nbbo))
    return out


def golden_lines(n_cycles: int = 360, seed: int = 20261012) -> list[str]:
    """Deterministic synthetic cycles -> one repr line per (cycle, cfg) for
    compute_composite and evaluate_signal. Uses only public APIs and no env."""
    rng = random.Random(seed)
    cfgs = _cfgs()
    lines: list[str] = []
    for i in range(n_cycles):
        u = _case(rng)
        cfg = cfgs[i % len(cfgs)]
        vix = rng.choice([14.0, 22.0, 35.0, None])
        snap = MarketSnapshot(
            asof=1_700_000_000.0,
            underlyings={"SPY": u},
            vix=vix,
            vix_age_seconds=rng.choice([10.0, 500.0, None]),
            data_healthy=rng.random() < 0.9,
        )
        comp = compute_composite(u, cfg)
        lines.append(f"{i}:composite:{comp!r}")
        sig = evaluate_signal(snap, cfg)
        lines.append(f"{i}:signal:{sig!r}")
    return lines


def golden_digest() -> str:
    return hashlib.sha256("\n".join(golden_lines()).encode()).hexdigest()


def test_flag_off_golden_digest_matches_base_commit():
    assert golden_digest() == GOLDEN_DIGEST_FLAG_OFF


# ─────────────────────────────────────────────────────────────────────────
# Flag-on behaviour (hand-computed expectations, static weights 0.40/0.25/0.20/0.15)
# ─────────────────────────────────────────────────────────────────────────

from agent.dpof.config import load_dpof_config  # noqa: E402
from agent.dpof.signals import composite_flat_variants  # noqa: E402

_FLAG = "CLIO_DPOF_DARKPOOL_FLAT_UNAVAILABLE"


def _u(nofa=0.4, premium=0.25, dp=None, dp_source="uw:/darkpool/SPY", gex=False):
    pillars = {
        "nofa": PillarReading(name="nofa", value=nofa, available=True, source="uw:/nofa/SPY"),
        "premium": PillarReading(name="premium", value=premium, available=True, source="uw:/premium/SPY"),
    }
    if dp is not None:
        pillars["darkpool"] = PillarReading(name="darkpool", value=dp, available=True, source=dp_source)
    if gex:
        # negative_gamma -> gex factor 0.5 (pin_risk None: not halved)
        return UnderlyingSnapshot(symbol="SPY", pillars=pillars, gex_regime="negative_gamma", pin_risk=None)
    return UnderlyingSnapshot(symbol="SPY", pillars=pillars)


def _cfg(flag: bool, **kw):
    return dataclasses.replace(DPOFConfig(), darkpool_flat_unavailable=flag, **kw)


def test_flag_off_flat_darkpool_is_still_included():
    """Flag off: the 0.0 darkpool stays in the composite: (0.2225 + 0.5*0.15)/(0.85+0.15)."""
    comp, used, reasons = compute_composite(_u(dp=0.0, gex=True), _cfg(False))
    assert comp == pytest.approx(0.2975)
    assert used == {"nofa": 0.4, "premium": 0.25, "darkpool": 0.0, "gex": 0.5}
    assert "darkpool_flat_unavailable" not in reasons


def test_flag_on_excludes_exact_zero_and_half_milli_darkpool():
    for dp in (0.0, 0.0005, -0.0005, 0.00099):
        comp, used, reasons = compute_composite(_u(dp=dp, gex=True), _cfg(True))
        assert "darkpool" not in used, dp
        assert "darkpool_flat_unavailable" in reasons, dp
        # renormalised over nofa/premium/gex: (0.2225 + 0.075) / (0.65 + 0.15)
        assert comp == pytest.approx(0.371875), dp


def test_flag_on_keeps_darkpool_at_and_above_threshold():
    comp, used, reasons = compute_composite(_u(dp=0.002, gex=True), _cfg(True))
    assert used["darkpool"] == 0.002
    assert "darkpool_flat_unavailable" not in reasons
    assert comp == pytest.approx(0.2979)  # (0.2225 + 0.2*0.002 + 0.075) / (0.85 + 0.15)
    # boundary: exactly 1e-3 is not |v| < 1e-3, so it stays
    _, used_b, _ = compute_composite(_u(dp=0.001, gex=True), _cfg(True))
    assert used_b["darkpool"] == 0.001


def test_guard_keeps_flat_darkpool_when_exclusion_would_drop_below_min_pillars():
    """nofa+premium+darkpool(flat), no GEX: excluding leaves 2 < min_pillars 3 -> kept."""
    comp, used, reasons = compute_composite(_u(dp=0.0), _cfg(True))
    assert used["darkpool"] == 0.0
    assert "darkpool_flat_unavailable" not in reasons
    assert comp == pytest.approx(0.2225 / 0.85)


def test_guard_drops_flat_darkpool_when_gex_keeps_count_at_min():
    """4 available (nofa, premium, darkpool, gex) -> excluding leaves 3 == min_pillars -> excluded."""
    comp, used, _ = compute_composite(_u(dp=0.0, gex=True), _cfg(True))
    assert "darkpool" not in used and comp == pytest.approx(0.371875)


def test_nbbo_reading_is_never_affected_by_flat_rule():
    nb = _u(dp=0.0, dp_source="uw:/darkpool/SPY#nbbo", gex=True)
    comp, used, reasons = compute_composite(nb, _cfg(True, darkpool_nbbo=True))
    assert used["darkpool"] == 0.0
    assert "darkpool_flat_unavailable" not in reasons
    assert comp == pytest.approx(0.2975)


def test_nbbo_mode_flag_on_ignores_legacy_flat_reading():
    comp, used, _ = compute_composite(_u(dp=0.0, gex=True), _cfg(True, darkpool_nbbo=True))
    assert used["darkpool"] == 0.0 and comp == pytest.approx(0.2975)


def test_insufficient_pillars_veto_is_not_created_by_flag():
    """nofa + flat darkpool only (2 < 3): None both with and without the flag."""
    assert compute_composite(_u(premium=None, dp=0.0), _cfg(True))[0] is None
    assert compute_composite(_u(premium=None, dp=0.0), _cfg(False))[0] is None


def test_composite_flat_variants_differ_only_when_rule_applies():
    with_f, without_f = composite_flat_variants(_u(dp=0.0, gex=True), _cfg(False))
    assert (with_f, without_f) == pytest.approx((0.2975, 0.371875))
    # guard blocks: identical
    assert composite_flat_variants(_u(dp=0.0), _cfg(False)) == pytest.approx((0.2225 / 0.85, 0.2225 / 0.85))
    # static weights are used even when the snapshot carries dynamic weights
    dyn = dataclasses.replace(
        _u(dp=0.0, gex=True),
        pillar_weights={"nofa": 0.1, "premium": 0.1, "darkpool": 0.1, "kronos": 0.0},
    )
    assert composite_flat_variants(dyn, _cfg(False)) == pytest.approx((0.2975, 0.371875))


def test_env_flag_default_off_and_bad_values_normalise_to_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert DPOFConfig().darkpool_flat_unavailable is False
    assert load_dpof_config().darkpool_flat_unavailable is False
    for bad in ("maybe", "2", "off", "no-thanks"):
        monkeypatch.setenv(_FLAG, bad)
        assert load_dpof_config().darkpool_flat_unavailable is False, bad
    monkeypatch.setenv(_FLAG, "1")
    assert load_dpof_config().darkpool_flat_unavailable is True


# ─────────────────────────────────────────────────────────────────────────
# Shadow evidence: composite_with_flat / composite_without_flat in the dynamic-weights row
# ─────────────────────────────────────────────────────────────────────────

import asyncio  # noqa: E402
import datetime as dt  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

from agent.dpof.dynamic_weights import DynamicWeightsManager  # noqa: E402
from agent.dpof.pillar_weights_io import ShadowLog  # noqa: E402

_ET = ZoneInfo("America/New_York")
_DAY = "2026-10-12"


def _shadow_u(dp=0.0, premium=0.2, gex=True):
    """nofa 0.3 (w .40), premium (w .25), darkpool (w .20), optional negative-gamma GEX (factor 0.5, w .15)."""
    pillars = {
        "nofa": PillarReading(name="nofa", value=0.3, available=True, source="uw:/nofa"),
        "darkpool": PillarReading(name="darkpool", value=dp, available=True, source="uw:/darkpool"),
    }
    if premium is not None:
        pillars["premium"] = PillarReading(name="premium", value=premium, available=True, source="uw:/flow")
    return UnderlyingSnapshot(
        symbol="SPY", pillars=pillars,
        gex_regime="negative_gamma" if gex else None, pin_risk=None,
        last_price=100.0, price_age_seconds=5.0, ret_15m=0.002,
    )


def _shadow_row(tmp_path, u, flag: bool):
    cfg = dataclasses.replace(
        DPOFConfig(), data_dir=str(tmp_path), dynamic_weights="shadow",
        kronos_pillar="off", darkpool_flat_unavailable=flag,
    )
    m = DynamicWeightsManager(cfg, str(tmp_path), session_fn=lambda now: (_DAY, True), sync_io=True)
    asyncio.run(m.ensure_loaded())
    ts = dt.datetime(2026, 10, 12, 10, 0, tzinfo=_ET).timestamp()
    m.finish_snapshot(ts, {"SPY": u})
    return ShadowLog(str(tmp_path)).read_tail(1)[-1]


def test_shadow_row_carries_both_flat_variants_hand_computed(tmp_path):
    """Flag off: static = with_flat = 0.245 ((0.17 + 0.15*0.5)/1.0); without_flat = (0.17 + 0.075)/0.8 = 0.30625."""
    row = _shadow_row(tmp_path, _shadow_u(dp=0.0), flag=False)
    assert row["static"] == pytest.approx(0.245)
    assert row["composite_with_flat"] == pytest.approx(0.245)
    assert row["composite_without_flat"] == pytest.approx(0.30625)


def test_shadow_row_static_follows_flag_while_variants_do_not(tmp_path):
    row = _shadow_row(tmp_path, _shadow_u(dp=0.0), flag=True)
    assert row["static"] == pytest.approx(0.30625)  # decision composite now excludes the flat pillar
    assert row["composite_with_flat"] == pytest.approx(0.245)
    assert row["composite_without_flat"] == pytest.approx(0.30625)


def test_shadow_variants_are_none_when_not_computable_not_zero(tmp_path):
    """nofa + flat darkpool only (2 < min_pillars 3): both variants None, never a 0.0 stand-in."""
    row = _shadow_row(tmp_path, _shadow_u(dp=0.0, premium=None, gex=False), flag=True)
    assert row["composite_with_flat"] is None
    assert row["composite_without_flat"] is None


def test_shadow_variants_differ_only_when_flat_rule_applies(tmp_path):
    row = _shadow_row(tmp_path, _shadow_u(dp=0.002), flag=False)
    assert row["composite_with_flat"] == pytest.approx(row["composite_without_flat"])


def _report_module():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "dpof_dynamic_weights_report_w7a", root / "scripts" / "dpof_dynamic_weights_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_replay_is_identical_with_and_without_new_keys(tmp_path):
    rep = _report_module()
    ts = dt.datetime(2026, 10, 12, 10, 0, tzinfo=_ET).isoformat()
    sig = {"ts": ts, "underlying": "SPY", "composite": 0.325, "signal_basis": None,
           "pillars": {"nofa": {"value": 0.4}, "premium": {"value": 0.25}}}
    plain = {"ts": ts, "u": "SPY", "static": 0.325, "dyn": 0.325, "dead": [],
             "w": {"nofa": 0.5, "premium": 0.5}, "avail": ["nofa", "premium"]}
    keyed = dict(plain, composite_with_flat=0.325, composite_without_flat=None)
    results = []
    for name, rows in (("plain", [plain]), ("keyed", [keyed]), ("mixed", [plain, keyed])):
        d = tmp_path / name
        d.mkdir()
        (d / "pillar_weights_shadow.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / "signals.jsonl").write_text(json.dumps(sig) + "\n")
        shadow = rep.load_jsonl(str(d / "pillar_weights_shadow.jsonl"))
        signals = rep.load_jsonl(str(d / "signals.jsonl"))
        results.append(rep.replay_check(shadow, signals))
    assert results[0]["mismatches"] == 0 and results[0]["dyn_match"] == 1 and results[0]["static_match"] == 1
    assert results[1]["dyn_match"] == 1 and results[1]["static_match"] == 1 and results[1]["mismatches"] == 0
    assert results[2]["shadow_rows"] == 2 and results[2]["mismatches"] == 0


# ─────────────────────────────────────────────────────────────────────────
# R-36 fix-up (orchestrator mutants): exemption paths and veto semantics pinned one by one
# ─────────────────────────────────────────────────────────────────────────

from agent.dpof.signals import _flat_legacy_darkpool  # noqa: E402


def test_nbbo_source_tag_alone_exempts_the_reading_even_with_the_nbbo_config_off():
    # data.py tags the NBBO reading '<source>#nbbo'; a snapshot carrying it must never be judged flat,
    # whatever the config says (e.g. a replayed row or a config flipped mid-session).
    nb = _u(dp=0.0, dp_source="uw:/darkpool/SPY#nbbo", gex=True)
    comp, used, reasons = compute_composite(nb, _cfg(True, darkpool_nbbo=False))
    assert used["darkpool"] == 0.0 and "darkpool_flat_unavailable" not in reasons
    assert comp == pytest.approx(0.2975)


def test_flat_helper_ignores_an_unavailable_reading_and_a_missing_value():
    cfg = _cfg(True)
    unavailable = UnderlyingSnapshot(symbol="SPY", pillars={
        "darkpool": PillarReading(name="darkpool", value=0.0, available=False, source="uw:/darkpool/SPY")})
    assert _flat_legacy_darkpool(unavailable, cfg) is False
    flat = UnderlyingSnapshot(symbol="SPY", pillars={
        "darkpool": PillarReading(name="darkpool", value=0.0, available=True, source="uw:/darkpool/SPY")})
    assert _flat_legacy_darkpool(flat, cfg) is True


def test_missing_nofa_leaves_the_flat_darkpool_in_the_diagnostic_pillars_used():
    # No nofa -> the composite is None whatever happens; the exclusion must not also hide darkpool from pillars_used.
    u = UnderlyingSnapshot(symbol="SPY", pillars={
        "premium": PillarReading(name="premium", value=0.25, available=True, source="uw:/premium/SPY"),
        "darkpool": PillarReading(name="darkpool", value=0.0, available=True, source="uw:/darkpool/SPY")},
        gex_regime="negative_gamma", pin_risk=None)
    comp, used, reasons = compute_composite(u, _cfg(True, min_pillars=2))
    assert comp is None and reasons == ["nofa_unavailable"]
    assert "darkpool" in used


def _veto_snapshot(dp):
    pillars = {
        "nofa": PillarReading(name="nofa", value=0.4, available=True, source="uw:/nofa/SPY"),
        "premium": PillarReading(name="premium", value=0.25, available=True, source="uw:/premium/SPY"),
        "darkpool": PillarReading(name="darkpool", value=dp, available=True, source="uw:/darkpool/SPY"),
    }
    u = UnderlyingSnapshot(symbol="SPY", pillars=pillars, gex_regime="negative_gamma", pin_risk=None,
                           last_price=500.0, vwap=500.0, ret_15m=0.001)
    return MarketSnapshot(asof=1_700_000_000.0, underlyings={"SPY": u}, vix=14.0, vix_age_seconds=10.0, data_healthy=True)


def test_an_excluded_flat_darkpool_cannot_veto_but_an_included_one_still_can():
    # Tiny veto threshold so a flat -0.0007 reading would veto a bullish composite if it were still counted.
    tiny = dict(dp_veto_threshold=0.0005)
    off = evaluate_signal(_veto_snapshot(-0.0007), _cfg(False, **tiny))
    on = evaluate_signal(_veto_snapshot(-0.0007), _cfg(True, **tiny))
    assert "dp_veto" in off.vetoes          # flag off: legacy behaviour unchanged
    assert "dp_veto" not in on.vetoes       # flag on: the excluded pillar is out of the veto too
    assert "darkpool_flat_unavailable" in on.reasons or "darkpool" not in on.pillars_used
