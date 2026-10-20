"""Unit tests for the pure DPOF pillar-weight core (agent/dpof/pillar_weights.py)."""

import itertools
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.dpof.pillar_weights import (  # noqa: E402
    LivenessTracker,
    QualityStats,
    WeightParams,
    compute_weights,
)

P = WeightParams()
KEY = "SPY|premium|uw"
DAY = "2026-10-09"
PILLARS = ("nofa", "premium", "darkpool", "kronos")


def _obs(tr, values, t0=0.0, step=180.0, day=DAY, key=KEY):
    """Feed values at evenly spaced timestamps."""
    for i, v in enumerate(values):
        tr.observe(key, v, t0 + i * step, day, True)


def _wobble(i):
    """A healthy-looking reading that is never identical twice in a row (a frozen value must not revive a pillar)."""
    return 0.02 + 0.001 * (i % 2)


def _dead_tracker():
    """Tracker flagged dead at ts 39*180 by 40 zeros."""
    tr = LivenessTracker(P)
    _obs(tr, [0.0] * 40)
    assert tr.is_dead(KEY)
    return tr


def test_not_dead_before_ring_is_full():
    """A pillar is not dead while fewer than ring_size refreshes have been seen."""
    tr = LivenessTracker(P)
    _obs(tr, [0.0] * 39)
    assert not tr.is_dead(KEY)
    assert tr.aliveness(KEY) == 1.0


def test_dead_exactly_when_full_and_all_small():
    """Dead on the 40th refresh when every value in the ring is below dead_eps in magnitude."""
    tr = LivenessTracker(P)
    _obs(tr, [0.001, -0.002] * 20)
    assert tr.is_dead(KEY)
    assert tr.aliveness(KEY) == 0.0


def test_dead_on_identical_nonzero_values():
    """A stuck non-zero value (identical across a full ring) is flagged dead."""
    tr = LivenessTracker(P)
    _obs(tr, [0.3] * 40)
    assert tr.is_dead(KEY)


def test_dead_threshold_is_strict_less_than():
    """Alternating +/-dead_eps exactly (max|v| == eps) is not dead; just under eps is dead."""
    tr = LivenessTracker(P)
    _obs(tr, [P.dead_eps, -P.dead_eps] * 20)
    assert not tr.is_dead(KEY)
    tr2 = LivenessTracker(P)
    _obs(tr2, [0.0049, -0.0049] * 20)
    assert tr2.is_dead(KEY)


def test_healthy_varying_series_never_dead():
    """An alternating +/-0.3 series never flags dead."""
    tr = LivenessTracker(P)
    _obs(tr, [0.3 if i % 2 else -0.3 for i in range(200)])
    assert not tr.is_dead(KEY)
    assert tr.keys() == [KEY]


def test_out_of_session_observations_ignored():
    """Observations with in_session False create no state and never flip a pillar."""
    tr = LivenessTracker(P)
    for i in range(50):
        tr.observe(KEY, 0.0, i * 180.0, DAY, False)
    assert tr.keys() == []
    assert not tr.is_dead(KEY)


def test_invalid_values_and_timestamps_ignored():
    """bool, NaN, inf, None, 10**400 and strings are ignored and create no state."""
    bad_values = [True, False, float("nan"), float("inf"), float("-inf"), None, 10**400, "abc", "0.5"]
    for bad in bad_values:
        tr = LivenessTracker(P)
        tr.observe(KEY, bad, 0.0, DAY, True)
        assert tr.keys() == [], f"value {bad!r} was not ignored"
    tr = LivenessTracker(P)
    for bad_ts in [float("nan"), None, True, 10**400]:
        tr.observe(KEY, 0.0, bad_ts, DAY, True)
    assert tr.keys() == []


def test_new_session_clears_ring_but_keeps_dead_flag():
    """A new session_day clears the ring, while a pillar dead at yesterday's close stays dead."""
    tr = _dead_tracker()
    tr.observe(KEY, 0.0, 99999.0, "2026-10-10", True)
    assert tr.is_dead(KEY)
    assert len(tr.to_dict()["keys"][KEY]["ring"]) == 1


def test_revival_needs_consecutive_values_and_hysteresis():
    """Revival needs revive_n consecutive |v| >= 0.01; values in [0.005, 0.01) reset the counter."""
    tr = _dead_tracker()
    t = 7020.0 + 1800.0
    for i in range(6):
        tr.observe(KEY, 0.007, t + i * 180.0, DAY, True)
    assert tr.is_dead(KEY)
    for i in range(5):
        tr.observe(KEY, _wobble(i), t + (6 + i) * 180.0, DAY, True)
    tr.observe(KEY, 0.007, t + 11 * 180.0, DAY, True)
    assert tr.is_dead(KEY)
    for i in range(5):
        tr.observe(KEY, _wobble(i), t + (12 + i) * 180.0, DAY, True)
    assert tr.is_dead(KEY)
    tr.observe(KEY, _wobble(5), t + 17 * 180.0, DAY, True)
    assert not tr.is_dead(KEY)


def test_a_frozen_non_zero_value_never_revives_a_dead_pillar():
    """Frozen at 0.8 for hours: the stuck arm kills it and it stays dead (no dead/alive duty cycle every dwell)."""
    tr = LivenessTracker(P)
    for i in range(400):
        tr.observe(KEY, 0.8, i * 180.0, DAY, True)
    assert tr.is_dead(KEY)
    assert len(tr.transitions(KEY)) == 1


def test_a_frozen_value_revives_once_it_varies():
    """The same dead pillar revives after the dwell as soon as the last revive_n values include a change."""
    tr = LivenessTracker(P)
    for i in range(60):
        tr.observe(KEY, 0.8, i * 180.0, DAY, True)
    assert tr.is_dead(KEY)
    t = 60 * 180.0 + P.dwell_seconds
    for i in range(6):
        tr.observe(KEY, 0.8 if i < 5 else 0.7, t + i * 180.0, DAY, True)
    assert not tr.is_dead(KEY)


def test_dwell_blocks_flip_inside_window_and_allows_at_boundary():
    """No flip within dwell_seconds of the previous flip; a flip is allowed at exactly 1800 s."""
    tr = LivenessTracker(P)
    # Kill at t=0 (needs full ring): fill with zeros at 1 s spacing -> flip is at the 40th obs.
    _obs(tr, [0.0] * 40, step=1.0)
    assert tr.is_dead(KEY)
    dead_ts = tr.transitions(KEY)[-1][0]
    # Revive with 6 big values one second apart, well inside the dwell window.
    for i in range(6):
        tr.observe(KEY, _wobble(i), dead_ts + 100.0 + i, DAY, True)
    assert tr.is_dead(KEY), "revival must be blocked inside dwell"
    tr.observe(KEY, _wobble(6), dead_ts + 1800.0, DAY, True)
    assert not tr.is_dead(KEY)
    revive_ts = tr.transitions(KEY)[-1][0]
    assert revive_ts == dead_ts + 1800.0
    # Kill again: refill the ring with zeros at 40 s spacing (inside dwell) -> must stay alive.
    for i in range(40):
        tr.observe(KEY, 0.0, revive_ts + 40.0 * (i + 1), DAY, True)
    assert not tr.is_dead(KEY), "kill must be blocked inside dwell"
    tr.observe(KEY, 0.0, revive_ts + 1800.0, DAY, True)
    assert tr.is_dead(KEY), "kill must be allowed at exactly the dwell boundary"


def test_ramp_sequence_is_one_third_two_thirds_one():
    """After revival aliveness reads 1/3, 2/3, 1.0 over the revival observation and the next two."""
    tr = _dead_tracker()
    t = 7020.0 + 1800.0
    seq = []
    for i in range(6):
        tr.observe(KEY, _wobble(i), t + i * 180.0, DAY, True)
        seq.append(tr.aliveness(KEY))
    assert seq[:5] == [0.0] * 5
    assert seq[5] == 1.0 / 3.0
    tr.observe(KEY, _wobble(6), t + 6 * 180.0, DAY, True)
    assert tr.aliveness(KEY) == 2.0 / 3.0
    tr.observe(KEY, _wobble(7), t + 7 * 180.0, DAY, True)
    assert tr.aliveness(KEY) == 1.0
    tr.observe(KEY, _wobble(8), t + 8 * 180.0, DAY, True)
    assert tr.aliveness(KEY) == 1.0


def test_aliveness_unknown_key_is_one():
    """Unknown keys report full aliveness."""
    assert LivenessTracker(P).aliveness("NEVER|premium|uw") == 1.0


def test_round_trip_through_json_preserves_state():
    """to_dict -> json -> from_dict keeps dead flags, rings, ramp stage and transitions."""
    tr = _dead_tracker()
    t = 7020.0 + 1800.0
    for i in range(6):
        tr.observe(KEY, 0.02, t + i * 180.0, DAY, True)
    tr.observe("QQQ|darkpool|uw", 0.4, 0.0, DAY, True)
    payload = json.dumps(tr.to_dict(), allow_nan=False)
    back = LivenessTracker.from_dict(json.loads(payload), P)
    assert back.to_dict() == tr.to_dict()
    assert back.is_dead(KEY) == tr.is_dead(KEY)
    assert back.aliveness(KEY) == tr.aliveness(KEY)
    assert back.transitions(KEY) == tr.transitions(KEY)
    assert back.keys() == tr.keys()


def test_from_dict_garbage_never_raises_and_is_empty():
    """Malformed, unknown-version or corrupt input yields an empty tracker and never raises."""
    good = _dead_tracker().to_dict()
    ring_too_long = json.loads(json.dumps(good))
    ring_too_long["keys"][KEY]["ring"] = [0.0] * 41
    nan_ring = json.loads(json.dumps(good))
    nan_ring["keys"][KEY]["ring"] = [float("nan")] * 40
    ramp_zero = json.loads(json.dumps(good))
    ramp_zero["keys"][KEY]["ramp_stage"] = 0
    dead_int = json.loads(json.dumps(good))
    dead_int["keys"][KEY]["dead"] = 1
    version_bool = json.loads(json.dumps(good))
    version_bool["schema_version"] = True
    garbage = [None, [], "x", 3, {}, {"schema_version": 99}, {"schema_version": 1, "keys": None},
               ring_too_long, nan_ring, ramp_zero, dead_int, version_bool]
    for d in garbage:
        tr = LivenessTracker.from_dict(d, P)
        assert isinstance(tr, LivenessTracker)
        assert tr.keys() == [], f"garbage {d!r} produced keys"


class _ExplodingMapping(dict):
    """A mapping whose lookups raise, to prove from_dict guards its own reads."""

    def get(self, *args, **kwargs):
        raise RuntimeError("boom")

    def __getitem__(self, key):
        raise RuntimeError("boom")


def test_from_dict_never_raises_even_when_mapping_explodes():
    """from_dict and QualityStats.from_dict return clean empty objects when reads raise."""
    tr = LivenessTracker.from_dict(_ExplodingMapping(schema_version=1, keys={}), P)
    assert tr.keys() == []
    assert QualityStats.from_dict(_ExplodingMapping(n=1.0)).n_eff() == 0.0


def test_from_dict_skips_only_bad_key():
    """A corrupt key is skipped while a valid sibling key is kept."""
    d = json.loads(json.dumps(_dead_tracker().to_dict()))
    d["keys"]["QQQ|darkpool|uw"] = {"ring": "nope"}
    tr = LivenessTracker.from_dict(d, P)
    assert tr.keys() == [KEY]
    assert tr.is_dead(KEY)


def test_quality_add_block_idempotent_and_rejects_bad_inputs():
    """add_block accepts each (day, block) once and rejects bool, non-finite and bad indices."""
    q = QualityStats()
    assert q.add_block(DAY, 0, 0.1, 0.2, 0.3) is True
    assert q.add_block(DAY, 0, 0.1, 0.2, 0.3) is False
    assert q.n_eff() == 1.0
    rejects = [
        (DAY, 1, True, 0.0, 0.0),
        (DAY, 1, float("nan"), 0.0, 0.0),
        (DAY, 1, 0.0, 10**400, 0.0),
        (DAY, -1, 0.0, 0.0, 0.0),
        (DAY, True, 0.0, 0.0, 0.0),
        (DAY, 1.5, 0.0, 0.0, 0.0),
        ("", 1, 0.0, 0.0, 0.0),
        (None, 1, 0.0, 0.0, 0.0),
    ]
    for args in rejects:
        assert q.add_block(*args) is False, args
    assert q.n_eff() == 1.0


def _planted(seed, n, slope):
    rng = random.Random(seed)
    q = QualityStats()
    for i in range(n):
        z = rng.gauss(0.0, 1.0)
        x = rng.gauss(0.0, 1.0)
        y = slope * x + rng.gauss(0.0, 1.0)
        assert q.add_block(DAY, i, x, z, y)
    return q


def test_planted_signal_detected_with_ci_excluding_zero():
    """A planted y = 0.3 x + noise gives partial IC > 0.2 with a CI excluding 0."""
    q = _planted(7, 300, 0.3)
    r = q.partial_ic()
    lo, hi = q.ci95()
    assert r is not None and r > 0.2
    assert lo > 0.0


def test_pure_momentum_has_no_partial_ic():
    """x = z + small noise and y = 0.5 z + noise: strong corr(x, y) but partial IC near 0."""
    rng = random.Random(11)
    q = QualityStats()
    for i in range(300):
        z = rng.gauss(0.0, 1.0)
        x = z + 0.1 * rng.gauss(0.0, 1.0)
        y = 0.5 * z + 0.1 * rng.gauss(0.0, 1.0)
        q.add_block(DAY, i, x, z, y)
    # Plain Pearson r(x, y) is large here; the control must remove it.
    sxy, sx, sy = q.sxy / q.n, q.sx / q.n, q.sy / q.n
    cov = sxy - sx * sy
    vx = q.sxx / q.n - sx * sx
    vy = q.syy / q.n - sy * sy
    assert cov / math.sqrt(vx * vy) > 0.8
    assert abs(q.partial_ic()) < 0.15


def test_null_false_positive_rate_within_budget():
    """With y independent of x and z, the 95% CI excludes 0 in at most 3 of 30 seeds."""
    hits = 0
    for seed in range(30):
        q = _planted(seed + 100, 300, 0.0)
        ci = q.ci95()
        if ci is not None and (ci[0] > 0.0 or ci[1] < 0.0):
            hits += 1
    assert hits <= 3


def test_roll_session_decays_once_per_day_and_is_idempotent():
    """roll_session multiplies sums by 0.5**(1/half_life) once per new day and ignores repeats."""
    q = _planted(3, 100, 0.0)
    q.roll_session(DAY, P)
    before = q.n_eff()
    q.roll_session(DAY, P)
    assert q.n_eff() == before
    q.roll_session("2026-10-10", P)
    lam = 0.5 ** (1.0 / P.half_life_sessions)
    assert math.isclose(q.n_eff(), before * lam, rel_tol=1e-12)
    q.roll_session("2026-10-10", P)
    assert math.isclose(q.n_eff(), before * lam, rel_tol=1e-12)
    q.roll_session("2026-10-13", P)
    assert math.isclose(q.n_eff(), before * lam * lam, rel_tol=1e-12)


def test_multiplier_is_one_when_quality_apply_off():
    """With quality_apply False the multiplier is exactly 1.0 even for a strong planted signal."""
    q = _planted(5, 400, 0.3)
    assert q.multiplier(P) == 1.0


def test_multiplier_is_one_below_n_floor():
    """With quality_apply True but n_eff below n_floor the multiplier is exactly 1.0."""
    q = _planted(5, 100, 0.3)
    assert q.multiplier(replace(P, quality_apply=True)) == 1.0


def test_multiplier_is_one_when_ci_includes_zero():
    """With quality_apply True and enough samples but a null signal the multiplier is exactly 1.0."""
    q = _planted(21, 400, 0.0)
    ci = q.ci95()
    assert ci is not None and ci[0] <= 0.0 <= ci[1]
    assert q.multiplier(replace(P, quality_apply=True)) == 1.0


def test_multiplier_applies_for_strong_signal_and_is_clipped():
    """With quality_apply True, n_eff >= n_floor and a CI excluding 0 the multiplier is in (1, m_hi]."""
    q = _planted(5, 400, 0.3)
    m = q.multiplier(replace(P, quality_apply=True))
    assert 1.0 < m <= P.m_hi


def test_diagnostic_multiplier_ignores_gates():
    """diagnostic_multiplier reports the posterior-based value even when quality_apply is off."""
    q = _planted(5, 400, 0.3)
    assert q.multiplier(P) == 1.0
    d = q.diagnostic_multiplier(P)
    assert P.m_lo <= d <= P.m_hi and d > 1.0


def test_quality_round_trip_and_garbage():
    """QualityStats survives JSON round trip; garbage yields fresh empty stats without raising."""
    q = _planted(9, 50, 0.2)
    back = QualityStats.from_dict(json.loads(json.dumps(q.to_dict(), allow_nan=False)))
    assert back.to_dict() == q.to_dict()
    for bad in [None, [], {}, {"n": "x"}, {"n": float("nan")}]:
        fresh = QualityStats.from_dict(bad)
        assert fresh.n_eff() == 0.0 and fresh.added == []


def test_identity_for_every_nonempty_subset_of_pillars():
    """All pillars alive with multiplier 1 gives back the static prior for every subset."""
    for r in range(1, 5):
        for subset in itertools.combinations(PILLARS, r):
            w = compute_weights({p: 1.0 for p in subset}, {}, P)
            assert set(w) == set(subset)
            for p in subset:
                assert abs(w[p] - P.prior[p]) <= 1e-12, (subset, p)


def test_dead_premium_redistributes_proportionally():
    """A dead premium gets weight 0 and the others scale by 1/0.75 to keep the total at 1.0."""
    w = compute_weights({"nofa": 1, "premium": 0, "darkpool": 1, "kronos": 1}, {}, P)
    assert w["premium"] == 0.0
    assert math.isclose(w["nofa"], 0.40 / 0.75, abs_tol=1e-12)
    assert math.isclose(w["darkpool"], 0.20 / 0.75, abs_tol=1e-12)
    assert math.isclose(w["kronos"], 0.15 / 0.75, abs_tol=1e-12)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-12)


def test_cap_binds_with_ramping_premium():
    """nofa would be 0.6316 with a ramping premium and dead darkpool; the cap clamps it to 0.60."""
    w = compute_weights({"nofa": 1, "premium": 1 / 3, "darkpool": 0, "kronos": 1}, {}, P)
    assert math.isclose(w["nofa"], 0.60, abs_tol=1e-12)
    assert math.isclose(w["premium"], 0.25 / 3 / (0.25 / 3 + 0.15) * 0.40, abs_tol=1e-12)
    assert math.isclose(w["kronos"], 0.15 / (0.25 / 3 + 0.15) * 0.40, abs_tol=1e-12)
    assert w["darkpool"] == 0.0
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-12)


def test_floor_binds_with_tiny_premium():
    """premium at aliveness 0.02 would be ~0.009 and is lifted to the floor 0.05; nofa hits the cap, kronos takes the rest."""
    w = compute_weights({"nofa": 1, "premium": 0.02, "darkpool": 0, "kronos": 1}, {}, P)
    assert math.isclose(w["nofa"], 0.60, abs_tol=1e-12)
    assert math.isclose(w["premium"], 0.05, abs_tol=1e-12)
    assert math.isclose(w["kronos"], 0.35, abs_tol=1e-12)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-12)


def test_cap_excess_is_redistributed_proportionally_before_the_floor_is_judged():
    """premium at aliveness 0.1 is 0.0435 of the raw mass, but once nofa's excess above the cap is handed out it clears
    the floor: premium:kronos keep their 1:6 proportion and only nofa is pinned (the old same-pass clamp lifted premium)."""
    w = compute_weights({"nofa": 1, "premium": 0.1, "darkpool": 0, "kronos": 1}, {}, P)
    assert math.isclose(w["nofa"], 0.60, abs_tol=1e-12)
    assert math.isclose(w["premium"], 0.4 / 7.0, abs_tol=1e-12)
    assert math.isclose(w["kronos"], 0.4 * 6.0 / 7.0, abs_tol=1e-12)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-12)


def test_skewed_shares_that_the_old_clamp_could_not_bound_are_bounded():
    """One dominant pillar and two near-zero ones: cap and floor violations coexist, the old pass returned None."""
    from agent.dpof.pillar_weights import _apply_bounds
    out = _apply_bounds({"a": 0.9999384310113245, "b": 3.907676657152258e-05, "c": 2.249222210405888e-05}, 0.60, 0.05)
    assert out is not None
    assert math.isclose(out["a"], 0.60, abs_tol=1e-12)
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-12)
    assert all(0.05 - 1e-12 <= v <= 0.60 + 1e-12 for v in out.values())
    assert math.isclose(out["b"] / out["c"], 3.907676657152258e-05 / 2.249222210405888e-05, rel_tol=1e-9)


def test_bounds_solver_is_a_true_projection_on_random_inputs():
    """Random 3-5 pillar shares: result sums to 1, sits inside [floor, cap], and unpinned pillars keep their ratios."""
    from agent.dpof.pillar_weights import _apply_bounds
    rng = random.Random(11)
    for _ in range(3000):
        n = rng.choice((3, 4, 5))
        raw = [rng.random() ** rng.choice((1, 3, 8)) + 1e-12 for _ in range(n)]
        total = sum(raw)
        shares = {f"p{i}": raw[i] / total for i in range(n)}
        out = _apply_bounds(shares, 0.60, 0.05)
        assert out is not None, shares
        assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)
        assert all(0.05 - 1e-12 <= v <= 0.60 + 1e-12 for v in out.values())
        free = [k for k, v in out.items() if 0.05 + 1e-9 < v < 0.60 - 1e-9]
        for a in free:
            for b in free:
                assert math.isclose(out[a] / out[b], shares[a] / shares[b], rel_tol=1e-6)


def test_bounds_solver_rejects_infeasible_and_malformed_input():
    from agent.dpof.pillar_weights import _apply_bounds
    assert _apply_bounds({}, 0.6, 0.05) is None
    assert _apply_bounds({"a": 0.5, "b": 0.5}, 0.4, 0.05) is None          # 2 * cap < 1
    assert _apply_bounds({"a": 0.2, "b": 0.3, "c": 0.5}, 0.6, 0.4) is None  # 3 * floor > 1
    assert _apply_bounds({"a": float("nan"), "b": 0.5, "c": 0.5}, 0.6, 0.05) is None
    assert _apply_bounds({"a": 0.0, "b": 0.5, "c": 0.5}, 0.6, 0.05) is None
    assert _apply_bounds({"a": True, "b": 0.5, "c": 0.5}, 0.6, 0.05) is None


def test_non_finite_aliveness_and_multipliers_are_sanitised():
    """NaN aliveness counts as dead; NaN, inf, negative and zero multipliers fall back to 1.0."""
    w = compute_weights({"nofa": float("nan"), "premium": 1, "darkpool": 1, "kronos": 1}, {}, P)
    assert w["nofa"] == 0.0
    assert math.isclose(w["premium"], 0.25 / 0.6, abs_tol=1e-12)
    assert math.isclose(w["darkpool"], 0.20 / 0.6, abs_tol=1e-12)
    assert math.isclose(w["kronos"], 0.15 / 0.6, abs_tol=1e-12)
    all_alive = {p: 1.0 for p in PILLARS}
    bad_mult = {"nofa": float("inf"), "premium": float("nan"), "darkpool": -1.0, "kronos": 0.0}
    w2 = compute_weights(all_alive, bad_mult, P)
    for p in PILLARS:
        assert abs(w2[p] - P.prior[p]) <= 1e-12


def test_multipliers_are_clamped_to_band():
    """A multiplier of 5.0 is clamped to m_hi = 1.2 before weighting."""
    w = compute_weights({p: 1.0 for p in PILLARS}, {"nofa": 5.0}, P)
    assert math.isclose(w["nofa"], 0.48 / 1.08, abs_tol=1e-12)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-12)


def test_empty_zero_and_unknown_inputs_return_empty():
    """No usable pillars, zero total or only unknown pillar names return {} (caller uses static)."""
    assert compute_weights({}, {}, P) == {}
    assert compute_weights({p: 0.0 for p in PILLARS}, {}, P) == {}
    assert compute_weights({"foo": 1.0}, {}, P) == {}


def test_identity_survives_a_cap_that_all_alive_shares_violate():
    """With cap 0.45 the all-alive nofa share (0.47) would clamp, but the identity case must not clamp."""
    tight = replace(P, cap=0.45)
    w = compute_weights({"nofa": 1, "premium": 1, "darkpool": 1}, {}, tight)
    for p in ("nofa", "premium", "darkpool"):
        assert abs(w[p] - P.prior[p]) <= 1e-12, p


# ------------------------------------------- reviewer additions (mutation gaps)

def test_revive_boundary_is_inclusive_at_revive_eps():
    """Six consecutive values of exactly revive_eps revive a dead pillar; 0.0099 does not."""
    tr = _dead_tracker()
    t0 = 39 * 180.0 + P.dwell_seconds
    _obs(tr, [0.0099] * 6, t0=t0)
    assert tr.is_dead(KEY)
    tr2 = _dead_tracker()
    _obs(tr2, [0.01, 0.011] * 3, t0=t0)
    assert not tr2.is_dead(KEY)


def test_one_tiny_value_in_an_otherwise_healthy_ring_is_not_dead():
    """Dead needs ALL values tiny (max |v| < eps); a ring with a single tiny value, or a single large one, behaves accordingly."""
    tr = LivenessTracker(P)
    _obs(tr, [0.3] * 39 + [0.0])
    assert not tr.is_dead(KEY)
    tr2 = LivenessTracker(P)
    _obs(tr2, [0.001, -0.002] * 19 + [0.001, 0.3])
    assert not tr2.is_dead(KEY)


def test_partial_ic_needs_at_least_six_samples():
    """With 5 samples partial_ic is None; with 6 non-degenerate samples it is a float."""
    rng = random.Random(3)
    q = QualityStats()
    for i in range(5):
        assert q.add_block(DAY, i, rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
    assert q.partial_ic() is None
    assert q.add_block(DAY, 5, rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
    assert isinstance(q.partial_ic(), float)


def test_ci95_uses_fisher_z_with_n_minus_4_degrees_of_freedom():
    """The CI equals tanh(atanh(r) +- 1.96/sqrt(n-4)) for the partial IC r."""
    q = _planted(11, 120, 0.3)
    r, ne = q.partial_ic(), q.n_eff()
    lo, hi = q.ci95()
    se = 1.0 / math.sqrt(ne - 4.0)
    assert lo == pytest.approx(math.tanh(math.atanh(r) - 1.96 * se), abs=1e-12)
    assert hi == pytest.approx(math.tanh(math.atanh(r) + 1.96 * se), abs=1e-12)


def test_posterior_shrinks_by_n_over_n_plus_n0_and_diagnostic_multiplier_follows():
    """posterior == n/(n+n0) * partial_ic, and the diagnostic multiplier is clip(1 + kappa*posterior)."""
    q = _planted(5, 200, 0.2)
    r, ne = q.partial_ic(), q.n_eff()
    post = q.posterior(P)
    assert post == pytest.approx(ne / (ne + P.n0) * r, abs=1e-12)
    assert abs(post) < abs(r)
    expected = min(max(1.0 + P.kappa * post, P.m_lo), P.m_hi)
    assert q.diagnostic_multiplier(P) == pytest.approx(expected, abs=1e-12)


def test_n_floor_boundary_is_inclusive():
    """With n_eff exactly equal to n_floor and a strong significant signal the multiplier already applies."""
    params = replace(P, quality_apply=True, n_floor=60)
    q = _planted(2, 60, 1.0)
    assert q.n_eff() == 60.0
    assert q.multiplier(params) > 1.0
    params_hi = replace(params, n_floor=61)
    assert q.multiplier(params_hi) == 1.0


def test_significant_negative_ic_lowers_the_multiplier():
    """A strongly significant NEGATIVE partial IC gives a multiplier below 1 (never below m_lo)."""
    params = replace(P, quality_apply=True, n_floor=100, n0=100.0)
    q = _planted(4, 300, -0.9)
    ci = q.ci95()
    assert ci[1] < 0.0
    m = q.multiplier(params)
    assert params.m_lo <= m < 1.0


def test_roll_session_keeps_previous_day_keys_so_late_duplicates_stay_rejected():
    """After rolling to the next day, re-sending a block from the previous day is still rejected."""
    q = QualityStats()
    q.roll_session("2026-10-08", P)  # first roll only sets the day; pruning happens on the NEXT roll
    assert q.add_block("2026-10-08", 0, 0.1, 0.2, 0.3)
    q.roll_session("2026-10-09", P)
    assert q.add_block("2026-10-08", 0, 0.1, 0.2, 0.3) is False
    q.roll_session("2026-10-10", P)
    assert q.add_block("2026-10-09", 3, 0.5, 0.1, 0.2) is True
    assert q.add_block("2026-10-08", 0, 0.1, 0.2, 0.3) is True  # two days later the old key has aged out


def test_two_positive_pillars_are_not_bounded_only_three_or_more_are():
    """With only NOFA and Kronos carrying weight the shares are the plain 0.727/0.273, not clamped to the 0.60 cap."""
    w = compute_weights({"nofa": 1.0, "premium": 0.0, "darkpool": 0.0, "kronos": 1.0}, {}, P)
    assert w["nofa"] == pytest.approx(0.40 / 0.55, abs=1e-12)
    assert w["kronos"] == pytest.approx(0.15 / 0.55, abs=1e-12)
    assert w["premium"] == 0.0 and w["darkpool"] == 0.0
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)
