"""Seeded property tests for the pure DPOF pillar-weight core (random.Random only)."""

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.dpof.pillar_weights import (  # noqa: E402
    LivenessTracker,
    WeightParams,
    compute_weights,
)

P = WeightParams()
PILLARS = ("nofa", "premium", "darkpool", "kronos")
SEEDS = range(200)
_BAD_NUMS = [float("nan"), float("inf"), float("-inf"), -1.0, 0.0, True, None, "x", 10**400]


def _scenario(rng):
    """Random available/multiplier inputs, sanitised exactly as compute_weights does."""
    k = rng.randint(1, 4)
    subset = rng.sample(PILLARS, k)
    available = {}
    for p in subset:
        roll = rng.random()
        if roll < 0.25:
            available[p] = 0.0
        elif roll < 0.4:
            available[p] = 1.0
        elif roll < 0.5:
            available[p] = rng.choice(_BAD_NUMS)
        else:
            available[p] = rng.random()
    multipliers = {}
    for p in subset:
        roll = rng.random()
        if roll < 0.4:
            continue
        if roll < 0.5:
            multipliers[p] = rng.choice(_BAD_NUMS)
        else:
            multipliers[p] = rng.uniform(0.5, 1.5)
    return available, multipliers


def _as_finite(v):
    """Independent mirror of the sanitiser: finite float or None (bool/str/overflow rejected)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


def _sanitised_alive(available):
    out = {}
    for p, v in available.items():
        f = _as_finite(v)
        out[p] = 0.0 if f is None else min(max(f, 0.0), 1.0)
    return out


def _sanitised_mult(multipliers, subset):
    out = {}
    for p in subset:
        f = _as_finite(multipliers.get(p))
        if f is None or f <= 0.0:
            out[p] = 1.0
        else:
            out[p] = min(max(f, P.m_lo), P.m_hi)
    return out


def test_weights_conserve_target_and_are_finite_nonnegative():
    """Weights sum to the available prior total within 1e-12 and are finite and non-negative."""
    for seed in SEEDS:
        rng = random.Random(seed)
        available, multipliers = _scenario(rng)
        w = compute_weights(available, multipliers, P)
        if not w:
            assert not any(a > 0 for a in _sanitised_alive(available).values()), seed
            continue
        target = sum(P.prior[p] for p in available)
        assert abs(sum(w.values()) - target) <= 1e-12, seed
        for p, v in w.items():
            assert math.isfinite(v) and v >= 0.0, (seed, p, v)


def test_dead_pillars_get_zero_weight():
    """A pillar with sanitised aliveness 0 always gets weight exactly 0.0."""
    for seed in SEEDS:
        rng = random.Random(seed)
        available, multipliers = _scenario(rng)
        w = compute_weights(available, multipliers, P)
        alive = _sanitised_alive(available)
        for p, a in alive.items():
            if a == 0.0 and w:
                assert w[p] == 0.0, (seed, p)


def test_result_keys_match_available_prior_pillars():
    """Result covers exactly the available pillars that have a prior, or is empty."""
    for seed in SEEDS:
        rng = random.Random(seed)
        available, multipliers = _scenario(rng)
        w = compute_weights(available, multipliers, P)
        if w:
            assert set(w) == set(available)


def test_bounds_hold_for_three_or_more_positive_pillars_unless_identity():
    """With >= 3 pillars carrying weight, every share lies in [floor, cap] unless the identity case applies."""
    checked = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        available, multipliers = _scenario(rng)
        w = compute_weights(available, multipliers, P)
        if not w:
            continue
        alive = _sanitised_alive(available)
        mult = _sanitised_mult(multipliers, available.keys())
        positive = [p for p in available if alive[p] * P.prior[p] * mult[p] > 0]
        identity = all(alive[p] == 1.0 and mult[p] == 1.0 for p in available)
        if len(positive) < 3 or identity:
            continue
        target = sum(P.prior[p] for p in available)
        for p in positive:
            share = w[p] / target
            assert P.floor - 1e-12 <= share <= P.cap + 1e-12, (seed, p, share)
        checked += 1
    assert checked > 0


def test_ordering_invariance_and_determinism():
    """Shuffling input dict order gives identical output, and repeated calls agree exactly."""
    for seed in SEEDS:
        rng = random.Random(seed)
        available, multipliers = _scenario(rng)
        w1 = compute_weights(available, multipliers, P)
        w2 = compute_weights(available, multipliers, P)
        assert w1 == w2, seed
        items = list(available.items())
        rng.shuffle(items)
        shuffled = dict(items)
        mitems = list(multipliers.items())
        rng.shuffle(mitems)
        w3 = compute_weights(shuffled, dict(mitems), P)
        assert w3 == w1, seed


def test_liveness_fuzz_never_raises_and_stays_serialisable():
    """Random observation streams never raise, aliveness stays in [0, 1], and state is JSON-safe."""
    values = [0.0, 0.001, -0.002, 0.3, -0.3, float("nan"), float("inf"), True, None, 10**400, "x"]
    keys = ["SPY|premium|uw", "QQQ|darkpool|uw", "SPY|darkpool|selfflow"]
    for seed in SEEDS:
        rng = random.Random(seed)
        tr = LivenessTracker(P)
        ts = 0.0
        day = "2026-10-09"
        for _ in range(300):
            ts += rng.choice([0.0, 1.0, 40.0, 180.0, 400.0, 1800.0])
            if rng.random() < 0.02:
                day = f"2026-10-{rng.randint(10, 30):02d}"
            value = rng.choice(values) if rng.random() < 0.3 else rng.uniform(-1.0, 1.0)
            tr.observe(rng.choice(keys), value, ts, day, rng.random() < 0.9)
        for key in tr.keys():
            assert 0.0 <= tr.aliveness(key) <= 1.0
        payload = json.dumps(tr.to_dict(), allow_nan=False)
        back = LivenessTracker.from_dict(json.loads(payload), P)
        assert back.to_dict() == tr.to_dict(), seed
