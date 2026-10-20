"""Regression test for the P1-9 retry-jitter max_delay overshoot bug.

``_compute_delay`` (agent/rate_limiter.py) adds jitter to de-correlate
lockstep retries. The jitter MUST be clamped so the returned delay never
exceeds the caller-supplied ``max_delay`` contract — some callers pass
``max_delay`` as a hard SLA/UX bound (e.g. ``retry_on_transient``'s default
30s), and code elsewhere clamps a ``Retry-After`` header to ``max_delay * 2``
on the assumption that an un-jittered/base delay never gets close to that
ceiling on its own.

A prior version added jitter AFTER the max_delay clamp
(``min(base, max_delay) + random.uniform(0, delay * 0.25)``), which could
overshoot ``max_delay`` by up to 25%. The fix clamps AFTER adding jitter.
"""
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.rate_limiter import _compute_delay  # noqa: E402


def test_compute_delay_never_exceeds_max_delay():
    """Across many attempts/statuses/seeds, the jittered delay must stay <= max_delay."""
    max_delay = 30.0
    base_delay = 1.0
    statuses = [None, 429, 500, 503]

    rng_state = random.getstate()
    try:
        random.seed(0)
        for attempt in range(1, 12):  # exponential growth quickly saturates max_delay
            for status in statuses:
                for _ in range(200):  # sample the jitter distribution
                    delay = _compute_delay(attempt, status, base_delay, max_delay)
                    assert delay <= max_delay, (
                        f"jittered delay {delay} exceeded max_delay={max_delay} "
                        f"(attempt={attempt}, status={status})"
                    )
                    assert delay >= 0
    finally:
        random.setstate(rng_state)


def test_compute_delay_still_applies_jitter_below_cap():
    """Sanity check the fix didn't neuter jitter entirely: small attempts (delay well
    under max_delay) should still show variance across draws, proving the clamp only
    bites when jitter would overshoot the cap, not unconditionally."""
    base_delay = 1.0
    max_delay = 30.0

    rng_state = random.getstate()
    try:
        random.seed(1)
        samples = {_compute_delay(1, None, base_delay, max_delay) for _ in range(50)}
        # attempt=1 => base delay before jitter is small (well under max_delay), so
        # jitter should produce more than one distinct value across 50 draws.
        assert len(samples) > 1
    finally:
        random.setstate(rng_state)
