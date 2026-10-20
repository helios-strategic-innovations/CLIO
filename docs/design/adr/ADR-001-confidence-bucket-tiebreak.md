# ADR-001 — Confidence Bucket Snap: Midpoint Tie Behavior

**Date**: 2026-04-29 (S185 W4)
**Status**: Accepted
**Decider**: Operator (Dominic) via S185 remediation plan acceptance
**Revised**: 2026-04-29 — initial draft incorrectly asserted "always-downward" tie-break; replaced with empirical reality after W4 functional test exposed float-precision behavior.

## Context

CLIO normalizes LLM confidence values to a 7-bucket discrete set:

```
_CONFIDENCE_BUCKETS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
```

`learning_engine._normalize_confidence(value)` snaps an arbitrary float to the nearest bucket via `min(_CONFIDENCE_BUCKETS, key=lambda b: abs(b - v))`. When the input lands exactly on a midpoint (`0.30`, `0.40`, `0.50`, `0.60`, `0.70`, `0.80`), two buckets are mathematically equidistant — but in IEEE 754 floats, the distances are ALMOST never byte-equal, so a tie-break rule emerges from float precision rather than from intentional design.

S181 regime arithmetic and S182 anti-cluster bumps were both audited as midpoint-reachable concerns (S185 P2-4). Empirical investigation during S185 W4 functional verification disproved that worry: those operators move BETWEEN bucket values (±0.10 on already-bucketed inputs), so the result is always another bucket value, never a midpoint.

## Decision

**Midpoint inputs are treated as unreachable in deployed callers.** No code change. No deterministic tie-break rule is added. Future contributors must NOT rely on midpoint snap direction.

If a future caller becomes capable of producing midpoint inputs (e.g., averaging two confidences, applying multiplicative decay, etc.), this ADR must be re-opened and an explicit tie-break implemented.

## Empirical Behavior (current Python + CPython)

| Midpoint | Snaps to | Direction |
|---|---|---|
| 0.30 | 0.25 | down (exact tie, stable min) |
| 0.40 | 0.45 | up (IEEE 754: \|0.35-0.40\|=5.0e-2+ε, \|0.45-0.40\|=5.0e-2-ε) |
| 0.50 | 0.45 | down |
| 0.60 | 0.55 | down |
| 0.70 | 0.65 | down |
| 0.80 | 0.85 | up |

4-of-6 snap down, 2-of-6 snap up. The pattern is reproducible on a given Python build but is NOT a documented Python contract — it could shift across CPython versions.

## Rationale

1. **Practical reachability**: All current callers feed `_normalize_confidence` either LLM-emitted 2-decimal floats (already bucket-aligned) or post-arithmetic bucket values. Midpoint inputs do not occur.
2. **No behavior change**: S185 is hygiene; introducing an explicit tie-break would alter a code path that's only theoretically reachable. Risk > reward.
3. **Honesty over false determinism**: the prior version of this ADR asserted "always-downward" — that was wrong. Misdocumented behavior is worse than acknowledged uncertainty.

## Consequences

- Future callers MUST NOT rely on a specific midpoint tie direction.
- If midpoints become reachable, the audit-driven fix is to use explicit bucket-index math, e.g.:
  ```python
  for i, b in enumerate(_CONFIDENCE_BUCKETS):
      if v <= b:
          # tie-break rule: <= snaps DOWN to b if v < b, UP to b if v == b
          ...
  ```
- A regression test that pins midpoint behavior would also pin a Python implementation detail; not recommended.

## Verification

```python
from agent.integrations.learning_engine import _CONFIDENCE_BUCKETS
midpoints = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
for v in midpoints:
    snapped = min(_CONFIDENCE_BUCKETS, key=lambda b: abs(b - v))
    distances = {b: abs(b - v) for b in _CONFIDENCE_BUCKETS if abs(b - v) <= 0.06}
    print(f'{v} -> {snapped}  (distances: {distances})')
# Expected (current Python build):
#   0.3 -> 0.25, 0.4 -> 0.45, 0.5 -> 0.45, 0.6 -> 0.55, 0.7 -> 0.65, 0.8 -> 0.85
```

## Related

- `agent/integrations/learning_engine.py:_normalize_confidence` (S185 W4 inline comment)
- S181 A2 (regime arithmetic — bucket-aligned, never produces midpoints)
- S182 M3 (anti-cluster undo guard — same)
- S185 audit P2-4 (raised the concern; W4 verification disproved practical reachability)
