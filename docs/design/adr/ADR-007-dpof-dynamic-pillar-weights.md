# ADR-007 — DPOF dynamic pillar weights (liveness-gated, shadow-first)

**Date**: 2026-10-10 (Wave B3)
**Status**: Proposed (builder draft; NOT operator-approved; nothing is live. Default mode is `shadow` and decisions keep static weights.)
**Decider**: Operator (Dominic). Direction set 2026-10-10: "static weights → dynamic, automatic weights, even if it takes longer."
**Supersedes**: none. Full design: `docs/design/clio_dpof_dynamic_weights.md` (v2; section numbers below refer to it).

## Context

DPOF's composite (`agent/dpof/signals.py`) weights NOFA .40, premium .25, darkpool .20, Kronos .15, plus a GEX term (.15) that amplifies or dampens by `sign(pre)`. The composite renormalizes only over **available** pillars, so it cannot see a pillar that is available but dead.

- **Dead-pillar evidence**: premium scored exactly 0.0 on all 6,476 logged UW SPY cycles (before the ADR-006 field fix). The legacy darkpool score sits at ±0.002. Those two pillars held 0.45 of the weight with no information.
- **Research (R1, 10 logged sessions)**: no pillar shows a detectable predictive edge at 5/15/30/60 minutes. NOFA IC −0.017 (CI −0.149..+0.116); corrected premium +0.008 (−0.131..+0.146). The corrected premium is contemporaneous momentum (corr +0.38 with the trailing 15-minute return; partial IC about −0.02 after removing that return). The 80%-power minimum detectable |IC| is about 0.19. The data neither refutes nor confirms an edge.
- **Liveness rule** (K=40 refreshes, ε=0.005) flags the logged-dead pillars in every session with UW data, within about 2 hours of the open, and flags 0 of 20 healthy series.

## Decision

1. **L1 liveness gate (live-capable).** Premium and darkpool are tracked per (underlying, pillar, source). A pillar is dead when its last 40 in-session values are all near zero or identical. Revival needs 6 consecutive healthy values, a 30-minute dwell applies, and the weight ramps back in over 3 refreshes. Dead state persists across sessions.
2. **L2 quality diagnostic (built, persisted and logged; NOT applied by default).** A partial-IC multiplier is clipped to [0.8, 1.2] and applied only when `CLIO_DPOF_DW_QUALITY_APPLY=1`, `n_eff >= 350`, and the 95% CI excludes 0. At the current sample sizes it is inert by design.
3. **L3 conservation target.** The target is the sum of prior weights over the available pillars, dead ones included. When nothing is dead and L2 is inert, `dynamic == static` to 1e-12. A dead pillar's weight is redistributed, not removed. Shares are bounded to [0.05, 0.60] when at least 3 pillars carry weight.
4. **Mode** `CLIO_DPOF_DYNAMIC_WEIGHTS = off | shadow | live`. Code default `shadow`; an invalid value means `off`. Masking happens inside `_build_underlying_snapshot`, not by popping cache entries. Shadow rows go to a sidecar file, `pillar_weights_shadow.jsonl`, so `engine.py` is not edited.
5. **Offline report** `scripts/dpof_dynamic_weights_report.py` (stdlib only, read-only, stdout only) replays stored sessions and prints the design §6 checklist.

## Consequences

**Positive**: a dead premium or darkpool loses its weight within about 2 hours and regains it only on evidence. Every liveness transition is logged (WARNING on death, INFO on revival). Shadow mode leaves decisions unchanged, and the replay tool can verify the shadow log against the signal log.

**Negative**:
- **`min_pillars` consequence.** If premium and darkpool are both dead and not substituted by self-flow, `count_available` = NOFA + GEX = 2 < 3. The composite becomes unavailable and the engine vetoes with `data_unhealthy`. This is intended (the engine goes quiet visibly), but it needs explicit operator confirmation at promotion. In shadow it changes nothing.
- **Stale prose.** `agent/dpof/prompts.py:63` and `scripts/decisions_replay_corpus.py:244` still describe the 40/25/20/15 weights. Update both at promotion.
- **Amplitude-based liveness fits ratio-scale pillars.** The legacy darkpool (±0.002 to ±0.02) can flicker near the threshold while `CLIO_DPOF_DARKPOOL_NBBO` is off. Expect flip noise there until the NBBO flag is decided.
- **Evidence limit.** The data cannot show that any pillar adds predictive value. Dynamic weights are justified by liveness, not by an IC gain.
- **Revival needs variation (review panel, 2026-10-10).** The last 6 readings must not all be identical, otherwise a frozen non-zero feed would be killed by the stuck arm, revived by magnitude after the 30-minute dwell and killed again, a dead/alive duty cycle. A healthy pillar that saturates at exactly +/-1.0 for 6 consecutive refreshes and was flagged dead stays dead until one value differs (the 20-tick premium ratio is exactly +/-1.0 only about 71% of the time with cent-level data, so this clears quickly).
- **Shadow `dyn` is not the live decision when self-flow substitutes.** In live mode a dead UW pillar is masked in the snapshot and, with `selfflow_fallback` on, replaced by a self-flow reading that carries weight; shadow keeps the UW pillar in the snapshot and logs `dyn` without it. Compare shadow `dyn` with `static`, not with what live would have done, when a dead episode coincides with a self-flow substitution.
- **L2 cannot open under the default decay.** At most 11 block samples per (underlying, pillar) per session with a 20-session half-life bound `n_eff` at 11/(1-0.9659) = 322.9, below `n_floor = 350`, so the quality multiplier can never apply. L2 is a logged diagnostic only (`CLIO_DPOF_DW_QUALITY_APPLY=0`, and criterion 6 below is a separate decision); `n_floor` and the half-life must be reconciled before that decision is taken.
- **Shadow record schema.** `m` is the diagnostic multiplier (moves off 1.0 as samples accrue); `ma` is the multiplier actually applied (1.0 while quality is off). The morning-check identity invariant (`dyn == static` with nothing dead and nothing applied) reads `ma` and falls back to `m` on older rows.
- **Backup growth.** `pillar_weights_shadow.jsonl` reaches its 40,000-line cap (about 14.5 MB) after about 25 sessions. `~/.claude/hooks/runtime_state_backup.sh` rsyncs `openclaw/data` hourly with `--link-dest`, so a changed file is stored in full in every hourly copy during market hours, the same shape as the existing 17 MB `signals.jsonl`. Exclude it in the hook if disk becomes a concern (the hook is operator-owned).
- **I/O plumbing.** Shadow appends and state saves run on the default executor, capped at 4 writes in flight; beyond that a write is dropped with one hourly WARNING (shadow rows are diagnostics, a dropped state save stays dirty and is retried next cycle). A dedicated 1-worker pool (the CLAUDE.md #37 pattern) is the next step if that warning ever appears. The first snapshot waits at most 3 s for the state load and otherwise continues with empty state.

## Rollback

- Set `CLIO_DPOF_DYNAMIC_WEIGHTS=off` and restart moltbot (`off` is byte-identical to the static path).
- Code: `git apply -R <wave B3 patch>`.
- The state file `data/dpof/pillar_weights.json` and the shadow log are runtime data. Tooling never deletes them; they are kept for audit.

## Promotion criteria (summary of design §6; the operator decides)

1. At least **15 RTH sessions** of shadow data, zero exceptions in the dynamic path, and a replay that reproduces the logged shadow values.
2. Every dead episode detected within K and recovered within R on planted-episode replay. At most 1 liveness flip per pillar per 30 minutes.
3. Nomination episodes within [0.67×, 1.5×] of static in every session, with every divergence explained.
4. No pillar above the cap. With nothing dead and L2 inert, dynamic equals static on every cycle.
5. Operator confirms the `min_pillars` consequence and updates the stale prose listed above.
6. **L2 → live is a separate decision**: block-bootstrap 95% CI of partial IC excludes 0 over at least **120 sessions**, with the same sign in both halves.

## Open questions

1. Whether any pillar has a real predictive edge. This needs at least 16 pooled sessions at 22 blocks per session to reach MDE 0.15.
2. Healthy darkpool and premium value distributions on regular-session UW data.
3. UW refresh phase, and whether the Alpaca bar timestamps are open-stamped (R1 assumed so; the choice is conservative under either).
4. A frozen UW feed with a fresh fetch timestamp is not caught by value-based liveness. The existing 3× refresh staleness check covers fetch failures only.
5. Whether to keep `min_pillars = 3` once premium and darkpool can legitimately be unavailable.
