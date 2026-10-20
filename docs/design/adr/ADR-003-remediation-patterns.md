# ADR-003 — Standing Remediation Patterns

**Status**: Accepted (S189, 2026-05-20)
**Author**: Buddy (CC) under operator direction
**Related**: ADR-001 (confidence bucket tie-break), ADR-002 (log-level discipline), CLAUDE.md "Always Do" / "Anti-Patterns" sections.

## Context

Across S170B, S184, S185, S186, S187, and this session (S189), several patterns have recurred during remediation work without being formally codified. Each session re-derives them ad hoc, and audits occasionally re-flag the same shape of issue. This ADR pulls those into one place so future sessions reference rather than re-discover.

## Decision

The following four patterns are standing remediation discipline for CLIO. They are non-negotiable when the conditions below apply.

### Pattern 1 — Module-level dedup is acceptable when the alert window dominates the restart rate

**Condition**: A repeated event (alert, log line, circuit-breaker state, quarantine) is being deduped over a window W, and the service's typical restart rate R has W >> 1/R.

**Decision**: Use module-level state (a dict or `_last_ts` global). Do NOT persist to JSON.

**Reasoning**: Persistence adds operational surface (file lifecycle, lock semantics, atomic-write discipline, schema drift) for a benefit that is wiped by the next restart cycle anyway. The dedup window absorbs the loss.

**Precedents**:
- S184-followup `_athena_stale_alert_dedup` (6h alert window vs ~2 restarts/day)
- S186 W3.1 Lighthouse per-feed quarantine (24h window vs ~2 restarts/day)
- S189 SEAL multigen circuit breaker (10-min window vs ~2 restarts/day)
- S189 SEAL feedback freeze visibility (6h dedup vs ~2 restarts/day)

**When NOT to apply**: any dedup whose window meaningfully exceeds the restart rate (e.g. a 14-day window cannot live in module memory).

### Pattern 2 — New validators ship LOG_ONLY by default; operator promotes to active

**Condition**: A new validator, filter, or rejection gate is being introduced for content quality, integrity, or formatting.

**Decision**: Add an env flag `<NAMESPACE>_<VALIDATOR>_LOG_ONLY` defaulting to `"1"`. In LOG_ONLY mode, the validator logs WARNING with the detected pattern but does NOT mutate or reject the content. Operator inspects the WARNING volume over 24-72h and decides whether to flip the flag.

**Reasoning**: Validators ship with unknown false-positive rates. Active enforcement on day-1 risks blocking legitimate content. LOG_ONLY lets operator gather empirical data before activating. Mirror of CLAUDE.md "All `_LOG_ONLY` flags set to `0` in `.env` (operator-approved Mar 18 2026). Code defaults remain `"1"` for safety."

**Precedents**:
- S189 C1 slop-phrase post-processor (`CLIO_SLOP_FILTER_LOG_ONLY=1`)
- S189 C3 quantified-risk-factor validator (`CLIO_TRADE_IDEA_VALIDATOR_LOG_ONLY=1`)
- All MOLTBOT_*_LOG_ONLY flags listed in CLAUDE.md "Feature Flags"

**Anti-pattern**: shipping a validator that strips content on first deploy, then the operator finds out the next morning that 500 reports lost their conclusions.

### Pattern 3 — Pessimistic-merge rule for verification disagreement

**Condition**: Two verification passes (or two sub-agents, or pass-1 and pass-2 audits) reach different conclusions about whether a finding is real, severe, or already-fixed.

**Decision**: The more pessimistic assessment wins. Document the caveat even on PASS-marked items.

**Reasoning**: A false-pass risks shipping broken code. A false-fail wastes investigation time. The former is strictly worse — investigations are cheap, regressions are expensive. Codified in S170B + S185 verification protocol.

**Precedents**:
- S170B P0-3 v1 marked PASS by pass-1, FAIL by pass-2 → in-session RETRY repair (in-place truncate kept inode stable)
- S185 W4 ADR-001 "always-downward" claim disproved on functional matrix → corrected mid-session
- S189 audit sub-agent claims (RSSHub down, memory tier degenerate) verified in main thread before acting

### Pattern 4 — Read-only investigation BEFORE code change for ambiguous root cause

**Condition**: An audit surfaces an apparent issue, but the proximate cause is unclear (competing hypotheses, multiple files, cross-system dependency).

**Decision**: Phase 1 ALWAYS reads + documents in `memory/ref/sNNN_*.md` before any source edit. Phase 2 code changes reference the memo's determination.

**Reasoning**: Sub-agent audit FP rate is ~55% per CLAUDE.md. Shipping "fixes" against unverified root causes ships at least 1 false fix per session. The 1-2h read-only pass is the cheapest insurance available.

**Precedents**:
- S185 W0 diagnostic pivot — caught audit's "metadata-ghost" misread of intel state files
- S186 W0 diagnostic — root-caused staleness to TE budget exhaustion (resolved in S187)
- S189 Phase 0 — caught audit's "UW TTL = 600s" misread (actual: dynamic via Hydrinia-written `freshness_ttl_seconds`)

## Consequences

**Positive**:
- Future remediation sessions can cite these by ID instead of re-deriving.
- Audits that re-flag already-resolved items can be triaged by referencing the relevant pattern.
- Operator gating is preserved at the explicit boundary (env flag flip).

**Negative**:
- Patterns may calcify; future sessions should still question them if the condition changes (e.g. if multi-day dedup becomes important, Pattern 1 needs revisiting).

**Neutral**:
- ADR-003 itself becomes a load-bearing reference. Treat as authoritative; if reality diverges, amend in place rather than write a contradicting ADR-004.

## Validation

- Future audit reports cite ADR-003 by pattern number when classifying fixes.
- Future session summaries that introduce LOG_ONLY validators reference Pattern 2 in their decision-rationale section.
- Future cross-instance summaries note ADR-003 in the §1 "Templates / patterns" list.

## Revision History

- **v1 (S189, 2026-05-20)**: Initial codification of 4 patterns from S170B → S189 precedents.
