# ADR-010 — Test isolation of live audit state

**Date**: 2026-10-12 (S215 wave 9)
**Status**: Accepted (pending deploy D1)
**Deciders**: Dom + Buddy
**Related**: ADR-008 (gateway; its safety notes), `agent/safety/audit_log.py`, `tests/conftest.py`, `tests/test_audit_log_isolation.py`, `tests/test_native_b2a_hydrinia_bridge_gates.py`, `scripts/morning_check.py`, CLAUDE.md "Runtime-State Protection" (test rules for live paths)

## Context

- `agent/safety/audit_log.py` held a hard-coded audit directory. Line 37 sets `AUDIT_LOG_DIR = os.environ.get("CLIO_AUDIT_LOG_DIR") or "/home/agent/openclaw/data/audit"`. Before the fix there was no test redirect for this path.
- Between 2026-09-27 and 2026-10-11, pytest runs wrote unsigned `directive_dispatched` records (EXECUTE on SPY) into the live audit chain. The counts by day were 1 (09-27), 31 (09-29), 1 (09-30), 4 (10-10) and 6 (10-11). The same runs also wrote `HMAC SECRET MISSING` and `(UNSIGNED)` startup rows. The cause was `tests/test_native_b2a_hydrinia_bridge_gates.py`, whose bridge-on dispatch reached the real audit logger. `tests/conftest.py` redirected other live paths but not this one. (Historical counts are from the operator's incident review. They were not re-derived from the audit files in this wave.)
- The chain stayed hash-intact across the October files (0 breaks), and the service's own rows are signed. The injected rows are protected records and cannot be deleted under the runtime-state rule.
- A second incident the same day: a builder ran a mutation of the guard itself, with the guard removed. That run appended 4 unsigned rows to the live chain.
- `scripts/morning_check.py` tracks unsigned rows per day as `AUDIT_UNSIGNED_KNOWN` (`morning_check.py:291`), with an entry for each recent day. A single global count cannot work: about 1,443 rows across the whole chain have an empty `hmac_signature`, mostly legacy. (The 1,443 figure is from the orchestrator's review, not re-counted here.)

## Decision drivers

1. Tests must never write to a live audit path, even by accident.
2. A guard that protects a live path must be tested without touching that path.
3. The live chain is append-only; recovery from a bad write is not possible, so prevention is the only control.

## Considered options

| Option | Verdict |
|---|---|
| Leave the injected rows and document them | Adopted for the existing rows. They are protected and cannot be removed. |
| Annotate the chain with a signed note | Rejected for now: requires the service HMAC secret, which was not used in this wave. |
| Make every test use a fake audit module | Rejected as the only control: it relies on each test author remembering. Kept as a longer-term option for new tests. |
| Env override plus autouse redirect plus guard test (adopted) | Adopted. See Decision. |

## Decision

1. **Env override.** `CLIO_AUDIT_LOG_DIR` overrides the audit directory. The default is unchanged (`audit_log.py:37`).
2. **Autouse redirect.** `tests/conftest.py` defines the autouse fixture `_isolate_audit_log` (`conftest.py:66-84`). For each test it sets `CLIO_AUDIT_LOG_DIR` and the module attribute `audit_log.AUDIT_LOG_DIR` to a `tmp_path_factory` directory, and resets the `audit_log._audit_log` singleton before and after the test. It uses its own `pytest.MonkeyPatch()`, not the `monkeypatch` fixture, to avoid reordering fixture teardown.
3. **Guard tests.** `tests/test_audit_log_isolation.py` covers the redirect. It checks a fresh interpreter with the env var set, that the bridge-on dispatch writes only into the temp chain (`test_bridge_on_directive_dispatch_audits_into_tmp_chain_only`), and that the singleton is latched within one test and unlatched at the start of the next.
4. **Precondition assert (not yet in the bridge test).** A test that can reach the audit log must assert the redirect before any write. The guard tests above do this. `tests/test_native_b2a_hydrinia_bridge_gates.py` contains no audit assertion at all (grep finds none), so it relies on the autouse fixture alone. Adding an explicit precondition there is an open item.
5. **Mutation runs.** A mutant of a guard that protects a live path must run in a sandbox export in which the live default is rewritten to a temp path. A mutant run must never run against the live tree.
6. **Morning check baseline.** `AUDIT_UNSIGNED_KNOWN` records the known unsigned count per audit day (`morning_check.py:291`). Only the last two days are judged (`morning_check.py:341-347`). Each day is compared with its own baseline.

## Consequences

**Positive**
- Pytest runs no longer write to the live audit chain, given the autouse fixture is in place.
- The morning check can flag new unsigned rows without being fooled by legacy history.

**Negative**
- The per-day baseline must be updated by hand when a known, accepted set of unsigned rows is recorded. This is operator maintenance.
- A new test that writes through a different path (a fresh subprocess without the env var, or a module that caches the path at import) is not covered by the autouse fixture. Only the subprocess case is covered by the env override.

**Risks**
- The conftest fixture resets a module singleton. A test that depends on singleton state across tests would break. The guard tests check this.

## Rollback

- Revert the commit that adds the conftest fixture and the guard test, then rerun the suite.
- The env override is harmless when unset. Reverting it is not required.
- The existing unsigned rows remain. Rollback does not remove them, and cannot.

## Verification

- `tests/test_audit_log_isolation.py` passes with the fixture in place.
- A mutation run that removes the redirect must write only into a sandbox export, never the live `data/audit`.
- After any test run, the live audit directory for the current day should show no new unsigned rows with a `directive_dispatched` event type.

## Lessons

1. A test fixture that redirects one live path is not enough. Every live path a test can reach needs its own redirect, and the redirect should be checked by a test that fails if it is removed.
2. Verifying a guard by mutating it is itself a write path. The mutation must run against a copy whose live default has been rewritten, never against the live tree.
3. A per-day baseline in the morning check is the only workable signal for unsigned rows. A global count mixes legacy history with new incidents.

## Open items

- Add an explicit redirect assertion at the top of `tests/test_native_b2a_hydrinia_bridge_gates.py`, so a regression in conftest fails there as well.
- Decide whether to annotate the 2026-09-27 to 2026-10-11 injected rows with a signed note. This needs the service HMAC secret and operator approval.
- Consider a fake audit module for new tests that do not need the real chain. This is a style decision, not a fix for the incident.

## Verification status of claims in this ADR

- Verified in code: `audit_log.py:37` env override and default; `conftest.py:66-84` autouse fixture and its behaviour; `test_audit_log_isolation.py` tests; `morning_check.py:291` baseline dict; `morning_check.py:341-347` two-day judging rule.
- Not verified in this wave: the historical unsigned counts per day, the 4-row mutation incident, the 1,443 global figure, and the 0 chain breaks. These are from the operator's incident review.
- Not present in code: an audit assertion in `tests/test_native_b2a_hydrinia_bridge_gates.py`.
