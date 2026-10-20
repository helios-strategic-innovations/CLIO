# ADR-009 — DPOF decider deadline, fallback policy and attempt visibility

**Date**: 2026-10-12 (S215 wave 9)
**Status**: Accepted (pending deploy D1)
**Deciders**: Dom + Buddy
**Related**: ADR-008 (CLIO Anthropic gateway; its appendix lists the timeouts this ADR changes), ADR-007 (DPOF dynamic weights), `agent/dpof/config.py`, `agent/dpof/decider.py`, `agent/dpof/engine.py`, `agent/dpof/attempt_log.py`, `agent/dpof/claim_check.py`, `agent/dpof/reflection.py`

## Context

- Since the gateway cutover on 2026-10-11 (ADR-008), the DPOF decider's primary model (Opus, via `api.oneprovider.dev`) has been slow. The gateway stats line at 12:24 ET on 2026-10-12 reported p50 17.8 s and p95 79.3 s for Opus. Sonnet p95 was 57.8 s. (Source: gateway stats log line, not re-measured in this wave.)
- The per-provider deadline is `decision_timeout_seconds = 60.0` (`agent/dpof/config.py:447`), read from `CLIO_DPOF_DECISION_TIMEOUT` (`config.py:787`). At p95 near 79 s, the primary misses its deadline on a meaningful share of calls.
- On 2026-10-12, 5 of 19 entry deliberations had fallen back to OpenAI `gpt-6-sol` by about 13:00 ET (5 of 29 by 15:10 ET, when the figures below were re-derived from `data/dpof/decisions.jsonl`, `finished_at - launched_at`). Three fallbacks took 67.3, 71.8 and 67.3 s (the 60 s primary timeout plus a short fallback call) and two took 7.4 and 8.1 s. Four of the five fallback verdicts were `enter` and one was `pass`: every `enter` verdict of the day was a fallback verdict. All 24 Anthropic entry verdicts at 15:10 ET were `pass`. The fallback path therefore decided every entry, which is not an intended outcome.
- The primary failure was invisible. The only record was the in-memory `last_error` on the provider status. No log line was written.
- The decider's record callback was never wired. `agent/dpof/engine.py:622` constructs `DPOFDecider(cfg)` with no `record=` argument, and `DPOFDecider.__init__` takes `record` as an optional parameter (`decider.py:704`).
- Worst-case latency per entry deliberation is the debate (bounded by the env deadline, 60 s) plus the primary (60 s) plus the fallback (60 s), about 3 minutes.
- The Anthropic SDK default `max_retries=2` was never overridden. `asyncio.wait_for` over `run_in_executor` cancels the awaiting coroutine, not the executor thread, so a timed-out SDK call keeps running in the background.

## Decision drivers

1. Entries should be decided by the primary model when it is healthy and merely slow, not by the fallback.
2. Every provider attempt must leave a record, including failed attempts and their usage.
3. No new failure surface may be added to the event loop or to the trading cycle.
4. Rollback must be a set of environment variables plus a restart, not a code revert.

## Considered options

| Option | Verdict |
|---|---|
| Raise the timeout only (`CLIO_DPOF_DECISION_TIMEOUT`) | Partly adopted. To be raised to 90 s through `.env` at deploy (not applied as of this writing; the live value is the 60 s code default). Alone it does not fix the invisible failure or the SDK-internal retries. |
| Hedged parallel request (primary and fallback at once) | Rejected: doubles spend on every decision and makes the primary's verdict non-deterministic relative to the fallback. |
| Lower effort to `low` | Rejected for now: no token data exists yet to show what it saves or costs in quality. Effort stays `medium` until that data exists. |
| Wire `record=` in `engine.py` | Rejected in this wave: `engine.py` belongs to another lane with uncommitted work. Deferred (see Consequences). |
| Switch the primary to Sonnet | Rejected: changes the decision model, which is a separate operator decision; not a deadline fix. |

## Decision

1. **SDK retries off by default.** New config field `decider_sdk_max_retries`, default `0` (`agent/dpof/config.py:456`), read from `CLIO_DPOF_DECISION_SDK_RETRIES` and clamped to 0..3 (`config.py:795`). The client is built with `max_retries=retries` (`decider.py:888-889`). The OpenAI fallback is the retry. Gateway empty-reply retries stay inside the HTTP transport (ADR-008, `CLIO_ANTHROPIC_GATEWAY_EMPTY_RETRIES`).
2. **Deadline raised by an `.env` change at deploy.** `CLIO_DPOF_DECISION_TIMEOUT=90` is added to `.env` when the code deploys (not applied as of this writing). The code default stays `60.0` so that a fresh checkout behaves as before. Effort stays `medium`.
3. **Per-attempt ledger.** `agent/dpof/attempt_log.py` writes one row per provider attempt to `data/dpof/decider_attempts.jsonl`. Fields: provider, requested_model, served_model, effort, timeout_s, latency_ms, status, error_class, usage, cost_usd, circuit, fallback_used, final_provider, input_digest, claim_violations. Controlled by `CLIO_DPOF_ATTEMPT_LOG` (default on, `config.py:457`) and `CLIO_DPOF_ATTEMPT_LOG_MAX_LINES` (default 20000, clamped 100..200000, `config.py:458,797`). `record()` schedules the write with `run_in_executor` and never raises (`attempt_log.py:36-44`). Rows are emitted from `decider.py:_emit_attempts` (`decider.py:1244`).
4. **Visibility logs.** A WARNING `DPOF decider primary failed provider=... class=... latency=... circuit=...` is emitted at most once per error class per minute (`decider.py:_warn_primary_failed`). Circuit transitions log at INFO: `circuit OPEN` (`decider.py:876`) and `circuit CLOSED` (`decider.py:862`).
5. **Claim checker, log only.** `agent/dpof/claim_check.py` flags two classes of contradiction in decider text: VWAP-relation and gap-value mismatches, and named single-pillar claims contradicted by another flow pillar at |value| >= 0.5. Controlled by `CLIO_DPOF_CLAIM_CHECK` with values `log` (default) or `off`. Any other value degrades to `log` (`config.py:117-122`). There is no enforce mode. Violations are logged and written to the attempt ledger as `claim_violations`. On the orchestrator's replay over 441 real texts, the checker flagged 6, all judged genuine (the four 2026-10-12 errors, one text that matched the position record, and one 2026-10-01 arithmetic slip). A first version flagged 138. (Replay counts were supplied by the orchestrator's 2026-10-12 review and were not re-run in this wave.)
6. **Thesis and lesson limits.** The persisted pass thesis limit goes from 300 to 1,200 characters (`decider.py:89`, `thesis_full`). The PASS log line keeps the 300-character `_one_line` (`decider.py:77,197`). Lessons are capped at 600 characters and cut at a sentence boundary. A cut must keep at least 40 percent of the cap, otherwise the hard cut is used (`reflection.py:72-89`).

## Consequences

**Positive**
- A healthy-but-slow primary gets 90 s before the fallback takes over.
- Every attempt, including failed attempts that report usage, is visible in one ledger.
- The primary's failure class and circuit state appear in the journal.
- The persisted pass thesis is no longer truncated at 300 characters.

**Negative**
- More fallbacks on transient primary errors, because SDK-internal retries no longer absorb them.
- An entry may wait up to 90 s for the primary before the fallback starts.
- Per-attempt cost now includes failed attempts that report usage. Cost totals will rise relative to the old per-decision view.

**Risks**
- The fallback model may decide more entries than before. This needs monitoring in `decider_attempts.jsonl` (`final_provider` and `fallback_used`).
- The claim checker is heuristic. Its log-only mode limits harm, but a false positive will still show up in ledger rows.

**Deferred (engine-dependent, not in this wave)**
- A `no_candidates` INFO log.
- `requested_model` on decision rows.
- Provenance for the event-countdown.
- Wiring `record=` in `engine.py:622`. Until it is wired, the decider's own record callback stays unused and the attempt ledger is the only record.

## Deploy and verification

- Deploy D1: merge, then restart with `sudo /usr/bin/systemctl restart moltbot` (exact NOPASSWD form).
- Operator sets `CLIO_DPOF_DECISION_TIMEOUT=90` in `.env` before the restart. (Operator action; not done by this ADR.)
- Verify after the first live session: `data/dpof/decider_attempts.jsonl` has rows; `journalctl -u moltbot` shows at most one primary-failed WARNING per class per minute.

## Rollback

- `CLIO_DPOF_DECISION_SDK_RETRIES=2` restores the SDK default.
- `CLIO_DPOF_DECISION_TIMEOUT=60` restores the old deadline.
- `CLIO_DPOF_ATTEMPT_LOG=0` disables the ledger.
- `CLIO_DPOF_CLAIM_CHECK=off` disables the claim checker.
- Revert the merge commit if the code itself must go. Then restart moltbot.

## Verification status of claims in this ADR

- Verified in code on the S215 base (`2dc0dca`): `decision_timeout_seconds` default and env; `decider_sdk_max_retries` default, env and clamp; attempt-log config, module and write path; claim-check config and module; `_warn_primary_failed` rate limit; circuit OPEN and CLOSED logs; `DPOFDecider(cfg)` built without `record=` at `engine.py:622`; thesis limit 1200; lesson cap 600 and 40 percent rule.
- Re-derived by the orchestrator at 15:10 ET from `data/dpof/decisions.jsonl` (read-only): the fallback count, the fallback latencies and the verdict split quoted in Context.
- Not verified in this wave: the gateway latency percentiles and the replay counts (441 / 6 / 138). These come from the 2026-10-12 log review and were supplied by the orchestrator.
- Not verified: the operator's `.env` value. Only the code default is verified.

## Appendix — timeouts after this ADR

| Setting | Code default | Operator value | Env |
|---|---|---|---|
| `decision_timeout_seconds` | 60 | 90 | `CLIO_DPOF_DECISION_TIMEOUT` |
| `decider_sdk_max_retries` | 0 | 0 | `CLIO_DPOF_DECISION_SDK_RETRIES` (0..3) |
| `debate_timeout_seconds` | 20 | unchanged here | `CLIO_DPOF_DEBATE_TIMEOUT` |

ADR-008's appendix still lists `decision_timeout_seconds` at 60. That row is superseded for the live service by this ADR.
