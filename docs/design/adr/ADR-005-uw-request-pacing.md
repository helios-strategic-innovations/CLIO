# ADR-005 — Process-wide pacing and short backoff for Unusual Whales requests

**Date**: 2026-09-29 (S211)
**Status**: Accepted (implemented on `fix/clio-uw-basic-plan-pacing`; deploy operator-gated)
**Decider**: Operator (Dominic) — "yes, remediate" after the 2026-09-29 investigation; plan in `docs/plans/S211_uw_basic_plan_remediation.md`
**Supersedes**: the S133 HI-10 "sleep for `x-uw-req-per-minute-reset`, capped at 5 min" 429 handler in `uw_flow_intel._uw_request`

## Context

The old UW key was effectively dead until 2026-09-28 (Monday 09:30–12:26 ET: 20 UW calls, all 401/denied), so CLIO's UW polling had never run at full volume against the **UW Basic** plan ($150/month, **40,000 requests/day**, resets 8 PM ET). UW documents no per-minute or burst limit and no 429 semantics (checked: `unusualwhales.com/public-api`, `api.unusualwhales.com/docs/{conventions,errors}.md`, `llms.txt`). The response headers are uninformative: `x-uw-req-per-minute-remaining` is a constant 1,000,000 and `x-uw-req-per-minute-reset` a constant 60000, even on a 429.

On 2026-09-29, 216 of 1,184 requests (18%) were rejected with 429 while volume was tiny (peak 52/min, ~3K/day of 40K). Grouping the quota ledger's reservations by wall-clock second:

| Requests in the same second | Requests | 429s |
|---|---|---|
| 1–3 | 488 | **0** |
| 12–14 | 680 | ~31% |

So the plan rejects **bursts**. CLIO produces exactly that burst: DPOF fires 12–14 requests at the same instant every 180 s (`asyncio.gather` over six refreshers × two tickers), and `_REQUEST_DELAY = 0.3` was an `asyncio.sleep` inside each concurrent coroutine — all slept together and fired together. The 429 handler then slept 60 s, but DPOF cancels each call at 8 s (`asyncio.wait_for`), so **the retry never ran** (zero `*_retry` ledger rows). The last requests in the gather (NOPE, max-pain) failed 41–42% of the time; the first (net-prem-ticks) 0%. The effect was DPOF flapping between UW and its self-computed fallback flow source.

## Decision

1. **Pace at the one choke point.** All UW traffic goes through `uw_flow_intel._uw_request` (six modules, ~31 call sites). A process-wide **slot-reservation pacer** (`agent/integrations/uw_pacer.py`) hands out send slots 1/rate apart: default **3 req/s, burst 1**. It uses a `threading.Lock` and a token balance that may go negative (a reserved future slot), so it is loop-agnostic and FIFO; a cancelled waiter just forfeits its slot.
2. **Short, honest 429 handling.** On a 429: back off `Retry-After` if numeric, else 1.5 s (clamped 0.5–10 s, + ≤0.5 s jitter); ignore the constant `x-uw-req-per-minute-reset`; halve the pacer rate for 60 s (at most once per 5 s, floor 0.5 req/s); take a fresh paced slot; retry **once**. One WARNING per 30 s with a suppressed-count.
3. **Rollback switch**: `CLIO_UW_PACER_ENABLED=0` restores the legacy fixed sleep. Tunables: `CLIO_UW_RATE_PER_SEC`, `CLIO_UW_BURST`, `CLIO_UW_PENALTY_S`, `CLIO_UW_429_BACKOFF_S`.

`burst = 1` is deliberate. With burst 3 at 3 req/s a single calendar second can hold 5–6 sends, above the only regime observed safe (≤3). The cost of burst 1 is small: DPOF's 14-request refresh takes ≈4.3 s (13 slots × 1/3 s) instead of ≈3.7 s, inside its 8 s per-call and 15 s refresh budgets.

## Considered options

| Option | Verdict |
|---|---|
| Stagger only DPOF's refreshers | Rejected: other modules (flow scan, vol surface, bridge, newsflash) share the key and can burst too; DPOF-only leaves the choke point unprotected. |
| `asyncio.Semaphore`/queue-based limiter | Rejected: loop-bound primitives; some callers run on other loops/threads. |
| Raise DPOF's 8 s timeout to outlast the 60 s wait | Rejected: makes a stalled UW block the 30 s DPOF cycle; the 60 s wait itself is baseless (constant header). |
| Upgrade to UW Advanced ($375, "unlimited requests") | Not chosen: the daily quota is not the problem and burst limits are undocumented for every tier. |
| Websocket streaming (included in Basic) | Deferred: a redesign, not a remediation. |

## Consequences

**Positive**: bursts disappear (expected 429 rate <2%), DPOF's UW pillars stop flapping to the fallback, the retry finally runs, ~130/day `UW_HEADER_COUNT_DECREASE` reorder errors shrink (fewer concurrent responses), warning volume drops.
**Negative**: every UW call may wait up to a few seconds behind others in a burst; a DPOF refresh is ~0.6 s slower.
**Risks**: the true burst threshold (between 3 and 12/s) is unknown — 3 req/s is a starting value to be tuned from the ledger's 429 telemetry; the `x-uw-*` headers cannot be used to self-tune. `agent/integrations/unusual_whales.py` (chat tool) and `econ_calendar` were not audited for a private HTTP path that bypasses the choke point.
**Known limitation (cancelled waiters)**: DPOF cancels a call at 8 s; the slot that call had reserved is forfeited, not refunded (refunding a mid-queue slot would let two requests share an instant). At 3 req/s a 14-request burst ends at ≈4.3 s, so nothing is cancelled. If 429s halve the pacer to ~0.5–1.5 req/s, a burst can outlast 8 s, waste up to ~14 slots and delay other callers by up to ~28 s of "debt"; with DPOF's 180 s cadence this self-heals before the next burst. Revisit with ledger telemetry if it ever shows.
**Verification (live, read-only)**: `python3 scripts/uw_ledger_report.py` (429 rate, same-second burst histogram, retry rows) — ledger rowid-bisect query — 429 rate, same-second burst histogram, `*_retry` rows > 0, `flow_source_counts` mostly `uw`.

## Related

- `docs/plans/S211_uw_basic_plan_remediation.md` — issues UW-1, UW-2 (and UW-3…UW-7 follow-ons)
- ADR-002 (log-level discipline) — the 429 WARNING is deduplicated, the rest at DEBUG
- ADR-003 (remediation patterns)

## Addendum 2026-10-12: minute-only headers

**Date**: 2026-10-12 (S215 wave 9). Appended; the decision above is unchanged in substance.

**Context.** Since about 2026-10-07, UW 200 responses have carried only the per-minute headers (`x-uw-req-per-minute-*`). The daily-usage headers have not been present. `UWQuotaClient`'s observe path had no `observed_at` update for this shape, so the ledger stayed stale on every cycle. `reserve()` then denied all non-P0 calls except one probe per 60 s. On 2026-10-12, 273 of 1,932 attempts were denied, with about 39.7k of the 40k daily budget unused. `/historical-risk-reversal-skew` is P1, so the vol-surface skew was null for that day.

**Decision.**
- A 200 that is minute-only refreshes `observed_at`. Controlled by `UW_QUOTA_MINUTE_ONLY_FRESH` (default `1`, `agent/integrations/uw_quota_client.py:865`). The refresh is applied with `COALESCE` (`uw_quota_client.py:499-511`).
- `observed_count`, entitlement and `header_schema_ok` are unchanged. `local_unconfirmed` still counts. A 429 or 5xx never refreshes the ledger.
- `remaining` and the P0/P1 reserves still bind. The refresh restores freshness only; it does not add budget.
- The `DEGRADED_STALE_CACHE` denial logs at INFO once per episode, then at most every 30 minutes (`agent/integrations/uw_flow_intel.py:225-236`, `_QUOTA_DENIED_STALE_LOG_INTERVAL_S = 1800`). Real denials (budget and priority reserves) keep the WARNING.
- The morning-check tolerance for `UW_ACCOUNT_QUOTA_DENIED` is 3 (`scripts/morning_check.py:45`). The earlier value of 10 was not re-verified in this wave.

**Risks.**
- The ledger no longer reconciles to UW's own daily count while the daily headers are absent.
- Calls that bypass the client stay invisible. `agent/integrations/unusual_whales.py` was not audited by ADR-005 and still is not.
- If UW restores the daily headers, the schema branch takes over automatically. No code change is needed for that case.

**Rollback.** `UW_QUOTA_MINUTE_ONLY_FRESH=0` plus a restart of moltbot. The pre-fix behaviour returns: every cycle stays stale, and non-P0 calls are denied except one probe per 60 s.

**Verification.** After the next live session, `scripts/morning_check.py` should show `UW_ACCOUNT_QUOTA_DENIED` counts at or below 3 per day. The 273-of-1,932 pattern should not recur. (Those 2026-10-12 counts come from the orchestrator's review and were not re-derived here.)
