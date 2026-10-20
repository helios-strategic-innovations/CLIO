# ADR-008 — CLIO routes Anthropic traffic through the OneProvider gateway

**Date**: 2026-10-11 (OP-16, OP-19)
**Status**: Accepted (operator-initiated)
**Deciders**: Dom + Buddy
**Related**: `docs/api/oneprovider.md` (reference), `docs/api/anthropic.txt` (stale-banner refresh), ADR-006, ADR-007 (earlier DPOF ADRs; no change to their decisions), merge commit `385239c` (feat/clio-oneprovider-gateway-s214)

**Operator acceptance**: initiated by Dom on 2026-10-11 (he supplied the gateway key and base URL for CLIO).

## Context

- CLIO's Anthropic key was the same key Claude Code uses, and it was nearly out of usage. Claude Code and CLIO therefore shared a quota that was close to exhausted.
- Dom issued a separate OneProvider key for CLIO. OneProvider is an Anthropic-compatible reseller at `https://api.oneprovider.dev` (no `/v1` suffix). Auth is a raw `sk-` + 64 hex key sent in `x-api-key` or as `Authorization: Bearer`. Keys are valid for up to 90 days; the live key expires 2027-01-09. Billing is prepaid, with about $350 of balance at the start. The balance endpoint `GET /v1/dashboard/balance` accepts Bearer auth only.
- CLIO moved to Opus 5.5 and Sonnet 5.5. Canonical gateway IDs are `claude-opus-5-5` and `claude-sonnet-5-5`. The dotted form `claude-sonnet-5.5` returns HTTP 400 on the gateway. `claude-haiku-5-5` does not exist on the gateway; `claude-haiku-4-5-20251001` does.

**Live probe findings** (about 250 probe calls on 2026-10-11). The gateway is **not** a transparent proxy:

1. `output_config.format` (structured outputs) is ignored. The DPOF charter and messages never name the answer keys, and the real decider payload returned wrong-shaped JSON 4 of 4 times. With the JSON schema embedded in the prompt, 40 of 40 replies were schema-valid across decider entry and review, debate bull and bear, and reflection.
2. `tool_choice: {"type": "none"}` is ignored; the reply is either empty or a `tool_use`. An explicit "no more tools" text block gave 10 of 10 text replies. Omitting `tools` entirely also worked on the gateway.
3. The response `model` differs from the requested model. Every Opus ID returns `claude-opus-5`; every Sonnet ID returns `claude-sonnet-5`. **What is billed is unverified**; it must be checked on the cabinet Usage page.
4. Intermittent empty HTTP 200 replies (0 output tokens). Observed 0 of 72 on generic request shapes and up to about 30% on some Opus tool-loop follow-ups. The SDK never retries an empty 200.
5. No prompt-cache read hits were observed, and about 63 hidden prefix tokens appear per call. **Billing impact unverified.**
6. Latency is 2.5–12 s per call. A bull plus bear debate pair took 8–24 s wall-clock, depending on load.
7. Haiku returns a thinking block first, so `response.content[0].text` is unsafe on that model.

**Anthropic documentation** (fetched 2026-10-11) relevant to the design:
- Opus 5.5 and Sonnet 5.5 have adaptive thinking on by default (always on for Opus). Default effort is medium for Opus and high for Sonnet.
- Non-default `temperature`, `top_p` or `top_k` returns 400 on Sonnet 5.5. So `sampling_model` stays `claude-sonnet-4-6`.
- Forced tool use (`tool_choice` `any` or `tool`) returns 400 on 5.5 models. `auto` and `none` are accepted.
- Within a tool-use turn, thinking blocks must be passed back unmodified.
- Structured outputs are GA via `output_config.format`.
- Cache reads cost 0.05x on 5.5 models; 5-minute cache writes cost 1.25x. The minimum cacheable prompt is 512 tokens on Opus 5.5 and Sonnet 5.5.

## Decision

1. **One seam.** `Config.anthropic_client_kwargs(timeout, **extra)` in `agent/config.py` is the only way to build an Anthropic client. `tests/test_anthropic_client_invariant.py` enforces this with an AST check over committed files.
2. **Route resolution** via `CLIO_ANTHROPIC_GATEWAY=auto|on|off`. `auto` uses the gateway if and only if `ONEPROVIDER_API_KEY` is set.
3. **Host pinning.** The gateway key may be sent only to `https://api.oneprovider.dev`. Userinfo, query, fragment, port and path are rejected. Invalid gateway config fails closed: the key is empty, and `main.py` refuses to start.
4. **No automatic fallback** to the shared direct key, which is Claude Code's key.
5. **Compensations, active only in gateway mode:**
   - Schema-in-prompt: `agent/dpof/schema_prompt.py`, used by the decider and advisory payload builders.
   - Guarded httpx transport in `agent/anthropic_gateway.py`: bounded retry of empty replies, served-model mismatch log, per-model latency stats, and an hourly `gateway stats:` log line. Kill switch `CLIO_ANTHROPIC_GATEWAY_TRANSPORT=0`. Retry count `CLIO_ANTHROPIC_GATEWAY_EMPTY_RETRIES`, default 2.
   - Forced-summary turn hardening in `agent/core.py`: an explicit instruction block plus a tool-free retry.
   - Thinking blocks are preserved on the direct API and stripped on the gateway (`CLIO_PRESERVE_THINKING_BLOCKS=auto|1|0`).
   - `agent/anthropic_utils.first_text` replaces direct indexing of `content[0].text`.
6. **Log scrubbing.** Bare `sk-` keys are scrubbed in `agent/log_scrub.py`.
7. **Monitoring.** `agent/integrations/oneprovider_balance.py` checks balance, expiry and a served-model canary every 6 hours. Balance alerts: warn below $75, critical below $25. Expiry alerts: warn within 14 days, critical within 3 days. State is in `data/oneprovider_balance_state.json`.
8. **Health classification.** `agent/integrations/anthropic_health.py` is gateway-aware.
9. **Probe tool.** `scripts/gateway_probe.py` is repeatable, with suites `models`, `structured`, `toolchoice`, `empty`, `latency` and `balance`.

**Alternatives rejected:**
- Bridging through `os.environ` (`ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`) at `agent.config` import time. This is hidden process-global state, misses processes that do not import `agent.config`, and cannot carry the compensations.
- A single point of fallback to the direct key.

## Privacy

Every CLIO prompt now transits a third-party reseller. This includes DPOF context with positions and quantities, memory tiers, news and knowledge content, Telegram chat content, and voice transcripts. The vendor states the service is hosted in EU/Frankfurt with "no prompt logging". **Those statements are unverifiable from our side.**

Mitigations:
- https only, with host pinning.
- The key is never logged, and bare keys are scrubbed.
- Kill switch: `CLIO_ANTHROPIC_GATEWAY=off`.
- A per-key spend limit is recommended in the cabinet (operator action; not yet confirmed as set).

## Consequences

**Positive**
- CLIO has its own funded credential. Claude Code's quota is no longer shared with CLIO.

**Negative**
- The served and billed model identity is unverified. Decision rows record the **served** model string (for example `claude-opus-5` instead of `claude-opus-5-5`). This is a value change visible to the decision record consumers of `decisions.*.model`. It is not a schema change. Dom must relay it to the decision-record seats.
- Cache economics are unverified. The cost estimate is handled separately.
- The key expires 2027-01-09. Renewal runbook: issue a new key in the cabinet at app.oneprovider.dev, or through the Telegram bot @oneprovider_robot. Update `ONEPROVIDER_API_KEY` in `.env`, then run `sudo /usr/bin/systemctl restart moltbot`, then `moltbot-voice` and `moltbot-control`.
- Rolling back to the direct key is only as good as its remaining credit. That credit is unknown and reported as nearly spent.

## Rollback

- `CLIO_ANTHROPIC_GATEWAY_TRANSPORT=0` drops the transport guard.
- `CLIO_DPOF_DEBATE=0` disables the debate path.
- `CLIO_ANTHROPIC_GATEWAY=off` returns to the direct key.
- `git revert -m 1 385239c` reverts the merge. `bf5f7ad` is the pre-merge main hash (the parent of merge commit `385239c` on main).
- No data migration is involved. The only new runtime file is `data/oneprovider_balance_state.json`.

## Appendix — Timeouts vs measured latency

| Setting | Value | Note |
|---|---|---|
| `decision_timeout_seconds` | 60 | DPOF decider |
| `reflection_timeout_seconds` | 60 | DPOF reflection |
| `debate_timeout_seconds` | 60 | was 20 |
| `debate_review_timeout_seconds` | 45 | was 12; set in `.env` on 2026-10-11 via `CLIO_DPOF_DEBATE_TIMEOUT` / `CLIO_DPOF_DEBATE_REVIEW_TIMEOUT` |
| core `_API_TIMEOUT` | 120 | `agent/core.py` |
| SDK per-call timeout elsewhere | 120 | all Anthropic clients |
| Measured gateway latency | 2.5–12 s per call | 2026-10-11 probes |
| Measured debate pair | 8–24 s wall-clock | bull plus bear, load-dependent |

The review-side debate adds up to 45 s before an exit REVIEW decision. This applies when the trigger is `alignment_against` and at least 30 minutes remain to flatten. It is advisory and fails open.

## Open questions

1. What is actually billed per model alias (`claude-opus-5-5` vs served `claude-opus-5`)? Check the cabinet Usage page.
2. What is the real prompt-cache behaviour through the gateway?
3. Does thinking-block pass-back work correctly on the gateway?
4. Should `ANTHROPIC_API_KEY` be removed from CLIO's `.env` after a clean week on the gateway?
5. Should a spare second gateway key be provisioned?
6. Whether the gateway's per-key spend limit should be set in the cabinet (recommended; not confirmed as set).
