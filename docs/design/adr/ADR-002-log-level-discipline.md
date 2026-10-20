# ADR-002 — Log-Level Discipline for Production Observability

**Date**: 2026-04-29 (S186 W5)
**Status**: Accepted
**Decider**: Operator (Dominic) via S186 remediation plan acceptance

## Context

CLIO ships with `agent/logging_config.py` setting `root_logger.setLevel(INFO)` from `LOG_LEVEL` env (default INFO). DEBUG records are filtered at the logger hierarchy BEFORE reaching any handler — `file_handler.setLevel(DEBUG)` is moot. This was codified in S185 as anti-pattern #27 (G17).

The Apr 29, 2026 audit (S186) found this pattern manifesting as **silent staleness cascades**:

1. `trading_economics` monthly budget exhausted at 70/70 → `econ_calendar.fetch_calendar()` returns `[]` → `run_calendar_refresh` keeps cached events at WARNING (correct).
2. Cached events become stale (dates pass) → `get_upcoming_events()` returns `[]` → 7 downstream intel modules' `if not upcoming: return` branches at **DEBUG** (invisible).
3. The S185 mtime-gate WARNING fires correctly on all 7 modules — but operator has no signal of WHY each exits without saving.

**Core principle revealed**: A diagnostic log buried at DEBUG is functionally identical to no log at all in production.

## Decision

For early-exit paths in scheduled / escalation-triggered callbacks:

| Path category | Log level | Rationale |
|---|---|---|
| Upstream-data-empty (`if not upcoming: return`, `if not research_results: continue`) | **INFO** | Fires at most once per cycle per branch; surfaces cascades |
| Upstream-classification-miss (e.g. crisis classifier returns None) | **INFO** | Diagnostic value high, frequency low |
| Anthropic call returned None / generated nothing | **WARNING** | Recoverable but suspicious; potential silent failure of the analysis layer |
| Dedup window suppression ("Already analyzed X recently") | **DEBUG** | Fires every cycle within window; INFO would create noise |
| Per-loop-iteration heartbeat that fires hundreds of times per cycle | **DEBUG** | Already low-signal-per-line |
| Recovered-on-rotation API errors (DDG Yahoo backend, primp-handled) | **WARNING** | Eventually-successful failures are not ERROR; preserve error detail |
| Provider-side intermittent 5xx (UW net-prem-ticks etc.) | **WARNING** | Transient; track via counter if frequency rises |
| Genuine non-recoverable errors | **ERROR** | Reserved for "cannot proceed without operator" |
| Module init-time crashes, kill-switch firing, security violations | **CRITICAL** | Page operator |

**Counter pattern**: When a transient failure is observable but not actionable per-event, augment with a **module-level daily counter** (e.g., `scheduled_tasks._timeout_counter`) that resets on local-date change. Surface count in the existing log line (`...today_count=N`). Avoid creating a separate JSON file for low-value telemetry.

## Consequences

### Positive
- Operator reading INFO-level production journal sees WHY each scheduled job exits without state advance, not just THAT it exited.
- Matches Hydrinia's S181 Fix B `📥 Moltbot idea scan:` precedent (operator-visible canary at INFO).
- Compatible with S185 mtime-gated dedup: WARNING from the gate + INFO from the runner = full diagnostic chain.
- No log-volume explosion: only ~3-5 additional INFO lines per cycle when stale.

### Negative
- INFO log volume increases when upstream is degraded (e.g. TE budget exhausted for week → 7 modules × ~1 cycle/day = ~50 lines/week).
- Cosmetic increase in journal size; trivial vs. retention rotation.

### Neutral
- "Already analyzed recently" stays at DEBUG by design — invisible at default LOG_LEVEL=INFO. If operator wants visibility into dedup window behavior, they can bump LOG_LEVEL=DEBUG temporarily (env reload required per logging_config.py).

## Validation

After S186 deploy + 24h observation:
- Expect `INFO` lines from each of macro_research / fed_analysis / commodity_review / earnings_intel / bond_intel for at least one cycle (window-dependent; weekday-only modules will only fire Mon-Fri).
- Expect 0 new WARNING lines from `learning_cycle` timeout counter unless real timeouts occur (currently 2/day at session start).
- Expect ≤4 quarantine WARNING lines from Lighthouse (one per chronically-broken feed).
- Expect ≤1 quarantine WARNING from news_intel date-coherence guard (the existing Apr 29 Trump tariff item should be re-flagged on next refresh — though item is already saved, so flag will only apply to NEW items).

## Re-open trigger

Re-open this ADR if any of the following:
- Operator reports INFO-level journal volume is too noisy.
- Two or more modules adopt a "summarize the last N cycles in one log line" pattern (then INFO becomes redundant; consolidate).
- A new asyncio scheduler / replacement for apscheduler changes the call/return contract.
- LOG_LEVEL default changes from INFO (e.g., to WARNING) — would re-bury the upgraded INFO records.

## Related anti-patterns

- CLAUDE.md anti-pattern #27 (S185 G17): DEBUG-level logs invisible at INFO root.
- CLAUDE.md anti-pattern #25 (S184): grep -c counts lines, not matches — applies when verifying multiple INFO patterns are present in a single journal scan.

## Implementation references

- `agent/integrations/macro_research.py` — 3 paths upgraded (S186 commit `d2ab931`)
- `agent/integrations/fed_analysis.py` — 4 paths (S186 `f108849`)
- `agent/integrations/geopolitical_research.py` — 3 paths (S186 `73c849e`)
- `agent/integrations/commodity_research.py` — 2 paths (S186 `df8df96`)
- `agent/integrations/bond_intel.py` — 2 paths (S186 `8d07c7f`)
- `agent/integrations/trade_policy.py` — 2 paths (S186 `3dad36d`)
- `agent/integrations/earnings_intel.py` — 3 paths (S186 `46591eb`)
- `agent/integrations/web_search.py` — DDG Yahoo backend demote (S186 `6e29113`)
- `agent/integrations/scheduled_tasks.py` — `_timeout_counter` + `_record_timeout()` (S186 `7addc95`)
- `agent/integrations/global_news_feeds.py` — Lighthouse circuit breaker + WARNING quarantine log (S186 `6e1b112`)
- `agent/integrations/news_intel.py` — `_detect_off_year_only` + WARNING quarantine (S186 `3e95789`)
