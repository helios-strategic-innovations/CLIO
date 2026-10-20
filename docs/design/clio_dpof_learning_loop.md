# CLIO DPOF — Learning Loop design frame (post-trade lessons + borderline debate)

**Author:** Buddy (Atlas Genius, four-persona deliberation; Jev consulted per the 2026-09-26 standing directive) · **Date:** 2026-09-26 · **Class:** DESIGN FRAME → build spec for three Sonnet 5 builders.
**Origin:** operator "Let's do it" on the TradingAgents evaluation (`memory/ref/tradingagents_eval_2026-09-26.md`): borrow (1) the post-trade reflection loop and (2) optional bull/bear context for borderline calls — as patterns, hand-rolled, no LangChain.
**Posture:** paper (DPOF default); both features ACTIVE by default (operator: build active; paper is the soak); purely additive to the decision model's context.

## 0. One paragraph
DPOF remembers positions but not judgment. The decider sees the last five closes as `{label, pnl, exit_reason, held_minutes}` (`engine.py:915-932`) — never *why* a trade worked or failed — and the only outcome hook (`_learn_from_exit`, `engine.py:627`) writes a one-line P&L note into CLIO's general memory, which DPOF never reads back. This frame adds the missing feedback edge: after every closed trade, the decision model writes a short lesson from **code-computed** outcome metrics; recent lessons (plus a code-computed tally) enter future entry/review prompts as context. Separately, on genuinely borderline calls only, two short parallel "bull case / bear case" arguments from a faster model are added to the decider's context. Neither feature decides anything: the single decision model (Opus 5.5 → GPT-6 Sol) still decides, code rails stay solvency + the 15:50 flatten, and every new call fails open.

## 1. Evidence (read this session, `[PRIMARY]`)
- `agent/dpof/decider.py`: provider order, per-provider circuit breakers, shared `max_decisions_per_day` cap persisted in `decider_state.json` (`:1033-1041`), CHARTER as cached system block (`:747-758`), `anthropic.Anthropic(api_key, timeout)` from `agent.config.config.anthropic_api_key` (`:732-745`), OpenAI via raw httpx.
- `agent/dpof/prompts.py`: CHARTER (`:57-86`), `_CTX_KEY_ORDER` + external-text envelope (`:156-176`, `:271-299`), `build_entry_message`/`build_review_message` (`:318-383`).
- `agent/dpof/engine.py`: `_build_ctx` (`:900-990`, sync, on-loop), `recent_closed` (`:915-932`), `_build_review_position_dict` (`:1003-1018`), review launch (`:1200-1220`), `_review_triggers` incl. `alignment_against`/`jev_escalation` (`:1050-1121`), entry launch (`:1580-1608`), `_run_decision_task` (`:744-762`), `decision_max_age_seconds` measured from decision *finish* (`:1251`, `:1652`), `exit_filled` close handling (`:2175-2196`), marks loop (`:2225-2243`), `_fire_and_forget` (`:586`).
- `agent/dpof/models.py:254-309` `PositionRecord` (no underlying price at entry/exit; no observed P&L extremes; `from_dict` filters unknown keys, so new fields are backward-compatible).
- `agent/dpof/orders.py:322` `entry_signal=signal.to_dict()` (composite/strength/direction/regime/pillars/reasons/vetoes; no underlying price).

## 2. Scope
**F1 — Post-trade lessons (reflection loop).** **F2 — Borderline debate context.**
**Non-goals / invariants (all builders):** no change to thresholds, gates, sizing, order handling, the decider's transports/circuit breakers/daily cap, or the flatten; lessons and debate are CONTEXT, never rules or votes; every new LLM call fails open (feature silently absent, logged at INFO); no sync file I/O or flock on the event loop; every model-written or external text reaching a prompt passes `sanitize_ingested_content(..., source=...)`; tests hermetic (tmp data dirs, fake transports, never `data/`, never network); plain-text Telegram only.

## 3. Rulings (Jev-advised, Buddy-ruled; Jev advises, does not rule)
| # | Decision | Ruling | Jev tool → choice (confidence) |
|---|---|---|---|
| R1 | Model that writes lessons | Same as the decision model: `decision_model` (Opus 5.5) → `fallback_model` (GPT-6 Sol) | `jev_next_step` → same_as_decision_model (0.98) |
| R2 | Reflection trigger | Status-driven scan each cycle (closed positions without a lesson), single-flight, off the cycle's critical path; restart-safe | `jev_next_step` → status_scan (0.96) |
| R3 | Debate on reviews | Only when the review trigger includes `alignment_against`, ≥ 30 min to the flatten, and no `jev_escalation` | `jev_next_step` → alignment_flip_only (0.90) |
| R4 | Debate model | `claude-sonnet-5` → `gpt-6-sol` fallback | `jev_next_step` → fast_model (0.88) |
| R5 | Where advisory calls live | New `agent/dpof/advisory_llm.py` with its own clients, deadlines, circuit breakers, budgets; `decider.py` untouched | `jev_next_step` → separate_advisory_client (0.98) |
(`jev_decide` rejected the first payload as `invalid_request`; all rulings used `jev_next_step`.)

## 4. F1 — Post-trade lessons
### 4.1 Data capture (engine, observability only)
New `PositionRecord` fields (defaults `None`; old `positions.json` loads unchanged):
- `entry_underlying_price` — the underlying's `last_price` from the snapshot when the entry is applied.
- `exit_underlying_price` — the underlying's `last_price` at the cycle that records the full close.
- `obs_max_pnl_pct`, `obs_min_pnl_pct` — observed extremes of the position's mark-based P&L % while open, tracked **in memory** each cycle in the marks loop and written onto the position once, at close (best-effort; reset by a restart — the lesson says "observed since <time>"). **Named `obs_*` on purpose:** the removed `peak_pnl_pct` backed the retired mechanical ladder; nothing in exits/review triggers may read `obs_*` (grep-tested).
### 4.2 Trade record (pure, `reflection.build_trade_record`)
From a closed `PositionRecord` + its review trail (decisions.jsonl `kind="position_review"` rows for that `position_id`, read off-loop) compute **in code**: underlying, direction, structure, qty, entry/exit times (ET), held minutes, minutes-to-flatten at entry, realized P&L $ and % of premium paid, underlying move in the trade's direction (bps) = `sign × (exit_u − entry_u) / entry_u`, observed max/min P&L %, exit reason category (`decider_exit` / `decider_trim` / `eod_flatten` / other), entry composite/strength/regime/pillars (from `entry_signal`), entry conviction/qty/provider (from `entry_decision`), thesis + invalidation, review count and the review actions/reasons/thesis_status sequence (capped at 6, text-capped). Math stays in code; the model only interprets.
### 4.3 Eligibility + discovery (R2)
Eligible: `status == "closed"`, filled (`qty_original > 0` and `entry_price` not None), `closed_time` within `reflection_lookback_days` (7), no stored lesson for that `position_id`, attempts < `reflection_max_attempts` (3), next-attempt backoff elapsed (10 min × attempt). Excluded exit reasons: `log_only`, `submit_failed:*`, `submit_unconfirmed`, `reconciled_absent`. Discovery each cycle, **after** step 13 (risk persist) and 13b (signal log): `engine` fires `reflection.run_pending(...)` via `_fire_and_forget` guarded by a single-flight flag; at most `reflection_max_per_run` (2) positions per run; everything off-loop.
### 4.4 The reflection call
`AdvisoryLLM.complete_json(purpose="reflection", ...)`, model per R1, timeout `reflection_timeout_seconds` (60), budget `dpof_reflection`. System prompt (static, in `reflection.py`): you are reviewing one of your OWN closed paper trades; state what the evidence says, not rules; one trade is a small sample; no generic advice. Input: the trade record rendered as a compact block inside `sanitize_ingested_content(..., source="dpof_trade_record")` (the thesis/reasons are model text that can echo headlines). Output schema (validated + clamped in code): `{"lesson": str ≤ 400 chars, "entry_quality": "good"|"mixed"|"poor", "exit_quality": "good"|"mixed"|"poor"|"forced", "key_factor": one of ["flow_signal","price_confirmation","timing","theta","event_risk","vol_pricing","sizing","exit_discipline","regime","other"], "confidence": 0..1}`. Invalid/timeout/both providers down → attempt recorded, retried later; never raises.
### 4.5 Store (`reflection.LessonStore`)
`<data_dir>/lessons.jsonl` (append-only, capped 5,000 lines, in-memory line counter + trim-to-90% like `signal_log.py`; sidecar-lock `.dpof_lessons.lock`, never flock + os.replace on the locked file) and `<data_dir>/reflection_state.json` (attempts/backoff per position_id; atomic write). Record: `{position_id, underlying, direction, closed_at (ISO ET), pnl_usd, pnl_pct, underlying_move_bps, held_minutes, exit_category, lesson, entry_quality, exit_quality, key_factor, confidence, provider, model, mode, created_at}`. Loaded once at startup (off-loop) into an in-memory cache; every new lesson updates the cache. **`_build_ctx` reads only the cache** (no I/O on the loop).
### 4.6 Prompt injection
`ctx["lessons"] = {"same": [...≤5 for the nominated/position underlying, newest first], "cross": [...≤3 other underlyings], "tally": {per underlying over the last 20 lessons: n, wins, losses, net_pnl_usd, avg_pnl_pct}}` (tally computed in code). `prompts.py` renders it in its own section after VOL SURFACE: `=== LESSONS FROM YOUR OWN RECENT CLOSED TRADES (context, not rules; small samples) ===`, one line per lesson (`date · SPY bullish · −$120 (−38%) · exit decider_exit · key: theta · lesson…`) plus the tally line, the whole block inside ONE `sanitize_ingested_content(..., source="dpof_lessons")` envelope, capped at `lessons_prompt_max_chars` (2,500). `lessons` joins `_CTX_SKIP_GENERIC_KEYS`. Entry AND review messages. CHARTER gains two sentences under WHAT YOU WILL SEE (lessons = your own past trades, observations not rules; weigh against n).
### 4.7 Visibility
One plain-text Telegram line per new lesson (`lesson_alerts`, default on): `DPOF lesson — SPY bullish −$120 (−38%, 42 min): <lesson>`. `/dpof lessons` (last 10 + tally, via `readmodel`, executor). EOD digest gains "Lessons today: n" + the day's lessons (truncated).

## 5. F2 — Borderline debate context
### 5.1 Gate (pure, `debate.should_debate_entry` / `should_debate_review`)
Entry: nominated signal with `strength == "normal"` and `|composite| − threshold_normal ≤ debate_margin` (0.05). (Strong signals and the regime-requires-strong path never debate.) Review (R3): `"alignment_against" in triggers`, `"jev_escalation" not in triggers`, `minutes_to_flatten ≥ debate_review_min_minutes_to_flatten` (30). Global: `debate_enabled` / `debate_on_reviews`.
### 5.2 Execution
Inside the existing decision Task (never the cycle): `engine` wraps the decider coroutine — `debate.run_debate(...)` first (bull and bear calls **in parallel**, `asyncio.wait_for` total `debate_timeout_seconds` 20 for entries / `debate_review_timeout_seconds` 12 for reviews), then `decide_entry` / `review_position` with `ctx["debate"]` set. Staleness is unaffected (measured from decision finish, `engine.py:1251`); the engine re-quotes before submit. Any failure → `ctx["debate"] = {"available": false, "reason": ...}` and the decision proceeds.
Input to each side (built in code, compact): the underlying's composite/pillars/strength, price vs VWAP + 15-min momentum, regime label, vol-surface line, event-risk summary, and the candidate menu (entry) or position summary (review), plus the lessons tally — inside one `sanitize_ingested_content(..., source="dpof_debate_input")` envelope. Side prompts: argue the strongest honest case FOR (bull) / AGAINST (bear) this specific trade/hold; no hedging; ≤ 120 words. Output schema: `{"argument": str ≤ 700 chars, "key_risk": str ≤ 200 chars, "strength": 0..1}`.
### 5.3 Rendering + logging
`prompts.py`: section `=== BULL CASE / BEAR CASE (arguments from a faster model — weigh them, they are not votes) ===` inside `sanitize_ingested_content(..., source="dpof_debate")`, only when `ctx["debate"].available`; `debate` joins `_CTX_SKIP_GENERIC_KEYS`. The decision_log for that deliberation/review gains `debate: {ran, available, latency_ms, bull_strength, bear_strength, provider, reason}`; `signals.jsonl` records are unchanged.
### 5.4 Model + budget
R4: `debate_model` `claude-sonnet-5` → `debate_fallback_model` `gpt-6-sol`; budget `dpof_debate` (200 calls/day = 100 debates). Over budget → skip (fail open).

## 6. `agent/dpof/advisory_llm.py` (R5) — interface both features code against
```python
@dataclass
class AdvisoryResult:
    ok: bool; parsed: dict | None; provider: str; model: str; latency_ms: int | None; error: str

class AdvisoryLLM:
    def __init__(self, *, anthropic_key_fn: Callable[[], str], openai_key_fn: Callable[[], str],
                 anthropic_transport=None, openai_transport=None, budget_fn=None,
                 clock=time.time, circuit_failures: int = 3, circuit_cooldown_seconds: float = 300.0): ...
    async def complete_json(self, *, purpose: str, system: str, user: str, schema: dict, schema_name: str,
                            primary_model: str, fallback_model: str, timeout_seconds: float,
                            max_tokens: int = 800, budget: str | None = None) -> AdvisoryResult: ...
    def status(self) -> dict: ...      # per-provider circuit state, calls/failures today (no secrets)
    async def close(self) -> None: ...
```
Never raises (returns `ok=False` with a redacted error). Own per-provider circuit breakers (separate from the decider's). Anthropic: `anthropic.Anthropic(api_key, timeout=...)` in `run_in_executor`, structured output via `output_config.format` json_schema with the same one-time downgrade on a 400 that `decider._is_output_config_400` handles (then parse the first JSON object from text); OpenAI: raw `httpx.AsyncClient`, `follow_redirects=False`, response size cap, `response_format` json_schema. `budget_fn(budget)` defaults to `agent.rate_limiter.check_and_record`; `BudgetExhaustedError` → `ok=False, error="budget"`, no call. Deadlines enforced with `asyncio.wait_for` per provider attempt: each attempt gets its own `timeout_seconds` (same as the live decider's two separate deadlines). The debate's outer `run_debate` deadline (20 s entry / 12 s review) is the real ceiling on its wall time; a reflection can take up to 2 × `reflection_timeout_seconds` in the background. `effort` (low/medium/high, default low) is passed through to both providers; reflection uses medium. Secrets never logged (`Bearer` + configured keys redacted).

## 7. Config (DPOFConfig — dataclass default == `load_dpof_config` default; env names)
F1: `reflection_enabled` True (`CLIO_DPOF_REFLECTION`), `lessons_in_prompt` True (`CLIO_DPOF_LESSONS_IN_PROMPT`), `reflection_model` "" → decision_model (`CLIO_DPOF_REFLECTION_MODEL`), `reflection_fallback_model` "" → fallback_model (`CLIO_DPOF_REFLECTION_FALLBACK_MODEL`), `reflection_timeout_seconds` 60, `reflection_lookback_days` 7, `reflection_max_attempts` 3, `reflection_max_per_run` 2, `lessons_same_underlying` 5, `lessons_cross` 3, `lessons_prompt_max_chars` 2500, `lesson_alerts` True (`CLIO_DPOF_LESSON_ALERTS`).
F2: `debate_enabled` True (`CLIO_DPOF_DEBATE`), `debate_on_reviews` True (`CLIO_DPOF_DEBATE_REVIEWS`), `debate_model` "claude-sonnet-5", `debate_fallback_model` "gpt-6-sol", `debate_margin` 0.05, `debate_timeout_seconds` 20, `debate_review_timeout_seconds` 12, `debate_review_min_minutes_to_flatten` 30.
Budgets — three sites each: `agent/config.py` dataclass field + `from_env` (`BUDGET_DPOF_REFLECTION`, `BUDGET_DPOF_DEBATE`) and `agent/rate_limiter.py` `budget_tracker.register(...)`: `dpof_reflection` 50/day, `dpof_debate` 200/day. Exhausted → `BudgetExhaustedError` (`agent/rate_limiter.py:60`).

## 8. Build slices
- **B1 (parallel)** — `agent/dpof/advisory_llm.py`, `agent/dpof/debate.py` (+ tests). Pure/injectable: every knob is an explicit parameter (no `DPOFConfig` field dependency).
- **B2 (parallel)** — `agent/dpof/reflection.py` (trade record, reflection prompt/schema/validation, `LessonStore`, `ReflectionService.run_pending` / `lessons_ctx` / `lessons_text`), `models.py` fields, `config.py` (ALL §7 knobs), `agent/config.py` + `agent/rate_limiter.py` budgets, `prompts.py` (lessons + debate sections, `_CTX_SKIP_GENERIC_KEYS`, CHARTER sentences), `readmodel.py` + `telegram_commands.py` (`/dpof lessons`, digest section), `.env.template` (+ tests). Codes against the §6 interface with a fake.
- **B3 (after B1+B2)** — `engine.py` only (+ tests): collaborators, data capture (§4.1), reflection scheduling (§4.3), ctx injection (§4.6), debate wrapper + decision_log fields (§5.2–5.3), lesson alerts; `docs/design/clio_dpof.md` §5c pointer.

## 9. Atlas Genius record (four-persona)
- **Odegaard (frames):** the gap is the missing feedback edge — DPOF records outcomes but never turns them into judgment the decider can use. Build the edge (F1) first; F2 is a cheap second opinion on the one decision class where the model is most likely to be wrong: the borderline. Confidence 9. Change-my-mind: if the decider already reads its own decision history, F1 shrinks to formatting.
- **Aidan (quants):** 0–10 closes/day means lessons are anecdotes; every lesson carries code-computed n/wins/net P&L so the model can see sample size, and lessons state the evidence, not rules. Cost: ~1 Opus call per close + ≤2 Sonnet calls per borderline nomination — noise next to the decider's own spend. Measure after ~30 closed trades: pass rate, win rate and average P&L on lesson-bearing vs early decisions (signals.jsonl + decisions.jsonl + lessons.jsonl). Confidence 8. Change-my-mind: if lessons converge on one generic line ("theta"), cut them to the tally.
- **Antony (kills):** (1) lessons hardening into de-facto rules → "context, not rules" framing, schema forbids imperatives-as-rules, n shown; (2) injection via lesson/debate text that echoes headlines → sanitize envelopes; (3) debate delaying an exit → alignment-only, ≥ 30 min, never with Jev escalation, 12 s cap, fail-open; (4) touching the live decider → separate client (R5); (5) sync I/O in `_build_ctx` → in-memory cache only; (6) lost/duplicate lessons across restarts → status-driven discovery keyed by `position_id` (R2); (7) `obs_*` extremes drifting into an exit rule → grep test. Confidence 8.
- **Harlan (structures):** B1 ∥ B2, then B3; one PR; pass-2 reviews before merge; deploy = pull + `sudo /usr/bin/systemctl restart moltbot` (collector untouched); rollback = `CLIO_DPOF_REFLECTION=0` / `CLIO_DPOF_DEBATE=0` + restart, or revert the merge. Confidence 9.
- **Consensus 4/4 (avg 8.5): build.**
