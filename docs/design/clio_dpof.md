# CLIO DPOF — Design (v2, 2026-09-24)

**v2 supersedes v1's mechanical trading model.** Operator directive
(2026-09-24, same day as v1): *no mechanical trading*. Entries AND exits are
judged by CLIO's own decision model (Anthropic Opus 5.5, fallback OpenAI
GPT-6 Sol — `agent/dpof/decider.py`); code only enforces SOLVENCY (no
hard percentage/count caps left) and the one kept rail, the 15:50 ET forced
flatten. §§4-8 below describe the v2 flow; §1 keeps its v1 recon table as
history (the NOFA composite/regime/price-confirmation SIGNAL math it
describes is unchanged — only what code does with that signal changed).

Long-options-only execution lane inside CLIO. Trades **long calls / long puts**
(optional **debit verticals**, off by default) on 1-2 liquid underlyings
(default `SPY`, `QQQ`) in a **dedicated Alpaca account** (~$20K target), driven
by the Hydrinia DPOF method, with an optional **Jev (TypeSafe)** fast
advisory/veto layer.

Status: **active by default** — `CLIO_DPOF_MODE` defaults to `paper`
(operator directive, 2026-09-24: "paper mode IS the soak" — there is no
separate log-only staging period — and "everything is built active, not
shadow" — every default is the ACTIVE mode). Ladder: `off → log_only → paper
→ live`, still available via `CLIO_DPOF_MODE` for an operator who wants to
stage manually. Live requires every gate in §7 regardless of this default —
nothing here weakens them.

## 1. Why not port Hydrinia's DPOF code as-is

Recon (`hydrinia/dpof_machine.py` + `hydrinia.py` DPOF handlers, 2026-09-24):

| Hydrinia defect (evidence) | CLIO DPOF answer |
|---|---|
| Every NOFA pillar fails open to `0.0`; Sep 7 2026 ran a full day on flat `FPS=0.00` (`dpof_machine.py:2325-2331`) | `PillarReading.value=None` when unavailable; signal requires `min_pillars` real readings incl. NOFA; stale-data alarm after 30 min |
| Two sizing formulas; the logged superlinear one is never used (`hydrinia.py:85546`) | One sizing function (`sizing.py`) |
| `options_target_delta=0.65` configured, never read; selection by $-distance to ATM | Delta-banded selection from broker greeks; no greeks ⇒ no trade |
| No price confirmation ("VWAP never actually checked", `dpof_machine.py:2113`) | Session VWAP + 15-min momentum must agree with direction |
| GEX pillar adds +0.5 in negative gamma regardless of direction (biases bullish), −0.3 in positive gamma (biases bearish) | GEX is direction-agnostic: amplifies/dampens the sign of the other three pillars |
| Config drift across two files (`DPOF_MAX_POSITIONS=2` vs enforced `3`) | One frozen `DPOFConfig` (`agent/dpof/config.py`) |
| 1-DTE entries churned: 5 stop-outs, −$349 in 10 min (2026-06-10) | `min_dte=5`, `max_dte=14`, theta-per-day cap |
| −$2,000 daily breaker trip; exits 403'd (2026-09-06) | Breakers on account equity (realized+unrealized), weekly + drawdown hard-halt, exit escalation ladder, reconciliation |
| Zero-vote autonomous execution | Deterministic gates + optional Jev veto; exits never gated by any model |

Kept from the method: the NOFA 4-pillar composite (NOFA .40 / premium .25 /
dark pool .20 / GEX .15), dark-pool veto, normal/strong thresholds (.15/.30),
VIX regime thresholds, QQQ divergence selection (1.5×), take-profit tiering,
per-symbol entry caps and post-stop re-entry cooldown.

## 2. Account & regulatory facts (verified 2026-09-24)

- **PDT is gone.** FINRA Reg. Notice 26-10: SEC Rel. 34-105226 (2026-04-14)
  eliminated the $25K PDT minimum and day-trade-count designation, effective
  **2026-06-04**. Alpaca removed `pattern_day_trader`/`daytrade_count`/
  `daytrading_buying_power` from the API by 2026-07-06. Do **not** read those
  fields. No day-trade counting anywhere in DPOF.
- Alpaca options: Level 2 = long calls/puts; Level 3 = spreads (`order_class=mleg`,
  always limit). TIF `day`. Whole contracts. No extended hours. Market orders
  allowed single-leg. New orders on an expiring contract stop 15:30 ET.
- Data: free "indicative" options feed vs OPRA (Algo Trader Plus, $99/mo) —
  **OPRA strongly recommended before live** (mid-price limits depend on it).
- Regulatory fees ≈ $0.09/contract round trip; no commission.
- CLIO's LLM trade tools stay **paper-only by invariant** (`agent/security.py`
  `_validate_trade`). DPOF is a *separate* lane with its own keys and gates; it
  never calls the LLM trade tools and never relaxes that invariant.

## 3. Package layout (`agent/dpof/`)

| Module | Kind | Responsibility |
|---|---|---|
| `config.py` | pure | `DPOFConfig` (frozen), `load_dpof_config()`, `DpofMode`, `JevMode`, `validate()` |
| `models.py` | pure | shared dataclasses (`PillarReading`, `MarketSnapshot`, `Signal`, `OptionQuote`, `ContractChoice`, `OrderIntent`, `OrderResult`, `JevVerdict`, `PositionRecord`, `RiskState`, `ExitDecision`, `GateResult`) |
| `signals.py` | pure | regime, composite, DP veto, price confirmation, instrument selection, thesis-invalidation votes |
| `contracts.py` | pure | expiration window, delta-banded single-leg and debit-vertical selection |
| `sizing.py` | pure | premium-at-risk sizing |
| `exits.py` | pure | exit ladder (EOD, stop, breakeven, trail, tiers, max-hold, thesis) |
| `risk.py` | pure | day/week rollover, entry gate, breakers, streaks, cooldowns, operator resume |
| `jev.py` | I/O + pure | TypeSafe client (raw httpx, verified contract), state bucketing, questions, strict answer validation, verdict |
| `data.py` | I/O | `DPOFDataFeed.snapshot()`: UW pillars, bars→VWAP, VIX, econ calendar/breaking news/vol regime readings, headlines, health. `get_intel_ctx()`: advisory events/vol_surface/breaking/calendar_status (§4a) |
| `broker.py` | I/O | dedicated-account Alpaca adapter; **the single order chokepoint**; live/paper gate verification |
| `orders.py` | I/O | entry/exit order lifecycle (reprice, cancel, escalate), fills → ledger, reconciliation |
| `ledger.py` | I/O | atomic JSON persistence (sidecar lock), decisions/jev JSONL |
| `engine.py` | I/O | cycle orchestration, scheduler entrypoint, status, memory hooks |
| `alerts.py` | I/O | plain-text Telegram alerts via `scheduled_tasks._safe_send` |
| `telegram_commands.py` | I/O | `/dpof [status\|positions\|halt\|resume\|config]` |

Pure modules import only stdlib + `agent.dpof.config/models`. I/O modules
lazy-import CLIO integrations inside functions (startup isolation).

## 4. Signal (signals.py)

Per cycle, for each configured underlying `u`:

- Directional pillars (each in [-1,+1], `None` if unavailable):
  `nofa` (net call−put premium / total, flow alerts), `premium` (net premium
  flow), `darkpool` (DP bias).
- `pre = Σ w_i·p_i / Σ w_i` over available directional pillars.
- GEX factor: `negative_gamma|accelerating → +0.5`, `positive_gamma|stabilizing
  → −0.3`, else `0`; halved if pin risk. GEX contribution
  `= gex_factor · sign(pre)` (amplifies/dampens, never picks direction).
- `composite = (Σ w_i·p_i + w_gex·gex_contrib) / (Σ w_i + w_gex)` over available.
- Unavailable if fewer than `min_pillars` available (GEX counts) or NOFA missing.

Decision on the **primary** underlying's composite `c`:

1. Data unhealthy (flow pillars/price data only — see §4a) → veto.
2. Regime from VIX: `≥vix_extreme_label_at → HALT ("EXTREME" label)`,
   `≥choppy → CHOPPY`, `<trending → TRENDING`, else `NORMAL`; VIX
   missing/stale → `UNKNOWN` (treated as CHOPPY). HALT/"EXTREME" is **not**
   an automatic veto any more — see §4a.
3. `|c| ≥ strong → strong`, `≥ normal → normal`, else none. CHOPPY/UNKNOWN/
   HALT("EXTREME") all require strong.
4. Dark-pool veto: `darkpool·sign(c) ≤ −dp_veto_threshold`.
5. Instrument: secondary (QQQ) if `|nofa_qqq| ≥ ratio·|nofa_spy|` and same sign
   as `c`; else primary.
6. Price confirmation on the chosen instrument: bullish needs `last > vwap` and
   `ret_15m > 0` (bearish mirrored). Missing ⇒ not confirmed.
7. `conviction = min(1, |c| / threshold_strong)`.

Thesis-invalidation votes (exits), for a held direction `d`: (a)
`composite·sign(d) ≤ −exit_threshold`, (b) price on the wrong side of VWAP,
(c) `ret_15m·sign(d) < 0`. Missing data casts no vote.

## 4a. Former hard blocks → decision context, + advisory intel (2026-09-25,
INTEL_V3_SPEC.md builder G)

**Operator directive (2026-09-25): the 4 former hard entry blocks — econ
blackout (`±event_context_window_minutes` around an importance-3 release),
breaking news, VIX HALT/"EXTREME", and a blocked vol regime — no longer
block a nomination or entry anywhere.** They are DECISION CONTEXT for the
decider (and, unchanged, for Jev's own `snap.vix`/`snap.headlines` reads)
now:

- `signals.evaluate_signal()` no longer vetoes on `snap.risk_flags`
  (econ_blackout/breaking_news/vol_regime) or `Regime.HALT` — these are
  surfaced as diagnosable `Signal.reasons` (never `Signal.vetoes`) instead.
  HALT/"EXTREME" gets the SAME strong-composite-only treatment CHOPPY/
  UNKNOWN already had (§4 step 3), in place of its old outright veto.
- `data.py`'s `data_healthy` is strictly about the flow pillars/price data
  DPOF actually trades on now (`primary_composite_ok`/`primary_price_ok`) —
  econ-calendar/learning-state gate-SOURCE availability no longer forces it
  False (still recorded as a `health_issues` diagnostic note, just no longer
  a veto input).
- Retired config knobs (existed only to block): `block_on_breaking_news`,
  `block_on_vol_regimes`. Renamed to make their now-advisory role explicit:
  `vix_halt_at` → `vix_extreme_label_at` (regime label threshold only),
  `econ_blackout_minutes` → `event_context_window_minutes` (context window
  only). Both keep their original default values.

**`DPOFDataFeed.get_intel_ctx()`** (data.py) is a new, separate, best-effort
ADVISORY method — awaited once per cycle by engine.py, never inside
`snapshot()`'s own budget — that assembles:

- `events`: `{"upcoming": [...], "recent_surprises": [...], "source": ...}`
  — upcoming releases (minutes_until/importance/consensus/previous) from
  Builder E's `agent.integrations.econ_surprise.get_upcoming_releases(6.0)`,
  else a fallback derived from the same econ-calendar cache `_check_econ_gate`
  already reads; `recent_surprises` from `get_recent_surprises(240)` (empty
  when E is unavailable — no DPOF-native fallback for that half).
- `vol_surface`: `{SYM: vol_surface.get_vol_surface(SYM) | None}` per
  underlying (Builder V) — figures are in VOL POINTS (18.0 = 18% vol), never
  fractions; rendered with explicit units in prompts.py (never raw JSON).
- `breaking`: Builder N's `newsflash_lite.get_breaking_state(900)`, else the
  existing Lighthouse/learning_state breaking-news read this module already
  performs, normalized to N's contract shape (severity synthesized from the
  size of the triggering move).
- `calendar_status`: `{"status": "ok"|"stale"|"missing", "age_hours", "sources"}`
  — the econ calendar's own freshness, now purely advisory.

`engine.py`'s `_regime_ctx()` adds `regime: {"vix", "vix_age_seconds",
"label"}` (HALT's enum value stays `"halt"` — only its presented `label` is
`"EXTREME"`). `_build_intel_ctx()` merges these into `self._last_intel_ctx`
once per cycle (step 10b of `_run_cycle_locked`); `_build_ctx()` merges it
into every decider ctx (entries and reviews) unchanged.

**New review triggers** (`_review_triggers`, never trade by themselves — they
only pull a position's next decider review forward):
(a) a NEW recent surprise (`|surprise_z| ≥ 1.5`, or — when `z` is
unavailable — importance-3 with a non-inline `direction`) released after the
position's `last_review_at` (or `entry_time` if never reviewed);
(b) a NEW breaking event (`id` not seen before FOR THIS position) with
`severity ≥ 2` affecting the position's underlying or macro-wide (no
`tickers` listed). Seen-event bookkeeping is per-position, in-memory only
(`self._seen_breaking_event_ids`), cleared on that position's full close.

**Entry re-deliberation nudge** (`_cooldown_allows`): a new `|surprise_z| ≥
1.5` released after the entry-deliberation cooldown record was set bypasses
that cooldown for the next cycle — the decider still decides (enter/pass),
this only gets it deliberating sooner.

All of the above is grep-provably advisory: none of `signals.py`, `data.py`,
`engine.py`'s new code touches `agent.dpof.broker`/`orders` — see
`docs/design/clio_dpof.md` §5b/§5a for the unchanged entry/exit paths those
still exclusively own.

## 4b. Flow fallback: self-computed pillars (2026-09-25, operator directive)

**Operator directive (2026-09-25):** UW (Unusual Whales) API access is down
until ~Tuesday (subscription/key reissue). Without UW, DPOF's flow pillars
(NOFA/premium/darkpool/GEX, plus NOPE/max-pain context) go missing → data
unhealthy → no nominations. While UW is unavailable, `data.py` falls back to
`agent.flowcollector`'s self-computed Alpaca OPRA/SIP flow
(`agent.flowcollector.state.get_selfflow(symbol, max_age_seconds)`) as a
fallback pillar source. **Paper only** (DPOF is paper by default; this is
part of the soak). **Active by default** (`CLIO_DPOF_SELFFLOW_FALLBACK=true`,
`CLIO_DPOF_SELFFLOW_MAX_AGE=120`). When UW data returns, UW is primary again
automatically — nothing about decision authority changes: the signal only
nominates; Opus 5.5 (GPT-6 Sol fallback) + Jev still decide.

**Precedence, per pillar, per underlying, per cycle** (`_build_underlying_
snapshot`): UW wins whenever it produced a fresh (≤ `3×uw_refresh_seconds`)
reading for that specific pillar — the two sources are **never blended** for
the same pillar. Only a pillar UW left missing/stale this cycle is filled
from selfflow, and only when `get_selfflow()` itself is fresh (≤
`selfflow_max_age_seconds`, default 120s — selfflow's own state file is
written every 15s by `clio-flowcollector.service` during market hours). A
pillar neither source can supply stays `PillarReading.missing()` exactly as
before (missing is never zero).

**Per-pillar mapping** (`_selfflow_pillar_reading`, `agent/dpof/data.py`):

| DPOF pillar | selfflow field | Mapping |
|---|---|---|
| `nofa` | `nofa_ratio` | Signed classified call-minus-put premium / gross classified premium, in `[-1,+1]` — the same form as UW's NOFA ((call − put) / total). selfflow's plain `nofa` field is `tanh(dollars / an uncalibrated $ scale)` and is NOT used: nomination thresholds were tuned on UW ratios. Empty window → 0.0 (as UW's empty alerts); missing ratio key → unavailable. |
| `premium` | `premium_ratio` | (aggressor buys − sells) / (buys + sells) premium, in `[-1,+1]` — same form as UW's premium score (net / (\|call\| + \|put\|)). |
| `darkpool` | `darkpool_ratio` | Signed classified dark notional / gross classified dark notional, in `[-1,+1]`, positive = net buying. Methodology differs from UW (UW: print price vs the window median; selfflow: each D-tape print's side vs its NBBO midpoint at ingest) but both read "net buying (+) / selling (−)". |
| `gex` | `gex_regime` (+`gex`, `gex_detail.pin_strike` as context) | **Not used directly** — `PillarReading("gex").value` is never the raw signal, it's already the `gex_factor` (see `signals.compute_composite`'s own docstring). Selfflow's regime vocabulary (`"positive"`/`"negative"`/`"neutral"`, a relative-band dealer-gamma sign) is DIFFERENT from UW's (`"ACCELERATING"`/`"STABILIZING"`) that `signals.gex_factor()` understands — translated explicitly via `_SELFFLOW_GEX_REGIME_TO_UW`: selfflow `"negative"` (dealers net short gamma, amplifies moves) → UW `"ACCELERATING"` (`gex_factor` `+0.5`); selfflow `"positive"` (dealers net long gamma, dampens moves) → UW `"STABILIZING"` (`gex_factor` `-0.3`); `"neutral"` → `"NEUTRAL"` (`gex_factor` `0.0`, same else-branch a raw pass-through would already hit). Passing the raw strings through unmapped would have silently zeroed GEX's directional contribution for both `"positive"` and `"negative"`. |
| `nope` (context) | `nope` | Used directly (same Lily Francus definition, computed independently) only when UW's own NOPE read is still `None` after its own attempt. |
| `max_pain` (context) | `max_pain` | Used directly (same textbook max-pain strike definition) only when UW's own max-pain read is still `None` after its own attempt. |

**Minimum sample:** fewer than `selfflow_min_trades` (default 20, `CLIO_DPOF_SELFFLOW_MIN_TRADES`) classified trades in the 5-min window — option trades for nofa/premium, D-tape prints for darkpool — gives a "thin" 0.0 reading (mirrors the UW path's `_NOFA_MIN_ALERTS`); a ratio of one or two trades is always ±1.0 and would otherwise read as a strong signal after the open or a collector restart. Unknown sample size → unavailable. **UW 401 backoff:** while `uw_flow_intel.uw_auth_backoff_active()` is true, `_uw_fetch` returns "not attempted" without charging DPOF's daily UW cap.

**Selfflow GEX / NOPE greeks (2026-09-25 fix).** On the first fallback day
the selfflow GEX read exactly 0.0 / "neutral" all session (`deltas_loaded:
0`) and still counted as an AVAILABLE pillar, diluting the composite by the
GEX weight every cycle. Causes, verified on the VPS: (1) the collector
counted `max_dte` in calendar days, so on a Friday its universe held 0DTE
contracts only; (2) Alpaca returns no greeks for same-day (0DTE) expiries —
the heaviest-gamma slice — while for the next 1–3 sessions they cover ~77–92%
of in-band open interest (deploy dry run, 2026-09-25; zero-bid far-OTM
contracts get none); (3) `compute_gex` turned all-zero gammas into
"neutral". Now:
- `max_dte` counts **trading sessions** like DPOF's 1DTE (Friday → Fri/Mon/Tue; holidays skipped).
- Per-contract greeks precedence: Alpaca (sane delta+gamma) → **self-computed Black-Scholes-Merton** from the freshest quote (live NBBO, else the snapshot quote; IV solved from the mid; `risk_free_rate` 0.04 / `dividend_yield` 0.012 config) → **smile fill** (IV interpolated linearly in strike between solved/Alpaca anchors of the same expiry, flat to the band edge; ≥2 anchors) → missing. Self-computed greeks refresh every `greeks_recompute_seconds` (60 s) from live quotes, off the event loop.
- **No fake zeros**: GEX is unavailable (regime `None`) when OI-weighted gamma coverage < `gex_min_coverage` (0.5) or all OI is 0; NOPE is unavailable when traded-volume delta coverage < `nope_min_coverage` (0.5) or the window has no option prints. DPOF maps a `None` regime / NOPE to unavailable (never 0.0).
- **Effect once GEX is live** (it is direction-agnostic: `contrib = gex_factor × sign(composite)`): a positive-gamma ("STABILIZING", −0.3) regime *damps* nominations, a negative-gamma ("ACCELERATING", +0.5) one amplifies them. With all three directional pillars at the same value p, the normal threshold (0.15) needs p ≈ 0.23 in positive gamma, 0.088 in negative gamma, 0.176 under the old fake "neutral", 0.15 with GEX unavailable (strong 0.30: 0.41 / 0.27 / 0.35 / 0.30). The dry run read "positive" for SPY and QQQ going into 2026-09-28. By design; any threshold change is the operator's call, with `signals.jsonl` as the evidence.
- `quality_notes` add `greeks_source` {alpaca, computed, interpolated, missing}, `gammas_loaded`, `gamma_oi_coverage`, `delta_volume_coverage`; the collector logs one INFO `flowcollector: greeks refresh -- SPY: alpaca=… computed=… interpolated=… missing=… gamma_oi_coverage=…` line at most every 5 min, and a once-a-day WARNING when a whole Alpaca batch has no greeks.

**Health**: when primary pillars come from selfflow, the snapshot can still
be `data_healthy` — that's the point. If selfflow is missing/stale too,
behavior is unchanged from before this fallback existed (unhealthy, existing
stale-data alarm).

**Visibility**: `PillarReading.source` reads `"selfflow:<symbol>"` (vs
`"uw:..."`) per pillar; `UnderlyingSnapshot.flow_selfflow` is `True` when
≥1 of {pillars, nope, max_pain} for that underlying came from selfflow this
cycle. `engine.py`'s per-underlying decider ctx adds a plain-words
`flow_note` ("flow pillars self-computed from Alpaca OPRA/SIP trades
(quote-rule classification; not validated against Unusual Whales)") only
when `flow_selfflow` is true — the static CHARTER text is unchanged (prompt
caching preserved). `/dpof status`'s data-health line names the affected
underlyings. A UW↔selfflow transition is logged once (INFO) per underlying
per process, never every cycle.

**Isolation**: `data.py` only ever *reads* `agent.flowcollector.state.
get_selfflow()` — it never imports `agent.dpof.broker`/`orders`, and the
selfflow read for a given underlying is skipped entirely (grep-provable: no
call) whenever UW is already healthy for every pillar/context field that
underlying needs this cycle.

## 5. Contract candidates, sizing, exits (v2 — judgment layer)

**Contract candidates (contracts.py `choose_candidates`)**: a MENU, not an
auto-pick. 1DTE standard (§0/Hydrinia design intent) — `target_expiries()`
walks `expiry_sessions_ahead..max_expiry_sessions_ahead` trading sessions
ahead (skipping weekends + `MARKET_HOLIDAYS`); the first expiry with liquid
candidates wins. Liquidity filter only (no theta gate — theta is decision
CONTEXT for 1DTE, never a mechanical filter): valid two-sided quote,
`spread/mid ≤ max_spread_pct`, `OI ≥ min_open_interest`. Up to
`entry_candidates` `EntryCandidate`s: the closest-to-`target_delta` single
plus its nearest lower/higher-delta neighbors (or an ATM-ish strike-distance
fallback when greeks are absent from the whole chain), plus one debit
vertical when `allow_verticals` AND the broker-reported account options
level ≥ 3 (checked at runtime by engine.py, never a config-time error) and
one can be built. The engine assigns stable keys ("C1"..) and computes each
candidate's `max_affordable_qty` (see Sizing below) as informational
context, never a filter.

**Sizing (operator directive, no-hard-caps amendment, 2026-09-24): NO
risk-pct band, no per-trade/open-premium/max-contracts cap.** The decision
model picks `qty` directly (its own sizing judgment, informed by the
candidate's `max_affordable_qty` context). The ONE code-side bound left is
SOLVENCY (`sizing.py`):

```
capital_available = dpof_equity − open_premium_committed   # incl. PENDING entries
max_affordable_qty = floor(capital_available / (unit_debit·100))
```

`clamp_to_solvency(qty, capital_available, unit_debit)` re-checks this FRESH
at actual-entry time (equity/open-premium may have moved during the model's
10-60s latency) and clamps the model's `qty` down if needed (logged, not a
failure) — `qty < 1 ⇒ no trade` (never floors up). `dpof_equity` is DPOF's
own equity (`engine._dpof_equity()`, §6), not necessarily the raw Alpaca
account equity.

**Exits — the ONE kept rail**: forced flatten of `qty_open` at `flatten_et`
(urgent), half-days clamp via `engine._session_cfg`. **No other mechanical
exit exists** — no stop-loss, breakeven, trailing stop, take-profit tiers,
max-hold timer, or thesis-vote auto-exit. Every open position is instead
reviewed by the decision model on a cadence (§ Position reviews below);
`ReviewDecision.action` (`hold`/`trim`/`exit`) is the ONLY thing that can
close or trim a position before the flatten. **No model is ever consulted on
the flatten itself** — it is pure wall-clock arithmetic.

## 5a. Position reviews (v2 — judgment layer, engine.py)

Each OPEN position (skip pending/closing — no new decision is applied while
an exit batch is already working) is reviewed by the decider on
`review_interval_minutes` cadence by default; the decider's own
`ReviewDecision.next_review_minutes` sets the next one, clamped to
`[review_min_minutes, review_max_minutes]`. Triggers pull a review forward
(never trade by themselves): due (`now ≥ next_review_at`), a Jev SENTINEL
escalation (§8), `|mark move| / last_review_mark ≥ review_trigger_move_pct`,
`signals.position_alignment` flipping flow/VWAP "against" the position, or a
`"breaking_news"` risk flag on the fresh snapshot.

Decider calls run as background `asyncio.Task`s (10-60s latency) — the
~30s cycle never awaits one; results are harvested and applied on a LATER
cycle, only if fresh (`≤ decision_max_age_seconds`) and the position is
STILL `open` with the SAME `qty_open` as when the review launched (order-
safety rule: a position that moved to `closing` — e.g. the EOD flatten fired
first that cycle — discards the harvested decision rather than submitting a
second, conflicting close). `trim` sells `ceil(qty_open · trim_fraction)`
(≥ 1; ≥ `qty_open` ⇒ treated as `exit`); `urgency == "urgent"` lets that
close escalate to marketable/market sooner (orders.py), same as the old
ladder's `urgent` flag. `DecisionUnavailable` on a review ⇒ HOLD,
`review_failures += 1`, retry in ~60s (not the full interval), operator
alerted (re-alert every `model_down_alert_minutes`); the 15:50 flatten still
fires regardless.

## 5b. Entry deliberation (v2 — judgment layer, engine.py)

Each cycle, for the signal's nominated underlying (`evaluate_signal`, §4 —
unchanged), if entry-permitted (§6) and no in-flight deliberation for that
underlying (one in-flight per underlying — concurrency control, not a size
cap) and the cooldown allows (`entry_deliberation_cooldown_seconds` per
underlying+direction, bypassed by the composite strengthening
`≥ entry_redeliberate_delta` or the decider's own `recheck_minutes`
elapsing) → the engine builds the candidate menu + ctx and launches
`decider.decide_entry(ctx, candidates)` in the background. On harvest: stale/
inapplicable (underlying now has a position, gate fails) ⇒ discard+log;
`action == "enter"` ⇒ the EXISTING Jev entry veto (`jev_entry_check`,
§8 — now includes a sizing question) runs against the chosen candidate,
re-quoted fresh (discard on `> entry_max_slippage_pct` vs the decided mid),
sized via `clamp_to_solvency`, and submitted through the unchanged
OrderManager entry path; `action == "pass"` ⇒ remembers the cooldown/
recheck time. Both providers down ⇒ no entry this cycle, operator alerted
(same alert as the review path).

## 5c. Learning loop (F1 post-trade lessons + F2 borderline debate)

Task ORCH-7Q4K-DPOF (2026-09-26): the missing feedback edge on top of §5a/§5b
— **F1** writes a short, code-computed-evidence lesson from the decision
model after every closed trade (`agent/dpof/reflection.py`); recent lessons
plus a code-computed per-underlying tally then enter future entry/review
prompts as context (never a rule, gate, or vote). **F2** adds a short bull-
case/bear-case second opinion from a faster model on genuinely borderline
calls only — a "normal"-strength entry nomination within `debate_margin` of
`threshold_normal`, or a review with an `alignment_against` trigger at least
`debate_review_min_minutes_to_flatten` from the flatten and no
`jev_escalation` (`agent/dpof/debate.py`). Both features run through their
own `agent/dpof/advisory_llm.py` client — separate circuit breakers/budgets
from `decider.py`, which stays untouched and read-only to them. The debate
runs INSIDE the existing background decision Task (engine.py wraps
`decide_entry`/`review_position`, never inline in the cycle); reflection
scheduling is fire-and-forget, never awaited by the cycle. Full design frame,
rulings and the exact prompt-injection/decision_log shapes:
`docs/design/clio_dpof_learning_loop.md`. Flags: `CLIO_DPOF_REFLECTION`,
`CLIO_DPOF_LESSONS_IN_PROMPT`, `CLIO_DPOF_DEBATE`, `CLIO_DPOF_DEBATE_REVIEWS`
(all default on — see `agent/dpof/config.py`).

## 6. Risk envelope (risk.py) — v2, no-hard-caps amendment

**Operator directive (no-hard-caps amendment, 2026-09-24): NO mechanical
breakers and NO entry-count caps.** `check_entry_allowed` is now just:
not hard-halted (operator `/dpof halt`, manual only — there is no automatic
breaker left to trip it), `data/dpof/HALT` file absent, and inside
`[entry_start_et, entry_end_et]`. There is no
`max_concurrent_positions`/`max_entries_per_day`/`max_entries_per_symbol_day`
count cap, no daily/weekly/drawdown breaker, no loss-streak cooldown, no
per-symbol re-entry cooldown. The ONE size bound left is SOLVENCY (§5),
enforced by `sizing.py` at actual entry time, not duplicated in this gate.
`risk.py` still does day/week P&L accounting (`realized_pnl_today/_week/
_total`, `entries_today`, `entries_by_symbol`, start-of-day/week equity) for
`/dpof status` and reporting, and still owns operator halt/resume
(`operator_halt`/`operator_resume`) — the only way `hard_halted` is ever set
or cleared now.

**Virtual capital for a SHARED Alpaca account** (`cfg.capital_base`,
`CLIO_DPOF_CAPITAL`, default $20,000): CLIO's shared paper account (the
`clio_paper` key fallback in §7) may hold other, non-DPOF positions and a
much larger balance than DPOF is sized for. Sizing by the full account
equity would oversize; the breakers above compare equity *deltas*, so
non-DPOF P&L would trip (or mask) DPOF's own breakers. `engine._dpof_equity()`
is the ONE place every equity use site (`roll_state`, `evaluate_breakers`,
`size_position`, the `/dpof resume` re-baseline, status text) routes through:
with `capital_base > 0` it returns `min(account_equity, capital_base +
risk.realized_pnl_total + unrealized)` — a virtual ledger seeded at
`capital_base` and tracked forward purely by DPOF's own trades, capped at
the real account equity so DPOF can never claim more capital exists than the
account actually has. `capital_base = 0` disables this and uses the real
account equity unchanged (the right setting for a dedicated account). Unknown
real equity is never turned into a number — `None` propagates, and the
breakers above already treat that as "unarmed," not zero.
`RiskState.realized_pnl_total` is DPOF's cumulative realized P&L since
inception (or the last `/dpof resume` re-baseline, which resets it to 0) —
unlike `realized_pnl_today`/`realized_pnl_week`, day/week rollover never
touches it.

## 7. Modes and live gates (broker.py = single chokepoint)

- `off`: engine returns immediately (only the ledger + alerts are constructed
  — never the broker — so a stranded open position from a prior paper/live
  run is still alerted on: "DPOF is OFF with N open position(s)... NOT being
  managed"). Config-invalid similarly surfaces "N open position(s) NOT being
  managed" when positions exist.
- `log_only`: full pipeline; `broker.submit()` returns `would_submit=True`
  without any network call.
- `paper` (**the default**): requires `CLIO_DPOF_ALPACA_PAPER=true` and the
  account to report paper. Alpaca keys: `CLIO_DPOF_ALPACA_KEY_ID`/
  `_SECRET_KEY` (dedicated account) preferred; if both are unset, falls back
  to CLIO's existing shared paper keys (`ALPACA_PAPER_KEY_ID`/
  `_SECRET_KEY`) as a complete pair from one source only (never a dedicated
  id mixed with the shared secret, or vice versa) — see
  `DPOFConfig.alpaca_key_source` ("dedicated" | "clio_paper" | "none",
  shown in `/dpof config`) and §6 for how a shared account's balance is kept
  from corrupting DPOF's own solvency sizing. Options data feed
  (`CLIO_DPOF_OPTIONS_FEED`) defaults to **`opra`** (operator confirmed
  Algo Trader Plus / real-time OPRA + options level 3 on the account,
  2026-09-24); `auto` tries OPRA and falls back to indicative for the rest
  of the session on a 401/403 (one INFO log + shown in `/dpof status`);
  `indicative` is explicit, no OPRA attempt. `CLIO_DPOF_STOCK_FEED` defaults
  to `sip` (Algo Trader Plus includes full SIP).
- `live`: requires ALL of: `CLIO_DPOF_ALPACA_PAPER=false`;
  `CLIO_DPOF_LIVE_ACK=I_ACCEPT_REAL_MONEY_RISK`; `CLIO_DPOF_LIVE_ACCOUNT` equal
  to the broker's `account_number`; CLIO kill switch `live_trading` enabled
  (`LIVE_TRADING_ENABLED=true` and no `LIVE_TRADING_KILLED` file); options
  level ≥ `min_options_level` (≥3 if verticals); equity ≥ `min_account_equity`;
  account not trading-blocked. Any failure ⇒ effective mode `log_only` + alert.
  **Never** falls back to CLIO's shared paper keys — a live account always
  requires its own dedicated `CLIO_DPOF_ALPACA_KEY_ID`/`_SECRET_KEY`.
- Every submit: prefix-checked client order id, `audit_tool_execution(...)`
  (HMAC-chained audit log), mode re-check. Orders without the DPOF prefix are
  never cancelled/replaced.

## 8. Jev fast decision layer (jev.py)

Verified contract: `docs/api/typesafe_jev.md`. Modes: `off | shadow | veto`,
defaulting to **`veto`** (operator directive, 2026-09-24: "everything is
built active, not shadow"). Two roles now:

1. **Entry veto** (`jev_entry_check`) — unchanged semantics, called against
   the decider's chosen candidate AFTER it picks `qty` (§5b). Gains a sizing
   question (no-hard-caps amendment): bucketed `premium_pct_of_capital`,
   `open_exposure_pct`, `positions_open`, `entries_today`,
   `minutes_to_flatten`, `realized_today_pct`, `conviction` — a validated
   `size_reckless ≥ jev_size_veto_threshold` (default 0.70) adds
   `"size_reckless"` to the veto reasons. Jev still never sizes or picks a
   contract itself — it can only veto a size the decision model already
   chose.
2. **Position-review SENTINEL** (`jev_position_check`, §5a) — screens each
   open position every cycle (when `jev_effective_mode != off`) and may only
   set `escalate=True`, pulling that position's NEXT DECIDER REVIEW forward
   to now. It can never exit, trim, enter, or size a position by itself —
   the hard boundary below covers both roles.

- **`jev_effective_mode`** (`DPOFConfig` property): `veto`/`shadow` degrades
  to `off` when Jev is completely UNCONFIGURED — no `TYPESAFE_API_KEY`, no
  `CLIO_DPOF_JEV_KEY_FILE`, no `CLIO_DPOF_JEV_FALLBACK_URL`. This is what
  makes the `veto` default safe to ship inert: with no key set, Jev is never
  called and never blocks an entry — identical to `off`. It becomes active
  automatically the instant one of those three is configured; no other flag
  flip needed. A key/key_file that IS configured but fails to resolve at
  RUNTIME (e.g. an unreadable key file) is a real failure, not an
  unconfigured state — it stays whatever `jev_mode` is (fail-closed in
  `veto`, per `jev_fail_closed`). Every place the engine/jev.py decides
  whether to call Jev, or whether a Jev failure blocks an entry, reads
  `jev_effective_mode`, never the raw `jev_mode`. `/dpof status` and
  `/dpof config` show e.g. `jev: veto (inactive — no TYPESAFE_API_KEY)`, and
  the engine logs once at INFO on the first cycle it detects this.
- Called only for an **actionable entry** after all deterministic gates pass.
  One request, questions evaluated in parallel:
  - `headline_conflict` (noul) — headlines point against the trade in the next hour.
  - `event_risk` (noul) — market-moving event now / within the hour.
  - `setup` (choice: `aligned | mixed | late_or_exhausted`) over **bucketed**
    signal descriptions (vendor: "Jev is not a calculator" — all math stays in
    code; state carries named buckets, not raw floats).
- Veto if `headline_conflict ≥ 0.70`, `event_risk ≥ 0.70`, or
  `setup = late_or_exhausted` with confidence `≥ 0.60`.
- `shadow`: logged to `data/dpof/jev.jsonl`, never blocks. `veto`: may block
  the entry; on error/timeout blocks only if `jev_fail_closed`.
- Hard boundary: an entry verdict can only turn an entry from allowed to
  blocked; a position-sentinel `JevEscalation` can only pull the next
  decider review forward. Neither ever initiates, sizes, selects contracts,
  or exits/trims a position directly — every trade decision (enter/pass,
  hold/trim/exit) comes from the decision model, never from Jev.
- Strict validation (fail-closed): noul in [0,1]; choice ∈ criteria keys,
  probabilities sum≈1, argmax = choice; confidence in [0,1]. Unknown extra
  fields ignored (a fallback may add `confidence` to noul answers).
- Deadline 2.5 s total; one retry on 429/529 honoring `retry-after` within the
  deadline; circuit opens after 3 consecutive failures for 10 min.
- Optional wire-compatible fallback (`CLIO_DPOF_JEV_FALLBACK_URL`) — e.g.
  a self-hosted server (serving the same `POST /v1/systemone`),
  tried once inside the same deadline.
- Headlines pass through `sanitize_ingested_content(..., source="dpof_jev_headlines")`
  and are capped; they are untrusted data (vendor-documented adversarial risk).

## 9. Persistence (`data/dpof/`, ledger.py)

`positions.json`, `risk_state.json` (atomic `.tmp` + `os.replace`, guarded by a
**sidecar** lock file `.dpof.lock` — never flock the replaced file, CLAUDE.md
anti-pattern #22), `decisions.jsonl` and `jev.jsonl` (append, capped). All file
I/O via `run_in_executor` (anti-patterns #19/#20).

## 9a. Observability (`signal_log.py`, Task ORCH-7Q4K-DPOF, 2026-09-25)

**Problem this closes**: every pre-decider rejection reason in the entry
pipeline (§5b) was `logger.debug(...)` only — invisible at production INFO
(CLAUDE.md anti-pattern #27) — and `decisions.jsonl` only records cycles
where a decider call actually happened. A "quiet tape" day (no nomination at
all, every cycle) left NO trace anywhere; an operator watching the live
session could only guess ("quiet tape + a dead GEX pillar").

**OBSERVABILITY ONLY** (operator hard constraint): `signal_log.py` is a
standalone, purely-additive collaborator. Nothing in it is ever read by a
gate, threshold, size, order, or decider/Jev call — engine.py's
`_log_cycle_signals` runs LAST in each cycle, after `_handle_entry_pipeline`
(§5b) and after the risk-state persist (so neither an exception nor a
cycle-timeout cancellation landing in it can skip a real step), and only
RECORDS the outcome that pipeline already reached, via a
handful of one-line, non-reordering additions at its real return points
(`_note_entry_outcome`) — never a second/parallel gate evaluation.

- **`data/dpof/signals.jsonl`** — one record per cycle per underlying (`ts`,
  `underlying`, `composite`, `strength`, `direction`, per-pillar
  `{value, available, source}` for all 4 `PILLAR_NAMES`, `flow_source`
  ("uw"/"selfflow"/"mixed"/"unavailable"), `regime`, `price_confirm`,
  `vetoes`, `reasons`, `is_signal_symbol`, `nominated`, **`outcome`**, and
  `decider`/`jev` summaries when a deliberation was harvested this cycle).
  `outcome` is one of: `outside_entry_window`, `halted`, `data_unhealthy`,
  `composite_unavailable`, `below_threshold`, `regime_requires_strong`,
  `dp_veto`, `price_not_confirmed`, `position_open`,
  `deliberation_in_flight`, `cooldown`, `insufficient_capital`,
  `decider_unavailable`, `candidate_schema_unavailable`,
  `expiry_window_error`, `chain_fetch_error`, `no_chain`,
  `candidate_build_error`, `no_candidates`, `deliberating` (this cycle's
  fresh-nomination outcomes — engine.py's `_signal_block_outcome`/
  `_gate_outcome`/`_note_entry_outcome` call sites), or `decider_pass`,
  `decider_enter`, `stale_decision`, `decider_invalid_action`,
  `decider_invalid_candidate`, `signal_unavailable`, `requote_failed`,
  `requote_unusable`, `slippage_exceeded`, `no_size`, `jev_veto`,
  `open_position_failed`, `open_position_rejected`, `position_gone`,
  `entered` (a harvested prior deliberation — `_classify_harvest_outcome`
  reading the SAME `decision_log` dict `_apply_entry_outcome` already
  appends to `decisions.jsonl`), or `not_selected_instrument` (an
  underlying neither nominated nor harvested this cycle —
  `_fallback_entry_outcome`, real ledger/gate facts only, never a
  synthesized signal verdict for a symbol the pipeline didn't touch).
  Capped at `signal_log_max_lines` (`CLIO_DPOF_SIGNAL_LOG_MAX_LINES`,
  default 20,000).
- **`data/dpof/signal_day_stats.json`** — same-day rollup folded from every
  record (`fold_record_into_day_stats`, pure): cycles evaluated,
  nominations, decider enter/pass/unavailable, Jev vetoes, entries, and per
  underlying: outcome counts, max `|composite|` (with time + direction),
  pillar availability counts, flow-source mix. Resets on an ET day
  rollover. Both files share the ledger's own I/O conventions (sidecar
  lock, atomic `.tmp`+`os.replace`, `run_in_executor`) but their OWN
  sidecar (`.dpof_signal.lock`) — never the ledger's.
- **INFO visibility without spam**: `NearMissLogger` (in-memory, per
  process) logs one INFO line per underlying when its outcome CHANGES (at
  most 4 change lines per underlying per window, so an outcome flapping
  every cycle can't log every 30 s), and at most one INFO line per
  `(underlying, outcome)` pair per
  `near_miss_log_minutes` (`CLIO_DPOF_NEAR_MISS_LOG_MINUTES`, default 10)
  while it stays unchanged — e.g. `DPOF SPY: no entry — below_threshold
  (composite +0.12; nofa +0.21 premium +0.08 darkpool -0.05 gex n/a; flow
  selfflow)`.
- **`/dpof why`** (`agent/dpof/telegram_commands.py`, filesystem-only via
  `agent.dpof.readmodel.dpof_why_text()` — never touches the live
  engine/broker, so Mission Control can call the same reader directly):
  today's day stats, the current outcome + blocking reason per configured
  underlying, and the last ~5 outcome CHANGES.
- **EOD digest** (`agent/integrations/scheduled_tasks.py`, `dpof_eod_digest`
  job, `run_daily` at 16:10 ET Mon–Fri with a tz-aware time so it follows
  DST, catch-up run after a restart between 16:10 and 18:10, skips
  `MARKET_HOLIDAYS` — the same set the engine itself uses — gated on
  `eod_digest_enabled`/`CLIO_DPOF_EOD_DIGEST`, default on): plain-text
  Telegram message(s), split under Telegram's 4096-char limit — entries, nominations,
  decider outcomes, max composite per underlying (with time), top blocking
  outcomes with counts, flow-source mix, pillar availability. The automatic
  answer to "why didn't CLIO trade today." If no cycle ran today, it says
  so ("DPOF did not evaluate the market today") instead of showing an
  earlier day's figures; `/dpof why` labels stale figures the same way.

## 10. Operations

- Scheduler: one `jq.run_repeating(_with_timeout(run_dpof_cycle, cycle−5,
  "dpof_cycle"), interval=cycle_seconds)` in the unconditional block of
  `register_scheduled_jobs()`; shutdown hook in `shutdown_background_tasks()`.
  The `dpof_eod_digest` job (§9a) registers alongside it, independent of
  `cycle_seconds`.
- Telegram: `/dpof`, `/dpof positions`, `/dpof halt`, `/dpof resume`,
  `/dpof config`, `/dpof why` (§9a) (authorized users only; plain text).
- Memory: `remember()` on entry, `learn_from_outcome()` on close (fire-and-forget).
- **Operator visibility**: once per process, on the first cycle `mode` is
  `paper`/`live`, a single Telegram alert announces activation — e.g. `DPOF
  active: PAPER | keys: clio_paper | capital $20,000 (account equity $X) |
  underlyings SPY,QQQ | Jev: inactive (no key)` — so an operator who did
  nothing beyond a restart still finds out DPOF is now actively trading.
- **Unmanaged-position alerts** (deduped daily): `mode=off` with open
  paper/live positions in the ledger (checked WITHOUT touching the broker —
  see §7); a config-invalid cycle with open positions appends "N open
  position(s) NOT being managed" to the existing alert.

## 11. Go-live checklist (operator)

Paper trading now starts from the DEFAULT config — there is no separate
log-only staging period to opt into (operator directive, 2026-09-24: "paper
mode IS the soak"). `log_only` stays available for an operator who wants to
review decisions before any paper order is placed.

1. (Optional but recommended) Open a dedicated Alpaca account for DPOF and
   set `CLIO_DPOF_ALPACA_KEY_ID`/`_SECRET_KEY`. Without dedicated keys, DPOF
   automatically falls back to CLIO's existing shared paper account
   (`alpaca_key_source=clio_paper`) — fine for the paper soak, but review §6
   (virtual capital) since a shared account may hold other positions/balance.
   Request options Level 2 (Level 3 for verticals — operator confirmed level
   3 on the account, 2026-09-24, so verticals default on). OPRA data (Algo
   Trader Plus) — `CLIO_DPOF_OPTIONS_FEED` already defaults to `opra`.
2. Set `ANTHROPIC_API_KEY` (decision model, `CLIO_DPOF_DECISION_MODEL`
   default `claude-opus-5-5`) and `OPENAI_API_KEY` (fallback,
   `CLIO_DPOF_FALLBACK_MODEL` default `gpt-6-sol`) — both providers down means
   no new entries and open positions HELD (never auto-closed) until one
   recovers; `/dpof status` shows decider provider health.
3. Leave `CLIO_DPOF_MODE` unset/`paper` (the default) and let it run; watch
   the one-time activation alert and `/dpof status`/`/dpof positions` for
   entries, reviews (thesis, last review action, next review ETA), fills,
   and slippage. Optionally set `CLIO_DPOF_MODE=log_only` first if a
   no-orders dry run is wanted.
4. Optional: set `TYPESAFE_API_KEY` (or `CLIO_DPOF_JEV_KEY_FILE`) — Jev
   activates automatically at `jev_mode`'s default (`veto`) for BOTH the
   entry veto and the position-review sentinel; set `CLIO_DPOF_JEV_MODE=shadow`
   first to review `jev.jsonl` before trusting it to block entries.
5. Live: swap to live keys, `CLIO_DPOF_ALPACA_PAPER=false`,
   `CLIO_DPOF_LIVE_ACCOUNT=<acct#>`, `CLIO_DPOF_LIVE_ACK=I_ACCEPT_REAL_MONEY_RISK`,
   `LIVE_TRADING_ENABLED=true`, `CLIO_DPOF_MODE=live`; restart; confirm
   `/dpof` shows `mode=live (gates OK)`. Live never falls back to shared
   keys (§7) and, if trading a dedicated account, set `CLIO_DPOF_CAPITAL=0`
   to use the real account equity unchanged.

Kill: `/dpof halt`, or `touch data/dpof/HALT`, or `CLIO_DPOF_MODE=off` + restart
(`off` still alerts if positions are left unmanaged — see §10).
