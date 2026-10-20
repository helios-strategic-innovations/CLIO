# CLIO DPOF — Dynamic pillar weights (design)

**Status**: Draft v2 — 2026-10-10, after the research wave (R1 data, R2 critique, R3 integration map). v1 is superseded.
**Decision owner**: operator (Dominic). Direction chosen 2026-10-10: *static weights → dynamic, automatic weights, even if it takes longer*.
**Builders**: Haiku 5.5 agents in isolated worktrees; the orchestrating session verifies every patch (own mutation matrix, live-data replay, byte-compare) before it reaches the live tree.
**Related**: `docs/design/clio_dpof.md` (§4), `docs/design/adr/ADR-006-uw-pillar-scoring-fixes.md`, `docs/plans/S213_dpof_dynamic_weights_and_followups.md`.

## 1. Problem and evidence

Composite = `Σ wᵢ·pᵢ / Σ wᵢ` over *available* directional pillars (NOFA .40, premium .25, darkpool .20, Kronos .15) plus a GEX term (`w_gex` .15, amplifies/dampens by `sign(pre)`). The weights are read in exactly one place, `agent/dpof/signals.py:164-178`.

* It renormalizes over **available** pillars but cannot detect **available-but-dead** ones. Premium scored exactly 0.0 on all 6,476 UW-sourced SPY cycles (fixed 2026-10-09: wrong field names); the legacy darkpool score sits at ±0.002. Dead pillars held 0.45 of the weight.
* **Measured on the 10 logged sessions (R1, spot-checked by the orchestrator)**:
  * No pillar shows a detectable predictive edge at 5/15/30/60 minutes. Block-sampled IC vs the next 30 minutes (pooled SPY+QQQ, n≈220 blocks): NOFA −0.017 [−0.149, +0.116]; corrected premium +0.008…+0.014 [≈−0.13, +0.15]; Kronos +0.018 [−0.115, +0.150]; composite −0.006 [−0.138, +0.126]. 80%-power minimum detectable |IC| ≈ 0.19 at that n. **Not refuted, not confirmed — the data cannot say.**
  * The corrected premium is **contemporaneous momentum**: corr +0.38 with the trailing 15-minute return, partial IC vs the next 30 minutes after removing that return ≈ −0.02. Same-5-minute return corr +0.57 (SPY) / +0.42 (QQQ); next-5-minute ≈ 0.
  * The liveness rule (K=40 refreshes, ε=0.005) flags the logged-dead premium and darkpool in every session that has UW data (117–141 minutes after the open) and flags the healthy corrected premium **0 of 20** times at every tested K/ε. ε=0.001 misses the legacy darkpool (values ±0.002).
  * Conserving total weight vs removing a dead pillar changes nomination episodes by only 1–4; replacing the zero premium with the corrected score changes them most.
* Not a defect (verified): the logged QQQ composite equals SPY's on 1,418 cycles because `select_instrument` attributes the single primary-basis signal to QQQ when QQQ is selected (`signal_basis: SPY`, `is_signal_symbol: True`).

## 2. Goals and non-goals
**Goals** — G1 a dead/stuck pillar loses its weight within ~2 h, regains it automatically with evidence, and is always visible; G2 weights adapt to measured usefulness only as fast as the evidence allows; G3 bounded, smooth, deterministic, explainable, replayable; G4 fail static; G5 shadow first with numeric promotion criteria; G6 everything the operator fixed stays fixed (PAPER, 09:45 earliest entry, thresholds 0.15/0.30, `min_pillars`, quantities, exits, guards, sizing, decider, Jev).
**Non-goals** — tuning thresholds; changing the GEX modulator's form; touching execution, `engine.py` (another lane's uncommitted work), Hydrinia/tradingbot; training on Jev outputs; claiming profitability.

## 3. Architecture (v2): three layers, two of them live-capable

The forecast-combination literature (DeMiguel–Garlappi–Uppal 2009, RFS; Smith–Wallis 2009, Oxford Bull. Econ. Stat.) and R1's power analysis agree: with ~117–220 independent blocks, estimated weights add more variance than they remove. So **automatic dynamics come from liveness (L1) and conservation/bounds (L3); learned quality (L2) is built, persisted and measured but cannot move live weights until it earns it.**

### L1 Liveness gate — live-capable
* Pillars: **premium, darkpool**, per `(underlying, pillar, source)` where source ∈ {uw, selfflow}. NOFA exempt (flat zero is legitimate; absence handled by `nofa_unavailable`). Kronos exempt (own freshness gate). GEX is a modulator.
* Ring of the last `K = 40` in-session refresh values (UW cadence 180 s ⇒ ~2 h). **Dead** ⇔ ring full and (`max|v| < ε` or all values identical), `ε = 0.005`.
* **Hysteresis**: revive only after `R = 6` consecutive refreshes with `|v| ≥ 2ε = 0.01` **and** at least two distinct values among those last 6 (a frozen non-zero feed must not duty-cycle dead/alive; added after the 2026-10-10 review); **dwell**: no flip within 30 min of the previous flip; after revival the weight **ramps** in over 3 refreshes (1/3, 2/3, 1).
* **State persists across sessions**: a pillar dead at yesterday's close stays dead until it proves alive (otherwise the dead pillar would dilute the first ~2 h of every day). Rings persist within a session and are restored on restart (same ET `session_day`, state file younger than 15 min); with no usable state a pillar is treated as alive — exactly today's behaviour, never worse.
* Always emits a WARNING (deduped hourly) when a pillar turns dead and an INFO when it revives, in **every** mode.

### L2 Quality diagnostic — built and logged, **not applied** (default)
* Per `(underlying, pillar)`, once per 30-minute block (non-overlapping) the data layer records `x` = pillar value at the block start, `z` = trailing 15-minute return, `y` = forward 30-minute return, admitted only after the horizon has closed (no look-ahead). Samples are keyed `(session_day, block_index)` ⇒ **idempotent** (a crash/restart cannot double count).
* Sufficient statistics only: `n, Σx, Σz, Σy, Σxx, Σzz, Σyy, Σxz, Σxy, Σzy`, exponentially decayed at each session roll (half-life 20 sessions). Quality metric = **partial IC** `r(x,y | z)` (controls for the momentum echo R1 found).
* Posterior `IC_post = n_eff/(n_eff + n₀)·IC_partial`, `n₀ = 1000`; multiplier `m = clip(1 + κ·IC_post, 0.8, 1.2)`, `κ = 4`. **Applied only if** `CLIO_DPOF_DW_QUALITY_APPLY=1` **and** `n_eff ≥ 350` (note: with 11 samples per session and a 20-session half-life `n_eff` is bounded by 322.9, so this gate is unreachable until `n_floor` or the decay is revisited; see ADR-007) **and** the 95% CI of `IC_partial` excludes 0; otherwise `m = 1` exactly. R1: at 120–220 blocks the multipliers would be 0.85–1.12 — L2 is inert today by design.
* Whether L2 ever moves live weights is a **separate operator decision** (§6).

### L3 Combine, conserve, bound
* `rawᵢ = priorᵢ · alivenessᵢ · mᵢ` over **available** pillars (aliveness ∈ [0,1]: 0 dead, ramp while reviving, 1 alive).
* **Conservation target = Σ prior over the available pillars** (dead ones included). Shares `sᵢ = rawᵢ/Σ raw`; `wᵢ = sᵢ · target`. Consequence: when nothing is dead and no multiplier applies, `wᵢ = priorᵢ` — **bit-for-bit within 1e-12 the static composite** — and when a pillar dies its weight is *redistributed*, so the GEX term and the 0.15/0.30 thresholds keep their calibration.
* Bounds when ≥ 3 pillars carry weight: `s ≤ 0.60`, `s ≥ 0.05` as an exact projection `w_i = clip(λ·s_i, 0.05, 0.60)` with the one λ that makes the sum 1 (unpinned pillars keep their proportions; solved between the sorted breakpoints, no iteration; infeasible ⇒ unclamped shares). The first build used a same-pass clamp that returned no result for skewed shares and lifted pillars the cap redistribution would already have cleared; replaced after the 2026-10-10 review. Per-session move cap 0.10 and EMA `α = 0.3` apply to the **quality** component only; liveness changes are immediate (dwell-limited).
* If `Σ raw == 0` or any input is non-finite ⇒ return no weights ⇒ static.

### Dead pillars and the rest of the engine (R3 findings)
* **Mask, don't pop**: evicting `_pillar_cache` does not stick (the next 180 s refresh re-inserts it). A dead pillar is masked **inside `_build_underlying_snapshot`**: treated as missing *before* the existing per-pillar self-flow substitution (UW wins when alive; sources are never blended); if self-flow cannot supply it the pillar is `PillarReading.missing`.
* **`min_pillars` is unchanged.** A masked pillar counts as unavailable, so if both premium and darkpool are dead and not substituted, `count_available` = NOFA + GEX = 2 < 3 → `primary_composite_unavailable` → `data_unhealthy` veto (the engine goes quiet *visibly*, which is `min_pillars`' purpose). This is a **promotion-time operator confirmation item**, not a build decision; in shadow it changes nothing.
* The weights live on `UnderlyingSnapshot` (a new defaulted, never-persisted field, `models.py`), and `compute_composite(u, cfg, weights=None)` uses `weights` else `u.pillar_weights` else static. Every existing caller passes `(u, cfg)` positionally (R3), so both engine call sites are reached **without editing `engine.py`**.
* Modes `CLIO_DPOF_DYNAMIC_WEIGHTS = off | shadow | live`, code default **shadow**, invalid ⇒ **off**. `off` = today's path byte-for-byte; `shadow` = dynamic weights and the resulting composite are computed and logged beside the static one while **decisions use static**; `live` = decisions use dynamic. Kill-switch: `off` + restart.

## 4. Components and API contract

| Piece | File | Notes |
|---|---|---|
| Pure core | NEW `agent/dpof/pillar_weights.py` | stdlib only; **no `agent.*` imports**; no I/O, no clock reads (all times/days are arguments); deterministic |
| State + shadow I/O | NEW `agent/dpof/pillar_weights_io.py` | atomic JSON store + trimmed JSONL; sidecar lock; sync core + executor wrappers |
| Model field + weights param | `agent/dpof/models.py`, `agent/dpof/signals.py` | defaulted field; `compute_composite(u, cfg, weights=None)` |
| Config | `agent/dpof/config.py` | knobs normalized in `load_dpof_config`; dataclass default == `from_env` default |
| Integration | `agent/dpof/data.py` | observe hooks after each pillar cache write, session roll, masking, shadow record |
| Report/replay | NEW `scripts/dpof_dynamic_weights_report.py` | read-only |

### `agent/dpof/pillar_weights.py` (public surface)
```
SCHEMA_VERSION = 1
DIRECTIONAL = ("nofa", "premium", "darkpool", "kronos")
LIVENESS_PILLARS = ("premium", "darkpool")

@dataclass(frozen=True) class WeightParams:   # defaults = §3 values
    prior: Mapping = {"nofa": .40, "premium": .25, "darkpool": .20, "kronos": .15}
    ring_size=40; dead_eps=0.005; revive_eps=0.01; revive_n=6; dwell_seconds=1800.0; ramp_refreshes=3
    cap=0.60; floor=0.05
    block_minutes=30; n0=1000.0; n_floor=350; kappa=4.0; m_lo=0.8; m_hi=1.2
    half_life_sessions=20.0; quality_apply=False

class LivenessTracker:
    observe(key, value, ts, session_day, in_session) -> None      # value None/NaN/inf/bool ignored; not in_session ignored
    aliveness(key) -> float                                       # 1.0 alive/unknown, 0.0 dead, ramp in (0,1) while reviving
    is_dead(key) -> bool;  transitions(key) -> list[...]          # for logging
    to_dict() -> dict;  @classmethod from_dict(d, params, now_ts, max_age_seconds) -> LivenessTracker  # corrupt/old => empty
class QualityStats:                                               # one per (underlying, pillar)
    add_block(session_day, block_index, x, z, y) -> bool          # False if already added (idempotent) or non-finite
    roll_session(new_session_day) -> None                         # decay once per new day (idempotent per day)
    partial_ic() -> float|None;  n_eff() -> float;  ci95() -> (lo, hi)|None
    multiplier(params) -> float                                   # 1.0 unless params.quality_apply and n_eff>=n_floor and CI excludes 0
    to_dict()/from_dict()
def compute_weights(available: Mapping[str, float], multipliers: Mapping[str, float], params) -> dict[str, float]
    # available: pillar -> aliveness in [0,1] (only pillars present in the snapshot). Returns {} => caller uses static.
```
### State file `data/dpof/pillar_weights.json`
`{"schema_version":1,"written_at":<epoch>,"session_day":"YYYY-MM-DD","liveness":{"SPY|premium|uw":{...}},"quality":{"SPY|premium":{...}},"last_weights":{"SPY":{"nofa":..}}}` — atomic tmp + fsync + `os.replace`; sidecar lock `.dpof_pillar_weights.lock`; corrupt ⇒ rename `.corrupt-<epoch>` and start empty; writes only from an executor, throttled (≥ 60 s), at session roll and on state change. **Runtime state: never deleted by tooling.**
### Shadow sidecar `data/dpof/pillar_weights_shadow.jsonl`
One line per underlying per cycle: `{"ts": isoformat(now in ET) (joins signals.jsonl), "u", "mode", "static", "dyn", "w": {...}, "alive": {...}, "m": {...}, "avail": [...]}`; trimmed to ≤ 40,000 lines (~25 sessions); same executor + sidecar-lock pattern as `signal_log.py`. (`signal_log.py` is opaque to extra keys and the row is built in `engine.py`, so a sidecar is the engine-free route — R3.)

## 5. Failure semantics
Corrupt/missing/too-old state, non-finite values, any exception in the dynamic path ⇒ static weights + one WARNING per hour; weights never persisted with NaN/inf; shares always sum to 1 and respect bounds; no I/O on the event loop thread (anti-patterns #19, #37); no I/O inside `compute_composite`.

## 6. Promotion criteria (shadow → live) — all required; the operator decides
1. ≥ 15 RTH sessions of shadow data, zero exceptions in the dynamic path; replay of stored sessions reproduces the logged shadow composite within 1e-12.
2. Every dead episode detected within `K` and recovered within `R` (planted-episode replay); false-dead < 1% of minutes on known-healthy windows; ≤ 1 liveness flip per pillar per 30 min.
3. `|pre|` distribution and threshold-crossing rate within ±15% of static on replay; nomination episodes within [0.67×, 1.5×] of static with every divergence explained; placebo (permuted labels) change within the noise band.
4. No pillar above the cap; with nothing dead and L2 inert, dynamic == static to 1e-12 on **every** cycle.
5. Operator confirms the `min_pillars` consequence (§3) and updates the stale 40/25/20/15 prose in `agent/dpof/prompts.py:63` and the comment in `scripts/decisions_replay_corpus.py:244`.
6. **L2 → live is a separate decision**: block-bootstrap 95% CI of the partial IC excludes 0 over ≥ 120 sessions, sign-stable across halves.

## 7. Risks
| Risk | Mitigation |
|---|---|
| False "dead" on a quiet tape | K≈2 h + near-zero/identical requirement; hysteresis + dwell; WARNING always on; R1: 0 false positives on the healthy series; real-darkpool FP rate unmeasurable until regular-session data exists (shadow measures it) |
| Overfitting ~10 sessions | L2 inert by default; n₀ = 1000, n_floor = 350, CI-excludes-0 gate, clip [0.8, 1.2] |
| Look-ahead | blocks admitted after the horizon; planted future-leak mutant must fail |
| Momentum echo / double counting with the VWAP+15-min gate | partial IC controls for trailing return |
| Restart mid-session / state corruption | persisted rings + dead flags; atomic writes; quarantine; fail-static |
| Self-flow substitution churn | key by (pillar, source); dwell; source switch visible in the shadow log |
| Event-loop blocking | executor-only I/O; throttled writes |
| Behavioural surprise at restart | default `shadow` ⇒ decisions unchanged |

## 8. Test strategy
Seeded property tests (shares sum to 1 within 1e-12; bounds; dead ⇒ 0 immediately; revival ramp; ordering invariance; NaN/inf/bool/`10**400` ⇒ ignored — assert on results; persist/restore keeps dead state; idempotent block/session updates; determinism). Mutation matrix with a planted future-leak mutant, sign-flip, dead/alive flip fuzzing, stuck-value and frozen-input cases. Concurrency stress on the state writer (10 threads × 50 ops × 5 runs, zero corruption). Replay of stored sessions through the pure core. Import-isolation (`data.py` must not import `agent.config`/`httpx`; the new module must import nothing from `agent.*`).

## 9. Known unknowns (carried)
Whether any pillar has a real predictive edge (needs ≥ 16 pooled sessions at 22 blocks/session for MDE 0.15); healthy-darkpool/premium value distributions on regular-session UW data; UW refresh phase; Alpaca bar-timestamp convention (R1 assumed open-stamped, conservative under either); `as_of` staleness of UW data (a frozen feed with a fresh fetch stamp is not detected by value-based liveness; the existing 3×refresh staleness covers fetch failures only).
