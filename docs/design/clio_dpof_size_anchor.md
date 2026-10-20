# CLIO DPOF — advisory size anchor

**Date**: 2026-09-29 (ET) · **Operator (Dom) decisions**: lever = advisory anchor (not a cap); scale = **~10% of the DPOF slice in premium at conviction 0.50, ~20% at 0.80**.
**Status**: built on `feat/clio-dpof-size-anchor`, NOT deployed (needs a moltbot restart).

## Why
Recent DPOF entries were 3–4 contracts (about $700–1,000 of premium, 4–5% of the ~$20K slice) at conviction 0.42–0.50, and the decision model has no reference for what a "normal" size is: the 2026-09-24 no-hard-caps directive removed every mechanical sizing rule and the charter only says to size from conviction, premium at risk, exposure and the day so far.

A read-only probe of the Hydrinia tree (2026-09-29) found no aggressive Kelly recipe to port: Hydrinia's live options entries used a linear conviction ramp under a small per-trade cap (tighter than CLIO's current sizing), its Kelly sizer served a different path and ended clamped to a few percent, the Perseus flow loop (Pe1, "DPOF-style") has no sizing of its own, and two sizers are dead code. The useful idea that carries over is the **conviction ramp**; a Kelly fraction cannot be estimated from CLIO's handful of closed trades.

## What it does
`agent/dpof/sizing.py` `anchor_share()` / `anchor_qty()` (pure): premium share of `dpof_equity` for a full entry = linear through (0.50, 10%) and (0.80, 20%), clamped to [2%, 30%]. For every entry candidate the decision model now sees the contracts that share would buy at conviction 0.40 / 0.50 / 0.65 / 0.80 / 0.90, bounded by the candidate's `max_affordable_qty`, plus one advisory section ("NOT a limit and NOT an instruction") stating the scale. About 1,000 characters per entry deliberation.

**It is not a cap.** The model still picks its own qty; `clamp_to_solvency` remains the only code-side size bound; the anchor does not shrink for open positions or a bad day (the section tells the model to weigh those itself). No track-record shrinkage on purpose: with ~4 closed trades it would cancel the requested scale, and the model already sees its own closes and lessons. Review (exit) messages and the static charter are untouched (prompt cache stays valid).

## Knobs (all normalised inside `load_dpof_config`; a typo falls back to the default, never stops a cycle)
| Env | Default | Meaning |
|---|---|---|
| `CLIO_DPOF_SIZE_ANCHOR` | `1` | `0` = off: no ctx key, decider message byte-identical to before (rollback = set `0` + restart) |
| `CLIO_DPOF_ANCHOR_SHARE_C50` | `0.10` | share at conviction 0.50, in (0, 0.5] |
| `CLIO_DPOF_ANCHOR_SHARE_C80` | `0.20` | share at conviction 0.80, in (0, 0.5], must be ≥ the c50 share (else both reset) |

Sizing is against `dpof_equity` = min(real equity, `CLIO_DPOF_CAPITAL` $20,000 + DPOF's own P&L). Raising the slice is a separate lever (an `.env` edit) and multiplies everything here.

## Risks / things to watch
* **Jev `size_reckless` veto** (`CLIO_DPOF_JEV_SIZE_T`, default 0.70) judges whether a size is "proportionate to capital and conviction": anchor-scale entries may draw more vetoes. Watch `reasons: ['size_reckless']` in `data/dpof/decisions.jsonl`; raise the threshold or lower the shares if it bites.
* **Concentration**: at 20% of the slice per entry, a handful of 1DTE entries that expire worthless in one day can cost a large share of the slice. The model sees the anchor as a scale, not a target; there is still no per-day loss limit by directive.
* Bounded by solvency exactly as before; paper mode.

Tests: `tests/dpof/test_size_anchor.py` (77 tests, 41/41 mutants killed).
