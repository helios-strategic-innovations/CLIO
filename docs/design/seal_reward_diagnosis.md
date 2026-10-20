# SEAL reward state freeze: diagnosis (S215 / R-28)

Question: why has `data/knowledge/seal_reward_state.json` not been written since
2026-03-27 (`total_evaluated` 27, `evaluated_ids` 4)? Read-only. The live file
was NOT opened (protected state). Sources: code, git, moltbot journal.

## Writer and gates

Only `SEALRewardEngine.record_binary_reward()` (`agent/integrations/seal_reward_engine.py`)
increments `total_evaluated` / `evaluated_ids`, and it saves after each record.

| Gate | Location | Env flag (code default) | Effect |
|---|---|---|---|
| flag off | reward_engine ~L161 | `MOLTBOT_SEAL_REWARDS` (`"1"`) | no write |
| unknown/empty id | ~L166-168 | none | WARNING, no write |
| dedup hit | ~L170-171 | none | DEBUG only, no write |

`MOLTBOT_SEAL_REWARD_LOG_ONLY` (`"1"`) only changes the log tag; it does not
block the save. It gates adjustment use in `learning_engine.py` ~L4516.

## Callers

1. Closed-trade debrief: `agent/integrations/trade_idea_feedback.py`
   `process_trade_idea_feedback()`. Top gate ~L26: `if not bridge_enabled(): return`.
   `bridge_enabled()` (`agent/hydrinia_bridge.py`) needs
   `CLIO_HYDRINIA_BRIDGE_ENABLED` in `1/true/yes/on` (code default: off). Reads
   `tradingbot/moltbot_knowledge/trade_debriefs/`. Reward hook ~L739 needs
   `MOLTBOT_SEAL_REWARDS == "1"` and a matched idea; otherwise INFO
   `SEAL reward skipped` (~L753).
2. Gap warning: same file, `_apply_gap_conviction_filter` ~L437. Needs
   `MOLTBOT_GAP_CONVICTION_LOG_ONLY != "1"` (default `"1"`, so off).
3. `MemoryEngine.integrate_seal_signals()` reads the summary only.

## Hypotheses

**H1, flag or LOG_ONLY blocks writes: not supported.** Only `MOLTBOT_SEAL_REWARDS`
gates writes (default on). The live `.env` was not read; if the running process
has it set to `0`, that alone would freeze writes. Operator to check the process
environment.

**H2, no closed trade matches a stored idea: supported, but upstream.** The
debrief path is dead while the bridge is off. Journal (coverage from 2026-09-29):
`trade_feedback` runs ~321 times with zero `binary reward`, `reward skipped`,
`skipping trade with unknown`, or gap-reward lines. The debrief directory is empty,
last changed 2026-09-16. Startup logs `Hydrinia bridge disabled
(CLIO_HYDRINIA_BRIDGE_ENABLED=0)`. Hydrinia retired 2026-09-24. This explains the
September silence, not a freeze from 2026-03-27 (no March journal here).

**H3, dedup swallows everything: not the cause.** Dedup returns after a DEBUG
line; the hook's `except` is also DEBUG (~L759, anti-pattern #27), so a silent
failure cannot be excluded from the journal. Dedup only fires for an id already
recorded.

**H4, another writer reset `evaluated_ids`: code-inconsistent as stated.**
`evaluated_ids` grows only with `total_evaluated`, and the 500 cap keeps the
newest entries. So 27 evaluations with 4 ids cannot come from the current code.
Possibilities: a file written by a pre-S151-B-C3 version, or an external edit.
Neither is verifiable here.

## Conclusion

- Since September: no writes because the debrief feed sits behind the retired
  bridge gate (H2). Expected design state, not a defect.
- March 27 freeze: not determinable from code or the current journal. The 27-vs-4
  mismatch points to an external rewrite or a version change (H4).
- Next step (operator, read-only): compare the live file mtime and the running
  process's `MOLTBOT_SEAL_REWARDS` against March deploy records.

## Fix decision

No reward-logic change: no unambiguous flag or path bug found. The `options`
bandit fix and calibration id fix are separate (commit 7ae958a).
