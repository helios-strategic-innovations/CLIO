# ADR-006 — UW pillar scoring fixes: premium signal, darkpool NBBO, empty-pillar handling

**Date**: 2026-10-09 (Wave 2A)
**Status**: Proposed (builder draft; figures re-checked by the reviewing session against live UW responses on 2026-10-09; NOT operator-approved; nothing is live until `moltbot` restarts)
**Decider**: Operator (Dominic) — to approve flag rollout and the premium-pillar behaviour change
**Supersedes**: none. Builds on the Wave-1 premium compute fix in `uw_flow_intel._compute_premium_signals`.

## Context

DPOF's composite (`agent/dpof/signals.py`) weights NOFA .40, premium .25, darkpool .20, Kronos .15, GEX .15; `min_pillars=3` counts NOFA, premium, darkpool and GEX.

1. **Premium was a dead pillar.** `_compute_premium_signals` read `net_premium`/`call_premium`/`put_premium`, which do not exist in the live UW `net-prem-ticks` rows. The score was exactly 0.0 for all 6,476 logged UW-sourced SPY cycles (2026-09-28 to 2026-10-09); every non-zero premium value in `signals.jsonl` came from the self-flow fallback. Verified live row schema: `net_call_premium`, `net_put_premium`. Sign convention verified by per-tick correlation with `net_delta`: (nc − np) correlates 0.76, (nc + np) 0.32. Wave 1 fixed the compute.
2. **Darkpool was effectively dead.** The legacy pillar uses `bias_score = (VWAP − median) / median`, which sits near ±0.002 on live data (the UW data is fine). `agent/dpof/darkpool_nbbo.py` provides a scale-free signed NBBO-midpoint ratio in [−1, 1], or None when fewer than 20 prints classify. On the only sample available (after-hours) it read −0.86 to −0.96. It has **not** been calibrated on regular-session prints.
3. **Empty responses were stored as fake zeros.** Empty or unrecognized UW responses became `PillarReading(value=0.0, available=True, detail={"empty": True})`. This silently dilutes the composite. A failed fetch instead records `_pillar_last_error` and writes no cache entry, so after 3× `uw_refresh_seconds` the self-flow fallback (`_needs_selfflow`) takes over.
4. **Dilution arithmetic.** Premium and darkpool at 0.0 carry 0.45 of total weight with no information. For NOFA alone to clear the 0.15 threshold, |NOFA| must be about 0.43 (0.15 / 0.35 of the informative weight).

## Decision

1. **Premium fix ships unflagged.** It corrects a defect (scoring from keys that do not exist), not a new behaviour choice.
2. **Darkpool NBBO ratio behind `CLIO_DPOF_DARKPOOL_NBBO` (default OFF).** Flag off: legacy value used, plus one INFO line per ticker per refresh: `DPOF darkpool shadow <T>: legacy=… nbbo=… classified=… at_mid=…`. Flag on: NBBO ratio used (`source` ends `#nbbo`); under-sampled (<20 classified) becomes an empty reading.
3. **Empty and schema-drifted pillars behind `CLIO_DPOF_EMPTY_PILLAR_UNAVAILABLE` (default OFF).** Flag on: the reading follows the failed-fetch path (no cache write, `_pillar_last_error = "empty:<reason>"`, INFO log), so the self-flow fallback can apply. Flag off: legacy stored 0.0 reading, byte-for-byte.
4. **Schema canary.** Non-empty premium rows that carry none of the recognized fields log one WARNING per (pillar, ticker) per hour, naming field names only (never values).
5. **GEX regime transition INFO log** (`DPOF gex regime <T>: <prev> -> <new> (factor …)`), emitted only on change. No change to GEX values or empty-GEX handling.

## Consequences

- **Premium pillar is now volatile.** Scores saturate at ±1 in many 20-tick windows (191 bullish and 195 bearish readings on 2026-10-09). Entries and deliberations are likely to rise. Mitigation: lower `CLIO_DPOF_W_PREMIUM`.
- **With both new pillars unavailable** (`empty_pillar_unavailable=1` plus a failing NBBO sample), only NOFA and GEX remain, which is below `min_pillars=3`. The composite becomes None and `data_unhealthy` by design. This is the documented consequence, pinned by `tests/dpof/test_dpof_pillar_hardening.py` H9.
- **The premium flow is contemporaneous, not (yet shown to be) predictive.** Over 10 sessions, flow in a 5-minute block correlates about +0.5 with the *same* block's return but about 0 with the *next* block's (SPY +0.004, QQQ −0.072). It behaves like a momentum/confirmation signal. Restoring it will make the composite follow the tape and raise activity; whether it adds predictive value at the 30-minute trading horizon is being measured (research wave R1; see `docs/design/clio_dpof_dynamic_weights.md`).
- Decisions recorded 2026-10-10: (a) weights become dynamic and automatic (design doc above; shadow-first); (b) the decider's pass rationale is logged at INFO from `decider.py`; (c) the timeline sign is fixed and the tide block probed (inconclusive, untouched); (d) Hydrinia `tscore_collect.compute_net_premium` is left untouched — no CLIO consumer of its output was found, Hydrinia is retired, and a tradingbot edit would be a cross-repo write needing explicit approval.
- Log volume rises by one INFO line per ticker per UW refresh (darkpool shadow) and one WARNING per hour per drifted pillar at most.
- **CLIO's own UW composite changes too, outside DPOF and unflagged.** `_compute_premium_signals` is shared: `uw_flow_intel._compute_composite` (premium weight 0.25) now sees a live premium term for the 20 single-stock `_PREMIUM_TICKERS`. This takes effect at the next restart in every DPOF mode. Adversarial review (2026-10-10) traced the consumers: `learning_engine` U-3 direction-flip override (score > 0.20) can now let a flip trade idea through where a saturated premium score alone previously kept it suppressed (a narrow dedup relaxation on advisory idea files; nothing in `agent/dpof` reads them and the Hydrinia bridge is retired); the cross-instrument conflict gate (score >= 0.4) never sees these tickers (it only runs for `_ETF_UNDERLYING`, no overlap); `flow_alignment` is write-only; the premarket and system-prompt flow text and `alignment_checker` are advisory text. No order path is affected.

## Rollback

- Flags: unset or `0` for `CLIO_DPOF_DARKPOOL_NBBO` and `CLIO_DPOF_EMPTY_PILLAR_UNAVAILABLE`, then restart moltbot. Behaviour returns to legacy except for the new log lines.
- Code: `git apply -R <wave2a patch>`.

## Open questions

1. Weight renormalization over the informative pillars, instead of letting zero-information pillars dilute the composite.
2. `min_pillars` value once premium and darkpool can legitimately be unavailable.
3. Regular-session calibration of the NBBO ratio, before any flag flip. Ext-hours prints must be excluded or tagged.
4. GEX regime hysteresis (flip-flopping between regimes is not yet damped).
5. Decider pass-reason logging is deferred: it touches `engine.py`, which carries another lane's uncommitted work.
6. `get_ticker_flow_timeline` still used the `nc + np` convention; the 2026-10-10 decision is to fix it (`net_flow = nc − np`, wave B5, plan `docs/plans/S213_dpof_dynamic_weights_and_followups.md`). The market-tide block is **left untouched**: its premiums show |corr| ≤ 0.03 with 5-minute SPY/QQQ returns under either convention (813 intervals), so the probe is inconclusive; both consumers are advisory text and DPOF never reads them.
7. Hydrinia `tscore_collect.compute_net_premium` sums nc + np. Cross-repo, untouched by this ADR.
8. The premium sign convention (positive `net_put_premium` = puts net bought = bearish) is **confirmed against realized returns** (2026-10-10, 10 sessions, SPY/QQQ 1-minute SIP bars): the same 5-minute return correlates +0.565 (SPY) / +0.416 (QQQ) with `net_call − net_put` versus +0.146 / +0.116 with the sum, and the difference wins on 10 of 10 days for both symbols (also 0.76 vs 0.32 against `net_delta`). UW's own field definitions are still not in the repo (`docs/api/uw.txt` lists no response schema) — an external confirmation would still be good.
9. The corrected score is a 20-tick ratio that saturates at ±1; whether to smooth it (longer window, shrinkage toward 0) before it carries 0.25 of the composite is a tuning decision for the operator.
