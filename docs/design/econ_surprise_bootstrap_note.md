# econ_surprise z-score sparsity: bootstrap options (R-22)

Status: note only. No code change, no API calls. Written 2026-10-12 (S215 / R-22).

## Observed (read-only check of data/intelligence/econ_surprise.json)
- 151 surprises recorded; 1 has a non-null `surprise_z`.
- 96 history series; the 40 shortest all hold exactly one point.

## Why z is None
`_update_history` (agent/integrations/econ_surprise.py ~L429-449) returns a z only when
`len(prior) >= _MIN_HISTORY_FOR_Z` (= 8, L206) and the stdev over the last
`_Z_LOOKBACK` (= 36) prior points is above 1e-9. The live pipeline adds one point per
canonical series per release, so a new series needs 8 releases before its first z.
The None is the designed "no fake zeros" behaviour, not a bug.

## What bootstrap_history does (L1507-1557)
- Off by default. `refresh_econ_surprise` never calls it (pinned by
  tests/test_econ_surprise_refresh_and_readers.py::test_bootstrap_history_never_called_by_refresh).
- One TradingView range request per invocation (`_fetch_tradingview_raw`, default
  lookback 730 days, `countries=US`). The code issues a single `client.get`; no pagination loop.
- Gated by `check_and_record("econ_surprise")` (budget 1500/day, config.py L283).
- Seeds `state["history"][key]` with up to 60 values (`actual - consensus`, same-unit rows
  only). `canonical_key` scopes the seed to one series.
- The same TradingView range endpoint is already used by the slow refresh, so the seeded
  surprise definition matches the live path.

## Call count
- One HTTP request per run, about 1 of the 1500/day budget. Whole backfill = one run.
- Unverified: whether TradingView caps rows per response. If it does, seeded depth may
  fall below 8 for many series. The operator run must report the seeded counts.

## Options
1. Enable once: the operator runs `bootstrap_history()` once (no key), then live
   releases append. No code change. Cost: one request. Series without enough TradingView
   history stay sparse.
2. Lower `_MIN_HISTORY_FOR_Z` (e.g. to 4): z appears sooner, but stdev from 4 points is
   noisy. This changes a documented contract ("None if <8 history pts") and needs code
   and test changes.
3. Accept: z fills in as releases accrue (about 8 per series). No change.

## Recommendation
Option 1, run once by the operator, after checking the TradingView response shape and
that the seeded keys match live canonical keys. Do not change `_MIN_HISTORY_FOR_Z`. If
the seeded counts stay below 8 for most series, fall back to option 3.
