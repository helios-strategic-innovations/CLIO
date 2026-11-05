# CLIO decision desk

A read-only demonstration of CLIO's decision evidence. It is a standalone presentation, not the previously deployed CLIO interface and not a live account dashboard.

## Open and explore

Open `index.html` in a browser. No server, installation or external library is required. Select **Decision journal**, **Synthetic cases** or **Evidence & scope** in the sidebar. Filter the journal by entry or review, inspect a row, and export the selected sanitized event as JSON. The synthetic pane switches between four saved evaluator outputs and can export a selected case. Marble and Basalt change the local display theme; **Reload snapshot** resets the journal to its bundled records and does not query a broker.

The Telegram link opens the existing `@DN_Moltbot` conversation outside this page. **Copy command** copies `/dpof status` when clipboard access is available, or selects it for manual copying. This dashboard does not send, poll, authenticate, trade or create a Telegram bot.

## What the data means

The journal contains six selected, sanitized historical PAPER events through November 4, 2026 at 12:15 ET, before the later shadow installation. Their display order is not a per-event timestamp. Meta Muse Wonder 1.0 appears in selected entry deliberations and Spark 1.3 in selected position reviews. The snapshot has no account balances, private prompts, fill details or financial outcome values; missing information is shown as unknown.

The four separate synthetic cases are saved outputs from the existing re-entry evaluator. They are not live market data or a fresh calculation in this page. The observer is diagnostic and does not submit, block or resize orders. Normal post-restart shadow emission and economic benefit remain unverified.

`PROVENANCE.md` describes the source, saved run and the derived offline adapter. To run that adapter separately with Python 3.14 and bundled IANA time-zone data, use `python -B run_demo.py`. The page itself does not run Python or invoke a model, broker or provider. `META_EVIDENCE.json` is the sanitized categorical evidence used for the journal.

## Checks and rights

`checks/interaction_result.txt`, `checks/static_result.txt` and `checks/lint_result.txt` capture local synthetic-DOM and static checks. They are not browser-rendering, accessibility audit, current runtime or live-account evidence. The included IANA notice is in `tzdata/TZDATA_COPYRIGHT.txt`. CLIO was created by Dominic Newman with AI assistance. No Helios code license is assigned in this draft; public release remains an owner decision.
