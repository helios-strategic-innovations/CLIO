r"""
CLIO flowcollector — self-computed UW-style options-flow pillars from Alpaca
Algo Trader Plus market data (OPRA options + SIP equities), SPY/QQQ only.

SHADOW ONLY (2026-09-25 phase, INTEL_V3_SPEC.md): this package's own
long-running process (`python -m agent.flowcollector`) streams and
classifies Alpaca market data and writes a state file backing
`get_selfflow()`; nothing in CLIO reads that output yet. This package never
imports `agent.dpof.broker` / `agent.dpof.orders` / `agent.dpof.engine` (the
DPOF order chokepoints), never imports the `alpaca.trading.client` order-
submission surface, and contains no order/position/cancel code anywhere --
it is a pure market-data reader/producer, exactly like
`agent.integrations.uw_flow_intel` is for UW data. Grep-provable:
    grep -rn "dpof.broker\|dpof.orders\|dpof.engine\|submit_order\|cancel_order" agent/flowcollector/
must return nothing.

Public contract (see agent/flowcollector/state.py for the full docstring):
    get_selfflow(symbol: str, max_age_seconds: int = 300) -> dict | None
returning {nofa, premium, darkpool, gex, nope, max_pain, gex_regime, as_of,
quality_notes} (plus additive `windows`/`darkpool_detail`/`gex_detail` --
see agent/flowcollector/aggregator.py).

Research basis: scratchpad recon doc "SELF_COMPUTE_FLOW.md" (alpaca-py
0.44.0, Apache-2.0, read directly from the installed package). No AGPL or
otherwise-unlicensed third-party code is copied anywhere in this package --
GEX/max-pain/NOPE/quote-rule/Lee-Ready are standard, uncopyrightable
market-microstructure formulas/methodology, implemented from scratch against
Alpaca's own field names.
"""

from __future__ import annotations

from agent.flowcollector.state import get_selfflow

__all__ = ["get_selfflow"]
