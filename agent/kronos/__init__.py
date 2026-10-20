"""CLIO Kronos — candlestick-foundation-model forecaster (docs/design/clio_kronos.md).

Runs as its own service (`python -m agent.kronos`, systemd unit `clio-kronos.service`) and writes
its forecast and track record to JSON files that DPOF reads with a freshness check
(`agent.kronos.state`). Advisory only: it never nominates, sizes, vetoes or exits a trade.
"""
