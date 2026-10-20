"""
CLIO DPOF — Directional Premium / Options-Flow engine.

A long-options-only (long calls, long puts, optional debit verticals)
execution lane for 1-2 liquid underlyings (default SPY/QQQ), driven by the
Hydrinia DPOF method (NOFA 4-pillar options-flow composite + dark-pool veto),
hardened against the failure classes that method showed in Hydrinia, with an
optional Jev (TypeSafe) fast advisory/veto layer.

Mode is OFF by default (CLIO_DPOF_MODE). See docs/design/clio_dpof.md.
"""
