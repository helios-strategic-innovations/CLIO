"""
Hydrinia bridge switch — single gate for every CLIO <-> Hydrinia touchpoint.

Hydrinia (/home/agent/tradingbot) is permanently off (operator direction,
2026-09-24): CLIO trades natively via agent/dpof/. Its files still exist on
the VPS and go stale, so CLIO must neither READ them as current state nor
keep WRITING into its bridge directories.

Every remaining Hydrinia read/write/health-check is gated on
`bridge_enabled()`. Default OFF; set CLIO_HYDRINIA_BRIDGE_ENABLED=1 only if
Hydrinia is ever revived. Trading-facing reads (positions, P&L, regime) are
NOT gated — they were repointed to CLIO-native sources (agent/dpof/readmodel.py,
the CLIO Alpaca account, agent/integrations/market_regime.py).
"""

from __future__ import annotations

import os


def bridge_enabled() -> bool:
    """True only when the operator explicitly re-enables the Hydrinia bridge."""
    raw = os.environ.get("CLIO_HYDRINIA_BRIDGE_ENABLED", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")
