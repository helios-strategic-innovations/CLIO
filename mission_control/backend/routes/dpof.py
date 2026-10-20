"""DPOF routes — CLIO's own SPY/QQQ long-options engine (read-only).

CLIO-native (Hydrinia-independence build, 2026-09-24): DPOF is CLIO's own
trading engine, running in-process inside moltbot. Mission Control is a
SEPARATE process, so these routes only ever go through
`agent.dpof.readmodel` (sync, read-only, never raises) — never
`agent.dpof.engine`, which is in-process only.
"""

import logging

from fastapi import APIRouter, Depends, Query

from mission_control.backend.auth import require_auth
from mission_control.backend.deps import run_in_executor

router = APIRouter(prefix="/api/dpof", tags=["dpof"])
logger = logging.getLogger("mission_control.dpof")


async def _market_regime_block() -> dict | None:
    """CLIO-native market regime, if the module is available — never raises."""
    try:
        from agent.integrations.market_regime import get_market_regime
    except Exception as e:
        logger.debug(f"market_regime module unavailable: {e}")
        return None
    try:
        return await run_in_executor(get_market_regime)
    except Exception as e:
        logger.debug(f"market_regime.get_market_regime() failed: {e}")
        return None


@router.get("/status")
async def dpof_status(_=Depends(require_auth)):
    """DPOF snapshot (mode, halt state, open/closed counts, P&L) + market regime."""
    snapshot: dict = {}
    try:
        from agent.dpof import readmodel

        snapshot = await run_in_executor(readmodel.snapshot)
    except Exception as e:
        logger.error(f"DPOF snapshot read failed: {e}", exc_info=True)

    regime = await _market_regime_block()
    return {
        "dpof": snapshot,
        "market_regime": regime,
        "source": "clio_dpof",
    }


@router.get("/positions")
async def dpof_positions(_=Depends(require_auth)):
    """Open DPOF positions, plus positions closed today — formatted + raw."""
    try:
        from agent.dpof import readmodel

        open_positions = await run_in_executor(readmodel.open_positions)
        closed_today = await run_in_executor(readmodel.closed_today)
    except Exception as e:
        logger.error(f"DPOF positions read failed: {e}", exc_info=True)
        return {
            "open": [],
            "open_lines": [],
            "closed_today": [],
            "closed_today_lines": [],
            "source": "clio_dpof",
        }

    try:
        from agent.dpof.readmodel import format_position_line
    except Exception as e:
        logger.debug(f"DPOF line formatter unavailable: {e}")
        format_position_line = None

    def _line(p) -> str:
        if format_position_line is None:
            return f"{p.underlying} {p.direction} {p.structure} [{p.status}]"
        try:
            return format_position_line(p)
        except Exception as e:
            logger.debug(f"DPOF position line format failed for {getattr(p, 'id', '?')}: {e}")
            return f"{p.underlying} {p.direction} {p.structure} [{p.status}]"

    return {
        "open": [p.to_dict() for p in open_positions],
        "open_lines": [_line(p) for p in open_positions],
        "closed_today": [p.to_dict() for p in closed_today],
        "closed_today_lines": [_line(p) for p in closed_today],
        "source": "clio_dpof",
    }


@router.get("/decisions")
async def dpof_decisions(limit: int = Query(50, ge=1, le=500), _=Depends(require_auth)):
    """Recent DPOF cycle decisions (signal/entry/exit reasoning), newest last."""
    decisions: list = []
    try:
        from agent.dpof import readmodel

        decisions = await run_in_executor(readmodel.recent_decisions, limit)
    except Exception as e:
        logger.error(f"DPOF decisions read failed: {e}", exc_info=True)

    return {
        "decisions": decisions,
        "count": len(decisions),
        "source": "clio_dpof",
    }
