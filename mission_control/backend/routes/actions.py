"""Quick actions — flatten, halt, resume."""

import asyncio
import json
import logging
import os
import subprocess
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from mission_control.backend.auth import require_auth, require_trade_execution_enabled
from mission_control.backend.deps import get_trading_client, run_in_executor
from mission_control.backend.config import mc_config

router = APIRouter(prefix="/api/actions", tags=["actions"])
logger = logging.getLogger("mission_control.actions")

_last_halt_time: float = 0.0
_halt_lock = asyncio.Lock()  # S162 C-1: Serialize halt operations

_EMERGENCY_STOP_PATH = os.path.join(mc_config.tradingbot_root, "state", "EMERGENCY_STOP")


class ConfirmAction(BaseModel):
    confirmation: str  # Must match expected string


@router.post("/flatten")
async def flatten_all(
    req: ConfirmAction,
    _=Depends(require_auth),
    _gate=Depends(require_trade_execution_enabled),
):
    """Close ALL positions in the CLIO Alpaca account — requires typing FLATTEN to confirm.

    This is account-wide (not DPOF-scoped): it closes every open position on the
    account DPOF trades from, not only DPOF's own book.
    """
    if req.confirmation != "FLATTEN":
        raise HTTPException(status_code=400, detail="Type FLATTEN to confirm")
    try:
        client = get_trading_client()
        await run_in_executor(lambda: client.close_all_positions(cancel_orders=True))
        logger.warning("FLATTEN ALL executed from Mission Control")
        return {
            "status": "all_positions_closed",
            "cancel_orders": True,
            "note": "Closed every position on the CLIO Alpaca account, not just DPOF's book.",
        }
    except Exception as e:
        logger.error(f"Flatten failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/halt")
async def halt_trading(_=Depends(require_auth)):
    """Block new DPOF entries from the next cycle (the '/dpof halt' equivalent).

    This does NOT force-close open DPOF positions — those stay under the engine's
    own exit logic (stops/targets/thesis-invalidation) until they close naturally
    or an operator explicitly flattens the account. Hydrinia's EMERGENCY_STOP +
    systemctl stop are only touched when CLIO_HYDRINIA_BRIDGE_ENABLED=1.
    """
    global _last_halt_time
    import time as _time_mod
    # S162 C-1: Serialize halt operations to prevent TOCTOU on cooldown check
    async with _halt_lock:
        if _time_mod.time() - _last_halt_time < 300:  # 5-minute cooldown
            raise HTTPException(status_code=429, detail="Halt cooldown: wait 5 minutes between halt commands")
        logger.critical("S162 C-1: HALT TRADING issued from Mission Control")
        _last_halt_time = _time_mod.time()

    reason = "HALT issued from Mission Control"
    dpof_halt_written = False
    hard_halted = False
    halt_reason = ""
    try:
        from agent.dpof import readmodel
        dpof_halt_written = bool(await run_in_executor(readmodel.write_halt_file, reason))
        risk = await run_in_executor(readmodel.load_risk)
        hard_halted = bool(risk.hard_halted)
        halt_reason = risk.halt_reason
    except Exception as e:
        logger.error(f"DPOF halt failed: {e}", exc_info=True)

    hydrinia_stopped = False
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if bridge_enabled():
            # Write EMERGENCY_STOP file (atomic, JSON matching Hydrinia format — S136 C1)
            tmp_path = _EMERGENCY_STOP_PATH + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump({"activated_at": time.time(), "reason": reason}, f)
            os.replace(tmp_path, _EMERGENCY_STOP_PATH)
            await run_in_executor(lambda: subprocess.run(["sudo", "systemctl", "stop", "hydrinia"], timeout=30))
            hydrinia_stopped = True
    except Exception as e:
        logger.error(f"Hydrinia halt path failed: {e}", exc_info=True)

    logger.warning("HALT TRADING executed from Mission Control (DPOF entry halt)")
    message = (
        "New DPOF entries are blocked starting next cycle. Open DPOF positions are still "
        "managed by the engine's own exits — this does not flatten them; use Flatten for "
        "that. Equivalent to '/dpof halt' in Telegram."
    )
    if hard_halted:
        message += (
            f" NOTE: an in-process hard halt is ALSO active ({halt_reason or 'reason unavailable'}) "
            "— clear it with '/dpof resume' in Telegram."
        )
    return {
        "status": "trading_halted",
        "dpof_halt_file": dpof_halt_written,
        "hard_halted": hard_halted,
        "halt_reason": halt_reason,
        "hydrinia_stopped": hydrinia_stopped,
        "emergency_stop": hydrinia_stopped,  # legacy field name, Hydrinia-bridge-scoped
        "message": message,
    }


@router.post("/resume")
async def resume_trading(_=Depends(require_auth)):
    """Allow new DPOF entries again (the '/dpof resume' equivalent for the HALT file).

    Cannot clear an in-process hard halt (weekly loss / drawdown / manual) — that
    requires '/dpof resume' in Telegram. Hydrinia's EMERGENCY_STOP + systemctl start
    are only touched when CLIO_HYDRINIA_BRIDGE_ENABLED=1.
    """
    dpof_halt_cleared = False
    hard_halted = False
    halt_reason = ""
    try:
        from agent.dpof import readmodel
        dpof_halt_cleared = bool(await run_in_executor(readmodel.clear_halt_file))
        risk = await run_in_executor(readmodel.load_risk)
        hard_halted = bool(risk.hard_halted)
        halt_reason = risk.halt_reason
    except Exception as e:
        logger.error(f"DPOF resume failed: {e}", exc_info=True)

    hydrinia_started = False
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if bridge_enabled():
            if os.path.exists(_EMERGENCY_STOP_PATH):
                os.remove(_EMERGENCY_STOP_PATH)
            await run_in_executor(lambda: subprocess.run(["sudo", "systemctl", "start", "hydrinia"], timeout=30))
            hydrinia_started = True
    except Exception as e:
        logger.error(f"Hydrinia resume path failed: {e}", exc_info=True)

    logger.warning("RESUME TRADING executed from Mission Control (DPOF entry halt cleared)")
    message = "New DPOF entries are allowed again (HALT file cleared)."
    if hard_halted:
        message += (
            f" NOTE: an in-process hard halt is STILL active ({halt_reason or 'reason unavailable'}) "
            "— this action cannot clear it, use '/dpof resume' in Telegram."
        )
    return {
        "status": "trading_resumed",
        "dpof_halt_cleared": dpof_halt_cleared,
        "hard_halted": hard_halted,
        "halt_reason": halt_reason,
        "hydrinia_started": hydrinia_started,
        "emergency_stop": False,
        "message": message,
    }


@router.post("/restart/{service}")
async def restart_service(service: str, _=Depends(require_auth)):
    """Restart a specific service."""
    from mission_control.backend.config import effective_allowed_services
    if service not in effective_allowed_services():
        raise HTTPException(status_code=400, detail=f"Unknown service: {service}")
    try:
        result = await run_in_executor(lambda: subprocess.run(
            ["sudo", "systemctl", "restart", service],
            capture_output=True, text=True, timeout=30,
        ))
        logger.warning(f"Service {service} restarted from Mission Control")
        return {"service": service, "restarted": result.returncode == 0}
    except Exception as e:
        logger.error(f"Service restart failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status")
async def get_trading_status(_=Depends(require_auth)):
    """DPOF-native trading status: mode, HALT file, hard-halt, open positions."""
    mode = "unknown"
    halt_file = False
    hard_halted = False
    halt_reason = ""
    open_count = 0
    try:
        from agent.dpof import readmodel
        from agent.dpof.config import load_dpof_config

        cfg = await run_in_executor(load_dpof_config)
        mode = cfg.mode.value
        halt_file = bool(await run_in_executor(readmodel.halt_file_present))
        risk = await run_in_executor(readmodel.load_risk)
        hard_halted = bool(risk.hard_halted)
        halt_reason = risk.halt_reason
        open_positions = await run_in_executor(readmodel.open_positions)
        open_count = len(open_positions)
    except Exception as e:
        logger.error(f"DPOF status read failed: {e}", exc_info=True)

    moltbot_active = False
    try:
        result = await run_in_executor(lambda: subprocess.run(
            ["systemctl", "is-active", "moltbot"],
            capture_output=True, text=True, timeout=5,
        ))
        moltbot_active = result.stdout.strip() == "active"
    except Exception:
        moltbot_active = False

    # Legacy Hydrinia fields — only meaningful (and only checked) when the bridge
    # is explicitly re-enabled; otherwise stay False/no-op (no log spam).
    emergency = False
    hydrinia_active = False
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if bridge_enabled():
            emergency = os.path.exists(_EMERGENCY_STOP_PATH)
            result = await run_in_executor(lambda: subprocess.run(
                ["systemctl", "is-active", "hydrinia"],
                capture_output=True, text=True, timeout=5,
            ))
            hydrinia_active = result.stdout.strip() == "active"
    except Exception as e:
        logger.debug(f"Hydrinia status check skipped: {e}")

    trading_active = mode in ("paper", "live") and not halt_file and not hard_halted

    return {
        "source": "clio_dpof",
        "mode": mode,
        "halt_file": halt_file,
        "hard_halted": hard_halted,
        "halt_reason": halt_reason,
        "open_count": open_count,
        "moltbot_active": moltbot_active,
        "trading_active": trading_active,
        # Legacy field names kept for the current frontend; Hydrinia-bridge-scoped.
        "emergency_stop": emergency,
        "hydrinia_active": hydrinia_active,
    }
