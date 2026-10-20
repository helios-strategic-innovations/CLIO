"""Directive routes — issue commands, view active/history, service control."""

import json
import logging
import os
import subprocess

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional

from mission_control.backend.auth import require_auth
from mission_control.backend.config import mc_config
from mission_control.backend.deps import run_in_executor

router = APIRouter(prefix="/api", tags=["directives"])
logger = logging.getLogger("mission_control.directives")

_DIRECTIVES_DIR = os.path.join(mc_config.tradingbot_root, "moltbot_knowledge", "directives")
_RESPONSES_DIR = os.path.join(mc_config.tradingbot_root, "moltbot_knowledge", "directive_responses")
_AUDIT_LOG = os.path.join(mc_config.openclaw_root, "logs", "directives.jsonl")


class DirectiveRequest(BaseModel):
    command: str
    type: Optional[str] = None  # EXECUTE, VOTE_INFLUENCE, CONSIDER — auto-classified if omitted

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v is not None and v not in ("EXECUTE", "VOTE_INFLUENCE", "CONSIDER"):
            raise ValueError(f"Invalid directive type: {v}. Must be EXECUTE, VOTE_INFLUENCE, or CONSIDER.")
        return v


class ServiceControlRequest(BaseModel):
    action: str  # start, stop, restart
    service: str = "hydrinia"


@router.get("/directives")
async def get_directives(_=Depends(require_auth)):
    """List active directives."""
    directives = []
    try:
        if os.path.isdir(_DIRECTIVES_DIR):
            for fname in os.listdir(_DIRECTIVES_DIR):
                if fname.endswith(".json") and not fname.startswith("."):
                    fpath = os.path.join(_DIRECTIVES_DIR, fname)
                    try:
                        with open(fpath) as f:
                            directives.append(json.load(f))
                    except Exception:
                        continue
    except Exception as e:
        logger.error(f"Failed to read directives: {e}")
    return {"directives": directives}


@router.post("/directives")
async def issue_directive(req: DirectiveRequest, _=Depends(require_auth)):
    """Issue a new override directive (Hydrinia voting-influence bridge).

    Hydrinia is permanently retired (operator direction, 2026-09-24). This
    dispatch path only does anything when CLIO_HYDRINIA_BRIDGE_ENABLED=1.
    """
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            raise HTTPException(
                status_code=409,
                detail="Hydrinia is retired — override directives have no consumer. "
                       "Set CLIO_HYDRINIA_BRIDGE_ENABLED=1 to re-enable this path.",
            )

        from datetime import datetime, timezone

        from agent.integrations.directive_delivery import dispatch_directive, classify_directive_type
        from agent.override_directive import OverrideEvent

        directive_type = req.type or classify_directive_type(req.command)

        # dispatch_directive() is async and reads override_event.command_text /
        # .timestamp — it needs a real OverrideEvent, not a plain dict (a plain
        # dict here would AttributeError once the bridge is back on). Mission
        # Control's operator auth (require_auth) is the authorization check for
        # this request, so it is authorized by construction.
        override_event = OverrideEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="mission_control",
            user_id="operator",
            is_authorized=True,
            context="legitimate_command",
            message_preview=req.command[:300],
            action_taken="dispatched_from_mission_control",
            command_text=req.command,
        )
        result = await dispatch_directive(override_event, req.command)
        logger.info(f"Directive issued: {directive_type} — {req.command[:100]}")
        return {"status": "dispatched", "type": directive_type, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Directive dispatch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/directives/history")
async def get_directive_history(limit: int = 50, _=Depends(require_auth)):
    """Read directive audit log."""
    entries = []
    try:
        if os.path.isfile(_AUDIT_LOG):
            with open(_AUDIT_LOG) as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    continue
    except Exception as e:
        logger.error(f"Failed to read directive history: {e}")
    return {"history": entries}


@router.get("/directives/responses")
async def get_directive_responses(_=Depends(require_auth)):
    """Read directive compliance responses from Hydrinia."""
    responses = []
    try:
        if os.path.isdir(_RESPONSES_DIR):
            for fname in os.listdir(_RESPONSES_DIR):
                if fname.endswith(".json"):
                    fpath = os.path.join(_RESPONSES_DIR, fname)
                    try:
                        with open(fpath) as f:
                            responses.append(json.load(f))
                    except Exception:
                        continue
    except Exception as e:
        logger.error(f"Failed to read responses: {e}")
    return {"responses": responses}


@router.post("/hydrinia/control")
async def service_control(req: ServiceControlRequest, _=Depends(require_auth)):
    """Start, stop, or restart a service."""
    if req.service not in mc_config.allowed_services:
        raise HTTPException(status_code=400, detail=f"Unknown service: {req.service}")
    if req.service == "hydrinia":
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            raise HTTPException(
                status_code=409,
                detail="Hydrinia is retired — set CLIO_HYDRINIA_BRIDGE_ENABLED=1 to control it.",
            )
    if req.action not in ("start", "stop", "restart"):
        raise HTTPException(status_code=400, detail=f"Invalid action: {req.action}")

    try:
        result = await run_in_executor(lambda: subprocess.run(
            ["sudo", "systemctl", req.action, req.service],
            capture_output=True, text=True, timeout=30,
        ))
        success = result.returncode == 0
        logger.info(f"Service control: {req.action} {req.service} — {'OK' if success else 'FAILED'}")
        return {
            "service": req.service,
            "action": req.action,
            "success": success,
            "output": result.stdout.strip() or result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Service control timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
