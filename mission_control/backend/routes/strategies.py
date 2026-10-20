"""Strategy bank routes — CRUD + push to Hydrinia."""

import json
import logging
import os
import tempfile
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from mission_control.backend.auth import require_auth
from mission_control.backend.deps import read_json, read_text, run_in_executor
from mission_control.backend.config import mc_config

router = APIRouter(prefix="/api/strategies", tags=["strategies"])
logger = logging.getLogger("mission_control.strategies")

_KNOWLEDGE_DIR = os.path.join(mc_config.openclaw_root, "data", "knowledge")
_INDEX_PATH = os.path.join(_KNOWLEDGE_DIR, "knowledge_index.json")
_VAULT_DIR = os.path.join(mc_config.hydrinia_vault, "strategies")
_BRIDGE_STRATEGIES = os.path.join(mc_config.tradingbot_root, "moltbot_knowledge", "strategies")


class StrategyCreate(BaseModel):
    title: str
    content: str
    tags: list[str] = []


class HydriniaPush(BaseModel):
    title: str
    content: str


@router.get("")
async def list_strategies(_=Depends(require_auth)):
    """List Moltbot strategy entries."""
    index = await run_in_executor(lambda: read_json(_INDEX_PATH, lock=True)) or {}
    entries = index.get("entries", [])
    strategies = [e for e in entries if e.get("category") == "trading_strategies"]
    strategies.sort(key=lambda e: e.get("created", ""), reverse=True)
    return {"strategies": strategies, "count": len(strategies)}


@router.get("/hydrinia")
async def list_hydrinia_strategies(_=Depends(require_auth)):
    """List Hydrinia vault strategies."""
    files = []
    for d in [_VAULT_DIR, _BRIDGE_STRATEGIES]:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.endswith((".md", ".json", ".txt")):
                    fpath = os.path.join(d, fname)
                    content = read_text(fpath)
                    files.append({
                        "filename": fname,
                        "source": "vault" if d == _VAULT_DIR else "bridge",
                        "content_preview": content[:300],
                        "size": len(content),
                    })
    return {"strategies": files}


@router.post("")
async def create_strategy(req: StrategyCreate, _=Depends(require_auth)):
    """Add a new strategy to Moltbot's knowledge base."""
    try:
        from agent.tools.knowledge_tools import handle_save_knowledge
        params = {
            "title": req.title,
            "content": req.content,
            "category": "trading_strategies",
            "tags": req.tags,
            "source": "mission_control",
        }
        result = await handle_save_knowledge(params)
        logger.info(f"Strategy created: {req.title}")
        return {"status": "created", "result": result}
    except Exception as e:
        logger.error(f"Strategy creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/hydrinia")
async def push_to_hydrinia(req: HydriniaPush, _=Depends(require_auth)):
    """Push a strategy to Hydrinia's bridge directory.

    Hydrinia is permanently retired (operator direction, 2026-09-24); nothing
    consumes this directory unless the bridge is explicitly re-enabled, so this
    is a no-write 409 by default rather than writing into a dead mailbox.
    """
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            raise HTTPException(
                status_code=409,
                detail="Hydrinia is retired — nothing reads moltbot_knowledge/strategies/. "
                       "Set CLIO_HYDRINIA_BRIDGE_ENABLED=1 to re-enable this path.",
            )
        os.makedirs(_BRIDGE_STRATEGIES, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in req.title)
        fname = f"{safe_title.replace(' ', '_').lower()}.md"
        fpath = os.path.join(_BRIDGE_STRATEGIES, fname)

        # S162 H-1: Use unique temp file to prevent concurrent write collisions
        fd, tmp = tempfile.mkstemp(dir=_BRIDGE_STRATEGIES, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(f"# {req.title}\n\n{req.content}\n")
            os.replace(tmp, fpath)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

        logger.info(f"Strategy pushed to Hydrinia: {fname}")
        return {"status": "pushed", "filename": fname}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hydrinia push failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
