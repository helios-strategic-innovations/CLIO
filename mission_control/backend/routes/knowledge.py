"""Knowledge routes — learning summaries, living perspective, priors."""

import logging
import os

from fastapi import APIRouter, Depends, Query
from typing import Optional

from mission_control.backend.auth import require_auth
from mission_control.backend.deps import read_json, read_text, run_in_executor
from mission_control.backend.config import mc_config

router = APIRouter(prefix="/api", tags=["knowledge"])
logger = logging.getLogger("mission_control.knowledge")

_KNOWLEDGE_DIR = os.path.join(mc_config.openclaw_root, "data", "knowledge")
_INDEX_PATH = os.path.join(_KNOWLEDGE_DIR, "knowledge_index.json")
_PERSPECTIVE_PATH = os.path.join(_KNOWLEDGE_DIR, "living_perspective.md")
_PRIORS_PATH = os.path.join(_KNOWLEDGE_DIR, "current_priors.md")
_LEARNING_STATE_PATH = os.path.join(_KNOWLEDGE_DIR, "learning_state.json")


@router.get("/knowledge")
async def list_knowledge(
    category: Optional[str] = None,
    limit: int = Query(30, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _=Depends(require_auth),
):
    """List knowledge entries, optionally filtered by category."""
    index = await run_in_executor(lambda: read_json(_INDEX_PATH, lock=True))
    if not index:
        return {"entries": [], "total": 0}
    entries = index.get("entries", [])
    if not isinstance(entries, list):
        return {"entries": [], "total": 0}

    if category:
        entries = [e for e in entries if e.get("category") == category]

    # Sort by created date descending
    entries.sort(key=lambda e: e.get("created", ""), reverse=True)
    total = len(entries)
    page = entries[offset:offset + limit]

    return {"entries": page, "total": total, "offset": offset, "limit": limit}


@router.get("/knowledge/summary")
async def get_living_perspective(_=Depends(require_auth)):
    """Living perspective — synthesized macro view."""
    text = read_text(_PERSPECTIVE_PATH)
    return {"content": text, "length": len(text)}


@router.get("/knowledge/priors")
async def get_current_priors(_=Depends(require_auth)):
    """Current operational priors."""
    text = read_text(_PRIORS_PATH)
    return {"content": text, "length": len(text)}


@router.get("/knowledge/stats")
async def get_learning_stats(_=Depends(require_auth)):
    """Learning pipeline statistics."""
    state = await run_in_executor(lambda: read_json(_LEARNING_STATE_PATH, lock=True)) or {}
    return {
        "total_cycles": state.get("total_cycles", 0),
        "last_cycle": state.get("last_cycle_time"),
        "save_rate": state.get("save_rate"),
        "category_weights": state.get("category_weights", {}),
        "last_queries": state.get("last_queries", [])[-10:],
    }


@router.get("/knowledge/{entry_id}")
async def get_knowledge_entry(entry_id: str, _=Depends(require_auth)):
    """Get full content of a knowledge entry."""
    index = await run_in_executor(lambda: read_json(_INDEX_PATH, lock=True)) or {}
    entries = index.get("entries", [])
    entry = next((e for e in entries if e.get("id") == entry_id), None)
    if not entry:
        return {"error": "Entry not found"}

    # S162 M-12: Use file_path (not filename) — actual field in knowledge index entries
    file_rel = entry.get("file_path", "") or entry.get("filename", "")
    category = entry.get("category", "")
    # S162 M-12: Validate BEFORE path construction (defense-in-depth)
    if not file_rel or ".." in file_rel or "\\" in file_rel:
        return {"error": "Invalid file path in knowledge entry"}
    content_path = os.path.join(_KNOWLEDGE_DIR, file_rel) if file_rel else ""
    # S153: Validate path stays within knowledge directory (prevent traversal via malicious index entries)
    if content_path:
        from pathlib import Path
        _resolved = Path(content_path).resolve()
        if not str(_resolved).startswith(str(Path(_KNOWLEDGE_DIR).resolve())):
            return {"error": "Path outside knowledge directory"}
    content = ""
    if content_path and os.path.isfile(content_path):
        try:
            with open(content_path) as f:
                data = __import__("json").load(f)
                content = data.get("content", "")
        except Exception:
            content = read_text(content_path)

    return {**entry, "content": content}
