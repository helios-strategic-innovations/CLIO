"""Audit agent routes — on-demand Claude-powered system audit."""

import asyncio
import json
import logging
import subprocess
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from mission_control.backend.auth import require_auth
from agent.anthropic_utils import first_text
from mission_control.backend.deps import run_in_executor

router = APIRouter(prefix="/api/audit", tags=["audit"])
logger = logging.getLogger("mission_control.audit")

# In-memory audit run storage
_audit_runs: dict[str, dict] = {}

_AUDIT_SYSTEM_PROMPT = """You are an expert trading system auditor analyzing CLIO's DPOF engine — a live, autonomous long-options (SPY/QQQ) trading system running inside the CLIO/Moltbot agent on its own Alpaca account. You have access to the system's logs and source code.

AUDIT FRAMEWORK:
1. Identify anomalies: unexpected behavior, error patterns, performance degradation
2. Check signal quality: is the NOFA 4-pillar composite (nofa/premium/darkpool/gex) producing coherent, non-degenerate readings?
3. Validate risk controls: are stop-losses, daily/weekly loss limits, the HALT file, and hard-halt circuit breakers functioning?
4. Detect data quality issues: stale UW/price data, missing fills, phantom positions, mismatched order chains
5. Review P&L attribution: does DPOF's reported realized P&L match actual account changes?

OUTPUT FORMAT:
- 🔴 CRITICAL: Immediate action required (risk to capital)
- 🟡 WARNING: Should be addressed soon
- 🟢 INFO: Observations and patterns
- Provide specific evidence (log lines, timestamps, values)
- End with recommended actions"""


class AuditRequest(BaseModel):
    scope: str = "logs"  # logs, code, full
    focus: Optional[str] = None
    timeframe: str = "4h"  # 1h, 4h, 24h


@router.post("")
async def start_audit(req: AuditRequest, _=Depends(require_auth)):
    """Start an audit run — returns run_id for polling."""
    run_id = str(uuid.uuid4())[:12]
    _audit_runs[run_id] = {
        "status": "running",
        "started": time.time(),
        "scope": req.scope,
        "focus": req.focus,
        "result": None,
    }

    # Launch audit in background
    asyncio.create_task(_run_audit(run_id, req))

    return {"run_id": run_id, "status": "running"}


@router.get("/status")
async def audit_status(run_id: str, _=Depends(require_auth)):
    """Poll audit run status."""
    run = _audit_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Audit run not found")
    return run


@router.get("/history")
async def audit_history(_=Depends(require_auth)):
    """List past audit runs."""
    runs = [
        {"run_id": rid, "status": r["status"], "started": r["started"], "scope": r["scope"]}
        for rid, r in _audit_runs.items()
    ]
    runs.sort(key=lambda r: r["started"], reverse=True)
    return {"runs": runs[:20]}


async def _run_audit(run_id: str, req: AuditRequest):
    """Background audit execution."""
    try:
        # Gather context based on scope
        context_parts = []

        if req.scope in ("logs", "full"):
            timeframe_map = {"1h": 3600, "4h": 14400, "24h": 86400}
            since_seconds = timeframe_map.get(req.timeframe, 14400)
            try:
                result = await run_in_executor(lambda: subprocess.run(
                    ["journalctl", "-u", "moltbot", "--no-pager",
                     f"--since={since_seconds} seconds ago", "-n", "2000"],
                    capture_output=True, text=True, timeout=30,
                ))
                logs = result.stdout[-50000:]  # Cap at 50K chars
                context_parts.append(f"=== MOLTBOT/DPOF LOGS (last {req.timeframe}) ===\n{logs}")
            except Exception as e:
                context_parts.append(f"[Failed to read moltbot logs: {e}]")

        if req.scope in ("code", "full"):
            # DPOF's own read-only state (CLIO-native — Hydrinia's state/*.json is retired)
            try:
                from agent.dpof import readmodel

                snapshot = readmodel.snapshot()
                context_parts.append(f"=== DPOF SNAPSHOT ===\n{json.dumps(snapshot, indent=2, default=str)}")
                decisions = readmodel.recent_decisions(limit=50)
                context_parts.append(f"=== DPOF RECENT DECISIONS (last 50) ===\n{json.dumps(decisions, indent=2, default=str)}")
                risk = readmodel.load_risk()
                context_parts.append(f"=== DPOF RISK STATE ===\n{json.dumps(risk.to_dict(), indent=2, default=str)}")
            except Exception as e:
                context_parts.append(f"[Failed to read DPOF state: {e}]")

        context = "\n\n".join(context_parts)

        # Build prompt
        user_msg = f"Audit this trading system. Scope: {req.scope}."
        if req.focus:
            user_msg += f"\n\nFOCUS AREA: {req.focus}"
        user_msg += f"\n\n{context}"

        # Call Claude
        import anthropic
        from agent.config import config
        client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))
        response = await run_in_executor(
            lambda: client.messages.create(
                model=config.default_model,
                max_tokens=4096,
                system=_AUDIT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
        )

        result_text = first_text(response, default="No response")
        _audit_runs[run_id]["status"] = "complete"
        _audit_runs[run_id]["result"] = result_text
        _audit_runs[run_id]["completed"] = time.time()

    except Exception as e:
        logger.error(f"Audit run {run_id} failed: {e}", exc_info=True)
        _audit_runs[run_id]["status"] = "error"
        _audit_runs[run_id]["result"] = "Audit run failed — check server logs"
