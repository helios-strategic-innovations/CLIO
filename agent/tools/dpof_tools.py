"""
OpenClaw/Moltbot — CLIO-native DPOF status tool.

Read-only view of CLIO's own SPY/QQQ long-options engine (agent/dpof/),
exposed to the LLM as get_dpof_status. Prefers the in-process engine
singleton's live text methods (same call pattern the /dpof Telegram command
uses — agent.dpof.engine.get_engine(), safe to call in-process because
DPOFEngine.__init__ has no I/O side effects) and falls back to the on-disk
readmodel (agent/dpof/readmodel.py — sync, read-only, never raises) when the
engine import/call fails for any reason.
"""

import logging

from agent.security import sanitize_ingested_content

logger = logging.getLogger("openclaw.dpof_tools")

_VALID_SECTIONS = ("status", "positions", "decisions", "config")


async def handle_dpof_status(p: dict) -> str:
    """Read-only DPOF status/positions/decisions/config. Never requires confirmation."""
    section = str(p.get("section") or "status").strip().lower()
    if section not in _VALID_SECTIONS:
        section = "status"

    if section == "decisions":
        # No live-engine equivalent exists (status_text/positions_text/config_text
        # only) — the decision log is always read from the ledger via readmodel.
        return sanitize_ingested_content(_decisions_text(), source="dpof_status")

    try:
        from agent.dpof.engine import get_engine

        engine = get_engine()
        if section == "status":
            text = await engine.status_text()
        elif section == "positions":
            text = await engine.positions_text()
        else:  # config
            text = await engine.config_text()
        return sanitize_ingested_content(text, source="dpof_status")
    except Exception as e:
        logger.debug(
            f"DPOF engine unavailable for section={section}, falling back to "
            f"readmodel: {type(e).__name__}: {e}"
        )
        return sanitize_ingested_content(_fallback_text(section), source="dpof_status")


def _decisions_text(limit: int = 20) -> str:
    from agent.dpof.readmodel import recent_decisions

    decisions = recent_decisions(limit=limit)
    if not decisions:
        return "DPOF: no recent decisions logged."
    lines = [f"DPOF recent decisions ({len(decisions)}):"]
    for d in decisions:
        lines.append(f"- {d}")
    return "\n".join(lines)


def _fallback_text(section: str) -> str:
    """readmodel-only rendering, used when the in-process engine is unavailable
    (e.g. a separate process, or the engine hasn't been constructed yet)."""
    from agent.dpof.readmodel import positions_brief, snapshot

    if section == "positions":
        brief = positions_brief()
        return brief or "DPOF: no open positions."
    if section == "config":
        return "DPOF: config unavailable (engine not running in this process)."

    snap = snapshot()
    lines = [
        f"DPOF mode={snap.get('mode', 'unknown')} (from readmodel — engine not running in this process)",
        "halt_file={} hard_halted={}{}".format(
            snap.get("halt_file"),
            snap.get("hard_halted"),
            f" reason={snap.get('halt_reason')}" if snap.get("halt_reason") else "",
        ),
        f"open positions: {snap.get('open_count', 0)}",
        *snap.get("open_lines", []),
        f"realized P&L today: ${snap.get('realized_pnl_today', 0.0):+,.2f} "
        f"({snap.get('closed_today_count', 0)} closed, {snap.get('closed_today_wins', 0)} wins)",
        f"realized P&L week: ${snap.get('realized_pnl_week', 0.0):+,.2f}",
        f"entries today: {snap.get('entries_today', 0)}",
    ]
    return "\n".join(lines)
