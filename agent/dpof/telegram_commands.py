"""
CLIO DPOF — /dpof Telegram command (I/O).

Plain text only (CLAUDE.md: no parse_mode with dynamic content). Authorized
users only; unauthorized callers get silence, matching every other DPOF-
adjacent command in agent/integrations/telegram_bot.py.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("clio.dpof.telegram_commands")

_MAX_CHUNK_CHARS = 4000

_USAGE = "Usage: /dpof [status|positions|config|why|lessons|stats|kronos|halt [reason]|resume]"


async def dpof_command(update, context) -> None:
    try:
        from agent.integrations.telegram_bot import is_authorized  # lazy
    except Exception as e:
        logger.error(f"DPOF telegram command: is_authorized import failed: {type(e).__name__}: {e}")
        return

    user = getattr(update, "effective_user", None)
    user_id = getattr(user, "id", None)
    if user_id is None or not is_authorized(user_id):
        return

    args = list(getattr(context, "args", None) or [])
    sub = args[0].lower() if args else "status"

    try:
        from agent.dpof.engine import get_engine

        engine = get_engine()

        if sub in ("", "status"):
            text = await engine.status_text()
        elif sub == "positions":
            text = await engine.positions_text()
        elif sub == "config":
            text = await engine.config_text()
        elif sub == "why":
            # Read-only, filesystem-only (agent.dpof.readmodel) -- never
            # touches the live engine/broker, same as Mission Control could
            # call it directly (Task ORCH-7Q4K-DPOF observability).
            from agent.dpof import readmodel

            import asyncio

            text = await asyncio.get_running_loop().run_in_executor(None, readmodel.dpof_why_text)
        elif sub == "lessons":
            # Same read-only/off-loop contract as "why" (Task ORCH-7Q4K-DPOF
            # builder B2, 2026-09-26 -- learning loop F1).
            from agent.dpof import readmodel

            import asyncio

            text = await asyncio.get_running_loop().run_in_executor(None, readmodel.lessons_text)
        elif sub == "stats":
            # Same read-only/off-loop contract as "why"/"lessons" (Task
            # ORCH-7Q4K-DPOF builder B2, docs/design/
            # clio_dpof_followups_2026-09-26.md §4, 2026-09-26 -- F2).
            from agent.dpof import readmodel

            import asyncio

            text = await asyncio.get_running_loop().run_in_executor(None, readmodel.trade_stats_text)
        elif sub == "kronos":
            # Same read-only/off-loop contract as "why"/"lessons"/"stats"
            # (Task K2, agent/kronos -- a separate CLIO service).
            from agent.dpof import readmodel

            import asyncio

            text = await asyncio.get_running_loop().run_in_executor(None, readmodel.kronos_text)
        elif sub == "halt":
            reason = " ".join(args[1:]).strip() or "operator"
            text = await engine.operator_halt(reason)
        elif sub == "resume":
            text = await engine.operator_resume()
        else:
            text = _USAGE
    except Exception as e:
        logger.error(f"DPOF telegram command '{sub}' failed: {type(e).__name__}: {e}", exc_info=True)
        text = f"DPOF command failed: {type(e).__name__}: {e}"

    await _reply_chunks(update, text)


async def _reply_chunks(update, text: str) -> None:
    message = getattr(update, "message", None)
    if message is None:
        return
    remaining = text or "(empty)"
    while remaining:
        if len(remaining) <= _MAX_CHUNK_CHARS:
            await message.reply_text(remaining)
            break
        split_at = remaining.rfind("\n", 0, _MAX_CHUNK_CHARS)
        if split_at <= 0:
            split_at = _MAX_CHUNK_CHARS
        await message.reply_text(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
