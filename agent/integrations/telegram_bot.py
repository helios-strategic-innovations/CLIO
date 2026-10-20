"""
OpenClaw/Moltbot Telegram Bot
"""

import asyncio
import json
import time
import uuid
import logging
from collections import deque

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters

from agent.config import config
from agent.core import agent_respond, classify_task, conversation_manager
from agent.security import log_security_event

logger = logging.getLogger("openclaw.telegram")
pending_confirmations: dict[str, dict] = {}
_confirmation_lock = asyncio.Lock()  # S153: serialize concurrent confirmation access
bot_app: Application | None = None
_shutdown_cleanup_completed = False

# S182 L3: Telegram ReadError rate tracker — downgrade transients to DEBUG,
# but alert on storms (>5 errors per rolling 60s window).
# S170B P1-7: maxlen 100 → 500 so massive storms (>1.6/sec) don't cap the count
# and miss the threshold; 500 covers ~8 min of 1-err-per-sec without rolloff.
_telegram_read_error_times: deque[float] = deque(maxlen=500)


def shutdown_cleanup_completed() -> bool:
    """Return whether post_shutdown attempted every cleanup stage."""
    return _shutdown_cleanup_completed


async def _shutdown_default_executor(timeout_s: float = 5.0) -> None:
    """Bound default-executor join so shutdown cannot wait until systemd SIGKILL."""
    loop = asyncio.get_running_loop()
    shutdown = getattr(loop, "shutdown_default_executor", None)
    if shutdown is None:
        logger.info("Default executor shutdown skipped: unsupported event loop")
        return
    try:
        await shutdown(timeout=timeout_s)
    except TypeError:
        await asyncio.wait_for(shutdown(), timeout=timeout_s)


def is_authorized(user_id: int) -> bool:
    return user_id in config.telegram_authorized_users


async def cmd_start(update: Update, context):
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        log_security_event("unauthorized_telegram", user_id=update.effective_user.id)
        return

    # Hydrinia is permanently off (operator direction, 2026-09-24) — its
    # shortcut commands are only listed when the bridge is explicitly re-enabled.
    try:
        from agent.hydrinia_bridge import bridge_enabled
        _hydrinia_on = bridge_enabled()
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, excluding hydrinia help lines: {e}")
        _hydrinia_on = False

    trading_lines = "/portfolio — Current positions (Alpaca + DPOF)\n"
    if _hydrinia_on:
        trading_lines = "/hydrinia — Hydrinia bot status\n/logs — Recent bot logs\n" + trading_lines

    system_lines = "/status — System status\n/codehealth — Code health report\n"
    if _hydrinia_on:
        system_lines += "/stop — Stop Hydrinia\n/botstart — Start Hydrinia\n"
    system_lines += "/clear — Clear conversation\n\n"

    await update.message.reply_text(
        "CLIO Online\n\n"
        "Modes:\n"
        "/trading — Market intel & trading\n"
        "/academic — Coursework & homework\n"
        "/assistant — Email, calendar, reminders\n"
        "/career — Job search & networking\n\n"
        "Trading:\n"
        f"{trading_lines}"
        "/directive — Override directives\n"
        "/worldbrief — Global intelligence brief\n"
        "/dpof — DPOF long-options engine status\n\n"
        "System:\n"
        f"{system_lines}"
        "Personality: /personality, /trait_propose, "
        "/trait_confirm, /trait_deny\n\n"
        "Send text or voice messages.",
    )


async def cmd_status(update: Update, context):
    if not is_authorized(update.effective_user.id):
        return

    # Build API budget section
    budget_lines = []
    try:
        from agent.rate_limiter import budget_tracker
        budgets = await budget_tracker.get_status()
        for name, info in budgets.items():
            remaining_pct = info["remaining"] / info["max"] if info["max"] else 0
            if remaining_pct > 0.2:
                indicator = "🟢"
            elif remaining_pct > 0:
                indicator = "🟡"
            else:
                indicator = "🔴"
            label = name.replace("_", " ").title()
            budget_lines.append(
                f"{indicator} {label}: {info['used']}/{info['max']} ({info['period']})"
            )
    except Exception as e:
        logger.debug(f"Budget data load failed: {e}")
        budget_lines.append("Could not load budget data")

    budget_section = "\n".join(budget_lines)

    await update.message.reply_text(
        "Moltbot Status\n\n"
        "Agent Core: Running\n"
        "Telegram: Connected\n"
        "News Intelligence: Active\n"
        f"Alpaca: Connected ({'paper trading' if config.alpaca_paper else 'live trading'})\n"
        f"{'Email: ' + str(len(config.email_accounts)) + ' account(s)' if config.email_accounts else 'Email: Not configured'}\n\n"
        f"API Budgets:\n{budget_section}\n\n"
        f"Conversation manager: SQLite (persistent)",
    )


async def cmd_clear(update: Update, context):
    if not is_authorized(update.effective_user.id):
        return
    await conversation_manager.clear(str(update.effective_chat.id))
    await update.message.reply_text("🧹 History cleared.")


async def cmd_trading(update: Update, context):
    if not is_authorized(update.effective_user.id):
        return
    await update.message.reply_text("Trading Mode — Ask about markets, news, flow, portfolio, or DPOF.")


async def cmd_academic(update: Update, context):
    if not is_authorized(update.effective_user.id):
        return
    await update.message.reply_text("Academic Mode — Ask about assignments, readings, homework, or exams.")


async def cmd_assistant(update: Update, context):
    if not is_authorized(update.effective_user.id):
        return
    await update.message.reply_text("Assistant Mode — Email, calendar, reminders.")


async def cmd_career(update: Update, context):
    if not is_authorized(update.effective_user.id):
        return
    await update.message.reply_text("Career Mode — Jobs, alumni search, applications.")


async def handle_message(update: Update, context):
    user_id = update.effective_user.id
    logger.info(f"S163-B: Message received from {user_id}: {(update.message.text or '')[:50]!r}")
    if not is_authorized(user_id):
        log_security_event("unauthorized_message", user_id=user_id)
        return

    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    await update.effective_chat.send_action("typing")
    task_type = classify_task(user_text)

    async def request_confirmation(tool_name: str, tool_input: dict) -> bool:
        return await send_confirmation_request(update.effective_chat.id, tool_name, tool_input)

    try:
        # S163-B: Outer timeout on entire agent_respond — last resort if SDK timeout fails
        response = await asyncio.wait_for(agent_respond(
            user_message=user_text, conversation_id=chat_id,
            task_type=task_type, is_auto_mode=False, confirmation_callback=request_confirmation,
            source="telegram_direct", user_id=str(user_id),
        ), timeout=180)  # 3 min max for full response including tool loops
        if len(response) > 4000:
            for i in range(0, len(response), 4000):
                await update.message.reply_text(response[i:i+4000])
        else:
            await update.message.reply_text(response)
    except asyncio.TimeoutError:
        logger.error("S163-B: agent_respond timed out after 180s")
        await update.message.reply_text("⚠️ Request timed out (180s). Please try again with a simpler question.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Error: {str(e)[:200]}")


async def send_confirmation_request(chat_id: int, tool_name: str, tool_input: dict) -> bool:
    # S170B P3-6: explicit init-order guard. bot_app is module-global set by
    # create_bot(); without this guard a bad startup order yields AttributeError
    # deep inside the function (line 189). Fail fast with a clear message.
    if bot_app is None:
        logger.error("send_confirmation_request called before create_bot() — bot_app is None")
        return False
    callback_id = str(uuid.uuid4())[:8]
    details = json.dumps(tool_input, indent=2, default=str)
    if len(details) > 500:
        details = details[:500] + "..."

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Approve", callback_data=f"confirm_{callback_id}"),
         InlineKeyboardButton("❌ Deny", callback_data=f"deny_{callback_id}")]
    ])

    event = asyncio.Event()
    import time as _time
    async with _confirmation_lock:  # S153: serialize concurrent confirmation access
        pending_confirmations[callback_id] = {"event": event, "result": False, "created_at": _time.time()}
    logger.info(f"Confirmation {callback_id} created for tool={tool_name}")

    try:
        msg = await bot_app.bot.send_message(
            chat_id=chat_id,
            text=f"⚠️ Confirmation Required\n\nTool: {tool_name}\n\n{details}\n\nApprove or Deny?",
            reply_markup=keyboard,
        )
    except Exception as e:
        logger.error(f"Failed to send confirmation message for {callback_id}: {e}", exc_info=True)
        pending_confirmations.pop(callback_id, None)
        return False

    logger.info(f"Confirmation {callback_id} sent, awaiting response (120s timeout)")

    try:
        await asyncio.wait_for(event.wait(), timeout=120.0)
        result = pending_confirmations[callback_id]["result"]
        logger.info(f"Confirmation {callback_id} resolved: {'approved' if result else 'denied'}")
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Confirmation {callback_id} TIMED OUT (120s)")
        try:
            await msg.edit_text(f"⏰ Timed out. Cancelled.\n\nTool: {tool_name}")
        except Exception as e:
            logger.debug(f"Fire-and-forget timeout edit failed: {e}")
            try:
                await bot_app.bot.send_message(chat_id=chat_id, text="⏰ Timed out. Cancelled.")
            except Exception as e2:
                logger.debug(f"Fire-and-forget timeout fallback send failed: {e2}")
        return False
    finally:
        pending_confirmations.pop(callback_id, None)


async def handle_confirmation_callback(update: Update, context):
    # Sweep stale confirmations (older than 5 min) — defense against leaked entries
    import time as _time
    _stale = [k for k, v in pending_confirmations.items()
              if v.get("created_at", 0) < _time.time() - 300]
    for k in _stale:
        pending_confirmations.pop(k, None)

    query = update.callback_query
    data = query.data or ""

    # Acknowledge callback query (non-critical — must not block event setting)
    try:
        await query.answer()
    except Exception as e:
        logger.warning(f"Failed to answer callback query: {e}")

    if not is_authorized(query.from_user.id):
        logger.warning(f"Unauthorized confirmation callback from user {query.from_user.id}")
        return

    # Parse action and callback ID
    cid = None
    approved = None
    if data.startswith("confirm_"):
        cid = data[len("confirm_"):]
        approved = True
    elif data.startswith("deny_"):
        cid = data[len("deny_"):]
        approved = False

    if cid is None:
        logger.warning(f"Unknown callback data format: {data}")
        return

    logger.info(f"Confirmation callback: cid={cid}, approved={approved}, pending_keys={list(pending_confirmations.keys())}")

    if cid not in pending_confirmations:
        logger.warning(f"Confirmation {cid} not in pending (expired or already handled)")
        try:
            await query.edit_message_text("⏰ This confirmation has expired.")
        except Exception as e:
            logger.debug(f"Fire-and-forget expired confirmation edit failed: {e}")
        return

    # CRITICAL: Set event first — this unblocks send_confirmation_request
    try:
        pending_confirmations[cid]["result"] = approved
        pending_confirmations[cid]["event"].set()
        logger.info(f"Confirmation {cid} event SET — {'APPROVED' if approved else 'DENIED'}")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to set confirmation event {cid}: {e}", exc_info=True)
        return

    # Update message text (non-critical cosmetic update)
    try:
        if approved:
            await query.edit_message_text("✅ Approved. Executing...")
        else:
            await query.edit_message_text("❌ Denied. Cancelled.")
    except Exception as e:
        logger.warning(f"Failed to edit confirmation message {cid}: {e}")


async def handle_voice(update: Update, context):
    """Handle voice messages — transcribe via Whisper then process as text."""
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        log_security_event("unauthorized_voice", user_id=user_id)
        return

    await update.effective_chat.send_action("typing")
    await update.message.reply_text("🎙️ Transcribing...")

    try:
        voice = update.message.voice or update.message.audio
        if not voice:
            await update.message.reply_text("No audio found.")
            return

        file = await voice.get_file()
        file_bytes = await file.download_as_bytearray()

        from agent.integrations.voice import transcribe_voice
        text = await transcribe_voice(bytes(file_bytes))

        if text.startswith("ERROR:"):
            await update.message.reply_text(f"⚠️ {text}")
            return

        await update.message.reply_text(f"📝 {text}")

        # Process transcribed text through agent
        chat_id = str(update.effective_chat.id)
        task_type = classify_task(text)

        async def request_confirmation(tool_name: str, tool_input: dict) -> bool:
            return await send_confirmation_request(update.effective_chat.id, tool_name, tool_input)

        # S233 HI-3: Timeout wrapper matching handle_message pattern (180s)
        response = await asyncio.wait_for(agent_respond(
            user_message=text, conversation_id=chat_id,
            task_type=task_type, is_auto_mode=False, confirmation_callback=request_confirmation,
            voice_mode=True, source="telegram_direct", user_id=str(user_id),
        ), timeout=180)

        if len(response) > 4000:
            for i in range(0, len(response), 4000):
                await update.message.reply_text(response[i:i+4000])
        else:
            await update.message.reply_text(response)

        # Voice reply: synthesize TTS if ElevenLabs configured
        if config.elevenlabs_api_key and update.message.voice:
            try:
                from agent.integrations.tts import synthesize_speech, strip_markdown
                import io
                audio_bytes = await synthesize_speech(strip_markdown(response))
                await update.message.reply_voice(voice=io.BytesIO(audio_bytes))
            except Exception as e:
                logger.warning(f"TTS voice reply failed (text already sent): {e}")

    except asyncio.TimeoutError:
        logger.error("Voice handler: agent_respond timed out after 180s")
        await update.message.reply_text("⏱️ Sorry, the voice response timed out. Please try again.")
    except Exception as e:
        logger.error(f"Voice error: {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Voice processing error: {str(e)[:200]}")


async def _shortcut_command(update: Update, prompt: str, task_type: str = "trade_decision"):
    """Helper for shortcut commands that route through agent_respond."""
    if not is_authorized(update.effective_user.id):
        return
    chat_id = str(update.effective_chat.id)
    await update.effective_chat.send_action("typing")

    async def request_confirmation(tool_name: str, tool_input: dict) -> bool:
        return await send_confirmation_request(update.effective_chat.id, tool_name, tool_input)

    try:
        # S233-V2: Timeout wrapper matching handle_message/handle_voice pattern (180s)
        response = await asyncio.wait_for(agent_respond(
            user_message=prompt, conversation_id=chat_id,
            task_type=task_type, is_auto_mode=False, confirmation_callback=request_confirmation,
            source="telegram_direct", user_id=str(update.effective_user.id),
        ), timeout=180)
        if len(response) > 4000:
            for i in range(0, len(response), 4000):
                await update.message.reply_text(response[i:i+4000])
        else:
            await update.message.reply_text(response)
    except asyncio.TimeoutError:
        logger.error("Shortcut command timed out after 180s")
        await update.message.reply_text("Request timed out (180s). Please try again.")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)[:200]}")


async def cmd_hydrinia(update: Update, context):
    await _shortcut_command(update, "Check the Hydrinia trading bot status. Is it running? What's the PID and uptime?")


async def cmd_logs(update: Update, context):
    await _shortcut_command(update, "Show me the last 30 important Hydrinia log lines (WARNING and ERROR level).")


async def cmd_portfolio(update: Update, context):
    await _shortcut_command(
        update,
        "Show CLIO's Alpaca paper portfolio and the DPOF book (open positions, today's realized P&L).",
    )


async def cmd_stop(update: Update, context):
    await _shortcut_command(update, "Stop the Hydrinia trading bot gracefully.")


async def cmd_botstart(update: Update, context):
    await _shortcut_command(update, "Start the Hydrinia trading bot.")


async def cmd_directive(update: Update, context):
    """Show active operator directives."""
    if not is_authorized(update.effective_user.id):
        return
    try:
        from agent.integrations.directive_delivery import get_active_directives
        directives = get_active_directives()
        if not directives:
            await update.message.reply_text("No active directives.")
            return

        lines = [f"Active Directives ({len(directives)}):"]
        for d in directives:
            d_id = d.get("id", "?")
            d_type = d.get("type", "?")
            symbol = d.get("symbol") or "global"
            status = d.get("status", "?")
            command = d.get("command", "")[:80]
            expires = d.get("expires_at", "")[:19]
            lines.append(
                f"\n  {d_id}\n"
                f"  Type: {d_type} | Symbol: {symbol} | Status: {status}\n"
                f"  Expires: {expires}\n"
                f"  Command: {command}{'...' if len(d.get('command', '')) > 80 else ''}"
            )

        message = "\n".join(lines)
        if len(message) > 4000:
            message = message[:3997] + "..."
        await update.message.reply_text(message)
    except Exception as e:
        await update.message.reply_text(f"Error loading directives: {str(e)[:200]}")


async def _post_shutdown(application) -> None:
    """Clean up background tasks on graceful shutdown."""
    global _shutdown_cleanup_completed
    _shutdown_cleanup_completed = False
    logger.info("CLIO shutdown stage ENTER: post_shutdown")
    _background_ok = False
    try:
        logger.info("CLIO shutdown stage ENTER: shutdown_background_tasks")
        from agent.integrations.scheduled_tasks import shutdown_background_tasks
        await shutdown_background_tasks()
        logger.info("Background tasks shut down cleanly")
        _background_ok = True
    except Exception as e:
        logger.warning(f"Error during background task shutdown: {e}")
    finally:
        logger.info(
            "CLIO shutdown stage EXIT: shutdown_background_tasks status=%s",
            "ok" if _background_ok else "error",
        )
    _memory_ok = False
    try:
        logger.info("CLIO shutdown stage ENTER: memory_flush")
        from agent.memory.memory_engine import get_memory_engine
        me = get_memory_engine()
        me.flush()
        logger.info("Memory engine flushed on shutdown")
        _memory_ok = True
    except Exception as e:
        logger.debug(f"Memory flush on shutdown failed (non-critical): {e}")
    finally:
        logger.info(
            "CLIO shutdown stage EXIT: memory_flush status=%s",
            "ok" if _memory_ok else "error",
        )
    _conversation_ok = False
    try:
        logger.info("CLIO shutdown stage ENTER: conversation_persistence_close")
        await conversation_manager.close()
        logger.info("Conversation persistence closed on shutdown")
        _conversation_ok = True
    except Exception as e:
        logger.warning("Conversation persistence close failed: %s", str(e)[:200])
    finally:
        logger.info(
            "CLIO shutdown stage EXIT: conversation_persistence_close status=%s",
            "ok" if _conversation_ok else "error",
        )
    _search_ok = False
    try:
        logger.info("CLIO shutdown stage ENTER: web_search_executor_shutdown")
        from agent.integrations.web_search import shutdown_search_executors
        shutdown_search_executors(wait=False)
        _search_ok = True
    except Exception as e:
        logger.warning("Web search executor shutdown failed: %s", str(e)[:200])
    finally:
        logger.info(
            "CLIO shutdown stage EXIT: web_search_executor_shutdown status=%s",
            "ok" if _search_ok else "error",
        )
    _executor_ok = False
    try:
        logger.info("CLIO shutdown stage ENTER: default_executor_shutdown")
        await _shutdown_default_executor(timeout_s=5.0)
        _executor_ok = True
    except Exception as e:
        logger.warning("Default executor shutdown failed: %s", str(e)[:200])
    finally:
        logger.info(
            "CLIO shutdown stage EXIT: default_executor_shutdown status=%s",
            "ok" if _executor_ok else "error",
        )
        _shutdown_cleanup_completed = True
        logger.info("CLIO shutdown cleanup completed flag set")
        logger.info("CLIO shutdown stage EXIT: post_shutdown")


def create_bot() -> Application:
    global bot_app
    app = (
        Application.builder()
        .token(config.telegram_bot_token)
        .concurrent_updates(True)
        .post_shutdown(_post_shutdown)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("trading", cmd_trading))
    app.add_handler(CommandHandler("academic", cmd_academic))
    app.add_handler(CommandHandler("assistant", cmd_assistant))
    app.add_handler(CommandHandler("career", cmd_career))
    app.add_handler(CommandHandler("portfolio", cmd_portfolio))
    app.add_handler(CommandHandler("directive", cmd_directive))
    # Hydrinia shortcut commands — Hydrinia is permanently off (operator
    # direction, 2026-09-24); only registered when the bridge is explicitly
    # re-enabled (matches the /start help text gating above).
    try:
        from agent.hydrinia_bridge import bridge_enabled
        _hydrinia_bridge_on = bridge_enabled()
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, excluding hydrinia commands: {e}")
        _hydrinia_bridge_on = False
    if _hydrinia_bridge_on:
        app.add_handler(CommandHandler("hydrinia", cmd_hydrinia))
        app.add_handler(CommandHandler("logs", cmd_logs))
        app.add_handler(CommandHandler("stop", cmd_stop))
        app.add_handler(CommandHandler("botstart", cmd_botstart))
        logger.info("Hydrinia bridge commands registered: ['hydrinia', 'logs', 'stop', 'botstart']")

    # Personality commands (graceful — doesn't crash if module fails)
    try:
        from agent.personality.engine import get_personality_engine
        from agent.personality.telegram_commands import PersonalityTelegramCommands
        _pe = get_personality_engine()
        _personality_cmds = PersonalityTelegramCommands(
            registry=_pe.registry, engine=_pe, config=_pe.config,
        )
        for _cmd_name, _handler in _personality_cmds.get_command_map().items():
            app.add_handler(CommandHandler(_cmd_name, _handler))
        logger.info(f"Personality commands registered: {list(_personality_cmds.get_command_map().keys())}")
    except Exception as e:
        logger.error(f"Failed to register personality commands: {e}", exc_info=True)

    # Moltbook commands (graceful — doesn't crash if module fails)
    try:
        from agent.moltbook.config import get_moltbook_config
        from agent.moltbook.kill_switch import get_kill_switch
        from agent.moltbook.quarantine import get_quarantine_manager
        from agent.moltbook.audit import AuditReporter
        from agent.moltbook.telegram_commands import MoltbookTelegramCommands
        _mb_config = get_moltbook_config()
        _moltbook_cmds = MoltbookTelegramCommands(
            heartbeat=None,  # Heartbeat starts later via scheduler; status handles None gracefully
            quarantine_manager=get_quarantine_manager(),
            kill_switch=get_kill_switch(),
            audit_reporter=AuditReporter(),
            config=_mb_config,
        )
        for _cmd_name, _handler in _moltbook_cmds.get_command_map().items():
            app.add_handler(CommandHandler(_cmd_name, _handler))
        logger.info(f"Moltbook commands registered: {list(_moltbook_cmds.get_command_map().keys())}")
    except Exception as e:
        logger.error(f"Failed to register moltbook commands: {e}", exc_info=True)

    # News Intelligence commands
    try:
        from agent.integrations.news_intel import get_worldbrief

        async def _worldbrief_handler(update, context):
            """Show last 5 intelligence items."""
            if not is_authorized(update.effective_user.id):
                return
            brief = get_worldbrief()
            await update.message.reply_text(brief)

        app.add_handler(CommandHandler("worldbrief", _worldbrief_handler))
        logger.info("News intelligence commands registered: ['worldbrief']")
    except Exception as e:
        logger.error(f"Failed to register news intel commands: {e}", exc_info=True)

    # Code Sentinel — on-demand code health check (Operation FORGE)
    try:
        from agent.integrations.code_sentinel import handle_codehealth_command, SENTINEL_ENABLED

        async def _cmd_codehealth(update, context):
            """Run code health scan and report findings."""
            if not is_authorized(update.effective_user.id):
                return
            try:
                args = context.args
                scan_type = args[0].lower() if args else "changed"
                if scan_type not in ("changed", "full"):
                    scan_type = "changed"
                report = await handle_codehealth_command(scan_type)
                # Send in chunks to avoid Telegram 4096-char limit
                remaining = report
                while remaining:
                    if len(remaining) <= 4000:
                        await update.message.reply_text(remaining)
                        break
                    # Find a newline near the 4000 boundary to split cleanly
                    split_at = remaining.rfind("\n", 0, 4000)
                    if split_at == -1:
                        split_at = 4000
                    await update.message.reply_text(remaining[:split_at])
                    remaining = remaining[split_at:].lstrip("\n")
            except Exception as e:
                await update.message.reply_text(f"Code health check failed: {e}")

        if SENTINEL_ENABLED():
            app.add_handler(CommandHandler("codehealth", _cmd_codehealth))
            logger.info("Code Sentinel command registered: ['codehealth']")
        else:
            logger.info("Code Sentinel disabled — /codehealth not registered")
    except Exception as e:
        logger.error(f"Failed to register Code Sentinel command: {e}", exc_info=True)

    # CLIO DPOF long-options engine — /dpof [status|positions|config|why|halt|resume]
    try:
        from agent.dpof.telegram_commands import dpof_command

        app.add_handler(CommandHandler("dpof", dpof_command))
        logger.info("DPOF commands registered: ['dpof']")
    except Exception as e:
        logger.error(f"Failed to register DPOF command: {e}", exc_info=True)

    # Atlas Autonomy Bridge — CLIO Telegram approval channel (Phase 4, Gate G-B).
    # Wires /approve + /bridge_pending into THIS Application (the existing
    # getUpdates poller — no second poller, no second bot). Handlers are inert
    # no-ops unless BRIDGE_APPROVAL_ENABLED=1, so registration changes no live
    # behavior until the operator opts in. T3 stays desk-only inside the module.
    try:
        from agent.integrations.bridge_approval import (
            get_command_map as _bridge_cmd_map,
            BRIDGE_APPROVAL_ENABLED as _bridge_enabled,
        )
        for _cmd_name, _handler in _bridge_cmd_map().items():
            app.add_handler(CommandHandler(_cmd_name, _handler))
        logger.info(
            "Bridge approval commands registered: %s (enabled=%s)",
            list(_bridge_cmd_map().keys()), _bridge_enabled(),
        )
    except Exception as e:
        logger.error(f"Failed to register bridge approval commands: {e}", exc_info=True)

    # Atlas Autonomy Bridge — proactive pending-request push (M6, default-disabled).
    # Schedules an idempotent JobQueue poll on THIS Application (no second poller) that
    # pings the operator once per new pending request. No-op unless BOTH
    # BRIDGE_APPROVAL_ENABLED and BRIDGE_APPROVAL_PUSH_ENABLED are set, and it never
    # raises into create_bot() (JobQueue-absent or any error is logged + skipped).
    try:
        from agent.integrations.bridge_approval import schedule_proactive_push as _bridge_sched_push
        _bridge_sched_push(app)
    except Exception as e:
        logger.error(f"Failed to schedule bridge proactive push: {e}", exc_info=True)

    # Callbacks and message handlers
    app.add_handler(CallbackQueryHandler(handle_confirmation_callback))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Global error handler
    async def _error_handler(update, context):
        # S182 L3: Downgrade benign transient network errors (ReadError, etc.) to
        # DEBUG — PTB's network_retry_loop recovers automatically. Keep ERROR for
        # real issues. Storm detection: if >5/60s, re-promote to WARNING.
        err = context.error
        err_name = type(err).__name__ if err else ""
        _TRANSIENT = ("ReadError", "ConnectError", "ConnectTimeout", "ReadTimeout",
                      "WriteTimeout", "PoolTimeout", "NetworkError")
        if any(t in err_name for t in _TRANSIENT):
            now = time.time()
            _telegram_read_error_times.append(now)
            # Count errors in last 60s
            recent = sum(1 for t in _telegram_read_error_times if now - t <= 60)
            if recent > 5:
                logger.warning(
                    f"Telegram transient-error storm: {recent}/60s ({err_name}). "
                    f"Possible upstream Telegram API issue."
                )
            else:
                logger.debug(f"Telegram transient {err_name} (PTB will retry): {err}")
            return
        logger.error(f"Telegram handler error: {context.error}", exc_info=context.error)

    app.add_error_handler(_error_handler)

    # Scheduled monitoring tasks
    from agent.integrations.scheduled_tasks import register_scheduled_jobs
    register_scheduled_jobs(app)

    bot_app = app
    return app
