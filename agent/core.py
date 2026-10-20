"""
OpenClaw/Moltbot Core Agent Loop
"""

import asyncio
import json
import logging
import os
import re
import threading
from typing import Any, Callable, Optional

import anthropic

from agent.anthropic_utils import first_text
from agent.config import config
from agent.security import sanitize_user_input, validate_tool_call, CONFIRMATION_REQUIRED
from agent.prompts.system_prompt import get_system_prompt, get_model_for_task
from agent.tools.definitions import get_all_tools
from agent.tools.executor import execute_tool

logger = logging.getLogger("openclaw.core")

# S163-B: Timeout for Anthropic API calls via run_in_executor.
# Prevents hung API calls from freezing the event loop indefinitely.
# Normal response: 5-30s. Complex tool use: up to 60s. 120s is generous.
_API_TIMEOUT = 120

# S214: appended as a text block to the forced-summary user turn. The gateway
# ignores tool_choice=none, so the instruction has to be in the message itself.
_FORCED_SUMMARY_INSTRUCTION = (
    "You have used all available tool calls. Do not call any more tools. "
    "Using only what you already have, write the final answer to the user now in plain text."
)


def _log_fire_and_forget_exception(fut: "asyncio.Future", label: str) -> None:
    """Done-callback for un-awaited run_in_executor futures (episodic.record /
    learn_from_outcome). Without this, an exception raised inside the executor
    thread is only ever surfaced via asyncio's default "Future exception was
    never retrieved" GC-time warning — never reaching this module's own
    try/except-at-call-site logging (that only catches pre-schedule failures).
    Fire-and-forget by design: this only logs, it never re-raises or blocks."""
    try:
        exc = fut.exception()
        if exc is not None:
            logger.warning("Fire-and-forget %s failed: %s: %s", label, type(exc).__name__, exc)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass

VOICE_MODE_INSTRUCTION = (
    "You are on a live phone call. Keep responses under 3 sentences unless asked "
    "for detail. No markdown, no bullet points, no tables, no code blocks. "
    "Speak naturally — your text is being converted to audio. "
    "IMPORTANT: Respond directly from your knowledge and context. Do NOT use tools "
    "like recall_knowledge or search_web unless the user explicitly asks you to look "
    "something up. Every tool call adds seconds of latency on a live call."
)

_cached_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()

def _get_client() -> anthropic.Anthropic:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                # S163-B: Total operation timeout (all phases)
                _cached_client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=_API_TIMEOUT))
    return _cached_client


# S170B P1-3: Robust context-window error detection. Substring-only matches are
# brittle to SDK message reformatting; prefer typed checks first, fall back to
# canonical-phrase substring (NOT bare "token" — collides with auth errors).
_CONTEXT_OVERFLOW_PHRASES = (
    "context window",
    "context length",
    "too many tokens",
    "prompt is too long",
    "maximum context length",
    "input is too long",
    "too long",
)


def _is_context_overflow_error(e: Exception) -> bool:
    # Typed check first — anthropic.BadRequestError carries structured body
    if isinstance(e, getattr(anthropic, "BadRequestError", anthropic.APIError)):
        body = getattr(e, "body", None) or {}
        if isinstance(body, dict):
            err = body.get("error", {})
            if isinstance(err, dict):
                err_type = (err.get("type") or "").lower()
                err_msg = (err.get("message") or "").lower()
                if "context_length" in err_type or "context_window" in err_type:
                    return True
                if any(p in err_msg for p in _CONTEXT_OVERFLOW_PHRASES):
                    return True
    msg = str(e).lower()
    return any(p in msg for p in _CONTEXT_OVERFLOW_PHRASES)


def _get_system(model: str, voice_mode: bool, conversation_id: str = "") -> list[dict]:
    """Get system prompt as content blocks, optionally prepending voice mode instructions."""
    blocks = get_system_prompt(model=model, conversation_id=conversation_id)
    if voice_mode:
        blocks = [{"type": "text", "text": VOICE_MODE_INSTRUCTION}] + blocks
    return blocks


from agent.memory.persistence import PersistentConversationManager

conversation_manager = PersistentConversationManager()

# Tools whose success is worth recording in episodic memory (module-level constant)
_HIGH_VALUE_TOOLS = frozenset({"execute_trade", "save_knowledge", "update_priors", "send_email", "modify_hydrinia", "control_hydrinia"})


def _derive_priority(message: str, task_type: str) -> float:
    """Derive memory priority from content signals."""
    msg_lower = message[:300].lower()
    if any(re.search(r'\b' + re.escape(w) + r'\b', msg_lower) for w in ("wrong", "incorrect", "that's not", "you're wrong", "actually no")) or re.search(r'\bno!', msg_lower):
        return 0.9
    if task_type == "trade_decision":
        return 0.85
    if re.search(r'\b(strategy|thesis|setup|position size|stop loss|take profit)\b', msg_lower):
        return 0.8
    if re.search(r"\b(i prefer|i like|always do|never do|don't ever|i want you to)\b", msg_lower):
        return 0.7
    if task_type != "general" and "?" in message:
        return 0.6
    if len(message) < 10:
        return 0.0
    if len(message) < 30:
        return 0.3
    return 0.5


async def agent_respond(
    user_message: str,
    conversation_id: str,
    task_type: str = "general",
    is_auto_mode: bool = False,
    confirmation_callback: Optional[Callable] = None,
    voice_mode: bool = False,
    source: str = "unknown",
    user_id: str = "",
) -> str:
    # Override directive detection — BEFORE any sanitization (fire-and-forget)
    if config.override_directives_enabled:
        try:
            from agent.override_directive import (
                check_pin_lockout,
                process_override,
                record_pin_attempt,
                redact_passphrase_from_content,
                validate_voice_pin_value,
            )
            override_event = process_override(
                message=user_message,
                source=source,
                user_id=user_id,
                authorized_user_ids=config.telegram_authorized_users,
            )
            if override_event:
                # Audit log (fire-and-forget)
                try:
                    from agent.safety.audit_log import audit_security_event
                    audit_security_event(
                        f"override_{override_event.context}",
                        f"Override {override_event.context}: {override_event.action_taken}",
                        details={"source": source, "user_id": user_id,
                                 "command_preview": (override_event.command_text or "")[:200]},
                    )
                except Exception as e:
                    logger.warning(f"S162 H-6: Fire-and-forget override audit failed: {e}")

                if override_event.context == "legitimate_command":
                    # Twilio voice calls require PIN authentication.
                    # On this path user_id carries the allowlisted CALLER NUMBER
                    # (threaded from voice_server via the TwiML <Parameter>), which
                    # keys the brute-force lockout per-caller so it survives re-dials.
                    if source == "twilio_voice":
                        locked, retry_after = check_pin_lockout(
                            caller=user_id, conversation_id=conversation_id,
                        )
                        if locked:
                            logger.warning(
                                "Voice override BLOCKED by PIN lockout — caller=%s retry_after=%ds",
                                user_id or "<unknown>", retry_after,
                            )
                            return (
                                "Too many failed PIN attempts. Voice overrides are locked for "
                                f"{retry_after // 60 + 1} more minute(s)."
                            )
                        try:
                            pin_valid = validate_voice_pin_value(
                                getattr(override_event, "_voice_pin_candidate", None),
                                config.override_voice_pin,
                            )
                        finally:
                            if hasattr(override_event, "_voice_pin_candidate"):
                                delattr(override_event, "_voice_pin_candidate")
                        record_pin_attempt(
                            pin_valid, caller=user_id, conversation_id=conversation_id,
                        )
                        if not pin_valid:
                            # Caller number is an identifier, not a credential — log it
                            # so a dialling campaign is visible. NEVER log the attempted
                            # PIN value.
                            logger.warning(
                                "Voice override FAILED PIN check — caller=%s source=%s",
                                user_id or "<unknown>", source,
                            )
                            return "Voice override requires valid PIN. Command not executed."

                    # Dispatch to directive delivery (Feature 2)
                    try:
                        from agent.integrations.directive_delivery import dispatch_directive
                        result = await dispatch_directive(override_event, user_message)
                        return result
                    except Exception as e:
                        logger.error(f"Directive dispatch failed: {e}")
                        return f"Override accepted but dispatch failed: {str(e)[:200]}"

                elif override_event.context == "injection_attempt":
                    logger.warning(f"Override passphrase in external content from {source} — redacted")
                    user_message = redact_passphrase_from_content(user_message)

                elif override_event.context == "unauthorized_attempt":
                    logger.critical(f"Unauthorized override attempt from user_id={user_id}")
                    return "Unauthorized."
        except Exception as e:
            logger.debug(f"Fire-and-forget override detection failed: {e}")

    clean_message = sanitize_user_input(user_message)
    model = get_model_for_task(task_type)

    # Set topic for knowledge context injection (before system prompt build)
    try:
        from agent.prompts.system_prompt import set_last_user_message
        set_last_user_message(clean_message, conversation_id=conversation_id)
    except Exception as e:
        logger.debug(f"Fire-and-forget set_last_user_message failed: {e}")

    history = await conversation_manager.get_history(conversation_id)

    # S159 M-2: Session detection per-conversation_id — benign race on same cid (both set _is_new=True)
    # Session gap detection — inject recall instruction after >4h idle
    try:
        from agent.prompts.system_prompt import set_session_fresh
        import time as _time
        _is_new = True  # Default: new session (empty history, voice calls, etc.)
        if history and not voice_mode:
            last_ts = await conversation_manager.get_last_timestamp(conversation_id)
            if last_ts is not None:
                _is_new = (_time.time() - last_ts) > 4 * 3600
        set_session_fresh(_is_new, conversation_id=conversation_id)
        # Clear session-scoped memory state to prevent cross-conversation leakage (Fix S103-2A)
        if _is_new:
            try:
                from agent.memory.memory_engine import get_memory_engine
                get_memory_engine().clear_session_state()
            except Exception as e:
                logger.debug(f"Fire-and-forget clear_session_state failed: {e}")
    except Exception as e:
        logger.debug(f"Fire-and-forget session gap detection failed: {e}")

    await conversation_manager.add_message(conversation_id, "user", clean_message)
    messages = history + [{"role": "user", "content": clean_message}]
    # Anthropic API requires first message to have role "user"
    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "[conversation continued]"})
    # Ensure alternating roles (no consecutive same-role messages)
    _deduped = [messages[0]]
    for _m in messages[1:]:
        if _m["role"] == _deduped[-1]["role"]:
            logger.warning(f"Role dedup: discarding older {_deduped[-1].get('role', '?')} message (len={len(str(_deduped[-1].get('content', '')))})")
            _deduped[-1] = _m  # Keep newer message
        else:
            _deduped.append(_m)
    # S118 W-003: Re-check first message role after dedup (synthetic user may have been merged)
    if _deduped and _deduped[0]["role"] != "user":
        _deduped.insert(0, {"role": "user", "content": "[conversation continued]"})
    messages = _deduped
    loop = asyncio.get_running_loop()

    # Cache system prompt for this call (avoid rebuilding per tool iteration — Fix S104-J)
    # P0-2: _get_system does synchronous multi-file reads + keyword scoring inline; offload
    # off the event loop (loop bound at get_running_loop() above). Awaited → behavior identical.
    _cached_system = await loop.run_in_executor(None, _get_system, model, voice_mode, conversation_id)
    # Adaptive thinking: interleaved reasoning between tool calls (feature-flagged)
    _thinking_kwarg = {"thinking": {"type": "adaptive"}} if config.adaptive_thinking else {}

    try:
        response = await asyncio.wait_for(loop.run_in_executor(None, lambda: _get_client().messages.create(
            model=model, max_tokens=config.max_tokens,
            system=_cached_system, messages=messages, tools=get_all_tools(),
            **_thinking_kwarg,
        )), timeout=_API_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error(f"S163-B: Anthropic API call timed out after {_API_TIMEOUT}s")
        return f"I'm sorry, the request timed out after {_API_TIMEOUT}s. Please try again."
    except anthropic.APIError as e:
        logger.error(f"API error: {e}")
        # If input too long, trim history and retry with just the current message
        if _is_context_overflow_error(e):  # S170B P1-3
            logger.warning("Input too long — retrying with compaction/truncation")
            _compaction_ok = False
            if config.server_compaction:
                try:
                    response = await asyncio.wait_for(loop.run_in_executor(None, lambda: _get_client().messages.create(
                        model=model, max_tokens=config.max_tokens,
                        system=_cached_system, messages=messages, tools=get_all_tools(),
                        context_management={"edits": [{"type": "compact_20260112"}]},
                        **_thinking_kwarg,
                    )), timeout=_API_TIMEOUT)
                    _compaction_ok = True
                except Exception as e_compact:
                    logger.warning(f"Server-side compaction failed ({e_compact}), falling back to history drop")
            if not _compaction_ok:
                try:
                    response = await asyncio.wait_for(loop.run_in_executor(None, lambda: _get_client().messages.create(
                        model=model, max_tokens=config.max_tokens,
                        system=_cached_system,
                        messages=[{"role": "user", "content": clean_message}],
                        tools=get_all_tools(),
                        **_thinking_kwarg,
                    )), timeout=_API_TIMEOUT)
                except (anthropic.APIError, asyncio.TimeoutError) as e2:
                    return f"API error (retry also failed): {str(e2)[:200]}"
        else:
            return f"API error: {str(e)[:200]}"  # S233 CR-5: str(e) not e.message (TimeoutError has no .message)

    iteration = 0
    max_iterations = 2 if voice_mode else 10  # Limit tool loops on live calls for latency

    while response.stop_reason == "tool_use" and iteration < max_iterations:
        iteration += 1
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []

        for tb in tool_use_blocks:
            logger.info(f"Tool: {tb.name}({json.dumps(tb.input, default=str)[:200]})")
            is_valid, reason = validate_tool_call(tb.name, tb.input, is_auto_mode, confirmation_callback)

            if not is_valid:
                logger.warning(f"Blocked: {tb.name} - {reason}")
                try:
                    from agent.safety.audit_log import audit_security_event
                    audit_security_event("tool_blocked", f"Tool blocked: {tb.name} — {reason}", details={"tool": tb.name, "reason": reason})
                except Exception as e:
                    logger.warning(f"S162 H-6: Fire-and-forget tool_blocked audit failed: {e}")
                tool_results.append({"type": "tool_result", "tool_use_id": tb.id, "content": f"BLOCKED: {reason}"})
                continue

            if tb.name in CONFIRMATION_REQUIRED and confirmation_callback:
                try:
                    confirmed = await confirmation_callback(tb.name, tb.input)
                except Exception as e:
                    logger.error(f"Confirmation callback error for {tb.name}: {e}", exc_info=True)
                    tool_results.append({"type": "tool_result", "tool_use_id": tb.id, "content": f"CANCELLED: Confirmation error: {str(e)[:100]}"})
                    continue
                if not confirmed:
                    tool_results.append({"type": "tool_result", "tool_use_id": tb.id, "content": "CANCELLED: User declined."})
                    continue

            try:
                # Audit log pre-execution for high-risk tools
                # P0-4: audit_tool_execution does flock+fsync (~5-10ms) on every tool call;
                # offload off the event loop. AWAITED (not fire-and-forget) so the pre-execution
                # audit record still completes BEFORE the tool runs (log-before-execute forensic
                # ordering), while the flock+fsync runs in the executor so the loop isn't blocked.
                # Default-arg binding snapshots tb.name/tb.input per iteration.
                try:
                    from agent.safety.audit_log import audit_tool_execution
                    await loop.run_in_executor(None, lambda name=tb.name, inp=tb.input: audit_tool_execution(name, inp))
                except Exception as e:
                    logger.warning(f"S162 H-6: Fire-and-forget audit_tool_execution failed: {e}")
                result = await execute_tool(tb.name, tb.input)
                tool_results.append({"type": "tool_result", "tool_use_id": tb.id, "content": str(result)})
                # Record high-value tool successes in episodic memory (fire-and-forget)
                if tb.name in _HIGH_VALUE_TOOLS:
                    try:
                        from agent.memory.memory_engine import get_memory_engine
                        # P0-3: offload episodic.record off the loop (matches remember() at
                        # ~:548). Snapshot the tb-derived content per iteration via default-arg
                        # binding (fire-and-forget → not awaited).
                        _me = get_memory_engine()
                        _ep_content = f"Tool {tb.name} succeeded: {str(tb.input)[:100]}"
                        _ep_future = loop.run_in_executor(None, lambda me=_me, content=_ep_content: me.episodic.record(
                            event_type="tool_execution",
                            content=content,
                            category="system",
                            outcome="success",
                            importance=0.6,
                            domain="tool_execution",
                        ))
                        _ep_future.add_done_callback(
                            lambda f: _log_fire_and_forget_exception(f, "episodic.record")
                        )
                    except Exception as e:
                        logger.warning("Fire-and-forget episodic record failed: %s: %s", type(e).__name__, e, exc_info=True)
            except Exception as e:
                logger.error(f"Tool error ({tb.name}): {e}")
                try:
                    from agent.safety.audit_log import audit_security_event
                    audit_security_event("tool_failed", f"Tool error: {tb.name} — {str(e)[:200]}", details={"tool": tb.name, "error": str(e)[:200]})
                except Exception as e2:
                    logger.warning(f"S162 H-6: Fire-and-forget audit event failed: {e2}")
                tool_results.append({"type": "tool_result", "tool_use_id": tb.id, "content": f"ERROR: {str(e)}"})
                # Learn from tool failure (fire-and-forget)
                try:
                    from agent.memory.memory_engine import get_memory_engine
                    # P0-3: offload learn_from_outcome off the loop (matches remember() at
                    # ~:548). Default-arg binding snapshots action/outcome/domain now —
                    # critical because `e` is cleared at except-block exit (fire-and-forget).
                    _me = get_memory_engine()
                    _lfo_future = loop.run_in_executor(None, lambda me=_me, action=tb.name, outcome=str(e)[:200], domain=task_type: me.learn_from_outcome(
                        action=action, outcome=outcome,
                        domain=domain, was_correct=False,
                    ))
                    _lfo_future.add_done_callback(
                        lambda f: _log_fire_and_forget_exception(f, "learn_from_outcome")
                    )
                except Exception as e2:
                    logger.debug(f"Fire-and-forget learn_from_outcome failed: {e2}")
                # ANVIL error lesson (fire-and-forget)
                try:
                    from agent.tools.error_lesson_moltbot import get_moltbot_lesson_adapter
                    _err_str = str(e).lower()
                    _anvil_class = "API_FAILURE" if any(k in _err_str for k in ("timeout", "status", "http", "429", "500", "503", "529", "rate_limit", "connection")) else "LOGIC_ERROR"
                    await get_moltbot_lesson_adapter().record_error(
                        error_class=_anvil_class,
                        description=f"Tool {tb.name} failed: {str(e)[:150]}",
                        affected_files=[],
                        context={"tool": tb.name, "error": str(e)[:200]},
                    )
                except Exception as e2:
                    logger.debug(f"Fire-and-forget ANVIL record_error failed: {e2}")

        _content_for_loop = _assistant_content_for_loop(response)
        if not _content_for_loop:
            # S159 M-1: Prevent empty assistant content (API schema violation).
            # S170B P3-9: prefer a thinking-summary excerpt over the opaque
            # "(Processing...)" stub — gives the next turn at least some context.
            _thinking_text = ""
            try:
                for _b in response.content:
                    if getattr(_b, "type", None) == "thinking":
                        _thinking_text += getattr(_b, "thinking", "") or ""
            except Exception as e:
                logger.debug("Failed to extract thinking-summary blocks for stub fallback: %s", e)
            _stub = (_thinking_text[:200].strip() if _thinking_text else "") or "(Processing...)"
            logger.debug("Empty content after thinking-filter; using stub (thinking len=%d)", len(_thinking_text))
            _content_for_loop = [{"type": "text", "text": _stub}]
        messages.append({"role": "assistant", "content": _content_for_loop})
        messages.append({"role": "user", "content": tool_results})

        try:
            response = await asyncio.wait_for(loop.run_in_executor(None, lambda: _get_client().messages.create(
                model=model, max_tokens=config.max_tokens,
                system=_cached_system, messages=messages, tools=get_all_tools(),
                **_thinking_kwarg,
            )), timeout=_API_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"S163-B: Tool loop API call timed out after {_API_TIMEOUT}s (iteration {iteration})")
            break  # Exit tool loop gracefully, return whatever we have
        except anthropic.APIError as e:
            if _is_context_overflow_error(e):  # S170B P1-3
                logger.warning(f"Tool loop context overflow ({e}), retrying with compaction/trim")
                _compaction_ok = False
                if config.server_compaction:
                    try:
                        response = await asyncio.wait_for(loop.run_in_executor(None, lambda: _get_client().messages.create(
                            model=model, max_tokens=config.max_tokens,
                            system=_cached_system, messages=messages, tools=get_all_tools(),
                            context_management={"edits": [{"type": "compact_20260112"}]},
                            **_thinking_kwarg,
                        )), timeout=_API_TIMEOUT)
                        _compaction_ok = True
                    except Exception as e_compact:
                        logger.warning(f"Tool loop compaction failed ({e_compact}), falling back to trim")
                if not _compaction_ok:
                    # Fallback: trim to last complete user message (not tool_result)
                    _trim_idx = len(messages) - 1
                    while _trim_idx > 0:
                        m = messages[_trim_idx]
                        if m["role"] == "user" and isinstance(m.get("content"), str):
                            break
                        _trim_idx -= 1
                    if _trim_idx == 0 and not (messages[0]["role"] == "user" and isinstance(messages[0].get("content"), str)):
                        # No string user message found — fall back to synthetic
                        messages = [{"role": "user", "content": "[conversation continued]"}]
                    else:
                        messages = messages[_trim_idx:]
                    if not messages or messages[0]["role"] != "user":
                        messages = [{"role": "user", "content": "[conversation continued]"}]
                    # Role dedup (S118 W-003)
                    _deduped = [messages[0]]
                    for m in messages[1:]:
                        if m["role"] == _deduped[-1]["role"]:
                            _deduped[-1] = m
                        else:
                            _deduped.append(m)
                    if _deduped and _deduped[0]["role"] != "user":
                        _deduped.insert(0, {"role": "user", "content": "[conversation continued]"})
                    messages = _deduped
                    try:
                        response = await asyncio.wait_for(loop.run_in_executor(None, lambda: _get_client().messages.create(
                            model=model, max_tokens=config.max_tokens,
                            system=_cached_system, messages=messages, tools=get_all_tools(),
                            **_thinking_kwarg,
                        )), timeout=_API_TIMEOUT)
                    except (anthropic.APIError, asyncio.TimeoutError) as e2:
                        return f"API error in tool loop (retry failed): {str(e2)[:200]}"
            else:
                return f"API error in tool loop: {str(e)[:200]}"

    # S189 NEW-1: If the tool loop exited at the iteration cap with stop_reason
    # still == "tool_use", Claude wanted more turns but ran out of budget. The
    # final response then has ONLY tool_use blocks and no text — so
    # extract_text_response would fall back to "Processed but no text response."
    # This was the operator-reported symptom: "CLIO says 'Processed text but no
    # response'; duplicate message required."
    # Fix: do one forced-summary turn with tool_choice=none so Claude must
    # produce text from what it already has.
    if (
        getattr(response, "stop_reason", None) == "tool_use"
        and iteration >= max_iterations
    ):
        try:
            unresolved_tool_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            logger.warning(
                f"S189 NEW-1: tool loop hit max_iterations={max_iterations} with "
                f"stop_reason=tool_use ({len(unresolved_tool_blocks)} unresolved tool_use blocks); "
                f"forcing summary turn (conv={conversation_id})."
            )
            # Append the last (unresolved) assistant response + synthetic tool_results
            # so the message sequence remains API-valid.
            _last_content = _assistant_content_for_loop(response)
            if not _last_content:
                _last_content = [{"type": "text", "text": "(Tool budget reached)"}]
            messages.append({"role": "assistant", "content": _last_content})
            synth_tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": "INTERRUPTED: tool-call budget exhausted for this turn. Summarize what you've found and answer the user from current findings; do NOT request more tools.",
                }
                for tb in unresolved_tool_blocks
            ]
            messages.append({"role": "user", "content": _forced_summary_user_turn(synth_tool_results)})
            response = await _forced_summary_turn(loop, model, _cached_system, messages, _thinking_kwarg)
        except Exception as _summ_e:
            logger.error(f"S189 NEW-1: force-summary turn failed: {_summ_e}; falling through")

    final_text = extract_text_response(response)
    # S189 NEW-1: Diagnostic if extract_text_response STILL returns the fallback —
    # surfaces the issue with stop_reason context so operator can correlate.
    if final_text == "Processed but no text response.":
        logger.warning(
            f"S189 NEW-1: empty-text response delivered (stop_reason={getattr(response, 'stop_reason', '?')}, "
            f"iteration={iteration}, conv={conversation_id}); user will see fallback string."
        )
    # Strip markdown for voice callers (clean TTS output)
    if voice_mode:
        try:
            from agent.integrations.tts import strip_markdown
            final_text = strip_markdown(final_text)
        except Exception as e:
            logger.debug(f"Fire-and-forget strip_markdown failed: {e}")
    # Only save text blocks to persistence — tool_use blocks cause API errors
    # when loaded back without matching tool_result blocks
    await conversation_manager.add_message(conversation_id, "assistant", _serialize_content_text_only(response.content))

    # Record calibration prediction when confidence language detected (fire-and-forget)
    # S163-B: meta operations use fcntl.flock — run in executor to avoid blocking event loop
    try:
        from agent.memory.memory_engine import get_memory_engine
        _me = get_memory_engine()
        def _calibration_task():
            _conf = _me.meta.detect_confidence_level(final_text[:1000])
            if _conf is not None:
                import hashlib
                _pred_id = f"cal_{hashlib.md5(final_text[:100].encode()).hexdigest()[:12]}"
                _me.meta.record_prediction(
                    prediction_id=_pred_id,
                    domain=task_type,
                    stated_confidence=_conf,
                    description=f"{task_type} conf={_conf}",
                )
        loop.run_in_executor(None, _calibration_task)
    except Exception as e:
        logger.debug(f"Fire-and-forget calibration failed: {e}")

    # S163-B/S233: Fire-and-forget executor tasks below. Exceptions are lost because
    # the Future is not awaited — intentional. These operations (calibration, memory,
    # hot context export) must not block the response. Errors logged inside each task.

    # Update working memory with conversation context (fire-and-forget)
    # S163-B: remember() uses fcntl.flock — MUST run in executor to avoid blocking event loop
    try:
        from agent.memory.memory_engine import get_memory_engine
        from agent.override_directive import redact_passphrase_from_content
        _safe_message = redact_passphrase_from_content(clean_message)
        _priority = _derive_priority(_safe_message, task_type)
        if _priority > 0.0:
            _me = get_memory_engine()
            loop.run_in_executor(None, lambda: _me.remember(
                key=f"conv_{conversation_id[-12:]}",
                content=_safe_message[:250],
                category=task_type, domain=task_type,
                priority=_priority,
                source=source,
            ))
    except Exception as e:
        logger.warning("Fire-and-forget memory update failed: %s: %s", type(e).__name__, e, exc_info=True)

    # S163-B: ALL post-response fire-and-forget operations use flock or file I/O.
    # Bundle them into a single executor task to avoid blocking the event loop.
    def _post_response_tasks():
        """Run all post-response fire-and-forget ops in a thread (off event loop)."""
        # Positive user feedback updates recent episodic outcomes
        try:
            _msg_lower = clean_message[:200].lower()
            _NEGATION_PREFIXES = ("no ", "not ", "don't ", "doesn't ", "didn't ", "never ")
            for _fw in ("thanks", "perfect", "that worked", "great", "exactly"):
                _match = re.search(r'\b' + _fw + r'\b', _msg_lower)
                if _match:
                    _pre = _msg_lower[max(0, _match.start() - 20):_match.start()]
                    if any(_neg in _pre for _neg in _NEGATION_PREFIXES):
                        continue
                    from agent.memory.memory_engine import get_memory_engine
                    get_memory_engine().episodic.update_outcome(
                        event_type="memory", keyword="", outcome="success",
                    )
                    break
        except Exception as e:
            logger.warning("Fire-and-forget feedback update failed: %s: %s", type(e).__name__, e, exc_info=True)

        # Record trait activations from response
        try:
            if os.environ.get("PERSONALITY_ENABLED", "").lower() == "true":
                from agent.personality.registry import get_trait_registry
                _registry = get_trait_registry()
                if _registry:
                    _confirmed = _registry.get_confirmed_traits()
                    _response_lower = final_text[:2000].lower()
                    for _trait in _confirmed:
                        if _trait.name and _trait.name.lower() in _response_lower:
                            _registry.record_activation(_trait.id)
        except Exception as e:
            logger.debug(f"Fire-and-forget trait activation recording failed: {e}")

        # Export hot context to Hydrinia mirror
        try:
            from agent.memory.memory_engine import get_memory_engine
            get_memory_engine().export_hot_context()
        except Exception as e:
            logger.debug(f"Fire-and-forget hot context export failed: {e}")

    loop.run_in_executor(None, _post_response_tasks)

    logger.info(f"Response: model={model}, conv={conversation_id}, tools={iteration}, in={response.usage.input_tokens}, out={response.usage.output_tokens}")
    return final_text


def _serialize_content_text_only(content):
    """Convert Anthropic SDK content blocks to plain dicts, stripping tool_use and thinking blocks.

    tool_use blocks in persisted assistant messages cause 400 errors when loaded
    back into conversation history (the API requires matching tool_result blocks).
    thinking blocks are internal reasoning and shouldn't bloat stored history.
    We only persist the text content for conversation continuity.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for block in content:
            block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if block_type in ("tool_use", "thinking"):
                continue
            if hasattr(block, "model_dump"):
                result.append(block.model_dump())
            elif isinstance(block, dict):
                result.append(block)
            else:
                result.append({"type": "text", "text": str(block)})
        if not result:
            result = [{"type": "text", "text": "(Tool calls processed)"}]
        return result
    return str(content)


def _preserve_thinking() -> bool:
    """S214: whether thinking blocks are passed back unmodified in tool-loop turns.

    CLIO_PRESERVE_THINKING_BLOCKS=1 -> True, =0 -> False, anything else (auto/unset)
    -> preserve on direct api.anthropic.com, strip on the gateway.
    """
    raw = os.environ.get("CLIO_PRESERVE_THINKING_BLOCKS", "").strip()
    if raw == "1":
        return True
    if raw == "0":
        return False
    return not bool(getattr(config, "anthropic_gateway", False))


def _assistant_content_for_loop(response) -> list:
    """S214: assistant content to replay in a tool loop.

    Preserving: every block is kept as the SDK object (thinking included, unmodified).
    Otherwise: thinking blocks are dropped (the pre-S214 behaviour).
    """
    content = list(getattr(response, "content", None) or [])
    if _preserve_thinking():
        return content
    return [b for b in content if getattr(b, "type", None) != "thinking"]


def _has_nonblank_text(response) -> bool:
    return bool(first_text(response).strip())


def _has_tool_use(response) -> bool:
    return any(getattr(b, "type", None) == "tool_use" for b in (getattr(response, "content", None) or []))


def _forced_summary_user_turn(synth_tool_results: list) -> list:
    """S214: tool_result blocks first, then the explicit instruction as the last text block."""
    return list(synth_tool_results) + [{"type": "text", "text": _FORCED_SUMMARY_INSTRUCTION}]


async def _forced_summary_turn(loop, model, system, messages, thinking_kwarg):
    """S214: one forced-summary API turn, with a gateway-only tool-free retry.

    First attempt keeps tools + tool_choice=none (correct on api.anthropic.com).
    On the gateway, if that reply has no text or contains tool_use, retry once
    without tools/tool_choice. A failed or empty retry keeps the first response.
    Timeouts from the first attempt propagate to the caller (existing handling).
    """
    response = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: _get_client().messages.create(
            model=model,
            max_tokens=config.max_tokens,
            system=system,
            messages=messages,
            tools=get_all_tools(),
            tool_choice={"type": "none"},
            **thinking_kwarg,
        )),
        timeout=_API_TIMEOUT,
    )
    if not getattr(config, "anthropic_gateway", False):
        return response
    if _has_nonblank_text(response) and not _has_tool_use(response):
        return response
    try:
        logger.info("S214 forced-summary retry (gateway, no tools)")
        retry = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _get_client().messages.create(
                model=model,
                max_tokens=config.max_tokens,
                system=system,
                messages=messages,
                **thinking_kwarg,
            )),
            timeout=_API_TIMEOUT,
        )
    except Exception as _retry_e:
        logger.warning(f"S214 forced-summary retry failed: {_retry_e}; keeping first response")
        return response
    if _has_nonblank_text(retry):
        return retry
    return response


def extract_text_response(response) -> str:
    parts = [b.text for b in response.content if hasattr(b, "text") and getattr(b, "type", None) != "thinking"]
    return "\n".join(parts) if parts else "Processed but no text response."


def classify_task(message: str) -> str:
    m = message.lower()
    # S121 W-08: Use word-boundary regex instead of substring `in` matching
    def _has(keywords):
        return any(re.search(r'\b' + re.escape(k) + r'\b', m) for k in keywords)
    if _has(["trade", "buy", "sell", "position", "portfolio", "options", "dark pool", "flow", "strategy", "hedge", "backtest", "hydrinia", "dpof", "algo", "bot code"]):
        return "trade_decision"
    if _has(["homework", "assignment", "essay", "paper", "reading", "canvas", "course", "exam", "study", "physics", "discussion post"]) or (re.search(r'\bclass\b', m) and not re.search(r'\basset\s+class\b', m)):
        return "homework"
    if _has(["code", "debug", "function", "refactor", "deploy", "git"]):
        return "code_modification"
    return "general"
