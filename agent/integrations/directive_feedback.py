"""
Directive Feedback — Polls Hydrinia directive responses and notifies Dominic via Telegram.

Runs as a 300s scheduled task. Scans moltbot_knowledge/directive_responses/ for new response files
written by Hydrinia after agent deliberations. Sends compliance summary to Telegram.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("moltbot.directive_feedback")


def _get_responses_dir():
    try:
        from agent.config import config
        return os.path.join(config.tradingbot_dir, "moltbot_knowledge", "directive_responses")
    except Exception:
        return "/home/agent/tradingbot/moltbot_knowledge/directive_responses"


RESPONSES_DIR = _get_responses_dir()

# S190 H4: bounded LRU dedup (was unbounded set). Disk-level rename to .notified.json
# is the primary durable mechanism — this in-memory cache is the secondary fast-path.
# 2000 entries ~= many weeks of directive activity, well above realistic restart cadence.
import collections
_MAX_NOTIFIED_RESPONSES = 2000
_notified_responses: "collections.OrderedDict[str, None]" = collections.OrderedDict()

# Cap notifications per cycle to prevent spam
MAX_NOTIFICATIONS_PER_CYCLE = 5


async def poll_directive_feedback(context) -> None:
    """
    Scheduled task: scan directive_responses/ for new files and notify via Telegram.
    Called every 300s by APScheduler (registered in scheduled_tasks.py).
    """
    # Hydrinia is permanently off (2026-09-24) — this whole module reads its
    # moltbot_knowledge/directive_responses/ bridge dir, which nothing writes
    # to any more. Defense in depth beyond the job-registration gate.
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            return
    except Exception as e:
        logger.debug(f"Bridge check failed (treating as disabled): {e}")
        return

    # S142-B W4-D: Skip polling if Hydrinia is not running
    try:
        from agent.integrations.scheduled_tasks import _is_hydrinia_running
        _running, _ = _is_hydrinia_running()
        if not _running:
            return
    except Exception as e:
        logger.debug("Hydrinia liveness check skipped: %s", e)

    responses_dir = Path(RESPONSES_DIR)
    if not responses_dir.is_dir():
        return

    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        chat_ids = _get_chat_ids()

        notifications_sent = 0

        for json_file in sorted(responses_dir.glob("*.json")):
            if notifications_sent >= MAX_NOTIFICATIONS_PER_CYCLE:
                break

            # Skip already-notified files (in-memory dedup)
            if json_file.name in _notified_responses:
                continue

            # Skip .notified.json files (disk-level dedup — survives restart)
            if ".notified." in json_file.name:
                continue

            try:
                with open(json_file) as f:
                    response = json.load(f)

                directive_id = response.get("directive_id", "?")
                trade_id = response.get("trade_id", "?")
                summary = response.get("summary", {})
                total = summary.get("total", 0)
                complied = summary.get("complied", 0)
                dissented = summary.get("dissented", 0)
                try:
                    dissent_rate = float(summary.get("dissent_rate", 0))
                except (ValueError, TypeError):
                    dissent_rate = 0.0

                # Build notification message
                lines = [
                    f"Directive Response -- {directive_id}",
                    f"Trade: {trade_id[:40]}",
                    f"Result: {complied}/{total} complied, {dissented}/{total} dissented "
                    f"({dissent_rate:.0%} dissent rate)",
                ]

                # List dissenters with reasons
                dissenters = [
                    r for r in response.get("responses", [])
                    if r.get("compliance") == "DISSENTED"
                ]
                if dissenters:
                    lines.append("")
                    lines.append("Dissenters:")
                    for d in dissenters[:5]:  # Cap at 5 dissenters
                        reason = d.get("dissent_reason", "no reason given")[:200]
                        lines.append(f"  {d.get('agent_name', '?')}: {reason}")

                message = "\n".join(lines)

                # S190 H4: rename-BEFORE-send. Previous order (send → rename → in-memory
                # mark) had a crash window: if the bot crashed after send but before
                # rename, the file stayed *.json and got re-notified on next start.
                # Now: rename first → disk dedup is durable even on crash. If send
                # fails the warning at line below makes the missed notification
                # operator-visible; one missed alert is better than restart-spam.
                notified_path = json_file.with_suffix(".notified.json")
                try:
                    json_file.rename(notified_path)
                except OSError as e:
                    logger.warning("directive_feedback: rename failed for %s — skipping this cycle: %s",
                                   json_file.name, e)
                    continue

                # Send to all configured chats (only if we have chat_ids)
                if chat_ids:
                    for chat_id in chat_ids:
                        try:
                            await _safe_send(context.bot, chat_id, text=message)
                        except Exception as e:
                            logger.warning(f"Failed to send directive feedback to {chat_id}: {e}")

                # S190 H4: bounded LRU add — in-memory dedup, secondary to disk rename.
                _notified_responses[json_file.name] = None
                if len(_notified_responses) > _MAX_NOTIFIED_RESPONSES:
                    _notified_responses.popitem(last=False)
                notifications_sent += 1
                logger.info(f"Directive feedback processed: {directive_id} ({complied}/{total} complied)")

            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Directive feedback: skip {json_file.name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Directive feedback poll failed: {e}")
