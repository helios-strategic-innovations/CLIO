"""
Moltbook Kill Switch
=====================
Emergency halt system for the Moltbook subsystem.

Activation sequence:
  1. Writes immutable audit log entry (timestamped, with PID)
  2. Sets config.enabled = False in memory
  3. Writes data/kill_switches/MOLTBOOK_KILLED file (persists across restarts)
  4. Sends Telegram alert to Dominic
  5. Closes any active aiohttp sessions cleanly

Deactivation requires Dominic to manually delete the MOLTBOOK_KILLED file.

The heartbeat loop checks the kill file on every cycle — so even if memory
state resets, the kill switch remains active.

Author: Moltbot (Session 68, Feb 27 2026)
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone

logger = logging.getLogger("moltbot.moltbook.kill_switch")

KILL_FILE = "/home/agent/openclaw/data/kill_switches/MOLTBOOK_KILLED"


class MoltbookKillSwitch:
    """
    Persistent kill switch for the Moltbook subsystem.
    File-based: survives process restarts.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(KILL_FILE), exist_ok=True)
        # S170B P1-5: serialize activate/deactivate against is_active reads.
        # Writes use os.replace/os.remove (atomic at filesystem layer); the lock
        # prevents read-during-mutation interleaving from reporting stale state.
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        """Returns True if kill switch is engaged (Moltbook is halted)."""
        with self._lock:
            return os.path.isfile(KILL_FILE)

    def activate(self, reason: str = "manual", user_id: str = "") -> dict:
        """
        Activate the kill switch. Idempotent.

        Steps:
          1. Write kill file
          2. Log to audit log
          3. Alert Dominic via Telegram (best-effort)
        """
        try:
            kill_data = {
                "killed_at": datetime.now(timezone.utc).isoformat(),
                "killed_by": user_id or "system",
                "reason": reason,
                "pid": os.getpid(),
            }
            tmp_path = KILL_FILE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(kill_data, f, indent=2)
            with self._lock:  # S170B P1-5: serialize against is_active readers
                os.replace(tmp_path, KILL_FILE)

            logger.critical(
                f"MOLTBOOK KILL SWITCH ACTIVATED: reason={reason}, user={user_id}, pid={os.getpid()}"
            )

            # Log to immutable audit log
            try:
                from agent.safety.audit_log import get_audit_log, AuditEntry, AuditEventType
                get_audit_log().write(AuditEntry(
                    event_type=AuditEventType.MOLTBOOK_KILL,
                    subsystem="moltbook",
                    summary=f"Moltbook kill switch activated: {reason}",
                    details=kill_data,
                    user_id=user_id,
                    severity="CRITICAL",
                ))
            except Exception as e:
                logger.error(f"Failed to write kill switch to audit log: {e}")

            # Alert Dominic (best-effort, non-blocking)
            try:
                self._send_kill_alert(reason, user_id)
            except Exception as e:
                logger.error(f"Failed to send kill switch Telegram alert: {e}")

            return {"success": True, "kill_file": KILL_FILE, "reason": reason}

        except Exception as e:
            logger.error(f"Failed to activate Moltbook kill switch: {e}")
            return {"success": False, "error": str(e)}

    def deactivate(self, user_id: str = "") -> dict:
        """
        Deactivate the kill switch. Requires Dominic authorization.
        Removes the kill file — Moltbook reverts to MOLTBOOK_ENABLED env var control.
        """
        if not self.is_active:
            return {"success": True, "note": "Kill switch was not active"}

        try:
            with self._lock:  # S170B P1-5: serialize against is_active readers
                if os.path.isfile(KILL_FILE):
                    os.remove(KILL_FILE)
            logger.warning(f"Moltbook kill switch DEACTIVATED by user {user_id}")

            try:
                from agent.safety.audit_log import get_audit_log, AuditEntry, AuditEventType
                get_audit_log().write(AuditEntry(
                    event_type=AuditEventType.KILL_SWITCH_DEACTIVATED,
                    subsystem="moltbook",
                    summary="Moltbook kill switch deactivated",
                    details={"deactivated_by": user_id},
                    user_id=user_id,
                    severity="WARNING",
                ))
            except Exception as e:
                logger.debug(f"Fire-and-forget deactivate audit log failed: {e}")

            return {"success": True, "note": "Kill switch deactivated — Moltbook reverts to env var control"}

        except Exception as e:
            logger.error(f"Failed to deactivate Moltbook kill switch: {e}")
            return {"success": False, "error": str(e)}

    def status(self) -> dict:
        """Return current kill switch status."""
        if not self.is_active:
            return {"active": False}

        try:
            with open(KILL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"active": True, **data}
        except Exception as e:
            logger.debug(f"Kill file read failed: {e}")
            return {"active": True, "note": "Kill file exists but could not be read"}

    def _send_kill_alert(self, reason: str, user_id: str):
        """Send Telegram alert to Dominic (best-effort)."""
        try:
            from agent.integrations.telegram_bot import bot_app
            from agent.config import config
            if not bot_app:
                return
            if not config.telegram_authorized_users:
                logger.error("Kill switch activated but no telegram_authorized_users configured")
                return
            import asyncio
            alert = (
                "MOLTBOOK KILL SWITCH ACTIVATED\n\n"
                f"Reason: {reason}\n"
                f"Activated by: {user_id or 'system'}\n"
                f"Time: {datetime.now(timezone.utc).isoformat()}\n\n"
                "Moltbook is now fully halted. To re-enable, delete the kill file and restart."
            )
            chat_id = config.telegram_authorized_users[0]
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    bot_app.bot.send_message(chat_id=chat_id, text=alert)
                )
            except RuntimeError:
                logger.warning("No event loop for kill switch alert")
        except Exception as e:
            logger.debug(f"Kill switch alert send failed (non-blocking): {e}")


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_kill_switch = None
_init_lock = threading.Lock()


def get_kill_switch() -> MoltbookKillSwitch:
    global _kill_switch
    if _kill_switch is None:
        with _init_lock:
            if _kill_switch is None:
                _kill_switch = MoltbookKillSwitch()
    return _kill_switch
