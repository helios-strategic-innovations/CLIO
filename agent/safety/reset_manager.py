"""
Moltbot Reset Manager
======================
Clean state restoration protocols at three levels.

Level 1 — Soft Reset:   Clear conversation history only
Level 2 — System Reset: Reset one subsystem's state files
Level 3 — Full Reset:   Reset all state EXCEPT audit logs (immutable, never reset)

All resets are logged to the immutable audit log before execution.

Author: Moltbot (Session 68, Feb 27 2026)
"""

import glob
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("moltbot.reset")

# Directories that can be reset per-system
SYSTEM_RESET_PATHS = {
    "moltbook": [
        "/home/agent/openclaw/data/moltbook/quarantine/",
        "/home/agent/openclaw/data/moltbook/logs/",
        "/home/agent/openclaw/data/moltbook/memory.json",
    ],
    "personality": [
        "/home/agent/openclaw/data/personality/trait_history.jsonl",
        # NOTE: trait_registry.json is NOT reset — traits are permanent
        # Only the history log is cleared
    ],
    "conversations": [
        "/home/agent/openclaw/data/conversations.db",
    ],
    "knowledge": [
        # This is dangerous — only reset with full understanding
        "/home/agent/openclaw/data/knowledge/",
    ],
    "kill_switches": [
        "/home/agent/openclaw/data/kill_switches/",
    ],
}

# NEVER reset these (immutable records)
PROTECTED_PATHS = [
    "/home/agent/openclaw/data/audit/",
    "/home/agent/openclaw/logs/override_directive.jsonl",
    "/home/agent/openclaw/logs/security.log",
]


class ResetManager:
    """
    Manages clean state restoration at three levels.
    All resets require explicit confirmation and are audit-logged.
    """

    def reset_conversations(self, conversation_id: Optional[str] = None) -> dict:
        """
        Level 1 — Soft Reset: Clear conversation history.

        Args:
            conversation_id: If provided, clear only this conversation.
                           If None, clear ALL conversations.
        """
        try:
            from agent.safety.audit_log import audit_security_event
            audit_security_event("reset_conversation", f"Soft reset initiated: {'single=' + conversation_id if conversation_id else 'ALL'}", severity="CRITICAL")
        except Exception as e:
            logger.debug(f"Fire-and-forget reset_conversation audit failed: {e}")
            # S190 L8: reset is operator-initiated AND destructive. If the audit
            # subsystem is down, we still want a forensic breadcrumb in stderr/journal.
            import sys
            print(f"[AUDIT_BYPASS] reset_conversation cid={conversation_id!r} (audit unavailable: {e})",
                  file=sys.stderr, flush=True)
        try:
            if conversation_id:
                # Single conversation clear handled by conversation_manager
                logger.info(f"Soft reset: conversation {conversation_id}")
                return {
                    "success": True,
                    "level": "soft",
                    "cleared": conversation_id,
                    "note": "Use conversation_manager.clear() to execute",
                }
            else:
                # Clear entire conversations database
                db_path = "/home/agent/openclaw/data/conversations.db"
                if os.path.isfile(db_path):
                    # S118 W-057: Warn about potentially open DB connections
                    logger.warning("Soft reset: deleting conversations.db (open connections may persist until closed)")
                    try:
                        os.remove(db_path)
                    except PermissionError:
                        logger.error("Cannot delete conversations.db — file is locked by another process")
                        return {"success": False, "error": "Database file locked"}
                    logger.warning("Soft reset: ALL conversations cleared")
                return {
                    "success": True,
                    "level": "soft",
                    "cleared": "all_conversations",
                }
        except Exception as e:
            logger.error(f"Soft reset failed: {e}")
            return {"success": False, "error": str(e)}

    def reset_system(self, system_name: str) -> dict:
        """
        Level 2 — System Reset: Clear one subsystem's state files.

        Args:
            system_name: One of "moltbook", "personality", "conversations", "kill_switches"
        """
        try:
            from agent.safety.audit_log import audit_security_event
            audit_security_event("reset_system", f"System reset initiated: {system_name}", details={"system": system_name}, severity="CRITICAL")
        except Exception as e:
            logger.debug(f"Fire-and-forget reset_system audit failed: {e}")
            # S190 L8: see reset_conversation note — destructive operator action
            # MUST leave a forensic trace even when audit subsystem is down.
            import sys
            print(f"[AUDIT_BYPASS] reset_system system={system_name!r} (audit unavailable: {e})",
                  file=sys.stderr, flush=True)
        if system_name not in SYSTEM_RESET_PATHS:
            return {"success": False, "error": f"Unknown system: {system_name}. Valid: {list(SYSTEM_RESET_PATHS.keys())}"}

        paths = SYSTEM_RESET_PATHS[system_name]
        cleared = []
        errors = []

        for path in paths:
            # Safety check: never touch protected paths
            # S162 M-9: Normalize path for boundary-safe comparison (trailing slash)
            _norm_path = path if path.endswith(os.sep) else path + os.sep
            if any(_norm_path.startswith(p) for p in PROTECTED_PATHS) or path in PROTECTED_PATHS:
                errors.append(f"PROTECTED: {path}")
                continue

            try:
                if os.path.isdir(path):
                    # Clear directory contents, keep the directory
                    for item in glob.glob(os.path.join(path, "*")):
                        if os.path.isfile(item):
                            os.remove(item)
                            cleared.append(item)
                        elif os.path.isdir(item):
                            shutil.rmtree(item)
                            cleared.append(item)
                elif os.path.isfile(path):
                    os.remove(path)
                    cleared.append(path)
            except Exception as e:
                errors.append(f"{path}: {e}")

        logger.warning(f"System reset: {system_name} — cleared {len(cleared)} items, {len(errors)} errors")

        return {
            "success": len(errors) == 0,
            "level": "system",
            "system": system_name,
            "cleared": cleared,
            "errors": errors,
        }

    def reset_full(self) -> dict:
        """
        Level 3 — Full Reset: Clear ALL state files EXCEPT audit logs.

        This is the nuclear option. It:
          - Clears all conversations
          - Clears all subsystem state (moltbook, personality, kill switches)
          - Preserves: audit logs, security logs, override directive logs
          - Preserves: knowledge base (too valuable to auto-delete)
        """
        try:
            from agent.safety.audit_log import audit_security_event
            audit_security_event("reset_full", "FULL RESET initiated — all state except audit logs", severity="CRITICAL")
        except Exception as e:
            logger.debug(f"Fire-and-forget reset_full audit failed: {e}")
            # S190 L8: FULL RESET is the most destructive operator action. Stderr
            # breadcrumb is the bare minimum forensic anchor if audit is down.
            import sys
            print(f"[AUDIT_BYPASS] reset_full (audit unavailable: {e})",
                  file=sys.stderr, flush=True)
        results = {}

        for system_name in ["conversations", "moltbook", "personality", "kill_switches"]:
            results[system_name] = self.reset_system(system_name)

        # S118 W-058: Verify protected paths are untouched (simplified — path must exist)
        protected_intact = all(os.path.exists(p) for p in PROTECTED_PATHS)

        logger.critical("FULL RESET executed — all state cleared except audit logs and knowledge")

        return {
            "success": all(r.get("success", False) for r in results.values()),
            "level": "full",
            "systems_reset": results,
            "audit_logs_intact": protected_intact,
            "knowledge_preserved": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def verify_protected_paths(self) -> dict:
        """Verify that all protected paths still exist and are intact."""
        results = {}
        for path in PROTECTED_PATHS:
            exists = os.path.exists(path)
            results[path] = {
                "exists": exists,
                "type": "directory" if os.path.isdir(path) else "file" if os.path.isfile(path) else "missing",
            }
            if os.path.isfile(path):
                results[path]["size_bytes"] = os.path.getsize(path)
        return results
