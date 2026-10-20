"""
Moltbot Kill Switch Registry
==============================
Unified registry of all system kill switches.
Single source of truth for system enable/disable state.

Switches:
  - moltbook:      Moltbook social participation (default: OFF)
  - personality:   Personality trait injection (default: OFF)
  - auto_learn:    Autonomous learning engine (default: ON)
  - live_trading:  Live (non-paper) trading (default: OFF)
  - email_send:    Outgoing email capability (default: ON)

Each switch is backed by:
  1. Environment variable (primary source of truth)
  2. File-based override (for kill switches that survive process restarts)

Author: Moltbot (Session 68, Feb 27 2026)
"""

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("moltbot.kill_switches")

KILL_SWITCH_DIR = "/home/agent/openclaw/data/kill_switches"


# ─────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class KillSwitch:
    """A single kill switch."""
    name: str
    env_var: str
    default: bool
    description: str
    # File-based override: if this file exists, switch is forced OFF
    # regardless of env var state
    kill_file: str = ""

    @property
    def is_enabled(self) -> bool:
        """Check if this switch is currently enabled."""
        # File-based kill takes absolute precedence
        if self.kill_file and os.path.isfile(self.kill_file):
            return False
        # Then check environment variable
        # S162 M-8: Distinguish unset (use default) from empty string (treat as disabled)
        env_val = os.getenv(self.env_var)
        if env_val is None:
            return self.default  # Truly not set — use default
        if not env_val:
            return False  # Explicitly empty — treat as disabled
        if env_val.lower() in ("true", "1", "yes", "on"):
            return True
        return False  # Explicit: "false", "no", "0", "off", etc.


# ─────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────

class KillSwitchRegistry:
    """
    Unified registry of all system kill switches.
    Thread-safe. Provides status queries and toggle operations.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ensure_dir()

        self.switches: dict[str, KillSwitch] = {
            "moltbook": KillSwitch(
                name="moltbook",
                env_var="MOLTBOOK_ENABLED",
                default=False,
                description="Moltbook social participation subsystem",
                kill_file=os.path.join(KILL_SWITCH_DIR, "MOLTBOOK_KILLED"),
            ),
            "moltbook_dry_run": KillSwitch(
                name="moltbook_dry_run",
                env_var="MOLTBOOK_DRY_RUN",
                default=True,  # Dry run ON by default = safe
                description="Moltbook dry run mode (true = no real API calls)",
            ),
            "personality": KillSwitch(
                name="personality",
                env_var="PERSONALITY_ENABLED",
                default=False,
                description="Personality trait injection into system prompt",
                kill_file=os.path.join(KILL_SWITCH_DIR, "PERSONALITY_KILLED"),
            ),
            "auto_learn": KillSwitch(
                name="auto_learn",
                env_var="AUTO_LEARN_ENABLED",
                default=True,
                description="Autonomous internet learning engine",
            ),
            "live_trading": KillSwitch(
                name="live_trading",
                env_var="LIVE_TRADING_ENABLED",
                default=False,
                description="Live (non-paper) trading capability",
                kill_file=os.path.join(KILL_SWITCH_DIR, "LIVE_TRADING_KILLED"),
            ),
            "email_send": KillSwitch(
                name="email_send",
                env_var="EMAIL_SEND_ENABLED",
                default=True,
                description="Outgoing email send capability",
            ),
        }

    def _ensure_dir(self):
        """Create kill switch directory if needed."""
        try:
            os.makedirs(KILL_SWITCH_DIR, exist_ok=True)
        except Exception as e:
            logger.error(f"Cannot create kill switch dir: {e}")

    def is_enabled(self, switch_name: str) -> bool:
        """Check if a specific switch is enabled."""
        switch = self.switches.get(switch_name)
        if switch is None:
            logger.warning(f"Unknown kill switch queried: {switch_name}")
            return False  # Unknown switches default to OFF (safe)
        return switch.is_enabled

    def status_all(self) -> dict:
        """Return status of all switches."""
        result = {}
        for name, switch in self.switches.items():
            enabled = switch.is_enabled
            has_kill_file = bool(switch.kill_file and os.path.isfile(switch.kill_file))
            result[name] = {
                "enabled": enabled,
                "default": switch.default,
                "env_var": switch.env_var,
                "env_value": os.getenv(switch.env_var, "(not set)"),
                "kill_file_active": has_kill_file,
                "description": switch.description,
            }
        return result

    def activate_kill(self, switch_name: str, user_id: str = "") -> dict:
        """
        Activate a kill switch (force system OFF).
        Creates a kill file that persists across restarts.
        """
        switch = self.switches.get(switch_name)
        if switch is None:
            return {"success": False, "error": f"Unknown switch: {switch_name}"}

        if not switch.kill_file:
            return {"success": False, "error": f"Switch '{switch_name}' does not support file-based kill"}

        with self._lock:
            try:
                tmp_path = switch.kill_file + ".tmp"
                with open(tmp_path, "w") as f:
                    f.write(json.dumps({
                        "killed_at": datetime.now(timezone.utc).isoformat(),
                        "killed_by": user_id or "system",
                        "switch": switch_name,
                    }))
                    # S190 L7: fsync the kill-file body before the atomic rename.
                    # A kernel crash between buffered-write and os.replace could
                    # leave the kill file empty, which means the switch wouldn't
                    # re-arm on restart. Cheap durability win for a safety
                    # mechanism that NEEDS to survive crashes.
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, switch.kill_file)
                logger.warning(f"Kill switch ACTIVATED: {switch_name} by user {user_id}")
                try:
                    from agent.safety.audit_log import audit_kill_switch
                    audit_kill_switch(switch_name, activated=True, user_id=user_id)
                except Exception as e:
                    logger.debug(f"Fire-and-forget activate_kill audit failed: {e}")
                return {"success": True, "switch": switch_name, "state": "KILLED"}
            except Exception as e:
                logger.error(f"Failed to activate kill switch {switch_name}: {e}")
                return {"success": False, "error": str(e)}

    def deactivate_kill(self, switch_name: str, user_id: str = "") -> dict:
        """
        Deactivate a kill switch (remove kill file, revert to env var control).
        """
        switch = self.switches.get(switch_name)
        if switch is None:
            return {"success": False, "error": f"Unknown switch: {switch_name}"}

        if not switch.kill_file:
            return {"success": False, "error": f"Switch '{switch_name}' does not have a kill file"}

        with self._lock:
            try:
                if os.path.isfile(switch.kill_file):
                    os.remove(switch.kill_file)
                logger.info(f"Kill switch deactivated: {switch_name} by user {user_id}")
                try:
                    from agent.safety.audit_log import audit_kill_switch
                    audit_kill_switch(switch_name, activated=False, user_id=user_id)
                except Exception as e:
                    logger.debug(f"Fire-and-forget deactivate_kill audit failed: {e}")
                return {"success": True, "switch": switch_name, "state": "ACTIVE" if switch.is_enabled else "DISABLED_BY_ENV"}
            except Exception as e:
                logger.error(f"Failed to deactivate kill switch {switch_name}: {e}")
                return {"success": False, "error": str(e)}

    def emergency_halt(self, user_id: str = "") -> dict:
        """
        NUCLEAR OPTION: Activate ALL kill switches simultaneously.
        Used for emergency situations only.
        """
        results = {}
        for name, switch in self.switches.items():
            if switch.kill_file:
                results[name] = self.activate_kill(name, user_id)
        logger.critical(f"EMERGENCY HALT executed by user {user_id}")
        try:
            from agent.safety.audit_log import audit_security_event
            audit_security_event("emergency_halt", f"Emergency halt executed by {user_id}", details={"results": results}, severity="CRITICAL")
        except Exception as e:
            logger.debug(f"Fire-and-forget emergency_halt audit failed: {e}")
        return results

    def format_status(self) -> str:
        """Format all switch statuses for Telegram display."""
        lines = ["System Kill Switches:\n"]
        for name, status in self.status_all().items():
            icon = "🟢" if status["enabled"] else "🔴"
            killed = " ⚠️ KILLED" if status["kill_file_active"] else ""
            lines.append(f"{icon} {name} — {'ON' if status['enabled'] else 'OFF'}{killed}")
            lines.append(f"    {status['description']}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_registry: Optional[KillSwitchRegistry] = None
_init_lock = threading.Lock()


def get_kill_registry() -> KillSwitchRegistry:
    """Get or create the singleton kill switch registry."""
    global _registry
    if _registry is None:
        with _init_lock:
            if _registry is None:
                _registry = KillSwitchRegistry()
    return _registry
