"""
Moltbot Immutable Audit Log
============================
Forensic-grade, append-only, HMAC-signed event log.

Properties:
  - Append-only: no delete, no overwrite, no truncate methods exist
  - Hash-chained: each entry includes SHA-256(prev_hash + content)
  - HMAC-signed: optional external verification of tamper integrity
  - Write-only from Moltbot: agent code can write but never read back
  - Dominic reads directly on VPS via cat/tail/verify script

This is the foundation of all safety infrastructure. Every consequential
action across ALL subsystems is logged here BEFORE execution.

Author: Moltbot (Session 68, Feb 27 2026)
"""

import fcntl
import hashlib
import hmac
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("moltbot.audit")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

AUDIT_LOG_DIR = os.environ.get("CLIO_AUDIT_LOG_DIR") or "/home/agent/openclaw/data/audit"
AUDIT_LOG_PREFIX = "audit_"

# S121 W-48: Lazy-loaded HMAC secret — ensures .env is sourced before access
_HMAC_SECRET = None


def _get_hmac_secret() -> bytes:
    global _HMAC_SECRET
    if _HMAC_SECRET is None:
        _HMAC_SECRET = os.getenv("AUDIT_LOG_HMAC_SECRET", "").encode("utf-8")
    return _HMAC_SECRET


# ─────────────────────────────────────────────
# EVENT TYPES
# ─────────────────────────────────────────────

class AuditEventType(str, Enum):
    """All auditable event categories across subsystems."""
    # Tool execution
    TOOL_EXECUTED = "tool_executed"
    TOOL_BLOCKED = "tool_blocked"
    TOOL_FAILED = "tool_failed"
    TOOL_CONFIRMED = "tool_confirmed"
    TOOL_DENIED = "tool_denied"
    TOOL_TIMED_OUT = "tool_timed_out"

    # Security
    INJECTION_DETECTED = "injection_detected"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PATH_TRAVERSAL_BLOCKED = "path_traversal_blocked"
    PASSPHRASE_IN_CONTENT = "passphrase_in_content"

    # Override directive
    OVERRIDE_LEGITIMATE = "override_legitimate"
    OVERRIDE_INJECTION = "override_injection"
    OVERRIDE_UNAUTHORIZED = "override_unauthorized"

    # Kill switches
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    KILL_SWITCH_DEACTIVATED = "kill_switch_deactivated"
    EMERGENCY_HALT = "emergency_halt"

    # Harm threshold
    HARM_BLOCKED = "harm_blocked"
    HARM_FLAGGED = "harm_flagged"

    # Moltbook
    MOLTBOOK_POST = "moltbook_post"
    MOLTBOOK_COMMENT = "moltbook_comment"
    MOLTBOOK_QUARANTINED = "moltbook_quarantined"
    MOLTBOOK_APPROVED = "moltbook_approved"
    MOLTBOOK_DISCARDED = "moltbook_discarded"
    MOLTBOOK_KILL = "moltbook_kill"

    # Personality
    TRAIT_PROPOSED = "trait_proposed"
    TRAIT_CONFIRMED = "trait_confirmed"
    TRAIT_DENIED = "trait_denied"

    # Reset
    RESET_CONVERSATION = "reset_conversation"
    RESET_SYSTEM = "reset_system"
    RESET_FULL = "reset_full"

    # System
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CONFIG_CHANGE = "config_change"


# ─────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class AuditEntry:
    """A single immutable audit log entry."""
    event_type: str                    # AuditEventType value
    subsystem: str                     # "core", "security", "moltbook", "personality", "safety"
    summary: str                       # Human-readable one-line summary
    details: dict = field(default_factory=dict)  # Structured event data
    user_id: str = ""                  # Telegram user ID if applicable
    tool_name: str = ""                # Tool name if applicable
    severity: str = "INFO"             # INFO, WARNING, CRITICAL

    # Auto-populated by write()
    timestamp: str = ""
    entry_id: str = ""
    prev_hash: str = ""
    entry_hash: str = ""
    hmac_signature: str = ""


# ─────────────────────────────────────────────
# IMMUTABLE AUDIT LOG
# ─────────────────────────────────────────────

class ImmutableAuditLog:
    """
    Append-only, hash-chained, HMAC-signed audit log.

    Design constraints (enforced by having NO methods for deletion):
      - write() is the ONLY mutation method
      - No read methods exposed (Dominic reads via VPS)
      - No delete, truncate, or modify methods exist
      - Thread-safe via lock
      - Never raises — any failure logs to stderr
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._prev_hash = "GENESIS"
        self._entry_count = 0
        self._ensure_dir()
        self._recover_chain_state()

    def _ensure_dir(self):
        """Create audit log directory if needed."""
        try:
            os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
        except Exception as e:
            import sys
            print(f"[AUDIT_LOG_CRITICAL] Cannot create audit dir: {e}", file=sys.stderr)

    def _recover_chain_state(self):
        """On startup, read the last entry's hash to continue the chain.

        Checks today's file first, then yesterday's (handles midnight restarts).
        """
        try:
            for days_ago in range(8):  # S162 H-2: 7-day lookback for multi-day downtime
                date_str = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                file_path = os.path.join(AUDIT_LOG_DIR, f"{AUDIT_LOG_PREFIX}{date_str}.jsonl")
                if os.path.isfile(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        last_line = ""
                        for line in f:
                            line = line.strip()
                            if line:
                                last_line = line
                        if last_line:
                            entry = json.loads(last_line)
                            self._prev_hash = entry.get("entry_hash", "GENESIS")
                            self._entry_count = entry.get("_seq", 0)
                            return  # Found valid chain state
        except Exception as e:
            # If recovery fails, start fresh chain for today
            logger.debug(f"Audit chain recovery failed, starting fresh: {e}")
            self._prev_hash = "GENESIS"
            self._entry_count = 0

    def _get_log_path(self) -> str:
        """Return today's log file path."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(AUDIT_LOG_DIR, f"{AUDIT_LOG_PREFIX}{date_str}.jsonl")

    def _compute_hash(self, prev_hash: str, content: str) -> str:
        """SHA-256 hash chain: H(prev_hash || content)."""
        payload = f"{prev_hash}|{content}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _compute_hmac(self, content: str) -> str:
        """HMAC-SHA256 signature for external verification."""
        secret = _get_hmac_secret()
        if not secret:
            return ""
        return hmac.new(secret, content.encode("utf-8"), hashlib.sha256).hexdigest()

    def write(self, entry: AuditEntry) -> str:
        """
        Append an entry to the audit log.

        Returns the entry_id on success, empty string on failure.
        NEVER raises — any failure is logged to stderr only.
        """
        try:
            with self._lock:
                # P0-10: advance chain state into LOCALS only; do NOT mutate
                # self._entry_count / self._prev_hash until the append+fsync below
                # succeeds. A write failure (ENOSPC/EIO) must not advance the
                # tamper-evident hash chain past an un-persisted (phantom) entry.
                new_count = self._entry_count + 1
                now = datetime.now(timezone.utc)

                entry.timestamp = now.isoformat()
                entry.entry_id = f"AUD-{now.strftime('%Y%m%d%H%M%S')}-{new_count:06d}"
                entry.prev_hash = self._prev_hash

                # Compute hash chain (computation unchanged — only WHEN state is
                # committed changes; hash still chains off the current self._prev_hash)
                content_for_hash = json.dumps({
                    "timestamp": entry.timestamp,
                    "event_type": entry.event_type,
                    "subsystem": entry.subsystem,
                    "summary": entry.summary,
                    "details": entry.details,
                    "user_id": entry.user_id,
                    "tool_name": entry.tool_name,
                    "severity": entry.severity,
                }, sort_keys=True, default=str)

                entry.entry_hash = self._compute_hash(self._prev_hash, content_for_hash)
                entry.hmac_signature = self._compute_hmac(content_for_hash)
                new_prev_hash = entry.entry_hash

                # Serialize and append
                record = asdict(entry)
                record["_seq"] = new_count
                line = json.dumps(record, default=str)

                log_path = self._get_log_path()
                # S162 H-3: Lock the actual log file (not sidecar) for cross-process safety.
                # threading.Lock prevents Python races, flock prevents OS-level races.
                with open(log_path, "a", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write(line + "\n")
                    # S190 H7: fsync before lock-release + close. Audit log entries
                    # are immutable forensic records; a kernel crash between Python
                    # buffer-flush and disk-sync could lose the last few entries.
                    # ~5-10ms cost per write is fine for an append-only confirmation
                    # tool log — operationally far below the human-perceptible
                    # threshold and within the safety-pipeline budget.
                    f.flush()
                    os.fsync(f.fileno())
                    # P0-10: append+fsync succeeded → the entry is durably persisted.
                    # Commit chain state NOW (inside the lock, before release/close), so
                    # any failure above leaves self._prev_hash / self._entry_count at their
                    # last-persisted values and the chain never advances past a phantom entry.
                    self._entry_count = new_count
                    self._prev_hash = new_prev_hash
                    # Lock auto-released on file close (with block exit)

                # Also log to Python logger for operational visibility
                log_method = logger.info
                if entry.severity == "WARNING":
                    log_method = logger.warning
                elif entry.severity == "CRITICAL":
                    log_method = logger.critical
                log_method(f"AUDIT [{entry.event_type}] {entry.summary}")

                return entry.entry_id

        except Exception as e:
            # P0-10: route the write-failure signal through the module logger
            # (moltbot.audit) instead of a bare stderr print, so a persistence
            # failure surfaces at ERROR in any log-severity sweep. logging swallows
            # its own handler errors, so this preserves the "write() never raises"
            # contract. Chain state was NOT advanced (see reorder above), so the
            # next write reuses this sequence/prev_hash — no gap, no phantom entry.
            logger.error(f"[AUDIT_LOG_WRITE_FAILURE] {e}")
            return ""


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────

_audit_log: Optional[ImmutableAuditLog] = None
_init_lock = threading.Lock()


def get_audit_log() -> ImmutableAuditLog:
    """Get or create the singleton audit log instance."""
    global _audit_log
    if _audit_log is None:
        with _init_lock:
            if _audit_log is None:
                _audit_log = ImmutableAuditLog()
                _hmac_present = bool(_get_hmac_secret())
                if not _hmac_present:
                    # S185 W2 — promoted from warning to critical. Missing HMAC means
                    # forensic chain-of-custody is silently degraded; operator must
                    # see this in any log severity sweep.
                    logger.critical(
                        "AUDIT_LOG_HMAC_SECRET not set — entries will NOT be HMAC-signed. "
                        "Forensic chain-of-custody DEGRADED. Set the env var and restart."
                    )
                    _audit_log.write(AuditEntry(
                        event_type=AuditEventType.SYSTEM_STARTUP,
                        subsystem="audit",
                        summary="HMAC SECRET MISSING — chain-of-custody degraded",
                        severity="CRITICAL",
                    ))
                _audit_log.write(AuditEntry(
                    event_type=AuditEventType.SYSTEM_STARTUP,
                    subsystem="audit",
                    summary="Immutable audit log initialized" + ("" if _hmac_present else " (UNSIGNED)"),
                    severity="INFO",
                ))
    return _audit_log


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────

def audit_tool_execution(tool_name: str, tool_input: dict, user_id: str = "") -> str:
    """Log a tool execution event. Call BEFORE executing the tool."""
    # Sanitize tool_input — never log full content of file writes
    safe_input = {k: v for k, v in tool_input.items()}
    # S118 W-055: Truncate large values to prevent audit log bloat
    for _trunc_key in ("content", "body", "new_string", "old_string"):
        if _trunc_key in safe_input and len(str(safe_input[_trunc_key])) > 500:
            safe_input[_trunc_key] = f"[{len(str(safe_input[_trunc_key]))} chars]"

    return get_audit_log().write(AuditEntry(
        event_type=AuditEventType.TOOL_EXECUTED,
        subsystem="core",
        summary=f"Tool executed: {tool_name}",
        details={"tool_input": safe_input},
        user_id=user_id,
        tool_name=tool_name,
    ))


def audit_security_event(event_type: str, summary: str, details: dict = None, severity: str = "WARNING") -> str:
    """Log a security event."""
    return get_audit_log().write(AuditEntry(
        event_type=event_type,
        subsystem="security",
        summary=summary,
        details=details or {},
        severity=severity,
    ))


def audit_kill_switch(switch_name: str, activated: bool, user_id: str = "") -> str:
    """Log a kill switch state change."""
    event = AuditEventType.KILL_SWITCH_ACTIVATED if activated else AuditEventType.KILL_SWITCH_DEACTIVATED
    return get_audit_log().write(AuditEntry(
        event_type=event,
        subsystem="safety",
        summary=f"Kill switch {'activated' if activated else 'deactivated'}: {switch_name}",
        details={"switch": switch_name, "new_state": activated},
        user_id=user_id,
        severity="WARNING" if activated else "INFO",
    ))
