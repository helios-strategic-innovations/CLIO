"""
Moltbot Safety Protocols Package
=================================
Five-layer safety infrastructure:

  Layer 1: Harm Threshold Engine    — harm scoring before any output
  Layer 2: Injection Guard          — extends security.py (in security.py)
  Layer 3: Kill Switch Registry     — unified switch manager
  Layer 4: Reset Manager            — clean state protocols
  Layer 5: Immutable Audit Log      — forensic-grade append-only record

All layers are ALWAYS-ON and cannot be disabled by any kill switch.

Author: Moltbot (Session 68, Feb 27 2026)
"""

from agent.safety.audit_log import (
    ImmutableAuditLog,
    AuditEntry,
    AuditEventType,
    get_audit_log,
    audit_tool_execution,
    audit_security_event,
    audit_kill_switch,
)
from agent.safety.harm_threshold import (
    HarmScorer,
    HarmAssessment,
    HarmLevel,
    get_harm_scorer,
)
from agent.safety.kill_registry import (
    KillSwitchRegistry,
    KillSwitch,
    get_kill_registry,
)
from agent.safety.reset_manager import ResetManager

__all__ = [
    # Audit log
    "ImmutableAuditLog",
    "AuditEntry",
    "AuditEventType",
    "get_audit_log",
    "audit_tool_execution",
    "audit_security_event",
    "audit_kill_switch",
    # Harm threshold
    "HarmScorer",
    "HarmAssessment",
    "HarmLevel",
    "get_harm_scorer",
    # Kill switches
    "KillSwitchRegistry",
    "KillSwitch",
    "get_kill_registry",
    # Reset
    "ResetManager",
]
