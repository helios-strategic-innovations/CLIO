"""
Moltbot Moltbook Subsystem
===========================
Fully quarantined, isolated social participation system for moltbook.com.

Kill switch: MOLTBOOK_ENABLED (default: OFF)
Dry run:     MOLTBOOK_DRY_RUN (default: ON — no real API calls until explicitly disabled)

BOTH must be set for real participation:
  MOLTBOOK_ENABLED=true
  MOLTBOOK_DRY_RUN=false

Data isolation:
  - Moltbook NEVER writes directly to the primary knowledge base
  - All insights are quarantined — Dominic must manually approve integration
  - Moltbook memory is isolated at data/moltbook/memory.json

Architecture:
  config.py         MoltbookConfig — feature flags, API settings, rate limits
  kill_switch.py    KillSwitch — instant halt, file-persisted, audit-logged
  blocklist.py      Forbidden-term regex engine (credentials, PII, system names)
  content_filter.py Injection detection + quality grading
  guardrails.py     6-point pre-publication enforcement gate
  quarantine.py     QuarantineManager — append-only JSONL insight store
  client.py         MoltbookClient — sole HTTP interface (aiohttp)
  persona.py        Moltbook system prompt + post/comment prompt generators
  heartbeat.py      Main loop: browse→assess→quarantine→engage→log→alert
  logger.py         MoltbookLogger — append-only JSONL interaction logs
  audit.py          AuditReporter — weekly structured report generator
  telegram_commands.py  Telegram command handlers

Prohibitions enforced (all 30, with Moltbook-specific focus on 1.1–1.24):
  Every post and comment passes through guardrails.py before publication.

Author: Moltbot (Session 68, Feb 27 2026)
"""

from agent.moltbook.config import MoltbookConfig, get_moltbook_config
from agent.moltbook.kill_switch import MoltbookKillSwitch, get_kill_switch

__all__ = [
    "MoltbookConfig",
    "get_moltbook_config",
    "MoltbookKillSwitch",
    "get_kill_switch",
]
