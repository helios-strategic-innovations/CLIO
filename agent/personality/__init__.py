"""
Moltbot Personality Engine
==========================
A persistent, cumulative trait registry that shapes Moltbot's voice and approach.

Kill switch: PERSONALITY_ENABLED (default: OFF)
When OFF: trait registry still accumulates, context injection is simply skipped.
When ON: confirmed traits are injected into every system prompt.

Architecture:
  - config.py        PersonalityConfig dataclass
  - registry.py      TraitRegistry — append-only trait store
  - engine.py        PersonalityEngine — context generation, trait application
  - telegram_commands.py  /personality, /trait_propose, /trait_confirm, /trait_deny

Author: Moltbot (Session 68, Feb 27 2026)
"""

from agent.personality.config import PersonalityConfig, get_personality_config
from agent.personality.registry import TraitRegistry, Trait, get_trait_registry
from agent.personality.engine import PersonalityEngine, get_personality_engine, get_personality_context

__all__ = [
    "PersonalityConfig",
    "get_personality_config",
    "TraitRegistry",
    "Trait",
    "get_trait_registry",
    "PersonalityEngine",
    "get_personality_engine",
    "get_personality_context",
]
