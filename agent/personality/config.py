"""
Personality Engine Configuration
"""

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("moltbot.personality.config")

_OVERRIDE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "personality_override.json")


def _safe_int(val, default: int, label: str = "") -> int:
    """Parse int from env var, return default on failure."""
    try:
        return int(val)
    except (ValueError, TypeError):
        if label:
            logger.debug(f"_safe_int parse failed for {label}: {val!r} → default {default}")
        return default


@dataclass
class PersonalityConfig:
    enabled: bool
    daily_trait_limit: int
    context_max_chars: int
    registry_path: str
    history_path: str
    reading_log_path: str
    summary_path: str


def _load_personality_override() -> bool | None:
    """Load persisted personality toggle override, if it exists. Returns None if no override."""
    try:
        if os.path.isfile(_OVERRIDE_PATH):
            with open(_OVERRIDE_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "enabled" in data:
                return bool(data["enabled"])
    except Exception as e:
        import logging
        logging.getLogger("openclaw.personality.config").debug(f"Personality override load failed: {e}")
    return None


def get_personality_config() -> PersonalityConfig:
    env_enabled = os.getenv("PERSONALITY_ENABLED", "false").lower() in ("true", "1", "yes")
    # Persisted override from /personality_on or /personality_off takes precedence over env var
    override = _load_personality_override()
    enabled = override if override is not None else env_enabled

    return PersonalityConfig(
        enabled=enabled,
        daily_trait_limit=_safe_int(os.getenv("PERSONALITY_DAILY_TRAIT_LIMIT", "1"), 1, "PERSONALITY_DAILY_TRAIT_LIMIT"),
        context_max_chars=_safe_int(os.getenv("PERSONALITY_CONTEXT_MAX_CHARS", "800"), 800, "PERSONALITY_CONTEXT_MAX_CHARS"),
        registry_path="/home/agent/openclaw/data/personality/trait_registry.json",
        history_path="/home/agent/openclaw/data/personality/trait_history.jsonl",
        reading_log_path="/home/agent/openclaw/data/personality/reading_log.json",
        summary_path="/home/agent/openclaw/data/personality/personality_summary.md",
    )
