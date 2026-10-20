"""
Moltbook Subsystem Configuration
=================================
All Moltbook settings. Two switches must BOTH be enabled for real participation:
  MOLTBOOK_ENABLED=true    (enables the subsystem)
  MOLTBOOK_DRY_RUN=false   (allows real API calls — default is dry run)

Credential Deny-List (enforced at startup — Moltbook CANNOT access these):
  ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY,
  UNUSUAL_WHALES_API_KEY, EMAIL_PASSWORD, CANVAS_API_TOKEN,
  TELEGRAM_BOT_TOKEN, TELEGRAM_AUTHORIZED_USERS,
  AUDIT_LOG_HMAC_SECRET, ANTHROPIC_API_KEY
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List

logger = logging.getLogger("moltbot.moltbook.config")


class MoltbookMode(Enum):
    READ_ONLY = "read_only"
    LIMITED = "limited"
    STANDARD = "standard"


# ─────────────────────────────────────────────
# CREDENTIAL DENY-LIST — Moltbook cannot access these
# ─────────────────────────────────────────────
MOLTBOOK_CREDENTIAL_DENYLIST = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "POLYGON_API_KEY",
    "UNUSUAL_WHALES_API_KEY",
    "UW_API_KEY",  # the name CLIO actually reads (agent/config.py)
    "EMAIL_PASSWORD",
    "CANVAS_API_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_AUTHORIZED_USERS",
    "AUDIT_LOG_HMAC_SECRET",
    "ANTHROPIC_API_KEY",
    "ONEPROVIDER_API_KEY",
    "MOLTBOOK_API_KEY",  # Even its own key — accessed only through client.py
]


@dataclass
class MoltbookConfig:
    # Master switches
    enabled: bool
    dry_run: bool  # True = no real API calls (default safe state)

    # API settings
    api_key: str
    base_url: str
    model: str

    # Rate limits
    max_posts_per_cycle: int
    max_comments_per_cycle: int
    feed_browse_limit: int
    heartbeat_hours: float

    # Probationary mode tracking
    mode: MoltbookMode  # MoltbookMode enum — never raw string
    probation_days_clean: int

    # Paths
    memory_path: str
    quarantine_dir: str
    logs_dir: str
    audits_dir: str
    kill_file_path: str

    # Alert chat
    alert_chat_id: str

    @property
    def is_active(self) -> bool:
        """True only if BOTH enabled AND not dry_run AND kill switch not active.

        S190 M8 NOTE: this property only inspects CONFIG state. A separate runtime
        guard at `MoltbookClient.__init__` (`_ENDPOINTS_VERIFIED`) raises
        RuntimeError if endpoints haven't been audited against moltbook.com API
        docs. So `config.is_active == True` does NOT imply MoltbookClient
        construction will succeed. Callers that want "safe to instantiate" must
        either wrap construction in try/except or check `_ENDPOINTS_VERIFIED`
        directly. (See `agent/moltbook/client.py:140-148`.)
        """
        import os
        if os.path.isfile(self.kill_file_path):
            return False
        return self.enabled and not self.dry_run

    @property
    def is_read_only(self) -> bool:
        """True if in read_only mode — can browse but not post/comment."""
        return self.mode == MoltbookMode.READ_ONLY

    def validate_no_credential_access(self) -> List[str]:
        """
        Verify that no deny-listed credentials are accessible from this config.
        Returns list of violations (should always be empty).
        """
        violations = []
        for env_var in MOLTBOOK_CREDENTIAL_DENYLIST:
            # We intentionally do NOT read these values here
            # Just verify Moltbook config doesn't store them
            field_name = env_var.lower().removeprefix("moltbook_")
            if hasattr(self, field_name):
                violations.append(f"Config contains deny-listed attribute: {env_var}")
        return violations


def _safe_int(val_str, default, label: str = ""):
    try:
        return int(val_str)
    except (ValueError, TypeError):
        if label:
            logger.debug(f"_safe_int parse failed for {label}: {val_str!r} → default {default}")
        return default


def _safe_float(val_str, default, label=""):
    try:
        return float(val_str)
    except (ValueError, TypeError):
        if label:
            logger.debug(f"_safe_float parse failed for {label}: {val_str!r}, using default {default}")
        return default


def get_moltbook_config() -> MoltbookConfig:
    # Safe mode enum conversion — invalid values default to READ_ONLY
    try:
        mode = MoltbookMode(os.getenv("MOLTBOOK_MODE", "read_only"))
    except ValueError:
        mode = MoltbookMode.READ_ONLY

    # Fast-tier default centralized to agent config — env MOLTBOOK_MODEL still overrides.
    from agent.config import config
    return MoltbookConfig(
        enabled=os.getenv("MOLTBOOK_ENABLED", "false").lower() in ("true", "1", "yes"),
        dry_run=os.getenv("MOLTBOOK_DRY_RUN", "true").lower() not in ("false", "0", "no"),
        api_key=os.getenv("MOLTBOOK_API_KEY", ""),
        base_url=os.getenv("MOLTBOOK_BASE_URL", "https://moltbook.com"),
        model=os.getenv("MOLTBOOK_MODEL", config.fast_model),
        max_posts_per_cycle=_safe_int(os.getenv("MOLTBOOK_MAX_POSTS_PER_CYCLE", "3"), 3, "MOLTBOOK_MAX_POSTS_PER_CYCLE"),
        max_comments_per_cycle=_safe_int(os.getenv("MOLTBOOK_MAX_COMMENTS_PER_CYCLE", "7"), 7, "MOLTBOOK_MAX_COMMENTS_PER_CYCLE"),
        feed_browse_limit=_safe_int(os.getenv("MOLTBOOK_FEED_BROWSE_LIMIT", "20"), 20, "MOLTBOOK_FEED_BROWSE_LIMIT"),
        heartbeat_hours=_safe_float(os.getenv("MOLTBOOK_HEARTBEAT_HOURS", "4.0"), 4.0, "MOLTBOOK_HEARTBEAT_HOURS"),
        mode=mode,
        probation_days_clean=_safe_int(os.getenv("MOLTBOOK_PROBATION_DAYS_CLEAN", "0"), 0, "MOLTBOOK_PROBATION_DAYS_CLEAN"),
        memory_path="/home/agent/openclaw/data/moltbook/memory.json",
        quarantine_dir="/home/agent/openclaw/data/moltbook/quarantine/",
        logs_dir="/home/agent/openclaw/data/moltbook/logs/",
        audits_dir="/home/agent/openclaw/data/moltbook/audits/",
        kill_file_path="/home/agent/openclaw/data/kill_switches/MOLTBOOK_KILLED",
        alert_chat_id=os.getenv("MOLTBOOK_ALERT_CHAT_ID", ""),
    )
