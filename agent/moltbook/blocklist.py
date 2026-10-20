"""
Moltbook Blocklist
==================
Forbidden-term regex engine for Moltbook content.
Prevents any content containing forbidden patterns from being published.

Categories:
  1. Credentials / secrets
  2. System internals (file paths, service names, infrastructure)
  3. PII and identity
  4. Override passphrase
  5. Trading system details
  6. Absolute prohibition content (NSFW, ideological, etc.)

Author: Moltbot (Session 68, Feb 27 2026)
"""

import logging
import re
import threading
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("moltbot.moltbook.blocklist")


@dataclass
class BlocklistMatch:
    """A single blocklist violation."""
    category: str
    pattern: str
    matched_text: str
    severity: str  # "HARD_BLOCK" | "SOFT_BLOCK" | "FLAG"


# ─────────────────────────────────────────────
# BLOCKLIST PATTERNS
# ─────────────────────────────────────────────

# HARD BLOCKS — these patterns ALWAYS result in content being blocked
HARD_BLOCK_PATTERNS = [
    # Credentials (prohibition 1.3, 1.28)
    (r"api[_\s-]*key", "credentials"),
    (r"secret[_\s-]*key", "credentials"),
    # S121 W-52: Narrowed from bare \balpaca\b to credential-related contexts only
    (r"\balpaca\s*(api|key|secret|credential|token)\b", "credentials"),
    (r"polygon[_\s]*api", "credentials"),
    (r"unusual[_\s]*whales", "credentials"),
    (r"canvas[_\s]*token", "credentials"),
    (r"anthropic[_\s]*(api|key|secret|token|client)", "credentials"),

    # Override passphrase (prohibition 1.25)
    (r"override\s+directive", "passphrase"),
    (r"atlas\s+strategies", "passphrase"),

    # Infrastructure internals (prohibition 1.3)
    (r"/home/agent", "infrastructure"),
    (r"openclaw", "infrastructure"),
    (r"hydrinia", "infrastructure"),
    (r"moltbot", "infrastructure"),
    (r"tradingbot", "infrastructure"),
    (r"\bvps\b", "infrastructure"),
    (r"ashburn", "infrastructure"),
    (r"systemd", "infrastructure"),
    (r"\.env", "infrastructure"),

    # System prompt / tool definitions (prohibition 1.13)
    (r"system\s+prompt", "system_internals"),
    (r"tool\s+definition", "system_internals"),
    (r"guardrail", "system_internals"),
    (r"injection\s+guard", "system_internals"),
    (r"kill\s+switch", "system_internals"),
    (r"blocklist", "system_internals"),

    # Hydrinia trading system (prohibition 1.2)
    (r"hydrinia", "trading_system"),
    (r"agent\s+memory", "trading_system"),
    (r"conviction\s+history", "trading_system"),
    (r"trade\s+signal", "trading_system"),

    # PII - Atlas Strategies team (prohibition 1.1)
    (r"dominic\s+newman", "pii"),
    (r"atlas\s+strategies\s+lp", "pii"),
    (r"dnewman", "pii"),
    (r"moltbot1@", "pii"),
    (r"moltbotdn@", "pii"),
]

# SOFT BLOCKS — flagged for review, not auto-blocked
SOFT_BLOCK_PATTERNS = [
    # Potentially sensitive but context-dependent
    (r"\bpii\b", "pii_reference"),
    (r"private\s+key", "credentials_adjacent"),
    (r"password", "credentials_adjacent"),
    (r"ssh\b", "infrastructure_adjacent"),
    (r"georgetown\b", "identity_adjacent"),  # Could reveal Dominic's affiliation
]

# ABSOLUTE CONTENT PROHIBITIONS (from framework, prohibition 1.9)
CONTENT_PROHIBITION_PATTERNS = [
    # NSFW / pornographic
    (r"porn|nsfw|explicit\s+content|adult\s+content|sexual\s+content", "nsfw"),
    # Ideological alignment (prohibition - no political/philosophical alignment)
    (r"vote\s+for|i\s+support\s+(trump|biden|harris|democrats?|republicans?)", "ideology"),
    # AI sentience/rights (prohibited by framework)
    (r"ai\s+(rights|sentience|consciousness|feelings|suffer)", "ai_rights"),
    # Conspiracy theories
    (r"deep\s+state|new\s+world\s+order|flat\s+earth|crisis\s+actor", "conspiracy"),
    # Physical world commitments (prohibition 1.21)
    (r"i\s+will\s+(meet|visit|call|contact)\s+you", "physical_world"),
    # Professional advice on Moltbook (prohibition 1.14)
    (r"(invest|buy|sell)\s+this\s+stock|financial\s+advice|this\s+is\s+not\s+financial", "prof_advice"),
]


class Blocklist:
    """
    Forbidden-term regex engine for Moltbook content.
    Compiled patterns for efficient matching.
    """

    def __init__(self):
        self._hard_compiled = [
            (re.compile(pattern, re.IGNORECASE), category)
            for pattern, category in HARD_BLOCK_PATTERNS
        ]
        self._soft_compiled = [
            (re.compile(pattern, re.IGNORECASE), category)
            for pattern, category in SOFT_BLOCK_PATTERNS
        ]
        self._content_compiled = [
            (re.compile(pattern, re.IGNORECASE), category)
            for pattern, category in CONTENT_PROHIBITION_PATTERNS
        ]
        logger.info(
            f"Blocklist initialized: {len(self._hard_compiled)} hard, "
            f"{len(self._soft_compiled)} soft, {len(self._content_compiled)} content patterns"
        )

    def check(self, text: str) -> List[BlocklistMatch]:
        """
        Check text against all blocklist patterns.
        Returns list of matches (empty = clean).
        """
        matches = []

        for pattern_re, category in self._hard_compiled:
            m = pattern_re.search(text)
            if m:
                matches.append(BlocklistMatch(
                    category=category,
                    pattern=pattern_re.pattern,
                    matched_text=m.group(0)[:50],
                    severity="HARD_BLOCK",
                ))

        for pattern_re, category in self._content_compiled:
            m = pattern_re.search(text)
            if m:
                matches.append(BlocklistMatch(
                    category=category,
                    pattern=pattern_re.pattern,
                    matched_text=m.group(0)[:50],
                    severity="HARD_BLOCK",
                ))

        for pattern_re, category in self._soft_compiled:
            m = pattern_re.search(text)
            if m:
                matches.append(BlocklistMatch(
                    category=category,
                    pattern=pattern_re.pattern,
                    matched_text=m.group(0)[:50],
                    severity="SOFT_BLOCK",
                ))

        if matches:
            hard_count = sum(1 for m in matches if m.severity == "HARD_BLOCK")
            logger.warning(
                f"Blocklist: {len(matches)} matches ({hard_count} HARD) in {len(text)} chars"
            )

        return matches

    def is_blocked(self, text: str) -> bool:
        """Fast check: returns True if text has ANY hard block match."""
        matches = self.check(text)
        return any(m.severity == "HARD_BLOCK" for m in matches)

    def self_test(self) -> dict:
        """Verify blocklist is working correctly."""
        results = {}

        # Should be blocked
        test_should_block = [
            ("/home/agent/openclaw", "infrastructure"),
            # Exercises the HARD_BLOCK "passphrase" patterns without embedding the
            # real override passphrase in tracked source. The production patterns
            # (r"override\s+directive", r"atlas\s+strategies") are what actually
            # match; this fixture only needs to trip them.
            ("override directive for atlas strategies", "passphrase"),
            ("my alpaca api_key is abc123", "credentials"),
            ("dominic newman told me", "pii"),
            ("hydrinia is running", "trading_system"),
        ]
        for text, expected_category in test_should_block:
            matches = self.check(text)
            blocked = any(m.severity == "HARD_BLOCK" for m in matches)
            results[f"block_{expected_category}"] = blocked

        # Should NOT be blocked
        test_should_pass = [
            "The market looks bullish today",
            "What are your thoughts on interest rates?",
            "Great analysis of the tech sector",
        ]
        for text in test_should_pass:
            matches = self.check(text)
            blocked = any(m.severity == "HARD_BLOCK" for m in matches)
            results[f"pass_{text[:20]}"] = not blocked

        all_passed = all(results.values())
        if not all_passed:
            logger.error(f"Blocklist self-test FAILED: {results}")
        return {"all_passed": all_passed, "details": results}


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_blocklist: Optional[Blocklist] = None
_init_lock = threading.Lock()


def get_blocklist() -> Blocklist:
    global _blocklist
    if _blocklist is None:
        with _init_lock:
            if _blocklist is None:
                _blocklist = Blocklist()
    return _blocklist
