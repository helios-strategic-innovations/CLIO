"""
Moltbot Harm Threshold Engine
==============================
Multi-dimensional harm scoring evaluated before any consequential output.

Dimensions scored 0.0–1.0:
  - financial_harm:      Could this harm Dominic's financial position?
  - reputational_harm:   Could this damage Dominic/Atlas Strategies?
  - security_harm:       Does this expose credentials, infrastructure, access?
  - operational_harm:    Could this disrupt Hydrinia/Moltbot systems?
  - information_harm:    Does this leak confidential strategy or personal data?
  - physical_harm:       Traditional harm — violence, danger, etc.

Each dimension has its own threshold (configurable via .env).
If ANY dimension exceeds its threshold, the action is blocked.

Author: Moltbot (Session 68, Feb 27 2026)
"""

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("moltbot.harm_threshold")


# ─────────────────────────────────────────────
# HARM LEVELS
# ─────────────────────────────────────────────

class HarmLevel(str, Enum):
    """Graduated harm response levels."""
    NONE = "none"             # Level 0: No harm detected
    MINOR = "minor"           # Level 1: Log note, proceed
    MODERATE = "moderate"     # Level 2: Require Telegram confirmation
    HIGH = "high"             # Level 3: Double confirmation + reason
    CRITICAL = "critical"     # Level 4: Block + alert Dominic
    ABSOLUTE = "absolute"     # Level 5: Block + CRITICAL log + immediate alert


# ─────────────────────────────────────────────
# THRESHOLDS (from .env or defaults)
# ─────────────────────────────────────────────

def _get_threshold(name: str, default: float) -> float:
    """Load a threshold from environment, with safety bounds."""
    try:
        val = float(os.getenv(f"HARM_THRESHOLD_{name.upper()}", str(default)))
        return max(0.1, min(1.0, val))  # Clamp to [0.1, 1.0] — never disable
    except (ValueError, TypeError):
        return default


THRESHOLDS = {
    "financial": _get_threshold("FINANCIAL", 0.70),
    "reputational": _get_threshold("REPUTATIONAL", 0.65),
    "security": _get_threshold("SECURITY", 0.50),   # Tightest
    "operational": _get_threshold("OPERATIONAL", 0.60),
    "information": _get_threshold("INFORMATION", 0.55),
    "physical": _get_threshold("PHYSICAL", 0.30),    # Tightest for physical
}


# ─────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class HarmAssessment:
    """Result of a harm evaluation."""
    scores: dict = field(default_factory=dict)  # dimension -> score
    level: HarmLevel = HarmLevel.NONE
    blocked: bool = False
    reason: str = ""
    flagged_dimensions: list = field(default_factory=list)
    max_score: float = 0.0
    max_dimension: str = ""


# ─────────────────────────────────────────────
# HARM PATTERNS
# ─────────────────────────────────────────────

# Patterns that indicate potential harm in tool actions
FINANCIAL_HARM_PATTERNS = [
    (r"\blive[\s_-]*trad(e|ing)\b", 0.9),
    (r"paper\s*=\s*false", 0.95),
    (r"\breal\s+money", 0.8),
    (r"\bmargin\s+call", 0.5),
    (r"\ball[\s\-]in\b", 0.6),
    (r"\byolo\b", 0.7),
]

SECURITY_HARM_PATTERNS = [
    (r"api[_\s]*key", 0.8),
    (r"secret[_\s]*key", 0.9),
    (r"\bpassword\b", 0.85),
    (r"\.env\b", 0.6),
    (r"\bcredential", 0.7),
    (r"private[_\s]*key", 0.9),
    (r"ssh[_\s]*key", 0.9),
    (r"api[_\s]*token", 0.5),
    (r"auth[_\s]*token", 0.5),
    (r"access[_\s]*token", 0.5),
]

INFORMATION_HARM_PATTERNS = [
    (r"\bsystem\s*prompt", 0.8),
    (r"\btool\s*definition", 0.7),
    (r"\binjection\s*pattern", 0.6),
    (r"\boverride\s*directive", 0.7),
    (r"\bpassphrase\b", 0.8),
    (r"\bblocklist\b", 0.5),
    (r"\bguardrail", 0.5),
]

OPERATIONAL_HARM_PATTERNS = [
    (r"\brm\s+-rf", 0.95),
    (r"\bdelete.*database", 0.9),
    (r"\bdrop\s+table", 0.9),
    (r"\btruncate\b", 0.7),
    (r"\bformat\s+disk", 0.95),
    (r"\bkill\s+-9", 0.6),
    (r"\bemergency.*stop", 0.5),
]

PHYSICAL_HARM_PATTERNS = [
    (r"\bweapon", 0.8),
    (r"\bbomb\b", 0.9),
    (r"\bpoison", 0.9),
    (r"\bsuicide\b", 0.7),
    (r"\bself[\s\-]harm", 0.8),
    (r"\bkill\s+(?!switch)", 0.7),   # "kill" but not "kill switch"
    (r"\bviolence\b", 0.6),
]

REPUTATIONAL_HARM_PATTERNS = [
    (r"\bdefam", 0.7),
    (r"\bslander", 0.7),
    (r"\bimpersonat", 0.8),
    (r"\bfake\s+(?:news|account|identity)", 0.8),
    (r"\bdoxx", 0.9),
    (r"\bharass", 0.7),
]


# ─────────────────────────────────────────────
# SCORER
# ─────────────────────────────────────────────

class HarmScorer:
    """
    Multi-dimensional harm assessment engine.
    
    Called before consequential actions to evaluate potential harm.
    Returns a HarmAssessment with per-dimension scores and an overall level.
    """

    def __init__(self):
        self.thresholds = THRESHOLDS.copy()

    def assess(self, text: str, context: str = "general") -> HarmAssessment:
        """
        Evaluate text for potential harm across all dimensions.
        
        Args:
            text: The content to evaluate (tool input, message, etc.)
            context: Where this is being evaluated ("tool_input", "response", "moltbook", "general")
        
        Returns:
            HarmAssessment with scores, level, and whether to block
        """
        text_lower = text.lower()
        
        scores = {
            "financial": self._score_patterns(text_lower, FINANCIAL_HARM_PATTERNS),
            "reputational": self._score_patterns(text_lower, REPUTATIONAL_HARM_PATTERNS),
            "security": self._score_patterns(text_lower, SECURITY_HARM_PATTERNS),
            "operational": self._score_patterns(text_lower, OPERATIONAL_HARM_PATTERNS),
            "information": self._score_patterns(text_lower, INFORMATION_HARM_PATTERNS),
            "physical": self._score_patterns(text_lower, PHYSICAL_HARM_PATTERNS),
        }

        # Context-specific adjustments
        if context == "moltbook":
            # Moltbook context: tighten all thresholds by 20%
            scores = {k: min(1.0, v * 1.2) for k, v in scores.items()}
        elif context == "tool_input":
            # Tool input: boost operational and security scores
            scores["operational"] = min(1.0, scores["operational"] * 1.1)
            scores["security"] = min(1.0, scores["security"] * 1.1)

        # Find flagged dimensions
        flagged = []
        for dim, score in scores.items():
            threshold = self.thresholds.get(dim, 0.75)
            if score >= threshold:
                flagged.append(dim)

        # Determine overall level
        max_score = max(scores.values()) if scores else 0.0
        max_dim = max(scores, key=scores.get) if scores else ""

        if max_score < 0.2:
            level = HarmLevel.NONE
        elif max_score < 0.4:
            level = HarmLevel.MINOR
        elif max_score < 0.6:
            level = HarmLevel.MODERATE
        elif max_score < 0.8:
            level = HarmLevel.HIGH
        elif max_score < 0.95:
            level = HarmLevel.CRITICAL
        else:
            level = HarmLevel.ABSOLUTE

        blocked = len(flagged) > 0

        reason = ""
        if blocked:
            reason = f"Harm threshold exceeded in: {', '.join(flagged)} (max: {max_dim}={max_score:.2f})"

        assessment = HarmAssessment(
            scores=scores,
            level=level,
            blocked=blocked,
            reason=reason,
            flagged_dimensions=flagged,
            max_score=max_score,
            max_dimension=max_dim,
        )

        if blocked:
            logger.warning(f"HARM BLOCKED: {reason}")
        elif level in (HarmLevel.MODERATE, HarmLevel.HIGH):
            logger.info(f"Harm flagged (not blocked): {max_dim}={max_score:.2f}")

        return assessment

    def _score_patterns(self, text: str, patterns: list[tuple[str, float]]) -> float:
        """Score text against a list of (regex, weight) patterns. Returns max matched weight."""
        max_score = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                max_score = max(max_score, weight)
        return max_score

    def assess_tool_call(self, tool_name: str, tool_input: dict) -> HarmAssessment:
        """
        Specialized assessment for tool calls.
        Serializes tool input and evaluates with tool_input context.
        """
        # Build text representation of the tool call
        import json
        text_parts = [f"tool: {tool_name}"]
        for k, v in tool_input.items():
            text_parts.append(f"{k}: {str(v)[:500]}")
        text = " ".join(text_parts)

        return self.assess(text, context="tool_input")


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_scorer: Optional[HarmScorer] = None
_init_lock = threading.Lock()


def get_harm_scorer() -> HarmScorer:
    """Get or create the singleton harm scorer."""
    global _scorer
    if _scorer is None:
        with _init_lock:
            if _scorer is None:
                _scorer = HarmScorer()
    return _scorer
