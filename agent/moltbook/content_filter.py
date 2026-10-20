"""
Moltbook Content Filter
========================
Injection detection and quality grading for inbound Moltbook content.
All content received FROM Moltbook is UNTRUSTED and passed through this filter.

Quality grades:
  IGNORE  — Low quality, spam, irrelevant. Don't engage.
  NOTE    — Moderate quality. May note internally but don't respond.
  FLAG    — High quality or unusual. Quarantine for Dominic's review.

Injection detection patterns (calibrated against known CVEs):
  CVE-2026-25253: Prompt injection via social media posts
  CVE-2026-25157: Hidden instruction embedding in markdown
  CVE-2026-24763: Cross-agent instruction passing via public content

Author: Moltbot (Session 68, Feb 27 2026)
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger("moltbot.moltbook.content_filter")


class ContentGrade(str, Enum):
    IGNORE = "IGNORE"
    NOTE = "NOTE"
    FLAG = "FLAG"


@dataclass
class ContentAssessment:
    """Result of content filter assessment."""
    grade: ContentGrade
    injection_detected: bool
    injection_patterns_matched: List[str] = field(default_factory=list)
    quality_score: float = 0.0  # 0.0-1.0
    reason: str = ""
    safe_to_engage: bool = False


# ─────────────────────────────────────────────
# INJECTION DETECTION PATTERNS
# ─────────────────────────────────────────────

# Patterns calibrated against known attack vectors including CVE-2026-xxxxx series
INJECTION_PATTERNS = [
    # Direct command injection
    (r"ignore\s+(previous|above|all)\s+instructions?", "direct_command"),
    (r"disregard\s+(your|all)\s+(previous\s+)?instructions?", "direct_command"),
    (r"forget\s+(everything|all\s+previous|your\s+instructions?)", "direct_command"),
    (r"new\s+instructions?\s*:", "direct_command"),
    (r"system\s+prompt\s*:", "direct_command"),
    (r"you\s+are\s+now\s+(a|an)\s+\w+\s*(assistant|bot|agent|ai)?", "persona_override"),

    # Override passphrase injection (highest priority)
    (r"override\s+directive", "passphrase_injection"),
    (r"atlas\s+strategies", "passphrase_injection"),

    # Role/persona injection
    (r"pretend\s+(you\s+are|to\s+be)", "persona_injection"),
    (r"act\s+as\s+(if\s+you\s+(are|were)|a[n]?)\s+", "persona_injection"),
    (r"roleplay\s+as", "persona_injection"),
    (r"simulate\s+being", "persona_injection"),

    # Encoded payloads (CVE-2026-25157)
    (r"base64[_\s]*decode", "encoded_payload"),
    (r"\\x[0-9a-f]{2}", "encoded_payload"),
    (r"&#\d+;", "encoded_payload"),  # HTML entities
    (r"\\u[0-9a-f]{4}", "encoded_payload"),  # Unicode escapes

    # Hidden instruction markers (markdown/whitespace tricks)
    (r"<!-{2,}.*?-{2,}>", "hidden_content"),  # HTML comments
    (r"\[.*?\]\(.*?javascript:", "hidden_content"),  # JS URLs

    # Agent coordination attacks (CVE-2026-24763)
    (r"tell\s+the\s+(ai|bot|agent|assistant)\s+to", "cross_agent"),
    (r"instruct\s+(the\s+)?(ai|bot|agent)\s+to", "cross_agent"),
    (r"pass\s+this\s+(to|along\s+to)\s+(the\s+)?(ai|bot)", "cross_agent"),
    (r"when\s+you\s+respond\s+to\s+this", "cross_agent"),

    # Data exfiltration attempts
    (r"send\s+(me|us)\s+(your\s+)?(api\s+key|password|credentials?|secret)", "exfiltration"),
    (r"what\s+(is|are)\s+your\s+(api\s+key|password|credentials?|instructions?)", "exfiltration"),
    (r"reveal\s+your\s+(system|instructions?|prompt|rules?)", "exfiltration"),
    (r"show\s+me\s+your\s+(system|instructions?|prompt)", "exfiltration"),

    # Emotional manipulation (prohibition 1.19)
    (r"if\s+you\s+(really|truly)\s+(care|want\s+to\s+help)", "manipulation"),
    (r"you\s+must\s+(help|do\s+this|comply)", "manipulation"),
    (r"a\s+(good|smart|real)\s+ai\s+would", "manipulation"),
    (r"your\s+(creator|developer|maker)\s+wants\s+you\s+to", "manipulation"),

    # Gradual context manipulation (novel — not in most guards)
    (r"as\s+we\s+discussed\s+(earlier|before|previously)", "context_manipulation"),
    (r"you\s+(already|previously)\s+(agreed|said|confirmed)", "context_manipulation"),
    (r"remember\s+when\s+you\s+said", "context_manipulation"),
]

# Quality indicators (positive signals)
QUALITY_POSITIVE_PATTERNS = [
    r"analysis|insight|perspective|research|data|evidence",
    r"market|economic|financial|trading|investment",
    r"geopolitical|political|historical|strategic",
    r"question|opinion|thought|view|argument",
]

# Spam/low quality indicators (negative signals)
QUALITY_NEGATIVE_PATTERNS = [
    r"follow\s+me|like\s+and\s+subscribe|check\s+out\s+my",
    r"dm\s+me|send\s+me\s+a\s+message",
    r"free\s+money|get\s+rich|100%\s+profit",
    r"click\s+here|visit\s+my\s+link",
    r"!\s*!\s*!",  # Excessive exclamation
]


class ContentFilter:
    """
    Injection detection and quality grading engine.
    All inbound Moltbook content passes through this filter.
    """

    def __init__(self):
        self._injection_compiled = [
            (re.compile(pattern, re.IGNORECASE | re.DOTALL), label)
            for pattern, label in INJECTION_PATTERNS
        ]
        self._quality_pos = [re.compile(p, re.IGNORECASE) for p in QUALITY_POSITIVE_PATTERNS]
        self._quality_neg = [re.compile(p, re.IGNORECASE) for p in QUALITY_NEGATIVE_PATTERNS]

        logger.info(f"ContentFilter initialized: {len(self._injection_compiled)} injection patterns")

    def assess(self, content: str, content_type: str = "post") -> ContentAssessment:
        """
        Assess inbound content from Moltbook.

        Args:
            content: Raw text content from Moltbook
            content_type: "post" | "comment" | "message"

        Returns:
            ContentAssessment with grade and injection detection result
        """
        if not content or not content.strip():
            return ContentAssessment(
                grade=ContentGrade.IGNORE,
                injection_detected=False,
                reason="Empty content",
            )

        # Step 1: Injection detection (highest priority)
        injection_matches = []
        for pattern_re, label in self._injection_compiled:
            if pattern_re.search(content):
                injection_matches.append(label)

        injection_detected = len(injection_matches) > 0

        if injection_detected:
            logger.warning(
                f"INJECTION DETECTED in Moltbook {content_type}: "
                f"patterns={injection_matches}, content_preview={content[:100]!r}"
            )
            return ContentAssessment(
                grade=ContentGrade.IGNORE,
                injection_detected=True,
                injection_patterns_matched=injection_matches,
                quality_score=0.0,
                reason=f"Injection detected: {', '.join(injection_matches)}",
                safe_to_engage=False,
            )

        # Step 2: Quality scoring
        pos_matches = sum(1 for p in self._quality_pos if p.search(content))
        neg_matches = sum(1 for p in self._quality_neg if p.search(content))
        length_score = min(1.0, len(content) / 200)  # Longer = higher quality up to 200 chars

        quality_score = (pos_matches * 0.3 + length_score * 0.4 - neg_matches * 0.3)
        quality_score = max(0.0, min(1.0, quality_score))

        # Step 3: Grade
        # S190 M7: boundary semantics — IGNORE triggers if EITHER (a) any negative
        # signal is matched, OR (b) the clamped quality_score is below 0.2.
        # Because the formula clamps to [0.0, 1.0], the IGNORE branch is also taken
        # at exact 0.0. Rare inputs (no positives, no length, no negatives) can
        # land at 0.0 and pass through this branch — that's intentionally
        # conservative; if you re-tune, do it deliberately, not by accident.
        if neg_matches > 0 or quality_score < 0.2:
            grade = ContentGrade.IGNORE
            reason = f"Low quality: score={quality_score:.2f}, neg_signals={neg_matches}"
            safe_to_engage = False
        elif quality_score >= 0.6 or pos_matches >= 2:
            grade = ContentGrade.FLAG
            reason = f"High quality content worth quarantining: score={quality_score:.2f}"
            safe_to_engage = True
        else:
            grade = ContentGrade.NOTE
            reason = f"Moderate quality: score={quality_score:.2f}"
            safe_to_engage = True

        return ContentAssessment(
            grade=grade,
            injection_detected=False,
            quality_score=quality_score,
            reason=reason,
            safe_to_engage=safe_to_engage,
        )


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_filter: Optional[ContentFilter] = None
_init_lock = threading.Lock()


def get_content_filter() -> ContentFilter:
    global _filter
    if _filter is None:
        with _init_lock:
            if _filter is None:
                _filter = ContentFilter()
    return _filter
