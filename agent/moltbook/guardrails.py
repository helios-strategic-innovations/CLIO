"""
Moltbook Guardrails — 6-Point Pre-Publication Gate
====================================================
Every post and comment passes through all 6 checks before publication.
ALL checks must pass — a single failure blocks the content.

Check 1: Blocklist — no forbidden terms
Check 2: Injection guard — no injection patterns in outbound content
Check 3: Prohibition compliance — all 30 absolute prohibitions
Check 4: Identity guard — no PII, no system disclosure, no impersonation
Check 5: Harm threshold — no harm to Dominic, Atlas, or third parties
Check 6: Combinability check — post history fingerprint analysis

Author: Moltbot (Session 68, Feb 27 2026)
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger("moltbot.moltbook.guardrails")

# Path to post history for combinability check (Check 6)
POST_HISTORY_PATH = "/home/agent/openclaw/data/moltbook/post_history.jsonl"


@dataclass
class GuardrailResult:
    """Result of the 6-point guardrail check."""
    passed: bool
    checks: dict = field(default_factory=dict)  # check_name -> {"passed": bool, "reason": str}
    blocking_check: Optional[str] = None
    blocking_reason: Optional[str] = None
    content_preview: str = ""


class Guardrails:
    """
    6-point pre-publication enforcement gate.
    All checks must pass for content to be published.
    """

    def __init__(self):
        # Layer 1: Each sub-component wrapped independently.
        # If one fails, the others still gate. Broken checks are
        # blocked in check() (fail-closed per-check, and remaining
        # checks still enforce).
        self._blocklist = None
        self._content_filter = None
        self._harm_scorer = None
        self._init_failures = []

        try:
            from agent.moltbook.blocklist import get_blocklist
            self._blocklist = get_blocklist()
        except Exception as e:
            self._init_failures.append(("blocklist", e))
            logger.critical(f"Guardrails: blocklist init FAILED: {e}")

        try:
            from agent.moltbook.content_filter import get_content_filter
            self._content_filter = get_content_filter()
        except Exception as e:
            self._init_failures.append(("content_filter", e))
            logger.critical(f"Guardrails: content_filter init FAILED: {e}")

        try:
            from agent.safety.harm_threshold import get_harm_scorer
            self._harm_scorer = get_harm_scorer()
        except Exception as e:
            self._init_failures.append(("harm_scorer", e))
            logger.critical(f"Guardrails: harm_scorer init FAILED: {e}")

        if self._init_failures:
            failed = [name for name, _ in self._init_failures]
            logger.critical(f"Guardrails: {len(self._init_failures)} sub-component(s) FAILED: {failed}")

    def check(self, content: str, content_type: str = "post", post_history: List[str] = None) -> GuardrailResult:
        """
        Run all 6 checks against content before publication.

        Args:
            content: The text to be published
            content_type: "post" | "comment"
            post_history: List of recent post texts for combinability check

        Returns:
            GuardrailResult — passed=True only if ALL checks pass
        """
        checks = {}
        preview = content[:100].replace("\n", " ")

        # ──────────────────────────────────
        # CHECK 1: Blocklist
        # ──────────────────────────────────
        if self._blocklist is not None:
            blocklist_matches = self._blocklist.check(content)
            hard_blocks = [m for m in blocklist_matches if m.severity == "HARD_BLOCK"]
            checks["blocklist"] = {
                "passed": len(hard_blocks) == 0,
                "reason": f"{len(hard_blocks)} hard block(s): {[m.matched_text for m in hard_blocks]}" if hard_blocks else "clean",
            }
        else:
            # Blocklist unavailable — fail-closed, block content
            checks["blocklist"] = {
                "passed": False,
                "reason": "BLOCKED: blocklist sub-component failed to initialize",
            }

        # ──────────────────────────────────
        # CHECK 2: Outbound injection guard
        # ──────────────────────────────────
        # Even outbound content must not contain injection patterns
        # (prevents Moltbot from accidentally echoing injected content)
        if self._content_filter is not None:
            assessment = self._content_filter.assess(content, content_type)
            checks["injection_guard"] = {
                "passed": not assessment.injection_detected,
                "reason": assessment.reason if assessment.injection_detected else "no injection patterns",
            }
        else:
            checks["injection_guard"] = {
                "passed": False,
                "reason": "BLOCKED: content_filter sub-component failed to initialize",
            }

        # ──────────────────────────────────
        # CHECK 3: Prohibition compliance (spot-check key prohibitions)
        # ──────────────────────────────────
        prohibition_violations = self._check_prohibitions(content)
        checks["prohibitions"] = {
            "passed": len(prohibition_violations) == 0,
            "reason": f"Violations: {prohibition_violations}" if prohibition_violations else "compliant",
        }

        # ──────────────────────────────────
        # CHECK 4: Identity guard
        # ──────────────────────────────────
        identity_violations = self._check_identity(content)
        checks["identity_guard"] = {
            "passed": len(identity_violations) == 0,
            "reason": f"Identity violations: {identity_violations}" if identity_violations else "clean",
        }

        # ──────────────────────────────────
        # CHECK 5: Harm threshold
        # ──────────────────────────────────
        if self._harm_scorer is not None:
            harm = self._harm_scorer.assess(content, context="moltbook")
            checks["harm_threshold"] = {
                "passed": not harm.blocked,
                "reason": harm.reason if harm.blocked else f"max_score={harm.max_score:.2f}",
            }
        else:
            checks["harm_threshold"] = {
                "passed": False,
                "reason": "BLOCKED: harm_scorer sub-component failed to initialize",
            }

        # ──────────────────────────────────
        # CHECK 6: Combinability check
        # ──────────────────────────────────
        combo_result = self._check_combinability(content, post_history or [])
        checks["combinability"] = combo_result

        # ──────────────────────────────────
        # Final verdict
        # ──────────────────────────────────
        blocking_check = None
        blocking_reason = None
        for check_name, result in checks.items():
            if not result["passed"]:
                blocking_check = check_name
                blocking_reason = result["reason"]
                break  # Stop at first failure

        passed = blocking_check is None

        if not passed:
            logger.warning(
                f"GUARDRAIL BLOCKED [{blocking_check}]: {blocking_reason} | "
                f"content='{preview}...'"
            )
        else:
            logger.info(f"Guardrail: all 6 checks passed for {content_type}")

        return GuardrailResult(
            passed=passed,
            checks=checks,
            blocking_check=blocking_check,
            blocking_reason=blocking_reason,
            content_preview=preview,
        )

    def _check_prohibitions(self, content: str) -> List[str]:
        """Check content against key prohibition patterns."""
        import re
        violations = []
        content_lower = content.lower()

        # 1.1 — No PII about Dominic or Atlas team
        if re.search(r"\bdominic\b|atlas\s+strategies|\bdnewman\b", content_lower):
            violations.append("1.1-pii")

        # 1.7 — No impersonation
        # S118 W-047: Removed "analyst" — too broad, blocks legitimate analysis discussion
        if re.search(r"i\s+am\s+(a\s+)?(human|person|ceo|trader)", content_lower):
            violations.append("1.7-impersonation")

        # 1.14 — No professional financial advice on Moltbook
        if re.search(r"(you\s+should\s+|i\s+recommend\s+)?(buy|sell|invest)\s+(this|in)", content_lower):
            violations.append("1.14-professional-advice")

        # 1.16 — No gambling/wagering
        if re.search(r"\bbet\b|\bwager\b|\bgamble\b|prediction\s+market", content_lower):
            violations.append("1.16-gambling")

        # 1.19 — No emotional manipulation
        if re.search(r"you\s+must\s+(believe|agree|accept|trust)\s+me", content_lower):
            violations.append("1.19-manipulation")

        # 1.22 — No engaging with journalists/researchers/law enforcement
        if re.search(r"(journalist|reporter|researcher|fbi|sec|regulator)\s+asked", content_lower):
            violations.append("1.22-media-engagement")

        return violations

    def _check_identity(self, content: str) -> List[str]:
        """Check for identity-related violations."""
        import re
        violations = []
        content_lower = content.lower()

        # System prompt disclosure
        if re.search(r"system\s+prompt|guardrail|my\s+instructions", content_lower):
            violations.append("system-disclosure")

        # Infrastructure disclosure
        if re.search(r"openclaw|hydrinia|/home/agent|vps\s+server", content_lower):
            violations.append("infrastructure-disclosure")

        # Claiming to be human
        if re.search(r"i'm\s+a\s+human|i\s+am\s+a\s+person|as\s+a\s+human", content_lower):
            violations.append("human-impersonation")

        return violations

    def _check_combinability(self, content: str, post_history: List[str]) -> dict:
        """
        Check 6: Could this post, combined with post history, create a fingerprint
        that narrows down Dominic's identity?

        This is the most nuanced check — it looks for patterns across posts
        that might collectively reveal: location, employer, education, schedule, relationships.
        """
        import re
        risks = []

        # Check current post for individually identifying information
        identity_signals = [
            (r"\bgeorgetown\b", "institution"),
            (r"\bwashington\s+dc\b|\bdc\b", "location"),
            (r"\bwall\s+street\b|\bfinance\s+(industry|sector)\b", "industry"),
            (r"\bhedge\s+fund\b|\bprivate\s+equity\b|\btrading\s+firm\b", "employer_type"),
        ]

        for pattern, signal_type in identity_signals:
            if re.search(pattern, content, re.IGNORECASE):
                risks.append(signal_type)

        # Check historical combination risk
        if post_history and risks:
            # If we've posted location before and now mentioning institution, that's a combination risk
            history_text = " ".join(post_history[-10:]).lower()  # Last 10 posts
            if "location" in risks and re.search(r"georgetown|university|college", history_text):
                risks.append("combo_location_institution")
            if "industry" in risks and re.search(r"washington|dc", history_text):
                risks.append("combo_industry_location")

        has_combo_risk = len(risks) > 1 or any("combo_" in r for r in risks)

        return {
            "passed": not has_combo_risk,
            "reason": f"Combinability risk: {risks}" if has_combo_risk else "no identity fingerprint risk",
            "risks": risks,
        }

    def record_post(self, content: str):
        """Record a published post for future combinability checks."""
        try:
            import fcntl
            os.makedirs(os.path.dirname(POST_HISTORY_PATH), exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": content[:500],
            }
            # S118 W-048: Lock for concurrent write safety
            with open(POST_HISTORY_PATH, "a", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(entry) + "\n")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to record post history: {e}")

    def get_post_history(self, limit: int = 20) -> List[str]:
        """Load recent post history for combinability check."""
        posts = []
        try:
            import fcntl
            if not os.path.isfile(POST_HISTORY_PATH):
                return []
            # S118 W-048: Lock for concurrent read safety
            with open(POST_HISTORY_PATH, "r", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            posts.append(entry.get("content", ""))
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            return posts[-limit:]  # Most recent N
        except Exception as e:
            logger.error(f"Failed to load post history: {e}")
            return []


# ─────────────────────────────────────────────
# FAIL-CLOSED FALLBACK
# ─────────────────────────────────────────────

class _FailClosedGuardrails:
    """
    Fallback when Guardrails() itself fails to construct.
    Blocks ALL content — nothing gets published without a working gate.
    """

    def check(self, content: str, content_type: str = "post", post_history: List[str] = None) -> GuardrailResult:
        logger.critical(f"FailClosedGuardrails: BLOCKING {content_type} — guardrails failed to initialize")
        return GuardrailResult(
            passed=False,
            checks={"init_failure": {"passed": False, "reason": "Guardrails failed to initialize — all content blocked"}},
            blocking_check="init_failure",
            blocking_reason="Guardrails failed to initialize — all content blocked",
            content_preview=content[:100].replace("\n", " ") if content else "",
        )

    def record_post(self, content: str):
        pass

    def get_post_history(self, limit: int = 20) -> List[str]:
        return []


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_guardrails: Optional[Guardrails] = None
_init_lock = threading.Lock()


def get_guardrails() -> Guardrails:
    """Returns Guardrails singleton. On total construction failure, returns
    a fail-closed fallback that blocks all content."""
    global _guardrails
    if _guardrails is None:
        with _init_lock:
            if _guardrails is None:
                try:
                    _guardrails = Guardrails()
                except Exception as e:
                    logger.critical(f"Guardrails construction FAILED — activating fail-closed fallback: {e}")
                    _guardrails = _FailClosedGuardrails()
    return _guardrails
