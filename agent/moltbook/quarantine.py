"""
Moltbook Quarantine Manager
============================
Append-only JSONL store for insights from Moltbook that may be worth
integrating into Moltbot's primary knowledge base.

ISOLATION GUARANTEE:
  Moltbook NEVER writes directly to the primary knowledge base.
  All insights are quarantined here. Dominic must manually approve
  each insight via /moltbook_approve <id> before any integration.
  Integration itself requires Dominic to invoke save_knowledge explicitly.

Author: Moltbot (Session 68, Feb 27 2026)
"""

import fcntl
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields as dc_fields
from datetime import datetime, timezone
from typing import List, Optional

from enum import Enum

logger = logging.getLogger("moltbot.moltbook.quarantine")

QUARANTINE_DIR = "/home/agent/openclaw/data/moltbook/quarantine"
QUARANTINE_FILE = os.path.join(QUARANTINE_DIR, "insights.jsonl")


class QuarantineReason(Enum):
    INJECTION_PATTERN = "injection_pattern"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    CONTENT_FILTER = "content_filter"
    MANUAL = "manual"


@dataclass
class QuarantinedInsight:
    """An insight from Moltbook awaiting Dominic's approval."""
    id: str
    timestamp: str
    source_post_id: str          # ID of the Moltbook post/comment
    source_author: str           # Moltbook username (anonymized if possible)
    content_original: str        # Original post text (truncated to 1000 chars)
    insight_extracted: str       # What Moltbot extracted as potentially valuable
    category: str                # "market", "geopolitical", "strategy", "other"
    relevance_score: float       # 0.0-1.0 — Moltbot's estimated relevance
    status: str                  # "pending" | "approved" | "discarded"
    reviewed_at: str = ""
    reviewed_by: str = ""
    integration_note: str = ""   # Dominic's note if approved


class QuarantineManager:
    """
    Manages the insight quarantine pipeline.
    All writes are append-only to a JSONL file.
    Approved insights are flagged but NOT automatically integrated.
    """

    def __init__(self):
        os.makedirs(QUARANTINE_DIR, exist_ok=True)
        self._cache: dict[str, QuarantinedInsight] = {}
        self._lock = threading.Lock()  # S144 H-8: thread-safe cache access
        self._load_cache()

    def _load_cache(self):
        """Load existing quarantine entries into memory."""
        if not os.path.isfile(QUARANTINE_FILE):
            return
        # S121 W-46: Added fcntl.flock for concurrent read safety
        lock_path = QUARANTINE_FILE + ".lock"
        lock_fd = None
        try:
            lock_fd = open(lock_path, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            with open(QUARANTINE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        known = {f.name for f in dc_fields(QuarantinedInsight)}
                        insight = QuarantinedInsight(**{k: v for k, v in data.items() if k in known})
                        self._cache[insight.id] = insight
        except Exception as e:
            logger.error(f"Failed to load quarantine cache: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass

    def _append(self, insight: QuarantinedInsight):
        """Append insight to the JSONL file. Append-only."""
        # S121 W-46: Added fcntl.flock for concurrent write safety
        lock_path = QUARANTINE_FILE + ".lock"
        lock_fd = None
        try:
            lock_fd = open(lock_path, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            with open(QUARANTINE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(insight)) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to quarantine: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass

    def _rewrite(self):
        """Rewrite the JSONL file with current cache state (for status updates)."""
        # S144: Snapshot under threading lock, write under flock
        with self._lock:
            snapshot = [asdict(i) for i in self._cache.values()]
        # S121 W-46: Added fcntl.flock for concurrent write safety
        lock_path = QUARANTINE_FILE + ".lock"
        lock_fd = None
        try:
            lock_fd = open(lock_path, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            tmp_path = QUARANTINE_FILE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                for entry in snapshot:
                    f.write(json.dumps(entry) + "\n")
            os.replace(tmp_path, QUARANTINE_FILE)
        except Exception as e:
            logger.error(f"Failed to rewrite quarantine: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass

    def quarantine(
        self,
        source_post_id: str,
        source_author: str,
        content_original: str,
        insight_extracted: str,
        category: str = "other",
        relevance_score: float = 0.5,
    ) -> QuarantinedInsight:
        """Add a new insight to quarantine."""
        insight = QuarantinedInsight(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_post_id=source_post_id,
            source_author=source_author[:50],  # Truncate
            content_original=content_original[:1000],
            insight_extracted=insight_extracted[:500],
            category=category,
            relevance_score=relevance_score,
            status="pending",
        )
        with self._lock:
            self._cache[insight.id] = insight
        self._append(insight)
        logger.info(f"Quarantined insight: {insight.id} (category={category}, relevance={relevance_score:.2f})")
        return insight

    def approve(self, insight_id: str, integration_note: str = "") -> Optional[QuarantinedInsight]:
        """
        Dominic approves a quarantined insight.
        Note: Approval does NOT automatically integrate — Dominic must call save_knowledge.
        """
        with self._lock:
            insight = self._cache.get(insight_id)
            if not insight:
                logger.warning(f"Quarantine: insight not found: {insight_id}")
                return None
            insight.status = "approved"
            insight.reviewed_at = datetime.now(timezone.utc).isoformat()
            insight.reviewed_by = "Dominic"
            insight.integration_note = integration_note
        # Write outside lock (flock protects disk I/O)
        self._rewrite()
        logger.info(f"Insight approved: {insight_id}")
        return insight

    def discard(self, insight_id: str) -> bool:
        """Dominic discards a quarantined insight."""
        with self._lock:
            insight = self._cache.get(insight_id)
            if not insight:
                return False
            insight.status = "discarded"
            insight.reviewed_at = datetime.now(timezone.utc).isoformat()
            insight.reviewed_by = "Dominic"
        # Write outside lock (flock protects disk I/O)
        self._rewrite()
        logger.info(f"Insight discarded: {insight_id}")
        return True

    def add(self, content: str, source: str = "", reason=None, metadata: dict = None) -> QuarantinedInsight:
        """Convenience alias for quarantine() — accepts heartbeat-style keyword args."""
        reason_str = reason.value if hasattr(reason, 'value') else str(reason or "manual")
        meta = metadata or {}
        assessment = meta.get("assessment", {})
        return self.quarantine(
            source_post_id=source,
            source_author=meta.get("author", "unknown"),
            content_original=content[:1000],
            # S159 L-12: Handle ContentAssessment dataclass
            insight_extracted=(
                assessment.get("reason", content[:200]) if isinstance(assessment, dict)
                else getattr(assessment, "reason", content[:200]) if hasattr(assessment, "grade")
                else content[:200]
            ),
            category=reason_str,
            relevance_score=0.5,
        )

    def get_pending(self) -> List[QuarantinedInsight]:
        """Return all pending (unreviewed) insights."""
        with self._lock:
            return [i for i in self._cache.values() if i.status == "pending"]

    def get_approved(self) -> List[QuarantinedInsight]:
        """Return approved insights awaiting manual integration."""
        with self._lock:
            return [i for i in self._cache.values() if i.status == "approved"]

    def format_pending_for_telegram(self) -> str:
        """Format pending insights for Telegram display."""
        pending = self.get_pending()
        if not pending:
            return "✅ No pending insights in quarantine."

        lines = [f"Quarantine — {len(pending)} pending insight(s):\n"]
        for i in pending[:5]:  # Max 5 in Telegram
            lines.append(
                f"ID: {i.id} | Cat: {i.category} | Score: {i.relevance_score:.1f}\n"
                f"Insight: {i.insight_extracted[:150]}\n"
                f"/moltbook_approve {i.id} | /moltbook_discard {i.id}\n"
            )

        if len(pending) > 5:
            lines.append(f"...and {len(pending) - 5} more.")

        return "\n".join(lines)


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_manager: Optional[QuarantineManager] = None
_init_lock = threading.Lock()


def get_quarantine_manager() -> QuarantineManager:
    global _manager
    if _manager is None:
        with _init_lock:
            if _manager is None:
                _manager = QuarantineManager()
    return _manager
