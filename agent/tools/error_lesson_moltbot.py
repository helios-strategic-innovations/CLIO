"""
ANVIL Moltbot Error Lesson Adapter
====================================
Simplified error-to-lesson pipeline for Moltbot's knowledge index.
Stores lessons as knowledge entries with category 'system_lessons'.

No CLAUDE.md promotion queue (Moltbot's CLAUDE.md is operator-maintained).
No runtime scanner (Moltbot's log format differs — future enhancement).

Feature flags:
  MOLTBOT_ERROR_PIPELINE          — Master gate (default "1")
  MOLTBOT_ERROR_PIPELINE_LOG_ONLY — Log-only mode (default "1")

Author: Atlas Strategies — Aidan Tan (Momentum/Catalysts)
"""

import fcntl
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("openclaw.anvil")

_NY = ZoneInfo("America/New_York")

# ============================================================
# Paths
# ============================================================

_DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / "data"
_LESSONS_DIR = _DATA_DIR / "knowledge" / "system_lessons"
_HISTORY_FILE = _DATA_DIR / "knowledge" / "error_history.json"


class MoltbotLessonAdapter:
    """Adapts ANVIL lessons for Moltbot's knowledge index format."""

    def __init__(self):
        _LESSONS_DIR.mkdir(parents=True, exist_ok=True)

    async def record_error(self, error_class: str, description: str,
                           affected_files: list = None, context: dict = None) -> Optional[Dict]:
        """Record an error and generate a knowledge entry via handle_save_knowledge().

        Args:
            error_class: From error_taxonomy (e.g., 'API_FAILURE').
            description: Human-readable description.
            affected_files: List of files involved.
            context: Additional context dict.

        Returns:
            The knowledge entry dict, or None if disabled.
        """
        from agent.tools.error_taxonomy import ANVIL_ENABLED, ANVIL_LOG_ONLY, ERROR_CLASSES

        if not ANVIL_ENABLED():
            return None

        cls_info = ERROR_CLASSES.get(error_class, {})
        severity = cls_info.get("severity_default", "MEDIUM")
        now = datetime.now(_NY)
        entry_id = f"anvil_{error_class.lower()}_{now.strftime('%Y%m%d_%H%M%S')}"

        content = self._format_content(error_class, description, severity, affected_files, context)
        title = f"ANVIL: {error_class} — {description[:80]}"
        tags = ["anvil", error_class.lower(), severity.lower()]

        if ANVIL_LOG_ONLY():
            logger.info(f"ANVIL [LOG_ONLY]: Would store lesson {entry_id} ({error_class}, {severity})")
            return {"id": entry_id, "title": title, "category": "system_lessons",
                    "content": content, "tags": tags}

        # Use canonical save path — gets indexed, deduped, and bridge-mirrored
        try:
            from agent.tools.knowledge_tools import handle_save_knowledge
            result = await handle_save_knowledge({
                "title": title,
                "content": content,
                "category": "system_lessons",
                "tags": tags,
                "quality_score": 8 if severity in ("CRITICAL", "HIGH") else 6,
                "source": "error_pipeline",
            })
            logger.info(f"ANVIL: Lesson indexed: {entry_id} — {result[:80] if isinstance(result, str) else 'ok'}")
        except Exception as e:
            logger.debug(f"ANVIL: handle_save_knowledge failed, falling back to direct write: {e}")
            # Fallback: direct write (still creates the file even if indexing fails)
            entry_file = _LESSONS_DIR / f"{entry_id}.json"
            tmp_file = entry_file.with_suffix(".json.tmp")
            with open(tmp_file, "w") as f:
                json.dump({"id": entry_id, "title": title, "category": "system_lessons",
                           "content": content, "tags": tags, "created": now.isoformat(),
                           "source": "error_pipeline"}, f, indent=2)
            os.replace(tmp_file, entry_file)

        # Update history
        self._update_history(error_class, entry_id)

        return {"id": entry_id, "title": title, "category": "system_lessons",
                "content": content, "tags": tags}

    def _format_content(self, error_class: str, description: str,
                        severity: str, affected_files: list, context: dict) -> str:
        """Format lesson content for Moltbot's knowledge index."""
        lines = [
            f"# Error Lesson: {error_class}",
            f"**Severity:** {severity}",
            f"**Description:** {description}",
        ]
        if affected_files:
            lines.append(f"**Files:** {', '.join(affected_files)}")
        if context:
            ticker = context.get("ticker", "")
            if ticker:
                lines.append(f"**Ticker:** {ticker}")
        lines.append("")
        lines.append("## What Happened")
        lines.append(description)
        lines.append("")
        lines.append("## Prevention")

        # Prevention rules inline (avoid cross-repo import from error_lesson_generator)
        _prevention_rules = {
            "API_FAILURE": "Add retry with exponential backoff. Check rate limits.",
            "DATA_STALE": "Investigate why the data source stopped updating.",
            "BRIDGE_CORRUPT": "Verify atomic write pattern is used.",
            "SERVICE_CRASH": "Check OOM, unhandled exception, or resource exhaustion.",
            "FEATURE_FLAG_LEAK": "Audit all code paths gated by this flag.",
            "RESOURCE_EXHAUST": "Identify the resource consumer and add monitoring.",
            "IMPORT_FAIL": "Verify module exists before importing.",
        }
        rule = _prevention_rules.get(error_class, "Review and add specific prevention measures.")
        lines.append(rule)

        return "\n".join(lines)

    def _update_history(self, error_class: str, entry_id: str) -> None:
        """Update error history tracking (flock + atomic write)."""
        lock_fd = None
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            lock_path = str(_HISTORY_FILE) + ".lock"
            lock_fd = open(lock_path, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            history = {}
            if _HISTORY_FILE.exists():
                with open(_HISTORY_FILE) as f:
                    history = json.load(f)

            if "errors" not in history:
                history["errors"] = {}

            if error_class not in history["errors"]:
                history["errors"][error_class] = {"count": 0, "entries": []}

            history["errors"][error_class]["count"] += 1
            history["errors"][error_class]["entries"].append(entry_id)
            # Keep last 50 entries per class
            history["errors"][error_class]["entries"] = history["errors"][error_class]["entries"][-50:]

            _tmp = _HISTORY_FILE.with_suffix(".json.tmp")
            with open(_tmp, "w") as f:
                json.dump(history, f, indent=2)
            os.replace(_tmp, _HISTORY_FILE)

        except Exception as e:
            logger.warning(f"ANVIL: Failed to update history: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass


# Module-level singleton (thread-safe)
_adapter = None
_adapter_lock = threading.Lock()


def get_moltbot_lesson_adapter() -> MoltbotLessonAdapter:
    global _adapter
    if _adapter is None:
        with _adapter_lock:
            if _adapter is None:
                _adapter = MoltbotLessonAdapter()
    return _adapter
