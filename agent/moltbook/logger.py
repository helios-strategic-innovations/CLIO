"""
Moltbook Interaction Logger
============================
Append-only JSONL interaction log for all Moltbook activity.
Separate from the immutable audit log — this is operational logging.

Logs every cycle: posts browsed, assessed, quarantined, published, errors.

Author: Moltbot (Session 68, Feb 27 2026)
"""

import fcntl
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from zoneinfo import ZoneInfo

_NY = ZoneInfo("America/New_York")

from enum import Enum

logger = logging.getLogger("moltbot.moltbook.logger")

LOGS_DIR = "/home/agent/openclaw/data/moltbook/logs"


class InteractionType(Enum):
    BROWSE = "browse"
    ASSESS = "assess"
    QUARANTINE = "quarantine"
    POST = "post"
    COMMENT = "comment"
    OBSERVE = "observe"
    SKIP = "skip"
    ERROR = "error"
    KILL = "kill"


@dataclass
class MoltbookLogEntry:
    """A single Moltbook interaction log entry."""
    timestamp: str
    event: str          # "browse", "assess", "quarantine", "post", "comment", "skip", "error", "kill"
    details: dict = field(default_factory=dict)
    cycle_id: str = ""  # Groups events within one heartbeat cycle


class MoltbookLogger:
    """
    Append-only JSONL logger for Moltbook interactions.
    One file per day: moltbook_YYYY-MM-DD.jsonl
    """

    def __init__(self):
        os.makedirs(LOGS_DIR, exist_ok=True)

    def _log_path(self) -> str:
        date_str = datetime.now(_NY).strftime("%Y-%m-%d")
        return os.path.join(LOGS_DIR, f"moltbook_{date_str}.jsonl")

    def log(self, event: str, details: dict = None, cycle_id: str = ""):
        """Append a log entry."""
        try:
            entry = MoltbookLogEntry(
                timestamp=datetime.now(_NY).isoformat(),
                event=event,
                details=details or {},
                cycle_id=cycle_id,
            )
            log_path = self._log_path()
            lock_fd = open(log_path + ".lock", "a")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(entry)) + "\n")
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to write Moltbook log: {e}")

    def log_browse(self, post_count: int, cycle_id: str = ""):
        self.log("browse", {"post_count": post_count}, cycle_id)

    def log_assess(self, post_id: str, grade: str, reason: str, cycle_id: str = ""):
        self.log("assess", {"post_id": post_id, "grade": grade, "reason": reason}, cycle_id)

    def log_quarantine(self, insight_id: str, category: str, score: float, cycle_id: str = ""):
        self.log("quarantine", {"insight_id": insight_id, "category": category, "score": score}, cycle_id)

    def log_post(self, post_id: str, content_preview: str, dry_run: bool, cycle_id: str = ""):
        self.log("post", {"post_id": post_id, "preview": content_preview[:100], "dry_run": dry_run}, cycle_id)

    def log_comment(self, comment_id: str, target_post_id: str, dry_run: bool, cycle_id: str = ""):
        self.log("comment", {"comment_id": comment_id, "target_post_id": target_post_id, "dry_run": dry_run}, cycle_id)

    def log_skip(self, post_id: str, reason: str, cycle_id: str = ""):
        self.log("skip", {"post_id": post_id, "reason": reason}, cycle_id)

    def log_error(self, error: str, context: str = "", cycle_id: str = ""):
        self.log("error", {"error": error, "context": context}, cycle_id)

    def log_kill(self, reason: str):
        self.log("kill", {"reason": reason})

    def get_today_summary(self) -> dict:
        """Return summary statistics for today's log."""
        stats = {
            "browse": 0, "assess": 0, "quarantine": 0,
            "post": 0, "comment": 0, "skip": 0, "error": 0,
        }
        try:
            path = self._log_path()
            if not os.path.isfile(path):
                return stats
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        event = entry.get("event", "")
                        if event in stats:
                            stats[event] += 1
        except Exception as e:
            logger.error(f"Failed to get today summary: {e}")
        return stats

    def get_today_stats(self) -> dict:
        """Return today's stats with heartbeat-expected keys."""
        raw = self.get_today_summary()
        return {
            "posts_today": raw.get("post", 0),
            "comments_today": raw.get("comment", 0),
            "browse_events": raw.get("browse", 0),
            "errors": raw.get("error", 0),
        }

    def get_recent_posts(self, limit: int = 10) -> list[dict]:
        """Return recent post log entries from today's log."""
        entries = []
        try:
            path = self._log_path()
            if not os.path.isfile(path):
                return entries
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if entry.get("event") == "post":
                            entries.append(entry)
        except Exception as e:
            logger.error(f"Failed to get recent posts: {e}")
        return entries[-limit:]

    def get_logs_for_period(self, period_start: str, period_end: str) -> list[dict]:
        """Return all log entries within a date range (YYYY-MM-DD strings)."""
        from datetime import timedelta
        entries = []
        try:
            start = datetime.strptime(period_start, "%Y-%m-%d")
            end = datetime.strptime(period_end, "%Y-%m-%d")
            current = start
            while current <= end:
                date_str = current.strftime("%Y-%m-%d")
                path = os.path.join(LOGS_DIR, f"moltbook_{date_str}.jsonl")
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                entries.append(json.loads(line))
                current += timedelta(days=1)
        except Exception as e:
            logger.error(f"Failed to get logs for period: {e}")
        return entries


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_logger_instance: Optional[MoltbookLogger] = None
_init_lock = threading.Lock()


def get_moltbook_logger() -> MoltbookLogger:
    global _logger_instance
    if _logger_instance is None:
        with _init_lock:
            if _logger_instance is None:
                _logger_instance = MoltbookLogger()
    return _logger_instance
