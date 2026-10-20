"""
Episodic Memory — Warm Tier (Tier 2)
Append-only JSONL timestamped event log.

Records: conversations, trade outcomes, errors, decisions, discoveries.
Searchable by time range, category, outcome.
Compressed monthly into semantic memory clusters.

Modelled on Hydrinia's pattern_recognition — temporal track record of experiences.
"""

import fcntl
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("moltbot.memory.episodic")

# S233 ME-13: Simple dedup — (content_hash, timestamp) of last N records
_DEDUP_WINDOW_S = 60  # Suppress duplicates within 60 seconds
_recent_hashes: dict = {}  # hash -> timestamp

# S162 L-6: Use shared constant from package
from agent.memory import MEMORY_DATA_DIR as DATA_DIR


class EpisodicEntry:
    """A single episodic memory event."""

    def __init__(self, event_type: str, content: str, category: str = "general",
                 outcome: Optional[str] = None, importance: float = 0.5,
                 metadata: Optional[Dict] = None, domain: str = "unknown"):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.event_type = event_type  # "conversation", "trade", "error", "decision", "discovery"
        self.content = content
        self.category = category
        self.domain = domain  # S151-B H-3: domain classification
        self.outcome = outcome  # "success", "failure", "neutral", "pending"
        self.importance = min(max(importance, 0.0), 1.0)
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "content": self.content,
            "category": self.category,
            "domain": self.domain,
            "outcome": self.outcome,
            "importance": self.importance,
            "metadata": self.metadata,
        }


class EpisodicMemory:
    """
    Append-only episodic memory store.
    One JSONL file per month: episodic_YYYY-MM.jsonl
    """

    def __init__(self):
        self._dir = os.path.join(DATA_DIR, "episodic")
        os.makedirs(self._dir, exist_ok=True)
        self._lock_file = os.path.join(self._dir, "episodic.lock")

    def _current_file(self) -> str:
        """Get current month's file path."""
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        return os.path.join(self._dir, f"episodic_{month}.jsonl")

    def record(self, event_type: str, content: str, category: str = "general",
               outcome: Optional[str] = None, importance: float = 0.5,
               metadata: Optional[Dict] = None, domain: str = "unknown") -> Dict:
        """
        Record an episodic event. Append-only — never deletes, never overwrites.
        Returns the recorded entry (or empty dict if deduped).
        """
        # S233 ME-13: Suppress duplicate records within 60s window
        _dedup_key = hashlib.md5(f"{event_type}:{content[:100]}:{category}:{outcome}".encode()).hexdigest()[:12]
        _now = time.monotonic()
        if _dedup_key in _recent_hashes and (_now - _recent_hashes[_dedup_key]) < _DEDUP_WINDOW_S:
            logger.debug(f"Episodic dedup: suppressing duplicate {event_type}/{category} within {_DEDUP_WINDOW_S}s")
            return {}
        _recent_hashes[_dedup_key] = _now
        # Prune old entries periodically (keep dict small)
        if len(_recent_hashes) > 200:
            _cutoff = _now - _DEDUP_WINDOW_S
            for k in list(_recent_hashes):
                if _recent_hashes[k] < _cutoff:
                    del _recent_hashes[k]

        # S159 M-15: Default domain to category, then to "general"
        effective_domain = domain if domain and domain != "unknown" else (category if category and category != "unknown" else "general")
        entry = EpisodicEntry(
            event_type=event_type,
            content=content,
            category=category,
            outcome=outcome,
            importance=importance,
            metadata=metadata,
            domain=effective_domain,
        )
        data = entry.to_dict()

        lock_fd = None
        try:
            lock_fd = open(self._lock_file, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            with open(self._current_file(), "a") as f:
                f.write(json.dumps(data) + "\n")
            logger.debug(f"Episodic record: {event_type}/{category} — {content[:80]}")
        except Exception as e:
            logger.error(f"Failed to write episodic memory: {e}")
        finally:
            if lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()

        return data

    def _update_outcome_in_file(self, fpath: str, event_type: str, keyword: str, outcome: str) -> bool:
        """Scan a single JSONL file backward and update the most recent matching entry.
        Caller MUST hold LOCK_EX on self._lock_file. Returns True if updated."""
        if not os.path.exists(fpath):
            return False

        with open(fpath, "r") as f:
            lines = f.readlines()

        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if (entry.get("event_type") == event_type
                    and (not keyword or keyword.lower() in entry.get("content", "").lower())
                    and entry.get("outcome") in (None, "pending")):
                entry["outcome"] = outcome
                lines[i] = json.dumps(entry) + "\n"

                tmp = fpath + ".tmp"
                try:
                    with open(tmp, "w") as f:
                        f.writelines(lines)
                    os.replace(tmp, fpath)
                finally:
                    # S118 W-054: Clean up stale tmp on write failure
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError as e:
                            logger.debug(f"S162 L-2: Stale tmp cleanup failed: {e}")
                logger.debug(f"Episodic outcome updated: {event_type}/{keyword} -> {outcome}")
                return True

        return False

    def update_outcome(self, event_type: str, keyword: str, outcome: str) -> bool:
        """Update the outcome field of the most recent matching episodic entry.

        Searches backward through the current month's file (and previous month's
        for cross-month boundary entries) for an entry matching event_type and
        keyword (substring in content). Returns True if an entry was updated.
        """
        lock_fd = None
        try:
            lock_fd = open(self._lock_file, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Try current month first
            if self._update_outcome_in_file(self._current_file(), event_type, keyword, outcome):
                return True

            # S118 W-053: Try previous month for cross-month boundary entries
            _now = datetime.now(timezone.utc)
            prev = _now - timedelta(days=32)
            prev_file = os.path.join(self._dir, f"episodic_{prev.strftime('%Y-%m')}.jsonl")
            if self._update_outcome_in_file(prev_file, event_type, keyword, outcome):
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to update episodic outcome: {e}")
            return False
        finally:
            if lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()

    def search(self, category: Optional[str] = None, event_type: Optional[str] = None,
               outcome: Optional[str] = None, keyword: Optional[str] = None,
               min_importance: float = 0.0, limit: int = 20,
               months_back: int = 3) -> List[Dict]:
        """
        Search episodic memory across recent months.
        Returns newest-first.
        """
        results = []
        now = datetime.now(timezone.utc)

        # Collect files for the requested months
        files = []
        for i in range(months_back):
            m = now.month - i
            y = now.year
            while m <= 0:
                m += 12
                y -= 1
            month_dt = datetime(y, m, 1, tzinfo=timezone.utc)
            month_str = month_dt.strftime("%Y-%m")
            fpath = os.path.join(self._dir, f"episodic_{month_str}.jsonl")
            if os.path.exists(fpath):
                files.append(fpath)

        # Also check current month explicitly
        current = self._current_file()
        if current not in files and os.path.exists(current):
            files.insert(0, current)

        lock_fd = None
        try:
            lock_fd = open(self._lock_file, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            for fpath in files:
                try:
                    with open(fpath, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            # Apply filters
                            if category and entry.get("category") != category:
                                continue
                            if event_type and entry.get("event_type") != event_type:
                                continue
                            if outcome and entry.get("outcome") != outcome:
                                continue
                            if entry.get("importance", 0) < min_importance:
                                continue
                            if keyword and keyword.lower() not in entry.get("content", "").lower():
                                continue

                            results.append(entry)
                except Exception as e:
                    logger.error(f"Error reading episodic file {fpath}: {e}")
        finally:
            if lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()

        # Sort by timestamp descending (newest first), take limit
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:limit]

    def get_recent(self, limit: int = 10, months_back: int = 1,
                   event_types: Optional[List[str]] = None,
                   categories: Optional[List[str]] = None) -> List[Dict]:
        """Get N recent episodic entries, optionally filtered by broad type.

        Defaults intentionally preserve the legacy liveness/health behavior:
        get_recent(limit=1) still maps exactly to search(limit=1, months_back=1).
        """
        if not event_types and not categories:
            return self.search(limit=limit, months_back=months_back)

        event_type_set = set(event_types or [])
        category_set = set(categories or [])
        pool_limit = max(limit * 20, 100)
        entries = self.search(limit=pool_limit, months_back=months_back)
        filtered = []
        for entry in entries:
            if event_type_set and entry.get("event_type") not in event_type_set:
                continue
            if category_set and entry.get("category") not in category_set and entry.get("domain") not in category_set:
                continue
            filtered.append(entry)
            if len(filtered) >= limit:
                break
        return filtered

    def _get_month_files(self, months_back: int = 1) -> List[str]:
        """Return file paths for the requested months (newest first)."""
        now = datetime.now(timezone.utc)
        files = []
        for i in range(months_back):
            m = now.month - i
            y = now.year
            while m <= 0:
                m += 12
                y -= 1
            month_str = datetime(y, m, 1, tzinfo=timezone.utc).strftime("%Y-%m")
            fpath = os.path.join(self._dir, f"episodic_{month_str}.jsonl")
            if os.path.exists(fpath):
                files.append(fpath)
        # Also check current month explicitly
        current = self._current_file()
        if current not in files and os.path.exists(current):
            files.insert(0, current)
        return files

    def count(self, months_back: int = 1) -> int:
        """Count total entries in recent months (line-counting, avoids loading all entries)."""
        total = 0
        lock_fd = None
        try:
            lock_fd = open(self._lock_file, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            for path in self._get_month_files(months_back):
                try:
                    with open(path) as f:
                        total += sum(1 for _ in f)
                except Exception as e:
                    logger.error(f"Error counting episodic file {path}: {e}")
        finally:
            if lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
        return total

    def get_outcomes_summary(self, category: str, months_back: int = 3) -> Dict:
        """
        Summarize outcomes for a category — like Hydrinia's win/loss tracking.
        Returns: {success: N, failure: N, neutral: N, pending: N, total: N}
        """
        entries = self.search(category=category, months_back=months_back, limit=5000)
        summary = {"success": 0, "failure": 0, "neutral": 0, "pending": 0, "none": 0, "total": len(entries)}
        for e in entries:
            outcome = e.get("outcome", "none") or "none"
            summary[outcome] = summary.get(outcome, 0) + 1
        return summary

    def stats(self) -> Dict:
        """Return episodic memory statistics."""
        files = [f for f in os.listdir(self._dir) if f.startswith("episodic_") and f.endswith(".jsonl")]
        total_size = sum(os.path.getsize(os.path.join(self._dir, f)) for f in files)
        return {
            "month_files": len(files),
            "total_size_kb": round(total_size / 1024, 1),
            "recent_count": self.count(months_back=1),
        }
