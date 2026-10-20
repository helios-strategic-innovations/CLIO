"""
Working Memory — Hot Tier (Tier 1)
===================================
Short-term, session-scoped memory with a 50-entry sliding window.
Entries auto-expire after TTL (default 72h) unless promoted to episodic.

Modelled after Hydrinia's conviction_history arrays:
  - Recency-weighted retrieval
  - Domain-indexed entries
  - Automatic priority scoring

Author: Moltbot (Session 68, Feb 27 2026)
"""

import fcntl
import json
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field, asdict, fields as dc_fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("moltbot.memory.working")

# S162 L-6: Use shared constant from package
from agent.memory import MEMORY_DATA_DIR
WORKING_MEMORY_PATH = Path(os.path.join(MEMORY_DATA_DIR, "working_memory.json"))
MAX_ENTRIES = int(os.getenv("WORKING_MEMORY_MAX_ENTRIES", "50"))
TTL_HOURS = float(os.getenv("WORKING_MEMORY_TTL_HOURS", "72"))
# S209 WM-deadlock fix: write-behind flusher cadence. The flusher wakes on every
# mutation (event-driven) and also periodically as a safety net for read-triggered
# dirties (get()/access-count) that don't signal it.
_FLUSH_MAX_INTERVAL = float(os.getenv("WORKING_MEMORY_FLUSH_INTERVAL_S", "5.0"))
# Persistence is "broken" for liveness purposes if a dirty state has gone
# unpersisted longer than this (flusher dead / write erroring in a loop).
_PERSIST_STALE_SECONDS = float(os.getenv("WORKING_MEMORY_PERSIST_STALE_S", "60.0"))


@dataclass
class WorkingMemoryEntry:
    """A single hot-memory entry."""
    id: str
    content: str
    domain: str                      # trading, academic, personal, career, system
    priority: float                  # 0.0–1.0 (higher = kept longer)
    created_at: str
    expires_at: str
    access_count: int = 0
    last_accessed: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    promoted: bool = False           # True if promoted to episodic memory

    def is_expired(self) -> bool:
        try:
            exp = datetime.fromisoformat(self.expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) > exp
        except Exception as e:
            logger.debug(f"Entry expiration check failed: {e}")
            return True  # Treat unparseable entries as expired (safe default)

    def access(self):
        """Record an access — updates recency score."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc).isoformat()

    def recency_score(self) -> float:
        """Recency-weighted priority score. Decays over time like Hydrinia conviction."""
        try:
            created = datetime.fromisoformat(self.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
            decay = max(0.0, 1.0 - (age_hours / TTL_HOURS))
            access_boost = min(0.3, self.access_count * 0.05)
            return min(1.0, (self.priority * decay) + access_boost)
        except Exception as e:
            logger.debug(f"Entry recency score failed: {e}")
            return 0.0


class _GuardLock:
    """Non-reentrant lock that records its owning thread.

    S209 WM-deadlock fix: the S164 bug was `add()`/`promote()`/`clear_domain()`
    calling `_save()` while holding this lock, and `_save()` re-acquiring it —
    a re-entrant deadlock on a non-reentrant ``threading.Lock``. This wrapper
    makes that class of bug LOUD instead of silent: `_save()` checks
    ``owned_by_current_thread()`` and RAISES rather than hanging.

    Deliberately NOT ``threading.RLock`` — an RLock would *silently allow* a
    future `_save()`-under-lock (masking the bug), the opposite of what we want.
    Drop-in for ``threading.Lock`` (supports the context-manager + acquire/release
    API the rest of this module uses, including external ``with wm._lock:`` callers).
    """

    __slots__ = ("_lock", "_owner")

    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self._owner = threading.get_ident()
        return acquired

    def release(self) -> None:
        self._owner = None
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def locked(self) -> bool:
        return self._lock.locked()

    def owned_by_current_thread(self) -> bool:
        return self._owner is not None and self._owner == threading.get_ident()


class WorkingMemory:
    """
    Hot-tier memory. 50-entry sliding window.
    Lowest-recency-score entries are evicted when capacity is reached.
    Crash-safe via atomic writes (.tmp + os.replace). Thread-safe via fcntl.flock (S121).

    S190 M9 INVARIANT: `_load()` and `_save()` open SEPARATE FDs against the same
    `.lock` file. Per CLAUDE.md anti-pattern #20 (Linux 6.8+), holding flock from
    two FDs in the same process against the same file BLOCKS. The current code
    is safe because the two paths are never executed concurrently from the same
    thread — but DO NOT call `_save()` from inside the `_load()` lock block, or
    vice versa, or you will deadlock the calling thread.

    S209 WRITE-BEHIND CONTRACT: mutators (add/promote/clear_domain/_remove_entry)
    MUST NOT call `_save()` while holding `self._lock`. They set `self._dirty`
    under the lock, then call `self._signal_dirty()` OUTSIDE it; the `wm-flusher`
    thread persists. `self._lock` is a non-reentrant `_GuardLock`, and `_save()`
    RAISES (does not hang) if invoked while the current thread holds it — so the
    S164 deadlock (add()->_save()-under-lock) is impossible by construction.
    """

    def __init__(self):
        self._entries: Dict[str, WorkingMemoryEntry] = {}
        self._dirty = False
        # S209: non-reentrant owner-tracking lock (see _GuardLock). Re-entrant
        # _save()-under-lock now RAISES loud instead of deadlocking.
        self._lock = _GuardLock()  # S141: Thread safety for _entries (matches procedural/meta/registry)
        self._on_expire_callback = None  # Set by MemoryEngine to promote before purge
        # S209 write-behind: persistence is decoupled from the critical section.
        # Mutators mark _dirty under the lock and signal _flush_event; a dedicated
        # flusher thread persists OUTSIDE the lock so the critical section never
        # touches disk and can never re-enter _save() → the S164 deadlock is
        # structurally impossible.
        self._flush_event = threading.Event()
        self._stop_event = threading.Event()
        self._dirty_since: Optional[float] = None      # epoch when the current dirty run began
        self._last_persist_ok_ts: Optional[float] = None  # epoch of last successful disk write
        self._last_persist_error: Optional[str] = None    # last write error, None = healthy
        self._ensure_dirs()
        self._load()
        self._flusher = threading.Thread(
            target=self._flusher_loop, name="wm-flusher", daemon=True
        )
        self._flusher.start()

    def _ensure_dirs(self):
        WORKING_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if WORKING_MEMORY_PATH.exists():
            lock_path = str(WORKING_MEMORY_PATH) + ".lock"
            lock_fd = None
            try:
                lock_fd = open(lock_path, "a")
                fcntl.flock(lock_fd, fcntl.LOCK_SH)
                with open(WORKING_MEMORY_PATH, "r") as f:
                    raw = json.load(f)
                known = {f.name for f in dc_fields(WorkingMemoryEntry)}
                self._entries = {k: WorkingMemoryEntry(**{fk: fv for fk, fv in v.items() if fk in known}) for k, v in raw.items()}
                logger.debug(f"Working memory loaded: {len(self._entries)} entries")
                # Don't purge on load — wait for memory_engine to wire
                # _on_expire_callback so entries can be promoted before deletion
            except Exception as e:
                logger.warning(f"Working memory load failed: {e} — starting fresh")
                self._entries = {}
            finally:
                if lock_fd:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        lock_fd.close()
                    except Exception as e:
                        logger.debug(f"Lock cleanup: {e}")

    def _save(self):
        """Synchronous force-save (used by flush()/shutdown). MUST be called
        OUTSIDE the entries lock. Snapshots under the lock, writes outside it.

        S209: the re-entrancy guard converts the S164 deadlock class into a LOUD
        RuntimeError. If a future caller invokes _save() while holding self._lock
        (the add()/promote()/clear_domain() bug), this raises instead of hanging.
        """
        if self._lock.owned_by_current_thread():
            raise RuntimeError(
                "WorkingMemory._save() called while holding the entries lock — "
                "re-entrant deadlock prevented (S164 bug class). Mutators must use "
                "write-behind: set _dirty under the lock + _flush_event.set() OUTSIDE it; "
                "_save()/flush() run only outside the lock."
            )
        # S121 W-41: fcntl.flock for inter-process safety
        # S141: threading.Lock for intra-process safety (snapshot under lock, write outside)
        with self._lock:
            snapshot = {k: asdict(v) for k, v in self._entries.items()}
            self._dirty = False
        ok = self._write_snapshot(snapshot)
        if ok:
            self._last_persist_ok_ts = time.time()
            self._last_persist_error = None
            self._dirty_since = None
        else:
            # Write failed — re-arm the dirty flag so the flusher retries.
            with self._lock:
                self._dirty = True
        return ok

    def _write_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """Persist a pre-built snapshot to disk atomically. Holds NO threading
        lock (only the inter-process fcntl lock). Returns True on success.
        Never re-acquires self._lock — safe to call from any thread."""
        lock_path = str(WORKING_MEMORY_PATH) + ".lock"
        lock_fd = None
        try:
            lock_fd = open(lock_path, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            tmp_path = str(WORKING_MEMORY_PATH) + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(snapshot, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(WORKING_MEMORY_PATH))
            return True
        except Exception as e:
            logger.error(f"Working memory save failed: {e}")
            self._last_persist_error = str(e)[:200]
            return False
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception as e:
                    logger.debug(f"Lock cleanup: {e}")

    def _flusher_loop(self):
        """Write-behind flusher. Wakes on every mutation (event) and periodically
        as a safety net for read-triggered dirties (get()/access-count) that mark
        _dirty without signalling. Persists OUTSIDE the entries lock."""
        while not self._stop_event.is_set():
            self._flush_event.wait(timeout=_FLUSH_MAX_INTERVAL)
            if self._stop_event.is_set():
                break
            # Clear BEFORE snapshotting so a mutation racing in after our snapshot
            # re-sets the event and gets caught on the next iteration (no lost write).
            self._flush_event.clear()
            try:
                self._persist_if_dirty()
            except Exception as e:
                logger.error("Working memory flusher iteration failed: %s", e)

    def _persist_if_dirty(self):
        """Snapshot (under lock) + write (outside lock) iff dirty. Flusher worker."""
        with self._lock:
            if not self._dirty:
                return
            snapshot = {k: asdict(v) for k, v in self._entries.items()}
            self._dirty = False
        ok = self._write_snapshot(snapshot)
        if ok:
            self._last_persist_ok_ts = time.time()
            self._last_persist_error = None
            self._dirty_since = None
        else:
            # Re-arm so the next periodic wakeup (or next mutation) retries.
            # Do NOT re-signal the event here — that would busy-spin on a
            # persistent write failure; the _FLUSH_MAX_INTERVAL wakeup retries.
            with self._lock:
                self._dirty = True

    def _signal_dirty(self):
        """Mark the start of a dirty run (for staleness math) + wake the flusher.
        Call OUTSIDE the entries lock, after a mutation has set self._dirty."""
        if self._dirty_since is None:
            self._dirty_since = time.time()
        self._flush_event.set()

    def _remove_entry(self, entry_id: str) -> None:
        """Remove a single entry (used by the liveness probe to clean up its
        synthetic round-trip entry). Write-behind like every other mutator."""
        removed = False
        with self._lock:
            if entry_id in self._entries:
                del self._entries[entry_id]
                self._dirty = True
                removed = True
        if removed:
            self._signal_dirty()

    def stop(self):
        """Stop the flusher after a final synchronous flush. Best-effort; for
        clean shutdown and test teardown. Idempotent."""
        self._stop_event.set()
        self._flush_event.set()
        try:
            self.flush()
        except Exception as e:
            logger.debug(f"WorkingMemory.stop flush failed: {e}")
        try:
            if self._flusher.is_alive():
                self._flusher.join(timeout=2.0)
        except Exception as e:
            logger.debug(f"WorkingMemory.stop join failed: {e}")

    def persist_health(self) -> Dict[str, Any]:
        """Persistence-path health for the consumer-side liveness assertion.

        Reflects the REAL write path's last success/failure + flusher liveness —
        NOT a file-age proxy (file mtime can be recent for the wrong reasons;
        see the Phase-4 'file-age != liveness' rule). 'broken' is True when the
        flusher thread is dead, the last write errored, or a dirty state has gone
        unpersisted longer than _PERSIST_STALE_SECONDS.
        """
        now = time.time()
        pending_for = (now - self._dirty_since) if self._dirty_since is not None else 0.0
        flusher_alive = bool(self._flusher.is_alive())
        broken = (
            (not flusher_alive)
            or (self._last_persist_error is not None)
            or (self._dirty and pending_for > _PERSIST_STALE_SECONDS)
        )
        if broken:
            if not flusher_alive:
                reason = "flusher thread is dead"
            elif self._last_persist_error is not None:
                reason = f"last write errored: {self._last_persist_error}"
            else:
                reason = f"dirty unpersisted for {pending_for:.0f}s (> {_PERSIST_STALE_SECONDS:.0f}s)"
        else:
            reason = ""
        return {
            "broken": broken,
            "reason": reason,
            "flusher_alive": flusher_alive,
            "dirty": self._dirty,
            "pending_seconds": round(pending_for, 1),
            "last_persist_ok_ts": self._last_persist_ok_ts,
            "last_persist_error": self._last_persist_error,
        }

    def _mark_dirty(self):
        self._dirty = True

    def mark_dirty(self):
        """Thread-safe: mark state as needing flush (public API for encapsulation)."""
        with self._lock:
            self._mark_dirty()

    def flush(self):
        """Force-save if dirty. Called by export_hot_context and shutdown."""
        with self._lock:
            needs_save = self._dirty
        if needs_save:
            self._save()

    def purge_after_init(self):
        """Purge expired entries after callback wiring. Call from memory_engine.__init__."""
        self._purge_expired()

    def _purge_expired(self):
        """Remove expired entries. Promotes qualifying entries before deletion."""
        # S141: Collect expired + delete under lock; callbacks OUTSIDE lock (avoid deadlock)
        promote_candidates = []
        with self._lock:
            expired = [k for k, v in self._entries.items() if v.is_expired()]
            for k in expired:
                entry = self._entries[k]
                if self._on_expire_callback and not entry.promoted:
                    if entry.access_count >= 3 and entry.priority >= 0.3:
                        promote_candidates.append(entry)
                del self._entries[k]
            if expired:
                self._dirty = True
        # Callbacks outside lock — must NOT re-acquire _lock (non-reentrant threading.Lock).
        # Currently safe: callback calls episodic.record() which uses fcntl, not threading.Lock.
        promoted_count = 0
        for entry in promote_candidates:
            try:
                self._on_expire_callback(entry)
                promoted_count += 1
            except Exception as e:
                logger.debug(f"Working memory: expire callback failed for {entry.id[:12]}: {e}")
        if expired:
            logger.debug(f"Working memory: purged {len(expired)} expired entries ({promoted_count} promoted)")

    def _evict_if_needed(self):
        """Evict lowest-scored entries if over capacity."""
        # S141: Evict under lock; callbacks OUTSIDE lock
        promote_candidates = []
        evict_count = 0
        with self._lock:
            if len(self._entries) >= MAX_ENTRIES:
                sorted_entries = sorted(
                    self._entries.items(),
                    key=lambda x: x[1].recency_score()
                )
                evict_count = max(1, len(sorted_entries) // 10)
                for k, entry in sorted_entries[:evict_count]:
                    if self._on_expire_callback and not entry.promoted:
                        if entry.access_count >= 3 and entry.priority >= 0.3:
                            promote_candidates.append(entry)
                    del self._entries[k]
                self._dirty = True
        # Callbacks outside lock
        promoted_count = 0
        for entry in promote_candidates:
            try:
                self._on_expire_callback(entry)
                promoted_count += 1
            except Exception as e:
                logger.debug(f"Working memory: eviction callback failed for {entry.id[:12]}: {e}")
        if evict_count:
            logger.debug(f"Working memory: evicted {evict_count} entries ({promoted_count} promoted)")

    # Conversational filler patterns — not worth storing in working memory
    _FILLER_PREFIXES = (
        "what,", "also,", "okay", "hmm", "oh,", "oh ", "yeah", "no,",
        "sure", "yes,", "yes ", "please ", "sorry ", "thanks", "thank you",
        "ok,", "ok ", "got it", "and yes", "and no",
    )

    def add(self, content: str, domain: str = "general", priority: float = 0.5,
            tags: Optional[List[str]] = None, ttl_hours: Optional[float] = None) -> str:
        """Add an entry. Returns entry ID (empty string if skipped)."""
        stripped = content.strip()
        # Strip conv_XXXXXXXX: prefix before length check (actual content matters)
        _actual = stripped.split(": ", 1)[-1] if ": " in stripped else stripped
        if len(_actual) < 10:
            logger.debug(f"Working memory: skipping short entry ({len(_actual)} chars actual)")
            return ""

        # Skip garbage entries (repeated characters/words)
        unique_chars = set(stripped.replace(" ", ""))
        if len(unique_chars) <= 3 and len(stripped) >= 20:
            logger.debug(f"Working memory: skipping garbage entry (only {len(unique_chars)} unique chars)")
            return ""

        # Skip repetitive word entries like "B B B B B B" (Fix S103-2B)
        words = stripped.split()
        if len(words) >= 4:
            unique_words = set(w.lower() for w in words)
            if len(unique_words) <= 2:
                logger.debug(f"Working memory: skipping repetitive entry ({len(unique_words)} unique words)")
                return ""

        # Skip conversational filler (Fix C, Session 89)
        # Extract user message from "conv_XXXXX: <message>" format
        filler_check = stripped
        if ": " in filler_check:
            filler_check = filler_check.split(": ", 1)[1]
        filler_lower = filler_check.lower().strip()
        if len(filler_check) < 50:
            for prefix in self._FILLER_PREFIXES:
                if filler_lower.startswith(prefix):
                    logger.debug(f"Working memory: skipping filler entry: {stripped[:60]}")
                    return ""

        # Content dedup: skip if identical active entry exists
        with self._lock:
            for existing in self._entries.values():
                if not existing.is_expired() and existing.content == content and existing.domain == domain:
                    logger.debug(f"Working memory dedup: skipping duplicate in domain={domain}")
                    return ""

        self._purge_expired()
        self._evict_if_needed()

        # R-32(b) (S215): coerce at the write site so the persisted JSON keeps float types.
        try:
            priority = float(priority)
        except (TypeError, ValueError):
            priority = 0.5
        if not math.isfinite(priority):
            priority = 0.5

        entry_id = f"wm_{int(time.time() * 1000)}_{random.randint(0, 999):03d}_{domain[:4]}"
        now = datetime.now(timezone.utc)
        ttl = ttl_hours or TTL_HOURS
        expires = now + timedelta(hours=ttl)

        entry = WorkingMemoryEntry(
            id=entry_id,
            content=content,
            domain=domain,
            priority=priority,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            tags=tags or [],
        )
        with self._lock:
            self._entries[entry_id] = entry
            self._dirty = True
        # S209 write-behind: persistence runs on the flusher thread, OUTSIDE this
        # lock. The critical section does NO IO and NO re-entrant _save() call, so
        # the S164 add()->_save()-under-lock deadlock is structurally impossible.
        self._signal_dirty()
        logger.debug(f"Working memory: added entry {entry_id[:12]} (domain={domain}, priority={priority})")
        return entry_id

    def get(self, entry_id: str) -> Optional[WorkingMemoryEntry]:
        """Retrieve an entry by ID."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry and not entry.is_expired():
                entry.access()
                self._dirty = True
                return entry
        return None

    def get_by_domain(self, domain: str, limit: int = 10) -> List[WorkingMemoryEntry]:
        """Get top entries for a domain, sorted by recency score."""
        self._purge_expired()
        with self._lock:
            domain_entries = [e for e in self._entries.values() if e.domain == domain]
        return sorted(domain_entries, key=lambda x: x.recency_score(), reverse=True)[:limit]

    def get_context(self, max_chars: int = 800) -> str:
        """
        Generate compact context string for system prompt injection.
        Returns highest-priority entries across all domains.
        Does NOT increment access_count — bulk context reads are not real access.
        Only targeted get() lookups should count as access (Fix B, Session 89).
        """
        self._purge_expired()
        with self._lock:
            if not self._entries:
                return ""
            # S209: exclude synthetic liveness-probe entries (domain "_liveness")
            # from prompt context — they exist only to exercise the write→read path.
            sorted_entries = sorted(
                (e for e in self._entries.values() if e.domain != "_liveness"),
                key=lambda x: x.recency_score(),
                reverse=True
            )

        lines = ["[Working Memory — Hot Context]"]
        char_count = len(lines[0])

        for entry in sorted_entries:
            line = f"• [{entry.domain}] {entry.content[:200]}"
            if char_count + len(line) > max_chars:
                break
            lines.append(line)
            char_count += len(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    def promote(self, entry_id: str) -> Optional[WorkingMemoryEntry]:
        """Mark entry as promoted to episodic — keeps it from re-promotion."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry:
                entry.promoted = True
                self._dirty = True
        if entry is not None:
            self._signal_dirty()  # S209 write-behind: persist OUTSIDE the lock
        return entry

    def get_promotable(self, min_access: int = 5, min_priority: float = 0.5) -> List[WorkingMemoryEntry]:
        """Return entries eligible for promotion to episodic memory.
        Criteria: access_count >= min_access AND priority >= min_priority AND not already promoted.
        (Fix A, Session 89; threshold lowered from 0.6 to 0.5, Session 98)
        """
        with self._lock:
            return [
                e for e in self._entries.values()
                if e.access_count >= min_access
                and e.priority >= min_priority
                and not e.promoted
                and not e.is_expired()
            ]

    def clear_domain(self, domain: str) -> int:
        """Clear all entries for a domain. Returns count removed."""
        with self._lock:
            before = len(self._entries)
            self._entries = {k: v for k, v in self._entries.items() if v.domain != domain}
            removed = before - len(self._entries)
            if removed:
                self._dirty = True
        if removed:
            self._signal_dirty()  # S209 write-behind: persist OUTSIDE the lock
        return removed

    def stats(self) -> Dict[str, Any]:
        """Return working memory statistics."""
        self._purge_expired()
        with self._lock:
            domain_counts: Dict[str, int] = {}
            for e in self._entries.values():
                domain_counts[e.domain] = domain_counts.get(e.domain, 0) + 1
            return {
                "total_entries": len(self._entries),
                "max_capacity": MAX_ENTRIES,
                "utilization_pct": round(len(self._entries) / MAX_ENTRIES * 100, 1),
                "by_domain": domain_counts,
                "ttl_hours": TTL_HOURS,
            }

