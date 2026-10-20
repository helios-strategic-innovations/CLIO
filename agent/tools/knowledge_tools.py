"""
OpenClaw/Moltbot — Knowledge Management Tool Handlers

Persistent knowledge base with structured directory hierarchy,
flat JSON master index, and dual-write for trading categories.

Every dual-write into TRADINGBOT_MIRROR_DIR (the retired Hydrinia bridge) is
gated on agent.hydrinia_bridge.bridge_enabled() (default False, operator
direction 2026-09-24) — CLIO's own knowledge base write always happens;
only the Hydrinia mirror is skipped.
"""

import asyncio
import fcntl
import json
import logging
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger("openclaw.knowledge")

_NY = ZoneInfo("America/New_York")


def _bridge_enabled() -> bool:
    try:
        from agent.hydrinia_bridge import bridge_enabled
        return bridge_enabled()
    except Exception as e:
        logger.debug(f"Hydrinia bridge check failed, skipping mirror write: {e}")
        return False

# ============================================================
# Directory Configuration
# ============================================================

KNOWLEDGE_BASE_DIR = "/home/agent/openclaw/data/knowledge"
KNOWLEDGE_INDEX_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "knowledge_index.json")
KNOWLEDGE_ARCHIVE_INDEX_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "knowledge_index_archive.json")
TRADINGBOT_MIRROR_DIR = "/home/agent/tradingbot/moltbot_knowledge"
CURRENT_PRIORS_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "current_priors.md")

# Knowledge expiry — two-tier: quality-gated at 10 days, hard cap at 21 days
KNOWLEDGE_EXPIRY_DAYS_QUALITY = 10   # archive low-quality entries after 10 days
KNOWLEDGE_EXPIRY_DAYS_HARD = 21      # archive everything after 21 days
KNOWLEDGE_QUALITY_THRESHOLD = 6      # entries below this score get early archival
KNOWLEDGE_ARCHIVE_BATCH_SIZE = 30
KNOWLEDGE_ARCHIVE_INTERVAL_HOURS = 6
# P1-7: hard cap on the archive index. It grows unbounded and is fully rewritten
# each archival run; without a ceiling the rewrite cost climbs forever. Metadata
# only — capping drops the oldest index entries, never their on-disk files.
KNOWLEDGE_ARCHIVE_MAX_ENTRIES = 10000
_ARCHIVE_EXEMPT_CATEGORIES = {"trading_bot_status"}

# S148 C-1: Promoted to module-level so seal_knowledge_enhancer and contradiction_detector
# can import it (was local to _seal_impl_quality, causing ImportError fallback to empty set)
_NON_TICKER = frozenset({
    "IF", "THEN", "AND", "OR", "THE", "FOR", "BUT", "NOT", "ALL", "ANY",
    "MAY", "CAN", "GDP", "CPI", "FED", "IMF", "USD", "EUR", "BIS", "SEC",
    "ETF", "IPO", "CEO", "CFO", "BUY", "PUT", "OTC",
})


# ============================================================
# S189 C1: Slop-phrase post-processor (LOG_ONLY default per ADR-003 P2)
# ============================================================
# Detects reflex hedge language ("remains elevated/impaired/uncertain") that
# lacks a numeric anchor in the same sentence. Such sentences add no decision
# value — they're filler. S189 audit found 87 occurrences across the knowledge
# base. Operator promotes from LOG_ONLY to active by setting
# CLIO_SLOP_FILTER_LOG_ONLY=0 in .env.

_SLOP_HEDGE_RE = re.compile(
    r"\b(?:remains?|stays?|continues?\s+to\s+remain)\s+"
    r"(?:elevated|impaired|uncertain|firm|cautious|balanced|tight|loose|"
    r"volatile|stable|fragile|firm-ish|mixed)\b",
    re.IGNORECASE,
)
# Numeric anchor: any digit cluster (with optional %, $, bp/bps/pct suffixes
# OR spelled-out units like "basis points", "percent", "billion"). Tightened
# via S189 UL-3 — the old regex over-flagged content like "+18 basis points"
# because "basis points" wasn't in the suffix list. Spelled-out units are
# common in intel narrative prose.
_NUMERIC_ANCHOR_RE = re.compile(
    r"(?:"
        r"\$\d"  # $-prefixed number
        r"|\d+(?:[.,]\d+)?\s*"
        r"(?:%|bps?|pct|bp|pp|cents?|dollars?|bn|trn|mn|k|m|x"
            r"|basis\s+points?"
            r"|percent(?:age)?(?:\s+points?)?"
            r"|million|billion|trillion)"
    r")",
    re.IGNORECASE,
)
# Sentence splitter: stop at . ! ? but not inside abbreviations like "U.S."
# Keeping it simple — full NLP not warranted for hedge detection.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z\(\"\'])")


def _slop_filter_active() -> bool:
    """True only when operator has flipped CLIO_SLOP_FILTER_LOG_ONLY to 0."""
    return os.environ.get("CLIO_SLOP_FILTER_LOG_ONLY", "1") == "0"


def _detect_slop_sentences(text: str) -> list[str]:
    """Return slop sentences (hedge phrase without numeric anchor in same sentence).

    Returns the offending sentence strings (truncated to 200 chars each).
    Empty list if content is clean.
    """
    if not text or len(text) < 20:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    flagged: list[str] = []
    for s in sentences:
        s_strip = s.strip()
        if not s_strip:
            continue
        if _SLOP_HEDGE_RE.search(s_strip) and not _NUMERIC_ANCHOR_RE.search(s_strip):
            flagged.append(s_strip[:200])
    return flagged


def _filter_slop_from_text(text: str) -> str:
    """Remove slop sentences from text (only called when filter active)."""
    if not text:
        return text
    sentences = _SENTENCE_SPLIT_RE.split(text)
    kept: list[str] = []
    for s in sentences:
        s_strip = s.strip()
        if not s_strip:
            continue
        if _SLOP_HEDGE_RE.search(s_strip) and not _NUMERIC_ANCHOR_RE.search(s_strip):
            continue  # drop slop sentence
        kept.append(s_strip)
    return " ".join(kept)


def _seal_impl_quality(impl_text: str) -> int:
    """Dynamic quality score for SEAL implications based on specificity (S142-B W2-C).

    Base 5 + bonus for: tickers (+1), timeframes (+1), numeric levels (+1). Range 5-8.
    """
    score = 5
    _ticker_match = any(
        m.group() not in _NON_TICKER
        for m in re.finditer(r'\b[A-Z]{2,5}\b', impl_text)
    )
    if _ticker_match:
        score += 1  # Specific ticker mentioned (excludes common uppercase non-ticker words)
    if re.search(r'\b\d+[- ]?(day|week|month|quarter|year|[dwmqy])s?\b', impl_text, re.IGNORECASE):
        score += 1  # Timeframe mentioned
    if re.search(r'\$\d+|\d+%|\d+\.\d{2,}', impl_text):
        score += 1  # Numeric level/target mentioned
    return min(score, 8)


# Category → relative directory path
CATEGORY_DIRS = {
    "trading_strategies": "trading/strategies",
    "trading_trades": "trading/trades",
    "trading_market_analysis": "trading/market_analysis",
    "trading_conversations": "trading/conversations",
    "trading_bot_status": "trading/bot_status",
    "personal_preferences": "personal/preferences",
    "personal_conversations": "personal/conversations",
    "personal_notes": "personal/notes",
    "email_gmail_personal": "email/gmail_personal",
    "email_atlas_personal": "email/atlas_personal",
    "email_atlas_bot": "email/atlas_bot",
    "email_gmail_bot": "email/gmail_bot",
    "academic_courses": "academic/courses",
    "academic_research": "academic/research",
    "career_applications": "career/applications",
    "career_networking": "career/networking",
    "system_lessons": "system_lessons",  # ANVIL error pipeline (not in LLM tool schema)
}

# Categories that get dual-written to tradingbot mirror
TRADING_CATEGORIES = {k for k in CATEGORY_DIRS if k.startswith("trading_")}

MAX_CONTENT_LENGTH = 50_000

# Urgent channel — keywords that trigger immediate mirror to urgent/ for Hydrinia
URGENT_KEYWORDS = {
    "urgent", "critical", "breaking", "emergency", "crisis", "halt",
    "closed", "blockade", "war", "invasion", "crash", "circuit breaker",
}
URGENT_MIRROR_DIR = os.path.join(TRADINGBOT_MIRROR_DIR, "urgent")


# ============================================================
# Helpers
# ============================================================

def _sanitize_filename(title: str) -> str:
    """Convert title to safe filename component (lowercase, underscores, max 60 chars)."""
    sanitized = re.sub(r"[^a-zA-Z0-9_\s]", "", title)
    sanitized = re.sub(r"\s+", "_", sanitized).strip("_").lower()
    return sanitized[:60] if sanitized else "untitled"


_STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "will", "more",
    "when", "who", "did", "get", "may", "its", "than", "them", "some", "what",
    "this", "that", "with", "from", "your", "they", "about", "would", "there",
    "their", "which", "could", "other", "into", "just", "also", "how", "any",
}


def _extract_topic_keywords(text: str) -> list[str]:
    """Extract topic keywords from text for knowledge matching. Max 10."""
    words = set()
    for w in text.lower().split():
        w = re.sub(r"[^a-z0-9]", "", w)
        if len(w) >= 3 and w not in _STOP_WORDS:
            words.add(w)
    return list(words)[:10]


def _ensure_dirs() -> None:
    """Create all category directories if they don't exist."""
    for rel_dir in CATEGORY_DIRS.values():
        os.makedirs(os.path.join(KNOWLEDGE_BASE_DIR, rel_dir), exist_ok=True)
    # Trading mirror dirs — retired Hydrinia bridge, gated off by default.
    if not _bridge_enabled():
        return
    for cat, rel_dir in CATEGORY_DIRS.items():
        if cat in TRADING_CATEGORIES:
            # Strip "trading/" prefix for mirror
            mirror_subdir = rel_dir.split("/", 1)[1] if "/" in rel_dir else rel_dir
            os.makedirs(os.path.join(TRADINGBOT_MIRROR_DIR, mirror_subdir), exist_ok=True)


def _atomic_write_json(path: str, data) -> None:
    """Write JSON atomically via .tmp + fsync + os.replace()."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
        # P0-9: flush Python + OS buffers to disk BEFORE the rename. Without this,
        # a kernel crash / power loss can commit the rename while the tmp file's
        # data blocks are still only in the page cache — the file then recovers as
        # zero-length or torn. Model: audit_log.py:262-263. ~ms cost per write.
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _load_index() -> dict:
    """Load knowledge index. Returns empty structure on missing/corrupt file."""
    if not os.path.isfile(KNOWLEDGE_INDEX_FILE):
        return {"version": 1, "entries": []}
    try:
        with open(KNOWLEDGE_INDEX_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "entries" not in data or not isinstance(data.get("entries"), list):
            return {"version": 1, "entries": []}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Knowledge index corrupt/unreadable: {e}")
        return {"version": 1, "entries": []}


# Public alias for cross-module access (learning_engine uses this)
load_index = _load_index

_INDEX_LOCK_FILE = KNOWLEDGE_INDEX_FILE + ".lock"



# ============================================================
# Knowledge Archival
# ============================================================

def _load_archive_index() -> dict:
    """Load archive knowledge index. Returns empty structure on missing/corrupt."""
    if not os.path.isfile(KNOWLEDGE_ARCHIVE_INDEX_FILE):
        return {"version": 1, "entries": [], "last_archival": ""}
    try:
        with open(KNOWLEDGE_ARCHIVE_INDEX_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "entries" not in data:
            return {"version": 1, "entries": [], "last_archival": ""}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Archive index corrupt/unreadable: {e}")
        return {"version": 1, "entries": [], "last_archival": ""}


def load_all_entries() -> list:
    """Load entries from both active and archive indexes. For synthesis/historical queries."""
    active = _load_index()
    archive = _load_archive_index()
    return active.get("entries", []) + archive.get("entries", [])


def run_knowledge_archival() -> dict:
    """
    Move old/low-quality entries from active to archive index (two-tier: 10d+quality, 21d hard).
    Uses file locking for atomicity. Returns stats dict.
    Safe to call repeatedly — checks interval and skips if too soon.
    """
    stats = {"archived": 0, "active_remaining": 0, "skipped_too_soon": False}
    lock_fd = None

    try:
        lock_fd = open(_INDEX_LOCK_FILE, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # Check interval — skip if too soon
        archive = _load_archive_index()
        last_archival = archive.get("last_archival", "")
        if last_archival:
            try:
                last_dt = datetime.fromisoformat(last_archival)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=_NY)
                hours_since = (datetime.now(_NY) - last_dt).total_seconds() / 3600
                if hours_since < KNOWLEDGE_ARCHIVE_INTERVAL_HOURS:
                    stats["skipped_too_soon"] = True
                    return stats
            except (ValueError, TypeError):
                pass

        # Load active index
        index = _load_index()
        entries = index.get("entries", [])
        if not entries:
            return stats

        # Partition entries into keep vs archive (two-tier criteria)
        now = datetime.now(_NY)
        cutoff_quality = (now - timedelta(days=KNOWLEDGE_EXPIRY_DAYS_QUALITY)).isoformat()
        cutoff_hard = (now - timedelta(days=KNOWLEDGE_EXPIRY_DAYS_HARD)).isoformat()

        keep = []
        to_archive = []
        for entry in entries:
            category = entry.get("category", "")
            if category in _ARCHIVE_EXEMPT_CATEGORIES:
                keep.append(entry)
                continue
            # Entry-level permanence: durable reference docs (e.g. operator
            # worldview / conversation syntheses) opt out of aging via an
            # explicit per-entry flag, without exempting their whole category —
            # transient entries in the same category still age normally.
            # (Operator decision 2026-07-01 — theology synthesis ingest.)
            if entry.get("archive_exempt") is True:
                keep.append(entry)
                continue
            created = entry.get("created", "")
            if not created:
                keep.append(entry)
                continue
            quality = entry.get("quality_score", 0) or 0
            # Hard cap: archive everything older than 21 days
            if created < cutoff_hard:
                to_archive.append(entry)
            # Quality gate: archive low-quality after 10 days
            elif created < cutoff_quality and quality < KNOWLEDGE_QUALITY_THRESHOLD:
                to_archive.append(entry)
            else:
                keep.append(entry)

        # Respect batch size
        if len(to_archive) > KNOWLEDGE_ARCHIVE_BATCH_SIZE:
            to_archive.sort(key=lambda e: e.get("created", ""))
            overflow = to_archive[KNOWLEDGE_ARCHIVE_BATCH_SIZE:]
            to_archive = to_archive[:KNOWLEDGE_ARCHIVE_BATCH_SIZE]
            keep.extend(overflow)

        if not to_archive:
            stats["active_remaining"] = len(keep)
            return stats

        # Write archive FIRST (idempotent on crash — worst case: duplicates)
        archive["entries"].extend(to_archive)
        # P1-7: bound archive-index growth. Keep the most recent N entries by
        # created-timestamp (ISO-8601 strings sort chronologically). This is
        # index metadata only — the underlying entry files on disk are NOT
        # touched, so knowledge content is never destroyed here.
        _arch_entries = archive.get("entries", [])
        if len(_arch_entries) > KNOWLEDGE_ARCHIVE_MAX_ENTRIES:
            _arch_before = len(_arch_entries)
            _arch_entries.sort(key=lambda e: e.get("created", ""))
            archive["entries"] = _arch_entries[-KNOWLEDGE_ARCHIVE_MAX_ENTRIES:]
            logger.info(
                f"Archive index capped: {_arch_before} -> {KNOWLEDGE_ARCHIVE_MAX_ENTRIES} "
                f"entries (oldest metadata dropped; entry files untouched)"
            )
        archive["last_archival"] = datetime.now(_NY).isoformat()
        _atomic_write_json(KNOWLEDGE_ARCHIVE_INDEX_FILE, archive)

        # Then update active index
        index["entries"] = keep
        _atomic_write_json(KNOWLEDGE_INDEX_FILE, index)

        stats["archived"] = len(to_archive)
        stats["active_remaining"] = len(keep)

        logger.info(
            f"Knowledge archival: moved {stats['archived']} entries to archive, "
            f"{stats['active_remaining']} remain active"
        )
        return stats

    except Exception as e:
        logger.error(f"Knowledge archival failed: {e}", exc_info=True)
        return stats
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception as e:
                logger.debug(f"Fire-and-forget lock release failed: {e}")


def _tag_entry_as_parent_index(parent_id: str, child_count: int) -> bool:
    """S185 W4-2: Tag a knowledge-index entry as 'parent_index' once it has
    spawned enough SEAL implications that the parent thesis is redundant with
    its children in retrieval. Does NOT remove the entry — only annotates it.

    Retrieval code can later honor role=='parent_index' to suppress duplicates.
    Until then, this is an inert tag visible for operator review.

    Returns True if the tag was written, False otherwise.
    """
    if os.environ.get("CLIO_SEAL_PARENT_SUPPRESS_ENABLED", "1") != "1":
        return False
    lock_fd = None
    try:
        lock_fd = open(_INDEX_LOCK_FILE, "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        index = _load_index()
        entries = index.get("entries", [])
        updated = False
        for entry in entries:
            if entry.get("id") == parent_id:
                if entry.get("role") == "parent_index":
                    return False  # already tagged
                entry["role"] = "parent_index"
                entry["parent_of_count"] = child_count
                updated = True
                break
        if updated:
            _atomic_write_json(KNOWLEDGE_INDEX_FILE, index)
            logger.info(
                "SEAL parent tagged: id=%s children=%d (retrieval may de-duplicate)",
                parent_id, child_count,
            )
        return updated
    except Exception as e:
        logger.debug(f"parent-index tag failed: {e}")
        return False
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _append_or_update_index(index_entry: dict, entry_file_path: str, existing_lock_fd=None, index=None) -> int:
    """Append entry to index, or update existing if same title+category within 24h.

    This prevents duplicate entries when the same topic is saved multiple times
    in quick succession (e.g. learning engine re-discovering the same insight).
    Returns new entry count.

    S162 H-4: If existing_lock_fd is provided, skip lock acquisition (caller holds it).

    P1-6: If ``index`` is supplied AND the caller already holds the lock
    (``existing_lock_fd`` is not None), reuse that pre-loaded dict instead of
    re-reading the index from disk. When this function acquires its OWN lock, any
    caller-supplied ``index`` is discarded and re-read under that lock — a dict
    read before we held the lock could be stale and clobber concurrent writes.
    """
    os.makedirs(os.path.dirname(KNOWLEDGE_INDEX_FILE), exist_ok=True)
    lock_fd = None
    _owned_lock = existing_lock_fd is None
    try:
        if _owned_lock:
            lock_fd = open(_INDEX_LOCK_FILE, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        else:
            lock_fd = existing_lock_fd
        # P1-6: reuse the caller's pre-loaded index only when the caller holds the
        # lock (not _owned_lock). If we acquired our own lock above, re-read to
        # avoid writing a stale snapshot over concurrent updates.
        if index is None or _owned_lock:
            index = _load_index()
        entries = index.get("entries", [])

        title_lower = index_entry.get("title", "").lower().strip()
        category = index_entry.get("category", "")
        created_iso = index_entry.get("created", "")

        # Check for existing entry with same title+category within last 24h
        cutoff_iso = ""
        try:
            created_dt = datetime.fromisoformat(created_iso)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=_NY)
            cutoff_dt = created_dt - timedelta(hours=24)
            cutoff_iso = cutoff_dt.isoformat()
        except (ValueError, TypeError):
            pass

        duplicate_idx = None
        if cutoff_iso and title_lower:
            for i, existing in enumerate(entries):
                if (existing.get("category") == category
                        and existing.get("title", "").lower().strip() == title_lower
                        and existing.get("created", "") >= cutoff_iso):
                    duplicate_idx = i
                    break

        if duplicate_idx is not None:
            # Update existing entry in-place (newer data, same slot)
            old_entry = entries[duplicate_idx]
            old_file = os.path.join(KNOWLEDGE_BASE_DIR, old_entry.get("file_path", ""))
            entries[duplicate_idx] = index_entry
            index["entries"] = entries
            _atomic_write_json(KNOWLEDGE_INDEX_FILE, index)
            # Clean up old file if it differs from new
            if old_file != entry_file_path and os.path.isfile(old_file):
                # S159 H-6: Validate path before deletion
                _real_old = os.path.realpath(old_file)
                _real_base = os.path.realpath(KNOWLEDGE_BASE_DIR) + os.sep
                if _real_old.startswith(_real_base):
                    try:
                        os.remove(old_file)
                    except OSError:
                        pass
                else:
                    logger.warning(f"S159 H-6: Refused to delete file outside knowledge base: {old_file}")
            logger.info(f"Knowledge dedup: updated existing entry '{title_lower[:60]}' [{category}]")
            return len(entries)
        else:
            entries.append(index_entry)
            index["entries"] = entries
            _atomic_write_json(KNOWLEDGE_INDEX_FILE, index)
            return len(entries)
    finally:
        if _owned_lock and lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()


# ============================================================
# Tool Handlers
# ============================================================

async def handle_save_knowledge(p: dict) -> str:
    """Save a knowledge entry to the persistent knowledge base."""
    title = p.get("title", "").strip()
    category = p.get("category", "")
    content = p.get("content", "").strip()
    tags = p.get("tags", [])
    source = p.get("source", "conversation")

    # Validation
    if not title:
        return "ERROR: title is required."
    if not content:
        return "ERROR: content is required."
    if category not in CATEGORY_DIRS:
        return f"ERROR: Invalid category '{category}'. Valid: {', '.join(sorted(CATEGORY_DIRS.keys()))}"
    truncated = len(content) > MAX_CONTENT_LENGTH
    if truncated:
        content = content[:MAX_CONTENT_LENGTH]
        logger.warning(f"Knowledge content truncated to {MAX_CONTENT_LENGTH} chars")
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if t]

    # S189 C1: Slop-phrase detection (LOG_ONLY default — operator promotes via env).
    # Catches reflex hedge language without numeric anchors.
    try:
        _slop_sentences = _detect_slop_sentences(content)
        if _slop_sentences:
            _slop_active = _slop_filter_active()
            _mode = "stripped" if _slop_active else "LOG_ONLY"
            logger.warning(
                f"Slop filter [{_mode}] '{title[:60]}': {len(_slop_sentences)} "
                f"hedge sentence(s) without numeric anchor. First: "
                f"{_slop_sentences[0][:120]!r}"
            )
            if _slop_active:
                _filtered = _filter_slop_from_text(content)
                # S189-pass2 safety: never strip down to empty or trivially-tiny
                # content. If the filter would remove >75% OR leave <50 chars,
                # log additional WARNING and keep the original. Protects against
                # the bullet-list / all-slop edge case where the entire content
                # parses as a single sentence (no proper punctuation) and the
                # whole blob matches the slop regex.
                _min_keep = max(50, len(content) // 4)
                if len(_filtered) >= _min_keep:
                    content = _filtered
                else:
                    logger.warning(
                        f"Slop filter [active] '{title[:60]}': filter would "
                        f"reduce {len(content)} → {len(_filtered)} chars; "
                        f"keeping original (threshold ≥{_min_keep} chars). "
                        f"Likely cause: bullet list or all-slop content with no "
                        f"sentence punctuation."
                    )
    except Exception as e:
        logger.debug(f"Slop filter check failed (non-critical): {e}")

    # Skip operational telemetry — these pollute the knowledge base
    _SKIP_TITLE_PATTERNS = ("Trade Feedback", "Knowledge Improvements", "Test Entry", "Audit Verification")
    if any(title.startswith(pat) for pat in _SKIP_TITLE_PATTERNS):
        if category == "trading_bot_status":
            pass  # Allow bot_status saves — expected telemetry category
        else:
            logger.info(f"Knowledge skip: '{title}' — operational content, not market knowledge")
            return f"Skipped: operational content (use trading_bot_status category for telemetry)"

    # Title exact-match dedup: skip if same title exists within last 3 days (any category)
    # Catches patterns like "Options Research — Feb 26" x7 that similarity check misses
    # Skip this check when update_existing=true (upsert intentionally targets same title)
    # S118: Uses LOCK_EX for dedup check, released before index update to prevent flock deadlock
    title_lower = title.lower().strip()

    # P0-1: run the flock-holding synchronous region (dedup + upsert + create +
    # index write + contradiction check) in a thread executor, so a concurrent
    # cross-process LOCK_EX holder can never freeze the event loop (the S163-B
    # class). The SEAL enrichment await stays on the loop, after _do_save() returns.
    _seal_ctx = {}

    def _do_save() -> str:
        index = None  # P1-6: pre-load reused by _append_or_update_index under the held lock
        _create_lock_fd = None  # Held through dedup check + create path for non-upsert
        if not p.get("update_existing"):
            try:
                _create_lock_fd = open(_INDEX_LOCK_FILE, "a")
                fcntl.flock(_create_lock_fd, fcntl.LOCK_EX)
                index = _load_index()
                cutoff_3d = (datetime.now(_NY) - timedelta(days=3)).isoformat()
                for existing in index.get("entries", []):
                    if (existing.get("title", "").lower().strip() == title_lower
                            and existing.get("created", "") >= cutoff_3d):
                        logger.info(f"Knowledge title dedup: skipping '{title[:60]}' — exact match within 3 days")
                        if _create_lock_fd:
                            try:
                                fcntl.flock(_create_lock_fd, fcntl.LOCK_UN)
                            except Exception as e:
                                logger.debug(f"Lock cleanup: {e}")
                            _create_lock_fd.close()
                        return f"Skipped: duplicate title within 3 days (existing entry)"
            except Exception as e:
                logger.debug(f"Fire-and-forget title dedup check failed: {e}")
                # Release lock on dedup error — create path proceeds unlocked (best-effort)
                if _create_lock_fd:
                    try:
                        _create_lock_fd.close()
                    except Exception as e:
                        logger.debug(f"FD close: {e}")
                    _create_lock_fd = None

        # Upsert mode: find and update existing entry with same title
        # S100: entire upsert under exclusive lock to prevent concurrent update races
        if p.get("update_existing"):
            lock_fd = None
            try:
                lock_fd = open(_INDEX_LOCK_FILE, "a")
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

                upsert_index = _load_index()
                upsert_entries = upsert_index.get("entries", [])
                # Find all title matches (case-insensitive), pick most recent
                matches = []
                for i, existing in enumerate(upsert_entries):
                    if existing.get("title", "").lower().strip() == title_lower:
                        matches.append((i, existing))
                if matches:
                    matches.sort(key=lambda x: x[1].get("created", ""), reverse=True)
                    match_idx, match_entry = matches[0]
                    existing_file = os.path.join(KNOWLEDGE_BASE_DIR, match_entry.get("file_path", ""))
                    now = datetime.now(_NY)

                    # Read existing file and update content
                    existing_data = {}
                    if os.path.isfile(existing_file):
                        try:
                            with open(existing_file, "r") as f:
                                existing_data = json.load(f)
                        except (json.JSONDecodeError, OSError):
                            pass

                    existing_data["content"] = content
                    existing_data["tags"] = tags
                    existing_data["source"] = source
                    existing_data["updated"] = now.isoformat()
                    _atomic_write_json(existing_file, existing_data)

                    # Update index entry (already under lock)
                    idx_entries = upsert_index.get("entries", [])
                    for i, e in enumerate(idx_entries):
                        if e.get("id") == match_entry.get("id"):
                            idx_entries[i]["tags"] = tags
                            idx_entries[i]["source"] = source
                            idx_entries[i]["updated"] = now.isoformat()
                            break
                    upsert_index["entries"] = idx_entries
                    _atomic_write_json(KNOWLEDGE_INDEX_FILE, upsert_index)

                    # Mirror update for trading categories (retired Hydrinia bridge)
                    match_category = match_entry.get("category", "")
                    if match_category in TRADING_CATEGORIES and _bridge_enabled():
                        try:
                            match_rel = match_entry.get("file_path", "")
                            match_filename = os.path.basename(match_rel)
                            rel_dir = CATEGORY_DIRS.get(match_category, "")
                            mirror_subdir = rel_dir.split("/", 1)[1] if "/" in rel_dir else rel_dir
                            mirror_path = os.path.join(TRADINGBOT_MIRROR_DIR, mirror_subdir, match_filename)
                            _atomic_write_json(mirror_path, existing_data)
                        except OSError as e:
                            logger.warning(f"Trading mirror upsert failed: {e}")

                    logger.info(f"Knowledge upsert: updated '{title}' [{match_category}]")
                    return f"Updated existing entry: '{title}' [{match_category}] (ID: {match_entry.get('id', '?')})"
            except Exception as e:
                logger.warning(f"Knowledge upsert search failed, falling through to create: {e}")
                # S118 W-061: Re-check dedup before creating to avoid duplicate on exception fallthrough
                try:
                    _fb_index = _load_index()
                    _fb_entries = _fb_index.get("entries", [])
                    for _fb_e in _fb_entries:
                        _fb_created = _fb_e.get("created", "")
                        if _fb_e.get("title", "").lower().strip() == title_lower and _fb_created:
                            try:
                                # S121 W-11: Normalize naive datetimes to avoid TypeError
                                _fb_dt = datetime.fromisoformat(_fb_created)
                                if _fb_dt.tzinfo is None:
                                    _fb_dt = _fb_dt.replace(tzinfo=_NY)
                                if (datetime.now(_NY) - _fb_dt).days < 3:
                                    logger.info(f"Knowledge dedup: skipping create for '{title}' (exists within 3 days)")
                                    return f"Entry with similar title already exists (within 3 days): '{title}'"
                            except (ValueError, TypeError):
                                pass
                except Exception as e2:
                    logger.debug(f"Fire-and-forget upsert fallthrough dedup check failed: {e2}")
            finally:
                if lock_fd:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        lock_fd.close()
                    except Exception as e:
                        logger.debug(f"Fire-and-forget upsert lock release failed: {e}")

        # S118: Wrap entire create path in try/finally to release TOCTOU lock
        try:
            _ensure_dirs()

            # Generate ID and filename
            now = datetime.now(_NY)
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            safe_title = _sanitize_filename(title)
            entry_id = f"{timestamp_str}_{safe_title}"
            filename = f"{entry_id}.json"

            # Build entry
            entry = {
                "id": entry_id,
                "title": title,
                "category": category,
                "content": content,
                "tags": tags,
                "created": now.isoformat(),
                "source": source,
            }

            # Write entry to category directory
            rel_dir = CATEGORY_DIRS[category]
            entry_path = os.path.join(KNOWLEDGE_BASE_DIR, rel_dir, filename)
            _atomic_write_json(entry_path, entry)

            # Dual-write for trading categories (retired Hydrinia bridge, gated off by default)
            mirror_path = None
            if category in TRADING_CATEGORIES and _bridge_enabled():
                mirror_subdir = rel_dir.split("/", 1)[1] if "/" in rel_dir else rel_dir
                mirror_path = os.path.join(TRADINGBOT_MIRROR_DIR, mirror_subdir, filename)
                try:
                    _atomic_write_json(mirror_path, entry)
                except OSError as e:
                    logger.warning(f"Trading mirror write failed: {e}")
                    mirror_path = None

            # Urgent channel — mirror to urgent/ for time-sensitive entries (retired
            # Hydrinia bridge, gated off by default)
            urgent_path = None
            if _bridge_enabled():
                try:
                    tags_lower = {t.lower() for t in tags}
                    text_lower = (title + " " + content).lower()
                    is_urgent = (
                        tags_lower & {"urgent", "critical"}
                        or any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in URGENT_KEYWORDS)
                    )
                    if is_urgent:
                        os.makedirs(URGENT_MIRROR_DIR, exist_ok=True)
                        urgent_entry = {**entry, "priority": "URGENT"}
                        urgent_path = os.path.join(URGENT_MIRROR_DIR, filename)
                        _atomic_write_json(urgent_path, urgent_entry)
                        logger.info(f"URGENT knowledge mirrored: {entry_id}")
                except Exception as e:
                    logger.warning(f"Urgent mirror write failed: {e}")
                    urgent_path = None

            # S162 H-4: Keep TOCTOU dedup lock held through _append_or_update_index()
            # to prevent duplicate insertions. The lock FD is passed to the function
            # which skips acquiring its own lock (existing_lock_fd parameter).

            # Update master index (file-locked to prevent race conditions)
            index_entry = {
                "id": entry_id,
                "title": title,
                "category": category,
                "tags": tags,
                "created": now.isoformat(),
                "file_path": os.path.join(rel_dir, filename),
                "source": source,
            }
            # Include scores when provided (from learning engine) for index-level filtering
            # S156: Default to 5.0 when not provided — 34% of entries had no score
            # S215 R-30(a): mark the index entry when the 5.0 default was imputed (no real score).
            # Additive field only; no ranking or filter reads it.
            _qs_given = p.get("quality_score")
            index_entry["quality_score"] = _qs_given if _qs_given is not None else 5.0
            if _qs_given is None:
                index_entry["quality_imputed"] = True
            if p.get("actionability_score") is not None:
                index_entry["actionability_score"] = p["actionability_score"]

            # S122 1A: Commodity price divergence sanity check (warning only, never rejects)
            # S205 E1: when _create_lock_fd is held (non-upsert create path), this thread
            # already holds LOCK_EX on the index lock file — strictly stronger than the
            # LOCK_SH this read needs. Opening a SECOND FD on the same lock file and
            # requesting LOCK_SH self-deadlocks against our own LOCK_EX (anti-pattern #20)
            # on the event-loop thread (anti-pattern #19) — root cause of the daily morning
            # watchdog kill cascades (S204 probe, Class A). The previous comment here
            # ("outer TOCTOU lock released at line 551") was stale: S162 H-4 extended the
            # lock across this block. Only acquire LOCK_SH when no create lock is held
            # (upsert-fallthrough and dedup-error paths).
            _price_lock_fd = None
            try:
                _commodity_keywords = {"gold", "crude", "oil", "copper", "silver", "natural_gas", "wti", "brent",
                                       "crude_oil", "metals", "agriculture"}
                _tag_set = {t.lower() for t in tags} if tags else set()
                if category == "trading_market_analysis" and _tag_set & _commodity_keywords:
                    # Filter to commodity-plausible price range ($1-$15,000/unit)
                    _raw_prices = [float(m.replace(",", "")) for m in re.findall(r'\$(\d[\d,]*(?:\.\d{1,2})?)', content[:3000])]
                    _prices = [_pv for _pv in _raw_prices if 1.0 <= _pv <= 15000.0]
                    if _prices:
                        if _create_lock_fd is not None:
                            # Read under the already-held exclusive lock — no second flock.
                            _idx = _load_index()
                        else:
                            _price_lock_fd = open(_INDEX_LOCK_FILE, "a")
                            fcntl.flock(_price_lock_fd, fcntl.LOCK_SH)
                            _idx = _load_index()
                            fcntl.flock(_price_lock_fd, fcntl.LOCK_UN)
                            _price_lock_fd.close()
                            _price_lock_fd = None

                        _prior_prices = []
                        for _ie in reversed(_idx.get("entries", [])[-50:]):
                            _ie_tags = {t.lower() for t in (_ie.get("tags") or [])}
                            if _ie_tags & _tag_set & _commodity_keywords:
                                _ie_path = os.path.join(KNOWLEDGE_BASE_DIR, _ie.get("file_path", ""))
                                if os.path.isfile(_ie_path):
                                    try:
                                        with open(_ie_path, "r") as _pf:
                                            _pc = json.load(_pf).get("content", "")
                                        _pp = [float(m.replace(",", "")) for m in re.findall(r'\$(\d[\d,]*(?:\.\d{1,2})?)', _pc[:3000])]
                                        _pp = [_pv for _pv in _pp if 1.0 <= _pv <= 15000.0]
                                        if _pp:
                                            _prior_prices.extend(_pp)
                                            break  # Only need most recent prior
                                    except Exception as e:
                                        logger.debug(f"Prior price parse: {e}")
                        if _prior_prices:
                            _new_median = sorted(_prices)[len(_prices) // 2]
                            _prior_median = sorted(_prior_prices)[len(_prior_prices) // 2]
                            if _prior_median > 0:
                                _pct_diff = abs(_new_median - _prior_median) / _prior_median
                                if _pct_diff > 0.25:
                                    tags.append("price_divergence_warning")
                                    index_entry["tags"] = tags
                                    logger.warning(
                                        f"Price divergence: {list(_tag_set & _commodity_keywords)} "
                                        f"new=${_new_median:.0f} vs prior=${_prior_median:.0f} ({_pct_diff:.0%})"
                                    )
            except Exception as e:
                logger.debug(f"Price sanity check skipped: {e}")
            finally:
                if _price_lock_fd:
                    try:
                        fcntl.flock(_price_lock_fd, fcntl.LOCK_UN)
                        _price_lock_fd.close()
                    except Exception as e:
                        logger.debug(f"Lock cleanup: {e}")

            # Title-based dedup: if same title+category exists within last 24h, update it instead
            # S162 H-4: Pass held TOCTOU lock to prevent duplicate insertion window
            entry_count = _append_or_update_index(index_entry, entry_path, existing_lock_fd=_create_lock_fd, index=index)

            # S162 H-4: Release TOCTOU lock NOW — main index update is complete.
            # MUST release before SEAL impl calls which need their own lock on the same file.
            # flock blocks on same-file different-FD within same process (Linux 6.8+).
            if _create_lock_fd:
                try:
                    _create_lock_fd.close()
                except Exception as e:
                    logger.debug(f"FD close: {e}")
                _create_lock_fd = None

            # Contradiction check on new entry (fire-and-forget)
            try:
                from agent.integrations.contradiction_detector import check_new_entry_contradictions
                contradictions = check_new_entry_contradictions(index_entry)
                if contradictions:
                    logger.warning(f"Contradictions detected for '{title}': {len(contradictions)} entries")
            except Exception as e:
                logger.debug(f"Contradiction check failed: {e}")


            # P0-1: SEAL enrichment is async and must run on the event loop, not in
            # this executor thread. The locked region is already released here, so
            # hand the context back and let the enrichment await run after we return.
            _seal_ctx.update({
                "entry_id": entry_id,
                "entry_path": entry_path,
                "rel_dir": rel_dir,
                "now": now,
            })

            logger.info(f"Knowledge saved: {entry_id} [{category}] '{title}'")

            result = f"Saved: '{title}' [{category}] (ID: {entry_id})"
            if truncated:
                result += f"\nContent truncated to {MAX_CONTENT_LENGTH} chars."
            if mirror_path:
                result += f"\nMirrored to: {mirror_path}"
            if urgent_path:
                result += f"\nURGENT mirrored to: {urgent_path}"
            result += f"\nTags: {', '.join(tags)}" if tags else ""
            result += f"\nTotal entries: {entry_count}"
            return result
        finally:
            # S118: Release the TOCTOU dedup lock held since the dedup check
            if _create_lock_fd:
                try:
                    _create_lock_fd.close()
                except Exception as e:
                    logger.debug(f"FD close: {e}")

    result = await asyncio.get_running_loop().run_in_executor(None, _do_save)

    # P0-1: SEAL enrichment runs on the event loop AFTER the locked region is
    # released — the async enrichment call must never be awaited inside a held
    # flock (anti-patterns #19/#20). _seal_ctx is populated only when the create
    # path completed; every early-return (dedup skip / upsert) leaves it empty.
    if _seal_ctx:
        entry_id = _seal_ctx["entry_id"]
        entry_path = _seal_ctx["entry_path"]
        rel_dir = _seal_ctx["rel_dir"]
        now = _seal_ctx["now"]
        # --- SEAL ENRICHMENT HOOK (fire-and-forget, non-fatal) ---
        # Generates implications, atomic facts, and QA pairs for high-quality trading entries.
        # Controlled by MOLTBOT_SEAL_ENABLED and MOLTBOT_SEAL_LOG_ONLY env vars.
        try:
            from agent.tools.seal_knowledge_enhancer import (
                get_implication_engine, SEAL_ENABLED, SEAL_LOG_ONLY,
            )
            if SEAL_ENABLED():
                _seal_engine = get_implication_engine()
                _seal_quality = p.get("quality_score", 5) or 5
                if _seal_engine.should_enrich(category, quality_score=float(_seal_quality)):
                    # Extract ticker from tags — must be 1-5 uppercase alpha (optionally .X for BRK.B)
                    # S121 CR-2: Changed {2,5} to {1,5} to allow single-letter tickers (X, F, V)
                    import re as _seal_re
                    _seal_ticker = next(
                        (t for t in tags if _seal_re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", t)),
                        "",
                    )
                    _seal_enrichments = await _seal_engine.generate_enrichments(
                        category=category,
                        title=title,
                        content=content,
                        ticker=_seal_ticker,
                        quality_score=float(_seal_quality),
                    )
                    if _seal_enrichments and not SEAL_LOG_ONLY():
                        # Store implications as child knowledge entries
                        _seal_impl_written = 0  # S185 W4-2: count for parent tagging
                        for _i, _impl in enumerate(_seal_enrichments.get("implications", [])):
                            _seal_impl_id = f"seal_impl_{entry_id}_{_i:03d}"
                            _seal_impl_entry = {
                                "id": _seal_impl_id,
                                "title": f"[IMPL] {_impl[:80]}",
                                "category": category,
                                "tags": tags + ["seal_implication"],
                                "created": now.isoformat(),
                                "file_path": os.path.join(rel_dir, f"{_seal_impl_id}.json"),
                                "source": "seal_enrichment",
                                "quality_score": _seal_impl_quality(_impl),
                                "parent_id": entry_id,
                            }
                            # Write implication file
                            _seal_impl_path = os.path.join(
                                KNOWLEDGE_BASE_DIR, rel_dir, f"{_seal_impl_id}.json",
                            )
                            _atomic_write_json(_seal_impl_path, {
                                **_seal_impl_entry, "content": _impl,
                            })
                            # Add to knowledge index so recall/synthesis can find it
                            # S162 H-4: _create_lock_fd released after main index update (line ~678).
                            # Safe to acquire new lock here.
                            _append_or_update_index(_seal_impl_entry, _seal_impl_path)
                            # Mirror to Hydrinia bridge (all SEAL categories are trading_*;
                            # retired bridge, gated off by default)
                            if category in TRADING_CATEGORIES and _bridge_enabled():
                                _seal_mirror_sub = rel_dir.split("/", 1)[1] if "/" in rel_dir else rel_dir
                                _seal_mirror_path = os.path.join(
                                    TRADINGBOT_MIRROR_DIR, _seal_mirror_sub, f"{_seal_impl_id}.json",
                                )
                                try:
                                    _atomic_write_json(_seal_mirror_path, {
                                        **_seal_impl_entry, "content": _impl,
                                    })
                                except OSError:
                                    pass  # Non-fatal mirror failure
                            _seal_impl_written += 1
                        # S185 W4-2: Tag parent once we've spawned >=3 children.
                        # Annotation only; retrieval code will honor later.
                        if _seal_impl_written >= 3:
                            _tag_entry_as_parent_index(entry_id, _seal_impl_written)
                        # Store QA pairs for future verification
                        if _seal_enrichments.get("qa_pairs"):
                            _seal_engine.store_qa_pairs(
                                entry_id, title, _seal_enrichments["qa_pairs"],
                            )
                        # Append atomic facts to the entry's content on disk
                        _seal_facts = _seal_enrichments.get("atomic_facts", [])
                        if _seal_facts:
                            try:
                                with open(entry_path, "r") as _sf:
                                    _seal_entry_data = json.load(_sf)
                                _facts_block = "\n\n## Atomic Facts (SEAL)\n" + "\n".join(
                                    f"- {f}" for f in _seal_facts
                                )
                                _seal_entry_data["content"] = _seal_entry_data.get("content", "") + _facts_block
                                _atomic_write_json(entry_path, _seal_entry_data)
                            except (json.JSONDecodeError, OSError, KeyError):
                                pass  # Non-fatal: entry already saved without facts
        except Exception as e:
            logger.debug(f"SEAL enrichment hook failed (non-fatal): {e}")
        # --- END SEAL ENRICHMENT HOOK ---

    return result


async def handle_recall_knowledge(p: dict) -> str:
    """Search the knowledge base by category, keywords, or tags."""
    category_filter = p.get("category", "")
    keywords = p.get("keywords", "").lower().strip()
    tag_filter = p.get("tags", [])
    try:
        limit = max(1, min(int(p.get("limit", 10)), 50))
    except (ValueError, TypeError):
        limit = 10
    include_content = p.get("include_content", False)

    if not isinstance(tag_filter, list):
        tag_filter = []
    tag_filter_lower = {str(t).lower() for t in tag_filter if t}

    # Search both active and archive indexes for comprehensive recall
    entries = load_all_entries()

    if not entries:
        return "Knowledge base is empty. No entries saved yet."

    # Filter entries
    matches = []
    for entry in entries:
        # Category filter (exact match or prefix, e.g. "trading" matches "trading_strategies")
        if category_filter:
            if not entry.get("category", "").startswith(category_filter):
                continue

        # Keyword filter (search in title and tags) — word-boundary match
        if keywords:
            kw_pattern = r'\b' + re.escape(keywords) + r'\b'
            title_match = re.search(kw_pattern, entry.get("title", ""), re.IGNORECASE)
            tag_match = any(re.search(kw_pattern, str(t), re.IGNORECASE) for t in entry.get("tags", []))
            if not title_match and not tag_match:
                continue

        # Tag filter (any matching tag)
        if tag_filter_lower:
            entry_tags_lower = {str(t).lower() for t in entry.get("tags", [])}
            if not tag_filter_lower & entry_tags_lower:
                continue

        matches.append(entry)

    if not matches:
        parts = []
        if category_filter:
            parts.append(f"category='{category_filter}'")
        if keywords:
            parts.append(f"keywords='{keywords}'")
        if tag_filter:
            parts.append(f"tags={tag_filter}")
        return f"No entries found matching {', '.join(parts)}."

    # Sort by created (most recent first) and limit
    matches.sort(key=lambda e: e.get("created", ""), reverse=True)
    matches = matches[:limit]

    # Format results
    lines = [f"Found {len(matches)} matching entries:\n"]

    content_count = 0
    for entry in matches:
        tags_str = ", ".join(entry.get("tags", [])) if entry.get("tags") else "none"
        lines.append(
            f"  [{entry.get('category', '?')}] {entry.get('title', 'untitled')}\n"
            f"    ID: {entry.get('id', '?')} | Tags: {tags_str} | {entry.get('created', '?')}"
        )

        # Include content if requested (max 5)
        if include_content and content_count < 5:
            file_path = os.path.join(KNOWLEDGE_BASE_DIR, entry.get("file_path", ""))
            # Path validation: ensure file is within knowledge base dir
            real_path = os.path.realpath(file_path)
            if not real_path.startswith(os.path.realpath(KNOWLEDGE_BASE_DIR) + os.sep):
                lines.append("    Content: [access denied — path outside knowledge base]")
            elif os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as f:
                        full_entry = json.load(f)
                    content = full_entry.get("content", "")
                    if len(content) > 2000:
                        content = content[:2000] + "...(truncated)"
                    lines.append(f"    Content: {content}")
                    content_count += 1
                except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                    lines.append("    Content: [read error]")

    result = "\n".join(lines)
    # Sanitize ALL results (titles from ingested sources can contain injection attempts)
    try:
        from agent.security import sanitize_ingested_content
        result = sanitize_ingested_content(result, source="knowledge_recall")
    except Exception as e:
        logger.debug(f"Fire-and-forget recall sanitization failed: {e}")
    return result


async def handle_get_knowledge_summary(p: dict) -> str:
    """Return overview of all stored knowledge."""
    index = _load_index()
    entries = index.get("entries", [])

    if not entries:
        return "Knowledge base is empty. Use save_knowledge to start building your knowledge base."

    # Count by category
    category_counts = {}
    category_recent = {}
    for entry in entries:
        cat = entry.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if cat not in category_recent:
            category_recent[cat] = []
        category_recent[cat].append(entry)

    # Sort recent entries per category
    for cat in category_recent:
        category_recent[cat].sort(key=lambda e: e.get("created", ""), reverse=True)

    archive = _load_archive_index()
    archive_count = len(archive.get("entries", []))
    header = f"Knowledge Base Summary: {len(entries)} active entries"
    if archive_count > 0:
        header += f", {archive_count} archived"
    lines = [header + "\n"]

    # Group by domain
    domains = {
        "Trading": [c for c in sorted(category_counts) if c.startswith("trading_")],
        "Personal": [c for c in sorted(category_counts) if c.startswith("personal_")],
        "Email": [c for c in sorted(category_counts) if c.startswith("email_")],
        "Academic": [c for c in sorted(category_counts) if c.startswith("academic_")],
        "Career": [c for c in sorted(category_counts) if c.startswith("career_")],
    }

    for domain, cats in domains.items():
        if not cats:
            continue
        domain_total = sum(category_counts.get(c, 0) for c in cats)
        lines.append(f"{domain} ({domain_total} entries):")
        for cat in cats:
            count = category_counts[cat]
            short_name = cat.split("_", 1)[1] if "_" in cat else cat
            lines.append(f"  {short_name}: {count}")
            # Show up to 3 most recent
            for entry in category_recent[cat][:3]:
                lines.append(f"    - {entry.get('title', 'untitled')} ({entry.get('created', '?')[:10]})")
        lines.append("")

    return "\n".join(lines)


async def handle_update_priors(p: dict) -> str:
    """Update the current operational priors document."""
    content = p.get("content", "").strip()
    if not content:
        return "ERROR: content is required."
    if len(content) > 1500:
        truncated = content[:1500]
        # Find clean break point (newline > sentence > hard cut)
        nl_pos = truncated.rfind('\n')
        if nl_pos > 1000:
            content = truncated[:nl_pos]
        else:
            # S160-1: Find sentence boundary avoiding decimal points (e.g., "4.48%")
            sent_pos = -1
            for _si in range(len(truncated) - 1, 999, -1):
                if truncated[_si:_si + 2] == '. ':
                    if _si > 0 and truncated[_si - 1].isdigit():
                        continue  # Skip decimal: "4. " is likely "4.48" truncated
                    sent_pos = _si
                    break
            if sent_pos > 1000:
                content = truncated[:sent_pos + 1]
            else:
                content = truncated

    priors_path = CURRENT_PRIORS_PATH
    # Atomic write
    tmp_path = priors_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
    os.replace(tmp_path, priors_path)

    # Mirror to Hydrinia (retired bridge, gated off by default)
    if _bridge_enabled():
        try:
            mirror_path = os.path.join(TRADINGBOT_MIRROR_DIR, "current_priors.md")
            tmp_mirror = mirror_path + ".tmp"
            with open(tmp_mirror, "w") as f:
                f.write(content)
            os.replace(tmp_mirror, mirror_path)
        except OSError as e:
            logger.warning(f"Priors mirror write failed: {e}")

    logger.info(f"Operational priors updated ({len(content)} chars)")
    return f"Operational priors updated ({len(content)} chars). Will be injected into every system prompt."


# ============================================================
# Context Injection (called from system_prompt.py, NOT a tool)
# ============================================================

def _format_entries(entries: list, max_chars: int = 1800) -> list[str]:
    """Format index entries into display lines, respecting char budget."""
    lines = []
    char_count = 0
    for entry in entries:
        cat = entry.get("category", "?")
        title = entry.get("title", "?")
        date = entry.get("created", "?")[:10]
        tags = ", ".join(entry.get("tags", [])[:5])
        line = f"- [{cat}] {title} ({date})"
        if tags:
            line += f" #{tags}"
        if char_count + len(line) + 1 > max_chars:
            break
        lines.append(line)
        char_count += len(line) + 1
    return lines


def _score_entries(
    entries: list, keywords: list[str], *, min_score: float = 2.0
) -> list[tuple[float, dict]]:
    """Keyword-recall relevance scoring — the shared retrieve-stage seam.

    Scores each entry against ``keywords`` (title match ×2.0, tag match ×1.5,
    category match ×1.0, plus the S162 quality bonus) and returns ``(score, entry)``
    tuples sorted by descending score then ascending ``created`` timestamp.

    Entries scoring below ``min_score`` are dropped. The default (2.0) preserves
    the knowledge-base behaviour exactly; pass ``min_score=0.0`` to score the FULL
    candidate set — used by the personality retrieve stage, which needs every
    candidate's base_score and applies its own tie-break / budget-fill downstream.

    Extracted from _get_relevant_entries (behaviour-preserving) so this scoring
    formula lives in ONE place and is reused — never mirrored — by both the KB
    path and the personality retriever (DRY-at-root: change the formula here once).
    """
    if not keywords:
        return []
    kw_set = set(keywords)
    scored: list[tuple[float, dict]] = []
    for entry in entries:
        score = 0.0
        title_words = set(entry.get("title", "").lower().split())
        score += len(kw_set & title_words) * 2.0
        entry_tags = {str(t).lower() for t in entry.get("tags", [])}
        score += len(kw_set & entry_tags) * 1.5
        cat_words = set(entry.get("category", "").replace("_", " ").split())
        score += len(kw_set & cat_words) * 1.0
        # S162: Quality bonus — entries with quality_score > 6 get a ranking boost
        _quality = entry.get("quality_score")
        if isinstance(_quality, (int, float)) and _quality > 6:
            score += (_quality - 6) * 0.3  # +0.3 per point above 6 (max +1.2 at quality=10)
        if score >= min_score:
            scored.append((score, entry))
    scored.sort(key=lambda x: (-x[0], x[1].get("created", "")))
    return scored


def _get_relevant_entries(entries: list, keywords: list[str]) -> list[dict]:
    """Score and return entries relevant to keywords. Returns top 5 with score >= 2."""
    return [entry for _, entry in _score_entries(entries, keywords)[:5]]


def get_knowledge_context(topic: str = "") -> str:
    """
    Return a compact summary of knowledge for system prompt injection.
    If topic is provided, returns topic-relevant entries; falls back to recency.
    Max ~1800 chars. Returns empty string on any error (graceful degradation).
    """
    try:
        index = _load_index()
        entries = index.get("entries", [])
        if not entries:
            return ""

        # Try topic-relevant retrieval first
        if topic:
            keywords = _extract_topic_keywords(topic)
            if keywords:
                relevant = _get_relevant_entries(entries, keywords)
                if len(relevant) >= 2:
                    lines = _format_entries(relevant)
                    if lines:
                        return f"Relevant knowledge ({len(lines)} entries):\n" + "\n".join(lines)

        # Fall back to most recent 10 entries
        sorted_entries = sorted(entries, key=lambda e: e.get("created", ""), reverse=True)
        recent = sorted_entries[:10]
        lines = _format_entries(recent)

        if not lines:
            return ""

        return f"Recent knowledge ({len(lines)} entries):\n" + "\n".join(lines)

    except Exception as e:
        logger.debug(f"Knowledge context injection skipped: {e}")
        return ""


# ============================================================
# CC-CLIO-UW-5: UW Intelligence → Knowledge Entries (v2.0 — 8 generators)
# Supersedes: CC-MB-UW-5 Part A (3 generators → 8 generators)
# ============================================================

async def generate_uw_intelligence_entries() -> int:
    """Generate knowledge entries from UW signal intelligence.

    Called from scheduled_tasks on the hourly debrief cycle.
    Reads the UW signal bridge and creates knowledge entries for:
    - Congressional trades (late reports, notable trades)
    - FDA calendar events (PDUFA within 60 days)
    - Insider cluster detection (3+ insider buys within 30 days)
    - Unusual ticker analysis (overlap with Lazarus universe)

    Returns number of entries generated.
    """
    _uw_intel_enabled = os.environ.get("CLIO_UW_INTEL_ENABLED", os.environ.get("MOLTBOT_UW_INTEL_ENABLED", "1")) == "1"
    _uw_intel_log_only = os.environ.get("CLIO_UW_INTEL_LOG_ONLY", os.environ.get("MOLTBOT_UW_INTEL_LOG_ONLY", "1")) == "1"
    if not _uw_intel_enabled:
        return 0

    try:
        from agent.integrations.uw_signal_bridge import get_uw_bridge
        bridge = get_uw_bridge()
        if not bridge.is_fresh():
            logger.debug("MB-UW-5 intel: snapshot not fresh, skipping")
            return 0
    except Exception as e:
        logger.debug("MB-UW-5 intel: bridge unavailable: %s", e)
        return 0

    snapshot = bridge.get_snapshot()
    if not snapshot:
        return 0

    entries_generated = 0
    laz = snapshot.get("lazarus_signals", {})

    # --- Congressional late reports ---
    try:
        late_reports = laz.get("late_reports", [])
        for report in late_reports[:3]:
            politician = report.get("politician", "Unknown")
            ticker = report.get("ticker", "")
            days_late = report.get("filed_days_late", 0)
            if not ticker or days_late < 5:
                continue

            title = f"Congressional Late Filing: {politician} — {ticker} ({days_late}d late)"
            content = (
                f"Congressional insider trade alert: {politician} filed a trade in {ticker} "
                f"{days_late} days late. Late filings can indicate trades that politicians "
                f"preferred to keep quiet until after market-moving events. "
                f"Cross-reference with recent {ticker} catalysts and committee assignments."
            )

            if _uw_intel_log_only:
                logger.info("MB-UW-5 [LOG_ONLY] would create: %s", title)
            else:
                try:
                    if await _save_uw_intel_entry(title, content, "trading_market_analysis", ticker):
                        entries_generated += 1
                except Exception as e:
                    logger.debug("MB-UW-5 congressional entry failed: %s", e)
    except Exception as e:
        logger.debug("MB-UW-5 late reports scan failed: %s", e)

    # --- FDA calendar events ---
    try:
        fda_upcoming = laz.get("fda_upcoming", [])
        for event in fda_upcoming[:3]:
            ticker = event.get("ticker", "")
            drug = event.get("drug", "Unknown")
            target_date = event.get("target_date", "")
            action = event.get("action", "FDA")
            if not ticker or not target_date:
                continue

            # S146: Reject stale PDUFA events (target date > 6 months in the past)
            try:
                _td_clean = str(target_date).upper()
                _td_clean = _td_clean.replace("MID", "-06-15").replace("LATE", "-09-30").replace("EARLY", "-03-15")
                if len(_td_clean) == 4:  # Year only
                    _td_clean += "-06-15"
                elif len(_td_clean) == 7:  # Year-month
                    _td_clean += "-15"
                _pdufa_dt = datetime.fromisoformat(_td_clean)
                if _pdufa_dt.tzinfo is None:
                    _pdufa_dt = _pdufa_dt.replace(tzinfo=ZoneInfo("America/New_York"))
                if _pdufa_dt < datetime.now(ZoneInfo("America/New_York")) - timedelta(days=180):
                    logger.debug("UW PDUFA stale: %s target_date=%s — skipping", ticker, target_date)
                    continue
            except (ValueError, TypeError):
                pass  # Unparseable date — let it through (conservative)

            # S148-B W3-2: Pre-check — skip if PDUFA entry for this ticker exists within 3 days
            try:
                _pdufa_idx = load_index()
                _pdufa_cutoff = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=3)).isoformat()
                _pdufa_exists = any(
                    ticker.upper() in e.get("title", "").upper()
                    and "FDA" in e.get("title", "").upper()
                    and e.get("created", "") >= _pdufa_cutoff
                    for e in _pdufa_idx.get("entries", [])
                )
                if _pdufa_exists:
                    logger.debug("PDUFA pre-check: skipping %s (already exists within 3d)", ticker)
                    continue
            except Exception as e:
                logger.debug(f"PDUFA dedup pre-check: {e}")

            title = f"FDA {action}: {ticker} — {drug} (Target: {target_date})"
            content = (
                f"Upcoming FDA action for {ticker}: {drug} has a {action} target date of "
                f"{target_date}. Key considerations: (1) Historical approval rates for "
                f"this action type and therapeutic area affect base-rate probability. "
                f"(2) IV typically expands 2-3 weeks before PDUFA dates — monitor IV "
                f"percentile rank to assess options pricing. (3) CRL (Complete Response "
                f"Letter) risk factors include manufacturing issues, prior FDA feedback, "
                f"and advisory committee vote splits. (4) Sector contagion: approval/rejection "
                f"can move peers with similar mechanisms or targets."
            )

            if _uw_intel_log_only:
                logger.info("MB-UW-5 [LOG_ONLY] would create: %s", title)
            else:
                try:
                    if await _save_uw_intel_entry(title, content, "trading_market_analysis", ticker):
                        entries_generated += 1
                except Exception as e:
                    logger.debug("MB-UW-5 FDA entry failed: %s", e)
    except Exception as e:
        logger.debug("MB-UW-5 FDA scan failed: %s", e)

    # --- Insider cluster detection ---
    try:
        kill_scores = laz.get("top_kill_scores", [])
        for kill in kill_scores[:3]:
            ticker = kill.get("ticker", "")
            score = kill.get("score", 0)
            insider_cluster = kill.get("insider_cluster", False)
            si_float = kill.get("si_float", 0)
            if not ticker or not insider_cluster:
                continue

            title = f"Insider Cluster Alert: {ticker} (Kill Score {score}, SI {si_float}%)"
            content = (
                f"Lazarus insider cluster detected for {ticker}: kill score {score}/100 "
                f"with confirmed insider buying cluster. Short interest at {si_float}% of float. "
                f"Insider clusters combined with elevated short interest can signal "
                f"asymmetric upside potential if a catalyst triggers short covering."
            )

            if _uw_intel_log_only:
                logger.info("MB-UW-5 [LOG_ONLY] would create: %s", title)
            else:
                try:
                    if await _save_uw_intel_entry(title, content, "trading_trades", ticker):
                        entries_generated += 1
                except Exception as e:
                    logger.debug("MB-UW-5 insider cluster entry failed: %s", e)
    except Exception as e:
        logger.debug("MB-UW-5 insider cluster scan failed: %s", e)

    # --- Unusual tickers ---
    try:
        unusual = bridge.get_unusual_tickers()
        for ticker in unusual[:3]:
            if not ticker:
                continue

            title = f"Unusual Options Activity: {ticker}"
            content = (
                f"UW unusual activity screener flagged {ticker}. "
                f"Unusual options flow detected — cross-reference with existing theses "
                f"and open positions. Investigate whether the flow is directional "
                f"(new conviction) or hedging (portfolio protection)."
            )

            if _uw_intel_log_only:
                logger.info("MB-UW-5 [LOG_ONLY] would create: %s", title)
            else:
                try:
                    if await _save_uw_intel_entry(title, content, "trading_market_analysis", ticker):
                        entries_generated += 1
                except Exception as e:
                    logger.debug("MB-UW-5 unusual ticker entry failed: %s", e)
    except Exception as e:
        logger.debug("MB-UW-5 unusual tickers scan failed: %s", e)

    if entries_generated > 0:
        logger.info("CLIO-UW-5 intel: generated %d knowledge entries from original 4 generators",
                    entries_generated)
    return entries_generated


async def generate_uw_intelligence_entries_v2() -> int:
    """Generate knowledge entries from Trident v2.0 UW signals (5 new generators).

    Called from scheduled_tasks alongside generate_uw_intelligence_entries().
    Adds: prediction intelligence, technical summary, earnings straddle,
    short squeeze watchlist, ETF overlap report.

    Returns number of entries generated.
    """
    _uw_intel_enabled = os.environ.get("CLIO_UW_INTEL_ENABLED", os.environ.get("MOLTBOT_UW_INTEL_ENABLED", "1")) == "1"
    _uw_intel_log_only = os.environ.get("CLIO_UW_INTEL_LOG_ONLY", os.environ.get("MOLTBOT_UW_INTEL_LOG_ONLY", "1")) == "1"
    if not _uw_intel_enabled:
        return 0

    try:
        from agent.integrations.uw_signal_bridge import get_uw_bridge
        bridge = get_uw_bridge()
        if not bridge.is_fresh():
            logger.debug("CLIO-UW-5 v2: snapshot not fresh, skipping")
            return 0
    except Exception as e:
        logger.debug("CLIO-UW-5 v2: bridge unavailable: %s", e)
        return 0

    entries_generated = 0

    # --- Generator 4: Prediction intelligence ---
    try:
        predictions = bridge.get_predictions()
        unusual_finance = predictions.get("unusual_finance", [])
        smart_money = predictions.get("smart_money_consensus", "NEUTRAL")
        whale_conc = predictions.get("whale_concentration", {})

        if unusual_finance or smart_money != "NEUTRAL":
            top_unusual = unusual_finance[:3]
            top_tickers = [u.get("ticker", "?") for u in top_unusual if u.get("ticker")]

            whale_note = ""
            if isinstance(whale_conc, dict) and whale_conc.get("top_ticker"):
                whale_note = (f" Whale concentration: {whale_conc['top_ticker']} "
                              f"({whale_conc.get('pct_of_market', '?')}% of market).")

            title = f"Prediction Market Intelligence: Smart Money={smart_money}"
            if top_tickers:
                title += f" | Unusual: {', '.join(top_tickers[:2])}"
            content = (
                f"Prediction market signals update: Smart money consensus is {smart_money}. "
            )
            if top_unusual:
                content += f"Unusual finance activity flagged on: "
                content += ", ".join(
                    f"{u.get('ticker','?')} ({u.get('direction','?')}, "
                    f"activity score {u.get('activity_score','?')})"
                    for u in top_unusual
                ) + ". "
            content += (
                f"Prediction markets reflect aggregated positioning from retail and institutional "
                f"participants and can lead price action by 1-3 days.{whale_note}"
            )

            if _uw_intel_log_only:
                logger.info("CLIO-UW-5 [LOG_ONLY] prediction intel: sm=%s, %d unusual",
                            smart_money, len(unusual_finance))
            else:
                ticker_key = top_tickers[0] if top_tickers else ""
                try:
                    if await _save_uw_intel_entry(title, content, "trading_strategies", ticker_key):
                        entries_generated += 1
                except Exception as e:
                    logger.debug("CLIO-UW-5 prediction entry failed: %s", e)
    except Exception as e:
        logger.debug("CLIO-UW-5 prediction intel scan failed: %s", e)

    # --- Generator 5: Technical summary for held positions ---
    try:
        technicals = bridge.get_technicals()
        if technicals:
            signal_lines = []
            for ticker, td in list(technicals.items())[:5]:
                if not isinstance(td, dict):
                    continue
                signals = td.get("active_signals", [])
                rsi = td.get("rsi_14")
                macd_cross = "BULLISH_CROSS" in signals or "BEARISH_CROSS" in signals
                trend = "TREND_STRONG" in signals
                oversold = "OVERSOLD" in signals
                overbought = "OVERBOUGHT" in signals
                if signals:
                    sig_str = ", ".join(signals[:3])
                    rsi_str = f" (RSI={rsi})" if rsi is not None else ""
                    signal_lines.append(f"{ticker}: {sig_str}{rsi_str}")

            if signal_lines:
                title = f"Technical Indicator Summary: {len(signal_lines)} tickers with active signals"
                content = (
                    f"Technical analysis signals from Hydrinia's TA module:\n"
                    + "\n".join(f"- {s}" for s in signal_lines)
                    + "\n\nThese signals are derived from RSI-14, MACD, ADX, OBV, and Bollinger Bands. "
                    "OVERSOLD + BULLISH_CROSS = potential bounce setup. "
                    "TREND_STRONG = momentum continuation bias. "
                    "OVERBOUGHT + BEARISH_CROSS = potential reversal risk."
                )

                if _uw_intel_log_only:
                    logger.info("CLIO-UW-5 [LOG_ONLY] technical summary: %d tickers",
                                len(signal_lines))
                else:
                    try:
                        if await _save_uw_intel_entry(title, content, "trading_strategies", ""):
                            entries_generated += 1
                    except Exception as e:
                        logger.debug("CLIO-UW-5 technical summary entry failed: %s", e)
    except Exception as e:
        logger.debug("CLIO-UW-5 technical summary scan failed: %s", e)

    # --- Generator 6: Earnings straddle analysis ---
    try:
        earnings = bridge.get_earnings()
        upcoming = earnings.get("upcoming", [])
        straddle_accuracy = earnings.get("straddle_accuracy", {})

        if upcoming:
            entries_text = []
            for e in upcoming[:5]:
                e_ticker = e.get("ticker", "?")
                e_date = e.get("date", "?")
                e_move = e.get("expected_move_pct", "?")
                e_acc = straddle_accuracy.get(e_ticker, {})
                win_rate = e_acc.get("win_rate_pct", "?") if isinstance(e_acc, dict) else "?"
                entries_text.append(
                    f"{e_ticker} ({e_date}): expected ±{e_move}%, straddle win rate {win_rate}%"
                )

            title = f"Earnings Straddle Analysis: {len(upcoming)} upcoming events"
            content = (
                f"Upcoming earnings with options-implied expected moves:\n"
                + "\n".join(f"- {s}" for s in entries_text)
                + "\n\nStraddle win rate reflects how often buying the ATM straddle before earnings "
                "was profitable in prior quarters. High IV rank + low straddle win rate = "
                "IV crush risk. Low IV rank + high straddle win rate = potential vol buying opportunity."
            )

            if _uw_intel_log_only:
                logger.info("CLIO-UW-5 [LOG_ONLY] earnings straddle: %d upcoming", len(upcoming))
            else:
                try:
                    if await _save_uw_intel_entry(title, content, "trading_market_analysis", ""):
                        entries_generated += 1
                except Exception as e:
                    logger.debug("CLIO-UW-5 earnings straddle entry failed: %s", e)
    except Exception as e:
        logger.debug("CLIO-UW-5 earnings straddle scan failed: %s", e)

    # --- Generator 7: Short squeeze watchlist ---
    try:
        laz_enh = bridge.get_lazarus_enhanced()
        screener_hits = laz_enh.get("short_screener_hits", [])
        capitulation = laz_enh.get("short_volume_capitulation", [])
        ftd_spikes = laz_enh.get("ftd_spikes", [])

        squeeze_candidates = []
        for hit in screener_hits[:5]:
            t = hit.get("ticker", "")
            si = hit.get("si_float", 0)
            if t:
                squeeze_candidates.append(f"{t} (SI={si}%)")

        cap_tickers = [c.get("ticker", "") for c in capitulation[:3] if c.get("ticker")]
        ftd_tickers = [f.get("ticker", "") for f in ftd_spikes[:3] if f.get("ticker")]

        if squeeze_candidates or cap_tickers:
            title = f"Short Squeeze Watchlist: {len(squeeze_candidates)} screener hits"
            content = (
                f"Lazarus enhanced short squeeze analysis:\n"
            )
            if squeeze_candidates:
                content += f"Short screener hits: {', '.join(squeeze_candidates[:5])}. "
            if cap_tickers:
                content += f"Short volume capitulation detected: {', '.join(cap_tickers)}. "
            if ftd_tickers:
                content += f"FTD spikes (failure-to-deliver): {', '.join(ftd_tickers)}. "
            content += (
                "\nShort squeeze conditions: high SI float + FTD spikes + volume capitulation "
                "= elevated squeeze probability. Combine with catalyst analysis and insider cluster "
                "signals for higher conviction setups."
            )

            if _uw_intel_log_only:
                logger.info("CLIO-UW-5 [LOG_ONLY] short squeeze: %d candidates", len(squeeze_candidates))
            else:
                ticker_key = screener_hits[0].get("ticker", "") if screener_hits else ""
                try:
                    if await _save_uw_intel_entry(title, content, "trading_market_analysis", ticker_key):
                        entries_generated += 1
                except Exception as e:
                    logger.debug("CLIO-UW-5 short squeeze entry failed: %s", e)
    except Exception as e:
        logger.debug("CLIO-UW-5 short squeeze scan failed: %s", e)

    # --- Generator 8: ETF overlap report ---
    try:
        sector_enh = bridge.get_sector_enhanced()
        etf_overlap = sector_enh.get("etf_overlap", {})
        etf_tide = sector_enh.get("etf_tide", {})

        if etf_overlap or (isinstance(etf_tide, dict) and etf_tide.get("direction")):
            overlap_lines = []
            if isinstance(etf_overlap, dict):
                for ticker, overlaps in list(etf_overlap.items())[:5]:
                    if isinstance(overlaps, list) and overlaps:
                        etfs = ", ".join(str(o) for o in overlaps[:3])
                        overlap_lines.append(f"{ticker}: present in {etfs}")

            tide_dir = etf_tide.get("direction", "?") if isinstance(etf_tide, dict) else "?"
            tide_div = etf_tide.get("divergence_from_equity", False) if isinstance(etf_tide, dict) else False

            title = f"ETF Overlap & Tide Report: Tide={tide_dir}"
            if tide_div:
                title += " (DIVERGENCE DETECTED)"
            content = (
                f"Sector ETF intelligence update:\n"
                f"ETF flow tide direction: {tide_dir}. "
            )
            if tide_div:
                content += "ALERT: ETF flow is diverging from equity market direction — potential regime shift signal. "
            if overlap_lines:
                content += (
                    f"\nTicker-to-ETF membership (concentration risk):\n"
                    + "\n".join(f"- {s}" for s in overlap_lines)
                )
            content += (
                "\n\nETF overlap = concentration risk: a single stock move affects multiple ETFs "
                "simultaneously, amplifying flows. ETF tide divergence historically precedes "
                "equity trend reversals by 1-5 days."
            )

            if _uw_intel_log_only:
                logger.info("CLIO-UW-5 [LOG_ONLY] ETF overlap: %d tickers, tide=%s div=%s",
                            len(overlap_lines), tide_dir, tide_div)
            else:
                try:
                    if await _save_uw_intel_entry(title, content, "trading_market_analysis", ""):
                        entries_generated += 1
                except Exception as e:
                    logger.debug("CLIO-UW-5 ETF overlap entry failed: %s", e)
    except Exception as e:
        logger.debug("CLIO-UW-5 ETF overlap scan failed: %s", e)

    if entries_generated > 0:
        logger.info("CLIO-UW-5 v2 intel: generated %d entries from 5 Trident generators",
                    entries_generated)
    return entries_generated


async def _save_uw_intel_entry(title: str, content: str, category: str, ticker: str) -> bool:
    """Save a UW-generated intelligence entry to the knowledge base.

    Uses the existing handle_save_knowledge pathway for consistency
    (dedup, dual-write, urgent channel, etc.).
    Returns True if saved successfully.
    """
    entry = {
        "title": title,
        "content": content,
        "category": category,
        "tags": ["uw_intelligence", ticker.upper()] if ticker else ["uw_intelligence"],
        "source": "uw_signal_bridge",
        "quality_score": 7,  # S148 W-23: UW institutional flow data is high-quality structured intel
    }
    result = await handle_save_knowledge(entry)
    if isinstance(result, str) and (result.startswith("ERROR") or result.startswith("Skipped")):
        logger.debug("MB-UW-5 save not completed: %s", result)
        return False
    return True
