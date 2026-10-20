#!/usr/bin/env python3
"""One-time knowledge base cleanup script.

Identifies and archives:
1. Low-quality entries (quality + actionability < 10, no tickers in title)
2. Duplicate titles (3+ with same title — keep newest, archive rest)
3. Book corpus duplicates (same book within 7 days — keep newest per book)
4. Stale trade ideas that contradict active directives

Idempotent — safe to run multiple times.
"""

import fcntl
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_NY = ZoneInfo("America/New_York")

# Paths
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge")
KNOWLEDGE_INDEX_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "knowledge_index.json")
INDEX_LOCK_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "knowledge_index.json.lock")
ARCHIVE_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "archive")
TRADE_IDEAS_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "trade_ideas")

# Known prohibited instruments (from Dominic's directive)
PROHIBITED_TICKERS = {"GLD", "SLV", "AGQ", "IAU"}  # precious metals

# Common ticker-like patterns to detect tickers in titles
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z])?\b")

# Words that look like tickers but aren't
_TICKER_EXCLUSIONS = {
    "AND", "THE", "FOR", "NOT", "BUT", "ALL", "HAS", "HAD", "ARE", "WAS",
    "NEW", "OLD", "BIG", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
    "SEP", "OCT", "NOV", "DEC", "JAN", "MON", "TUE", "WED", "THU", "FRI",
    "SAT", "SUN", "ETF", "IPO", "CEO", "CFO", "SEC", "FED", "GDP", "CPI",
    "PPI", "USD", "EUR", "GBP", "JPY", "VIX", "P&L", "OPEC", "NATO",
    "RISK", "DEBT", "BOND", "RATE", "GOLD", "OIL", "GAS", "BOOK", "TEST",
    "SCRM", "DEFI", "SPAC", "ESG",
}


def _has_ticker_in_title(title: str) -> bool:
    """Check if title contains what looks like a stock ticker."""
    candidates = _TICKER_RE.findall(title)
    return any(c not in _TICKER_EXCLUSIONS for c in candidates)


def _extract_book_name(title: str) -> str | None:
    """Extract book name from 'Book: <name> — <date>' pattern."""
    if not title.lower().startswith("book:"):
        return None
    # "Book: Big Debt Crises — Mar 03" → "big debt crises"
    rest = title[5:].strip()
    # Split on em-dash or regular dash with date
    for sep in [" — ", " - "]:
        if sep in rest:
            return rest.split(sep)[0].strip().lower()
    return rest.strip().lower()


def main():
    # Ensure archive dir exists
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # Acquire lock
    lock_fd = open(INDEX_LOCK_FILE, "a")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        with open(KNOWLEDGE_INDEX_FILE, "r") as f:
            index = json.load(f)

        entries = index.get("entries", [])
        print(f"Total entries before cleanup: {len(entries)}")

        to_archive = set()  # indices to archive
        reasons = {}  # index -> reason

        # === 1. Low-quality entries with no tickers ===
        for i, entry in enumerate(entries):
            quality = entry.get("quality_score", 0) or 0
            actionability = entry.get("actionability_score", 0) or 0
            title = entry.get("title", "")

            # Only target entries that have scores (auto_learning entries)
            # Manual entries won't have scores and should be kept
            if quality == 0 and actionability == 0:
                # Check if it's from auto_learning by source or tags
                tags = entry.get("tags", [])
                source = entry.get("source", "")
                if source != "autonomous_learning" and "auto_learning" not in tags:
                    continue

            if (quality + actionability < 10) and not _has_ticker_in_title(title):
                to_archive.add(i)
                reasons[i] = f"low_quality (q={quality} a={actionability} sum={quality + actionability})"

        low_quality_count = len(to_archive)
        print(f"  Low-quality entries (no tickers, q+a<10): {low_quality_count}")

        # === 2. Duplicate titles (3+ copies — keep newest, archive rest) ===
        title_groups = defaultdict(list)
        for i, entry in enumerate(entries):
            title_lower = entry.get("title", "").lower().strip()
            if title_lower:
                title_groups[title_lower].append(i)

        dup_count = 0
        for title_lower, indices in title_groups.items():
            if len(indices) < 3:
                continue
            # Sort by created date descending (newest first)
            indices.sort(
                key=lambda i: entries[i].get("created", ""),
                reverse=True,
            )
            # Keep the newest, archive the rest
            for idx in indices[1:]:
                if idx not in to_archive:
                    to_archive.add(idx)
                    reasons[idx] = f"duplicate_title ({len(indices)}x: '{title_lower[:50]}')"
                    dup_count += 1

        print(f"  Duplicate title entries (3+, keeping newest): {dup_count}")

        # === 3. Book corpus duplicates (same book within 7 days — keep newest) ===
        book_groups = defaultdict(list)
        for i, entry in enumerate(entries):
            book_name = _extract_book_name(entry.get("title", ""))
            if book_name:
                book_groups[book_name].append(i)

        book_dup_count = 0
        cutoff_7d = (datetime.now(_NY) - timedelta(days=7)).isoformat()
        for book_name, indices in book_groups.items():
            # Filter to entries within last 7 days
            recent = [
                i for i in indices
                if entries[i].get("created", "") >= cutoff_7d
            ]
            if len(recent) < 2:
                continue
            # Sort by created date descending (newest first)
            recent.sort(
                key=lambda i: entries[i].get("created", ""),
                reverse=True,
            )
            # Keep the newest, archive the rest
            for idx in recent[1:]:
                if idx not in to_archive:
                    to_archive.add(idx)
                    reasons[idx] = f"book_duplicate ('{book_name[:40]}', {len(recent)} in 7d)"
                    book_dup_count += 1

        print(f"  Book corpus duplicates (7-day window): {book_dup_count}")

        # === Execute archival ===
        if not to_archive:
            print("\nNo entries to archive. Knowledge base is clean.")
            return

        # Sort indices descending for safe removal
        archive_indices = sorted(to_archive, reverse=True)
        archived_entries = []
        moved_files = 0
        missing_files = 0

        for idx in archive_indices:
            entry = entries[idx]
            archived_entries.append({
                **entry,
                "archived_at": datetime.now(_NY).isoformat(),
                "archive_reason": reasons.get(idx, "unknown"),
            })

            # Move the JSON file to archive
            file_path = entry.get("file_path", "")
            src = os.path.join(KNOWLEDGE_BASE_DIR, file_path)
            if os.path.isfile(src):
                # Preserve subdirectory structure in archive
                dest_subdir = os.path.dirname(file_path)
                dest_dir = os.path.join(ARCHIVE_DIR, dest_subdir)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(ARCHIVE_DIR, file_path)
                try:
                    shutil.move(src, dest)
                    moved_files += 1
                except OSError as e:
                    print(f"  WARNING: Failed to move {src}: {e}")
            else:
                missing_files += 1

            # Remove from index (descending order is safe)
            entries.pop(idx)

        # Write updated index atomically
        index["entries"] = entries
        tmp_path = KNOWLEDGE_INDEX_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(index, f, indent=2, default=str)
        os.replace(tmp_path, KNOWLEDGE_INDEX_FILE)

        # Write archive manifest
        manifest = {
            "archived_at": datetime.now(_NY).isoformat(),
            "total_archived": len(archived_entries),
            "by_reason": {},
            "entries": archived_entries,
        }
        for entry in archived_entries:
            reason_type = entry["archive_reason"].split(" ")[0]
            manifest["by_reason"][reason_type] = manifest["by_reason"].get(reason_type, 0) + 1

        manifest_path = os.path.join(
            ARCHIVE_DIR,
            f"cleanup_manifest_{datetime.now(_NY).strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        print(f"\n=== CLEANUP SUMMARY ===")
        print(f"  Archived: {len(archived_entries)} entries")
        print(f"  Files moved: {moved_files}")
        print(f"  Files missing (already gone): {missing_files}")
        print(f"  Remaining active: {len(entries)}")
        print(f"  Manifest: {manifest_path}")
        print(f"  By reason:")
        for reason, count in manifest["by_reason"].items():
            print(f"    {reason}: {count}")

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    # === 4. Stale trade ideas contradicting directives ===
    stale_archived = 0
    trade_archive_dir = os.path.join(ARCHIVE_DIR, "trade_ideas")
    if os.path.isdir(TRADE_IDEAS_DIR):
        for fname in os.listdir(TRADE_IDEAS_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(TRADE_IDEAS_DIR, fname)
            try:
                with open(fpath, "r") as f:
                    idea = json.load(f)

                ticker = idea.get("ticker", "").upper()
                consumed = idea.get("consumed_by_hydrinia", False)

                # Skip consumed ideas
                if consumed:
                    continue

                # Archive if ticker is in prohibited list
                if ticker in PROHIBITED_TICKERS:
                    os.makedirs(trade_archive_dir, exist_ok=True)
                    dest = os.path.join(trade_archive_dir, fname)
                    shutil.move(fpath, dest)
                    stale_archived += 1
                    print(f"  Archived prohibited trade idea: {ticker} ({fname})")
            except (json.JSONDecodeError, OSError):
                continue

    if stale_archived:
        print(f"\n  Stale/prohibited trade ideas archived: {stale_archived}")


if __name__ == "__main__":
    main()
