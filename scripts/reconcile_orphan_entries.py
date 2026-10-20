#!/usr/bin/env python3
"""S205 E6 — reconcile orphaned knowledge entry files into the live index.

Background: the knowledge_tools.py:781 second-FD flock self-deadlock (S204
probe, Class A) wrote entry files (line 715) but never reached the index
update (line 827), every weekday morning since 2026-04-06. This tool finds
those orphans and indexes them.

Discovery is STATUS-driven (anti-pattern #36): a file is an orphan iff its id
appears in NEITHER the live index NOR the archive index — no timestamp
windows, so late-found orphans stay discoverable until actually reconciled.
Re-running is a no-op once reconciled (idempotent).

Same-title grouping: the deadlock fired 2-3× per morning (budget-capped
retries), producing same-day duplicates with identical titles. Production
semantics would have kept exactly one (the first success would have set
_already_analyzed and suppressed the retries), so we index only ONE attempt
per title and skip the rest with reason superseded_same_day — files stay on
disk untouched.

IDEMPOTENCY GUARD (load-bearing): a candidate whose title+category already
exists in the live or archive index is skipped (title_already_indexed),
NEVER fed to _append_or_update_index. Without this, a re-run feeds the
still-orphaned sibling attempts into _append_or_update_index's same-title-
within-24h merge path, which REPLACES the indexed entry and DELETES the
previously-indexed file (its os.remove cleanup). The first version of this
script lacked the guard and its idempotency re-run deleted 39 later-attempt
duplicate files on 2026-06-12 (all recoverable from
~/backups/2026-06-12/16/openclaw_data/; system left consistent — one
entry/day, no dangling index references).

Usage (dry-run by default; nothing is written without --apply):
  ./venv/bin/python3 scripts/reconcile_orphan_entries.py
  ./venv/bin/python3 scripts/reconcile_orphan_entries.py --apply
  ./venv/bin/python3 scripts/reconcile_orphan_entries.py \
      --category trading_market_analysis --pattern commodity_natural_gas --apply

Note: reconciled entries older than the knowledge-expiry horizon (21d hard /
10d quality-gated) will be moved to the archive index by the normal archival
cycle — that is the intended lifecycle, not a failure of this tool.
"""

import argparse
import glob
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from agent.tools import knowledge_tools as kt  # noqa: E402

REQUIRED_FIELDS = ("id", "title", "category", "created")


def _load_ids(path: str) -> set:
    try:
        with open(path) as f:
            return {e.get("id") for e in json.load(f).get("entries", [])}
    except (OSError, json.JSONDecodeError):
        return set()


def _load_titles(path: str, category: str) -> set:
    """Lower/stripped titles already indexed for `category` (live or archive)."""
    try:
        with open(path) as f:
            return {
                e.get("title", "").lower().strip()
                for e in json.load(f).get("entries", [])
                if e.get("category") == category
            }
    except (OSError, json.JSONDecodeError):
        return set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--category", default="trading_market_analysis",
                    help="knowledge category to scan (default: trading_market_analysis)")
    ap.add_argument("--pattern", default="commodity_natural_gas",
                    help="filename substring filter (default: commodity_natural_gas)")
    ap.add_argument("--apply", action="store_true",
                    help="actually write to the index (default: dry-run)")
    args = ap.parse_args()

    rel_dir = kt.CATEGORY_DIRS.get(args.category)
    if not rel_dir:
        print(f"ERROR: unknown category {args.category!r}")
        return 2

    cat_dir = os.path.join(kt.KNOWLEDGE_BASE_DIR, rel_dir)
    live_ids = _load_ids(kt.KNOWLEDGE_INDEX_FILE)
    arc_ids = _load_ids(kt.KNOWLEDGE_ARCHIVE_INDEX_FILE)
    indexed_titles = _load_titles(kt.KNOWLEDGE_INDEX_FILE, args.category)
    indexed_titles |= _load_titles(kt.KNOWLEDGE_ARCHIVE_INDEX_FILE, args.category)

    found, skipped, candidates = [], [], []
    for fp in sorted(glob.glob(os.path.join(cat_dir, "*.json"))):
        name = os.path.basename(fp)
        if args.pattern not in name:
            continue
        found.append(fp)
        fid = name[:-5]
        if fid in live_ids:
            skipped.append((fid, "already_in_live_index"))
            continue
        if fid in arc_ids:
            skipped.append((fid, "already_in_archive_index"))
            continue
        try:
            with open(fp) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            skipped.append((fid, f"unreadable: {e}"))
            continue
        missing = [k for k in REQUIRED_FIELDS if not data.get(k)]
        if missing:
            skipped.append((fid, f"missing_fields: {','.join(missing)}"))
            continue
        if data.get("category") != args.category:
            skipped.append((fid, f"category_mismatch: {data.get('category')!r}"))
            continue
        if data.get("id") != fid:
            skipped.append((fid, f"id_filename_mismatch: {data.get('id')!r}"))
            continue
        if data["title"].lower().strip() in indexed_titles:
            # Idempotency guard — this title is already represented in the
            # index; feeding a sibling attempt to _append_or_update_index
            # would take its merge-and-delete path. Skip forever.
            skipped.append((fid, "title_already_indexed"))
            continue
        candidates.append((fp, data))

    # Earliest attempt per title wins — production's first success would have
    # set _already_analyzed and suppressed the later retries entirely.
    by_title: dict = {}
    for fp, data in candidates:
        key = data["title"].lower().strip()
        prev = by_title.get(key)
        if prev is None or data.get("created", "") < prev[1].get("created", ""):
            if prev is not None:
                skipped.append((prev[1]["id"], "superseded_same_day"))
            by_title[key] = (fp, data)
        else:
            skipped.append((data["id"], "superseded_same_day"))

    to_index = sorted(by_title.values(), key=lambda x: x[1].get("created", ""))

    indexed = []
    for fp, data in to_index:
        index_entry = {
            "id": data["id"],
            "title": data["title"],
            "category": data["category"],
            "tags": data.get("tags", []),
            "created": data["created"],
            "file_path": os.path.join(rel_dir, os.path.basename(fp)),
            "source": data.get("source", "reconciled"),
            "quality_score": data.get("quality_score") if data.get("quality_score") is not None else 5.0,
            "reconciled": "s205_e6",
        }
        if args.apply:
            kt._append_or_update_index(index_entry, fp)
        indexed.append(data["id"])

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\n[{mode}] category={args.category} pattern={args.pattern!r}")
    print(f"found:   {len(found)} matching files")
    print(f"indexed: {len(indexed)}" + ("" if args.apply else " (would be)"))
    reasons: dict = {}
    for _, r in skipped:
        key = r.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    print(f"skipped: {len(skipped)}")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {n}")
    if indexed:
        print(f"first indexed: {indexed[0]}")
        print(f"last indexed:  {indexed[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
