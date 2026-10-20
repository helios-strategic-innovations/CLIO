#!/usr/bin/env python3
"""S215 R-30(b) — READ-ONLY knowledge-base orphan report.

Reports:
  1. Entry JSON files on disk under the knowledge root that are referenced by
     NEITHER the live index (knowledge_index.json) NOR the archive index
     (knowledge_index_archive.json), grouped by top-level directory.
  2. Index entries (live and archive) whose file_path does not exist on disk.

Strictly read-only: it opens files for reading and never creates, moves, edits or
deletes anything (no lock files, no temp files). Stdlib only.

Scope rules:
  * Only files inside a subdirectory of the root are entry candidates. Root-level
    JSON (index files, learning_state.json, state files) is not an entry.
  * trade_ideas/ and trade_ideas_v2/ are trade-idea artifacts, not knowledge entries,
    and are skipped unless --include-trade-ideas is given.
  * An unreadable or malformed index aborts with exit code 2 instead of reporting
    every file as an orphan.

Usage:
  python3 scripts/kb_orphan_report.py                 # default root: <repo>/data/knowledge
  python3 scripts/kb_orphan_report.py --root /some/tree --limit 50
"""

import argparse
import json
import os
import sys

LIVE_INDEX = "knowledge_index.json"
ARCHIVE_INDEX = "knowledge_index_archive.json"
TRADE_IDEA_DIRS = ("trade_ideas", "trade_ideas_v2")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(_REPO_ROOT, "data", "knowledge")


class IndexFileError(Exception):
    pass


def _load_entries(path: str) -> list:
    """Return the entries list of an index file. Missing file = no entries; a corrupt
    file raises IndexFileError so the caller can abort rather than report false orphans."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise IndexFileError(f"{path}: {e}") from e
    if isinstance(data, dict):
        entries = data.get("entries", [])
    elif isinstance(data, list):
        entries = data
    else:
        raise IndexFileError(f"{path}: unexpected top-level type {type(data).__name__}")
    if not isinstance(entries, list):
        raise IndexFileError(f"{path}: 'entries' is not a list")
    return [e for e in entries if isinstance(e, dict)]


def _rel_key(root: str, file_path: str) -> str:
    fp = file_path
    if os.path.isabs(fp):
        fp = os.path.relpath(fp, root)
    return os.path.normpath(fp)


def _walk_entry_files(root: str, include_trade_ideas: bool) -> list:
    """Relative paths of *.json files that sit inside a subdirectory of root."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            continue  # root-level JSON is state, not an entry
        top = rel_dir.split(os.sep)[0]
        if not include_trade_ideas and top in TRADE_IDEA_DIRS:
            dirnames[:] = []
            continue
        for name in filenames:
            if name.endswith(".json"):
                found.append(os.path.normpath(os.path.join(rel_dir, name)))
    return sorted(found)


def build_report(root: str, include_trade_ideas: bool = False) -> dict:
    live = _load_entries(os.path.join(root, LIVE_INDEX))
    archive = _load_entries(os.path.join(root, ARCHIVE_INDEX))

    indexed = set()
    for e in live + archive:
        fp = e.get("file_path")
        if isinstance(fp, str) and fp:
            indexed.add(_rel_key(root, fp))

    on_disk = _walk_entry_files(root, include_trade_ideas)
    orphans = [p for p in on_disk if p not in indexed]

    by_top = {}
    for p in orphans:
        top = p.split(os.sep)[0]
        by_top[top] = by_top.get(top, 0) + 1

    missing = {}
    for label, entries in (("live", live), ("archive", archive)):
        missing_paths = []
        for e in entries:
            fp = e.get("file_path")
            if not isinstance(fp, str) or not fp:
                continue
            abs_fp = fp if os.path.isabs(fp) else os.path.join(root, fp)
            if not os.path.exists(abs_fp):
                missing_paths.append(fp)
        missing[label] = sorted(missing_paths)

    return {
        "root": root,
        "live_entries": len(live),
        "archive_entries": len(archive),
        "files_on_disk": len(on_disk),
        "orphan_files": len(orphans),
        "orphans_by_top_dir": dict(sorted(by_top.items(), key=lambda kv: (-kv[1], kv[0]))),
        "orphan_samples": orphans,
        "missing_targets": missing,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=DEFAULT_ROOT, help="knowledge root (default: data/knowledge)")
    ap.add_argument("--limit", type=int, default=20, help="max orphan/missing paths to print per list")
    ap.add_argument("--include-trade-ideas", action="store_true",
                    help="also count trade_ideas/ and trade_ideas_v2/ files as entries")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"ERROR: root is not a directory: {args.root}", file=sys.stderr)
        return 2
    try:
        rep = build_report(args.root, include_trade_ideas=args.include_trade_ideas)
    except IndexFileError as e:
        print(f"ERROR: unreadable index: {e}", file=sys.stderr)
        return 2

    print(f"root: {rep['root']}")
    print(f"live index entries: {rep['live_entries']}  archive index entries: {rep['archive_entries']}")
    print(f"entry files on disk: {rep['files_on_disk']}  orphans (in neither index): {rep['orphan_files']}")
    print("orphans by top-level directory:")
    for top, n in rep["orphans_by_top_dir"].items():
        print(f"  {top}: {n}")
    if rep["orphan_samples"]:
        print(f"orphan samples (first {args.limit}):")
        for p in rep["orphan_samples"][: args.limit]:
            print(f"  {p}")
    for label in ("live", "archive"):
        mp = rep["missing_targets"][label]
        print(f"{label} index entries whose file is missing: {len(mp)}")
        for p in mp[: args.limit]:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
