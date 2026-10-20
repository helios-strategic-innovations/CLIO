#!/usr/bin/env python3
"""One-time fix: normalize confidence values in existing trade idea JSON files.

Trade ideas written before _normalize_confidence() was added may have
confidence on an integer 1-10 scale (e.g., 5 or 7) instead of 0.0-1.0.
This script finds and fixes those files using atomic writes.

Usage: python3 scripts/fix_trade_idea_confidence.py [--dry-run]
"""

import json
import os
import sys

TRADE_IDEAS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge", "trade_ideas")
MIRROR_DIR = "/home/agent/tradingbot/moltbot_knowledge/trade_ideas"


def normalize_confidence(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v > 1.0:
        return round(min(v / 10.0, 1.0), 2)
    return round(max(v, 0.0), 2)


def fix_file(filepath: str, dry_run: bool) -> bool:
    """Fix confidence in a single trade idea file. Returns True if modified."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  SKIP (read error): {filepath} — {e}")
        return False

    confidence = data.get("confidence")
    if confidence is None:
        return False

    try:
        conf_float = float(confidence)
    except (TypeError, ValueError):
        return False

    if conf_float <= 1.0:
        return False

    new_conf = normalize_confidence(conf_float)
    print(f"  FIX: {os.path.basename(filepath)} — confidence {confidence} -> {new_conf}")

    if dry_run:
        return True

    data["confidence"] = new_conf
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, filepath)
    return True


def main():
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("DRY RUN — no files will be modified\n")

    fixed = 0
    scanned = 0

    for search_dir, label in [(TRADE_IDEAS_DIR, "data/knowledge/trade_ideas"), (MIRROR_DIR, "moltbot_knowledge/trade_ideas")]:
        abs_dir = os.path.abspath(search_dir)
        if not os.path.isdir(abs_dir):
            print(f"Directory not found: {abs_dir}")
            continue

        print(f"Scanning {label} ({abs_dir}):")
        for fname in sorted(os.listdir(abs_dir)):
            if not fname.endswith(".json"):
                continue
            scanned += 1
            fpath = os.path.join(abs_dir, fname)
            if fix_file(fpath, dry_run):
                fixed += 1

    print(f"\nDone. Scanned {scanned} files, {'would fix' if dry_run else 'fixed'} {fixed}.")


if __name__ == "__main__":
    main()
