#!/usr/bin/env python3
"""Collapse legacy SEAL trade-idea calibration rows in meta_memory.json.

Legacy rows have prediction_id "seal_ti_quality_<hash>" (one per 6 h write,
before the stable-id fix in memory_engine). All of them are collapsed into the
newest one, renamed to "seal_ti_quality". Every other row is untouched and keeps
its order.

Dry run by default: prints before/after counts and writes nothing.

--apply requires BOTH --i-understand-protected-state AND --backup-dir <existing
directory>. The backup is a hard link to the original inode (same filesystem,
no copy). The swap is tmp + os.replace under a non-blocking flock on
<file>.lock; the result is re-read and verified, and the original is restored
on any failure.

Stop moltbot before --apply: the running process holds its own in-memory copy
and rewrites the whole file on its next save, which would undo the change.
"""
import argparse
import collections
import copy
import fcntl
import json
import os
import sys
import time

LEGACY_PREFIX = "seal_ti_quality_"
TARGET_ID = "seal_ti_quality"
DEFAULT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "memory", "meta_memory.json",
)


class DedupeError(RuntimeError):
    pass


def _group_key(pid: str) -> str:
    if pid.startswith(LEGACY_PREFIX):
        return LEGACY_PREFIX + "*"
    return pid.split("_", 1)[0] if pid else "<empty>"


def _counts(rows) -> dict:
    return dict(collections.Counter(
        _group_key(str(r.get("prediction_id", ""))) for r in rows if isinstance(r, dict)
    ))


def plan(data: dict):
    """Pure planner: returns (new_calibration_list, report). Never mutates data."""
    cal = data.get("calibration")
    if not isinstance(cal, list):
        raise DedupeError("calibration is not a list")
    legacy = {
        i for i, r in enumerate(cal)
        if isinstance(r, dict) and str(r.get("prediction_id", "")).startswith(LEGACY_PREFIX)
    }
    keep_at = max(legacy) if legacy else None
    new = []
    for i, row in enumerate(cal):
        if i in legacy:
            if i == keep_at:
                fixed = copy.deepcopy(row)
                fixed["prediction_id"] = TARGET_ID
                new.append(fixed)
            continue
        new.append(copy.deepcopy(row))
    report = {
        "rows_before": len(cal),
        "rows_after": len(new),
        "legacy_rows_before": len(legacy),
        "by_prefix_before": _counts(cal),
        "by_prefix_after": _counts(new),
    }
    return new, report


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _parse(raw: bytes) -> dict:
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise DedupeError("top-level JSON is not an object")
    return data


def _verify_written(path: str, expected_cal: list) -> None:
    data = _parse(_read_bytes(path))
    cal = data.get("calibration")
    if not isinstance(cal, list) or len(cal) != len(expected_cal):
        raise DedupeError(f"row count mismatch after swap: expected {len(expected_cal)}")
    if cal != expected_cal:
        raise DedupeError("row contents mismatch after swap")


def _write_bytes_atomic(path: str, raw: bytes) -> None:
    tmp = path + ".dedupe.restore"
    with open(tmp, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def apply(path: str, backup_dir: str, _verify=None) -> dict:
    verify = _verify or _verify_written
    lock_fh = open(path + ".lock", "a")
    try:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise DedupeError("meta_memory lock is busy; another writer is active") from None
        orig = _read_bytes(path)
        data = _parse(orig)
        new_cal, report = plan(data)
        if report["rows_after"] == report["rows_before"]:
            return {"changed": False, **report}

        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup = os.path.join(backup_dir, f"meta_memory.{stamp}.json")
        if os.path.exists(backup):
            raise DedupeError(f"backup path already exists: {backup}")
        os.link(path, backup)  # hard link: same inode as the original, no copy
        if _read_bytes(backup) != orig:
            raise DedupeError("backup does not match the original bytes")

        new_data = dict(data)
        new_data["calibration"] = new_cal
        tmp = path + ".dedupe.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            verify(path, new_cal)
        except Exception as e:
            _write_bytes_atomic(path, orig)
            if _read_bytes(path) != orig:
                raise DedupeError(f"apply failed AND restore failed; backup at {backup}") from e
            raise DedupeError(f"apply failed, original restored: {e}") from e
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return {"changed": True, "backup": backup, **report}
    finally:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
        finally:
            lock_fh.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--apply", action="store_true", help="write the change (guarded)")
    ap.add_argument("--i-understand-protected-state", action="store_true", dest="ack")
    ap.add_argument("--backup-dir", help="existing directory for the hard-link backup")
    args = ap.parse_args(argv)

    if not args.apply:
        data = _parse(_read_bytes(args.file))
        _, report = plan(data)
        print(json.dumps({"mode": "dry-run", "file": args.file, **report}, indent=2))
        return 0

    if not args.ack or not args.backup_dir or not os.path.isdir(args.backup_dir):
        print("refusing --apply: pass --i-understand-protected-state and an existing "
              "--backup-dir", file=sys.stderr)
        return 2
    try:
        result = apply(args.file, args.backup_dir)
    except DedupeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"mode": "apply", "file": args.file, **result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
