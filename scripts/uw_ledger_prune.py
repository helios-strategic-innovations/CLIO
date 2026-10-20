#!/usr/bin/env python3
"""Prune the shared UW quota ledger (S211, LED-1; plan: docs/plans/S211_followups_ledger_retention_and_trade_policy.md).

The ledger (/home/agent/shared/provider_ops/uw_quota.sqlite3) grew to ~2 GB / 3.4 M rows with no retention.
Every outbox row is synced to Supabase; the local rows are only a buffer + local history.

    python3 scripts/uw_ledger_prune.py --older-than-days 14                     # DRY RUN (default, read-only)
    python3 scripts/uw_ledger_prune.py --older-than-days 14 --apply --vacuum     # backup -> delete -> vacuum -> verify

Guarantees
  * DRY RUN unless --apply. The dry run opens the DB read-only and never writes anything.
  * NEVER deletes an unsynced outbox row (synced_at IS NULL) or a pending reservation, whatever its age.
    Only rows in a terminal state (completed / denied / released / expired) are eligible. quota_cycles is untouched.
  * --apply requires a verified backup (made here with SQLite's online backup API) unless --backup-file names one.
  * Rows are selected by ROWID range (insertion order), bisecting the cutoff -- no full scans -- and deleted in short
    batches (default 10 000), one transaction each, with a pause and periodic WAL checkpoints so the single writer
    (moltbot) is never starved. Three consecutive lock timeouts abort the run (rows already deleted stay deleted; re-run).
  * Refuses --older-than-days < 7, a non-WAL database, or an unexpected schema.
  * --vacuum reclaims the file space (DELETE alone does not shrink it). It blocks writers for the (short) duration.

Exit codes: 0 ok / dry run, 2 refused, 3 aborted (lock contention), 4 verification failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

DEFAULT_DB = "/home/agent/shared/provider_ops/uw_quota.sqlite3"
DEFAULT_BACKUP_DIR = "/home/agent/backups/uw_quota"
TERMINAL_STATES = ("completed", "denied", "released", "expired")
MIN_DAYS = 7
MAX_CONSECUTIVE_LOCKS = 3

TABLES = {
    # table: (timestamp column, guard SQL appended to the rowid-range delete)
    "supabase_outbox": ("created_at", "AND synced_at IS NOT NULL"),
    "quota_reservations": ("reserved_at", "AND state IN ('completed','denied','released','expired')"),
}
REQUIRED_COLUMNS = {
    "supabase_outbox": {"event_id", "payload_json", "created_at", "synced_at"},
    "quota_reservations": {"request_id", "cycle_key", "state", "reserved_at"},
    "quota_cycles": {"cycle_key", "entitlement"},
}


class Refused(Exception):
    pass


class Aborted(Exception):
    pass


# ---------------------------------------------------------------------------------------------
def connect(path: str, *, readonly: bool, busy_ms: int = 5000) -> sqlite3.Connection:
    mode = "ro" if readonly else "rw"
    con = sqlite3.connect(f"file:{path}?mode={mode}", uri=True, timeout=busy_ms / 1000.0, isolation_level=None)
    con.execute(f"PRAGMA busy_timeout={int(busy_ms)}")
    if readonly:
        con.execute("PRAGMA query_only=1")
    return con


def check_schema(con: sqlite3.Connection) -> None:
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    if str(mode).lower() != "wal":
        raise Refused(f"database journal_mode is {mode!r}, expected 'wal' -- refusing (a writer could be starved)")
    for table, cols in REQUIRED_COLUMNS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if not cols <= have:
            raise Refused(f"table {table!r} lacks columns {sorted(cols - have)} -- unexpected schema")


def _ts_at(con, table, col, rowid):
    row = con.execute(f"SELECT {col} FROM {table} WHERE rowid >= ? ORDER BY rowid LIMIT 1", (rowid,)).fetchone()
    return row[0] if row else None


def cutoff_rowid(con: sqlite3.Connection, table: str, col: str, cutoff_iso: str) -> tuple[int, int]:
    """(max_rowid, first rowid whose timestamp >= cutoff_iso). ISO strings sort chronologically."""
    max_rowid = con.execute(f"SELECT max(rowid) FROM {table}").fetchone()[0] or 0
    lo, hi = 1, max_rowid + 1
    while lo < hi:
        mid = (lo + hi) // 2
        value = _ts_at(con, table, col, mid)
        if value is None or value >= cutoff_iso:
            hi = mid
        else:
            lo = mid + 1
    return max_rowid, lo


def plan(con: sqlite3.Connection, cutoff_iso: str) -> dict:
    out = {"cutoff_utc": cutoff_iso, "tables": {}}
    for table, (col, _guard) in TABLES.items():
        max_rowid, cut = cutoff_rowid(con, table, col, cutoff_iso)
        in_range = max(0, cut - 1)
        if table == "supabase_outbox":
            kept_guard = con.execute("SELECT count(*) FROM supabase_outbox WHERE synced_at IS NULL AND rowid < ?", (cut,)).fetchone()[0]
        else:
            kept_guard = con.execute("SELECT count(*) FROM quota_reservations WHERE state='pending' AND rowid < ?", (cut,)).fetchone()[0]
        out["tables"][table] = {
            "rows_total": max_rowid, "cutoff_rowid": cut, "rows_older_than_cutoff": in_range,
            "protected_in_range": kept_guard, "eligible_at_most": max(0, in_range - kept_guard),
        }
    out["file_bytes"] = os.path.getsize(con.execute("PRAGMA database_list").fetchone()[2])
    return out


def backup(src_path: str, dest_file: str) -> dict:
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    if os.path.exists(dest_file):
        raise Refused(f"backup target already exists: {dest_file} (never overwritten)")
    src = connect(src_path, readonly=True)
    dst = sqlite3.connect(dest_file)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    os.chmod(dest_file, 0o600)
    return verify_backup(src_path, dest_file)


def verify_backup(src_path: str, dest_file: str) -> dict:
    b = connect(dest_file, readonly=True)
    s = connect(src_path, readonly=True)
    try:
        ok = b.execute("PRAGMA integrity_check").fetchone()[0]
        if ok != "ok":
            raise Refused(f"backup integrity_check failed: {ok}")
        counts = {}
        for table in ("quota_reservations", "supabase_outbox", "quota_cycles"):
            bm = b.execute(f"SELECT coalesce(max(rowid),0) FROM {table}").fetchone()[0]
            sm = s.execute(f"SELECT coalesce(max(rowid),0) FROM {table}").fetchone()[0]
            # writers continue during the online backup: the backup may lag the source by a few rows, never lead it
            if bm > sm or sm - bm > 5000:
                raise Refused(f"backup of {table} looks wrong (backup max rowid {bm}, source {sm})")
            counts[table] = bm
        return {"file": dest_file, "bytes": os.path.getsize(dest_file), "integrity": ok, "max_rowid": counts}
    finally:
        b.close()
        s.close()


def prune(path: str, cutoff_iso: str, *, batch: int, pause_s: float, busy_ms: int, log=print) -> dict:
    con = connect(path, readonly=False, busy_ms=busy_ms)
    deleted = {}
    try:
        check_schema(con)
        for table, (col, guard) in TABLES.items():
            _max, cut = cutoff_rowid(con, table, col, cutoff_iso)
            first = con.execute(f"SELECT coalesce(min(rowid),1) FROM {table}").fetchone()[0]
            total, batches, locks, a = 0, 0, 0, first
            while a < cut:
                b = min(a + batch, cut)
                try:
                    con.execute("BEGIN IMMEDIATE")
                    cur = con.execute(f"DELETE FROM {table} WHERE rowid >= ? AND rowid < ? {guard}", (a, b))
                    n = cur.rowcount
                    con.execute("COMMIT")
                    locks = 0
                except sqlite3.OperationalError as exc:
                    try:
                        con.execute("ROLLBACK")
                    except sqlite3.OperationalError:
                        pass
                    if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                        locks += 1
                        log(f"  {table}: lock timeout {locks}/{MAX_CONSECUTIVE_LOCKS} at rowid {a}")
                        if locks >= MAX_CONSECUTIVE_LOCKS:
                            raise Aborted(f"{table}: {MAX_CONSECUTIVE_LOCKS} consecutive lock timeouts; "
                                          f"{total} rows deleted so far -- re-run later") from exc
                        time.sleep(1.0)
                        continue
                    raise
                total += n
                batches += 1
                a = b
                if batches % 20 == 0:
                    con.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    log(f"  {table}: {total:,} rows deleted (rowid < {a:,} of {cut:,})")
                if pause_s:
                    time.sleep(pause_s)
            deleted[table] = {"deleted": total, "batches": batches}
        con.execute("PRAGMA wal_checkpoint(PASSIVE)")
    finally:
        con.close()
    return deleted


def vacuum(path: str, *, busy_ms: int) -> dict:
    before = os.path.getsize(path)
    con = connect(path, readonly=False, busy_ms=busy_ms)
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        t0 = time.time()
        con.execute("VACUUM")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        took = time.time() - t0
    finally:
        con.close()
    return {"bytes_before": before, "bytes_after": os.path.getsize(path), "seconds": round(took, 2)}


def verify(path: str, before_plan: dict, cutoff_iso: str) -> dict:
    con = connect(path, readonly=True)
    try:
        ok = con.execute("PRAGMA integrity_check").fetchone()[0]
        res = {"integrity": ok, "tables": {}}
        problems = []
        if ok != "ok":
            problems.append(f"integrity_check: {ok}")
        for table, (col, _g) in TABLES.items():
            old_left = con.execute(f"SELECT count(*) FROM (SELECT {col} FROM {table} ORDER BY rowid LIMIT 200) WHERE {col} < ?", (cutoff_iso,)).fetchone()[0]
            rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            res["tables"][table] = {"rows_now": rows, "old_rows_in_first_200": old_left}
        unsynced = con.execute("SELECT count(*) FROM supabase_outbox WHERE synced_at IS NULL").fetchone()[0]
        pending = con.execute("SELECT count(*) FROM quota_reservations WHERE state='pending'").fetchone()[0]
        res.update(unsynced_outbox=unsynced, pending_reservations=pending,
                   quota_cycles=con.execute("SELECT count(*) FROM quota_cycles").fetchone()[0])
        if res["quota_cycles"] < 1:
            problems.append("quota_cycles is empty")
        res["problems"] = problems
        return res
    finally:
        con.close()


# ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--older-than-days", type=int, required=True)
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--vacuum", action="store_true", help="VACUUM afterwards (requires --apply)")
    ap.add_argument("--backup-dir", default=DEFAULT_BACKUP_DIR)
    ap.add_argument("--backup-file", help="use this existing, verified backup instead of making one")
    ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--pause-ms", type=int, default=25)
    ap.add_argument("--busy-ms", type=int, default=5000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    def say(msg=""):
        if not args.json:
            print(msg, flush=True)

    try:
        if args.older_than_days < MIN_DAYS:
            raise Refused(f"--older-than-days must be >= {MIN_DAYS}")
        if args.vacuum and not args.apply:
            raise Refused("--vacuum requires --apply")
        if not os.path.isfile(args.db):
            raise Refused(f"no such database: {args.db}")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.older_than_days)).strftime("%Y-%m-%dT%H:%M:%S")
        ro = connect(args.db, readonly=True)
        try:
            check_schema(ro)
            pl = plan(ro, cutoff)
        finally:
            ro.close()
        result = {"mode": "apply" if args.apply else "dry-run", "plan": pl}
        say(f"UW ledger prune — {result['mode']}   keep {args.older_than_days} days (cutoff {cutoff} UTC)   file {pl['file_bytes'] / 1e6:,.0f} MB")
        for table, t in pl["tables"].items():
            # rows_total is the table's MAX ROWID (an upper bound on its size), not a row count: once a table has been
            # pruned the two differ by orders of magnitude (11 K rows, max rowid 1.7 M) and "total" read as 1.7 M rows.
            say(f"  {table:<20} max rowid {t['rows_total']:>10,}   older {t['rows_older_than_cutoff']:>10,}   "
                f"protected (unsynced/pending) {t['protected_in_range']:>6,}   eligible <= {t['eligible_at_most']:>10,}")
        if not args.apply:
            say("  DRY RUN: nothing was written. Add --apply (and --vacuum) to delete.")
            if args.json:
                print(json.dumps(result, indent=1))
            return 0

        if args.backup_file:
            result["backup"] = verify_backup(args.db, args.backup_file)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            result["backup"] = backup(args.db, os.path.join(args.backup_dir, f"uw_quota_{stamp}.sqlite3"))
        say(f"  backup verified: {result['backup']['file']} ({result['backup']['bytes'] / 1e6:,.0f} MB, integrity ok)")

        t0 = time.time()
        result["deleted"] = prune(args.db, cutoff, batch=args.batch, pause_s=args.pause_ms / 1000.0,
                                  busy_ms=args.busy_ms, log=say)
        say(f"  deleted in {time.time() - t0:.1f}s: " + ", ".join(f"{k}={v['deleted']:,}" for k, v in result["deleted"].items()))
        if args.vacuum:
            result["vacuum"] = vacuum(args.db, busy_ms=args.busy_ms)
            say(f"  vacuum: {result['vacuum']['bytes_before'] / 1e6:,.0f} MB -> {result['vacuum']['bytes_after'] / 1e6:,.0f} MB "
                f"in {result['vacuum']['seconds']}s")
        result["verify"] = verify(args.db, pl, cutoff)
        say(f"  verify: integrity={result['verify']['integrity']} unsynced={result['verify']['unsynced_outbox']} "
            f"pending={result['verify']['pending_reservations']} cycles={result['verify']['quota_cycles']} "
            f"rows now: " + ", ".join(f"{k}={v['rows_now']:,}" for k, v in result["verify"]["tables"].items()))
        if args.json:
            print(json.dumps(result, indent=1))
        if result["verify"]["problems"]:
            say(f"  VERIFICATION FAILED: {result['verify']['problems']}")
            return 4
        return 0
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Aborted as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
