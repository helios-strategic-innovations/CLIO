#!/usr/bin/env python3
"""Read-only UW quota-ledger report (S211, ADR-005): is the pacer working?

Answers, for a time window, from the shared quota ledger (never written to):
  * how many UW requests, by HTTP status, and the 429 rate
  * requests per minute (median / p90 / max)
  * the same-second burst histogram (burst size -> requests, 429s, 429 rate)  <- the number that matters
  * retry rows (purpose *_retry) and their outcomes
  * 429 rate by endpoint family
  * the current cycle's entitlement / observed count / anomaly count

Safety: opens the ledger with ``mode=ro`` + ``query_only`` (WAL mode: readers never block the writers) and
finds the window by BISECTING ROWIDs (rows are in insertion order), so it never scans the ~2 GB file.

    python3 scripts/uw_ledger_report.py                       # today's session so far (09:30 ET -> now)
    python3 scripts/uw_ledger_report.py --since 2026-09-29T13:30:00 --until 2026-09-29T20:00:00   (UTC)
    python3 scripts/uw_ledger_report.py --json

Expected after the pacer is deployed (market hours): 429 rate < 2 %, max same-second burst <= 4, retry rows > 0
after any 429, no bursts of 12-14.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

DEFAULT_DB = "/home/agent/shared/provider_ops/uw_quota.sqlite3"
MAX_WINDOW_ROWS = 2_000_000
_NY = ZoneInfo("America/New_York")


def open_ro(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
    con.execute("PRAGMA query_only=1")
    return con


def _reserved_at(con: sqlite3.Connection, rowid: int):
    row = con.execute(
        "SELECT reserved_at FROM quota_reservations WHERE rowid >= ? ORDER BY rowid LIMIT 1", (rowid,)
    ).fetchone()
    return row[0] if row else None


def first_rowid_at_or_after(con: sqlite3.Connection, max_rowid: int, ts: str) -> int:
    """Smallest rowid whose reserved_at >= ts (ISO strings sort chronologically); max_rowid+1 if none."""
    lo, hi = 1, max_rowid + 1
    while lo < hi:
        mid = (lo + hi) // 2
        value = _reserved_at(con, mid)
        if value is None or value >= ts:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _family(endpoint: str) -> str:
    return re.sub(r"/(?:[A-Z]{1,5})(?=/|$)", "/{T}", endpoint or "")


def _pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))] if values else 0


def report(con: sqlite3.Connection, since_utc: str, until_utc: str) -> dict:
    max_rowid = con.execute("SELECT max(rowid) FROM quota_reservations").fetchone()[0] or 0
    a = first_rowid_at_or_after(con, max_rowid, since_utc)
    b = first_rowid_at_or_after(con, max_rowid, until_utc)
    if b - a > MAX_WINDOW_ROWS:
        raise SystemExit(f"window covers {b - a} rows (> {MAX_WINDOW_ROWS}); narrow --since/--until")
    rows = con.execute(
        "SELECT reserved_at, purpose, endpoint, state, response_status FROM quota_reservations "
        "WHERE rowid >= ? AND rowid < ?", (a, b),
    ).fetchall()
    done = [r for r in rows if r[3] == "completed"]
    per_minute = collections.Counter(r[0][:16] for r in done)
    per_second = collections.Counter(r[0][:19] for r in done)
    status = collections.Counter(r[4] for r in done)
    n429 = status.get(429, 0)

    burst = collections.defaultdict(lambda: [0, 0])   # burst size -> [requests, 429s]
    for reserved_at, _p, _e, _s, code in done:
        size = per_second[reserved_at[:19]]
        burst[size][0] += 1
        burst[size][1] += code == 429
    fam = collections.defaultdict(lambda: [0, 0])
    for _r, _p, endpoint, _s, code in done:
        f = _family(endpoint)
        fam[f][0] += 1
        fam[f][1] += code == 429
    retries = [r for r in done if (r[1] or "").endswith("_retry")]
    cycle = con.execute(
        "SELECT cycle_key, entitlement, observed_count, anomaly_count, observed_at FROM quota_cycles "
        "ORDER BY cycle_key DESC LIMIT 1"
    ).fetchone()
    return {
        "window_utc": [since_utc, until_utc],
        "requests": len(done),
        "denied": sum(1 for r in rows if r[3] == "denied"),
        "status": {str(k): v for k, v in sorted(status.items(), key=lambda kv: str(kv[0]))},
        "rate_429": (n429 / len(done)) if done else 0.0,
        "per_minute": {"median": _pct(per_minute.values(), 0.5), "p90": _pct(per_minute.values(), 0.9),
                       "max": max(per_minute.values(), default=0)},
        "burst_histogram": {str(k): {"requests": v[0], "429s": v[1], "rate": (v[1] / v[0]) if v[0] else 0.0}
                            for k, v in sorted(burst.items())},
        "max_same_second_burst": max(per_second.values(), default=0),
        "retries": {"rows": len(retries),
                    "status": {str(k): v for k, v in collections.Counter(r[4] for r in retries).items()}},
        "by_endpoint_family": {k: {"requests": v[0], "429s": v[1], "rate": (v[1] / v[0]) if v[0] else 0.0}
                               for k, v in sorted(fam.items(), key=lambda kv: -kv[1][0])},
        "cycle": dict(zip(("cycle_key", "entitlement", "observed_count", "anomaly_count", "observed_at"),
                          cycle)) if cycle else None,
    }


def render(rep: dict) -> str:
    out = [f"UW ledger report  window (UTC): {rep['window_utc'][0]} -> {rep['window_utc'][1]}",
           f"  requests: {rep['requests']}   denied: {rep['denied']}   status: {rep['status']}",
           f"  429 rate: {rep['rate_429']:.1%}",
           f"  per minute: median={rep['per_minute']['median']} p90={rep['per_minute']['p90']} max={rep['per_minute']['max']}",
           f"  max same-second burst: {rep['max_same_second_burst']}",
           "  burst size -> requests / 429s / rate"]
    for size, v in rep["burst_histogram"].items():
        out.append(f"    {size:>3}: {v['requests']:>5} / {v['429s']:>4} / {v['rate']:.0%}")
    out.append(f"  retry rows: {rep['retries']['rows']}   outcomes: {rep['retries']['status']}")
    out.append("  by endpoint family (requests / 429s / rate):")
    for fam, v in list(rep["by_endpoint_family"].items())[:12]:
        out.append(f"    {fam:<44} {v['requests']:>5} / {v['429s']:>4} / {v['rate']:.0%}")
    if rep["cycle"]:
        c = rep["cycle"]
        out.append(f"  cycle {c['cycle_key']}: entitlement={c['entitlement']} observed={c['observed_count']} "
                   f"anomalies={c['anomaly_count']} last_header={c['observed_at']}")
    return "\n".join(out)


def _default_window() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    open_et = datetime.now(_NY).replace(hour=9, minute=30, second=0, microsecond=0)
    return open_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), now.strftime("%Y-%m-%dT%H:%M:%S")


def main(argv=None) -> int:
    d_since, d_until = _default_window()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--since", default=d_since, help="UTC ISO timestamp (default: today 09:30 ET)")
    ap.add_argument("--until", default=d_until, help="UTC ISO timestamp (default: now)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    con = open_ro(args.db)
    try:
        rep = report(con, args.since, args.until)
    finally:
        con.close()
    print(json.dumps(rep, indent=1) if args.json else render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
