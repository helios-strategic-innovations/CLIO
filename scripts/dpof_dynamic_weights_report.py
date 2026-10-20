#!/usr/bin/env python3
"""Offline, read-only shadow-vs-static report for DPOF dynamic pillar weights.

Reads the shadow sidecar (pillar_weights_shadow.jsonl) and the engine signal log
(signals.jsonl) and prints the evidence for design doc
docs/design/clio_dpof_dynamic_weights.md section 6. Stdlib only, no agent.* imports,
never writes a file (output goes to stdout only).
"""

import argparse
import gzip
import json
import math
import os
import statistics
import sys
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "data", "dpof")
_SHADOW_NAME = "pillar_weights_shadow.jsonl"
_SIGNALS_NAME = "signals.jsonl"

DIRECTIONAL = ("nofa", "premium", "darkpool", "kronos")
GEX_WEIGHT = 0.15
THRESHOLD_1 = 0.15
THRESHOLD_2 = 0.30
EPISODE_GAP_S = 300
BAND_LO = 0.67
BAND_HI = 1.5
MIN_SESSIONS = 15
MAX_FLIPS_PER_30MIN = 1
MAX_WEIGHT_CHANGE = 0.10
MIN_JOIN_RATE = 0.99
RTH_GAP_S = 300
FLIP_BUCKET_S = 1800
MAX_EXAMPLES = 5


# ---------------------------------------------------------------- helpers

def _num(x):
    """Finite float or None (bools, NaN, inf and non-numbers are missing)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def _parse_ts(ts):
    if not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    return dt


def _epoch(ts):
    dt = _parse_ts(ts)
    return dt.timestamp() if dt is not None else None


def _u(rec):
    v = rec.get("u")
    return v if v is not None else rec.get("underlying")


def _timed(records):
    """(epoch, record) pairs for records with a parseable ts, sorted by time."""
    out = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        ep = _epoch(rec.get("ts"))
        if ep is not None:
            out.append((ep, rec))
    out.sort(key=lambda p: p[0])
    return out


def _clamp(x):
    return min(1.0, max(-1.0, x))


def _same(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def load_jsonl(path, max_lines=None):
    """Valid JSON-object lines of a file, in order; torn/invalid lines skipped.

    With max_lines, only the last max_lines valid records are kept. Missing,
    unreadable or empty files give []. Read-only ("r" only).
    """
    out = deque(maxlen=max_lines)
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return []
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return list(out)


_ARCHIVE_DIR_NAME = "archive"
_ARCHIVE_PREFIX = "signals_"
_ARCHIVE_SUFFIX = ".jsonl.gz"


def _parse_jsonl_lines(lines):
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _read_archive(path):
    """All records of one gzip archive. Any failure (OSError, truncated gzip
    EOFError, corrupt deflate zlib.error, ...) raises: the caller skips the
    WHOLE archive so a partial file never looks complete."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        return _parse_jsonl_lines(fh)


def load_signals(signals_path, include_archive=True, err=None):
    """Trimmed-head archives (<dir>/archive/signals_*.jsonl.gz, oldest name
    first) followed by the live signals.jsonl. Read-only. Unreadable or corrupt
    archives are skipped with a note on `err` (default stderr)."""
    err = err if err is not None else sys.stderr
    records = []
    if include_archive:
        archive_dir = os.path.join(os.path.dirname(os.path.abspath(signals_path)), _ARCHIVE_DIR_NAME)
        try:
            names = sorted(
                n for n in os.listdir(archive_dir)
                if n.startswith(_ARCHIVE_PREFIX) and n.endswith(_ARCHIVE_SUFFIX)
            )
        except OSError:
            names = []
        for name in names:
            try:
                records.extend(_read_archive(os.path.join(archive_dir, name)))
            except Exception as e:  # any corruption/unreadability: note and skip
                err.write(f"note: skipped unreadable signals archive {name} ({type(e).__name__})\n")
    records.extend(load_jsonl(signals_path))
    return records


def session_day(ts):
    """ET calendar date (YYYY-MM-DD) of an ISO timestamp, or None."""
    dt = _parse_ts(ts)
    if dt is None:
        return None
    return dt.astimezone(_ET).date().isoformat()


def _rth_bounds(day):
    y, m, d = (int(p) for p in day.split("-"))
    return (datetime(y, m, d, 9, 30, tzinfo=_ET).timestamp(),
            datetime(y, m, d, 16, 0, tzinfo=_ET).timestamp())


# ---------------------------------------------------------------- analyses

def coverage(records):
    """Per session: cycle counts per underlying, first/last ts, RTH gaps > 5 min."""
    by_key = defaultdict(list)
    for ep, rec in _timed(records):
        day = session_day(rec.get("ts"))
        if day is None:
            continue
        by_key[(day, _u(rec))].append((ep, rec.get("ts")))
    sessions = {}
    for (day, u), items in sorted(by_key.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        s = sessions.setdefault(day, {"cycles": {}, "first": None, "last": None, "gaps": []})
        s["cycles"][str(u)] = len(items)
        first, last = items[0], items[-1]
        if s["first"] is None or first[0] < s["_first_ep"]:
            s["first"], s["_first_ep"] = first[1], first[0]
        if s["last"] is None or last[0] > s["_last_ep"]:
            s["last"], s["_last_ep"] = last[1], last[0]
        open_, close_ = _rth_bounds(day)
        for (a_ep, a_ts), (b_ep, b_ts) in zip(items, items[1:]):
            lo, hi = max(a_ep, open_), min(b_ep, close_)
            if hi - lo > RTH_GAP_S:
                s["gaps"].append({"u": str(u), "start": a_ts, "end": b_ts,
                                  "minutes": round((b_ep - a_ep) / 60.0, 2)})
    for s in sessions.values():
        s.pop("_first_ep", None)
        s.pop("_last_ep", None)
    return sessions


def dead_episodes(records):
    """Dead-pillar episodes per (underlying, pillar) and flips per 30-minute bucket."""
    by_u = defaultdict(list)
    for ep, rec in _timed(records):
        by_u[str(_u(rec))].append((ep, rec.get("ts"), rec))
    episodes = []
    flip_bins = defaultdict(int)     # (u, pillar, bucket) -> count
    flip_total = defaultdict(int)    # "u|pillar" -> count
    for u in sorted(by_u):
        prev, opened = set(), {}
        for ep, ts, rec in by_u[u]:
            dead = rec.get("dead")
            if not isinstance(dead, list):
                continue
            cur = {str(p) for p in dead}
            for p in sorted(cur - prev):
                opened[p] = ts
                flip_bins[(u, p, int(ep // FLIP_BUCKET_S))] += 1
                flip_total[f"{u}|{p}"] += 1
            for p in sorted(prev - cur):
                episodes.append({"u": u, "pillar": p, "start": opened.pop(p, None), "end": ts})
                flip_bins[(u, p, int(ep // FLIP_BUCKET_S))] += 1
                flip_total[f"{u}|{p}"] += 1
            prev = cur
        for p in sorted(prev):
            episodes.append({"u": u, "pillar": p, "start": opened.pop(p, None), "end": None})
    max_by_pillar = defaultdict(int)
    for (u, p, _b), n in flip_bins.items():
        max_by_pillar[p] = max(max_by_pillar[p], n)
    return {
        "episodes": episodes,
        "flips": dict(sorted(flip_total.items())),
        "max_flips_per_30min": dict(sorted(max_by_pillar.items())),
        "max_flips_per_30min_overall": max(max_by_pillar.values()) if max_by_pillar else 0,
    }


def _identity_scan(records, tol):
    violations, max_dev, eligible = [], 0.0, 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        dead = rec.get("dead")
        alive = rec.get("alive")
        m = rec.get("m")
        if not (isinstance(dead, list) and dead == []):
            continue
        if not isinstance(alive, dict) or not all(_num(v) == 1.0 for v in alive.values()):
            continue
        if not isinstance(m, dict) or not all(_num(v) == 1.0 for v in m.values()):
            continue
        st, dy = _num(rec.get("static")), _num(rec.get("dyn"))
        if st is None or dy is None:
            continue
        eligible += 1
        dev = abs(dy - st)
        max_dev = max(max_dev, dev)
        if dev > tol:
            violations.append({"ts": rec.get("ts"), "u": _u(rec), "static": st, "dyn": dy, "dev": dev})
    return violations, max_dev, eligible


def identity_violations(records, tol=1e-12):
    """(violations, max deviation) over rows where nothing is dead and m == 1."""
    violations, max_dev, _n = _identity_scan(records, tol)
    return violations, max_dev


def episodes(series, threshold=THRESHOLD_1, gap_seconds=EPISODE_GAP_S):
    """Nomination episodes from (ts, value) pairs: list of (start_ts, sign).

    Each sample with |value| >= threshold is a crossing. Consecutive crossings
    with the same sign and a gap <= gap_seconds form one episode.
    """
    pts = []
    for ts, v in series:
        ep, x = _epoch(ts), _num(v)
        if ep is None or x is None or abs(x) < threshold:
            continue
        pts.append((ep, ts, 1 if x > 0 else -1))
    pts.sort(key=lambda p: p[0])
    out = []
    cur_start = cur_sign = last_ep = None
    for ep, ts, sign in pts:
        if cur_start is not None and sign == cur_sign and ep - last_ep <= gap_seconds:
            last_ep = ep
            continue
        if cur_start is not None:
            out.append((cur_start, cur_sign))
        cur_start, cur_sign, last_ep = ts, sign, ep
    if cur_start is not None:
        out.append((cur_start, cur_sign))
    return out


def _ratio_in_band(s_eps, d_eps):
    if s_eps == 0:
        ratio = 1.0 if d_eps == 0 else None
    else:
        ratio = d_eps / s_eps
    ok = ratio is not None and BAND_LO <= ratio <= BAND_HI
    return ratio, ok


def divergence(records):
    """Per (session, underlying): threshold cycle counts, episode counts, dyn/static ratio."""
    groups = defaultdict(list)
    for ep, rec in _timed(records):
        day = session_day(rec.get("ts"))
        if day is not None:
            groups[(day, str(_u(rec)))].append((rec.get("ts"), _num(rec.get("static")), _num(rec.get("dyn"))))
    rows = []
    for (day, u) in sorted(groups):
        items = groups[(day, u)]
        st = [(ts, s) for ts, s, _d in items]
        dy = [(ts, d) for ts, _s, d in items]
        s_eps = len(episodes(st))
        d_eps = len(episodes(dy))
        ratio, ok = _ratio_in_band(s_eps, d_eps)
        rows.append({
            "session": day, "u": u, "cycles": len(items),
            "static_ge_0.15": sum(1 for _t, s, _d in items if s is not None and abs(s) >= THRESHOLD_1),
            "dyn_ge_0.15": sum(1 for _t, _s, d in items if d is not None and abs(d) >= THRESHOLD_1),
            "static_ge_0.30": sum(1 for _t, s, _d in items if s is not None and abs(s) >= THRESHOLD_2),
            "dyn_ge_0.30": sum(1 for _t, _s, d in items if d is not None and abs(d) >= THRESHOLD_2),
            "static_episodes": s_eps, "dyn_episodes": d_eps,
            "ratio": ratio, "in_band": ok,
        })
    out_of_band = [{"session": r["session"], "u": r["u"], "ratio": r["ratio"]}
                   for r in rows if not r["in_band"]]
    return {"rows": rows, "out_of_band": out_of_band, "band": [BAND_LO, BAND_HI]}


def weight_stability(records):
    """Per session and pillar: min/mean/max weight; max change of consecutive session means."""
    vals = defaultdict(lambda: defaultdict(list))  # session -> pillar -> [w]
    for ep, rec in _timed(records):
        day = session_day(rec.get("ts"))
        w = rec.get("w")
        if day is None or not isinstance(w, dict):
            continue
        for p, v in w.items():
            x = _num(v)
            if x is not None:
                vals[day][str(p)].append(x)
    sessions = {}
    for day in sorted(vals):
        sessions[day] = {}
        for p, xs in sorted(vals[day].items()):
            sessions[day][p] = {"min": min(xs), "mean": math.fsum(xs) / len(xs),
                                "max": max(xs), "n": len(xs)}
    max_change = {}
    days = sorted(sessions)
    for prev, cur in zip(days, days[1:]):
        for p in sessions[cur]:
            if p in sessions[prev]:
                d = abs(sessions[cur][p]["mean"] - sessions[prev][p]["mean"])
                max_change[p] = max(max_change.get(p, 0.0), d)
    overall = max(max_change.values()) if max_change else None
    return {"sessions": sessions, "max_session_mean_change": max_change,
            "max_change": overall}


def _rebuild_dyn(rec, row):
    """Composite rebuilt from the signal row's pillar values and the shadow w/avail."""
    w, avail = rec.get("w"), rec.get("avail")
    if not isinstance(w, dict) or not isinstance(avail, list):
        return None
    pillars = row.get("pillars") if isinstance(row.get("pillars"), dict) else {}
    kron = row.get("kronos") if isinstance(row.get("kronos"), dict) else {}
    num = den = 0.0
    for p in avail:
        if p not in DIRECTIONAL:
            continue
        W = _num(w.get(p))
        if p == "kronos":
            val = _num(kron.get("pillar"))
        else:
            pv = pillars.get(p)
            val = _num(pv.get("value")) if isinstance(pv, dict) else None
        if W is None or val is None:
            return None
        num += W * val
        den += W
    if den <= 0:
        return None
    gex = pillars.get("gex")
    gval = _num(gex.get("value")) if isinstance(gex, dict) else None
    if gval is None:
        return _clamp(num / den)
    pre = num / den
    sign = (pre > 0) - (pre < 0)
    contrib = gval * sign
    return _clamp((num + GEX_WEIGHT * contrib) / (den + GEX_WEIGHT))


def replay_check(shadow_records, signal_rows, tol=1e-9):
    """Rebuild dyn from signals and compare; compare static with logged composite."""
    sig = {}
    for row in signal_rows:
        if not isinstance(row, dict):
            continue
        key = (row.get("ts"), row.get("underlying"))
        if key not in sig:
            sig[key] = row
    shadow = [r for r in shadow_records if isinstance(r, dict)]
    joined = dyn_checked = dyn_match = static_checked = static_match = 0
    examples = []
    for rec in shadow:
        u = _u(rec)
        row = sig.get((rec.get("ts"), u))
        if row is None:
            continue
        joined += 1
        rebuilt = _rebuild_dyn(rec, row)
        logged_dyn = _num(rec.get("dyn"))
        dyn_checked += 1
        if _same(rebuilt, logged_dyn, tol):
            dyn_match += 1
        elif len(examples) < MAX_EXAMPLES:
            examples.append({"ts": rec.get("ts"), "u": u, "kind": "dyn",
                             "expected": logged_dyn, "rebuilt": rebuilt})
        basis = row.get("signal_basis")
        if basis is None or basis == u:
            static = _num(rec.get("static"))
            logged = _num(row.get("composite"))
            static_checked += 1
            if _same(static, logged, tol):
                static_match += 1
            elif len(examples) < MAX_EXAMPLES:
                examples.append({"ts": rec.get("ts"), "u": u, "kind": "static",
                                 "expected": logged, "rebuilt": static})
    n = len(shadow)
    return {
        "shadow_rows": n, "joined": joined,
        "join_rate": (joined / n) if n else None,
        "dyn_checked": dyn_checked, "dyn_match": dyn_match,
        "dyn_mismatch": dyn_checked - dyn_match,
        "static_checked": static_checked, "static_match": static_match,
        "static_mismatch": static_checked - static_match,
        "mismatches": (dyn_checked - dyn_match) + (static_checked - static_match),
        "examples": examples,
    }


def l2_diagnostic(records):
    """Last n_eff and diagnostic multiplier per underlying and pillar."""
    last = {}
    for _ep, rec in _timed(records):
        last[str(_u(rec))] = rec
    out = {}
    for u in sorted(last):
        rec = last[u]
        ne = rec.get("n_eff") if isinstance(rec.get("n_eff"), dict) else {}
        mm = rec.get("m") if isinstance(rec.get("m"), dict) else {}
        pillars = sorted(set(ne) | set(mm))
        out[u] = {str(p): {"n_eff": _num(ne.get(p)), "m": _num(mm.get(p))} for p in pillars}
    return out


def promotion_checklist(report):
    """Design doc section 6 items 1-4 as PASS / FAIL / INSUFFICIENT rows."""
    n = report["session_count"]
    short = n < MIN_SESSIONS
    rep = report["replay"]
    ident = report["identity"]
    flips = report["dead_episodes"]["max_flips_per_30min_overall"]
    div = report["divergence"]
    wchg = report["weight_stability"]["max_change"]
    items = []

    def add(criterion, status, detail):
        items.append({"criterion": criterion, "status": status, "detail": detail})

    crit1 = (">= 15 sessions and replay reproduces logged shadow dyn/static "
             "(0 mismatches, join rate >= 0.99)")
    rd = (f"sessions={n}; replay mismatches={rep['mismatches']}; "
          f"join_rate={rep['join_rate']}")
    if short:
        add(crit1, "INSUFFICIENT", rd + f"; need {MIN_SESSIONS} sessions")
    elif rep["join_rate"] is None:
        add(crit1, "INSUFFICIENT", rd + "; no shadow rows")
    else:
        ok = rep["mismatches"] == 0 and rep["join_rate"] >= MIN_JOIN_RATE
        add(crit1, "PASS" if ok else "FAIL", rd)

    crit2 = f"liveness: <= {MAX_FLIPS_PER_30MIN} flip per pillar per 30 minutes"
    if short:
        add(crit2, "INSUFFICIENT", f"sessions={n}; max flips per 30 min={flips}")
    else:
        add(crit2, "PASS" if flips <= MAX_FLIPS_PER_30MIN else "FAIL",
            f"max flips per 30 min={flips}")

    crit3 = f"nomination episode ratio dyn/static within [{BAND_LO}, {BAND_HI}] every session"
    if short:
        add(crit3, "INSUFFICIENT", f"sessions={n}; rows out of band={len(div['out_of_band'])}")
    else:
        add(crit3, "PASS" if not div["out_of_band"] else "FAIL",
            f"rows out of band={len(div['out_of_band'])}")

    crit4a = "identity: dyn == static to 1e-12 wherever nothing is dead and m == 1"
    if ident["eligible"] == 0:
        add(crit4a, "INSUFFICIENT", "no eligible rows")
    else:
        add(crit4a, "PASS" if ident["violations"] == 0 else "FAIL",
            f"eligible={ident['eligible']}; violations={ident['violations']}; "
            f"max deviation={ident['max_dev']}")

    crit4b = f"max session-mean weight change <= {MAX_WEIGHT_CHANGE} (design section 3 move cap)"
    if short:
        add(crit4b, "INSUFFICIENT", f"sessions={n}; max change={wchg}")
    elif wchg is None:
        add(crit4b, "INSUFFICIENT", "fewer than two sessions with weights")
    else:
        add(crit4b, "PASS" if wchg <= MAX_WEIGHT_CHANGE else "FAIL", f"max change={wchg}")
    return items


# ---------------------------------------------------------------- descriptive alternatives (R-38)

_DESC_SAT = 0.999


def _desc_stats(vals):
    """n, saturation share (|v| >= 0.999), mean and population sd of the finite values."""
    xs = [x for x in vals if x is not None]
    if not xs:
        return {"n": 0, "saturation_share": None, "mean": None, "sd": None}
    return {
        "n": len(xs),
        "saturation_share": round(sum(1 for x in xs if abs(x) >= _DESC_SAT) / len(xs), 4),
        "mean": round(statistics.fmean(xs), 4),
        "sd": round(statistics.pstdev(xs), 4),
    }


def _sign3(x):
    return (x > 0) - (x < 0)


def descriptive_alternatives(records):
    """DESCRIPTIVE ONLY (no edge claims). Summarises the R-37 shadow keys: premium_alt (live value and the
    w60 / ewma / day alternatives), premium_sat_run and gex_cont. Rows without a key are skipped."""
    alt_rows = [r["premium_alt"] for r in records
                if isinstance(r, dict) and isinstance(r.get("premium_alt"), dict)]
    live = [_num(a.get("live")) for a in alt_rows]
    premium = {"live": dict(_desc_stats(live), sign_agreement=None)}
    for k in ("w60", "ewma", "day"):
        vals = [_num(a.get(k)) for a in alt_rows]
        pairs = [(v, lv) for v, lv in zip(vals, live) if v is not None and lv is not None]
        agree = round(sum(1 for v, lv in pairs if _sign3(v) == _sign3(lv)) / len(pairs), 4) if pairs else None
        premium[k] = dict(_desc_stats(vals), sign_agreement=agree)
    runs = [r["premium_sat_run"] for r in records
            if isinstance(r, dict) and isinstance(r.get("premium_sat_run"), int)
            and not isinstance(r.get("premium_sat_run"), bool)]
    gex = [_num(r.get("gex_cont")) for r in records if isinstance(r, dict)]
    gst = _desc_stats(gex)
    return {
        "note": "descriptive only; no edge claims",
        "premium_alt_rows": len(alt_rows),
        "premium": premium,
        "premium_sat_run": {"n": len(runs), "max": max(runs) if runs else None,
                            "mean": round(statistics.fmean(runs), 4) if runs else None},
        "gex_cont": {"n": gst["n"], "mean": gst["mean"], "sd": gst["sd"]},
    }


# ---------------------------------------------------------------- report

def build_report(shadow_records, signal_rows):
    """All analyses as one JSON-able dict (inputs already filtered)."""
    sessions = sorted({session_day(r.get("ts")) for r in shadow_records} - {None})
    viol, max_dev, eligible = _identity_scan(shadow_records, 1e-12)
    report = {
        "shadow_rows": len(shadow_records),
        "signal_rows": len(signal_rows),
        "sessions": sessions,
        "session_count": len(sessions),
        "coverage": coverage(shadow_records),
        "dead_episodes": dead_episodes(shadow_records),
        "identity": {"eligible": eligible, "violations": len(viol),
                     "max_dev": max_dev, "examples": viol[:MAX_EXAMPLES]},
        "divergence": divergence(shadow_records),
        "weight_stability": weight_stability(shadow_records),
        "replay": replay_check(shadow_records, signal_rows),
        "l2": l2_diagnostic(shadow_records),
        "descriptive_alternatives": descriptive_alternatives(shadow_records),
    }
    report["checklist"] = promotion_checklist(report)
    return report


def to_jsonable(obj):
    """Recursively map NaN/inf to None and tuples to lists; keys to str."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def dumps(obj):
    return json.dumps(to_jsonable(obj), allow_nan=False, sort_keys=True, indent=2)


def _fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def _text(report, paths, days):
    out = []
    w = out.append
    w("DPOF dynamic pillar weights - shadow vs static report")
    w(f"shadow: {paths['shadow']} ({report['shadow_rows']} rows)")
    w(f"signals: {paths['signals']} ({report['signal_rows']} rows)")
    w(f"sessions: {report['session_count']}" + (f" (last {days})" if days else ""))
    if report["shadow_rows"] == 0:
        w("NO DATA: no usable shadow records; the sections below are empty.")
    w("")
    w("== Coverage ==")
    for day, s in report["coverage"].items():
        cyc = " ".join(f"{u}={n}" for u, n in sorted(s["cycles"].items()))
        w(f"  {day}  {cyc}  {s['first']} .. {s['last']}  gaps={len(s['gaps'])}")
        for g in s["gaps"]:
            w(f"    gap {g['u']} {g['start']} .. {g['end']} ({g['minutes']} min)")
    if not report["coverage"]:
        w("  no data")
    w("")
    de = report["dead_episodes"]
    w("== Dead episodes ==")
    for e in de["episodes"]:
        w(f"  {e['u']} {e['pillar']}: {e['start']} .. {e['end'] if e['end'] else 'still dead'}")
    if not de["episodes"]:
        w("  none")
    w(f"  max flips per 30 min: {de['max_flips_per_30min_overall']}")
    w("")
    idt = report["identity"]
    w("== Identity check ==")
    w(f"  eligible={idt['eligible']} violations={idt['violations']} max deviation={_fmt(idt['max_dev'])}")
    w("")
    dv = report["divergence"]
    w("== Divergence ==")
    for r in dv["rows"]:
        w(f"  {r['session']} {r['u']}: cycles={r['cycles']} |s|>=0.15 {r['static_ge_0.15']}/{r['dyn_ge_0.15']} "
          f"|s|>=0.30 {r['static_ge_0.30']}/{r['dyn_ge_0.30']} episodes {r['static_episodes']}/{r['dyn_episodes']} "
          f"ratio={_fmt(r['ratio'])} {'in-band' if r['in_band'] else 'OUT'}")
    if not dv["rows"]:
        w("  no data")
    w("")
    ws = report["weight_stability"]
    w("== Weight stability ==")
    for day, pills in ws["sessions"].items():
        for p, st in pills.items():
            w(f"  {day} {p}: min={_fmt(st['min'])} mean={_fmt(st['mean'])} max={_fmt(st['max'])} n={st['n']}")
    w(f"  max session-mean change: {_fmt(ws['max_change'])}")
    w("")
    rp = report["replay"]
    w("== Replay check ==")
    w(f"  shadow rows={rp['shadow_rows']} joined={rp['joined']} join rate={_fmt(rp['join_rate'])}")
    w(f"  dyn: checked={rp['dyn_checked']} match={rp['dyn_match']} mismatch={rp['dyn_mismatch']}")
    w(f"  static vs logged: checked={rp['static_checked']} match={rp['static_match']} mismatch={rp['static_mismatch']}")
    for ex in rp["examples"]:
        w(f"  example {ex['ts']} {ex['u']} {ex['kind']}: logged={_fmt(ex['expected'])} rebuilt={_fmt(ex['rebuilt'])}")
    w("")
    w("== L2 diagnostic ==")
    for u, pills in report["l2"].items():
        for p, v in pills.items():
            w(f"  {u} {p}: n_eff={_fmt(v['n_eff'])} m={_fmt(v['m'])}")
    if not report["l2"]:
        w("  no data")
    w("")
    da = report["descriptive_alternatives"]
    w("== Descriptive premium/GEX alternatives (shadow only) ==")
    w("  descriptive only; no edge claims")
    if da["premium_alt_rows"] == 0 and da["premium_sat_run"]["n"] == 0 and da["gex_cont"]["n"] == 0:
        w("  no data (rows without the R-37 keys are skipped)")
    else:
        w(f"  premium rows with alternatives: {da['premium_alt_rows']}")
        for k in ("live", "w60", "ewma", "day"):
            st = da["premium"][k]
            agree = "" if k == "live" else f" sign-agree={_fmt(st['sign_agreement'])}"
            w(f"  {k}: n={st['n']} saturation={_fmt(st['saturation_share'])} mean={_fmt(st['mean'])} "
              f"sd={_fmt(st['sd'])}{agree}")
        sr = da["premium_sat_run"]
        w(f"  premium_sat_run: n={sr['n']} max={_fmt(sr['max'])} mean={_fmt(sr['mean'])}")
        gc = da["gex_cont"]
        w(f"  gex_cont: n={gc['n']} mean={_fmt(gc['mean'])} sd={_fmt(gc['sd'])}")
    w("")
    w("== Promotion checklist ==")
    for item in report["checklist"]:
        w(f"  [{item['status']}] {item['criterion']}")
        w(f"      {item['detail']}")
    return "\n".join(out) + "\n"


def _select_last_sessions(records, days):
    if not days:
        return records
    sessions = sorted({session_day(r.get("ts")) for r in records} - {None})
    keep = set(sessions[-days:])
    return [r for r in records if session_day(r.get("ts")) in keep]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only DPOF dynamic-weights shadow report.")
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    parser.add_argument("--shadow", default=None)
    parser.add_argument("--signals", default=None)
    parser.add_argument("--days", type=int, default=None, help="only the last N sessions")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--include-archive", dest="include_archive", action="store_true", default=True,
                        help="also read trimmed signals archives (default)")
    parser.add_argument("--no-archive", dest="include_archive", action="store_false",
                        help="read only the live signals.jsonl")
    args = parser.parse_args(argv)
    if args.days is not None and args.days < 1:
        parser.error("--days must be >= 1")

    shadow_path = args.shadow or os.path.join(args.data_dir, _SHADOW_NAME)
    signals_path = args.signals or os.path.join(args.data_dir, _SIGNALS_NAME)
    shadow = _select_last_sessions(load_jsonl(shadow_path), args.days)
    signals = _select_last_sessions(load_signals(signals_path, args.include_archive), args.days)
    report = build_report(shadow, signals)
    paths = {"shadow": shadow_path, "signals": signals_path}
    if args.json:
        sys.stdout.write(dumps(report) + "\n")
    else:
        sys.stdout.write(_text(report, paths, args.days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
