#!/usr/bin/env python3
"""CLIO morning check (S211, PROC-1): a durable, READ-ONLY health snapshot written to a log.

Run by `clio-morning-check.timer` (systemd --user, weekdays 09:50 ET) so the verification survives session
restarts (a CronCreate job dies with its Claude session). It changes nothing: no restarts, no writes except its log
line output; secrets are never printed.

Checks: services active / restart counts; the UW ledger report since the open (429 rate, same-second bursts, retry rows);
DPOF flow-source mix from data/dpof/signals.jsonl; journal warning-shape counts since the open; Kronos forecast
freshness; Supabase decision-outbox backlog; ledger file size; MEMORY.md size; DPOF pillar liveness (are the UW premium /
darkpool pillars moving, or flat?) and the dynamic-weights shadow log (coverage, dead pillars, dyn == static invariant).

Output: one line per check tagged PASS / WARN / FAIL, then a SUMMARY line. Exit 0 = no FAIL, 1 = at least one FAIL.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
ROOT = Path(os.environ.get("CLIO_ROOT", "/home/agent/openclaw"))
SERVICES = ("moltbot", "clio-flowcollector", "clio-kronos")
MEMORY_MD = Path("/home/agent/.claude/projects/-home-agent-openclaw/memory/MEMORY.md")

# journal patterns -> (label, level when seen: 'FAIL' | 'WARN' | None=info, max tolerated count before it counts)
JOURNAL_PATTERNS = (
    ("Traceback", "FAIL", 0),
    ("UW AUTH", "FAIL", 0),
    ("apiKey=", "FAIL", 0),
    ("UW rate limited", "WARN", 20),
    ("UW_HEADER_COUNT_DECREASE", "WARN", 0),
    ("UW_SUPABASE_OUTBOX_DEGRADED", "WARN", 20),
    ("CLIO-UW-1 STALE", "WARN", 5),
    ("UW_ACCOUNT_QUOTA_DENIED", "WARN", 3),
    ("Market snapshot failed", "WARN", 5),
    ("UW quota coordinator unavailable", "WARN", 0),
    # dynamic pillar weights (agent/dpof/dynamic_weights.py + the data.py hooks): every failure path ends in this suffix
    ("); static weights", "WARN", 0),
    ("dynamic weights disabled", "WARN", 0),
)

# A UW-sourced pillar reading counts as "not flat" when |value| >= this. A live premium score is a ratio over the last 20
# one-minute ticks, so exactly ~0.0 for a whole session is a defect (it read 0.0 on 6,476 cycles before 2026-10-09).
PILLAR_FLAT_EPS = 0.005
PILLAR_MIN_ROWS = 60       # UW-sourced readings needed before a pillar is judged (60 cycles = 30 minutes)


def _run(cmd: list[str], timeout: int = 30) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Report:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def add(self, level: str, text: str) -> None:
        self.lines.append((level, text))

    def counts(self) -> dict:
        return dict(collections.Counter(level for level, _ in self.lines))

    def render(self) -> str:
        out = [f"[{level}] {text}" for level, text in self.lines]
        c = self.counts()
        out.append(f"SUMMARY: {c.get('PASS', 0)} pass, {c.get('WARN', 0)} warn, {c.get('FAIL', 0)} fail")
        return "\n".join(out)

    @property
    def exit_code(self) -> int:
        return 1 if any(level == "FAIL" for level, _ in self.lines) else 0


# ---------------------------------------------------------------------------- pure evaluators (unit-tested)
def eval_services(states: dict[str, str], restarts: dict[str, str]) -> list[tuple[str, str]]:
    out = []
    for svc, state in states.items():
        out.append(("PASS" if state == "active" else "FAIL", f"service {svc}: {state} (restarts since start: {restarts.get(svc, '?')})"))
    return out


def eval_ledger(rep: dict, market_open_elapsed_min: float) -> list[tuple[str, str]]:
    n = rep["requests"]
    out = [("PASS" if n > 0 else "WARN", f"UW ledger: {n} requests since the open, status {rep['status']}")]
    if n >= 50:   # a rate over a handful of requests is noise
        rate = rep["rate_429"]
        level = "PASS" if rate < 0.02 else ("WARN" if rate < 0.10 else "FAIL")
        out.append((level, f"UW 429 rate {rate:.1%} (target < 2%)"))
        burst = rep["max_same_second_burst"]
        out.append(("PASS" if burst <= 4 else "WARN", f"max same-second burst {burst} (target <= 4)"))
    else:
        out.append(("WARN", f"only {n} requests since the open -- too few to judge the 429 rate"))
    out.append(("PASS", f"retry rows {rep['retries']['rows']} (outcomes {rep['retries']['status']})"))
    cyc = rep.get("cycle") or {}
    if cyc:
        out.append(("PASS", f"cycle {cyc.get('cycle_key')}: entitlement {cyc.get('entitlement')}, observed {cyc.get('observed_count')}, "
                            f"anomalies {cyc.get('anomaly_count')}"))
    return out


def eval_flow_sources(counts: dict[str, int], records: int) -> list[tuple[str, str]]:
    total = sum(counts.values())
    if not total:
        return [("WARN", f"DPOF: no available pillar readings today ({records} records)")]
    # keys are "<pillar>:<source>"; a UW reading's source is "uw:/endpoint...", the fallback's is "selfflow:<SYM>"
    uw = sum(v for k, v in counts.items() if k.split(":", 1)[-1].startswith("uw"))
    share = uw / total
    level = "PASS" if share >= 0.6 else "WARN"
    return [(level, f"DPOF flow pillars from UW: {share:.0%} of {total} readings ({records} records); sources {dict(sorted(counts.items()))}")]


def eval_pillar_liveness(stats: dict) -> list[tuple[str, str]]:
    """Is each UW-sourced premium/darkpool pillar actually moving? `stats[pillar] = {"rows": n, "nonflat": m}`.

    Premium flat on a live session is a FAIL (the 2026-10-09 field-name bug class). The LEGACY darkpool score lives at
    +-0.002 and is expected to be flat until CLIO_DPOF_DARKPOOL_NBBO is decided, so it is a WARN with an explanation."""
    out: list[tuple[str, str]] = []
    for name, ok_at, fail_below in (("premium", 0.80, 0.20), ("darkpool", 0.50, None)):
        s = stats.get(name) or {}
        rows, nonflat = int(s.get("rows", 0)), int(s.get("nonflat", 0))
        if rows < PILLAR_MIN_ROWS:
            out.append(("WARN", f"pillar {name}: only {rows} UW-sourced readings today -- too few to judge liveness"))
            continue
        share = nonflat / rows
        text = f"pillar {name} (UW-sourced): {share:.0%} of {rows} readings are not flat (|v| >= {PILLAR_FLAT_EPS})"
        if share >= ok_at:
            out.append(("PASS", text))
        elif fail_below is not None and share < fail_below:
            out.append(("FAIL", text + " -- looks DEAD (a scoring/field-name defect, not a quiet tape)"))
        elif name == "darkpool":
            out.append(("WARN", text + " -- the legacy darkpool score is flat by construction; see CLIO_DPOF_DARKPOOL_NBBO"))
        else:
            out.append(("WARN", text))
    return out


def eval_dynamic_weights(info: dict) -> list[tuple[str, str]]:
    """Shadow-log health: coverage, dead pillars, and the identity invariant (dyn == static when nothing is dead)."""
    out: list[tuple[str, str]] = []
    shadow, sig = int(info.get("shadow_rows", 0)), int(info.get("signal_rows", 0))
    if shadow == 0:
        out.append(("WARN", f"dynamic weights: no shadow records today ({sig} signal rows) -- mode off, manager disabled, or not restarted yet"))
        return out
    cover = shadow / sig if sig else 1.0
    out.append(("PASS" if cover >= 0.90 else "WARN", f"dynamic weights: {shadow} shadow records vs {sig} signal rows ({cover:.0%} coverage)"))
    viol, eligible = int(info.get("identity_violations", 0)), int(info.get("identity_eligible", 0))
    if eligible:
        if viol:
            out.append(("FAIL", f"dynamic weights: {viol} of {eligible} nothing-dead rows have |dyn - static| > 1e-9 "
                                f"(max {info.get('identity_max_dev')}) -- dynamic must equal static when nothing is dead"))
        else:
            out.append(("PASS", f"dynamic weights: identity holds on {eligible} nothing-dead rows"))
    dead = {u: sorted(d) for u, d in (info.get("dead_last") or {}).items() if d}
    if not dead:
        out.append(("PASS", "dynamic weights: no pillar currently flagged dead"))
    else:
        other = {u: [p for p in d if p != "darkpool"] for u, d in dead.items()}
        if any(other.values()):
            out.append(("WARN", f"dynamic weights: pillars flagged DEAD (weight redistributed in shadow): {dead}"))
        else:
            out.append(("PASS", f"dynamic weights: only the legacy darkpool is flagged dead (expected while CLIO_DPOF_DARKPOOL_NBBO is off): {dead}"))
    return out


def eval_journal(text: str) -> list[tuple[str, str]]:
    out = []
    for pattern, level, tolerated in JOURNAL_PATTERNS:
        n = text.count(pattern)
        if n > tolerated:
            out.append((level, f"journal: {n} x '{pattern}' since the open (tolerated <= {tolerated})"))
        else:
            out.append(("PASS", f"journal: {n} x '{pattern}'"))
    return out


def eval_kronos(age_s: float | None, in_session: bool) -> list[tuple[str, str]]:
    if age_s is None:
        return [("WARN", "Kronos forecast file missing")]
    if not in_session:
        return [("PASS", f"Kronos forecast age {age_s / 60:.0f} min (outside the session)")]
    return [("PASS" if age_s <= 600 else "WARN", f"Kronos forecast age {age_s / 60:.1f} min (target <= 10 during the session)")]


GATEWAY_STALE_S = 12 * 3600   # a good balance sample older than this is a WARN (the monitor samples every 6 h)
_GATEWAY_STATS_RE = re.compile(r"gateway stats: calls=(\d+) empty_retries=(\d+) empty_final=(\d+)")


def _gateway_expiry_text(expires_at, now_epoch: float) -> str:
    try:
        exp = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        days = (exp.timestamp() - now_epoch) / 86400
    except (TypeError, ValueError):
        return "expiry n/a"
    return "expired" if days < 0 else f"expires in {int(days)}d"


def _gateway_balance_line(info: dict, now_epoch: float) -> tuple[str, str]:
    sample = _num(info.get("sample_ts")) if info else None
    if sample is None:
        return ("WARN", "gateway: no balance sample yet (monitor runs every 6 h)")
    bal, burn = _num(info.get("balance_usd")), _num(info.get("burn_per_day"))
    detail = (f"balance ${bal:.2f}" if bal is not None else "balance n/a") + ", "
    detail += (f"burn ${burn:.2f}/day" if burn is not None else "burn n/a") + ", "
    detail += _gateway_expiry_text(info.get("expires_at"), now_epoch)
    level, active = info.get("level"), info.get("is_active")
    if level == "crit" or active is False:
        reasons = info.get("reasons")
        why = f" (reasons: {', '.join(str(r) for r in reasons)})" if isinstance(reasons, list) and reasons else ""
        return ("FAIL", f"gateway: {detail}, level {level}, is_active {active}{why}")
    age = now_epoch - sample
    if level != "ok" or age >= GATEWAY_STALE_S:
        why = f"last good sample {age / 3600:.1f} h ago" if age >= GATEWAY_STALE_S else f"level {level}"
        return ("WARN", f"gateway: {detail} -- {why}")
    return ("PASS", f"gateway: {detail}")


def _gateway_canary_line(info: dict) -> tuple[str, str]:
    canary = info.get("canary") if info else None
    if not isinstance(canary, dict):
        return ("WARN", "gateway canary: missing from balance state")
    results = [r for r in (canary.get("results") or []) if isinstance(r, dict)]
    pairs = "; ".join(f"{r.get('requested')} -> {r.get('served')}" for r in results)
    if not results or canary.get("family_changed") is True or canary.get("ok") is not True:
        return ("WARN", f"gateway canary: {pairs or 'no results'} (family_changed={canary.get('family_changed')}, ok={canary.get('ok')})")
    return ("PASS", f"gateway canary: {pairs} (version alias, same family)")


def _gateway_p95(line: str) -> int | None:
    m = re.search(r"latency_ms_p50/p95 by model=(\S+)", line)
    if not m:
        return None
    vals = [int(pm.group(2)) for part in m.group(1).split(",") if (pm := re.search(r"=(\d+)/(\d+)$", part))]
    return max(vals) if vals else None


def _gateway_stats_line(journal_text: str, in_session: bool = False) -> tuple[str, str]:
    lines = [ln for ln in (journal_text or "").splitlines() if "gateway stats:" in ln]
    if not lines:
        # The hourly transport line is call-driven: a silent window during the session is a finding, not a pass (R-12).
        if in_session:
            return ("WARN", "gateway stats: no gateway stats line in the last 2 h -- transport logging silent or no calls")
        return ("PASS", "gateway stats: none in the last 2 h (outside the session)")
    last = lines[-1]
    m = _GATEWAY_STATS_RE.search(last)
    if not m:
        return ("WARN", "gateway stats: unparseable line -- transport logging may have changed")
    calls, retries, empty_final = (int(x) for x in m.groups())
    if empty_final > 0:
        return ("WARN", f"gateway stats: empty_final={empty_final} of calls={calls} (empty_retries={retries}) -- empty replies reached callers")
    p95 = _gateway_p95(last)
    return ("PASS", f"gateway stats: calls={calls} empty_retries={retries} p95 {f'{p95}ms' if p95 is not None else 'n/a'}")


def _in_session_at(epoch: float) -> bool:
    t = datetime.fromtimestamp(epoch, NY)
    return 9 * 60 + 30 <= t.hour * 60 + t.minute <= 16 * 60


def eval_gateway(info: dict, now_epoch: float, journal_text: str = "", in_session: bool | None = None) -> list[tuple[str, str]]:
    """OneProvider gateway: balance/expiry from the monitor's state file, the served-model canary, and the transport-stats
    journal line from the last 2 h (`journal_text`). A missing state file (info == {}) is a WARN, never a PASS. A missing
    stats line is a WARN while the session is open (in_session defaults to the 09:30-16:00 ET clock of now_epoch)."""
    if in_session is None:
        in_session = _in_session_at(now_epoch)
    return [_gateway_balance_line(info or {}, now_epoch), _gateway_canary_line(info or {}), _gateway_stats_line(journal_text, in_session)]


# ---------------------------------------------------------------------------- R-12 (S215) pure evaluators
DECIDER_FALLBACK_WARN, DECIDER_FALLBACK_FAIL = 0.15, 0.40   # share of entry/review deliberations that used the fallback
# Unsigned (empty hmac_signature) audit rows already on record, per audit-day file. Legacy history is unsigned, so the
# check is per day: a day's count above its known value is a WARN. Unknown days are expected to be 0.
AUDIT_UNSIGNED_KNOWN = {"2026-09-27": 3, "2026-09-29": 93, "2026-09-30": 3, "2026-10-10": 12, "2026-10-11": 18, "2026-10-12": 4}
AUDIT_DAY_RE = re.compile(r"audit_(\d{4}-\d{2}-\d{2})\.jsonl\Z")
ALIGN_MAX_LAG = timedelta(days=1.5)                           # signals history may start this much AFTER the shadow log's first row
DISK_WARN_FREE, DISK_FAIL_FREE = 0.20, 0.10
BACKUP_TOTAL_MAX = 25 * 10**9
GATEWAY_STATS_WINDOW = "2 hours ago"


def _parse_ts(value) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return d.replace(tzinfo=NY) if d.tzinfo is None else d


def _p95(values: list[float]) -> float:
    s = sorted(values)
    return s[max(0, -(-95 * len(s) // 100) - 1)]   # nearest-rank: ceil(0.95 n) - 1


def eval_decider_attempts(rows: list[dict] | None) -> list[tuple[str, str]]:
    """Today's decider attempt ledger. `rows` is None when the file is absent (INFO, never a verdict)."""
    if rows is None:
        return [("INFO", "decider attempt ledger not present (data/dpof/decider_attempts.jsonl)")]
    cv = sum(len(r.get("claim_violations") or []) for r in rows if isinstance(r.get("claim_violations"), list))
    out: list[tuple[str, str]] = [("INFO", f"decider claim violations today: {cv}")]
    delib = [r for r in rows if r.get("kind") in ("entry", "review")]
    if not delib:
        out.append(("INFO", f"decider: no entry/review deliberations today ({len(rows)} attempt rows)"))
        return out
    # One PRIMARY row per deliberation; a deliberation that falls back adds a second row with fallback_used true.
    primary = [r for r in delib if r.get("fallback_used") is not True]
    fb = len(delib) - len(primary)
    if not primary:
        out.append(("INFO", f"decider: {fb} fallback rows but no primary rows today"))
        return out
    share = fb / len(primary)
    level = "FAIL" if share > DECIDER_FALLBACK_FAIL else ("WARN" if share > DECIDER_FALLBACK_WARN else "PASS")
    out.append((level, f"decider fallback: {fb} of {len(primary)} deliberations used the fallback ({share:.0%}; warn > 15%, fail > 40%)"))
    lat = [float(r["latency_ms"]) for r in primary if _num(r.get("latency_ms")) is not None]
    timeouts = [_num(r.get("timeout_s")) for r in primary if _num(r.get("timeout_s")) is not None]
    if lat and timeouts:
        p95_s, t = _p95(lat) / 1000, max(timeouts)
        level = "WARN" if p95_s > 0.8 * t else "PASS"
        out.append((level, f"decider primary p95 latency {p95_s:.1f} s vs timeout {t:.0f} s (warn > 0.8x)"))
    return out


def eval_audit_unsigned(counts_by_day: dict[str, int] | None) -> list[tuple[str, str]]:
    """Unsigned audit rows per audit-day file. Only the LAST TWO days are judged; a day above AUDIT_UNSIGNED_KNOWN.get(day, 0)
    is a WARN, otherwise PASS. None or empty = no audit-day files (INFO, never a verdict)."""
    if not counts_by_day:
        return [("INFO", "audit chain: no data/audit/audit_YYYY-MM-DD.jsonl files present")]
    out: list[tuple[str, str]] = []
    for day in sorted(counts_by_day)[-2:]:
        n, known = counts_by_day[day], AUDIT_UNSIGNED_KNOWN.get(day, 0)
        if n > known:
            out.append(("WARN", f"audit chain {day}: {n} unsigned rows vs known {known} -- new unsigned audit rows"))
        else:
            out.append(("PASS", f"audit chain {day}: {n} unsigned rows (known {known})"))
    return out


def eval_history_alignment(signals_first_ts, shadow_first_ts) -> list[tuple[str, str]]:
    """The replay gate needs the signals history to reach back at least as far as the shadow log. WARN when signals
    starts more than ALIGN_MAX_LAG AFTER the shadow log's first row; starting earlier is the healthy direction."""
    s, h = _parse_ts(signals_first_ts), _parse_ts(shadow_first_ts)
    if s is None or h is None:
        return [("INFO", "history alignment: signals or shadow log not present (or first line unreadable)")]
    lag = s - h
    if lag > ALIGN_MAX_LAG:
        return [("WARN", f"history alignment: signals starts {lag.total_seconds() / 86400:.1f} d after the shadow log: "
                         "signals history shorter than shadow log: replay gate at risk")]
    return [("PASS", f"history alignment: signals from {s:%Y-%m-%d}, shadow from {h:%Y-%m-%d}")]


def eval_disk(usage) -> list[tuple[str, str]]:
    """usage = (total, used, free) in bytes, as from shutil.disk_usage. Below 20% free warns; below 10% fails."""
    total, free = usage[0], usage[2]
    if not total:
        return [("INFO", "disk: capacity unknown")]
    frac = free / total
    level = "FAIL" if frac < DISK_FAIL_FREE else ("WARN" if frac < DISK_WARN_FREE else "PASS")
    return [(level, f"disk /home/agent: {frac:.0%} free ({free / 1e9:,.1f} GB of {total / 1e9:,.1f} GB)")]


def eval_backup_dedup(pair, total_bytes: int | None) -> list[tuple[str, str]]:
    """pair = ((st_ino, st_size), (st_ino, st_size)) of retrieval_shadow.jsonl in the two newest snapshots, or None.
    Same size on a different inode means rsync --link-dest did not hard-link the unchanged file."""
    out: list[tuple[str, str]] = []
    if pair is None:
        out.append(("INFO", "backup dedup: fewer than two snapshots carry retrieval_shadow.jsonl -- nothing to compare"))
    else:
        (i0, s0), (i1, s1) = pair
        if i0 != i1 and s0 == s1:
            out.append(("WARN", f"backup dedup: retrieval_shadow.jsonl is {s0} B in both newest snapshots on different inodes -- link-dest not applied"))
        else:
            out.append(("PASS", "backup dedup: unchanged file is hard-linked across the two newest snapshots (or it changed)"))
    if total_bytes is None:
        out.append(("INFO", "backup size: du unavailable"))
    else:
        level = "WARN" if total_bytes > BACKUP_TOTAL_MAX else "PASS"
        out.append((level, f"backup size: ~/backups apparent size {total_bytes / 1e9:.1f} GB (warn > 25 GB)"))
    return out


# ---------------------------------------------------------------------------- collectors
def _service_states() -> tuple[dict, dict]:
    states, restarts = {}, {}
    for svc in SERVICES:
        states[svc] = _run(["systemctl", "is-active", svc]).strip() or "unknown"
        for line in _run(["systemctl", "show", svc, "-p", "NRestarts"]).splitlines():
            if line.startswith("NRestarts="):
                restarts[svc] = line.split("=", 1)[1]
    return states, restarts


def _flow_sources(today: str) -> tuple[dict, int]:
    counts: collections.Counter = collections.Counter()
    records = 0
    path = ROOT / "data" / "dpof" / "signals.jsonl"
    if not path.is_file():
        return {}, 0
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            if not str(r.get("ts", "")).startswith(today):
                continue
            records += 1
            for name, p in (r.get("pillars") or {}).items():
                if isinstance(p, dict) and p.get("available"):
                    counts[f"{name}:{p.get('source')}"] += 1
    return dict(counts), records


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf") else None


def _pillar_liveness(today: str) -> dict:
    """Today's UW-sourced premium/darkpool readings from signals.jsonl: rows and how many are not flat."""
    stats = {"premium": {"rows": 0, "nonflat": 0}, "darkpool": {"rows": 0, "nonflat": 0}}
    path = ROOT / "data" / "dpof" / "signals.jsonl"
    if not path.is_file():
        return stats
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            if not str(r.get("ts", "")).startswith(today):
                continue
            for name in stats:
                p = (r.get("pillars") or {}).get(name)
                if not isinstance(p, dict) or not p.get("available") or not str(p.get("source") or "").startswith("uw"):
                    continue
                v = _num(p.get("value"))
                if v is None:
                    continue
                stats[name]["rows"] += 1
                stats[name]["nonflat"] += 1 if abs(v) >= PILLAR_FLAT_EPS else 0
    return stats


def _dynamic_weights_info(today: str) -> dict:
    """Shadow-log summary for today (read-only): coverage, the identity invariant, and each underlying's last dead list."""
    info = {"shadow_rows": 0, "signal_rows": 0, "identity_violations": 0, "identity_eligible": 0, "identity_max_dev": 0.0, "dead_last": {}}
    sig = ROOT / "data" / "dpof" / "signals.jsonl"
    if sig.is_file():
        with open(sig, "rb") as fh:
            for raw in fh:
                try:
                    if str(json.loads(raw).get("ts", "")).startswith(today):
                        info["signal_rows"] += 1
                except ValueError:
                    continue
    shadow = ROOT / "data" / "dpof" / "pillar_weights_shadow.jsonl"
    if not shadow.is_file():
        return info
    with open(shadow, "rb") as fh:
        for raw in fh:
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            if not str(r.get("ts", "")).startswith(today):
                continue
            info["shadow_rows"] += 1
            info["dead_last"][str(r.get("u"))] = list(r.get("dead") or [])
            # "ma" is the multiplier actually applied (1.0 until quality weighting is switched on); "m" is the logged
            # diagnostic and drifts off 1.0 as samples accrue, which would make the identity check vacuous. Old rows have only "m".
            alive = r.get("alive") or {}
            mult = r.get("ma") if isinstance(r.get("ma"), dict) else (r.get("m") or {})
            static, dyn = _num(r.get("static")), _num(r.get("dyn"))
            if (not r.get("dead") and static is not None and dyn is not None
                    and all(_num(a) == 1.0 for a in alive.values()) and all(_num(m) == 1.0 for m in mult.values())):
                info["identity_eligible"] += 1
                dev = abs(dyn - static)
                if dev > 1e-9:
                    info["identity_violations"] += 1
                    info["identity_max_dev"] = max(info["identity_max_dev"], dev)
    return info


def _gateway_info() -> dict:
    """The OneProvider balance monitor's state file (read-only). Missing, unreadable or non-object -> {}; never raises."""
    try:
        data = json.loads((ROOT / "data" / "oneprovider_balance_state.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an absent or corrupt state file is reported by eval_gateway, not raised
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------- R-12 (S215) read-only collectors
_SNAP_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_SNAP_HOUR_RE = re.compile(r"\d{2}\Z")
RETRIEVAL_SNAPSHOT_REL = Path("openclaw_data/personality/retrieval_shadow.jsonl")
BACKUPS_DIR = Path("/home/agent/backups")
DISK_PATH = "/home/agent"


def _decider_rows(today: str) -> list[dict] | None:
    """Today's rows of data/dpof/decider_attempts.jsonl; None when the ledger is absent. Corrupt lines are skipped."""
    path = ROOT / "data" / "dpof" / "decider_attempts.jsonl"
    if not path.is_file():
        return None
    rows: list[dict] = []
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            if isinstance(r, dict) and str(r.get("ts", "")).startswith(today):
                rows.append(r)
    return rows


def _audit_unsigned_by_day() -> dict[str, int] | None:
    """Unsigned (missing or empty hmac_signature) row counts for the LAST TWO audit-day files; None when there are none."""
    try:
        found = sorted(
            ((m.group(1), p) for p in (ROOT / "data" / "audit").iterdir() if (m := AUDIT_DAY_RE.search(p.name))),
            key=lambda t: t[0],
        )
    except OSError:
        return None
    counts: dict[str, int] = {}
    for day, path in found[-2:]:
        n = 0
        try:
            with open(path, "rb") as fh:
                for raw in fh:
                    try:
                        r = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(r, dict) and not r.get("hmac_signature"):
                        n += 1
        except OSError:
            continue
        counts[day] = n
    return counts or None


def _first_ts(path: Path) -> str | None:
    """The ts of the FIRST line only (the oldest row of an append-only log). None if absent or unreadable."""
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as fh:
            first = fh.readline()
        r = json.loads(first)
    except (OSError, ValueError):
        return None
    return str(r["ts"]) if isinstance(r, dict) and r.get("ts") else None


def _last_two_snapshots(base: Path) -> list[Path]:
    """The two newest ~/backups/YYYY-MM-DD/HH snapshot dirs, oldest first. Other names are ignored."""
    if not base.is_dir():
        return []
    snaps = []
    for d in base.iterdir():
        if d.is_dir() and _SNAP_DATE_RE.match(d.name):
            for h in d.iterdir():
                if h.is_dir() and _SNAP_HOUR_RE.match(h.name):
                    snaps.append((d.name, h.name, h))
    snaps.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in snaps[-2:]]


def _backup_dedup_inputs() -> tuple:
    """(pair, total_bytes): pair = (st_ino, st_size) of retrieval_shadow.jsonl in the two newest snapshots, or None;
    total_bytes = apparent size of ~/backups from `du -sb`, or None."""
    snaps = _last_two_snapshots(BACKUPS_DIR)
    pair = None
    if len(snaps) == 2:
        stats = []
        for s in snaps:
            try:
                st = os.stat(s / RETRIEVAL_SNAPSHOT_REL)
            except OSError:
                break
            stats.append((st.st_ino, st.st_size))
        else:
            pair = (stats[0], stats[1])
    out = _run(["du", "-sb", str(BACKUPS_DIR)], timeout=120).split()
    total = int(out[0]) if out and out[0].isdigit() else None
    return pair, total


def _disk_usage():
    return shutil.disk_usage(DISK_PATH)


def main() -> int:
    now_et = datetime.now(NY)
    today = now_et.date().isoformat()
    open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    in_session = 9 * 60 + 30 <= now_et.hour * 60 + now_et.minute <= 16 * 60
    rpt = Report()
    print(f"CLIO morning check  {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}  (root {ROOT})", flush=True)

    try:
        states, restarts = _service_states()
        for lv, tx in eval_services(states, restarts):
            rpt.add(lv, tx)
    except Exception as e:  # noqa: BLE001 - a broken check must not hide the others
        rpt.add("WARN", f"service check failed: {type(e).__name__}: {e}")

    try:
        report_mod = _load_module("uw_ledger_report", ROOT / "scripts" / "uw_ledger_report.py")
        since = open_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        until = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        db_path = os.environ.get("CLIO_UW_LEDGER_DB", report_mod.DEFAULT_DB)
        con = report_mod.open_ro(db_path)
        try:
            rep = report_mod.report(con, since, until)
        finally:
            con.close()
        for lv, tx in eval_ledger(rep, (now_et - open_et).total_seconds() / 60):
            rpt.add(lv, tx)
        rpt.add("PASS", f"ledger file {os.path.getsize(db_path) / 1e6:,.0f} MB")
    except Exception as e:  # noqa: BLE001
        rpt.add("WARN", f"ledger report failed: {type(e).__name__}: {e}")

    try:
        counts, records = _flow_sources(today)
        for lv, tx in eval_flow_sources(counts, records):
            rpt.add(lv, tx)
    except Exception as e:  # noqa: BLE001
        rpt.add("WARN", f"DPOF flow-source check failed: {type(e).__name__}: {e}")

    try:
        for lv, tx in eval_pillar_liveness(_pillar_liveness(today)):
            rpt.add(lv, tx)
    except Exception as e:  # noqa: BLE001
        rpt.add("WARN", f"pillar liveness check failed: {type(e).__name__}: {e}")

    try:
        for lv, tx in eval_dynamic_weights(_dynamic_weights_info(today)):
            rpt.add(lv, tx)
    except Exception as e:  # noqa: BLE001
        rpt.add("WARN", f"dynamic weights check failed: {type(e).__name__}: {e}")

    try:
        journal_text = _run(["journalctl", "-u", "moltbot", "--since", open_et.strftime("%Y-%m-%d %H:%M:%S"), "--no-pager"], timeout=60)
        for lv, tx in eval_journal(journal_text):
            rpt.add(lv, tx)
    except Exception as e:  # noqa: BLE001
        rpt.add("WARN", f"journal check failed: {type(e).__name__}: {e}")

    try:   # the gateway stats window is the last 2 h, not the session: the hourly line is call-driven (R-12)
        gw_journal = _run(["journalctl", "-u", "moltbot", "--since", GATEWAY_STATS_WINDOW, "--no-pager"], timeout=60)
        for lv, tx in eval_gateway(_gateway_info(), time.time(), gw_journal, in_session):
            rpt.add(lv, tx)
    except Exception as e:  # noqa: BLE001
        rpt.add("WARN", f"gateway check failed: {type(e).__name__}: {e}")

    try:
        kf = ROOT / "data" / "intelligence" / "kronos_forecast.json"
        age = (time.time() - kf.stat().st_mtime) if kf.is_file() else None
        for lv, tx in eval_kronos(age, in_session):
            rpt.add(lv, tx)
    except Exception as e:  # noqa: BLE001
        rpt.add("WARN", f"kronos check failed: {type(e).__name__}: {e}")

    try:
        outbox = ROOT / "data" / "decisions" / "outbox.jsonl"
        n = sum(1 for _ in open(outbox, "rb")) if outbox.is_file() else 0
        rpt.add("PASS" if n == 0 else "WARN", f"decision outbox backlog: {n} line(s) (non-zero = Supabase writes failing)")
    except Exception as e:  # noqa: BLE001
        rpt.add("WARN", f"decision outbox check failed: {type(e).__name__}: {e}")

    try:
        size = MEMORY_MD.stat().st_size
        rpt.add("PASS" if size <= 20000 else "WARN", f"MEMORY.md {size:,} B (budget 20,000)")
    except Exception as e:  # noqa: BLE001
        rpt.add("WARN", f"MEMORY.md check failed: {type(e).__name__}: {e}")

    checks = (
        ("decider check", lambda: eval_decider_attempts(_decider_rows(today))),
        ("audit check", lambda: eval_audit_unsigned(_audit_unsigned_by_day())),
        ("history alignment check", lambda: eval_history_alignment(
            _first_ts(ROOT / "data" / "dpof" / "signals.jsonl"),
            _first_ts(ROOT / "data" / "dpof" / "pillar_weights_shadow.jsonl"))),
        ("disk check", lambda: eval_disk(_disk_usage())),
        ("backup dedup check", lambda: eval_backup_dedup(*_backup_dedup_inputs())),
    )
    for label, run_check in checks:   # each isolated: a broken check must not hide the others (R-12)
        try:
            for lv, tx in run_check():
                rpt.add(lv, tx)
        except Exception as e:  # noqa: BLE001
            rpt.add("WARN", f"{label} failed: {type(e).__name__}: {e}")

    print(rpt.render(), flush=True)
    return rpt.exit_code


if __name__ == "__main__":
    sys.exit(main())
