#!/usr/bin/env python3
"""
CLIO DPOF — offline signal study (F3, `docs/design/clio_dpof_followups_2026-09-26.md` §5).

Measures what the underlying did 5/15/30/60 minutes after DPOF composite
threshold crossings recorded in `data/dpof/signals.jsonl`, against a
long-only baseline (market drift) computed over the same records —
descriptive statistics only (n reported next to every rate, PF-style "no
p-values on n<30", Aidan's framing in §7). NEVER reads, computes, or feeds
back into any gate/threshold/size/order/decider/Jev call — this is a
read-only research tool the operator runs by hand.

STANDALONE (operator hard constraint, §5/§7): never imported by `agent/`
— enforced by a grep test in `tests/test_dpof_signal_study.py`. Writes
nothing unless `--save-bars`/`--json-out` is given. Importing this module
performs no I/O and no network — the pure core (`load_signal_records`,
`find_events`, `price_at`, `forward_returns`, `summarize`, `render_report`)
is stdlib-only (no pandas, no numpy) and is exercised by the tests entirely
through `--bars-file` (offline) fixtures. `alpaca` and `agent.*` are
imported LAZILY, only on the live Alpaca-fetch path (no `--bars-file`):
CLAUDE.md anti-pattern #34 — `load_dotenv(<repo>/.env)` runs BEFORE any
`agent.*` import on that path, since `agent.dpof.config.load_dpof_config()`
re-reads `os.environ` on every call and a CLI invocation with no ambient
`.env` load would otherwise run silently keyless.

Event definition (R6, design-frame-ruled): per (underlying, sign(composite))
— the first record with |composite| >= threshold whose previous record for
that underlying was below threshold, on the opposite side, or more than 5
minutes earlier (a new session/restart) — then a `--min-gap-minutes`
refractory (default 15, 0 = none) before a new event fires in the same
(underlying, direction). This keeps events near-independent; composites
oscillate around a threshold at ~30s cadence, so pure first-crossing alone
would cluster events whose forward windows overlap.

Usage:
  python3 scripts/dpof_signal_study.py
  python3 scripts/dpof_signal_study.py --bars-file bars.jsonl --thresholds 0.15,0.30
  python3 scripts/dpof_signal_study.py --save-bars bars_cache.jsonl --json-out out.json
  python3 scripts/dpof_signal_study.py --signals /path/to/signals.jsonl \\
      --horizons 5,15,30,60 --min-gap-minutes 15 --since 2026-09-01 --until 2026-09-25
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

# scripts/dpof_signal_study.py -> 1x dirname = repo root (matches
# scripts/reconcile_orphan_entries.py's convention). This is a sys.path
# mutation only, never an import of agent.* — the module stays importable
# with zero third-party deps.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_DATA_DIR = os.environ.get("CLIO_DPOF_DATA_DIR", os.path.join(_REPO_ROOT, "data", "dpof"))
DEFAULT_SIGNALS_FILE = os.path.join(DEFAULT_DATA_DIR, "signals.jsonl")
DEFAULT_THRESHOLDS = (0.15, 0.30)
DEFAULT_HORIZONS = (5, 15, 30, 60)
DEFAULT_MIN_GAP_MINUTES = 15.0

_BAR_SECONDS = 60.0            # a 1-min bar's timestamp is its START (Alpaca convention)
_GAP_RESET_SECONDS = 5 * 60.0  # >5min between consecutive records for one underlying
                                # = a new session/restart, not a continuous crossing state
_MIN_DESCRIPTIVE_N = 30         # below this, a summary row is descriptive-only; no p-values


# ─────────────────────────────────────────────────────────────────────────
# Pure core — stdlib only, no I/O. Independently unit-testable.
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Event:
    """One threshold-crossing event for one underlying."""
    ts: float                      # epoch seconds
    underlying: str
    direction: int                 # +1 or -1 (sign of composite at the crossing)
    composite: float
    strength: object
    price_ok: Optional[bool]       # signals.jsonl's price_confirm.ok
    regime: object
    flow_source: object
    nominated: object


def _parse_ts(ts) -> Optional[float]:
    """epoch seconds from signals.jsonl's tz-aware ET ISO `ts` string, or an
    already-numeric epoch passed straight through. Never raises — a
    malformed/missing timestamp returns None so callers can skip the row
    (never treated as epoch 0)."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        dt = datetime.fromisoformat(str(ts))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _epoch_of(rec: dict) -> Optional[float]:
    """`_epoch` if load_signal_records already normalized this record,
    otherwise parses `ts` on the fly — keeps find_events usable directly on
    hand-built fixture dicts in tests, not only on load_signal_records
    output."""
    e = rec.get("_epoch")
    if e is not None:
        return e
    return _parse_ts(rec.get("ts"))


def _as_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _price_confirm_ok(rec: dict) -> Optional[bool]:
    pc = rec.get("price_confirm")
    if isinstance(pc, dict):
        ok = pc.get("ok")
        return ok if isinstance(ok, bool) else None
    return None


def load_signal_records(lines) -> list[dict]:
    """Parses an iterable of signals.jsonl lines (per engine.py:2366-2387's
    record shape) into records. Skips lines that aren't valid JSON objects,
    and records whose `composite` is None/missing or whose `ts` doesn't
    parse (composite None is a genuinely unmeasured cycle, never coerced to
    0 — CLAUDE.md 'no fake zeros'). Adds a normalized `_epoch` float field
    alongside the original `ts` string; never raises."""
    out: list[dict] = []
    for line in lines:
        line = line.strip() if isinstance(line, str) else line
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(rec, dict):
            continue
        composite = _as_float(rec.get("composite"))
        if composite is None:
            continue
        epoch = _parse_ts(rec.get("ts"))
        if epoch is None:
            continue
        rec = dict(rec)
        rec["composite"] = composite
        rec["_epoch"] = epoch
        out.append(rec)
    return out


def _sign(x: float) -> int:
    return 1 if x >= 0 else -1


def find_events(records: list[dict], threshold: float, min_gap_minutes: float) -> list[Event]:
    """Per (underlying, sign(composite)): an event at the first record with
    |composite| >= threshold whose previous record for that underlying had
    |composite| < threshold, the opposite sign, or was more than 5 minutes
    earlier (a new session/restart resets the crossing state — there is no
    market calendar here, so a >5min gap is the only signal a continuity
    tracker gets). A symbol's very first record is treated the same as a
    gap reset (nothing to compare against). After a crossing fires, no new
    event fires in the same (underlying, direction) within `min_gap_minutes`
    (0 = none, i.e. pure first-crossing) — this keeps events
    near-independent given signals.jsonl's ~30s cadence.

    Tolerant of records that were NOT pre-normalized by load_signal_records
    (parses `ts`/`composite` itself); still skips composite-None / bad-ts
    rows defensively. Input need not be pre-sorted. Output is chronological
    (interleaved across underlyings by ts)."""
    rows: list[tuple[float, dict, float]] = []
    for rec in records:
        composite = _as_float(rec.get("composite"))
        if composite is None:
            continue
        epoch = _epoch_of(rec)
        if epoch is None:
            continue
        if not rec.get("underlying"):
            continue
        rows.append((epoch, rec, composite))
    rows.sort(key=lambda r: r[0])

    prev: dict[str, tuple[float, float]] = {}       # underlying -> (epoch, composite)
    last_event: dict[tuple[str, int], float] = {}   # (underlying, direction) -> epoch
    events: list[Event] = []
    gap_s = float(min_gap_minutes) * 60.0

    for epoch, rec, composite in rows:
        underlying = rec["underlying"]
        direction = _sign(composite)

        is_crossing = False
        if abs(composite) >= threshold:
            prior = prev.get(underlying)
            if prior is None:
                is_crossing = True
            else:
                prior_epoch, prior_composite = prior
                if (
                    abs(prior_composite) < threshold
                    or _sign(prior_composite) != direction
                    or (epoch - prior_epoch) > _GAP_RESET_SECONDS
                ):
                    is_crossing = True

        if is_crossing:
            key = (underlying, direction)
            last = last_event.get(key)
            if last is None or (epoch - last) >= gap_s:
                events.append(Event(
                    ts=epoch,
                    underlying=underlying,
                    direction=direction,
                    composite=composite,
                    strength=rec.get("strength"),
                    price_ok=_price_confirm_ok(rec),
                    regime=rec.get("regime"),
                    flow_source=rec.get("flow_source"),
                    nominated=rec.get("nominated"),
                ))
                last_event[key] = epoch

        prev[underlying] = (epoch, composite)

    return events


def price_at(bars: list[dict], epoch: float, max_staleness_s: float = 120.0) -> Optional[float]:
    """Close of the latest 1-min bar whose END (`t + 60`) <= epoch — no
    look-ahead. Returns None if no such bar exists, or if the latest
    qualifying bar ended more than `max_staleness_s` before `epoch` (covers
    the session close, half-days, and data gaps without needing a market
    calendar). `bars` is one underlying's bar list (each `{"t": <epoch of
    bar START>, "c": <close>}`, per engine.py's broker.get_minute_bars /
    the --bars-file format); never raises."""
    best_t: Optional[float] = None
    best_c: Optional[float] = None
    for b in bars or []:
        t = _as_float(b.get("t"))
        c = _as_float(b.get("c"))
        if t is None or c is None:
            continue
        end = t + _BAR_SECONDS
        if end <= epoch and (best_t is None or t > best_t):
            best_t, best_c = t, c
    if best_t is None:
        return None
    if (epoch - (best_t + _BAR_SECONDS)) > max_staleness_s:
        return None
    return best_c


def forward_returns(
    events: list[Event],
    bars_by_symbol: dict,
    horizons: tuple = DEFAULT_HORIZONS,
) -> list[dict]:
    """One row per (event, horizon): `r_bps = direction * (p(t+h) - p(t)) /
    p(t) * 10000`. `r_bps` is None ('no_data') whenever either price is
    unavailable (missing bars, staleness, or a zero p0) — never coerced to
    0.0, per CLAUDE.md's 'no fake zeros' (an available 0.0 here would
    silently dilute every downstream hit-rate/mean)."""
    rows: list[dict] = []
    for ev in events:
        bars = bars_by_symbol.get(ev.underlying, [])
        p0 = price_at(bars, ev.ts)
        for h in horizons:
            r_bps: Optional[float] = None
            if p0 is not None and p0 != 0:
                p1 = price_at(bars, ev.ts + h * 60)
                if p1 is not None:
                    r_bps = ev.direction * (p1 - p0) / p0 * 10000.0
            rows.append({
                "ts": ev.ts,
                "underlying": ev.underlying,
                "direction": ev.direction,
                "composite": ev.composite,
                "strength": ev.strength,
                "price_ok": ev.price_ok,
                "regime": ev.regime,
                "flow_source": ev.flow_source,
                "nominated": ev.nominated,
                "horizon": h,
                "r_bps": r_bps,
            })
    return rows


def summarize(rows: list[dict], group_by: Callable[[dict], tuple]) -> list[dict]:
    """Aggregates forward_returns-shaped rows per `group_by(row)` key: `n`
    (rows with a real r_bps), `no_data` (rows where r_bps was None —
    counted, never folded into n as a 0), `hit_rate` (share of n with
    r_bps > 0), `mean_bps`, `median_bps` (None when n == 0). Groups with
    n < 30 carry `note = "descriptive only (n<30)"` — no p-values, ever, at
    any n. Deterministic output order (sorted by stringified key)."""
    groups: dict[tuple, dict] = {}
    for row in rows:
        key = group_by(row)
        g = groups.setdefault(key, {"values": [], "no_data": 0})
        r = row.get("r_bps")
        if r is None:
            g["no_data"] += 1
        else:
            g["values"].append(r)

    out: list[dict] = []
    for key, g in groups.items():
        vals = g["values"]
        n = len(vals)
        if n:
            hit_rate = sum(1 for v in vals if v > 0) / n
            mean_bps = sum(vals) / n
            median_bps = statistics.median(vals)
        else:
            hit_rate = None
            mean_bps = None
            median_bps = None
        out.append({
            "key": key,
            "n": n,
            "no_data": g["no_data"],
            "hit_rate": hit_rate,
            "mean_bps": mean_bps,
            "median_bps": median_bps,
            "note": f"descriptive only (n<{_MIN_DESCRIPTIVE_N})" if n < _MIN_DESCRIPTIVE_N else None,
        })
    out.sort(key=lambda o: tuple(str(k) for k in o["key"]))
    return out


def _price_ok_bucket(price_ok: Optional[bool]) -> str:
    if price_ok is True:
        return "confirmed"
    if price_ok is False:
        return "not_confirmed"
    return "unknown"


def summarize_signal(rows: list[dict]) -> list[dict]:
    """summarize() grouped per (threshold, horizon, price_ok bucket,
    underlying) — `rows` must already carry a `threshold` field (the CLI
    orchestration tags it per find_events() pass)."""
    out = summarize(
        rows,
        group_by=lambda r: (r["threshold"], r["horizon"], _price_ok_bucket(r.get("price_ok")), r["underlying"]),
    )
    for o in out:
        threshold, horizon, price_ok_bucket, underlying = o.pop("key")
        o["threshold"] = threshold
        o["horizon"] = horizon
        o["price_ok"] = price_ok_bucket
        o["underlying"] = underlying
    return out


def summarize_baseline(rows: list[dict]) -> list[dict]:
    """summarize() grouped per (underlying, horizon) only — the long-only
    market-drift reference, independent of any threshold. Every ~30 s
    record is a sample, so forward windows overlap heavily: its n is a
    sample count, not a count of independent observations, and is not
    comparable to the refractory-gated signal n."""
    out = summarize(rows, group_by=lambda r: (r["underlying"], r["horizon"]))
    for o in out:
        underlying, horizon = o.pop("key")
        o["underlying"] = underlying
        o["horizon"] = horizon
    return out


_ET = ZoneInfo("America/New_York")


def _iso(epoch: float) -> str:
    """ET ISO timestamp -- the zone signals.jsonl `ts`, --since/--until and
    the operator all use."""
    return datetime.fromtimestamp(epoch, tz=_ET).isoformat()


def build_report(
    records: list[dict],
    *,
    thresholds: tuple = DEFAULT_THRESHOLDS,
    horizons: tuple = DEFAULT_HORIZONS,
    min_gap_minutes: float = DEFAULT_MIN_GAP_MINUTES,
    bars_by_symbol: dict,
) -> dict:
    """Orchestrates find_events + forward_returns + summarize for every
    threshold, plus one long-only baseline pass over ALL records (market
    drift reference) — the whole thing is pure given `records` +
    `bars_by_symbol` (no I/O). Returns a JSON-serializable dict; see
    render_report() for the plain-text rendering."""
    signal_rows: list[dict] = []
    events_count: dict[float, int] = {}
    for threshold in thresholds:
        events = find_events(records, threshold=threshold, min_gap_minutes=min_gap_minutes)
        events_count[threshold] = len(events)
        rows = forward_returns(events, bars_by_symbol, horizons=horizons)
        for row in rows:
            row["threshold"] = threshold
        signal_rows.extend(rows)

    baseline_events = [
        Event(
            ts=epoch,
            underlying=rec["underlying"],
            direction=1,  # long-only market-drift reference
            composite=composite,
            strength=rec.get("strength"),
            price_ok=_price_confirm_ok(rec),
            regime=rec.get("regime"),
            flow_source=rec.get("flow_source"),
            nominated=rec.get("nominated"),
        )
        for rec in records
        for composite in [_as_float(rec.get("composite"))]
        for epoch in [_epoch_of(rec)]
        if rec.get("underlying") and composite is not None and epoch is not None
    ]
    baseline_rows = forward_returns(baseline_events, bars_by_symbol, horizons=horizons)

    underlyings = sorted({rec["underlying"] for rec in records if rec.get("underlying")})
    epochs = [e for e in (_epoch_of(r) for r in records) if e is not None]

    return {
        "n_records": len(records),
        "underlyings": underlyings,
        "time_span": {
            "first": _iso(min(epochs)) if epochs else None,
            "last": _iso(max(epochs)) if epochs else None,
        },
        "thresholds": list(thresholds),
        "horizons": list(horizons),
        "min_gap_minutes": min_gap_minutes,
        "events_count": events_count,
        "signal": summarize_signal(signal_rows),
        "baseline": summarize_baseline(baseline_rows),
    }


def _fmt_rate(x: Optional[float]) -> str:
    return f"{x * 100:.0f}%" if x is not None else "n/a"


def _fmt_bps(x: Optional[float]) -> str:
    return f"{x:+.1f}bps" if x is not None else "n/a"


def render_report(summary: dict) -> str:
    """Plain-text rendering of a build_report() dict."""
    lines: list[str] = []
    lines.append("DPOF signal study")
    lines.append(f"records: {summary['n_records']}  underlyings: {', '.join(summary['underlyings']) or 'none'}")
    span = summary["time_span"]
    lines.append(f"time span: {span['first'] or 'n/a'} .. {span['last'] or 'n/a'}")
    lines.append(
        f"thresholds: {summary['thresholds']}  horizons(min): {summary['horizons']}  "
        f"min_gap_minutes: {summary['min_gap_minutes']}"
    )
    lines.append("")

    for threshold in summary["thresholds"]:
        n_events = summary["events_count"].get(threshold, 0)
        lines.append(f"--- threshold {threshold} ({n_events} events) ---")
        rows = [r for r in summary["signal"] if r["threshold"] == threshold]
        rows.sort(key=lambda r: (r["underlying"], r["horizon"], r["price_ok"]))
        if not rows:
            lines.append("  (no events)")
        for r in rows:
            note = f"  ({r['note']})" if r["note"] else ""
            lines.append(
                f"  {r['underlying']:<6s} h={r['horizon']:>3}m price_ok={r['price_ok']:<13s} "
                f"n={r['n']:<4} no_data={r['no_data']:<3} "
                f"hit={_fmt_rate(r['hit_rate']):>5} mean={_fmt_bps(r['mean_bps']):>9} "
                f"median={_fmt_bps(r['median_bps']):>9}{note}"
            )
        lines.append("")

    lines.append(
        "--- baseline (long-only, every ~30 s record — market drift reference; "
        "overlapping samples, so n is not a count of independent observations) ---"
    )
    base_rows = sorted(summary["baseline"], key=lambda r: (r["underlying"], r["horizon"]))
    for r in base_rows:
        note = f"  ({r['note']})" if r["note"] else ""
        lines.append(
            f"  {r['underlying']:<6s} h={r['horizon']:>3}m "
            f"n={r['n']:<4} no_data={r['no_data']:<3} "
            f"hit={_fmt_rate(r['hit_rate']):>5} mean={_fmt_bps(r['mean_bps']):>9} "
            f"median={_fmt_bps(r['median_bps']):>9}{note}"
        )
    return "\n".join(lines)


def filter_by_date(records: list[dict], since: Optional[str], until: Optional[str]) -> list[dict]:
    """Keeps only records whose `ts` calendar date (in `ts`'s own tz — ET
    for signals.jsonl) falls within [since, until] (either bound optional).
    Records with an unparsable `ts` are dropped. NOTE: filtering happens
    BEFORE find_events, so the first record inside the window is always
    treated the same as a fresh session (§ find_events' gap-reset rule) —
    an expected boundary effect of windowing, not a bug."""
    if not since and not until:
        return records
    since_d: Optional[date] = datetime.strptime(since, "%Y-%m-%d").date() if since else None
    until_d: Optional[date] = datetime.strptime(until, "%Y-%m-%d").date() if until else None
    out: list[dict] = []
    for rec in records:
        try:
            d = datetime.fromisoformat(str(rec.get("ts"))).date()
        except (ValueError, TypeError):
            continue
        if since_d and d < since_d:
            continue
        if until_d and d > until_d:
            continue
        out.append(rec)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Bars I/O — offline file (stdlib only) and the live Alpaca fetch path
# ─────────────────────────────────────────────────────────────────────────

def load_bars_file(path: str) -> list[dict]:
    """Reads --bars-file / --save-bars format: JSONL lines
    `{"symbol","t","c"}` (default, any non-.csv extension) or CSV rows
    `symbol,t,c` (`.csv` extension, header row optional — a non-numeric `t`
    column, e.g. a header, is simply skipped). Never raises; malformed rows
    are skipped."""
    ext = os.path.splitext(path)[1].lower()
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".csv":
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                symbol = row[0]
                t = _as_float(row[1])
                c = _as_float(row[2])
                if not symbol or t is None or c is None:
                    continue
                records.append({"symbol": symbol, "t": t, "c": c})
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                symbol = rec.get("symbol")
                t = _as_float(rec.get("t"))
                c = _as_float(rec.get("c"))
                if not symbol or t is None or c is None:
                    continue
                records.append({"symbol": symbol, "t": t, "c": c})
    return records


def group_bars_by_symbol(bar_records: list[dict]) -> dict:
    out: dict[str, list[dict]] = {}
    for r in bar_records:
        out.setdefault(r["symbol"], []).append({"t": r["t"], "c": r["c"]})
    for lst in out.values():
        lst.sort(key=lambda b: b["t"])
    return out


def save_bars_file(path: str, bars_by_symbol: dict) -> None:
    """Writes fetched bars in the same JSONL shape load_bars_file() reads
    back (`--save-bars` cache). Only called from the live-fetch path."""
    with open(path, "w", encoding="utf-8") as f:
        for symbol, bars in bars_by_symbol.items():
            for b in bars:
                f.write(json.dumps({"symbol": symbol, "t": b["t"], "c": b["c"]}) + "\n")


def _load_dotenv_if_available() -> None:
    """Best-effort `load_dotenv(<repo>/.env)` — a no-op if python-dotenv
    isn't installed (this script's own unit-test environment has neither
    `dotenv` nor `alpaca`, deliberately, per §5's stdlib-only pure core).
    MUST run BEFORE any `agent.*` import (CLAUDE.md anti-pattern #34) so
    `agent.dpof.config.load_dpof_config()` sees a fully-populated
    environment instead of running silently keyless."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))


def _resolve_default_thresholds() -> tuple:
    """DPOFConfig's live threshold_normal/threshold_strong when the config
    module is importable, else the module's own 0.15/0.30 fallback. Reads
    only dataclass defaults (no keys), but still follows the
    load_dotenv-before-agent.* ordering for consistency."""
    try:
        _load_dotenv_if_available()
        from agent.dpof.config import DPOFConfig
        cfg = DPOFConfig()
        return (cfg.threshold_normal, cfg.threshold_strong)
    except Exception:
        return DEFAULT_THRESHOLDS


def _default_fetch_window(records: list[dict], horizons: tuple) -> tuple:
    """[earliest record, latest record + max(horizon) + 5min buffer] in
    UTC — wide enough that every event's forward-return window has bars to
    look up, without needing a market calendar."""
    epochs = [e for e in (_epoch_of(r) for r in records) if e is not None]
    if not epochs:
        now = datetime.now(timezone.utc)
        return now, now
    start = datetime.fromtimestamp(min(epochs), tz=timezone.utc)
    max_h = max(horizons) if horizons else 0
    end = datetime.fromtimestamp(max(epochs) + max_h * 60 + 300, tz=timezone.utc)
    return start, end


def et_day_windows(start_utc: datetime, end_utc: datetime) -> list[tuple[datetime, datetime]]:
    """Split [start_utc, end_utc) at Eastern-time midnights: one bars request
    per ET day (design frame §5.1 / R5). DST-safe (midnight is built per ET
    date, never by adding 24 h). Empty when end <= start."""
    out: list[tuple[datetime, datetime]] = []
    cur = start_utc
    while cur < end_utc:
        nd = cur.astimezone(_ET).date() + timedelta(days=1)
        next_midnight = datetime(nd.year, nd.month, nd.day, tzinfo=_ET).astimezone(timezone.utc)
        seg_end = min(next_midnight, end_utc)
        out.append((cur, seg_end))
        cur = seg_end
    return out


def _is_feed_subscription_error(e: Exception) -> bool:
    """401/403 only -- the same test agent/dpof/broker.py uses for its own
    IEX downgrade (status_code on the error or on its response)."""
    status = getattr(e, "status_code", None)
    if status is None:
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None) if resp is not None else None
    return status in (401, 403)


def fetch_bars_from_alpaca(
    underlyings: list[str],
    start_utc: datetime,
    end_utc: datetime,
    *,
    client=None,
    feed=None,
) -> dict:
    """Live 1-min bars via the SYNC alpaca-py client (no asyncio — this is
    a one-shot CLI, not the engine's async broker), one request per
    underlying per ET day (`et_day_windows`). `agent.*`/`alpaca` are
    imported here, lazily, AFTER _load_dotenv_if_available() (anti-pattern
    #34). Never prints/logs the resolved keys. A 401/403 on the configured
    feed switches to IEX for the rest of the run (sticky, like broker.py);
    any other failure is logged to stderr and that window comes back with
    no bars (price_at then reports None/no_data, never a fake 0).
    `client`/`feed` are injectable for tests; production passes neither."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    if client is None:
        _load_dotenv_if_available()
        from agent.dpof.config import load_dpof_config
        from alpaca.data.historical.stock import StockHistoricalDataClient

        cfg = load_dpof_config()
        client = StockHistoricalDataClient(api_key=cfg.alpaca_key_id, secret_key=cfg.alpaca_secret_key)
        feed_map = {
            "iex": DataFeed.IEX,
            "sip": DataFeed.SIP,
            "delayed_sip": DataFeed.DELAYED_SIP,
            "otc": DataFeed.OTC,
        }
        feed = feed_map.get(cfg.stock_feed, DataFeed.IEX)
    if feed is None:
        feed = DataFeed.IEX

    def _fetch(symbol: str, feed_choice, ws: datetime, we: datetime) -> list[dict]:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=ws,
            end=we,
            feed=feed_choice,
        )
        barset = client.get_stock_bars(req)
        try:
            bars = barset[symbol]
        except Exception:
            bars = []
        got: list[dict] = []
        for b in bars or []:
            ts = getattr(b, "timestamp", None)
            epoch = ts.timestamp() if ts is not None else None
            c = _as_float(getattr(b, "close", None))
            if epoch is None or c is None:
                continue
            got.append({"t": epoch, "c": c})
        return got

    windows = et_day_windows(start_utc, end_utc)
    current_feed = feed
    out: dict[str, list[dict]] = {}
    for symbol in underlyings:
        by_t: dict[float, dict] = {}
        for ws, we in windows:
            try:
                got = _fetch(symbol, current_feed, ws, we)
            except Exception as e:
                if _is_feed_subscription_error(e) and current_feed != DataFeed.IEX:
                    print(f"warning: {getattr(current_feed, 'value', current_feed)} feed refused "
                          f"({type(e).__name__}); using IEX for the rest of this run", file=sys.stderr)
                    current_feed = DataFeed.IEX
                    try:
                        got = _fetch(symbol, current_feed, ws, we)
                    except Exception as e2:
                        print(f"warning: bar fetch failed for {symbol} {ws.date()} (IEX too): {e2}", file=sys.stderr)
                        got = []
                else:
                    print(f"warning: bar fetch failed for {symbol} {ws.date()}: {e}", file=sys.stderr)
                    got = []
            for bar in got:
                by_t[bar["t"]] = bar
        out[symbol] = sorted(by_t.values(), key=lambda b: b["t"])
    return out


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    default_thresholds = _resolve_default_thresholds()
    ap = argparse.ArgumentParser(
        description="Offline study: what did the underlying do after a DPOF composite threshold crossing?",
    )
    ap.add_argument(
        "--signals", default=DEFAULT_SIGNALS_FILE,
        help=f"signals.jsonl path (default: {DEFAULT_SIGNALS_FILE}; honours CLIO_DPOF_DATA_DIR)",
    )
    ap.add_argument("--bars-file", default=None, help="offline bars JSONL/CSV; omit to fetch 1-min bars from Alpaca")
    ap.add_argument("--save-bars", default=None, help="cache fetched bars to this JSONL path (fetch path only)")
    ap.add_argument(
        "--thresholds", default=",".join(str(t) for t in default_thresholds),
        help=f"comma-separated composite thresholds (default: {','.join(str(t) for t in default_thresholds)})",
    )
    ap.add_argument(
        "--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help=f"comma-separated forward horizons in minutes (default: {','.join(str(h) for h in DEFAULT_HORIZONS)})",
    )
    ap.add_argument("--min-gap-minutes", type=float, default=DEFAULT_MIN_GAP_MINUTES)
    ap.add_argument("--since", default=None, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--until", default=None, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--json-out", default=None, help="optional machine-readable report path")
    return ap


def _read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def main(argv: Optional[list] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    try:
        thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip())
    except ValueError:
        print(f"error: --thresholds must be comma-separated numbers, got {args.thresholds!r}", file=sys.stderr)
        return 2
    if not thresholds:
        print("error: --thresholds resolved to an empty list", file=sys.stderr)
        return 2

    try:
        horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    except ValueError:
        print(f"error: --horizons must be comma-separated integers, got {args.horizons!r}", file=sys.stderr)
        return 2
    if not horizons:
        print("error: --horizons resolved to an empty list", file=sys.stderr)
        return 2

    if not os.path.exists(args.signals):
        print(f"error: signals file not found: {args.signals}", file=sys.stderr)
        return 2

    records = load_signal_records(_read_lines(args.signals))
    records = filter_by_date(records, args.since, args.until)

    if not records:
        print("no signal records with a usable composite/ts in the requested range", file=sys.stderr)
        return 1

    if args.bars_file:
        if args.save_bars:
            print("warning: --save-bars ignored — --bars-file already provided offline bars", file=sys.stderr)
        bars_by_symbol = group_bars_by_symbol(load_bars_file(args.bars_file))
    else:
        underlyings = sorted({r["underlying"] for r in records if r.get("underlying")})
        start_utc, end_utc = _default_fetch_window(records, horizons)
        bars_by_symbol = fetch_bars_from_alpaca(underlyings, start_utc, end_utc)
        if args.save_bars:
            save_bars_file(args.save_bars, bars_by_symbol)

    report = build_report(
        records,
        thresholds=thresholds,
        horizons=horizons,
        min_gap_minutes=args.min_gap_minutes,
        bars_by_symbol=bars_by_symbol,
    )
    print(render_report(report))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
