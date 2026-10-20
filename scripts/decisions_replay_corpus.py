#!/usr/bin/env python3
"""
CLIO DPOF -> the decision record -- replay corpus builder.

Reads a DPOF `signals.jsonl` observability log (agent/dpof/signal_log.py)
READ-ONLY and writes request JSONL (§4b's interchange format: one
`{"producer", "producer_ref", "state", "questions"}` object per line) for
every logged cycle record that carries a nominated/candidate signal
(`record["nominated"] is True`). Each request is built by rebuilding the
`Signal` and the parts of the `MarketSnapshot` that production actually used
and calling the SAME `agent.dpof.jev.build_entry_request(...)` production
code path -- same buckets, same vocabulary, same `Question` shapes -- with
`sizing_ctx=None` (a replay never invents position-sizing context; production
itself only adds that question when a real one is supplied).

FAITHFULNESS RULE (the core of this script): `signals.jsonl` is deliberately
OBSERVABILITY-ONLY (see signal_log.py's own module docstring) -- engine.py's
`_log_cycle_signals` never wrote every field `build_entry_request` reads.
Where production had a real value the log simply does not carry, this script
must NEVER let it silently become `build_entry_request`'s own "unavailable"/
"unknown" sentinel in the replayed state -- that would teach a false fact (a
value was there and we just don't know it, not that it was truly absent).
Instead the corresponding state key -- and every question whose instructions/
criteria depend on it by name -- is OMITTED from the emitted line and counted
in `producer_ref["omitted"]`. If a record can't even support the always-asked
`setup` question faithfully (its own underlying/direction/strength/timestamp,
or the primary underlying's own pillar readings, aren't reconstructible), the
whole record is skipped and counted by reason instead of emitted.

Field-by-field mapping (`build_entry_request`'s inputs -> where signals.jsonl
records them, or "not logged"; see the hand-back for the same table):

  signal.direction                     -> record["direction"]                     (reconstructible)
  signal.symbol                        -> record["underlying"]                    (reconstructible)
  signal.strength                      -> record["strength"]                      (reconstructible)
  signal.pillars_used["nofa"/"premium"/
    "darkpool"]                        -> the PRIMARY underlying's own
                                           record["pillars"][name] IN THE SAME
                                           CYCLE (same "ts") -- compute_composite
                                           (agent/dpof/signals.py) always reads
                                           the primary's own UnderlyingSnapshot,
                                           never the traded symbol's, so a
                                           divergence trade (signal.symbol !=
                                           record["signal_basis"]) needs the
                                           sibling record for the primary       (reconstructible; skip if the
                                                                                  sibling can't be found)
  signal.pillars_used["gex"]           -> not read by build_entry_request at all (irrelevant)
  now_et (session bucket)              -> record["ts"] (parsed in America/New_York) (reconstructible)
  state.signals.gamma_regime /
    price_vs_vwap / momentum_15m /
    volatility                         -> record["jev_words"][key] (engine.py
                                           computes them with jev.py's OWN
                                           helpers from the same snapshot and
                                           the same traded underlying that
                                           build_entry_request reads; logged
                                           since Kronos PR #74)                 (reconstructible when logged;
                                                                                  OMITTED per key otherwise --
                                                                                  older records never carried
                                                                                  them, and record["regime"] is a
                                                                                  different vocabulary that cannot
                                                                                  be losslessly translated)
  snap.headlines                       -> NOT LOGGED (no headline text
                                           anywhere in signals.jsonl)           (OMITTED, always; headline_conflict
                                                                                  and event_risk questions omitted too)
  sizing_ctx                           -> never supplied (this script always
                                           passes sizing_ctx=None, matching
                                           production's own "no sizing_ctx"
                                           behavior)                            (OMITTED, always; size_reckless
                                                                                  question omitted too)

Usage:
  python3 scripts/decisions_replay_corpus.py --out requests.jsonl
  python3 scripts/decisions_replay_corpus.py --signals /path/to/signals.jsonl \\
      --out requests.jsonl --since 2026-09-01 --limit 500
  python3 scripts/decisions_replay_corpus.py --out requests.jsonl --dry-run

canonical_json/state_hash come from `agent.decisions.records` -- the one
definition CLIO shares (byte-identical, golden-vector pinned) with Project
the consumer's reference records module.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime
from typing import Iterable, Iterator, Optional
from zoneinfo import ZoneInfo

# scripts/decisions_replay_corpus.py -> 1x dirname = repo root (matches
# scripts/dpof_signal_study.py's own convention). Import-only sys.path
# mutation; nothing here needs agent.config / any key at import time (see
# tests/test_decisions_replay_corpus.py's fresh-subprocess proof).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.dpof.config import DPOFConfig                          # noqa: E402
from agent.dpof.jev import build_entry_request                    # noqa: E402
from agent.dpof.models import Direction, MarketSnapshot, Signal   # noqa: E402
from agent.decisions.records import canonical_json, state_hash  # noqa: E402,F401

ET = ZoneInfo("America/New_York")

DEFAULT_DATA_DIR = os.environ.get("CLIO_DPOF_DATA_DIR", os.path.join(_REPO_ROOT, "data", "dpof"))
DEFAULT_SIGNALS_FILE = os.path.join(DEFAULT_DATA_DIR, "signals.jsonl")

PRODUCER = "corpus_replay"

# The three DIRECTIONAL pillars build_entry_request actually reads off
# Signal.pillars_used (bucket_pillar("nofa"/"premium"/"darkpool")); "gex" is
# never read by build_entry_request at all (agent/dpof/jev.py's
# _gamma_regime_words reads UnderlyingSnapshot.gex_regime/pin_risk directly,
# a completely separate computation from pillars_used["gex"]).
_PILLAR_KEYS = ("nofa", "premium", "darkpool")

# The bucket words engine.py logs per record as record["jev_words"] (Kronos
# PR #74). A record that lacks one (every record written before that change)
# gets that key OMITTED, never build_entry_request's "unknown" sentinel.
_JEV_WORD_KEYS = ("gamma_regime", "price_vs_vwap", "momentum_15m", "volatility")
# A logged word is only trusted when it has the shape jev.py's helpers emit
# (lowercase words, digits, spaces and "()+-." -- e.g. "calm (below 15)",
# "accelerating (pin risk)"); anything else is treated as not logged.
_JEV_WORD_RE = re.compile(r"[a-z0-9 ().+\-]{1,40}")  # fullmatch: "$" would admit a trailing newline
# Structural omissions -- see the module docstring's field-by-field mapping.
# signals.jsonl never logs headline text or sizing context, so these are
# ALWAYS omitted, never a per-record judgment call.
_OMITTED_STATE_KEYS = ("headlines",)
_OMITTED_QUESTION_IDS = ("headline_conflict", "event_risk", "size_reckless")


# ─────────────────────────────────────────────────────────────────────────
# Pure helpers (no I/O -- independently unit-testable)
# ─────────────────────────────────────────────────────────────────────────

def _num(v) -> Optional[float]:
    """isinstance-safe numeric coercion: rejects bool (bool is an int subclass) --
    same discipline as agent.dpof.jev._num."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _parse_ts_et(ts) -> Optional[datetime]:
    """Parses signals.jsonl's `ts` (a tz-aware ET ISO string, per engine.py's
    `now_et.isoformat()`) into an America/New_York datetime. Never reads the
    real clock -- a replay must use the record's OWN timestamp for every
    time-dependent bucket (session), never `datetime.now()`. A naive string
    (no offset) is localized to ET directly (CLAUDE.md's fromisoformat-naive
    gotcha); an offset-aware one is converted to ET (a no-op when it was
    already ET, which is what engine.py always writes). Returns None on
    anything unparsable -- callers must skip the record, never assume UTC."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)


def _parse_direction(value) -> Optional[Direction]:
    try:
        return Direction(value)
    except (ValueError, TypeError):
        return None


def _logged_jev_words(rec: dict) -> dict:
    """The record's own logged bucket words (record["jev_words"]), keeping
    only well-formed strings for the four keys build_entry_request derives
    from the traded underlying's snapshot and VIX. A word of "unknown" is
    kept: production computed and showed Jev exactly that word, so it is a
    real fact, unlike a key the log never carried."""
    words = rec.get("jev_words")
    if not isinstance(words, dict):
        return {}
    out = {}
    for key in _JEV_WORD_KEYS:
        w = words.get(key)
        if isinstance(w, str) and _JEV_WORD_RE.fullmatch(w):
            out[key] = w
    return out


def _omitted_fields(restored: Iterable[str] = ()) -> list[str]:
    """The `producer_ref["omitted"]` list for one emitted record: every
    bucket-word key the record's log line did not carry (`restored` lists the
    ones it did), plus the fields signals.jsonl never logs at all."""
    restored = set(restored)
    out = [f"signals.{k}" for k in _JEV_WORD_KEYS if k not in restored]
    out.extend(_OMITTED_STATE_KEYS)
    out.append("sizing")
    out.extend(f"questions.{q}" for q in _OMITTED_QUESTION_IDS)
    return sorted(out)


def _group_cycles(
    parsed: list[tuple[int, Optional[dict]]]
) -> Iterator[list[tuple[int, Optional[dict]]]]:
    """Groups (line_no, record) pairs into per-cycle groups: SignalLog.write_cycle
    appends every underlying's record for one cycle in a single write (one
    `"\\n".join(lines)` call over `records` built in `cfg.underlyings` order --
    see agent/dpof/signal_log.py's write_cycle and engine.py's
    `_log_cycle_signals`), so a cycle's records are always CONTIGUOUS lines
    sharing an identical `ts` string. A record with a missing/malformed `ts`
    (parse failure, or not a dict at all) is yielded as its own singleton
    group -- it can never be a sibling lookup match anyway."""
    current_ts: Optional[str] = None
    current: list[tuple[int, Optional[dict]]] = []
    for line_no, rec in parsed:
        ts = rec.get("ts") if isinstance(rec, dict) else None
        if not isinstance(ts, str) or not ts:
            if current:
                yield current
                current = []
                current_ts = None
            yield [(line_no, rec)]
            continue
        if ts != current_ts:
            if current:
                yield current
            current = [(line_no, rec)]
            current_ts = ts
        else:
            current.append((line_no, rec))
    if current:
        yield current


def _resolve_primary_pillars(
    rec: dict, group: list[tuple[int, Optional[dict]]]
) -> Optional[dict]:
    """Signal.pillars_used's directional pillars (nofa/premium/darkpool), taken
    from the PRIMARY underlying's own logged pillar readings for THIS cycle --
    never from `rec` itself unless `rec` already IS the primary's record.

    Production's compute_composite (agent/dpof/signals.py) always reads
    `snap.underlyings.get(cfg.primary)` -- the primary underlying -- to build
    Signal.pillars_used, regardless of which underlying select_instrument
    finally picked to trade (a "divergence" trade can pick the secondary).
    `record["signal_basis"]` is engine.py's own recording of `cfg.underlyings[0]`
    at the time (only ever set on the record where `is_signal_symbol` was
    True, i.e. exactly our nominated candidates) -- robust to config changes
    over the log's history, so we use it instead of reading live config.

    When `rec["underlying"] == rec["signal_basis"]` (the common case: no
    divergence trade this cycle), `rec` already carries the primary's own
    pillars and no sibling lookup is needed. Otherwise this searches `group`
    (every record sharing this cycle's `ts`, per `_group_cycles`) for the
    sibling whose `underlying` equals the primary symbol.

    A pillar only enters the returned dict when the log recorded it
    `available: True` with a real numeric `value` -- exactly
    `UnderlyingSnapshot.available_pillars()`'s own filter, so
    `bucket_pillar(pillars_used.get(name))` reproduces production's
    "unavailable" bucket faithfully (a REAL fact from the log, never a
    fabrication) whenever the pillar genuinely wasn't available that cycle.

    Returns None when the primary's own record can't be found in this cycle's
    group at all (e.g. signals.jsonl's line-count trim cut a cycle's records
    apart) -- callers must skip the whole record in that case; there is no
    faithful way to build pillars_used without it.
    """
    primary_sym = rec.get("signal_basis")
    if not isinstance(primary_sym, str) or not primary_sym:
        return None

    primary_rec: Optional[dict] = None
    if rec.get("underlying") == primary_sym:
        primary_rec = rec
    else:
        for _, other in group:
            if isinstance(other, dict) and other.get("underlying") == primary_sym:
                primary_rec = other
                break
    if primary_rec is None:
        return None

    pillars = primary_rec.get("pillars")
    if not isinstance(pillars, dict):
        return None

    out: dict = {}
    for name in _PILLAR_KEYS:
        p = pillars.get(name)
        if not isinstance(p, dict) or p.get("available") is not True:
            continue
        val = _num(p.get("value"))
        if val is not None:
            out[name] = val
    return out


# ─────────────────────────────────────────────────────────────────────────
# Per-record request building
# ─────────────────────────────────────────────────────────────────────────

def build_replay_request(
    rec: dict, line_no: int, group: list[tuple[int, Optional[dict]]], cfg: DPOFConfig,
) -> tuple[Optional[dict], Optional[str]]:
    """Builds one request-JSONL line (§4b) for a single nominated signals.jsonl
    record, or returns (None, skip_reason) when it can't be built faithfully.
    Never raises -- an unexpected failure inside build_entry_request itself
    is caught and reported as a skip reason rather than aborting the whole
    replay run over one bad record."""
    symbol = rec.get("underlying")
    if not isinstance(symbol, str) or not symbol:
        return None, "missing_underlying"

    now_et = _parse_ts_et(rec.get("ts"))
    if now_et is None:
        return None, "missing_or_bad_ts"

    direction = _parse_direction(rec.get("direction"))
    if direction is None or direction is Direction.NEUTRAL:
        return None, "missing_or_neutral_direction"

    strength = rec.get("strength")
    if not isinstance(strength, str) or strength == "none":
        return None, "missing_strength"

    pillars_used = _resolve_primary_pillars(rec, group)
    if pillars_used is None:
        return None, "primary_pillars_unavailable"

    signal = Signal(symbol=symbol, direction=direction, strength=strength, pillars_used=pillars_used)
    # underlyings={} / vix=None / headlines=[] on purpose: the log never
    # records the raw snapshot fields build_entry_request reads (see the
    # module docstring's mapping table). The empty snapshot makes it emit its
    # own "unknown" sentinels, which are then replaced by the record's logged
    # words or stripped -- never kept as if they were real values.
    snap = MarketSnapshot(asof=now_et.timestamp(), underlyings={}, vix=None, headlines=[])

    try:
        state, questions = build_entry_request(signal, snap, now_et, cfg, sizing_ctx=None)
    except Exception as e:  # defensive only -- build_entry_request is pure and should never raise
        return None, f"build_entry_request_error:{type(e).__name__}"

    signals_block = state.get("signals")
    if not isinstance(signals_block, dict):
        return None, "build_entry_request_malformed"
    # Production's words for the traded underlying (signal.symbol == this
    # record's own underlying, which is what build_entry_request reads) --
    # never the primary sibling's, even on a divergence trade.
    words = _logged_jev_words(rec)
    for key in _JEV_WORD_KEYS:
        if key in words:
            signals_block[key] = words[key]
        else:
            signals_block.pop(key, None)
    for key in _OMITTED_STATE_KEYS:
        state.pop(key, None)

    request = {
        "producer": PRODUCER,
        "producer_ref": {
            "source": "signals.jsonl",
            "line": line_no,
            "ts": rec.get("ts"),
            "symbol": symbol,
            "omitted": _omitted_fields(words),
        },
        "state": state,
        "questions": questions,
    }
    return request, None


# ─────────────────────────────────────────────────────────────────────────
# Pure orchestration core (no file I/O) -- independently testable
# ─────────────────────────────────────────────────────────────────────────

def process_signal_lines(
    lines: Iterable[str],
    cfg: DPOFConfig,
    *,
    since: Optional[date] = None,
    limit: Optional[int] = None,
    existing_hashes: Optional[Iterable[str]] = None,
) -> tuple[list[dict], dict]:
    """Consumes already-read signals.jsonl lines; returns (requests, summary).
    `existing_hashes` seeds the dedup set for the --append case -- the CLI
    owns reading them off --out. Never touches the filesystem itself."""
    seen_hashes: set = set(existing_hashes or ())
    preseeded = len(seen_hashes)

    parsed: list[tuple[int, Optional[dict]]] = []
    records_read = 0
    malformed_lines = 0
    for line_no, raw in enumerate(lines, start=1):
        raw = raw.strip() if isinstance(raw, str) else raw
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            rec = None
        if not isinstance(rec, dict):
            malformed_lines += 1
            parsed.append((line_no, None))
            continue
        records_read += 1
        parsed.append((line_no, rec))

    requests: list[dict] = []
    skipped: dict[str, int] = {}
    omitted_counts: dict[str, int] = {}
    candidates = 0
    since_filtered = 0
    deduped = 0
    limit_reached = False

    for group in _group_cycles(parsed):
        if limit_reached:
            break
        for line_no, rec in group:
            if limit is not None and len(requests) >= limit:
                limit_reached = True
                break
            if not isinstance(rec, dict) or rec.get("nominated") is not True:
                continue
            candidates += 1

            if since is not None:
                dt = _parse_ts_et(rec.get("ts"))
                if dt is None or dt.date() < since:
                    since_filtered += 1
                    continue

            request, reason = build_replay_request(rec, line_no, group, cfg)
            if request is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue

            h = state_hash(request["state"])
            if h in seen_hashes:
                deduped += 1
                continue
            seen_hashes.add(h)
            requests.append(request)
            for field in request["producer_ref"]["omitted"]:
                omitted_counts[field] = omitted_counts.get(field, 0) + 1

    summary = {
        "records_read": records_read,
        "malformed_lines": malformed_lines,
        "candidates": candidates,
        "since_filtered": since_filtered,
        "emitted": len(requests),
        "deduped": deduped,
        "limit_reached": limit_reached,
        "existing_hashes_loaded": preseeded,
        "skipped": skipped,
        "omitted_field_counts": omitted_counts,
    }
    return requests, summary


# ─────────────────────────────────────────────────────────────────────────
# File I/O + CLI
# ─────────────────────────────────────────────────────────────────────────

def _load_existing_hashes(path: str) -> set:
    """Best-effort: state_hash of every well-formed existing line's `state`
    (the --append dedup seed). Malformed lines are ignored, never fatal --
    this is a read of our OWN prior output, not untrusted input, but a
    partially-written prior run must not crash a later --append."""
    hashes: set = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and isinstance(obj.get("state"), dict):
                    hashes.add(state_hash(obj["state"]))
    except OSError:
        pass
    return hashes


def render_summary(args: argparse.Namespace, summary: dict) -> str:
    lines = ["DPOF -> decision-record replay corpus"]
    lines.append(f"signals: {args.signals}")
    if args.dry_run:
        lines.append(f"out: {args.out} (dry-run — nothing written)")
    else:
        lines.append(f"out: {args.out}" + (" (append)" if args.append else ""))
    lines.append("")
    lines.append(f"records read: {summary['records_read']}  (malformed lines: {summary['malformed_lines']})")
    lines.append(f"candidates (nominated): {summary['candidates']}")
    if summary["since_filtered"]:
        lines.append(f"since-filtered: {summary['since_filtered']}")
    lines.append(f"emitted: {summary['emitted']}")
    lines.append(f"deduped: {summary['deduped']} (existing hashes loaded: {summary['existing_hashes_loaded']})")
    if summary["limit_reached"]:
        lines.append("limit reached: yes (stopped early)")
    lines.append("")
    lines.append("skipped by reason:")
    if summary["skipped"]:
        for reason, count in sorted(summary["skipped"].items()):
            lines.append(f"  {reason}: {count}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("omitted-field counts (emitted records whose producer_ref.omitted lists the field):")
    for field, count in sorted(summary["omitted_field_counts"].items()):
        lines.append(f"  {field}: {count}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Replay a DPOF signals.jsonl into request JSONL for the downstream labeller.",
    )
    ap.add_argument(
        "--signals", default=DEFAULT_SIGNALS_FILE,
        help=f"signals.jsonl path, opened READ-ONLY (default: {DEFAULT_SIGNALS_FILE})",
    )
    ap.add_argument("--out", required=True, help="request-JSONL output path")
    ap.add_argument(
        "--append", action="store_true",
        help="add to an existing --out instead of refusing to overwrite it",
    )
    ap.add_argument("--since", default=None, help="YYYY-MM-DD, ET -- only records on/after this date")
    ap.add_argument("--limit", type=int, default=None, help="max requests to emit")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print the summary only; never touches --out",
    )
    return ap


def main(argv: Optional[list] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if not os.path.exists(args.signals):
        print(f"error: signals file not found: {args.signals}", file=sys.stderr)
        return 2

    since_date: Optional[date] = None
    if args.since:
        try:
            since_date = datetime.strptime(args.since, "%Y-%m-%d").date()
        except ValueError:
            print(f"error: --since must be YYYY-MM-DD, got {args.since!r}", file=sys.stderr)
            return 2

    if args.limit is not None and args.limit < 0:
        print("error: --limit must be >= 0", file=sys.stderr)
        return 2

    if not args.dry_run and os.path.exists(args.out) and not args.append:
        print(f"error: --out already exists: {args.out} (pass --append to add to it)", file=sys.stderr)
        return 2

    existing_hashes: set = set()
    if args.append and os.path.exists(args.out):
        existing_hashes = _load_existing_hashes(args.out)

    cfg = DPOFConfig()

    with open(args.signals, "r", encoding="utf-8") as f:
        lines = f.readlines()

    requests, summary = process_signal_lines(
        lines, cfg, since=since_date, limit=args.limit, existing_hashes=existing_hashes,
    )

    if not args.dry_run:
        mode = "a" if args.append else "x"
        try:
            with open(args.out, mode, encoding="utf-8") as f:
                for req in requests:
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
        except FileExistsError:
            print(f"error: --out already exists: {args.out} (pass --append to add to it)", file=sys.stderr)
            return 2

    print(render_summary(args, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
