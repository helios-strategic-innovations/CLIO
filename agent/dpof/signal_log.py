"""
CLIO DPOF — signal observability log (I/O).

Task ORCH-7Q4K-DPOF (2026-09-25, operator: "quiet tape + a diluting dead GEX
pillar" was the only explanation the live session could give for a zero-entry
day). Every pre-decider rejection reason in engine.py was DEBUG-only (CLAUDE.md
anti-pattern #27: DEBUG is invisible at production INFO level) and
`data/dpof/decisions.jsonl` only records cycles where a decider call actually
happened -- a "no nomination at all" cycle left no trace anywhere. This module
gives every cycle, for every configured underlying, a durable record of what
the signal/entry pipeline saw and why it did (or didn't) attempt an entry --
`data/dpof/signals.jsonl` -- plus a same-day rollup
(`data/dpof/signal_day_stats.json`) so `/dpof why` and the EOD digest can
answer "why didn't it trade today" in O(1) reads instead of re-scanning a
full day of raw records.

OBSERVABILITY ONLY (operator hard constraint): nothing in this module reads,
computes, or feeds back into any gate, threshold, size, order, or decider/Jev
call. engine.py calls this AFTER its real pipeline (_handle_entry_pipeline /
_launch_entry_deliberation / _apply_entry_outcome) has already run, and only
ever RECORDS what already happened -- see docs/design/clio_dpof.md's
Observability section for the exact record shape.

Task K2 (agent/kronos, docs/design/clio_kronos.md §4) added two ADDITIVE
per-underlying fields to the record engine.py._log_cycle_signals builds --
"kronos" ({asof, p_up_30, mean_return_30} or null when unavailable/disabled)
and "jev_words". Every existing field and its meaning is unchanged; this module
itself (write_cycle/fold_record_into_day_stats) treats both as opaque record
contents like everything else it writes -- no reader here needs updating.

I/O conventions mirror agent.dpof.ledger.DPOFLedger (its own docstring):
every write happens inside ONE run_in_executor call holding an exclusive
fcntl.flock on a SIDECAR lock file (never the file that gets os.replace()'d --
CLAUDE.md anti-pattern #22: flock locks the inode, os.replace swaps it, so a
thread holding the flock across a replace would write to the orphaned inode
and lose the write). JSONL append + line-count trim; atomic .tmp + fsync +
os.replace for the day-stats JSON.
"""

from __future__ import annotations

import asyncio
import fcntl
import gzip
import json
import logging
import os
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from agent.dpof.models import PILLAR_NAMES

logger = logging.getLogger("clio.dpof.signal_log")

_SIGNALS_FILE = "signals.jsonl"
_DAY_STATS_FILE = "signal_day_stats.json"
_LOCK_FILE = ".dpof_signal.lock"
_ARCHIVE_DIR = "archive"
_ARCHIVE_ENV = "CLIO_DPOF_SIGNAL_ARCHIVE"
_ET = ZoneInfo("America/New_York")

DEFAULT_MAX_LINES = 20_000


def signal_archive_enabled() -> bool:
    """CLIO_DPOF_SIGNAL_ARCHIVE: default ON. Only 0/false/off/no disable it;
    unset, blank or any other value keeps the archive ON (a bad value must
    never silently drop the record of trimmed rows)."""
    raw = os.environ.get(_ARCHIVE_ENV)
    if raw is None or raw.strip() == "":
        return True
    return raw.strip().lower() not in ("0", "false", "off", "no")


# ─────────────────────────────────────────────────────────────────────────
# Pure helpers (no I/O — independently unit-testable)
# ─────────────────────────────────────────────────────────────────────────

def flow_source_label(pillar_sources: list[str]) -> str:
    """'uw' | 'selfflow' | 'mixed' | 'unavailable' from the `.source` strings
    of whichever pillars were actually AVAILABLE this cycle for one
    underlying. data.py's self-computed fallback tags its readings
    "selfflow:<symbol>" (see docs/design/clio_dpof.md's Flow fallback
    section); every other non-empty source is treated as UW-derived. Never
    raises; empty/falsy input -> "unavailable"."""
    sources = [s for s in pillar_sources if s]
    if not sources:
        return "unavailable"
    is_self = [s.startswith("selfflow:") for s in sources]
    if all(is_self):
        return "selfflow"
    if not any(is_self):
        return "uw"
    return "mixed"


def format_outcome_log_line(record: dict) -> str:
    """One human-readable INFO line for a signals.jsonl record — the thing
    NearMissLogger.should_log() gates. Pure formatting only; never raises on
    a malformed/partial record (missing keys degrade to 'n/a', never a
    KeyError that could interrupt a cycle)."""
    sym = record.get("underlying", "?")
    outcome = record.get("outcome", "unknown")
    composite = record.get("composite")
    comp_str = f"{composite:+.2f}" if isinstance(composite, (int, float)) else "n/a"
    pillars = record.get("pillars") or {}
    pillar_bits = []
    for name in PILLAR_NAMES:
        p = pillars.get(name) or {}
        v = p.get("value")
        pillar_bits.append(f"{name} {v:+.2f}" if isinstance(v, (int, float)) else f"{name} n/a")
    flow = record.get("flow_source", "unavailable")
    verb = "entry" if outcome in ("entered", "deliberating") else "no entry"
    return (
        f"DPOF {sym}: {verb} — {outcome} (composite {comp_str}; "
        + " ".join(pillar_bits) + f"; flow {flow})"
    )


class NearMissLogger:
    """Per-process, in-memory (NEVER persisted — a restart just re-logs the
    current state once, which is correct) INFO-log rate limiter.

    should_log() returns True on an outcome CHANGE for a given key
    (underlying) -- but at most MAX_CHANGE_LOGS_PER_WINDOW change lines per
    key per `window_seconds`, so an outcome flapping every cycle (composite
    hovering at a threshold) can't log every 30 s -- and at most once per
    (key, outcome) pair per `window_seconds` while the outcome stays the
    same. Pure/cheap — callers do the actual `logger.info(...)` themselves
    when it returns True; this class only decides WHETHER, never WHAT."""

    MAX_CHANGE_LOGS_PER_WINDOW = 4

    def __init__(self) -> None:
        self._last_outcome: dict[str, str] = {}
        self._last_logged_at: dict[tuple, float] = {}
        self._change_log_times: dict[str, list[float]] = {}

    def should_log(self, key: str, outcome: str, now: float, window_seconds: float) -> bool:
        window = max(0.0, window_seconds)
        changed = self._last_outcome.get(key) != outcome
        self._last_outcome[key] = outcome
        if changed:
            recent = [t for t in self._change_log_times.get(key, []) if (now - t) < window]
            if len(recent) >= self.MAX_CHANGE_LOGS_PER_WINDOW:
                self._change_log_times[key] = recent
                return False
            recent.append(now)
            self._change_log_times[key] = recent
            self._last_logged_at[(key, outcome)] = now
            return True
        last = self._last_logged_at.get((key, outcome))
        if last is None or (now - last) >= max(0.0, window_seconds):
            self._last_logged_at[(key, outcome)] = now
            return True
        return False


def _default_day_stats(day: str) -> dict:
    return {
        "day": day,
        "cycles_evaluated": 0,
        "nominations": 0,
        "decider_pass": 0,
        "decider_enter": 0,
        "decider_unavailable": 0,
        "jev_vetoes": 0,
        "entries": 0,
        "underlyings": {},
    }


def _default_underlying_stats() -> dict:
    return {
        "outcome_counts": {},
        "max_abs_composite": None,
        "max_abs_composite_at": None,
        "max_abs_composite_direction": None,
        "pillar_available": {n: 0 for n in PILLAR_NAMES},
        "pillar_total": {n: 0 for n in PILLAR_NAMES},
        "flow_source_counts": {},
    }


def fold_record_into_day_stats(stats: dict, record: dict) -> dict:
    """Pure. Applies ONE signals.jsonl-shaped record to a day-stats dict,
    returning a NEW dict (never mutates the input — matches agent.dpof.risk's
    replace()-everywhere convention, which is what makes this trivially
    unit-testable without any I/O). `stats` should already be for the
    correct ET day — callers (write_cycle) reset to `_default_day_stats()`
    on a day rollover before folding."""
    stats = {**stats, "underlyings": dict(stats.get("underlyings", {}))}
    sym = record.get("underlying")
    if not sym:
        return stats
    u = dict(stats["underlyings"].get(sym) or _default_underlying_stats())
    u["outcome_counts"] = dict(u.get("outcome_counts", {}))
    outcome = record.get("outcome") or "unknown"
    u["outcome_counts"][outcome] = u["outcome_counts"].get(outcome, 0) + 1

    composite = record.get("composite")
    if isinstance(composite, (int, float)):
        ac = abs(composite)
        if u.get("max_abs_composite") is None or ac > u["max_abs_composite"]:
            u["max_abs_composite"] = ac
            u["max_abs_composite_at"] = record.get("ts")
            u["max_abs_composite_direction"] = record.get("direction")

    pillars = record.get("pillars") or {}
    pillar_available = dict(u.get("pillar_available") or {n: 0 for n in PILLAR_NAMES})
    pillar_total = dict(u.get("pillar_total") or {n: 0 for n in PILLAR_NAMES})
    for name in PILLAR_NAMES:
        p = pillars.get(name) or {}
        pillar_total[name] = pillar_total.get(name, 0) + 1
        if p.get("available"):
            pillar_available[name] = pillar_available.get(name, 0) + 1
    u["pillar_available"] = pillar_available
    u["pillar_total"] = pillar_total

    flow_source_counts = dict(u.get("flow_source_counts", {}))
    fs = record.get("flow_source") or "unavailable"
    flow_source_counts[fs] = flow_source_counts.get(fs, 0) + 1
    u["flow_source_counts"] = flow_source_counts

    stats["underlyings"][sym] = u

    if record.get("nominated"):
        stats["nominations"] = stats.get("nominations", 0) + 1
    decider = record.get("decider") or {}
    daction = decider.get("action")
    if daction == "pass":
        stats["decider_pass"] = stats.get("decider_pass", 0) + 1
    elif daction == "enter":
        stats["decider_enter"] = stats.get("decider_enter", 0) + 1
    if outcome == "decider_unavailable":
        stats["decider_unavailable"] = stats.get("decider_unavailable", 0) + 1
    jev = record.get("jev") or {}
    if jev.get("allow") is False:
        stats["jev_vetoes"] = stats.get("jev_vetoes", 0) + 1
    if outcome == "entered":
        stats["entries"] = stats.get("entries", 0) + 1
    return stats


# ─────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────

class SignalLog:
    """Owns `signals.jsonl` + `signal_day_stats.json` under `data_dir`.

    A standalone, purely-additive collaborator — mirrors
    agent.dpof.ledger.DPOFLedger's locking/atomic-write conventions but is
    NEVER consulted by any gate/decision path; engine.py only ever WRITES to
    it, after the fact. Readers (readmodel.py, Mission Control) read the
    files directly, same as they already do for positions.json/
    decisions.jsonl — no lock needed on the read side since every write is
    atomic (JSONL append+fsync; day-stats tmp+fsync+os.replace)."""

    def __init__(self, data_dir: str, clock: Optional[Callable[[], datetime]] = None) -> None:
        self.data_dir = data_dir
        self.near_miss = NearMissLogger()
        # Line count of signals.jsonl, counted once on the first write and
        # kept in memory after that (the engine is the only writer), so a
        # write doesn't re-read an up-to-cap file every cycle.
        self._line_count: Optional[int] = None
        # Names the trim archives (signals_<YYYYMMDD_HHMMSS>.jsonl.gz).
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(_ET))

    # ------------------------------------------------------------------
    # Locking / executor plumbing (mirrors DPOFLedger._with_lock/_run_locked)
    # ------------------------------------------------------------------

    def _lock_path(self) -> str:
        return os.path.join(self.data_dir, _LOCK_FILE)

    def _with_lock(self, fn):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self._lock_path(), "a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                return fn()
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    async def _run_locked(self, fn):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._with_lock(fn))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def write_cycle(self, records: list[dict], now_et: datetime, max_lines: Optional[int] = None) -> dict:
        """Appends every record (one JSONL line each) to signals.jsonl and
        folds them into today's day-stats file, all under ONE flock
        acquisition. Returns the updated day-stats dict (or {} on a no-op /
        failure). Never raises — callers (engine.py) already wrap this call
        in their own try/except per the fire-and-forget/defense-in-depth
        convention used throughout agent/dpof; this additionally degrades to
        a logged error + no-op rather than ever letting an observability
        write affect the cycle it's observing."""
        if not records:
            return {}
        day = now_et.strftime("%Y-%m-%d")
        cap = max(1, int(max_lines or DEFAULT_MAX_LINES))

        def _do() -> dict:
            os.makedirs(self.data_dir, exist_ok=True)
            signals_path = os.path.join(self.data_dir, _SIGNALS_FILE)
            lines = [json.dumps(r, default=str) for r in records]
            if self._line_count is None:
                self._line_count = self._count_lines(signals_path)
            with open(signals_path, "a") as f:
                f.write("\n".join(lines) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._line_count += len(lines)
            if self._line_count > cap:
                # Trim to 90% of the cap so the next trim is ~cap/10 writes away.
                archive_dir = (
                    os.path.join(self.data_dir, _ARCHIVE_DIR) if signal_archive_enabled() else None
                )
                kept = self._trim_signals_if_needed(
                    signals_path, cap, keep=max(1, (cap * 9) // 10),
                    archive_dir=archive_dir, clock=self._clock,
                )
                if kept is not None:
                    self._line_count = kept

            stats_path = os.path.join(self.data_dir, _DAY_STATS_FILE)
            stats = self._load_day_stats(stats_path)
            if stats.get("day") != day:
                stats = _default_day_stats(day)
            stats["cycles_evaluated"] = stats.get("cycles_evaluated", 0) + 1
            for r in records:
                stats = fold_record_into_day_stats(stats, r)
            self._atomic_write_json(stats_path, stats)
            return stats

        try:
            return await self._run_locked(_do)
        except Exception as e:
            logger.error(f"DPOF signal_log: write_cycle failed: {type(e).__name__}: {e}")
            return {}

    @staticmethod
    def _load_day_stats(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                raw = f.read()
            if not raw.strip():
                return {}
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.error(f"DPOF signal_log: corrupt day stats at {path}: {e}; starting fresh")
            return {}

    @staticmethod
    def _atomic_write_json(path: str, data: dict) -> None:
        text = json.dumps(data, default=str, indent=2)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @staticmethod
    def _count_lines(path: str) -> int:
        try:
            with open(path, "rb") as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0
        except OSError as e:
            logger.error(f"DPOF signal_log: failed to count lines in {path}: {e}")
            return 0

    @staticmethod
    def _archive_removed(archive_dir: str, clock: Callable[[], datetime], removed: list[str]) -> None:
        """Writes the trimmed head (byte-for-byte) to a new gzip archive file
        `signals_<YYYYMMDD_HHMMSS>.jsonl.gz` via tmp + fsync + os.replace.
        Never raises: any failure is logged as a WARNING (type only, no data)
        and the caller still performs the trim."""
        try:
            os.makedirs(archive_dir, exist_ok=True)
            base = f"signals_{clock():%Y%m%d_%H%M%S}"
            name = f"{base}.jsonl.gz"
            n = 1
            while os.path.exists(os.path.join(archive_dir, name)):
                n += 1
                name = f"{base}_{n}.jsonl.gz"
            final_path = os.path.join(archive_dir, name)
            tmp_path = f"{final_path}.tmp"
            with gzip.open(tmp_path, "wb") as gz:
                gz.write("".join(removed).encode("utf-8"))
            fd = os.open(tmp_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp_path, final_path)
        except Exception as e:
            logger.warning(
                f"DPOF signal_log: archive of trimmed signals rows failed "
                f"({type(e).__name__}); trimming anyway"
            )

    @staticmethod
    def _trim_signals_if_needed(
        path: str,
        max_lines: int,
        keep: Optional[int] = None,
        archive_dir: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> Optional[int]:
        """Keeps the newest `keep` lines (default `max_lines`) once the file
        holds more than `max_lines`. Returns the resulting line count, or None
        if the file couldn't be read. Only called when the in-memory count
        says the cap was crossed (see write_cycle). When `archive_dir` is set
        the removed head is archived BEFORE the replace; an archive failure
        never prevents the trim."""
        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except OSError as e:
            logger.error(f"DPOF signal_log: failed to read {path} for trim check: {e}")
            return None
        if len(lines) <= max_lines:
            return len(lines)
        kept = lines[-(keep or max_lines):]
        if archive_dir is not None:
            removed = lines[: len(lines) - len(kept)]
            SignalLog._archive_removed(archive_dir, clock or (lambda: datetime.now(_ET)), removed)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            f.writelines(kept)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        logger.info(f"DPOF signal_log: trimmed {path} from {len(lines)} to {len(kept)} lines")
        return len(kept)
