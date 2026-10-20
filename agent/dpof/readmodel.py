"""
CLIO DPOF — read-only view of DPOF state for the rest of CLIO.

The single CLIO-native source of "what are we holding / what did we trade"
for the system prompt, scheduled reports, LLM tools, email summaries,
learning-engine holdings context and Mission Control (a separate process).

Contract:
- Sync, read-only, NEVER raises (every public function degrades to an empty
  value and logs at DEBUG). Safe to call from any thread or process.
- No locks: the ledger writes positions.json / risk_state.json with
  tmp + fsync + os.replace, so a reader always sees a complete file.
- Parsed files are cached by (mtime_ns, size) so per-turn callers (system
  prompt) cost one os.stat() when nothing changed.
- Live marks live only in the engine's memory; unrealized P&L comes from the
  CLIO Alpaca account, not from here.

The only writes are the operator HALT sentinel helpers (`write_halt_file` /
`clear_halt_file`), which Mission Control uses to stop/allow new DPOF entries
from its own process — the engine checks that file every cycle.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from agent.dpof.models import PILLAR_NAMES, PositionRecord, RiskState
from agent.dpof.trade_stats import compute_trade_stats, is_real_trade, render_trade_stats

logger = logging.getLogger("clio.dpof.readmodel")

ET = ZoneInfo("America/New_York")

_POSITIONS_FILE = "positions.json"
_RISK_FILE = "risk_state.json"
_DECISIONS_FILE = "decisions.jsonl"
_HALT_FILE = "HALT"
_OPEN_STATUSES = ("pending", "open", "closing")

# Observability (Task ORCH-7Q4K-DPOF, 2026-09-25) — written by
# agent.dpof.signal_log.SignalLog; local constants here (not imported from
# there) match this module's existing convention of duplicating the ledger's
# own file-name constants above, so a reader never needs a writer's module.
_SIGNALS_FILE = "signals.jsonl"
_DAY_STATS_FILE = "signal_day_stats.json"

# Learning loop F1 (Task ORCH-7Q4K-DPOF builder B2, 2026-09-26) — written by
# agent.dpof.reflection.LessonStore. Same "duplicate the writer's file-name
# constant, never import its module" convention as signals.jsonl above.
_LESSONS_FILE = "lessons.jsonl"

# Follow-ups F2 (Task ORCH-7Q4K-DPOF builder B2, docs/design/
# clio_dpof_followups_2026-09-26.md §4, 2026-09-26) — written by
# agent.dpof.ledger.DPOFLedger._archive_old_closed. Same "duplicate the
# writer's file-name constant, never import its module" convention as above.
_ARCHIVE_FILE = "positions_archive.jsonl"

_OCC_RE = re.compile(r"^([A-Z.]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")

_cache_lock = threading.Lock()
_cache: dict[str, tuple[tuple[int, int, int], Any]] = {}


def data_dir() -> str:
    """DPOF data dir, honoring CLIO_DPOF_DATA_DIR exactly like the engine."""
    try:
        from agent.dpof.config import load_dpof_config

        return load_dpof_config().data_dir
    except Exception as e:  # config import must never break a reader
        logger.debug(f"readmodel: config unavailable ({type(e).__name__}: {e})")
        from agent.dpof.config import DEFAULT_DATA_DIR

        return DEFAULT_DATA_DIR


def _read_json_cached(path: str) -> Any:
    try:
        st = os.stat(path)
    except OSError:
        return None
    # Inode too: writers replace the file atomically (new inode per write), and two writes of the
    # same size can share an mtime tick -- (mtime, size) alone then serves the stale content.
    key = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _cache_lock:
        hit = _cache.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.debug(f"readmodel: unreadable {path} ({type(e).__name__}: {e})")
        return None
    with _cache_lock:
        _cache[path] = (key, data)
    return data


# ---------------------------------------------------------------------------
# Positions / risk
# ---------------------------------------------------------------------------


def load_positions(ddir: Optional[str] = None) -> list[PositionRecord]:
    """Every non-archived position (open and recently closed)."""
    raw = _read_json_cached(os.path.join(ddir or data_dir(), _POSITIONS_FILE))
    items = raw.get("positions", []) if isinstance(raw, dict) else []
    out: list[PositionRecord] = []
    for d in items:
        if not isinstance(d, dict):
            continue
        try:
            rec = PositionRecord.from_dict(d)
        except Exception as e:
            logger.debug(f"readmodel: skipping unparsable position ({type(e).__name__}: {e})")
            continue
        if rec.id:
            out.append(rec)
    return out


def open_positions(ddir: Optional[str] = None) -> list[PositionRecord]:
    return [p for p in load_positions(ddir) if p.status in _OPEN_STATUSES]


def open_underlyings(ddir: Optional[str] = None) -> list[str]:
    return sorted({p.underlying for p in open_positions(ddir) if p.underlying})


def _et_day_bounds(now_epoch: Optional[float] = None) -> tuple[float, float]:
    now_et = datetime.fromtimestamp(now_epoch if now_epoch is not None else time.time(), tz=ET)
    start = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp(), start.timestamp() + 86400.0


def closed_between(start_epoch: float, end_epoch: float, ddir: Optional[str] = None) -> list[PositionRecord]:
    return [
        p for p in load_positions(ddir)
        if p.status == "closed" and p.closed_time is not None and start_epoch <= p.closed_time < end_epoch
    ]


def closed_today(now_epoch: Optional[float] = None, ddir: Optional[str] = None) -> list[PositionRecord]:
    start, end = _et_day_bounds(now_epoch)
    return closed_between(start, end, ddir)


def opened_today(now_epoch: Optional[float] = None, ddir: Optional[str] = None) -> list[PositionRecord]:
    start, end = _et_day_bounds(now_epoch)
    return [p for p in load_positions(ddir) if p.entry_time is not None and start <= p.entry_time < end]


def load_risk(ddir: Optional[str] = None) -> RiskState:
    raw = _read_json_cached(os.path.join(ddir or data_dir(), _RISK_FILE))
    if not isinstance(raw, dict) or not raw:
        return RiskState()
    try:
        return RiskState.from_dict(raw)
    except Exception as e:
        logger.debug(f"readmodel: unparsable risk state ({type(e).__name__}: {e})")
        return RiskState()


def recent_decisions(limit: int = 20, ddir: Optional[str] = None) -> list[dict]:
    """Last `limit` decision records (newest last). Reads only the file tail."""
    path = os.path.join(ddir or data_dir(), _DECISIONS_FILE)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    out: list[dict] = []
    for line in tail.splitlines()[-limit:]:
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # first line of the tail window may be partial
        if isinstance(rec, dict):
            out.append(rec)
    return out


def position_reviews(position_id: str, ddir: Optional[str] = None) -> list[dict]:
    """Every `kind="position_review"` decision row for `position_id`,
    oldest-first. Streams the WHOLE decisions.jsonl (bounded by the ledger's
    own cap) rather than a byte tail: a reflection that runs days after the
    close (backlog after an outage) still gets the position's full review
    trail. Cheap substring prefilter before json.loads. Never raises."""
    if not position_id:
        return []
    path = os.path.join(ddir or data_dir(), _DECISIONS_FILE)
    needle = str(position_id)
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if needle not in line or "position_review" not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("kind") == "position_review" and rec.get("position_id") == position_id:
                    out.append(rec)
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# Observability (Task ORCH-7Q4K-DPOF, 2026-09-25) — signals.jsonl +
# signal_day_stats.json, written by agent.dpof.signal_log.SignalLog. Read-only,
# same "never raises, degrades to empty" contract as the rest of this module.
# Feeds `/dpof why` (agent/dpof/telegram_commands.py) and the EOD digest
# (agent/integrations/scheduled_tasks.py) — and, per the same contract, is
# safe for Mission Control (a separate process) to call directly instead of
# going through the engine.
# ---------------------------------------------------------------------------


def signal_day_stats(ddir: Optional[str] = None) -> dict:
    """Today's (per the day-stats file's own 'day' key -- NOT necessarily
    "today" if DPOF hasn't run yet since midnight ET) signal-pipeline rollup.
    {} when nothing has been recorded yet."""
    raw = _read_json_cached(os.path.join(ddir or data_dir(), _DAY_STATS_FILE))
    return raw if isinstance(raw, dict) else {}


def _tail_signal_records(ddir: Optional[str] = None, byte_window: int = 512 * 1024) -> list[dict]:
    """Every parsed signals.jsonl record within the last `byte_window` bytes
    of the file, oldest-first (matches the file's own append order) -- about
    3 hours of records at 2 underlyings / 30 s cycle (~680 B/record), plenty
    to derive "current outcome per underlying" and "recent outcome changes"
    without reading the
    whole (capped, but still potentially multi-MB) file on every call."""
    path = os.path.join(ddir or data_dir(), _SIGNALS_FILE)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - byte_window))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    out: list[dict] = []
    for line in tail.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # first line of the tail window may be partial
        if isinstance(rec, dict):
            out.append(rec)
    return out


def recent_signal_log(limit: int = 50, ddir: Optional[str] = None) -> list[dict]:
    """Last `limit` signals.jsonl records (newest last)."""
    return _tail_signal_records(ddir)[-limit:]


def _latest_per_underlying(records: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for rec in records:
        sym = rec.get("underlying")
        if sym:
            latest[sym] = rec  # records are oldest-first -> last write wins
    return latest


def _recent_outcome_changes(records: list[dict], limit: int = 5) -> list[dict]:
    """Every point, walking the records chronologically, where an
    underlying's outcome differs from its own immediately-PRIOR record --
    i.e. transitions, not just "the last N records" (which would mostly be
    repeats of an unchanged outcome under near_miss_log_minutes throttling
    upstream, and signals.jsonl itself has none — every cycle writes a line
    regardless). Returns the most recent `limit`, newest last."""
    last_outcome: dict[str, str] = {}
    changes: list[dict] = []
    for rec in records:
        sym = rec.get("underlying")
        outcome = rec.get("outcome")
        if not sym or outcome is None:
            continue
        prev = last_outcome.get(sym)
        if prev is not None and prev != outcome:
            changes.append({"ts": rec.get("ts"), "underlying": sym, "from": prev, "to": outcome})
        last_outcome[sym] = outcome
    return changes[-limit:]


def _fmt_ts(iso_ts: Any) -> str:
    """ISO timestamp -> 'HH:MM ET', tolerant of anything malformed/missing."""
    if not iso_ts:
        return "?"
    try:
        dt = datetime.fromisoformat(str(iso_ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        return dt.astimezone(ET).strftime("%H:%M ET")
    except (ValueError, TypeError):
        return str(iso_ts)


def _format_signal_line(rec: dict) -> str:
    sym = rec.get("underlying", "?")
    outcome = rec.get("outcome", "unknown")
    composite = rec.get("composite")
    comp_str = f"{composite:+.2f}" if isinstance(composite, (int, float)) else "n/a"
    strength = rec.get("strength", "none")
    pillars = rec.get("pillars") or {}
    pillar_bits = []
    for name in PILLAR_NAMES:
        p = pillars.get(name) or {}
        v = p.get("value")
        pillar_bits.append(f"{name} {v:+.2f}" if isinstance(v, (int, float)) else f"{name} n/a")
    flow = rec.get("flow_source", "unavailable")
    return (
        f"{sym}: {outcome} (composite {comp_str}, strength={strength}; "
        + " ".join(pillar_bits) + f"; flow {flow})"
    )


def _today_et() -> str:
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def dpof_why_text(ddir: Optional[str] = None, today: Optional[str] = None) -> str:
    """`/dpof why` body: today's day stats, the current outcome + blocking
    reason per configured underlying, and the last ~5 outcome CHANGES.
    Never raises -- degrades to an explanatory line when no signal data has
    been recorded yet (e.g. DPOF just started, or is OFF/market closed and
    never reached signal evaluation this process)."""
    try:
        d = ddir or data_dir()
        stats = signal_day_stats(d)
        records = _tail_signal_records(d)

        lines = []
        today = today or _today_et()
        day = stats.get("day") if stats else None
        lines.append(f"DPOF why{f' — {day} (ET)' if day else ''}")
        if stats and day != today:
            lines.append(f"Today ({today}): no signal cycles recorded yet — the figures below are from {day}.")
        if stats:
            lines.append(
                f"{'Today' if day == today else day}: {stats.get('cycles_evaluated', 0)} cycles evaluated, "
                f"{stats.get('nominations', 0)} nomination(s), decider "
                f"enter={stats.get('decider_enter', 0)} pass={stats.get('decider_pass', 0)} "
                f"unavailable={stats.get('decider_unavailable', 0)}, "
                f"jev vetoes={stats.get('jev_vetoes', 0)}, entries={stats.get('entries', 0)}"
            )
        else:
            lines.append("Today: no signal data recorded yet.")

        lines.append("")
        lines.append("Current per underlying:")
        latest = _latest_per_underlying(records)
        if not latest:
            lines.append("  (no signal-log records yet — has a cycle run since the last restart?)")
        else:
            for sym in sorted(latest):
                lines.append("  " + _format_signal_line(latest[sym]))

        changes = _recent_outcome_changes(records, limit=5)
        lines.append("")
        lines.append("Recent outcome changes:")
        if not changes:
            lines.append("  (none recorded in the current log window)")
        else:
            for c in changes:
                lines.append(f"  {_fmt_ts(c['ts'])} {c['underlying']}: {c['from']} -> {c['to']}")

        return "\n".join(lines)
    except Exception as e:  # never raise into the Telegram handler
        logger.debug(f"readmodel: dpof_why_text failed: {type(e).__name__}: {e}")
        return "DPOF why: unavailable (could not read signal log)."


def split_for_telegram(text: str, limit: int = 4000) -> list[str]:
    """Split plain text into Telegram-sized chunks (4096 max), on newlines
    where possible."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{cur}\n{line}" if cur else line
        if len(candidate) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Learning loop F1 — lessons.jsonl (Task ORCH-7Q4K-DPOF builder B2,
# 2026-09-26). Read-only, same "never raises, degrades to empty" contract as
# the rest of this module. Feeds `/dpof lessons` (agent/dpof/telegram_commands.py)
# and the EOD digest's "Lessons today" section below.
# ---------------------------------------------------------------------------


def _tail_lesson_records(ddir: Optional[str] = None, byte_window: int = 512 * 1024) -> list[dict]:
    """Every parsed lessons.jsonl record within the last `byte_window` bytes,
    oldest-first (matches the file's own append order)."""
    path = os.path.join(ddir or data_dir(), _LESSONS_FILE)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - byte_window))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    out: list[dict] = []
    for line in tail.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # first line of the tail window may be partial
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _lessons_tally(records: list[dict]) -> dict:
    """{underlying: {n, wins, losses, net_pnl_usd}} over `records`. Never a
    fake win/loss for an unusable pnl_usd (isinstance-checked, bool-excluded)."""
    tally: dict[str, dict] = {}
    for rec in records:
        sym = rec.get("underlying")
        if not sym:
            continue
        bucket = tally.setdefault(sym, {"n": 0, "wins": 0, "losses": 0, "net_pnl_usd": 0.0})
        bucket["n"] += 1
        pnl = rec.get("pnl_usd")
        if isinstance(pnl, (int, float)) and not isinstance(pnl, bool):
            bucket["net_pnl_usd"] += pnl
            if pnl > 0:
                bucket["wins"] += 1
            else:
                bucket["losses"] += 1
    return tally


def _format_lesson_line(rec: dict) -> str:
    sym = rec.get("underlying") or "?"
    direction = rec.get("direction") or "?"
    pnl_usd = rec.get("pnl_usd")
    pnl_pct = rec.get("pnl_pct")
    if isinstance(pnl_usd, (int, float)) and not isinstance(pnl_usd, bool):
        sign = "+" if pnl_usd >= 0 else "-"
        pnl_str = f"{sign}${abs(pnl_usd):,.0f}"
    else:
        pnl_str = "n/a"
    pct_str = f"{pnl_pct:+.0%}" if isinstance(pnl_pct, (int, float)) and not isinstance(pnl_pct, bool) else "n/a"
    key_factor = rec.get("key_factor") or "other"
    lesson = str(rec.get("lesson") or "").strip()
    return f"{sym} {direction} {pnl_str} ({pct_str}) key:{key_factor} — {lesson}"


def _lesson_day_et(rec: dict) -> Optional[str]:
    """closed_at parsed + converted to an ET 'YYYY-MM-DD' day key. None if
    missing/unparsable (never raises)."""
    raw = rec.get("closed_at")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        return dt.astimezone(ET).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def lessons_text(limit: int = 10, ddir: Optional[str] = None) -> str:
    """`/dpof lessons` body: the last `limit` stored lessons (newest first)
    plus a per-underlying tally over the returned read window. Never raises
    -- degrades to an explanatory line when nothing has been recorded yet."""
    try:
        records = _tail_lesson_records(ddir)
        if not records:
            return "DPOF lessons: none recorded yet."

        newest_first = list(reversed(records))
        shown = newest_first[: max(0, limit)]

        lines = [f"DPOF lessons (showing {len(shown)} of {len(records)}):"]
        for rec in shown:
            lines.append(f"- {_fmt_ts(rec.get('closed_at'))} {_format_lesson_line(rec)}")

        tally = _lessons_tally(records)
        if tally:
            lines.append("")
            lines.append("Tally (read window):")
            for sym in sorted(tally):
                t = tally[sym]
                lines.append(
                    f"  {sym}: n={t['n']} wins={t['wins']} losses={t['losses']} net=${t['net_pnl_usd']:+,.0f}"
                )
        return "\n".join(lines)
    except Exception as e:
        logger.debug(f"readmodel: lessons_text failed: {type(e).__name__}: {e}")
        return "DPOF lessons: unavailable (could not read lessons log)."


# ---------------------------------------------------------------------------
# Follow-ups F2 — trade-quality statistics (Task ORCH-7Q4K-DPOF builder B2,
# docs/design/clio_dpof_followups_2026-09-26.md §4, 2026-09-26). Read-only,
# same "never raises, degrades to empty" contract as the rest of this module.
# Feeds `/dpof stats` (agent/dpof/telegram_commands.py) and the EOD digest's
# "Trades" section below.
# ---------------------------------------------------------------------------


def load_archived_positions(ddir: Optional[str] = None) -> list[PositionRecord]:
    """Every closed/cancelled position ledger.py archived out of
    positions.json (positions_archive.jsonl -- one PositionRecord.to_dict()
    JSON per line, appended BEFORE positions.json is rewritten:
    ledger.py:172-199). Streaming line read; unparsable/idless lines are
    skipped, never raises."""
    path = os.path.join(ddir or data_dir(), _ARCHIVE_FILE)
    out: list[PositionRecord] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue  # a reader can meet a partial last line (crash mid-append)
                if not isinstance(d, dict):
                    continue
                try:
                    rec = PositionRecord.from_dict(d)
                except Exception as e:
                    logger.debug(f"readmodel: skipping unparsable archived position ({type(e).__name__}: {e})")
                    continue
                if rec.id:
                    out.append(rec)
    except OSError:
        return out
    return out


def _all_history_positions(ddir: Optional[str] = None) -> list[PositionRecord]:
    """positions.json + positions_archive.jsonl, deduped by id -- the
    positions.json record wins (R4: a crash can leave a record archived AND
    still in positions.json; positions.json is the more-recently-written,
    authoritative copy)."""
    merged: dict[str, PositionRecord] = {p.id: p for p in load_archived_positions(ddir) if p.id}
    merged.update({p.id: p for p in load_positions(ddir) if p.id})
    return list(merged.values())


# ---------------------------------------------------------------------------
# Kronos forecast (Task K2, agent/kronos -- a separate CLIO service; see
# docs/design/clio_kronos.md §4). Feeds `/dpof kronos`
# (agent/dpof/telegram_commands.py) and the EOD digest's one-line summary
# below. Read-only, filesystem-only (agent.kronos.state); never touches the
# live engine. Never imports agent.kronos.state at module level (mirrors this
# module's own "duplicate the writer's constants, don't import its module"
# convention isn't applicable here since state.py IS the read contract --
# but the import still stays lazy/inside the function so a missing
# agent.kronos package degrades to "unavailable" text instead of breaking
# this whole module's import).
# ---------------------------------------------------------------------------


def _kronos_underlyings_and_max_age() -> tuple[tuple[str, ...], float]:
    try:
        from agent.dpof.config import load_dpof_config

        cfg = load_dpof_config()
        return cfg.underlyings, cfg.kronos_max_age_seconds
    except Exception as e:
        logger.debug(f"readmodel: kronos config unavailable ({type(e).__name__}: {e})")
        return ("SPY", "QQQ"), 600.0


def kronos_text(now_epoch: Optional[float] = None) -> str:
    """`/dpof kronos` body: the latest Kronos forecast per configured
    underlying -- asof bar, per-horizon p_up/expected return/expected |move|,
    and the scored track record (n, hit rate). Never raises; missing/stale
    says so explicitly, never a zeroed/faked reading (CLAUDE.md "no fake
    zeros")."""
    underlyings, max_age = _kronos_underlyings_and_max_age()
    try:
        from agent.kronos import state as kronos_state
    except Exception as e:
        logger.debug(f"readmodel: agent.kronos.state unavailable ({type(e).__name__}: {e})")
        return "Kronos forecast: unavailable (agent.kronos is not installed)."

    try:
        forecast = kronos_state.read_forecast(now=now_epoch, max_age_seconds=max_age)
        scores = kronos_state.read_scores()
    except Exception as e:
        logger.debug(f"readmodel: kronos_text read failed ({type(e).__name__}: {e})")
        return "Kronos forecast: unavailable (read failed)."

    scores_by_symbol = scores.get("symbols") if isinstance(scores, dict) else None
    if not isinstance(scores_by_symbol, dict):
        scores_by_symbol = {}

    lines = ["Kronos forecast (advisory, candlestick model):"]
    any_ok = False
    for sym in underlyings:
        entry = kronos_state.symbol_forecast(forecast, sym)
        if entry is None:
            lines.append(f"{sym}: unavailable (no fresh forecast)")
            continue
        any_ok = True
        asof = entry.get("asof_bar") or "n/a"
        lines.append(f"{sym} (as of {asof}):")
        horizons = entry.get("horizons") if isinstance(entry.get("horizons"), dict) else {}
        sym_scores = scores_by_symbol.get(sym) if isinstance(scores_by_symbol.get(sym), dict) else {}
        for h in kronos_state.HORIZONS:
            hz = horizons.get(h)
            if not isinstance(hz, dict):
                continue
            p_up = hz.get("p_up")
            mean_r = hz.get("mean_return_pct")
            abs_move = hz.get("expected_abs_move_pct")
            p_up_txt = f"{p_up:.0%}" if isinstance(p_up, (int, float)) and not isinstance(p_up, bool) else "n/a"
            mean_txt = f"{mean_r:+.2f}%" if isinstance(mean_r, (int, float)) and not isinstance(mean_r, bool) else "n/a"
            abs_txt = f"{abs_move:.2f}%" if isinstance(abs_move, (int, float)) and not isinstance(abs_move, bool) else "n/a"
            track_txt = "n=0"
            hz_scores = sym_scores.get(h) if isinstance(sym_scores.get(h), dict) else None
            if hz_scores is not None:
                allm = hz_scores.get("all") if isinstance(hz_scores.get("all"), dict) else {}
                n = allm.get("n")
                if isinstance(n, (int, float)) and not isinstance(n, bool) and n > 0:
                    hr = allm.get("hit_rate")
                    hr_txt = f"{hr:.0%}" if isinstance(hr, (int, float)) and not isinstance(hr, bool) else "n/a"
                    track_txt = f"n={int(n)} hit={hr_txt}"
            lines.append(f"  {h}min: p_up={p_up_txt} mean={mean_txt} expected|move|={abs_txt} track(all)={track_txt}")
    if not any_ok:
        lines.append("(no fresh forecasts for any configured underlying)")
    return "\n".join(lines)


def _kronos_digest_line() -> Optional[str]:
    """One-line Kronos summary for the EOD digest -- omitted (None) unless
    kronos_scores.json has a scored 30-min track record for at least one
    configured underlying. Read-only, filesystem-only; never raises (any
    problem degrades to None, which the caller treats as "omit the line")."""
    underlyings, _max_age = _kronos_underlyings_and_max_age()
    try:
        from agent.kronos import state as kronos_state

        scores = kronos_state.read_scores()
    except Exception as e:
        logger.debug(f"readmodel: kronos digest scores unavailable ({type(e).__name__}: {e})")
        return None
    scores_by_symbol = scores.get("symbols") if isinstance(scores, dict) else None
    if not isinstance(scores_by_symbol, dict):
        return None

    bits = []
    for sym in underlyings:
        sym_scores = scores_by_symbol.get(sym)
        hz = sym_scores.get("30") if isinstance(sym_scores, dict) else None
        allm = hz.get("all") if isinstance(hz, dict) else None
        if not isinstance(allm, dict):
            continue
        n = allm.get("n")
        if not isinstance(n, (int, float)) or isinstance(n, bool) or n <= 0:
            continue
        hr = allm.get("hit_rate")
        hr_txt = f"{hr:.0%}" if isinstance(hr, (int, float)) and not isinstance(hr, bool) else "n/a"
        bits.append(f"{sym} 30m hit {hr_txt} (n={int(n)})")
    if not bits:
        return None
    return "Kronos: " + "; ".join(bits)


def trade_stats_text(ddir: Optional[str] = None, now_epoch: Optional[float] = None) -> str:
    """`/dpof stats` body and the EOD digest's "Trades" section: today's
    trade-quality stats plus all-history stats (positions.json +
    positions_archive.jsonl, deduped by id). Never raises -- degrades to a
    single "unavailable" line."""
    try:
        d = ddir or data_dir()
        today_stats = compute_trade_stats(closed_today(now_epoch, d))

        all_positions = _all_history_positions(d)
        all_stats = compute_trade_stats(all_positions)
        earliest = min(
            (p.closed_time for p in all_positions if is_real_trade(p) and p.closed_time is not None),
            default=None,
        )
        since_label = (
            "since " + datetime.fromtimestamp(earliest, tz=ET).strftime("%Y-%m-%d")
            if earliest is not None
            else "all-time"
        )

        return "\n".join([
            render_trade_stats("today", today_stats),
            render_trade_stats(since_label, all_stats),
        ])
    except Exception as e:
        logger.debug(f"readmodel: trade_stats_text failed: {type(e).__name__}: {e}")
        return "Trades: unavailable"


def eod_digest_text(ddir: Optional[str] = None, today: Optional[str] = None) -> str:
    """End-of-day digest body — the automatic answer to "why didn't CLIO
    trade today" (or a summary of what it did), sent once after the close by
    agent.integrations.scheduled_tasks' dpof_eod_digest job. Never raises."""
    try:
        stats = signal_day_stats(ddir)
        today = today or _today_et()
        if not stats or stats.get("day") != today:
            last = f" (last recorded day: {stats.get('day')})" if stats and stats.get("day") else ""
            return (
                f"DPOF EOD digest — {today} (ET): no signal cycles were recorded today{last}. "
                "DPOF did not evaluate the market today; check /dpof status (mode, halt, engine errors)."
            )

        day = stats.get("day", "")
        lines = [f"DPOF EOD digest — {day} (ET)" if day else "DPOF EOD digest"]
        lines.append(
            f"Entries: {stats.get('entries', 0)} | Nominations: {stats.get('nominations', 0)} | "
            f"Decider: enter={stats.get('decider_enter', 0)} pass={stats.get('decider_pass', 0)} "
            f"unavailable={stats.get('decider_unavailable', 0)} | Jev vetoes: {stats.get('jev_vetoes', 0)} | "
            f"Cycles evaluated: {stats.get('cycles_evaluated', 0)}"
        )

        # Follow-ups F2 (Task ORCH-7Q4K-DPOF builder B2, 2026-09-26): its own
        # try/except -- trade_stats_text() itself never raises, but this is a
        # second, defense-in-depth layer so the "Trades" section can NEVER
        # fail the rest of the digest.
        try:
            # Window "today" on the digest's own day (an explicit `today`
            # argument included) -- noon ET of that day.
            day_noon = datetime.strptime(today, "%Y-%m-%d").replace(hour=12, tzinfo=ET).timestamp()
            trades_text = trade_stats_text(ddir, now_epoch=day_noon)
        except Exception as e:
            logger.debug(f"readmodel: eod_digest_text trade stats failed: {type(e).__name__}: {e}")
            trades_text = "Trades: unavailable"
        lines.append("")
        lines.append(trades_text)

        underlyings = stats.get("underlyings") or {}
        for sym in sorted(underlyings):
            u = underlyings[sym] or {}
            mac = u.get("max_abs_composite")
            mac_str = f"{mac:+.2f}" if isinstance(mac, (int, float)) else "n/a"
            mac_dir = u.get("max_abs_composite_direction") or "n/a"
            mac_at = _fmt_ts(u.get("max_abs_composite_at"))
            lines.append("")
            lines.append(f"{sym}: max |composite| {mac_str} ({mac_dir}) at {mac_at}")

            counts = u.get("outcome_counts") or {}
            top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
            if top:
                lines.append("  top outcomes: " + ", ".join(f"{k}={v}" for k, v in top))

            pa = u.get("pillar_available") or {}
            pt = u.get("pillar_total") or {}
            pct_bits = []
            for name in PILLAR_NAMES:
                total = pt.get(name, 0) or 0
                avail = pa.get(name, 0) or 0
                pct = (avail / total * 100.0) if total else 0.0
                pct_bits.append(f"{name} {pct:.0f}%")
            lines.append("  pillar availability: " + " ".join(pct_bits))

            fs = u.get("flow_source_counts") or {}
            if fs:
                lines.append("  flow source: " + ", ".join(f"{k}={v}" for k, v in sorted(fs.items())))

        # Task K2 (agent/kronos): its own try/except, defense-in-depth same
        # as the Trades section above -- appended ONLY when kronos_scores.json
        # has a scored record for at least one configured underlying, so a
        # digest from before this feature (or a day kronos was never running)
        # renders identically.
        try:
            kronos_line = _kronos_digest_line()
        except Exception as e:
            logger.debug(f"readmodel: eod_digest_text kronos line failed: {type(e).__name__}: {e}")
            kronos_line = None
        if kronos_line:
            lines.append("")
            lines.append(kronos_line)

        # Learning loop F1 (Task ORCH-7Q4K-DPOF builder B2, 2026-09-26):
        # appended ONLY when there are lessons dated today -- an old digest
        # with no lessons.jsonl (or none closed today) renders identically
        # to before this feature existed.
        try:
            todays_lessons = [r for r in _tail_lesson_records(ddir) if _lesson_day_et(r) == today]
        except Exception as e:
            logger.debug(f"readmodel: eod_digest_text lessons lookup failed: {type(e).__name__}: {e}")
            todays_lessons = []
        if todays_lessons:
            lines.append("")
            lines.append(f"Lessons today: {len(todays_lessons)}")
            for rec in todays_lessons:
                lines.append(("  " + _format_lesson_line(rec))[:160])

        return "\n".join(lines)
    except Exception as e:
        logger.debug(f"readmodel: eod_digest_text failed: {type(e).__name__}: {e}")
        return "DPOF EOD digest: unavailable (could not read signal log)."


# ---------------------------------------------------------------------------
# Operator HALT sentinel (cross-process entry stop)
# ---------------------------------------------------------------------------


def halt_file_present(ddir: Optional[str] = None) -> bool:
    try:
        return os.path.exists(os.path.join(ddir or data_dir(), _HALT_FILE))
    except OSError:
        return False


def write_halt_file(reason: str, ddir: Optional[str] = None) -> bool:
    """Create data/dpof/HALT (blocks new DPOF entries next cycle). Returns success."""
    d = ddir or data_dir()
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, _HALT_FILE)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps({"reason": reason, "at": datetime.now(timezone.utc).isoformat()}))
        os.replace(tmp, path)
        return True
    except OSError as e:
        logger.warning(f"readmodel: could not write HALT file ({type(e).__name__}: {e})")
        return False


def clear_halt_file(ddir: Optional[str] = None) -> bool:
    """Remove data/dpof/HALT. Returns True if it is absent afterwards."""
    path = os.path.join(ddir or data_dir(), _HALT_FILE)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning(f"readmodel: could not remove HALT file ({type(e).__name__}: {e})")
        return False
    return not os.path.exists(path)


# ---------------------------------------------------------------------------
# Formatting (plain text — Telegram/prompt safe, no markup)
# ---------------------------------------------------------------------------


def contract_label(occ: str) -> str:
    """'SPY260925C00580000' -> 'SPY 2026-09-25 580C'. Unknown formats pass through."""
    m = _OCC_RE.match(occ or "")
    if not m:
        return occ or "?"
    root, yy, mm, dd, cp, strike = m.groups()
    k = int(strike) / 1000.0
    k_str = f"{k:g}"
    return f"{root} 20{yy}-{mm}-{dd} {k_str}{cp}"


def position_label(p: PositionRecord) -> str:
    """Human label for a position's contract(s)."""
    legs = [lg for lg in (p.legs or []) if isinstance(lg, dict) and lg.get("symbol")]
    if not legs:
        return f"{p.underlying} {p.direction} {p.structure}".strip()
    if len(legs) == 1:
        return contract_label(legs[0]["symbol"])
    parts = []
    for lg in legs:
        sign = "+" if str(lg.get("side", "")).lower() == "buy" else "-"
        parts.append(f"{sign}{contract_label(lg['symbol'])}")
    return " / ".join(parts)


def format_position_line(p: PositionRecord) -> str:
    entry = f"@{p.entry_price:.2f}" if isinstance(p.entry_price, (int, float)) else (
        f"limit {p.entry_limit_price:.2f}" if isinstance(p.entry_limit_price, (int, float)) else "unfilled"
    )
    opened = ""
    if isinstance(p.entry_time, (int, float)):
        opened = " opened " + datetime.fromtimestamp(p.entry_time, tz=ET).strftime("%m-%d %H:%M ET")
    line = f"{p.underlying} {p.direction} {position_label(p)} x{p.qty_open}/{p.qty_original} {entry} [{p.status}]{opened}"
    if p.status == "closed":
        line += f" realized ${p.realized_pnl:+,.2f}"
        if p.exit_reason:
            line += f" ({p.exit_reason})"
    return line


def positions_brief(max_items: int = 6, ddir: Optional[str] = None) -> str:
    """One compact block for the system prompt; '' when nothing is open."""
    opens = open_positions(ddir)
    if not opens:
        return ""
    lines = [f"DPOF OPEN POSITIONS ({len(opens)}, CLIO's own SPY/QQQ options book):"]
    for p in opens[:max_items]:
        lines.append("- " + format_position_line(p))
    if len(opens) > max_items:
        lines.append(f"- ... and {len(opens) - max_items} more")
    return "\n".join(lines)


def snapshot(now_epoch: Optional[float] = None, ddir: Optional[str] = None) -> dict:
    """JSON-safe summary for reports / Mission Control / email."""
    d = ddir or data_dir()
    risk = load_risk(d)
    opens = open_positions(d)
    closed = closed_today(now_epoch, d)
    mode = "unknown"
    try:
        from agent.dpof.config import load_dpof_config

        mode = load_dpof_config().mode.value
    except Exception as e:
        logger.debug(f"readmodel: mode unavailable ({type(e).__name__}: {e})")
    wins = [p for p in closed if p.realized_pnl > 0]
    return {
        "source": "clio_dpof",
        "mode": mode,
        "halt_file": halt_file_present(d),
        "hard_halted": bool(risk.hard_halted),
        "halt_reason": risk.halt_reason,
        "open_count": len(opens),
        "open_positions": [p.to_dict() for p in opens],
        "open_lines": [format_position_line(p) for p in opens],
        "closed_today_count": len(closed),
        "closed_today_lines": [format_position_line(p) for p in closed],
        "closed_today_wins": len(wins),
        "realized_pnl_today": sum(p.realized_pnl for p in closed),
        "risk_day": risk.day,
        "realized_pnl_week": risk.realized_pnl_week,
        "entries_today": risk.entries_today,
        "last_valid_snapshot": risk.last_valid_snapshot,
    }
