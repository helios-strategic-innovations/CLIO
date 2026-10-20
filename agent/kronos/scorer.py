"""
CLIO Kronos — status-driven scoring (CLAUDE.md anti-pattern #36).

Every forecast horizon is appended, PENDING, to an append-only-in-spirit
JSONL history file (`data/intelligence/kronos_history.jsonl` by default,
`CLIO_KRONOS_HISTORY_FILE`). Each run, BEFORE (or instead of) writing new
pending entries, `resolve_and_score` walks the WHOLE history looking for
any pending entry whose target bar has now closed — discovered by STATUS
(`resolved: false`), never by a fixed "created in the last N minutes"
window, so a run that was skipped/late (model load stall, a missed wake, a
service restart) still resolves and scores every pending entry whose
target time has since passed, using whatever fresh bars this run fetched.
This is the same discovery discipline CLAUDE.md's Intel v3 Gotchas
describe for status-driven work queues.

The history file has exactly ONE writer (this service, one process) so a
plain tempfile + `os.replace` atomic rewrite is sufficient — CLAUDE.md
anti-pattern #22 warns against combining `fcntl.flock` with `os.replace` on
the SAME file; the fix here is simpler still: never take a lock at all,
because there is no second writer to race against. The file is capped at
`cfg.history_max_lines` (oldest entries dropped first) so it never grows
unbounded.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from agent.kronos.bars import Bar

logger = logging.getLogger("clio.kronos.scorer")

_ET = ZoneInfo("America/New_York")

# A resolved bar must close within this many seconds of the entry's
# target_epoch to count as "the" target bar -- one extra bar-width of
# tolerance for a missing/thin print exactly at the target minute (a gap
# bigger than that is left unresolved rather than guessing at a stale
# price; CLAUDE.md "no fake zeros" cousin -- an unresolved entry reports
# null metrics, never a wrong-bar realized return).
_RESOLUTION_TOLERANCE_SECONDS = 2 * 5 * 60

# "recent" = the current session plus this many PRIOR trading sessions
# (state.py docstring: "recent covers the last 5 sessions").
_RECENT_PRIOR_SESSIONS = 4

_CONFIDENT_HIGH = 0.6
_CONFIDENT_LOW = 0.4


def _read_lines(path: str) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        pass
    return out


def _write_lines_atomic(path: str, records: list[dict]) -> None:
    """Single-writer atomic rewrite -- tempfile in the same directory +
    os.replace, no flock (see this module's docstring)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".kronos-history-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = -1
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_forecast(
    history_file: str, max_lines: int, *, symbol: str, anchor_epoch: float, anchor_close: float,
    horizons: dict[str, dict],
) -> None:
    """Appends one PENDING entry per horizon in `horizons` (as
    `agent.kronos.forecaster.build_forecast_horizons` returns it) to the
    history file. No-op when `horizons` is empty (no forecastable horizon
    this run -- CLAUDE.md "never extrapolate overnight": there is nothing
    to score). The file is capped here too (append immediately followed by
    a cap-and-rewrite) so a run that only records, and is never followed by
    a `resolve_and_score` call in the same process lifetime (impossible in
    normal operation, but defensive), still can't grow the file unbounded."""
    if not horizons:
        return
    existing = _read_lines(history_file)
    for label, metrics in horizons.items():
        minutes = int(metrics.get("minutes", 0) or 0)
        existing.append({
            "id": f"{symbol}:{label}:{anchor_epoch}",
            "symbol": symbol,
            "horizon_minutes": minutes,
            "anchor_epoch": anchor_epoch,
            "anchor_close": anchor_close,
            "target_epoch": anchor_epoch + minutes * 60,
            "p_up": metrics.get("p_up"),
            "expected_abs_move_pct": metrics.get("expected_abs_move_pct"),
            "resolved": False,
            "realized_return_pct": None,
            "realized_at": None,
        })
    if len(existing) > max_lines:
        existing = existing[-max_lines:]
    _write_lines_atomic(history_file, existing)


def _find_target_bar(bars: list[Bar], target_epoch: float) -> Optional[Bar]:
    """The bar whose close time is nearest target_epoch, at or after it,
    within `_RESOLUTION_TOLERANCE_SECONDS`. `bars` is ascending by
    `t_close` (agent.kronos.bars.filter_rth_bars's contract)."""
    best: Optional[Bar] = None
    for b in bars:
        if b.t_close < target_epoch:
            continue
        if b.t_close - target_epoch > _RESOLUTION_TOLERANCE_SECONDS:
            break  # bars are ascending -- everything after is even further away
        if best is None or b.t_close < best.t_close:
            best = b
    return best


def _n_sessions_ago(today: _date, sessions: int, holidays: frozenset) -> _date:
    """PURE. Walks back `sessions` trading days (Mon-Fri, not a NYSE
    holiday) from `today` (exclusive of `today` itself). `holidays`: a
    frozenset of `datetime.date` (agent.integrations.market_events.
    MARKET_HOLIDAYS)."""
    d = today
    count = 0
    while count < sessions:
        d = d - timedelta(days=1)
        if d.weekday() < 5 and d not in holidays:
            count += 1
    return d


def _market_holidays() -> frozenset:
    from agent.integrations.market_events import MARKET_HOLIDAYS

    return MARKET_HOLIDAYS


def _compute_metrics(entries: list[dict]) -> dict:
    """PURE. `entries`: resolved history records (resolved=True,
    realized_return_pct not None) already filtered to the desired window
    (all-time or recent). Every field is None when `entries` is empty
    (CLAUDE.md "no fake zeros" -- an unscored metric is null, never 0.0).

    Hit convention (not specified by the design frame -- documented, own
    decision): a forecast is "predicted up" when p_up >= 0.5, "realized up"
    when realized_return_pct > 0.0 (a flat/zero realized return counts as
    NOT up, matching p_up's own tie-handling being resolved before this
    point). `hit` = 1.0 when the two agree, else 0.0. Brier score uses the
    same realized-up indicator as its binary outcome. "confident" entries
    (p_up >= 0.6 or <= 0.4, per state.py's own docstring) get their own
    hit-rate sub-metric."""
    resolved = [
        e for e in entries
        if e.get("resolved") and isinstance(e.get("realized_return_pct"), (int, float))
        and isinstance(e.get("p_up"), (int, float))
    ]
    n = len(resolved)
    if n == 0:
        return {"n": 0, "hit_rate": None, "n_confident": 0, "hit_rate_confident": None,
                "brier": None, "mae_abs_move_pct": None, "since": None}

    hits: list[float] = []
    confident_hits: list[float] = []
    briers: list[float] = []
    maes: list[float] = []
    for e in resolved:
        p_up = float(e["p_up"])
        realized = float(e["realized_return_pct"])
        predicted_up = p_up >= 0.5
        realized_up = realized > 0.0
        hit = 1.0 if predicted_up == realized_up else 0.0
        hits.append(hit)
        briers.append((p_up - (1.0 if realized_up else 0.0)) ** 2)
        expected_abs = e.get("expected_abs_move_pct")
        if isinstance(expected_abs, (int, float)):
            maes.append(abs(float(expected_abs) - abs(realized)))
        if p_up >= _CONFIDENT_HIGH or p_up <= _CONFIDENT_LOW:
            confident_hits.append(hit)

    since_epoch = min(float(e["anchor_epoch"]) for e in resolved if isinstance(e.get("anchor_epoch"), (int, float)))
    since_iso = datetime.fromtimestamp(since_epoch, tz=timezone.utc).astimezone(_ET).isoformat()
    return {
        "n": n,
        "hit_rate": round(sum(hits) / n, 4),
        "n_confident": len(confident_hits),
        "hit_rate_confident": round(sum(confident_hits) / len(confident_hits), 4) if confident_hits else None,
        "brier": round(sum(briers) / n, 4),
        "mae_abs_move_pct": round(sum(maes) / len(maes), 4) if maes else None,
        "since": since_iso,
    }


def resolve_and_score(
    history_file: str, max_lines: int, *, bars_by_symbol: dict[str, list[Bar]], now_epoch: float,
) -> dict:
    """Resolves every PENDING entry (status-driven: any `resolved: false`
    row, regardless of when it was written) whose target bar is present in
    `bars_by_symbol`, rewrites the (capped) history file, and returns the
    full scores payload:
    `{"schema": 1, "written_at": now_epoch, "symbols": {sym: {label:
    {"all": <metrics>, "recent": <metrics>}}}}` -- ready for
    `agent.kronos.state.write_json_atomic` against `state.scores_path()`."""
    entries = _read_lines(history_file)
    changed = False
    for e in entries:
        if e.get("resolved"):
            continue
        sym = e.get("symbol")
        target_epoch = e.get("target_epoch")
        anchor_close = e.get("anchor_close")
        if not isinstance(target_epoch, (int, float)) or not isinstance(anchor_close, (int, float)) or anchor_close == 0:
            continue
        bars = bars_by_symbol.get(sym)
        if not bars:
            continue
        target_bar = _find_target_bar(bars, float(target_epoch))
        if target_bar is None:
            continue
        e["resolved"] = True
        e["realized_return_pct"] = round((target_bar.c / float(anchor_close) - 1.0) * 100.0, 6)
        e["realized_at"] = now_epoch
        changed = True

    if len(entries) > max_lines:
        entries = entries[-max_lines:]
        changed = True
    if changed:
        _write_lines_atomic(history_file, entries)

    now_et_date = datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(_ET).date()
    holidays = _market_holidays()
    recent_cutoff = _n_sessions_ago(now_et_date, _RECENT_PRIOR_SESSIONS, holidays)

    by_key: dict[tuple, list[dict]] = {}
    for e in entries:
        if not e.get("resolved"):
            continue
        sym = e.get("symbol")
        minutes = e.get("horizon_minutes")
        if sym is None or minutes is None:
            continue
        by_key.setdefault((sym, str(int(minutes))), []).append(e)

    symbols_out: dict[str, dict] = {}
    for (sym, label), rows in by_key.items():
        recent_rows = [
            r for r in rows
            if isinstance(r.get("anchor_epoch"), (int, float))
            and datetime.fromtimestamp(float(r["anchor_epoch"]), tz=timezone.utc).astimezone(_ET).date() >= recent_cutoff
        ]
        symbols_out.setdefault(sym, {})[label] = {
            "all": _compute_metrics(rows),
            "recent": _compute_metrics(recent_rows),
        }

    return {"schema": 1, "written_at": now_epoch, "symbols": symbols_out}
