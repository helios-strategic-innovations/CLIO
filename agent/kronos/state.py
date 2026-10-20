"""CLIO Kronos — the file contract between the forecaster service and its readers.

The service (`agent.kronos` package, its own process) is the ONLY writer of the two files below;
DPOF and the Telegram/read-model code are readers. Readers are sync, never raise, and cache the
parsed file by mtime (same pattern as `agent.flowcollector.state`); callers on the asyncio loop
run them in an executor. Schema changes bump SCHEMA and every reader rejects other versions.

kronos_forecast.json (rewritten after every forecast run)::

    {"schema": 1, "written_at": <epoch s>,
     "model": {"model", "model_revision", "tokenizer", "tokenizer_revision", "vendor_commit"},
     "params": {"bar_minutes", "lookback_bars", "sample_paths", "temperature", "top_p"},
     "symbols": {"SPY": {
         "status": "ok" | "no_data" | "error", "error": str | null,
         "asof_bar": "<ISO time the last input bar CLOSED, America/New_York>",
         "anchor_close": float, "feed": "iex" | "sip" | ..., "latency_ms": int,
         "horizons": {"15": {"minutes": 15, "p_up", "mean_return_pct", "median_return_pct",
                             "q10_return_pct", "q90_return_pct", "expected_abs_move_pct"},
                      "30": {...}, "60": {...}}}}}

    Return fields are percent (0.12 = +0.12 %) of `anchor_close`; p_up is the share of sampled
    paths whose close at that horizon is above `anchor_close` (ties count half). A horizon that
    would run past the regular-session close is omitted, never extrapolated overnight.

kronos_scores.json (rewritten after every scoring pass)::

    {"schema": 1, "written_at": <epoch s>,
     "symbols": {"SPY": {"15": {"all": <metrics>, "recent": <metrics>}, "30": {...}, "60": {...}}}}

    <metrics> = {"n", "hit_rate", "n_confident", "hit_rate_confident", "brier", "mae_abs_move_pct",
                 "since"}; "recent" covers the last 5 sessions. A metric with no scored forecasts is
    null (never 0.0 — CLAUDE.md "No fake zeros"). "confident" = p_up >= 0.6 or <= 0.4.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Optional

logger = logging.getLogger("clio.kronos.state")

SCHEMA = 1
HORIZONS = ("15", "30", "60")

# agent/kronos/state.py -> 3x dirname = repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FORECAST_FILE = os.path.join(_REPO_ROOT, "data", "intelligence", "kronos_forecast.json")
DEFAULT_SCORES_FILE = os.path.join(_REPO_ROOT, "data", "intelligence", "kronos_scores.json")

# A forecast older than this is stale: the service refreshes every 5 minutes.
DEFAULT_MAX_AGE_SECONDS = 600

_cache_lock = threading.Lock()
_cache: dict = {}


def forecast_path() -> str:
    return os.environ.get("CLIO_KRONOS_FORECAST_FILE", DEFAULT_FORECAST_FILE) or DEFAULT_FORECAST_FILE


def scores_path() -> str:
    return os.environ.get("CLIO_KRONOS_SCORES_FILE", DEFAULT_SCORES_FILE) or DEFAULT_SCORES_FILE


def _read_json_cached(path: str) -> Optional[dict]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    # Inode + ns mtime + size: the writer replaces the file atomically (new inode per write), and a
    # float mtime alone can repeat across two quick writes.
    key = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _cache_lock:
        hit = _cache.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        logger.debug(f"kronos state read failed for {os.path.basename(path)}: {type(e).__name__}")
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    with _cache_lock:
        _cache[path] = (key, data)
    return data


def read_forecast(now: Optional[float] = None, max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS) -> Optional[dict]:
    """The latest forecast file if it is fresh (written within `max_age_seconds`), else None.
    Never raises. The returned dict is shared with the cache: treat it as read-only."""
    data = _read_json_cached(forecast_path())
    if data is None:
        return None
    written = data.get("written_at")
    if isinstance(written, bool) or not isinstance(written, (int, float)):
        return None
    now = time.time() if now is None else now
    if now - float(written) > max_age_seconds or float(written) - now > 60:
        return None
    if not isinstance(data.get("symbols"), dict):
        return None
    return data


def read_scores() -> Optional[dict]:
    """The latest track-record file (any age: it summarises history), else None. Never raises."""
    data = _read_json_cached(scores_path())
    if data is None or not isinstance(data.get("symbols"), dict):
        return None
    return data


def symbol_forecast(forecast: Optional[dict], symbol: str) -> Optional[dict]:
    """One symbol's entry when its status is "ok", else None."""
    if not isinstance(forecast, dict):
        return None
    entry = (forecast.get("symbols") or {}).get(symbol)
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        return None
    return entry


def write_json_atomic(path: str, obj: Any) -> None:
    """Writer side (the service only; call off the event loop). tmp file in the same directory +
    os.replace, so a reader sees the old or the new file, never a torn one."""
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".kronos-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = -1
            fh.write(data)
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
