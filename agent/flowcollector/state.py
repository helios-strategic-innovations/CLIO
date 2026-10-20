"""
CLIO flowcollector — state file + comparison log (I/O).

`get_selfflow()` is the ONE contract other CLIO code will eventually read
(INTEL_V3_SPEC.md: "S ... get_selfflow(symbol, max_age_seconds=300) -> dict
| None ... NOT consumed by DPOF yet"). Per the shared spec's Contracts
section ("Readers: sync, never raise, mtime-cached"), this module never
raises to its caller and caches the parsed state by the file's mtime so a
hot caller costs one os.stat() when nothing changed -- same pattern as
`agent.dpof.readmodel`'s `_read_json_cached`.

Writes (`write_state_atomic`, `append_comparison_row`) are the collector
process's own, called only from collector.py, always off the asyncio event
loop (via `loop.run_in_executor`, never directly on it -- CLAUDE.md
anti-pattern #19) and always tmp+os.replace for the state file (CLAUDE.md:
"Use atomic writes for any JSON persistence").

Comparison log (`append_comparison_row`): side-by-side rows of this
package's own computed pillars against CLIO's UW-derived ones, for the 1-2
week shadow evaluation INTEL_V3_SPEC.md calls for. The only UW pillar
snapshot ever found written to disk anywhere in this repo (grep-verified
across agent/dpof/*.py: decider_state.json holds only a daily call-count
cap; positions.json/risk_state.json hold ledger/risk state, not a
continuous pillar snapshot; DPOFDataFeed's own pillar cache in
agent/dpof/data.py is in-process memory only, never persisted) is
`agent.integrations.uw_flow_intel._write_bridge_signals`'s daily
`moltbot_knowledge/intelligence/uw_flow_{YYYYMMDD}.json` export -- written
by CLIO's own learning cycle for Hydrinia's benefit, but readable back here
too. `_read_uw_bridge_snapshot_sync` is a best-effort, READ-ONLY reader of
that file; when it (or `agent.config`, or the file itself) is unavailable,
`append_comparison_row` degrades to a no-op (never raises, never fabricates
a comparison row) -- exactly the "else skip" the shared spec calls for.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger("clio.flowcollector.state")

# agent/flowcollector/state.py -> 3x dirname = repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STATE_FILE = os.path.join(_REPO_ROOT, "data", "intelligence", "selfflow_state.json")
DEFAULT_COMPARE_FILE = os.path.join(_REPO_ROOT, "data", "intelligence", "selfflow_compare.jsonl")

_COMPARE_CAP_LINES = 3000
_COMPARE_TRIM_TO_LINES = 2000
_COMPARE_MAX_BYTES = 4 * 1024 * 1024  # size-gate the trim check, same shape as agent.dpof.ledger

_state_cache_lock = threading.Lock()
_state_cache: dict = {"path": None, "key": None, "data": None}


def state_path() -> str:
    return os.environ.get("CLIO_SELFFLOW_FILE", DEFAULT_STATE_FILE) or DEFAULT_STATE_FILE


def compare_path() -> str:
    return os.environ.get("CLIO_SELFFLOW_COMPARE_FILE", DEFAULT_COMPARE_FILE) or DEFAULT_COMPARE_FILE


# ─────────────────────────────────────────────────────────────────────────
# State file: write (collector.py only) / read (get_selfflow, public API)
# ─────────────────────────────────────────────────────────────────────────


def write_state_atomic(state: dict, path: Optional[str] = None) -> None:
    """Blocking; caller (collector.py) must run this via run_in_executor,
    never directly on the event loop. tmp + os.replace, per CLAUDE.md.

    `path` lets collector.py pass its own `FlowCollectorConfig.state_file`
    explicitly (config is passed down, not re-derived from the ambient
    process environment a second time) -- defaults to the env/default path
    `state_path()` resolves, which is what `get_selfflow` (this module's
    public, config-free contract for OTHER CLIO code) always uses."""
    path = path or state_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data_str = json.dumps(state, default=str)  # serialize BEFORE touching any file handle
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(data_str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception as e:
        logger.error(f"flowcollector: state write failed: {type(e).__name__}: {e}")


def _read_state_sync() -> Optional[dict]:
    path = state_path()
    try:
        st = os.stat(path)
    except OSError:
        return None
    # Inode + ns mtime + size (same key as agent.kronos.state and the other file caches): the writer replaces the
    # file atomically (a new inode per write), and a float mtime alone repeats across two writes in one clock tick,
    # which served the first state for the second one.
    key = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _state_cache_lock:
        if _state_cache["path"] == path and _state_cache["key"] == key and _state_cache["data"] is not None:
            return _state_cache["data"]
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.debug(f"flowcollector: state read failed: {type(e).__name__}: {e}")
        return None
    if not isinstance(data, dict):
        return None
    with _state_cache_lock:
        _state_cache["path"] = path
        _state_cache["key"] = key
        _state_cache["data"] = data
    return data


def get_selfflow(symbol: str, max_age_seconds: int = 300) -> Optional[dict]:
    """Sync, never raises. Returns the pillar-shaped dict for `symbol`
    ({nofa, premium, darkpool, gex, nope, max_pain, gex_regime, as_of,
    quality_notes, ...}) if the state file exists, parses, contains that
    symbol, and its `as_of` is within `max_age_seconds` of now -- else None.
    Missing/stale/malformed is never distinguished from "not tracked" to the
    caller (there is nothing actionable to do differently -- this mirrors
    every other CLIO advisory reader's contract, e.g.
    agent.integrations.newsflash_lite's get_breaking_state degrading to
    empty rather than raising).
    """
    try:
        state = _read_state_sync()
        if not state:
            return None
        # The collector rewrites this file every tick even while idle (outside
        # market hours) or with its streams down; its windows are then empty,
        # not quiet. Never serve that as a fresh reading.
        status = state.get("collector_status")
        if isinstance(status, dict) and status.get("streams_active") is False:
            return None
        symbols = state.get("symbols")
        if not isinstance(symbols, dict):
            return None
        entry = symbols.get(symbol.upper())
        if not isinstance(entry, dict):
            return None
        as_of = entry.get("as_of")
        if as_of is None:
            return None
        age = time.time() - float(as_of)
        if age > max_age_seconds:
            return None
        return entry
    except Exception as e:
        logger.debug(f"flowcollector: get_selfflow({symbol!r}) failed: {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
# Comparison log (best-effort; see module docstring)
# ─────────────────────────────────────────────────────────────────────────


def _read_uw_bridge_snapshot_sync(tradingbot_dir: Optional[str] = None) -> Optional[dict]:
    """Blocking. Best-effort read of CLIO's own UW-flow bridge export (see
    module docstring). Never raises; returns None on any failure, including
    `agent.config` being unavailable, the file not existing yet today, or an
    unexpected shape."""
    try:
        if tradingbot_dir is None:
            try:
                from agent.config import config as _clio_config  # lazy: keep this module import-light

                tradingbot_dir = _clio_config.tradingbot_dir
            except Exception as e:
                logger.debug(f"flowcollector: agent.config unavailable for uw bridge read: {type(e).__name__}: {e}")
                return None
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ny = ZoneInfo("America/New_York")
        today = datetime.now(ny).strftime("%Y%m%d")
        path = os.path.join(tradingbot_dir, "moltbot_knowledge", "intelligence", f"uw_flow_{today}.json")
        if not os.path.isfile(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.debug(f"flowcollector: uw bridge snapshot read failed: {type(e).__name__}: {e}")
        return None


def build_comparison_row(underlying: str, self_entry: dict, uw_snapshot: Optional[dict], now_epoch: float) -> Optional[dict]:
    """PURE. Assembles one side-by-side JSONL row from this collector's own
    pillar entry and (if available) the matching slice of `uw_snapshot`
    (agent.integrations.uw_flow_intel._write_bridge_signals's shape:
    {"gex": {ticker: {...}}, "top_signals": [{"ticker":..., "score":...}],
    "dark_pool_top": {...}, "market_tide": {...}}). Returns None only if
    `self_entry` itself is falsy (nothing to compare) -- a missing/partial
    `uw_snapshot` still produces a row with `uw` fields set to None, which
    is itself useful shadow-evaluation signal ("UW side unavailable this
    tick"), never silently dropped.
    """
    if not self_entry:
        return None
    uw_gex = None
    uw_score = None
    if isinstance(uw_snapshot, dict):
        gex_map = uw_snapshot.get("gex")
        if isinstance(gex_map, dict):
            g = gex_map.get(underlying)
            if isinstance(g, dict):
                uw_gex = {"regime": g.get("regime"), "net_gex": g.get("net_gex"), "pin_strike": g.get("pin_strike")}
        for sig in uw_snapshot.get("top_signals") or []:
            if isinstance(sig, dict) and sig.get("ticker") == underlying:
                uw_score = sig.get("score")
                break
    return {
        "underlying": underlying,
        "as_of": now_epoch,
        "self": {
            "nofa": self_entry.get("nofa"),
            "premium": self_entry.get("premium"),
            "darkpool": self_entry.get("darkpool"),
            "gex": self_entry.get("gex"),
            "gex_regime": self_entry.get("gex_regime"),
            "nope": self_entry.get("nope"),
            "max_pain": self_entry.get("max_pain"),
        },
        "uw": {"gex": uw_gex, "score": uw_score, "available": uw_snapshot is not None},
    }


def append_comparison_row(row: dict, path: Optional[str] = None) -> None:
    """Blocking; caller runs via run_in_executor. Append-only JSONL + a
    size-gated trim (mirrors agent.dpof.ledger's decisions.jsonl pattern:
    only reads the whole file back when it has grown past
    _COMPARE_MAX_BYTES, so a normal append never pays that cost). `path`:
    see write_state_atomic's docstring -- same explicit-over-ambient
    rationale."""
    path = path or compare_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(row, default=str)
        with open(path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        _trim_compare_log_if_needed(path)
    except Exception as e:
        logger.error(f"flowcollector: comparison log append failed: {type(e).__name__}: {e}")


def _trim_compare_log_if_needed(path: str) -> None:
    # No fcntl.flock here (unlike agent.dpof.ledger's equivalent trim):
    # append + read-modify-write + os.replace is safe WITHOUT a lock only
    # because collector.py's single-writer discipline guarantees exactly
    # one process/thread ever calls append_comparison_row -- the collector
    # process's own tick() loop, never concurrently with itself (see
    # collector.py's module docstring). If a second writer to this same
    # file is ever introduced, this needs a lock first.
    try:
        if os.path.getsize(path) <= _COMPARE_MAX_BYTES:
            return
        with open(path, "r") as f:
            lines = f.readlines()
    except OSError as e:
        logger.error(f"flowcollector: comparison log trim read failed: {e}")
        return
    if len(lines) <= _COMPARE_CAP_LINES:
        return
    kept = lines[-_COMPARE_TRIM_TO_LINES:]
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.writelines(kept)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    logger.info(f"flowcollector: trimmed comparison log from {len(lines)} to {len(kept)} lines")
