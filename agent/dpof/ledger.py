"""
CLIO DPOF — ledger persistence (I/O).

Files under ``data_dir``:
  positions.json   {"version": 1, "positions": [PositionRecord.to_dict(), ...]}
  risk_state.json  RiskState.to_dict()
  decisions.jsonl  append-only; once > _JSONL_MAX_BYTES (and > _JSONL_CAP lines) trimmed to _JSONL_TRIM_TO
  positions_archive.jsonl  closed/cancelled records older than 14 days (moved on load)
  jev.jsonl        same shape/cap as decisions.jsonl
  .dpof.lock       sidecar lock file (never flock a file that gets os.replace()'d —
                   CLAUDE.md anti-pattern #22)
  HALT             presence blocks new entries (checked by risk.py's caller)

Every mutating operation runs inside ONE `run_in_executor` call that holds an
exclusive `fcntl.flock` on the sidecar lock file for its whole duration
(serialize -> write .tmp -> fsync -> os.replace, or the JSONL append + trim).
Corrupt JSON is never allowed to crash the engine: it is logged, quarantined
to `<file>.corrupt-<epoch>`, and the ledger continues with an empty/default
value for that file.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import time
from typing import Any, Callable, Optional

from agent.dpof.models import PositionRecord, RiskState

logger = logging.getLogger("clio.dpof.ledger")

_POSITIONS_FILE = "positions.json"
_RISK_FILE = "risk_state.json"
_DECISIONS_FILE = "decisions.jsonl"
_JEV_FILE = "jev.jsonl"
_LOCK_FILE = ".dpof.lock"
_HALT_FILE = "HALT"

_JSONL_CAP = 5000
_JSONL_TRIM_TO = 4000
# Trim check is size-based so an append never has to read the whole file.
# ~1 KB/line typical -> trim once the file passes ~5 MB.
_JSONL_MAX_BYTES = 5 * 1024 * 1024
_ARCHIVE_FILE = "positions_archive.jsonl"
_ARCHIVE_AFTER_SECONDS = 14 * 86400   # closed/cancelled records older than this leave positions.json

_OPEN_STATUSES = ("pending", "open", "closing")


class DPOFLedger:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._positions: dict[str, PositionRecord] = {}
        self._risk = RiskState()
        self._loaded = False
        # Serializes persists IN ORDER. Snapshots are taken under this lock, so an
        # older snapshot can never land on disk after a newer one (overlapping
        # shielded writes from a cancelled cycle + the next cycle).
        self._persist_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Locking / executor plumbing
    # ------------------------------------------------------------------

    def _lock_path(self) -> str:
        return os.path.join(self.data_dir, _LOCK_FILE)

    def _with_lock(self, fn: Callable[[], Any]) -> Any:
        """Run fn() synchronously while holding an exclusive sidecar flock.

        Intended to run inside a single run_in_executor call. Opens the lock
        file in "a" mode (never truncates) each time -- a fresh fd per call so
        we never hold two flocks on the same file from this process at once
        (CLAUDE.md anti-pattern #20).
        """
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self._lock_path(), "a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                return fn()
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    async def _run_locked(self, fn: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._with_lock(fn))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    async def load(self) -> None:
        def _do_load():
            os.makedirs(self.data_dir, exist_ok=True)
            positions_raw = self._load_json_quarantine(
                os.path.join(self.data_dir, _POSITIONS_FILE), default={"version": 1, "positions": []}
            )
            risk_raw = self._load_json_quarantine(os.path.join(self.data_dir, _RISK_FILE), default={})
            return positions_raw, risk_raw

        positions_raw, risk_raw = await self._run_locked(_do_load)

        pos_list = positions_raw.get("positions", []) if isinstance(positions_raw, dict) else []
        positions: dict[str, PositionRecord] = {}
        for d in pos_list:
            try:
                rec = PositionRecord.from_dict(d)
                if rec.id:
                    positions[rec.id] = rec
            except Exception as e:
                logger.error(f"DPOF ledger: dropping unparsable position record: {e}")
        self._positions = positions
        await self._archive_old_closed()

        try:
            self._risk = RiskState.from_dict(risk_raw) if risk_raw else RiskState()
        except Exception as e:
            logger.error(f"DPOF ledger: dropping unparsable risk state: {e}")
            self._risk = RiskState()

        self._loaded = True

    @staticmethod
    def _load_json_quarantine(path: str, default: Any) -> Any:
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r") as f:
                raw = f.read()
            if not raw.strip():
                return default
            return json.loads(raw)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.error(f"DPOF ledger: corrupt JSON at {path}: {e}; quarantining and starting empty")
            try:
                os.replace(path, f"{path}.corrupt-{int(time.time())}")
            except OSError as e2:
                logger.error(f"DPOF ledger: failed to quarantine {path}: {e2}")
            return default

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    @property
    def positions(self) -> list[PositionRecord]:
        return list(self._positions.values())

    def open_positions(self) -> list[PositionRecord]:
        return [p for p in self._positions.values() if p.status in _OPEN_STATUSES]

    def get(self, position_id: str) -> Optional[PositionRecord]:
        return self._positions.get(position_id)

    async def upsert(self, pos: PositionRecord) -> None:
        self._positions[pos.id] = pos
        await self._persist_positions()

    async def _persist_positions(self) -> None:
        async with self._persist_lock:
            snapshot = {"version": 1, "positions": [p.to_dict() for p in self._positions.values()]}

            def _do_write():
                self._atomic_write_json(os.path.join(self.data_dir, _POSITIONS_FILE), snapshot)

            await self._run_locked(_do_write)

    async def _archive_old_closed(self, now: Optional[float] = None) -> int:
        """Move closed/cancelled records older than _ARCHIVE_AFTER_SECONDS out of
        positions.json into positions_archive.jsonl (append-only history), so the
        hot file stays small. Never touches pending/open/closing records."""
        now = time.time() if now is None else now
        old = [
            p for p in self._positions.values()
            if p.status not in _OPEN_STATUSES
            and p.closed_time is not None
            and now - p.closed_time > _ARCHIVE_AFTER_SECONDS
        ]
        if not old:
            return 0
        archive_path = os.path.join(self.data_dir, _ARCHIVE_FILE)
        lines = [json.dumps(p.to_dict(), default=str) + "\n" for p in old]

        def _do_archive():
            with open(archive_path, "a") as f:
                f.writelines(lines)
                f.flush()
                os.fsync(f.fileno())

        await self._run_locked(_do_archive)
        for p in old:
            self._positions.pop(p.id, None)
        await self._persist_positions()
        logger.info(f"DPOF ledger: archived {len(old)} closed position record(s)")
        return len(old)

    # ------------------------------------------------------------------
    # Risk state
    # ------------------------------------------------------------------

    @property
    def risk(self) -> RiskState:
        return self._risk

    async def save_risk(self, state: RiskState) -> None:
        async with self._persist_lock:
            self._risk = state
            data = state.to_dict()

            def _do_write():
                self._atomic_write_json(os.path.join(self.data_dir, _RISK_FILE), data)

            await self._run_locked(_do_write)

    @staticmethod
    def _atomic_write_json(path: str, data: dict) -> None:
        text = json.dumps(data, default=str, indent=2)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    # ------------------------------------------------------------------
    # JSONL append logs
    # ------------------------------------------------------------------

    async def append_decision(self, record: dict) -> None:
        await self._append_jsonl(_DECISIONS_FILE, record)

    async def append_jev(self, record: dict) -> None:
        await self._append_jsonl(_JEV_FILE, record)

    async def _append_jsonl(self, filename: str, record: dict) -> None:
        path = os.path.join(self.data_dir, filename)

        def _do_append():
            os.makedirs(self.data_dir, exist_ok=True)
            line = json.dumps(record, default=str)
            with open(path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._trim_jsonl_if_needed(path)

        await self._run_locked(_do_append)

    @staticmethod
    def _trim_jsonl_if_needed(path: str) -> None:
        try:
            if os.path.getsize(path) <= _JSONL_MAX_BYTES:
                return
            with open(path, "r") as f:
                lines = f.readlines()
        except OSError as e:
            logger.error(f"DPOF ledger: failed to read {path} for trim check: {e}")
            return
        if len(lines) <= _JSONL_CAP:
            return
        kept = lines[-_JSONL_TRIM_TO:]
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            f.writelines(kept)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        logger.info(f"DPOF ledger: trimmed {path} from {len(lines)} to {len(kept)} lines")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def halt_file_present(self) -> bool:
        return os.path.exists(os.path.join(self.data_dir, _HALT_FILE))
