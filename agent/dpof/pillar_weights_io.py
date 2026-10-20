"""
CLIO DPOF -- dynamic pillar-weight state store and shadow log (I/O only).

Spec: docs/design/clio_dpof_dynamic_weights.md sections 4 and 5.

* StateStore: one atomic JSON document (tmp + fsync + os.replace) guarded by a
  SIDECAR flock that is never replaced, so the data file can be swapped safely.
* ShadowLog: trimmed JSONL, one append per cycle under the same sidecar pattern.

Sync cores never raise; *_async wrappers run the sync core in the default
executor so no flock or file I/O ever happens on the asyncio loop thread.
Stdlib only -- no imports from agent.*.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import math
import os
import threading
import time
from typing import Any

logger = logging.getLogger("clio.dpof.pillar_weights_io")

SCHEMA_VERSION = 1
_STATE_LOCK_NAME = ".dpof_pillar_weights.lock"
_SHADOW_LOCK_NAME = ".dpof_pillar_weights_shadow.lock"
_WARN_INTERVAL_S = 60.0
_TRIM_KEEP_FRACTION = 0.9
_FAIL_BACKOFF_S = 30.0
_TRIM_BACKOFF_S = 300.0


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite constant {name} in JSON")


def clean_for_json(obj: Any) -> Any:
    """Return a JSON-safe copy: non-finite floats -> None, tuples -> lists,
    non-str dict keys -> str(key). Other values are returned unchanged."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else str(k)): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    return obj


def _warn_deduped(seen: dict, key: str, msg: str) -> None:
    now = time.monotonic()
    last = seen.get(key)
    if last is not None and now - last < _WARN_INTERVAL_S:
        return
    seen[key] = now
    logger.warning(msg)


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError as e:
        logger.info("DPOF pillar weights: could not remove temp file %s: %s", path, e)


class StateStore:
    """Atomic JSON state document with a sidecar lock."""

    def __init__(self, data_dir: str, filename: str = "pillar_weights.json"):
        self.data_dir = data_dir
        self._filename = filename
        self.path = os.path.join(data_dir, filename)
        self._lock_path = os.path.join(data_dir, _STATE_LOCK_NAME)
        self._warned: dict[str, float] = {}

    # ------------------------------------------------------------------ load
    def load(self) -> dict | None:
        """Return the state dict, or None (missing, corrupt, other schema). Never raises."""
        try:
            return self._load()
        except Exception as e:
            _warn_deduped(self._warned, "load", f"DPOF pillar weights: load failed: {e}")
            return None

    def _load(self) -> dict | None:
        try:
            lf = open(self._lock_path, "a")
        except FileNotFoundError:
            return None
        try:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                if not os.path.exists(self.path):
                    return None
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        parsed = json.loads(f.read(), parse_constant=_reject_constant)
                    if not isinstance(parsed, dict):
                        raise ValueError("top-level JSON is not an object")
                except ValueError as e:  # unparseable / non-dict / NaN literal: genuinely corrupt
                    self._quarantine(str(e))
                    return None
                except OSError as e:  # a transient read error (EACCES, EIO, EMFILE) must NOT rename a good file away
                    _warn_deduped(self._warned, "read", f"DPOF pillar weights: could not read state ({type(e).__name__}); file left in place, starting empty")
                    return None
                ver = parsed.get("schema_version")
                if isinstance(ver, bool) or ver != SCHEMA_VERSION:
                    logger.info("DPOF pillar weights: schema_version %r != %d; ignoring state file",
                                ver, SCHEMA_VERSION)
                    return None
                return parsed
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        finally:
            lf.close()

    def _quarantine(self, reason: str) -> None:
        dest = f"{self.path}.corrupt-{int(time.time())}"
        try:
            os.replace(self.path, dest)
            logger.warning("DPOF pillar weights: corrupt state (%s); renamed to %s, starting empty",
                           reason, dest)
        except OSError as e:
            logger.warning("DPOF pillar weights: corrupt state (%s); could not rename: %s", reason, e)

    # ------------------------------------------------------------------ save
    def save(self, state: dict) -> bool:
        """Atomically persist state (schema_version/written_at stamped). Returns True on success. Never raises."""
        try:
            return self._save(state)
        except Exception as e:
            _warn_deduped(self._warned, "save", f"DPOF pillar weights: save failed: {e}")
            return False

    def _save(self, state: dict) -> bool:
        if not isinstance(state, dict):
            _warn_deduped(self._warned, "not_dict", "DPOF pillar weights: save refused, state is not a dict")
            return False
        doc = dict(state)
        doc["schema_version"] = SCHEMA_VERSION
        doc["written_at"] = time.time()
        try:
            text = json.dumps(doc, allow_nan=False, sort_keys=True, separators=(",", ":"))
        except (ValueError, TypeError, RuntimeError) as e:
            _warn_deduped(self._warned, "serialize",
                          f"DPOF pillar weights: refusing to persist ({type(e).__name__}); existing file untouched")
            return False
        os.makedirs(self.data_dir, exist_ok=True)
        tmp = os.path.join(self.data_dir,
                           f".{self._filename}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(self._lock_path, "a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)
            except OSError as e:
                _remove_quietly(tmp)
                _warn_deduped(self._warned, "write", f"DPOF pillar weights: write failed: {e}")
                return False
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        return True

    # ----------------------------------------------------------------- async
    async def load_async(self) -> dict | None:
        """load() in the default executor. Never raises."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.load)
        except Exception as e:
            logger.warning("DPOF pillar weights: load_async failed: %s", e)
            return None

    async def save_async(self, state: dict) -> bool:
        """save() in the default executor. Never raises."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.save, state)
        except Exception as e:
            logger.warning("DPOF pillar weights: save_async failed: %s", e)
            return False


class ShadowLog:
    """Trimmed JSONL shadow log, appended once per cycle under a sidecar lock."""

    def __init__(self, data_dir: str, filename: str = "pillar_weights_shadow.jsonl",
                 max_lines: int = 40000):
        self.data_dir = data_dir
        self._filename = filename
        self.path = os.path.join(data_dir, filename)
        self._lock_path = os.path.join(data_dir, _SHADOW_LOCK_NAME)
        self.max_lines = max(1, int(max_lines))
        self._lines: int | None = None
        self._warned: dict[str, float] = {}
        self._retry_after = 0.0  # monotonic: after a failed write, skip appends briefly instead of hammering a broken disk
        self._trim_after = 0.0   # monotonic: after a failed trim, do not retry the (expensive) rewrite for a while

    # ----------------------------------------------------------------- write
    def append_cycle(self, records: list[dict]) -> int:
        """Append one JSON line per record; returns lines written. Never raises."""
        try:
            return self._append(records)
        except Exception as e:
            self._retry_after = time.monotonic() + _FAIL_BACKOFF_S  # e.g. a read-only directory: back off, do not hammer
            _warn_deduped(self._warned, "append", f"DPOF shadow log: append failed: {e}")
            return 0

    def _append(self, records: list[dict]) -> int:
        if not records or time.monotonic() < self._retry_after:
            return 0
        lines: list[str] = []
        for rec in records:
            try:
                if not isinstance(rec, dict):
                    raise TypeError("record is not a dict")
                lines.append(json.dumps(clean_for_json(rec), allow_nan=False, separators=(",", ":")))
            except (ValueError, TypeError, RuntimeError) as e:
                _warn_deduped(self._warned, "bad_record",
                              f"DPOF shadow log: skipping unserializable record ({type(e).__name__})")
        if not lines:
            return 0
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self._lock_path, "a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                return self._write_locked(lines)
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _write_locked(self, lines: list[str]) -> int:
        if self._lines is None:
            self._lines = self._count_newlines()
        repaired = self._ends_without_newline()
        payload = ("\n" if repaired else "") + "\n".join(lines) + "\n"
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            self._lines = None
            self._retry_after = time.monotonic() + _FAIL_BACKOFF_S
            _warn_deduped(self._warned, "write", f"DPOF shadow log: write failed: {e}")
            return 0
        self._lines += len(lines) + (1 if repaired else 0)
        if self._lines > self.max_lines and time.monotonic() >= self._trim_after:
            self._trim_locked()
        return len(lines)

    def _count_newlines(self) -> int:
        n = 0
        try:
            with open(self.path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    n += chunk.count(b"\n")
        except FileNotFoundError:
            return 0
        return n

    def _ends_without_newline(self) -> bool:
        try:
            with open(self.path, "rb") as f:
                f.seek(0, 2)
                if f.tell() == 0:
                    return False
                f.seek(-1, 2)
                return f.read(1) != b"\n"
        except FileNotFoundError:
            return False

    def _trim_locked(self) -> None:
        keep = max(1, int(self.max_lines * _TRIM_KEEP_FRACTION))
        tmp = os.path.join(self.data_dir,
                           f".{self._filename}.{os.getpid()}.{threading.get_ident()}.trim.tmp")
        try:
            with open(self.path, "rb") as f:
                tail = f.readlines()[-keep:]
            with open(tmp, "wb") as f:
                f.write(b"".join(tail))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self._lines = len(tail)
        except OSError as e:
            _remove_quietly(tmp)
            self._lines = None
            self._trim_after = time.monotonic() + _TRIM_BACKOFF_S
            _warn_deduped(self._warned, "trim", f"DPOF shadow log: trim failed: {e}")

    # ------------------------------------------------------------------ read
    def read_tail(self, n: int) -> list[dict]:
        """Last n valid JSON object lines (torn/invalid lines skipped). For tests and reports."""
        if n <= 0:
            return []
        out: list[dict] = []
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        rec = json.loads(line, parse_constant=_reject_constant)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        out.append(rec)
        except OSError:
            return []
        return out[-n:]

    # ----------------------------------------------------------------- async
    async def append_cycle_async(self, records: list[dict]) -> int:
        """append_cycle() in the default executor. Never raises."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.append_cycle, records)
        except Exception as e:
            logger.warning("DPOF shadow log: append_cycle_async failed: %s", e)
            return 0
