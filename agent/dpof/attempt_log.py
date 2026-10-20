"""Bounded JSONL ledger of every DPOF decider provider attempt (stdlib only).

One row per provider attempt (Anthropic primary, OpenAI fallback), written to
`<data_dir>/decider_attempts.jsonl`. Writes are fire-and-forget on the
executor: `record()` never blocks the asyncio loop and never raises; a failed
write is logged at DEBUG and dropped. A single process-wide threading.Lock
serialises appends and the periodic trim. The trim rewrites the newest 90% of
lines through tmp + os.replace under that same lock (threading lock only, no
flock, so the os.replace swap is safe here).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger("clio.dpof.attempt_log")

_TRIM_EVERY = 200          # check the line count every N successful appends
_KEEP_FRACTION = 0.9       # a trim keeps the newest 90% of max_lines
_LOCK = threading.Lock()   # shared by every AttemptLog: one writer per process


class AttemptLog:
    def __init__(self, path: str, *, max_lines: int = 20000, enabled: bool = True) -> None:
        self.path = path
        self.max_lines = max(100, int(max_lines))
        self.enabled = bool(enabled)
        self._writes = 0
        self._pending: set = set()

    def record(self, row: dict[str, Any]) -> None:
        """Schedule one append on the executor. Must be called from the loop."""
        if not self.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(None, self._append_sync, row)
        except Exception as e:
            logger.debug(f"DPOF attempt log: schedule failed: {type(e).__name__}: {e}")
            return
        self._pending.add(fut)
        fut.add_done_callback(self._on_done)

    async def drain(self) -> None:
        """Await every scheduled append (used by tests and orderly shutdown)."""
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    def _on_done(self, fut: "asyncio.Future") -> None:
        self._pending.discard(fut)
        try:
            exc = fut.exception() if not fut.cancelled() else None
        except Exception:
            exc = None
        if exc is not None:
            logger.debug(f"DPOF attempt log: write failed: {type(exc).__name__}: {exc}")

    def _append_sync(self, row: dict[str, Any]) -> None:
        """Runs on an executor thread. Fail-open: errors are logged at DEBUG."""
        try:
            line = json.dumps(row, default=str, separators=(",", ":"), ensure_ascii=True) + "\n"
            with _LOCK:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
                self._writes += 1
                if self._writes % _TRIM_EVERY == 0:
                    self._trim_locked()
        except Exception as e:
            logger.debug(f"DPOF attempt log: append failed: {type(e).__name__}: {e}")

    def _trim_locked(self) -> None:
        """Caller must hold _LOCK."""
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= self.max_lines:
            return
        keep = lines[-max(1, int(self.max_lines * _KEEP_FRACTION)):]
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(keep)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)
