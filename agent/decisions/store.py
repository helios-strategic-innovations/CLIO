"""CLIO decisions producer — async write-behind store to PostgREST
.

Every write goes through `enqueue(table, rows)`: SYNC, non-blocking, never
raises. A single background writer task drains a bounded in-memory queue
(FIFO — this is what keeps `questions` writes ordered ahead of their
`answers`/`outcomes`, since a caller always enqueues a call's `questions`
rows before its `answers` rows). A batch that fails to POST is appended to a
JSONL outbox file instead (never lost, never blocks the caller); a later
successful write opportunistically tries to drain the outbox too, at most
once every 60s. Every row carries a client-generated id (or, for `answers`,
a `(question_id, model, role)` UNIQUE — see migration 020) precisely so a
replayed outbox batch is safe to resend: `Prefer: resolution=ignore-
duplicates` makes a duplicate POST a no-op, not a duplicate row.

No `SUPABASE_URL`/`SUPABASE_SERVICE_ROLE_KEY` -> the store is disabled: every
`enqueue()` call is a no-op (one INFO log, nothing written to the outbox).
`SUPABASE_URL`/`SUPABASE_SERVICE_ROLE_KEY` are re-read from the environment on
every call (never cached at construction) -- same pattern as
`agent.integrations.social_intel_enrich`'s own `_supa_base`/`_supa_key`
(copied here rather than imported, per the build spec: those are private
helpers of that module).
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import tempfile
import time
from typing import Any, Optional

import httpx

from agent.decisions.config import DecisionsConfig, from_env

logger = logging.getLogger("clio.decisions.store")

_QUEUE_MAXSIZE = 2000
_WRITE_TIMEOUT_SECONDS = 10.0
_DRAIN_MIN_INTERVAL_SECONDS = 60.0
_OUTCOME_IDS_FILENAME = "recorded_outcome_ids.json"


# _post_batch outcomes. 401/403 (key or grant not in place yet), 404 (schema not exposed to PostgREST
# yet), 408 and 429 are retryable; 409 (FK miss) is retried a bounded number of times (_CONFLICT).
_OK, _RETRY, _CONFLICT, _DROP = "ok", "retry", "conflict", "drop"
_RETRY_STATUSES = frozenset({401, 403, 404, 408, 429})
# 409 = FK miss: retried from the outbox, but only this many drains before the line is dropped.
_MAX_CONFLICTS = 20


def _error_summary(resp: "httpx.Response") -> str:
    """PostgREST's error code + message only. `details`/`hint` can quote the failing row (bucketed
    trading state), so they are never logged."""
    try:
        body = resp.json()
    except Exception:
        return "(non-JSON error body)"
    if not isinstance(body, dict):
        return "(unexpected error body)"
    return f"code={body.get('code')!s} message={str(body.get('message', ''))[:160]}"


def _supa_base() -> str:
    """Copied from agent.integrations.social_intel_enrich._supa_base (that
    module's own helper is private) -- SUPABASE_URL, or SUPABASE_PROJECT_REF
    as a fallback to derive the default *.supabase.co host."""
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not url:
        ref = os.environ.get("SUPABASE_PROJECT_REF", "")
        if ref:
            url = f"https://{ref}.supabase.co"
    return url


def _supa_key() -> str:
    return os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


# ─────────────────────────────────────────────────────────────────────────
# Outbox — sync helpers, always called via run_in_executor (never on the
# event loop thread). Sidecar lock file (`<outbox>.lock`), never flock the
# data file itself (CLAUDE.md anti-pattern #22: flock + os.replace on the
# SAME path races an inode swap). Atomic write: serialize first, mkstemp in
# the same dir, os.replace.
# ─────────────────────────────────────────────────────────────────────────

def _atomic_write_text(path: str, data: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = -1
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        if fd >= 0:
            os.close(fd)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _append_outbox_sync(path: str, max_lines: int, entry_line: str) -> None:
    lock_path = path + ".lock"
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with open(lock_path, "a") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            lines: list[str] = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            line = entry_line if entry_line.endswith("\n") else entry_line + "\n"
            lines.append(line)
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
            _atomic_write_text(path, "".join(lines))
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _read_outbox_sync(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    lock_path = path + ".lock"
    with open(lock_path, "a") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_SH)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.readlines()
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _rewrite_outbox_sync(path: str, lines: list[str]) -> None:
    lock_path = path + ".lock"
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with open(lock_path, "a") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            if not lines:
                # Nothing left to retry -- remove the file rather than leave
                # an empty one lying around (an operator eyeballing
                # data/decisions/ shouldn't have to open it to learn that).
                if os.path.exists(path):
                    os.unlink(path)
                return
            _atomic_write_text(path, "".join(lines))
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


# ─────────────────────────────────────────────────────────────────────────
# Recorded-outcome-id set — small persisted JSON set living next to the
# outbox (engine.py's status-driven outcome-linkage scan uses this so a
# closed position is never re-recorded on every cycle; the position-outcome
# row's own uuid5 id makes a LOST entry harmless -- a re-send is a
# no-op DB-side via ignore-duplicates, never a duplicate row).
# ─────────────────────────────────────────────────────────────────────────

def _outcome_ids_path(cfg: DecisionsConfig) -> str:
    d = os.path.dirname(cfg.outbox_path) or "."
    return os.path.join(d, _OUTCOME_IDS_FILENAME)


def load_recorded_outcome_ids_sync(cfg: DecisionsConfig) -> set:
    path = _outcome_ids_path(cfg)
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {str(x) for x in data}
    except Exception as e:
        logger.debug(f"agent.decisions store: recorded_outcome_ids read failed: {type(e).__name__}: {e}")
    return set()


def save_recorded_outcome_ids_sync(cfg: DecisionsConfig, ids: set, *, cap: int = 20000) -> None:
    items = sorted(str(x) for x in ids)
    if len(items) > cap:
        items = items[-cap:]
    path = _outcome_ids_path(cfg)
    _atomic_write_text(path, json.dumps(items, ensure_ascii=False))


class DecisionStore:
    def __init__(self, cfg: Optional[DecisionsConfig] = None, *, clock=time.monotonic) -> None:
        self._cfg = cfg or from_env()
        self._clock = clock
        self._queue: "asyncio.Queue[tuple[str, list[dict]]]" = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._writer_task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._dropped = 0
        self._last_drop_warn = 0.0
        self._last_drain_attempt = 0.0
        self._logged_disabled = False

    def update_config(self, cfg: DecisionsConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # enqueue — SYNC, non-blocking, never raises
    # ------------------------------------------------------------------

    def enqueue(self, table: str, rows: list[dict]) -> None:
        if not rows:
            return
        try:
            if not _supa_base() or not _supa_key():
                if not self._logged_disabled:
                    logger.info(
                        "agent.decisions store: SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY not set -- "
                        "decisions store disabled (no-op)"
                    )
                    self._logged_disabled = True
                return
            self._ensure_writer_started()
            self._queue.put_nowait((table, list(rows)))
        except asyncio.QueueFull:
            self._dropped += len(rows)
            now = self._clock()
            if now - self._last_drop_warn >= 60.0:
                logger.warning(
                    f"agent.decisions store: write queue full -- dropped {self._dropped} row(s) total so far"
                )
                self._last_drop_warn = now
        except Exception as e:
            logger.debug(f"agent.decisions store: enqueue failed for table={table}: {type(e).__name__}: {e}")

    def _ensure_writer_started(self) -> None:
        if self._writer_task is not None and not self._writer_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop yet -- the next enqueue() retries
        self._writer_task = loop.create_task(self._writer_loop())

    # ------------------------------------------------------------------
    # Background writer
    # ------------------------------------------------------------------

    async def _writer_loop(self) -> None:
        while True:
            table, rows = await self._queue.get()
            try:
                await self._write_batch(table, rows)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"agent.decisions store: _write_batch raised unexpectedly: {type(e).__name__}: {e}")
            finally:
                self._queue.task_done()

    async def _write_batch(self, table: str, rows: list[dict]) -> None:
        result = await self._post_batch(table, rows)
        if result in (_RETRY, _CONFLICT):
            await self._append_outbox(table, rows)
            return
        await self._maybe_drain_outbox()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_WRITE_TIMEOUT_SECONDS, follow_redirects=True)
        return self._client

    async def _post_batch(self, table: str, rows: list[dict]) -> str:
        """_OK, _RETRY (network, 5xx, 408/429, auth or schema not exposed yet -- worth keeping in the
        outbox), _CONFLICT (409, an FK miss: outboxed, bounded retries) or _DROP (any other 4xx: the rows themselves are bad, so replaying them can never succeed
        and would only block every later batch behind them in the outbox)."""
        base = _supa_base()
        key = _supa_key()
        if not base or not key or not rows:
            return _RETRY
        url = f"{base}/rest/v1/{table}"
        if table == "answers":
            url += "?on_conflict=question_id,model,role"
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Profile": "decisions",
            "Content-Type": "application/json",
            "Prefer": "resolution=ignore-duplicates,return=minimal",
        }
        try:
            client = self._get_client()
            resp = await client.post(url, headers=headers, json=rows)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"agent.decisions store: POST {table} failed: {type(e).__name__}: {e}")
            return _RETRY
        code = resp.status_code
        if code < 300:
            return _OK
        if code == 409:
            logger.debug(f"agent.decisions store: POST {table} returned 409: {_error_summary(resp)}")
            return _CONFLICT
        if code >= 500 or code in _RETRY_STATUSES:
            logger.debug(f"agent.decisions store: POST {table} returned {code}: {_error_summary(resp)}")
            return _RETRY
        logger.error(
            f"agent.decisions store: POST {table} rejected ({code}), dropping {len(rows)} row(s): {_error_summary(resp)}"
        )
        return _DROP

    async def _append_outbox(self, table: str, rows: list[dict]) -> None:
        try:
            entry = json.dumps({"table": table, "rows": rows}, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"agent.decisions store: outbox entry for table={table} not serializable: {type(e).__name__}: {e}")
            return
        try:
            loop = asyncio.get_running_loop()
            await asyncio.shield(
                loop.run_in_executor(None, _append_outbox_sync, self._cfg.outbox_path, max(1, self._cfg.outbox_max_lines), entry)
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"agent.decisions store: outbox append failed for table={table}: {type(e).__name__}: {e}")

    async def _maybe_drain_outbox(self) -> None:
        now = self._clock()
        if now - self._last_drain_attempt < _DRAIN_MIN_INTERVAL_SECONDS:
            return
        self._last_drain_attempt = now
        try:
            loop = asyncio.get_running_loop()
            lines = await loop.run_in_executor(None, _read_outbox_sync, self._cfg.outbox_path)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"agent.decisions store: outbox read (drain) failed: {type(e).__name__}: {e}")
            return
        if not lines:
            return

        remaining = list(lines)
        for line in lines:
            try:
                entry = json.loads(line)
                table = entry.get("table", "")
                rows = entry.get("rows", [])
                conflicts = int(entry.get("conflicts", 0) or 0)
            except Exception:
                # A corrupt line can never be replayed -- drop it (never
                # stall the drain on one bad line forever) and keep going.
                remaining.pop(0)
                continue
            result = await self._post_batch(table, rows)
            if result == _RETRY:
                break  # preserve FK order: stop at the first retryable failure, retry from here next time
            if result == _CONFLICT:
                # A 409 is an FK miss: the parent batch should have landed first (the outbox and the
                # queue are FIFO). If the parent was itself rejected it never will, so after
                # _MAX_CONFLICTS drains the line is dropped instead of blocking every later batch.
                # (Outages are _RETRY and never count.)
                conflicts += 1
                if conflicts < _MAX_CONFLICTS:
                    entry["conflicts"] = conflicts
                    # Lines keep their "\n" (read with it, rewritten by plain join).
                    remaining[0] = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
                    break
                logger.error(
                    f"agent.decisions store: dropping {len(rows)} {table} row(s) after {conflicts} FK conflicts "
                    f"(parent rows never landed)"
                )
            remaining.pop(0)  # _OK, _DROP (already logged) or exhausted conflicts: never replayed again

        if remaining != lines:
            try:
                await asyncio.shield(
                    loop.run_in_executor(None, _rewrite_outbox_sync, self._cfg.outbox_path, remaining)
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"agent.decisions store: outbox rewrite after drain failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Flush what's already queued with a short timeout; whatever
        remains goes to the outbox. Never raises, never blocks the event
        loop for long."""
        if self._writer_task is not None and not self._writer_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._queue.join()), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"agent.decisions store: aclose flush-wait failed: {type(e).__name__}: {e}")
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"agent.decisions store: writer task raised on aclose: {type(e).__name__}: {e}")

        leftover: list[tuple[str, list[dict]]] = []
        while True:
            try:
                leftover.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        for table, rows in leftover:
            try:
                await self._append_outbox(table, rows)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"agent.decisions store: aclose outbox flush failed: {type(e).__name__}: {e}")

        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.debug(f"agent.decisions store: client close failed: {type(e).__name__}: {e}")
            self._client = None
