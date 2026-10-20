"""CLIO decisions producer — shadow fan-out.

Fires the SAME `{state, questions}` a DPOF Jev call built at configured local
model endpoints (`CLIO_DECISIONS_SHADOW_ENDPOINTS="alpha=http://...,beta=http://..."`),
validates each response with `agent.dpof.jev.validate_answers` (the same
strict validation Jev's own responses go through), and records the result as
`decisions.answers` role="shadow". The served decision (the veto/sentinel
outcome DPOF actually acts on) is never touched by anything in this module --
`launch()` only ever enqueues background work, never returns a result to its
caller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import httpx

from agent.decisions import records
from agent.decisions.config import DecisionsConfig

logger = logging.getLogger("clio.decisions.shadow")

_MAX_INFLIGHT = 8


class ShadowFanout:
    def __init__(self, cfg: DecisionsConfig, store: Any, *, clock=time.monotonic) -> None:
        self._cfg = cfg
        self._store = store
        self._clock = clock
        self._client: Optional[httpx.AsyncClient] = None
        self._tasks: set[asyncio.Task] = set()
        self._skipped = 0
        self._last_skip_warn = 0.0

    def update_config(self, cfg: DecisionsConfig) -> None:
        self._cfg = cfg

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(follow_redirects=False)
        return self._client

    # ------------------------------------------------------------------
    # launch — fire-and-forget, never awaited by the caller, never raises
    # ------------------------------------------------------------------

    def launch(self, question_rows: list[dict], state: dict, questions: dict) -> None:
        """One background Task per configured endpoint. At most `_MAX_INFLIGHT`
        fan-out tasks in flight TOTAL across every call (never per call) --
        beyond that, an endpoint is skipped for this call and counted (at
        most one warning per minute)."""
        if not self._cfg.shadow or not self._cfg.shadow_endpoints or not question_rows:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop -- nothing to fire into

        for name, url in self._cfg.shadow_endpoints:
            if len(self._tasks) >= _MAX_INFLIGHT:
                self._skipped += 1
                now = self._clock()
                if now - self._last_skip_warn >= 60.0:
                    logger.warning(
                        f"agent.decisions shadow: at the {_MAX_INFLIGHT} in-flight cap -- "
                        f"skipped {self._skipped} fan-out(s) total so far"
                    )
                    self._last_skip_warn = now
                continue
            task = asyncio.ensure_future(self._run_one(name, url, question_rows, state, questions))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_one(self, name: str, url: str, question_rows: list[dict], state: dict, questions: dict) -> None:
        try:
            await self._run_one_impl(name, url, question_rows, state, questions)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"agent.decisions shadow: {name} fan-out failed unexpectedly: {type(e).__name__}: {e}")

    async def _run_one_impl(self, name: str, url: str, question_rows: list[dict], state: dict, questions: dict) -> None:
        timeout = max(0.2, min(10.0, self._cfg.shadow_timeout_seconds))
        started = self._clock()
        payload: Optional[dict] = None
        error = ""
        try:
            client = self._get_client()
            resp = await asyncio.wait_for(
                client.post(url, json={"state": state, "questions": questions}, timeout=timeout),
                timeout=timeout,
            )
            if resp.status_code == 200:
                payload = resp.json()
            else:
                error = f"http_{resp.status_code}"
        except asyncio.TimeoutError:
            error = "timeout"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error = f"{type(e).__name__}"
        latency_ms = (self._clock() - started) * 1000.0

        from agent.dpof.jev import validate_answers  # lazy import -- module-boundary discipline

        answers: Optional[dict] = None
        model_field = ""
        if payload is not None and not error:
            clean, val_errors = validate_answers(questions, payload)
            if val_errors:
                error = f"invalid_answers:{','.join(val_errors[:5])}"
            else:
                answers = clean
                if isinstance(payload, dict):
                    model_field = str(payload.get("model") or "")

        model_label = f"{name}@{model_field}" if model_field else name
        rows = records.build_answer_rows(
            question_rows, model=model_label, role="shadow", endpoint=name,
            answers=answers, error=error, latency_ms=latency_ms,
        )
        self._store.enqueue("answers", rows)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        tasks = [t for t in list(self._tasks) if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.debug(f"agent.decisions shadow: client close failed: {type(e).__name__}: {e}")
            self._client = None
