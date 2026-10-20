"""CLIO producer for the shared decision record.

Records every decision question DPOF's Jev veto/sentinel asks (and Jev's
answer, when it was actually called), fans the same question out to
configured local-model shadow endpoints, and links a closed position back to
its entry decision with an outcome row -- all fire-and-forget, so a DB outage
or a slow endpoint never touches DPOF's own latency or its veto/sentinel
result.

`DecisionRecorder` is the only thing `agent.dpof.engine` talks to.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from agent.decisions import records
from agent.decisions.config import DecisionsConfig, from_env
from agent.decisions.shadow import ShadowFanout
from agent.decisions.store import DecisionStore

logger = logging.getLogger("clio.decisions")

__all__ = ["DecisionRecorder", "DecisionsConfig", "from_env"]


class DecisionRecorder:
    def __init__(
        self,
        cfg: Optional[DecisionsConfig] = None,
        *,
        store: Optional[Any] = None,
        shadow: Optional[Any] = None,
    ) -> None:
        self._cfg = cfg or from_env()
        self._store = store if store is not None else DecisionStore(self._cfg)
        self._shadow = shadow if shadow is not None else ShadowFanout(self._cfg, self._store)

    @property
    def cfg(self) -> DecisionsConfig:
        return self._cfg

    def update_config(self, cfg: DecisionsConfig) -> None:
        self._cfg = cfg
        try:
            self._store.update_config(cfg)
        except Exception as e:
            logger.debug(f"agent.decisions: store.update_config failed: {type(e).__name__}: {e}")
        try:
            self._shadow.update_config(cfg)
        except Exception as e:
            logger.debug(f"agent.decisions: shadow.update_config failed: {type(e).__name__}: {e}")

    async def start(self) -> None:
        """No-op placeholder for symmetry with the engine's other
        collaborators' lifecycle -- the store's writer task starts lazily on
        first enqueue(), and the shadow fan-out has nothing to pre-warm."""
        return None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_call(
        self,
        producer: str,
        producer_ref: dict,
        state: dict,
        questions: dict,
        *,
        served: Optional[tuple] = None,
        position_id: Optional[str] = None,
    ) -> Optional[str]:
        """Records one `/v1/systemone`-shaped call: one `decisions.questions`
        row per question, an optional `decisions.answers` row per question
        (role="served") when `served` is given, and launches the shadow
        fan-out. SYNC, never raises, never blocks the event loop -- every
        actual write is enqueued to the store's own background writer.

        `served` is `(answers, model, endpoint, latency_ms, error)` -- the
        same shape a `JevVerdict`/`JevEscalation` already carries -- or None
        when nothing was served (Jev OFF: the questions are still recorded,
        just with no served answer, so the eval set doesn't depend on Jev
        being keyed).

        Returns the call_id (str uuid4), or None when decisions recording is
        disabled or something failed unexpectedly."""
        if not self._cfg.enabled:
            return None
        try:
            return self._record_call_impl(producer, producer_ref, state, questions, served=served, position_id=position_id)
        except Exception as e:
            logger.error(f"agent.decisions: record_call failed for producer={producer}: {type(e).__name__}: {e}")
            return None

    def _record_call_impl(
        self, producer: str, producer_ref: dict, state: dict, questions: dict, *,
        served: Optional[tuple], position_id: Optional[str],
    ) -> str:
        call_id = str(uuid.uuid4())
        question_rows = records.build_question_rows(
            producer=producer, producer_ref=producer_ref or {}, call_id=call_id,
            state=state, questions=questions, position_id=position_id,
        )
        if not question_rows:
            return call_id
        self._store.enqueue("questions", question_rows)

        if served is not None:
            answers, model, endpoint, latency_ms, error = served
            answer_rows = records.build_answer_rows(
                question_rows, model=model or "", role="served", endpoint=endpoint or "",
                answers=answers, error=error or "", latency_ms=latency_ms,
            )
            self._store.enqueue("answers", answer_rows)

        try:
            self._shadow.launch(question_rows, state, questions)
        except Exception as e:
            logger.debug(f"agent.decisions: shadow.launch failed: {type(e).__name__}: {e}")

        return call_id

    def record_position_outcome(
        self, position: Any, *, outcome_source: str = "live", call_id: Optional[str] = None,
    ) -> None:
        """Idempotent: the row's id is uuid5(position_id|outcome_source), so
        calling this more than once for the same closed position is safe
        (a DB-side no-op via ignore-duplicates, never a duplicate row).
        `position` may be a `PositionRecord`-shaped mapping, or any object
        with a `to_dict()` method."""
        if not self._cfg.enabled:
            return
        try:
            if isinstance(position, dict):
                pos_dict = position
            elif hasattr(position, "to_dict"):
                pos_dict = position.to_dict()
            else:
                pos_dict = dict(position)
            row = records.build_position_outcome_row(pos_dict, outcome_source=outcome_source, call_id=call_id)
            if row is not None:
                self._store.enqueue("outcomes", [row])
        except Exception as e:
            logger.error(f"agent.decisions: record_position_outcome failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        try:
            await self._shadow.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"agent.decisions: shadow.aclose failed: {type(e).__name__}: {e}")
        try:
            await self._store.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"agent.decisions: store.aclose failed: {type(e).__name__}: {e}")
