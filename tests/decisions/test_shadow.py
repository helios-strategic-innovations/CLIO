"""Tests for agent.decisions.shadow.ShadowFanout — fan-out to configured
local-model endpoints, validation via agent.dpof.jev.validate_answers,
in-flight bound, cancellation on aclose(). No network: httpx.MockTransport
(injected by monkeypatching ShadowFanout._get_client, mirroring the DecisionStore
test's own transport-injection style) plus a fake DecisionStore stand-in that
just records enqueue() calls.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import httpx
import pytest

from agent.decisions.config import DecisionsConfig
from agent.decisions.shadow import ShadowFanout


class FakeStore:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, list[dict]]] = []

    def enqueue(self, table: str, rows: list[dict]) -> None:
        self.enqueued.append((table, list(rows)))


def _cfg(**overrides) -> DecisionsConfig:
    base = dict(shadow=True, shadow_endpoints=(("alpha", "http://alpha.test"),), shadow_timeout_seconds=2.0)
    base.update(overrides)
    return replace(DecisionsConfig(), **base)


def _question_rows():
    from agent.decisions.records import build_question_rows

    return build_question_rows(
        producer="dpof_entry_veto", producer_ref={}, call_id="c1", state={"a": 1},
        questions={
            "setup": {"type": "choice", "instructions": "x", "criteria": {"aligned": "a", "mixed": "b"}},
        },
    )


def _install_transport(fanout: ShadowFanout, handler) -> None:
    fanout._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


# ─────────────────────────────────────────────────────────────────────────
# launch() — gating
# ─────────────────────────────────────────────────────────────────────────

def test_launch_noop_when_shadow_disabled():
    store = FakeStore()
    fanout = ShadowFanout(_cfg(shadow=False), store)

    async def _t():
        fanout.launch(_question_rows(), {"a": 1}, {"setup": {"type": "choice"}})
        await asyncio.sleep(0.02)
        assert store.enqueued == []
        assert len(fanout._tasks) == 0

    asyncio.run(_t())


def test_launch_noop_when_no_endpoints_configured():
    store = FakeStore()
    fanout = ShadowFanout(_cfg(shadow_endpoints=()), store)

    async def _t():
        fanout.launch(_question_rows(), {"a": 1}, {"setup": {"type": "choice"}})
        await asyncio.sleep(0.02)
        assert store.enqueued == []

    asyncio.run(_t())


def test_launch_noop_with_no_question_rows():
    store = FakeStore()
    fanout = ShadowFanout(_cfg(), store)

    async def _t():
        fanout.launch([], {"a": 1}, {"setup": {"type": "choice"}})
        await asyncio.sleep(0.02)
        assert store.enqueued == []

    asyncio.run(_t())


def test_launch_noop_without_running_loop():
    store = FakeStore()
    fanout = ShadowFanout(_cfg(), store)
    fanout.launch(_question_rows(), {"a": 1}, {"setup": {"type": "choice"}})  # no running loop -- must not raise
    assert len(fanout._tasks) == 0


# ─────────────────────────────────────────────────────────────────────────
# Success
# ─────────────────────────────────────────────────────────────────────────

def _valid_shadow_body(model="alpha-9b@rev1"):
    return {
        "model": model,
        "answers": {"setup": {"type": "choice", "choice": "aligned", "probabilities": {"aligned": 0.7, "mixed": 0.3}, "confidence": 0.8}},
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def test_launch_success_enqueues_answer_row_with_model_at_name(monkeypatch):
    store = FakeStore()
    fanout = ShadowFanout(_cfg(), store)
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json=_valid_shadow_body())

    async def _t():
        _install_transport(fanout, handler)
        qrows = _question_rows()
        questions = {"setup": {"type": "choice", "instructions": "x", "criteria": {"aligned": "a", "mixed": "b"}}}
        fanout.launch(qrows, {"a": 1}, questions)
        assert await _wait_until(lambda: bool(store.enqueued))
        assert captured["body"]["questions"] == questions
        table, rows = store.enqueued[0]
        assert table == "answers"
        assert len(rows) == 1
        assert rows[0]["role"] == "shadow"
        assert rows[0]["endpoint"] == "alpha"
        assert rows[0]["model"] == "alpha@alpha-9b@rev1"
        assert rows[0]["choice"] == "aligned"
        assert rows[0]["error"] is None
        await fanout.aclose()

    asyncio.run(_t())


def test_launch_model_missing_from_payload_uses_bare_name(monkeypatch):
    store = FakeStore()
    fanout = ShadowFanout(_cfg(), store)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = _valid_shadow_body()
        del body["model"]
        return httpx.Response(200, json=body)

    async def _t():
        _install_transport(fanout, handler)
        qrows = _question_rows()
        questions = {"setup": {"type": "choice", "instructions": "x", "criteria": {"aligned": "a", "mixed": "b"}}}
        fanout.launch(qrows, {"a": 1}, questions)
        assert await _wait_until(lambda: bool(store.enqueued))
        _, rows = store.enqueued[0]
        assert rows[0]["model"] == "alpha"
        await fanout.aclose()

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# Failure modes
# ─────────────────────────────────────────────────────────────────────────

def test_launch_timeout_enqueues_error_row():
    store = FakeStore()
    fanout = ShadowFanout(_cfg(shadow_timeout_seconds=0.2), store)

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json=_valid_shadow_body())  # pragma: no cover

    async def _t():
        _install_transport(fanout, handler)
        qrows = _question_rows()
        questions = {"setup": {"type": "choice", "instructions": "x", "criteria": {"aligned": "a", "mixed": "b"}}}
        fanout.launch(qrows, {"a": 1}, questions)
        assert await _wait_until(lambda: bool(store.enqueued), timeout=3.0)
        _, rows = store.enqueued[0]
        assert rows[0]["error"] == "timeout"
        assert rows[0]["probabilities"] is None
        await fanout.aclose()

    asyncio.run(_t())


def test_launch_http_error_enqueues_error_row():
    store = FakeStore()
    fanout = ShadowFanout(_cfg(), store)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async def _t():
        _install_transport(fanout, handler)
        qrows = _question_rows()
        questions = {"setup": {"type": "choice", "instructions": "x", "criteria": {"aligned": "a", "mixed": "b"}}}
        fanout.launch(qrows, {"a": 1}, questions)
        assert await _wait_until(lambda: bool(store.enqueued))
        _, rows = store.enqueued[0]
        assert rows[0]["error"] == "http_500"
        await fanout.aclose()

    asyncio.run(_t())


def test_launch_invalid_answers_enqueues_error_row():
    store = FakeStore()
    fanout = ShadowFanout(_cfg(), store)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "alpha-9b", "answers": {"setup": {"type": "choice", "choice": "nope", "probabilities": {}, "confidence": 5.0}}, "usage": {"input_tokens": 1, "output_tokens": 1}})

    async def _t():
        _install_transport(fanout, handler)
        qrows = _question_rows()
        questions = {"setup": {"type": "choice", "instructions": "x", "criteria": {"aligned": "a", "mixed": "b"}}}
        fanout.launch(qrows, {"a": 1}, questions)
        assert await _wait_until(lambda: bool(store.enqueued))
        _, rows = store.enqueued[0]
        assert "invalid_answers" in rows[0]["error"]
        assert rows[0]["probabilities"] is None
        await fanout.aclose()

    asyncio.run(_t())


def test_launch_connection_error_enqueues_error_row():
    store = FakeStore()
    fanout = ShadowFanout(_cfg(), store)

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated")

    async def _t():
        _install_transport(fanout, handler)
        qrows = _question_rows()
        questions = {"setup": {"type": "choice", "instructions": "x", "criteria": {"aligned": "a", "mixed": "b"}}}
        fanout.launch(qrows, {"a": 1}, questions)
        assert await _wait_until(lambda: bool(store.enqueued))
        _, rows = store.enqueued[0]
        assert rows[0]["error"]
        await fanout.aclose()

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# In-flight bound + never affects the served decision
# ─────────────────────────────────────────────────────────────────────────

def test_launch_bounds_in_flight_tasks_total():
    store = FakeStore()
    many_endpoints = tuple((f"ep{i}", f"http://ep{i}.test") for i in range(12))
    fanout = ShadowFanout(_cfg(shadow_endpoints=many_endpoints), store)

    async def never_respond(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(999)
        return httpx.Response(200, json={})  # pragma: no cover

    async def _t():
        _install_transport(fanout, never_respond)
        qrows = _question_rows()
        questions = {"setup": {"type": "choice", "instructions": "x", "criteria": {"aligned": "a", "mixed": "b"}}}
        fanout.launch(qrows, {"a": 1}, questions)
        await asyncio.sleep(0.02)
        assert len(fanout._tasks) == 8  # capped at _MAX_INFLIGHT, not 12
        await fanout.aclose()

    asyncio.run(_t())


def test_aclose_cancels_in_flight_tasks():
    store = FakeStore()
    fanout = ShadowFanout(_cfg(), store)

    async def never_respond(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(999)
        return httpx.Response(200, json={})  # pragma: no cover

    async def _t():
        _install_transport(fanout, never_respond)
        qrows = _question_rows()
        questions = {"setup": {"type": "choice", "instructions": "x", "criteria": {"aligned": "a", "mixed": "b"}}}
        fanout.launch(qrows, {"a": 1}, questions)
        await asyncio.sleep(0.02)
        assert len(fanout._tasks) == 1
        await fanout.aclose()
        assert len(fanout._tasks) == 0

    asyncio.run(_t())


def test_launch_never_raises_and_returns_none():
    """launch() itself is fire-and-forget -- it must never raise and never
    return anything the caller could mistakenly treat as the served result."""
    store = FakeStore()
    fanout = ShadowFanout(_cfg(), store)

    async def _t():
        result = fanout.launch(_question_rows(), {"a": 1}, {"setup": {"type": "choice"}})
        assert result is None
        await fanout.aclose()

    asyncio.run(_t())
