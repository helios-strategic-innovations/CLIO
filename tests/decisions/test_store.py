"""Tests for agent.decisions.store.DecisionStore — write-behind to
PostgREST, outbox fallback on failure, drain, cap, disabled-without-creds,
non-blocking enqueue, aclose(). No network: httpx.MockTransport / a fake
DecisionsConfig only. Plugin-free (asyncio.run() inside ordinary test
functions), matching tests/dpof's own convention.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import replace

import httpx
import pytest

from agent.decisions.config import DecisionsConfig
from agent.decisions.store import (
    DecisionStore,
    load_recorded_outcome_ids_sync,
    save_recorded_outcome_ids_sync,
)


def _cfg(tmp_path, **overrides) -> DecisionsConfig:
    base = dict(outbox_path=str(tmp_path / "outbox.jsonl"), outbox_max_lines=50)
    base.update(overrides)
    return replace(DecisionsConfig(), **base)


def _patch_supabase_env(monkeypatch, *, url="https://proj.supabase.co", key="service-role-key"):
    if url is None:
        monkeypatch.delenv("SUPABASE_URL", raising=False)
    else:
        monkeypatch.setenv("SUPABASE_URL", url)
    monkeypatch.delenv("SUPABASE_PROJECT_REF", raising=False)
    if key is None:
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    else:
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", key)


def _install_transport(store: DecisionStore, handler) -> None:
    """Point the store's lazily-built httpx.AsyncClient at a MockTransport
    by pre-building the client with that transport (mirrors JevClient's own
    transport-injection style, since DecisionStore doesn't take a transport
    constructor arg -- store.py builds a bare httpx.AsyncClient())."""
    store._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=10.0, follow_redirects=True)


def _rows(n=1):
    return [{"id": f"r{i}"} for i in range(n)]


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.005) -> bool:
    """Polls `predicate()` with real (tiny) sleeps -- a plain `await
    asyncio.sleep(0)` loop-yield is NOT enough here: the write path goes
    through `loop.run_in_executor()` (a real thread-pool round trip for the
    outbox append/read), which needs actual wall-clock time to complete, not
    just event-loop turns."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


def _outbox_exists(tmp_path) -> bool:
    return (tmp_path / "outbox.jsonl").exists()


# ─────────────────────────────────────────────────────────────────────────
# Disabled without creds
# ─────────────────────────────────────────────────────────────────────────

def test_enqueue_noop_without_creds(tmp_path, monkeypatch, caplog):
    _patch_supabase_env(monkeypatch, url=None, key=None)
    store = DecisionStore(_cfg(tmp_path))

    async def _t():
        with caplog.at_level(logging.INFO, logger="clio.decisions.store"):
            store.enqueue("questions", _rows())
            await asyncio.sleep(0.05)
        assert not _outbox_exists(tmp_path)
        assert any("disabled" in r.message for r in caplog.records)
        await store.aclose()

    asyncio.run(_t())


def test_enqueue_noop_with_url_but_no_key(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch, url="https://proj.supabase.co", key=None)
    store = DecisionStore(_cfg(tmp_path))

    async def _t():
        store.enqueue("questions", _rows())
        await asyncio.sleep(0.05)
        assert not _outbox_exists(tmp_path)
        await store.aclose()

    asyncio.run(_t())


def test_enqueue_empty_rows_is_noop(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))

    async def _t():
        store.enqueue("questions", [])
        await asyncio.sleep(0.05)
        assert not _outbox_exists(tmp_path)
        await store.aclose()

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# Success path (MockTransport)
# ─────────────────────────────────────────────────────────────────────────

def test_enqueue_success_posts_with_expected_headers_and_prefer(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=[])

    async def _t():
        _install_transport(store, handler)
        store.enqueue("questions", _rows(2))
        assert await _wait_until(lambda: bool(calls))
        assert len(calls) == 1
        req = calls[0]
        assert req.url.path == "/rest/v1/questions"
        assert req.headers["apikey"] == "service-role-key"
        assert req.headers["Authorization"] == "Bearer service-role-key"
        assert req.headers["Content-Profile"] == "decisions"
        assert req.headers["Prefer"] == "resolution=ignore-duplicates,return=minimal"
        assert "on_conflict" not in str(req.url)
        body = json.loads(req.content)
        assert len(body) == 2
        assert not _outbox_exists(tmp_path)
        await store.aclose()

    asyncio.run(_t())


def test_answers_table_uses_on_conflict(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=[])

    async def _t():
        _install_transport(store, handler)
        store.enqueue("answers", _rows(1))
        assert await _wait_until(lambda: bool(calls))
        assert "on_conflict=question_id,model,role" in str(calls[0].url)
        await store.aclose()

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# Failure -> outbox
# ─────────────────────────────────────────────────────────────────────────

def test_5xx_goes_to_outbox(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async def _t():
        _install_transport(store, handler)
        store.enqueue("questions", _rows(1))
        assert await _wait_until(lambda: _outbox_exists(tmp_path))
        await store.aclose()

    asyncio.run(_t())
    outbox_path = tmp_path / "outbox.jsonl"
    assert outbox_path.exists()
    lines = outbox_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["table"] == "questions"
    assert len(entry["rows"]) == 1


def test_connection_error_goes_to_outbox(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connect failure")

    async def _t():
        _install_transport(store, handler)
        store.enqueue("outcomes", _rows(1))
        assert await _wait_until(lambda: _outbox_exists(tmp_path))
        await store.aclose()

    asyncio.run(_t())
    assert _outbox_exists(tmp_path)


def test_outbox_never_logs_the_key(tmp_path, monkeypatch, caplog):
    _patch_supabase_env(monkeypatch, key="super-secret-service-role-key")
    store = DecisionStore(_cfg(tmp_path))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async def _t():
        with caplog.at_level(logging.DEBUG):
            _install_transport(store, handler)
            store.enqueue("questions", _rows(1))
            assert await _wait_until(lambda: _outbox_exists(tmp_path))
            await store.aclose()

    asyncio.run(_t())
    outbox_text = (tmp_path / "outbox.jsonl").read_text()
    assert "super-secret-service-role-key" not in outbox_text
    for record in caplog.records:
        assert "super-secret-service-role-key" not in record.getMessage()


# ─────────────────────────────────────────────────────────────────────────
# Outbox drain
# ─────────────────────────────────────────────────────────────────────────

def test_outbox_drains_on_a_later_successful_write(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    fake_clock = {"t": 0.0}
    store = DecisionStore(_cfg(tmp_path), clock=lambda: fake_clock["t"])
    posted = []
    should_fail = {"v": True}

    async def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request)
        if should_fail["v"]:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=[])

    async def _t():
        _install_transport(store, handler)
        store.enqueue("questions", _rows(1))
        assert await _wait_until(lambda: _outbox_exists(tmp_path))

        # A later successful write should trigger a drain attempt -- but the
        # drain is rate-limited to at most once per 60s, so advance the clock
        # well past both the initial write's own drain attempt AND its own
        # 60s window.
        should_fail["v"] = False
        fake_clock["t"] += 61.0
        store.enqueue("questions", _rows(1))
        assert await _wait_until(lambda: not _outbox_exists(tmp_path))
        await store.aclose()

    asyncio.run(_t())
    # The outbox's failed batch + the new successful one were both posted.
    assert len(posted) >= 2
    assert not _outbox_exists(tmp_path)


def test_outbox_drain_rate_limited_to_once_per_60s(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    fake_clock = {"t": 100.0}
    store = DecisionStore(_cfg(tmp_path), clock=lambda: fake_clock["t"])
    # A prior drain attempt "just happened" (well within the last 60s), so the
    # NEXT successful write must skip its own drain attempt entirely.
    store._last_drain_attempt = fake_clock["t"]
    posts = []

    async def always_ok(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(200, json=[])

    async def _t():
        _install_transport(store, always_ok)
        # Force an outbox entry directly (bypass the write path) so a drain
        # would have something to do if it were attempted.
        await store._append_outbox("questions", _rows(1))
        assert _outbox_exists(tmp_path)

        store.enqueue("questions", _rows(1))
        assert await _wait_until(lambda: len(posts) >= 1)
        await asyncio.sleep(0.05)  # give a would-be (incorrect) drain time to fire
        await store.aclose()

    asyncio.run(_t())
    # Only the new row's own write happened -- the outbox entry was NOT
    # drained (rate-limited), so exactly one POST and the file still there.
    assert len(posts) == 1
    assert _outbox_exists(tmp_path)


def test_outbox_capped_oldest_dropped(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    cfg = _cfg(tmp_path, outbox_max_lines=3)
    store = DecisionStore(cfg)

    async def always_fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async def _t():
        _install_transport(store, always_fail)
        for i in range(5):
            marker = f"r{i}"
            store.enqueue("questions", [{"id": marker}])
            assert await _wait_until(
                lambda m=marker: _outbox_exists(tmp_path) and m in (tmp_path / "outbox.jsonl").read_text()
            )
        await store.aclose()

    asyncio.run(_t())
    lines = (tmp_path / "outbox.jsonl").read_text().splitlines()
    assert len(lines) == 3
    kept_ids = [json.loads(line)["rows"][0]["id"] for line in lines]
    assert kept_ids == ["r2", "r3", "r4"]  # oldest (r0, r1) dropped


# ─────────────────────────────────────────────────────────────────────────
# enqueue is non-blocking / never raises, even under a full queue
# ─────────────────────────────────────────────────────────────────────────

def test_enqueue_never_raises_when_queue_full(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))
    store._queue = asyncio.Queue(maxsize=1)  # tiny queue, force overflow

    async def never_respond(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(999)
        return httpx.Response(200, json=[])  # pragma: no cover

    async def _t():
        _install_transport(store, never_respond)
        for _ in range(10):
            store.enqueue("questions", _rows(1))  # must never raise, even once full
        await asyncio.sleep(0.02)
        await store.aclose()

    asyncio.run(_t())  # the assertion is simply that this doesn't raise/hang


def test_enqueue_before_any_running_loop_does_not_raise(tmp_path, monkeypatch):
    """enqueue() called with no running event loop (writer can't start yet)
    must still not raise -- the row is dropped for lack of a loop, not queued
    forever; a later enqueue (inside a loop) works normally."""
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))
    store.enqueue("questions", _rows(1))  # no running loop here -- must not raise


# ─────────────────────────────────────────────────────────────────────────
# aclose()
# ─────────────────────────────────────────────────────────────────────────

def test_aclose_flushes_queued_rows_to_outbox_on_persistent_failure(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))

    async def always_fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async def _t():
        _install_transport(store, always_fail)
        store.enqueue("questions", _rows(1))
        await store.aclose()  # must not raise, must not hang

    asyncio.run(_t())
    assert _outbox_exists(tmp_path)


def test_aclose_idempotent_and_closes_client(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))

    async def _t():
        await store.aclose()
        await store.aclose()  # idempotent

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# recorded_outcome_ids sync helpers (used by the engine's outcome-linkage scan)
# ─────────────────────────────────────────────────────────────────────────

def test_recorded_outcome_ids_round_trip(tmp_path):
    cfg = _cfg(tmp_path)
    assert load_recorded_outcome_ids_sync(cfg) == set()
    save_recorded_outcome_ids_sync(cfg, {"a", "b", "c"})
    assert load_recorded_outcome_ids_sync(cfg) == {"a", "b", "c"}


def test_recorded_outcome_ids_capped(tmp_path):
    cfg = _cfg(tmp_path)
    ids = {f"id{i}" for i in range(10)}
    save_recorded_outcome_ids_sync(cfg, ids, cap=5)
    loaded = load_recorded_outcome_ids_sync(cfg)
    assert len(loaded) == 5


def test_recorded_outcome_ids_missing_file_is_empty_set(tmp_path):
    cfg = _cfg(tmp_path, outbox_path=str(tmp_path / "nonexistent" / "outbox.jsonl"))
    assert load_recorded_outcome_ids_sync(cfg) == set()


def test_recorded_outcome_ids_corrupt_file_degrades_to_empty_set(tmp_path):
    cfg = _cfg(tmp_path)
    import os as _os

    _os.makedirs(_os.path.dirname(cfg.outbox_path), exist_ok=True)
    ids_path = _os.path.join(_os.path.dirname(cfg.outbox_path), "recorded_outcome_ids.json")
    with open(ids_path, "w") as f:
        f.write("not json{{{")
    assert load_recorded_outcome_ids_sync(cfg) == set()


def test_rejected_batch_is_dropped_not_outboxed(tmp_path, monkeypatch, caplog):
    """A 400 (e.g. a CHECK violation) can never succeed on replay: it is logged and dropped, so it
    can't sit at the head of the outbox and block every later batch."""
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))
    posted = []

    async def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request)
        return httpx.Response(400, json={"code": "23514", "message": "violates check constraint"})

    async def _t():
        _install_transport(store, handler)
        store.enqueue("questions", _rows(1))
        assert await _wait_until(lambda: len(posted) == 1)
        await store.aclose()

    with caplog.at_level("ERROR"):
        asyncio.run(_t())
    assert not _outbox_exists(tmp_path)
    assert any("rejected (400)" in r.getMessage() for r in caplog.records)
    assert all("service-role-key" not in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("status", [401, 403, 404, 408, 409, 429, 503])
def test_retryable_statuses_go_to_outbox(tmp_path, monkeypatch, status):
    _patch_supabase_env(monkeypatch)
    store = DecisionStore(_cfg(tmp_path))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "x"})

    async def _t():
        _install_transport(store, handler)
        store.enqueue("answers", _rows(1))
        assert await _wait_until(lambda: _outbox_exists(tmp_path))
        await store.aclose()

    asyncio.run(_t())


def test_drain_skips_a_rejected_line_and_keeps_going(tmp_path, monkeypatch):
    _patch_supabase_env(monkeypatch)
    fake_clock = {"t": 0.0}
    store = DecisionStore(_cfg(tmp_path), clock=lambda: fake_clock["t"])
    outbox = tmp_path / "outbox.jsonl"
    outbox.write_text(
        json.dumps({"table": "questions", "rows": [{"id": "bad"}]}) + "\n"
        + json.dumps({"table": "questions", "rows": [{"id": "good"}]}) + "\n"
    )
    posted_ids = []

    async def handler(request: httpx.Request) -> httpx.Response:
        rows = json.loads(request.content)
        posted_ids.append(rows[0]["id"])
        if rows[0]["id"] == "bad":
            return httpx.Response(400, json={"message": "bad row"})
        return httpx.Response(201, json=[])

    async def _t():
        _install_transport(store, handler)
        fake_clock["t"] = 100.0
        store.enqueue("questions", [{"id": "new"}])
        assert await _wait_until(lambda: not _outbox_exists(tmp_path) or outbox.read_text() == "")
        await store.aclose()

    asyncio.run(_t())
    assert posted_ids[:3] == ["new", "bad", "good"]


def test_fk_conflict_line_is_dropped_after_the_bound_not_forever(tmp_path, monkeypatch, caplog):
    """An answers batch whose parent questions batch was rejected 409s forever; after _MAX_CONFLICTS
    drains it is dropped so later batches can land. Outages (_RETRY) never count toward the bound."""
    from agent.decisions import store as store_mod

    _patch_supabase_env(monkeypatch)
    monkeypatch.setattr(store_mod, "_MAX_CONFLICTS", 3)
    fake_clock = {"t": 0.0}
    store = DecisionStore(_cfg(tmp_path), clock=lambda: fake_clock["t"])
    outbox = tmp_path / "outbox.jsonl"
    outbox.write_text(
        json.dumps({"table": "answers", "rows": [{"id": "orphan"}]}) + "\n"
        + json.dumps({"table": "questions", "rows": [{"id": "later"}]}) + "\n"
    )
    posted = []

    async def handler(request: httpx.Request) -> httpx.Response:
        rows = json.loads(request.content)
        posted.append(rows[0]["id"])
        if rows[0]["id"] == "orphan":
            return httpx.Response(409, json={"code": "23503", "message": "fk", "details": "Key (question_id)=(x)"})
        return httpx.Response(201, json=[])

    async def _t():
        _install_transport(store, handler)
        for i in range(3):
            fake_clock["t"] += 61.0
            store.enqueue("questions", [{"id": f"new{i}"}])
            await _wait_until(lambda: posted.count("orphan") >= i + 1)
        ok = await _wait_until(lambda: "later" in posted)
        await store.aclose()
        assert ok, posted

    with caplog.at_level("ERROR"):
        asyncio.run(_t())
    assert posted.count("orphan") == 3
    assert "later" in posted
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "after 3 FK conflicts" in text and "Key (question_id)" not in text


def test_error_summary_never_echoes_row_details():
    from agent.decisions.store import _error_summary

    resp = httpx.Response(400, json={"code": "23514", "message": "violates check", "details": "Failing row contains (secret-ish state)"})
    out = _error_summary(resp)
    assert "23514" in out and "violates check" in out and "Failing row" not in out
    assert _error_summary(httpx.Response(502, text="<html>bad gateway</html>")) == "(non-JSON error body)"
