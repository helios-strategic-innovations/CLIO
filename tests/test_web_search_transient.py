"""R-26: transient DuckDuckGo network errors are retried with backoff, fail over once,
and never logged as ERROR tracebacks. The ddgs client is faked; no network, no real sleeps."""

import asyncio
import logging
import sys
import time as real_time
import types
from types import SimpleNamespace

import httpx
import pytest

import agent.integrations.web_search as ws

LOGGER_NAME = "openclaw.web_search"
QUERY = "private-context-query-xyz"
RESULT = {"href": "https://example.com/page", "title": "T", "body": "B"}


class _ScriptedDDGS:
    """Fake ddgs.DDGS. Per-backend outcome queues: an Exception is raised, a list is returned."""

    script: dict = {}
    calls: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results=5, backend=None):
        _ScriptedDDGS.calls.append(backend)
        queue = _ScriptedDDGS.script.get(backend, [])
        if not queue:
            return []
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return list(outcome)


@pytest.fixture
def env(monkeypatch):
    sleeps: list = []
    _ScriptedDDGS.script = {}
    _ScriptedDDGS.calls = []

    fake_ddgs = types.ModuleType("ddgs")
    fake_ddgs.DDGS = _ScriptedDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_ddgs)

    # Patch the module's time facade: sleeps are recorded, never performed.
    monkeypatch.setattr(ws, "_time", SimpleNamespace(time=real_time.time, sleep=sleeps.append))
    monkeypatch.setattr(ws, "_MIN_SEARCH_INTERVAL", 0)
    monkeypatch.setattr(ws, "_last_search_ts", 0.0)
    monkeypatch.setattr(ws, "_search_cache", {})
    monkeypatch.setattr(ws, "_backend_idx", 0)
    monkeypatch.setattr(ws, "_is_allowed_url", lambda url: (True, ""))

    async def _no_budget(*_a, **_k):
        return None

    monkeypatch.setattr(ws, "check_and_record", _no_budget)
    return SimpleNamespace(sleeps=sleeps, script=_ScriptedDDGS.script, calls=_ScriptedDDGS.calls)


def _search():
    return asyncio.run(ws.search_web(QUERY))


def _records(caplog, level):
    return [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno == level]


def _transient_warnings(caplog):
    return [r for r in _records(caplog, logging.WARNING) if "transient network error" in r.getMessage()]


def test_connect_error_twice_then_success_retries_with_backoff(env, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    env.script["duckduckgo"] = [
        httpx.ConnectError("[Errno 104] Connection reset by peer"),
        httpx.ConnectError("[Errno 104] Connection reset by peer"),
        [RESULT],
    ]
    results = _search()
    assert [r["url"] for r in results] == [RESULT["href"]]
    assert env.sleeps == [1, 2]
    assert env.calls == ["duckduckgo", "duckduckgo", "duckduckgo"]
    assert _records(caplog, logging.ERROR) == []
    assert _transient_warnings(caplog) == []


def test_primary_exhausted_then_other_backend_succeeds(env, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    env.script["duckduckgo"] = [httpx.ConnectError("x")] * 3
    env.script["auto"] = [[RESULT]]
    results = _search()
    assert len(results) == 1
    assert env.sleeps == [1, 2]
    assert env.calls == ["duckduckgo", "duckduckgo", "duckduckgo", "auto"]
    assert _records(caplog, logging.ERROR) == []


def test_failure_everywhere_gives_empty_list_one_warning_no_traceback(env, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    env.script["duckduckgo"] = [httpx.ConnectError("x")] * 3
    env.script["auto"] = [httpx.ConnectError("x")]
    assert _search() == []
    assert env.calls == ["duckduckgo", "duckduckgo", "duckduckgo", "auto"]
    assert env.sleeps == [1, 2]
    warnings = _transient_warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].exc_info is None
    msg = warnings[0].getMessage()
    assert "class=ConnectError" in msg
    assert "backend=auto" in msg
    assert QUERY not in msg
    assert _records(caplog, logging.ERROR) == []


def test_generic_exception_is_transient_by_errno_text_alone(env, caplog):
    # The class (RuntimeError) is NOT on the transient list: only the 'Errno 104 ... reset by peer' text marks it.
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    env.script["duckduckgo"] = [RuntimeError("[Errno 104] Connection reset by peer"), [RESULT]]
    assert len(_search()) == 1
    assert env.sleeps == [1]
    assert _records(caplog, logging.ERROR) == []


def test_connection_reset_error_is_transient(env, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    env.script["duckduckgo"] = [ConnectionResetError("[Errno 104] Connection reset by peer"), [RESULT]]
    assert len(_search()) == 1
    assert env.sleeps == [1]
    assert _records(caplog, logging.ERROR) == []


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("x"),
        httpx.ConnectTimeout("x"),
        httpx.ReadTimeout("x"),
        httpx.RemoteProtocolError("x"),
        ConnectionResetError("[Errno 104] Connection reset by peer"),
        OSError(104, "Connection reset by peer"),
    ],
)
def test_transient_classification(exc):
    assert ws._transient_class(exc) is not None


def test_transient_classification_follows_cause_chain():
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = httpx.ConnectError("x")
    assert ws._transient_class(wrapper) == "ConnectError"


def test_runtime_error_stays_error_with_traceback(env, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    env.script["duckduckgo"] = [RuntimeError("boom")]
    assert _search() == []
    assert env.sleeps == []
    assert env.calls == ["duckduckgo"]
    errors = _records(caplog, logging.ERROR)
    assert len(errors) == 1
    assert errors[0].getMessage() == "DuckDuckGo search error: boom"
    assert errors[0].exc_info is not None
    assert _transient_warnings(caplog) == []


def test_429_path_unchanged(env, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    env.script["duckduckgo"] = [
        RuntimeError("429 Too Many Requests"),
        RuntimeError("429 Too Many Requests"),
        [RESULT],
    ]
    assert len(_search()) == 1
    assert env.calls == ["duckduckgo", "duckduckgo", "duckduckgo"]
    # 429 path backoff is 2**attempt + jitter: 1..2 s then 2..3 s, recorded through the same facade.
    assert len(env.sleeps) == 2
    assert 1 <= env.sleeps[0] < 2
    assert 2 <= env.sleeps[1] < 3
    assert _records(caplog, logging.ERROR) == []
    assert _transient_warnings(caplog) == []


def test_no_results_path_unchanged(env, caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    env.script["duckduckgo"] = [RuntimeError("No results.")]
    env.script["auto"] = [[RESULT]]
    assert len(_search()) == 1
    assert env.sleeps == []
    assert env.calls == ["duckduckgo", "auto"]
    assert _records(caplog, logging.ERROR) == []
