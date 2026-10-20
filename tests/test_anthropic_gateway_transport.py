"""OP-06: gateway httpx transport guard (agent/anthropic_gateway.py).

All traffic is scripted in-process (httpx.BaseTransport fakes); sleep/clock/rng
are injected. No network.
"""

import json
import logging
import os
import sys

import anthropic
import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.anthropic_gateway as gw  # noqa: E402
from agent.anthropic_gateway import GatewayStats, GatewayTransport  # noqa: E402

FAKE_KEY = "sk-" + "a" * 64
LOGGER = "openclaw.anthropic_gateway"
REQ_MODEL = "claude-opus-5-5"


class Scripted(httpx.BaseTransport):
    """Returns scripted responses in order; the last one repeats once exhausted."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def handle_request(self, request):
        self.calls.append(request)
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return item() if callable(item) else item


def msg(content=None, output_tokens=7, stop="end_turn", model=REQ_MODEL):
    if content is None:
        content = [{"type": "text", "text": "hello"}]
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": output_tokens},
    }


EMPTY = lambda: msg(content=[], output_tokens=0)  # noqa: E731


def ok_resp(payload):
    return lambda: httpx.Response(200, json=payload)


def json_resp(payload, status):
    return lambda: httpx.Response(status, json=payload)


def request(body=None, path="/v1/messages", method="POST", model=REQ_MODEL):
    if body is None:
        body = {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    return httpx.Request(method, "http://gw.invalid" + path, json=body)


def make(inner, **kw):
    sleeps = []
    kw.setdefault("sleep", sleeps.append)
    kw.setdefault("rng", lambda: 0.5)
    kw.setdefault("stats", GatewayStats())
    kw.setdefault("clock", lambda: 0.0)
    return GatewayTransport(inner, **kw), sleeps


def test_empty_then_ok_retries_once(monkeypatch):
    inner = Scripted([ok_resp(msg(content=[], output_tokens=0)), ok_resp(msg())])
    t, sleeps = make(inner)
    resp = t.handle_request(request())
    assert resp.json()["content"][0]["text"] == "hello"
    assert len(inner.calls) == 2
    assert sleeps == [0.4]
    assert t._stats.empty_retries == 1
    assert t._stats.empty_final == 0


def test_empty_three_times_returns_empty_after_cap():
    inner = Scripted([ok_resp(EMPTY()), ok_resp(EMPTY()), ok_resp(EMPTY())])
    stats = GatewayStats()
    t, sleeps = make(inner, max_empty_retries=2, stats=stats)
    resp = t.handle_request(request())
    assert resp.json()["content"] == []
    assert len(inner.calls) == 3
    assert len(sleeps) == 2
    assert stats.empty_final == 1
    assert stats.empty_retries == 2


def test_backoff_grows_exponentially_with_jitter_factor():
    inner = Scripted([ok_resp(EMPTY())])
    t, sleeps = make(inner, max_empty_retries=2, rng=lambda: 0.0)
    t.handle_request(request())
    assert sleeps == pytest.approx([0.4 * 0.75, 0.8 * 0.75])


def test_max_empty_retries_zero_no_retry_but_counts_final():
    inner = Scripted([ok_resp(EMPTY())])
    stats = GatewayStats()
    t, sleeps = make(inner, max_empty_retries=0, stats=stats)
    resp = t.handle_request(request())
    assert resp.json()["content"] == []
    assert len(inner.calls) == 1
    assert sleeps == []
    assert stats.empty_retries == 0
    assert stats.empty_final == 1


def test_tool_use_only_reply_not_retried():
    block = {"type": "tool_use", "id": "tu_1", "name": "x", "input": {}}
    inner = Scripted([ok_resp(msg(content=[block], output_tokens=0))])
    t, sleeps = make(inner)
    t.handle_request(request())
    assert len(inner.calls) == 1
    assert sleeps == []


def test_thinking_only_reply_not_retried():
    block = {"type": "thinking", "thinking": "let me think", "signature": "sig"}
    inner = Scripted([ok_resp(msg(content=[block], output_tokens=0))])
    t, sleeps = make(inner)
    t.handle_request(request())
    assert len(inner.calls) == 1
    assert sleeps == []


def test_blank_text_only_reply_is_empty():
    blank = {"type": "text", "text": "   \n"}
    inner = Scripted([ok_resp(msg(content=[blank], output_tokens=0)), ok_resp(msg())])
    t, sleeps = make(inner)
    resp = t.handle_request(request())
    assert len(inner.calls) == 2
    assert resp.json()["content"][0]["text"] == "hello"
    assert sleeps == [0.4]


@pytest.mark.parametrize("status", [429, 500])
def test_error_status_passthrough_untouched(status):
    payload = {"type": "error", "error": {"type": "overloaded_error", "message": "x"}}
    inner = Scripted([json_resp(payload, status)])
    stats = GatewayStats()
    t, sleeps = make(inner, stats=stats)
    resp = t.handle_request(request())
    assert resp.status_code == status
    assert resp.json() == payload
    assert len(inner.calls) == 1
    assert sleeps == []
    assert stats.empty_retries == 0


def test_streaming_request_passthrough_body_untouched():
    body = {"model": REQ_MODEL, "max_tokens": 16, "stream": True, "messages": []}
    inner = Scripted([ok_resp(EMPTY())])
    t, sleeps = make(inner)
    req = request(body)
    original = req.read()
    t.handle_request(req)
    assert len(inner.calls) == 1
    assert sleeps == []
    assert inner.calls[0].read() == original
    assert json.loads(original)["stream"] is True


def test_non_messages_path_passthrough():
    inner = Scripted([ok_resp(EMPTY())])
    t, sleeps = make(inner)
    t.handle_request(request(path="/v1/complete"))
    assert len(inner.calls) == 1
    assert sleeps == []


def test_inspection_exception_returns_original_response_consumable():
    raw = b"{not valid json"
    inner = Scripted([lambda: httpx.Response(200, content=raw, headers={"content-type": "application/json"})])
    t, sleeps = make(inner)
    resp = t.handle_request(request())
    assert len(inner.calls) == 1
    assert sleeps == []
    assert resp.status_code == 200
    assert resp.read() == raw
    assert resp.content == raw


def test_mismatch_logged_once_per_pair_per_hour(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    inner = Scripted([ok_resp(msg(model="claude-opus-5"))])
    stats = GatewayStats()
    t, _ = make(inner, stats=stats)
    for _ in range(5):
        t.handle_request(request())
    lines = [r.getMessage() for r in caplog.records if "served model differs" in r.getMessage()]
    assert lines == ["gateway served model differs: requested=claude-opus-5-5 served=claude-opus-5"]
    assert stats.mismatches == 5

    caplog.clear()
    inner2 = Scripted([ok_resp(msg(model="claude-sonnet-4-6"))])
    t2, _ = make(inner2, stats=stats)
    t2.handle_request(request())
    # opus -> sonnet is a FAMILY change: WARNING (not the alias INFO line).
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warns == ["gateway served family changed: requested=claude-opus-5-5 served=claude-sonnet-4-6"]
    assert not [r for r in caplog.records if "served model differs" in r.getMessage()]


def test_stats_percentiles_and_status_histogram():
    stats = GatewayStats()
    for v in range(1, 101):
        stats.record_call("m", v, 200)
    stats.record_call("m", 999, 429)
    snap = stats.snapshot()
    assert snap["calls"] == 101
    assert snap["status"] == {200: 100, 429: 1}
    assert snap["latency_ms"]["m"]["n"] == 101
    assert isinstance(snap["latency_ms"]["m"]["p50"], int)
    stats2 = GatewayStats()
    for v in range(1, 101):
        stats2.record_call("m", v, 200)
    snap2 = stats2.snapshot()
    assert snap2["latency_ms"]["m"]["p50"] == 50
    assert snap2["latency_ms"]["m"]["p95"] == 95


def test_maybe_log_emits_at_most_once_per_hour(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    stats = GatewayStats()
    stats.record_call("m", 10, 200)
    assert stats.maybe_log(0.0) is True
    assert stats.maybe_log(10.0) is False
    assert stats.maybe_log(3599.0) is False
    assert stats.maybe_log(3600.0) is True
    lines = [r for r in caplog.records if r.getMessage().startswith("gateway stats:")]
    assert len(lines) == 2
    assert "calls=1" in lines[0].getMessage()
    assert "m=10/10" in lines[0].getMessage()


def test_real_sdk_round_trip_empty_then_message():
    inner = Scripted([ok_resp(EMPTY()), ok_resp(msg(content=[{"type": "text", "text": "hi there"}]))])
    transport = GatewayTransport(inner, sleep=lambda s: None, rng=lambda: 0.5, stats=GatewayStats(), clock=lambda: 0.0)
    client = anthropic.Anthropic(
        api_key=FAKE_KEY,
        base_url="http://gw.invalid",
        http_client=anthropic.DefaultHttpxClient(transport=transport),
    )
    out = client.messages.create(
        model=REQ_MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": "hello"}],
    )
    assert out.content[0].text == "hi there"
    assert len(inner.calls) == 2


@pytest.mark.parametrize("stop", ["max_tokens", "refusal", "pause_turn"])
def test_empty_content_with_non_end_turn_stop_not_retried(stop):
    inner = Scripted([ok_resp(msg(content=[], output_tokens=0, stop=stop))])
    t, sleeps = make(inner)
    resp = t.handle_request(request())
    assert resp.json()["stop_reason"] == stop
    assert len(inner.calls) == 1
    assert sleeps == []


def test_blank_text_with_positive_output_tokens_not_retried():
    blank = {"type": "text", "text": "  "}
    inner = Scripted([ok_resp(msg(content=[blank], output_tokens=3))])
    t, sleeps = make(inner)
    t.handle_request(request())
    assert len(inner.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "error", "error": {"type": "api_error", "message": "x"}},
        {},
    ],
)
def test_non_message_200_body_not_retried(payload):
    inner = Scripted([ok_resp(payload)])
    t, sleeps = make(inner)
    resp = t.handle_request(request())
    assert resp.json() == payload
    assert len(inner.calls) == 1
    assert sleeps == []


def test_get_request_to_messages_path_passthrough_not_inspected():
    inner = Scripted([ok_resp(EMPTY())])
    t, sleeps = make(inner)
    resp = t.handle_request(request(method="GET"))
    assert len(inner.calls) == 1
    assert sleeps == []
    assert resp.json()["content"] == []


def test_one_shot_generator_body_passthrough_never_retried():
    payload = json.dumps({"model": REQ_MODEL, "max_tokens": 16, "messages": []}).encode()

    def gen():
        yield payload

    req = httpx.Request("POST", "http://gw.invalid/v1/messages", content=gen())
    assert not isinstance(req.stream, httpx.ByteStream)
    inner = Scripted([ok_resp(EMPTY())])
    t, sleeps = make(inner)
    resp = t.handle_request(req)
    assert len(inner.calls) == 1
    assert sleeps == []
    assert resp.json()["content"] == []


def test_close_delegates_to_inner_transport():
    class Closable(Scripted):
        closed = False

        def close(self):
            self.closed = True

    inner = Closable([ok_resp(msg())])
    t, _ = make(inner)
    t.close()
    assert inner.closed is True


def test_unknown_block_type_is_content_not_retried():
    block = {"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {}}
    inner = Scripted([ok_resp(msg(content=[block], output_tokens=0))])
    t, sleeps = make(inner)
    t.handle_request(request())
    assert len(inner.calls) == 1
    assert sleeps == []


@pytest.fixture
def reset_shared(monkeypatch):
    monkeypatch.setattr(gw, "_shared_client", None)
    monkeypatch.delenv("CLIO_ANTHROPIC_GATEWAY_EMPTY_RETRIES", raising=False)
    yield
    monkeypatch.setattr(gw, "_shared_client", None)


def test_shared_client_is_singleton(reset_shared):
    a = gw.get_shared_http_client()
    b = gw.get_shared_http_client()
    assert a is b
    assert isinstance(a, anthropic.DefaultHttpxClient)
    assert isinstance(a._transport, GatewayTransport)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 2),
        ("", 2),
        ("abc", 2),
        ("0", 0),
        ("1", 1),
        ("3", 3),
        ("9", 3),
        ("-1", 0),
        (" 2 ", 2),
    ],
)
def test_shared_client_empty_retry_env_clamp(reset_shared, monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv("CLIO_ANTHROPIC_GATEWAY_EMPTY_RETRIES", raw)
    client = gw.get_shared_http_client()
    assert client._transport._max_empty_retries == expected
    assert client._transport._retry_empty is True


# ---------------------------------------------------------------------------
# R-09: non-200 causes, per-host counters, alias vs family split, stats tick
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import re  # noqa: E402


def err_resp(status, body, headers=None):
    return lambda: httpx.Response(status, json=body, headers=headers or {})


SECRET_BODY = {"type": "error", "error": {"type": "invalid_request_error", "message": "SECRET-PROMPT-TEXT"}}


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == LOGGER and r.levelno == logging.WARNING and "http_status=" in r.getMessage()]


def test_non_200_warning_has_only_safe_fields_and_never_message(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    inner = Scripted([err_resp(400, SECRET_BODY, {"request-id": "req_abc123"})])
    t, _ = make(inner)
    resp = t.handle_request(request())
    assert resp.status_code == 400
    assert json.loads(resp.content)["error"]["message"] == "SECRET-PROMPT-TEXT"  # caller still sees body
    lines = _warnings(caplog)
    assert len(lines) == 1
    assert lines[0] == ("gateway http_status=400 endpoint=/v1/messages model=claude-opus-5-5 "
                        "request_id=req_abc123 error_type=invalid_request_error")
    assert "SECRET-PROMPT-TEXT" not in "\n".join(r.getMessage() for r in caplog.records)


def test_hostile_request_id_cannot_inject_log_lines(caplog):
    # A header value with spaces/newlines must be reduced to identifier characters (log-injection guard).
    caplog.set_level(logging.INFO, logger=LOGGER)
    inner = Scripted([err_resp(400, SECRET_BODY, {"request-id": "req 1\nINJECTED line sk-abc"})])
    t, _ = make(inner)
    t.handle_request(request())
    lines = _warnings(caplog)
    assert len(lines) == 1
    assert "\n" not in lines[0] and " line" not in lines[0].split("request_id=")[1]
    assert "request_id=req1INJECTEDlinesk-abc" in lines[0]


def test_error_warning_deduped_per_status_and_type(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    clk = [0.0]
    stats = GatewayStats()
    t, _ = make(Scripted([err_resp(400, SECRET_BODY)]), stats=stats, clock=lambda: clk[0])
    for _ in range(5):
        t.handle_request(request())
    assert len(_warnings(caplog)) == 1
    # Different error type under the same status is a new key.
    t2, _ = make(Scripted([err_resp(400, {"error": {"type": "not_found_error"}})]), stats=stats, clock=lambda: clk[0])
    t2.handle_request(request())
    # Same type under a different status is a new key too.
    t3, _ = make(Scripted([err_resp(429, SECRET_BODY)]), stats=stats, clock=lambda: clk[0])
    t3.handle_request(request())
    assert len(_warnings(caplog)) == 3


def test_error_warning_reappears_after_ten_minutes(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    clk = [0.0]
    t, _ = make(Scripted([err_resp(400, SECRET_BODY)]), clock=lambda: clk[0])
    t.handle_request(request())
    clk[0] = 599.0
    t.handle_request(request())
    assert len(_warnings(caplog)) == 1
    clk[0] = 600.0
    t.handle_request(request())
    assert len(_warnings(caplog)) == 2


def test_non_messages_path_error_is_also_logged(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    t, _ = make(Scripted([err_resp(500, {"error": {"type": "api_error"}})]))
    t.handle_request(request(path="/v1/models", method="GET"))
    lines = _warnings(caplog)
    assert len(lines) == 1 and "error_type=api_error" in lines[0] and "model=-" in lines[0]


class _BoomRecordHost(GatewayStats):
    def record_host(self, host):
        raise RuntimeError("boom-host")


class _BoomDedupe(GatewayStats):
    def should_log_error(self, status, error_type, now):
        raise RuntimeError("boom-dedupe")


@pytest.mark.parametrize("stats_cls", [_BoomRecordHost, _BoomDedupe])
def test_inspection_exceptions_never_escape_guard(stats_cls):
    t, _ = make(Scripted([err_resp(400, SECRET_BODY)]), stats=stats_cls())
    resp = t.handle_request(request())
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_hosts_counted_and_in_stats_line(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    stats = GatewayStats()
    t, _ = make(Scripted([ok_resp(msg())]), stats=stats)
    t.handle_request(request())
    t.handle_request(request())
    assert stats.snapshot()["hosts"] == {"gw.invalid": 2}
    stats.emit_stats_tick(0.0)
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gateway stats:")][-1]
    assert line.endswith("family_mismatch=0 alias=0 hosts={gw.invalid:2}")


def test_alias_same_family_is_not_a_family_warning(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    stats = GatewayStats()
    t, _ = make(Scripted([ok_resp(msg(model="claude-opus-5"))]), stats=stats)
    for _ in range(3):
        t.handle_request(request())
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == LOGGER]
    snap = stats.snapshot()
    assert snap["alias_same_family"] == 3 and snap["family_mismatch"] == 0 and snap["mismatches"] == 3


def test_family_change_is_warning_once_per_pair(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    stats = GatewayStats()
    t, _ = make(Scripted([ok_resp(msg(model="claude-sonnet-4-6"))]), stats=stats)
    for _ in range(3):
        t.handle_request(request())
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name == LOGGER]
    assert warns == ["gateway served family changed: requested=claude-opus-5-5 served=claude-sonnet-4-6"]
    snap = stats.snapshot()
    assert snap["family_mismatch"] == 3 and snap["alias_same_family"] == 0


def test_stats_tick_emits_even_with_zero_calls(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    stats = GatewayStats()
    stats.emit_stats_tick(0.0)
    stats.emit_stats_tick(10.0)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gateway stats:")]
    assert len(lines) == 2
    assert lines[0].startswith("gateway stats: calls=0 empty_retries=0 empty_final=0 mismatches=0 ")
    assert lines[0].endswith("status={-} family_mismatch=0 alias=0 hosts={}")


def test_stats_line_prefix_contract_matches_morning_check(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    stats = GatewayStats()
    stats.record_call("claude-opus-5-5", 120, 200)
    stats.emit_stats_tick(0.0)
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gateway stats:")][-1]
    m = re.search(r"gateway stats: calls=(\d+) empty_retries=(\d+) empty_final=(\d+)", line)
    assert m and m.groups() == ("1", "0", "0")
    assert re.search(r"status=\{200:1\} family_mismatch=0 alias=0 hosts=\{\}$", line)


def test_gateway_stats_tick_job_respects_monitor_flag(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    monkeypatch.setenv("CLIO_GATEWAY_MONITOR_ENABLED", "0")
    before = len([r for r in caplog.records if r.getMessage().startswith("gateway stats:")])
    asyncio.run(gw.gateway_stats_tick_job(None))
    assert len([r for r in caplog.records if r.getMessage().startswith("gateway stats:")]) == before
    monkeypatch.delenv("CLIO_GATEWAY_MONITOR_ENABLED", raising=False)
    asyncio.run(gw.gateway_stats_tick_job(None))
    assert len([r for r in caplog.records if r.getMessage().startswith("gateway stats:")]) == before + 1
