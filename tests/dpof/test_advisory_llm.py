"""Tests for agent.dpof.advisory_llm -- AdvisoryLLM (Anthropic primary /
OpenAI fallback advisory client used by F1 reflection and F2 debate).

Plugin-free: async code is driven with asyncio.run() inside ordinary test
functions (see tests/dpof/conftest.py). No network: every provider call goes
through an injected fake anthropic_transport/openai_transport, except the
oversized-response test which wires a real httpx.AsyncClient to
httpx.MockTransport (same technique as tests/dpof/test_decider.py).
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import time
from typing import Optional

import httpx
import pytest

from agent.dpof.advisory_llm import (
    AdvisoryLLM,
    AdvisoryResult,
    _extract_first_json_object,
    _is_output_config_400,
    _redact,
    _shape_matches,
    _try_parse_json,
    _valid_model,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]


_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "string"},
        "y": {"type": "number"},
    },
    "required": ["x", "y"],
    "additionalProperties": False,
}


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _answer(x: str = "hello", y: float = 1.0) -> dict:
    return {"x": x, "y": y}


def _result(text_obj, model: str = "claude-opus-5-5") -> dict:
    text = text_obj if isinstance(text_obj, str) else json.dumps(text_obj)
    return {"text": text, "usage": {"input_tokens": 10, "output_tokens": 5}, "model": model}


def _fixed_transport(payload_or_exc, calls: Optional[list] = None):
    async def transport(payload):
        if calls is not None:
            calls.append(payload)
        if isinstance(payload_or_exc, Exception):
            raise payload_or_exc
        return payload_or_exc
    return transport


def _sequence_transport(items, default=None, calls: Optional[list] = None):
    it = iter(items)

    async def transport(payload):
        if calls is not None:
            calls.append(payload)
        try:
            item = next(it)
        except StopIteration:
            item = default
        if isinstance(item, Exception):
            raise item
        return item
    return transport


class _Fake400(Exception):
    def __init__(self, msg: str):
        self.status_code = 400
        super().__init__(msg)


def _llm(**kwargs) -> AdvisoryLLM:
    kwargs.setdefault("anthropic_key_fn", lambda: "sk-ant-test-key")
    kwargs.setdefault("openai_key_fn", lambda: "sk-oai-test-key")
    return AdvisoryLLM(**kwargs)


async def _complete(
    llm: AdvisoryLLM,
    *,
    primary_model: str = "claude-opus-5-5",
    fallback_model: str = "gpt-6-sol",
    timeout_seconds: float = 5.0,
    budget: Optional[str] = None,
) -> AdvisoryResult:
    return await llm.complete_json(
        purpose="test_purpose", system="you are a test system prompt", user="user prompt text",
        schema=_SCHEMA, schema_name="test_schema",
        primary_model=primary_model, fallback_model=fallback_model,
        timeout_seconds=timeout_seconds, budget=budget,
    )


# ─────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────

def test_happy_path_anthropic():
    calls: list = []
    llm = _llm(anthropic_transport=_fixed_transport(_result(_answer()), calls))
    res = asyncio.run(_complete(llm))

    assert res.ok is True
    assert res.provider == "anthropic"
    assert res.model == "claude-opus-5-5"
    assert res.parsed == _answer()
    assert res.latency_ms is not None and res.latency_ms >= 0
    assert res.error == ""
    assert len(calls) == 1
    assert calls[0]["system"] == "you are a test system prompt"
    assert calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_happy_path_openai_only():
    # Anthropic key empty -> skip straight to OpenAI, still a "happy path".
    calls: list = []
    llm = _llm(anthropic_key_fn=lambda: "", openai_transport=_fixed_transport(_result(_answer(), model="gpt-6-sol"), calls))
    res = asyncio.run(_complete(llm))
    assert res.ok is True
    assert res.provider == "openai"
    assert len(calls) == 1
    assert calls[0]["text"]["format"]["strict"] is True
    assert calls[0]["input"][0]["role"] == "developer"


# ─────────────────────────────────────────────────────────────────────────
# output_config 400 downgrade
# ─────────────────────────────────────────────────────────────────────────

def test_output_config_400_downgrades_and_parses_prose():
    calls: list = []
    prose = "Here is my answer:\n" + json.dumps(_answer()) + "\nThat's it."
    transport = _sequence_transport(
        [_Fake400("400 Bad Request: output_config.format is not supported"), _result(prose)],
        default=_result(_answer()),
        calls=calls,
    )
    llm = _llm(anthropic_transport=transport)

    res = asyncio.run(_complete(llm))
    assert res.ok is True
    assert res.parsed == _answer()
    assert llm._anthropic_format_downgraded is True
    assert "format" in calls[0]["output_config"]
    assert "format" not in calls[1]["output_config"]

    # The downgrade persists for the rest of the instance's life.
    res2 = asyncio.run(_complete(llm))
    assert res2.ok is True
    assert "format" not in calls[2]["output_config"]


# ─────────────────────────────────────────────────────────────────────────
# Fallback behavior
# ─────────────────────────────────────────────────────────────────────────

def test_anthropic_error_falls_back_to_openai():
    calls_a, calls_o = [], []
    llm = _llm(
        anthropic_transport=_fixed_transport(RuntimeError("anthropic down"), calls_a),
        openai_transport=_fixed_transport(_result(_answer(), model="gpt-6-sol"), calls_o),
    )
    res = asyncio.run(_complete(llm))

    assert res.ok is True
    assert res.provider == "openai"
    assert res.model == "gpt-6-sol"
    assert len(calls_a) == 1
    assert len(calls_o) == 1
    assert llm._anthropic_status.consecutive_failures == 1


def test_schema_shape_mismatch_falls_back_to_openai():
    calls_o: list = []
    llm = _llm(
        anthropic_transport=_fixed_transport(_result('{"not": "the right shape"}')),
        openai_transport=_fixed_transport(_result(_answer(), model="gpt-6-sol"), calls_o),
    )
    res = asyncio.run(_complete(llm))

    assert res.ok is True
    assert res.provider == "openai"
    assert len(calls_o) == 1
    assert llm._anthropic_status.last_error == "invalid_json"


def test_both_providers_fail_returns_ok_false():
    llm = _llm(
        anthropic_transport=_fixed_transport(RuntimeError("anthropic down")),
        openai_transport=_fixed_transport(RuntimeError("openai down")),
    )
    res = asyncio.run(_complete(llm))

    assert res.ok is False
    assert res.parsed is None
    assert "anthropic:" in res.error
    assert "openai:" in res.error


def test_fallback_model_empty_disables_fallback():
    calls_o: list = []
    llm = _llm(
        anthropic_transport=_fixed_transport(RuntimeError("anthropic down")),
        openai_transport=_fixed_transport(_result(_answer()), calls_o),
    )
    res = asyncio.run(_complete(llm, fallback_model=""))

    assert res.ok is False
    assert "fallback_disabled" in res.error
    assert calls_o == []


def test_empty_anthropic_key_skips_to_openai():
    calls_a, calls_o = [], []
    llm = _llm(
        anthropic_key_fn=lambda: "",
        anthropic_transport=_fixed_transport(RuntimeError("must never be called"), calls_a),
        openai_transport=_fixed_transport(_result(_answer(), model="gpt-6-sol"), calls_o),
    )
    res = asyncio.run(_complete(llm))

    assert res.ok is True
    assert res.provider == "openai"
    assert calls_a == []
    assert len(calls_o) == 1


def test_empty_openai_key_disables_fallback():
    llm = _llm(
        openai_key_fn=lambda: "",
        anthropic_transport=_fixed_transport(RuntimeError("anthropic down")),
    )
    res = asyncio.run(_complete(llm))

    assert res.ok is False
    assert "fallback_disabled" in res.error


def test_invalid_primary_model_type_skips_to_openai():
    calls_o: list = []
    llm = _llm(openai_transport=_fixed_transport(_result(_answer(), model="gpt-6-sol"), calls_o))
    res = asyncio.run(llm.complete_json(
        purpose="t", system="s", user="u", schema=_SCHEMA, schema_name="x",
        primary_model=123, fallback_model="gpt-6-sol", timeout_seconds=5.0,
    ))
    assert res.ok is True
    assert res.provider == "openai"
    assert len(calls_o) == 1


# ─────────────────────────────────────────────────────────────────────────
# Circuit breaker
# ─────────────────────────────────────────────────────────────────────────

def test_circuit_opens_after_n_failures_and_half_opens_after_cooldown():
    clock = FakeClock()
    calls: list = []
    transport = _sequence_transport(
        [RuntimeError("boom1"), RuntimeError("boom2"), _result(_answer())], calls=calls,
    )
    llm = _llm(anthropic_transport=transport, clock=clock, circuit_failures=2, circuit_cooldown_seconds=100)

    res1 = asyncio.run(_complete(llm, fallback_model=""))
    assert res1.ok is False
    assert llm._anthropic_status.consecutive_failures == 1
    assert llm._anthropic_status.circuit_open_until is None

    res2 = asyncio.run(_complete(llm, fallback_model=""))
    assert res2.ok is False
    assert llm._anthropic_status.consecutive_failures == 2
    assert llm._anthropic_status.circuit_open_until is not None

    calls_before = len(calls)
    res3 = asyncio.run(_complete(llm, fallback_model=""))
    assert res3.ok is False
    assert "circuit_open" in res3.error
    assert len(calls) == calls_before  # circuit open -> transport never invoked

    clock.advance(101)
    res4 = asyncio.run(_complete(llm, fallback_model=""))
    assert res4.ok is True
    assert llm._anthropic_status.consecutive_failures == 0
    assert llm._anthropic_status.circuit_open_until is None


# ─────────────────────────────────────────────────────────────────────────
# Budget
# ─────────────────────────────────────────────────────────────────────────

def test_budget_exhausted_skips_all_providers_no_call():
    calls_a, calls_o = [], []

    async def exhausted_budget_fn(name: str) -> None:
        raise RuntimeError(f"{name} budget exhausted")

    llm = _llm(
        anthropic_transport=_fixed_transport(_result(_answer()), calls_a),
        openai_transport=_fixed_transport(_result(_answer()), calls_o),
        budget_fn=exhausted_budget_fn,
    )
    res = asyncio.run(_complete(llm, budget="dpof_debate"))

    assert res.ok is False
    assert res.error == "budget"
    assert calls_a == []
    assert calls_o == []


def test_no_budget_param_never_calls_budget_fn():
    budget_calls: list = []

    async def tracking_budget_fn(name: str) -> None:
        budget_calls.append(name)

    llm = _llm(anthropic_transport=_fixed_transport(_result(_answer())), budget_fn=tracking_budget_fn)
    res = asyncio.run(_complete(llm, budget=None))

    assert res.ok is True
    assert budget_calls == []


# ─────────────────────────────────────────────────────────────────────────
# Deadlines
# ─────────────────────────────────────────────────────────────────────────

def test_primary_timeout_falls_back_and_fallback_gets_its_own_budget():
    async def slow_anthropic(payload):
        await asyncio.sleep(5.0)
        return _result(_answer())

    calls_o: list = []
    llm = _llm(
        anthropic_transport=slow_anthropic,
        openai_transport=_fixed_transport(_result(_answer(), model="gpt-6-sol"), calls_o),
    )

    t0 = time.monotonic()
    res = asyncio.run(_complete(llm, timeout_seconds=0.2))
    elapsed = time.monotonic() - t0

    assert res.ok is True
    assert res.provider == "openai"
    assert len(calls_o) == 1
    assert llm._anthropic_status.last_error == "timeout"
    # The slow primary's 5s sleep must have been CANCELLED at ~0.2s, not
    # awaited to completion -- proves asyncio.wait_for actually bounds it
    # and the fallback still gets its own (fast) attempt afterward.
    assert elapsed < 1.0


# ─────────────────────────────────────────────────────────────────────────
# Oversized OpenAI response
# ─────────────────────────────────────────────────────────────────────────

def test_openai_response_size_cap_rejects_oversized():
    oversized_body = json.dumps({"output_text": "x" * (2 * 1024 * 1024)}).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized_body, headers={"content-length": str(len(oversized_body))})

    llm = _llm()
    llm._openai_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        asyncio.run(llm._openai_http_call({"model": "gpt-6-sol"}, 5.0, "sk-oai-test-key"))


# ─────────────────────────────────────────────────────────────────────────
# Secrets never logged
# ─────────────────────────────────────────────────────────────────────────

def test_openai_failure_never_leaks_api_key(caplog):
    async def bad_transport(payload):
        raise RuntimeError("auth failed for Bearer sk-super-secret-value token")

    llm = _llm(
        anthropic_transport=_fixed_transport(RuntimeError("anthropic down")),
        openai_key_fn=lambda: "sk-super-secret-value",
        openai_transport=bad_transport,
    )
    res = asyncio.run(_complete(llm))

    assert res.ok is False
    assert "sk-super-secret-value" not in res.error
    assert "sk-super-secret-value" not in llm._openai_status.last_error
    assert "sk-super-secret-value" not in caplog.text


def test_redact_strips_bearer_and_named_secrets():
    msg = "unauthorized Bearer sk-verysecret123 and also sk-verysecret123 again"
    out = _redact(msg, "sk-verysecret123")
    assert "sk-verysecret123" not in out
    assert "Bearer ***" in out


# ─────────────────────────────────────────────────────────────────────────
# status() -- no secrets
# ─────────────────────────────────────────────────────────────────────────

def test_status_reports_calls_and_no_secrets():
    llm = _llm(anthropic_transport=_fixed_transport(_result(_answer())))
    asyncio.run(_complete(llm))

    status = llm.status()
    assert status["providers"]["anthropic"]["calls_today"] == 1
    assert status["providers"]["anthropic"]["failures_today"] == 0
    assert status["providers"]["anthropic"]["consecutive_failures"] == 0
    dumped = json.dumps(status)
    assert "sk-ant-test-key" not in dumped
    assert "sk-oai-test-key" not in dumped


def test_status_counts_failures():
    llm = _llm(anthropic_transport=_fixed_transport(RuntimeError("down")), openai_key_fn=lambda: "")
    asyncio.run(_complete(llm))
    status = llm.status()
    assert status["providers"]["anthropic"]["calls_today"] == 1
    assert status["providers"]["anthropic"]["failures_today"] == 1


# ─────────────────────────────────────────────────────────────────────────
# close()
# ─────────────────────────────────────────────────────────────────────────

def test_close_is_safe_with_nothing_constructed():
    llm = _llm()
    asyncio.run(llm.close())  # must not raise


def test_close_closes_openai_client():
    llm = _llm(anthropic_transport=_fixed_transport(RuntimeError("primary down")))
    client = llm._get_openai_client()
    asyncio.run(llm.close())
    assert client.is_closed
    assert llm._openai_client is None


# ─────────────────────────────────────────────────────────────────────────
# Never raises
# ─────────────────────────────────────────────────────────────────────────

def test_never_raises_on_internal_bug():
    llm = _llm(anthropic_transport=_fixed_transport(_result(_answer())))
    res = asyncio.run(llm.complete_json(
        purpose="t", system="s", user="u", schema="not-a-dict", schema_name="x",
        primary_model="claude-opus-5-5", fallback_model="", timeout_seconds=5.0,
    ))
    assert res.ok is False
    assert "internal_error" in res.error


def test_cancelled_error_propagates():
    async def cancels(payload):
        raise asyncio.CancelledError()

    llm = _llm(anthropic_transport=cancels)

    async def runner():
        with pytest.raises(asyncio.CancelledError):
            await _complete(llm, fallback_model="")

    asyncio.run(runner())


# ─────────────────────────────────────────────────────────────────────────
# Pure helper unit tests
# ─────────────────────────────────────────────────────────────────────────

def test_extract_first_json_object_handles_nested_and_quoted_braces():
    text = 'prose {"a": 1, "b": {"c": "}nested} literal"}, "d": [1,2]} trailing'
    obj = _extract_first_json_object(text)
    assert obj == {"a": 1, "b": {"c": "}nested} literal"}, "d": [1, 2]}


def test_extract_first_json_object_none_when_no_brace():
    assert _extract_first_json_object("no json here") is None


def test_try_parse_json_direct_and_fallback():
    assert _try_parse_json('{"x": 1}') == {"x": 1}
    assert _try_parse_json("prose then {\"x\": 2} more prose") == {"x": 2}
    assert _try_parse_json("") is None
    assert _try_parse_json("not json at all") is None


def test_shape_matches_requires_fields_and_types():
    good = _answer()
    assert _shape_matches(good, _SCHEMA) is True
    missing = dict(good)
    del missing["y"]
    assert _shape_matches(missing, _SCHEMA) is False
    wrong_type = dict(good)
    wrong_type["y"] = "not a number"
    assert _shape_matches(wrong_type, _SCHEMA) is False
    assert _shape_matches("not a dict", _SCHEMA) is False


def test_is_output_config_400_detects_and_rejects_others():
    assert _is_output_config_400(_Fake400("400: output_config.format unsupported")) is True

    class _NotIt(Exception):
        status_code = 400

    assert _is_output_config_400(_NotIt("unrelated 400 error")) is False
    assert _is_output_config_400(RuntimeError("some other failure")) is False


def test_valid_model_helper():
    assert _valid_model("claude-opus-5-5") is True
    assert _valid_model("") is False
    assert _valid_model("   ") is False
    assert _valid_model(None) is False
    assert _valid_model(123) is False


# ─────────────────────────────────────────────────────────────────────────
# No forbidden imports (grep test) -- advisory_llm.py must not touch the
# live decider or the engine/orders/broker execution path.
# ─────────────────────────────────────────────────────────────────────────

def test_no_forbidden_imports():
    src = (_ROOT / "agent" / "dpof" / "advisory_llm.py").read_text()
    for forbidden in ("agent.dpof.engine", "agent.dpof.orders", "agent.dpof.broker", "agent.dpof.decider"):
        assert forbidden not in src, f"advisory_llm.py must not import {forbidden}"


def test_effort_is_plumbed_to_both_providers_and_validated():
    """complete_json(effort=...) reaches the Anthropic output_config and the
    OpenAI reasoning block; an unknown value falls back to "low"."""
    import asyncio

    from agent.dpof import advisory_llm as al

    seen = []

    async def _anthropic(payload):
        seen.append(("anthropic", payload["output_config"]["effort"]))
        raise RuntimeError("force fallback")

    async def _openai(payload):
        seen.append(("openai", payload["reasoning"]["effort"]))
        return {"output_text": '{"x": 1}', "model": "gpt-6-sol"}

    llm = al.AdvisoryLLM(anthropic_key_fn=lambda: "k", openai_key_fn=lambda: "k",
                         anthropic_transport=_anthropic, openai_transport=_openai)
    for effort, expected in (("medium", "medium"), ("bogus", "low")):
        seen.clear()
        asyncio.run(llm.complete_json(purpose="t", system="s", user="u", schema={"type": "object", "required": ["x"]},
                                      schema_name="t", primary_model="claude-opus-5-5", fallback_model="gpt-6-sol",
                                      timeout_seconds=5, effort=effort))
        assert seen[0] == ("anthropic", expected)
        assert seen[1][0] == "openai" and seen[1][1] == expected
