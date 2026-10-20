"""S214: forced-summary turn, gateway retry, thinking-block preservation (agent/core.py).

All API traffic is a scripted fake client patched in via agent.core._get_client.
Config is swapped with dataclasses.replace (no env mutation). No network.
"""

import asyncio
import dataclasses
import os
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.core as core  # noqa: E402

INSTR = core._FORCED_SUMMARY_INSTRUCTION
THINK_KW = {"thinking": {"type": "adaptive"}}


def _text(t):
    return NS(type="text", text=t)


def _tool_use():
    return NS(type="tool_use", id="toolu_9", name="get_weather", input={"city": "Paris"})


def _thinking(t="pondering"):
    return NS(type="thinking", thinking=t, signature="sig")


def _reply(*blocks):
    return NS(content=list(blocks), stop_reason="end_turn", model="served-model")


class _Messages:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.scripted.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _Client:
    def __init__(self, scripted):
        self.messages = _Messages(scripted)


def _set_gateway(monkeypatch, flag):
    monkeypatch.setattr(core, "config", dataclasses.replace(core.config, anthropic_gateway=flag))


def _install(monkeypatch, scripted):
    client = _Client(scripted)
    monkeypatch.setattr(core, "_get_client", lambda: client)
    return client


def _history():
    return [
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": [_text("Checking."), _tool_use()]},
        {"role": "user", "content": core._forced_summary_user_turn([
            {"type": "tool_result", "tool_use_id": "toolu_9", "content": "INTERRUPTED"},
        ])},
    ]


def _run_turn(messages=None):
    msgs = messages if messages is not None else _history()

    async def _go():
        loop = asyncio.get_running_loop()
        return await core._forced_summary_turn(loop, "claude-sonnet-5-5", "SYS", msgs, dict(THINK_KW))

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Instruction placement
# ---------------------------------------------------------------------------

def test_user_turn_has_tool_results_first_and_instruction_last():
    turn = core._forced_summary_user_turn([{"type": "tool_result", "tool_use_id": "t", "content": "x"}])
    assert turn[0]["type"] == "tool_result"
    assert turn[-1] == {"type": "text", "text": INSTR}
    assert len(turn) == 2


def test_first_call_keeps_tool_choice_none_and_tools_and_instruction_last(monkeypatch):
    _set_gateway(monkeypatch, False)
    client = _install(monkeypatch, [_reply(_text("Final answer."))])
    resp = _run_turn()
    assert resp.content[0].text == "Final answer."
    assert len(client.messages.calls) == 1
    kw = client.messages.calls[0]
    assert kw["tool_choice"] == {"type": "none"}
    assert kw["tools"]  # get_all_tools() non-empty
    final_user = kw["messages"][-1]
    assert final_user["role"] == "user"
    assert final_user["content"][-1] == {"type": "text", "text": INSTR}
    assert final_user["content"][0]["type"] == "tool_result"
    assert kw["thinking"] == {"type": "adaptive"}


# ---------------------------------------------------------------------------
# Gateway retry
# ---------------------------------------------------------------------------

def test_gateway_empty_first_reply_retries_without_tools(monkeypatch):
    _set_gateway(monkeypatch, True)
    client = _install(monkeypatch, [_reply(), _reply(_text("Retry text."))])
    resp = _run_turn()
    assert resp.content[0].text == "Retry text."
    assert len(client.messages.calls) == 2
    retry = client.messages.calls[1]
    assert "tools" not in retry
    assert "tool_choice" not in retry
    assert retry["system"] == "SYS"


def test_gateway_tool_use_first_reply_retries(monkeypatch):
    _set_gateway(monkeypatch, True)
    client = _install(monkeypatch, [_reply(_text("partial"), _tool_use()), _reply(_text("Clean summary."))])
    resp = _run_turn()
    assert resp.content[0].text == "Clean summary."
    assert len(client.messages.calls) == 2


def test_gateway_text_first_reply_does_not_retry(monkeypatch):
    _set_gateway(monkeypatch, True)
    client = _install(monkeypatch, [_reply(_text("Good answer."))])
    resp = _run_turn()
    assert resp.content[0].text == "Good answer."
    assert len(client.messages.calls) == 1


def test_direct_mode_empty_first_reply_does_not_retry(monkeypatch):
    _set_gateway(monkeypatch, False)
    first = _reply()
    client = _install(monkeypatch, [first, _reply(_text("should not be used"))])
    resp = _run_turn()
    assert resp is first
    assert len(client.messages.calls) == 1


def test_direct_mode_tool_use_first_reply_does_not_retry(monkeypatch):
    _set_gateway(monkeypatch, False)
    client = _install(monkeypatch, [_reply(_tool_use()), _reply(_text("unused"))])
    resp = _run_turn()
    assert resp.content[0].type == "tool_use"
    assert len(client.messages.calls) == 1


def test_gateway_retry_raising_returns_first_response(monkeypatch):
    _set_gateway(monkeypatch, True)
    first = _reply()
    client = _install(monkeypatch, [first, RuntimeError("boom")])
    resp = _run_turn()
    assert resp is first
    assert len(client.messages.calls) == 2


def test_gateway_retry_empty_returns_first_response(monkeypatch):
    _set_gateway(monkeypatch, True)
    first = _reply(_tool_use())
    second = _reply()
    client = _install(monkeypatch, [first, second])
    resp = _run_turn()
    assert resp is first
    assert len(client.messages.calls) == 2


def test_first_call_exception_propagates_to_caller(monkeypatch):
    _set_gateway(monkeypatch, True)
    _install(monkeypatch, [RuntimeError("first failed")])
    with pytest.raises(RuntimeError):
        _run_turn()


# ---------------------------------------------------------------------------
# Thinking preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "env_value, gateway, expected",
    [
        ("1", True, True),
        ("1", False, True),
        ("0", True, False),
        ("0", False, False),
        ("auto", True, False),
        ("auto", False, True),
        (None, True, False),
        (None, False, True),
    ],
)
def test_preserve_thinking_matrix(monkeypatch, env_value, gateway, expected):
    if env_value is None:
        monkeypatch.delenv("CLIO_PRESERVE_THINKING_BLOCKS", raising=False)
    else:
        monkeypatch.setenv("CLIO_PRESERVE_THINKING_BLOCKS", env_value)
    _set_gateway(monkeypatch, gateway)
    assert core._preserve_thinking() is expected


def test_assistant_content_keeps_thinking_when_preserving(monkeypatch):
    monkeypatch.setenv("CLIO_PRESERVE_THINKING_BLOCKS", "1")
    th = _thinking()
    tu = _tool_use()
    resp = _reply(th, tu)
    out = core._assistant_content_for_loop(resp)
    assert out == [th, tu]
    assert out[0] is th  # passed through as the SDK object, not rebuilt


def test_assistant_content_drops_thinking_when_not_preserving(monkeypatch):
    monkeypatch.setenv("CLIO_PRESERVE_THINKING_BLOCKS", "0")
    tu = _tool_use()
    out = core._assistant_content_for_loop(_reply(_thinking(), tu))
    assert out == [tu]


def test_assistant_content_auto_on_gateway_strips(monkeypatch):
    monkeypatch.delenv("CLIO_PRESERVE_THINKING_BLOCKS", raising=False)
    _set_gateway(monkeypatch, True)
    tu = _tool_use()
    assert core._assistant_content_for_loop(_reply(_thinking(), tu)) == [tu]


def test_extract_text_response_ignores_thinking():
    resp = _reply(_thinking("internal"), _text("visible"))
    assert core.extract_text_response(resp) == "visible"
