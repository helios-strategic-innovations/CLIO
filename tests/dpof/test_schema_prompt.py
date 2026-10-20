"""OP-05: JSON schema embedded in the prompt when structured outputs are unavailable.

Direct Anthropic payloads (use_format=True) must stay byte-identical; gateway
or downgraded payloads (use_format=False) must carry the schema text.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from typing import Optional

import pytest

from agent.config import config as clio_config
from agent.dpof.advisory_llm import AdvisoryLLM
from agent.dpof.debate import _DEBATE_SCHEMA
from agent.dpof.decider import ENTRY_SCHEMA, DPOFDecider, EntryCandidate
from agent.dpof.prompts import CHARTER
from agent.dpof.schema_prompt import schema_instruction


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

class _Fake400(Exception):
    def __init__(self, msg: str):
        self.status_code = 400
        super().__init__(msg)


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


def _result(text_obj, model: str = "claude-opus-5-5") -> dict:
    text = text_obj if isinstance(text_obj, str) else json.dumps(text_obj)
    return {"text": text, "usage": {"input_tokens": 10, "output_tokens": 5}, "model": model}


def _entry_answer(candidate_key: str = "C1") -> dict:
    return {
        "action": "enter", "candidate_key": candidate_key, "conviction": 0.7, "qty": 1,
        "thesis": "thesis text", "invalidation": "invalidation text", "recheck_minutes": 5,
    }


def _debate_answer() -> dict:
    return {"argument": "the case", "key_risk": "the risk", "strength": 0.5}


def _entry_ctx() -> dict:
    return {
        "now_et": "2026-09-24T12:00:00-04:00",
        "minutes_to_flatten": 230,
        "mode": "paper",
        "underlyings": {"SPY": {"composite": 0.42, "direction": "bullish", "strength": "strong"}},
        "market_regime": "normal",
        "open_positions": [],
        "day": {"realized_today": 0.0, "entries_today": 0, "open_premium": 0.0},
        "recent_closed": [],
        "econ_events": [],
        "jev": None,
        "headlines": [],
    }


def _candidates() -> list:
    return [
        EntryCandidate(
            key="C1", underlying="SPY", direction="bullish", structure="single",
            legs=[{"symbol": "SPY261002C00580000", "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}],
            expiry="2026-09-25", sessions_to_expiry=1, strike_desc="580C",
            bid=1.18, ask=1.28, mid=1.23, spread_pct=0.08,
            delta=0.5, theta=-0.12, iv=0.18, open_interest=1200, unit_premium=123.0,
            max_affordable_qty=5,
        )
    ]


@pytest.fixture
def gateway_flag(monkeypatch):
    """Set agent.config.config.anthropic_gateway for one test; monkeypatch restores it."""
    def _set(value: bool) -> None:
        monkeypatch.setattr(clio_config, "anthropic_gateway", value)
    return _set


def _llm(**kwargs) -> AdvisoryLLM:
    kwargs.setdefault("anthropic_key_fn", lambda: "sk-ant-test-key")
    kwargs.setdefault("openai_key_fn", lambda: "sk-oai-test-key")
    return AdvisoryLLM(**kwargs)


# ─────────────────────────────────────────────────────────────────────────
# schema_instruction: content, determinism, no mutation
# ─────────────────────────────────────────────────────────────────────────

def test_instruction_contains_every_required_key_of_entry_and_debate():
    text = schema_instruction(ENTRY_SCHEMA)
    for key in ENTRY_SCHEMA["required"]:
        assert f'"{key}"' in text
    text_d = schema_instruction(_DEBATE_SCHEMA)
    for key in _DEBATE_SCHEMA["required"]:
        assert f'"{key}"' in text_d


def test_instruction_exact_text_is_the_documented_contract():
    expected_prefix = (
        "Answer with exactly one JSON object that matches this JSON schema "
        "(include every required key; no prose, no markdown, no code fences):\n"
    )
    assert schema_instruction(ENTRY_SCHEMA) == expected_prefix + json.dumps(
        ENTRY_SCHEMA, separators=(",", ":"), sort_keys=True,
    )


def test_instruction_is_deterministic_and_key_order_independent():
    reordered = dict(reversed(list(ENTRY_SCHEMA.items())))
    assert schema_instruction(ENTRY_SCHEMA) == schema_instruction(ENTRY_SCHEMA)
    assert schema_instruction(ENTRY_SCHEMA) == schema_instruction(reordered)


def test_instruction_does_not_mutate_its_argument():
    schema = copy.deepcopy(ENTRY_SCHEMA)
    before = copy.deepcopy(schema)
    schema_instruction(schema)
    assert schema == before


def test_instruction_tolerates_arbitrary_dict():
    assert "Answer with exactly one JSON object" in schema_instruction({})
    assert "Answer with exactly one JSON object" in schema_instruction({"weird": object()})


# ─────────────────────────────────────────────────────────────────────────
# Decider payloads
# ─────────────────────────────────────────────────────────────────────────

def test_decider_direct_payload_golden_is_unchanged(cfg):
    decider = DPOFDecider(cfg)
    payload = decider._build_anthropic_payload("USER TEXT", ENTRY_SCHEMA, use_format=True)
    assert payload == {
        "model": cfg.decision_model,
        "max_tokens": cfg.decision_max_tokens,
        "system": [{"type": "text", "text": CHARTER, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "USER TEXT"}],
        "output_config": {
            "effort": cfg.decision_effort,
            "format": {"type": "json_schema", "schema": ENTRY_SCHEMA},
        },
    }
    assert "Answer with exactly one JSON object" not in json.dumps(payload)


def test_decider_no_format_payload_embeds_schema_in_user_message(cfg):
    decider = DPOFDecider(cfg)
    payload = decider._build_anthropic_payload("USER TEXT", ENTRY_SCHEMA, use_format=False)
    assert payload["messages"][0]["content"] == "USER TEXT\n\n" + schema_instruction(ENTRY_SCHEMA)
    assert payload["system"] == [{"type": "text", "text": CHARTER, "cache_control": {"type": "ephemeral"}}]
    assert payload["output_config"] == {"effort": cfg.decision_effort}
    assert "format" not in payload["output_config"]


# ─────────────────────────────────────────────────────────────────────────
# Advisory payloads
# ─────────────────────────────────────────────────────────────────────────

def test_advisory_direct_payload_golden_is_unchanged():
    llm = _llm()
    payload = llm._build_anthropic_payload(
        system="SYS", user="USR", schema=_DEBATE_SCHEMA, model="claude-opus-5-5",
        max_tokens=100, use_format=True, effort="high",
    )
    assert payload == {
        "model": "claude-opus-5-5",
        "max_tokens": 100,
        "system": "SYS",
        "messages": [{"role": "user", "content": "USR"}],
        "output_config": {
            "effort": "high",
            "format": {"type": "json_schema", "schema": _DEBATE_SCHEMA},
        },
    }


def test_advisory_no_format_payload_embeds_schema_in_system():
    llm = _llm()
    payload = llm._build_anthropic_payload(
        system="SYS", user="USR", schema=_DEBATE_SCHEMA, model="claude-opus-5-5",
        max_tokens=100, use_format=False, effort="high",
    )
    assert payload["system"] == "SYS\n\n" + schema_instruction(_DEBATE_SCHEMA)
    assert payload["messages"] == [{"role": "user", "content": "USR"}]
    assert payload["output_config"] == {"effort": "high"}


# ─────────────────────────────────────────────────────────────────────────
# Gateway-mode init flag
# ─────────────────────────────────────────────────────────────────────────

def test_gateway_mode_initialises_downgrade_flag_true_for_both_classes(cfg, gateway_flag):
    gateway_flag(True)
    assert DPOFDecider(cfg)._anthropic_format_downgraded is True
    assert _llm()._anthropic_format_downgraded is True


def test_direct_mode_initialises_downgrade_flag_false_for_both_classes(cfg, gateway_flag):
    gateway_flag(False)
    assert DPOFDecider(cfg)._anthropic_format_downgraded is False
    assert _llm()._anthropic_format_downgraded is False


def test_gateway_mode_first_call_sends_schema_in_prompt_not_format(cfg, gateway_flag):
    gateway_flag(True)
    calls: list = []
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer()), calls))
    asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert len(calls) == 1
    assert "format" not in calls[0]["output_config"]
    assert calls[0]["messages"][0]["content"].endswith(schema_instruction(ENTRY_SCHEMA))


# ─────────────────────────────────────────────────────────────────────────
# 400 downgrade: retry payload must now carry the schema
# ─────────────────────────────────────────────────────────────────────────

def test_decider_400_downgrade_retry_carries_schema(cfg, gateway_flag):
    gateway_flag(False)
    calls: list = []
    transport = _sequence_transport(
        [_Fake400("400 Bad Request: output_config.format is not supported"), _result(_entry_answer())],
        default=_result(_entry_answer()),
        calls=calls,
    )
    decider = DPOFDecider(cfg, anthropic_transport=transport)
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.action == "enter"
    assert decider._anthropic_format_downgraded is True
    assert "format" in calls[0]["output_config"]
    assert "format" not in calls[1]["output_config"]
    retry_user = calls[1]["messages"][0]["content"]
    assert retry_user.endswith(schema_instruction(ENTRY_SCHEMA))
    for key in ENTRY_SCHEMA["required"]:
        assert f'"{key}"' in retry_user


def test_advisory_400_downgrade_retry_carries_schema(gateway_flag):
    gateway_flag(False)
    calls: list = []
    transport = _sequence_transport(
        [_Fake400("400 Bad Request: output_config.format is not supported"), _result(_debate_answer())],
        default=_result(_debate_answer()),
        calls=calls,
    )
    llm = _llm(anthropic_transport=transport)
    res = asyncio.run(llm.complete_json(
        purpose="test_purpose", system="SYS", user="USR", schema=_DEBATE_SCHEMA, schema_name="debate",
        primary_model="claude-opus-5-5", fallback_model="gpt-6-sol", timeout_seconds=5.0, budget=None,
    ))
    assert res.ok is True
    assert llm._anthropic_format_downgraded is True
    assert "format" in calls[0]["output_config"]
    assert "format" not in calls[1]["output_config"]
    assert calls[1]["system"] == "SYS\n\n" + schema_instruction(_DEBATE_SCHEMA)


# ─────────────────────────────────────────────────────────────────────────
# Gateway-flag helper failure modes: any failure must default to "direct"
# ─────────────────────────────────────────────────────────────────────────

def test_gateway_helper_false_when_config_import_fails(cfg, monkeypatch):
    import sys

    # `from agent.config import config` raises ImportError when the module is None.
    monkeypatch.setitem(sys.modules, "agent.config", None)
    dec = DPOFDecider(cfg)
    llm = _llm()
    assert dec._gateway_needs_prompt_schema() is False
    assert llm._gateway_needs_prompt_schema() is False
    assert dec._anthropic_format_downgraded is False
    assert llm._anthropic_format_downgraded is False


def test_gateway_helper_false_when_attribute_read_raises(cfg, monkeypatch):
    import agent.config as cfg_mod

    class _Broken:
        @property
        def anthropic_gateway(self):
            raise RuntimeError("config attribute unavailable")

    monkeypatch.setattr(cfg_mod, "config", _Broken())
    dec = DPOFDecider(cfg)
    llm = _llm()
    assert dec._gateway_needs_prompt_schema() is False
    assert llm._gateway_needs_prompt_schema() is False
    assert dec._anthropic_format_downgraded is False
    assert llm._anthropic_format_downgraded is False
