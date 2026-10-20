"""OP-04b: DPOF decider / advisory LLM rebuild their Anthropic client on
gateway/key/timeout/base_url changes, and build via config.anthropic_client_kwargs.
"""
import dataclasses

import anthropic
import pytest

from agent.config import config as app_config
from agent.dpof.advisory_llm import AdvisoryLLM
from agent.dpof.config import DPOFConfig
from agent.dpof.decider import DPOFDecider


class FakeAnthropic:
    built: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeAnthropic.built.append(kwargs)


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeAnthropic.built = []
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    return FakeAnthropic


@pytest.fixture
def cfg_state(monkeypatch):
    """Patch the live config singleton's Anthropic routing fields for one test."""
    monkeypatch.setattr(app_config, "anthropic_api_key", "key-A")
    monkeypatch.setattr(app_config, "anthropic_gateway", False)
    monkeypatch.setattr(app_config, "anthropic_base_url", "")
    return app_config


def _decider() -> DPOFDecider:
    return DPOFDecider(DPOFConfig(), data_dir="/tmp")


# ---------------- decider ----------------

def test_decider_direct_mode_builds_key_and_timeout_only(fake_sdk, cfg_state):
    d = _decider()
    d._get_anthropic_client()
    assert len(fake_sdk.built) == 1
    kw = fake_sdk.built[0]
    assert kw["api_key"] == "key-A"
    assert kw["timeout"] == d.cfg.decision_timeout_seconds
    assert "base_url" not in kw


def test_decider_gateway_mode_passes_base_url(fake_sdk, cfg_state, monkeypatch):
    monkeypatch.setattr(cfg_state, "anthropic_gateway", True)
    monkeypatch.setattr(cfg_state, "anthropic_base_url", "https://gw.example/v1")
    d = _decider()
    d._get_anthropic_client()
    assert fake_sdk.built[0]["base_url"] == "https://gw.example/v1"


def test_decider_reuses_client_when_signature_unchanged(fake_sdk, cfg_state):
    d = _decider()
    first = d._get_anthropic_client()
    second = d._get_anthropic_client()
    assert first is second
    assert len(fake_sdk.built) == 1


def test_decider_rebuilds_on_key_base_url_or_timeout_change(fake_sdk, cfg_state, monkeypatch):
    d = _decider()
    d._get_anthropic_client()
    monkeypatch.setattr(cfg_state, "anthropic_api_key", "key-B")
    d._get_anthropic_client()
    assert len(fake_sdk.built) == 2
    assert fake_sdk.built[1]["api_key"] == "key-B"

    monkeypatch.setattr(cfg_state, "anthropic_gateway", True)
    monkeypatch.setattr(cfg_state, "anthropic_base_url", "https://gw.example/v1")
    d._get_anthropic_client()
    assert len(fake_sdk.built) == 3
    assert fake_sdk.built[2]["base_url"] == "https://gw.example/v1"

    monkeypatch.setattr(cfg_state, "anthropic_base_url", "https://gw2.example/v1")
    d._get_anthropic_client()
    assert len(fake_sdk.built) == 4
    assert fake_sdk.built[3]["base_url"] == "https://gw2.example/v1"

    d.cfg = dataclasses.replace(d.cfg, decision_timeout_seconds=d.cfg.decision_timeout_seconds + 7)
    d._get_anthropic_client()
    assert len(fake_sdk.built) == 5
    assert fake_sdk.built[4]["timeout"] == d.cfg.decision_timeout_seconds


# ---------------- advisory ----------------

def _advisory(key="key-A") -> AdvisoryLLM:
    return AdvisoryLLM(anthropic_key_fn=lambda: key, openai_key_fn=lambda: "")


def test_advisory_direct_mode_builds_key_and_timeout_only(fake_sdk, cfg_state):
    llm = _advisory()
    llm._get_anthropic_client("key-A", 45.0)
    assert fake_sdk.built == [{"api_key": "key-A", "timeout": 45.0}]


def test_advisory_gateway_mode_passes_base_url(fake_sdk, cfg_state, monkeypatch):
    monkeypatch.setattr(cfg_state, "anthropic_gateway", True)
    monkeypatch.setattr(cfg_state, "anthropic_base_url", "https://gw.example/v1")
    llm = _advisory()
    llm._get_anthropic_client("key-A", 45.0)
    assert fake_sdk.built[0]["base_url"] == "https://gw.example/v1"


def test_advisory_reuses_client_when_signature_unchanged(fake_sdk, cfg_state):
    llm = _advisory()
    a = llm._get_anthropic_client("key-A", 45.0)
    b = llm._get_anthropic_client("key-A", 45.0)
    assert a is b
    assert len(fake_sdk.built) == 1


def test_advisory_rebuilds_on_key_base_url_or_timeout_change(fake_sdk, cfg_state, monkeypatch):
    llm = _advisory()
    llm._get_anthropic_client("key-A", 45.0)
    llm._get_anthropic_client("key-B", 45.0)
    assert len(fake_sdk.built) == 2
    assert fake_sdk.built[1]["api_key"] == "key-B"

    monkeypatch.setattr(cfg_state, "anthropic_gateway", True)
    monkeypatch.setattr(cfg_state, "anthropic_base_url", "https://gw.example/v1")
    llm._get_anthropic_client("key-B", 45.0)
    assert len(fake_sdk.built) == 3
    assert fake_sdk.built[2]["base_url"] == "https://gw.example/v1"

    monkeypatch.setattr(cfg_state, "anthropic_base_url", "https://gw2.example/v1")
    llm._get_anthropic_client("key-B", 45.0)
    assert len(fake_sdk.built) == 4

    llm._get_anthropic_client("key-B", 90.0)
    assert len(fake_sdk.built) == 5
    assert fake_sdk.built[4]["timeout"] == 90.0
