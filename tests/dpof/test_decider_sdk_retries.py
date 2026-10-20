"""R-04: the decider's Anthropic client honours cfg.decider_sdk_max_retries.

No network: only client construction is inspected (anthropic.Anthropic does not
call the API on __init__).
"""

from __future__ import annotations

from dataclasses import replace

from agent.dpof.config import DPOFConfig, load_dpof_config
from agent.dpof.decider import DPOFDecider


def test_config_default_is_zero_retries(cfg):
    assert DPOFConfig().decider_sdk_max_retries == 0
    assert load_dpof_config().decider_sdk_max_retries == 0


def test_env_retries_clamped_and_bad_value_defaults_zero(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_DECISION_SDK_RETRIES", "9")
    assert load_dpof_config().decider_sdk_max_retries == 3
    monkeypatch.setenv("CLIO_DPOF_DECISION_SDK_RETRIES", "-4")
    assert load_dpof_config().decider_sdk_max_retries == 0
    monkeypatch.setenv("CLIO_DPOF_DECISION_SDK_RETRIES", "two")
    assert load_dpof_config().decider_sdk_max_retries == 0
    monkeypatch.setenv("CLIO_DPOF_DECISION_SDK_RETRIES", "2")
    assert load_dpof_config().decider_sdk_max_retries == 2


def test_client_uses_default_zero_retries(cfg):
    decider = DPOFDecider(cfg)
    client = decider._get_anthropic_client()
    assert client.max_retries == 0


def test_client_uses_configured_retries(cfg):
    decider = DPOFDecider(replace(cfg, decider_sdk_max_retries=2))
    assert decider._get_anthropic_client().max_retries == 2


def test_client_rebuilt_only_when_retries_change(cfg):
    decider = DPOFDecider(replace(cfg, decider_sdk_max_retries=1))
    first = decider._get_anthropic_client()
    assert decider._get_anthropic_client() is first
    decider.update_config(replace(cfg, decider_sdk_max_retries=3))
    second = decider._get_anthropic_client()
    assert second is not first
    assert second.max_retries == 3
