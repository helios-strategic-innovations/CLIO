"""Tests for agent.dpof.config's Kronos knobs (Task K2, agent/kronos --
docs/design/clio_kronos.md §4): kronos_context / kronos_max_age_seconds.
Dataclass-default/from_env-fallback parity (CLAUDE.md coding standard) and
the [120, 1800] clamp on CLIO_DPOF_KRONOS_MAX_AGE.
"""

from __future__ import annotations

import pytest

from agent.dpof.config import DPOFConfig, load_dpof_config


def test_dataclass_default_kronos_context_is_true():
    assert DPOFConfig().kronos_context is True


def test_dataclass_default_kronos_max_age_is_600():
    assert DPOFConfig().kronos_max_age_seconds == 600


def test_from_env_fallback_agrees_with_dataclass_default_kronos():
    """CLAUDE.md coding standard: the dataclass default AND from_env()'s
    fallback must agree -- exercised here with a fully clean environment
    (autouse _clean_dpof_env fixture strips CLIO_DPOF_*)."""
    d = DPOFConfig()
    env_cfg = load_dpof_config()
    assert env_cfg.kronos_context == d.kronos_context is True
    assert env_cfg.kronos_max_age_seconds == d.kronos_max_age_seconds == 600


def test_kronos_context_env_off(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_CONTEXT", "0")
    assert load_dpof_config().kronos_context is False


def test_kronos_context_env_on(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_CONTEXT", "true")
    assert load_dpof_config().kronos_context is True


def test_kronos_max_age_env_in_range(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_MAX_AGE", "300")
    assert load_dpof_config().kronos_max_age_seconds == 300


def test_kronos_max_age_env_clamped_below_120(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_MAX_AGE", "10")
    assert load_dpof_config().kronos_max_age_seconds == 120


def test_kronos_max_age_env_clamped_above_1800(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_MAX_AGE", "5000")
    assert load_dpof_config().kronos_max_age_seconds == 1800


def test_kronos_max_age_env_exactly_at_bounds(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_KRONOS_MAX_AGE", "120")
    assert load_dpof_config().kronos_max_age_seconds == 120
    monkeypatch.setenv("CLIO_DPOF_KRONOS_MAX_AGE", "1800")
    assert load_dpof_config().kronos_max_age_seconds == 1800


def test_validate_flags_out_of_range_kronos_max_age():
    from dataclasses import replace

    bad = replace(DPOFConfig(), kronos_max_age_seconds=50)
    problems = bad.validate()
    assert any("kronos_max_age_seconds" in p for p in problems)


def test_validate_ok_with_default_kronos_settings():
    problems = DPOFConfig().validate()
    assert not any("kronos" in p for p in problems)
