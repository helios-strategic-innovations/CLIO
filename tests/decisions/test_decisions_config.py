"""Tests for agent.decisions.config — DecisionsConfig / from_env(), clamps,
and default-parity (CLAUDE.md rule: dataclass defaults must equal from_env()'s
fallbacks -- from_env() derives its fallbacks straight off DecisionsConfig()'s
own defaults, so this is close to tautological, but a change to one site
without the other would still be caught by test_default_parity below)."""

from __future__ import annotations

import os

import pytest

from agent.decisions.config import DecisionsConfig, _parse_shadow_endpoints, from_env


@pytest.fixture(autouse=True)
def _clean_decisions_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("CLIO_DECISIONS_"):
            monkeypatch.delenv(key, raising=False)
    yield


def test_default_parity():
    """from_env() with nothing set must equal DecisionsConfig()'s own defaults."""
    assert from_env() == DecisionsConfig()


def test_enabled_bool_parsing(monkeypatch):
    monkeypatch.setenv("CLIO_DECISIONS_ENABLED", "0")
    assert from_env().enabled is False
    monkeypatch.setenv("CLIO_DECISIONS_ENABLED", "true")
    assert from_env().enabled is True
    monkeypatch.setenv("CLIO_DECISIONS_ENABLED", "yes")
    assert from_env().enabled is True


def test_shadow_bool_parsing(monkeypatch):
    monkeypatch.setenv("CLIO_DECISIONS_SHADOW", "0")
    assert from_env().shadow is False


def test_shadow_timeout_clamped(monkeypatch):
    monkeypatch.setenv("CLIO_DECISIONS_SHADOW_TIMEOUT", "0.01")
    assert from_env().shadow_timeout_seconds == 0.2  # clamped to the lo bound
    monkeypatch.setenv("CLIO_DECISIONS_SHADOW_TIMEOUT", "100")
    assert from_env().shadow_timeout_seconds == 10.0  # clamped to the hi bound
    monkeypatch.setenv("CLIO_DECISIONS_SHADOW_TIMEOUT", "3.5")
    assert from_env().shadow_timeout_seconds == 3.5


def test_shadow_timeout_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CLIO_DECISIONS_SHADOW_TIMEOUT", "not-a-number")
    assert from_env().shadow_timeout_seconds == DecisionsConfig().shadow_timeout_seconds


def test_outbox_path_override(monkeypatch, tmp_path):
    p = str(tmp_path / "outbox.jsonl")
    monkeypatch.setenv("CLIO_DECISIONS_OUTBOX_PATH", p)
    assert from_env().outbox_path == p


def test_outbox_max_lines_override_and_floor(monkeypatch):
    monkeypatch.setenv("CLIO_DECISIONS_OUTBOX_MAX_LINES", "500")
    assert from_env().outbox_max_lines == 500
    monkeypatch.setenv("CLIO_DECISIONS_OUTBOX_MAX_LINES", "0")
    assert from_env().outbox_max_lines == 1  # floored at 1, never 0/negative
    monkeypatch.setenv("CLIO_DECISIONS_OUTBOX_MAX_LINES", "-5")
    assert from_env().outbox_max_lines == 1


def test_serve_mode_only_jev_recognized(monkeypatch):
    monkeypatch.setenv("CLIO_DECISIONS_SERVE_MODE", "ensemble")
    assert from_env().serve_mode == "jev"  # only "jev" exists per DESIGN.md §6
    monkeypatch.setenv("CLIO_DECISIONS_SERVE_MODE", "jev")
    assert from_env().serve_mode == "jev"


# ─────────────────────────────────────────────────────────────────────────
# Shadow endpoints parsing
# ─────────────────────────────────────────────────────────────────────────

def test_parse_shadow_endpoints_empty():
    assert _parse_shadow_endpoints("") == ()
    assert _parse_shadow_endpoints("   ") == ()


def test_parse_shadow_endpoints_single_and_multiple():
    assert _parse_shadow_endpoints("alpha=http://127.0.0.1:8008") == (("alpha", "http://127.0.0.1:8008"),)
    assert _parse_shadow_endpoints("alpha=http://a,beta=https://b") == (
        ("alpha", "http://a"), ("beta", "https://b"),
    )


def test_parse_shadow_endpoints_skips_malformed_entries_keeps_good_ones():
    # "noequals" has no "=", "bad=ftp://x" isn't http(s), "=novalue" has no name.
    out = _parse_shadow_endpoints("alpha=http://a,noequals,bad=ftp://x,=novalue,beta=https://b")
    assert out == (("alpha", "http://a"), ("beta", "https://b"))


def test_parse_shadow_endpoints_whitespace_trimmed():
    assert _parse_shadow_endpoints(" alpha = http://a , beta=http://b ") == (
        ("alpha", "http://a"), ("beta", "http://b"),
    )


def test_shadow_endpoints_from_env(monkeypatch):
    monkeypatch.setenv("CLIO_DECISIONS_SHADOW_ENDPOINTS", "alpha=http://127.0.0.1:8008,beta=http://127.0.0.1:8009")
    cfg = from_env()
    assert cfg.shadow_endpoints == (
        ("alpha", "http://127.0.0.1:8008"), ("beta", "http://127.0.0.1:8009"),
    )


def test_dataclass_is_frozen():
    cfg = DecisionsConfig()
    with pytest.raises(Exception):
        cfg.enabled = False  # type: ignore[misc]
