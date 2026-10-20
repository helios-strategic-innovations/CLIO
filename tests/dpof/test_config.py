"""Tests for agent.dpof.config: defaults, load_dpof_config() env resolution
(Alpaca key/feed fallback, capital_base), validate(), and jev_effective_mode.

Operator directive (2026-09-24): "paper mode IS the soak" (no separate
log-only staging) and "everything is built active, not shadow" (defaults are
the ACTIVE modes) — mode defaults to PAPER, jev_mode defaults to VETO (but is
inert until a TypeSafe key is configured; see jev_effective_mode).
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from agent.dpof.config import (
    DPOFConfig,
    DpofMode,
    JevMode,
    load_dpof_config,
)


# ─────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────

def test_dataclass_default_mode_is_paper():
    assert DPOFConfig().mode == DpofMode.PAPER


def test_dataclass_default_jev_mode_is_veto():
    assert DPOFConfig().jev_mode == JevMode.VETO


def test_dataclass_default_capital_base_is_20k():
    assert DPOFConfig().capital_base == 20_000.0


def test_dataclass_default_alpaca_key_source_is_none():
    assert DPOFConfig().alpaca_key_source == "none"


def test_from_env_fallback_agrees_with_dataclass_default(monkeypatch):
    """CLAUDE.md coding standard: the dataclass default AND from_env()'s
    fallback must agree — both sites are exercised here with a fully clean
    environment (autouse _clean_dpof_env fixture strips CLIO_DPOF_*/TYPESAFE_*;
    ALPACA_PAPER_* must be cleared too so a developer's real env can't leak
    into "no fallback configured" assertions)."""
    monkeypatch.delenv("ALPACA_PAPER_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_OPTIONS_FEED", raising=False)
    monkeypatch.delenv("ALPACA_STOCK_FEED", raising=False)

    d = DPOFConfig()
    env_cfg = load_dpof_config()
    assert env_cfg.mode == d.mode == DpofMode.PAPER
    assert env_cfg.jev_mode == d.jev_mode == JevMode.VETO
    assert env_cfg.capital_base == d.capital_base == 20_000.0
    assert env_cfg.options_feed == d.options_feed == "opra"
    assert env_cfg.stock_feed == d.stock_feed == "sip"
    assert env_cfg.alpaca_key_source == "none"


# ─────────────────────────────────────────────────────────────────────────
# Alpaca key fallback (load_dpof_config)
# ─────────────────────────────────────────────────────────────────────────

def _clear_alpaca(monkeypatch):
    for k in ("ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_no_keys_at_all_resolves_to_none_source(monkeypatch):
    _clear_alpaca(monkeypatch)
    c = load_dpof_config()
    assert c.alpaca_key_id == "" and c.alpaca_secret_key == "" and c.alpaca_key_source == "none"


def test_dedicated_keys_win_over_clio_paper_fallback(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_ALPACA_KEY_ID", "ded-id")
    monkeypatch.setenv("CLIO_DPOF_ALPACA_SECRET_KEY", "ded-secret")
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "clio-id")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "clio-secret")
    c = load_dpof_config()
    assert c.alpaca_key_id == "ded-id"
    assert c.alpaca_secret_key == "ded-secret"
    assert c.alpaca_key_source == "dedicated"


def test_clio_paper_fallback_when_dedicated_unset_and_paper(monkeypatch):
    _clear_alpaca(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "clio-id")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "clio-secret")
    c = load_dpof_config()  # alpaca_paper defaults True
    assert c.alpaca_key_id == "clio-id"
    assert c.alpaca_secret_key == "clio-secret"
    assert c.alpaca_key_source == "clio_paper"


def test_clio_paper_fallback_never_applies_to_live(monkeypatch):
    _clear_alpaca(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "clio-id")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "clio-secret")
    monkeypatch.setenv("CLIO_DPOF_ALPACA_PAPER", "false")
    c = load_dpof_config()
    assert c.alpaca_key_id == "" and c.alpaca_secret_key == ""
    assert c.alpaca_key_source == "none"
    assert c.alpaca_paper is False


def test_partial_dedicated_id_only_falls_back_to_full_clio_pair_never_mixed(monkeypatch):
    """A lone dedicated id (no dedicated secret) must NOT pair with the
    shared secret -- the fallback is all-or-nothing from ONE source."""
    monkeypatch.setenv("CLIO_DPOF_ALPACA_KEY_ID", "ded-id")
    monkeypatch.delenv("CLIO_DPOF_ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "clio-id")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "clio-secret")
    c = load_dpof_config()
    assert c.alpaca_key_id == "clio-id"
    assert c.alpaca_secret_key == "clio-secret"
    assert c.alpaca_key_source == "clio_paper"
    # Never "ded-id" + "clio-secret".
    assert c.alpaca_key_id != "ded-id"


def test_partial_dedicated_secret_only_falls_back_to_full_clio_pair_never_mixed(monkeypatch):
    monkeypatch.delenv("CLIO_DPOF_ALPACA_KEY_ID", raising=False)
    monkeypatch.setenv("CLIO_DPOF_ALPACA_SECRET_KEY", "ded-secret")
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "clio-id")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "clio-secret")
    c = load_dpof_config()
    assert c.alpaca_key_id == "clio-id"
    assert c.alpaca_secret_key == "clio-secret"
    assert c.alpaca_key_source == "clio_paper"
    assert c.alpaca_secret_key != "ded-secret"


def test_partial_dedicated_with_no_clio_fallback_available_stays_partial_none_source(monkeypatch):
    _clear_alpaca(monkeypatch)
    monkeypatch.setenv("CLIO_DPOF_ALPACA_KEY_ID", "ded-id")
    monkeypatch.delenv("CLIO_DPOF_ALPACA_SECRET_KEY", raising=False)
    c = load_dpof_config()
    assert c.alpaca_key_id == "ded-id"
    assert c.alpaca_secret_key == ""
    assert c.alpaca_key_source == "none"


# ─────────────────────────────────────────────────────────────────────────
# Feed fallbacks (ALPACA_OPTIONS_FEED / ALPACA_STOCK_FEED)
# ─────────────────────────────────────────────────────────────────────────

def test_options_feed_falls_back_to_shared_alpaca_env(monkeypatch):
    monkeypatch.setenv("ALPACA_OPTIONS_FEED", "opra")
    c = load_dpof_config()
    assert c.options_feed == "opra"


def test_options_feed_dpof_specific_env_wins_over_shared(monkeypatch):
    monkeypatch.setenv("ALPACA_OPTIONS_FEED", "opra")
    monkeypatch.setenv("CLIO_DPOF_OPTIONS_FEED", "indicative")
    c = load_dpof_config()
    assert c.options_feed == "indicative"


def test_options_feed_defaults_opra_with_no_env(monkeypatch):
    """Operator confirmed Algo Trader Plus (real-time OPRA) + options level 3
    on the account (2026-09-24) -- "opra" is the default now, not
    "indicative"."""
    monkeypatch.delenv("ALPACA_OPTIONS_FEED", raising=False)
    c = load_dpof_config()
    assert c.options_feed == "opra"


def test_stock_feed_falls_back_to_shared_alpaca_env(monkeypatch):
    monkeypatch.setenv("ALPACA_STOCK_FEED", "sip")
    c = load_dpof_config()
    assert c.stock_feed == "sip"


def test_stock_feed_dpof_specific_env_wins_over_shared(monkeypatch):
    monkeypatch.setenv("ALPACA_STOCK_FEED", "sip")
    monkeypatch.setenv("CLIO_DPOF_STOCK_FEED", "iex")
    c = load_dpof_config()
    assert c.stock_feed == "iex"


def test_stock_feed_defaults_sip_with_no_env(monkeypatch):
    """Algo Trader Plus includes full SIP (operator confirmed 2026-09-24)."""
    monkeypatch.delenv("ALPACA_STOCK_FEED", raising=False)
    c = load_dpof_config()
    assert c.stock_feed == "sip"


# ─────────────────────────────────────────────────────────────────────────
# capital_base (CLIO_DPOF_CAPITAL, validate())
# ─────────────────────────────────────────────────────────────────────────

def test_capital_base_from_env(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_CAPITAL", "35000")
    c = load_dpof_config()
    assert c.capital_base == 35_000.0


def test_capital_base_zero_disables_virtual_capital_and_is_valid(cfg):
    c = replace(cfg, capital_base=0.0, alpaca_key_id="k", alpaca_secret_key="s")
    assert "capital_base" not in " ".join(c.validate())


def test_capital_base_negative_is_invalid(cfg):
    c = replace(cfg, capital_base=-1.0)
    problems = c.validate()
    assert any("capital_base" in p for p in problems)


def test_capital_base_non_finite_is_invalid(cfg):
    for bad in (float("nan"), float("inf")):
        c = replace(cfg, capital_base=bad)
        problems = c.validate()
        assert any("capital_base" in p for p in problems)


def test_capital_base_below_min_account_equity_is_invalid(cfg):
    c = replace(cfg, capital_base=100.0, min_account_equity=2_000.0)
    problems = c.validate()
    assert any("capital_base" in p and "min_account_equity" in p for p in problems)


def test_capital_base_at_or_above_min_account_equity_is_valid(cfg):
    c = replace(cfg, capital_base=2_000.0, min_account_equity=2_000.0, alpaca_key_id="k", alpaca_secret_key="s")
    problems = c.validate()
    assert not any("capital_base" in p for p in problems)


# ─────────────────────────────────────────────────────────────────────────
# validate() no longer flags an unconfigured Jev
# ─────────────────────────────────────────────────────────────────────────

def test_validate_never_flags_unconfigured_jev(cfg):
    """The new VETO default with no key must never make the whole engine
    report as config-invalid — jev_effective_mode degrading to OFF is
    silent, not a validate() problem."""
    c = replace(cfg, jev_mode=JevMode.VETO, alpaca_key_id="k", alpaca_secret_key="s")
    problems = c.validate()
    assert not any("jev" in p.lower() or "typesafe" in p.lower() for p in problems)


# ─────────────────────────────────────────────────────────────────────────
# jev_effective_mode
# ─────────────────────────────────────────────────────────────────────────

def test_jev_effective_mode_off_stays_off_even_with_key(cfg):
    c = replace(cfg, jev_mode=JevMode.OFF, jev_api_key="k")
    assert c.jev_effective_mode == JevMode.OFF


def test_jev_effective_mode_veto_with_no_config_degrades_to_off(cfg):
    c = replace(cfg, jev_mode=JevMode.VETO)
    assert c.jev_api_key == "" and c.jev_key_file == "" and c.jev_fallback_base_url == ""
    assert c.jev_effective_mode == JevMode.OFF


@pytest.mark.parametrize("field", ["jev_api_key", "jev_key_file", "jev_fallback_base_url"])
def test_jev_effective_mode_any_single_knob_keeps_it_active(cfg, field):
    c = replace(cfg, jev_mode=JevMode.VETO, **{field: "x"})
    assert c.jev_effective_mode == JevMode.VETO


def test_jev_effective_mode_shadow_with_no_config_degrades_to_off(cfg):
    c = replace(cfg, jev_mode=JevMode.SHADOW)
    assert c.jev_effective_mode == JevMode.OFF


# ─────────────────────────────────────────────────────────────────────────
# Task 1 (2026-09-25, operator directive): retired/renamed knobs
# ─────────────────────────────────────────────────────────────────────────

def test_retired_blocking_knobs_are_gone():
    """block_on_breaking_news/block_on_vol_regimes only ever existed to
    BLOCK an entry; both are retired outright -- nothing surfaces them any
    more (not the dataclass, not from_env(), not config_summary())."""
    d = DPOFConfig()
    assert not hasattr(d, "block_on_breaking_news")
    assert not hasattr(d, "block_on_vol_regimes")
    from dataclasses import fields
    field_names = {f.name for f in fields(d)}
    assert "block_on_breaking_news" not in field_names
    assert "block_on_vol_regimes" not in field_names


def test_renamed_advisory_label_knobs_exist_with_original_defaults():
    """vix_halt_at -> vix_extreme_label_at, econ_blackout_minutes ->
    event_context_window_minutes: same default VALUES, advisory role now."""
    d = DPOFConfig()
    assert not hasattr(d, "vix_halt_at")
    assert not hasattr(d, "econ_blackout_minutes")
    assert d.vix_extreme_label_at == 35.0
    assert d.event_context_window_minutes == 30


def test_renamed_env_vars_resolve_into_the_renamed_fields(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_VIX_EXTREME_LABEL", "40.0")
    monkeypatch.setenv("CLIO_DPOF_EVENT_CONTEXT_WINDOW_MIN", "45")
    c = load_dpof_config()
    assert c.vix_extreme_label_at == 40.0
    assert c.event_context_window_minutes == 45


def test_old_retired_env_vars_are_silently_ignored(monkeypatch):
    """The old env var NAMES (CLIO_DPOF_VIX_HALT / _BLOCK_BREAKING_NEWS /
    _BLOCK_VOL_REGIMES / _ECON_BLACKOUT_MIN) no longer map to anything --
    setting them must not raise and must not change the new fields'
    defaults (they are simply unread now)."""
    monkeypatch.setenv("CLIO_DPOF_VIX_HALT", "50.0")
    monkeypatch.setenv("CLIO_DPOF_ECON_BLACKOUT_MIN", "99")
    monkeypatch.setenv("CLIO_DPOF_BLOCK_BREAKING_NEWS", "false")
    monkeypatch.setenv("CLIO_DPOF_BLOCK_VOL_REGIMES", "HIGH")
    c = load_dpof_config()  # must not raise
    assert c.vix_extreme_label_at == 35.0
    assert c.event_context_window_minutes == 30


def test_validate_still_clean_after_knob_retirement_rename(cfg):
    """validate() never referenced the retired/renamed knobs in the first
    place -- confirms the rename didn't silently break validation."""
    c = replace(cfg, alpaca_key_id="k", alpaca_secret_key="s")
    assert c.validate() == []


def test_config_summary_reflects_renamed_fields_not_old_names(cfg):
    from agent.dpof.config import config_summary
    summary = config_summary(cfg)
    assert "vix_extreme_label_at" in summary
    assert "event_context_window_minutes" in summary
    assert "vix_halt_at" not in summary
    assert "econ_blackout_minutes" not in summary
    assert "block_on_breaking_news" not in summary
    assert "block_on_vol_regimes" not in summary
