"""LOG-ONLY claim checker (agent.dpof.claim_check) and its decider wiring (R-07).

No network: decider calls go through injected fake transports (see test_decider.py).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, replace

import pytest

import agent.dpof.decider as decider_mod
from agent.dpof.claim_check import check_claims
from agent.dpof.decider import DPOFDecider
from tests.dpof.test_attempt_log import _cfg_with_fallback, _rows, _run
from tests.dpof.test_decider import (
    FakeClock,
    _candidates,
    _entry_answer,
    _entry_ctx,
    _fixed_transport,
    _result,
)

LOGGER_NAME = "clio.dpof.decider"
NOFA_ONLY = {"underlyings": {"SPY": {"pillars": {"nofa": -0.7493, "premium": -1.0}}}}
QUIET = {"underlyings": {"SPY": {"pillars": {"nofa": -0.7493, "premium": 0.05}}}}


# ─────────────────────────────────────────────────────────────────────────
# The four real contradictions seen on 2026-10-12
# ─────────────────────────────────────────────────────────────────────────
def test_real_reclaimed_vwap_contradiction_is_flagged():
    assert check_claims("SPY has reclaimed VWAP (778.3 vs 778.91)", {}) == ["vwap_relation:778.3<778.91:reclaimed"]


def test_real_single_pillar_claim_against_premium_is_flagged():
    assert check_claims("SPY bearish nomination is single-pillar NOFA (-0.7493)", NOFA_ONLY) == ["single_pillar_claim:premium=-1.00"]


def test_real_gap_mismatch_is_flagged():
    assert check_claims("SPY is only 0.77 below VWAP (778.55 vs 778.77)", {}) == ["gap_mismatch:claimed0.77:actual0.22"]


def test_real_below_vwap_when_price_is_above_is_flagged():
    assert check_claims("SPY is barely below VWAP (778.43 vs 777.96)", {}) == ["vwap_relation:778.43>777.96:below"]


# ─────────────────────────────────────────────────────────────────────────
# Correct statements stay clean
# ─────────────────────────────────────────────────────────────────────────
def test_correct_reclaim_statement_is_clean():
    assert check_claims("SPY has reclaimed VWAP (778.91 vs 778.3)", {}) == []


def test_correct_gap_statement_is_clean():
    assert check_claims("SPY is only 0.22 below VWAP (778.55 vs 778.77)", {}) == []


def test_single_pillar_claim_with_quiet_other_pillars_is_clean():
    assert check_claims("SPY nomination is single-pillar NOFA (-0.7493)", QUIET) == []


def test_option_mark_pairs_are_not_vwap_claims():
    # Replay of real review text: 'mark 3.175 vs 2.91 ... below ...' is an option mark against entry, not VWAP.
    assert check_claims("Six minutes in, the put is +9% (mark 3.175 vs 2.91), still below the give-back line.", {}) == []
    assert check_claims("Down 21% (mark 2.07 vs 2.62), the gap is 0.95 below entry.", {}) == []


def test_a_pair_without_vwap_nearby_is_not_checked_even_with_a_relation_word():
    assert check_claims("SPY has reclaimed 778.3 vs 778.91", {}) == []
    assert check_claims("SPY has reclaimed VWAP (778.3 vs 778.91)", {}) == ["vwap_relation:778.3<778.91:reclaimed"]


def test_gex_and_kronos_never_contradict_a_single_pillar_claim():
    ctx = {"underlyings": {"SPY": {"pillars": {"nofa": -0.7, "premium": 0.05, "gex": 0.5, "kronos": 0.4}}}}
    assert check_claims("SPY nomination is single-pillar NOFA", ctx) == []
    assert check_claims("SPY: only NOFA supports the trade", ctx) == []


def test_vwap_mentioned_far_from_the_pair_does_not_make_it_a_vwap_claim():
    far = ("VWAP was mentioned long ago. The market is quiet and nothing else matters here today at all. "
           "Separately, the put has reclaimed 778.3 vs 778.91 on the mark.")
    assert check_claims(far, {}) == []


def test_text_beyond_the_input_cap_is_not_evaluated():
    padded = ("filler " * 300) + " SPY has reclaimed VWAP (778.3 vs 778.91)"
    assert len(padded) > 1500
    assert check_claims(padded, {}) == []


def test_the_same_violation_repeated_is_reported_once():
    twice = "SPY has reclaimed VWAP (778.3 vs 778.91). Again: SPY has reclaimed VWAP (778.3 vs 778.91)."
    assert check_claims(twice, {}) == ["vwap_relation:778.3<778.91:reclaimed"]


def test_relation_word_after_the_pair_is_checked_too():
    assert check_claims("VWAP check: (778.3 vs 778.91) SPY reclaimed the level", {}) == ["vwap_relation:778.3<778.91:reclaimed"]


def test_relation_word_before_the_pair_with_the_other_direction():
    assert check_claims("SPY lost VWAP (778.91 vs 778.3)", {}) == ["vwap_relation:778.91>778.3:lost"]


def test_consistent_relation_in_the_after_position_is_clean():
    assert check_claims("(778.3 vs 778.91) is still below VWAP", {}) == []


# ─────────────────────────────────────────────────────────────────────────
# Tolerance boundary (0.02 inclusive, 0.03 flagged)
# ─────────────────────────────────────────────────────────────────────────
def test_gap_exactly_at_tolerance_is_clean():
    # |0.20 - 0.22| == 0.02 is not greater than 0.02
    assert check_claims("SPY is 0.20 below VWAP (778.55 vs 778.77)", {}) == []


def test_gap_over_tolerance_is_flagged():
    # |0.19 - 0.22| == 0.03 > 0.02
    assert check_claims("SPY is 0.19 below VWAP (778.55 vs 778.77)", {}) == ["gap_mismatch:claimed0.19:actual0.22"]


# ─────────────────────────────────────────────────────────────────────────
# Single-pillar threshold and bare claims
# ─────────────────────────────────────────────────────────────────────────
def test_other_pillar_just_under_point_five_is_not_a_contradiction():
    ctx = {"underlyings": {"SPY": {"pillars": {"nofa": -0.7, "premium": 0.49}}}}
    assert check_claims("SPY single-pillar NOFA", ctx) == []


def test_other_pillar_at_point_five_is_a_contradiction():
    ctx = {"underlyings": {"SPY": {"pillars": {"nofa": -0.7, "premium": 0.5}}}}
    assert check_claims("SPY single-pillar NOFA", ctx) == ["single_pillar_claim:premium=0.50"]


def test_bare_single_pillar_prose_is_not_a_claim_and_is_never_judged():
    # Replay of 441 real texts: "the same single-pillar fade seen this morning" is prose, not a checkable claim.
    strong_two = {"underlyings": {"SPY": {"pillars": {"nofa": -0.7, "premium": -0.9}}}}
    assert check_claims("SPY: this is the same single-pillar fade seen this morning", strong_two) == []
    assert check_claims("SPY: this is a single-pillar setup", strong_two) == []


def test_named_single_pillar_claim_needs_a_named_underlying_and_uses_only_its_pillars():
    both = {"underlyings": {"SPY": {"pillars": {"nofa": -0.7, "premium": 0.05}},
                            "QQQ": {"pillars": {"nofa": 0.6, "premium": -0.9}}}}
    assert check_claims("single-pillar NOFA", both) == []                       # no underlying named: not checkable
    assert check_claims("SPY single-pillar NOFA", both) == []                   # SPY's other pillars are quiet
    assert check_claims("QQQ single-pillar NOFA", both) == ["single_pillar_claim:premium=-0.90"]


def test_only_premium_claim_with_quiet_other_flow_pillars_is_clean():
    # Real 10-12 10:21 text: "only premium at -1.0 is strong" while NOFA/darkpool were ~0 (GEX -0.3 is not a flow pillar).
    ctx = {"underlyings": {"SPY": {"pillars": {"nofa": 0.0, "premium": -1.0, "darkpool": 0.0001, "gex": -0.3}}}}
    assert check_claims("SPY composite is weak with only premium at -1.0 strong", ctx) == []


def test_non_numeric_and_nan_pillar_values_are_ignored():
    ctx = {"underlyings": {"SPY": {"pillars": {"nofa": -0.7, "premium": float("nan"), "gex": True, "darkpool": "x"}}}}
    assert check_claims("single-pillar NOFA", ctx) == []


def test_claim_check_env_is_normalised_to_log_or_off(monkeypatch):
    from agent.dpof.config import load_dpof_config

    monkeypatch.delenv("CLIO_DPOF_CLAIM_CHECK", raising=False)
    assert load_dpof_config().claim_check_mode == "log"
    monkeypatch.setenv("CLIO_DPOF_CLAIM_CHECK", "OFF")
    assert load_dpof_config().claim_check_mode == "off"
    monkeypatch.setenv("CLIO_DPOF_CLAIM_CHECK", "enforce")
    assert load_dpof_config().claim_check_mode == "log"


def test_empty_or_non_string_text_is_clean():
    assert check_claims("", {}) == []
    assert check_claims(None, {}) == []  # type: ignore[arg-type]
    assert check_claims("SPY reclaimed VWAP", None) == []  # type: ignore[arg-type]


def test_hostile_ten_thousand_char_text_is_bounded_and_does_not_hang():
    hostile = ("1 vs 2 below " * 800) + ("   " * 3000) + ("0.1 " * 1000)
    assert len(hostile) > 10_000
    started = time.perf_counter()
    check_claims(hostile, NOFA_ONLY)
    check_claims("single-pillar " + "a" * 10_000, NOFA_ONLY)
    assert time.perf_counter() - started < 1.0


# ─────────────────────────────────────────────────────────────────────────
# Decider wiring: logging, ledger placement, flag, fail-open
# ─────────────────────────────────────────────────────────────────────────
VIOLATING = "SPY has reclaimed VWAP (778.3 vs 778.91)"


def _decide(cfg, thesis, *, action="pass", fallback_transport=None, anthropic_transport=None):
    decider = DPOFDecider(
        cfg,
        anthropic_transport=anthropic_transport or _fixed_transport(_result(_entry_answer(
            action=action, candidate_key="" if action == "pass" else "C1", qty=0 if action == "pass" else 1,
            thesis=thesis))),
        openai_transport=fallback_transport,
        clock=FakeClock(),
    )
    result = _run(decider, lambda: decider.decide_entry(_entry_ctx(), _candidates()))
    return result


def _warnings(caplog):
    return [r for r in caplog.records if r.name == LOGGER_NAME and "DPOF claim check" in r.getMessage()]


def test_violating_thesis_logs_warning_and_ledgers_the_codes(cfg, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    result = _decide(cfg, VIOLATING)
    assert result.action == "pass"
    warns = _warnings(caplog)
    assert len(warns) == 1 and warns[0].levelno == logging.WARNING
    assert "vwap_relation:778.3<778.91:reclaimed" in warns[0].getMessage()
    rows = _rows(cfg)
    assert rows[0]["claim_violations"] == ["vwap_relation:778.3<778.91:reclaimed"]


def test_clean_thesis_logs_nothing_and_ledgers_an_empty_list(cfg, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    _decide(cfg, "SPY has reclaimed VWAP (778.91 vs 778.3)")
    assert _warnings(caplog) == []
    assert _rows(cfg)[0]["claim_violations"] == []


def test_flag_off_leaves_the_decision_identical_and_logs_and_ledgers_nothing(cfg, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    on = _decide(cfg, VIOLATING)
    on_rows = _rows(cfg)
    off_cfg = replace(cfg, claim_check_mode="off", data_dir=cfg.data_dir + "_off")
    caplog.clear()
    off = _decide(off_cfg, VIOLATING)
    assert _warnings(caplog) == []
    assert _rows(off_cfg)[0]["claim_violations"] == []
    assert on_rows[0]["claim_violations"] != []
    a, b = asdict(on), asdict(off)
    for k in ("latency_ms", "raw"):
        a.pop(k), b.pop(k)
    assert a == b


def test_violations_attach_to_the_attempt_that_produced_the_decision(cfg):
    """Primary times out; the OpenAI fallback decides with the violating thesis: only its row carries the codes."""
    cfg2 = _cfg_with_fallback(cfg)
    result = _decide(
        cfg2, VIOLATING,
        anthropic_transport=_fixed_transport(asyncio.TimeoutError()),
        fallback_transport=_fixed_transport(_result(_entry_answer(action="pass", candidate_key="", qty=0,
                                                                  thesis=VIOLATING), model="gpt-6-sol")),
    )
    assert result.provider == "openai"
    primary, fallback = _rows(cfg2)
    assert primary["provider"] == "anthropic" and primary["claim_violations"] == []
    assert fallback["provider"] == "openai" and fallback["claim_violations"] == ["vwap_relation:778.3<778.91:reclaimed"]


def test_a_raising_checker_never_reaches_the_decision_path(cfg, caplog, monkeypatch):
    def boom(text, ctx):
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(decider_mod, "check_claims", boom)
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    result = _decide(cfg, VIOLATING)
    assert result.action == "pass" and result.thesis == VIOLATING
    assert _rows(cfg)[0]["claim_violations"] == []
    assert [r for r in _warnings(caplog) if r.levelno >= logging.WARNING] == []
