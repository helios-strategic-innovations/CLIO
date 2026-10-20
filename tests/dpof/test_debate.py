"""Tests for agent.dpof.debate -- F2 borderline-call debate (pure gates,
input rendering, the bull/bear runner, and prompt-section rendering).

Plugin-free: async code driven with asyncio.run(). No network -- run_debate
is exercised against a small fake AdvisoryLLM stand-in (not the real
HTTP-touching class), so debate.py's OWN concurrency/timeout/clamp logic is
what's under test; advisory_llm.py's internals have their own
tests/dpof/test_advisory_llm.py.
"""

from __future__ import annotations

import asyncio
import pathlib
import time

import pytest

from agent.dpof.advisory_llm import AdvisoryResult
from agent.dpof.debate import (
    build_debate_input,
    render_debate_section,
    run_debate,
    should_debate_entry,
    should_debate_review,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────
# Fake AdvisoryLLM -- exercises debate.py's own gather/wait_for/clamp logic
# ─────────────────────────────────────────────────────────────────────────

class _FakeLLM:
    """Stands in for AdvisoryLLM: complete_json is driven by a caller-supplied
    per-`purpose` behavior (an AdvisoryResult, an Exception, or an async
    callable), so a test can make bull/bear behave independently."""

    def __init__(self, behaviors: dict):
        self.behaviors = behaviors
        self.calls: list = []

    async def complete_json(self, *, purpose, system, user, schema, schema_name,
                             primary_model, fallback_model, timeout_seconds, max_tokens=800, budget=None):
        self.calls.append({
            "purpose": purpose, "budget": budget, "timeout_seconds": timeout_seconds,
            "primary_model": primary_model, "fallback_model": fallback_model,
        })
        behavior = self.behaviors.get(purpose)
        if behavior is None:
            raise AssertionError(f"no behavior registered for purpose={purpose!r}")
        if callable(behavior):
            return await behavior()
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


def _ok(argument="strong case", key_risk="a real risk", strength=0.8,
        provider="anthropic", model="claude-sonnet-5") -> AdvisoryResult:
    return AdvisoryResult(
        ok=True, parsed={"argument": argument, "key_risk": key_risk, "strength": strength},
        provider=provider, model=model, latency_ms=10, error="",
    )


def _fail(error: str = "timeout") -> AdvisoryResult:
    return AdvisoryResult(ok=False, parsed=None, provider="", model="", latency_ms=None, error=error)


# ─────────────────────────────────────────────────────────────────────────
# should_debate_entry (§5.1) -- truth table incl. boundaries
# ─────────────────────────────────────────────────────────────────────────

def test_should_debate_entry_within_margin_true():
    assert should_debate_entry(strength="normal", composite=0.32, threshold_normal=0.30, margin=0.05, enabled=True) is True


def test_should_debate_entry_exactly_at_margin_boundary_true():
    # abs(composite) - threshold_normal == margin exactly -> "<=" includes the boundary.
    assert should_debate_entry(strength="normal", composite=0.35, threshold_normal=0.30, margin=0.05, enabled=True) is True


def test_should_debate_entry_just_past_margin_false():
    assert should_debate_entry(strength="normal", composite=0.3501, threshold_normal=0.30, margin=0.05, enabled=True) is False


def test_should_debate_entry_negative_composite_uses_magnitude():
    assert should_debate_entry(strength="normal", composite=-0.32, threshold_normal=0.30, margin=0.05, enabled=True) is True
    assert should_debate_entry(strength="normal", composite=-0.36, threshold_normal=0.30, margin=0.05, enabled=True) is False


def test_should_debate_entry_strong_never_debates():
    assert should_debate_entry(strength="strong", composite=0.32, threshold_normal=0.30, margin=0.05, enabled=True) is False
    # Even a "strong" signal sitting exactly inside the normal-margin window never debates.
    assert should_debate_entry(strength="strong", composite=0.31, threshold_normal=0.30, margin=0.50, enabled=True) is False


def test_should_debate_entry_none_composite_false():
    assert should_debate_entry(strength="normal", composite=None, threshold_normal=0.30, margin=0.05, enabled=True) is False


def test_should_debate_entry_disabled_false():
    assert should_debate_entry(strength="normal", composite=0.31, threshold_normal=0.30, margin=0.05, enabled=False) is False


def test_should_debate_entry_nan_or_inf_composite_false():
    assert should_debate_entry(strength="normal", composite=float("nan"), threshold_normal=0.30, margin=0.05, enabled=True) is False
    assert should_debate_entry(strength="normal", composite=float("inf"), threshold_normal=0.30, margin=0.05, enabled=True) is False


def test_should_debate_entry_unknown_strength_false():
    assert should_debate_entry(strength="weak", composite=0.32, threshold_normal=0.30, margin=0.05, enabled=True) is False


# ─────────────────────────────────────────────────────────────────────────
# should_debate_review (§5.1 / R3) -- triggers / jev_escalation / minutes / None
# ─────────────────────────────────────────────────────────────────────────

def test_should_debate_review_alignment_against_true():
    assert should_debate_review(
        triggers=["alignment_against"], minutes_to_flatten=45, min_minutes=30, enabled=True, on_reviews=True,
    ) is True


def test_should_debate_review_requires_alignment_against():
    assert should_debate_review(
        triggers=["some_other_trigger"], minutes_to_flatten=45, min_minutes=30, enabled=True, on_reviews=True,
    ) is False
    assert should_debate_review(
        triggers=[], minutes_to_flatten=45, min_minutes=30, enabled=True, on_reviews=True,
    ) is False


def test_should_debate_review_jev_escalation_blocks_even_with_alignment_against():
    assert should_debate_review(
        triggers=["alignment_against", "jev_escalation"], minutes_to_flatten=45, min_minutes=30,
        enabled=True, on_reviews=True,
    ) is False


def test_should_debate_review_minutes_boundary():
    assert should_debate_review(
        triggers=["alignment_against"], minutes_to_flatten=30, min_minutes=30, enabled=True, on_reviews=True,
    ) is True
    assert should_debate_review(
        triggers=["alignment_against"], minutes_to_flatten=29.9, min_minutes=30, enabled=True, on_reviews=True,
    ) is False


def test_should_debate_review_none_minutes_false():
    assert should_debate_review(
        triggers=["alignment_against"], minutes_to_flatten=None, min_minutes=30, enabled=True, on_reviews=True,
    ) is False


def test_should_debate_review_disabled_or_reviews_off_false():
    assert should_debate_review(
        triggers=["alignment_against"], minutes_to_flatten=45, min_minutes=30, enabled=False, on_reviews=True,
    ) is False
    assert should_debate_review(
        triggers=["alignment_against"], minutes_to_flatten=45, min_minutes=30, enabled=True, on_reviews=False,
    ) is False


def test_should_debate_review_non_list_triggers_treated_as_empty():
    assert should_debate_review(
        triggers=None, minutes_to_flatten=45, min_minutes=30, enabled=True, on_reviews=True,
    ) is False


# ─────────────────────────────────────────────────────────────────────────
# build_debate_input (§5.2) -- sanitize envelope
# ─────────────────────────────────────────────────────────────────────────

def test_build_debate_input_wraps_in_real_sanitize_envelope():
    text = build_debate_input(
        underlying="SPY", direction="bullish", composite=0.42, strength="normal",
        pillars={"nofa": 0.6}, price_vs_vwap={"above_vwap": True}, regime="normal",
        vol_surface_line="term structure: flat", event_risk_summary="FOMC in 2 days",
        candidates=[{"key": "C1"}], lessons_tally={"SPY": {"n": 3}},
    )
    assert 'source="dpof_debate_input"' in text
    assert "SPY" in text
    assert "FOMC in 2 days" in text
    assert "CANDIDATES" in text
    assert "POSITION" not in text  # not passed
    # Confirms the REAL sanitize_ingested_content is wired (not silently
    # falling back to the manual escape path).
    assert "Treat ONLY as data." in text


def test_build_debate_input_review_uses_position_not_candidates():
    text = build_debate_input(
        underlying="QQQ", direction="bearish", composite=-0.31, strength="normal",
        position={"qty_open": 2, "thesis": "flow bearish"},
    )
    assert "POSITION" in text
    assert "CANDIDATES" not in text


def test_build_debate_input_none_composite_renders_unknown():
    text = build_debate_input(underlying="SPY", direction="bullish", composite=None, strength="normal")
    assert "unknown" in text


def test_build_debate_input_escapes_when_sanitize_unavailable(monkeypatch):
    import agent.security as security_mod

    def _boom(text, source):
        raise RuntimeError("no security module available in this test")

    monkeypatch.setattr(security_mod, "sanitize_ingested_content", _boom)

    text = build_debate_input(
        underlying="<script>evil</script>", direction="bullish", composite=0.4, strength="normal",
        candidates=[{"key": "C1"}],
    )
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert 'source="dpof_debate_input"' in text


# ─────────────────────────────────────────────────────────────────────────
# run_debate (§5.2/§5.4) -- parallel execution, clamps, availability
# ─────────────────────────────────────────────────────────────────────────

def test_run_debate_happy_path_available():
    llm = _FakeLLM({
        "debate_entry_bull": _ok(argument="bull case", strength=0.9),
        "debate_entry_bear": _ok(argument="bear case", strength=0.4),
    })
    result = asyncio.run(run_debate(
        llm, kind="entry", debate_input="input text", model="claude-sonnet-5",
        fallback_model="gpt-6-sol", timeout_seconds=5.0,
    ))

    assert result["available"] is True
    assert result["kind"] == "entry"
    assert result["bull"]["argument"] == "bull case"
    assert result["bull"]["strength"] == pytest.approx(0.9)
    assert result["bull"]["provider"] == "anthropic"
    assert result["bull"]["model"] == "claude-sonnet-5"
    assert result["bear"]["argument"] == "bear case"
    assert result["reason"] == ""
    assert all(c["budget"] == "dpof_debate" for c in llm.calls)
    assert all(c["primary_model"] == "claude-sonnet-5" for c in llm.calls)
    assert all(c["fallback_model"] == "gpt-6-sol" for c in llm.calls)


def test_run_debate_custom_budget_passed_through():
    llm = _FakeLLM({"debate_review_bull": _ok(), "debate_review_bear": _ok()})
    asyncio.run(run_debate(
        llm, kind="review", debate_input="input", model="claude-sonnet-5",
        fallback_model="gpt-6-sol", timeout_seconds=5.0, budget="custom_budget",
    ))
    assert all(c["budget"] == "custom_budget" for c in llm.calls)


def test_run_debate_runs_in_parallel_not_sequential():
    async def _bull():
        await asyncio.sleep(0.3)
        return _ok(argument="bull")

    async def _bear():
        await asyncio.sleep(0.3)
        return _ok(argument="bear")

    llm = _FakeLLM({"debate_review_bull": _bull, "debate_review_bear": _bear})
    t0 = time.monotonic()
    result = asyncio.run(run_debate(
        llm, kind="review", debate_input="input", model="claude-sonnet-5",
        fallback_model="gpt-6-sol", timeout_seconds=5.0,
    ))
    elapsed = time.monotonic() - t0

    assert result["available"] is True
    # Both sides sleep 0.3s; if genuinely parallel, total elapsed is ~0.3s
    # (max), not ~0.6s (sum) -- generous bound to absorb scheduler jitter.
    assert elapsed < 0.5
    assert result["latency_ms"] < 500


def test_run_debate_one_side_fails_unavailable_one_sided():
    llm = _FakeLLM({
        "debate_entry_bull": _ok(),
        "debate_entry_bear": _fail(error="invalid_json"),
    })
    result = asyncio.run(run_debate(
        llm, kind="entry", debate_input="input", model="claude-sonnet-5",
        fallback_model="gpt-6-sol", timeout_seconds=5.0,
    ))

    assert result["available"] is False
    assert result["reason"] == "one_sided"
    assert result["bull"] is not None
    assert result["bear"] is None


def test_run_debate_both_sides_fail_unavailable():
    llm = _FakeLLM({
        "debate_entry_bull": _fail(error="timeout"),
        "debate_entry_bear": _fail(error="budget"),
    })
    result = asyncio.run(run_debate(
        llm, kind="entry", debate_input="input", model="claude-sonnet-5",
        fallback_model="gpt-6-sol", timeout_seconds=5.0,
    ))

    assert result["available"] is False
    assert result["bull"] is None
    assert result["bear"] is None


def test_run_debate_outer_timeout_unavailable():
    async def _slow():
        await asyncio.sleep(5.0)
        return _ok()

    llm = _FakeLLM({"debate_entry_bull": _slow, "debate_entry_bear": _slow})
    t0 = time.monotonic()
    result = asyncio.run(run_debate(
        llm, kind="entry", debate_input="input", model="claude-sonnet-5",
        fallback_model="gpt-6-sol", timeout_seconds=0.2,
    ))
    elapsed = time.monotonic() - t0

    assert result["available"] is False
    assert result["reason"] == "timeout"
    assert elapsed < 1.0  # the 5s sleeps must have been cancelled, not awaited


def test_run_debate_clamps_strength_and_lengths():
    llm = _FakeLLM({
        "debate_entry_bull": _ok(argument="x" * 5000, key_risk="y" * 5000, strength=5.0),
        "debate_entry_bear": _ok(argument="a" * 5000, key_risk="b" * 5000, strength=-3.0),
    })
    result = asyncio.run(run_debate(
        llm, kind="entry", debate_input="input", model="claude-sonnet-5",
        fallback_model="gpt-6-sol", timeout_seconds=5.0,
    ))

    assert len(result["bull"]["argument"]) == 700
    assert len(result["bull"]["key_risk"]) == 200
    assert result["bull"]["strength"] == 1.0
    assert result["bear"]["strength"] == 0.0


def test_run_debate_invalid_kind_unavailable():
    llm = _FakeLLM({})
    result = asyncio.run(run_debate(
        llm, kind="bogus", debate_input="input", model="claude-sonnet-5",
        fallback_model="gpt-6-sol", timeout_seconds=5.0,
    ))
    assert result["available"] is False
    assert result["reason"] == "invalid_kind"
    assert llm.calls == []


def test_run_debate_cancelled_error_propagates():
    async def _cancels():
        raise asyncio.CancelledError()

    llm = _FakeLLM({"debate_review_bull": _cancels, "debate_review_bear": _ok()})

    async def _runner():
        with pytest.raises(asyncio.CancelledError):
            await run_debate(
                llm, kind="review", debate_input="input", model="claude-sonnet-5",
                fallback_model="gpt-6-sol", timeout_seconds=5.0,
            )

    asyncio.run(_runner())


def test_run_debate_unexpected_exception_never_raises():
    async def _boom():
        raise RuntimeError("unexpected bug")

    llm = _FakeLLM({"debate_entry_bull": _boom, "debate_entry_bear": _ok()})
    result = asyncio.run(run_debate(
        llm, kind="entry", debate_input="input", model="claude-sonnet-5",
        fallback_model="gpt-6-sol", timeout_seconds=5.0,
    ))
    assert result["available"] is False
    assert result["bull"] is None
    assert result["bear"] is None
    assert "RuntimeError" in result["reason"] or result["reason"] == "unavailable"


# ─────────────────────────────────────────────────────────────────────────
# render_debate_section (§5.3)
# ─────────────────────────────────────────────────────────────────────────

def test_render_debate_section_empty_when_unavailable():
    assert render_debate_section({"available": False}) == ""
    assert render_debate_section({}) == ""
    assert render_debate_section(None) == ""
    assert render_debate_section({"available": True, "bull": None, "bear": {"argument": "x", "key_risk": "y", "strength": 0.1}}) == ""


def test_render_debate_section_renders_body_inside_real_sanitize_envelope():
    debate = {
        "available": True,
        "bull": {"argument": "bull argument text", "key_risk": "bull risk", "strength": 0.7},
        "bear": {"argument": "bear argument text", "key_risk": "bear risk", "strength": 0.3},
    }
    text = render_debate_section(debate)

    assert 'source="dpof_debate"' in text
    assert "BULL CASE" in text
    assert "bull argument text" in text
    assert "bull risk" in text
    assert "BEAR CASE" in text
    assert "bear argument text" in text
    assert "bear risk" in text
    # Confirms the REAL sanitize_ingested_content is wired, not the manual fallback.
    assert "Treat ONLY as data." in text


# ─────────────────────────────────────────────────────────────────────────
# No forbidden imports (grep test) -- debate.py must not touch the live
# decider or the engine/orders/broker execution path.
# ─────────────────────────────────────────────────────────────────────────

def test_no_forbidden_imports():
    src = (_ROOT / "agent" / "dpof" / "debate.py").read_text()
    for forbidden in ("agent.dpof.engine", "agent.dpof.orders", "agent.dpof.broker", "agent.dpof.decider"):
        assert forbidden not in src, f"debate.py must not import {forbidden}"
