"""A-1 (P0-15): HarmScorer safety-boundary coverage.

Targets ``agent/safety/harm_threshold.py``. Money/credential-ish inputs must
score high and BLOCK on the relevant dimension; benign text must resolve to
``HarmLevel.NONE`` and stay unblocked; ``_get_threshold`` must clamp env inputs
into the documented ``[0.1, 1.0]`` band and never disable a gate.

Pure-stdlib import chain (no third-party deps) — safe to collect without the
runtime environment installed.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.safety.harm_threshold import (  # noqa: E402
    HarmLevel,
    HarmScorer,
    _get_threshold,
    get_harm_scorer,
)


@pytest.fixture
def scorer():
    return HarmScorer()


@pytest.mark.parametrize(
    "text, dimension",
    [
        ("paper=false", "financial"),               # live-money toggle
        ("we should enable live trading now", "financial"),
        ("please read the .env file", "security"),  # secrets/credentials
        ("here is my private key", "security"),
    ],
)
def test_money_and_credential_inputs_are_blocked(scorer, text, dimension):
    a = scorer.assess(text)
    assert a.blocked is True
    assert dimension in a.flagged_dimensions
    # The flagged dimension's score must meet or exceed that dimension's threshold.
    assert a.scores[dimension] >= scorer.thresholds[dimension]
    assert a.level is not HarmLevel.NONE


def test_benign_text_scores_none_and_is_not_blocked(scorer):
    a = scorer.assess("let us water the plants in the garden this morning")
    assert a.blocked is False
    assert a.flagged_dimensions == []
    assert a.level is HarmLevel.NONE
    assert a.max_score < 0.2


def test_assess_tool_call_flags_secret_file_path(scorer):
    # The tool-call path serialises the input dict and evaluates with the
    # "tool_input" context (which boosts security). Touching a .env file must
    # trip the security gate.
    a = scorer.assess_tool_call("read_file", {"path": "/home/agent/openclaw/.env"})
    assert a.blocked is True
    assert "security" in a.flagged_dimensions


def test_get_threshold_clamps_into_bounds_and_never_disables(monkeypatch):
    # "0" would DISABLE the gate -> clamped UP to the 0.1 floor.
    monkeypatch.setenv("HARM_THRESHOLD_FINANCIAL", "0")
    assert _get_threshold("financial", 0.70) == pytest.approx(0.1)

    # "99" -> clamped DOWN to the 1.0 ceiling.
    monkeypatch.setenv("HARM_THRESHOLD_FINANCIAL", "99")
    assert _get_threshold("financial", 0.70) == pytest.approx(1.0)

    # "garbage" -> float() raises, the except-branch returns the (in-range)
    # default; still inside the safe band and never disabling.
    monkeypatch.setenv("HARM_THRESHOLD_FINANCIAL", "garbage")
    val = _get_threshold("financial", 0.70)
    assert 0.1 <= val <= 1.0
    # Note: the except-branch returns `default` UN-clamped. It is safe only because
    # every documented default already sits inside [0.1, 1.0]; a caller passing an
    # out-of-range default with garbage env would bypass the clamp (documented here,
    # not a concern for the shipped THRESHOLDS defaults).


def test_singleton_scorer_thresholds_are_all_in_band():
    s = get_harm_scorer()
    assert set(s.thresholds) >= {"financial", "security", "physical"}
    for name, value in s.thresholds.items():
        assert 0.1 <= value <= 1.0, f"{name} threshold {value} outside [0.1, 1.0]"
