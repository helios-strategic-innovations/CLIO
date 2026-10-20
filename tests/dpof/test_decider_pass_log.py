"""Tests for the INFO log line emitted on every successful entry PASS in agent.dpof.decider.

No network: provider calls go through injected fake transports (see test_decider.py).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from agent.dpof.decider import (
    DecisionUnavailable,
    DPOFDecider,
    EntryDecision,
    _one_line,
)
from tests.dpof.test_decider import (
    _candidates,
    _entry_answer,
    _entry_ctx,
    _fixed_transport,
    _result,
)

LOGGER_NAME = "clio.dpof.decider"
PASS_MARK = "entry decision PASS"


def _pass_records(caplog):
    return [r for r in caplog.records if r.name == LOGGER_NAME and PASS_MARK in r.getMessage()]


def _decide_pass(cfg, caplog, thesis="thesis text", conviction=0.4):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    answer = _entry_answer(action="pass", candidate_key="", conviction=conviction, qty=0,
                           thesis=thesis, recheck_minutes=7)
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(answer)))
    return asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))


def test_p1_pass_logs_one_info_record_with_thesis_conviction_model(cfg, caplog):
    """A PASS emits exactly one INFO record carrying thesis, conviction and model."""
    result = _decide_pass(cfg, caplog, thesis="flow is mixed, wait for VWAP")
    assert isinstance(result, EntryDecision) and result.action == "pass"
    records = _pass_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    msg = records[0].getMessage()
    assert "flow is mixed, wait for VWAP" in msg
    assert "conviction=0.4" in msg
    assert "model=claude-opus-5-5" in msg
    assert "provider=anthropic" in msg
    assert "candidates=1" in msg
    assert "recheck_min=7" in msg
    assert "nomination=SPY:bullish" in msg  # label comes from the offered candidates


def test_p1b_ctx_label_is_the_fallback_when_there_are_no_candidates(cfg, caplog):
    """With no candidates, identifying ctx keys name the nomination."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    answer = _entry_answer(action="pass", candidate_key="", qty=0, thesis="t")
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(answer)))
    ctx = dict(_entry_ctx(), underlying="SPY", direction="bullish")
    asyncio.run(decider.decide_entry(ctx, []))
    msg = _pass_records(caplog)[0].getMessage()
    assert "nomination=underlying=SPY|direction=bullish" in msg


def test_p2_enter_does_not_log_pass(cfg, caplog):
    """An ENTER decision emits no 'entry decision PASS' record."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer())))
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.action == "enter"
    assert _pass_records(caplog) == []


def test_p3_control_characters_logged_on_one_line(cfg, caplog):
    """Newlines, tabs and ANSI escape bytes in the thesis do not reach the log record."""
    thesis = "line one\nline two\r\n\tindented\x1b[31m red \x1b[0m\x00end"
    _decide_pass(cfg, caplog, thesis=thesis)
    records = _pass_records(caplog)
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "\n" not in msg and "\r" not in msg and "\t" not in msg
    assert all(ch.isprintable() or ch == " " for ch in msg)
    assert "line one line two" in msg


def test_p4_long_thesis_truncated_short_thesis_unchanged(cfg, caplog):
    """A 1000-char thesis is cut to <=300 chars ending in an ellipsis; a short one is untouched."""
    _decide_pass(cfg, caplog, thesis="a" * 1000)
    msg = _pass_records(caplog)[0].getMessage()
    logged_thesis = msg.split("thesis=", 1)[1]
    assert len(logged_thesis) <= 300
    assert logged_thesis.endswith("…")

    caplog.clear()
    _decide_pass(cfg, caplog, thesis="short thesis")
    msg = _pass_records(caplog)[0].getMessage()
    assert msg.endswith("thesis=short thesis")


def test_p5_format_specifiers_and_braces_logged_verbatim(cfg, caplog):
    """Thesis text with %s, %d and {} is logged literally and does not raise."""
    thesis = "odds 100%s and %d items, keep {} braces {x}"
    _decide_pass(cfg, caplog, thesis=thesis)
    msg = _pass_records(caplog)[0].getMessage()
    assert msg.endswith("thesis=" + thesis)


def test_p6_decider_unavailable_logs_no_pass_and_still_raises(cfg, caplog):
    """When both providers fail, no PASS record is written and DecisionUnavailable propagates."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    cfg_fb = cfg
    decider = DPOFDecider(
        cfg_fb,
        anthropic_transport=_fixed_transport(RuntimeError("anthropic down")),
    )
    with pytest.raises(DecisionUnavailable):
        asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert _pass_records(caplog) == []


def test_p7_one_line_unit_cases():
    """_one_line handles None, printable filtering, whitespace collapse and the exact limit boundary."""
    assert _one_line(None) == ""
    assert _one_line("  a\tb\n\nc  ") == "a b c"
    assert _one_line("x\x07y\x7fz") == "x y z"
    assert _one_line(42) == "42"
    exact = "b" * 300
    assert _one_line(exact) == exact
    over = "c" * 301
    out = _one_line(over)
    assert len(out) == 300 and out.endswith("…")
    assert out == "c" * 299 + "…"


def test_p8_label_is_unique_underlying_direction_pairs_from_candidates():
    """Duplicates collapse, order is kept, dict and object candidates both work, and at most 4 pairs are shown."""
    from agent.dpof.decider import EntryCandidate, _pass_label
    spy_bull = EntryCandidate(key="C1", underlying="SPY", direction="bullish")
    spy_bull_2 = EntryCandidate(key="C2", underlying="SPY", direction="bullish")
    qqq_bear = {"key": "C3", "underlying": "QQQ", "direction": "bearish"}
    assert _pass_label({}, [spy_bull, spy_bull_2, qqq_bear]) == "SPY:bullish,QQQ:bearish"
    many = [EntryCandidate(key=f"C{i}", underlying=f"U{i}", direction="bullish") for i in range(9)]
    assert _pass_label({}, many).count(",") == 3
    assert _pass_label({}, [EntryCandidate(key="C1")]) == "n/a"
    assert _pass_label({"underlying": "SPY"}, []) == "underlying=SPY"
    assert _pass_label(None, None) == "n/a"


def test_p9_label_is_sanitized_and_bounded():
    """Hostile candidate fields cannot inject newlines or unbounded text into the log line."""
    from agent.dpof.decider import _pass_label
    label = _pass_label({}, [{"underlying": "SP\nY" + "x" * 100, "direction": "bull\tish"}])
    assert "\n" not in label and "\t" not in label and len(label) <= 41


def test_p10_provider_supplied_model_and_provider_strings_cannot_forge_log_lines(cfg, caplog, monkeypatch):
    """A hostile model/provider string (newline + fake log line, 500 chars) is collapsed to one bounded line in the PASS record."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    evil_model = "gpt-x\n2026-10-10 15:00:00 INFO clio.dpof.engine DPOF entry FILLED " + "y" * 300
    answer = _entry_answer(action="pass", candidate_key="", qty=0, thesis="t")
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(answer, model=evil_model)))
    asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    msg = _pass_records(caplog)[0].getMessage()
    assert "\n" not in msg and "\r" not in msg
    model_part = msg.split("model=", 1)[1].split(" provider=", 1)[0]
    assert len(model_part) <= 80 and model_part.endswith("…")


def test_p11_hostile_provider_string_is_sanitised_and_bounded_too(cfg, caplog, monkeypatch):
    """The provider field is as provider-controlled as the model: newline, a fake log line and 400 chars collapse to one bounded line."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    hostile = "openai\n2026-10-10 15:00:00 INFO clio.dpof.engine DPOF entry FILLED " + "z" * 400

    async def fake_decide(self, kind, ctx, **kw):
        return EntryDecision(action="pass", conviction=0.2, thesis="t", model="m", provider=hostile)

    monkeypatch.setattr(DPOFDecider, "_decide", fake_decide)
    asyncio.run(DPOFDecider(cfg).decide_entry(_entry_ctx(), _candidates()))
    msg = _pass_records(caplog)[0].getMessage()
    assert "\n" not in msg and "\r" not in msg
    provider_part = msg.split(" provider=", 1)[1].split(" candidates=", 1)[0]
    assert len(provider_part) <= 80 and provider_part.endswith("…")
