"""The entry decider's thesis is persisted on every PASS row of decisions.jsonl (sanitized, single-spaced, <= 1,200 chars)."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import timezone
from types import SimpleNamespace

from agent.dpof.decider import EntryDecision, thesis_full
from agent.dpof.ledger import DPOFLedger
from tests.dpof.test_engine_judgment import _base_dt, _engine


def _run(cfg, decision, *, age_seconds=0.0):
    """Drive the real _apply_entry_outcome for SPY with a real ledger; return the decisions.jsonl row it wrote."""
    async def go():
        ledger = DPOFLedger(cfg.data_dir)
        await ledger.load()
        eng = _engine(cfg, ledger=ledger)
        now_et = _base_dt()
        now = now_et.astimezone(timezone.utc).timestamp()
        eng._entry_task_meta["SPY"] = {"direction": "bullish", "launched_at": now - 5}
        outcome = {"ok": True, "result": decision, "finished": now - age_seconds}
        await eng._apply_entry_outcome("SPY", outcome, now, now_et, cfg, None, {}, None)

    asyncio.run(go())
    with open(os.path.join(cfg.data_dir, "decisions.jsonl")) as f:
        return json.loads(f.read().strip().splitlines()[-1])


def test_pass_row_carries_the_sanitized_thesis(cfg):
    """A PASS row records the model's thesis as one printable line next to reasons ['pass']."""
    row = _run(cfg, EntryDecision(action="pass", thesis="flow is mixed; wait for VWAP", provider="anthropic", model="m"))
    assert row["reasons"] == ["pass"] and row["action"] == "pass"
    assert row["thesis"] == "flow is mixed; wait for VWAP"


def test_hostile_thesis_is_sanitized_and_kept_whole_under_the_cap(cfg):
    """Newlines, control bytes and a 1,000-char thesis are collapsed to one printable line of <= 1,200 chars (not cut at 300)."""
    raw = "line one\nline two\r\n\tx\x1b[31m red \x00" + "a" * 1000
    row = _run(cfg, EntryDecision(action="pass", thesis=raw))
    assert "\n" not in row["thesis"] and "\x1b" not in row["thesis"] and "\x00" not in row["thesis"]
    assert len(row["thesis"]) <= 1200 and not row["thesis"].endswith("…")
    assert row["thesis"] == thesis_full(raw)


def test_a_very_long_thesis_is_cut_at_a_sentence_boundary_in_the_row(cfg):
    """A 3,000-char thesis is persisted as <= 1,200 chars ending at a sentence boundary, never mid-word."""
    raw = "Flow is mixed and VWAP is contested. " * 100
    row = _run(cfg, EntryDecision(action="pass", thesis=raw))
    assert len(row["thesis"]) <= 1200
    assert row["thesis"].endswith("contested.")


def test_an_empty_thesis_adds_no_key(cfg):
    """No thesis means the row is exactly what it was before this change (no 'thesis' key)."""
    row = _run(cfg, EntryDecision(action="pass", thesis=""))
    assert "thesis" not in row and row["reasons"] == ["pass"]


def test_a_decision_object_without_thesis_line_is_unaffected(cfg):
    """Any other decision shape (no thesis_line attribute) still logs a normal pass row with no error and no thesis key."""
    row = _run(cfg, SimpleNamespace(action="pass", recheck_minutes=5, provider="p", model="m"))
    assert row["reasons"] == ["pass"] and "thesis" not in row


def test_a_stale_decision_does_not_persist_a_thesis(cfg):
    """The stale-decision early return happens before the pass branch, so it never carries a thesis."""
    row = _run(cfg, EntryDecision(action="pass", thesis="should not be stored"), age_seconds=10_000.0)
    assert row["reasons"] == ["stale_decision"] and "thesis" not in row


def test_thesis_line_property_matches_the_sanitizer():
    """EntryDecision.thesis_line is exactly thesis_full(thesis): 1,200-char bound, no 300-char cut."""
    for t in ("", "plain", "a\nb", "z" * 301, "w" * 1300):
        assert EntryDecision(action="pass", thesis=t).thesis_line == thesis_full(t)
    assert EntryDecision(action="pass", thesis="z" * 301).thesis_line == "z" * 301


def test_thesis_full_short_text_is_unchanged_and_whitespace_collapsed():
    assert thesis_full("short thesis") == "short thesis"
    assert thesis_full("a\n\n  b\tc") == "a b c"
    assert thesis_full(None) == ""


def test_thesis_full_cuts_at_the_last_sentence_boundary_that_keeps_40_percent():
    """A sentence boundary past 40% of the cap is used; the sentence ends cleanly with no ellipsis."""
    head = "Alpha beta gamma delta. " * 30            # sentence ends well past 480 chars
    out = thesis_full(head + "tail " * 400)
    assert len(out) <= 1200 and out.endswith("delta.") and "…" not in out


def test_thesis_full_falls_back_to_the_last_space_with_an_ellipsis_never_mid_word():
    """With no usable sentence boundary, the cut lands on a space and ends in an ellipsis; the last word is whole."""
    words = " ".join(f"word{i:04d}" for i in range(400))   # ~3,000 chars, no sentence boundary
    out = thesis_full(words)
    assert len(out) <= 1200 and out.endswith("…")
    last = out[:-1].rsplit(" ", 1)[-1]
    assert last.startswith("word") and len(last) == 8
