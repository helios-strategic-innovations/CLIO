"""Tests for agent.dpof.readmodel's learning-loop F1 readers/formatters
(Task ORCH-7Q4K-DPOF builder B2, 2026-09-26): `lessons_text`, and the EOD
digest's "Lessons today" section. Also covers `/dpof lessons`
(agent.dpof.telegram_commands) since it's a thin executor wrapper around
`lessons_text`.

Plugin-free (no pytest-asyncio), matching tests/dpof/test_readmodel_dpof_why.py.
Writes raw JSONL fixture files directly (not through LessonStore), so these
tests pin the exact ON-DISK shape readmodel.py must keep reading.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types

import pytest

from agent.dpof import readmodel


def _write_lessons(ddir: str, records: list[dict]) -> None:
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "lessons.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_day_stats(ddir: str, stats: dict) -> None:
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "signal_day_stats.json"), "w") as f:
        json.dump(stats, f)


def _lesson(pid, underlying, pnl_usd, pnl_pct, closed_at, key_factor="theta", lesson="a lesson"):
    return {
        "position_id": pid, "underlying": underlying, "direction": "bullish",
        "closed_at": closed_at, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
        "exit_category": "decider_exit", "key_factor": key_factor, "lesson": lesson,
    }


# ─────────────────────────────────────────────────────────────────────────
# lessons_text
# ─────────────────────────────────────────────────────────────────────────

def test_lessons_text_no_data(tmp_path):
    text = readmodel.lessons_text(ddir=str(tmp_path))
    assert "none recorded yet" in text


def test_lessons_text_shows_last_n_newest_first(tmp_path):
    records = [
        _lesson(f"p{i}", "SPY", 10.0 * i, 0.1 * i, f"2026-09-2{i}T10:00:00-04:00", lesson=f"lesson-{i}")
        for i in range(5)
    ]
    _write_lessons(str(tmp_path), records)
    text = readmodel.lessons_text(limit=3, ddir=str(tmp_path))
    assert "showing 3 of 5" in text
    # newest (lesson-4) appears before oldest-of-the-shown (lesson-2)
    assert text.index("lesson-4") < text.index("lesson-3") < text.index("lesson-2")
    assert "lesson-0" not in text
    assert "lesson-1" not in text


def test_lessons_text_fields_rendered(tmp_path):
    _write_lessons(str(tmp_path), [_lesson("p1", "SPY", -120.0, -0.38, "2026-09-20T14:30:00-04:00",
                                            key_factor="theta", lesson="theta ate it")])
    text = readmodel.lessons_text(ddir=str(tmp_path))
    assert "SPY bullish" in text
    assert "-$120" in text
    assert "-38%" in text
    assert "key:theta" in text
    assert "theta ate it" in text


def test_lessons_text_tally_rendered(tmp_path):
    records = [
        _lesson("w1", "SPY", 100.0, 0.3, "2026-09-20T10:00:00-04:00"),
        _lesson("l1", "SPY", -40.0, -0.13, "2026-09-21T10:00:00-04:00"),
        _lesson("w2", "QQQ", 20.0, 0.05, "2026-09-21T11:00:00-04:00"),
    ]
    _write_lessons(str(tmp_path), records)
    text = readmodel.lessons_text(ddir=str(tmp_path))
    assert "SPY: n=2 wins=1 losses=1 net=$+60" in text
    assert "QQQ: n=1 wins=1 losses=0 net=$+20" in text


def test_lessons_text_never_raises_on_corrupt_file(tmp_path):
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(os.path.join(str(tmp_path), "lessons.jsonl"), "w") as f:
        f.write("not json\n{also not json\n")
    text = readmodel.lessons_text(ddir=str(tmp_path))
    assert isinstance(text, str)


def test_lessons_text_limit_zero_shows_none_but_tally_present(tmp_path):
    _write_lessons(str(tmp_path), [_lesson("p1", "SPY", 10.0, 0.1, "2026-09-20T10:00:00-04:00")])
    text = readmodel.lessons_text(limit=0, ddir=str(tmp_path))
    assert "showing 0 of 1" in text
    assert "Tally" in text


def test_lessons_text_malformed_pnl_never_raises(tmp_path):
    _write_lessons(str(tmp_path), [{"position_id": "p1", "underlying": "SPY", "pnl_usd": "not_a_number"}])
    text = readmodel.lessons_text(ddir=str(tmp_path))
    assert "n/a" in text


# ─────────────────────────────────────────────────────────────────────────
# EOD digest "Lessons today" section
# ─────────────────────────────────────────────────────────────────────────

def test_eod_digest_no_lessons_section_when_no_lessons_file(tmp_path):
    _write_day_stats(str(tmp_path), {"day": "2026-09-25", "cycles_evaluated": 10, "entries": 0, "underlyings": {}})
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")
    assert "Lessons today" not in text


def test_eod_digest_no_lessons_section_when_lessons_are_from_a_different_day(tmp_path):
    _write_day_stats(str(tmp_path), {"day": "2026-09-25", "cycles_evaluated": 10, "entries": 0, "underlyings": {}})
    _write_lessons(str(tmp_path), [_lesson("p1", "SPY", 10.0, 0.1, "2026-09-24T10:00:00-04:00")])
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")
    assert "Lessons today" not in text


def test_eod_digest_includes_lessons_today_section(tmp_path):
    _write_day_stats(str(tmp_path), {"day": "2026-09-25", "cycles_evaluated": 10, "entries": 1, "underlyings": {}})
    _write_lessons(str(tmp_path), [
        _lesson("p1", "SPY", -120.0, -0.38, "2026-09-25T14:30:00-04:00", lesson="theta ate it"),
        _lesson("p2", "QQQ", 50.0, 0.2, "2026-09-25T15:00:00-04:00", lesson="flow paid off"),
        _lesson("p3", "SPY", 10.0, 0.05, "2026-09-24T15:00:00-04:00", lesson="not today"),
    ])
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")
    assert "Lessons today: 2" in text
    assert "theta ate it" in text
    assert "flow paid off" in text
    assert "not today" not in text


def test_eod_digest_lessons_truncated_to_160_chars(tmp_path):
    _write_day_stats(str(tmp_path), {"day": "2026-09-25", "cycles_evaluated": 10, "entries": 1, "underlyings": {}})
    long_lesson = "z" * 500
    _write_lessons(str(tmp_path), [_lesson("p1", "SPY", 10.0, 0.1, "2026-09-25T10:00:00-04:00", lesson=long_lesson)])
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")
    lesson_lines = [ln for ln in text.splitlines() if "z" in ln]
    assert lesson_lines
    assert all(len(ln) <= 160 for ln in lesson_lines)


def test_eod_digest_no_lessons_section_never_raises_when_lessons_file_corrupt(tmp_path):
    _write_day_stats(str(tmp_path), {"day": "2026-09-25", "cycles_evaluated": 10, "entries": 1, "underlyings": {}})
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(os.path.join(str(tmp_path), "lessons.jsonl"), "w") as f:
        f.write("not json at all\n")
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")  # must not raise
    assert "DPOF EOD digest" in text


def test_eod_digest_full_shape_test_unaffected_by_lessons_absence(tmp_path):
    """Guards the PRE-EXISTING full-shape assertions in
    tests/dpof/test_readmodel_dpof_why.py::test_eod_digest_text_full_shape --
    without a lessons.jsonl file, the digest must be unchanged."""
    _write_day_stats(str(tmp_path), {
        "day": "2026-09-25", "cycles_evaluated": 780, "nominations": 6,
        "decider_pass": 5, "decider_enter": 1, "decider_unavailable": 0,
        "jev_vetoes": 0, "entries": 1,
        "underlyings": {"SPY": {
            "max_abs_composite": 0.34, "max_abs_composite_at": "2026-09-25T10:15:00-04:00",
            "max_abs_composite_direction": "bullish",
            "outcome_counts": {"below_threshold": 600}, "pillar_available": {}, "pillar_total": {},
            "flow_source_counts": {},
        }},
    })
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")
    assert "Lessons today" not in text
    assert "DPOF EOD digest — 2026-09-25" in text


# ─────────────────────────────────────────────────────────────────────────
# /dpof lessons Telegram subcommand
# ─────────────────────────────────────────────────────────────────────────

class FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class FakeUpdate:
    def __init__(self, user_id):
        self.effective_user = types.SimpleNamespace(id=user_id) if user_id is not None else None
        self.message = FakeMessage()


class FakeContext:
    def __init__(self, args=None):
        self.args = args or []


def _patch_telegram_bot_module(monkeypatch, authorized_ids):
    fake_module = types.ModuleType("agent.integrations.telegram_bot")
    fake_module.is_authorized = lambda uid: uid in authorized_ids
    monkeypatch.setitem(sys.modules, "agent.integrations.telegram_bot", fake_module)
    import agent.integrations as integrations_pkg
    monkeypatch.setattr(integrations_pkg, "telegram_bot", fake_module, raising=False)


def _patch_get_engine(monkeypatch, engine):
    fake_module = types.ModuleType("agent.dpof.engine")
    fake_module.get_engine = lambda: engine
    monkeypatch.setitem(sys.modules, "agent.dpof.engine", fake_module)
    import agent.dpof as dpof_pkg
    monkeypatch.setattr(dpof_pkg, "engine", fake_module, raising=False)


def test_dpof_lessons_subcommand_reads_readmodel_off_loop(monkeypatch):
    import threading

    import agent.dpof.telegram_commands as tc_mod

    _patch_telegram_bot_module(monkeypatch, {42})
    _patch_get_engine(monkeypatch, object())  # never touched by "lessons"
    seen = {}

    def _fake_lessons_text():
        seen["thread"] = threading.current_thread() is threading.main_thread()
        return "LESSONS TEXT"

    monkeypatch.setattr(readmodel, "lessons_text", _fake_lessons_text)
    update = FakeUpdate(42)
    asyncio.run(tc_mod.dpof_command(update, FakeContext(["lessons"])))
    assert update.message.replies == ["LESSONS TEXT"]
    assert seen["thread"] is False


def test_dpof_lessons_unauthorized_is_silent(monkeypatch):
    import agent.dpof.telegram_commands as tc_mod

    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})
    _patch_get_engine(monkeypatch, object())
    update = FakeUpdate(user_id=999)
    asyncio.run(tc_mod.dpof_command(update, FakeContext(args=["lessons"])))
    assert update.message.replies == []


def test_usage_string_mentions_lessons():
    import agent.dpof.telegram_commands as tc_mod

    assert "lessons" in tc_mod._USAGE


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(pytest.main([__file__, "-v"]))


def test_position_reviews_reads_whole_file_not_a_tail(tmp_path):
    """A backlogged reflection needs ALL of a position's review rows, even when
    later decisions push them far beyond a 256 KB tail."""
    from agent.dpof import readmodel

    d = tmp_path / "dpof"
    d.mkdir()
    path = d / "decisions.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "position_review", "position_id": "OLD", "action": "hold", "ts": 1}) + "\n")
        f.write(json.dumps({"kind": "entry_deliberation", "underlying": "SPY", "ts": 2}) + "\n")
        filler = {"kind": "position_review", "position_id": "OTHER", "reason": "z" * 900}
        for i in range(600):  # ~560 KB of other positions' rows after it
            f.write(json.dumps({**filler, "ts": 10 + i}) + "\n")
        f.write(json.dumps({"kind": "position_review", "position_id": "OLD", "action": "exit", "ts": 9999}) + "\n")
    assert os.path.getsize(path) > 512 * 1024
    rows = readmodel.position_reviews("OLD", ddir=str(d))
    assert [r["action"] for r in rows] == ["hold", "exit"]  # oldest-first, both found
    assert readmodel.position_reviews("MISSING", ddir=str(d)) == []
    assert readmodel.position_reviews("OLD", ddir=str(tmp_path / "nope")) == []
