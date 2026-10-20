"""Tests for agent.dpof.readmodel's observability readers/formatters
(Task ORCH-7Q4K-DPOF): signal_day_stats, recent_signal_log,
_recent_outcome_changes, dpof_why_text, eod_digest_text.

Plugin-free (no pytest-asyncio), matching the rest of tests/dpof/. Writes
raw JSON/JSONL fixture files directly rather than going through SignalLog,
so these tests pin the exact ON-DISK shape readmodel.py must keep reading.
"""

from __future__ import annotations

import json
import os

from agent.dpof import readmodel


def _write_signals(ddir: str, records: list[dict]) -> None:
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "signals.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_day_stats(ddir: str, stats: dict) -> None:
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "signal_day_stats.json"), "w") as f:
        json.dump(stats, f)


def _rec(ts, underlying, outcome, composite=0.12, strength="none", flow_source="uw", pillars=None):
    return {
        "ts": ts, "underlying": underlying, "outcome": outcome, "composite": composite,
        "strength": strength, "flow_source": flow_source,
        "pillars": pillars or {
            "nofa": {"value": composite, "available": True, "source": "uw:x"},
            "premium": {"value": None, "available": False, "source": ""},
            "darkpool": {"value": None, "available": False, "source": ""},
            "gex": {"value": None, "available": False, "source": ""},
        },
    }


# ─────────────────────────────────────────────────────────────────────────
# signal_day_stats / recent_signal_log
# ─────────────────────────────────────────────────────────────────────────

def test_signal_day_stats_missing_file_returns_empty(tmp_path):
    assert readmodel.signal_day_stats(str(tmp_path)) == {}


def test_signal_day_stats_reads_written_file(tmp_path):
    _write_day_stats(str(tmp_path), {"day": "2026-09-25", "cycles_evaluated": 5})
    stats = readmodel.signal_day_stats(str(tmp_path))
    assert stats["day"] == "2026-09-25"
    assert stats["cycles_evaluated"] == 5


def test_recent_signal_log_empty_when_no_file(tmp_path):
    assert readmodel.recent_signal_log(ddir=str(tmp_path)) == []


def test_recent_signal_log_respects_limit_newest_last(tmp_path):
    records = [_rec(f"2026-09-25T10:0{i}:00-04:00", "SPY", "below_threshold") for i in range(5)]
    _write_signals(str(tmp_path), records)
    out = readmodel.recent_signal_log(limit=2, ddir=str(tmp_path))
    assert len(out) == 2
    assert out[-1]["ts"] == records[-1]["ts"]


# ─────────────────────────────────────────────────────────────────────────
# _latest_per_underlying / _recent_outcome_changes
# ─────────────────────────────────────────────────────────────────────────

def test_latest_per_underlying_last_write_wins(tmp_path):
    records = [
        _rec("2026-09-25T10:00:00-04:00", "SPY", "below_threshold"),
        _rec("2026-09-25T10:00:30-04:00", "QQQ", "cooldown"),
        _rec("2026-09-25T10:01:00-04:00", "SPY", "deliberating"),
    ]
    latest = readmodel._latest_per_underlying(records)
    assert latest["SPY"]["outcome"] == "deliberating"
    assert latest["QQQ"]["outcome"] == "cooldown"


def test_recent_outcome_changes_only_records_transitions():
    records = [
        _rec("2026-09-25T10:00:00-04:00", "SPY", "below_threshold"),
        _rec("2026-09-25T10:00:30-04:00", "SPY", "below_threshold"),  # unchanged -- not a transition
        _rec("2026-09-25T10:01:00-04:00", "SPY", "cooldown"),         # changed
        _rec("2026-09-25T10:01:30-04:00", "SPY", "cooldown"),         # unchanged
        _rec("2026-09-25T10:02:00-04:00", "SPY", "deliberating"),     # changed
    ]
    changes = readmodel._recent_outcome_changes(records, limit=5)
    assert len(changes) == 2
    assert changes[0] == {"ts": "2026-09-25T10:01:00-04:00", "underlying": "SPY", "from": "below_threshold", "to": "cooldown"}
    assert changes[1] == {"ts": "2026-09-25T10:02:00-04:00", "underlying": "SPY", "from": "cooldown", "to": "deliberating"}


def test_recent_outcome_changes_respects_limit_keeps_most_recent():
    records = [_rec(f"t{i}", "SPY", f"o{i}") for i in range(10)]  # every record is a change
    changes = readmodel._recent_outcome_changes(records, limit=5)
    assert len(changes) == 5
    assert changes[-1]["to"] == "o9"


def test_recent_outcome_changes_independent_per_underlying():
    records = [
        _rec("t0", "SPY", "below_threshold"),
        _rec("t1", "QQQ", "cooldown"),
        _rec("t2", "SPY", "deliberating"),  # SPY change
        _rec("t3", "QQQ", "cooldown"),      # QQQ unchanged
    ]
    changes = readmodel._recent_outcome_changes(records, limit=5)
    assert len(changes) == 1
    assert changes[0]["underlying"] == "SPY"


# ─────────────────────────────────────────────────────────────────────────
# dpof_why_text
# ─────────────────────────────────────────────────────────────────────────

def test_dpof_why_text_no_data_yet(tmp_path):
    text = readmodel.dpof_why_text(ddir=str(tmp_path), today="2026-09-25")
    assert "no signal data recorded yet" in text
    assert "no signal-log records yet" in text


def test_dpof_why_text_includes_day_stats_and_current_outcomes(tmp_path):
    _write_day_stats(str(tmp_path), {
        "day": "2026-09-25", "cycles_evaluated": 40, "nominations": 2,
        "decider_enter": 0, "decider_pass": 2, "decider_unavailable": 0,
        "jev_vetoes": 0, "entries": 0, "underlyings": {},
    })
    _write_signals(str(tmp_path), [
        _rec("2026-09-25T14:00:00-04:00", "SPY", "below_threshold", composite=0.12),
        _rec("2026-09-25T14:00:00-04:00", "QQQ", "not_selected_instrument", composite=0.05),
    ])
    text = readmodel.dpof_why_text(ddir=str(tmp_path), today="2026-09-25")
    assert "2026-09-25" in text
    assert "40 cycles evaluated" in text
    assert "2 nomination(s)" in text
    assert "SPY: below_threshold" in text
    assert "QQQ: not_selected_instrument" in text
    assert "composite +0.12" in text


def test_dpof_why_text_shows_recent_outcome_changes(tmp_path):
    _write_signals(str(tmp_path), [
        _rec("2026-09-25T13:40:05-04:00", "SPY", "deliberating"),
        _rec("2026-09-25T13:41:05-04:00", "SPY", "decider_pass"),
    ])
    text = readmodel.dpof_why_text(ddir=str(tmp_path), today="2026-09-25")
    assert "Recent outcome changes:" in text
    assert "deliberating -> decider_pass" in text


def test_dpof_why_text_never_raises_on_corrupt_files(tmp_path):
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(os.path.join(str(tmp_path), "signal_day_stats.json"), "w") as f:
        f.write("{not json")
    with open(os.path.join(str(tmp_path), "signals.jsonl"), "w") as f:
        f.write("not json either\n")
    text = readmodel.dpof_why_text(ddir=str(tmp_path), today="2026-09-25")
    assert isinstance(text, str) and text


# ─────────────────────────────────────────────────────────────────────────
# eod_digest_text
# ─────────────────────────────────────────────────────────────────────────

def test_eod_digest_text_no_data(tmp_path):
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")
    assert "no signal cycles were recorded today" in text


def test_eod_digest_text_full_shape(tmp_path):
    _write_day_stats(str(tmp_path), {
        "day": "2026-09-25",
        "cycles_evaluated": 780,
        "nominations": 6,
        "decider_pass": 5,
        "decider_enter": 1,
        "decider_unavailable": 0,
        "jev_vetoes": 0,
        "entries": 1,
        "underlyings": {
            "SPY": {
                "max_abs_composite": 0.34,
                "max_abs_composite_at": "2026-09-25T10:15:00-04:00",
                "max_abs_composite_direction": "bullish",
                "outcome_counts": {"below_threshold": 600, "cooldown": 150, "entered": 1},
                "pillar_available": {"nofa": 780, "premium": 400, "darkpool": 780, "gex": 0},
                "pillar_total": {"nofa": 780, "premium": 780, "darkpool": 780, "gex": 780},
                "flow_source_counts": {"uw": 700, "selfflow": 80},
            },
        },
    })
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")
    assert "DPOF EOD digest — 2026-09-25" in text
    assert "Entries: 1" in text
    assert "Nominations: 6" in text
    assert "enter=1 pass=5" in text
    assert "SPY: max |composite| +0.34 (bullish)" in text
    assert "top outcomes:" in text
    assert "below_threshold=600" in text
    assert "gex 0%" in text  # 0/780 -> 0%, never a ZeroDivisionError
    assert "nofa 100%" in text
    assert "flow source: selfflow=80, uw=700" in text


def test_eod_digest_text_zero_total_pillar_no_div_by_zero(tmp_path):
    _write_day_stats(str(tmp_path), {
        "day": "2026-09-25", "underlyings": {
            "SPY": {
                "pillar_available": {}, "pillar_total": {}, "outcome_counts": {}, "flow_source_counts": {},
            }
        },
    })
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")
    assert "nofa 0%" in text


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))


def test_eod_digest_flags_a_day_with_no_cycles(tmp_path):
    """Stats left over from an earlier day must not be reported as today's:
    the digest says DPOF didn't evaluate the market today."""
    _write_day_stats(str(tmp_path), {"day": "2026-09-24", "cycles_evaluated": 780, "entries": 2})
    text = readmodel.eod_digest_text(ddir=str(tmp_path), today="2026-09-25")
    assert "2026-09-25" in text
    assert "no signal cycles were recorded today" in text
    assert "last recorded day: 2026-09-24" in text
    assert "Entries: 2" not in text


def test_dpof_why_labels_stale_day(tmp_path):
    _write_day_stats(str(tmp_path), {"day": "2026-09-24", "cycles_evaluated": 780})
    text = readmodel.dpof_why_text(ddir=str(tmp_path), today="2026-09-25")
    assert "Today (2026-09-25): no signal cycles recorded yet" in text
    assert "2026-09-24: 780 cycles evaluated" in text


def test_split_for_telegram():
    assert readmodel.split_for_telegram("short") == ["short"]
    text = "\n".join(f"line {i} " + "x" * 90 for i in range(200))  # ~20k chars
    chunks = readmodel.split_for_telegram(text, limit=4000)
    assert all(len(c) <= 4000 for c in chunks)
    assert "\n".join(chunks) == text
    long_line = "y" * 9000
    chunks = readmodel.split_for_telegram(long_line, limit=4000)
    assert all(len(c) <= 4000 for c in chunks) and "".join(chunks) == long_line
