"""Tests for agent.dpof.signal_log — observability log (Task ORCH-7Q4K-DPOF).

Covers: NearMissLogger rate limiting, flow_source_label, format_outcome_log_line,
fold_record_into_day_stats (pure), and SignalLog.write_cycle I/O (record shape,
line-count cap/trim, off-loop execution, day-stats rollover at ET midnight).

Plugin-free (no pytest-asyncio): async code driven with asyncio.run(), matching
the rest of this suite (see tests/dpof/conftest.py).
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from agent.dpof import signal_log as signal_log_mod
from agent.dpof.signal_log import (
    NearMissLogger,
    SignalLog,
    fold_record_into_day_stats,
    format_outcome_log_line,
    flow_source_label,
    signal_archive_enabled,
)


# ─────────────────────────────────────────────────────────────────────────
# flow_source_label
# ─────────────────────────────────────────────────────────────────────────

def test_flow_source_label_empty_is_unavailable():
    assert flow_source_label([]) == "unavailable"
    assert flow_source_label(["", None, ""]) == "unavailable"


def test_flow_source_label_all_uw():
    assert flow_source_label(["uw:/option-trades/flow-alerts", "uw:/gex"]) == "uw"


def test_flow_source_label_all_selfflow():
    assert flow_source_label(["selfflow:SPY", "selfflow:SPY"]) == "selfflow"


def test_flow_source_label_mixed():
    assert flow_source_label(["uw:/gex", "selfflow:SPY"]) == "mixed"


# ─────────────────────────────────────────────────────────────────────────
# format_outcome_log_line
# ─────────────────────────────────────────────────────────────────────────

def test_format_outcome_log_line_below_threshold():
    record = {
        "underlying": "SPY",
        "outcome": "below_threshold",
        "composite": 0.12,
        "pillars": {
            "nofa": {"value": 0.21, "available": True, "source": "uw:x"},
            "premium": {"value": 0.08, "available": True, "source": "uw:y"},
            "darkpool": {"value": -0.05, "available": True, "source": "selfflow:SPY"},
            "gex": {"value": None, "available": False, "source": ""},
        },
        "flow_source": "mixed",
    }
    line = format_outcome_log_line(record)
    assert line.startswith("DPOF SPY: no entry — below_threshold")
    assert "composite +0.12" in line
    assert "nofa +0.21" in line
    assert "premium +0.08" in line
    assert "darkpool -0.05" in line
    assert "gex n/a" in line
    assert "flow mixed" in line


def test_format_outcome_log_line_entered_says_entry_not_no_entry():
    record = {"underlying": "QQQ", "outcome": "entered", "composite": 0.35, "pillars": {}, "flow_source": "uw"}
    line = format_outcome_log_line(record)
    assert line.startswith("DPOF QQQ: entry — entered")


def test_format_outcome_log_line_never_raises_on_malformed_record():
    assert format_outcome_log_line({}) is not None
    assert "n/a" in format_outcome_log_line({"underlying": "SPY"})


# ─────────────────────────────────────────────────────────────────────────
# NearMissLogger
# ─────────────────────────────────────────────────────────────────────────

def test_near_miss_logs_on_first_seen():
    nm = NearMissLogger()
    assert nm.should_log("SPY", "below_threshold", now=1000.0, window_seconds=600) is True


def test_near_miss_logs_on_outcome_change():
    nm = NearMissLogger()
    assert nm.should_log("SPY", "below_threshold", now=1000.0, window_seconds=600) is True
    assert nm.should_log("SPY", "below_threshold", now=1010.0, window_seconds=600) is False
    assert nm.should_log("SPY", "cooldown", now=1020.0, window_seconds=600) is True  # changed -> log


def test_near_miss_suppresses_same_outcome_within_window():
    nm = NearMissLogger()
    assert nm.should_log("SPY", "below_threshold", now=0.0, window_seconds=600) is True
    for t in (30.0, 120.0, 300.0, 599.0):
        assert nm.should_log("SPY", "below_threshold", now=t, window_seconds=600) is False


def test_near_miss_relogs_after_window_elapses():
    nm = NearMissLogger()
    assert nm.should_log("SPY", "below_threshold", now=0.0, window_seconds=600) is True
    assert nm.should_log("SPY", "below_threshold", now=599.0, window_seconds=600) is False
    assert nm.should_log("SPY", "below_threshold", now=600.0, window_seconds=600) is True


def test_near_miss_keys_are_independent_per_underlying():
    nm = NearMissLogger()
    assert nm.should_log("SPY", "below_threshold", now=0.0, window_seconds=600) is True
    assert nm.should_log("QQQ", "below_threshold", now=0.0, window_seconds=600) is True
    assert nm.should_log("QQQ", "below_threshold", now=1.0, window_seconds=600) is False
    assert nm.should_log("SPY", "below_threshold", now=1.0, window_seconds=600) is False


# ─────────────────────────────────────────────────────────────────────────
# fold_record_into_day_stats (pure)
# ─────────────────────────────────────────────────────────────────────────

def _record(underlying="SPY", outcome="below_threshold", composite=0.12, direction="neutral",
            nominated=False, decider=None, jev=None, flow_source="uw",
            pillars=None) -> dict:
    default_pillars = {
        "nofa": {"value": 0.2, "available": True, "source": "uw:x"},
        "premium": {"value": None, "available": False, "source": ""},
        "darkpool": {"value": -0.1, "available": True, "source": "uw:y"},
        "gex": {"value": None, "available": False, "source": ""},
    }
    return {
        "ts": "2026-09-25T10:00:00-04:00",
        "underlying": underlying,
        "composite": composite,
        "strength": "none",
        "direction": direction,
        "pillars": pillars if pillars is not None else default_pillars,
        "flow_source": flow_source,
        "regime": "normal",
        "price_confirm": {"ok": None, "reasons": []},
        "vetoes": [],
        "reasons": [],
        "is_signal_symbol": True,
        "nominated": nominated,
        "outcome": outcome,
        "decider": decider,
        "jev": jev,
    }


def test_fold_never_mutates_input():
    stats = {"day": "2026-09-25", "underlyings": {}}
    before = json.loads(json.dumps(stats))
    fold_record_into_day_stats(stats, _record())
    assert stats == before


def test_fold_increments_outcome_counts_per_underlying():
    stats = {"day": "2026-09-25", "underlyings": {}}
    stats = fold_record_into_day_stats(stats, _record(underlying="SPY", outcome="below_threshold"))
    stats = fold_record_into_day_stats(stats, _record(underlying="SPY", outcome="below_threshold"))
    stats = fold_record_into_day_stats(stats, _record(underlying="SPY", outcome="cooldown"))
    counts = stats["underlyings"]["SPY"]["outcome_counts"]
    assert counts == {"below_threshold": 2, "cooldown": 1}


def test_fold_tracks_max_abs_composite_with_time_and_direction():
    stats = {"day": "2026-09-25", "underlyings": {}}
    stats = fold_record_into_day_stats(stats, _record(composite=0.12, direction="bullish"))
    stats = fold_record_into_day_stats(stats, _record(composite=-0.30, direction="bearish"))
    stats = fold_record_into_day_stats(stats, _record(composite=0.05, direction="bullish"))
    u = stats["underlyings"]["SPY"]
    assert u["max_abs_composite"] == pytest.approx(0.30)
    assert u["max_abs_composite_direction"] == "bearish"


def test_fold_pillar_availability_counts():
    stats = {"day": "2026-09-25", "underlyings": {}}
    stats = fold_record_into_day_stats(stats, _record())
    stats = fold_record_into_day_stats(stats, _record())
    u = stats["underlyings"]["SPY"]
    assert u["pillar_total"]["nofa"] == 2
    assert u["pillar_available"]["nofa"] == 2
    assert u["pillar_total"]["premium"] == 2
    assert u["pillar_available"]["premium"] == 0


def test_fold_flow_source_counts():
    stats = {"day": "2026-09-25", "underlyings": {}}
    stats = fold_record_into_day_stats(stats, _record(flow_source="uw"))
    stats = fold_record_into_day_stats(stats, _record(flow_source="selfflow"))
    stats = fold_record_into_day_stats(stats, _record(flow_source="uw"))
    assert stats["underlyings"]["SPY"]["flow_source_counts"] == {"uw": 2, "selfflow": 1}


def test_fold_nominations_and_decider_counters():
    stats = {"day": "2026-09-25", "underlyings": {}}
    stats = fold_record_into_day_stats(stats, _record(nominated=True, outcome="deliberating"))
    stats = fold_record_into_day_stats(
        stats, _record(nominated=False, outcome="decider_pass", decider={"action": "pass"})
    )
    stats = fold_record_into_day_stats(
        stats, _record(nominated=False, outcome="entered", decider={"action": "enter"})
    )
    assert stats["nominations"] == 1
    assert stats["decider_pass"] == 1
    assert stats["decider_enter"] == 1
    assert stats["entries"] == 1


def test_fold_decider_unavailable_counter():
    stats = {"day": "2026-09-25", "underlyings": {}}
    stats = fold_record_into_day_stats(stats, _record(outcome="decider_unavailable"))
    assert stats["decider_unavailable"] == 1


def test_fold_jev_veto_counter():
    stats = {"day": "2026-09-25", "underlyings": {}}
    stats = fold_record_into_day_stats(
        stats, _record(outcome="jev_veto", jev={"allow": False, "veto_reasons": ["size_reckless"]})
    )
    assert stats["jev_vetoes"] == 1
    stats = fold_record_into_day_stats(
        stats, _record(outcome="entered", jev={"allow": True, "veto_reasons": []})
    )
    assert stats["jev_vetoes"] == 1  # allow=True never counts as a veto


def test_fold_missing_underlying_is_a_noop():
    stats = {"day": "2026-09-25", "underlyings": {}}
    out = fold_record_into_day_stats(stats, {"outcome": "below_threshold"})
    assert out["underlyings"] == {}


# ─────────────────────────────────────────────────────────────────────────
# SignalLog I/O
# ─────────────────────────────────────────────────────────────────────────

from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def test_write_cycle_appends_jsonl_and_day_stats(tmp_path):
    async def _t():
        sl = SignalLog(str(tmp_path))
        now_et = datetime(2026, 9, 25, 10, 0, tzinfo=ET)
        records = [_record(underlying="SPY"), _record(underlying="QQQ", outcome="cooldown")]
        stats = await sl.write_cycle(records, now_et, max_lines=1000)

        signals_path = os.path.join(str(tmp_path), "signals.jsonl")
        with open(signals_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        parsed = [json.loads(l) for l in lines]
        assert parsed[0]["underlying"] == "SPY"
        assert parsed[1]["underlying"] == "QQQ"

        assert stats["day"] == "2026-09-25"
        assert stats["cycles_evaluated"] == 1
        assert "SPY" in stats["underlyings"]
        assert "QQQ" in stats["underlyings"]

        stats_path = os.path.join(str(tmp_path), "signal_day_stats.json")
        with open(stats_path) as f:
            on_disk = json.load(f)
        assert on_disk["cycles_evaluated"] == 1

    asyncio.run(_t())


def test_write_cycle_empty_records_is_noop(tmp_path):
    async def _t():
        sl = SignalLog(str(tmp_path))
        now_et = datetime(2026, 9, 25, 10, 0, tzinfo=ET)
        result = await sl.write_cycle([], now_et, max_lines=1000)
        assert result == {}
        assert not os.path.exists(os.path.join(str(tmp_path), "signals.jsonl"))

    asyncio.run(_t())


def test_write_cycle_multiple_cycles_increments_cycles_evaluated(tmp_path):
    async def _t():
        sl = SignalLog(str(tmp_path))
        now_et = datetime(2026, 9, 25, 10, 0, tzinfo=ET)
        for _ in range(3):
            stats = await sl.write_cycle([_record(underlying="SPY")], now_et, max_lines=1000)
        assert stats["cycles_evaluated"] == 3
        assert stats["underlyings"]["SPY"]["outcome_counts"]["below_threshold"] == 3

    asyncio.run(_t())


def test_write_cycle_day_rollover_resets_stats(tmp_path):
    async def _t():
        sl = SignalLog(str(tmp_path))
        day1 = datetime(2026, 9, 25, 15, 55, tzinfo=ET)
        day2 = datetime(2026, 9, 26, 9, 45, tzinfo=ET)
        await sl.write_cycle([_record(underlying="SPY", outcome="entered")], day1, max_lines=1000)
        stats2 = await sl.write_cycle([_record(underlying="SPY", outcome="cooldown")], day2, max_lines=1000)
        assert stats2["day"] == "2026-09-26"
        assert stats2["cycles_evaluated"] == 1
        assert stats2["underlyings"]["SPY"]["outcome_counts"] == {"cooldown": 1}
        # JSONL is append-only across the rollover -- both days' lines remain.
        with open(os.path.join(str(tmp_path), "signals.jsonl")) as f:
            assert len(f.readlines()) == 2

    asyncio.run(_t())


def test_write_cycle_caps_and_trims_signals_jsonl(tmp_path):
    async def _t():
        sl = SignalLog(str(tmp_path))
        now_et = datetime(2026, 9, 25, 10, 0, tzinfo=ET)
        for i in range(25):
            await sl.write_cycle([_record(underlying="SPY", outcome=f"o{i}")], now_et, max_lines=10)
        with open(os.path.join(str(tmp_path), "signals.jsonl")) as f:
            lines = f.readlines()
        assert len(lines) <= 10
        # The newest record must survive the trim (trim keeps the TAIL).
        last = json.loads(lines[-1])
        assert last["outcome"] == "o24"

    asyncio.run(_t())


def test_write_cycle_never_raises_on_corrupt_day_stats(tmp_path):
    async def _t():
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(os.path.join(str(tmp_path), "signal_day_stats.json"), "w") as f:
            f.write("{not json")
        sl = SignalLog(str(tmp_path))
        now_et = datetime(2026, 9, 25, 10, 0, tzinfo=ET)
        stats = await sl.write_cycle([_record(underlying="SPY")], now_et, max_lines=1000)
        assert stats["cycles_evaluated"] == 1  # started fresh, didn't raise/propagate the corruption

    asyncio.run(_t())


def test_write_cycle_runs_off_the_event_loop(tmp_path):
    """CLAUDE.md: never fcntl.flock / blocking I/O on the asyncio event loop
    thread — mirrors tests/dpof/test_ledger.py's
    test_all_file_io_goes_through_run_in_executor."""
    async def _t():
        loop = asyncio.get_running_loop()
        calls = []
        original = loop.run_in_executor

        def tracking(executor, func, *a):
            calls.append(func)
            return original(executor, func, *a)

        loop.run_in_executor = tracking
        try:
            sl = SignalLog(str(tmp_path))
            now_et = datetime(2026, 9, 25, 10, 0, tzinfo=ET)
            await sl.write_cycle([_record(underlying="SPY")], now_et, max_lines=1000)
        finally:
            loop.run_in_executor = original

        assert len(calls) >= 1

    asyncio.run(_t())


def test_write_cycle_default_max_lines_when_none_passed(tmp_path):
    async def _t():
        sl = SignalLog(str(tmp_path))
        now_et = datetime(2026, 9, 25, 10, 0, tzinfo=ET)
        stats = await sl.write_cycle([_record(underlying="SPY")], now_et, max_lines=None)
        assert stats["cycles_evaluated"] == 1

    asyncio.run(_t())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


def test_trim_reads_file_only_when_cap_crossed(tmp_path, monkeypatch):
    """The line count is kept in memory: a write must not re-read an up-to-cap
    signals.jsonl every cycle, only when the cap is crossed (with hysteresis)."""
    calls = {"n": 0}
    real_trim = SignalLog._trim_signals_if_needed

    def _spy(path, max_lines, keep=None, **kwargs):
        calls["n"] += 1
        return real_trim(path, max_lines, keep=keep, **kwargs)

    monkeypatch.setattr(SignalLog, "_trim_signals_if_needed", staticmethod(_spy))

    async def _t():
        sl = SignalLog(str(tmp_path))
        now_et = datetime(2026, 9, 25, 10, 0, tzinfo=ET)
        for i in range(250):
            await sl.write_cycle([_record(underlying="SPY", outcome=f"o{i}")], now_et, max_lines=100)
        with open(os.path.join(str(tmp_path), "signals.jsonl")) as f:
            lines = f.readlines()
        assert len(lines) <= 100
        assert json.loads(lines[-1])["outcome"] == "o249"
        assert calls["n"] <= 20  # ~every cap/10 writes, not every write

    asyncio.run(_t())


def test_existing_file_is_counted_on_first_write(tmp_path):
    path = os.path.join(str(tmp_path), "signals.jsonl")
    with open(path, "w") as f:
        for i in range(95):
            f.write(json.dumps({"i": i}) + "\n")

    async def _t():
        sl = SignalLog(str(tmp_path))  # fresh process: count starts unknown
        now_et = datetime(2026, 9, 25, 10, 0, tzinfo=ET)
        for i in range(10):
            await sl.write_cycle([_record(underlying="SPY", outcome=f"n{i}")], now_et, max_lines=100)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) <= 100
        assert json.loads(lines[-1])["outcome"] == "n9"

    asyncio.run(_t())


def test_flapping_outcome_is_bounded():
    """An outcome alternating every 30 s cycle must not log every cycle."""
    nm = NearMissLogger()
    logged = sum(
        nm.should_log("SPY", "below_threshold" if i % 2 else "price_not_confirmed", now=i * 30.0, window_seconds=600)
        for i in range(20)
    )
    assert logged == NearMissLogger.MAX_CHANGE_LOGS_PER_WINDOW
    # after the window a change logs again
    assert nm.should_log("SPY", "cooldown", now=20 * 30.0 + 600, window_seconds=600) is True


# ─────────────────────────────────────────────────────────────────────────
# Trim archive (CLIO_DPOF_SIGNAL_ARCHIVE, default ON)
# ─────────────────────────────────────────────────────────────────────────

_CAP = 100
_KEEP = 90  # (cap * 9) // 10
_ARCHIVE_SUBDIR = "archive"


def _live_lines(data_dir):
    with open(os.path.join(data_dir, "signals.jsonl"), encoding="utf-8") as f:
        return f.readlines()


def _archive_files(data_dir):
    d = os.path.join(data_dir, _ARCHIVE_SUBDIR)
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if n.startswith("signals_") and n.endswith(".jsonl.gz"))


def _read_archive(data_dir, name):
    with gzip.open(os.path.join(data_dir, _ARCHIVE_SUBDIR, name), "rb") as gz:
        return gz.read().decode("utf-8").splitlines(keepends=True)


def _fixed_clock(stamp):
    return lambda: stamp


def _write_n(sl, n, start=0):
    async def _t():
        now_et = datetime(2026, 9, 25, 10, 0, tzinfo=ET)
        for i in range(start, start + n):
            await sl.write_cycle([_record(underlying="SPY", outcome=f"o{i}")], now_et, max_lines=_CAP)

    asyncio.run(_t())


def test_archive_holds_exactly_the_removed_head_and_nothing_is_lost(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIO_DPOF_SIGNAL_ARCHIVE", raising=False)
    data_dir = str(tmp_path)
    sl = SignalLog(data_dir, clock=_fixed_clock(datetime(2026, 9, 25, 10, 0, 0, tzinfo=ET)))
    _write_n(sl, _CAP)  # exactly at the cap: no trim yet
    before = _live_lines(data_dir)
    assert len(before) == _CAP and _archive_files(data_dir) == []

    _write_n(sl, 1, start=_CAP)  # crosses the cap -> trim
    after = _live_lines(data_dir)
    full = before + [json.dumps(_record(underlying="SPY", outcome=f"o{_CAP}"), default=str) + "\n"]
    removed_count = len(full) - _KEEP
    assert len(after) == _KEEP

    names = _archive_files(data_dir)
    assert names == ["signals_20260925_100000.jsonl.gz"]
    archived = _read_archive(data_dir, names[0])
    assert len(archived) == removed_count
    assert hashlib.sha256("".join(archived).encode()).hexdigest() == \
        hashlib.sha256("".join(full[:removed_count]).encode()).hexdigest()
    # archive + live file == the full original sequence, in order
    assert archived + after == full


def test_live_file_keeps_exactly_the_newest_ninety_percent(tmp_path):
    sl = SignalLog(str(tmp_path))
    _write_n(sl, _CAP + 1)
    lines = _live_lines(str(tmp_path))
    assert len(lines) == _KEEP
    assert json.loads(lines[-1])["outcome"] == f"o{_CAP}"
    assert json.loads(lines[0])["outcome"] == f"o{_CAP + 1 - _KEEP}"


def test_archive_write_failure_warns_and_still_trims(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("CLIO_DPOF_SIGNAL_ARCHIVE", raising=False)

    def _boom(*args, **kwargs):
        raise OSError("archive dir unwritable")

    monkeypatch.setattr(signal_log_mod.gzip, "open", _boom)
    sl = SignalLog(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="clio.dpof.signal_log"):
        _write_n(sl, _CAP + 1)  # must not raise into the caller
    assert len(_live_lines(str(tmp_path))) == _KEEP
    assert _archive_files(str(tmp_path)) == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("archive" in r.getMessage() and "OSError" in r.getMessage() for r in warnings)
    # no record content leaks into the warning
    assert not any("outcome" in r.getMessage() for r in warnings)


@pytest.mark.parametrize("value", ["0", "false", "OFF", " no "])
def test_flag_off_writes_no_archive_but_still_trims(tmp_path, monkeypatch, value):
    monkeypatch.setenv("CLIO_DPOF_SIGNAL_ARCHIVE", value)
    sl = SignalLog(str(tmp_path))
    _write_n(sl, _CAP + 1)
    assert len(_live_lines(str(tmp_path))) == _KEEP
    assert not os.path.exists(os.path.join(str(tmp_path), _ARCHIVE_SUBDIR))


@pytest.mark.parametrize("value", [None, "", "1", "true", "banana"])
def test_flag_default_and_bad_values_keep_archive_on(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CLIO_DPOF_SIGNAL_ARCHIVE", raising=False)
    else:
        monkeypatch.setenv("CLIO_DPOF_SIGNAL_ARCHIVE", value)
    assert signal_archive_enabled() is True


def test_two_trims_make_two_distinct_archive_names(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIO_DPOF_SIGNAL_ARCHIVE", raising=False)
    sl = SignalLog(str(tmp_path), clock=_fixed_clock(datetime(2026, 9, 25, 10, 0, 0, tzinfo=ET)))
    _write_n(sl, _CAP + 1)           # trim 1 -> live holds 90
    _write_n(sl, _CAP - _KEEP + 1, start=_CAP + 1)  # 11 more -> 101 -> trim 2
    names = _archive_files(str(tmp_path))
    assert names == ["signals_20260925_100000.jsonl.gz", "signals_20260925_100000_2.jsonl.gz"]
    assert len(_read_archive(str(tmp_path), names[0])) == _CAP + 1 - _KEEP
    assert len(_read_archive(str(tmp_path), names[1])) == _CAP + 1 - _KEEP
    assert len(_live_lines(str(tmp_path))) == _KEEP


def test_archive_is_written_before_the_live_file_is_replaced(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIO_DPOF_SIGNAL_ARCHIVE", raising=False)
    data_dir = str(tmp_path)
    seen = {}
    real_replace = os.replace

    def _spy(src, dst, *a, **kw):
        if os.path.basename(dst) == "signals.jsonl":
            seen["archives_at_live_replace"] = _archive_files(data_dir)
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(signal_log_mod.os, "replace", _spy)
    _write_n(SignalLog(data_dir), _CAP + 1)
    assert len(seen["archives_at_live_replace"]) == 1
