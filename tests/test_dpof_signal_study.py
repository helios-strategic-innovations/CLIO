"""Tests for scripts/dpof_signal_study.py (F3 —
docs/design/clio_dpof_followups_2026-09-26.md §5/§5.4).

Synthetic data only: no network, no `data/` reads or writes. Everything
runs off `tmp_path` fixtures and `--bars-file`, matching the design frame's
"never reads/writes data/" and "stdlib only for the pure core" constraints
(this sandbox has neither `alpaca` nor `dotenv` installed, which is itself
part of what proves the module-import purity below).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts import dpof_signal_study as dss  # noqa: E402

ET = ZoneInfo("America/New_York")
_BASE = datetime(2026, 9, 1, 9, 30, 0, tzinfo=ET)


def _epoch(dt) -> float:
    return dt.timestamp()


def _ts(dt) -> str:
    return dt.isoformat()


def _rec(underlying: str, minute_offset: float, composite, **overrides) -> dict:
    dt = _BASE + timedelta(minutes=minute_offset)
    rec = {
        "ts": _ts(dt),
        "underlying": underlying,
        "composite": composite,
        "strength": "normal",
        "direction": None,
        "price_confirm": {"ok": None, "reasons": []},
        "regime": "trend",
        "flow_source": "uw",
        "nominated": False,
    }
    rec.update(overrides)
    return rec


def _load(records: list[dict]) -> list[dict]:
    return dss.load_signal_records(json.dumps(r) for r in records)


def _bar(t_offset_s: float, c: float) -> dict:
    return {"t": _epoch(_BASE) + t_offset_s, "c": c}


def _event(underlying="SPY", direction=1, ts=None, **kw) -> dss.Event:
    return dss.Event(
        ts=ts if ts is not None else _epoch(_BASE),
        underlying=underlying,
        direction=direction,
        composite=kw.pop("composite", 0.2),
        strength=kw.pop("strength", "normal"),
        price_ok=kw.pop("price_ok", None),
        regime=kw.pop("regime", "trend"),
        flow_source=kw.pop("flow_source", "uw"),
        nominated=kw.pop("nominated", False),
    )


# ─────────────────────────────────────────────────────────────────────────
# Import purity — no I/O, no network, no third-party deps at import time
# ─────────────────────────────────────────────────────────────────────────

def test_module_imports_without_alpaca_dotenv_pandas_numpy():
    """A bare import pulls in none of alpaca/dotenv/pandas/numpy/agent (they
    are only imported lazily, inside functions on the live-fetch path).
    Checked in a FRESH interpreter: in the shared pytest process other test
    modules have usually imported dotenv/numpy already, so inspecting this
    process's sys.modules would say nothing about the script."""
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import scripts.dpof_signal_study; "
        "print(','.join(n for n in ('alpaca', 'dotenv', 'pandas', 'numpy', 'agent') if n in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, REPO_ROOT],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"bare import pulled in: {result.stdout.strip()}"


def test_agent_never_imports_this_script():
    """grep test (design frame §5.4): nothing under agent/ imports this
    standalone research script."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "agent/*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    offenders = []
    for rel in filter(None, listed.stdout.split("\0")):
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8", errors="replace") as f:
            if "dpof_signal_study" in f.read():
                offenders.append(rel)
    assert offenders == [], f"agent/ must never import dpof_signal_study: {offenders}"


# ─────────────────────────────────────────────────────────────────────────
# load_signal_records
# ─────────────────────────────────────────────────────────────────────────

def test_load_signal_records_skips_none_composite_and_bad_json():
    lines = [
        json.dumps(_rec("SPY", 0, 0.2)),
        "not json at all",
        json.dumps({"ts": _ts(_BASE + timedelta(minutes=1)), "underlying": "SPY", "composite": None}),
        json.dumps(_rec("SPY", 2, 0.3)),
        "",
    ]
    out = dss.load_signal_records(lines)
    assert len(out) == 2
    assert all(r["composite"] is not None for r in out)
    assert out[0]["composite"] == pytest.approx(0.2)
    assert out[1]["composite"] == pytest.approx(0.3)


def test_load_signal_records_skips_unparsable_ts():
    lines = [json.dumps({"ts": "not-a-date", "underlying": "SPY", "composite": 0.2})]
    assert dss.load_signal_records(lines) == []


# ─────────────────────────────────────────────────────────────────────────
# find_events (R6: crossing, refractory, direction flip, gap reset, None)
# ─────────────────────────────────────────────────────────────────────────

def test_find_events_first_crossing_fires_once():
    records = _load([
        _rec("SPY", 0, 0.05),
        _rec("SPY", 1, 0.20),   # crosses 0.15 here
        _rec("SPY", 2, 0.22),   # still above -- not a new crossing
    ])
    events = dss.find_events(records, threshold=0.15, min_gap_minutes=0)
    assert len(events) == 1
    assert events[0].underlying == "SPY"
    assert events[0].direction == 1
    assert events[0].composite == pytest.approx(0.20)


def test_find_events_refractory_suppresses_second_crossing_within_gap():
    records = _load([
        _rec("SPY", 0, 0.05),
        _rec("SPY", 1, 0.20),   # event 1
        _rec("SPY", 2, 0.05),   # dips below threshold
        _rec("SPY", 3, 0.20),   # re-crosses 2 min later -- inside a 15 min refractory
    ])
    events = dss.find_events(records, threshold=0.15, min_gap_minutes=15)
    assert len(events) == 1


def test_find_events_refractory_allows_after_gap_elapses():
    records = _load([
        _rec("SPY", 0, 0.05),
        _rec("SPY", 1, 0.20),    # event 1
        _rec("SPY", 2, 0.05),
        _rec("SPY", 20, 0.20),   # 19 min after event 1 -- outside a 15 min refractory
    ])
    events = dss.find_events(records, threshold=0.15, min_gap_minutes=15)
    assert len(events) == 2


def test_find_events_min_gap_zero_is_pure_first_crossing():
    records = _load([
        _rec("SPY", 0, 0.05),
        _rec("SPY", 1, 0.20),
        _rec("SPY", 2, 0.05),
        _rec("SPY", 3, 0.20),
    ])
    events = dss.find_events(records, threshold=0.15, min_gap_minutes=0)
    assert len(events) == 2


def test_find_events_direction_flip_fires_immediately_no_refractory_carryover():
    records = _load([
        _rec("SPY", 0, 0.20),     # event, bullish
        _rec("SPY", 1, -0.20),    # opposite sign -- fires despite only 1 min later
    ])
    events = dss.find_events(records, threshold=0.15, min_gap_minutes=15)
    assert len(events) == 2
    assert events[0].direction == 1
    assert events[1].direction == -1


def test_find_events_gap_reset_after_5_minutes_creates_new_crossing():
    """Two records for one underlying, >5 min apart, both above threshold
    and the same sign: WITHOUT the gap-reset rule these wouldn't be a new
    crossing at all (never dipped below / flipped sign in between); the
    >5min-gap = new-session rule is what makes the second record a fresh
    crossing. min_gap_minutes=0 isolates this from the refractory check."""
    records = _load([
        _rec("SPY", 0, 0.20),
        _rec("SPY", 20, 0.20),   # 20 min later -- gap-reset (>5min) makes this a new crossing
    ])
    events = dss.find_events(records, threshold=0.15, min_gap_minutes=0)
    assert len(events) == 2


def test_find_events_gap_reset_still_respects_refractory():
    """The gap-reset rule and the refractory are independent, ANDed
    conditions: a gap-reset crossing can still be suppressed by an
    unexpired refractory in the same (underlying, direction)."""
    records = _load([
        _rec("SPY", 0, 0.20),
        _rec("SPY", 20, 0.20),   # gap-reset makes this a crossing, but the 30min refractory suppresses it
    ])
    events = dss.find_events(records, threshold=0.15, min_gap_minutes=30)
    assert len(events) == 1


def test_find_events_no_gap_reset_within_5_minutes_is_not_a_crossing():
    records = _load([
        _rec("SPY", 0, 0.20),
        _rec("SPY", 4, 0.20),   # 4 min later -- no gap reset, still above threshold, same sign
    ])
    events = dss.find_events(records, threshold=0.15, min_gap_minutes=0)
    assert len(events) == 1  # 2nd record isn't even a crossing (no dip below threshold in between)


def test_find_events_none_composite_skipped():
    records = [
        {"ts": _ts(_BASE), "underlying": "SPY", "composite": None},
        *_load([_rec("SPY", 1, 0.20)]),
    ]
    events = dss.find_events(records, threshold=0.15, min_gap_minutes=0)
    assert len(events) == 1


def test_find_events_independent_per_underlying():
    records = _load([
        _rec("SPY", 0, 0.20),
        _rec("QQQ", 0, 0.20),
    ])
    events = dss.find_events(records, threshold=0.15, min_gap_minutes=0)
    assert {e.underlying for e in events} == {"SPY", "QQQ"}
    assert len(events) == 2


def test_find_events_below_threshold_never_fires():
    records = _load([_rec("SPY", m, 0.05) for m in range(5)])
    assert dss.find_events(records, threshold=0.15, min_gap_minutes=0) == []


# ─────────────────────────────────────────────────────────────────────────
# price_at (no look-ahead, staleness)
# ─────────────────────────────────────────────────────────────────────────

def test_price_at_no_look_ahead():
    bars = [_bar(0, 100.0), _bar(60, 101.0), _bar(120, 102.0)]
    # bar starting at t=60 ends at t=120; at epoch 119 it hasn't ended yet,
    # so the only usable bar is the one starting at t=0.
    epoch = _epoch(_BASE) + 119
    assert dss.price_at(bars, epoch) == 100.0


def test_price_at_uses_latest_qualifying_bar_at_exact_boundary():
    bars = [_bar(0, 100.0), _bar(60, 101.0)]
    epoch = _epoch(_BASE) + 120  # bar starting at 60 ends exactly at 120 -- inclusive
    assert dss.price_at(bars, epoch) == 101.0


def test_price_at_staleness_returns_none_beyond_max():
    bars = [_bar(0, 100.0)]  # ends at t=60
    epoch = _epoch(_BASE) + 60 + 200  # 200s stale > 120s default
    assert dss.price_at(bars, epoch, max_staleness_s=120) is None


def test_price_at_within_staleness_window_returns_price():
    bars = [_bar(0, 100.0)]
    epoch = _epoch(_BASE) + 60 + 100  # 100s stale, under the 120s default
    assert dss.price_at(bars, epoch, max_staleness_s=120) == 100.0


def test_price_at_empty_bars_returns_none():
    assert dss.price_at([], _epoch(_BASE)) is None


def test_price_at_malformed_bar_entries_skipped():
    bars = [{"t": None, "c": 100.0}, {"t": 0, "c": None}, _bar(0, 100.0)]
    assert dss.price_at(bars, _epoch(_BASE) + 60) == 100.0


# ─────────────────────────────────────────────────────────────────────────
# forward_returns (sign, bps, None propagation -- no fake zeros)
# ─────────────────────────────────────────────────────────────────────────

def test_forward_returns_bullish_positive_move_is_positive_bps():
    t0 = _epoch(_BASE) + 300
    ev = _event(direction=1, ts=t0, underlying="SPY")
    bars = {"SPY": [_bar(300 - 60, 100.0), _bar(300 + 5 * 60 - 60, 105.0)]}
    rows = dss.forward_returns([ev], bars, horizons=(5,))
    assert len(rows) == 1
    assert rows[0]["r_bps"] == pytest.approx(500.0)  # (105-100)/100 * 10000


def test_forward_returns_bearish_flips_sign():
    t0 = _epoch(_BASE) + 300
    ev = _event(direction=-1, ts=t0, underlying="SPY")
    bars = {"SPY": [_bar(300 - 60, 100.0), _bar(300 + 5 * 60 - 60, 105.0)]}
    rows = dss.forward_returns([ev], bars, horizons=(5,))
    assert rows[0]["r_bps"] == pytest.approx(-500.0)


def test_forward_returns_missing_symbol_is_no_data_not_zero():
    ev = _event(underlying="QQQ", ts=_epoch(_BASE) + 300)
    rows = dss.forward_returns([ev], {"SPY": [_bar(0, 100.0)]}, horizons=(5,))
    assert rows[0]["r_bps"] is None


def test_forward_returns_missing_p1_is_no_data():
    t0 = _epoch(_BASE) + 300
    ev = _event(direction=1, ts=t0, underlying="SPY")
    bars = {"SPY": [_bar(300 - 60, 100.0)]}  # no bar exists near t0 + 5*60
    rows = dss.forward_returns([ev], bars, horizons=(5,))
    assert rows[0]["r_bps"] is None


def test_forward_returns_zero_p0_is_no_data_never_divides():
    t0 = _epoch(_BASE) + 300
    ev = _event(direction=1, ts=t0, underlying="SPY")
    bars = {"SPY": [_bar(300 - 60, 0.0), _bar(300 + 5 * 60 - 60, 105.0)]}
    rows = dss.forward_returns([ev], bars, horizons=(5,))
    assert rows[0]["r_bps"] is None


def test_forward_returns_one_row_per_horizon():
    ev = _event(ts=_epoch(_BASE) + 300, underlying="SPY")
    rows = dss.forward_returns([ev], {}, horizons=(5, 15, 30, 60))
    assert [r["horizon"] for r in rows] == [5, 15, 30, 60]
    assert all(r["r_bps"] is None for r in rows)


# ─────────────────────────────────────────────────────────────────────────
# summarize (counts, hit rate, median, n<30 marking -- no p-values)
# ─────────────────────────────────────────────────────────────────────────

def test_summarize_hand_computed_stats():
    rows = [
        {"r_bps": 10.0, "k": "a"},
        {"r_bps": -5.0, "k": "a"},
        {"r_bps": 20.0, "k": "a"},
        {"r_bps": None, "k": "a"},
        {"r_bps": 30.0, "k": "a"},
    ]
    out = dss.summarize(rows, group_by=lambda r: (r["k"],))
    assert len(out) == 1
    g = out[0]
    assert g["n"] == 4
    assert g["no_data"] == 1
    assert g["hit_rate"] == pytest.approx(0.75)          # 3 of 4 positive
    assert g["mean_bps"] == pytest.approx(13.75)
    assert g["median_bps"] == pytest.approx(15.0)         # sorted -5,10,20,30 -> (10+20)/2
    assert g["note"] == "descriptive only (n<30)"


def test_summarize_all_no_data_group_is_all_none():
    rows = [{"r_bps": None, "k": "a"} for _ in range(3)]
    out = dss.summarize(rows, group_by=lambda r: (r["k"],))
    g = out[0]
    assert g["n"] == 0
    assert g["no_data"] == 3
    assert g["hit_rate"] is None
    assert g["mean_bps"] is None
    assert g["median_bps"] is None
    assert g["note"] == "descriptive only (n<30)"


def test_summarize_n_at_30_has_no_descriptive_note():
    rows = [{"r_bps": float(i - 15), "k": "a"} for i in range(30)]
    out = dss.summarize(rows, group_by=lambda r: (r["k"],))
    assert out[0]["n"] == 30
    assert out[0]["note"] is None


def test_summarize_n_at_29_has_descriptive_note():
    rows = [{"r_bps": float(i - 14), "k": "a"} for i in range(29)]
    out = dss.summarize(rows, group_by=lambda r: (r["k"],))
    assert out[0]["n"] == 29
    assert out[0]["note"] == "descriptive only (n<30)"


def test_summarize_groups_and_sorts_deterministically():
    rows = [{"r_bps": 1.0, "k": "b"}, {"r_bps": 2.0, "k": "a"}]
    out = dss.summarize(rows, group_by=lambda r: (r["k"],))
    assert [o["key"] for o in out] == [("a",), ("b",)]


def test_summarize_signal_and_baseline_unpack_named_fields():
    rows = [{"r_bps": 5.0, "threshold": 0.15, "horizon": 5, "price_ok": True, "underlying": "SPY"}]
    out = dss.summarize_signal(rows)
    assert out[0]["threshold"] == 0.15
    assert out[0]["horizon"] == 5
    assert out[0]["price_ok"] == "confirmed"
    assert out[0]["underlying"] == "SPY"
    assert "key" not in out[0]

    base_rows = [{"r_bps": 5.0, "horizon": 5, "underlying": "SPY"}]
    base_out = dss.summarize_baseline(base_rows)
    assert base_out[0]["underlying"] == "SPY"
    assert base_out[0]["horizon"] == 5
    assert "key" not in base_out[0]


# ─────────────────────────────────────────────────────────────────────────
# CLI end-to-end (--bars-file, --json-out)
# ─────────────────────────────────────────────────────────────────────────

def test_cli_end_to_end_bars_file_and_json_out(tmp_path):
    sig_path = tmp_path / "signals.jsonl"
    bars_path = tmp_path / "bars.jsonl"
    json_out = tmp_path / "out.json"

    records = [
        _rec("SPY", 0, 0.05),
        _rec("SPY", 1, 0.20),    # event 1
        _rec("SPY", 2, 0.22),
        _rec("SPY", 30, 0.05),
        _rec("SPY", 31, 0.20),   # event 2 (well past any refractory)
    ]
    sig_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    bars = [
        {"symbol": "SPY", "t": _epoch(_BASE) + minute * 60, "c": 100.0 + minute * 0.1}
        for minute in range(0, 60)
    ]
    bars_path.write_text("\n".join(json.dumps(b) for b in bars) + "\n", encoding="utf-8")

    script = os.path.join(REPO_ROOT, "scripts", "dpof_signal_study.py")
    result = subprocess.run(
        [
            sys.executable, script,
            "--signals", str(sig_path),
            "--bars-file", str(bars_path),
            "--thresholds", "0.15",
            "--horizons", "5,15",
            "--min-gap-minutes", "0",
            "--json-out", str(json_out),
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=30, check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "DPOF signal study" in result.stdout
    assert "baseline" in result.stdout.lower()

    assert json_out.exists()
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["n_records"] == len(records)
    assert report["events_count"]["0.15"] == 2
    assert "signal" in report and "baseline" in report
    assert report["underlyings"] == ["SPY"]
    assert report["time_span"]["first"] is not None
    assert report["time_span"]["last"] is not None


def test_cli_errors_cleanly_on_missing_signals_file(tmp_path):
    script = os.path.join(REPO_ROOT, "scripts", "dpof_signal_study.py")
    result = subprocess.run(
        [sys.executable, script, "--signals", str(tmp_path / "nope.jsonl")],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=30, check=False,
    )
    assert result.returncode == 2
    assert "not found" in result.stderr


def test_cli_csv_bars_file_with_header(tmp_path):
    sig_path = tmp_path / "signals.jsonl"
    bars_path = tmp_path / "bars.csv"

    records = [_rec("SPY", 0, 0.05), _rec("SPY", 1, 0.20)]
    sig_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    lines = ["symbol,t,c"] + [
        f"SPY,{_epoch(_BASE) + minute * 60},{100.0 + minute * 0.1}" for minute in range(0, 20)
    ]
    bars_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    script = os.path.join(REPO_ROOT, "scripts", "dpof_signal_study.py")
    result = subprocess.run(
        [
            sys.executable, script,
            "--signals", str(sig_path),
            "--bars-file", str(bars_path),
            "--thresholds", "0.15",
            "--horizons", "5",
            "--min-gap-minutes", "0",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=30, check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


# ─────────────────────────────────────────────────────────────────────────
# filter_by_date
# ─────────────────────────────────────────────────────────────────────────

def test_filter_by_date_since_and_until():
    records = _load([
        _rec("SPY", 0, 0.2),                    # 2026-09-01
        _rec("SPY", 24 * 60, 0.2),              # 2026-09-02
        _rec("SPY", 2 * 24 * 60, 0.2),           # 2026-09-03
    ])
    out = dss.filter_by_date(records, since="2026-09-02", until="2026-09-02")
    assert len(out) == 1


def test_filter_by_date_no_bounds_is_identity():
    records = _load([_rec("SPY", 0, 0.2)])
    assert dss.filter_by_date(records, None, None) == records


# ─────────────────────────────────────────────────────────────────────────
# Pass-2 review follow-ups (2026-09-26)
# ─────────────────────────────────────────────────────────────────────────

def test_find_events_exact_threshold_fires():
    recs = _load([_rec("SPY", 0, 0.10), _rec("SPY", 0.5, 0.15)])
    events = dss.find_events(recs, threshold=0.15, min_gap_minutes=15)
    assert len(events) == 1 and events[0].composite == 0.15


def test_forward_return_across_the_close_is_no_data_not_next_open():
    # Event at 15:55 ET; bars stop at 16:00 and resume 09:30 next day.
    close_evt = _BASE.replace(hour=15, minute=55)
    bars = [{"t": _epoch(close_evt) - 60 * k, "c": 100.0} for k in range(1, 6)]
    bars += [{"t": _epoch(close_evt.replace(hour=15, minute=55 + k)), "c": 100.0} for k in range(0, 5)]
    next_open = (close_evt + timedelta(days=1)).replace(hour=9, minute=30)
    bars += [{"t": _epoch(next_open) + 60 * k, "c": 150.0} for k in range(0, 120)]
    rows = dss.forward_returns([_event(ts=_epoch(close_evt))], {"SPY": bars}, horizons=(30,))
    assert rows[0]["r_bps"] is None


def test_time_span_is_printed_in_eastern_time():
    assert dss._iso(datetime(2026, 9, 2, 13, 30, tzinfo=timezone.utc).timestamp()) == "2026-09-02T09:30:00-04:00"


def test_baseline_section_says_samples_overlap():
    recs = _load([_rec("SPY", m * 0.5, 0.05) for m in range(10)])
    bars = [_bar(60 * k, 100.0 + k) for k in range(-5, 90)]
    text = dss.render_report(dss.build_report(
        recs, thresholds=(0.15,), horizons=(5,), min_gap_minutes=15,
        bars_by_symbol={"SPY": bars},
    ))
    assert "overlapping" in text


def test_et_day_windows_split_at_eastern_midnight():
    start = datetime(2026, 9, 2, 13, 30, tzinfo=timezone.utc)
    end = datetime(2026, 9, 4, 20, 0, tzinfo=timezone.utc)
    wins = dss.et_day_windows(start, end)
    assert [(a.isoformat(), b.isoformat()) for a, b in wins] == [
        ("2026-09-02T13:30:00+00:00", "2026-09-03T04:00:00+00:00"),
        ("2026-09-03T04:00:00+00:00", "2026-09-04T04:00:00+00:00"),
        ("2026-09-04T04:00:00+00:00", "2026-09-04T20:00:00+00:00"),
    ]
    assert dss.et_day_windows(end, start) == []


class _StatusError(Exception):
    def __init__(self, status):
        super().__init__(f"http {status}")
        self.status_code = status


class _FakeBar:
    def __init__(self, epoch, close):
        self.timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
        self.close = close


class _FakeBarsClient:
    """Records every request; `fail` maps (feed value, call index) -> status."""

    def __init__(self, fail=None):
        self.requests = []
        self.fail = fail or {}

    def get_stock_bars(self, req):
        idx = len(self.requests)
        feed = getattr(req.feed, "value", req.feed)
        self.requests.append((req.start, req.end, feed))
        status = self.fail.get((feed, idx))
        if status is not None:
            raise _StatusError(status)
        return {req.symbol_or_symbols: [_FakeBar(req.start.timestamp() + 60, 101.0)]}


def test_fetch_bars_requests_one_window_per_et_day():
    pytest.importorskip("alpaca")
    from alpaca.data.enums import DataFeed

    client = _FakeBarsClient()
    start = datetime(2026, 9, 2, 13, 30, tzinfo=timezone.utc)
    end = datetime(2026, 9, 4, 20, 0, tzinfo=timezone.utc)
    out = dss.fetch_bars_from_alpaca(["SPY"], start, end, client=client, feed=DataFeed.SIP)
    assert len(client.requests) == 3
    assert len(out["SPY"]) == 3


def test_fetch_bars_falls_back_to_iex_only_on_401_403_and_sticks():
    pytest.importorskip("alpaca")
    from alpaca.data.enums import DataFeed

    start = datetime(2026, 9, 2, 13, 30, tzinfo=timezone.utc)
    end = datetime(2026, 9, 3, 20, 0, tzinfo=timezone.utc)
    client = _FakeBarsClient(fail={("sip", 0): 403})
    out = dss.fetch_bars_from_alpaca(["SPY"], start, end, client=client, feed=DataFeed.SIP)
    assert [r[2] for r in client.requests] == ["sip", "iex", "iex"]
    assert len(out["SPY"]) == 2

    client = _FakeBarsClient(fail={("sip", 0): 500})
    out = dss.fetch_bars_from_alpaca(["SPY"], start, end, client=client, feed=DataFeed.SIP)
    assert [r[2] for r in client.requests] == ["sip", "sip"]
    assert len(out["SPY"]) == 1
