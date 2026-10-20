"""Tests for scripts/decisions_replay_corpus.py.

Fixtures are built through agent.dpof.signal_log.SignalLog's own writer
(pointed at tmp_path) so their shape is exactly what engine.py's
_log_cycle_signals actually produces, per the build task's own instruction
to reuse the log's real writer rather than hand-rolling a divergent shape.

No network, no real database, no GPU, no live `data/` reads or writes:
every --signals/--out path used here is under tmp_path.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.dpof.config import DPOFConfig                        # noqa: E402
from agent.dpof.jev import build_entry_request                  # noqa: E402
from agent.dpof.models import Direction, MarketSnapshot, Signal  # noqa: E402
from agent.dpof.signal_log import SignalLog                     # noqa: E402
from scripts import decisions_replay_corpus as rc                # noqa: E402

ET = ZoneInfo("America/New_York")


# ─────────────────────────────────────────────────────────────────────────
# Fixture builders -- signals.jsonl records shaped exactly like
# engine.py:_log_cycle_signals (see agent/dpof/engine.py ~L2280-2393)
# ─────────────────────────────────────────────────────────────────────────

def _pillars(nofa=0.5, premium=0.3, darkpool=0.2, available=True):
    return {
        "nofa": {"value": nofa, "available": available, "source": "uw:x" if available else ""},
        "premium": {"value": premium, "available": available, "source": "uw:y" if available else ""},
        "darkpool": {"value": darkpool, "available": available, "source": "uw:z" if available else ""},
        "gex": {"value": 0.1, "available": True, "source": "uw:gex"},
    }


def _rec(
    underlying, *, nominated=False, signal_basis=None, direction="bullish", strength="strong",
    pillars=None, ts="2026-09-25T11:15:00.123456-04:00",
) -> dict:
    """One signals.jsonl record, matching engine.py's exact field set."""
    return {
        "ts": ts,
        "underlying": underlying,
        "composite": 0.5,
        "strength": strength,
        "direction": direction,
        "pillars": pillars if pillars is not None else _pillars(),
        "flow_source": "uw",
        "regime": "normal",
        "price_confirm": {"ok": True, "reasons": []},
        "vetoes": [],
        "reasons": [],
        "is_signal_symbol": nominated,
        "signal_basis": signal_basis,
        "nominated": nominated,
        "outcome": "entered" if nominated else "below_threshold",
        "decider": None,
        "jev": None,
    }


def _write_cycle(data_dir, records, now_et, max_lines=20_000):
    sl = SignalLog(str(data_dir))
    asyncio.run(sl.write_cycle(records, now_et, max_lines=max_lines))


def _read_jsonl(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────────────────
# canonical_json / state_hash -- golden vector
# ─────────────────────────────────────────────────────────────────────────

def test_golden_state_hash_vector():
    state = {
        "underlying": "SPY",
        "signals": {"options_flow": "strong bullish", "vix": "normal (15-20)"},
        "headlines": ["Fed holds rates — «steady»"],
        "n": 3,
    }
    assert rc.state_hash(state) == (
        "3df70c3d9e795bb32ecf338c059bcf924bbbb4e8d61b39d4524f376076098eae"
    )


def test_canonical_json_is_sorted_compact_no_ascii_escape():
    assert rc.canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'


# ─────────────────────────────────────────────────────────────────────────
# _parse_ts_et
# ─────────────────────────────────────────────────────────────────────────

def test_parse_ts_et_offset_aware_converts_to_et():
    dt = rc._parse_ts_et("2026-09-25T15:15:00.123456-04:00")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.hour == 15 and dt.minute == 15

    # A UTC timestamp (different offset) must convert, not just re-tag.
    dt_utc = rc._parse_ts_et("2026-09-25T15:15:00+00:00")
    assert dt_utc.hour == 11  # 15:15 UTC == 11:15 ET (EDT, UTC-4) in September


def test_parse_ts_et_naive_is_localized_not_utc():
    dt = rc._parse_ts_et("2026-09-25T11:15:00")
    assert dt is not None
    assert dt.hour == 11 and dt.minute == 15
    assert dt.tzinfo is not None


@pytest.mark.parametrize("bad", [None, "", "not-a-timestamp", 12345, object()])
def test_parse_ts_et_invalid_returns_none(bad):
    assert rc._parse_ts_et(bad) is None


# ─────────────────────────────────────────────────────────────────────────
# _parse_direction
# ─────────────────────────────────────────────────────────────────────────

def test_parse_direction_valid_and_invalid():
    assert rc._parse_direction("bullish") is Direction.BULLISH
    assert rc._parse_direction("bearish") is Direction.BEARISH
    assert rc._parse_direction("neutral") is Direction.NEUTRAL
    assert rc._parse_direction("sideways") is None
    assert rc._parse_direction(None) is None
    assert rc._parse_direction(123) is None


# ─────────────────────────────────────────────────────────────────────────
# _group_cycles
# ─────────────────────────────────────────────────────────────────────────

def test_group_cycles_groups_contiguous_identical_ts():
    parsed = [
        (1, {"ts": "T1", "underlying": "SPY"}),
        (2, {"ts": "T1", "underlying": "QQQ"}),
        (3, {"ts": "T2", "underlying": "SPY"}),
    ]
    groups = list(rc._group_cycles(parsed))
    assert len(groups) == 2
    assert [ln for ln, _ in groups[0]] == [1, 2]
    assert [ln for ln, _ in groups[1]] == [3]


def test_group_cycles_malformed_and_missing_ts_are_singletons():
    parsed = [
        (1, {"ts": "T1", "underlying": "SPY"}),
        (2, None),
        (3, {"underlying": "QQQ"}),  # no ts key at all
        (4, {"ts": "T1", "underlying": "SPY"}),  # same ts as line 1 but NOT contiguous -> own group
    ]
    groups = list(rc._group_cycles(parsed))
    assert [ln for group in groups for ln, _ in group] == [1, 2, 3, 4]
    assert [len(g) for g in groups] == [1, 1, 1, 1]


# ─────────────────────────────────────────────────────────────────────────
# _resolve_primary_pillars
# ─────────────────────────────────────────────────────────────────────────

def test_resolve_primary_pillars_no_divergence_uses_own_record():
    rec = _rec("SPY", nominated=True, signal_basis="SPY", pillars=_pillars(0.4, 0.3, 0.2))
    group = [(1, rec)]
    out = rc._resolve_primary_pillars(rec, group)
    assert out == {"nofa": 0.4, "premium": 0.3, "darkpool": 0.2}


def test_resolve_primary_pillars_divergence_uses_sibling_not_own_record():
    primary_rec = _rec("SPY", nominated=False, signal_basis=None, pillars=_pillars(0.2, 0.1, 0.05))
    traded_rec = _rec("QQQ", nominated=True, signal_basis="SPY", pillars=_pillars(0.9, 0.9, 0.9))
    group = [(1, primary_rec), (2, traded_rec)]
    out = rc._resolve_primary_pillars(traded_rec, group)
    assert out == {"nofa": 0.2, "premium": 0.1, "darkpool": 0.05}  # SPY's own, not QQQ's


def test_resolve_primary_pillars_sibling_missing_returns_none():
    traded_rec = _rec("QQQ", nominated=True, signal_basis="SPY", pillars=_pillars(0.9, 0.9, 0.9))
    group = [(1, traded_rec)]  # SPY sibling never made it into this group
    assert rc._resolve_primary_pillars(traded_rec, group) is None


def test_resolve_primary_pillars_unavailable_pillar_excluded_not_zeroed():
    pillars = _pillars(0.4, 0.3, 0.2)
    pillars["premium"] = {"value": None, "available": False, "source": ""}
    rec = _rec("SPY", nominated=True, signal_basis="SPY", pillars=pillars)
    out = rc._resolve_primary_pillars(rec, [(1, rec)])
    assert "premium" not in out
    assert out == {"nofa": 0.4, "darkpool": 0.2}


def test_resolve_primary_pillars_missing_signal_basis_returns_none():
    rec = _rec("SPY", nominated=True, signal_basis=None)
    assert rc._resolve_primary_pillars(rec, [(1, rec)]) is None


# ─────────────────────────────────────────────────────────────────────────
# build_replay_request -- matches build_entry_request; never fabricates
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> DPOFConfig:
    return DPOFConfig()


def test_build_replay_request_matches_build_entry_request_directly(cfg):
    rec = _rec("SPY", nominated=True, signal_basis="SPY", direction="bullish", strength="strong",
               pillars=_pillars(0.5, 0.3, 0.2))
    request, reason = rc.build_replay_request(rec, 7, [(7, rec)], cfg)
    assert reason is None
    assert request is not None

    now_et = rc._parse_ts_et(rec["ts"])
    signal = Signal(symbol="SPY", direction=Direction.BULLISH, strength="strong",
                     pillars_used={"nofa": 0.5, "premium": 0.3, "darkpool": 0.2})
    snap = MarketSnapshot(asof=now_et.timestamp(), underlyings={}, vix=None, headlines=[])
    expected_state, expected_questions = build_entry_request(signal, snap, now_et, cfg, sizing_ctx=None)
    for k in ("gamma_regime", "price_vs_vwap", "momentum_15m", "volatility"):
        expected_state["signals"].pop(k, None)
    expected_state.pop("headlines", None)

    assert request["state"] == expected_state
    assert request["questions"] == expected_questions


def test_build_replay_request_never_fabricates_omitted_buckets(cfg):
    rec = _rec("SPY", nominated=True, signal_basis="SPY")
    request, reason = rc.build_replay_request(rec, 1, [(1, rec)], cfg)
    assert reason is None
    sig = request["state"]["signals"]
    # Absent entirely -- never "unknown"/"unavailable" placeholders.
    for key in ("gamma_regime", "price_vs_vwap", "momentum_15m", "volatility"):
        assert key not in sig
    assert "headlines" not in request["state"]
    assert "sizing" not in request["state"]
    assert "headline_conflict" not in request["questions"]
    assert "event_risk" not in request["questions"]
    assert "size_reckless" not in request["questions"]
    # What DOES survive is real, reconstructed data, not sentinels either.
    assert sig["options_flow"] == "strongly bullish"  # bucket_pillar(0.5) >= 0.45 threshold


def test_build_replay_request_reports_omitted_fields(cfg):
    rec = _rec("SPY", nominated=True, signal_basis="SPY")
    request, _ = rc.build_replay_request(rec, 1, [(1, rec)], cfg)
    omitted = set(request["producer_ref"]["omitted"])
    assert omitted == {
        "signals.gamma_regime", "signals.price_vs_vwap", "signals.momentum_15m", "signals.volatility",
        "headlines", "sizing",
        "questions.headline_conflict", "questions.event_risk", "questions.size_reckless",
    }


def test_build_replay_request_producer_ref_shape(cfg):
    rec = _rec("SPY", nominated=True, signal_basis="SPY", ts="2026-09-25T11:15:00-04:00")
    request, _ = rc.build_replay_request(rec, 42, [(42, rec)], cfg)
    ref = request["producer_ref"]
    assert ref["source"] == "signals.jsonl"
    assert ref["line"] == 42
    assert ref["ts"] == "2026-09-25T11:15:00-04:00"
    assert ref["symbol"] == "SPY"
    assert request["producer"] == "corpus_replay"


@pytest.mark.parametrize(
    "mutate,expected_reason",
    [
        (lambda r: r.__setitem__("underlying", None), "missing_underlying"),
        (lambda r: r.__setitem__("ts", "garbage"), "missing_or_bad_ts"),
        (lambda r: r.__setitem__("direction", "neutral"), "missing_or_neutral_direction"),
        (lambda r: r.__setitem__("direction", "sideways"), "missing_or_neutral_direction"),
        (lambda r: r.__setitem__("strength", "none"), "missing_strength"),
        (lambda r: r.__setitem__("strength", None), "missing_strength"),
        (lambda r: r.__setitem__("signal_basis", None), "primary_pillars_unavailable"),
    ],
)
def test_build_replay_request_skip_reasons(cfg, mutate, expected_reason):
    rec = _rec("SPY", nominated=True, signal_basis="SPY")
    mutate(rec)
    request, reason = rc.build_replay_request(rec, 1, [(1, rec)], cfg)
    assert request is None
    assert reason == expected_reason


# ─────────────────────────────────────────────────────────────────────────
# process_signal_lines (pure core)
# ─────────────────────────────────────────────────────────────────────────

def _lines_for(records: list[dict]) -> list[str]:
    return [json.dumps(r) + "\n" for r in records]


def test_process_signal_lines_dedups_identical_states(cfg):
    rec1 = _rec("SPY", nominated=True, signal_basis="SPY", ts="2026-09-25T11:15:00-04:00")
    rec2 = _rec("SPY", nominated=True, signal_basis="SPY", ts="2026-09-25T11:15:30-04:00")  # different ts, same bucket -> same state
    requests, summary = rc.process_signal_lines(_lines_for([rec1, rec2]), cfg)
    assert summary["emitted"] == 1
    assert summary["deduped"] == 1
    assert len(requests) == 1


def test_process_signal_lines_existing_hashes_seed_dedup(cfg):
    rec = _rec("SPY", nominated=True, signal_basis="SPY")
    requests, summary = rc.process_signal_lines(_lines_for([rec]), cfg)
    h = rc.state_hash(requests[0]["state"])

    requests2, summary2 = rc.process_signal_lines(_lines_for([rec]), cfg, existing_hashes={h})
    assert requests2 == []
    assert summary2["deduped"] == 1
    assert summary2["existing_hashes_loaded"] == 1


def test_process_signal_lines_since_filter(cfg):
    old = _rec("SPY", nominated=True, signal_basis="SPY", ts="2026-09-01T11:15:00-04:00")
    new = _rec("SPY", nominated=True, signal_basis="SPY", ts="2026-09-25T11:15:00-04:00", strength="normal")
    from datetime import date
    requests, summary = rc.process_signal_lines(_lines_for([old, new]), cfg, since=date(2026, 9, 10))
    assert summary["since_filtered"] == 1
    assert summary["emitted"] == 1
    assert requests[0]["producer_ref"]["ts"] == new["ts"]


def test_process_signal_lines_limit_caps_emitted(cfg):
    recs = [
        _rec("SPY", nominated=True, signal_basis="SPY", ts=f"2026-09-25T11:{15+i}:00-04:00", strength="normal" if i else "strong")
        for i in range(3)
    ]
    requests, summary = rc.process_signal_lines(_lines_for(recs), cfg, limit=1)
    assert summary["emitted"] == 1
    assert summary["limit_reached"] is True


def test_process_signal_lines_non_nominated_records_are_not_candidates(cfg):
    rec = _rec("SPY", nominated=False, signal_basis=None)
    requests, summary = rc.process_signal_lines(_lines_for([rec]), cfg)
    assert summary["candidates"] == 0
    assert summary["emitted"] == 0
    assert summary["skipped"] == {}


def test_process_signal_lines_malformed_line_counted_not_candidate(cfg):
    lines = ["not json {{{\n", json.dumps(_rec("SPY", nominated=True, signal_basis="SPY")) + "\n"]
    requests, summary = rc.process_signal_lines(lines, cfg)
    assert summary["malformed_lines"] == 1
    assert summary["records_read"] == 1
    assert summary["emitted"] == 1


def test_process_signal_lines_skip_counts_by_reason(cfg):
    bad1 = _rec("SPY", nominated=True, signal_basis="SPY")
    bad1["direction"] = "neutral"
    bad2 = _rec("QQQ", nominated=True, signal_basis="QQQ")
    bad2["strength"] = "none"
    requests, summary = rc.process_signal_lines(_lines_for([bad1, bad2]), cfg)
    assert summary["skipped"] == {"missing_or_neutral_direction": 1, "missing_strength": 1}
    assert summary["emitted"] == 0


# ─────────────────────────────────────────────────────────────────────────
# CLI (main) -- file I/O behavior
# ─────────────────────────────────────────────────────────────────────────

def test_cli_end_to_end_writes_expected_line(tmp_path):
    now_et = datetime(2026, 9, 25, 11, 15, tzinfo=ET)
    data_dir = tmp_path / "dpof"
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY")], now_et)
    out = tmp_path / "out.jsonl"

    rv = rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(out)])
    assert rv == 0
    lines = _read_jsonl(out)
    assert len(lines) == 1
    assert lines[0]["producer"] == "corpus_replay"
    assert lines[0]["producer_ref"]["symbol"] == "SPY"


def test_cli_refuses_to_overwrite_existing_out(tmp_path):
    now_et = datetime(2026, 9, 25, 11, 15, tzinfo=ET)
    data_dir = tmp_path / "dpof"
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY")], now_et)
    out = tmp_path / "out.jsonl"
    out.write_text('{"sentinel": true}\n')

    rv = rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(out)])
    assert rv != 0
    # untouched
    assert out.read_text() == '{"sentinel": true}\n'


def test_cli_append_adds_to_existing_out(tmp_path):
    now_et = datetime(2026, 9, 25, 11, 15, tzinfo=ET)
    data_dir = tmp_path / "dpof"
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY", strength="strong")], now_et)
    out = tmp_path / "out.jsonl"

    rv1 = rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(out)])
    assert rv1 == 0
    assert len(_read_jsonl(out)) == 1

    now_et2 = datetime(2026, 9, 25, 11, 45, tzinfo=ET)
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY", strength="normal")], now_et2, max_lines=20_000)
    rv2 = rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(out), "--append"])
    assert rv2 == 0
    lines = _read_jsonl(out)
    assert len(lines) == 2  # the first (strong, from cycle 1) plus the new (normal, from cycle 2)


def test_cli_append_dedups_against_existing_file(tmp_path):
    now_et = datetime(2026, 9, 25, 11, 15, tzinfo=ET)
    data_dir = tmp_path / "dpof"
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY")], now_et)
    out = tmp_path / "out.jsonl"

    rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(out)])
    assert len(_read_jsonl(out)) == 1

    # Re-running --append against the SAME signals file must not duplicate.
    rv = rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(out), "--append"])
    assert rv == 0
    assert len(_read_jsonl(out)) == 1


def test_cli_dry_run_never_writes_out(tmp_path, capsys):
    now_et = datetime(2026, 9, 25, 11, 15, tzinfo=ET)
    data_dir = tmp_path / "dpof"
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY")], now_et)
    out = tmp_path / "out.jsonl"

    rv = rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(out), "--dry-run"])
    assert rv == 0
    assert not out.exists()
    captured = capsys.readouterr()
    assert "emitted: 1" in captured.out


def test_cli_dry_run_does_not_refuse_when_out_already_exists(tmp_path):
    now_et = datetime(2026, 9, 25, 11, 15, tzinfo=ET)
    data_dir = tmp_path / "dpof"
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY")], now_et)
    out = tmp_path / "out.jsonl"
    out.write_text('{"sentinel": true}\n')

    rv = rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(out), "--dry-run"])
    assert rv == 0
    assert out.read_text() == '{"sentinel": true}\n'  # untouched


def test_cli_since_and_limit_flags(tmp_path):
    data_dir = tmp_path / "dpof"
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY", ts="2026-09-01T11:15:00-04:00")],
                 datetime(2026, 9, 1, 11, 15, tzinfo=ET))
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY", ts="2026-09-25T11:15:00-04:00", strength="normal")],
                 datetime(2026, 9, 25, 11, 15, tzinfo=ET))
    out = tmp_path / "out.jsonl"

    rv = rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(out), "--since", "2026-09-10"])
    assert rv == 0
    lines = _read_jsonl(out)
    assert len(lines) == 1
    assert lines[0]["producer_ref"]["ts"].startswith("2026-09-25")


def test_cli_missing_signals_file_errors(tmp_path):
    rv = rc.main(["--signals", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "out.jsonl")])
    assert rv != 0
    assert not (tmp_path / "out.jsonl").exists()


def test_cli_bad_since_format_errors(tmp_path):
    now_et = datetime(2026, 9, 25, 11, 15, tzinfo=ET)
    data_dir = tmp_path / "dpof"
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY")], now_et)
    rv = rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(tmp_path / "out.jsonl"), "--since", "09/25/2026"])
    assert rv != 0


def test_cli_writes_nowhere_except_out(tmp_path):
    """The script must never write anywhere except --out (per the build task)."""
    now_et = datetime(2026, 9, 25, 11, 15, tzinfo=ET)
    data_dir = tmp_path / "dpof"
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY")], now_et)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    out = out_dir / "out.jsonl"

    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    rc.main(["--signals", str(data_dir / "signals.jsonl"), "--out", str(out)])
    after = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert after - before == {out}


# ─────────────────────────────────────────────────────────────────────────
# Import isolation -- fresh subprocess (CLAUDE.md "DPOF Follow-ups Gotchas":
# import-isolation tests probe a fresh interpreter, never the shared pytest
# process's sys.modules)
# ─────────────────────────────────────────────────────────────────────────

def test_cli_runs_keyless_in_fresh_subprocess(tmp_path):
    """Proves the whole import chain (agent.dpof.config/jev/models) needs no
    TYPESAFE_*/ANTHROPIC_*/any key at import time -- no load_dotenv() is
    called anywhere in this script on purpose. A stripped environment must
    still run --dry-run cleanly end to end."""
    now_et = datetime(2026, 9, 25, 11, 15, tzinfo=ET)
    data_dir = tmp_path / "dpof"
    _write_cycle(data_dir, [_rec("SPY", nominated=True, signal_basis="SPY")], now_et)
    out = tmp_path / "out.jsonl"

    script = os.path.join(REPO_ROOT, "scripts", "decisions_replay_corpus.py")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/root")}
    result = subprocess.run(
        [sys.executable, script, "--signals", str(data_dir / "signals.jsonl"), "--out", str(out), "--dry-run"],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "emitted: 1" in result.stdout
    assert not out.exists()



# ─────────────────────────────────────────────────────────────────────────
# record["jev_words"] (logged since Kronos PR #74): restored, never invented
# ─────────────────────────────────────────────────────────────────────────

from agent.dpof import jev as _jev                                # noqa: E402
from agent.dpof.models import UnderlyingSnapshot                  # noqa: E402

_WORD_KEYS = ("gamma_regime", "price_vs_vwap", "momentum_15m", "volatility")


def _engine_words(u, vix) -> dict:
    """What engine.py:_log_cycle_signals logs (tests/dpof/test_kronos_engine.py
    pins that it uses exactly these jev.py helpers)."""
    return {
        "gamma_regime": _jev._gamma_regime_words(u),
        "price_vs_vwap": _jev._price_vs_vwap_words(u),
        "momentum_15m": _jev._momentum_words(u),
        "volatility": _jev.bucket_vix(vix),
    }


def test_logged_words_make_the_replay_state_equal_production(cfg):
    """End-to-end oracle: the replayed state is exactly what build_entry_request
    showed Jev from the real snapshot, minus only the never-logged headlines."""
    u = UnderlyingSnapshot(symbol="SPY", gex_regime="ACCELERATING", last_price=501.0, vwap=500.0, ret_15m=-0.002)
    vix = 22.4
    rec = _rec("SPY", nominated=True, signal_basis="SPY", direction="bearish", strength="normal",
               pillars=_pillars(-0.5, -0.3, 0.1))
    rec["jev_words"] = _engine_words(u, vix)

    request, reason = rc.build_replay_request(rec, 3, [(3, rec)], cfg)
    assert reason is None

    now_et = rc._parse_ts_et(rec["ts"])
    signal = Signal(symbol="SPY", direction=Direction.BEARISH, strength="normal",
                    pillars_used={"nofa": -0.5, "premium": -0.3, "darkpool": 0.1})
    snap = MarketSnapshot(asof=now_et.timestamp(), underlyings={"SPY": u}, vix=vix, headlines=[])
    expected_state, expected_questions = build_entry_request(signal, snap, now_et, cfg, sizing_ctx=None)
    expected_state.pop("headlines", None)

    assert request["state"] == expected_state
    assert request["questions"] == expected_questions
    assert request["state"]["signals"]["gamma_regime"] == "accelerating"
    assert request["state"]["signals"]["volatility"] == "elevated (20-25)"
    assert set(request["producer_ref"]["omitted"]) == {
        "headlines", "sizing", "questions.headline_conflict", "questions.event_risk", "questions.size_reckless",
    }


def test_divergence_trade_uses_the_traded_symbols_own_words(cfg):
    """build_entry_request reads snap.underlyings[signal.symbol] (the TRADED
    symbol), so the words come from that record, never the primary sibling."""
    ts = "2026-09-25T11:15:00-04:00"
    spy = _rec("SPY", ts=ts)
    spy["jev_words"] = {"gamma_regime": "stabilizing", "price_vs_vwap": "below",
                        "momentum_15m": "falling", "volatility": "calm (below 15)"}
    qqq = _rec("QQQ", nominated=True, signal_basis="SPY", ts=ts)
    qqq["jev_words"] = {"gamma_regime": "accelerating", "price_vs_vwap": "above",
                        "momentum_15m": "rising", "volatility": "calm (below 15)"}
    group = [(1, spy), (2, qqq)]
    request, reason = rc.build_replay_request(qqq, 2, group, cfg)
    assert reason is None
    sig = request["state"]["signals"]
    assert (sig["gamma_regime"], sig["price_vs_vwap"], sig["momentum_15m"]) == ("accelerating", "above", "rising")


def test_unknown_is_kept_but_missing_or_malformed_words_are_omitted(cfg):
    rec = _rec("SPY", nominated=True, signal_basis="SPY")
    rec["jev_words"] = {
        "gamma_regime": "unknown",            # production really showed Jev "unknown": a fact, kept
        "price_vs_vwap": "Above",             # wrong case: not a shape jev.py emits
        "momentum_15m": "rising\n",           # trailing newline must not slip past the check
        # "volatility" missing entirely
    }
    request, reason = rc.build_replay_request(rec, 1, [(1, rec)], cfg)
    assert reason is None
    sig = request["state"]["signals"]
    assert sig["gamma_regime"] == "unknown"
    for key in ("price_vs_vwap", "momentum_15m", "volatility"):
        assert key not in sig
    omitted = set(request["producer_ref"]["omitted"])
    assert {"signals.price_vs_vwap", "signals.momentum_15m", "signals.volatility"} <= omitted
    assert "signals.gamma_regime" not in omitted

    for junk in (None, [], "words", {"gamma_regime": 5}):
        rec["jev_words"] = junk
        request, _ = rc.build_replay_request(rec, 1, [(1, rec)], cfg)
        assert not set(_WORD_KEYS) & set(request["state"]["signals"])


def test_every_word_jev_emits_passes_the_shape_check():
    """Parity oracle for _JEV_WORD_RE: if a jev.py helper ever emits a new
    shape, this fails instead of the replay silently omitting real words."""
    words = {_jev.bucket_vix(v) for v in (None, 10, 15, 19.9, 20, 24.9, 25, 34.9, 35, 80)}
    for regime in (None, "", "ACCELERATING", "STABILIZING", "NEUTRAL", "some_new_label"):
        for pin in (None, False, True):
            words.add(_jev._gamma_regime_words(UnderlyingSnapshot(symbol="SPY", gex_regime=regime, pin_risk=pin)))
    for price, vwap in ((None, 1.0), (1.0, None), (2.0, 1.0), (1.0, 2.0), (1.0, 1.0)):
        words.add(_jev._price_vs_vwap_words(UnderlyingSnapshot(symbol="SPY", last_price=price, vwap=vwap)))
    for ret in (None, 0.1, -0.1, 0.0):
        words.add(_jev._momentum_words(UnderlyingSnapshot(symbol="SPY", ret_15m=ret)))
    assert words, "no words generated"
    assert [w for w in words if not rc._JEV_WORD_RE.fullmatch(w)] == []


def test_summary_counts_omissions_only_for_records_without_words(cfg):
    old = _rec("SPY", nominated=True, signal_basis="SPY", ts="2026-09-25T11:15:00-04:00")
    new = _rec("SPY", nominated=True, signal_basis="SPY", ts="2026-09-25T11:20:00-04:00", strength="normal")
    new["jev_words"] = {"gamma_regime": "neutral", "price_vs_vwap": "above",
                        "momentum_15m": "flat", "volatility": "normal (15-20)"}
    lines = [json.dumps(old), json.dumps(new)]
    requests, summary = rc.process_signal_lines(lines, cfg)
    assert summary["emitted"] == 2
    counts = summary["omitted_field_counts"]
    for key in _WORD_KEYS:
        assert counts[f"signals.{key}"] == 1
    assert counts["headlines"] == 2

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
