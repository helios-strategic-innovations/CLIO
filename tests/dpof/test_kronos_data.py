"""Tests for agent.dpof.data's Kronos advisory read (Task K2, agent/kronos --
docs/design/clio_kronos.md §4): DPOFDataFeed._fetch_kronos / get_intel_ctx's
"kronos" key, against the REAL agent.kronos.state contract (the finished
K1 file, read only -- never written to from here except via its own
write_json_atomic helper, exactly as the real service would).

Plugin-free (no pytest-asyncio): async code is driven with asyncio.run().
Every Kronos file this suite touches is written under CLIO_KRONOS_FORECAST_FILE/
CLIO_KRONOS_SCORES_FILE pointed at tmp_path (tests/conftest.py's repo-wide
isolation already does this to a per-session temp dir; tests here override it
per-test with their own explicit setenv, which wins).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import agent.dpof.data as data
import agent.kronos.state as kronos_state

pytestmark = pytest.mark.filterwarnings("ignore")


def _run(coro):
    return asyncio.run(coro)


class _DummyBroker:
    """get_intel_ctx() never touches the broker -- placeholder only."""


def _feed(cfg) -> data.DPOFDataFeed:
    return data.DPOFDataFeed(cfg, _DummyBroker())


def _point_kronos_files(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(tmp_path / "kronos_forecast.json"))
    monkeypatch.setenv("CLIO_KRONOS_SCORES_FILE", str(tmp_path / "kronos_scores.json"))
    # state.py caches parsed files by (path, mtime) at module scope -- a
    # fresh path per test (tmp_path) already avoids cross-test contamination,
    # but clear it anyway so a test is never affected by cache state a sibling
    # test left behind for the SAME path (e.g. two tests sharing a tmp_path
    # fixture instance would not, but this is cheap insurance).
    with kronos_state._cache_lock:
        kronos_state._cache.clear()


def _write_forecast(now_epoch: float, symbols: dict, written_at: float | None = None) -> None:
    kronos_state.write_json_atomic(kronos_state.forecast_path(), {
        "schema": kronos_state.SCHEMA,
        "written_at": written_at if written_at is not None else now_epoch,
        "model": {"model": "Kronos-small", "model_revision": "x", "tokenizer": "t", "tokenizer_revision": "y", "vendor_commit": "67b630e"},
        "params": {"bar_minutes": 5, "lookback_bars": 400, "sample_paths": 16, "temperature": 1.0, "top_p": 0.9},
        "symbols": symbols,
    })


def _write_scores(symbols: dict) -> None:
    kronos_state.write_json_atomic(kronos_state.scores_path(), {
        "schema": kronos_state.SCHEMA, "written_at": 1_800_000_000.0, "symbols": symbols,
    })


def _spy_forecast_entry(asof="2026-09-25T12:00:00-04:00") -> dict:
    return {
        "status": "ok", "error": None, "asof_bar": asof, "anchor_close": 580.0,
        "feed": "iex", "latency_ms": 120,
        "horizons": {
            "15": {"minutes": 15, "p_up": 0.62, "mean_return_pct": 0.05, "median_return_pct": 0.04,
                   "q10_return_pct": -0.20, "q90_return_pct": 0.30, "expected_abs_move_pct": 0.12},
            "30": {"minutes": 30, "p_up": 0.58, "mean_return_pct": 0.09, "median_return_pct": 0.08,
                   "q10_return_pct": -0.35, "q90_return_pct": 0.55, "expected_abs_move_pct": 0.20},
            "60": {"minutes": 60, "p_up": 0.55, "mean_return_pct": 0.14, "median_return_pct": 0.12,
                   "q10_return_pct": -0.60, "q90_return_pct": 0.90, "expected_abs_move_pct": 0.35},
        },
    }


def _spy_scores_entry() -> dict:
    metrics_all = {"n": 48, "hit_rate": 0.62, "n_confident": 20, "hit_rate_confident": 0.70,
                   "brier": 0.21, "mae_abs_move_pct": 0.08, "since": "2026-09-01"}
    metrics_recent = {"n": 12, "hit_rate": 0.58, "n_confident": 5, "hit_rate_confident": 0.60,
                      "brier": 0.24, "mae_abs_move_pct": 0.09, "since": "2026-09-20"}
    return {h: {"all": metrics_all, "recent": metrics_recent} for h in kronos_state.HORIZONS}


# ═══════════════════════════════════════════════════════════════════════════
#  Flag off -> key absent entirely
# ═══════════════════════════════════════════════════════════════════════════

def test_kronos_context_off_omits_key_entirely(cfg, monkeypatch, tmp_path):
    from dataclasses import replace

    _point_kronos_files(monkeypatch, tmp_path)
    _write_forecast(1_800_000_000.0, {"SPY": _spy_forecast_entry()})
    c = replace(cfg, kronos_context=False)

    intel = _run(_feed(c).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert "kronos" not in intel


# ═══════════════════════════════════════════════════════════════════════════
#  Missing / stale -> "unavailable" per symbol, never zeros
# ═══════════════════════════════════════════════════════════════════════════

def test_kronos_file_missing_all_unavailable(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)  # never written
    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["kronos"] == {sym: "unavailable" for sym in cfg.underlyings}


def test_kronos_file_stale_all_unavailable(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    now_epoch = 1_800_000_000.0
    _write_forecast(now_epoch, {"SPY": _spy_forecast_entry()}, written_at=now_epoch - cfg.kronos_max_age_seconds - 60)
    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=now_epoch))
    assert intel["kronos"] == {sym: "unavailable" for sym in cfg.underlyings}


def test_kronos_symbol_status_not_ok_is_unavailable(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    now_epoch = 1_800_000_000.0
    _write_forecast(now_epoch, {"SPY": {"status": "no_data", "error": None, "asof_bar": None,
                                          "anchor_close": None, "feed": "iex", "latency_ms": 5, "horizons": {}}})
    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=now_epoch))
    assert intel["kronos"]["SPY"] == "unavailable"


# ═══════════════════════════════════════════════════════════════════════════
#  Fresh data present -> trimmed dict, per-symbol
# ═══════════════════════════════════════════════════════════════════════════

def test_kronos_fresh_forecast_present_for_one_symbol_other_unavailable(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    now_epoch = 1_800_000_000.0
    _write_forecast(now_epoch, {"SPY": _spy_forecast_entry()})

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=now_epoch))
    kronos = intel["kronos"]
    assert kronos["SPY"] != "unavailable"
    assert kronos["SPY"]["asof_bar"] == "2026-09-25T12:00:00-04:00"
    assert kronos["SPY"]["horizons"]["30"]["p_up"] == 0.58
    assert kronos["SPY"]["horizons"]["30"]["mean_return_pct"] == 0.09
    assert kronos["SPY"]["horizons"]["30"]["q10_return_pct"] == -0.35
    assert kronos["SPY"]["horizons"]["30"]["q90_return_pct"] == 0.55
    assert kronos["SPY"]["horizons"]["30"]["expected_abs_move_pct"] == 0.20
    if "QQQ" in cfg.underlyings:
        assert kronos["QQQ"] == "unavailable"


def test_kronos_track_record_trimmed_to_four_fields_per_bucket(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    now_epoch = 1_800_000_000.0
    _write_forecast(now_epoch, {"SPY": _spy_forecast_entry()})
    _write_scores({"SPY": _spy_scores_entry()})

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=now_epoch))
    rec = intel["kronos"]["SPY"]["track_record"]["30"]
    assert set(rec.keys()) == {"all", "recent"}
    assert set(rec["all"].keys()) == {"n", "hit_rate", "hit_rate_confident", "brier"}
    assert rec["all"]["n"] == 48
    assert rec["all"]["hit_rate"] == 0.62
    assert rec["all"]["hit_rate_confident"] == 0.70
    assert rec["all"]["brier"] == 0.21
    assert rec["recent"]["n"] == 12
    # mae_abs_move_pct/since/n_confident are dropped, never leak through
    assert "mae_abs_move_pct" not in rec["all"]
    assert "since" not in rec["all"]


def test_kronos_track_record_absent_horizon_omitted_not_zeroed(cfg, monkeypatch, tmp_path):
    """Scored history exists for 30/60 but not 15 -- "15" is simply absent
    from track_record (never {"n": 0, ...})."""
    _point_kronos_files(monkeypatch, tmp_path)
    now_epoch = 1_800_000_000.0
    _write_forecast(now_epoch, {"SPY": _spy_forecast_entry()})
    metrics = {"n": 10, "hit_rate": 0.5, "n_confident": 2, "hit_rate_confident": 0.5, "brier": 0.25,
               "mae_abs_move_pct": 0.1, "since": "2026-09-01"}
    _write_scores({"SPY": {"30": {"all": metrics, "recent": metrics}, "60": {"all": metrics, "recent": metrics}}})

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=now_epoch))
    track = intel["kronos"]["SPY"]["track_record"]
    assert "15" not in track
    assert "30" in track and "60" in track


def test_kronos_scores_present_but_forecast_missing_still_unavailable(cfg, monkeypatch, tmp_path):
    """A scored record with no fresh forecast file is still "unavailable" --
    the forecast's OWN freshness gate always wins; scores never substitute
    for a live reading."""
    _point_kronos_files(monkeypatch, tmp_path)
    _write_scores({"SPY": _spy_scores_entry()})  # forecast file never written
    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["kronos"]["SPY"] == "unavailable"


# ═══════════════════════════════════════════════════════════════════════════
#  agent.kronos not installed -> unavailable, never raises
# ═══════════════════════════════════════════════════════════════════════════

def test_kronos_module_absent_all_unavailable(cfg, monkeypatch, tmp_path):
    """Simulates agent.kronos not being installed yet (K1 not landed/deployed
    on this host). `from agent.kronos import state` resolves the already-
    cached `state` attribute on the `agent.kronos` package object directly
    (Python's _handle_fromlist skips re-import when hasattr() is already
    true) -- this test's own `import agent.kronos.state as kronos_state` at
    module scope leaves exactly that attribute cached, so it must be cleared
    too, not just sys.modules (same fixture pattern as test_intel_ctx_data.py's
    _isolate_intel_modules)."""
    import sys

    import agent.kronos as kronos_pkg

    _point_kronos_files(monkeypatch, tmp_path)
    monkeypatch.delattr(kronos_pkg, "state", raising=False)
    monkeypatch.setitem(sys.modules, "agent.kronos.state", None)

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["kronos"] == {sym: "unavailable" for sym in cfg.underlyings}


def test_kronos_read_raising_never_raises_caller(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)

    def _boom(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(kronos_state, "read_forecast", _boom)
    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))  # must not raise
    assert intel["kronos"] == {sym: "unavailable" for sym in cfg.underlyings}


# ═══════════════════════════════════════════════════════════════════════════
#  Off-loop dispatch (CLAUDE.md anti-pattern #19)
# ═══════════════════════════════════════════════════════════════════════════

def test_kronos_read_dispatched_off_loop(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    main_thread = threading.current_thread()
    seen_threads = []

    real_read_forecast = kronos_state.read_forecast

    def _spy(*a, **kw):
        seen_threads.append(threading.current_thread())
        return real_read_forecast(*a, **kw)

    monkeypatch.setattr(kronos_state, "read_forecast", _spy)

    _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert len(seen_threads) == 1
    assert seen_threads[0] is not main_thread


# ═══════════════════════════════════════════════════════════════════════════
#  Complete-shape guarantee (updates the pre-existing test_intel_ctx_data.py
#  assertion for the new key; see that file's own test for the full check)
# ═══════════════════════════════════════════════════════════════════════════

def test_kronos_key_present_alongside_other_intel_keys(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert "kronos" in intel
    assert isinstance(intel["kronos"], dict)
    assert set(intel["kronos"].keys()) == set(cfg.underlyings)
