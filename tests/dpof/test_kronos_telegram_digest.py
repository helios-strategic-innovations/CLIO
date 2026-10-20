"""Tests for Task K2's (agent/kronos -- docs/design/clio_kronos.md §4)
Telegram + digest surfaces:

  - agent.dpof.telegram_commands.dpof_command's "kronos" subcommand (plain
    text, off-loop, same contract as "why"/"lessons"/"stats").
  - agent.dpof.readmodel.kronos_text (the body it reads) against the REAL
    agent.kronos.state contract.
  - agent.dpof.readmodel._kronos_digest_line + its one-line appearance in
    eod_digest_text().
"""

from __future__ import annotations

import asyncio
import threading
import types

import agent.dpof.readmodel as readmodel
import agent.dpof.telegram_commands as tc_mod
import agent.kronos.state as kronos_state

from tests.dpof.test_telegram_commands import FakeContext, FakeEngine, FakeUpdate, _patch_get_engine, _patch_telegram_bot_module


def _point_kronos_files(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(tmp_path / "kronos_forecast.json"))
    monkeypatch.setenv("CLIO_KRONOS_SCORES_FILE", str(tmp_path / "kronos_scores.json"))
    with kronos_state._cache_lock:
        kronos_state._cache.clear()


def _write_forecast(now_epoch, symbols):
    kronos_state.write_json_atomic(kronos_state.forecast_path(), {
        "schema": kronos_state.SCHEMA, "written_at": now_epoch,
        "model": {"model": "m", "model_revision": "r", "tokenizer": "t", "tokenizer_revision": "y", "vendor_commit": "c"},
        "params": {"bar_minutes": 5, "lookback_bars": 400, "sample_paths": 16, "temperature": 1.0, "top_p": 0.9},
        "symbols": symbols,
    })


def _write_scores(symbols):
    kronos_state.write_json_atomic(kronos_state.scores_path(), {
        "schema": kronos_state.SCHEMA, "written_at": 1_800_000_000.0, "symbols": symbols,
    })


def _spy_forecast_entry(asof="2026-09-25T12:00:00-04:00"):
    return {
        "status": "ok", "error": None, "asof_bar": asof, "anchor_close": 580.0,
        "feed": "iex", "latency_ms": 100,
        "horizons": {
            "15": {"minutes": 15, "p_up": 0.6, "mean_return_pct": 0.05, "median_return_pct": 0.04,
                   "q10_return_pct": -0.2, "q90_return_pct": 0.3, "expected_abs_move_pct": 0.12},
            "30": {"minutes": 30, "p_up": 0.58, "mean_return_pct": 0.09, "median_return_pct": 0.08,
                   "q10_return_pct": -0.35, "q90_return_pct": 0.55, "expected_abs_move_pct": 0.2},
            "60": {"minutes": 60, "p_up": 0.55, "mean_return_pct": 0.14, "median_return_pct": 0.12,
                   "q10_return_pct": -0.6, "q90_return_pct": 0.9, "expected_abs_move_pct": 0.35},
        },
    }


def _spy_scores_all_metrics(n=48, hit_rate=0.62):
    metrics = {"n": n, "hit_rate": hit_rate, "n_confident": 20, "hit_rate_confident": 0.70,
               "brier": 0.21, "mae_abs_move_pct": 0.08, "since": "2026-09-01"}
    return {h: {"all": metrics, "recent": metrics} for h in kronos_state.HORIZONS}


# ═══════════════════════════════════════════════════════════════════════════
#  readmodel.kronos_text
# ═══════════════════════════════════════════════════════════════════════════

def test_kronos_text_unavailable_when_no_forecast_file(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    monkeypatch.setattr("agent.dpof.config.load_dpof_config", lambda: cfg)
    text = readmodel.kronos_text(now_epoch=1_800_000_000.0)
    assert "SPY" in text
    assert "unavailable" in text.lower()


def test_kronos_text_renders_fresh_forecast_and_track_record(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    now_epoch = 1_800_000_000.0
    _write_forecast(now_epoch, {"SPY": _spy_forecast_entry()})
    _write_scores({"SPY": _spy_scores_all_metrics()})
    monkeypatch.setattr("agent.dpof.config.load_dpof_config", lambda: cfg)

    text = readmodel.kronos_text(now_epoch=now_epoch)
    assert "SPY (as of 2026-09-25T12:00:00-04:00)" in text
    assert "30min: p_up=58%" in text
    assert "n=48 hit=62%" in text


def test_kronos_text_agent_kronos_not_installed(cfg, monkeypatch, tmp_path):
    import sys

    import agent.kronos as kronos_pkg

    _point_kronos_files(monkeypatch, tmp_path)
    monkeypatch.setattr("agent.dpof.config.load_dpof_config", lambda: cfg)
    monkeypatch.delattr(kronos_pkg, "state", raising=False)
    monkeypatch.setitem(sys.modules, "agent.kronos.state", None)

    text = readmodel.kronos_text(now_epoch=1_800_000_000.0)
    assert "unavailable" in text.lower()


def test_kronos_text_never_raises_when_config_unavailable(monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr("agent.dpof.config.load_dpof_config", _boom)
    text = readmodel.kronos_text(now_epoch=1_800_000_000.0)  # must not raise
    assert isinstance(text, str)


# ═══════════════════════════════════════════════════════════════════════════
#  readmodel._kronos_digest_line + eod_digest_text integration
# ═══════════════════════════════════════════════════════════════════════════

def test_kronos_digest_line_none_when_no_scores(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    monkeypatch.setattr("agent.dpof.config.load_dpof_config", lambda: cfg)
    assert readmodel._kronos_digest_line() is None


def test_kronos_digest_line_present_when_scores_exist(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    _write_scores({"SPY": _spy_scores_all_metrics(n=48, hit_rate=0.62)})
    monkeypatch.setattr("agent.dpof.config.load_dpof_config", lambda: cfg)

    line = readmodel._kronos_digest_line()
    assert line is not None
    assert "SPY 30m hit 62% (n=48)" in line


def test_kronos_digest_line_omits_zero_n_symbols(cfg, monkeypatch, tmp_path):
    """A symbol with an all-null/zero-n scores bucket contributes nothing to
    the line (never a fabricated 0%/n=0 entry)."""
    from dataclasses import replace

    c = replace(cfg, underlyings=("SPY",))
    _point_kronos_files(monkeypatch, tmp_path)
    zero_metrics = {"n": 0, "hit_rate": None, "n_confident": 0, "hit_rate_confident": None,
                     "brier": None, "mae_abs_move_pct": None, "since": None}
    _write_scores({"SPY": {h: {"all": zero_metrics, "recent": zero_metrics} for h in kronos_state.HORIZONS}})
    monkeypatch.setattr("agent.dpof.config.load_dpof_config", lambda: c)

    assert readmodel._kronos_digest_line() is None


def test_eod_digest_includes_kronos_line_when_scores_exist(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    _write_scores({"SPY": _spy_scores_all_metrics(n=48, hit_rate=0.62)})
    monkeypatch.setattr("agent.dpof.config.load_dpof_config", lambda: cfg)

    # A minimal "today had cycles" day-stats file so eod_digest_text doesn't
    # short-circuit on "no signal cycles were recorded today".
    import json
    import os

    today = "2026-09-25"
    os.makedirs(cfg.data_dir, exist_ok=True)
    stats = {"day": today, "cycles_evaluated": 3, "nominations": 0, "decider_pass": 0, "decider_enter": 0,
              "decider_unavailable": 0, "jev_vetoes": 0, "entries": 0, "underlyings": {}}
    with open(os.path.join(cfg.data_dir, "signal_day_stats.json"), "w") as f:
        json.dump(stats, f)

    text = readmodel.eod_digest_text(ddir=cfg.data_dir, today=today)
    assert "Kronos: SPY 30m hit 62% (n=48)" in text


def test_eod_digest_omits_kronos_line_when_no_scores(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)  # no scores file written
    monkeypatch.setattr("agent.dpof.config.load_dpof_config", lambda: cfg)

    import json
    import os

    today = "2026-09-25"
    os.makedirs(cfg.data_dir, exist_ok=True)
    stats = {"day": today, "cycles_evaluated": 3, "nominations": 0, "decider_pass": 0, "decider_enter": 0,
              "decider_unavailable": 0, "jev_vetoes": 0, "entries": 0, "underlyings": {}}
    with open(os.path.join(cfg.data_dir, "signal_day_stats.json"), "w") as f:
        json.dump(stats, f)

    text = readmodel.eod_digest_text(ddir=cfg.data_dir, today=today)
    assert "Kronos:" not in text


# ═══════════════════════════════════════════════════════════════════════════
#  /dpof kronos Telegram subcommand
# ═══════════════════════════════════════════════════════════════════════════

def test_usage_mentions_kronos():
    assert "kronos" in tc_mod._USAGE


def test_kronos_subcommand_reads_readmodel_off_loop(monkeypatch):
    """Same off-loop/plain-text contract as "why"/"lessons"/"stats"."""
    _patch_telegram_bot_module(monkeypatch, {42})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)
    seen = {}

    def _kronos(now_epoch=None):
        seen["thread"] = threading.current_thread() is threading.main_thread()
        return "KRONOS TEXT"

    monkeypatch.setattr(readmodel, "kronos_text", _kronos)
    update = FakeUpdate(42)
    asyncio.run(tc_mod.dpof_command(update, FakeContext(["kronos"])))
    assert update.message.replies == ["KRONOS TEXT"]
    assert seen["thread"] is False
    assert engine.status_calls == 0


def test_kronos_subcommand_unauthorized_is_silent(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, {1})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)
    monkeypatch.setattr(readmodel, "kronos_text", lambda now_epoch=None: "KRONOS TEXT")

    update = FakeUpdate(user_id=999)
    asyncio.run(tc_mod.dpof_command(update, FakeContext(["kronos"])))
    assert update.message.replies == []
