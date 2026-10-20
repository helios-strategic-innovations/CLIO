"""Tests for the engine.py half of Task K2 (agent/kronos -- docs/design/
clio_kronos.md §4): _build_intel_ctx's "kronos" merge, _build_ctx's
conditional inclusion (absent when disabled/unset), _log_cycle_signals's
"kronos" + "jev_words" additive record fields, and the K2 ruling's isolation
guarantee -- composite/nomination/Jev request/veto outcome are byte-identical
whether or not Kronos forecast files are present (agent.dpof.signals/jev
never read agent.kronos.state at all; this test PINS that contract).

Reuses test_engine.py's/test_engine_judgment.py's fakes and helpers, same
pattern test_intel_ctx_engine.py already follows.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace

import pytest

import agent.kronos.state as kronos_state
from agent.dpof import data as data_module
from agent.dpof import jev as jev_module
from agent.dpof import signals as signals_module
from agent.dpof.config import DpofMode, JevMode
from agent.dpof.engine import DPOFEngine
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import MarketSnapshot, PillarReading, Regime, Signal, UnderlyingSnapshot

from tests.dpof.test_engine import (
    FakeBroker,
    FakeOrderManager,
    _base_dt,
    _bullish_snapshot,
    _bullish_underlying,
)
from tests.dpof.test_engine_judgment import _engine
from tests.dpof.test_engine_signal_log import _base_paper_cfg, _read_signals

pytestmark = pytest.mark.filterwarnings("ignore")


def _run(coro):
    return asyncio.run(coro)


class FakeIntelFeed:
    def __init__(self, intel=None):
        self.intel = intel if intel is not None else {}

    async def get_intel_ctx(self, now_epoch=None):
        return self.intel


class FakeFeedWithRealKronos:
    """A market-data FakeFeed (real snapshot_fn, no other intel) whose
    get_intel_ctx() carries a REAL "kronos" read via the tested
    data._kronos_ctx_read_sync helper against `cfg` -- everything else in
    the intel bundle is the standard "unavailable"/empty complete shape, so
    only the kronos wiring under test is real."""

    def __init__(self, snapshot_fn, cfg):
        self.cfg = cfg
        self.snapshot_fn = snapshot_fn
        self.closed = False

    def update_config(self, cfg):
        self.cfg = cfg

    async def snapshot(self, now_epoch):
        return self.snapshot_fn(now_epoch)

    async def close(self):
        self.closed = True

    async def get_intel_ctx(self, now_epoch=None):
        import asyncio as _asyncio

        loop = _asyncio.get_running_loop()
        kronos = None
        if self.cfg.kronos_context:
            kronos = await loop.run_in_executor(
                None, data_module._kronos_ctx_read_sync, tuple(self.cfg.underlyings),
                self.cfg.kronos_max_age_seconds, now_epoch,
            )
        out = {
            "events": {"upcoming": [], "recent_surprises": [], "source": "unavailable"},
            "vol_surface": {sym: None for sym in self.cfg.underlyings},
            "breaking": {"active": False, "max_severity": 0, "events": [], "as_of": None, "stale": True, "source": "unavailable"},
            "calendar_status": {"status": "missing", "age_hours": None, "sources": []},
        }
        if kronos is not None:
            out["kronos"] = kronos
        return out


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


def _spy_forecast_entry():
    return {
        "status": "ok", "error": None, "asof_bar": "2026-09-25T12:00:00-04:00", "anchor_close": 580.0,
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


# ─────────────────────────────────────────────────────────────────────────
# _build_intel_ctx merges "kronos"
# ─────────────────────────────────────────────────────────────────────────

def test_build_intel_ctx_merges_kronos_from_feed(cfg):
    intel = {
        "events": {"upcoming": [], "recent_surprises": [], "source": "x"},
        "vol_surface": {}, "breaking": {"active": False, "max_severity": 0, "events": [], "as_of": None, "stale": True, "source": "x"},
        "calendar_status": {"status": "ok", "age_hours": 0.1, "sources": []},
        "kronos": {"SPY": {"asof_bar": "x", "horizons": {}, "track_record": {}}},
    }
    eng = _engine(cfg, feed=FakeIntelFeed(intel))
    snap = MarketSnapshot(asof=0.0, vix=16.0, vix_age_seconds=10.0)
    out = _run(eng._build_intel_ctx(cfg, snap, 1000.0))
    assert out["kronos"] == intel["kronos"]


def test_build_intel_ctx_kronos_none_when_feed_omits_it(cfg):
    """A feed output with no "kronos" key at all (kronos disabled, or an
    older/partial feed) -> _build_intel_ctx's own "complete shape" default
    for kronos is None, never a fabricated per-symbol dict."""
    intel = {"events": {"upcoming": [], "recent_surprises": [], "source": "x"}}
    eng = _engine(cfg, feed=FakeIntelFeed(intel))
    out = _run(eng._build_intel_ctx(cfg, None, 1000.0))
    assert out["kronos"] is None


def test_build_intel_ctx_kronos_none_when_no_feed(cfg):
    eng = _engine(cfg, feed=None)
    out = _run(eng._build_intel_ctx(cfg, None, 1000.0))
    assert out["kronos"] is None


# ─────────────────────────────────────────────────────────────────────────
# _build_ctx: "kronos" present only when intel actually carries one
# ─────────────────────────────────────────────────────────────────────────

def _call_build_ctx(eng, cfg, snap=None):
    from agent.dpof.models import RiskState

    return eng._build_ctx(
        now_et=_base_dt(), cfg=cfg, snap=snap, underlyings_ctx={}, market_regime=None,
        risk=RiskState(), open_positions=[], dpof_equity=10_000.0, open_premium=0.0,
    )


def test_build_ctx_includes_kronos_when_present(cfg):
    eng = _engine(cfg)
    eng._last_intel_ctx = {"kronos": {"SPY": "unavailable"}}
    ctx = _call_build_ctx(eng, cfg)
    assert ctx["kronos"] == {"SPY": "unavailable"}


def test_build_ctx_omits_kronos_when_last_intel_ctx_has_none(cfg):
    """Matches disabled/no-cycle-yet: intel.get("kronos") is None ->
    "kronos" is ABSENT from the decider ctx entirely (never a "kronos": None
    key either -- the decider prompt renderer treats key-absence, not a None
    value, as the "omit this section" signal)."""
    eng = _engine(cfg)
    eng._last_intel_ctx = {}
    ctx = _call_build_ctx(eng, cfg)
    assert "kronos" not in ctx


def test_build_ctx_omits_kronos_when_intel_ctx_explicitly_none(cfg):
    eng = _engine(cfg)
    eng._last_intel_ctx = {"kronos": None, "events": {}}
    ctx = _call_build_ctx(eng, cfg)
    assert "kronos" not in ctx


def test_build_ctx_kronos_present_for_review_ctx_too(cfg):
    """_build_ctx is the SAME builder used for entries and reviews (see
    engine.py's _run_entry_debate_then_decide / _run_review_debate_then_decide,
    both of which pass their `ctx` straight from one _build_ctx call) -- a
    single assertion on _build_ctx's own output covers both call sites; this
    test just pins that there is no separate/divergent review-ctx builder
    that would need its own kronos wiring."""
    eng = _engine(cfg)
    eng._last_intel_ctx = {"kronos": {"SPY": "unavailable", "QQQ": "unavailable"}}
    ctx = _call_build_ctx(eng, cfg)
    assert ctx["kronos"] == {"SPY": "unavailable", "QQQ": "unavailable"}


# ─────────────────────────────────────────────────────────────────────────
# _log_cycle_signals: "kronos" + "jev_words" additive record fields
# ─────────────────────────────────────────────────────────────────────────

def test_signal_record_gains_kronos_and_jev_words_fields(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    now_epoch = _base_dt().timestamp()
    _write_forecast(now_epoch, {"SPY": _spy_forecast_entry()})

    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        feed = FakeFeedWithRealKronos(lambda now: _bullish_snapshot(now, "SPY"), c)
        eng = _engine(c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: now_epoch)
        await eng.run_cycle(bot=None)
        recs = _read_signals(c.data_dir)
        assert len(recs) == 1
        rec = recs[0]

        # Existing fields/meaning unchanged (additive-only contract).
        for key in ("ts", "underlying", "composite", "strength", "direction", "pillars", "flow_source",
                    "regime", "price_confirm", "vetoes", "reasons", "is_signal_symbol", "nominated",
                    "outcome", "decider", "jev"):
            assert key in rec

        # New additive fields.
        assert "kronos" in rec
        assert rec["kronos"]["asof"] == "2026-09-25T12:00:00-04:00"
        assert rec["kronos"]["p_up_30"] == 0.58
        assert rec["kronos"]["mean_return_30"] == 0.09

        assert "jev_words" in rec
        jw = rec["jev_words"]
        assert set(jw.keys()) == {"gamma_regime", "price_vs_vwap", "momentum_15m", "volatility"}
        # _bullish_underlying: last_price=100.5 > vwap=100.0 -> "above";
        # ret_15m=0.01 > 0 -> "rising"; gex_regime=None -> "unknown";
        # vix=16.0 -> bucket_vix's "normal (15-20)" band.
        assert jw["price_vs_vwap"] == "above"
        assert jw["momentum_15m"] == "rising"
        assert jw["gamma_regime"] == "unknown"
        assert jw["volatility"] == "normal (15-20)"
        # Pin these against jev.py's OWN helpers directly (reuse, not
        # reimplementation) -- see next test for the exhaustive equality
        # version across a matrix of snapshots.

    asyncio.run(_t())


def test_signal_record_kronos_null_when_disabled(cfg, monkeypatch, tmp_path):
    _point_kronos_files(monkeypatch, tmp_path)
    now_epoch = _base_dt().timestamp()
    _write_forecast(now_epoch, {"SPY": _spy_forecast_entry()})

    c = _base_paper_cfg(cfg, kronos_context=False)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        feed = FakeFeedWithRealKronos(lambda now: _bullish_snapshot(now, "SPY"), c)
        eng = _engine(c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: now_epoch)
        await eng.run_cycle(bot=None)
        recs = _read_signals(c.data_dir)
        assert recs[0]["kronos"] is None
        # jev_words is unconditional (never gated by kronos_context).
        assert recs[0]["jev_words"]["price_vs_vwap"] == "above"

    asyncio.run(_t())


def test_signal_record_kronos_null_when_symbol_unavailable(cfg, monkeypatch, tmp_path):
    """Kronos enabled and the file present+fresh, but with no entry for THIS
    symbol (or a non-"ok" status) -> null, never zeros."""
    _point_kronos_files(monkeypatch, tmp_path)
    now_epoch = _base_dt().timestamp()
    _write_forecast(now_epoch, {})  # no SPY entry at all

    c = _base_paper_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        feed = FakeFeedWithRealKronos(lambda now: _bullish_snapshot(now, "SPY"), c)
        eng = _engine(c, broker=broker, feed=feed, orders=orders, ledger=ledger, clock=lambda: now_epoch)
        await eng.run_cycle(bot=None)
        recs = _read_signals(c.data_dir)
        assert recs[0]["kronos"] is None

    asyncio.run(_t())


def test_jev_words_computed_with_jevs_own_helpers_not_duplicated(cfg):
    """Direct pin: engine.py's jev_words values equal jev.py's own bucket
    helper outputs for the SAME UnderlyingSnapshot/vix -- proves reuse, not a
    parallel re-implementation that could silently drift."""
    from agent.dpof import engine as engine_module

    u = _bullish_underlying("SPY")
    vix = 16.0
    assert engine_module._gamma_regime_words is jev_module._gamma_regime_words
    assert engine_module._price_vs_vwap_words is jev_module._price_vs_vwap_words
    assert engine_module._momentum_words is jev_module._momentum_words
    assert engine_module.bucket_vix is jev_module.bucket_vix
    assert engine_module._gamma_regime_words(u) == jev_module._gamma_regime_words(u)
    assert engine_module._price_vs_vwap_words(u) == jev_module._price_vs_vwap_words(u)
    assert engine_module._momentum_words(u) == jev_module._momentum_words(u)
    assert engine_module.bucket_vix(vix) == jev_module.bucket_vix(vix)


# ─────────────────────────────────────────────────────────────────────────
# K2 isolation guarantee: composite/nomination/Jev request/veto outcome are
# byte-identical with and without Kronos forecast files present. agent.dpof.
# signals/jev never import or reference agent.kronos at all (grep-proven,
# see the build report) -- this test additionally PINS the runtime behavior.
# ─────────────────────────────────────────────────────────────────────────

def _isolation_snapshot(now_epoch: float) -> MarketSnapshot:
    return _bullish_snapshot(now_epoch, "SPY")


def _compute_all(cfg, snap: MarketSnapshot, now_et) -> dict:
    signal = signals_module.evaluate_signal(snap, cfg)
    u = snap.underlyings.get(signal.symbol)
    composite, pillars_used, reasons = signals_module.compute_composite(u, cfg)
    state, questions = jev_module.build_entry_request(signal, snap, now_et, cfg)
    # A representative answer set -- decide() is pure over (answers, cfg), so
    # any fixed answers prove the point; these happen to trip every veto path
    # so the resulting `reasons` list is non-trivial (a stronger pin than an
    # always-empty veto list).
    answers = {
        "setup": {"type": "choice", "choice": "late_or_exhausted", "confidence": 0.9},
        "headline_conflict": {"type": "noul", "noul": 0.8},
        "event_risk": {"type": "noul", "noul": 0.9},
    }
    veto_reasons = jev_module.decide(answers, cfg)
    return {
        "signal": {
            "symbol": signal.symbol, "direction": signal.direction.value, "composite": signal.composite,
            "conviction": signal.conviction, "strength": signal.strength, "regime": signal.regime.value,
            "reasons": signal.reasons, "vetoes": signal.vetoes, "actionable": signal.actionable,
        },
        "composite": composite, "pillars_used": pillars_used, "compute_reasons": reasons,
        "jev_state": state, "jev_questions": questions, "jev_veto_reasons": veto_reasons,
    }


def test_composite_nomination_jev_request_and_veto_byte_identical_with_and_without_kronos(cfg, monkeypatch, tmp_path):
    now_epoch = _base_dt().timestamp()
    now_et = _base_dt()
    c = replace(cfg, underlyings=("SPY",))

    # Pass 1: Kronos files ABSENT (never written; env points at a tmp dir
    # that never gets a forecast/scores file).
    _point_kronos_files(monkeypatch, tmp_path / "absent")
    snap_absent = _isolation_snapshot(now_epoch)
    result_absent = _compute_all(c, snap_absent, now_et)

    # Pass 2: Kronos files PRESENT (a real, fresh, "ok" forecast for SPY).
    _point_kronos_files(monkeypatch, tmp_path / "present")
    _write_forecast(now_epoch, {"SPY": _spy_forecast_entry()})
    assert kronos_state.read_forecast(now=now_epoch) is not None  # sanity: the file really is readable
    snap_present = _isolation_snapshot(now_epoch)
    result_present = _compute_all(c, snap_present, now_et)

    assert json.dumps(result_absent, sort_keys=True, default=str) == json.dumps(result_present, sort_keys=True, default=str)
    # Non-trivial pin: the veto path actually fired (proves this isn't a
    # vacuously-empty comparison).
    assert result_absent["jev_veto_reasons"] == ["headline_conflict", "event_risk", "late_or_exhausted"]
