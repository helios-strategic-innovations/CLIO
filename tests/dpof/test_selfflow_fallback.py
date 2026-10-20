"""Tests for the self-computed flow fallback (Task F, 2026-09-25, operator
directive): while UW (Unusual Whales) is unavailable, agent.dpof.data falls
back to agent.flowcollector.state.get_selfflow() (Alpaca OPRA/SIP
self-computed pillars) for a given underlying/pillar, instead of leaving it
missing. UW always wins whenever it is available; the two sources are never
blended for the same pillar. See docs/design/clio_dpof.md's "Flow fallback"
section for the exact per-pillar mapping this file pins.

Plugin-free (no pytest-asyncio), matching tests/dpof/test_data.py. Reuses
that file's Fakes/FakeBroker/_small_cfg/_bar helpers (cross-file import is
an established pattern in this suite -- see test_engine_judgment.py). The
REAL agent.flowcollector.state module is used (hermetic: CLIO_SELFFLOW_FILE
always points at a tmp_path file, and its module-level mtime cache is reset
between tests -- mirrors tests/test_flowcollector_state.py's own fixture).
"""

from __future__ import annotations

import asyncio
import pathlib
import threading
import time
from dataclasses import replace

import pytest

import agent.dpof.data as data
import agent.flowcollector.state as flowcollector_state
from agent.dpof.engine import DPOFEngine
from agent.dpof.models import MarketSnapshot, PillarReading, UnderlyingSnapshot

from tests.dpof.test_data import FakeBroker, Fakes, _bar, _run, _small_cfg
from tests.dpof.test_engine import FakeJevClient, FakeMemory, _fake_candidate_fn, _fake_jev_sentinel_no_escalate

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_selfflow_state_cache(monkeypatch, tmp_path):
    """Isolate every test from the real environment/cache, mirroring
    tests/test_flowcollector_state.py's own autouse fixture -- the module
    caches its last-read (path, mtime) pair process-wide, so a stale entry
    from another test module could otherwise leak in by chance."""
    # A temp path, never delenv: unset falls back to the LIVE state file.
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(tmp_path / "no_selfflow_state.json"))
    with flowcollector_state._state_cache_lock:
        flowcollector_state._state_cache["path"] = None
        flowcollector_state._state_cache["mtime"] = None
        flowcollector_state._state_cache["data"] = None
    yield


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    return Fakes(monkeypatch, tmp_path)


@pytest.fixture(autouse=True)
def _fast_timeouts(monkeypatch):
    """Same rationale as test_data.py's own fixture of the same name: keep
    per-call timeouts small so an accidental real hang fails fast."""
    monkeypatch.setattr(data, "_UW_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_BARS_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_VIX_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_FILE_GATE_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_SNAPSHOT_BUDGET_SECONDS", 5.0)


# Sample counts comfortably above cfg.selfflow_min_trades (20).
_AMPLE_SAMPLE = {"option_trades_5m": 500, "dark_classified_5m": 200}


def _write_selfflow(tmp_path, monkeypatch, symbol: str, as_of: float, **fields) -> None:
    """Writes a real selfflow_state.json (via the REAL write_state_atomic)
    for `symbol`, pointed at by a fresh CLIO_SELFFLOW_FILE env override."""
    path = str(tmp_path / "selfflow_state.json")
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", path)
    # The DPOF fallback reads the scale-free *_ratio fields; tests pass the
    # intended pillar value as nofa=/premium=/darkpool= and it is written to
    # both the tanh field and the ratio field.
    for name in ("nofa", "premium", "darkpool"):
        if name in fields and f"{name}_ratio" not in fields:
            fields[f"{name}_ratio"] = fields[name]
    fields.setdefault("quality_notes", _AMPLE_SAMPLE)
    entry = {"as_of": as_of, **fields}
    flowcollector_state.write_state_atomic({"symbols": {symbol.upper(): entry}})


def _fail_all_uw(fakes: Fakes, tickers) -> None:
    """Blanket-fails EVERY UW endpoint data.py touches, for every ticker --
    the "UW is completely down" scenario the operator directive describes.
    {} (no 'data' key) is data.py's own FAILURE shape (_shape_check_uw_
    response), never confused with a genuine empty-list result."""
    fakes.uw_responses[data._EP_FLOW_ALERTS] = {}
    fakes.uw_responses[data._EP_MARKET_TIDE] = {}
    for t in tickers:
        fakes.uw_responses[f"/stock/{t}/net-prem-ticks"] = {}
        fakes.uw_responses[f"/darkpool/{t}"] = {}
        fakes.uw_responses[f"/stock/{t}/spot-exposures/strike"] = {}
        fakes.uw_responses[data._EP_NOPE_TICKER.format(ticker=t)] = {}
        fakes.uw_responses[data._EP_MAX_PAIN_TICKER.format(ticker=t)] = {}


def _spy_get_selfflow(monkeypatch, return_value=None):
    """Replaces the REAL agent.flowcollector.state.get_selfflow with a spy
    that records every (symbol, max_age_seconds) call -- used to prove data.py
    either DID or DID NOT touch the flowcollector state layer at all this
    cycle, independent of file-based staleness timing."""
    calls: list[tuple[str, int]] = []

    def _spy(symbol, max_age_seconds=300):
        calls.append((symbol, max_age_seconds))
        return return_value

    monkeypatch.setattr(flowcollector_state, "get_selfflow", _spy)
    return calls


def _one_symbol_cfg(cfg, **overrides):
    return replace(cfg, underlyings=("SPY",), min_pillars=2, uw_refresh_seconds=100, bars_refresh_seconds=50, **overrides)


def _spy_price_bars(now_epoch=1000.0):
    return FakeBroker(bars_by_symbol={"SPY": [_bar(now_epoch - 1, 100, 101, 99, 100, 10)]})


# ═══════════════════════════════════════════════════════════════════════════
#  Pure mapping: DPOFDataFeed._selfflow_pillar_reading
# ═══════════════════════════════════════════════════════════════════════════

def _feed(cfg) -> data.DPOFDataFeed:
    return data.DPOFDataFeed(cfg, FakeBroker())


def test_mapping_uses_scale_free_ratios_not_tanh_values(cfg):
    """nofa/premium/darkpool come from the *_ratio fields (same form as UW's
    ratios); the tanh(dollars/scale) fields must never reach the composite."""
    feed = _feed(cfg)
    entry = {"nofa": 0.99, "premium": 0.99, "darkpool": 0.99,
             "nofa_ratio": 0.42, "premium_ratio": -0.31, "darkpool_ratio": 0.15,
             "quality_notes": _AMPLE_SAMPLE}
    nofa = feed._selfflow_pillar_reading("SPY", "nofa", entry)
    premium = feed._selfflow_pillar_reading("SPY", "premium", entry)
    darkpool = feed._selfflow_pillar_reading("SPY", "darkpool", entry)

    assert nofa.value == pytest.approx(0.42)
    assert nofa.available is True
    assert nofa.source == "selfflow:SPY"
    assert nofa.detail["selfflow"] is True
    assert nofa.detail["raw"] == 0.42

    assert premium.value == pytest.approx(-0.31)
    assert darkpool.value == pytest.approx(0.15)


def test_mapping_missing_ratio_key_is_unavailable_never_the_tanh_value(cfg):
    feed = _feed(cfg)
    entry = {"nofa": 0.5, "premium": 0.5, "darkpool": 0.5}  # older collector: no *_ratio keys
    for name in ("nofa", "premium", "darkpool"):
        assert feed._selfflow_pillar_reading("SPY", name, entry) is None


def test_mapping_empty_window_is_zero_like_uw_empty(cfg):
    feed = _feed(cfg)
    entry = {"nofa_ratio": None, "premium_ratio": None, "darkpool_ratio": None,
             "quality_notes": {"nofa_empty": True, "premium_empty": True, "darkpool_empty": False}}
    assert feed._selfflow_pillar_reading("SPY", "nofa", entry).value == 0.0
    assert feed._selfflow_pillar_reading("SPY", "premium", entry).value == 0.0
    # darkpool had trades but none classifiable -> ratio None and not "empty" -> unavailable
    assert feed._selfflow_pillar_reading("SPY", "darkpool", entry) is None


def test_mapping_clamps_out_of_range_values(cfg):
    feed = _feed(cfg)
    entry = {"nofa_ratio": 1.5, "premium_ratio": -2.3, "darkpool_ratio": 1.0, "quality_notes": _AMPLE_SAMPLE}
    nofa = feed._selfflow_pillar_reading("SPY", "nofa", entry)
    premium = feed._selfflow_pillar_reading("SPY", "premium", entry)
    darkpool = feed._selfflow_pillar_reading("SPY", "darkpool", entry)
    assert nofa.value == 1.0
    assert premium.value == -1.0
    assert darkpool.value == 1.0


def test_mapping_missing_field_returns_none_never_zero(cfg):
    feed = _feed(cfg)
    assert feed._selfflow_pillar_reading("SPY", "nofa", {}) is None
    assert feed._selfflow_pillar_reading("SPY", "premium", {"nofa": 0.1}) is None
    assert feed._selfflow_pillar_reading("SPY", "darkpool", {}) is None


def test_mapping_entry_none_returns_none(cfg):
    feed = _feed(cfg)
    assert feed._selfflow_pillar_reading("SPY", "nofa", None) is None


def test_mapping_gex_negative_maps_to_accelerating_plus_half(cfg):
    feed = _feed(cfg)
    reading = feed._selfflow_pillar_reading("SPY", "gex", {"gex_regime": "negative", "gex": -9000.0})
    assert reading.value == pytest.approx(0.5)
    assert reading.source == "selfflow:SPY"
    assert reading.detail["regime"] == "ACCELERATING"
    assert reading.detail["selfflow_regime"] == "negative"
    assert reading.detail["net_gex"] == -9000.0


def test_mapping_gex_positive_maps_to_stabilizing_minus_point3(cfg):
    feed = _feed(cfg)
    reading = feed._selfflow_pillar_reading("SPY", "gex", {"gex_regime": "positive", "gex": 12345.0})
    assert reading.value == pytest.approx(-0.3)
    assert reading.detail["regime"] == "STABILIZING"


def test_mapping_gex_neutral_maps_to_zero(cfg):
    feed = _feed(cfg)
    reading = feed._selfflow_pillar_reading("SPY", "gex", {"gex_regime": "neutral"})
    assert reading.value == pytest.approx(0.0)
    assert reading.detail["regime"] == "NEUTRAL"


def test_mapping_gex_case_insensitive():
    from agent.dpof.config import DPOFConfig

    feed = _feed(DPOFConfig())
    reading = feed._selfflow_pillar_reading("SPY", "gex", {"gex_regime": "NEGATIVE"})
    assert reading.value == pytest.approx(0.5)


def test_mapping_gex_unknown_regime_string_returns_none(cfg):
    feed = _feed(cfg)
    # Never a guessed 0.0 for a regime string this mapping doesn't recognize
    # -- an unrecognized value is treated as unmappable, not "neutral".
    assert feed._selfflow_pillar_reading("SPY", "gex", {"gex_regime": "sideways"}) is None


def test_mapping_gex_missing_regime_returns_none(cfg):
    feed = _feed(cfg)
    assert feed._selfflow_pillar_reading("SPY", "gex", {"gex": 100.0}) is None


def test_mapping_pin_strike_surfaced_from_gex_detail(cfg):
    feed = _feed(cfg)
    reading = feed._selfflow_pillar_reading(
        "SPY", "gex", {"gex_regime": "negative", "gex": -1.0, "gex_detail": {"pin_strike": 575.0}},
    )
    assert reading.detail["pin_strike"] == 575.0


# ═══════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════

def test_config_defaults():
    from agent.dpof.config import DPOFConfig

    d = DPOFConfig()
    assert d.selfflow_fallback is True
    assert d.selfflow_max_age_seconds == 120


def test_config_env_overrides(monkeypatch):
    from agent.dpof.config import load_dpof_config

    monkeypatch.setenv("CLIO_DPOF_SELFFLOW_FALLBACK", "false")
    monkeypatch.setenv("CLIO_DPOF_SELFFLOW_MAX_AGE", "45")
    c = load_dpof_config()
    assert c.selfflow_fallback is False
    assert c.selfflow_max_age_seconds == 45


def test_config_summary_includes_new_flags(cfg):
    from agent.dpof.config import config_summary

    summary = config_summary(cfg)
    assert summary["selfflow_fallback"] is True
    assert summary["selfflow_max_age_seconds"] == 120


# ═══════════════════════════════════════════════════════════════════════════
#  snapshot()-level: UW available -> selfflow never touched
# ═══════════════════════════════════════════════════════════════════════════

def test_uw_available_selfflow_never_read(fakes, cfg, monkeypatch, tmp_path):
    """Every UW endpoint defaults to a genuine {"data": []} (Fakes' own
    default) -- a legitimate, fresh, available=True reading for every pillar/
    NOPE/max-pain field, for both configured underlyings. get_selfflow()
    must never be called at all this cycle."""
    calls = _spy_get_selfflow(monkeypatch, return_value={"nofa": 0.99, "as_of": time.time()})
    c = _small_cfg(cfg)  # underlyings=("SPY", "QQQ"), min_pillars=2
    broker = FakeBroker(bars_by_symbol={
        "SPY": [_bar(999, 100, 101, 99, 100, 10)],
        "QQQ": [_bar(999, 400, 401, 399, 400, 10)],
    })
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=1000.0))

    assert calls == []  # the key assertion: selfflow was never even read

    for sym in ("SPY", "QQQ"):
        u = snap.underlyings[sym]
        assert u.flow_selfflow is False
        for name in ("nofa", "premium", "darkpool", "gex"):
            p = u.pillars[name]
            assert p.available is True
            assert p.source.startswith("uw:")


# ═══════════════════════════════════════════════════════════════════════════
#  snapshot()-level: UW fully unavailable + fresh selfflow
# ═══════════════════════════════════════════════════════════════════════════

def test_uw_unavailable_fresh_selfflow_fills_all_pillars(fakes, cfg, monkeypatch, tmp_path):
    now_epoch = 1000.0
    c = _one_symbol_cfg(cfg)
    _fail_all_uw(fakes, c.underlyings)
    # get_selfflow()'s OWN freshness gate is wall-clock (time.time()), NOT
    # the fake simulation now_epoch feed.snapshot() otherwise runs on -- the
    # written as_of must be REAL-clock-fresh, decoupled from now_epoch.
    _write_selfflow(
        tmp_path, monkeypatch, "SPY", as_of=time.time(),
        nofa=0.6, premium=0.4, darkpool=0.2, gex=55000.0, gex_regime="negative",
        nope=0.05, max_pain=570.0,
    )
    broker = _spy_price_bars(now_epoch)
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))

    u = snap.underlyings["SPY"]
    assert u.flow_selfflow is True

    nofa, premium, darkpool, gex = (u.pillars[n] for n in ("nofa", "premium", "darkpool", "gex"))
    assert nofa.available and nofa.value == pytest.approx(0.6) and nofa.source == "selfflow:SPY"
    assert premium.available and premium.value == pytest.approx(0.4) and premium.source == "selfflow:SPY"
    assert darkpool.available and darkpool.value == pytest.approx(0.2) and darkpool.source == "selfflow:SPY"
    assert gex.available and gex.value == pytest.approx(0.5) and gex.source == "selfflow:SPY"  # negative -> ACCELERATING -> +0.5

    assert u.gex_regime == "ACCELERATING"
    assert u.nope == pytest.approx(0.05)
    assert u.max_pain == pytest.approx(570.0)

    # Composite is computable and the snapshot is healthy on selfflow alone.
    from agent.dpof.signals import compute_composite

    composite, used, reasons = compute_composite(u, c)
    assert composite is not None
    assert reasons == []
    assert snap.data_healthy is True
    assert snap.health_issues == []


def test_uw_partially_available_only_missing_pillar_falls_back(fakes, cfg, monkeypatch, tmp_path):
    """nofa (and premium/gex) succeed via UW's own genuine-empty default;
    darkpool alone fails -- only darkpool should come from selfflow, and the
    selfflow entry's OWN (deliberately different) nofa value must never leak
    through UW's real, fresh nofa reading."""
    now_epoch = 1000.0
    c = _one_symbol_cfg(cfg)
    fakes.uw_responses["/darkpool/SPY"] = {}  # only darkpool fails
    _write_selfflow(
        tmp_path, monkeypatch, "SPY", as_of=time.time(),  # real-clock-fresh, see the note above
        nofa=0.99, premium=0.99, darkpool=0.33, gex=1.0, gex_regime="positive",
    )
    broker = _spy_price_bars(now_epoch)
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))

    u = snap.underlyings["SPY"]
    nofa = u.pillars["nofa"]
    darkpool = u.pillars["darkpool"]

    assert nofa.source.startswith("uw:")
    assert nofa.value == 0.0  # UW's genuine empty {"data": []} -- NOT selfflow's 0.99
    assert nofa.detail.get("empty") is True

    assert darkpool.source == "selfflow:SPY"
    assert darkpool.value == pytest.approx(0.33)
    assert u.flow_selfflow is True


# ═══════════════════════════════════════════════════════════════════════════
#  selfflow itself missing/stale -> unchanged from today's behaviour
# ═══════════════════════════════════════════════════════════════════════════

def test_selfflow_missing_file_behaves_like_today(fakes, cfg, monkeypatch, tmp_path):
    now_epoch = 1000.0
    c = _one_symbol_cfg(cfg)
    _fail_all_uw(fakes, c.underlyings)
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(tmp_path / "does_not_exist.json"))
    broker = _spy_price_bars(now_epoch)
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))

    u = snap.underlyings["SPY"]
    assert u.flow_selfflow is False
    for name in ("nofa", "premium", "darkpool", "gex"):
        assert u.pillars[name].available is False
        assert u.pillars[name].value is None
    assert snap.data_healthy is False


def test_selfflow_stale_beyond_max_age_behaves_like_today(fakes, cfg, monkeypatch, tmp_path):
    now_epoch = 1000.0
    c = _one_symbol_cfg(cfg, selfflow_max_age_seconds=120)
    _fail_all_uw(fakes, c.underlyings)
    # Real-clock-stale (300s ago), well beyond selfflow_max_age_seconds=120 --
    # see the wall-clock-vs-fake-now_epoch note in the "fresh" test above.
    _write_selfflow(tmp_path, monkeypatch, "SPY", as_of=time.time() - 300, nofa=0.9, premium=0.9, darkpool=0.9, gex_regime="negative")
    broker = _spy_price_bars(now_epoch)
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))

    u = snap.underlyings["SPY"]
    assert u.flow_selfflow is False
    assert u.pillars["nofa"].available is False
    assert snap.data_healthy is False


# ═══════════════════════════════════════════════════════════════════════════
#  Flag off -> today's behaviour exactly
# ═══════════════════════════════════════════════════════════════════════════

def test_flag_off_never_touches_selfflow(fakes, cfg, monkeypatch, tmp_path):
    now_epoch = 1000.0
    c = _one_symbol_cfg(cfg, selfflow_fallback=False)
    _fail_all_uw(fakes, c.underlyings)
    calls = _spy_get_selfflow(monkeypatch, return_value={"nofa": 0.5, "as_of": now_epoch})
    broker = _spy_price_bars(now_epoch)
    feed = data.DPOFDataFeed(c, broker)
    snap = _run(feed.snapshot(now_epoch=now_epoch))

    assert calls == []  # flag off -- get_selfflow must never even be called
    u = snap.underlyings["SPY"]
    assert u.flow_selfflow is False
    for name in ("nofa", "premium", "darkpool", "gex"):
        assert u.pillars[name].available is False
    assert snap.data_healthy is False


# ═══════════════════════════════════════════════════════════════════════════
#  Off-loop dispatch
# ═══════════════════════════════════════════════════════════════════════════

def test_selfflow_read_dispatched_off_loop(cfg, monkeypatch):
    main_thread = threading.current_thread()
    seen_threads = []

    def _spy(symbol, max_age_seconds=300):
        seen_threads.append(threading.current_thread())
        return None

    monkeypatch.setattr(flowcollector_state, "get_selfflow", _spy)

    c = _one_symbol_cfg(cfg)
    feed = data.DPOFDataFeed(c, FakeBroker())
    # A freshly constructed feed has empty pillar/nope/max_pain caches for
    # every underlying, so _needs_selfflow is True without any UW setup --
    # call the fallback refresher directly (unit-level, no snapshot() needed).
    _run(feed._maybe_refresh_selfflow(1000.0))

    assert len(seen_threads) == 1
    assert seen_threads[0] is not main_thread


def test_maybe_refresh_selfflow_skips_when_uw_already_fresh(cfg, monkeypatch):
    """_needs_selfflow gates the read per-symbol -- a symbol with every
    pillar/NOPE/max-pain already fresh from UW is never even passed to
    get_selfflow (matches the snapshot()-level "UW available" test, but at
    the unit level against _maybe_refresh_selfflow directly)."""
    calls = _spy_get_selfflow(monkeypatch, return_value=None)
    c = _one_symbol_cfg(cfg)
    feed = data.DPOFDataFeed(c, FakeBroker())
    now_epoch = 1000.0
    for name in ("nofa", "premium", "darkpool", "gex"):
        feed._pillar_cache[("SPY", name)] = PillarReading(name=name, value=0.1, available=True, source="uw:test")
        feed._pillar_cache_time[("SPY", name)] = now_epoch
    feed._nope_cache_time["SPY"] = now_epoch
    feed._max_pain_cache_time["SPY"] = now_epoch

    _run(feed._maybe_refresh_selfflow(now_epoch))
    assert calls == []


# ═══════════════════════════════════════════════════════════════════════════
#  Decider ctx / status_text visibility
# ═══════════════════════════════════════════════════════════════════════════

def _underlying(symbol="SPY", flow_selfflow=False) -> UnderlyingSnapshot:
    return UnderlyingSnapshot(
        symbol=symbol,
        pillars={
            "nofa": PillarReading(name="nofa", value=0.4, available=True, source="selfflow:SPY" if flow_selfflow else "uw:test"),
        },
        last_price=100.5, vwap=100.0, ret_5m=0.001, ret_15m=0.002,
        flow_selfflow=flow_selfflow,
    )


def _snapshot(symbol="SPY", flow_selfflow=False, now_epoch=1000.0) -> MarketSnapshot:
    return MarketSnapshot(
        asof=now_epoch, underlyings={symbol: _underlying(symbol, flow_selfflow)},
        vix=16.0, vix_age_seconds=10.0, data_healthy=True,
    )


def _make_engine(cfg) -> DPOFEngine:
    return DPOFEngine(
        memory=FakeMemory(), broker=None, feed=None, orders=None, jev=FakeJevClient(), alerts=None, ledger=None,
        decider=None, candidate_fn=_fake_candidate_fn, jev_sentinel=_fake_jev_sentinel_no_escalate,
        position_alignment_fn=lambda *_a, **_kw: {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"},
        clock=lambda: 1000.0, market_open_fn=lambda: True, config_loader=lambda: cfg,
    )


def test_underlyings_ctx_carries_flow_note_when_selfflow_active(cfg):
    engine = _make_engine(cfg)
    snap = _snapshot(flow_selfflow=True)
    ctx = engine._build_underlyings_ctx(snap, cfg, signal=None)
    assert "flow_note" in ctx["SPY"]
    assert "self-computed" in ctx["SPY"]["flow_note"]
    assert "not validated against Unusual Whales" in ctx["SPY"]["flow_note"]


def test_underlyings_ctx_omits_flow_note_when_uw_active(cfg):
    engine = _make_engine(cfg)
    snap = _snapshot(flow_selfflow=False)
    ctx = engine._build_underlyings_ctx(snap, cfg, signal=None)
    assert "flow_note" not in ctx["SPY"]


def test_status_text_names_selfflow_underlyings(cfg):
    from agent.dpof.ledger import DPOFLedger

    ledger = DPOFLedger(cfg.data_dir)  # mode=PAPER (cfg fixture default), never .load()ed -- fine, all-empty state
    engine = DPOFEngine(
        memory=FakeMemory(), broker=None, feed=None, orders=None, jev=FakeJevClient(), alerts=None, ledger=ledger,
        decider=None, candidate_fn=_fake_candidate_fn, jev_sentinel=_fake_jev_sentinel_no_escalate,
        position_alignment_fn=lambda *_a, **_kw: {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"},
        clock=lambda: 1000.0, market_open_fn=lambda: True, config_loader=lambda: cfg,
    )
    engine._last_snapshot = _snapshot(flow_selfflow=True)
    text = _run(engine.status_text())
    assert "flow: self-computed fallback active for SPY" in text


def test_status_text_omits_flow_line_when_uw_active(cfg):
    from agent.dpof.ledger import DPOFLedger

    ledger = DPOFLedger(cfg.data_dir)
    engine = DPOFEngine(
        memory=FakeMemory(), broker=None, feed=None, orders=None, jev=FakeJevClient(), alerts=None, ledger=ledger,
        decider=None, candidate_fn=_fake_candidate_fn, jev_sentinel=_fake_jev_sentinel_no_escalate,
        position_alignment_fn=lambda *_a, **_kw: {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"},
        clock=lambda: 1000.0, market_open_fn=lambda: True, config_loader=lambda: cfg,
    )
    engine._last_snapshot = _snapshot(flow_selfflow=False)
    text = _run(engine.status_text())
    assert "self-computed fallback" not in text


# ═══════════════════════════════════════════════════════════════════════════
#  Isolation
# ═══════════════════════════════════════════════════════════════════════════

def test_data_py_never_references_broker_or_orders_modules():
    """Source-text check (mirrors tests/dpof/test_wiring.py's pattern): the
    selfflow-fallback code added to data.py must not have introduced a
    dependency on agent.dpof.broker/orders -- data.py stays I/O-in, never
    order-execution-adjacent."""
    text = (ROOT / "agent" / "dpof" / "data.py").read_text(encoding="utf-8")
    assert "agent.dpof.broker" not in text
    assert "agent.dpof.orders" not in text
    assert "agent.flowcollector.state" in text  # sanity: the fallback import IS present


# ── Pass-2 review: minimum-sample guard ─────────────────────────────────────

def test_thin_sample_is_zero_not_a_full_ratio(cfg):
    """Two option trades / one dark print give ratios of exactly +/-1.0 -- that
    must read as "thin" (0.0), never a strong directional pillar."""
    feed = _feed(cfg)
    entry = {"nofa_ratio": 1.0, "premium_ratio": 1.0, "darkpool_ratio": 1.0,
             "quality_notes": {"option_trades_5m": 2, "dark_classified_5m": 1}}
    for name in ("nofa", "premium", "darkpool"):
        r = feed._selfflow_pillar_reading("SPY", name, entry)
        assert r.available and r.value == 0.0 and r.detail["thin"] is True, (name, r)


def test_sample_at_threshold_uses_the_ratio(cfg):
    feed = _feed(cfg)
    n = cfg.selfflow_min_trades
    entry = {"nofa_ratio": 0.4, "premium_ratio": 0.2, "darkpool_ratio": -0.3,
             "quality_notes": {"option_trades_5m": n, "dark_classified_5m": n}}
    assert feed._selfflow_pillar_reading("SPY", "nofa", entry).value == pytest.approx(0.4)
    assert feed._selfflow_pillar_reading("SPY", "darkpool", entry).value == pytest.approx(-0.3)


def test_unknown_sample_size_is_unavailable(cfg):
    feed = _feed(cfg)
    entry = {"nofa_ratio": 0.9, "premium_ratio": 0.9, "darkpool_ratio": 0.9, "quality_notes": {}}
    for name in ("nofa", "premium", "darkpool"):
        assert feed._selfflow_pillar_reading("SPY", name, entry) is None


def test_min_trades_env_knob(monkeypatch):
    from agent.dpof.config import load_dpof_config
    monkeypatch.setenv("CLIO_DPOF_SELFFLOW_MIN_TRADES", "7")
    assert load_dpof_config().selfflow_min_trades == 7


# ── Pass-2 review: UW auth backoff must not burn DPOF's daily UW cap ─────────

def test_uw_fetch_skips_cap_during_auth_backoff(cfg, monkeypatch):
    import agent.integrations.uw_flow_intel as uw
    feed = _feed(cfg)
    called = {"n": 0}

    async def _never(*_a, **_kw):
        called["n"] += 1
        return {"data": []}

    monkeypatch.setattr(uw, "_uw_get", _never)
    monkeypatch.setattr(uw, "_auth_blocked_until", 1e18)   # backoff active
    before = feed._uw_call_count
    res, err = asyncio.run(feed._uw_fetch("/x", None, 1_000_000.0, "probe"))
    assert res is None and "backoff" in err
    assert feed._uw_call_count == before and called["n"] == 0

    monkeypatch.setattr(uw, "_auth_blocked_until", 0.0)    # backoff over -> counted as usual
    asyncio.run(feed._uw_fetch("/x", None, 1_000_000.0, "probe"))
    assert feed._uw_call_count == before + 1 and called["n"] == 1


def test_zero_prints_in_window_is_unavailable_not_zero(cfg):
    """No prints at all in the 5-min window = stalled/idle stream, not a quiet
    tape: unavailable, never a 0.0 that counts as an available pillar."""
    feed = _feed(cfg)
    entry = {"nofa_ratio": None, "premium_ratio": None, "darkpool_ratio": None,
             "quality_notes": {"option_trades_5m": 0, "dark_classified_5m": 0,
                               "nofa_empty": True, "premium_empty": True, "darkpool_empty": True}}
    for name in ("nofa", "premium", "darkpool"):
        assert feed._selfflow_pillar_reading("SPY", name, entry) is None


def test_prints_but_none_classified_is_still_a_zero_reading(cfg):
    feed = _feed(cfg)
    entry = {"nofa_ratio": None, "quality_notes": {"option_trades_5m": 50, "nofa_empty": True}}
    reading = feed._selfflow_pillar_reading("SPY", "nofa", entry)
    assert reading is not None and reading.available is True and reading.value == 0.0
