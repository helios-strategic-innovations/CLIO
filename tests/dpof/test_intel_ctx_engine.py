"""Tests for the engine.py half of Task 2/Task 3 (INTEL_V3_SPEC.md / G
prompt): regime labeling + advisory intel ctx assembly (_regime_ctx,
_build_intel_ctx, _build_ctx's merge of self._last_intel_ctx), and the two
new review triggers + the entry-cooldown "new surprise" bypass in
_review_triggers/_cooldown_allows.

Reuses test_engine.py's/test_engine_judgment.py's fakes and helpers (same
pattern test_engine_judgment.py itself already follows) rather than
redefining them, so this file only ever tests NEW behavior added by G.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from agent.dpof.config import DpofMode
from agent.dpof.engine import DPOFEngine, _REGIME_LABELS, _parse_iso_epoch
from agent.dpof.models import MarketSnapshot, Regime, Signal, Direction

from tests.dpof.test_engine import (
    FakeMemory,
    _base_dt,
    _bullish_snapshot,
    _neutral_snapshot,
    _position,
)
from tests.dpof.test_engine_judgment import _engine


def _run(coro):
    return asyncio.run(coro)


_EMPTY_EVENTS = {"upcoming": [], "recent_surprises": [], "source": "unavailable"}
_EMPTY_BREAKING = {"active": False, "max_severity": 0, "events": [], "as_of": None, "stale": True, "source": "unavailable"}
_EMPTY_CALENDAR_STATUS = {"status": "missing", "age_hours": None, "sources": []}


class FakeIntelFeed:
    """Only implements get_intel_ctx() -- everything else engine.py might
    call on self._feed is untouched by these tests."""

    def __init__(self, intel=None, raise_exc=None):
        self.intel = intel if intel is not None else {}
        self.raise_exc = raise_exc
        self.calls: list[float] = []

    async def get_intel_ctx(self, now_epoch=None):
        self.calls.append(now_epoch)
        if self.raise_exc:
            raise self.raise_exc
        return self.intel


# ─────────────────────────────────────────────────────────────────────────
# _regime_ctx
# ─────────────────────────────────────────────────────────────────────────

def test_regime_ctx_labels_halt_as_extreme(cfg):
    eng = _engine(cfg)
    snap = MarketSnapshot(asof=0.0, vix=cfg.vix_extreme_label_at, vix_age_seconds=10.0)
    out = eng._regime_ctx(snap, cfg)
    assert out["label"] == "EXTREME"
    assert out["vix"] == cfg.vix_extreme_label_at
    assert out["vix_age_seconds"] == 10.0
    assert _REGIME_LABELS[Regime.HALT] == "EXTREME"


@pytest.mark.parametrize("vix,expected_label", [
    (10.0, "TRENDING"),
    (20.0, "NORMAL"),
    (26.0, "CHOPPY"),
])
def test_regime_ctx_labels_match_classify_regime(cfg, vix, expected_label):
    eng = _engine(cfg)
    snap = MarketSnapshot(asof=0.0, vix=vix, vix_age_seconds=10.0)
    out = eng._regime_ctx(snap, cfg)
    assert out["label"] == expected_label


def test_regime_ctx_unknown_when_vix_missing(cfg):
    eng = _engine(cfg)
    snap = MarketSnapshot(asof=0.0, vix=None, vix_age_seconds=None)
    out = eng._regime_ctx(snap, cfg)
    assert out["label"] == "UNKNOWN"


def test_regime_ctx_none_snapshot_is_unknown(cfg):
    eng = _engine(cfg)
    out = eng._regime_ctx(None, cfg)
    assert out == {"vix": None, "vix_age_seconds": None, "label": "UNKNOWN"}


# ─────────────────────────────────────────────────────────────────────────
# _build_intel_ctx
# ─────────────────────────────────────────────────────────────────────────

def test_build_intel_ctx_merges_feed_output_and_regime(cfg):
    intel = {
        "events": {"upcoming": [{"event": "CPI"}], "recent_surprises": [], "source": "econ_surprise"},
        "vol_surface": {"SPY": {"iv_front": 18.0}},
        "breaking": {"active": True, "max_severity": 3, "events": [], "as_of": None, "stale": False, "source": "x"},
        "calendar_status": {"status": "ok", "age_hours": 0.5, "sources": ["econ_calendar"]},
    }
    eng = _engine(cfg, feed=FakeIntelFeed(intel))
    snap = MarketSnapshot(asof=0.0, vix=16.0, vix_age_seconds=10.0)
    out = _run(eng._build_intel_ctx(cfg, snap, 1000.0))
    assert out["events"] == intel["events"]
    assert out["vol_surface"] == intel["vol_surface"]
    assert out["breaking"] == intel["breaking"]
    assert out["calendar_status"] == intel["calendar_status"]
    assert out["regime"]["label"] == "TRENDING"  # vix=16.0 < cfg.vix_trending_below (18.0)
    assert eng._feed.calls == [1000.0]


def test_build_intel_ctx_defends_against_no_feed(cfg):
    eng = _engine(cfg, feed=None)
    out = _run(eng._build_intel_ctx(cfg, None, 1000.0))
    assert out["events"] == _EMPTY_EVENTS
    assert out["vol_surface"] == {sym: None for sym in cfg.underlyings}
    assert out["breaking"] == _EMPTY_BREAKING
    assert out["calendar_status"] == _EMPTY_CALENDAR_STATUS
    assert out["regime"] == {"vix": None, "vix_age_seconds": None, "label": "UNKNOWN"}


def test_build_intel_ctx_defends_against_feed_raising(cfg):
    eng = _engine(cfg, feed=FakeIntelFeed(raise_exc=RuntimeError("boom")))
    out = _run(eng._build_intel_ctx(cfg, None, 1000.0))  # must not raise
    assert out["events"] == _EMPTY_EVENTS
    assert out["breaking"] == _EMPTY_BREAKING


def test_build_intel_ctx_defends_against_non_dict_feed_output(cfg):
    eng = _engine(cfg, feed=FakeIntelFeed(intel="not a dict"))
    out = _run(eng._build_intel_ctx(cfg, None, 1000.0))
    assert out["events"] == _EMPTY_EVENTS


def test_build_intel_ctx_defends_against_partial_feed_output(cfg):
    """A feed that only returns SOME keys (e.g. an in-progress sibling
    module) must not KeyError anywhere downstream -- every key defaults."""
    eng = _engine(cfg, feed=FakeIntelFeed(intel={"events": {"upcoming": [], "recent_surprises": [], "source": "x"}}))
    out = _run(eng._build_intel_ctx(cfg, None, 1000.0))
    assert out["vol_surface"] == {sym: None for sym in cfg.underlyings}
    assert out["breaking"] == _EMPTY_BREAKING
    assert out["calendar_status"] == _EMPTY_CALENDAR_STATUS


# ─────────────────────────────────────────────────────────────────────────
# _build_ctx merges self._last_intel_ctx
# ─────────────────────────────────────────────────────────────────────────

def _call_build_ctx(eng, cfg, snap=None):
    return eng._build_ctx(
        now_et=_base_dt(), cfg=cfg, snap=snap, underlyings_ctx={}, market_regime=None,
        risk=__import__("agent.dpof.models", fromlist=["RiskState"]).RiskState(),
        open_positions=[], dpof_equity=10_000.0, open_premium=0.0,
    )


def test_build_ctx_includes_intel_keys_from_last_intel_ctx(cfg):
    eng = _engine(cfg)
    eng._last_intel_ctx = {
        "events": {"upcoming": [{"event": "CPI"}], "recent_surprises": [], "source": "econ_surprise"},
        "vol_surface": {"SPY": {"iv_front": 18.0}},
        "breaking": {"active": True, "max_severity": 2, "events": [], "as_of": None, "stale": False, "source": "x"},
        "regime": {"vix": 16.0, "vix_age_seconds": 10.0, "label": "NORMAL"},
        "calendar_status": {"status": "ok", "age_hours": 0.5, "sources": ["econ_calendar"]},
    }
    ctx = _call_build_ctx(eng, cfg)
    assert ctx["events"] == eng._last_intel_ctx["events"]
    assert ctx["vol_surface"] == eng._last_intel_ctx["vol_surface"]
    assert ctx["breaking"] == eng._last_intel_ctx["breaking"]
    assert ctx["regime"] == eng._last_intel_ctx["regime"]
    assert ctx["calendar_status"] == eng._last_intel_ctx["calendar_status"]


def test_build_ctx_defaults_when_no_intel_ctx_yet(cfg):
    """A fresh engine (no cycle has run) must still return a complete ctx
    shape for every intel key -- callers never need to special-case this."""
    eng = _engine(cfg)
    assert eng._last_intel_ctx == {}
    ctx = _call_build_ctx(eng, cfg)
    assert ctx["events"] == _EMPTY_EVENTS
    assert ctx["vol_surface"] == {}
    assert ctx["breaking"] == _EMPTY_BREAKING
    assert ctx["regime"]["label"] == "UNKNOWN"
    assert ctx["calendar_status"] == _EMPTY_CALENDAR_STATUS


# ─────────────────────────────────────────────────────────────────────────
# _review_triggers Task 3(a): new surprise
# ─────────────────────────────────────────────────────────────────────────

def _pos_with_review(last_review_at=None, entry_time=1000.0, underlying="SPY"):
    return _position(
        "P1", underlying, f"{underlying}TESTC00500000",
        next_review_at=_base_dt().timestamp() + 999999,  # not "due"
        last_review_at=last_review_at, entry_time=entry_time,
    )


def test_review_trigger_new_surprise_with_high_z_after_last_review(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=now - 3600)
    eng._last_intel_ctx = {"events": {"upcoming": [], "recent_surprises": [{
        "event": "PPI", "release_time_et": _iso(now - 60), "surprise_z": 1.8, "importance": 3, "direction": "above",
    }]}}
    triggers = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_surprise" in triggers


def test_review_trigger_no_surprise_trigger_when_released_before_last_review(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=now - 60)  # reviewed recently
    eng._last_intel_ctx = {"events": {"upcoming": [], "recent_surprises": [{
        "event": "PPI", "release_time_et": _iso(now - 3600), "surprise_z": 2.5, "importance": 3, "direction": "above",
    }]}}
    triggers = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_surprise" not in triggers


def test_review_trigger_no_surprise_trigger_when_z_below_threshold(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=now - 3600)
    eng._last_intel_ctx = {"events": {"upcoming": [], "recent_surprises": [{
        "event": "PPI", "release_time_et": _iso(now - 60), "surprise_z": 0.9, "importance": 3, "direction": "above",
    }]}}
    triggers = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_surprise" not in triggers


def test_review_trigger_new_surprise_z_none_importance_3_non_inline(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=now - 3600)
    eng._last_intel_ctx = {"events": {"upcoming": [], "recent_surprises": [{
        "event": "NFP", "release_time_et": _iso(now - 60), "surprise_z": None, "importance": 3, "direction": "below",
    }]}}
    triggers = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_surprise" in triggers


def test_review_trigger_no_surprise_when_z_none_and_inline(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=now - 3600)
    eng._last_intel_ctx = {"events": {"upcoming": [], "recent_surprises": [{
        "event": "NFP", "release_time_et": _iso(now - 60), "surprise_z": None, "importance": 3, "direction": "inline",
    }]}}
    triggers = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_surprise" not in triggers


def test_review_trigger_surprise_baseline_falls_back_to_entry_time(cfg):
    """No review has happened yet (last_review_at is None) -- baseline falls
    back to entry_time, never to epoch 0 (which would spuriously fire on
    every historical surprise ever recorded)."""
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=None, entry_time=now - 120)
    eng._last_intel_ctx = {"events": {"upcoming": [], "recent_surprises": [
        {"event": "old", "release_time_et": _iso(now - 3600), "surprise_z": 3.0, "importance": 3, "direction": "above"},
        {"event": "new", "release_time_et": _iso(now - 30), "surprise_z": 1.9, "importance": 3, "direction": "above"},
    ]}}
    triggers = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_surprise" in triggers  # the "new" one alone is enough to fire


# ─────────────────────────────────────────────────────────────────────────
# _review_triggers Task 3(b): new breaking event
# ─────────────────────────────────────────────────────────────────────────

def test_review_trigger_new_breaking_event_severity_2_affecting_underlying(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=now - 60, underlying="SPY")
    eng._last_intel_ctx = {"breaking": {"active": True, "events": [
        {"id": "e1", "severity": 2, "tickers": ["SPY"]},
    ]}}
    triggers = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_breaking_event" in triggers


def test_review_trigger_breaking_event_macro_wide_no_tickers_still_fires(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=now - 60, underlying="SPY")
    eng._last_intel_ctx = {"breaking": {"active": True, "events": [
        {"id": "e2", "severity": 3, "tickers": []},
    ]}}
    triggers = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_breaking_event" in triggers


def test_review_trigger_breaking_event_wrong_ticker_does_not_fire(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=now - 60, underlying="SPY")
    eng._last_intel_ctx = {"breaking": {"active": True, "events": [
        {"id": "e3", "severity": 3, "tickers": ["TSLA"]},
    ]}}
    triggers = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_breaking_event" not in triggers


def test_review_trigger_breaking_event_low_severity_does_not_fire(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=now - 60, underlying="SPY")
    eng._last_intel_ctx = {"breaking": {"active": True, "events": [
        {"id": "e4", "severity": 1, "tickers": ["SPY"]},
    ]}}
    triggers = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_breaking_event" not in triggers


def test_review_trigger_breaking_event_only_fires_once_per_position(cfg):
    """The same event id must never re-fire the trigger for the SAME
    position across cycles (the 'not seen before for this position' rule)."""
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    now = _base_dt().timestamp()
    pos = _pos_with_review(last_review_at=now - 60, underlying="SPY")
    eng._last_intel_ctx = {"breaking": {"active": True, "events": [{"id": "e5", "severity": 3, "tickers": ["SPY"]}]}}

    first = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_breaking_event" in first

    second = eng._review_triggers(pos, mark=2.0, now=now, jev_escalated=False, snap=snap, underlyings_ctx={}, cfg=cfg)
    assert "new_breaking_event" not in second


# Seen-set cleanup on close is exercised through a real cycle in
# tests/dpof/test_intel_ctx_review_fixes.py (swept for every non-open position).


def test_review_triggers_never_place_orders(cfg):
    """Grep-provable (spec convention): _review_triggers is a pure trigger-
    naming function -- it must never reference the broker/order manager."""
    src = inspect.getsource(DPOFEngine._review_triggers)
    for forbidden in ("self._broker", "self._orders", "submit_order", "open_position", "close_position"):
        assert forbidden not in src


# ─────────────────────────────────────────────────────────────────────────
# _cooldown_allows Task 3: new-surprise bypass
# ─────────────────────────────────────────────────────────────────────────

def _bullish_signal(composite=0.5) -> Signal:
    return Signal(symbol="SPY", direction=Direction.BULLISH, composite=composite, conviction=1.0, strength="strong")


def test_cooldown_bypassed_by_new_high_z_surprise_after_set_at(cfg):
    eng = _engine(cfg)
    now = _base_dt().timestamp()
    key = "SPY:bullish"
    eng._entry_cooldowns[key] = {
        "next_allowed_at": now + 3600, "recheck_at": None, "last_composite": 0.5, "set_at": now - 120,
    }
    eng._last_intel_ctx = {"events": {"recent_surprises": [{
        "event": "CPI", "release_time_et": _iso(now - 30), "surprise_z": 2.0,
    }]}}
    assert eng._cooldown_allows("SPY", _bullish_signal(), now, cfg) is True


def test_cooldown_not_bypassed_by_surprise_before_set_at(cfg):
    eng = _engine(cfg)
    now = _base_dt().timestamp()
    key = "SPY:bullish"
    eng._entry_cooldowns[key] = {
        "next_allowed_at": now + 3600, "recheck_at": None, "last_composite": 0.5, "set_at": now - 30,
    }
    eng._last_intel_ctx = {"events": {"recent_surprises": [{
        "event": "CPI", "release_time_et": _iso(now - 120), "surprise_z": 2.0,
    }]}}
    assert eng._cooldown_allows("SPY", _bullish_signal(), now, cfg) is False


def test_cooldown_not_bypassed_by_low_z_surprise(cfg):
    eng = _engine(cfg)
    now = _base_dt().timestamp()
    key = "SPY:bullish"
    eng._entry_cooldowns[key] = {
        "next_allowed_at": now + 3600, "recheck_at": None, "last_composite": 0.5, "set_at": now - 120,
    }
    eng._last_intel_ctx = {"events": {"recent_surprises": [{
        "event": "CPI", "release_time_et": _iso(now - 30), "surprise_z": 0.5,
    }]}}
    assert eng._cooldown_allows("SPY", _bullish_signal(), now, cfg) is False


def test_cooldown_still_open_when_no_record_exists(cfg):
    eng = _engine(cfg)
    now = _base_dt().timestamp()
    assert eng._cooldown_allows("SPY", _bullish_signal(), now, cfg) is True


def test_cooldown_allows_never_places_orders(cfg):
    src = inspect.getsource(DPOFEngine._cooldown_allows)
    for forbidden in ("self._broker", "self._orders", "submit_order", "open_position", "close_position"):
        assert forbidden not in src


# ─────────────────────────────────────────────────────────────────────────
# _parse_iso_epoch
# ─────────────────────────────────────────────────────────────────────────

def test_parse_iso_epoch_none_and_empty_are_none():
    assert _parse_iso_epoch(None) is None
    assert _parse_iso_epoch("") is None


def test_parse_iso_epoch_malformed_is_none():
    assert _parse_iso_epoch("not-a-timestamp") is None


def test_parse_iso_epoch_naive_assumed_et():
    from datetime import datetime
    from agent.dpof.engine import ET
    epoch = _parse_iso_epoch("2026-09-25T12:00:00")
    assert epoch == datetime(2026, 9, 25, 12, 0, 0, tzinfo=ET).timestamp()


def _iso(epoch: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
