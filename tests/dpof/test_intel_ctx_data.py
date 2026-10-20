"""Tests for agent.dpof.data.DPOFDataFeed.get_intel_ctx() (Task 2,
INTEL_V3_SPEC.md): advisory events/vol_surface/breaking/calendar_status
context for the decider, assembled from Builders E/V/N's landed modules
(agent.integrations.econ_surprise / vol_surface / newsflash_lite) with a
harmless fallback whenever any of them is absent or raises.

Plugin-free (no pytest-asyncio): async code is driven with asyncio.run().
E/V/N's own public reader functions are monkeypatched directly on the real
(already-landed) module objects -- this exercises ONLY data.py's own call/
fallback wiring, never E/V/N's internal file-parsing (that is their own test
suites' job). "Absent" (simulating a sibling module not landed yet, which is
how G was built against the shared INTEL_V3_SPEC.md contract while E/V/N
were still in flight) is simulated via `sys.modules[<name>] = None`, which
makes Python's import machinery raise ImportError on `import <name>` --
exactly the failure mode data.py's lazy imports must survive.

econ_calendar/learning_engine are always explicitly monkeypatched to a
tmp_path file (never the real module defaults, which resolve to real
on-disk/deployment paths) -- hermetic per CLAUDE.md / the shared build spec.
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timezone

import pytest

import agent.dpof.data as data

pytestmark = pytest.mark.filterwarnings("ignore")


def _run(coro):
    return asyncio.run(coro)


class _DummyBroker:
    """get_intel_ctx() never touches the broker -- placeholder only, to
    satisfy DPOFDataFeed's constructor."""


def _feed(cfg) -> data.DPOFDataFeed:
    return data.DPOFDataFeed(cfg, _DummyBroker())


@pytest.fixture(autouse=True)
def _isolate_intel_modules(monkeypatch):
    """Every test starts from a clean import slate for the 3 advisory
    modules -- undoes any `sys.modules[name] = None` absence-simulation a
    prior test left behind, and undoes real-module monkeypatches via
    pytest's own monkeypatch teardown.

    Also clears the cached attribute `agent.integrations.<name>` that a real
    `import agent.integrations.<name>` leaves on the PARENT package object.
    Without this, `from agent.integrations import <name>` (data.py's own
    lazy-import form) resolves that attribute directly and never re-consults
    sys.modules -- so a later `sys.modules[full_name] = None` absence
    simulation would silently do nothing once a prior test in the same
    process had really imported that module once.
    """
    import agent.integrations as integrations_pkg

    for name in ("econ_surprise", "vol_surface", "newsflash_lite"):
        monkeypatch.delitem(sys.modules, f"agent.integrations.{name}", raising=False)
        monkeypatch.delattr(integrations_pkg, name, raising=False)
    yield


def _fake_econ_calendar(monkeypatch, tmp_path, events, present=True):
    import agent.integrations.econ_calendar as econ_calendar

    path = tmp_path / "econ_calendar.json"
    if present:
        path.write_text("{}")
    monkeypatch.setattr(econ_calendar, "CALENDAR_CACHE_PATH", str(path))
    monkeypatch.setattr(econ_calendar, "load_calendar", lambda: events)
    return econ_calendar


def _fake_learning_engine(monkeypatch, tmp_path, state, present=True):
    import agent.integrations.learning_engine as learning_engine

    path = tmp_path / "learning_state.json"
    if present:
        path.write_text("{}")
    monkeypatch.setattr(learning_engine, "LEARNING_STATE_FILE", str(path))
    monkeypatch.setattr(learning_engine, "_load_learning_state", lambda: state)
    return learning_engine


# ═══════════════════════════════════════════════════════════════════════════
#  events (Builder E: econ_surprise)
# ═══════════════════════════════════════════════════════════════════════════

def test_events_from_econ_surprise_when_present(cfg, monkeypatch):
    import agent.integrations.econ_surprise as econ_surprise

    upcoming = [{
        "event": "CPI", "release_time_et": "2026-10-13T08:30:00-04:00",
        "importance": 3, "consensus": 3.1, "previous": 3.0, "minutes_until": 45.0,
    }]
    surprises = [{
        "event": "PPI", "release_time_et": "2026-09-24T08:30:00-04:00",
        "surprise_z": 2.1, "importance": 3, "direction": "above",
        "as_of": "2026-09-24T08:31:00-04:00",
    }]
    monkeypatch.setattr(econ_surprise, "get_upcoming_releases", lambda hours: upcoming)
    monkeypatch.setattr(econ_surprise, "get_recent_surprises", lambda max_age_minutes: surprises)

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["events"]["source"] == "econ_surprise"
    assert intel["events"]["upcoming"] == upcoming
    assert intel["events"]["recent_surprises"] == surprises


def test_events_fall_back_to_econ_calendar_when_econ_surprise_absent(cfg, monkeypatch, tmp_path):
    sys.modules["agent.integrations.econ_surprise"] = None  # not-landed-yet simulation

    now_epoch = 1_800_000_000.0
    now_et = datetime.fromtimestamp(now_epoch, tz=timezone.utc).astimezone(data.ET)
    soon_iso = now_et.replace(microsecond=0).isoformat()
    _fake_econ_calendar(monkeypatch, tmp_path, events=[
        {"name": "FOMC Rate Decision", "importance": 3, "datetime_et": soon_iso, "forecast": "5.25", "previous": "5.00"},
        {"name": "Consumer Sentiment", "importance": 2, "datetime_et": soon_iso, "forecast": None, "previous": "70.1"},
    ])

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=now_epoch))
    assert intel["events"]["source"] == "econ_calendar_fallback"
    assert intel["events"]["recent_surprises"] == []
    upcoming = intel["events"]["upcoming"]
    assert len(upcoming) == 2  # unlike the old blackout gate, ALL importances surface, not just 3
    fomc = next(e for e in upcoming if e["event"] == "FOMC Rate Decision")
    assert fomc["consensus"] == "5.25"
    assert fomc["previous"] == "5.00"
    assert fomc["importance"] == 3
    assert fomc["minutes_until"] == pytest.approx(0.0, abs=1.0)
    sentiment = next(e for e in upcoming if e["event"] == "Consumer Sentiment")
    assert sentiment["consensus"] is None  # falsy forecast -> None, never a fabricated value


def test_events_fall_back_and_empty_when_econ_calendar_also_absent(cfg, monkeypatch, tmp_path):
    sys.modules["agent.integrations.econ_surprise"] = None
    _fake_econ_calendar(monkeypatch, tmp_path, events=[], present=False)

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["events"] == {"upcoming": [], "recent_surprises": [], "source": "econ_calendar_fallback"}


def test_events_never_raises_when_econ_surprise_getters_raise(cfg, monkeypatch, tmp_path):
    import agent.integrations.econ_surprise as econ_surprise

    def _boom(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(econ_surprise, "get_upcoming_releases", _boom)
    monkeypatch.setattr(econ_surprise, "get_recent_surprises", _boom)
    _fake_econ_calendar(monkeypatch, tmp_path, events=[], present=False)

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))  # must not raise
    assert intel["events"]["source"] == "econ_calendar_fallback"
    assert intel["events"]["recent_surprises"] == []


# ═══════════════════════════════════════════════════════════════════════════
#  vol_surface (Builder V)
# ═══════════════════════════════════════════════════════════════════════════

def test_vol_surface_from_module_per_symbol_when_present(cfg, monkeypatch):
    import agent.integrations.vol_surface as vol_surface

    readings = {
        "SPY": {
            "symbol": "SPY", "iv_front": 18.2, "iv_30d": 17.5, "term_slope": 0.8,
            "term_state": "backwardation", "rv_20d": 14.9, "iv_rv_spread": 3.3,
            "iv_rank": 62.0, "skew_25d": 4.1, "richness": "rich",
            "as_of": "2026-09-25T12:00:00-04:00", "sources": ["uw"],
        },
    }
    monkeypatch.setattr(vol_surface, "get_vol_surface", lambda sym, max_age_seconds=3600: readings.get(sym))

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["vol_surface"]["SPY"] == readings["SPY"]
    assert intel["vol_surface"]["QQQ"] is None  # no reading for QQQ -> None, never fabricated


def test_vol_surface_all_none_when_module_absent(cfg):
    sys.modules["agent.integrations.vol_surface"] = None
    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["vol_surface"] == {sym: None for sym in cfg.underlyings}


def test_vol_surface_never_raises_when_getter_raises(cfg, monkeypatch):
    import agent.integrations.vol_surface as vol_surface

    def _boom(sym, max_age_seconds=3600):
        raise RuntimeError("boom")

    monkeypatch.setattr(vol_surface, "get_vol_surface", _boom)
    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["vol_surface"] == {sym: None for sym in cfg.underlyings}


# ═══════════════════════════════════════════════════════════════════════════
#  breaking (Builder N: newsflash_lite, falling back to Lighthouse/
#  learning_state -- the existing breaking-news read this module already
#  performs for `_check_breaking_news`, normalized to N's contract shape)
# ═══════════════════════════════════════════════════════════════════════════

def test_breaking_from_newsflash_lite_when_present(cfg, monkeypatch):
    import agent.integrations.newsflash_lite as newsflash_lite

    state = {
        "active": True, "max_severity": 3,
        "events": [{"id": "e1", "headline": "x", "category": "geopolitical", "tier": 1, "severity": 3,
                    "confidence": 0.8, "source": "reuters", "ts": "2026-09-25T12:00:00Z",
                    "tickers": ["SPY"], "direction_hint": "risk_off"}],
        "as_of": "2026-09-25T12:00:00Z", "stale": False,
    }
    monkeypatch.setattr(newsflash_lite, "get_breaking_state", lambda max_age_seconds=900: state)

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["breaking"] == state


def test_breaking_falls_back_to_lighthouse_when_newsflash_lite_absent(cfg, monkeypatch, tmp_path):
    sys.modules["agent.integrations.newsflash_lite"] = None
    _fake_learning_engine(monkeypatch, tmp_path, state={"breaking_news": {
        "active": True, "triggered_at": "2026-09-25T12:00:00-04:00",
        "event_type": "geopolitical_shock", "spy_pct": -2.4, "vix_change": 9.0,
    }})

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    breaking = intel["breaking"]
    assert breaking["active"] is True
    assert breaking["max_severity"] == 3  # |spy_pct| >= 2.0 (or |vix_change| >= 8.0) -> severity 3
    assert breaking["source"] == "lighthouse_fallback"
    assert breaking["events"][0]["category"] == "geopolitical_shock"
    assert breaking["events"][0]["tickers"] == []  # Lighthouse's state is market-wide, never per-ticker


def test_breaking_lighthouse_fallback_moderate_severity_for_a_smaller_move(cfg, monkeypatch, tmp_path):
    sys.modules["agent.integrations.newsflash_lite"] = None
    _fake_learning_engine(monkeypatch, tmp_path, state={"breaking_news": {
        "active": True, "triggered_at": "2026-09-25T12:00:00-04:00",
        "event_type": "earnings_miss", "spy_pct": -0.6, "vix_change": 1.2,
    }})

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["breaking"]["max_severity"] == 2  # below both thresholds -> the moderate default


def test_breaking_falls_back_and_inactive_when_lighthouse_state_absent_too(cfg, monkeypatch, tmp_path):
    sys.modules["agent.integrations.newsflash_lite"] = None
    _fake_learning_engine(monkeypatch, tmp_path, state={}, present=False)

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["breaking"] == {
        "active": False, "max_severity": 0, "events": [], "as_of": None,
        "stale": True, "source": "lighthouse_fallback",
    }


def test_breaking_never_raises_when_newsflash_lite_getter_raises(cfg, monkeypatch, tmp_path):
    import agent.integrations.newsflash_lite as newsflash_lite

    def _boom(max_age_seconds=900):
        raise RuntimeError("boom")

    monkeypatch.setattr(newsflash_lite, "get_breaking_state", _boom)
    _fake_learning_engine(monkeypatch, tmp_path, state={}, present=False)

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))  # must not raise
    assert intel["breaking"]["source"] == "lighthouse_fallback"
    assert intel["breaking"]["active"] is False


# ═══════════════════════════════════════════════════════════════════════════
#  calendar_status (advisory, never a data_healthy input -- see test_data.py
#  for the "no longer forces unhealthy" side of Task 1)
# ═══════════════════════════════════════════════════════════════════════════

def test_calendar_status_ok_when_cache_present_and_fresh(cfg, monkeypatch, tmp_path):
    _fake_econ_calendar(monkeypatch, tmp_path, events=[], present=True)
    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    assert intel["calendar_status"]["status"] == "ok"
    assert intel["calendar_status"]["age_hours"] < 1.0
    assert intel["calendar_status"]["sources"] == ["econ_calendar"]


# ═══════════════════════════════════════════════════════════════════════════
#  Complete-shape guarantee (all sources absent at once) + full-cycle
#  wiring smoke test
# ═══════════════════════════════════════════════════════════════════════════

def test_intel_ctx_complete_shape_when_everything_absent(cfg, monkeypatch, tmp_path):
    sys.modules["agent.integrations.econ_surprise"] = None
    sys.modules["agent.integrations.vol_surface"] = None
    sys.modules["agent.integrations.newsflash_lite"] = None
    _fake_econ_calendar(monkeypatch, tmp_path, events=[], present=False)
    _fake_learning_engine(monkeypatch, tmp_path, state={}, present=False)

    intel = _run(_feed(cfg).get_intel_ctx(now_epoch=1_800_000_000.0))
    # Task K2 (agent/kronos): "kronos" joins the complete shape when enabled
    # (cfg's default) -- agent.kronos.state itself is real (not simulated
    # absent here), but the repo-wide test isolation points its files at a
    # nonexistent tmp path, so every symbol reads "unavailable".
    assert set(intel.keys()) == {"events", "vol_surface", "breaking", "calendar_status", "kronos"}
    assert intel["events"] == {"upcoming": [], "recent_surprises": [], "source": "econ_calendar_fallback"}
    assert intel["vol_surface"] == {sym: None for sym in cfg.underlyings}
    assert intel["breaking"]["active"] is False
    assert intel["calendar_status"]["status"] == "missing"
    assert intel["kronos"] == {sym: "unavailable" for sym in cfg.underlyings}
