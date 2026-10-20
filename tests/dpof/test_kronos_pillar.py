"""Kronos LIVE pillar (docs/design/clio_kronos.md §4b; Dom 2026-09-26 "We build Kronos for live use").

The pillar is 2*p_up - 1 at the configured horizon from a FRESH forecast only, weighed beside the flow
pillars in compute_composite's weighted average, never counted toward min_pillars, and absent (never
0.0) whenever the forecast is off, stale, missing or unusable.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import agent.dpof.data as data
import agent.kronos.state as kronos_state
from agent.dpof.config import DPOFConfig, load_dpof_config
from agent.dpof.models import PILLAR_NAMES, MarketSnapshot, PillarReading, UnderlyingSnapshot
from agent.dpof.signals import KRONOS_PILLAR, compute_composite, kronos_pillar_from_forecast

ET = ZoneInfo("America/New_York")


def _entry(**horizons):
    return {"status": "ok", "asof_bar": "2026-09-28T10:35:00-04:00",
            "horizons": {h: {"minutes": int(h), "p_up": p} for h, p in horizons.items()}}


def _u(**pillars) -> UnderlyingSnapshot:
    return UnderlyingSnapshot(
        symbol="SPY",
        pillars={name: PillarReading(name=name, value=v, available=True, source="t") for name, v in pillars.items()},
    )


# ── pure mapping ─────────────────────────────────────────────────────────

def test_value_is_two_p_up_minus_one_at_the_requested_horizon():
    value, detail = kronos_pillar_from_forecast(_entry(**{"15": 0.9, "30": 0.75, "60": 0.1}), 30)
    assert value == pytest.approx(0.5)
    assert detail["horizon_min"] == 30 and detail["p_up"] == 0.75


def test_falls_back_to_the_longest_shorter_horizon_near_the_close():
    value, detail = kronos_pillar_from_forecast(_entry(**{"15": 0.25}), 30)
    assert value == pytest.approx(-0.5) and detail["horizon_min"] == 15
    # A longer horizon never stands in for a shorter one that was asked for.
    assert kronos_pillar_from_forecast(_entry(**{"60": 0.9}), 30) is None


@pytest.mark.parametrize("bad", [None, True, "0.6", float("nan"), -0.1, 1.2])
def test_unusable_p_up_is_skipped_never_zero(bad):
    assert kronos_pillar_from_forecast(_entry(**{"30": bad}), 30) is None
    value, detail = kronos_pillar_from_forecast(_entry(**{"30": bad, "15": 0.6}), 30)
    assert detail["horizon_min"] == 15 and value == pytest.approx(0.2)


def test_garbage_entries_give_none():
    for entry in (None, {}, {"horizons": []}, {"horizons": {"30": "x"}}):
        assert kronos_pillar_from_forecast(entry, 30) is None


# ── composite ────────────────────────────────────────────────────────────

def test_kronos_is_not_a_flow_pillar():
    # PILLAR_NAMES drives UW caches, the self-flow fallback trigger and data_healthy: Kronos stays out.
    assert KRONOS_PILLAR not in PILLAR_NAMES


def test_composite_weighs_kronos_beside_the_flow_pillars(cfg):
    c = replace(cfg, kronos_pillar="live", w_kronos=0.15)
    u = _u(nofa=0.5, premium=0.2, darkpool=-0.1, kronos=0.8)
    composite, used, reasons = compute_composite(u, c)
    expected = (0.40 * 0.5 + 0.25 * 0.2 + 0.20 * -0.1 + 0.15 * 0.8) / (0.40 + 0.25 + 0.20 + 0.15)
    assert composite == pytest.approx(expected)
    assert used["kronos"] == 0.8 and reasons == []


def test_composite_without_kronos_is_unchanged(cfg):
    c = replace(cfg, kronos_pillar="live")
    base = compute_composite(_u(nofa=0.5, premium=0.2, darkpool=-0.1), c)
    off = compute_composite(_u(nofa=0.5, premium=0.2, darkpool=-0.1, kronos=0.8), replace(c, kronos_pillar="off"))
    assert base == off
    assert "kronos" not in off[1]


def test_kronos_never_counts_toward_min_pillars(cfg):
    c = replace(cfg, kronos_pillar="live", min_pillars=3)
    composite, used, reasons = compute_composite(_u(nofa=0.5, premium=0.2, kronos=0.9), c)
    assert composite is None
    assert reasons == ["insufficient_pillars:2<3"]


def test_kronos_can_move_the_composite_across_a_threshold(cfg):
    # The point of a live pillar: it changes nominations. Flow alone is just under threshold_normal (.15).
    c = replace(cfg, kronos_pillar="live", w_kronos=0.15)
    flow_only, _, _ = compute_composite(_u(nofa=0.2, premium=0.1, darkpool=0.1), c)
    with_kronos, _, _ = compute_composite(_u(nofa=0.2, premium=0.1, darkpool=0.1, kronos=0.9), c)
    assert flow_only < c.threshold_normal < with_kronos


# ── config ───────────────────────────────────────────────────────────────

def test_defaults_are_live_point_fifteen_thirty():
    d = DPOFConfig()
    assert (d.kronos_pillar, d.w_kronos, d.kronos_horizon) == ("live", 0.15, 30)
    assert d.validate() == [] or all("kronos" not in p for p in d.validate())


@pytest.mark.parametrize("env,expected", [
    ({"CLIO_DPOF_KRONOS_PILLAR": "OFF"}, ("off", 0.15, 30)),
    ({"CLIO_DPOF_KRONOS_PILLAR": "shadow"}, ("off", 0.15, 30)),    # typo degrades to off, never invalid
    ({"CLIO_DPOF_W_KRONOS": "2.5"}, ("live", 1.0, 30)),
    ({"CLIO_DPOF_W_KRONOS": "-1"}, ("live", 0.0, 30)),
    ({"CLIO_DPOF_KRONOS_HORIZON": "45"}, ("live", 0.15, 30)),
    ({"CLIO_DPOF_KRONOS_HORIZON": "60"}, ("live", 0.15, 60)),
])
def test_bad_env_values_normalize_so_the_cycle_never_stops(monkeypatch, env, expected):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    c = load_dpof_config()
    assert (c.kronos_pillar, c.w_kronos, c.kronos_horizon) == expected
    assert not [p for p in c.validate() if "kronos" in p]


def test_validate_flags_programmatic_bad_values():
    bad = replace(DPOFConfig(), kronos_pillar="maybe", w_kronos=1.5, kronos_horizon=45)
    problems = " ".join(bad.validate())
    assert "kronos_pillar" in problems and "w_kronos" in problems and "kronos_horizon" in problems


# ── snapshot injection ───────────────────────────────────────────────────

class _DummyBroker:
    pass


def _write_forecast(tmp_path, monkeypatch, written_at, symbols):
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(tmp_path / "kronos_forecast.json"))
    with kronos_state._cache_lock:
        kronos_state._cache.clear()
    kronos_state.write_json_atomic(kronos_state.forecast_path(), {
        "schema": kronos_state.SCHEMA, "written_at": written_at, "model": {}, "params": {}, "symbols": symbols,
    })


def _attach(cfg, now):
    feed = data.DPOFDataFeed(cfg, _DummyBroker())
    unds = {"SPY": UnderlyingSnapshot(symbol="SPY"), "QQQ": UnderlyingSnapshot(symbol="QQQ")}
    asyncio.run(feed._attach_kronos_pillar(unds, now))
    return unds


def test_fresh_forecast_attaches_the_pillar(cfg, tmp_path, monkeypatch):
    now = 1_800_000_000.0
    _write_forecast(tmp_path, monkeypatch, now - 30, {"SPY": _entry(**{"30": 0.7}), "QQQ": {"status": "error"}})
    unds = _attach(replace(cfg, underlyings=("SPY", "QQQ"), kronos_pillar="live"), now)
    reading = unds["SPY"].pillars[KRONOS_PILLAR]
    assert reading.available and reading.value == pytest.approx(0.4) and reading.source == "kronos"
    assert KRONOS_PILLAR not in unds["QQQ"].pillars


def test_stale_or_off_attaches_nothing(cfg, tmp_path, monkeypatch):
    now = 1_800_000_000.0
    _write_forecast(tmp_path, monkeypatch, now - 3600, {"SPY": _entry(**{"30": 0.7})})
    assert KRONOS_PILLAR not in _attach(replace(cfg, underlyings=("SPY", "QQQ")), now)["SPY"].pillars
    _write_forecast(tmp_path, monkeypatch, now - 30, {"SPY": _entry(**{"30": 0.7})})
    off = _attach(replace(cfg, underlyings=("SPY", "QQQ"), kronos_pillar="off"), now)
    assert KRONOS_PILLAR not in off["SPY"].pillars


def test_a_broken_reader_attaches_nothing_and_never_raises(cfg, monkeypatch):
    def boom(*_a, **_kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(data, "_kronos_pillar_read_sync", boom)
    unds = _attach(replace(cfg, underlyings=("SPY", "QQQ")), 1_800_000_000.0)
    assert all(KRONOS_PILLAR not in u.pillars for u in unds.values())


# ── stale alert ──────────────────────────────────────────────────────────

class _Alerts:
    def __init__(self):
        self.sent = []

    async def send(self, bot, text, dedupe_key=None, dedupe_seconds=86400):
        self.sent.append((text, dedupe_key))


def _engine(cfg, alerts):
    from agent.dpof.engine import DPOFEngine

    return DPOFEngine(broker=None, feed=None, alerts=alerts, config_loader=lambda: cfg)


def _snap(with_kronos):
    u = UnderlyingSnapshot(symbol="SPY")
    if with_kronos:
        u.pillars[KRONOS_PILLAR] = PillarReading(name=KRONOS_PILLAR, value=0.2, available=True, source="kronos")
    return MarketSnapshot(asof=0.0, underlyings={"SPY": u})


def test_stale_alert_fires_once_after_fifteen_minutes_in_the_entry_window(cfg):
    alerts = _Alerts()
    c = replace(cfg, kronos_pillar="live")
    eng = _engine(c, alerts)
    now_et = datetime(2026, 9, 28, 11, 0, tzinfo=ET)
    t0 = now_et.timestamp()

    async def _t():
        await eng._maybe_alert_kronos_stale(_snap(False), t0, now_et, c, None)
        await eng._maybe_alert_kronos_stale(_snap(False), t0 + 600, now_et, c, None)
        assert alerts.sent == []
        await eng._maybe_alert_kronos_stale(_snap(False), t0 + 900, now_et, c, None)
        assert len(alerts.sent) == 1 and alerts.sent[0][1] == "dpof_kronos_stale_2026-09-28"
        assert "clio-kronos.service" in alerts.sent[0][0]
        # A fresh forecast resets the clock.
        await eng._maybe_alert_kronos_stale(_snap(True), t0 + 960, now_et, c, None)
        assert eng._kronos_missing_since is None

    asyncio.run(_t())


def test_stale_alert_silent_when_off_or_outside_the_window(cfg):
    alerts = _Alerts()
    eng = _engine(cfg, alerts)
    early = datetime(2026, 9, 28, 8, 0, tzinfo=ET)
    midday = datetime(2026, 9, 28, 11, 0, tzinfo=ET)

    async def _t():
        for i in range(5):
            await eng._maybe_alert_kronos_stale(_snap(False), early.timestamp() + i * 900, early, cfg, None)
            off = replace(cfg, kronos_pillar="off")
            await eng._maybe_alert_kronos_stale(_snap(False), midday.timestamp() + i * 900, midday, off, None)
        assert alerts.sent == []

    asyncio.run(_t())
