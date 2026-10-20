"""Tests for agent/dpof/dynamic_weights.py (DynamicWeightsManager). All files live under tmp_path."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import random
import threading
import time
from zoneinfo import ZoneInfo

import pytest

from agent.dpof import dynamic_weights as dwmod
from agent.dpof.config import DPOFConfig
from agent.dpof.dynamic_weights import DynamicWeightsManager, _default_session
from agent.dpof.models import PillarReading, UnderlyingSnapshot
from agent.dpof.pillar_weights import QualityStats
from agent.dpof.pillar_weights_io import ShadowLog, StateStore
from agent.dpof.signals import compute_composite
from agent.integrations.market_events import MARKET_EARLY_CLOSE, MARKET_HOLIDAYS

ET = ZoneInfo("America/New_York")
DAY = "2026-10-06"  # Tuesday
DAY2 = "2026-10-07"
LOGGER = "clio.dpof.dynamic_weights"


def et_epoch(h: int, m: int, s: int = 0, day: str = DAY) -> float:
    y, mo, d = (int(x) for x in day.split("-"))
    return dt.datetime(y, mo, d, h, m, s, tzinfo=ET).timestamp()


def fixed(day: str = DAY, in_session: bool = True):
    return lambda now: (day, in_session)


def make_cfg(tmp_path, mode: str = "shadow", **kw):
    base = dict(data_dir=str(tmp_path), dynamic_weights=mode, dw_ring_size=10, kronos_pillar="live")
    base.update(kw)
    return dataclasses.replace(DPOFConfig(), **base)


def snap(sym="SPY", nofa=0.3, premium=0.2, darkpool=0.1, kronos=0.15, premium_source="uw:/flow",
         last_price=100.0, age=5.0, ret=0.002, gex=None, pin=None):
    pillars = {}

    def add(name, val, src):
        if val is None:
            pillars[name] = PillarReading.missing(name, "absent", src)
        else:
            pillars[name] = PillarReading(name=name, value=val, available=True, source=src)

    add("nofa", nofa, "uw:/nofa")
    add("premium", premium, premium_source)
    add("darkpool", darkpool, "uw:/darkpool")
    add("kronos", kronos, "kronos")
    return UnderlyingSnapshot(symbol=sym, pillars=pillars, last_price=last_price,
                              price_age_seconds=age, ret_15m=ret, gex_regime=gex, pin_risk=pin)


def pillar_table(u):
    return {k: (v.name, v.value, v.available, v.source) for k, v in u.pillars.items()}


def kill_premium(m, sym, base, n=10, step=180.0):
    """Feed n zero observations at distinct in-session timestamps so premium is dead."""
    for i in range(n):
        m.observe_uw(sym, "premium", 0.0, base + i * step)
    return base + (n - 1) * step


def last_record(tmp_path, n=1):
    return ShadowLog(str(tmp_path)).read_tail(n)[-1]


# ----------------------------------------------------------------- M1 off mode
@pytest.mark.parametrize("mode", ["off", "bogus"])
def test_off_mode_creates_nothing_and_leaves_snapshot_alone(tmp_path, mode):
    """Mode off (and an invalid mode string) is a no-op: no files, no mask, snapshot untouched."""
    m = DynamicWeightsManager(make_cfg(tmp_path, mode), str(tmp_path), session_fn=fixed())
    assert m.mode == "off"
    m.observe_uw("SPY", "premium", 0.0, et_epoch(10, 0))
    u = snap()
    before = pillar_table(u)
    m.finish_snapshot(et_epoch(10, 0), {"SPY": u})
    asyncio.run(m.ensure_loaded())
    assert m.is_masked("SPY", "premium", et_epoch(10, 0)) is False
    assert os.listdir(tmp_path) == []
    assert u.pillar_weights is None
    assert pillar_table(u) == before


# ----------------------------------------------------------------- M2 shadow, nothing dead
def test_shadow_nothing_dead_dyn_equals_static_for_random_scenarios(tmp_path):
    """Shadow with nothing dead: dyn equals static within 1e-12 across 40 random scenarios."""
    rng = random.Random(20261010)
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    base = et_epoch(10, 0)
    for i in range(40):
        vals = {}
        if rng.random() < 0.85:
            vals["nofa"] = rng.uniform(-1, 1)
        if rng.random() < 0.7:
            vals["premium"] = rng.uniform(-1, 1)
        if rng.random() < 0.7:
            vals["darkpool"] = rng.uniform(-1, 1)
        if rng.random() < 0.6:
            vals["kronos"] = rng.uniform(-1, 1)
        u = snap(nofa=vals.get("nofa"), premium=vals.get("premium"), darkpool=vals.get("darkpool"),
                 kronos=vals.get("kronos"), gex=rng.choice(["positive_gamma", "negative_gamma", None]),
                 pin=rng.choice([None, True, False]))
        before = pillar_table(u)
        now = base + i * 60
        m.finish_snapshot(now, {"SPY": u})
        rec = last_record(tmp_path)
        assert u.pillar_weights is None
        assert pillar_table(u) == before
        assert rec["ts"] == dt.datetime.fromtimestamp(now, ET).isoformat()
        if rec["static"] is None:
            assert rec["dyn"] is None
        else:
            assert abs(rec["dyn"] - rec["static"]) <= 1e-12
    records = ShadowLog(str(tmp_path)).read_tail(100)
    assert len(records) == 40
    assert all(r["u"] == "SPY" and r["mode"] == "shadow" and r["dead"] == [] for r in records)
    assert os.path.exists(os.path.join(str(tmp_path), "pillar_weights_shadow.jsonl"))


# ----------------------------------------------------------------- M3 dead detection, shadow
def test_dead_premium_is_detected_once_logged_and_redistributed_in_shadow(tmp_path, caplog):
    """Ten zero refreshes flag premium DEAD (one deduped WARNING); shadow weights conserve the prior total."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    base = et_epoch(10, 0)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        last = kill_premium(m, "SPY", base)
        dead_warnings = [r for r in caplog.records if "looks DEAD" in r.getMessage()]
        assert len(dead_warnings) == 1
        for i in range(30):
            m.observe_uw("SPY", "premium", 0.0, last + 180 * (i + 1))
        assert len([r for r in caplog.records if "looks DEAD" in r.getMessage()]) == 1
    assert m._liveness.is_dead("SPY|premium|uw")
    assert m.is_masked("SPY", "premium", last) is False  # shadow never masks
    # negative_gamma gives the GEX factor 0.5, so three flow-or-GEX pillars count toward min_pillars.
    u = snap(nofa=0.3, premium=0.0, darkpool=0.1, kronos=0.15, gex="negative_gamma")
    before = pillar_table(u)
    m.finish_snapshot(last + 200, {"SPY": u})
    rec = last_record(tmp_path)
    assert rec["dead"] == ["premium"]
    assert rec["alive"]["premium"] == 0.0
    # Target is the prior total of ALL available pillars (dead included): 0.40+0.25+0.20+0.15 = 1.0.
    # Living shares are 0.40/0.75, 0.20/0.75, 0.15/0.75, so nofa 0.533333, darkpool 0.266667, kronos 0.2.
    assert rec["w"]["premium"] == 0.0
    assert rec["w"]["nofa"] == pytest.approx(0.533333, abs=1e-6)
    assert rec["w"]["darkpool"] == pytest.approx(0.266667, abs=1e-6)
    assert rec["w"]["kronos"] == pytest.approx(0.2, abs=1e-6)
    assert sum(rec["w"].values()) == pytest.approx(1.0, abs=1e-5)
    # dyn: directional mean 13/60 with GEX term 0.15*0.5 over denominator 1.15 -> (7/24)/(23/20) = 35/138.
    # static: directional sum 0.1625 (premium 0.25*0 included) -> (0.1625 + 0.075)/1.15 = 19/92.
    assert rec["dyn"] == pytest.approx(35 / 138, abs=1e-9)
    assert rec["static"] == pytest.approx(19 / 92, abs=1e-9)
    assert rec["static"] != pytest.approx(rec["dyn"])
    assert u.pillar_weights is None
    assert pillar_table(u) == before


# ----------------------------------------------------------------- M4 live mode
def test_live_mode_masks_dead_uw_pillar_and_sets_conserved_weights(tmp_path):
    """Live: is_masked is true only for the dead UW key; weights are set; self-flow readings are never dead."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=fixed(), sync_io=True)
    base = et_epoch(10, 0)
    assert m.is_masked("SPY", "premium", base) is False
    last = kill_premium(m, "SPY", base)
    assert m.is_masked("SPY", "premium", last) is True
    for p in ("nofa", "darkpool", "kronos"):
        assert m.is_masked("SPY", p, last) is False
    assert m.is_masked("QQQ", "premium", last) is False

    u = snap(premium=0.0, gex="negative_gamma")
    m.finish_snapshot(last + 200, {"SPY": u})
    assert isinstance(u.pillar_weights, dict)
    assert u.pillar_weights["premium"] == 0.0
    assert sum(u.pillar_weights.values()) == pytest.approx(1.0, abs=1e-9)
    # dead premium carries weight zero, so compute_composite masks it even though the snapshot still holds it
    assert compute_composite(u, m._cfg)[0] == pytest.approx(35 / 138, abs=1e-9)

    # Self-flow premium is never dead, even though the UW key is dead: weights stay at the priors.
    u2 = snap(premium=0.4, premium_source="selfflow:SPY")
    m.finish_snapshot(last + 200, {"SPY": u2})
    rec = last_record(tmp_path)
    assert rec["dead"] == []
    assert u2.pillar_weights == pytest.approx({"nofa": 0.4, "premium": 0.25, "darkpool": 0.2, "kronos": 0.15}, abs=1e-12)


def test_is_masked_never_true_for_non_liveness_pillars_even_if_tracker_flags_them(tmp_path):
    """is_masked refuses nofa/kronos even when the tracker holds a dead flag for their key (defensive)."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=fixed(), sync_io=True)
    base = et_epoch(10, 0)
    for i in range(10):
        m._liveness.observe("SPY|nofa|uw", 0.0, base + i * 180, DAY, True)
        m._liveness.observe("SPY|kronos|uw", 0.0, base + i * 180, DAY, True)
    assert m._liveness.is_dead("SPY|nofa|uw") and m._liveness.is_dead("SPY|kronos|uw")
    assert m.is_masked("SPY", "nofa", base) is False
    assert m.is_masked("SPY", "kronos", base) is False


def test_live_mode_with_nothing_dead_equals_priors(tmp_path):
    """Live with nothing dead: u.pillar_weights equals the static priors within 1e-12."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=fixed(), sync_io=True)
    u = snap()
    m.finish_snapshot(et_epoch(10, 0), {"SPY": u})
    expect = {"nofa": 0.40, "premium": 0.25, "darkpool": 0.20, "kronos": 0.15}
    assert set(u.pillar_weights) == set(expect)
    for k, v in expect.items():
        assert abs(u.pillar_weights[k] - v) <= 1e-12


# ----------------------------------------------------------------- M5 revival
def test_revival_needs_hysteresis_and_ramps_in_over_three_refreshes(tmp_path):
    """After six refreshes at or above 2*eps (past the dwell), premium revives and ramps 1/3, 2/3, 1."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    base = et_epoch(10, 0)
    t_dead = kill_premium(m, "SPY", base)
    assert m._liveness.is_dead("SPY|premium|uw")
    ramp = []
    for k in range(8):
        ts = t_dead + 1800 + 180 * k
        v = 0.02 + 0.001 * (k % 2)                                  # a frozen value must not revive the pillar
        m.observe_uw("SPY", "premium", v, ts)
        u = snap(premium=v)
        m.finish_snapshot(ts + 1, {"SPY": u})
        ramp.append(last_record(tmp_path)["alive"]["premium"])
    # first five are not yet revived (dead, 0.0), sixth flips alive at 1/3, then 2/3, then 1.0
    assert ramp[:5] == [0.0] * 5
    assert ramp[5] == pytest.approx(1 / 3, abs=1e-6)
    assert ramp[6] == pytest.approx(2 / 3, abs=1e-6)
    assert ramp[7] == pytest.approx(1.0, abs=1e-6)
    assert not m._liveness.is_dead("SPY|premium|uw")


# ----------------------------------------------------------------- M6 kronos excluded
def test_kronos_excluded_when_pillar_off(tmp_path):
    """With kronos_pillar off, kronos is neither available nor weighted."""
    cfg = make_cfg(tmp_path, "shadow", kronos_pillar="off")
    m = DynamicWeightsManager(cfg, str(tmp_path), session_fn=fixed(), sync_io=True)
    u = snap()
    m.finish_snapshot(et_epoch(10, 0), {"SPY": u})
    rec = last_record(tmp_path)
    assert "kronos" not in rec["avail"]
    assert "kronos" not in rec["w"]
    assert rec["static"] == pytest.approx(rec["dyn"], abs=1e-12)


# ----------------------------------------------------------------- M7 persistence
def _save_spy(monkeypatch):
    calls = []
    orig = StateStore.save

    def spy(self, state):
        calls.append(state)
        return orig(self, state)

    monkeypatch.setattr(StateStore, "save", spy)
    return calls


def test_state_persists_throttles_and_restores_by_session(tmp_path, monkeypatch):
    """Save throttle, restore on same day (ring kept), next day (ring empty, dead kept), stale and corrupt state."""
    saves = _save_spy(monkeypatch)
    base = time.time()
    cfg = make_cfg(tmp_path, "shadow")
    m = DynamicWeightsManager(cfg, str(tmp_path), session_fn=fixed(DAY), sync_io=True)
    last = kill_premium(m, "SPY", base)
    m.finish_snapshot(last + 1, {"SPY": snap()})
    path = os.path.join(str(tmp_path), "pillar_weights.json")
    assert os.path.exists(path)
    doc = json.load(open(path))
    assert doc["schema_version"] == 1 and "liveness" in doc and "quality" in doc
    assert doc["liveness"]["keys"]["SPY|premium|uw"]["dead"] is True
    assert len(saves) == 1

    m.observe_uw("SPY", "premium", 0.0, last + 10)
    m.finish_snapshot(last + 20, {"SPY": snap()})
    assert len(saves) == 1  # inside 60 s: throttled
    m.finish_snapshot(last + 90, {"SPY": snap()})
    assert len(saves) == 2

    # same day, fresh state: ring restored and still dead
    m2 = DynamicWeightsManager(cfg, str(tmp_path), session_fn=fixed(DAY), sync_io=True)
    m2.load_sync(time.time())
    assert m2._liveness.is_dead("SPY|premium|uw")
    assert len(m2._liveness.to_dict()["keys"]["SPY|premium|uw"]["ring"]) > 0

    # next ET day: dead flag kept, ring empty
    m3 = DynamicWeightsManager(cfg, str(tmp_path), session_fn=fixed(DAY2), sync_io=True)
    m3.load_sync(time.time())
    assert m3._liveness.is_dead("SPY|premium|uw")
    assert m3._liveness.to_dict()["keys"]["SPY|premium|uw"]["ring"] == []


def test_state_older_than_five_days_is_ignored(tmp_path):
    """A state file stamped more than five days ago is ignored: the pillar starts alive."""
    cfg = make_cfg(tmp_path, "shadow")
    m = DynamicWeightsManager(cfg, str(tmp_path), session_fn=fixed(), sync_io=True)
    base = time.time()
    kill_premium(m, "SPY", et_epoch(10, 0))
    m.finish_snapshot(base, {"SPY": snap()})
    path = os.path.join(str(tmp_path), "pillar_weights.json")
    doc = json.load(open(path))
    doc["written_at"] = base - 6 * 86400
    json.dump(doc, open(path, "w"))
    m2 = DynamicWeightsManager(cfg, str(tmp_path), session_fn=fixed(), sync_io=True)
    m2.load_sync(base)
    assert not m2._liveness.is_dead("SPY|premium|uw")


def test_corrupt_state_is_quarantined_and_manager_starts_empty(tmp_path):
    """A corrupt state file is renamed aside and the manager starts empty without raising."""
    path = os.path.join(str(tmp_path), "pillar_weights.json")
    with open(path, "w") as f:
        f.write("{not json")
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    m.load_sync(time.time())
    assert not m._liveness.is_dead("SPY|premium|uw")
    assert not os.path.exists(path)
    assert any(name.startswith("pillar_weights.json.corrupt-") for name in os.listdir(tmp_path))


def test_ensure_loaded_is_idempotent_under_asyncio(tmp_path, monkeypatch):
    """ensure_loaded loads once even when awaited twice inside asyncio.run."""
    calls = []
    orig = StateStore.load_async

    async def spy(self):
        calls.append(1)
        return await orig(self)

    monkeypatch.setattr(StateStore, "load_async", spy)
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed())

    async def main():
        await m.ensure_loaded()
        await m.ensure_loaded()

    asyncio.run(main())
    assert len(calls) == 1


# ----------------------------------------------------------------- M8 quality sampling
def test_quality_block_samples_horizon_and_look_ahead(tmp_path):
    """One sample per block after the 30-minute horizon; no completion from a price inside the horizon."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    t0 = et_epoch(9, 45)
    m.finish_snapshot(t0, {"SPY": snap(premium=0.5, last_price=100.0, age=5, ret=0.002)})
    assert "SPY|premium" in m._pending
    # look-ahead canary: a price at +900 s must not close the block
    m.finish_snapshot(t0 + 900, {"SPY": snap(premium=0.5, last_price=105.0, age=5, ret=0.002)})
    assert m._quality.get("SPY|premium") is None or m._quality["SPY|premium"].n == 0
    assert "SPY|premium" in m._pending
    m.finish_snapshot(t0 + 1800, {"SPY": snap(premium=0.5, last_price=101.0, age=5, ret=0.002)})
    q = m._quality["SPY|premium"]
    assert q.n == pytest.approx(1.0)
    assert q.sx == pytest.approx(0.5)
    assert q.sz == pytest.approx(0.002)
    assert q.sy == pytest.approx(0.01)
    # repeating the horizon call adds nothing new for block 0 (block 1 may open, not count)
    m.finish_snapshot(t0 + 1800, {"SPY": snap(premium=0.5, last_price=101.0, age=5, ret=0.002)})
    assert m._quality["SPY|premium"].n == pytest.approx(1.0)


def test_quality_stale_price_and_drop_and_gates(tmp_path):
    """Stale price, out-of-session, missing price: nothing opens; an unfinished pending drops after 3600 s."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    t0 = et_epoch(9, 45)
    m.finish_snapshot(t0, {"SPY": snap(age=400)})
    assert "SPY|premium" not in m._pending

    m.finish_snapshot(t0, {"SPY": snap(last_price=None)})
    assert "SPY|premium" not in m._pending

    m2 = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path / "b"),
                               session_fn=fixed(in_session=False), sync_io=True)
    m2.finish_snapshot(t0, {"SPY": snap()})
    assert m2._pending == {}

    m.finish_snapshot(t0, {"SPY": snap()})
    assert "SPY|premium" in m._pending
    m.finish_snapshot(t0 + 3600, {"SPY": snap(age=400)})
    assert "SPY|premium" not in m._pending
    assert m._quality.get("SPY|premium") is None or m._quality["SPY|premium"].n == 0


# ----------------------------------------------------------------- M9 session roll
def test_session_roll_decays_once_clears_pending_and_forces_save(tmp_path, monkeypatch):
    """First snapshot of a new session day decays quality once, clears pending and forces a save."""
    saves = _save_spy(monkeypatch)
    day = {"d": DAY}
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path),
                              session_fn=lambda now: (day["d"], True), sync_io=True)
    q = QualityStats()
    q.roll_session(DAY, m._params)
    q.add_block(DAY, 0, 0.1, 0.2, 0.3)
    m._quality["SPY|premium"] = q
    m.finish_snapshot(et_epoch(10, 0), {"SPY": snap()})
    n_before = q.n
    m._pending["SPY|premium"] = {"idx": 0, "t0": 1.0, "x": 0.1, "z": 0.2, "p0": 100.0}
    saves_before = len(saves)
    day["d"] = DAY2
    m.finish_snapshot(et_epoch(10, 0, day=DAY2), {"SPY": snap()})
    assert q.n == pytest.approx(n_before * 0.5 ** (1.0 / 20.0))
    # the old pending (t0=1.0) is gone; the same call may open a fresh block-0 pending for the new day
    assert all(v["t0"] != 1.0 for v in m._pending.values())
    assert len(saves) == saves_before + 1


def test_premium_dead_without_gex_is_unavailable_in_shadow_by_min_pillars(tmp_path):
    """Dead premium with no GEX leaves two counted pillars; the composite is None (min_pillars, design 3)."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    last = kill_premium(m, "SPY", et_epoch(10, 0))
    u = snap(premium=0.0)
    m.finish_snapshot(last + 200, {"SPY": u})
    rec = last_record(tmp_path)
    assert rec["static"] is not None
    assert rec["dyn"] is None


# ----------------------------------------------------------------- M10 fail-static
def test_compute_failure_is_fail_static_and_other_underlying_proceeds(tmp_path, monkeypatch, caplog):
    """A raising weight computation leaves that underlying on static weights; the next one still runs."""
    real = dwmod.compute_weights
    state = {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(dwmod, "compute_weights", flaky)
    m = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=fixed(), sync_io=True)
    spy, qqq = snap("SPY"), snap("QQQ")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        m.finish_snapshot(et_epoch(10, 0), {"SPY": spy, "QQQ": qqq})
    assert spy.pillar_weights is None
    assert isinstance(qqq.pillar_weights, dict)
    assert len([r for r in caplog.records if "SPY failed" in r.getMessage()]) == 1
    assert [r["u"] for r in ShadowLog(str(tmp_path)).read_tail(5)] == ["QQQ"]


# ----------------------------------------------------------------- M11 event loop
def test_writes_run_on_executor_not_loop_thread(tmp_path, monkeypatch):
    """Under a running loop, shadow and state writes run off the loop thread and are drained."""
    seen = []
    orig_append = ShadowLog.append_cycle
    orig_save = StateStore.save

    def spy_append(self, records):
        seen.append(("append", threading.get_ident()))
        return orig_append(self, records)

    def spy_save(self, state):
        seen.append(("save", threading.get_ident()))
        return orig_save(self, state)

    monkeypatch.setattr(ShadowLog, "append_cycle", spy_append)
    monkeypatch.setattr(StateStore, "save", spy_save)
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=False)

    async def main():
        loop_tid = threading.get_ident()
        m.finish_snapshot(et_epoch(10, 0), {"SPY": snap()})
        m.drain(timeout=5.0)
        return loop_tid

    loop_tid = asyncio.run(main())
    kinds = {k for k, _ in seen}
    assert kinds == {"append", "save"}
    assert all(tid != loop_tid for _, tid in seen)


# ----------------------------------------------------------------- M12 default session
def test_default_session_calendar_rules():
    """Weekends, holidays, early closes and the 09:30 to 16:00 boundaries of the default session."""
    sat = dt.date(2026, 10, 10)
    assert sat.weekday() == 5
    assert _default_session(et_epoch(10, 0, day=sat.isoformat()))[1] is False

    fri = dt.date(2026, 10, 9)
    assert fri.weekday() == 4
    assert _default_session(et_epoch(9, 29, day=fri.isoformat()))[1] is False
    assert _default_session(et_epoch(9, 30, day=fri.isoformat()))[1] is True
    assert _default_session(et_epoch(15, 59, day=fri.isoformat()))[1] is True
    assert _default_session(et_epoch(16, 0, day=fri.isoformat()))[1] is False

    hol = sorted(d for d in MARKET_HOLIDAYS if d.weekday() < 5)[0]
    assert _default_session(et_epoch(10, 0, day=hol.isoformat()))[1] is False

    early = sorted(d for d in MARKET_EARLY_CLOSE if d.weekday() < 5 and d not in MARKET_HOLIDAYS)[0]
    assert _default_session(et_epoch(12, 59, day=early.isoformat()))[1] is True
    assert _default_session(et_epoch(13, 1, day=early.isoformat()))[1] is False


# ----------------------------------------------------------------- M13 observe_uw filters
def test_observe_uw_ignores_non_liveness_bad_values_and_out_of_session(tmp_path):
    """Only premium/darkpool are tracked; NaN, inf, bool and out-of-session values create no key."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    base = et_epoch(10, 0)
    m.observe_uw("SPY", "nofa", 0.0, base)
    m.observe_uw("SPY", "kronos", 0.0, base)
    for bad in (float("nan"), float("inf"), True, "x"):
        m.observe_uw("SPY", "premium", bad, base)
    assert m._liveness.keys() == []
    m_out = DynamicWeightsManager(make_cfg(tmp_path, "shadow", ), str(tmp_path / "o"),
                                  session_fn=fixed(in_session=False), sync_io=True)
    m_out.observe_uw("SPY", "premium", 0.0, base)
    assert m_out._liveness.keys() == []
    m.observe_uw("SPY", "premium", 0.3, base)
    assert m._liveness.keys() == ["SPY|premium|uw"]


# ----------------------------------------------------------------- M14 mutation-probe helper
def test_warning_dedupe_key_is_per_manager_and_rate_limited(tmp_path, caplog):
    """The same warning key logs once per hour per manager instance."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(3):
            m._warn("k", "same message")
    assert len([r for r in caplog.records if r.getMessage() == "same message"]) == 1
    assert math.isfinite(dwmod._WARN_INTERVAL_S)


# ----------------------------------------------------------------- M15 never raise
def test_public_methods_never_raise_when_session_fn_raises(tmp_path):
    """A failing session function is swallowed by every public method (fail-static, never raises)."""

    def broken(now):
        raise RuntimeError("calendar exploded")

    m = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=broken, sync_io=True)
    m.observe_uw("SPY", "premium", 0.0, et_epoch(10, 0))
    u = snap()
    m.finish_snapshot(et_epoch(10, 0), {"SPY": u})
    m.load_sync(et_epoch(10, 0))
    assert u.pillar_weights is None
    assert m.is_masked("SPY", "premium", et_epoch(10, 0)) is False


# ----------------------------------------------------------------- M16 in-block sample dedupe
def test_block_is_sampled_once_even_if_its_pending_is_lost(tmp_path):
    """A block that was opened once is not re-opened later in the same block (sampled-key dedupe)."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    t0 = et_epoch(9, 45)
    m.finish_snapshot(t0, {"SPY": snap()})
    assert "SPY|premium" in m._pending
    m._pending.clear()
    m.finish_snapshot(t0 + 300, {"SPY": snap()})
    assert "SPY|premium" not in m._pending


# ------------------------------------------- reviewer additions (design-conformance gaps)

def test_masked_dead_pillar_absent_from_the_snapshot_still_conserves_the_weight_total(tmp_path):
    """Live: a dead premium that the data layer already REMOVED still counts in the target, so living weights sum to 1.0."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=fixed(), sync_io=True)
    last = kill_premium(m, "SPY", et_epoch(10, 0))
    assert m.is_masked("SPY", "premium", last) is True
    u = snap(premium=None)
    m.finish_snapshot(last + 60, {"SPY": u})
    w = u.pillar_weights
    assert w["premium"] == 0.0
    assert w["nofa"] == pytest.approx(0.40 / 0.75, abs=1e-9)
    assert w["darkpool"] == pytest.approx(0.20 / 0.75, abs=1e-9)
    assert w["kronos"] == pytest.approx(0.15 / 0.75, abs=1e-9)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    rec = last_record(tmp_path)
    assert rec["dead"] == ["premium"] and "premium" not in rec["avail"]


def test_dead_uw_pillar_replaced_by_selfflow_is_alive_and_not_listed_dead(tmp_path):
    """A dead UW premium replaced by a self-flow reading keeps its prior weight and is not reported dead."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=fixed(), sync_io=True)
    last = kill_premium(m, "SPY", et_epoch(10, 0))
    u = snap(premium=0.2, premium_source="selfflow:SPY")
    m.finish_snapshot(last + 60, {"SPY": u})
    assert u.pillar_weights["premium"] == pytest.approx(0.25, abs=1e-9)
    assert sum(u.pillar_weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert last_record(tmp_path)["dead"] == []


def test_plain_missing_pillar_that_is_not_dead_is_not_redistributed(tmp_path):
    """A pillar that is merely missing (never flagged dead) keeps today's semantics: the total is the living priors."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=fixed(), sync_io=True)
    u = snap(premium=None)
    m.finish_snapshot(et_epoch(10, 0), {"SPY": u})
    w = u.pillar_weights
    assert "premium" not in w
    assert w["nofa"] == pytest.approx(0.40, abs=1e-9) and sum(w.values()) == pytest.approx(0.75, abs=1e-9)


def test_quality_sample_horizon_runs_from_the_actual_sample_time_not_the_block_start(tmp_path):
    """A first snapshot at 09:58 (restart mid-block) is closed at 10:28, a full 30 minutes later, not at 10:15."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    t_open = et_epoch(9, 58)
    m.finish_snapshot(t_open, {"SPY": snap(last_price=100.0, ret=0.001)})
    assert m._pending["SPY|nofa"]["t0"] == pytest.approx(t_open)
    m.finish_snapshot(et_epoch(10, 15), {"SPY": snap(last_price=101.0, ret=0.001)})
    assert "SPY|nofa" in m._pending and "SPY|nofa" not in m._quality
    m.finish_snapshot(et_epoch(10, 28), {"SPY": snap(last_price=101.0, ret=0.001)})
    assert "SPY|nofa" not in m._pending
    assert m._quality["SPY|nofa"].n_eff() == 1.0


def test_session_roll_decays_only_on_in_session_days_never_on_a_weekend(tmp_path):
    """Quality stats decay once on the first IN-SESSION snapshot of a new trading day, not on a weekend calendar flip."""
    state = {"day": "2026-10-09", "in": True}  # Friday
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path),
                              session_fn=lambda now: (state["day"], state["in"]), sync_io=True)
    q = QualityStats()
    q.roll_session("2026-10-09", m._params)
    assert q.add_block("2026-10-09", 0, 0.1, 0.0, 0.01)
    m._quality["SPY|nofa"] = q
    m.finish_snapshot(et_epoch(10, 0, day="2026-10-09"), {"SPY": snap()})
    assert q.n_eff() == pytest.approx(1.0)
    state.update(day="2026-10-10", **{"in": False})  # Saturday, market closed
    m.finish_snapshot(et_epoch(10, 0, day="2026-10-10"), {"SPY": snap()})
    assert q.n_eff() == pytest.approx(1.0)
    state.update(day="2026-10-12", **{"in": True})  # Monday
    m.finish_snapshot(et_epoch(10, 0, day="2026-10-12"), {"SPY": snap()})
    assert q.n_eff() == pytest.approx(0.5 ** (1.0 / 20.0))
    m.finish_snapshot(et_epoch(10, 1, day="2026-10-12"), {"SPY": snap()})
    assert q.n_eff() == pytest.approx(0.5 ** (1.0 / 20.0))


def test_a_pending_block_is_dropped_not_completed_when_the_gap_exceeds_the_give_up_window(tmp_path):
    """A pending block seen again 2 hours later with a fresh price is DROPPED (not closed with a stale-horizon return)."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    m.finish_snapshot(et_epoch(10, 0), {"SPY": snap(last_price=100.0)})
    assert "SPY|nofa" in m._pending
    m.finish_snapshot(et_epoch(12, 0), {"SPY": snap(last_price=105.0)})
    assert "SPY|nofa" not in m._quality
    assert "SPY|nofa" not in m._pending or m._pending["SPY|nofa"]["p0"] == 105.0


def test_pending_block_does_not_complete_out_of_session(tmp_path):
    """Out of session nothing opens or closes: a Friday-afternoon pending is not closed by a Saturday snapshot."""
    state = {"day": "2026-10-09", "in": True}
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path),
                              session_fn=lambda now: (state["day"], state["in"]), sync_io=True)
    m.finish_snapshot(et_epoch(14, 50, day="2026-10-09"), {"SPY": snap(last_price=100.0)})
    assert "SPY|nofa" in m._pending
    state.update(day="2026-10-10", **{"in": False})
    m.finish_snapshot(et_epoch(14, 50, day="2026-10-10"), {"SPY": snap(last_price=140.0)})
    assert "SPY|nofa" not in m._quality


def _saved_state_with_dead_premium(tmp_path):
    """Run a manager until premium is dead and the state file is written; return the file's written_at."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=fixed(), sync_io=True)
    last = kill_premium(m, "SPY", et_epoch(10, 0))
    m.finish_snapshot(last + 61, {"SPY": snap(premium=None)})
    return json.load(open(os.path.join(str(tmp_path), "pillar_weights.json")))["written_at"]


@pytest.mark.parametrize("age,ring_len", [(800.0, 10), (1200.0, 0)])
def test_restore_reuses_the_ring_only_within_fifteen_minutes_but_always_keeps_the_dead_flag(tmp_path, age, ring_len):
    """Same-day state younger than 900 s restores the ring; older state keeps the dead flag but an EMPTY ring."""
    written = _saved_state_with_dead_premium(tmp_path)
    m2 = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=fixed(), sync_io=True)
    m2.load_sync(written + age)
    assert m2._liveness.is_dead("SPY|premium|uw") is True
    assert len(m2._liveness.to_dict()["keys"]["SPY|premium|uw"]["ring"]) == ring_len


def test_late_state_load_never_overwrites_observations_already_made(tmp_path, caplog):
    """If observations arrived before the load finished, the persisted state is skipped (with a WARNING), not applied."""
    written = _saved_state_with_dead_premium(tmp_path)
    m2 = DynamicWeightsManager(make_cfg(tmp_path, "live"), str(tmp_path), session_fn=fixed(), sync_io=True)
    m2.observe_uw("SPY", "premium", 0.3, et_epoch(10, 0))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        m2.load_sync(written + 10)
    assert m2._liveness.is_dead("SPY|premium|uw") is False
    assert any("loaded after first observation" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("h,mnt", [(9, 40), (15, 20), (15, 59)])
def test_no_quality_block_opens_before_0945_or_after_the_last_block(tmp_path, h, mnt):
    """Blocks exist only from 09:45 ET through block index 10 (up to 15:15); outside that no sample opens."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    m.finish_snapshot(et_epoch(h, mnt), {"SPY": snap()})
    assert m._pending == {}


def test_a_quality_block_does_open_at_0945_and_in_the_last_block(tmp_path):
    """Control: 09:45 (block 0) and 14:50 (block 10) do open a sample."""
    for hh, mm, idx in ((9, 45, 0), (14, 50, 10)):
        m = DynamicWeightsManager(make_cfg(tmp_path / f"c{idx}", "shadow"), str(tmp_path / f"c{idx}"),
                                  session_fn=fixed(), sync_io=True)
        os.makedirs(tmp_path / f"c{idx}", exist_ok=True)
        m.finish_snapshot(et_epoch(hh, mm), {"SPY": snap()})
        assert m._pending["SPY|nofa"]["idx"] == idx


# ------------------------------------------- reviewer-panel hardening (2026-10-10)

def test_a_stalled_disk_cannot_exhaust_the_executor_writes_are_capped_and_a_dropped_save_is_retried(tmp_path, caplog, monkeypatch):
    """With writes blocked, at most _MAX_INFLIGHT_IO are ever in flight (one deduped WARNING), and a DROPPED state save stays dirty."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed())
    gate = threading.Event()

    def slow_append(records):
        gate.wait(10)
        return len(records)

    monkeypatch.setattr(m._shadow, "append_cycle", slow_append)
    monkeypatch.setattr(m._store, "save", lambda doc: (gate.wait(10), True)[1])

    async def go():
        base = et_epoch(10, 0)
        for i in range(8):
            m._dirty = True                                         # state changed this cycle, so a save is due
            m.finish_snapshot(base + i * 61, {"SPY": snap()})      # 61 s apart: past the save interval every cycle
        assert m._io_pending == dwmod._MAX_INFLIGHT_IO
        assert m._dirty is True                                     # the dropped save was NOT marked clean
        gate.set()
        m.drain(5.0)
        assert m._io_pending == 0
        m.finish_snapshot(base + 9 * 61, {"SPY": snap()})           # capacity is back: the save goes through
        m.drain(5.0)
        assert m._dirty is False

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        asyncio.run(go())
    assert len([r for r in caplog.records if "already in flight" in r.getMessage()]) == 1


def test_restore_ignores_non_numeric_last_weights_and_clamps_a_future_flip_timestamp(tmp_path, caplog):
    """A hand-edited/foreign state file cannot crash a cycle (non-numeric last_weights) or freeze the dwell (future flip ts)."""
    store = StateStore(str(tmp_path))
    far_future = time.time() + 1e9
    state = {
        "session_day": DAY,
        "liveness": {"schema_version": 1, "keys": {"SPY|premium|uw": {
            "ring": [], "dead": True, "last_flip_ts": far_future, "consec_alive": 0, "ramp_stage": 3,
            "last_session_day": DAY, "transitions": []}}},
        "quality": {}, "pending": {},
        "last_weights": {"SPY": {"nofa": "x", "premium": None, "darkpool": 0.2, "kronos": float("inf") if False else 0.1}},
    }
    assert store.save(state)
    written = json.load(open(os.path.join(str(tmp_path), "pillar_weights.json")))["written_at"]
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    m.load_sync(written + 10)
    assert m._last_weights["SPY"] == {"darkpool": 0.2, "kronos": 0.1}
    key = m._liveness.to_dict()["keys"]["SPY|premium|uw"]
    assert key["dead"] is True and key["last_flip_ts"] is None
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        m.finish_snapshot(et_epoch(10, 0), {"SPY": snap()})
    assert not [r for r in caplog.records if "failed" in r.getMessage()]


# ------------------------------------------- bounded first load, applied multiplier (2026-10-10 review panel)

def test_a_hung_state_load_never_stalls_the_first_snapshot_and_is_picked_up_after(tmp_path, caplog, monkeypatch):
    """ensure_loaded returns after _LOAD_TIMEOUT_S while the disk read hangs; the manager keeps working (empty state) and a
    later call completes once the read returns, without starting a second load."""
    monkeypatch.setattr(dwmod, "_LOAD_TIMEOUT_S", 0.05)
    gate = threading.Event()
    calls = []

    def hung_load(self):
        calls.append(1)
        gate.wait(10)
        return None

    monkeypatch.setattr(StateStore, "load", hung_load)
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed())

    async def main():
        t0 = time.monotonic()
        await m.ensure_loaded()
        assert time.monotonic() - t0 < 2.0 and m._loaded is False
        m.finish_snapshot(et_epoch(10, 0), {"SPY": snap()})         # the cycle proceeds with empty state
        await m.ensure_loaded()                                       # still hung: same task, no second load
        assert m._loaded is False and len(calls) == 1
        gate.set()
        await m.ensure_loaded()
        assert m._loaded is True and len(calls) == 1

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        asyncio.run(main())
        m.drain(5.0)
    assert len([r for r in caplog.records if "still running" in r.getMessage()]) == 1


def test_a_load_that_finishes_before_any_observation_is_restored_even_if_the_first_wait_timed_out(tmp_path, monkeypatch):
    """Late but before the first observation: the persisted dead flag is restored."""
    store = StateStore(str(tmp_path))
    m0 = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    kill_premium(m0, "SPY", et_epoch(10, 0))
    assert m0._liveness.is_dead("SPY|premium|uw")
    assert store.save(m0._state_doc(DAY)) is True
    monkeypatch.setattr(dwmod, "_LOAD_TIMEOUT_S", 0.0)
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed())

    async def main():
        await m.ensure_loaded()          # may or may not have finished within a zero wait
        for _ in range(200):
            if m._loaded:
                break
            await asyncio.sleep(0.01)
            await m.ensure_loaded()

    asyncio.run(main())
    assert m._loaded is True
    assert m._liveness.is_dead("SPY|premium|uw")


def test_shadow_record_carries_the_applied_multiplier_next_to_the_diagnostic_one(tmp_path):
    """'ma' is 1.0 per pillar while quality weighting is off, even when the diagnostic 'm' has moved off 1.0."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    m.finish_snapshot(et_epoch(10, 0), {"SPY": snap()})
    rec = last_record(tmp_path)
    assert set(rec["ma"]) == set(rec["m"]) == set(rec["avail"])
    assert all(v == 1.0 for v in rec["ma"].values())


def test_a_cancelled_load_task_is_treated_as_empty_state_not_raised(tmp_path):
    """If the background load task was cancelled (loop shutdown), ensure_loaded finishes with empty state instead of raising."""
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed())

    async def main():
        loop = asyncio.get_running_loop()
        task = loop.create_task(asyncio.sleep(30))
        m._load_task, m._load_loop = task, loop
        task.cancel()
        await asyncio.sleep(0)
        await m.ensure_loaded()
        assert m._loaded is True and m._load_task is None

    asyncio.run(main())


def test_applied_multiplier_stays_one_while_the_logged_diagnostic_moves(tmp_path):
    """With real samples the diagnostic 'm' leaves 1.0, but 'ma' (what the weights use) stays 1.0 until quality is applied."""
    from tests.dpof.test_pillar_weights import _planted
    m = DynamicWeightsManager(make_cfg(tmp_path, "shadow"), str(tmp_path), session_fn=fixed(), sync_io=True)
    m._quality["SPY|premium"] = _planted(5, 400, 0.3)
    m.finish_snapshot(et_epoch(10, 0), {"SPY": snap()})
    rec = last_record(tmp_path)
    assert rec["m"]["premium"] > 1.0
    assert rec["ma"]["premium"] == 1.0
    assert rec["static"] == pytest.approx(rec["dyn"], abs=1e-12)
