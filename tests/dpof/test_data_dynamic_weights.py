"""Dynamic pillar weights wired into the DPOF data feed (agent/dpof/data.py hooks).

Plugin-free (asyncio.run). Only agent.integrations.uw_flow_intel._uw_get and the
other integration fakes from tests/dpof/test_data.py are faked; every data.py
path runs for real. All files live under tmp_path; no live state is read.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

import agent.dpof.data as data
import agent.flowcollector.state as flowcollector_state
from agent.dpof.config import DPOFConfig
from agent.dpof.dynamic_weights import DynamicWeightsManager
from agent.dpof.models import PILLAR_NAMES, PillarReading
from agent.dpof.signals import compute_composite
from tests.dpof.test_data import FakeBroker, Fakes

ET = ZoneInfo("America/New_York")
# Monday 2026-10-05 11:00 EDT: inside regular trading hours.
BASE = datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc).timestamp()
DAY = "2026-10-05"
STEP = 120.0  # > uw_refresh_seconds (100): every cycle refreshes UW
_AMPLE = {"option_trades_5m": 500, "dark_classified_5m": 200}


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    return Fakes(monkeypatch, tmp_path)


@pytest.fixture(autouse=True)
def _fast_timeouts(monkeypatch):
    monkeypatch.setattr(data, "_UW_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_BARS_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_VIX_CALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_FILE_GATE_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(data, "_SNAPSHOT_BUDGET_SECONDS", 5.0)


@pytest.fixture(autouse=True)
def _isolate_selfflow_file(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(tmp_path / "no_selfflow_state.json"))


def _run(coro):
    return asyncio.run(coro)


def _make_feed(tmp_path, name, mode, *, selfflow=False, kronos="off", min_pillars=2, empty_unavail=False):
    data_dir = tmp_path / name
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(
        DPOFConfig(), data_dir=str(data_dir), min_pillars=min_pillars, uw_refresh_seconds=100,
        bars_refresh_seconds=50, underlyings=("SPY", "QQQ"), dynamic_weights=mode, dw_ring_size=10,
        selfflow_fallback=selfflow, kronos_pillar=kronos, empty_pillar_unavailable=empty_unavail,
    )
    return data.DPOFDataFeed(cfg, FakeBroker()), cfg


def _in_session(_ts):
    return (DAY, True)


def _digest(snap):
    return {
        "underlyings": {
            sym: {n: (r.value, r.available, r.source, r.error, r.age_seconds) for n, r in sorted(u.pillars.items())}
            for sym, u in sorted(snap.underlyings.items())
        },
        "data_healthy": snap.data_healthy,
        "health_issues": list(snap.health_issues),
        "risk_flags": list(snap.risk_flags),
    }


async def _cycles(feed, n, start=BASE, drain=False):
    """n snapshots STEP seconds apart. drain=True lets each cycle's writes finish first (production cycles are 30 s
    apart; back-to-back cycles would trip the manager's in-flight write cap and drop shadow records by design)."""
    snaps = []
    for i in range(n):
        snaps.append(await feed.snapshot(start + i * STEP))
        if drain and feed._dw is not None:
            feed._dw.drain()
    return snaps


def _seed_dead(feed, sym, pillar, now, n=10, value=0.0):
    """value=0.0 seeds a dead ring (|v|<eps); any other value seeds an alive ring alternating in sign (identical values read as stuck)."""
    for i in range(n):
        v = value if value == 0.0 else (value if i % 2 == 0 else -value)
        feed._dw.observe_uw(sym, pillar, v, now - (n - i) * 60.0)


# D1 ─ OFF: no files, no masking, snapshot matches the off reference.
def test_off_mode_writes_nothing_masks_nothing_and_matches_reference(tmp_path, fakes):
    """With dynamic_weights off the feed writes no dynamic-weights files and never masks a pillar."""
    feed, _ = _make_feed(tmp_path, "off", "off")
    snaps = _run(_cycles(feed, 3))
    feed._dw.drain()
    names = os.listdir(tmp_path / "off")
    assert not [n for n in names if "pillar_weights" in n or "dynamic" in n]
    assert feed._dw_masked("SPY", "premium", BASE) is False
    assert all(u.pillar_weights is None for s in snaps for u in s.underlyings.values())


# D2 ─ A/B equivalence: off vs shadow give identical decision inputs.
def test_shadow_mode_leaves_every_decision_input_identical_to_off(tmp_path, fakes):
    """Twenty cycles in shadow produce the same snapshot as off, with shadow records written beside it."""
    off_feed, _ = _make_feed(tmp_path, "off", "off")
    sh_feed, _ = _make_feed(tmp_path, "shadow", "shadow")
    off_snaps = _run(_cycles(off_feed, 20))
    sh_snaps = _run(_cycles(sh_feed, 20, drain=True))
    sh_feed._dw.drain()
    assert [_digest(s) for s in sh_snaps] == [_digest(s) for s in off_snaps]
    assert all(u.pillar_weights is None for s in sh_snaps for u in s.underlyings.values())
    lines = (tmp_path / "shadow" / "pillar_weights_shadow.jsonl").read_text().splitlines()
    assert len(lines) == 40  # 2 underlyings x 20 cycles
    expected_ts = {datetime.fromtimestamp(BASE + i * STEP, ET).isoformat() for i in range(20)}
    assert {json.loads(line)["ts"] for line in lines} == expected_ts


# D3 ─ HOOKS: observations land where and only where the spec says.
def test_refresh_observes_premium_and_darkpool_once_each(tmp_path, fakes):
    """A single premium and darkpool refresh each record exactly one UW liveness observation."""
    feed, _ = _make_feed(tmp_path, "hooks", "shadow")

    async def go():
        await feed._refresh_premium(BASE)
        await feed._refresh_darkpool(BASE)

    _run(go())
    keys = feed._dw._liveness.to_dict()["keys"]
    assert len(keys["SPY|premium|uw"]["ring"]) == 1
    assert len(keys["SPY|darkpool|uw"]["ring"]) == 1


def test_empty_pillar_unavailable_observes_nothing(tmp_path, fakes):
    """With empty_pillar_unavailable and empty UW data, the empty path writes no cache and observes nothing."""
    feed, _ = _make_feed(tmp_path, "empty", "shadow", empty_unavail=True)
    _run(feed._refresh_premium(BASE))
    _run(feed._refresh_darkpool(BASE))
    assert feed._dw._liveness.to_dict()["keys"] == {}


def test_selfflow_substituted_pillar_creates_no_selfflow_key(tmp_path, fakes, monkeypatch):
    """When UW premium fails and self-flow fills it, no liveness key with a selfflow source is created."""
    feed, _ = _make_feed(tmp_path, "sf", "shadow", selfflow=True)
    fakes.uw_responses["/stock/SPY/net-prem-ticks"] = {}  # UW failure shape

    def _sf(sym, max_age=300):
        if sym != "SPY":
            return None
        return {"as_of": BASE, "premium_ratio": 0.4, "quality_notes": _AMPLE}

    monkeypatch.setattr(flowcollector_state, "get_selfflow", _sf)
    _run(_cycles(feed, 2))
    feed._dw.drain()
    keys = feed._dw._liveness.to_dict()["keys"]
    assert "SPY|premium|uw" not in keys
    assert not [k for k in keys if "selfflow" in k]


# D4 ─ LIVE MASK + SELF-FLOW SUBSTITUTION.
def _manager(cfg, data_dir, mode_session=_in_session):
    return DynamicWeightsManager(cfg, str(data_dir), session_fn=mode_session)


def _premium_cached(feed, now):
    """Every SPY pillar, NOPE and max-pain fresh, so the mask is the only reason _needs_selfflow can be True."""
    for name in PILLAR_NAMES:
        feed._pillar_cache[("SPY", name)] = PillarReading(name=name, value=0.3, available=True, source="uw:x")
        feed._pillar_cache_time[("SPY", name)] = now
    feed._nope_cache_time["SPY"] = now
    feed._max_pain_cache_time["SPY"] = now


def test_live_mask_then_selfflow_substitution(tmp_path, fakes, monkeypatch):
    """In live a dead UW premium is masked despite a fresh cache, forces self-flow, and is replaced by it."""
    feed, cfg = _make_feed(tmp_path, "d4live", "live", selfflow=True)
    feed._dw = _manager(cfg, cfg.data_dir)
    _premium_cached(feed, BASE)
    _seed_dead(feed, "SPY", "premium", BASE)
    assert feed._dw_masked("SPY", "premium", BASE) is True
    u = feed._build_underlying_snapshot("SPY", BASE)
    assert u.pillars["premium"].available is False
    assert u.pillars["premium"].error.startswith("dead")
    assert feed._needs_selfflow("SPY", BASE) is True

    def _sf(sym, max_age=300):
        return {"as_of": BASE, "premium_ratio": 0.4, "quality_notes": _AMPLE} if sym == "SPY" else None

    monkeypatch.setattr(flowcollector_state, "get_selfflow", _sf)
    _run(feed._maybe_refresh_selfflow(BASE))
    u2 = feed._build_underlying_snapshot("SPY", BASE)
    assert u2.pillars["premium"].source == "selfflow:SPY"
    assert u2.pillars["premium"].available is True
    assert u2.flow_selfflow is True


def test_shadow_mode_masks_nothing_on_the_same_dead_state(tmp_path, fakes):
    """The same dead state in shadow masks nothing and leaves the cached premium in place."""
    feed, cfg = _make_feed(tmp_path, "d4shadow", "shadow", selfflow=True)
    feed._dw = _manager(cfg, cfg.data_dir)
    _premium_cached(feed, BASE)
    _seed_dead(feed, "SPY", "premium", BASE)
    assert feed._dw_masked("SPY", "premium", BASE) is False
    assert feed._needs_selfflow("SPY", BASE) is False
    u = feed._build_underlying_snapshot("SPY", BASE)
    assert u.pillars["premium"].available is True
    assert u.pillars["premium"].source == "uw:x"


# D5 ─ LIVE WEIGHTS END-TO-END.
def test_live_weights_end_to_end_with_dead_premium(tmp_path, fakes, monkeypatch):
    """In live with premium dead and no self-flow, the snapshot carries the redistributed weights used by compute_composite."""
    feed, cfg = _make_feed(tmp_path, "d5", "live", kronos="live")
    feed._dw = _manager(cfg, cfg.data_dir)
    _seed_dead(feed, "SPY", "premium", BASE)
    _seed_dead(feed, "SPY", "darkpool", BASE, value=0.5)  # alive
    monkeypatch.setattr(
        data, "_kronos_pillar_read_sync",
        lambda symbols, max_age, horizon, now=None: {
            s: PillarReading(name="kronos", value=0.5, available=True, source="kronos") for s in symbols
        },
    )
    snap = _run(feed.snapshot(BASE))
    feed._dw.drain()
    u = snap.underlyings["SPY"]
    w = u.pillar_weights
    assert w is not None
    assert w.get("premium", 0.0) == pytest.approx(0.0)
    assert w["nofa"] == pytest.approx(0.40 / 0.75)
    assert w["darkpool"] == pytest.approx(0.20 / 0.75)
    assert w["kronos"] == pytest.approx(0.15 / 0.75)
    assert sum(w.values()) == pytest.approx(1.0)
    _, _, reasons = compute_composite(u, cfg)
    assert "weights:dynamic" in reasons


# D6 ─ MIN_PILLARS CONSEQUENCE.
def _both_dead(feed, cfg):
    feed._dw = _manager(cfg, cfg.data_dir)
    _seed_dead(feed, "SPY", "premium", BASE)
    _seed_dead(feed, "SPY", "darkpool", BASE)


def test_both_pillars_dead_makes_primary_unavailable_in_live_only(tmp_path, fakes):
    """With premium and darkpool both dead and no self-flow, live reports primary_composite_unavailable; shadow does not."""
    live, lcfg = _make_feed(tmp_path, "d6live", "live", min_pillars=3)
    _both_dead(live, lcfg)
    lsnap = _run(live.snapshot(BASE))
    assert "primary_composite_unavailable" in lsnap.health_issues
    assert lsnap.data_healthy is False

    shadow, scfg = _make_feed(tmp_path, "d6shadow", "shadow", min_pillars=3)
    _both_dead(shadow, scfg)
    ssnap = _run(shadow.snapshot(BASE))
    assert "primary_composite_unavailable" not in ssnap.health_issues


# D7 ─ FAIL-STATIC.
def test_manager_failures_are_swallowed_and_match_off(tmp_path, fakes):
    """Each raising manager method leaves refresh and snapshot completing with the off-mode results."""
    off, _ = _make_feed(tmp_path, "d7off", "off")
    ref = [_digest(s) for s in _run(_cycles(off, 3))]

    async def _boom_async():
        raise RuntimeError("boom")

    def _boom(*a, **k):
        raise RuntimeError("boom")

    for attr, patch in (("observe_uw", _boom), ("is_masked", _boom), ("ensure_loaded", _boom_async), ("finish_snapshot", _boom)):
        feed, _ = _make_feed(tmp_path, f"d7{attr}", "live")
        setattr(feed._dw, attr, patch)
        got = [_digest(s) for s in _run(_cycles(feed, 3))]
        assert got == ref, attr


def test_cancelled_error_from_ensure_loaded_propagates(tmp_path, fakes):
    """A CancelledError raised while loading state is not swallowed by the feed hook."""
    feed, _ = _make_feed(tmp_path, "d7cancel", "live")

    async def _cancel():
        raise asyncio.CancelledError()

    feed._dw.ensure_loaded = _cancel
    with pytest.raises(asyncio.CancelledError):
        _run(feed.snapshot(BASE))


# D8 ─ PERSISTENCE across a restart.
def test_dead_flag_survives_a_new_feed_on_the_same_data_dir(tmp_path, fakes):
    """A dead premium saved by one feed is restored by a new feed on the same data_dir."""
    real_now = datetime.now(timezone.utc).timestamp()
    feed1, cfg = _make_feed(tmp_path, "d8", "live")
    feed1._dw = _manager(cfg, cfg.data_dir)
    _seed_dead(feed1, "SPY", "premium", real_now)
    _run(_cycles(feed1, 3, start=real_now))
    feed1._dw.drain()

    feed2, cfg2 = _make_feed(tmp_path, "d8", "live")
    feed2._dw = _manager(cfg2, cfg2.data_dir)

    orig_refresh = feed2._maybe_refresh_uw
    seen: dict = {}

    async def spy_refresh(now):
        # Recorded BEFORE the UW refresh of the snapshot: proves snapshot() itself loaded the state.
        seen["loaded"] = feed2._dw._loaded
        seen["dead"] = feed2._dw._liveness.is_dead("SPY|premium|uw")
        return await orig_refresh(now)

    feed2._maybe_refresh_uw = spy_refresh
    _run(feed2.snapshot(real_now + 3 * STEP))
    assert seen == {"loaded": True, "dead": True}


# D9 ─ MANAGER INIT FAILURE.
def test_manager_init_failure_leaves_feed_working(tmp_path, fakes, monkeypatch):
    """If the manager cannot be built the feed has _dw None and snapshots still complete with no weights."""

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no manager")

    monkeypatch.setattr(data, "DynamicWeightsManager", _Boom)
    feed, _ = _make_feed(tmp_path, "d9", "live")
    assert feed._dw is None
    snaps = _run(_cycles(feed, 2))
    assert all(u.pillar_weights is None for s in snaps for u in s.underlyings.values())


# ------------------------------------------- reviewer additions (wiring gaps)

def _premium_rows(nc, np_):
    return [{"net_call_premium": nc, "net_put_premium": np_, "tape_time": "2026-10-05T14:30:00.000000Z"} for _ in range(20)]


def test_healthy_varying_premium_through_the_refresh_path_is_never_flagged_dead(tmp_path, fakes):
    """Twelve real refreshes of a premium score that swings between +1 and -1 never turn the pillar dead or masked."""
    feed, cfg = _make_feed(tmp_path, "alive", "live")
    feed._dw = _manager(cfg, cfg.data_dir)
    for i in range(12):
        sign = 1 if i % 2 == 0 else -1
        fakes.uw_responses["/stock/SPY/net-prem-ticks"] = {"data": _premium_rows(f"{sign * 500}.00", f"{-sign * 100}.00")}
        _run(feed._refresh_premium(BASE + i * STEP))
    ring = feed._dw._liveness.to_dict()["keys"]["SPY|premium|uw"]["ring"]
    assert len(ring) == 10 and ring[0] == pytest.approx(1.0) and ring[1] == pytest.approx(-1.0)
    assert feed._dw._liveness.is_dead("SPY|premium|uw") is False
    assert feed._dw_masked("SPY", "premium", BASE + 12 * STEP) is False


def test_a_flat_premium_through_the_refresh_path_is_flagged_dead_and_masked_in_live(tmp_path, fakes):
    """Ten real refreshes whose rows carry no recognised field (score exactly 0.0) flag premium dead; live masks it."""
    feed, cfg = _make_feed(tmp_path, "flat", "live")
    feed._dw = _manager(cfg, cfg.data_dir)
    fakes.uw_responses["/stock/SPY/net-prem-ticks"] = {"data": [{"foo": 1, "tape_time": "2026-10-05T14:30:00.000000Z"}] * 20}
    for i in range(10):
        _run(feed._refresh_premium(BASE + i * STEP))
    ring = feed._dw._liveness.to_dict()["keys"]["SPY|premium|uw"]["ring"]
    assert ring == [0.0] * 10
    assert feed._dw._liveness.is_dead("SPY|premium|uw") is True
    assert feed._dw_masked("SPY", "premium", BASE + 10 * STEP) is True
    assert feed._dw_masked("SPY", "darkpool", BASE + 10 * STEP) is False


def test_the_observation_is_made_after_the_cache_write(tmp_path, fakes, monkeypatch):
    """When the observe hook fires, the premium/darkpool reading and its cache time are already stored."""
    feed, _ = _make_feed(tmp_path, "order", "shadow")
    seen = []
    orig = feed._dw_observe

    def spy(ticker, pillar, value, now):
        seen.append((pillar, (ticker, pillar) in feed._pillar_cache and feed._pillar_cache_time.get((ticker, pillar)) == now))
        return orig(ticker, pillar, value, now)

    monkeypatch.setattr(feed, "_dw_observe", spy)
    _run(feed._refresh_premium(BASE))
    _run(feed._refresh_darkpool(BASE))
    assert {p for p, _ok in seen} == {"premium", "darkpool"}
    assert all(ok for _p, ok in seen)


def test_healthy_varying_darkpool_through_the_refresh_path_is_never_flagged_dead(tmp_path, fakes):
    """Twelve refreshes of NBBO-ratio darkpool alternating between all-buy and all-sell windows never flag it dead."""
    feed, cfg = _make_feed(tmp_path, "dpalive", "live")
    feed.update_config(replace(cfg, darkpool_nbbo=True))
    feed._dw = _manager(cfg, cfg.data_dir)
    buy = {"price": "101.00", "size": 100, "nbbo_bid": "99.00", "nbbo_ask": "100.00", "canceled": False}
    sell = {"price": "98.00", "size": 100, "nbbo_bid": "99.00", "nbbo_ask": "100.00", "canceled": False}
    for i in range(12):
        fakes.uw_responses["/darkpool/SPY"] = {"data": [dict(buy if i % 2 == 0 else sell) for _ in range(25)]}
        _run(feed._refresh_darkpool(BASE + i * STEP))
    ring = feed._dw._liveness.to_dict()["keys"]["SPY|darkpool|uw"]["ring"]
    assert len(ring) == 10 and ring[0] == pytest.approx(1.0) and ring[1] == pytest.approx(-1.0)
    assert feed._dw._liveness.is_dead("SPY|darkpool|uw") is False
