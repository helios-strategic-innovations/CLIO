"""UW-7 (S211): the CLIO-built fallback snapshot must stay fresh between its 15-minute refreshes.

The fallback snapshot is derived from CLIO's own uw_flow_intel daily file (`created` = last job run,
every ~900 s). With a 600 s TTL it was stale for a third of every market-hours cycle, flooding the log
and making DPOF skip its vol-regime context. The Hydrinia-native snapshot.json contract is unchanged.

Every path is redirected to tmp_path: the real bridge dir lives under /home/agent/tradingbot (Hydrinia),
which tests must never read or write.
"""

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from agent.integrations import uw_signal_bridge as bridge_mod

NY = ZoneInfo("America/New_York")


@pytest.fixture()
def bridge(tmp_path, monkeypatch):
    snap_dir = tmp_path / "uw_signals"
    intel_dir = tmp_path / "intelligence"
    snap_dir.mkdir()
    intel_dir.mkdir()
    monkeypatch.setattr(bridge_mod, "_SNAPSHOT_DIR", str(snap_dir))
    monkeypatch.setattr(bridge_mod, "_SNAPSHOT_FILE", str(snap_dir / "snapshot.json"))
    monkeypatch.delenv("CLIO_UW_BRIDGE_ENABLED", raising=False)
    monkeypatch.delenv("MOLTBOT_UW_BRIDGE_ENABLED", raising=False)
    b = bridge_mod.UWSignalBridge()
    monkeypatch.setattr(b, "_is_market_hours", lambda: True)  # no off-hours TTL multiplier
    b.snap_dir, b.intel_dir = snap_dir, intel_dir
    return b


def _write_flow_file(bridge, age_s):
    created = (datetime.now(NY) - timedelta(seconds=age_s)).isoformat()
    (bridge.intel_dir / "uw_flow_20260929.json").write_text(json.dumps({
        "created": created,
        "market_tide": {"sentiment": "BULLISH"},
        "top_signals": [],
        "gex": {"SPY": {"net_gex": -1.0}},
    }))


def test_fallback_snapshot_carries_the_longer_ttl(bridge):
    _write_flow_file(bridge, age_s=10)
    snap = bridge._read_flow_intel_fallback()
    assert snap["source"] == "uw_flow_intel_fallback"
    assert snap["freshness_ttl_seconds"] == bridge_mod._FALLBACK_TTL_SECONDS == 1200


@pytest.mark.parametrize("age_s", [30, 606, 899, 1100])
def test_fallback_is_fresh_across_a_full_refresh_cycle(bridge, age_s):
    """606-899 s are exactly the stale-warning ages seen live (n=130); all must now read fresh."""
    _write_flow_file(bridge, age_s)
    assert bridge.is_fresh() is True


def test_fallback_still_goes_stale_when_the_producer_stalls(bridge):
    _write_flow_file(bridge, age_s=1300)  # > 1200 s: the job missed a cycle and then some
    assert bridge.is_fresh() is False


def test_hydrinia_native_snapshot_keeps_its_own_ttl_contract(bridge):
    """A real snapshot.json carries freshness_ttl_seconds=600; 700 s old must still be stale."""
    ts = (datetime.now(NY) - timedelta(seconds=700)).isoformat()
    (bridge.snap_dir / "snapshot.json").write_text(json.dumps({
        "timestamp": ts, "freshness_ttl_seconds": 600, "market_context": {},
    }))
    assert bridge.is_fresh() is False
    fresh_ts = (datetime.now(NY) - timedelta(seconds=300)).isoformat()
    (bridge.snap_dir / "snapshot.json").write_text(json.dumps({
        "timestamp": fresh_ts, "freshness_ttl_seconds": 600, "market_context": {},
    }))
    assert bridge.is_fresh() is True


def test_default_ttl_constant_is_unchanged():
    assert bridge_mod._DEFAULT_TTL_SECONDS == 600
    assert bridge_mod._FALLBACK_TTL_SECONDS > 900  # must exceed the 15-minute producer cadence
