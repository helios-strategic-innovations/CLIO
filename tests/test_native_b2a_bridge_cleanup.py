"""
B2a: _cleanup_bridge_files() split behavior.

Hydrinia (/home/agent/tradingbot) is permanently off. The daily bridge_cleanup
job stays registered (see test_native_b2a_job_gating.py), but the function it
runs must skip pruning Hydrinia's moltbot_knowledge/ bridge dir when the bridge
is off while continuing to run CLIO-native housekeeping (trade idea / SEAL impl
pruning, WAL checkpoint) unconditionally.

This test never touches the real repo's data/knowledge/ — the real project's
data/knowledge paths are guarded off (os.path.isdir forced False for them) so
the CLIO-native prune loops see "directory absent" and no-op, while the
moltbot_knowledge/ bridge dir under a tmp_path IS exercised for real.
"""

import asyncio
import os
import time

import pytest

from agent.integrations import scheduled_tasks as st


class FakeContext:
    bot = None


def _guard_real_data_knowledge(monkeypatch):
    """Force os.path.isdir(...) False for any path under the real repo's
    data/knowledge/ tree, so the CLIO-native prune loops (which derive their
    path from __file__, not from config.tradingbot_dir) never touch real data."""
    real_isdir = os.path.isdir

    def _guarded(path):
        normalized = str(path).replace(os.sep, "/")
        if "/data/knowledge/" in normalized:
            return False
        return real_isdir(path)

    monkeypatch.setattr(os.path, "isdir", _guarded)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    monkeypatch.setattr(
        st.config, "tradingbot_dir", str(tmp_path / "tradingbot"), raising=False
    )
    _guard_real_data_knowledge(monkeypatch)

    # Neutralize the WAL checkpoint call — unrelated to this test and must not
    # touch the real conversation DB under data/.
    from agent.core import conversation_manager

    async def _noop_checkpoint():
        return None

    monkeypatch.setattr(conversation_manager, "checkpoint", _noop_checkpoint)


def _run(coro):
    return asyncio.run(coro)


def _make_old_file(path, days_old):
    path.write_text("x")
    old_time = time.time() - days_old * 86400
    os.utime(path, (old_time, old_time))


def test_bridge_off_leaves_moltbot_knowledge_untouched(tmp_path):
    bridge_dir = tmp_path / "tradingbot" / "moltbot_knowledge" / "market_analysis"
    bridge_dir.mkdir(parents=True)
    stale = bridge_dir / "stale.json"
    _make_old_file(stale, days_old=30)

    _run(st._cleanup_bridge_files(FakeContext()))

    assert stale.exists(), "moltbot_knowledge/ must not be pruned while the bridge is off"


def test_bridge_on_prunes_moltbot_knowledge(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    bridge_dir = tmp_path / "tradingbot" / "moltbot_knowledge" / "market_analysis"
    bridge_dir.mkdir(parents=True)
    stale = bridge_dir / "stale.json"
    _make_old_file(stale, days_old=30)
    fresh = bridge_dir / "fresh.json"
    fresh.write_text("x")

    _run(st._cleanup_bridge_files(FakeContext()))

    assert not stale.exists(), "moltbot_knowledge/ must be pruned once the bridge is on"
    assert fresh.exists()


def test_wal_checkpoint_always_runs(tmp_path, monkeypatch):
    """CLIO-native housekeeping (WAL checkpoint) must run regardless of bridge state."""
    calls = []

    from agent.core import conversation_manager

    async def _tracked_checkpoint():
        calls.append(1)

    monkeypatch.setattr(conversation_manager, "checkpoint", _tracked_checkpoint)

    _run(st._cleanup_bridge_files(FakeContext()))
    assert calls == [1]
