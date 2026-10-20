"""SEAL bandit disabled-category filter and calibration upsert (S215 / R-28).

All state is redirected to tmp_path; the clock is frozen by patching the
module-level ``datetime`` names. No network, no live data files.
"""
import random
import types
from datetime import datetime, timedelta, timezone

import pytest


T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


class _FrozenClock:
    def __init__(self, start):
        self.now = start


def _fake_datetime(clock):
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now if tz is not None else clock.now.replace(tzinfo=None)
    return FakeDT


# ---------------------------------------------------------------------------
# (a) SEAL bandit never selects a disabled category
# ---------------------------------------------------------------------------

def test_options_is_in_disabled_categories():
    import agent.integrations.seal_category_bandit as bmod
    assert "options" in bmod.DISABLED_CATEGORIES
    assert {"equity", "history"} <= set(bmod.DISABLED_CATEGORIES)


def test_select_category_never_returns_disabled_over_2000_pulls(tmp_path, monkeypatch):
    import agent.integrations.seal_category_bandit as bmod

    monkeypatch.setattr(bmod, "_BANDIT_STATE_FILE", str(tmp_path / "bandit.json"))
    monkeypatch.setenv("MOLTBOT_SEAL_CATEGORY_BANDIT", "1")
    # LOG_ONLY=1 (the code default) makes select_category return "" by design;
    # turn it off so the returned category is the one under test.
    monkeypatch.setenv("MOLTBOT_SEAL_CATEGORY_BANDIT_LOG_ONLY", "0")
    random.seed(20261012)

    bandit = bmod.CategorySelectionBandit()
    # Leave the exploration phase so the Thompson-sampling branch runs.
    bandit._meta["total_pulls"] = bmod.MIN_PULLS_FOR_EXPLOITATION + 1
    base_weights = {c: 1.0 for c in bmod.CATEGORY_NAMES}

    seen = set()
    for _ in range(2000):
        seen.add(bandit.select_category("BULL", base_weights))

    disabled = set(bmod.DISABLED_CATEGORIES)
    assert seen, "bandit returned nothing"
    assert seen.isdisjoint(disabled), f"disabled categories selected: {seen & disabled}"
    assert seen <= set(bmod.CATEGORY_NAMES) - disabled


# ---------------------------------------------------------------------------
# (b) calibration row for SEAL trade-idea win rate is a single upserted row
# ---------------------------------------------------------------------------

class _FakeRewardEngine:
    def __init__(self, holder):
        self._holder = holder

    def get_reward_summary(self):
        return dict(self._holder["summary"])


class _FakeProcedural:
    def get_all_patterns(self):
        return []

    def add_pattern(self, *a, **kw):
        return None


@pytest.fixture
def calib_env(tmp_path, monkeypatch):
    import agent.memory.meta as meta_mod
    import agent.memory.memory_engine as me_mod
    import agent.integrations.seal_reward_engine as sre

    monkeypatch.setattr(meta_mod, "DATA_DIR", str(tmp_path / "memdata"))
    monkeypatch.setenv("MOLTBOT_SEAL_REWARDS", "1")

    clock = _FrozenClock(T0)
    FakeDT = _fake_datetime(clock)
    monkeypatch.setattr(meta_mod, "datetime", FakeDT)
    monkeypatch.setattr(me_mod, "datetime", FakeDT)

    holder = {"summary": {"total_evaluated": 27, "win_rate": 0.0, "total_wins": 0}}
    monkeypatch.setattr(sre, "get_reward_engine", lambda: _FakeRewardEngine(holder))

    meta = meta_mod.MetaMemory()
    engine = types.SimpleNamespace(meta=meta, procedural=_FakeProcedural())
    return types.SimpleNamespace(
        clock=clock, holder=holder, meta=meta, engine=engine, me_mod=me_mod,
    )


def _seal_rows(meta):
    return [c for c in meta._data.get("calibration", [])
            if str(c.get("prediction_id", "")).startswith("seal_t")]


def test_two_writes_seven_hours_apart_leave_one_updated_row(calib_env):
    env = calib_env
    env.me_mod.MemoryEngine.integrate_seal_signals(env.engine)
    rows = _seal_rows(env.meta)
    assert len(rows) == 1
    assert rows[0]["prediction_id"] == "seal_ti_quality"
    assert "0% (0/27)" in rows[0]["description"]

    env.clock.now = T0 + timedelta(hours=7)
    env.holder["summary"] = {"total_evaluated": 40, "win_rate": 0.5, "total_wins": 20}
    stats = env.me_mod.MemoryEngine.integrate_seal_signals(env.engine)

    assert stats["meta_updated"] is True
    rows = _seal_rows(env.meta)
    assert len(rows) == 1, f"expected one row, got {len(rows)}"
    assert rows[0]["prediction_id"] == "seal_ti_quality"
    assert "50% (20/40)" in rows[0]["description"]
    assert rows[0]["timestamp"] == (T0 + timedelta(hours=7)).isoformat()


def test_throttle_blocks_second_write_inside_six_hours(calib_env):
    env = calib_env
    env.me_mod.MemoryEngine.integrate_seal_signals(env.engine)
    first_ts = _seal_rows(env.meta)[0]["timestamp"]

    env.clock.now = T0 + timedelta(hours=1)
    env.holder["summary"] = {"total_evaluated": 40, "win_rate": 0.5, "total_wins": 20}
    stats = env.me_mod.MemoryEngine.integrate_seal_signals(env.engine)

    assert stats["meta_updated"] is False
    rows = _seal_rows(env.meta)
    assert len(rows) == 1
    assert rows[0]["timestamp"] == first_ts
    assert "0% (0/27)" in rows[0]["description"]


def test_record_prediction_default_still_only_replaces_unresolved(calib_env):
    """Default callers keep the S122 4B semantics: a resolved row is not replaced."""
    meta = calib_env.meta
    meta.record_prediction("pid_x", "trade_ideas", 0.6, "first")
    meta.record_outcome("pid_x", True)
    meta.record_prediction("pid_x", "trade_ideas", 0.6, "second")
    rows = meta._data["calibration"]
    assert len(rows) == 2
    meta.record_prediction("pid_y", "trade_ideas", 0.6, "a")
    meta.record_prediction("pid_y", "trade_ideas", 0.7, "b")
    assert len([r for r in rows if r["prediction_id"] == "pid_y"]) == 1
