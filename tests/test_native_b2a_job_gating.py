"""
B2a (native CLIO / Hydrinia-independence build): job-registration gating in
agent/integrations/scheduled_tasks.py.

Hydrinia (/home/agent/tradingbot) is permanently off. Jobs that only read/write
Hydrinia's process, logs, or bridge files must be registered ONLY when
CLIO_HYDRINIA_BRIDGE_ENABLED is truthy; CLIO-native jobs (DPOF, market regime,
bridge cleanup) must always register regardless.

register_scheduled_jobs() does real (lazy) imports of most CLIO subsystems, each
wrapped in its own try/except that logs on failure — so calling it end-to-end
against a fake Application/JobQueue is safe even when a sibling subsystem is
mid-development; a failed sibling import just means that job doesn't appear in
the registered set; it never propagates.
"""

import logging

import pytest

from agent.integrations import scheduled_tasks as st

_HYDRINIA_ONLY_JOBS = {
    "hydrinia_status_check",
    "hydrinia_error_scan",
    "hydrinia_portfolio_snapshot",
    "directive_feedback",
    "brier_resolution",
    "trade_idea_feedback",
    "athena_health_check",
}

_ALWAYS_ON_JOBS = {
    "bridge_cleanup",
    "market_regime_refresh",
    "dpof_cycle",
    "self_check",
    "system_health_report",
}


class FakeJobQueue:
    def __init__(self):
        self.repeating: list[str] = []
        self.once: list[str] = []

    def run_repeating(self, func, interval=None, first=None, name=None, **kw):
        self.repeating.append(name)

    def run_once(self, func, when=None, name=None, **kw):
        self.once.append(name)


class FakeApp:
    def __init__(self):
        self.job_queue = FakeJobQueue()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Isolate from the real environment/data dir and give every job at least
    one chat_id so the Telegram-dependent branch (which the Hydrinia-only jobs
    live under) is exercised."""
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof"))
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    monkeypatch.setattr(st, "_get_chat_ids", lambda: ["test_chat"])


def _registered_names(app: FakeApp) -> set[str]:
    return set(app.job_queue.repeating) | set(app.job_queue.once)


def test_bridge_off_skips_hydrinia_only_jobs(monkeypatch, caplog):
    """Default (bridge unset) — Hydrinia-only jobs are never registered; CLIO-native
    jobs still are; one consolidated INFO line names every skipped job."""
    caplog.set_level(logging.INFO, logger="openclaw.scheduled")
    app = FakeApp()
    st.register_scheduled_jobs(app)

    names = _registered_names(app)
    for job in _HYDRINIA_ONLY_JOBS:
        assert job not in names, f"{job} should not register with the bridge off"
    for job in _ALWAYS_ON_JOBS:
        assert job in names, f"{job} should register regardless of the bridge"

    summary = [r for r in caplog.records if "skipped" in r.message and "Hydrinia" in r.message]
    assert len(summary) == 1, "expected exactly one consolidated skipped-jobs summary line"
    for job in _HYDRINIA_ONLY_JOBS:
        assert job in summary[0].message


def test_bridge_on_registers_hydrinia_only_jobs(monkeypatch):
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    app = FakeApp()
    st.register_scheduled_jobs(app)

    names = _registered_names(app)
    for job in _HYDRINIA_ONLY_JOBS:
        assert job in names, f"{job} should register with the bridge on"
    for job in _ALWAYS_ON_JOBS:
        assert job in names


def test_personality_snapshot_cold_sync_bridge_gated(monkeypatch):
    """Personality snapshot export targets the Hydrinia bridge dir — only fires
    when both personality AND the bridge are enabled."""
    monkeypatch.setenv("PERSONALITY_ENABLED", "true")

    app_off = FakeApp()
    st.register_scheduled_jobs(app_off)
    assert "personality_snapshot_cold_sync" not in _registered_names(app_off)

    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    app_on = FakeApp()
    st.register_scheduled_jobs(app_on)
    assert "personality_snapshot_cold_sync" in _registered_names(app_on)


def test_market_regime_refresh_registers_with_timeout_wrapper(monkeypatch):
    """New job: lazily imports market_regime.refresh_market_regime, 1800s interval,
    first run ~60s, uses the same _with_timeout convention as sibling jobs."""
    app = FakeApp()
    st.register_scheduled_jobs(app)
    assert "market_regime_refresh" in app.job_queue.repeating


def test_market_regime_module_missing_does_not_crash(monkeypatch):
    """If agent.integrations.market_regime can't be imported, registration must
    log a warning and continue — never raise."""
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "agent.integrations.market_regime":
            raise ImportError("simulated missing module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    app = FakeApp()
    # Must not raise even though market_regime import fails.
    st.register_scheduled_jobs(app)
    assert "market_regime_refresh" not in _registered_names(app)
