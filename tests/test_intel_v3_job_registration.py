"""
Intel v3 (2026-09-25): scheduler registration for the advisory intel jobs —
vol_surface_refresh, econ_surprise_refresh, newsflash_refresh — in
agent/integrations/scheduled_tasks.register_scheduled_jobs().

Each job must register with the _with_timeout wrapper at its module's
advertised cadence, and a missing/broken module must only skip its own job.
"""

import builtins

import pytest

from agent.integrations import scheduled_tasks as st

_INTEL_JOBS = ("vol_surface_refresh", "econ_surprise_refresh", "newsflash_refresh")


class FakeJobQueue:
    def __init__(self):
        self.repeating: dict[str, dict] = {}
        self.once: list[str] = []

    def run_repeating(self, func, interval=None, first=None, name=None, **kw):
        self.repeating[name] = {"func": func, "interval": interval, "first": first}

    def run_once(self, func, when=None, name=None, **kw):
        self.once.append(name)


class FakeApp:
    def __init__(self):
        self.job_queue = FakeJobQueue()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof"))
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    monkeypatch.setattr(st, "_get_chat_ids", lambda: ["test_chat"])


def test_intel_jobs_register_with_timeout_wrapper():
    app = FakeApp()
    st.register_scheduled_jobs(app)
    for name in _INTEL_JOBS:
        assert name in app.job_queue.repeating, f"{name} not registered"
        job = app.job_queue.repeating[name]
        assert job["func"].__name__ == f"{name}_timeout_wrapped"
        assert isinstance(job["interval"], (int, float)) and job["interval"] > 0


def test_intel_job_intervals_follow_module_constants():
    from agent.integrations import econ_surprise, newsflash_lite

    app = FakeApp()
    st.register_scheduled_jobs(app)
    jobs = app.job_queue.repeating
    assert jobs["vol_surface_refresh"]["interval"] == 1800
    assert jobs["econ_surprise_refresh"]["interval"] == getattr(econ_surprise, "JOB_INTERVAL_SECONDS", 15)
    assert jobs["newsflash_refresh"]["interval"] == getattr(newsflash_lite, "JOB_INTERVAL_SECONDS", 60)


@pytest.mark.parametrize("module_name,job_name", [
    ("vol_surface", "vol_surface_refresh"),
    ("econ_surprise", "econ_surprise_refresh"),
    ("newsflash_lite", "newsflash_refresh"),
])
def test_missing_intel_module_skips_only_its_job(monkeypatch, module_name, job_name):
    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        target = f"agent.integrations.{module_name}"
        if name == target or (name == "agent.integrations" and module_name in (fromlist or ())):
            raise ImportError(f"simulated missing {target}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    app = FakeApp()
    st.register_scheduled_jobs(app)  # must not raise
    assert job_name not in app.job_queue.repeating
    for other in _INTEL_JOBS:
        if other != job_name:
            assert other in app.job_queue.repeating, f"{other} should still register"
    assert "dpof_cycle" in app.job_queue.repeating
