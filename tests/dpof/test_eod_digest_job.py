"""Tests for the DPOF EOD digest scheduled job (Task ORCH-7Q4K-DPOF,
2026-09-25) -- agent.integrations.scheduled_tasks's `dpof_eod_digest`
registration, the `_with_timeout` wrapper, and _guarded_dpof_eod_digest's
own gating (CLIO_DPOF_EOD_DIGEST, MARKET_HOLIDAYS skip,
_daily_task_already_ran dedup), mirroring tests/test_intel_v3_job_registration.py's
FakeJobQueue pattern.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from agent.integrations import scheduled_tasks as st


class FakeJobQueue:
    def __init__(self):
        self.repeating: dict[str, dict] = {}
        self.daily: dict[str, dict] = {}
        self.once: list[str] = []

    def run_daily(self, func, time=None, days=None, name=None, **kw):
        self.daily[name] = {"func": func, "time": time, "days": days}

    def run_repeating(self, func, interval=None, first=None, name=None, **kw):
        self.repeating[name] = {"func": func, "interval": interval, "first": first}

    def run_once(self, func, when=None, name=None, **kw):
        self.once.append(name)


class FakeApp:
    def __init__(self):
        self.job_queue = FakeJobQueue()


class FakeBot:
    def __init__(self):
        self.sent: list[tuple] = []

    async def send_message(self, chat_id=None, text=None, **kw):
        self.sent.append((chat_id, text))


class FakeContext:
    def __init__(self, bot):
        self.bot = bot


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof"))
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    monkeypatch.setattr(st, "_get_chat_ids", lambda: ["test_chat"])
    # Never touch the real repo's data/daily_task_runs.json from this test file.
    monkeypatch.setattr(st, "_DAILY_TASK_RAN_FILE", str(tmp_path / "daily_task_runs.json"))
    monkeypatch.setattr(st, "_DAILY_TASK_LOCK_FILE", str(tmp_path / "daily_task_runs.json.lock"))
    # A fixed weekday after the close, so the job's own gating is deterministic.
    monkeypatch.setattr(st, "_dpof_eod_now", lambda: dt.datetime(2026, 9, 25, 16, 10, tzinfo=_ET))


_ET = __import__("zoneinfo").ZoneInfo("America/New_York")


# ─────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────

def test_dpof_eod_digest_registers_with_timeout_wrapper():
    app = FakeApp()
    st.register_scheduled_jobs(app)
    assert "dpof_eod_digest" in app.job_queue.daily
    job = app.job_queue.daily["dpof_eod_digest"]
    assert job["func"].__name__ == "dpof_eod_digest_timeout_wrapped"
    # 16:10 ET, tz-aware (follows DST), Mon-Fri in PTB's 0=Sunday numbering
    assert (job["time"].hour, job["time"].minute) == (16, 10)
    assert str(job["time"].tzinfo) == "America/New_York"
    assert job["days"] == (1, 2, 3, 4, 5)


def test_dpof_eod_digest_registers_alongside_dpof_cycle():
    app = FakeApp()
    st.register_scheduled_jobs(app)
    assert "dpof_cycle" in app.job_queue.repeating
    assert "dpof_eod_digest" in app.job_queue.daily


def test_missing_readmodel_module_skips_only_its_job(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agent.dpof.readmodel" or (name == "agent.dpof" and "readmodel" in (fromlist or ())):
            raise ImportError("simulated missing agent.dpof.readmodel")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    app = FakeApp()
    st.register_scheduled_jobs(app)  # must not raise
    assert "dpof_eod_digest" not in app.job_queue.daily
    assert "dpof_cycle" in app.job_queue.repeating  # unaffected


# ─────────────────────────────────────────────────────────────────────────
# _guarded_dpof_eod_digest behavior (extracted from the registered job)
# ─────────────────────────────────────────────────────────────────────────

def _get_guarded_fn(app: FakeApp):
    return app.job_queue.daily["dpof_eod_digest"]["func"]


def test_guarded_digest_sends_to_every_chat_id(monkeypatch, tmp_path):
    from agent.dpof import readmodel

    monkeypatch.setattr(readmodel, "eod_digest_text", lambda: "DPOF EOD digest — test body")
    monkeypatch.setattr(st, "_get_chat_ids", lambda: ["chat1", "chat2"])

    app = FakeApp()
    st.register_scheduled_jobs(app)
    fn = _get_guarded_fn(app)
    bot = FakeBot()

    asyncio.run(fn(FakeContext(bot)))

    assert len(bot.sent) == 2
    assert all(text == "DPOF EOD digest — test body" for _cid, text in bot.sent)


def test_guarded_digest_skips_when_eod_digest_disabled(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_EOD_DIGEST", "false")
    app = FakeApp()
    st.register_scheduled_jobs(app)
    fn = _get_guarded_fn(app)
    bot = FakeBot()

    asyncio.run(fn(FakeContext(bot)))

    assert bot.sent == []


def test_guarded_digest_skips_on_market_holiday(monkeypatch):
    import agent.integrations.market_events as market_events

    monkeypatch.setattr(market_events, "MARKET_HOLIDAYS", frozenset({dt.date(2026, 9, 25)}))

    app = FakeApp()
    st.register_scheduled_jobs(app)
    fn = _get_guarded_fn(app)
    bot = FakeBot()

    asyncio.run(fn(FakeContext(bot)))

    assert bot.sent == []


def test_guarded_digest_proceeds_when_holiday_check_unavailable(monkeypatch):
    """A broken/missing market_events import must not silently swallow the
    digest -- it proceeds (fail-open on the ADVISORY holiday check; the real
    gate is _daily_task_already_ran + the job's own 16:10 ET schedule)."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agent.integrations.market_events":
            raise ImportError("simulated")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    from agent.dpof import readmodel

    monkeypatch.setattr(readmodel, "eod_digest_text", lambda: "body")

    app = FakeApp()
    st.register_scheduled_jobs(app)
    fn = _get_guarded_fn(app)
    bot = FakeBot()

    asyncio.run(fn(FakeContext(bot)))

    assert len(bot.sent) == 1


def test_guarded_digest_runs_once_per_day(monkeypatch):
    from agent.dpof import readmodel

    monkeypatch.setattr(readmodel, "eod_digest_text", lambda: "body")

    app = FakeApp()
    st.register_scheduled_jobs(app)
    fn = _get_guarded_fn(app)
    bot = FakeBot()

    asyncio.run(fn(FakeContext(bot)))
    assert len(bot.sent) == 1

    asyncio.run(fn(FakeContext(bot)))  # second run same day -- must be a no-op
    assert len(bot.sent) == 1


@pytest.mark.parametrize("now", [
    dt.datetime(2026, 9, 26, 16, 10, tzinfo=_ET),   # Saturday
    dt.datetime(2026, 9, 27, 16, 10, tzinfo=_ET),   # Sunday
    dt.datetime(2026, 9, 25, 15, 10, tzinfo=_ET),   # weekday, before the close
])
def test_guarded_digest_skips_weekends_and_mid_session(monkeypatch, now):
    from agent.dpof import readmodel

    monkeypatch.setattr(readmodel, "eod_digest_text", lambda: "body")
    monkeypatch.setattr(st, "_dpof_eod_now", lambda: now)
    app = FakeApp()
    st.register_scheduled_jobs(app)
    bot = FakeBot()
    asyncio.run(_get_guarded_fn(app)(FakeContext(bot)))
    assert bot.sent == []
    # skipped days are not marked as ran
    assert st._daily_task_already_ran("dpof_eod_digest") is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


def test_restart_catch_up_uses_clock_hook(monkeypatch):
    """A restart between 16:10 and 18:10 on a weekday schedules a catch-up run."""
    monkeypatch.setattr(st, "_dpof_eod_now", lambda: dt.datetime(2026, 9, 25, 16, 30, tzinfo=_ET))
    app = FakeApp()
    st.register_scheduled_jobs(app)
    assert "dpof_eod_digest_catchup" in app.job_queue.once


def test_no_catch_up_outside_window(monkeypatch):
    monkeypatch.setattr(st, "_dpof_eod_now", lambda: dt.datetime(2026, 9, 25, 12, 0, tzinfo=_ET))
    app = FakeApp()
    st.register_scheduled_jobs(app)
    assert "dpof_eod_digest_catchup" not in app.job_queue.once


def test_long_digest_is_chunked(monkeypatch):
    from agent.dpof import readmodel

    body = "\n".join("z" * 100 for _ in range(100))  # ~10k chars
    monkeypatch.setattr(readmodel, "eod_digest_text", lambda: body)
    app = FakeApp()
    st.register_scheduled_jobs(app)
    bot = FakeBot()
    asyncio.run(_get_guarded_fn(app)(FakeContext(bot)))
    assert len(bot.sent) >= 3 and all(len(t) <= 4096 for _cid, t in bot.sent)
