"""S215 W6A: per-job duration logging in scheduled_tasks._with_timeout.

Uses an injected clock (scheduled_tasks._clock) and caplog; no real sleeps, no wall clock.
"""

import asyncio
import logging

import pytest

from agent.integrations import scheduled_tasks

LOGGER_NAME = "openclaw.scheduled"


@pytest.fixture
def fake_clock(monkeypatch):
    """Returns a setter: feed the start/end readings the wrapper will see."""

    def set_readings(*readings):
        it = iter(readings)
        monkeypatch.setattr(scheduled_tasks, "_clock", lambda: next(it))

    monkeypatch.setattr(scheduled_tasks, "_record_timeout", lambda name: 1)
    return set_readings


@pytest.fixture
def caplog_info(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return caplog


def _run(job, timeout_s, name="j"):
    wrapped = scheduled_tasks._with_timeout(job, timeout_s, name)
    return asyncio.run(wrapped(None))


def _job_lines(caplog):
    return [r for r in caplog.records if "job_duration" in r.getMessage()]


async def _ok(context):
    return None


def test_six_second_job_logs_one_info_line(fake_clock, caplog_info):
    fake_clock(100.0, 106.0)
    assert _run(_ok, 60) is None
    lines = _job_lines(caplog_info)
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert lines[0].getMessage() == "job_duration name=j s=6.0"


def test_one_second_job_logs_nothing(fake_clock, caplog_info):
    fake_clock(100.0, 101.0)
    assert _run(_ok, 60) is None
    assert _job_lines(caplog_info) == []


def test_exactly_five_seconds_logs_info(fake_clock, caplog_info):
    fake_clock(0.0, 5.0)
    _run(_ok, 60)
    lines = _job_lines(caplog_info)
    assert [r.getMessage() for r in lines] == ["job_duration name=j s=5.0"]


def test_job_at_eighty_percent_of_timeout_logs_warning(fake_clock, caplog_info):
    # timeout 10 s -> 80% is 8.0 s; 8.0 must WARN (the boundary is inclusive).
    fake_clock(0.0, 8.0)
    _run(_ok, 10)
    lines = _job_lines(caplog_info)
    assert len(lines) == 1
    assert lines[0].levelno == logging.WARNING
    assert lines[0].getMessage() == (
        "job_duration name=j s=8.0 timeout=10s (>=80% of the timeout)"
    )


def test_job_just_under_eighty_percent_logs_info_not_warning(fake_clock, caplog_info):
    fake_clock(0.0, 7.9)
    _run(_ok, 10)
    lines = _job_lines(caplog_info)
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert lines[0].getMessage() == "job_duration name=j s=7.9"


def test_timeout_still_logs_existing_error_and_returns_none(fake_clock, caplog_info):
    fake_clock(0.0)  # success-path clock read must not happen on the timeout path

    async def times_out(context):
        raise asyncio.TimeoutError()

    assert _run(times_out, 10, name="slow") is None
    errs = [r for r in caplog_info.records if r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errs] == [
        "Scheduled task 'slow' timed out after 10s (today_count=1)"
    ]
    assert _job_lines(caplog_info) == []


def test_exception_still_logs_existing_error_and_returns_none(fake_clock, caplog_info):
    fake_clock(0.0)

    async def boom(context):
        raise ValueError("kaboom")

    assert _run(boom, 10, name="bad") is None
    errs = [r for r in caplog_info.records if r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errs] == ["Scheduled task 'bad' failed: kaboom"]
    assert _job_lines(caplog_info) == []


def test_wrapped_name_and_context_passthrough(fake_clock, caplog_info):
    fake_clock(0.0, 0.0)
    seen = []

    async def record(context):
        seen.append(context)

    wrapped = scheduled_tasks._with_timeout(record, 60, "named")
    assert wrapped.__name__ == "named_timeout_wrapped"
    asyncio.run(wrapped("ctx"))
    assert seen == ["ctx"]


def test_duration_spans_the_job_itself_not_just_two_fixed_readings(monkeypatch, caplog_info):
    # The start reading must be taken BEFORE the job runs: here the clock only moves while the job runs, so a start
    # reading taken after the await measures 0 s and the line vanishes (R-33 fix-up: pins reading order).
    now = {"t": 100.0}
    monkeypatch.setattr(scheduled_tasks, "_clock", lambda: now["t"])
    monkeypatch.setattr(scheduled_tasks, "_record_timeout", lambda name: 1)

    async def job(context):
        now["t"] += 6.0

    _run(job, 60, name="moves_clock")
    assert [r.getMessage() for r in _job_lines(caplog_info)] == ["job_duration name=moves_clock s=6.0"]
