"""S215 W6A: CLIO_LOG_APSCHEDULER_LEVEL in setup_logging.

setup_logging hard-codes the LIVE log dir, so the file handler is replaced with a
StreamHandler and makedirs is stubbed; root/security handlers and logger levels are restored.
"""

import io
import logging

import pytest

from agent import logging_config

APS = "apscheduler.executors.default"
APS_SCHED = "apscheduler.scheduler"


@pytest.fixture
def isolated_logging(monkeypatch):
    monkeypatch.setattr(logging_config, "RotatingFileHandler", lambda *a, **k: logging.StreamHandler(io.StringIO()))
    monkeypatch.setattr(logging_config.os, "makedirs", lambda *a, **k: None)
    monkeypatch.delenv("CLIO_LOG_APSCHEDULER_LEVEL", raising=False)

    root = logging.getLogger()
    sec = logging.getLogger("openclaw.security")
    tracked = [logging.getLogger(APS), logging.getLogger(APS_SCHED), root]
    saved_levels = [lg.level for lg in tracked]
    saved_root_handlers = root.handlers[:]
    saved_sec_handlers = sec.handlers[:]
    root.handlers = []
    try:
        yield
    finally:
        root.handlers = saved_root_handlers
        sec.handlers = saved_sec_handlers
        for lg, lvl in zip(tracked, saved_levels):
            lg.setLevel(lvl)


def test_default_is_warning(isolated_logging):
    logging_config.setup_logging("INFO")
    assert logging.getLogger(APS).level == logging.WARNING


@pytest.mark.parametrize("raw", ["INFO", "info", " Info "])
def test_env_info_is_applied(isolated_logging, monkeypatch, raw):
    monkeypatch.setenv("CLIO_LOG_APSCHEDULER_LEVEL", raw)
    logging_config.setup_logging("INFO")
    assert logging.getLogger(APS).level == logging.INFO


@pytest.mark.parametrize("raw,expected", [("DEBUG", logging.DEBUG), ("error", logging.ERROR), ("warning", logging.WARNING)])
def test_other_valid_levels_are_applied(isolated_logging, monkeypatch, raw, expected):
    monkeypatch.setenv("CLIO_LOG_APSCHEDULER_LEVEL", raw)
    logging_config.setup_logging("INFO")
    assert logging.getLogger(APS).level == expected


@pytest.mark.parametrize("raw", ["garbage", "", "CRITICAL", "10x"])
def test_unrecognised_env_falls_back_to_warning(isolated_logging, monkeypatch, raw):
    monkeypatch.setenv("CLIO_LOG_APSCHEDULER_LEVEL", raw)
    logging_config.setup_logging("INFO")
    assert logging.getLogger(APS).level == logging.WARNING


def test_scheduler_logger_and_root_untouched(isolated_logging, monkeypatch):
    sched = logging.getLogger(APS_SCHED)
    sched.setLevel(logging.NOTSET)
    monkeypatch.setenv("CLIO_LOG_APSCHEDULER_LEVEL", "INFO")
    logging_config.setup_logging("ERROR")
    assert sched.level == logging.NOTSET
    assert logging.getLogger().level == logging.ERROR


def test_repeated_call_still_applies_level(isolated_logging, monkeypatch):
    logging_config.setup_logging("INFO")
    monkeypatch.setenv("CLIO_LOG_APSCHEDULER_LEVEL", "INFO")
    logging_config.setup_logging("INFO")  # second call takes the duplicate-handler early return
    assert logging.getLogger(APS).level == logging.INFO
