"""
P1-4: Tests for the Hydrinia error-log scanner (agent/integrations/scheduled_tasks.py).

All fixtures live under pytest's tmp_path — never a real production log/vault path.
"""

import asyncio
import logging
import os

import pytest

from agent.integrations import scheduled_tasks as st


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text))


class FakeContext:
    def __init__(self, bot):
        self.bot = bot


@pytest.fixture(autouse=True)
def scanner_state(monkeypatch, tmp_path):
    """Point the scanner at an isolated tmp log file and reset module state."""
    log_path = tmp_path / "hydrinia_errors.log"
    monkeypatch.setattr(st, "_LOG_FILE", str(log_path))
    monkeypatch.setattr(st, "_last_log_size", 0)
    monkeypatch.setattr(st, "_last_log_ino", 0)
    monkeypatch.setattr(st, "_LOG_SCAN_SIGNAL_KEYS", set())
    monkeypatch.setattr(st, "_get_chat_ids", lambda: ["test_chat"])
    bot = FakeBot()
    ctx = FakeContext(bot)
    return {"log_path": log_path, "bot": bot, "ctx": ctx}


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _scan(ctx):
    asyncio.run(st.scan_hydrinia_errors(ctx))


def test_padded_error_fires(scanner_state):
    log_path = scanner_state["log_path"]
    ctx = scanner_state["ctx"]
    bot = scanner_state["bot"]

    _write(log_path, "2026-08-04 12:00:00 | ERROR    | Hydrinia             | boom\n")

    _scan(ctx)

    assert len(bot.sent) == 1
    assert "boom" in bot.sent[0][1]


def test_padded_critical_fires(scanner_state):
    log_path = scanner_state["log_path"]
    ctx = scanner_state["ctx"]
    bot = scanner_state["bot"]

    _write(log_path, "2026-08-04 12:00:00 | CRITICAL | Hydrinia             | meltdown\n")

    _scan(ctx)

    assert len(bot.sent) == 1
    assert "meltdown" in bot.sent[0][1]


def test_clean_fixture_is_silent(scanner_state):
    log_path = scanner_state["log_path"]
    ctx = scanner_state["ctx"]
    bot = scanner_state["bot"]

    _write(
        log_path,
        "2026-08-04 12:00:00 | INFO     | Hydrinia             | all good\n"
        "2026-08-04 12:00:01 | DEBUG    | Hydrinia             | still good\n",
    )

    _scan(ctx)

    assert bot.sent == []


def test_missing_target_emits_warn_once(scanner_state, caplog):
    ctx = scanner_state["ctx"]
    bot = scanner_state["bot"]
    # log_path was never created -> os.stat() raises FileNotFoundError

    caplog.set_level(logging.WARNING, logger="openclaw.scheduled")

    _scan(ctx)
    _scan(ctx)
    _scan(ctx)

    assert bot.sent == []
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_records) == 1, (
        f"expected exactly one warn-once signal across 3 calls, got {len(warn_records)}"
    )


def test_rename_rotation_skips_no_lines(scanner_state):
    log_path = scanner_state["log_path"]
    ctx = scanner_state["ctx"]
    bot = scanner_state["bot"]

    # Initial file, fully scanned.
    _write(log_path, "2026-08-04 12:00:00 | ERROR    | Hydrinia             | first\n")
    _scan(ctx)
    assert len(bot.sent) == 1
    assert "first" in bot.sent[0][1]

    # logrotate 'create' style: rename old file aside, create a brand-new file at
    # the same path (new inode), then append to it — must not skip these lines,
    # even though the new file starts smaller than the old offset.
    rotated_path = str(log_path) + ".1"
    os.rename(log_path, rotated_path)
    _write(log_path, "2026-08-04 12:05:00 | CRITICAL | Hydrinia             | second\n")

    _scan(ctx)

    assert len(bot.sent) == 2
    assert "second" in bot.sent[1][1]

    # Further appends to the new (post-rotation) file continue to be picked up.
    with open(log_path, "a") as f:
        f.write("2026-08-04 12:06:00 | ERROR    | Hydrinia             | third\n")

    _scan(ctx)

    assert len(bot.sent) == 3
    assert "third" in bot.sent[2][1]
