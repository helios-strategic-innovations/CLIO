#!/usr/bin/env python3
"""Step-2 tests for fail-visible, row-scoped history recovery.

These tests use only tmp_path SQLite files with synthetic dummy rows. They do
not read or mutate CLIO runtime data. The production mutations they catch are:
one malformed persisted history row collapsing the whole conversation to [],
and the outer get_history handler hiding the exception class/traceback or
logging the conversation id.
"""

import asyncio
import importlib.util
import json
import logging
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _load_persistence_module():
    spec = importlib.util.spec_from_file_location(
        "step2_persistence_under_test",
        REPO / "agent/memory/persistence.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_history_skips_poison_row_and_warns_rowid_only(tmp_path, monkeypatch, caplog):
    persistence = _load_persistence_module()

    old_db_path = persistence.DB_PATH
    persistence.DB_PATH = tmp_path / "conversations.db"
    manager = persistence.PersistentConversationManager(max_history=10)
    cid = "step2-poison-cid"
    poison_content = "{not-json"

    async def scenario():
        try:
            db = await manager._ensure_db()
            await db.execute(
                "INSERT INTO conversations (conversation_id, role, content, timestamp, token_estimate) "
                "VALUES (?, ?, ?, ?, ?)",
                (cid, "user", json.dumps("good-one"), 1.0, 2),
            )
            cursor = await db.execute(
                "INSERT INTO conversations (conversation_id, role, content, timestamp, token_estimate) "
                "VALUES (?, ?, ?, ?, ?)",
                (cid, "assistant", poison_content, 2.0, 2),
            )
            poison_rowid = cursor.lastrowid
            await db.execute(
                "INSERT INTO conversations (conversation_id, role, content, timestamp, token_estimate) "
                "VALUES (?, ?, ?, ?, ?)",
                (cid, "assistant", json.dumps({"ok": True}), 3.0, 2),
            )
            await db.commit()

            with caplog.at_level(logging.WARNING, logger="openclaw.memory"):
                result = await manager.get_history(cid)

            return poison_rowid, result
        finally:
            await manager.close()
            persistence.DB_PATH = old_db_path

    poison_rowid, result = asyncio.run(scenario())

    assert result == [
        {"role": "user", "content": "good-one"},
        {"role": "assistant", "content": {"ok": True}},
    ]

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert any(str(poison_rowid) in message for message in warning_messages)
    assert all(cid not in message for message in warning_messages)
    assert all(poison_content not in message for message in warning_messages)


def test_get_history_outer_failure_logs_exception_class_and_traceback_without_cid(monkeypatch, caplog):
    persistence = _load_persistence_module()

    manager = persistence.PersistentConversationManager(max_history=10)
    cid = "step2-outer-cid"

    async def fail_ensure_db():
        raise RuntimeError("connection broke")

    monkeypatch.setattr(manager, "_ensure_db", fail_ensure_db)

    async def scenario():
        with caplog.at_level(logging.ERROR, logger="openclaw.memory"):
            return await manager.get_history(cid)

    result = asyncio.run(scenario())

    assert result == []
    error_records = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR and "Failed to load history" in record.getMessage()
    ]
    assert len(error_records) == 1
    record = error_records[0]
    assert "RuntimeError" in record.getMessage()
    assert cid not in record.getMessage()
    assert record.exc_info is not None
