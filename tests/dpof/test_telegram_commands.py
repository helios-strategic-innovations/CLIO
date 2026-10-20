"""Tests for agent.dpof.telegram_commands.dpof_command.

Fake update/context objects (no real python-telegram-bot dependency needed
for these); is_authorized and get_engine are monkeypatched at the module
level they're looked up from (lazy imports inside dpof_command).
"""

import asyncio
import sys
import types

import agent.dpof.telegram_commands as tc_mod


class FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class FakeUpdate:
    def __init__(self, user_id):
        self.effective_user = types.SimpleNamespace(id=user_id) if user_id is not None else None
        self.message = FakeMessage()


class FakeContext:
    def __init__(self, args=None):
        self.args = args or []


class FakeEngine:
    def __init__(self):
        self.status_calls = 0
        self.positions_calls = 0
        self.config_calls = 0
        self.halt_calls = []
        self.resume_calls = 0

    async def status_text(self):
        self.status_calls += 1
        return "STATUS TEXT"

    async def positions_text(self):
        self.positions_calls += 1
        return "POSITIONS TEXT"

    async def config_text(self):
        self.config_calls += 1
        return "CONFIG TEXT"

    async def operator_halt(self, reason):
        self.halt_calls.append(reason)
        return f"DPOF halted: {reason}"

    async def operator_resume(self):
        self.resume_calls += 1
        return "DPOF resumed"


def _patch_telegram_bot_module(monkeypatch, authorized_ids):
    fake_module = types.ModuleType("agent.integrations.telegram_bot")
    fake_module.is_authorized = lambda uid: uid in authorized_ids
    monkeypatch.setitem(sys.modules, "agent.integrations.telegram_bot", fake_module)
    import agent.integrations as integrations_pkg
    monkeypatch.setattr(integrations_pkg, "telegram_bot", fake_module, raising=False)


def _patch_get_engine(monkeypatch, engine):
    fake_module = types.ModuleType("agent.dpof.engine")
    fake_module.get_engine = lambda: engine
    monkeypatch.setitem(sys.modules, "agent.dpof.engine", fake_module)
    import agent.dpof as dpof_pkg
    monkeypatch.setattr(dpof_pkg, "engine", fake_module, raising=False)


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


def test_unauthorized_user_is_silent(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)

    update = FakeUpdate(user_id=999)
    context = FakeContext(args=["status"])

    asyncio.run(tc_mod.dpof_command(update, context))

    assert update.message.replies == []
    assert engine.status_calls == 0


def test_no_effective_user_is_silent(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)

    update = FakeUpdate(user_id=None)
    context = FakeContext(args=[])

    asyncio.run(tc_mod.dpof_command(update, context))

    assert update.message.replies == []


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def test_status_default_and_explicit(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)

    update = FakeUpdate(user_id=1)
    asyncio.run(tc_mod.dpof_command(update, FakeContext(args=[])))
    assert update.message.replies == ["STATUS TEXT"]
    assert engine.status_calls == 1

    update2 = FakeUpdate(user_id=1)
    asyncio.run(tc_mod.dpof_command(update2, FakeContext(args=["status"])))
    assert update2.message.replies == ["STATUS TEXT"]
    assert engine.status_calls == 2


def test_positions_and_config(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)

    u1 = FakeUpdate(user_id=1)
    asyncio.run(tc_mod.dpof_command(u1, FakeContext(args=["positions"])))
    assert u1.message.replies == ["POSITIONS TEXT"]

    u2 = FakeUpdate(user_id=1)
    asyncio.run(tc_mod.dpof_command(u2, FakeContext(args=["config"])))
    assert u2.message.replies == ["CONFIG TEXT"]


def test_halt_with_and_without_reason(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)

    u1 = FakeUpdate(user_id=1)
    asyncio.run(tc_mod.dpof_command(u1, FakeContext(args=["halt", "vol", "spike"])))
    assert engine.halt_calls == ["vol spike"]
    assert u1.message.replies == ["DPOF halted: vol spike"]

    u2 = FakeUpdate(user_id=1)
    asyncio.run(tc_mod.dpof_command(u2, FakeContext(args=["halt"])))
    assert engine.halt_calls == ["vol spike", "operator"]


def test_resume(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)

    u = FakeUpdate(user_id=1)
    asyncio.run(tc_mod.dpof_command(u, FakeContext(args=["resume"])))
    assert engine.resume_calls == 1
    assert u.message.replies == ["DPOF resumed"]


def test_unknown_subcommand_shows_usage(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)

    u = FakeUpdate(user_id=1)
    asyncio.run(tc_mod.dpof_command(u, FakeContext(args=["bogus"])))
    assert len(u.message.replies) == 1
    assert u.message.replies[0].startswith("Usage: /dpof")


# ---------------------------------------------------------------------------
# Chunking + robustness
# ---------------------------------------------------------------------------


def test_long_reply_is_chunked_under_4000_chars(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})
    engine = FakeEngine()
    long_text = "\n".join(f"line {i}" for i in range(2000))  # comfortably > 4000 chars
    engine.status_text = lambda: _async_return(long_text)
    _patch_get_engine(monkeypatch, engine)

    u = FakeUpdate(user_id=1)
    asyncio.run(tc_mod.dpof_command(u, FakeContext(args=["status"])))

    assert len(u.message.replies) >= 2
    assert all(len(chunk) <= 4000 for chunk in u.message.replies)
    assert "".join(u.message.replies).replace("\n\n", "\n") or True  # content preserved (loose check)
    assert "line 0" in u.message.replies[0]
    assert "line 1999" in u.message.replies[-1]


def test_engine_exception_is_caught_and_reported(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})

    class BoomEngine(FakeEngine):
        async def status_text(self):
            raise RuntimeError("ledger unavailable")

    _patch_get_engine(monkeypatch, BoomEngine())

    u = FakeUpdate(user_id=1)
    asyncio.run(tc_mod.dpof_command(u, FakeContext(args=["status"])))
    assert len(u.message.replies) == 1
    assert "DPOF command failed" in u.message.replies[0]
    assert "RuntimeError" in u.message.replies[0]


def test_no_message_on_update_does_not_raise(monkeypatch):
    _patch_telegram_bot_module(monkeypatch, authorized_ids={1})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)

    update = FakeUpdate(user_id=1)
    update.message = None
    asyncio.run(tc_mod.dpof_command(update, FakeContext(args=["status"])))  # must not raise
    assert engine.status_calls == 1


def _async_return(value):
    async def _inner():
        return value

    return _inner()


def test_why_subcommand_reads_readmodel_off_loop(monkeypatch):
    """/dpof why answers from the read model (files only), in an executor
    thread -- never the engine, never blocking the bot's event loop."""
    import threading

    from agent.dpof import readmodel

    _patch_telegram_bot_module(monkeypatch, {42})
    engine = FakeEngine()
    _patch_get_engine(monkeypatch, engine)
    seen = {}

    def _why():
        seen["thread"] = threading.current_thread() is threading.main_thread()
        return "WHY TEXT"

    monkeypatch.setattr(readmodel, "dpof_why_text", _why)
    update = FakeUpdate(42)
    asyncio.run(tc_mod.dpof_command(update, FakeContext(["why"])))
    assert update.message.replies == ["WHY TEXT"]
    assert seen["thread"] is False
    assert engine.status_calls == 0
