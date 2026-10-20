"""
Tests for agent/integrations/telegram_bot.py Hydrinia-bridge gating (B1 scope).

Hydrinia is permanently off (operator direction, 2026-09-24):
- /hydrinia, /logs, /stop, /botstart are registered as Telegram commands, and
  listed in the /start help text, ONLY when the bridge is explicitly
  re-enabled.
- /portfolio is CLIO-native (Alpaca + DPOF), not Hydrinia-specific, and is
  always registered.
- /trading mode text references DPOF, not hydrinia.py.

No network: create_bot() is exercised with a syntactically-valid-but-fake bot
token (never used to make a real Telegram API call by .build()/add_handler()).
"""

import asyncio

import pytest

import agent.config as config_mod
from agent.integrations import telegram_bot as tb


def _run(coro):
    return asyncio.run(coro)


class _FakeUser:
    def __init__(self, uid):
        self.id = uid


class _FakeChat:
    def __init__(self, cid):
        self.id = cid

    async def send_action(self, action):
        pass


class _FakeMessage:
    def __init__(self):
        self.sent: list[str] = []

    async def reply_text(self, text, **kwargs):
        self.sent.append(text)


class _FakeUpdate:
    def __init__(self, uid):
        self.effective_user = _FakeUser(uid)
        self.effective_chat = _FakeChat(uid)
        self.message = _FakeMessage()


AUTHORIZED_UID = 100000001


@pytest.fixture(autouse=True)
def _authorized_user(monkeypatch):
    monkeypatch.setattr(config_mod.config, "telegram_authorized_users", [AUTHORIZED_UID])


# ---------------------------------------------------------------------------
# /start help text
# ---------------------------------------------------------------------------

class TestStartHelpText:
    def test_hydrinia_commands_omitted_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        update = _FakeUpdate(AUTHORIZED_UID)
        _run(tb.cmd_start(update, None))
        text = update.message.sent[0]
        assert "/hydrinia" not in text
        assert "/logs" not in text
        assert "/stop —" not in text
        assert "/botstart" not in text
        assert "/portfolio" in text
        assert "/dpof" in text

    def test_hydrinia_commands_listed_when_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        update = _FakeUpdate(AUTHORIZED_UID)
        _run(tb.cmd_start(update, None))
        text = update.message.sent[0]
        assert "/hydrinia" in text
        assert "/logs" in text
        assert "/stop" in text
        assert "/botstart" in text
        assert "/portfolio" in text

    def test_unauthorized_user_gets_no_help_text(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        update = _FakeUpdate(999)
        _run(tb.cmd_start(update, None))
        assert update.message.sent == ["Unauthorized."]


# ---------------------------------------------------------------------------
# /trading mode text
# ---------------------------------------------------------------------------

class TestTradingModeText:
    def test_mentions_dpof_not_hydrinia_py(self):
        update = _FakeUpdate(AUTHORIZED_UID)
        _run(tb.cmd_trading(update, None))
        text = update.message.sent[0]
        assert "DPOF" in text
        assert "hydrinia.py" not in text


# ---------------------------------------------------------------------------
# /portfolio shortcut — CLIO-native instruction
# ---------------------------------------------------------------------------

class TestPortfolioShortcut:
    def test_portfolio_instruction_is_clio_native(self, monkeypatch):
        captured = {}

        async def _fake_agent_respond(**kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(tb, "agent_respond", _fake_agent_respond)
        update = _FakeUpdate(AUTHORIZED_UID)
        _run(tb.cmd_portfolio(update, None))
        prompt = captured.get("user_message", "")
        assert "DPOF" in prompt
        assert "Hydrinia" not in prompt
        assert "Alpaca" in prompt


# ---------------------------------------------------------------------------
# create_bot() handler registration
# ---------------------------------------------------------------------------

def _registered_commands(app) -> set:
    names = set()
    for handlers in app.handlers.values():
        for h in handlers:
            names |= set(getattr(h, "commands", None) or [])
    return names


class TestCreateBotRegistration:
    def test_hydrinia_commands_not_registered_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        monkeypatch.setattr(config_mod.config, "telegram_bot_token", "123456:FAKE-TOKEN-FOR-TESTS")
        app = tb.create_bot()
        names = _registered_commands(app)
        assert "hydrinia" not in names
        assert "logs" not in names
        assert "stop" not in names
        assert "botstart" not in names
        # portfolio and dpof are CLIO-native / DPOF-native — always registered
        assert "portfolio" in names

    def test_hydrinia_commands_registered_when_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        monkeypatch.setattr(config_mod.config, "telegram_bot_token", "123456:FAKE-TOKEN-FOR-TESTS")
        app = tb.create_bot()
        names = _registered_commands(app)
        assert "hydrinia" in names
        assert "logs" in names
        assert "stop" in names
        assert "botstart" in names
