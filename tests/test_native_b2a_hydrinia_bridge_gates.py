"""
B2a: defense-in-depth bridge_enabled() gates in directive_delivery.py,
directive_feedback.py, trade_idea_feedback.py, and brier_resolver.py.

Hydrinia (/home/agent/tradingbot) is permanently off. These modules only
read/write Hydrinia's bridge directories or state files, so each must no-op
when CLIO_HYDRINIA_BRIDGE_ENABLED is unset/0, and behave as before when it's
explicitly re-enabled.
"""

import asyncio
import json
import os

import pytest

from agent.integrations import (
    brier_resolver,
    directive_delivery,
    directive_feedback,
    trade_idea_feedback,
)


@pytest.fixture(autouse=True)
def _clean_bridge_env(monkeypatch):
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# directive_delivery.py
# ---------------------------------------------------------------------------


class FakeOverrideEvent:
    command_text = "sell SPY now"
    timestamp = "2026-09-24T12:00:00Z"


def test_dispatch_directive_noop_when_bridge_off(monkeypatch, tmp_path):
    directives_dir = tmp_path / "directives"
    monkeypatch.setattr(directive_delivery, "DIRECTIVES_DIR", str(directives_dir))

    result = _run(directive_delivery.dispatch_directive(FakeOverrideEvent(), "sell SPY now"))

    assert "bridge is off" in result.lower()
    assert not directives_dir.exists() or list(directives_dir.glob("*.json")) == []


def test_dispatch_directive_writes_when_bridge_on(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    directives_dir = tmp_path / "directives"
    monkeypatch.setattr(directive_delivery, "DIRECTIVES_DIR", str(directives_dir))
    monkeypatch.setattr(directive_delivery, "DIRECTIVES_LOG", str(tmp_path / "directives.jsonl"))

    result = _run(directive_delivery.dispatch_directive(FakeOverrideEvent(), "sell SPY now"))

    assert "Directive dispatched" in result
    assert list(directives_dir.glob("*.json"))


def test_get_active_directives_empty_when_bridge_off(monkeypatch, tmp_path):
    directives_dir = tmp_path / "directives"
    directives_dir.mkdir()
    (directives_dir / "DIR-1.json").write_text(json.dumps({
        "id": "DIR-1", "expires_at": "2099-01-01T00:00:00+00:00",
    }))
    monkeypatch.setattr(directive_delivery, "DIRECTIVES_DIR", str(directives_dir))
    monkeypatch.setattr(directive_delivery, "DIRECTIVE_RESPONSES_DIR", str(tmp_path / "resp"))

    assert directive_delivery.get_active_directives() == []


def test_get_active_directives_reads_when_bridge_on(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    directives_dir = tmp_path / "directives"
    directives_dir.mkdir()
    (directives_dir / "DIR-1.json").write_text(json.dumps({
        "id": "DIR-1", "expires_at": "2099-01-01T00:00:00+00:00",
    }))
    monkeypatch.setattr(directive_delivery, "DIRECTIVES_DIR", str(directives_dir))
    monkeypatch.setattr(directive_delivery, "DIRECTIVE_RESPONSES_DIR", str(tmp_path / "resp"))

    active = directive_delivery.get_active_directives()
    assert len(active) == 1
    assert active[0]["id"] == "DIR-1"


# ---------------------------------------------------------------------------
# directive_feedback.py
# ---------------------------------------------------------------------------


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text))


class FakeContext:
    def __init__(self, bot):
        self.bot = bot


def test_poll_directive_feedback_noop_when_bridge_off(monkeypatch, tmp_path):
    responses_dir = tmp_path / "responses"
    responses_dir.mkdir()
    (responses_dir / "resp1.json").write_text(json.dumps({
        "directive_id": "DIR-1", "trade_id": "t1", "summary": {"total": 1, "complied": 1, "dissented": 0},
    }))
    monkeypatch.setattr(directive_feedback, "RESPONSES_DIR", str(responses_dir))

    bot = FakeBot()
    _run(directive_feedback.poll_directive_feedback(FakeContext(bot)))

    assert bot.sent == []
    # File must not have been renamed/consumed either — a true no-op.
    assert (responses_dir / "resp1.json").exists()


# ---------------------------------------------------------------------------
# trade_idea_feedback.py
# ---------------------------------------------------------------------------


def test_process_trade_idea_feedback_noop_when_bridge_off(monkeypatch, tmp_path):
    debrief_dir = tmp_path / "trade_debriefs"
    debrief_dir.mkdir()
    (debrief_dir / "d1.json").write_text(json.dumps({"symbol": "SPY", "pnl": 10}))
    monkeypatch.setattr(trade_idea_feedback, "_DEBRIEF_DIR", str(debrief_dir))

    calls = []
    monkeypatch.setattr(trade_idea_feedback, "_load_state", lambda: calls.append(1))

    _run(trade_idea_feedback.process_trade_idea_feedback(context=None))

    assert calls == [], "must return before ever touching state when the bridge is off"


# ---------------------------------------------------------------------------
# brier_resolver.py
# ---------------------------------------------------------------------------


def test_run_brier_resolution_noop_when_bridge_off(monkeypatch, tmp_path):
    pred_log = tmp_path / "prediction_log.json"
    pred_log.write_text(json.dumps({"predictions": []}))
    monkeypatch.setattr(brier_resolver, "PREDICTION_LOG_PATH", str(pred_log))

    calls = []
    monkeypatch.setattr(
        brier_resolver, "_run_brier_resolution_sync", lambda: calls.append(1)
    )

    _run(brier_resolver.run_brier_resolution(context=None))

    assert calls == [], "must return before dispatching to the executor when the bridge is off"


def test_run_brier_resolution_runs_when_bridge_on(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    pred_log = tmp_path / "prediction_log.json"
    pred_log.write_text(json.dumps({"predictions": []}))
    monkeypatch.setattr(brier_resolver, "PREDICTION_LOG_PATH", str(pred_log))

    calls = []
    monkeypatch.setattr(
        brier_resolver, "_run_brier_resolution_sync", lambda: calls.append(1)
    )

    # Force a weekday so the function doesn't early-return on a weekend.
    import datetime as _dt
    from zoneinfo import ZoneInfo

    class _FixedDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 24, 16, 30, tzinfo=tz)  # Thursday

    monkeypatch.setattr(brier_resolver, "datetime", _FixedDatetime)

    _run(brier_resolver.run_brier_resolution(context=None))

    assert calls == [1]
