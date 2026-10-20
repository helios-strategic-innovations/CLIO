"""
B2a: bridge-gated Hydrinia reads inside run_self_check() and the CLIO-native
DPOF line replacing the Hydrinia status line in run_system_health_report().

Hydrinia (/home/agent/tradingbot) is permanently off (2026-09-24 operator
direction) — a stale EMERGENCY_STOP file or unprocessed directive_responses/
left behind on disk must not keep re-appearing in alerts once the bridge is
off, and the hourly health report must describe CLIO's own DPOF book instead
of a Hydrinia PID/RUNNING/STOPPED line.
"""

import asyncio
import json
import os

import pytest

from agent.integrations import scheduled_tasks as st


class FakeBot:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text))


class FakeContext:
    def __init__(self, bot):
        self.bot = bot


class _FakeMeta:
    def get_counts(self):
        return {"corrections": 0, "gaps": 0}


class FakeMemoryEngine:
    """Stub so self-check / health-report memory sections don't touch real data/."""

    meta = _FakeMeta()

    def health_check(self):
        return {
            "overall_healthy": True,
            "working": {"total": 0},
            "episodic": {"stale": False},
            "procedural": {"patterns": 0, "any_triggered": False},
            "meta": {"corrections": 0, "gaps": 0},
        }


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof"))
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    monkeypatch.setattr(st, "_get_chat_ids", lambda: ["test_chat"])
    monkeypatch.setattr(st, "_last_self_check_state", {})
    monkeypatch.setattr(
        "agent.memory.memory_engine.get_memory_engine", lambda: FakeMemoryEngine()
    )
    # No urgent bridge files / directive responses dirs by default.
    monkeypatch.setattr(
        st.config, "tradingbot_dir", str(tmp_path / "tradingbot"), raising=False
    )


def _run(coro):
    return asyncio.run(coro)


def _write_dpof_fixture(data_dir: str, *, open_spy=True, closed_qqq_pnl=None):
    os.makedirs(data_dir, exist_ok=True)
    positions = []
    if open_spy:
        positions.append({
            "id": "p_open_spy",
            "underlying": "SPY",
            "direction": "bullish",
            "structure": "single",
            "legs": [{"symbol": "SPY261002C00580000", "side": "buy"}],
            "status": "open",
            "qty_original": 1,
            "qty_open": 1,
            "entry_price": 2.5,
        })
    if closed_qqq_pnl is not None:
        import time
        positions.append({
            "id": "p_closed_qqq",
            "underlying": "QQQ",
            "direction": "bearish",
            "structure": "single",
            "legs": [{"symbol": "QQQ261002P00500000", "side": "buy"}],
            "status": "closed",
            "qty_original": 1,
            "qty_open": 0,
            "realized_pnl": closed_qqq_pnl,
            "closed_time": time.time(),
        })
    with open(os.path.join(data_dir, "positions.json"), "w") as f:
        json.dump({"positions": positions}, f)
    with open(os.path.join(data_dir, "risk_state.json"), "w") as f:
        json.dump({"day": "2026-09-24", "realized_pnl_today": closed_qqq_pnl or 0.0}, f)


# ---------------------------------------------------------------------------
# run_self_check()
# ---------------------------------------------------------------------------


def test_self_check_stale_estop_silent_when_bridge_off(monkeypatch, tmp_path):
    estop = tmp_path / "EMERGENCY_STOP"
    estop.write_text('{"reason": "stale"}')
    monkeypatch.setattr(st, "_ESTOP_FILE", str(estop))

    bot = FakeBot()
    ctx = FakeContext(bot)
    _run(st.run_self_check(ctx))

    assert bot.sent == [], "a stale Hydrinia EMERGENCY_STOP must not alert with the bridge off"


def test_self_check_estop_alerts_when_bridge_on(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    estop = tmp_path / "EMERGENCY_STOP"
    estop.write_text('{"reason": "circuit breaker"}')
    monkeypatch.setattr(st, "_ESTOP_FILE", str(estop))

    bot = FakeBot()
    ctx = FakeContext(bot)
    _run(st.run_self_check(ctx))

    assert len(bot.sent) == 1
    assert "EMERGENCY STOP is active" in bot.sent[0][1]


def test_self_check_directive_responses_ignored_when_bridge_off(monkeypatch, tmp_path, caplog):
    import logging
    responses_dir = tmp_path / "tradingbot" / "moltbot_knowledge" / "directive_responses"
    responses_dir.mkdir(parents=True)
    (responses_dir / "resp1.json").write_text("{}")

    caplog.set_level(logging.INFO, logger="openclaw.scheduled")
    bot = FakeBot()
    ctx = FakeContext(bot)
    _run(st.run_self_check(ctx))

    assert not any("unprocessed directive response" in r.message for r in caplog.records)


def test_self_check_directive_responses_scanned_when_bridge_on(monkeypatch, tmp_path, caplog):
    import logging
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    responses_dir = tmp_path / "tradingbot" / "moltbot_knowledge" / "directive_responses"
    responses_dir.mkdir(parents=True)
    (responses_dir / "resp1.json").write_text("{}")

    caplog.set_level(logging.INFO, logger="openclaw.scheduled")
    bot = FakeBot()
    ctx = FakeContext(bot)
    _run(st.run_self_check(ctx))

    assert any("unprocessed directive response" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# run_system_health_report()
# ---------------------------------------------------------------------------


def test_health_report_shows_dpof_not_hydrinia(monkeypatch, tmp_path):
    dpof_dir = str(tmp_path / "dpof")
    _write_dpof_fixture(dpof_dir, open_spy=True, closed_qqq_pnl=42.5)

    bot = FakeBot()
    ctx = FakeContext(bot)
    _run(st.run_system_health_report(ctx))

    assert len(bot.sent) == 1
    text = bot.sent[0][1]
    assert "DPOF:" in text
    assert "open 1" in text
    assert "closed today 1" in text
    assert "realized today $+42.50" in text
    # The old Hydrinia PID/RUNNING/STOPPED line must be gone.
    assert "Hydrinia: RUNNING" not in text
    assert "Hydrinia: STOPPED" not in text


def test_health_report_dpof_flat_book(monkeypatch, tmp_path):
    dpof_dir = str(tmp_path / "dpof")
    _write_dpof_fixture(dpof_dir, open_spy=False, closed_qqq_pnl=None)

    bot = FakeBot()
    ctx = FakeContext(bot)
    _run(st.run_system_health_report(ctx))

    text = bot.sent[0][1]
    assert "DPOF: mode=" in text
    assert "open 0" in text
    assert "HALT file: no" in text


def test_health_report_clio_idea_outcomes_hidden_when_bridge_off(monkeypatch, tmp_path):
    ideas_dir = tmp_path / "tradingbot" / "moltbot_knowledge" / "trade_ideas"
    ideas_dir.mkdir(parents=True)
    (ideas_dir / "ti_1.json").write_text(json.dumps({
        "ticker": "SPY", "direction": "LONG", "confidence": 0.7,
        "consumed_by_hydrinia": True, "hydrinia_direction_agreement": "agreed",
        "hydrinia_result": "win",
    }))
    _write_dpof_fixture(str(tmp_path / "dpof"))

    bot = FakeBot()
    ctx = FakeContext(bot)
    _run(st.run_system_health_report(ctx))

    text = bot.sent[0][1]
    assert "CLIO Ideas" not in text
