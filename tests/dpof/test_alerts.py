"""Tests for agent.dpof.alerts."""

import asyncio

from agent.dpof.alerts import (
    DPOFAlerts,
    fmt_entry,
    fmt_exit,
    fmt_gates,
    fmt_model_down,
    fmt_model_recovered,
    fmt_stale,
)
from agent.dpof.models import Direction, JevVerdict, PositionRecord, Signal


def _pos(**overrides) -> PositionRecord:
    kwargs = dict(
        id="p1", underlying="SPY", direction="bullish", structure="single",
        legs=[{"symbol": "SPY261016C00500000", "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}],
        status="open", qty_original=2, qty_open=2, entry_price=2.0, mode="paper",
    )
    kwargs.update(overrides)
    return PositionRecord(**kwargs)


def _signal(**overrides) -> Signal:
    kwargs = dict(symbol="SPY", direction=Direction.BULLISH, composite=0.55, conviction=1.0, strength="strong")
    kwargs.update(overrides)
    return Signal(**kwargs)


# ---------------------------------------------------------------------------
# DPOFAlerts.send
# ---------------------------------------------------------------------------


def test_send_no_bot_logs_and_returns_false_without_consuming_dedupe():
    sent = []

    async def sender(bot, text):
        sent.append(text)

    clock_box = {"t": 0.0}
    alerts = DPOFAlerts(sender=sender, clock=lambda: clock_box["t"])

    async def _t():
        ok = await alerts.send(None, "hello", dedupe_key="k")
        assert ok is False
        assert sent == []
        # bot now present -- since bot=None never consumed the dedupe window, this must send.
        ok2 = await alerts.send(object(), "hello", dedupe_key="k")
        assert ok2 is True
        assert sent == ["hello"]

    asyncio.run(_t())


def test_send_dedupes_within_window_and_reopens_after():
    sent = []

    async def sender(bot, text):
        sent.append(text)

    clock_box = {"t": 1000.0}
    alerts = DPOFAlerts(sender=sender, clock=lambda: clock_box["t"])

    async def _t():
        assert await alerts.send(object(), "a", dedupe_key="k", dedupe_seconds=60) is True
        clock_box["t"] += 30
        assert await alerts.send(object(), "b", dedupe_key="k", dedupe_seconds=60) is False
        clock_box["t"] += 31  # now 61s after the first send
        assert await alerts.send(object(), "c", dedupe_key="k", dedupe_seconds=60) is True
        assert sent == ["a", "c"]

    asyncio.run(_t())


def test_send_without_dedupe_key_always_sends():
    sent = []

    async def sender(bot, text):
        sent.append(text)

    alerts = DPOFAlerts(sender=sender, clock=lambda: 0.0)

    async def _t():
        assert await alerts.send(object(), "x") is True
        assert await alerts.send(object(), "x") is True
        assert sent == ["x", "x"]

    asyncio.run(_t())


def test_send_sender_exception_is_caught_and_returns_false():
    async def bad_sender(bot, text):
        raise RuntimeError("telegram is down")

    alerts = DPOFAlerts(sender=bad_sender, clock=lambda: 0.0)

    async def _t():
        ok = await alerts.send(object(), "x", dedupe_key="k")
        assert ok is False

    asyncio.run(_t())


def test_default_sender_lazy_imports_scheduled_tasks(monkeypatch):
    """The default sender must fan out via scheduled_tasks._get_chat_ids/_safe_send."""
    import agent.dpof.alerts as alerts_mod

    calls = []

    class FakeScheduledTasks:
        @staticmethod
        def _get_chat_ids():
            return ["111", "222"]

        @staticmethod
        async def _safe_send(bot, chat_id, **kwargs):
            calls.append((chat_id, kwargs.get("text")))
            return True

    import sys
    import agent.integrations as integrations_pkg

    monkeypatch.setitem(sys.modules, "agent.integrations.scheduled_tasks", FakeScheduledTasks)
    # Also patch the package attribute directly: if the real submodule was already
    # imported earlier in this test session, `from agent.integrations import
    # scheduled_tasks` resolves via getattr(agent.integrations, "scheduled_tasks")
    # rather than re-consulting sys.modules, so the sys.modules patch alone would
    # be silently ineffective depending on test order.
    monkeypatch.setattr(integrations_pkg, "scheduled_tasks", FakeScheduledTasks, raising=False)

    alerts = DPOFAlerts()  # uses the real default sender

    async def _t():
        ok = await alerts.send(object(), "hi", dedupe_key=None)
        assert ok is True

    asyncio.run(_t())
    assert calls == [("111", "hi"), ("222", "hi")]


# ---------------------------------------------------------------------------
# Formatting (plain text, mode tag first)
# ---------------------------------------------------------------------------


def test_fmt_entry_submitted_vs_log_only():
    verdict = JevVerdict(mode="off", called=False, allow=True)

    submitted = fmt_entry(_pos(), _signal(), verdict, "paper")
    assert submitted.startswith("[PAPER]")
    assert "ENTRY SUBMITTED" in submitted
    assert "SPY261016C00500000" in submitted

    would_enter = fmt_entry(_pos(exit_reason="log_only"), _signal(), verdict, "log_only")
    assert would_enter.startswith("[LOG_ONLY]")
    assert "would enter" in would_enter


def test_fmt_entry_includes_jev_notes_when_called():
    verdict = JevVerdict(mode="shadow", called=True, allow=True, veto_reasons=["would_veto:event_risk"])
    text = fmt_entry(_pos(), _signal(), verdict, "paper")
    assert "jev=shadow" in text
    assert "would_veto:event_risk" in text


def test_fmt_entry_handles_missing_composite():
    verdict = JevVerdict(mode="off", called=False, allow=True)
    text = fmt_entry(_pos(), _signal(composite=None), verdict, "paper")
    assert "composite=n/a" in text


def test_fmt_exit_uses_position_mode_and_no_parse_mode_markup():
    text = fmt_exit(_pos(mode="live"), "stop_loss", -180.0)
    assert text.startswith("[LIVE]")
    assert "stop_loss" in text
    assert "$-180.00" in text
    assert "<" not in text and ">" not in text  # never markup-shaped


def test_fmt_exit_handles_none_pnl():
    text = fmt_exit(_pos(mode="paper"), "max_hold", None)
    assert "n/a" in text


def test_fmt_model_down_includes_open_count_and_flatten_time():
    text = fmt_model_down(3, "15:50")
    assert "3 open position" in text
    assert "15:50 ET" in text
    assert "Opus 5.5" in text and "GPT-6 Sol" in text
    assert "no new entries" in text


def test_fmt_model_down_zero_open_positions():
    text = fmt_model_down(0, "15:50")
    assert "0 open position" in text


def test_fmt_model_recovered_text():
    text = fmt_model_recovered()
    assert "back online" in text


def test_fmt_gates_and_fmt_stale():
    gates_text = fmt_gates(["options_level_too_low:1<2", "keys_not_set"])
    assert gates_text.startswith("[LOG_ONLY]")
    assert "options_level_too_low:1<2" in gates_text

    stale_text = fmt_stale(45.3, ["primary_composite_unavailable", "gate_source_unavailable:econ_calendar"])
    assert stale_text.startswith("[DATA]")
    assert "45 min" in stale_text
    assert "primary_composite_unavailable" in stale_text


def test_fmt_gates_empty_reasons_does_not_crash():
    assert "unknown" in fmt_gates([])
