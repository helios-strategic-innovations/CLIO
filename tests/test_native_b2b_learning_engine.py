"""
B2b: agent/integrations/learning_engine.py repointed off frozen Hydrinia state
(Hydrinia permanently off, operator direction 2026-09-24) onto CLIO-native sources:
market_regime.py, agent/dpof/readmodel.py, and the hydrinia_bridge gate.

No network. DPOF fixtures use CLIO_DPOF_DATA_DIR (same pattern as other native-work
builders' tests). Never writes into the real data/ tree.
"""

import asyncio
import json
import os
import time

import pytest

from agent.integrations import learning_engine as le


def _write_dpof_positions(data_dir: str, positions: list[dict]) -> None:
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "positions.json"), "w") as f:
        json.dump({"positions": positions}, f)


# ---------------------------------------------------------------------------
# _get_current_market_regime() — shared helper for the two regime-aware sites
# ---------------------------------------------------------------------------


class TestGetCurrentMarketRegime:
    def test_returns_label_when_fresh(self, monkeypatch):
        monkeypatch.setattr(
            "agent.integrations.market_regime.get_market_regime",
            lambda **kw: {"regime": "BULL", "confidence": 0.9},
        )
        assert le._get_current_market_regime() == "BULL"

    def test_returns_none_by_default_when_unavailable(self, monkeypatch):
        monkeypatch.setattr("agent.integrations.market_regime.get_market_regime", lambda **kw: None)
        assert le._get_current_market_regime() is None

    def test_returns_given_default_when_unavailable(self, monkeypatch):
        monkeypatch.setattr("agent.integrations.market_regime.get_market_regime", lambda **kw: None)
        assert le._get_current_market_regime(default="CHOPPY") == "CHOPPY"

    def test_invalid_regime_label_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(
            "agent.integrations.market_regime.get_market_regime",
            lambda **kw: {"regime": "SIDEWAYS"},
        )
        assert le._get_current_market_regime(default="CHOPPY") == "CHOPPY"

    def test_import_error_never_raises(self, monkeypatch):
        def boom(**kw):
            raise RuntimeError("boom")

        monkeypatch.setattr("agent.integrations.market_regime.get_market_regime", boom)
        assert le._get_current_market_regime(default="BULL") == "BULL"


# ---------------------------------------------------------------------------
# _get_portfolio_symbols() — DPOF readmodel, not Hydrinia position_log.json
# ---------------------------------------------------------------------------


class TestGetPortfolioSymbols:
    def test_returns_open_underlyings_as_dicts(self, monkeypatch, tmp_path):
        d = str(tmp_path / "dpof")
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", d)
        _write_dpof_positions(d, [
            {"id": "p1", "underlying": "SPY", "direction": "bullish", "structure": "single",
             "legs": [], "status": "open", "qty_original": 1, "qty_open": 1},
            {"id": "p2", "underlying": "QQQ", "direction": "bearish", "structure": "single",
             "legs": [], "status": "open", "qty_original": 1, "qty_open": 1},
            {"id": "p3", "underlying": "SPY", "direction": "bullish", "structure": "single",
             "legs": [], "status": "closed", "qty_original": 1, "qty_open": 0,
             "realized_pnl": 10.0, "closed_time": time.time()},
        ])
        result = le._get_portfolio_symbols()
        symbols = sorted(p["symbol"] for p in result)
        assert symbols == ["QQQ", "SPY"]  # closed position excluded; dedup'd by underlying

    def test_no_open_positions_returns_empty_list(self, monkeypatch, tmp_path):
        d = str(tmp_path / "dpof")
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", d)
        _write_dpof_positions(d, [])
        assert le._get_portfolio_symbols() == []

    def test_never_raises_on_readmodel_failure(self, monkeypatch):
        def boom():
            raise RuntimeError("dpof unavailable")

        monkeypatch.setattr("agent.dpof.readmodel.open_underlyings", boom)
        assert le._get_portfolio_symbols() == []

    def test_get_open_tickers_uses_it(self, monkeypatch, tmp_path):
        d = str(tmp_path / "dpof")
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", d)
        _write_dpof_positions(d, [
            {"id": "p1", "underlying": "SPY", "direction": "bullish", "structure": "single",
             "legs": [], "status": "open", "qty_original": 1, "qty_open": 1},
        ])
        monkeypatch.setattr(le, "_ticker_cache", [])
        monkeypatch.setattr(le, "_ticker_cache_ts", 0.0)
        assert le._get_open_tickers() == ["SPY"]


# ---------------------------------------------------------------------------
# _load_feedback_hint() — DPOF closed positions, not hydrinia_feedback.json
# ---------------------------------------------------------------------------


class TestLoadFeedbackHint:
    def _closed(self, n_wins, n_losses, base_time):
        out = []
        for i in range(n_wins):
            out.append({"id": f"w{i}", "underlying": "SPY", "direction": "bullish",
                        "structure": "single", "legs": [], "status": "closed",
                        "qty_original": 1, "qty_open": 0, "realized_pnl": 50.0,
                        "closed_time": base_time + i})
        for i in range(n_losses):
            out.append({"id": f"l{i}", "underlying": "QQQ", "direction": "bearish",
                        "structure": "single", "legs": [], "status": "closed",
                        "qty_original": 1, "qty_open": 0, "realized_pnl": -20.0,
                        "closed_time": base_time + 1000 + i})
        return out

    def test_below_five_gate_returns_empty(self, monkeypatch, tmp_path):
        d = str(tmp_path / "dpof")
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", d)
        _write_dpof_positions(d, self._closed(2, 1, time.time()))
        assert le._load_feedback_hint() == ""

    def test_at_least_five_produces_win_rate_hint(self, monkeypatch, tmp_path):
        d = str(tmp_path / "dpof")
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", d)
        _write_dpof_positions(d, self._closed(3, 2, time.time()))
        hint = le._load_feedback_hint()
        assert "5 DPOF trades" in hint
        assert "60%" in hint  # 3/5 wins

    def test_no_closed_positions_returns_empty(self, monkeypatch, tmp_path):
        d = str(tmp_path / "dpof")
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", d)
        _write_dpof_positions(d, [])
        assert le._load_feedback_hint() == ""

    def test_caps_at_max_trades(self, monkeypatch, tmp_path):
        d = str(tmp_path / "dpof")
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", d)
        _write_dpof_positions(d, self._closed(60, 0, time.time()))
        hint = le._load_feedback_hint()
        assert f"{le._FEEDBACK_HINT_MAX_TRADES} DPOF trades" in hint

    def test_never_raises_on_readmodel_failure(self, monkeypatch):
        def boom():
            raise RuntimeError("boom")

        monkeypatch.setattr("agent.dpof.readmodel.load_positions", boom)
        assert le._load_feedback_hint() == ""


# ---------------------------------------------------------------------------
# _extract_hydrinia_adaptations() — new "For CLIO Trading" + old "For Hydrinia"
# ---------------------------------------------------------------------------


class TestExtractAdaptations:
    def test_new_heading_parses(self):
        text = (
            "## 7. Actionable Adaptations\n"
            "### For CLIO Trading\n"
            "1. Reduce DPOF size on VIX>30\n"
            "2. Add SPY hedge above 0.85 confidence\n"
            "### For Moltbot\n"
            "1. Research more macro\n"
            "## 8. Knowledge Gaps\n"
        )
        out = le._extract_hydrinia_adaptations(text)
        assert "Reduce DPOF size" in out
        assert "Research more macro" not in out  # stops at "### For Moltbot"

    def test_old_heading_still_parses(self):
        text = (
            "### For Hydrinia\n"
            "1. Old-style adaptation\n"
            "### For Moltbot\n"
            "1. Other\n"
        )
        out = le._extract_hydrinia_adaptations(text)
        assert "Old-style adaptation" in out

    def test_new_heading_wins_when_both_present_and_new_is_first(self):
        text = (
            "### For CLIO Trading\n"
            "1. New item\n"
            "### For Hydrinia\n"
            "1. Should not be reached (old heading treated as end-of-doc marker text)\n"
        )
        out = le._extract_hydrinia_adaptations(text)
        assert "New item" in out

    def test_neither_heading_present_returns_empty(self):
        assert le._extract_hydrinia_adaptations("no adaptations section here") == ""

    def test_new_items_tagged(self):
        text = "### For CLIO Trading\n1. Totally new adaptation about gamma exposure\n"
        out = le._extract_hydrinia_adaptations(text, previous_adaptations="")
        assert "[NEW]" in out

    def test_seen_items_not_tagged_new(self):
        text = "### For CLIO Trading\n1. Reduce size near FOMC\n"
        prev = "[P0] Reduce size near FOMC"
        out = le._extract_hydrinia_adaptations(text, previous_adaptations=prev)
        assert "[NEW]" not in out


# ---------------------------------------------------------------------------
# SYNTHESIS_PROMPT text — no more literal "Hydrinia items"/"Moltbot items" framing
# ---------------------------------------------------------------------------


def test_synthesis_prompt_reframed_to_clio_dpof():
    assert "CLIO" in le.SYNTHESIS_PROMPT
    assert "For CLIO Trading" in le.SYNTHESIS_PROMPT
    assert "max 10 Hydrinia items, max 5 Moltbot items" not in le.SYNTHESIS_PROMPT


# ---------------------------------------------------------------------------
# LearningModeScheduler.scan_for_surgical_events() — bridge gating + DPOF trigger
# ---------------------------------------------------------------------------


class TestScanForSurgicalEvents:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(le, "_BRIDGE_URGENT_DIR", str(tmp_path / "urgent"))
        monkeypatch.setattr(le, "_BRIDGE_TRADES_DIR", str(tmp_path / "trades"))
        monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof"))
        self.tmp_path = tmp_path

    def _write_urgent_file(self, name="NVDA_alert.md", content="Ticker: NVDA\nBreaking news"):
        d = self.tmp_path / "urgent"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(content)

    def test_hydrinia_dirs_ignored_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        self._write_urgent_file()
        _write_dpof_positions(str(self.tmp_path / "dpof"), [])
        sched = le.LearningModeScheduler()
        events = sched.scan_for_surgical_events()
        assert events == []

    def test_hydrinia_dirs_read_when_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        self._write_urgent_file()
        _write_dpof_positions(str(self.tmp_path / "dpof"), [])
        sched = le.LearningModeScheduler()
        events = sched.scan_for_surgical_events()
        assert any(e["ticker"] == "NVDA" and e["event_type"] == "urgent_bridge" for e in events)

    def test_dpof_opened_position_triggers_surgical_event(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        _write_dpof_positions(str(self.tmp_path / "dpof"), [
            {"id": "new1", "underlying": "SPY", "direction": "bullish", "structure": "single",
             "legs": [], "status": "open", "qty_original": 1, "qty_open": 1,
             "entry_time": time.time()},
        ])
        sched = le.LearningModeScheduler()
        events = sched.scan_for_surgical_events()
        assert len(events) == 1
        assert events[0]["ticker"] == "SPY"
        assert events[0]["event_type"] == "position_open"
        assert events[0]["source_file"] == "new1"

    def test_dpof_trigger_only_fires_once_per_position(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        _write_dpof_positions(str(self.tmp_path / "dpof"), [
            {"id": "new1", "underlying": "SPY", "direction": "bullish", "structure": "single",
             "legs": [], "status": "open", "qty_original": 1, "qty_open": 1,
             "entry_time": time.time()},
        ])
        sched = le.LearningModeScheduler()
        first = sched.scan_for_surgical_events()
        second = sched.scan_for_surgical_events()
        assert len(first) == 1
        assert second == []  # already seen -- not re-emitted

    def test_dpof_scan_failure_never_raises(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

        def boom(**kw):
            raise RuntimeError("boom")

        monkeypatch.setattr("agent.dpof.readmodel.opened_today", boom)
        sched = le.LearningModeScheduler()
        assert sched.scan_for_surgical_events() == []


# ---------------------------------------------------------------------------
# _save_trade_idea() — mirror write gated on bridge_enabled()
# ---------------------------------------------------------------------------


_VALID_IDEA = {
    "ticker": "AAPL", "direction": "LONG", "confidence": 0.55,
    "thesis": "test thesis", "entry_trigger": "break 200", "stop_loss": "190",
    "take_profit": "220", "timeframe": "1w",
}


class TestSaveTradeIdeaMirrorGate:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(le, "TRADE_IDEAS_DIR", str(tmp_path / "trade_ideas"))
        monkeypatch.setattr(le, "TRADE_IDEAS_V2_DIR", str(tmp_path / "trade_ideas_v2"))
        self.mirror_dir = tmp_path / "hydrinia_mirror" / "trade_ideas"
        self.mirror_v2_dir = tmp_path / "hydrinia_mirror" / "trade_ideas_v2"
        monkeypatch.setattr(le, "_HYDRINIA_TRADE_IDEAS_MIRROR_DIR", str(self.mirror_dir))
        monkeypatch.setattr(le, "_HYDRINIA_TRADE_IDEAS_V2_MIRROR_DIR", str(self.mirror_v2_dir))
        monkeypatch.setattr(
            "agent.integrations.market_regime.get_market_regime", lambda **kw: None
        )
        self.tmp_path = tmp_path

    def test_clio_copy_always_written_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        asyncio.run(le._save_trade_idea(dict(_VALID_IDEA), "equities_intraday", ["http://x"]))
        assert len(list((self.tmp_path / "trade_ideas").glob("ti_*.json"))) == 1
        assert len(list((self.tmp_path / "trade_ideas_v2").glob("*.json"))) == 1

    def test_mirror_not_written_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        asyncio.run(le._save_trade_idea(dict(_VALID_IDEA), "equities_intraday", ["http://x"]))
        assert not self.mirror_dir.exists()
        assert not self.mirror_v2_dir.exists()

    def test_mirror_written_when_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        asyncio.run(le._save_trade_idea(dict(_VALID_IDEA), "equities_intraday", ["http://x"]))
        assert len(list(self.mirror_dir.glob("ti_*.json"))) == 1
        assert len(list(self.mirror_v2_dir.glob("*.json"))) == 1
        # CLIO's own copy still written unconditionally
        assert len(list((self.tmp_path / "trade_ideas").glob("ti_*.json"))) == 1

    def test_consumed_by_hydrinia_field_kept_in_schema(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        asyncio.run(le._save_trade_idea(dict(_VALID_IDEA), "equities_intraday", ["http://x"]))
        files = list((self.tmp_path / "trade_ideas").glob("ti_*.json"))
        data = json.loads(files[0].read_text())
        assert data["consumed_by_hydrinia"] is False


# ---------------------------------------------------------------------------
# R-23: adaptive query prompt uses the real current month (frozen clock 2026-10-12)
# ---------------------------------------------------------------------------

from datetime import datetime as _real_datetime
from zoneinfo import ZoneInfo as _ZoneInfo

_FROZEN_ET = _real_datetime(2026, 10, 12, 10, 0, tzinfo=_ZoneInfo("America/New_York"))


class _FrozenDT(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        return _FROZEN_ET.astimezone(tz) if tz is not None else _FROZEN_ET.replace(tzinfo=None)


class _FakeMsgResp:
    def __init__(self, text):
        self.content = [type("B", (), {"text": text})()]


class _FakeClient:
    def __init__(self, text, sink):
        self._text = text
        self._sink = sink

    def with_options(self, **kw):
        return self

    @property
    def messages(self):
        outer = self

        class _M:
            def create(self_inner, **kw):
                outer._sink.append(kw)
                return _FakeMsgResp(outer._text)

        return _M()


def _patch_adaptive(monkeypatch, model_text: str):
    sent: list[dict] = []

    async def _ok(*a, **kw):
        return None

    monkeypatch.setattr(le, "datetime", _FrozenDT)
    monkeypatch.setattr("agent.rate_limiter.check_and_record", _ok)
    monkeypatch.setattr(le, "_get_client", lambda: _FakeClient(model_text, sent))
    monkeypatch.setattr(le, "_log_api_usage", lambda *a, **k: None)
    return sent


_CAT = {"name": "macro_fed", "knowledge_category": "trading_market_analysis", "queries": ["x"]}


class TestAdaptiveQueryCurrentMonth:
    def test_prompt_injects_current_month_not_march(self, monkeypatch):
        sent = _patch_adaptive(monkeypatch, "Fed rate outlook latest data")
        asyncio.run(le._generate_adaptive_query(_CAT, [], [], []))
        prompt = sent[0]["messages"][0]["content"]
        assert "October 2026" in prompt
        assert "March 2026" not in prompt

    def test_past_month_token_rewritten_to_current(self, monkeypatch):
        _patch_adaptive(monkeypatch, "March 2026 Fed outlook")
        out = asyncio.run(le._generate_adaptive_query(_CAT, [], [], []))
        assert out == "October 2026 Fed outlook"

    def test_future_month_token_becomes_latest_as_before(self, monkeypatch):
        # Original behaviour kept: a month that has not happened yet is never searched, it becomes "latest".
        _patch_adaptive(monkeypatch, "December 2026 CPI print preview")
        out = asyncio.run(le._generate_adaptive_query(_CAT, [], [], []))
        assert out == "latest CPI print preview"

    def test_next_year_month_token_becomes_latest(self, monkeypatch):
        _patch_adaptive(monkeypatch, "January 2027 rate outlook")
        out = asyncio.run(le._generate_adaptive_query(_CAT, [], [], []))
        assert out == "latest rate outlook"

    def test_past_year_token_rewritten_to_current(self, monkeypatch):
        _patch_adaptive(monkeypatch, "July 2025 earnings recap")
        out = asyncio.run(le._generate_adaptive_query(_CAT, [], [], []))
        assert out == "October 2026 earnings recap"

    def test_year_only_mention_untouched(self, monkeypatch):
        _patch_adaptive(monkeypatch, "Fed 2025 outlook for rate cuts")
        out = asyncio.run(le._generate_adaptive_query(_CAT, [], [], []))
        assert out == "Fed 2025 outlook for rate cuts"

    def test_current_month_token_unchanged(self, monkeypatch):
        _patch_adaptive(monkeypatch, "october 2026 Treasury auction results")
        out = asyncio.run(le._generate_adaptive_query(_CAT, [], [], []))
        assert out == "october 2026 Treasury auction results"
