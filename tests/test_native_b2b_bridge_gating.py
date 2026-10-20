"""
B2b: gap_signal_bridge.py, book_pipeline.py, trade_feedback.py, news_intel.py --
every remaining Hydrinia bridge read/write gated on hydrinia_bridge.bridge_enabled()
(default False; Hydrinia is permanently off, operator direction 2026-09-24).

No network. All file I/O goes through tmp_path. Never writes into the real data/ tree
or the real /home/agent/tradingbot mirror paths.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from agent.integrations import gap_signal_bridge as gsb
from agent.integrations import book_pipeline as bp
from agent.integrations import trade_feedback as tf
from agent.integrations import news_intel as ni

_NY = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# gap_signal_bridge.py
# ---------------------------------------------------------------------------


class TestGapSignalBridgeGating:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        self.signals_dir = tmp_path / "gap_signals"
        self.newsflash_dir = tmp_path / "newsflash"
        monkeypatch.setattr(gsb, "_GAP_SIGNALS_DIR", str(self.signals_dir))
        monkeypatch.setattr(gsb, "_NEWSFLASH_DIR", str(self.newsflash_dir))
        monkeypatch.setenv("MOLTBOT_GAP_BRIDGE_ENABLED", "1")
        self.signals_dir.mkdir(parents=True)
        (self.signals_dir / "stress_summary.json").write_text(json.dumps({
            "timestamp": datetime.now(_NY).isoformat(),
            "portfolio": {"num_positions": 3, "total_equity": 100000.0},
        }))
        (self.signals_dir / "drawdown_state.json").write_text(json.dumps({
            "timestamp": datetime.now(_NY).isoformat(),
            "tier": "CAUTION",
        }))

    def test_all_accessors_none_when_hydrinia_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        b = gsb.GapSignalBridge()
        assert b.get_stress_summary() is None
        assert b.get_drawdown_state() is None
        assert b.get_current_tier() == "NORMAL"
        assert b.get_min_conviction() == 0.30
        assert b.is_leveraged_etf_allowed() is True
        assert b.get_concentration_flags() == []
        assert b.get_analysis_context_block() == ""

    def test_accessors_populated_when_hydrinia_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        monkeypatch.setenv("MOLTBOT_GAP_BRIDGE_LOG_ONLY", "0")
        b = gsb.GapSignalBridge()
        assert b.get_stress_summary() is not None
        dd = b.get_drawdown_state()
        assert dd is not None
        assert b.get_current_tier() == "CAUTION"

    def test_read_cached_never_raises_on_bridge_check_failure(self, monkeypatch):
        def boom():
            raise RuntimeError("boom")

        monkeypatch.setattr("agent.hydrinia_bridge.bridge_enabled", boom)
        b = gsb.GapSignalBridge()
        assert b.get_stress_summary() is None

    def test_staleness_flagged_regardless_of_hour(self, monkeypatch):
        """S205-B2b fix: staleness used to only be checked during 8:00-20:00 ET
        (self._is_market_hours()). Now it is checked unconditionally."""
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        old_ts = (datetime.now(_NY) - timedelta(hours=5)).isoformat()
        (self.signals_dir / "drawdown_state.json").write_text(json.dumps({
            "timestamp": old_ts,
            "tier": "NORMAL",
            "freshness_ttl_seconds": 60,  # 60s TTL, data is 5h old -> always stale
        }))
        b = gsb.GapSignalBridge()
        dd = b.get_drawdown_state()
        assert dd is not None
        assert dd.get("_is_stale") is True

    def test_fresh_data_not_flagged_stale(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        b = gsb.GapSignalBridge()
        dd = b.get_drawdown_state()
        assert dd is not None
        assert "_is_stale" not in dd

    def test_is_market_hours_helper_removed(self):
        assert not hasattr(gsb.GapSignalBridge, "_is_market_hours")


# ---------------------------------------------------------------------------
# book_pipeline.py
# ---------------------------------------------------------------------------

_BOOK = {
    "id": "test_book", "title": "Test Book", "author": "Test Author", "tier": "A",
    "knowledge_category": "trading_strategies", "key_themes": ["risk", "sizing", "timing"],
}

_ANALYSIS = {
    "summary": "A test summary.",
    "mental_models": ["model one"],
    "trading_heuristics": [{"trigger": "VIX>30", "action": "reduce size", "rationale": "vol"}],
    "trading_adaptations": ["Reduce DPOF size when VIX>30"],
    "cross_references": [],
    "quality_score": 8,
}


class TestBookPipelineNativeStorage:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        self.native_dir = tmp_path / "clio_books"
        self.mirror_dir = tmp_path / "hydrinia_mirror_books"
        monkeypatch.setattr(bp, "BOOK_BASE_DIR", str(self.native_dir))
        monkeypatch.setattr(bp, "SYNTHESIS_DIR", str(self.native_dir / "synthesis"))
        monkeypatch.setattr(bp, "_HYDRINIA_BOOK_MIRROR_DIR", str(self.mirror_dir))

        async def _fake_save_knowledge(payload):
            return "ok"

        monkeypatch.setattr(bp, "handle_save_knowledge", _fake_save_knowledge, raising=False)
        # _save_book_analysis imports handle_save_knowledge lazily from knowledge_tools;
        # patch it at the source module too.
        import agent.tools.knowledge_tools as kt
        async def _fake(payload):
            return "ok"
        monkeypatch.setattr(kt, "handle_save_knowledge", _fake)

    def test_clio_native_dir_is_not_under_tradingbot(self):
        assert "tradingbot" not in bp.BOOK_BASE_DIR

    def test_clio_copy_written_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        asyncio.run(bp._save_book_analysis(dict(_ANALYSIS), dict(_BOOK), ["http://x"]))
        book_file = self.native_dir / "tier_a_trading" / "test_book.json"
        assert book_file.exists()
        data = json.loads(book_file.read_text())
        assert data["analyses"][0]["analysis"]["trading_adaptations"] == _ANALYSIS["trading_adaptations"]

    def test_mirror_not_written_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        asyncio.run(bp._save_book_analysis(dict(_ANALYSIS), dict(_BOOK), ["http://x"]))
        assert not self.mirror_dir.exists()

    def test_mirror_written_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        asyncio.run(bp._save_book_analysis(dict(_ANALYSIS), dict(_BOOK), ["http://x"]))
        mirror_file = self.mirror_dir / "tier_a_trading" / "test_book.json"
        assert mirror_file.exists()
        # CLIO's own copy still written unconditionally
        assert (self.native_dir / "tier_a_trading" / "test_book.json").exists()

    def test_old_hydrinia_adaptations_key_still_parses(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        old_style = dict(_ANALYSIS)
        del old_style["trading_adaptations"]
        old_style["hydrinia_adaptations"] = ["Old-style adaptation"]
        asyncio.run(bp._save_book_analysis(old_style, dict(_BOOK), ["http://x"]))
        book_file = self.native_dir / "tier_a_trading" / "test_book.json"
        data = json.loads(book_file.read_text())
        stored = data["analyses"][0]["analysis"]
        assert stored.get("hydrinia_adaptations") == ["Old-style adaptation"]

    def test_low_quality_skipped_entirely(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        low = dict(_ANALYSIS)
        low["quality_score"] = 2
        asyncio.run(bp._save_book_analysis(low, dict(_BOOK), ["http://x"]))
        book_file = self.native_dir / "tier_a_trading" / "test_book.json"
        assert not book_file.exists()

    def test_prompts_reference_clio_dpof_not_hydrinia(self):
        assert "trading_adaptations" in bp.BOOK_ANALYSIS_PROMPT
        assert "CLIO's own DPOF" in bp.BOOK_ANALYSIS_PROMPT
        assert "CLIO's own trading engine" in bp.BOOK_SYNTHESIS_PROMPT
        assert "CLIO Trading Adaptations" in bp.BOOK_SYNTHESIS_PROMPT


# ---------------------------------------------------------------------------
# trade_feedback.py — early-return no-op when bridge disabled
# ---------------------------------------------------------------------------


class TestTradeFeedbackGate:
    def test_noop_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

        def boom():
            raise AssertionError("_load_positions must not be called when bridge is off")

        monkeypatch.setattr(tf, "_load_positions", boom)
        asyncio.run(tf.analyze_trade_feedback(context=None))  # must not raise

    def test_proceeds_when_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        calls = {"n": 0}

        def fake_load_positions():
            calls["n"] += 1
            return {}  # empty -> analyze_trade_feedback logs-and-returns right after

        monkeypatch.setattr(tf, "_load_positions", fake_load_positions)
        asyncio.run(tf.analyze_trade_feedback(context=None))
        assert calls["n"] == 1

    def test_bridge_check_failure_is_a_noop_not_a_crash(self, monkeypatch):
        def boom():
            raise RuntimeError("boom")

        monkeypatch.setattr("agent.hydrinia_bridge.bridge_enabled", boom)
        asyncio.run(tf.analyze_trade_feedback(context=None))  # must not raise


# ---------------------------------------------------------------------------
# news_intel.py — bridge write gated
# ---------------------------------------------------------------------------


class TestNewsIntelBridgeGate:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ni, "BRIDGE_DIR", str(tmp_path / "bridge_intel"))

        async def _fake_save_knowledge(payload):
            return "ok"

        import agent.tools.knowledge_tools as kt
        monkeypatch.setattr(kt, "handle_save_knowledge", _fake_save_knowledge)
        self.bridge_dir = tmp_path / "bridge_intel"

    def test_no_bridge_write_when_disabled(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        asyncio.run(ni._deliver_intel("fed", "Some real analysis text.", ["http://x"]))
        assert not self.bridge_dir.exists()

    def test_bridge_write_when_enabled(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        asyncio.run(ni._deliver_intel("fed", "Some real analysis text.", ["http://x"]))
        files = list(self.bridge_dir.glob("*_fed.json"))
        assert len(files) == 1

    def test_null_intel_still_rejected_before_bridge_check(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        asyncio.run(ni._deliver_intel("fed", "SKIP\n\nNo new info.", ["http://x"]))
        assert not self.bridge_dir.exists()

    def test_bridge_check_failure_disables_write_not_crash(self, monkeypatch):
        def boom():
            raise RuntimeError("boom")

        monkeypatch.setattr("agent.hydrinia_bridge.bridge_enabled", boom)
        asyncio.run(ni._deliver_intel("fed", "Some real analysis text.", ["http://x"]))
        assert not self.bridge_dir.exists()
