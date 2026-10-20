"""
S215 R-29: the learning log reports the REAL trade-idea outcome (saved or the reason a
gate rejected it), and per-gate counters persist as an additive learning-state field.

No network, no live state: every path (trade ideas, learning_state, knowledge index) is
redirected to tmp_path or replaced by an in-memory fake. Thresholds are not changed.
"""

import asyncio
import json
import logging

import pytest

from agent.integrations import learning_engine as le


_VALID_IDEA = {
    "ticker": "AAPL", "direction": "LONG", "confidence": 0.55,
    "thesis": "test thesis", "entry_trigger": "break 200", "stop_loss": "190",
    "take_profit": "220", "timeframe": "1w",
}


@pytest.fixture
def ti_env(monkeypatch, tmp_path):
    monkeypatch.setattr(le, "TRADE_IDEAS_DIR", str(tmp_path / "trade_ideas"))
    monkeypatch.setattr(le, "TRADE_IDEAS_V2_DIR", str(tmp_path / "trade_ideas_v2"))
    monkeypatch.setattr(le, "_HYDRINIA_TRADE_IDEAS_MIRROR_DIR", str(tmp_path / "mirror" / "ti"))
    monkeypatch.setattr(le, "_HYDRINIA_TRADE_IDEAS_V2_MIRROR_DIR", str(tmp_path / "mirror" / "tiv2"))
    monkeypatch.setattr(le, "LEARNING_STATE_FILE", str(tmp_path / "learning_state.json"))
    monkeypatch.setattr(le, "_GATE_COUNTS", {})
    monkeypatch.setattr(le, "_GATE_COUNTS_SEEDED", False)
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    monkeypatch.delenv("TRADE_IDEA_MIN_CONFIDENCE", raising=False)
    monkeypatch.setenv("MOLTBOT_COST_AWARE_ENABLED", "0")
    # Injected fakes: no regime file, no UW/gap/concentration reads, no live index.
    monkeypatch.setattr("agent.integrations.market_regime.get_market_regime", lambda **kw: None)
    monkeypatch.setattr("agent.integrations.uw_flow_intel.get_ticker_flow", lambda t: None)
    monkeypatch.setattr("agent.integrations.gap_signal_bridge.get_gap_bridge",
                        lambda: type("_G", (), {"get_newsflash_for_ticker": lambda self, t: None})())
    monkeypatch.setattr(le, "_check_gap_concentration_for_idea", lambda t, s: None)

    saved_calls = []

    async def _fake_save_knowledge(p):
        saved_calls.append(p)
        return "ok"

    monkeypatch.setattr("agent.tools.knowledge_tools.handle_save_knowledge", _fake_save_knowledge)
    monkeypatch.setattr("agent.tools.knowledge_tools.load_index", lambda: {"entries": []})
    monkeypatch.setattr("agent.integrations.source_authority.compute_authority_weighted_quality",
                        lambda q, urls: float(q))
    monkeypatch.setattr(le, "_check_content_duplicate", lambda a, urls, cat: (False, 0.0, None))
    return {"tmp": tmp_path, "saved_calls": saved_calls}


def _analysis(trade_idea=None, quality=9, actionability=9):
    return {
        "summary": "Summary sentence about SPY 5% move.", "key_insights": ["ins"],
        "actionable_items": ["act"], "themes": ["macro"], "quality_score": quality,
        "actionability_score": actionability, "contrarian_view": "none",
        "trade_idea": trade_idea,
    }


def _save_idea(idea):
    return asyncio.run(le._save_trade_idea(dict(idea), "equities_intraday", ["http://x"]))


def _run_analysis(analysis, category="equities_intraday", knowledge_category="trading_market_analysis"):
    return asyncio.run(le._save_analysis_to_knowledge(analysis, category, knowledge_category, ["http://x"]))


# ---------------------------------------------------------------------------
# _save_trade_idea reports why it rejected (return value)
# ---------------------------------------------------------------------------

class TestSaveTradeIdeaReturnsOutcome:
    def test_saved_returns_saved(self, ti_env):
        assert _save_idea(_VALID_IDEA) == "saved"
        assert len(list((ti_env["tmp"] / "trade_ideas").glob("ti_*.json"))) == 1

    def test_hold_is_non_directional(self, ti_env):
        assert _save_idea({**_VALID_IDEA, "direction": "HOLD"}) == "non_directional"
        assert not (ti_env["tmp"] / "trade_ideas").exists() or not list(
            (ti_env["tmp"] / "trade_ideas").glob("ti_*.json"))

    def test_unknown_direction(self, ti_env):
        assert _save_idea({**_VALID_IDEA, "direction": "SIDEWAYS"}) == "unknown_direction"

    def test_prohibited_ticker(self, ti_env):
        assert _save_idea({**_VALID_IDEA, "ticker": "GLD"}) == "prohibited_ticker"

    def test_confidence_floor(self, ti_env):
        assert _save_idea({**_VALID_IDEA, "confidence": 0.1}) == "confidence_floor"

    def test_vague_entry(self, ti_env):
        assert _save_idea({**_VALID_IDEA, "entry_trigger": "at current levels"}) == "vague_entry"

    def test_same_direction_dedup(self, ti_env):
        assert _save_idea(_VALID_IDEA) == "saved"
        assert _save_idea(_VALID_IDEA) == "dedup_same_direction"


# ---------------------------------------------------------------------------
# Log line and counters on the knowledge-save path
# ---------------------------------------------------------------------------

class TestLogReportsRealOutcome:
    def test_rejected_idea_is_not_logged_as_yes(self, ti_env, caplog):
        caplog.set_level(logging.INFO, logger=le.logger.name)
        assert _run_analysis(_analysis({**_VALID_IDEA, "direction": "HOLD"})) is True
        lines = [r.getMessage() for r in caplog.records if "Knowledge saved" in r.getMessage()]
        assert len(lines) == 1
        assert lines[0].endswith("trade_idea=rejected:non_directional")
        assert "trade_idea=yes" not in lines[0]
        assert "trade_idea=saved" not in lines[0]

    def test_saved_idea_logged_as_saved(self, ti_env, caplog):
        caplog.set_level(logging.INFO, logger=le.logger.name)
        assert _run_analysis(_analysis(_VALID_IDEA)) is True
        lines = [r.getMessage() for r in caplog.records if "Knowledge saved" in r.getMessage()]
        assert lines[0].endswith("trade_idea=saved")

    def test_no_idea_logged_as_none(self, ti_env, caplog):
        caplog.set_level(logging.INFO, logger=le.logger.name)
        assert _run_analysis(_analysis(None)) is True
        lines = [r.getMessage() for r in caplog.records if "Knowledge saved" in r.getMessage()]
        assert lines[0].endswith("trade_idea=none")

    def test_rejected_idea_still_saves_knowledge_entry(self, ti_env):
        _run_analysis(_analysis({**_VALID_IDEA, "ticker": "GLD"}))
        assert len(ti_env["saved_calls"]) == 1


class TestGateCounters:
    def test_quality_floor_counted_and_no_save(self, ti_env):
        assert _run_analysis(_analysis(_VALID_IDEA, quality=3, actionability=9)) is False
        assert le._gate_counts_snapshot() == {"quality_floor": 1}
        assert ti_env["saved_calls"] == []

    def test_combined_floor_counted(self, ti_env):
        assert _run_analysis(_analysis(_VALID_IDEA, quality=5, actionability=5)) is False
        assert le._gate_counts_snapshot() == {"combined_floor": 1}

    def test_content_dedup_counted(self, ti_env, monkeypatch):
        monkeypatch.setattr(le, "_check_content_duplicate", lambda a, urls, cat: (True, 0.99, "x"))
        assert _run_analysis(_analysis(_VALID_IDEA)) is False
        assert le._gate_counts_snapshot() == {"content_dedup": 1}

    def test_category_cap_counted(self, ti_env, monkeypatch):
        from datetime import datetime
        today = datetime.now(le._NY).strftime("%Y-%m-%d") + "T09:00:00"
        entries = [{"category": "trading_market_analysis", "created": today}] * 2
        monkeypatch.setattr("agent.tools.knowledge_tools.load_index", lambda: {"entries": entries})
        assert _run_analysis(_analysis(_VALID_IDEA), category="crypto") is False
        assert le._gate_counts_snapshot() == {"category_cap": 1}

    def test_trade_idea_outcomes_counted(self, ti_env):
        _run_analysis(_analysis({**_VALID_IDEA, "direction": "HOLD"}))
        _run_analysis(_analysis(_VALID_IDEA))
        assert le._gate_counts_snapshot() == {
            "ti_rejected:non_directional": 1, "ti_saved": 1,
        }

    def test_counts_persist_in_learning_state(self, ti_env):
        le._load_learning_state()  # the learning cycle always loads first (seeds the counters)
        _run_analysis(_analysis(_VALID_IDEA))
        le._save_learning_state({"cycle_count": 3})
        data = json.loads(open(le.LEARNING_STATE_FILE).read())
        assert data["cycle_count"] == 3
        assert data["trade_idea_gate_counts"] == {"ti_saved": 1}

    def test_persisted_counts_are_seeded_once_and_accumulate(self, ti_env):
        with open(le.LEARNING_STATE_FILE, "w") as f:
            json.dump({"trade_idea_gate_counts": {"ti_saved": 5}}, f)
        le._load_learning_state()
        le._bump_gate("ti_saved")
        le._load_learning_state()  # second load must not re-add the disk value
        le._save_learning_state({})
        data = json.loads(open(le.LEARNING_STATE_FILE).read())
        assert data["trade_idea_gate_counts"] == {"ti_saved": 6}

    def test_no_counter_write_before_seed(self, ti_env):
        # An unseeded process must not overwrite persisted counts with zeros.
        with open(le.LEARNING_STATE_FILE, "w") as f:
            json.dump({"trade_idea_gate_counts": {"ti_saved": 9}}, f)
        le._save_learning_state({"cycle_count": 1})
        data = json.loads(open(le.LEARNING_STATE_FILE).read())
        assert "trade_idea_gate_counts" not in data
