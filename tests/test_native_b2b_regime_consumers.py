"""
B2b: seal_reward_engine.py and social_intel_divergence.py repointed off Hydrinia's
frozen state/daily_metrics.json onto agent/integrations/market_regime.py (CLIO-native).
Hydrinia is permanently off (operator direction 2026-09-24).

No network, no real state files touched.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from agent.integrations import seal_reward_engine as sre
from agent.integrations import social_intel_divergence as sid


# ---------------------------------------------------------------------------
# seal_reward_engine.py — regime lookup feeding the category bandit
# ---------------------------------------------------------------------------


class _FakeBandit:
    def __init__(self):
        self.calls = []

    def record_outcome(self, regime, category, won):
        self.calls.append((regime, category, won))


class TestSealRewardEngineRegime:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sre, "REWARD_STATE_FILE", str(tmp_path / "seal_reward_state.json"))
        self.fake_bandit = _FakeBandit()
        monkeypatch.setattr(
            "agent.integrations.seal_category_bandit.get_category_bandit",
            lambda: self.fake_bandit,
        )
        self.engine = sre.SEALRewardEngine()

    def test_uses_fresh_market_regime_label(self, monkeypatch):
        monkeypatch.setattr(
            "agent.integrations.market_regime.get_market_regime_label",
            lambda **kw: "BEAR",
        )
        self.engine.record_binary_reward(
            ticker="AAPL", won=True, trade_idea_id="ti_1", source_category="equities_intraday",
        )
        assert self.fake_bandit.calls == [("BEAR", "equities_intraday", True)]

    def test_defaults_to_choppy_when_regime_lookup_fails(self, monkeypatch):
        def boom(**kw):
            raise RuntimeError("boom")

        monkeypatch.setattr("agent.integrations.market_regime.get_market_regime_label", boom)
        self.engine.record_binary_reward(
            ticker="AAPL", won=False, trade_idea_id="ti_2", source_category="macro",
        )
        assert self.fake_bandit.calls == [("CHOPPY", "macro", False)]

    def test_no_bandit_call_without_source_category(self, monkeypatch):
        monkeypatch.setattr(
            "agent.integrations.market_regime.get_market_regime_label",
            lambda **kw: "BULL",
        )
        self.engine.record_binary_reward(ticker="AAPL", won=True, trade_idea_id="ti_3")
        assert self.fake_bandit.calls == []


# ---------------------------------------------------------------------------
# social_intel_divergence.py — _bot_regime_component() staleness handling
# ---------------------------------------------------------------------------


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


class TestBotRegimeComponent:
    def test_missing_returns_none(self, monkeypatch):
        monkeypatch.setattr("agent.integrations.market_regime.get_market_regime", lambda **kw: None)
        assert sid._bot_regime_component() is None

    def test_fresh_read_full_weight(self, monkeypatch):
        monkeypatch.setattr(
            "agent.integrations.market_regime.get_market_regime",
            lambda **kw: {"regime": "BEAR", "confidence": 0.8, "as_of": _iso(datetime.now(timezone.utc))},
        )
        comp = sid._bot_regime_component()
        assert comp is not None
        assert comp["weight"] == 1.0
        assert comp["stale"] is False
        assert comp["raw"]["regime"] == "BEAR"
        assert comp["value"] == pytest.approx(round(sid._BOT_REGIME_RISK["BEAR"] * 0.8, 3))

    def test_stale_but_within_max_age_zero_weighted_and_flagged(self, monkeypatch):
        old = datetime.now(timezone.utc) - timedelta(seconds=sid._BOT_REGIME_STALE_S + 3600)
        monkeypatch.setattr(
            "agent.integrations.market_regime.get_market_regime",
            lambda **kw: {"regime": "BULL", "confidence": 0.6, "as_of": _iso(old)},
        )
        comp = sid._bot_regime_component()
        assert comp is not None
        assert comp["weight"] == 0.0
        assert comp["stale"] is True

    def test_unknown_regime_label_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "agent.integrations.market_regime.get_market_regime",
            lambda **kw: {"regime": "SIDEWAYS", "confidence": 0.5, "as_of": _iso(datetime.now(timezone.utc))},
        )
        assert sid._bot_regime_component() is None

    def test_passes_max_age_to_market_regime(self, monkeypatch):
        seen = {}

        def fake_get(**kw):
            seen.update(kw)
            return None

        monkeypatch.setattr("agent.integrations.market_regime.get_market_regime", fake_get)
        sid._bot_regime_component()
        assert seen.get("max_age_seconds") == sid._BOT_REGIME_MAX_AGE_S

    def test_never_raises_on_failure(self, monkeypatch):
        def boom(**kw):
            raise RuntimeError("boom")

        monkeypatch.setattr("agent.integrations.market_regime.get_market_regime", boom)
        assert sid._bot_regime_component() is None

    def test_no_daily_metrics_path_reference_remains(self):
        assert not hasattr(sid, "_DAILY_METRICS_PATH")
