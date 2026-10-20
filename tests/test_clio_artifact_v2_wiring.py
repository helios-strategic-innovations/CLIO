"""
learning_engine.py trade-idea dual-emission wiring: CLIO's own clio_artifact_v2
(TRADE_IDEAS_V2_DIR) is always emitted; the legacy Hydrinia moltbot_knowledge mirror
(trade_ideas + trade_ideas_v2) is gated on agent.hydrinia_bridge.bridge_enabled()
(Hydrinia is permanently off, operator direction 2026-09-24 -- S205-B2b).

Behavioral (not source-grep): actually calls _save_trade_idea() and inspects what
landed on disk, so a future refactor of the mirror-dir plumbing can't silently drop
coverage the way a literal string match did.
"""

import asyncio
import json
from pathlib import Path

import pytest

from agent.integrations import learning_engine as le

_VALID_IDEA = {
    "ticker": "AAPL", "direction": "LONG", "confidence": 0.55,
    "thesis": "test thesis", "entry_trigger": "break 200", "stop_loss": "190",
    "take_profit": "220", "timeframe": "1w",
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(le, "TRADE_IDEAS_DIR", str(tmp_path / "trade_ideas"))
    monkeypatch.setattr(le, "TRADE_IDEAS_V2_DIR", str(tmp_path / "trade_ideas_v2"))
    monkeypatch.setattr(le, "_HYDRINIA_TRADE_IDEAS_MIRROR_DIR", str(tmp_path / "mirror" / "trade_ideas"))
    monkeypatch.setattr(le, "_HYDRINIA_TRADE_IDEAS_V2_MIRROR_DIR", str(tmp_path / "mirror" / "trade_ideas_v2"))
    monkeypatch.setattr("agent.integrations.market_regime.get_market_regime", lambda **kw: None)
    return tmp_path


def _run(coro):
    return asyncio.run(coro)


def test_v2_clio_local_artifact_always_emitted_bridge_off(monkeypatch, _isolate):
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    _run(le._save_trade_idea(dict(_VALID_IDEA), "equities_intraday", ["http://x"]))

    v2_files = list((_isolate / "trade_ideas_v2").glob("*.json"))
    assert len(v2_files) == 1, "CLIO-local v2 artifact must be emitted regardless of bridge state"
    artifact = json.loads(v2_files[0].read_text())
    assert artifact["schema_version"]
    assert artifact["artifact_id"]

    legacy_files = list((_isolate / "trade_ideas").glob("ti_*.json"))
    assert len(legacy_files) == 1, "CLIO's own legacy-format copy must also be emitted unconditionally"


def test_v2_clio_local_artifact_emitted_bridge_on_too(monkeypatch, _isolate):
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    _run(le._save_trade_idea(dict(_VALID_IDEA), "equities_intraday", ["http://x"]))
    v2_files = list((_isolate / "trade_ideas_v2").glob("*.json"))
    assert len(v2_files) == 1


def test_hydrinia_mirror_not_written_when_bridge_off(monkeypatch, _isolate):
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    _run(le._save_trade_idea(dict(_VALID_IDEA), "equities_intraday", ["http://x"]))

    assert not (_isolate / "mirror" / "trade_ideas").exists()
    assert not (_isolate / "mirror" / "trade_ideas_v2").exists()


def test_hydrinia_mirror_written_when_bridge_on(monkeypatch, _isolate):
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    _run(le._save_trade_idea(dict(_VALID_IDEA), "equities_intraday", ["http://x"]))

    mirror_legacy = list((_isolate / "mirror" / "trade_ideas").glob("ti_*.json"))
    mirror_v2 = list((_isolate / "mirror" / "trade_ideas_v2").glob("*.json"))
    assert len(mirror_legacy) == 1, "Hydrinia legacy mirror must be written when the bridge is enabled"
    assert len(mirror_v2) == 1, "Hydrinia v2 mirror must be written when the bridge is enabled"

    # Bridge-on doesn't stop CLIO's own copies from being written too.
    assert len(list((_isolate / "trade_ideas").glob("ti_*.json"))) == 1
    assert len(list((_isolate / "trade_ideas_v2").glob("*.json"))) == 1


def test_bridge_check_failure_never_raises_and_skips_mirror(monkeypatch, _isolate):
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr("agent.hydrinia_bridge.bridge_enabled", boom)
    _run(le._save_trade_idea(dict(_VALID_IDEA), "equities_intraday", ["http://x"]))  # must not raise

    assert len(list((_isolate / "trade_ideas_v2").glob("*.json"))) == 1  # CLIO-local copy unaffected
    assert not (_isolate / "mirror" / "trade_ideas_v2").exists()  # bridge check failure -> no mirror


@pytest.mark.parametrize("direction", ["HOLD", "NEUTRAL", "SIDEWAYS", "WOBBLE"])
def test_rejects_unknown_or_non_directional(monkeypatch, _isolate, direction):
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    idea = dict(_VALID_IDEA)
    idea["direction"] = direction
    _run(le._save_trade_idea(idea, "equities_intraday", ["http://x"]))

    assert list((_isolate / "trade_ideas").glob("ti_*.json")) == []
    assert list((_isolate / "trade_ideas_v2").glob("*.json")) == []


def test_wiring_present_in_source():
    """Cheap smoke check that the dual-emission machinery is still wired up at all
    (build_trade_idea_artifact / atomic_write_artifact / TRADE_IDEAS_V2_DIR) -- kept
    alongside the behavioral tests above, not as a substitute for them."""
    root = Path(__file__).parents[1]
    source = (root / "agent" / "integrations" / "learning_engine.py").read_text(encoding="utf-8")
    assert "TRADE_IDEAS_V2_DIR" in source
    assert "build_trade_idea_artifact(" in source
    assert "atomic_write_artifact(" in source
    assert "_HYDRINIA_TRADE_IDEAS_V2_MIRROR_DIR" in source
    assert "bridge_enabled" in source
