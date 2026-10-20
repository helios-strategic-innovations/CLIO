"""
Tests for agent/prompts/system_prompt.py (CLIO-native repoint, B1 scope).

Covers:
- The static SYSTEM_PROMPT text no longer claims CLIO is a "digital embodiment"/
  "personification" of hydrinia.py, and does reference DPOF/Atlas Strategies.
- The dynamic "current open positions" block is sourced from
  agent.dpof.readmodel.positions_brief() (CLIO's own DPOF book), not the frozen
  Hydrinia position_log.json — present when DPOF has open positions, absent
  when flat.
- The OBS-2 fail-visible logging contract (exact wrapper text + warning +
  exc_info + exception class) survives the repoint — see
  tests/test_obs2_sysprompt_fail_visible_logging.py for the AST-level check;
  this file adds a runtime check that a readmodel failure is swallowed and
  produces no injected block (graceful degradation), not a crash.

No network, no real data/ writes: DPOF data dir is redirected to tmp_path via
CLIO_DPOF_DATA_DIR for every test.
"""

import json
import os
import time

import pytest

from agent.prompts import system_prompt as sp


@pytest.fixture(autouse=True)
def _isolated_dpof_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path))
    # readmodel caches parsed JSON by (path, mtime_ns, size); a fresh tmp_path
    # per test gives each test a distinct path, so no cross-test cache bleed.
    yield tmp_path


def _write_position(d, **overrides):
    pos = {
        "id": "p1",
        "underlying": "SPY",
        "direction": "bullish",
        "structure": "single",
        "legs": [{"symbol": "SPY261002C00580000", "side": "buy"}],
        "status": "open",
        "qty_original": 2,
        "qty_open": 2,
        "entry_price": 3.50,
        "entry_time": time.time(),
    }
    pos.update(overrides)
    with open(os.path.join(d, "positions.json"), "w") as f:
        json.dump({"positions": [pos]}, f)
    return pos


class TestStaticPromptText:
    def test_no_longer_claims_hydrinia_embodiment(self):
        assert "digital embodiment of the hydrinia.py" not in sp.SYSTEM_PROMPT
        assert "personification of hydrinia.py" not in sp.SYSTEM_PROMPT

    def test_references_dpof_and_atlas_strategies(self):
        assert "DPOF" in sp.SYSTEM_PROMPT
        assert "Atlas Strategies" in sp.SYSTEM_PROMPT

    def test_key_files_section_is_clio_native(self):
        assert "KEY CLIO TRADING FILES" in sp.SYSTEM_PROMPT
        assert "KEY HYDRINIA FILES" not in sp.SYSTEM_PROMPT
        assert "data/dpof/HALT" in sp.SYSTEM_PROMPT

    def test_file_operations_default_to_openclaw(self):
        assert "sandboxed to /home/agent/openclaw/" in sp.SYSTEM_PROMPT

    def test_trading_dpof_section_present_not_hydrinia_bot_management(self):
        assert "TRADING (DPOF):" in sp.SYSTEM_PROMPT
        assert "HYDRINIA BOT MANAGEMENT" not in sp.SYSTEM_PROMPT
        assert "get_dpof_status" in sp.SYSTEM_PROMPT


class TestPositionsInjection:
    def test_open_dpof_position_is_injected(self, tmp_path):
        _write_position(tmp_path)
        blocks = sp.get_system_prompt(model="test-model", conversation_id="c-open")
        assert len(blocks) == 2
        dynamic_text = blocks[1]["text"]
        assert "DPOF OPEN POSITIONS" in dynamic_text
        assert "SPY" in dynamic_text

    def test_flat_book_omits_positions_block(self, tmp_path):
        with open(os.path.join(tmp_path, "positions.json"), "w") as f:
            json.dump({"positions": []}, f)
        blocks = sp.get_system_prompt(model="test-model", conversation_id="c-flat")
        dynamic_text = blocks[1]["text"] if len(blocks) > 1 else ""
        assert "DPOF OPEN POSITIONS" not in dynamic_text
        assert "CURRENT OPEN POSITIONS" not in dynamic_text  # old Hydrinia label gone

    def test_missing_positions_file_is_graceful(self, tmp_path):
        # No positions.json written at all — readmodel degrades to [].
        blocks = sp.get_system_prompt(model="test-model", conversation_id="c-missing")
        dynamic_text = blocks[1]["text"] if len(blocks) > 1 else ""
        assert "DPOF OPEN POSITIONS" not in dynamic_text

    def test_readmodel_import_failure_degrades_silently(self, monkeypatch, caplog):
        """If agent.dpof.readmodel itself is unimportable, the positions block
        is simply omitted — get_system_prompt() must never raise."""
        import builtins

        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "agent.dpof.readmodel":
                raise ImportError("simulated import failure")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked_import)
        blocks = sp.get_system_prompt(model="test-model", conversation_id="c-importfail")
        assert isinstance(blocks, list)
        dynamic_text = blocks[1]["text"] if len(blocks) > 1 else ""
        assert "DPOF OPEN POSITIONS" not in dynamic_text
