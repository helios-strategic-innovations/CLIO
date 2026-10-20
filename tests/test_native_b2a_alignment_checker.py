"""
B2a: alignment_checker.py repointed off Hydrinia's position_log.json onto
CLIO's own DPOF book (agent/dpof/readmodel.py).
"""

import json
import os

import pytest

from agent.integrations import alignment_checker as ac


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof"))


def _write_positions(data_dir, positions):
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "positions.json"), "w") as f:
        json.dump({"positions": positions}, f)


def test_no_open_positions_returns_empty(tmp_path):
    _write_positions(str(tmp_path / "dpof"), [])
    assert ac._get_open_positions() == []


def test_open_single_leg_position_mapped(tmp_path):
    _write_positions(str(tmp_path / "dpof"), [{
        "id": "pos1",
        "underlying": "SPY",
        "direction": "bullish",
        "structure": "single",
        "legs": [{"symbol": "SPY261002C00580000", "side": "buy"}],
        "status": "open",
        "qty_original": 1,
        "qty_open": 1,
        "entry_price": 3.25,
        "entry_time": 1758700000.0,
        "entry_signal": {"conviction": 0.72},
    }])

    positions = ac._get_open_positions()
    assert len(positions) == 1
    p = positions[0]
    assert p["symbol"] == "SPY"
    assert p["direction"] == "bullish"
    assert p["category"] == "DPOF single"
    assert p["entry_price"] == 3.25
    assert p["conviction"] == 0.72
    assert p["trade_id"] == "pos1"
    assert "580C" in p["contract"] or "580.0C" in p["contract"] or "580" in p["contract"]


def test_closed_and_pending_excluded(tmp_path):
    _write_positions(str(tmp_path / "dpof"), [
        {"id": "a", "underlying": "SPY", "direction": "bullish", "structure": "single",
         "legs": [], "status": "closed", "qty_original": 1, "qty_open": 0},
        {"id": "b", "underlying": "QQQ", "direction": "bearish", "structure": "single",
         "legs": [], "status": "cancelled", "qty_original": 1, "qty_open": 0},
        {"id": "c", "underlying": "SPY", "direction": "bullish", "structure": "single",
         "legs": [], "status": "pending", "qty_original": 1, "qty_open": 0},
    ])
    positions = ac._get_open_positions()
    # "pending" IS one of readmodel's open statuses (pending/open/closing) — only
    # the fully closed/cancelled ones must drop out.
    symbols = {p["symbol"] for p in positions}
    assert "QQQ" not in symbols  # cancelled
    ids = {p["trade_id"] for p in positions}
    assert "a" not in ids  # closed
    assert "c" in ids  # pending is still "open" from the trading-desk's perspective


def test_missing_conviction_defaults_unknown(tmp_path):
    _write_positions(str(tmp_path / "dpof"), [{
        "id": "pos2",
        "underlying": "QQQ",
        "direction": "bearish",
        "structure": "debit_vertical",
        "legs": [],
        "status": "open",
        "qty_original": 1,
        "qty_open": 1,
    }])
    positions = ac._get_open_positions()
    assert positions[0]["conviction"] == "unknown"
    assert positions[0]["category"] == "DPOF debit_vertical"


def test_alignment_prompt_includes_contract_and_structure():
    assert "{contract}" in ac.ALIGNMENT_PROMPT
    assert "Structure: {category}" in ac.ALIGNMENT_PROMPT


def test_dpof_unavailable_degrades_to_empty(monkeypatch):
    """readmodel import failure must degrade to an empty list, never raise."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "agent.dpof.readmodel" or name == "agent.dpof":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert ac._get_open_positions() == []
