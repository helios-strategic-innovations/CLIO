"""
B2a: analytics_fun_facts.py rebuilt on CLIO-native data (agent/dpof/readmodel.py
+ CLIO's own Alpaca paper account) instead of Hydrinia's state/ files.
"""

import asyncio
import json
import os

import pytest

from agent.integrations import analytics_fun_facts as aff


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof"))


def _write_dpof(data_dir, *, open_count=0, closed_today=None, realized_week=0.0,
                 halt=False, mode="paper"):
    os.makedirs(data_dir, exist_ok=True)
    import time
    positions = []
    for i in range(open_count):
        positions.append({
            "id": f"open{i}", "underlying": "SPY", "direction": "bullish",
            "structure": "single", "legs": [], "status": "open",
            "qty_original": 1, "qty_open": 1,
        })
    if closed_today is not None:
        positions.append({
            "id": "closed1", "underlying": "QQQ", "direction": "bearish",
            "structure": "single", "legs": [], "status": "closed",
            "qty_original": 1, "qty_open": 0,
            "realized_pnl": closed_today, "closed_time": time.time(),
        })
    with open(os.path.join(data_dir, "positions.json"), "w") as f:
        json.dump({"positions": positions}, f)
    risk = {"day": "2026-09-24", "realized_pnl_today": closed_today or 0.0,
            "realized_pnl_week": realized_week}
    with open(os.path.join(data_dir, "risk_state.json"), "w") as f:
        json.dump(risk, f)
    if halt:
        with open(os.path.join(data_dir, "HALT"), "w") as f:
            json.dump({"reason": "test halt"}, f)


class FakeAccount:
    def __init__(self, equity, last_equity):
        self.equity = equity
        self.last_equity = last_equity


class FakeTradingClient:
    def get_account(self):
        raise NotImplementedError("stubbed by _retry_alpaca_call in tests")


def _patch_alpaca(monkeypatch, account):
    async def _fake_retry(fn, *args, **kwargs):
        return account

    monkeypatch.setattr("agent.integrations.alpaca._get_trading_client", lambda: FakeTradingClient())
    monkeypatch.setattr("agent.integrations.alpaca._retry_alpaca_call", _fake_retry)


def test_dynamic_facts_include_dpof_and_account(monkeypatch, tmp_path):
    _write_dpof(str(tmp_path / "dpof"), open_count=2, closed_today=15.5, realized_week=100.0)
    _patch_alpaca(monkeypatch, FakeAccount(equity="105000.00", last_equity="104500.00"))

    facts = asyncio.run(aff._generate_dynamic_facts())
    joined = " | ".join(facts)
    assert "Currently holding 2 open DPOF position" in joined
    assert "1 DPOF trade(s) closed today" in joined
    assert "realized +$15.50" in joined
    assert "DPOF realized P&L this week: +$100.00" in joined
    assert "Current CLIO paper account equity: $105,000." in joined
    assert "Today's account P&L: +$500.00" in joined


def test_halted_book_surfaces_fact(monkeypatch, tmp_path):
    _write_dpof(str(tmp_path / "dpof"), halt=True)
    _patch_alpaca(monkeypatch, FakeAccount(equity="0", last_equity="0"))

    facts = asyncio.run(aff._generate_dynamic_facts())
    assert any("HALTED" in f for f in facts)


def test_flat_book_and_account_unavailable_degrades_gracefully(monkeypatch, tmp_path):
    _write_dpof(str(tmp_path / "dpof"))

    async def _raise(fn, *a, **kw):
        raise RuntimeError("alpaca down")

    monkeypatch.setattr("agent.integrations.alpaca._get_trading_client", lambda: object())
    monkeypatch.setattr("agent.integrations.alpaca._retry_alpaca_call", _raise)

    facts = asyncio.run(aff._generate_dynamic_facts())
    assert isinstance(facts, list)  # never raises


def test_no_hydrinia_state_reads_remain():
    """The old Hydrinia state/*.json reader is gone entirely (module docstring
    is allowed to still name the retired path for context)."""
    import inspect

    src = inspect.getsource(aff)
    assert "_STATE_DIR" not in src
    assert "_read_state" not in src
    assert "Hydrinia v" not in src  # the old ever-growing uptime fact is gone


def test_static_facts_are_clio_native():
    joined = " ".join(aff.STATIC_FACTS)
    assert "DPOF" in joined
    assert "Hydrinia" not in joined


def test_generate_fun_facts_combines_static_and_dynamic(monkeypatch, tmp_path):
    _write_dpof(str(tmp_path / "dpof"), open_count=1)
    _patch_alpaca(monkeypatch, FakeAccount(equity="100000", last_equity="100000"))

    facts = asyncio.run(aff.generate_fun_facts())
    assert len(facts) >= len(aff.STATIC_FACTS)
    for f in aff.STATIC_FACTS:
        assert f in facts


def test_refresh_writes_output_file(monkeypatch, tmp_path):
    _write_dpof(str(tmp_path / "dpof"), open_count=1)
    _patch_alpaca(monkeypatch, FakeAccount(equity="100000", last_equity="99000"))

    out_path = tmp_path / "analytics_fun_facts.json"
    monkeypatch.setattr(aff, "_OUTPUT_PATH", str(out_path))

    asyncio.run(aff.refresh_analytics_fun_facts(context=None))

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "facts" in data and isinstance(data["facts"], list)
    assert data["static_count"] == len(aff.STATIC_FACTS)
