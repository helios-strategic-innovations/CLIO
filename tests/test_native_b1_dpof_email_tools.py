"""
Tests for agent/tools/dpof_tools.py (new get_dpof_status tool) and the
CLIO-native rebuild of agent/tools/email_tools.py::handle_email_trade_summary
(B1 scope).

No network: the in-process DPOF engine is never constructed (import failure is
simulated to force the readmodel fallback path), and the Alpaca client/SMTP
send are monkeypatched. All DPOF data reads go through a tmp CLIO_DPOF_DATA_DIR.
"""

import asyncio
import json
import os
import time

import pytest

import agent.tools.dpof_tools as dpof_tools
import agent.tools.email_tools as email_tools


def _run(coro):
    return asyncio.run(coro)


def _write_open_position(ddir, **overrides):
    pos = {
        "id": "p1",
        "underlying": "QQQ",
        "direction": "bearish",
        "structure": "single",
        "legs": [{"symbol": "QQQ261010P00480000", "side": "buy"}],
        "status": "open",
        "qty_original": 3,
        "qty_open": 3,
        "entry_price": 2.10,
        "entry_time": time.time(),
    }
    pos.update(overrides)
    with open(os.path.join(ddir, "positions.json"), "w") as f:
        json.dump({"positions": [pos]}, f)
    return pos


@pytest.fixture(autouse=True)
def _isolated_dpof_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture(autouse=True)
def _force_engine_unavailable(monkeypatch):
    """Simulate the in-process engine being unimportable so handle_dpof_status
    exercises its readmodel fallback path deterministically (no live engine
    singleton construction in a unit test)."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "agent.dpof.engine":
            raise ImportError("simulated: engine unavailable in this process")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    yield


# ---------------------------------------------------------------------------
# dpof_tools.handle_dpof_status
# ---------------------------------------------------------------------------

class TestDpofStatusTool:
    def test_default_section_is_status(self, tmp_path):
        _write_open_position(tmp_path)
        result = _run(dpof_tools.handle_dpof_status({}))
        assert "DPOF mode=" in result
        assert "open positions: 1" in result

    def test_invalid_section_falls_back_to_status(self, tmp_path):
        result = _run(dpof_tools.handle_dpof_status({"section": "bogus"}))
        assert "DPOF mode=" in result

    def test_positions_section_lists_open_position(self, tmp_path):
        _write_open_position(tmp_path)
        result = _run(dpof_tools.handle_dpof_status({"section": "positions"}))
        assert "DPOF OPEN POSITIONS" in result
        assert "QQQ" in result

    def test_positions_section_flat_book(self, tmp_path):
        with open(os.path.join(tmp_path, "positions.json"), "w") as f:
            json.dump({"positions": []}, f)
        result = _run(dpof_tools.handle_dpof_status({"section": "positions"}))
        assert "no open positions" in result.lower()

    def test_decisions_section_reads_ledger(self, tmp_path):
        decisions_path = os.path.join(tmp_path, "decisions.jsonl")
        with open(decisions_path, "w") as f:
            f.write(json.dumps({"ts": 1, "action": "entry", "symbol": "SPY"}) + "\n")
            f.write(json.dumps({"ts": 2, "action": "exit", "symbol": "QQQ"}) + "\n")
        result = _run(dpof_tools.handle_dpof_status({"section": "decisions"}))
        assert "DPOF recent decisions (2)" in result

    def test_decisions_section_empty_ledger(self, tmp_path):
        result = _run(dpof_tools.handle_dpof_status({"section": "decisions"}))
        assert "no recent decisions" in result.lower()

    def test_config_section_degrades_gracefully_without_engine(self, tmp_path):
        result = _run(dpof_tools.handle_dpof_status({"section": "config"}))
        assert "config unavailable" in result.lower()

    def test_output_is_sanitized_untrusted_wrapper(self, tmp_path):
        result = _run(dpof_tools.handle_dpof_status({}))
        assert "<ingested_data" in result
        assert 'source="dpof_status"' in result

    def test_never_requires_confirmation(self):
        """get_dpof_status is a read-only status tool — never in
        CONFIRMATION_REQUIRED."""
        from agent.security import CONFIRMATION_REQUIRED
        assert "get_dpof_status" not in CONFIRMATION_REQUIRED


# ---------------------------------------------------------------------------
# email_tools.handle_email_trade_summary
# ---------------------------------------------------------------------------

class _FakeAccount:
    equity = "10500.25"
    last_equity = "10400.00"
    buying_power = "5000.00"


class _FakeTradingClient:
    def get_account(self):
        return _FakeAccount()


@pytest.fixture
def _fake_send_account(monkeypatch):
    """Route email_tools' account resolution to a fake, always-sendable
    account so handle_email_trade_summary doesn't need real config.email_accounts."""
    from agent.config import EmailAccount

    fake_acct = EmailAccount(
        name="bot_atlas", imap_host="imap.example.com", smtp_host="smtp.example.com",
        address="bot@example.org", password="x", can_send=True,
    )
    monkeypatch.setattr(
        email_tools, "_resolve_account", lambda account_name, need_send=False: (fake_acct, ""),
    )
    return fake_acct


@pytest.fixture(autouse=True)
def _no_real_smtp(monkeypatch):
    sent = {}

    async def _fake_send(msg, **kwargs):
        sent["msg"] = msg
        sent["kwargs"] = kwargs

    monkeypatch.setattr(email_tools.aiosmtplib, "send", _fake_send)
    return sent


class TestEmailTradeSummary:
    def test_subject_and_sender_are_clio_native(self, tmp_path, monkeypatch, _fake_send_account, _no_real_smtp):
        monkeypatch.setattr("agent.integrations.alpaca._get_trading_client", lambda: _FakeTradingClient())

        async def _fake_retry(fn, *a, **kw):
            return fn()

        monkeypatch.setattr("agent.integrations.alpaca._retry_alpaca_call", _fake_retry)
        with open(os.path.join(tmp_path, "positions.json"), "w") as f:
            json.dump({"positions": []}, f)

        result = _run(email_tools.handle_email_trade_summary({
            "to": "operator@example.com", "summary_type": "daily",
        }))
        assert "Trade summary sent" in result
        sent_msg = _no_real_smtp["msg"]
        assert sent_msg["Subject"].startswith("CLIO DPOF Daily Summary")
        assert "Hydrinia" not in sent_msg["Subject"]

    def test_body_includes_equity_day_pnl_and_dpof_positions(self, tmp_path, monkeypatch, _fake_send_account, _no_real_smtp):
        monkeypatch.setattr("agent.integrations.alpaca._get_trading_client", lambda: _FakeTradingClient())

        async def _fake_retry(fn, *a, **kw):
            return fn()

        monkeypatch.setattr("agent.integrations.alpaca._retry_alpaca_call", _fake_retry)
        _write_open_position(tmp_path)

        result = _run(email_tools.handle_email_trade_summary({
            "to": "operator@example.com",
        }))
        assert "Trade summary sent" in result
        body = _no_real_smtp["msg"].get_content()
        assert "Equity: $10,500.25" in body
        # day P&L = equity - last_equity = 10500.25 - 10400.00 = 100.25
        assert "$+100.25" in body
        assert "QQQ" in body  # DPOF open position surfaced

    def test_alpaca_failure_degrades_gracefully(self, tmp_path, monkeypatch, _fake_send_account, _no_real_smtp):
        def _boom():
            raise RuntimeError("simulated Alpaca outage")

        monkeypatch.setattr("agent.integrations.alpaca._get_trading_client", _boom)
        with open(os.path.join(tmp_path, "positions.json"), "w") as f:
            json.dump({"positions": []}, f)

        result = _run(email_tools.handle_email_trade_summary({
            "to": "operator@example.com",
        }))
        assert "Trade summary sent" in result
        body = _no_real_smtp["msg"].get_content()
        assert "Account: unavailable" in body
