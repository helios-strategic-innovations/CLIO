"""Tests for agent.dpof.broker.DPOFBroker -- the order chokepoint."""

import asyncio
import sys
from dataclasses import replace

import pytest

from agent.dpof.broker import DPOFBroker
from agent.dpof.config import DpofMode
from agent.dpof.models import OrderIntent, OrderLeg


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class RaisingClient:
    """Raises on ANY attribute access -- proves the broker never touches it."""

    def __getattr__(self, name):
        raise AssertionError(f"trading/data client touched in OFF/LOG_ONLY mode: .{name}")


class FakeAccount:
    def __init__(self, **kw):
        self.account_number = kw.get("account_number", "PA000DPOF")
        self.equity = kw.get("equity", "20000")
        self.last_equity = kw.get("last_equity", "19500")
        self.buying_power = kw.get("buying_power", "40000")
        self.options_buying_power = kw.get("options_buying_power", "20000")
        self.options_trading_level = kw.get("options_trading_level", 3)
        self.options_approved_level = kw.get("options_approved_level", 3)
        self.status = kw.get("status", "ACTIVE")
        self.trading_blocked = kw.get("trading_blocked", False)
        self.account_blocked = kw.get("account_blocked", False)


class FakeStatus:
    def __init__(self, value):
        self.value = value


class FakeOrder:
    def __init__(self, id="b-1", client_order_id="DPOF-EN-1", status="new", filled_qty=0,
                 filled_avg_price=None, limit_price=None, legs=None):
        self.id = id
        self.client_order_id = client_order_id
        self.status = FakeStatus(status)
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.limit_price = limit_price
        self.legs = legs or []


class FakeTradingClient:
    def __init__(self, account=None, account_error=None, submit_error=None, get_order_error=None):
        self.account = account or FakeAccount()
        self.account_error = account_error
        self.submit_error = submit_error
        self.get_order_error = get_order_error
        self.submitted = []
        self.cancelled = []
        self.replaced = []
        self._orders = {}

    def get_account(self):
        if self.account_error:
            raise self.account_error
        return self.account

    def submit_order(self, order_data):
        if self.submit_error:
            raise self.submit_error
        self.submitted.append(order_data)
        cid = getattr(order_data, "client_order_id", "") or ""
        order = FakeOrder(id=f"b-{len(self.submitted)}", client_order_id=cid, status="new")
        self._orders[cid] = order
        return order

    def get_order_by_client_id(self, client_id):
        if self.get_order_error:
            raise self.get_order_error
        if client_id not in self._orders:
            raise KeyError(client_id)
        return self._orders[client_id]

    def cancel_order_by_id(self, order_id):
        self.cancelled.append(order_id)

    def replace_order_by_id(self, order_id, order_data):
        self.replaced.append((order_id, order_data))
        return FakeOrder(id=order_id, status="new")

    def get_orders(self, filter=None):
        return list(self._orders.values())

    def get_all_positions(self):
        return []

    def get_option_contracts(self, request):
        class Resp:
            option_contracts = []
            next_page_token = None

        return Resp()


# ---------------------------------------------------------------------------
# Requirement 1: OFF / LOG_ONLY never touch the trading client
# ---------------------------------------------------------------------------


def _entry_intent(cfg, cid=None):
    return OrderIntent(
        client_order_id=cid or f"{cfg.client_order_prefix}-EN-20260924000000-abcdef",
        legs=[OrderLeg(symbol="SPY261002C00575000", side="buy", position_intent="buy_to_open")],
        qty=1,
        limit_price=1.05,
        purpose="entry",
        position_id="pos1",
    )


def test_off_mode_never_touches_trading_client(cfg):
    off_cfg = replace(cfg, mode=DpofMode.OFF)
    broker = DPOFBroker(off_cfg, trading_client=RaisingClient(), stock_data_client=RaisingClient(), option_data_client=RaisingClient())

    async def _t():
        result = await broker.submit(_entry_intent(off_cfg))
        assert result.ok is True
        assert result.would_submit is True

    asyncio.run(_t())


def test_log_only_mode_never_touches_trading_client(cfg):
    lo_cfg = replace(cfg, mode=DpofMode.LOG_ONLY)
    broker = DPOFBroker(lo_cfg, trading_client=RaisingClient(), stock_data_client=RaisingClient(), option_data_client=RaisingClient())

    async def _t():
        result = await broker.submit(_entry_intent(lo_cfg))
        assert result.ok is True
        assert result.would_submit is True
        gr = await broker.verify_gates(1000.0)
        assert gr.allowed is True

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Requirement 2: LIVE mode gates, tested individually
# ---------------------------------------------------------------------------


def _live_cfg(cfg, **overrides):
    base = dict(
        mode=DpofMode.LIVE,
        alpaca_paper=False,
        alpaca_key_id="key",
        alpaca_secret_key="secret",
        live_ack="I_ACCEPT_REAL_MONEY_RISK",
        live_account_number="PA000DPOF",
        min_options_level=2,
        min_account_equity=2000.0,
    )
    base.update(overrides)
    return replace(cfg, **base)


def _live_broker(live_cfg, account=None, account_error=None, kill_enabled=True, kill_import_fails=False):
    client = FakeTradingClient(account=account, account_error=account_error)
    broker = DPOFBroker(live_cfg, trading_client=client, stock_data_client=object(), option_data_client=object())
    return broker, client


async def _assert_live_blocked(monkeypatch, live_cfg, account=None, account_error=None,
                                kill_enabled=True, kill_import_fails=False, expect_reason_substr=None):
    _patch_kill_registry(monkeypatch, enabled=kill_enabled, import_fails=kill_import_fails)
    broker, client = _live_broker(live_cfg, account=account, account_error=account_error)
    gr = await broker.verify_gates(1000.0)
    assert gr.allowed is False
    if expect_reason_substr:
        assert any(expect_reason_substr in r for r in gr.reasons), gr.reasons
    assert broker.effective_mode() == DpofMode.LOG_ONLY

    result = await broker.submit(_entry_intent(live_cfg))
    assert result.would_submit is True
    assert client.submitted == []  # nothing sent
    return gr


def _patch_kill_registry(monkeypatch, enabled=True, import_fails=False):
    if import_fails:
        monkeypatch.setitem(sys.modules, "agent.safety.kill_registry", None)
        return

    class FakeRegistry:
        def is_enabled(self, name):
            return enabled if name == "live_trading" else False

    fake_module = type(sys)("agent.safety.kill_registry")
    fake_module.get_kill_registry = lambda: FakeRegistry()
    monkeypatch.setitem(sys.modules, "agent.safety.kill_registry", fake_module)


def test_live_gate_fails_on_paper_flag_true(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg, alpaca_paper=True)

    async def _t():
        await _assert_live_blocked(monkeypatch, live_cfg, expect_reason_substr="live_requires_paper_false")

    asyncio.run(_t())


def test_live_gate_fails_on_missing_ack_phrase(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg, live_ack="wrong-phrase")

    async def _t():
        await _assert_live_blocked(monkeypatch, live_cfg, expect_reason_substr="live_ack_missing")

    asyncio.run(_t())


def test_live_gate_fails_on_account_number_mismatch(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg, live_account_number="PA_EXPECTED")

    async def _t():
        await _assert_live_blocked(
            monkeypatch, live_cfg, account=FakeAccount(account_number="PA_ACTUAL"),
            expect_reason_substr="account_number_mismatch",
        )

    asyncio.run(_t())


def test_live_gate_fails_when_kill_switch_disabled(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg)

    async def _t():
        await _assert_live_blocked(
            monkeypatch, live_cfg, kill_enabled=False, expect_reason_substr="kill_switch_live_trading_disabled"
        )

    asyncio.run(_t())


def test_live_gate_fails_when_kill_registry_import_fails(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg)

    async def _t():
        await _assert_live_blocked(
            monkeypatch, live_cfg, kill_import_fails=True, expect_reason_substr="kill_switch_live_trading_disabled"
        )

    asyncio.run(_t())


def test_live_gate_fails_on_options_level_too_low(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg, min_options_level=3)

    async def _t():
        await _assert_live_blocked(
            monkeypatch, live_cfg, account=FakeAccount(options_trading_level=2),
            expect_reason_substr="options_level_too_low",
        )

    asyncio.run(_t())


def test_live_gate_passes_on_level_2_with_verticals_enabled(cfg, monkeypatch):
    # Pass-2 verifier V2 fix: min_options_level=2 (long calls/puts) + a
    # level-2 account must NOT be blocked just because allow_verticals
    # defaults to True -- verticals are gated separately, per-cycle, by
    # engine.py's own runtime check (options_level >= 3 -> offered;
    # otherwise singles only, no error). The account's basic ability to
    # trade the core strategy at all is min_options_level, not a
    # verticals-driven bump to 3 that would silently collapse the whole
    # mode to LOG_ONLY. Was test_live_gate_fails_on_level_2_with_verticals_
    # enabled, which asserted the OLD (buggy) blocking behavior.
    live_cfg = _live_cfg(cfg, min_options_level=2, allow_verticals=True)

    async def _t():
        _patch_kill_registry(monkeypatch, enabled=True)
        broker, client = _live_broker(live_cfg, account=FakeAccount(options_trading_level=2))
        gr = await broker.verify_gates(1000.0)
        assert gr.allowed is True, gr.reasons
        assert broker.effective_mode() == DpofMode.LIVE

    asyncio.run(_t())


def test_live_gate_fails_on_level_1_even_with_verticals_disabled(cfg, monkeypatch):
    # A level-1 account (Covered Call/Cash-Secured Put only) is genuinely
    # below the DPOF strategy's basic requirement (long calls/puts, level
    # 2) regardless of allow_verticals -- this must still block.
    live_cfg = _live_cfg(cfg, min_options_level=2, allow_verticals=False)

    async def _t():
        await _assert_live_blocked(
            monkeypatch, live_cfg, account=FakeAccount(options_trading_level=1),
            expect_reason_substr="options_level_too_low:1<2",
        )

    asyncio.run(_t())


def test_live_gate_fails_on_equity_below_min(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg, min_account_equity=50000.0)

    async def _t():
        await _assert_live_blocked(
            monkeypatch, live_cfg, account=FakeAccount(equity="20000"),
            expect_reason_substr="equity_below_min",
        )

    asyncio.run(_t())


def test_live_gate_fails_on_trading_blocked(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg)

    async def _t():
        await _assert_live_blocked(
            monkeypatch, live_cfg, account=FakeAccount(trading_blocked=True),
            expect_reason_substr="trading_blocked",
        )

    asyncio.run(_t())


def test_live_gate_fails_on_account_fetch_failure(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg)

    async def _t():
        await _assert_live_blocked(
            monkeypatch, live_cfg, account_error=RuntimeError("network down"),
            expect_reason_substr="account_fetch_failed",
        )

    asyncio.run(_t())


def test_live_gate_passes_when_everything_is_fine(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg)

    async def _t():
        _patch_kill_registry(monkeypatch, enabled=True)
        broker, client = _live_broker(live_cfg, account=FakeAccount())
        gr = await broker.verify_gates(1000.0)
        assert gr.allowed is True, gr.reasons
        assert broker.effective_mode() == DpofMode.LIVE

    asyncio.run(_t())


def test_gate_result_is_cached_for_300_seconds(cfg, monkeypatch):
    live_cfg = _live_cfg(cfg)

    async def _t():
        calls = {"n": 0}

        class CountingRegistry:
            def is_enabled(self, name):
                calls["n"] += 1
                return True

        fake_module = type(sys)("agent.safety.kill_registry")
        fake_module.get_kill_registry = lambda: CountingRegistry()
        monkeypatch.setitem(sys.modules, "agent.safety.kill_registry", fake_module)

        broker, client = _live_broker(live_cfg, account=FakeAccount())
        await broker.verify_gates(1000.0)
        await broker.verify_gates(1200.0)  # within 300s -- cached
        assert calls["n"] == 1
        await broker.verify_gates(1301.0)  # past the TTL -- re-checked
        assert calls["n"] == 2

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Requirement 3: PAPER mode refuses if cfg.alpaca_paper is False
# ---------------------------------------------------------------------------


def test_paper_mode_refuses_when_alpaca_paper_flag_false(cfg):
    paper_cfg = replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=False, alpaca_key_id="k", alpaca_secret_key="s"
    )
    client = FakeTradingClient(account=FakeAccount())
    broker = DPOFBroker(paper_cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    async def _t():
        gr = await broker.verify_gates(1000.0)
        assert gr.allowed is False
        assert "paper_flag_false" in gr.reasons
        assert broker.effective_mode() == DpofMode.LOG_ONLY

        result = await broker.submit(_entry_intent(paper_cfg))
        assert result.would_submit is True
        assert client.submitted == []

    asyncio.run(_t())


def test_paper_mode_passes_with_paper_flag_true(cfg):
    paper_cfg = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, alpaca_key_id="k", alpaca_secret_key="s")
    client = FakeTradingClient(account=FakeAccount())
    broker = DPOFBroker(paper_cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    async def _t():
        gr = await broker.verify_gates(1000.0)
        assert gr.allowed is True, gr.reasons
        assert broker.effective_mode() == DpofMode.PAPER

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Requirement 4: prefix enforcement on submit/cancel/replace
# ---------------------------------------------------------------------------


def test_submit_refuses_client_order_id_without_prefix(cfg):
    paper_cfg = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, alpaca_key_id="k", alpaca_secret_key="s")
    client = FakeTradingClient(account=FakeAccount())
    broker = DPOFBroker(paper_cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    async def _t():
        await broker.verify_gates(1000.0)
        bad_intent = _entry_intent(paper_cfg, cid="NOT-DPOF-PREFIXED-123")
        result = await broker.submit(bad_intent)
        assert result.ok is False
        assert "prefix" in result.error
        assert client.submitted == []

    asyncio.run(_t())


def test_cancel_refuses_non_dpof_client_order_id(cfg):
    client = FakeTradingClient()
    broker = DPOFBroker(cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    async def _t():
        ok = await broker.cancel_order("broker-order-1", "SOME-OTHER-SYSTEM-123")
        assert ok is False
        assert client.cancelled == []

    asyncio.run(_t())


def test_cancel_allows_dpof_prefixed_ids(cfg):
    client = FakeTradingClient()
    broker = DPOFBroker(cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    async def _t():
        ok = await broker.cancel_order("broker-order-1", f"{cfg.client_order_prefix}-EN-x")
        assert ok is True
        assert client.cancelled == ["broker-order-1"]

    asyncio.run(_t())


def test_broker_has_no_replace_order_method(cfg):
    """Alpaca's PATCH /v2/orders/{id} does NOT edit an order in place -- it
    creates a brand-new order/id (Order.replaces/replaced_by,
    OrderStatus.REPLACED). Using such a method from orders.py would silently
    orphan the ledger's tracked client_order_id on every reprice, so it must
    not exist to be accidentally reintroduced."""
    client = FakeTradingClient()
    broker = DPOFBroker(cfg, trading_client=client, stock_data_client=object(), option_data_client=object())
    assert not hasattr(broker, "replace_order")


# ---------------------------------------------------------------------------
# Requirement 5: audit hook
# ---------------------------------------------------------------------------


def test_audit_hook_called_before_real_submission(cfg, monkeypatch):
    paper_cfg = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, alpaca_key_id="k", alpaca_secret_key="s")
    client = FakeTradingClient(account=FakeAccount())
    broker = DPOFBroker(paper_cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    calls = []

    def fake_audit(tool_name, tool_input, user_id=""):
        calls.append((tool_name, tool_input, user_id))
        return "entry-id"

    import agent.safety.audit_log as audit_log_module

    monkeypatch.setattr(audit_log_module, "audit_tool_execution", fake_audit)

    async def _t():
        await broker.verify_gates(1000.0)
        result = await broker.submit(_entry_intent(paper_cfg))
        assert result.ok is True
        assert len(calls) == 1
        assert calls[0][0] == "dpof_submit_order"
        assert calls[0][2] == "dpof_engine"
        assert len(client.submitted) == 1  # audit happened BEFORE the real submit went out

    asyncio.run(_t())


def test_audit_hook_failure_refuses_entry(cfg, monkeypatch):
    paper_cfg = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, alpaca_key_id="k", alpaca_secret_key="s")
    client = FakeTradingClient(account=FakeAccount())
    broker = DPOFBroker(paper_cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    def raising_audit(tool_name, tool_input, user_id=""):
        raise RuntimeError("audit log unavailable")

    import agent.safety.audit_log as audit_log_module

    monkeypatch.setattr(audit_log_module, "audit_tool_execution", raising_audit)

    async def _t():
        await broker.verify_gates(1000.0)
        result = await broker.submit(_entry_intent(paper_cfg))
        assert result.ok is False
        assert client.submitted == []  # refused, never reached the broker

    asyncio.run(_t())


def test_audit_hook_failure_does_not_block_exits(cfg, monkeypatch):
    paper_cfg = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, alpaca_key_id="k", alpaca_secret_key="s")
    client = FakeTradingClient(account=FakeAccount())
    broker = DPOFBroker(paper_cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    def raising_audit(tool_name, tool_input, user_id=""):
        raise RuntimeError("audit log unavailable")

    import agent.safety.audit_log as audit_log_module

    monkeypatch.setattr(audit_log_module, "audit_tool_execution", raising_audit)

    async def _t():
        await broker.verify_gates(1000.0)
        exit_intent = OrderIntent(
            client_order_id=f"{paper_cfg.client_order_prefix}-EX-1",
            legs=[OrderLeg(symbol="SPY261002C00575000", side="sell", position_intent="sell_to_close")],
            qty=1,
            limit_price=1.05,
            purpose="exit",
            position_id="pos1",
            position_mode="paper",
        )
        result = await broker.submit(exit_intent)
        assert result.ok is True  # exits go out regardless of audit failure
        assert len(client.submitted) == 1

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Requirement 6: broker clients come from cfg, never agent.config
# ---------------------------------------------------------------------------


def test_construction_does_not_import_agent_config(cfg, monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.config", None)  # any `import agent.config` now raises

    live_cfg = _live_cfg(cfg)
    broker = DPOFBroker(live_cfg, trading_client=RaisingClient(), stock_data_client=object(), option_data_client=object())
    broker.update_config(live_cfg)  # must also not import agent.config
    assert broker is not None  # got this far without ImportError -- proves it


def test_lazy_client_build_uses_cfg_keys_not_agent_config(cfg, monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.config", None)

    built = {}

    class FakeSDKTradingClient:
        def __init__(self, api_key, secret_key, paper):
            built["api_key"] = api_key
            built["secret_key"] = secret_key
            built["paper"] = paper

    import alpaca.trading.client as trading_client_module

    monkeypatch.setattr(trading_client_module, "TradingClient", FakeSDKTradingClient)

    dedicated_cfg = replace(
        cfg, alpaca_key_id="dpof-dedicated-key", alpaca_secret_key="dpof-dedicated-secret", alpaca_paper=True
    )
    broker = DPOFBroker(dedicated_cfg)  # no injected clients -> must lazy-build
    client = broker._get_trading_client()
    assert isinstance(client, FakeSDKTradingClient)
    assert built["api_key"] == "dpof-dedicated-key"
    assert built["secret_key"] == "dpof-dedicated-secret"
    assert built["paper"] is True


def test_update_config_rebuilds_only_non_injected_clients(cfg):
    injected_trading = FakeTradingClient()
    broker = DPOFBroker(cfg, trading_client=injected_trading)  # stock/option not injected

    new_cfg = replace(cfg, alpaca_key_id="new-key", alpaca_secret_key="new-secret")
    broker.update_config(new_cfg)
    # Injected trading client must be untouched even though keys changed.
    assert broker._trading_client is injected_trading


# ---------------------------------------------------------------------------
# get_account / get_order_by_client_id shape
# ---------------------------------------------------------------------------


def test_get_account_never_reads_pdt_fields(cfg):
    client = FakeTradingClient(account=FakeAccount())
    broker = DPOFBroker(cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    async def _t():
        acct = await broker.get_account()
        assert acct is not None
        assert "pattern_day_trader" not in acct
        assert "daytrade_count" not in acct
        assert acct["account_number"] == "PA000DPOF"
        assert acct["equity"] == 20000.0
        assert acct["options_trading_level"] == 3
        assert acct["trading_blocked"] is False

    asyncio.run(_t())


def test_get_order_by_client_id_raises_lookup_error_on_failure(cfg):
    """Orchestrator review: a lookup FAILURE must never read as "not found" (None) —
    the order manager resends on not-found, which on a transient error could put a
    second live order in the market."""
    from agent.dpof.broker import BrokerLookupError
    client = FakeTradingClient(get_order_error=RuntimeError("boom"))
    broker = DPOFBroker(cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    async def _t():
        with pytest.raises(BrokerLookupError):
            await broker.get_order_by_client_id("DPOF-EN-missing")

    asyncio.run(_t())


def test_get_order_by_client_id_returns_none_only_on_404(cfg):
    class _NotFound(Exception):
        status_code = 404

    client = FakeTradingClient(get_order_error=_NotFound("order not found"))
    broker = DPOFBroker(cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    async def _t():
        assert await broker.get_order_by_client_id("DPOF-EN-missing") is None

    asyncio.run(_t())


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(pytest.main([__file__, "-q"]))



# ── Orchestrator review: gates govern ENTRIES; exits for real positions always go out ──

def _gate_exit_intent(prefix, pmode):
    return OrderIntent(
        client_order_id=f"{prefix}-EX-gate", legs=[OrderLeg(symbol="SPY261002C00575000", side="sell",
        position_intent="sell_to_close")], qty=1, limit_price=1.05, purpose="exit",
        position_id="p1", position_mode=pmode)


def _gate_entry_intent(prefix):
    return OrderIntent(
        client_order_id=f"{prefix}-EN-gate", legs=[OrderLeg(symbol="SPY261002C00575000", side="buy",
        position_intent="buy_to_open")], qty=1, limit_price=1.05, purpose="entry", position_id="p1")


def test_exit_for_paper_position_sent_even_when_gates_fail(cfg, monkeypatch):
    import agent.safety.audit_log as audit_log_module
    monkeypatch.setattr(audit_log_module, "audit_tool_execution", lambda *a, **k: "ok")
    paper_cfg = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, alpaca_key_id="k", alpaca_secret_key="s")
    client = FakeTradingClient(account_error=RuntimeError("account api down"))  # gates fail
    broker = DPOFBroker(paper_cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    async def _t():
        gates = await broker.verify_gates(1000.0)
        assert gates.allowed is False and broker.effective_mode() == DpofMode.LOG_ONLY
        ent = await broker.submit(_gate_entry_intent(paper_cfg.client_order_prefix))
        assert ent.would_submit is True                 # entry blocked by gates
        ext = await broker.submit(_gate_exit_intent(paper_cfg.client_order_prefix, "paper"))
        assert ext.would_submit is False and len(client.submitted) == 1   # exit still sent

    asyncio.run(_t())


def test_exit_refused_when_account_type_differs_from_position(cfg, monkeypatch):
    import agent.safety.audit_log as audit_log_module
    monkeypatch.setattr(audit_log_module, "audit_tool_execution", lambda *a, **k: "ok")
    paper_cfg = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, alpaca_key_id="k", alpaca_secret_key="s")
    client = FakeTradingClient(account=FakeAccount())
    broker = DPOFBroker(paper_cfg, trading_client=client, stock_data_client=object(), option_data_client=object())

    async def _t():
        res = await broker.submit(_gate_exit_intent(paper_cfg.client_order_prefix, "live"))   # live position, paper keys
        assert res.would_submit is True and client.submitted == []
        res2 = await broker.submit(_gate_exit_intent(paper_cfg.client_order_prefix, "log_only"))
        assert res2.would_submit is True and client.submitted == []

    asyncio.run(_t())
