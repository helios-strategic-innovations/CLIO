"""B3 — Mission Control repointed off Hydrinia onto CLIO's own DPOF engine.

Covers: agent/dpof/readmodel.py-backed reads (positions, risk, decisions,
snapshot), the DPOF-native /api/actions/*, /api/dpof/*, /api/hydrinia/*
(repointed) routes, and the CLIO_HYDRINIA_BRIDGE_ENABLED gate on every
remaining Hydrinia write/health-check/control path in mission_control/backend/.

These tests import the REAL mission_control.backend.* modules (fastapi,
pydantic, bcrypt, jwt, alpaca are all installed in this environment) rather
than stubbing them — only `psutil` (unused by the modules under test here)
is absent, and none of these route files import it at module scope. Each
module is loaded fresh per test via spec_from_file_location so module-level
mutable state (e.g. actions._last_halt_time) never leaks between tests.
"""

import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.dpof.models import PositionRecord, RiskState  # noqa: E402
import agent.dpof.readmodel as readmodel  # noqa: E402


def _load_fresh(monkeypatch, tmp_path, rel_path: str, name: str):
    """Load a mission_control.backend module fresh, isolated from other tests."""
    monkeypatch.setenv("MISSION_CONTROL_JWT_SECRET", "test-secret")
    monkeypatch.setenv("MISSION_CONTROL_PASSWORD_HASH", "test-hash")
    module_name = f"_native_b3_{name}_{id(tmp_path)}"
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_positions(ddir: str, positions: list[PositionRecord]) -> None:
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "positions.json"), "w") as f:
        json.dump({"positions": [p.to_dict() for p in positions]}, f)


def _write_risk(ddir: str, risk: RiskState) -> None:
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "risk_state.json"), "w") as f:
        json.dump(risk.to_dict(), f)


def _open_position(pid="p1", underlying="SPY", entry_price=1.5, qty_open=2, qty_original=2) -> PositionRecord:
    return PositionRecord(
        id=pid,
        underlying=underlying,
        direction="bullish",
        structure="single",
        legs=[{"symbol": f"{underlying}261002C00580000", "side": "buy"}],
        status="open",
        qty_original=qty_original,
        qty_open=qty_open,
        entry_price=entry_price,
        entry_time=time.time() - 3600,
        entry_client_order_ids=["DPOF-entry-1"],
        thesis="NOFA composite strongly bullish, VWAP confirms",
        entry_decision={"provider": "opus", "conviction": 0.72},
        mode="paper",
    )


def _closed_position(pid="p2", underlying="QQQ", realized_pnl=42.5) -> PositionRecord:
    return PositionRecord(
        id=pid,
        underlying=underlying,
        direction="bearish",
        structure="single",
        legs=[{"symbol": f"{underlying}261002P00480000", "side": "buy"}],
        status="closed",
        qty_original=3,
        qty_open=0,
        entry_price=2.1,
        entry_time=time.time() - 7200,
        entry_client_order_ids=["DPOF-entry-2"],
        exit_client_order_ids=["DPOF-exit-2"],
        realized_pnl=realized_pnl,
        closed_time=time.time() - 60,
        exit_reason="profit_tier_1",
        mode="paper",
    )


# ---------------------------------------------------------------------------
# agent/dpof/readmodel.py wiring — real end-to-end via a tmp CLIO_DPOF_DATA_DIR
# ---------------------------------------------------------------------------


def test_readmodel_reads_real_tmp_data_dir(monkeypatch, tmp_path):
    ddir = str(tmp_path / "dpof")
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", ddir)
    monkeypatch.setenv("CLIO_DPOF_MODE", "paper")
    _write_positions(ddir, [_open_position(), _closed_position()])
    _write_risk(ddir, RiskState(day="2026-09-24", hard_halted=False, realized_pnl_today=42.5))

    opens = readmodel.open_positions()
    assert [p.id for p in opens] == ["p1"]
    snap = readmodel.snapshot()
    assert snap["mode"] == "paper"
    assert snap["open_count"] == 1
    assert snap["halt_file"] is False

    assert readmodel.write_halt_file("test halt", ddir=ddir) is True
    assert readmodel.halt_file_present(ddir=ddir) is True
    assert readmodel.clear_halt_file(ddir=ddir) is True
    assert readmodel.halt_file_present(ddir=ddir) is False


# ---------------------------------------------------------------------------
# routes/portfolio.py — tax lots derived from DPOF, not Hydrinia position_log
# ---------------------------------------------------------------------------


def test_portfolio_tax_lots_translate_dpof_positions(monkeypatch, tmp_path):
    portfolio = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/portfolio.py", "portfolio")

    positions = [_open_position(qty_open=2, entry_price=1.5), _closed_position(realized_pnl=42.5)]
    legacy = portfolio._dpof_positions_to_legacy_lot_dict(positions)
    assert set(legacy["positions"].keys()) == {"p1", "p2"}
    assert legacy["positions"]["p1"]["is_options"] is True
    assert legacy["positions"]["p1"]["is_short"] is False
    assert legacy["positions"]["p1"]["status"] == "open"
    assert legacy["positions"]["p2"]["status"] == "closed"
    assert legacy["positions"]["p2"]["pnl"] == 42.5

    result = portfolio._build_tax_lots_from_positions(legacy, limit=10, now=time.time())
    assert result["source"] == "clio_dpof_ledger"
    assert "CLIO's own DPOF ledger" in result["warning"]
    lot = next(lot for lot in result["lots"] if lot["lot_id"] == "p1")
    # 2 contracts * $1.50 * 100 multiplier (options) = $300 estimated cost basis
    assert lot["estimated_cost_basis"] == 300.0
    assert lot["source"] == "clio_dpof_ledger"


def test_portfolio_tax_lots_skips_pending_and_cancelled_dpof_positions(monkeypatch, tmp_path):
    """Pending (order not yet filled) and cancelled DPOF positions never held
    real risk and must not show up as phantom 'open' tax lots (pass-2 fix:
    portfolio.py previously mapped every non-closed status to 'open')."""
    portfolio = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/portfolio.py", "portfolio_pending")

    pending = _open_position(pid="p3", qty_open=0, qty_original=0)
    pending.status = "pending"
    pending.entry_price = None
    cancelled = _open_position(pid="p4", qty_open=0, qty_original=1)
    cancelled.status = "cancelled"

    positions = [_open_position(), _closed_position(), pending, cancelled]
    legacy = portfolio._dpof_positions_to_legacy_lot_dict(positions)
    assert set(legacy["positions"].keys()) == {"p1", "p2"}


def test_portfolio_tax_lots_route_reads_real_dpof_dir(monkeypatch, tmp_path):
    portfolio = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/portfolio.py", "portfolio_route")
    ddir = str(tmp_path / "dpof")
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", ddir)
    _write_positions(ddir, [_open_position()])
    _write_risk(ddir, RiskState())

    result = asyncio.run(portfolio.get_tax_lots(limit=80, status="all", _=None))
    assert result["total_count"] == 1
    assert result["lots"][0]["lot_id"] == "p1"
    assert result["source_path"].endswith("positions.json")


# ---------------------------------------------------------------------------
# routes/trades.py — enrichment from DPOF client_order_id chains
# ---------------------------------------------------------------------------


def test_trades_dpof_client_order_index(monkeypatch, tmp_path):
    trades = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/trades.py", "trades")
    pos = _closed_position()
    monkeypatch.setattr(readmodel, "load_positions", lambda: [pos])

    index = trades._dpof_client_order_index()
    assert index["DPOF-entry-2"] is pos
    assert index["DPOF-exit-2"] is pos
    assert "unrelated-order" not in index


# ---------------------------------------------------------------------------
# routes/risk.py — theses/risk/regime repointed, market-structure/decisions/
# alerts gated to an empty "hydrinia_retired" payload when bridge is off
# ---------------------------------------------------------------------------


def test_risk_theses_from_dpof_open_positions(monkeypatch, tmp_path):
    risk = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/risk.py", "risk")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    pos = _open_position()
    monkeypatch.setattr(readmodel, "open_positions", lambda: [pos])

    async def no_broker():
        return None, {"source": "alpaca_trading_cached", "stale": True}

    monkeypatch.setattr(risk, "_cached_broker_state", no_broker)
    result = asyncio.run(risk.get_theses(_=None))
    assert result["source"] == "clio_dpof"
    thesis = result["theses"][0]
    assert thesis["symbol"] == "SPY"
    assert thesis["direction"] == "BULLISH"
    assert thesis["dpof_id"] == "p1"
    assert "NOFA composite" in thesis["thesis"]
    assert thesis["conviction"] == 0.72


def test_risk_get_risk_is_dpof_native(monkeypatch, tmp_path):
    risk = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/risk.py", "risk2")

    async def no_broker():
        return None, {"source": "alpaca_trading_cached", "stale": True}

    monkeypatch.setattr(risk, "_cached_broker_state", no_broker)
    monkeypatch.setattr(readmodel, "snapshot", lambda now_epoch=None: {
        "mode": "paper", "halt_file": True, "hard_halted": False,
        "halt_reason": "", "realized_pnl_today": -5.0, "realized_pnl_week": 10.0,
        "entries_today": 1, "open_count": 0, "closed_today_count": 1, "closed_today_wins": 0,
    })
    result = asyncio.run(risk.get_risk(_=None))
    assert result["emergency_stop"] is True  # halt_file True -> emergency_stop True
    assert result["daily_metrics"]["source"] == "clio_dpof"
    assert result["daily_metrics"]["mode"] == "paper"
    assert result["dpof"]["halt_file"] is True


def test_risk_regime_prefers_clio_native(monkeypatch, tmp_path):
    risk = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/risk.py", "risk_regime")

    fake_module = type(sys)("agent.integrations.market_regime")
    fake_module.get_market_regime = lambda max_age_seconds=None: {
        "regime": "BULL", "confidence": 0.8, "vix": 14.2,
        "spy_close": 620.1, "spy_sma20": 610.0, "spy_sma50": 600.0,
        "as_of": "2026-09-24T12:00:00+00:00", "source": "clio_native",
    }
    monkeypatch.setitem(sys.modules, "agent.integrations.market_regime", fake_module)

    result = asyncio.run(risk.get_regime(_=None))
    assert result["regime"] == "BULL"
    assert result["confidence_source"] == "clio_native"
    assert result["source"] == "clio_native"
    assert result["vix"] == 14.2


def test_risk_regime_falls_back_to_unknown_when_bridge_disabled_and_native_missing(monkeypatch, tmp_path):
    risk = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/risk.py", "risk_regime2")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

    async def _no_native():
        return None

    monkeypatch.setattr(risk, "_clio_native_regime", _no_native)
    result = asyncio.run(risk.get_regime(_=None))
    assert result["regime"] == "UNKNOWN"
    assert result["source"] == "hydrinia_retired"


def test_risk_market_structure_and_decisions_and_alerts_empty_when_bridge_disabled(monkeypatch, tmp_path):
    risk = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/risk.py", "risk_gated")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

    ms = asyncio.run(risk.get_market_structure(_=None))
    assert ms["source"] == "hydrinia_retired"
    assert ms["gex"] == []

    decisions = asyncio.run(risk.get_decisions(limit=20, _=None))
    assert decisions["source"] == "hydrinia_retired"
    assert decisions["decisions"] == []

    alerts = asyncio.run(risk.get_alerts(limit=20, _=None))
    assert alerts["source"] == "hydrinia_retired"
    assert alerts["alerts"] == []
    assert alerts["reason_code"] == "HYDRINIA_RETIRED"


def test_risk_market_structure_and_alerts_run_legacy_chain_when_bridge_enabled(monkeypatch, tmp_path):
    risk = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/risk.py", "risk_bridge_on")
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    monkeypatch.setattr(risk, "read_json", lambda *_a, **_k: {})
    monkeypatch.setattr(risk.os.path, "isfile", lambda _p: False)
    monkeypatch.setattr(risk.os.path, "exists", lambda _p: False)

    ms = asyncio.run(risk.get_market_structure(_=None))
    assert ms["source"] == "hydrinia_legacy_bridge"

    alerts = asyncio.run(risk.get_alerts(limit=20, _=None))
    assert alerts["source"] == "hydrinia_legacy_bridge"


# ---------------------------------------------------------------------------
# routes/actions.py — halt/resume/status are DPOF-native; Hydrinia touched
# only when the bridge is explicitly enabled
# ---------------------------------------------------------------------------


def test_actions_status_is_dpof_native(monkeypatch, tmp_path):
    actions = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/actions.py", "actions_status")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    ddir = str(tmp_path / "dpof")
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", ddir)
    monkeypatch.setenv("CLIO_DPOF_MODE", "paper")
    _write_positions(ddir, [_open_position()])
    _write_risk(ddir, RiskState())

    async def fake_run_in_executor(fn, *args):
        return fn(*args)

    monkeypatch.setattr(actions, "run_in_executor", fake_run_in_executor)

    result = asyncio.run(actions.get_trading_status(_=None))
    assert result["source"] == "clio_dpof"
    assert result["mode"] == "paper"
    assert result["open_count"] == 1
    assert result["hydrinia_active"] is False  # bridge disabled -> never checked
    assert result["emergency_stop"] is False


def test_actions_halt_and_resume_touch_only_dpof_when_bridge_disabled(monkeypatch, tmp_path):
    actions = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/actions.py", "actions_halt")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    ddir = str(tmp_path / "dpof")
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", ddir)
    _write_risk(ddir, RiskState())

    async def fake_run_in_executor(fn, *args):
        return fn(*args)

    monkeypatch.setattr(actions, "run_in_executor", fake_run_in_executor)

    def _boom(*_a, **_k):
        raise AssertionError("must not touch Hydrinia when bridge disabled")

    monkeypatch.setattr(actions.subprocess, "run", _boom)

    result = asyncio.run(actions.halt_trading(_=None))
    assert result["dpof_halt_file"] is True
    assert result["hydrinia_stopped"] is False
    assert readmodel.halt_file_present(ddir=ddir) is True
    assert not os.path.exists(actions._EMERGENCY_STOP_PATH)

    result = asyncio.run(actions.resume_trading(_=None))
    assert result["dpof_halt_cleared"] is True
    assert result["hydrinia_started"] is False
    assert readmodel.halt_file_present(ddir=ddir) is False


def test_actions_restart_service_respects_bridge_gate(monkeypatch, tmp_path):
    actions = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/actions.py", "actions_restart")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

    async def fake_run_in_executor(fn, *args):
        return fn(*args)

    monkeypatch.setattr(actions, "run_in_executor", fake_run_in_executor)

    with pytest.raises(actions.HTTPException) as exc:
        asyncio.run(actions.restart_service("hydrinia", _=None))
    assert exc.value.status_code == 400

    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")

    def fake_subprocess_run(cmd, **_kwargs):
        class Result:
            returncode = 0
        assert cmd[-1] == "hydrinia"
        return Result()

    monkeypatch.setattr(actions.subprocess, "run", fake_subprocess_run)
    result = asyncio.run(actions.restart_service("hydrinia", _=None))
    assert result["restarted"] is True


# ---------------------------------------------------------------------------
# routes/directives.py + routes/strategies.py — Hydrinia writes refused (409)
# when the bridge is disabled, never silently no-op'd
# ---------------------------------------------------------------------------


def test_directives_post_directive_and_control_refused_when_bridge_disabled(monkeypatch, tmp_path):
    directives = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/directives.py", "directives")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

    req = directives.DirectiveRequest(command="buy AAPL")
    with pytest.raises(directives.HTTPException) as exc:
        asyncio.run(directives.issue_directive(req, _=None))
    assert exc.value.status_code == 409

    ctl = directives.ServiceControlRequest(action="restart", service="hydrinia")
    with pytest.raises(directives.HTTPException) as exc2:
        asyncio.run(directives.service_control(ctl, _=None))
    assert exc2.value.status_code == 409


def test_directives_dispatch_awaits_async_dispatch_directive_with_real_override_event(monkeypatch, tmp_path):
    """dispatch_directive() is `async def` — must be awaited directly, not run
    via run_in_executor (which would hand back an un-awaited coroutine), and it
    reads override_event.command_text/.timestamp, so it needs a real
    OverrideEvent, not a plain dict (S/O coordinator fix)."""
    directives = _load_fresh(
        monkeypatch, tmp_path, "mission_control/backend/routes/directives.py", "directives_dispatch"
    )
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")

    import agent.integrations.directive_delivery as directive_delivery
    from agent.override_directive import OverrideEvent

    captured = {}

    async def fake_dispatch_directive(override_event, raw_message):
        assert isinstance(override_event, OverrideEvent)
        captured["command_text"] = override_event.command_text
        captured["timestamp"] = override_event.timestamp
        captured["raw_message"] = raw_message
        return "Directive dispatched.\n  ID: DIR-test"

    monkeypatch.setattr(directive_delivery, "dispatch_directive", fake_dispatch_directive)

    req = directives.DirectiveRequest(command="buy AAPL")
    result = asyncio.run(directives.issue_directive(req, _=None))

    assert result["status"] == "dispatched"
    assert result["result"] == "Directive dispatched.\n  ID: DIR-test"
    assert captured["command_text"] == "buy AAPL"
    assert captured["raw_message"] == "buy AAPL"
    assert captured["timestamp"]  # non-empty ISO timestamp set


def test_directives_dispatch_bridge_off_message_surfaces_when_bridge_flips_mid_flight(monkeypatch, tmp_path):
    """If dispatch_directive's own internal bridge check ever fires (e.g. a
    race with the route's pre-check), its "not delivered" string must pass
    through unmodified rather than being swallowed as an error."""
    directives = _load_fresh(
        monkeypatch, tmp_path, "mission_control/backend/routes/directives.py", "directives_dispatch_race"
    )
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")

    import agent.integrations.directive_delivery as directive_delivery

    async def fake_dispatch_directive(_override_event, _raw_message):
        return "Hydrinia bridge is off — directive not delivered."

    monkeypatch.setattr(directive_delivery, "dispatch_directive", fake_dispatch_directive)

    req = directives.DirectiveRequest(command="consider bearish pressure")
    result = asyncio.run(directives.issue_directive(req, _=None))
    assert result["result"] == "Hydrinia bridge is off — directive not delivered."


def test_strategies_push_to_hydrinia_refused_when_bridge_disabled(monkeypatch, tmp_path):
    strategies = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/strategies.py", "strategies")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

    req = strategies.HydriniaPush(title="Test", content="body")
    with pytest.raises(strategies.HTTPException) as exc:
        asyncio.run(strategies.push_to_hydrinia(req, _=None))
    assert exc.value.status_code == 409
    # And nothing was written.
    assert not os.path.isdir(strategies._BRIDGE_STRATEGIES) or not os.listdir(strategies._BRIDGE_STRATEGIES)


# ---------------------------------------------------------------------------
# routes/files.py — Hydrinia roots read-gated by the bridge, never writable
# ---------------------------------------------------------------------------


def test_files_hydrinia_roots_gated(monkeypatch, tmp_path):
    files = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/files.py", "files")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

    with pytest.raises(files.HTTPException) as exc:
        files._resolve_safe("/home/agent/tradingbot/state/position_log.json")
    assert exc.value.status_code == 403

    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    resolved = files._resolve_safe("/home/agent/tradingbot/state/position_log.json")
    assert str(resolved) == "/home/agent/tradingbot/state/position_log.json"

    # Even with the bridge on, writes into Hydrinia's roots are never allowed.
    with pytest.raises(files.HTTPException) as exc2:
        files._resolve_safe("/home/agent/tradingbot/state/position_log.json", for_write=True)
    assert exc2.value.status_code == 403


# ---------------------------------------------------------------------------
# routes/dpof.py — new CLIO-native read-only routes
# ---------------------------------------------------------------------------


def test_dpof_routes_status_positions_decisions(monkeypatch, tmp_path):
    dpof = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/dpof.py", "dpof")

    async def fake_run_in_executor(fn, *args):
        return fn(*args)

    monkeypatch.setattr(dpof, "run_in_executor", fake_run_in_executor)
    monkeypatch.setattr(readmodel, "snapshot", lambda now_epoch=None: {"mode": "paper", "open_count": 1})
    monkeypatch.setattr(readmodel, "open_positions", lambda: [_open_position()])
    monkeypatch.setattr(readmodel, "closed_today", lambda now_epoch=None: [_closed_position()])
    monkeypatch.setattr(readmodel, "recent_decisions", lambda limit=20: [{"symbol": "SPY", "action": "enter"}])

    status = asyncio.run(dpof.dpof_status(_=None))
    assert status["dpof"]["mode"] == "paper"
    assert status["source"] == "clio_dpof"

    positions = asyncio.run(dpof.dpof_positions(_=None))
    assert len(positions["open"]) == 1
    assert positions["open"][0]["id"] == "p1"
    assert len(positions["closed_today"]) == 1
    assert positions["open_lines"] and "SPY" in positions["open_lines"][0]

    decisions = asyncio.run(dpof.dpof_decisions(limit=50, _=None))
    assert decisions["count"] == 1
    assert decisions["decisions"][0]["symbol"] == "SPY"


def test_dpof_status_route_registered_in_api_key_allowlist():
    import mission_control.backend.auth as real_auth

    assert "/api/dpof/" in real_auth._API_KEY_ALLOWED_READ_PREFIXES


# ---------------------------------------------------------------------------
# deps.py / config.py — small pure-Python bridge gates
# ---------------------------------------------------------------------------


def test_load_symbols_universe_dpof_native_by_default(monkeypatch, tmp_path):
    deps = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/deps.py", "deps")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    monkeypatch.setenv("CLIO_DPOF_UNDERLYINGS", "SPY,QQQ")

    universe = deps.load_symbols_universe()
    assert universe == {"SPY", "QQQ"}


def test_load_symbols_universe_falls_back_to_hydrinia_symbols_txt_when_bridge_enabled(monkeypatch, tmp_path):
    deps = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/deps.py", "deps2")
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    (tmp_path / "symbols.txt").write_text("AAPL,NASDAQ\nMSFT\n# comment\n")
    deps.mc_config.tradingbot_root = str(tmp_path)

    universe = deps.load_symbols_universe()
    assert universe == {"AAPL", "MSFT"}


def test_agents_routes_empty_and_marked_when_bridge_disabled(monkeypatch, tmp_path):
    agents = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/agents.py", "agents")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

    result = asyncio.run(agents.get_agents(_=None))
    assert result["agents"] == {}
    assert result["total_agents"] == 0
    assert result["reason_code"] == "HYDRINIA_RETIRED"

    named = asyncio.run(agents.get_agents(name="Aidan Tan", _=None))
    assert named["agent"] == "Aidan Tan"
    assert named["metrics"] == {}

    consensus = asyncio.run(agents.get_consensus(limit=20, _=None))
    assert consensus == {"consensus": [], "count": 0, "source": "hydrinia_retired"}


def test_effective_allowed_services_gates_hydrinia(monkeypatch):
    import mission_control.backend.config as real_config

    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    assert "hydrinia" not in real_config.effective_allowed_services()

    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    assert "hydrinia" in real_config.effective_allowed_services()


# ---------------------------------------------------------------------------
# ws/logs.py, ws/alerts.py — Hydrinia unit/source gated by the bridge
# ---------------------------------------------------------------------------


def test_ws_logs_default_unit_and_bridge_gated_allowlist(monkeypatch, tmp_path):
    ws_logs = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/ws/logs.py", "ws_logs")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    assert "hydrinia" not in ws_logs._ALLOWED_UNITS
    assert "hydrinia" not in (ws_logs._ALLOWED_UNITS | set())
    assert ws_logs._hydrinia_bridge_enabled() is False

    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    assert ws_logs._hydrinia_bridge_enabled() is True
    allowed = ws_logs._ALLOWED_UNITS | ws_logs._BRIDGE_ONLY_UNITS
    assert "hydrinia" in allowed


def test_ws_alerts_bridge_gate_helper(monkeypatch, tmp_path):
    ws_alerts = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/ws/alerts.py", "ws_alerts")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    assert ws_alerts._hydrinia_bridge_enabled() is False
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    assert ws_alerts._hydrinia_bridge_enabled() is True


# ---------------------------------------------------------------------------
# routes/system.py — DPOF-native bot-status, bridge-gated managed services
# ---------------------------------------------------------------------------


def _load_fresh_with_psutil_stub(monkeypatch, tmp_path, rel_path: str, name: str):
    import types

    monkeypatch.setitem(sys.modules, "psutil", types.ModuleType("psutil"))
    return _load_fresh(monkeypatch, tmp_path, rel_path, name)


def test_system_managed_services_and_bot_status_are_dpof_native(monkeypatch, tmp_path):
    system = _load_fresh_with_psutil_stub(
        monkeypatch, tmp_path, "mission_control/backend/routes/system.py", "system"
    )
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    assert "hydrinia" not in system._managed_services()
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    assert "hydrinia" in system._managed_services()
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

    monkeypatch.setattr(readmodel, "load_risk", lambda: RiskState(hard_halted=False))
    monkeypatch.setattr(readmodel, "halt_file_present", lambda: True)
    monkeypatch.setattr(readmodel, "load_positions", lambda: [_open_position(), _closed_position()])
    monkeypatch.setattr(readmodel, "open_positions", lambda: [_open_position()])

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        class Proc:
            async def communicate(self):
                return b"active\n", b""
        return Proc()

    monkeypatch.setattr(system.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(system, "run_in_executor", lambda fn, *a: asyncio.sleep(0, result=fn(*a)))

    result = asyncio.run(system.bot_status(_=None))
    assert result["emergency_stop"] is True  # halt_file True
    assert result["dpof_halt_file"] is True
    assert result["open_positions"] == 1
    assert result["total_positions"] == 2
    assert "hydrinia" not in result["services"]


# ---------------------------------------------------------------------------
# routes/audit.py — DPOF-native audit context (no Hydrinia journalctl/state)
# ---------------------------------------------------------------------------


def test_audit_system_prompt_and_context_are_dpof_native(monkeypatch, tmp_path):
    audit = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/audit.py", "audit")
    assert "Hydrinia" not in audit._AUDIT_SYSTEM_PROMPT
    assert "DPOF" in audit._AUDIT_SYSTEM_PROMPT

    monkeypatch.setattr(readmodel, "snapshot", lambda now_epoch=None: {"mode": "paper"})
    monkeypatch.setattr(readmodel, "recent_decisions", lambda limit=20: [])
    monkeypatch.setattr(readmodel, "load_risk", lambda: RiskState())

    req = audit.AuditRequest(scope="code", timeframe="4h")
    run_id = "test-run"
    audit._audit_runs[run_id] = {"status": "running", "started": time.time(), "scope": "code", "focus": None, "result": None}

    class FakeContent:
        type = "text"  # S214: first_text() filters on block type, as SDK blocks carry it
        text = "audit ok"

    class FakeResponse:
        content = [FakeContent()]

    class FakeMessages:
        def create(self, **_kwargs):
            return FakeResponse()

    class FakeClient:
        messages = FakeMessages()

    import types
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda **kw: FakeClient()
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    fake_config_mod = types.ModuleType("agent.config")
    fake_config_mod.config = types.SimpleNamespace(
        default_model="claude-test",
        anthropic_client_kwargs=lambda timeout=120, **extra: {"api_key": "test-key", "timeout": timeout, **extra},
    )
    monkeypatch.setitem(sys.modules, "agent.config", fake_config_mod)

    asyncio.run(audit._run_audit(run_id, req))
    assert audit._audit_runs[run_id]["status"] == "complete"
    assert audit._audit_runs[run_id]["result"] == "audit ok"


# ---------------------------------------------------------------------------
# routes/briefing.py — DPOF-native metadata/regime/emergency_stop
# ---------------------------------------------------------------------------


def test_briefing_metadata_and_regime_and_emergency_stop_are_dpof_native(monkeypatch, tmp_path):
    import types
    monkeypatch.setitem(sys.modules, "psutil", types.ModuleType("psutil"))
    sys.modules["psutil"].virtual_memory = lambda: types.SimpleNamespace(percent=10.0)
    sys.modules["psutil"].cpu_percent = lambda interval=0.3: 5.0
    sys.modules["psutil"].disk_usage = lambda _path: types.SimpleNamespace(percent=20.0)

    briefing = _load_fresh(monkeypatch, tmp_path, "mission_control/backend/routes/briefing.py", "briefing")
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

    class Account:
        equity = "20000"
        cash = "5000"
        buying_power = "10000"
        portfolio_value = "20000"
        last_equity = "19500"

    class Client:
        def get_account(self):
            return Account()

        def get_all_positions(self):
            return []

    monkeypatch.setattr(briefing, "get_trading_client", lambda: Client())

    import types
    fake_regime_mod = types.ModuleType("agent.integrations.market_regime")
    fake_regime_mod.get_market_regime = lambda max_age_seconds=None: {"regime": "BULL", "confidence": 0.9}
    monkeypatch.setitem(sys.modules, "agent.integrations.market_regime", fake_regime_mod)

    monkeypatch.setattr(readmodel, "halt_file_present", lambda: False)
    monkeypatch.setattr(readmodel, "load_risk", lambda: RiskState(hard_halted=True))

    result = asyncio.run(briefing.full_briefing(_=None))
    assert "Hydrinia" not in result["system"]
    assert "DPOF" in result["mode"]
    assert result["market_regime"] == {"regime": "BULL", "confidence": 0.9}
    assert result["active_directives"] == []
    assert result["system_health"]["emergency_stop"] is True  # hard_halted True
