"""
Tests for the Hydrinia-bridge tool gating in agent/tools/{definitions,executor,
hydrinia_tools,bot_control}.py (B1 scope).

Hydrinia is permanently off (operator direction, 2026-09-24). Its monitoring
(get_hydrinia_*) and control (control_hydrinia, modify_hydrinia) tools must:
- be OMITTED from the tool list the LLM sees when the bridge is disabled
  (default), and INCLUDED when CLIO_HYDRINIA_BRIDGE_ENABLED=1;
- return a clear "RETIRED" string (never execute the real handler) when
  called while the bridge is disabled, both through agent.tools.executor's
  dispatch table AND when a caller imports the handler directly (defense in
  depth — agent/integrations/scheduled_tasks.py does exactly that for
  handle_hydrinia_portfolio, bypassing execute_tool()).

The new get_dpof_status tool must always be present and never gated.
"""

import asyncio

import pytest

import agent.tools.definitions as definitions
import agent.tools.executor as executor
import agent.tools.hydrinia_tools as hydrinia_tools
import agent.tools.bot_control as bot_control


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# definitions.py — tool list gating
# ---------------------------------------------------------------------------

class TestDefinitionsGating:
    def test_bridge_off_by_default_omits_hydrinia_tools(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        names = {t["name"] for t in definitions.get_all_tools()}
        for gated in (
            "get_hydrinia_logs", "get_hydrinia_state", "get_hydrinia_status",
            "get_hydrinia_portfolio", "get_hydrinia_performance",
            "control_hydrinia", "modify_hydrinia",
        ):
            assert gated not in names, f"{gated} should be omitted when bridge is off"

    def test_bridge_on_includes_hydrinia_tools(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        names = {t["name"] for t in definitions.get_all_tools()}
        for gated in (
            "get_hydrinia_logs", "get_hydrinia_state", "get_hydrinia_status",
            "get_hydrinia_portfolio", "get_hydrinia_performance",
            "control_hydrinia", "modify_hydrinia",
        ):
            assert gated in names, f"{gated} should be present when bridge is on"

    def test_get_dpof_status_always_present(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        names_off = {t["name"] for t in definitions.get_all_tools()}
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        names_on = {t["name"] for t in definitions.get_all_tools()}
        assert "get_dpof_status" in names_off
        assert "get_dpof_status" in names_on

    def test_get_bot_processes_always_present_both_states(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        assert "get_bot_processes" in {t["name"] for t in definitions.get_all_tools()}
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        assert "get_bot_processes" in {t["name"] for t in definitions.get_all_tools()}

    def test_get_dpof_status_schema_has_section_enum(self):
        tool = next(t for t in definitions.get_all_tools() if t["name"] == "get_dpof_status")
        props = tool["input_schema"]["properties"]
        assert set(props["section"]["enum"]) == {"status", "positions", "decisions", "config"}
        assert tool["input_schema"]["required"] == []


# ---------------------------------------------------------------------------
# executor.py — dispatch-level gate ("never execute")
# ---------------------------------------------------------------------------

class TestExecutorDispatchGate:
    @pytest.mark.parametrize("tool_name", [
        "get_hydrinia_logs", "get_hydrinia_state", "get_hydrinia_status",
        "get_hydrinia_portfolio", "get_hydrinia_performance",
        "control_hydrinia", "modify_hydrinia",
    ])
    def test_gated_tool_returns_retired_string_when_bridge_off(self, monkeypatch, tool_name):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        result = _run(executor.execute_tool(tool_name, {"action": "status"}))
        assert result.startswith("RETIRED:")
        assert "Hydrinia" in result

    def test_gated_tool_never_reaches_handler_when_bridge_off(self, monkeypatch):
        """The dispatch gate must short-circuit BEFORE the handler is looked up
        — patch the real handler to raise, and confirm it's never called."""
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

        async def _boom(p):
            raise AssertionError("handler must not be called when bridge is off")

        monkeypatch.setattr(hydrinia_tools, "handle_hydrinia_status", _boom)
        result = _run(executor.execute_tool("get_hydrinia_status", {}))
        assert result.startswith("RETIRED:")

    def test_get_dpof_status_not_gated(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        result = _run(executor.execute_tool("get_dpof_status", {"section": "status"}))
        assert not result.startswith("RETIRED:")


# ---------------------------------------------------------------------------
# hydrinia_tools.py / bot_control.py — handler-level defense-in-depth gate
# (covers callers that import the handler directly, bypassing execute_tool())
# ---------------------------------------------------------------------------

class TestHandlerLevelGate:
    def test_handle_hydrinia_portfolio_retired_when_bridge_off_called_directly(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        result = _run(hydrinia_tools.handle_hydrinia_portfolio({}))
        assert result.startswith("RETIRED:")

    def test_handle_hydrinia_status_retired_when_bridge_off_called_directly(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        result = _run(hydrinia_tools.handle_hydrinia_status({}))
        assert result.startswith("RETIRED:")

    def test_handle_control_hydrinia_retired_when_bridge_off_called_directly(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        result = _run(bot_control.handle_control_hydrinia({"action": "start"}))
        assert result.startswith("RETIRED:")

    def test_handle_control_hydrinia_never_popens_when_bridge_off(self, monkeypatch):
        """Regression guard for the 'could REVIVE Hydrinia' risk: _start_hydrinia
        must never be reached while the bridge is disabled."""
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

        async def _boom():
            raise AssertionError("_start_hydrinia must not run when bridge is off")

        monkeypatch.setattr(bot_control, "_start_hydrinia", _boom)
        result = _run(bot_control.handle_control_hydrinia({"action": "start"}))
        assert result.startswith("RETIRED:")


# ---------------------------------------------------------------------------
# bot_control.py — get_bot_processes keyword filter
# ---------------------------------------------------------------------------

class TestBotProcessesFilter:
    def test_hydrinia_excluded_from_filter_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)

        class _FakeProc:
            async def communicate(self):
                return (b"agent  123  hydrinia.py\nagent  456  moltbot main.py\n", b"")

        async def _fake_exec(*args, **kwargs):
            return _FakeProc()

        monkeypatch.setattr(bot_control.asyncio, "create_subprocess_exec", _fake_exec)
        result = _run(bot_control.handle_get_bot_processes({}))
        assert "hydrinia.py" not in result
        assert "moltbot main.py" in result

    def test_hydrinia_included_in_filter_when_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")

        class _FakeProc:
            async def communicate(self):
                return (b"agent  123  hydrinia.py\nagent  456  moltbot main.py\n", b"")

        async def _fake_exec(*args, **kwargs):
            return _FakeProc()

        monkeypatch.setattr(bot_control.asyncio, "create_subprocess_exec", _fake_exec)
        result = _run(bot_control.handle_get_bot_processes({}))
        assert "hydrinia.py" in result
        assert "moltbot main.py" in result
