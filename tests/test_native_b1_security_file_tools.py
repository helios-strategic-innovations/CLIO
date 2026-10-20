"""
Tests for agent/security.py + agent/tools/file_tools.py Hydrinia-bridge gating
(B1 scope).

Hydrinia is permanently off (operator direction, 2026-09-24):
- security.py: /home/agent/tradingbot/ drops out of the effective allowed
  file-path set (validate_tool_call's file_path check) unless the bridge is
  explicitly re-enabled. modify_hydrinia is in BLOCKED_IN_AUTO_MODE (gap fix).
- file_tools.py: DEFAULT_BASE is /home/agent/openclaw; the retired tradingbot
  root is only in the resolvable/allowed root set when the bridge is on.

No network, no real writes: only path resolution/validation is exercised.
"""

import os

import agent.security as security
import agent.tools.file_tools as file_tools


# ---------------------------------------------------------------------------
# security.py
# ---------------------------------------------------------------------------

class TestSecurityAllowedFilePaths:
    def test_tradingbot_excluded_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        paths = security._current_allowed_file_paths()
        assert "/home/agent/tradingbot/" not in paths
        assert "/home/agent/openclaw/" in paths

    def test_tradingbot_included_when_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        paths = security._current_allowed_file_paths()
        assert "/home/agent/tradingbot/" in paths

    def test_validate_tool_call_rejects_tradingbot_path_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        ok, reason = security.validate_tool_call(
            "read_file", {"file_path": "/home/agent/tradingbot/hydrinia.py"},
        )
        assert ok is False
        assert "outside allowed directories" in reason

    def test_validate_tool_call_accepts_tradingbot_path_when_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        ok, reason = security.validate_tool_call(
            "read_file", {"file_path": "/home/agent/tradingbot/hydrinia.py"},
        )
        assert ok is True

    def test_validate_tool_call_accepts_openclaw_path_regardless_of_bridge(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        ok, _ = security.validate_tool_call(
            "read_file", {"file_path": "/home/agent/openclaw/agent/config.py"},
        )
        assert ok is True


class TestBlockedInAutoMode:
    def test_modify_hydrinia_blocked_in_auto_mode(self):
        assert "modify_hydrinia" in security.BLOCKED_IN_AUTO_MODE

    def test_control_hydrinia_blocked_in_auto_mode(self):
        assert "control_hydrinia" in security.BLOCKED_IN_AUTO_MODE


class TestKnownToolsFallbackConsistency:
    def test_fallback_set_includes_dpof_status(self):
        """_build_known_tools()'s hardcoded fallback (used only if the real
        get_all_tools() import fails) must stay in sync with the new tool."""
        import inspect
        src = inspect.getsource(security._build_known_tools)
        assert "get_dpof_status" in src


# ---------------------------------------------------------------------------
# file_tools.py
# ---------------------------------------------------------------------------

class TestFileToolsDefaultBase:
    def test_default_base_is_openclaw(self):
        assert file_tools.DEFAULT_BASE == "/home/agent/openclaw"

    def test_allowed_roots_excludes_tradingbot_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        roots = file_tools._allowed_roots()
        assert "/home/agent/tradingbot/" not in roots
        assert "/home/agent/openclaw/" in roots

    def test_allowed_roots_includes_tradingbot_when_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        roots = file_tools._allowed_roots()
        assert "/home/agent/tradingbot/" in roots


class TestResolvePath:
    def test_relative_agent_path_resolves_under_openclaw(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        ok, resolved, err = file_tools._resolve_path("agent/config.py")
        assert ok is True
        assert resolved == "/home/agent/openclaw/agent/config.py"

    def test_absolute_tradingbot_path_rejected_when_bridge_off(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        ok, resolved, err = file_tools._resolve_path("/home/agent/tradingbot/hydrinia.py")
        assert ok is False
        assert "outside allowed directories" in err

    def test_absolute_tradingbot_path_accepted_when_bridge_on(self, monkeypatch):
        monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
        ok, resolved, err = file_tools._resolve_path("/home/agent/tradingbot/hydrinia.py")
        assert ok is True
        assert resolved == "/home/agent/tradingbot/hydrinia.py"

    def test_path_traversal_lookalike_still_rejected(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        ok, _, err = file_tools._resolve_path("/home/agent/openclaw-evil/secrets.txt")
        assert ok is False

    def test_secret_file_still_blocked_under_openclaw(self, monkeypatch):
        monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
        ok, _, err = file_tools._resolve_path("/home/agent/openclaw/.env")
        assert ok is False
        assert "secret file" in err.lower()
