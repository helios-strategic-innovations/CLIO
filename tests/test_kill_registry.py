"""A-2 (P0-16): KillSwitch env-parser + registry coverage.

Targets ``agent/safety/kill_registry.py`` ``KillSwitch.is_enabled`` and
``KillSwitchRegistry.is_enabled``. The parser must distinguish unset (use
default), empty string (disabled), truthy tokens (enabled) and everything else
(disabled); a present kill-file forces OFF regardless of env; an unknown switch
name is OFF (fail-safe).

Pure-stdlib import chain — safe to collect without runtime deps installed.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.safety import kill_registry as kr  # noqa: E402
from agent.safety.kill_registry import KillSwitch, KillSwitchRegistry  # noqa: E402

_ENV = "TEST_KILL_SWITCH_X"


def _switch(default=True, kill_file=""):
    return KillSwitch(name="t", env_var=_ENV, default=default, description="d", kill_file=kill_file)


def test_unset_uses_default(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert _switch(default=True).is_enabled is True
    assert _switch(default=False).is_enabled is False


def test_empty_string_is_disabled_not_default(monkeypatch):
    # Explicit empty means "disabled" — it must NOT fall back to the default
    # (default=True here) (S162 M-8).
    monkeypatch.setenv(_ENV, "")
    assert _switch(default=True).is_enabled is False


@pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE", "On", "yES"])
def test_truthy_tokens_enable(monkeypatch, val):
    monkeypatch.setenv(_ENV, val)
    assert _switch(default=False).is_enabled is True


@pytest.mark.parametrize("val", ["false", "no", "0", "off", "junk", "enabled", "2"])
def test_non_truthy_tokens_disable(monkeypatch, val):
    monkeypatch.setenv(_ENV, val)
    assert _switch(default=True).is_enabled is False


def test_kill_file_forces_off_even_when_env_true(monkeypatch, tmp_path):
    kf = tmp_path / "SWITCH_KILLED"
    kf.write_text("{}")
    monkeypatch.setenv(_ENV, "true")
    assert _switch(default=True, kill_file=str(kf)).is_enabled is False


def test_absent_kill_file_falls_through_to_env(monkeypatch, tmp_path):
    kf = tmp_path / "does_not_exist"
    monkeypatch.setenv(_ENV, "true")
    assert _switch(default=False, kill_file=str(kf)).is_enabled is True


def test_registry_unknown_switch_is_off(monkeypatch, tmp_path):
    # Route the registry's kill-switch dir to a disposable tmp dir so construction
    # never touches the real /home/agent path.
    monkeypatch.setattr(kr, "KILL_SWITCH_DIR", str(tmp_path))
    reg = KillSwitchRegistry()
    assert reg.is_enabled("this_switch_does_not_exist") is False
