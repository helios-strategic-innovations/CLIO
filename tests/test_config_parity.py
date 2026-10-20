"""A-4 (P1): Config defaults-parity guard.

Targets ``agent/config.py``. With the environment cleared, ``Config.from_env()``
must reproduce every declared dataclass-field default — scalar defaults AND
``default_factory`` values alike. This single loop catches the whole
"dataclass default drifted from ``from_env()`` default" class that the project
CLAUDE.md warns about ("Dataclass defaults MUST match from_env() defaults").

Pure-stdlib import chain — safe to collect without runtime deps installed.
"""
import dataclasses
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Config  # noqa: E402


def _expected_default(f):
    """Return (value, has_default) for a dataclass Field, handling factories."""
    if f.default is not dataclasses.MISSING:
        return f.default, True
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory(), True
    return None, False


def test_from_env_matches_every_declared_default(monkeypatch):
    # Clear the environment so from_env() takes every fallback default (rebinding
    # os.environ to {} makes os.getenv see an empty environment; monkeypatch
    # restores the real mapping at teardown).
    monkeypatch.setattr(os, "environ", {})
    cfg = Config.from_env()

    mismatches = []
    for f in dataclasses.fields(Config):
        expected, has_default = _expected_default(f)
        if not has_default:
            continue
        actual = getattr(cfg, f.name)
        if actual != expected:
            mismatches.append(f"{f.name}: dataclass_default={expected!r} from_env={actual!r}")

    assert not mismatches, (
        "Config.from_env() drifted from dataclass field defaults:\n  " + "\n  ".join(mismatches)
    )


def test_import_time_singleton_is_a_config():
    # The module-level `config` singleton exists and is a Config instance.
    from agent.config import config
    assert isinstance(config, Config)
