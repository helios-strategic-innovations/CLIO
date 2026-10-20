"""Shared fixtures for the CLIO DPOF test suite.

Tests are plugin-free (no pytest-asyncio): async code is driven with
asyncio.run() inside ordinary test functions, matching the VPS venv.
"""

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.dpof.config import DPOFConfig, load_dpof_config  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_dpof_env(monkeypatch, tmp_path):
    """Isolate every test from the real environment and the live data dir."""
    for key in list(os.environ):
        if key.startswith("CLIO_DPOF_") or key.startswith("TYPESAFE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof"))
    yield


@pytest.fixture
def cfg(tmp_path) -> DPOFConfig:
    """Default config pointed at a temp data dir."""
    return replace(DPOFConfig(), data_dir=str(tmp_path / "dpof"))


@pytest.fixture
def env_cfg():
    """Build a config from the (cleaned + test-modified) environment."""
    return load_dpof_config
