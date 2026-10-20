"""Grep-provable / fresh-interpreter isolation guarantees for agent/kronos/:
importing the package (or any of its non-vendor submodules) must never pull
in `torch`, `agent.config`, or `alpaca` at import time -- CLAUDE.md's own
"import agent.kronos in a fresh subprocess does not import torch" bar.

These tests must give the same answer in a torch-free test venv AND in the
VPS serving venv, where the deploy installs torch. Every subprocess therefore
runs with a stub `torch` package first on PYTHONPATH, shadowing any real one:
- the RECORDER stub imports fine, so ANY import attempt (even a guarded
  `try: import torch`) shows up in sys.modules;
- the BLOCKER stub raises ImportError, so `python -m agent.kronos` sees torch
  as missing and must exit 3 before touching .env-driven config, Alpaca or
  data/ (with real torch present it would otherwise do a live run)."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "agent" / "kronos"


def _py_files():
    return sorted(p for p in PKG_DIR.glob("*.py"))


_STUB_MARKER = "KRONOS_TEST_TORCH_STUB"


@pytest.fixture
def recorder_stub(tmp_path):
    """A `torch` package that imports cleanly and marks itself."""
    pkg = tmp_path / "recorder" / "torch"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"{_STUB_MARKER} = True\n")
    return str(tmp_path / "recorder")


@pytest.fixture
def blocker_stub(tmp_path):
    """A `torch` package whose import fails, as if torch were not installed."""
    pkg = tmp_path / "blocker" / "torch"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise ImportError('torch blocked by test stub')\n")
    return str(tmp_path / "blocker")


def _env_with(stub_dir: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = stub_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _run(script: str, stub_dir: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
                          env=_env_with(stub_dir))


def test_recorder_stub_shadows_any_real_torch(recorder_stub):
    """Precondition for the sys.modules checks below: the subprocess imports
    the stub, not a real torch, whatever this venv has installed."""
    result = _run(f"import torch; assert getattr(torch, {_STUB_MARKER!r}, False); print('STUB_OK')", recorder_stub)
    assert result.returncode == 0 and "STUB_OK" in result.stdout, f"stderr={result.stderr!r}"


def test_blocker_stub_makes_torch_unimportable(blocker_stub):
    result = _run("import torch", blocker_stub)
    assert result.returncode != 0 and "blocked by test stub" in result.stderr


def test_importing_package_alone_never_imports_torch_agent_config_or_alpaca(recorder_stub):
    script = (
        "import sys; sys.path.insert(0, %r); import agent.kronos; "
        "bad = [m for m in ('torch', 'agent.config', 'alpaca') if m in sys.modules]; "
        "assert not bad, f'unexpectedly imported: {bad}'; "
        "print('ISOLATION_OK')" % str(ROOT)
    )
    result = _run(script, recorder_stub)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ISOLATION_OK" in result.stdout


def test_importing_every_submodule_individually_stays_torch_free(recorder_stub):
    for mod in ("config", "bars", "forecaster", "scorer", "service", "state"):
        script = (
            "import sys; sys.path.insert(0, %r); import agent.kronos.%s; "
            "assert 'torch' not in sys.modules, 'torch imported by agent.kronos.%s'; "
            "print('ISOLATION_OK')" % (str(ROOT), mod, mod)
        )
        result = _run(script, recorder_stub)
        assert result.returncode == 0, f"module={mod} stdout={result.stdout!r} stderr={result.stderr!r}"
        assert "ISOLATION_OK" in result.stdout


def test_constructing_kronos_predictor_without_calling_predict_batch_stays_torch_free(recorder_stub):
    script = (
        "import sys; sys.path.insert(0, %r); "
        "from agent.kronos.config import KronosConfig; from agent.kronos.forecaster import KronosPredictor; "
        "p = KronosPredictor(KronosConfig()); "
        "assert 'torch' not in sys.modules; print('ISOLATION_OK')" % str(ROOT)
    )
    result = _run(script, recorder_stub)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ISOLATION_OK" in result.stdout


def test_main_module_missing_deps_check_exits_3(blocker_stub):
    """With torch unimportable, `python -m agent.kronos --once` must refuse to
    run at all and exit 3, with one clear log line naming what's missing,
    BEFORE touching Alpaca keys, config or any network/file I/O. The blocker
    stub makes this hold in the torch-installed serving venv too, where an
    unguarded run would be a live forecast pass."""
    result = subprocess.run(
        [sys.executable, "-m", "agent.kronos", "--once"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=30, env=_env_with(blocker_stub),
    )
    assert result.returncode == 3, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "missing optional dependencies" in result.stderr.lower()
    assert "torch" in result.stderr.lower()


def test_main_module_missing_deps_check_exits_3_for_download_too(blocker_stub):
    result = subprocess.run(
        [sys.executable, "-m", "agent.kronos", "--download"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=30, env=_env_with(blocker_stub),
    )
    assert result.returncode == 3


def test_no_forbidden_order_or_trading_client_calls_anywhere_in_package():
    """This package is market-data READ-ONLY (agent/kronos/bars.py's own
    module docstring) -- the same order-submission grep flowcollector's
    isolation test runs."""
    forbidden = re.compile(r"\.(submit_order|cancel_order|replace_order|cancel_order_by_id)\s*\(")
    hits = []
    for f in _py_files():
        text = f.read_text()
        for m in forbidden.finditer(text):
            hits.append(f"{f.name}: {m.group(0)}")
    assert hits == [], f"order-submission-shaped call found: {hits}"


def test_all_expected_module_files_present():
    names = {f.name for f in _py_files()}
    for expected in ("__init__.py", "config.py", "bars.py", "forecaster.py", "scorer.py", "service.py", "state.py", "__main__.py"):
        assert expected in names, f"missing {expected}"


def test_vendor_package_files_present():
    vendor_model = PKG_DIR / "vendor" / "model"
    for expected in ("__init__.py", "kronos.py", "module.py"):
        assert (vendor_model / expected).exists(), f"missing vendor/model/{expected}"
    assert (PKG_DIR / "vendor" / "LICENSE").exists()
    assert (PKG_DIR / "vendor" / "VENDOR_PIN.md").exists()


def test_vendor_model_has_no_leftover_sys_path_manipulation():
    """The two documented Hydrinia patches (relative import + removed
    sys.path.append) must both be present in this copy."""
    kronos_py = (PKG_DIR / "vendor" / "model" / "kronos.py").read_text()
    assert "from .module import *" in kronos_py
    assert "from model.module import *" not in kronos_py
    assert "sys.path.append" not in kronos_py
    assert re.search(r"^\s*import sys\b", kronos_py, re.MULTILINE) is None
