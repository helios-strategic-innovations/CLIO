"""Grep-provable isolation guarantees for agent/flowcollector/ (INTEL_V3_SPEC.md:
"new modules never import agent.dpof.broker/orders/engine"; this package's own
docstrings additionally promise no order-submission calls anywhere).
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "agent" / "flowcollector"

_FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(from|import)\s+agent\.dpof\.(broker|orders|engine)\b", re.MULTILINE,
)
_FORBIDDEN_CALLS = re.compile(r"\.(submit_order|cancel_order|replace_order|cancel_order_by_id)\s*\(")


def _py_files():
    return sorted(p for p in PKG_DIR.glob("*.py"))


def test_no_forbidden_dpof_order_module_imports():
    hits = []
    for f in _py_files():
        text = f.read_text()
        if _FORBIDDEN_IMPORTS.search(text):
            hits.append(f.name)
    assert hits == [], f"forbidden agent.dpof.{{broker,orders,engine}} import found in: {hits}"


def test_no_order_submission_calls_anywhere_in_package():
    hits = []
    for f in _py_files():
        text = f.read_text()
        for m in _FORBIDDEN_CALLS.finditer(text):
            hits.append(f"{f.name}: {m.group(0)}")
    assert hits == [], f"order-submission-shaped call found: {hits}"


def test_package_never_imports_alpaca_trading_orders_module():
    """alpaca.trading.orders defines OrderRequest subclasses used to
    construct/submit trading orders -- this package never needs it (it only
    ever calls the read-only get_option_contracts on the trading client)."""
    hits = []
    for f in _py_files():
        text = f.read_text()
        if re.search(r"from\s+alpaca\.trading\.requests\s+import.*OrderRequest", text):
            hits.append(f.name)
        if re.search(r"\balpaca\.trading\.orders\b", text):
            hits.append(f.name)
    assert hits == []


def test_importing_package_alone_never_imports_agent_config_or_httpx_or_alpaca():
    """Fresh-interpreter subprocess check (mirrors tests/dpof/test_data.py's
    pattern): importing agent.flowcollector by itself must not drag in
    agent.config, httpx, or the alpaca SDK -- all of collector.py's
    config/Alpaca-touching imports are lazy, inside functions/methods (per
    this package's own module docstrings and CLAUDE.md anti-pattern #34)."""
    script = (
        "import sys; sys.path.insert(0, %r); import agent.flowcollector; "
        "bad = [m for m in ('agent.config', 'httpx', 'alpaca') if m in sys.modules]; "
        "assert not bad, f'unexpectedly imported: {bad}'; "
        "print('ISOLATION_OK')" % str(ROOT)
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ISOLATION_OK" in result.stdout


def test_importing_collector_module_directly_also_stays_lazy():
    script = (
        "import sys; sys.path.insert(0, %r); import agent.flowcollector.collector; "
        "bad = [m for m in ('agent.config', 'httpx', 'alpaca') if m in sys.modules]; "
        "assert not bad, f'unexpectedly imported: {bad}'; "
        "print('ISOLATION_OK')" % str(ROOT)
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ISOLATION_OK" in result.stdout


def test_all_expected_module_files_present():
    names = {f.name for f in _py_files()}
    for expected in ("__init__.py", "config.py", "universe.py", "classifier.py", "aggregator.py", "state.py", "collector.py", "__main__.py", "greeks.py"):
        assert expected in names, f"missing {expected}"


def test_greeks_module_never_imports_agent_config_or_httpx_or_alpaca():
    """greeks.py (BSM/implied-vol math) is stdlib-only per its own module
    docstring -- its one optional dependency (agent.integrations.
    market_events's early-close dates) is a LAZY import inside
    `_early_close_dates()`, never at module import time."""
    script = (
        "import sys; sys.path.insert(0, %r); import agent.flowcollector.greeks; "
        "bad = [m for m in ('agent.config', 'httpx', 'alpaca') if m in sys.modules]; "
        "assert not bad, f'unexpectedly imported: {bad}'; "
        "print('ISOLATION_OK')" % str(ROOT)
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ISOLATION_OK" in result.stdout
