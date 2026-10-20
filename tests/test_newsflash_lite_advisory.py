"""
Advisory-only boundary test for agent/integrations/newsflash_lite.py (INTEL_V3_SPEC.md
contract "N"). Mirrors tests/test_social_intel_divergence.py::test_no_risk_write_path —
grep-provable that this module never imports the DPOF trade-execution surface and never
contains order/position/authority-write tokens in its EXECUTABLE code (docstrings and
comments, where the prohibition is documented, are stripped first).
"""

import os
import re


_SRC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "agent", "integrations", "newsflash_lite.py",
)


def _executable_source() -> str:
    with open(_SRC_PATH) as f:
        text = f.read()
    code = re.sub(r'"""[\s\S]*?"""', "", text)
    code = re.sub(r"'''[\s\S]*?'''", "", code)
    code = "\n".join(ln.split("#", 1)[0] for ln in code.splitlines())
    return code


def test_no_dpof_trading_surface_imports():
    code = _executable_source()
    banned_imports = (
        "agent.dpof.broker", "agent.dpof.orders", "agent.dpof.engine",
        "from agent.dpof.broker", "from agent.dpof.orders", "from agent.dpof.engine",
        "import agent.dpof.broker", "import agent.dpof.orders", "import agent.dpof.engine",
    )
    offenders = [b for b in banned_imports if b in code]
    assert not offenders, offenders


def test_no_order_or_authority_write_tokens():
    code = _executable_source()
    banned = (
        "EMERGENCY_STOP", "execute_trade", "execute_options_trade", "atlas_team",
        "risk_envelope", "place_order", "submit_order", "control_hydrinia", "modify_hydrinia",
        "position_log", "alpaca_options", "alpaca.py",
    )
    offenders = [b for b in banned if b in code]
    assert not offenders, offenders


def test_module_docstring_declares_advisory_only():
    with open(_SRC_PATH) as f:
        text = f.read()
    assert "ADVISORY ONLY" in text
