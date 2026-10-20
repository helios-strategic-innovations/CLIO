"""Pins the repo-wide test isolation in tests/conftest.py: on the VPS a test must never see the real
.env or read live state files (2026-09-27: three VPS-only failures traced to exactly these leaks)."""

import os

from agent.dpof.config import load_dpof_config
from agent.integrations import market_regime


def test_load_dotenv_is_a_no_op_during_tests(tmp_path):
    import dotenv

    env_file = tmp_path / ".env"
    env_file.write_text("CLIO_TEST_DOTENV_PROBE=loaded\n")
    assert dotenv.load_dotenv(env_file) is False
    assert "CLIO_TEST_DOTENV_PROBE" not in os.environ


def test_modules_that_load_dotenv_at_import_get_the_no_op():
    from agent.integrations import social_intel_divergence

    assert social_intel_divergence.load_dotenv.__name__ == "_no_load_dotenv"


def test_audit_chain_resolves_to_a_per_test_temp_dir_not_the_live_path():
    from agent.safety import audit_log

    assert audit_log.AUDIT_LOG_DIR != "/home/agent/openclaw/data/audit"
    assert "audit_chain" in audit_log.AUDIT_LOG_DIR
    assert os.environ["CLIO_AUDIT_LOG_DIR"] == audit_log.AUDIT_LOG_DIR


def test_market_regime_and_dpof_data_paths_are_per_test_temp_paths():
    regime_path = market_regime._file_path()
    assert "live_state" in regime_path and not os.path.exists(regime_path)
    assert "live_state" in load_dpof_config().data_dir
