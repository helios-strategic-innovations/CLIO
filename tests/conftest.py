"""Repo-wide test isolation.

agent/rate_limiter.py keeps its budget counters in /home/agent/openclaw/logs/
rate_limits.json — on the production VPS that is the LIVE counter file, so a
test run that reaches check_and_record() (e.g. email_trade_summary) would burn
real daily budgets. pytest_configure runs before any test module is imported,
so pointing CLIO_RATE_LIMITS_FILE at a per-session temp file here covers every
import path (module-level and lazy).
"""

import os
import tempfile

import pytest

_RATE_LIMITS_DIR = None

# Live advisory state files that CLIO modules READ from data/intelligence/ by
# default. On the VPS they exist and are fresh (the flow collector rewrites
# selfflow_state.json every 15 s even while idle), so a test that doesn't set
# its own path would read live data -- 13 tests/dpof failures on 2026-09-25.
_LIVE_STATE_ENV = (
    "CLIO_SELFFLOW_FILE",
    "CLIO_SELFFLOW_COMPARE_FILE",
    "CLIO_VOL_SURFACE_FILE",
    "CLIO_ECON_SURPRISE_FILE",
    "CLIO_NEWSFLASH_FILE",
    "CLIO_KRONOS_FORECAST_FILE",
    "CLIO_KRONOS_SCORES_FILE",
    "CLIO_KRONOS_HISTORY_FILE",
    "CLIO_MARKET_REGIME_FILE",
)


@pytest.fixture(autouse=True)
def _isolate_live_state_files(tmp_path_factory):
    """Point every live state path at a per-test temp dir (files absent
    unless a test writes them). A test's own setenv still wins; a test must
    never delenv these (that falls back to the live default path).

    Uses its own MonkeyPatch, not the `monkeypatch` fixture: requesting that
    here would set it up before every module's own autouse fixtures and so
    tear it down after them, changing their teardown order."""
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("live_state")
    for key in _LIVE_STATE_ENV:
        mp.setenv(key, str(d / f"{key.lower()}.json"))
    # DPOF's data dir (positions, ledger, signal log) defaults to the live data/dpof.
    mp.setenv("CLIO_DPOF_DATA_DIR", str(d / "dpof"))

    # decision-record producer (agent/decisions/): on the VPS pytest
    # may run with real SUPABASE_* creds in the environment -- no existing
    # test may ever reach the real database. Disabled + no shadow endpoints +
    # a per-test outbox path by default; a test that wants to exercise the
    # store/shadow fan-out opts in with its own monkeypatch.setenv (and an
    # injected fake transport), never the real Supabase project.
    mp.setenv("CLIO_DECISIONS_ENABLED", "0")
    mp.setenv("CLIO_DECISIONS_SHADOW_ENDPOINTS", "")
    mp.setenv("CLIO_DECISIONS_OUTBOX_PATH", str(d / "decisions_outbox.jsonl"))

    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _isolate_audit_log(tmp_path_factory):
    """Point the immutable audit chain (agent/safety/audit_log.py) at a per-test temp dir.
    AUDIT_LOG_DIR is read as a module global at call time, so the module attribute is what
    redirects writes; the env var covers fresh-interpreter subprocesses. The singleton is reset
    before and after each test so no test latches a path (or chain state) into the next one.
    Uses its own MonkeyPatch, not the `monkeypatch` fixture (see _isolate_live_state_files)."""
    from agent.safety import audit_log

    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("audit_chain")
    mp.setenv("CLIO_AUDIT_LOG_DIR", str(d))
    mp.setattr(audit_log, "AUDIT_LOG_DIR", str(d))
    audit_log._audit_log = None
    yield
    audit_log._audit_log = None
    mp.undo()


@pytest.fixture(autouse=True)
def _reset_uw_pacer():
    """The UW pacer (agent/integrations/uw_pacer.py) is a process-wide singleton: a test that spends
    its burst must not make the next test sleep, and a test that sets CLIO_UW_* env expects them to
    be re-read. Pure stdlib module -- importing it here has no side effects."""
    from agent.integrations import uw_pacer

    uw_pacer.reset_uw_pacer_for_tests()
    yield
    uw_pacer.reset_uw_pacer_for_tests()


def _no_load_dotenv(*_args, **_kwargs):
    return False


def _disable_dotenv() -> None:
    """Several dual-use modules call load_dotenv() at import (CLAUDE.md #34). Imported by a test on
    the VPS, that loads the real .env into this whole pytest process: every later test then sees live
    values (JUDGMENT_MODEL, API keys) -- test_model_hygiene failed there on 2026-09-27. Tests never
    need the real .env, so load_dotenv is a no-op for the session. Subprocess-based tests run in
    fresh interpreters and are unaffected."""
    try:
        import dotenv
        import dotenv.main
    except ImportError:
        return
    dotenv.load_dotenv = _no_load_dotenv
    dotenv.main.load_dotenv = _no_load_dotenv


def pytest_configure(config):
    global _RATE_LIMITS_DIR
    _disable_dotenv()
    if not os.environ.get("CLIO_RATE_LIMITS_FILE"):
        _RATE_LIMITS_DIR = tempfile.TemporaryDirectory(prefix="clio-test-ratelimits-")
        os.environ["CLIO_RATE_LIMITS_FILE"] = os.path.join(_RATE_LIMITS_DIR.name, "rate_limits.json")


def pytest_unconfigure(config):
    if _RATE_LIMITS_DIR is not None:
        _RATE_LIMITS_DIR.cleanup()
