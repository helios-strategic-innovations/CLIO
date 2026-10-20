"""R-01: pytest must never append to the LIVE audit chain (/home/agent/openclaw/data/audit).

agent/safety/audit_log.py hard-coded the live directory; a test that reached the audit path
(e.g. dispatching a directive with the Hydrinia bridge on) wrote unsigned records into the
production chain. tests/conftest.py now redirects AUDIT_LOG_DIR per test. These tests pin that
redirect, both in-process and in a fresh interpreter.
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from agent.safety import audit_log

LIVE_AUDIT_DIR = "/home/agent/openclaw/data/audit"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _audit_files(directory):
    return sorted(glob.glob(os.path.join(directory, f"{audit_log.AUDIT_LOG_PREFIX}*.jsonl")))


def test_fresh_interpreter_with_env_writes_only_under_env_dir(tmp_path):
    target = tmp_path / "chain"
    env = dict(os.environ)
    env["CLIO_AUDIT_LOG_DIR"] = str(target)
    code = (
        "from agent.safety.audit_log import audit_security_event, AUDIT_LOG_DIR;"
        "print(AUDIT_LOG_DIR);"
        "audit_security_event('isolation_probe', 'probe write', severity='INFO')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[0] == str(target)

    files = _audit_files(str(target))
    assert len(files) == 1
    records = [json.loads(line) for line in open(files[0], encoding="utf-8") if line.strip()]
    assert any(r.get("event_type") == "isolation_probe" for r in records)


def test_bridge_on_directive_dispatch_audits_into_tmp_chain_only(monkeypatch, tmp_path):
    import asyncio

    from agent.integrations import directive_delivery

    # Precondition, checked BEFORE any write: the audit chain must already be redirected.
    # If it is not, the dispatch below would append to the live chain, so fail first.
    assert audit_log.AUDIT_LOG_DIR != LIVE_AUDIT_DIR
    assert "audit_chain" in audit_log.AUDIT_LOG_DIR
    chain_dir = audit_log.AUDIT_LOG_DIR

    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    directives_dir = tmp_path / "directives"
    monkeypatch.setattr(directive_delivery, "DIRECTIVES_DIR", str(directives_dir))
    monkeypatch.setattr(directive_delivery, "DIRECTIVES_LOG", str(tmp_path / "directives.jsonl"))

    class FakeOverrideEvent:
        command_text = "sell SPY now"
        timestamp = "2026-09-24T12:00:00Z"

    result = asyncio.run(directive_delivery.dispatch_directive(FakeOverrideEvent(), "sell SPY now"))
    assert "Directive dispatched" in result

    files = _audit_files(chain_dir)
    assert files, "dispatch should have written an audit record into the temp chain"
    records = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh if line.strip())
    assert any(r.get("event_type") == "directive_dispatched" for r in records)


def test_singleton_latches_within_this_test_only():
    # Precondition BEFORE touching the singleton: get_audit_log() appends startup rows, so a
    # missing redirect would write into the live chain (S215: a builder's mutant run did exactly
    # that). Fail first, write never.
    assert audit_log.AUDIT_LOG_DIR != LIVE_AUDIT_DIR
    assert "audit_chain" in audit_log.AUDIT_LOG_DIR
    first = audit_log.get_audit_log()
    assert first is audit_log.get_audit_log()
    assert audit_log._audit_log is first


def test_singleton_is_unlatched_at_start_of_next_test():
    # Runs after test_singleton_latches_within_this_test_only in file order. The autouse
    # fixture must reset the singleton on entry, so no earlier test's chain leaks forward.
    assert audit_log._audit_log is None
    assert "audit_chain" in audit_log.AUDIT_LOG_DIR
