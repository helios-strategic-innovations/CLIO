"""S211 LED-1: the weekly ledger-prune systemd units.

The unit files are configuration nobody runs until Sunday 04:00, so a typo (a moved script path, a flag the script
refuses, retention under its floor) would only surface as a failed unit days later. Instead of checking strings, the
tests take the EXACT argv from the .service, point it at a synthetic ledger (tmp_path; the production ledger and backup
directory are never touched) and run the script's real main(): it must parse, back up, prune to 30 days and exit 0.
"""

import json
import re
import shlex
from pathlib import Path

import pytest

import test_uw_ledger_prune as base   # sibling test module: synthetic-ledger builder (tests/ has no __init__.py)

ROOT = Path(__file__).resolve().parents[1]
UNIT_DIR = ROOT / "systemd" / "user"
SERVICE = UNIT_DIR / "clio-ledger-prune.service"
TIMER = UNIT_DIR / "clio-ledger-prune.timer"
prune_mod = base.prune_mod


def _exec_argv() -> list[str]:
    m = re.search(r"^ExecStart=(.+)$", SERVICE.read_text(), re.MULTILINE)
    assert m, "no ExecStart"
    return shlex.split(m.group(1))


def _count(db, table):
    import sqlite3

    con = sqlite3.connect(db)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def test_exec_points_at_the_repo_script_and_a_venv_interpreter():
    argv = _exec_argv()
    assert argv[0].endswith("/venv/bin/python3")
    assert argv[1] == "/home/agent/openclaw/scripts/uw_ledger_prune.py"
    assert (ROOT / "scripts" / "uw_ledger_prune.py").is_file()


def test_the_exact_unit_argv_runs_and_prunes_a_synthetic_ledger_to_30_days(tmp_path, capsys):
    tail = _exec_argv()[2:]
    days = int(tail[tail.index("--older-than-days") + 1])
    assert days == 30
    db = base._build(tmp_path)
    before = _count(db, "quota_reservations")
    backups = tmp_path / "backups"
    rc = prune_mod.main(tail + ["--db", db, "--backup-dir", str(backups), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert _count(db, "quota_reservations") == base.kept_rows(30) < before      # pruned to exactly the 30-day window
    assert _count(db, "supabase_outbox") == base.kept_rows(30)
    assert len(list(backups.glob("uw_quota_*.sqlite3"))) == 1                   # backed up first, once
    assert out["backup"]["file"].startswith(str(backups))


def test_without_apply_the_same_argv_would_delete_nothing(tmp_path):
    """Pins why --apply matters: drop it and the weekly run is a silent no-op."""
    tail = [a for a in _exec_argv()[2:] if a not in ("--apply", "--vacuum")]
    db = base._build(tmp_path)
    before = _count(db, "quota_reservations")
    assert prune_mod.main(tail + ["--db", db, "--backup-dir", str(tmp_path / "b")]) == 0
    assert _count(db, "quota_reservations") == before
    assert "--apply" in _exec_argv() and "--vacuum" in _exec_argv()


def test_the_script_refuses_retention_under_its_floor_so_a_typo_cannot_hollow_the_ledger(tmp_path, capsys):
    db = base._build(tmp_path)
    before = _count(db, "quota_reservations")
    rc = prune_mod.main(["--older-than-days", "3", "--db", db, "--backup-dir", str(tmp_path / "b")])
    assert rc != 0
    assert "must be >= 7" in capsys.readouterr().err
    assert _count(db, "quota_reservations") == before


def test_the_service_fails_loudly_and_logs_to_the_repo_log():
    text = SERVICE.read_text()
    assert "Type=oneshot" in text                      # a failed run marks the unit failed (systemctl --user --failed)
    log = "/home/agent/openclaw/logs/uw_ledger_prune.log"
    assert re.search(rf"^StandardOutput=append:{re.escape(log)}$", text, re.MULTILINE)
    assert re.search(rf"^StandardError=append:{re.escape(log)}$", text, re.MULTILINE)
    assert re.search(r"^TimeoutStartSec=\d+", text, re.MULTILINE)


def test_the_timer_is_weekly_on_sunday_early_morning_eastern():
    text = TIMER.read_text()
    assert re.search(r"^OnCalendar=Sun 04:00:00 America/New_York$", text, re.MULTILINE)
    assert re.search(r"^Unit=clio-ledger-prune\.service$", text, re.MULTILINE)
    assert "WantedBy=timers.target" in text
