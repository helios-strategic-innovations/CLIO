"""S211 (LED-1): scripts/uw_ledger_prune.py -- prune the shared UW quota ledger safely.

Synthetic ledgers are built in tmp_path through the real SupabaseUWQuotaClient (fake clock relative to now).
The production ledger is never opened. Rules under test: dry-run writes nothing; an unsynced outbox row and a
pending reservation are NEVER deleted (whatever their age); quota_cycles is untouched; a verified backup is required.
"""

import atexit
import importlib.util
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.integrations.uw_quota_client import SupabaseUWQuotaClient

_SPEC = importlib.util.spec_from_file_location(
    "uw_ledger_prune", Path(__file__).resolve().parents[1] / "scripts" / "uw_ledger_prune.py"
)
prune_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prune_mod)

NOW = datetime.now(timezone.utc)
PER_DAY = 5
DAYS = 40   # ledger spans DAYS..0 days ago; every row sits 12 h off a day boundary (never on a cutoff)


def old_rows(retention):
    """rows strictly older than `retention` days: days-ago d >= retention (d = retention -> retention+0.5 days old)."""
    return (DAYS - retention + 1) * PER_DAY


def kept_rows(retention):
    return retention * PER_DAY          # d = 0 .. retention-1


_TEMPLATES: dict = {}


def _build(tmp_path, *, days=DAYS, per_day=PER_DAY, payload_pad=0):
    """A fresh copy of a cached synthetic ledger (building through the real client is slow; copying is not)."""
    key = (days, per_day, payload_pad)
    if key not in _TEMPLATES:
        tpl_dir = tempfile.mkdtemp(prefix="uwprune-tpl-")
        atexit.register(shutil.rmtree, tpl_dir, ignore_errors=True)
        _TEMPLATES[key] = _make(Path(tpl_dir), days=days, per_day=per_day, payload_pad=payload_pad)
    dst = str(tmp_path / "ledger.sqlite3")
    shutil.copyfile(_TEMPLATES[key], dst)
    return dst


def _make(tmp_path, *, days, per_day, payload_pad):
    """days..0 days ago, per_day requests each, oldest first (rowids ascend with time). All outbox rows synced."""
    box = [NOW - timedelta(days=days, hours=12)]
    db = str(tmp_path / "ledger.sqlite3")
    client = SupabaseUWQuotaClient(caller="t", db_path=db, enabled=True, clock=lambda: box[0], start_worker=False)
    for d in range(days, -1, -1):
        for i in range(per_day):
            box[0] = NOW - timedelta(days=d, hours=12) + timedelta(seconds=i)
            dec = client.reserve(consumer="t", endpoint="/stock/SPY/nope", priority=0)
            client.observe_response(
                dec, consumer="t", endpoint="/stock/SPY/nope", purpose="unit", status_code=200,
                headers={"x-uw-daily-req-count": str(d * 100 + i), "x-uw-token-req-limit": "40000"}, latency_ms=1.0,
            )
    con = sqlite3.connect(db)
    con.execute("UPDATE supabase_outbox SET synced_at = created_at")
    if payload_pad:
        con.execute("UPDATE supabase_outbox SET payload_json = payload_json || ?", ("x" * payload_pad,))
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # single self-contained file, safe to copy
    con.close()
    return db


def _protect_old_rows(db):
    """Inside the OLD delete range: one UNSYNCED outbox row, one PENDING reservation and one reservation in an
    unknown state -- none may ever be deleted (they sit at low rowids, i.e. squarely in the range being pruned)."""
    con = sqlite3.connect(db)
    con.execute("UPDATE supabase_outbox SET synced_at = NULL WHERE rowid = (SELECT min(rowid) + 3 FROM supabase_outbox)")
    con.execute("UPDATE quota_reservations SET state = 'pending', resolved_at = NULL "
                "WHERE rowid = (SELECT min(rowid) + 2 FROM quota_reservations)")
    con.execute("UPDATE quota_reservations SET state = 'mystery' WHERE rowid = (SELECT min(rowid) + 4 FROM quota_reservations)")
    con.commit()
    con.close()


def _counts(db):
    con = sqlite3.connect(db)
    out = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
           for t in ("quota_reservations", "supabase_outbox", "quota_cycles")}
    con.close()
    return out


def _run(argv):
    return prune_mod.main(argv)


def _apply_args(db, tmp_path, days=14, extra=()):
    return ["--db", db, "--older-than-days", str(days), "--apply", "--backup-dir", str(tmp_path / "bk"),
            "--pause-ms", "0", *extra]


# ---------------------------------------------------------------------------------- dry run
def test_dry_run_reports_and_writes_nothing(tmp_path, capsys):
    db = _build(tmp_path)
    before = (os.stat(db).st_mtime_ns, os.path.getsize(db), _counts(db))
    assert _run(["--db", db, "--older-than-days", "14"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "supabase_outbox" in out and "quota_reservations" in out
    assert (os.stat(db).st_mtime_ns, os.path.getsize(db), _counts(db)) == before
    assert not (tmp_path / "bk").exists()


def test_dry_run_counts_match_the_synthetic_ledger(tmp_path):
    db = _build(tmp_path)
    con = prune_mod.connect(db, readonly=True)
    cutoff = (NOW - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%S")
    pl = prune_mod.plan(con, cutoff)
    con.close()
    assert pl["tables"]["quota_reservations"]["rows_older_than_cutoff"] == old_rows(14)
    assert pl["tables"]["supabase_outbox"]["rows_older_than_cutoff"] == old_rows(14)
    assert pl["tables"]["supabase_outbox"]["protected_in_range"] == 0


# ---------------------------------------------------------------------------------- apply
def test_apply_deletes_only_old_synced_terminal_rows(tmp_path):
    db = _build(tmp_path)
    _protect_old_rows(db)
    before = _counts(db)
    assert _run(_apply_args(db, tmp_path, extra=["--batch", "7"])) == 0

    after = _counts(db)
    assert after["quota_cycles"] == before["quota_cycles"]                       # untouched
    con = sqlite3.connect(db)
    # protected rows survive
    assert con.execute("SELECT count(*) FROM supabase_outbox WHERE synced_at IS NULL").fetchone()[0] == 1
    states = sorted(r[0] for r in con.execute(
        "SELECT state FROM quota_reservations WHERE state IN ('pending','mystery')"))
    assert states == ["mystery", "pending"]                                       # both survive, though old
    # everything from the last 14 days survives, every other old synced/terminal row is gone
    cutoff_iso = (NOW - timedelta(days=14)).isoformat()
    assert con.execute("SELECT count(*) FROM quota_reservations WHERE reserved_at >= ?", (cutoff_iso,)).fetchone()[0] \
        == kept_rows(14)
    left_old_res = con.execute("SELECT count(*) FROM quota_reservations WHERE reserved_at < ?", (cutoff_iso,)).fetchone()[0]
    assert left_old_res == 2                                                     # the pending + unknown-state rows only
    left_old_out = con.execute("SELECT count(*) FROM supabase_outbox WHERE created_at < ?", (cutoff_iso,)).fetchone()[0]
    assert left_old_out == 1                                                     # the unsynced one only
    con.close()
    assert after["quota_reservations"] == kept_rows(14) + 2          # kept days + the two protected old rows
    assert after["supabase_outbox"] == kept_rows(14) + 1             # kept days + the one unsynced old row
    assert before["quota_reservations"] - after["quota_reservations"] == old_rows(14) - 2
    assert before["supabase_outbox"] - after["supabase_outbox"] == old_rows(14) - 1   # the unsynced row stays
    assert (tmp_path / "bk").is_dir() and len(list((tmp_path / "bk").glob("uw_quota_*.sqlite3"))) == 1


def test_backup_holds_every_original_row(tmp_path):
    db = _build(tmp_path)
    before = _counts(db)
    assert _run(_apply_args(db, tmp_path)) == 0
    backup_file = next((tmp_path / "bk").glob("uw_quota_*.sqlite3"))
    assert _counts(str(backup_file)) == before
    con = sqlite3.connect(str(backup_file))
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close()


@pytest.mark.parametrize("batch", [1, 7, 10000])
def test_result_is_independent_of_batch_size(tmp_path, batch):
    work = tmp_path / f"b{batch}"
    work.mkdir()
    db = _build(work)
    assert _run(_apply_args(db, work, extra=["--batch", str(batch)])) == 0
    after = _counts(db)
    assert after["quota_reservations"] == kept_rows(14)
    assert after["supabase_outbox"] == kept_rows(14)


def test_second_apply_is_a_noop(tmp_path):
    db = _build(tmp_path)
    assert _run(_apply_args(db, tmp_path)) == 0
    first = _counts(db)
    assert _run(_apply_args(db, tmp_path / "second")) == 0
    assert _counts(db) == first


def test_recent_rows_are_never_touched_even_with_a_huge_batch(tmp_path):
    db = _build(tmp_path)
    assert _run(_apply_args(db, tmp_path, days=30, extra=["--batch", "1000000"])) == 0
    assert _counts(db)["quota_reservations"] == kept_rows(30)


# ---------------------------------------------------------------------------------- vacuum
def test_vacuum_reclaims_the_file(tmp_path):
    db = _build(tmp_path, per_day=60, payload_pad=400)
    size_before = os.path.getsize(db)
    assert _run(_apply_args(db, tmp_path, extra=["--vacuum"])) == 0
    size_after = os.path.getsize(db)
    assert size_after < size_before * 0.6
    con = sqlite3.connect(db)
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert con.execute("PRAGMA freelist_count").fetchone()[0] == 0
    con.close()


def test_delete_without_vacuum_keeps_the_file_size(tmp_path):
    db = _build(tmp_path, per_day=60, payload_pad=400)
    size_before = os.path.getsize(db)
    assert _run(_apply_args(db, tmp_path)) == 0
    assert os.path.getsize(db) >= size_before * 0.95      # freed pages are reused, the file does not shrink


# ---------------------------------------------------------------------------------- refusals & safety
def test_refuses_retention_below_seven_days(tmp_path, capsys):
    db = _build(tmp_path)
    assert _run(["--db", db, "--older-than-days", "6"]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert _counts(db)["quota_reservations"] == kept_rows(DAYS + 1)


def test_vacuum_requires_apply(tmp_path):
    db = _build(tmp_path)
    assert _run(["--db", db, "--older-than-days", "14", "--vacuum"]) == 2


def test_refuses_a_missing_database(tmp_path):
    assert _run(["--db", str(tmp_path / "nope.sqlite3"), "--older-than-days", "14"]) == 2


def test_refuses_a_non_wal_database(tmp_path):
    db = str(tmp_path / "plain.sqlite3")
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=DELETE")
    for ddl in (
        "CREATE TABLE supabase_outbox (event_id TEXT PRIMARY KEY, payload_json TEXT, created_at TEXT, synced_at TEXT)",
        "CREATE TABLE quota_reservations (request_id TEXT PRIMARY KEY, cycle_key TEXT, state TEXT, reserved_at TEXT)",
        "CREATE TABLE quota_cycles (cycle_key TEXT PRIMARY KEY, entitlement INTEGER)",
    ):
        con.execute(ddl)
    con.commit()
    con.close()
    assert _run(["--db", db, "--older-than-days", "14"]) == 2


def test_refuses_an_unexpected_schema(tmp_path):
    db = str(tmp_path / "odd.sqlite3")
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE supabase_outbox (event_id TEXT)")
    con.commit()
    con.close()
    assert _run(["--db", db, "--older-than-days", "14"]) == 2


def test_backup_is_never_overwritten(tmp_path):
    db = _build(tmp_path)
    target = tmp_path / "existing.sqlite3"
    target.write_text("do not touch")
    with pytest.raises(prune_mod.Refused):
        prune_mod.backup(db, str(target))
    assert target.read_text() == "do not touch"


def test_existing_backup_file_can_be_supplied_and_is_verified(tmp_path):
    db = _build(tmp_path)
    info = prune_mod.backup(db, str(tmp_path / "manual.sqlite3"))
    assert info["integrity"] == "ok"
    assert _run(["--db", db, "--older-than-days", "14", "--apply", "--backup-file", str(tmp_path / "manual.sqlite3"),
                 "--pause-ms", "0"]) == 0
    assert not (tmp_path / "bk").exists()          # no second backup was made


def test_a_bad_backup_file_stops_the_run_before_any_delete(tmp_path):
    db = _build(tmp_path)
    bad = tmp_path / "bad.sqlite3"
    con = sqlite3.connect(str(bad))
    con.execute("CREATE TABLE quota_reservations(request_id)")
    con.commit()
    con.close()
    with pytest.raises(Exception):
        _run(["--db", db, "--older-than-days", "14", "--apply", "--backup-file", str(bad), "--pause-ms", "0"])
    assert _counts(db)["quota_reservations"] == kept_rows(DAYS + 1)


def test_lock_contention_aborts_and_deletes_nothing(tmp_path, capsys):
    db = _build(tmp_path)
    blocker = sqlite3.connect(db, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")                    # another writer holds the lock
    try:
        code = _run(_apply_args(db, tmp_path, extra=["--busy-ms", "50"]))
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert code == 3
    assert "ABORTED" in capsys.readouterr().err
    assert _counts(db)["quota_reservations"] == kept_rows(DAYS + 1)


def test_json_output_is_valid_and_complete(tmp_path, capsys):
    db = _build(tmp_path)
    assert _run(_apply_args(db, tmp_path, extra=["--json", "--vacuum"])) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "apply" and data["verify"]["integrity"] == "ok"
    assert data["deleted"]["supabase_outbox"]["deleted"] == old_rows(14)
    assert data["vacuum"]["bytes_after"] <= data["vacuum"]["bytes_before"]


def test_verification_failure_returns_exit_4(tmp_path, monkeypatch):
    db = _build(tmp_path)
    monkeypatch.setattr(prune_mod, "verify", lambda *a, **k: {
        "integrity": "ok", "tables": {}, "unsynced_outbox": 0, "pending_reservations": 0, "quota_cycles": 0,
        "problems": ["quota_cycles is empty"]})
    assert _run(_apply_args(db, tmp_path)) == 4


def test_dry_run_labels_the_rowid_bound_as_max_rowid_not_a_row_count(tmp_path, capsys):
    """After a prune the table holds thousands of rows but its max rowid is still in the millions; the report must not
    call that number a total (the first weekly log would have read "1.7 M rows" on an 11 MB ledger)."""
    db = _build(tmp_path)
    con = sqlite3.connect(db)
    con.execute("DELETE FROM quota_reservations WHERE rowid <= ?", (base_max := con.execute(
        "SELECT MAX(rowid) FROM quota_reservations").fetchone()[0] - 20,))
    con.commit()
    remaining = con.execute("SELECT COUNT(*) FROM quota_reservations").fetchone()[0]
    con.close()
    assert remaining == 20 and base_max > 100
    rc = prune_mod.main(["--older-than-days", "30", "--db", db, "--backup-dir", str(tmp_path / "b")])
    out = capsys.readouterr().out
    assert rc == 0
    line = next(l for l in out.splitlines() if l.strip().startswith("quota_reservations"))
    assert "max rowid" in line and "total" not in line
    assert f"{base_max + 20:,}" in line
