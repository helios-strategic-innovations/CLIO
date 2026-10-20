"""S211: scripts/uw_ledger_report.py -- the read-only post-deploy proof that the UW pacer works.

A synthetic ledger is built in tmp_path through the real SupabaseUWQuotaClient (fake clock), so the
report is checked against numbers we control. The production ledger is never opened.
"""

import importlib.util
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.integrations.uw_quota_client import SupabaseUWQuotaClient

_SPEC = importlib.util.spec_from_file_location(
    "uw_ledger_report", Path(__file__).resolve().parents[1] / "scripts" / "uw_ledger_report.py"
)
report_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report_mod)

T0 = datetime(2026, 9, 29, 13, 30, 0, tzinfo=timezone.utc)   # 09:30 ET


def _ledger(tmp_path, plan):
    """plan: list of (seconds_from_T0, endpoint, status, purpose). Returns db path."""
    now = [T0]
    db = str(tmp_path / "ledger.sqlite3")
    client = SupabaseUWQuotaClient(caller="test", db_path=db, enabled=True, clock=lambda: now[0], start_worker=False)
    for offset, endpoint, status, purpose in plan:
        now[0] = T0 + timedelta(seconds=offset)
        decision = client.reserve(consumer="t", endpoint=endpoint, purpose=purpose, priority=0)
        client.observe_response(
            decision, consumer="t", endpoint=endpoint, purpose=purpose, status_code=status,
            headers={"x-uw-daily-req-count": "10", "x-uw-token-req-limit": "40000"}, latency_ms=1.0,
        )
    return db


def _report(db, since=None, until=None):
    con = report_mod.open_ro(db)
    try:
        return report_mod.report(con, since or "2026-09-29T13:00:00", until or "2026-09-29T15:00:00")
    finally:
        con.close()


def test_counts_status_rates_and_burst_histogram(tmp_path):
    plan = []
    # 10 lone requests, one per 10 s: no bursts, all 200
    plan += [(i * 10, f"/stock/SPY/nope", 200, "clio_flow_intel") for i in range(10)]
    # one burst of 12 in the same second, 4 of them 429
    plan += [(200, f"/darkpool/{'QQQ' if i % 2 else 'SPY'}", 429 if i < 4 else 200, "clio_flow_intel") for i in range(12)]
    # a retry row that succeeded
    plan += [(300, "/stock/SPY/max-pain", 200, "clio_flow_intel_retry")]
    rep = _report(_ledger(tmp_path, plan))

    assert rep["requests"] == 23
    assert rep["status"] == {"200": 19, "429": 4}
    assert rep["rate_429"] == pytest.approx(4 / 23)
    assert rep["max_same_second_burst"] == 12
    assert rep["burst_histogram"]["1"] == {"requests": 11, "429s": 0, "rate": 0.0}
    assert rep["burst_histogram"]["12"]["requests"] == 12 and rep["burst_histogram"]["12"]["429s"] == 4
    assert rep["retries"] == {"rows": 1, "status": {"200": 1}}
    assert rep["by_endpoint_family"]["/darkpool/{T}"] == {"requests": 12, "429s": 4, "rate": pytest.approx(4 / 12)}
    assert rep["cycle"]["entitlement"] == 40000 and rep["cycle"]["observed_count"] == 10


def test_window_selection_by_rowid_bisection(tmp_path):
    plan = [(i * 60, "/stock/SPY/nope", 200, "clio_flow_intel") for i in range(30)]   # one per minute for 30 min
    db = _ledger(tmp_path, plan)
    start = (T0 + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S")
    end = (T0 + timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S")
    rep = _report(db, start, end)
    assert rep["requests"] == 10                       # minutes 10..19 inclusive of start, exclusive of end
    assert _report(db, "2030-01-01T00:00:00", "2030-01-02T00:00:00")["requests"] == 0   # empty window
    assert _report(db, "2000-01-01T00:00:00", "2000-01-02T00:00:00")["requests"] == 0


def test_per_minute_statistics(tmp_path):
    plan = [(i, "/stock/SPY/nope", 200, "clio_flow_intel") for i in range(5)]           # 5 in minute 0
    plan += [(60 + i, "/stock/SPY/nope", 200, "clio_flow_intel") for i in range(2)]      # 2 in minute 1
    rep = _report(_ledger(tmp_path, plan))
    assert rep["per_minute"]["max"] == 5


def test_report_never_writes_and_opens_read_only(tmp_path):
    db = _ledger(tmp_path, [(0, "/stock/SPY/nope", 200, "clio_flow_intel")])
    before = os.stat(db).st_mtime_ns
    con = report_mod.open_ro(db)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM quota_reservations")     # a read-only connection cannot write
    con.close()
    _report(db)
    assert os.stat(db).st_mtime_ns == before


def test_render_and_cli_run(tmp_path, capsys):
    db = _ledger(tmp_path, [(i, "/stock/SPY/nope", 200, "clio_flow_intel") for i in range(3)])
    assert report_mod.main(["--db", db, "--since", "2026-09-29T13:00:00", "--until", "2026-09-29T15:00:00"]) == 0
    text = capsys.readouterr().out
    assert "429 rate: 0.0%" in text and "max same-second burst" in text
    assert report_mod.main(["--db", db, "--since", "2026-09-29T13:00:00", "--until", "2026-09-29T15:00:00", "--json"]) == 0
    assert '"requests": 3' in capsys.readouterr().out
