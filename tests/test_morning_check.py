"""S211 (PROC-1): scripts/morning_check.py -- the durable, read-only morning health log run by a systemd --user timer.

Pure evaluators are tested directly; main() runs end to end against a fake root, a synthetic ledger, canned
systemctl/journalctl output and a fixed clock. Nothing live is read.
"""

import importlib.util
import json
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.integrations.uw_quota_client import SupabaseUWQuotaClient

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SPEC = importlib.util.spec_from_file_location("morning_check", SCRIPTS / "morning_check.py")
mc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mc)
NY = ZoneInfo("America/New_York")


# ---------------------------------------------------------------- pure evaluators
def test_services():
    out = mc.eval_services({"moltbot": "active", "clio-kronos": "failed"}, {"moltbot": "0", "clio-kronos": "3"})
    assert out[0][0] == "PASS" and out[1][0] == "FAIL"


def _rep(n=200, rate=0.0, burst=3):
    return {"requests": n, "status": {"200": n}, "rate_429": rate, "max_same_second_burst": burst,
            "retries": {"rows": 0, "status": {}}, "cycle": {"cycle_key": "k", "entitlement": 40000,
                                                            "observed_count": 10, "anomaly_count": 0}}


@pytest.mark.parametrize("rate, level", [(0.0, "PASS"), (0.019, "PASS"), (0.02, "WARN"), (0.09, "WARN"), (0.10, "FAIL"), (0.5, "FAIL")])
def test_429_rate_thresholds(rate, level):
    lines = dict((t.split(" (")[0], lv) for lv, t in mc.eval_ledger(_rep(rate=rate), 30))
    assert lines[f"UW 429 rate {rate:.1%}"] == level


@pytest.mark.parametrize("burst, level", [(1, "PASS"), (4, "PASS"), (5, "WARN"), (16, "WARN")])
def test_burst_thresholds(burst, level):
    got = [lv for lv, t in mc.eval_ledger(_rep(burst=burst), 30) if "same-second" in t]
    assert got == [level]


def test_too_few_requests_is_a_warning_not_a_verdict():
    out = mc.eval_ledger(_rep(n=10, rate=0.5, burst=9), 30)
    assert any(lv == "WARN" and "too few" in t for lv, t in out)
    assert not any(t.startswith("UW 429 rate") for _lv, t in out)   # no verdict from 10 requests


def test_flow_sources_use_the_real_source_string_shapes():
    counts = {"darkpool:uw:/darkpool/SPY": 90, "nofa:uw:/option-trades/flow-alerts?ticker_symbol=SPY": 90,
              "premium:selfflow:SPY": 10, "gex:selfflow:QQQ": 10}
    (lv, tx), = mc.eval_flow_sources(counts, 100)
    assert lv == "PASS" and "90%" in tx
    (lv, tx), = mc.eval_flow_sources({"nofa:selfflow:SPY": 50}, 10)
    assert lv == "WARN" and "0%" in tx
    (lv, _), = mc.eval_flow_sources({}, 0)
    assert lv == "WARN"


def test_journal_patterns_and_tolerances():
    text = "x\n" * 3 + "Traceback (most recent call last)\n" + "UW rate limited\n" * 20 + "apiKey=REDACTED\n"
    got = {t.split("'")[1]: lv for lv, t in mc.eval_journal(text)}
    assert got["Traceback"] == "FAIL"
    assert got["apiKey="] == "FAIL"
    assert got["UW rate limited"] == "PASS"                      # 20 is tolerated
    assert got["UW AUTH"] == "PASS"
    got2 = {t.split("'")[1]: lv for lv, t in mc.eval_journal("UW rate limited\n" * 21)}
    assert got2["UW rate limited"] == "WARN"


def test_uw_account_quota_denied_tolerance_is_three():
    """R-13: real denials are now WARNING-only at the source; a 4th denied line in the window is a WARN."""
    tol = {name: t for name, _lv, t in mc.JOURNAL_PATTERNS}
    assert tol["UW_ACCOUNT_QUOTA_DENIED"] == 3
    got = {t.split("'")[1]: lv for lv, t in mc.eval_journal("UW_ACCOUNT_QUOTA_DENIED caller=clio\n" * 3)}
    assert got["UW_ACCOUNT_QUOTA_DENIED"] == "PASS"
    got4 = {t.split("'")[1]: lv for lv, t in mc.eval_journal("UW_ACCOUNT_QUOTA_DENIED caller=clio\n" * 4)}
    assert got4["UW_ACCOUNT_QUOTA_DENIED"] == "WARN"


def test_kronos_freshness():
    assert mc.eval_kronos(None, True)[0][0] == "WARN"
    assert mc.eval_kronos(500, True)[0][0] == "PASS"
    assert mc.eval_kronos(700, True)[0][0] == "WARN"
    assert mc.eval_kronos(7200, False)[0][0] == "PASS"           # outside the session it may legitimately be old


def test_report_summary_and_exit_code():
    r = mc.Report()
    r.add("PASS", "a")
    r.add("WARN", "b")
    assert r.exit_code == 0 and r.render().endswith("SUMMARY: 1 pass, 1 warn, 0 fail")
    r.add("FAIL", "c")
    assert r.exit_code == 1 and r.render().endswith("SUMMARY: 1 pass, 1 warn, 1 fail")


# ---------------------------------------------------------------- end to end on a fake root
FIXED = datetime(2026, 9, 29, 10, 0, 0, tzinfo=NY)          # 10:00 ET Tuesday (inside the session)


class _FixedDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED.astimezone(tz) if tz else FIXED.replace(tzinfo=None)


def _fake_root(tmp_path, *, ledger_requests, status=200):
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPTS / "uw_ledger_report.py", root / "scripts" / "uw_ledger_report.py")
    (root / "data" / "dpof").mkdir(parents=True)
    (root / "data" / "intelligence").mkdir(parents=True)
    rec = {"ts": FIXED.isoformat(), "pillars": {"nofa": {"available": True, "source": "uw:/x"},
                                                 "gex": {"available": True, "source": "selfflow:SPY"}}}
    (root / "data" / "dpof" / "signals.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps({"ts": "2020-01-01T00:00:00"}) + "\n")
    kf = root / "data" / "intelligence" / "kronos_forecast.json"
    kf.write_text("{}")
    db = str(tmp_path / "ledger.sqlite3")
    box = [FIXED.astimezone(timezone.utc) - timedelta(minutes=20)]
    client = SupabaseUWQuotaClient(caller="t", db_path=db, enabled=True, clock=lambda: box[0], start_worker=False)
    for i in range(ledger_requests):
        box[0] = FIXED.astimezone(timezone.utc) - timedelta(minutes=20) + timedelta(seconds=i * 7)
        d = client.reserve(consumer="t", endpoint="/stock/SPY/nope", priority=0)
        client.observe_response(d, consumer="t", endpoint="/stock/SPY/nope", purpose="unit",
                                status_code=status, headers={"x-uw-daily-req-count": str(i), "x-uw-token-req-limit": "40000"},
                                latency_ms=1.0)
    memory = tmp_path / "MEMORY.md"
    memory.write_text("x" * 5000)
    return root, db, memory


def _patch(monkeypatch, root, db, memory, journal_text="", services="active"):
    def fake_run(cmd, timeout=30):
        if cmd[:2] == ["systemctl", "is-active"]:
            return services + "\n"
        if cmd[:2] == ["systemctl", "show"]:
            return "NRestarts=0\n"
        if cmd[0] == "journalctl":
            return journal_text
        return ""

    monkeypatch.setattr(mc, "ROOT", root)
    monkeypatch.setattr(mc, "MEMORY_MD", memory)
    monkeypatch.setattr(mc, "_run", fake_run)
    monkeypatch.setattr(mc, "datetime", _FixedDT)
    monkeypatch.setenv("CLIO_UW_LEDGER_DB", db)
    monkeypatch.setattr(mc, "BACKUPS_DIR", root.parent / "no_backups")          # never read the live ~/backups
    monkeypatch.setattr(mc, "_disk_usage", lambda: (100 * 10**9, 50 * 10**9, 50 * 10**9))


def test_end_to_end_healthy_day_exits_zero(tmp_path, monkeypatch, capsys):
    root, db, memory = _fake_root(tmp_path, ledger_requests=60)
    _patch(monkeypatch, root, db, memory)
    assert mc.main() == 0
    out = capsys.readouterr().out
    assert "SUMMARY:" in out and "0 fail" in out
    assert "UW ledger: 60 requests" in out and "UW 429 rate 0.0%" in out
    assert "DPOF flow pillars from UW: 50%" in out                  # one uw + one selfflow reading ...
    assert "(1 records)" in out                                     # ... and the other-day row is ignored
    assert "MEMORY.md 5,000 B" in out and "decision outbox backlog: 0" in out


def test_end_to_end_bad_day_exits_one(tmp_path, monkeypatch, capsys):
    root, db, memory = _fake_root(tmp_path, ledger_requests=60, status=429)
    _patch(monkeypatch, root, db, memory, journal_text="Traceback\n", services="failed")
    assert mc.main() == 1
    out = capsys.readouterr().out
    assert "[FAIL] service moltbot: failed" in out
    assert "[FAIL] UW 429 rate 100.0%" in out
    assert "[FAIL] journal: 1 x 'Traceback'" in out


def test_a_broken_check_does_not_hide_the_others(tmp_path, monkeypatch, capsys):
    root, db, memory = _fake_root(tmp_path, ledger_requests=5)
    _patch(monkeypatch, root, db, memory)
    monkeypatch.setenv("CLIO_UW_LEDGER_DB", str(tmp_path / "does_not_exist.sqlite3"))
    code = mc.main()
    out = capsys.readouterr().out
    assert "ledger report failed" in out                              # reported as a WARN ...
    assert "service moltbot: active" in out and "Kronos forecast age" in out   # ... and the rest still ran
    assert code == 0


# ---------------------------------------------------------------- DPOF pillar liveness + dynamic weights (2026-10-10)
def _pl(prem_rows, prem_nonflat, dp_rows=200, dp_nonflat=0):
    return {"premium": {"rows": prem_rows, "nonflat": prem_nonflat}, "darkpool": {"rows": dp_rows, "nonflat": dp_nonflat}}


@pytest.mark.parametrize("nonflat, level", [(200, "PASS"), (160, "PASS"), (159, "WARN"), (40, "WARN"), (39, "FAIL"), (0, "FAIL")])
def test_premium_liveness_thresholds(nonflat, level):
    """Premium: >= 80% not-flat passes, 20-80% warns, < 20% is a FAIL (the dead-pillar defect class)."""
    out = mc.eval_pillar_liveness(_pl(200, nonflat, dp_rows=200, dp_nonflat=200))
    prem = [lv for lv, tx in out if "pillar premium" in tx][0]
    assert prem == level


def test_premium_with_too_few_readings_is_a_warning_not_a_verdict():
    """Fewer than PILLAR_MIN_ROWS UW-sourced readings cannot be judged (WARN, never FAIL)."""
    out = mc.eval_pillar_liveness(_pl(59, 0))
    assert [lv for lv, tx in out if "premium" in tx] == ["WARN"]
    assert "too few" in [tx for lv, tx in out if "premium" in tx][0]


def test_flat_legacy_darkpool_warns_but_never_fails_and_a_live_one_passes():
    """The legacy darkpool is flat by construction: WARN with an explanation, never FAIL; a moving one passes."""
    flat = [x for x in mc.eval_pillar_liveness(_pl(200, 200, dp_rows=500, dp_nonflat=3)) if "darkpool" in x[1]][0]
    assert flat[0] == "WARN" and "CLIO_DPOF_DARKPOOL_NBBO" in flat[1]
    moving = [x for x in mc.eval_pillar_liveness(_pl(200, 200, dp_rows=500, dp_nonflat=450)) if "darkpool" in x[1]][0]
    assert moving[0] == "PASS"


def test_flat_eps_boundary_counts_as_not_flat(tmp_path, monkeypatch):
    """A reading with |v| exactly 0.005 counts as moving; 0.0049 as flat; non-UW and non-numeric readings are ignored."""
    root = tmp_path / "r"
    (root / "data" / "dpof").mkdir(parents=True)
    today = "2026-10-12"
    rows = []
    for v in (0.005, 0.0049, -0.5, 0.0):
        rows.append({"ts": f"{today}T10:00:00-04:00", "pillars": {"premium": {"available": True, "source": "uw:/x", "value": v}}})
    rows += [
        {"ts": f"{today}T10:00:30-04:00", "pillars": {"premium": {"available": True, "source": "selfflow:SPY", "value": 0.9}}},
        {"ts": f"{today}T10:01:00-04:00", "pillars": {"premium": {"available": False, "source": "uw:/x", "value": 0.9}}},
        {"ts": f"{today}T10:01:30-04:00", "pillars": {"premium": {"available": True, "source": "uw:/x", "value": True}}},
        {"ts": f"{today}T10:02:00-04:00", "pillars": {"premium": {"available": True, "source": "uw:/x", "value": "nan"}}},
        {"ts": "2020-01-01T10:00:00-04:00", "pillars": {"premium": {"available": True, "source": "uw:/x", "value": 0.9}}},
    ]
    (root / "data" / "dpof" / "signals.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    monkeypatch.setattr(mc, "ROOT", root)
    stats = mc._pillar_liveness(today)
    assert stats["premium"] == {"rows": 4, "nonflat": 2}
    assert stats["darkpool"] == {"rows": 0, "nonflat": 0}


def _dw(**kw):
    base = {"shadow_rows": 1000, "signal_rows": 1000, "identity_violations": 0, "identity_eligible": 800, "identity_max_dev": 0.0, "dead_last": {}}
    base.update(kw)
    return base


def test_dynamic_weights_no_shadow_records_is_a_warning():
    """No shadow records today (mode off / not restarted) is a WARN with the likely causes, not a FAIL."""
    out = mc.eval_dynamic_weights(_dw(shadow_rows=0))
    assert [lv for lv, _ in out] == ["WARN"] and "not restarted" in out[0][1]


@pytest.mark.parametrize("shadow, level", [(1000, "PASS"), (900, "PASS"), (899, "WARN")])
def test_dynamic_weights_coverage_threshold(shadow, level):
    """Shadow coverage of >= 90% of the signal rows passes; below it warns."""
    out = mc.eval_dynamic_weights(_dw(shadow_rows=shadow))
    assert out[0][0] == level and "coverage" in out[0][1]


def test_dynamic_weights_identity_violation_is_a_fail():
    """Any nothing-dead row where dyn differs from static is a FAIL; none is a PASS."""
    bad = mc.eval_dynamic_weights(_dw(identity_violations=3, identity_max_dev=0.01))
    assert any(lv == "FAIL" and "dynamic must equal static" in tx for lv, tx in bad)
    good = mc.eval_dynamic_weights(_dw())
    assert any(lv == "PASS" and "identity holds on 800" in tx for lv, tx in good)
    none_eligible = mc.eval_dynamic_weights(_dw(identity_eligible=0))
    assert not any("identity" in tx for _lv, tx in none_eligible)


def test_dynamic_weights_dead_pillar_levels():
    """Only-darkpool-dead is the expected legacy state (PASS); any other dead pillar warns."""
    legacy = mc.eval_dynamic_weights(_dw(dead_last={"SPY": ["darkpool"], "QQQ": ["darkpool"]}))
    assert legacy[-1][0] == "PASS" and "legacy darkpool" in legacy[-1][1]
    other = mc.eval_dynamic_weights(_dw(dead_last={"SPY": ["premium", "darkpool"], "QQQ": []}))
    assert other[-1][0] == "WARN" and "premium" in other[-1][1]
    none_dead = mc.eval_dynamic_weights(_dw(dead_last={"SPY": [], "QQQ": []}))
    assert none_dead[-1] == ("PASS", "dynamic weights: no pillar currently flagged dead")


def test_dynamic_weights_collector_reads_today_only_and_checks_the_identity_invariant(tmp_path, monkeypatch):
    """The collector counts today's shadow/signal rows, flags a planted |dyn-static| violation, and ignores dead/ramping rows."""
    root = tmp_path / "r"
    (root / "data" / "dpof").mkdir(parents=True)
    today = "2026-10-12"
    sig = [{"ts": f"{today}T10:00:0{i}-04:00", "underlying": "SPY"} for i in range(5)] + [{"ts": "2020-01-01T00:00:00"}]
    (root / "data" / "dpof" / "signals.jsonl").write_text("\n".join(json.dumps(r) for r in sig) + "\n")
    ok = {"alive": {"nofa": 1.0, "premium": 1.0}, "m": {"nofa": 1.0}, "dead": []}
    shadow = [
        {"ts": f"{today}T10:00:00-04:00", "u": "SPY", "static": 0.2, "dyn": 0.2, **ok},
        {"ts": f"{today}T10:00:01-04:00", "u": "SPY", "static": 0.2, "dyn": 0.2000001, **ok},                     # violation
        {"ts": f"{today}T10:00:02-04:00", "u": "QQQ", "static": 0.2, "dyn": 0.9, "alive": {"nofa": 1.0}, "m": {}, "dead": ["darkpool"]},  # dead: ignored
        {"ts": f"{today}T10:00:03-04:00", "u": "QQQ", "static": 0.2, "dyn": 0.9, "alive": {"nofa": 0.5}, "m": {}, "dead": []},         # ramping: ignored
        {"ts": "2020-01-01T10:00:00-04:00", "u": "SPY", "static": 0.0, "dyn": 0.5, **ok},
    ]
    (root / "data" / "dpof" / "pillar_weights_shadow.jsonl").write_text("\n".join(json.dumps(r) for r in shadow) + "\ngarbage\n")
    monkeypatch.setattr(mc, "ROOT", root)
    info = mc._dynamic_weights_info(today)
    assert info["signal_rows"] == 5 and info["shadow_rows"] == 4
    assert info["identity_eligible"] == 2 and info["identity_violations"] == 1
    assert info["identity_max_dev"] == pytest.approx(1e-7, rel=1e-3)
    assert info["dead_last"] == {"SPY": [], "QQQ": []}


def test_journal_patterns_cover_the_dynamic_weights_failure_messages():
    """Both the data-layer and manager failure messages end in '); static weights' and trip the journal WARN."""
    text = ("DPOF dynamic weights observe failed (RuntimeError: x); static weights for this cycle\n"
            "DPOF dynamic weights: finish_snapshot failed (KeyError: y); static weights\n"
            "DPOF dynamic weights disabled: manager init failed (TypeError: z)\n")
    out = dict((tx.split("'")[1], lv) for lv, tx in mc.eval_journal(text) if "'" in tx and lv != "PASS")
    assert out.get("); static weights") == "WARN" and out.get("dynamic weights disabled") == "WARN"


def test_end_to_end_flat_premium_day_exits_one_with_a_clear_line(tmp_path, monkeypatch, capsys):
    """A session whose UW premium readings are all flat makes the morning check FAIL and say so."""
    root, db, memory = _fake_root(tmp_path, ledger_requests=60)
    rows = [{"ts": (FIXED + timedelta(seconds=30 * i)).isoformat(),
             "pillars": {"premium": {"available": True, "source": "uw:/market/market-tide+/stock/SPY/net-prem-ticks", "value": 0.0}}}
            for i in range(80)]
    (root / "data" / "dpof" / "signals.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    _patch(monkeypatch, root, db, memory)
    assert mc.main() == 1
    out = capsys.readouterr().out
    assert "[FAIL] pillar premium (UW-sourced): 0% of 80 readings are not flat" in out and "looks DEAD" in out


def test_end_to_end_alive_premium_day_has_no_pillar_fail(tmp_path, monkeypatch, capsys):
    """A session with a moving premium pillar and matching shadow log produces PASS lines and exit 0."""
    root, db, memory = _fake_root(tmp_path, ledger_requests=60)
    rows = [{"ts": (FIXED + timedelta(seconds=30 * i)).isoformat(),
             "pillars": {"premium": {"available": True, "source": "uw:/x", "value": 0.4 if i % 2 else -0.4}}} for i in range(80)]
    (root / "data" / "dpof" / "signals.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    shadow = [{"ts": r["ts"], "u": "SPY", "static": 0.1, "dyn": 0.1, "alive": {"premium": 1.0}, "m": {}, "dead": []} for r in rows]
    (root / "data" / "dpof" / "pillar_weights_shadow.jsonl").write_text("\n".join(json.dumps(r) for r in shadow) + "\n")
    _patch(monkeypatch, root, db, memory)
    assert mc.main() == 0
    out = capsys.readouterr().out
    assert "[PASS] pillar premium (UW-sourced): 100% of 80 readings are not flat" in out
    assert "[PASS] dynamic weights: 80 shadow records vs 80 signal rows (100% coverage)" in out
    assert "[PASS] dynamic weights: identity holds on 80 nothing-dead rows" in out


def test_darkpool_pass_cutoff_is_half_and_exactly_min_rows_is_judged():
    """Darkpool passes at exactly 50% not-flat and warns just below; exactly PILLAR_MIN_ROWS readings are judged, not 'too few'."""
    below = [x for x in mc.eval_pillar_liveness(_pl(200, 200, dp_rows=500, dp_nonflat=249)) if "darkpool" in x[1]][0]
    at = [x for x in mc.eval_pillar_liveness(_pl(200, 200, dp_rows=500, dp_nonflat=250)) if "darkpool" in x[1]][0]
    assert below[0] == "WARN" and at[0] == "PASS"
    judged = [x for x in mc.eval_pillar_liveness(_pl(60, 60)) if "pillar premium" in x[1]][0]
    assert judged[0] == "PASS" and "too few" not in judged[1]
    assert [x for x in mc.eval_pillar_liveness(_pl(60, 0)) if "pillar premium" in x[1]][0][0] == "FAIL"


def test_identity_check_uses_the_applied_multiplier_not_the_diagnostic_one(tmp_path, monkeypatch):
    """A row whose diagnostic 'm' drifted off 1.0 but whose applied 'ma' is 1.0 still counts (and can flag a violation);
    a legacy row without 'ma' falls back to 'm'; a row whose applied multiplier is not 1.0 is skipped."""
    root = tmp_path / "r"
    (root / "data" / "dpof").mkdir(parents=True)
    today = "2026-10-12"
    base = {"alive": {"nofa": 1.0, "premium": 1.0}, "dead": []}
    rows = [
        {"ts": f"{today}T10:00:00-04:00", "u": "SPY", "static": 0.2, "dyn": 0.2, "m": {"nofa": 1.07}, "ma": {"nofa": 1.0}, **base},        # eligible, ok
        {"ts": f"{today}T10:00:01-04:00", "u": "SPY", "static": 0.2, "dyn": 0.3, "m": {"nofa": 0.93}, "ma": {"nofa": 1.0}, **base},        # eligible, VIOLATION
        {"ts": f"{today}T10:00:02-04:00", "u": "SPY", "static": 0.2, "dyn": 0.9, "m": {"nofa": 1.0}, "ma": {"nofa": 1.1}, **base},         # applied != 1: skipped
        {"ts": f"{today}T10:00:03-04:00", "u": "SPY", "static": 0.2, "dyn": 0.2, "m": {"nofa": 1.0}, **base},                              # legacy row, m == 1: eligible
        {"ts": f"{today}T10:00:04-04:00", "u": "SPY", "static": 0.2, "dyn": 0.9, "m": {"nofa": 1.1}, **base},                              # legacy row, m != 1: skipped
    ]
    (root / "data" / "dpof" / "pillar_weights_shadow.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(mc, "ROOT", root)
    info = mc._dynamic_weights_info(today)
    assert info["identity_eligible"] == 3 and info["identity_violations"] == 1
    assert info["identity_max_dev"] == pytest.approx(0.1, rel=1e-6)


# ---------------------------------------------------------------- OneProvider gateway (OP-11)
GW_NOW = 1_800_000_000.0


def _gw_info(**kw):
    """A healthy balance-state dict at GW_NOW; keyword overrides replace fields (None removes one)."""
    base = {
        "balance_usd": 349.91, "expires_at": datetime.fromtimestamp(GW_NOW + 89.5 * 86400, timezone.utc).isoformat(),
        "is_active": True, "sample_ts": GW_NOW - 60, "level": "ok", "reasons": [], "burn_per_day": 2.10,
        "last_error": None, "canary": {"results": [{"requested": "claude-opus-5-5", "served": "claude-opus-5",
                                                   "latency_ms": 900, "family_changed": False, "ok": True}],
                                       "family_changed": False, "ok": True},
        "last_alert": None, "checked_at": "2026-10-11T10:00:00+00:00",
    }
    for k, v in kw.items():
        if v is None:
            base.pop(k, None)
        else:
            base[k] = v
    return base


def test_gateway_balance_pass_text_is_exact():
    lv, tx = mc.eval_gateway(_gw_info(), GW_NOW, "")[0]
    assert lv == "PASS"
    assert tx == "gateway: balance $349.91, burn $2.10/day, expires in 89d"


@pytest.mark.parametrize("age_s, level", [
    (12 * 3600, "WARN"),          # exactly 12 h old is stale (>= boundary)
    (12 * 3600 - 1, "PASS"),      # one second younger is still fresh
])
def test_gateway_balance_12h_boundary(age_s, level):
    lv, tx = mc.eval_gateway(_gw_info(sample_ts=GW_NOW - age_s), GW_NOW, "")[0]
    assert lv == level
    if level == "WARN":
        assert "last good sample 12.0 h ago" in tx


@pytest.mark.parametrize("state_level, level", [("ok", "PASS"), ("warn", "WARN"), ("crit", "FAIL"), (None, "WARN")])
def test_gateway_balance_level_mapping(state_level, level):
    lv, tx = mc.eval_gateway(_gw_info(level=state_level), GW_NOW, "")[0]
    assert lv == level
    assert ("level crit" in tx) if state_level == "crit" else True


def test_gateway_balance_crit_fails_even_with_a_fresh_sample():
    lv, _ = mc.eval_gateway(_gw_info(level="crit", reasons=["low_balance"]), GW_NOW, "")[0]
    assert lv == "FAIL"


def test_gateway_inactive_key_is_a_fail_even_when_level_ok():
    lv, tx = mc.eval_gateway(_gw_info(is_active=False), GW_NOW, "")[0]
    assert lv == "FAIL" and "is_active False" in tx


def test_gateway_warn_reason_text_and_expiry_past_is_expired():
    past = datetime.fromtimestamp(GW_NOW - 3600, timezone.utc).isoformat()
    lv, tx = mc.eval_gateway(_gw_info(level="warn", expires_at=past), GW_NOW, "")[0]
    assert lv == "WARN" and "expired" in tx and "level warn" in tx


def test_gateway_no_sample_yet_is_a_warning_not_a_pass():
    """Missing state file ({}) and a file without sample_ts are both WARN."""
    for info in ({}, _gw_info(sample_ts=None)):
        out = mc.eval_gateway(info, GW_NOW, "")
        assert out[0] == ("WARN", "gateway: no balance sample yet (monitor runs every 6 h)")


def test_gateway_canary_pass_text_is_exact():
    lv, tx = mc.eval_gateway(_gw_info(), GW_NOW, "")[1]
    assert lv == "PASS"
    assert tx == "gateway canary: claude-opus-5-5 -> claude-opus-5 (version alias, same family)"


def test_gateway_canary_family_changed_is_a_warning_even_when_ok():
    canary = {"results": [{"requested": "claude-opus-5-5", "served": "claude-sonnet-5", "family_changed": True, "ok": True}],
              "family_changed": True, "ok": True}
    lv, tx = mc.eval_gateway(_gw_info(canary=canary), GW_NOW, "")[1]
    assert lv == "WARN" and "claude-sonnet-5" in tx and "family_changed=True" in tx


@pytest.mark.parametrize("canary", [
    {"results": [{"requested": "a", "served": "b"}], "family_changed": False, "ok": False},
    {"results": [{"requested": "a", "served": "b"}], "family_changed": False},            # ok missing
    {"results": [], "family_changed": False, "ok": True},                                 # nothing was probed
])
def test_gateway_canary_not_ok_is_a_warning(canary):
    assert mc.eval_gateway(_gw_info(canary=canary), GW_NOW, "")[1][0] == "WARN"


def test_gateway_canary_missing_is_a_warning():
    assert mc.eval_gateway(_gw_info(canary=None), GW_NOW, "")[1][0] == "WARN"
    assert mc.eval_gateway({}, GW_NOW, "")[1][0] == "WARN"


GW_STATS = ("2026-10-11T14:00:00 host moltbot[1]: gateway stats: calls=12 empty_retries=1 empty_final={ef} "
            "mismatches=9 latency_ms_p50/p95 by model=claude-opus-5-5=2424/8161,claude-sonnet-5-5=2275/2571 status={{200:12}}")


def test_gateway_stats_clean_pass_uses_max_p95():
    lv, tx = mc.eval_gateway({}, GW_NOW, GW_STATS.format(ef=0))[2]
    assert lv == "PASS"
    assert tx == "gateway stats: calls=12 empty_retries=1 p95 8161ms"


@pytest.mark.parametrize("ef", [1, 3])
def test_gateway_stats_empty_final_is_a_warning(ef):
    lv, tx = mc.eval_gateway({}, GW_NOW, GW_STATS.format(ef=ef))[2]
    assert lv == "WARN" and f"empty_final={ef} of calls=12" in tx


def test_gateway_stats_uses_the_last_stats_line():
    text = GW_STATS.format(ef=0) + "\nunrelated line\n" + GW_STATS.format(ef=2) + "\n"
    assert mc.eval_gateway({}, GW_NOW, text)[2][0] == "WARN"
    assert mc.eval_gateway({}, GW_NOW, GW_STATS.format(ef=2) + "\n" + GW_STATS.format(ef=0))[2][0] == "PASS"


def test_gateway_stats_no_line_outside_the_session_is_a_pass_with_a_note():
    """GW_NOW is 03:00 ET, outside the session: no stats line is a PASS that says so."""
    assert mc.eval_gateway({}, GW_NOW, "nothing relevant\n")[2] == ("PASS", "gateway stats: none in the last 2 h (outside the session)")
    assert mc.eval_gateway({}, GW_NOW, "")[2] == ("PASS", "gateway stats: none in the last 2 h (outside the session)")


def test_gateway_stats_malformed_line_is_a_warning():
    lv, tx = mc.eval_gateway({}, GW_NOW, "gateway stats: calls=x empty_retries=1\n")[2]
    assert lv == "WARN" and "unparseable" in tx


def test_gateway_info_missing_corrupt_and_non_object_are_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "ROOT", tmp_path)
    assert mc._gateway_info() == {}                                       # missing file
    (tmp_path / "data").mkdir()
    state = tmp_path / "data" / "oneprovider_balance_state.json"
    state.write_text("{not json")
    assert mc._gateway_info() == {}                                       # corrupt
    state.write_text("[1, 2]")
    assert mc._gateway_info() == {}                                       # valid JSON, not an object


def test_gateway_info_reads_the_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "oneprovider_balance_state.json").write_text(json.dumps(_gw_info()))
    info = mc._gateway_info()
    assert info["balance_usd"] == 349.91 and info["level"] == "ok"
    assert mc.eval_gateway(info, GW_NOW, "")[0][0] == "PASS"


def test_gateway_end_to_end_reads_journal_for_its_own_window_and_reports_gateway(tmp_path, monkeypatch, capsys):
    root, db, memory = _fake_root(tmp_path, ledger_requests=60)
    (root / "data" / "oneprovider_balance_state.json").write_text(
        json.dumps(_gw_info(sample_ts=time.time() - 60, expires_at=datetime.fromtimestamp(time.time() + 30 * 86400, timezone.utc).isoformat())))
    _patch(monkeypatch, root, db, memory, journal_text=GW_STATS.format(ef=0) + "\n")
    journal_calls = []
    real_run = mc._run

    def counting_run(cmd, timeout=30):
        if cmd and cmd[0] == "journalctl":
            journal_calls.append(cmd)
        return real_run(cmd, timeout)

    monkeypatch.setattr(mc, "_run", counting_run)
    assert mc.main() == 0
    out = capsys.readouterr().out
    # the journal-pattern check reads since the open; the gateway stats read its own 2-hour window
    assert [c[c.index("--since") + 1] for c in journal_calls] == ["2026-09-29 09:30:00", "2 hours ago"]
    assert "[PASS] gateway: balance $349.91, burn $2.10/day, expires in 29d" in out
    assert "[PASS] gateway canary: claude-opus-5-5 -> claude-opus-5" in out
    assert "[PASS] gateway stats: calls=12 empty_retries=1 p95 8161ms" in out


@pytest.mark.parametrize("target", ["eval_gateway", "_gateway_info"])
def test_gateway_failure_is_isolated_in_main(tmp_path, monkeypatch, capsys, target):
    """If the gateway check raises (in its evaluator or its collector), main() must not raise: it reports a WARN
    and every other check still produces its line."""
    root, db, memory = _fake_root(tmp_path, ledger_requests=60)
    _patch(monkeypatch, root, db, memory, journal_text=GW_STATS.format(ef=0) + "\n")

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(mc, target, boom)
    assert mc.main() == 0
    out = capsys.readouterr().out
    assert "[WARN] gateway check failed: RuntimeError: boom" in out
    assert "journal: 0 x 'Traceback'" in out
    assert "MEMORY.md 5,000 B" in out


# ---------------------------------------------------------------- R-12 (S215): gateway window, decider, audit, alignment, disk, backups
SESSION_EPOCH = datetime(2026, 10, 12, 10, 0, tzinfo=NY).timestamp()
OUT_EPOCH = datetime(2026, 10, 12, 20, 0, tzinfo=NY).timestamp()


def test_gateway_stats_missing_in_session_is_a_warning_not_a_pass():
    lv, tx = mc.eval_gateway(_gw_info(), SESSION_EPOCH, "nothing relevant\n")[2]
    assert lv == "WARN" and "no gateway stats line in the last 2 h" in tx


def test_gateway_stats_missing_outside_session_is_a_pass_and_the_flag_overrides_the_clock():
    assert mc.eval_gateway(_gw_info(), OUT_EPOCH, "")[2][0] == "PASS"
    assert mc.eval_gateway(_gw_info(), OUT_EPOCH, "", in_session=True)[2][0] == "WARN"


def test_gateway_stats_accepts_the_appended_family_and_host_fields():
    line = GW_STATS.format(ef=0) + " family_mismatch=0 alias=2 hosts={api.oneprovider.dev:12}"
    lv, tx = mc.eval_gateway({}, SESSION_EPOCH, line + "\n", in_session=True)[2]
    assert lv == "PASS" and tx == "gateway stats: calls=12 empty_retries=1 p95 8161ms"


def _dec(fallback=False, kind="entry", latency_ms=2000, timeout_s=60, claims=()):
    return {"ts": "2026-10-12T10:00:00-04:00", "kind": kind, "provider": "oneprovider", "requested_model": "claude-opus-5-5",
            "served_model": "claude-opus-5", "status": "ok", "latency_ms": latency_ms, "timeout_s": timeout_s, "usage": {},
            "cost_usd": 0.1, "fallback_used": fallback, "final_provider": "fallback" if fallback else "oneprovider",
            "claim_violations": list(claims)}


@pytest.mark.parametrize("fallback, level", [(3, "PASS"), (4, "WARN"), (8, "WARN"), (9, "FAIL")])
def test_decider_fallback_thresholds_are_exact_at_15_and_40_percent(fallback, level):
    """20 deliberations (20 primary rows) plus k fallback rows: 3 (= 15%) passes, 4 (20%) warns, 8 (= 40%) still warns, 9 fails."""
    rows = [_dec() for _ in range(20)] + [_dec(fallback=True) for _ in range(fallback)]
    assert mc.eval_decider_attempts(rows)[1][0] == level


def test_decider_share_divides_by_deliberations_not_by_rows():
    # 3 fallbacks among 17 deliberations = 17.6% -> WARN. Dividing by all 20 rows would give exactly 15% -> PASS (the bug).
    rows = [_dec() for _ in range(17)] + [_dec(fallback=True) for _ in range(3)]
    lv, tx = mc.eval_decider_attempts(rows)[1]
    assert lv == "WARN" and "3 of 17 deliberations" in tx


def test_decider_absent_or_empty_ledger_is_info():
    assert mc.eval_decider_attempts(None) == [("INFO", "decider attempt ledger not present (data/dpof/decider_attempts.jsonl)")]
    assert mc.eval_decider_attempts([])[-1][0] == "INFO"


def test_decider_primary_p95_latency_and_claim_violations():
    slow = [_dec(latency_ms=50_000) for _ in range(10)] + [_dec(fallback=True, latency_ms=500_000)]   # p95 50 s > 0.8 x 60 s
    out = mc.eval_decider_attempts(slow)
    assert ("WARN", "decider primary p95 latency 50.0 s vs timeout 60 s (warn > 0.8x)") in out
    fast = [_dec(latency_ms=40_000) for _ in range(10)]                                              # 40 s <= 48 s
    assert [lv for lv, tx in mc.eval_decider_attempts(fast) if "primary p95" in tx] == ["PASS"]
    cv = mc.eval_decider_attempts([_dec(claims=["x"]), _dec(claims=["a", "b"])])
    assert ("INFO", "decider claim violations today: 3") in cv


AUDIT_KNOWN_DAY = "2026-09-29"    # known baseline 93
AUDIT_KNOWN = mc.AUDIT_UNSIGNED_KNOWN[AUDIT_KNOWN_DAY]


@pytest.mark.parametrize("day, count, level", [
    (AUDIT_KNOWN_DAY, AUDIT_KNOWN, "PASS"),          # known day, equal
    (AUDIT_KNOWN_DAY, AUDIT_KNOWN + 1, "WARN"),      # known day, +1
    ("2026-10-13", 0, "PASS"),                       # unknown day, zero
    ("2026-10-13", 2, "WARN"),                       # unknown day, two
])
def test_audit_unsigned_is_compared_per_day_against_the_known_value(day, count, level):
    out = mc.eval_audit_unsigned({day: count})
    assert out == [(level, out[0][1])] and day in out[0][1]


def test_audit_unsigned_judges_only_the_last_two_days_and_names_counts():
    out = mc.eval_audit_unsigned({"2026-10-09": 999, "2026-10-11": 18, "2026-10-12": 5})
    assert len(out) == 2 and out[0][0] == "PASS" and "2026-10-11" in out[0][1] and out[1][0] == "WARN"
    assert "5 unsigned rows vs known 4" in out[1][1]


def test_decider_collector_reads_only_todays_rows_and_skips_corrupt_lines(tmp_path, monkeypatch):
    (tmp_path / "data" / "dpof").mkdir(parents=True)
    ledger = tmp_path / "data" / "dpof" / "decider_attempts.jsonl"
    ledger.write_text("\n".join([
        json.dumps({"ts": "2026-10-11T15:00:00-04:00", "kind": "entry", "fallback_used": False}),
        "{not json",
        json.dumps({"ts": "2026-10-12T10:00:00-04:00", "kind": "entry", "fallback_used": False}),
        json.dumps({"ts": "2026-10-12T10:00:30-04:00", "kind": "entry", "fallback_used": True}),
        json.dumps(["not", "a", "dict"]),
    ]) + "\n")
    monkeypatch.setattr(mc, "ROOT", tmp_path)
    rows = mc._decider_rows("2026-10-12")
    assert [r["ts"][:10] for r in rows] == ["2026-10-12", "2026-10-12"]
    assert mc._decider_rows("2026-10-13") == []


def test_audit_unsigned_missing_directory_is_info(tmp_path, monkeypatch):
    assert mc.eval_audit_unsigned(None)[0][0] == "INFO"
    assert mc.eval_audit_unsigned({})[0][0] == "INFO"
    monkeypatch.setattr(mc, "ROOT", tmp_path)                      # no data/audit at all
    assert mc._audit_unsigned_by_day() is None
    assert mc.eval_audit_unsigned(mc._audit_unsigned_by_day())[0][0] == "INFO"


def test_audit_collector_counts_missing_and_empty_signatures_per_day_and_tolerates_corrupt_lines(tmp_path, monkeypatch):
    adir = tmp_path / "data" / "audit"
    adir.mkdir(parents=True)
    (adir / "audit_2026-10-11.jsonl").write_text('{"hmac_signature": "abc"}\n{"hmac_signature": ""}\n')    # ignored: not last two
    (adir / "audit_2026-10-12.jsonl").write_text('{"hmac_signature": ""}\n{"x": 1}\nnot json\n[1]\n{"hmac_signature": "ok"}\n')
    (adir / "audit_2026-10-13.jsonl").write_text('{"x": 1}\n{"hmac_signature": null}\n')
    (adir / "audit_notes.jsonl").write_text('{"x": 1}\n')                                             # not an audit-day file
    monkeypatch.setattr(mc, "ROOT", tmp_path)
    # 10-12: empty-string + missing key counted; corrupt, non-object and signed rows are not
    assert mc._audit_unsigned_by_day() == {"2026-10-12": 2, "2026-10-13": 2}


def test_history_alignment_flags_only_signals_starting_later_than_the_shadow_log():
    """WARN when signals starts more than 1.5 d AFTER the shadow log; starting earlier is the healthy direction."""
    cases = [
        ("2026-10-12T10:00:00-04:00", "2026-10-10T10:00:00-04:00", "WARN"),   # signals 2 d newer
        ("2026-10-10T10:00:00-04:00", "2026-10-12T10:00:00-04:00", "PASS"),   # signals 2 d older (healthy)
        ("2026-10-11T10:00:00-04:00", "2026-10-10T10:00:00-04:00", "PASS"),   # 1 d lag, within tolerance
        ("2026-10-11T22:00:00-04:00", "2026-10-10T10:00:00-04:00", "PASS"),   # exactly 1.5 d is not over the limit
    ]
    for sig, shadow, level in cases:
        assert mc.eval_history_alignment(sig, shadow)[0][0] == level, (sig, shadow)
    assert "replay gate at risk" in mc.eval_history_alignment(cases[0][0], cases[0][1])[0][1]
    assert mc.eval_history_alignment(None, cases[0][1]) [0][0] == "INFO"


def test_first_ts_reads_only_the_first_line(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"ts": "2026-10-10T10:00:00-04:00"}\n{"ts": "2026-10-12T10:00:00-04:00"}\n')
    assert mc._first_ts(p) == "2026-10-10T10:00:00-04:00"
    assert mc._first_ts(tmp_path / "absent.jsonl") is None


@pytest.mark.parametrize("free, level", [(25, "PASS"), (20, "PASS"), (19, "WARN"), (10, "WARN"), (9, "FAIL")])
def test_disk_free_thresholds_are_20_and_10_percent_strictly_below(free, level):
    assert mc.eval_disk((100, 100 - free, free))[0][0] == level


def test_backup_dedup_flags_same_size_on_a_new_inode_only():
    assert mc.eval_backup_dedup(((1, 500), (1, 500)), 1)[0][0] == "PASS"     # hard-linked
    assert mc.eval_backup_dedup(((1, 500), (2, 900)), 1)[0][0] == "PASS"     # the file changed
    lv, tx = mc.eval_backup_dedup(((1, 500), (2, 500)), 1)[0]
    assert lv == "WARN" and "link-dest not applied" in tx
    assert mc.eval_backup_dedup(None, 1)[0][0] == "INFO"


def test_backup_total_size_threshold_is_strictly_above_25_gb():
    size_line = lambda total: [lv for lv, tx in mc.eval_backup_dedup(None, total) if tx.startswith("backup size")][0]
    assert size_line(25 * 10**9) == "PASS"
    assert size_line(25 * 10**9 + 1) == "WARN"
    assert size_line(None) == "INFO"


def test_backup_snapshots_are_the_last_two_by_date_and_hour(tmp_path):
    base = tmp_path / "backups"
    for d, h in [("2026-10-11", "23"), ("2026-10-12", "00"), ("2026-10-12", "01"), ("tmp", "02"), ("2026-10-09", "09")]:
        (base / d / h).mkdir(parents=True)
    (base / "2026-10-12" / "notes").mkdir()
    picks = mc._last_two_snapshots(base)
    assert [f"{p.parent.name}/{p.name}" for p in picks] == ["2026-10-12/00", "2026-10-12/01"]
    assert mc._last_two_snapshots(tmp_path / "none") == []


def test_backup_dedup_inputs_compare_the_retrieval_shadow_inodes(tmp_path, monkeypatch):
    base = tmp_path / "backups"
    for h, body in (("00", "a" * 10), ("01", "b" * 10)):
        f = base / "2026-10-12" / h / "openclaw_data" / "personality" / "retrieval_shadow.jsonl"
        f.parent.mkdir(parents=True)
        f.write_text(body)
    monkeypatch.setattr(mc, "BACKUPS_DIR", base)
    monkeypatch.setattr(mc, "_run", lambda cmd, timeout=30: f"12345\t{base}\n")
    pair, total = mc._backup_dedup_inputs()
    assert pair[0][1] == pair[1][1] == 10 and pair[0][0] != pair[1][0]   # different inodes, same size
    assert total == 12345


def test_missing_ledgers_and_backups_are_none_not_exceptions(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "ROOT", tmp_path)
    monkeypatch.setattr(mc, "BACKUPS_DIR", tmp_path / "backups")
    monkeypatch.setattr(mc, "_run", lambda cmd, timeout=30: "")
    assert mc._decider_rows("2026-10-12") is None
    assert mc._audit_unsigned_by_day() is None
    assert mc._first_ts(tmp_path / "data" / "dpof" / "signals.jsonl") is None
    assert mc._backup_dedup_inputs() == (None, None)


def test_end_to_end_new_checks_are_wired_and_absent_inputs_are_info(tmp_path, monkeypatch, capsys):
    root, db, memory = _fake_root(tmp_path, ledger_requests=60)
    _patch(monkeypatch, root, db, memory, journal_text=GW_STATS.format(ef=0) + "\n")
    assert mc.main() == 0
    out = capsys.readouterr().out
    assert "[INFO] decider attempt ledger not present" in out
    assert "[INFO] audit chain: no data/audit/audit_YYYY-MM-DD.jsonl files present" in out
    assert "[INFO] history alignment" in out
    assert "[PASS] disk /home/agent: 50% free (50.0 GB of 100.0 GB)" in out
    assert "[INFO] backup dedup" in out
