"""Tests for scripts/dpof_dynamic_weights_report.py (synthetic data under tmp_path only)."""

import importlib.util
import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dpof_dynamic_weights_report.py"
_spec = importlib.util.spec_from_file_location("dpof_dynamic_weights_report", _SCRIPT)
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)

_ET = ZoneInfo("America/New_York")
_PRIOR = {"nofa": 0.40, "premium": 0.25, "darkpool": 0.20, "kronos": 0.15}
_DEAD_W = {"nofa": 0.40 / 0.75, "darkpool": 0.20 / 0.75, "kronos": 0.15 / 0.75}
_ALL = ["nofa", "premium", "darkpool", "kronos"]


def _composite(values, weights, gex):
    """Independent copy of the composite formula (test oracle)."""
    num = sum(weights[p] * values[p] for p in values)
    den = sum(weights[p] for p in values)
    pre = num / den
    if gex is None:
        comp = num / den
    else:
        sign = (pre > 0) - (pre < 0)
        comp = (num + 0.15 * gex * sign) / (den + 0.15)
    return max(-1.0, min(1.0, comp))


def _values(s, i, u):
    k = i + 7 * s + (0 if u == "SPY" else 3)
    return {
        "nofa": round(0.5 * math.sin(0.3 * k), 4),
        "premium": round(0.6 * math.cos(0.17 * k), 4),
        "darkpool": round(0.4 * math.sin(0.11 * k + 1.0), 4),
        "kronos": round(0.7 * math.cos(0.23 * k + 2.0), 4),
        "gex": round(0.5 * math.sin(0.19 * k + 0.5) + 0.05, 4),
    }


def _write_dataset(data_dir, n_sessions=3, cycles=60, dead=None, tamper=None):
    """Write coherent signals.jsonl + pillar_weights_shadow.jsonl; return their paths."""
    dead = dead or (lambda s, i, u: False)
    tamper = tamper or (lambda s, i, u: None)
    sig_lines, sh_lines = [], []
    for s in range(n_sessions):
        day = date(2026, 10, 1) + timedelta(days=s)
        for i in range(cycles):
            ts = (datetime(day.year, day.month, day.day, 9, 45, tzinfo=_ET)
                  + timedelta(seconds=30 * i)).isoformat()
            spy_static = None
            for u in ("SPY", "QQQ"):
                v = _values(s, i, u)
                is_dead = dead(s, i, u)
                prem = 0.0 if is_dead else v["premium"]
                vals = {"nofa": v["nofa"], "premium": prem,
                        "darkpool": v["darkpool"], "kronos": v["kronos"]}
                static = _composite(vals, _PRIOR, v["gex"])
                if u == "SPY":
                    spy_static = static
                    basis, logged = None, static
                else:
                    basis, logged = "SPY", spy_static
                if is_dead:
                    w = dict(_DEAD_W)
                    avail = ["nofa", "darkpool", "kronos"]
                    dead_list = ["premium"]
                    alive = {"nofa": 1.0, "premium": 0.0, "darkpool": 1.0, "kronos": 1.0}
                    dyn = _composite({p: vals[p] for p in avail}, w, v["gex"])
                else:
                    w = dict(_PRIOR)
                    avail = list(_ALL)
                    dead_list = []
                    alive = {p: 1.0 for p in _ALL}
                    dyn = _composite(vals, _PRIOR, v["gex"])
                t = tamper(s, i, u)
                if t == "identity" and not is_dead:
                    dyn = dyn + 1e-6
                if t == "w":
                    w["nofa"] = w["nofa"] + 0.05
                if t == "dyn_zero":
                    dyn = 0.0
                sh_lines.append({
                    "ts": ts, "u": u, "mode": "shadow", "static": static, "dyn": dyn,
                    "w": w, "alive": alive, "m": {p: 1.0 for p in _ALL},
                    "n_eff": {"nofa": 100.0 + i, "premium": 50.0, "darkpool": 40.0, "kronos": 30.0},
                    "avail": avail, "dead": dead_list,
                })
                sig_lines.append({
                    "ts": ts, "underlying": u, "composite": logged, "direction": "bullish",
                    "signal_basis": basis, "is_signal_symbol": u == "SPY",
                    "pillars": {
                        "nofa": {"value": v["nofa"], "available": True, "source": "uw"},
                        "premium": {"value": prem, "available": True, "source": "uw"},
                        "darkpool": {"value": v["darkpool"], "available": True, "source": "uw"},
                        "gex": {"value": v["gex"], "available": True, "source": "uw"},
                    },
                    "kronos": {"pillar": v["kronos"]},
                })
    os.makedirs(data_dir, exist_ok=True)
    sig_path = os.path.join(data_dir, "signals.jsonl")
    sh_path = os.path.join(data_dir, "pillar_weights_shadow.jsonl")
    with open(sig_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in sig_lines)
    with open(sh_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in sh_lines)
    return sh_path, sig_path


def _load(data_dir):
    return (rep.load_jsonl(os.path.join(data_dir, "pillar_weights_shadow.jsonl")),
            rep.load_jsonl(os.path.join(data_dir, "signals.jsonl")))


def _ts(day, hh, mm, ss=0):
    return datetime(2026, 10, day, hh, mm, ss, tzinfo=_ET).isoformat()


def _rec(ts, u="SPY", static=0.0, dyn=0.0, dead=(), w=None, alive=None, m=None):
    return {"ts": ts, "u": u, "static": static, "dyn": dyn, "dead": list(dead),
            "w": w if w is not None else dict(_PRIOR),
            "alive": alive if alive is not None else {p: 1.0 for p in _ALL},
            "m": m if m is not None else {p: 1.0 for p in _ALL},
            "avail": list(_ALL)}


def _tree_state(root):
    """Relative path -> (size, mtime_ns) for every entry under root."""
    state = {}
    for dp, dns, fns in os.walk(root):
        for name in dns + fns:
            full = os.path.join(dp, name)
            st = os.lstat(full)
            state[os.path.relpath(full, root)] = (st.st_size, st.st_mtime_ns)
    return state


def test_loader_skips_torn_lines_and_handles_missing_or_empty(tmp_path):
    """Torn, blank and non-object lines are skipped; missing and empty files give []."""
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"b": 2\n\n[1, 2]\n{"c": 3}\n', encoding="utf-8")
    assert rep.load_jsonl(str(p)) == [{"a": 1}, {"c": 3}]
    assert rep.load_jsonl(str(p), max_lines=1) == [{"c": 3}]
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert rep.load_jsonl(str(empty)) == []
    assert rep.load_jsonl(str(tmp_path / "missing.jsonl")) == []


def test_session_day_uses_et_date_for_both_offsets():
    """ET calendar date is taken from -04:00, -05:00 and UTC-stamped timestamps."""
    assert rep.session_day("2026-10-01T23:59:00-04:00") == "2026-10-01"
    assert rep.session_day("2026-03-02T09:30:00-05:00") == "2026-03-02"
    assert rep.session_day("2026-10-02T01:00:00+00:00") == "2026-10-01"
    assert rep.session_day("not a time") is None


def test_coverage_detects_planted_ten_minute_gap():
    """A 20-cycle hole in SPY during RTH is reported as a gap of at least 10 minutes."""
    recs = []
    for i in range(60):
        ts = (datetime(2026, 10, 1, 9, 45, tzinfo=_ET) + timedelta(seconds=30 * i)).isoformat()
        if not 20 <= i < 40:
            recs.append({"ts": ts, "u": "SPY"})
        recs.append({"ts": ts, "u": "QQQ"})
    cov = rep.coverage(recs)
    gaps = cov["2026-10-01"]["gaps"]
    assert [g["u"] for g in gaps] == ["SPY"]
    assert gaps[0]["minutes"] >= 10
    assert cov["2026-10-01"]["cycles"] == {"SPY": 40, "QQQ": 60}


def test_dead_episode_start_end_and_flip_count(tmp_path):
    """The planted premium dead run in session 1 has the right start, revival end and flip count."""
    sh, _sig = _write_dataset(str(tmp_path), n_sessions=3,
                              dead=lambda s, i, u: s == 0 and i >= 20)
    out = rep.dead_episodes(rep.load_jsonl(sh))
    spy = [e for e in out["episodes"] if e["u"] == "SPY"]
    assert len(spy) == 1
    assert spy[0]["pillar"] == "premium"
    assert spy[0]["start"] == _ts(1, 9, 55) and spy[0]["end"] == _ts(2, 9, 45)
    assert out["flips"]["SPY|premium"] == 2
    assert out["max_flips_per_30min_overall"] == 1


def test_identity_ignores_dead_rows_and_is_empty_when_identical():
    """Identical rows give no violation; a row with a dead pillar is never checked."""
    recs = [_rec(_ts(1, 10, k, 0), static=0.2, dyn=0.2) for k in range(5)]
    recs.append(_rec(_ts(1, 10, 5), static=0.2, dyn=0.9, dead=["premium"],
                     alive={"nofa": 1.0, "premium": 0.0, "darkpool": 1.0, "kronos": 1.0}))
    # Inconsistent row: dead list set but alive and m all 1.0, so only the dead-list guard excludes it.
    recs.append(_rec(_ts(1, 10, 6), static=0.2, dyn=0.9, dead=["premium"]))
    viol, max_dev = rep.identity_violations(recs)
    assert viol == []
    assert max_dev == 0.0


def test_identity_reports_planted_deviation_and_respects_tolerance():
    """A 1e-6 deviation is reported with its size; a 1e-13 deviation is within tolerance."""
    recs = [_rec(_ts(1, 10, 0), static=0.2, dyn=0.2),
            _rec(_ts(1, 10, 1), static=0.3, dyn=0.3 + 1e-6),
            _rec(_ts(1, 10, 2), static=0.4, dyn=0.4 + 1e-13)]
    viol, max_dev = rep.identity_violations(recs)
    assert len(viol) == 1
    assert viol[0]["ts"] == _ts(1, 10, 1)
    assert max_dev == pytest.approx(1e-6, abs=1e-9)


def test_episodes_merge_within_gap_including_exact_boundary():
    """Crossings 30 s and exactly 300 s apart form one episode that starts at the first crossing."""
    series = [(_ts(1, 10, 0, 0), 0.2), (_ts(1, 10, 0, 30), 0.25),
              (_ts(1, 10, 5, 30), 0.3), (_ts(1, 10, 5, 45), 0.1)]
    assert rep.episodes(series) == [(_ts(1, 10, 0, 0), 1)]


def test_episodes_split_on_gap_above_300_seconds():
    """A crossing 301 s after the previous one starts a new episode."""
    series = [(_ts(1, 10, 0, 0), 0.2), (_ts(1, 10, 5, 1), 0.2)]
    assert rep.episodes(series) == [(_ts(1, 10, 0, 0), 1), (_ts(1, 10, 5, 1), 1)]


def test_episodes_split_on_sign_flip():
    """Consecutive crossings of opposite sign form two episodes with signs +1 and -1."""
    series = [(_ts(1, 10, 0, 0), 0.2), (_ts(1, 10, 0, 30), -0.2)]
    assert rep.episodes(series) == [(_ts(1, 10, 0, 0), 1), (_ts(1, 10, 0, 30), -1)]


def test_divergence_band_edges_and_out_of_band_flag():
    """Ratios of exactly 0.67 and 1.5 are in band; 0.66 is flagged out of band."""
    recs = []

    def session(day, n_static, n_dyn):
        # 301 s spacing keeps each crossing its own episode, all within one ET day.
        for k in range(100):
            ts = (datetime(2026, 10, day, 0, 0, tzinfo=_ET) + timedelta(seconds=301 * k)).isoformat()
            recs.append(_rec(ts, static=0.35 if k < n_static else 0.0,
                             dyn=0.35 if k < n_dyn else 0.0))

    session(1, 100, 67)   # ratio 0.67 exactly
    session(2, 100, 66)   # ratio 0.66 -> out of band
    session(3, 2, 3)      # ratio 1.5 exactly
    out = rep.divergence(recs)
    by_day = {r["session"]: r for r in out["rows"]}
    assert by_day["2026-10-01"]["ratio"] == 0.67 and by_day["2026-10-01"]["in_band"]
    assert by_day["2026-10-02"]["in_band"] is False
    assert by_day["2026-10-03"]["ratio"] == 1.5 and by_day["2026-10-03"]["in_band"]
    assert [o["session"] for o in out["out_of_band"]] == ["2026-10-02"]
    assert by_day["2026-10-01"]["static_ge_0.30"] == 100


def test_weight_stability_uses_session_mean_not_max():
    """Max session-mean change is 0.10 (means 0.5 vs 0.4), not the 0.30 max-to-max change."""
    recs = [
        _rec(_ts(1, 10, 0), w={"nofa": 0.40, "premium": 0.25}),
        _rec(_ts(1, 10, 1), w={"nofa": 0.40, "premium": 0.25}),
        _rec(_ts(1, 10, 2), w={"nofa": 0.70, "premium": 0.25}),
        _rec(_ts(2, 10, 0), w={"nofa": 0.40, "premium": 0.25}),
    ]
    out = rep.weight_stability(recs)
    assert out["sessions"]["2026-10-01"]["nofa"]["max"] == 0.70
    assert out["sessions"]["2026-10-01"]["nofa"]["mean"] == pytest.approx(0.5)
    assert out["max_session_mean_change"]["nofa"] == pytest.approx(0.10)
    assert out["max_session_mean_change"]["premium"] == pytest.approx(0.0)


def test_replay_reproduces_dyn_and_static_for_every_row(tmp_path):
    """Replay rebuilds all 360 dyn values and matches every static, with join rate 1.0."""
    _write_dataset(str(tmp_path), n_sessions=3)
    shadow, signals = _load(str(tmp_path))
    out = rep.replay_check(shadow, signals)
    assert out["shadow_rows"] == 360
    assert out["dyn_match"] == 360 and out["dyn_mismatch"] == 0
    assert out["join_rate"] == 1.0
    assert out["mismatches"] == 0


def test_replay_detects_planted_wrong_weight(tmp_path):
    """A wrong w on one shadow row is reported as exactly one dyn mismatch with an example."""
    _write_dataset(str(tmp_path), n_sessions=3,
                   tamper=lambda s, i, u: "w" if (s, i, u) == (1, 5, "SPY") else None)
    shadow, signals = _load(str(tmp_path))
    out = rep.replay_check(shadow, signals)
    assert out["dyn_mismatch"] == 1
    assert out["examples"][0]["kind"] == "dyn"


def test_replay_excludes_qqq_rows_with_spy_signal_basis(tmp_path):
    """QQQ rows whose logged composite is SPY's are excluded from the static comparison."""
    _write_dataset(str(tmp_path), n_sessions=3)
    shadow, signals = _load(str(tmp_path))
    out = rep.replay_check(shadow, signals)
    assert out["static_checked"] == 180
    assert out["static_mismatch"] == 0


def test_l2_reports_last_n_eff_and_multiplier(tmp_path):
    """The last record per underlying supplies n_eff and m for each pillar."""
    _write_dataset(str(tmp_path), n_sessions=2, cycles=10)
    shadow, _sig = _load(str(tmp_path))
    out = rep.l2_diagnostic(shadow)
    assert out["SPY"]["premium"] == {"n_eff": 50.0, "m": 1.0}
    assert out["QQQ"]["nofa"]["n_eff"] == 109.0


def test_promotion_insufficient_with_three_and_five_sessions(tmp_path):
    """Three or five sessions leave the session-count-dependent items INSUFFICIENT."""
    for n in (3, 5):
        d = tmp_path / f"n{n}"
        _write_dataset(str(d), n_sessions=n, cycles=20)
        shadow, signals = _load(str(d))
        report = rep.build_report(shadow, signals)
        assert report["session_count"] == n
        # items 1, 2, 3 and 4b depend on the session count; 4a (identity) does not.
        assert [c["status"] for c in report["checklist"]] == [
            "INSUFFICIENT", "INSUFFICIENT", "INSUFFICIENT", "PASS", "INSUFFICIENT"]


def test_promotion_passes_on_clean_fifteen_sessions(tmp_path):
    """With 15 clean sessions (nothing dead, dyn == static) every checklist item is PASS."""
    _write_dataset(str(tmp_path), n_sessions=15, cycles=60)
    shadow, signals = _load(str(tmp_path))
    report = rep.build_report(shadow, signals)
    assert [c["status"] for c in report["checklist"]] == ["PASS"] * 5


def test_promotion_fails_on_flip_storm(tmp_path):
    """A premium flip every cycle breaks the <= 1 flip per 30 minutes item."""
    _write_dataset(str(tmp_path), n_sessions=15, cycles=60, dead=lambda s, i, u: i % 2 == 0)
    shadow, signals = _load(str(tmp_path))
    report = rep.build_report(shadow, signals)
    assert report["checklist"][1]["status"] == "FAIL"
    assert report["checklist"][3]["status"] == "PASS"


def test_promotion_fails_on_planted_identity_deviation(tmp_path):
    """A 1e-6 deviation on a healthy row fails the identity item."""
    _write_dataset(str(tmp_path), n_sessions=15, cycles=60,
                   tamper=lambda s, i, u: "identity" if (s, i, u) == (3, 5, "SPY") else None)
    shadow, signals = _load(str(tmp_path))
    report = rep.build_report(shadow, signals)
    assert report["checklist"][3]["status"] == "FAIL"
    assert report["identity"]["violations"] == 1


def test_promotion_fails_on_replay_mismatch(tmp_path):
    """A planted wrong weight fails the replay item."""
    _write_dataset(str(tmp_path), n_sessions=15, cycles=60,
                   tamper=lambda s, i, u: "w" if (s, i, u) == (3, 5, "SPY") else None)
    shadow, signals = _load(str(tmp_path))
    report = rep.build_report(shadow, signals)
    assert report["checklist"][0]["status"] == "FAIL"


def test_promotion_fails_on_weight_move_above_cap(tmp_path):
    """A whole session with premium dead moves the nofa session mean by 0.133 and fails the cap."""
    _write_dataset(str(tmp_path), n_sessions=15, cycles=60, dead=lambda s, i, u: s == 5)
    shadow, signals = _load(str(tmp_path))
    report = rep.build_report(shadow, signals)
    assert report["checklist"][4]["status"] == "FAIL"


def test_promotion_fails_on_episode_ratio_out_of_band(tmp_path):
    """Dyn forced to zero in one session drops its episode ratio out of the band."""
    _write_dataset(str(tmp_path), n_sessions=15, cycles=60,
                   tamper=lambda s, i, u: "dyn_zero" if s == 3 else None)
    shadow, signals = _load(str(tmp_path))
    report = rep.build_report(shadow, signals)
    assert report["checklist"][2]["status"] == "FAIL"
    assert report["divergence"]["out_of_band"]


def test_dumps_maps_nan_and_inf_to_null():
    """NaN and infinity become null and the output parses as strict JSON."""
    text = rep.dumps({"x": float("nan"), "y": [float("inf"), 1.5], "z": (1, 2)})
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"x": None, "y": [None, 1.5], "z": [1, 2]}


def test_main_text_and_json_write_nothing_and_emit_no_nan(tmp_path, monkeypatch, capsys):
    """Text and JSON runs leave the data dir and cwd unchanged and emit strict JSON."""
    data = tmp_path / "data"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_dataset(str(data), n_sessions=3)
    monkeypatch.chdir(cwd)
    before_data, before_cwd = _tree_state(str(data)), _tree_state(str(cwd))

    assert rep.main(["--data-dir", str(data)]) == 0
    text = capsys.readouterr().out
    assert "== Promotion checklist ==" in text and "Replay check" in text

    assert rep.main(["--data-dir", str(data), "--json"]) == 0
    raw = capsys.readouterr().out
    assert "NaN" not in raw and "Infinity" not in raw
    parsed = json.loads(raw)
    assert parsed["session_count"] == 3
    assert parsed["replay"]["dyn_match"] == 360

    assert _tree_state(str(data)) == before_data
    assert _tree_state(str(cwd)) == before_cwd


def test_main_no_data_is_clean_exit_zero(tmp_path, capsys):
    """Missing or empty inputs give a clear no-data report and exit 0."""
    (tmp_path / "pillar_weights_shadow.jsonl").write_text("", encoding="utf-8")
    assert rep.main(["--data-dir", str(tmp_path / "absent")]) == 0
    assert "NO DATA" in capsys.readouterr().out
    assert rep.main(["--data-dir", str(tmp_path)]) == 0
    assert "NO DATA" in capsys.readouterr().out


def test_main_days_selects_last_sessions(tmp_path, capsys):
    """--days 1 keeps only the newest session in the JSON report."""
    _write_dataset(str(tmp_path), n_sessions=3, cycles=20)
    assert rep.main(["--data-dir", str(tmp_path), "--json", "--days", "1"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["sessions"] == ["2026-10-03"]
    assert parsed["session_count"] == 1


# ------------------------------------------- reviewer additions (mutation gaps)

def _mk_rec(ts, u="SPY", static=0.0, dyn=0.0, dead=(), w=None, avail=None):
    return {"ts": ts, "u": u, "mode": "shadow", "static": static, "dyn": dyn, "w": w or {}, "alive": {},
            "m": {}, "n_eff": {}, "avail": list(avail or []), "dead": list(dead)}


def _checklist_report(**over):
    base = {
        "session_count": 15,
        "replay": {"mismatches": 0, "join_rate": 1.0},
        "identity": {"eligible": 10, "violations": 0, "max_dev": 0.0},
        "dead_episodes": {"max_flips_per_30min_overall": 0},
        "divergence": {"out_of_band": []},
        "weight_stability": {"max_change": 0.0},
    }
    base.update(over)
    return base


def _pick_status(items, needle):
    return next(i["status"] for i in items if needle in i["criterion"])


def test_episode_threshold_is_inclusive_a_value_exactly_at_0_15_is_a_crossing():
    """|value| == 0.15 counts as a nomination crossing (>= threshold)."""
    assert len(rep.episodes([("2026-10-09T10:00:00-04:00", 0.15)])) == 1
    assert rep.episodes([("2026-10-09T10:00:00-04:00", 0.1499999)]) == []


def test_flip_bucket_is_thirty_minutes_not_an_hour():
    """Two flips inside one 30-minute bucket and a third in the next bucket give max 2 per 30 minutes (not 3)."""
    recs = [
        _mk_rec("2026-10-09T10:00:00-04:00"),
        _mk_rec("2026-10-09T10:01:00-04:00", dead=["premium"]),
        _mk_rec("2026-10-09T10:15:00-04:00"),
        _mk_rec("2026-10-09T10:40:00-04:00", dead=["premium"]),
    ]
    assert rep.dead_episodes(recs)["max_flips_per_30min_overall"] == 2


def test_divergence_ratio_above_the_upper_band_edge_is_out_of_band():
    """One static episode versus two dyn episodes is ratio 2.0, which is above the 1.5 upper edge."""
    recs = [
        _mk_rec("2026-10-09T10:00:00-04:00", static=0.2, dyn=0.2),
        _mk_rec("2026-10-09T10:10:00-04:00", static=0.0, dyn=0.2),
    ]
    res = rep.divergence(recs)
    assert res["rows"][0]["static_episodes"] == 1 and res["rows"][0]["dyn_episodes"] == 2
    assert res["rows"][0]["ratio"] == 2.0 and res["rows"][0]["in_band"] is False
    assert len(res["out_of_band"]) == 1


@pytest.mark.parametrize("rate,status", [(0.99, "PASS"), (0.9899, "FAIL"), (0.8, "FAIL"), (1.0, "PASS")])
def test_promotion_join_rate_boundary_is_inclusive_at_0_99(rate, status):
    """With 15 sessions and no replay mismatches, join rate >= 0.99 passes and anything below fails."""
    items = rep.promotion_checklist(_checklist_report(replay={"mismatches": 0, "join_rate": rate}))
    assert _pick_status(items, ">= 15 sessions") == status


@pytest.mark.parametrize("flips,status", [(0, "PASS"), (1, "PASS"), (2, "FAIL")])
def test_promotion_flip_rule_allows_at_most_one_flip_per_30_minutes(flips, status):
    """One flip per pillar per 30 minutes passes; two fails."""
    items = rep.promotion_checklist(_checklist_report(dead_episodes={"max_flips_per_30min_overall": flips}))
    assert _pick_status(items, "liveness") == status


@pytest.mark.parametrize("change,status", [(0.1, "PASS"), (0.09, "PASS"), (0.1001, "FAIL")])
def test_promotion_weight_change_cap_is_inclusive_at_0_10(change, status):
    """A session-mean weight change of exactly 0.10 passes; above it fails."""
    items = rep.promotion_checklist(_checklist_report(weight_stability={"max_change": change}))
    assert _pick_status(items, "weight change") == status


def test_replay_examples_are_capped_at_five_even_with_many_mismatches():
    """Eight planted mismatches list at most MAX_EXAMPLES (5) examples but count all of them."""
    sigs, shadow = [], []
    for i in range(8):
        ts = f"2026-10-09T10:{i:02d}:00-04:00"
        sigs.append({"ts": ts, "underlying": "SPY", "composite": 0.1, "signal_basis": None,
                     "pillars": {"nofa": {"value": 0.3, "available": True, "source": "uw"}}, "kronos": {"pillar": None}})
        shadow.append(_mk_rec(ts, static=0.3, dyn=0.9, w={"nofa": 0.4}, avail=["nofa"]))
    res = rep.replay_check(shadow, sigs)
    assert res["mismatches"] >= 8
    assert len(res["examples"]) == 5


# ---------------------------------------------------------------- signal archives

import gzip as _gzip
import io as _io


def _archive_head(data_dir, k, name="signals_20261001_000000.jsonl.gz"):
    """Moves the first k live signals lines into <data_dir>/archive/<name>, as the
    engine's trim does (the live file keeps the tail)."""
    live = os.path.join(data_dir, "signals.jsonl")
    with open(live, encoding="utf-8") as f:
        lines = f.readlines()
    arc_dir = os.path.join(data_dir, "archive")
    os.makedirs(arc_dir, exist_ok=True)
    with _gzip.open(os.path.join(arc_dir, name), "wb") as gz:
        gz.write("".join(lines[:k]).encode("utf-8"))
    with open(live, "w", encoding="utf-8") as f:
        f.writelines(lines[k:])
    return len(lines)


def _json_report(capsys, argv):
    capsys.readouterr()
    assert rep.main(argv) == 0
    cap = capsys.readouterr()
    return json.loads(cap.out), cap.err


def test_load_signals_reads_archives_in_name_order_then_live(tmp_path):
    d = str(tmp_path)
    os.makedirs(os.path.join(d, "archive"))
    for name, rows in (("signals_20261002_000000.jsonl.gz", [{"i": 3}]),
                       ("signals_20261001_000000.jsonl.gz", [{"i": 1}, {"i": 2}])):
        with _gzip.open(os.path.join(d, "archive", name), "wb") as gz:
            gz.write("".join(json.dumps(r) + "\n" for r in rows).encode())
    with open(os.path.join(d, "signals.jsonl"), "w") as f:
        f.write('{"i": 4}\n{"i": 5\n')  # torn live line skipped
    got = rep.load_signals(os.path.join(d, "signals.jsonl"), True, _io.StringIO())
    assert [r["i"] for r in got] == [1, 2, 3, 4]
    assert [r["i"] for r in rep.load_signals(os.path.join(d, "signals.jsonl"), False)] == [4]


def test_join_rate_is_one_only_when_archives_are_included(tmp_path, capsys):
    data = tmp_path / "data"
    _write_dataset(str(data), n_sessions=2)
    _archive_head(str(data), k=len(open(data / "signals.jsonl").readlines()) // 2)

    with_arch, err = _json_report(capsys, ["--data-dir", str(data), "--json"])
    assert with_arch["replay"]["join_rate"] == 1.0
    assert err == ""

    no_arch, _ = _json_report(capsys, ["--data-dir", str(data), "--json", "--no-archive"])
    assert no_arch["replay"]["join_rate"] < 1.0
    assert no_arch["replay"]["joined"] < with_arch["replay"]["joined"]


def test_corrupt_and_truncated_archives_are_skipped_with_a_note(tmp_path, capsys):
    data = tmp_path / "data"
    _write_dataset(str(data), n_sessions=2)
    _archive_head(str(data), k=len(open(data / "signals.jsonl").readlines()) // 2)
    arc = data / "archive"
    (arc / "signals_20250101_000000.jsonl.gz").write_bytes(b"not a gzip file at all")
    good = (arc / "signals_20261001_000000.jsonl.gz").read_bytes()
    (arc / "signals_20261003_000000.jsonl.gz").write_bytes(good[: len(good) // 2])  # truncated

    report, err = _json_report(capsys, ["--data-dir", str(data), "--json"])
    assert "signals_20250101_000000.jsonl.gz" in err
    assert "signals_20261003_000000.jsonl.gz" in err
    assert "signals_20261001_000000.jsonl.gz" not in err
    assert report["replay"]["join_rate"] == 1.0
    assert report["replay"]["mismatches"] == 0


def test_report_writes_nothing_with_or_without_archives(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_dataset(str(data), n_sessions=2)
    _archive_head(str(data), k=20)
    monkeypatch.chdir(cwd)
    before_data, before_cwd = _tree_state(str(data)), _tree_state(str(cwd))

    assert rep.main(["--data-dir", str(data)]) == 0
    assert rep.main(["--data-dir", str(data), "--no-archive", "--json"]) == 0
    capsys.readouterr()

    assert _tree_state(str(data)) == before_data
    assert _tree_state(str(cwd)) == before_cwd


def test_only_signals_archives_are_read_not_foreign_gz_or_leftover_tmp(tmp_path):
    # R-34 fix-up: the archive dir may hold a failed-write leftover (.gz.tmp) or an unrelated gzip; neither is signals data.
    d = str(tmp_path)
    arc = os.path.join(d, "archive")
    os.makedirs(arc)
    good = [{"i": 1}]
    for name, rows in (("signals_20261001_000000.jsonl.gz", good),
                       ("notes_20261001.jsonl.gz", [{"i": 98}]),
                       ("other.gz", [{"i": 97}])):
        with _gzip.open(os.path.join(arc, name), "wb") as gz:
            gz.write("".join(json.dumps(r) + "\n" for r in rows).encode())
    with _gzip.open(os.path.join(arc, "signals_20261002_000000.jsonl.gz.tmp"), "wb") as gz:
        gz.write(b'{"i": 99}\n')  # a complete-looking leftover from a crashed write must still be ignored
    with open(os.path.join(d, "signals.jsonl"), "w") as f:
        f.write('{"i": 2}\n')
    got = rep.load_signals(os.path.join(d, "signals.jsonl"), True, _io.StringIO())
    assert [r["i"] for r in got] == [1, 2]


# ---------------------------------------------------------------- R-38 descriptive alternatives (shadow only)

def _alt_row(live, w60, ewma, day, sat=0, gex=None, ts="2026-10-05T10:00:00-04:00", u="SPY"):
    return {"ts": ts, "u": u, "premium_sat_run": sat,
            "premium_alt": {"live": live, "w60": w60, "ewma": ewma, "day": day}, "gex_cont": gex}


def test_descriptive_stats_on_synthetic_rows_skip_old_and_unavailable_rows():
    recs = [
        _alt_row(1.0, 0.5, 0.8784, 0.5, sat=1, gex=3.0),
        _alt_row(-1.0, -0.2, 0.25, -0.1, sat=2, gex=-1.0),
        _alt_row(0.4, 0.4, -0.4, 0.0, sat=0, gex=None),
        {"ts": "2026-10-05T10:01:00-04:00", "u": "SPY", "static": 0.1},  # old row: no new keys
        {"ts": "2026-10-05T10:02:00-04:00", "u": "SPY", "premium_alt": None, "premium_sat_run": 0,
         "gex_cont": None},  # before the first observation
        _alt_row(float("nan"), 0.1, 0.1, 0.1, sat=True, gex=float("inf")),  # NaN live, bool sat run, inf gex
    ]
    d = rep.descriptive_alternatives(recs)
    assert d["note"] == "descriptive only; no edge claims"
    assert d["premium_alt_rows"] == 4  # the old row and the None premium_alt are skipped
    live = d["premium"]["live"]
    assert live["n"] == 3  # the NaN live value is excluded
    assert live["saturation_share"] == round(2 / 3, 4)  # |1.0| and |-1.0| saturate; 0.4 does not
    assert live["sign_agreement"] is None
    assert d["premium"]["w60"]["n"] == 4
    # w60 vs live pairs where both finite: (0.5,1) yes, (-0.2,-1) yes, (0.4,0.4) yes
    assert d["premium"]["w60"]["sign_agreement"] == 1.0
    # ewma vs live: (0.8784,1) yes, (0.25,-1) no, (-0.4,0.4) no
    assert d["premium"]["ewma"]["sign_agreement"] == round(1 / 3, 4)
    sr = d["premium_sat_run"]
    # runs 1, 2, 0 and the int 0 of the None-premium row count; the bool sat run and the old row do not
    assert sr["n"] == 4 and sr["max"] == 2 and sr["mean"] == 0.75
    gc = d["gex_cont"]
    assert gc["n"] == 2 and gc["mean"] == 1.0 and gc["sd"] == 2.0


def test_descriptive_on_empty_and_old_only_input_is_empty_not_an_error():
    d = rep.descriptive_alternatives([{"ts": "x", "u": "SPY", "static": 0.2}, "junk"])
    assert d["premium_alt_rows"] == 0
    assert d["premium"]["w60"]["n"] == 0 and d["premium"]["w60"]["mean"] is None
    assert d["premium_sat_run"] == {"n": 0, "max": None, "mean": None}
    assert d["gex_cont"]["n"] == 0


def test_text_section_is_labelled_and_old_only_file_reports_no_data(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    _write_dataset(str(data), n_sessions=2)
    assert rep.main(["--data-dir", str(data)]) == 0
    text = capsys.readouterr().out
    assert "== Descriptive premium/GEX alternatives (shadow only) ==" in text
    assert "descriptive only; no edge claims" in text
    assert "no data (rows without the R-37 keys are skipped)" in text


def test_main_with_mixed_old_and_new_rows_reports_section_and_writes_nothing(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    cwd = tmp_path / "cwd"
    data.mkdir()
    cwd.mkdir()
    _write_dataset(str(data), n_sessions=2)
    old_rows = rep.load_jsonl(str(data / "pillar_weights_shadow.jsonl"))
    mixed = []
    for i, r in enumerate(old_rows):
        r = dict(r)
        if i % 2 == 0:  # every other row carries the R-37 keys; the rest stay old-format
            v = 1.0 if i % 4 == 0 else -0.3
            r.update({"premium_alt": {"live": v, "w60": v / 2, "ewma": 0.1, "day": v},
                      "premium_sat_run": 1 if abs(v) >= 0.999 else 0, "gex_cont": 2.0})
        mixed.append(r)
    shadow = tmp_path / "mixed_shadow.jsonl"
    shadow.write_text("".join(json.dumps(r) + "\n" for r in mixed))
    monkeypatch.chdir(cwd)
    before_data, before_cwd = _tree_state(str(data)), _tree_state(str(cwd))
    before_shadow = _tree_state(str(tmp_path))

    assert rep.main(["--data-dir", str(data), "--shadow", str(shadow)]) == 0
    text = capsys.readouterr().out
    assert "== Descriptive premium/GEX alternatives (shadow only) ==" in text
    assert "premium rows with alternatives:" in text

    assert rep.main(["--data-dir", str(data), "--shadow", str(shadow), "--json"]) == 0
    raw = capsys.readouterr().out
    assert "NaN" not in raw and "Infinity" not in raw
    d = json.loads(raw)["descriptive_alternatives"]
    assert d["premium_alt_rows"] == len([r for r in mixed if "premium_alt" in r])
    assert d["gex_cont"]["mean"] == 2.0
    assert d["premium"]["live"]["n"] == d["premium_alt_rows"]

    assert _tree_state(str(data)) == before_data
    assert _tree_state(str(cwd)) == before_cwd
    assert _tree_state(str(tmp_path)) == before_shadow  # the mixed file and the rest of tmp are untouched


def test_descriptive_section_does_not_change_the_existing_checklist(tmp_path):
    """Adding the new keys must leave the promotion checklist and the replay result exactly as before."""
    data = tmp_path / "data"
    data.mkdir()
    _write_dataset(str(data), n_sessions=3)
    shadow = rep.load_jsonl(str(data / "pillar_weights_shadow.jsonl"))
    signals = rep.load_signals(str(data / "signals.jsonl"), include_archive=False)
    base = rep.build_report(shadow, signals)
    with_new = [dict(r, premium_alt={"live": 0.1, "w60": 0.1, "ewma": 0.1, "day": 0.1}, premium_sat_run=0,
                     gex_cont=1.0) for r in shadow]
    again = rep.build_report(with_new, signals)
    assert again["checklist"] == base["checklist"]
    assert again["replay"] == base["replay"]
