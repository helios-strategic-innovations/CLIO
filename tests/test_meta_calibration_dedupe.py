"""Tests for scripts/meta_calibration_dedupe.py (S215 / R-28).

Every test works on a copy under tmp_path. The live data/ tree is never touched.
"""
import copy
import importlib.util
import json
import os
import shutil

import pytest

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "meta_calibration_dedupe.py",
)


@pytest.fixture(scope="module")
def dd():
    spec = importlib.util.spec_from_file_location("meta_calibration_dedupe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _legacy(i):
    return {
        "prediction_id": f"seal_ti_quality_{i:08x}",
        "timestamp": f"2026-03-{1 + i % 27:02d}T{i % 24:02d}:00:00+00:00",
        "domain": "trade_ideas",
        "stated_confidence": 0.54,
        "description": "SEAL trade idea win rate: 0% (0/27)",
        "actual_outcome": False,
        "calibration_error": 0.54,
    }


def _other(pid):
    return {
        "prediction_id": pid,
        "timestamp": "2026-03-01T00:00:00+00:00",
        "domain": "trade_ideas",
        "stated_confidence": 0.65,
        "description": f"{pid} idea",
        "actual_outcome": True,
        "calibration_error": 0.35,
    }


def _build_data():
    cal = [_legacy(i) for i in range(249)]
    cal.append(_other("ti_MID"))
    cal.extend(_legacy(i) for i in range(249, 497))
    cal.append(_other("ti_END"))
    cal.append(_other("trade_77"))
    return {"gaps": [{"topic": "x"}], "calibration": cal, "last_updated": "2026-03-27"}


def _make_copy(tmp_path, data=None):
    root = tmp_path / "copy"
    root.mkdir()
    path = root / "meta_memory.json"
    payload = data if data is not None else _build_data()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    bk = tmp_path / "bk"
    bk.mkdir()
    return str(path), str(bk)


def test_dry_run_leaves_original_byte_identical(dd, tmp_path, capsys):
    path, _ = _make_copy(tmp_path)
    before_bytes = open(path, "rb").read()
    before_listing = sorted(os.listdir(os.path.dirname(path)))
    rc = dd.main(["--file", path])
    assert rc == 0
    assert open(path, "rb").read() == before_bytes
    assert sorted(os.listdir(os.path.dirname(path))) == before_listing


def test_dry_run_reports_497_legacy_rows_collapsed_to_one(dd, tmp_path, capsys):
    path, _ = _make_copy(tmp_path)
    assert dd.main(["--file", path]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry-run"
    assert report["rows_before"] == 500
    assert report["rows_after"] == 4
    assert report["legacy_rows_before"] == 497
    assert report["by_prefix_before"]["seal_ti_quality_*"] == 497
    assert report["by_prefix_after"].get("seal_ti_quality_*", 0) == 0


def test_plan_does_not_mutate_input(dd):
    data = _build_data()
    snapshot = copy.deepcopy(data)
    new, report = dd.plan(data)
    assert data == snapshot
    assert len(new) == 4


def test_apply_refused_without_guard_flags(dd, tmp_path):
    path, bk = _make_copy(tmp_path)
    before = open(path, "rb").read()
    assert dd.main(["--file", path, "--apply"]) == 2
    assert dd.main(["--file", path, "--apply", "--backup-dir", bk]) == 2
    assert dd.main(["--file", path, "--apply", "--i-understand-protected-state"]) == 2
    assert dd.main(["--file", path, "--apply", "--i-understand-protected-state",
                    "--backup-dir", str(tmp_path / "missing")]) == 2
    assert open(path, "rb").read() == before
    assert os.listdir(bk) == []


def test_apply_collapses_497_to_one_and_leaves_others_untouched(dd, tmp_path):
    path, bk = _make_copy(tmp_path)
    original = json.loads(open(path, "rb").read())
    last_legacy = original["calibration"][-3]  # index 497 is the newest legacy row
    assert last_legacy["prediction_id"].startswith("seal_ti_quality_")

    rc = dd.main(["--file", path, "--apply", "--i-understand-protected-state",
                  "--backup-dir", bk])
    assert rc == 0

    after = json.loads(open(path, "rb").read())
    cal = after["calibration"]
    assert [r["prediction_id"] for r in cal] == [
        "ti_MID", "seal_ti_quality", "ti_END", "trade_77",
    ]
    collapsed = cal[1]
    assert collapsed["timestamp"] == last_legacy["timestamp"]
    assert collapsed["description"] == last_legacy["description"]
    assert collapsed["actual_outcome"] == last_legacy["actual_outcome"]
    for kept, orig in zip([cal[0], cal[2], cal[3]],
                          [original["calibration"][249], original["calibration"][-2],
                           original["calibration"][-1]]):
        assert kept == orig
    assert after["gaps"] == original["gaps"]
    assert after["last_updated"] == original["last_updated"]

    backups = os.listdir(bk)
    assert len(backups) == 1
    assert open(os.path.join(bk, backups[0]), "rb").read() == \
        json.dumps(original, indent=2).encode("utf-8")


def test_backup_is_hard_link_to_original_inode(dd, tmp_path):
    path, bk = _make_copy(tmp_path)
    ino = os.stat(path).st_ino
    assert dd.main(["--file", path, "--apply", "--i-understand-protected-state",
                    "--backup-dir", bk]) == 0
    (name,) = os.listdir(bk)
    assert os.stat(os.path.join(bk, name)).st_ino == ino


def test_apply_restores_original_when_verification_fails(dd, tmp_path):
    path, bk = _make_copy(tmp_path)
    before = open(path, "rb").read()

    def bad_verify(p, expected):
        raise dd.DedupeError("simulated post-swap failure")

    with pytest.raises(dd.DedupeError, match="original restored"):
        dd.apply(path, bk, _verify=bad_verify)
    assert open(path, "rb").read() == before


def test_apply_is_noop_without_legacy_rows(dd, tmp_path):
    data = {"calibration": [_other("ti_A"), _other("trade_9")]}
    path, bk = _make_copy(tmp_path, data)
    before = open(path, "rb").read()
    assert dd.main(["--file", path, "--apply", "--i-understand-protected-state",
                    "--backup-dir", bk]) == 0
    assert open(path, "rb").read() == before
    assert os.listdir(bk) == []
