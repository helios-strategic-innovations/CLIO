"""Tests for scripts/selfflow_gex_reconcile.py (read-only GEX reconciliation).

All inputs are synthetic files under tmp_path; the live
data/intelligence/selfflow_compare.jsonl is never touched.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "selfflow_gex_reconcile.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("selfflow_gex_reconcile", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


recon = _load_script()


def _row(symbol, self_gex, uw_net):
    return {
        "underlying": symbol,
        "as_of": 1.0,
        "self": {"gex": self_gex, "gex_regime": None},
        "uw": {"gex": {"regime": "X", "net_gex": uw_net, "pin_strike": 1.0}, "available": True},
    }


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n" if isinstance(r, dict) else r + "\n")


def test_known_60x_ratio_and_40pct_sign_agreement(tmp_path, capsys):
    """5 SPY rows: |self| = 60 x |UW| every row; self agrees in sign with UW on
    2 of 5 rows (40%)."""
    inp = tmp_path / "compare.jsonl"
    uw = 1.0e9
    rows = [
        _row("SPY", 60.0e9, uw),
        _row("SPY", 60.0e9, uw),
        _row("SPY", -60.0e9, uw),
        _row("SPY", -60.0e9, uw),
        _row("SPY", -60.0e9, uw),
    ]
    _write(inp, rows)
    pairs, skipped = recon.load_pairs(str(inp))
    assert skipped == 0
    s = recon.summarise(pairs["SPY"])
    assert s["n"] == 5
    assert s["median_ratio"] == pytest.approx(60.0)
    assert s["sign_agree"] == pytest.approx(0.4)
    assert s["proposed_scale"] == pytest.approx(60.0)

    assert recon.main(["--input", str(inp)]) == 0
    out = capsys.readouterr().out
    assert "SPY: n=5 median_ratio(self/UW)=60.00 sign_agree=40.0%" in out


def test_ratio_is_median_not_mean(tmp_path):
    """Ratios [60, 60, 60, 60, 600]: median 60, mean 168. The proposed scale
    must follow the median so one outlier tick cannot inflate it."""
    inp = tmp_path / "compare.jsonl"
    uw = 2.0e9
    selfs = [60.0 * uw, 60.0 * uw, 60.0 * uw, 60.0 * uw, 600.0 * uw]
    _write(inp, [_row("QQQ", s, uw) for s in selfs])
    pairs, _ = recon.load_pairs(str(inp))
    s = recon.summarise(pairs["QQQ"])
    assert s["median_ratio"] == pytest.approx(60.0)
    assert s["proposed_scale"] == pytest.approx(60.0)


def test_null_and_malformed_rows_are_skipped_and_counted(tmp_path):
    inp = tmp_path / "compare.jsonl"
    good = _row("SPY", 6.0e9, 1.0e8)
    null_self = _row("SPY", None, 1.0e8)
    zero_uw = _row("SPY", 6.0e9, 0.0)
    no_uw_block = {"underlying": "SPY", "self": {"gex": 1.0}, "uw": {"gex": None}}
    _write(inp, [good, null_self, zero_uw, no_uw_block, "{not json", ""])
    pairs, skipped = recon.load_pairs(str(inp))
    assert len(pairs["SPY"]) == 1
    assert skipped == 4  # null self, zero UW, missing UW gex, unparsable (blank line not counted)


def test_script_writes_nothing_and_leaves_input_unchanged(tmp_path, capsys):
    """Read-only contract: the input's size and mtime are unchanged and the
    input directory gains no new entries."""
    inp = tmp_path / "compare.jsonl"
    _write(inp, [_row("SPY", 6.0e9, 1.0e8), _row("SPY", -6.0e9, 1.0e8)])
    before_stat = os.stat(inp)
    before_listing = sorted(os.listdir(tmp_path))
    before_bytes = inp.read_bytes()

    assert recon.main(["--input", str(inp)]) == 0
    capsys.readouterr()

    after_stat = os.stat(inp)
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert inp.read_bytes() == before_bytes
    assert sorted(os.listdir(tmp_path)) == before_listing


def test_empty_input_reports_no_usable_pair(tmp_path, capsys):
    inp = tmp_path / "compare.jsonl"
    _write(inp, [])
    assert recon.main(["--input", str(inp)]) == 0
    assert "no symbol has a usable self/UW GEX pair" in capsys.readouterr().out
