import json
import os

from agent.kronos import state


def _write(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _forecast(written_at, status="ok"):
    return {
        "schema": 1, "written_at": written_at, "model": {}, "params": {},
        "symbols": {"SPY": {"status": status, "error": None, "asof_bar": "2026-09-28T10:35:00-04:00",
                            "anchor_close": 600.0, "feed": "iex", "latency_ms": 2900,
                            "horizons": {"15": {"minutes": 15, "p_up": 0.6}}}},
    }


def test_paths_come_from_env_isolated_per_test(tmp_path):
    # conftest points both files at a per-test temp dir; nothing there yet.
    assert "live_state" in state.forecast_path()
    assert state.read_forecast() is None
    assert state.read_scores() is None


def test_fresh_forecast_is_returned_and_stale_is_not(monkeypatch, tmp_path):
    path = tmp_path / "f.json"
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(path))
    _write(path, _forecast(1000.0))
    assert state.read_forecast(now=1000.0 + 599)["symbols"]["SPY"]["anchor_close"] == 600.0
    assert state.read_forecast(now=1000.0 + 601) is None
    assert state.read_forecast(now=1000.0 - 120) is None  # written in the future by > 60 s


def test_wrong_schema_bad_json_and_bad_types_are_rejected(monkeypatch, tmp_path):
    path = tmp_path / "f.json"
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(path))
    bad = _forecast(1000.0)
    bad["schema"] = 2
    _write(path, bad)
    assert state.read_forecast(now=1000.0) is None
    path.write_text("{not json")
    os.utime(path, (2000, 2000))
    assert state.read_forecast(now=1000.0) is None
    odd = _forecast(True)
    _write(path, odd)
    os.utime(path, (3000, 3000))
    assert state.read_forecast(now=1000.0) is None


def test_symbol_forecast_only_returns_ok_entries(monkeypatch, tmp_path):
    ok = _forecast(1000.0)
    assert state.symbol_forecast(ok, "SPY")["feed"] == "iex"
    assert state.symbol_forecast(ok, "QQQ") is None
    assert state.symbol_forecast(_forecast(1000.0, status="error"), "SPY") is None
    assert state.symbol_forecast(None, "SPY") is None


def test_write_json_atomic_round_trips_and_leaves_no_temp(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "scores.json"
    monkeypatch.setenv("CLIO_KRONOS_SCORES_FILE", str(path))
    state.write_json_atomic(str(path), {"schema": 1, "written_at": 1.0, "symbols": {"SPY": {}}})
    assert state.read_scores()["symbols"] == {"SPY": {}}
    assert [p.name for p in path.parent.iterdir()] == ["scores.json"]


def test_cache_sees_a_same_size_rewrite_within_one_mtime_tick(monkeypatch, tmp_path):
    """Two atomic writes of the same size can share an mtime tick (seen on the VPS 2026-09-27);
    the reader's cache key includes the inode, which every atomic write changes."""
    path = tmp_path / "f.json"
    monkeypatch.setenv("CLIO_KRONOS_FORECAST_FILE", str(path))
    first_obj = _forecast(1000.0)
    state.write_json_atomic(str(path), first_obj)
    first = os.stat(path)
    assert state.read_forecast(now=1001.0)["symbols"]["SPY"]["anchor_close"] == 600.0
    second_obj = _forecast(1000.0)
    second_obj["symbols"]["SPY"]["anchor_close"] = 601.0
    state.write_json_atomic(str(path), second_obj)
    os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns))
    second = os.stat(path)
    assert (second.st_mtime_ns, second.st_size) == (first.st_mtime_ns, first.st_size)  # old key collides
    assert state.read_forecast(now=1001.0)["symbols"]["SPY"]["anchor_close"] == 601.0
