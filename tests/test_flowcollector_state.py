"""Tests for agent/flowcollector/state.py. Hermetic: tmp_path only, no
network, never touches the real repo data/ dir (env-overridden paths)."""

import json
import os
import time

import pytest

import agent.flowcollector.state as state


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    # Temp paths, never delenv: unset falls back to the LIVE state files.
    for key in ("CLIO_SELFFLOW_FILE", "CLIO_SELFFLOW_COMPARE_FILE"):
        monkeypatch.setenv(key, str(tmp_path / f"unset_{key.lower()}"))
    # Always reset the module-level mtime cache between tests -- otherwise a
    # prior test's cached parse (keyed by absolute path) could leak into a
    # later test that reuses the same tmp_path-derived filename by chance.
    with state._state_cache_lock:
        state._state_cache["path"] = None
        state._state_cache["key"] = None
        state._state_cache["data"] = None
    yield


def test_state_path_and_compare_path_env_override(tmp_path, monkeypatch):
    sf = str(tmp_path / "selfflow_state.json")
    cf = str(tmp_path / "selfflow_compare.jsonl")
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", sf)
    monkeypatch.setenv("CLIO_SELFFLOW_COMPARE_FILE", cf)
    assert state.state_path() == sf
    assert state.compare_path() == cf


def test_state_path_defaults_under_repo_data_intelligence():
    assert state.DEFAULT_STATE_FILE.endswith(os.path.join("data", "intelligence", "selfflow_state.json"))
    assert state.DEFAULT_COMPARE_FILE.endswith(os.path.join("data", "intelligence", "selfflow_compare.jsonl"))


# ── write_state_atomic / get_selfflow ──────────────────────────────────────


def test_get_selfflow_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(tmp_path / "nope.json"))
    assert state.get_selfflow("SPY") is None


def test_write_then_get_selfflow_fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(tmp_path / "selfflow_state.json"))
    now = time.time()
    payload = {"symbols": {"SPY": {"nofa": 0.3, "premium": 0.1, "as_of": now}}}
    state.write_state_atomic(payload)
    entry = state.get_selfflow("SPY", max_age_seconds=300)
    assert entry is not None
    assert entry["nofa"] == 0.3
    # case-insensitive lookup
    assert state.get_selfflow("spy", max_age_seconds=300) is not None


def test_get_selfflow_stale_entry_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(tmp_path / "selfflow_state.json"))
    old = time.time() - 10_000
    state.write_state_atomic({"symbols": {"SPY": {"nofa": 0.1, "as_of": old}}})
    assert state.get_selfflow("SPY", max_age_seconds=300) is None


def test_get_selfflow_unknown_symbol_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(tmp_path / "selfflow_state.json"))
    state.write_state_atomic({"symbols": {"SPY": {"nofa": 0.1, "as_of": time.time()}}})
    assert state.get_selfflow("QQQ") is None


def test_get_selfflow_malformed_file_never_raises(tmp_path, monkeypatch):
    path = tmp_path / "selfflow_state.json"
    path.write_text("{not valid json")
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(path))
    assert state.get_selfflow("SPY") is None


def test_get_selfflow_wrong_top_level_shape_never_raises(tmp_path, monkeypatch):
    path = tmp_path / "selfflow_state.json"
    path.write_text(json.dumps(["not", "a", "dict"]))
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(path))
    assert state.get_selfflow("SPY") is None


def test_write_state_atomic_uses_tmp_and_replace(tmp_path, monkeypatch):
    target = tmp_path / "selfflow_state.json"
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(target))
    state.write_state_atomic({"symbols": {}})
    assert target.exists()
    assert not (tmp_path / "selfflow_state.json.tmp").exists()  # tmp cleaned up by os.replace


def test_write_state_atomic_creates_parent_dirs(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "dir" / "selfflow_state.json"
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(target))
    state.write_state_atomic({"symbols": {}})
    assert target.exists()


def test_get_selfflow_reflects_updated_file_after_rewrite(tmp_path, monkeypatch):
    path = tmp_path / "selfflow_state.json"
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(path))
    state.write_state_atomic({"symbols": {"SPY": {"nofa": 0.1, "as_of": time.time()}}})
    assert state.get_selfflow("SPY")["nofa"] == 0.1
    state.write_state_atomic({"symbols": {"SPY": {"nofa": 0.9, "as_of": time.time()}}})
    # No sleep and no manual cache invalidation: an immediate rewrite must be picked up on any filesystem clock
    # (this used to be papered over, and the paper hid a stale-cache flake -- see the key tests below).
    assert state.get_selfflow("SPY")["nofa"] == 0.9


def _seed(path, nofa):
    # a whole-second as_of keeps the JSON length identical across seeds (a float repr's length varies)
    state.write_state_atomic({"symbols": {"SPY": {"nofa": nofa, "as_of": float(int(time.time()))}}}, path=str(path))


def test_an_unchanged_file_is_parsed_once_not_on_every_read(tmp_path, monkeypatch):
    """The cache exists so the 30 s DPOF cycle does not re-parse a file the collector rewrites every 15 s."""
    path = tmp_path / "selfflow_state.json"
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(path))
    _seed(path, 0.1)
    loads = []
    real_load = json.load
    monkeypatch.setattr(state.json, "load", lambda fh: (loads.append(1), real_load(fh))[1])
    for _ in range(5):
        assert state.get_selfflow("SPY")["nofa"] == 0.1
    assert len(loads) == 1


def test_cache_sees_an_atomic_rewrite_even_when_the_mtime_is_forced_equal(tmp_path, monkeypatch):
    """Two writes in one clock tick share an mtime. The inode differs (tmp + os.replace), so the reader must refresh."""
    path = tmp_path / "selfflow_state.json"
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(path))
    _seed(path, 0.1)
    first = os.stat(path)
    assert state.get_selfflow("SPY")["nofa"] == 0.1
    _seed(path, 0.2)                                        # same JSON length as the first
    os.utime(path, ns=(first.st_mtime_ns, first.st_mtime_ns))   # ... and the same mtime, as in a fast test / coarse clock
    now = os.stat(path)
    assert now.st_mtime_ns == first.st_mtime_ns and now.st_size == first.st_size and now.st_ino != first.st_ino
    assert state.get_selfflow("SPY")["nofa"] == 0.2


def test_cache_sees_an_in_place_rewrite_with_a_different_size(tmp_path, monkeypatch):
    """Same inode, same forced mtime, different length: the size component must invalidate."""
    path = tmp_path / "selfflow_state.json"
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(path))
    _seed(path, 0.1)
    st = os.stat(path)
    assert state.get_selfflow("SPY")["nofa"] == 0.1
    with open(path, "r+") as fh:                            # in place: inode unchanged
        data = json.load(fh)
        data["symbols"]["SPY"]["nofa"] = 0.123456789
        fh.seek(0)
        fh.truncate()
        json.dump(data, fh)
    os.utime(path, ns=(st.st_mtime_ns, st.st_mtime_ns))
    assert os.stat(path).st_ino == st.st_ino and os.stat(path).st_size != st.st_size
    assert state.get_selfflow("SPY")["nofa"] == 0.123456789


def test_cache_sees_an_in_place_rewrite_one_nanosecond_later(tmp_path, monkeypatch):
    """Same inode, same size, mtime differing by 1 ns: invisible to a float mtime, visible to st_mtime_ns."""
    path = tmp_path / "selfflow_state.json"
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(path))
    _seed(path, 0.1)
    st = os.stat(path)
    assert state.get_selfflow("SPY")["nofa"] == 0.1
    raw = path.read_text().replace('"nofa": 0.1', '"nofa": 0.7')   # same length
    with open(path, "r+") as fh:
        fh.write(raw)
    os.utime(path, ns=(st.st_mtime_ns + 1, st.st_mtime_ns + 1))
    now = os.stat(path)
    assert now.st_ino == st.st_ino and now.st_size == st.st_size and float(now.st_mtime) == float(st.st_mtime)
    assert state.get_selfflow("SPY")["nofa"] == 0.7


# ── comparison log ─────────────────────────────────────────────────────────


def test_build_comparison_row_none_when_no_self_entry():
    assert state.build_comparison_row("SPY", {}, None, 1000.0) is None
    assert state.build_comparison_row("SPY", None, None, 1000.0) is None


def test_build_comparison_row_without_uw_snapshot():
    row = state.build_comparison_row("SPY", {"nofa": 0.2, "gex": 100.0, "gex_regime": "positive"}, None, 1000.0)
    assert row["underlying"] == "SPY"
    assert row["self"]["nofa"] == 0.2
    assert row["uw"]["available"] is False
    assert row["uw"]["gex"] is None
    assert row["uw"]["score"] is None


def test_build_comparison_row_with_uw_snapshot():
    uw = {
        "gex": {"SPY": {"regime": "ACCELERATING", "net_gex": 123.0, "pin_strike": 580.0}},
        "top_signals": [{"ticker": "SPY", "score": 0.42}, {"ticker": "QQQ", "score": -0.1}],
    }
    row = state.build_comparison_row("SPY", {"nofa": 0.2}, uw, 1000.0)
    assert row["uw"]["available"] is True
    assert row["uw"]["gex"] == {"regime": "ACCELERATING", "net_gex": 123.0, "pin_strike": 580.0}
    assert row["uw"]["score"] == 0.42


def test_append_comparison_row_and_read_back(tmp_path, monkeypatch):
    path = tmp_path / "selfflow_compare.jsonl"
    monkeypatch.setenv("CLIO_SELFFLOW_COMPARE_FILE", str(path))
    state.append_comparison_row({"underlying": "SPY", "as_of": 1.0})
    state.append_comparison_row({"underlying": "QQQ", "as_of": 2.0})
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["underlying"] == "SPY"
    assert json.loads(lines[1])["underlying"] == "QQQ"


def test_comparison_log_trims_when_capped(tmp_path, monkeypatch):
    path = tmp_path / "selfflow_compare.jsonl"
    monkeypatch.setenv("CLIO_SELFFLOW_COMPARE_FILE", str(path))
    monkeypatch.setattr(state, "_COMPARE_MAX_BYTES", 1000)
    monkeypatch.setattr(state, "_COMPARE_CAP_LINES", 20)
    monkeypatch.setattr(state, "_COMPARE_TRIM_TO_LINES", 10)
    for i in range(60):
        state.append_comparison_row({"underlying": "SPY", "as_of": float(i), "pad": "x" * 20})
    lines = path.read_text().splitlines()
    # The size-gated trim (mirrors agent.dpof.ledger's decisions.jsonl
    # pattern) only re-checks line count once the file has ALSO crossed the
    # byte threshold, so it can fire more than once across 60 appends and
    # the file can sit anywhere between _COMPARE_TRIM_TO_LINES and
    # _COMPARE_CAP_LINES right after the last append -- the invariant this
    # test actually cares about is "far fewer than 60 lines survive" (a
    # trim genuinely happened) and "the newest row survived" (a trim never
    # drops the tail).
    assert len(lines) < 60
    assert len(lines) <= state._COMPARE_CAP_LINES
    last = json.loads(lines[-1])
    assert last["as_of"] == 59.0


def test_read_uw_bridge_snapshot_sync_missing_file_returns_none(tmp_path):
    assert state._read_uw_bridge_snapshot_sync(tradingbot_dir=str(tmp_path)) is None


def test_read_uw_bridge_snapshot_sync_reads_todays_file(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    today = datetime.now(ny).strftime("%Y%m%d")
    d = tmp_path / "moltbot_knowledge" / "intelligence"
    d.mkdir(parents=True)
    (d / f"uw_flow_{today}.json").write_text(json.dumps({"gex": {"SPY": {"regime": "ACCELERATING"}}}))
    data = state._read_uw_bridge_snapshot_sync(tradingbot_dir=str(tmp_path))
    assert data is not None
    assert data["gex"]["SPY"]["regime"] == "ACCELERATING"


def test_read_uw_bridge_snapshot_sync_malformed_json_returns_none(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    today = datetime.now(ny).strftime("%Y%m%d")
    d = tmp_path / "moltbot_knowledge" / "intelligence"
    d.mkdir(parents=True)
    (d / f"uw_flow_{today}.json").write_text("{not json")
    assert state._read_uw_bridge_snapshot_sync(tradingbot_dir=str(tmp_path)) is None


def test_get_selfflow_ignores_state_written_with_streams_down(tmp_path, monkeypatch):
    """The collector rewrites its state every tick even while idle / with its
    streams down; that is not a fresh reading."""
    monkeypatch.setenv("CLIO_SELFFLOW_FILE", str(tmp_path / "selfflow_state.json"))
    entry = {"nofa": 0.0, "as_of": time.time()}
    state.write_state_atomic({"collector_status": {"streams_active": False}, "symbols": {"SPY": entry}})
    assert state.get_selfflow("SPY", max_age_seconds=60) is None
    state.write_state_atomic({"collector_status": {"streams_active": True}, "symbols": {"SPY": entry}})
    assert state.get_selfflow("SPY", max_age_seconds=60) is not None
