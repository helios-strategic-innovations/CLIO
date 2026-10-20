"""Tests for agent.dpof.pillar_weights_io -- StateStore and ShadowLog (tmp_path only)."""

from __future__ import annotations

import asyncio
import fcntl
import glob
import json
import os
import threading
import time

import pytest

from agent.dpof.pillar_weights_io import ShadowLog, StateStore, clean_for_json


def _tmp_files(d):
    """All leftover temp files, INCLUDING dot-prefixed ones (glob('*.tmp') silently skips dotfiles)."""
    return sorted(f for f in os.listdir(str(d)) if f.endswith(".tmp"))


def _hold_flock(lock_path):
    """Open and LOCK_EX a sidecar lock from the test; returns the open file."""
    lf = open(lock_path, "a")
    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
    return lf


# ---------------------------------------------------------------- StateStore

def test_state_round_trip(tmp_path):
    """save then load returns the same payload (plus stamped schema fields)."""
    st = StateStore(str(tmp_path))
    assert st.save({"liveness": {"SPY|premium": {"k": 1.5}}}) is True
    got = st.load()
    assert got["liveness"] == {"SPY|premium": {"k": 1.5}}


def test_state_load_missing_returns_none(tmp_path):
    """A missing state file loads as None without creating it."""
    st = StateStore(str(tmp_path))
    assert st.load() is None
    assert not os.path.exists(st.path)


def test_state_corrupt_json_quarantined_then_save_succeeds(tmp_path):
    """Corrupt JSON is renamed to *.corrupt-<epoch>, load returns None, and a fresh save works."""
    st = StateStore(str(tmp_path))
    with open(st.path, "w") as f:
        f.write("{not json")
    assert st.load() is None
    assert not os.path.exists(st.path)
    assert glob.glob(st.path + ".corrupt-*")
    assert st.save({"a": 1}) is True
    assert st.load()["a"] == 1


def test_state_wrong_schema_version_returns_none_not_renamed(tmp_path):
    """A dict with another schema_version is ignored and the file is left in place."""
    st = StateStore(str(tmp_path))
    with open(st.path, "w") as f:
        json.dump({"schema_version": 2, "a": 1}, f)
    before = open(st.path, "rb").read()
    assert st.load() is None
    assert open(st.path, "rb").read() == before
    assert not glob.glob(st.path + ".corrupt-*")


def test_state_non_dict_file_quarantined(tmp_path):
    """A top-level JSON list is treated as corrupt and quarantined."""
    st = StateStore(str(tmp_path))
    with open(st.path, "w") as f:
        f.write("[1, 2]")
    assert st.load() is None
    assert glob.glob(st.path + ".corrupt-*")


def test_state_nan_inf_and_unserializable_refused_file_untouched(tmp_path):
    """NaN, inf and non-serializable values make save return False and leave the old bytes intact."""
    st = StateStore(str(tmp_path))
    assert st.save({"v": 1}) is True
    before = open(st.path, "rb").read()
    for bad in ({"v": float("nan")}, {"v": float("inf")}, {"v": object()}, {"v": {1, 2}}):
        assert st.save(bad) is False
        assert open(st.path, "rb").read() == before
    assert _tmp_files(tmp_path) == []


def test_state_save_refuses_non_dict(tmp_path):
    """save of a non-dict returns False and writes nothing."""
    st = StateStore(str(tmp_path))
    assert st.save([1, 2, 3]) is False
    assert not os.path.exists(st.path)


def test_state_save_stamps_schema_and_written_at(tmp_path):
    """save stamps schema_version=1 and a numeric written_at into the persisted copy."""
    st = StateStore(str(tmp_path))
    caller = {"schema_version": 9, "written_at": 0}
    before = time.time()
    assert st.save(caller) is True
    got = st.load()
    assert got["schema_version"] == 1
    assert isinstance(got["written_at"], float) and got["written_at"] >= before
    assert caller == {"schema_version": 9, "written_at": 0}  # caller dict not mutated


def test_state_no_tmp_residue_after_saves(tmp_path):
    """Normal saves leave no *.tmp files behind."""
    st = StateStore(str(tmp_path))
    for i in range(20):
        assert st.save({"i": i})
    assert _tmp_files(tmp_path) == []


def test_state_lock_inode_persists_data_inode_changes(tmp_path):
    """Sidecar lock keeps its inode across saves while the data file is replaced (new inode)."""
    st = StateStore(str(tmp_path))
    st.save({"i": 0})
    lock_ino = os.stat(st._lock_path).st_ino
    data_ino = os.stat(st.path).st_ino
    st.save({"i": 1})
    assert os.stat(st._lock_path).st_ino == lock_ino
    assert os.stat(st.path).st_ino != data_ino


def test_state_save_blocks_while_sidecar_lock_held(tmp_path):
    """save waits for the sidecar flock: it cannot finish while the test holds the lock."""
    st = StateStore(str(tmp_path))
    st.save({"i": 0})
    holder = _hold_flock(st._lock_path)
    done = threading.Event()

    def _run():
        st.save({"i": 1})
        done.set()

    t = threading.Thread(target=_run)
    try:
        t.start()
        assert not done.wait(0.4), "save finished while the sidecar lock was held"
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()
    t.join(5)
    assert done.is_set()
    assert st.load()["i"] == 1


def test_state_async_runs_off_loop_thread(tmp_path, monkeypatch):
    """save_async and load_async execute the blocking core on a worker thread, not the loop thread."""
    st = StateStore(str(tmp_path))
    seen = {}
    real_save, real_load = st.save, st.load

    def spy_save(state):
        seen["save"] = threading.get_ident()
        return real_save(state)

    def spy_load():
        seen["load"] = threading.get_ident()
        return real_load()

    monkeypatch.setattr(st, "save", spy_save)
    monkeypatch.setattr(st, "load", spy_load)

    async def _main():
        loop_tid = threading.get_ident()
        ok = await st.save_async({"x": 1})
        got = await st.load_async()
        return loop_tid, ok, got

    loop_tid, ok, got = asyncio.run(_main())
    assert ok is True and got["x"] == 1
    assert seen["save"] != loop_tid
    assert seen["load"] != loop_tid


def test_state_concurrent_saves_never_corrupt(tmp_path):
    """10 threads x 50 saves x 3 runs: the file always parses, no tmp residue."""
    for run in range(3):
        d = tmp_path / f"run{run}"
        st = StateStore(str(d))
        errors = []

        def _worker(tid):
            try:
                for op in range(50):
                    assert st.save({"tid": tid, "op": op, "pad": "x" * 200})
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert errors == []
        with open(st.path) as f:
            parsed = json.load(f)
        assert parsed["schema_version"] == 1
        assert _tmp_files(d) == []


# ---------------------------------------------------------------- ShadowLog

def test_shadow_round_trip_read_tail(tmp_path):
    """Appended records come back in order from read_tail."""
    sl = ShadowLog(str(tmp_path))
    assert sl.append_cycle([{"seq": 1}, {"seq": 2}]) == 2
    assert sl.append_cycle([{"seq": 3}]) == 1
    assert [r["seq"] for r in sl.read_tail(10)] == [1, 2, 3]
    assert [r["seq"] for r in sl.read_tail(2)] == [2, 3]


def test_shadow_nan_inf_tuple_cleaned(tmp_path):
    """NaN/inf become null, tuples become lists, and the record is written without raising."""
    sl = ShadowLog(str(tmp_path))
    assert sl.append_cycle([{"a": float("nan"), "b": float("inf"), "c": (1, 2), "d": True}]) == 1
    rec = sl.read_tail(1)[0]
    assert rec == {"a": None, "b": None, "c": [1, 2], "d": True}


def test_shadow_unserializable_record_skipped_neighbours_written(tmp_path):
    """A record that cannot be serialized is skipped; its neighbours are still written."""
    sl = ShadowLog(str(tmp_path))
    n = sl.append_cycle([{"seq": 1}, {"seq": 2, "bad": object()}, {"seq": 3}])
    assert n == 2
    assert [r["seq"] for r in sl.read_tail(10)] == [1, 3]


def test_shadow_non_dict_record_skipped(tmp_path):
    """A non-dict record is skipped and does not abort the batch."""
    sl = ShadowLog(str(tmp_path))
    assert sl.append_cycle([{"seq": 1}, "junk", {"seq": 2}]) == 2
    assert [r["seq"] for r in sl.read_tail(10)] == [1, 2]


def test_shadow_trim_keeps_newest_within_bounds(tmp_path):
    """With max_lines=100, 250 single appends leave 90..100 lines and the newest record."""
    sl = ShadowLog(str(tmp_path), max_lines=100)
    for i in range(250):
        sl.append_cycle([{"seq": i}])
    with open(sl.path) as f:
        n = sum(1 for _ in f)
    assert 90 <= n <= 100
    assert sl.read_tail(1)[0]["seq"] == 249
    seqs = [r["seq"] for r in sl.read_tail(n)]
    assert seqs == list(range(249 - n + 1, 250))


def test_shadow_trim_keeps_newest_for_oversized_batch(tmp_path):
    """One batch larger than max_lines trims to the newest records of that batch."""
    sl = ShadowLog(str(tmp_path), max_lines=50)
    assert sl.append_cycle([{"seq": i} for i in range(120)]) == 120
    seqs = [r["seq"] for r in sl.read_tail(1000)]
    assert seqs[-1] == 119
    assert len(seqs) <= 50
    assert seqs == sorted(seqs)


def test_shadow_torn_tail_repaired_no_merge(tmp_path):
    """A partial line without newline is terminated before the next append; nothing merges."""
    sl = ShadowLog(str(tmp_path))
    sl.append_cycle([{"seq": 1}])
    with open(sl.path, "a") as f:
        f.write('{"seq": 2, "torn')
    sl.append_cycle([{"seq": 3}])
    recs = sl.read_tail(10)
    assert [r["seq"] for r in recs] == [1, 3]
    with open(sl.path) as f:
        lines = f.read().split("\n")
    assert lines[-1] == ""
    assert lines[1] == '{"seq": 2, "torn'


def test_shadow_lock_blocks_appends_while_held(tmp_path):
    """append_cycle waits for the shadow sidecar flock."""
    sl = ShadowLog(str(tmp_path))
    sl.append_cycle([{"seq": 0}])
    holder = _hold_flock(sl._lock_path)
    done = threading.Event()

    def _run():
        sl.append_cycle([{"seq": 1}])
        done.set()

    t = threading.Thread(target=_run)
    try:
        t.start()
        assert not done.wait(0.4), "append finished while the sidecar lock was held"
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()
    t.join(5)
    assert [r["seq"] for r in sl.read_tail(10)] == [0, 1]


def test_shadow_concurrent_appenders_with_trim(tmp_path):
    """10 threads x 50 ops x 3 records x 3 runs with trimming: all lines parse, unique seq, bounded."""
    for run in range(3):
        d = tmp_path / f"trim{run}"
        sl = ShadowLog(str(d), max_lines=400)
        errors = []

        def _worker(tid):
            try:
                for op in range(50):
                    base = (tid * 50 + op) * 3
                    sl.append_cycle([{"seq": base + k} for k in range(3)])
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert errors == []
        with open(sl.path) as f:
            raw = f.read().splitlines()
        assert len(raw) <= 400
        parsed = [json.loads(line) for line in raw]
        seqs = [r["seq"] for r in parsed]
        assert len(seqs) == len(set(seqs))


def test_shadow_concurrent_appenders_no_trim_exact_count(tmp_path):
    """10 threads x 50 ops x 3 records x 3 runs without trimming: exactly 1500 lines per run, all parse."""
    for run in range(3):
        d = tmp_path / f"notrim{run}"
        sl = ShadowLog(str(d), max_lines=100000)
        errors = []

        def _worker(tid):
            try:
                for op in range(50):
                    base = (tid * 50 + op) * 3
                    sl.append_cycle([{"seq": base + k} for k in range(3)])
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert errors == []
        with open(sl.path) as f:
            raw = f.read().splitlines()
        assert len(raw) == 1500
        seqs = [json.loads(line)["seq"] for line in raw]
        assert len(seqs) == len(set(seqs))


def test_shadow_async_runs_off_loop_thread(tmp_path, monkeypatch):
    """append_cycle_async runs the blocking core on a worker thread and returns the count."""
    sl = ShadowLog(str(tmp_path))
    seen = {}
    real = sl.append_cycle

    def spy(records):
        seen["tid"] = threading.get_ident()
        return real(records)

    monkeypatch.setattr(sl, "append_cycle", spy)

    async def _main():
        loop_tid = threading.get_ident()
        n = await sl.append_cycle_async([{"seq": 7}])
        return loop_tid, n

    loop_tid, n = asyncio.run(_main())
    assert n == 1
    assert seen["tid"] != loop_tid
    assert sl.read_tail(1)[0]["seq"] == 7


def test_shadow_empty_batch_creates_no_file(tmp_path):
    """append_cycle([]) returns 0 and creates neither the data file nor the lock file."""
    d = tmp_path / "fresh"
    sl = ShadowLog(str(d))
    assert sl.append_cycle([]) == 0
    assert not os.path.exists(sl.path)
    assert not os.path.exists(sl._lock_path)


# ------------------------------------------------------------- clean_for_json

def test_clean_for_json_cases():
    """Each cleaning rule maps its input as documented."""
    assert clean_for_json(float("nan")) is None
    assert clean_for_json(float("-inf")) is None
    assert clean_for_json(1.5) == 1.5
    assert clean_for_json(True) is True
    assert clean_for_json(3) == 3
    assert clean_for_json("s") == "s"
    assert clean_for_json((1, (2.0, float("nan")))) == [1, [2.0, None]]
    assert clean_for_json({1: "a", "b": {2: float("inf")}}) == {"1": "a", "b": {"2": None}}
    assert json.dumps(clean_for_json({"k": [float("nan")]}), allow_nan=False)


# ------------------------------------------- reviewer additions (mutation gaps)

@pytest.mark.parametrize("version", [0, 2, "1", True, None, 1.5])
def test_state_any_schema_version_other_than_integer_one_is_ignored_not_renamed(tmp_path, version):
    """Only the integer 1 is accepted: 0, 2, '1', True, None and 1.5 load as None and the file is left in place."""
    st = StateStore(str(tmp_path))
    st.save({"x": 1})
    doc = json.load(open(st.path))
    doc["schema_version"] = version
    open(st.path, "w").write(json.dumps(doc))
    assert st.load() is None
    assert os.path.exists(st.path)
    assert not glob.glob(st.path + ".corrupt-*")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_state_non_finite_json_literal_is_treated_as_corrupt(tmp_path, literal):
    """A state file containing a NaN/Infinity literal is quarantined as corrupt, never loaded."""
    st = StateStore(str(tmp_path))
    st.save({"x": 1})
    open(st.path, "w").write('{"schema_version":1,"x":%s}' % literal)
    assert st.load() is None
    assert glob.glob(st.path + ".corrupt-*")


def test_state_failed_replace_leaves_no_tmp_and_keeps_old_file(tmp_path, monkeypatch):
    """If os.replace fails the save returns False, no tmp file remains and the previous document is intact."""
    st = StateStore(str(tmp_path))
    assert st.save({"x": 1}) is True
    before = open(st.path, "rb").read()

    def boom(*a, **k):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)
    assert st.save({"x": 2}) is False
    monkeypatch.undo()
    assert _tmp_files(tmp_path) == []
    assert open(st.path, "rb").read() == before


def test_shadow_counts_lines_already_on_disk_so_trim_still_triggers(tmp_path):
    """A new ShadowLog instance counts the existing file, so appending past the cap trims instead of growing forever."""
    path = os.path.join(str(tmp_path), "pillar_weights_shadow.jsonl")
    with open(path, "w") as f:
        for i in range(95):
            f.write(json.dumps({"seq": i}) + "\n")
    log = ShadowLog(str(tmp_path), max_lines=100)
    log.append_cycle([{"seq": 1000 + i} for i in range(10)])
    n = sum(1 for _ in open(path))
    assert n == 90
    assert log.read_tail(1)[0]["seq"] == 1009


def test_shadow_trim_boundary_is_strictly_above_cap_and_leaves_headroom(tmp_path):
    """Exactly max_lines lines are kept; one more triggers a trim down to int(0.9 * max_lines)."""
    log = ShadowLog(str(tmp_path), max_lines=100)
    log.append_cycle([{"seq": i} for i in range(100)])
    assert sum(1 for _ in open(log.path)) == 100
    log.append_cycle([{"seq": 100}])
    assert sum(1 for _ in open(log.path)) == 90
    assert log.read_tail(1)[0]["seq"] == 100
    assert log.read_tail(90)[0]["seq"] == 11


# ------------------------------------------- reviewer-panel hardening (2026-10-10)

class _Clock:
    """Stand-in for the module's `time`: lets a test advance monotonic time (backoff / dedupe windows) deterministically."""
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def time(self):
        return time.time()


@pytest.mark.skipif(os.geteuid() == 0, reason="chmod 000 does not block root")
def test_state_unreadable_file_is_left_in_place_not_quarantined(tmp_path):
    """A transient read error (EACCES) must NOT rename a valid state file away: load returns None, the file stays."""
    st = StateStore(str(tmp_path))
    st.save({"x": 1})
    os.chmod(st.path, 0)
    try:
        assert st.load() is None
        assert os.path.exists(st.path) and not glob.glob(st.path + ".corrupt-*")
    finally:
        os.chmod(st.path, 0o644)
    assert st.load()["x"] == 1


def test_repeated_state_write_failures_warn_once_and_keep_the_old_file(tmp_path, monkeypatch, caplog):
    """Six failing saves emit ONE 'write failed' WARNING (deduped), leave the old document intact and no tmp file."""
    st = StateStore(str(tmp_path))
    assert st.save({"x": 1}) is True
    before = open(st.path, "rb").read()

    def boom(*a, **k):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)
    with caplog.at_level("WARNING", logger="clio.dpof.pillar_weights_io"):
        assert [st.save({"x": i}) for i in range(2, 8)] == [False] * 6
    monkeypatch.undo()
    assert len([r for r in caplog.records if "write failed" in r.getMessage()]) == 1
    assert open(st.path, "rb").read() == before and _tmp_files(tmp_path) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="chmod 555 does not block root")
def test_shadow_log_on_a_read_only_dir_warns_once_backs_off_then_recovers(tmp_path, monkeypatch, caplog):
    """A read-only data dir: five appends warn ONCE and write nothing; after the 30 s backoff a fixed dir writes again."""
    import agent.dpof.pillar_weights_io as iomod
    clock = _Clock()
    monkeypatch.setattr(iomod, "time", clock)
    d = tmp_path / "ro"
    d.mkdir()
    log = ShadowLog(str(d))
    os.chmod(d, 0o555)
    try:
        with caplog.at_level("WARNING", logger="clio.dpof.pillar_weights_io"):
            assert [log.append_cycle([{"seq": i}]) for i in range(5)] == [0] * 5
        assert len([r for r in caplog.records if "append failed" in r.getMessage()]) == 1
    finally:
        os.chmod(d, 0o755)
    assert log.append_cycle([{"seq": 99}]) == 0          # still inside the backoff window
    clock.now += 31.0
    assert log.append_cycle([{"seq": 100}]) == 1
    assert log.read_tail(1)[0]["seq"] == 100


def test_failed_trim_is_not_retried_every_append_but_resumes_after_the_backoff(tmp_path, monkeypatch):
    """When the trim rewrite fails, later appends do not re-attempt it until the 300 s backoff has passed."""
    import agent.dpof.pillar_weights_io as iomod
    clock = _Clock()
    monkeypatch.setattr(iomod, "time", clock)
    log = ShadowLog(str(tmp_path), max_lines=10)
    calls = []
    real_trim = log._trim_locked

    def counting_trim():
        calls.append(clock.now)
        real_trim()

    monkeypatch.setattr(log, "_trim_locked", counting_trim)
    real_replace = os.replace

    def failing_replace(src, dst, *a, **k):
        if ".trim.tmp" in str(src):
            raise OSError("simulated trim failure")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", failing_replace)
    log.append_cycle([{"seq": i} for i in range(20)])
    assert len(calls) == 1
    log.append_cycle([{"seq": 100}])
    log.append_cycle([{"seq": 101}])
    assert len(calls) == 1                                # backed off
    clock.now += 301.0
    log.append_cycle([{"seq": 102}])
    assert len(calls) == 2                                # resumed


@pytest.mark.skipif(os.geteuid() == 0, reason="chmod 444 does not block root")
def test_a_failing_append_to_an_existing_log_backs_off_instead_of_retrying_every_cycle(tmp_path, monkeypatch, caplog):
    """The log file itself is unwritable (directory fine): the first append fails and warns ONCE; the next appends are
    skipped inside the backoff, and succeed again once the file is writable and the backoff has passed."""
    import agent.dpof.pillar_weights_io as iomod
    clock = _Clock()
    monkeypatch.setattr(iomod, "time", clock)
    log = ShadowLog(str(tmp_path))
    assert log.append_cycle([{"seq": 1}]) == 1
    os.chmod(log.path, 0o444)
    try:
        with caplog.at_level("WARNING", logger="clio.dpof.pillar_weights_io"):
            assert [log.append_cycle([{"seq": i}]) for i in range(2, 7)] == [0] * 5
        assert len([r for r in caplog.records if "write failed" in r.getMessage() or "append failed" in r.getMessage()]) == 1
    finally:
        os.chmod(log.path, 0o644)
    assert log.append_cycle([{"seq": 50}]) == 0            # still inside the backoff window
    clock.now += 31.0
    assert log.append_cycle([{"seq": 51}]) == 1
