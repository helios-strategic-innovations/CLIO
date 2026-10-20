#!/usr/bin/env python3
"""S209 — working-memory deadlock + silent-death-hardening contract tests.

Run (no pytest needed):  /home/agent/openclaw/venv/bin/python3 tests/test_working_memory_deadlock.py

Sibling of test_knowledge_save_deadlock.py (same bounded-join discipline). ALL
file I/O is redirected into a throwaway temp dir — production memory paths are
NEVER touched (CLAUDE.md runtime-state rule).

The bug (S164, b1a9784d 2026-04-04): WorkingMemory.add()/promote()/clear_domain()
called self._save() while holding self._lock; _save() re-acquired the same
non-reentrant lock → deadlock. Dead ~12 weeks (working_memory.json == {}).

These tests are CONTRACT tests — each runs the real path in a daemon worker with a
bounded join, so a re-introduction of the bug FAILS within JOIN_TIMEOUT instead of
hanging the runner.
"""

import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

JOIN_TIMEOUT = 15.0

import agent.memory.working_memory as wmmod  # noqa: E402
from agent.memory.working_memory import WorkingMemory, WorkingMemoryEntry  # noqa: E402
from agent.memory import liveness  # noqa: E402
from agent.memory.memory_engine import MemoryEngine  # noqa: E402


def fresh_wm(base: str) -> WorkingMemory:
    """A WorkingMemory pointed entirely at a temp dir."""
    from pathlib import Path
    wmmod.WORKING_MEMORY_PATH = Path(base) / "working_memory.json"
    return WorkingMemory()


def bounded(fn, timeout=JOIN_TIMEOUT):
    """Run fn() in a daemon worker; return ('ok', ret) / ('err', exc) /
    ('deadlock', None) if still alive after timeout."""
    box = {}

    def worker():
        try:
            box["ret"] = fn()
        except Exception as e:
            box["exc"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return ("deadlock", None)
    if "exc" in box:
        return ("err", box["exc"])
    return ("ok", box.get("ret"))


def _seed(wm: WorkingMemory, eid: str, domain="trading") -> None:
    wm._entries[eid] = WorkingMemoryEntry(
        id=eid, content="x" * 40, domain=domain, priority=0.6,
        created_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=72)).isoformat(),
    )


# ──────────────────────────────────────────────────────────────────────────
# 1. The deadlock repro — add / promote / clear_domain must all return.
# ──────────────────────────────────────────────────────────────────────────
def test_add_promote_clear_no_deadlock() -> str:
    with tempfile.TemporaryDirectory(prefix="clio_wm_dl_") as base:
        wm = fresh_wm(base)
        try:
            status, ret = bounded(lambda: wm.add(
                content="A genuine working-memory entry for the deadlock contract test",
                domain="trading", priority=0.6))
            assert status == "ok" and ret, f"add() {status} (S164 deadlock?) ret={ret!r}"
            eid = ret
            status, _ = bounded(lambda: wm.promote(eid))
            assert status == "ok", f"promote() {status} (S164 deadlock?)"
            status, n = bounded(lambda: wm.clear_domain("trading"))
            assert status == "ok" and n == 1, f"clear_domain() {status} removed={n}"
        finally:
            wm.stop()
    return "add/promote/clear_domain all return (no re-entrant deadlock)"


# ──────────────────────────────────────────────────────────────────────────
# 2. Re-entrancy guard — _save() under the lock RAISES (loud), never hangs.
# ──────────────────────────────────────────────────────────────────────────
def test_reentrancy_guard_raises() -> str:
    with tempfile.TemporaryDirectory(prefix="clio_wm_guard_") as base:
        wm = fresh_wm(base)
        try:
            def reentrant():
                with wm._lock:          # simulate the S164 bug
                    wm._save()
            status, exc = bounded(reentrant, timeout=5.0)
            assert status == "err", f"expected RAISE, got {status} (a hang means the guard is dead)"
            assert isinstance(exc, RuntimeError), f"expected RuntimeError, got {type(exc).__name__}"
            assert "re-entrant" in str(exc).lower()
        finally:
            wm.stop()
    return "guard raises RuntimeError on _save()-under-lock (no hang)"


# ──────────────────────────────────────────────────────────────────────────
# 3. CONTRACT — the critical section performs NO IO and NO re-entrant acquire.
#    Instrument _write_snapshot to assert the entries lock is NOT held by the
#    calling thread at the moment any disk write happens.
# ──────────────────────────────────────────────────────────────────────────
def test_contract_no_io_under_lock() -> str:
    with tempfile.TemporaryDirectory(prefix="clio_wm_contract_") as base:
        wm = fresh_wm(base)
        violations = []
        real_write = wm._write_snapshot

        def spy(snapshot):
            if wm._lock.owned_by_current_thread():
                violations.append("write happened while critical-section lock held")
            return real_write(snapshot)

        wm._write_snapshot = spy
        try:
            for i in range(5):
                wm.add(content=f"contract probe entry number {i} with enough length", domain="trading")
            wm.flush()                 # synchronous path
            time.sleep(0.3)            # let the flusher thread run too
            wm.flush()
            assert not violations, f"CONTRACT VIOLATION: {violations}"
            # And at least one write actually occurred (the spy was exercised).
            assert wm.persist_health()["last_persist_ok_ts"] is not None, "no write ever succeeded"
        finally:
            wm.stop()
    return "critical section did NO disk IO under the lock (write-behind contract holds)"


# ──────────────────────────────────────────────────────────────────────────
# 4. Persistence round-trip — add → flush → on-disk file carries the entry.
# ──────────────────────────────────────────────────────────────────────────
def test_persist_roundtrip() -> str:
    with tempfile.TemporaryDirectory(prefix="clio_wm_persist_") as base:
        wm = fresh_wm(base)
        try:
            eid = wm.add(content="Persisted entry exercising the write-behind flusher path", domain="research")
            wm.flush()
            data = json.load(open(wmmod.WORKING_MEMORY_PATH))
            assert eid in data, "entry not persisted to disk"
            # Reload in a fresh instance → survives "restart".
            wm2 = fresh_wm(base)
            try:
                assert eid in wm2._entries, "entry not reloaded after restart"
            finally:
                wm2.stop()
        finally:
            wm.stop()
    return "entry persists to disk and survives reload"


# ──────────────────────────────────────────────────────────────────────────
# 5/6/7. Liveness probe — LIVE healthy; TIMEOUT on planted hang; BROKEN on error.
# ──────────────────────────────────────────────────────────────────────────
def test_liveness_probe_live() -> str:
    with tempfile.TemporaryDirectory(prefix="clio_wm_probe_live_") as base:
        wm = fresh_wm(base)
        try:
            res = liveness.probe_working_memory(wm, timeout=3.0)
            assert res["status"] == "LIVE", f"healthy tier probed {res}"
            # the probe must clean up after itself
            assert not [e for e in wm._entries.values() if e.domain == "_liveness"], "probe entry left behind"
        finally:
            wm.stop()
    return "probe LIVE on healthy tier + cleans up its synthetic entry"


def test_liveness_probe_timeout() -> str:
    """PLANT a hang (add() sleeps past the timeout) → probe must TIMEOUT, not hang."""
    with tempfile.TemporaryDirectory(prefix="clio_wm_probe_to_") as base:
        wm = fresh_wm(base)
        try:
            def hung_add(*a, **k):
                time.sleep(30)  # simulate the deadlocked add()
                return "never"
            wm.add = hung_add
            status, res = bounded(lambda: liveness.probe_working_memory(wm, timeout=1.0), timeout=8.0)
            assert status == "ok", f"probe itself hung ({status}) — it inherited the failure it must catch"
            assert res["status"] == "TIMEOUT", f"planted hang not caught: {res}"
        finally:
            # restore add before stop (stop()->flush doesn't use add, but be clean)
            try: del wm.__dict__["add"]
            except Exception: pass
            wm.stop()
    return "probe returns TIMEOUT on a planted hang WITHOUT hanging the caller"


def test_liveness_probe_broken() -> str:
    with tempfile.TemporaryDirectory(prefix="clio_wm_probe_brk_") as base:
        wm = fresh_wm(base)
        try:
            def boom(*a, **k):
                raise OSError("disk gone")
            wm.add = boom
            res = liveness.probe_working_memory(wm, timeout=3.0)
            assert res["status"] == "BROKEN", f"planted error not caught: {res}"
            assert "disk gone" in res["detail"]
        finally:
            try: del wm.__dict__["add"]
            except Exception: pass
            wm.stop()
    return "probe returns BROKEN on a planted write error"


# ──────────────────────────────────────────────────────────────────────────
# 8. Consumer-side liveness assertion — healthy = no marker; broken = LOUD marker.
# ──────────────────────────────────────────────────────────────────────────
def test_consumer_marker_red_green() -> str:
    with tempfile.TemporaryDirectory(prefix="clio_wm_consumer_") as base:
        wm = fresh_wm(base)

        class FakeEngine:  # avoids constructing a full MemoryEngine (production paths)
            pass
        fe = FakeEngine()
        fe.working = wm
        try:
            # GREEN: healthy → no marker
            assert MemoryEngine._memory_liveness_marker(fe) == "", "false-alarm on healthy tier"
            # RED (planted error): last_persist_error set
            wm._last_persist_error = "simulated disk failure"
            m = MemoryEngine._memory_liveness_marker(fe)
            assert "MEMORY DEGRADED" in m and "UNAVAILABLE" in m, f"marker did not fire: {m!r}"
            wm._last_persist_error = None
            assert MemoryEngine._memory_liveness_marker(fe) == "", "marker stuck after restore"
            # RED (planted dead flusher)
            wm._stop_event.set(); wm._flush_event.set(); wm._flusher.join(timeout=2.0)
            assert not wm._flusher.is_alive(), "flusher should be dead for this probe"
            m2 = MemoryEngine._memory_liveness_marker(fe)
            assert "MEMORY DEGRADED" in m2, f"dead-flusher not surfaced: {m2!r}"
        finally:
            wm.stop()
    return "consumer marker: silent when healthy, LOUD on planted error AND dead flusher"


# ──────────────────────────────────────────────────────────────────────────
# 9. Concurrency stress — CLIO bar: 10 threads × 50 ops × 5 runs.
#    No deadlock, valid JSON, ≤ MAX_ENTRIES, no duplicate ids, concurrent flush().
# ──────────────────────────────────────────────────────────────────────────
def test_concurrency_stress() -> str:
    THREADS, OPS, RUNS = 10, 50, 5
    for run in range(RUNS):
        with tempfile.TemporaryDirectory(prefix=f"clio_wm_stress_{run}_") as base:
            wm = fresh_wm(base)
            errors = []

            def hammer(tid):
                try:
                    for i in range(OPS):
                        wm.add(content=f"stress run{run} t{tid} op{i} — a sufficiently long unique entry",
                                domain=("trading", "research", "personal")[i % 3], priority=0.5)
                        if i % 17 == 0:
                            wm.get_context(max_chars=800)   # concurrent read
                        if i % 23 == 0:
                            wm.flush()                       # concurrent synchronous flush vs flusher
                except Exception as e:
                    errors.append(repr(e))

            threads = [threading.Thread(target=hammer, args=(t,)) for t in range(THREADS)]
            t0 = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join(JOIN_TIMEOUT)
            alive = [t for t in threads if t.is_alive()]
            assert not alive, f"run {run}: {len(alive)} threads DEADLOCKED after {JOIN_TIMEOUT}s"
            assert not errors, f"run {run}: worker errors {errors[:3]}"
            wm.flush()
            # File integrity
            data = json.load(open(wmmod.WORKING_MEMORY_PATH))
            assert isinstance(data, dict), f"run {run}: corrupt file (not a dict)"
            assert len(data) <= wmmod.MAX_ENTRIES, f"run {run}: {len(data)} > MAX_ENTRIES {wmmod.MAX_ENTRIES}"
            ids = list(data.keys())
            assert len(ids) == len(set(ids)), f"run {run}: duplicate ids on disk"
            for k, v in data.items():
                assert v.get("id") == k and "content" in v, f"run {run}: malformed entry {k}"
            wm.stop()
    return f"{THREADS}t×{OPS}op×{RUNS}runs: no deadlock, no corruption, ≤{wmmod.MAX_ENTRIES} entries, no dup ids"


# ──────────────────────────────────────────────────────────────────────────
# WAVE 2 — Phase 3/4 hardening contract tests.
# ──────────────────────────────────────────────────────────────────────────
def test_promotion_drop_and_flag() -> str:
    """Phase 3: a DEAD working source must DROP-AND-FLAG (loud) and must NOT call
    get_promotable() (no silent zero-promotion). An INFRA error on the health read
    SKIPS (retryable)."""
    with tempfile.TemporaryDirectory(prefix="clio_wm_promote_") as base:
        wm = fresh_wm(base)

        class FakeEngine:
            pass
        fe = FakeEngine()
        fe.working = wm
        called = {"get_promotable": 0}
        wm.get_promotable = lambda *a, **k: called.__setitem__("get_promotable", called["get_promotable"] + 1) or []
        try:
            # healthy → promotion path runs (get_promotable called)
            MemoryEngine._check_promotions(fe)
            assert called["get_promotable"] == 1, "healthy source should run promotion"
            # DEAD source → DROP-AND-FLAG: get_promotable must NOT be called again
            wm._last_persist_error = "planted dead source"
            MemoryEngine._check_promotions(fe)
            assert called["get_promotable"] == 1, "dead source must SUSPEND promotion (not silently empty)"
            wm._last_persist_error = None
            MemoryEngine._check_promotions(fe)
            assert called["get_promotable"] == 2, "promotion should resume once source healthy"
        finally:
            wm.stop()
    return "promotion drop-and-flags a dead source (no silent zero-promotion); resumes when healthy"


def test_file_age_not_liveness() -> str:
    """Phase 4: persist_health reflects the REAL write path, NOT file mtime. A fresh
    file on disk does NOT mask a broken write path."""
    with tempfile.TemporaryDirectory(prefix="clio_wm_fileage_") as base:
        wm = fresh_wm(base)
        try:
            wm.add(content="entry to create a fresh on-disk file for the file-age test", domain="trading")
            wm.flush()
            assert os.path.getmtime(wmmod.WORKING_MEMORY_PATH)  # file exists, mtime fresh
            assert not wm.persist_health()["broken"], "healthy write path"
            # Plant a write-path failure: file mtime is still fresh, but health must be BROKEN.
            wm._last_persist_error = "write path down despite fresh file"
            ph = wm.persist_health()
            assert ph["broken"], "persist_health was fooled by a fresh file mtime (file-age != liveness)"
        finally:
            wm._last_persist_error = None
            wm.stop()
    return "persist_health tracks the write path, not file mtime (fresh file does not mask breakage)"


def test_schema_drift_roundtrip() -> str:
    """Phase 4: write→read contract. Unknown fields on disk are filtered on load;
    known fields survive intact (no silent unreadable entries on schema drift)."""
    with tempfile.TemporaryDirectory(prefix="clio_wm_schema_") as base:
        from pathlib import Path
        wmmod.WORKING_MEMORY_PATH = Path(base) / "working_memory.json"
        # Plant a file with an entry carrying an UNKNOWN future field + a real entry.
        planted = {
            "wm_known": {
                "id": "wm_known", "content": "a normal entry that must survive reload intact",
                "domain": "trading", "priority": 0.6,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=72)).isoformat(),
                "future_field_from_v2": {"nested": True},  # unknown → must be filtered, not crash
            },
        }
        with open(wmmod.WORKING_MEMORY_PATH, "w") as f:
            json.dump(planted, f)
        wm = WorkingMemory()  # loads the planted file
        try:
            assert "wm_known" in wm._entries, "known entry dropped on schema drift (load too strict)"
            e = wm._entries["wm_known"]
            assert e.content == "a normal entry that must survive reload intact"
            assert not hasattr(e, "future_field_from_v2"), "unknown field leaked onto the dataclass"
            # round-trip: re-save → reload → still intact
            wm.add(content="second entry to force a fresh save round-trip now", domain="research")
            wm.flush()
            wm2 = WorkingMemory()
            assert "wm_known" in wm2._entries, "known entry lost across save/reload round-trip"
            wm2.stop()
        finally:
            wm.stop()
    return "unknown on-disk fields filtered on load; known fields survive the write→read round-trip"


def test_cross_tier_liveness_degraded() -> str:
    """Phase 3/4: _collect_liveness flags a broken working tier (degraded=True +
    alarm) and stays clean when healthy."""
    with tempfile.TemporaryDirectory(prefix="clio_wm_xtier_") as base:
        wm = fresh_wm(base)

        class StubEp:
            def get_recent(self, limit=1):
                return [{"timestamp": datetime.now(timezone.utc).isoformat()}]
            def stats(self):
                return {"total_size_kb": 1.0}

        class FakeEngine:
            pass
        fe = FakeEngine()
        fe.working = wm
        fe.episodic = StubEp()
        fe._last_working_probe = {"status": "LIVE"}
        try:
            rep = MemoryEngine._collect_liveness(fe)
            assert rep["degraded"] is False and not rep["alarms"], f"false alarm when healthy: {rep}"
            wm._last_persist_error = "planted breakage"
            rep2 = MemoryEngine._collect_liveness(fe)
            assert rep2["degraded"] is True and any("working" in a for a in rep2["alarms"]), f"did not flag: {rep2}"
        finally:
            wm._last_persist_error = None
            wm.stop()
    return "cross-tier liveness flags a broken working tier (degraded+alarm), silent when healthy"


def main() -> int:
    tests = [
        ("1  add/promote/clear no-deadlock", test_add_promote_clear_no_deadlock),
        ("2  re-entrancy guard raises", test_reentrancy_guard_raises),
        ("3  CONTRACT no-IO-under-lock", test_contract_no_io_under_lock),
        ("4  persistence round-trip", test_persist_roundtrip),
        ("5  liveness probe LIVE", test_liveness_probe_live),
        ("6  liveness probe TIMEOUT (planted hang)", test_liveness_probe_timeout),
        ("7  liveness probe BROKEN (planted error)", test_liveness_probe_broken),
        ("8  consumer marker RED→GREEN", test_consumer_marker_red_green),
        ("9  promotion drop-and-flag (Phase 3)", test_promotion_drop_and_flag),
        ("10 file-age != liveness (Phase 4)", test_file_age_not_liveness),
        ("11 schema-drift write→read contract (Phase 4)", test_schema_drift_roundtrip),
        ("12 cross-tier liveness degraded (Phase 3/4)", test_cross_tier_liveness_degraded),
        ("13 concurrency stress 10×50×5", test_concurrency_stress),
    ]
    failed = 0
    for name, fn in tests:
        try:
            detail = fn()
            print(f"PASS  {name} — {detail}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name} — {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name} — {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
