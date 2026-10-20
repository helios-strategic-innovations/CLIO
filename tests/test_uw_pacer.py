"""UW-1 (S211, ADR-005): the process-wide UW request pacer.

UW Basic rejects bursts (measured: 0/488 429s at <=3 same-second requests, ~31% at 12-14).
The pacer hands out send slots ~1/rate apart. Everything here uses a fake clock -- no real sleeping.
"""

import asyncio
import threading
import time

import pytest

from agent.integrations import uw_pacer
from agent.integrations.uw_pacer import UWPacer


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _pacer(rate=3.0, burst=1, penalty_s=60.0, enabled=True):
    clock = Clock()
    return UWPacer(rate_per_s=rate, burst=burst, penalty_s=penalty_s, enabled=enabled, clock=clock), clock


def test_simultaneous_callers_get_slots_one_over_rate_apart():
    pacer, _ = _pacer()
    waits = [pacer.reserve() for _ in range(14)]  # DPOF's 14-request refresh, all at t0
    assert waits[0] == 0.0
    for k, w in enumerate(waits):
        assert w == pytest.approx(k / 3.0)
    assert waits[-1] == pytest.approx(13 / 3.0)  # ~4.3 s: inside DPOF's 8 s per-call budget


def test_no_calendar_second_holds_more_than_four_sends():
    """The measured-safe regime is <=3 same-second requests; allow 4 for float edge effects."""
    pacer, clock = _pacer()
    sends = sorted(clock() + pacer.reserve() for _ in range(90))
    per_second = {}
    for t in sends:
        per_second[int(t)] = per_second.get(int(t), 0) + 1
    assert max(per_second.values()) <= 4
    assert 2.5 <= len(sends) / (sends[-1] - sends[0] + 1e-9) <= 3.6  # averages ~3 req/s


def test_bucket_refills_after_idle_and_never_exceeds_burst():
    pacer, clock = _pacer(burst=2)
    assert [pacer.reserve() for _ in range(2)] == [0.0, 0.0]
    assert pacer.reserve() > 0.0  # third caller must wait
    clock.advance(600.0)  # long idle: debt repaid, bucket capped at the burst size
    assert [pacer.reserve() for _ in range(2)] == [0.0, 0.0]
    assert pacer.reserve() > 0.0


def test_note_429_halves_once_per_cooldown_floors_and_restores():
    pacer, clock = _pacer(rate=3.0, penalty_s=60.0)
    assert pacer.note_429() == pytest.approx(1.5)
    assert pacer.note_429() == pytest.approx(1.5)  # same congestion event: no second halving
    assert pacer.n_429 == 2
    clock.advance(5.1)
    assert pacer.note_429() == pytest.approx(0.75)
    clock.advance(5.1)
    assert pacer.note_429() == pytest.approx(0.5)  # floored
    clock.advance(5.1)
    assert pacer.note_429() == pytest.approx(0.5)  # never below the floor
    clock.advance(59.0)
    assert pacer.effective_rate() == pytest.approx(0.5)  # still inside the penalty window
    clock.advance(2.0)
    assert pacer.effective_rate() == pytest.approx(3.0)  # quiet for penalty_s -> base rate restored


def test_slower_rate_during_penalty_spaces_slots_wider():
    pacer, clock = _pacer(rate=3.0)
    pacer.note_429()  # 1.5 req/s
    waits = [pacer.reserve() for _ in range(4)]
    assert waits[1] == pytest.approx(1 / 1.5)
    assert waits[3] == pytest.approx(3 / 1.5)


def test_disabled_pacer_never_waits(monkeypatch):
    pacer, _ = _pacer(enabled=False)

    async def must_not_sleep(_s):
        raise AssertionError("disabled pacer must not sleep")

    monkeypatch.setattr(asyncio, "sleep", must_not_sleep)
    assert asyncio.run(pacer.acquire()) == 0.0


def test_acquire_sleeps_exactly_the_reserved_delay(monkeypatch):
    pacer, _ = _pacer()
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def three():
        return [await pacer.acquire() for _ in range(3)]

    waits = asyncio.run(three())
    assert waits == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3)]
    assert slept == [pytest.approx(1 / 3), pytest.approx(2 / 3)]


def test_cancelled_waiter_forfeits_its_slot_without_corrupting_state(monkeypatch):
    pacer, _ = _pacer()

    async def cancelled_sleep(_s):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", cancelled_sleep)
    pacer.reserve()  # take the free slot

    async def victim():
        await pacer.acquire()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(victim())
    nxt = pacer.reserve()  # state is still consistent: the next slot is simply later
    assert nxt == pytest.approx(2 / 3.0)


class _YieldingClock(Clock):
    """Frozen time, but every read yields the GIL -- without the pacer's lock the threads interleave
    inside reserve() and hand out duplicate slots (plain frozen-clock threads hide the race)."""

    def __call__(self):
        time.sleep(0.0003)
        return self.t


def test_threaded_reservations_are_unique_and_gap_free():
    """8 threads x 25 reservations against a GIL-yielding clock: every slot exactly once."""
    clock = _YieldingClock()
    pacer = UWPacer(rate_per_s=3.0, burst=1, clock=clock)
    waits, lock = [], threading.Lock()

    def worker():
        mine = [pacer.reserve() for _ in range(25)]
        with lock:
            waits.extend(mine)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(round(w * 3.0) for w in waits) == list(range(200))


class _SpyLock:
    """Counts acquisitions; lets a test pin lock discipline deterministically (a GIL race is too
    narrow to hit reliably)."""

    def __init__(self):
        self.enters = 0

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, *exc):
        return False


def test_every_state_access_takes_the_lock():
    pacer, _ = _pacer()
    spy = _SpyLock()
    pacer._lock = spy
    pacer.reserve()
    pacer.note_429()
    pacer.effective_rate()
    assert spy.enters == 3


def test_env_configuration_and_garbage_tolerance(monkeypatch):
    monkeypatch.setenv("CLIO_UW_RATE_PER_SEC", "not-a-number")
    monkeypatch.setenv("CLIO_UW_BURST", "nan")
    monkeypatch.setenv("CLIO_UW_PENALTY_S", "-5")
    uw_pacer.reset_uw_pacer_for_tests()
    p = uw_pacer.get_uw_pacer()
    assert p.enabled is True and p.effective_rate() == pytest.approx(3.0)

    monkeypatch.setenv("CLIO_UW_PACER_ENABLED", "0")
    monkeypatch.setenv("CLIO_UW_RATE_PER_SEC", "50")  # clamped to the module ceiling
    uw_pacer.reset_uw_pacer_for_tests()
    p = uw_pacer.get_uw_pacer()
    assert p.enabled is False and p.effective_rate() == pytest.approx(20.0)


def test_singleton_is_stable_until_reset():
    uw_pacer.reset_uw_pacer_for_tests()
    assert uw_pacer.get_uw_pacer() is uw_pacer.get_uw_pacer()
