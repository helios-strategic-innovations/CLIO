"""Tests for the DPOF learning loop wiring in agent/dpof/engine.py (Task
ORCH-7Q4K-DPOF builder B3, 2026-09-26): F1 post-trade lessons (reflection)
and F2 borderline debate context. Design frame:
docs/design/clio_dpof_learning_loop.md.

Hermetic: every collaborator (broker, feed, decider, OrderManager, jev
sentinel, advisory LLMs, reflection service) is faked or, where the real
`agent.dpof.reflection`/`agent.dpof.ledger` modules are exercised, pointed at
a tmp_path data dir -- never network, never the repo's real `data/` tree
(the `_no_data_dir_pollution` guard below mirrors test_engine.py's own).

Reuses test_engine.py's fakes/builders (`FakeBroker`, `FakeDecider`,
`FakeOrderManager`, `FakeFeed`, `FakeJevClient`, `FakeMemory`, `_base_dt`,
`_bullish_snapshot`, `_make_chain`, `_position`, `_dedupe_alerts`,
`_patch_entry_candidate_cls`, `_patch_jev_entry_check`, `_fake_candidate_fn`,
`_fake_jev_sentinel_no_escalate`) via the same `from tests.dpof.test_engine
import (...)` pattern test_engine_signal_log.py already uses.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import replace
from typing import Any, Optional

import pytest

from agent.dpof.advisory_llm import AdvisoryResult
from agent.dpof.alerts import DPOFAlerts
from agent.dpof.config import DpofMode, JevMode
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import (
    JevVerdict,
    MarketSnapshot,
    OptionQuote,
    PillarReading,
    UnderlyingSnapshot,
)

from tests.dpof.test_engine import (
    FakeBroker,
    FakeDecider,
    FakeEntryDecision,
    FakeFeed,
    FakeJevClient,
    FakeMemory,
    FakeOrderManager,
    FakeReviewDecision,
    _base_dt,
    _bullish_snapshot,
    _dedupe_alerts,
    _fake_candidate_fn,
    _fake_jev_sentinel_no_escalate,
    _make_chain,
    _neutral_snapshot,
    _patch_entry_candidate_cls,
    _patch_jev_entry_check,
    _position,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────
# Hermeticity guard (mirrors test_engine.py's own)
# ─────────────────────────────────────────────────────────────────────────

def _data_dir_git_status() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", "data/"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


@pytest.fixture(autouse=True)
def _no_data_dir_pollution():
    before = _data_dir_git_status()
    yield
    after = _data_dir_git_status()
    assert after == before, (
        "DPOF learning-loop test wrote into the repo's data/ directory -- before:\n"
        f"{before!r}\nafter:\n{after!r}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Frozen ledger clock (OP-15)
#
# DPOFLedger.load() archives closed records older than 14 days against the
# wall clock. The fixtures in test_engine.py are pinned to 2026-09-24, so
# from 2026-10-08 the closed test position was archived at load and
# has_lesson() went False. Freeze the ledger's `time.time()` at the fixed
# base instant; every other attribute still delegates to the real module.
# ─────────────────────────────────────────────────────────────────────────

class _TimeShim:
    """Stands in for the `time` module inside `agent.dpof.ledger`."""

    def __init__(self, fixed_epoch: float):
        self._fixed_epoch = fixed_epoch

    def time(self) -> float:
        return self._fixed_epoch

    def __getattr__(self, name: str):
        import time as _real_time
        return getattr(_real_time, name)


@pytest.fixture(autouse=True)
def _frozen_ledger_clock(monkeypatch):
    shim = _TimeShim(_base_dt().timestamp())
    monkeypatch.setattr("agent.dpof.ledger.time", shim)
    yield shim


def test_frozen_ledger_clock_shim_is_active():
    import time as _real_time
    import agent.dpof.ledger as _ledger

    assert _ledger.time.time() == _base_dt().timestamp()
    assert _ledger.time.monotonic is _real_time.monotonic


# ─────────────────────────────────────────────────────────────────────────
# Fakes: AdvisoryLLM (F1 + F2) + ReflectionService
# ─────────────────────────────────────────────────────────────────────────

class FakeAdvisoryLLM:
    """Stands in for `agent.dpof.advisory_llm.AdvisoryLLM`. `results` is
    either a single `AdvisoryResult` reused for every call, or a zero-arg-
    minus-purpose callable `fn(purpose) -> AdvisoryResult` for scripting
    per-side (bull/bear) or per-attempt responses."""

    def __init__(self, results: Any = None):
        self.calls: list[dict] = []
        self.results = results
        self.closed = False

    async def complete_json(
        self, *, purpose, system, user, schema, schema_name, primary_model,
        fallback_model, timeout_seconds, max_tokens=800, budget=None, effort="low",
    ) -> AdvisoryResult:
        self.calls.append({
            "purpose": purpose, "budget": budget, "effort": effort,
            "primary_model": primary_model, "fallback_model": fallback_model,
        })
        if callable(self.results):
            return self.results(purpose)
        if self.results is not None:
            return self.results
        return AdvisoryResult(ok=False, parsed=None, provider="", model="", latency_ms=None, error="unscripted")

    def status(self) -> dict:
        return {"providers": {}}

    async def close(self) -> None:
        self.closed = True


def _advisory_ok(parsed: dict, provider="anthropic", model="claude-sonnet-5", latency_ms=100) -> AdvisoryResult:
    return AdvisoryResult(ok=True, parsed=parsed, provider=provider, model=model, latency_ms=latency_ms, error="")


def _advisory_fail(error="boom") -> AdvisoryResult:
    return AdvisoryResult(ok=False, parsed=None, provider="", model="", latency_ms=None, error=error)


_DEBATE_SIDE = {"argument": "strongest honest case", "key_risk": "the flow could reverse", "strength": 0.65}
_LESSON_PARSED = {
    "lesson": "Flow confirmed early and the move followed through as expected.",
    "entry_quality": "good", "exit_quality": "good", "key_factor": "flow_signal", "confidence": 0.7,
}


class FakeReflectionService:
    """Stands in for `agent.dpof.reflection.ReflectionService` when a test
    needs full control over `run_pending`/`lessons_ctx` without exercising
    real storage. `block_forever=True` makes `run_pending` await an
    `asyncio.Event` the test never sets until `release()` -- used to prove
    engine-level single-flight scheduling."""

    def __init__(self, block_forever: bool = False, run_pending_result: Optional[dict] = None,
                 lessons_ctx_result: Optional[dict] = None):
        self.run_pending_calls: list[tuple] = []
        self.lessons_ctx_calls: list[tuple] = []
        self.block_forever = block_forever
        self.run_pending_result = run_pending_result if run_pending_result is not None else {
            "attempted": 0, "stored": 0, "failed": 0, "positions": [],
        }
        self.lessons_ctx_result = lessons_ctx_result if lessons_ctx_result is not None else {
            "same": [], "cross": [], "tally": {},
        }
        self._event = asyncio.Event() if block_forever else None

    async def run_pending(self, positions, now):
        self.run_pending_calls.append((list(positions), now))
        if self.block_forever:
            await self._event.wait()
        return self.run_pending_result

    def release(self) -> None:
        if self._event is not None:
            self._event.set()

    def lessons_ctx(self, underlying, *, same_n=5, cross_n=3):
        self.lessons_ctx_calls.append((underlying, same_n, cross_n))
        return self.lessons_ctx_result


async def _fake_jev_sentinel_escalate(cfg, client, position_ctx):
    from agent.dpof.jev import JevEscalation

    return JevEscalation(escalate=True, skipped=False)


def _align_against_fn(*_a, **_kw) -> dict:
    return {"flow": "against", "momentum": "neutral", "vwap": "neutral"}


def _align_against_once_fn():
    """"against" on the first call, "neutral" from the second call onward --
    used where a test needs the SECOND (harvest) cycle's re-evaluation of
    triggers to NOT relaunch a second review (engine.py relaunches within the
    same harvest cycle once a position's review Task is popped, before that
    cycle's trigger loop runs again for the same still-open position)."""
    calls = {"n": 0}

    def _fn(*_a, **_kw) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"flow": "against", "momentum": "neutral", "vwap": "neutral"}
        return {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"}

    return _fn


def _align_neutral_fn(*_a, **_kw) -> dict:
    return {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"}


# ─────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────

def _make_ll_engine(
    *, cfg, broker=None, feed=None, orders=None, ledger=None, jev=None, alerts=None,
    decider=None, memory=None, clock=None, market_open_fn=None,
    candidate_fn=None, jev_sentinel=None, position_alignment_fn=None,
    advisory_reflection=None, advisory_debate=None, reflection_service=None, lesson_store=None,
):
    """Same defaulting shape as test_engine.py's `_make_engine`, extended
    with the learning-loop collaborator injection points added to
    `DPOFEngine.__init__` by this builder."""
    from agent.dpof.engine import DPOFEngine

    return DPOFEngine(
        memory=memory or FakeMemory(),
        broker=broker, feed=feed, orders=orders, jev=jev or FakeJevClient(), alerts=alerts, ledger=ledger,
        decider=decider,
        advisory_reflection=advisory_reflection, advisory_debate=advisory_debate,
        reflection_service=reflection_service, lesson_store=lesson_store,
        candidate_fn=candidate_fn or _fake_candidate_fn,
        jev_sentinel=jev_sentinel or _fake_jev_sentinel_no_escalate,
        position_alignment_fn=position_alignment_fn or _align_neutral_fn,
        clock=clock or (lambda: _base_dt().timestamp()),
        market_open_fn=market_open_fn or (lambda: True),
        config_loader=lambda: cfg,
    )


def _borderline_underlying(symbol="SPY") -> UnderlyingSnapshot:
    """Same shape as test_engine.py's `_bullish_underlying`, but weak enough
    that the composite lands just inside threshold_normal + debate_margin
    (a genuinely borderline "normal" nomination) instead of "strong"."""
    return UnderlyingSnapshot(
        symbol=symbol,
        pillars={
            "nofa": PillarReading(name="nofa", value=0.20, available=True, source="test"),
            "premium": PillarReading(name="premium", value=0.15, available=True, source="test"),
            "darkpool": PillarReading(name="darkpool", value=0.15, available=True, source="test"),
        },
        gex_regime=None, pin_risk=None, last_price=100.5, vwap=100.0,
        ret_5m=0.005, ret_15m=0.01, bars_count=20, price_age_seconds=10.0,
    )


def _borderline_snapshot(now_epoch, symbol="SPY") -> MarketSnapshot:
    return MarketSnapshot(
        asof=now_epoch, underlyings={symbol: _borderline_underlying(symbol)},
        vix=16.0, vix_age_seconds=30.0, risk_flags=[], headlines=[], econ_events_soon=[],
        data_healthy=True, health_issues=[],
    )


def _read_decisions(data_dir: str) -> list[dict]:
    path = os.path.join(data_dir, "decisions.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _base_entry_cfg(cfg, **overrides):
    base = dict(
        mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",),
        alpaca_key_id="k", alpaca_secret_key="s", allow_verticals=False,
        reflection_enabled=False,
    )
    base.update(overrides)
    return replace(cfg, **base)


# ─────────────────────────────────────────────────────────────────────────
# A. Entry debate gating (§5.1/§5.2, R3)
# ─────────────────────────────────────────────────────────────────────────

def test_entry_debate_runs_on_borderline_normal_signal(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = _base_entry_cfg(cfg, debate_enabled=True)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _borderline_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=True))
    decider = FakeDecider(entry_result=FakeEntryDecision(action="pass"))
    advisory_debate = FakeAdvisoryLLM(results=_advisory_ok(dict(_DEBATE_SIDE)))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            advisory_debate=advisory_debate, clock=lambda: clock_box["t"],
        )

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        # Bull + bear, in parallel -- exactly two complete_json calls.
        assert len(advisory_debate.calls) == 2
        assert {c["purpose"] for c in advisory_debate.calls} == {"debate_entry_bull", "debate_entry_bear"}

        ctx = decider.entry_calls[-1][0]
        assert "debate" in ctx
        assert ctx["debate"]["available"] is True
        assert ctx["debate"]["rendered"]

        records = _read_decisions(c.data_dir)
        last = next(r for r in reversed(records) if r.get("kind") == "entry_deliberation")
        assert last["debate"]["ran"] is True
        assert last["debate"]["available"] is True
        assert last["debate"]["bull_strength"] == pytest.approx(0.65)
        assert last["debate"]["bear_strength"] == pytest.approx(0.65)

    asyncio.run(_t())


def test_entry_debate_skipped_on_strong_signal(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = _base_entry_cfg(cfg, debate_enabled=True)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))  # composite ~0.547 -> "strong"
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=True))
    decider = FakeDecider(entry_result=FakeEntryDecision(action="pass"))
    advisory_debate = FakeAdvisoryLLM()
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            advisory_debate=advisory_debate, clock=lambda: clock_box["t"],
        )

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        assert advisory_debate.calls == []
        ctx = decider.entry_calls[-1][0]
        assert "debate" not in ctx

        records = _read_decisions(c.data_dir)
        last = next(r for r in reversed(records) if r.get("kind") == "entry_deliberation")
        assert "debate" not in last

    asyncio.run(_t())


def test_entry_debate_failure_still_decides_and_marks_unavailable(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = _base_entry_cfg(cfg, debate_enabled=True)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _borderline_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=True))
    decider = FakeDecider(entry_result=FakeEntryDecision(action="pass"))
    advisory_debate = FakeAdvisoryLLM(results=_advisory_fail("no_key"))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            advisory_debate=advisory_debate, clock=lambda: clock_box["t"],
        )

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        # The decision still happened (decider.entry_calls not empty) even
        # though every debate call failed.
        assert len(decider.entry_calls) == 1
        ctx = decider.entry_calls[-1][0]
        assert "debate" not in ctx

        records = _read_decisions(c.data_dir)
        last = next(r for r in reversed(records) if r.get("kind") == "entry_deliberation")
        assert last["debate"]["ran"] is True
        assert last["debate"]["available"] is False

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# B. Review debate gating (§5.1/§5.2, R3)
# ─────────────────────────────────────────────────────────────────────────

def _review_cfg(cfg, **overrides):
    base = dict(
        mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",),
        alpaca_key_id="k", alpaca_secret_key="s", reflection_enabled=False,
    )
    base.update(overrides)
    return replace(cfg, **base)


def _make_review_scenario(c, *, jev_sentinel, position_alignment_fn, advisory_debate,
                           review_result=None):
    base_epoch = _base_dt().timestamp()
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=review_result or FakeReviewDecision(action="hold"))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, qty_open=2, qty_original=2, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            jev_sentinel=jev_sentinel, position_alignment_fn=position_alignment_fn,
            advisory_debate=advisory_debate, clock=lambda: clock_box["t"],
        )
        # Launch cycle, then a second cycle to harvest the (already-resolved
        # fake) review Task and write its decision_log -- same two-cycle
        # pattern test_engine.py's own review tests use.
        await eng.run_cycle(bot=None)
        await eng._review_tasks["P1"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)
        return eng, decider

    return asyncio.run(_t())


def test_review_debate_runs_on_alignment_against_with_enough_time(cfg):
    c = _review_cfg(cfg)  # default flatten_et="15:50", now=11:00 -> 290 min, well over 30
    advisory_debate = FakeAdvisoryLLM(results=_advisory_ok(dict(_DEBATE_SIDE)))
    eng, decider = _make_review_scenario(
        c, jev_sentinel=_fake_jev_sentinel_no_escalate, position_alignment_fn=_align_against_once_fn(),
        advisory_debate=advisory_debate,
    )

    assert len(advisory_debate.calls) == 2
    assert {call["purpose"] for call in advisory_debate.calls} == {"debate_review_bull", "debate_review_bear"}
    ctx = decider.review_calls[0][0]
    assert "debate" in ctx
    assert ctx["debate"]["rendered"]

    records = _read_decisions(c.data_dir)
    last = next(r for r in reversed(records) if r.get("kind") == "position_review")
    assert last["debate"]["available"] is True


def test_review_debate_skipped_with_jev_escalation(cfg):
    c = _review_cfg(cfg, jev_api_key="test-key")  # jev must be ACTIVE for the sentinel to be called at all
    advisory_debate = FakeAdvisoryLLM()
    eng, decider = _make_review_scenario(
        c, jev_sentinel=_fake_jev_sentinel_escalate, position_alignment_fn=_align_against_fn,
        advisory_debate=advisory_debate,
    )

    assert advisory_debate.calls == []
    ctx = decider.review_calls[-1][0]
    assert "debate" not in ctx


def test_review_debate_skipped_too_close_to_flatten(cfg):
    # now=11:00 -> 20 minutes to flatten, below the 30-min floor. entry_end_et
    # must stay < flatten_et (DPOFConfig.validate()), so it moves in lockstep.
    c = _review_cfg(cfg, flatten_et="11:20", entry_end_et="11:00")
    advisory_debate = FakeAdvisoryLLM()
    eng, decider = _make_review_scenario(
        c, jev_sentinel=_fake_jev_sentinel_no_escalate, position_alignment_fn=_align_against_fn,
        advisory_debate=advisory_debate,
    )

    assert advisory_debate.calls == []
    ctx = decider.review_calls[-1][0]
    assert "debate" not in ctx


def test_review_debate_skipped_when_debate_on_reviews_false(cfg):
    c = _review_cfg(cfg, debate_on_reviews=False)
    advisory_debate = FakeAdvisoryLLM()
    eng, decider = _make_review_scenario(
        c, jev_sentinel=_fake_jev_sentinel_no_escalate, position_alignment_fn=_align_against_fn,
        advisory_debate=advisory_debate,
    )

    assert advisory_debate.calls == []
    ctx = decider.review_calls[-1][0]
    assert "debate" not in ctx


# ─────────────────────────────────────────────────────────────────────────
# C. Lessons in ctx (§4.6)
# ─────────────────────────────────────────────────────────────────────────

_LESSONS_CTX_WITH_DATA = {
    "same": [{
        "underlying": "SPY", "direction": "bullish", "closed_at": "2026-09-20T10:00:00-04:00",
        "pnl_usd": -50.0, "pnl_pct": -0.1, "exit_category": "decider_exit", "key_factor": "theta",
        "lesson": "Theta ate the position faster than the thesis played out.",
    }],
    "cross": [],
    "tally": {"SPY": {"n": 1, "wins": 0, "losses": 1, "net_pnl_usd": -50.0, "avg_pnl_pct": -0.1}},
}


def test_entry_lessons_ctx_present_with_stored_lesson(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = _base_entry_cfg(cfg, debate_enabled=False, lessons_in_prompt=True)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=True))
    decider = FakeDecider(entry_result=FakeEntryDecision(action="pass"))
    fake_reflection = FakeReflectionService(lessons_ctx_result=_LESSONS_CTX_WITH_DATA)
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            reflection_service=fake_reflection, clock=lambda: clock_box["t"],
        )
        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]

        ctx = decider.entry_calls[-1][0]
        assert "lessons" in ctx
        assert ctx["lessons"]["same"] == _LESSONS_CTX_WITH_DATA["same"]
        assert ctx["lessons"]["tally"] == _LESSONS_CTX_WITH_DATA["tally"]
        assert ctx["lessons"]["max_chars"] == c.lessons_prompt_max_chars
        assert fake_reflection.lessons_ctx_calls == [("SPY", c.lessons_same_underlying, c.lessons_cross)]

    asyncio.run(_t())


def test_entry_lessons_ctx_absent_when_none_stored(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = _base_entry_cfg(cfg, debate_enabled=False, lessons_in_prompt=True)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=True))
    decider = FakeDecider(entry_result=FakeEntryDecision(action="pass"))
    fake_reflection = FakeReflectionService()  # default: all-empty lessons_ctx
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            reflection_service=fake_reflection, clock=lambda: clock_box["t"],
        )
        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]

        ctx = decider.entry_calls[-1][0]
        assert "lessons" not in ctx

    asyncio.run(_t())


def test_entry_lessons_ctx_absent_when_lessons_in_prompt_false(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = _base_entry_cfg(cfg, debate_enabled=False, lessons_in_prompt=False)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=True))
    decider = FakeDecider(entry_result=FakeEntryDecision(action="pass"))
    fake_reflection = FakeReflectionService(lessons_ctx_result=_LESSONS_CTX_WITH_DATA)
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            reflection_service=fake_reflection, clock=lambda: clock_box["t"],
        )
        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]

        ctx = decider.entry_calls[-1][0]
        assert "lessons" not in ctx
        assert fake_reflection.lessons_ctx_calls == []  # gate blocks BEFORE calling lessons_ctx

    asyncio.run(_t())


def test_review_lessons_ctx_present_with_stored_lesson(cfg):
    """_make_review_scenario doesn't take a `reflection_service=` override, so
    this test builds the review scenario directly instead."""
    c = _review_cfg(cfg, lessons_in_prompt=True)
    fake_reflection = FakeReflectionService(lessons_ctx_result=_LESSONS_CTX_WITH_DATA)

    base_epoch = _base_dt().timestamp()
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=FakeReviewDecision(action="hold"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, qty_open=2, qty_original=2, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            reflection_service=fake_reflection, clock=lambda: base_epoch,
        )
        await eng.run_cycle(bot=None)
        await eng._review_tasks["P1"]

        ctx = decider.review_calls[-1][0]
        assert "lessons" in ctx
        assert ctx["lessons"]["tally"] == _LESSONS_CTX_WITH_DATA["tally"]

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# D. Data capture (§4.1)
# ─────────────────────────────────────────────────────────────────────────

def test_entry_sets_entry_underlying_price_on_open(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = _base_entry_cfg(cfg, debate_enabled=False)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))  # last_price=100.5
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=True))
    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=0.8, qty=1))

    async def open_position_paper(mgr, choice, qty, signal, jev, now_epoch, judgment):
        from agent.dpof.models import PositionRecord

        pos = PositionRecord(
            id="P1", underlying=choice.underlying, direction=choice.direction.value, structure=choice.structure,
            legs=[{"symbol": l.symbol, "side": l.side, "ratio_qty": l.ratio_qty, "position_intent": l.position_intent} for l in choice.legs],
            status="pending", qty_original=qty, qty_open=0, mode="paper",
            thesis=judgment["thesis"], invalidation=judgment["invalidation"],
            entry_decision=judgment["entry_decision"], next_review_at=judgment["next_review_at"],
        )
        await mgr.ledger.upsert(pos)
        return pos

    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        orders.open_position_return = open_position_paper
        eng = _make_ll_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        pos = eng._ledger.get("P1")
        assert pos is not None
        assert pos.entry_underlying_price == pytest.approx(100.5)

    asyncio.run(_t())


def test_full_close_sets_exit_price_and_obs_extremes_and_pops(cfg):
    base_epoch = _base_dt().timestamp()
    c = replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",),
        alpaca_key_id="k", alpaca_secret_key="s", reflection_enabled=False, debate_enabled=False,
    )
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    price_box = {"px": 100.0}
    feed = FakeFeed(lambda now: MarketSnapshot(
        asof=now, underlyings={"SPY": UnderlyingSnapshot(symbol="SPY", pillars={}, last_price=price_box["px"], vwap=100.0, ret_15m=0.0)},
        vix=16.0, vix_age_seconds=30.0, data_healthy=True,
    ))
    decider = FakeDecider()

    async def _idle_step(ledger, now_epoch):
        return []

    def _close_step(pos_id, realized_pnl):
        async def _step(ledger, now_epoch):
            pos = ledger.get(pos_id)
            updated = replace(pos, status="closed", qty_open=0, realized_pnl=realized_pnl, exit_reason="decider_exit", closed_time=now_epoch)
            await ledger.upsert(updated)
            return [{"type": "exit_filled", "position_id": pos_id, "qty": pos.qty_open, "realized_pnl_delta": realized_pnl, "closed": True}]
        return _step

    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position(
            "P1", "SPY", leg_symbol, entry_price=2.0, qty_open=2, qty_original=2,
            entry_time=base_epoch - 3600, next_review_at=base_epoch + 100_000,
        ))
        orders = FakeOrderManager(ledger=ledger)
        orders.work_orders_script = [_idle_step, _idle_step, _close_step("P1", 20.0)]
        eng = _make_ll_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        # Cycle 1: mark down (pnl_pct = -0.04), snapshot last_price=100.0.
        broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.90, ask=1.94)}
        price_box["px"] = 100.0
        await eng.run_cycle(bot=None)
        assert eng._pnl_extremes["P1"] == pytest.approx([-0.04, -0.04])

        # Cycle 2: mark up (pnl_pct = +0.06), snapshot last_price=101.0 -- this
        # becomes self._last_snapshot for the CLOSING cycle below (the fresh
        # snapshot for cycle 3 isn't fetched until after the close is applied).
        clock_box["t"] += 30
        broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=2.10, ask=2.14)}
        price_box["px"] = 101.0
        await eng.run_cycle(bot=None)
        assert eng._pnl_extremes["P1"] == pytest.approx([0.06, -0.04])

        # Cycle 3: work_orders_script closes the position.
        clock_box["t"] += 30
        price_box["px"] = 102.0  # irrelevant -- exit_underlying_price uses cycle 2's snapshot
        await eng.run_cycle(bot=None)

        pos = eng._ledger.get("P1")
        assert pos.status == "closed"
        assert pos.exit_underlying_price == pytest.approx(101.0)
        assert pos.obs_max_pnl_pct == pytest.approx(0.06)
        assert pos.obs_min_pnl_pct == pytest.approx(-0.04)
        assert "P1" not in eng._pnl_extremes

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# E. Reflection scheduling (§4.3)
# ─────────────────────────────────────────────────────────────────────────

def _sched_cfg(cfg, **overrides):
    base = dict(mode=DpofMode.LOG_ONLY, alpaca_key_id="k", alpaca_secret_key="s")
    base.update(overrides)
    return replace(cfg, **base)


def test_reflection_scheduled_after_a_cycle(cfg):
    base_epoch = _base_dt().timestamp()
    c = _sched_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.LOG_ONLY)
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    fake_reflection = FakeReflectionService()

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=FakeDecider(), ledger=ledger,
            reflection_service=fake_reflection, clock=lambda: base_epoch,
        )
        await eng.run_cycle(bot=None)
        assert eng._reflection_task is not None
        await eng._reflection_task
        await asyncio.sleep(0)

        assert len(fake_reflection.run_pending_calls) == 1
        positions, now = fake_reflection.run_pending_calls[0]
        assert positions == list(ledger.positions)
        assert now == base_epoch

    asyncio.run(_t())


def test_reflection_disabled_never_schedules(cfg):
    base_epoch = _base_dt().timestamp()
    c = _sched_cfg(cfg, reflection_enabled=False)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.LOG_ONLY)
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    fake_reflection = FakeReflectionService()

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=FakeDecider(), ledger=ledger,
            reflection_service=fake_reflection, clock=lambda: base_epoch,
        )
        await eng.run_cycle(bot=None)
        assert eng._reflection_task is None
        assert fake_reflection.run_pending_calls == []

    asyncio.run(_t())


def test_reflection_runs_on_market_closed_path(cfg):
    base_epoch = _base_dt().timestamp()
    c = _sched_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.LOG_ONLY)
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    fake_reflection = FakeReflectionService()

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=FakeDecider(), ledger=ledger,
            reflection_service=fake_reflection, clock=lambda: base_epoch, market_open_fn=lambda: False,
        )
        r = await eng.run_cycle(bot=None)
        assert r["skipped"] == "market_closed"
        assert eng._reflection_task is not None
        await eng._reflection_task
        await asyncio.sleep(0)

        assert len(fake_reflection.run_pending_calls) == 1

    asyncio.run(_t())


def test_reflection_single_flight_across_back_to_back_cycles(cfg):
    base_epoch = _base_dt().timestamp()
    c = _sched_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.LOG_ONLY)
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    fake_reflection = FakeReflectionService(block_forever=True)
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=FakeDecider(), ledger=ledger,
            reflection_service=fake_reflection, clock=lambda: clock_box["t"],
        )

        await eng.run_cycle(bot=None)
        await asyncio.sleep(0)  # let the fire-and-forget Task actually start running
        assert len(fake_reflection.run_pending_calls) == 1
        task = eng._reflection_task
        assert task is not None and not task.done()

        clock_box["t"] += 30
        await eng.run_cycle(bot=None)  # still in flight -- must NOT schedule a second one
        await asyncio.sleep(0)
        assert len(fake_reflection.run_pending_calls) == 1
        assert eng._reflection_task is task

        fake_reflection.release()
        await task
        await asyncio.sleep(0)
        assert eng._reflection_task is None

    asyncio.run(_t())


def test_reflection_stores_lesson_and_sends_alert(cfg):
    base_epoch = _base_dt().timestamp()
    c = replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",),
        alpaca_key_id="k", alpaca_secret_key="s", debate_enabled=False,
    )
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    sent, sender = _dedupe_alerts()
    alerts = DPOFAlerts(sender=sender)
    fake_advisory = FakeAdvisoryLLM(results=_advisory_ok(dict(_LESSON_PARSED)))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position(
            "P1", "SPY", "SPYTESTC00500000", status="closed", qty_open=0, qty_original=2,
            entry_price=2.0, entry_time=base_epoch - 3600, closed_time=base_epoch - 60,
            realized_pnl=150.0, exit_reason="decider_exit", mode="paper",
        ))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=FakeDecider(), ledger=ledger,
            alerts=alerts, advisory_reflection=fake_advisory, clock=lambda: base_epoch,
        )

        await eng.run_cycle(bot=object())
        assert eng._reflection_task is not None
        await eng._reflection_task
        await asyncio.sleep(0)

        assert eng._lesson_store.has_lesson("P1")
        lesson_alerts = [s for s in sent if s.startswith("DPOF lesson")]
        assert len(lesson_alerts) == 1
        assert "SPY" in lesson_alerts[0]

    asyncio.run(_t())


def test_reflection_new_engine_over_same_data_dir_does_not_reflect_twice(cfg):
    c = replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",),
        alpaca_key_id="k", alpaca_secret_key="s", debate_enabled=False,
    )

    async def _run_one(clock_value, advisory):
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        if ledger.get("P1") is None:
            await ledger.upsert(_position(
                "P1", "SPY", "SPYTESTC00500000", status="closed", qty_open=0, qty_original=2,
                entry_price=2.0, entry_time=clock_value - 3600, closed_time=clock_value - 60,
                realized_pnl=150.0, exit_reason="decider_exit", mode="paper",
            ))
        broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
        feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=FakeDecider(), ledger=ledger,
            advisory_reflection=advisory, clock=lambda: clock_value,
        )
        await eng.run_cycle(bot=None)
        if eng._reflection_task is not None:
            await eng._reflection_task
            await asyncio.sleep(0)
        return eng

    base_epoch = _base_dt().timestamp()
    advisory1 = FakeAdvisoryLLM(results=_advisory_ok(dict(_LESSON_PARSED)))
    advisory2 = FakeAdvisoryLLM(results=_advisory_ok(dict(_LESSON_PARSED)))

    async def _t():
        eng1 = await _run_one(base_epoch, advisory1)
        assert eng1._lesson_store.has_lesson("P1")
        assert len(advisory1.calls) == 1

        eng2 = await _run_one(base_epoch + 120, advisory2)
        assert eng2._lesson_store.has_lesson("P1")  # loaded from disk, not re-derived
        assert advisory2.calls == []  # never re-reflected

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# F. Invariance: both features off reproduces the pre-existing baseline
# ─────────────────────────────────────────────────────────────────────────

def test_learning_loop_disabled_matches_pre_existing_entry_behavior(cfg, monkeypatch):
    """Mirrors test_engine.py's test_entry_happy_path_launches_then_applies_
    on_next_cycle with debate/reflection/lessons explicitly OFF -- proves the
    learning loop introduces no behavior change when disabled."""
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)

    c = replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",),
        alpaca_key_id="k", alpaca_secret_key="s", allow_verticals=False,
        debate_enabled=False, reflection_enabled=False, lessons_in_prompt=False,
    )
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=True))

    decider = FakeDecider(entry_result=FakeEntryDecision(
        action="enter", candidate_key="C1", conviction=0.8, qty=3, thesis="strong flow", invalidation="composite reverses",
    ))
    submitted = {}

    async def open_position_paper(mgr, choice, qty, signal, jev, now_epoch, judgment):
        from agent.dpof.models import PositionRecord

        submitted["qty"] = qty
        submitted["judgment"] = judgment
        pos = PositionRecord(
            id="P1", underlying=choice.underlying, direction=choice.direction.value, structure=choice.structure,
            legs=[{"symbol": l.symbol, "side": l.side, "ratio_qty": l.ratio_qty, "position_intent": l.position_intent} for l in choice.legs],
            status="pending", qty_original=qty, qty_open=0, mode="paper",
            thesis=judgment["thesis"], invalidation=judgment["invalidation"],
            entry_decision=judgment["entry_decision"], next_review_at=judgment["next_review_at"],
        )
        await mgr.ledger.upsert(pos)
        return pos

    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        orders.open_position_return = open_position_paper

        eng = _make_ll_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        r1 = await eng.run_cycle(bot=object())
        assert "SPY" in eng._entry_tasks
        assert orders.open_calls == []

        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        r2 = await eng.run_cycle(bot=object())

        assert len(orders.open_calls) == 1
        assert submitted["qty"] == 3
        assert submitted["judgment"]["thesis"] == "strong flow"
        assert submitted["judgment"]["entry_decision"]["candidate_key"] == "C1"
        assert eng._ledger.get("P1") is not None
        assert eng._entry_tasks == {}

        # No "debate"/"lessons" ever entered the ctx the decider saw, and no
        # debate summary landed in decisions.jsonl -- byte-identical to the
        # pre-learning-loop shape.
        ctx = decider.entry_calls[-1][0]
        assert "debate" not in ctx
        assert "lessons" not in ctx
        records = _read_decisions(c.data_dir)
        last = next(r for r in reversed(records) if r.get("kind") == "entry_deliberation")
        assert "debate" not in last

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# G. obs_* grep invariant
# ─────────────────────────────────────────────────────────────────────────

_OBS_FIELD_RE = re.compile(r"\bobs_(?:max|min)_pnl_pct\b")


def test_obs_pnl_fields_read_only_by_data_capture_and_reflection():
    """obs_max_pnl_pct/obs_min_pnl_pct back ONLY post-trade learning records --
    the lesson (reflection.py) and the decision-record outcome row's detail
    (agent/decisions/records.py); nothing in exits.py/review triggers/gates may
    ever read them (CLAUDE.md-style invariant named in the design frame's §4.1)."""
    # Only the repo's own TRACKED .py files -- a virtualenv inside the checkout
    # (the VPS has ./venv and ./atlas_rag_sidecar/.venv) is full of unrelated
    # "obs_" substrings in third-party packages.
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=str(ROOT), capture_output=True, text=True, check=True,
    )
    hits = set()
    for rel in filter(None, listed.stdout.split("\0")):
        if rel.startswith("tests/"):
            continue
        with open(ROOT / rel, encoding="utf-8", errors="replace") as f:
            # The field names themselves, word-bounded: a bare "obs_" substring also matches
            # unrelated identifiers (e.g. "probs_per_dim" in the vendored Kronos model).
            if _OBS_FIELD_RE.search(f.read()):
                hits.add(rel)
    allowed = {
        "agent/dpof/models.py", "agent/dpof/engine.py", "agent/dpof/reflection.py",
        "agent/decisions/records.py",
    }
    assert hits == allowed, f"unexpected obs_ references outside the allowed files: {hits ^ allowed}"


def test_pnl_extremes_pruned_for_positions_no_longer_open(cfg):
    """A position closed outside the exit_filled path (e.g. reconciled_absent)
    must not leave its extremes entry in memory forever."""
    base_epoch = _base_dt().timestamp()
    c = _sched_cfg(cfg)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.LOG_ONLY)
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_ll_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=FakeDecider(), ledger=ledger,
            reflection_service=FakeReflectionService(), clock=lambda: base_epoch,
        )
        eng._pnl_extremes["gone_position"] = [0.2, -0.1]
        await eng.run_cycle(bot=None)
        assert "gone_position" not in eng._pnl_extremes

    asyncio.run(_t())
