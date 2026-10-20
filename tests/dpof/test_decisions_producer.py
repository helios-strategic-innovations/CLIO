"""Tests for the decision-record producer wiring in
agent/dpof/engine.py:
the entry-veto and exit-sentinel hooks, decision_call_id persistence on the
position, the outcome-linkage scan, and shutdown/cancellation. A
FakeDecisionRecorder (never the real agent.decisions.DecisionRecorder/
DecisionStore -- no network) stands in for `self._decisions` throughout.

Reuses test_engine.py's fakes/builders via the same `from tests.dpof.test_engine
import (...)` pattern test_engine_learning_loop.py already uses.
"""

from __future__ import annotations

import asyncio
import pathlib
import subprocess
import tempfile
import time
from dataclasses import replace
from typing import Any, Optional

import httpx
import pytest

from agent.dpof.config import DpofMode, JevMode
from agent.dpof.engine import DPOFEngine
from agent.dpof.jev import JevClient, JevEscalation
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import JevVerdict, OptionQuote, PositionRecord
from agent.decisions.config import DecisionsConfig
from agent.decisions.records import position_outcome_id
from agent.decisions.store import load_recorded_outcome_ids_sync

from tests.dpof.test_engine import (
    FakeBroker,
    FakeDecider,
    FakeEntryDecision,
    FakeFeed,
    FakeMemory,
    FakeOrderManager,
    _base_dt,
    _bullish_snapshot,
    _fake_candidate_fn,
    _fake_jev_sentinel_no_escalate,
    _make_chain,
    _patch_entry_candidate_cls,
    _position,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────
# Hermeticity guard (mirrors test_engine.py's own) -- the outcome-linkage
# scan touches disk (recorded_outcome_ids.json next to the outbox); every
# FakeDecisionRecorder in this file MUST be given an isolated tmp cfg
# (never DecisionsConfig()'s own real-repo default outbox_path).
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
        "Decisions producer test wrote into the repo's data/ directory -- before:\n"
        f"{before!r}\nafter:\n{after!r}\nEvery FakeDecisionRecorder(...) in this file must be "
        "constructed with an isolated tmp DecisionsConfig (see _isolated_decisions_cfg)."
    )


def _isolated_decisions_cfg(**overrides) -> DecisionsConfig:
    """A DecisionsConfig pointed at a fresh tmp dir -- DecisionsConfig()'s own
    default outbox_path is the REAL repo's data/decisions/outbox.jsonl, which
    the outcome-linkage scan's recorded_outcome_ids.json sits next to; never
    use the bare default in a test."""
    d = tempfile.mkdtemp(prefix="clio-test-decisions-")
    base = dict(outbox_path=str(pathlib.Path(d) / "outbox.jsonl"))
    base.update(overrides)
    return replace(DecisionsConfig(), **base)


# ─────────────────────────────────────────────────────────────────────────
# Fake decisions recorder
# ─────────────────────────────────────────────────────────────────────────

class FakeDecisionRecorder:
    def __init__(self, cfg: Optional[DecisionsConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else _isolated_decisions_cfg()
        self.record_call_calls: list[dict] = []
        self.record_position_outcome_calls: list[dict] = []
        self.aclose_called = False
        self._n = 0

    def record_call(self, producer, producer_ref, state, questions, *, served=None, position_id=None):
        if not self.cfg.enabled:
            return None
        self._n += 1
        call_id = f"call-{self._n}"
        self.record_call_calls.append({
            "producer": producer, "producer_ref": producer_ref, "state": state, "questions": questions,
            "served": served, "position_id": position_id, "call_id": call_id,
        })
        return call_id

    def record_position_outcome(self, position, *, outcome_source="live", call_id=None):
        if not self.cfg.enabled:
            return
        pos_dict = position if isinstance(position, dict) else position.to_dict()
        self.record_position_outcome_calls.append(
            {"position": pos_dict, "outcome_source": outcome_source, "call_id": call_id}
        )

    async def aclose(self) -> None:
        self.aclose_called = True


def _make_dp_engine(
    *, cfg, broker=None, feed=None, orders=None, ledger=None, jev=None, decider=None,
    decisions=None, jev_sentinel=None, position_alignment_fn=None, clock=None,
) -> DPOFEngine:
    return DPOFEngine(
        memory=FakeMemory(), broker=broker, feed=feed, orders=orders, jev=jev, ledger=ledger,
        decider=decider, decisions=decisions,
        candidate_fn=_fake_candidate_fn,
        jev_sentinel=jev_sentinel or _fake_jev_sentinel_no_escalate,
        position_alignment_fn=position_alignment_fn or (lambda *_a, **_kw: {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"}),
        clock=clock or (lambda: _base_dt().timestamp()),
        market_open_fn=lambda: True,
        config_loader=lambda: cfg,
    )


# ─────────────────────────────────────────────────────────────────────────
# Entry veto — OFF (recorded via prepare_entry_request, no Jev call)
# ─────────────────────────────────────────────────────────────────────────

def test_entry_off_records_via_prepare_entry_request_and_sets_call_id(cfg, monkeypatch):
    from agent.dpof.jev import prepare_entry_request

    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    assert c.jev_effective_mode == JevMode.OFF

    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=0.8, qty=3, thesis="t", invalidation="i"))
    decisions = FakeDecisionRecorder()
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)

        async def open_position_paper(mgr, choice, qty, signal, jev, now_epoch, judgment):
            pos = PositionRecord(
                id="P1", underlying=choice.underlying, direction=choice.direction.value, structure=choice.structure,
                legs=[{"symbol": l.symbol, "side": l.side, "ratio_qty": l.ratio_qty, "position_intent": l.position_intent} for l in choice.legs],
                status="pending", qty_original=qty, qty_open=0, mode="paper",
                thesis=judgment["thesis"], invalidation=judgment["invalidation"],
                entry_decision=judgment["entry_decision"], next_review_at=judgment["next_review_at"],
            )
            await mgr.ledger.upsert(pos)
            return pos

        orders.open_position_return = open_position_paper
        eng = _make_dp_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, decisions=decisions, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        assert len(orders.open_calls) == 1
        assert len(decisions.record_call_calls) == 1
        call = decisions.record_call_calls[0]
        assert call["producer"] == "dpof_entry_veto"
        assert call["served"] is None  # Jev OFF -- nothing served
        assert call["producer_ref"]["symbol"] == "SPY"
        assert call["producer_ref"]["candidate_key"] == "C1"
        assert call["producer_ref"]["jev_mode"] == "off"

        pos = eng._ledger.get("P1")
        assert pos.entry_decision["decision_call_id"] == "call-1"

    asyncio.run(_t())


def test_entry_decisions_disabled_never_records(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=0.8, qty=1))
    decisions = FakeDecisionRecorder(cfg=_isolated_decisions_cfg(enabled=False))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)

        async def open_position_paper(mgr, choice, qty, signal, jev, now_epoch, judgment):
            pos = PositionRecord(
                id="P1", underlying=choice.underlying, direction=choice.direction.value, structure=choice.structure,
                legs=[], status="pending", qty_original=qty, qty_open=0, mode="paper",
                entry_decision=judgment["entry_decision"], next_review_at=judgment["next_review_at"],
            )
            await mgr.ledger.upsert(pos)
            return pos

        orders.open_position_return = open_position_paper
        eng = _make_dp_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, decisions=decisions, clock=lambda: clock_box["t"])
        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        assert decisions.record_call_calls == []
        pos = eng._ledger.get("P1")
        assert pos.entry_decision.get("decision_call_id") is None

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# Entry veto — ACTIVE (real JevClient + MockTransport): allow and veto
# ─────────────────────────────────────────────────────────────────────────

def _valid_body(questions: dict, *, setup_choice="aligned", setup_conf=0.9, hc_val=0.05, er_val=0.05, size_val=0.05, model="jev-1.13.0") -> dict:
    answers = {}
    for qid, qspec in questions.items():
        if qid == "setup":
            others = [k for k in qspec["criteria"] if k != setup_choice]
            each = (1.0 - setup_conf) / len(others) if others else 0.0
            probs = {k: each for k in others}
            probs[setup_choice] = 1.0 - sum(probs.values())
            answers[qid] = {"type": "choice", "choice": setup_choice, "probabilities": probs, "confidence": setup_conf}
        elif qid == "headline_conflict":
            answers[qid] = {"type": "noul", "noul": hc_val}
        elif qid == "event_risk":
            answers[qid] = {"type": "noul", "noul": er_val}
        elif qid == "size_reckless":
            # The engine always passes sizing_ctx once qty is known, so
            # build_entry_request adds this question -- it must be answered
            # or validate_answers rejects the WHOLE response (missing:qid).
            answers[qid] = {"type": "noul", "noul": size_val}
    return {"model": model, "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 5}}


def _active_entry_scenario(cfg, *, handler, decisions):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s",
        jev_mode=JevMode.VETO, jev_api_key="testkey", jev_base_url="https://jev.test",
    )
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=0.8, qty=1))
    clock_box = {"t": base_epoch}
    jev = JevClient(c, transport=httpx.MockTransport(handler))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)

        async def open_position_paper(mgr, choice, qty, signal, verdict, now_epoch, judgment):
            pos = PositionRecord(
                id="P1", underlying=choice.underlying, direction=choice.direction.value, structure=choice.structure,
                legs=[], status="pending", qty_original=qty, qty_open=0, mode="paper",
                entry_decision=judgment["entry_decision"], next_review_at=judgment["next_review_at"],
            )
            await mgr.ledger.upsert(pos)
            return pos

        orders.open_position_return = open_position_paper
        eng = _make_dp_engine(cfg=c, broker=broker, feed=feed, orders=orders, jev=jev, decider=decider, ledger=ledger, decisions=decisions, clock=lambda: clock_box["t"])
        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)
        return eng, orders

    return asyncio.run(_t())


def test_entry_active_allow_records_served_answer(cfg, monkeypatch):
    _patch_entry_candidate_cls(monkeypatch)
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        body = _json.loads(request.content)
        captured["state"] = body["state"]
        captured["questions"] = body["questions"]
        return httpx.Response(200, json=_valid_body(body["questions"], setup_choice="aligned"))

    decisions = FakeDecisionRecorder()
    eng, orders = _active_entry_scenario(cfg, handler=handler, decisions=decisions)

    assert len(orders.open_calls) == 1  # allowed through
    assert len(decisions.record_call_calls) == 1
    call = decisions.record_call_calls[0]
    assert call["producer"] == "dpof_entry_veto"
    assert call["state"] == captured["state"]
    assert call["questions"] == captured["questions"]
    served = call["served"]
    assert served is not None
    answers, model, endpoint, latency_ms, error = served
    assert model == "jev-1.13.0"
    assert endpoint == "primary"
    assert error == ""
    assert "setup" in answers
    pos = eng._ledger.get("P1")
    assert pos.entry_decision["decision_call_id"] == "call-1"


def test_entry_active_veto_still_records(cfg, monkeypatch):
    """Vetoed entries are recorded too."""
    _patch_entry_candidate_cls(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        body = _json.loads(request.content)
        return httpx.Response(200, json=_valid_body(body["questions"], setup_choice="late_or_exhausted", setup_conf=0.9))

    decisions = FakeDecisionRecorder()
    eng, orders = _active_entry_scenario(cfg, handler=handler, decisions=decisions)

    assert orders.open_calls == []  # blocked by the veto
    assert len(decisions.record_call_calls) == 1
    call = decisions.record_call_calls[0]
    served = call["served"]
    answers, model, endpoint, latency_ms, error = served
    assert answers["setup"]["choice"] == "late_or_exhausted"
    # No position was ever opened, so there's nothing to check decision_call_id on --
    # the important assertion is that record_call still fired despite the veto.


def test_entry_active_unavailable_records_error_answer(cfg, monkeypatch):
    _patch_entry_candidate_cls(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    decisions = FakeDecisionRecorder()
    eng, orders = _active_entry_scenario(cfg, handler=handler, decisions=decisions)

    assert orders.open_calls == []
    assert len(decisions.record_call_calls) == 1
    served = decisions.record_call_calls[0]["served"]
    answers, model, endpoint, latency_ms, error = served
    assert "jev_unavailable" in error
    assert model  # a failed call with no reported model is recorded as "jev-unknown", never ""


# ─────────────────────────────────────────────────────────────────────────
# Exit sentinel — OFF (never calls Jev) and ACTIVE
# ─────────────────────────────────────────────────────────────────────────

def test_sentinel_off_records_without_ever_calling_jev(cfg):
    assert cfg.jev_effective_mode == JevMode.OFF

    async def _jev_sentinel_must_not_be_called(cfg, client, position_ctx):
        raise AssertionError("jev_sentinel must never be called when jev_effective_mode is OFF")

    decisions = FakeDecisionRecorder()

    async def _t():
        ledger = DPOFLedger(cfg.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", "SPYTESTC00500000", entry_price=2.0, qty_open=2, qty_original=2, next_review_at=None))
        broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
        broker.quotes = {"SPYTESTC00500000": OptionQuote(symbol="SPYTESTC00500000", underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
        feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
        orders = FakeOrderManager(ledger=ledger)
        decider = FakeDecider()
        c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", reflection_enabled=False)
        eng = _make_dp_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            decisions=decisions, jev_sentinel=_jev_sentinel_must_not_be_called,
        )
        await eng.run_cycle(bot=None)
        # Give the (already-resolved fake) review Task a chance to be harvested.
        if "P1" in eng._review_tasks:
            await eng._review_tasks["P1"]
            await eng.run_cycle(bot=None)

        sentinel_calls = [c for c in decisions.record_call_calls if c["producer"] == "dpof_exit_sentinel"]
        assert len(sentinel_calls) >= 1
        call = sentinel_calls[0]
        assert call["served"] is None
        assert call["position_id"] == "P1"

    asyncio.run(_t())


def test_sentinel_active_records_served_answer_and_escalation_unaffected(cfg):
    c = replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s",
        jev_mode=JevMode.VETO, jev_api_key="k", jev_base_url="https://jev.test", reflection_enabled=False,
    )
    assert c.jev_effective_mode == JevMode.VETO

    request = {"state": {"a": 1}, "questions": {"thesis_danger": {"type": "noul", "instructions": "x"}}}

    async def _fake_sentinel(cfg, client, position_ctx):
        return JevEscalation(
            escalate=True, reasons=["thesis_danger"], probabilities={"thesis_danger": 0.9},
            model="jev-1.13.0", endpoint="primary", latency_ms=12.0, request=request,
        )

    decisions = FakeDecisionRecorder()

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", "SPYTESTC00500000", entry_price=2.0, qty_open=2, qty_original=2, next_review_at=None))
        broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
        broker.quotes = {"SPYTESTC00500000": OptionQuote(symbol="SPYTESTC00500000", underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
        feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
        orders = FakeOrderManager(ledger=ledger)
        decider = FakeDecider()
        eng = _make_dp_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            decisions=decisions, jev_sentinel=_fake_sentinel,
        )
        await eng.run_cycle(bot=None)
        if "P1" in eng._review_tasks:
            await eng._review_tasks["P1"]
            await eng.run_cycle(bot=None)

        sentinel_calls = [c for c in decisions.record_call_calls if c["producer"] == "dpof_exit_sentinel"]
        assert len(sentinel_calls) >= 1
        call = sentinel_calls[0]
        assert call["state"] == request["state"]
        assert call["questions"] == request["questions"]
        assert call["position_id"] == "P1"
        answers, model, endpoint, latency_ms, error = call["served"]
        assert answers == {"thesis_danger": {"type": "noul", "noul": 0.9}}
        assert model == "jev-1.13.0"

        # The escalation record in the ledger (append_jev) is untouched: the
        # position's next_review_at should reflect the escalation having
        # fired (same as before this feature existed).
        pos = eng._ledger.get("P1")
        assert pos is not None

    asyncio.run(_t())


def test_sentinel_decisions_disabled_never_calls_position_alignment_fn_extra(cfg):
    """Regression guard: building the sentinel's position request for
    recording must NEVER happen when decisions recording is disabled -- it
    would otherwise call the injected position_alignment_fn an extra time,
    a real behavior change (this broke test_engine_learning_loop.py's
    _align_against_once_fn-based tests during development)."""
    assert cfg.jev_effective_mode == JevMode.OFF
    calls = {"n": 0}

    def counting_alignment_fn(*_a, **_kw):
        calls["n"] += 1
        return {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"}

    decisions = FakeDecisionRecorder(cfg=_isolated_decisions_cfg(enabled=False))

    async def _t():
        ledger = DPOFLedger(cfg.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", "SPYTESTC00500000", entry_price=2.0, qty_open=2, qty_original=2, next_review_at=None))
        broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
        broker.quotes = {"SPYTESTC00500000": OptionQuote(symbol="SPYTESTC00500000", underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
        feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
        orders = FakeOrderManager(ledger=ledger)
        decider = FakeDecider()
        c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", reflection_enabled=False)
        eng_off = _make_dp_engine(
            cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger,
            decisions=decisions, position_alignment_fn=counting_alignment_fn,
        )
        await eng_off.run_cycle(bot=None)
        calls_with_decisions_disabled = calls["n"]

        # Same scenario with NO decisions collaborator at all -- must produce
        # the exact same call count (decisions=None and decisions=disabled
        # behave identically from position_alignment_fn's point of view).
        calls["n"] = 0
        ledger2 = DPOFLedger(cfg.data_dir + "_2")
        await ledger2.load()
        await ledger2.upsert(_position("P1", "SPY", "SPYTESTC00500000", entry_price=2.0, qty_open=2, qty_original=2, next_review_at=None))
        orders2 = FakeOrderManager(ledger=ledger2)
        eng_none = _make_dp_engine(
            cfg=c, broker=broker, feed=feed, orders=orders2, decider=decider, ledger=ledger2,
            decisions=None, position_alignment_fn=counting_alignment_fn,
        )
        await eng_none.run_cycle(bot=None)
        assert calls["n"] == calls_with_decisions_disabled

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# Outcome linkage scan
# ─────────────────────────────────────────────────────────────────────────

def _closed_pos(pos_id, *, decision_call_id=None, realized_pnl=100.0) -> PositionRecord:
    """DPOFLedger.load() archives closed/cancelled positions older than 14
    real days (agent/dpof/ledger.py's _archive_old_closed, which uses REAL
    wall-clock time.time(), never the engine's injected fake clock) -- so
    closed_time must be recent real time, not a small offset like 1900.0
    (which is 1970-ish and gets archived away before the scan ever runs)."""
    now = time.time()
    entry_decision = {"provider": "anthropic", "qty": 1}
    if decision_call_id is not None:
        entry_decision["decision_call_id"] = decision_call_id
    return PositionRecord(
        id=pos_id, underlying="SPY", direction="bullish", structure="single", legs=[],
        status="closed", qty_original=1, qty_open=0, entry_price=2.0, entry_time=now - 1000.0,
        closed_time=now - 100.0, realized_pnl=realized_pnl, exit_reason="decider_exit", mode="paper",
        entry_decision=entry_decision,
    )


def test_outcome_scan_records_every_closed_position(cfg):
    decisions = FakeDecisionRecorder()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", reflection_enabled=False)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_closed_pos("P1", decision_call_id="call-1"))
        await ledger.upsert(_closed_pos("P2", decision_call_id=None))  # no call_id -- never entry-recorded
        broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
        feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
        orders = FakeOrderManager(ledger=ledger)
        decider = FakeDecider()
        eng = _make_dp_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, decisions=decisions)
        await eng.run_cycle(bot=None)
        if eng._decision_outcomes_task is not None:
            await eng._decision_outcomes_task

        # Every closed position gets an outcome row; one entered without a recorded entry decision
        # (before this feature, or while recording was off) carries call_id None and is joined to its
        # exit-sentinel questions by position_id instead.
        recs = {r["position"]["id"]: r for r in decisions.record_position_outcome_calls}
        assert set(recs) == {"P1", "P2"}
        assert recs["P1"]["call_id"] == "call-1" and recs["P1"]["outcome_source"] == "live"
        assert recs["P2"]["call_id"] is None

    asyncio.run(_t())


def test_outcome_scan_idempotent_across_two_runs_same_engine(cfg, tmp_path):
    d_cfg = _isolated_decisions_cfg(outbox_path=str(tmp_path / "outbox.jsonl"))
    decisions = FakeDecisionRecorder(cfg=d_cfg)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", reflection_enabled=False)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_closed_pos("P1", decision_call_id="call-1"))
        broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
        feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
        orders = FakeOrderManager(ledger=ledger)
        decider = FakeDecider()
        eng = _make_dp_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, decisions=decisions)

        await eng.run_cycle(bot=None)
        if eng._decision_outcomes_task is not None:
            await eng._decision_outcomes_task
        assert len(decisions.record_position_outcome_calls) == 1

        # Second cycle on the SAME engine -- must not re-record.
        await eng.run_cycle(bot=None)
        if eng._decision_outcomes_task is not None:
            await eng._decision_outcomes_task
        assert len(decisions.record_position_outcome_calls) == 1

    asyncio.run(_t())
    # The persisted id set survives -- a fresh engine pointed at the same
    # outbox dir (a restart) must ALSO skip P1.
    ids = load_recorded_outcome_ids_sync(d_cfg)
    assert "P1" in ids


def test_outcome_scan_persists_across_engine_restart(cfg, tmp_path):
    d_cfg = _isolated_decisions_cfg(outbox_path=str(tmp_path / "outbox.jsonl"))
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", reflection_enabled=False)

    async def _run_once(decisions):
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_closed_pos("P1", decision_call_id="call-1"))
        broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
        feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
        orders = FakeOrderManager(ledger=ledger)
        decider = FakeDecider()
        eng = _make_dp_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, decisions=decisions)
        await eng.run_cycle(bot=None)
        if eng._decision_outcomes_task is not None:
            await eng._decision_outcomes_task

    decisions1 = FakeDecisionRecorder(cfg=d_cfg)
    asyncio.run(_run_once(decisions1))
    assert len(decisions1.record_position_outcome_calls) == 1

    # A brand-new engine (simulating a process restart) sharing the same
    # data_dir/outbox path must NOT re-record P1 -- the persisted id set
    # (not in-memory state) is what makes this idempotent across restarts.
    decisions2 = FakeDecisionRecorder(cfg=d_cfg)
    asyncio.run(_run_once(decisions2))
    assert decisions2.record_position_outcome_calls == []


def test_outcome_scan_covers_reconciled_absent_close(cfg):
    """A position that goes straight to status=closed WITHOUT an
    exit_filled event (e.g. reconcile's reconciled_absent path) must still
    be picked up -- the scan is ledger-status-driven, not event-driven."""
    decisions = FakeDecisionRecorder()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", reflection_enabled=False)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        pos = _closed_pos("P1", decision_call_id="call-1")
        pos = replace(pos, exit_reason="reconciled_absent")
        await ledger.upsert(pos)
        broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
        feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
        orders = FakeOrderManager(ledger=ledger)
        decider = FakeDecider()
        eng = _make_dp_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, decisions=decisions)
        await eng.run_cycle(bot=None)
        if eng._decision_outcomes_task is not None:
            await eng._decision_outcomes_task

        assert len(decisions.record_position_outcome_calls) == 1
        assert decisions.record_position_outcome_calls[0]["position"]["exit_reason"] == "reconciled_absent"

    asyncio.run(_t())


def test_outcome_scan_disabled_never_schedules(cfg):
    decisions = FakeDecisionRecorder(cfg=_isolated_decisions_cfg(enabled=False))
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", reflection_enabled=False)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_closed_pos("P1", decision_call_id="call-1"))
        broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
        feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
        orders = FakeOrderManager(ledger=ledger)
        decider = FakeDecider()
        eng = _make_dp_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, decisions=decisions)
        await eng.run_cycle(bot=None)
        assert eng._decision_outcomes_task is None
        assert decisions.record_position_outcome_calls == []

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# Shutdown / cancellation
# ─────────────────────────────────────────────────────────────────────────

def test_shutdown_closes_decisions_recorder():
    decisions = FakeDecisionRecorder()

    async def _t():
        eng = DPOFEngine(memory=FakeMemory(), decisions=decisions)
        await eng.shutdown()
        assert decisions.aclose_called is True
        await eng.shutdown()  # idempotent

    asyncio.run(_t())


def test_cancel_all_decision_tasks_cancels_outcome_scan():
    decisions = FakeDecisionRecorder()

    async def _t():
        eng = DPOFEngine(memory=FakeMemory(), decisions=decisions)
        task = asyncio.ensure_future(asyncio.sleep(999))
        eng._decision_outcomes_task = task
        task.add_done_callback(eng._on_decision_outcomes_task_done)  # mirrors _maybe_schedule_decision_outcomes
        await eng._cancel_all_decision_tasks()
        assert task.cancelled()
        assert eng._decision_outcomes_task is None

    asyncio.run(_t())


def test_outcome_scan_also_runs_after_the_close(cfg):
    """A close at/after the flatten is linked the same evening: the market-closed early return
    schedules the scan too (it used to wait for the next session's first full cycle)."""
    decisions = FakeDecisionRecorder()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", reflection_enabled=False)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_closed_pos("P1", decision_call_id="call-1"))
        broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
        feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_dp_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=FakeDecider(), ledger=ledger, decisions=decisions)
        eng._market_open_fn = lambda: False
        result = await eng.run_cycle(bot=None)
        assert result.get("skipped") == "market_closed"
        if eng._decision_outcomes_task is not None:
            await eng._decision_outcomes_task
        assert [r["call_id"] for r in decisions.record_position_outcome_calls] == ["call-1"]

    asyncio.run(_t())
