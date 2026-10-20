"""Tests for agent.dpof.engine.DPOFEngine.

Judgment layer (operator directive 2026-09-24): every collaborator
(broker, feed, jev client, OrderManager, decider, candidate_fn,
jev_sentinel, position_alignment_fn) is FAKED so these tests are isolated
from the concurrent work on decider.py/contracts.py/signals.py/jev.py and
from network I/O. The ledger is a REAL agent.dpof.ledger.DPOFLedger pointed
at a tmp_path, so persistence/derivation semantics are exercised for real.
"""

from __future__ import annotations

import asyncio
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pytest

from agent.dpof import engine as engine_module
from agent.dpof.alerts import DPOFAlerts
from agent.dpof.config import DPOFConfig, DpofMode, JevMode
from agent.dpof.engine import DPOFEngine
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import (
    ContractChoice,
    Direction,
    GateResult,
    JevVerdict,
    MarketSnapshot,
    OptionQuote,
    OrderLeg,
    PillarReading,
    PositionRecord,
    UnderlyingSnapshot,
)

ET = ZoneInfo("America/New_York")
ROOT = pathlib.Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────
# Hermeticity guard
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
        "DPOF engine test wrote into the repo's data/ directory -- before:\n"
        f"{before!r}\nafter:\n{after!r}\nEvery DPOFEngine(...) in this file must "
        "inject memory=FakeMemory() rather than let the default _RealMemoryHooks() "
        "reach agent.memory.memory_engine.get_memory_engine()."
    )


class FakeMemory:
    def __init__(self, raise_on_remember=False, raise_on_learn=False):
        self.remember_calls: list[dict] = []
        self.learn_calls: list[dict] = []
        self.raise_on_remember = raise_on_remember
        self.raise_on_learn = raise_on_learn

    def remember(self, **kwargs):
        self.remember_calls.append(kwargs)
        if self.raise_on_remember:
            raise RuntimeError("boom: memory engine unavailable")

    def learn_from_outcome(self, **kwargs):
        self.learn_calls.append(kwargs)
        if self.raise_on_learn:
            raise RuntimeError("boom: memory engine unavailable")


# ─────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────

class FakeBroker:
    def __init__(self, gates_allowed=True, gate_reasons=None, effective=DpofMode.PAPER, account=None, chain=None, options_level=3):
        self.cfg = None
        self.gates_allowed = gates_allowed
        self.gate_reasons = gate_reasons or []
        self.effective_value = effective
        self.account = account if account is not None else {"equity": 20000.0, "options_trading_level": options_level}
        self.chain = chain
        self.quotes: dict[str, OptionQuote] = {}
        self.verify_gates_calls = 0
        self.get_account_calls = 0
        self.get_option_chain_calls: list[tuple] = []

    def update_config(self, cfg):
        self.cfg = cfg

    async def verify_gates(self, now_epoch):
        self.verify_gates_calls += 1
        return GateResult(allowed=self.gates_allowed, reasons=list(self.gate_reasons))

    def effective_mode(self):
        return self.effective_value

    async def get_account(self):
        self.get_account_calls += 1
        return self.account

    async def get_option_quotes(self, symbols):
        return {s: self.quotes[s] for s in symbols if s in self.quotes}

    async def get_option_chain(self, underlying, exp_min, exp_max, right):
        self.get_option_chain_calls.append((underlying, exp_min, exp_max, right))
        return self.chain


class FakeFeed:
    def __init__(self, snapshot_fn):
        self.cfg = None
        self.snapshot_fn = snapshot_fn
        self.closed = False

    def update_config(self, cfg):
        self.cfg = cfg

    async def snapshot(self, now_epoch):
        return self.snapshot_fn(now_epoch)

    async def close(self):
        self.closed = True


class FakeJevClient:
    """Stands in for JevClient itself -- jev_entry_check/jev_position_check
    are the real functions (imported directly in engine.py); this fake is
    what those functions would talk to, but since D2's jev_entry_check /
    jev_sentinel are exercised via monkeypatch in these tests, this class
    mainly needs to exist as `self._jev` for update_config()/close()."""

    def __init__(self):
        self.cfg = None
        self.closed = False

    def update_config(self, cfg):
        self.cfg = cfg

    async def close(self):
        self.closed = True


class FakeOrderManager:
    """Public surface: open_position/close_position/work_orders/reconcile/
    update_config. open_position accepts judgment= (optional) exactly like
    the real OrderManager."""

    def __init__(self, ledger=None):
        self.cfg = None
        self.ledger = ledger
        self.open_calls: list[tuple] = []
        self.close_calls: list[tuple] = []
        self.flatten_entry_calls: list[tuple] = []
        self.reconcile_calls = 0
        self.reconcile_now_epochs: list = []
        self.reconcile_return = []
        self.pending_qty_mismatch = False
        self.open_position_return = None  # PositionRecord|None, or async callable
        self.close_position_return = True
        self.close_position_side_effect = None
        self.flatten_entry_return = True
        self.flatten_entry_side_effect = None
        self.work_orders_script = []

    def update_config(self, cfg):
        self.cfg = cfg

    async def work_orders(self, now_epoch):
        if self.work_orders_script:
            step = self.work_orders_script.pop(0)
            return await step(self.ledger, now_epoch)
        return []

    async def open_position(self, choice, qty, signal, jev, now_epoch, *, judgment=None):
        self.open_calls.append((choice, qty, signal, jev, now_epoch, judgment))
        ret = self.open_position_return
        if callable(ret):
            return await ret(self, choice, qty, signal, jev, now_epoch, judgment)
        return ret

    async def close_position(self, pos, qty, reason, urgent, now_epoch):
        self.close_calls.append((pos.id, qty, reason, urgent, now_epoch))
        if self.close_position_side_effect is not None:
            await self.close_position_side_effect(self.ledger, pos, qty, reason, urgent, now_epoch)
        return self.close_position_return

    async def flatten_entry(self, pos, reason, now_epoch):
        self.flatten_entry_calls.append((pos.id, reason, now_epoch))
        if self.flatten_entry_side_effect is not None:
            await self.flatten_entry_side_effect(self.ledger, pos, reason, now_epoch)
        return self.flatten_entry_return

    def has_pending_qty_mismatch(self):
        return self.pending_qty_mismatch

    async def reconcile(self, now_epoch=None):
        self.reconcile_calls += 1
        self.reconcile_now_epochs.append(now_epoch)
        return self.reconcile_return


@dataclass
class FakeEntryDecision:
    action: str = "pass"
    candidate_key: Optional[str] = None
    conviction: float = 0.0
    qty: int = 0
    thesis: str = ""
    invalidation: str = ""
    recheck_minutes: int = 5
    provider: str = "anthropic"
    model: str = "claude-opus-5-5"
    latency_ms: int = 1200


@dataclass
class FakeReviewDecision:
    action: str = "hold"
    trim_fraction: float = 0.0
    urgency: str = "normal"
    thesis_status: str = "intact"
    reason: str = ""
    next_review_minutes: int = 5
    provider: str = "anthropic"
    model: str = "claude-opus-5-5"
    latency_ms: int = 900


class FakeDecider:
    """Injected in place of the real DPOFDecider. `entry_result`/
    `review_result` may be a FakeEntryDecision/FakeReviewDecision, an
    exception INSTANCE to raise, or a zero-arg callable returning either
    (for scripting successive calls). `block_forever=True` makes
    decide_entry/review_position await an asyncio.Event that the test
    never sets -- used to prove the cycle never awaits the decider."""

    def __init__(self, entry_result=None, review_result=None, block_forever=False):
        self.cfg = None
        self.entry_result = entry_result
        self.review_result = review_result
        self.block_forever = block_forever
        self.entry_calls: list[tuple] = []
        self.review_calls: list[tuple] = []
        self.closed = False
        self._block_event = asyncio.Event() if block_forever else None

    def update_config(self, cfg):
        self.cfg = cfg

    def _resolve(self, result):
        if callable(result) and not isinstance(result, (FakeEntryDecision, FakeReviewDecision)):
            result = result()
        if isinstance(result, Exception):
            raise result
        return result

    async def decide_entry(self, ctx, candidates):
        self.entry_calls.append((ctx, candidates))
        if self.block_forever:
            await self._block_event.wait()
        return self._resolve(self.entry_result if self.entry_result is not None else FakeEntryDecision())

    async def review_position(self, ctx, position):
        self.review_calls.append((ctx, position))
        if self.block_forever:
            await self._block_event.wait()
        return self._resolve(self.review_result if self.review_result is not None else FakeReviewDecision())

    def status(self) -> dict:
        return {"providers": {}, "calls_today": len(self.entry_calls) + len(self.review_calls), "est_cost_today_usd": 0.0}

    async def close(self):
        self.closed = True

    def release(self):
        if self._block_event is not None:
            self._block_event.set()


@dataclass
class FakeCandidate:
    """Duck-types agent.dpof.decider.EntryCandidate closely enough for the
    engine's own field reads (key/underlying/direction/structure/legs/mid/
    max_affordable_qty) without depending on decider.py's exact dataclass
    shape (which is mid-migration per the orchestrator's no-hard-caps
    directive: max_qty -> max_affordable_qty)."""
    key: str
    underlying: str
    direction: str
    structure: str
    legs: list
    expiry: str
    sessions_to_expiry: int
    strike_desc: str
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    spread_pct: Optional[float]
    delta: Optional[float]
    theta: Optional[float]
    iv: Optional[float]
    open_interest: Optional[int]
    unit_premium: Optional[float]
    max_affordable_qty: int


def _fake_candidate_cls(key, underlying, direction, structure, legs, expiry, sessions_to_expiry, strike_desc, bid, ask, mid, spread_pct, delta, theta, iv, open_interest, unit_premium, max_affordable_qty):
    return FakeCandidate(
        key=key, underlying=underlying, direction=direction, structure=structure, legs=legs, expiry=expiry,
        sessions_to_expiry=sessions_to_expiry, strike_desc=strike_desc, bid=bid, ask=ask, mid=mid,
        spread_pct=spread_pct, delta=delta, theta=theta, iv=iv, open_interest=open_interest,
        unit_premium=unit_premium, max_affordable_qty=max_affordable_qty,
    )


# ─────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────

def _base_dt() -> datetime:
    return datetime(2026, 9, 24, 11, 0, tzinfo=ET)


def _bullish_underlying(symbol="SPY") -> UnderlyingSnapshot:
    return UnderlyingSnapshot(
        symbol=symbol,
        pillars={
            "nofa": PillarReading(name="nofa", value=0.6, available=True, source="test"),
            "premium": PillarReading(name="premium", value=0.5, available=True, source="test"),
            "darkpool": PillarReading(name="darkpool", value=0.5, available=True, source="test"),
        },
        gex_regime=None,
        pin_risk=None,
        last_price=100.5,
        vwap=100.0,
        ret_5m=0.005,
        ret_15m=0.01,
        bars_count=20,
        price_age_seconds=10.0,
    )


def _bullish_snapshot(now_epoch, symbol="SPY", data_healthy=True, health_issues=None, headlines=None, risk_flags=None) -> MarketSnapshot:
    return MarketSnapshot(
        asof=now_epoch,
        underlyings={symbol: _bullish_underlying(symbol)},
        vix=16.0,
        vix_age_seconds=30.0,
        risk_flags=risk_flags or [],
        headlines=headlines or [],
        econ_events_soon=[],
        data_healthy=data_healthy,
        health_issues=health_issues or [],
    )


def _neutral_snapshot(now_epoch, symbol="SPY", data_healthy=True, health_issues=None) -> MarketSnapshot:
    u = UnderlyingSnapshot(symbol=symbol, pillars={}, last_price=100.0, vwap=100.0, ret_15m=0.0)
    return MarketSnapshot(
        asof=now_epoch, underlyings={symbol: u}, vix=16.0, vix_age_seconds=30.0,
        data_healthy=data_healthy, health_issues=health_issues or [],
    )


def _make_chain(symbol, today, bid=1.98, ask=2.02):
    from datetime import timedelta

    exp = (today + timedelta(days=3)).isoformat()
    q = OptionQuote(
        symbol=f"{symbol}TESTC00500000", underlying=symbol, expiration=exp, strike=500.0, right="call",
        bid=bid, ask=ask, delta=0.5, theta=-0.05, iv=0.2, open_interest=1000, volume=100,
    )
    return [q], q.symbol


def _position(pos_id, underlying, leg_symbol, **overrides):
    kwargs = dict(
        id=pos_id, underlying=underlying, direction="bullish", structure="single",
        legs=[{"symbol": leg_symbol, "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}],
        status="open", qty_original=2, qty_open=2, entry_price=2.0, entry_time=1000.0, mode="paper",
    )
    kwargs.update(overrides)
    return PositionRecord(**kwargs)


def _dedupe_alerts():
    sent = []

    async def sender(bot, text):
        sent.append(text)

    return sent, sender


def _make_engine(*, cfg, broker=None, feed=None, orders=None, ledger=None, jev=None, alerts=None,
                  decider=None, memory=None, clock=None, market_open_fn=None,
                  candidate_fn=None, jev_sentinel=None, position_alignment_fn=None) -> DPOFEngine:
    return DPOFEngine(
        memory=memory or FakeMemory(),
        broker=broker, feed=feed, orders=orders, jev=jev or FakeJevClient(), alerts=alerts, ledger=ledger,
        decider=decider,
        candidate_fn=candidate_fn or _fake_candidate_fn,
        jev_sentinel=jev_sentinel or _fake_jev_sentinel_no_escalate,
        position_alignment_fn=position_alignment_fn or (lambda *_a, **_kw: {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"}),
        clock=clock or (lambda: _base_dt().timestamp()),
        market_open_fn=market_open_fn or (lambda: True),
        config_loader=lambda: cfg,
    )


def _fake_candidate_fn(chain, *, underlying, direction, spot, today, cfg, now_et=None):
    """One single-leg candidate per liquid quote in `chain`, matching `direction`'s right."""
    from agent.dpof.contracts import option_right

    right = option_right(direction)
    out = []
    for q in chain:
        if q.right != right:
            continue
        leg = OrderLeg(symbol=q.symbol, side="buy", ratio_qty=1, position_intent="buy_to_open")
        out.append(ContractChoice(
            structure="single", legs=[leg], underlying=q.underlying, direction=direction,
            expiration=q.expiration, dte=3, est_debit=q.mid, est_bid=q.bid, est_ask=q.ask, long_delta=q.delta,
        ))
    return out


async def _fake_jev_sentinel_no_escalate(cfg, client, position_ctx):
    from agent.dpof.jev import JevEscalation

    return JevEscalation(escalate=False, skipped=(cfg.jev_effective_mode == JevMode.OFF))


def _patch_entry_candidate_cls(monkeypatch):
    """decider.EntryCandidate is imported lazily inside engine.py's own
    functions -- patch the module attribute D1 actually defines so the
    engine's `from agent.dpof.decider import EntryCandidate` picks up the
    fake constructor regardless of whether decider.py's own field names have
    caught up to the no-hard-caps `max_affordable_qty` rename yet."""
    import agent.dpof.decider as decider_mod

    monkeypatch.setattr(decider_mod, "EntryCandidate", _fake_candidate_cls, raising=False)


def _patch_jev_entry_check(monkeypatch, verdict_or_fn):
    async def _fake(signal, snap, now_et, cfg, client, sizing_ctx=None):
        if callable(verdict_or_fn) and not isinstance(verdict_or_fn, JevVerdict):
            return verdict_or_fn(signal, snap, now_et, cfg, client, sizing_ctx)
        return verdict_or_fn

    monkeypatch.setattr(engine_module, "jev_entry_check", _fake)


# ─────────────────────────────────────────────────────────────────────────
# 1. OFF
# ─────────────────────────────────────────────────────────────────────────

def test_off_does_nothing_and_constructs_nothing(cfg):
    off_cfg = replace(cfg, mode=DpofMode.OFF)

    async def _t():
        eng = _make_engine(cfg=off_cfg)
        r = await eng.run_cycle(bot=None)
        assert r == {"skipped": "off", "mode": "off", "effective_mode": "off", "unmanaged_positions": 0}
        assert eng._broker is None
        assert eng._feed is None
        assert eng._orders is None
        assert eng._decider is None
        assert eng._ledger is not None
        assert eng._alerts is not None

    asyncio.run(_t())


def test_off_alerts_once_on_stranded_positions(cfg):
    off_cfg = replace(cfg, mode=DpofMode.OFF)
    leg_symbol = "SPYTESTC00500000"

    async def _t():
        ledger = DPOFLedger(off_cfg.data_dir)
        await ledger.upsert(_position("STRANDED", "SPY", leg_symbol, mode="paper"))
        sent, sender = _dedupe_alerts()
        alerts = DPOFAlerts(sender=sender)
        eng = _make_engine(cfg=off_cfg, ledger=ledger, alerts=alerts)

        r = await eng.run_cycle(bot=object())
        assert r["unmanaged_positions"] == 1
        assert eng._broker is None
        assert len(sent) == 1
        assert "OFF with 1 open position" in sent[0]

        r2 = await eng.run_cycle(bot=object())
        assert r2["unmanaged_positions"] == 1
        assert len(sent) == 1

    asyncio.run(_t())


def test_off_cancels_in_flight_decision_tasks(cfg):
    """Task lifecycle (spec F): mode turning OFF cancels every in-flight
    entry/review decision task."""
    async def _t():
        eng = _make_engine(cfg=cfg)
        fake_task = asyncio.ensure_future(asyncio.sleep(999))
        eng._entry_tasks["SPY"] = fake_task
        eng._review_tasks["P1"] = asyncio.ensure_future(asyncio.sleep(999))

        off_cfg = replace(cfg, mode=DpofMode.OFF)
        eng._config_loader = lambda: off_cfg
        await eng.run_cycle(bot=None)

        assert eng._entry_tasks == {}
        assert eng._review_tasks == {}
        assert fake_task.cancelled()

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# 2. config_invalid / no keys / market closed (unaffected by the judgment
# rewrite -- these gates run before any decider/candidate work).
# ─────────────────────────────────────────────────────────────────────────

def test_config_invalid_skips_and_alerts_once(cfg):
    bad_cfg = replace(cfg, mode=DpofMode.LOG_ONLY, w_gex=0.99)
    sent, sender = _dedupe_alerts()
    clock_box = {"t": 1_000_000.0}
    alerts = DPOFAlerts(sender=sender, clock=lambda: clock_box["t"])

    async def _t():
        eng = _make_engine(cfg=bad_cfg, alerts=alerts, clock=lambda: clock_box["t"])
        r1 = await eng.run_cycle(bot=object())
        assert r1["skipped"] == "config_invalid"
        assert any("weights sum" in p for p in r1["problems"])
        assert len(sent) == 1

        clock_box["t"] += 10
        r2 = await eng.run_cycle(bot=object())
        assert r2["skipped"] == "config_invalid"
        assert len(sent) == 1

    asyncio.run(_t())


def test_log_only_without_alpaca_keys_skips(cfg):
    c = replace(cfg, mode=DpofMode.LOG_ONLY, alpaca_key_id="", alpaca_secret_key="")

    async def _t():
        eng = _make_engine(cfg=c, market_open_fn=lambda: True)
        r = await eng.run_cycle(bot=None)
        assert r == {"skipped": "no_alpaca_keys", "mode": "log_only"}

    asyncio.run(_t())


def test_market_closed_skips(cfg):
    c = replace(cfg, mode=DpofMode.LOG_ONLY, alpaca_key_id="k", alpaca_secret_key="s")

    async def _t():
        eng = _make_engine(cfg=c, market_open_fn=lambda: False)
        r = await eng.run_cycle(bot=None)
        assert r == {"skipped": "market_closed", "mode": "log_only", "stranded": 0}

    asyncio.run(_t())


def test_market_open_fn_raising_treated_as_closed(cfg):
    c = replace(cfg, mode=DpofMode.LOG_ONLY, alpaca_key_id="k", alpaca_secret_key="s")

    def _boom():
        raise RuntimeError("calendar unavailable")

    async def _t():
        eng = _make_engine(cfg=c, market_open_fn=_boom)
        r = await eng.run_cycle(bot=None)
        assert r["skipped"] == "market_closed"

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# 3. _dpof_equity (unaffected by the judgment rewrite)
# ─────────────────────────────────────────────────────────────────────────

def test_dpof_equity_capital_base_zero_passes_through_unchanged(cfg):
    from agent.dpof.models import RiskState

    c = replace(cfg, capital_base=0.0)
    out = engine_module._dpof_equity(15_000.0, RiskState(), 500.0, c)
    assert out == 15_000.0


def test_dpof_equity_uses_virtual_when_smaller_than_account(cfg):
    from agent.dpof.models import RiskState

    c = replace(cfg, capital_base=20_000.0)
    risk = RiskState(realized_pnl_total=-500.0)
    out = engine_module._dpof_equity(50_000.0, risk, -200.0, c)
    assert out == 19_300.0  # 20000 - 500 - 200, capped at 50000


def test_dpof_equity_capped_at_real_account_equity(cfg):
    from agent.dpof.models import RiskState

    c = replace(cfg, capital_base=20_000.0)
    risk = RiskState(realized_pnl_total=50_000.0)
    out = engine_module._dpof_equity(21_000.0, risk, 0.0, c)
    assert out == 21_000.0  # virtual (70000) capped at real account equity


def test_dpof_equity_none_account_equity_stays_none(cfg):
    from agent.dpof.models import RiskState

    c = replace(cfg, capital_base=20_000.0)
    assert engine_module._dpof_equity(None, RiskState(), 0.0, c) is None


# ─────────────────────────────────────────────────────────────────────────
# 4. Entry flow: happy path -- fake decider says "enter" -> Jev pass -> sized order
# ─────────────────────────────────────────────────────────────────────────

def test_entry_happy_path_launches_then_applies_on_next_cycle(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)

    c = replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",),
        alpaca_key_id="k", alpaca_secret_key="s", allow_verticals=False,
    )
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=True))

    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=0.8, qty=3, thesis="strong flow", invalidation="composite reverses"))

    submitted = {}

    async def open_position_paper(mgr, choice, qty, signal, jev, now_epoch, judgment):
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

        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        r1 = await eng.run_cycle(bot=object())
        assert "SPY" in eng._entry_tasks  # launched, not yet applied
        assert orders.open_calls == []  # cycle never awaited the decider inline

        # Let the (already-resolved fake) task actually finish.
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        r2 = await eng.run_cycle(bot=object())

        assert len(orders.open_calls) == 1
        assert submitted["qty"] == 3
        assert submitted["judgment"]["thesis"] == "strong flow"
        assert submitted["judgment"]["entry_decision"]["candidate_key"] == "C1"
        assert eng._ledger.get("P1") is not None
        assert eng._entry_tasks == {}

    asyncio.run(_t())


def test_entry_counters_survive_the_cycle_end_save(cfg, monkeypatch):
    """DPOF-1 (S211): _apply_entry_outcome bumps entries_today on the LEDGER's risk and saves it, but
    run_cycle had rolled its own RiskState at step 7, before the entry pipeline. The cycle-end
    save_risk(risk) then wrote the pre-entry counters back, so entries_today read 0 all day (2026-09-29:
    risk_state.json showed entries_today=0, entries_by_symbol={} with realized_pnl_today=130). Jev's
    sizing context and the decider's context both consume these counters."""
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)

    c = replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",),
        alpaca_key_id="k", alpaca_secret_key="s", allow_verticals=False,
    )
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=True))
    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=0.8, qty=3, thesis="strong flow", invalidation="composite reverses"))

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

    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        orders.open_position_return = open_position_paper
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=object())
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=object())   # the entry is applied in this cycle
        assert len(orders.open_calls) == 1

        # in memory ...
        assert eng._ledger.risk.entries_today == 1
        assert eng._ledger.risk.entries_by_symbol == {"SPY": 1}
        # ... and what a restart / the operator would read back from disk
        fresh = DPOFLedger(c.data_dir)
        await fresh.load()
        assert fresh.risk.entries_today == 1
        assert fresh.risk.entries_by_symbol == {"SPY": 1}

    asyncio.run(_t())


def test_entry_pass_sets_cooldown_and_recheck(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)

    decider = FakeDecider(entry_result=FakeEntryDecision(action="pass", recheck_minutes=10))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        assert orders.open_calls == []
        key = "SPY:bullish"
        assert key in eng._entry_cooldowns
        rec = eng._entry_cooldowns[key]
        assert rec["recheck_at"] == pytest.approx(clock_box["t"] + 10 * 60)

        # Cooldown active -> no relaunch on the very next cycle.
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)
        assert "SPY" not in eng._entry_tasks

    asyncio.run(_t())


def test_step_8c_reconciles_on_tenth_cycle_and_while_qty_mismatch_pending(cfg, monkeypatch):
    """Drives the real run_cycle through step 8c (followups frame §3.3):
    reconcile runs every 10th cycle, and on EVERY cycle while the order
    manager reports a pending quantity mismatch, with the cycle's epoch."""
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, _ = _make_chain("SPY", today)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    decider = FakeDecider(entry_result=FakeEntryDecision(action="pass", recheck_minutes=10))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)                      # cycle 1, nothing pending
        assert orders.reconcile_calls == 0

        orders.pending_qty_mismatch = True
        clock_box["t"] += 30
        await eng.run_cycle(bot=None)                      # cycle 2, mismatch pending
        assert orders.reconcile_calls == 1
        assert orders.reconcile_now_epochs == [clock_box["t"]]

        orders.pending_qty_mismatch = False
        clock_box["t"] += 30
        await eng.run_cycle(bot=None)                      # cycle 3, nothing pending
        assert orders.reconcile_calls == 1

        eng._cycle_count = 9
        clock_box["t"] += 30
        await eng.run_cycle(bot=None)                      # cycle 10
        assert orders.reconcile_calls == 2
        assert orders.reconcile_now_epochs[-1] == clock_box["t"]

    asyncio.run(_t())


def test_entry_stale_decision_discarded(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", decision_max_age_seconds=5)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)

    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=0.9, qty=1))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += c.decision_max_age_seconds + 30  # well past staleness
        await eng.run_cycle(bot=None)

        assert orders.open_calls == []

    asyncio.run(_t())


def test_entry_invalid_candidate_key_discarded(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)

    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C99", conviction=0.9, qty=1))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        assert orders.open_calls == []

    asyncio.run(_t())


def test_entry_jev_veto_blocks(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="veto", called=True, allow=False, veto_reasons=["size_reckless"]))

    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=0.9, qty=5))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        assert orders.open_calls == []
        assert eng._last_jev_verdict is not None
        assert eng._last_jev_verdict.allow is False

    asyncio.run(_t())


def test_entry_quote_moved_discards_on_slippage(cfg, monkeypatch):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today, bid=1.98, ask=2.02)  # mid ~= 2.00
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", entry_max_slippage_pct=0.05)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    # Re-quote moves the mid from ~2.00 to 3.00 -- a 50% slip, well past the 5% cap.
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=2.98, ask=3.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="off", called=False, allow=True))

    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=0.9, qty=1))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        assert orders.open_calls == []

    asyncio.run(_t())


def test_entry_solvency_clamps_qty_no_hard_caps(cfg, monkeypatch):
    """Operator directive (no-hard-caps amendment): the decider's qty is
    honored up to SOLVENCY only -- no max_contracts/premium-pct cap left."""
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today, bid=1.98, ask=2.02)  # mid ~2.00 -> $200/contract
    c = replace(
        cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",),
        alpaca_key_id="k", alpaca_secret_key="s", capital_base=0.0,
    )
    # Account equity only $500 -> solvency ceiling is floor(500/200) = 2 contracts,
    # even though the decider asks for 100.
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, account={"equity": 500.0, "options_trading_level": 2})
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    _patch_jev_entry_check(monkeypatch, JevVerdict(mode="off", called=False, allow=True))

    decider = FakeDecider(entry_result=FakeEntryDecision(action="enter", candidate_key="C1", conviction=1.0, qty=100))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)

        async def open_pos(mgr, choice, qty, signal, jev, now_epoch, judgment):
            return PositionRecord(id="P1", underlying=choice.underlying, direction=choice.direction.value, structure="single", status="open", qty_original=qty, qty_open=qty, mode="paper")

        orders.open_position_return = open_pos
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        assert len(orders.open_calls) == 1
        assert orders.open_calls[0][1] == 2  # clamped to solvency, not the requested 100

    asyncio.run(_t())


def test_entry_zero_solvency_skips_entirely(cfg, monkeypatch):
    """Solvency pre-check (operator directive): DPOF never even bothers
    deliberating an underlying it can't afford one contract of."""
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", capital_base=0.0)
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, account={"equity": 0.0, "options_trading_level": 2})
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    decider = FakeDecider()

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: base_epoch)

        await eng.run_cycle(bot=None)
        assert "SPY" not in eng._entry_tasks
        assert decider.entry_calls == []

    asyncio.run(_t())


def test_entry_no_position_count_cap_blocks(cfg, monkeypatch):
    """Operator directive (no-hard-caps amendment): a huge existing-position
    count doesn't itself block a NEW underlying's deliberation -- only an
    existing position on the SAME underlying does (checked separately)."""
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _make_chain("SPY", today)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY", "QQQ"), alpaca_key_id="k", alpaca_secret_key="s")
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    _patch_entry_candidate_cls(monkeypatch)
    decider = FakeDecider()

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        # Pre-existing positions on OTHER underlyings/symbols -- would have
        # tripped the old max_concurrent_positions cap; must not block SPY now.
        for i in range(5):
            await ledger.upsert(_position(f"OLD{i}", "TLT", f"TLTTESTC0050000{i}"))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: base_epoch)

        await eng.run_cycle(bot=None)
        assert "SPY" in eng._entry_tasks

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# 5. Position reviews
# ─────────────────────────────────────────────────────────────────────────

def test_review_due_trigger_launches_and_hold_reschedules(cfg):
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=FakeReviewDecision(action="hold", next_review_minutes=7))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        assert "P1" in eng._review_tasks
        await eng._review_tasks["P1"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        pos = eng._ledger.get("P1")
        assert pos.review_count == 1
        assert pos.review_failures == 0
        assert pos.next_review_at == pytest.approx(clock_box["t"] + 7 * 60)
        assert orders.close_calls == []

    asyncio.run(_t())


def test_review_move_trigger_fires_between_due_reviews(cfg):
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", review_trigger_move_pct=0.10)
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=2.48, ask=2.52)}  # mark ~2.50, +25% vs last_review_mark=2.0
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=FakeReviewDecision(action="hold"))

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, next_review_at=base_epoch + 3600, last_review_mark=2.0))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: base_epoch)

        await eng.run_cycle(bot=None)
        assert "P1" in eng._review_tasks
        ctx, position_dict = decider.review_calls[0]
        assert "mark_move" in position_dict["trigger"]

    asyncio.run(_t())


def test_review_apply_trim(cfg):
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=2.98, ask=3.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=FakeReviewDecision(action="trim", trim_fraction=0.5, thesis_status="weakening", reason="momentum fading"))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, qty_open=4, qty_original=4, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._review_tasks["P1"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        assert len(orders.close_calls) == 1
        pos_id, qty, reason, urgent, _now = orders.close_calls[0]
        assert pos_id == "P1"
        assert qty == 2  # ceil(4 * 0.5)
        assert reason == "decider_trim"
        assert urgent is False

    asyncio.run(_t())


def test_review_trim_consuming_whole_position_becomes_exit(cfg):
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=2.98, ask=3.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=FakeReviewDecision(action="trim", trim_fraction=0.9))  # ceil(3*0.9)=3 >= qty_open
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, qty_open=3, qty_original=3, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._review_tasks["P1"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        pos_id, qty, reason, urgent, _now = orders.close_calls[0]
        assert qty == 3
        assert reason == "decider_exit"

    asyncio.run(_t())


def test_review_apply_exit_urgent_flag_propagates(cfg):
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=0.98, ask=1.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=FakeReviewDecision(action="exit", urgency="urgent", thesis_status="broken", reason="thesis invalidated"))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, qty_open=2, qty_original=2, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._review_tasks["P1"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        pos_id, qty, reason, urgent, _now = orders.close_calls[0]
        assert qty == 2
        assert urgent is True

    asyncio.run(_t())


def test_review_order_safety_no_apply_while_exit_batch_working(cfg):
    """The rule: a harvested review decision is discarded (never applied) if
    the position moved to 'closing' (an exit batch already working, e.g. the
    EOD flatten fired first this cycle) since the review was launched."""
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=FakeReviewDecision(action="trim", trim_fraction=0.5))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, qty_open=4, qty_original=4, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)  # launches the review
        await eng._review_tasks["P1"]

        # Simulate another exit batch starting on this position before harvest
        # (e.g. a concurrent EOD flatten submission) -- status flips to "closing".
        pos = eng._ledger.get("P1")
        await eng._ledger.upsert(replace(pos, status="closing"))

        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        # The harvested trim must NOT have been applied on top of the working batch.
        assert orders.close_calls == []
        final = eng._ledger.get("P1")
        assert final.review_count == 0  # last_review/next_review_at untouched too

    asyncio.run(_t())


def test_review_qty_changed_since_launch_discards(cfg):
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=FakeReviewDecision(action="trim", trim_fraction=0.5))
    clock_box = {"t": base_epoch}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, qty_open=4, qty_original=4, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        await eng._review_tasks["P1"]

        # qty_open changed (e.g. a partial fill on a prior exit) since launch.
        pos = eng._ledger.get("P1")
        await eng._ledger.upsert(replace(pos, qty_open=3))

        clock_box["t"] += 1
        await eng.run_cycle(bot=None)
        assert orders.close_calls == []

    asyncio.run(_t())


def test_review_jev_escalation_triggers_review_never_trades(cfg):
    """Jev escalation only pulls the next review forward -- it must never
    itself close/trim/enter anything."""
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", jev_mode=JevMode.VETO, jev_api_key="test-key")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=FakeReviewDecision(action="hold"))

    escalations = []

    async def sentinel(cfg_, client, position_ctx):
        from agent.dpof.jev import JevEscalation
        escalations.append(position_ctx)
        return JevEscalation(escalate=True, reasons=["thesis_danger"])

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        # next_review_at far in the future -- only the Jev escalation should trigger a review.
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, next_review_at=base_epoch + 3600))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, jev_sentinel=sentinel, clock=lambda: base_epoch)

        await eng.run_cycle(bot=None)
        assert len(escalations) == 1
        assert "P1" in eng._review_tasks  # escalation ONLY scheduled a review
        assert orders.close_calls == []  # never trades by itself

    asyncio.run(_t())


def test_no_jev_call_when_effective_off(cfg):
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", jev_mode=JevMode.OFF)
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(review_result=FakeReviewDecision(action="hold"))

    async def sentinel_must_not_be_called(cfg_, client, position_ctx):
        raise AssertionError("jev sentinel must never be called when jev_effective_mode is OFF")

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, jev_sentinel=sentinel_must_not_be_called, clock=lambda: base_epoch)

        r = await eng.run_cycle(bot=None)
        assert "position_reviews" not in r["errors"]

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# 6. EOD flatten (the one kept rail)
# ─────────────────────────────────────────────────────────────────────────

def test_eod_flatten_cancels_in_flight_review_and_flattens(cfg):
    now_et = datetime(2026, 9, 24, 15, 55, 0, tzinfo=ET)  # past cfg.flatten_et default 15:50
    base_epoch = now_et.timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(block_forever=True)  # would hang forever if ever awaited

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, qty_open=2, qty_original=2))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: base_epoch)

        r = await asyncio.wait_for(eng.run_cycle(bot=None), timeout=5.0)

        assert len(orders.close_calls) == 1
        pos_id, qty, reason, urgent, _now = orders.close_calls[0]
        assert reason == "eod_flatten"
        assert urgent is True
        assert "P1" not in eng._review_tasks  # never even launched -- flatten wins outright
        decider.release()

    asyncio.run(_t())


def test_eod_flatten_cancels_already_in_flight_review_task(cfg):
    """A review already launched a prior cycle (before flatten_et) must be
    CANCELLED, not left to complete and get applied later, once flatten_et
    arrives."""
    before_flatten = datetime(2026, 9, 24, 15, 40, 0, tzinfo=ET)
    after_flatten = datetime(2026, 9, 24, 15, 55, 0, tzinfo=ET)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(block_forever=True)
    clock_box = {"t": before_flatten.timestamp()}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, qty_open=2, qty_original=2, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        task = eng._review_tasks.get("P1")
        assert task is not None and not task.done()

        clock_box["t"] = after_flatten.timestamp()
        await asyncio.wait_for(eng.run_cycle(bot=None), timeout=5.0)

        assert task.cancelled() or task.done()
        assert "P1" not in eng._review_tasks
        assert len(orders.close_calls) == 1
        assert orders.close_calls[0][2] == "eod_flatten"
        decider.release()

    asyncio.run(_t())


def test_eod_flatten_force_cancels_pending_entry_not_close_position(cfg):
    """Pass-2 order-safety finding: a still-PENDING entry (a working, unfilled
    limit order -- no fill yet exists to "close") was previously invisible to
    the flatten rail entirely, because the review loop gated the whole
    evaluate_exit() check on status=="open". At/after flatten_et a PENDING
    position must be force-cancelled via OrderManager.flatten_entry(), never
    routed through close_position() (which requires status=="open" and would
    refuse/warn), so it can never keep working -- or fill -- past 15:50 and
    sit undiscovered as a 1DTE contract until the next session."""
    now_et = datetime(2026, 9, 24, 15, 55, 0, tzinfo=ET)  # past cfg.flatten_et default 15:50
    base_epoch = now_et.timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(block_forever=True)  # would hang forever if ever awaited

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        pending = _position(
            "P1", "SPY", leg_symbol, status="pending", qty_original=2, qty_open=0,
            entry_price=None, entry_limit_price=1.90, entry_client_order_id="DPOF-EN-1",
            entry_client_order_ids=["DPOF-EN-1"],
        )
        await ledger.upsert(pending)
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: base_epoch)

        r = await asyncio.wait_for(eng.run_cycle(bot=None), timeout=5.0)

        assert orders.close_calls == []  # never treated as a fillable "open" position
        assert len(orders.flatten_entry_calls) == 1
        pos_id, reason, _now = orders.flatten_entry_calls[0]
        assert pos_id == "P1"
        assert reason == "eod_flatten"
        assert "P1" not in eng._review_tasks  # never even launched a review for a pending entry
        assert any(e.get("action") == "flatten_entry" for e in r["exits"])
        decider.release()

    asyncio.run(_t())


def test_eod_flatten_escalates_working_closing_batch_to_urgent(cfg):
    """A position already 'closing' (an exit batch launched by an earlier
    review) must NEVER get a second, conflicting close submitted by the
    flatten rail -- but past flatten_et it should be escalated to
    exit_urgent=True so orders.py's own reprice-exhaustion ladder is allowed
    to go marketable/market once its budget runs out, instead of silently
    staying non-urgent past the close deadline."""
    now_et = datetime(2026, 9, 24, 15, 55, 0, tzinfo=ET)
    base_epoch = now_et.timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(block_forever=True)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        closing = _position(
            "P1", "SPY", leg_symbol, status="closing", qty_original=2, qty_open=2,
            exit_target_qty=2, exit_urgent=False,
        )
        await ledger.upsert(closing)
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: base_epoch)

        r = await asyncio.wait_for(eng.run_cycle(bot=None), timeout=5.0)

        assert orders.close_calls == []            # no duplicate close submitted
        assert orders.flatten_entry_calls == []     # not treated as a pending entry either
        final = eng._ledger.get("P1")
        assert final.status == "closing"
        assert final.exit_urgent is True            # escalated
        assert any(e.get("action") == "flatten_escalate_closing" for e in r["exits"])
        decider.release()

    asyncio.run(_t())


def test_eod_flatten_closing_already_urgent_is_a_noop(cfg):
    """No redundant ledger write / exits_log entry once a closing batch is
    already urgent."""
    now_et = datetime(2026, 9, 24, 15, 55, 0, tzinfo=ET)
    base_epoch = now_et.timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    decider = FakeDecider(block_forever=True)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        closing = _position(
            "P1", "SPY", leg_symbol, status="closing", qty_original=2, qty_open=2,
            exit_target_qty=2, exit_urgent=True,
        )
        await ledger.upsert(closing)
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: base_epoch)

        r = await asyncio.wait_for(eng.run_cycle(bot=None), timeout=5.0)

        assert orders.close_calls == []
        assert orders.flatten_entry_calls == []
        assert not any(e.get("action") == "flatten_escalate_closing" for e in r["exits"])
        decider.release()

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# 7. Model-down / recovery
# ─────────────────────────────────────────────────────────────────────────

def test_model_down_when_decider_none_holds_and_alerts(cfg, monkeypatch):
    """Whether agent.dpof.decider is importable or not, engine.py must
    degrade the SAME way once construction fails -- forced here by making
    DPOFDecider.__init__ itself raise, so this stays hermetic (no real
    provider client, no network) regardless of whether D1's module exists
    on disk in this checkout."""
    import agent.dpof.decider as decider_mod

    def _boom(cfg):
        raise RuntimeError("decider construction unavailable for test")

    monkeypatch.setattr(decider_mod, "DPOFDecider", _boom)

    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    sent, sender = _dedupe_alerts()
    alerts = DPOFAlerts(sender=sender, clock=lambda: base_epoch)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, ledger=ledger, alerts=alerts, clock=lambda: base_epoch)
        assert eng._decider is None

        await eng.run_cycle(bot=object())

        assert eng._decider is None  # construction kept failing -- never silently swapped in
        pos = eng._ledger.get("P1")
        assert pos.review_failures == 1
        assert pos.next_review_at == pytest.approx(base_epoch + 60.0)
        model_down = [t for t in sent if "decision model unavailable" in t]
        assert len(model_down) == 1
        assert "1 open position" in model_down[0]

    asyncio.run(_t())


def test_model_down_alert_dedupes_and_recovery_alert_fires_once(cfg):
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", model_down_alert_minutes=15)
    leg_symbol = "SPYTESTC00500000"
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _neutral_snapshot(now, "SPY"))
    sent, sender = _dedupe_alerts()
    clock_box = {"t": base_epoch}
    alerts = DPOFAlerts(sender=sender, clock=lambda: clock_box["t"])
    decider = FakeDecider(review_result=RuntimeError("boom"))  # ok=False, unavailable=False (generic error)

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        await ledger.upsert(_position("P1", "SPY", leg_symbol, entry_price=2.0, next_review_at=None))
        orders = FakeOrderManager(ledger=ledger)
        eng = _make_engine(cfg=c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, alerts=alerts, clock=lambda: clock_box["t"])

        # Force "unavailable" classification for a generic error by pointing the
        # engine's own sentinel exception class at the real DecisionUnavailable.
        from agent.dpof.decider import DecisionUnavailable
        decider.review_result = DecisionUnavailable("both providers down")
        eng._decision_unavailable_exc = DecisionUnavailable

        await eng.run_cycle(bot=object())
        await eng._review_tasks["P1"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=object())

        model_down = [t for t in sent if "unavailable" in t]
        assert len(model_down) == 1

        # A second review-failure within the dedupe window does not re-alert.
        pos = eng._ledger.get("P1")
        clock_box["t"] = pos.next_review_at
        await eng.run_cycle(bot=object())
        await eng._review_tasks["P1"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=object())
        assert len([t for t in sent if "unavailable" in t]) == 1

        # Now the decider recovers.
        decider.review_result = FakeReviewDecision(action="hold")
        pos = eng._ledger.get("P1")
        clock_box["t"] = pos.next_review_at
        await eng.run_cycle(bot=object())
        await eng._review_tasks["P1"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=object())

        recovered = [t for t in sent if "back online" in t]
        assert len(recovered) == 1

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# 8. Shutdown / cancellation
# ─────────────────────────────────────────────────────────────────────────

def test_shutdown_cancels_in_flight_tasks_and_closes_collaborators():
    async def _t():
        decider = FakeDecider()
        jev = FakeJevClient()
        feed = FakeFeed(lambda now: _neutral_snapshot(now))
        eng = DPOFEngine(memory=FakeMemory(), decider=decider, jev=jev, feed=feed)

        t1 = asyncio.ensure_future(asyncio.sleep(999))
        t2 = asyncio.ensure_future(asyncio.sleep(999))
        eng._entry_tasks["SPY"] = t1
        eng._review_tasks["P1"] = t2

        await eng.shutdown()

        assert t1.cancelled()
        assert t2.cancelled()
        assert decider.closed is True
        assert jev.closed is True
        assert feed.closed is True

        # Idempotent.
        await eng.shutdown()

    asyncio.run(_t())


def test_shutdown_dpof_noop_when_singleton_never_created(monkeypatch):
    monkeypatch.setattr(engine_module, "_engine_singleton", None)
    asyncio.run(engine_module.shutdown_dpof())  # must not raise


# ─────────────────────────────────────────────────────────────────────────
# 9. Misc engine plumbing (singleton, job interval, operator halt/resume)
# ─────────────────────────────────────────────────────────────────────────

def test_job_interval_and_timeout_clamped():
    assert 10 <= engine_module.DPOF_JOB_INTERVAL <= 300
    assert engine_module.DPOF_JOB_TIMEOUT == max(8, engine_module.DPOF_JOB_INTERVAL - 5)


def test_get_engine_singleton_is_stable():
    a = engine_module.get_engine()
    b = engine_module.get_engine()
    assert a is b


def test_run_dpof_cycle_extracts_bot_from_context(monkeypatch):
    calls = []

    class FakeEngine:
        async def run_cycle(self, bot=None):
            calls.append(bot)
            return {}

    monkeypatch.setattr(engine_module, "get_engine", lambda: FakeEngine())

    class Ctx:
        bot = "the-bot"

    asyncio.run(engine_module.run_dpof_cycle(Ctx()))
    assert calls == ["the-bot"]


def test_operator_halt_mid_cycle_is_not_overwritten(cfg):
    async def _t():
        eng = _make_engine(cfg=replace(cfg, mode=DpofMode.OFF))
        msg = await eng.operator_halt("manual test")
        assert "manual test" in msg
        assert eng._ledger.risk.hard_halted is True

    asyncio.run(_t())
