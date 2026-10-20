"""Integration-flavoured tests for the DPOF judgment layer (engine.py).

Complements test_engine.py: where that file fakes every collaborator for
isolation, these tests wire engine.py against the REAL landed D1/D4/D2
modules (agent.dpof.decider.EntryCandidate/EntryDecision/ReviewDecision,
agent.dpof.contracts.choose_candidates/target_expiries,
agent.dpof.signals.position_alignment, agent.dpof.jev.jev_position_check)
wherever they're importable, to catch field-name/shape drift between
builders that a fully-faked test would hide. The decider's OWN provider
calls are still faked (no network) — only its data contracts are real.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from agent.dpof.config import DpofMode, JevMode
from agent.dpof.engine import DPOFEngine
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import Direction, MarketSnapshot, OptionQuote, PillarReading, PositionRecord, UnderlyingSnapshot
from agent.dpof.sizing import max_affordable_qty

from tests.dpof.test_engine import (
    FakeBroker,
    FakeFeed,
    FakeJevClient,
    FakeMemory,
    FakeOrderManager,
    _base_dt,
    _bullish_snapshot,
    _dedupe_alerts,
    _make_chain,
    _neutral_snapshot,
    _position,
)

ET = ZoneInfo("America/New_York")

decider_mod = pytest.importorskip("agent.dpof.decider", reason="D1's decider.py not landed yet")
contracts_mod = pytest.importorskip("agent.dpof.contracts", reason="D4's contracts.py not landed yet")
signals_mod = pytest.importorskip("agent.dpof.signals", reason="D4's signals.py not landed yet")


class RealShapeDecider:
    """Returns REAL agent.dpof.decider.EntryDecision/ReviewDecision instances
    (never calls a real provider) so engine.py's field reads
    (getattr(decision, "qty", ...) etc.) are exercised against D1's actual
    dataclass shape, not a hand-rolled fake."""

    def __init__(self, entry=None, review=None, block_forever=False):
        self.cfg = None
        self.entry = entry
        self.review = review
        self.entry_calls: list[tuple] = []
        self.review_calls: list[tuple] = []
        self._event = asyncio.Event() if block_forever else None
        self.closed = False

    def update_config(self, cfg):
        self.cfg = cfg

    async def decide_entry(self, ctx, candidates):
        self.entry_calls.append((ctx, candidates))
        if self._event is not None:
            await self._event.wait()
        if isinstance(self.entry, Exception):
            raise self.entry
        return self.entry if self.entry is not None else decider_mod.EntryDecision(action="pass")

    async def review_position(self, ctx, position):
        self.review_calls.append((ctx, position))
        if self._event is not None:
            await self._event.wait()
        if isinstance(self.review, Exception):
            raise self.review
        return self.review if self.review is not None else decider_mod.ReviewDecision(action="hold")

    def status(self) -> dict:
        return {"providers": {}, "calls_today": 0, "est_cost_today_usd": 0.0}

    async def close(self):
        self.closed = True

    def release(self):
        if self._event is not None:
            self._event.set()


def _engine(cfg, **kw) -> DPOFEngine:
    return DPOFEngine(
        memory=FakeMemory(),
        jev=kw.pop("jev", None) or FakeJevClient(),
        clock=kw.pop("clock", None) or (lambda: _base_dt().timestamp()),
        market_open_fn=kw.pop("market_open_fn", None) or (lambda: True),
        config_loader=lambda: cfg,
        **kw,
    )


# ─────────────────────────────────────────────────────────────────────────
# Real EntryCandidate/EntryDecision/ReviewDecision shape compatibility
# ─────────────────────────────────────────────────────────────────────────

def test_entry_candidate_built_with_real_dataclass_and_correct_solvency_field(cfg):
    """engine._build_entry_candidate must construct the REAL
    agent.dpof.decider.EntryCandidate (no TypeError on field names) with
    max_affordable_qty computed via sizing.max_affordable_qty."""
    from agent.dpof.models import ContractChoice, OrderLeg

    choice = ContractChoice(
        structure="single",
        legs=[OrderLeg(symbol="SPY261002C00580000", side="buy", ratio_qty=1, position_intent="buy_to_open")],
        underlying="SPY", direction=Direction.BULLISH, expiration="2026-10-02", dte=1,
        est_debit=2.00, est_bid=1.98, est_ask=2.02, long_delta=0.5,
    )
    quote = OptionQuote(
        symbol="SPY261002C00580000", underlying="SPY", expiration="2026-10-02", strike=580.0, right="call",
        bid=1.98, ask=2.02, delta=0.5, theta=-0.05, iv=0.22, open_interest=1200,
    )
    eng = _engine(cfg)
    capital_available = 2000.0
    cand = eng._build_entry_candidate(
        "C1", choice, {quote.symbol: quote}, _base_dt().date(), capital_available, decider_mod.EntryCandidate,
    )
    assert isinstance(cand, decider_mod.EntryCandidate)
    assert cand.key == "C1"
    assert cand.underlying == "SPY"
    assert cand.max_affordable_qty == max_affordable_qty(capital_available, choice.est_debit)
    assert cand.theta == -0.05
    assert cand.iv == 0.22
    assert cand.open_interest == 1200
    assert cand.unit_premium == pytest.approx(200.0)


def _real_1dte_chain(symbol, today, cfg, bid=1.98, ask=2.02):
    """A chain quote at the REAL target_expiries()[0] date -- needed when a
    test wires the REAL contracts.choose_candidates/expiry_fetch_window in,
    since those only look at the 1DTE-session window, not an arbitrary
    +N-calendar-days offset (test_engine.py's _make_chain helper picks +3
    calendar days, which is fine for the FAKE candidate_fn but not the real
    1DTE-pinned one)."""
    expiry = contracts_mod.target_expiries(today, cfg)[0]
    q = OptionQuote(
        symbol=f"{symbol}TESTC00500000", underlying=symbol, expiration=expiry.isoformat(), strike=500.0, right="call",
        bid=bid, ask=ask, delta=0.5, theta=-0.05, iv=0.2, open_interest=1000, volume=100,
    )
    return [q], q.symbol


def test_entry_decision_qty_field_drives_sizing_end_to_end(cfg, monkeypatch):
    """Full path with the REAL EntryCandidate/EntryDecision: decider.qty is
    clamped fresh by clamp_to_solvency at apply time, not size_fraction."""
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _real_1dte_chain("SPY", today, cfg)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0, right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))

    import agent.dpof.engine as engine_module

    async def _jev_allow(signal, snap, now_et, cfg_, client, sizing_ctx=None):
        from agent.dpof.models import JevVerdict
        return JevVerdict(mode="off", called=False, allow=True)

    monkeypatch.setattr(engine_module, "jev_entry_check", _jev_allow)

    decider = RealShapeDecider(entry=decider_mod.EntryDecision(
        action="enter", candidate_key="C1", conviction=0.9, qty=4, thesis="t", invalidation="i",
    ))
    clock_box = {"t": base_epoch}
    submitted = {}

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)

        async def open_pos(mgr, choice, qty, signal, jev, now_epoch, judgment):
            submitted["qty"] = qty
            return PositionRecord(id="P1", underlying=choice.underlying, direction=choice.direction.value, structure="single", status="open", qty_original=qty, qty_open=qty, mode="paper")

        orders.open_position_return = open_pos
        eng = _engine(c, broker=broker, feed=feed, orders=orders, ledger=ledger, decider=decider, candidate_fn=engine_module._default_candidate_fn, expiry_fetch_window_fn=engine_module._default_expiry_fetch_window_fn, position_alignment_fn=engine_module._default_position_alignment_fn, clock=lambda: clock_box["t"])

        await eng.run_cycle(bot=None)
        assert "SPY" in eng._entry_tasks
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

        assert submitted.get("qty") == 4  # well within solvency at $20k equity / $200 contract

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# Cycle never awaits the decider
# ─────────────────────────────────────────────────────────────────────────

def test_cycle_returns_promptly_even_with_a_permanently_blocked_decider(cfg):
    base_epoch = _base_dt().timestamp()
    today = _base_dt().date()
    chain, leg_symbol = _real_1dte_chain("SPY", today, cfg)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s")
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain)
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    decider = RealShapeDecider(block_forever=True)

    import agent.dpof.engine as engine_module

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)
        eng = _engine(
            c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: base_epoch,
            candidate_fn=engine_module._default_candidate_fn, expiry_fetch_window_fn=engine_module._default_expiry_fetch_window_fn,
        )

        # If run_cycle ever awaited decide_entry inline, this would hang until
        # the wait_for timeout fires and raises -- it must complete well inside it.
        result = await asyncio.wait_for(eng.run_cycle(bot=None), timeout=3.0)
        assert "SPY" in eng._entry_tasks
        assert not eng._entry_tasks["SPY"].done()
        decider.release()
        await eng._entry_tasks["SPY"]

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# Review triggers (unit-level, real position_alignment)
# ─────────────────────────────────────────────────────────────────────────

def _snap_with_underlying(now_epoch, composite_sign_against: bool, symbol="SPY") -> MarketSnapshot:
    """A snapshot whose pillars compose to a NEGATIVE composite (bearish),
    used to test 'alignment_against' for a held BULLISH position via the
    real signals.position_alignment + compute_composite pipeline."""
    if composite_sign_against:
        pillars = {
            "nofa": PillarReading(name="nofa", value=-0.6, available=True, source="test"),
            "premium": PillarReading(name="premium", value=-0.5, available=True, source="test"),
            "darkpool": PillarReading(name="darkpool", value=-0.5, available=True, source="test"),
        }
    else:
        pillars = {
            "nofa": PillarReading(name="nofa", value=0.6, available=True, source="test"),
            "premium": PillarReading(name="premium", value=0.5, available=True, source="test"),
            "darkpool": PillarReading(name="darkpool", value=0.5, available=True, source="test"),
        }
    u = UnderlyingSnapshot(symbol=symbol, pillars=pillars, last_price=100.0, vwap=100.0, ret_15m=0.0)
    return MarketSnapshot(asof=now_epoch, underlyings={symbol: u}, vix=16.0, vix_age_seconds=30.0, data_healthy=True)


def test_review_trigger_alignment_against_uses_real_position_alignment(cfg):
    eng = _engine(cfg)
    snap = _snap_with_underlying(_base_dt().timestamp(), composite_sign_against=True)
    underlyings_ctx = eng._build_underlyings_ctx(snap, cfg, signal=None)
    pos = _position("P1", "SPY", "SPYTESTC00500000", direction="bullish", next_review_at=_base_dt().timestamp() + 999999)
    triggers = eng._review_triggers(pos, mark=2.0, now=_base_dt().timestamp(), jev_escalated=False, snap=snap, underlyings_ctx=underlyings_ctx, cfg=cfg)
    assert "alignment_against" in triggers
    assert "due" not in triggers


def test_review_trigger_breaking_news(cfg):
    eng = _engine(cfg)
    snap = _bullish_snapshot(_base_dt().timestamp(), "SPY", risk_flags=["breaking_news"])
    underlyings_ctx = eng._build_underlyings_ctx(snap, cfg, signal=None)
    pos = _position("P1", "SPY", "SPYTESTC00500000", next_review_at=_base_dt().timestamp() + 999999)
    triggers = eng._review_triggers(pos, mark=2.0, now=_base_dt().timestamp(), jev_escalated=False, snap=snap, underlyings_ctx=underlyings_ctx, cfg=cfg)
    assert "breaking_news" in triggers


def test_review_trigger_jev_escalation(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    underlyings_ctx = eng._build_underlyings_ctx(snap, cfg, signal=None)
    pos = _position("P1", "SPY", "SPYTESTC00500000", next_review_at=_base_dt().timestamp() + 999999)
    triggers = eng._review_triggers(pos, mark=2.0, now=_base_dt().timestamp(), jev_escalated=True, snap=snap, underlyings_ctx=underlyings_ctx, cfg=cfg)
    assert triggers == ["jev_escalation"]


def test_review_trigger_none_when_nothing_fires(cfg):
    eng = _engine(cfg)
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")
    underlyings_ctx = eng._build_underlyings_ctx(snap, cfg, signal=None)
    now = _base_dt().timestamp()
    pos = _position("P1", "SPY", "SPYTESTC00500000", next_review_at=now + 3600, last_review_mark=2.0)
    triggers = eng._review_triggers(pos, mark=2.01, now=now, jev_escalated=False, snap=snap, underlyings_ctx=underlyings_ctx, cfg=cfg)
    assert triggers == []


# ─────────────────────────────────────────────────────────────────────────
# Jev sentinel real wiring (D2) — escalation-only, never trades
# ─────────────────────────────────────────────────────────────────────────

def test_jev_sentinel_off_mode_never_calls_real_jev_position_check(cfg):
    """cfg.jev_effective_mode == OFF (no key configured) must short-circuit
    BEFORE the default jev_sentinel (agent.dpof.jev.jev_position_check) is
    even called."""
    from agent.dpof.jev import jev_position_check

    assert cfg.jev_effective_mode == JevMode.OFF  # default fixture has no TypeSafe key

    eng = DPOFEngine(memory=FakeMemory(), jev_sentinel=jev_position_check, clock=lambda: _base_dt().timestamp())
    snap = _neutral_snapshot(_base_dt().timestamp(), "SPY")

    async def _t():
        pos = _position("P1", "SPY", "SPYTESTC00500000")
        escalated = await eng._jev_sentinel_screen(pos, pnl_pct=0.0, now_et=_base_dt(), cfg=cfg, snap=snap, underlyings_ctx={})
        assert escalated is False

    asyncio.run(_t())


# ─────────────────────────────────────────────────────────────────────────
# Solvency, not hard caps, is the ONLY entry-permission size gate
# ─────────────────────────────────────────────────────────────────────────

def test_entry_permission_skips_launch_when_capital_available_nonpositive(cfg):
    """The engine's solvency pre-check (before even fetching a chain) must
    key off dpof_equity minus committed open premium (pending + open),
    not any entry-count cap -- and must skip launching entirely at <= 0."""
    base_epoch = _base_dt().timestamp()
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k", alpaca_secret_key="s", capital_base=0.0)
    # Real account equity equals what's already committed in a pending SPY-adjacent
    # position's premium -- capital_available == 0 -- must not even attempt a chain fetch.
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, account={"equity": 400.0, "options_trading_level": 2})
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    decider = RealShapeDecider()

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        # Pending QQQ position (different underlying) priced at $400 total premium,
        # consuming all of the $400 equity -- capital_available for a NEW SPY
        # deliberation is 0.
        await ledger.upsert(PositionRecord(
            id="PEND", underlying="QQQ", direction="bullish", structure="single",
            legs=[{"symbol": "QQQTESTC00400000", "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}],
            status="pending", qty_original=2, qty_open=0, entry_limit_price=2.0, mode="paper",
        ))
        orders = FakeOrderManager(ledger=ledger)
        eng = _engine(c, broker=broker, feed=feed, orders=orders, decider=decider, ledger=ledger, clock=lambda: base_epoch)

        await eng.run_cycle(bot=None)

        assert "SPY" not in eng._entry_tasks
        assert decider.entry_calls == []
        assert broker.get_option_chain_calls == []  # never even fetched a chain

    asyncio.run(_t())
