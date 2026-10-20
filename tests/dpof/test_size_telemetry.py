"""S212 SZ-1: sizing telemetry on every DPOF entry decision row.

Yesterday's first live day showed 2 of 4 entry intents vetoed by Jev for `size_reckless` with nothing recorded about the
sizes involved. Every entry decision row now carries a `sizing` block: the quantity asked for, the solvency-clamped
quantity, premium as a share of capital, the advisory anchor's quantity at the model's own conviction, and Jev's
`size_reckless` noul next to its threshold. It is telemetry only: it never changes a quantity or a veto and can never
raise into the entry path.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

import agent.dpof.engine as engine_module
from agent.dpof import sizing
from agent.dpof.config import DpofMode
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import JevVerdict, OptionQuote, PositionRecord
from tests.dpof.test_engine import FakeBroker, FakeFeed, FakeOrderManager, _base_dt, _bullish_snapshot
from tests.dpof.test_engine_judgment import RealShapeDecider, _engine, _real_1dte_chain, decider_mod

pytestmark = pytest.mark.filterwarnings("ignore")

NOUL_83 = {"size_reckless": {"type": "noul", "noul": 0.83}}


# ─────────────────────────────────────────────────────────────────────────
# sizing.size_telemetry / jev_size_reckless_noul (pure)
# ─────────────────────────────────────────────────────────────────────────

def _tele(**over):
    kw = dict(qty_requested=8, qty=8, unit_premium=240.0, dpof_equity=20000.0, capital_available=19000.0,
              conviction=0.5, anchor_shares=(0.10, 0.20), jev_answers=NOUL_83, jev_size_veto_threshold=0.70)
    kw.update(over)
    return sizing.size_telemetry(**kw)


def test_full_block_values():
    t = _tele()
    assert t["qty_requested"] == 8 and t["qty"] == 8
    assert t["unit_premium"] == 240.0 and t["premium"] == 1920.0
    assert t["premium_pct_of_capital"] == pytest.approx(0.096)
    assert t["dpof_equity"] == 20000.0 and t["capital_available"] == 19000.0
    assert t["max_affordable_qty"] == 79                                   # floor(19000 / 240)
    assert t["conviction"] == 0.5
    assert t["anchor"] == {"enabled": True, "share_c50": 0.1, "share_c80": 0.2, "share": 0.1, "qty": 8}
    assert t["jev_size_reckless_noul"] == 0.83 and t["jev_size_veto_threshold"] == 0.7


def test_requested_and_clamped_quantities_are_kept_apart():
    t = _tele(qty_requested=200, qty=79)
    assert t["qty_requested"] == 200 and t["qty"] == 79 and t["premium"] == pytest.approx(79 * 240.0)


def test_anchor_quantity_follows_the_models_own_conviction_and_is_bounded_by_solvency():
    hi = _tele(conviction=0.8)["anchor"]
    assert hi["share"] == 0.2 and hi["qty"] == 16                         # 20% of 20000 / 240
    tight = _tele(conviction=0.8, capital_available=720.0)["anchor"]
    assert tight["qty"] == 3                                               # bounded by max_affordable_qty (720 / 240)


def test_anchor_disabled_records_only_the_flag():
    assert _tele(anchor_shares=None)["anchor"] == {"enabled": False}


def test_unknown_inputs_are_none_never_zero():
    t = _tele(qty=None, qty_requested=None, unit_premium=None, dpof_equity=None, capital_available=None,
              conviction=None, jev_answers=None, jev_size_veto_threshold=None)
    for key in ("qty", "qty_requested", "unit_premium", "premium", "premium_pct_of_capital", "dpof_equity",
                "capital_available", "max_affordable_qty", "conviction", "jev_size_reckless_noul",
                "jev_size_veto_threshold"):
        assert t[key] is None, key
    assert t["anchor"]["enabled"] is True and t["anchor"]["share"] is None and t["anchor"]["qty"] is None


@pytest.mark.parametrize("equity", [0.0, -5.0, float("nan")])
def test_nonpositive_equity_gives_no_percentage(equity):
    assert _tele(dpof_equity=equity)["premium_pct_of_capital"] is None


def test_bool_and_nonfinite_quantities_are_unknown():
    t = _tele(qty_requested=True, qty=float("nan"))
    assert t["qty_requested"] is None and t["qty"] is None and t["premium"] is None


def test_zero_quantity_is_a_real_zero_not_unknown():
    t = _tele(qty_requested=5, qty=0)
    assert t["qty"] == 0 and t["premium"] == 0.0 and t["premium_pct_of_capital"] == 0.0


@pytest.mark.parametrize("answers,want", [
    ({"size_reckless": {"type": "noul", "noul": 0.71}}, 0.71),
    ({"size_reckless": {"type": "noul", "noul": 0}}, 0.0),
    ({"size_reckless": {"type": "noul", "noul": float("nan")}}, None),
    ({"size_reckless": {"type": "noul", "noul": "0.9"}}, None),
    ({"size_reckless": {"type": "choice", "noul": 0.9}}, None),
    ({"size_reckless": "x"}, None),
    ({"headline_conflict": {"type": "noul", "noul": 0.9}}, None),
    ({}, None), (None, None), ("x", None),
])
def test_jev_size_reckless_noul_extraction(answers, want):
    assert sizing.jev_size_reckless_noul(answers) == want


# ─────────────────────────────────────────────────────────────────────────
# engine: the row carries it, including vetoed and no_size decisions, and it never changes the outcome
# ─────────────────────────────────────────────────────────────────────────

_RUN = {"n": 0}


def _scenario(cfg, monkeypatch, *, qty=4, conviction=0.5, verdict=None, anchor=True, telemetry_raises=False, equity_cfg=None,
              open_premium=0.0):
    """One full entry through the real engine, in its OWN data dir (a second run in the same dir would see the
    first run's open position). Returns (rows, open_qty_list)."""
    _RUN["n"] += 1
    run_dir = Path(cfg.data_dir) / f"run{_RUN['n']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    base_epoch = _base_dt().timestamp()
    chain, leg_symbol = _real_1dte_chain("SPY", _base_dt().date(), cfg)
    c = replace(cfg, mode=DpofMode.PAPER, alpaca_paper=True, underlyings=("SPY",), alpaca_key_id="k",
                alpaca_secret_key="s", size_anchor_enabled=anchor, data_dir=str(run_dir), **(equity_cfg or {}))
    broker = FakeBroker(gates_allowed=True, effective=DpofMode.PAPER, chain=chain, options_level=2)
    broker.quotes = {leg_symbol: OptionQuote(symbol=leg_symbol, underlying="SPY", expiration="", strike=500.0,
                                             right="call", bid=1.98, ask=2.02)}
    feed = FakeFeed(lambda now: _bullish_snapshot(now, "SPY"))
    verdict = verdict if verdict is not None else JevVerdict(mode="off", called=False, allow=True)

    async def _jev(signal, snap, now_et, cfg_, client, sizing_ctx=None):
        return verdict

    monkeypatch.setattr(engine_module, "jev_entry_check", _jev)
    if telemetry_raises:
        def boom(**_k):
            raise RuntimeError("telemetry bug")
        monkeypatch.setattr(engine_module, "size_telemetry", boom)

    decider = RealShapeDecider(entry=decider_mod.EntryDecision(
        action="enter", candidate_key="C1", conviction=conviction, qty=qty, thesis="t", invalidation="i"))
    clock_box = {"t": base_epoch}
    opened: list[int] = []

    async def _t():
        ledger = DPOFLedger(c.data_dir)
        await ledger.load()
        orders = FakeOrderManager(ledger=ledger)

        async def open_pos(mgr, choice, qty_, signal, jev, now_epoch, judgment):
            opened.append(qty_)
            return PositionRecord(id="P1", underlying=choice.underlying, direction=choice.direction.value,
                                  structure="single", status="open", qty_original=qty_, qty_open=qty_, mode="paper")

        orders.open_position_return = open_pos
        if open_premium:
            await ledger.upsert(PositionRecord(id="OLD", underlying="QQQ", direction="bullish", structure="single",
                                               status="open", qty_original=1, qty_open=1, mode="paper"))
            monkeypatch.setattr(engine_module.DPOFEngine, "_position_premium", lambda self, p: open_premium)
        eng = _engine(c, broker=broker, feed=feed, orders=orders, ledger=ledger, decider=decider,
                      candidate_fn=engine_module._default_candidate_fn,
                      expiry_fetch_window_fn=engine_module._default_expiry_fetch_window_fn,
                      position_alignment_fn=engine_module._default_position_alignment_fn,
                      clock=lambda: clock_box["t"])
        await eng.run_cycle(bot=None)
        assert "SPY" in eng._entry_tasks
        await eng._entry_tasks["SPY"]
        clock_box["t"] += 1
        await eng.run_cycle(bot=None)

    asyncio.run(_t())
    path = Path(c.data_dir) / "decisions.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    return [r for r in rows if r.get("kind") == "entry_deliberation"], opened


def test_entered_row_carries_the_sizing_block(cfg, monkeypatch):
    rows, opened = _scenario(cfg, monkeypatch, qty=4, conviction=0.5)
    assert opened == [4] and len(rows) == 1
    s = rows[0]["sizing"]
    assert s["qty_requested"] == 4 and s["qty"] == 4
    assert s["unit_premium"] == pytest.approx(200.0, rel=0.02)            # mid of 1.98 / 2.02
    assert s["premium"] == pytest.approx(800.0, rel=0.02)
    assert s["premium_pct_of_capital"] == pytest.approx(s["premium"] / s["dpof_equity"], abs=1e-4)
    assert s["anchor"]["enabled"] is True and s["anchor"]["share"] == 0.1 and s["anchor"]["qty"] >= 4
    assert s["jev_size_reckless_noul"] is None                             # Jev off: unknown, not 0


def test_jev_vetoed_row_carries_sizing_and_the_noul(cfg, monkeypatch):
    verdict = JevVerdict(mode="veto", called=True, allow=False, veto_reasons=["size_reckless"], answers=NOUL_83)
    rows, opened = _scenario(cfg, monkeypatch, qty=5, conviction=0.5, verdict=verdict)
    assert opened == [] and len(rows) == 1
    r = rows[0]
    assert r["reasons"] == ["jev_veto"] and r["jev_veto_reasons"] == ["size_reckless"]
    assert r["sizing"]["qty_requested"] == 5 and r["sizing"]["qty"] == 5
    assert r["sizing"]["jev_size_reckless_noul"] == 0.83 and r["sizing"]["jev_size_veto_threshold"] == 0.7
    assert "sizing" not in r["jev"]                                        # never inside the dict the signal log copies


def test_no_size_row_carries_the_requested_quantity(cfg, monkeypatch):
    rows, opened = _scenario(cfg, monkeypatch, qty=0, conviction=0.5)
    assert opened == [] and len(rows) == 1
    r = rows[0]
    assert "no_size" in r["reasons"]
    assert r["sizing"]["qty_requested"] == 0 and r["sizing"]["qty"] == 0
    assert r["sizing"]["unit_premium"] == pytest.approx(200.0, rel=0.02)      # the FIRST call's arguments (x100, not raw mid)
    assert r["sizing"]["capital_available"] == r["sizing"]["dpof_equity"]      # nothing open: equal


def test_a_clamped_quantity_keeps_both_the_request_and_the_result(cfg, monkeypatch):
    rows, opened = _scenario(cfg, monkeypatch, qty=5000, conviction=0.5)
    r = rows[0]
    assert r["sizing"]["qty_requested"] == 5000
    assert 0 < r["sizing"]["qty"] < 5000 and opened == [r["sizing"]["qty"]]
    assert r["sizing"]["qty"] == r["sizing"]["max_affordable_qty"]             # clamped down to solvency
    assert any(x.startswith("clamped_by_solvency") for x in r["size_reasons"])


def test_open_exposure_lowers_capital_available_but_not_equity(cfg, monkeypatch):
    rows, _ = _scenario(cfg, monkeypatch, qty=0, open_premium=1500.0)        # no_size row: only the first call ran
    s = rows[0]["sizing"]
    assert s["capital_available"] == pytest.approx(s["dpof_equity"] - 1500.0)
    assert s["max_affordable_qty"] == int(s["capital_available"] // s["unit_premium"])


def test_anchor_off_records_the_flag_only(cfg, monkeypatch):
    rows, opened = _scenario(cfg, monkeypatch, qty=4, anchor=False)
    assert opened == [4]
    assert rows[0]["sizing"]["anchor"] == {"enabled": False}


def test_a_failing_telemetry_helper_changes_nothing(cfg, monkeypatch):
    base_rows, base_opened = _scenario(cfg, monkeypatch, qty=4)
    rows, opened = _scenario(cfg, monkeypatch, qty=4, telemetry_raises=True)
    assert opened == base_opened == [4]                                    # the entry still went through
    assert "sizing" not in rows[0]                                         # and no half-written block
    assert rows[0]["reasons"] == base_rows[0]["reasons"]
    assert rows[0]["outcome"].startswith("entry_submitted")


def test_telemetry_never_changes_quantity_or_veto(cfg, monkeypatch):
    for verdict in (None, JevVerdict(mode="veto", called=True, allow=False, veto_reasons=["size_reckless"], answers=NOUL_83)):
        with_t, opened_t = _scenario(cfg, monkeypatch, qty=6, verdict=verdict)
        without_t, opened_n = _scenario(cfg, monkeypatch, qty=6, verdict=verdict, telemetry_raises=True)
        assert opened_t == opened_n
        assert with_t[0]["reasons"] == without_t[0]["reasons"]
