"""Tests for agent.dpof.ledger.DPOFLedger."""

import asyncio
import json
import os

import pytest

from agent.dpof import ledger as ledger_module
from agent.dpof.ledger import DPOFLedger
from agent.dpof.models import PositionRecord, RiskState


def _make_pos(pos_id="p1", **overrides) -> PositionRecord:
    kwargs = dict(
        id=pos_id,
        underlying="SPY",
        direction="bullish",
        structure="single",
        legs=[{"symbol": "SPY261002C00575000", "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}],
        status="open",
        qty_original=2,
        qty_open=2,
        entry_price=1.5,
        entry_time=1000.0,
        entry_client_order_id="DPOF-EN-20260924000000-abcdef",
    )
    kwargs.update(overrides)
    return PositionRecord(**kwargs)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_upsert_and_round_trip(cfg):
    async def _t():
        led = DPOFLedger(cfg.data_dir)
        await led.load()
        assert led.positions == []
        assert led.open_positions() == []

        pos = _make_pos()
        await led.upsert(pos)
        assert led.get("p1") is not None
        assert led.get("missing") is None

        risk = RiskState(day="2026-09-24", realized_pnl_today=12.5, entries_today=1)
        await led.save_risk(risk)

        # Fresh instance -- proves it round-trips through the files, not
        # just in-memory state.
        led2 = DPOFLedger(cfg.data_dir)
        await led2.load()
        got = led2.get("p1")
        assert got is not None
        assert got.underlying == "SPY"
        assert got.qty_open == 2
        assert got.entry_price == 1.5
        assert got.legs[0]["symbol"] == "SPY261002C00575000"
        assert led2.risk.day == "2026-09-24"
        assert led2.risk.realized_pnl_today == 12.5
        assert led2.risk.entries_today == 1

    asyncio.run(_t())


def test_open_positions_filters_by_status(cfg):
    async def _t():
        led = DPOFLedger(cfg.data_dir)
        await led.load()
        await led.upsert(_make_pos("a", status="pending"))
        await led.upsert(_make_pos("b", status="open"))
        await led.upsert(_make_pos("c", status="closing"))
        await led.upsert(_make_pos("d", status="closed"))
        await led.upsert(_make_pos("e", status="cancelled"))

        open_ids = {p.id for p in led.open_positions()}
        assert open_ids == {"a", "b", "c"}
        assert {p.id for p in led.positions} == {"a", "b", "c", "d", "e"}

    asyncio.run(_t())


def test_round_trip_unknown_fields_ignored(cfg):
    """Old ledger files must survive schema additions (CLAUDE.md
    dataclass-deserialization rule) -- unknown top-level keys are dropped by
    PositionRecord.from_dict / RiskState.from_dict, never crash the load."""

    async def _t():
        os.makedirs(cfg.data_dir, exist_ok=True)
        raw_positions = {
            "version": 1,
            "positions": [
                {
                    "id": "legacy1",
                    "underlying": "SPY",
                    "direction": "bullish",
                    "structure": "single",
                    "legs": [],
                    "status": "open",
                    "qty_original": 1,
                    "qty_open": 1,
                    "future_field_from_a_newer_schema": {"nested": True},
                }
            ],
        }
        with open(os.path.join(cfg.data_dir, "positions.json"), "w") as f:
            json.dump(raw_positions, f)

        raw_risk = {"day": "2026-09-24", "realized_pnl_today": 5.0, "some_future_key": "xyz"}
        with open(os.path.join(cfg.data_dir, "risk_state.json"), "w") as f:
            json.dump(raw_risk, f)

        led = DPOFLedger(cfg.data_dir)
        await led.load()
        pos = led.get("legacy1")
        assert pos is not None
        assert pos.underlying == "SPY"
        assert not hasattr(pos, "future_field_from_a_newer_schema")
        assert led.risk.day == "2026-09-24"
        assert led.risk.realized_pnl_today == 5.0

    asyncio.run(_t())


def test_old_risk_state_without_realized_pnl_total_loads_as_zero(cfg):
    """A risk_state.json persisted BEFORE realized_pnl_total existed (S-DPOF
    virtual-capital work) has no such key at all -- from_dict's dataclass
    default (0.0) must apply, never a crash or a missing-attribute error."""

    async def _t():
        os.makedirs(cfg.data_dir, exist_ok=True)
        raw_risk = {
            "day": "2026-09-24", "week": "2026-W39",
            "realized_pnl_today": 12.0, "realized_pnl_week": 40.0,
            "start_of_day_equity": 20_000.0, "hwm_equity": 20_500.0,
            # deliberately no "realized_pnl_total" key
        }
        with open(os.path.join(cfg.data_dir, "risk_state.json"), "w") as f:
            json.dump(raw_risk, f)

        led = DPOFLedger(cfg.data_dir)
        await led.load()
        assert led.risk.realized_pnl_total == 0.0
        assert led.risk.realized_pnl_today == 12.0
        assert led.risk.hwm_equity == 20_500.0

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Corruption handling
# ---------------------------------------------------------------------------


def test_corrupt_positions_json_is_quarantined_and_engine_continues(cfg):
    async def _t():
        os.makedirs(cfg.data_dir, exist_ok=True)
        bad_path = os.path.join(cfg.data_dir, "positions.json")
        with open(bad_path, "w") as f:
            f.write("{not valid json!!")

        led = DPOFLedger(cfg.data_dir)
        await led.load()  # must NOT raise

        assert led.positions == []
        assert not os.path.exists(bad_path)  # original renamed away
        quarantined = [f for f in os.listdir(cfg.data_dir) if f.startswith("positions.json.corrupt-")]
        assert len(quarantined) == 1

        # Engine keeps working after a corrupt load.
        await led.upsert(_make_pos())
        assert led.get("p1") is not None

    asyncio.run(_t())


def test_corrupt_risk_state_json_is_quarantined(cfg):
    async def _t():
        os.makedirs(cfg.data_dir, exist_ok=True)
        bad_path = os.path.join(cfg.data_dir, "risk_state.json")
        with open(bad_path, "w") as f:
            f.write("]] garbage [[")

        led = DPOFLedger(cfg.data_dir)
        await led.load()

        assert led.risk.day == ""
        quarantined = [f for f in os.listdir(cfg.data_dir) if f.startswith("risk_state.json.corrupt-")]
        assert len(quarantined) == 1

    asyncio.run(_t())


def test_missing_files_load_as_empty_defaults(cfg):
    async def _t():
        led = DPOFLedger(cfg.data_dir)
        await led.load()  # neither file exists yet
        assert led.positions == []
        assert led.risk.day == ""
        assert led.halt_file_present() is False

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def test_atomic_write_leaves_no_tmp_file(cfg):
    async def _t():
        led = DPOFLedger(cfg.data_dir)
        await led.load()
        await led.upsert(_make_pos())
        await led.save_risk(RiskState(day="2026-09-24"))

        entries = os.listdir(cfg.data_dir)
        assert "positions.json" in entries
        assert "risk_state.json" in entries
        assert not any(e.endswith(".tmp") for e in entries)

        with open(os.path.join(cfg.data_dir, "positions.json")) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["positions"]) == 1

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# JSONL cap
# ---------------------------------------------------------------------------


def test_jsonl_append_caps_and_trims(cfg, monkeypatch):
    monkeypatch.setattr(ledger_module, "_JSONL_CAP", 5)
    monkeypatch.setattr(ledger_module, "_JSONL_TRIM_TO", 3)
    monkeypatch.setattr(ledger_module, "_JSONL_MAX_BYTES", 0)  # size gate always open -> line cap governs

    async def _t():
        led = DPOFLedger(cfg.data_dir)
        await led.load()
        for i in range(7):
            await led.append_decision({"i": i})

        path = os.path.join(cfg.data_dir, "decisions.jsonl")
        with open(path) as f:
            lines = f.readlines()
        # Trim only fires once the file exceeds the cap (5), cutting to 3;
        # i=0..4 -> 5 lines (no trim yet), i=5 -> 6 lines (>5, trim to 3:
        # [3,4,5]), i=6 -> append -> 4 lines (3+1, not >5, no trim).
        assert len(lines) == 4
        kept = [json.loads(l)["i"] for l in lines]
        assert kept == [3, 4, 5, 6]
        assert len(kept) <= 7  # never grows unbounded

    asyncio.run(_t())


def test_append_jev_is_a_separate_file(cfg):
    async def _t():
        led = DPOFLedger(cfg.data_dir)
        await led.load()
        await led.append_decision({"kind": "decision"})
        await led.append_jev({"kind": "jev"})

        dec_path = os.path.join(cfg.data_dir, "decisions.jsonl")
        jev_path = os.path.join(cfg.data_dir, "jev.jsonl")
        with open(dec_path) as f:
            assert json.loads(f.readline())["kind"] == "decision"
        with open(jev_path) as f:
            assert json.loads(f.readline())["kind"] == "jev"

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# run_in_executor discipline (CLAUDE.md: never flock/blocking I/O on the loop)
# ---------------------------------------------------------------------------


def test_all_file_io_goes_through_run_in_executor(cfg):
    async def _t():
        loop = asyncio.get_running_loop()
        calls = []
        original = loop.run_in_executor

        def tracking(executor, func, *a):
            calls.append(func)
            return original(executor, func, *a)

        loop.run_in_executor = tracking
        try:
            led = DPOFLedger(cfg.data_dir)
            await led.load()
            await led.upsert(_make_pos())
            await led.save_risk(RiskState(day="2026-09-24"))
            await led.append_decision({"x": 1})
        finally:
            loop.run_in_executor = original

        assert len(calls) >= 4  # load, upsert, save_risk, append_decision

    asyncio.run(_t())


def test_halt_file_present(cfg):
    async def _t():
        led = DPOFLedger(cfg.data_dir)
        await led.load()
        assert led.halt_file_present() is False
        os.makedirs(cfg.data_dir, exist_ok=True)
        open(os.path.join(cfg.data_dir, "HALT"), "w").close()
        assert led.halt_file_present() is True

    asyncio.run(_t())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


def test_jsonl_append_skips_full_read_below_size_gate(cfg, monkeypatch):
    """Orchestrator review: below _JSONL_MAX_BYTES an append must not trim (or read) the file."""
    monkeypatch.setattr(ledger_module, "_JSONL_CAP", 2)
    async def _t():
        led = DPOFLedger(cfg.data_dir)
        await led.load()
        for i in range(6):
            await led.append_decision({"i": i})
        with open(os.path.join(cfg.data_dir, "decisions.jsonl")) as f:
            assert len(f.readlines()) == 6
    asyncio.run(_t())


def test_load_archives_old_closed_positions_only(cfg):
    import time as _time
    from agent.dpof.models import PositionRecord
    async def _t():
        led = DPOFLedger(cfg.data_dir)
        await led.load()
        old = _time.time() - 30 * 86400
        await led.upsert(PositionRecord(id="old-closed", underlying="SPY", direction="bullish", structure="single",
                                        status="closed", closed_time=old))
        await led.upsert(PositionRecord(id="old-open", underlying="SPY", direction="bullish", structure="single",
                                        status="open", closed_time=None))
        await led.upsert(PositionRecord(id="recent-closed", underlying="SPY", direction="bullish", structure="single",
                                        status="closed", closed_time=_time.time() - 3600))
        led2 = DPOFLedger(cfg.data_dir)
        await led2.load()
        ids = {p.id for p in led2.positions}
        assert ids == {"old-open", "recent-closed"}
        with open(os.path.join(cfg.data_dir, "positions_archive.jsonl")) as f:
            archived = [json.loads(l)["id"] for l in f]
        assert archived == ["old-closed"]
    asyncio.run(_t())


def test_concurrent_upserts_last_state_wins(cfg):
    """Overlapping upserts must never leave an older snapshot on disk."""
    from agent.dpof.models import PositionRecord
    async def _t():
        led = DPOFLedger(cfg.data_dir)
        await led.load()
        recs = [PositionRecord(id=f"p{i}", underlying="SPY", direction="bullish", structure="single", status="open")
                for i in range(20)]
        await asyncio.gather(*(led.upsert(r) for r in recs))
        led2 = DPOFLedger(cfg.data_dir)
        await led2.load()
        assert {p.id for p in led2.positions} == {f"p{i}" for i in range(20)}
    asyncio.run(_t())
