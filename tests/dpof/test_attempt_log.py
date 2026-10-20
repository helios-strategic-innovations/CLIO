"""R-05: per-attempt decider ledger (agent/dpof/attempt_log.py + decider hooks),
circuit transition logs, primary-failure WARNING, and failed-attempt cost booking.

No network: provider calls go through injected fake transports. All files live
under pytest's tmp_path (cfg fixture -> data_dir=tmp_path/dpof).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import replace

import pytest

from agent.dpof.attempt_log import AttemptLog
from agent.dpof.config import DPOFConfig, load_dpof_config
from agent.dpof.decider import DecisionUnavailable, DPOFDecider
from tests.dpof.test_decider import (
    FakeClock,
    _candidates,
    _entry_answer,
    _entry_ctx,
    _fixed_transport,
    _result,
)

DECIDER_LOGGER = "clio.dpof.decider"
OPENAI_KEY = "sk-openai-TESTKEY"


def _cfg_with_fallback(cfg) -> DPOFConfig:
    return replace(cfg, openai_api_key=OPENAI_KEY)


def _ledger_path(cfg) -> str:
    return os.path.join(cfg.data_dir, "decider_attempts.jsonl")


def _rows(cfg) -> list:
    path = _ledger_path(cfg)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run(decider, coro_factory):
    """Run a coroutine, then drain the fire-and-forget ledger writes."""
    async def go():
        out = await coro_factory()
        await decider._attempt_log.drain()
        return out
    return asyncio.run(go())


# ─────────────────────────────────────────────────────────────────────────
# Config normalisation (inside load_dpof_config)
# ─────────────────────────────────────────────────────────────────────────

def test_config_defaults_match_env_loader():
    dataclass_defaults = DPOFConfig()
    loaded = load_dpof_config()
    assert dataclass_defaults.attempt_log_enabled is True
    assert dataclass_defaults.attempt_log_max_lines == 20000
    assert loaded.attempt_log_enabled == dataclass_defaults.attempt_log_enabled
    assert loaded.attempt_log_max_lines == dataclass_defaults.attempt_log_max_lines


def test_config_env_flag_and_max_lines_are_normalised(monkeypatch):
    monkeypatch.setenv("CLIO_DPOF_ATTEMPT_LOG", "0")
    assert load_dpof_config().attempt_log_enabled is False
    monkeypatch.setenv("CLIO_DPOF_ATTEMPT_LOG", "1")
    assert load_dpof_config().attempt_log_enabled is True

    monkeypatch.setenv("CLIO_DPOF_ATTEMPT_LOG_MAX_LINES", "5")
    assert load_dpof_config().attempt_log_max_lines == 100
    monkeypatch.setenv("CLIO_DPOF_ATTEMPT_LOG_MAX_LINES", "999999")
    assert load_dpof_config().attempt_log_max_lines == 200000
    monkeypatch.setenv("CLIO_DPOF_ATTEMPT_LOG_MAX_LINES", "lots")
    assert load_dpof_config().attempt_log_max_lines == 20000


# ─────────────────────────────────────────────────────────────────────────
# One row per attempt
# ─────────────────────────────────────────────────────────────────────────

def test_primary_success_writes_one_ok_row(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer())), clock=FakeClock())
    _run(decider, lambda: decider.decide_entry(_entry_ctx(), _candidates()))

    rows = _rows(cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "anthropic"
    assert row["status"] == "ok"
    assert row["final_provider"] == "anthropic"
    assert row["fallback_used"] is False
    assert row["served_model"] == "claude-opus-5-5"
    assert row["requested_model"] == cfg.decision_model
    assert row["error_class"] == "" and row["error"] == ""
    assert row["claim_violations"] == []
    assert set(row) == {
        "ts", "kind", "provider", "requested_model", "served_model", "effort", "timeout_s",
        "latency_ms", "status", "error_class", "error", "usage", "cost_usd", "circuit",
        "fallback_used", "final_provider", "input_digest", "claim_violations",
    }


def test_timeout_then_fallback_writes_two_rows(cfg):
    cfg2 = _cfg_with_fallback(cfg)
    decider = DPOFDecider(
        cfg2,
        anthropic_transport=_fixed_transport(asyncio.TimeoutError()),
        openai_transport=_fixed_transport(_result(_entry_answer(), model="gpt-6-sol")),
        clock=FakeClock(),
    )
    result = _run(decider, lambda: decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.provider == "openai"

    rows = _rows(cfg2)
    assert [r["provider"] for r in rows] == ["anthropic", "openai"]
    primary, fallback = rows
    assert primary["status"] == "timeout"
    assert primary["error_class"] == "timeout"
    assert primary["served_model"] is None
    assert primary["fallback_used"] is False
    assert primary["final_provider"] == "openai"
    assert fallback["status"] == "ok"
    assert fallback["fallback_used"] is True
    assert fallback["served_model"] == "gpt-6-sol"
    assert fallback["requested_model"] == cfg2.fallback_model
    assert fallback["final_provider"] == "openai"


def test_rows_carry_each_provider_s_own_timeout_and_effort(cfg):
    # Distinct values per provider so a swapped field cannot hide behind equal defaults.
    cfg2 = replace(_cfg_with_fallback(cfg), decision_timeout_seconds=91.0, fallback_timeout_seconds=37.0,
                   decision_effort="medium", fallback_effort="low")
    decider = DPOFDecider(
        cfg2,
        anthropic_transport=_fixed_transport(asyncio.TimeoutError()),
        openai_transport=_fixed_transport(_result(_entry_answer(), model="gpt-6-sol")),
        clock=FakeClock(),
    )
    _run(decider, lambda: decider.decide_entry(_entry_ctx(), _candidates()))
    primary, fallback = _rows(cfg2)
    assert (primary["timeout_s"], primary["effort"]) == (91.0, "medium")
    assert (fallback["timeout_s"], fallback["effort"]) == (37.0, "low")


def test_both_providers_fail_rows_final_none_and_failed_cost_booked(cfg):
    cfg2 = _cfg_with_fallback(cfg)
    bad_json = _result("not json at all", usage={"input_tokens": 1_000_000, "output_tokens": 0,
                                                 "cache_read_input_tokens": 0})
    decider = DPOFDecider(
        cfg2,
        anthropic_transport=_fixed_transport(bad_json),
        openai_transport=_fixed_transport(RuntimeError("upstream exploded")),
        clock=FakeClock(),
    )

    async def go():
        with pytest.raises(DecisionUnavailable):
            await decider.decide_entry(_entry_ctx(), _candidates())
    _run(decider, go)

    rows = _rows(cfg2)
    assert [r["final_provider"] for r in rows] == ["none", "none"]
    assert rows[0]["status"] == "invalid_shape"
    assert rows[0]["usage"]["in"] == 1_000_000
    assert rows[0]["cost_usd"] == pytest.approx(4.0)
    assert rows[1]["status"] == "error"
    assert rows[1]["fallback_used"] is True
    # The failed anthropic call's usage is booked once; no decision is counted.
    status = decider.status()
    assert status["calls_today"] == 1
    assert status["est_cost_today_usd"] == pytest.approx(4.0)


def test_failed_attempt_usage_is_booked_once_and_winner_not_double_counted(cfg):
    cfg2 = _cfg_with_fallback(cfg)
    bad_json = _result("not json", usage={"input_tokens": 1_000_000, "output_tokens": 0,
                                          "cache_read_input_tokens": 0})
    decider = DPOFDecider(
        cfg2,
        anthropic_transport=_fixed_transport(bad_json),
        openai_transport=_fixed_transport(_result(_entry_answer(), model="gpt-6-sol",
                                                  usage={"input_tokens": 0, "output_tokens": 0,
                                                         "cache_read_input_tokens": 0})),
        clock=FakeClock(),
    )
    result = _run(decider, lambda: decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.provider == "openai"
    status = decider.status()
    assert status["calls_today"] == 1
    assert status["est_cost_today_usd"] == pytest.approx(4.0)


def test_totals_unchanged_when_no_failed_usage(cfg):
    usage = {"input_tokens": 300, "output_tokens": 60, "cache_read_input_tokens": 0}
    decider = DPOFDecider(
        cfg,
        anthropic_transport=_fixed_transport(_result(_entry_answer(), usage=usage)),
        clock=FakeClock(),
    )
    _run(decider, lambda: decider.decide_entry(_entry_ctx(), _candidates()))
    status = decider.status()
    assert status["calls_today"] == 1
    assert status["est_cost_today_usd"] == pytest.approx(0.0024)


# ─────────────────────────────────────────────────────────────────────────
# Redaction, off-loop writes, enable flag, trim
# ─────────────────────────────────────────────────────────────────────────

def test_attempt_row_builder_redacts_error_text(cfg):
    decider = DPOFDecider(_cfg_with_fallback(cfg), clock=FakeClock())
    row = decider._attempt_row(
        kind="entry", provider="openai",
        meta={"error": "HTTPError: Bearer sk-live-SECRET123 key=" + OPENAI_KEY, "usage": {}},
        elapsed_ms=5, input_digest="d", fallback_used=True, final_provider="none",
        circuit={"failures": 1, "open_until": None},
    )
    text = json.dumps(row)
    assert "sk-live-SECRET123" not in text
    assert OPENAI_KEY not in text
    assert row["error"].startswith("HTTPError: Bearer ***")
    assert row["error_class"] == "HTTPError"
    assert row["status"] == "error"


def test_attempt_row_error_is_capped_at_120_chars(cfg):
    decider = DPOFDecider(cfg, clock=FakeClock())
    row = decider._attempt_row(
        kind="entry", provider="anthropic", meta={"error": "X: " + "a" * 500, "usage": {}},
        elapsed_ms=1, input_digest="d", fallback_used=False, final_provider="none",
        circuit={"failures": 1, "open_until": None},
    )
    assert len(row["error"]) <= 120


def test_secret_does_not_reach_ledger_file(cfg):
    decider = DPOFDecider(
        cfg,
        anthropic_transport=_fixed_transport(RuntimeError("auth failed Bearer sk-live-SECRET123")),
        openai_transport=None,
        clock=FakeClock(),
    )

    async def go():
        with pytest.raises(DecisionUnavailable):
            await decider.decide_entry(_entry_ctx(), _candidates())
    _run(decider, go)
    with open(_ledger_path(cfg), encoding="utf-8") as f:
        text = f.read()
    assert "sk-live-SECRET123" not in text
    assert "Bearer ***" in text


def test_row_write_happens_off_the_loop(cfg, monkeypatch):
    gate = threading.Event()
    original = AttemptLog._append_sync

    def gated(self, row):
        gate.wait(2.0)
        return original(self, row)

    monkeypatch.setattr(AttemptLog, "_append_sync", gated)
    path = os.path.join(cfg.data_dir, "off_loop.jsonl")
    log = AttemptLog(path, max_lines=100)

    async def go():
        log.record({"i": 1})
        before = os.path.exists(path)
        gate.set()
        await log.drain()
        return before, os.path.exists(path)

    assert asyncio.run(go()) == (False, True)


def test_disabled_flag_writes_nothing(cfg):
    cfg_off = replace(cfg, attempt_log_enabled=False)
    decider = DPOFDecider(cfg_off, anthropic_transport=_fixed_transport(_result(_entry_answer())), clock=FakeClock())
    _run(decider, lambda: decider.decide_entry(_entry_ctx(), _candidates()))
    assert not os.path.exists(_ledger_path(cfg_off))


def test_trim_keeps_newest_lines(tmp_path):
    path = str(tmp_path / "trim.jsonl")
    log = AttemptLog(path, max_lines=100)

    async def go():
        for i in range(200):
            log.record({"i": i})
            await log.drain()
    asyncio.run(go())

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 90
    assert rows[0]["i"] == 110
    assert rows[-1]["i"] == 199


# ─────────────────────────────────────────────────────────────────────────
# Logs: circuit transitions and the rate-limited primary-failure WARNING
# ─────────────────────────────────────────────────────────────────────────

def test_circuit_open_and_close_are_logged(cfg, caplog):
    caplog.set_level(logging.INFO, logger=DECIDER_LOGGER)
    state = {"fail": True}

    async def anthropic(payload):
        if state["fail"]:
            raise RuntimeError("upstream down")
        return _result(_entry_answer())

    clock = FakeClock()
    decider = DPOFDecider(_cfg_with_fallback(cfg), anthropic_transport=anthropic,
                          openai_transport=_fixed_transport(_result(_entry_answer(), model="gpt-6-sol")),
                          clock=clock)

    async def go():
        for _ in range(3):              # default circuit threshold = 3
            await decider.decide_entry(_entry_ctx(), _candidates())
        await decider.decide_entry(_entry_ctx(), _candidates())   # circuit open: no primary call
        state["fail"] = False
        clock.advance(301)              # past the 300 s cooldown: half-open probe succeeds
        await decider.decide_entry(_entry_ctx(), _candidates())
    _run(decider, go)

    messages = [r.getMessage() for r in caplog.records if r.name == DECIDER_LOGGER]
    assert sum("circuit OPEN provider=anthropic" in m for m in messages) == 1
    assert sum("circuit CLOSED provider=anthropic" in m for m in messages) == 1
    statuses = [r["status"] for r in _rows(_cfg_with_fallback(cfg)) if r["provider"] == "anthropic"]
    assert "circuit_open" in statuses
    assert statuses[-1] == "ok"


def test_primary_failure_warning_is_rate_limited_and_redacted(cfg, caplog):
    caplog.set_level(logging.WARNING, logger=DECIDER_LOGGER)
    clock = FakeClock()
    decider = DPOFDecider(
        _cfg_with_fallback(cfg),
        anthropic_transport=_fixed_transport(RuntimeError("boom Bearer sk-live-SECRET123")),
        openai_transport=_fixed_transport(_result(_entry_answer(), model="gpt-6-sol")),
        clock=clock,
    )

    async def go():
        await decider.decide_entry(_entry_ctx(), _candidates())   # t=1000: warns
        await decider.decide_entry(_entry_ctx(), _candidates())   # same window: suppressed
        clock.advance(61)
        await decider.decide_entry(_entry_ctx(), _candidates())   # new window: warns again
    _run(decider, go)

    warnings = [r.getMessage() for r in caplog.records
                if r.name == DECIDER_LOGGER and "primary failed" in r.getMessage()]
    assert len(warnings) == 2
    assert all("provider=anthropic class=RuntimeError" in w for w in warnings)
    assert "circuit=closed" in warnings[0]   # the 3rd failure (second warning) opens the circuit
    assert not any("sk-live-SECRET123" in w for w in warnings)
