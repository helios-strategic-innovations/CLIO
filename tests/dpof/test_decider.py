"""Tests for agent.dpof.decider — DPOFDecider (Anthropic primary / OpenAI
fallback judgment layer) and its pure validation/parsing helpers.

Plugin-free: async code is driven with asyncio.run() inside ordinary test
functions (see tests/dpof/conftest.py). No network: every provider call
goes through an injected fake `anthropic_transport`/`openai_transport`.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
import pytest

from agent.dpof.config import DPOFConfig
from agent.dpof.decider import (
    DecisionUnavailable,
    DPOFDecider,
    ENTRY_SCHEMA,
    EntryCandidate,
    EntryDecision,
    REVIEW_SCHEMA,
    ReviewDecision,
    _cost_usd,
    _extract_first_json_object,
    _floor_qty,
    _is_output_config_400,
    _redact,
    _shape_matches,
    _try_parse_json,
    _validate_entry,
    _validate_review,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

class FakeClock:
    def __init__(self, start: float = 1_000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _et_epoch(hour: int = 12, minute: int = 0, day: int = 24, month: int = 9, year: int = 2026) -> float:
    return datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/New_York")).timestamp()


def _entry_ctx(headlines=None) -> dict:
    return {
        "now_et": "2026-09-24T12:00:00-04:00",
        "minutes_to_flatten": 230,
        "mode": "paper",
        "underlyings": {"SPY": {"composite": 0.42, "direction": "bullish", "strength": "strong"}},
        "market_regime": "normal",
        "open_positions": [],
        "day": {"realized_today": 0.0, "entries_today": 0, "open_premium": 0.0},
        "recent_closed": [],
        "econ_events": [],
        "jev": None,
        "headlines": headlines or [],
    }


def _candidates(max_affordable_qty: int = 5) -> list:
    return [
        EntryCandidate(
            key="C1", underlying="SPY", direction="bullish", structure="single",
            legs=[{"symbol": "SPY261002C00580000", "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}],
            expiry="2026-09-25", sessions_to_expiry=1, strike_desc="580C",
            bid=1.18, ask=1.28, mid=1.23, spread_pct=0.08,
            delta=0.5, theta=-0.12, iv=0.18, open_interest=1200, unit_premium=123.0,
            max_affordable_qty=max_affordable_qty,
        )
    ]


def _position() -> dict:
    return {
        "id": "pos_1", "underlying": "SPY", "direction": "bullish", "qty_open": 2,
        "entry_price": 1.10, "thesis": "flow bullish", "invalidation": "loses VWAP",
    }


def _entry_answer(
    action="enter", candidate_key="C1", conviction=0.7, qty=1,
    thesis="thesis text", invalidation="invalidation text", recheck_minutes=5,
) -> dict:
    return {
        "action": action, "candidate_key": candidate_key, "conviction": conviction,
        "qty": qty, "thesis": thesis, "invalidation": invalidation,
        "recheck_minutes": recheck_minutes,
    }


def _review_answer(
    action="hold", trim_fraction=0.0, urgency="normal", thesis_status="intact",
    reason="reason text", next_review_minutes=5,
) -> dict:
    return {
        "action": action, "trim_fraction": trim_fraction, "urgency": urgency,
        "thesis_status": thesis_status, "reason": reason, "next_review_minutes": next_review_minutes,
    }


def _result(text_obj, model="claude-opus-5-5", usage=None) -> dict:
    text = text_obj if isinstance(text_obj, str) else json.dumps(text_obj)
    return {
        "text": text,
        "usage": usage or {"input_tokens": 300, "output_tokens": 60, "cache_read_input_tokens": 0},
        "model": model,
    }


def _fixed_transport(payload_or_exc, calls: Optional[list] = None):
    async def transport(payload):
        if calls is not None:
            calls.append(payload)
        if isinstance(payload_or_exc, Exception):
            raise payload_or_exc
        return payload_or_exc
    return transport


def _sequence_transport(items, default=None, calls: Optional[list] = None):
    it = iter(items)

    async def transport(payload):
        if calls is not None:
            calls.append(payload)
        try:
            item = next(it)
        except StopIteration:
            item = default
        if isinstance(item, Exception):
            raise item
        return item
    return transport


class _Fake400(Exception):
    def __init__(self, msg: str):
        self.status_code = 400
        super().__init__(msg)


# ─────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────

def test_happy_path_entry(cfg):
    calls: list = []
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer()), calls))
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))

    assert isinstance(result, EntryDecision)
    assert result.action == "enter"
    assert result.candidate_key == "C1"
    assert result.conviction == pytest.approx(0.7)
    assert result.qty == 1
    assert result.clamped is False
    assert result.provider == "anthropic"
    assert result.model == "claude-opus-5-5"
    assert result.latency_ms >= 0
    assert result.raw["action"] == "enter"
    assert len(calls) == 1
    # System block carries the CHARTER with prompt-cache control.
    assert calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_happy_path_review(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_review_answer(action="trim", trim_fraction=0.4))))
    result = asyncio.run(decider.review_position(_entry_ctx(), _position()))

    assert isinstance(result, ReviewDecision)
    assert result.action == "trim"
    assert result.trim_fraction == pytest.approx(0.4)
    assert result.provider == "anthropic"


# ─────────────────────────────────────────────────────────────────────────
# Fallback behavior
# ─────────────────────────────────────────────────────────────────────────

def test_schema_invalid_primary_falls_back_to_openai(cfg):
    cfg = replace(cfg, fallback_model="gpt-6-sol", openai_api_key="sk-fallback-key")
    anthropic_calls: list = []
    openai_calls: list = []
    decider = DPOFDecider(
        cfg,
        anthropic_transport=_fixed_transport(_result('{"not": "the right shape"}'), anthropic_calls),
        openai_transport=_fixed_transport(_result(_entry_answer(), model="gpt-6-sol"), openai_calls),
    )
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))

    assert result.provider == "openai"
    assert result.model == "gpt-6-sol"
    assert len(anthropic_calls) == 1
    assert len(openai_calls) == 1
    assert decider._anthropic_status.consecutive_failures == 1
    assert decider._anthropic_status.last_error == "invalid_json"


def test_primary_timeout_falls_back_to_openai(cfg):
    cfg = replace(cfg, decision_timeout_seconds=0.2, fallback_model="gpt-6-sol", openai_api_key="sk-fallback-key")

    async def slow_anthropic(payload):
        await asyncio.sleep(1.0)
        return _result(_entry_answer())

    decider = DPOFDecider(
        cfg,
        anthropic_transport=slow_anthropic,
        openai_transport=_fixed_transport(_result(_entry_answer(), model="gpt-6-sol")),
    )
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))

    assert result.provider == "openai"
    assert decider._anthropic_status.last_error == "timeout"


def test_both_providers_fail_raises_decision_unavailable(cfg):
    cfg = replace(cfg, fallback_model="gpt-6-sol", openai_api_key="sk-fallback-key")
    decider = DPOFDecider(
        cfg,
        anthropic_transport=_fixed_transport(RuntimeError("anthropic down")),
        openai_transport=_fixed_transport(RuntimeError("openai down")),
    )
    with pytest.raises(DecisionUnavailable):
        asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))


def test_fallback_disabled_when_model_empty(cfg):
    cfg = replace(cfg, fallback_model="", openai_api_key="sk-would-be-used")
    openai_calls: list = []
    decider = DPOFDecider(
        cfg,
        anthropic_transport=_fixed_transport(RuntimeError("anthropic down")),
        openai_transport=_fixed_transport(_result(_entry_answer()), openai_calls),
    )
    with pytest.raises(DecisionUnavailable) as exc_info:
        asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert "fallback_disabled" in str(exc_info.value)
    assert openai_calls == []


def test_fallback_disabled_when_no_key(cfg):
    cfg = replace(cfg, fallback_model="gpt-6-sol", openai_api_key="")
    openai_calls: list = []
    decider = DPOFDecider(
        cfg,
        anthropic_transport=_fixed_transport(RuntimeError("anthropic down")),
        openai_transport=_fixed_transport(_result(_entry_answer()), openai_calls),
    )
    with pytest.raises(DecisionUnavailable) as exc_info:
        asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert "fallback_disabled" in str(exc_info.value)
    assert openai_calls == []


# ─────────────────────────────────────────────────────────────────────────
# Circuit breaker
# ─────────────────────────────────────────────────────────────────────────

def test_circuit_opens_after_n_failures_and_recovers_after_cooldown(cfg, tmp_path):
    cfg = replace(
        cfg, data_dir=str(tmp_path), fallback_model="", decider_circuit_failures=2,
        decider_circuit_cooldown_seconds=100, decision_timeout_seconds=5,
    )
    clock = FakeClock(_et_epoch())
    calls: list = []
    transport = _sequence_transport(
        [RuntimeError("boom1"), RuntimeError("boom2"), _result(_entry_answer())], calls=calls,
    )
    decider = DPOFDecider(cfg, anthropic_transport=transport, clock=clock, data_dir=str(tmp_path))

    with pytest.raises(DecisionUnavailable):
        asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert decider._anthropic_status.consecutive_failures == 1
    assert decider._anthropic_status.circuit_open_until is None

    with pytest.raises(DecisionUnavailable):
        asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert decider._anthropic_status.consecutive_failures == 2
    assert decider._anthropic_status.circuit_open_until is not None

    calls_before = len(calls)
    with pytest.raises(DecisionUnavailable):
        asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert len(calls) == calls_before  # circuit open -> transport never invoked

    clock.advance(cfg.decider_circuit_cooldown_seconds + 1)
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.action == "enter"
    assert decider._anthropic_status.consecutive_failures == 0
    assert decider._anthropic_status.circuit_open_until is None


# ─────────────────────────────────────────────────────────────────────────
# Daily cap persistence
# ─────────────────────────────────────────────────────────────────────────

def test_daily_cap_persists_across_new_instance_and_resets_next_day(cfg, tmp_path):
    cfg = replace(cfg, data_dir=str(tmp_path), max_decisions_per_day=1, fallback_model="")
    clock = FakeClock(_et_epoch())

    decider1 = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer())), clock=clock, data_dir=str(tmp_path))
    result = asyncio.run(decider1.decide_entry(_entry_ctx(), _candidates()))
    assert result.action == "enter"

    with pytest.raises(DecisionUnavailable) as exc1:
        asyncio.run(decider1.decide_entry(_entry_ctx(), _candidates()))
    assert "daily_cap" in str(exc1.value)

    # Fresh instance, same data_dir, same (fake) day: the cap survives a restart.
    decider2 = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer())), clock=clock, data_dir=str(tmp_path))
    with pytest.raises(DecisionUnavailable) as exc2:
        asyncio.run(decider2.decide_entry(_entry_ctx(), _candidates()))
    assert "daily_cap" in str(exc2.value)

    # Cross into the next ET day: the cap resets.
    clock.advance(24 * 3600)
    decider3 = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer())), clock=clock, data_dir=str(tmp_path))
    result3 = asyncio.run(decider3.decide_entry(_entry_ctx(), _candidates()))
    assert result3.action == "enter"


# ─────────────────────────────────────────────────────────────────────────
# Validation / clamping
# ─────────────────────────────────────────────────────────────────────────

def test_invalid_candidate_key_forces_pass(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer(candidate_key="ZZZ"))))
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.action == "pass"
    assert result.candidate_key is None


def test_missing_candidate_key_on_enter_forces_pass(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer(candidate_key=""))))
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.action == "pass"


def test_clamping_conviction(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer(conviction=5.0))))
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.conviction == 1.0


# ─────────────────────────────────────────────────────────────────────────
# qty / solvency clamping (operator amendment, 2026-09-24: NO hard size
# caps -- the model's qty IS the size; the only code bound is solvency)
# ─────────────────────────────────────────────────────────────────────────

def test_qty_less_than_1_on_enter_forces_pass(cfg):
    candidates = _candidates()
    fields, reasons = _validate_entry(_entry_answer(qty=0), candidates, cfg)
    assert fields["action"] == "pass"
    assert fields["candidate_key"] is None
    assert fields["qty"] == 0
    assert fields["clamped"] is False
    assert "invalid_qty" in reasons


def test_negative_qty_on_enter_forces_pass(cfg):
    candidates = _candidates()
    fields, reasons = _validate_entry(_entry_answer(qty=-4), candidates, cfg)
    assert fields["action"] == "pass"
    assert fields["qty"] == 0
    assert "invalid_qty" in reasons


def test_qty_above_max_affordable_qty_is_clamped(cfg):
    candidates = _candidates(max_affordable_qty=3)
    fields, reasons = _validate_entry(_entry_answer(qty=50), candidates, cfg)
    assert fields["action"] == "enter"
    assert fields["qty"] == 3
    assert fields["clamped"] is True
    # Clamping is NOT a failure -- it never lands in `reasons`.
    assert reasons == []


def test_decide_entry_clamps_qty_to_max_affordable_qty(cfg):
    candidates = _candidates(max_affordable_qty=3)
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer(qty=50))))
    result = asyncio.run(decider.decide_entry(_entry_ctx(), candidates))
    assert result.action == "enter"
    assert result.qty == 3
    assert result.clamped is True


def test_qty_fractional_is_floored(cfg):
    candidates = _candidates(max_affordable_qty=10)
    fields, reasons = _validate_entry(_entry_answer(qty=3.9), candidates, cfg)
    assert fields["action"] == "enter"
    assert fields["qty"] == 3
    assert fields["clamped"] is False
    assert reasons == []


def test_qty_nan_inf_and_non_numeric_force_pass(cfg):
    candidates = _candidates()
    for bad in (float("nan"), float("inf"), float("-inf"), "not-a-number", None):
        fields, reasons = _validate_entry(_entry_answer(qty=bad), candidates, cfg)
        assert fields["action"] == "pass", f"qty={bad!r} should force pass"
        assert fields["qty"] == 0
        assert "invalid_qty" in reasons


def test_floor_qty_helper():
    assert _floor_qty(3.9) == 3
    assert _floor_qty(3) == 3
    assert _floor_qty("4") == 4
    assert _floor_qty(float("nan")) is None
    assert _floor_qty(float("inf")) is None
    assert _floor_qty("nope") is None
    assert _floor_qty(None) is None
    assert _floor_qty(True) is None


def test_recheck_minutes_clamped_to_review_bounds(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer(recheck_minutes=99999))))
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.recheck_minutes == cfg.review_max_minutes

    decider2 = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer(recheck_minutes=0))))
    result2 = asyncio.run(decider2.decide_entry(_entry_ctx(), _candidates()))
    assert result2.recheck_minutes == cfg.review_min_minutes


def test_next_review_minutes_clamped_to_review_bounds(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_review_answer(next_review_minutes=-5))))
    result = asyncio.run(decider.review_position(_entry_ctx(), _position()))
    assert result.next_review_minutes == cfg.review_min_minutes


def test_unknown_action_enum_forces_safest(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer(action="yolo"))))
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.action == "pass"

    decider2 = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_review_answer(action="obliterate"))))
    result2 = asyncio.run(decider2.review_position(_entry_ctx(), _position()))
    assert result2.action == "hold"


def test_trim_fraction_le_0_forces_hold(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_review_answer(action="trim", trim_fraction=0.0))))
    result = asyncio.run(decider.review_position(_entry_ctx(), _position()))
    assert result.action == "hold"
    assert result.trim_fraction == 0.0


def test_trim_fraction_ge_1_forces_exit(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_review_answer(action="trim", trim_fraction=1.0))))
    result = asyncio.run(decider.review_position(_entry_ctx(), _position()))
    assert result.action == "exit"
    assert result.trim_fraction == 0.0


def test_trim_fraction_mid_range_kept(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_review_answer(action="trim", trim_fraction=0.35))))
    result = asyncio.run(decider.review_position(_entry_ctx(), _position()))
    assert result.action == "trim"
    assert result.trim_fraction == pytest.approx(0.35)


# ─────────────────────────────────────────────────────────────────────────
# Anthropic output_config 400 downgrade path
# ─────────────────────────────────────────────────────────────────────────

def test_output_config_400_downgrades_and_extracts_json_from_prose(cfg):
    calls: list = []
    prose_wrapped = "Here is my decision:\n" + json.dumps(_entry_answer(candidate_key="C1")) + "\nEnd of answer."
    transport = _sequence_transport(
        [
            _Fake400("400 Bad Request: output_config.format is not supported for this model"),
            _result(prose_wrapped),
        ],
        default=_result(_entry_answer(candidate_key="C1")),
        calls=calls,
    )
    decider = DPOFDecider(cfg, anthropic_transport=transport)

    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.action == "enter"
    assert decider._anthropic_format_downgraded is True
    assert "format" in calls[0]["output_config"]
    assert "format" not in calls[1]["output_config"]
    assert calls[1]["output_config"]["effort"] == cfg.decision_effort

    # The downgrade persists for the rest of the process: a later call never
    # re-sends output_config.format either.
    result2 = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result2.action == "enter"
    assert "format" not in calls[2]["output_config"]


# ─────────────────────────────────────────────────────────────────────────
# OpenAI payload shape
# ─────────────────────────────────────────────────────────────────────────

def test_openai_payload_never_contains_temperature_or_top_p(cfg):
    cfg = replace(cfg, fallback_model="gpt-6-sol", openai_api_key="sk-fallback-key")
    calls: list = []
    decider = DPOFDecider(
        cfg,
        anthropic_transport=_fixed_transport(RuntimeError("primary down")),
        openai_transport=_fixed_transport(_result(_entry_answer(), model="gpt-6-sol"), calls),
    )
    asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))

    assert len(calls) == 1
    payload = calls[0]
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert payload["model"] == "gpt-6-sol"
    assert payload["reasoning"] == {"effort": cfg.fallback_effort}
    assert payload["text"]["format"]["strict"] is True
    assert payload["input"][0]["role"] == "developer"
    assert payload["input"][1]["role"] == "user"


def test_openai_response_size_cap_rejects_oversized(cfg):
    oversized_body = json.dumps({"output_text": "x" * (2 * 1024 * 1024)}).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized_body, headers={"content-length": str(len(oversized_body))})

    decider = DPOFDecider(cfg)
    decider._openai_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        asyncio.run(decider._openai_http_call({"model": "gpt-6-sol"}, timeout=5.0))


# ─────────────────────────────────────────────────────────────────────────
# Ledger / record callback
# ─────────────────────────────────────────────────────────────────────────

def test_record_callback_receives_ledger_entry(cfg):
    records: list = []

    async def record(entry):
        records.append(entry)

    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer())), record=record)
    asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))

    assert len(records) == 1
    rec = records[0]
    assert rec["kind"] == "entry"
    assert rec["provider"] == "anthropic"
    assert rec["error"] == ""
    assert len(rec["input_digest"]) == 64
    assert rec["validated"]["action"] == "enter"


def test_record_callback_exception_does_not_break_decide(cfg):
    async def bad_record(entry):
        raise RuntimeError("ledger down")

    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer())), record=bad_record)
    result = asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))
    assert result.action == "enter"


# ─────────────────────────────────────────────────────────────────────────
# Secrets never logged
# ─────────────────────────────────────────────────────────────────────────

def test_openai_failure_never_logs_api_key(cfg, caplog):
    cfg = replace(cfg, fallback_model="gpt-6-sol", openai_api_key="sk-super-secret-value")

    async def bad_transport(payload):
        raise RuntimeError("auth failed for Bearer sk-super-secret-value token")

    decider = DPOFDecider(
        cfg,
        anthropic_transport=_fixed_transport(RuntimeError("primary down")),
        openai_transport=bad_transport,
    )
    with pytest.raises(DecisionUnavailable):
        asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))

    assert "sk-super-secret-value" not in decider._openai_status.last_error
    assert "sk-super-secret-value" not in caplog.text


def test_redact_strips_bearer_and_named_secrets():
    msg = "unauthorized Bearer sk-verysecret123 and also sk-verysecret123 again"
    out = _redact(msg, "sk-verysecret123")
    assert "sk-verysecret123" not in out
    assert "Bearer ***" in out


# ─────────────────────────────────────────────────────────────────────────
# status()
# ─────────────────────────────────────────────────────────────────────────

def test_status_reports_calls_and_cost(cfg):
    decider = DPOFDecider(cfg, anthropic_transport=_fixed_transport(_result(_entry_answer(), usage={"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 0})))
    asyncio.run(decider.decide_entry(_entry_ctx(), _candidates()))

    status = decider.status()
    assert status["calls_today"] == 1
    assert status["est_cost_today_usd"] > 0
    assert status["providers"]["anthropic"]["consecutive_failures"] == 0
    assert status["providers"]["anthropic"]["model"] == cfg.decision_model
    assert status["providers"]["openai"]["model"] == cfg.fallback_model


# ─────────────────────────────────────────────────────────────────────────
# update_config / close
# ─────────────────────────────────────────────────────────────────────────

def test_update_config_replaces_cfg(cfg):
    decider = DPOFDecider(cfg)
    new_cfg = replace(cfg, decision_model="claude-opus-5-5-preview")
    decider.update_config(new_cfg)
    assert decider.cfg is new_cfg


def test_close_is_safe_with_nothing_constructed(cfg):
    decider = DPOFDecider(cfg)
    asyncio.run(decider.close())  # must not raise


def test_close_closes_openai_client(cfg):
    cfg2 = replace(cfg, fallback_model="gpt-6-sol", openai_api_key="sk-fallback-key")
    decider = DPOFDecider(
        cfg2,
        anthropic_transport=_fixed_transport(RuntimeError("primary down")),
    )
    # Force the real (non-transport) OpenAI client to be constructed.
    client = decider._get_openai_client()
    asyncio.run(decider.close())
    assert client.is_closed
    assert decider._openai_client is None


# ─────────────────────────────────────────────────────────────────────────
# Pure helper unit tests
# ─────────────────────────────────────────────────────────────────────────

def test_extract_first_json_object_handles_nested_and_quoted_braces():
    text = 'prose {"a": 1, "b": {"c": "}nested} literal"}, "d": [1,2]} trailing'
    obj = _extract_first_json_object(text)
    assert obj == {"a": 1, "b": {"c": "}nested} literal"}, "d": [1, 2]}


def test_extract_first_json_object_none_when_no_brace():
    assert _extract_first_json_object("no json here") is None


def test_try_parse_json_direct_and_fallback():
    assert _try_parse_json('{"x": 1}') == {"x": 1}
    assert _try_parse_json("prose then {\"x\": 2} more prose") == {"x": 2}
    assert _try_parse_json("") is None
    assert _try_parse_json("not json at all") is None


def test_shape_matches_requires_fields_and_types():
    good = _entry_answer()
    assert _shape_matches(good, ENTRY_SCHEMA) is True
    missing = dict(good)
    del missing["thesis"]
    assert _shape_matches(missing, ENTRY_SCHEMA) is False
    wrong_type = dict(good)
    wrong_type["conviction"] = "high"
    assert _shape_matches(wrong_type, ENTRY_SCHEMA) is False
    assert _shape_matches("not a dict", ENTRY_SCHEMA) is False


def test_shape_matches_review_schema():
    good = _review_answer()
    assert _shape_matches(good, REVIEW_SCHEMA) is True


def test_is_output_config_400_detects_and_rejects_others():
    assert _is_output_config_400(_Fake400("400: output_config.format unsupported")) is True

    class _NotIt(Exception):
        status_code = 400

    assert _is_output_config_400(_NotIt("unrelated 400 error")) is False
    assert _is_output_config_400(RuntimeError("some other failure")) is False


def test_cost_usd_uses_pricing_table():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000}
    cost = _cost_usd(usage, {"input": 4.0, "output": 20.0, "cache_read": 0.20})
    assert cost == pytest.approx(4.0 + 20.0 + 0.20)


def test_cost_usd_handles_missing_or_bad_usage():
    assert _cost_usd({}, {"input": 4.0, "output": 20.0, "cache_read": 0.20}) == 0.0
    assert _cost_usd(None, {"input": 4.0, "output": 20.0, "cache_read": 0.20}) == 0.0
    assert _cost_usd({"input_tokens": "bad"}, {"input": 4.0, "output": 20.0, "cache_read": 0.20}) == 0.0


def test_validate_entry_pure_clamping(cfg):
    candidates = _candidates()
    fields, reasons = _validate_entry(_entry_answer(conviction=2.0, qty=1), candidates, cfg)
    assert fields["conviction"] == 1.0
    assert fields["qty"] == 1
    assert fields["clamped"] is False
    assert reasons == []


def test_validate_review_pure_clamping(cfg):
    fields, reasons = _validate_review(_review_answer(action="trim", trim_fraction=1.5), cfg)
    assert fields["action"] == "exit"  # >= 1 clamps up to exit
    assert fields["trim_fraction"] == 0.0


# ─────────────────────────────────────────────────────────────────────────
# OP-12: prompt-cache WRITES are priced in the decider cost estimate
# ─────────────────────────────────────────────────────────────────────────

from types import SimpleNamespace as _OP12NS  # noqa: E402

from agent.dpof.decider import _ANTHROPIC_PRICING as _OP12_ANT, _OPENAI_PRICING as _OP12_OAI  # noqa: E402
from agent.dpof.decider import _anthropic_response_to_dict as _op12_resp_to_dict  # noqa: E402


def test_op12_cache_write_priced_at_5_per_mtok():
    # 1000*4 + 500*20 + 2000*0.20 + 4000*5, all per-MTok
    # = 0.004 + 0.010 + 0.0004 + 0.020 = 0.0344
    usage = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 4000,
    }
    assert _cost_usd(usage, _OP12_ANT) == pytest.approx(0.0344)


def test_op12_missing_creation_key_keeps_previous_total():
    usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 2000}
    # previous golden total: 0.004 + 0.010 + 0.0004
    assert _cost_usd(usage, _OP12_ANT) == pytest.approx(0.0144)


def test_op12_creation_tokens_tolerate_none_and_bad_types():
    base = {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 2000}
    expected = pytest.approx(0.0144)
    assert _cost_usd({**base, "cache_creation_input_tokens": None}, _OP12_ANT) == expected
    assert _cost_usd({**base, "cache_creation_input_tokens": "bad"}, _OP12_ANT) == expected
    assert _cost_usd({**base, "cache_creation_input_tokens": True}, _OP12_ANT) == expected


def test_op12_openai_pricing_unchanged_even_with_creation_key():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 4000,
    }
    without = {k: v for k, v in usage.items() if k != "cache_creation_input_tokens"}
    assert _cost_usd(usage, _OP12_OAI) == _cost_usd(without, _OP12_OAI)
    assert _cost_usd(usage, _OP12_OAI) == pytest.approx(
        1000 / 1e6 * 2.0 + 500 / 1e6 * 10.0 + 2000 / 1e6 * 0.20
    )


def test_op12_response_to_dict_captures_creation_tokens():
    resp = _OP12NS(
        content=[_OP12NS(type="text", text="hi")],
        usage=_OP12NS(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=40,
        ),
        model="claude-opus-5-5",
        stop_reason="end_turn",
    )
    out = _op12_resp_to_dict(resp, "fallback")
    assert out["usage"]["cache_creation_input_tokens"] == 40
    assert out["usage"]["cache_read_input_tokens"] == 30


def test_op12_response_to_dict_tolerates_absent_creation_field():
    resp = _OP12NS(
        content=[_OP12NS(type="text", text="hi")],
        usage=_OP12NS(input_tokens=10, output_tokens=20, cache_read_input_tokens=0),
        model="claude-opus-5-5",
        stop_reason="end_turn",
    )
    out = _op12_resp_to_dict(resp, "fallback")
    assert out["usage"]["cache_creation_input_tokens"] is None
    assert _cost_usd(out["usage"], _OP12_ANT) == pytest.approx(10 / 1e6 * 4.0 + 20 / 1e6 * 20.0)
