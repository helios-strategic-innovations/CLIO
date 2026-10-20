"""
CLIO DPOF — judgment layer: the decision model itself (Opus 5.5 primary,
GPT-6 Sol fallback).

Operator directive (2026-09-24, AMENDED later that day): NO mechanical
trading, and — per the AMENDMENTS section of DPOF_V2_SPEC.md, which
supersedes the earlier "hard size caps stay in code" wording — NO hard
size caps either. Entries AND exits, INCLUDING how many contracts/spreads
to trade, are judged by CLIO's own decision model (`EntryDecision.qty`).
Code enforces only two things: the one kept timing rail (15:50 ET flatten)
and a single solvency bound — a `qty` above the chosen candidate's
`EntryCandidate.max_affordable_qty` (= floor(capital_available /
unit_premium), computed by the engine) is clamped DOWN, never up, and a
`qty` below 1 on "enter" is treated as a pass. This module is the thing
that actually calls the model and turns its answer into a validated,
code-clampable decision. It never trades on its own — the engine applies
`EntryDecision`/`ReviewDecision` under the usual gates.

Provider order: Anthropic (`cfg.decision_model`, default claude-opus-5-5) ->
on error/timeout/invalid-JSON/circuit-open -> OpenAI (`cfg.fallback_model`,
default gpt-6-sol) -> `DecisionUnavailable`. Each provider has its own
circuit breaker (`cfg.decider_circuit_failures` / `_cooldown_seconds`,
same shape as jev.py's JevClient) and its own deadline
(`cfg.decision_timeout_seconds` / `cfg.fallback_timeout_seconds`).
A shared daily cap (`cfg.max_decisions_per_day`, counted per ET day across
both providers) is persisted to `<data_dir>/decider_state.json` so a
restart mid-day doesn't reset the cost guard.

Reused from jev.py's patterns (deadline enforcement, circuit breaker,
header hygiene, never logging secrets): the `_ProviderStatus`/circuit-open/
half-open shape, `_MAX_RESPONSE_BYTES` + a size cap ahead of any JSON parse
on the raw-httpx (OpenAI) leg, `follow_redirects=False`, and a `_redact()`
helper that strips `Bearer <token>` and any configured secret before an
error message is logged.

The model can never exceed what code offers: every field is validated and
clamped in `_validate_entry`/`_validate_review` before an `EntryDecision`/
`ReviewDecision` is returned — an out-of-menu `candidate_key`, an
out-of-range number, an unrecognized enum, or a `qty` above the chosen
candidate's solvency bound never reaches the caller unfiltered.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

import anthropic
import httpx

from agent.dpof.attempt_log import AttemptLog
from agent.dpof.claim_check import check_claims
from agent.dpof.config import DPOFConfig
from agent.dpof.prompts import CHARTER, build_entry_message, build_review_message
from agent.dpof.schema_prompt import schema_instruction

logger = logging.getLogger("clio.dpof.decider")


def _collapse(text) -> str:
    """Printable, single-spaced text: control/ANSI bytes and newlines become spaces, runs collapse."""
    raw = "" if text is None else str(text)
    cleaned = "".join(ch if ch.isprintable() else " " for ch in raw)
    return " ".join(cleaned.split())


def _one_line(text, limit: int = 300) -> str:
    """Collapse model text to one printable line of at most `limit` chars.

    Non-printable characters (newlines, tabs, control/ANSI bytes) become a
    single space, whitespace runs collapse to one space, and anything longer
    than `limit` is cut to `limit` characters ending in an ellipsis."""
    collapsed = _collapse(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(limit - 1, 0)].rstrip() + "…"


_THESIS_LIMIT = 1200


def thesis_full(text, limit: int = _THESIS_LIMIT) -> str:
    """The persisted form of a pass thesis: printable, single-spaced, at most `limit` chars.

    Short text is returned unchanged. Longer text is cut at the last sentence
    boundary ('. ' or '; ') that keeps at least 40% of the cap, with no ellipsis
    (the sentence ends cleanly). Otherwise it is cut at the last space and ends
    in an ellipsis. It is never cut mid-word unless the text has no space at all."""
    collapsed = _collapse(text)
    if len(collapsed) <= limit:
        return collapsed
    head = collapsed[: max(limit - 1, 0)]
    cut = max(head.rfind(". "), head.rfind("; "))
    if cut >= 0 and cut + 1 >= 0.4 * limit:
        return head[: cut + 1]
    space = head.rfind(" ")
    if space > 0:
        return head[:space].rstrip() + "…"
    return head + "…"


def _pass_label(ctx: dict, candidates: list) -> str:
    """Short nomination label for the PASS log line.

    The engine's entry ctx carries no top-level underlying/direction -- those live on
    the offered EntryCandidate objects -- so the label is the unique `underlying:direction`
    pairs of the candidates (at most 4), falling back to identifying ctx keys, then "n/a"."""
    pairs: list[str] = []
    for cand in candidates or []:
        get = cand.get if isinstance(cand, dict) else (lambda k, _c=cand: getattr(_c, k, None))
        und, direction = get("underlying"), get("direction")
        if und or direction:
            pair = f"{_one_line(und or '?', 20)}:{_one_line(direction or '?', 20)}"
            if pair not in pairs:
                pairs.append(pair)
        if len(pairs) >= 4:
            break
    if pairs:
        return ",".join(pairs)
    parts = []
    for key in ("underlying", "symbol", "direction"):
        val = (ctx or {}).get(key)
        if val is not None and val != "":
            parts.append(f"{key}={_one_line(val, 40)}")
    return "|".join(parts) if parts else "n/a"

_MIN_CALL_BUDGET_SECONDS = 0.05
_MAX_RESPONSE_BYTES = 1 * 1024 * 1024  # 1 MiB — a decider answer is never legitimately this large
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_STATE_FILE = "decider_state.json"
_ATTEMPT_FILE = "decider_attempts.jsonl"
_WARN_INTERVAL_SECONDS = 60.0  # primary-failure WARNING: at most one per error class per minute

# Per-1M-token USD pricing used only for the advisory est_cost_today_usd in
# status(). claude-opus-5-5: $4 input / $20 output / $0.20 cache-read.
# gpt-6-sol: $2 input / $10 output / $0.20 cached (per spec).
_ANTHROPIC_PRICING = {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0}
_OPENAI_PRICING = {"input": 2.0, "output": 10.0, "cache_read": 0.20}


# ─────────────────────────────────────────────────────────────────────────
# Public data contracts
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class EntryCandidate:
    """Built by the engine from contracts.choose_candidates(...)."""
    key: str = ""                        # stable label offered to the model, e.g. "C1"
    underlying: str = ""
    direction: str = ""                  # Direction value ("bullish"/"bearish")
    structure: str = "single"            # "single" | "debit_vertical"
    legs: list = field(default_factory=list)   # OrderLeg dicts
    expiry: str = ""                     # YYYY-MM-DD
    sessions_to_expiry: int = 0
    strike_desc: str = ""                # e.g. "580C" or "580/585C"
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    spread_pct: Optional[float] = None
    delta: Optional[float] = None
    theta: Optional[float] = None
    iv: Optional[float] = None
    open_interest: Optional[int] = None
    unit_premium: Optional[float] = None       # $ per 1 contract/spread at mid
    max_affordable_qty: int = 0          # solvency bound = floor(capital_available / unit_premium); NOT a risk cap


@dataclass
class EntryDecision:
    action: str = "pass"                 # "enter" | "pass"
    candidate_key: Optional[str] = None  # must be one of the offered keys when action == "enter"
    conviction: float = 0.0              # 0..1
    qty: int = 0                         # contracts/spreads the MODEL chose; >=1 on "enter" (its own sizing judgment)
    clamped: bool = False                # True when code clamped qty DOWN to the candidate's max_affordable_qty
    thesis: str = ""
    invalidation: str = ""
    recheck_minutes: int = 5             # when to deliberate again if "pass"
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def thesis_line(self) -> str:
        """The model's thesis as printable text of <= 1,200 chars, persisted on a pass row.

        The PASS log line still uses the 300-char one-liner (`_one_line`); only the
        persisted form is the longer `thesis_full`."""
        return thesis_full(self.thesis)


@dataclass
class ReviewDecision:
    action: str = "hold"                 # "hold" | "trim" | "exit"
    trim_fraction: float = 0.0           # (0,1) when action == "trim"
    urgency: str = "normal"              # "normal" | "urgent"
    thesis_status: str = "intact"        # "intact" | "weakening" | "broken"
    reason: str = ""
    next_review_minutes: int = 5         # clamped to [review_min_minutes, review_max_minutes]
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    raw: dict = field(default_factory=dict)


class DecisionUnavailable(Exception):
    """All providers failed / open circuits / daily cap reached."""


# ─────────────────────────────────────────────────────────────────────────
# JSON schemas — additionalProperties: false, every field required.
#
# NO `minimum`/`maximum` (or any other numeric constraint) on any field here.
# Anthropic's structured outputs (`output_config.format`) do NOT support
# numeric constraints — they're listed under "Not supported" in the JSON
# Schema Limitations for the feature, alongside minLength/maxLength/
# multipleOf. The SDK's own schema transform that strips/relocates
# unsupported constraints into the description (`anthropic.lib._parse.
# _transform.transform_schema`) only runs inside `client.messages.parse()`
# (converting a pydantic-derived schema) — NOT when a raw dict schema is
# handed to `output_config.format.schema` on `client.messages.create()`,
# which is what this module does (see `_build_anthropic_payload`). A
# `minimum`/`maximum` here would therefore be sent to Anthropic verbatim,
# unvalidated by the SDK and unsupported by the API's own documented
# contract for this call shape (confirmed by reading transform_schema's
# only two call sites in `anthropic` 0.125.0). The numeric bound is instead
# stated in the `description` (a supported keyword) so the model still sees
# it, and is ENFORCED, unconditionally, in code — `_validate_entry`/
# `_validate_review` below clamp every numeric field regardless of what the
# model returns. OpenAI's Responses API `text.format` (json_schema, strict)
# is a separate request (see `_call_openai`) that DOES support minimum/
# maximum, but this module deliberately uses ONE schema shape for both
# providers rather than special-casing what OpenAI happens to accept.
# ─────────────────────────────────────────────────────────────────────────

ENTRY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["enter", "pass"]},
        "candidate_key": {
            "type": "string",
            "description": "One of the offered candidate keys (e.g. 'C1') when action is 'enter'; empty string when 'pass'.",
        },
        "conviction": {"type": "number", "description": "0.0 to 1.0 — your conviction in this trade."},
        "qty": {
            "type": "integer",
            "description": "Number of contracts/spreads to trade when action is 'enter' (your own sizing judgment — no fixed percentage limit; the only hard bound is the offered candidate's max_affordable_qty, shown per-candidate). Must be >= 1 on 'enter'. 0 when action is 'pass'.",
        },
        "thesis": {"type": "string"},
        "invalidation": {"type": "string"},
        "recheck_minutes": {
            "type": "integer",
            "description": "When to deliberate again if 'pass', in minutes. Must be between 1 and 120.",
        },
    },
    "required": ["action", "candidate_key", "conviction", "qty", "thesis", "invalidation", "recheck_minutes"],
    "additionalProperties": False,
}

REVIEW_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["hold", "trim", "exit"]},
        "trim_fraction": {
            "type": "number",
            "description": "Fraction of the position to trim, strictly between 0.0 and 1.0. Only meaningful when action is 'trim'.",
        },
        "urgency": {"type": "string", "enum": ["normal", "urgent"]},
        "thesis_status": {"type": "string", "enum": ["intact", "weakening", "broken"]},
        "reason": {"type": "string"},
        "next_review_minutes": {
            "type": "integer",
            "description": "When to review this position again, in minutes. Must be between 1 and 120.",
        },
    },
    "required": ["action", "trim_fraction", "urgency", "thesis_status", "reason", "next_review_minutes"],
    "additionalProperties": False,
}


# ─────────────────────────────────────────────────────────────────────────
# PURE — validation / clamping ("the model can never exceed what code
# offers")
# ─────────────────────────────────────────────────────────────────────────

def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except (TypeError, ValueError):
        v = default
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, v))


def _floor_qty(x: Any) -> Optional[int]:
    """Floor a numeric qty to int. None for NaN/inf/non-numeric/bool — the
    caller then forces the safest interpretation ("invalid_qty" -> pass),
    same shape as every other unusable-field case in this module."""
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return math.floor(v)


def _validate_entry(parsed: dict, candidates: list, cfg: DPOFConfig) -> tuple[dict, list[str]]:
    """No hard size caps (operator amendment, 2026-09-24): the model's `qty`
    IS the size. The only code bound is solvency — `qty` above the chosen
    candidate's `max_affordable_qty` is clamped down (logged via `clamped`,
    not treated as a failure); `qty` < 1 on "enter" (including a NaN/inf/
    non-numeric/missing value) is treated the same as an invalid candidate:
    forced to "pass"."""
    reasons: list[str] = []
    action = parsed.get("action")
    if action not in ("enter", "pass"):
        reasons.append(f"invalid_action:{action!r}")
        action = "pass"

    candidate_key = parsed.get("candidate_key")
    offered = {getattr(c, "key", None): c for c in (candidates or [])}
    if action == "enter":
        if not candidate_key or candidate_key not in offered:
            reasons.append("invalid_candidate")
            action = "pass"
            candidate_key = None
    else:
        candidate_key = None

    conviction = _clamp01(parsed.get("conviction"))
    thesis = str(parsed.get("thesis") or "")
    invalidation = str(parsed.get("invalidation") or "")
    recheck_minutes = _clamp_int(
        parsed.get("recheck_minutes"), cfg.review_min_minutes, cfg.review_max_minutes, cfg.review_interval_minutes
    )

    clamped = False
    if action == "enter":
        qty = _floor_qty(parsed.get("qty"))
        if qty is None or qty < 1:
            reasons.append("invalid_qty")
            action = "pass"
            candidate_key = None
            qty = 0
        else:
            candidate = offered.get(candidate_key)
            max_affordable = int(getattr(candidate, "max_affordable_qty", 0) or 0) if candidate is not None else 0
            if qty > max_affordable:
                clamped = True
                logger.info(
                    f"DPOF decider: entry qty {qty} for candidate {candidate_key!r} exceeds its "
                    f"max_affordable_qty {max_affordable}; clamping down (solvency bound, not a failure)"
                )
                qty = max_affordable
    else:
        qty = 0

    return {
        "action": action,
        "candidate_key": candidate_key,
        "conviction": conviction,
        "qty": qty,
        "clamped": clamped,
        "thesis": thesis,
        "invalidation": invalidation,
        "recheck_minutes": recheck_minutes,
    }, reasons


def _validate_review(parsed: dict, cfg: DPOFConfig) -> tuple[dict, list[str]]:
    reasons: list[str] = []
    action = parsed.get("action")
    if action not in ("hold", "trim", "exit"):
        reasons.append(f"invalid_action:{action!r}")
        action = "hold"

    try:
        tf = float(parsed.get("trim_fraction"))
        if not math.isfinite(tf):
            tf = 0.0
    except (TypeError, ValueError):
        tf = 0.0

    if action == "trim":
        if tf <= 0.0:
            reasons.append("trim_fraction_le_0")
            action = "hold"
            tf = 0.0
        elif tf >= 1.0:
            reasons.append("trim_fraction_ge_1")
            action = "exit"
            tf = 0.0
    else:
        tf = 0.0

    urgency = parsed.get("urgency")
    if urgency not in ("normal", "urgent"):
        reasons.append(f"invalid_urgency:{urgency!r}")
        urgency = "normal"

    thesis_status = parsed.get("thesis_status")
    if thesis_status not in ("intact", "weakening", "broken"):
        reasons.append(f"invalid_thesis_status:{thesis_status!r}")
        thesis_status = "intact"

    reason_text = str(parsed.get("reason") or "")
    next_review_minutes = _clamp_int(
        parsed.get("next_review_minutes"), cfg.review_min_minutes, cfg.review_max_minutes, cfg.review_interval_minutes
    )

    return {
        "action": action,
        "trim_fraction": tf,
        "urgency": urgency,
        "thesis_status": thesis_status,
        "reason": reason_text,
        "next_review_minutes": next_review_minutes,
    }, reasons


# ─────────────────────────────────────────────────────────────────────────
# PURE — JSON extraction / shape checking
# ─────────────────────────────────────────────────────────────────────────

def _extract_first_json_object(text: str) -> Optional[dict]:
    """Scan for the first balanced {...} object, respecting quoted strings.

    Used only on the downgrade path (Anthropic rejected output_config with a
    400) and as a last-resort parse for a malformed/fake response, so the
    text may have prose or markdown fencing around the JSON.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _try_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return _extract_first_json_object(text)


def _shape_matches(parsed: Any, schema: dict) -> bool:
    """Rudimentary structural check (required keys present, gross type match).

    This is a defense-in-depth check, not a full JSON-schema validator — the
    provider's own structured-output enforcement (or the strict-mode
    fallback) is what actually guarantees the shape on the happy path; this
    catches a bad downgrade-path extraction or a malformed fake in tests.
    """
    if not isinstance(parsed, dict):
        return False
    for key in schema.get("required", []):
        if key not in parsed:
            return False
    props = schema.get("properties", {})
    for key, val in parsed.items():
        spec = props.get(key)
        if not isinstance(spec, dict):
            continue
        t = spec.get("type")
        if t == "string" and not isinstance(val, str):
            return False
        if t == "number" and (isinstance(val, bool) or not isinstance(val, (int, float))):
            return False
        if t == "integer" and (isinstance(val, bool) or not isinstance(val, int)):
            return False
    return True


def _is_output_config_400(exc: Exception) -> bool:
    """True if `exc` looks like an Anthropic 400 rejecting output_config/format."""
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    msg = str(exc)
    mentions = "output_config" in msg or "output config" in msg.lower()
    if status == 400 and mentions:
        return True
    if status is None and "400" in msg and mentions:
        return True
    return False


def _cost_usd(usage: dict, pricing: dict) -> float:
    if not isinstance(usage, dict):
        return 0.0

    def _n(v: Any) -> float:
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

    inp = _n(usage.get("input_tokens"))
    out = _n(usage.get("output_tokens"))
    cached = _n(usage.get("cache_read_input_tokens"))
    cost = (inp / 1_000_000.0) * pricing["input"] + (cached / 1_000_000.0) * pricing["cache_read"] + (out / 1_000_000.0) * pricing["output"]
    # Prompt-cache WRITES (5-min tier, 1.25x base input). Only priced when the
    # pricing table defines a cache_write rate, so OpenAI totals stay identical.
    if "cache_write" in pricing:
        created = _n(usage.get("cache_creation_input_tokens"))
        cost += (created / 1_000_000.0) * pricing["cache_write"]
    return cost


def _redact(msg: str, *secrets: str) -> str:
    if not msg:
        return msg
    out = _BEARER_RE.sub("Bearer ***", msg)
    for s in secrets:
        if s:
            out = out.replace(s, "***")
    return out


def _redact_anthropic(msg: str) -> str:
    try:
        from agent.config import config as _clio_cfg  # lazy import

        return _redact(msg, _clio_cfg.anthropic_api_key)
    except Exception:
        return _redact(msg)


def _anthropic_stop_reason_note(resp: Any) -> str:
    """"" when there's nothing worth flagging; otherwise a short diagnostic
    tag for a response that structurally can't have produced a usable
    decision — surfaced in the decider's error string so "invalid_json" (the
    generic JSON-parse-failure bucket) doesn't hide WHY the text was empty or
    truncated. `stop_details` is populated only when stop_reason == "refusal"
    (null otherwise) — always guarded before reading, per the API's own
    documented contract for that field."""
    stop_reason = getattr(resp, "stop_reason", None)
    if stop_reason == "max_tokens":
        return "max_tokens"
    if stop_reason == "refusal":
        details = getattr(resp, "stop_details", None)
        category = getattr(details, "category", None) if details is not None else None
        return f"refusal:{category}" if category else "refusal"
    return ""


def _anthropic_response_to_dict(resp: Any, default_model: str) -> dict:
    text_parts: list[str] = []
    for block in getattr(resp, "content", None) or []:
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", "") or "")
        # thinking blocks are intentionally ignored — Opus 5.5 always thinks
    usage_obj = getattr(resp, "usage", None)
    usage: dict = {}
    if usage_obj is not None:
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
            "cache_read_input_tokens": getattr(usage_obj, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(usage_obj, "cache_creation_input_tokens", None),
        }
    return {
        "text": "".join(text_parts),
        "usage": usage,
        "model": getattr(resp, "model", default_model) or default_model,
        "stop_note": _anthropic_stop_reason_note(resp),
    }


# ─────────────────────────────────────────────────────────────────────────
# I/O helpers — state file (daily cap), sync (always via run_in_executor)
# ─────────────────────────────────────────────────────────────────────────

def _error_class(error: str) -> str:
    """Short, non-secret class label for an (already redacted) provider error."""
    if not error:
        return ""
    return error.split(":", 1)[0].strip()[:60] or "unknown"


def _attempt_status(error: str) -> str:
    if not error:
        return "ok"
    if error.startswith("circuit_open"):
        return "circuit_open"
    if error.startswith("timeout"):
        return "timeout"
    if error.startswith("invalid_json"):
        return "invalid_shape"
    return "error"


def _tok(value: Any) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _provider_cost(provider: str, usage: Any) -> float:
    if provider == "anthropic":
        return _cost_usd(usage, _ANTHROPIC_PRICING)
    if provider == "openai":
        return _cost_usd(usage, _OPENAI_PRICING)
    return 0.0


def _load_json_sync(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            raw = f.read()
        if not raw.strip():
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _atomic_write_json_sync(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


@dataclass
class _ProviderStatus:
    circuit_open_until: Optional[float] = None
    consecutive_failures: int = 0
    last_error: str = ""
    last_latency_ms: Optional[int] = None
    name: str = ""
    tripped: bool = False           # set when the circuit opens, cleared on success (for the CLOSED log)


# ─────────────────────────────────────────────────────────────────────────
# DPOFDecider
# ─────────────────────────────────────────────────────────────────────────

class DPOFDecider:
    def __init__(
        self,
        cfg: DPOFConfig,
        *,
        anthropic_transport: Optional[Callable[[dict], Awaitable[dict]]] = None,
        openai_transport: Optional[Callable[[dict], Awaitable[dict]]] = None,
        clock: Callable[[], float] = time.time,
        data_dir: Optional[str] = None,
        record: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> None:
        self.cfg = cfg
        self._anthropic_transport = anthropic_transport
        self._openai_transport = openai_transport
        self._clock = clock
        self.data_dir = data_dir or cfg.data_dir
        self._record = record

        self._anthropic_status = _ProviderStatus(name="anthropic")
        self._openai_status = _ProviderStatus(name="openai")
        self._anthropic_format_downgraded = self._gateway_needs_prompt_schema()
        self._attempt_log = self._make_attempt_log()
        self._warn_last: dict[str, float] = {}

        self._anthropic_client: Optional[anthropic.Anthropic] = None
        self._anthropic_client_key: Optional[str] = None
        self._anthropic_client_timeout: Optional[float] = None
        self._openai_client: Optional[httpx.AsyncClient] = None

        self._state: Optional[dict] = None
        self._state_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def update_config(self, cfg: DPOFConfig) -> None:
        self.cfg = cfg
        self._attempt_log = self._make_attempt_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decide_entry(self, ctx: dict, candidates: list) -> EntryDecision:
        decision = await self._decide("entry", ctx or {}, candidates=candidates or [])
        assert isinstance(decision, EntryDecision)
        if decision.action == "pass":
            logger.info(
                "DPOF entry decision PASS: conviction=%s recheck_min=%s model=%s provider=%s candidates=%d nomination=%s thesis=%s",
                decision.conviction,
                decision.recheck_minutes,
                _one_line(decision.model or "n/a", 80),
                _one_line(decision.provider or "n/a", 80),
                len(candidates or []),
                _pass_label(ctx or {}, candidates or []),
                _one_line(decision.thesis),
            )
        return decision

    async def review_position(self, ctx: dict, position: dict) -> ReviewDecision:
        decision = await self._decide("review", ctx or {}, position=position or {})
        assert isinstance(decision, ReviewDecision)
        return decision

    def status(self) -> dict:
        state = self._state or {"calls": 0, "cost_usd": 0.0}
        return {
            "providers": {
                "anthropic": {
                    "name": "anthropic",
                    "model": self.cfg.decision_model,
                    "circuit_open_until": self._anthropic_status.circuit_open_until,
                    "consecutive_failures": self._anthropic_status.consecutive_failures,
                    "last_error": self._anthropic_status.last_error,
                    "last_latency_ms": self._anthropic_status.last_latency_ms,
                },
                "openai": {
                    "name": "openai",
                    "model": self.cfg.fallback_model,
                    "circuit_open_until": self._openai_status.circuit_open_until,
                    "consecutive_failures": self._openai_status.consecutive_failures,
                    "last_error": self._openai_status.last_error,
                    "last_latency_ms": self._openai_status.last_latency_ms,
                },
            },
            "calls_today": int(state.get("calls", 0)),
            "est_cost_today_usd": round(float(state.get("cost_usd", 0.0)), 4),
        }

    async def close(self) -> None:
        if self._openai_client is not None:
            client, self._openai_client = self._openai_client, None
            try:
                await client.aclose()
            except Exception as e:
                logger.debug(f"DPOFDecider.close: openai client aclose failed: {type(e).__name__}: {e}")
        if self._anthropic_client is not None:
            client, self._anthropic_client = self._anthropic_client, None
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, client.close)
            except Exception as e:
                logger.debug(f"DPOFDecider.close: anthropic client close failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Daily cap / cost-guard state
    # ------------------------------------------------------------------

    def _state_path(self) -> str:
        return os.path.join(self.data_dir, _STATE_FILE)

    def _et_day_str(self) -> str:
        dt = datetime.fromtimestamp(self._clock(), tz=ZoneInfo("America/New_York"))
        return dt.strftime("%Y-%m-%d")

    async def _ensure_state_loaded(self) -> None:
        if self._state is not None:
            return
        async with self._state_lock:
            if self._state is not None:
                return
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _load_json_sync, self._state_path())
            today = self._et_day_str()
            if isinstance(data, dict) and data.get("day") == today:
                self._state = {
                    "day": today,
                    "calls": int(data.get("calls") or 0),
                    "cost_usd": float(data.get("cost_usd") or 0.0),
                }
            else:
                self._state = {"day": today, "calls": 0, "cost_usd": 0.0}

    def _roll_day_if_needed(self) -> bool:
        """Caller must hold self._state_lock."""
        today = self._et_day_str()
        if self._state is None or self._state.get("day") != today:
            self._state = {"day": today, "calls": 0, "cost_usd": 0.0}
            return True
        return False

    async def _finish_attempt(self, *, provider: str, usage: dict, failed: list = ()) -> None:
        async with self._state_lock:
            self._roll_day_if_needed()
            cost = _provider_cost(provider, usage)
            # Failed attempts that still carried usage (e.g. invalid JSON) cost real money:
            # their cost joins the day total, but they are not counted as a decision.
            for failed_provider, failed_usage in failed:
                cost += _provider_cost(failed_provider, failed_usage)
            self._state["calls"] = int(self._state.get("calls", 0)) + 1
            self._state["cost_usd"] = float(self._state.get("cost_usd", 0.0)) + cost
            snapshot = dict(self._state)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _atomic_write_json_sync, self._state_path(), snapshot)

    # ------------------------------------------------------------------
    # Circuit breaker bookkeeping (shared shape between both providers)
    # ------------------------------------------------------------------

    def _record_provider_success(self, status: _ProviderStatus) -> None:
        was_tripped = status.tripped
        status.consecutive_failures = 0
        status.circuit_open_until = None
        status.last_error = ""
        status.tripped = False
        if was_tripped:
            logger.info("DPOF decider circuit CLOSED provider=%s", status.name)

    def _record_provider_failure(self, status: _ProviderStatus, error: str) -> None:
        status.consecutive_failures += 1
        status.last_error = error
        if status.consecutive_failures >= max(1, self.cfg.decider_circuit_failures):
            # A half-open probe cleared circuit_open_until before the call, so a failed
            # probe logs OPEN again; a failure while already open does not.
            was_open = status.circuit_open_until is not None
            cooldown = max(0.0, self.cfg.decider_circuit_cooldown_seconds)
            status.circuit_open_until = self._clock() + cooldown
            status.tripped = True
            if not was_open:
                logger.info(
                    "DPOF decider circuit OPEN provider=%s failures=%d cooldown=%ss error_class=%s",
                    status.name, status.consecutive_failures, cooldown, _error_class(error) or "unknown",
                )

    # ------------------------------------------------------------------
    # Anthropic (primary)
    # ------------------------------------------------------------------

    def _get_anthropic_client(self) -> anthropic.Anthropic:
        from agent.config import config as clio_config  # lazy import

        timeout = self.cfg.decision_timeout_seconds
        retries = max(0, min(3, int(self.cfg.decider_sdk_max_retries)))
        kwargs = clio_config.anthropic_client_kwargs(timeout=timeout, max_retries=retries)
        sig = (kwargs.get("api_key"), kwargs.get("base_url"), timeout, retries)
        if self._anthropic_client is None or self._anthropic_client_key != sig:
            self._anthropic_client = anthropic.Anthropic(**kwargs)
            self._anthropic_client_key = sig
            self._anthropic_client_timeout = timeout
        return self._anthropic_client

    def _gateway_needs_prompt_schema(self) -> bool:
        """True when routed via a gateway that ignores output_config.format (OP-05)."""
        try:
            from agent.config import config as clio_config  # lazy import

            return bool(getattr(clio_config, "anthropic_gateway", False))
        except Exception:
            return False

    def _build_anthropic_payload(self, user_text: str, schema: dict, *, use_format: bool) -> dict:
        cfg = self.cfg
        output_config: dict = {"effort": cfg.decision_effort}
        if use_format:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        else:
            # Gateway / downgraded path: the schema must be stated in the prompt.
            user_text = user_text + "\n\n" + schema_instruction(schema)
        return {
            "model": cfg.decision_model,
            "max_tokens": cfg.decision_max_tokens,
            "system": [{"type": "text", "text": CHARTER, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_text}],
            "output_config": output_config,
        }

    def _sync_anthropic_call(self, payload: dict) -> dict:
        """Runs on the executor thread pool — never on the event loop."""
        client = self._get_anthropic_client()
        kwargs = dict(payload)
        try:
            resp = client.messages.create(**kwargs)
        except TypeError as e:
            # Defensive: an older installed SDK without a native output_config
            # kwarg. Retry once via extra_body (same request shape on the wire).
            if "output_config" not in str(e):
                raise
            oc = kwargs.pop("output_config", None)
            resp = client.messages.create(extra_body={"output_config": oc}, **kwargs)
        return _anthropic_response_to_dict(resp, self.cfg.decision_model)

    async def _run_anthropic_transport(self, payload: dict, timeout: float) -> dict:
        t = max(0.01, timeout)
        if self._anthropic_transport is not None:
            return await asyncio.wait_for(self._anthropic_transport(payload), timeout=t)
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(loop.run_in_executor(None, lambda: self._sync_anthropic_call(payload)), timeout=t)

    async def _call_anthropic(self, user_text: str, schema: dict, deadline_at: float) -> tuple[Optional[dict], dict]:
        cfg = self.cfg
        status = self._anthropic_status

        if status.circuit_open_until is not None:
            if self._clock() < status.circuit_open_until:
                return None, {"error": "circuit_open", "model": cfg.decision_model, "latency_ms": None, "usage": {}, "raw_text": ""}
            status.circuit_open_until = None  # half-open: let exactly this call through

        budget = deadline_at - self._clock()
        if budget <= _MIN_CALL_BUDGET_SECONDS:
            self._record_provider_failure(status, "no_budget")
            return None, {"error": "no_budget", "model": cfg.decision_model, "latency_ms": None, "usage": {}, "raw_text": ""}

        use_format = not self._anthropic_format_downgraded
        payload = self._build_anthropic_payload(user_text, schema, use_format=use_format)

        t0 = self._clock()
        try:
            result = await self._run_anthropic_transport(payload, budget)
        except asyncio.TimeoutError:
            self._record_provider_failure(status, "timeout")
            return None, {"error": "timeout", "model": cfg.decision_model, "latency_ms": None, "usage": {}, "raw_text": ""}
        except Exception as e:
            if use_format and _is_output_config_400(e):
                if not self._anthropic_format_downgraded:
                    self._anthropic_format_downgraded = True
                    logger.warning(
                        "DPOF decider: Anthropic rejected output_config.format (HTTP 400); "
                        "downgrading to prompt-only JSON + text extraction for the rest of this process."
                    )
                budget2 = deadline_at - self._clock()
                if budget2 <= _MIN_CALL_BUDGET_SECONDS:
                    self._record_provider_failure(status, "no_budget_after_downgrade")
                    return None, {"error": "no_budget_after_downgrade", "model": cfg.decision_model, "latency_ms": None, "usage": {}, "raw_text": ""}
                payload2 = self._build_anthropic_payload(user_text, schema, use_format=False)
                t0 = self._clock()
                try:
                    result = await self._run_anthropic_transport(payload2, budget2)
                except asyncio.TimeoutError:
                    self._record_provider_failure(status, "timeout_after_downgrade")
                    return None, {"error": "timeout_after_downgrade", "model": cfg.decision_model, "latency_ms": None, "usage": {}, "raw_text": ""}
                except Exception as e2:
                    err = _redact_anthropic(f"{type(e2).__name__}: {e2}")
                    self._record_provider_failure(status, err)
                    return None, {"error": err, "model": cfg.decision_model, "latency_ms": None, "usage": {}, "raw_text": ""}
            else:
                err = _redact_anthropic(f"{type(e).__name__}: {e}")
                self._record_provider_failure(status, err)
                return None, {"error": err, "model": cfg.decision_model, "latency_ms": None, "usage": {}, "raw_text": ""}

        latency_ms = int((self._clock() - t0) * 1000)
        status.last_latency_ms = latency_ms

        text = result.get("text", "") if isinstance(result, dict) else ""
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        model_used = (result.get("model") if isinstance(result, dict) else None) or cfg.decision_model

        parsed = _try_parse_json(text)
        if parsed is None or not _shape_matches(parsed, schema):
            # stop_note ("" for an ordinary end_turn) surfaces WHY the text
            # was unusable when it's diagnosable — max_tokens truncation or a
            # safety refusal look identical to a malformed answer otherwise.
            # Missing on an injected test transport's plain dict (no attr) —
            # .get() degrades to "", matching the pre-existing behavior.
            stop_note = result.get("stop_note", "") if isinstance(result, dict) else ""
            reason = f"invalid_json:{stop_note}" if stop_note else "invalid_json"
            self._record_provider_failure(status, reason)
            return None, {"error": reason, "model": model_used, "latency_ms": latency_ms, "usage": usage, "raw_text": text}

        self._record_provider_success(status)
        return parsed, {"error": "", "model": model_used, "latency_ms": latency_ms, "usage": usage, "raw_text": text}

    # ------------------------------------------------------------------
    # OpenAI (fallback)
    # ------------------------------------------------------------------

    def _get_openai_client(self) -> httpx.AsyncClient:
        if self._openai_client is None or self._openai_client.is_closed:
            self._openai_client = httpx.AsyncClient(follow_redirects=False)
        return self._openai_client

    def _openai_response_to_dict(self, body: Any) -> dict:
        text = ""
        if isinstance(body, dict):
            ot = body.get("output_text")
            if isinstance(ot, str) and ot:
                text = ot
            else:
                chunks: list[str] = []
                for item in body.get("output", []) or []:
                    if not isinstance(item, dict):
                        continue
                    for c in item.get("content", []) or []:
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            t = c.get("text")
                            if isinstance(t, str):
                                chunks.append(t)
                text = "".join(chunks)
        usage_raw = body.get("usage") if isinstance(body, dict) else None
        usage: dict = {}
        if isinstance(usage_raw, dict):
            cached = 0
            details = usage_raw.get("input_tokens_details")
            if isinstance(details, dict):
                cached = details.get("cached_tokens") or 0
            usage = {
                "input_tokens": usage_raw.get("input_tokens"),
                "output_tokens": usage_raw.get("output_tokens"),
                "cache_read_input_tokens": cached,
            }
        model = (body.get("model") if isinstance(body, dict) else None) or self.cfg.fallback_model
        return {"text": text, "usage": usage, "model": model}

    async def _openai_http_call(self, payload: dict, timeout: float) -> dict:
        cfg = self.cfg
        client = self._get_openai_client()
        headers = {"Authorization": f"Bearer {cfg.openai_api_key}", "Content-Type": "application/json"}
        url = f"{cfg.openai_base_url}/responses"
        resp = await client.post(url, headers=headers, json=payload, timeout=timeout)

        oversized = False
        cl = resp.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > _MAX_RESPONSE_BYTES:
                    oversized = True
            except (TypeError, ValueError):
                pass
        if not oversized and len(resp.content) > _MAX_RESPONSE_BYTES:
            oversized = True
        if oversized:
            raise ValueError("openai_response_oversized")

        if resp.status_code != 200:
            detail = ""
            try:
                j = resp.json()
                if isinstance(j, dict):
                    err = j.get("error")
                    if isinstance(err, dict):
                        detail = str(err.get("message") or "")
                    elif isinstance(err, str):
                        detail = err
            except Exception:
                pass
            raise _OpenAIHTTPError(resp.status_code, detail[:200])

        try:
            body = resp.json()
        except Exception as e:
            raise ValueError(f"invalid_json_response:{type(e).__name__}") from e
        return self._openai_response_to_dict(body)

    async def _run_openai_transport(self, payload: dict, timeout: float) -> dict:
        t = max(0.01, timeout)
        if self._openai_transport is not None:
            return await asyncio.wait_for(self._openai_transport(payload), timeout=t)
        return await asyncio.wait_for(self._openai_http_call(payload, t), timeout=t)

    async def _call_openai(self, user_text: str, schema: dict, schema_name: str, deadline_at: float) -> tuple[Optional[dict], dict]:
        cfg = self.cfg
        if not cfg.fallback_model or not cfg.openai_api_key:
            return None, {"error": "fallback_disabled", "model": cfg.fallback_model, "latency_ms": None, "usage": {}, "raw_text": ""}

        status = self._openai_status
        if status.circuit_open_until is not None:
            if self._clock() < status.circuit_open_until:
                return None, {"error": "circuit_open", "model": cfg.fallback_model, "latency_ms": None, "usage": {}, "raw_text": ""}
            status.circuit_open_until = None

        budget = deadline_at - self._clock()
        if budget <= _MIN_CALL_BUDGET_SECONDS:
            self._record_provider_failure(status, "no_budget")
            return None, {"error": "no_budget", "model": cfg.fallback_model, "latency_ms": None, "usage": {}, "raw_text": ""}

        payload = {
            "model": cfg.fallback_model,
            "input": [
                {"role": "developer", "content": CHARTER},
                {"role": "user", "content": user_text},
            ],
            "reasoning": {"effort": cfg.fallback_effort},
            "text": {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
            "max_output_tokens": cfg.decision_max_tokens,
        }

        t0 = self._clock()
        try:
            result = await self._run_openai_transport(payload, budget)
        except asyncio.TimeoutError:
            self._record_provider_failure(status, "timeout")
            return None, {"error": "timeout", "model": cfg.fallback_model, "latency_ms": None, "usage": {}, "raw_text": ""}
        except Exception as e:
            err = _redact(f"{type(e).__name__}: {e}", cfg.openai_api_key)
            self._record_provider_failure(status, err)
            return None, {"error": err, "model": cfg.fallback_model, "latency_ms": None, "usage": {}, "raw_text": ""}

        latency_ms = int((self._clock() - t0) * 1000)
        status.last_latency_ms = latency_ms

        text = result.get("text", "") if isinstance(result, dict) else ""
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        model_used = (result.get("model") if isinstance(result, dict) else None) or cfg.fallback_model

        parsed = _try_parse_json(text)
        if parsed is None or not _shape_matches(parsed, schema):
            self._record_provider_failure(status, "invalid_json")
            return None, {"error": "invalid_json", "model": model_used, "latency_ms": latency_ms, "usage": usage, "raw_text": text}

        self._record_provider_success(status)
        return parsed, {"error": "", "model": model_used, "latency_ms": latency_ms, "usage": usage, "raw_text": text}

    # ------------------------------------------------------------------
    # Ledger
    # ------------------------------------------------------------------

    def _make_attempt_log(self) -> AttemptLog:
        return AttemptLog(
            os.path.join(self.data_dir, _ATTEMPT_FILE),
            max_lines=self.cfg.attempt_log_max_lines,
            enabled=self.cfg.attempt_log_enabled,
        )

    @staticmethod
    def _circuit_snapshot(status: _ProviderStatus) -> dict:
        open_until = status.circuit_open_until
        return {
            "failures": int(status.consecutive_failures),
            "open_until": round(open_until, 1) if open_until is not None else None,
        }

    def _attempt_row(
        self,
        *,
        kind: str,
        provider: str,
        meta: dict,
        elapsed_ms: int,
        input_digest: str,
        fallback_used: bool,
        final_provider: str,
        circuit: dict,
        claim_violations: Optional[list] = None,
    ) -> dict:
        """One ledger row for one provider attempt. Error text is redacted here too."""
        cfg = self.cfg
        is_anthropic = provider == "anthropic"
        error_raw = str(meta.get("error") or "")
        error = _redact(_redact_anthropic(error_raw), cfg.openai_api_key)[:120]
        usage_raw = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
        responded = meta.get("latency_ms") is not None
        return {
            "ts": datetime.fromtimestamp(self._clock(), tz=ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
            "kind": kind,
            "provider": provider,
            "requested_model": cfg.decision_model if is_anthropic else cfg.fallback_model,
            "served_model": (meta.get("model") or None) if responded else None,
            "effort": cfg.decision_effort if is_anthropic else cfg.fallback_effort,
            "timeout_s": cfg.decision_timeout_seconds if is_anthropic else cfg.fallback_timeout_seconds,
            "latency_ms": int(elapsed_ms),
            "status": _attempt_status(error_raw),
            "error_class": _error_class(error),
            "error": error,
            "usage": {
                "in": _tok(usage_raw.get("input_tokens")),
                "out": _tok(usage_raw.get("output_tokens")),
                "cache_read": _tok(usage_raw.get("cache_read_input_tokens")),
                "cache_write": _tok(usage_raw.get("cache_creation_input_tokens")),
            },
            "cost_usd": round(_provider_cost(provider, usage_raw), 6),
            "circuit": circuit,
            "fallback_used": bool(fallback_used),
            "final_provider": final_provider,
            "input_digest": input_digest,
            "claim_violations": [str(v) for v in (claim_violations or [])],
        }

    def _claim_violations(self, kind: str, decision: Any, ctx: dict) -> list[str]:
        """LOG-ONLY claim check on the parsed decision text. Fail-open: never raises into the decision path."""
        if getattr(self.cfg, "claim_check_mode", "log") == "off":
            return []
        try:
            text = decision.thesis if kind == "entry" else getattr(decision, "reason", "")
            violations = check_claims(str(text or ""), ctx or {})
            if violations:
                logger.warning("DPOF claim check: %s decision contradicts its inputs: %s", kind, ", ".join(violations))
            return violations
        except Exception as e:
            logger.debug(f"DPOF claim check skipped: {type(e).__name__}: {e}")
            return []

    def _emit_attempts(self, attempts: list, final_provider: str) -> None:
        """Build and schedule one ledger row per attempt (fire-and-forget)."""
        for attempt in attempts:
            try:
                row = self._attempt_row(final_provider=final_provider, **attempt)
            except Exception as e:
                logger.debug(f"DPOF decider: attempt row build failed: {type(e).__name__}: {e}")
                continue
            self._attempt_log.record(row)

    def _warn_primary_failed(self, status: _ProviderStatus, meta: dict, elapsed_ms: int) -> None:
        """WARNING for a primary failure, at most once per error class per minute."""
        error = _redact(_redact_anthropic(str(meta.get("error") or "")), self.cfg.openai_api_key)
        cls = _error_class(error) or "unknown"
        now = self._clock()
        last = self._warn_last.get(cls)
        if last is not None and now - last < _WARN_INTERVAL_SECONDS:
            return
        self._warn_last[cls] = now
        logger.warning(
            "DPOF decider primary failed provider=%s class=%s latency=%.1fs circuit=%s",
            status.name, cls, elapsed_ms / 1000.0,
            "open" if status.circuit_open_until is not None else "closed",
        )

    async def _record_decision(
        self,
        *,
        kind: str,
        provider: str,
        model: str,
        latency_ms: Optional[int],
        input_digest: str,
        validated: Optional[dict],
        raw_text: str,
        error: str,
    ) -> None:
        if self._record is None:
            return
        record = {
            "ts": self._clock(),
            "kind": kind,
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
            "input_digest": input_digest,
            "validated": validated,
            "raw": (raw_text or "")[:4096],
            "error": error,
        }
        try:
            await self._record(record)
        except Exception as e:
            logger.debug(f"DPOF decider: record callback failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def _decide(self, kind: str, ctx: dict, *, candidates: Optional[list] = None, position: Optional[dict] = None):
        await self._ensure_state_loaded()
        async with self._state_lock:
            self._roll_day_if_needed()
            if self._state["calls"] >= self.cfg.max_decisions_per_day:
                raise DecisionUnavailable("daily_cap")

        if kind == "entry":
            user_text = await build_entry_message(ctx, candidates or [])
            schema = ENTRY_SCHEMA
        else:
            user_text = await build_review_message(ctx, position or {})
            schema = REVIEW_SCHEMA

        digest = hashlib.sha256(user_text.encode("utf-8")).hexdigest()

        attempts: list[dict] = []
        deadline_a = self._clock() + max(0.0, self.cfg.decision_timeout_seconds)
        t_a = self._clock()
        parsed, meta = await self._call_anthropic(user_text, schema, deadline_a)
        elapsed_a = int((self._clock() - t_a) * 1000)
        attempts.append({
            "kind": kind, "provider": "anthropic", "meta": meta, "elapsed_ms": elapsed_a,
            "input_digest": digest, "fallback_used": False,
            "circuit": self._circuit_snapshot(self._anthropic_status),
        })
        provider_used = "anthropic"

        if parsed is None:
            if meta.get("error") != "circuit_open":
                self._warn_primary_failed(self._anthropic_status, meta, elapsed_a)
            deadline_f = self._clock() + max(0.0, self.cfg.fallback_timeout_seconds)
            t_f = self._clock()
            parsed2, meta2 = await self._call_openai(user_text, schema, f"dpof_{kind}", deadline_f)
            attempts.append({
                "kind": kind, "provider": "openai", "meta": meta2,
                "elapsed_ms": int((self._clock() - t_f) * 1000), "input_digest": digest,
                "fallback_used": True, "circuit": self._circuit_snapshot(self._openai_status),
            })
            if parsed2 is not None:
                parsed, meta, provider_used = parsed2, meta2, "openai"
            else:
                combined_error = f"anthropic:{meta.get('error')}|openai:{meta2.get('error')}"
                failed = [("anthropic", meta.get("usage") or {}), ("openai", meta2.get("usage") or {})]
                await self._finish_attempt(provider="none", usage={}, failed=failed)
                self._emit_attempts(attempts, final_provider="none")
                await self._record_decision(
                    kind=kind,
                    provider="none",
                    model="",
                    latency_ms=None,
                    input_digest=digest,
                    validated=None,
                    raw_text=(meta2.get("raw_text") or meta.get("raw_text") or ""),
                    error=combined_error,
                )
                raise DecisionUnavailable(combined_error)

        if kind == "entry":
            fields, invalid_reasons = _validate_entry(parsed, candidates or [], self.cfg)
            decision: Any = EntryDecision(
                action=fields["action"],
                candidate_key=fields["candidate_key"],
                conviction=fields["conviction"],
                qty=fields["qty"],
                clamped=fields["clamped"],
                thesis=fields["thesis"],
                invalidation=fields["invalidation"],
                recheck_minutes=fields["recheck_minutes"],
                provider=provider_used,
                model=meta.get("model", ""),
                latency_ms=int(meta.get("latency_ms") or 0),
                raw=parsed,
            )
        else:
            fields, invalid_reasons = _validate_review(parsed, self.cfg)
            decision = ReviewDecision(
                action=fields["action"],
                trim_fraction=fields["trim_fraction"],
                urgency=fields["urgency"],
                thesis_status=fields["thesis_status"],
                reason=fields["reason"],
                next_review_minutes=fields["next_review_minutes"],
                provider=provider_used,
                model=meta.get("model", ""),
                latency_ms=int(meta.get("latency_ms") or 0),
                raw=parsed,
            )

        if invalid_reasons:
            logger.warning(f"DPOF decider: {kind} decision had invalid field(s) {invalid_reasons}; safest interpretation applied")

        violations = self._claim_violations(kind, decision, ctx)
        for att in attempts:
            if att["provider"] == provider_used:
                att["claim_violations"] = violations

        failed = [
            (a["provider"], a["meta"].get("usage") or {})
            for a in attempts if a["provider"] != provider_used
        ]
        await self._finish_attempt(provider=provider_used, usage=meta.get("usage", {}), failed=failed)
        self._emit_attempts(attempts, final_provider=provider_used)
        await self._record_decision(
            kind=kind,
            provider=provider_used,
            model=meta.get("model", ""),
            latency_ms=meta.get("latency_ms"),
            input_digest=digest,
            validated=asdict(decision),
            raw_text=meta.get("raw_text", ""),
            error="",
        )
        return decision


class _OpenAIHTTPError(Exception):
    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"http_{status_code}" + (f":{detail}" if detail else ""))
