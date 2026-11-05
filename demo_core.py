"""Derived offline demo adapter; not the CLIO engine, model or broker.

Extracted source bodies from private CLIO commit 46da00609029298c2bca6d7d0f723b103f50272f.
Rights, ownership and any external release of this subset await operator review.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
import math
import numbers
import re
from typing import Optional, Protocol
from zoneinfo import ZoneInfo


class UnderlyingSnapshot(Protocol):
    last_price: float | None
    vwap: float | None
    ret_15m: float | None


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.BULLISH else (-1 if self is Direction.BEARISH else 0)


def _price_input_code(value) -> Optional[str]:
    """One price input -> its fail-closed code, or None when it is usable.

    None -> price_data_missing; bool or a non-numeric -> price_data_invalid;
    NaN/inf float -> price_data_nonfinite. Integers are finite by definition
    and carry no cap. numbers.Real (not just int/float) keeps numpy scalars
    working exactly as they did before this check existed.
    """
    if value is None:
        return "price_data_missing"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return "price_data_invalid"
    if isinstance(value, numbers.Integral):
        return None
    try:
        finite = math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return "price_data_invalid"
    return None if finite else "price_data_nonfinite"

def price_confirms(u: UnderlyingSnapshot, direction: Direction) -> tuple[bool, list[str]]:
    """Session-VWAP side + 15-min momentum must agree with `direction`.

    Any of last_price/vwap/ret_15m missing -> (False, ["price_data_missing"]).
    Invalid (bool/non-numeric) -> price_data_invalid; NaN/inf -> price_data_nonfinite.
    Input codes are checked BEFORE any comparison, so a NaN can never read as a
    real "price_below_vwap" market condition. Codes follow the fixed order
    last_price, vwap, ret_15m (deduplicated).
    A Direction.NEUTRAL input has nothing to confirm -> (False, ["direction_neutral"]).
    """
    input_codes: list[str] = []
    for value in (u.last_price, u.vwap, u.ret_15m):
        code = _price_input_code(value)
        if code is not None and code not in input_codes:
            input_codes.append(code)
    if input_codes:
        return False, input_codes

    reasons: list[str] = []
    if direction is Direction.BULLISH:
        if not (u.last_price > u.vwap):
            reasons.append("price_below_vwap")
        if not (u.ret_15m > 0):
            reasons.append("momentum_not_positive")
    elif direction is Direction.BEARISH:
        if not (u.last_price < u.vwap):
            reasons.append("price_above_vwap")
        if not (u.ret_15m < 0):
            reasons.append("momentum_not_negative")
    else:
        return False, ["direction_neutral"]

    return (len(reasons) == 0, reasons)


_ET = ZoneInfo("America/New_York")

_REASONS = frozenset({"no_prior_close", "history_incomplete", "invalid_input", "prior_broken",
                      "prior_weakening", "flow_conflict", "source_time_unknown", "reconfirmed",
                      "waiting_price", "waiting_alignment"})

_FIELDS = frozenset({"version", "posture", "disposition", "reason_codes", "captured_at", "snapshot_asof",
                     "price_bar_at", "nofa_cache_at", "premium_cache_at", "provider_freshness",
                     "history_coverage", "history_scope", "anchor_id", "anchor_closed_at", "prior_exit_class", "nofa",
                     "premium", "price_confirmed", "existing_cooldown_gate"})

def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (OverflowError, ValueError):
        return None

def _time(asof, age):
    age = _number(age)
    return asof - age if asof is not None and age is not None and age >= 0 and math.isfinite(asof - age) else None

def _flow(reading, direction):
    if reading is None or getattr(reading, "available", None) is not True:
        return "unknown"
    value = _number(getattr(reading, "value", None))
    if value is None:
        return "unknown"
    if value == 0:
        return "neutral"
    sign = 1 if direction is Direction.BULLISH else -1
    return "aligned" if value * sign > 0 else "opposed"

def validate_reentry(value: object) -> dict | None:
    """Return only detached primitive data with internally consistent evidence."""
    if type(value) is not dict or any(type(key) is not str for key in value) or set(value) != _FIELDS:
        return None
    enums = {
        "posture": {"observational"},
        "disposition": {"WOULD_DEFER", "RECONFIRMED", "NO_REENTRY_CONTEXT", "INSUFFICIENT"},
        "provider_freshness": {"unknown"}, "history_coverage": {"complete", "incomplete"},
        "history_scope": {"retained_positions"}, "prior_exit_class": {"broken", "weakening", "none", "unknown"},
        "existing_cooldown_gate": {"passed"}, "nofa": {"aligned", "opposed", "neutral", "unknown"},
        "premium": {"aligned", "opposed", "neutral", "unknown"},
    }
    if type(value["version"]) is not int or value["version"] != 1:
        return None
    if any(type(value[key]) is not str or value[key] not in allowed for key, allowed in enums.items()):
        return None
    reasons = value["reason_codes"]
    if (type(reasons) is not list or len(reasons) > 6
            or any(type(reason) is not str or reason not in _REASONS for reason in reasons)):
        return None
    if value["price_confirmed"] is not None and type(value["price_confirmed"]) is not bool:
        return None
    anchor = value["anchor_id"]
    if anchor is not None and (type(anchor) is not str or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", anchor) is None):
        return None
    time_keys = ("captured_at", "snapshot_asof", "price_bar_at", "nofa_cache_at", "premium_cache_at", "anchor_closed_at")
    times = {}
    for key in time_keys:
        raw = value[key]
        number = _number(raw) if raw is not None else None
        if raw is not None and number is None:
            return None
        times[key] = number
    disposition = value["disposition"]
    if disposition in ("WOULD_DEFER", "RECONFIRMED"):
        now, asof, closed = times["captured_at"], times["snapshot_asof"], times["anchor_closed_at"]
        if (value["history_coverage"] != "complete" or anchor is None
                or value["prior_exit_class"] not in ("broken", "weakening")
                or "prior_" + value["prior_exit_class"] not in reasons
                or now is None or asof != now or closed is None
                or any(times[key] is None or not closed < times[key] <= now
                       for key in ("price_bar_at", "nofa_cache_at", "premium_cache_at"))):
            return None
        try:
            if datetime.fromtimestamp(closed, _ET).date() != datetime.fromtimestamp(now, _ET).date():
                return None
        except (OverflowError, OSError, ValueError):
            return None
        flow = (value["nofa"], value["premium"])
        expected_reason = "flow_conflict" if disposition == "WOULD_DEFER" else "reconfirmed"
        if len(reasons) != 2 or set(reasons) != {"prior_" + value["prior_exit_class"], expected_reason}:
            return None
        if disposition == "WOULD_DEFER" and (set(flow) != {"aligned", "opposed"} or "flow_conflict" not in reasons):
            return None
        if disposition == "RECONFIRMED" and (value["price_confirmed"] is not True or "unknown" in flow
                or "opposed" in flow or "aligned" not in flow or "reconfirmed" not in reasons):
            return None
    elif disposition == "NO_REENTRY_CONTEXT":
        if (value["history_coverage"] != "complete" or anchor is not None
                or value["prior_exit_class"] != "none" or reasons != ["no_prior_close"]):
            return None
    return {**value, **times, "reason_codes": list(reasons)}

def evaluate_reentry(*, symbol: str, direction: Direction, captured_at: float, snapshot: object,
                     history: list, history_complete: bool = False) -> dict:
    """Evaluate retained worker-local rows; completeness is never inferred from their count."""
    now = _number(captured_at)
    result = {"version": 1, "posture": "observational", "disposition": "INSUFFICIENT",
              "reason_codes": [], "captured_at": now, "snapshot_asof": None, "price_bar_at": None,
              "nofa_cache_at": None, "premium_cache_at": None, "provider_freshness": "unknown",
              "history_coverage": "complete" if history_complete is True else "incomplete",
              "history_scope": "retained_positions", "anchor_id": None,
              "anchor_closed_at": None, "prior_exit_class": "unknown", "nofa": "unknown", "premium": "unknown",
              "price_confirmed": None, "existing_cooldown_gate": "passed"}
    if (now is None or type(symbol) is not str or not symbol or len(symbol) > 32
            or direction not in (Direction.BULLISH, Direction.BEARISH) or type(history) is not list
            or len(history) > 512 or type(history_complete) is not bool):
        result["reason_codes"] = ["invalid_input"]
        return result
    try:
        day = datetime.fromtimestamp(now, _ET).date()
        matching, unknown = [], False
        for position in history:
            if position.underlying != symbol or position.direction != direction.value or position.status != "closed":
                continue
            closed = _number(position.closed_time)
            if closed is None or closed >= now:
                unknown = True
                continue
            if datetime.fromtimestamp(closed, _ET).date() != day or position.exit_reason != "decider_exit":
                continue
            review = position.last_review
            if type(review) is not dict or review.get("action") != "exit" or review.get("thesis_status") not in ("broken", "weakening"):
                unknown = True
                continue
            matching.append((closed, position))
        if not matching:
            if not unknown and history_complete:
                result.update(disposition="NO_REENTRY_CONTEXT", prior_exit_class="none", reason_codes=["no_prior_close"])
            else:
                result["reason_codes"] = ["history_incomplete"]
            return result
        closed, anchor = max(matching, key=lambda item: item[0])
        if unknown:
            result["reason_codes"] = ["history_incomplete"]
            return result
        anchor_id = anchor.id
        if type(anchor_id) is not str or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", anchor_id) is None:
            result["reason_codes"] = ["invalid_input"]
            return result
        status = anchor.last_review["thesis_status"]
        result.update(anchor_id=anchor_id, anchor_closed_at=closed, prior_exit_class=status,
                      reason_codes=["prior_" + status])
        asof = _number(snapshot.asof)
        result["snapshot_asof"] = asof
        u = snapshot.underlyings.get(symbol)
        if u is None or type(getattr(u, "symbol", None)) is not str or u.symbol != symbol:
            result["reason_codes"].append("invalid_input")
            return result
        nofa, premium = u.pillars.get("nofa"), u.pillars.get("premium")
        result["nofa"], result["premium"] = _flow(nofa, direction), _flow(premium, direction)
        conflict = {result["nofa"], result["premium"]} == {"aligned", "opposed"}
        if conflict:
            result["reason_codes"].append("flow_conflict")
        result["price_bar_at"] = _time(asof, u.price_age_seconds)
        result["nofa_cache_at"] = _time(asof, getattr(nofa, "age_seconds", None))
        result["premium_cache_at"] = _time(asof, getattr(premium, "age_seconds", None))
        result["price_confirmed"] = bool(price_confirms(u, direction)[0])
        times = (result["price_bar_at"], result["nofa_cache_at"], result["premium_cache_at"])
        if (not history_complete or snapshot.data_healthy is not True or asof != now or any(t is None or not closed < t <= now for t in times)
                or "unknown" in (result["nofa"], result["premium"])):
            result["reason_codes"].append("source_time_unknown")
        elif conflict:
            result["disposition"] = "WOULD_DEFER"
        elif ("opposed" not in (result["nofa"], result["premium"])
              and "aligned" in (result["nofa"], result["premium"]) and result["price_confirmed"]):
            result["disposition"] = "RECONFIRMED"
            result["reason_codes"].append("reconfirmed")
        else:
            result["reason_codes"].append("waiting_price" if not result["price_confirmed"] else "waiting_alignment")
        return result
    except (AttributeError, TypeError, ValueError, OverflowError, OSError):
        result["reason_codes"] = ["invalid_input"]
        return result
