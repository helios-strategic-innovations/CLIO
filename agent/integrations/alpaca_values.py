"""Strict Alpaca enum normalization helpers."""

from __future__ import annotations

from typing import Any, Iterable


class AlpacaValueError(ValueError):
    pass


def enum_value(value: Any, *, field: str, allowed: Iterable[str]) -> str:
    if value is None:
        raise AlpacaValueError(f"{field}:missing")
    raw = value.value if hasattr(value, "value") else value
    if not isinstance(raw, str):
        raise AlpacaValueError(f"{field}:unknown_type")
    normalized = raw.strip().lower()
    allowed_values = {str(item).strip().lower() for item in allowed}
    if normalized not in allowed_values:
        raise AlpacaValueError(f"{field}:unknown:{normalized or 'empty'}")
    return normalized


def normalize_position_side(value: Any) -> str:
    return enum_value(value, field="position_side", allowed=("long", "short"))


def normalize_order_side(value: Any) -> str:
    return enum_value(value, field="order_side", allowed=("buy", "sell"))


__all__ = [
    "AlpacaValueError",
    "enum_value",
    "normalize_order_side",
    "normalize_position_side",
]
