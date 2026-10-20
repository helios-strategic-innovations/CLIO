import pytest

from agent.integrations.alpaca_values import (
    AlpacaValueError,
    normalize_order_side,
    normalize_position_side,
)


class EnumLike:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"PositionSide.{str(self.value).upper()}"


class StringOnlyEnumLike:
    def __str__(self):
        return "PositionSide.LONG"


def test_prefers_value_and_normalizes_documented_values():
    assert normalize_position_side(EnumLike("long")) == "long"
    assert normalize_position_side(EnumLike("SHORT")) == "short"
    assert normalize_order_side(EnumLike("buy")) == "buy"


def test_unknown_enum_representation_rejects_instead_of_guessing_short():
    with pytest.raises(AlpacaValueError, match="unknown"):
        normalize_position_side(StringOnlyEnumLike())
    with pytest.raises(AlpacaValueError, match="unknown"):
        normalize_position_side("PositionSide.LONG")
    with pytest.raises(AlpacaValueError, match="unknown"):
        normalize_position_side("flat")


def test_enum_representation_change_cannot_classify_every_position_short():
    positions = [EnumLike("long"), EnumLike("long"), EnumLike("short")]
    normalized = [normalize_position_side(side) for side in positions]
    assert normalized == ["long", "long", "short"]
    assert normalized.count("short") == 1
