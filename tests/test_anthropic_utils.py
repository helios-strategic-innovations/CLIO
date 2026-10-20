"""S214: first_text helper (agent/anthropic_utils.py). Pure, no network."""

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.anthropic_utils import first_text  # noqa: E402


def _resp(*blocks):
    return NS(content=list(blocks))


def test_thinking_then_text_returns_text():
    r = _resp(NS(type="thinking", thinking="hmm", signature="s"), NS(type="text", text="Answer"))
    assert first_text(r) == "Answer"


def test_text_only():
    assert first_text(_resp(NS(type="text", text="Hi"))) == "Hi"


def test_empty_content_returns_default():
    assert first_text(_resp()) == ""
    assert first_text(_resp(), default="fallback") == "fallback"


def test_tool_use_only_returns_default():
    r = _resp(NS(type="tool_use", id="t1", name="x", input={}))
    assert first_text(r, default="none") == "none"


def test_dict_blocks_supported():
    r = {"content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "dict text"}]}
    assert first_text(r) == "dict text"


def test_none_content_and_none_response_tolerated():
    assert first_text(NS(content=None), default="d") == "d"
    assert first_text(None, default="d") == "d"
    assert first_text(NS(), default="d") == "d"


def test_multiple_text_blocks_concatenated_in_order():
    r = _resp(NS(type="text", text="Part one. "), NS(type="thinking", thinking="x"), NS(type="text", text="Part two."))
    assert first_text(r) == "Part one. Part two."


def test_whitespace_only_returns_default():
    assert first_text(_resp(NS(type="text", text="   \n\t")), default="blank") == "blank"


def test_text_with_surrounding_whitespace_is_not_stripped_by_helper():
    # Callers strip; the helper returns the concatenation as-is when non-blank.
    assert first_text(_resp(NS(type="text", text="  padded  "))) == "  padded  "
