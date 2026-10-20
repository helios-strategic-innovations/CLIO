"""Small helpers for reading Anthropic Messages responses.

Kept free of agent imports so any module can use it. A response may start
with a ``thinking`` block (the gateway's Haiku always does), so ``content[0]``
is not safe to index for text.
"""

from typing import Any


def _block_type(block: Any) -> Any:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_text(block: Any) -> Any:
    if isinstance(block, dict):
        return block.get("text")
    return getattr(block, "text", None)


def first_text(response: Any, default: str = "") -> str:
    """Concatenate the text of every ``type == "text"`` block; ``default`` if blank.

    Works on SDK objects and plain dicts. Ignores ``thinking``, ``tool_use`` and
    any other block type. Tolerates ``response`` or ``.content`` being None or
    not a list.
    """
    if response is None:
        return default
    content = response.get("content") if isinstance(response, dict) else getattr(response, "content", None)
    if not isinstance(content, list):
        return default
    parts = []
    for block in content:
        if _block_type(block) != "text":
            continue
        text = _block_text(block)
        if isinstance(text, str):
            parts.append(text)
    joined = "".join(parts)
    return joined if joined.strip() else default
