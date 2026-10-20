"""Prompt-embedded JSON schema for providers without structured outputs (OP-05).

Direct Anthropic enforces ``output_config.format``; a gateway (OneProvider)
ignores it. When structured outputs are unavailable the answer shape must be
stated in the prompt itself. Stdlib only, no agent imports, and it never
mutates its argument.
"""

from __future__ import annotations

import json


def schema_instruction(schema: dict) -> str:
    """Return the prompt text that tells the model the exact answer schema."""
    rendered = json.dumps(schema, separators=(",", ":"), sort_keys=True, default=str)
    return (
        "Answer with exactly one JSON object that matches this JSON schema "
        "(include every required key; no prose, no markdown, no code fences):\n"
        + rendered
    )
