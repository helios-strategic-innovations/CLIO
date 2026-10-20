"""Pass-2 verifier V2 (external API contracts / decision quality lens)
regression tests for agent.dpof.decider's ENTRY_SCHEMA/REVIEW_SCHEMA.

Confirmed bug (fixed): ENTRY_SCHEMA/REVIEW_SCHEMA carried `minimum`/`maximum`
numeric constraints. Anthropic's structured outputs (`output_config.format`)
do NOT support numeric constraints -- verified against the bundled
`claude-api` skill's JSON Schema Limitations table AND, more concretely,
against the installed `anthropic` SDK (0.125.0) source: the schema-transform
helper that strips/relocates unsupported constraints
(`anthropic.lib._parse._transform.transform_schema`) is called ONLY from
`Messages.parse()`/`Messages.stream()`'s typed `output_format` path (which
converts a pydantic-derived schema) -- never from `Messages.create()`, which
is what `DPOFDecider._sync_anthropic_call` uses via `client.messages.create(
**payload)` with a hand-written raw dict schema in
`output_config["format"]["schema"]`. A raw dict schema handed to `.create()`
therefore reaches the API exactly as written, with `minimum`/`maximum`
un-stripped and unsupported by Anthropic's own documented contract for this
call shape. These tests pin the schema's shape so the constraint can't
silently come back, and confirm the numeric bound is still fully enforced in
code (the model can never exceed what code offers) regardless of the schema.
"""

from __future__ import annotations

import json

from agent.dpof.decider import ENTRY_SCHEMA, REVIEW_SCHEMA, _validate_entry, _validate_review
from agent.dpof.config import DPOFConfig


def _walk_schema(node):
    """Yield every dict-shaped subschema reachable from `node` (properties,
    items, $defs, anyOf/allOf/oneOf) -- exhaustive, so a constraint hidden a
    level deep can't slip past a shallow check."""
    if isinstance(node, dict):
        yield node
        for key in ("properties", "$defs", "definitions"):
            for sub in node.get(key, {}).values():
                yield from _walk_schema(sub)
        for key in ("items",):
            if key in node:
                yield from _walk_schema(node[key])
        for key in ("anyOf", "allOf", "oneOf"):
            for sub in node.get(key, []):
                yield from _walk_schema(sub)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_schema(item)


_UNSUPPORTED_ANTHROPIC_KEYWORDS = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "minItems", "maxItems", "patternProperties",
)


def test_entry_schema_has_no_unsupported_anthropic_keywords():
    for sub in _walk_schema(ENTRY_SCHEMA):
        for kw in _UNSUPPORTED_ANTHROPIC_KEYWORDS:
            assert kw not in sub, (
                f"ENTRY_SCHEMA contains unsupported Anthropic structured-output "
                f"keyword {kw!r} in {sub!r} -- Anthropic's output_config.format "
                f"does not support this on a raw dict schema sent via "
                f"client.messages.create()"
            )


def test_review_schema_has_no_unsupported_anthropic_keywords():
    for sub in _walk_schema(REVIEW_SCHEMA):
        for kw in _UNSUPPORTED_ANTHROPIC_KEYWORDS:
            assert kw not in sub, (
                f"REVIEW_SCHEMA contains unsupported Anthropic structured-output "
                f"keyword {kw!r} in {sub!r}"
            )


def test_entry_schema_still_json_serializable_and_well_formed():
    # additionalProperties: false is required on every object per the docs'
    # "Supported" list -- a regression here would silently break BOTH
    # providers (OpenAI's strict mode requires it too).
    blob = json.dumps(ENTRY_SCHEMA)
    reparsed = json.loads(blob)
    assert reparsed["additionalProperties"] is False
    assert set(reparsed["required"]) == set(reparsed["properties"].keys())


def test_review_schema_still_json_serializable_and_well_formed():
    blob = json.dumps(REVIEW_SCHEMA)
    reparsed = json.loads(blob)
    assert reparsed["additionalProperties"] is False
    assert set(reparsed["required"]) == set(reparsed["properties"].keys())


def test_entry_schema_bounds_still_documented_in_descriptions():
    # The numeric bound removed from `minimum`/`maximum` must not just vanish
    # -- it belongs in `description` (a supported keyword) so the model still
    # sees the intended range even though code is the actual enforcer.
    qty_desc = ENTRY_SCHEMA["properties"]["qty"]["description"]
    assert "max_affordable_qty" in qty_desc
    recheck_desc = ENTRY_SCHEMA["properties"]["recheck_minutes"]["description"]
    assert "1" in recheck_desc and "120" in recheck_desc


def test_review_schema_bounds_still_documented_in_descriptions():
    trim_desc = REVIEW_SCHEMA["properties"]["trim_fraction"]["description"]
    assert "0" in trim_desc and "1" in trim_desc
    next_review_desc = REVIEW_SCHEMA["properties"]["next_review_minutes"]["description"]
    assert "1" in next_review_desc and "120" in next_review_desc


# ─────────────────────────────────────────────────────────────────────────
# Code-side enforcement is unconditional -- the schema no longer being able
# to express a bound must not weaken what _validate_entry/_validate_review
# actually clamp. These mirror (a subset of) test_decider.py's existing
# clamping coverage, scoped tightly to "the schema change didn't regress
# enforcement".
# ─────────────────────────────────────────────────────────────────────────

def test_conviction_out_of_range_still_clamped_without_schema_minimum(cfg=None):
    cfg = DPOFConfig()
    candidates = []
    fields, reasons = _validate_entry(
        {"action": "pass", "candidate_key": "", "conviction": 5.0, "qty": 0,
         "thesis": "t", "invalidation": "i", "recheck_minutes": 5},
        candidates, cfg,
    )
    assert fields["conviction"] == 1.0

    fields2, _ = _validate_entry(
        {"action": "pass", "candidate_key": "", "conviction": -5.0, "qty": 0,
         "thesis": "t", "invalidation": "i", "recheck_minutes": 5},
        candidates, cfg,
    )
    assert fields2["conviction"] == 0.0


def test_recheck_minutes_out_of_range_still_clamped_without_schema_bounds():
    cfg = DPOFConfig()
    fields, _ = _validate_entry(
        {"action": "pass", "candidate_key": "", "conviction": 0.5, "qty": 0,
         "thesis": "t", "invalidation": "i", "recheck_minutes": 999999},
        [], cfg,
    )
    assert fields["recheck_minutes"] <= cfg.review_max_minutes
    assert fields["recheck_minutes"] >= cfg.review_min_minutes


def test_trim_fraction_out_of_range_still_clamped_without_schema_bounds():
    cfg = DPOFConfig()
    fields, reasons = _validate_review(
        {"action": "trim", "trim_fraction": -1.0, "urgency": "normal",
         "thesis_status": "intact", "reason": "r", "next_review_minutes": 5},
        cfg,
    )
    # trim_fraction <= 0 is treated as invalid -> forced to hold, tf=0.0
    assert fields["action"] == "hold"
    assert fields["trim_fraction"] == 0.0
