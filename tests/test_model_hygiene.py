#!/usr/bin/env python3
"""Guard CLIO model defaults against retired model strings.

Run: python3 tests/test_model_hygiene.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(name)


def test_judgment_model_defaults_to_supported_opus():
    from agent.config import Config

    check(
        "dataclass judgment default uses Opus 5.5",
        Config.__dataclass_fields__["judgment_model"].default == "claude-opus-5-5",
        Config.__dataclass_fields__["judgment_model"].default,
    )
    # The code default, not a deployment's override (the VPS .env sets JUDGMENT_MODEL).
    saved = os.environ.pop("JUDGMENT_MODEL", None)
    try:
        cfg = Config.from_env()
    finally:
        if saved is not None:
            os.environ["JUDGMENT_MODEL"] = saved
    check("from_env judgment default uses Opus 5.5", cfg.judgment_model == "claude-opus-5-5", cfg.judgment_model)


def test_norm_model_id_hyphenates_dotted_claude_ids():
    from agent.config import _norm_model_id

    check("dotted sonnet normalized", _norm_model_id("claude-sonnet-5.5") == "claude-sonnet-5-5")
    check("dotted haiku normalized", _norm_model_id("claude-haiku-4.5") == "claude-haiku-4-5")
    check("hyphenated id unchanged", _norm_model_id("claude-opus-5-5") == "claude-opus-5-5")
    check("whitespace stripped", _norm_model_id("  claude-opus-5-5 ") == "claude-opus-5-5")
    check("non-claude value untouched", _norm_model_id("gpt-6.1") == "gpt-6.1")


def test_config_default_model_ids_are_canonical():
    import re
    from agent.config import Config

    pattern = re.compile(r"^claude-[a-z]+(-\d+)+(-\d{8})?$")
    for name in ("default_model", "fast_model", "judgment_model"):
        value = Config.__dataclass_fields__[name].default
        check(f"{name} default is canonical hyphenated id", bool(pattern.match(value)), value)


def test_config_source_has_no_dotted_claude_literal_outside_normalizer():
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "agent", "config.py"), "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    # Drop the _norm_model_id function body (the only place a dotted pattern may live).
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def _norm_model_id"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace())
    rest = lines[:start] + lines[end:]
    dotted = re.compile(r"claude-[a-z]+-\d+\.\d+")
    hits = [ln for ln in rest if dotted.search(ln)]
    check("no dotted claude model literal in config.py", not hits, str(hits[:2]))


def test_retired_fable_not_in_config_source():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "agent", "config.py")
    with open(path, "r", encoding="utf-8") as handle:
        src = handle.read()
    check("retired Fable string absent from config.py", "claude-fable-5" not in src)


if __name__ == "__main__":
    test_judgment_model_defaults_to_supported_opus()
    test_retired_fable_not_in_config_source()
    test_norm_model_id_hyphenates_dotted_claude_ids()
    test_config_default_model_ids_are_canonical()
    test_config_source_has_no_dotted_claude_literal_outside_normalizer()
    print("ALL PASS")
