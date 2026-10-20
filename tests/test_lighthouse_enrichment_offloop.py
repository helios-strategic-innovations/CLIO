"""S212 LOOP-1: the learning cycle must not run the Lighthouse enrichment on the event loop.

`_build_rss_enrichment_context` -> `get_enrichment_context` -> `scan_current_window()` is a SYNC, sequential httpx
fetch of ~46 RSS feeds plus HeadlineManager.process. Called directly from the async `_analyze_with_claude` it stalled the
event loop 12-23 s whenever the feed cache was cold (5 missed `newsflash_refresh` / `econ_surprise_refresh` runs in 16 h on
2026-09-30). The fix runs it in an executor, like the FD block right below it.

Driving `_analyze_with_claude` end to end would read live state (UW bridge, the tradingbot gap bridge, priors files), so the
contract is pinned on the source's AST instead -- deterministic, no I/O beyond reading this checkout's file.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "agent" / "integrations" / "learning_engine.py"
HELPER = "_build_rss_enrichment_context"


def _tree(text=None):
    return ast.parse(text if text is not None else SRC.read_text())


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _direct_calls(node, name):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name]


def _executor_calls(node, name):
    """`await <x>.run_in_executor(None, name, ...)` calls whose callable argument is exactly `name`."""
    out = []
    for aw in (n for n in ast.walk(node) if isinstance(n, ast.Await)):
        c = aw.value
        if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "run_in_executor"
                and len(c.args) >= 2 and isinstance(c.args[1], ast.Name) and c.args[1].id == name):
            out.append(c)
    return out


def _check(text):
    tree = _tree(text)
    fn = _func(tree, "_analyze_with_claude")
    assert isinstance(fn, ast.AsyncFunctionDef)
    assert _direct_calls(fn, HELPER) == [], "the sync helper is called directly inside the async function"
    assert len(_executor_calls(fn, HELPER)) == 1, "expected exactly one awaited run_in_executor call of the helper"
    # ... inside a try whose handler catches Exception (fire-and-forget: enrichment can never break the analysis)
    guarded = False
    for t in (n for n in ast.walk(fn) if isinstance(n, ast.Try)):
        body_has = any(_executor_calls(stmt, HELPER) for stmt in t.body)
        catches = any(h.type is not None and getattr(h.type, "id", None) == "Exception" for h in t.handlers)
        guarded = guarded or (body_has and catches)
    assert guarded, "the executor call must sit inside try/except Exception"
    # nowhere else in the module may the helper be called synchronously (other than through the executor)
    module_direct = [c for c in _direct_calls(tree, HELPER)]
    assert module_direct == [], "a direct call of the sync helper exists somewhere in learning_engine.py"


def test_the_learning_cycle_runs_the_enrichment_in_an_executor():
    _check(SRC.read_text())


def test_the_helper_is_still_a_plain_sync_function():
    """If it ever becomes async, the executor wrapping (and this test) must be revisited."""
    assert isinstance(_func(_tree(), HELPER), ast.FunctionDef)


# ---- the checker itself must catch the regressions it exists for ----------------------------------------------
GOOD = SRC.read_text()
DIRECT = "rss_enrichment = _build_rss_enrichment_context(category)"
WRAPPED = (
    "rss_enrichment = await asyncio.get_running_loop().run_in_executor(\n"
    "            None, _build_rss_enrichment_context, category,\n"
    "        )"
)


def test_checker_baseline_uses_the_wrapped_form():
    assert WRAPPED in GOOD and DIRECT not in GOOD


@pytest.mark.parametrize("label,mutant", [
    ("direct sync call restored", GOOD.replace(WRAPPED, DIRECT)),
    ("wrong callable in the executor", GOOD.replace("None, _build_rss_enrichment_context, category,", "None, _build_fd_research_block, category,")),
    ("executor call not awaited", GOOD.replace(WRAPPED, WRAPPED.replace("await ", ""))),
    ("try/except removed", GOOD.replace(
        "    try:\n        rss_enrichment = await asyncio.get_running_loop().run_in_executor(\n            None, _build_rss_enrichment_context, category,\n        )\n    except Exception as e:\n        logger.debug(\"CC-LH-3 RSS enrichment failed: %s\", e)\n",
        "    rss_enrichment = await asyncio.get_running_loop().run_in_executor(\n        None, _build_rss_enrichment_context, category,\n    )\n")),
    ("extra direct call elsewhere", GOOD + f"\n\ndef _stray():\n    return {HELPER}('x')\n"),
])
def test_checker_rejects_each_regression(label, mutant):
    assert mutant != GOOD, f"mutant '{label}' did not change the source (anchor drift)"
    with pytest.raises(AssertionError):
        _check(mutant)
