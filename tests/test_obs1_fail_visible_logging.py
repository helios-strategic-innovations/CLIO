#!/usr/bin/env python3
"""OBS-1 contract tests for fail-visible non-fatal memory wrappers.

These tests parse source structure only; they do not import CLIO modules, read
runtime memory, call providers, or touch live stores. The production mutation
they catch is the OBS-1 regression class: a memory-state fire-and-forget
exception path dropping below WARNING, omitting the exception class, or omitting
traceback context.
"""

import ast
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]

OBS1_WRAPPERS = [
    ("agent/core.py", "Fire-and-forget episodic record failed"),
    ("agent/core.py", "Fire-and-forget memory update failed"),
    ("agent/core.py", "Fire-and-forget feedback update failed"),
    ("agent/memory/memory_engine.py", "Fire-and-forget working memory promotion check failed"),
    ("agent/memory/memory_engine.py", "SEAL episodic milestone recording failed"),
    ("agent/integrations/trade_idea_feedback.py", "Episodic record failed"),
    ("agent/integrations/trade_idea_feedback.py", "CLIO-UW-3 episodic record failed"),
]


def _source_tree(rel_path: str) -> ast.Module:
    return ast.parse((REPO / rel_path).read_text(encoding="utf-8"), filename=rel_path)


def _message_template(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("{}")
        return "".join(parts)
    return ""


def _logger_calls_for(tree: ast.AST, needle: str):
    calls = []
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler) or not handler.name:
            continue
        for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "logger"
            ):
                continue
            if node.args and needle in _message_template(node.args[0]):
                calls.append((handler.name, node))
    return calls


def _kw_true(call: ast.Call, name: str) -> bool:
    return any(
        kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in call.keywords
    )


def _is_exception_class_arg(node: ast.AST, exc_name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "__name__"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "type"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == exc_name
    )


@pytest.mark.parametrize(("rel_path", "needle"), OBS1_WRAPPERS)
def test_obs1_memory_wrappers_warn_with_exception_class_and_traceback(rel_path, needle):
    calls = _logger_calls_for(_source_tree(rel_path), needle)

    assert len(calls) == 1, f"{needle!r} should identify one OBS-1 exception handler"
    exc_name, call = calls[0]

    assert call.func.attr == "warning"
    assert _kw_true(call, "exc_info")
    assert any(_is_exception_class_arg(arg, exc_name) for arg in call.args[1:])
    assert any(isinstance(arg, ast.Name) and arg.id == exc_name for arg in call.args[1:])


def test_health_check_episodic_freshness_debug_handler_remains_excluded():
    calls = _logger_calls_for(
        _source_tree("agent/memory/memory_engine.py"),
        "Health check episodic freshness failed",
    )

    assert len(calls) == 1
    _exc_name, call = calls[0]
    assert call.func.attr == "debug"
    assert not _kw_true(call, "exc_info")
