"""Repo-scanning invariants for Anthropic client construction and sampling.

Scans COMMITTED python files (``git ls-files``) with the ``ast`` module.

Rule A: every direct ``anthropic.Anthropic`` / ``AsyncAnthropic`` construction must
        spread ``**<x>.anthropic_client_kwargs(...)`` so gateway routing (base_url)
        cannot be bypassed.
Rule B: ``<x>.messages.create(..., temperature=...)`` must pass ``model=`` containing
        ``sampling_model`` (temperature is only valid on the sampling route).
Rule C: no dotted Claude model IDs (``claude-<family>-N.M``) outside ``agent/config.py``
        (the dotted-ID normalizer lives there).
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_CLIENT_NAMES = {"Anthropic", "AsyncAnthropic"}

_RULE_A_EXEMPT_FILES = {"agent/integrations/code_sentinel.py"}

_DOTTED_ID_RE = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-\d+\.\d+")


def _is_tests_path(path: str) -> bool:
    return path.replace("\\", "/").startswith("tests/")


def _is_client_ctor(func: ast.expr) -> bool:
    if isinstance(func, ast.Attribute):
        return (
            func.attr in _CLIENT_NAMES
            and isinstance(func.value, ast.Name)
            and func.value.id == "anthropic"
        )
    if isinstance(func, ast.Name):
        return func.id in _CLIENT_NAMES
    return False


def _is_kwargs_call(expr: ast.expr | None) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "anthropic_client_kwargs"
    )


def _parent_map(tree: ast.AST) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _nearest_scope(node: ast.AST, parents: dict, tree: ast.AST) -> ast.AST:
    """Nearest enclosing function, or the module when there is none."""
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
        cur = parents.get(cur)
    return tree


def _kwargs_names_in_scope(scope: ast.AST, tree: ast.AST, parents: dict) -> set[str]:
    """Names assigned from <x>.anthropic_client_kwargs(...) within this scope."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_kwargs_call(node.value):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and _is_kwargs_call(node.value) and isinstance(node.target, ast.Name):
            targets = [node.target]
        else:
            continue
        if _nearest_scope(node, parents, tree) is scope:
            names.update(t.id for t in targets)
    return names


def _has_wired_kwargs(call: ast.Call, tree: ast.AST, parents: dict) -> bool:
    scope = None
    for kw in call.keywords:
        if kw.arg is not None:
            continue
        if _is_kwargs_call(kw.value):
            return True
        if isinstance(kw.value, ast.Name):
            if scope is None:
                scope = _nearest_scope(call, parents, tree)
            if kw.value.id in _kwargs_names_in_scope(scope, tree, parents):
                return True
    return False


def find_direct_anthropic_violations(path: str, source: str) -> list[str]:
    """Rule A: direct client construction without anthropic_client_kwargs."""
    path = path.replace("\\", "/")
    if _is_tests_path(path) or path in _RULE_A_EXEMPT_FILES:
        return []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [f"{path}: unparseable ({exc.msg} line {exc.lineno})"]
    parents = _parent_map(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_client_ctor(node.func) and not _has_wired_kwargs(node, tree, parents):
            out.append(f"{path}:{node.lineno}: direct Anthropic client construction")
    return out


def find_sampling_violations(path: str, source: str) -> list[str]:
    """Rule B: temperature= on messages.create must route through sampling_model."""
    path = path.replace("\\", "/")
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [f"{path}: unparseable ({exc.msg} line {exc.lineno})"]
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "create"
                and isinstance(fn.value, ast.Attribute) and fn.value.attr == "messages"):
            continue
        kwargs = {kw.arg: kw for kw in node.keywords if kw.arg is not None}
        if "temperature" not in kwargs:
            continue
        model_kw = kwargs.get("model")
        if model_kw is None or "sampling_model" not in ast.unparse(model_kw.value):
            out.append(f"{path}:{node.lineno}: temperature without model=...sampling_model")
    return out


def find_dotted_model_violations(path: str, source: str) -> list[str]:
    """Rule C: dotted Claude model IDs outside agent/config.py and tests/."""
    path = path.replace("\\", "/")
    if _is_tests_path(path) or path == "agent/config.py":
        return []
    out = []
    for lineno, line in enumerate(source.splitlines(), 1):
        m = _DOTTED_ID_RE.search(line)
        if m:
            out.append(f"{path}:{lineno}: dotted model ID {m.group(0)!r}")
    return out


def _committed_python_files(root: Path) -> list[str]:
    res = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=str(root), capture_output=True, text=True, check=True, timeout=60,
    )
    return [line for line in res.stdout.splitlines() if line.strip()]


def scan_repo(root: Path = REPO_ROOT) -> list[str]:
    violations: list[str] = []
    for rel in _committed_python_files(root):
        p = root / rel
        if not p.is_file():
            continue
        source = p.read_text(encoding="utf-8", errors="replace")
        violations += find_direct_anthropic_violations(rel, source)
        violations += find_sampling_violations(rel, source)
        violations += find_dotted_model_violations(rel, source)
    return violations


# ---------------------------------------------------------------- unit tests

def test_rule_a_flags_direct_construction():
    src = 'import anthropic\nc = anthropic.Anthropic(api_key="x", timeout=120)\n'
    assert find_direct_anthropic_violations("agent/foo.py", src)


def test_rule_a_flags_bare_async_name():
    src = "from anthropic import AsyncAnthropic\nc = AsyncAnthropic(api_key='x')\n"
    assert find_direct_anthropic_violations("agent/foo.py", src)


def test_rule_a_allows_wired_form():
    src = (
        "import anthropic\n"
        "c = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=120))\n"
    )
    assert find_direct_anthropic_violations("agent/foo.py", src) == []


def test_rule_a_exempts_tests_and_code_sentinel():
    src = 'import anthropic\nc = anthropic.Anthropic(api_key="x")\n'
    assert find_direct_anthropic_violations("tests/test_x.py", src) == []
    assert find_direct_anthropic_violations("agent/integrations/code_sentinel.py", src) == []


def test_rule_a_no_pending_exemption_for_unwired_core():
    src = 'import anthropic\nc = anthropic.Anthropic(api_key="x")\n'
    assert find_direct_anthropic_violations("agent/core.py", src)


def test_rule_a_same_function_kwargs_name_accepted():
    src = (
        "import anthropic\n"
        "def f():\n"
        "    kw = cfg.anthropic_client_kwargs(timeout=5)\n"
        "    return anthropic.Anthropic(**kw)\n"
    )
    assert find_direct_anthropic_violations("agent/foo.py", src) == []


def test_rule_a_name_from_plain_dict_flagged():
    src = (
        "import anthropic\n"
        "def f():\n"
        '    kw = {"api_key": "x"}\n'
        "    return anthropic.Anthropic(**kw)\n"
    )
    assert find_direct_anthropic_violations("agent/foo.py", src)


def test_rule_a_name_assigned_in_different_function_flagged():
    src = (
        "import anthropic\n"
        "def f():\n"
        "    kw = cfg.anthropic_client_kwargs(timeout=5)\n"
        "    return kw\n"
        "def g():\n"
        "    return anthropic.Anthropic(**kw)\n"
    )
    assert find_direct_anthropic_violations("agent/foo.py", src)


def test_rule_a_annassign_kwargs_accepted():
    src = (
        "import anthropic\n"
        "def f():\n"
        "    kw: dict = cfg.anthropic_client_kwargs(timeout=5)\n"
        "    return anthropic.AsyncAnthropic(**kw)\n"
    )
    assert find_direct_anthropic_violations("agent/foo.py", src) == []


def test_rule_a_advisory_override_pattern_accepted():
    src = (
        "import anthropic\n"
        "def f(k):\n"
        "    kw = cfg.anthropic_client_kwargs(timeout=60)\n"
        '    kw["api_key"] = k\n'
        "    return anthropic.Anthropic(**kw)\n"
    )
    assert find_direct_anthropic_violations("agent/dpof/advisory_llm.py", src) == []


def test_rule_a_module_level_kwargs_accepted_only_at_module_scope():
    src = (
        "import anthropic\n"
        "KW = cfg.anthropic_client_kwargs(timeout=5)\n"
        "client = anthropic.Anthropic(**KW)\n"
    )
    assert find_direct_anthropic_violations("agent/foo.py", src) == []


def test_rule_b_flags_temperature_on_fast_model():
    src = "client.messages.create(model=config.fast_model, temperature=0.2, messages=[])\n"
    assert find_sampling_violations("agent/foo.py", src)


def test_rule_b_allows_temperature_on_sampling_model():
    src = "client.messages.create(model=config.sampling_model, temperature=0.2, messages=[])\n"
    assert find_sampling_violations("agent/foo.py", src) == []


def test_rule_b_ignores_calls_without_temperature():
    src = "client.messages.create(model=config.fast_model, messages=[])\n"
    assert find_sampling_violations("agent/foo.py", src) == []


def test_rule_c_flags_dotted_id_in_non_config_file():
    src = 'MODEL = "claude-opus-4.8"\n'
    assert find_dotted_model_violations("agent/foo.py", src)


def test_rule_c_allows_dotted_id_in_config_and_hyphenated_ids():
    assert find_dotted_model_violations("agent/config.py", 'X = "claude-opus-4.8"\n') == []
    assert find_dotted_model_violations("agent/foo.py", 'X = "claude-opus-4-8"\n') == []


# ---------------------------------------------------------------- real repo

def test_real_repo_has_no_invariant_violations():
    violations = scan_repo(REPO_ROOT)
    assert violations == [], "\n".join(violations)
