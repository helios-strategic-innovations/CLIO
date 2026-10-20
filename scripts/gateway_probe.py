#!/usr/bin/env python3
"""READ-ONLY compatibility probe for the Anthropic gateway (OneProvider).

Costs cents: small max_tokens, a handful of calls per suite. Makes no writes to
the repo, the bridge or any state file.

Usage:
    python3 scripts/gateway_probe.py --suite models|structured|toolchoice|empty|latency|balance|all [--n N]

Prints one line per check: ``PASS|WARN|FAIL <suite>: <detail>``. Exit code 1
if any check is FAIL.

Structure: the ``judge_*`` functions are pure (no network, no agent imports) and
unit-tested in tests/test_gateway_probe_logic.py. The network suites take an
injectable client factory. ``.env`` is loaded inside ``main()`` before any
``agent`` import (CLAUDE.md anti-pattern #34), so importing this module has no
side effects.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as _dt
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS_PING = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001"]
HAIKU_5_5 = "claude-haiku-5-5"
TOOLCHOICE_MODELS = ["claude-sonnet-5-5", "claude-opus-5-5"]
ALL_SUITES = ["models", "structured", "toolchoice", "empty", "latency", "balance"]

EMPTY_CALLS_PER_CELL = 12  # probe_empty.py: 3 cases x 2 models x 12 = 72
DEBATE_WALL_SHARE_S = 20.0  # entry-side reference point for the latency report

Check = tuple  # (level, detail)


# ---------------------------------------------------------------------------
# Pure judges (no network). Integer math avoids float boundary surprises.
# ---------------------------------------------------------------------------

def judge_models(status_by_model: dict) -> list:
    """Map model -> {"status": int|None, "served": str|None, "error": str} to checks."""
    out = []
    for model, info in status_by_model.items():
        st = info.get("status")
        err = str(info.get("error") or "")
        if st == 200:
            out.append(("PASS", f"{model} 200 served={info.get('served') or '?'}"))
        elif model == HAIKU_5_5 and st == 400:
            out.append(("WARN", f"{model} 400 (expected on gateway): {err[:120]}"))
        elif st in (401, 403):
            out.append(("FAIL", f"{model} HTTP {st}: {err[:120]}"))
        elif st == 429 and "insufficient" in err.lower():
            out.append(("FAIL", f"{model} 429 insufficient balance: {err[:120]}"))
        elif st == 429:
            out.append(("WARN", f"{model} 429 rate-limited (transient): {err[:120]}"))
        else:
            out.append(("FAIL", f"{model} status={st}: {err[:120]}"))
    return out


def judge_structured(results: dict, n: int) -> list:
    """results: surface -> list of bool (schema-valid). PASS needs >= 7/8 (ceil(7n/8))."""
    out = []
    for surface, oks in results.items():
        ok = sum(1 for x in oks if x)
        passed = ok * 8 >= n * 7  # ok >= 7n/8
        level = "PASS" if passed else "FAIL"
        out.append((level, f"{surface} schema-valid {ok}/{n} (need >= 7/8)"))
    return out


def judge_toolchoice(text_count: int, total: int) -> Check:
    """Production shape (tool_choice none + instruction text block): TEXT >= 9/10."""
    if total <= 0:
        return ("FAIL", "production shape: no calls")
    passed = text_count * 10 >= total * 9
    return ("PASS" if passed else "FAIL", f"production shape TEXT {text_count}/{total} (need >= 9/10)")


def judge_empty(snapshot: dict, calls: int) -> Check:
    final = int(snapshot.get("empty_final", 0) or 0)
    retries = int(snapshot.get("empty_retries", 0) or 0)
    if calls <= 0:
        return ("FAIL", "no calls made")
    rate = final / calls
    detail = f"calls={calls} empty_retries={retries} empty_final={final} rate={rate:.2%}"
    if final == 0:
        return ("PASS", detail)
    if rate <= 0.01:
        return ("WARN", detail)
    return ("FAIL", detail)


def judge_latency(times: list, deadline: float) -> Check:
    if not times:
        return ("WARN", "no latency samples")
    within = sum(1 for t in times if t <= DEBATE_WALL_SHARE_S)
    detail = (
        f"pairs={len(times)} min={min(times):.1f}s median={statistics.median(times):.1f}s "
        f"max={max(times):.1f}s deadline={deadline:.1f}s within_20s={within}/{len(times)}"
    )
    if max(times) <= deadline:
        return ("PASS", detail)
    return ("WARN", detail)


def _parse_iso(value: Any) -> Optional[_dt.datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = _dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def judge_balance(payload: dict, now: _dt.datetime) -> Check:
    """payload: {"status": int|None, "body": dict|None}. Field names are tolerant
    because the dashboard schema is not documented in this repo."""
    status = payload.get("status")
    body = payload.get("body") or {}
    if status != 200:
        return ("FAIL", f"balance endpoint HTTP {status}")
    if not isinstance(body, dict):
        body = {}
    active = body.get("is_active")
    balance = None
    for key in ("balance_usd", "balance", "available_usd"):
        if isinstance(body.get(key), (int, float)):
            balance = float(body[key])
            break
    expiry = None
    for key in ("expires_at", "expiry", "expiration"):
        expiry = _parse_iso(body.get(key))
        if expiry is not None:
            break
    days = "n/a" if expiry is None else f"{int((expiry - now).total_seconds() // 86400)}"
    bal = "n/a" if balance is None else f"{balance:.2f}"
    detail = f"balance_usd={bal} days_to_expiry={days} is_active={active}"
    if active is False:
        return ("FAIL", detail + " (inactive)")
    if active is None:
        return ("WARN", detail + " (is_active not reported)")
    return ("PASS", detail)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scrub(text: Any, key: str) -> str:
    s = str(text)
    if key:
        s = s.replace(key, "***")
    return s[:200]


def _error_status(exc: BaseException) -> Optional[int]:
    return getattr(exc, "status_code", None)


def _error_text(exc: BaseException, key: str) -> str:
    return _scrub(getattr(exc, "message", None) or exc, key)


def _response_text(resp: Any) -> str:
    from agent.anthropic_utils import first_text  # lazy: agent import after .env load

    return first_text(resp)


# ---------------------------------------------------------------------------
# Network suites (client factories are injectable)
# ---------------------------------------------------------------------------

def run_models(client_factory: Callable[[], Any], models: list, key: str = "") -> dict:
    client = client_factory()
    out = {}
    for model in models:
        try:
            r = client.messages.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": "ping"}]
            )
            out[model] = {"status": 200, "served": getattr(r, "model", None), "error": ""}
        except Exception as e:  # noqa: BLE001 - probe records every failure class
            out[model] = {"status": _error_status(e), "served": None, "error": _error_text(e, key)}
    return out


def suite_models(client_factory, key: str = "") -> list:
    return judge_models(run_models(client_factory, MODELS_PING, key))


def _structured_payloads(cfg) -> dict:
    """surface -> (builder() -> (payload, schema)). Uses the REAL builders."""
    from agent.dpof import advisory_llm as a
    from agent.dpof import debate as dbt
    from agent.dpof import decider as d
    from agent.dpof import reflection as rf

    dec = d.DPOFDecider(cfg)
    llm = a.AdvisoryLLM()
    entry_ctx = (
        "ENTRY DELIBERATION (stand-in context for a compatibility probe, no real market data): "
        "SPY composite +0.19 (normal, near threshold 0.15), QQQ composite -0.02, regime TRENDING, "
        "VIX 17.2, no open positions, 1 candidate: key=SPY-C570-1S SPY C 570 1-session mid 1.45 "
        "delta 0.48. Decide enter or pass."
    )
    review_ctx = (
        "REVIEW (stand-in): open position SPY C 570 qty 2 entry 1.45 mark 1.62 pnl +11.7%; "
        "composite now +0.04 neutral; 45 minutes to flatten. Decide hold, trim or exit."
    )
    dinp = dbt.build_debate_input(
        underlying="SPY", direction="bullish", composite=0.19, strength="normal",
        pillars={"nofa": 0.31, "premium": 0.10, "darkpool": 0.02, "kronos": 0.20},
        regime="TRENDING",
        candidates=[{"key": "SPY-C570-1S", "unit_premium": 1.45, "delta": 0.48}],
    )
    trade = (
        '{"underlying": "SPY", "structure": "long call", "entry": 1.45, "exit": 1.62, '
        '"pnl_pct": 11.7, "held_min": 38, "exit_reason": "decider_exit", "thesis": "flow + kronos aligned"}'
    )
    use_fmt_dec = lambda: not dec._anthropic_format_downgraded  # noqa: E731
    use_fmt_llm = lambda: not llm._anthropic_format_downgraded  # noqa: E731
    return {
        "decider_entry": lambda: (
            dec._build_anthropic_payload(entry_ctx, d.ENTRY_SCHEMA, use_format=use_fmt_dec()),
            d.ENTRY_SCHEMA,
        ),
        "decider_review": lambda: (
            dec._build_anthropic_payload(review_ctx, d.REVIEW_SCHEMA, use_format=use_fmt_dec()),
            d.REVIEW_SCHEMA,
        ),
        "debate_bull": lambda: (
            llm._build_anthropic_payload(
                system=dbt._bull_system("entry"), user=dinp, schema=dbt._DEBATE_SCHEMA,
                model=cfg.debate_model, max_tokens=400, use_format=use_fmt_llm(), effort="low",
            ),
            dbt._DEBATE_SCHEMA,
        ),
        "debate_bear": lambda: (
            llm._build_anthropic_payload(
                system=dbt._bear_system("entry"), user=dinp, schema=dbt._DEBATE_SCHEMA,
                model=cfg.debate_model, max_tokens=400, use_format=use_fmt_llm(), effort="low",
            ),
            dbt._DEBATE_SCHEMA,
        ),
        "reflection": lambda: (
            llm._build_anthropic_payload(
                system=rf.REFLECTION_SYSTEM_PROMPT, user=trade, schema=rf.REFLECTION_SCHEMA,
                model=cfg.decision_model, max_tokens=1200, use_format=use_fmt_llm(), effort="low",
            ),
            rf.REFLECTION_SCHEMA,
        ),
    }


def run_structured(client_factory, cfg, n: int, workers: int = 6, key: str = "") -> dict:
    from agent.dpof import decider as d

    builders = _structured_payloads(cfg)
    client = client_factory()

    def call(surface: str) -> bool:
        try:
            payload, schema = builders[surface]()
            r = client.messages.create(**payload)
            parsed = d._extract_first_json_object(_response_text(r))
            return bool(parsed) and d._shape_matches(parsed, schema)
        except Exception as e:  # noqa: BLE001
            print(f"  note structured/{surface}: {type(e).__name__} {_error_text(e, key)[:120]}")
            return False

    jobs = [s for s in builders for _ in range(n)]
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(call, jobs))
    out = collections.defaultdict(list)
    for surface, ok in zip(jobs, res):
        out[surface].append(ok)
    return dict(out)


def suite_structured(client_factory, cfg, n: int = 8, key: str = "") -> list:
    return judge_structured(run_structured(client_factory, cfg, n, key=key), n)


_TOOLS = [
    {"name": "get_weather", "description": "Get weather for a city",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "get_time", "description": "Get local time for a city",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
]
_HIST = [
    {"role": "user", "content": "What is the weather and the local time in Paris?"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": {"city": "Paris"}},
    ]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01A", "content": "18C and sunny"}]},
]


def _toolchoice_shapes(summary: str) -> list:
    def with_instruction(tool_choice):
        h = [dict(m) for m in _HIST]
        h[-1] = {"role": "user", "content": h[-1]["content"] + [{"type": "text", "text": summary}]}
        return dict(tools=_TOOLS, tool_choice=tool_choice, messages=h)

    flat = [
        _HIST[0],
        {"role": "assistant", "content": "Let me check. [I called get_weather for Paris]"},
        {"role": "user", "content": "Tool result for get_weather(Paris): 18C and sunny.\n\n" + summary},
    ]
    return [
        ("V1 tools+auto+instruction in tool_result turn", lambda: with_instruction({"type": "auto"})),
        ("V2 NO tools param, tool blocks in history", lambda: dict(messages=list(_HIST))),
        ("V3 flattened plain text, NO tools", lambda: dict(messages=flat)),
        ("V4 PRODUCTION tool_choice none + instruction", lambda: with_instruction({"type": "none"})),
        ("V5 tools+auto+system instruction only",
         lambda: dict(tools=_TOOLS, tool_choice={"type": "auto"}, system=summary, messages=list(_HIST))),
    ]


def _outcome(r: Any) -> str:
    kinds = [getattr(b, "type", None) for b in (getattr(r, "content", None) or [])]
    if "tool_use" in kinds:
        return "TOOL_USE"
    return "TEXT" if _response_text(r).strip() else "EMPTY"


def run_toolchoice(client_factory, n: int, key: str = "") -> dict:
    from agent.core import _FORCED_SUMMARY_INSTRUCTION  # production instruction text

    client = client_factory()
    shapes = _toolchoice_shapes(_FORCED_SUMMARY_INSTRUCTION)

    def one(job):
        name, build, model = job
        try:
            r = client.messages.create(model=model, max_tokens=300, **build())
            return name, _outcome(r)
        except Exception as e:  # noqa: BLE001
            return name, f"ERR{_error_status(e) or type(e).__name__}"

    jobs = [(name, build, m) for name, build in shapes for m in TOOLCHOICE_MODELS for _ in range(n)]
    with cf.ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    counts = collections.defaultdict(collections.Counter)
    for name, outcome in res:
        counts[name][outcome] += 1
    return dict(counts)


def suite_toolchoice(client_factory, n: int = 5, key: str = "") -> list:
    counts = run_toolchoice(client_factory, n, key)
    out = []
    prod = None
    for name, c in counts.items():
        total = sum(c.values())
        detail = f"{name}: {dict(c)}"
        if name.startswith("V4"):
            prod = (c.get("TEXT", 0), total)
        else:
            out.append(("INFO", detail))
    if prod is None:
        out.append(("FAIL", "production shape V4 produced no results"))
    else:
        out.append(judge_toolchoice(*prod))
    return out


def run_empty(client_factory, cfg, key: str = "") -> tuple:
    """Returns (client_visible_empty_count, calls) for the 72-call matrix."""
    client = client_factory()
    tools = [{"name": "get_weather", "description": "Get the current weather for a city.",
              "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
    sys_prompt = "You are CLIO, an analyst assistant. Be concise and factual. " * 20
    cases = {
        "plain": dict(system=sys_prompt, messages=[{"role": "user", "content": "In one sentence, what does a rising VIX usually signal?"}]),
        "with_tools_no_call": dict(system=sys_prompt, tools=tools,
                                   messages=[{"role": "user", "content": "In one sentence, what does a rising VIX usually signal?"}]),
        "after_tool_result": dict(system=sys_prompt, tools=tools, messages=[
            {"role": "user", "content": "What is the weather in Paris? Use the tool then answer in one sentence."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01X", "name": "get_weather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01X", "content": "18C and sunny"}]},
        ]),
    }

    def one(job):
        model, case = job
        try:
            r = client.messages.create(model=model, max_tokens=300, **cases[case])
            return _outcome(r) == "EMPTY"
        except Exception:  # noqa: BLE001 - errors are not empties
            return False

    jobs = [(m, c) for m in TOOLCHOICE_MODELS for c in cases for _ in range(EMPTY_CALLS_PER_CELL)]
    with cf.ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    return sum(1 for x in res if x), len(jobs)


def suite_empty(client_factory, cfg, guard_snapshot_fn: Callable[[], dict], key: str = "") -> list:
    """guard_snapshot_fn returns the gateway stats snapshot; deltas are judged."""
    before = guard_snapshot_fn()
    client_empty, calls = run_empty(client_factory, cfg, key)
    after = guard_snapshot_fn()
    guarded = bool(getattr(cfg, "anthropic_gateway", False)) and os.environ.get("CLIO_ANTHROPIC_GATEWAY_TRANSPORT", "") != "0"
    delta = {
        "empty_retries": int(after.get("empty_retries", 0)) - int(before.get("empty_retries", 0)),
        "empty_final": int(after.get("empty_final", 0)) - int(before.get("empty_final", 0)),
    }
    out = []
    if not guarded:
        delta["empty_final"] = client_empty
        out.append(("WARN", f"guard transport not active; client-visible empties={client_empty}/{calls}"))
    out.append(judge_empty(delta, calls))
    return out


def run_latency(client_factory, cfg, pairs: int = 8, key: str = "") -> list:
    from agent.dpof import advisory_llm as a
    from agent.dpof import debate as dbt

    llm = a.AdvisoryLLM()
    client = client_factory()
    dinp = dbt.build_debate_input(
        underlying="SPY", direction="bullish", composite=0.19, strength="normal",
        pillars={"nofa": 0.31, "premium": 0.10, "darkpool": 0.02, "kronos": 0.20},
        regime="TRENDING",
        candidates=[{"key": "SPY-C570-1S", "unit_premium": 1.45, "delta": 0.48}],
    )
    sides = [
        llm._build_anthropic_payload(system=dbt._bull_system("entry"), user=dinp, schema=dbt._DEBATE_SCHEMA,
                                     model=cfg.debate_model, max_tokens=400,
                                     use_format=not llm._anthropic_format_downgraded, effort="low"),
        llm._build_anthropic_payload(system=dbt._bear_system("entry"), user=dinp, schema=dbt._DEBATE_SCHEMA,
                                     model=cfg.debate_model, max_tokens=400,
                                     use_format=not llm._anthropic_format_downgraded, effort="low"),
    ]

    def one_side(payload):
        try:
            client.messages.create(**payload)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  note latency: {type(e).__name__} {_error_text(e, key)[:120]}")
            return False

    def one_pair(_):
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(2) as inner:
            list(inner.map(one_side, sides))
        return time.monotonic() - t0

    with cf.ThreadPoolExecutor(pairs) as ex:
        times = list(ex.map(one_pair, range(pairs)))
    return times


def suite_latency(client_factory, cfg, pairs: int = 8, key: str = "") -> list:
    deadline = float(cfg.debate_timeout_seconds)  # env-driven via load_dpof_config()
    return [judge_latency(run_latency(client_factory, cfg, pairs, key), deadline)]


def run_balance(base_url: str, key: str, timeout: float = 20.0) -> dict:
    import httpx

    url = base_url.rstrip("/") + "/v1/dashboard/balance"
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=timeout, follow_redirects=True)
    except Exception as e:  # noqa: BLE001
        return {"status": None, "body": None, "error": _scrub(e, key)}
    try:
        body = r.json()
    except ValueError:
        body = None
    return {"status": r.status_code, "body": body if isinstance(body, dict) else None}


def suite_balance(base_url: str, key: str, now: Optional[_dt.datetime] = None) -> list:
    if not base_url:
        return [("WARN", "balance skipped: no gateway base URL configured")]
    payload = run_balance(base_url, key)
    if payload.get("status") is None:
        return [("FAIL", f"balance request failed: {payload.get('error', '')}")]
    return [judge_balance(payload, now or _dt.datetime.now(_dt.timezone.utc))]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_client_factory(cfg, timeout: float = 60.0) -> Callable[[], Any]:
    import anthropic

    def factory():
        return anthropic.Anthropic(**cfg.anthropic_client_kwargs(timeout=timeout, max_retries=0))

    return factory


def _print_checks(suite: str, checks: list) -> bool:
    failed = False
    for level, detail in checks:
        if level == "FAIL":
            failed = True
        print(f"{level} {suite}: {detail}")
    return failed


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only Anthropic gateway compatibility probe.")
    ap.add_argument("--suite", required=True, choices=ALL_SUITES + ["all"])
    ap.add_argument("--n", type=int, default=None, help="calls per cell (suite default if omitted)")
    args = ap.parse_args(argv)

    # Load .env BEFORE any agent import (config snapshots env at import time).
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    sys.path.insert(0, str(REPO_ROOT))
    from agent.config import config  # noqa: E402
    import agent.anthropic_gateway as gw  # noqa: E402

    key = config.anthropic_api_key or ""
    factory = _make_client_factory(config, timeout=60.0)
    suites = ALL_SUITES if args.suite == "all" else [args.suite]
    failed = False
    for suite in suites:
        try:
            if suite == "models":
                checks = suite_models(factory, key)
            elif suite == "structured":
                checks = suite_structured(factory, config_dpof(), args.n or 8, key=key)
            elif suite == "toolchoice":
                checks = suite_toolchoice(factory, args.n or 5, key)
            elif suite == "empty":
                checks = suite_empty(factory, config, lambda: gw.get_stats().snapshot(), key)
            elif suite == "latency":
                checks = suite_latency(factory, config_dpof(), args.n or 8, key)
            else:
                checks = suite_balance(getattr(config, "anthropic_base_url", "") or "", key)
        except Exception as e:  # noqa: BLE001 - one suite must not hide the others
            checks = [("FAIL", f"suite crashed: {type(e).__name__} {_scrub(e, key)}")]
        failed = _print_checks(suite, checks) or failed
    return 1 if failed else 0


def config_dpof():
    from agent.dpof.config import load_dpof_config

    return load_dpof_config()


if __name__ == "__main__":
    sys.exit(main())
