"""
CLIO DPOF -- advisory LLM client (R5, design frame `docs/design/clio_dpof_learning_loop.md`).

A separate, self-contained Anthropic/OpenAI client used ONLY by the "learning
loop" additions on top of DPOF: F1 post-trade reflection (`reflection.py`,
built by B2) and F2 borderline-call debate (`debate.py`, this same builder).
It never touches the live decider (`agent/dpof/decider.py`) and is never on
the path that turns a decision into an order -- everything `complete_json`
returns is CONTEXT for the decider's own decision model, never a vote or a
trade. `decider.py` is READ-ONLY reference material here; nothing is
imported from it (its private helpers stay private) and it is never modified.

Provider order per call: Anthropic (`primary_model`) -> on error / timeout /
invalid-JSON / open circuit -> OpenAI (`fallback_model`) -> `ok=False`. Each
provider has its OWN circuit breaker (`circuit_failures` / a shared
`circuit_cooldown_seconds`), separate from the decider's own breakers, and is
keyed by PROVIDER only (not by model) -- reflection and debate share one
Anthropic model slot's circuit even though they pass different
`primary_model` values, matching the decider's "one breaker per provider"
shape.

Deadlines: each provider attempt gets its OWN fresh budget of `timeout_seconds`
computed from "now" at the moment that attempt starts (mirrors decider.py's
`decision_timeout_seconds` / `fallback_timeout_seconds` split exactly, just
with one shared knob instead of two cfg fields) -- a primary that times out
using its full budget does not shrink the fallback's own budget. Every
`asyncio.wait_for` uses this per-attempt budget; a budget at or below
`_MIN_CALL_BUDGET_SECONDS` skips that attempt entirely ("no_budget").

Reused from `decider.py`'s (in turn from `jev.py`'s) proven patterns --
copied and adapted, never imported, per R5:
  - `_get_anthropic_client` (rebuild the client only when key/timeout change)
  - `_build_anthropic_payload` / one-time `output_config.format` downgrade on
    a 400 (`_is_output_config_400`) then parse the first JSON object from
    text (`_extract_first_json_object`)
  - `_sync_anthropic_call` always run via `run_in_executor` -- never on the
    event loop
  - `_anthropic_response_to_dict` / `_anthropic_stop_reason_note`
  - the raw-httpx OpenAI leg (`follow_redirects=False`, `_MAX_RESPONSE_BYTES`
    cap BEFORE any JSON parse, `text.format` json_schema strict)
  - the `_ProviderStatus` circuit breaker (failures -> open for cooldown ->
    half-open lets exactly one call through)
  - `_redact` (`Bearer <token>` + any configured secret stripped before an
    error message is ever logged or stored)

Never raises: every path returns an `AdvisoryResult` with `ok=False` and a
redacted `error` string. Budget: `budget_fn(name)` is awaited ONCE per
`complete_json` call, gating the WHOLE primary-then-fallback attempt
sequence (a fallback after a primary failure is still one logical
reflection/debate-side call, not two budget units) -- any exception from
`budget_fn` (typically `agent.rate_limiter.BudgetExhaustedError`, but any
exception is treated the same way, fail-closed) means `ok=False,
error="budget"`, no provider is called at all.

System prompt is passed PER CALL (there is no CHARTER here -- reflection and
debate each own their own short system prompt); it is sent to Anthropic as a
plain `system` string (no prompt-cache block -- these prompts are short and
purpose-specific, unlike the decider's large cached CHARTER) and to OpenAI as
a `developer` role message, matching the Responses API shape `decider.py`
already uses.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

import anthropic
import httpx

from agent.dpof.schema_prompt import schema_instruction
import json

logger = logging.getLogger("clio.dpof.advisory_llm")

_MIN_CALL_BUDGET_SECONDS = 0.05
_MAX_RESPONSE_BYTES = 1 * 1024 * 1024  # 1 MiB -- an advisory answer is never legitimately this large
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Not exposed as complete_json() parameters (the §6 interface has none) --
# fixed, conservative defaults for the "faster/lighter" advisory calls this
# module serves (reflection, debate) as opposed to the decider's own
# judgment-heavy call. Hardcoded here, NOT read from DPOFConfig (B1 stays
# config-free per the build slice).
_DEFAULT_ANTHROPIC_EFFORT = "low"
_DEFAULT_OPENAI_EFFORT = "low"
# complete_json(effort=...) accepts these (same vocabulary on both providers);
# anything else falls back to "low". Reflection asks for more than the debate.
_VALID_EFFORTS = ("low", "medium", "high")


# ─────────────────────────────────────────────────────────────────────────
# Public contract (design frame §6)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class AdvisoryResult:
    ok: bool
    parsed: Optional[dict]
    provider: str
    model: str
    latency_ms: Optional[int]
    error: str


class _AdvisoryHTTPError(Exception):
    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"http_{status_code}" + (f":{detail}" if detail else ""))


@dataclass
class _ProviderStatus:
    circuit_open_until: Optional[float] = None
    consecutive_failures: int = 0
    last_error: str = ""
    last_latency_ms: Optional[int] = None
    day: str = ""
    calls_today: int = 0
    failures_today: int = 0


# ─────────────────────────────────────────────────────────────────────────
# PURE helpers -- copied/adapted from decider.py, never imported (R5)
# ─────────────────────────────────────────────────────────────────────────

def _valid_model(model: Any) -> bool:
    """Model ids must be plain, non-empty strings."""
    return isinstance(model, str) and bool(model.strip())


def _extract_first_json_object(text: str) -> Optional[dict]:
    """Scan for the first balanced {...} object, respecting quoted strings.

    Used on the Anthropic output_config-downgrade path and as a last-resort
    parse for a malformed/fake response, so the text may have prose or
    markdown fencing around the JSON.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _try_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return _extract_first_json_object(text)


def _shape_matches(parsed: Any, schema: dict) -> bool:
    """Rudimentary structural check (required keys present, gross type match).

    Defense-in-depth, not a full JSON-schema validator -- the provider's own
    structured-output enforcement (or the strict-mode fallback) is what
    actually guarantees the shape on the happy path; this catches a bad
    downgrade-path extraction or a malformed fake in tests.
    """
    if not isinstance(parsed, dict):
        return False
    for key in schema.get("required", []):
        if key not in parsed:
            return False
    props = schema.get("properties", {})
    for key, val in parsed.items():
        spec = props.get(key)
        if not isinstance(spec, dict):
            continue
        t = spec.get("type")
        if t == "string" and not isinstance(val, str):
            return False
        if t == "number" and (isinstance(val, bool) or not isinstance(val, (int, float))):
            return False
        if t == "integer" and (isinstance(val, bool) or not isinstance(val, int)):
            return False
    return True


def _is_output_config_400(exc: Exception) -> bool:
    """True if `exc` looks like an Anthropic 400 rejecting output_config/format."""
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    msg = str(exc)
    mentions = "output_config" in msg or "output config" in msg.lower()
    if status == 400 and mentions:
        return True
    if status is None and "400" in msg and mentions:
        return True
    return False


def _redact(msg: str, *secrets: str) -> str:
    if not msg:
        return msg
    out = _BEARER_RE.sub("Bearer ***", msg)
    for s in secrets:
        if s:
            out = out.replace(s, "***")
    return out


def _anthropic_stop_reason_note(resp: Any) -> str:
    """"" when there's nothing worth flagging; otherwise a short diagnostic
    tag for a response that structurally can't have produced a usable
    answer. `stop_details` is populated only when stop_reason == "refusal"
    (null otherwise) -- always guarded before reading."""
    stop_reason = getattr(resp, "stop_reason", None)
    if stop_reason == "max_tokens":
        return "max_tokens"
    if stop_reason == "refusal":
        details = getattr(resp, "stop_details", None)
        category = getattr(details, "category", None) if details is not None else None
        return f"refusal:{category}" if category else "refusal"
    return ""


def _anthropic_response_to_dict(resp: Any, default_model: str) -> dict:
    text_parts: list[str] = []
    for block in getattr(resp, "content", None) or []:
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", "") or "")
        # thinking blocks are intentionally ignored
    usage_obj = getattr(resp, "usage", None)
    usage: dict = {}
    if usage_obj is not None:
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
            "cache_read_input_tokens": getattr(usage_obj, "cache_read_input_tokens", None),
        }
    return {
        "text": "".join(text_parts),
        "usage": usage,
        "model": getattr(resp, "model", default_model) or default_model,
        "stop_note": _anthropic_stop_reason_note(resp),
    }


def _openai_response_to_dict(body: Any, default_model: str) -> dict:
    text = ""
    if isinstance(body, dict):
        ot = body.get("output_text")
        if isinstance(ot, str) and ot:
            text = ot
        else:
            chunks: list[str] = []
            for item in body.get("output", []) or []:
                if not isinstance(item, dict):
                    continue
                for c in item.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        t = c.get("text")
                        if isinstance(t, str):
                            chunks.append(t)
            text = "".join(chunks)
    model = (body.get("model") if isinstance(body, dict) else None) or default_model
    return {"text": text, "usage": {}, "model": model}


# ─────────────────────────────────────────────────────────────────────────
# Default key / budget functions (lazy imports -- avoid import cycles &
# startup failures if these subsystems aren't wired yet)
# ─────────────────────────────────────────────────────────────────────────

def _default_anthropic_key() -> str:
    try:
        from agent.config import config as _clio_cfg  # lazy import

        return _clio_cfg.anthropic_api_key or ""
    except Exception as e:
        logger.debug(f"advisory_llm: default anthropic_key_fn failed: {type(e).__name__}: {e}")
        return ""


def _default_openai_key() -> str:
    import os

    return os.environ.get("OPENAI_API_KEY", "") or ""


async def _default_budget_fn(name: str) -> None:
    from agent.rate_limiter import check_and_record  # lazy import

    await check_and_record(name)


# ─────────────────────────────────────────────────────────────────────────
# AdvisoryLLM
# ─────────────────────────────────────────────────────────────────────────

class AdvisoryLLM:
    def __init__(
        self,
        *,
        anthropic_key_fn: Callable[[], str] = _default_anthropic_key,
        openai_key_fn: Callable[[], str] = _default_openai_key,
        anthropic_transport: Optional[Callable[[dict], Awaitable[dict]]] = None,
        openai_transport: Optional[Callable[[dict], Awaitable[dict]]] = None,
        budget_fn: Optional[Callable[[str], Awaitable[None]]] = None,
        clock: Callable[[], float] = time.time,
        circuit_failures: int = 3,
        circuit_cooldown_seconds: float = 300.0,
        openai_base_url: str = _DEFAULT_OPENAI_BASE_URL,
    ) -> None:
        self._anthropic_key_fn = anthropic_key_fn
        self._openai_key_fn = openai_key_fn
        self._anthropic_transport = anthropic_transport
        self._openai_transport = openai_transport
        self._budget_fn = budget_fn or _default_budget_fn
        self._clock = clock
        self._circuit_failures = circuit_failures
        self._circuit_cooldown_seconds = circuit_cooldown_seconds
        self._openai_base_url = (openai_base_url or _DEFAULT_OPENAI_BASE_URL).rstrip("/")

        self._anthropic_status = _ProviderStatus()
        self._openai_status = _ProviderStatus()
        self._anthropic_format_downgraded = self._gateway_needs_prompt_schema()

        self._anthropic_client: Optional[anthropic.Anthropic] = None
        self._anthropic_client_key: Optional[str] = None
        self._anthropic_client_timeout: Optional[float] = None
        self._openai_client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        schema: dict,
        schema_name: str,
        primary_model: str,
        fallback_model: str,
        timeout_seconds: float,
        max_tokens: int = 800,
        budget: Optional[str] = None,
        effort: str = _DEFAULT_ANTHROPIC_EFFORT,
    ) -> AdvisoryResult:
        try:
            return await self._complete_json_inner(
                purpose=purpose,
                system=system,
                user=user,
                schema=schema,
                schema_name=schema_name,
                primary_model=primary_model,
                fallback_model=fallback_model,
                timeout_seconds=timeout_seconds,
                max_tokens=max_tokens,
                budget=budget,
                effort=effort,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Belt-and-braces: complete_json must NEVER raise. Every known
            # failure mode is already handled inside _complete_json_inner /
            # _call_anthropic / _call_openai; this only catches a genuine bug.
            logger.warning(f"advisory_llm[{purpose}]: unexpected internal error: {type(e).__name__}: {e}")
            return AdvisoryResult(
                ok=False, parsed=None, provider="", model="", latency_ms=None,
                error=f"internal_error:{type(e).__name__}",
            )

    async def _complete_json_inner(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        schema: dict,
        schema_name: str,
        primary_model: str,
        fallback_model: str,
        timeout_seconds: float,
        max_tokens: int,
        budget: Optional[str],
        effort: str = _DEFAULT_ANTHROPIC_EFFORT,
    ) -> AdvisoryResult:
        if budget:
            try:
                await self._budget_fn(budget)
            except Exception as e:
                # Any budget_fn exception (typically BudgetExhaustedError, but
                # a custom budget_fn may raise something else) is treated the
                # same way: fail closed, no provider call at all.
                logger.info(f"advisory_llm[{purpose}]: budget check failed for {budget!r}: {type(e).__name__}: {e}")
                return AdvisoryResult(ok=False, parsed=None, provider="", model="", latency_ms=None, error="budget")

        effort = effort if effort in _VALID_EFFORTS else _DEFAULT_ANTHROPIC_EFFORT
        parsed, meta = await self._call_anthropic(
            purpose=purpose, system=system, user=user, schema=schema,
            model=primary_model, max_tokens=max_tokens, timeout_seconds=timeout_seconds, effort=effort,
        )
        if parsed is not None:
            self._log_outcome(purpose, "anthropic", meta.get("model", primary_model), meta.get("latency_ms"), True)
            return AdvisoryResult(
                ok=True, parsed=parsed, provider="anthropic",
                model=meta.get("model", primary_model), latency_ms=meta.get("latency_ms"), error="",
            )
        self._log_outcome(purpose, "anthropic", meta.get("model", primary_model), meta.get("latency_ms"), False)

        parsed2, meta2 = await self._call_openai(
            purpose=purpose, system=system, user=user, schema=schema, schema_name=schema_name,
            model=fallback_model, max_tokens=max_tokens, timeout_seconds=timeout_seconds, effort=effort,
        )
        if parsed2 is not None:
            self._log_outcome(purpose, "openai", meta2.get("model", fallback_model), meta2.get("latency_ms"), True)
            return AdvisoryResult(
                ok=True, parsed=parsed2, provider="openai",
                model=meta2.get("model", fallback_model), latency_ms=meta2.get("latency_ms"), error="",
            )
        self._log_outcome(purpose, "openai", meta2.get("model", fallback_model), meta2.get("latency_ms"), False)

        combined_error = f"anthropic:{meta.get('error')}|openai:{meta2.get('error')}"
        return AdvisoryResult(ok=False, parsed=None, provider="", model="", latency_ms=None, error=combined_error)

    def status(self) -> dict:
        return {
            "providers": {
                "anthropic": self._status_dict(self._anthropic_status),
                "openai": self._status_dict(self._openai_status),
            },
        }

    async def close(self) -> None:
        if self._openai_client is not None:
            client, self._openai_client = self._openai_client, None
            try:
                await client.aclose()
            except Exception as e:
                logger.debug(f"AdvisoryLLM.close: openai client aclose failed: {type(e).__name__}: {e}")
        if self._anthropic_client is not None:
            client, self._anthropic_client = self._anthropic_client, None
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, client.close)
            except Exception as e:
                logger.debug(f"AdvisoryLLM.close: anthropic client close failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Logging (never the prompt text)
    # ------------------------------------------------------------------

    def _log_outcome(self, purpose: str, provider: str, model: str, latency_ms: Optional[int], ok: bool) -> None:
        logger.info(f"advisory_llm: purpose={purpose} provider={provider} model={model} latency_ms={latency_ms} ok={ok}")

    # ------------------------------------------------------------------
    # Circuit breaker + daily counters (in-memory only -- no persistence;
    # a restart naturally clears both, which is fine for observability
    # counters backed by the durable, cross-restart budget_fn/rate_limiter)
    # ------------------------------------------------------------------

    def _et_day_str(self) -> str:
        dt = datetime.fromtimestamp(self._clock(), tz=ZoneInfo("America/New_York"))
        return dt.strftime("%Y-%m-%d")

    def _roll_day(self, status: _ProviderStatus) -> None:
        today = self._et_day_str()
        if status.day != today:
            status.day = today
            status.calls_today = 0
            status.failures_today = 0

    def _record_success(self, status: _ProviderStatus) -> None:
        status.consecutive_failures = 0
        status.circuit_open_until = None
        status.last_error = ""
        status.calls_today += 1

    def _record_failure(self, status: _ProviderStatus, error: str) -> None:
        status.consecutive_failures += 1
        status.last_error = error
        status.calls_today += 1
        status.failures_today += 1
        if status.consecutive_failures >= max(1, self._circuit_failures):
            status.circuit_open_until = self._clock() + max(0.0, self._circuit_cooldown_seconds)

    def _status_dict(self, status: _ProviderStatus) -> dict:
        self._roll_day(status)
        return {
            "circuit_open_until": status.circuit_open_until,
            "consecutive_failures": status.consecutive_failures,
            "last_error": status.last_error,
            "last_latency_ms": status.last_latency_ms,
            "calls_today": status.calls_today,
            "failures_today": status.failures_today,
        }

    # ------------------------------------------------------------------
    # Anthropic (primary)
    # ------------------------------------------------------------------

    def _get_anthropic_client(self, api_key: str, timeout: float) -> anthropic.Anthropic:
        from agent.config import config as _c  # lazy import

        kwargs = _c.anthropic_client_kwargs(timeout=timeout)
        kwargs["api_key"] = api_key  # caller-supplied key wins (injection in tests)
        sig = (api_key, kwargs.get("base_url"), timeout)
        if self._anthropic_client is None or self._anthropic_client_key != sig:
            self._anthropic_client = anthropic.Anthropic(**kwargs)
            self._anthropic_client_key = sig
            self._anthropic_client_timeout = timeout
        return self._anthropic_client

    def _gateway_needs_prompt_schema(self) -> bool:
        """True when routed via a gateway that ignores output_config.format (OP-05)."""
        try:
            from agent.config import config as clio_config  # lazy import

            return bool(getattr(clio_config, "anthropic_gateway", False))
        except Exception:
            return False

    def _build_anthropic_payload(
        self, *, system: str, user: str, schema: dict, model: str, max_tokens: int, use_format: bool,
        effort: str = _DEFAULT_ANTHROPIC_EFFORT,
    ) -> dict:
        output_config: dict = {"effort": effort}
        if use_format:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        else:
            # Gateway / downgraded path: the schema must be stated in the prompt.
            system = system + "\n\n" + schema_instruction(schema)
        return {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }

    def _sync_anthropic_call(self, payload: dict, api_key: str, client_timeout: float) -> dict:
        """Runs on the executor thread pool -- never on the event loop."""
        client = self._get_anthropic_client(api_key, client_timeout)
        kwargs = dict(payload)
        try:
            resp = client.messages.create(**kwargs)
        except TypeError as e:
            # Defensive: an older installed SDK without a native output_config
            # kwarg. Retry once via extra_body (same request shape on the wire).
            if "output_config" not in str(e):
                raise
            oc = kwargs.pop("output_config", None)
            resp = client.messages.create(extra_body={"output_config": oc}, **kwargs)
        return _anthropic_response_to_dict(resp, payload.get("model", ""))

    async def _run_anthropic_transport(self, payload: dict, budget: float, api_key: str, client_timeout: float) -> dict:
        t = max(0.01, budget)
        if self._anthropic_transport is not None:
            return await asyncio.wait_for(self._anthropic_transport(payload), timeout=t)
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: self._sync_anthropic_call(payload, api_key, client_timeout)),
            timeout=t,
        )

    async def _call_anthropic(
        self, *, purpose: str, system: str, user: str, schema: dict, model: str, max_tokens: int, timeout_seconds: float,
        effort: str = _DEFAULT_ANTHROPIC_EFFORT,
    ) -> tuple[Optional[dict], dict]:
        status = self._anthropic_status
        self._roll_day(status)

        if not _valid_model(model):
            return None, {"error": "invalid_model", "model": model, "latency_ms": None}

        api_key = self._anthropic_key_fn() or ""
        if not api_key:
            return None, {"error": "no_key", "model": model, "latency_ms": None}

        if status.circuit_open_until is not None:
            if self._clock() < status.circuit_open_until:
                return None, {"error": "circuit_open", "model": model, "latency_ms": None}
            status.circuit_open_until = None  # half-open: let exactly this call through

        deadline_at = self._clock() + max(0.0, timeout_seconds)
        budget = deadline_at - self._clock()
        if budget <= _MIN_CALL_BUDGET_SECONDS:
            self._record_failure(status, "no_budget")
            return None, {"error": "no_budget", "model": model, "latency_ms": None}

        use_format = not self._anthropic_format_downgraded
        payload = self._build_anthropic_payload(
            system=system, user=user, schema=schema, model=model, max_tokens=max_tokens, use_format=use_format, effort=effort,
        )

        t0 = self._clock()
        try:
            result = await self._run_anthropic_transport(payload, budget, api_key, timeout_seconds)
        except asyncio.TimeoutError:
            self._record_failure(status, "timeout")
            return None, {"error": "timeout", "model": model, "latency_ms": None}
        except Exception as e:
            if use_format and _is_output_config_400(e):
                if not self._anthropic_format_downgraded:
                    self._anthropic_format_downgraded = True
                    logger.warning(
                        "advisory_llm: Anthropic rejected output_config.format (HTTP 400); "
                        "downgrading to prompt-only JSON + text extraction for the rest of this process."
                    )
                budget2 = deadline_at - self._clock()
                if budget2 <= _MIN_CALL_BUDGET_SECONDS:
                    self._record_failure(status, "no_budget_after_downgrade")
                    return None, {"error": "no_budget_after_downgrade", "model": model, "latency_ms": None}
                payload2 = self._build_anthropic_payload(
                    system=system, user=user, schema=schema, model=model, max_tokens=max_tokens, use_format=False, effort=effort,
                )
                t0 = self._clock()
                try:
                    result = await self._run_anthropic_transport(payload2, budget2, api_key, timeout_seconds)
                except asyncio.TimeoutError:
                    self._record_failure(status, "timeout_after_downgrade")
                    return None, {"error": "timeout_after_downgrade", "model": model, "latency_ms": None}
                except Exception as e2:
                    err = _redact(f"{type(e2).__name__}: {e2}", api_key)
                    self._record_failure(status, err)
                    return None, {"error": err, "model": model, "latency_ms": None}
            else:
                err = _redact(f"{type(e).__name__}: {e}", api_key)
                self._record_failure(status, err)
                return None, {"error": err, "model": model, "latency_ms": None}

        latency_ms = int((self._clock() - t0) * 1000)
        status.last_latency_ms = latency_ms

        text = result.get("text", "") if isinstance(result, dict) else ""
        model_used = (result.get("model") if isinstance(result, dict) else None) or model

        parsed = _try_parse_json(text)
        if parsed is None or not _shape_matches(parsed, schema):
            stop_note = result.get("stop_note", "") if isinstance(result, dict) else ""
            reason = f"invalid_json:{stop_note}" if stop_note else "invalid_json"
            self._record_failure(status, reason)
            return None, {"error": reason, "model": model_used, "latency_ms": latency_ms}

        self._record_success(status)
        return parsed, {"error": "", "model": model_used, "latency_ms": latency_ms}

    # ------------------------------------------------------------------
    # OpenAI (fallback)
    # ------------------------------------------------------------------

    def _get_openai_client(self) -> httpx.AsyncClient:
        if self._openai_client is None or self._openai_client.is_closed:
            self._openai_client = httpx.AsyncClient(follow_redirects=False)
        return self._openai_client

    async def _openai_http_call(self, payload: dict, timeout: float, api_key: str) -> dict:
        client = self._get_openai_client()
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = f"{self._openai_base_url}/responses"
        resp = await client.post(url, headers=headers, json=payload, timeout=timeout)

        oversized = False
        cl = resp.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > _MAX_RESPONSE_BYTES:
                    oversized = True
            except (TypeError, ValueError):
                pass
        if not oversized and len(resp.content) > _MAX_RESPONSE_BYTES:
            oversized = True
        if oversized:
            raise ValueError("openai_response_oversized")

        if resp.status_code != 200:
            detail = ""
            try:
                j = resp.json()
                if isinstance(j, dict):
                    err = j.get("error")
                    if isinstance(err, dict):
                        detail = str(err.get("message") or "")
                    elif isinstance(err, str):
                        detail = err
            except Exception:
                pass
            raise _AdvisoryHTTPError(resp.status_code, detail[:200])

        try:
            body = resp.json()
        except Exception as e:
            raise ValueError(f"invalid_json_response:{type(e).__name__}") from e
        return _openai_response_to_dict(body, payload.get("model", ""))

    async def _run_openai_transport(self, payload: dict, timeout: float, api_key: str) -> dict:
        t = max(0.01, timeout)
        if self._openai_transport is not None:
            return await asyncio.wait_for(self._openai_transport(payload), timeout=t)
        return await asyncio.wait_for(self._openai_http_call(payload, t, api_key), timeout=t)

    async def _call_openai(
        self, *, purpose: str, system: str, user: str, schema: dict, schema_name: str, model: str,
        max_tokens: int, timeout_seconds: float, effort: str = _DEFAULT_OPENAI_EFFORT,
    ) -> tuple[Optional[dict], dict]:
        status = self._openai_status
        self._roll_day(status)

        api_key = self._openai_key_fn() or ""
        if not _valid_model(model) or not api_key:
            return None, {"error": "fallback_disabled", "model": model, "latency_ms": None}

        if status.circuit_open_until is not None:
            if self._clock() < status.circuit_open_until:
                return None, {"error": "circuit_open", "model": model, "latency_ms": None}
            status.circuit_open_until = None

        deadline_at = self._clock() + max(0.0, timeout_seconds)
        budget = deadline_at - self._clock()
        if budget <= _MIN_CALL_BUDGET_SECONDS:
            self._record_failure(status, "no_budget")
            return None, {"error": "no_budget", "model": model, "latency_ms": None}

        payload = {
            "model": model,
            "input": [
                {"role": "developer", "content": system},
                {"role": "user", "content": user},
            ],
            "reasoning": {"effort": effort},
            "text": {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
            "max_output_tokens": max_tokens,
        }

        t0 = self._clock()
        try:
            result = await self._run_openai_transport(payload, budget, api_key)
        except asyncio.TimeoutError:
            self._record_failure(status, "timeout")
            return None, {"error": "timeout", "model": model, "latency_ms": None}
        except Exception as e:
            err = _redact(f"{type(e).__name__}: {e}", api_key)
            self._record_failure(status, err)
            return None, {"error": err, "model": model, "latency_ms": None}

        latency_ms = int((self._clock() - t0) * 1000)
        status.last_latency_ms = latency_ms

        text = result.get("text", "") if isinstance(result, dict) else ""
        model_used = (result.get("model") if isinstance(result, dict) else None) or model

        parsed = _try_parse_json(text)
        if parsed is None or not _shape_matches(parsed, schema):
            self._record_failure(status, "invalid_json")
            return None, {"error": "invalid_json", "model": model_used, "latency_ms": latency_ms}

        self._record_success(status)
        return parsed, {"error": "", "model": model_used, "latency_ms": latency_ms}
