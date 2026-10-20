"""CLIO decisions producer — config.

Runtime-read like `agent.dpof.config.DPOFConfig`: `from_env()` calls
`os.environ.get()` fresh each time it is invoked, not at module-import time —
a restart is the only way a `.env` change here takes effect, matching the
rest of `agent/dpof/`'s own convention. Dataclass defaults and `from_env()`'s
fallbacks are the SAME values (CLAUDE.md's "both sites must agree" rule) —
`from_env()` builds its fallbacks directly off `DecisionsConfig()`'s own
defaults, so they can never drift apart.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("clio.decisions.config")


def _repo_root() -> str:
    """agent/decisions/config.py needs 3 levels of dirname to reach the repo
    root (same depth as agent/integrations/*.py — CLAUDE.md gotcha)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_OUTBOX_PATH = os.path.join(_repo_root(), "data", "decisions", "outbox.jsonl")

_VALID_SERVE_MODES = ("jev",)  # DESIGN.md §6: only "jev" exists until a local model first passes the gate


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val.strip() == "" else val.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning(f"agent.decisions config: invalid float for {name}={raw!r}; using default {default}")
        return default
    return max(lo, min(hi, val))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"agent.decisions config: invalid int for {name}={raw!r}; using default {default}")
        return default


def _parse_shadow_endpoints(raw: str) -> tuple[tuple[str, str], ...]:
    """"name=url,name=url" -> (("name", "url"), ...). A malformed entry
    (no "=", empty name, or a url that isn't http(s)) is skipped; the whole
    var is never rejected wholesale, so one typo doesn't silently disable
    every OTHER configured endpoint. At most one warning per call."""
    if not raw or not raw.strip():
        return ()
    out: list[tuple[str, str]] = []
    any_bad = False
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            any_bad = True
            continue
        name, _, url = part.partition("=")
        name = name.strip()
        url = url.strip()
        if not name or not url.lower().startswith(("http://", "https://")):
            any_bad = True
            continue
        out.append((name, url))
    if any_bad:
        logger.warning(
            "agent.decisions config: CLIO_DECISIONS_SHADOW_ENDPOINTS had one or more malformed "
            "entries ('name=http(s)://...' expected); those were skipped"
        )
    return tuple(out)


@dataclass(frozen=True)
class DecisionsConfig:
    enabled: bool = True
    shadow: bool = True
    shadow_endpoints: tuple[tuple[str, str], ...] = ()
    shadow_timeout_seconds: float = 2.0
    outbox_path: str = DEFAULT_OUTBOX_PATH
    outbox_max_lines: int = 20000
    serve_mode: str = "jev"


def from_env() -> DecisionsConfig:
    d = DecisionsConfig()  # defaults — single source for fallbacks

    serve_mode = _env_str("CLIO_DECISIONS_SERVE_MODE", d.serve_mode)
    if serve_mode not in _VALID_SERVE_MODES:
        logger.warning(
            f"agent.decisions config: CLIO_DECISIONS_SERVE_MODE={serve_mode!r} not recognized "
            f" -- serving 'jev'"
        )
        serve_mode = "jev"

    return DecisionsConfig(
        enabled=_env_bool("CLIO_DECISIONS_ENABLED", d.enabled),
        shadow=_env_bool("CLIO_DECISIONS_SHADOW", d.shadow),
        shadow_endpoints=_parse_shadow_endpoints(_env_str("CLIO_DECISIONS_SHADOW_ENDPOINTS", "")),
        shadow_timeout_seconds=_env_float(
            "CLIO_DECISIONS_SHADOW_TIMEOUT", d.shadow_timeout_seconds, lo=0.2, hi=10.0,
        ),
        outbox_path=_env_str("CLIO_DECISIONS_OUTBOX_PATH", d.outbox_path),
        outbox_max_lines=max(1, _env_int("CLIO_DECISIONS_OUTBOX_MAX_LINES", d.outbox_max_lines)),
        serve_mode=serve_mode,
    )
