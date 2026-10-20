"""
CLIO Kronos — configuration (env-driven; dataclass defaults MUST match
from_env() defaults, per CLAUDE.md's coding standard).

Deliberately independent of `agent.config` and `agent.dpof.config`: this
package is a separate long-running process with its own systemd unit (see
`systemd/clio-kronos.service`), same shape as `agent.flowcollector.config`
(see that module's own docstring for the rationale — a separate process
never wants to inherit the main agent's whole config surface). Every knob
is read fresh in `from_env()` (runtime-evaluated, matching the CLIO_* flag
family convention) — a `.env` edit + service restart is the only deploy
step.

Alpaca credentials: reuses CLIO's existing shared PAPER data keys
(`ALPACA_PAPER_KEY_ID`/`ALPACA_PAPER_SECRET_KEY`), the same fallback
`agent.dpof.broker`/`agent.flowcollector` use — see `bars.py`'s module
docstring for the feed-downgrade pattern this mirrors. Never logs a key
value (see `config_summary` below).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields

logger = logging.getLogger("clio.kronos.config")

# agent/kronos/config.py -> 3x dirname = repo root (same depth as
# agent/integrations/*.py -- CLAUDE.md "dirname level counting").
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pinned upstream/HF identifiers -- see agent/kronos/vendor/VENDOR_PIN.md
# §1/§3 for the provenance record these must match. Module-level constants
# (not just defaults) so tests and VENDOR_PIN.md can be cross-checked
# against one source.
UPSTREAM_VENDOR_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
DEFAULT_MODEL_ID = "NeoQuasar/Kronos-small"
DEFAULT_MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
DEFAULT_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"

DEFAULT_HISTORY_FILE = os.path.join(_REPO_ROOT, "data", "intelligence", "kronos_history.jsonl")


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val.strip() == "" else val.strip()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(f"kronos config: invalid float for {name}={raw!r}; using default {default}")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"kronos config: invalid int for {name}={raw!r}; using default {default}")
        return default


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    items = tuple(s.strip().upper() for s in raw.split(",") if s.strip())
    return items or default


def _clamp_int(value: int, lo: int, hi: int, name: str) -> int:
    clamped = max(lo, min(hi, value))
    if clamped != value:
        logger.warning(f"kronos config: {name}={value} outside [{lo}, {hi}]; using {clamped}")
    return clamped


@dataclass(frozen=True)
class KronosConfig:
    # ── Universe ─────────────────────────────────────────────────────────
    symbols: tuple[str, ...] = ("SPY", "QQQ")                     # CLIO_KRONOS_SYMBOLS

    # ── Sampling / model ─────────────────────────────────────────────────
    sample_paths: int = 16                # CLIO_KRONOS_SAMPLE_PATHS, clamp 2-64
    lookback_bars: int = 400              # CLIO_KRONOS_LOOKBACK_BARS, clamp 64-512
    torch_threads: int = 6                # CLIO_KRONOS_TORCH_THREADS, clamp 1-8 (VPS sweep 2026-09-27: 4/6/8 -> ~94/77/69 s)
    run_timeout_seconds: float = 240.0    # CLIO_KRONOS_RUN_TIMEOUT (runs are 300 s apart; a VPS run measured ~91 s)
    temperature: float = 1.0              # CLIO_KRONOS_TEMPERATURE (sampling T)
    top_p: float = 0.9                    # CLIO_KRONOS_TOP_P (nucleus sampling)

    # ── Pinned model/tokenizer (see vendor/VENDOR_PIN.md) ───────────────
    model_id: str = DEFAULT_MODEL_ID                       # CLIO_KRONOS_MODEL
    model_revision: str = DEFAULT_MODEL_REVISION           # CLIO_KRONOS_MODEL_REVISION
    tokenizer_id: str = DEFAULT_TOKENIZER_ID               # CLIO_KRONOS_TOKENIZER
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION   # CLIO_KRONOS_TOKENIZER_REVISION
    vendor_commit: str = UPSTREAM_VENDOR_COMMIT            # CLIO_KRONOS_VENDOR_COMMIT (loggable provenance tag)
    # KRONOS_HF_LOCAL_ONLY-equivalent: never a supply-chain-open default.
    # False only for `--download` (an explicit, one-time deploy step) or an
    # operator-shell cache refresh -- never set true in .env (mirrors
    # Hydrinia's kronos_vendor/VENDOR_PIN.md §3 "operator-SHELL override only").
    hf_local_files_only: bool = True                       # CLIO_KRONOS_HF_LOCAL_ONLY

    # ── Alpaca market data (shared paper keys; same fallback DPOF/flowcollector use) ──
    alpaca_key_id: str = ""                                # ALPACA_PAPER_KEY_ID
    alpaca_secret_key: str = ""                             # ALPACA_PAPER_SECRET_KEY
    stock_feed: str = "iex"                                 # CLIO_KRONOS_STOCK_FEED / ALPACA_STOCK_FEED, "iex" or "sip"

    # ── History (scorer's append-only log; kronos_forecast.json/kronos_scores.json
    # paths are NOT duplicated here -- state.py's forecast_path()/scores_path()
    # are the single source of truth for those two, per its own module docstring) ──
    history_file: str = DEFAULT_HISTORY_FILE                # CLIO_KRONOS_HISTORY_FILE
    history_max_lines: int = 20_000                          # CLIO_KRONOS_HISTORY_MAX_LINES

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.symbols:
            issues.append("symbols is empty")
        if not (2 <= self.sample_paths <= 64):
            issues.append(f"sample_paths out of range: {self.sample_paths}")
        if not (64 <= self.lookback_bars <= 512):
            issues.append(f"lookback_bars out of range: {self.lookback_bars}")
        if not (1 <= self.torch_threads <= 8):
            issues.append(f"torch_threads out of range: {self.torch_threads}")
        if self.run_timeout_seconds <= 0:
            issues.append(f"run_timeout_seconds must be positive: {self.run_timeout_seconds}")
        if not (0.0 < self.temperature <= 5.0):
            issues.append(f"temperature out of range: {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            issues.append(f"top_p out of range: {self.top_p}")
        if self.stock_feed not in ("iex", "sip"):
            issues.append(f"unsupported stock_feed (must be 'iex' or 'sip'): {self.stock_feed!r}")
        if self.history_max_lines < 1:
            issues.append(f"history_max_lines must be >= 1: {self.history_max_lines}")
        if not self.model_id or not self.model_revision:
            issues.append("model_id/model_revision must be set")
        if not self.tokenizer_id or not self.tokenizer_revision:
            issues.append("tokenizer_id/tokenizer_revision must be set")
        return issues


def load_config() -> KronosConfig:
    """Build config from the current environment. Re-reads env every call
    (matches agent.flowcollector.config.load_config's contract)."""
    d = KronosConfig()  # defaults -- single source for fallbacks
    return KronosConfig(
        symbols=_env_list("CLIO_KRONOS_SYMBOLS", d.symbols),
        sample_paths=_clamp_int(_env_int("CLIO_KRONOS_SAMPLE_PATHS", d.sample_paths), 2, 64, "sample_paths"),
        lookback_bars=_clamp_int(_env_int("CLIO_KRONOS_LOOKBACK_BARS", d.lookback_bars), 64, 512, "lookback_bars"),
        torch_threads=_clamp_int(_env_int("CLIO_KRONOS_TORCH_THREADS", d.torch_threads), 1, 8, "torch_threads"),
        run_timeout_seconds=_env_float("CLIO_KRONOS_RUN_TIMEOUT", d.run_timeout_seconds),
        temperature=_env_float("CLIO_KRONOS_TEMPERATURE", d.temperature),
        top_p=_env_float("CLIO_KRONOS_TOP_P", d.top_p),
        model_id=_env_str("CLIO_KRONOS_MODEL", d.model_id),
        model_revision=_env_str("CLIO_KRONOS_MODEL_REVISION", d.model_revision),
        tokenizer_id=_env_str("CLIO_KRONOS_TOKENIZER", d.tokenizer_id),
        tokenizer_revision=_env_str("CLIO_KRONOS_TOKENIZER_REVISION", d.tokenizer_revision),
        vendor_commit=_env_str("CLIO_KRONOS_VENDOR_COMMIT", d.vendor_commit),
        hf_local_files_only=_env_str("CLIO_KRONOS_HF_LOCAL_ONLY", "1" if d.hf_local_files_only else "0") not in ("0", "false", "no", "off"),
        alpaca_key_id=_env_str("ALPACA_PAPER_KEY_ID", d.alpaca_key_id),
        alpaca_secret_key=_env_str("ALPACA_PAPER_SECRET_KEY", d.alpaca_secret_key),
        stock_feed=_env_str("CLIO_KRONOS_STOCK_FEED", _env_str("ALPACA_STOCK_FEED", d.stock_feed)).lower(),
        history_file=_env_str("CLIO_KRONOS_HISTORY_FILE", d.history_file),
        history_max_lines=_env_int("CLIO_KRONOS_HISTORY_MAX_LINES", d.history_max_lines),
    )


def config_summary(cfg: KronosConfig) -> dict:
    """Loggable view of the config with secrets redacted (never log a key value)."""
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if f.name in ("alpaca_key_id", "alpaca_secret_key"):
            v = "set" if v else "unset"
        out[f.name] = v
    return out
