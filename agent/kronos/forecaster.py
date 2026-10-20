"""
CLIO Kronos — forecast metrics (pure, torch-free) + the torch-backed
predictor wrapper (all torch/vendor-model imports LAZY, inside methods).

Two halves, deliberately split per the design frame (docs/design/
clio_kronos.md §3):

1. Pure metric code (module level, safe to import and unit-test on a venv
   with no `torch` installed): horizon truncation at the regular-session
   close, the deterministic per-run sampling seed, and the per-horizon
   statistics (p_up, mean/median/q10/q90 return %, expected |move| %) from
   a set of sampled future-close paths. `numpy`/`pandas` ARE imported at
   module level (they are ordinary runtime dependencies of this repo
   already — CLAUDE.md's own `requirements.txt` pins numpy; see this
   module's own `predict_batch` for the ONE place `torch` is imported,
   lazily). `tests/test_kronos_isolation.py` proves `import agent.kronos`
   never pulls in `torch`.

2. `KronosPredictor`, a thin wrapper around the vendored model
   (`agent.kronos.vendor.model`) and `torch`. Built once per service
   process; `torch.set_num_threads()` is applied exactly once, at the
   first successful load (a process-global torch setting — later
   `CLIO_KRONOS_TORCH_THREADS` env edits need a service restart to take
   effect, same caveat Hydrinia's `kronos_vendor/VENDOR_PIN.md` documents
   for its own `KRONOS_TORCH_THREADS`). `predict_batch` runs ONE batched
   `generate()` call across every (symbol, sample path) pair — the input
   bars are replicated `sample_paths` times per symbol (`sample_count=1`
   each), never handed to the vendor model's own `sample_count` averaging,
   which would discard exactly the per-path spread this service needs
   (mirrors Hydrinia's `kronos_forecast_agent.compute_forecast`'s
   documented reason for the same replication trick).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from agent.kronos.bars import BAR_MINUTES, Bar
from agent.kronos.config import KronosConfig

logger = logging.getLogger("clio.kronos.forecaster")

# Horizons offered, in minutes -- matches agent.kronos.state.HORIZONS
# ("15","30","60") exactly; the string labels used in the forecast file are
# derived from these, never hand-duplicated elsewhere.
HORIZONS_MINUTES: tuple[int, ...] = (15, 30, 60)


def seed_from_anchor(anchor_epoch: float) -> int:
    """PURE, deterministic: the same anchor bar always seeds the same way,
    so a retried or re-run forecast for the same bar samples the same
    paths (reproducibility, not cryptographic determinism — sampling still
    varies ACROSS anchors, which is the point). Masked to a non-negative
    31-bit value, well inside torch's manual_seed range."""
    return int(anchor_epoch) & 0x7FFFFFFF


def resolve_pred_plan(anchor_et: datetime, close_et: datetime) -> tuple[int, list[tuple[str, int, int]]]:
    """PURE. Given the anchor bar's ET close time and the regular session's
    ET close on that day, returns `(pred_len_bars, horizons)` where
    `horizons` is the subset of `HORIZONS_MINUTES` whose target time
    (`anchor_et + minutes`) does not cross `close_et` — as
    `(label, minutes, bar_count)` triples, `label` matching
    `agent.kronos.state.HORIZONS`. `pred_len_bars` is the max `bar_count`
    needed to cover every included horizon (0, with an empty `horizons`
    list, when even the shortest horizon would cross the close — the
    service then skips the model call for this symbol/run entirely rather
    than ever extrapolating overnight, CLAUDE.md Intel v3 Gotchas)."""
    horizons: list[tuple[str, int, int]] = []
    for minutes in HORIZONS_MINUTES:
        target_et = anchor_et + timedelta(minutes=minutes)
        if target_et <= close_et:
            horizons.append((str(minutes), minutes, minutes // BAR_MINUTES))
    pred_len = max((bars for _, _, bars in horizons), default=0)
    return pred_len, horizons


def p_up_from_returns(returns_pct: list[float]) -> float:
    """PURE. Share of sampled paths whose return is positive; a return of
    exactly 0.0 counts as half-up (a flat path is direction-neutral, not
    bearish — same convention Hydrinia's `_derive_vote_from_samples` uses)."""
    n = len(returns_pct)
    up = sum(1 for r in returns_pct if r > 0.0)
    flat = sum(1 for r in returns_pct if r == 0.0)
    return (up + 0.5 * flat) / n


@dataclass(frozen=True)
class HorizonMetrics:
    minutes: int
    p_up: float
    mean_return_pct: float
    median_return_pct: float
    q10_return_pct: float
    q90_return_pct: float
    expected_abs_move_pct: float

    def as_dict(self) -> dict:
        return {
            "minutes": self.minutes,
            "p_up": self.p_up,
            "mean_return_pct": self.mean_return_pct,
            "median_return_pct": self.median_return_pct,
            "q10_return_pct": self.q10_return_pct,
            "q90_return_pct": self.q90_return_pct,
            "expected_abs_move_pct": self.expected_abs_move_pct,
        }


def compute_horizon_metrics(minutes: int, returns_pct: list[float]) -> Optional[HorizonMetrics]:
    """PURE. `returns_pct`: one sampled-path return (percent of anchor
    close) per path, e.g. 0.12 for +0.12%. Non-finite entries are dropped
    first (a single inf/nan sample must not poison the whole horizon —
    CLAUDE.md anti-pattern #1). Returns None when fewer than 2 finite
    samples remain (a single surviving sample can't support a spread
    statistic; the caller treats None as "this horizon has no result this
    run", never a fabricated 0.0 — CLAUDE.md "no fake zeros")."""
    finite = [r for r in returns_pct if isinstance(r, (int, float)) and math.isfinite(r)]
    if len(finite) < 2:
        return None
    arr = np.asarray(finite, dtype=float)
    return HorizonMetrics(
        minutes=minutes,
        p_up=round(p_up_from_returns(finite), 4),
        mean_return_pct=round(float(np.mean(arr)), 4),
        median_return_pct=round(float(np.median(arr)), 4),
        q10_return_pct=round(float(np.percentile(arr, 10)), 4),
        q90_return_pct=round(float(np.percentile(arr, 90)), 4),
        expected_abs_move_pct=round(float(np.mean(np.abs(arr))), 4),
    )


def build_forecast_horizons(
    anchor_close: float, anchor_et: datetime, close_et: datetime, path_closes: list[list[float]],
) -> dict[str, dict]:
    """PURE. `path_closes`: one list per sampled path, each holding the
    predicted close price for bars 1..pred_len (ascending). Returns
    `{"15": {...}, "30": {...}, ...}` for every horizon `resolve_pred_plan`
    says fits before `close_et`, each value a `HorizonMetrics.as_dict()` (or
    omitted — never a zero-filled placeholder — when
    `compute_horizon_metrics` returns None for that horizon, e.g. a
    predictor that returned fewer than 2 usable paths)."""
    _, horizons = resolve_pred_plan(anchor_et, close_et)
    out: dict[str, dict] = {}
    if anchor_close is None or not math.isfinite(anchor_close) or anchor_close <= 0:
        return out
    for label, minutes, bar_count in horizons:
        returns_pct: list[float] = []
        for path in path_closes:
            if len(path) < bar_count:
                continue
            close_at_h = path[bar_count - 1]
            if close_at_h is None or not isinstance(close_at_h, (int, float)) or not math.isfinite(close_at_h):
                continue
            returns_pct.append((close_at_h / anchor_close - 1.0) * 100.0)
        metrics = compute_horizon_metrics(minutes, returns_pct)
        if metrics is not None:
            out[label] = metrics.as_dict()
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Torch-backed predictor (lazy imports only — never at module import time)
# ═══════════════════════════════════════════════════════════════════════════


class ModelUnavailable(Exception):
    """torch, the vendored model package, or the pinned HF weights could
    not be loaded — the caller maps this to a per-symbol "error" status,
    never a crash of the whole run."""


class KronosPredictor:
    """Wraps the vendored Kronos model. Built once per service process
    (`KronosConfig` is immutable, so one instance's lifetime == one
    process's). `_ensure_loaded` is idempotent and cheap to call every run
    (a no-op once loaded)."""

    def __init__(self, cfg: KronosConfig) -> None:
        self._cfg = cfg
        self._predictor = None  # lazy vendor KronosPredictor instance
        self._threads_set = False

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch
        except ImportError as e:
            raise ModelUnavailable(f"torch not installed: {e}") from e

        if not self._threads_set:
            # Process-global torch setting -- binds at FIRST successful load
            # only (mirrors kronos_vendor/VENDOR_PIN.md's documented
            # caveat: a later CLIO_KRONOS_TORCH_THREADS env edit needs a
            # service restart).
            torch.set_num_threads(self._cfg.torch_threads)
            self._threads_set = True

        try:
            from agent.kronos.vendor.model import Kronos, KronosPredictor as _VendorPredictor, KronosTokenizer
        except ImportError as e:
            raise ModelUnavailable(f"vendored kronos model package unavailable: {e}") from e

        try:
            tokenizer = KronosTokenizer.from_pretrained(
                self._cfg.tokenizer_id, revision=self._cfg.tokenizer_revision,
                local_files_only=self._cfg.hf_local_files_only,
            )
            model = Kronos.from_pretrained(
                self._cfg.model_id, revision=self._cfg.model_revision,
                local_files_only=self._cfg.hf_local_files_only,
            )
            predictor = _VendorPredictor(model, tokenizer, device="cpu", max_context=512)
        except Exception as e:
            raise ModelUnavailable(f"kronos model/tokenizer load failed: {type(e).__name__}: {e}") from e

        self._predictor = predictor
        logger.info(f"kronos: loaded {self._cfg.model_id}@{self._cfg.model_revision[:8]} on cpu")

    def download_weights(self) -> None:
        """`python -m agent.kronos --download`: fetch the pinned
        model+tokenizer revisions into the local HF cache once (needs
        network; an explicit, operator-run deploy step — never called from
        the run loop)."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ModelUnavailable(f"huggingface_hub not installed: {e}") from e
        for repo, rev in (
            (self._cfg.model_id, self._cfg.model_revision),
            (self._cfg.tokenizer_id, self._cfg.tokenizer_revision),
        ):
            path = snapshot_download(repo, revision=rev)
            logger.info(f"kronos --download: {repo}@{rev[:8]} -> {path}")

    def predict_batch(
        self, bars_by_symbol: dict[str, list[Bar]], pred_len: int, seed: int,
    ) -> dict[str, list[list[float]]]:
        """One batched `generate()` call across every (symbol, sample path)
        pair. Returns `{symbol: [[close_bar1, ..., close_bar_pred_len], ...]}`
        — `cfg.sample_paths` inner lists per symbol, each of length
        `pred_len`. A symbol with an empty bar list is simply absent from
        the result (never a fabricated entry); `pred_len <= 0` or an empty
        `bars_by_symbol` short-circuits to `{}` without loading the model
        at all."""
        symbols = [s for s, bars in bars_by_symbol.items() if bars]
        if not symbols or pred_len <= 0:
            return {}

        self._ensure_loaded()
        import pandas as pd
        import torch

        torch.manual_seed(seed)

        replicas = self._cfg.sample_paths
        df_list: list = []
        x_ts_list: list = []
        y_ts_list: list = []
        for sym in symbols:
            bars = bars_by_symbol[sym]
            df = pd.DataFrame({
                "open": [b.o for b in bars], "high": [b.h for b in bars], "low": [b.l for b in bars],
                "close": [b.c for b in bars], "volume": [b.v for b in bars], "amount": [b.amount for b in bars],
            })
            # ET WALL-CLOCK, tz-naive after conversion: calc_time_stamps
            # (vendor/model/kronos.py) reads .dt.hour/.dt.minute/etc -- these
            # must reflect the exchange's local time, not UTC.
            x_ts = pd.to_datetime([b.t_open for b in bars], unit="s", utc=True) \
                .tz_convert("America/New_York").tz_localize(None)
            last_ts = x_ts[-1]
            y_ts = pd.date_range(start=last_ts + pd.Timedelta(minutes=BAR_MINUTES), periods=pred_len, freq=f"{BAR_MINUTES}min")
            for _ in range(replicas):
                df_list.append(df)
                x_ts_list.append(pd.Series(x_ts))
                y_ts_list.append(pd.Series(y_ts))

        pred_dfs = self._predictor.predict_batch(
            df_list, x_ts_list, y_ts_list, pred_len=pred_len,
            T=self._cfg.temperature, top_k=0, top_p=self._cfg.top_p,
            sample_count=1, verbose=False,
        )

        out: dict[str, list[list[float]]] = {}
        idx = 0
        for sym in symbols:
            paths: list[list[float]] = []
            for _ in range(replicas):
                pdf = pred_dfs[idx]
                paths.append([float(v) for v in pdf["close"].tolist()])
                idx += 1
            out[sym] = paths
        return out
