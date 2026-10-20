"""Feed-side manager of the DPOF dynamic pillar weights (docs/design/clio_dpof_dynamic_weights.md).

Modes (CLIO_DPOF_DYNAMIC_WEIGHTS; invalid values mean off):
- off: every public method returns at once; no state, no files, no objects created.
- shadow: liveness and weights are computed; the dynamic composite is logged beside
  the static one. The snapshot is never modified, so decisions use static weights.
- live: the snapshot gets u.pillar_weights, so decisions use dynamic weights.

Fail-static: any exception on the dynamic path leaves the underlying on static
weights (live mode clears u.pillar_weights) and logs one deduplicated WARNING per
hour. Public methods never raise.

Conservation: dead pillars stay in the weight target with aliveness 0, so their
weight is redistributed to the living pillars. In live mode the dead pillar's
weight is 0.0, which masks it inside compute_composite even when the data layer
has not removed it from the snapshot.

I/O: state and shadow writes run on the default executor (or inline when sync_io
is set or no loop is running). finish_snapshot is synchronous and never awaits,
so a cycle-timeout CancelledError cannot skip later steps.

This class owns no network and no engine access.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from datetime import date, datetime, time as dtime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from agent.dpof.config import dynamic_weights_mode
from agent.dpof.models import PillarReading, UnderlyingSnapshot
from agent.dpof.pillar_weights import (
    DIRECTIONAL,
    LIVENESS_PILLARS,
    SCHEMA_VERSION,
    LivenessTracker,
    QualityStats,
    WeightParams,
    compute_weights,
)
from agent.dpof.pillar_weights_io import ShadowLog, StateStore
from agent.dpof.signals import KRONOS_PILLAR, compute_composite, composite_flat_variants

logger = logging.getLogger("clio.dpof.dynamic_weights")
ET = ZoneInfo("America/New_York")

_WARN_INTERVAL_S = 3600.0
_SAVE_INTERVAL_S = 60.0
_STATE_MAX_AGE_S = 5 * 86400.0
_RESTORE_RING_FRESH_S = 900.0
_PRICE_MAX_AGE_S = 180.0
_HORIZON_S = 1800.0
_GIVE_UP_S = 3600.0
_BLOCK_START = dtime(9, 45)
_MAX_BLOCK_INDEX = 10
_WEIGHT_LOG_DELTA = 0.02
_LOAD_TIMEOUT_S = 3.0  # the first snapshot never waits longer than this for the state file (fail-static, like every other file gate)
_MAX_INFLIGHT_IO = 4  # writes allowed in the SHARED default executor at once: a stalled disk must not exhaust it
_SAT_THRESHOLD = 0.999  # shadow evidence (R-37): |live premium| at or above this counts as saturated
_GEX_WINDOW = 200  # shadow evidence (R-37): GEX observations kept per underlying (bounded memory)
_GEX_MIN_OBS = 20  # gex_cont is None until this many observations exist


def _finite(v: Any) -> float | None:
    """v as a finite float, or None for bool, non-numbers, overflow, NaN or inf."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except (OverflowError, ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def _round4(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def _default_session(now_epoch: float) -> tuple[str, bool]:
    """(ET session_day, in_session) for an epoch; RTH is 09:30 to 16:00 (13:00 on early-close days)."""
    now = datetime.fromtimestamp(float(now_epoch), ET)
    day: date = now.date()
    holidays: frozenset = frozenset()
    early: frozenset = frozenset()
    try:
        from agent.integrations import market_events as _me

        holidays = frozenset(_me.MARKET_HOLIDAYS)
        early = frozenset(_me.MARKET_EARLY_CLOSE)
    except Exception as e:  # fall back to weekday + 09:30-16:00
        logger.warning("DPOF dynamic weights: market calendar unavailable (%s); weekday hours only", e)
    if day.weekday() >= 5 or day in holidays:
        return day.isoformat(), False
    close = dtime(13, 0) if day in early else dtime(16, 0)
    return day.isoformat(), dtime(9, 30) <= now.time() < close


def _is_uw(r: PillarReading) -> bool:
    return isinstance(r.source, str) and r.source.startswith("uw")


def _price_fresh(u: UnderlyingSnapshot) -> bool:
    lp = _finite(u.last_price)
    age = _finite(u.price_age_seconds)
    return lp is not None and lp > 0.0 and age is not None and age <= _PRICE_MAX_AGE_S


def _block(now_et: datetime, block_minutes: int) -> tuple[int, datetime] | None:
    """(block_index, block_start) for a 30-minute block from 09:45 ET; None outside blocks 0..10."""
    start = datetime.combine(now_et.date(), _BLOCK_START, tzinfo=ET)
    mins = (now_et - start).total_seconds() / 60.0
    if mins < 0:
        return None
    idx = int(mins // block_minutes)
    if idx > _MAX_BLOCK_INDEX:
        return None
    return idx, start + timedelta(minutes=idx * block_minutes)


def _valid_pending(rec: Any) -> bool:
    if not isinstance(rec, Mapping):
        return False
    idx = rec.get("idx")
    if isinstance(idx, bool) or not isinstance(idx, int) or not (0 <= idx <= _MAX_BLOCK_INDEX):
        return False
    return all(_finite(rec.get(k)) is not None for k in ("t0", "x", "z", "p0")) and float(rec["p0"]) > 0.0


class DynamicWeightsManager:
    """Owns liveness (L1), quality diagnostics (L2) and weight computation (L3) for DPOF pillars."""

    def __init__(
        self,
        cfg: Any,
        data_dir: str,
        *,
        session_fn: Callable[[float], tuple[str, bool]] | None = None,
        sync_io: bool = False,
    ) -> None:
        self._cfg = cfg
        self._mode = dynamic_weights_mode(cfg)
        self._session_fn = session_fn or _default_session
        self._sync_io = bool(sync_io)
        self._params = WeightParams(
            prior={
                "nofa": cfg.w_nofa,
                "premium": cfg.w_premium,
                "darkpool": cfg.w_darkpool,
                "kronos": getattr(cfg, "w_kronos", 0.0),
            },
            ring_size=int(cfg.dw_ring_size),
            dead_eps=float(cfg.dw_dead_eps),
            revive_eps=2 * float(cfg.dw_dead_eps),
            quality_apply=bool(cfg.dw_quality_apply),
        )
        self._kronos_live = getattr(cfg, "kronos_pillar", "off") == "live"
        self._liveness = LivenessTracker(self._params)
        self._quality: dict[str, QualityStats] = {}
        self._pending: dict[str, dict] = {}
        self._sampled: set[tuple] = set()
        self._last_weights: dict[str, dict] = {}
        self._store: StateStore | None = None
        self._shadow: ShadowLog | None = None
        if self._mode != "off":
            self._store = StateStore(data_dir)
            self._shadow = ShadowLog(data_dir)
        self._loaded = False
        self._load_task: asyncio.Future | None = None
        self._load_loop: Any = None
        self._touched = False
        self._last_session_day: str | None = None
        self._last_save_ts = 0.0
        self._dirty = False
        self._inflight: set = set()
        self._io_cv = threading.Condition()
        self._io_pending = 0
        self._warned: dict[str, float] = {}
        # Shadow-only evidence (R-37). Memory-only, never persisted, never read by any decision path.
        self._premium_alt: dict[str, dict] = {}
        self._premium_sat_run: dict[str, int] = {}
        self._gex_obs: dict[str, deque] = {}

    @property
    def mode(self) -> str:
        """Normalised mode: off, shadow or live."""
        return self._mode

    # ------------------------------------------------------------------ public API
    def observe_uw(
        self,
        underlying: str,
        pillar: str,
        value: Any,
        now_epoch: float,
        source: str = "uw",
    ) -> None:
        """Feed one refresh value of a liveness pillar (premium/darkpool) into the L1 tracker."""
        if self._mode == "off" or pillar not in LIVENESS_PILLARS:
            return
        try:
            key = f"{underlying}|{pillar}|{source}"
            session_day, in_session = self._session_fn(now_epoch)
            was_dead = self._liveness.is_dead(key)
            self._liveness.observe(key, value, now_epoch, session_day, in_session)
            self._touched = True
            self._dirty = True
            now_dead = self._liveness.is_dead(key)
            if now_dead and not was_dead:
                self._warn(
                    f"dead:{key}",
                    f"DPOF dynamic weights: {key} looks DEAD (last {self._params.ring_size} refreshes "
                    f"|v|<eps or identical); weight redistributed (mode={self._mode})",
                )
            elif was_dead and not now_dead:
                logger.info("DPOF dynamic weights: %s revived (mode=%s); weight ramps back in", key, self._mode)
        except Exception as e:
            self._warn("observe", f"DPOF dynamic weights: observe_uw failed ({type(e).__name__}: {e})")

    def observe_premium_alt(
        self,
        underlying: str,
        live_value: Any,
        alt: Mapping,
        now_epoch: float,
    ) -> None:
        """Shadow (R-37): keep the latest premium alternatives beside the live refresh value. Never raises."""
        if self._mode == "off":
            return
        try:
            rec: dict = {"live": _round4(_finite(live_value))}
            for k in ("w60", "ewma", "day"):
                rec[k] = _round4(_finite(alt.get(k)))
            self._premium_alt[underlying] = rec
        except Exception as e:
            self._warn("premium_alt", f"DPOF dynamic weights: observe_premium_alt failed ({type(e).__name__}: {e})")

    def observe_gex(self, underlying: str, net_gex: Any, now_epoch: float) -> None:
        """Shadow (R-37): one raw net_gex observation (bounded window). Never raises."""
        if self._mode == "off":
            return
        try:
            v = _finite(net_gex)
            if v is None:
                return
            dq = self._gex_obs.get(underlying)
            if dq is None:
                dq = deque(maxlen=_GEX_WINDOW)
                self._gex_obs[underlying] = dq
            dq.append(v)
        except Exception as e:
            self._warn("gex_obs", f"DPOF dynamic weights: observe_gex failed ({type(e).__name__}: {e})")

    def is_masked(self, underlying: str, pillar: str, now_epoch: float) -> bool:
        """True in live mode when a UW liveness pillar is dead (the data layer then treats it as missing)."""
        if self._mode != "live" or pillar not in LIVENESS_PILLARS:
            return False
        try:
            return self._liveness.is_dead(f"{underlying}|{pillar}|uw")
        except Exception:
            return False

    async def ensure_loaded(self) -> None:
        """Restore persisted state once (idempotent), waiting at most _LOAD_TIMEOUT_S.

        A slow or hung disk never stalls the snapshot: the cycle proceeds with empty state and the still-running load
        is picked up (or, once observation has started, discarded) on a later call. A cancelled wait leaves the load
        running and is retried on the next call."""
        if self._mode == "off" or self._loaded:
            return
        if self._sync_io:
            self.load_sync(time.time())
            return
        loop = asyncio.get_running_loop()
        task = self._load_task
        if task is None or self._load_loop is not loop:
            task = loop.create_task(self._load_state())
            self._load_task, self._load_loop = task, loop
        done, _ = await asyncio.wait({task}, timeout=_LOAD_TIMEOUT_S)
        if not done:
            self._warn("load_slow", f"DPOF dynamic weights: state load still running after {_LOAD_TIMEOUT_S:.0f}s; continuing with empty state")
            return
        if self._loaded:  # a concurrent caller already finished the restore
            return
        self._load_task = None
        self._finish_load(None if task.cancelled() else task.result(), time.time())

    async def _load_state(self) -> dict | None:
        try:
            return await self._store.load_async()
        except Exception as e:
            logger.warning("DPOF dynamic weights: state load failed (%s); starting empty", e)
            return None

    def load_sync(self, now_epoch: float) -> None:
        """Synchronous equivalent of ensure_loaded (sync_io managers and tests)."""
        if self._mode == "off" or self._loaded:
            return
        try:
            state = self._store.load()
        except Exception as e:
            logger.warning("DPOF dynamic weights: state load failed (%s); starting empty", e)
            state = None
        self._finish_load(state, now_epoch)

    def finish_snapshot(self, now_epoch: float, underlyings: Mapping[str, UnderlyingSnapshot]) -> None:
        """Synchronous, CPU-only per-cycle step: masks, weights, shadow record, quality sampling, saves."""
        if self._mode == "off":
            return
        try:
            self._touched = True
            self._finish(now_epoch, underlyings)
        except Exception as e:
            self._warn("finish", f"DPOF dynamic weights: finish_snapshot failed ({type(e).__name__}: {e}); static weights")

    def drain(self, timeout: float = 2.0) -> None:
        """Wait (blocking, bounded) for in-flight writes. Test and shutdown helper; never raises."""
        try:
            deadline = time.monotonic() + max(0.0, float(timeout))
            with self._io_cv:
                while self._io_pending > 0:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self._io_cv.wait(left)
        except Exception as e:
            logger.debug("DPOF dynamic weights: drain failed: %s", e)

    # ------------------------------------------------------------------ restore
    def _finish_load(self, state: Any, now_epoch: float) -> None:
        try:
            if self._touched:
                logger.warning("DPOF dynamic weights: state loaded after first observation; skipping restore")
            else:
                self._restore(state, now_epoch)
        except Exception as e:
            logger.warning("DPOF dynamic weights: restore failed (%s); starting empty", e)
        self._loaded = True

    def _restore(self, state: Any, now_epoch: float) -> None:
        if not isinstance(state, Mapping):
            return
        ver = state.get("schema_version")
        if isinstance(ver, bool) or ver != SCHEMA_VERSION:
            logger.info("DPOF dynamic weights: state schema_version %r ignored", ver)
            return
        written = _finite(state.get("written_at"))
        if written is None or now_epoch - written > _STATE_MAX_AGE_S:
            logger.info("DPOF dynamic weights: state too old or unstamped; starting empty")
            return
        today, _ = self._session_fn(now_epoch)
        saved_day = state.get("session_day")
        ring_ok = saved_day == today and (now_epoch - written) <= _RESTORE_RING_FRESH_S
        tracker = LivenessTracker(self._params)
        liv = state.get("liveness")
        if isinstance(liv, Mapping) and isinstance(liv.get("keys"), Mapping):
            keys: dict[str, Any] = {}
            for key, rec in liv["keys"].items():
                if isinstance(rec, Mapping) and not ring_ok:
                    rec = dict(rec)
                    rec["ring"] = []  # a stale ring is never reused; a dead flag still is
                if isinstance(rec, Mapping):
                    flip = _finite(rec.get("last_flip_ts"))
                    if flip is not None and flip > now_epoch:  # clock skew or a hand-edited file must not freeze the dwell forever
                        rec = dict(rec)
                        rec["last_flip_ts"] = None
                keys[key] = rec
            tracker = LivenessTracker.from_dict({"schema_version": liv.get("schema_version"), "keys": keys}, self._params)
        quality: dict[str, QualityStats] = {}
        raw_q = state.get("quality")
        if isinstance(raw_q, Mapping):
            for key, rec in raw_q.items():
                if isinstance(key, str):
                    quality[key] = QualityStats.from_dict(rec)
        pending: dict[str, dict] = {}
        raw_p = state.get("pending")
        if saved_day == today and isinstance(raw_p, Mapping):
            for key, rec in raw_p.items():
                if isinstance(key, str) and _valid_pending(rec):
                    pending[key] = dict(rec)
        last_w: dict[str, dict] = {}
        raw_lw = state.get("last_weights")
        if isinstance(raw_lw, Mapping):
            for sym, w in raw_lw.items():
                if isinstance(sym, str) and isinstance(w, Mapping):
                    last_w[sym] = {p: float(v) for p, v in w.items() if isinstance(p, str) and _finite(v) is not None}
        self._liveness = tracker
        self._quality = quality
        self._pending = pending
        self._last_weights = last_w
        self._last_session_day = saved_day if isinstance(saved_day, str) else None

    # ------------------------------------------------------------------ per-cycle
    def _finish(self, now_epoch: float, underlyings: Mapping[str, UnderlyingSnapshot]) -> None:
        session_day, in_session = self._session_fn(now_epoch)
        now_et = datetime.fromtimestamp(float(now_epoch), ET)
        force_save = False
        if in_session:  # roll on the first IN-SESSION snapshot of a new trading day, never on a weekend/holiday flip
            if self._last_session_day is not None and session_day != self._last_session_day:
                self._roll(session_day)
                force_save = True
            self._last_session_day = session_day
        records: list[dict] = []
        for sym, u in underlyings.items():
            try:
                records.append(self._process(sym, u, now_epoch, now_et, session_day, in_session))
            except Exception as e:
                if self._mode == "live":
                    u.pillar_weights = None
                self._warn(f"proc:{sym}", f"DPOF dynamic weights: {sym} failed ({type(e).__name__}: {e}); static weights")
        if records and self._shadow is not None:
            self._submit(self._shadow.append_cycle, records)
        if self._store is not None and (force_save or (self._dirty and now_epoch - self._last_save_ts >= _SAVE_INTERVAL_S)):
            if self._submit(self._store.save, self._state_doc(session_day)):  # a dropped save stays dirty and is retried
                self._last_save_ts = now_epoch
                self._dirty = False

    def _roll(self, session_day: str) -> None:
        for q in self._quality.values():
            q.roll_session(session_day, self._params)
        self._pending.clear()
        self._sampled.clear()
        self._dirty = True

    def _readings(self, u: UnderlyingSnapshot) -> dict[str, PillarReading]:
        out: dict[str, PillarReading] = {}
        for p in DIRECTIONAL:
            r = u.pillars.get(p)
            if r is None or not r.available or r.value is None:
                continue
            if p == KRONOS_PILLAR and not self._kronos_live:
                continue
            out[p] = r
        return out

    def _process(
        self,
        sym: str,
        u: UnderlyingSnapshot,
        now_epoch: float,
        now_et: datetime,
        session_day: str,
        in_session: bool,
    ) -> dict:
        params = self._params
        readings = self._readings(u)
        dead: set[str] = set()
        alive: dict[str, float] = {}
        for p, r in readings.items():
            if p in LIVENESS_PILLARS and _is_uw(r):
                key = f"{sym}|{p}|uw"
                if self._liveness.is_dead(key):
                    dead.add(p)
                alive[p] = self._liveness.aliveness(key)
            else:
                alive[p] = 1.0
        for p in LIVENESS_PILLARS:
            # A dead UW pillar the data layer already removed (live masking) and that nothing replaced still
            # owns its prior in the conservation target; otherwise its weight would silently not be redistributed.
            if p not in readings and self._liveness.is_dead(f"{sym}|{p}|uw"):
                alive[p] = 0.0
                dead.add(p)
        mult: dict[str, float] = {}
        for p in readings:
            stats = self._quality.get(f"{sym}|{p}")
            mult[p] = stats.multiplier(params) if stats is not None else 1.0
        weights = compute_weights(alive, mult, params)
        static_c = compute_composite(dataclasses.replace(u, pillar_weights=None), self._cfg)[0]
        masked = dataclasses.replace(
            u,
            pillars={k: v for k, v in u.pillars.items() if k not in dead},
            pillar_weights=None,
        )
        dyn_c = compute_composite(masked, self._cfg, weights=weights or None)[0]
        # Shadow evidence for CLIO_DPOF_DARKPOOL_FLAT_UNAVAILABLE: static-weight composites
        # with and without the flat legacy darkpool, whatever the flag says (never decisional).
        with_flat_c, without_flat_c = composite_flat_variants(u, self._cfg)
        rounded = {p: round(w, 6) for p, w in weights.items()}
        for p, r in readings.items():
            self._sample(sym, p, r.value, u, now_epoch, now_et, session_day, in_session)
        if self._mode == "live":
            u.pillar_weights = dict(weights) if weights else None
        self._log_weight_change(sym, rounded)
        m_out: dict[str, float] = {}
        n_out: dict[str, float] = {}
        for p in readings:
            stats = self._quality.get(f"{sym}|{p}")
            m_out[p] = round(stats.diagnostic_multiplier(params), 6) if stats is not None else 1.0
            n_out[p] = round(stats.n_eff(), 3) if stats is not None else 0.0
        return {
            "ts": now_et.isoformat(),
            "u": sym,
            "mode": self._mode,
            "static": static_c,
            "dyn": dyn_c,
            "composite_with_flat": with_flat_c,
            "composite_without_flat": without_flat_c,
            "w": rounded,
            "alive": {p: round(a, 6) for p, a in alive.items()},
            "m": m_out,
            "ma": {p: round(mult[p], 6) for p in readings},
            "n_eff": n_out,
            "avail": sorted(readings),
            "dead": sorted(dead),
            **self._alt_fields(sym, u),
        }

    def _alt_fields(self, sym: str, u: UnderlyingSnapshot) -> dict:
        """Shadow-only row keys (R-37): premium_alt, premium_sat_run, gex_cont. Read-only with respect to the
        snapshot and the live weights; every step is guarded separately, so none can change the live result."""
        out: dict = {"premium_alt": None, "premium_sat_run": 0, "gex_cont": None}
        try:
            pr = u.pillars.get("premium")
            pv = _finite(pr.value) if pr is not None and pr.available else None
            run = self._premium_sat_run.get(sym, 0) + 1 if (pv is not None and abs(pv) >= _SAT_THRESHOLD) else 0
            self._premium_sat_run[sym] = run
            out["premium_sat_run"] = run
        except Exception as e:
            self._warn("sat_run", f"DPOF dynamic weights: premium_sat_run failed ({type(e).__name__}: {e})")
        try:
            rec = self._premium_alt.get(sym)
            out["premium_alt"] = dict(rec) if rec is not None else None
        except Exception as e:
            self._warn("premium_alt_row", f"DPOF dynamic weights: premium_alt row failed ({type(e).__name__}: {e})")
        try:
            dq = self._gex_obs.get(sym)
            if dq is not None and len(dq) >= _GEX_MIN_OBS:
                med = statistics.median([abs(x) for x in dq])
                if med != 0:
                    out["gex_cont"] = _round4(dq[-1] / med)
        except Exception as e:
            self._warn("gex_cont", f"DPOF dynamic weights: gex_cont failed ({type(e).__name__}: {e})")
        return out

    def _log_weight_change(self, sym: str, rounded: dict[str, float]) -> None:
        if not rounded:
            return
        prev = self._last_weights.get(sym)
        if prev is not None and any(
            _finite(prev.get(p)) is not None and abs(rounded[p] - _finite(prev.get(p))) > _WEIGHT_LOG_DELTA for p in rounded
        ):
            logger.info("DPOF dynamic weights: %s weights moved %s -> %s (mode=%s)", sym, prev, rounded, self._mode)
        if prev != rounded:
            self._last_weights[sym] = rounded
            self._dirty = True

    def _quality_for(self, qk: str, session_day: str) -> QualityStats:
        stats = self._quality.get(qk)
        if stats is None:
            stats = QualityStats()
            stats.roll_session(session_day, self._params)
            self._quality[qk] = stats
        return stats

    def _sample(
        self,
        sym: str,
        p: str,
        value: Any,
        u: UnderlyingSnapshot,
        now_epoch: float,
        now_et: datetime,
        session_day: str,
        in_session: bool,
    ) -> None:
        """L2 diagnostic: open one pending block sample per (sym, p, block); close it only after the horizon."""
        if not in_session:
            return  # nothing opens or closes out of session: a horizon must never span the close
        qk = f"{sym}|{p}"
        pend = self._pending.get(qk)
        fresh = _price_fresh(u)
        if pend is not None:
            if now_epoch >= pend["t0"] + _GIVE_UP_S:
                # too late to be a clean 30-minute return (data gap, restart): drop, never complete
                del self._pending[qk]
                self._dirty = True
            elif fresh and now_epoch >= pend["t0"] + _HORIZON_S:
                y = float(u.last_price) / pend["p0"] - 1.0
                self._quality_for(qk, session_day).add_block(session_day, pend["idx"], pend["x"], pend["z"], y)
                del self._pending[qk]
                self._dirty = True
            return
        blk = _block(now_et, self._params.block_minutes)
        if blk is None:
            return
        idx, start = blk
        sk = (session_day, sym, p, idx)
        if sk in self._sampled or not fresh:
            return
        x = _finite(value)
        z = _finite(u.ret_15m)
        if x is None or z is None:
            return
        # t0 is the ACTUAL sample time (not the block start): p0 is the price seen now, so the 30-minute
        # horizon must run from now even after a mid-block restart.
        self._pending[qk] = {"idx": idx, "t0": float(now_epoch), "x": x, "z": z, "p0": float(u.last_price)}
        self._sampled.add(sk)
        self._dirty = True

    def _state_doc(self, session_day: str) -> dict:
        return {
            "session_day": session_day,
            "liveness": self._liveness.to_dict(),
            "quality": {k: q.to_dict() for k, q in self._quality.items()},
            "pending": {k: dict(v) for k, v in self._pending.items()},
            "last_weights": {k: dict(v) for k, v in self._last_weights.items()},
            "mode": self._mode,
        }

    # ------------------------------------------------------------------ I/O plumbing
    def _submit(self, fn: Callable[..., Any], *args: Any) -> bool:
        """Run a write inline (sync_io or no loop) or on the default executor; never awaited.

        Returns False when the write was DROPPED because _MAX_INFLIGHT_IO writes are already in flight (a stalled disk):
        shadow records are diagnostics, and a dropped state save is retried on the next cycle."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._sync_io or loop is None:
            try:
                fn(*args)
            except Exception as e:
                self._warn("io", f"DPOF dynamic weights: write failed ({type(e).__name__}: {e})")
            return True

        with self._io_cv:
            backlog = self._io_pending >= _MAX_INFLIGHT_IO
            if not backlog:
                self._io_pending += 1
        if backlog:
            self._warn("io_backlog", f"DPOF dynamic weights: {_MAX_INFLIGHT_IO} writes already in flight (slow disk?); dropping this write")
            return False

        def _run() -> None:
            try:
                fn(*args)
            except Exception as e:
                self._warn("io", f"DPOF dynamic weights: write failed ({type(e).__name__}: {e})")
            finally:
                with self._io_cv:
                    self._io_pending -= 1
                    self._io_cv.notify_all()

        try:
            fut = loop.run_in_executor(None, _run)
        except Exception as e:
            with self._io_cv:
                self._io_pending -= 1
                self._io_cv.notify_all()
            self._warn("io", f"DPOF dynamic weights: executor submit failed ({type(e).__name__}: {e})")
            return False
        self._inflight.add(fut)
        fut.add_done_callback(self._inflight.discard)
        return True

    def _warn(self, key: str, msg: str) -> None:
        """Log a WARNING at most once per hour per key."""
        now = time.monotonic()
        last = self._warned.get(key)
        if last is not None and now - last < _WARN_INTERVAL_S:
            return
        self._warned[key] = now
        logger.warning("%s", msg)
