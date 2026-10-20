"""Pure core of the DPOF dynamic pillar weights (docs/design/clio_dpof_dynamic_weights.md).

Three layers: L1 LivenessTracker (dead/stuck pillar detection with hysteresis,
dwell and a revival ramp), L2 QualityStats (decayed partial-IC diagnostic,
inert unless quality_apply), L3 compute_weights (prior x aliveness x multiplier,
conservation to the available prior total, cap/floor bounds).

Pure core: callers own I/O. Nothing here reads files, sockets, the clock or
randomness; every time and session day is an argument.

Fail-static contract: malformed input is ignored or answered with the unknown
sentinel, from_dict never raises (corrupt state yields an empty object), and
compute_weights returns {} whenever the caller must fall back to static weights.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1
DIRECTIONAL = ("nofa", "premium", "darkpool", "kronos")
LIVENESS_PILLARS = ("premium", "darkpool")

_TRANSITION_KEEP = 20
_SUM_FIELDS = ("n", "sx", "sz", "sy", "sxx", "szz", "syy", "sxz", "sxy", "szy")


def _default_prior() -> dict[str, float]:
    return {"nofa": 0.40, "premium": 0.25, "darkpool": 0.20, "kronos": 0.15}


@dataclass(frozen=True)
class WeightParams:
    """Tunable parameters; defaults are the design-doc values."""

    prior: Mapping[str, float] = field(default_factory=_default_prior)
    ring_size: int = 40
    dead_eps: float = 0.005
    revive_eps: float = 0.01
    revive_n: int = 6
    dwell_seconds: float = 1800.0
    ramp_refreshes: int = 3
    cap: float = 0.60
    floor: float = 0.05
    block_minutes: int = 30
    n0: float = 1000.0
    n_floor: int = 350
    kappa: float = 4.0
    m_lo: float = 0.8
    m_hi: float = 1.2
    half_life_sessions: float = 20.0
    quality_apply: bool = False


def _num(v: Any) -> float | None:
    """Return v as a finite float, or None for bool, non-numbers, overflow, NaN or inf."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _clip(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


class _KeyState:
    __slots__ = (
        "ring",
        "dead",
        "last_flip_ts",
        "consec_alive",
        "ramp_stage",
        "last_session_day",
        "transitions",
    )

    def __init__(self, ring_size: int, ramp_stage: int) -> None:
        self.ring: deque[float] = deque(maxlen=ring_size)
        self.dead = False
        self.last_flip_ts: float | None = None
        self.consec_alive = 0
        self.ramp_stage = ramp_stage
        self.last_session_day: str | None = None
        self.transitions: list[tuple[float, str]] = []


class LivenessTracker:
    """L1: per-key dead/stuck detection with hysteresis, dwell and revival ramp."""

    def __init__(self, params: WeightParams) -> None:
        self._p = params
        self._states: dict[str, _KeyState] = {}

    def _state(self, key: str) -> _KeyState:
        st = self._states.get(key)
        if st is None:
            st = _KeyState(self._p.ring_size, self._p.ramp_refreshes)
            self._states[key] = st
        return st

    def _dwell_ok(self, st: _KeyState, ts: float) -> bool:
        return st.last_flip_ts is None or ts - st.last_flip_ts >= self._p.dwell_seconds

    @staticmethod
    def _flip(st: _KeyState, ts: float, to_dead: bool) -> None:
        st.dead = to_dead
        st.last_flip_ts = ts
        st.consec_alive = 0
        if not to_dead:
            st.ramp_stage = 1
        st.transitions.append((ts, "dead" if to_dead else "alive"))
        del st.transitions[:-_TRANSITION_KEEP]

    def observe(
        self,
        key: str,
        value: Any,
        ts: Any,
        session_day: Any,
        in_session: bool,
    ) -> None:
        """Feed one refresh value; invalid input or out-of-session calls are ignored."""
        if not in_session or not isinstance(key, str):
            return
        v = _num(value)
        t = _num(ts)
        if v is None or t is None:
            return
        if not isinstance(session_day, str) or not session_day:
            return
        st = self._state(key)
        if session_day != st.last_session_day:
            st.ring.clear()
            st.consec_alive = 0
            st.last_session_day = session_day
        st.ring.append(v)
        p = self._p
        was_dead = st.dead
        revived_now = False
        if not was_dead:
            full = len(st.ring) == p.ring_size
            small = max(abs(x) for x in st.ring) < p.dead_eps
            stuck = len(set(st.ring)) == 1
            if full and (small or stuck) and self._dwell_ok(st, t):
                self._flip(st, t, True)
        else:
            if abs(v) >= p.revive_eps:
                st.consec_alive += 1
            else:
                st.consec_alive = 0
            # A frozen non-zero value must not revive: the stuck arm would re-kill it one dwell later and the
            # pillar would duty-cycle dead/alive. Revival needs the last revive_n values to actually vary.
            varied = len(set(list(st.ring)[-p.revive_n:])) > 1
            if st.consec_alive >= p.revive_n and varied and self._dwell_ok(st, t):
                self._flip(st, t, False)
                revived_now = True
        if not st.dead and not revived_now and st.ramp_stage < p.ramp_refreshes:
            st.ramp_stage += 1

    def aliveness(self, key: str) -> float:
        """1.0 for alive or unknown keys, 0.0 when dead, a ramp fraction while reviving."""
        st = self._states.get(key)
        if st is None:
            return 1.0
        if st.dead:
            return 0.0
        if st.ramp_stage >= self._p.ramp_refreshes:
            return 1.0
        return st.ramp_stage / self._p.ramp_refreshes

    def is_dead(self, key: str) -> bool:
        """True when the key is currently flagged dead."""
        st = self._states.get(key)
        return bool(st is not None and st.dead)

    def transitions(self, key: str) -> list:
        """Copy of the last recorded (ts, 'dead'|'alive') transitions for key."""
        st = self._states.get(key)
        return list(st.transitions) if st is not None else []

    def keys(self) -> list[str]:
        """Sorted list of tracked keys."""
        return sorted(self._states)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot (finite floats only)."""
        keys: dict[str, dict] = {}
        for key in sorted(self._states):
            st = self._states[key]
            keys[key] = {
                "ring": [float(x) for x in st.ring],
                "dead": bool(st.dead),
                "last_flip_ts": st.last_flip_ts,
                "consec_alive": int(st.consec_alive),
                "ramp_stage": int(st.ramp_stage),
                "last_session_day": st.last_session_day,
                "transitions": [[float(t), label] for t, label in st.transitions],
            }
        return {"schema_version": SCHEMA_VERSION, "keys": keys}

    @classmethod
    def from_dict(cls, d: Any, params: WeightParams) -> LivenessTracker:
        """Restore from to_dict output; never raises (bad input yields an empty or partial tracker)."""
        tracker = cls(params)
        try:
            if not isinstance(d, Mapping):
                return tracker
            version = d.get("schema_version")
            if not _is_int(version) or version != SCHEMA_VERSION:
                return tracker
            keys = d.get("keys")
            if not isinstance(keys, Mapping):
                return tracker
            for key, rec in keys.items():
                st = _restore_key(key, rec, params)
                if st is not None:
                    tracker._states[key] = st
        except Exception:  # fail-static: corrupt state must never raise
            return cls(params)
        return tracker


def _restore_key(key: Any, rec: Any, params: WeightParams) -> _KeyState | None:
    if not isinstance(key, str) or not isinstance(rec, Mapping):
        return None
    ring_raw = rec.get("ring")
    if not isinstance(ring_raw, list) or len(ring_raw) > params.ring_size:
        return None
    ring: list[float] = []
    for x in ring_raw:
        f = _num(x)
        if f is None:
            return None
        ring.append(f)
    dead = rec.get("dead")
    if not isinstance(dead, bool):
        return None
    last_flip = rec.get("last_flip_ts")
    if last_flip is not None:
        last_flip = _num(last_flip)
        if last_flip is None:
            return None
    consec = rec.get("consec_alive")
    if not _is_int(consec) or consec < 0:
        return None
    ramp = rec.get("ramp_stage")
    if not _is_int(ramp) or not (1 <= ramp <= params.ramp_refreshes):
        return None
    last_day = rec.get("last_session_day")
    if last_day is not None and not isinstance(last_day, str):
        return None
    trs = rec.get("transitions")
    if not isinstance(trs, list):
        return None
    transitions: list[tuple[float, str]] = []
    for item in trs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        t = _num(item[0])
        label = item[1]
        if t is None or label not in ("dead", "alive"):
            return None
        transitions.append((t, label))
    st = _KeyState(params.ring_size, ramp)
    st.ring.extend(ring)
    st.dead = dead
    st.last_flip_ts = last_flip
    st.consec_alive = consec
    st.last_session_day = last_day
    st.transitions = transitions[-_TRANSITION_KEEP:]
    return st


class QualityStats:
    """L2: decayed sufficient statistics for the partial IC of r(x, y | z) for one (underlying, pillar)."""

    def __init__(self) -> None:
        self.n = 0.0
        self.sx = 0.0
        self.sz = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.szz = 0.0
        self.syy = 0.0
        self.sxz = 0.0
        self.sxy = 0.0
        self.szy = 0.0
        self.last_day: str | None = None
        self.added: list[str] = []

    def add_block(self, session_day: Any, block_index: Any, x: Any, z: Any, y: Any) -> bool:
        """Add one (x, z, y) sample once per (session_day, block_index); False if rejected or duplicate."""
        if not isinstance(session_day, str) or not session_day:
            return False
        if not _is_int(block_index) or block_index < 0:
            return False
        xv, zv, yv = _num(x), _num(z), _num(y)
        if xv is None or zv is None or yv is None:
            return False
        key = f"{session_day}#{block_index}"
        if key in self.added:
            return False
        delta = {
            "n": 1.0,
            "sx": xv,
            "sz": zv,
            "sy": yv,
            "sxx": xv * xv,
            "szz": zv * zv,
            "syy": yv * yv,
            "sxz": xv * zv,
            "sxy": xv * yv,
            "szy": zv * yv,
        }
        new = {name: getattr(self, name) + delta[name] for name in _SUM_FIELDS}
        if not all(math.isfinite(v) for v in new.values()):
            return False
        for name, value in new.items():
            setattr(self, name, value)
        self.added.append(key)
        return True

    def roll_session(self, new_session_day: Any, params: WeightParams) -> None:
        """Decay once per new session day (idempotent for the same day)."""
        if not isinstance(new_session_day, str) or not new_session_day:
            return
        if new_session_day == self.last_day:
            return
        if self.last_day is not None:
            lam = 0.5 ** (1.0 / params.half_life_sessions)
            for name in _SUM_FIELDS:
                setattr(self, name, getattr(self, name) * lam)
            old_day = self.last_day
            self.added = [k for k in self.added if k.rsplit("#", 1)[0] == old_day]
        self.last_day = new_session_day

    def n_eff(self) -> float:
        """Decayed sample count."""
        return self.n

    def partial_ic(self) -> float | None:
        """Partial correlation r(x, y | z), clamped to [-1, 1]; None when undefined."""
        n = self.n
        if not (n >= 6):
            return None
        mx, mz, my = self.sx / n, self.sz / n, self.sy / n
        vx = self.sxx / n - mx * mx
        vz = self.szz / n - mz * mz
        vy = self.syy / n - my * my
        if min(vx, vy, vz) <= 1e-18:
            return None
        cxy = self.sxy / n - mx * my
        cxz = self.sxz / n - mx * mz
        czy = self.szy / n - mz * my
        r_xy = _clip(cxy / math.sqrt(vx * vy), -1.0, 1.0)
        r_xz = _clip(cxz / math.sqrt(vx * vz), -1.0, 1.0)
        r_yz = _clip(czy / math.sqrt(vy * vz), -1.0, 1.0)
        d2 = (1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)
        denom = math.sqrt(max(d2, 0.0))
        if denom <= 1e-12:
            return None
        r = (r_xy - r_xz * r_yz) / denom
        if not math.isfinite(r):
            return None
        return _clip(r, -1.0, 1.0)

    def ci95(self) -> tuple[float, float] | None:
        """Fisher-z 95% interval for the partial IC (one control variable); None when undefined."""
        ne = self.n
        if not (ne > 6):
            return None
        r = self.partial_ic()
        if r is None:
            return None
        rc = _clip(r, -0.999999, 0.999999)
        z = math.atanh(rc)
        se = 1.0 / math.sqrt(ne - 4.0)
        return (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))

    def posterior(self, params: WeightParams) -> float | None:
        """Shrunk IC: n_eff/(n_eff+n0) * partial IC; None when partial IC is undefined."""
        r = self.partial_ic()
        if r is None:
            return None
        ne = self.n
        return ne / (ne + params.n0) * r

    def diagnostic_multiplier(self, params: WeightParams) -> float:
        """Multiplier from the posterior, ignoring all gates (for logging only)."""
        post = self.posterior(params)
        if post is None:
            return 1.0
        return _clip(1.0 + params.kappa * post, params.m_lo, params.m_hi)

    def multiplier(self, params: WeightParams) -> float:
        """1.0 unless quality_apply, n_eff >= n_floor and the 95% CI excludes 0."""
        if not params.quality_apply:
            return 1.0
        if not (self.n >= params.n_floor):
            return 1.0
        ci = self.ci95()
        if ci is None:
            return 1.0
        lo, hi = ci
        if not (lo > 0.0 or hi < 0.0):
            return 1.0
        return self.diagnostic_multiplier(params)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot."""
        out: dict[str, Any] = {name: float(getattr(self, name)) for name in _SUM_FIELDS}
        out["last_day"] = self.last_day
        out["added"] = list(self.added)
        return out

    @classmethod
    def from_dict(cls, d: Any) -> QualityStats:
        """Restore from to_dict output; never raises (bad input yields fresh empty stats)."""
        try:
            if not isinstance(d, Mapping):
                return cls()
            vals: dict[str, float] = {}
            for name in _SUM_FIELDS:
                f = _num(d.get(name))
                if f is None:
                    return cls()
                vals[name] = f
            if vals["n"] < 0.0:
                return cls()
            last_day = d.get("last_day")
            if last_day is not None and not isinstance(last_day, str):
                return cls()
            added = d.get("added")
            if not isinstance(added, list) or not all(isinstance(k, str) for k in added):
                return cls()
            out = cls()
            for name, value in vals.items():
                setattr(out, name, value)
            out.last_day = last_day
            out.added = list(added)
            return out
        except Exception:  # fail-static: corrupt state must never raise
            return cls()


def _apply_bounds(shares: Mapping[str, float], cap: float, floor: float) -> dict[str, float] | None:
    """Project positive shares onto weights in [floor, cap] that sum to 1, or None if that is impossible.

    The answer is w_i = clip(lam * share_i, floor, cap) for the one lam that makes the sum 1: unclamped pillars keep
    their proportions and only the pillars that hit a bound are pinned. The sum is piecewise linear and non-decreasing
    in lam, so the crossing is solved exactly between the sorted breakpoints (no iteration, no failure on skewed shares)."""
    items = {p: v for p, v in shares.items()}
    n = len(items)
    if n == 0 or not (floor * n <= 1.0 <= cap * n):
        return None
    for v in items.values():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0.0:
            return None

    def total(lam: float) -> float:
        return sum(min(cap, max(floor, lam * v)) for v in items.values())

    points = sorted({0.0} | {floor / v for v in items.values()} | {cap / v for v in items.values()})
    lam = None
    for lo, hi in zip(points, points[1:]):
        f_hi = total(hi)
        if f_hi >= 1.0:
            f_lo = total(lo)
            lam = hi if f_hi <= f_lo else lo + (1.0 - f_lo) / (f_hi - f_lo) * (hi - lo)
            break
    if lam is None:
        lam = points[-1]
    out = {p: min(cap, max(floor, lam * v)) for p, v in items.items()}
    if abs(sum(out.values()) - 1.0) > 1e-9:
        return None
    return out


def compute_weights(
    available: Mapping[str, Any],
    multipliers: Mapping[str, Any] | None,
    params: WeightParams,
) -> dict[str, float]:
    """L3: combine prior, aliveness and multiplier into conserved, bounded weights; {} means use static."""
    prior = params.prior
    pillars: list[str] = []
    prior_val: dict[str, float] = {}
    for p in sorted(k for k in available if isinstance(k, str)):
        if p not in prior:
            continue
        pv = _num(prior[p])
        if pv is None or pv < 0.0:
            return {}
        pillars.append(p)
        prior_val[p] = pv
    if not pillars:
        return {}

    alive: dict[str, float] = {}
    mult: dict[str, float] = {}
    for p in pillars:
        a = _num(available[p])
        alive[p] = 0.0 if a is None else _clip(a, 0.0, 1.0)
        m = _num(multipliers.get(p)) if multipliers is not None else None
        if m is None or m <= 0.0:
            m = 1.0
        mult[p] = _clip(m, params.m_lo, params.m_hi)

    raw = {p: prior_val[p] * alive[p] * mult[p] for p in pillars}
    total = sum(raw[p] for p in pillars)
    if not math.isfinite(total) or total <= 0.0:
        return {}
    target = sum(prior_val[p] for p in pillars)
    if not math.isfinite(target):
        return {}
    shares = {p: raw[p] / total for p in pillars}

    positive = [p for p in pillars if raw[p] > 0.0]
    identity = all(alive[p] == 1.0 and mult[p] == 1.0 for p in pillars)
    if not identity and len(positive) >= 3:
        feasible = params.floor * len(positive) <= 1.0 <= params.cap * len(positive)
        if feasible:
            bounded = _apply_bounds({p: shares[p] for p in positive}, params.cap, params.floor)
            if bounded is not None:
                for p in positive:
                    shares[p] = bounded[p]

    return {p: shares[p] * target for p in pillars}
