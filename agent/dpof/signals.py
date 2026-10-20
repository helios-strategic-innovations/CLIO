"""
CLIO DPOF — signal computation (pure).

Regime classification, the NOFA/premium/dark-pool/GEX composite, price
confirmation, instrument selection and thesis-invalidation voting. See
docs/design/clio_dpof.md §4.

No I/O, no clock reads: every function takes its inputs (including any
`now`) as parameters. Imports only stdlib + agent.dpof.config/models.

MISSING IS NOT ZERO (models.py): an unavailable pillar/price/VIX reading is
never coerced to 0.0. `compute_composite` requires NOFA plus >= cfg.min_pillars
real readings (GEX counts) before it will produce a number at all; short of
that it returns None and a machine-readable reason.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping

from agent.dpof.config import DPOFConfig
from agent.dpof.models import (
    Direction,
    MarketSnapshot,
    Regime,
    Signal,
    UnderlyingSnapshot,
)

logger = logging.getLogger("clio.dpof.signals")

# The three directional pillars that feed `pre` in compute_composite. GEX is
# handled separately — it is direction-agnostic (design doc §4): it never
# picks a side, only amplifies/dampens the sign already set by these three.
_FLOW_DIRECTIONAL_PILLARS = ("nofa", "premium", "darkpool")
# "kronos" (docs/design/clio_kronos.md §4b) is a directional pillar too, but a
# technical one: it adds direction to a composite the FLOW pillars already
# qualify (NOFA + min_pillars), never counts toward that minimum.
KRONOS_PILLAR = "kronos"
_DIRECTIONAL_PILLARS = _FLOW_DIRECTIONAL_PILLARS + (KRONOS_PILLAR,)


def _sign(x: float) -> int:
    """+1 / -1 / 0. Plain math sign, not Direction.sign."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def classify_regime(vix: float | None, vix_age_seconds: float | None, cfg: DPOFConfig) -> Regime:
    """VIX-based regime. Missing/stale VIX -> UNKNOWN (treated like CHOPPY downstream).

    Interpretation note: the spec's "None, or age > vix_max_age_seconds -> UNKNOWN"
    is read here as covering BOTH `vix is None` and `vix_age_seconds is None` (not
    just `vix is None`) — a reading with no recorded age is treated as unknown
    freshness, not implicitly fresh. This matches the "missing is not zero"
    fail-closed posture used throughout agent/dpof.

    Regime.HALT is presented to the decider as the "EXTREME" label (engine.py's
    regime ctx) — the enum member keeps its original name/value ("halt") since
    it is persisted/logged history, but per operator directive (Task 1,
    2026-09-25) it is no longer an automatic entry veto; evaluate_signal()
    below requires a STRONG composite to nominate at/above this threshold,
    exactly like CHOPPY/UNKNOWN already did, instead of vetoing outright.
    """
    if vix is None or vix_age_seconds is None or vix_age_seconds > cfg.vix_max_age_seconds:
        return Regime.UNKNOWN
    if vix >= cfg.vix_extreme_label_at:
        return Regime.HALT
    if vix >= cfg.vix_choppy_at:
        return Regime.CHOPPY
    if vix < cfg.vix_trending_below:
        return Regime.TRENDING
    return Regime.NORMAL


def gex_factor(regime: str | None, pin_risk: bool | None) -> float | None:
    """Direction-agnostic GEX factor (design doc §4 — fixes the Hydrinia bug where
    GEX biased direction). None regime -> None (unavailable). Case-insensitive.
    pin_risk True (and only True — not None, not False) halves the magnitude.
    """
    if regime is None:
        return None
    r = regime.strip().lower()
    if r in ("negative_gamma", "accelerating"):
        factor = 0.5
    elif r in ("positive_gamma", "stabilizing"):
        factor = -0.3
    else:
        factor = 0.0
    if pin_risk is True:
        factor *= 0.5
    return factor


def kronos_pillar_from_forecast(entry: dict | None, horizon: int) -> tuple[float, dict] | None:
    """The Kronos pillar value for one symbol's forecast entry (agent.kronos.state schema, status "ok"):
    2*p_up - 1 at `horizon` minutes, or -- when that horizon was omitted because it would run past the
    session close -- at the longest available horizon below it. None when no usable horizon exists
    (never a 0.0 stand-in). Returns (value in [-1, 1], detail)."""
    if not isinstance(entry, dict):
        return None
    horizons = entry.get("horizons")
    if not isinstance(horizons, dict):
        return None
    for h in sorted((m for m in (15, 30, 60) if m <= horizon), reverse=True):
        hz = horizons.get(str(h))
        p_up = hz.get("p_up") if isinstance(hz, dict) else None
        if isinstance(p_up, bool) or not isinstance(p_up, (int, float)):
            continue
        p_up = float(p_up)
        if not (0.0 <= p_up <= 1.0):  # NaN fails both comparisons
            continue
        value = max(-1.0, min(1.0, 2.0 * p_up - 1.0))
        return value, {"horizon_min": h, "p_up": p_up, "asof_bar": entry.get("asof_bar")}
    return None


def _usable_weight(v) -> float | None:
    """A dynamic weight is usable only if it is a real, finite, non-negative number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    if not math.isfinite(f) or f < 0.0:
        return None
    return f


# Legacy (non-NBBO) darkpool readings with |v| below this are treated as flat
# (no information) when CLIO_DPOF_DARKPOOL_FLAT_UNAVAILABLE is on (S215 W7A / R-36).
FLAT_DARKPOOL_EPS = 1e-3
FLAT_DARKPOOL_REASON = "darkpool_flat_unavailable"

# _composite flat-darkpool modes: "auto" = obey cfg.darkpool_flat_unavailable
# (production); "keep" = never exclude (shadow "with flat"); "exclude" = apply
# the rule regardless of the flag (shadow "without flat").
_FLAT_AUTO, _FLAT_KEEP, _FLAT_EXCLUDE = "auto", "keep", "exclude"


def _flat_legacy_darkpool(u: UnderlyingSnapshot, cfg: DPOFConfig) -> bool:
    """True when the darkpool pillar is a legacy (non-NBBO) reading with |v| < FLAT_DARKPOOL_EPS."""
    r = u.pillars.get("darkpool")
    if r is None or not r.available or r.value is None:
        return False
    if cfg.darkpool_nbbo or "#nbbo" in (r.source or ""):
        return False  # the NBBO path is never affected by the flat rule
    return abs(r.value) < FLAT_DARKPOOL_EPS


def _flat_darkpool_excluded(
    u: UnderlyingSnapshot, cfg: DPOFConfig, directional: dict, gex_available: bool, mode: str
) -> bool:
    """Exclude a flat legacy darkpool only when that keeps count_available >= min_pillars
    (never creates a new insufficient-pillars veto)."""
    if mode == _FLAT_KEEP or "nofa" not in directional or "darkpool" not in directional:
        return False
    if mode == _FLAT_AUTO and not getattr(cfg, "darkpool_flat_unavailable", False):
        return False
    if not _flat_legacy_darkpool(u, cfg):
        return False
    count_full = sum(1 for k in directional if k in _FLOW_DIRECTIONAL_PILLARS) + (1 if gex_available else 0)
    return count_full - 1 >= cfg.min_pillars


def compute_composite(
    u: UnderlyingSnapshot, cfg: DPOFConfig, weights=None
) -> tuple[float | None, dict[str, float], list[str]]:
    """Weighted NOFA/premium/darkpool(/Kronos)/GEX composite in [-1, +1] for one underlying.

    Returns (composite_or_None, used_values_by_name, reasons). `used_values_by_name`
    is populated with whatever directional/GEX pillars were actually available even
    when composite ends up None, so callers get a diagnosable Signal.pillars_used
    regardless of outcome.
    """
    return _composite(u, cfg, weights, _FLAT_AUTO)


def composite_flat_variants(u: UnderlyingSnapshot, cfg: DPOFConfig) -> tuple[float | None, float | None]:
    """Static-weight composites for the same inputs: (with_flat, without_flat).

    with_flat keeps the darkpool pillar as today; without_flat applies the flat-darkpool
    rule (guard included) whatever the flag says. Shadow evidence only: neither value is
    used for decisions. Each is None when not computable.
    """
    static_u = dataclasses.replace(u, pillar_weights=None)
    with_flat = _composite(static_u, cfg, None, _FLAT_KEEP)[0]
    without_flat = _composite(static_u, cfg, None, _FLAT_EXCLUDE)[0]
    return with_flat, without_flat


def _composite(
    u: UnderlyingSnapshot, cfg: DPOFConfig, weights, flat_mode: str
) -> tuple[float | None, dict[str, float], list[str]]:
    avail = u.available_pillars()
    directional = {k: v for k, v in avail.items() if k in _DIRECTIONAL_PILLARS}
    if getattr(cfg, "kronos_pillar", "off") != "live":
        directional.pop(KRONOS_PILLAR, None)

    gex_available = False
    gex_val = 0.0
    if "gex" in avail:
        # Pre-computed by an upstream producer (data.py) — its .value IS the
        # gex_factor already (design doc: "its .value is the gex_factor").
        gex_available = True
        gex_val = avail["gex"]
    else:
        factor = gex_factor(u.gex_regime, u.pin_risk)
        if factor is not None:
            gex_available = True
            gex_val = factor

    flat_excluded = _flat_darkpool_excluded(u, cfg, directional, gex_available, flat_mode)
    if flat_excluded:
        directional.pop("darkpool", None)

    used_values: dict[str, float] = dict(directional)
    if gex_available:
        used_values["gex"] = gex_val

    reasons: list[str] = []

    if "nofa" not in directional:
        reasons.append("nofa_unavailable")
        return None, used_values, reasons

    count_available = sum(1 for k in directional if k in _FLOW_DIRECTIONAL_PILLARS) + (1 if gex_available else 0)
    if count_available < cfg.min_pillars:
        reasons.append(f"insufficient_pillars:{count_available}<{cfg.min_pillars}")
        return None, used_values, reasons

    static = {"nofa": cfg.w_nofa, "premium": cfg.w_premium, "darkpool": cfg.w_darkpool,
              KRONOS_PILLAR: getattr(cfg, "w_kronos", 0.0)}
    # Explicit `weights` argument wins over the snapshot field; per key, an
    # unusable value falls back to the static weight for that key only.
    override = weights if weights is not None else getattr(u, "pillar_weights", None)
    used_w = static
    dynamic_applied = False
    if isinstance(override, Mapping):
        mixed = {}
        for k, sv in static.items():
            dv = _usable_weight(override.get(k))
            if dv is None:
                mixed[k] = sv
            else:
                mixed[k] = dv
                dynamic_applied = True
        if dynamic_applied:
            used_w = mixed
    sum_w = sum(used_w[k] for k in directional)
    sum_wp = sum(used_w[k] * directional[k] for k in directional)
    if sum_w == 0:
        # Defensive only — NOFA is always in `directional` here, so this needs a
        # zero-weighted config (invalid per DPOFConfig.validate()) to hit.
        reasons.append("zero_weight_sum")
        return None, used_values, reasons

    pre = sum_wp / sum_w
    contrib = gex_val * _sign(pre) if gex_available else 0.0

    numerator = sum_wp + (cfg.w_gex * contrib if gex_available else 0.0)
    denominator = sum_w + (cfg.w_gex if gex_available else 0.0)

    composite = numerator / denominator
    composite = max(-1.0, min(1.0, composite))
    if flat_excluded:
        reasons.append(FLAT_DARKPOOL_REASON)
    if dynamic_applied:
        reasons.append("weights:dynamic")
    return composite, used_values, reasons


def price_confirms(u: UnderlyingSnapshot, direction: Direction) -> tuple[bool, list[str]]:
    """Session-VWAP side + 15-min momentum must agree with `direction`.

    Any of last_price/vwap/ret_15m missing -> (False, ["price_data_missing"]).
    A Direction.NEUTRAL input has nothing to confirm -> (False, ["direction_neutral"]).
    """
    if u.last_price is None or u.vwap is None or u.ret_15m is None:
        return False, ["price_data_missing"]

    reasons: list[str] = []
    if direction is Direction.BULLISH:
        if not (u.last_price > u.vwap):
            reasons.append("price_below_vwap")
        if not (u.ret_15m > 0):
            reasons.append("momentum_not_positive")
    elif direction is Direction.BEARISH:
        if not (u.last_price < u.vwap):
            reasons.append("price_above_vwap")
        if not (u.ret_15m < 0):
            reasons.append("momentum_not_negative")
    else:
        return False, ["direction_neutral"]

    return (len(reasons) == 0, reasons)


def select_instrument(
    snap: MarketSnapshot, composite_sign: int, cfg: DPOFConfig
) -> tuple[str, list[str]]:
    """Primary underlying, unless the secondary (e.g. QQQ) diverges strongly enough
    and in the same direction as the primary's composite (design doc §4).
    """
    primary = cfg.underlyings[0]
    if len(cfg.underlyings) < 2:
        return primary, ["single_underlying_configured"]

    secondary = cfg.underlyings[1]
    u1 = snap.underlyings.get(primary)
    u2 = snap.underlyings.get(secondary)
    nofa1 = u1.pillars.get("nofa") if u1 is not None else None
    nofa2 = u2.pillars.get("nofa") if u2 is not None else None

    if nofa1 is None or not nofa1.available or nofa1.value is None:
        return primary, [f"{primary}_nofa_unavailable"]
    if nofa2 is None or not nofa2.available or nofa2.value is None:
        return primary, [f"{secondary}_nofa_unavailable"]

    v1, v2 = nofa1.value, nofa2.value
    if abs(v2) >= cfg.qqq_divergence_ratio * abs(v1) and _sign(v2) == composite_sign:
        return secondary, [f"divergence_selected:{secondary}"]
    return primary, [f"primary_selected:{primary}"]


def evaluate_signal(snap: MarketSnapshot, cfg: DPOFConfig) -> Signal:
    """The full ordered decision per docs/design/clio_dpof.md §4. See BUILD_SPEC.md
    signals.py docstring for the exact 9-step order this follows.
    """
    reasons: list[str] = []
    vetoes: list[str] = []

    if not snap.data_healthy:
        vetoes.append("data_unhealthy")
        vetoes.extend(snap.health_issues)

    # Task 1 (2026-09-25, operator directive): econ blackout / breaking news /
    # blocked vol regime are no longer deterministic vetoes -- they are
    # DECISION CONTEXT for the decider now (engine.py's ctx["events"]/
    # ["breaking"]/["regime"], and snap.risk_flags/econ_events_soon are still
    # passed through to Jev unchanged). Surfaced here as diagnosable `reasons`
    # (never `vetoes`) so status_text()/decision logs still show them without
    # blocking a nomination. See docs/design/clio_dpof.md §4/Task1.
    for f in snap.risk_flags:
        reasons.append(f"risk_flag:{f}")

    primary = cfg.primary
    u = snap.underlyings.get(primary)
    if u is None:
        composite, pillars_used, comp_reasons = None, {}, ["underlying_snapshot_missing"]
    else:
        composite, pillars_used, comp_reasons = compute_composite(u, cfg)
    reasons.extend(comp_reasons)

    # Task 1 (2026-09-25): VIX HALT/"EXTREME" no longer vetoes outright -- see
    # the strength-downgrade rule below (regime in (CHOPPY, UNKNOWN, HALT)),
    # which now applies the SAME "requires STRONG composite" treatment HALT's
    # veto used to enforce by itself. `regime` is still returned on the Signal
    # either way, so callers always see it.
    regime = classify_regime(snap.vix, snap.vix_age_seconds, cfg)

    if composite is None:
        # Not computable (NOFA missing / insufficient pillars / no snapshot):
        # composite stays None — never a 0.0 stand-in (models.Signal).
        reasons.append("composite_unavailable")
        return Signal(
            symbol=primary,
            direction=Direction.NEUTRAL,
            composite=None,
            conviction=0.0,
            strength="none",
            regime=regime,
            pillars_used=pillars_used,
            price_confirmed=False,
            reasons=reasons,
            vetoes=vetoes,
        )

    assert u is not None  # composite is not None only when u was resolved above
    c = composite
    csign = _sign(c)
    ac = abs(c)

    if ac >= cfg.threshold_strong:
        strength = "strong"
    elif ac >= cfg.threshold_normal:
        strength = "normal"
    else:
        strength = "none"
        reasons.append("composite_below_normal_threshold")

    if strength == "normal" and regime in (Regime.CHOPPY, Regime.UNKNOWN, Regime.HALT):
        # HALT ("EXTREME" label, Task 1) gets the SAME strong-only treatment
        # CHOPPY/UNKNOWN already had, in place of the old outright veto.
        strength = "none"
        reasons.append("regime_requires_strong")

    # Signal.direction reflects only an ACTIONABLE lean (strength != "none").
    # DP veto / instrument selection / price confirmation still use the raw
    # composite sign ("candidate_direction") below — a weak/gated signal still
    # has a directional lean worth diagnosing even though it won't fire.
    direction = Direction.NEUTRAL
    if strength != "none":
        direction = Direction.BULLISH if csign > 0 else (Direction.BEARISH if csign < 0 else Direction.NEUTRAL)

    candidate_direction = Direction.BULLISH if csign > 0 else (Direction.BEARISH if csign < 0 else Direction.NEUTRAL)

    dp_reading = u.pillars.get("darkpool")
    if (
        dp_reading is not None
        and dp_reading.available
        and dp_reading.value is not None
        and FLAT_DARKPOOL_REASON not in comp_reasons  # an excluded flat pillar cannot veto
    ):
        if dp_reading.value * csign <= -cfg.dp_veto_threshold:
            vetoes.append("dp_veto")

    symbol, instr_reasons = select_instrument(snap, csign, cfg)
    reasons.extend(instr_reasons)

    price_confirmed = False
    if cfg.require_price_confirmation:
        target_u = snap.underlyings.get(symbol)
        if target_u is None:
            reasons.append("price_data_missing")
            vetoes.append("price_not_confirmed")
        else:
            price_confirmed, pc_reasons = price_confirms(target_u, candidate_direction)
            reasons.extend(pc_reasons)
            if not price_confirmed:
                vetoes.append("price_not_confirmed")

    conviction = min(1.0, ac / cfg.threshold_strong) if cfg.threshold_strong > 0 else 0.0

    return Signal(
        symbol=symbol,
        direction=direction,
        composite=c,
        conviction=conviction,
        strength=strength,
        regime=regime,
        pillars_used=pillars_used,
        price_confirmed=price_confirmed,
        reasons=reasons,
        vetoes=vetoes,
    )


def _get_field(obj, key: str):
    """Accepts either a dict (the ctx-shaped per-underlying mapping the engine
    builds for the decider — DPOF_V2_SPEC.md's Interfaces section:
    {SYM: composite, pillars{...}, ..., vwap, last, above_vwap, ret_5m,
    ret_15m, ...}) or an object exposing the same names as attributes (e.g.
    UnderlyingSnapshot). Missing on either shape -> None, never raises.
    """
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


# Small neutral bands (not exact-zero comparisons) so a composite/return/
# price that is genuinely flat reads as "neutral" rather than a coin-flip
# "against" from float noise.
_ALIGN_NEUTRAL_COMPOSITE_BAND = 0.05      # |composite| below this -> neutral
_ALIGN_NEUTRAL_RET_BAND = 0.0005          # |ret_15m| below this (0.05%) -> neutral
_ALIGN_NEUTRAL_VWAP_BAND_PCT = 0.0005     # |last-vwap|/vwap below this -> neutral


def position_alignment(snapshot_for_underlying, direction: Direction) -> dict:
    """PURE, no I/O. Words {"flow": "aligned"|"neutral"|"against", "momentum":
    ..., "vwap": ...} describing whether the CURRENT market read still agrees
    with a HELD position's direction — for the engine to fold into a review's
    trigger/context (DPOF_V2_SPEC.md's "review triggers": composite flipped
    against the position, price crossed VWAP against the position). This is
    NEVER a gate or an exit decision by itself — per operator directive
    (2026-09-24), triggers only ever SCHEDULE a review; the decider is the
    only thing that can act on what this says.

    Accepts either the ctx-shaped per-underlying dict the engine already
    builds for decider.py (keyed "composite"/"ret_15m"/"vwap"/"last") or an
    UnderlyingSnapshot-like object exposing the same names as attributes
    (falls back to "last_price" if "last" isn't present, since that's
    UnderlyingSnapshot's own field name for the same value) — see
    _get_field(). A NEUTRAL `direction` has nothing to align against and
    returns all-"neutral" immediately (mirrors price_confirms' treatment of
    Direction.NEUTRAL).

    Missing/non-numeric inputs never raise and never guess: that axis simply
    stays "neutral" (unknown alignment must never masquerade as agreement or
    disagreement — the same "missing is not zero" discipline models.py and
    the rest of this module apply to pillars/composite applies here too).
    """
    sign = direction.sign
    out = {"flow": "neutral", "momentum": "neutral", "vwap": "neutral"}
    if sign == 0:
        return out

    composite = _get_field(snapshot_for_underlying, "composite")
    if isinstance(composite, (int, float)) and abs(composite) >= _ALIGN_NEUTRAL_COMPOSITE_BAND:
        out["flow"] = "aligned" if _sign(composite) == sign else "against"

    ret_15m = _get_field(snapshot_for_underlying, "ret_15m")
    if isinstance(ret_15m, (int, float)) and abs(ret_15m) >= _ALIGN_NEUTRAL_RET_BAND:
        out["momentum"] = "aligned" if _sign(ret_15m) == sign else "against"

    last = _get_field(snapshot_for_underlying, "last")
    if last is None:
        last = _get_field(snapshot_for_underlying, "last_price")
    vwap = _get_field(snapshot_for_underlying, "vwap")
    if isinstance(last, (int, float)) and isinstance(vwap, (int, float)) and vwap > 0:
        rel = (last - vwap) / vwap
        if abs(rel) >= _ALIGN_NEUTRAL_VWAP_BAND_PCT:
            out["vwap"] = "aligned" if _sign(rel) == sign else "against"

    return out
