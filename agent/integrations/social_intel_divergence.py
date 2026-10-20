"""
§6.D — Narrative-vs-Statistical Divergence Detector (THE CONSUMER of the enrichment substrate).
Directive: CC_CLIO_6D_DIVERGENCE_DETECTOR_2026-06-25. **ADVISORY ONLY.**

Compares what the social narrative is saying (CLIO talk-leg: tweet_signals + stories + signal_links,
DISTINCT-VOICE-weighted) against the statistical/positioning picture (bot market_regime + CFTC COT
percentiles + VIX) on a common risk_on(-1)..risk_off(+1) axis, and renders a persistence-gated,
evidence-linked verdict when they DIVERGE — e.g. a narrative storm against a calm tape = possible
pre-repricing. Emits a Telegram advisory to the operator + an additive bridge file Hydrinia MAY read.

BINDING INVARIANTS (non-negotiable — see the directive):
1. ADVISORY ONLY. The detector NEVER writes atlas_team authority, the risk envelope, EMERGENCY_STOP,
   or any position/order API. Authority stack: operator > atlas_team > bot > clio_divergence(advisory).
   The only writes here are: judge_verdicts (an advisory LOG), a Telegram message, and an additive
   bridge file in a NEW moltbot_knowledge/ subdir. (Grep-proven no risk-write path in acceptance.)
2. PI3 — every verdict is claim -> ATTRIBUTED EVIDENCE -> confidence; evidence is a NON-EMPTY array,
   CHECK-enforced at the DB (migration 018). build_verdict raises UngroundedVerdict on empty evidence.
3. PERSISTENCE-GATED (anti-flap) — a verdict fires only after k consecutive passes hold the same
   divergence class. Kills the classifier flap mode.
4. DISTINCT-VOICE weighting (the 6/25 dedup-caveat refinement) — narrative intensity/velocity weight
   each signal by VOICE: near_dup-original (different author) = full; followup (self-repeat) = 0;
   echo / is_amplification (retweet) = 0. A single account spamming a take CANNOT fake a storm. The
   weighting lives server-side in social_intel.narrative_regime_agg (migration 018).

Model policy: ZERO LLM in the hot path (heuristic risk-axis mapping over server-side aggregates) —
the verdict is fully reproducible and externally checkable. Kill-switch CLIO_DIVERGENCE_ENABLED=0.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

# BEFORE agent.config: config snapshots env at import; standalone CLI runs need load_dotenv here
# (S203 finding — a load_dotenv in __main__ runs after imports and leaves the keys empty).
from dotenv import load_dotenv
load_dotenv()

import httpx

from agent.config import config
from agent.integrations.social_intel_enrich import (  # shared Supabase plumbing
    _SCHEMA,
    _TIMEOUT,
    _read_headers,
    _req,
    _supa_base,
    _supa_key,
)

logger = logging.getLogger("openclaw.social_intel_divergence")

_METHOD_VER = "clio_divergence_v1"

# ---- Knobs (restart-pinned by design — read once at import) -------------------------------------
_ENABLED = os.getenv("CLIO_DIVERGENCE_ENABLED", "0") not in ("0", "false", "no", "off")
_WINDOW_DAYS = int(os.getenv("CLIO_DIVERGENCE_WINDOW_DAYS", "3"))
_PERSIST_K = int(os.getenv("CLIO_DIVERGENCE_PERSIST_K", "3"))          # consecutive holds to fire
_PERSIST_MAX_AGE_S = float(os.getenv("CLIO_DIVERGENCE_PERSIST_MAX_AGE_S", "21600"))  # 6h: older buffer entries aren't "consecutive"
_ADVISORY_FLOOR = float(os.getenv("CLIO_DIVERGENCE_ADVISORY_FLOOR", "0.6"))  # min conf to advise
_AGREE_THRESHOLD = float(os.getenv("CLIO_DIVERGENCE_AGREE_THRESHOLD", "0.40"))  # |narr-stat| < this = AGREE
_CONV_FLOOR = float(os.getenv("CLIO_DIVERGENCE_CONV_FLOOR", "0.45"))   # min conviction to "lead"
_COT_STALE_DAYS = int(os.getenv("CLIO_DIVERGENCE_COT_STALE_DAYS", "10"))
_REEMIT_HOURS = float(os.getenv("CLIO_DIVERGENCE_REEMIT_HOURS", "12"))  # cooldown re-advising same class
_TOP_K = 5
# CLIO-native market_regime.py staleness knobs for _bot_regime_component() (mirrors
# _cot_component()'s stale-but-still-surfaced / zero-weighted pattern): a read is
# accepted up to _BOT_REGIME_MAX_AGE_S old (else market_regime.get_market_regime()
# itself returns None and the component is dropped entirely -- "missing"), but only
# treated as FRESH (full weight) within _BOT_REGIME_STALE_S.
_BOT_REGIME_STALE_S = float(os.getenv("CLIO_DIVERGENCE_REGIME_STALE_S", str(6 * 3600)))
_BOT_REGIME_MAX_AGE_S = float(os.getenv("CLIO_DIVERGENCE_REGIME_MAX_AGE_S", str(24 * 3600)))

# Paths (local CLIO data + the Hydrinia bridge). os.getenv only; no hardcoded secrets (CG4).
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
_STATE_PATH = os.path.join(_DATA_DIR, "clio_divergence_state.json")
_COT_PATH = os.getenv("CLIO_COT_DERIVED_PATH", "/home/agent/tradingbot/state/cot/cot_derived.json")

_run_lock = asyncio.Lock()

# ---- Theme -> risk-axis map (v1 advisory heuristic; TRANSPARENT — the full theme breakdown ships in
#      the verdict evidence so the operator/grader can audit every contribution). +risk_off / -risk_on.
_THEME_RISK = {
    # risk-off
    "geopolitics": 0.8, "middle_east": 0.9, "war": 1.0, "conflict": 0.9, "military": 0.8,
    "defense": 0.6, "sanctions": 0.7, "recession": 0.9, "crisis": 1.0, "crash": 1.0,
    "inflation": 0.6, "stagflation": 0.9, "hawkish": 0.5, "tariff": 0.6, "tariffs": 0.6,
    "trade_war": 0.7, "supply_chain": 0.5, "energy": 0.3, "contagion": 0.9, "default": 0.9,
    "selloff": 0.8, "layoffs": 0.6, "bankruptcy": 0.7,
    # risk-on
    "earnings": -0.5, "ai": -0.5, "growth": -0.6, "dovish": -0.5, "stimulus": -0.6,
    "easing": -0.5, "rally": -0.7, "ipo": -0.4, "semiconductors": -0.4, "buyback": -0.4,
    "melt_up": -0.8, "breakout": -0.6,
}

# Bot market_regime (daily_metrics.json) -> risk axis. Short + canonical labels (CLAUDE.md).
_BOT_REGIME_RISK = {
    "BULL": -0.5, "BULL_TREND": -0.5, "MELT_UP": -0.7, "RECOVERY": -0.6,
    "BEAR": 0.5, "BEAR_TREND": 0.5, "DISTRIBUTION": 0.3,
    "CHOPPY": 0.0, "CHOPPY_RANGE": 0.0,
    "CRISIS": 1.0, "HIGH_VOL_CRASH": 1.0,
}


class UngroundedVerdict(RuntimeError):
    """PI3 guard: a verdict the detector cannot attribute to evidence is rejected (no naked verdicts)."""


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ============================================================
# Narrative leg (CLIO talk-leg; distinct-voice-weighted, server-side aggregate)
# ============================================================

async def _rpc(http: httpx.AsyncClient, fn: str, args: dict, label: str):
    url = f"{_supa_base()}/rest/v1/rpc/{fn}"
    headers = {"apikey": _supa_key(), "Authorization": f"Bearer {_supa_key()}",
               "Content-Profile": _SCHEMA, "Accept-Profile": _SCHEMA,
               "Content-Type": "application/json"}
    resp = await _req(http, "POST", url, label, headers=headers, json=args)
    if resp.status_code != 200:
        raise RuntimeError(f"{label} failed {resp.status_code}: {resp.text[:300]}")
    return resp.json()


async def narrative_regime(http: httpx.AsyncClient, window_days: int = _WINDOW_DAYS) -> dict:
    """Distinct-voice-weighted narrative on the risk axis. score in [-1,1] (+risk_off); intensity =
    total voice; velocity = (recent-prior)/prior; conviction blends |score| and velocity. Evidence =
    top story_ids + the theme breakdown that produced the score."""
    agg = await _rpc(http, "narrative_regime_agg",
                     {"p_window_days": window_days, "p_top_k": _TOP_K}, "narrative_regime_agg")
    agg = agg if isinstance(agg, dict) else (agg[0] if agg else {})
    # PostgREST returns the scalar jsonb directly; some clients wrap — normalize.
    if "themes" not in agg and isinstance(agg, dict) and len(agg) == 1:
        agg = next(iter(agg.values()))

    themes = agg.get("themes") or []
    total_voice = float(agg.get("total_voice") or 0.0)
    recent = float(agg.get("recent_voice") or 0.0)
    prior = float(agg.get("prior_voice") or 0.0)

    # risk score = voice-weighted theme risk valence (themes not in the map contribute 0)
    risk_num = 0.0
    mapped_voice = 0.0
    for t in themes:
        w = _THEME_RISK.get(str(t.get("theme", "")).lower())
        if w is None:
            continue
        v = float(t.get("voice") or 0.0)
        risk_num += w * v
        mapped_voice += v
    score = _clamp(risk_num / mapped_voice) if mapped_voice > 0 else 0.0

    velocity = (recent - prior) / prior if prior > 0 else (1.0 if recent > 0 else 0.0)
    # conviction: a loud, accelerating, directional narrative scores high
    conviction = _clamp(0.55 * abs(score) + 0.45 * _clamp(velocity, 0.0, 1.0), 0.0, 1.0)

    stories = agg.get("stories") or []
    evidence = []
    for s in stories[:_TOP_K]:
        evidence.append({"type": "narrative_story", "story_id": s.get("story_id"),
                         "label": (s.get("label") or "")[:80], "voice": s.get("voice")})
    top_themes = [{"theme": t.get("theme"), "voice": t.get("voice")}
                  for t in themes[:6] if _THEME_RISK.get(str(t.get("theme", "")).lower()) is not None]

    return {"score": round(score, 3), "intensity": round(total_voice, 1),
            "velocity": round(velocity, 3), "conviction": round(conviction, 3),
            "net_stance": agg.get("net_stance"), "signals": agg.get("signals"),
            "muted": agg.get("muted"), "top_themes": top_themes, "evidence": evidence}


# ============================================================
# Statistical leg (bot regime + COT + VIX)  — fail-OPEN per-component (advisory; degrade, don't crash)
# ============================================================

def _read_json_file(path: str) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 — advisory: a missing/stale source degrades, never crashes
        logger.debug("divergence: could not read %s: %s", path, e)
        return None


def _bot_regime_component() -> dict | None:
    """CLIO-native market regime (agent.integrations.market_regime), NOT Hydrinia's
    frozen daily_metrics.json["market_regime"] (Hydrinia retired 2026-09-24; that file
    never updates again). Mirrors _cot_component()'s staleness handling: a read within
    _BOT_REGIME_STALE_S is used at full weight; an older-but-not-yet-fully-expired read
    is still surfaced (never silently dropped from the evidence trail) but zero-weighted
    and flagged stale, mirroring _cot_component()'s degrade-not-crash pattern."""
    try:
        from agent.integrations.market_regime import get_market_regime
        mr = get_market_regime(max_age_seconds=_BOT_REGIME_MAX_AGE_S)
    except Exception as e:  # noqa: BLE001 — advisory: degrade, never crash
        logger.debug("divergence: market_regime unavailable: %s", e)
        return None
    if not isinstance(mr, dict):
        return None
    label = str(mr.get("regime") or "").upper()
    risk = _BOT_REGIME_RISK.get(label)
    if risk is None:
        return None
    conf_raw = mr.get("confidence")
    conf = float(conf_raw) if isinstance(conf_raw, (int, float)) else 0.5
    stale = False
    age_seconds = None
    as_of = mr.get("as_of")
    if isinstance(as_of, str):
        try:
            d = datetime.fromisoformat(as_of)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - d).total_seconds()
            stale = age_seconds > _BOT_REGIME_STALE_S
        except ValueError:
            pass
    return {"name": "bot_regime", "value": round(risk * conf, 3),
            "weight": 0.0 if stale else 1.0, "stale": stale,
            "raw": {"regime": label, "confidence": round(conf, 3), "age_seconds": age_seconds}}


def _cot_component() -> dict | None:
    cot = _read_json_file(_COT_PATH)
    if not isinstance(cot, dict):
        return None
    es = cot.get("ES")
    if not isinstance(es, dict):
        return None
    mm_pctile = es.get("mm_pctile_104wk")
    if not isinstance(mm_pctile, (int, float)):
        return None
    # managed-money equity positioning: low percentile = positioned risk-OFF (short equities) -> +;
    # high = risk-ON (long) -> -. (positioning STATE on the risk axis, in [-1,1]).
    risk = _clamp((0.5 - float(mm_pctile)) * 2.0)
    # staleness: report_date age (CFTC weekly; >COT_STALE_DAYS = degrade the weight + flag)
    stale = False
    age_days = None
    rd = es.get("report_date")
    if isinstance(rd, str):
        try:
            d = datetime.fromisoformat(rd)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - d).days
            stale = age_days > _COT_STALE_DAYS
        except ValueError:
            pass
    return {"name": "cot_es_positioning", "value": round(risk, 3),
            "weight": 0.4 if stale else 1.0, "stale": stale,
            "raw": {"es_mm_pctile_104wk": round(float(mm_pctile), 3), "report_date": rd,
                    "age_days": age_days}}


async def _vix_component(http: httpx.AsyncClient) -> dict | None:
    """VIX level -> stress on the risk axis. VIX3M is not sourced (see directive) so we use the VIX
    LEVEL (and, when present, Hydrinia's vol_regime term-structure flag). Fail-open."""
    vix = None
    try:
        from agent.integrations.polygon import get_snapshot_raw
        snap = await get_snapshot_raw("VIX")
        if isinstance(snap, dict):
            vix = snap.get("last_price") or snap.get("day_close")
    except Exception as e:  # noqa: BLE001 — advisory: degrade
        logger.debug("divergence: VIX fetch failed: %s", e)
    if not isinstance(vix, (int, float)) or vix <= 0:
        return None
    v = float(vix)
    if v < 15:
        risk = -0.3
    elif v < 20:
        risk = 0.0
    elif v < 30:
        risk = 0.4
    else:
        risk = 0.8
    return {"name": "vix_level", "value": round(risk, 3), "weight": 0.8,
            "raw": {"vix": round(v, 2)}}


async def statistical_regime(http: httpx.AsyncClient) -> dict:
    """Fuse bot regime + COT + VIX onto the risk axis. score in [-1,1] (+risk_off). conviction reflects
    component agreement + freshness. Evidence = the components[] (each with raw values). Degrades
    gracefully: a missing source is simply absent from the fusion (advisory, fail-open per-component)."""
    comps = []
    for c in (_bot_regime_component(), _cot_component()):
        if c:
            comps.append(c)
    vix_c = await _vix_component(http)
    if vix_c:
        comps.append(vix_c)

    if not comps:
        return {"score": 0.0, "conviction": 0.0, "components": [], "evidence": [],
                "degraded": True}

    wsum = sum(c["weight"] for c in comps) or 1.0
    score = _clamp(sum(c["value"] * c["weight"] for c in comps) / wsum)
    # conviction: magnitude + how aligned the components are (low spread = high conviction)
    vals = [c["value"] for c in comps]
    spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    conviction = _clamp(0.6 * abs(score) + 0.4 * (1.0 - _clamp(spread, 0.0, 1.0)), 0.0, 1.0)
    evidence = [{"type": "stat_component", **c} for c in comps]
    return {"score": round(score, 3), "conviction": round(conviction, 3),
            "components": comps, "evidence": evidence,
            "degraded": any(c.get("stale") for c in comps) or len(comps) < 2}


# ============================================================
# Classify -> persistence-gate -> PI3 verdict -> insert -> advise
# ============================================================

def classify_divergence(narr: dict, stat: dict) -> str:
    """AGREE / NARRATIVE_LEADS / STATISTICAL_LEADS / CONFLICT on the shared risk axis."""
    d = narr["score"] - stat["score"]
    if abs(d) < _AGREE_THRESHOLD:
        return "AGREE"
    narr_strong = narr["conviction"] >= _CONV_FLOOR
    stat_strong = stat["conviction"] >= _CONV_FLOOR
    opposite = (narr["score"] > 0) != (stat["score"] > 0) and abs(narr["score"]) > 0.15 and abs(stat["score"]) > 0.15
    if narr_strong and stat_strong and opposite:
        return "CONFLICT"
    if narr_strong and not stat_strong:
        return "NARRATIVE_LEADS"          # narrative moved, stats calm = pre-repricing watch
    if stat_strong and not narr_strong:
        return "STATISTICAL_LEADS"
    # both/neither strong: the louder side leads
    return "NARRATIVE_LEADS" if narr["conviction"] >= stat["conviction"] else "STATISTICAL_LEADS"


_STANCE = {"AGREE": "confirm", "NARRATIVE_LEADS": "watch",
           "STATISTICAL_LEADS": "watch", "CONFLICT": "contradict"}


def build_verdict(div_class: str, narr: dict, stat: dict) -> dict:
    """PI3 verdict: claim -> evidence (non-empty) -> confidence. Raises UngroundedVerdict if there is
    no evidence to attribute (no naked verdicts — mirrors the DB CHECK)."""
    evidence = list(narr.get("evidence") or []) + list(stat.get("evidence") or [])
    if not evidence:
        raise UngroundedVerdict(f"{div_class}: no narrative stories or stat components to ground the verdict")

    d = abs(narr["score"] - stat["score"])
    confidence = _clamp(0.30 + 0.40 * _clamp(d, 0.0, 1.0)
                        + 0.30 * max(narr["conviction"], stat["conviction"]), 0.0, 1.0)

    axis = lambda s: ("risk-off" if s > 0.15 else "risk-on" if s < -0.15 else "neutral")
    themes = ", ".join(f"{t['theme']}" for t in (narr.get("top_themes") or [])[:3]) or "—"
    if div_class == "NARRATIVE_LEADS":
        claim = (f"Narrative is {axis(narr['score'])} (themes: {themes}) and accelerating "
                 f"(voice {narr['intensity']}, vel {narr['velocity']:+.0%}) while the statistical/"
                 f"positioning tape reads {axis(stat['score'])} — possible pre-repricing. ADVISORY.")
    elif div_class == "STATISTICAL_LEADS":
        claim = (f"Statistical/positioning tape is {axis(stat['score'])} while the narrative stays "
                 f"{axis(narr['score'])}/quiet — stats leading. ADVISORY.")
    elif div_class == "CONFLICT":
        claim = (f"CONFLICT: narrative {axis(narr['score'])} vs positioning {axis(stat['score'])} — "
                 f"opposed risk signals. ADVISORY.")
    else:  # AGREE
        claim = f"Narrative and statistical regimes AGREE ({axis(narr['score'])}). ADVISORY."

    return {"subject": "market_regime", "claim": claim, "stance": _STANCE[div_class],
            "divergence_class": div_class, "confidence": round(confidence, 3),
            "evidence": evidence, "narrative_score": narr["score"],
            "statistical_score": stat["score"], "method_ver": _METHOD_VER}


# ---- persistence buffer (local state file; atomic write; survives restarts) ----------------------

def _load_state() -> dict:
    st = _read_json_file(_STATE_PATH)
    return st if isinstance(st, dict) else {"buffer": [], "last_emitted_class": None, "last_emit_ts": 0}


def _save_state(state: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    tmp = _STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, _STATE_PATH)


def _persistence_gate(state: dict, div_class: str) -> bool:
    """Append the class to the rolling buffer; True iff the last k entries ALL equal div_class AND are
    all recent. Stale entries (older than _PERSIST_MAX_AGE_S) are dropped first, so 'k consecutive' means
    k consecutive RECENT holds — a detector re-enabled after a gap cannot fire on stale buffer entries."""
    now = time.time()
    buf = [b for b in (state.get("buffer") or []) if (now - float(b.get("ts", 0))) <= _PERSIST_MAX_AGE_S]
    buf.append({"class": div_class, "ts": now})
    buf = buf[-max(_PERSIST_K, 1):]
    state["buffer"] = buf
    return len(buf) >= _PERSIST_K and all(b["class"] == div_class for b in buf)


# ---- emit (advisory ONLY: judge_verdicts log + Telegram + additive bridge file) -----------------

async def _insert_verdict(http: httpx.AsyncClient, verdict: dict) -> int | None:
    url = f"{_supa_base()}/rest/v1/judge_verdicts"
    headers = {"apikey": _supa_key(), "Authorization": f"Bearer {_supa_key()}",
               "Content-Profile": _SCHEMA, "Accept-Profile": _SCHEMA,
               "Content-Type": "application/json", "Prefer": "return=representation"}
    body = {k: verdict[k] for k in ("subject", "claim", "stance", "divergence_class", "confidence",
                                    "evidence", "narrative_score", "statistical_score", "method_ver")}
    resp = await _req(http, "POST", url, "judge_verdicts insert", headers=headers, json=body)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"judge_verdicts insert failed {resp.status_code}: {resp.text[:300]}")
    rows = resp.json()
    return rows[0]["id"] if isinstance(rows, list) and rows else None


def _write_bridge_file(verdict: dict, verdict_id) -> str | None:
    """Additive bridge file Hydrinia MAY read — a NEW moltbot_knowledge/clio_divergence/ subdir, so NO
    existing bridge schema changes (anti-pattern #3). Advisory only; never a risk/order signal."""
    try:
        from agent.tools.knowledge_tools import TRADINGBOT_MIRROR_DIR
    except Exception as e:  # noqa: BLE001
        logger.debug("divergence: TRADINGBOT_MIRROR_DIR import failed, using default path: %s", e)
        TRADINGBOT_MIRROR_DIR = "/home/agent/tradingbot/moltbot_knowledge"
    out_dir = os.path.join(TRADINGBOT_MIRROR_DIR, "clio_divergence")
    payload = {"id": verdict_id, "subject": verdict["subject"], "claim": verdict["claim"],
               "divergence_class": verdict["divergence_class"], "stance": verdict["stance"],
               "confidence": verdict["confidence"], "narrative_score": verdict["narrative_score"],
               "statistical_score": verdict["statistical_score"], "evidence": verdict["evidence"],
               "method_ver": verdict["method_ver"], "advisory": True,
               "authority": "clio_divergence(advisory)",
               "created": datetime.now(timezone.utc).isoformat()}
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "latest_verdict.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path)
        return path
    except Exception as e:  # noqa: BLE001 — advisory: bridge write is best-effort
        logger.warning("divergence bridge write failed: %s", e)
        return None


async def _send_advisory(context, verdict: dict) -> None:
    """Plain-text Telegram advisory to the operator (no parse_mode — CLAUDE.md). Best-effort."""
    if context is None or getattr(context, "bot", None) is None:
        return
    try:
        from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send
        msg = (f"CLIO DIVERGENCE ADVISORY — {verdict['divergence_class']} "
               f"(conf {verdict['confidence']:.0%})\n\n{verdict['claim']}\n\n"
               f"narrative {verdict['narrative_score']:+.2f} vs statistical "
               f"{verdict['statistical_score']:+.2f} (risk axis). Advisory only — not a trade signal.")
        for cid in _get_chat_ids():
            await _safe_send(context.bot, cid, text=msg)
    except Exception as e:  # noqa: BLE001 — advisory: never fail the pass on a send error
        logger.warning("divergence advisory send failed: %s", e)


# ============================================================
# divergence_pass — the §6.D orchestration (READ + classify + gate + advise; NO risk write)
# ============================================================

async def divergence_pass(http: httpx.AsyncClient, context=None, dry: bool = False) -> dict:
    """One detector pass. Reads narrative + statistical legs, classifies, applies the persistence gate,
    builds the PI3 verdict, inserts judge_verdicts, and (only on a non-AGREE stable-class TRANSITION
    with confidence >= ADVISORY_FLOOR) emits a Telegram advisory + additive bridge file.

    PROHIBITED — and absent from this code path: any write to atlas_team / the risk envelope /
    EMERGENCY_STOP / a position or order API. The detector is advisory only."""
    narr = await narrative_regime(http, _WINDOW_DAYS)
    stat = await statistical_regime(http)
    div_class = classify_divergence(narr, stat)
    verdict = build_verdict(div_class, narr, stat)   # raises UngroundedVerdict if unattributable

    result = {"divergence_class": div_class, "confidence": verdict["confidence"],
              "narrative_score": narr["score"], "statistical_score": stat["score"],
              "narr_conviction": narr["conviction"], "stat_conviction": stat["conviction"],
              "stat_degraded": stat.get("degraded"), "gated": False, "emitted": False,
              "verdict_id": None, "claim": verdict["claim"], "evidence_n": len(verdict["evidence"]),
              "dry": dry}

    if dry:
        result["verdict"] = verdict
        return result

    state = _load_state()
    stable = _persistence_gate(state, div_class)
    result["gated"] = stable

    if stable:
        prev = state.get("last_emitted_class")
        is_transition = (div_class != prev)
        # insert a verdict row only on a stable-class TRANSITION (the meaningful change), not every tick
        if is_transition:
            try:
                result["verdict_id"] = await _insert_verdict(http, verdict)
            except Exception as e:  # noqa: BLE001 — advisory: log, don't crash the tick
                logger.error("divergence verdict insert failed: %s", e, exc_info=True)
            # advise only on a non-AGREE, high-confidence transition (with a re-emit cooldown)
            cooled = (time.time() - float(state.get("last_emit_ts") or 0)) >= _REEMIT_HOURS * 3600
            if (div_class != "AGREE" and verdict["confidence"] >= _ADVISORY_FLOOR and cooled):
                await _send_advisory(context, verdict)
                bridge = _write_bridge_file(verdict, result["verdict_id"])
                result["emitted"] = True
                result["bridge_file"] = bridge
                state["last_emit_ts"] = time.time()
            state["last_emitted_class"] = div_class

    _save_state(state)
    logger.info("divergence: class=%s conf=%.2f narr=%+.2f stat=%+.2f gated=%s emitted=%s",
                div_class, verdict["confidence"], narr["score"], stat["score"],
                result["gated"], result["emitted"])
    return result


async def run_divergence(context=None, dry: bool = False) -> dict:
    """Scheduled/manual entry point. Single-flight via _run_lock; never raises in scheduled context."""
    if not _ENABLED and not dry:
        return {"status": "disabled"}
    if _run_lock.locked():
        return {"status": "skipped_concurrent"}
    async with _run_lock:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as http:
                return await divergence_pass(http, context=context, dry=dry)
        except UngroundedVerdict as e:
            logger.warning("divergence: ungrounded verdict rejected: %s", e)
            return {"status": "ungrounded", "error": str(e)}
        except Exception as e:  # noqa: BLE001 — scheduled context: report, never raise
            logger.error("divergence pass failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)[:500]}


# ============================================================
# Phase 5 — forward-repricing grader (scaffold; pre-registered decision set is the locked spec doc)
# ============================================================

async def run_divergence_grader(http: httpx.AsyncClient | None = None, grade_after_days: int = 5) -> dict:
    """Join each ungraded NARRATIVE_LEADS verdict older than grade_after_days to the FORWARD realized
    move, and write verifier_grade (precision/recall scorecard input). Decision set is PRE-REGISTERED
    in docs/design/clio_divergence_decision_set.md (locked BEFORE the first live verdict — no grading on
    a moving target). Scaffold: the forward-vol/price join lands with the datafeeds wiring (H1 cross-
    check). No self-assessment — the grade is the realized outcome, not the detector's opinion."""
    own = http is None
    if own:
        http = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True)
    try:
        cutoff = datetime.now(timezone.utc).timestamp() - grade_after_days * 86400
        url = (f"{_supa_base()}/rest/v1/judge_verdicts?select=id,created_at,narrative_score"
               f"&divergence_class=eq.NARRATIVE_LEADS&verifier_grade=is.null&order=created_at.asc&limit=100")
        resp = await _req(http, "GET", url, "grader read", headers=_read_headers())
        rows = resp.json() if resp.status_code == 200 else []
        due = []
        for r in rows:
            _cat = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            if _cat.tzinfo is None:
                _cat = _cat.replace(tzinfo=timezone.utc)
            if _cat.timestamp() <= cutoff:
                due.append(r)
        # Forward realized-move join is scaffolded (datafeeds wiring pending). Until then the grader
        # only reports the queue depth so the scorecard cadence is visible; it writes NO grade.
        logger.info("divergence grader: %d NARRATIVE_LEADS verdicts due for grading (forward join pending)",
                    len(due))
        return {"due_for_grading": len(due), "scanned": len(rows)}
    except Exception as e:  # noqa: BLE001
        logger.error("divergence grader failed: %s", e, exc_info=True)
        return {"error": str(e)[:500]}
    finally:
        if own:
            await http.aclose()


if __name__ == "__main__":
    # CLI: `dry` (default) → one pass, print verdict, NO emit/insert; `grade` → run the grader.
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    if mode == "grade":
        out = asyncio.run(run_divergence_grader())
    else:
        out = asyncio.run(run_divergence(dry=(mode == "dry")))
    print(json.dumps(out, indent=2, default=str))
