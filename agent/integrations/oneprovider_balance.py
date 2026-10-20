"""
OneProvider gateway monitor — balance / expiry watch plus a served-model canary.

Observation only. Nothing here may raise into a caller or touch a trading or
chat path. Every public coroutine swallows its own errors (debug log). The
Authorization header and the API key are never logged; reasons are fixed words.

Signals:
    1. fetch_balance()  -> GET <gateway>/v1/dashboard/balance (Bearer auth only).
    2. evaluate()       -> pure threshold / expiry / spend-rate verdict.
    3. run_canary()     -> tiny ping per configured model; records the SERVED
                           model id so a silent family swap is visible.

Runs every 6 h via oneprovider_monitor_job (registered only when
config.anthropic_gateway is on). Debounces Telegram alerts to one per level per
12 h (crit also re-alerts on a level change). State file:
    data/oneprovider_balance_state.json

Flag: CLIO_GATEWAY_MONITOR_ENABLED (default "1"; "0" disables the job body).
Thresholds: CLIO_GATEWAY_BALANCE_WARN_USD (75), CLIO_GATEWAY_BALANCE_CRIT_USD (25).
"""

import asyncio
import fcntl
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import anthropic
import httpx

logger = logging.getLogger("openclaw.oneprovider_balance")

_STATE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "oneprovider_balance_state.json")
)

_ALERT_DEBOUNCE_SEC = 12 * 3600
_MIN_BURN_ELAPSED_SEC = 3600
_EXPIRY_WARN_DAYS = 14.0
_EXPIRY_CRIT_DAYS = 3.0
_BALANCE_WARN_DEFAULT = 75.0
_BALANCE_CRIT_DEFAULT = 25.0
_CANARY_TIMEOUT_SEC = 30
_FAMILY_RE = re.compile(r"^claude-(opus|sonnet|haiku|fable)")
_LEVEL_RANK = {"ok": 0, "warn": 1, "crit": 2}
_ET = ZoneInfo("America/New_York")
_SAMPLE_RING = 48                       # balance samples kept in the state file (~12 days at 6 h)
_RTH_OPEN = (9, 30)
_RTH_CLOSE = (16, 0)
_LAT_BREACH_FRACTION = 0.8              # Opus p95 alert threshold = 0.8 x decider timeout
_LAT_MIN_N = 20                         # ignore p95 over fewer calls than this
_DECISION_TIMEOUT_DEFAULT_S = 60.0


def _lock_path() -> str:
    # Computed at call time so tests that monkeypatch _STATE_PATH never touch the real lock.
    return _STATE_PATH + ".lock"


# ---------------------------------------------------------------------------
# Balance fetch
# ---------------------------------------------------------------------------

async def fetch_balance(timeout: float = 15.0, transport: Any = None) -> dict:
    """Fetch the OneProvider balance. Never raises.

    Returns {"balance_usd", "expires_at", "is_active", "last_synced_at"} on success,
    or {"error": <fixed word>} where the word is one of gateway_disabled,
    invalid_key, rate_limited, malformed, unavailable.
    """
    try:
        from agent.config import config

        if not config.anthropic_gateway or not config.anthropic_api_key:
            return {"error": "gateway_disabled"}
        url = config.anthropic_base_url.rstrip("/") + "/v1/dashboard/balance"
        headers = {"Authorization": f"Bearer {config.anthropic_api_key}"}
    except Exception:
        return {"error": "gateway_disabled"}

    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            resp = await client.get(url, headers=headers)
    except Exception:
        # Transport / timeout / connect errors: message deliberately not logged.
        return {"error": "unavailable"}

    try:
        status = resp.status_code
        if status == 401:
            return {"error": "invalid_key"}
        if status == 429:
            return {"error": "rate_limited"}
        if status != 200:
            return {"error": "unavailable"}
        return _parse_balance_body(resp)
    except Exception:
        return {"error": "malformed"}


def _parse_balance_body(resp: httpx.Response) -> dict:
    data = resp.json()
    if not isinstance(data, dict):
        return {"error": "malformed"}
    for key in ("balance_usd", "expires_at", "is_active", "last_synced_at"):
        if key not in data:
            return {"error": "malformed"}
    bal = data["balance_usd"]
    if isinstance(bal, bool) or not isinstance(bal, (int, float)) or not math.isfinite(bal):
        return {"error": "malformed"}
    active = data["is_active"]
    if not isinstance(active, bool):
        return {"error": "malformed"}
    expires = data["expires_at"]
    if expires is not None and not isinstance(expires, str):
        return {"error": "malformed"}
    synced = data["last_synced_at"]
    if synced is not None and not isinstance(synced, str):
        return {"error": "malformed"}
    return {
        "balance_usd": float(bal),
        "expires_at": expires,
        "is_active": active,
        "last_synced_at": synced,
    }


# ---------------------------------------------------------------------------
# State file (flock on a sidecar .lock; atomic tmp + os.replace)
# ---------------------------------------------------------------------------

def read_state_sync() -> dict:
    """Blocking. Run via run_in_executor. Corrupt/missing file -> {}."""
    lock_fd = None
    try:
        lock_fd = open(_lock_path(), "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if not os.path.isfile(_STATE_PATH):
            return {}
        with open(_STATE_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug(f"oneprovider state read failed: {e}")
        return {}
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def write_state_sync(state: dict) -> None:
    """Blocking. Run via run_in_executor. Atomic tmp + os.replace under flock."""
    lock_fd = None
    tmp = _STATE_PATH + ".tmp"
    try:
        os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
        lock_fd = open(_lock_path(), "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, _STATE_PATH)
    except Exception as e:
        logger.debug(f"oneprovider state write failed: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Evaluation (pure)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return value


def _parse_expiry(raw: Any, now_utc: datetime) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        exp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp


def _as_aware(now_utc: datetime) -> datetime:
    return now_utc if now_utc.tzinfo is not None else now_utc.replace(tzinfo=timezone.utc)


def evaluate(balance: dict, now_utc: datetime, prev_state: dict) -> dict:
    """Pure verdict on one balance sample.

    Returns {"level": ok|warn|crit, "reasons": [...], "days_to_expiry": float|None,
    "burn_per_day": float|None}.
    """
    prev_state = prev_state if isinstance(prev_state, dict) else {}
    err = balance.get("error")
    if err == "invalid_key":
        return {"level": "crit", "reasons": ["invalid_key"], "days_to_expiry": None, "burn_per_day": None}
    if err == "gateway_disabled":
        return {"level": "ok", "reasons": ["gateway_disabled"], "days_to_expiry": None, "burn_per_day": None}
    if err is not None:
        # unavailable / malformed / rate_limited / any unknown sample error
        return {"level": "warn", "reasons": ["no_sample"], "days_to_expiry": None, "burn_per_day": None}

    now_utc = _as_aware(now_utc)
    balance_usd = float(balance["balance_usd"])
    warn_usd = _env_float("CLIO_GATEWAY_BALANCE_WARN_USD", _BALANCE_WARN_DEFAULT)
    crit_usd = _env_float("CLIO_GATEWAY_BALANCE_CRIT_USD", _BALANCE_CRIT_DEFAULT)

    reasons: list[str] = []
    level = "ok"

    def _raise(to: str) -> None:
        nonlocal level
        if _LEVEL_RANK[to] > _LEVEL_RANK[level]:
            level = to

    exp = _parse_expiry(balance.get("expires_at"), now_utc)
    days_to_expiry = None
    if exp is not None:
        days_to_expiry = (exp - now_utc).total_seconds() / 86400.0

    if balance.get("is_active") is False:
        reasons.append("inactive")
        _raise("crit")
    if balance_usd < crit_usd:
        reasons.append("balance_critical")
        _raise("crit")
    elif balance_usd < warn_usd:
        reasons.append("balance_low")
        _raise("warn")
    if days_to_expiry is not None:
        if days_to_expiry < _EXPIRY_CRIT_DAYS:
            reasons.append("expiry_critical")
            _raise("crit")
        elif days_to_expiry < _EXPIRY_WARN_DAYS:
            reasons.append("expiry_soon")
            _raise("warn")

    burn = _burn_per_day(balance_usd, now_utc, prev_state)
    return {
        "level": level,
        "reasons": reasons,
        "days_to_expiry": days_to_expiry,
        "burn_per_day": burn,
    }


def _burn_per_day(balance_usd: float, now_utc: datetime, prev_state: dict) -> float | None:
    """Spend rate vs the previous successful sample. Top-ups (balance went up) yield None."""
    prev_bal = prev_state.get("balance_usd")
    prev_ts = prev_state.get("sample_ts")
    if isinstance(prev_bal, bool) or not isinstance(prev_bal, (int, float)):
        return None
    if isinstance(prev_ts, bool) or not isinstance(prev_ts, (int, float)):
        return None
    elapsed = now_utc.timestamp() - float(prev_ts)
    if elapsed < _MIN_BURN_ELAPSED_SEC:
        return None
    spent = float(prev_bal) - balance_usd
    if spent < 0:
        # Top-up (or a re-priced balance) between samples: not a spend signal.
        return None
    return spent / (elapsed / 86400.0)


# ---------------------------------------------------------------------------
# Served-model canary
# ---------------------------------------------------------------------------

def _is_num(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _valid_samples(raw: Any) -> list[dict]:
    """Ring entries from the state file; anything malformed is dropped, never raised on."""
    if not isinstance(raw, list):
        return []
    out = []
    for s in raw:
        if isinstance(s, dict) and _is_num(s.get("ts")) and _is_num(s.get("balance")):
            out.append({"ts": float(s["ts"]), "balance": float(s["balance"])})
    return out


def _append_sample(raw: Any, ts: float, balance_usd: float) -> list[dict]:
    """Append one sample and keep only the newest _SAMPLE_RING entries (bounded)."""
    samples = _valid_samples(raw)
    samples.append({"ts": float(ts), "balance": float(balance_usd)})
    samples.sort(key=lambda s: s["ts"])
    return samples[-_SAMPLE_RING:]


def _in_rth(ts: float) -> bool:
    local = datetime.fromtimestamp(ts, tz=_ET)
    return _RTH_OPEN <= (local.hour, local.minute) < _RTH_CLOSE


def _window_burn(pairs: list) -> float | None:
    """Spend per day over consecutive (earlier, later) sample pairs. Top-ups are not spend."""
    spent = 0.0
    elapsed = 0.0
    for a, b in pairs:
        elapsed += b["ts"] - a["ts"]
        delta = a["balance"] - b["balance"]
        if delta > 0:
            spent += delta
    if elapsed < _MIN_BURN_ELAPSED_SEC:
        return None
    return spent / (elapsed / 86400.0)


def compute_burns(samples: list[dict], now_ts: float) -> tuple[float | None, float | None]:
    """(burn_24h, burn_rth). burn_24h: pairs starting in the last 24 h.
    burn_rth: pairs where both samples are inside 09:30-16:00 ET on the same ET date."""
    pairs = list(zip(samples, samples[1:]))
    last_24h = [(a, b) for a, b in pairs if a["ts"] >= now_ts - 86400.0]
    rth = []
    for a, b in pairs:
        if not (_in_rth(a["ts"]) and _in_rth(b["ts"])):
            continue
        same_day = datetime.fromtimestamp(a["ts"], tz=_ET).date() == datetime.fromtimestamp(b["ts"], tz=_ET).date()
        if same_day:
            rth.append((a, b))
    return _window_burn(last_24h), _window_burn(rth)


def _fmt_burn(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}/d"


def _decision_timeout_s() -> float:
    """The DPOF decider timeout as the dpof config exposes it (default 60 s)."""
    try:
        from agent.dpof.config import load_dpof_config

        value = float(load_dpof_config().decision_timeout_seconds)
        if math.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    return _DECISION_TIMEOUT_DEFAULT_S


def _gateway_snapshot() -> dict | None:
    try:
        from agent.anthropic_gateway import get_stats

        return get_stats().snapshot()
    except Exception:
        return None


def opus_p95_breach(snapshot: dict | None, timeout_s: float) -> dict | None:
    """Worst Opus per-model p95 above 0.8 x timeout (with enough calls), else None."""
    if not isinstance(snapshot, dict):
        return None
    latency = snapshot.get("latency_ms")
    if not isinstance(latency, dict):
        return None
    threshold_ms = _LAT_BREACH_FRACTION * float(timeout_s) * 1000.0
    worst = None
    for model, stats in latency.items():
        if _family(model) != "opus" or not isinstance(stats, dict):
            continue
        n, p95 = stats.get("n"), stats.get("p95")
        if isinstance(n, bool) or not isinstance(n, int) or n < _LAT_MIN_N or not _is_num(p95):
            continue
        if p95 > threshold_ms and (worst is None or p95 > worst[1]):
            worst = (model, p95)
    if worst is None:
        return None
    return {"model": worst[0], "p95_ms": int(worst[1]), "threshold_ms": int(threshold_ms)}


def _family(model: str | None) -> str | None:
    if not model:
        return None
    m = _FAMILY_RE.match(model)
    return m.group(1) if m else None


def _ping_model_sync(model: str, client_factory: Callable[[], Any] | None) -> dict:
    """Blocking. Run via run_in_executor. Never raises."""
    try:
        if client_factory is not None:
            client = client_factory()
        else:
            from agent.config import config

            client = anthropic.Anthropic(**config.anthropic_client_kwargs(timeout=_CANARY_TIMEOUT_SEC))
        t0 = time.monotonic()
        resp = client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": "ping"}],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        served = str(getattr(resp, "model", "") or "") or None
    except Exception as e:
        # Exception type only: messages may carry request context.
        logger.debug(f"oneprovider canary ping failed for {model}: {type(e).__name__}")
        return {"requested": model, "served": None, "latency_ms": None, "family_changed": False, "ok": False}

    fam_req = _family(model)
    fam_srv = _family(served)
    family_changed = fam_srv is not None and fam_req is not None and fam_req != fam_srv
    return {
        "requested": model,
        "served": served,
        "latency_ms": latency_ms,
        "family_changed": family_changed,
        "ok": True,
    }


async def run_canary(client_factory: Callable[[], Any] | None = None) -> dict:
    """Ping default_model and fast_model (distinct). Never raises."""
    try:
        from agent.config import config

        models: list[str] = []
        for m in (config.default_model, config.fast_model):
            if m and m not in models:
                models.append(m)
        loop = asyncio.get_running_loop()
        results = []
        for model in models:
            r = await loop.run_in_executor(None, _ping_model_sync, model, client_factory)
            results.append(r)
            if not r["ok"]:
                continue
            if r["family_changed"]:
                logger.warning(
                    f"oneprovider canary: served family changed requested={r['requested']} served={r['served']}"
                )
            elif r["served"] != r["requested"]:
                logger.info(
                    f"oneprovider canary: served version differs requested={r['requested']} served={r['served']}"
                )
        return {
            "results": results,
            "family_changed": any(r["family_changed"] for r in results),
            "ok": bool(results) and all(r["ok"] for r in results),
        }
    except Exception as e:
        logger.debug(f"oneprovider canary failed: {type(e).__name__}")
        return {"results": [], "family_changed": False, "ok": False}


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _should_alert(level: str, prior_level: str | None, last_alert: dict, now_ts: float) -> bool:
    if level == "ok":
        return False
    if now_ts - _to_float(last_alert.get(level)) >= _ALERT_DEBOUNCE_SEC:
        return True
    # Crit re-alerts immediately when the level changes into crit.
    return level == "crit" and prior_level != "crit"


def _format_alert(ev: dict, balance: dict, canary: dict, level: str,
                  level_alert: bool, canary_changed: bool) -> str:
    lines = []
    if level_alert and level != "ok":
        if "balance_usd" in balance:
            bal_txt = f"balance ${balance['balance_usd']:.2f}"
        else:
            bal_txt = "balance unknown"
        days = ev.get("days_to_expiry")
        exp_txt = f"expires in {days:.1f}d" if days is not None else "expiry unknown"
        burn = ev.get("burn_per_day")
        burn_txt = f"burn ${burn:.2f}/day" if burn is not None else "burn n/a"
        reasons = ", ".join(ev.get("reasons") or []) or "none"
        lines.append(f"CLIO gateway {level.upper()}: {bal_txt}, {exp_txt}, {burn_txt}; {reasons}")
    if canary_changed:
        for r in canary.get("results", []):
            if r.get("family_changed"):
                lines.append(f"Canary: served family changed {r['requested']} -> {r['served']}")
    return "\n".join(lines)


async def _monitor_once(context: Any) -> None:
    loop = asyncio.get_running_loop()
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()

    balance = await fetch_balance()
    canary = await run_canary()
    prev = await loop.run_in_executor(None, read_state_sync)
    if not isinstance(prev, dict):
        prev = {}

    ev = evaluate(balance, now, prev)
    level = ev["level"]
    prior_level = prev.get("level")
    last_alert = prev.get("last_alert") if isinstance(prev.get("last_alert"), dict) else {}
    last_alert = dict(last_alert)

    alert_text_needed = False
    level_alert = _should_alert(level, prior_level, last_alert, now_ts)
    if level_alert:
        alert_text_needed = True
        last_alert[level] = now_ts
    canary_alert = False
    if bool(canary.get("family_changed")) and now_ts - _to_float(last_alert.get("canary")) >= _ALERT_DEBOUNCE_SEC:
        alert_text_needed = True
        canary_alert = True
        last_alert["canary"] = now_ts

    ok_sample = "error" not in balance
    new_state = dict(prev)
    samples = (
        _append_sample(prev.get("samples"), now_ts, balance["balance_usd"])
        if ok_sample else _valid_samples(prev.get("samples"))
    )
    burn_24h, burn_rth = compute_burns(samples, now_ts)
    if ok_sample:
        new_state.update({
            "balance_usd": balance["balance_usd"],
            "expires_at": balance.get("expires_at"),
            "is_active": balance.get("is_active"),
            "sample_ts": now_ts,
        })
        logger.info(
            "gateway balance $%.2f burn24h=%s burn_rth=%s",
            balance["balance_usd"], _fmt_burn(burn_24h), _fmt_burn(burn_rth),
        )

    # R-10 latency alert: Opus per-model p95 vs 0.8 x decider timeout, 12 h debounce.
    latency_text = None
    breach = opus_p95_breach(_gateway_snapshot(), _decision_timeout_s())
    if breach is not None and now_ts - _to_float(last_alert.get("latency")) >= _ALERT_DEBOUNCE_SEC:
        last_alert["latency"] = now_ts
        alert_text_needed = True
        latency_text = (
            f"CLIO gateway LATENCY: {breach['model']} p95 {breach['p95_ms']}ms "
            f"over {breach['threshold_ms']}ms (0.8 x decider timeout)"
        )

    new_state.update({
        "level": level,
        "reasons": ev["reasons"],
        "burn_per_day": ev["burn_per_day"],
        "burn_24h": burn_24h,
        "burn_rth": burn_rth,
        "samples": samples,
        "last_error": balance.get("error"),
        "canary": canary,
        "last_alert": last_alert,
        "checked_at": now.isoformat(),
    })
    await loop.run_in_executor(None, write_state_sync, new_state)

    if alert_text_needed:
        text = _format_alert(ev, balance, canary, level, level_alert, canary_alert)
        if latency_text:
            text = "\n".join(part for part in (text, latency_text) if part)
        if text:
            try:
                from agent.integrations.scheduled_tasks import _get_chat_ids, _safe_send

                bot = getattr(context, "bot", None)
                for cid in _get_chat_ids():
                    await _safe_send(bot, cid, text=text)
            except Exception as e:
                logger.debug(f"oneprovider alert send failed: {type(e).__name__}")

    if level == "ok":
        logger.debug("oneprovider monitor: ok")
    else:
        logger.warning(f"oneprovider monitor: level={level} reasons={ev['reasons']}")


async def oneprovider_monitor_job(context) -> None:
    """PTB JobQueue callback. Never raises."""
    try:
        if os.getenv("CLIO_GATEWAY_MONITOR_ENABLED", "1") == "0":
            return
        from agent.config import config

        if not config.anthropic_gateway:
            return
        await _monitor_once(context)
    except Exception as e:
        logger.debug(f"oneprovider monitor raised: {type(e).__name__}")
