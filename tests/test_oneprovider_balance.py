"""
OP-10: OneProvider balance/expiry monitor + served-model canary
(agent/integrations/oneprovider_balance.py).

No network: httpx.MockTransport for the balance fetch, fake SDK clients for the
canary, fake Telegram send/chat-id functions. The state file is always redirected
to tmp_path. Keys are fake.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from agent.config import config
from agent.integrations import oneprovider_balance as ob
from agent.integrations import scheduled_tasks as st

_FAKE_KEY = "sk-" + "a" * 64
_BASE = "https://gw.invalid"
_NOW = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(ob, "_STATE_PATH", str(tmp_path / "oneprovider_balance_state.json"))
    monkeypatch.setattr(config, "anthropic_gateway", True)
    monkeypatch.setattr(config, "anthropic_api_key", _FAKE_KEY)
    monkeypatch.setattr(config, "anthropic_base_url", _BASE)
    monkeypatch.setattr(config, "default_model", "claude-opus-5-5")
    monkeypatch.setattr(config, "fast_model", "claude-sonnet-5-5")
    monkeypatch.delenv("CLIO_GATEWAY_MONITOR_ENABLED", raising=False)
    monkeypatch.delenv("CLIO_GATEWAY_BALANCE_WARN_USD", raising=False)
    monkeypatch.delenv("CLIO_GATEWAY_BALANCE_CRIT_USD", raising=False)


def _balance_body(**over):
    body = {
        "object": "balance",
        "balance_usd": 349.9121,
        "expires_at": "2027-01-09T17:43:23.283795",
        "is_active": True,
        "last_synced_at": "2026-10-11T18:41:59.842191",
    }
    body.update(over)
    return body


def _transport(status=200, body=None, raw=None, exc=None):
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers.get("authorization")
        seen["x_api_key"] = request.headers.get("x-api-key")
        seen["url"] = str(request.url)
        if exc is not None:
            raise exc
        if raw is not None:
            return httpx.Response(status, content=raw, request=request)
        return httpx.Response(status, json=body, request=request)

    return httpx.MockTransport(handler), seen


# ---------------------------------------------------------------------------
# fetch_balance
# ---------------------------------------------------------------------------

def test_fetch_200_parses_and_uses_bearer_only():
    transport, seen = _transport(200, _balance_body())
    out = asyncio.run(ob.fetch_balance(transport=transport))
    assert out == {
        "balance_usd": 349.9121,
        "expires_at": "2027-01-09T17:43:23.283795",
        "is_active": True,
        "last_synced_at": "2026-10-11T18:41:59.842191",
    }
    assert seen["url"] == _BASE + "/v1/dashboard/balance"
    assert seen["auth"] == "Bearer " + _FAKE_KEY
    assert seen["x_api_key"] is None


def test_fetch_401_invalid_key():
    transport, _ = _transport(401, {"error": {"type": "authentication_error"}})
    assert asyncio.run(ob.fetch_balance(transport=transport)) == {"error": "invalid_key"}


def test_fetch_429_rate_limited():
    transport, _ = _transport(429, {"error": {"type": "rate_limit_error"}})
    assert asyncio.run(ob.fetch_balance(transport=transport)) == {"error": "rate_limited"}


def test_fetch_500_unavailable():
    transport, _ = _transport(500, {"error": {"type": "server_error"}})
    assert asyncio.run(ob.fetch_balance(transport=transport)) == {"error": "unavailable"}


def test_fetch_timeout_returns_unavailable_not_raise():
    req = httpx.Request("GET", _BASE + "/v1/dashboard/balance")
    transport, _ = _transport(exc=httpx.ReadTimeout("slow", request=req))
    assert asyncio.run(ob.fetch_balance(transport=transport)) == {"error": "unavailable"}


def test_fetch_malformed_json_is_malformed():
    transport, _ = _transport(200, raw=b"<html>not json")
    assert asyncio.run(ob.fetch_balance(transport=transport)) == {"error": "malformed"}


@pytest.mark.parametrize("drop", ["balance_usd", "expires_at", "is_active", "last_synced_at"])
def test_fetch_missing_field_is_malformed(drop):
    body = _balance_body()
    del body[drop]
    transport, _ = _transport(200, body)
    assert asyncio.run(ob.fetch_balance(transport=transport)) == {"error": "malformed"}


def test_fetch_wrong_types_are_malformed():
    for bad in (
        _balance_body(balance_usd="349"),
        _balance_body(balance_usd=True),
        _balance_body(is_active="yes"),
        _balance_body(expires_at=12),
    ):
        transport, _ = _transport(200, bad)
        assert asyncio.run(ob.fetch_balance(transport=transport)) == {"error": "malformed"}


def test_fetch_gateway_disabled(monkeypatch):
    monkeypatch.setattr(config, "anthropic_gateway", False)
    assert asyncio.run(ob.fetch_balance(transport=_transport(200, _balance_body())[0])) == {
        "error": "gateway_disabled"
    }


def test_fetch_no_key_is_gateway_disabled(monkeypatch):
    monkeypatch.setattr(config, "anthropic_api_key", "")
    assert asyncio.run(ob.fetch_balance(transport=_transport(200, _balance_body())[0])) == {
        "error": "gateway_disabled"
    }


# ---------------------------------------------------------------------------
# evaluate (pure)
# ---------------------------------------------------------------------------

def _sample(balance=349.9, days=89.0, active=True, **kw):
    exp = (_NOW + timedelta(days=days)).isoformat() if days is not None else None
    s = {"balance_usd": balance, "expires_at": exp, "is_active": active, "last_synced_at": None}
    s.update(kw)
    return s


def test_evaluate_healthy_is_ok():
    ev = ob.evaluate(_sample(), _NOW, {})
    assert ev["level"] == "ok"
    assert ev["reasons"] == []
    assert ev["days_to_expiry"] == pytest.approx(89.0)
    assert ev["burn_per_day"] is None


def test_balance_warn_boundary_75():
    assert ob.evaluate(_sample(balance=75.0), _NOW, {})["level"] == "ok"
    ev = ob.evaluate(_sample(balance=74.99), _NOW, {})
    assert ev["level"] == "warn" and "balance_low" in ev["reasons"]


def test_balance_crit_boundary_25():
    ev = ob.evaluate(_sample(balance=25.0), _NOW, {})
    assert ev["level"] == "warn" and "balance_low" in ev["reasons"]
    ev = ob.evaluate(_sample(balance=24.99), _NOW, {})
    assert ev["level"] == "crit" and "balance_critical" in ev["reasons"]


def test_expiry_warn_boundary_14d():
    assert ob.evaluate(_sample(days=14.0), _NOW, {})["level"] == "ok"
    ev = ob.evaluate(_sample(days=13.9), _NOW, {})
    assert ev["level"] == "warn" and "expiry_soon" in ev["reasons"]


def test_expiry_crit_boundary_3d():
    ev = ob.evaluate(_sample(days=3.0), _NOW, {})
    assert ev["level"] == "warn" and "expiry_soon" in ev["reasons"]
    ev = ob.evaluate(_sample(days=2.9), _NOW, {})
    assert ev["level"] == "crit" and "expiry_critical" in ev["reasons"]


def test_inactive_key_is_crit():
    ev = ob.evaluate(_sample(active=False), _NOW, {})
    assert ev["level"] == "crit" and "inactive" in ev["reasons"]


def test_invalid_key_is_crit():
    ev = ob.evaluate({"error": "invalid_key"}, _NOW, {})
    assert ev["level"] == "crit" and ev["reasons"] == ["invalid_key"]


@pytest.mark.parametrize("err", ["rate_limited", "unavailable", "malformed"])
def test_sample_errors_are_warn_no_sample(err):
    ev = ob.evaluate({"error": err}, _NOW, {})
    assert ev["level"] == "warn" and ev["reasons"] == ["no_sample"]


def test_gateway_disabled_is_ok():
    assert ob.evaluate({"error": "gateway_disabled"}, _NOW, {})["level"] == "ok"


def test_burn_counts_spend():
    prev = {"balance_usd": 350.0, "sample_ts": (_NOW - timedelta(days=2)).timestamp()}
    ev = ob.evaluate(_sample(balance=340.0), _NOW, prev)
    assert ev["burn_per_day"] == pytest.approx(5.0)


def test_burn_ignores_top_up():
    prev = {"balance_usd": 300.0, "sample_ts": (_NOW - timedelta(days=2)).timestamp()}
    ev = ob.evaluate(_sample(balance=349.9), _NOW, prev)
    assert ev["burn_per_day"] is None


def test_burn_needs_one_hour_elapsed():
    prev = {"balance_usd": 350.0, "sample_ts": (_NOW - timedelta(minutes=30)).timestamp()}
    assert ob.evaluate(_sample(balance=349.0), _NOW, prev)["burn_per_day"] is None


def test_bad_env_thresholds_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("CLIO_GATEWAY_BALANCE_WARN_USD", "lots")
    monkeypatch.setenv("CLIO_GATEWAY_BALANCE_CRIT_USD", "nan")
    assert ob.evaluate(_sample(balance=80.0), _NOW, {})["level"] == "ok"
    ev = ob.evaluate(_sample(balance=74.0), _NOW, {})
    assert ev["level"] == "warn"  # default warn 75 still applies
    assert ob.evaluate(_sample(balance=24.0), _NOW, {})["level"] == "crit"  # default crit 25


def test_env_thresholds_are_honoured(monkeypatch):
    monkeypatch.setenv("CLIO_GATEWAY_BALANCE_WARN_USD", "100")
    assert ob.evaluate(_sample(balance=90.0), _NOW, {})["level"] == "warn"


# ---------------------------------------------------------------------------
# state file
# ---------------------------------------------------------------------------

def test_state_round_trip_and_no_tmp_left(tmp_path):
    state = {"balance_usd": 349.9, "level": "ok", "last_alert": {"warn": 1.0}}
    ob.write_state_sync(state)
    assert ob.read_state_sync() == state
    assert not (tmp_path / "oneprovider_balance_state.json.tmp").exists()


def test_missing_state_is_empty(tmp_path):
    assert ob.read_state_sync() == {}


def test_corrupt_state_is_empty(tmp_path):
    (tmp_path / "oneprovider_balance_state.json").write_text("{not json")
    assert ob.read_state_sync() == {}


def test_state_write_replaces_existing(tmp_path):
    ob.write_state_sync({"v": 1})
    ob.write_state_sync({"v": 2})
    assert json.loads((tmp_path / "oneprovider_balance_state.json").read_text()) == {"v": 2}
    assert not (tmp_path / "oneprovider_balance_state.json.tmp").exists()


# ---------------------------------------------------------------------------
# canary
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, model):
        self.model = model


class _Msgs:
    def __init__(self, served_map=None, exc=None):
        self.served_map = served_map or {}
        self.exc = exc
        self.calls = []

    def create(self, model, max_tokens, messages):
        self.calls.append((model, max_tokens))
        if self.exc is not None:
            raise self.exc
        return _Resp(self.served_map.get(model, model))


class _Client:
    def __init__(self, msgs):
        self.messages = msgs


def test_canary_version_only_diff_is_info_no_warning(caplog):
    msgs = _Msgs({"claude-opus-5-5": "claude-opus-5", "claude-sonnet-5-5": "claude-sonnet-5-5"})
    caplog.set_level(logging.INFO, logger="openclaw.oneprovider_balance")
    out = asyncio.run(ob.run_canary(client_factory=lambda: _Client(msgs)))
    assert out["family_changed"] is False
    assert out["ok"] is True
    assert [c[0] for c in msgs.calls] == ["claude-opus-5-5", "claude-sonnet-5-5"]
    assert all(c[1] == 8 for c in msgs.calls)
    levels = [r.levelno for r in caplog.records if r.name == "openclaw.oneprovider_balance"]
    assert logging.WARNING not in levels
    assert logging.INFO in levels


def test_canary_family_change_is_warning_and_flagged(caplog):
    msgs = _Msgs({"claude-opus-5-5": "claude-sonnet-4-6"})
    caplog.set_level(logging.INFO, logger="openclaw.oneprovider_balance")
    out = asyncio.run(ob.run_canary(client_factory=lambda: _Client(msgs)))
    assert out["family_changed"] is True
    flagged = [r for r in out["results"] if r["requested"] == "claude-opus-5-5"][0]
    assert flagged["family_changed"] is True
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_canary_exception_is_ok_false_no_raise():
    msgs = _Msgs(exc=RuntimeError("boom " + _FAKE_KEY))
    out = asyncio.run(ob.run_canary(client_factory=lambda: _Client(msgs)))
    assert out["ok"] is False
    assert all(r["ok"] is False for r in out["results"])
    assert out["family_changed"] is False


def test_canary_models_are_distinct(monkeypatch):
    monkeypatch.setattr(config, "fast_model", "claude-opus-5-5")
    msgs = _Msgs()
    asyncio.run(ob.run_canary(client_factory=lambda: _Client(msgs)))
    assert [c[0] for c in msgs.calls] == ["claude-opus-5-5"]


# ---------------------------------------------------------------------------
# monitor job
# ---------------------------------------------------------------------------

class _Ctx:
    class bot:  # noqa: N801 - mimic PTB context.bot
        pass


@pytest.fixture
def sends(monkeypatch):
    sent = []

    async def fake_send(bot, chat_id, **kwargs):
        sent.append((chat_id, kwargs.get("text")))
        return True

    monkeypatch.setattr(st, "_get_chat_ids", lambda: ["chat1"])
    monkeypatch.setattr(st, "_safe_send", fake_send)
    return sent


def _patch_fetch_and_canary(monkeypatch, balance, canary=None, calls=None):
    async def fake_fetch(*a, **k):
        if calls is not None:
            calls.append("fetch")
        return balance

    async def fake_canary(*a, **k):
        return canary or {"results": [], "family_changed": False, "ok": True}

    monkeypatch.setattr(ob, "fetch_balance", fake_fetch)
    monkeypatch.setattr(ob, "run_canary", fake_canary)


def test_job_alerts_once_then_debounces(monkeypatch, sends):
    _patch_fetch_and_canary(monkeypatch, {"error": "invalid_key"})
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert len(sends) == 1
    assert "CLIO gateway CRIT" in sends[0][1]
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert len(sends) == 1  # same level within 12h: debounced
    state = ob.read_state_sync()
    assert state["level"] == "crit"
    assert "crit" in state["last_alert"]


def test_job_crit_realerts_on_level_change(monkeypatch, sends):
    _patch_fetch_and_canary(monkeypatch, _sample(balance=50.0))  # warn
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert len(sends) == 1 and "WARN" in sends[0][1]
    _patch_fetch_and_canary(monkeypatch, _sample(balance=10.0))  # crit, within debounce
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert len(sends) == 2 and "CRIT" in sends[1][1]


def test_job_ok_sends_nothing(monkeypatch, sends):
    _patch_fetch_and_canary(monkeypatch, _sample())
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert sends == []


def test_job_canary_family_change_alerts(monkeypatch, sends):
    canary = {
        "results": [{"requested": "claude-opus-5-5", "served": "claude-sonnet-4-6",
                     "latency_ms": 5, "family_changed": True, "ok": True}],
        "family_changed": True,
        "ok": True,
    }
    _patch_fetch_and_canary(monkeypatch, _sample(), canary=canary)
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert len(sends) == 1
    assert "Canary" in sends[0][1]
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert len(sends) == 1  # canary debounced


def test_job_disabled_flag_skips_fetch(monkeypatch, sends):
    monkeypatch.setenv("CLIO_GATEWAY_MONITOR_ENABLED", "0")
    calls = []
    _patch_fetch_and_canary(monkeypatch, {"error": "invalid_key"}, calls=calls)
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert calls == []
    assert sends == []


def test_job_gateway_off_skips_fetch(monkeypatch, sends):
    monkeypatch.setattr(config, "anthropic_gateway", False)
    calls = []
    _patch_fetch_and_canary(monkeypatch, {"error": "invalid_key"}, calls=calls)
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert calls == []


def test_job_never_raises_when_fetch_raises(monkeypatch, sends):
    async def boom(*a, **k):
        raise RuntimeError("fetch exploded")

    monkeypatch.setattr(ob, "fetch_balance", boom)
    monkeypatch.setattr(ob, "run_canary", lambda *a, **k: _none_coro())
    # Must not raise into the JobQueue.
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert sends == []


async def _none_coro():
    return {"results": [], "family_changed": False, "ok": True}


def test_job_never_raises_when_send_raises(monkeypatch):
    async def bad_send(bot, chat_id, **kwargs):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(st, "_get_chat_ids", lambda: ["chat1"])
    monkeypatch.setattr(st, "_safe_send", bad_send)
    _patch_fetch_and_canary(monkeypatch, {"error": "invalid_key"})
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))  # no raise
    assert ob.read_state_sync()["level"] == "crit"


def test_job_debounce_reads_corrupt_last_alert_safely(monkeypatch, sends):
    ob.write_state_sync({"level": "crit", "last_alert": "garbage"})
    _patch_fetch_and_canary(monkeypatch, {"error": "invalid_key"})
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    # Prior level was crit (no level-change alert) and the debounce stamp is unreadable
    # (treated as 0) -> alert fires once; the call must not raise.
    assert len(sends) == 1


# ---------------------------------------------------------------------------
# registration (scheduled_tasks)
# ---------------------------------------------------------------------------

class _FakeJQ:
    def __init__(self):
        self.repeating = {}
        self.once = []

    def run_repeating(self, func, interval=None, first=None, name=None, **kw):
        self.repeating[name] = {"func": func, "interval": interval, "first": first}

    def run_once(self, func, when=None, name=None, **kw):
        self.once.append(name)


class _FakeApp:
    def __init__(self):
        self.job_queue = _FakeJQ()


def _register(monkeypatch, tmp_path, gateway_on: bool):
    monkeypatch.setenv("CLIO_DPOF_DATA_DIR", str(tmp_path / "dpof"))
    monkeypatch.delenv("CLIO_HYDRINIA_BRIDGE_ENABLED", raising=False)
    monkeypatch.setattr(config, "anthropic_gateway", gateway_on)
    monkeypatch.setattr(st, "_get_chat_ids", lambda: ["test_chat"])
    app = _FakeApp()
    st.register_scheduled_jobs(app)
    return app.job_queue.repeating


def test_monitor_registration_skipped_when_gateway_off(monkeypatch, tmp_path):
    repeating = _register(monkeypatch, tmp_path, gateway_on=False)
    assert "oneprovider_monitor" not in repeating


def test_monitor_registration_when_gateway_on(monkeypatch, tmp_path):
    repeating = _register(monkeypatch, tmp_path, gateway_on=True)
    assert "oneprovider_monitor" in repeating
    job = repeating["oneprovider_monitor"]
    assert job["interval"] == 21600
    assert job["first"] == 240
    assert job["func"] is ob.oneprovider_monitor_job


# ---------------------------------------------------------------------------
# R-10: balance sample ring, burn windows, per-sample INFO, Opus latency alert
# ---------------------------------------------------------------------------

from zoneinfo import ZoneInfo  # noqa: E402

_ET_TZ = ZoneInfo("America/New_York")
_OLD_STATE = {
    "balance_usd": 320.0,
    "expires_at": "2026-12-31T00:00:00+00:00",
    "is_active": True,
    "sample_ts": 1_781_000_000.0,
    "level": "ok",
    "reasons": [],
    "burn_per_day": None,
    "last_error": None,
    "canary": {"results": [], "family_changed": False, "ok": True},
    "last_alert": {},
    "checked_at": "2026-10-11T12:00:00+00:00",
}


def test_ring_is_bounded_to_48_newest():
    samples = []
    for i in range(60):
        samples = ob._append_sample(samples, 1000.0 + i, 100.0)
    assert len(samples) == ob._SAMPLE_RING == 48
    assert samples[-1]["ts"] == 1059.0 and samples[0]["ts"] == 1012.0


def test_ring_drops_malformed_entries():
    raw = [{"ts": 1, "balance": 2}, "x", {"ts": True, "balance": 1}, {"ts": 3}]
    assert ob._valid_samples(raw) == [{"ts": 1.0, "balance": 2.0}]
    assert ob._valid_samples("not-a-list") == []


def test_burn_24h_and_rth_windows():
    base = datetime(2026, 10, 12, 9, 0, tzinfo=_ET_TZ).timestamp()
    samples = [
        {"ts": base, "balance": 100.0},                  # 09:00 ET (before RTH open)
        {"ts": base + 3 * 3600, "balance": 97.0},        # 12:00 ET
        {"ts": base + 6 * 3600, "balance": 96.0},        # 15:00 ET
    ]
    b24, brth = ob.compute_burns(samples, base + 7 * 3600)
    # 24h: spent 4.0 over 6 h -> 16.0 per day
    assert b24 == pytest.approx(16.0)
    # RTH: only the 12:00 -> 15:00 pair has both ends inside 09:30-16:00 ET: 1.0 over 3 h -> 8.0/day
    assert brth == pytest.approx(8.0)


def test_top_ups_are_not_spend_and_rth_none_without_pairs():
    base = datetime(2026, 10, 12, 20, 0, tzinfo=_ET_TZ).timestamp()  # evening: no RTH pairs
    samples = [{"ts": base, "balance": 50.0}, {"ts": base + 6 * 3600, "balance": 80.0}]
    b24, brth = ob.compute_burns(samples, base + 6 * 3600)
    assert b24 == pytest.approx(0.0)  # a top-up is never counted as spend (6 h window, zero spend)
    assert brth is None  # evening samples are outside RTH: no RTH pairs


def test_old_state_format_still_reads_and_evaluates(tmp_path, monkeypatch):
    path = tmp_path / "oneprovider_balance_state.json"
    path.write_text(json.dumps(_OLD_STATE))
    monkeypatch.setattr(ob, "_STATE_PATH", str(path))
    prev = ob.read_state_sync()
    assert prev["balance_usd"] == 320.0 and prev["sample_ts"] == _OLD_STATE["sample_ts"]
    assert "samples" not in prev  # old file has no ring: readers must tolerate its absence
    now = datetime.fromtimestamp(_OLD_STATE["sample_ts"] + 86400, tz=timezone.utc)
    ev = ob.evaluate(_sample(balance=300.0), now, prev)
    assert ev["level"] == "ok"
    assert ev["burn_per_day"] == pytest.approx(20.0)


def test_job_writes_ring_and_logs_per_sample(monkeypatch, sends, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="openclaw.oneprovider_balance")
    _patch_fetch_and_canary(monkeypatch, _sample(balance=340.0))
    for _ in range(3):
        asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    state = ob.read_state_sync()
    assert len(state["samples"]) == 3
    assert state["burn_24h"] is None and state["burn_rth"] is None  # sub-hour pairs: no burn
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gateway balance $")]
    assert len(lines) == 3
    assert lines[-1] == "gateway balance $340.00 burn24h=n/a burn_rth=n/a"


def _lat_snapshot(model, n, p95):
    return {"latency_ms": {model: {"n": n, "p50": 1, "p95": p95}}}


def test_opus_p95_breach_is_strict_and_opus_only():
    assert ob.opus_p95_breach(_lat_snapshot("claude-opus-5-5", 25, 48000), 60.0) is None
    assert ob.opus_p95_breach(_lat_snapshot("claude-opus-5-5", 25, 48001), 60.0)["p95_ms"] == 48001
    assert ob.opus_p95_breach(_lat_snapshot("claude-sonnet-5-5", 25, 90000), 60.0) is None
    assert ob.opus_p95_breach(_lat_snapshot("claude-opus-5-5", 5, 90000), 60.0) is None
    assert ob.opus_p95_breach(None, 60.0) is None


def test_latency_alert_sent_plain_text_and_debounced(monkeypatch, sends, tmp_path):
    monkeypatch.setattr(ob, "_gateway_snapshot", lambda: _lat_snapshot("claude-opus-5-5", 25, 50000))
    monkeypatch.setattr(ob, "_decision_timeout_s", lambda: 60.0)
    _patch_fetch_and_canary(monkeypatch, _sample())  # healthy balance: only the latency text goes out
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))  # within 12 h: debounced
    latency = [t for _, t in sends if t and "LATENCY" in t]
    assert latency == ["CLIO gateway LATENCY: claude-opus-5-5 p95 50000ms over 48000ms (0.8 x decider timeout)"]


def test_no_latency_alert_below_threshold(monkeypatch, sends):
    monkeypatch.setattr(ob, "_gateway_snapshot", lambda: _lat_snapshot("claude-opus-5-5", 25, 10000))
    monkeypatch.setattr(ob, "_decision_timeout_s", lambda: 60.0)
    _patch_fetch_and_canary(monkeypatch, _sample())
    asyncio.run(ob.oneprovider_monitor_job(_Ctx()))
    assert not [t for _, t in sends if t and "LATENCY" in t]


def test_gateway_stats_tick_registered_hourly_in_gateway_mode(monkeypatch, tmp_path):
    repeating = _register(monkeypatch, tmp_path, gateway_on=True)
    assert repeating["gateway_stats_tick"]["interval"] == 3600
    assert repeating["gateway_stats_tick"]["first"] == 3600


def test_gateway_stats_tick_not_registered_when_gateway_off(monkeypatch, tmp_path):
    repeating = _register(monkeypatch, tmp_path, gateway_on=False)
    assert "gateway_stats_tick" not in repeating


def test_state_io_runs_off_the_event_loop(monkeypatch):
    """Flock/file I/O must go through run_in_executor, never run on the loop thread."""
    import threading

    seen = []

    def rec_read():
        seen.append(("read", threading.get_ident()))
        return {}

    def rec_write(state):
        seen.append(("write", threading.get_ident()))

    monkeypatch.setattr(ob, "read_state_sync", rec_read)
    monkeypatch.setattr(ob, "write_state_sync", rec_write)
    _patch_fetch_and_canary(monkeypatch, _sample())

    async def _run():
        loop_ident = threading.get_ident()
        await ob._monitor_once(_Ctx())
        return loop_ident

    loop_ident = asyncio.run(_run())
    kinds = [k for k, _ in seen]
    assert kinds == ["read", "write"]
    assert all(ident != loop_ident for _, ident in seen)


# ---------------------------------------------------------------------------
# alert debounce: level-change re-alert semantics
# ---------------------------------------------------------------------------

def test_crit_realerts_on_level_change_even_within_debounce():
    now = _NOW.timestamp()
    # crit was alerted an hour ago, then warn, now crit again -> re-alert now.
    last = {"crit": now - 3600, "warn": now - 1800}
    assert ob._should_alert("crit", "warn", last, now) is True


def test_crit_not_realerted_when_prior_was_crit_within_debounce():
    now = _NOW.timestamp()
    assert ob._should_alert("crit", "crit", {"crit": now - 3600}, now) is False


def test_warn_not_realerted_within_debounce():
    now = _NOW.timestamp()
    assert ob._should_alert("warn", "warn", {"warn": now - 3600}, now) is False


def test_ok_never_alerts():
    now = _NOW.timestamp()
    assert ob._should_alert("ok", "crit", {}, now) is False


def test_warn_realerts_after_debounce_window():
    now = _NOW.timestamp()
    assert ob._should_alert("warn", "warn", {"warn": now - 13 * 3600}, now) is True


# ---------------------------------------------------------------------------
# served-model family regex (haiku included)
# ---------------------------------------------------------------------------

def _canary_one(requested, served):
    msgs = _Msgs({requested: served})
    return ob._ping_model_sync(requested, lambda: _Client(msgs))


def test_family_haiku_same_is_not_changed():
    r = _canary_one("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001")
    assert r["ok"] is True and r["family_changed"] is False


def test_family_opus_to_haiku_is_changed():
    r = _canary_one("claude-opus-5-5", "claude-haiku-4-5-20251001")
    assert r["family_changed"] is True


def test_family_opus_version_only_is_not_changed():
    r = _canary_one("claude-opus-5-5", "claude-opus-5")
    assert r["family_changed"] is False


class _CallJQ:
    def __init__(self, raises=None):
        self.calls = []
        self._raises = raises

    def run_repeating(self, func, interval=None, first=None, name=None, **kw):
        if self._raises is not None:
            raise self._raises
        self.calls.append({"func": func, "interval": interval, "first": first, "name": name})


def test_helper_gateway_off_no_registration(monkeypatch):
    monkeypatch.setattr(config, "anthropic_gateway", False)
    jq = _CallJQ()
    assert st._register_oneprovider_monitor(jq) is False
    assert jq.calls == []


def test_helper_gateway_on_registers_once(monkeypatch):
    monkeypatch.setattr(config, "anthropic_gateway", True)
    jq = _CallJQ()
    assert st._register_oneprovider_monitor(jq) is True
    assert len(jq.calls) == 1
    call = jq.calls[0]
    assert call["interval"] == 21600
    assert call["first"] == 240
    assert call["name"] == "oneprovider_monitor"
    assert call["func"] is ob.oneprovider_monitor_job


def test_helper_swallows_registration_error(monkeypatch):
    monkeypatch.setattr(config, "anthropic_gateway", True)
    jq = _CallJQ(raises=RuntimeError("jobqueue down"))
    assert st._register_oneprovider_monitor(jq) is False  # no exception propagates
