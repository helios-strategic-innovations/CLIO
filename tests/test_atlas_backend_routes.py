import asyncio
import importlib.util
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class EnumLike:
    def __init__(self, value):
        self.value = value


def _load_route(monkeypatch, tmp_path, name):
    fastapi = types.ModuleType("fastapi")

    class APIRouter:
        def __init__(self, prefix="", tags=None):
            self.prefix = prefix
            self.tags = tags or []
            self.routes = []

        def _route(self, path):
            def decorator(function):
                self.routes.append(SimpleNamespace(path=f"{self.prefix}{path}"))
                return function
            return decorator

        get = _route
        post = _route

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi.APIRouter = APIRouter
    fastapi.Depends = lambda _dependency: None
    fastapi.HTTPException = HTTPException
    fastapi.Query = lambda default=None, **_kwargs: default

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    psutil = types.ModuleType("psutil")

    auth = types.ModuleType("mission_control.backend.auth")
    auth.require_auth = lambda: None

    config = types.ModuleType("mission_control.backend.config")
    config.mc_config = SimpleNamespace(
        openclaw_root=str(tmp_path / "openclaw"),
        tradingbot_root=str(tmp_path / "tradingbot"),
    )

    deps = types.ModuleType("mission_control.backend.deps")

    async def run_in_executor(function, *args):
        return function(*args)

    deps.run_in_executor = run_in_executor
    deps.get_trading_client = lambda: None
    deps.read_json = lambda *_args, **_kwargs: None
    deps.read_text = lambda *_args, **_kwargs: ""

    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    monkeypatch.setitem(sys.modules, "mission_control.backend.auth", auth)
    monkeypatch.setitem(sys.modules, "mission_control.backend.config", config)
    monkeypatch.setitem(sys.modules, "mission_control.backend.deps", deps)

    module_name = f"_atlas_backend_test_{name}_{id(tmp_path)}"
    path = ROOT / "mission_control" / "backend" / "routes" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _position(symbol, side, market_value=1000):
    return SimpleNamespace(
        symbol=symbol,
        side=side,
        qty="2",
        avg_entry_price="100",
        current_price="105",
        market_value=str(market_value),
        unrealized_pl="10",
        unrealized_plpc="0.01",
        asset_class="us_equity",
    )


def _account():
    return SimpleNamespace(
        account_number="PA-123",
        equity="10000",
        cash="2000",
        buying_power="4000",
        portfolio_value="10000",
        last_equity="9900",
        status=EnumLike("ACTIVE"),
        pattern_day_trader=False,
        trading_blocked=False,
    )


def test_agent_performance_preserves_compatibility_and_adds_model_diagnostics(monkeypatch, tmp_path):
    # Hydrinia's 12-agent vote history is retired (S/O 2026-09-24); this route
    # gates to an empty payload unless the bridge is explicitly re-enabled —
    # exercise the legacy analysis pipeline with it back on.
    monkeypatch.setenv("CLIO_HYDRINIA_BRIDGE_ENABLED", "1")
    agents = _load_route(monkeypatch, tmp_path, "agents")
    state_dir = Path(agents._STATE_DIR)
    state_dir.mkdir(parents=True)
    for filename in ("agent_metrics.json", "vote_history.json", "prediction_log.json"):
        (state_dir / filename).write_text("{}", encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()
    payloads = {
        "agent_metrics.json": {
            "Aidan Tan": {
                "total_votes": 2,
                "trades": [
                    {"vote": "BUY", "trade_direction": "BUY"},
                    {"vote": "SELL", "trade_direction": "BUY"},
                ],
            }
        },
        "vote_history.json": [
            {"agent_name": "Aidan Tan", "vote": "BUY", "platform": "deepseek", "timestamp": now},
            {"agent_name": "Aidan Tan", "vote": "BUY", "platform": "deepseek", "timestamp": now},
            {
                "agent_name": "Aidan Tan",
                "vote": "SELL",
                "platform": "deepseek",
                "requested_model": "deepseek-v4-pro",
                "served_model": "deepseek-v4-pro-202607",
                "served_model_verified": True,
                "timestamp": now,
            },
        ],
        "prediction_log.json": {"predictions": []},
    }
    monkeypatch.setattr(agents, "read_json", lambda path: payloads[Path(path).name])

    result = asyncio.run(agents.get_agents(_=None))
    row = result["agents"]["Aidan Tan"]

    assert row["total_votes"] == 3
    assert (row["buy_votes"], row["sell_votes"], row["hold_votes"]) == (2, 1, 0)
    assert row["platform"] == "deepseek"
    assert row["requested_model"] == "deepseek-v4-pro"
    assert row["served_model"] == "deepseek-v4-pro-202607"
    assert row["served_model_verified"] is True
    assert row["vote_skew"] == 1 / 3
    assert row["accuracy"] is None

    assert result["vote_window_size"] == 3
    assert result["agent_metrics_updated"] is not None
    assert result["prediction_log_updated"] is not None
    assert result["as_of"] is not None
    assert result["stale"] is False
    assert result["reason_code"] == "OK"


def test_portfolio_strict_sides_history_aliases_and_estimated_lots(monkeypatch, tmp_path):
    portfolio = _load_route(monkeypatch, tmp_path, "portfolio")
    positions = [
        _position("LONG", EnumLike("long"), 2500),
        _position("UNKNOWN", "PositionSide.LONG", 500),
        _position("SHORT", EnumLike("short"), -1200),
    ]

    class Client:
        def get_account(self):
            return _account()

        def get_all_positions(self):
            return positions

    monkeypatch.setattr(portfolio, "get_trading_client", Client)
    result = asyncio.run(portfolio.get_portfolio(_=None))

    assert [row["side"] for row in result["positions"]] == [
        "PositionSide.LONG",
        "PositionSide.SHORT",
    ]
    assert [row["side_normalized"] for row in result["positions"]] == ["long", "short"]
    assert result["position_rejections"] == [{
        "symbol": "UNKNOWN",
        "reason_code": "UNKNOWN_POSITION_SIDE",
        "detail": "position_side:unknown:positionside.long",
    }]

    canonical, request_kwargs = portfolio._portfolio_history_request_kwargs("5D", "1D")
    assert canonical == "5D"
    assert request_kwargs == {"period": "1W", "timeframe": "1D"}
    ytd, ytd_kwargs = portfolio._portfolio_history_request_kwargs(
        "YTD",
        "1D",
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
    )
    assert ytd == "YTD"
    assert ytd_kwargs["start"] == datetime(2026, 1, 1, tzinfo=timezone.utc)

    history = SimpleNamespace(
        timestamp=[1_700_000_000, 1_700_086_400],
        equity=[10000, 10100],
        profit_loss=[0, 100],
        profit_loss_pct=[0, 0.01],
    )
    points, legacy = portfolio._portfolio_history_points(history)
    envelope = portfolio._portfolio_history_response(
        points=points,
        legacy=legacy,
        requested_period="5D",
        period="5D",
        timeframe="1D",
        reason_code="OK",
        stale=False,
    )
    assert set(("points", "source", "as_of", "stale", "reason_code")) <= envelope.keys()
    assert envelope["bars"] == envelope["points"]
    assert envelope["timestamps"] == history.timestamp
    assert envelope["points"][0]["timestamp"].endswith("+00:00")

    lots = portfolio._build_tax_lots_from_positions({
        "positions": {
            "lot-1": {
                "symbol": "SPY",
                "status": "open",
                "quantity": 2,
                "entry_price": 500,
                "entry_time": 1_700_000_000,
            }
        }
    }, limit=10, now=1_700_086_400)
    assert lots["broker_verified"] is False
    assert lots["lots"][0]["broker_verified"] is False


def test_quote_candles_5d_are_structured_and_ttl_cached(monkeypatch, tmp_path):
    quotes = _load_route(monkeypatch, tmp_path, "quotes")
    quotes._QUOTE_CACHE.clear()
    calls = {"history": 0}
    bars = [{
        "timestamp": "2026-07-08T16:00:00-04:00",
        "open": 620.0,
        "high": 625.0,
        "low": 618.0,
        "close": 624.0,
        "volume": 123456,
    }]

    async def polygon_history(_symbol, days, limit):
        calls["history"] += 1
        assert days == 12
        assert limit == 5
        return bars

    monkeypatch.setattr(quotes, "_polygon_history", polygon_history)
    first = asyncio.run(quotes.get_quote_candles("spy", period="5D", range_=None, limit=None, _=None))
    second = asyncio.run(quotes.get_quote_candles("SPY", period="3M", range_="5D", limit=None, _=None))

    assert calls["history"] == 1
    assert first["bars"] == bars
    assert first["points"] == [{
        "t": bars[0]["timestamp"],
        "o": bars[0]["open"],
        "h": bars[0]["high"],
        "l": bars[0]["low"],
        "c": bars[0]["close"],
        "v": bars[0]["volume"],
    }]
    assert first["period"] == second["period"] == "5D"
    assert first["source"] == "polygon_aggregates"
    assert first["reason_code"] == "OK"
    assert first["stale"] is False
    assert second["freshness"]["cache_hit"] is True
    assert {"t", "o", "h", "l", "c", "v"} == set(first["points"][0])

    paths = {route.path for route in quotes.router.routes}
    assert {
        "/api/quote/{symbol}",
        "/api/quote/{symbol}/details",
        "/api/quote/{symbol}/history",
        "/api/quote/{symbol}/bars",
        "/api/quote/{symbol}/candles",
    } <= paths


def test_quote_missing_or_previous_close_values_never_render_as_live_zero(monkeypatch, tmp_path):
    quotes = _load_route(monkeypatch, tmp_path, "quotes")

    async def no_snapshot(_symbol):
        return {}

    async def no_quote(_symbol):
        return {}

    monkeypatch.setattr(quotes, "_polygon_snapshot", no_snapshot)
    monkeypatch.setattr(quotes, "_alpaca_quote", no_quote)
    missing = asyncio.run(quotes._fetch_equity_market_snapshot("SPY"))
    assert missing["display_price"] is None
    assert missing["bid"] is None
    assert missing["ask"] is None
    assert missing["spread"] is None
    assert missing["change"] is None
    assert missing["stale"] is True
    assert missing["reason_code"] == "QUOTE_DATA_UNAVAILABLE"

    async def previous_close_only(_symbol):
        return {"prev_close": 624.12}

    monkeypatch.setattr(quotes, "_polygon_snapshot", previous_close_only)
    previous = asyncio.run(quotes._fetch_equity_market_snapshot("SPY"))
    assert previous["display_price"] == 624.12
    assert previous["price_source"] == "prev_close"
    assert previous["stale"] is True
    assert previous["reason_code"] == "PREVIOUS_CLOSE_ONLY"


def test_quote_details_keep_legacy_beta_and_52w_fields(monkeypatch, tmp_path):
    quotes = _load_route(monkeypatch, tmp_path, "quotes")

    async def snapshot(symbol):
        return {
            "symbol": symbol,
            "source": "polygon_snapshot+alpaca_quote",
            "as_of": "2026-07-09T14:00:00+00:00",
            "stale": False,
            "reason_code": "OK",
        }

    async def details(_symbol):
        return {"name": "Example"}, {
            "stale": False,
            "stored_at": "2026-07-09T14:00:00+00:00",
            "reason_code": "OK",
        }

    async def cached_bars(*_args, **_kwargs):
        return [
            {"timestamp": "2025-07-09T16:00:00-04:00", "high": 100, "low": 50, "close": 75},
            {"timestamp": "2026-07-08T16:00:00-04:00", "high": 125, "low": 60, "close": 120},
        ], {"stale": False, "stored_at": "2026-07-09T14:00:00+00:00", "reason_code": "OK"}

    async def beta(_symbol, _bars):
        return 0.87

    monkeypatch.setattr(quotes, "get_equity_market_snapshot", snapshot)
    monkeypatch.setattr(quotes, "_cached_polygon_details", details)
    monkeypatch.setattr(quotes, "_cached_polygon_bars", cached_bars)
    monkeypatch.setattr(quotes, "_beta_1y", beta)

    result = asyncio.run(quotes.get_quote_details("spy", _=None))
    assert result["week52_high"] == 125
    assert result["week52_low"] == 50
    assert result["beta_1y"] == result["beta"] == 0.87
    assert result["week52"]["reason_code"] == "OK"
    assert result["beta_details"]["benchmark"] == "SPY"


def test_risk_strict_sides_share_broker_cache(monkeypatch, tmp_path):
    risk = _load_route(monkeypatch, tmp_path, "risk")
    risk._BROKER_STATE_CACHE.update({"stored_monotonic": -1e9, "stored_at": None, "data": None})
    calls = {"account": 0, "positions": 0}

    class Client:
        def get_account(self):
            calls["account"] += 1
            return _account()

        def get_all_positions(self):
            calls["positions"] += 1
            return [
                _position("A", EnumLike("long"), 2500),
                _position("B", EnumLike("short"), -1500),
                _position("BAD", "PositionSide.LONG", 500),
            ]

    monkeypatch.setattr(risk, "get_trading_client", Client)
    monkeypatch.setattr(risk, "read_json", lambda *_args, **_kwargs: {})

    first = asyncio.run(risk.get_risk(_=None))
    second = asyncio.run(risk.get_risk(_=None))

    assert calls == {"account": 1, "positions": 1}
    assert first["exposure"]["long"] == 2500
    assert first["exposure"]["short"] == 1500
    assert first["position_rejections"][0]["reason_code"] == "UNKNOWN_POSITION_SIDE"
    assert second["exposure_provenance"]["cache_hit"] is True


def test_risk_cached_indicator_provenance_and_alert_lifecycle(monkeypatch, tmp_path):
    risk = _load_route(monkeypatch, tmp_path, "risk")
    now = datetime.now(timezone.utc)
    now_epoch = now.timestamp()
    monkeypatch.setattr(risk.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(risk, "_file_mtime_epoch", lambda _path: now_epoch)
    monkeypatch.setattr(risk, "_market_hours_now", lambda: True)

    daily = {"vix": 19.4, "last_updated": now.isoformat()}
    journal = {"transitions": []}
    uw = {
        "timestamp": now.isoformat(),
        "freshness_ttl_seconds": 300,
        "market_context": {"market_tide_direction": "BULLISH"},
    }
    vix, sentiment = risk._market_indicator_details(daily, journal, uw)
    assert vix["value"] == 19.4
    assert vix["delivery"] == "cached_file"
    assert vix["vendor_call"] is False
    assert vix["reason_code"] == "OK"
    assert sentiment["value"] == "BULLISH"
    assert sentiment["source"] == "hydrinia_uw_snapshot"
    assert sentiment["reason_code"] == "OK"

    alerts = [
        {
            "severity": "warning",
            "type": "risk_limit",
            "message": "risk limit exceeded",
            "timestamp": datetime.fromtimestamp(now_epoch - 60, timezone.utc).isoformat(),
            "source": "test",
        },
        {
            "severity": "critical",
            "type": "engine_error",
            "message": "old engine failure",
            "timestamp": datetime.fromtimestamp(now_epoch - 7200, timezone.utc).isoformat(),
            "source": "test",
        },
        {
            "severity": "info",
            "type": "trade_event",
            "message": "trade filled",
            "timestamp": datetime.fromtimestamp(now_epoch - 10, timezone.utc).isoformat(),
            "source": "test",
        },
    ]
    active, diagnostics = risk._partition_alerts(alerts, limit=10, now=now_epoch)
    assert [row["type"] for row in active] == ["risk_limit"]
    assert all(row["reason_code"] for row in active + diagnostics)
    assert {row["resolution"]["reason_code"] for row in diagnostics} == {
        "TTL_EXPIRED",
        "INFORMATIONAL_EVENT",
    }


def test_risk_prefers_structured_health_and_hides_stale_sentinel(monkeypatch, tmp_path):
    risk = _load_route(monkeypatch, tmp_path, "risk")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(risk.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(risk, "_file_mtime_epoch", lambda _path: now.timestamp())
    monkeypatch.setattr(risk, "_market_hours_now", lambda: True)

    daily = {
        "vix": 30.0,
        "vix_health": {
            "value": 30.0,
            "source": "fail_closed",
            "valid": False,
            "current": False,
            "stale": True,
            "reason": "not_recorded",
            "timestamp": None,
            "age_seconds": None,
            "freshness_ttl_seconds": 180,
            "provenance": {"record": "MarketData.get_vix_health"},
        },
        "sentiment": "BEARISH",
        "sentiment_health": {
            "value": "BEARISH",
            "source": "unusual_whales.market_tide_snapshot",
            "valid": True,
            "current": False,
            "stale": True,
            "reason": "ok",
            "timestamp": (now - timedelta(minutes=20)).isoformat(),
            "age_seconds": 1200,
            "freshness_ttl_seconds": 300,
        },
        "last_updated": now.isoformat(),
    }

    vix, sentiment = risk._market_indicator_details(daily, {}, {})

    assert vix["value"] is None
    assert vix["last_value"] == 30.0
    assert vix["reason_code"] == "NOT_RECORDED"
    assert vix["provenance"]["record"] == "MarketData.get_vix_health"
    assert sentiment["value"] is None
    assert sentiment["last_value"] == "BEARISH"
    assert sentiment["reason_code"] == "SOURCE_STALE"


def test_system_health_restores_legacy_contract_and_account_cache(monkeypatch, tmp_path):
    system = _load_route(monkeypatch, tmp_path, "system")
    calls = {"account": 0}

    class Client:
        def get_account(self):
            calls["account"] += 1
            return _account()

    monkeypatch.setattr(system, "get_trading_client", Client)
    system._ACCOUNT_CACHE.update({"ts": -1e9, "data": None})
    first_account = asyncio.run(system._account_health())
    second_account = asyncio.run(system._account_health())
    assert first_account == second_account
    assert calls["account"] == 1

    async def account_health():
        return first_account

    async def run_in_executor(function, *args):
        return function(*args)

    monkeypatch.setattr(system, "_account_health", account_health)
    monkeypatch.setattr(system, "run_in_executor", run_in_executor)
    monkeypatch.setattr(system, "os", SimpleNamespace(environ={}, path=os.path))
    result = asyncio.run(system.system_health(_=None))

    assert set(result) == {"generated_at", "account", "sources"}
    assert result["account"]["account_number"] == "PA-123"
    assert [source["key"] for source in result["sources"]] == [
        "alpaca",
        "polygon",
        "uw",
        "news",
        "files",
    ]
    assert {route.path for route in system.router.routes} >= {"/api/system/health"}
