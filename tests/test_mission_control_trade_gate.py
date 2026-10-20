import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


class DependencyMarker:
    def __init__(self, dependency):
        self.dependency = dependency


def _install_stubs(monkeypatch, tmp_path):
    fastapi = types.ModuleType("fastapi")

    class APIRouter:
        def __init__(self, prefix="", tags=None):
            self.prefix = prefix
            self.tags = tags or []
            self.routes = []

        def _route(self, path):
            def decorator(function):
                self.routes.append(SimpleNamespace(path=f"{self.prefix}{path}", endpoint=function))
                return function
            return decorator

        get = _route
        post = _route
        delete = _route

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    def depends(dependency):
        return DependencyMarker(dependency)

    fastapi.APIRouter = APIRouter
    fastapi.Depends = depends
    fastapi.HTTPException = HTTPException
    fastapi.Query = lambda default=None, **_kwargs: default
    fastapi.Request = object

    security = types.ModuleType("fastapi.security")

    class HTTPBearer:
        def __init__(self, auto_error=True):
            self.auto_error = auto_error

    class HTTPAuthorizationCredentials:
        def __init__(self, credentials="token"):
            self.credentials = credentials

    security.HTTPBearer = HTTPBearer
    security.HTTPAuthorizationCredentials = HTTPAuthorizationCredentials

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})

    bcrypt = types.ModuleType("bcrypt")
    bcrypt.checkpw = lambda _plain, _hashed: True

    jwt = types.ModuleType("jwt")

    class ExpiredSignatureError(Exception):
        pass

    class InvalidTokenError(Exception):
        pass

    jwt.ExpiredSignatureError = ExpiredSignatureError
    jwt.InvalidTokenError = InvalidTokenError
    jwt.encode = lambda payload, _secret, algorithm=None: f"encoded:{payload['sub']}"
    jwt.decode = lambda token, _secret, algorithms=None: {"sub": "operator"} if token else None

    config = types.ModuleType("mission_control.backend.config")
    config.mc_config = SimpleNamespace(
        external_api_key="external-key",
        password_hash="hash",
        jwt_secret="secret",
        jwt_expiry_hours=24,
        tradingbot_root=str(tmp_path / "tradingbot"),
        allowed_services=("hydrinia", "moltbot", "moltbot-voice"),
    )

    deps = types.ModuleType("mission_control.backend.deps")

    async def run_in_executor(function, *args):
        return function(*args)

    deps.get_trading_client = lambda: None
    deps.run_in_executor = run_in_executor

    alpaca_values = types.ModuleType("agent.integrations.alpaca_values")
    alpaca_values.normalize_order_side = lambda side: getattr(side, "value", side)

    modules = {
        "fastapi": fastapi,
        "fastapi.security": security,
        "pydantic": pydantic,
        "bcrypt": bcrypt,
        "jwt": jwt,
        "mission_control.backend.config": config,
        "mission_control.backend.deps": deps,
        "agent.integrations.alpaca_values": alpaca_values,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return fastapi


def _load_module(monkeypatch, module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_auth(monkeypatch, tmp_path):
    _install_stubs(monkeypatch, tmp_path)
    return _load_module(
        monkeypatch,
        f"_mc_gate_auth_{id(tmp_path)}",
        ROOT / "mission_control" / "backend" / "auth.py",
    )


def _load_route(monkeypatch, tmp_path, auth_module, name):
    monkeypatch.setitem(sys.modules, "mission_control.backend.auth", auth_module)
    return _load_module(
        monkeypatch,
        f"_mc_gate_route_{name}_{id(tmp_path)}",
        ROOT / "mission_control" / "backend" / "routes" / f"{name}.py",
    )


def _dependency_names(function):
    defaults = list(function.__defaults__ or [])
    defaults.extend((function.__kwdefaults__ or {}).values())
    return [
        marker.dependency.__name__
        for marker in defaults
        if isinstance(marker, DependencyMarker)
    ]


def test_trade_execution_dependency_fails_closed_and_passes_only_when_enabled(monkeypatch, tmp_path):
    auth = _load_auth(monkeypatch, tmp_path)
    assert hasattr(auth, "require_trade_execution_enabled")

    monkeypatch.delenv("MC_TRADE_EXECUTION", raising=False)
    with pytest.raises(auth.HTTPException) as absent:
        auth.require_trade_execution_enabled()
    assert absent.value.status_code == 403

    monkeypatch.setenv("MC_TRADE_EXECUTION", "0")
    with pytest.raises(auth.HTTPException) as disabled:
        auth.require_trade_execution_enabled()
    assert disabled.value.status_code == 403

    monkeypatch.setenv("MC_TRADE_EXECUTION", "1")
    assert auth.require_trade_execution_enabled() is True


def test_trade_execution_dependency_is_attached_only_to_submit_and_flatten_routes(monkeypatch, tmp_path):
    auth = _load_auth(monkeypatch, tmp_path)
    orders = _load_route(monkeypatch, tmp_path, auth, "orders")
    options = _load_route(monkeypatch, tmp_path, auth, "options")
    actions = _load_route(monkeypatch, tmp_path, auth, "actions")

    assert _dependency_names(orders.submit_order) == [
        "require_auth",
        "require_trade_execution_enabled",
    ]
    assert _dependency_names(options.submit_options_order) == [
        "require_auth",
        "require_trade_execution_enabled",
    ]
    assert _dependency_names(actions.flatten_all) == [
        "require_auth",
        "require_trade_execution_enabled",
    ]

    assert _dependency_names(orders.cancel_order) == ["require_auth"]
    assert _dependency_names(actions.halt_trading) == ["require_auth"]
    assert _dependency_names(actions.resume_trading) == ["require_auth"]
    assert _dependency_names(actions.restart_service) == ["require_auth"]
