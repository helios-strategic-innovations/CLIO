import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


# The checkout that CONTAINS this test (see test_voice_pin_non_persistence.py: a sibling "openclaw" dir is the serving tree).
ROOT = Path(__file__).resolve().parents[1]
AUTH_PATH = ROOT / "mission_control" / "backend" / "auth.py"


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class HTTPAuthorizationCredentials:
    def __init__(self, scheme, credentials):
        self.scheme = scheme
        self.credentials = credentials


class HTTPBearer:
    def __init__(self, auto_error=True):
        self.auto_error = auto_error


def _request(method, path):
    return SimpleNamespace(
        method=method,
        url=SimpleNamespace(path=path),
        query_params={},
    )


def _load_auth(monkeypatch):
    fastapi = types.ModuleType("fastapi")
    fastapi.Request = object
    fastapi.HTTPException = HTTPException
    fastapi.Depends = lambda dependency: None

    fastapi_security = types.ModuleType("fastapi.security")
    fastapi_security.HTTPBearer = HTTPBearer
    fastapi_security.HTTPAuthorizationCredentials = HTTPAuthorizationCredentials

    bcrypt = types.ModuleType("bcrypt")
    bcrypt.checkpw = lambda _plain, _hashed: False

    jwt = types.ModuleType("jwt")
    jwt.ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
    jwt.InvalidTokenError = type("InvalidTokenError", (Exception,), {})
    jwt.encode = lambda payload, _secret, algorithm=None: "token"
    jwt.decode = lambda _token, _secret, algorithms=None: {"sub": "operator"}

    mission_control = types.ModuleType("mission_control")
    backend = types.ModuleType("mission_control.backend")
    config = types.ModuleType("mission_control.backend.config")
    config.mc_config = SimpleNamespace(
        external_api_key="review-only-key",
        jwt_expiry_hours=1,
        jwt_secret="not-used",
        password_hash="",
    )

    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.security", fastapi_security)
    monkeypatch.setitem(sys.modules, "bcrypt", bcrypt)
    monkeypatch.setitem(sys.modules, "jwt", jwt)
    monkeypatch.setitem(sys.modules, "mission_control", mission_control)
    monkeypatch.setitem(sys.modules, "mission_control.backend", backend)
    monkeypatch.setitem(sys.modules, "mission_control.backend.config", config)

    module_name = f"_mc_auth_under_test_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(module_name, AUTH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_real_auth_module_loaded(auth):
    assert Path(auth.__file__).resolve() == AUTH_PATH
    assert Path(auth.require_auth.__code__.co_filename).resolve() == AUTH_PATH
    assert Path(auth._api_key_route_allowed.__code__.co_filename).resolve() == AUTH_PATH


def test_external_api_key_cannot_read_sensitive_system_surfaces(monkeypatch):
    auth = _load_auth(monkeypatch)
    _assert_real_auth_module_loaded(auth)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="review-only-key")

    allowed = asyncio.run(
        auth.require_auth(_request("GET", "/api/system/services"), credentials)
    )
    assert allowed == {"sub": "external_api", "scope": "readonly"}

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth.require_auth(_request("POST", "/api/system/services"), credentials))
    assert exc_info.value.status_code == 403

    for path in (
        "/api/system/services",
        "/api/system/metrics",
        "/api/system/bot-status",
        "/api/portfolio",
        "/api/news",
        "/api/health",
        "/api/health/detailed",
    ):
        allowed = asyncio.run(auth.require_auth(_request("GET", path), credentials))
        assert allowed == {"sub": "external_api", "scope": "readonly"}

    for path in ("/api/system/logs", "/api/system/env"):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(auth.require_auth(_request("GET", path), credentials))
        assert exc_info.value.status_code == 403
