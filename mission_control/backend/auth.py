"""JWT authentication for Mission Control."""

import hmac
import os
import time
import logging
import posixpath
from typing import Optional

import bcrypt
import jwt
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from mission_control.backend.config import mc_config

logger = logging.getLogger("mission_control.auth")
_bearer_scheme = HTTPBearer(auto_error=False)

# External API keys are strictly read-only. Exact paths avoid prefix collisions;
# prefixes cover only route families whose mutating methods are separately denied.
_API_KEY_READ_METHODS = frozenset({"GET", "HEAD"})
_API_KEY_ALLOWED_EXACT_PATHS = frozenset({
    "/api/system/services",
    "/api/system/metrics",
    "/api/system/bot-status",
    "/api/portfolio",
    "/api/news",
    "/api/health",
    "/api/health/detailed",
})
_API_KEY_ALLOWED_READ_PREFIXES = (
    "/api/files/",
    "/api/briefing/",
    "/api/portfolio/",
    "/api/hydrinia/",
    "/api/news/",
    "/api/dpof/",
)


def _api_key_route_allowed(method: str, raw_path: str) -> bool:
    """Return whether a static external key may access this read route."""
    if str(method or "").upper() not in _API_KEY_READ_METHODS:
        return False
    path = posixpath.normpath(raw_path or "/")
    return path in _API_KEY_ALLOWED_EXACT_PATHS or any(
        path.startswith(prefix) for prefix in _API_KEY_ALLOWED_READ_PREFIXES
    )


def verify_password(plain: str) -> bool:
    if not mc_config.password_hash:
        logger.error("No password hash configured — login impossible")
        return False
    return bcrypt.checkpw(plain.encode(), mc_config.password_hash.encode())


def create_token() -> str:
    payload = {
        "sub": "operator",
        "iat": int(time.time()),
        "exp": int(time.time()) + mc_config.jwt_expiry_hours * 3600,
    }
    return jwt.encode(payload, mc_config.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, mc_config.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def _check_api_key(token: str) -> Optional[dict]:
    """Check if token matches the static external API key. Constant-time comparison."""
    if not mc_config.external_api_key:
        return None
    if hmac.compare_digest(token.encode(), mc_config.external_api_key.encode()):
        return {"sub": "external_api", "scope": "readonly"}
    return None


def require_trade_execution_enabled() -> bool:
    """Fail closed unless Mission Control trade execution is explicitly enabled."""
    if os.getenv("MC_TRADE_EXECUTION", "0") != "1":
        logger.warning("MC trade execution blocked (MC_TRADE_EXECUTION disabled)")
        raise HTTPException(status_code=403, detail="Trade execution is disabled")
    return True


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """Dependency that enforces JWT or API key auth on protected routes."""
    token = None

    # Try Authorization header first
    if credentials:
        token = credentials.credentials

    # Fallback: query param (for WebSocket connections)
    # S118 W-065: JWT in URL is standard for WebSocket auth (no header support).
    # Mitigated by: short-lived tokens, HTTPS-only, no URL logging in Caddy.
    if not token:
        token = request.query_params.get("token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Try API key first, then JWT
    payload = _check_api_key(token)
    if payload is not None:
        if not _api_key_route_allowed(request.method, request.url.path):
            raise HTTPException(status_code=403, detail="API key not authorized for this endpoint")
        return payload

    payload = decode_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return payload
