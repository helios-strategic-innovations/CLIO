"""Auth routes — login endpoint."""

import time
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from mission_control.backend.auth import verify_password, create_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

# S153: In-memory login rate limiter
_login_attempts: dict[str, list[float]] = {}
_MAX_LOGIN_ATTEMPTS = 5
_LOGIN_WINDOW = 300  # 5 minutes


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    token: str
    expires_in: int


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request):
    # S153: Rate limit by IP
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    attempts = _login_attempts.get(client_ip, [])
    attempts = [t for t in attempts if now - t < _LOGIN_WINDOW]
    if len(attempts) >= _MAX_LOGIN_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")

    if not verify_password(req.password):
        attempts.append(now)
        _login_attempts[client_ip] = attempts
        raise HTTPException(status_code=401, detail="Invalid password")

    _login_attempts.pop(client_ip, None)  # Reset on success
    token = create_token()
    return LoginResponse(token=token, expires_in=24 * 3600)
