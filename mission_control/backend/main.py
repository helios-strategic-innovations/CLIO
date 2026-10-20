"""Mission Control — FastAPI application."""

import logging
import sys
import os

# Ensure project root is on sys.path so `agent.*` imports work
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(os.path.join(_project_root, ".env"))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mission_control.backend.config import mc_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("mission_control")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Mission Control starting on port {mc_config.port}")
    if not mc_config.jwt_secret:
        raise RuntimeError("MISSION_CONTROL_JWT_SECRET not set — cannot start without auth secret")
    if not mc_config.password_hash:
        raise RuntimeError("S159 M-19: MISSION_CONTROL_PASSWORD_HASH must be set")
    yield
    logger.info("Mission Control shutting down")


app = FastAPI(
    title="Mission Control",
    description="Atlas Strategies Trading Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("MC_FRONTEND_URL", "https://control.atlasstrategies-voice.com"),
    ] + (["http://localhost:5173"] if os.getenv("MC_DEV_MODE", "0") == "1" else []),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Register route modules
# ---------------------------------------------------------------------------
from mission_control.backend.routes.auth import router as auth_router
from mission_control.backend.routes.health import router as health_router
from mission_control.backend.routes.portfolio import router as portfolio_router
from mission_control.backend.routes.orders import router as orders_router
from mission_control.backend.routes.quotes import router as quotes_router
from mission_control.backend.routes.trades import router as trades_router
from mission_control.backend.routes.directives import router as directives_router
from mission_control.backend.routes.actions import router as actions_router
from mission_control.backend.routes.knowledge import router as knowledge_router
from mission_control.backend.routes.strategies import router as strategies_router
from mission_control.backend.routes.calendar import router as calendar_router
from mission_control.backend.routes.news import router as news_router
from mission_control.backend.routes.agents import router as agents_router
from mission_control.backend.routes.risk import router as risk_router
from mission_control.backend.routes.audit import router as audit_router
from mission_control.backend.routes.options import router as options_router
from mission_control.backend.routes.files import router as files_router
from mission_control.backend.routes.system import router as system_router
from mission_control.backend.routes.briefing import router as briefing_router
from mission_control.backend.routes.analytics import router as analytics_router
from mission_control.backend.routes.dpof import router as dpof_router

app.include_router(auth_router)
app.include_router(health_router)
app.include_router(portfolio_router)
app.include_router(orders_router)
app.include_router(options_router)
app.include_router(quotes_router)
app.include_router(trades_router)
app.include_router(directives_router)
app.include_router(actions_router)
app.include_router(knowledge_router)
app.include_router(strategies_router)
app.include_router(calendar_router)
app.include_router(news_router)
app.include_router(agents_router)
app.include_router(risk_router)
app.include_router(audit_router)
app.include_router(files_router)
app.include_router(system_router)
app.include_router(briefing_router)
app.include_router(analytics_router)
app.include_router(dpof_router)

# ---------------------------------------------------------------------------
# WebSocket streams
# ---------------------------------------------------------------------------
from mission_control.backend.ws.manager import ws_manager  # noqa: F401
from mission_control.backend.ws import logs as ws_logs  # noqa: F401
from mission_control.backend.ws import prices as ws_prices  # noqa: F401
from mission_control.backend.ws import portfolio as ws_portfolio  # noqa: F401
from mission_control.backend.ws import news as ws_news  # noqa: F401
from mission_control.backend.ws import alerts as ws_alerts  # noqa: F401

app.include_router(ws_logs.router)
app.include_router(ws_prices.router)
app.include_router(ws_portfolio.router)
app.include_router(ws_news.router)
app.include_router(ws_alerts.router)
