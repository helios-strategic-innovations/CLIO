"""Mission Control dashboard configuration."""

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env EARLY so all config reads see the values
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


@dataclass
class MCConfig:
    port: int = 8001
    jwt_secret: str = ""
    password_hash: str = ""
    jwt_expiry_hours: int = 24
    external_api_key: str = ""  # Static API key for external tools (Perplexity, etc.)
    # Paths
    openclaw_root: str = "/home/agent/openclaw"
    tradingbot_root: str = "/home/agent/tradingbot"
    hydrinia_vault: str = "/home/agent/.hydrinia/vault"
    # Services
    allowed_services: tuple = ("hydrinia", "moltbot", "moltbot-voice")

    @classmethod
    def from_env(cls) -> "MCConfig":
        return cls(
            port=int(os.getenv("MISSION_CONTROL_PORT", "8001")),
            jwt_secret=os.getenv("MISSION_CONTROL_JWT_SECRET", ""),
            password_hash=os.getenv("MISSION_CONTROL_PASSWORD_HASH", ""),
            jwt_expiry_hours=int(os.getenv("MISSION_CONTROL_JWT_EXPIRY_HOURS", "24")),
            external_api_key=os.getenv("EXTERNAL_API_KEY", ""),
        )


mc_config = MCConfig.from_env()


def effective_allowed_services() -> tuple:
    """`allowed_services`, minus "hydrinia" unless the Hydrinia bridge is enabled.

    Hydrinia (/home/agent/tradingbot) is permanently retired (operator direction,
    2026-09-24); CLIO trades natively via agent/dpof/. Service-restart/listing
    routes must not offer to manage a service nobody wants running unless the
    bridge is explicitly re-enabled.
    """
    try:
        from agent.hydrinia_bridge import bridge_enabled

        if not bridge_enabled():
            return tuple(s for s in mc_config.allowed_services if s != "hydrinia")
    except Exception:
        pass
    return mc_config.allowed_services
