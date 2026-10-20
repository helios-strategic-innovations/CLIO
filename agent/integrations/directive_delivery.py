"""
Moltbot → Hydrinia Operator Directive Delivery
================================================
Writes structured directive files to the moltbot_knowledge bridge
for Hydrinia agent consumption. Atomic writes, audit trail.
"""

import asyncio
import fcntl
import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("moltbot.directive_delivery")


def _get_tradingbot_dir():
    try:
        from agent.config import config
        return config.tradingbot_dir
    except Exception:
        return "/home/agent/tradingbot/"


def _get_project_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_TB_DIR = _get_tradingbot_dir()
DIRECTIVES_DIR = os.path.join(_TB_DIR, "moltbot_knowledge", "directives")
DIRECTIVES_ARCHIVE_DIR = os.path.join(DIRECTIVES_DIR, "archive")
DIRECTIVE_RESPONSES_DIR = os.path.join(_TB_DIR, "moltbot_knowledge", "directive_responses")
# S190 L12: DIRECTIVES_LOG is CLIO-side audit trail only. Path lives outside
# Hydrinia's systemd whitelist (`/home/agent/openclaw/logs/`), and that is
# intentional — Hydrinia never reads CLIO's audit logs. Directives themselves
# write to the whitelisted /tradingbot/moltbot_knowledge/directives/ path above.
DIRECTIVES_LOG = os.path.join(_get_project_dir(), "logs", "directives.jsonl")

# TTL defaults per directive type (seconds)
TTL_DEFAULTS = {
    "EXECUTE": 3600,         # 1 hour
    "VOTE_INFLUENCE": 86400,  # 24 hours
    "CONSIDER": 172800,       # 48 hours
}

# Keywords for directive type classification
EXECUTE_KEYWORDS = {
    "sell", "buy", "close", "exit", "execute", "immediately",
    "now", "liquidate", "dump", "open", "enter", "take profit",
    "force execute",
}
VOTE_INFLUENCE_KEYWORDS = {
    "bearish", "bullish", "bias", "lean", "favor", "against",
    "for the next", "sentiment", "cautious", "aggressive",
    "overweight", "underweight", "reduce exposure",
    "vote influence", "vote_influence",
}
# Default is CONSIDER if no keywords match

# Common English words to exclude from symbol extraction
_SYMBOL_EXCLUSIONS = {
    "THE", "FOR", "AND", "ALL", "ARE", "BUT", "CAN", "DID", "GET",
    "HAS", "HER", "HIM", "HIS", "HOW", "ITS", "LET", "MAY", "NEW",
    "NOT", "NOW", "OLD", "OUR", "OUT", "OWN", "SAY", "SHE", "TOO",
    "USE", "WAY", "WHO", "BOY", "DID", "MAN", "DAY", "PIN", "TTL",
    "RED", "ONE", "TWO", "SET", "RUN", "TOP", "LOW", "BIG", "END",
    "PUT", "SAY", "ADD", "ANY", "FEW", "GOT", "YES", "YET", "TRY",
    "ASK", "OWN", "LET", "BE", "DO", "GO", "IF", "IN", "IS", "IT",
    "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "AT", "BY", "AN", "AS", "AM", "ME", "HE",
    "SELL", "BUY", "HOLD", "LONG", "SHORT", "CALL", "NOTE",
    "NEXT", "OVER", "THEM", "THAN", "EACH", "MADE", "LIKE",
    "TIME", "VERY", "WHEN", "COME", "BEEN", "SOME", "TAKE",
    "HAVE", "FROM", "WILL", "WITH", "WHAT", "THIS", "THAT",
    "ALSO", "BACK", "INTO", "JUST", "ONLY", "SAID", "THEY",
    "MOST", "MUCH", "MUST", "VOTE", "TYPE", "VETO", "STOP",
}


@dataclass
class Directive:
    id: str
    timestamp: str
    type: str               # EXECUTE, VOTE_INFLUENCE, CONSIDER
    target_agents: list     # ["all"] or ["Aidan Tan", "Victor Rostov"]
    symbol: Optional[str]
    command: str
    reasoning: Optional[str]
    ttl_seconds: int
    expires_at: str
    status: str             # "pending", "acknowledged", "expired"
    source_event_id: str
    executor: str


def classify_directive_type(command: str) -> str:
    """Classify directive type from command keywords.

    Explicit type prefixes (first 30 chars) take priority over body keywords.
    This lets the user say "Consider bearish pressure..." without the "bearish"
    keyword overriding their explicit "Consider" intent.
    """
    lower = command.lower()
    prefix = lower[:30]
    # Explicit type prefix takes priority
    if prefix.startswith("execute") or prefix.startswith("force execute"):
        return "EXECUTE"
    if prefix.startswith("vote influence") or prefix.startswith("vote_influence"):
        return "VOTE_INFLUENCE"
    if prefix.startswith("consider"):
        return "CONSIDER"
    # Fall back to keyword scanning (word-boundary to avoid substring false positives — Fix S104-I)
    for kw in EXECUTE_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', lower):
            return "EXECUTE"
    for kw in VOTE_INFLUENCE_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', lower):
            return "VOTE_INFLUENCE"
    return "CONSIDER"


def extract_symbol(command: str) -> Optional[str]:
    """Extract a stock ticker symbol from command text."""
    # Find all 2-5 uppercase letter words (min 2 chars to prevent single-letter false positives)
    candidates = re.findall(r'\b([A-Z]{2,5}(?:\.[A-Z])?)\b', command)
    for candidate in candidates:
        if candidate not in _SYMBOL_EXCLUSIONS:
            return candidate
    return None


def extract_target_agents(command: str) -> list[str]:
    """Extract target agents from command text, or default to ["all"]."""
    try:
        from agent.override_directive import parse_agent_directive
        directive = parse_agent_directive(command)
        if directive and directive.target_agent:
            return [directive.target_agent]
    except Exception as e:
        logger.warning(f"Agent directive parse failed (defaulting to all): {type(e).__name__}: {e}")
    return ["all"]


def _write_directive_file(directive: Directive) -> None:
    """Write directive JSON atomically to directives directory."""
    os.makedirs(DIRECTIVES_DIR, exist_ok=True)
    filename = f"{directive.id}.json"
    filepath = os.path.join(DIRECTIVES_DIR, filename)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(asdict(directive), f, indent=2, default=str)
    os.replace(tmp_path, filepath)
    logger.info(f"Directive written: {filepath}")


def _log_directive(directive: Directive) -> None:
    """Append directive to JSONL audit log."""
    lock_fd = None
    try:
        os.makedirs(os.path.dirname(DIRECTIVES_LOG), exist_ok=True)
        lock_fd = open(DIRECTIVES_LOG + ".lock", "a")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        with open(DIRECTIVES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(directive), default=str) + "\n")
    except Exception as e:
        logger.warning(f"Directive log write failed: {e}")
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


async def dispatch_directive(override_event, raw_message: str) -> str:
    """
    Main entry point: classify, write, and log an operator directive.

    Called from core.py when a legitimate override command is detected.
    Returns a confirmation message for the Telegram reply.
    """
    command_text = override_event.command_text or ""
    if not command_text:
        return "Override acknowledged. No command found after passphrase."

    try:
        from agent.hydrinia_bridge import bridge_enabled
        _bridge_on = bridge_enabled()
    except Exception:
        _bridge_on = False
    if not _bridge_on:
        logger.info(
            "Directive dispatch skipped — Hydrinia bridge is off "
            "(CLIO_HYDRINIA_BRIDGE_ENABLED unset/0); no directive file written."
        )
        return (
            "Hydrinia bridge is off — directive not delivered. "
            "Hydrinia is permanently retired; CLIO trades its own DPOF book "
            "and does not yet take operator directives for it."
        )

    directive_type = classify_directive_type(command_text)
    symbol = extract_symbol(command_text)
    target_agents = extract_target_agents(command_text)
    ttl = TTL_DEFAULTS[directive_type]

    now = datetime.now(timezone.utc)
    directive_id = f"DIR-{now.strftime('%Y%m%d-%H%M%S')}-{os.getpid() % 1000:03d}"

    directive = Directive(
        id=directive_id,
        timestamp=now.isoformat(),
        type=directive_type,
        target_agents=target_agents,
        symbol=symbol,
        command=command_text[:500],
        reasoning=None,
        ttl_seconds=ttl,
        expires_at=(now + timedelta(seconds=ttl)).isoformat(),
        status="pending",
        source_event_id=override_event.timestamp,
        executor="Dominic",
    )

    _write_directive_file(directive)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _log_directive, directive)

    # Audit trail (fire-and-forget)
    try:
        from agent.safety.audit_log import audit_security_event
        audit_security_event(
            "directive_dispatched",
            f"Directive {directive_id}: {directive_type} on {symbol or 'global'}",
            details={"directive_id": directive_id, "type": directive_type,
                     "symbol": symbol, "target_agents": target_agents},
        )
    except Exception as e:
        logger.debug(f"Fire-and-forget directive audit log failed: {e}")

    target_desc = "all agents" if target_agents == ["all"] else f"agents: {', '.join(target_agents)}"
    ttl_desc = f"{ttl // 3600}h" if ttl >= 3600 else f"{ttl // 60}m"

    return (
        f"Directive dispatched.\n"
        f"  ID: {directive_id}\n"
        f"  Type: {directive_type}\n"
        f"  Symbol: {symbol or 'global'}\n"
        f"  Target: {target_desc}\n"
        f"  TTL: {ttl_desc}\n"
        f"  Command: {command_text[:100]}{'...' if len(command_text) > 100 else ''}"
    )


def get_active_directives() -> list[dict]:
    """Return all non-expired directive files. Used by /directive command.

    Expired directives are moved to archive/ automatically.
    Also ensures directive_responses/ directory exists on first run.
    Empty when the Hydrinia bridge is off (Hydrinia is permanently retired).
    """
    try:
        from agent.hydrinia_bridge import bridge_enabled
        if not bridge_enabled():
            return []
    except Exception as e:
        logger.debug(f"Bridge check failed (treating as disabled): {e}")
        return []

    # Ensure directories exist
    os.makedirs(DIRECTIVES_DIR, exist_ok=True)
    os.makedirs(DIRECTIVE_RESPONSES_DIR, exist_ok=True)

    result = []
    now = datetime.now(timezone.utc)
    for f in sorted(os.listdir(DIRECTIVES_DIR)):
        if not f.endswith(".json"):
            continue
        filepath = os.path.join(DIRECTIVES_DIR, f)
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath) as fh:
                d = json.load(fh)
            expires_at = d.get("expires_at", "")
            if expires_at:
                exp = datetime.fromisoformat(expires_at)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if now >= exp:
                    # Archive expired directive
                    _archive_directive(filepath, f)
                    continue
            result.append(d)
        except Exception as e:
            logger.debug(f"Fire-and-forget directive file parse failed: {e}")
            continue
    return result


def _archive_directive(filepath: str, filename: str) -> None:
    """Move an expired directive file to the archive directory."""
    try:
        os.makedirs(DIRECTIVES_ARCHIVE_DIR, exist_ok=True)
        dest = os.path.join(DIRECTIVES_ARCHIVE_DIR, filename)
        shutil.move(filepath, dest)
        logger.info(f"Directive archived: {filename}")
    except Exception as e:
        logger.warning(f"Failed to archive directive {filename}: {e}")
