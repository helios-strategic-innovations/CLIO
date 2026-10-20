"""
OpenClaw/Moltbot Logging Configuration
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_level: str = "INFO"):
    log_dir = "/home/agent/openclaw/logs"
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # S215 R-33: apscheduler's per-run 'Running job' / 'executed successfully' INFO pairs are
    # ~80% of log volume. Default WARNING; CLIO_LOG_APSCHEDULER_LEVEL overrides (anything
    # unrecognised -> WARNING). Set before the early return so repeated calls still apply it.
    _aps_raw = os.environ.get("CLIO_LOG_APSCHEDULER_LEVEL", "WARNING").strip().upper()
    _aps_name = _aps_raw if _aps_raw in ("DEBUG", "INFO", "WARNING", "ERROR") else "WARNING"
    logging.getLogger("apscheduler.executors.default").setLevel(getattr(logging, _aps_name))

    # Prevent duplicate handlers on repeated calls
    if root_logger.handlers:
        return

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "openclaw.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    security_handler = RotatingFileHandler(
        os.path.join(log_dir, "security.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
    )
    security_handler.setLevel(logging.WARNING)
    security_handler.setFormatter(file_format)

    # SEC-1 (S211): scrub credentials from every record on every handler. Must be a
    # HANDLER filter -- a filter on the root logger is not consulted for child loggers.
    from agent.log_scrub import SecretScrubFilter
    _scrub = SecretScrubFilter()
    for _h in (console_handler, file_handler, security_handler):
        _h.addFilter(_scrub)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    security_logger = logging.getLogger("openclaw.security")
    security_logger.addHandler(security_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
