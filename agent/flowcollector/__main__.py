"""
CLIO flowcollector — process entrypoint: `python -m agent.flowcollector`.

Dual-use-module discipline (CLAUDE.md anti-pattern #34): `load_dotenv()`
runs FIRST, before any import that could touch `agent.config` or read
Alpaca/env-derived config, so a bare `systemd` invocation (no shell that
pre-sources `.env`) never runs silently keyless. This mirrors `main.py`'s
own top-of-file `from dotenv import load_dotenv; load_dotenv()` pattern
exactly.

Graceful SIGTERM shutdown: sets an asyncio.Event via
`loop.call_soon_threadsafe` (signal handlers run outside the event loop),
which `FlowCollectorService.run_forever` polls each tick and unwinds
cleanly (stops stream supervisors, joins their threads, writes one final
state snapshot) rather than being killed mid-write.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
import signal

logger = logging.getLogger("clio.flowcollector")


def _setup_logging() -> None:
    # stdout only (-> journald via the unit's StandardOutput=journal). Deliberately NOT
    # agent.logging_config.setup_logging(): that attaches RotatingFileHandlers to
    # /home/agent/openclaw/logs/openclaw.log, which moltbot also rotates (two processes
    # rotating one file loses/garbles lines), and this unit's sandbox keeps logs/ read-only.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


async def _main() -> None:
    from agent.flowcollector.collector import FlowCollectorService
    from agent.flowcollector.config import load_config

    cfg = load_config()
    issues = cfg.validate()
    if issues:
        logger.warning(f"flowcollector: config issues (continuing anyway): {issues}")
    if not (cfg.alpaca_key_id and cfg.alpaca_secret_key):
        logger.warning(
            "flowcollector: no Alpaca keys configured (ALPACA_PAPER_KEY_ID/ALPACA_PAPER_SECRET_KEY) -- "
            "streams will fail to authenticate; the process still runs (idle/retry) rather than exiting, "
            "consistent with 'never go blind'."
        )

    logger.info(
        f"flowcollector: starting -- underlyings={cfg.underlyings} band=+/-{cfg.strike_band_pct:.1%} "
        f"max_dte={cfg.max_dte} options_feed={cfg.options_feed} stock_feed={cfg.stock_feed} "
        f"state_file={cfg.state_file}"
    )

    service = FlowCollectorService(cfg)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_sigterm(*_args) -> None:
        logger.info("flowcollector: SIGTERM received, shutting down")
        loop.call_soon_threadsafe(stop_event.set)

    try:
        loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
        loop.add_signal_handler(signal.SIGINT, _handle_sigterm)
    except NotImplementedError:
        # add_signal_handler is unavailable on some platforms (e.g. Windows
        # ProactorEventLoop) -- fall back to the classic signal module. Not
        # exercised on the VPS (Linux), but never crash the process over it.
        signal.signal(signal.SIGTERM, lambda *_: loop.call_soon_threadsafe(stop_event.set))
        signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(stop_event.set))

    await service.run_forever(stop_event)


def main() -> None:
    _setup_logging()
    asyncio.run(_main())


if __name__ == "__main__":
    main()
