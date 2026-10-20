"""
CLIO Kronos — process entrypoint: `python -m agent.kronos`.

Dual-use-module discipline (CLAUDE.md anti-pattern #34): `load_dotenv()`
runs FIRST, before any import that could touch `agent.config` or read
Alpaca/env-derived config, so a bare `systemd` invocation (no shell that
pre-sources `.env`) never runs silently keyless. Mirrors `main.py`'s and
`agent.flowcollector.__main__`'s own top-of-file pattern exactly.

CLI:
  python -m agent.kronos                 # run forever (the systemd unit's mode)
  python -m agent.kronos --once           # one run right now, even outside market hours; prints a summary
  python -m agent.kronos --once --dry-run # one run, no file writes
  python -m agent.kronos --download       # fetch the pinned model/tokenizer revisions into the HF cache, then exit

Missing optional dependencies (torch/pandas/einops/safetensors/
huggingface_hub — see requirements-kronos.txt) log ONE clear line and exit
with code 3, for every subcommand (including `--download`, which strictly
only needs `huggingface_hub` — checking the whole set uniformly is the
conservative, simpler choice: a deploy that hasn't `pip install -r
requirements-kronos.txt`-ed yet should fail loudly on ANY invocation, not
just the ones that happen to need torch).
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone as _tz

logger = logging.getLogger("clio.kronos")

_REQUIRED_OPTIONAL_DEPS = ("torch", "pandas", "einops", "safetensors", "huggingface_hub")


def _setup_logging() -> None:
    # stdout only (-> journald via the unit's StandardOutput=journal) --
    # same rationale as agent.flowcollector.__main__._setup_logging: this
    # unit's sandbox keeps the repo's logs/ read-only, and a second process
    # rotating openclaw.log would garble moltbot's own lines.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _missing_dependencies() -> list[str]:
    missing = []
    for mod in _REQUIRED_OPTIONAL_DEPS:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m agent.kronos", description="CLIO Kronos forecaster service")
    p.add_argument("--once", action="store_true", help="run one forecast+score pass now (even outside market hours) and print a summary")
    p.add_argument("--dry-run", action="store_true", help="with --once: compute but never write kronos_forecast.json/kronos_scores.json/history")
    p.add_argument("--download", action="store_true", help="fetch the pinned model/tokenizer revisions into the local HF cache, then exit")
    return p.parse_args(argv)


async def _run_once(cfg, *, dry_run: bool) -> int:
    from agent.kronos.service import KronosService, is_active_now
    from zoneinfo import ZoneInfo

    service = KronosService(cfg)
    result = await service.run_once(dry_run=dry_run)
    et = ZoneInfo("America/New_York")
    now_et = datetime.fromtimestamp(result["as_of"], tz=_tz.utc).astimezone(et)
    from agent.integrations.market_events import MARKET_EARLY_CLOSE, MARKET_HOLIDAYS

    active = is_active_now(now_et, MARKET_HOLIDAYS, MARKET_EARLY_CLOSE)
    summary = {
        "as_of_et": now_et.isoformat(),
        "market_active_window": active,
        "ok_count": result["ok_count"],
        "total": result["total"],
        "dry_run": result["dry_run"],
        "symbols": {
            sym: {"status": e.get("status"), "horizons": sorted((e.get("horizons") or {}).keys()), "error": e.get("error")}
            for sym, e in result["symbols"].items()
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


async def _run_forever(cfg) -> int:
    from agent.kronos.service import KronosService

    service = KronosService(cfg)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_sigterm(*_args) -> None:
        logger.info("kronos: SIGTERM received, shutting down")
        loop.call_soon_threadsafe(stop_event.set)

    try:
        loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
        loop.add_signal_handler(signal.SIGINT, _handle_sigterm)
    except NotImplementedError:
        # add_signal_handler is unavailable on some platforms -- fall back
        # to the classic signal module (mirrors agent.flowcollector.__main__).
        signal.signal(signal.SIGTERM, lambda *_: loop.call_soon_threadsafe(stop_event.set))
        signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(stop_event.set))

    from agent.kronos.forecaster import ModelUnavailable

    try:
        await service.startup_check()
    except ModelUnavailable as e:
        logger.error(f"kronos: model failed to load at startup: {e} -- run `python -m agent.kronos --download` first")
        return 4
    await service.run_forever(stop_event)
    return 0


def main(argv=None) -> int:
    _setup_logging()
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    missing = _missing_dependencies()
    if missing:
        logger.error(
            f"kronos: missing optional dependencies: {', '.join(missing)} -- "
            f"run `pip install -r requirements-kronos.txt` first"
        )
        return 3

    from agent.kronos.config import config_summary, load_config

    cfg = load_config()
    issues = cfg.validate()
    if issues:
        logger.warning(f"kronos: config issues (continuing anyway): {issues}")
    logger.info(f"kronos: starting -- {config_summary(cfg)}")

    if args.download:
        from agent.kronos.forecaster import KronosPredictor

        KronosPredictor(cfg).download_weights()
        return 0

    if args.once:
        return asyncio.run(_run_once(cfg, dry_run=bool(args.dry_run)))

    return asyncio.run(_run_forever(cfg))


if __name__ == "__main__":
    sys.exit(main())
