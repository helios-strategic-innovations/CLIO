# CLIO

CLIO (Continuous Learning Intelligence Operations) is an AI-assisted options research and paper-trading system for SPY and QQQ, built by Helios Strategic Innovations. It runs as a long-lived Python service with a Telegram interface, a phone line, and a web dashboard.

**Status:** paper trading only. Nothing in this repository is investment advice or a record of trading returns.

## What it does

- **DPOF options engine** (`agent/dpof/`). Every 30 seconds, CLIO scores options flow (net options flow, premium, dark pool, and gamma exposure) and a Kronos price forecast into one composite signal. When the signal clears its gates, a language model decides whether to enter, which next-session-expiry contract to buy, and how many. The same model later decides when to trim or exit. A second, independent model can veto entries, and borderline calls get a bull-versus-bear debate. Code enforces solvency and closes every position before the market close; there are no mechanical stop-losses or take-profits.
- **Market data and intelligence** (`agent/integrations/`). Unusual Whales flow data behind a shared request pacer and quota ledger, Alpaca market data, an economic calendar with surprise scoring, a volatility-surface monitor, a breaking-news classifier, and a registry of 54 global news feeds.
- **Self-computed flow** (`agent/flowcollector/`). Computes options flow and Black-Scholes greeks from Alpaca's options feed; DPOF falls back to it when vendor data goes stale.
- **Kronos forecaster** (`agent/kronos/`). Runs the open-source Kronos candlestick model on 5-minute SPY and QQQ bars, sampling 16 price paths every five minutes during market hours.
- **Learning and memory** (`agent/memory/`, `agent/integrations/learning_engine.py`). Research cycles every 15 minutes feed a knowledge base, a five-tier memory (working, episodic, semantic, procedural, and meta) carries context between conversations, and DPOF writes a lesson after each closed trade.
- **Decision record** (`agent/decisions/`). When Supabase is configured, every decision question, answer, and outcome is recorded.
- **Interfaces.** A Telegram bot (`agent/integrations/telegram_bot.py`), a Twilio voice server (`voice_server.py`), and Mission Control, a FastAPI and React dashboard (`mission_control/`). Mission Control's trade buttons stay disabled unless `MC_TRADE_EXECUTION=1`.

## Running it

CLIO targets Python 3.12.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-mission-control.txt
cp .env.template .env        # fill in the keys for the services you use
python main.py               # the agent, Telegram bot, and scheduler
uvicorn mission_control.backend.main:app   # Mission Control backend
```

Kronos needs `requirements-kronos.txt` (PyTorch) and the model weights pinned in `agent/kronos/vendor/VENDOR_PIN.md`. Example systemd units are in `systemd/`.

DPOF starts in paper mode. Live trading stays off unless several independent settings agree, including an explicit acknowledgment and an account match; see `agent/dpof/broker.py`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

On Python 3.12 the suite runs 4,402 tests: 4,399 pass, and 3 are skipped when run as root. Tests never touch live state: `tests/conftest.py` redirects every state file to a temporary path and disables `.env` loading.

## Design notes

Architecture decision records are in `docs/design/adr/`, and the DPOF, Kronos, and learning-loop designs are in `docs/design/`. Some of these documents refer to internal notes that are not part of this repository.

## Third-party code

`agent/kronos/vendor/` contains the Kronos model (github.com/shiyu-coder/Kronos), MIT License, Copyright (c) 2025 ShiYu. Its license file is in that directory.

## License

Copyright (c) 2026 Helios Strategic Innovations. All rights reserved.

This repository is published for reference. No license is granted to use, copy, modify, or distribute the code.
