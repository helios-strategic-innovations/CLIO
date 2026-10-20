"""Pass-2 verifier V2 (external API contracts / data lens) regression tests.

Confirmed gap: agent/dpof/broker.py's `_feed_for_options`/`_feed_for_stock`
picked OPRA/SIP (or indicative/iex) from config but never handled a 401/403
subscription error at the actual call sites (`get_option_chain`,
`get_option_quotes`, `get_minute_bars`) -- a revoked/missing OPRA or SIP
entitlement would silently starve DPOF of option quotes / bars every cycle
with no fallback (config.py's own docstring promised this belongs in
broker.py/data.py: "Feed selection/fallback itself lives in broker.py/
data.py, not here"). Fixed: a 401/403 on the configured premium feed
downgrades it to indicative/iex, sticky for the life of the DPOFBroker
instance, with exactly ONE warning log and status visible via
`broker.feed_status()`. A non-auth error (network, 5xx, etc.) must NOT
trigger a downgrade -- only a subscription-shaped error should.
"""

import asyncio
import logging

import pytest

from agent.dpof.broker import DPOFBroker
from agent.dpof.config import DpofMode
from agent.dpof.models import OrderLeg  # noqa: F401  (keeps parity with test_broker.py's import style)


class _FeedError(Exception):
    """Mimics alpaca-py's APIError: exposes .status_code, per broker.py's
    own _is_not_found/_is_transient status_code lookup pattern."""

    def __init__(self, status_code):
        super().__init__(f"http_{status_code}")
        self.status_code = status_code


class FakeOptionDataClient:
    """get_option_snapshot fails `fail_times` times with `fail_status`, then
    succeeds. Records every request's .feed so the test can assert exactly
    which feed each call actually used."""

    def __init__(self, fail_times=0, fail_status=403):
        self.feeds_used = []
        self.calls = 0
        self._fail_times = fail_times
        self._fail_status = fail_status

    def get_option_snapshot(self, req):
        self.calls += 1
        self.feeds_used.append(req.feed)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise _FeedError(self._fail_status)
        symbols = req.symbol_or_symbols
        if isinstance(symbols, str):
            symbols = [symbols]
        return {s: None for s in symbols}


class FakeStockDataClient:
    """get_stock_bars fails `fail_times` times with `fail_status`, then succeeds."""

    def __init__(self, fail_times=0, fail_status=401):
        self.feeds_used = []
        self.calls = 0
        self._fail_times = fail_times
        self._fail_status = fail_status

    def get_stock_bars(self, req):
        self.calls += 1
        self.feeds_used.append(req.feed)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise _FeedError(self._fail_status)
        return {req.symbol_or_symbols: []}


def _broker(cfg, *, option_data_client=None, stock_data_client=None):
    return DPOFBroker(
        cfg,
        trading_client=object(),
        stock_data_client=stock_data_client or object(),
        option_data_client=option_data_client or object(),
    )


# ---------------------------------------------------------------------------
# Options feed (OPRA -> indicative)
# ---------------------------------------------------------------------------


def test_options_feed_defaults_to_configured_value(cfg):
    broker = _broker(cfg)
    assert broker._cfg.options_feed == "opra"
    from alpaca.data.enums import OptionsFeed

    assert broker._feed_for_options() == OptionsFeed.OPRA
    status = broker.feed_status()
    assert status["options_feed_downgraded"] is False
    assert status["options_feed_effective"] == "opra"


def test_options_feed_downgrades_on_403_and_retries_once(cfg, caplog):
    client = FakeOptionDataClient(fail_times=1, fail_status=403)
    broker = _broker(cfg, option_data_client=client)

    async def _t():
        with caplog.at_level(logging.WARNING, logger="clio.dpof.broker"):
            out = await broker.get_option_quotes(["SPY261002C00575000"])
        assert out is not None  # succeeded after the retry -- never blind
        assert client.calls == 2  # 1 failing OPRA attempt + 1 retry
        from alpaca.data.enums import OptionsFeed

        assert client.feeds_used[0] == OptionsFeed.OPRA
        assert client.feeds_used[1] == OptionsFeed.INDICATIVE

        status = broker.feed_status()
        assert status["options_feed_downgraded"] is True
        assert status["options_feed_effective"] == "indicative"
        assert status["options_feed_downgrade_reason"] == "http_403"

        # Exactly one WARNING for the downgrade.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "opra" in warnings[0].message.lower()

    asyncio.run(_t())


def test_options_feed_downgrade_is_sticky_no_repeat_warning(cfg, caplog):
    client = FakeOptionDataClient(fail_times=1, fail_status=401)
    broker = _broker(cfg, option_data_client=client)

    async def _t():
        with caplog.at_level(logging.WARNING, logger="clio.dpof.broker"):
            await broker.get_option_quotes(["SPY261002C00575000"])  # triggers the downgrade
            client.calls = 0
            client.feeds_used = []
            await broker.get_option_quotes(["SPY261002C00580000"])  # already downgraded

        from alpaca.data.enums import OptionsFeed

        assert client.calls == 1  # no retry needed -- indicative used straight away
        assert client.feeds_used == [OptionsFeed.INDICATIVE]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1  # still only the ONE warning from the first downgrade

    asyncio.run(_t())


def test_options_feed_non_auth_error_never_downgrades(cfg):
    # 400 is deliberately NOT in _call()'s own transient-retry set (429/5xx) --
    # this isolates the feed-fallback decision from DPOFBroker._call()'s
    # unrelated built-in retry-on-5xx behavior.
    client = FakeOptionDataClient(fail_times=99, fail_status=400)
    broker = _broker(cfg, option_data_client=client)

    async def _t():
        out = await broker.get_option_quotes(["SPY261002C00575000"])
        assert out is None  # genuine failure, correctly propagated
        assert client.calls == 1  # no fallback retry attempted for a non-auth error
        status = broker.feed_status()
        assert status["options_feed_downgraded"] is False

    asyncio.run(_t())


def test_options_feed_no_downgrade_when_already_configured_indicative(cfg):
    from dataclasses import replace as dc_replace

    indicative_cfg = dc_replace(cfg, options_feed="indicative")
    client = FakeOptionDataClient(fail_times=99, fail_status=403)
    broker = _broker(indicative_cfg, option_data_client=client)

    async def _t():
        out = await broker.get_option_quotes(["SPY261002C00575000"])
        assert out is None
        assert client.calls == 1  # nothing to downgrade TO -- already indicative
        assert broker.feed_status()["options_feed_downgraded"] is False

    asyncio.run(_t())


def test_get_option_chain_snapshot_leg_also_falls_back(cfg):
    """get_option_chain's snapshot fetch (separate from its contracts fetch)
    goes through the same fallback path."""
    from alpaca.trading.enums import AssetStatus

    class FakeContract:
        def __init__(self, symbol):
            self.symbol = symbol
            self.expiration_date = "2026-09-25"
            self.strike_price = "575.0"
            self.open_interest = None

    class FakeContractsResp:
        def __init__(self, contracts):
            self.option_contracts = contracts
            self.next_page_token = None

    class FakeTradingClient:
        def get_option_contracts(self, req):
            return FakeContractsResp([FakeContract("SPY261002C00575000")])

    option_client = FakeOptionDataClient(fail_times=1, fail_status=403)
    cfg2 = cfg
    broker = DPOFBroker(cfg2, trading_client=FakeTradingClient(), stock_data_client=object(),
                         option_data_client=option_client)

    async def _t():
        out = await broker.get_option_chain("SPY", "2026-09-25", "2026-09-25", "call")
        assert out is not None
        assert len(out) == 1
        assert option_client.calls == 2
        assert broker.feed_status()["options_feed_downgraded"] is True

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# Stock feed (SIP -> IEX)
# ---------------------------------------------------------------------------


def test_stock_feed_defaults_to_configured_value(cfg):
    broker = _broker(cfg)
    assert broker._cfg.stock_feed == "sip"
    from alpaca.data.enums import DataFeed

    assert broker._feed_for_stock() == DataFeed.SIP


def test_stock_feed_downgrades_on_401_and_retries_once(cfg, caplog):
    from datetime import datetime, timezone

    client = FakeStockDataClient(fail_times=1, fail_status=401)
    broker = _broker(cfg, stock_data_client=client)

    async def _t():
        with caplog.at_level(logging.WARNING, logger="clio.dpof.broker"):
            out = await broker.get_minute_bars("SPY", datetime.now(timezone.utc))
        assert out is not None
        assert client.calls == 2
        from alpaca.data.enums import DataFeed

        assert client.feeds_used[0] == DataFeed.SIP
        assert client.feeds_used[1] == DataFeed.IEX
        status = broker.feed_status()
        assert status["stock_feed_downgraded"] is True
        assert status["stock_feed_effective"] == "iex"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    asyncio.run(_t())


def test_stock_feed_iex_never_downgrades_further(cfg):
    """An account configured for iex has nothing further to fall back to --
    a 403 on iex itself is a genuine failure, not a feed-selection problem."""
    from dataclasses import replace as dc_replace
    from datetime import datetime, timezone

    iex_cfg = dc_replace(cfg, stock_feed="iex")
    client = FakeStockDataClient(fail_times=99, fail_status=403)
    broker = _broker(iex_cfg, stock_data_client=client)

    async def _t():
        out = await broker.get_minute_bars("SPY", datetime.now(timezone.utc))
        assert out is None
        assert client.calls == 1
        assert broker.feed_status()["stock_feed_downgraded"] is False

    asyncio.run(_t())


# ---------------------------------------------------------------------------
# update_config() reset semantics
# ---------------------------------------------------------------------------


def test_update_config_same_feed_value_keeps_downgrade(cfg):
    client = FakeOptionDataClient(fail_times=1, fail_status=403)
    broker = _broker(cfg, option_data_client=client)

    async def _t():
        await broker.get_option_quotes(["SPY261002C00575000"])
        assert broker.feed_status()["options_feed_downgraded"] is True
        broker.update_config(cfg)  # same options_feed value -- downgrade must persist
        assert broker.feed_status()["options_feed_downgraded"] is True

    asyncio.run(_t())


def test_update_config_changed_feed_value_resets_downgrade(cfg):
    from dataclasses import replace as dc_replace

    client = FakeOptionDataClient(fail_times=1, fail_status=403)
    broker = _broker(cfg, option_data_client=client)

    async def _t():
        await broker.get_option_quotes(["SPY261002C00575000"])
        assert broker.feed_status()["options_feed_downgraded"] is True
        broker.update_config(dc_replace(cfg, options_feed="indicative"))
        assert broker.feed_status()["options_feed_downgraded"] is False
        assert broker.feed_status()["options_feed_downgrade_reason"] == ""

    asyncio.run(_t())


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(pytest.main([__file__, "-q"]))
