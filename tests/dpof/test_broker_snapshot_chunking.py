"""Option snapshot requests must respect Alpaca's 100-symbol limit (2026-09-28).

get_option_chain asked for every contract's snapshot in ONE request; Alpaca answered
{"message":"symbol limit is 100"} for SPY/QQQ chains (hundreds of contracts), the error
was swallowed, every quote came back without bid/ask/delta, and every nomination ended
"no_candidates" from go-live until the fix. The fake below enforces the real limit, which
the earlier fakes did not.
"""

import asyncio

from agent.dpof.broker import DPOFBroker

_LIMIT = 100


class _APIError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class _Quote:
    def __init__(self, bid, ask):
        self.bid_price, self.ask_price = bid, ask


class _Greeks:
    def __init__(self, delta):
        self.delta, self.theta = delta, -0.05


class _Snap:
    def __init__(self, i):
        self.latest_quote = _Quote(1.0 + i / 1000, 1.1 + i / 1000)
        self.greeks = _Greeks(0.4)
        self.implied_volatility = 0.2


class LimitedOptionDataClient:
    """Behaves like Alpaca's option snapshot endpoint: more than 100 symbols is a 400.
    `fail_on` makes every request containing that symbol fail (422: not retried), so one
    chunk fails for good while the others succeed."""

    def __init__(self, fail_on=None):
        self.request_sizes = []
        self._fail_on = fail_on

    def get_option_snapshot(self, req):
        symbols = req.symbol_or_symbols
        symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        self.request_sizes.append(len(symbols))
        if len(symbols) > _LIMIT:
            raise _APIError(400, '{"message":"symbol limit is 100"}')
        if self._fail_on is not None and self._fail_on in symbols:
            raise _APIError(422, "unprocessable")
        return {s: _Snap(i) for i, s in enumerate(symbols)}


class _Contract:
    def __init__(self, i):
        self.symbol = f"SPY260929C{500000 + i * 1000:08d}"
        self.expiration_date = "2026-09-29"
        self.strike_price = str(500 + i)
        self.open_interest = 100


class _ContractsResp:
    def __init__(self, contracts, token):
        self.option_contracts = contracts
        self.next_page_token = token


class PagedTradingClient:
    """get_option_contracts returns `n` contracts in pages of 100."""

    def __init__(self, n):
        self._contracts = [_Contract(i) for i in range(n)]

    def get_option_contracts(self, req):
        start = int(req.page_token or 0)
        page = self._contracts[start:start + 100]
        nxt = start + 100
        return _ContractsResp(page, str(nxt) if nxt < len(self._contracts) else None)


def _broker(cfg, option_client, n_contracts=0):
    return DPOFBroker(cfg, trading_client=PagedTradingClient(n_contracts), stock_data_client=object(),
                      option_data_client=option_client)


def test_chain_of_250_contracts_is_quoted_in_chunks_of_at_most_100(cfg):
    client = LimitedOptionDataClient()
    broker = _broker(cfg, client, n_contracts=250)
    chain = asyncio.run(broker.get_option_chain("SPY", "2026-09-29", "2026-09-29", "call"))
    assert chain is not None and len(chain) == 250
    assert client.request_sizes == [100, 100, 50]
    assert all(q.bid is not None and q.ask is not None and q.delta is not None for q in chain)


def test_quotes_for_150_symbols_are_all_returned(cfg):
    client = LimitedOptionDataClient()
    broker = _broker(cfg, client)
    symbols = [f"SPY260929P{500000 + i * 1000:08d}" for i in range(150)]
    quotes = asyncio.run(broker.get_option_quotes(symbols))
    assert quotes is not None and set(quotes) == set(symbols)
    assert client.request_sizes == [100, 50]


def test_exactly_100_symbols_is_one_request(cfg):
    client = LimitedOptionDataClient()
    broker = _broker(cfg, client, n_contracts=100)
    chain = asyncio.run(broker.get_option_chain("SPY", "2026-09-29", "2026-09-29", "call"))
    assert len(chain) == 100 and client.request_sizes == [100]


def test_a_failed_chunk_makes_the_chain_unavailable_not_illiquid(cfg):
    """A partial or quote-less chain would read as illiquid contracts ("no_candidates");
    an unavailable chain is retried next cycle."""
    client = LimitedOptionDataClient(fail_on=_Contract(150).symbol)  # second chunk
    broker = _broker(cfg, client, n_contracts=250)
    assert asyncio.run(broker.get_option_chain("SPY", "2026-09-29", "2026-09-29", "call")) is None
    assert client.request_sizes[:2] == [100, 100]  # stopped at the failed chunk


def test_a_failed_chunk_makes_quotes_unavailable_not_missing(cfg):
    """A lookup failure is never "not found": a partial dict would make the failed
    chunk's symbols look absent to the exit/reconcile paths."""
    symbols = [f"SPY260929P{500000 + i * 1000:08d}" for i in range(150)]
    client = LimitedOptionDataClient(fail_on=symbols[0])  # first chunk
    broker = _broker(cfg, client)
    assert asyncio.run(broker.get_option_quotes(symbols)) is None
