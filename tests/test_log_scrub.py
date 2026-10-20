"""SEC-1 (S211): credentials must never reach a log handler.

Polygon authenticates with ``?apiKey=`` and an httpx error string embeds the URL, so a
plain ``logger.warning(f"...{e}")`` leaked the key (355 journal lines in 7 days).
All keys below are fake.
"""

import io
import logging

import httpx

from agent import logging_config
from agent.log_scrub import SecretScrubFilter, scrub_secrets

FAKE = "sk_test_FAKEKEY_1234567890"


def _logger_with_filter(name):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(SecretScrubFilter())
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, stream


def test_scrubs_url_query_key():
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/SPY?apiKey={FAKE}&x=1"
    out = scrub_secrets(f"Client error '403 Forbidden' for url '{url}'")
    assert FAKE not in out
    assert "apiKey=REDACTED&x=1" in out
    assert out.endswith("x=1'")  # the closing quote survives


def test_scrubs_variants_and_json_and_bearer():
    assert FAKE not in scrub_secrets(f"api_key={FAKE}")
    assert FAKE not in scrub_secrets(f"API-KEY={FAKE}")
    assert FAKE not in scrub_secrets(f"access_token={FAKE} ok")
    assert FAKE not in scrub_secrets(f'{{"apiKey": "{FAKE}", "n": 1}}')
    assert FAKE not in scrub_secrets(f"Authorization: Bearer {FAKE}")
    assert scrub_secrets(f"password={FAKE}").startswith("password=REDACTED")


def test_no_false_positives_on_ordinary_log_text():
    for text in (
        "max_tokens=1000 input_tokens=5 tokens=7",
        "monkey=1 turkey=2 key=abc",
        "status=ok price=12.5 symbol=SPY",
        "",
    ):
        assert scrub_secrets(text) == text


def test_idempotent_and_non_string_passthrough():
    once = scrub_secrets(f"apiKey={FAKE}")
    assert scrub_secrets(once) == once
    assert scrub_secrets(None) is None
    assert scrub_secrets(123) == 123


def test_filter_scrubs_message_args_and_exception_text():
    logger, stream = _logger_with_filter("test_log_scrub.filter")
    url = f"https://api.polygon.io/v2/x?apiKey={FAKE}"
    logger.warning("call failed for %s", url)
    try:
        request = httpx.Request("GET", url)
        raise httpx.HTTPStatusError(
            f"Client error '403 Forbidden' for url '{url}'",
            request=request,
            response=httpx.Response(403, request=request),
        )
    except httpx.HTTPStatusError:
        logger.error("boom", exc_info=True)
    out = stream.getvalue()
    assert FAKE not in out
    assert out.count("apiKey=REDACTED") >= 2  # message arg + traceback line


def test_filter_never_raises_and_never_drops():
    class Boom:
        def __str__(self):
            raise RuntimeError("cannot stringify")

    logger, stream = _logger_with_filter("test_log_scrub.failclosed")
    logger.warning("value: %s", Boom())  # logging swallows its own format error
    filt = SecretScrubFilter()
    record = logging.LogRecord("t", logging.WARNING, __file__, 1, "x %s", (Boom(),), None)
    assert filt.filter(record) is True
    assert "scrub failed" in record.msg


def test_setup_logging_attaches_filter_to_every_handler(monkeypatch):
    """setup_logging hard-codes the LIVE log dir; patch it so this test writes nothing there."""
    made = []

    def fake_rotating(*args, **kwargs):
        h = logging.StreamHandler(io.StringIO())
        made.append(h)
        return h

    monkeypatch.setattr(logging_config, "RotatingFileHandler", fake_rotating)
    monkeypatch.setattr(logging_config.os, "makedirs", lambda *a, **k: None)

    root = logging.getLogger()
    sec_logger = logging.getLogger("openclaw.security")
    saved_root, saved_level = root.handlers[:], root.level
    saved_sec = sec_logger.handlers[:]
    root.handlers = []
    try:
        logging_config.setup_logging("INFO")
        handlers = list(root.handlers) + [h for h in sec_logger.handlers if h not in saved_sec]
        assert len(handlers) == 3  # console, file, security
        for h in handlers:
            assert any(isinstance(f, SecretScrubFilter) for f in h.filters), h
        # end to end: a leaked key emitted through the configured file handler is scrubbed
        logging.getLogger("openclaw.market_events").warning("failed url=https://x/y?apiKey=%s", FAKE)
        written = "".join(h.stream.getvalue() for h in made)
        assert FAKE not in written and "apiKey=REDACTED" in written
    finally:
        root.handlers = saved_root
        root.setLevel(saved_level)
        sec_logger.handlers = saved_sec


def test_market_events_fetch_snapshot_logs_scrubbed(monkeypatch, caplog):
    import asyncio

    from agent.integrations import market_events, market_snapshot

    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/SPY?apiKey={FAKE}"
    request = httpx.Request("GET", url)

    async def boom(symbol):
        raise httpx.HTTPStatusError(
            f"Client error '403 Forbidden' for url '{url}'",
            request=request,
            response=httpx.Response(403, request=request),
        )

    monkeypatch.setattr(market_snapshot, "get_snapshot_raw", boom)
    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(market_events._fetch_snapshot("SPY")) is None
    assert FAKE not in caplog.text
    assert "apiKey=REDACTED" in caplog.text
