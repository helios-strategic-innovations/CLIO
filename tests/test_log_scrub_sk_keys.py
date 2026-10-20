"""OP-03 (S214): bare sk- keys and key-bearing header text must not reach logs.

All keys below are fake (a/b characters only).
"""

import io
import logging
import py_compile

from agent.log_scrub import SecretScrubFilter, scrub_secrets

GW_KEY = "sk-" + "a" * 64
ANT_KEY = "sk-ant-" + "b" * 40


def test_bare_gateway_key_in_plain_text_redacted():
    out = scrub_secrets(f"calling provider with {GW_KEY} now")
    assert GW_KEY not in out
    assert "sk-REDACTED" in out


def test_bare_key_in_exception_text_redacted():
    out = scrub_secrets(str(ValueError(f"bad key {GW_KEY} rejected")))
    assert GW_KEY not in out


def test_ant_prefixed_key_redacted():
    out = scrub_secrets(f"anthropic said {ANT_KEY}")
    assert ANT_KEY not in out


def test_x_api_key_header_redacted():
    out = scrub_secrets(f"request headers x-api-key: {GW_KEY}")
    assert GW_KEY not in out
    assert "x-api-key: REDACTED" in out


def test_authorization_bearer_header_redacted():
    out = scrub_secrets(f"Authorization: Bearer {GW_KEY}")
    assert GW_KEY not in out
    assert "Authorization: REDACTED" in out


def test_lowercase_authorization_header_redacted():
    out = scrub_secrets(f"authorization: {GW_KEY}")
    assert GW_KEY not in out


def test_query_string_token_redacted():
    out = scrub_secrets(f"GET https://host/v1/x?token={GW_KEY}&limit=5")
    assert GW_KEY not in out
    assert "&limit=5" in out


def test_idempotent():
    samples = [
        f"plain {GW_KEY}",
        f"x-api-key: {GW_KEY} and Authorization: Bearer {ANT_KEY}",
        f"?token={GW_KEY}",
        '{"apiKey": "' + GW_KEY + '"}',
    ]
    for s in samples:
        once = scrub_secrets(s)
        assert scrub_secrets(once) == once


def test_non_strings_unchanged():
    assert scrub_secrets(None) is None
    assert scrub_secrets(42) == 42
    assert scrub_secrets("") == ""


def test_false_positive_guards_unchanged():
    for s in [
        "task-force-alpha-delta-omega-sierra-tango",
        "desk-lamp-with-a-very-long-descriptive-name",
        "risk-adjusted-return-metrics-per-quarter",
    ]:
        assert scrub_secrets(s) == s


def test_filter_redacts_record_message():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretScrubFilter())
    lg = logging.getLogger("test_op03_sk_keys")
    lg.handlers = [handler]
    lg.propagate = False
    lg.setLevel(logging.INFO)
    try:
        lg.info("upstream error for key %s", GW_KEY)
        handler.flush()
        text = stream.getvalue()
    finally:
        lg.handlers = []
    assert GW_KEY not in text
    assert "sk-REDACTED" in text


def test_api_key_header_lowercase_does_not_leak():
    secret = "q" * 24
    text = "api-key: " + secret
    assert secret not in scrub_secrets(text)


def test_api_key_header_uppercase_does_not_leak():
    secret = "q" * 24
    text = "API-KEY: " + secret
    assert secret not in scrub_secrets(text)


def test_x_api_key_header_non_sk_value_does_not_leak():
    secret = "z" * 32
    text = "X-API-KEY: " + secret
    assert secret not in scrub_secrets(text)


def test_authorization_bearer_non_sk_token_does_not_leak():
    secret = "abcdefghijklmnopqrstuvwx0123456789"
    text = "Authorization: Bearer " + secret
    assert secret not in scrub_secrets(text)


def test_main_py_compiles():
    py_compile.compile("main.py", doraise=True, cfile="/tmp/_op03_main.pyc")
