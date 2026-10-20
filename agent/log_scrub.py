"""Secret scrubbing for log output (SEC-1, S211).

Some upstream APIs authenticate with a *query parameter* (Polygon: ``?apiKey=...``).
An ``httpx.HTTPStatusError`` puts the full request URL in ``str(e)``, so a plain
``logger.warning(f"...{e}")`` writes the key to the journal and to logs/openclaw.log.

This module gives one pure function and one ``logging.Filter``. The filter must be
attached to every *handler* (see ``logging_config.setup_logging``): a filter on the
root *logger* is not consulted for records emitted by child loggers.

Bare ``sk-...`` keys and ``x-api-key:`` / ``authorization:`` header text are also redacted
(OP-03, S214), because a key pasted into an exception message carries no ``name=`` prefix.
Scrubbing is a defense in depth, not a substitute for rotating a leaked key.
Pure stdlib -- safe to import from anywhere (no agent.config, no httpx).
"""

from __future__ import annotations

import logging
import re

_REDACTED = "REDACTED"

# name=value pairs (URL query strings, "key=value" log text)
_KV_RE = re.compile(
    r"(?i)\b(api[_-]?key|apikey|access[_-]?token|token|secret|password)=([^&\s'\")]+)"
)
# JSON-ish  "apiKey": "value"
_JSON_RE = re.compile(
    r"(?i)(\"(?:api[_-]?key|apikey|access[_-]?token|token|secret|password)\"\s*:\s*\")([^\"]+)(\")"
)
# Authorization: Bearer <token>
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}")
# Bare provider keys (OpenAI/gateway-style ``sk-...``, Anthropic ``sk-ant-...``).
# The leading \b keeps words such as "task-force" or "desk-lamp" untouched.
_SK_RE = re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}")
# Header text: ``x-api-key: <v>``, ``authorization: Bearer <v>``, ``authorization: <v>``
_HEADER_RE = re.compile(
    r"(?i)\b(x-api-key|api-key|authorization)\s*:\s*(?:Bearer\s+)?[^\s,;'\")]+"
)


def scrub_secrets(text):
    """Return ``text`` with credential-looking values replaced by ``REDACTED``.

    Non-strings are returned unchanged. Idempotent.
    """
    if not isinstance(text, str) or not text:
        return text
    out = _HEADER_RE.sub(lambda m: f"{m.group(1)}: {_REDACTED}", text)
    out = _BEARER_RE.sub(f"Bearer {_REDACTED}", out)
    out = _SK_RE.sub("sk-REDACTED", out)
    out = _KV_RE.sub(lambda m: f"{m.group(1)}={_REDACTED}", out)
    out = _JSON_RE.sub(lambda m: f"{m.group(1)}{_REDACTED}{m.group(3)}", out)
    return out


class SecretScrubFilter(logging.Filter):
    """Scrub the message, exception text and stack text of every record.

    Fails closed on content: if scrubbing itself raises, the record is replaced by a
    fixed marker line rather than emitted unscrubbed. Never raises, never drops a record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_secrets_scrubbed", False):
            return True
        try:
            msg = record.getMessage()
            scrubbed = scrub_secrets(msg)
            if scrubbed != msg:
                record.msg = scrubbed
                record.args = ()
            if record.exc_info and not record.exc_text:
                record.exc_text = scrub_secrets(
                    logging.Formatter().formatException(record.exc_info)
                )
            elif record.exc_text:
                record.exc_text = scrub_secrets(record.exc_text)
            if record.stack_info:
                record.stack_info = scrub_secrets(record.stack_info)
        except Exception as e:  # noqa: BLE001 - a log filter must never raise
            record.msg = f"[log record withheld: scrub failed ({type(e).__name__})]"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        record._secrets_scrubbed = True
        return True
