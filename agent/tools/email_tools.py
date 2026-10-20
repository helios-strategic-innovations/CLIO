"""
OpenClaw/Moltbot — Email Tools (IMAP Read, SMTP Send, Trade Summary)

Provides handle_read_emails, handle_send_email, handle_email_trade_summary
for the Claude tool-use loop. Uses aioimaplib (async IMAP) and aiosmtplib
(async SMTP) with defense-in-depth security.
"""

import asyncio
import email
import logging
import re
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any

import aioimaplib
import aiosmtplib

from agent.config import config, EmailAccount
from agent.rate_limiter import check_and_record
from agent.security import log_security_event, sanitize_ingested_content

logger = logging.getLogger("openclaw.email")

# ============================================================
# Constants
# ============================================================

MAX_BODY_SIZE = 50_000          # 50KB max outgoing email body
MAX_RESULTS = 20                # Max emails per read request
MAX_SUBJECT_LENGTH = 200        # Subject line cap
MAX_RECIPIENTS = 5              # Max To + CC recipients per email
DEFAULT_SEND_ACCOUNT = "bot_atlas"
DEFAULT_READ_ACCOUNT = "personal_gmail"
SEND_TIMEOUT = 30               # seconds
IMAP_TIMEOUT = 30               # seconds

# ============================================================
# Security helpers
# ============================================================

_HEADER_INJECTION_RE = re.compile(r"[\r\n]")


def _sanitize_header(value: str, field_name: str) -> str:
    """Strip CR/LF from header values to prevent header injection."""
    clean = _HEADER_INJECTION_RE.sub("", value)
    if clean != value:
        log_security_event("email_header_injection_attempt", field=field_name)
        logger.warning(f"Header injection blocked in {field_name}")
    return clean.strip()


def _validate_email_address(addr: str) -> bool:
    """Basic email format validation."""
    return bool(re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", addr))


def _resolve_account(account_name: str | None, need_send: bool = False) -> tuple[EmailAccount | None, str]:
    """Resolve account name to EmailAccount. Returns (account, error_msg)."""
    if not account_name:
        account_name = DEFAULT_SEND_ACCOUNT if need_send else DEFAULT_READ_ACCOUNT

    acct = config.email_accounts.get(account_name)
    if not acct:
        available = ", ".join(config.email_accounts.keys()) or "(none configured)"
        return None, f"Unknown account '{account_name}'. Available: {available}"

    if not acct.address:
        return None, f"Account '{account_name}' has no email address configured."

    if not acct.password:
        return None, f"Account '{account_name}' has no password configured."

    if need_send and not acct.can_send:
        return None, f"Account '{account_name}' is read-only (can_send=False). Use bot_atlas or bot_gmail."

    return acct, ""


def _extract_body(msg: email.message.Message) -> str:
    """Extract plain text body from an email message."""
    if msg.is_multipart():
        # Prefer text/plain
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")
        # Fallback: strip HTML
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    text = payload.decode("utf-8", errors="replace")
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()
                    return text
        return "(no readable body)"
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="replace")
    return "(empty)"


# ============================================================
# IMAP Read
# ============================================================

async def handle_read_emails(p: dict) -> str:
    """Read emails via IMAP. Supports folder, search, unread-only, account selection."""
    await check_and_record("email_read")

    account_name = p.get("account")
    acct, err = _resolve_account(account_name, need_send=False)
    if err:
        return f"ERROR: {err}"

    # Sanitize folder to safe Gmail IMAP names
    folder_raw = (p.get("folder") or "inbox").lower()
    folder_map = {
        "inbox": "INBOX",
        "sent": '"[Gmail]/Sent Mail"',
        "flagged": '"[Gmail]/Starred"',
    }
    folder = folder_map.get(folder_raw, "INBOX")

    try:
        limit = min(int(p.get("limit", 10)), MAX_RESULTS)
    except (ValueError, TypeError):
        limit = 10
    unread_only = p.get("unread_only", False)
    search_query = p.get("search_query", "")

    imap_client = None
    try:
        imap_client = aioimaplib.IMAP4_SSL(
            host=acct.imap_host, port=acct.imap_port, timeout=IMAP_TIMEOUT
        )
        await imap_client.wait_hello_from_server()
        resp = await imap_client.login(acct.address, acct.password)
        if resp.result != "OK":
            return f"ERROR: IMAP login failed for {acct.name}. Check App Password."

        await imap_client.select(folder)

        # Build IMAP search criteria
        criteria_parts = []
        if unread_only:
            criteria_parts.append("UNSEEN")
        if search_query:
            safe_query = _HEADER_INJECTION_RE.sub("", search_query)
            safe_query = safe_query.replace("\\", "").replace('"', "")[:100]
            # Search in both subject and from
            criteria_parts.append(f'OR SUBJECT "{safe_query}" FROM "{safe_query}"')
        if not criteria_parts:
            criteria_parts.append("ALL")

        search_str = " ".join(criteria_parts)
        _, data = await imap_client.search(search_str)

        if not data or not data[0]:
            await imap_client.logout()
            return f"No emails found in {folder_raw} for {acct.address}."

        message_ids = data[0].split()
        # Take most recent N
        message_ids = message_ids[-limit:]

        results = []
        for mid in reversed(message_ids):  # Newest first
            mid_str = mid.decode() if isinstance(mid, bytes) else str(mid)
            try:
                _, msg_data = await imap_client.fetch(mid_str, "(RFC822)")
                if not msg_data or len(msg_data) < 2:
                    continue
                raw = msg_data[1]
                if isinstance(raw, tuple):
                    raw = raw[1] if len(raw) > 1 else raw[0]
                if isinstance(raw, bytes):
                    msg = email.message_from_bytes(raw)
                else:
                    continue
            except Exception as e:
                logger.debug(f"Fire-and-forget email parse failed: {e}")
                continue

            body = _extract_body(msg)
            if len(body) > 2000:
                body = body[:2000] + "... [truncated]"

            results.append({
                "from": msg.get("From", ""),
                "to": msg.get("To", ""),
                "subject": msg.get("Subject", "(no subject)"),
                "date": msg.get("Date", ""),
                "body": body,
                "id": mid_str,
            })

        await imap_client.logout()

        if not results:
            return f"No emails found in {folder_raw} for {acct.address}."

        # Format output
        output_lines = [f"{len(results)} email(s) from {acct.address} ({folder_raw}):\n"]
        for i, r in enumerate(results, 1):
            output_lines.append(
                f"--- Email {i} ---\n"
                f"From: {r['from']}\n"
                f"To: {r['to']}\n"
                f"Subject: {r['subject']}\n"
                f"Date: {r['date']}\n"
                f"Body:\n{r['body']}\n"
            )
        raw_output = "\n".join(output_lines)

        # Wrap as untrusted ingested content
        return sanitize_ingested_content(raw_output, f"email:{acct.name}:{folder_raw}")

    except asyncio.TimeoutError:
        return f"ERROR: IMAP timeout for {acct.name} after {IMAP_TIMEOUT}s."
    except Exception as e:
        logger.error(f"Email read error ({acct.name}): {e}", exc_info=True)
        return f"ERROR: Failed to read emails from {acct.name}: {str(e)[:200]}"
    finally:
        if imap_client:
            try:
                await imap_client.logout()
            except Exception as e:
                logger.debug(f"Fire-and-forget IMAP logout failed: {e}")


# ============================================================
# SMTP Send
# ============================================================

async def handle_send_email(p: dict) -> str:
    """Send email via SMTP. Requires confirmation (enforced by security.py)."""
    await check_and_record("email_send")

    account_name = p.get("from_account", DEFAULT_SEND_ACCOUNT)
    acct, err = _resolve_account(account_name, need_send=True)
    if err:
        return f"ERROR: {err}"

    to_raw = p.get("to", "")
    subject = p.get("subject", "")
    body = p.get("body", "")
    cc_raw = p.get("cc", "")

    # --- Validation ---
    if not to_raw or not subject or not body:
        return "ERROR: 'to', 'subject', and 'body' are all required."

    to_list = [addr.strip() for addr in to_raw.split(",") if addr.strip()]
    cc_list = [addr.strip() for addr in cc_raw.split(",") if addr.strip()] if cc_raw else []
    all_recipients = to_list + cc_list

    if len(all_recipients) > MAX_RECIPIENTS:
        return f"ERROR: Max {MAX_RECIPIENTS} recipients allowed (got {len(all_recipients)})."

    for addr in all_recipients:
        if not _validate_email_address(addr):
            return f"ERROR: Invalid email address: {addr}"

    # Sanitize headers
    subject = _sanitize_header(subject, "subject")
    if len(subject) > MAX_SUBJECT_LENGTH:
        subject = subject[:MAX_SUBJECT_LENGTH]

    if len(body) > MAX_BODY_SIZE:
        return f"ERROR: Body exceeds {MAX_BODY_SIZE} character limit."

    # --- Build message ---
    msg = EmailMessage()
    msg["From"] = acct.address
    msg["To"] = ", ".join(to_list)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg.set_content(body)

    # --- Send ---
    try:
        await aiosmtplib.send(
            msg,
            hostname=acct.smtp_host,
            port=acct.smtp_port,
            start_tls=True,
            username=acct.address,
            password=acct.password,
            timeout=SEND_TIMEOUT,
        )

        log_security_event(
            "email_sent",
            from_account=acct.name,
            to=to_raw,
            cc=cc_raw or "(none)",
            subject=subject[:50],
        )
        logger.info(f"Email sent from {acct.name} to {to_raw}, subject: {subject[:50]}")
        return f"Email sent from {acct.address} to {to_raw}. Subject: {subject}"

    except aiosmtplib.SMTPAuthenticationError:
        logger.error(f"SMTP auth failed for {acct.name}")
        return f"ERROR: SMTP authentication failed for {acct.name}. Check App Password."
    except asyncio.TimeoutError:
        return f"ERROR: SMTP timeout for {acct.name} after {SEND_TIMEOUT}s."
    except Exception as e:
        logger.error(f"SMTP error ({acct.name}): {e}", exc_info=True)
        return f"ERROR: Failed to send from {acct.name}: {str(e)[:200]}"


# ============================================================
# Trade Summary Email
# ============================================================

async def handle_email_trade_summary(p: dict) -> str:
    """Build and send a CLIO DPOF trade summary email from the CLIO-native
    Alpaca paper account and the DPOF readmodel (agent/dpof/readmodel.py).

    Hydrinia is permanently retired (operator direction, 2026-09-24) — this
    no longer reads its frozen state files.
    """
    await check_and_record("email_send")

    account_name = p.get("from_account", DEFAULT_SEND_ACCOUNT)
    acct, err = _resolve_account(account_name, need_send=True)
    if err:
        return f"ERROR: {err}"

    to_raw = p.get("to", "")
    if not to_raw:
        return "ERROR: 'to' recipient required for trade summary."

    to_list = [addr.strip() for addr in to_raw.split(",") if addr.strip()]
    if len(to_list) > MAX_RECIPIENTS:
        return f"ERROR: Max {MAX_RECIPIENTS} recipients."
    for addr in to_list:
        if not _validate_email_address(addr):
            return f"ERROR: Invalid email: {addr}"

    summary_type = p.get("summary_type", "daily")
    summary_parts = []

    # CLIO's own Alpaca paper account — day P&L = equity - last_equity
    # (CLAUDE.md: "Daily P&L ... equity - last_equity gives true daily P&L").
    try:
        from agent.integrations.alpaca import _get_trading_client, _retry_alpaca_call
        client = _get_trading_client()
        account = await _retry_alpaca_call(client.get_account)
        equity = float(account.equity)
        last_equity = float(account.last_equity) if account.last_equity is not None else None
        summary_parts.append("=== Account (Alpaca paper) ===")
        summary_parts.append(f"  Equity: ${equity:,.2f}")
        if last_equity is not None:
            summary_parts.append(f"  Day P&L (equity - last_equity): ${equity - last_equity:+,.2f}")
        summary_parts.append(f"  Buying Power: ${float(account.buying_power):,.2f}")
    except Exception as e:
        logger.warning(f"Trade summary: Alpaca account fetch failed: {e}")
        summary_parts.append("=== Account: unavailable ===")

    # CLIO's own DPOF book (agent/dpof/readmodel.py — sync, read-only, never raises)
    try:
        from agent.dpof.readmodel import snapshot
        snap = snapshot()
        summary_parts.append(f"\n=== DPOF (mode={snap.get('mode', 'unknown')}) ===")
        open_lines = snap.get("open_lines") or []
        if open_lines:
            summary_parts.append(f"Open Positions ({snap.get('open_count', 0)}):")
            summary_parts.extend(f"  {line}" for line in open_lines)
        else:
            summary_parts.append("No open positions.")
        closed_lines = snap.get("closed_today_lines") or []
        if closed_lines:
            summary_parts.append(
                f"\nClosed Today ({snap.get('closed_today_count', 0)}, "
                f"{snap.get('closed_today_wins', 0)} wins):"
            )
            summary_parts.extend(f"  {line}" for line in closed_lines)
        summary_parts.append(
            f"\nRealized P&L today: ${snap.get('realized_pnl_today', 0.0):+,.2f}  "
            f"week: ${snap.get('realized_pnl_week', 0.0):+,.2f}"
        )
        if snap.get("hard_halted") or snap.get("halt_file"):
            summary_parts.append(
                f"HALTED: hard_halted={snap.get('hard_halted')} halt_file={snap.get('halt_file')} "
                f"reason={snap.get('halt_reason') or 'n/a'}"
            )
    except Exception as e:
        logger.warning(f"Trade summary: DPOF readmodel unavailable: {e}")
        summary_parts.append("\n=== DPOF: unavailable ===")

    body = "\n".join(summary_parts) or "No data available."
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject = f"CLIO DPOF {summary_type.title()} Summary — {now_str}"

    msg = EmailMessage()
    msg["From"] = acct.address
    msg["To"] = ", ".join(to_list)
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        await aiosmtplib.send(
            msg,
            hostname=acct.smtp_host,
            port=acct.smtp_port,
            start_tls=True,
            username=acct.address,
            password=acct.password,
            timeout=SEND_TIMEOUT,
        )
        log_security_event(
            "trade_summary_sent",
            from_account=acct.name,
            to=to_raw,
            summary_type=summary_type,
        )
        logger.info(f"Trade summary sent from {acct.name} to {to_raw}")
        return f"Trade summary sent from {acct.address} to {to_raw}."

    except aiosmtplib.SMTPAuthenticationError:
        return f"ERROR: SMTP auth failed for {acct.name}. Check App Password."
    except asyncio.TimeoutError:
        return f"ERROR: SMTP timeout for {acct.name} after {SEND_TIMEOUT}s."
    except Exception as e:
        logger.error(f"Trade summary send error: {e}", exc_info=True)
        return f"ERROR: Failed to send trade summary: {str(e)[:200]}"
