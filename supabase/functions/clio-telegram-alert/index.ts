// supabase/functions/clio-telegram-alert/index.ts
// Source: CLIO_PLAN_APIFY_SUPABASE_SOCIAL_INTEL_2026-05-24.md §6.1
//
// Lightweight forwarder: accepts {message, severity} from internal callers
// (initially pg_cron jobs in Phase 4), forwards to Telegram Bot API.
//
// S199 Pass-2 security hardening (fix branch fix/s199-pipeline-hardening — PREPARED FOR
// REVIEW, not deployed). Brings this function to parity with its hardened siblings
// (social-intel-ingest / social-intel-alert):
//   H2  fail-closed empty-secret guard — an unset/empty ALERT_TOKEN must NOT authenticate
//       `Bearer undefined` / `Bearer ` (the siblings already do `if (!SECRET || ...)`).
//   H3/H4  outer try/catch + redactSecrets — a network-level fetch THROW (TypeError)
//       embeds the request URL, and the Telegram bot token lives in the sendMessage URL
//       PATH (/bot<token>/), so an unhandled throw or echoed error body would leak it.
//       Mirrors social-intel-alert's /bot<token>/ path-redaction rule.

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";

const TELEGRAM_BOT_TOKEN = Deno.env.get("TELEGRAM_BOT_TOKEN")!;
const TELEGRAM_CHAT_ID = Deno.env.get("TELEGRAM_CHAT_ID")!;
const ALERT_TOKEN = Deno.env.get("ALERT_TOKEN")!;

// Redact the Telegram bot token (URL path /bot<token>/) and any ?secret=/?token= query
// params before anything is logged or returned. The bot token sits in the sendMessage URL
// path, so a thrown fetch error (which embeds the URL) would otherwise leak it.
function redactSecrets(s: string): string {
  return s
    .replace(/\/bot[^/]+\//gi, "/bot[REDACTED]/")
    .replace(/([?&](?:secret|token)=)[^&\s)"']+/gi, "$1[REDACTED]");
}

serve(async (req) => {
  try {
    if (req.method !== "POST") return new Response("method not allowed", { status: 405 });

    // Fail-closed auth (H2): an unset/empty ALERT_TOKEN must never authenticate.
    const auth = req.headers.get("authorization");
    if (!ALERT_TOKEN || auth !== `Bearer ${ALERT_TOKEN}`) {
      return new Response("unauthorized", { status: 401 });
    }

    let body: any;
    try { body = await req.json(); } catch { return new Response("bad json", { status: 400 }); }

    const { message, severity } = body;
    if (!message) return new Response("missing message", { status: 400 });

    const text = `[${severity || "INFO"}] ${message}`;

    // Per CLIO CLAUDE.md gotchas — DO NOT use parse_mode="Markdown"/"HTML" with
    // user-generated/dynamic content (causes BadRequest). Plain text only.
    const tgRes = await fetch(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: TELEGRAM_CHAT_ID,
        text: text,
      }),
    });

    if (!tgRes.ok) {
      const errText = await tgRes.text();
      // Log redacted detail server-side; return a GENERIC error to the caller (H3 — do not
      // echo the raw upstream body to whatever invoked the forwarder).
      console.error(redactSecrets(`Telegram send failed: ${tgRes.status} ${errText}`));
      return new Response(JSON.stringify({ status: "telegram_failed", code: tgRes.status }), { status: 502 });
    }

    return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
  } catch (err) {
    // H4: a network-level fetch throw embeds the bot-token URL in its message — redact
    // before logging, and never surface it to the caller.
    console.error(redactSecrets(`clio-telegram-alert error: ${String((err as any)?.message || err)}`));
    return new Response(JSON.stringify({ status: "error" }), { status: 500 });
  }
});
