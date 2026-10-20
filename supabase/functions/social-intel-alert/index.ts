// supabase/functions/social-intel-alert/index.ts
// S193 Phase 4.5 — observability layer for the social_intel pipeline.
// Invoked every 30 minutes by pg_cron via pg_net.http_post. Checks ingest_runs
// state for alert conditions; sends Telegram alerts with 24h idempotency cooldown
// per (alert_type, alert_key).
//
// Auth: STALENESS_CHECK_SECRET via ?secret=... query param. Same model as
// social-intel-ingest's APIFY_WEBHOOK_SECRET pattern.
//
// HTML safety (CLIO CLAUDE.md "parse_mode + dynamic content → BadRequest"): every
// dynamic field interpolated into the parse_mode:"HTML" body is run through esc()
// first. An unescaped '<', '>' or '&' in error_message / run_id would make Telegram
// reject the whole sendMessage with a 400 — silently losing the alert. Markdown is
// banned in CLIO for this reason; escaped HTML is the robust alternative.

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const STALENESS_CHECK_SECRET = Deno.env.get("STALENESS_CHECK_SECRET")!;
const TELEGRAM_BOT_TOKEN = Deno.env.get("TELEGRAM_BOT_TOKEN")!;
const TELEGRAM_ALERT_CHAT_ID = Deno.env.get("TELEGRAM_ALERT_CHAT_ID")!;

const COOLDOWN_HOURS = 24;
const STALENESS_THRESHOLD_HOURS = 24;
const STUCK_THRESHOLD_MINUTES = 30;
const RECENT_WINDOW_HOURS = 24;

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { autoRefreshToken: false, persistSession: false }
});

type Alert = { type: string; key: string; message: string };

// HTML-escape dynamic content before it enters a parse_mode:"HTML" message body.
// Telegram's HTML parser rejects the entire sendMessage (HTTP 400) on an unbalanced
// or stray '<' / '>' / '&'. Escaping these three is sufficient per the Bot API spec.
function esc(s: unknown): string {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

// Redact secrets from any error string before it enters the HTTP response body (captured
// by pg_net's net._http_response.content + Edge Function logs) or a console log. The
// Telegram bot token lives in the sendTelegram URL PATH (/bot<token>/sendMessage), so a
// network-level fetch THROW embeds the live token in the Deno TypeError — redact the path
// form. Also strip ?secret=/?token= query params defensively. (S196 P4.6 Pass-2 security:
// Pass-1 scoped redaction to the ingest fn and missed this alert-side path-token leak.)
function redactSecrets(s: unknown): string {
  return String(s ?? "")
    .replace(/\/bot[^/\s)"']+\//gi, "/bot[REDACTED]/")
    .replace(/([?&](?:secret|token)=)[^&\s)"']+/gi, "$1[REDACTED]");
}

async function sendTelegram(text: string): Promise<string> {
  const url = `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chat_id: TELEGRAM_ALERT_CHAT_ID,
      text,
      parse_mode: "HTML",
      disable_web_page_preview: true,
    }),
  });
  const body = await res.text();
  if (!res.ok) throw new Error(`Telegram API ${res.status}: ${body.slice(0, 500)}`);
  return body.slice(0, 500);
}

async function hasRecentAlert(alertType: string, alertKey: string): Promise<boolean> {
  const cutoff = new Date(Date.now() - COOLDOWN_HOURS * 3600 * 1000).toISOString();
  const { data, error } = await supabase
    .schema("social_intel")
    .from("alerts_log")
    .select("id")
    .eq("alert_type", alertType)
    .eq("alert_key", alertKey)
    .gte("sent_at", cutoff)
    .limit(1);
  if (error) throw new Error(`alerts_log SELECT error: ${error.message}`);
  return (data?.length ?? 0) > 0;
}

async function logAlert(alertType: string, alertKey: string, message: string, telegramResp: string): Promise<void> {
  const { error } = await supabase
    .schema("social_intel")
    .from("alerts_log")
    .insert({ alert_type: alertType, alert_key: alertKey, message, telegram_response: telegramResp });
  if (error) console.error(`alerts_log INSERT error: ${error.message}`);
}

async function checkStaleness(): Promise<Alert | null> {
  const { data, error } = await supabase
    .schema("social_intel")
    .from("ingest_runs")
    .select("completed_at")
    // LIVENESS = a run that actually LANDED data, not just a status. Count succeeded OR
    // partial_success (a drift-filtered run still landed the bulk), but REQUIRE
    // tweets_count > 0 — otherwise a 0-tweet run (empty Apify scrape → succeeded(0), or an
    // all-drift page → partial_success(0)) would mask a silently-dead pipeline for 48h.
    // (S196 P4.6 Pass-2: Pass-1's status-only liveness opened a 24-48h silent-death blind
    // zone — checkPartialSuccess only covers 24h, this window is 48h.) Drift itself is
    // surfaced separately by checkPartialSuccess.
    .in("status", ["succeeded", "partial_success"])
    .gt("tweets_count", 0)
    .gte("completed_at", new Date(Date.now() - STALENESS_THRESHOLD_HOURS * 3600 * 1000).toISOString())
    .limit(1);
  if (error) throw new Error(`staleness SELECT: ${error.message}`);
  if (data && data.length > 0) return null;  // recent data-landing run exists → alive
  const todayKey = new Date().toISOString().slice(0, 10);
  return {
    type: "staleness",
    key: todayKey,
    message: `<b>⚠ social_intel pipeline STALE</b>\nNo data-landing ingest in past ${STALENESS_THRESHOLD_HOURS}h (no runs, or all runs landed 0 usable tweets).\nCheck the Apify schedule, task input, and seed accounts.\nDashboard: ${esc(SUPABASE_URL)}/project/_/functions`,
  };
}

async function checkFailures(): Promise<Alert[]> {
  const cutoff = new Date(Date.now() - RECENT_WINDOW_HOURS * 3600 * 1000).toISOString();
  const { data, error } = await supabase
    .schema("social_intel")
    .from("ingest_runs")
    .select("run_id, error_message, completed_at")
    .eq("status", "failed")
    .gte("completed_at", cutoff)
    .not("run_id", "ilike", "phase4-smoketest-%");
  if (error) throw new Error(`failures SELECT: ${error.message}`);
  return (data ?? []).map(r => ({
    type: "failure",
    key: r.run_id,
    message: `<b>🔴 social_intel run FAILED</b>\nrunId: <code>${esc(r.run_id)}</code>\ncompleted: ${esc(r.completed_at)}\nerror: <code>${esc((r.error_message ?? '(empty)').slice(0, 400))}</code>`,
  }));
}

async function checkStuck(): Promise<Alert[]> {
  const cutoff = new Date(Date.now() - STUCK_THRESHOLD_MINUTES * 60 * 1000).toISOString();
  const { data, error } = await supabase
    .schema("social_intel")
    .from("ingest_runs")
    .select("run_id, started_at, error_message")
    .eq("status", "in_progress")
    .lt("started_at", cutoff)
    .not("run_id", "ilike", "phase4-smoketest-%");  // smoke-test rows never page (house pattern, S196 Pass-2)
  if (error) throw new Error(`stuck SELECT: ${error.message}`);
  return (data ?? []).map(r => ({
    type: "stuck",
    key: r.run_id,
    // Surface any in-progress error_message — e.g. a 'DRIFT DETECTED … ACTION REQUIRED'
    // recorded by a drift pass before the isolate died — so the actionable drift signal
    // is not downgraded to a generic 'stuck' notice. (S196 P4.6 observability fix.)
    message: `<b>🟡 social_intel run STUCK</b>\nrunId: <code>${esc(r.run_id)}</code>\nstarted: ${esc(r.started_at)} (>${STUCK_THRESHOLD_MINUTES}min ago)\nBackground isolate likely died. Idempotent re-fire to recover.${r.error_message ? `\nlast state: <code>${esc(String(r.error_message).slice(0, 300))}</code>` : ""}`,
  }));
}

// Intentional tripwire: 'aborted_by_wall_clock' is NOT written by the current Phase-4
// background-task design (a dead isolate leaves the row in_progress → checkStuck; MAX_PASSES
// marks failed — see ingest header). This check returns [] by design today; it exists to
// fire ONLY if a future regression re-introduces a wall-clock-abort status. Kept deliberately,
// not dead-by-accident. (S196 P4.6 Pass-2.)
async function checkAborted(): Promise<Alert[]> {
  const cutoff = new Date(Date.now() - RECENT_WINDOW_HOURS * 3600 * 1000).toISOString();
  const { data, error } = await supabase
    .schema("social_intel")
    .from("ingest_runs")
    .select("run_id, error_message, completed_at")
    .eq("status", "aborted_by_wall_clock")
    .gte("completed_at", cutoff)
    .not("run_id", "ilike", "phase4-smoketest-%");
  if (error) throw new Error(`aborted SELECT: ${error.message}`);
  return (data ?? []).map(r => ({
    type: "aborted_by_wall_clock",
    key: r.run_id,
    message: `<b>🟠 social_intel run ABORTED (wall-clock)</b>\nrunId: <code>${esc(r.run_id)}</code>\nPhase 2A-class IDLE_TIMEOUT — Phase 4 background-task pattern may have regressed.\nerror: <code>${esc((r.error_message ?? '(empty)').slice(0, 400))}</code>`,
  }));
}

// P4.6 (S195 Pass-2 follow-up): handle drift. social-intel-ingest now drops tweets
// from handles missing in social_intel.accounts (instead of FK-aborting the whole
// run) and marks the run 'partial_success'. Surface those loudly — most of the data
// landed, but drift exists and the operator must reconcile the handle list before
// the dropped handles silently accumulate. error_message already carries the
// 'DRIFT FILTERED — … ACTION REQUIRED' detail from the ingest finalizer.
async function checkPartialSuccess(): Promise<Alert[]> {
  const cutoff = new Date(Date.now() - RECENT_WINDOW_HOURS * 3600 * 1000).toISOString();
  const { data, error } = await supabase
    .schema("social_intel")
    .from("ingest_runs")
    .select("run_id, error_message, completed_at, tweets_count")
    .eq("status", "partial_success")
    .gte("completed_at", cutoff)
    .not("run_id", "ilike", "phase4-smoketest-%");  // smoke-test rows never page (house pattern, S196 Pass-2)
  if (error) throw new Error(`partial_success SELECT: ${error.message}`);
  return (data ?? []).map(r => ({
    type: "partial_success",
    key: r.run_id,
    message: `<b>⚠ social_intel run PARTIAL — ACTION REQUIRED</b>\nrunId: <code>${esc(r.run_id)}</code>\ntweets landed: ${esc(r.tweets_count ?? 0)}\ncompleted: ${esc(r.completed_at ?? '')}\n<code>${esc((r.error_message ?? '(empty)').slice(0, 400))}</code>`,
  }));
}

serve(async (req) => {
  if (req.method !== "POST") return new Response("method not allowed", { status: 405 });

  // Auth gate (fail-closed on an empty/unset secret so a misdeploy can't open the gate).
  const url = new URL(req.url);
  if (!STALENESS_CHECK_SECRET || url.searchParams.get("secret") !== STALENESS_CHECK_SECRET) {
    return new Response(JSON.stringify({ error: "unauthorized" }), { status: 401 });
  }

  const results = { checked: 0, sent: 0, suppressed_by_cooldown: 0, errors: [] as string[] };

  try {
    // Gather candidate alerts from all 4 checks. allSettled (not all) so a transient
    // failure in ONE check does not suppress real alerts from the others, nor 500 the
    // whole tick — mirrors the per-item graceful degradation the send loop below already
    // uses. A failed check is surfaced in results.errors (and Edge Function logs).
    const checks: [string, Promise<Alert | Alert[] | null>][] = [
      ["staleness", checkStaleness()],
      ["failures", checkFailures()],
      ["stuck", checkStuck()],
      ["aborted", checkAborted()],
      ["partial_success", checkPartialSuccess()],
    ];
    const settled = await Promise.allSettled(checks.map(c => c[1]));
    const candidates: Alert[] = [];
    settled.forEach((s, i) => {
      if (s.status === "fulfilled") {
        const v = s.value;
        if (v == null) return;
        if (Array.isArray(v)) candidates.push(...v);
        else candidates.push(v);
      } else {
        results.errors.push(`check:${checks[i][0]}: ${redactSecrets(String((s.reason as any)?.message || s.reason)).slice(0, 300)}`);
      }
    });
    results.checked = candidates.length;

    // Fire alerts that aren't in cooldown.
    for (const a of candidates) {
      try {
        if (await hasRecentAlert(a.type, a.key)) { results.suppressed_by_cooldown++; continue; }
        const tgResp = await sendTelegram(a.message);
        await logAlert(a.type, a.key, a.message, tgResp);
        results.sent++;
      } catch (e) {
        results.errors.push(`${a.type}/${a.key}: ${redactSecrets(String((e as any)?.message || e)).slice(0, 300)}`);
      }
    }

    return new Response(JSON.stringify({ ok: true, ...results }), {
      status: 200, headers: { "content-type": "application/json" },
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: redactSecrets(String((err as any)?.message || err)).slice(0, 500) }), { status: 500 });
  }
});
