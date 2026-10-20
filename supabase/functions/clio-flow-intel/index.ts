// supabase/functions/clio-flow-intel/index.ts
// RFC-CLIO-BRIDGE-v1 (P2B path A) — A1: CLIO read path (additive enriched lens).
//
// READ-ONLY. Serves CLIO's enriched social-intel substrate (tweet_signals ⋈ tweets ⋈ stories,
// status='ok') to Hydrinia's out-of-band mirror, selected by ticker | theme | entity | scope +
// since. The contract (request + response shape) is versioned and hides the schema: a future
// tweet_signals/tweets/stories migration changes only the SQL RPC body, not this contract — so
// it never ripples into Hydrinia (the #3 bridge-decoupling the operator chose Edge-Function for).
//
// ADVISORY WALL (RFC §2 / Antony): this function exposes ONLY a read endpoint. The single DB
// call is the SELECT-only RPC social_intel.get_flow_intel — there is NO insert/update/delete/
// upsert/rpc-write anywhere in this file. Hydrinia gets read access, never a write surface.
//
// Auth: CLIO_FLOW_READ_TOKEN via ?secret=... query param, fail-closed on an unset/empty token
// (mirrors social-intel-alert's STALENESS_CHECK_SECRET / social-intel-ingest's APIFY_WEBHOOK_SECRET).

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const CLIO_FLOW_READ_TOKEN = Deno.env.get("CLIO_FLOW_READ_TOKEN")!;

const CONTRACT = "RFC-CLIO-BRIDGE-v1";
const VERSION = 1;
const DEFAULT_LIMIT = 200;
const MAX_LIMIT = 500;
const DEFAULT_SINCE_HOURS = 72;

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { autoRefreshToken: false, persistSession: false },
});

// Redact ?secret=/?token= query params from any error string before it is logged or returned.
function redactSecrets(s: unknown): string {
  return String(s ?? "").replace(/([?&](?:secret|token)=)[^&\s)"']+/gi, "$1[REDACTED]");
}

function jsonResponse(obj: unknown, status: number): Response {
  return new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });
}

// Coerce a request field to a clean string[] (trims, drops blanks); non-array/empty → null
// so the RPC's "no selector" branch and per-selector cardinality()>0 guards behave correctly.
function toStrArray(v: unknown): string[] | null {
  if (!Array.isArray(v)) return null;
  const out = v.map((x) => String(x).trim()).filter((x) => x.length > 0);
  return out.length > 0 ? out : null;
}

serve(async (req) => {
  if (req.method !== "POST") return new Response("method not allowed", { status: 405 });

  // Fail-closed auth: an unset/empty token must never authenticate.
  const url = new URL(req.url);
  if (!CLIO_FLOW_READ_TOKEN || url.searchParams.get("secret") !== CLIO_FLOW_READ_TOKEN) {
    return jsonResponse({ error: "unauthorized" }, 401);
  }

  // Empty body is legitimate (→ broad recent lens). Only non-empty bodies must parse as JSON.
  let body: any = {};
  try {
    const raw = await req.text();
    if (raw && raw.trim().length > 0) body = JSON.parse(raw);
  } catch {
    return jsonResponse({ error: "bad json" }, 400);
  }

  // Bounded params. Tickers are upper-cased to match mapped_tickers (uppercase symbols);
  // themes/scope pass through; entity match is case-insensitive in the RPC.
  const tickers = toStrArray(body.tickers)?.map((t) => t.toUpperCase()) ?? null;
  const themes = toStrArray(body.themes);
  const entities = toStrArray(body.entities);
  const scope = toStrArray(body.scope);

  let since: string;
  if (typeof body.since === "string" && body.since.length > 0) {
    const d = new Date(body.since);
    if (isNaN(d.getTime())) return jsonResponse({ error: "invalid 'since' (expect ISO8601)" }, 400);
    since = d.toISOString();
  } else {
    since = new Date(Date.now() - DEFAULT_SINCE_HOURS * 3600 * 1000).toISOString();
  }

  const rawLimit = typeof body.limit === "number" && Number.isFinite(body.limit) ? Math.floor(body.limit) : DEFAULT_LIMIT;
  const limit = Math.min(Math.max(rawLimit, 1), MAX_LIMIT);
  const offset = typeof body.offset === "number" && Number.isFinite(body.offset) ? Math.max(Math.floor(body.offset), 0) : 0;
  const novelOnly = body.novel_only === undefined ? true : Boolean(body.novel_only);

  try {
    const { data, error } = await supabase
      .schema("social_intel")
      .rpc("get_flow_intel", {
        p_tickers: tickers,
        p_themes: themes,
        p_entities: entities,
        p_scope: scope,
        p_since: since,
        p_limit: limit,
        p_offset: offset,
        p_novel_only: novelOnly,
      });

    if (error) {
      console.error(redactSecrets(`get_flow_intel RPC error: ${error.message}`));
      return jsonResponse({ error: "query failed" }, 502);
    }

    const signals = data ?? [];
    return jsonResponse({
      contract: CONTRACT,
      version: VERSION,
      generated_at: new Date().toISOString(),
      query: { tickers, themes, entities, scope, since, limit, offset, novel_only: novelOnly },
      count: signals.length,
      // pagination hint: a full page MAY have more; a short page is the last page.
      next_offset: signals.length === limit ? offset + limit : null,
      signals,
    }, 200);
  } catch (err) {
    console.error(redactSecrets(`clio-flow-intel error: ${String((err as any)?.message || err)}`));
    return jsonResponse({ error: "internal error" }, 500);
  }
});
