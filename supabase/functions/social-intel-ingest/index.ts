// supabase/functions/social-intel-ingest/index.ts
// Source: CLIO_PLAN_APIFY_SUPABASE_SOCIAL_INTEL_2026-05-24.md §5.2
//         + S193 Phase 2A (handle lowercase) + S193 Phase 4 (background-task + paged continuation)
//
// Apify webhook → 200 in <2s → background task (EdgeRuntime.waitUntil) processes
// the dataset in OFFSET-PAGED passes. Each pass embeds + upserts a bounded window
// of MAX_PER_PASS tweets (≈120s of HNSW index insertion, safely under the ~300s
// background-isolate budget), then dispatches the next pass to a fresh isolate.
//
// Why paging: the bottleneck is incremental HNSW vector-index maintenance on
// insert — `tweets` has TWO HNSW indexes (embedding 1536-dim + embedding_512
// 512-dim, m=16/ef_construction=64). Measured ~0.37s/row. A full 3000-tweet run
// = ~19min of insert, which exceeds any single isolate's lifetime. OpenAI
// embedding is NOT the bottleneck (100 embeds in ~0.7s, 5000 RPM / 5M TPM limits).
// Atlas Team consensus 4-0 Option B (background-task); paging is the §13.4
// mitigation, implemented after Phase 4's first real-data replay confirmed the
// isolate dies at ~800 rows.
//
// Canonical operator surface for run outcomes is social_intel.ingest_runs:
//   status ∈ {in_progress, succeeded, partial_success, failed}
//     partial_success = some unseeded-handle tweets were drift-filtered (P4.6).
//     'aborted_by_wall_clock' is RESERVED — the Phase-4 background-task design never
//     writes it (isolate death leaves the row in_progress; MAX_PASSES marks failed).
//   error_message captures failures + drift detail (background errors otherwise swallowed).
// Idempotent: ON CONFLICT (tweet_id) DO NOTHING — replays / overlapping pages add
// zero duplicate rows.

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// EdgeRuntime is a Supabase Edge Runtime global; declared for the type checker.
// deno-lint-ignore no-explicit-any
declare const EdgeRuntime: any;

const APIFY_TOKEN = Deno.env.get("APIFY_TOKEN")!;
const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY")!;
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const WEBHOOK_SECRET = Deno.env.get("APIFY_WEBHOOK_SECRET")!;

const EMBED_BATCH_SIZE = 100;   // OpenAI text-embedding-3-small accepts array input
const UPSERT_CHUNK_SIZE = 100;  // PostgREST body size + per-chunk HNSW cost
const MAX_PER_PASS = 300;       // tweets per isolate pass (~120s of HNSW insert)
const MAX_PASSES = 30;          // safety cap: 30 × 300 = 9000 >> any real run

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { autoRefreshToken: false, persistSession: false }
});

// Redact secret/token query params from any string before it is logged, persisted to
// ingest_runs.error_message, or forwarded to Telegram. A network-level fetch THROW
// (TypeError — distinct from the !res.ok path) embeds the FULL request URL, including
// ?secret=…/?token=…, in the error text; that string flows through markFailed →
// error_message → social-intel-alert → operator Telegram + alerts_log. Without this the
// live APIFY_WEBHOOK_SECRET / APIFY_TOKEN would land in plaintext across two DB columns,
// the function logs, and Telegram history. (S196 P4.6 security review.)
function redactSecrets(s: string): string {
  return s.replace(/([?&](?:secret|token)=)[^&\s)"']+/gi, "$1[REDACTED]");
}

async function fetchApifyRunDetails(runId: string): Promise<any> {
  const url = `https://api.apify.com/v2/actor-runs/${runId}?token=${APIFY_TOKEN}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Apify run details fetch failed: ${res.status} ${await res.text()}`);
  return (await res.json()).data;
}

// Offset-paged dataset fetch — only pulls the window this pass needs.
async function fetchApifyDatasetPage(datasetId: string, offset: number, limit: number): Promise<any[]> {
  const url = `https://api.apify.com/v2/datasets/${datasetId}/items` +
    `?token=${APIFY_TOKEN}&format=json&clean=true&offset=${offset}&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Apify dataset page fetch failed: ${res.status} ${await res.text()}`);
  return await res.json();
}

async function embedBatch(texts: string[]): Promise<{ full: number[]; small: number[] }[]> {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: "text-embedding-3-small",
      input: texts.map(t => t.slice(0, 8000)),
      dimensions: 1536,
    }),
  });
  if (!res.ok) throw new Error(`OpenAI embed failed: ${res.status} ${await res.text()}`);
  const json = await res.json();
  const sorted = (json.data as any[]).slice().sort((a, b) => a.index - b.index);
  // Misalignment guard (S193 Pass-2 B2): OpenAI returns exactly one embedding per
  // input or errors the whole call — but assert it. A silent partial batch would
  // shift every subsequent row's embedding onto the wrong tweet (undetectable by a
  // null-scan: vectors stay non-null + unit-norm, just attached to the wrong tweet).
  // Fail loud instead of poisoning semantic search.
  if (sorted.length !== texts.length) {
    throw new Error(`OpenAI returned ${sorted.length} embeddings for ${texts.length} inputs — aborting (misalignment guard)`);
  }
  return sorted.map((d: any) => {
    const full = d.embedding as number[];
    const truncated = full.slice(0, 512);
    const norm = Math.sqrt(truncated.reduce((s: number, v: number) => s + v * v, 0)) || 1;
    const small = truncated.map((v: number) => v / norm);
    return { full, small };
  });
}

async function embedBatchWithRetry(texts: string[], attempt = 0): Promise<{ full: number[]; small: number[] }[]> {
  try {
    return await embedBatch(texts);
  } catch (e) {
    const msg = String((e as any)?.message || e);
    // Retry on 429 (rate limit) AND transient 5xx (S193 Pass-2 A4) — a one-off
    // OpenAI 503 shouldn't fail an entire ingest pass. The misalignment guard
    // (length assert) is NOT retryable and rethrows immediately.
    const retryable = /failed: (429|5\d\d)/.test(msg);
    if (retryable && attempt < 2) {
      await new Promise(r => setTimeout(r, 2000 * (attempt + 1)));
      return embedBatchWithRetry(texts, attempt + 1);
    }
    throw e;
  }
}

function extractHandleFromUrl(url: string): string {
  // apidojo/tweet-scraper omits the author object; parse handle from the canonical
  // tweet URL. S193 patch: lowercase the slug — Twitter URLs preserve user-canonical
  // case (@Hedgeye) but the FK is case-sensitive; lowercase is the canonical form.
  // Migration 009 enforces the invariant via CHECK; this is the producer side.
  if (!url) return "@unknown";
  const m = url.match(/https?:\/\/(?:x|twitter)\.com\/([^/]+)\/status\//i);
  return m ? `@${m[1].toLowerCase()}` : "@unknown";
}

function extractTweetId(t: any): string {
  // Prefer the url-embedded id (always a string → precision-safe). apidojo returns
  // `id` as a string today (verified 2026-05-28: url id == tweet_id for 2999/2999
  // rows), but 19-digit tweet IDs exceed Number.MAX_SAFE_INTEGER (9e15) — if apidojo
  // ever switches `id` to a JSON number, res.json() would silently round it BEFORE
  // String(t.id) runs, corrupting the PK + breaking ON CONFLICT dedup. Deriving from
  // the url eliminates that latent landmine. (S193 Pass-2 B3.)
  const url = t.url || t.twitterUrl || "";
  const m = url.match(/status\/(\d+)/);
  if (m) return m[1];
  return t?.id != null ? String(t.id) : "";
}

function parseTwitterDate(dateStr: string): string | null {
  if (!dateStr) return null;
  const d = new Date(dateStr);
  return isNaN(d.getTime()) ? null : d.toISOString();
}

function extractTickers(text: string): string[] {
  const matches = text.match(/\$[A-Z]{1,5}\b/g) || [];
  return Array.from(new Set(matches));
}

function extractTopics(text: string): string[] {
  const topicKeywords: Record<string, string[]> = {
    fed: ["fed", "powell", "fomc", "rate cut", "rate hike", "qt", "qe"],
    geopolitical: ["iran", "israel", "russia", "ukraine", "china", "taiwan", "war"],
    earnings: ["earnings", "guidance", "beat", "miss", "raised", "lowered"],
    crypto: ["bitcoin", "btc", "ethereum", "eth", "crypto"],
    macro: ["cpi", "ppi", "gdp", "nfp", "jobless", "pmi", "ism"],
  };
  const textLower = text.toLowerCase();
  const topics: string[] = [];
  for (const [topic, kws] of Object.entries(topicKeywords)) {
    if (kws.some(kw => textLower.includes(kw))) topics.push(topic);
  }
  return topics;
}

function buildRow(t: any, handle: string, runId: string, embeds: { full: number[]; small: number[] } | null) {
  const handleRaw = handle.replace(/^@/, "");
  const tweetUrl = t.url || t.twitterUrl || "";
  const text = t.text || t.fullText || "";
  const tweetId = extractTweetId(t);
  return {
    tweet_id: tweetId,
    account_handle: handle,
    text: text,
    url: tweetUrl || `https://x.com/${handleRaw}/status/${tweetId}`,
    created_at: parseTwitterDate(t.createdAt) || new Date().toISOString(),
    likes_count: t.likeCount || 0,
    retweets_count: t.retweetCount || 0,
    replies_count: t.replyCount || 0,
    views_count: t.viewCount ?? null,
    bookmarks_count: t.bookmarkCount ?? null,
    is_retweet: !!t.isRetweet,
    is_reply: !!t.isReply || !!t.inReplyToId,
    is_quote: !!t.isQuote || !!t.quotedStatusId,
    reply_to_tweet_id: t.inReplyToId || null,
    reply_to_user: t.inReplyToUsername || null,
    quote_tweet_id: t.quotedStatusId || null,
    quote_tweet_url: t.quotedStatusUrl || null,
    media_urls: (t.media || []).map((m: any) => m?.url || m?.media_url_https).filter(Boolean),
    media_types: (t.media || []).map((m: any) => m?.type).filter(Boolean),
    extracted_tickers: extractTickers(text),
    extracted_topics: extractTopics(text),
    language: t.lang || null,
    embedding: embeds ? embeds.full : null,
    embedding_512: embeds ? embeds.small : null,
    apify_run_id: runId,
    raw_payload: t,
  };
}

// Fire the next pass to a fresh isolate. The target returns 200 in <2s after
// dispatching its own background work, so this await is quick.
async function dispatchContinuation(runId: string, nextPass: number): Promise<void> {
  // Auth via header (the receiver's gate accepts x-apify-webhook-secret), NOT a ?secret=
  // query param: a network-level fetch throw would otherwise embed the secret in the URL
  // inside the error text (S196 security review). redactSecrets is the backstop; keeping
  // it out of the URL is the primary defense.
  const url = `${SUPABASE_URL}/functions/v1/social-intel-ingest`;
  await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-apify-webhook-secret": WEBHOOK_SECRET },
    body: JSON.stringify({ resource: { id: runId }, _pass: nextPass }),
  });
}

async function markFailed(runId: string, msg: string): Promise<void> {
  await supabase.schema("social_intel").from("ingest_runs").update({
    status: "failed",
    completed_at: new Date().toISOString(),
    error_message: redactSecrets(msg).slice(0, 2000),
  }).eq("run_id", runId).then(({ error }) => {
    if (error) console.error(`ingest_runs failed-update error: ${error.message}`);
  });
}

// Pre-flight handle-drift detector (S195 P4.6 — Atlas Team 4-0, option "c":
// drop-and-mark-partial). The Apify task scrape list and the social_intel.accounts
// seed are independently hand-managed; drift between them is silent until ingest.
// On 2026-05-29 (@credaily, runId 3Rj0fETs2aIGlzWOB) an unseeded handle FK-violated
// the upsert mid-run: 37 tweets landed, ~946 lost when the whole ingest aborted.
// This filters out unseeded handles' tweets so the rest still land; the run is then
// marked 'partial_success' so the alert layer surfaces the drift loudly.
//
// Re-derived per pass against the LIVE accounts table (self-healing: if the operator
// adds the handle mid-chain, later passes will land it). Bounded <100 handles/page in
// practice, so the PostgREST .in() 1000-row cap is N/A by construction (CLAUDE.md rule).
async function computeDroppedHandles(items: any[]): Promise<string[]> {
  const datasetHandles = new Set<string>();
  for (const t of items) {
    const h = extractHandleFromUrl(t.url || t.twitterUrl || "");
    if (h && h !== "@unknown") datasetHandles.add(h);
  }
  const handleList = [...datasetHandles];
  if (handleList.length === 0) return [];
  const { data: existing, error } = await supabase
    .schema("social_intel")
    .from("accounts")
    .select("handle")
    .in("handle", handleList);
  if (error) {
    // FAIL-OPEN: the accounts lookup is ADVISORY — the tweets FK is the real
    // backstop. On a transient lookup error, drop nothing (return []); a genuinely
    // unseeded handle still FK-aborts the run exactly as pre-P4.6 (no regression),
    // while a no-drift page proceeds instead of being failed by a blip. Failing
    // closed here would add a brand-new transient-failure surface to every
    // previously-succeeding run (review finding, happy-path lens).
    console.error(`accounts seed lookup failed (fail-open, no drop): ${error.message}`);
    return [];
  }
  const seeded = new Set((existing ?? []).map((r: any) => r.handle));
  // seeded.size===0 for a non-empty handle list is AMBIGUOUS: either (a) this page is
  // genuinely 100% drift (every handle unseeded — dropping is CORRECT → partial_success)
  // or (b) the lookup is broken (grant/schema regression returning an empty set without
  // an error — dropping would mass-drop real data). A blanket "no drop" mishandles (a):
  // the unseeded tweets reach the upsert and FK-abort the ENTIRE run = the exact pre-P4.6
  // failure P4.6 exists to prevent (reachable on a single-handle scrape or an author-
  // grouped all-drift page). Disambiguate with a sentinel COUNT of the accounts table:
  // if it has rows, table+grant are healthy so an empty .in() means genuine drift → DROP
  // all (run becomes the actionable partial_success). If the sentinel is 0 or errors,
  // treat as untrustworthy → fail-open (drop nothing). (S196 P4.6 data-shape review.)
  if (seeded.size === 0) {
    const { count: acctCount, error: cntErr } = await supabase
      .schema("social_intel").from("accounts")
      .select("handle", { count: "exact", head: true });
    if (cntErr || !acctCount || acctCount === 0) {
      console.error(`accounts all-unseeded page but sentinel untrustworthy (count=${acctCount ?? "err"}${cntErr ? ", " + cntErr.message : ""}) — fail-open, no drop`);
      return [];
    }
    console.error(`accounts all-unseeded page (${handleList.length} handles) but table healthy (${acctCount} seeded) — genuine drift, dropping all → partial_success`);
    return handleList;
  }
  return handleList.filter(h => !seeded.has(h));
}

// Process ONE offset-paged window. Continues to the next pass or finalizes.
async function processIngestInBackground(runId: string, pass: number): Promise<void> {
  try {
    if (pass >= MAX_PASSES) {
      await markFailed(runId, `Exceeded MAX_PASSES (${MAX_PASSES}) — aborting paged ingest to prevent runaway.`);
      return;
    }

    const runDetails = await fetchApifyRunDetails(runId);
    const datasetId = runDetails?.defaultDatasetId;
    const apifyStatus = runDetails?.status;
    const costUsd = parseFloat(runDetails?.usageTotalUsd || "0");

    if (pass === 0 && (apifyStatus !== "SUCCEEDED" || !datasetId)) {
      await markFailed(runId, `Apify run not usable: status=${apifyStatus} datasetId=${datasetId ?? "null"}`);
      return;
    }
    if (!datasetId) {
      await markFailed(runId, `No datasetId on Apify run (pass ${pass}).`);
      return;
    }

    // E-OBS (S195 Pass-2, operator-approved 2026-05-30): persist apify_dataset_id
    // as soon as we have it. Without this, a failure mid-flight (e.g., the upsert
    // FK violation that surfaced on runId 3Rj0fETs2aIGlzWOB) leaves the column NULL
    // — operator can't jump straight to the dataset for post-mortem. Pass-0 only;
    // continuation passes already have it from pass 0.
    if (pass === 0) {
      await supabase.schema("social_intel").from("ingest_runs")
        .update({ apify_dataset_id: datasetId })
        .eq("run_id", runId)
        .then(({ error }) => {
          if (error) console.error(`apify_dataset_id early-set failed: ${error.message}`);
        });
    }

    const offset = pass * MAX_PER_PASS;
    const items = await fetchApifyDatasetPage(datasetId, offset, MAX_PER_PASS);

    if (items.length > 0) {
      // P4.6 drift filter: drop tweets whose handle is not seeded in accounts —
      // those would FK-violate the upsert and abort the whole run. Re-derived per
      // pass against the live accounts table (self-healing). Cost: one extra accounts
      // read per page (cheap, <100 handles); no extra writes unless drift is found.
      const dropped = await computeDroppedHandles(items);
      const droppedSet = new Set(dropped);

      // Build the embed/upsert set; count unparseable-handle skips (M2 — never silent).
      const pending: { t: any; handle: string; text: string }[] = [];
      let unparseableThisPass = 0;
      for (const t of items) {
        const tweetId = extractTweetId(t);
        const handle = extractHandleFromUrl(t.url || t.twitterUrl || "");
        const text = t.text || t.fullText || "";
        // @unknown = apidojo emitted a null-author / unparseable URL (deleted/suspended/
        // protected account, or a future URL-format change). Count it so a SYSTEMATIC
        // break surfaces at finalization instead of silently dropping ~100% of a run.
        if (handle === "@unknown") unparseableThisPass++;
        if (!tweetId || handle === "@unknown" || !text) continue;
        if (droppedSet.has(handle)) continue;  // drift filter — unseeded handle
        pending.push({ t, handle, text });
      }

      // M1+M2 (S199 Pass-2): atomic per-pass accumulation of the drifted-handle set AND the
      // unparseable count via a single row-locked RPC (social_intel.record_pass_drift,
      // migration 012). Replaces the prior select-then-update on accounts_attempted, which
      // raced across a >10min in_progress re-claim spawning a second chain → a lost drift
      // entry → run mis-marked 'succeeded' (silent drift loss; the exact failure P4.6
      // prevents). FAIL-CLOSED: this is the drift RECORD — throw → markFailed → re-fireable.
      // The no-drift / no-unparseable happy path skips the RPC entirely → zero extra writes.
      if (dropped.length > 0 || unparseableThisPass > 0) {
        const { error: recErr } = await supabase.schema("social_intel")
          .rpc("record_pass_drift", { p_run_id: runId, p_dropped: dropped, p_unparseable: unparseableThisPass });
        if (recErr) throw new Error(`drift/unparseable accumulator RPC failed (pass ${pass}): ${recErr.message}`);
      }

      if (pending.length > 0) {
        const embedBatches: Promise<{ full: number[]; small: number[] }[]>[] = [];
        for (let i = 0; i < pending.length; i += EMBED_BATCH_SIZE) {
          embedBatches.push(embedBatchWithRetry(pending.slice(i, i + EMBED_BATCH_SIZE).map(p => p.text)));
        }
        const embedded = (await Promise.all(embedBatches)).flat();
        const rows = pending.map((p, i) => buildRow(p.t, p.handle, runId, embedded[i] ?? null));

        for (let i = 0; i < rows.length; i += UPSERT_CHUNK_SIZE) {
          const chunk = rows.slice(i, i + UPSERT_CHUNK_SIZE);
          const { error } = await supabase
            .schema("social_intel")
            .from("tweets")
            .upsert(chunk, { onConflict: "tweet_id", ignoreDuplicates: true });
          if (error) throw new Error(`tweets upsert failed (pass ${pass}, chunk @${i}): ${error.message}`);
        }

        const distinctHandles = [...new Set(pending.map(p => p.handle))];
        if (distinctHandles.length > 0) {
          await supabase.schema("social_intel").from("accounts")
            .update({ last_scraped_at: new Date().toISOString() })
            .in("handle", distinctHandles)
            .then(({ error }) => {
              // Advisory (no throw) — last_scraped_at is metadata — but make a silent
              // freeze observable, matching the file's advisory→console.error discipline.
              if (error) console.error(`accounts last_scraped_at update failed (pass ${pass}): ${error.message}`);
            });
        }
      }
    }

    // Continue while the page returned ANY items; finalize only on an empty page.
    // (S193 Pass-2 B1: the prior `=== MAX_PER_PASS` test would stop early + silently
    //  under-ingest if Apify clean=true ever drops items mid-page, returning a short
    //  middle page. `> 0` is robust to that; costs one extra empty-page fetch.)
    const morePages = items.length > 0;
    if (morePages) {
      await dispatchContinuation(runId, pass + 1);
      return;  // leave status in_progress; the next isolate continues the chain
    }

    // Last page (empty) — finalize with the accurate cumulative landed count +
    // the distinct account set for this run (S193 Pass-2 F1: prior paged
    // finalization left accounts_attempted/succeeded empty, a lying observability
    // column. Derive from the tweets actually landed for this run).
    const { count: landedCount } = await supabase
      .schema("social_intel")
      .from("tweets")
      .select("tweet_id", { count: "exact", head: true })
      .eq("apify_run_id", runId);

    // Collect the DISTINCT account set for this run. PostgREST caps each response
    // at the project's db-max-rows (observed 1000), so a single select would
    // silently truncate to the first 1000 tweets' handles (S193 Pass-2: caught
    // pre-deploy). Page until an empty page; advance by ACTUAL rows returned so
    // this is correct for any server-side max-rows. Iteration-capped to avoid runaway.
    const runHandleSet = new Set<string>();
    {
      let from = 0;
      const STEP = 1000;
      for (let iter = 0; iter < 100; iter++) {
        const { data: hr, error: hrErr } = await supabase
          .schema("social_intel")
          .from("tweets")
          .select("account_handle")
          .eq("apify_run_id", runId)
          .range(from, from + STEP - 1);
        if (hrErr) { console.error(`handle page @${from} failed: ${hrErr.message}`); break; }
        const n = hr?.length ?? 0;
        if (n === 0) break;
        for (const r of hr) runHandleSet.add((r as any).account_handle);
        from += n;
      }
    }
    const runHandles = [...runHandleSet];

    // P4.6: read back the drift accumulator (dropped handles stashed in
    // accounts_attempted across data passes). On this empty finalization page our
    // local view of drift is empty, so the persisted set is authoritative.
    const { data: driftRow, error: driftErr } = await supabase.schema("social_intel").from("ingest_runs")
      .select("accounts_attempted, unparseable_count").eq("run_id", runId).maybeSingle();
    // FAIL-CLOSED: this read decides succeeded vs partial_success. Swallowing the
    // error would silently mark a drifted run 'succeeded' (terminal — the pass-0 gate
    // refuses to re-fire a succeeded run), re-introducing the exact silent drift-loss
    // P4.6 exists to prevent. Throw → markFailed → re-fireable recompute.
    if (driftErr) throw new Error(`drift accumulator read-back failed: ${driftErr.message}`);
    // The accumulator IS the dropped set. We do NOT subtract landed handles: a handle
    // dropped on an early page but seeded mid-chain and landed late is STILL drift — its
    // early-page tweets were filtered and (forward-only paging) are not re-fetched within
    // this chain, so they are lost unless the run is re-fired. Marking partial_success
    // (re-claimable → recoverable) rather than 'succeeded' (terminal) is what makes that
    // recovery possible; subtracting landed would bury the data loss as success. (S196.)
    const droppedHandles = ((driftRow?.accounts_attempted as string[]) ?? []);
    const attempted = [...new Set([...runHandles, ...droppedHandles])];

    // M2 (S199 Pass-2): surface a SYSTEMATIC unparseable-handle break. Routine null-author
    // tweets (deleted/suspended/protected accounts) number a handful per run; a wholesale
    // apidojo URL-format change would skip a large fraction. unparseable_count is ALWAYS
    // persisted (the RPC accumulates it) so it is never silent; the floor only gates the
    // status/alert so routine drops don't flap partial_success. Tune after a baseline.
    const UNPARSEABLE_ALERT_FLOOR = 25;  // operator-review value
    const unparseableCount = (driftRow?.unparseable_count as number) ?? 0;
    const unparseableSystemic = unparseableCount >= UNPARSEABLE_ALERT_FLOOR;

    const finalStatus = (droppedHandles.length > 0 || unparseableSystemic) ? "partial_success" : "succeeded";
    const msgParts: string[] = [];
    if (droppedHandles.length > 0) {
      msgParts.push(`DRIFT FILTERED — ${droppedHandles.length} unseeded handle(s): ${droppedHandles.slice(0, 10).join(", ")}${droppedHandles.length > 10 ? "… (truncated)" : ""}. ACTION REQUIRED: add to social_intel.accounts OR remove from the Apify task input.`);
    }
    if (unparseableSystemic) {
      msgParts.push(`UNPARSEABLE HANDLES — ${unparseableCount} tweet(s) had a null-author / unparseable URL and were dropped (not stored). A systematic apidojo URL-format change manifests this way; inspect the dataset.`);
    }
    const finalErrorMsg = msgParts.length > 0 ? msgParts.join(" | ") : null;

    const { error: updErr } = await supabase.schema("social_intel").from("ingest_runs").update({
      status: finalStatus,
      tweets_count: landedCount ?? 0,
      accounts_attempted: attempted,
      accounts_succeeded: runHandles,
      cost_usd: costUsd,
      apify_dataset_id: datasetId,
      completed_at: new Date().toISOString(),
      error_message: finalErrorMsg,
    }).eq("run_id", runId);
    if (updErr) console.error(`ingest_runs ${finalStatus} update failed: ${updErr.message}`);

  } catch (err) {
    const errMsg = redactSecrets(`Background pass ${pass}: ${String((err as any)?.message || err).slice(0, 1900)}`);
    console.error(`ingest run ${runId} ${errMsg}`);
    await markFailed(runId, errMsg);
  }
}

serve(async (req) => {
  if (req.method !== "POST") return new Response("method not allowed", { status: 405 });

  let runId: string | null = null;
  try {
    // 1. Auth gate
    const url = new URL(req.url);
    const providedSecret = url.searchParams.get("secret") || req.headers.get("x-apify-webhook-secret");
    if (!WEBHOOK_SECRET || providedSecret !== WEBHOOK_SECRET) {
      return new Response(JSON.stringify({ error: "unauthorized" }), { status: 401 });
    }

    // 2. Parse payload
    let body: any;
    try { body = await req.json(); } catch { return new Response(JSON.stringify({ error: "bad json" }), { status: 400 }); }
    runId = body?.resource?.id ?? null;
    const pass: number = Number.isInteger(body?._pass) ? body._pass : 0;
    if (!runId) return new Response(JSON.stringify({ error: "missing resource.id" }), { status: 400 });

    // 3. Continuation passes (_pass > 0) bypass dedup/claim — the row is already
    //    in_progress from pass 0, and this is our own internal fan-out.
    if (pass === 0) {
      const { data: existing } = await supabase
        .schema("social_intel")
        .from("ingest_runs")
        .select("run_id, status, started_at")
        .eq("run_id", runId)
        .maybeSingle();

      if (existing) {
        const startedMs = existing.started_at ? new Date(existing.started_at).getTime() : 0;
        const ageMs = Date.now() - startedMs;
        if (existing.status === "succeeded") {
          return new Response(JSON.stringify({ ok: true, runId, skipped: "already_succeeded" }), { status: 200 });
        }
        if (existing.status === "in_progress" && ageMs < 10 * 60 * 1000) {
          return new Response(JSON.stringify({ ok: true, runId, skipped: "in_progress_younger_than_10min" }), { status: 200 });
        }
        // Stuck in_progress / failed / partial_success → retry (upsert below).
        // partial_success is intentionally re-claimable (NOT terminal): after the operator
        // seeds the missing handle, re-firing the same runId re-fetches from offset 0
        // (ON CONFLICT DO NOTHING skips already-landed rows; previously-dropped rows now
        // land) and the run flips to succeeded — the drift-recovery path.
      }

      const { error: upErr } = await supabase
        .schema("social_intel")
        .from("ingest_runs")
        .upsert({
          run_id: runId,
          status: "in_progress",
          started_at: new Date().toISOString(),
          completed_at: null,
          tweets_count: 0,
          accounts_attempted: [],
          accounts_succeeded: [],
          // M2 (S199 Pass-3): reset the unparseable accumulator on (re-)claim, parallel to
          // accounts_attempted — otherwise a re-fired run double-counts onto the stale value
          // (and could wrongly stay partial_success). Fresh runs start at 0 via the column
          // default; this only matters for the re-claim/drift-recovery path.
          unparseable_count: 0,
          cost_usd: 0,
          error_message: null,
        }, { onConflict: "run_id" });
      if (upErr) {
        return new Response(JSON.stringify({ error: redactSecrets(`ingest_runs claim failed: ${upErr.message}`) }), { status: 500 });
      }
    }

    // 4. Dispatch this pass to the background; return 200 immediately.
    if (typeof EdgeRuntime !== "undefined" && EdgeRuntime && typeof EdgeRuntime.waitUntil === "function") {
      EdgeRuntime.waitUntil(processIngestInBackground(runId, pass));
    } else {
      processIngestInBackground(runId, pass).catch((e) => console.error(`bg fallback error: ${e}`));
    }

    return new Response(JSON.stringify({ ok: true, runId, pass, dispatched: "background" }), {
      status: 200, headers: { "content-type": "application/json" },
    });

  } catch (err) {
    const errMsg = redactSecrets(String((err as any)?.message || err).slice(0, 2000));
    if (runId) {
      await supabase.schema("social_intel").from("ingest_runs").upsert({
        run_id: runId,
        status: "failed",
        error_message: `Sync entry: ${errMsg}`,
        completed_at: new Date().toISOString(),
      }, { onConflict: "run_id" }).then(({ error }) => {
        if (error) console.error(`ingest_runs sync-fail upsert error: ${error.message}`);
      });
    }
    return new Response(JSON.stringify({ error: errMsg.slice(0, 500) }), { status: 500 });
  }
});
