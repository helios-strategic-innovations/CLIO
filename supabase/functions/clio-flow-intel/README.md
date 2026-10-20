# clio-flow-intel — RFC-CLIO-BRIDGE-v1 read contract (P2B path A · A1)

Read-only Edge Function that serves CLIO's enriched social-intel substrate to **Hydrinia's
out-of-band mirror** (A2). Versioned, schema-hiding seam: a future `tweet_signals`/`tweets`/
`stories` migration changes only `social_intel.get_flow_intel` (migration 019), never this
contract or Hydrinia's consumer. **No write surface to Hydrinia** — the only DB call is the
SELECT-only RPC.

## Endpoint
`POST /functions/v1/clio-flow-intel?secret=<CLIO_FLOW_READ_TOKEN>`
- `POST` only (405 otherwise). Auth fail-closed: unset/empty token ⇒ 401.
- Body optional; an **empty body returns the broad recent lens** (all themes, last 72h).

## Request body (all fields optional)
| field | type | default | meaning |
|---|---|---|---|
| `tickers` | string[] | — | match vs `mapped_tickers` (caller-cased→UPPER; array overlap) |
| `themes` | string[] | — | match vs `theme_tags` (array overlap) — e.g. `geopolitics`,`energy` |
| `entities` | string[] | — | match vs `mapped_entity` (case-insensitive **exact**) — e.g. `Iran`,`SpaceX` |
| `scope` | string[] | — | match vs `scope` — `macro`\|`market`\|`sector` |
| `since` | ISO8601 | now−72h | `extracted_at > since` |
| `limit` | int | 200 | **hard-capped at 500** |
| `offset` | int | 0 | pagination |
| `novel_only` | bool | `true` | skip near-dup/echo signals (`novel=false`) |

Selectors are an **OR-union**: a row matches if it satisfies *any* supplied selector (plus
`status='ok' AND since AND novel`). With **no** selector → broad recent lens. Rationale
(P0): only 13.6% of signals carry a ticker; `theme_tags` (86.2%) is the dominant axis, so a
ticker-only read would starve the lens of the macro/geopolitical OSINT color that is its point.

## Response `200`
```jsonc
{
  "contract": "RFC-CLIO-BRIDGE-v1", "version": 1,
  "generated_at": "…Z",
  "query": { /* echo of the resolved params */ },
  "count": 5,
  "next_offset": null,            // offset+limit if a full page, else null
  "signals": [{
    "tweet_id","handle","tweet_text","url","created_at",
    "mapped_tickers","mapped_entity","direction","strength","scope","theme_tags",
    "novelty","novel","is_amplification",
    "engagement": {"likes","retweets","replies","views"},
    "story_id","story_label","story_velocity_24h","story_status","extracted_at"
  }]
}
```
Errors: `400` bad json / invalid `since` · `401` unauthorized · `502` query failed · `500` internal.

## Secret
`CLIO_FLOW_READ_TOKEN` — set via `supabase secrets set` (NOT committed). Plus the standard
`SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` (already present for the sibling functions).

## Deploy (gated — from `main`/deployed ref only, per CLAUDE.md)
1. `git branch --show-current` ⇒ must be `main` (deploy-from-feature-branch silently regresses).
2. Apply migration `019_social_intel_flow_intel_read.sql`; then `NOTIFY pgrst, 'reload schema';`
3. `supabase secrets set CLIO_FLOW_READ_TOKEN=…`
4. `supabase functions deploy clio-flow-intel`
5. Smoke: `POST …/clio-flow-intel?secret=…` `-d '{"themes":["geopolitics"],"limit":5}'` ⇒ 200, 5 rows.

A1 is CLIO-side only. A2 (Hydrinia mirror → `state/clio_flow_intel.json`) and A3 (consumer
wiring) are Hydrinia-side and each need their own operator go + the #19 cross-system ritual.
