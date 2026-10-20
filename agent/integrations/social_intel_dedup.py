"""
P2A-core — dedup-on-write + novelty backfill (CLIO Enrichment/Consumer Layer; the §6.D
divergence-detector prerequisite). Directive CC_CLIO_P2A_CORE_DEDUP_2026-06-25.

Flags each enriched signal `novel` (a distinct take) vs a near-dup of an earlier take, and records
the near-dup graph in social_intel.signal_links (PROVENANCE-PRESERVING — a dupe is flagged + linked,
NEVER dropped). The §6.D detector reads `novel = true` so near-duplicate original takes don't inflate
the talk-leg's narrative velocity into false divergences against COT positioning.

Division of labor (mirrors E.2 stories — see social_intel_stories.py):
- GEOMETRY IS SERVER-SIDE: the chronological-greedy KNN + novelty/novel write + link insert run as a
  pgvector plpgsql RPC (migration 017: dedup_signals). No embedding payload ever crosses the wire;
  this module only orchestrates the batched drive. ZERO LLM cost.
- STATUS-BASED WORK DISCOVERY (R4 / anti-pattern #36): the unit of work is
  `status='ok' AND novelty IS NULL`, drained to zero — never a rolling created_at window, so
  late-ingested / late-healed rows stay discoverable until processed. novelty (the score) doubles as
  the processed marker; novel is the detector-facing flag.
- SAME LANE AS E.1/E.2: shares _run_lock, the enrich_runs ledger (run_type='dedup'), the cross-process
  in-flight guard, and the spend cap (dedup spend is $0, but the ledger row is heartbeat evidence and
  keeps the lane single-writer). Ledger mapping: seen = signals processed; written_ok = near-dups
  linked; dropped/skipped = 0.
- Wired on-write from social_intel_enrich.run_incremental (dedup_after_enrich) so the daily incremental
  keeps novel/signal_links populated for new rows; run_dedup() is the standalone backfill / heal path.

Knobs (restart-pinned by design — read once at import):
- CLIO_DEDUP_SIM_THRESHOLD (0.95): cosine similarity >= threshold to an entity-sharing older neighbor
  = near-dup → novel=false + linked. (novel is set by the link decision in the RPC; novelty is the
  cosine distance to the nearest neighbor.)
- CLIO_DEDUP_WINDOW_DAYS (14): only compare against neighbors within this many days (narrative time).
- CLIO_DEDUP_ENABLED (default on): kill-switch (set 0 + restart to stop on-write dedup).
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

# BEFORE agent.config: the config singleton snapshots env at import. Standalone CLI runs need
# load_dotenv HERE (a load_dotenv inside __main__ runs after imports — S203 finding).
from dotenv import load_dotenv
load_dotenv()

import httpx

# Same-lane plumbing (deliberate coupling: one lane lock, one ledger, one in-flight guard — E.1/E.2/
# P2A are phases of the same pipeline; see the enrich module docstring).
from agent.integrations.social_intel_enrich import (  # noqa: F401
    _RunLedger,
    _SCHEMA,
    _TIMEOUT,
    _finish_ledger_best_effort,
    _inflight_run_exists,
    _read_headers,
    _req,
    _run_lock,
    _supa_base,
    _supa_key,
)

logger = logging.getLogger("openclaw.social_intel_dedup")

_ENABLED = os.getenv("CLIO_DEDUP_ENABLED", "1") not in ("0", "false", "no", "off")
_SIM_THRESHOLD = float(os.getenv("CLIO_DEDUP_SIM_THRESHOLD", "0.95"))
_WINDOW_DAYS = int(os.getenv("CLIO_DEDUP_WINDOW_DAYS", "14"))
_RPC_BATCH = int(os.getenv("CLIO_DEDUP_RPC_BATCH", "150"))   # signals per RPC call (bounded ef_search → well under the 8s PostgREST statement timeout)
_MAX_RPC_ITER = 800                                          # 800 x 150 = 120K >> any real corpus


def _rpc_headers() -> dict:
    key = _supa_key()
    return {"apikey": key, "Authorization": f"Bearer {key}",
            "Content-Profile": _SCHEMA, "Accept-Profile": _SCHEMA,
            "Content-Type": "application/json", "Prefer": "return=representation"}


async def _rpc(http: httpx.AsyncClient, fn: str, args: dict, label: str) -> list:
    """Call a social_intel RPC via PostgREST. Fail-loud: raises on non-200."""
    url = f"{_supa_base()}/rest/v1/rpc/{fn}"
    resp = await _req(http, "POST", url, label, headers=_rpc_headers(), json=args)
    if resp.status_code != 200:
        raise RuntimeError(f"{label} failed {resp.status_code}: {resp.text[:300]}")
    return resp.json()


async def _dedup_drive(http: httpx.AsyncClient, result: dict,
                       ledger: "_RunLedger | None" = None) -> None:
    """Drive dedup_signals to exhaustion over the undeduped (status='ok' AND novelty IS NULL) set.
    Zero LLM cost. Updates `result` (processed/linked/novel/iter_capped) IN PLACE every iteration so
    the error/cancel finish records the TRUE floor of progress (the enrich module's _merge_persisted
    precedent — a mid-drive exception must not let the finish PATCH a 0 over real progress)."""
    for _ in range(_MAX_RPC_ITER):
        rows = await _rpc(http, "dedup_signals",
                          {"p_sim_threshold": _SIM_THRESHOLD, "p_window_days": _WINDOW_DAYS,
                           "p_batch": _RPC_BATCH}, "dedup pass")
        r = rows[0] if rows else {}
        n = int(r.get("processed") or 0)
        result["processed"] += n
        result["linked"] += int(r.get("linked") or 0)
        result["novel"] += int(r.get("novel") or 0)
        if ledger is not None:
            await ledger.update(seen=result["processed"], written_ok=result["linked"])
        if n == 0:
            return
    logger.warning("dedup hit _MAX_RPC_ITER backstop (%d batches) — more remain; next run continues",
                   _MAX_RPC_ITER)
    result["iter_capped"] = True


async def _dedup_core(http: httpx.AsyncClient, run_type: str) -> dict:
    """Open a 'dedup' ledger row, drive the RPC to exhaustion, finish the row. NO lane lock — the
    caller owns concurrency (run_dedup holds _run_lock; the on-write hook already holds it). Never
    raises in scheduled context: errors land in the result dict + the ledger row (the dedup pass is
    idempotent and self-heals the leftover novelty-NULL rows next run)."""
    result = {"run_type": run_type, "run_id": None, "processed": 0, "linked": 0, "novel": 0,
              "iter_capped": False, "error": None}
    ledger = None
    try:
        ledger = _RunLedger(http, run_type)
        await ledger.open()
        result["run_id"] = ledger.run_id
        try:
            await _dedup_drive(http, result, ledger)
            await ledger.finish(seen=result["processed"], written_ok=result["linked"],
                                dropped=0, skipped=0, spend_usd=0)
            logger.info("%s done: processed=%d linked=%d novel=%d%s", run_type,
                        result["processed"], result["linked"], result["novel"],
                        " (iter-capped)" if result["iter_capped"] else "")
            return result
        except asyncio.CancelledError:
            # CancelledError is BaseException — without this the ledger row orphans with a fresh
            # heartbeat and suppresses the shared lane for the in-flight window. Shield the close,
            # then re-raise (symmetric with E.1/E.2, S203 pass-2 F4).
            await asyncio.shield(_finish_ledger_best_effort(
                ledger, {"seen": result["processed"], "written_ok": result["linked"],
                         "dropped": 0, "skipped": 0, "spend_usd": 0},
                error="cancelled: scheduled-tick timeout"))
            raise
        except Exception as e:  # noqa: BLE001 — report, never raise (idempotent self-heal next run)
            result["error"] = str(e)[:500]
            logger.error("%s failed: %s", run_type, e, exc_info=True)
            await _finish_ledger_best_effort(
                ledger, {"seen": result["processed"], "written_ok": result["linked"],
                         "dropped": 0, "skipped": 0, "spend_usd": 0}, error=str(e)[:500])
            return result
    except Exception as e:  # noqa: BLE001 — ledger open / client teardown; nothing to close
        result["error"] = result.get("error") or str(e)[:500]
        logger.error("%s outer failure: %s", run_type, e, exc_info=True)
        return result


async def dedup_after_enrich(http: httpx.AsyncClient) -> dict:
    """Phase-3 on-write hook: called from run_incremental (which already holds _run_lock) right after
    new signals land. Drives the dedup pass over the just-written (novelty-NULL) rows. Disabled via
    CLIO_DEDUP_ENABLED=0. The caller wraps this in try/except so a hiccup never fails enrichment."""
    if not _ENABLED:
        return {"status": "disabled"}
    return await _dedup_core(http, "dedup")


async def run_dedup(run_type: str = "dedup") -> dict:
    """Standalone dedup run — the Phase-2 backfill and manual heal path. Takes the shared lane lock +
    cross-process in-flight guard (mirrors run_stories), then drives the RPC to exhaustion. Idempotent
    (processes only novelty-NULL rows; never duplicates a link)."""
    if not _ENABLED:
        logger.info("social_intel_dedup: disabled (CLIO_DEDUP_ENABLED=0); skipping %s", run_type)
        return {"status": "disabled", "run_type": run_type}
    if _run_lock.locked():
        logger.info("social_intel_dedup: run already in flight (in-process); skipping %s", run_type)
        return {"status": "skipped_concurrent", "run_type": run_type}
    async with _run_lock:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as http:
                if await _inflight_run_exists(http):
                    logger.info("social_intel_dedup: in-flight ledger row found; skipping %s", run_type)
                    return {"status": "skipped_inflight", "run_type": run_type}
                return await _dedup_core(http, run_type)
        except Exception as e:  # noqa: BLE001 — client construction/teardown
            logger.error("%s outer failure: %s", run_type, e, exc_info=True)
            return {"run_type": run_type, "error": str(e)[:500]}


if __name__ == "__main__":
    # CLI: `backfill` (default) → drive the dedup pass to exhaustion over the whole undeduped set.
    # (dotenv is loaded at module top — before the config singleton snapshots env.)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = asyncio.run(run_dedup())
    print(json.dumps(out, indent=2))
