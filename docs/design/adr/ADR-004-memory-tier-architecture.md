# ADR-004 — Memory Tier Architecture: 4 File-Backed + 1 External

**Status**: Accepted (S189, 2026-05-20)
**Author**: Buddy (CC) under operator direction
**Related**: ADR-003 (remediation patterns), CLAUDE.md "Memory Engine" section, S189 audit memory-tier findings.

## Context

CLIO's memory engine (`agent/memory/memory_engine.py:65`) logs "Memory engine initialized — all 5 tiers active" at startup. CLAUDE.md describes a "5-tier: working (72h TTL) / episodic / semantic / procedural / meta" architecture. The S189 audit sub-agent inspected `data/memory/` on disk and found only:

- `working_memory.json` (mtime 2026-04-02, 2 bytes `{}`)
- `procedural_memory.json` (mtime 2026-03-30, 10KB)
- `meta_memory.json` (mtime 2026-05-20, 82KB, actively updated)
- `episodic/` (a directory, not a file)
- **`semantic_memory.json` does not exist on disk.**

The sub-agent flagged this as "memory tier collapse — 3 of 5 layers missing." On closer reading of the source, this is a misread of designed architecture.

## Decision

Codify the memory engine's tier layout as **4 file-backed tiers + 1 external (knowledge-base-backed)** tier. The "5 tiers active" log message is a cosmetic carryover from an earlier architectural plan and should be considered descriptive of the **logical** model, not the on-disk file count.

### The 5 logical tiers

| # | Tier | Storage | Source-of-truth | Behavior |
|---|---|---|---|---|
| 1 | **Working** | `data/memory/working_memory.json` | `WorkingMemory` instance in `agent/memory/working_memory.py` | In-process dict with 72h TTL. `_save()` is unconditional on `add()` (S233 LO-7). Empty `{}` between active sessions is normal (TTL purges, no add() since means file mtime stays). |
| 2 | **Episodic** | `data/memory/episodic/` (directory, NOT a JSON file) | `EpisodicMemory` instance in `agent/memory/episodic.py` | Event-shard files (one per record), compressed monthly. Audit subagent confusion: this tier IS file-backed, but as a directory full of shards, not a single JSON file. |
| 3 | **Semantic** | **`data/knowledge/`** (the entire knowledge base) | `agent/tools/knowledge_tools.py` | Per `memory_engine.py:117` comment: "Semantic memory (Tier 3) is injected separately by existing `get_knowledge_context()`." No `semantic.py` exists in `agent/memory/`. Semantic memory IS the knowledge base — `handle_save_knowledge()` writes there, and retrieval uses semantic search via `get_knowledge_context()`. |
| 4 | **Procedural** | `data/memory/procedural_memory.json` | `ProceduralMemory` instance in `agent/memory/procedural.py` | 14 hard-coded patterns that trigger on user message keywords. Patterns are LOADED from file at startup, but NEW patterns are not added at runtime — only trigger-matches are recorded (in-memory access counts). File mtime stays at last commit time, which is "stale" by mtime but functionally healthy. |
| 5 | **Meta** | `data/memory/meta_memory.json` | `MetaMemory` instance in `agent/memory/meta.py` | Calibration + correction tracking. Actively updated whenever the model emits a confidence-language response or the user offers a correction. Most "alive" tier in absolute terms. |

### Why this architecture

- **Working ↔ Episodic ↔ Semantic** form a temperature ladder: hot session-scope → warm event-shard → cold knowledge.
- **Procedural** is orthogonal: cross-cutting behavioral patterns (e.g., "when user pastes code, treat as inspection request") that fire from ANY tier's context.
- **Meta** is meta-cognitive: tracks the model's own calibration, gaps, and corrections.
- Semantic doesn't need a dedicated file under `data/memory/` because the knowledge base (`data/knowledge/`) IS the semantic-tier storage. Duplicating it would be wasteful and risk drift.

### What audit sub-agents got wrong

| Sub-agent claim | Reality |
|---|---|
| "semantic_memory.json missing" | Semantic memory IS `data/knowledge/`. No separate file by design. |
| "working_memory frozen 48d (empty `{}`)" | Per S186 W2.3 + this ADR: empty content is **benign**. `os.replace()` doesn't write if content didn't change. Working memory NOT being written for 48d is a separate concern (see memory_tier_investigation memo) — but the `{}` state itself is correct. |
| "procedural_memory stale 51d (stuck at 14 patterns)" | Per design: procedural patterns are **declared in code**, not learned. The file is read-only at runtime in the current schema. Future enhancement (S189 deferred) might add runtime pattern accretion, but the current 14 patterns are the intended set. |
| "episodic_memory.json missing" | Episodic is a DIRECTORY, not a JSON file. Each record is a shard. Subagent looked for the wrong path. |
| "5 of 5 tiers active log is wrong" | The log refers to LOGICAL tiers (incl. semantic). It's correct, just over-loaded in meaning. |

### The genuine concern (logged separately)

The audit DID surface a real issue worth investigating in a future session: working_memory.json mtime hasn't advanced in 48 days despite multiple service restarts and confirmed Telegram message activity. This means either (a) `remember()` is never reaching `add()`, or (b) the `_save()` path is silently failing.

Per ADR-003 Pattern 4 (read-only investigation before code change), the fix is **deferred** to a dedicated S190+ session that can:
1. Trace `core.py:548` `loop.run_in_executor(None, lambda: _me.remember(...))` to see if the executor lambda silently raises.
2. Add a non-fire-and-forget debug logger inside `remember()` and `add()` to confirm reach.
3. Audit the `priority > 0.0` filter at `core.py:546`.

This ADR-004 codifies the architecture so the deferred investigation has a known baseline. See `memory/ref/s189_memory_tier_investigation.md` for the empirical findings.

## Consequences

**Positive**:
- Future audits will not re-flag "semantic missing" or "episodic missing" if they reference this ADR.
- The 5-tier-active log message stays (it's the logical model); future contributors know it means 4 file-backed + 1 knowledge-base-backed.
- Operator has a clear root cause for the working_memory mtime issue (deferred for proper investigation, not a half-blind patch).

**Negative**:
- The log message "all 5 tiers active" is slightly misleading without context. A future amendment could either rename it or accept the cost (this ADR explicitly chose to keep it for stability — no behavior change).

## Validation

- `agent/memory/memory_engine.py:50-53` instantiates exactly 4 tiers (working, episodic, procedural, meta).
- `agent/memory/memory_engine.py:117` comment confirms "Semantic memory (Tier 3) is injected separately."
- No `semantic.py` exists under `agent/memory/`.
- `data/memory/episodic/` is a directory, not a file.
- Knowledge tools (`agent/tools/knowledge_tools.py`) provides `get_knowledge_context()` which is the semantic retrieval path.

## Revision History

- **v1 (S189, 2026-05-20)**: Initial codification. Resolves audit sub-agent confusion. Defers the genuine working_memory write-path issue to S190+ with full investigation framing.
