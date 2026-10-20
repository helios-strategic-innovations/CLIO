"""
Retrieval-based personality (INTERIM) — two-stage retrieve → re-rank port
=========================================================================
CLIO's personality is a flat injection of ALL confirmed traits, capped at
PERSONALITY_CONTEXT_MAX_CHARS (.env:229, currently 15,000). That is an
*injection-budget* limit, not a *hold* limit — and once the confirmed set
exceeds the cap the flat path truncates a trait's description mid-sentence
(engine.py:64-65), breaking verbatim integrity.

This module makes personality RETRIEVAL-based: hold ALL confirmed traits,
surface the situationally-relevant window per turn, controlling HOW MANY
traits are injected — never truncating a trait's text.

Architecture — a generic two-stage retrieve → re-rank port:

    Encoder            encode(text) -> repr            (rebuildable cache)
    TraitStore         index(...) + query(cue_repr,k)  (the *retrieve* stage)
    RetrievedTrait     {trait_id, base_score, rerank_score, text, embedding, meta}
    Reranker           rerank(cue_ctx, items)          (THE swap seam)
      PassthroughReranker                              (identity — the interim)
    RetrievalResult    {items, query_repr, n_returned, health, all_scored}
    PersonalityRetriever.retrieve(cue, k, budget)      encode→query→rerank→budget-fill

The *retrieve* stage reuses CLIO's OWN live keyword-recall scorer
(`agent.tools.knowledge_tools._score_entries`) over an in-process candidate
list of confirmed traits — no new engine, zero network, zero cost. The
*re-rank* stage is a NAMED SEAM: today it is the identity `PassthroughReranker`;
a future semantic scorer drops in HERE and only here (and may run as a
firewalled cross-boundary call — only DATA crosses, never code). The RAW
VERBATIM trait description is the source-of-truth; the `embedding` field is a
rebuildable cache of the encoder's repr.

Tiered assembly atop the port:
    CORE  — top-K highest-conviction confirmed traits, always-on identity
            anchor (~2k char floor). Cue-INDEPENDENT and independent of the
            retrieval backend: `health != ok` degrades the TAIL, NEVER the CORE.
    TAIL  — PersonalityRetriever.retrieve(cue, k, tail_budget): verbatim
            confirmed traits, situationally relevant, budget-filled.
    When the FULL confirmed set fits within the injection budget, the block is
    emitted in the flat order (byte-identical to the current flat block, minus
    the truncation bug) — windowing engages only once the set exceeds budget.

SHADOW-FIRST (house convention): gated behind two flags defaulting to
shadow-observe. `CLIO_RETRIEVAL_PERSONALITY_ENABLED=0` (off) — nothing runs;
`=1` begins shadow assembly + trace logging. `CLIO_RETRIEVAL_PERSONALITY_LOG_ONLY=1`
(default) — assemble + LOG beside the live flat context, NEVER inject; `=0`
(operator-gated promotion) — inject the retrieval block instead of flat.
Zero behavioral change on deploy.

Guardrails: confirmed-only · verbatim-whole (raw source-of-truth) · CORE-always
+ survives-degradation · reuse-KB-retrieval-as-retrieve-stage (re-rank = named
Passthrough seam) · zero external-engine vocab / no imported foreign code ·
/trait_confirm untouched.

Author: CLIO / Buddy (interim retrieval-personality build, 2026-07-01)
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("moltbot.personality.retrieval")

# ── Versioned identifiers (carry-forward trace / cross-val) ──────────────────
ENCODING_ID = "kw-topic-v1"            # Encoder: knowledge_tools._extract_topic_keywords
BASE_SCORER_ID = "kb-keyword-recall-v1"  # retrieve: knowledge_tools._score_entries formula
RERANK_SCORER_ID = "passthrough-v1"    # re-rank: identity (the interim swap seam)

# ── Assembly budgets ─────────────────────────────────────────────────────────
CORE_CHAR_FLOOR = 2000                 # CORE always-on anchor budget (~1-2k, directive)
CORE_MAX_TRAITS = 10                   # hard cap on CORE trait count (anchor, not the whole set)

# ── Shadow trace (CLIO-side observability; NOT the Hydrinia snapshot) ─────────
_TRACE_PATH = "/home/agent/openclaw/data/personality/retrieval_shadow.jsonl"
_TRACE_MAX_BYTES = 10 * 1024 * 1024    # rotate at ~10MB (keep one .1 backup)


# ── Flag helpers (runtime-evaluated; code defaults = shadow) ─────────────────
def is_enabled() -> bool:
    """Master gate. Default OFF → the shadow layer does no work at all."""
    return os.environ.get("CLIO_RETRIEVAL_PERSONALITY_ENABLED", "0") == "1"


def is_log_only() -> bool:
    """Shadow gate. Default TRUE → assemble + log, NEVER inject.
    Only an explicit '0' (operator-gated promotion) flips to active injection."""
    return os.environ.get("CLIO_RETRIEVAL_PERSONALITY_LOG_ONLY", "1") != "0"


# ─────────────────────────────────────────────────────────────────────────────
# Port dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RetrievedTrait:
    """A candidate trait returned by the retrieve stage.

    A deliberate SUPERSET of what any re-ranker needs. ``base_score`` (retrieval
    relevance) and ``rerank_score`` (post-rerank relevance) are DISTINCT fields —
    never overwrite one with the other. ``text`` is the RAW VERBATIM trait
    description (source-of-truth); ``embedding`` is the rebuildable encoder repr.
    Plain-serializable + boundary-tolerant by construction.
    """
    trait_id: str
    base_score: float
    text: str                                   # verbatim description (source-of-truth)
    embedding: Optional[Any] = None             # rebuildable repr cache (keyword set for the interim)
    meta: dict = field(default_factory=dict)    # {name, conviction, date_confirmed, domain_applications, ...}
    rerank_score: Optional[float] = None        # set by the re-rank stage; distinct from base_score

    def to_row(self) -> dict:
        return {
            "trait_id": self.trait_id,
            "base_score": self.base_score,
            "rerank_score": self.rerank_score,
        }


@dataclass
class Cue:
    """The per-turn retrieval cue. ``text`` = current topic / last user message.
    ``context`` = opaque per-turn affective/focus slot — EMPTY placeholder in the
    interim (no such state exists in the assembly context; not fabricated). It is
    the forward-compat modulation hook a future re-ranker may read. ``exclude_ids``
    = trait ids already claimed by the CORE (never re-surface them in the TAIL)."""
    text: str = ""
    context: dict = field(default_factory=dict)
    exclude_ids: frozenset = frozenset()


@dataclass
class RetrievalResult:
    items: list                 # budget-filled TAIL selection (list[RetrievedTrait])
    query_repr: list            # the encoded cue (keyword list)
    n_returned: int
    health: str                 # "ok" | "degraded_fallback" | "empty"
    all_scored: list            # FULL candidate ranking with base_scores (trace: highest value)
    rerank_reason: Optional[str] = None  # re-rank backend last_reason ("timeout"⇒deadline-truncated); trace-only


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1a: Encoder  (encode(text) -> repr; rebuildable cache)
# ─────────────────────────────────────────────────────────────────────────────
class Encoder:
    """Interim encoder = CLIO's keyword extractor. repr = keyword set. The raw
    text remains source-of-truth, so re-encoding later needs zero re-index."""

    encoding_id = ENCODING_ID

    def encode(self, text: str) -> list:
        if not text:
            return []
        try:
            from agent.tools.knowledge_tools import _extract_topic_keywords
            return list(_extract_topic_keywords(text))
        except Exception as e:  # fail-open: no repr → degraded (never crash the hot path)
            logger.debug("Encoder.encode failed: %s", e)
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b: TraitStore  (the *retrieve* stage — CLIO's KB scorer, in-process)
# ─────────────────────────────────────────────────────────────────────────────
class TraitStore:
    """In-memory store of confirmed traits behind a generic index/query interface.
    ``query`` reuses `knowledge_tools._score_entries` (the shared KB scoring seam)
    over the trait candidate set — the SAME formula the KB path uses, never a
    mirror. Fully in-process: zero network, zero cost."""

    def __init__(self) -> None:
        self._entries: list[dict] = []          # scoring rows: {id,title,tags,category,created}
        self._text: dict[str, str] = {}         # trait_id -> verbatim description
        self._embedding: dict[str, Any] = {}    # trait_id -> encoder repr (cache)
        self._meta: dict[str, dict] = {}        # trait_id -> meta

    def size(self) -> int:
        return len(self._entries)

    def index(self, trait_id: str, text: str, embedding: Any, meta: dict) -> None:
        """Register a trait. ``text`` = raw verbatim description (source-of-truth).
        The scoring row folds name + domains + description keywords into the fields
        the KB scorer matches (title/tags/category); base_score stays PURE keyword
        relevance (conviction lives in the CORE ordering / TAIL tie-break, not here)."""
        name = meta.get("name", "")
        domains = list((meta.get("domain_applications") or {}).keys())
        try:
            from agent.tools.knowledge_tools import _extract_topic_keywords
            kw = _extract_topic_keywords(f"{name} {text}")
        except Exception as e:
            logger.debug(f"TraitStore keyword extraction failed: {e}")
            kw = []
        self._entries.append({
            "id": trait_id,
            "title": name,
            "tags": domains + kw,
            "category": ("personality " + " ".join(domains)).strip(),
            # no quality_score → base_score is pure keyword relevance
            "created": meta.get("date_confirmed", ""),
        })
        self._text[trait_id] = text
        self._embedding[trait_id] = embedding
        self._meta[trait_id] = meta

    def _to_retrieved(self, trait_id: str, base_score: float) -> RetrievedTrait:
        return RetrievedTrait(
            trait_id=trait_id,
            base_score=float(base_score),
            text=self._text.get(trait_id, ""),
            embedding=self._embedding.get(trait_id),
            meta=self._meta.get(trait_id, {}),
        )

    def _tie_break_key(self, rt: RetrievedTrait):
        # (base_score desc, conviction desc, date_confirmed asc, id asc) — fully
        # deterministic even under uniform conviction (all confirmed == 1.0 today).
        # Keys coerced (never None) so a malformed trait can't raise during sort.
        m = rt.meta or {}
        return (-float(rt.base_score or 0.0), -float(m.get("conviction") or 0.0),
                (m.get("date_confirmed") or ""), (rt.trait_id or ""))

    def query(self, query_repr: list, k: Optional[int] = None) -> list:
        """Score the FULL candidate set against the cue repr (min_score=0.0), map
        to RetrievedTrait, sort by the deterministic tie-break. Returns all when
        k is None (the trace needs every candidate's base_score)."""
        from agent.tools.knowledge_tools import _score_entries
        # min_score=0.0 → every candidate scored (0.0 for no keyword overlap)
        scored = _score_entries(self._entries, list(query_repr or []), min_score=0.0)
        got = {e["id"]: s for s, e in scored}
        items = [self._to_retrieved(e["id"], got.get(e["id"], 0.0)) for e in self._entries]
        items.sort(key=self._tie_break_key)
        return items if k is None else items[:k]

    def all_deterministic(self) -> list:
        """Cue-independent ordering: every candidate at base_score 0.0, sorted by
        (conviction desc, date_confirmed asc, id asc). Used when the cue yields no
        retrieval signal — coverage preserved, relevance degraded."""
        items = [self._to_retrieved(e["id"], 0.0) for e in self._entries]
        items.sort(key=self._tie_break_key)
        return items


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Reranker  (THE swap seam — identity for the interim)
# ─────────────────────────────────────────────────────────────────────────────
class Reranker:
    """Re-rank a candidate set given the opaque cue context. THE strategic scorer
    drops in here and ONLY here (possibly as a firewalled cross-boundary call —
    only DATA crosses, never code). Contract: higher = more relevant, comparable
    WITHIN one call only, NOT a probability."""

    scorer_id = "abstract"

    def rerank(self, cue_ctx: dict, items: list) -> list:  # pragma: no cover
        raise NotImplementedError


class PassthroughReranker(Reranker):
    """Identity re-ranker: keep the retrieve-stage order. Sets rerank_score =
    base_score (a DISTINCT field — never overwrites base_score)."""

    scorer_id = RERANK_SCORER_ID

    def rerank(self, cue_ctx: dict, items: list) -> list:
        for it in items:
            it.rerank_score = it.base_score
        return items


# ─────────────────────────────────────────────────────────────────────────────
# The retriever  (encode → query → rerank → budget-fill)
# ─────────────────────────────────────────────────────────────────────────────
class PersonalityRetriever:
    def __init__(self, encoder: Encoder, store: TraitStore, reranker: Reranker) -> None:
        self.encoder = encoder
        self.store = store
        self.reranker = reranker

    def retrieve(self, cue: Cue, k: Optional[int], budget: int) -> RetrievalResult:
        query_repr = self.encoder.encode(cue.text)

        if self.store.size() == 0:
            return RetrievalResult([], query_repr, 0, "empty", [])

        health = "ok"
        try:
            if query_repr:
                ranked = self.store.query(query_repr, k=None)   # relevance-ranked, full set
            else:
                # no cue signal → cue-independent deterministic fill (coverage kept)
                ranked = self.store.all_deterministic()
                health = "degraded_fallback"
        except Exception as e:
            # retrieve-stage failure → TAIL degrades to EMPTY; CORE is unaffected upstream
            logger.debug("TraitStore.query failed, TAIL degraded to empty: %s", e)
            return RetrievalResult([], query_repr, 0, "degraded_fallback", [])

        # TAIL-only invariant: the CORE trait ids (cue.exclude_ids) are held OUT of
        # the re-rank set entirely — they are never re-ranked and never cross the
        # (possibly cross-boundary) re-rank interface. The assembler enforces
        # rerank_set ∩ CORE == ∅ HERE (structurally, via the comprehension; the
        # assert is a loud tripwire on top). A swapped-in backend therefore can never
        # see, score, or transmit a CORE trait — CORE is never on the wire.
        exclude = cue.exclude_ids or frozenset()
        rerank_input = [it for it in ranked if it.trait_id not in exclude]
        if exclude:
            assert not (exclude & {it.trait_id for it in rerank_input}), \
                "TAIL-only invariant violated: re-rank set intersects CORE"

        # Re-rank the TAIL-eligible set. cue_ctx is a plain dict carrying the raw cue
        # text (a semantic backend needs it to score relevance) alongside the opaque
        # per-turn context — the identity PassthroughReranker ignores it entirely, so
        # this stays backend-agnostic (byte-identical behaviour for the interim).
        ranked = self.reranker.rerank({"text": cue.text, "context": cue.context}, rerank_input)
        all_scored = list(ranked)   # the re-ranked TAIL set (trace: highest value)

        # A re-rank backend that degraded to its local floor (e.g. wire timeout)
        # reports it via an OPTIONAL `last_health` attribute. Surface it — but never
        # override an already-degraded retrieve-stage health. PassthroughReranker has
        # no such attribute (getattr → None → no change), so the interim is unaffected.
        rr_health = getattr(self.reranker, "last_health", None)
        rr_reason = getattr(self.reranker, "last_reason", None)   # e.g. "timeout" ⇒ deadline-truncated
        if rr_health and health == "ok":
            health = rr_health

        # Budget-fill: never truncate a trait; skip-and-continue so smaller traits
        # can still pack the remaining budget (maximises coverage when the set fits).
        selected: list = []
        used = 0
        for it in ranked:
            if it.trait_id in exclude:   # defensive: CORE already held out above
                continue
            piece = format_trait_flat(it.meta.get("name", ""), it.text,
                                      it.meta.get("domain_applications") or {})
            cost = len(piece) + 1  # +1 for the joining newline
            if used + cost > budget:
                continue
            selected.append(it)
            used += cost

        return RetrievalResult(selected, query_repr, len(selected), health, all_scored,
                               rerank_reason=rr_reason)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting — MUST match engine.get_context() byte-for-byte (for zero-delta A/B)
# ─────────────────────────────────────────────────────────────────────────────
_DOMAIN_ORDER = ["trading", "research", "conversation", "code"]


def format_trait_flat(name: str, description: str, domain_applications: dict) -> str:
    """Render one trait exactly as engine.get_context() does: the bullet line plus
    (at most) the first present domain-application line. Verbatim — never truncated."""
    lines = [f"• {name}: {description}"]
    if domain_applications:
        for domain in _DOMAIN_ORDER:
            if domain in domain_applications:
                lines.append(f"  → {domain.title()}: {domain_applications[domain]}")
                break
    return "\n".join(lines)


_HEADER = "PERSONALITY (confirmed traits):"


def _flat_block(traits: list) -> str:
    """The full flat block for a list of Trait objects (engine.get_context() order/format)."""
    parts = [_HEADER]
    for t in traits:
        parts.append(format_trait_flat(t.name, t.description, t.domain_applications or {}))
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CORE + TAIL assembly
# ─────────────────────────────────────────────────────────────────────────────
def _core_order(traits: list) -> list:
    """CORE ordering: (conviction desc, date_confirmed asc, id asc). Deterministic
    under uniform conviction; date_confirmed asc == the registry's confirmed sort,
    so the CORE lead == the flat block's lead traits today. Sort keys are coerced
    (never None) and the sort is guarded, so a single malformed trait can never
    sink the CORE anchor — identity must not depend on clean data any more than on
    the retrieve backend."""
    try:
        return sorted(traits, key=lambda t: (
            -float(getattr(t, "conviction", 0.0) or 0.0),
            (getattr(t, "date_confirmed", "") or ""),
            (getattr(t, "id", "") or ""),
        ))
    except Exception as e:
        logger.debug("CORE ordering fell back to input order: %s", e)
        return list(traits)  # get_confirmed_traits() already returns date_confirmed asc


def _fill_core(ranked_traits: list, core_budget: int):
    """Pack the CORE anchor up to core_budget / CORE_MAX_TRAITS, verbatim, never
    truncating. Returns (core_trait_list, core_ids, chars_used)."""
    core: list = []
    used = 0
    for t in ranked_traits:
        if len(core) >= CORE_MAX_TRAITS:
            break
        piece = format_trait_flat(t.name, t.description, t.domain_applications or {})
        cost = len(piece) + 1
        if core and used + cost > core_budget:
            continue  # try to fit a smaller trait; but always keep at least one anchor
        core.append(t)
        used += cost
    core_ids = [t.id for t in core]
    return core, core_ids, used


def _assemble(confirmed: list, cue_text: str, total_budget: int,
              core_budget: int, retriever: Optional[PersonalityRetriever] = None):
    """Pure, testable assembly. Returns (block_str, RetrievalResult, meta_dict).

    If the full confirmed set fits within total_budget → emit it in flat order
    (byte-identical to engine.get_context() when it does not truncate). Otherwise
    → CORE anchor (always) + TAIL (retrieved, budget-filled). CORE is materialised
    independently of the retrieve stage, so a degraded retrieval never touches it.
    """
    flat_block = _flat_block(confirmed)

    if not confirmed:
        empty_result = RetrievalResult([], [], 0, "empty", [])
        return "", empty_result, {"windowed": False, "core_ids": [], "flat_block": flat_block}

    # Build the retriever (fresh per call at 20-215 scale: <1ms) unless injected.
    if retriever is None:
        store = TraitStore()
        for t in confirmed:
            meta = {
                "name": t.name,
                "conviction": float(getattr(t, "conviction", 0.0) or 0.0),
                "date_confirmed": t.date_confirmed,
                "domain_applications": t.domain_applications or {},
                "activation_count": getattr(t, "activation_count", 0),
            }
            enc = Encoder().encode(f"{t.name} {t.description}")
            store.index(t.id, t.description, enc, meta)
        # Re-rank seam (backend SWAP only — retriever/assembler unchanged). Flag-gated:
        # CLIO_SEMANTIC_RERANK_ENABLED=1 → the firewalled SemanticReranker (SHADOW when
        # LOG_ONLY=1: records rerank_score but injects the base/keyword order = ZERO
        # behavioral change); default/off/import-error → the identity PassthroughReranker
        # (rollback floor). Lazy import breaks the retrieval<->reranker_semantic cycle;
        # any failure fails OPEN to Passthrough (assembly never breaks). deadline_ms
        # (env CLIO_RERANK_DEADLINE_MS, default 120) bounds the ON-LOOP wire cost — with
        # the personality ACTIVE the assembly runs synchronously on the event loop
        # (F-2 / gotcha #37). The CORE holdout + the ∩CORE==∅ assert live in
        # PersonalityRetriever.retrieve, untouched by this swap.
        reranker = PassthroughReranker()
        try:
            from agent.personality import reranker_semantic as _semantic
            if _semantic.is_enabled():
                try:
                    _deadline_ms = int(os.getenv("CLIO_RERANK_DEADLINE_MS", "120"))
                except (TypeError, ValueError):
                    _deadline_ms = 120
                reranker = _semantic.SemanticReranker(deadline_ms=_deadline_ms)
        except Exception as e:
            logger.debug("semantic reranker unavailable; using passthrough: %s", e)
        retriever = PersonalityRetriever(Encoder(), store, reranker)

    # Always run the retriever (the trace is the supervised ranking dataset).
    # windowed=False path emits flat; windowed=True path uses CORE + TAIL.
    windowed = len(flat_block) > total_budget

    if not windowed:
        # Zero-delta: full set fits → flat order/format, byte-identical to the
        # live flat block (minus the mid-string truncation the flat path applies
        # only when it OVERFLOWS — which, by definition, it does not here).
        cue = Cue(text=cue_text, context={}, exclude_ids=frozenset())
        result = retriever.retrieve(cue, k=None, budget=total_budget)
        meta = {"windowed": False, "core_ids": [], "flat_block": flat_block,
                "core_used": 0, "tail_budget": total_budget}
        return flat_block, result, meta

    # Windowed: CORE anchor (deterministic, cue-independent) + TAIL (retrieved).
    core_traits, core_ids, core_used = _fill_core(_core_order(confirmed), core_budget)
    core_block_parts = [format_trait_flat(t.name, t.description, t.domain_applications or {})
                        for t in core_traits]

    tail_budget = max(0, total_budget - core_used - len(_HEADER) - 1)
    cue = Cue(text=cue_text, context={}, exclude_ids=frozenset(core_ids))
    result = retriever.retrieve(cue, k=None, budget=tail_budget)
    tail_block_parts = [format_trait_flat(it.meta.get("name", ""), it.text,
                                          it.meta.get("domain_applications") or {})
                        for it in result.items]

    block = "\n".join([_HEADER] + core_block_parts + tail_block_parts)
    if len(block) > total_budget:
        # Only reachable if a SINGLE core trait's verbatim text alone exceeds the
        # whole budget (pathological; real descriptions are ~250 chars). We NEVER
        # truncate a trait (guardrail 2) — surface the overrun instead of hiding it.
        logger.warning(
            "personality retrieval block %d chars exceeds budget %d "
            "(oversized single trait kept verbatim, not truncated)",
            len(block), total_budget,
        )
    meta = {"windowed": True, "core_ids": core_ids, "flat_block": flat_block,
            "core_used": core_used, "tail_budget": tail_budget}
    return block, result, meta


# ─────────────────────────────────────────────────────────────────────────────
# Shadow entry point + trace
# ─────────────────────────────────────────────────────────────────────────────
def assemble_shadow_context(cue_text: str, conversation_id: str = "") -> tuple:
    """Assemble the retrieval block from the LIVE confirmed set and emit the
    carry-forward trace. Returns (block_str, RetrievalResult, meta). Read-only on
    the trait registry; touches NOTHING the Hydrinia snapshot reads."""
    from agent.personality.config import get_personality_config
    from agent.personality.registry import get_trait_registry

    cfg = get_personality_config()
    total_budget = int(getattr(cfg, "context_max_chars", 15000) or 15000)
    core_budget = min(CORE_CHAR_FLOOR, total_budget)

    confirmed = get_trait_registry().get_confirmed_traits()  # confirmed-only, date_confirmed asc
    block, result, meta = _assemble(confirmed, cue_text, total_budget, core_budget)

    _emit_trace(cue_text, conversation_id, confirmed, result, meta, block, total_budget)
    return block, result, meta


def _emit_trace(cue_text, conversation_id, confirmed, result, meta, block, total_budget) -> None:
    try:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "conversation_id": conversation_id,
            # (2) raw cue = source-of-truth + versioned encoding id
            "cue_raw": cue_text,
            "cue_encoding_id": ENCODING_ID,
            "query_repr": list(result.query_repr),
            # (3) opaque per-turn affective/focus blob — EMPTY placeholder (none exists)
            "context": {},
            # (4) versioned scorer ids (base vs rerank → agreement is computable later)
            "scorer_ids": {"base": BASE_SCORER_ID, "rerank": RERANK_SCORER_ID},
            "health": result.health,
            "windowed": meta.get("windowed", False),
            "n_confirmed": len(confirmed),
            "core_ids": meta.get("core_ids", []),
            # (1) FULL candidate set base/rerank scores — highest-value signal
            "candidates": [rt.to_row() for rt in result.all_scored],
            # (5) flat-baseline set beside the retrieved subset (the A/B needs both)
            "flat_baseline_ids": [t.id for t in confirmed],
            "tail_selected_ids": [rt.trait_id for rt in result.items],
            "shadow_len": len(block),
            "flat_len": len(meta.get("flat_block", "")),
            "byte_identical_to_flat": (not meta.get("windowed", False)
                                       and block == meta.get("flat_block", "")),
            "total_budget": total_budget,
            # re-rank deadline instrumentation (F-2 hardening): truncated ⟺ the wire
            # call hit the deadline (biases flip evidence + the determinism bar → watch
            # its rate once the scorer is live).
            "rerank_reason": getattr(result, "rerank_reason", None),
            "truncated": (getattr(result, "rerank_reason", None) == "timeout"),
            "within_deadline": (getattr(result, "rerank_reason", None) != "timeout"),
        }
        _enqueue_trace(rec)
    except Exception as e:  # observability must never break assembly
        logger.debug("Shadow trace emit skipped: %s", e)


# Single-worker pool for OFF-LOOP shadow assembly. Lazily created only when the
# master gate is on, so a disabled deploy spawns nothing. The event-loop thread
# only submits (returns immediately) — no assembly CPU, no I/O, no flock on-loop.
_shadow_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
_shadow_pool_lock = threading.Lock()


def _get_shadow_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _shadow_pool
    if _shadow_pool is None:
        with _shadow_pool_lock:
            if _shadow_pool is None:
                _shadow_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="personality-shadow")
    return _shadow_pool


def shutdown_shadow_executor(wait: bool = False) -> None:
    """Release shadow personality worker threads during process shutdown."""
    global _shadow_pool
    with _shadow_pool_lock:
        pool = _shadow_pool
        _shadow_pool = None
    if pool is not None:
        pool.shutdown(wait=wait, cancel_futures=True)


def _run_shadow_offloop(cue_text: str, conversation_id: str, flat_context: str) -> None:
    """Shadow assembly + trace, run OFF the event-loop thread. Fire-and-forget:
    the result is only logged (never injected), so nothing crosses back."""
    try:
        block, result, meta = assemble_shadow_context(cue_text, conversation_id)
        logger.info(
            "[personality-shadow] health=%s windowed=%s n_conf=%d core=%d tail=%d "
            "shadow_len=%d flat_len=%d byte_identical=%s",
            result.health, meta.get("windowed"), len(result.all_scored),
            len(meta.get("core_ids", [])), result.n_returned,
            len(block), len(flat_context),
            (not meta.get("windowed") and block == flat_context),
        )
    except Exception as e:
        logger.debug("shadow off-loop assembly failed: %s", e)


def maybe_shadow_personality(cue_text: str, conversation_id: str = "",
                             flat_context: str = "") -> Optional[str]:
    """Called from the system-prompt personality injection (on the event-loop thread).

    Returns None when the flat context should be used as-is — the default in
    LOG_ONLY / shadow mode (ZERO behavioral change: the injected prompt is the live
    flat block). Returns the retrieval block ONLY when the master gate is on AND
    LOG_ONLY has been explicitly flipped off (operator-gated promotion).

    In SHADOW mode the assembly + trace run OFF the event-loop thread (fire-and-
    forget to a single-worker pool), so the loop bears only a submit — never the
    assembly CPU, file I/O, or any lock held beyond microseconds. Only the ACTIVE
    (promoted) path assembles synchronously, because its block must be RETURNED for
    injection. Never raises into the assembly path."""
    if not is_enabled():
        return None
    try:
        if is_log_only():
            # SHADOW: off-loop assemble + log; never inject. On-loop cost ≈ a submit.
            try:
                _get_shadow_pool().submit(
                    _run_shadow_offloop, cue_text, conversation_id, flat_context)
            except Exception as e:
                logger.debug("shadow dispatch skipped: %s", e)
            return None
        # ACTIVE (operator-gated promotion): assemble synchronously to inject.
        block, _result, _meta = assemble_shadow_context(cue_text, conversation_id)
        return block or None
    except Exception as e:
        logger.debug("maybe_shadow_personality failed (using flat): %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Write-behind trace logger — off the event loop (no flock, no blocking I/O).
# A single daemon writer drains a bounded queue; the event-loop thread only
# enqueues (non-blocking). Mirrors the S209 write-behind flusher pattern.
# ─────────────────────────────────────────────────────────────────────────────
_trace_q: "queue.Queue" = queue.Queue(maxsize=2000)
_writer_started = False
_writer_lock = threading.Lock()


def _trace_writer_loop() -> None:
    while True:
        try:
            line = _trace_q.get()
        except Exception:
            continue
        try:
            _rotate_if_needed()
            os.makedirs(os.path.dirname(_TRACE_PATH), exist_ok=True)
            with open(_TRACE_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.debug("Shadow trace write failed: %s", e)
        finally:
            try:
                _trace_q.task_done()
            except Exception:
                pass


def _rotate_if_needed() -> None:
    try:
        if os.path.isfile(_TRACE_PATH) and os.path.getsize(_TRACE_PATH) > _TRACE_MAX_BYTES:
            os.replace(_TRACE_PATH, _TRACE_PATH + ".1")  # rename, not delete (hook-safe)
    except Exception:
        pass


def _ensure_writer() -> None:
    global _writer_started
    if _writer_started:
        return
    with _writer_lock:
        if _writer_started:
            return
        t = threading.Thread(target=_trace_writer_loop, name="personality-shadow-trace", daemon=True)
        t.start()
        _writer_started = True


def _enqueue_trace(record: dict) -> None:
    """Non-blocking enqueue from the event-loop thread. Drops on a full queue or
    serialize error (fail-open — the trace is observability, never load-bearing)."""
    try:
        _ensure_writer()
        _trace_q.put_nowait(json.dumps(record, default=str))
    except Exception:
        pass
