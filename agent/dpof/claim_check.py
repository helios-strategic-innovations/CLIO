"""LOG-ONLY claim checker for DPOF decider text (stdlib only, pure, no I/O).

Flags statements in a decider thesis or review reason that contradict the
decider's own inputs (the ctx it was given). It never changes a decision: the
decider logs the violation codes and writes them to the attempt ledger row.

Rules (each returns a short code; an empty list means nothing contradicted):
  (i)   "A vs/against/versus [VWAP] B" with a relation word (above/below/
        reclaimed/lost/over/under) within 40 chars: contradicted when the word
        disagrees with sign(A - B).  -> vwap_relation:A<B:word
  (ii)  "X below/above" within 60 chars of an "A vs B" pair: contradicted when
        |X - |A - B|| > 0.02.           -> gap_mismatch:claimedX:actualG
  (iii) "single-pillar" / "only NOFA" style claims: contradicted when another
        pillar in scope has |value| >= 0.2 (bare "single-pillar" with no named
        pillar: contradicted when two or more pillars have |value| >= 0.2).
        -> single_pillar_claim:name=value

Input is capped at 1,500 characters, so the regex work is bounded for any text.
"""

from __future__ import annotations

import math
import re

MAX_CHARS = 1500
_GAP_TOLERANCE = 0.02
_RELATION_WINDOW = 40
_GAP_WINDOW = 60
_PILLAR_STRONG = 0.5   # another flow pillar must be STRONG to contradict a named single-pillar claim
_VWAP_CONTEXT_CHARS = 30   # a "A vs B" pair is a VWAP claim only when 'VWAP' is inside it or this close before it
# A "single-pillar" claim is about FLOW pillars. GEX is a two-state regime factor (+0.5 / -0.3, constant for hours)
# and Kronos is a forecast, so neither can contradict "single-pillar NOFA" (replay over 441 real texts: every GEX flag
# was a false positive).
_FLOW_PILLARS = frozenset({"nofa", "premium", "darkpool", "dark_pool"})

_NUM = r"\d+(?:\.\d+)?"
_PAIR_RE = re.compile(
    rf"(?<![\w.])({_NUM})\s*(?:vs\.?|against|versus)\s+(?:vwap\s*)?({_NUM})(?!\w|\.\d)",
    re.IGNORECASE,
)
_UP_RE = re.compile(r"\b(?:above|over|reclaimed|reclaims|reclaiming|reclaim)\b", re.IGNORECASE)
_DOWN_RE = re.compile(r"\b(?:below|under|lost|loses|losing|lose)\b", re.IGNORECASE)
_GAP_RE = re.compile(rf"({_NUM})\s*(?:points?|pts?|dollars?|usd|\$)?\s*(below|above|under|over)\b", re.IGNORECASE)
# Only EXPLICIT, named claims are judged ("single-pillar NOFA", "only premium"). Bare "single-pillar" is ordinary prose
# ("the same single-pillar fade seen this morning"): replay over 441 real texts showed it is not a checkable claim.
_SINGLE_RE = re.compile(
    r"\bsingle[- ]pillar\s+(nofa|premium|dark\s*pool|darkpool)\b"
    r"|\bonly\s+(nofa|premium|dark\s*pool|darkpool)\b",
    re.IGNORECASE,
)


def _pillars_in_scope(text: str, ctx: dict) -> list[tuple[str, float]]:
    """(pillar, value) pairs for the underlyings the text names (all of them when it names none)."""
    unds = ctx.get("underlyings") if isinstance(ctx, dict) else None
    if not isinstance(unds, dict):
        return []
    named = [sym for sym in unds if isinstance(sym, str) and re.search(rf"\b{re.escape(sym)}\b", text, re.IGNORECASE)]
    if not named:
        return []   # no underlying named: SPY's and QQQ's pillars would be mixed, so the claim is not checkable
    scope = named
    out: list[tuple[str, float]] = []
    for sym in scope:
        u = unds.get(sym)
        pillars = u.get("pillars") if isinstance(u, dict) else None
        if not isinstance(pillars, dict):
            continue
        for name, raw in pillars.items():
            if str(name).lower() not in _FLOW_PILLARS:
                continue
            val = raw.get("value") if isinstance(raw, dict) else raw
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                continue
            out.append((str(name).lower(), float(val)))
    return out


def _pair_relation(text: str, start: int, end: int) -> str | None:
    """The relation word nearest the pair, searching 40 chars before first, then after."""
    before = text[max(0, start - _RELATION_WINDOW): start]
    hits = list(_UP_RE.finditer(before)) + list(_DOWN_RE.finditer(before))
    if hits:
        last = max(hits, key=lambda m: m.end())
        return last.group(0).lower()
    after = text[end: end + _RELATION_WINDOW]
    hits = [m for m in list(_UP_RE.finditer(after)) + list(_DOWN_RE.finditer(after))]
    if hits:
        return min(hits, key=lambda m: m.start()).group(0).lower()
    return None


def check_claims(text: str, ctx: dict) -> list[str]:
    """Violation codes for statements in `text` that contradict `ctx`. Empty list when clean."""
    if not isinstance(text, str) or not text:
        return []
    text = text[:MAX_CHARS]
    codes: list[str] = []

    pairs = []
    for m in _PAIR_RE.finditer(text):
        if "vwap" not in text[max(0, m.start() - _VWAP_CONTEXT_CHARS): m.end()].lower():
            continue   # option marks ("mark 2.315 vs 2.29"), entries and other 'A vs B' pairs are not VWAP claims
        a_str, b_str = m.group(1), m.group(2)
        a_val, b_val = float(a_str), float(b_str)
        pairs.append((m.start(), m.end(), a_str, b_str, a_val, b_val))
        word = _pair_relation(text, m.start(), m.end())
        if word is None:
            continue
        if word in ("above", "over", "reclaimed", "reclaims", "reclaiming", "reclaim") and a_val < b_val:
            codes.append(f"vwap_relation:{a_str}<{b_str}:{word}")
        elif word in ("below", "under", "lost", "loses", "losing", "lose") and a_val > b_val:
            codes.append(f"vwap_relation:{a_str}>{b_str}:{word}")

    for m in _GAP_RE.finditer(text):
        claimed_str = m.group(1)
        claimed = float(claimed_str)
        near = [p for p in pairs if (m.end() <= p[0] and p[0] - m.end() <= _GAP_WINDOW)
                or (p[1] <= m.start() and m.start() - p[1] <= _GAP_WINDOW)]
        if not near:
            continue
        p = min(near, key=lambda q: min(abs(q[0] - m.end()), abs(m.start() - q[1])))
        actual = abs(p[4] - p[5])
        if round(abs(claimed - actual), 6) > _GAP_TOLERANCE:
            codes.append(f"gap_mismatch:claimed{claimed_str}:actual{actual:.2f}")

    single = [m for m in _SINGLE_RE.finditer(text)]
    if single:
        pillars = _pillars_in_scope(text, ctx)
        for m in single:
            named = next((g for g in m.groups() if g), None)
            if not named:
                continue
            named = re.sub(r"\s+", "", named.lower())
            others = [(n, v) for n, v in pillars if n.replace(" ", "") != named and abs(v) >= _PILLAR_STRONG]
            if others:
                n, v = max(others, key=lambda nv: abs(nv[1]))
                codes.append(f"single_pillar_claim:{n}={v:.2f}")

    seen: set[str] = set()
    unique: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique
