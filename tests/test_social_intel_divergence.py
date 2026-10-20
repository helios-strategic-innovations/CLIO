#!/usr/bin/env python3
"""Standalone tests for the §6.D divergence detector (CLIO has no pytest suite — runtime script).
Pure-logic + a source grep; NO production DB/file writes (CG10). Run: python3 tests/test_social_intel_divergence.py
Exit 0 = all pass, 1 = failure."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.integrations.social_intel_divergence import (  # noqa: E402
    UngroundedVerdict, build_verdict, classify_divergence, _persistence_gate, _PERSIST_K,
)

_FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        _FAILS.append(name)


def _voice_weight(is_amplification: bool, novel: bool, link_kind: str | None) -> float:
    """Mirrors social_intel.narrative_regime_agg's CASE (migration 018). Kept in lockstep with the SQL.
    is_amplification (echo/RT) -> 0; novel original -> 1; near_dup link (different author) -> 1;
    followup (same author repeating) / echo -> 0."""
    if is_amplification:
        return 0.0
    if novel:
        return 1.0
    if link_kind == "near_dup":
        return 1.0
    return 0.0  # followup / echo


def test_distinct_voice_single_account_spam():
    """THE binding refinement: a single account spamming one take must NOT read as a storm."""
    # single account posts the same take 50x: first novel (canonical), 49 followup (same author)
    spam = [_voice_weight(False, True, None)] + [_voice_weight(False, False, "followup") for _ in range(49)]
    # 50 DIFFERENT accounts post ~the same take: first novel, 49 near_dup (independent voices)
    storm = [_voice_weight(False, True, None)] + [_voice_weight(False, False, "near_dup") for _ in range(49)]
    # 50 retweets/amplifications of one take: all near-zero voice
    amp = [_voice_weight(True, False, "echo") for _ in range(50)]

    print("\n[distinct-voice weighting]")
    check("single-account spam totals 1 voice (not 50)", sum(spam) == 1.0, f"sum={sum(spam)}")
    check("multi-author storm totals 50 voice", sum(storm) == 50.0, f"sum={sum(storm)}")
    check("pure amplification totals ~0 voice", sum(amp) == 0.0, f"sum={sum(amp)}")
    check("spam << storm (spam cannot fake a storm)", sum(spam) < sum(storm) / 10)

    # and the downstream effect: a spam-driven 'narrative' has tiny intensity -> cannot lead
    narr_spam = {"score": 0.9, "intensity": sum(spam), "velocity": 0.0, "conviction": 0.05,
                 "top_themes": [{"theme": "geopolitics", "voice": 1}], "evidence": [{"type": "x"}]}
    narr_storm = {"score": 0.9, "intensity": sum(storm), "velocity": 1.0, "conviction": 0.9,
                  "top_themes": [{"theme": "geopolitics", "voice": 50}], "evidence": [{"type": "x"}]}
    calm_stat = {"score": 0.0, "conviction": 0.1, "evidence": [{"type": "y"}]}
    check("single-account spam does NOT classify NARRATIVE_LEADS",
          classify_divergence(narr_spam, calm_stat) != "NARRATIVE_LEADS",
          classify_divergence(narr_spam, calm_stat))
    check("genuine multi-author storm DOES classify NARRATIVE_LEADS",
          classify_divergence(narr_storm, calm_stat) == "NARRATIVE_LEADS")


def test_classify():
    print("\n[classify_divergence]")
    storm = {"score": 0.8, "conviction": 0.8}
    calm = {"score": 0.05, "conviction": 0.1}
    check("storm+calm -> NARRATIVE_LEADS", classify_divergence(storm, calm) == "NARRATIVE_LEADS")
    check("aligned -> AGREE", classify_divergence({"score": 0.5, "conviction": 0.6},
                                                  {"score": 0.5, "conviction": 0.6}) == "AGREE")
    check("opposed+both-strong -> CONFLICT",
          classify_divergence({"score": 0.7, "conviction": 0.7},
                              {"score": -0.7, "conviction": 0.7}) == "CONFLICT")
    check("stat strong, narr quiet -> STATISTICAL_LEADS",
          classify_divergence({"score": 0.1, "conviction": 0.1},
                              {"score": 0.8, "conviction": 0.8}) == "STATISTICAL_LEADS")


def test_persistence_gate():
    print("\n[persistence gate (anti-flap, k=%d)]" % _PERSIST_K)
    state = {"buffer": []}
    fired = [_persistence_gate(state, "NARRATIVE_LEADS") for _ in range(_PERSIST_K)]
    check("first k-1 passes do NOT fire", not any(fired[:-1]), str(fired[:-1]))
    check("k-th consecutive pass FIRES", fired[-1] is True)
    # a flap breaks the streak
    state2 = {"buffer": []}
    _persistence_gate(state2, "NARRATIVE_LEADS")
    _persistence_gate(state2, "AGREE")
    last = _persistence_gate(state2, "NARRATIVE_LEADS")
    check("a flap resets the streak (no fire)", last is False)


def test_evidence_required():
    print("\n[PI3 evidence required]")
    narr = {"score": 0.8, "intensity": 100, "velocity": 0.5, "conviction": 0.8,
            "top_themes": [], "evidence": []}
    stat = {"score": 0.0, "conviction": 0.1, "evidence": []}
    raised = False
    try:
        build_verdict("NARRATIVE_LEADS", narr, stat)
    except UngroundedVerdict:
        raised = True
    check("build_verdict raises UngroundedVerdict on empty evidence", raised)
    # with evidence it succeeds and stays non-empty
    narr["evidence"] = [{"type": "narrative_story", "story_id": 1}]
    v = build_verdict("NARRATIVE_LEADS", narr, stat)
    check("verdict with evidence is grounded (>=1 item)", len(v["evidence"]) >= 1)
    check("verdict confidence in [0,1]", 0.0 <= v["confidence"] <= 1.0)


def test_no_risk_write_path():
    print("\n[advisory-only: grep-proven no risk-write path]")
    import re
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "agent", "integrations", "social_intel_divergence.py")
    with open(src) as f:
        text = f.read()
    # strip ALL triple-quoted docstrings (where the PROHIBITION is *documented*) then line comments,
    # so the test inspects EXECUTABLE code only — a banned token surviving here would be a real write.
    code = re.sub(r'"""[\s\S]*?"""', "", text)
    code = re.sub(r"'''[\s\S]*?'''", "", code)
    code = "\n".join(ln.split("#", 1)[0] for ln in code.splitlines())
    banned = ("EMERGENCY_STOP", "execute_trade", "execute_options_trade", "atlas_team",
              "risk_envelope", "place_order", "submit_order", "control_hydrinia", "modify_hydrinia",
              "position_log")
    offenders = [b for b in banned if b in code]
    check("no risk-authority write tokens in executable code", not offenders, str(offenders))


if __name__ == "__main__":
    print("=== §6.D divergence detector tests ===")
    test_distinct_voice_single_account_spam()
    test_classify()
    test_persistence_gate()
    test_evidence_required()
    test_no_risk_write_path()
    print(f"\n{'ALL PASS' if not _FAILS else 'FAILURES: ' + ', '.join(_FAILS)}")
    sys.exit(1 if _FAILS else 0)
