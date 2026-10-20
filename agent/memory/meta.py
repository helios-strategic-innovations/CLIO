"""
Meta-Memory — Tier 5
Memory about memory: what I know, what I don't, how well-calibrated I am.

Tracks:
- Knowledge gaps: topics I've been asked about where my knowledge was thin
- Confidence calibration: were my stated confidence levels accurate?
- Domain coverage: which domains am I strong/weak in?

Modelled on Hydrinia's sector_performance tracking — quantitative self-awareness.
"""

import copy
import fcntl
import json
import logging
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("moltbot.memory.meta")

# S162 L-6: Use shared constant from package
from agent.memory import MEMORY_DATA_DIR as DATA_DIR


class MetaMemory:
    """
    Self-awareness about knowledge state.
    Tracks gaps, calibration, and domain coverage.
    """

    def __init__(self):
        self._file_path = os.path.join(DATA_DIR, "meta_memory.json")
        self._lock = threading.Lock()  # S136: protects _data dict across threads
        self._dirty = False
        self._data = {
            "gaps": [],             # Topics where I lacked knowledge
            "calibration": [],      # Confidence predictions vs actual outcomes
            "domain_coverage": {},  # Domain -> {queries: N, confident: N, uncertain: N}
            "corrections": [],      # Times Dominic corrected me
            "last_updated": None,
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._file_path):
            # S121 W-43: Added fcntl.flock for concurrent read safety
            lock_path = self._file_path + ".lock"
            lock_fd = None
            try:
                lock_fd = open(lock_path, "a")
                fcntl.flock(lock_fd, fcntl.LOCK_SH)
                with open(self._file_path, "r") as f:
                    self._data = json.load(f)
                logger.info(f"Meta-memory loaded: {len(self._data.get('gaps', []))} gaps, "
                           f"{len(self._data.get('calibration', []))} calibration entries")
            except Exception as e:
                logger.error(f"Failed to load meta-memory: {e}")
            finally:
                if lock_fd:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        lock_fd.close()
                    except Exception:
                        pass

    def _save(self) -> None:
        # S148 W-1: Standardized lock ordering — threading.Lock first (snapshot), then fcntl (write)
        # Previous: fcntl -> threading.Lock. Now matches working_memory.py pattern.
        with self._lock:
            # S122 4C: Clean up old unresolved minor gaps (>14 days)
            try:
                gaps = self._data.get("gaps", [])
                if gaps:
                    _cutoff = datetime.now(timezone.utc) - timedelta(days=14)
                    def _gap_not_expired(g):
                        if g.get("resolved") or g.get("severity") != "minor":
                            return True
                        ts = g.get("timestamp", "")
                        if not ts:
                            return True
                        try:
                            dt = datetime.fromisoformat(ts)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            return dt > _cutoff
                        except (ValueError, TypeError):
                            return True
                    self._data["gaps"] = [g for g in gaps if _gap_not_expired(g)]
            except Exception as e:
                logger.debug(f"Gap TTL cleanup skipped: {e}")

            self._data["last_updated"] = datetime.now(timezone.utc).isoformat()
            data_copy = copy.deepcopy(self._data)  # S162 M-7: Proper deep copy under lock
            self._dirty = False
        # S121 W-43: fcntl.flock for inter-process write safety (outside threading.Lock)
        lock_path = self._file_path + ".lock"
        lock_fd = None
        try:
            lock_fd = open(lock_path, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            tmp_path = self._file_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data_copy, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._file_path)
        except Exception as e:
            logger.error(f"Failed to save meta-memory: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass

    def _mark_dirty(self):
        self._dirty = True

    def flush(self):
        """Force-save if dirty. Called by export_hot_context and shutdown."""
        with self._lock:
            needs_save = self._dirty
        if needs_save:
            self._save()

    # === GAP TRACKING ===

    def record_gap(self, topic: str, domain: str, context: str = "",
                   severity: str = "minor") -> None:
        """
        Record a knowledge gap — something I didn't know or was uncertain about.
        severity: "minor" (could improve), "moderate" (noticeably weak), "major" (failed to answer)
        """
        gap = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "domain": domain,
            "context": context[:200],
            "severity": severity,
            "resolved": False,
            "resolution": None,
        }
        with self._lock:
            self._data.setdefault("gaps", []).append(gap)
            self._data["gaps"] = self._data["gaps"][-200:]
        self._save()
        logger.info(f"Knowledge gap recorded: {topic} ({domain}, {severity})")

    def resolve_gap(self, topic: str, resolution: str = "learned") -> bool:
        """Mark a knowledge gap as resolved."""
        with self._lock:
            for gap in reversed(self._data.get("gaps", [])):
                if gap["topic"].lower() == topic.lower() and not gap["resolved"]:
                    gap["resolved"] = True
                    gap["resolution"] = resolution
                    gap["resolved_at"] = datetime.now(timezone.utc).isoformat()
                    break
            else:
                return False
        self._save()
        return True

    def get_unresolved_gaps(self, domain: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get unresolved knowledge gaps, newest first."""
        with self._lock:
            gaps = [g for g in self._data.get("gaps", [])
                    if not g.get("resolved")]
        if domain:
            gaps = [g for g in gaps if g.get("domain") == domain]
        return sorted(gaps, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

    # === CALIBRATION TRACKING ===

    # S156: Only accept calibration from trade-relevant domains
    _VALID_CALIBRATION_DOMAINS = {
        "trade_decision", "market_forecast", "trade_idea", "trade_ideas",
        "position_sizing", "uw_calibration",
    }

    def record_prediction(self, prediction_id: str, domain: str,
                          stated_confidence: float, description: str = "",
                          replace_resolved: bool = False) -> None:
        """Record a confidence prediction for later calibration.

        S122 4B: Upserts on prediction_id — replaces existing unresolved entry
        with same ID instead of appending duplicates.
        S156: Filters non-prediction domains (personality, books, general chat).
        replace_resolved: also replace an entry whose outcome is already recorded
        (used by callers that resolve the same stable prediction_id right after
        every write, so the default unresolved-only match would never dedupe).
        """
        if not domain or domain not in self._VALID_CALIBRATION_DOMAINS:
            logger.debug("Skipping calibration for non-prediction domain: %s", domain)
            return
        entry = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domain": domain,
            "stated_confidence": stated_confidence,
            "description": description[:200],
            "actual_outcome": None,  # Filled in later
            "calibration_error": None,
        }
        with self._lock:
            cals = self._data.setdefault("calibration", [])
            # Upsert: replace existing unresolved entry with same prediction_id
            for i, existing in enumerate(cals):
                if (existing.get("prediction_id") == prediction_id
                        and (replace_resolved or existing.get("actual_outcome") is None)):
                    cals[i] = entry
                    break
            else:
                cals.append(entry)
            self._data["calibration"] = cals[-500:]
        self._save()

    def record_outcome(self, prediction_id: str, was_correct: bool) -> bool:
        """Record whether a prediction was correct. Computes calibration error."""
        with self._lock:
            for entry in reversed(self._data.get("calibration", [])):
                if entry.get("prediction_id") == prediction_id and entry.get("actual_outcome") is None:
                    entry["actual_outcome"] = was_correct
                    actual = 1.0 if was_correct else 0.0
                    entry["calibration_error"] = abs(entry["stated_confidence"] - actual)
                    break
            else:
                return False
        self._save()
        return True

    def get_calibration_score(self, domain: Optional[str] = None) -> Dict:
        """
        How well-calibrated am I? Returns average calibration error.
        Perfect calibration = 0.0, worst = 1.0.
        Modelled on Hydrinia's conviction accuracy tracking.
        """
        with self._lock:
            entries = [e for e in self._data.get("calibration", [])
                       if e.get("calibration_error") is not None]
        if domain:
            entries = [e for e in entries if e.get("domain") == domain]

        if not entries:
            return {"calibration_error": None, "sample_size": 0, "status": "insufficient_data"}

        avg_error = sum(e["calibration_error"] for e in entries) / len(entries)
        accuracy = sum(1 for e in entries if e.get("actual_outcome")) / len(entries)

        return {
            "calibration_error": round(avg_error, 3),
            "accuracy": round(accuracy, 3),
            "sample_size": len(entries),
            "status": "well_calibrated" if avg_error < 0.2 else "overconfident" if avg_error > 0.4 else "moderate",
        }

    # === CORRECTION TRACKING ===

    def record_correction(self, topic: str, domain: str, what_i_said: str,
                          what_was_correct: str) -> None:
        """Record when Dominic corrects me — critical for learning."""
        correction = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "domain": domain,
            "what_i_said": what_i_said[:300],
            "what_was_correct": what_was_correct[:300],
        }
        with self._lock:
            self._data.setdefault("corrections", []).append(correction)
            self._data["corrections"] = self._data["corrections"][-100:]
        self._save()
        logger.info(f"Correction recorded: {topic} ({domain})")

    # === DOMAIN COVERAGE ===

    # Confidence levels for calibration tracking (checked highest-first)
    _CONFIDENCE_VERY_HIGH = ("i'm certain", "definitely", "100%", "no doubt")
    _CONFIDENCE_HIGH = ("i'm confident", "high probability", "very likely", "strongly believe")
    _CONFIDENCE_MODERATE = ("i think", "likely", "probably", "should be")
    _CONFIDENCE_LOW = ("possibly", "might", "could be", "uncertain")

    # Hedging phrases that indicate uncertainty — if present, classify as uncertain
    # NOTE: "i think" is in _CONFIDENCE_MODERATE (0.70), NOT here — it's moderate confidence, not hedging
    # "probably" and "possibly" intentionally EXCLUDED — they have calibrated confidence levels
    # (0.70 and 0.50 respectively) in _CONFIDENCE_MODERATE/_LOW. Including them here would
    # create contradictory signals (confident + uncertain simultaneously).
    _HEDGING_PATTERNS = (
        "not sure", "may be", "might", "perhaps",
        "unclear", "uncertain", "i believe", "it seems",
        "i'm not certain", "hard to say", "difficult to determine",
    )

    def record_domain_query(self, domain: str, confident: bool,
                            content: str = "") -> None:
        """Track that I was queried in a domain, and whether I was confident.

        If content is provided, hedging language overrides the confident flag
        to reduce calibration inflation. When uncertainty is detected, also
        records a knowledge gap (Fix E, Session 89).
        """
        # Hedging language in content overrides the confident flag
        detected_hedging = False
        if confident and content:
            content_lower = content.lower()
            if self._phrase_match(self._HEDGING_PATTERNS, content_lower):
                confident = False
                detected_hedging = True

        with self._lock:
            coverage = self._data.setdefault("domain_coverage", {})
            if domain not in coverage:
                coverage[domain] = {"queries": 0, "confident": 0, "uncertain": 0}
            coverage[domain]["queries"] += 1
            if confident:
                coverage[domain]["confident"] += 1
            else:
                coverage[domain]["uncertain"] += 1
            self._mark_dirty()  # H-6: inside lock, not outside

        # Auto-record knowledge gap when hedging is detected (Fix E, Session 89)
        # S141: Snapshot gaps under lock, check outside, call record_gap outside (non-reentrant lock)
        if detected_hedging and content:
            topic = content[:80].strip()
            with self._lock:
                recent_gaps = list(self._data.get("gaps", [])[-20:])
            already_recorded = any(
                g.get("topic", "")[:40] == topic[:40] and g.get("domain") == domain
                for g in recent_gaps
            )
            if not already_recorded:
                self.record_gap(topic=topic, domain=domain,
                                context="Auto-detected from hedging language",
                                severity="minor")

        # S159 M-14: _save() outside lock (standardized) — _save() acquires its own lock to snapshot
        # M-7: force save after domain_coverage mutation (crash safety)
        self._save()

    def get_domain_coverage(self) -> Dict:
        """Get domain coverage statistics."""
        with self._lock:
            coverage = dict(self._data.get("domain_coverage", {}))  # Snapshot under lock
        result = {}
        for domain, stats in coverage.items():
            total = stats.get("queries", 0)
            confident = stats.get("confident", 0)
            result[domain] = {
                "queries": total,
                "confidence_rate": round(confident / total, 2) if total > 0 else 0,
            }
        return result

    # === SUMMARY ===

    def get_self_assessment(self) -> Dict:
        """
        Full self-assessment: where am I strong, where am I weak?
        The meta-cognition report.
        """
        gaps = self.get_unresolved_gaps(limit=50)
        calibration = self.get_calibration_score()
        coverage = self.get_domain_coverage()
        with self._lock:
            corrections = list(self._data.get("corrections", []))

        # Identify weakest domains
        weak_domains = sorted(
            [(d, s) for d, s in coverage.items()],
            key=lambda x: x[1].get("confidence_rate", 0)
        )[:3]

        return {
            "calibration": calibration,
            "unresolved_gaps": len(gaps),
            "top_gap_domains": list(set(g.get("domain", "?") for g in gaps[:10])),
            "total_corrections": len(corrections),
            "recent_corrections": corrections[-3:],
            "weakest_domains": [d for d, _ in weak_domains],
            "domain_coverage": coverage,
        }

    @staticmethod
    def _phrase_match(phrases: tuple, text: str) -> bool:
        """Check if any phrase matches with word boundaries."""
        for p in phrases:
            if re.search(r'\b' + re.escape(p.strip()) + r'\b', text, re.IGNORECASE):
                return True
        return False

    def detect_confidence_level(self, text: str) -> Optional[float]:
        """Detect stated confidence level in text. Returns float or None."""
        text_lower = text[:1000].lower()
        for phrases, level in (
            (self._CONFIDENCE_VERY_HIGH, 0.95),
            (self._CONFIDENCE_HIGH, 0.85),
            (self._CONFIDENCE_MODERATE, 0.70),
            (self._CONFIDENCE_LOW, 0.50),
        ):
            if self._phrase_match(phrases, text_lower):
                return level
        return None

    def get_unresolved_calibration(self, domain: str) -> List[Dict]:
        """Get unresolved calibration entries for a domain."""
        with self._lock:
            return [e for e in self._data.get("calibration", [])
                    if e.get("actual_outcome") is None and e.get("domain") == domain]

    def get_calibration_by_prediction_id(self, prediction_id: str) -> List[Dict]:
        """Get calibration entries matching a prediction_id."""
        with self._lock:
            return [c for c in self._data.get("calibration", [])
                    if c.get("prediction_id") == prediction_id]

    def get_calibration_by_prefix(self, prefix: str) -> List[Dict]:
        """Get calibration entries whose prediction_id starts with prefix (thread-safe)."""
        with self._lock:
            return [c for c in self._data.get("calibration", [])
                    if str(c.get("prediction_id", "")).startswith(prefix)]

    def get_counts(self) -> Dict[str, int]:
        """Get counts for corrections, gaps, calibration."""
        with self._lock:
            return {
                "corrections": len(self._data.get("corrections", [])),
                "gaps": len(self._data.get("gaps", [])),
                "calibration": len(self._data.get("calibration", [])),
            }

    def stats(self) -> Dict:
        with self._lock:
            return {
                "total_gaps": len(self._data.get("gaps", [])),
                "unresolved_gaps": len([g for g in self._data.get("gaps", []) if not g.get("resolved")]),
                "calibration_entries": len(self._data.get("calibration", [])),
                "corrections": len(self._data.get("corrections", [])),
                "domains_tracked": len(self._data.get("domain_coverage", {})),
            }
