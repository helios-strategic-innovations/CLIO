"""
Cross-Category Prediction Tracker — Financial ML Insight #22
==============================================================
Kelly et al. (2020a) Principal Portfolios: off-diagonal cross-prediction
contains massive exploitable alpha distinct from self-prediction.

Moltbot adaptation: tracks which knowledge CATEGORY PAIRS co-occur
around successful trade ideas. When categories A and B are both active
in the ±48h window around a trade idea, and the trade wins, the
(A, B) pair gets a win recorded.

Over time, this reveals which information combinations produce
synergistic alpha — the knowledge-domain cross-prediction matrix.

Feature flags:
  MOLTBOT_CROSS_CATEGORY_ENABLED (default "0" — disabled per CLAUDE.md;
                                   25% win rate observed in earlier evaluation)
  MOLTBOT_CROSS_CATEGORY_LOG_ONLY (default "1")

Cost: ZERO API calls — pure bookkeeping on existing reward signals.
Author: Atlas Strategies — Aidan Tan (Momentum/Catalysts)
Paper: Kelly & Xiu (2024) SSRN #4501707, Insight #22
"""

import fcntl
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("openclaw.cross_category")

# Configuration
COOCCURRENCE_WINDOW_HOURS = 48   # Look ±48h around trade idea timestamp
MIN_SAMPLES_FOR_SIGNAL = 8       # Minimum pair observations before trusting
SYNERGY_THRESHOLD = 0.10          # Pair win rate must exceed solo by this much
STATE_FILE = "cross_category_state.json"

# Knowledge index path
KNOWLEDGE_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "knowledge", "knowledge_index.json",
)


class CategoryPairStats:
    """Tracks win/loss statistics for a category pair co-occurrence."""

    def __init__(self, cat_a: str, cat_b: str):
        self.cat_a = cat_a
        self.cat_b = cat_b
        self.wins = 0
        self.losses = 0
        self.last_updated = ""

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        if self.total == 0:
            return 0.5  # Bayesian prior
        return self.wins / self.total

    @property
    def is_significant(self) -> bool:
        return self.total >= MIN_SAMPLES_FOR_SIGNAL

    def to_dict(self) -> dict:
        return {
            "cat_a": self.cat_a,
            "cat_b": self.cat_b,
            "wins": self.wins,
            "losses": self.losses,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CategoryPairStats":
        obj = cls(data.get("cat_a", ""), data.get("cat_b", ""))
        obj.wins = data.get("wins", 0)
        obj.losses = data.get("losses", 0)
        obj.last_updated = data.get("last_updated", "")
        return obj


class CrossCategoryPredictor:
    """
    Tracks cross-category co-occurrence patterns around trade outcomes.

    When a binary reward fires (trade win/loss), this engine:
    1. Looks up all knowledge entries ±48h of the trade idea creation
    2. Groups entries by category
    3. Records win/loss for every category PAIR that was active
    4. Maintains rolling pair accuracy statistics
    5. Surfaces synergistic pairs (pairs that outperform solo categories)
    """

    def __init__(self, state_dir: Optional[str] = None):
        if state_dir is None:
            state_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "knowledge",
            )
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Category pair stats: key = "catA|catB" (alphabetically sorted)
        self._pair_stats: Dict[str, CategoryPairStats] = {}

        # Solo category stats for comparison
        self._solo_stats: Dict[str, CategoryPairStats] = {}  # Reuse struct, cat_b = "SOLO"

        self._load_state()
        logger.info(
            f"CrossCategoryPredictor initialized: "
            f"pairs={len(self._pair_stats)}, "
            f"solo={len(self._solo_stats)}, "
            f"log_only={os.environ.get('MOLTBOT_CROSS_CATEGORY_LOG_ONLY', '1') == '1'}"
        )

    @staticmethod
    def _pair_key(cat_a: str, cat_b: str) -> str:
        """Create canonical key for category pair (alphabetical order)."""
        return "|".join(sorted([cat_a, cat_b]))

    def record_trade_outcome(
        self,
        ticker: str,
        won: bool,
        source_category: str,
        trade_idea_timestamp: str,
    ):
        """
        Record a trade outcome and update cross-category statistics.

        Called from seal_reward_engine after recording a binary reward.

        Args:
            ticker: The traded ticker
            won: Whether the trade was profitable
            source_category: The primary knowledge category of the trade idea
            trade_idea_timestamp: ISO timestamp of when the trade idea was created
        """
        # S151-B H-5: Default disabled — 25% win rate across 3,243 evaluations (worse than random)
        if os.environ.get("MOLTBOT_CROSS_CATEGORY_ENABLED", "0") != "1":
            return

        try:
            idea_dt = datetime.fromisoformat(
                trade_idea_timestamp.replace("Z", "+00:00")
            )
            if idea_dt.tzinfo is None:
                idea_dt = idea_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            logger.debug(f"CrossCategory: invalid timestamp {trade_idea_timestamp}")
            return

        now = datetime.now(timezone.utc).isoformat()

        # Find active categories in the co-occurrence window
        # (I/O outside lock — reading knowledge_index is read-only)
        active_categories = self._find_active_categories(idea_dt)

        # Always include the source category
        active_categories.add(source_category)

        with self._lock:
            # Record solo stats for each active category
            for cat in active_categories:
                if cat not in self._solo_stats:
                    self._solo_stats[cat] = CategoryPairStats(cat, "SOLO")
                solo = self._solo_stats[cat]
                if won:
                    solo.wins += 1
                else:
                    solo.losses += 1
                solo.last_updated = now

            # Record pair stats for every combination
            active_list = sorted(active_categories)
            pairs_recorded = 0
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    key = self._pair_key(active_list[i], active_list[j])
                    if key not in self._pair_stats:
                        self._pair_stats[key] = CategoryPairStats(
                            active_list[i], active_list[j]
                        )
                    pair = self._pair_stats[key]
                    if won:
                        pair.wins += 1
                    else:
                        pair.losses += 1
                    pair.last_updated = now
                    pairs_recorded += 1

            self._save_state()

        logger.info(
            f"CROSS_CATEGORY: {ticker} {'WIN' if won else 'LOSS'} — "
            f"active_categories={len(active_categories)} "
            f"({', '.join(sorted(active_categories)[:4])}{'...' if len(active_categories) > 4 else ''}), "
            f"pairs_updated={pairs_recorded} "
            f"{'[LOG_ONLY]' if os.environ.get('MOLTBOT_CROSS_CATEGORY_LOG_ONLY', '1') == '1' else '[LIVE]'}"
        )

    def _find_active_categories(self, center_dt: datetime) -> Set[str]:
        """
        Find knowledge categories with entries within ±COOCCURRENCE_WINDOW_HOURS
        of the given timestamp.
        """
        window_start = center_dt - timedelta(hours=COOCCURRENCE_WINDOW_HOURS)
        window_end = center_dt + timedelta(hours=COOCCURRENCE_WINDOW_HOURS)

        active: Set[str] = set()
        # S122 1D / S121 CR-3: Add LOCK_SH for concurrent read safety
        lock_fd = None
        try:
            lock_path = str(KNOWLEDGE_INDEX_PATH) + ".lock"
            lock_fd = open(lock_path, "a")
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            with open(KNOWLEDGE_INDEX_PATH, "r") as f:
                index = json.load(f)

            entries = index.get("entries", [])
            if not isinstance(entries, list):
                return active
            for entry in entries:
                ts_str = entry.get("created", entry.get("timestamp", ""))
                if not ts_str:
                    continue
                try:
                    entry_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue

                if window_start <= entry_dt <= window_end:
                    cat = entry.get("category", "")
                    if cat:
                        active.add(cat)

        except Exception as e:
            logger.debug(f"CrossCategory: failed to scan knowledge index: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception as e:
                    logger.debug(f"Lock cleanup: {e}")

        return active

    def _get_synergistic_pairs_unlocked(self) -> List[dict]:
        """Internal helper — caller MUST hold self._lock."""
        synergies = []
        for key, pair in self._pair_stats.items():
            if not pair.is_significant:
                continue

            solo_a = self._solo_stats.get(pair.cat_a)
            solo_b = self._solo_stats.get(pair.cat_b)

            solo_a_rate = solo_a.win_rate if solo_a and solo_a.is_significant else 0.5
            solo_b_rate = solo_b.win_rate if solo_b and solo_b.is_significant else 0.5

            # Synergy = pair rate - max(solo_a, solo_b)
            max_solo = max(solo_a_rate, solo_b_rate)
            synergy = pair.win_rate - max_solo

            if synergy >= SYNERGY_THRESHOLD:
                synergies.append({
                    "pair": f"{pair.cat_a} + {pair.cat_b}",
                    "pair_win_rate": round(pair.win_rate, 3),
                    "solo_a_rate": round(solo_a_rate, 3),
                    "solo_b_rate": round(solo_b_rate, 3),
                    "synergy": round(synergy, 3),
                    "samples": pair.total,
                })

        return sorted(synergies, key=lambda x: x["synergy"], reverse=True)

    def get_synergistic_pairs(self) -> List[dict]:
        """
        Return category pairs whose co-occurrence win rate significantly
        exceeds both solo rates. This is the off-diagonal alpha.

        Returns list of dicts: {pair, pair_win_rate, solo_a_rate, solo_b_rate, synergy, samples}
        """
        if os.environ.get("MOLTBOT_CROSS_CATEGORY_ENABLED", "0") != "1":
            return []
        with self._lock:
            return self._get_synergistic_pairs_unlocked()

    def get_category_pair_context(self, current_category: str) -> str:
        """
        Generate context for the learning engine about which categories
        synergize with the current one.

        Used to inform Living Perspective synthesis and category scheduling.
        """
        lines = []
        with self._lock:
            for key, pair in self._pair_stats.items():
                if not pair.is_significant:
                    continue
                if pair.cat_a != current_category and pair.cat_b != current_category:
                    continue

                partner = pair.cat_b if pair.cat_a == current_category else pair.cat_a
                solo = self._solo_stats.get(current_category)
                solo_rate = solo.win_rate if solo and solo.is_significant else 0.5

                if pair.win_rate > solo_rate + 0.05:
                    lines.append(
                        f"  SYNERGY: {current_category} + {partner} → "
                        f"{pair.win_rate:.0%} win ({pair.total} samples) "
                        f"vs {current_category} solo {solo_rate:.0%}"
                    )

        if lines:
            return f"CROSS-CATEGORY SIGNALS for {current_category}:\n" + "\n".join(lines)
        return ""

    def get_stats(self) -> dict:
        """Return engine statistics for MCP dashboard."""
        with self._lock:
            synergies = self._get_synergistic_pairs_unlocked()
            return {
                "enabled": os.environ.get("MOLTBOT_CROSS_CATEGORY_ENABLED", "0") == "1",
                "log_only": os.environ.get("MOLTBOT_CROSS_CATEGORY_LOG_ONLY", "1") == "1",
                "total_pairs_tracked": len(self._pair_stats),
                "significant_pairs": sum(1 for p in self._pair_stats.values() if p.is_significant),
                "solo_categories_tracked": len(self._solo_stats),
                "synergistic_pairs": len(synergies),
                "top_synergies": synergies[:5],
            }

    def _load_state(self):
        """Load persistent state."""
        state_file = self._state_dir / STATE_FILE
        if not state_file.exists():
            return
        lock_fd = None
        try:
            lock_fd = open(str(state_file) + ".lock", "a")
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            data = json.loads(state_file.read_text())
            for key, pair_data in data.get("pairs", {}).items():
                self._pair_stats[key] = CategoryPairStats.from_dict(pair_data)
            for key, solo_data in data.get("solo", {}).items():
                self._solo_stats[key] = CategoryPairStats.from_dict(solo_data)
        except Exception as e:
            logger.warning(f"CrossCategory: failed to load state: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception as e:
                    logger.debug(f"Lock cleanup: {e}")

    def _save_state(self):
        """Persist state atomically."""
        state_file = self._state_dir / STATE_FILE
        lock_fd = None
        try:
            data = {
                "pairs": {k: v.to_dict() for k, v in self._pair_stats.items()},
                "solo": {k: v.to_dict() for k, v in self._solo_stats.items()},
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            lock_fd = open(str(state_file) + ".lock", "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            tmp = state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(state_file)
        except Exception as e:
            logger.error(f"CrossCategory: failed to save state: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception as e:
                    logger.debug(f"Lock cleanup: {e}")


# Module singleton — thread-safe double-checked locking
_predictor: Optional[CrossCategoryPredictor] = None
_predictor_lock = threading.Lock()


def get_cross_category_predictor() -> CrossCategoryPredictor:
    """Get or create singleton instance."""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = CrossCategoryPredictor()
    return _predictor
