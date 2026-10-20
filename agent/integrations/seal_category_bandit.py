"""
SEAL Phase 6 — CategorySelectionBandit: Learned Per-Regime Category Rotation
=============================================================================
Replaces static weighted random category selection in the learning engine with
Thompson Sampling over Beta distributions. Learns which of the 15 knowledge
categories produce winning trade ideas per market regime.

Context buckets: 4 regimes x 15 categories = 60 Beta distributions.

Over time, the bandit converges on the optimal category allocation per regime:
  - BULL: maybe trading (momentum) and investing (targets) score highest
  - BEAR: maybe geopolitical and bonds score highest
  - CRISIS: maybe commodities and options (hedging) score highest

The bandit doesn't replace the category_weights system — it ENHANCES it.
Base weights (from M2-B reward adjustments) are multiplied by bandit scores,
giving a two-layer optimization: M2-B adjusts globally, bandit adjusts per-regime.

Feature flags (environment variables):
  MOLTBOT_SEAL_CATEGORY_BANDIT          — Enable bandit selection (default "1")
  MOLTBOT_SEAL_CATEGORY_BANDIT_LOG_ONLY — Log selections without overriding (default "1")

Author: Atlas Strategies — Phase 6 SEAL Implementation
"""

import json
import logging
import os
import random
import threading
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("openclaw.seal_category_bandit")

_NY = ZoneInfo("America/New_York")

# ============================================================
# Configuration
# ============================================================

# Regime labels (must match Hydrinia's market regime detection)
REGIMES = ("BULL", "BEAR", "CHOPPY", "CRISIS")

# Category names (must match LEARNING_CATEGORIES[i]["name"] in learning_engine.py)
CATEGORY_NAMES = (
    "trading", "geopolitical", "us_domestic_political", "finance", "investing",
    "history", "strategy", "psychology", "bonds", "equity",
    "crypto", "commodities", "futures", "options", "supply_chain",
    "global_geopolitical_intelligence",  # S135: Lighthouse RSS category
)

# S148-B W1-1: Disabled categories — excluded from bandit selection entirely
DISABLED_CATEGORIES = frozenset({"equity", "history", "options"})

# S148-B W1-1: Diversity penalty — recently-selected categories get score discount
_DIVERSITY_WINDOW = 8        # S156 M-B: Increased from 3 (~15min) to 8 (~40min) to reduce category spam
_DIVERSITY_DISCOUNT = 0.5    # Multiply score by this if selected recently

# State persistence
_BANDIT_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "knowledge", "seal_category_bandit_state.json",
)

# Minimum total pulls before bandit influences selection
# (below this, use uniform exploration to gather data)
MIN_PULLS_FOR_EXPLOITATION = 20


# ============================================================
# BetaParams
# ============================================================

class BetaParams:
    """Beta distribution parameters for Thompson Sampling."""

    __slots__ = ("alpha", "beta")

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def sample(self) -> float:
        """Draw from Beta(alpha, beta)."""
        return random.betavariate(max(self.alpha, 0.01), max(self.beta, 0.01))

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    @property
    def mean(self) -> float:
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    @property
    def pulls(self) -> int:
        """Total observations (subtract prior of 2)."""
        return max(0, int(self.alpha + self.beta - 2))

    def to_dict(self) -> dict:
        return {"alpha": round(self.alpha, 4), "beta": round(self.beta, 4)}

    @classmethod
    def from_dict(cls, d: dict) -> "BetaParams":
        return cls(alpha=d.get("alpha", 1.0), beta=d.get("beta", 1.0))


# ============================================================
# CategorySelectionBandit
# ============================================================

class CategorySelectionBandit:
    """Thompson Sampling bandit for adaptive learning category selection.

    Maintains Beta(alpha, beta) per (regime, category) pair.
    Updated after each binary trade reward with the source category.

    Thread-safe via _lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[str, dict[str, BetaParams]] = {}
        self._meta: dict = {"total_pulls": 0, "total_rewards": 0}
        self._recent_selections: list[str] = []  # S148-B: diversity tracking
        self._initialize_priors()
        self._load_state()
        logger.info(
            f"SEAL CategorySelectionBandit initialized: "
            f"{self._meta.get('total_pulls', 0)} total pulls across "
            f"{len(REGIMES) * len(CATEGORY_NAMES)} buckets"
        )

    def _initialize_priors(self) -> None:
        """Initialize Beta(1,1) uniform priors for active regime-category pairs."""
        for regime in REGIMES:
            if regime not in self._state:
                self._state[regime] = {}
            for cat in CATEGORY_NAMES:
                if cat in DISABLED_CATEGORIES:
                    continue
                if cat not in self._state[regime]:
                    self._state[regime][cat] = BetaParams()

    def _load_state(self) -> None:
        """Load persisted bandit state from disk."""
        try:
            if os.path.isfile(_BANDIT_STATE_FILE):
                with open(_BANDIT_STATE_FILE, "r") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    return
                self._meta = data.get("meta", self._meta)
                self._recent_selections = data.get("recent_selections", [])
                distributions = data.get("distributions", {})
                for regime in REGIMES:
                    regime_data = distributions.get(regime, {})
                    for cat in CATEGORY_NAMES:
                        cat_data = regime_data.get(cat)
                        if cat_data and isinstance(cat_data, dict):
                            self._state[regime][cat] = BetaParams.from_dict(cat_data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"SEAL bandit state load failed: {e}")

    def _save_state(self) -> None:
        """Persist bandit state atomically."""
        try:
            distributions = {}
            for regime in REGIMES:
                distributions[regime] = {}
                for cat in CATEGORY_NAMES:
                    if cat in DISABLED_CATEGORIES:
                        continue
                    distributions[regime][cat] = self._state[regime][cat].to_dict()

            data = {
                "meta": self._meta,
                "recent_selections": self._recent_selections[-50:],  # cap at 50
                "distributions": distributions,
            }
            os.makedirs(os.path.dirname(_BANDIT_STATE_FILE), exist_ok=True)
            tmp = _BANDIT_STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, _BANDIT_STATE_FILE)
        except OSError as e:
            logger.warning(f"SEAL bandit state save failed: {e}")

    def select_category(
        self,
        regime: str,
        base_weights: dict[str, float],
    ) -> str:
        """Select the next category to study via Thompson Sampling.

        Args:
            regime: Current market regime (BULL/BEAR/CHOPPY/CRISIS)
            base_weights: Static category_weights from learning_state.json

        Returns: Category name string (e.g., "trading", "geopolitical")
                 Empty string = caller uses default selection.

        Selection process:
        1. Sample from Beta(alpha, beta) for each category in the current regime
        2. Multiply sampled score by base_weight (two-layer optimization)
        3. Select the category with highest combined score
        """
        if os.environ.get("MOLTBOT_SEAL_CATEGORY_BANDIT", "1") != "1":
            return ""

        with self._lock:
            regime = regime.upper() if regime else "CHOPPY"
            if regime not in self._state:
                regime = "CHOPPY"

            total_pulls = self._meta.get("total_pulls", 0)

            # S151 H-1: Increment total_pulls on every selection attempt.
            # A "pull" in bandit terms is an arm selection. Without this,
            # MIN_PULLS_FOR_EXPLOITATION gate is dead code (total_pulls stuck at 0).
            self._meta["total_pulls"] = total_pulls + 1

            # Exploration phase: not enough data yet, return empty (use default)
            if total_pulls < MIN_PULLS_FOR_EXPLOITATION:
                logger.debug(
                    f"SEAL bandit: exploration phase ({total_pulls}/{MIN_PULLS_FOR_EXPLOITATION}), "
                    f"using default selection"
                )
                self._save_state()
                return ""

            # S148-B W1-1: Filter disabled categories from selection
            active_cats = [c for c in CATEGORY_NAMES if c not in DISABLED_CATEGORIES]

            # Thompson Sampling with diversity penalty
            _recent = self._recent_selections[-_DIVERSITY_WINDOW:]
            scores: dict[str, float] = {}
            for cat in active_cats:
                beta_params = self._state[regime].get(cat, BetaParams())
                sampled = beta_params.sample()
                base = base_weights.get(cat, 1.0)
                score = sampled * base
                # S148-B: Diversity discount — penalize recently-selected categories
                if cat in _recent:
                    score *= _DIVERSITY_DISCOUNT
                scores[cat] = score

            # Select highest-scoring category
            best_cat = max(scores, key=scores.get)  # type: ignore[arg-type]

            # Record selection for diversity tracking
            self._recent_selections.append(best_cat)
            if len(self._recent_selections) > 50:
                self._recent_selections = self._recent_selections[-50:]

            mode_tag = "[LOG_ONLY] " if os.environ.get("MOLTBOT_SEAL_CATEGORY_BANDIT_LOG_ONLY", "1") == "1" else ""
            logger.info(
                f"SEAL {mode_tag}bandit selected: {best_cat} "
                f"(regime={regime}, score={scores[best_cat]:.3f}, "
                f"mean={self._state[regime][best_cat].mean:.3f}, "
                f"pulls={self._state[regime][best_cat].pulls})"
            )

            # Save state to persist recent_selections
            self._save_state()

            if os.environ.get("MOLTBOT_SEAL_CATEGORY_BANDIT_LOG_ONLY", "1") == "1":
                return ""

            return best_cat

    def record_outcome(
        self,
        regime: str,
        category: str,
        success: bool,
    ) -> None:
        """Update the Beta distribution for a regime-category pair.

        Called when a trade closes and the source knowledge category is known.
        Typically wired from seal_reward_engine.py -> record_binary_reward().

        NOTE: Updates occur even in LOG_ONLY mode. This is intentional —
        the bandit pre-trains while observing, so when LOG_ONLY is disabled,
        it already has learned distributions and can make informed selections
        immediately rather than starting from uniform priors.

        Args:
            regime: Market regime at time of the original analysis
            category: Learning category that generated the knowledge
            success: True if the trade was profitable
        """
        if os.environ.get("MOLTBOT_SEAL_CATEGORY_BANDIT", "1") != "1":
            return

        with self._lock:
            regime = regime.upper() if regime else "CHOPPY"
            if regime not in self._state:
                regime = "CHOPPY"
            if category not in CATEGORY_NAMES or category in DISABLED_CATEGORIES:
                logger.debug(f"SEAL bandit: unknown/disabled category '{category}', ignoring")
                return

            self._state[regime][category].update(success)
            # S151: total_pulls is incremented in select_category() only (a "pull" = selection).
            # Removed duplicate increment here to avoid inflating win_rate denominator.
            if success:
                self._meta["total_rewards"] = self._meta.get("total_rewards", 0) + 1

            mode_tag = "[LOG_ONLY] " if os.environ.get("MOLTBOT_SEAL_CATEGORY_BANDIT_LOG_ONLY", "1") == "1" else ""
            logger.info(
                f"SEAL {mode_tag}bandit update: {category}@{regime} "
                f"{'WIN' if success else 'LOSS'} -> "
                f"mean={self._state[regime][category].mean:.3f} "
                f"(a={self._state[regime][category].alpha:.1f}, "
                f"b={self._state[regime][category].beta:.1f})"
            )

            self._save_state()

    def get_regime_report(self, regime: str) -> dict[str, dict]:
        """Return bandit state for a specific regime (for meta-memory/reporting)."""
        with self._lock:
            regime = regime.upper() if regime else "CHOPPY"
            if regime not in self._state:
                return {}
            report = {}
            for cat in CATEGORY_NAMES:
                if cat in DISABLED_CATEGORIES:
                    continue
                bp = self._state[regime].get(cat, BetaParams())
                report[cat] = {
                    "mean": round(bp.mean, 3),
                    "pulls": bp.pulls,
                    "alpha": round(bp.alpha, 2),
                    "beta": round(bp.beta, 2),
                }
            # Sort by mean descending
            return dict(sorted(report.items(), key=lambda x: x[1]["mean"], reverse=True))

    def get_status(self) -> dict:
        """Return overall bandit status."""
        with self._lock:
            return {
                "total_pulls": self._meta.get("total_pulls", 0),
                "total_rewards": self._meta.get("total_rewards", 0),
                "overall_win_rate": round(
                    self._meta.get("total_rewards", 0)
                    / max(self._meta.get("total_pulls", 0), 1),
                    3,
                ),
                "buckets": len(REGIMES) * (len(CATEGORY_NAMES) - len(DISABLED_CATEGORIES)),
            }


# ============================================================
# Module-level singleton
# ============================================================

_bandit: Optional[CategorySelectionBandit] = None
_bandit_lock = threading.Lock()


def get_category_bandit() -> CategorySelectionBandit:
    """Get or create the singleton CategorySelectionBandit."""
    global _bandit
    if _bandit is None:
        with _bandit_lock:
            if _bandit is None:
                _bandit = CategorySelectionBandit()
    return _bandit
