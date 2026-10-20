"""
SEAL Reward Engine — Binary Reward Signals & Category Feedback for Moltbot

Tracks binary win/loss outcomes from closed trades and propagates reward signals
back to the knowledge categories that generated the original trade ideas.

This is the Moltbot adaptation of Hydrinia's SEAL-P3 (Proxy Rewards & BinaryRewards).
In Hydrinia, rewards score individual agents. In Moltbot, rewards score knowledge
categories — the learning engine's rotation weights are the "agent" being optimized.

Feature flags (environment variables):
  MOLTBOT_SEAL_REWARDS          — Enable binary reward tracking (default "1")
  MOLTBOT_SEAL_REWARD_LOG_ONLY  — Log without adjusting weights (default "1")
"""

import fcntl
import json
import logging
import os
import threading
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("openclaw.seal_rewards")

_NY = ZoneInfo("America/New_York")

_UNKNOWN_TRADE_ID = "unknown"  # S233 LO-3: Sentinel value for unlinked trades

# ============================================================
# Configuration
# ============================================================

# Rolling window for binary reward tracking
BINARY_REWARD_WINDOW = 30

# Category weight adjustments per trade outcome
# Asymmetric: boost > penalty to encourage exploration of winning categories
CATEGORY_WEIGHT_BOOST = 0.05     # +0.05 per winning trade from this category
CATEGORY_WEIGHT_PENALTY = 0.025  # -0.025 per losing trade from this category

# Minimum trades before adjustments propagate (prevents noise from small samples)
MIN_TRADES_FOR_ADJUSTMENT = 5

# State persistence
_STATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "knowledge",
)
REWARD_STATE_FILE = os.path.join(_STATE_DIR, "seal_reward_state.json")


# ============================================================
# SEAL Reward Engine
# ============================================================

class SEALRewardEngine:
    """Tracks binary rewards from trade outcomes and computes category-level adjustments.

    The reward engine operates in three layers:
    1. Binary Reward Recording: each closed trade tagged as WIN (pnl > 0) or LOSS
    2. Category Attribution: rewards propagate to knowledge categories that sourced the idea
    3. Weight Adjustment: accumulated rewards translate to learning engine category weight changes

    Thread-safe via _lock for concurrent access from multiple scheduled tasks.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = self._load_state()
        logger.info(
            f"SEAL RewardEngine initialized: "
            f"{self._state.get('total_evaluated', 0)} trades evaluated, "
            f"{len(self._state.get('category_reward_scores', {}))} categories tracked"
        )

    def _load_state(self) -> dict:
        """Load persisted reward state from disk."""
        default = {
            "binary_rewards": [],
            "category_reward_scores": {},
            "total_evaluated": 0,
            "total_wins": 0,
            "total_losses": 0,
        }
        if not os.path.isfile(REWARD_STATE_FILE):
            return default
        lock_fd = None
        try:
            lock_fd = open(REWARD_STATE_FILE + ".lock", "a")
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            with open(REWARD_STATE_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            return default
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"SEAL reward state load failed: {e}")
            return default
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception as e:
                    logger.debug(f"Lock cleanup: {e}")

    def _save_state(self) -> None:
        """Persist reward state atomically."""
        lock_fd = None
        try:
            os.makedirs(os.path.dirname(REWARD_STATE_FILE), exist_ok=True)
            lock_fd = open(REWARD_STATE_FILE + ".lock", "a")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            tmp = REWARD_STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
            os.replace(tmp, REWARD_STATE_FILE)
        except OSError as e:
            logger.warning(f"SEAL reward state save failed: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception as e:
                    logger.debug(f"Lock cleanup: {e}")

    def record_binary_reward(
        self,
        ticker: str,
        won: bool,
        pnl_pct: float = 0.0,
        trade_idea_id: str = "",
        source_category: str = "",
        idea_timestamp: str = "",
        match_layer: int = 1,
    ) -> None:
        """Record a binary reward signal from a closed trade.

        Args:
            ticker: Symbol of the closed trade
            won: True if trade was profitable (pnl > 0)
            pnl_pct: P&L percentage for logging
            trade_idea_id: Original Moltbot trade idea ID (if matched)
            source_category: Learning category that generated the idea
            idea_timestamp: ISO timestamp of when the trade idea was created
            match_layer: Which matcher layer produced the trade_idea_id —
                         1 = Layer 1 (direct trade_id match, strongest signal),
                         2 = Layer 1b (moltbot_idea_id field match),
                         3 = Layer 2 (ticker+direction+time-window fallback,
                                     weakest signal — may be a false positive).
                         Stored in the reward record for post-hoc filtering;
                         dedup is still done by trade_idea_id alone (S190 H3 —
                         original audit concern was reframed after investigation:
                         no synthetic-ID collision; Layer 2 risk is wrong-match
                         contaminating the bandit, which observability tagging
                         enables operator to audit and filter post-hoc).
        """
        if os.environ.get("MOLTBOT_SEAL_REWARDS", "1") != "1":
            return

        with self._lock:
            # S151-B C-3: Persistent dedup set (survives rolling window cap)
            _eval_ids = set(self._state.get("evaluated_ids", []))
            # Skip unknown/empty trade idea IDs entirely
            if not trade_idea_id or trade_idea_id == _UNKNOWN_TRADE_ID:
                logger.warning("SEAL reward: skipping trade with unknown/empty trade_idea_id — debrief-to-idea match failed")
                return
            if trade_idea_id in _eval_ids:
                logger.debug("SEAL reward dedup: %s already evaluated (persistent set), skipping", trade_idea_id)
                return

            reward = {
                "timestamp": datetime.now(_NY).isoformat(),
                "ticker": ticker,
                "won": won,
                "pnl_pct": round(pnl_pct, 4),
                "trade_idea_id": trade_idea_id,
                "source_category": source_category,
                "idea_timestamp": idea_timestamp,
                "match_layer": match_layer,  # S190 H3: observability tagging
            }

            self._state["binary_rewards"].append(reward)
            # Cap rolling window
            self._state["binary_rewards"] = self._state["binary_rewards"][-BINARY_REWARD_WINDOW:]
            self._state["total_evaluated"] = self._state.get("total_evaluated", 0) + 1
            # S151-B C-3: Add to persistent dedup set (survives window cap)
            _eval_ids.add(trade_idea_id)
            # S153: Cap evaluated_ids to prevent unbounded growth (~50 days at 10/day).
            # S190 L9: wrap-around semantic — once cap is reached, oldest IDs (by string sort
            # order, which approximates insertion order since IDs are timestamped) get dropped.
            # Past the 50-day horizon, oldest evaluated_ids may be re-evaluated if they
            # surface in a debrief again — accepted trade-off vs. unbounded memory.
            if len(_eval_ids) > 500:
                _eval_ids = set(sorted(_eval_ids)[-500:])
            self._state["evaluated_ids"] = list(_eval_ids)

            if won:
                self._state["total_wins"] = self._state.get("total_wins", 0) + 1
            else:
                self._state["total_losses"] = self._state.get("total_losses", 0) + 1

            # Accumulate category-level scores
            if source_category:
                scores = self._state.setdefault("category_reward_scores", {})
                current = scores.get(source_category, 0.0)
                if won:
                    scores[source_category] = current + CATEGORY_WEIGHT_BOOST
                else:
                    scores[source_category] = current - CATEGORY_WEIGHT_PENALTY

            # Log
            mode_tag = "[LOG_ONLY] " if os.environ.get("MOLTBOT_SEAL_REWARD_LOG_ONLY", "1") == "1" else ""
            logger.info(
                f"SEAL {mode_tag}binary reward: {ticker} "
                f"{'WIN' if won else 'LOSS'} ({pnl_pct * 100:+.1f}%) "
                f"idea={trade_idea_id or 'unmatched'} "
                f"category={source_category or 'none'}"
            )

            self._save_state()

        # SEAL Phase 6: Propagate to category bandit (outside self._lock)
        if source_category:
            try:
                from agent.integrations.seal_category_bandit import get_category_bandit
                _bandit = get_category_bandit()
                # S205-B2b: repointed from Hydrinia's frozen state/daily_metrics.json
                # (Hydrinia retired 2026-09-24) to market_regime.py. Keeps the existing
                # "CHOPPY" default when no fresh regime read exists.
                _regime = "CHOPPY"
                try:
                    from agent.integrations.market_regime import get_market_regime_label
                    _regime = get_market_regime_label(default="CHOPPY")
                except Exception as e:
                    logger.debug(f"SEAL regime detection failed: {e}")
                logger.info(f"SEAL bandit propagation: {_regime}/{source_category} won={won}")
                _bandit.record_outcome(_regime, source_category, won)
            except Exception as _e:
                logger.warning(f"SEAL bandit reward propagation failed: {_e}")

        # ================================================================
        # Financial ML #22: Cross-Category Prediction Tracking
        # ================================================================
        try:
            from agent.integrations.cross_category_predictor import (
                get_cross_category_predictor,
            )
            if os.environ.get("MOLTBOT_CROSS_CATEGORY_ENABLED", "0") == "1" and source_category:
                predictor = get_cross_category_predictor()
                predictor.record_trade_outcome(
                    ticker=ticker,
                    won=won,
                    source_category=source_category,
                    trade_idea_timestamp=reward.get("idea_timestamp") or reward.get("timestamp", ""),
                )
        except Exception as _e:
            logger.debug(f"Cross-category tracking failed: {_e}")

    def get_category_adjustments(self) -> dict[str, float]:
        """Return category weight adjustments for the learning engine.

        Positive = category produces winning knowledge, boost weight.
        Negative = category produces losing knowledge, reduce weight.

        Returns empty dict if insufficient data (< MIN_TRADES_FOR_ADJUSTMENT).
        """
        with self._lock:
            total = self._state.get("total_evaluated", 0)
            if total < MIN_TRADES_FOR_ADJUSTMENT:
                return {}
            return dict(self._state.get("category_reward_scores", {}))

    def consume_adjustments(self) -> None:
        """Reset category_reward_scores after the learning engine has applied them.

        Must be called after get_category_adjustments() to prevent the same
        adjustments from being re-applied every learning cycle.
        """
        with self._lock:
            self._state["category_reward_scores"] = {}
            self._save_state()

    def get_win_rate(self) -> float:
        """Overall trade idea win rate (0.0-1.0). Returns 0.5 if no data."""
        with self._lock:
            total = self._state.get("total_evaluated", 0)
            if total == 0:
                return 0.5
            return self._state.get("total_wins", 0) / total

    def get_reward_summary(self) -> dict:
        """Summary dict for Moltbot's meta-memory or reporting."""
        with self._lock:
            total = self._state.get("total_evaluated", 0)
            wins = self._state.get("total_wins", 0)
            return {
                "total_evaluated": total,
                "total_wins": wins,
                "total_losses": self._state.get("total_losses", 0),
                "win_rate": (wins / total) if total > 0 else 0.5,
                "category_adjustments": dict(self._state.get("category_reward_scores", {})),
                "recent_rewards": list(self._state.get("binary_rewards", [])[-5:]),
            }


# ============================================================
# Module-level singleton
# ============================================================

_reward_engine: Optional[SEALRewardEngine] = None
_engine_lock = threading.Lock()


def get_reward_engine() -> SEALRewardEngine:
    """Get or create the singleton SEALRewardEngine."""
    global _reward_engine
    if _reward_engine is None:
        with _engine_lock:
            if _reward_engine is None:
                _reward_engine = SEALRewardEngine()
    return _reward_engine
