import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config import Config
import agent.rate_limiter as rl


class UWBudgetRetuneTests(unittest.TestCase):
    def test_config_default_uses_startup_guard(self):
        self.assertEqual(Config().budget_unusual_whales, 16000)

    def test_env_default_uses_startup_guard_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Config.from_env().budget_unusual_whales, 16000)

    def test_budget_warns_once_at_80_percent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_quota_file = rl.QUOTA_FILE
            rl.QUOTA_FILE = Path(tmpdir) / "rate_limits.json"
            try:
                tracker = rl.BudgetTracker()
                tracker.register("unusual_whales", max_requests=10, period="daily")
                with self.assertLogs("openclaw.rate_limiter", level="WARNING") as captured:
                    asyncio.run(tracker.record_usage("unusual_whales", 8))
                    asyncio.run(tracker.record_usage("unusual_whales", 1))
                threshold_lines = [
                    line for line in captured.output if "Budget 80% threshold for unusual_whales" in line
                ]
                self.assertEqual(len(threshold_lines), 1)
            finally:
                rl.QUOTA_FILE = old_quota_file

    def test_budget_exhaustion_still_blocks_at_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_quota_file = rl.QUOTA_FILE
            rl.QUOTA_FILE = Path(tmpdir) / "rate_limits.json"
            try:
                tracker = rl.BudgetTracker()
                tracker.register("unusual_whales", max_requests=2, period="daily")
                asyncio.run(tracker.record_usage("unusual_whales", 2))
                allowed, msg = asyncio.run(tracker.check_budget("unusual_whales"))
                self.assertFalse(allowed)
                self.assertIn("Budget exhausted for unusual_whales", msg)
            finally:
                rl.QUOTA_FILE = old_quota_file


if __name__ == "__main__":
    unittest.main()
