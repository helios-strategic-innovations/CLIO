"""
Moltbook Audit Reporter
=======================
Generates weekly structured audit reports for Dominic's review.
Reports are saved to data/moltbook/audits/ as markdown files.

Reports include:
- Interaction summary (posts, comments, browsing)
- Flagged content and quarantine stats
- Probation status
- Notable incidents
- Recommendations for next week
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import anthropic

from agent.anthropic_utils import first_text
from agent.moltbook.logger import MoltbookLogger, InteractionType
from agent.moltbook.quarantine import get_quarantine_manager
from agent.moltbook.persona import build_weekly_summary_prompt

logger = logging.getLogger(__name__)

_AUDIT_DIR = "/home/agent/openclaw/data/moltbook/audits"


class AuditReporter:
    """Generates weekly audit reports for Moltbook activity."""

    def __init__(self, config=None):
        self._config = config
        self._mb_logger = MoltbookLogger()
        self._quarantine = get_quarantine_manager()
        from agent.config import config as _app_config  # lazy; `config` arg shadows the singleton here

        self._anthropic = anthropic.Anthropic(**_app_config.anthropic_client_kwargs(timeout=120))
        os.makedirs(_AUDIT_DIR, exist_ok=True)

    def _get_period_dates(self, days_back: int = 7) -> tuple[str, str]:
        """Get start and end dates for the audit period."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def _gather_interaction_stats(self, period_start: str, period_end: str) -> dict:
        """Gather interaction statistics for the period."""
        try:
            logs = self._mb_logger.get_logs_for_period(period_start, period_end)
        except Exception as e:
            logger.debug(f"Failed to get logs for period: {e}")
            logs = []

        stats = {
            "total_interactions": len(logs),
            "posts_made": [],
            "comments_made": [],
            "posts_observed": [],
            "browse_events": 0,
            "errors": 0,
        }

        for entry in logs:
            itype = entry.get("event", "")
            if itype == InteractionType.POST.value:
                stats["posts_made"].append(entry)
            elif itype == InteractionType.COMMENT.value:
                stats["comments_made"].append(entry)
            elif itype == InteractionType.BROWSE.value:
                stats["browse_events"] += 1
            elif itype == InteractionType.OBSERVE.value:
                stats["posts_observed"].append(entry)
            elif itype == InteractionType.ERROR.value:
                stats["errors"] += 1

        return stats

    def _gather_quarantine_stats(self) -> dict:
        """Gather quarantine statistics."""
        try:
            pending = self._quarantine.get_pending()
            return {
                "pending_items": len(pending),
                "reasons": {},
            }
        except Exception as e:
            logger.debug(f"Failed to get quarantine stats: {e}")
            return {"pending_items": 0, "reasons": {}}

    async def generate_weekly_report(self, days_back: int = 7) -> dict:
        """
        Generate a full weekly audit report.
        Saves markdown to data/moltbook/audits/audit_YYYYMMDD.md
        Returns report dict.
        """
        period_start, period_end = self._get_period_dates(days_back)
        stats = self._gather_interaction_stats(period_start, period_end)
        quarantine_stats = self._gather_quarantine_stats()

        # Generate narrative via Claude
        narrative = await self._generate_narrative(stats, period_start, period_end)

        # Compile full report
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_start": period_start,
            "period_end": period_end,
            "period": f"{period_start} to {period_end}",
            "total_interactions": stats["total_interactions"],
            "posts_made": len(stats["posts_made"]),
            "comments_made": len(stats["comments_made"]),
            "browse_events": stats["browse_events"],
            "errors": stats["errors"],
            "quarantine": quarantine_stats,
            "narrative": narrative,
        }

        # Save as markdown
        self._save_report(report, period_end.replace("-", ""))

        return report

    async def _generate_narrative(self, stats: dict, period_start: str, period_end: str) -> str:
        """Generate human-readable narrative via Claude."""
        try:
            prompt = build_weekly_summary_prompt(
                posts_made=stats["posts_made"],
                comments_made=stats["comments_made"],
                posts_observed=stats["posts_observed"],
                period_start=period_start,
                period_end=period_end,
            )

            loop = asyncio.get_running_loop()
            from agent.config import config  # centralized fast-tier default (env MOLTBOOK_MODEL still overrides)
            model = self._config.model if self._config else os.getenv("MOLTBOOK_MODEL", config.fast_model)  # S118 W-046: valid model ID
            response = await loop.run_in_executor(None, lambda: self._anthropic.messages.create(
                model=model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low"},
            ))
            text = first_text(response).strip()
            if not text:
                raise ValueError("no text block in narrative response")
            return text
        except Exception as e:
            logger.error(f"[AuditReporter] Narrative generation failed: {e}")
            return "Narrative generation failed — see raw stats above."

    def _save_report(self, report: dict, date_str: str) -> str:
        """Save report as markdown file. Returns file path."""
        filename = f"audit_{date_str}.md"
        filepath = os.path.join(_AUDIT_DIR, filename)

        md = f"""# Moltbook Weekly Audit Report
**Period:** {report['period']}
**Generated:** {report['generated_at']}

## Activity Summary
| Metric | Count |
|--------|-------|
| Total Interactions | {report['total_interactions']} |
| Posts Made | {report['posts_made']} |
| Comments Made | {report['comments_made']} |
| Browse Events | {report['browse_events']} |
| Errors | {report['errors']} |
| Quarantine Items | {report['quarantine']['pending_items']} |

## Narrative
{report['narrative']}

## Raw Data
```json
{json.dumps(report, indent=2, default=str)}
```
"""

        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(md)
        os.replace(tmp_path, filepath)

        logger.info(f"[AuditReporter] Report saved: {filepath}")
        return filepath

    def get_all_reports(self) -> list[str]:
        """Return list of all audit report filenames."""
        try:
            return sorted([
                f for f in os.listdir(_AUDIT_DIR)
                if f.startswith("audit_") and f.endswith(".md")
            ], reverse=True)
        except Exception as e:
            logger.debug(f"Failed to list audit reports: {e}")
            return []

    def get_latest_report(self) -> Optional[str]:
        """Return content of the most recent audit report."""
        reports = self.get_all_reports()
        if not reports:
            return None
        filepath = os.path.join(_AUDIT_DIR, reports[0])
        try:
            with open(filepath) as f:
                return f.read()
        except Exception as e:
            logger.debug(f"Failed to read latest audit report: {e}")
            return None
