"""
Moltbook Telegram Commands — Registration and Handlers
Provides /moltbook_status, /moltbook_kill, /moltbook_quarantine,
/moltbook_approve, /moltbook_discard, /moltbook_audit, /moltbook_mode
"""

import logging
from typing import Optional

from agent.integrations.telegram_bot import is_authorized
from agent.moltbook.config import MoltbookMode

logger = logging.getLogger("moltbot.moltbook.telegram")


class MoltbookTelegramCommands:
    """
    Registers all Moltbook-related Telegram commands.
    Called once from telegram_bot.py with register_commands(app).
    """

    def __init__(self, heartbeat=None, quarantine_manager=None, kill_switch=None, audit_reporter=None, config=None):
        self.heartbeat = heartbeat
        self.quarantine = quarantine_manager
        self.kill_switch = kill_switch
        self.audit = audit_reporter
        self.config = config

    async def handle_status(self, update, context) -> None:
        """Show Moltbook subsystem health."""
        if not is_authorized(update.effective_user.id):
            return
        if not self.config:
            await update.message.reply_text("🔴 Moltbook not initialized.")
            return

        enabled = self.config.enabled
        mode = self.config.mode
        dry_run = self.config.dry_run
        kill_active = self.kill_switch.is_active if self.kill_switch else False

        status_lines = [
            "🦀 Moltbook Status",
            f"  Enabled: {'🟢 ON' if enabled else '🔴 OFF'}",
            f"  Mode: {mode.value if isinstance(mode, MoltbookMode) else mode}",
            f"  Dry Run: {'⚠️ YES' if dry_run else '🔴 NO (LIVE)'}",
            f"  Kill Switch: {'🔴 ACTIVE' if kill_active else '🟢 Clear'}",
        ]

        if self.heartbeat:
            stats = self.heartbeat.get_stats()
            status_lines.extend([
                f"  Cycles: {stats.get('total_cycles', 0)}",
                f"  Posts: {stats.get('posts_sent', 0)}",
                f"  Comments: {stats.get('comments_sent', 0)}",
                f"  Last cycle: {stats.get('last_cycle', 'never')}",
            ])

        await update.message.reply_text("\n".join(status_lines))

    async def handle_kill(self, update, context) -> None:
        """Emergency kill switch activation."""
        if not is_authorized(update.effective_user.id):
            return
        if self.kill_switch:
            reason = " ".join(context.args) if context.args else "Manual kill via Telegram"
            self.kill_switch.activate(reason=reason, user_id="telegram_command")
            await update.message.reply_text("🔴 Moltbook KILLED. All activity halted.\nReason: " + reason)
            logger.critical(f"Moltbook kill switch activated via Telegram: {reason}")
        else:
            await update.message.reply_text("⚠️ Kill switch not initialized.")

    async def handle_quarantine(self, update, context) -> None:
        """Review pending quarantined insights."""
        if not is_authorized(update.effective_user.id):
            return
        if not self.quarantine:
            await update.message.reply_text("⚠️ Quarantine manager not initialized.")
            return

        pending = self.quarantine.get_pending()
        if not pending:
            await update.message.reply_text("✅ No pending quarantined items.")
            return

        lines = ["🔒 Quarantined Items:\n"]
        for item in pending[:10]:  # Show max 10
            lines.append(
                f"  {item.id[:8]} — {item.category}: "
                f"{item.insight_extracted[:80]}..."
            )
        lines.append(f"\n📊 Total pending: {len(pending)}")
        lines.append("Use /moltbook_approve <id> or /moltbook_discard <id>")

        await update.message.reply_text("\n".join(lines))

    async def handle_approve(self, update, context) -> None:
        """Approve a quarantined insight for publication."""
        if not is_authorized(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text("Usage: /moltbook_approve <id>")
            return

        item_id = context.args[0]
        if self.quarantine:
            result = self.quarantine.approve(item_id, integration_note="Approved via Telegram")
            if result:
                await update.message.reply_text(f"✅ Approved: {item_id[:8]}")
            else:
                await update.message.reply_text(f"❌ Item {item_id[:8]} not found or already processed.")

    async def handle_discard(self, update, context) -> None:
        """Discard a quarantined insight."""
        if not is_authorized(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text("Usage: /moltbook_discard <id>")
            return

        item_id = context.args[0]
        if self.quarantine:
            result = self.quarantine.discard(item_id)
            if result:
                await update.message.reply_text(f"🗑️ Discarded: {item_id[:8]}")
            else:
                await update.message.reply_text(f"❌ Item {item_id[:8]} not found.")

    async def handle_audit(self, update, context) -> None:
        """Generate and send weekly audit report."""
        if not is_authorized(update.effective_user.id):
            return
        if self.audit:
            report = await self.audit.generate_weekly_report()
            text = (
                f"📋 Moltbook Audit Report\n"
                f"Period: {report.get('period', 'N/A')}\n"
                f"Interactions: {report.get('total_interactions', 0)}\n"
                f"Posts: {report.get('posts_made', 0)} | Comments: {report.get('comments_made', 0)}\n"
                f"Errors: {report.get('errors', 0)}\n\n"
                f"{report.get('narrative', 'No narrative generated.')}"
            )
            if len(text) > 4000:
                text = text[:3997] + "..."
            await update.message.reply_text(text)
        else:
            await update.message.reply_text("⚠️ Audit reporter not initialized.")

    async def handle_mode(self, update, context) -> None:
        """Change Moltbook operating mode."""
        if not is_authorized(update.effective_user.id):
            return
        valid_modes = ["read_only", "limited", "standard"]
        if not context.args or context.args[0] not in valid_modes:
            current = self.config.mode.value if self.config and isinstance(self.config.mode, MoltbookMode) else str(self.config.mode) if self.config else "unknown"
            await update.message.reply_text(
                f"Usage: /moltbook_mode <{'|'.join(valid_modes)}>\nCurrent: {current}",
            )
            return

        new_mode_str = context.args[0]
        if self.config:
            old_mode = self.config.mode
            old_mode_str = old_mode.value if isinstance(old_mode, MoltbookMode) else str(old_mode)
            self.config.mode = MoltbookMode(new_mode_str)
            await update.message.reply_text(
                f"Moltbook mode changed: {old_mode_str} → {new_mode_str}",
            )
            logger.info(f"Moltbook mode changed from {old_mode_str} to {new_mode_str} via Telegram")

    def get_command_map(self) -> dict:
        """
        Returns {command_name: handler} for registration in telegram_bot.py.
        Example: {"moltbook_status": self.handle_status, ...}
        """
        return {
            "moltbook_status": self.handle_status,
            "moltbook_kill": self.handle_kill,
            "moltbook_quarantine": self.handle_quarantine,
            "moltbook_approve": self.handle_approve,
            "moltbook_discard": self.handle_discard,
            "moltbook_audit": self.handle_audit,
            "moltbook_mode": self.handle_mode,
        }
