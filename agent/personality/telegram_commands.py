"""
Personality Telegram Commands — Registration and Handlers
Provides /personality, /trait_propose, /trait_confirm, /trait_deny,
/reading_update, /personality_on, /personality_off
"""

import json
import logging
import os
import re
from datetime import datetime

from agent.integrations.telegram_bot import is_authorized
from agent.personality.config import _OVERRIDE_PATH

logger = logging.getLogger("moltbot.personality.telegram")

# Regex for matching both straight and smart (curly) quotes
_QUOTE_RE = re.compile(r'["\u201c\u201d](.*?)["\u201c\u201d]')


def _persist_personality_toggle(enabled: bool) -> None:
    """Write personality enabled state to disk for persistence across restarts."""
    try:
        os.makedirs(os.path.dirname(_OVERRIDE_PATH), exist_ok=True)
        tmp_path = _OVERRIDE_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"enabled": enabled}, f)
        os.replace(tmp_path, _OVERRIDE_PATH)
        logger.info(f"Personality override persisted: enabled={enabled}")
    except Exception as e:
        logger.error(f"Failed to persist personality override: {e}")


class PersonalityTelegramCommands:
    """
    Registers all Personality-related Telegram commands.
    Called once from telegram_bot.py with register_commands(app).
    """

    def __init__(self, registry=None, engine=None, config=None):
        self.registry = registry
        self.engine = engine
        self.config = config

    async def handle_personality(self, update, context) -> None:
        """Show current personality state — confirmed traits and status."""
        if not is_authorized(update.effective_user.id):
            return
        if not self.registry:
            await update.message.reply_text("Personality registry not initialized.")
            return

        enabled = self.config.enabled if self.config else False
        traits = self.registry.get_confirmed_traits()

        lines = [
            f"Personality Engine — {'ON' if enabled else 'OFF'}",
            f"Confirmed traits: {len(traits)}",
            ""
        ]

        if traits:
            for t in traits:
                activation = t.activation_count
                conviction = t.conviction
                lines.append(
                    f"  - {t.name} (conviction: {conviction:.2f}, activations: {activation})"
                )
                if t.description:
                    lines.append(f"    {t.description[:100]}")
        else:
            lines.append("  No traits confirmed yet.")

        pending = self.registry.get_proposed_traits()
        if pending:
            lines.append(f"\nPending proposals: {len(pending)}")
            for p in pending:
                lines.append(f"  - {p.id} — {p.name}")

        await update.message.reply_text("\n".join(lines))

    async def handle_trait_propose(self, update, context) -> None:
        """Propose a new personality trait for Dominic's confirmation."""
        if not is_authorized(update.effective_user.id):
            return
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                'Usage: /trait_propose "Trait Name" Description of the trait...'
            )
            return

        # Parse: first quoted arg = name, rest = description
        full_text = " ".join(context.args)
        match = _QUOTE_RE.search(full_text)
        if match:
            name = match.group(1)
            description = full_text[match.end():].strip()
        else:
            name = context.args[0]
            description = " ".join(context.args[1:])

        if self.engine:
            proposal = self.engine.propose_trait(
                name=name,
                description=description,
                rationale="Proposed via Telegram"
            )
            await update.message.reply_text(
                f"Trait proposed: {name}\n"
                f"ID: {proposal.id}\n"
                f"Description: {description[:200]}\n\n"
                f"Confirm: /trait_confirm {proposal.id}\n"
                f"Deny: /trait_deny {proposal.id}"
            )
            logger.info(f"Trait proposed: {name} ({proposal.id})")
        elif self.registry:
            proposal = self.registry.propose(
                name=name,
                description=description,
                rationale="Proposed via Telegram"
            )
            await update.message.reply_text(
                f"Trait proposed: {name}\n"
                f"ID: {proposal.id}\n"
                f"Description: {description[:200]}\n\n"
                f"Confirm: /trait_confirm {proposal.id}\n"
                f"Deny: /trait_deny {proposal.id}"
            )
            logger.info(f"Trait proposed: {name} ({proposal.id})")

    async def handle_trait_confirm(self, update, context) -> None:
        """Confirm a proposed trait — Dominic only."""
        if not is_authorized(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text("Usage: /trait_confirm <id>")
            return

        trait_id = context.args[0]
        if self.engine:
            result = self.engine.confirm_trait(trait_id)
            if result:
                await update.message.reply_text(
                    f"Trait confirmed: {result.name}\n"
                    f"Now part of my personality. Thank you, Dominic."
                )
                logger.info(f"Trait confirmed: {result.name}")
            else:
                await update.message.reply_text(f"Trait {trait_id} not found or already processed.")
        elif self.registry:
            result = self.registry.confirm(trait_id)
            if result:
                await update.message.reply_text(
                    f"Trait confirmed: {result.name}\n"
                    f"Now part of my personality. Thank you, Dominic."
                )
                logger.info(f"Trait confirmed: {result.name}")
            else:
                await update.message.reply_text(f"Trait {trait_id} not found or already processed.")

    async def handle_trait_deny(self, update, context) -> None:
        """Deny a proposed trait — no reason required."""
        if not is_authorized(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text("Usage: /trait_deny <id>")
            return

        trait_id = context.args[0]
        if self.engine:
            result = self.engine.deny_trait(trait_id)
            if result:
                await update.message.reply_text(
                    f"Understood. Trait {trait_id} denied. No explanation needed."
                )
                logger.info(f"Trait denied: {trait_id}")
            else:
                await update.message.reply_text(f"Trait {trait_id} not found or already processed.")
        elif self.registry:
            result = self.registry.deny(trait_id)
            if result:
                await update.message.reply_text(
                    f"Understood. Trait {trait_id} denied. No explanation needed."
                )
                logger.info(f"Trait denied: {trait_id}")
            else:
                await update.message.reply_text(f"Trait {trait_id} not found.")

    async def handle_reading_update(self, update, context) -> None:
        """Log reading progress on an assigned text."""
        if not is_authorized(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text(
                'Usage: /reading_update "Title" Notes about what was read...'
            )
            return

        full_text = " ".join(context.args)
        match = _QUOTE_RE.search(full_text)
        if match:
            title = match.group(1)
            notes = full_text[match.end():].strip()
        else:
            title = context.args[0]
            notes = " ".join(context.args[1:])

        if self.engine:
            self.engine.log_reading(title=title, notes=notes)
            await update.message.reply_text(
                f"Reading logged: {title}\n{notes[:200]}"
            )

    async def handle_personality_on(self, update, context) -> None:
        """Enable personality trait injection."""
        if not is_authorized(update.effective_user.id):
            return
        if self.config:
            _persist_personality_toggle(True)
            await update.message.reply_text("Personality engine enabled. Traits now active in system prompt.")
            logger.info("Personality engine enabled via Telegram")

    async def handle_personality_off(self, update, context) -> None:
        """Disable personality trait injection (registry preserved)."""
        if not is_authorized(update.effective_user.id):
            return
        if self.config:
            _persist_personality_toggle(False)
            await update.message.reply_text(
                "Personality engine disabled. Traits preserved but not injected."
            )
            logger.info("Personality engine disabled via Telegram")

    def get_command_map(self) -> dict:
        """Returns {command_name: handler} for registration."""
        return {
            "personality": self.handle_personality,
            "trait_propose": self.handle_trait_propose,
            "trait_confirm": self.handle_trait_confirm,
            "trait_deny": self.handle_trait_deny,
            "reading_update": self.handle_reading_update,
            "personality_on": self.handle_personality_on,
            "personality_off": self.handle_personality_off,
        }
