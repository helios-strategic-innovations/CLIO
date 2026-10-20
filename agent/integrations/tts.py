"""
OpenClaw/Moltbot — Text-to-Speech (ElevenLabs API)

Converts text responses to audio for Telegram voice replies and phone calls.
"""

import logging
import re

import httpx

from agent.config import config

logger = logging.getLogger("openclaw.tts")

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
TTS_MODEL = "eleven_multilingual_v2"
TTS_TIMEOUT = 30.0
MAX_TEXT_CHARS = 4500


async def synthesize_speech(text: str, output_format: str = "mp3_22050_32") -> bytes:
    """Convert text to audio bytes via ElevenLabs TTS API.

    Args:
        text: Text to synthesize (truncated at 4500 chars).
        output_format: ElevenLabs output format. Default mp3_22050_32 for Telegram.
                       Use ulaw_8000 for Twilio phone calls.

    Returns:
        Raw audio bytes in the requested format.

    Raises:
        ValueError: If API key or voice ID not configured, or empty text.
        BudgetExhaustedError: If daily TTS budget exhausted.
        httpx.HTTPStatusError: On API errors (4xx, 5xx).
        httpx.TimeoutException: On network timeout.
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text")

    if not config.elevenlabs_api_key:
        raise ValueError("ELEVENLABS_API_KEY not configured")

    if not config.elevenlabs_voice_id:
        raise ValueError("ELEVENLABS_VOICE_ID not configured")

    # Budget check — raises BudgetExhaustedError if over limit
    from agent.rate_limiter import check_and_record
    await check_and_record("elevenlabs_tts")

    # Truncate long text (ElevenLabs limit ~5000 chars; 4500 safety margin)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "..."
        logger.warning(f"TTS text truncated to {MAX_TEXT_CHARS} chars")

    url = f"{ELEVENLABS_BASE_URL}/{config.elevenlabs_voice_id}"

    async with httpx.AsyncClient(timeout=TTS_TIMEOUT, follow_redirects=True) as client:
        response = await client.post(
            url,
            params={"output_format": output_format},
            headers={
                "xi-api-key": config.elevenlabs_api_key,
                "Content-Type": "application/json",
                # S190 L5: identify ourselves to ElevenLabs. Anonymous httpx UA
                # invites rate-limit / WAF heuristics down the road.
                "User-Agent": "CLIO-Buddy/1.0 (+https://atlas.strategies)",
            },
            json={
                "text": text,
                "model_id": TTS_MODEL,
            },
        )
        response.raise_for_status()
        audio_bytes = response.content
        logger.info(
            f"TTS synthesized {len(text)} chars -> {len(audio_bytes)} bytes "
            f"(format={output_format})"
        )
        return audio_bytes


def strip_markdown(text: str) -> str:
    """Remove markdown formatting for clean TTS output."""
    text = re.sub(r'```[\s\S]*?```', '', text)        # Code blocks
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)      # Bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)           # Italic
    text = re.sub(r'#{1,6}\s', '', text)               # Headers
    text = re.sub(r'^\|(?:[^|\n]*\|){2,}$', '', text, flags=re.MULTILINE)  # S156 M-D: Tables (2+ cells, per-line)
    text = re.sub(r'`(.+?)`', r'\1', text)             # Inline code
    text = re.sub(r'\n{2,}', '. ', text)               # Paragraph breaks → pauses
    text = re.sub(r'[-•]\s', '', text)                 # Bullet points
    return text.strip()
