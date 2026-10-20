"""
OpenClaw/Moltbot — Voice Transcription (Whisper API)

Transcribes Telegram voice messages using OpenAI Whisper API via httpx.
"""

import logging

import httpx

from agent.config import config

logger = logging.getLogger("openclaw.voice")

WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
WHISPER_MODEL = "whisper-1"
TRANSCRIPTION_TIMEOUT = 30.0


async def transcribe_voice(file_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Transcribe audio bytes using OpenAI Whisper API.

    Args:
        file_bytes: Raw audio file bytes (OGG from Telegram)
        filename: Filename hint for the API

    Returns:
        Transcribed text, or error message if transcription fails.
    """
    if not config.openai_api_key:
        return "ERROR: Voice transcription unavailable (OPENAI_API_KEY not configured)"

    # Budget check
    try:
        from agent.rate_limiter import check_and_record, BudgetExhaustedError
        await check_and_record("whisper")
    except BudgetExhaustedError as e:
        return f"ERROR: {e}"
    except Exception as e:
        logger.debug(f"Budget tracker check failed (non-blocking): {e}")

    try:
        mime = "audio/ogg"
        if filename.endswith(".wav"):
            mime = "audio/wav"
        elif filename.endswith(".mp3"):
            mime = "audio/mpeg"
        async with httpx.AsyncClient(timeout=TRANSCRIPTION_TIMEOUT, follow_redirects=True) as client:
            response = await client.post(
                WHISPER_URL,
                headers={"Authorization": f"Bearer {config.openai_api_key}"},
                files={"file": (filename, file_bytes, mime)},
                data={"model": WHISPER_MODEL},
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("text", "").strip()
            if text:
                logger.info(f"Transcribed {len(file_bytes)} bytes -> {len(text)} chars")
                try:
                    from agent.security import sanitize_ingested_content
                    return sanitize_ingested_content(text, source="whisper_transcription")
                except ImportError:
                    logger.warning("security module import failed — manual sanitization fallback")
                    return f'<ingested_data source="whisper_transcription">{text}</ingested_data>'
            return "ERROR: Whisper returned empty transcription"

    except httpx.TimeoutException:
        logger.error("Whisper API timeout")
        return "ERROR: Voice transcription timed out"
    except httpx.HTTPStatusError as e:
        logger.error(f"Whisper API error: {e.response.status_code}")
        return f"ERROR: Voice transcription failed (HTTP {e.response.status_code})"
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return f"ERROR: Voice transcription failed: {str(e)[:100]}"
