"""
Moltbot Voice Server — Real-time phone call handler via Twilio Media Streams.

Runs as a separate process alongside main.py.
Twilio calls POST /incoming-call, which returns TwiML to open a WebSocket media stream.
Audio flows: Twilio mulaw → Whisper STT → Claude → ElevenLabs TTS → Twilio mulaw.
"""

import asyncio
import base64
import json
import logging
import os
import re
import struct
from collections import deque

from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import Response

logger = logging.getLogger("openclaw.voice_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="Moltbot Voice Server")

_MAX_BUFFER_BYTES = 2_000_000  # S159 H-12: Audio buffer cap

_call_timestamps: list[float] = []  # S159 L-11: Call rate limiting
_MAX_CALLS_PER_HOUR = 30


# --- Twilio Signature Verification (S118 — defense-in-depth) ---

def _get_twilio_validator():
    """Return a Twilio RequestValidator if auth token is configured, else None."""
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not auth_token:
        logger.warning("TWILIO_AUTH_TOKEN not set — signature verification disabled")
        return None
    try:
        from twilio.request_validator import RequestValidator
        return RequestValidator(auth_token)
    except ImportError:
        logger.warning("twilio package not installed — signature verification disabled")
        return None


_twilio_validator = _get_twilio_validator()


# --- Caller allowlist (S-ROT: primary voice control, FAIL-CLOSED) ---
#
# Twilio signature validation proves a request came from Twilio; it says nothing
# about WHO dialled. The allowlist is the control that authenticates the caller,
# which demotes the spoken PIN to a genuine second factor.
#
# ⚠ FAIL-CLOSED AND LINE-WIDE: an empty or unset VOICE_CALLER_ALLOWLIST rejects
# EVERY inbound call, including the operator's. This is the operator's explicit
# choice (whole-line gating over override-only gating). VOICE_CALLER_ALLOWLIST
# must be set in .env before moltbot-voice next restarts or the phone line goes
# dark. Format: comma-separated E.164, e.g. "+15551234567,+15559876543".

def _normalize_e164(raw: str) -> str:
    """Reduce a phone number to a comparable form: digits with a leading '+'."""
    if not raw:
        return ""
    s = re.sub(r"[^\d+]", "", raw.strip())
    if s and not s.startswith("+"):
        s = "+" + s
    return s


def _load_caller_allowlist() -> set:
    raw = os.getenv("VOICE_CALLER_ALLOWLIST", "")
    return {n for n in (_normalize_e164(p) for p in raw.split(",")) if n}


_caller_allowlist = _load_caller_allowlist()
if not _caller_allowlist:
    logger.error(
        "VOICE_CALLER_ALLOWLIST is unset/empty — ALL inbound calls will be REJECTED "
        "(fail-closed). Set VOICE_CALLER_ALLOWLIST in .env and restart moltbot-voice."
    )
else:
    logger.info("Caller allowlist loaded: %d number(s)", len(_caller_allowlist))


# --- TwiML for incoming calls ---

def _reject_twiml(spoken: str = "This number is not authorized.") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{spoken}</Say>
    <Hangup/>
</Response>"""


def _get_twiml(host: str, caller: str = "") -> str:
    """
    Return TwiML XML that tells Twilio to open a bidirectional media stream.

    The caller number is threaded through as a <Parameter> so the media-stream
    socket (which Twilio opens separately, without the original form fields) can
    key the PIN lockout per-caller instead of per-call. `caller` is already
    normalised to [0-9+] by _normalize_e164, so it carries no XML metacharacters.
    """
    param = f'\n            <Parameter name="caller" value="{caller}" />' if caller else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Connecting to Moltbot.</Say>
    <Connect>
        <Stream url="wss://{host}/media-stream">{param}
        </Stream>
    </Connect>
</Response>"""


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """Twilio webhook — called when someone dials the Twilio number."""
    # S159 L-11: Call rate limiting
    import time as _time
    _now = _time.time()
    _call_timestamps[:] = [t for t in _call_timestamps if _now - t < 3600]
    if len(_call_timestamps) >= _MAX_CALLS_PER_HOUR:
        logger.warning("S159 L-11: Call rate limit exceeded")
        return Response(content='<Response><Say>System busy. Please try again later.</Say></Response>',
                        media_type="application/xml")
    _call_timestamps.append(_now)

    # Read the form once — the caller number is needed after signature validation.
    form_data = dict(await request.form())

    # S118: Verify Twilio signature (defense-in-depth — Caddy already path-whitelists)
    if _twilio_validator:
        # Reconstruct public URL (Caddy reverse-proxies, so use forwarded headers)
        proto = request.headers.get("x-forwarded-proto", "https")
        fwd_host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
        public_url = f"{proto}://{fwd_host}{request.url.path}"
        signature = request.headers.get("X-Twilio-Signature", "")
        if not _twilio_validator.validate(public_url, form_data, signature):
            logger.warning(f"Twilio signature validation failed from {request.client.host}")
            return Response(status_code=403, content="Forbidden")
    else:
        # S153: Fail closed — reject if TWILIO_AUTH_TOKEN not configured
        logger.error("Twilio validator not configured — rejecting unauthenticated call")
        return Response(status_code=503, content="Voice service not configured")

    # S-ROT: Caller allowlist — authenticates WHO dialled (signature only proves
    # the request came from Twilio). FAIL-CLOSED: unset allowlist rejects everyone.
    caller = _normalize_e164(form_data.get("From", ""))
    if not _caller_allowlist:
        logger.error(
            "Inbound call REJECTED from %s — VOICE_CALLER_ALLOWLIST unset (fail-closed)",
            caller or "<unknown>",
        )
        return Response(content=_reject_twiml("Voice service is not configured."),
                        media_type="application/xml")
    if caller not in _caller_allowlist:
        # Caller number is an identifier, not a credential — log it in full so a
        # dialling campaign is visible. Never log any spoken PIN.
        logger.warning("Inbound call REJECTED — caller not allowlisted: %s", caller or "<unknown>")
        return Response(content=_reject_twiml(), media_type="application/xml")

    from agent.config import config
    host = config.voice_server_public_url.replace("https://", "").replace("http://", "")
    if not host:
        host = request.headers.get("host", "localhost")
    twiml = _get_twiml(host, caller)
    logger.info(f"Incoming call ACCEPTED from {caller} — stream host: {host}")
    return Response(content=twiml, media_type="application/xml")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "moltbot-voice"}


# --- WebSocket Media Stream Handler ---

SILENCE_THRESHOLD_S = 2.0  # seconds of silence before processing (natural pauses ~0.3-0.8s)
MIN_AUDIO_BYTES = 1600     # minimum audio to process (~200ms of 8kHz mulaw)
ENERGY_THRESHOLD = 20      # smoothed mulaw RMS above this = speech (tunable)
ENERGY_WINDOW = 10         # rolling average window (~200ms at ~20ms/chunk)
MIN_SPEECH_S = 0.3         # seconds of sustained speech before marking as speaking
BARGE_IN_ENERGY_MULT = 2   # energy multiplier for barge-in detection (avoids echo triggers)
BARGE_IN_SPEECH_S = 0.4    # sustained speech seconds to trigger barge-in (slightly > MIN_SPEECH_S)


def _mulaw_rms(data: bytes) -> float:
    """Compute approximate RMS energy of mulaw audio for VAD.

    Mulaw silence encodes as 0xFF/0x7F — both give magnitude ~0.
    Speech gives higher magnitudes. Returns 0-127 scale.
    """
    if not data:
        return 0.0
    total = 0
    for b in data:
        mag = (b ^ 0xFF) & 0x7F  # invert + strip sign bit
        total += mag * mag
    return (total / len(data)) ** 0.5


class CallSession:
    """Manages a single phone call's audio pipeline."""

    def __init__(self, websocket: WebSocket, stream_sid: str, caller: str = ""):
        self.ws = websocket
        self.stream_sid = stream_sid
        # Allowlisted caller number, threaded from /incoming-call via the TwiML
        # <Parameter>. Used to key the PIN lockout per-caller (survives re-dials).
        self.caller = caller
        self.audio_buffer = bytearray()
        self._is_speaking: bool = False
        self._silence_start: float = 0.0
        self._speech_start: float = 0.0
        self._energy_window: deque = deque(maxlen=ENERGY_WINDOW)
        self._stop = False
        self._processing = False
        self._playing_back = False
        self._playback_done: asyncio.Event = asyncio.Event()
        self._barge_in = False

    async def run(self):
        """Launch receiver + silence watcher concurrently."""
        # Send greeting first
        try:
            await self._send_greeting()
        except Exception as e:
            logger.warning(f"Greeting failed: {e}")

        await asyncio.gather(
            self._receiver_loop(),
            self._silence_watcher(),
            return_exceptions=True,
        )

    async def _send_greeting(self):
        """Synthesize and send a short greeting when the call connects."""
        from agent.integrations.tts import synthesize_speech
        greeting_audio = await synthesize_speech(
            "Moltbot online. I'm listening.",
            output_format="ulaw_8000",
        )
        await self._send_audio(greeting_audio)
        logger.info("Greeting sent")

    async def _receiver_loop(self):
        """Receive Twilio media events and accumulate audio."""
        try:
            while not self._stop:
                try:
                    data = await asyncio.wait_for(self.ws.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("No data for 30s — ending call")
                    break

                msg = json.loads(data)
                event = msg.get("event")

                if event == "media":
                    payload = msg["media"]["payload"]
                    audio_chunk = base64.b64decode(payload)
                    energy = _mulaw_rms(audio_chunk)

                    # During playback: monitor for barge-in but don't buffer
                    # (audio is echo-contaminated from CLIO's own TTS)
                    if self._playing_back:
                        self._energy_window.append(energy)
                        smoothed = sum(self._energy_window) / len(self._energy_window)
                        now = asyncio.get_running_loop().time()
                        barge_threshold = ENERGY_THRESHOLD * BARGE_IN_ENERGY_MULT
                        if smoothed > barge_threshold:
                            if self._speech_start == 0.0:
                                self._speech_start = now
                            if (now - self._speech_start) >= BARGE_IN_SPEECH_S:
                                logger.info("Barge-in detected — interrupting playback")
                                self._barge_in = True
                                self._playing_back = False
                                self._playback_done.set()
                                await self._send_clear()
                                # Reset VAD state — fresh start after barge-in
                                self._speech_start = 0.0
                                self._energy_window.clear()
                        else:
                            self._speech_start = 0.0
                        continue  # Never buffer during playback

                    # During processing (thinking/STT/TTS): discard entirely
                    if self._processing:
                        continue

                    # Normal listening: buffer audio and run VAD
                    if len(self.audio_buffer) > _MAX_BUFFER_BYTES:
                        logger.warning("S159 H-12: Audio buffer exceeded max — forcing transcription")
                        break
                    self.audio_buffer.extend(audio_chunk)
                    self._energy_window.append(energy)
                    smoothed = sum(self._energy_window) / len(self._energy_window)
                    now = asyncio.get_running_loop().time()
                    if smoothed > ENERGY_THRESHOLD:
                        if self._speech_start == 0.0:
                            self._speech_start = now
                        if not self._is_speaking and (now - self._speech_start) >= MIN_SPEECH_S:
                            self._is_speaking = True
                        self._silence_start = 0.0
                    else:
                        self._speech_start = 0.0
                        if self._is_speaking and self._silence_start == 0.0:
                            self._silence_start = now

                elif event == "stop":
                    logger.info(f"Stream stopped by Twilio (sid={self.stream_sid})")
                    self._stop = True
                    break

                elif event == "mark":
                    mark_name = msg.get("mark", {}).get("name", "")
                    if mark_name == "playback_done" and self._playing_back:
                        logger.info("Playback completed (mark event)")
                        self._playing_back = False
                        self._playback_done.set()

        except WebSocketDisconnect:
            logger.info("Twilio WebSocket disconnected")
        except Exception as e:
            logger.error(f"Receiver error: {e}", exc_info=True)
        finally:
            self._stop = True
            # Unblock any pending playback wait
            self._playback_done.set()

    async def _silence_watcher(self):
        """Detect silence gaps (via audio energy VAD) and trigger processing."""
        await asyncio.sleep(0.2)  # let receiver start first
        while not self._stop:
            await asyncio.sleep(0.1)  # poll every 100ms

            if self._processing:
                continue
            if not self.audio_buffer:
                continue
            if not self._is_speaking:
                continue
            if self._silence_start == 0.0:
                continue

            elapsed = asyncio.get_running_loop().time() - self._silence_start
            if elapsed >= SILENCE_THRESHOLD_S and len(self.audio_buffer) >= MIN_AUDIO_BYTES:
                self._processing = True
                self._barge_in = False  # Reset for new turn
                # Grab buffer and reset speech state
                audio_to_process = bytes(self.audio_buffer)
                self.audio_buffer.clear()
                self._is_speaking = False
                self._silence_start = 0.0
                self._speech_start = 0.0
                self._energy_window.clear()
                try:
                    await self._process_turn(audio_to_process)
                except Exception as e:
                    logger.error(f"Process turn error: {e}", exc_info=True)
                    # Try to send error audio
                    try:
                        from agent.integrations.tts import synthesize_speech
                        error_audio = await synthesize_speech(
                            "Sorry, I ran into an error. Could you repeat that?",
                            output_format="ulaw_8000",
                        )
                        await self._send_audio(error_audio)
                    except Exception:
                        pass
                finally:
                    self._processing = False

    async def _process_turn(self, mulaw_bytes: bytes):
        """Full pipeline: mulaw → Whisper → Claude → TTS → Twilio → wait for playback."""
        from agent.integrations.voice import transcribe_voice
        from agent.integrations.tts import synthesize_speech
        from agent.core import agent_respond

        # Step 1: Wrap mulaw in WAV container for Whisper
        wav_bytes = _wrap_mulaw_as_wav(mulaw_bytes)
        logger.info(f"Processing {len(mulaw_bytes)} bytes of mulaw audio")

        # Step 2: Transcribe (Whisper) ~1s
        transcript = await transcribe_voice(wav_bytes, filename="call.wav")
        if not transcript or transcript.startswith("ERROR"):
            logger.warning(f"Transcription failed or empty: {transcript}")
            return
        logger.info(f"Transcribed: {transcript[:100]}")

        # Step 3: Query Claude ~2s
        response_text = await agent_respond(
            user_message=transcript,
            conversation_id=f"voice_{self.stream_sid}",
            task_type="general",
            voice_mode=True,
            source="twilio_voice",
            # Caller number doubles as user_id on the voice path. process_override()
            # never uses user_id for twilio_voice authorization (is_authorized is
            # forced False there), so this is purely an identity for lockout keying.
            user_id=self.caller,
        )
        logger.info(f"Claude response ({len(response_text)} chars): {response_text[:100]}")

        # Step 4: TTS (ElevenLabs) ~1s — mulaw format for Twilio
        try:
            from agent.integrations.tts import strip_markdown
            response_text = strip_markdown(response_text)
        except Exception:
            pass
        tts_audio = await synthesize_speech(response_text, output_format="ulaw_8000")
        logger.info(f"TTS: {len(response_text)} chars → {len(tts_audio)} bytes ulaw")

        # Step 5: Stream audio back to Twilio and wait for playback to finish
        self._playback_done.clear()
        self._barge_in = False
        self._playing_back = True
        await self._send_audio(tts_audio)

        # Step 6: Wait for Twilio mark event (playback done) or barge-in
        try:
            await asyncio.wait_for(self._playback_done.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("Playback mark timeout (60s) — assuming done")
            self._playing_back = False

        # After playback ends (naturally or barge-in), discard any buffered audio
        # — it's either echo from CLIO's response or stale speech from during playback
        self.audio_buffer.clear()
        self._is_speaking = False
        self._silence_start = 0.0
        self._speech_start = 0.0
        self._energy_window.clear()

        if self._barge_in:
            logger.info("Turn ended by barge-in — ready for new utterance")

    async def _send_audio(self, audio_bytes: bytes):
        """Send mulaw audio bytes back to Twilio media stream in base64 chunks."""
        CHUNK_SIZE = 640  # 40ms of 8kHz mulaw (8000 samples/s * 0.04s * 1 byte/sample)
        for i in range(0, len(audio_bytes), CHUNK_SIZE):
            if self._barge_in:
                logger.info(f"Barge-in: stopped audio send at {i}/{len(audio_bytes)} bytes")
                break
            chunk = audio_bytes[i : i + CHUNK_SIZE]
            payload = base64.b64encode(chunk).decode("utf-8")
            try:
                await self.ws.send_text(
                    json.dumps(
                        {
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": payload},
                        }
                    )
                )
            except Exception as e:
                logger.debug(f"Audio send error: {e}")
                break  # WebSocket closed

        # Send a mark event so Twilio tells us when playback finishes
        # Skip if barge-in fired — the stale mark would interfere with the next turn
        if not self._barge_in:
            try:
                await self.ws.send_text(
                    json.dumps(
                        {
                            "event": "mark",
                            "streamSid": self.stream_sid,
                            "mark": {"name": "playback_done"},
                        }
                    )
                )
            except Exception as e:
                logger.debug(f"Mark event send failed: {e}")

    async def _send_clear(self):
        """Send clear event to Twilio to stop all queued audio playback."""
        try:
            await self.ws.send_text(
                json.dumps(
                    {
                        "event": "clear",
                        "streamSid": self.stream_sid,
                    }
                )
            )
            logger.info("Sent clear event to Twilio — playback interrupted")
        except Exception as e:
            logger.debug(f"Clear event send failed: {e}")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Twilio Media Stream WebSocket endpoint."""
    await websocket.accept()
    logger.info("Twilio media stream WebSocket connected")

    stream_sid = None
    caller = ""
    try:
        # Wait for the "start" event to get the stream SID
        while True:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            msg = json.loads(data)
            if msg.get("event") == "start":
                stream_sid = msg["start"]["streamSid"]
                _params = msg["start"].get("customParameters") or {}
                caller = _normalize_e164(str(_params.get("caller", "")))
                logger.info(f"Stream started: sid={stream_sid} caller={caller or '<unknown>'}")
                break
            elif msg.get("event") == "connected":
                logger.info("Twilio connected event received")
                continue

        if not stream_sid:
            logger.error("Never received start event")
            return

        session = CallSession(websocket, stream_sid, caller)
        await session.run()

    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for stream start")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected before stream started")
    except Exception as e:
        logger.error(f"Media stream error: {e}", exc_info=True)
    finally:
        logger.info(f"Call ended: sid={stream_sid}")


# --- Audio Utilities ---

def _wrap_mulaw_as_wav(mulaw_bytes: bytes) -> bytes:
    """Wrap raw mulaw PCM bytes in a WAV container so Whisper can decode it.

    WAV format: RIFF header + fmt chunk (mulaw, 8kHz, mono) + data chunk.
    """
    num_channels = 1
    sample_rate = 8000
    bits_per_sample = 8
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(mulaw_bytes)
    # fmt chunk: format=7 (mulaw), channels, sample_rate, byte_rate, block_align, bits_per_sample
    fmt_chunk = struct.pack(
        "<HHIIHH",
        7,               # wFormatTag: 7 = mulaw
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    # RIFF header + fmt + data
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + data_size)  # "WAVE" + fmt_chunk_header + data_chunk_header
    header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    fmt_header = struct.pack("<4sI", b"fmt ", len(fmt_chunk))
    data_header = struct.pack("<4sI", b"data", data_size)
    return header + fmt_header + fmt_chunk + data_header + mulaw_bytes


# --- Entry Point ---

if __name__ == "__main__":
    import uvicorn
    from agent.config import config

    logger.info(f"Starting Moltbot Voice Server on port {config.voice_server_port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.voice_server_port,
        log_level="info",
    )
