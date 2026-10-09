from __future__ import annotations

import os
import queue
import threading
import logging
import time
from pathlib import Path

import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v1.types import SpeakV1Text

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in environment variables.")

VOICE_MODEL = os.getenv("DEEPGRAM_VOICE_MODEL", "aura-2-jupiter-en")
SAMPLE_RATE = 24000


def speak_streaming(text: str, model: str = VOICE_MODEL) -> None:
    """Stream Deepgram TTS audio and play it at the correct sample rate."""

    if not text or not text.strip():
        return

    client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    audio_q: "queue.Queue[bytes]" = queue.Queue()

    # Persistent audio buffer
    audio_buffer = bytearray()
    buffer_lock = threading.Lock()

    # Bookkeeping: how many bytes Deepgram sent vs. how many PortAudio consumed
    state_cv = threading.Condition()
    received_bytes = 0
    played_bytes = 0

    # ------------------------------------------------------------------
    # PortAudio callback
    # ------------------------------------------------------------------
    def output_callback(outdata, frames, time_info, status):
        nonlocal played_bytes

        if status:
            logger.warning("Audio stream status: %s", status)

        needed_bytes = frames * 2  # int16, mono

        with buffer_lock:
            # Pull from the network queue until we have enough for this block
            while len(audio_buffer) < needed_bytes:
                try:
                    chunk = audio_q.get_nowait()
                except queue.Empty:
                    break
                audio_buffer.extend(chunk)

            take = min(len(audio_buffer), needed_bytes)
            chunk = bytes(audio_buffer[:take])
            del audio_buffer[:take]

        # Write audio + pad with silence if underrunning
        if chunk:
            outdata[: len(chunk)] = chunk
        if len(chunk) < needed_bytes:
            outdata[len(chunk):needed_bytes] = b"\x00" * (needed_bytes - len(chunk))

        if take:
            with state_cv:
                played_bytes += take
                state_cv.notify_all()

    stream = sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=1024,
        callback=output_callback,
    )
    stream.start()

    try:
        with client.speak.v1.connect(
            model=model,
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
        ) as dg_connection:

            def on_open(_):
                logger.debug("Deepgram TTS socket opened")
                dg_connection.send_text(SpeakV1Text(text=text))
                dg_connection.send_flush()
                # IMPORTANT: do NOT send_close() here. Wait for "Flushed".

            def on_message(message):
                nonlocal received_bytes

                if isinstance(message, bytes):
                    # Audio payload
                    audio_q.put(message)
                    with state_cv:
                        received_bytes += len(message)
                        state_cv.notify_all()
                else:
                    msg_type = getattr(message, "type", None) or type(message).__name__
                    logger.debug("Deepgram event: %s", msg_type)

                    # All audio for our text has been synthesized; now it's
                    # safe to close the socket.
                    if "flush" in str(msg_type).lower():
                        try:
                            dg_connection.send_close()
                        except Exception:
                            logger.debug("send_close() failed (likely already closing)")

            def on_close(_):
                logger.debug("Deepgram TTS socket closed")

            def on_error(error):
                logger.error("Deepgram TTS error: %s", error)

            dg_connection.on(EventType.OPEN, on_open)
            dg_connection.on(EventType.MESSAGE, on_message)
            dg_connection.on(EventType.CLOSE, on_close)
            dg_connection.on(EventType.ERROR, on_error)

            # Blocks until the socket is closed
            dg_connection.start_listening()

    except Exception:
        logger.exception("Deepgram TTS failed")

    finally:
        # ------------------------------------------------------------------
        # Wait until EVERY byte Deepgram sent has been consumed by the
        # PortAudio callback. Deepgram synthesizes far faster than real-time,
        # so the queue can still hold many seconds of audio at this point.
        # ------------------------------------------------------------------
        deadline = time.time() + 120.0   # generous safety cap
        with state_cv:
            while played_bytes < received_bytes and time.time() < deadline:
                state_cv.wait(timeout=0.5)

        # Small extra drain so PortAudio's own internal buffer empties
        # (blocksize=1024 @ 24 kHz ≈ 43 ms per callback).
        time.sleep(0.5)

        stream.stop()
        stream.close()