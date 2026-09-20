from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Callable, Optional

import sounddevice as sd
from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v2.types import ListenV2CloseStream, ListenV2TurnInfo

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in environment variables.")

# ── Audio capture ────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
CHANNELS     = 1
DTYPE        = "int16"
BLOCKSIZE    = 1024
QUEUE_WAIT_S = 0.1
QUEUE_MAX    = 200          # ~12 s of headroom before we drop chunks

# ── Flux turn detection ──────────────────────────────────────────────
EOT_THRESHOLD        = 0.85 # default 0.7 — stricter, avoids cutting off commands
EOT_TIMEOUT_MS       = 6000 # default 5000
MIN_WORDS_BEFORE_EOT = 2    # fragments shorter than this don't close a turn
SHORT_EOT_GRACE_S    = 2.0  # keep listening briefly after a fragmentary EOT
HARD_TIMEOUT_S       = 30.0 # absolute ceiling on one listen_once() call

_EVT_UPDATE    = "Update"
_EVT_EAGER_EOT = "EagerEndOfTurn"
_EVT_EOT       = "EndOfTurn"


class _TurnState:
    """Thread-safe state shared between the audio loop and Flux callbacks."""

    __slots__ = ("_lock", "_finals", "_partial", "_done", "_errored", "_tail_timer")

    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self._finals: list[str] = []
        self._partial    = ""
        self._done       = threading.Event()
        self._errored    = threading.Event()
        self._tail_timer: Optional[threading.Timer] = None

    # ── writers (called from listener thread) ────────────────────
    def add_final(self, text: str) -> None:
        with self._lock:
            self._finals.append(text)

    def set_partial(self, text: str) -> None:
        with self._lock:
            self._partial = text

    def clear_partial(self) -> None:
        with self._lock:
            self._partial = ""

    def commit(self) -> None:
        self._cancel_tail()
        self._done.set()

    def fail(self) -> None:
        self._cancel_tail()
        self._errored.set()
        self._done.set()

    def arm_short_eot_grace(self, seconds: float) -> None:
        """Fragmentary EOT — give Flux one more chance to finish the thought."""
        self._cancel_tail()
        self._tail_timer = threading.Timer(seconds, self._done.set)
        self._tail_timer.daemon = True
        self._tail_timer.start()

    def _cancel_tail(self) -> None:
        if self._tail_timer is not None:
            self._tail_timer.cancel()
            self._tail_timer = None

    # ── readers ──────────────────────────────────────────────────
    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    @property
    def errored(self) -> bool:
        return self._errored.is_set()

    def total_words(self) -> int:
        with self._lock:
            return sum(len(p.split()) for p in self._finals)

    def result(self) -> str:
        with self._lock:
            if self._finals:
                return " ".join(self._finals).strip()
            return self._partial.strip()


def _unwrap(message) -> tuple[Optional[str], str]:
    """Normalise a Flux message into (event, transcript)."""
    if isinstance(message, ListenV2TurnInfo):
        return message.event, (message.transcript or "").strip()
    if isinstance(message, dict) and message.get("type") == "TurnInfo":
        return message.get("event"), (message.get("transcript") or "").strip()
    return None, ""


def listen_once(
    *,
    on_partial: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> str:
    """
    Capture one user utterance via Deepgram Flux.

    Returns the finalized transcript, or the latest partial if Flux never
    emitted a clean EndOfTurn before the hard timeout. Returns "" on a
    connection-level failure.

    Args:
        on_partial:  optional callback invoked with the latest interim text.
        stop_event:  optional external cancel flag. If set, listen_once
                     returns as soon as the audio loop notices it.
    """
    client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    state = _TurnState()
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=QUEUE_MAX)
    stop = stop_event or threading.Event()

    def mic_callback(indata, frames, time_info, status):
        if status:
            logger.warning("Microphone status: %s", status)
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            logger.warning("Audio queue full — dropping chunk")

    def on_open(_):
        logger.debug("Flux connection opened")

    def on_message(message):
        try:
            event, transcript = _unwrap(message)
            if not event or not transcript:
                return

            logger.debug("Flux [%s]: %s", event, transcript)

            if event in (_EVT_UPDATE, _EVT_EAGER_EOT):
                state.set_partial(transcript)
                if on_partial is not None:
                    on_partial(transcript)
                return

            if event == _EVT_EOT:
                state.clear_partial()
                state.add_final(transcript)
                if state.total_words() >= MIN_WORDS_BEFORE_EOT:
                    state.commit()
                else:
                    # Fragmentary end-of-turn — keep listening briefly in
                    # case Flux is about to emit a fuller continuation.
                    state.arm_short_eot_grace(SHORT_EOT_GRACE_S)
        except Exception:
            logger.exception("Error handling Flux turn info")

    def on_close(_):
        logger.debug("Flux connection closed")
        state.commit()

    def on_error(error):
        logger.error("Flux error: %s", error)
        state.fail()

    try:
        with client.listen.v2.connect(
            model="flux-general-en",
            encoding="linear16",
            sample_rate=str(SAMPLE_RATE),
            eot_threshold=EOT_THRESHOLD,
            eot_timeout_ms=EOT_TIMEOUT_MS,
        ) as dg_connection:
            dg_connection.on(EventType.OPEN,    on_open)
            dg_connection.on(EventType.MESSAGE, on_message)
            dg_connection.on(EventType.CLOSE,   on_close)
            dg_connection.on(EventType.ERROR,   on_error)

            listener = threading.Thread(
                target=dg_connection.start_listening, daemon=True
            )
            listener.start()

            deadline = time.monotonic() + HARD_TIMEOUT_S

            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCKSIZE,
                callback=mic_callback,
            ):
                logger.info("Listening… speak now.")

                while not state.is_done:
                    if stop.is_set():
                        logger.info("listen_once: external stop requested")
                        break
                    if time.monotonic() >= deadline:
                        logger.info("listen_once: hard timeout reached")
                        break

                    try:
                        chunk = audio_queue.get(timeout=QUEUE_WAIT_S)
                    except queue.Empty:
                        continue

                    try:
                        dg_connection.send_media(chunk)
                    except Exception:
                        logger.exception("Failed to send audio chunk")
                        state.fail()
                        break

            # Tell Deepgram we're done so it flushes any buffered audio.
            try:
                dg_connection.send_close_stream(
                    ListenV2CloseStream(type="CloseStream")
                )
            except Exception as e:
                logger.debug("CloseStream send failed: %s", e)

            listener.join(timeout=2.0)

    except Exception:
        logger.exception("Deepgram Flux connection failed")
        return ""

    if state.errored:
        return ""

    text = state.result()
    logger.info("Utterance: %r", text)
    return text