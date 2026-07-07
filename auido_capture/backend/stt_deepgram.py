"""
stt_deepgram.py — wraps one Deepgram live-streaming connection.

Diagram maps to this as the two "Deepgram WS" boxes:
  - mic channel     -> speaker = agent
  - system channel  -> speaker = customer

Each instance owns exactly one Deepgram connection. The Audio Router creates
two of these per session (one per channel) and streams raw PCM straight
through (no client-side gating — Deepgram's own endpointing handles
segmentation).
"""

import asyncio
import logging
from typing import Callable, Optional

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)

from errors import classify_error, retry_with_backoff

logger = logging.getLogger("insureassist.layer1")


class DeepgramChannelSTT:
    # Deepgram closes a live connection with code 1011 if it receives NO
    # audio bytes and NO text/KeepAlive message within its timeout window
    # (~10-12s, see https://dpgr.am/net0001). Relying on VAD-triggered
    # keepalives is NOT enough — if a channel never receives any PCM at all
    # (e.g. a shared tab with no live audio track), handle_pcm() is never
    # even called, so no keepalive would ever fire. This background loop
    # runs for the lifetime of the connection regardless of audio flow.
    KEEPALIVE_INTERVAL_S = 5.0

    def __init__(
        self,
        api_key: str,
        speaker_label: str,          # "agent" or "customer"
        on_final_segment: Callable[[dict], None],
        session_start_time: float,   # time.time() when the session/router started
        sample_rate: int = 16000,
    ):
        self.speaker_label = speaker_label
        self.sample_rate = sample_rate
        self.on_final_segment = on_final_segment
        self.session_start_time = session_start_time
        self._client = DeepgramClient(api_key)
        self._connection = None
        self._connected = False
        self._keepalive_task: Optional[asyncio.Task] = None

    async def connect(self):
        async def _do_connect():
            try:
                self._connection = self._client.listen.asynclive.v("1")

                async def on_message(_, result, **kwargs):
                    alt = result.channel.alternatives[0]
                    if not alt.transcript:
                        return

                    # result.start is Deepgram's offset (seconds) of this
                    # segment's audio relative to when THIS stream began —
                    # so session_start_time + result.start ≈ wall-clock time
                    # the words were actually spoken. transcribed_at is
                    # simply "now" — when the final event reached us.
                    import time as _time
                    spoken_at = self.session_start_time + (result.start or 0.0)
                    transcribed_at = _time.time()

                    segment = {
                        "speaker": self.speaker_label,
                        "text": alt.transcript,
                        "is_final": result.is_final,
                        "timestamp": result.start,
                        "spoken_at": spoken_at,
                        "transcribed_at": transcribed_at,
                        "latency_ms": (transcribed_at - spoken_at) * 1000.0,
                    }
                    if result.is_final:
                        self.on_final_segment(segment)

                async def on_error(_, error, **kwargs):
                    status_code = getattr(error, "code", None)
                    raise classify_error(status_code, str(error))

                self._connection.on(LiveTranscriptionEvents.Transcript, on_message)
                self._connection.on(LiveTranscriptionEvents.Error, on_error)

                options = LiveOptions(
                    model="nova-2",
                    language="en-US",
                    encoding="linear16",
                    sample_rate=self.sample_rate,
                    channels=1,
                    interim_results=True,
                    endpointing=300,   # ms of silence before Deepgram finalizes a segment
                )
                started = await self._connection.start(options)
                if not started:
                    raise classify_error(None, "Deepgram connection failed to start")
                self._connected = True
            except Exception as e:
                # normalize any raw exception into our classified error scheme
                if not hasattr(e, "action"):
                    raise classify_error(None, str(e))
                raise

        await retry_with_backoff(_do_connect, max_attempts=5, base_delay_s=1.0)
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self):
        """
        Runs independently of handle_pcm()/VAD — guarantees Deepgram always
        sees a message at least every KEEPALIVE_INTERVAL_S seconds, even if
        this channel never receives a single real audio byte (e.g. a shared
        browser tab with no live audio track).
        """
        try:
            while self._connected:
                await asyncio.sleep(self.KEEPALIVE_INTERVAL_S)
                if self._connected and self._connection:
                    ok = await self._connection.keep_alive()
                    if not ok:
                        logger.warning(f"[{self.speaker_label}] keepalive send failed")
        except asyncio.CancelledError:
            pass

    async def send_pcm(self, pcm_bytes: bytes):
        if self._connected and self._connection:
            await self._connection.send(pcm_bytes)

    async def send_keepalive(self):
        # Deepgram supports a KeepAlive message to hold the socket open
        # during silence without sending real audio (matches the "keepalive
        # every 100ms" note in the VAD box of the diagram).
        if self._connected and self._connection:
            await self._connection.keep_alive()

    async def close(self):
        self._connected = False
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        if self._connection:
            await self._connection.finish()