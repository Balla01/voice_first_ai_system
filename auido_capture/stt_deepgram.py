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

from errors import classify_error, retry_with_backoff, ErrorAction

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
        encoding: str = "linear16",  # "linear16" for browser PCM16, "mulaw" for Twilio phone audio
    ):
        self.speaker_label = speaker_label
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.on_final_segment = on_final_segment
        self.session_start_time = session_start_time
        self._client = DeepgramClient(api_key)
        self._connection = None
        self._connected = False
        self._keepalive_task: Optional[asyncio.Task] = None
        # Reconnect bookkeeping: distinguish an intentional close() from a
        # dropped/idle-closed connection, avoid overlapping reconnects, and
        # stop retrying on a fatal (e.g. 401 auth) error.
        self._closing = False
        self._reconnecting = False
        self._fatal = False

    async def connect(self):
        await retry_with_backoff(self._do_connect, max_attempts=5, base_delay_s=1.0)
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _do_connect(self):
        """Establish (or re-establish) the Deepgram connection and register
        handlers. Re-runnable — used for both the initial connect and every
        reconnect."""
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
                # Classify: an auth/other fatal error must NOT trigger an
                # endless reconnect loop; a transient/connection error should.
                status_code = getattr(error, "code", None)
                err = classify_error(status_code, str(error))
                if getattr(err, "action", None) == ErrorAction.FATAL_NO_RETRY:
                    logger.error(f"[{self.speaker_label}] Deepgram fatal error (no reconnect): {err}")
                    self._fatal = True
                    self._connected = False
                else:
                    logger.warning(f"[{self.speaker_label}] Deepgram error, will reconnect: {err}")
                    self._trigger_reconnect()

            async def on_close(_, close, **kwargs):
                # The 1011 idle-timeout / any unexpected close lands here. Unless
                # we're closing on purpose (or hit a fatal error), reconnect so
                # the channel self-heals instead of going silent for the session.
                if self._closing or self._fatal:
                    return
                logger.warning(f"[{self.speaker_label}] Deepgram connection closed unexpectedly; reconnecting")
                self._trigger_reconnect()

            self._connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self._connection.on(LiveTranscriptionEvents.Error, on_error)
            self._connection.on(LiveTranscriptionEvents.Close, on_close)

            options = LiveOptions(
                model="nova-2",
                language="en-US",
                encoding=self.encoding,
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

    def _trigger_reconnect(self):
        """Schedule a single reconnect. Guarded so overlapping error/close/
        keepalive signals don't spawn competing reconnect tasks."""
        if self._closing or self._fatal or self._reconnecting:
            return
        self._reconnecting = True
        self._connected = False
        asyncio.create_task(self._reconnect())

    async def _reconnect(self):
        try:
            if self._keepalive_task:
                self._keepalive_task.cancel()
                self._keepalive_task = None
            try:
                if self._connection:
                    await self._connection.finish()
            except Exception:
                pass   # best-effort teardown of the dead connection

            await retry_with_backoff(self._do_connect, max_attempts=5, base_delay_s=1.0)
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            logger.info(f"[{self.speaker_label}] Deepgram reconnected")
        except Exception as e:
            logger.error(f"[{self.speaker_label}] Deepgram reconnect failed after retries: {e}")
        finally:
            self._reconnecting = False

    async def _keepalive_loop(self):
        """
        Runs independently of handle_pcm()/VAD — guarantees Deepgram always
        sees a message at least every KEEPALIVE_INTERVAL_S seconds, even if
        this channel never receives a single real audio byte (e.g. a shared
        browser tab with no live audio track). Resilient by design: a single
        failed keep_alive() reconnects rather than killing the loop forever
        (the old version only caught CancelledError, so any other raise left
        the channel with no keepalive until session restart).
        """
        try:
            while self._connected:
                await asyncio.sleep(self.KEEPALIVE_INTERVAL_S)
                if not self._connected or not self._connection:
                    break
                try:
                    ok = await self._connection.keep_alive()
                    if ok is False:
                        logger.warning(f"[{self.speaker_label}] keepalive send failed; reconnecting")
                        self._trigger_reconnect()
                        break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"[{self.speaker_label}] keepalive raised ({e}); reconnecting")
                    self._trigger_reconnect()
                    break
        except asyncio.CancelledError:
            pass

    async def send_pcm(self, pcm_bytes: bytes):
        if not (self._connected and self._connection):
            return   # dropped mid-reconnect — skip this frame, audio resumes once reconnected
        try:
            await self._connection.send(pcm_bytes)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[{self.speaker_label}] send_pcm failed ({e}); reconnecting")
            self._trigger_reconnect()

    async def send_keepalive(self):
        # Deepgram supports a KeepAlive message to hold the socket open
        # during silence without sending real audio (matches the "keepalive
        # every 100ms" note in the VAD box of the diagram).
        if self._connected and self._connection:
            await self._connection.keep_alive()

    async def close(self):
        self._closing = True   # suppress reconnect on the resulting Close event
        self._connected = False
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        if self._connection:
            try:
                await self._connection.finish()
            except Exception:
                pass