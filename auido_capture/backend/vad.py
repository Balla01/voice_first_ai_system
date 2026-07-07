"""
vad.py — Voice Activity Detection (VAD) layer.

Matches the diagram's "VAD" box:
  - adaptive RMS threshold
  - keepalive every 100ms (so downstream STT sockets don't time out during silence)

This is intentionally a lightweight, dependency-free VAD (pure numpy) rather than
webrtcvad/silero — good enough for a hackathon, and easy to explain/demo live.
If you have time budget left, swapping in silero-vad is a drop-in upgrade later
(same interface: bytes in -> (is_speech: bool, pcm: bytes) out).
"""

import time
import numpy as np


class AdaptiveRMSVAD:
    """
    Tracks a rolling noise floor per audio stream and classifies each incoming
    PCM16 chunk as speech / silence based on how far above the floor it is.

    One instance per (session_id, channel) — e.g. one for mic, one for system.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        noise_floor_alpha: float = 0.05,   # EMA smoothing for the noise floor
        speech_multiplier: float = 2.5,    # speech = floor * multiplier
        keepalive_interval_s: float = 0.1, # 100ms keepalive, per the diagram
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.noise_floor_alpha = noise_floor_alpha
        self.speech_multiplier = speech_multiplier
        self.keepalive_interval_s = keepalive_interval_s

        self._noise_floor = 50.0  # seed value, adapts quickly in first ~1s
        self._last_emit_ts = 0.0

    @staticmethod
    def _rms(pcm16_bytes: bytes) -> float:
        if not pcm16_bytes:
            return 0.0
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples**2)))

    def process(self, pcm16_bytes: bytes) -> dict:
        """
        Returns:
            {
              "is_speech": bool,
              "pcm": bytes,          # the same chunk, passed through if speech
              "send_keepalive": bool # true if we should ping STT socket to hold it open
            }
        """
        rms = self._rms(pcm16_bytes)
        now = time.monotonic()

        is_speech = rms > (self._noise_floor * self.speech_multiplier)

        # Only adapt the floor on quiet frames, so a sustained loud voice
        # doesn't drag the floor upward and desensitize the detector.
        if not is_speech:
            self._noise_floor = (
                (1 - self.noise_floor_alpha) * self._noise_floor
                + self.noise_floor_alpha * rms
            )

        send_keepalive = False
        if is_speech:
            self._last_emit_ts = now
        elif now - self._last_emit_ts >= self.keepalive_interval_s:
            send_keepalive = True
            self._last_emit_ts = now

        return {
            "is_speech": is_speech,
            "pcm": pcm16_bytes,
            "send_keepalive": send_keepalive,
        }