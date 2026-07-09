"""
audio_router.py — the green "Audio Router" box.

One AudioRouter per session_id. Owns:
  - two DeepgramChannelSTT instances (mic->agent, system->customer)
  - one TranscriptMerger shared across both channels

Wiring:  PCM bytes in (from WebSocket) -> Deepgram (continuous, ungated)
         Deepgram finals -> TranscriptMerger -> merged conversation out

NOTE on VAD: an earlier version of this file gated audio through a custom
adaptive-RMS VAD and only forwarded frames classified as "speech" to
Deepgram. In testing this clipped the start/end of words whenever the RMS
dipped below threshold mid-utterance ("not capturing all audio"). Deepgram
already does its own voice-activity/endpointing server-side, so we now
stream ALL audio through continuously and let Deepgram handle segmentation.
Keepalive is handled independently by a background loop inside
DeepgramChannelSTT, so it no longer depends on VAD state either.
"""

from stt_deepgram import DeepgramChannelSTT
from transcript_merger import TranscriptMerger


class AudioRouter:
    def __init__(self, session_id: str, deepgram_api_key: str, on_merged_final, session_start_time: float):
        self.session_id = session_id

        self.merger = TranscriptMerger(on_merged_final=on_merged_final)

        self.stt = {
            "mic": DeepgramChannelSTT(
                api_key=deepgram_api_key,
                speaker_label="agent",
                on_final_segment=self.merger.add_segment,
                session_start_time=session_start_time,
            ),
            "system": DeepgramChannelSTT(
                api_key=deepgram_api_key,
                speaker_label="customer",
                on_final_segment=self.merger.add_segment,
                session_start_time=session_start_time,
            ),
        }

    async def start(self):
        await self.stt["mic"].connect()
        await self.stt["system"].connect()

    async def handle_pcm(self, stream_id: str, pcm_bytes: bytes):
        """
        stream_id: "mic" or "system" — matches the WSS query param from the
        browser client (see diagram: "WSS | stream_id: mic/system | session_id header")
        """
        if stream_id not in ("mic", "system"):
            raise ValueError(f"Unknown stream_id: {stream_id}")

        await self.stt[stream_id].send_pcm(pcm_bytes)

    async def close(self):
        await self.stt["mic"].close()
        await self.stt["system"].close()