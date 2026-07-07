"""
transcript_merger.py — the purple "Transcript Merger" box.

Takes final segments arriving asynchronously from the mic channel (agent)
and system channel (customer) Deepgram connections, and merges them into
a single time-ordered conversation stream:

    { speaker, text, is_final, timestamp }

- dedup window: 500ms — if two segments from the *same* speaker arrive
  within 500ms of each other with overlapping text (e.g. an interim that
  slipped through, or a duplicate final on reconnect), they're collapsed
  into one.
- only finals are pushed onward to context (e.g. the RAG / policy-lookup
  layer downstream) — interim results are used for live UI display only,
  not for downstream reasoning.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class TranscriptSegment:
    speaker: str        # "agent" | "customer"
    text: str
    is_final: bool
    timestamp: float
    spoken_at: Optional[float] = None       # wall-clock time (epoch s) speech began
    transcribed_at: Optional[float] = None  # wall-clock time final was received
    latency_ms: Optional[float] = None      # transcribed_at - spoken_at, in ms


class TranscriptMerger:
    DEDUP_WINDOW_S = 0.5

    def __init__(self, on_merged_final: Callable[[TranscriptSegment], None]):
        """
        on_merged_final: callback invoked with each deduped final segment,
        in arrival order. This is what feeds "to context" downstream
        (e.g. the RAG layer that surfaces policy info to the agent).
        """
        self.on_merged_final = on_merged_final
        self._history: List[TranscriptSegment] = []

    def _is_duplicate(self, candidate: TranscriptSegment) -> bool:
        for prior in reversed(self._history):
            if prior.speaker != candidate.speaker:
                continue
            if candidate.timestamp - prior.timestamp > self.DEDUP_WINDOW_S:
                break  # history is time-ordered; nothing older can match either
            if prior.text.strip() == candidate.text.strip():
                return True
        return False

    def add_segment(self, segment: dict):
        """
        segment: {"speaker": str, "text": str, "is_final": bool, "timestamp": float}
        Only finals are processed — interims should be routed straight to the
        live UI elsewhere and never reach this merger.
        """
        if not segment.get("is_final"):
            return

        candidate = TranscriptSegment(
            speaker=segment["speaker"],
            text=segment["text"],
            is_final=True,
            timestamp=segment.get("timestamp") or time.monotonic(),
            spoken_at=segment.get("spoken_at"),
            transcribed_at=segment.get("transcribed_at"),
            latency_ms=segment.get("latency_ms"),
        )

        if self._is_duplicate(candidate):
            return

        self._history.append(candidate)
        self.on_merged_final(candidate)

    def get_conversation(self) -> List[TranscriptSegment]:
        return list(self._history)