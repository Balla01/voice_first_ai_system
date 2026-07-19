"""
turn_accumulator.py — debounces the stream of final transcript segments from
TranscriptMerger into complete *turns* before anything downstream (Layer 3
memory + Layer 4 router) sees them.

Why: Deepgram emits a single spoken sentence as several final segments split
on short pauses (e.g. "what type of plan like" + "a lic new pension plus
cover"). Routing each fragment separately caused duplicate fires, wasted
LLM calls, and aborted answers. Buffering until a short silence collapses
those fragments into one turn -> one routing decision -> one answer.

Design (speaker-agnostic — either speaker can trigger a query in this tool):
  - Segments are buffered per current speaker. Each new segment resets a
    silence timer; after SILENCE_FLUSH_S with no new segment, the buffer is
    joined into one turn and emitted via on_turn.
  - A segment from a DIFFERENT speaker is a hard flush: the in-progress turn
    is emitted immediately, then the new speaker's buffer starts.
  - on_turn receives one synthesized TranscriptSegment carrying the joined
    text, the earliest spoken_at and latest transcribed_at of the fragments.
"""

import asyncio
from typing import Awaitable, Callable, List, Optional, Set

from transcript_merger import TranscriptSegment


class TurnAccumulator:
    SILENCE_FLUSH_S = 1.5

    def __init__(self, on_turn: Callable[[TranscriptSegment], Awaitable[None]]):
        self.on_turn = on_turn
        self._buffer: List[TranscriptSegment] = []
        self._speaker: Optional[str] = None
        self._flush_task: Optional[asyncio.Task] = None
        # Keep strong refs to fire-and-forget tasks so they aren't GC'd mid-flight.
        self._background_tasks: Set[asyncio.Task] = set()

    def _fire_and_forget(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def add_segment(self, segment: TranscriptSegment):
        """Called synchronously from TranscriptMerger's callback chain."""
        # Speaker switch — flush whatever the previous speaker had buffered first.
        if self._buffer and segment.speaker != self._speaker:
            self._flush()

        self._speaker = segment.speaker
        self._buffer.append(segment)
        self._reset_flush_timer()

    def _reset_flush_timer(self):
        if self._flush_task is not None:
            self._flush_task.cancel()
        self._flush_task = self._fire_and_forget(self._flush_after_silence())

    async def _flush_after_silence(self):
        try:
            await asyncio.sleep(self.SILENCE_FLUSH_S)
        except asyncio.CancelledError:
            return
        self._flush()

    def _flush(self):
        if not self._buffer:
            return
        segs = self._buffer
        self._buffer = []
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None

        text = " ".join(s.text for s in segs).strip()
        if not text:
            return

        first, last = segs[0], segs[-1]
        latency_ms = None
        if first.spoken_at is not None and last.transcribed_at is not None:
            latency_ms = (last.transcribed_at - first.spoken_at) * 1000.0

        merged = TranscriptSegment(
            speaker=self._speaker,
            text=text,
            is_final=True,
            timestamp=last.timestamp,
            spoken_at=first.spoken_at,
            transcribed_at=last.transcribed_at,
            latency_ms=latency_ms,
        )
        self._fire_and_forget(self.on_turn(merged))

    def close(self):
        """Cancel the pending silence timer on session teardown. Any partially
        buffered turn is dropped (the sockets are closing — nowhere to send an
        answer)."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None
        self._buffer = []
