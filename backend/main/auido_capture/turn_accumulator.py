"""
turn_accumulator.py — decides "what counts as one RAG query" from a live
stream of speaker-tagged final transcript segments coming out of TranscriptMerger.

Design:
  - Customer segments are buffered. Each new customer segment resets a
    silence timer; when SILENCE_FLUSH_S passes with no new segment, the
    buffered text is joined into one "final query" and flushed.
  - An agent segment is a hard flush trigger too (speaker switch) — if the
    customer was mid-turn, whatever's buffered is flushed immediately before
    the agent segment is forwarded.
  - Only customer turns trigger a RAG query. Agent speech is forwarded
    on_agent_segment (for conversational memory) but never itself queries.
"""

import asyncio
from typing import Awaitable, Callable, List, Optional, Set


class TurnAccumulator:
    SILENCE_FLUSH_S = 1.5

    def __init__(
        self,
        on_customer_turn: Callable[[str], Awaitable[None]],
        on_agent_segment: Callable[[str], Awaitable[None]],
    ):
        self.on_customer_turn = on_customer_turn
        self.on_agent_segment = on_agent_segment
        self._buffer: List[str] = []
        self._flush_task: Optional[asyncio.Task] = None
        # Holds references to fire-and-forget callback tasks so they aren't
        # garbage-collected mid-flight (asyncio only weakly references a task
        # once nothing else does — see create_task()'s own docs warning).
        self._background_tasks: Set[asyncio.Task] = set()

    def _fire_and_forget(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def add_segment(self, segment):
        """
        segment: TranscriptSegment (has .speaker, .text). Called synchronously
        from TranscriptMerger's callback chain (itself invoked from an async
        Deepgram message handler) — schedules async work via
        asyncio.create_task rather than blocking the caller.
        """
        if segment.speaker == "agent":
            self._fire_and_forget(self.on_agent_segment(segment.text))
            self._flush()  # speaker switch — flush any in-progress customer turn
            return

        # customer segment
        self._buffer.append(segment.text)
        self._reset_flush_timer()

    def _reset_flush_timer(self):
        if self._flush_task is not None:
            self._flush_task.cancel()
        self._flush_task = asyncio.create_task(self._flush_after_silence())

    async def _flush_after_silence(self):
        try:
            await asyncio.sleep(self.SILENCE_FLUSH_S)
        except asyncio.CancelledError:
            return
        self._flush()

    def _flush(self):
        if not self._buffer:
            return
        turn_text = " ".join(self._buffer).strip()
        self._buffer = []
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None
        if turn_text:
            self._fire_and_forget(self.on_customer_turn(turn_text))
