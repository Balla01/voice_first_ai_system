"""
layer3/session_tracker.py — orchestrator wiring the whole Layer 3 together.

This is the only module the rest of the app (main.py, and eventually Layer 4)
needs to import. Everything else in this package is a component it composes:

    Deduplicator     -> is this turn a repeat?
    ContextWindow     -> in-memory rolling turns + epoch summaries
    EpochCompactor    -> async Claude calls to summarize
    Persistence       -> Postgres read/write

Compaction runs as fire-and-forget background asyncio tasks so a slow Claude
call never blocks new turns from being processed — matching the "async,
non-blocking" requirement from the design doc.
"""

import asyncio
from typing import Optional

from .models import Turn, EpochSummary
from .dedup import Deduplicator
from .context_window import ContextWindow, TURNS_PER_EPOCH
from .epoch_compaction import EpochCompactor
from .persistence import Persistence

MAX_EPOCH_SUMMARIES = 10
SUMMARIES_MERGED_PER_COMPACTION = 2


class SessionTracker:
    def __init__(self, session_id: str, persistence: Persistence, compactor: EpochCompactor):
        self.session_id = session_id
        self._persistence = persistence
        self._compactor = compactor

        self._dedup = Deduplicator()
        self._window = ContextWindow()
        self._epoch_counter = 0

        # Guards against overlapping compaction tasks stepping on each other
        # (e.g. a burst of turns each independently deciding "we're over budget").
        self._compaction_lock = asyncio.Lock()
        self._meta_compaction_lock = asyncio.Lock()

    # ---- lifecycle ----

    async def load_history(self) -> None:
        """Reconnect support: rebuild in-memory state from Postgres."""
        turns, summaries = await self._persistence.load_session_history(self.session_id)
        for t in turns:
            self._window.add_turn(t)
            self._dedup.seed(t)
        for s in summaries:
            self._window.add_epoch_summary(s)
        if summaries:
            self._epoch_counter = max(s.epoch_index for s in summaries) + 1

    # ---- writes ----

    async def add_turn(self, speaker: str, text: str, timestamp: float) -> Optional[Turn]:
        """
        Returns the Turn if it was added, or None if it was dropped as a duplicate.
        Only call this with FINAL transcript segments (is_final=True upstream).
        """
        turn = Turn(session_id=self.session_id, speaker=speaker, text=text, timestamp=timestamp)

        if self._dedup.is_duplicate(turn):
            return None

        self._window.add_turn(turn)
        turn.db_id = await self._persistence.insert_turn(turn)

        if self._window.is_over_budget():
            asyncio.create_task(self._run_epoch_compaction())

        return turn

    async def mark_important(self, turn: Turn) -> None:
        """Called by Layer 4 when it fires a trigger on this turn."""
        self._window.mark_turn_important(turn)
        await self._persistence.update_turn_importance(turn)

    # ---- reads ----

    def get_formatted_context(self) -> str:
        return self._window.format_context()

    # ---- compaction internals ----

    async def _run_epoch_compaction(self) -> None:
        async with self._compaction_lock:
            oldest = self._window.pop_oldest_turns(TURNS_PER_EPOCH)
            if not oldest:
                return

            summary_text = await self._compactor.compact_turns(oldest)
            summary = EpochSummary(
                session_id=self.session_id,
                text=summary_text,
                covers_turn_count=len(oldest),
                epoch_index=self._epoch_counter,
            )
            self._epoch_counter += 1

            self._window.add_epoch_summary(summary)
            summary.db_id = await self._persistence.insert_epoch_summary(summary)

        if len(self._window.epoch_summaries) > MAX_EPOCH_SUMMARIES:
            asyncio.create_task(self._run_meta_compaction())

    async def _run_meta_compaction(self) -> None:
        async with self._meta_compaction_lock:
            oldest = self._window.pop_oldest_summaries(SUMMARIES_MERGED_PER_COMPACTION)
            if len(oldest) < SUMMARIES_MERGED_PER_COMPACTION:
                # Not enough to merge (shouldn't normally happen) — put back untouched.
                for s in reversed(oldest):
                    self._window.add_merged_summary_at_front(s)
                return

            merged_text = await self._compactor.compact_summaries(oldest)
            merged = EpochSummary(
                session_id=self.session_id,
                text=merged_text,
                covers_turn_count=sum(s.covers_turn_count for s in oldest),
                epoch_index=oldest[0].epoch_index,  # keep it sorted as the oldest entry
            )
            self._window.add_merged_summary_at_front(merged)
            merged.db_id = await self._persistence.insert_epoch_summary(merged)
            await self._persistence.mark_summaries_superseded(oldest)