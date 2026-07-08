"""
layer3/context_window.py — Step 2: Context Window (token-budgeted deque).

Pure in-memory data structure: owns the rolling turns and epoch summaries,
counts tokens, formats the string handed to Layer 4 / Layer 5, and exposes
primitives for evicting the oldest turns/summaries. It does NOT call Claude
and does NOT talk to Postgres — those live in epoch_compaction.py and
persistence.py respectively, and session_tracker.py wires everything
together. Keeping this file dependency-free makes it trivial to unit test.
"""

from collections import deque
from typing import Deque, List

from .models import Turn, EpochSummary
from .tokens import count_tokens, AVAILABLE_FOR_CONTEXT
from .dedup import Deduplicator


TURNS_PER_EPOCH = 500  # how many oldest turns get compacted into one summary


class ContextWindow:
    def __init__(self):
        self.turns: Deque[Turn] = deque()
        self.epoch_summaries: Deque[EpochSummary] = deque()

    # ---- writes ----

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def pop_oldest_turns(self, n: int = TURNS_PER_EPOCH) -> List[Turn]:
        """Remove and return up to n oldest turns, for compaction."""
        n = min(n, len(self.turns))
        return [self.turns.popleft() for _ in range(n)]

    def add_epoch_summary(self, summary: EpochSummary) -> None:
        self.epoch_summaries.append(summary)

    def pop_oldest_summaries(self, n: int) -> List[EpochSummary]:
        n = min(n, len(self.epoch_summaries))
        return [self.epoch_summaries.popleft() for _ in range(n)]

    def add_merged_summary_at_front(self, summary: EpochSummary) -> None:
        """Meta-compaction result goes back at the oldest position."""
        self.epoch_summaries.appendleft(summary)

    def mark_turn_important(self, turn: Turn) -> None:
        # turn is the same object reference already sitting in self.turns,
        # so mutating it here is visible everywhere else it's referenced.
        turn.is_important = True

    # ---- reads ----

    def total_tokens(self) -> int:
        turn_tokens = sum(count_tokens(t.text) for t in self.turns)
        summary_tokens = sum(count_tokens(s.text) for s in self.epoch_summaries)
        return turn_tokens + summary_tokens

    def is_over_budget(self) -> bool:
        return self.total_tokens() > AVAILABLE_FOR_CONTEXT

    def format_context(self) -> str:
        lines = [f"[SESSION HISTORY]: {s.text}" for s in self.epoch_summaries]
        for t in self.turns:
            tag = t.speaker.upper() + (" \u2014 IMPORTANT" if t.is_important else "")
            lines.append(f"[{tag}]: {t.text}")
        return "\n".join(lines)