"""
layer3/models.py — shared data structures for the Session Tracker.

Kept dependency-free (no tiktoken, no asyncpg, no anthropic imports here) so
every other module can import these without pulling in unrelated dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    """One final transcript segment, as it lives inside Layer 3's context window."""
    session_id: str
    speaker: str          # "agent" | "customer"
    text: str
    timestamp: float       # unix epoch seconds, when the words were spoken (from Layer 1/2)
    is_important: bool = False   # flipped True only if Layer 4 fires a trigger on this turn
    db_id: Optional[int] = None  # set once persisted; used for UPDATE (importance flip)


@dataclass
class EpochSummary:
    """A compacted summary standing in for a block of older turns."""
    session_id: str
    text: str
    covers_turn_count: int
    epoch_index: int = 0
    db_id: Optional[int] = None
    superseded: bool = False   # set True once merged into a meta-summary