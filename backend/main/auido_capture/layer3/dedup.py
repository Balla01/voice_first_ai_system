"""
layer3/dedup.py — Step 1: Deduplication (500ms window).

If the same speaker says the exact same text again within 500ms of their
previous turn, it's dropped. Guards against duplicate finals from STT
reconnects or double-delivery. Pure logic, no I/O — easy to unit test.
"""

from typing import Dict, Optional
from .models import Turn


class Deduplicator:
    DEDUP_WINDOW_S = 0.5

    def __init__(self):
        self._last_turn_by_speaker: Dict[str, Turn] = {}

    def is_duplicate(self, turn: Turn) -> bool:
        prior = self._last_turn_by_speaker.get(turn.speaker)
        if (
            prior is not None
            and (turn.timestamp - prior.timestamp) <= self.DEDUP_WINDOW_S
            and turn.text.strip() == prior.text.strip()
        ):
            return True
        self._last_turn_by_speaker[turn.speaker] = turn
        return False

    def seed(self, turn: Turn) -> None:
        """Rehydrate state from persisted history (e.g. on reconnect) without
        running the actual duplicate check — just primes 'last turn per speaker'."""
        self._last_turn_by_speaker[turn.speaker] = turn