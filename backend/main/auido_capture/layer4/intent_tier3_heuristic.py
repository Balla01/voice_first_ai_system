"""
layer4/intent_tier3_heuristic.py — Tier 3: context-state heuristic.

If the assistant has already answered twice in this session and a short new
customer utterance comes in (e.g. "and for kids?"), treat it as a follow-up
continuation even without a keyword match. 0ms, pure Python, stateful
per-session.

"Assistant answered" is approximated here as "TriggerGate fired a trigger" —
Layer 5 (actual answer generation) doesn't exist yet, so this is the closest
available signal. Revisit once Layer 5 can confirm an answer was actually
delivered.
"""

import logging

logger = logging.getLogger("insureassist.layer4")

FOLLOWUP_MAX_WORDS = 6
ANSWER_COUNT_THRESHOLD = 2


class Tier3Heuristic:
    def __init__(self):
        self.assistant_answer_count = 0

    def record_answer(self) -> None:
        self.assistant_answer_count += 1
        logger.debug(f"Tier3: assistant_answer_count now {self.assistant_answer_count}")

    def is_followup(self, text: str) -> bool:
        word_count = len(text.split())
        result = (
            self.assistant_answer_count >= ANSWER_COUNT_THRESHOLD
            and word_count <= FOLLOWUP_MAX_WORDS
        )
        logger.debug(
            f"Tier3 heuristic: answer_count={self.assistant_answer_count} "
            f"word_count={word_count} -> {'FOLLOWUP' if result else 'no match'}"
        )
        return result