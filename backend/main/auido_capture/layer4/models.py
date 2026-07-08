"""
layer4/models.py — shared data structures for the Smart Trigger Gate.

Kept dependency-free, same reasoning as layer3/models.py: every other module
should be importable and unit-testable without pulling in unrelated deps.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TriggerAction(Enum):
    FIRE = "fire"                # a new RAG + LLM call should happen
    NO_TRIGGER = "no_trigger"    # drop silently, buffer keeps accumulating
    REFINE = "refine"            # edit the last answer in place, not a new call
    # NOTE: there is no separate ABORT action here. Aborting an in-flight
    # generation is a side effect of GenerationController.start_generation()
    # being called again on a FIRE — it always cancels whatever's still
    # running for that session first. The gate itself only ever decides
    # FIRE / NO_TRIGGER / REFINE.


@dataclass
class IntentMatch:
    intent: str
    rag_collections: List[str]
    response_tone: str
    priority: int
    confidence: float = 1.0   # Tier 1 matches are deterministic -> always 1.0


@dataclass
class TriggerResult:
    action: TriggerAction
    matches: List[IntentMatch] = field(default_factory=list)
    reason: str = ""   # short human-readable explanation, useful for logs/demo narration