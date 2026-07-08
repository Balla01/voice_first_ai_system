"""
layer4/trigger_gate.py — Smart Trigger Gate orchestrator.

Gate order (see design doc Section 3.1):
  speaker filter (is_final only)
    -> refinement check (agent only, takes precedence)
    -> cooldown check
    -> Tier 1 regex (deterministic, multi-match)
    -> Tier 2 embedding (only if Tier 1 found nothing)
    -> confidence gate (only meaningful for Tier 2's genuine score)
    -> Tier 3 heuristic (only if Tiers 1 & 2 both found nothing)

This is the only module main.py needs to import from layer4 for the
decision-making part (GenerationController is separate, used alongside it
once Layer 5 exists).
"""

import logging
from typing import Optional

from .models import TriggerResult, TriggerAction, IntentMatch
from .cooldown import CooldownTracker
from .refinement import is_refinement_command
from .intent_tier1_regex import classify_intent
from .intent_tier2_embedding import Tier2EmbeddingClassifier
from .intent_tier3_heuristic import Tier3Heuristic

logger = logging.getLogger("insureassist.layer4")

CONFIDENCE_THRESHOLD = 0.5


class TriggerGate:
    def __init__(self, session_id: str, tier2_classifier: Optional[Tier2EmbeddingClassifier] = None):
        """
        tier2_classifier: pass a shared instance across sessions if you want
        to avoid loading the MiniLM model once per session (it's not
        session-specific state, just the classifier logic + reference
        embeddings). Defaults to creating its own if not provided.
        """
        self.session_id = session_id
        self._cooldown = CooldownTracker()
        self._tier3 = Tier3Heuristic()
        self._tier2 = tier2_classifier if tier2_classifier is not None else Tier2EmbeddingClassifier()

    def check(self, speaker: str, text: str, is_final: bool, now: float) -> TriggerResult:
        logger.debug(f"[{self.session_id}] ---- TriggerGate.check speaker={speaker!r} text={text!r} ----")

        if not is_final:
            logger.debug(f"[{self.session_id}] Speaker filter: turn is not final -> NO_TRIGGER")
            return TriggerResult(action=TriggerAction.NO_TRIGGER, reason="not final")

        if is_refinement_command(speaker, text):
            logger.info(f"[{self.session_id}] TriggerGate decision: REFINE")
            return TriggerResult(action=TriggerAction.REFINE, reason="agent issued a refinement command")

        if self._cooldown.is_in_cooldown(now):
            logger.debug(f"[{self.session_id}] TriggerGate decision: NO_TRIGGER (cooldown active)")
            return TriggerResult(action=TriggerAction.NO_TRIGGER, reason="cooldown active")

        # Tier 1 — regex registry, deterministic, multi-match
        matches = classify_intent(text)
        if matches:
            self._fire(now)
            logger.info(
                f"[{self.session_id}] TriggerGate decision: FIRE via Tier1 "
                f"intents={[m.intent for m in matches]}"
            )
            return TriggerResult(action=TriggerAction.FIRE, matches=matches, reason="Tier1 regex match")

        # Tier 2 — local embedding classifier, only since Tier 1 found nothing
        tier2_match = self._tier2.classify(text)
        if tier2_match is not None:
            if tier2_match.confidence >= CONFIDENCE_THRESHOLD:
                self._fire(now)
                logger.info(
                    f"[{self.session_id}] TriggerGate decision: FIRE via Tier2 "
                    f"intent={tier2_match.intent} confidence={tier2_match.confidence:.3f}"
                )
                return TriggerResult(action=TriggerAction.FIRE, matches=[tier2_match], reason="Tier2 embedding match")
            else:
                logger.debug(
                    f"[{self.session_id}] Confidence gate: Tier2 confidence "
                    f"{tier2_match.confidence:.3f} < {CONFIDENCE_THRESHOLD} -> drop silently"
                )

        # Tier 3 — context-state heuristic, only since Tiers 1 & 2 both found nothing confident
        if self._tier3.is_followup(text):
            self._fire(now)
            match = IntentMatch(
                intent="follow_up", rag_collections=[], response_tone="conversational, contextual",
                priority=2, confidence=0.6,
            )
            logger.info(f"[{self.session_id}] TriggerGate decision: FIRE via Tier3 heuristic (follow-up)")
            return TriggerResult(action=TriggerAction.FIRE, matches=[match], reason="Tier3 context heuristic")

        logger.debug(f"[{self.session_id}] TriggerGate decision: NO_TRIGGER (no tier matched)")
        return TriggerResult(action=TriggerAction.NO_TRIGGER, reason="no tier matched")

    def _fire(self, now: float) -> None:
        self._cooldown.record_trigger(now)
        self._tier3.record_answer()