"""
layer4/trigger_gate.py — Smart Trigger Gate orchestrator.

Gate order:
  speaker filter (is_final only)
    -> refinement check (agent only, takes precedence)
    -> cooldown check
    -> Tier 3 heuristic (context-state follow-up detection)

Tier 1 (regex registry) and Tier 2 (MiniLM embedding classifier) were removed:
this gate now only matters as ToolRouter's last-resort fallback (after both
the primary and fallback LLM routers have failed — see tool_router.py), a
case rare enough that a full keyword/embedding classifier wasn't earning its
keep, and Tier 2 also cost real startup time/memory (loading all-MiniLM-L6-v2)
on every boot whether or not it was ever exercised. Tier 3 stays: it's pure
Python, no model load, no dependency.

This is the only module main.py needs to import from layer4 for the
decision-making part (GenerationController is separate, used alongside it
once Layer 5 exists).
"""

import logging

from .models import TriggerResult, TriggerAction, IntentMatch
from .cooldown import CooldownTracker
from .refinement import is_refinement_command
from .intent_tier3_heuristic import Tier3Heuristic

logger = logging.getLogger("insureassist.layer4")


class TriggerGate:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._cooldown = CooldownTracker()
        self._tier3 = Tier3Heuristic()

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

        # Tier 3 — context-state heuristic, the only remaining tier
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