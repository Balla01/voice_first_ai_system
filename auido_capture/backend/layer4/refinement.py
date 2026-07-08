"""
layer4/refinement.py — refinement detection (Section 3.8 of the design doc).

Checked before the normal trigger path. If the AGENT (not the customer) says
something like "make it shorter", "rephrase", or "add an example", this
isn't a new question — it's an instruction to edit the last answer Layer 5
already gave, in place.
"""

import logging

logger = logging.getLogger("insureassist.layer4")

REFINEMENT_PHRASES = ["make it shorter", "rephrase", "add example", "add an example"]


def is_refinement_command(speaker: str, text: str) -> bool:
    if speaker != "agent":
        return False

    lowered = text.lower()
    matched = [p for p in REFINEMENT_PHRASES if p in lowered]

    if matched:
        logger.debug(f"Refinement: agent text matched phrase(s) {matched} -> REFINE")
        return True

    logger.debug("Refinement: agent text did not match any refinement phrase")
    return False