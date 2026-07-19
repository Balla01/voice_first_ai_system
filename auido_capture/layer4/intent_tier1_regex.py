"""
layer4/intent_tier1_regex.py — Tier 1: business-intent regex registry.

Real, demo-ready registry of 6 named business intents. ALL patterns are
checked and every intent that matches fires together — merged into one RAG
retrieval and one LLM call downstream, rather than picking just the first or
highest-priority match. 0ms, deterministic, no API call.

NOTE: patterns below are a representative subset (per intent) for build
speed — the design doc flags pulling the full 8-10 patterns per intent from
the original reference before the final demo (see Open Items).
"""

import re
import logging
from typing import List

from .models import IntentMatch

logger = logging.getLogger("insureassist.layer4")

TRIGGER_REGISTRY = {
    "policy_inquiry": {
        "patterns": [
            r"what (does|is) (this|the) plan cover",
            r"what(\'s| is) included",
            r"explain (the|this) policy",
            r"benefits? (of|in) (this|the) plan",
        ],
        "rag_collection": ["policy_docs", "plan_comparisons"],
        "response_tone": "informative, comprehensive",
        "priority": 2,
    },
    "objection": {
        "patterns": [
            r"too expensive",
            r"(better|cheaper) (option|plan|deal) (elsewhere|outside|online)",
            r"my (current|existing|old) plan (is|was) better",
            r"why (should|would) (i|we) (buy|take|choose) (this|your)",
        ],
        "rag_collection": ["plan_comparisons", "value_props", "competitor_notes"],
        "response_tone": "empathetic, persuasive, value-focused",
        "priority": 1,
    },
    "premium_concern": {
        "patterns": [
            r"(how much|what is the) premium",
            r"(monthly|annual|yearly) (cost|payment)",
            r"(reduce|lower|decrease) (the|my) premium",
            r"(tax|80d|deduction|exemption)",
        ],
        "rag_collection": ["premium_tables", "discount_rules", "tax_benefits"],
        "response_tone": "factual, number-specific, budget-aware",
        "priority": 2,
    },
    "claim_question": {
        "patterns": [
            r"(how|when|where) (do|can) (i|we) (file|raise|submit) a claim",
            r"cashless (treatment|facility|claim)",
            r"(network|empanelled) hospital",
            r"(claim|insurance) (rejected|denied|declined)",
        ],
        "rag_collection": ["claim_docs", "hospital_network", "tpa_process"],
        "response_tone": "process-oriented, step-by-step, reassuring",
        "priority": 2,
    },
    "exclusion_concern": {
        "patterns": [
            r"(what is|what\'s) not covered",
            r"exclusion(s)?",
            r"(dental|vision|cosmetic|plastic surgery)",
            r"(pre.?existing).{0,20}(not|never|won\'t)",
        ],
        "rag_collection": ["exclusion_lists", "policy_wordings", "faq_docs"],
        "response_tone": "honest, clear, offer alternatives",
        "priority": 2,
    },
    "buying_signal": {
        "patterns": [
            r"(how|where|when) (do|can) (i|we) (buy|purchase|apply)",
            r"(let\'s|ready to) (proceed|go ahead|buy)",
            r"(what|which) documents? (do i need|required)",
            r"(payment|pay|upi|card) (link|option|method)",
        ],
        "rag_collection": ["onboarding_docs", "document_checklist", "payment_process"],
        "response_tone": "action-oriented, fast, remove friction",
        "priority": 3,
    },
}


def classify_intent(text: str) -> List[IntentMatch]:
    matched = []
    for intent_name, config in TRIGGER_REGISTRY.items():
        for pattern in config["patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                matched.append(IntentMatch(
                    intent=intent_name,
                    rag_collections=config["rag_collection"],
                    response_tone=config["response_tone"],
                    priority=config["priority"],
                    confidence=1.0,  # deterministic regex match -> full confidence
                ))
                logger.debug(f"Tier1 regex: '{pattern}' matched intent={intent_name} (priority={config['priority']})")
                break  # one match per intent is enough, but ALL intents are checked

    matched.sort(key=lambda m: m.priority, reverse=True)

    if matched:
        logger.debug(f"Tier1 regex: {len(matched)} intent(s) matched -> {[m.intent for m in matched]}")
    else:
        logger.debug("Tier1 regex: no intents matched -> falling through to Tier 2")

    return matched