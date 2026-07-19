"""
layer4/pregate.py — cheap, deterministic pre-filter that runs BEFORE the LLM
router, so we don't spend a full LLM round-trip on turns that plainly aren't
worth answering.

Backchannel skip: acknowledgements / filler that should never trigger a RAG
lookup. Deliberately Hinglish-aware, not English-only — real Indian sales-call
speech is full of "haan", "theek hai", "achha", "sahi hai" etc.; an
English-only list under-filters badly on this traffic.

A turn is treated as a backchannel only when it is SHORT and made up entirely
of backchannel/filler tokens — so "haan theek hai" is skipped but "haan, but
what does it cover?" is not (it carries a real question past the filler).
"""

import re
import logging

logger = logging.getLogger("insureassist.layer4")

# Multi-word phrases checked as whole-utterance matches first.
BACKCHANNEL_PHRASES = {
    "theek hai", "thik hai", "sahi hai", "achha theek hai", "haan theek hai",
    "ok thik hai", "let me think", "one sec", "one second", "hold on",
    "got it", "makes sense", "fair enough", "go on", "carry on",
}

# Single-token filler. If EVERY token of a short turn is in here, it's a backchannel.
BACKCHANNEL_WORDS = {
    # English
    "yeah", "yea", "yes", "yep", "yup", "ok", "okay", "k", "kk", "right",
    "sure", "hmm", "hm", "mhm", "mmhmm", "uh", "uhh", "um", "umm", "oh",
    "cool", "nice", "alright", "fine", "true", "exactly", "correct", "great",
    # Hinglish / Hindi-romanized
    "haan", "haa", "han", "ha", "ji", "hnn", "achha", "accha", "acha",
    "theek", "thik", "sahi", "bilkul", "bas", "arre", "arey", "chalo",
    "matlab", "waise", "toh", "na", "naa",
}

# Strip leading/trailing punctuation and collapse whitespace for tokenizing.
_TOKEN_RE = re.compile(r"[a-z]+", re.IGNORECASE)
MAX_BACKCHANNEL_WORDS = 4


def is_backchannel(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return True  # empty -> nothing to answer

    normalized = re.sub(r"[^\w\s]", "", stripped).strip()
    normalized = re.sub(r"\s+", " ", normalized)

    if normalized in BACKCHANNEL_PHRASES:
        logger.debug(f"Pre-gate: backchannel phrase match {normalized!r} -> skip")
        return True

    tokens = _TOKEN_RE.findall(stripped)
    if not tokens:
        # e.g. "..." or "!!" — no real words
        logger.debug(f"Pre-gate: no word tokens in {text!r} -> skip")
        return True

    if len(tokens) <= MAX_BACKCHANNEL_WORDS and all(t in BACKCHANNEL_WORDS for t in tokens):
        logger.debug(f"Pre-gate: all-filler short turn {tokens} -> skip")
        return True

    return False
