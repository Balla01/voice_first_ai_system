"""
Deterministic (regex-only, no LLM) detection of ambiguous references in an
Ask-AI query — e.g. "correct this suggestion", "what's the answer" — where
it's unclear whether "this/that X" means the AI Copilot's live suggestion card
(session history + summaries) or the current Ask-AI thread's own conversation
(chat_bot_ask_ai). Used only when advanced_filter=True and the caller hasn't
already resolved it via context_source (see api.py's /query).

Regex over LLM classification here for the same reason as email_trigger.py:
this is a keyword-shaped signal, not a semantic judgment call — a regex is
free, instant, and can't misfire in a way that costs an extra Groq round-trip.

NOTE: the referent noun alone ("point", "thread", "card", ...) is NOT enough —
it has to be preceded by a determiner (this/that/these/those/the) that's
actually pointing at something. A bare noun anywhere in the sentence produced
too many false positives in practice: "summarize this call in 3 bullet
points", "is this new thread?", and similar ordinary phrasing were all being
flagged as needing clarification even though nothing was actually ambiguous.
"""
import re

_REFERENT_RE = re.compile(
    r"\b(this|that|these|those|the)\s+(suggestion|conversation|point|thread|answer|response|card)s?\b",
    re.IGNORECASE,
)


def is_ambiguous_reference(query: str) -> bool:
    """True if `query` contains a determiner + referent noun ("this suggestion",
    "the answer", ...) that could mean either the live suggestion card (session
    history + summaries) or the current Ask-AI thread (chat_bot_ask_ai) — the
    caller should ask the user which one before answering, rather than
    guessing."""
    return bool(_REFERENT_RE.search(query))
