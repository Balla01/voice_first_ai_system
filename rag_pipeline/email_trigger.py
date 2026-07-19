"""
Deterministic (regex-only, no LLM) detection of "email me the answer to X"
requests inside a query — used only when advanced_filter=True (api.py).

Regex over LLM classification here specifically for cost/latency/hallucination
reasons: unlike classify_web_search (query_understanding.py), which needs
semantic judgment, "is there an email address plus an intent-to-send word in
this string" is exactly what a regex is for, at zero extra Groq calls.
"""
import re
from typing import Optional

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_TRIGGER_RE = re.compile(r"\b(e-?mail|mail|send)\b", re.IGNORECASE)


def detect_email_request(query: str) -> Optional[str]:
    """Return the first email address in `query` if the query also contains an
    email/mail/send trigger word, else None. Whole-query match (not a proximity
    window) — both signals present anywhere in one short query is already a
    strong, low-false-positive signal for this domain."""
    match = _EMAIL_RE.search(query)
    if not match:
        return None
    if not _TRIGGER_RE.search(query):
        return None
    return match.group(0)
