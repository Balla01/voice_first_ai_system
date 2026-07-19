"""
profile_extractor.py — LLM-based customer-profile extraction for GET /profile (api.py).

Given all available context for a session — an earlier summary (if the
conversation ran long enough to evict) followed by the recent turns verbatim,
see api.py's _build_session_transcript — asks Groq to pull out structured
customer facts. Mirrors auido_capture/profile_extractor.py's approach
(LLM-only: name/profession/etc. phrasing is too open-ended for regex) but adds
the two insurance-specific fields the RAG side needs: policy_product and
category, both validated against values that actually exist in the docs
collection (same match-or-drop pattern as query_understanding.py's plan_name
filter) so a hallucinated plan/category is dropped rather than trusted.
"""

import logging
import os
from typing import Dict, List, Optional

from groq import Groq

from constants import GROQ_MODEL
from query_understanding import _parse_json, _match_known_plan as _match_known

logger = logging.getLogger("rag_api.profile")

PROFILE_FIELDS = ("name", "age", "profession", "location", "policy_product", "category")

_SYSTEM = """You extract a customer's profile from an insurance chatbot conversation.

You are given all available context for this session: an earlier summary (if the
conversation is long) followed by the recent turns verbatim. Return ONLY a JSON
object with exactly these keys:
  "name":            customer's full name if stated, else null
  "age":             integer age if stated, else null
  "profession":      job/occupation if stated, else null
  "location":        city/area/state if stated, else null
  "policy_product":  the specific LIC plan name the customer is discussing or
                     asking about, if one is clearly named, else null
  "category":        the plan category (e.g. "endowment-plans", "term_assurance_plans",
                     "pension") if it can be inferred, else null

Rules:
- Only include facts actually present in the conversation. Do not guess or
  invent a plan name/category that was not mentioned or clearly implied.
- If a field is not mentioned, use null.
- Output raw JSON only. No prose, no markdown fences."""


def _empty_profile() -> Dict[str, Optional[str]]:
    return {f: None for f in PROFILE_FIELDS}


def extract_profile(transcript: str, known_plans: List[str], known_categories: List[str]) -> Dict[str, Optional[str]]:
    """
    Best-effort profile extraction via Groq. Returns an all-null profile on any
    failure (no API key, LLM error, unparseable reply) — profile building must
    never break the caller.

    known_plans / known_categories: policy_product/category are dropped (set
    to None) unless they match one of these (case-insensitive, exact-or-substring)
    — same defensive pattern as query_understanding.auto_docs_filter, since an
    LLM will happily invent a plausible-looking plan name.
    """
    if not transcript.strip():
        return _empty_profile()

    api_key = os.getenv("groq_api")
    if not api_key:
        return _empty_profile()

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Conversation:\n{transcript[-6000:]}"},
            ],
            temperature=0,
            max_completion_tokens=200,
            top_p=1,
            stream=False,
        )
        data = _parse_json(resp.choices[0].message.content or "")
    except Exception as e:
        logger.warning(f"profile extraction failed: {e}")
        return _empty_profile()

    if not isinstance(data, dict):
        return _empty_profile()

    profile = {f: data.get(f) for f in PROFILE_FIELDS}
    profile["policy_product"] = _match_known(profile.get("policy_product") or "", known_plans)
    profile["category"] = _match_known(profile.get("category") or "", known_categories)
    return profile
