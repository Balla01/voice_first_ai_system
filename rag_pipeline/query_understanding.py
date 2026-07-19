"""
LLM-driven query -> metadata filter for docs retrieval (feature c).

Given a user query, a fast LLM (Groq) extracts the plan_name / doc_type /
product_type mentioned in it (e.g. "surrender value for Jeevan Anand" ->
plan_name="Jeevan Anand"). Those are validated against the values actually
present in the collection and turned into a Qdrant Filter, so a query about
one plan doesn't retrieve chunks from every other plan.

Safety (an LLM will happily invent a plan name):
  - plan_name is accepted ONLY if it matches a plan actually in the DB
    (case-insensitive, exact-or-substring), so a hallucinated name can't zero
    out the results;
  - doc_type / product_type are constrained to known vocab;
  - if nothing validates, the filter is None (search everything);
  - callers additionally fall back to unfiltered search if a filter yields 0 hits
    (see history_pipeline.search_docs_scored).

Groq is used deliberately (fast, cloud) — NOT the CPU Qwen metadata model, which
would add ~50s to every query.
"""
import json
import os
import re
from typing import Dict, List, Optional

from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from constants import (
    QUERY_FILTER_MODEL, QDRANT_COLLECTION,
    KNOWN_DOC_TYPES, KNOWN_PRODUCT_TYPES,
)

_SYSTEM = """You extract retrieval filters from an insurance customer's question.
Return ONLY a JSON object, no prose, no markdown fences, with exactly these keys:
{"plan_name": string|null, "doc_type": string|null, "product_type": string|null}

Rules:
- plan_name: the specific insurance/pension plan named in the question (e.g. "Jeevan Anand",
  "New Pension Plus", "Single Premium Endowment"). null if no specific plan is named.
- doc_type: one of "Sales Brochure", "Customer Information Sheet", "Policy Document" if the
  question clearly refers to that document kind; else null.
- product_type: "pension" if the question is about a pension/annuity plan, "insurance" if it is
  clearly about a life-insurance plan; else null.
Only use what is explicitly in the question. When unsure, use null."""


# ── Known field values (cached) ───────────────────────────────────────────────

def known_field_values(client: QdrantClient, collection: str, field: str) -> List[str]:
    """Distinct non-empty values of `field` across the collection, scrolled once.
    Not cached itself — callers with a stable (collection, field) pair (below)
    cache the result; ad-hoc callers can call this directly."""
    values = set()
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=collection, limit=500, offset=offset,
                with_vectors=False, with_payload=[field],
            )
            for p in points:
                v = (p.payload or {}).get(field)
                if v:
                    values.add(v)
            if offset is None:
                break
    except Exception:
        pass
    return sorted(values)


_known_plans_cache: Optional[List[str]] = None
_known_categories_cache: Optional[List[str]] = None


def known_plan_names(client: QdrantClient, collection: str = QDRANT_COLLECTION) -> List[str]:
    """Distinct plan_name payload values in the collection, scrolled once and cached."""
    global _known_plans_cache
    if _known_plans_cache is None:
        _known_plans_cache = known_field_values(client, collection, "plan_name")
    return _known_plans_cache


def known_categories(client: QdrantClient, collection: str = QDRANT_COLLECTION) -> List[str]:
    """Distinct category payload values (only populated in COMPLETE_FOLDER_STRUCTURE
    ingestion mode — see constants.py; returns [] otherwise)."""
    global _known_categories_cache
    if _known_categories_cache is None:
        _known_categories_cache = known_field_values(client, collection, "category")
    return _known_categories_cache


# ── Query -> raw extracted fields (Groq) ──────────────────────────────────────

def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def extract_query_filter(query: str) -> Dict[str, Optional[str]]:
    """Ask Groq for {plan_name, doc_type, product_type}; returns {} on any failure."""
    api_key = os.getenv("groq_api")
    if not api_key:
        return {}
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=QUERY_FILTER_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_completion_tokens=128,
            top_p=1,
            stream=False,
        )
        return _parse_json(resp.choices[0].message.content or "")
    except Exception:
        return {}


# ── Validate + build Qdrant Filter ────────────────────────────────────────────

def _match_known_plan(candidate: str, known: List[str]) -> Optional[str]:
    """Return the stored plan_name matching `candidate` (case-insensitive, exact or
    substring either way), or None. Prevents a hallucinated name from filtering to nothing."""
    if not candidate:
        return None
    c = candidate.strip().lower()
    for k in known:
        if k.lower() == c:
            return k
    for k in known:
        kl = k.lower()
        if c in kl or kl in c:
            return k
    return None


def _match_vocab(candidate: Optional[str], vocab: List[str]) -> Optional[str]:
    if not candidate:
        return None
    c = candidate.strip().lower()
    for v in vocab:
        if v.lower() == c:
            return v
    return None


def build_docs_filter(extracted: Dict[str, Optional[str]], known_plans: List[str]):
    """
    Turn validated extracted fields into a Qdrant Filter.
    Returns (filter_or_None, applied_dict) — applied_dict is the validated
    fields that made it into the filter (for logging / eval visibility).
    """
    must = []
    applied: Dict[str, str] = {}

    plan = _match_known_plan(extracted.get("plan_name") or "", known_plans)
    if plan:
        must.append(FieldCondition(key="plan_name", match=MatchValue(value=plan)))
        applied["plan_name"] = plan

    doc_type = _match_vocab(extracted.get("doc_type"), KNOWN_DOC_TYPES)
    if doc_type and doc_type != "Other":
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
        applied["doc_type"] = doc_type

    product_type = _match_vocab(extracted.get("product_type"), KNOWN_PRODUCT_TYPES)
    if product_type:
        must.append(FieldCondition(key="product_type", match=MatchValue(value=product_type)))
        applied["product_type"] = product_type

    if not must:
        return None, {}
    return Filter(must=must), applied


def auto_docs_filter(query: str, known_plans: List[str]):
    """
    Full query -> filter path: Groq-extract fields, validate, build a Qdrant Filter.
    Returns (filter_or_None, description_str) — description for logs / eval CSV.
    """
    extracted = extract_query_filter(query)
    doc_filter, applied = build_docs_filter(extracted, known_plans)
    desc = ", ".join(f"{k}={v}" for k, v in applied.items()) if applied else "none"
    return doc_filter, desc


# ── Web-search trigger (advanced_filter mode, api.py) ─────────────────────────

_WEB_SEARCH_SYSTEM = """Decide if the customer's message needs a live web search rather than the internal LIC insurance/pension knowledge base.

Answer "yes" (needs web search) when the question:
- is NOT about LIC insurance or pension plans, premiums, claims, coverage, exclusions, riders, or policy documents, OR
- asks for current/general-knowledge information (news, rates, dates, external company info, unrelated topics), OR
- is too vague/incomplete/partial to search the insurance knowledge base meaningfully (no clear insurance intent).

Answer "no" when the question is clearly about LIC insurance/pension products, premiums, claims, coverage, or policy details.

Reply with exactly one word: yes or no."""


def classify_web_search(query: str) -> bool:
    """True if `query` should be answered via a live web search instead of (or in
    addition to) the insurance knowledge base. Defaults to False (stay in-domain)
    on any failure, since the RAG pipeline is the safe default."""
    api_key = os.getenv("groq_api")
    if not api_key:
        return False
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=QUERY_FILTER_MODEL,
            messages=[
                {"role": "system", "content": _WEB_SEARCH_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_completion_tokens=4,
            top_p=1,
            stream=False,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False
