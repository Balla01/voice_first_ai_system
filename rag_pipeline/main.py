"""
Main pipeline — parallel 3-collection RAG with recency-weighted re-ranking.

Per query:
  1. Embed query once
  2. Search 3 collections in parallel (ThreadPoolExecutor):
       a. runtime_history  — filtered by session_id + customer_id
       b. session_summaries — filtered by session_id + customer_id
       c. insurance_docs   — no filter (general knowledge)
  3. Re-rank history + summary results with recency formula:
       finalScore = 0.7 × similarity + 0.3 × exp(-ageHours / 168)
  4. Build combined context string
  5. Call Groq LLM (streaming)
  6. Store user + assistant turns in history (eviction auto-triggers)

After all queries:
  7. End session → summarize remaining history → save to summary DB
"""

import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from history.history_pipeline import RuntimeHistory, _embed
from constants import GROQ_MODEL, DOCS_SEARCH_K

load_dotenv()

# ── Session identifiers ───────────────────────────────────────────────────────

SESSION_ID  = "session_main_001"
CUSTOMER_ID = "customer_main_001"

# ── Test queries ──────────────────────────────────────────────────────────────

QUERIES = [
    "What is the premium for a term life insurance policy?",
    "How does the claim process work after an accident?",
    "What is the waiting period for pre-existing medical conditions?",
    "Can I add riders to my current policy?",
    "What happens if I miss a premium payment?",
    "Is mental health treatment covered under my health plan?",
    "How do I update my nominee details?",
    "What is the grace period for premium payments?",
    "Are maternity benefits included in the health plan?",
    "How do I cancel my policy and get a refund?",
]

# ── Re-ranking ────────────────────────────────────────────────────────────────

def _recency_score(timestamp_iso: str) -> float:
    """exp(-ageHours / 168). Returns 1.0 when no timestamp (docs have none)."""
    if not timestamp_iso:
        return 1.0
    try:
        ts = datetime.fromisoformat(timestamp_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        age_hours = (now - ts).total_seconds() / 3600.0
        return math.exp(-age_hours / 168.0)
    except Exception:
        return 1.0


def rerank(results: list) -> list:
    """
    results: list of (text, similarity_score, timestamp_iso)
    Returns sorted list of (text, final_score) highest first.
    """
    scored = []
    for text, sim, ts in results:
        rec = _recency_score(ts)
        final = 0.7 * sim + 0.3 * rec
        scored.append((text, final))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ── Parallel search ───────────────────────────────────────────────────────────

def parallel_search(query: str, query_vec: list, history: RuntimeHistory, k: int = 4,
                    docs_k: int = DOCS_SEARCH_K, doc_filter=None, timing: dict = None):
    """
    Fire all 3 searches concurrently.
    docs_k is decoupled from k (history/summary breadth) — defaults to
    DOCS_SEARCH_K (constants.py) since the docs collection benefits from a
    wider candidate pool.
    query_vec (gte-large, pre-embedded) serves history/summary; docs is
    embedded separately inside search_docs_scored (BGE-M3, hybrid dense+sparse)
    since it's a different model — see history_pipeline.py.
    doc_filter (Qdrant Filter or None): explicit metadata filter for docs; when
    None, search_docs_scored auto-derives one from the query (USE_QUERY_FILTER).
    Docs are also cross-encoder reranked inside search_docs_scored (USE_RERANKER).
    Returns (history_ranked, summary_ranked, docs_raw).

    timing (optional): if a dict is passed, populated in place with
    history_retrieval_ms / summary_retrieval_ms / docs_retrieval_ms (each is
    wall-clock elapsed from pool submission to that branch's own completion —
    valid even though the 3 branches run concurrently, since each is measured
    against its own future, not the others') plus the docs sub-stage keys
    (embed_ms/filter_ms/qdrant_ms/merge_ms/rerank_ms) from search_docs_scored.
    """
    t_pool_start = time.perf_counter()
    docs_timing = {} if timing is not None else None
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_hist = pool.submit(history.search_history_scored,  query_vec, k)
        fut_summ = pool.submit(history.search_summary_scored,  query_vec, k)
        fut_docs = pool.submit(history.search_docs_scored,     query, docs_k, doc_filter, timing=docs_timing)

        history_results = fut_hist.result()
        if timing is not None:
            timing["history_retrieval_ms"] = (time.perf_counter() - t_pool_start) * 1000
        summary_results = fut_summ.result()
        if timing is not None:
            timing["summary_retrieval_ms"] = (time.perf_counter() - t_pool_start) * 1000
        docs_results = fut_docs.result()
        if timing is not None:
            timing["docs_retrieval_ms"] = (time.perf_counter() - t_pool_start) * 1000
            timing.update(docs_timing)

    history_ranked = rerank(history_results)
    summary_ranked = rerank(summary_results)
    # Docs have no timestamps; sort by raw similarity (already descending from Qdrant)
    docs_ranked = [(text, score) for text, score, _ in docs_results]

    return history_ranked, summary_ranked, docs_ranked

# ── Context builder ───────────────────────────────────────────────────────────

def build_context(
    recent_turns: list,
    history_ranked,
    summary_ranked,
    docs_ranked,
    top_k: int = 3,
) -> str:
    parts = []

    if recent_turns:
        parts.append("--- Last conversation turns (chronological) ---")
        for turn in recent_turns:
            parts.append(f"  {turn}")

    if history_ranked:
        parts.append("--- Semantically relevant history ---")
        for text, _ in history_ranked[:top_k]:
            parts.append(f"  • {text}")

    if summary_ranked:
        parts.append("--- Past session summaries ---")
        for text, _ in summary_ranked[:top_k]:
            parts.append(f"  • {text}")

    if docs_ranked:
        parts.append("--- Insurance document knowledge ---")
        for text, _ in docs_ranked[:top_k]:
            parts.append(f"  • {text[:300]}")

    return "\n".join(parts)

# ── LLM call ─────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = "You are a helpful insurance assistant. Answer concisely using the provided context."


def stream_llm_chunks(query: str, context: str, system_prompt: str = None):
    """
    Generator over raw content deltas from the Groq streaming call — the
    single place that owns the actual API call. call_llm() (below) and the
    FastAPI streaming endpoint (api.py) both consume this instead of each
    duplicating the Groq request.

    system_prompt: override the default instruction (e.g. api.py's advanced_filter
    mode, which needs the LLM to weigh multiple context sources). Defaults to
    DEFAULT_SYSTEM_PROMPT when None.
    """
    api_key = os.getenv("groq_api")
    if not api_key:
        yield "[LLM unavailable — groq_api not set in .env]"
        return

    user_content = query
    if context.strip():
        user_content = f"Context:\n{context}\n\nCustomer question: {query}"

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt or DEFAULT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        temperature=1,
        max_completion_tokens=256,
        top_p=1,
        stream=True,
    )

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def call_llm(query: str, context: str, system_prompt: str = None) -> str:
    parts = []
    for content in stream_llm_chunks(query, context, system_prompt):
        # errors="replace": a Windows console's cp1252 codepage can't encode
        # characters like the Rupee sign the LLM legitimately emits when
        # discussing premiums — this print is debug/CLI-visibility only, so a
        # console-display quirk must never crash the actual API call.
        try:
            print(content, end="", flush=True)
        except UnicodeEncodeError:
            print(content.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)
        parts.append(content)
    print()
    return "".join(parts)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    history = RuntimeHistory(session_id=SESSION_ID, customer_id=CUSTOMER_ID)

    for i, query in enumerate(QUERIES, start=1):
        print(f"\n{'─'*65}")
        print(f"Query {i:02d}/{len(QUERIES)} | session={SESSION_ID} | customer={CUSTOMER_ID}")
        print(f"User: {query}")

        t_start = time.perf_counter()

        # 1. Embed query once, reuse across all 3 searches
        query_vec = _embed([query])[0]

        # 2. Always fetch last 5 turns by recency (no embedding needed)
        recent_turns = history.get_recent_history(n=5)

        # 3. Parallel semantic search + re-rank
        history_ranked, summary_ranked, docs_ranked = parallel_search(query, query_vec, history)

        # 4. Build combined context
        context = build_context(recent_turns, history_ranked, summary_ranked, docs_ranked)

        t_retrieval = time.perf_counter()

        # 5. LLM call
        print("Assistant: ", end="")
        response = call_llm(query, context)

        t_end = time.perf_counter()

        print(f"  [retrieval: {t_retrieval - t_start:.2f}s | llm: {t_end - t_retrieval:.2f}s | total: {t_end - t_start:.2f}s]")

        # 6. Store both turns in history (eviction auto-triggers if cap hit)
        history.add("user",      query)
        history.add("assistant", response)

    # 6. End session → summarize remaining → save to summary DB
    print(f"\n{'─'*65}")
    print("Ending session...")
    history.end_session()
    history.close()


if __name__ == "__main__":
    main()
