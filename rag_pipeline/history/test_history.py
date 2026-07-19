"""
Test: 10 queries through RuntimeHistory pipeline.

Flow per query:
  1. Retrieve relevant context from history (filtered by session + customer)
  2. Call Groq LLM with context + query
  3. Store user query in history
  4. Store assistant response in history
  5. (Eviction triggers automatically if RAM cap is hit)

After all 10 queries:
  6. End session → summarize remaining history → save to summary DB
  7. Print summary DB contents
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from history.history_pipeline import RuntimeHistory
from constants import SUMMARY_COLLECTION, HISTORY_SUMMARY_DIR, GROQ_MODEL

load_dotenv()

# ── Session identifiers ───────────────────────────────────────────────────────

SESSION_ID  = "session_002"
CUSTOMER_ID = "customer_002"

# ── 10 test queries ───────────────────────────────────────────────────────────

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

# ── LLM call (follows llm_test.py pattern) ───────────────────────────────────

def call_llm(query: str, context: list) -> str:
    """Call Groq with retrieved context + current query. Returns full response."""
    api_key = os.getenv("groq_api")
    if not api_key:
        return "[LLM unavailable — groq_api not set in .env]"

    context_block = ""
    if context:
        context_block = "Relevant conversation history:\n" + "\n".join(
            f"  - {c}" for c in context
        ) + "\n\n"

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful insurance assistant. Answer concisely.",
            },
            {
                "role": "user",
                "content": f"{context_block}Customer question: {query}",
            },
        ],
        temperature=1,
        max_completion_tokens=256,
        top_p=1,
        stream=True,
    )

    # Stream response (as per llm_test.py reference)
    response_parts = []
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            response_parts.append(content)
    print()

    return "".join(response_parts)


# ── Print summary DB ──────────────────────────────────────────────────────────

def print_summary_db():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = QdrantClient(path=str(HISTORY_SUMMARY_DIR))
    count = client.count(SUMMARY_COLLECTION).count
    print(f"\n{'='*60}")
    print(f"SUMMARY DB — {count} entries")
    print(f"{'='*60}")

    results, _ = client.scroll(
        collection_name=SUMMARY_COLLECTION,
        limit=50,
        with_vectors=False,
        with_payload=True,
        scroll_filter=Filter(must=[
            FieldCondition(key="session_id",  match=MatchValue(value=SESSION_ID)),
            FieldCondition(key="customer_id", match=MatchValue(value=CUSTOMER_ID)),
        ]),
    )

    for point in results:
        p = point.payload
        print(f"\n  ID          : {point.id}")
        print(f"  session_id  : {p['session_id']}")
        print(f"  customer_id : {p['customer_id']}")
        print(f"  reason      : {p['reason']}")
        print(f"  chunk_count : {p['chunk_count']}")
        print(f"  timestamp   : {p['timestamp']}")
        print(f"  summary     : {p['summary'][:300]}...")

    client.close()


# ── Main test ─────────────────────────────────────────────────────────────────

def main():
    history = RuntimeHistory(session_id=SESSION_ID, customer_id=CUSTOMER_ID)

    for i, query in enumerate(QUERIES, start=1):
        print(f"\n{'─'*60}")
        print(f"Query {i:02d}/{len(QUERIES)} | session={SESSION_ID} | customer={CUSTOMER_ID}")
        print(f"User: {query}")

        # 1. Retrieve relevant context from history (filtered by session + customer)
        context = history.retrieve(query, k=3)

        # 2. Call LLM
        print("Assistant: ", end="")
        response = call_llm(query, context)

        # 3 & 4. Store both turns in history
        history.add("user",      query)
        history.add("assistant", response)

    # 5. End session → summarize all remaining → save to summary DB
    print(f"\n{'─'*60}")
    print("Ending session...")
    history.end_session()
    history.close()

    # 6. Show what's in summary DB
    print_summary_db()


if __name__ == "__main__":
    main()
