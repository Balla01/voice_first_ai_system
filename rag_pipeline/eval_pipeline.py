"""
Batch evaluation harness for the RAG pipeline.

Runs every question in questions.json through the same underlying retrieval/
rerank/context/LLM primitives as main.py (search_*_scored, rerank, build_context,
call_llm — all imported unmodified) and logs every stage to CSV, segregating:

  - runtime_history_recent : fixed, most-recent-5 turns (chronological, NOT a
                             semantic search, NOT reranked — history.get_recent_history)
  - history_retrieval      : semantic search over runtime_history (raw)
  - history_retrieval_rerank : same, after the 0.7*sim + 0.3*recency rerank
  - summary_retrieval      : semantic search over session_summaries (raw)
  - summary_retrieval_rerank : same, after rerank
  - docs_retrieval         : semantic search over lic_insurance_docs (k=10).
                             No "_rerank" column: main.py never reranks docs
                             (no timestamps), so there is nothing to log separately.
  - final_answer_docs_only : LLM answer using ONLY the docs_retrieval chunks
                             (all 10, not the usual top-3) as context
  - final_answer_all       : LLM answer using the full production context
                             (recent turns + reranked history + reranked summary
                             + top-3 docs) — exactly what main.py itself builds

Note on docs k=10: this only changes what's visible in docs_retrieval / feeds
final_answer_docs_only. final_answer_all still uses build_context's default
top_k=3, and Qdrant's top-3 is identical whether the candidate pool is 4 or 10
(nearest-neighbor results are prefix-stable), so it is unaffected by the bump.

All questions run in ONE shared session (session_id="eval_run_shared"), so
runtime_history/session_summaries accumulate across questions exactly like a
real multi-turn conversation. Only the "all" answer is written back into
history (history.add) — the docs-only answer is an ablation for this eval and
is not treated as part of the real conversation.

Usage:
    cd main/src
    python eval_pipeline.py
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from history.history_pipeline import RuntimeHistory, _embed
from main import rerank, build_context, call_llm
from query_understanding import auto_docs_filter
from constants import OUTPUT_DIR, QUESTIONS_JSON_PATH, USE_QUERY_FILTER

QUESTIONS_PATH = QUESTIONS_JSON_PATH
OUTPUT_CSV = OUTPUT_DIR / "eval_results.csv"

SESSION_ID = "eval_run_shared"
CUSTOMER_ID = "eval_customer"

# Retrieval breadth per collection — decoupled since docs needs a bigger pool
# than history/summary (parallel_search() in main.py applies one k to all three,
# so we call the underlying search_*_scored methods directly instead).
HISTORY_K = 4
SUMMARY_K = 4
DOCS_K = 10

# How many docs chunks feed the docs-only answer (all of them, not the usual top-3).
DOCS_ONLY_TOP_K = DOCS_K

FIELDNAMES = [
    "id", "category", "query",
    "runtime_history_recent",
    "history_retrieval", "history_retrieval_rerank",
    "summary_retrieval", "summary_retrieval_rerank",
    "docs_filter_used",
    "docs_retrieval",
    "context_docs_only", "final_answer_docs_only",
    "context_all", "final_answer_all",
]


def _format_scored(items, limit: int = 10) -> str:
    """items: list of (text, score) or (text, score, ...). One 'score | snippet' line per item."""
    lines = []
    for entry in items[:limit]:
        text, score = entry[0], entry[1]
        snippet = " ".join(text.split())[:250]
        lines.append(f"[{score:.4f}] {snippet}")
    return "\n".join(lines)


def run(question_bank: list, limit: int = None) -> list:
    """
    Runs the pipeline over question_bank, returns the list of result rows.

    question_bank: list of {"category": str, "questions": [{"id", "question"}, ...]}
    """
    total = sum(len(entry["questions"]) for entry in question_bank)
    if limit:
        total = min(total, limit)

    history = RuntimeHistory(session_id=SESSION_ID, customer_id=CUSTOMER_ID)
    rows = []
    done = 0

    try:
        for entry in question_bank:
            category = entry["category"]
            questions = entry["questions"]
            for item in questions:
                if limit and done >= limit:
                    break
                done += 1
                qid = item["id"]
                query = item["question"]
                print(f"\n{'='*65}")
                print(f"[{done}/{total}] ({category} #{qid}) {query}")

                t0 = time.perf_counter()

                # 1. Embed query once (same as main.py)
                query_vec = _embed([query])[0]

                # 2. Fixed, most-recent-5 turns — chronological, NOT a semantic
                #    search, NOT reranked. Segregated from history_retrieval below.
                recent_turns = history.get_recent_history(n=5)

                # 3. RAW retrieval, one call per collection, independent k per collection.
                #    Docs: derive the LLM metadata filter here (so it can be logged),
                #    then pass it explicitly (auto_filter=False) to avoid a duplicate
                #    extraction call. search_docs_scored applies hybrid+RRF, the
                #    metadata filter (with 0-hit fallback), and cross-encoder rerank.
                if USE_QUERY_FILTER:
                    doc_filter, filter_desc = auto_docs_filter(query, history._known_plans())
                else:
                    doc_filter, filter_desc = None, "disabled"
                raw_history = history.search_history_scored(query_vec, HISTORY_K)
                raw_summary = history.search_summary_scored(query_vec, SUMMARY_K)
                raw_docs    = history.search_docs_scored(query, DOCS_K, doc_filter=doc_filter, auto_filter=False)

                # 4. Rerank history/summary with the recency reranker (main.py's, unmodified).
                #    Docs are already cross-encoder-reranked inside search_docs_scored.
                history_ranked = rerank(raw_history)
                summary_ranked = rerank(raw_summary)
                docs_ranked = [(text, score) for text, score, _ in raw_docs]

                # 5a. Docs-only context/answer — ablation: only the doc chunks,
                #     using all DOCS_ONLY_TOP_K retrieved (not the usual top-3).
                context_docs_only = build_context([], [], [], docs_ranked, top_k=DOCS_ONLY_TOP_K)
                print("Assistant (docs-only): ", end="")
                answer_docs_only = call_llm(query, context_docs_only)

                # 5b. Full context/answer — identical to main.py's real production
                #     call (recent turns + reranked history + reranked summary +
                #     top-3 docs, build_context's default top_k=3).
                context_all = build_context(recent_turns, history_ranked, summary_ranked, docs_ranked)
                print("Assistant (all): ", end="")
                answer_all = call_llm(query, context_all)

                elapsed = time.perf_counter() - t0
                print(f"  [{elapsed:.2f}s]")

                rows.append({
                    "id": qid,
                    "category": category,
                    "query": query,
                    "runtime_history_recent": "\n".join(recent_turns),
                    "history_retrieval": _format_scored(raw_history, limit=HISTORY_K),
                    "history_retrieval_rerank": _format_scored(history_ranked, limit=HISTORY_K),
                    "summary_retrieval": _format_scored(raw_summary, limit=SUMMARY_K),
                    "summary_retrieval_rerank": _format_scored(summary_ranked, limit=SUMMARY_K),
                    "docs_filter_used": filter_desc,
                    "docs_retrieval": _format_scored(docs_ranked, limit=DOCS_K),
                    "context_docs_only": context_docs_only,
                    "final_answer_docs_only": answer_docs_only,
                    "context_all": context_all,
                    "final_answer_all": answer_all,
                })

                # 6. Write back to history exactly as main.py does — only the
                #    production-faithful "all" answer becomes part of the real
                #    conversation; the docs-only answer is an eval-only ablation.
                history.add("user", query)
                history.add("assistant", answer_all)

            if limit and done >= limit:
                break

        print(f"\n{'='*65}")
        print("Ending session...")
        history.end_session()
    finally:
        history.close()

    return rows


def save_csv(rows: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {output_path}")


def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        question_bank = json.load(f)

    rows = run(question_bank)
    save_csv(rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()
