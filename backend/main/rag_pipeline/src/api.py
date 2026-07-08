"""
api.py — FastAPI wrapper around the RAG pipeline (main.py).

POST /query
  body:   {query, session_id, customer_id, stream}
  stream=False -> single JSON: {answer, retrieval_time_s, llm_time_s, total_time_s}
  stream=True  -> text/event-stream: one "data:" event per token as it
                  arrives from Groq, then a final "event: done" carrying the
                  full answer + timing breakdown.

One RuntimeHistory per (session_id, customer_id), created on first use and
cached for the life of the process — RuntimeHistory's on-disk Qdrant client
(CLEAR_RUNTIME_HISTORY=False, see constants.py) holds a directory lock, so it
must stay open rather than being recreated per request.

Run:
    cd rag_pipeline/src
    uvicorn api:app --reload --port 8001
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from history.history_pipeline import RuntimeHistory, _embed, _get_model
from main import parallel_search, build_context, call_llm, stream_llm_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(title="RAG Pipeline API")

_histories: Dict[Tuple[str, str], RuntimeHistory] = {}
_histories_lock = Lock()

# Must match build_context()'s own top_k default (main.py) — kept as an
# explicit constant here so the retrieval log ("N/M sent to LLM") and the
# actual build_context() call can never drift apart.
CONTEXT_TOP_K = 3
CHUNK_LOG_WORD_LIMIT = 30


def _truncate_words(text: str, limit: int = CHUNK_LOG_WORD_LIMIT) -> str:
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + " ..."


def _log_chunks(label: str, chunks: List[Tuple[str, float]]) -> None:
    """chunks: list of (text, score), already reranked/sorted — logs every one, not just the ones used."""
    logger.info(f"[retrieval:{label}] {len(chunks)} chunk(s) retrieved")
    for i, (text, score) in enumerate(chunks, start=1):
        logger.info(f"  [{label} #{i}] score={score:.4f} | {_truncate_words(text)}")


@app.on_event("startup")
async def startup():
    # Force the SentenceTransformer to load now (it's normally lazy, on first
    # _embed() call) so the app doesn't report ready until it's actually
    # ready to serve a query at full speed.
    await asyncio.to_thread(_get_model)


def _get_history(session_id: str, customer_id: str) -> RuntimeHistory:
    key = (session_id, customer_id)
    with _histories_lock:
        history = _histories.get(key)
        if history is None:
            history = RuntimeHistory(session_id=session_id, customer_id=customer_id)
            _histories[key] = history
        return history


@app.on_event("shutdown")
async def shutdown():
    with _histories_lock:
        for history in _histories.values():
            history.close()
        _histories.clear()


class QueryRequest(BaseModel):
    query: str
    session_id: str
    customer_id: str
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    retrieval_time_s: float
    llm_time_s: float
    total_time_s: float


def _retrieve_context(query: str, history: RuntimeHistory) -> str:
    """Embed + search + rerank + build_context — same steps as main.py's main(), synchronous/blocking."""
    query_vec = _embed([query])[0]
    recent_turns = history.get_recent_history(n=5)
    history_ranked, summary_ranked, docs_ranked = parallel_search(query_vec, history)

    logger.info(f"[retrieval:recent_history] {len(recent_turns)} turn(s) (chronological, not reranked)")
    _log_chunks("history", history_ranked)
    _log_chunks("summary", summary_ranked)
    _log_chunks("docs", docs_ranked)

    context = build_context(recent_turns, history_ranked, summary_ranked, docs_ranked, top_k=CONTEXT_TOP_K)

    used_history = min(len(history_ranked), CONTEXT_TOP_K)
    used_summary = min(len(summary_ranked), CONTEXT_TOP_K)
    used_docs = min(len(docs_ranked), CONTEXT_TOP_K)
    logger.info(
        f"[context] sent to LLM (top_k={CONTEXT_TOP_K} per collection): "
        f"recent_turns={len(recent_turns)}/{len(recent_turns)} (all), "
        f"history={used_history}/{len(history_ranked)}, "
        f"summary={used_summary}/{len(summary_ranked)}, "
        f"docs={used_docs}/{len(docs_ranked)}, "
        f"total_chunks_to_llm={len(recent_turns) + used_history + used_summary + used_docs}"
    )

    return context


@app.post("/query")
async def query(request: QueryRequest):
    history = _get_history(request.session_id, request.customer_id)

    t_start = time.perf_counter()
    context = await asyncio.to_thread(_retrieve_context, request.query, history)
    t_retrieval = time.perf_counter()

    if request.stream:
        return StreamingResponse(
            _stream_answer(request.query, history, context, t_start, t_retrieval),
            media_type="text/event-stream",
        )

    answer = await asyncio.to_thread(call_llm, request.query, context)
    t_end = time.perf_counter()

    await asyncio.to_thread(history.add, "user", request.query)
    await asyncio.to_thread(history.add, "assistant", answer)

    return QueryResponse(
        answer=answer,
        retrieval_time_s=round(t_retrieval - t_start, 3),
        llm_time_s=round(t_end - t_retrieval, 3),
        total_time_s=round(t_end - t_start, 3),
    )


def _stream_answer(query: str, history: RuntimeHistory, context: str, t_start: float, t_retrieval: float):
    """Sync generator (Starlette runs it in a threadpool) — yields SSE token events, then a final done event."""
    parts = []
    for token in stream_llm_chunks(query, context):
        parts.append(token)
        yield f"data: {json.dumps({'token': token})}\n\n"

    answer = "".join(parts)
    t_end = time.perf_counter()

    history.add("user", query)
    history.add("assistant", answer)

    done_payload = {
        "answer": answer,
        "retrieval_time_s": round(t_retrieval - t_start, 3),
        "llm_time_s": round(t_end - t_retrieval, 3),
        "total_time_s": round(t_end - t_start, 3),
    }
    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(_histories)}
