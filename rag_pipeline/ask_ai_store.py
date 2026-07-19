"""
ask_ai_store.py — chat_bot_ask_ai: the advanced_filter=True Q&A log.

Deliberately independent of RuntimeHistory (the call-transcript pipeline in
history/history_pipeline.py): one process-wide Qdrant client, threads keyed
by (customer_id, ask_ai_session_id) — NOT by any live-call session_id. That
lets a customer hold multiple "Ask AI" conversations at once (ChatGPT-style)
and switch between them freely, independent of whatever call/session happens
to be active. One instance is created at api.py startup and reused for the
life of the process (see api.py).
"""

import threading
from datetime import datetime
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff,
    Filter, FieldCondition, MatchValue, OrderBy, Direction,
)

from constants import (
    CHAT_ASK_AI_DIR, CHAT_ASK_AI_COLLECTION,
    EMBEDDING_DIM, QDRANT_HNSW_M, QDRANT_HNSW_EF_CONSTRUCT,
)
from history.history_pipeline import _embed


class AskAIStore:
    def __init__(self):
        CHAT_ASK_AI_DIR.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(CHAT_ASK_AI_DIR))

        existing = {c.name for c in self._client.get_collections().collections}
        if CHAT_ASK_AI_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=CHAT_ASK_AI_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=QDRANT_HNSW_M, ef_construct=QDRANT_HNSW_EF_CONSTRUCT),
            )

        # Always-incrementing point id, never reused — guarded since save() can
        # be called from concurrent request threads (asyncio.to_thread).
        self._id_counter = self._client.count(CHAT_ASK_AI_COLLECTION).count
        self._id_lock = threading.Lock()

    def _thread_filter(self, customer_id: str, ask_ai_session_id: str) -> Filter:
        return Filter(must=[
            FieldCondition(key="customer_id", match=MatchValue(value=customer_id)),
            FieldCondition(key="ask_ai_session_id", match=MatchValue(value=ask_ai_session_id)),
        ])

    def save(self, customer_id: str, ask_ai_session_id: str, query: str, answer: str) -> None:
        """Save one {query, answer} pair, embedded as combined Q+A text so
        semantic search over this collection can match on either side."""
        combined = f"Q: {query}\nA: {answer}"
        vector = _embed([combined])[0]
        with self._id_lock:
            point_id = self._id_counter
            self._id_counter += 1

        self._client.upsert(
            collection_name=CHAT_ASK_AI_COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "customer_id":       customer_id,
                    "ask_ai_session_id": ask_ai_session_id,
                    "query":             query,
                    "answer":            answer,
                    "turn":              point_id,
                    "timestamp":         datetime.now().isoformat(),
                },
            )],
            wait=True,
        )

    def get_recent(self, customer_id: str, ask_ai_session_id: str, n: int = 5) -> List[str]:
        """Most recent n {query, answer} pairs in this thread, chronological."""
        results, _ = self._client.scroll(
            collection_name=CHAT_ASK_AI_COLLECTION,
            scroll_filter=self._thread_filter(customer_id, ask_ai_session_id),
            limit=n,
            order_by=OrderBy(key="turn", direction=Direction.DESC),
            with_vectors=False,
            with_payload=True,
        )
        return [
            f"Q: {p.payload['query']}\nA: {p.payload['answer']}"
            for p in reversed(results)
        ]

    def search_relevant(self, customer_id: str, ask_ai_session_id: str, query_vec: List[float], k: int = 5) -> List[tuple]:
        """Semantically relevant past {query, answer} pairs in this thread."""
        response = self._client.query_points(
            collection_name=CHAT_ASK_AI_COLLECTION,
            query=query_vec,
            query_filter=self._thread_filter(customer_id, ask_ai_session_id),
            limit=k,
            with_payload=True,
        )
        return [
            (f"Q: {r.payload['query']}\nA: {r.payload['answer']}", r.score, r.payload.get("timestamp", ""))
            for r in response.points
        ]

    def get_thread(self, customer_id: str, ask_ai_session_id: str, limit: int = 200) -> List[dict]:
        """Every {query, answer, timestamp} pair in this thread, chronological —
        structured (unlike get_recent's combined "Q:.../A:..." strings, which
        are meant for LLM context, not UI rendering). Powers "reopen a past
        Ask-AI thread and see the full conversation" in the frontend."""
        results, _ = self._client.scroll(
            collection_name=CHAT_ASK_AI_COLLECTION,
            scroll_filter=self._thread_filter(customer_id, ask_ai_session_id),
            limit=limit,
            order_by=OrderBy(key="turn", direction=Direction.ASC),
            with_vectors=False,
            with_payload=True,
        )
        return [
            {"query": p.payload["query"], "answer": p.payload["answer"], "timestamp": p.payload.get("timestamp", "")}
            for p in results
        ]

    def list_sessions(self, customer_id: str) -> List[dict]:
        """Distinct ask_ai_session_id threads for this customer, most-recent-first —
        powers a ChatGPT-style session switcher. Demo-scale: scrolls this
        customer's whole chat_bot_ask_ai history and groups in Python."""
        points, _ = self._client.scroll(
            collection_name=CHAT_ASK_AI_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="customer_id", match=MatchValue(value=customer_id))]),
            limit=10_000,
            order_by=OrderBy(key="turn", direction=Direction.ASC),
            with_vectors=False,
            with_payload=True,
        )
        sessions: Dict[str, dict] = {}
        for p in points:
            sid = p.payload.get("ask_ai_session_id")
            if not sid:
                continue
            entry = sessions.setdefault(sid, {"ask_ai_session_id": sid, "turn_count": 0})
            entry["turn_count"] += 1
            entry["last_query"] = p.payload["query"]
            entry["last_timestamp"] = p.payload["timestamp"]
        return sorted(sessions.values(), key=lambda s: s["last_timestamp"], reverse=True)

    def close(self) -> None:
        self._client.close()
