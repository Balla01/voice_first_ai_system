"""
Runtime History Pipeline

Two collections:
  1. runtime_history  — in-memory Qdrant (QdrantClient(":memory:"))
                        Fast retrieval during active session.
                        Evicts oldest chunks to summary DB when RAM cap is hit.

  2. session_summaries — persistent Qdrant (disk)
                         Receives summarized chunks on eviction and on session end.

Each point is tagged with session_id + customer_id for filtered retrieval.
"""

import gc
import logging
import os
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, HnswConfigDiff,
    Filter, FieldCondition, MatchValue,
    PointIdsList, OrderBy, Direction,
    Prefetch, FusionQuery, Fusion,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (
    HISTORY_SUMMARY_DIR, DOCS_VECTOR_DIR, EMBEDDING_MODEL, EMBEDDING_DIM,
    QDRANT_HNSW_M, QDRANT_HNSW_EF_CONSTRUCT,
    HISTORY_COLLECTION, SUMMARY_COLLECTION, QDRANT_COLLECTION,
    DOCS_DENSE_VECTOR_NAME, DOCS_SPARSE_VECTOR_NAME,
    MAX_HISTORY_CHUNKS, EVICT_COUNT,
    GROQ_MODEL, EMBED_BATCH_SIZE,
    CLEAR_RUNTIME_HISTORY,
    USE_RERANKER, RERANK_CANDIDATE_POOL, RERANK_PREFETCH_LIMIT,
    USE_QUERY_FILTER,
)
from data_dump.embedder import embed_query_hybrid

logger = logging.getLogger("rag_api.history")

# Separate sub-folder so history DB and summary DB don't share the same
# Qdrant storage when both are persistent.
HISTORY_DB_DIR = HISTORY_SUMMARY_DIR / "history"

load_dotenv()


# ── Embedding model (loaded once, shared across the session) ──────────────────

_model: SentenceTransformer = None
# Guards _get_model()'s check-then-construct against a genuine race: callers
# via asyncio.to_thread() run on real OS threads (a plain "if _model is None"
# check is not atomic), so two near-simultaneous first calls could otherwise
# both pass the check and construct the model concurrently.
_model_lock = threading.Lock()

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # re-check: another thread may have won the race
                print(f"Loading embedding model: {EMBEDDING_MODEL}")
                model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
                model[0].auto_model.config.unpad_inputs = False
                _model = model
                print("Model ready.")
    return _model

def _embed(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    arr = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    result = arr.tolist()
    del arr
    return result


# ── Groq LLM ─────────────────────────────────────────────────────────────────

def _summarize_via_llm(chunks: List[str]) -> str:
    """Call Groq to summarize a list of conversation chunks."""
    api_key = os.getenv("groq_api")
    if not api_key:
        return "[summary unavailable — groq_api not set]"

    conversation = "\n---\n".join(chunks)
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise summarizer. "
                    "Summarize the following conversation turns into a single, "
                    "dense paragraph capturing key topics, decisions, and context."
                ),
            },
            {
                "role": "user",
                "content": f"Conversation:\n{conversation[:4000]}",
            },
        ],
        temperature=0.3,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content.strip()


# ── RuntimeHistory ────────────────────────────────────────────────────────────

class RuntimeHistory:
    """
    Manages conversation history for one session.

    history_client  — in-memory Qdrant, fast retrieval
    summary_client  — persistent Qdrant on disk, survives restarts
    """

    # Every RuntimeHistory instance (one per active session_id+customer_id)
    # shares these three clients at the CLASS level rather than opening its
    # own — history_client/summary_client/docs_client all point at the same
    # physical Qdrant storage folders regardless of which session opens them
    # (rows are scoped by the session_id/customer_id payload filter above,
    # not by separate storage), and Qdrant's embedded/local mode only allows
    # one open handle per folder. Two sessions alive at once (e.g. a live
    # call plus an Ask-AI ad-hoc lookup) used to crash the second one's
    # QdrantClient(path=...) with "already accessed by another instance of
    # Qdrant client" — mirrors how AskAIStore next door is already a single
    # process-wide instance for the same reason.
    _shared_lock = threading.Lock()
    _shared_history_client = None
    _shared_summary_client = None
    _shared_docs_client = None

    @classmethod
    def _get_shared_clients(cls):
        with cls._shared_lock:
            if cls._shared_summary_client is None:
                if CLEAR_RUNTIME_HISTORY:
                    cls._shared_history_client = QdrantClient(":memory:")
                else:
                    HISTORY_DB_DIR.mkdir(parents=True, exist_ok=True)
                    cls._shared_history_client = QdrantClient(path=str(HISTORY_DB_DIR))

                HISTORY_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
                cls._shared_summary_client = QdrantClient(path=str(HISTORY_SUMMARY_DIR))

                # Product-docs collection — separate on-disk Qdrant storage,
                # populated by the data_dump ingestion pipeline (read-only from here).
                DOCS_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
                cls._shared_docs_client = QdrantClient(path=str(DOCS_VECTOR_DIR))
        return cls._shared_history_client, cls._shared_summary_client, cls._shared_docs_client

    @classmethod
    def close_shared_clients(cls):
        """Actually releases the Qdrant locks — call once at process shutdown,
        never per-session (other sessions may still be using them)."""
        with cls._shared_lock:
            for client in (cls._shared_history_client, cls._shared_summary_client, cls._shared_docs_client):
                if client is not None:
                    client.close()
            cls._shared_history_client = cls._shared_summary_client = cls._shared_docs_client = None

    def __init__(self, session_id: str, customer_id: str):
        self.session_id  = session_id
        self.customer_id = customer_id
        self._id_counter = 0   # always incrementing, never reused → no collision on evict
        self._known_plans_cache = None      # distinct docs plan_names, lazily scrolled once
        self._known_categories_cache = None # distinct docs categories, lazily scrolled once

        self.history_client, self.summary_client, self.docs_client = self._get_shared_clients()
        self._create_collection(self.history_client, HISTORY_COLLECTION)
        self._create_collection(self.summary_client, SUMMARY_COLLECTION)

        mode = "RAM (wiped on session end)" if CLEAR_RUNTIME_HISTORY else f"disk ({HISTORY_DB_DIR})"
        print(f"Session started  | session_id={session_id} | customer_id={customer_id}")
        print(f"History mode     | {mode}")

    # ── Collection setup ─────────────────────────────────────────────────────

    def _create_collection(self, client: QdrantClient, name: str):
        existing = {c.name for c in client.get_collections().collections}
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=QDRANT_HNSW_M,
                    ef_construct=QDRANT_HNSW_EF_CONSTRUCT,
                ),
            )

    # ── Filters ──────────────────────────────────────────────────────────────

    def _session_filter(self) -> Filter:
        return Filter(must=[
            FieldCondition(key="session_id",  match=MatchValue(value=self.session_id)),
            FieldCondition(key="customer_id", match=MatchValue(value=self.customer_id)),
        ])

    # ── Add message ──────────────────────────────────────────────────────────

    def add(self, role: str, content: str):
        """
        Add one conversation turn to runtime history.
        role: "user" or "assistant"
        Triggers eviction check after every insert.
        """
        vector = _embed([content])[0]
        point_id = self._id_counter
        self._id_counter += 1

        self.history_client.upsert(
            collection_name=HISTORY_COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "session_id":  self.session_id,
                    "customer_id": self.customer_id,
                    "role":        role,
                    "content":     content,
                    "turn":        point_id,
                    "timestamp":   datetime.now().isoformat(),
                },
            )],
            wait=True,
        )
        self._check_evict()

    # ── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """
        Semantic search over runtime history filtered by session + customer.
        Returns top-k matching content strings.
        """
        vector = _embed([query])[0]
        response = self.history_client.query_points(
            collection_name=HISTORY_COLLECTION,
            query=vector,
            query_filter=self._session_filter(),
            limit=k,
            with_payload=True,
        )
        return [r.payload["content"] for r in response.points]

    # ── Eviction ─────────────────────────────────────────────────────────────

    def _check_evict(self):
        """Evict oldest chunks if THIS session's history exceeds MAX_HISTORY_CHUNKS.

        Must be scoped to this session (not a collection-wide count): runtime_history
        is one shared on-disk collection across every session_id/customer_id ever run
        (CLEAR_RUNTIME_HISTORY=False persists it), so a global count would trigger
        eviction of this session's turns because unrelated sessions filled the cap.
        """
        count = self.history_client.count(
            HISTORY_COLLECTION, count_filter=self._session_filter()
        ).count
        if count > MAX_HISTORY_CHUNKS:
            print(f"  [history] RAM cap reached ({count}/{MAX_HISTORY_CHUNKS}) — evicting {EVICT_COUNT} oldest chunks")
            self._evict(EVICT_COUNT)

    def _evict(self, n: int):
        """
        Take the n oldest chunks (lowest turn IDs) for this session,
        summarize them, save to summary DB, then delete from RAM.
        """
        oldest, _ = self.history_client.scroll(
            collection_name=HISTORY_COLLECTION,
            scroll_filter=self._session_filter(),
            limit=n,
            order_by=OrderBy(key="turn", direction=Direction.ASC),
            with_vectors=False,
            with_payload=True,
        )

        if not oldest:
            return

        texts = [
            f"[{p.payload['role']}]: {p.payload['content']}"
            for p in oldest
        ]
        summary = _summarize_via_llm(texts)
        self._save_summary(summary, reason="eviction", chunk_count=len(oldest))

        ids_to_delete = [p.id for p in oldest]
        self.history_client.delete(
            collection_name=HISTORY_COLLECTION,
            points_selector=PointIdsList(points=ids_to_delete),
            wait=True,
        )
        print(f"  [history] Evicted {len(ids_to_delete)} chunks → summarized → summary DB")
        gc.collect()

    # ── Session end ───────────────────────────────────────────────────────────

    def end_session(self):
        """
        Summarize all remaining history for this session and save to summary DB.

        CLEAR_RUNTIME_HISTORY = True  → history was in RAM; it disappears on close().
        CLEAR_RUNTIME_HISTORY = False → history stays on disk; turns remain queryable
                                        via session_id/customer_id filter for future sessions.
        """
        remaining, _ = self.history_client.scroll(
            collection_name=HISTORY_COLLECTION,
            scroll_filter=self._session_filter(),
            limit=MAX_HISTORY_CHUNKS + EVICT_COUNT + 10,
            order_by=OrderBy(key="turn", direction=Direction.ASC),
            with_vectors=False,
            with_payload=True,
        )

        if remaining:
            texts = [
                f"[{p.payload['role']}]: {p.payload['content']}"
                for p in remaining
            ]
            summary = _summarize_via_llm(texts)
            self._save_summary(summary, reason="session_end", chunk_count=len(remaining))
            print(f"  [session] Summarized {len(remaining)} remaining chunks → summary DB")

        if CLEAR_RUNTIME_HISTORY:
            print(f"  [session] History cleared from RAM (CLEAR_RUNTIME_HISTORY=True)")
        else:
            count = self.history_client.count(HISTORY_COLLECTION).count
            print(f"  [session] History kept on disk — {count} turns remain (CLEAR_RUNTIME_HISTORY=False)")

        print(f"Session ended    | session_id={self.session_id} | customer_id={self.customer_id}")

    # ── Save summary ──────────────────────────────────────────────────────────

    def _save_summary(self, summary: str, reason: str, chunk_count: int):
        """Embed summary and upsert to persistent summary collection."""
        vector = _embed([summary])[0]

        # Summary collection is never deleted from, so count() == next safe ID.
        next_id = self.summary_client.count(SUMMARY_COLLECTION).count

        self.summary_client.upsert(
            collection_name=SUMMARY_COLLECTION,
            points=[PointStruct(
                id=next_id,
                vector=vector,
                payload={
                    "session_id":   self.session_id,
                    "customer_id":  self.customer_id,
                    "summary":      summary,
                    "reason":       reason,        # "eviction" | "session_end"
                    "chunk_count":  chunk_count,
                    "timestamp":    datetime.now().isoformat(),
                },
            )],
            wait=True,
        )

    # ── Scored search methods (return (text, score, timestamp) tuples) ────────

    def get_recent_history(self, n: int = 5) -> List[str]:
        """
        Return the n most recent conversation turns (text only) for this session,
        ordered oldest-first so they read chronologically in context.
        """
        results, _ = self.history_client.scroll(
            collection_name=HISTORY_COLLECTION,
            scroll_filter=self._session_filter(),
            limit=n,
            order_by=OrderBy(key="turn", direction=Direction.DESC),
            with_vectors=False,
            with_payload=True,
        )
        # results come back newest-first; reverse so context reads in order
        turns = [
            f"[{p.payload['role']}]: {p.payload['content']}"
            for p in reversed(results)
        ]
        return turns

    def search_history_scored(self, query_vec: List[float], k: int = 5) -> List[tuple]:
        response = self.history_client.query_points(
            collection_name=HISTORY_COLLECTION,
            query=query_vec,
            query_filter=self._session_filter(),
            limit=k,
            with_payload=True,
        )
        return [
            (r.payload["content"], r.score, r.payload.get("timestamp", ""))
            for r in response.points
        ]

    def search_summary_scored(self, query_vec: List[float], k: int = 5) -> List[tuple]:
        response = self.summary_client.query_points(
            collection_name=SUMMARY_COLLECTION,
            query=query_vec,
            query_filter=self._session_filter(),
            limit=k,
            with_payload=True,
        )
        return [
            (r.payload["summary"], r.score, r.payload.get("timestamp", ""))
            for r in response.points
        ]

    def get_all_summaries(self) -> List[str]:
        """Every saved summary for this session (from evictions + session end),
        chronological. Used for "all available context" reads (e.g. profile
        building, api.py) where a query-driven top-k search doesn't apply —
        there's no question to rank summaries against, we want all of them.
        Sorted client-side (timestamp is an ISO string, not a Qdrant-orderable
        numeric field) rather than via query_points' order_by."""
        results, _ = self.summary_client.scroll(
            collection_name=SUMMARY_COLLECTION,
            scroll_filter=self._session_filter(),
            limit=1000,
            with_vectors=False,
            with_payload=True,
        )
        results.sort(key=lambda p: p.payload.get("timestamp", ""))
        return [p.payload["summary"] for p in results]

    def _known_plans(self) -> List[str]:
        """Distinct plan_name values in the docs collection, cached on first use."""
        if self._known_plans_cache is None:
            from query_understanding import known_plan_names
            self._known_plans_cache = known_plan_names(self.docs_client, QDRANT_COLLECTION)
        return self._known_plans_cache

    def _known_categories(self) -> List[str]:
        """Distinct category values in the docs collection, cached on first use
        (see profile_extractor.py — used to validate the LLM's extracted
        category rather than trust it outright)."""
        if self._known_categories_cache is None:
            from query_understanding import known_categories
            self._known_categories_cache = known_categories(self.docs_client, QDRANT_COLLECTION)
        return self._known_categories_cache

    def _hybrid_docs_search(self, dense_vec, sparse_vec, limit, doc_filter) -> List[tuple]:
        """
        One hybrid dense+sparse + RRF query, optionally metadata-filtered.
        The filter is applied INSIDE each prefetch branch (where the actual
        vector search runs) — a top-level query_filter on a FusionQuery does
        not propagate to the prefetch stages, so it would leak unfiltered hits.
        """
        response = self.docs_client.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=[
                Prefetch(query=dense_vec, using=DOCS_DENSE_VECTOR_NAME,
                         filter=doc_filter, limit=RERANK_PREFETCH_LIMIT),
                Prefetch(query=sparse_vec, using=DOCS_SPARSE_VECTOR_NAME,
                         filter=doc_filter, limit=RERANK_PREFETCH_LIMIT),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [(r.payload.get("text", ""), r.score, "") for r in response.points]

    def search_docs_scored(self, query: str, k: int = 5, doc_filter=None, auto_filter: bool = True,
                            timing: Optional[dict] = None) -> List[tuple]:
        """
        Full docs retrieval, all four features:
          (a) hybrid dense+sparse search  (b) RRF fusion
          (c) metadata filtering — auto-derived from the query via an LLM when
              doc_filter is None, auto_filter=True and USE_QUERY_FILTER
              (validated + 0-hit fallback)
          (d) cross-encoder reranking of the fused shortlist (when USE_RERANKER)

        Pass auto_filter=False when the caller already derived the filter (e.g.
        eval, which logs it) to avoid a duplicate LLM extraction call.

        Embeds `query` with BGE-M3 (docs use a different model than history/summary).
        Returns (text, score, timestamp="") — timestamp always empty for docs.

        timing (optional): if a dict is passed, it is populated in place with
        elapsed milliseconds for each sub-stage — embed_ms (BGE-M3 embedding),
        filter_ms (auto_docs_filter LLM call, only present when it actually
        runs), qdrant_ms (the hybrid dense+sparse + RRF query_points call —
        note RRF fusion (b) happens server-side inside this same call, so it
        has no separate cost of its own), merge_ms (time spent on the 0-hit
        unfiltered fallback re-query, 0.0 when the first query already had
        hits), rerank_ms (cross-encoder rerank, 0.0 when USE_RERANKER is off).
        Callers that don't care about the breakdown simply omit `timing`.
        """
        try:
            t0 = time.perf_counter()
            dense_vec, sparse_vec = embed_query_hybrid(query)
            if timing is not None:
                timing["embed_ms"] = (time.perf_counter() - t0) * 1000

            # (c) metadata filter: use the caller's, else auto-derive from the query.
            if doc_filter is None and auto_filter and USE_QUERY_FILTER:
                t_filter0 = time.perf_counter()
                from query_understanding import auto_docs_filter
                doc_filter, _desc = auto_docs_filter(query, self._known_plans())
                if timing is not None:
                    timing["filter_ms"] = (time.perf_counter() - t_filter0) * 1000

            # (a)+(b) hybrid RRF; pull a wider pool when a reranker will trim it.
            pool = RERANK_CANDIDATE_POOL if USE_RERANKER else k
            t_q0 = time.perf_counter()
            fused = self._hybrid_docs_search(dense_vec, sparse_vec, pool, doc_filter)
            if timing is not None:
                timing["qdrant_ms"] = (time.perf_counter() - t_q0) * 1000
                timing["merge_ms"] = 0.0

            # Filtering must never make results worse: fall back to unfiltered on 0 hits.
            if not fused and doc_filter is not None:
                t_fb0 = time.perf_counter()
                fused = self._hybrid_docs_search(dense_vec, sparse_vec, pool, None)
                if timing is not None:
                    timing["merge_ms"] = (time.perf_counter() - t_fb0) * 1000

            # (d) cross-encoder rerank the shortlist, then truncate to k.
            if USE_RERANKER and fused:
                t_r0 = time.perf_counter()
                from data_dump.reranker import rerank
                reranked = rerank(query, [(t, s) for t, s, _ in fused], top_k=k)
                if timing is not None:
                    timing["rerank_ms"] = (time.perf_counter() - t_r0) * 1000
                return [(t, s, "") for t, s in reranked]

            if timing is not None:
                timing["rerank_ms"] = 0.0
            return fused[:k]
        except Exception as e:
            # Previously silent (bare `except: return []`) — a missing/broken
            # dependency (e.g. FlagEmbedding not installed) or any other
            # retrieval failure looked identical to "no matching docs" in the
            # logs, with zero indication anything was actually wrong. Log the
            # real error (with traceback) so a genuine 0-result query is still
            # distinguishable from a broken retrieval path.
            logger.error(f"[docs] search_docs_scored failed for query={query!r}: {e}", exc_info=True)
            return []

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        """No-op: history_client/summary_client/docs_client are process-wide
        singletons shared by every active session (see _get_shared_clients) —
        closing them here on one session's end would break every other
        session still using them. Use RuntimeHistory.close_shared_clients()
        at actual process shutdown instead."""
        pass
