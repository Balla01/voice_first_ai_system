"""
api.py — FastAPI wrapper around the RAG pipeline (main.py).

POST /query
  body:   {query, session_id, customer_id, stream, advanced_filter, ask_ai_session_id}
  stream=False -> single JSON: {answer, retrieval_time_s, llm_time_s, total_time_s, ask_ai_session_id}
  stream=True  -> text/event-stream: one "data:" event per token as it
                  arrives from Groq, then a final "event: done" carrying the
                  full answer + timing breakdown + ask_ai_session_id.

  advanced_filter=True turns on chat_bot_ask_ai: recent + semantically relevant
  past Q&A from an Ask-AI "thread" (ask_ai_session_id, ChatGPT-style — see
  ask_ai_store.py), plus a live web search when the query is classified as
  off-domain/general/incomplete (query_understanding.classify_web_search).
  Omit ask_ai_session_id to start a new thread; the server mints one and
  returns it for the client to reuse on the next message in that thread.

  Ambiguous-reference clarification (advanced_filter only): if the query reads
  like "correct this suggestion"/"fix that point" (ambiguous_reference.py) and
  context_source isn't set yet, the response comes back immediately with
  needs_clarification=true + clarification_options instead of an answer — no
  LLM call is made. The client re-sends the SAME query with context_source set
  to the user's pick to get the real answer:
    "suggestion_card" -> context = this call's session history + summaries only
                          (RuntimeHistory; no docs, no chat_bot_ask_ai, no web search)
    "current_thread"  -> context = this Ask-AI thread's own history only
                          (chat_bot_ask_ai; no RuntimeHistory, no docs, no web search)
  context_source is ignored (has no effect) when the query wasn't flagged as
  ambiguous — the normal multi-source advanced_filter context is used instead.

GET /ask-ai/sessions?customer_id=...
  Lists a customer's Ask-AI threads (most-recent-first) for a session switcher.

GET /ask-ai/thread?customer_id=...&ask_ai_session_id=...
  Full {query, answer, timestamp} history of one Ask-AI thread, chronological —
  for reopening a past thread and replaying it in the UI (see /ask-ai/sessions
  for the list of thread ids to choose from).

GET /profile?session_id=...&customer_id=...
  Builds a customer profile (name, age, profession, location, policy_product,
  category) from all available context for that session — earlier summaries
  plus recent raw turns — via profile_extractor.py. Best-effort: unset fields
  come back null, never an error.

POST /session/{session_id}/end?customer_id=...
  Called when a call ends: summarizes any remaining runtime history for this
  (session_id, customer_id) into the persistent summary DB and closes its
  Qdrant clients, then evicts it from the process-wide RuntimeHistory cache.
  No-ops successfully (not an error) if that session was never actually used.

One RuntimeHistory per (session_id, customer_id), created on first use and
cached for the life of the process — RuntimeHistory's on-disk Qdrant client
(CLEAR_RUNTIME_HISTORY=False, see constants.py) holds a directory lock, so it
must stay open rather than being recreated per request. AskAIStore (chat_bot_ask_ai)
is a single process-wide instance instead, since its threads are scoped by
customer_id + ask_ai_session_id, not by call session_id.

Run:
    cd rag_pipeline/src
    uvicorn api:app --reload --port 8001
"""

import asyncio
import json
import logging
import sys
import threading
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client.models import Filter, FieldCondition, MatchValue

sys.path.insert(0, str(Path(__file__).resolve().parent))

from history.history_pipeline import RuntimeHistory, _embed, _get_model
from main import parallel_search, build_context, call_llm, stream_llm_chunks, rerank
from query_understanding import classify_web_search
from web_search_call import web_search_answer
from ask_ai_store import AskAIStore
from profile_extractor import extract_profile
from email_trigger import detect_email_request
from email_sending_test import send_email
from ambiguous_reference import is_ambiguous_reference
from constants import DEBUG, MAX_HISTORY_CHUNKS, EVICT_COUNT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(title="RAG Pipeline API")

_histories: Dict[Tuple[str, str], RuntimeHistory] = {}
_histories_lock = Lock()

# chat_bot_ask_ai (advanced_filter mode): one store for the whole process,
# NOT keyed by call session_id — see ask_ai_store.py for why threads are
# scoped to (customer_id, ask_ai_session_id) instead.
_ask_ai_store: Optional[AskAIStore] = None

# Must match build_context()'s own top_k default (main.py) — kept as an
# explicit constant here so the retrieval log ("N/M sent to LLM") and the
# actual build_context() call can never drift apart.
CONTEXT_TOP_K = 3
CHUNK_LOG_WORD_LIMIT = 30

# advanced_filter=True system prompt: the base prompt only tells the LLM to use
# "the provided context" — under advanced_filter the context can carry FOUR
# kinds of sections (insurance docs/history/summary, recent chat_bot_ask_ai
# turns, semantically relevant past Q&A, and live web search results), so the
# LLM needs to be told to pick only the sections relevant to this question.
ADVANCED_SYSTEM_PROMPT = (
    "You are an insurance assistant with access to multiple context sources, "
    "each under its own '--- section ---' heading: recent conversation, "
    "insurance document knowledge, past session summaries, recent past Q&A "
    "exchanges, semantically relevant past Q&A, and (only when the question "
    "needs it) live web search results.\n\n"
    "Use ONLY the sections that are actually relevant to the customer's "
    "current question — ignore sections that don't apply, and don't mention "
    "the sections themselves. If the question is about LIC insurance/pension "
    "plans, premiums, claims, or policy details, prefer the insurance "
    "document/history sections. If it is a general or off-domain question, "
    "prefer the web search section. Answer directly and concisely."
)

# Appended to ADVANCED_SYSTEM_PROMPT only when the query itself was detected
# (email_trigger.detect_email_request, BEFORE the LLM call) as an email-send
# request. Without this, the raw query text (e.g. "...send this to
# abc@gmail.com") reads to the LLM as an instruction it must fulfill itself,
# and since it has no email tool it responds with a capability disclaimer
# ("I can't send emails") instead of the actual informational answer — even
# though _send_answer_email() already handles the send afterward, entirely
# independent of what the LLM says. This note tells the LLM that instruction
# is handled automatically, so it should just answer the real question.
EMAIL_HANDLING_NOTE = (
    "\n\nNote: this message also asks to email/send the answer somewhere. "
    "That delivery is handled automatically by the system after you respond — "
    "you do not have email-sending capability and must NOT say so, apologize "
    "for it, or mention email/delivery at all. Simply answer the underlying "
    "question/information request directly, exactly as you would if the "
    "'send/email this to ...' instruction were not present."
)


def _truncate_words(text: str, limit: int = CHUNK_LOG_WORD_LIMIT) -> str:
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + " ..."


def _log_chunks(label: str, chunks: List[Tuple[str, float]], full: bool = False) -> None:
    """
    chunks: list of (text, score), already reranked/sorted — logs every one,
    not just the ones used. full=True (only used for docs when DEBUG=True)
    logs each chunk's untruncated text instead of the normal 30-word preview.
    """
    logger.info(f"[retrieval:{label}] {len(chunks)} chunk(s) retrieved")
    for i, (text, score) in enumerate(chunks, start=1):
        display = text if full else _truncate_words(text)
        logger.info(f"  [{label} #{i}] score={score:.4f} | {display}")


@app.on_event("startup")
async def startup():
    global _ask_ai_store
    # Force the SentenceTransformer to load now (it's normally lazy, on first
    # _embed() call) so the app doesn't report ready until it's actually
    # ready to serve a query at full speed.
    await asyncio.to_thread(_get_model)
    _ask_ai_store = await asyncio.to_thread(AskAIStore)


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
    if _ask_ai_store is not None:
        _ask_ai_store.close()


class QueryRequest(BaseModel):
    query: str
    session_id: str
    customer_id: str
    stream: bool = False
    # Optional explicit docs metadata filter. When any is set, it overrides the
    # LLM-derived query filter (search_docs_scored's auto path is skipped).
    plan_name: Optional[str] = None
    doc_type: Optional[str] = None
    product_type: Optional[str] = None
    tenant_id: Optional[str] = None
    # When True: adds chat_bot_ask_ai (recent + semantically relevant past Q&A)
    # to the context, triggers a live web search for non-insurance/incomplete
    # queries, and logs this {query, answer} pair to chat_bot_ask_ai. Default
    # False preserves the exact existing behavior.
    advanced_filter: bool = False
    # Ask-AI "thread" id (only meaningful when advanced_filter=True) — like a
    # ChatGPT conversation id. Omit on the first message of a new thread; the
    # server mints one and returns it in QueryResponse so the client can pass
    # it back on subsequent messages in the same thread. Pass a previously
    # returned id to continue that thread, or a different one to switch threads.
    ask_ai_session_id: Optional[str] = None
    # Resolves an ambiguous-reference clarification (see module docstring) —
    # "suggestion_card" or "current_thread". Only has an effect when the query
    # is actually flagged as ambiguous; ignored otherwise. Leave unset on a
    # fresh query — set it only when re-sending after the user picked an
    # option in response to needs_clarification=true.
    context_source: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    retrieval_time_s: float
    llm_time_s: float
    total_time_s: float
    # Echoed/minted only when advanced_filter=True — persist this and send it
    # back as ask_ai_session_id on the next call to stay in the same thread.
    ask_ai_session_id: Optional[str] = None
    # Set only when advanced_filter=True and the query was detected as an
    # email-send request (email_trigger.detect_email_request) — the recipient
    # the answer was dispatched to. Fire-and-forget: this means "queued", not
    # "confirmed delivered" — actual success/failure only appears in server
    # logs ([advanced:email] sent/failed), never as an API error.
    emailed_to: Optional[str] = None
    # True when the query was flagged as an ambiguous reference (see module
    # docstring) and context_source wasn't already given — answer is "" in
    # this case; no LLM call was made. Re-send the same query with
    # context_source set to one of clarification_options' values to get the
    # real answer.
    needs_clarification: bool = False
    clarification_options: Optional[List[dict]] = None


def _build_explicit_filter(request: "QueryRequest"):
    """Build a Qdrant Filter from explicit request fields, or None if none set."""
    conds = []
    for key in ("plan_name", "doc_type", "product_type", "tenant_id"):
        val = getattr(request, key, None)
        if val:
            conds.append(FieldCondition(key=key, match=MatchValue(value=val)))
    return Filter(must=conds) if conds else None


def _retrieve_context(query: str, history: RuntimeHistory, doc_filter=None, advanced: bool = False,
                      customer_id: str = None, ask_ai_session_id: str = None) -> str:
    """Embed + search + rerank + build_context — same steps as main.py's main(), synchronous/blocking.

    advanced=True additionally appends chat_bot_ask_ai context (recent + semantically
    relevant past Q&A for this ask_ai_session_id thread) and, when the query is
    classified as web-search-worthy (non-insurance / general / incomplete), live
    web search results.
    """
    query_vec = _embed([query])[0]
    recent_turns = history.get_recent_history(n=5)
    history_ranked, summary_ranked, docs_ranked = parallel_search(query, query_vec, history, doc_filter=doc_filter)

    logger.info(f"[retrieval:docs_filter] explicit={'yes' if doc_filter is not None else 'no (auto LLM filter)'}")
    logger.info(f"[retrieval:recent_history] {len(recent_turns)} turn(s) (chronological, not reranked)")
    _log_chunks("history", history_ranked)
    _log_chunks("summary", summary_ranked)
    # docs get full-text logging when DEBUG=True (constants.py) — the other
    # two collections stay truncated since they're rarely what you're debugging.
    _log_chunks("docs", docs_ranked, full=DEBUG)

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

    if advanced:
        context += _advanced_context(query, query_vec, customer_id, ask_ai_session_id)

    return context


def _advanced_context(query: str, query_vec: list, customer_id: str, ask_ai_session_id: str) -> str:
    """chat_bot_ask_ai (recent + relevant past Q&A for this thread) + a live web
    search when the query is classified as web-search-worthy. Appended to the
    base context string."""
    extra = []

    recent_qas = _ask_ai_store.get_recent(customer_id, ask_ai_session_id, n=5)
    logger.info(f"[advanced:chat_ask_ai_recent] thread={ask_ai_session_id} {len(recent_qas)} turn(s)")
    if recent_qas:
        extra.append("--- Recent chat_bot_ask_ai turns (chronological) ---")
        extra.extend(f"  {qa}" for qa in recent_qas)

    qa_ranked = _ask_ai_store.search_relevant(customer_id, ask_ai_session_id, query_vec, k=CONTEXT_TOP_K)
    _log_chunks("chat_ask_ai_relevant", [(text, score) for text, score, _ts in qa_ranked])
    if qa_ranked:
        extra.append("--- Relevant past Q&A (chat_bot_ask_ai) ---")
        extra.extend(f"  • {text}" for text, _score, _ts in qa_ranked)

    web_triggered = classify_web_search(query)
    logger.info(f"[advanced:web_search_trigger] {web_triggered}")
    if web_triggered:
        try:
            web_text = web_search_answer(query)
        except Exception as e:
            logger.warning(f"[advanced:web_search] failed: {e}")
            web_text = ""
        if web_text:
            extra.append("--- Live web search results ---")
            extra.append(f"  {web_text}")

    return ("\n" + "\n".join(extra)) if extra else ""


CLARIFICATION_OPTIONS = [
    {"value": "suggestion_card", "label": "The suggestion card"},
    {"value": "current_thread", "label": "This chat thread"},
]


def _suggestion_card_context(query: str, history: RuntimeHistory) -> str:
    """Narrow context for the 'suggestion_card' disambiguation choice: this
    call's own recent turns + session summaries only (RuntimeHistory's two
    collections) — no insurance docs, no chat_bot_ask_ai, no web search.
    Deliberately calls history's own scored-search methods directly (not
    parallel_search) to skip the docs/BGE-M3 hybrid search entirely, since
    it's out of scope here and would otherwise waste that retrieval work."""
    query_vec = _embed([query])[0]
    recent_turns = history.get_recent_history(n=5)
    history_ranked = rerank(history.search_history_scored(query_vec, k=4))
    summary_ranked = rerank(history.search_summary_scored(query_vec, k=4))

    logger.info(f"[clarify:suggestion_card] recent_turns={len(recent_turns)}")
    _log_chunks("history", history_ranked)
    _log_chunks("summary", summary_ranked)

    return build_context(recent_turns, history_ranked, summary_ranked, [], top_k=CONTEXT_TOP_K)


def _current_thread_context(customer_id: str, ask_ai_session_id: str, query: str) -> str:
    """Narrow context for the 'current_thread' disambiguation choice: only
    this Ask-AI thread's own history (chat_bot_ask_ai) — no RuntimeHistory, no
    docs, no web search."""
    query_vec = _embed([query])[0]
    parts = []

    recent_qas = _ask_ai_store.get_recent(customer_id, ask_ai_session_id, n=5)
    logger.info(f"[clarify:current_thread] thread={ask_ai_session_id} recent={len(recent_qas)}")
    if recent_qas:
        parts.append("--- Recent chat_bot_ask_ai turns (chronological) ---")
        parts.extend(f"  {qa}" for qa in recent_qas)

    qa_ranked = _ask_ai_store.search_relevant(customer_id, ask_ai_session_id, query_vec, k=CONTEXT_TOP_K)
    _log_chunks("chat_ask_ai_relevant", [(text, score) for text, score, _ts in qa_ranked])
    if qa_ranked:
        parts.append("--- Relevant past Q&A (chat_bot_ask_ai) ---")
        parts.extend(f"  • {text}" for text, _score, _ts in qa_ranked)

    return "\n".join(parts)


EMAIL_SUBJECT_QUERY_CHARS = 60


def _derive_email_subject(query: str) -> str:
    """Deterministic subject line from the query text — no LLM call, per the
    regex-only decision for this whole feature."""
    text = " ".join(query.split())
    if len(text) > EMAIL_SUBJECT_QUERY_CHARS:
        text = text[:EMAIL_SUBJECT_QUERY_CHARS].rstrip() + "..."
    return f"Insurance Assistant: {text}"


def _send_answer_email(recipient: str, query: str, answer: str) -> None:
    """Best-effort + fire-and-forget: dispatch `answer` verbatim as the email
    body to `recipient` on a background thread. Never raises, never blocks the
    caller — mirrors the classify_web_search/web_search_answer try/except
    pattern already used in _advanced_context, but doesn't even wait on the
    outcome. `recipient` is detected by the caller (once, before the LLM call —
    see EMAIL_HANDLING_NOTE) rather than re-detected here."""

    def _worker():
        try:
            send_email(recipient, _derive_email_subject(query), answer)
            logger.info(f"[advanced:email] sent to {recipient}")
        except Exception as e:
            logger.warning(f"[advanced:email] failed to send to {recipient}: {e}")

    threading.Thread(target=_worker, daemon=True).start()


@app.post("/query")
async def query(request: QueryRequest):
    history = _get_history(request.session_id, request.customer_id)

    # A new thread id is minted here (not left to the client) so the first
    # message of an Ask-AI conversation doesn't need one pre-generated —
    # mirrors ChatGPT starting a new conversation on the first message.
    ask_ai_session_id = None
    if request.advanced_filter:
        ask_ai_session_id = request.ask_ai_session_id or str(uuid.uuid4())

    # Ambiguous-reference check: only on a query's FIRST pass (context_source
    # not already set — a resend after the user picked an option skips this
    # entirely, even if the resent text still matches the pattern). No LLM
    # call, no retrieval — just tell the client which context sources to
    # disambiguate between.
    if (request.advanced_filter and request.context_source is None
            and is_ambiguous_reference(request.query)):
        logger.info(f"[clarify] ambiguous reference detected: {request.query!r}")
        return QueryResponse(
            answer="",
            retrieval_time_s=0.0, llm_time_s=0.0, total_time_s=0.0,
            ask_ai_session_id=ask_ai_session_id,
            needs_clarification=True,
            clarification_options=CLARIFICATION_OPTIONS,
        )

    t_start = time.perf_counter()
    doc_filter = _build_explicit_filter(request)

    if request.context_source == "suggestion_card":
        context = await asyncio.to_thread(_suggestion_card_context, request.query, history)
    elif request.context_source == "current_thread":
        context = await asyncio.to_thread(
            _current_thread_context, request.customer_id, ask_ai_session_id, request.query
        )
    else:
        context = await asyncio.to_thread(
            _retrieve_context, request.query, history, doc_filter, request.advanced_filter,
            request.customer_id, ask_ai_session_id,
        )
    t_retrieval = time.perf_counter()

    # Detected BEFORE the LLM call (not after, as originally) so the system
    # prompt can be adjusted when the query itself contains an email-send
    # instruction — otherwise the raw query text reads to the LLM as something
    # it must fulfill itself, and it responds with an "I can't send emails"
    # disclaimer instead of the actual answer. See EMAIL_HANDLING_NOTE.
    email_recipient = detect_email_request(request.query) if request.advanced_filter else None
    system_prompt = None
    if request.advanced_filter:
        system_prompt = ADVANCED_SYSTEM_PROMPT + (EMAIL_HANDLING_NOTE if email_recipient else "")

    if request.stream:
        return StreamingResponse(
            _stream_answer(request.query, history, context, t_start, t_retrieval,
                           system_prompt, request.advanced_filter, request.customer_id, ask_ai_session_id,
                           email_recipient),
            media_type="text/event-stream",
        )

    answer = await asyncio.to_thread(call_llm, request.query, context, system_prompt)
    t_end = time.perf_counter()

    await asyncio.to_thread(history.add, "user", request.query)
    await asyncio.to_thread(history.add, "assistant", answer)

    emailed_to = None
    if request.advanced_filter:
        await asyncio.to_thread(_ask_ai_store.save, request.customer_id, ask_ai_session_id, request.query, answer)
        if email_recipient:
            _send_answer_email(email_recipient, request.query, answer)
            emailed_to = email_recipient

    return QueryResponse(
        answer=answer,
        retrieval_time_s=round(t_retrieval - t_start, 3),
        llm_time_s=round(t_end - t_retrieval, 3),
        total_time_s=round(t_end - t_start, 3),
        ask_ai_session_id=ask_ai_session_id,
        emailed_to=emailed_to,
    )


def _stream_answer(query: str, history: RuntimeHistory, context: str, t_start: float, t_retrieval: float,
                   system_prompt: str = None, advanced_filter: bool = False,
                   customer_id: str = None, ask_ai_session_id: str = None,
                   email_recipient: str = None):
    """Sync generator (Starlette runs it in a threadpool) — yields SSE token events, then a final done event."""
    parts = []
    for token in stream_llm_chunks(query, context, system_prompt):
        parts.append(token)
        yield f"data: {json.dumps({'token': token})}\n\n"

    answer = "".join(parts)
    t_end = time.perf_counter()

    history.add("user", query)
    history.add("assistant", answer)

    emailed_to = None
    if advanced_filter:
        _ask_ai_store.save(customer_id, ask_ai_session_id, query, answer)
        if email_recipient:
            _send_answer_email(email_recipient, query, answer)
            emailed_to = email_recipient

    done_payload = {
        "answer": answer,
        "retrieval_time_s": round(t_retrieval - t_start, 3),
        "llm_time_s": round(t_end - t_retrieval, 3),
        "total_time_s": round(t_end - t_start, 3),
        "ask_ai_session_id": ask_ai_session_id,
        "emailed_to": emailed_to,
    }
    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"


@app.get("/ask-ai/sessions")
async def ask_ai_sessions(customer_id: str):
    """List this customer's Ask-AI threads (most-recent-first) — powers a
    ChatGPT-style session switcher in the frontend."""
    sessions = await asyncio.to_thread(_ask_ai_store.list_sessions, customer_id)
    return {"customer_id": customer_id, "sessions": sessions}


@app.get("/ask-ai/thread")
async def ask_ai_thread(customer_id: str, ask_ai_session_id: str):
    """Full chronological {query, answer, timestamp} history of one Ask-AI
    thread — for reopening a thread picked from /ask-ai/sessions and replaying
    it in the UI."""
    turns = await asyncio.to_thread(_ask_ai_store.get_thread, customer_id, ask_ai_session_id)
    return {"customer_id": customer_id, "ask_ai_session_id": ask_ai_session_id, "turns": turns}


def _build_session_transcript(history: RuntimeHistory) -> str:
    """All available context for this session: earlier summaries (evicted/
    session-end digests) followed by every raw turn still in runtime_history,
    oldest-first. Unlike _retrieve_context, this has no query to rank against —
    profile building wants everything, not a top-k slice."""
    parts = []

    summaries = history.get_all_summaries()
    if summaries:
        parts.append("--- Earlier session summary ---")
        parts.extend(summaries)

    # runtime_history only ever holds up to MAX_HISTORY_CHUNKS+EVICT_COUNT turns
    # for one session before eviction (see history_pipeline.py); this cap just
    # asks for "all of it" without an arbitrary small default like the n=5 used
    # for live-query context.
    recent_turns = history.get_recent_history(n=MAX_HISTORY_CHUNKS + EVICT_COUNT + 10)
    if recent_turns:
        parts.append("--- Recent turns ---")
        parts.extend(recent_turns)

    return "\n".join(parts)


@app.get("/profile")
async def profile(session_id: str, customer_id: str):
    """Build a customer profile from all available context for this session —
    see profile_extractor.py. Best-effort: fields the conversation never
    mentioned come back null, extraction failures return an all-null profile
    rather than an error."""
    history = _get_history(session_id, customer_id)

    transcript = await asyncio.to_thread(_build_session_transcript, history)
    known_plans = await asyncio.to_thread(history._known_plans)
    known_cats = await asyncio.to_thread(history._known_categories)
    result = await asyncio.to_thread(extract_profile, transcript, known_plans, known_cats)

    return {"session_id": session_id, "customer_id": customer_id, "profile": result}


@app.post("/session/{session_id}/end")
async def end_session(session_id: str, customer_id: str):
    """Called when a call ends (proxied from auido_capture's /session/{id}/reset).
    Summarizes any remaining runtime history into the persistent summary DB and
    closes this session's Qdrant clients, then evicts it from the process-wide
    _histories cache so a long-running server doesn't accumulate open Qdrant
    clients across many calls. If this (session_id, customer_id) was never
    actually used, there's nothing to end — responds successfully with
    ended=False rather than a 404."""
    key = (session_id, customer_id)
    with _histories_lock:
        history = _histories.pop(key, None)

    if history is None:
        return {"session_id": session_id, "customer_id": customer_id, "ended": False}

    await asyncio.to_thread(history.end_session)
    await asyncio.to_thread(history.close)
    return {"session_id": session_id, "customer_id": customer_id, "ended": True}


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(_histories)}
