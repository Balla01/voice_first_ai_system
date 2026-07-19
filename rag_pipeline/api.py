"""
api.py — FastAPI wrapper around the RAG pipeline (main.py).

Two entirely independent flows share this one /query endpoint, distinguished
by advanced_filter:

  advanced_filter=False -> the AI Copilot's own auto-triggered suggestions
  during a live call. Call-scoped via RuntimeHistory (session_id + customer_id
  — today always both set to the call's session id, no separate customer
  identity — see auido_capture). Context = recent turns + semantically
  relevant history/summaries for THIS call + insurance docs. See
  _retrieve_context().

  advanced_filter=True -> Ask-AI. Owned entirely by agent_id (the logged-in
  sales agent), NOT by any call/session — an agent can use Ask-AI with no
  call active at all, exactly like a standalone chatbot. Threads
  (ask_ai_session_id) are independent of each other (no cross-thread memory,
  ChatGPT-"New Chat"-style) and persist permanently per agent_id in
  chat_bot_ask_ai (ask_ai_store.py) — reopening one later shows its full
  history. Context = insurance docs + this thread's own chat_bot_ask_ai
  memory + a live web search when the query needs it (classify_web_search).
  Never touches RuntimeHistory's session-scoped recent_turns/history/summary,
  and never calls history.add() — Ask-AI's only persisted state is
  {query, answer} in its own thread. See _thread_context().

  Optional live_context: built by the CALLER (frontend, from its own live
  transcript/suggestion/profile state — see the "Include Current Call
  Context" toggle) and appended to the prompt for that ONE request only. Never
  written to chat_bot_ask_ai or anywhere else — the thread's permanent memory
  never includes it.

POST /query
  body:   {query, stream, advanced_filter, agent_id, ask_ai_session_id,
           live_context, session_id, customer_id}
  session_id/customer_id only matter (and are required) when
  advanced_filter=False. agent_id is required when advanced_filter=True.
  stream=False -> single JSON: {answer, retrieval_time_s, llm_time_s, total_time_s, ask_ai_session_id}
  stream=True  -> text/event-stream: one "data:" event per token as it
                  arrives from Groq, then a final "event: done" carrying the
                  full answer + timing breakdown + ask_ai_session_id.
  Omit ask_ai_session_id to start a new thread; the server mints one and
  returns it for the client to reuse on the next message in that thread.

GET /ask-ai/sessions?agent_id=...
  Lists this agent's Ask-AI threads (most-recent-first) for a session switcher.

GET /ask-ai/thread?agent_id=...&ask_ai_session_id=...
  Full {query, answer, timestamp} history of one Ask-AI thread, chronological —
  for reopening a past thread and replaying it in the UI (see /ask-ai/sessions
  for the list of thread ids to choose from).

DELETE /ask-ai/thread?agent_id=...&ask_ai_session_id=...
  Permanently deletes one Ask-AI thread. Idempotent.

GET /profile?session_id=...&customer_id=...
  Builds a customer profile (name, age, profession, location, policy_product,
  category) from all available context for that session — earlier summaries
  plus recent raw turns — via profile_extractor.py. Best-effort: unset fields
  come back null, never an error.

POST /session/{session_id}/end?customer_id=...
  Called when a call ends: summarizes any remaining runtime history for this
  (session_id, customer_id) into the persistent summary DB, then evicts it
  from the process-wide RuntimeHistory cache (the underlying Qdrant clients
  are NOT closed here — see below). No-ops successfully (not an error) if
  that session was never actually used. Unrelated to Ask-AI/agent_id.

One RuntimeHistory per (session_id, customer_id) — or per (agent_id, agent_id)
for the Ask-AI docs-only lookup, see _thread_context() — created on first use
and cached for the life of the process for the session-scoped bookkeeping
(_id_counter, caches) — but its history/summary/docs Qdrant clients are
process-wide singletons shared by every RuntimeHistory instance (see
history_pipeline.RuntimeHistory._get_shared_clients), because they all point
at the same on-disk storage folders (rows are scoped by a payload filter, not
by separate storage) and Qdrant's embedded mode only allows one open handle
per folder — two sessions each opening their own client to the same folder
used to crash with "already accessed by another instance of Qdrant client"
the moment a second session was alive concurrently. AskAIStore
(chat_bot_ask_ai) was already a single process-wide instance for the same
underlying reason, since its threads are scoped by agent_id + ask_ai_session_id,
not by call session_id.

Run:
    cd rag_pipeline/src
    uvicorn api:app --reload --port 8001
"""

import asyncio
import json
import logging
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client.models import Filter, FieldCondition, MatchValue

sys.path.insert(0, str(Path(__file__).resolve().parent))

from history.history_pipeline import RuntimeHistory, _embed, _get_model
from data_dump.embedder import _get_model as _get_docs_embedding_model
from main import parallel_search, build_context, call_llm, stream_llm_chunks
from query_understanding import classify_web_search
from web_search_call import web_search_answer
from ask_ai_store import AskAIStore
from profile_extractor import extract_profile
from email_trigger import detect_email_request
from email_sending_test import send_email
from constants import DEBUG, MAX_HISTORY_CHUNKS, EVICT_COUNT, DOCS_SEARCH_K
# ambiguous_reference.is_ambiguous_reference is deliberately NOT imported —
# Ask-AI no longer has any ambiguity to resolve (it never touches RuntimeHistory
# automatically; live call context is an explicit toggle, not inferred from
# wording). Kept in the repo, unused, in case it's needed again later.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(title="RAG Pipeline API")

_histories: Dict[Tuple[str, str], RuntimeHistory] = {}
_histories_lock = Lock()

# chat_bot_ask_ai (advanced_filter mode): one store for the whole process,
# NOT keyed by call session_id — see ask_ai_store.py for why threads are
# scoped to (agent_id, ask_ai_session_id) instead.
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
# though the actual dispatch (once the agent confirms — see
# needs_email_confirmation/confirm_email) is handled entirely independently of
# what the LLM says. This note tells the LLM that instruction is handled
# elsewhere, so it should just answer the real question.
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
    # Force both embedding models to load now (each is normally lazy, on
    # first use) so the app doesn't report ready until it's actually ready to
    # serve a query at full speed — and so a first-ever cold-start model
    # download (BGE-M3 is ~3GB) shows up clearly in the startup logs instead
    # of silently blocking a live agent's first Ask-AI question.
    await asyncio.to_thread(_get_model)                # gte-large — runtime_history/session_summaries
    await asyncio.to_thread(_get_docs_embedding_model)  # BGE-M3 — insurance docs (search_docs_scored)
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
        _histories.clear()
    # history/summary/docs Qdrant clients are process-wide singletons shared
    # by every session (see RuntimeHistory._get_shared_clients) — close them
    # once here, not per-session (individual RuntimeHistory.close() is a no-op).
    RuntimeHistory.close_shared_clients()
    if _ask_ai_store is not None:
        _ask_ai_store.close()


class QueryRequest(BaseModel):
    query: str
    stream: bool = False
    # Required when advanced_filter=False (the AI Copilot's own call-scoped
    # suggestions), ignored otherwise. Today always both set to the call's
    # session id by auido_capture (no separate customer identity) — see
    # module docstring.
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    # Optional explicit docs metadata filter. When any is set, it overrides the
    # LLM-derived query filter (search_docs_scored's auto path is skipped).
    plan_name: Optional[str] = None
    doc_type: Optional[str] = None
    product_type: Optional[str] = None
    tenant_id: Optional[str] = None
    # When True: this is Ask-AI, not the AI Copilot — see module docstring.
    # agent_id becomes required; session_id/customer_id are ignored.
    advanced_filter: bool = False
    # Required when advanced_filter=True. The logged-in sales agent's stable
    # identity (from auido_capture's /auth/login) — Ask-AI threads belong to
    # this, never to any call/session. See module docstring.
    agent_id: Optional[str] = None
    # Ask-AI "thread" id (only meaningful when advanced_filter=True) — like a
    # ChatGPT conversation id. Omit on the first message of a new thread; the
    # server mints one and returns it in QueryResponse so the client can pass
    # it back on subsequent messages in the same thread. Pass a previously
    # returned id to continue that thread, or a different one to switch threads.
    # Threads are fully independent of each other — no cross-thread memory.
    ask_ai_session_id: Optional[str] = None
    # Optional, advanced_filter=True only: the current call's live transcript/
    # latest suggestion/customer profile, built by the CALLER (frontend's
    # "Include Current Call Context" toggle) and appended to the prompt for
    # THIS request only. Never persisted — chat_bot_ask_ai only ever saves
    # {query, answer}, never this.
    live_context: Optional[str] = None
    # Resolves an email-send confirmation (see needs_email_confirmation below).
    # Leave False on a fresh query. Set True + confirmed_answer (the exact
    # answer text the agent reviewed and approved) only when re-sending after
    # the agent confirmed — this sends verbatim rather than re-running
    # retrieval/LLM a second time, so what gets emailed always matches exactly
    # what the agent saw and approved.
    confirm_email: bool = False
    confirmed_answer: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    retrieval_time_s: float
    llm_time_s: float
    total_time_s: float
    # Echoed/minted only when advanced_filter=True — persist this and send it
    # back as ask_ai_session_id on the next call to stay in the same thread.
    ask_ai_session_id: Optional[str] = None
    # Set only once the email has actually been dispatched (confirm_email=True
    # round-trip) — the recipient the answer was sent to. Fire-and-forget:
    # this means "queued", not "confirmed delivered" — actual success/failure
    # only appears in server logs ([advanced:email] sent/failed), never as an
    # API error.
    emailed_to: Optional[str] = None
    # True when the query was detected as an email-send request
    # (email_trigger.detect_email_request) but hasn't been sent yet — answer
    # still holds the drafted text for the agent to review. Re-send the same
    # query with confirm_email=True + confirmed_answer=<this answer> to
    # actually dispatch it, or just don't — nothing is sent unless confirmed.
    needs_email_confirmation: bool = False
    pending_email: Optional[dict] = None


def _build_explicit_filter(request: "QueryRequest"):
    """Build a Qdrant Filter from explicit request fields, or None if none set."""
    conds = []
    for key in ("plan_name", "doc_type", "product_type", "tenant_id"):
        val = getattr(request, key, None)
        if val:
            conds.append(FieldCondition(key=key, match=MatchValue(value=val)))
    return Filter(must=conds) if conds else None


def _retrieve_context(query: str, history: RuntimeHistory, doc_filter=None) -> Tuple[str, Dict]:
    """Embed + search + rerank + build_context — same steps as main.py's main(),
    synchronous/blocking. Used ONLY for the AI Copilot's own call-scoped
    suggestions (advanced_filter=False) — Ask-AI (advanced_filter=True) uses
    _thread_context() instead and never reaches this function; see module
    docstring.

    Returns (context, timing) — timing carries per-stage elapsed ms (see
    _log_pipeline_timing) plus retrieval-count stats, for the structured
    per-request timing log built in query()."""
    timing: Dict = {}

    t0 = time.perf_counter()
    query_vec = _embed([query])[0]
    timing["query_embed_gte_ms"] = (time.perf_counter() - t0) * 1000

    recent_turns = history.get_recent_history(n=5)
    history_ranked, summary_ranked, docs_ranked = parallel_search(
        query, query_vec, history, doc_filter=doc_filter, timing=timing
    )

    logger.info(f"[retrieval:docs_filter] explicit={'yes' if doc_filter is not None else 'no (auto LLM filter)'}")
    logger.info(f"[retrieval:recent_history] {len(recent_turns)} turn(s) (chronological, not reranked)")
    _log_chunks("history", history_ranked)
    _log_chunks("summary", summary_ranked)
    # docs get full-text logging when DEBUG=True (constants.py) — the other
    # two collections stay truncated since they're rarely what you're debugging.
    _log_chunks("docs", docs_ranked, full=DEBUG)

    t_prompt0 = time.perf_counter()
    context = build_context(recent_turns, history_ranked, summary_ranked, docs_ranked, top_k=CONTEXT_TOP_K)
    timing["prompt_construction_ms"] = (time.perf_counter() - t_prompt0) * 1000

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

    # No intent/trigger gate exists inside rag_pipeline for this flow either —
    # every /query request that reaches here always embeds (BGE-M3) and
    # searches the docs collection, regardless of query type. (Layer4's own
    # "Smart Trigger Gate" lives upstream in auido_capture and decides whether
    # to call this endpoint AT ALL for a given transcript turn — it is not a
    # gate inside this function.)
    timing["trigger_gate_ms"] = 0.0
    timing["trigger_decision"] = "USE_RAG"
    timing["trigger_reason"] = (
        "No intent/trigger gate in rag_pipeline — docs retrieval always runs. "
        "(auido_capture's Layer4 gate, if any, decides upstream whether to call this endpoint at all.)"
    )

    timing["history_chunks"] = len(history_ranked)
    timing["summary_chunks"] = len(summary_ranked)
    timing["docs_chunks"] = len(docs_ranked)
    timing["docs_sent_to_llm"] = used_docs

    return context, timing


# Deterministic (regex-only, no LLM) pre-gate for Ask-AI's docs/web retrieval —
# same rationale as email_trigger.py/ambiguous_reference.py: this is a
# keyword-shaped signal ("is this just small talk?"), not a semantic judgment
# call, so a free/instant regex beats spending a Groq round-trip (or worse, a
# BGE-M3 embed + Qdrant query) to find out a greeting needs neither.
#
# Deliberately EXCLUDE-only: only skips retrieval for a narrow, well-defined
# set of non-informational patterns (greetings/thanks/casual acknowledgements
# and thread-mechanics questions like "what did I ask before?"). Anything else
# — including odd or unrecognized phrasing — falls through to the existing
# safe default of actually searching, so a real insurance question is never
# accidentally starved of context just because the gate didn't recognize it.
_GREETING_WORDS = (
    r"hi|hello+|hey+|hiya|yo|namaste|"
    r"good\s*(morning|afternoon|evening|night)|"
    r"thanks?|thank\s*you|thx|ty|"
    r"bye|goodbye|see\s*you|take\s*care|"
    r"ok(ay)?|cool|great|nice|got\s*it|sounds\s*good|alright|"
    r"how\s*are\s*you|what'?s\s*up"
)
_GREETING_RE = re.compile(rf"^\s*({_GREETING_WORDS})\s*[!.,?]*\s*$", re.IGNORECASE)

_THREAD_MEMORY_RE = re.compile(
    r"\b(what (did|have|has) (i|you|we) (ask|say|tell|talk)(ed|ing)?|"
    r"what (was|is) (my|your|the|our)?\s*(last|previous|first)\s*(question|message|answer|query)|"
    r"how many (question|message)s? (have|did|has) (i|you|we) (ask|send)|"
    r"summarize (this|our) (conversation|chat|thread)|"
    r"what.{0,20}(talk(ed)?|discuss(ed)?).{0,15}about|"
    r"is this (a )?new thread)\b",
    re.IGNORECASE,
)


def _needs_external_retrieval(query: str) -> Tuple[bool, str]:
    """False for greetings/thanks/casual chit-chat and thread-memory questions
    ("what did I ask before?", "is this a new thread?") — these need only the
    LLM plus thread history (already always included, separately from this
    gate). True for everything else, including real insurance questions,
    off-domain questions that might need web search, and anything the gate
    doesn't specifically recognize (safe default — see module comment above).

    Returns (needs_retrieval, reason) — reason is a short human-readable label
    for the structured per-request timing log (see _log_pipeline_timing)."""
    q = query.strip()
    if _GREETING_RE.match(q):
        return False, "Greeting/casual chit-chat (regex gate)"
    if _THREAD_MEMORY_RE.search(q):
        return False, "Thread-memory question — answered from thread history only (regex gate)"
    return True, "Needs external retrieval — insurance/product question or unrecognized phrasing (safe default)"


def _thread_context(query: str, history: RuntimeHistory, agent_id: str, ask_ai_session_id: str,
                     doc_filter=None, live_context: Optional[str] = None) -> Tuple[str, Dict]:
    """The ONLY context builder for Ask-AI (advanced_filter=True): insurance
    product docs (global knowledge, not session-scoped) + this thread's own
    chat_bot_ask_ai memory (recent + semantically relevant past Q&A, scoped by
    agent_id + ask_ai_session_id) + a live web search when the query needs it.

    `history` is a RuntimeHistory instance used ONLY for its shared docs_client
    (search_docs_scored) — its session-scoped recent_turns/history/summary
    methods are deliberately never called here, and .add() is never called on
    it for Ask-AI turns. This is what makes Ask-AI fully decoupled from any
    call/session: the only identity that matters is agent_id.

    live_context (optional): the current call's live transcript/suggestion/
    profile, built by the caller from its own live state (see the frontend's
    "Include Current Call Context" toggle) — appended as a clearly-labeled
    TEMPORARY section for this one request only. Never saved to chat_bot_ask_ai
    or anywhere else.

    Docs search and web-search classification are both skipped entirely for
    greetings/thanks/casual chat and thread-memory questions — see
    _needs_external_retrieval(). Thread memory (recent_qas/qa_ranked below)
    always runs regardless — it's cheap (local Qdrant, no LLM call) and is
    exactly what a memory question like "what did I ask before?" needs.

    Returns (context, timing) — timing carries per-stage elapsed ms plus the
    gate's decision/reason and retrieval-count stats, for the structured
    per-request timing log built in query() (_log_pipeline_timing).
    """
    timing: Dict = {}

    t0 = time.perf_counter()
    query_vec = _embed([query])[0]
    timing["query_embed_gte_ms"] = (time.perf_counter() - t0) * 1000

    t_gate0 = time.perf_counter()
    do_retrieval, gate_reason = _needs_external_retrieval(query)
    timing["trigger_gate_ms"] = (time.perf_counter() - t_gate0) * 1000
    timing["trigger_decision"] = "USE_RAG" if do_retrieval else "SKIP_RAG"
    timing["trigger_reason"] = gate_reason
    logger.info(f"[thread:retrieval_gate] query={query!r} needs_retrieval={do_retrieval} reason={gate_reason!r}")

    context = ""
    if do_retrieval:
        docs_timing: Dict = {}
        t_docs0 = time.perf_counter()
        docs_results = history.search_docs_scored(query, DOCS_SEARCH_K, doc_filter, timing=docs_timing)
        timing["docs_retrieval_ms"] = (time.perf_counter() - t_docs0) * 1000
        timing.update(docs_timing)
        docs_ranked = [(text, score) for text, score, _ in docs_results]
        _log_chunks("docs", docs_ranked, full=DEBUG)

        t_prompt0 = time.perf_counter()
        context = build_context([], [], [], docs_ranked, top_k=CONTEXT_TOP_K)
        timing["prompt_construction_ms"] = (time.perf_counter() - t_prompt0) * 1000
        timing["docs_chunks"] = len(docs_ranked)
        timing["docs_sent_to_llm"] = min(len(docs_ranked), CONTEXT_TOP_K)
    else:
        # Gate skipped retrieval entirely — BGE-M3 embed/Qdrant search/rerank
        # never ran for this request, so every docs sub-timing is genuinely 0.
        timing["docs_retrieval_ms"] = 0.0
        timing["embed_ms"] = 0.0
        timing["qdrant_ms"] = 0.0
        timing["merge_ms"] = 0.0
        timing["rerank_ms"] = 0.0
        timing["prompt_construction_ms"] = 0.0
        timing["docs_chunks"] = 0
        timing["docs_sent_to_llm"] = 0

    extra = []

    t_hist0 = time.perf_counter()
    recent_qas = _ask_ai_store.get_recent(agent_id, ask_ai_session_id, n=5)
    timing["history_retrieval_ms"] = (time.perf_counter() - t_hist0) * 1000
    logger.info(f"[thread] agent={agent_id} thread={ask_ai_session_id} recent={len(recent_qas)} turn(s)")
    if recent_qas:
        extra.append("--- Recent turns in this Ask-AI thread (chronological) ---")
        extra.extend(f"  {qa}" for qa in recent_qas)

    t_summ0 = time.perf_counter()
    qa_ranked = _ask_ai_store.search_relevant(agent_id, ask_ai_session_id, query_vec, k=CONTEXT_TOP_K)
    timing["summary_retrieval_ms"] = (time.perf_counter() - t_summ0) * 1000
    _log_chunks("thread_relevant", [(text, score) for text, score, _ts in qa_ranked])
    if qa_ranked:
        extra.append("--- Semantically relevant past turns in this thread ---")
        extra.extend(f"  • {text}" for text, _score, _ts in qa_ranked)

    timing["history_chunks"] = len(recent_qas)
    timing["summary_chunks"] = len(qa_ranked)

    if do_retrieval:
        t_web_gate0 = time.perf_counter()
        web_triggered = classify_web_search(query)
        timing["web_search_gate_ms"] = (time.perf_counter() - t_web_gate0) * 1000
        logger.info(f"[thread:web_search_trigger] {web_triggered}")
        if web_triggered:
            t_web0 = time.perf_counter()
            try:
                web_text = web_search_answer(query)
            except Exception as e:
                logger.warning(f"[thread:web_search] failed: {e}")
                web_text = ""
            timing["web_search_ms"] = (time.perf_counter() - t_web0) * 1000
            if web_text:
                extra.append("--- Live web search results ---")
                extra.append(f"  {web_text}")
    else:
        logger.info("[thread:web_search_trigger] skipped (retrieval gate)")

    if live_context:
        extra.append("--- Live call context (temporary — this request only, not saved) ---")
        extra.append(live_context)

    t_prompt1 = time.perf_counter()
    if extra:
        context += ("\n" if context else "") + "\n".join(extra)
    timing["prompt_construction_ms"] += (time.perf_counter() - t_prompt1) * 1000

    return context, timing


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


def _log_pipeline_timing(query: str, session_label: str, timing: Dict, total_ms: float) -> None:
    """Structured, comparable-across-requests timing log for one /query
    request (both advanced_filter modes). All *_ms values come from
    time.perf_counter() deltas captured at each stage; a stage that never ran
    for this request (e.g. docs retrieval skipped by the Ask-AI trigger gate,
    or reranking when USE_RERANKER is off) defaults to 0.0/0 rather than being
    omitted, so every request logs the same fixed set of lines and stays
    diffable across requests."""

    def ms(key: str) -> float:
        return timing.get(key, 0.0)

    def count(key: str) -> int:
        return timing.get(key, 0)

    lines = [
        "================ ASK AI REQUEST ================",
        f'Query              : "{query}"',
        f"Session            : {session_label}",
        "",
        "TIMING",
        "-----------------------------------------------",
        f"Intent / Trigger Gate      : {ms('trigger_gate_ms'):.1f} ms",
        f"History Retrieval          : {ms('history_retrieval_ms'):.1f} ms",
        f"Summary Retrieval          : {ms('summary_retrieval_ms'):.1f} ms",
        f"Query Embedding (BGE-M3)   : {ms('embed_ms'):.1f} ms",
        f"Docs Retrieval (Qdrant)    : {ms('qdrant_ms'):.1f} ms",
        f"Docs Reranking             : {ms('rerank_ms'):.1f} ms",
        f"Prompt Construction        : {ms('prompt_construction_ms'):.1f} ms",
        f"Groq API                   : {ms('groq_ms'):.1f} ms",
        f"Response Parsing           : {ms('response_parsing_ms'):.1f} ms",
        "",
        f"TOTAL BACKEND              : {total_ms:.1f} ms",
        "===============================================",
        "",
        f"Trigger Decision : {timing.get('trigger_decision', 'USE_RAG')}",
        f"Reason           : {timing.get('trigger_reason', 'n/a')}",
        "",
        f"History Chunks Retrieved : {count('history_chunks')}",
        f"Summary Chunks Retrieved : {count('summary_chunks')}",
        f"Docs Retrieved           : {count('docs_chunks')}",
        f"Docs Sent to LLM         : {count('docs_sent_to_llm')}",
        "",
        f"Embedding Time           : {ms('embed_ms'):.1f} ms",
        f"Qdrant Search Time       : {ms('qdrant_ms'):.1f} ms",
        f"Hybrid Merge Time        : {ms('merge_ms'):.1f} ms",
        f"Rerank Time              : {ms('rerank_ms'):.1f} ms",
    ]
    logger.info("\n" + "\n".join(lines))


@app.post("/query")
async def query(request: QueryRequest):
    # A new thread id is minted here (not left to the client) so the first
    # message of an Ask-AI conversation doesn't need one pre-generated —
    # mirrors ChatGPT starting a new conversation on the first message.
    ask_ai_session_id = None
    if request.advanced_filter:
        if not request.agent_id:
            raise HTTPException(status_code=400, detail="agent_id is required when advanced_filter=True")
        ask_ai_session_id = request.ask_ai_session_id or str(uuid.uuid4())

    # Email confirmation round-trip: the agent already reviewed confirmed_answer
    # (returned as this same query's answer on the first pass, with
    # needs_email_confirmation=True) and approved sending it. Short-circuits
    # BEFORE retrieval/LLM — sends verbatim rather than regenerating the answer
    # a second time, which could otherwise drift from what was actually shown
    # to and approved by the agent. No chat_bot_ask_ai save here either: the
    # first pass already saved this {query, answer} pair.
    if request.advanced_filter and request.confirm_email and request.confirmed_answer:
        email_recipient = detect_email_request(request.query)
        if email_recipient:
            _send_answer_email(email_recipient, request.query, request.confirmed_answer)
        return QueryResponse(
            answer=request.confirmed_answer,
            retrieval_time_s=0.0, llm_time_s=0.0, total_time_s=0.0,
            ask_ai_session_id=ask_ai_session_id,
            emailed_to=email_recipient,
        )

    t_start = time.perf_counter()
    doc_filter = _build_explicit_filter(request)

    # Ask-AI (advanced_filter=True) is owned by agent_id alone, fully decoupled
    # from any call/session — see module docstring. The AI Copilot's own
    # auto-triggered suggestions (advanced_filter=False) stay call-scoped via
    # RuntimeHistory, unchanged.
    if request.advanced_filter:
        session_label = f"agent={request.agent_id} thread={ask_ai_session_id}"
        # RuntimeHistory here is used ONLY for its shared docs_client (product
        # knowledge search, global and session-independent) — never its
        # session-scoped recent_turns/history/summary methods, and .add() is
        # never called on it for Ask-AI turns.
        history = _get_history(request.agent_id, request.agent_id)
        context, timing = await asyncio.to_thread(
            _thread_context, request.query, history, request.agent_id, ask_ai_session_id,
            doc_filter, request.live_context,
        )
    else:
        if not request.session_id or not request.customer_id:
            raise HTTPException(
                status_code=400, detail="session_id and customer_id are required when advanced_filter=False"
            )
        session_label = f"session={request.session_id} customer={request.customer_id}"
        history = _get_history(request.session_id, request.customer_id)
        context, timing = await asyncio.to_thread(_retrieve_context, request.query, history, doc_filter)
    t_retrieval = time.perf_counter()

    # Detected BEFORE the LLM call (not after, as originally) so the system
    # prompt can be adjusted when the query itself contains an email-send
    # instruction — otherwise the raw query text reads to the LLM as something
    # it must fulfill itself, and it responds with an "I can't send emails"
    # disclaimer instead of the actual answer. See EMAIL_HANDLING_NOTE.
    t_prompt0 = time.perf_counter()
    email_recipient = detect_email_request(request.query) if request.advanced_filter else None
    system_prompt = None
    if request.advanced_filter:
        system_prompt = ADVANCED_SYSTEM_PROMPT + (EMAIL_HANDLING_NOTE if email_recipient else "")
    timing["prompt_construction_ms"] = timing.get("prompt_construction_ms", 0.0) + (time.perf_counter() - t_prompt0) * 1000

    if request.stream:
        return StreamingResponse(
            _stream_answer(request.query, context, t_start, t_retrieval, timing, session_label,
                           system_prompt, request.advanced_filter, history,
                           request.agent_id, ask_ai_session_id, email_recipient),
            media_type="text/event-stream",
        )

    t_groq0 = time.perf_counter()
    answer = await asyncio.to_thread(call_llm, request.query, context, system_prompt)
    t_groq1 = time.perf_counter()
    timing["groq_ms"] = (t_groq1 - t_groq0) * 1000

    t_parse0 = time.perf_counter()
    if not request.advanced_filter:
        await asyncio.to_thread(history.add, "user", request.query)
        await asyncio.to_thread(history.add, "assistant", answer)

    needs_email_confirmation = False
    pending_email = None
    if request.advanced_filter:
        await asyncio.to_thread(_ask_ai_store.save, request.agent_id, ask_ai_session_id, request.query, answer)
        if email_recipient:
            # Don't send yet — the agent needs to confirm first (see
            # QueryRequest.confirm_email). answer already holds the drafted
            # text for them to review.
            needs_email_confirmation = True
            pending_email = {"to": email_recipient, "subject": _derive_email_subject(request.query)}
    t_end = time.perf_counter()
    timing["response_parsing_ms"] = (t_end - t_parse0) * 1000

    _log_pipeline_timing(request.query, session_label, timing, (t_end - t_start) * 1000)

    return QueryResponse(
        answer=answer,
        retrieval_time_s=round(t_retrieval - t_start, 3),
        llm_time_s=round(t_end - t_retrieval, 3),
        total_time_s=round(t_end - t_start, 3),
        ask_ai_session_id=ask_ai_session_id,
        needs_email_confirmation=needs_email_confirmation,
        pending_email=pending_email,
    )


def _stream_answer(query: str, context: str, t_start: float, t_retrieval: float, timing: Dict, session_label: str,
                   system_prompt: str = None, advanced_filter: bool = False,
                   history: RuntimeHistory = None, agent_id: str = None, ask_ai_session_id: str = None,
                   email_recipient: str = None):
    """Sync generator (Starlette runs it in a threadpool) — yields SSE token events, then a final done event."""
    t_groq0 = time.perf_counter()
    parts = []
    for token in stream_llm_chunks(query, context, system_prompt):
        parts.append(token)
        yield f"data: {json.dumps({'token': token})}\n\n"

    answer = "".join(parts)
    t_groq1 = time.perf_counter()
    timing["groq_ms"] = (t_groq1 - t_groq0) * 1000

    t_parse0 = time.perf_counter()
    if not advanced_filter and history is not None:
        history.add("user", query)
        history.add("assistant", answer)

    needs_email_confirmation = False
    pending_email = None
    if advanced_filter:
        _ask_ai_store.save(agent_id, ask_ai_session_id, query, answer)
        if email_recipient:
            # Don't send yet — mirrors the non-streaming /query path: the
            # agent must resend with confirm_email=True + confirmed_answer to
            # actually dispatch it.
            needs_email_confirmation = True
            pending_email = {"to": email_recipient, "subject": _derive_email_subject(query)}
    t_end = time.perf_counter()
    timing["response_parsing_ms"] = (t_end - t_parse0) * 1000

    _log_pipeline_timing(query, session_label, timing, (t_end - t_start) * 1000)

    done_payload = {
        "answer": answer,
        "retrieval_time_s": round(t_retrieval - t_start, 3),
        "llm_time_s": round(t_end - t_retrieval, 3),
        "total_time_s": round(t_end - t_start, 3),
        "ask_ai_session_id": ask_ai_session_id,
        "needs_email_confirmation": needs_email_confirmation,
        "pending_email": pending_email,
    }
    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"


@app.get("/ask-ai/sessions")
async def ask_ai_sessions(agent_id: str):
    """List this agent's Ask-AI threads (most-recent-first) — powers a
    ChatGPT-style session switcher in the frontend."""
    sessions = await asyncio.to_thread(_ask_ai_store.list_sessions, agent_id)
    return {"agent_id": agent_id, "sessions": sessions}


@app.get("/ask-ai/thread")
async def ask_ai_thread(agent_id: str, ask_ai_session_id: str):
    """Full chronological {query, answer, timestamp} history of one Ask-AI
    thread — for reopening a thread picked from /ask-ai/sessions and replaying
    it in the UI."""
    turns = await asyncio.to_thread(_ask_ai_store.get_thread, agent_id, ask_ai_session_id)
    return {"agent_id": agent_id, "ask_ai_session_id": ask_ai_session_id, "turns": turns}


@app.delete("/ask-ai/thread")
async def delete_ask_ai_thread(agent_id: str, ask_ai_session_id: str):
    """Permanently deletes one Ask-AI thread. Scoped by BOTH agent_id and
    ask_ai_session_id (see AskAIStore.delete_thread) so an agent can only ever
    delete their own threads. Idempotent — deleting an already-gone/unknown
    thread still returns ok=True rather than a 404."""
    await asyncio.to_thread(_ask_ai_store.delete_thread, agent_id, ask_ai_session_id)
    return {"ok": True, "agent_id": agent_id, "ask_ai_session_id": ask_ai_session_id}


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
