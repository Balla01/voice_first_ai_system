"""
main.py — Python Backend (FastAPI) box in the diagram.

Exposes:  wss://.../ws/audio?stream_id=mic|system&session_id=<id>

The browser opens TWO connections per session (one for mic, one for system
audio), both tagged with the same session_id. This file routes each
connection's bytes into the right AudioRouter/channel.

Run:
    uvicorn main:app --reload --port 8000

Env:
    DEEPGRAM_API_KEY=your_key_here   (put in a .env file, see .env.example)
"""

import os
import time
import base64
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()   # must run before importing twilio_voice — it reads Twilio env vars at module import time

from audio_router import AudioRouter
from turn_accumulator import TurnAccumulator
from errors import ClassifiedError, ErrorAction
from layer3 import SessionTracker, EpochCompactor, get_llm_client, Persistence
from layer4 import (
    TriggerGate,
    Tier2EmbeddingClassifier,
    GenerationController,
    GenerationManager,
    ToolRouter,
    get_router_client,
    build_default_registry,
    execute_tool_calls,
    ExecutionContext,
)
from layer5_client import Layer5Client
from twilio_voice import router as twilio_router
from profile_extractor import extract_profile, merge_profile, new_profile

from logging_config import setup_logging
setup_logging()
logger = logging.getLogger("insureassist.layer1")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
LAYER5_URL = os.getenv("LAYER5_URL", "http://127.0.0.1:8001")

# Layer 4 trigger mode: "tiers" = deterministic regex/embedding/heuristic gate
# (demo-safe default, reliability net). "router" = LLM tool-calling router.
# Flip this one env var to fall back if the router misbehaves. The router path
# ALSO falls back to the tiers per-turn on its own timeout/error.
LAYER4_TRIGGER_MODE = os.getenv("LAYER4_TRIGGER_MODE", "tiers").lower()

app = FastAPI(title="InsureAssist AI — Layer 1&2: Audio Capture + STT, Layer 3: Session Tracker")

# The UI (frontend/) is served from a different origin (e.g. Live Server :5500)
# than this API (:8000), so the browser needs CORS for the /email, /ask,
# /intent and /session REST calls. WebSockets aren't subject to CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(twilio_router)

# session_id -> AudioRouter / SessionTracker / TriggerGate  (created on first
# connection for a session, torn down when both mic and system sockets for
# that session close)
_sessions: dict[str, AudioRouter] = {}
_trackers: dict[str, SessionTracker] = {}
_gates: dict[str, TriggerGate] = {}
_routers: dict[str, ToolRouter] = {}   # only populated when LAYER4_TRIGGER_MODE == "router"
_accumulators: dict[str, TurnAccumulator] = {}   # debounces STT fragments into complete turns
_session_socket_count: dict[str, int] = {}

# Per-session customer profile (built live from speech) and manual intent
# override (set when the agent clicks an intent card). Both are cleared when the
# call ends (socket close or POST /session/{id}/reset).
_profiles: dict[str, dict] = {}
_manual_intents: dict[str, str | None] = {}
_profile_locks: dict[str, asyncio.Lock] = {}   # one extraction in flight per session

# The mic socket doubles as the UI's push channel — the frontend renders
# {"type": "transcript", ...} and {"type": "suggestion", ...} JSON frames
# pushed back over it (see frontend/capture-client.js). Only mic is tracked
# here since that's the only socket the UI listens on for server pushes.
_mic_sockets: dict[str, WebSocket] = {}

# Layer 3 shares ONE Postgres pool and ONE Anthropic client across all
# sessions — SessionTracker instances are per-session (they hold per-session
# state), but the underlying connections are expensive to set up per-session,
# so they're created once at app startup and injected into each tracker.
_persistence: Persistence | None = None
_compactor: EpochCompactor | None = None

# Layer 4's Tier 2 embedding classifier loads the MiniLM model once and holds
# no session-specific state itself — shared across all TriggerGate instances
# so we're not reloading the model per session.
_tier2_classifier: Tier2EmbeddingClassifier | None = None

# Layer 5 client (HTTP) and the abort controller that wraps every call to it.
# Both are stateless-enough to share across all sessions — GenerationController
# tracks per-session in-flight tasks internally via a dict keyed by session_id.
_layer5_client: Layer5Client | None = None
# Tiers path: abort-always (latest-wins). Router path: concurrent + queued, so
# separate questions asked during an in-flight answer all get answered.
_generation_controller = GenerationController()
_generation_manager = GenerationManager(
    max_concurrent=int(os.getenv("LAYER4_MAX_CONCURRENT_GENERATIONS", "3"))
)

# Layer 4 router (tool-calling). The tool registry holds no per-session state
# and the LLM client is a shared HTTP/SDK client, so both are created once at
# startup and shared across every ToolRouter instance. Only built when
# LAYER4_TRIGGER_MODE == "router".
_tool_registry = build_default_registry()
_router_llm = None

# Shared chat LLM for live profile extraction (same provider as Layer 3, chosen
# via LLM_PROVIDER). Built at startup; None if no provider key is configured, in
# which case profile-building simply stays empty (never crashes the call).
_profile_llm = None

# Runtime-mutable trigger mode. Starts at the env value but can be flipped live
# via POST /admin/trigger-mode WITHOUT restarting — the demo-day kill switch for
# a misbehaving router (router -> tiers takes effect on the very next turn,
# since the deterministic gate always exists per session).
_runtime_trigger_mode = LAYER4_TRIGGER_MODE


def _fmt(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s).strftime("%H:%M:%S.%f")[:-3]


async def _send_to_session(session_id: str, payload: dict) -> None:
    """Best-effort push of a JSON frame to the session's mic socket (the UI's
    push channel). Silently no-ops if the socket isn't open — a slow/dead
    frontend connection shouldn't break the STT/RAG pipeline."""
    ws = _mic_sockets.get(session_id)
    if ws is None:
        return
    try:
        await ws.send_json(payload)
    except Exception as e:
        logger.debug(f"[{session_id}] Failed to push {payload.get('type')} to UI: {e}")


async def _run_layer5_query(session_id: str, query_text: str) -> None:
    """
    Calls Layer 5's non-streaming /query endpoint (query(), not query_stream())
    since that's the mode that returns retrieval_time_s/llm_time_s/total_time_s —
    the timing breakdown we want to log. Wrapped by GenerationController: if a
    new trigger fires while this is still awaiting a response, it gets
    cancelled cleanly (verified — the in-flight HTTP request aborts properly,
    doesn't hang).
    """
    fire_time = time.time()
    logger.debug(f"[{session_id}] Layer 4 -> Layer 5: sending query {query_text!r}")

    try:
        response = await _layer5_client.query(query=query_text, session_id=session_id, customer_id=session_id)
        end_to_end_s = time.time() - fire_time

        logger.info(f"  <- Layer 5 answer: {response.answer}")

        def _fmt_s(v):
            return f"{v:.2f}s" if v is not None else "n/a"

        logger.info(
            f"     timing: retrieval={_fmt_s(response.retrieval_time_s)} | "
            f"llm={_fmt_s(response.llm_time_s)} | "
            f"api_total={_fmt_s(response.total_time_s)} | "
            f"end_to_end={end_to_end_s:.2f}s"
        )
        if response.total_time_s is not None:
            overhead_s = end_to_end_s - response.total_time_s
            logger.debug(
                f"[{session_id}] Layer5 network/overhead beyond API's own total_time_s: {overhead_s:.2f}s"
            )

        await _send_to_session(
            session_id, {"type": "suggestion", "query": query_text, "answer": response.answer}
        )
    except asyncio.CancelledError:
        logger.debug(f"[{session_id}] Layer 5 call aborted (a new trigger arrived mid-request)")
        raise
    except Exception as e:
        logger.error(f"[{session_id}] Layer 5 call failed: {e}")


async def _handle_tool_result(session_id: str, result, fire_time: float) -> None:
    """Router-mode analogue of _run_layer5_query's result handling: log the
    tool's answer + timing and push it to the UI. Called once per tool as it
    completes (multiple tools can fire on one turn)."""
    end_to_end_s = time.time() - fire_time
    if not result.ok:
        logger.error(f"[{session_id}] tool {result.tool} failed: {result.error}")
        return

    logger.info(f"  <- {result.tool} answer: {result.answer}")

    def _fmt_s(v):
        return f"{v:.2f}s" if v is not None else "n/a"

    m = result.meta
    logger.info(
        f"     timing: retrieval={_fmt_s(m.get('retrieval_time_s'))} | "
        f"llm={_fmt_s(m.get('llm_time_s'))} | "
        f"api_total={_fmt_s(m.get('total_time_s'))} | "
        f"end_to_end={end_to_end_s:.2f}s"
    )
    await _send_to_session(
        session_id,
        {"type": "suggestion", "query": result.query, "answer": result.answer, "tool": result.tool},
    )


async def _run_router_tools(session_id: str, tool_calls) -> None:
    """Execute the router's chosen tool calls. Wrapped by GenerationController,
    so a newer trigger cancels the whole batch cleanly."""
    fire_time = time.time()
    ctx = ExecutionContext(session_id=session_id, layer5_client=_layer5_client, customer_id=session_id)

    async def on_result(result):
        await _handle_tool_result(session_id, result, fire_time)

    try:
        await execute_tool_calls(tool_calls, _tool_registry, ctx, on_result)
    except asyncio.CancelledError:
        logger.debug(f"[{session_id}] router tool batch aborted (a new trigger arrived mid-request)")
        raise


@app.on_event("startup")
async def startup():
    global _persistence, _compactor, _tier2_classifier, _layer5_client, _router_llm, _profile_llm
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set — see .env.example")
    print("DATABASE_URL =", DATABASE_URL)
    _persistence = Persistence(dsn=DATABASE_URL)
    await _persistence.connect()   # also runs schema init (CREATE TABLE IF NOT EXISTS)

    _compactor = EpochCompactor(client=get_llm_client())   # provider chosen via LLM_PROVIDER env var

    try:
        _profile_llm = get_llm_client()   # reused for live customer-profile extraction
    except Exception as e:
        logger.warning(f"Profile extraction disabled — no LLM client: {e}")
        _profile_llm = None

    _tier2_classifier = Tier2EmbeddingClassifier()   # logs a warning + degrades gracefully if MiniLM can't load

    _layer5_client = Layer5Client(base_url=LAYER5_URL)

    # Build the router client whenever we CAN (not only when starting in router
    # mode), so the /admin/trigger-mode hot-switch can turn the router ON later
    # without a restart. If it can't be built (e.g. no OLLAMA_API_KEY), the
    # switch-to-router path is refused with a clear message and we stay on tiers.
    try:
        _router_llm = get_router_client()
        logger.info(f"Layer 4 router client ready (model={_router_llm.model})")
    except Exception as e:
        _router_llm = None
        logger.warning(f"Layer 4 router client unavailable ({e}); router mode disabled until fixed")

    if _runtime_trigger_mode == "router" and _router_llm is None:
        logger.warning("LAYER4_TRIGGER_MODE=router but router client unavailable -> running in TIERS")
    logger.info(f"Layer 4 trigger mode: {_runtime_trigger_mode.upper()}")

    logger.info("Layer 3, 4 & 5 ready: SQLite connected, LLM client initialized, trigger gate ready, Layer 5 client ready")


@app.on_event("shutdown")
async def shutdown():
    if _persistence:
        await _persistence.close()
    if _layer5_client:
        await _layer5_client.close()
    if _router_llm:
        await _router_llm.close()


def _make_segment_handler(session_id: str):
    """
    Two things happen per deduped final segment from TranscriptMerger:
      1. the transcript is pushed to the UI IMMEDIATELY (per fragment) so the
         Live Conversation panel stays snappy — no debounce delay;
      2. the segment is fed to the session's TurnAccumulator, which debounces
         STT fragments (~1.5s silence) into one complete turn before Layer 3
         memory + the Layer 4 router see it.
    Both are scheduled async so this callback stays synchronous and never
    blocks the STT event loop.
    """

    def _on_merged_segment(segment):
        asyncio.create_task(_push_transcript_now(session_id, segment))
        _accumulators[session_id].add_segment(segment)

    return _on_merged_segment


async def _push_transcript_now(session_id: str, segment) -> None:
    """Immediate, per-fragment transcript render (display only — memory and
    routing run off the debounced complete turn in _handle_merged_segment)."""
    await _send_to_session(
        session_id, {"type": "transcript", "speaker": segment.speaker, "text": segment.text}
    )


async def _maybe_update_profile(session_id: str, tracker) -> None:
    """Re-extract the customer profile from the running transcript and push any
    new facts to the UI. Best-effort: lock-guarded (one at a time per session),
    never raises, no-ops if the LLM client isn't configured."""
    if _profile_llm is None:
        return
    lock = _profile_locks.setdefault(session_id, asyncio.Lock())
    if lock.locked():
        return  # an extraction is already running; the next turn will refresh
    async with lock:
        transcript_text = tracker.get_formatted_context()
        incoming = await extract_profile(_profile_llm, transcript_text)
        if not incoming:
            return
        profile = _profiles.setdefault(session_id, new_profile())
        if merge_profile(profile, incoming):
            await _send_to_session(session_id, {"type": "profile", "profile": profile})


async def _handle_merged_segment(session_id: str, segment):
    tracker = _trackers[session_id]
    gate = _gates[session_id]
    spoken_at = segment.spoken_at if segment.spoken_at is not None else time.time()

    turn = await tracker.add_turn(segment.speaker, segment.text, spoken_at)
    if turn is None:
        logger.debug(f"Layer 3 dropped duplicate turn: [{segment.speaker}] {segment.text}")
        return

    # NOTE: transcript is already pushed to the UI per-fragment in
    # _push_transcript_now (immediate). This handler runs on the debounced
    # complete turn and only does memory + routing.

    if segment.spoken_at is not None and segment.transcribed_at is not None:
        logger.info(
            f"[{segment.speaker}] {segment.text}  "
            f"(spoken {_fmt(segment.spoken_at)} -> transcribed {_fmt(segment.transcribed_at)}, "
            f"latency {segment.latency_ms:.0f}ms)"
        )
    else:
        logger.info(f"[{segment.speaker}] {segment.text}")

    logger.debug(f"Layer 3 formatted context now:\n{tracker.get_formatted_context()}")

    # Build the customer profile live: whenever the CUSTOMER speaks, re-extract
    # facts from the running transcript and push any updates to the UI card.
    # Scheduled async + lock-guarded so it never blocks routing and only one
    # extraction runs per session at a time.
    if segment.speaker == "customer" and _profile_llm is not None:
        asyncio.create_task(_maybe_update_profile(session_id, tracker))

    # Layer 4: decide whether this turn is worth triggering Layer 5 on.
    # Reads the RUNTIME mode (flippable live via /admin/trigger-mode), not the
    # startup constant. Falls back to tiers if a router-mode session somehow has
    # no ToolRouter (e.g. router client failed to build).
    if _runtime_trigger_mode == "router" and session_id in _routers:
        await _handle_segment_router(session_id, tracker, segment, turn)
        return

    result = gate.check(speaker=segment.speaker, text=segment.text, is_final=True, now=time.time())

    # Push detected intents to the UI's intent cards whenever the classifier
    # matched something (not only on FIRE), so the cards stay responsive. The
    # agent can still manually override by clicking a card (POST /intent/override).
    detected = [m.intent for m in result.matches]
    if detected:
        await _send_to_session(session_id, {"type": "intent", "intents": detected, "source": "auto"})

    if result.action.value == "fire":
        await tracker.mark_important(turn)   # feeds back into Layer 3's IMPORTANT tagging
        intents = detected
        logger.info(f"  -> Layer 4 FIRE: {intents} -> calling Layer 5...")
        await _generation_controller.start_generation(
            session_id, lambda: _run_layer5_query(session_id, segment.text)
        )
    elif result.action.value == "refine":
        logger.info("  -> Layer 4 REFINE: agent asked to edit the last answer -> calling Layer 5...")
        # NOTE: sent as a plain follow-up query in the same session_id — this
        # assumes Layer 5 keeps its own conversation memory keyed by
        # session_id and will interpret "make it shorter" etc. as an edit
        # instruction on its last answer, since the /query contract has no
        # separate "refine" mode field. Revisit if Layer 5 needs an explicit
        # signal instead.
        await _generation_controller.start_generation(
            session_id, lambda: _run_layer5_query(session_id, segment.text)
        )
    else:
        logger.debug(f"Layer 4: no trigger ({result.reason})")


async def _handle_segment_router(session_id: str, tracker, segment, turn) -> None:
    """LAYER4_TRIGGER_MODE == 'router' path: the LLM picks the tool(s) to call.

    Continuation vs separate question (out-of-order, query-tagged, no holdback):
      - continuation of the in-flight question -> abort the latest generation
        and reissue the LLM's merged/enhanced query in its place;
      - separate question -> run it CONCURRENTLY (bounded/queued) so it gets
        answered too, surfaced as soon as it completes.
    """
    router = _routers[session_id]
    context = tracker.get_formatted_context()
    in_flight_query = _generation_manager.latest_query(session_id)
    decision = await router.route(
        speaker=segment.speaker, text=segment.text, context=context,
        now=time.time(), in_flight_query=in_flight_query,
    )

    if decision.action != "fire":
        logger.debug(f"Layer 4 router: no trigger ({decision.reason})")
        return

    await tracker.mark_important(turn)   # feeds back into Layer 3's IMPORTANT tagging
    tool_calls = decision.tool_calls
    tool_names = [c.name for c in tool_calls]
    # Query used for continuation tracking/tagging (first search call, else raw text).
    gen_query = next(
        (c.arguments.get("query") for c in tool_calls
         if c.name == "search_knowledge_base" and c.arguments.get("query")),
        segment.text,
    )

    if decision.is_continuation and _generation_manager.has_active(session_id):
        _generation_manager.abort_latest(session_id)
        logger.info(f"  -> Layer 4 router CONTINUATION: {tool_names} -> superseding in-flight, reissuing...")
    else:
        logger.info(f"  -> Layer 4 router FIRE via {decision.source}: {tool_names} -> executing (concurrent)...")

    _generation_manager.submit(
        session_id, gen_query, lambda: _run_router_tools(session_id, tool_calls)
    )


async def _get_or_create_router(
    session_id: str,
    system_encoding: str = "linear16",
    system_sample_rate: int = 16000,
) -> AudioRouter:
    """
    system_encoding/system_sample_rate only matter the FIRST time a session
    is created — whichever endpoint (ws_audio's "system" stream, or
    ws_twilio) connects first for a given session_id decides the "system"
    (customer) channel's audio format. In practice these two sources are
    mutually exclusive per session: a phone-call session only ever gets
    ws_twilio traffic on "system", a browser-only session only ever gets
    ws_audio traffic. Mixing the two within one session_id isn't supported.
    """
    if session_id not in _sessions:
        if not DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY is not set — see .env.example")

        tracker = SessionTracker(session_id=session_id, persistence=_persistence, compactor=_compactor)
        await tracker.load_history()   # no-op for a brand new session, rebuilds state on reconnect
        _trackers[session_id] = tracker

        # The deterministic gate is always created: it's the tiers-mode gate AND
        # the per-turn fallback the router delegates to on timeout/error.
        gate = TriggerGate(session_id=session_id, tier2_classifier=_tier2_classifier)
        _gates[session_id] = gate
        # Create the ToolRouter whenever a router client exists (regardless of the
        # current mode), so a live switch TO router works for sessions that were
        # opened while in tiers mode. Cheap: just holds refs + a CooldownTracker.
        if _router_llm is not None:
            _routers[session_id] = ToolRouter(
                session_id=session_id,
                llm_client=_router_llm,
                registry=_tool_registry,
                fallback_gate=gate,
            )

        # Debounce STT fragments into complete turns before Layer 3/Layer 4 see
        # them. Created before the AudioRouter so it exists when segments start
        # arriving. on_turn hands the completed turn to _handle_merged_segment.
        _accumulators[session_id] = TurnAccumulator(
            on_turn=lambda merged: _handle_merged_segment(session_id, merged)
        )

        router = AudioRouter(
            session_id=session_id,
            deepgram_api_key=DEEPGRAM_API_KEY,
            on_merged_final=_make_segment_handler(session_id),
            session_start_time=time.time(),
            system_encoding=system_encoding,
            system_sample_rate=system_sample_rate,
        )
        await router.start()
        _sessions[session_id] = router
        _session_socket_count[session_id] = 0
    return _sessions[session_id]


async def _decrement_and_maybe_close(session_id: str) -> None:
    """Shared teardown for both ws_audio and ws_twilio's finally blocks."""
    _session_socket_count[session_id] -= 1
    if _session_socket_count[session_id] <= 0:
        await _sessions[session_id].close()
        _sessions.pop(session_id, None)
        _trackers.pop(session_id, None)
        _gates.pop(session_id, None)
        _routers.pop(session_id, None)
        acc = _accumulators.pop(session_id, None)
        if acc is not None:
            acc.close()   # cancel any pending silence-flush timer
        _generation_controller.clear(session_id)
        _generation_manager.clear(session_id)   # cancel any in-flight/queued router generations
        _session_socket_count.pop(session_id, None)
        _mic_sockets.pop(session_id, None)
        _profiles.pop(session_id, None)
        _manual_intents.pop(session_id, None)
        _profile_locks.pop(session_id, None)
        logger.info(f"Session fully closed and cleaned up: {session_id}")


@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    """Called by the UI when the agent ends a call — drop the live profile and
    manual-intent state so a re-join starts clean, and tell Layer 5 to
    summarize + close its RuntimeHistory for this session (best-effort; never
    raises, see Layer5Client.end_session)."""
    _profiles.pop(session_id, None)
    _manual_intents.pop(session_id, None)
    if _layer5_client is not None:
        await _layer5_client.end_session(session_id=session_id, customer_id=session_id)
    return {"ok": True}


@app.post("/intent/override")
async def intent_override(payload: dict = Body(...)):
    """Agent clicked an intent card to override the auto-detected intent.
    Stored per session; `intent: null` clears the override (back to auto)."""
    session_id = payload.get("session_id") or ""
    _manual_intents[session_id] = payload.get("intent")
    return {"ok": True, "intent": _manual_intents[session_id]}


@app.post("/ask")
async def ask_ai(payload: dict = Body(...)):
    """Ad-hoc 'Ask AI' box in the sidebar — forwards a free-text question to
    Layer 5 (RAG) in the same session so it has call context. Degrades to a
    clear message if Layer 5 is unavailable.

    advanced_filter / ask_ai_session_id: pass-through for Layer 5's Ask-AI
    "thread" mode. When advanced_filter=True and ask_ai_session_id is
    omitted/None, Layer 5 mints a new thread id, returned below so the
    frontend can send it back on the next question in that thread.

    context_source: pass-through for resolving an ambiguous-reference
    clarification ("suggestion_card" | "current_thread") — see
    rag_pipeline/api.py. When the question is flagged ambiguous and
    context_source wasn't given, needs_clarification=True comes back below
    with no real answer; the frontend re-sends with context_source set."""
    question = (payload.get("question") or "").strip()
    session_id = payload.get("session_id") or "ask"
    advanced_filter = bool(payload.get("advanced_filter", False))
    ask_ai_session_id = payload.get("ask_ai_session_id")
    context_source = payload.get("context_source")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if _layer5_client is None:
        raise HTTPException(status_code=503, detail="RAG service not available")
    try:
        resp = await _layer5_client.query(
            query=question,
            session_id=session_id,
            customer_id=session_id,
            advanced_filter=advanced_filter,
            ask_ai_session_id=ask_ai_session_id,
            context_source=context_source,
        )
        return {
            "answer": resp.answer,
            "ask_ai_session_id": resp.ask_ai_session_id,
            "needs_clarification": resp.needs_clarification,
            "clarification_options": resp.clarification_options,
        }
    except Exception as e:
        logger.error(f"/ask failed: {e}")
        raise HTTPException(status_code=502, detail=f"Ask failed: {e}")


@app.get("/profile")
async def profile(session_id: str = Query(...)):
    """Proxies Layer 5's GET /profile so the frontend never talks to
    rag_pipeline directly (no CORS there — see layer5_client.py). No CRM, so
    customer_id=session_id, same convention as /ask."""
    if _layer5_client is None:
        raise HTTPException(status_code=503, detail="RAG service not available")
    try:
        return await _layer5_client.get_profile(session_id=session_id, customer_id=session_id)
    except Exception as e:
        logger.error(f"/profile failed: {e}")
        raise HTTPException(status_code=502, detail=f"Profile fetch failed: {e}")


@app.get("/ask-ai/sessions")
async def ask_ai_sessions(session_id: str = Query(...)):
    """Proxies Layer 5's GET /ask-ai/sessions — lists this call's Ask-AI
    threads (most-recent-first) for the frontend's thread switcher. No CRM,
    so customer_id=session_id, same convention as /ask and /profile."""
    if _layer5_client is None:
        raise HTTPException(status_code=503, detail="RAG service not available")
    try:
        return await _layer5_client.list_ask_ai_sessions(customer_id=session_id)
    except Exception as e:
        logger.error(f"/ask-ai/sessions failed: {e}")
        raise HTTPException(status_code=502, detail=f"Ask-AI sessions fetch failed: {e}")


@app.get("/ask-ai/thread")
async def ask_ai_thread(session_id: str = Query(...), ask_ai_session_id: str = Query(...)):
    """Proxies Layer 5's GET /ask-ai/thread — full history of one Ask-AI
    thread, for reopening it in the UI."""
    if _layer5_client is None:
        raise HTTPException(status_code=503, detail="RAG service not available")
    try:
        return await _layer5_client.get_ask_ai_thread(customer_id=session_id, ask_ai_session_id=ask_ai_session_id)
    except Exception as e:
        logger.error(f"/ask-ai/thread failed: {e}")
        raise HTTPException(status_code=502, detail=f"Ask-AI thread fetch failed: {e}")


def _send_gmail(to: str, subject: str, body: str) -> None:
    """Blocking SMTP send via Gmail. Requires GMAIL_ADDRESS + GMAIL_APP_PASSWORD
    (a Google App Password, not the account password) in the environment.
    Called via asyncio.to_thread so it doesn't block the event loop."""
    import smtplib
    from email.message import EmailMessage

    sender = os.getenv("GMAIL_ADDRESS", "")
    app_password = os.getenv("GMAIL_APP_PASSWORD", "")
    if not sender or not app_password:
        raise RuntimeError(
            "Email not configured — set GMAIL_ADDRESS and GMAIL_APP_PASSWORD "
            "(a Google App Password) in the backend .env"
        )

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, app_password)
        server.send_message(msg)


@app.post("/email/send")
async def email_send(payload: dict = Body(...)):
    """Send the follow-up email via Gmail (SMTP + App Password)."""
    to = (payload.get("to") or "").strip()
    subject = payload.get("subject") or "Follow-up from your InsureAssist agent"
    body = payload.get("body") or ""
    if not to:
        raise HTTPException(status_code=400, detail="recipient 'to' is required")
    try:
        await asyncio.to_thread(_send_gmail, to, subject, body)
        return {"ok": True}
    except Exception as e:
        logger.error(f"/email/send failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws/audio")
async def ws_audio(
    websocket: WebSocket,
    stream_id: str = Query(..., description="mic | system"),
    session_id: str = Query(...),
):
    await websocket.accept()
    logger.info(f"WS connected: session={session_id} stream={stream_id}")

    try:
        router = await _get_or_create_router(session_id)
    except (RuntimeError, ClassifiedError) as e:
        # Auth (401) or startup failure -> fatal, close immediately (no retry)
        logger.error(f"Fatal startup error for session {session_id}: {e}")
        await websocket.close(code=4401)
        return
    except Exception as e:
        # Covers Layer 3 failures (e.g. Postgres briefly unreachable) that
        # aren't already classified above.
        logger.error(f"Unexpected startup error for session {session_id}: {e}")
        await websocket.close(code=4500)
        return

    _session_socket_count[session_id] += 1
    if stream_id == "mic":
        _mic_sockets[session_id] = websocket

    try:
        while True:
            pcm_bytes = await websocket.receive_bytes()
            await router.handle_pcm(stream_id, pcm_bytes)
    except WebSocketDisconnect:
        logger.info(f"WS disconnected: session={session_id} stream={stream_id}")
    except ClassifiedError as e:
        if e.action == ErrorAction.FATAL_NO_RETRY:
            logger.error(f"Fatal error, closing session {session_id}: {e}")
        else:
            logger.warning(f"Retryable error surfaced to client {session_id}: {e}")
        await websocket.close(code=4500)
    finally:
        if stream_id == "mic" and _mic_sockets.get(session_id) is websocket:
            _mic_sockets.pop(session_id, None)
        await _decrement_and_maybe_close(session_id)


@app.websocket("/ws/twilio")
async def ws_twilio(websocket: WebSocket, session_id: str = Query(...)):
    """
    Twilio Media Streams — the phone-call equivalent of ws_audio's "system"
    (customer) leg. Twilio speaks its own JSON-over-WebSocket protocol
    (connected/start/media/stop events, base64-encoded mulaw payloads)
    rather than raw binary PCM frames, so it gets its own handler instead of
    reusing ws_audio. See twilio_voice.py for how a call gets routed here
    (session_id == the call's CallSid).
    """
    await websocket.accept()
    logger.info(f"Twilio WS connected: session={session_id}")

    try:
        router = await _get_or_create_router(session_id, system_encoding="mulaw", system_sample_rate=8000)
    except (RuntimeError, ClassifiedError) as e:
        logger.error(f"Fatal startup error for Twilio session {session_id}: {e}")
        await websocket.close(code=4401)
        return
    except Exception as e:
        logger.error(f"Unexpected startup error for Twilio session {session_id}: {e}")
        await websocket.close(code=4500)
        return

    _session_socket_count[session_id] += 1

    try:
        while True:
            message = await websocket.receive_json()
            event = message.get("event")

            if event == "media":
                mulaw_bytes = base64.b64decode(message["media"]["payload"])
                await router.handle_pcm("system", mulaw_bytes)
            elif event == "start":
                call_sid = message.get("start", {}).get("callSid")
                logger.info(f"[{session_id}] Twilio stream started (callSid={call_sid})")
            elif event == "stop":
                logger.info(f"[{session_id}] Twilio stream stopped")
                break
    except WebSocketDisconnect:
        logger.info(f"Twilio WS disconnected: session={session_id}")
    except ClassifiedError as e:
        if e.action == ErrorAction.FATAL_NO_RETRY:
            logger.error(f"Fatal error, closing Twilio session {session_id}: {e}")
        else:
            logger.warning(f"Retryable error on Twilio session {session_id}: {e}")
    finally:
        await _decrement_and_maybe_close(session_id)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
        "active_trackers": len(_trackers),
        "trigger_mode": _runtime_trigger_mode,
        "router_available": _router_llm is not None,
        "router_model": getattr(_router_llm, "model", None),
    }


@app.get("/admin/trigger-mode")
async def get_trigger_mode():
    return {"trigger_mode": _runtime_trigger_mode, "router_available": _router_llm is not None}


@app.post("/admin/trigger-mode")
async def set_trigger_mode(mode: str = Query(..., description="tiers | router")):
    """Demo-day kill switch: flip Layer 4 between the LLM router and the
    deterministic tiers live, no restart. Takes effect on the next turn for
    every session. Switching to 'router' is refused if the router client
    couldn't be built (e.g. missing key) — we stay on tiers rather than
    silently no-op.

    NOTE: intentionally unauthenticated, same posture as the rest of this
    demo backend (see /twilio/call in CLAUDE.md). Gate before any public
    deployment.
    """
    global _runtime_trigger_mode
    mode = mode.lower()
    if mode not in ("tiers", "router"):
        return {"ok": False, "error": f"invalid mode {mode!r} (expected tiers|router)"}
    if mode == "router" and _router_llm is None:
        return {"ok": False, "error": "router client unavailable; staying on tiers",
                "trigger_mode": _runtime_trigger_mode}
    previous = _runtime_trigger_mode
    _runtime_trigger_mode = mode
    logger.warning(f"Layer 4 trigger mode switched live: {previous} -> {mode}")
    return {"ok": True, "trigger_mode": _runtime_trigger_mode, "previous": previous}
