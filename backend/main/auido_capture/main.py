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
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from dotenv import load_dotenv

from audio_router import AudioRouter
from errors import ClassifiedError, ErrorAction
from layer3 import SessionTracker, EpochCompactor, get_llm_client, Persistence
from layer4 import TriggerGate, Tier2EmbeddingClassifier, GenerationController
from layer5_client import Layer5Client

load_dotenv()

from logging_config import setup_logging
setup_logging()
logger = logging.getLogger("insureassist.layer1")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
LAYER5_URL = os.getenv("LAYER5_URL", "http://127.0.0.1:8001")

app = FastAPI(title="InsureAssist AI — Layer 1&2: Audio Capture + STT, Layer 3: Session Tracker")

# session_id -> AudioRouter / SessionTracker / TriggerGate  (created on first
# connection for a session, torn down when both mic and system sockets for
# that session close)
_sessions: dict[str, AudioRouter] = {}
_trackers: dict[str, SessionTracker] = {}
_gates: dict[str, TriggerGate] = {}
_session_socket_count: dict[str, int] = {}

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
_generation_controller = GenerationController()


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


@app.on_event("startup")
async def startup():
    global _persistence, _compactor, _tier2_classifier, _layer5_client
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set — see .env.example")

    _persistence = Persistence(dsn=DATABASE_URL)
    await _persistence.connect()   # also runs schema init (CREATE TABLE IF NOT EXISTS)

    _compactor = EpochCompactor(client=get_llm_client())   # provider chosen via LLM_PROVIDER env var

    _tier2_classifier = Tier2EmbeddingClassifier()   # logs a warning + degrades gracefully if MiniLM can't load

    _layer5_client = Layer5Client(base_url=LAYER5_URL)

    logger.info("Layer 3, 4 & 5 ready: Postgres connected, LLM client initialized, trigger gate ready, Layer 5 client ready")


@app.on_event("shutdown")
async def shutdown():
    if _persistence:
        await _persistence.close()
    if _layer5_client:
        await _layer5_client.close()


def _make_segment_handler(session_id: str):
    """
    Bridges the synchronous on_merged_final callback (called directly inside
    Deepgram's async message handler — see transcript_merger.py) into Layer 3's
    async SessionTracker.add_turn(), without blocking the STT event loop.
    """

    def _on_merged_segment(segment):
        asyncio.create_task(_handle_merged_segment(session_id, segment))

    return _on_merged_segment


async def _handle_merged_segment(session_id: str, segment):
    tracker = _trackers[session_id]
    gate = _gates[session_id]
    spoken_at = segment.spoken_at if segment.spoken_at is not None else time.time()

    turn = await tracker.add_turn(segment.speaker, segment.text, spoken_at)
    if turn is None:
        logger.debug(f"Layer 3 dropped duplicate turn: [{segment.speaker}] {segment.text}")
        return

    await _send_to_session(
        session_id, {"type": "transcript", "speaker": segment.speaker, "text": segment.text}
    )

    if segment.spoken_at is not None and segment.transcribed_at is not None:
        logger.info(
            f"[{segment.speaker}] {segment.text}  "
            f"(spoken {_fmt(segment.spoken_at)} -> transcribed {_fmt(segment.transcribed_at)}, "
            f"latency {segment.latency_ms:.0f}ms)"
        )
    else:
        logger.info(f"[{segment.speaker}] {segment.text}")

    logger.debug(f"Layer 3 formatted context now:\n{tracker.get_formatted_context()}")

    # Layer 4: decide whether this turn is worth triggering Layer 5 on.
    result = gate.check(speaker=segment.speaker, text=segment.text, is_final=True, now=time.time())

    if result.action.value == "fire":
        await tracker.mark_important(turn)   # feeds back into Layer 3's IMPORTANT tagging
        intents = [m.intent for m in result.matches]
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


async def _get_or_create_router(session_id: str) -> AudioRouter:
    if session_id not in _sessions:
        if not DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY is not set — see .env.example")

        tracker = SessionTracker(session_id=session_id, persistence=_persistence, compactor=_compactor)
        await tracker.load_history()   # no-op for a brand new session, rebuilds state on reconnect
        _trackers[session_id] = tracker

        _gates[session_id] = TriggerGate(session_id=session_id, tier2_classifier=_tier2_classifier)

        router = AudioRouter(
            session_id=session_id,
            deepgram_api_key=DEEPGRAM_API_KEY,
            on_merged_final=_make_segment_handler(session_id),
            session_start_time=time.time(),
        )
        await router.start()
        _sessions[session_id] = router
        _session_socket_count[session_id] = 0
    return _sessions[session_id]


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
        _session_socket_count[session_id] -= 1
        if _session_socket_count[session_id] <= 0:
            await router.close()
            _sessions.pop(session_id, None)
            _trackers.pop(session_id, None)
            _gates.pop(session_id, None)
            _session_socket_count.pop(session_id, None)
            _mic_sockets.pop(session_id, None)
            logger.info(f"Session fully closed and cleaned up: {session_id}")


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(_sessions), "active_trackers": len(_trackers)}
