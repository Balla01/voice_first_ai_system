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
from layer4 import TriggerGate, Tier2EmbeddingClassifier

load_dotenv()

from logging_config import setup_logging
setup_logging()
logger = logging.getLogger("insureassist.layer1")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

app = FastAPI(title="InsureAssist AI — Layer 1&2: Audio Capture + STT, Layer 3: Session Tracker")

# session_id -> AudioRouter / SessionTracker / TriggerGate  (created on first
# connection for a session, torn down when both mic and system sockets for
# that session close)
_sessions: dict[str, AudioRouter] = {}
_trackers: dict[str, SessionTracker] = {}
_gates: dict[str, TriggerGate] = {}
_session_socket_count: dict[str, int] = {}

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


def _fmt(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s).strftime("%H:%M:%S.%f")[:-3]


@app.on_event("startup")
async def startup():
    global _persistence, _compactor, _tier2_classifier
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set — see .env.example")

    _persistence = Persistence(dsn=DATABASE_URL)
    await _persistence.connect()   # also runs schema init (CREATE TABLE IF NOT EXISTS)

    _compactor = EpochCompactor(client=get_llm_client())   # provider chosen via LLM_PROVIDER env var

    _tier2_classifier = Tier2EmbeddingClassifier()   # logs a warning + degrades gracefully if MiniLM can't load
    logger.info("Layer 3 & 4 ready: Postgres connected, LLM client initialized, trigger gate ready")


@app.on_event("shutdown")
async def shutdown():
    if _persistence:
        await _persistence.close()


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
    # Layer 5 (RAG + Prompt Builder) doesn't exist yet, so for now we just
    # log the decision — this is the hook point it plugs into next.
    result = gate.check(speaker=segment.speaker, text=segment.text, is_final=True, now=time.time())

    if result.action.value == "fire":
        await tracker.mark_important(turn)   # feeds back into Layer 3's IMPORTANT tagging
        intents = [m.intent for m in result.matches]
        logger.info(f"  -> Layer 4 FIRE: {intents} (would call Layer 5 here)")
    elif result.action.value == "refine":
        logger.info("  -> Layer 4 REFINE: agent asked to edit the last answer (would call Layer 5 here)")
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
        _session_socket_count[session_id] -= 1
        if _session_socket_count[session_id] <= 0:
            await router.close()
            _sessions.pop(session_id, None)
            _trackers.pop(session_id, None)
            _gates.pop(session_id, None)
            _session_socket_count.pop(session_id, None)
            logger.info(f"Session fully closed and cleaned up: {session_id}")


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(_sessions), "active_trackers": len(_trackers)}