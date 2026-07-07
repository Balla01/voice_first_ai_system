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
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from dotenv import load_dotenv

from audio_router import AudioRouter
from errors import ClassifiedError, ErrorAction

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insureassist.layer1")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

app = FastAPI(title="InsureAssist AI")

# session_id -> AudioRouter  (created on first connection for a session,
# torn down when both mic and system sockets for that session close)
_sessions: dict[str, AudioRouter] = {}
_session_socket_count: dict[str, int] = {}


def _fmt(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s).strftime("%H:%M:%S.%f")[:-3]


def _print_merged_segment(segment):
    # Placeholder for "final segments -> Transcript Merger -> to context".
    # In the real app, this hands off to Layer 3 (RAG / policy lookup).
    if segment.spoken_at is not None and segment.transcribed_at is not None:
        logger.info(
            f"[{segment.speaker}] {segment.text}  "
            f"(spoken {_fmt(segment.spoken_at)} -> transcribed {_fmt(segment.transcribed_at)}, "
            f"latency {segment.latency_ms:.0f}ms)"
        )
    else:
        logger.info(f"[{segment.speaker}] {segment.text}")


async def _get_or_create_router(session_id: str) -> AudioRouter:
    if session_id not in _sessions:
        if not DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY is not set — see .env.example")
        router = AudioRouter(
            session_id=session_id,
            deepgram_api_key=DEEPGRAM_API_KEY,
            on_merged_final=_print_merged_segment,
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
            _session_socket_count.pop(session_id, None)
            logger.info(f"Session fully closed and cleaned up: {session_id}")


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(_sessions)}