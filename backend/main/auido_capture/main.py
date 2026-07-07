"""
main.py — Python Backend (FastAPI) box in the diagram.

Exposes:  wss://.../ws/audio?stream_id=mic|system&session_id=<id>

The browser opens TWO connections per session (one for mic, one for system
audio), both tagged with the same session_id. This file routes each
connection's bytes into the right AudioRouter/channel.

Layer 3 wiring: each session's merged transcript segments feed a
TurnAccumulator, which buffers customer speech into full turns (flushed on
silence or agent speaker-switch) and hands each turn to a LiveRagWorker
(rag_bridge.py -> main/src/main.py's RAG pipeline, unmodified). The
generated answer is pushed back to the browser over the session's own
"mic" WebSocket as JSON: {"type": "suggestion", "query": ..., "answer": ...}.

Run:
    uvicorn main:app --port 8000
    (avoid --reload if you rely on "model loads before the API is reachable":
    --reload runs a separate parent "reloader" supervisor process that prints
    "Uvicorn running on http://..." immediately, before the actual child
    server process even starts — that line is NOT a readiness signal in
    --reload mode. The real signal, in both modes, is the
    "===> API READY <===" line below, printed only after the child process's
    lifespan/warmup has fully completed.)

Env:
    DEEPGRAM_API_KEY=your_key_here   (put in a .env file, see .env.example)
"""

import asyncio
import os
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from dotenv import load_dotenv

from audio_router import AudioRouter
from errors import ClassifiedError, ErrorAction
from turn_accumulator import TurnAccumulator
from rag_bridge import LiveRagWorker, rag_main

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insureassist.layer1")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Load the embedding model now, before the ASGI app finishes startup and
    uvicorn's *worker* process begins accepting connections — otherwise it
    lazy-loads on the FIRST real customer turn of the FIRST session, adding
    ~7-25s to that turn's latency.

    NOTE on --reload: uvicorn's reload supervisor prints "Uvicorn running on
    http://..." from its own PARENT process immediately on launch, before
    this function (which only runs in the CHILD worker process) is even
    called. That line is not a readiness signal when --reload is used — the
    "===> API READY <===" line below is, in both --reload and plain mode.
    """
    logger.info("Warming up embedding model — API is NOT ready yet...")
    t0 = time.time()
    await asyncio.to_thread(rag_main._embed, ["warmup"])
    logger.info(f"Embedding model ready ({time.time() - t0:.1f}s).")
    logger.info("===> API READY <=== now accepting connections.")
    yield


app = FastAPI(title="InsureAssist AI", lifespan=lifespan)

# session_id -> AudioRouter  (created on first connection for a session,
# torn down when both mic and system sockets for that session close)
_sessions: dict[str, AudioRouter] = {}
_session_socket_count: dict[str, int] = {}
_rag_workers: dict[str, LiveRagWorker] = {}
# session_id -> {stream_id: WebSocket} — lets us push suggestions back over
# a specific stream (mic) without double-delivering to both mic+system sockets,
# which belong to the same browser page.
_session_sockets: dict[str, dict[str, WebSocket]] = {}

# Holds references to fire-and-forget tasks (transcript pushes, etc.) so they
# aren't garbage-collected mid-flight — asyncio only holds a weak reference
# to a task once nothing else does, per the create_task() docs' own warning.
_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro):
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _fmt(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s).strftime("%H:%M:%S.%f")[:-3]


def _log_segment(segment):
    if segment.spoken_at is not None and segment.transcribed_at is not None:
        logger.info(
            f"[{segment.speaker}] {segment.text}  "
            f"(spoken {_fmt(segment.spoken_at)} -> transcribed {_fmt(segment.transcribed_at)}, "
            f"latency {segment.latency_ms:.0f}ms)"
        )
    else:
        logger.info(f"[{segment.speaker}] {segment.text}")


async def _send_to_session(session_id: str, payload: dict):
    """Push a JSON message back to the browser over the session's mic socket
    (falling back to whatever's open if mic isn't connected)."""
    sockets = _session_sockets.get(session_id, {})
    ws = sockets.get("mic") or next(iter(sockets.values()), None)
    if ws is None:
        logger.warning(f"No open socket to deliver suggestion for session {session_id}")
        return
    try:
        await ws.send_json(payload)
    except Exception as e:
        logger.warning(f"Failed to send suggestion to session {session_id}: {e}")


async def _get_or_create_router(session_id: str) -> AudioRouter:
    if session_id not in _sessions:
        if not DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY is not set — see .env.example")

        async def on_answer(query: str, answer: str):
            await _send_to_session(session_id, {
                "type": "suggestion",
                "query": query,
                "answer": answer,
            })

        # No separate CRM/customer identity is wired through yet, so the
        # browser-generated session_id doubles as both session_id and
        # customer_id for RuntimeHistory's scoping.
        rag_worker = LiveRagWorker(session_id=session_id, customer_id=session_id, on_answer=on_answer)
        _rag_workers[session_id] = rag_worker

        accumulator = TurnAccumulator(
            on_customer_turn=rag_worker.handle_customer_turn,
            on_agent_segment=rag_worker.handle_agent_segment,
        )

        def on_merged_final(segment):
            _log_segment(segment)  # keep existing console visibility
            _fire_and_forget(_send_to_session(session_id, {
                "type": "transcript",
                "speaker": segment.speaker,  # "agent" | "customer"
                "text": segment.text,
            }))
            accumulator.add_segment(segment)

        router = AudioRouter(
            session_id=session_id,
            deepgram_api_key=DEEPGRAM_API_KEY,
            on_merged_final=on_merged_final,
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
    _session_sockets.setdefault(session_id, {})[stream_id] = websocket

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
        sockets = _session_sockets.get(session_id)
        if sockets and sockets.get(stream_id) is websocket:
            del sockets[stream_id]
        if _session_socket_count[session_id] <= 0:
            await router.close()
            _sessions.pop(session_id, None)
            _session_socket_count.pop(session_id, None)
            _session_sockets.pop(session_id, None)
            worker = _rag_workers.pop(session_id, None)
            if worker:
                await asyncio.to_thread(worker.close)
            logger.info(f"Session fully closed and cleaned up: {session_id}")


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(_sessions)}