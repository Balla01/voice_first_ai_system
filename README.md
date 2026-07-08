# InsureAssist AI — Layer 1 & 2: Audio Capture + STT

Implements exactly the flow in your diagram:

```
Browser (Chrome)
  Microphone ──┐
               ├─▶ PCM Converter ─▶ WebSocket client (+session_id) ─▶ backend
  System Audio ┘
                       WSS | stream_id: mic/system | session_id header

Python Backend (FastAPI)
  Audio Router (per session_id)
    ├─ mic stream    ─▶ VAD (adaptive RMS) ─▶ Deepgram WS (speaker=agent)    ─┐
    └─ system stream ─▶ VAD (adaptive RMS) ─▶ Deepgram WS (speaker=customer)─┼─▶ Transcript Merger
                                                                              │   (dedup 500ms, finals only)
  Error Classification: Auth 401 → fatal no retry | Network/5xx → retry x5 backoff
```

## 1. Backend setup

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your real DEEPGRAM_API_KEY into .env
uvicorn main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/health`

## 2. Frontend demo

Serve the `frontend/` folder over HTTPS or `localhost` (required for
`getUserMedia`/`getDisplayMedia`), e.g.:

```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500`, click **Start Capture**, grant mic permission,
then choose a tab/window to share for system audio. Watch the backend
terminal — merged transcript segments print as `[agent] ...` / `[customer] ...`.

**Note:** update `BACKEND_WS_BASE` in `capture-client.js` to match wherever
you deploy the FastAPI app (`ws://` for local http, `wss://` for https).

## 3. What each file maps to on the diagram

| Diagram box            | File                          |
|-------------------------|-------------------------------|
| Microphone / System Audio | `frontend/capture-client.js` (`getUserMedia` / `getDisplayMedia`) |
| PCM Converter            | `frontend/pcm-worklet-processor.js` |
| WebSocket client + session_id | `frontend/capture-client.js` (`AudioStreamPipeline`) |
| Audio Router              | `backend/audio_router.py` |
| VAD                       | `backend/vad.py` |
| Deepgram WS (mic/system)  | `backend/stt_deepgram.py` |
| Transcript Merger         | `backend/transcript_merger.py` |
| Error Classification      | `backend/errors.py` |
| FastAPI wiring            | `backend/main.py` |

## 4. Next layer (not in this diagram)

`main.py`'s `_print_merged_segment()` is the hook point — replace it with a
call into your Layer 3 (RAG / policy retrieval over Qdrant) so merged final
segments feed the LLM context in real time instead of just logging.

## 5. Known shortcuts taken for hackathon speed (call these out if judges ask)

- VAD is a simple adaptive-RMS detector, not webrtcvad/silero — swap-in
  upgrade later if time allows, interface is unchanged.
- `getDisplayMedia` is used as a stand-in for "system audio" capture; for a
  real phone-based sales call you'd instead tap the softphone/PBX audio leg
  server-side (e.g. via Twilio Media Streams) rather than screen-share audio.
- No auth on the WebSocket endpoint yet — add a token check in `main.py`
  before the hackathon demo if you're presenting over a public URL.