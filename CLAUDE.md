# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

InsureAssist AI — a live-call insurance assistant. Three independent services, no shared build system, no root package manifest. `insureassist-react/` is an old reference UI, not part of current work — ignore it unless the user explicitly asks about it.

## Architecture

```
frontend (vanilla JS, :5500) → auido_capture (FastAPI, :8000) → rag_pipeline (FastAPI, :8001)
                                       ↕ Twilio, Deepgram, Gmail SMTP        ↕ Qdrant, Groq
```

- **`frontend/`** — plain HTML/CSS/JS, no build step. Talks to `auido_capture` only (WebSocket for audio, REST for `/ask`, `/ask-ai/sessions`, `/ask-ai/thread`, `/profile`, `/session/{id}/reset`). It never calls `rag_pipeline` directly — `auido_capture` proxies those calls to avoid CORS on 8001.
- **`auido_capture/`** — ingests mic/system audio over WebSocket, transcribes via Deepgram, and runs a layered pipeline:
  - Layer 3 (`layer3/`): session context window, dedup, token budgeting, Claude-based epoch summarization, SQLite persistence (`layer3_history.db`, tables `turns` + `epoch_summaries`).
  - Layer 4 (`layer4/`): "Smart Trigger Gate" deciding when to fire RAG — `tiers` mode (regex → MiniLM embedding → heuristic, deterministic) or `router` mode (LLM tool-calling), toggled via `LAYER4_TRIGGER_MODE` env var or live via `POST /admin/trigger-mode`.
  - Layer 5 (`layer5_client.py`): thin HTTP client to `rag_pipeline` (`LAYER5_URL`, default `http://127.0.0.1:8001`).
  - Also owns Twilio voice webhooks (`/twilio/voice`, `/twilio/call`, `/ws/twilio`) and outbound email (`/email/send` via Gmail SMTP).
- **`rag_pipeline/`** — the RAG backend. `api.py` is the FastAPI wrapper; `main.py` holds the actual retrieval/LLM logic (parallel search across 3 Qdrant collections, recency-weighted re-ranking, Groq completion). Vector store is Qdrant (embedded, on-disk under `rag_pipeline/data/` and `rag_pipeline/vector_data_docs/`, both gitignored): `lic_insurance_docs_v2` (hybrid dense+sparse BGE-M3 docs), `runtime_history` (in-memory per-session), `session_summaries` (persistent). A separate `chat_bot_ask_ai` collection (`ask_ai_store.py`) backs ChatGPT-style "Ask-AI" threads. LLM is Groq (`llama-3.1-8b-instant`); embeddings are `Alibaba-NLP/gte-large-en-v1.5` (history) and `BAAI/bge-m3` (docs).

Session correlation across all three services is by `session_id`/`customer_id` string passed through the HTTP chain — there's no shared database.

## Running locally

Three separate terminals, no fixed order:

```
# frontend
cd frontend && python serve.py            # http://localhost:5500

# auido_capture
cd auido_capture && uvicorn main:app --reload --port 8000

# rag_pipeline (from rag_pipeline/ root — NOT rag_pipeline/src, that path in api.py's docstring is stale)
cd rag_pipeline && uvicorn api:app --reload --port 8001
```

For `rag_pipeline`, use **Python 3.11** with the loose `requirements.txt` (not `requirements.docker.txt`, which is a frozen pip-freeze snapshot for the Docker image and drifts from `requirements.txt`).

Data ingestion (separate from the API): `python run_data_dump.py` (PDF → chunks.json) then `python run_pipeline.py` (PDF → embed → Qdrant), both in `rag_pipeline/`.

## Env vars (each service has its own gitignored `.env`, no `.env.example` committed)

- `auido_capture/.env`: `DEEPGRAM_API_KEY`, `DATABASE_URL` (legacy — see gotcha below), `LLM_PROVIDER` (anthropic|openai|gemini|ollama) + matching API key, `TWILIO_ACCOUNT_SID`/`TWILIO_AUTH_TOKEN`/`TWILIO_PHONE_NUMBER`, `AGENT_PHONE_NUMBER`, `PUBLIC_BASE_URL`, `FRONTEND_URL`.
- `rag_pipeline/.env`: `deep_gram_key`, `groq_api`, `email_sender_address`, `email_sender_app_password` (note: lowercase/underscore names, inconsistent with `auido_capture`'s convention).

## Known gotchas

- `auido_capture` was migrated from Postgres (Neon, via asyncpg) to local SQLite for dev. `DATABASE_URL` is still read but a leftover `postgres://` value silently falls back to `./layer3_history.db` with just a warning — don't assume Postgres is in play.
- `rag_pipeline/data_dump/constants.py` hardcodes an absolute Windows path (`C:\projects\audio_transition_projects\data\LIC`) for the LIC document folder walker — this only works on the original author's machine; treat as a config gap, not a bug to silently "fix" by guessing a new path.
- `rag_pipeline/requirements.txt` and `requirements.docker.txt` are out of sync (different `fastapi`/`uvicorn`/`qdrant-client` versions, extra packages like `docling`, `FlagEmbedding`, `torch` only in the Docker build). Don't assume the Docker freeze reflects what's needed for local dev, or vice versa.
- Secrets: never add real API keys/tokens to any tracked file. `auido_capture/voice_agent_twilio_converation.txt` currently has a live Twilio SID/token committed in git — flagged for rotation/history-scrub, not something to copy as a pattern.
