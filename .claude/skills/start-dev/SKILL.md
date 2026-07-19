---
name: start-dev
description: Start all three InsureAssist AI services (frontend, auido_capture, rag_pipeline) for local development. Use when the user asks to run, start, or spin up the app locally.
disable-model-invocation: true
---

Start the three services in separate background processes, each in its own working directory. Order doesn't matter — they don't block on each other at startup.

1. **rag_pipeline** (port 8001):
   ```
   cd rag_pipeline && uvicorn api:app --reload --port 8001
   ```
   Requires Python 3.11 with `rag_pipeline/requirements.txt` installed, and a `rag_pipeline/.env` with `deep_gram_key`, `groq_api`, `email_sender_address`, `email_sender_app_password`.

2. **auido_capture** (port 8000):
   ```
   cd auido_capture && uvicorn main:app --reload --port 8000
   ```
   Requires an `auido_capture/.env` with at least `DEEPGRAM_API_KEY` and `LLM_PROVIDER` + its matching API key. Talks to rag_pipeline over `LAYER5_URL` (defaults to `http://127.0.0.1:8001`, so start rag_pipeline on 8001 for the default to work).

3. **frontend** (port 5500):
   ```
   cd frontend && python serve.py
   ```

Run each with `run_in_background: true` (Bash tool) so all three stay up simultaneously. After starting, report the three URLs:
- Frontend: http://localhost:5500
- auido_capture: http://localhost:8000
- rag_pipeline: http://localhost:8001

If a `.env` file is missing for a service, say so and stop rather than starting a half-configured service.
