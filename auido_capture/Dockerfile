# auido_capture — FastAPI audio-capture/orchestration service (Layer 3/4/5 glue).
# No heavy ML deps (the old MiniLM tier-2 embedding step is gone — see
# layer4/intent_tier3_heuristic.py), so this image is much lighter than
# rag_pipeline's.
#
# Build:
#   docker build -t auido-capture:latest .
# Run (secrets passed at runtime, not baked into the image — see .env for the
# full list: DEEPGRAM_API_KEY, LLM_PROVIDER + matching key, TWILIO_* , etc.):
#   docker run --rm -p 8000:8000 --env-file .env auido-capture:latest
#
# LAYER5_URL defaults to http://127.0.0.1:8001, which is wrong inside a
# container — point it at the rag_pipeline service (see docker-compose.yml,
# which sets LAYER5_URL=http://rag_pipeline:8001 automatically).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code. .dockerignore keeps __pycache__/.env/layer3_history.db*
# out of the image — the SQLite DB is created fresh on first run unless a
# volume is mounted over /app/layer3_history.db (see docker-compose.yml).
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
