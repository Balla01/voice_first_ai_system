# RAG pipeline API — FastAPI (api.py) + Qdrant on-disk + Groq LLM.
# Mirrors the C:\voice_assistant_project\venv environment (pip freeze), minus
# Windows-only packages, with CPU-only torch.
#
# Build:
#   docker build -t rag-pipeline:latest .
# Run (Groq key passed at runtime, not baked into the image):
#   docker run --rm -p 8001:8001 -e groq_api=gsk_xxx rag-pipeline:latest
#
# The embedding models (gte-large-en-v1.5, BAAI/bge-m3) are ~4 GB and are NOT
# baked in — they download from HuggingFace on first request. Mount a volume at
# /root/.cache/huggingface to persist them across container restarts:
#   docker run -p 8001:8001 -e groq_api=gsk_xxx -v hf_cache:/root/.cache/huggingface rag-pipeline:latest

FROM python:3.12-slim

# libgomp1 is required by torch/onnxruntime (used by FlagEmbedding / fastembed).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# 1) torch (pinned to the frozen venv versions), installed from the default
#    PyPI index per project requirement.
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# 2) Everything else, pinned to the frozen venv (Windows-only pkgs stripped).
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# 3) Application code + baked-in vector data (vector_data_docs, data/).
#    .dockerignore keeps venv/__pycache__/.env out of the image.
COPY . .

EXPOSE 8001

# api.py inserts its own dir on sys.path, so running from /app resolves
# `main`, `constants`, and `history.history_pipeline` correctly.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
