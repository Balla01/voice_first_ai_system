#!/usr/bin/env bash
#
# run.sh — build & run the RAG pipeline Docker image.
#
# Usage:
#   ./run.sh build          # build the image
#   ./run.sh rebuild        # build with --no-cache
#   ./run.sh run            # run in foreground (Ctrl+C to stop)
#   ./run.sh start          # run detached (background)
#   ./run.sh logs           # follow logs of the detached container
#   ./run.sh stop           # stop & remove the detached container
#   ./run.sh health         # curl the /health endpoint
#
# The Groq API key is read from ./.env (groq_api=...) and passed at runtime,
# so it is never baked into the image.

set -euo pipefail

IMAGE="rag-pipeline:latest"
CONTAINER="rag-api"
PORT="8001"
HF_VOLUME="hf_cache"

# Always operate from this script's own directory (the build context).
cd "$(dirname "$0")"

# Pull groq_api out of .env if present, so `run`/`start` can inject it.
GROQ_KEY=""
if [ -f .env ]; then
  GROQ_KEY="$(grep -E '^[[:space:]]*groq_api[[:space:]]*=' .env | tail -n1 | cut -d= -f2- | tr -d '\r' | xargs || true)"
fi

build() {
  docker build -t "$IMAGE" .
}

rebuild() {
  docker build --no-cache -t "$IMAGE" .
}

run() {
  docker run --rm -p "${PORT}:${PORT}" \
    -e groq_api="${GROQ_KEY}" \
    -v "${HF_VOLUME}:/root/.cache/huggingface" \
    "$IMAGE"
}

start() {
  docker run -d --name "$CONTAINER" -p "${PORT}:${PORT}" \
    -e groq_api="${GROQ_KEY}" \
    -v "${HF_VOLUME}:/root/.cache/huggingface" \
    "$IMAGE"
  echo "Started '$CONTAINER' on http://localhost:${PORT}"
}

logs() {
  docker logs -f "$CONTAINER"
}

stop() {
  docker stop "$CONTAINER" && docker rm "$CONTAINER"
}

health() {
  curl -sS "http://localhost:${PORT}/health" && echo
}

case "${1:-run}" in
  build)   build ;;
  rebuild) rebuild ;;
  run)     run ;;
  start)   start ;;
  logs)    logs ;;
  stop)    stop ;;
  health)  health ;;
  *)
    echo "Usage: $0 {build|rebuild|run|start|logs|stop|health}" >&2
    exit 1
    ;;
esac
