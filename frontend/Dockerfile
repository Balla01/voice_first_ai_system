# frontend — plain HTML/CSS/JS, no build step. serve.py (stdlib http.server)
# just serves the directory with caching disabled; no pip dependencies.
#
# Talks to auido_capture only, over URLs hardcoded in capture-client.js as
# ws://localhost:8000/... — those resolve correctly from the *browser*
# (which hits the published host port), not from inside this container, so
# no extra config is needed as long as auido_capture is published on 8000
# on the same host (see docker-compose.yml).
#
# Build:
#   docker build -t insureassist-frontend:latest .
# Run:
#   docker run --rm -p 5500:5500 insureassist-frontend:latest

FROM python:3.11-slim

WORKDIR /app

COPY . .

EXPOSE 5500

CMD ["python", "serve.py", "5500"]
