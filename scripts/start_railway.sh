#!/bin/sh
set -eu

PORT="${PORT:-8080}"

exec uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --log-level info
