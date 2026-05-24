#!/bin/sh
set -eu

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-9000}"
WEB_PORT="${PORT:-8080}"

export API_HOST
export API_PORT
export WEB_PORT
export API_BASE_URL="${API_BASE_URL:-http://${API_HOST}:${API_PORT}}"

uvicorn src.api.main:app \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --workers 1 \
  --log-level info &

api_pid="$!"
web_pid=""

cleanup() {
  kill "${api_pid}" 2>/dev/null || true
  if [ -n "${web_pid}" ]; then
    kill "${web_pid}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

python - <<PY
import os
import time
import httpx

url = f"http://{os.environ.get('API_HOST', '127.0.0.1')}:{os.environ.get('API_PORT', '9000')}/api/v1/health"
deadline = time.time() + 60
last_error = None

while time.time() < deadline:
    try:
        response = httpx.get(url, timeout=2)
        if response.status_code < 500:
            raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(1)

raise SystemExit(f"Backend did not become ready: {last_error}")
PY

streamlit run frontend/app.py \
  --server.address 0.0.0.0 \
  --server.port "${WEB_PORT}" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false &

web_pid="$!"
wait "${web_pid}"
