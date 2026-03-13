#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -x "googlevenv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-googlevenv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected a usable Python at ${PYTHON_BIN}. Install dependencies into googlevenv or set PYTHON_BIN." >&2
  exit 1
fi

if [[ -z "${GCP_PROJECT_NAME:-}" ]]; then
  echo "GCP_PROJECT_NAME must be set in .env or the environment." >&2
  exit 1
fi

export GEMINI_VERTEX_HOST="${GEMINI_VERTEX_HOST:-127.0.0.1}"
export GEMINI_VERTEX_PORT="${GEMINI_VERTEX_PORT:-8010}"
export GEMINI_VERTEX_LOCATION="${GEMINI_VERTEX_LOCATION:-global}"
export GEMINI_VERTEX_MODEL="${GEMINI_VERTEX_MODEL:-gemini-3-pro-preview}"

MAX_EXAMPLES="${1:-20}"
RESULTS_FILE="${RESULTS_FILE:-results/google_humaneval_oneshot_${MAX_EXAMPLES}.jsonl}"
SUMMARY_DIR="${SUMMARY_DIR:-${RESULTS_FILE%.jsonl}_summary}"
PROXY_LOG="${PROXY_LOG:-/tmp/gemini_vertex_proxy.log}"
GOOGLE_EXPERIMENT_CONFIG="${GOOGLE_EXPERIMENT_CONFIG:-configs/mvp_humaneval_google_oneshot.yaml}"

"${PYTHON_BIN}" -m uvicorn servers.gemini_vertex_openai_server:app \
  --host "${GEMINI_VERTEX_HOST}" \
  --port "${GEMINI_VERTEX_PORT}" \
  >"${PROXY_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${PYTHON_BIN}" - <<'PY'
import os
import sys
import time
import urllib.request

host = os.environ["GEMINI_VERTEX_HOST"]
port = os.environ["GEMINI_VERTEX_PORT"]
url = f"http://{host}:{port}/health"
deadline = time.time() + 30
last_error = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status == 200:
                sys.exit(0)
    except Exception as exc:  # pragma: no cover - startup polling
        last_error = exc
        time.sleep(0.5)

print(f"Gemini proxy failed to start: {last_error}", file=sys.stderr)
sys.exit(1)
PY

export CODEGEN_BASE_URL="http://${GEMINI_VERTEX_HOST}:${GEMINI_VERTEX_PORT}/v1"
export CODEGEN_API_KEY="${CODEGEN_API_KEY:-${CLARIFYCODE_API_KEY:-dummy}}"
export CODEGEN_MODEL="${GEMINI_VERTEX_MODEL}"
# Keep Google runs independent from any Modal-era token caps in .env.
export CODEGEN_MAX_TOKENS="${GOOGLE_CODEGEN_MAX_TOKENS:-50000}"

"${PYTHON_BIN}" scripts/run_experiment.py \
  --config "${GOOGLE_EXPERIMENT_CONFIG}" \
  --strategies one-shot \
  --max-examples "${MAX_EXAMPLES}" \
  --output-file "${RESULTS_FILE}"

"${PYTHON_BIN}" scripts/summarize_results.py \
  --results "${RESULTS_FILE}" \
  --output-dir "${SUMMARY_DIR}"

echo "Results: ${RESULTS_FILE}"
echo "Summary: ${SUMMARY_DIR}"
echo "Proxy log: ${PROXY_LOG}"
