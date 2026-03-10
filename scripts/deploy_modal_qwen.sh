#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

APP_NAME="${1:-qwen25-7b-openai}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv/bin/python. Activate or create your virtualenv first."
  exit 1
fi

echo "Deploying Modal app '${APP_NAME}' from modal_qwen_openai_server.py ..."
DEPLOY_OUT="$(
  .venv/bin/python -m modal deploy modal_qwen_openai_server.py --name "${APP_NAME}" 2>&1 || true
)"
echo "${DEPLOY_OUT}"

if [[ "${DEPLOY_OUT}" == *"Error"* ]] || [[ "${DEPLOY_OUT}" == *"Traceback"* ]]; then
  echo ""
  echo "Deploy failed. Fix the error above and rerun."
  exit 1
fi

DEPLOY_OUT_CLEAN="$(echo "${DEPLOY_OUT}" | sed -E 's/\x1B\[[0-9;]*[mK]//g' | tr -d '\r')"
ENDPOINT="$(echo "${DEPLOY_OUT_CLEAN}" | rg -o 'https://[^[:space:]]*modal\.run' | head -n 1 || true)"
if [[ -n "${ENDPOINT}" ]]; then
  echo ""
  echo "Detected endpoint:"
  echo "  ${ENDPOINT}"
  echo ""
  echo "Update your .env:"
  echo "  MODAL_OPENAI_BASE_URL=${ENDPOINT}"
  echo "  CLARIFYCODE_BASE_URL=${ENDPOINT}/v1"
  echo "  MODAL_BASE_URL=${ENDPOINT}/v1"
else
  echo ""
  echo "Could not auto-detect endpoint URL from deploy output."
  echo "Open Modal Dashboard and copy your endpoint URL, then set:"
  echo "  MODAL_OPENAI_BASE_URL=<endpoint>"
  echo "  CLARIFYCODE_BASE_URL=<endpoint>/v1"
  echo "  MODAL_BASE_URL=<endpoint>/v1"
fi

echo ""
echo "Use dummy auth unless you add your own endpoint auth layer:"
echo "  CLARIFYCODE_API_KEY=dummy"
echo "  MODAL_API_KEY=dummy"
