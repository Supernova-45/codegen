#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ "$MODE" == "--venv" ]]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  echo "Created and initialized .venv"
  exit 0
fi

if [[ "$MODE" == "--conda" ]]; then
  conda create -n clarifycode python=3.11 -y
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate clarifycode
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  echo "Created and initialized conda env: clarifycode"
  exit 0
fi

echo "Usage: bash scripts/setup_env.sh --venv | --conda"
exit 1
