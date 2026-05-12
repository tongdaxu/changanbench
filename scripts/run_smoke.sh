#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"

exec "$PYTHON_BIN" scripts/run_smoke.py "$@"
