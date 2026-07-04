#!/usr/bin/env bash
set -euo pipefail

export EVONITH_RUNTIME_DIR="${EVONITH_RUNTIME_DIR:-/var/lib/evonith-bf}"
export EVONITH_DEPLOYMENT_PROFILE="${EVONITH_DEPLOYMENT_PROFILE:-edge}"
export EVONITH_RUNTIME_PROFILE="${EVONITH_RUNTIME_PROFILE:-edge}"
export EVONITH_EDGE_MODE="${EVONITH_EDGE_MODE:-true}"
export EVONITH_BACKEND_PROFILE="${EVONITH_BACKEND_PROFILE:-backend-base}"
export EVONITH_UVICORN_HOST="${EVONITH_UVICORN_HOST:-0.0.0.0}"
export EVONITH_UVICORN_PORT="${EVONITH_UVICORN_PORT:-8080}"
export EVONITH_UVICORN_WORKERS="${EVONITH_UVICORN_WORKERS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${EVONITH_RUNTIME_DIR}"
if [ ! -w "${EVONITH_RUNTIME_DIR}" ]; then
  echo "Runtime directory is not writable: ${EVONITH_RUNTIME_DIR}" >&2
  exit 1
fi

echo "Starting Evonith backend profile=${EVONITH_BACKEND_PROFILE} host=${EVONITH_UVICORN_HOST} port=${EVONITH_UVICORN_PORT} workers=${EVONITH_UVICORN_WORKERS}"
cd "$(dirname "$0")/.."
cmd=(uvicorn apps.backend_api.app.main:app \
  --host "${EVONITH_UVICORN_HOST}" \
  --port "${EVONITH_UVICORN_PORT}" \
  --workers "${EVONITH_UVICORN_WORKERS}")

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'DRY_RUN backend command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
