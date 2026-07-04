#!/usr/bin/env bash
set -euo pipefail

export EVONITH_RUNTIME_DIR="${EVONITH_RUNTIME_DIR:-/var/lib/evonith-bf}"
export EVONITH_RUNTIME_PROFILE="${EVONITH_RUNTIME_PROFILE:-edge}"
export EVONITH_EDGE_MODE="${EVONITH_EDGE_MODE:-true}"
export EVONITH_FRONTEND_PROFILE="${EVONITH_FRONTEND_PROFILE:-frontend}"
export EVONITH_FRONTEND_HOST="${EVONITH_FRONTEND_HOST:-0.0.0.0}"
export EVONITH_FRONTEND_PORT="${EVONITH_FRONTEND_PORT:-8501}"
export BACKEND_API_BASE_URL="${BACKEND_API_BASE_URL:-http://localhost:8080/api/v1}"

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

echo "Starting Evonith frontend profile=${EVONITH_FRONTEND_PROFILE} host=${EVONITH_FRONTEND_HOST} port=${EVONITH_FRONTEND_PORT} backend=${BACKEND_API_BASE_URL}"
exec streamlit run apps/frontend_streamlit/app.py \
  --server.address "${EVONITH_FRONTEND_HOST}" \
  --server.port "${EVONITH_FRONTEND_PORT}"
