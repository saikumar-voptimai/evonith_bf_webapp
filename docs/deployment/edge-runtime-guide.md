# Edge Runtime Guide

Edge deployments use conservative Raspberry Pi / NVIDIA Jetson style defaults and the canonical backend/frontend app locations.

## Environment

```bash
export EVONITH_RUNTIME_DIR=/var/lib/evonith-bf
export EVONITH_RUNTIME_PROFILE=edge
export EVONITH_EDGE_MODE=true
export EVONITH_BACKEND_PROFILE=backend-base
export EVONITH_FRONTEND_PROFILE=frontend
export EVONITH_UVICORN_WORKERS=1
export EVONITH_UVICORN_HOST=0.0.0.0
export EVONITH_UVICORN_PORT=8080
export EVONITH_FRONTEND_HOST=0.0.0.0
export EVONITH_FRONTEND_PORT=8501
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export EVONITH_ENABLE_OPTIONAL_AI=false
export EVONITH_ENABLE_OPTIONAL_VECTOR=false
export EVONITH_ENABLE_OPTIONAL_LOCAL_LLM=false
```

Set `EVONITH_AUTH_SECRET_KEY` securely in the device environment. Do not store
real secrets in docs, scripts, or service templates.

## Startup Scripts

```bash
scripts/edge_start_backend.sh
scripts/edge_start_frontend.sh
```

Both scripts:

- use `set -euo pipefail`
- create the runtime directory if possible
- fail if runtime storage is not writable
- set BLAS/OpenMP/thread defaults
- print only safe startup metadata
- use host/port/worker environment variables
- start `apps.backend_api.app.main:app` and `apps/frontend_streamlit/app.py`

Manual canonical backend command:

```bash
EVONITH_RUNTIME_DIR=./runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uv run uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080
```

The old service-local backend command remains a temporary compatibility path:

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Systemd Examples

Examples live under `infra/systemd/`:

- `evonith-backend.service.example`
- `evonith-frontend.service.example`

They use placeholder paths, a non-root `evonith` user, and environment files:

- `/etc/evonith-bf/backend.env`
- `/etc/evonith-bf/frontend.env`

These files are examples only and are not installed or enabled automatically.

## Optional Features

Optional AI/vector/local LLM features are disabled by default. Missing optional
packages appear as unavailable in `/api/v1/status/dependencies` rather than
breaking backend startup.
