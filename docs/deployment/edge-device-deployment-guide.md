# Edge Device Deployment Guide

## Target

Phase 13 targets Raspberry Pi / NVIDIA Jetson class devices with about 8 GB RAM
and a 256 GB SSD. Edge defaults are conservative and keep optional AI/vector/LLM
features disabled unless explicitly configured.

For a Jetson backend consumed by a separately hosted Streamlit frontend, follow
the complete [Jetson and hosted Streamlit guide](jetson-streamlit-deployment-guide.md).

## Environment

Start from `infra/env/edge.env.example` and place the real file outside git.

Important defaults:

```bash
EVONITH_RUNTIME_DIR=/var/lib/evonith-bf
EVONITH_UVICORN_WORKERS=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

## Dependency Profile

Install only the conservative edge backend profile on the device unless a
specific optional feature is being enabled:

```bash
uv sync --no-dev --group edge
# or
uv pip install -r requirements/edge.txt
```

The edge profile excludes Streamlit, provider SDKs, vector stores, document
parsers, and local embedding/LLM runtimes. Add optional groups such as
`backend-ai`, `backend-vector`, or `backend-documents` only when the matching
feature is configured and tested on the device.

## Runtime Bootstrap

```bash
python scripts/bootstrap_runtime.py --create
python scripts/validate_deployment.py --profile edge --offline
```

## Startup

```bash
scripts/edge_start_backend.sh
scripts/edge_start_frontend.sh
```

Dry-run check:

```bash
DRY_RUN=1 scripts/edge_start_backend.sh
DRY_RUN=1 scripts/edge_start_frontend.sh
```

## Resource Checks

The validation scripts check runtime writability, disk space, thread defaults,
canonical app paths, and placeholder secrets in production-like profiles.

## Model Assets

Active bundled model assets are packaged under:

```text
packages/furnace-data/furnace_data/assets/models/
```

Edge deployments can instead keep active model files on durable device storage
and point the app at that directory:

```bash
EVONITH_MODEL_DIR=/var/lib/evonith-bf/models
```

Only active model files should be present in the configured model directory.
Do not copy old archive folders such as `old_26_14` or `old_bmo_12062026` into
production source or the active model directory. Keep historical archives in an
external artifact store or backup location, and restore only the specific files
needed for rollback.

## Backup

Use `scripts/backup_runtime.py` before upgrades and copy archives off-device
where appropriate.
