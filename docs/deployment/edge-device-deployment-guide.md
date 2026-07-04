# Edge Device Deployment Guide

## Target

Phase 13 targets Raspberry Pi / NVIDIA Jetson class devices with about 8 GB RAM
and a 256 GB SSD. Edge defaults are conservative and keep optional AI/vector/LLM
features disabled unless explicitly configured.

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

## Backup

Use `scripts/backup_runtime.py` before upgrades and copy archives off-device
where appropriate.

