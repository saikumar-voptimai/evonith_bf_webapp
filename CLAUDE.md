# CLAUDE.md

This file gives coding agents current orientation for the Evonith BF webapp repository.

## Project Overview

Evonith BF is an industrial Blast Furnace web application with three canonical Python surfaces:

- Backend API: `apps/backend_api/app`
- Streamlit frontend: `apps/frontend_streamlit`
- Shared data/domain package: `packages/furnace-data/furnace_data`, imported as `furnace_data`

The backend exposes the FastAPI app at `apps.backend_api.app.main:app` and keeps `/api/v1` routes stable. The frontend entrypoint is `apps/frontend_streamlit/app.py`. Runtime-generated state belongs under `EVONITH_RUNTIME_DIR`, normally `./runtime` for local development.

Compatibility shims still exist for old imports and rollback. Treat `src/app.py`, `src/services/*`, `src/config/*`, `src/ui/*`, `src/custom_pages/*`, `furnace-data-service/app/*`, and root `furnace_data/*` as temporary compatibility surfaces unless a task explicitly targets shim cleanup.

## Current Commands

Install development dependencies:

```bash
uv sync --group dev
```

Start the backend:

```bash
EVONITH_RUNTIME_DIR=./runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uv run uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080
```

Start the frontend:

```bash
EVONITH_RUNTIME_DIR=./runtime \
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 \
uv run streamlit run apps/frontend_streamlit/app.py
```

Run tests and checks:

```bash
uv run pytest tests -q
uv run python scripts/check_repository_structure.py
uv run python scripts/check_import_boundaries.py
uv run python scripts/check_dependency_profiles.py
uv run python scripts/check_backend_minimal_startup.py
uv run python scripts/check_frontend_api_imports.py
```

Temporary compatibility commands may be used only for rollback checks:

```bash
uv run streamlit run src/app.py
```

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Dependency Boundaries

- `backend-base` must stay free of Streamlit, Qdrant, provider SDKs, document parsers, OCR packages, local embedding stacks, and heavy model runtimes.
- Frontend dependencies belong in the `frontend` group and should not import backend internals.
- Optional AI, vector, document, and ML stacks must remain lazy and grouped.
- Do not reintroduce Alembic dependencies or migration docs as active guidance.

## Runtime Rules

- Use `EVONITH_RUNTIME_DIR` for generated files, uploads, caches, SQLite databases, logs, and deployment artifacts.
- Source assets live under `apps/frontend_streamlit/assets` for UI assets or `packages/furnace-data/furnace_data/assets` for shared/model/prompt assets.
- Do not write runtime files under source folders.
- Do not commit secrets. Use placeholder development values only in docs and tests.

## Domain Notes

BF2 key KPIs include fuel rate, coke rate, PCI rate, ETA CO, production rate, RAFT, permeability, total pressure drop, heat load, and temperature spread. Shifts are fixed IST windows: A 06:00-14:00, B 14:00-22:00, and C 22:00-06:00. The shared package and `src` direct-mode modules contain the current data mapping and domain logic.

## Documentation Map

Use `docs/README.md` as the active documentation index. Historical phase-by-phase notes are archived under `docs/archive/migration-history/` and should not be treated as current operational instructions.
