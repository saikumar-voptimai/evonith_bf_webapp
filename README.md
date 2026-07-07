# Evonith BF Webapp

Evonith BF is a Blast Furnace intelligence web application with a FastAPI backend, a Streamlit frontend, and a shared furnace data/domain package. The repository has been restructured into canonical app and package paths.

## Current Architecture

| Area | Canonical path | Notes |
|---|---|---|
| Backend API | `apps/backend_api/app` | FastAPI app, `/api/v1` routes, legacy routes, services, repositories, tasks. |
| Frontend app | `apps/frontend_streamlit` | Streamlit entrypoint, page modules, frontend API adapters, UI helpers, frontend assets. |
| Shared package | `packages/furnace-data/furnace_data` | Shared data/domain package imported as `furnace_data`. |
| Tests | `tests` | Canonical backend, frontend, integration, dependency, deployment, structure, and fixture tests. |
| Runtime data | `runtime` | Local generated data root. Ignored except `runtime/.gitkeep`. |
| Infrastructure | `infra` | Deployment examples for edge, services, and reverse proxies. |
| Scripts | `scripts` | Validation, bootstrap, smoke test, migration, and release helpers. |
| Docs | `docs` | Active documentation plus archived migration history. |

Temporary compatibility paths still exist for rollback and old imports. They should not be used as primary commands and should be removed only in a later deprecation phase after tests and deployment telemetry confirm they are no longer needed.

## Local Setup

Install the full development environment:

```bash
uv sync --group dev
```

Create local runtime directories:

```bash
uv run python scripts/bootstrap_runtime.py --create
```

## Run Locally

Start the backend API:

```bash
EVONITH_RUNTIME_DIR=./runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uv run uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080
```

Start the Streamlit frontend:

```bash
EVONITH_RUNTIME_DIR=./runtime \
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 \
uv run streamlit run apps/frontend_streamlit/app.py
```


## Dependency Profiles

The root project metadata is intentionally slim. Use explicit groups or requirements profiles:

```bash
uv sync --no-dev --group backend-base
uv sync --no-dev --group frontend
uv sync --group dev
uv sync --no-dev --group edge
```

See [dependency profiles](docs/deployment/dependency-profiles.md) for backend-base, backend-data, backend-ml, backend-ai, backend-vector, backend-documents, frontend, dev, and edge details.

## Tests And Validation

Common checks:

```bash
uv run pytest tests -q
uv run python scripts/check_backend_minimal_startup.py
uv run python scripts/check_frontend_api_imports.py
uv run python scripts/check_dependency_profiles.py
uv run python scripts/check_repository_structure.py
uv run python scripts/verify_release_readiness.py --allow-dirty --skip-tests
```

Focused suites:

```bash
uv run pytest tests/backend -q
uv run pytest tests/frontend -q
uv run pytest tests/integration -q
uv run pytest tests/dependency -q
uv run pytest tests/deployment -q
uv run pytest tests/structure -q
```

## Documentation

Start with [docs/README.md](docs/README.md).

Key active guides:

- [Production deployment](docs/deployment/production-deployment-guide.md)
- [Edge device deployment](docs/deployment/edge-device-deployment-guide.md)
- [Local install](docs/deployment/local-install-guide.md)
- [Local and staging deployment](docs/deployment/local-staging-deployment-guide.md)
- [Dependency profiles](docs/deployment/dependency-profiles.md)
- [Testing guide](docs/testing/phase-13-testing-guide.md)
- [Model assets](docs/operations/model-assets.md)
- [Runtime cleanup](docs/operations/runtime-cleanup.md)
- [OpenAPI v1 export](docs/api/openapi-v1.json)
- [Post-phase-13 cleanup plan](docs/migration/post-phase-13-structure-cleanup-plan.md)
- [Migration history archive](docs/archive/migration-history/)

## Runtime And Secrets

Generated files, uploads, SQLite databases, caches, logs, and exported artifacts belong under `EVONITH_RUNTIME_DIR`, which defaults to `./runtime` for local development. Production secrets must be supplied through deployment secret management, not committed files.

## Compatibility Cleanup Note

Direct-mode fallback and legacy API routes remain covered through canonical app and package modules. The legacy frontend source folder, backend sidecar, and root shared-package shim have been removed.


