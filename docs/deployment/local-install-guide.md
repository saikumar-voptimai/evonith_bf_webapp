# Local Install Guide

## Full Development

```bash
uv sync --group dev
uv run pytest tests -q
```

This installs the full local workflow, including backend, frontend, optional feature, and test dependencies.

## Backend Base

```bash
uv pip install -r requirements/backend-base.txt
EVONITH_RUNTIME_DIR=./runtime EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
  uv run python scripts/check_backend_minimal_startup.py
```

Canonical backend command:

```bash
EVONITH_RUNTIME_DIR=./runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uv run uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080
```


Temporary backend compatibility command:

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Backend With Data And Compute

```bash
uv pip install -r requirements/backend-data.txt
uv pip install -r requirements/backend-ml.txt
```

Use this when data connectors and model-backed compute are required.

## Backend With AI Or Vector Features

```bash
uv pip install -r requirements/backend-ai.txt
uv pip install -r requirements/backend-vector.txt
```

Provider keys and vector credentials must be supplied through secure environment
variables outside git. These profiles are not required for backend startup.

## Frontend

```bash
uv pip install -r requirements/frontend.txt
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 uv run streamlit run apps/frontend_streamlit/app.py
```

The frontend profile is for Streamlit, UI libraries, and API adapters. It does
not require backend internals, database clients, vector clients, LLM providers,
or model loaders.

Temporary frontend compatibility command for rollback checks:

```bash
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 uv run streamlit run src/app.py
```

## Verification

```bash
uv run python scripts/check_dependency_profiles.py
uv run python scripts/check_import_boundaries.py
uv run python scripts/check_backend_minimal_startup.py
uv run python scripts/check_frontend_api_imports.py
uv run python scripts/check_repository_structure.py
uv run python scripts/export_backend_openapi.py
```
