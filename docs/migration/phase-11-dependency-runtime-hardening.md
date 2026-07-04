# Phase 11 Dependency And Runtime Hardening

## Phase 11 Goal

Slim, separate, and harden dependency/runtime profiles so the backend and
frontend can be installed, tested, packaged, and operated independently.

## Audit

1. Current dependency files and package-management approach:
   The repo uses root `pyproject.toml`, `uv.lock`, legacy `requirements.txt`,
   `furnace-data-service/pyproject.toml`, and `furnace_data/pyproject.toml`.
   The root dependency list remains the full/dev compatibility surface.

2. Current backend dependencies:
   The backend sidecar uses FastAPI, Uvicorn, Pydantic settings, pandas/numpy,
   SQLAlchemy, bcrypt, shared `furnace_data`, and optional data/ML/provider
   packages in the full environment.

3. Current frontend dependencies:
   Streamlit, plotting/UI libraries, pandas, HTTP clients, and API adapter code.

4. Current shared/domain dependencies:
   `furnace_data` contains data/runtime utilities and shared dataset/domain
   code. It still includes some data/ML dependencies for compatibility.

5. Heavy optional dependencies:
   Streamlit, Plotly, pydeck/pygwalker, scikit-learn/scipy/xgboost/joblib,
   torch, sentence-transformers, LangChain, OpenAI/Anthropic SDKs,
   qdrant-client, PyMuPDF/python-docx/python-pptx/pypdf. No OCR package was
   found as a required backend startup dependency.

6. Backend modules that still import frontend/UI packages:
   No `streamlit` import exists under `furnace-data-service/app`.

7. Frontend service modules that import backend internals:
   `src/services/*_api.py` imports only frontend API-client helpers and typing.

8. Backend startup imports that may pull heavy optional packages:
   Phase 7-9 services already lazy-import OpenAI, Qdrant, and joblib at call
   sites. Phase 11 wraps those lazy imports with `optional_dependency_service`.

9. Frontend startup imports that may pull backend/data/LLM/vector packages:
   API adapters do not import backend internals, DB clients, vector clients, LLM
   SDKs, or model loaders.

10. Current lockfile behavior:
    `uv.lock` is updated by `uv run` when dependency group metadata changes.

11. Current test setup and dependency assumptions:
    Tests run in the full local/dev environment. Phase 11 adds script checks
    that validate profile metadata and import boundaries without installing
    packages or requiring optional services.

12. Phase 11 changes implemented:
    Dependency groups, requirements profiles, optional dependency guard,
    runtime profile config, `/status/config`, enriched dependency status,
    boundary/profile/startup scripts, edge scripts, systemd examples, docs, and
    focused tests.

13. Deferred:
    Full folder restructuring, wheel slimming for each package, Docker images,
    production audit database backend, external metrics stacks, and removal of
    direct-mode fallbacks.

## Dependency Groups

Groups added to root `pyproject.toml`:

- `backend-base`
- `backend-data`
- `backend-ml`
- `backend-ai`
- `backend-vector`
- `backend-documents`
- `frontend`
- `dev`
- `edge`

Requirements profile files were also added under `requirements/`.

## Backend-Base Profile

`backend-base` includes FastAPI/Uvicorn, settings, auth hashing, SQLAlchemy,
pandas/numpy, shared package access, and safe runtime utilities. It excludes
Streamlit, Qdrant, provider SDKs, torch, sentence-transformers, and heavy ML
extras such as xgboost.

## Frontend Profile

The frontend profile includes Streamlit, UI/plotting packages, HTTP clients, and
display dependencies. It excludes backend internals, DB clients, vector clients,
LLM provider SDKs, and model loaders.

## Optional Profiles

- `backend-data`: data connectors and file/data export helpers.
- `backend-ml`: model/compute dependencies.
- `backend-ai`: cloud provider LLM SDKs.
- `backend-vector`: Qdrant and embedding/local vector dependencies.
- `backend-documents`: document text extraction dependencies.

Optional features remain disabled until explicit feature flags and secure
configuration are supplied.

## Edge Profile

`edge` is conservative: one Uvicorn worker, thread caps, runtime directory
under `/var/lib/evonith-bf` by convention, AI/vector/local LLM disabled by
default, and optional services reported as disabled/unconfigured.

## Lazy Imports And Optional Dependency Errors

`furnace-data-service/app/services/optional_dependency_service.py` provides:

- `is_module_available`
- `require_optional_module`
- `get_optional_dependency_status`
- `optional_dependency_error`

Missing optional packages return `DEPENDENCY_OPTIONAL_NOT_INSTALLED` with a
safe install-group recommendation.

## Checks Added

- `scripts/check_import_boundaries.py`
- `scripts/check_dependency_profiles.py`
- `scripts/check_backend_minimal_startup.py`
- `scripts/check_frontend_api_imports.py`
- `scripts/edge_start_backend.sh`
- `scripts/edge_start_frontend.sh`

Systemd examples were added under `infra/systemd/`. Docker/Compose templates
were intentionally not added because Docker is not currently required.

## Status Endpoint Updates

- `GET /api/v1/status/config` returns safe runtime profile metadata.
- `GET /api/v1/status/dependencies` now includes runtime profile, edge mode,
  dependency groups, optional dependency availability, and profile flags.

## Security And Compatibility

No secrets are stored in docs, scripts, `.env.example`, or service examples.
Operational endpoints remain admin-protected. Direct-mode fallbacks and legacy
routes remain available.

## OpenAPI And Tests

OpenAPI is exported with `uv run python scripts/export_backend_openapi.py`.
The test report is `docs/migration/phase-11-test-execution-report.md`.
