# Post-Phase 13 Structure Cleanup Plan

## Current Status

The repository now uses the canonical backend, frontend, shared package, runtime, script, documentation, and test layout:

```text
apps/
  backend_api/app/
  frontend_streamlit/
packages/
  furnace-data/furnace_data/
infra/
scripts/
docs/
tests/
runtime/
```

## Canonical Paths

| Area | Canonical path | Status |
|---|---|---|
| Backend API | `apps/backend_api/app` | Canonical FastAPI implementation. |
| Frontend | `apps/frontend_streamlit` | Canonical Streamlit app, pages, services, config, UI helpers, and assets. |
| Shared package | `packages/furnace-data/furnace_data` | Canonical package imported as `furnace_data`. |
| Runtime | `runtime` or `EVONITH_RUNTIME_DIR` | Generated state only; source folders must stay clean. |
| Tests | `tests` | Canonical test root for backend, frontend, integration, dependency, deployment, structure, and fixtures. |

## Legacy Surfaces

- The old backend sidecar has been removed from the active source tree.
- The legacy frontend source tree has been removed from the active source tree.
- The root shared-package shim has been removed; editable package metadata keeps `import furnace_data` stable.
- Legacy API routes remain supported inside the canonical backend.

## Current Backend Command

```bash
EVONITH_RUNTIME_DIR=./runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uv run uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080
```

## Current Frontend Command

```bash
EVONITH_RUNTIME_DIR=./runtime \
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 \
uv run streamlit run apps/frontend_streamlit/app.py
```


## Migration Map

| Current Path | Target Path | Action | Compatibility Strategy | Risk Level | Tests Required | Notes |
|---|---|---|---|---|---|---|
| Removed backend sidecar | `apps/backend_api/app` | Removed after canonical backend validation. | No active backend sidecar shim remains. | High | Backend, integration, OpenAPI, import-boundary, structure. | Backend work must use canonical imports. |
| Removed legacy frontend entrypoint | `apps/frontend_streamlit/app.py` | Removed after canonical frontend validation. | No active frontend entrypoint shim remains. | Medium | Frontend, structure. | Use the canonical Streamlit command only. |
| Removed legacy page wrappers | `apps/frontend_streamlit/custom_pages` | Removed after canonical page validation. | Canonical page registry points at app pages only. | Medium | Frontend, structure. | Page names and order remain canonical. |
| Removed legacy frontend service wrappers | `apps/frontend_streamlit/services` | Removed after canonical service validation. | Frontend imports canonical adapters only. | Medium | Frontend API import checks. | No backend internals in frontend adapters. |
| Root shared-package shim | `packages/furnace-data/furnace_data` | Removed after canonical package validation. | Editable package metadata keeps `import furnace_data` stable. | Medium | Shared package, backend startup, structure tests. | Source now lives only in the canonical package tree. |
| Runtime/generated data | `runtime` or configured runtime dir | Source copies removed. | Runtime bootstrap and migration scripts handle setup. | Medium | Structure and deployment tests. | Source folders must remain artifact-free. |

## Do Not Delete Yet

- Legacy route modules inside the canonical backend.
- Active model assets under `packages/furnace-data/furnace_data/assets` or configured model runtime directories.
- Runtime migration, backup, bootstrap, validation, and deployment scripts.
- Any files whose usage remains unclear after scans.

## Baseline Validation Commands

```bash
python scripts/export_backend_openapi.py
python scripts/check_repository_structure.py
python scripts/check_import_boundaries.py
python scripts/check_dependency_profiles.py
python scripts/check_backend_minimal_startup.py
python scripts/check_frontend_api_imports.py
pytest tests -q
```

