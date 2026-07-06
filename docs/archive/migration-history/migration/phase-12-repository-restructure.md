# Phase 12 Repository Restructure

## Purpose

Phase 12 introduces canonical monorepo-style app locations while preserving the
old backend and frontend startup paths. No API contracts, business logic,
runtime storage semantics, direct-mode fallbacks, or legacy backend routes were
removed.

## Structure Implemented

- Canonical backend entrypoint: `apps/backend_api/app/main.py`
- Backend compatibility shim: `furnace-data-service/app/main.py`
- Canonical frontend entrypoint: `apps/frontend_streamlit/app.py`
- Frontend compatibility shim: `src/app.py`
- Canonical frontend service wrappers: `apps/frontend_streamlit/services/`
- Canonical frontend page wrappers: `apps/frontend_streamlit/custom_pages/`
- Shared package strategy: `furnace_data` remains in place for Phase 12
- Runtime strategy: unchanged from Phase 1; generated files remain under `EVONITH_RUNTIME_DIR`

## Migration Map

| Old Path | New Path | Move Type | Compatibility Strategy | Tests Covering It | Notes |
|---|---|---|---|---|---|
| `furnace-data-service/app/main.py` | `apps/backend_api/app/main.py` | moved now | Old file re-exports `app` and `create_app` from canonical entrypoint | `tests/structure/test_backend_entrypoints.py`, `tests/structure/test_compatibility_shims.py` | Backend factory logic now lives in the canonical entrypoint |
| `furnace-data-service/app/api` | `apps/backend_api/app/api` | compatibility shim now, move later | `apps.backend_api.app.__path__` includes the legacy app tree | `tests/structure/test_backend_entrypoints.py`, `scripts/check_import_boundaries.py` | Avoids duplicating router business logic in Phase 12 |
| `furnace-data-service/app/core` | `apps/backend_api/app/core` | compatibility shim now, move later | Canonical package aliases legacy app modules | `tests/structure/test_backend_import_boundaries_phase12.py` | Boundary checks cover both trees |
| `furnace-data-service/app/services` | `apps/backend_api/app/services` | compatibility shim now, move later | Canonical package aliases legacy app modules | `tests/structure/test_backend_entrypoints.py` | Optional dependencies remain lazy in service modules |
| `furnace-data-service/app/repositories` | `apps/backend_api/app/repositories` | compatibility shim now, move later | Canonical package aliases legacy app modules | `tests/structure/test_backend_entrypoints.py` | Repository storage behavior unchanged |
| `furnace-data-service/app/tasks` | `apps/backend_api/app/tasks` | compatibility shim now, move later | Canonical package aliases legacy app modules | `scripts/check_repository_structure.py` | Runtime job storage remains under `EVONITH_RUNTIME_DIR` |
| `furnace-data-service/app/models` | `apps/backend_api/app/models` | compatibility shim now, move later | Canonical package aliases legacy app modules | `tests/structure/test_backend_entrypoints.py` | No schema or API contract changes |
| `src/app.py` | `apps/frontend_streamlit/app.py` | moved now | Old file runs the canonical Streamlit app via `runpy` | `tests/structure/test_frontend_entrypoints.py`, `tests/structure/test_compatibility_shims.py` | `streamlit run src/app.py` remains available |
| `src/custom_pages` | `apps/frontend_streamlit/custom_pages` | compatibility shim now, move later | Thin page wrappers execute legacy page files | `tests/structure/test_frontend_entrypoints.py` | Avoids duplicating page logic |
| `src/services` | `apps/frontend_streamlit/services` | compatibility shim now, move later | Canonical wrapper modules re-export legacy API adapters | `tests/structure/test_frontend_entrypoints.py`, `tests/structure/test_frontend_import_boundaries_phase12.py` | Old `src.services` imports remain available |
| `src/ui` | `apps/frontend_streamlit/ui` | compatibility shim now, move later | Canonical package path aliases `src/ui` | `scripts/check_repository_structure.py` | UI logic remains in legacy tree for direct-mode safety |
| `src/config` | `apps/frontend_streamlit/config` | compatibility shim now, move later | Canonical package path aliases `src/config` | `scripts/check_repository_structure.py` | Config files remain source assets |
| `src/utils` | `apps/frontend_streamlit/utils` | compatibility shim now, move later | Canonical package path aliases `src/utils` | `scripts/check_repository_structure.py` | Some utilities still support direct mode |
| `src/assets` | `apps/frontend_streamlit/assets` | compatibility shim now, move later | Canonical package path aliases `src/assets` | `scripts/check_repository_structure.py` | Static assets are not runtime-generated |
| `furnace_data` | `packages/furnace-data/furnace_data` | intentionally left in place | Root editable package remains configured in `pyproject.toml` | `scripts/check_dependency_profiles.py` | Move deferred to Phase 13/14 to avoid packaging churn |
| `scripts` | `scripts` | intentionally left in place | Scripts updated to canonical app paths | `tests/structure/test_repository_structure_script.py`, `tests/dependency/test_phase11_scripts.py` | No heavy deployment system added |
| `infra` | `infra` | intentionally left in place | Existing systemd examples use updated edge scripts | `scripts/check_repository_structure.py` | No deployment cutover attempted |
| `tests` | `tests` | intentionally left in place | Added `tests/structure` for Phase 12 | `pytest tests/structure -q` | Existing test layout preserved |
| `docs` | `docs` | intentionally left in place | Added Phase 12 docs and updated deployment docs | `scripts/check_repository_structure.py` | Older phase docs keep historical commands |

## Startup Commands

Canonical backend:

```bash
EVONITH_RUNTIME_DIR=./runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080
```

Compatibility backend:

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Canonical frontend:

```bash
streamlit run apps/frontend_streamlit/app.py
```

Compatibility frontend:

```bash
streamlit run src/app.py
```

## Boundary Decisions

- Backend code still must not import Streamlit.
- Frontend API adapters still must not import backend internals.
- Optional AI, vector, model, document, and LLM dependencies remain lazy.
- Runtime files still resolve through `EVONITH_RUNTIME_DIR`.
- Legacy backend routes and direct-mode frontend fallbacks remain available.

## Deferred Work

- Move backend deep modules from `furnace-data-service/app` into
  `apps/backend_api/app` after a later compatibility-removal phase.
- Move frontend page/UI/config/utils modules into `apps/frontend_streamlit`
  after direct-mode import paths are simplified.
- Move `furnace_data` under `packages/furnace-data` in a dedicated packaging
  phase.
- Add production image metadata once deployment cutover is in scope.

