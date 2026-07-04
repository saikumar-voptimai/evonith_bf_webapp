# Phase 12 Testing Guide

## Purpose

This guide verifies repository restructuring, backend/frontend canonical
entrypoints, compatibility shims, import boundaries, dependency profile
integrity, script/infra updates, OpenAPI export, and full regression safety for
Phases 1-11.

## Prerequisites

- Python 3.11 or the version pinned by the repo environment.
- `uv` package manager.
- Branch: `migration/backend-frontend-split`.
- Runtime directory: `EVONITH_RUNTIME_DIR=./runtime` for local tests.
- `EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me` for auth-protected tests.
- No production secrets are required.
- Full/dev dependencies are required for the full test suite.
- `backend-base` and `frontend` profiles are required for dependency checks.

## Environment Setup

Local test setup:

```bash
export EVONITH_RUNTIME_DIR=./runtime
export EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me
export BACKEND_API_BASE_URL=http://localhost:8080/api/v1
export EVONITH_RUNTIME_PROFILE=local
export EVONITH_EDGE_MODE=false
```

Edge-like setup:

```bash
export EVONITH_RUNTIME_PROFILE=edge
export EVONITH_EDGE_MODE=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

## New Startup Commands

Backend from repo root:

```bash
EVONITH_RUNTIME_DIR=./runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080
```

Backend from app directory, if supported:

```bash
cd apps/backend_api
EVONITH_RUNTIME_DIR=../../runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Frontend:

```bash
streamlit run apps/frontend_streamlit/app.py
```

## Compatibility Startup Commands

Backend compatibility:

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Frontend compatibility:

```bash
streamlit run src/app.py
```

Compatibility commands are temporary. Canonical commands should be used going
forward. Both compatibility commands must work in Phase 12 unless a future phase
explicitly removes them.

## Automated Test Commands

```bash
pytest furnace-data-service/tests -q
pytest tests/frontend -q
pytest tests/integration -q
pytest tests/dependency -q
pytest tests/structure -q
pytest tests -q
python scripts/export_backend_openapi.py
python scripts/check_repository_structure.py
python scripts/check_import_boundaries.py
python scripts/check_dependency_profiles.py
python scripts/check_backend_minimal_startup.py
python scripts/check_frontend_api_imports.py
```

Import checks:

```bash
python -c "from apps.backend_api.app.main import app; print(app.title)"
python -c "from apps.frontend_streamlit.services.status_api import get_status; print('new frontend import ok')"
python -c "from src.services.status_api import get_status; print('old frontend shim import ok')"
```

Boundary checks:

```bash
grep -R "import streamlit\|from streamlit" apps/backend_api/app furnace-data-service/app || true
grep -R "furnace-data-service\|from app\|import app" apps/frontend_streamlit/services src/services || true
grep -R "OPENAI_API_KEY=\|QDRANT_API_KEY=\|EVONITH_AUTH_SECRET_KEY=.*[A-Za-z0-9]" docs/migration/phase-12-test-execution-report.md docs/testing/phase-12-testing-guide.md scripts infra || true
```

## Manual Backend Verification

1. Start backend using the new canonical command.
2. Call `/api/v1/health`.
3. Call `/api/v1/readiness`.
4. Call `/api/v1/status`.
5. Login as admin if needed.
6. Call representative endpoints from all phases:
   - `/api/v1/data/sources`
   - `/api/v1/datasets`
   - `/api/v1/feedback/config`
   - `/api/v1/material-balance/config`
   - `/api/v1/recommendations/config`
   - `/api/v1/blend-optimizer/context`
   - `/api/v1/copilot/config`
   - `/api/v1/furnacemind/config`
7. Stop backend.
8. Start backend using compatibility command.
9. Repeat `/health` and `/openapi` check.
10. Confirm OpenAPI path set matches canonical app.

## Manual Frontend Verification

1. Start frontend using new canonical command.
2. Verify app loads.
3. Verify backend status badge appears.
4. Verify feature flags still parse.
5. Verify pages are listed.
6. Verify API adapters can call backend if backend is running.
7. Stop frontend.
8. Start frontend using compatibility command: `streamlit run src/app.py`.
9. Verify compatibility app loads.
10. Confirm direct-mode fallbacks are still available.

## Phase 11 Dependency/Runtime Regression Verification

Because Phase 12 changes paths and packaging metadata, verify Phase 11 still
works:

1. `python scripts/check_dependency_profiles.py` passes.
2. `python scripts/check_import_boundaries.py` passes.
3. `python scripts/check_backend_minimal_startup.py` passes.
4. `python scripts/check_frontend_api_imports.py` passes.
5. `backend-base` still excludes Streamlit.
6. `frontend` profile still includes Streamlit.
7. Optional provider/vector/model dependencies are still lazy.
8. `edge_start_backend.sh` points to the new backend app.
9. `edge_start_frontend.sh` points to the new frontend app.
10. No secrets appear in scripts or infra examples.

## Regression Verification

- Phase 1 runtime tests pass.
- Phase 2 API foundation tests pass.
- Phase 3 ApiClient/status tests pass.
- Phase 4 data/dataset tests pass.
- Phase 5 auth/admin tests pass.
- Phase 6 feedback tests pass.
- Phase 7 domain-compute tests pass.
- Phase 8 Copilot tests pass.
- Phase 9 FurnaceMind tests pass.
- Phase 10 operational tests pass.
- Phase 11 dependency/runtime tests pass.
- Full test suite passes.

## Expected Outcomes

- All tests pass.
- Canonical backend entrypoint works.
- Canonical frontend entrypoint works.
- Compatibility backend entrypoint works or is clearly documented.
- Compatibility frontend entrypoint works or is clearly documented.
- OpenAPI export works.
- No API path loss.
- `backend-base` dependency separation remains intact.
- Frontend API adapter boundaries remain intact.
- Edge scripts and infra examples point to the new structure.
- No secrets are introduced.
- No direct-mode fallback is removed.

## Troubleshooting

- Module import path errors: confirm commands run from repo root or documented
  app directories.
- `PYTHONPATH` issues: use canonical commands; shims add only entrypoint-level
  path compatibility.
- Compatibility shim import failure: inspect `furnace-data-service/app/main.py`
  and `src/app.py`.
- OpenAPI path mismatch: run `python scripts/export_backend_openapi.py`.
- Circular import after move: check old `app.main` imports only the canonical
  app.
- Edge script path wrong: run `python scripts/check_repository_structure.py`.
- Streamlit accidentally imported by backend: run
  `python scripts/check_import_boundaries.py`.
- Frontend adapter accidentally imports backend internals: run
  `python scripts/check_frontend_api_imports.py`.
- Dependency group package discovery issue: run
  `python scripts/check_dependency_profiles.py`.
- Test conftest path assumptions: ensure repo root, `src`, and
  `furnace-data-service` are inserted in the intended order.
- Old docs referencing old paths: historical phase docs may keep old commands;
  deployment docs and Phase 12 docs should show canonical commands.
- Runtime directory not writable: set `EVONITH_RUNTIME_DIR` to a writable path.

