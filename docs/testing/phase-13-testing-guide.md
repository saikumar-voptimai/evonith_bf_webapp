# Phase 13 Testing Guide

## Purpose

This guide verifies production deployment readiness, edge-device deployment
assets, runtime bootstrap, deployment validation, smoke tests, API cutover
validation, backup/restore, rollback readiness, release readiness, and
regression safety for Phases 1-12.

## Prerequisites

- Python 3.11 or the repo-pinned Python version.
- `uv` package manager.
- Branch: `migration/backend-frontend-split`.
- Runtime directory configured through `EVONITH_RUNTIME_DIR`.
- `EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me` for auth-protected tests.
- No production secrets required.
- Backend canonical path: `apps.backend_api.app.main:app`.
- Frontend canonical path: `apps/frontend_streamlit/app.py`.
- Full/dev dependencies for the full test suite.
- Docker/systemd/reverse proxy tools only if manually testing those optional paths.

## Environment Setup

Local setup:

```bash
export EVONITH_RUNTIME_DIR=./runtime
export EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me
export BACKEND_API_BASE_URL=http://localhost:8080/api/v1
export EVONITH_DEPLOYMENT_PROFILE=local
export EVONITH_DEPLOYMENT_ENV=local
```

Edge-like setup:

```bash
export EVONITH_RUNTIME_DIR=/var/lib/evonith-bf
export EVONITH_DEPLOYMENT_PROFILE=edge
export EVONITH_EDGE_MODE=true
export EVONITH_UVICORN_WORKERS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

API cutover setup:

```bash
export USE_BACKEND_API=true
export USE_BACKEND_API_AUTH=true
export USE_BACKEND_API_ADMIN=true
export USE_BACKEND_API_DATA_EXPLORER=true
export USE_BACKEND_API_DATASETS=true
export USE_BACKEND_API_FEEDBACK=true
export USE_BACKEND_API_MATERIAL_BALANCE=true
export USE_BACKEND_API_RECOMMENDATIONS=true
export USE_BACKEND_API_BLEND_OPTIMIZER=true
export USE_BACKEND_API_COPILOT=true
export USE_BACKEND_API_FURNACEMIND=true
export USE_BACKEND_API_OPS=true
```

Direct-mode rollback setup:

```bash
export USE_BACKEND_API=false
export EVONITH_ALLOW_DIRECT_MODE_FALLBACK=true
```

## Automated Test Commands

```bash
uv run pytest furnace-data-service/tests -q
uv run pytest tests/frontend -q
uv run pytest tests/integration -q
uv run pytest tests/dependency -q
uv run pytest tests/structure -q
uv run pytest tests/deployment -q
uv run pytest tests -q

uv run python scripts/export_backend_openapi.py
python scripts/check_repository_structure.py
python scripts/check_import_boundaries.py
python scripts/check_dependency_profiles.py
uv run python scripts/check_backend_minimal_startup.py
python scripts/check_frontend_api_imports.py
python scripts/bootstrap_runtime.py --dry-run
python scripts/validate_deployment.py --profile local --offline
python scripts/validate_api_cutover.py --allow-partial --json
python scripts/backup_runtime.py --dry-run
python scripts/verify_release_readiness.py --allow-dirty --skip-tests
```

## Manual Backend Verification

1. Bootstrap runtime: `python scripts/bootstrap_runtime.py --create`.
2. Start backend:
   `EVONITH_RUNTIME_DIR=./runtime EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080`.
3. Run smoke test:
   `python scripts/smoke_test_deployment.py --backend-url http://localhost:8080/api/v1 --skip-auth`.
4. Verify `/api/v1/health`, `/api/v1/readiness`, `/api/v1/status`, and representative endpoints from all phases.
5. Stop backend.
6. Start backend with compatibility command and verify `/api/v1/health`.

## Manual Frontend Verification

1. Start backend.
2. Start frontend:
   `BACKEND_API_BASE_URL=http://localhost:8080/api/v1 streamlit run apps/frontend_streamlit/app.py`.
3. Verify app loads.
4. Verify backend status badge.
5. Enable API cutover flags.
6. Verify pages use API mode.
7. Stop backend and verify frontend shows clean backend unavailable behavior.
8. Switch `USE_BACKEND_API=false` and verify direct-mode fallback remains available.
9. Start frontend compatibility command: `streamlit run src/app.py`.

## API Cutover Verification

1. Set all `USE_BACKEND_API_*` flags true.
2. Run `python scripts/validate_api_cutover.py --strict --backend-url http://localhost:8080/api/v1`.
3. Verify all required endpoints exist.
4. Verify frontend API adapters import.
5. Verify direct-mode rollback flags are documented.
6. Verify no live AI/vector/LLM services are required unless explicitly enabled.

## Backup and Restore Verification

1. Create sample files under a test runtime directory.
2. Run `python scripts/backup_runtime.py --dry-run`.
3. Run real test backup:
   `python scripts/backup_runtime.py --output ./runtime/backups/test-backup.tar.gz`.
4. Verify manifest exists in archive.
5. Run restore dry-run:
   `python scripts/restore_runtime.py --dry-run --backup ./runtime/backups/test-backup.tar.gz --target-runtime-dir ./runtime-restore-test`.
6. Run restore to isolated test directory with `--apply`.
7. Verify restored files.
8. Verify restore refuses path traversal fixture if tested.

## Rollback Verification

1. Run frontend in API mode.
2. Confirm backend-backed pages work.
3. Set `USE_BACKEND_API=false` or specific page flags false.
4. Restart frontend.
5. Confirm direct-mode fallback is available.
6. Confirm backend can remain running while frontend rolls back.
7. Confirm no runtime data is deleted.
8. Review rollback guide.

## Phase 12 Repository-Structure Regression Verification

Because Phase 13 deployment depends on Phase 12 canonical structure, verify:

1. `apps/backend_api/app/main.py` imports.
2. `apps/frontend_streamlit/app.py` exists.
3. Old backend shim imports.
4. Old frontend shim exists.
5. OpenAPI old/new paths match.
6. `check_repository_structure.py` passes.
7. `check_import_boundaries.py` passes.
8. `check_dependency_profiles.py` passes.
9. `check_backend_minimal_startup.py` passes.
10. `check_frontend_api_imports.py` passes.
11. Edge scripts point to canonical paths.

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
- Phase 12 repository-structure tests pass.
- Full test suite passes.

## Expected Outcomes

- All tests pass.
- Runtime bootstrap works safely.
- Deployment validation catches unsafe production config.
- Smoke tests verify backend health and representative APIs.
- API cutover validation verifies flags and endpoints.
- Backup/restore scripts are safe and non-destructive by default.
- Edge scripts use canonical paths and conservative resource defaults.
- Systemd/reverse proxy examples contain no secrets.
- Rollback guide supports direct-mode fallback.
- Phase 12 canonical and compatibility paths still work.
- No direct-mode fallback is removed.
- No API contract is changed.

## Troubleshooting

- Runtime directory not writable.
- Unsafe runtime path rejected.
- Placeholder secret in production-like profile.
- Backend not running for smoke test.
- `BACKEND_API_BASE_URL` incorrect.
- Missing API cutover flag.
- Frontend cannot reach backend.
- Backup archive permission error.
- Restore target already has files.
- Systemd `EnvironmentFile` path missing.
- Reverse proxy upload limit too small.
- Edge device low disk.
- Port already in use.
- Compatibility command broken.
- Optional service degraded but deployment otherwise healthy.
