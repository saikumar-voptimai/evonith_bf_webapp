# Phase 11 Testing Guide

## Purpose

This guide verifies dependency grouping, backend/frontend runtime separation,
optional dependency lazy imports, edge runtime defaults, import-boundary checks,
backend minimal startup, frontend API import safety, and regression safety for
Phases 1-10.

## Prerequisites

- Python 3.12 locally, compatible with the repo's `>=3.10` root metadata.
- Package manager: `uv`.
- Branch: `migration/backend-frontend-split`.
- Runtime directory: `./runtime` for local tests.
- Tests use local SQLite/runtime storage or in-memory repositories.
- Set `EVONITH_AUTH_SECRET_KEY` for auth-protected manual checks.
- No production secrets are required.
- Optional AI/vector/LLM/Qdrant/model dependencies are not required for
  backend-base checks.
- Full local/dev dependencies are required for the full test suite.

## Environment Setup

Backend minimal/edge-like setup:

```bash
export EVONITH_RUNTIME_DIR=./runtime
export EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me
export EVONITH_RUNTIME_PROFILE=edge
export EVONITH_EDGE_MODE=true
export EVONITH_ENABLE_OPTIONAL_AI=false
export EVONITH_ENABLE_OPTIONAL_VECTOR=false
export EVONITH_ENABLE_OPTIONAL_LOCAL_LLM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

Frontend setup:

```bash
export BACKEND_API_BASE_URL=http://localhost:8080/api/v1
export USE_BACKEND_API=false
```

Full development setup:

```bash
export EVONITH_RUNTIME_PROFILE=local
export EVONITH_EDGE_MODE=false
```

## Install/Profile Verification Commands

Implemented commands:

```bash
uv run python scripts/check_dependency_profiles.py
uv run python scripts/check_import_boundaries.py
uv run python scripts/check_backend_minimal_startup.py
uv run python scripts/check_frontend_api_imports.py
```

uv group examples:

```bash
uv sync --group backend-base
uv sync --group frontend
uv sync --group dev
```

Profile requirements examples:

```bash
uv pip install -r requirements/backend-base.txt
uv pip install -r requirements/frontend.txt
uv pip install -r requirements/dev.txt
```

## Automated Test Commands

```bash
uv run pytest furnace-data-service/tests -q
uv run pytest tests/frontend -q
uv run pytest tests/integration -q
uv run pytest tests/dependency -q
uv run pytest tests -q
uv run python scripts/export_backend_openapi.py
uv run python scripts/check_import_boundaries.py
uv run python scripts/check_dependency_profiles.py
uv run python scripts/check_backend_minimal_startup.py
uv run python scripts/check_frontend_api_imports.py
```

Import checks:

```bash
cd furnace-data-service
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me python -c "from app.main import app; print(app.title)"

cd ..
python -c "from src.services.status_api import get_status; print('status api import ok')"
python -c "from src.services.furnacemind_api import create_conversation; print('furnacemind api import ok')"
```

Boundary checks:

```bash
grep -R "import streamlit\|from streamlit" furnace-data-service/app || true
grep -R "furnace-data-service\|from app\|import app" src/services || true
grep -R "OPENAI_API_KEY=\|QDRANT_API_KEY=\|EVONITH_AUTH_SECRET_KEY=.*[A-Za-z0-9]" docs/migration/phase-11-test-execution-report.md docs/testing/phase-11-testing-guide.md scripts infra || true
```

The documented `dev-only-secret-change-me` placeholder is not a production
secret.

## Manual Backend Verification

1. Start backend with edge-like env vars.
2. Call `/api/v1/health`.
3. Call `/api/v1/readiness`.
4. Login as admin if required.
5. Call `/api/v1/status/config`.
6. Verify `runtime_profile` and `edge_mode` appear.
7. Call `/api/v1/status/dependencies`.
8. Verify optional AI/vector/local LLM features show disabled or unconfigured,
   not failure.
9. Verify OpenAPI export works.
10. Verify backend logs do not show Streamlit/provider/Qdrant initialization at
    startup.
11. Verify missing optional provider/vector package does not break startup.
12. Verify relevant endpoint returns a structured feature-unavailable or
    dependency-missing error if optional feature is requested without a package.

## Manual Frontend Verification

1. Import frontend API adapters with backend stopped.
2. Start Streamlit in normal direct mode.
3. Verify app starts.
4. Verify backend status badge still works.
5. Enable advanced backend status if needed.
6. Verify dependency profile summary is shown safely.
7. Verify no secrets or internal paths appear.
8. Verify existing direct-mode pages are unaffected.

## Edge Runtime Verification

1. Inspect `scripts/edge_start_backend.sh`.
2. Confirm thread limits are set.
3. Confirm runtime dir is configurable.
4. Confirm no secrets are embedded.
5. Inspect `scripts/edge_start_frontend.sh`.
6. Confirm `BACKEND_API_BASE_URL` is configurable.
7. Inspect systemd examples for placeholders and no secrets.
8. Confirm docs explain `/var/lib/evonith-bf` runtime usage.

## Phase 10 Operational Regression Verification

Because Phase 11 changes runtime/dependency profiles, verify Phase 10 still
works:

1. `/api/v1/status` works.
2. `/api/v1/status/runtime` works for admin.
3. `/api/v1/status/dependencies` works for admin.
4. `/api/v1/metrics` works for admin.
5. `/api/v1/jobs` works for admin.
6. `/api/v1/ops/cleanup/dry-run` works for admin and is non-destructive.
7. Audit events remain redacted.
8. Logs remain redacted.
9. Cleanup still cannot escape runtime.
10. Public health remains safe.

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
- Full test suite passes.

## Expected Outcomes

- All tests pass.
- Dependency groups are defined and documented.
- Backend-base profile excludes Streamlit and heavy optional packages.
- Frontend profile excludes backend internals and heavy backend integrations.
- Backend imports and starts without optional AI/vector/LLM/Qdrant/model
  packages.
- Frontend API adapters import without backend server.
- Edge runtime scripts exist and contain safe resource defaults.
- Optional features return structured unavailable errors when dependencies are
  missing.
- OpenAPI export still works.
- Phase 10 operational endpoints still work.
- No secrets are introduced in docs/scripts/config.

## Troubleshooting

- Missing `EVONITH_AUTH_SECRET_KEY`: set a development-only value for local
  auth checks.
- Wrong package manager command: use `uv run`, `uv sync`, or `uv pip install`
  as shown above.
- Lockfile mismatch: run the relevant `uv` command and review `uv.lock`.
- Missing optional dependency: install the matching profile.
- Feature group not installed: endpoint should return structured dependency
  metadata rather than breaking startup.
- Backend accidentally imports Streamlit: run `scripts/check_import_boundaries.py`.
- Frontend adapter accidentally imports backend internals: run
  `scripts/check_frontend_api_imports.py`.
- Qdrant/provider SDK imported at startup: run
  `scripts/check_backend_minimal_startup.py`.
- Edge script permission issue: execute with `bash scripts/edge_start_backend.sh`
  or set executable bits on Linux.
- Runtime directory not writable: update `EVONITH_RUNTIME_DIR` permissions.
- Phase 10 status endpoint requires admin token for detailed routes.
- Direct-mode imports still exist by design outside frontend API adapters.
