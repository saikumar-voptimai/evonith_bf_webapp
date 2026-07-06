# Phase 7 Testing Guide

## Purpose

This guide verifies backend-owned Material Balance, Recommendations, Blend
Optimizer APIs, model lazy-loading, compute jobs/artifacts, frontend API mode,
direct-mode backward compatibility, auth integration, and regression safety for
Phases 1-6.

## Prerequisites

- Python 3.12.
- Project dependencies installed in the local `uv` environment.
- Branch: `migration/backend-frontend-split`.
- Runtime directory available through `EVONITH_RUNTIME_DIR`.
- Test database or mocked mode for auth-protected compute APIs.
- `EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me` for auth checks.
- Optional model directory only when testing real local models.
- Backend command: `cd furnace-data-service && uvicorn app.main:app --host 0.0.0.0 --port 8080`.
- Frontend command: `streamlit run src/app.py`.
- No production secrets are required.

## Environment Setup

Direct mode setup:

```bash
export EVONITH_RUNTIME_DIR=./runtime
export BACKEND_API_BASE_URL=http://localhost:8080/api/v1
export EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me
export USE_BACKEND_API=false
export USE_BACKEND_API_MATERIAL_BALANCE=false
export USE_BACKEND_API_RECOMMENDATIONS=false
export USE_BACKEND_API_BLEND_OPTIMIZER=false
```

API mode setup:

```bash
export USE_BACKEND_API=true
export USE_BACKEND_API_AUTH=true
export USE_BACKEND_API_MATERIAL_BALANCE=true
export USE_BACKEND_API_RECOMMENDATIONS=true
export USE_BACKEND_API_BLEND_OPTIMIZER=true
export EVONITH_COMPUTE_REQUIRE_AUTH=true
```

Edge-like compute setup:

```bash
export EVONITH_COMPUTE_THREADPOOL_WORKERS=1
export EVONITH_MODEL_LAZY_LOAD=true
export EVONITH_MODEL_CACHE_MAX_ITEMS=2
export EVONITH_COMPUTE_MAX_JSON_ROWS=5000
export EVONITH_BLEND_OPTIMIZER_MAX_ITERATIONS=1000
```

## Automated Test Commands

```bash
pytest furnace-data-service/tests -q
pytest tests/frontend -q
pytest tests/integration -q
pytest tests -q
python scripts/export_backend_openapi.py
```

Import checks:

```bash
cd furnace-data-service
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me python -c "from app.main import app; print(app.title)"

cd ..
python -c "from src.services.material_balance_api import run_material_balance; print('material balance api import ok')"
python -c "from src.services.recommendations_api import run_recommendations; print('recommendations api import ok')"
python -c "from src.services.blend_optimizer_api import optimize_blend; print('blend optimizer api import ok')"
```

Boundary checks:

```bash
grep -R "import streamlit\|from streamlit" furnace-data-service/app || true
grep -R "furnace-data-service\|from app\|import app" src/services/material_balance_api.py src/services/recommendations_api.py src/services/blend_optimizer_api.py || true
grep -R "src/storage" furnace-data-service/app src/services src/custom_pages src/utils || true
```

## Manual Backend Verification

1. Start backend.
2. Call `/api/v1/health`.
3. Login using `/api/v1/auth/login` if compute auth is required.
4. Call `/api/v1/material-balance/config`.
5. Run `/api/v1/material-balance/run` with a small test payload.
6. Call `/api/v1/recommendations/config`.
7. Run `/api/v1/recommendations/run` with a small test payload.
8. Call `/api/v1/blend-optimizer/context`.
9. Call `/api/v1/blend-optimizer/models`.
10. Run `/api/v1/blend-optimizer/optimize` with a small mocked/test payload.
11. Verify request ID appears in success and error responses.
12. Verify no internal model/runtime paths appear in JSON responses.
13. Verify a large-output request creates an artifact or truncates safely.
14. Verify invalid model/artifact IDs do not allow path traversal.

Example:

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8080/api/v1/material-balance/config
curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer <token>" \
  http://localhost:8080/api/v1/recommendations/run \
  -d '{"input_data":{"signals":{"PCI_KG/THM":5}}}'
```

## Manual Frontend Verification

1. Run Streamlit in direct mode.
2. Verify Material Balance page still works.
3. Verify Recommendations page still works.
4. Verify Blend Optimizer page still works.
5. Start backend.
6. Run Streamlit with `USE_BACKEND_API_AUTH=true` and the three Phase 7 flags true.
7. Login through backend auth mode.
8. Open Material Balance page and run API mode workflow.
9. Open Recommendations page and run API mode workflow.
10. Open Blend Optimizer page and run API mode workflow.
11. Verify results render correctly.
12. Verify download links appear when artifacts are returned.
13. Stop backend and verify API mode shows clean backend unavailable errors.
14. Disable flags and verify direct mode still works without backend.

## Regression Verification

- Phase 1 runtime tests pass.
- Phase 2 API foundation tests pass.
- Phase 3 ApiClient tests pass.
- Phase 4 data/dataset tests pass.
- Phase 5 auth/admin tests pass.
- Phase 6 feedback tests pass.
- Full test suite passes.

## Expected Outcomes

- All tests pass.
- OpenAPI includes Material Balance, Recommendations, and Blend Optimizer endpoints.
- Backend starts independently.
- Backend does not load all models at startup.
- Missing optional models do not break startup.
- Compute runtime files are under `EVONITH_RUNTIME_DIR`.
- Direct-mode pages work.
- API-mode pages work behind flags.
- Unauthorized requests are rejected when compute auth is required.
- Large outputs are capped or exported.
- No internal paths, tokens, or raw full datasets are exposed.
- No AI Copilot/FurnaceMind migration occurred.

## Troubleshooting

- Missing `EVONITH_AUTH_SECRET_KEY`: set a dev-only value for local auth tests.
- Missing auth token in API mode: login through backend auth first.
- Optional model not found: set `EVONITH_MODEL_DIR` or disable model predictions.
- Optional ML dependency missing: install the dependency or rely on structured unavailable errors.
- Runtime directory not writable: set `EVONITH_RUNTIME_DIR` to a writable path.
- Large output truncated: download the returned artifact.
- Optimizer timeout: reduce iterations/candidates.
- No feasible blend solution: check material min/max shares.
- Feature flag not enabled: set the page-specific `USE_BACKEND_API_*` flag.
- Backend unavailable: start the backend or disable API mode.
- 401 means no/invalid authentication; 403 means authenticated but not allowed.
- Direct-mode imports still present by design.
