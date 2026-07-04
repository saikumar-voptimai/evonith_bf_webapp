# Phase 8 Testing Guide

## Purpose

This guide verifies backend-owned AI Copilot API mode, direct-mode backward
compatibility, auth integration, prompt/context safety, redaction, optional/mock
LLM behavior, jobs/artifacts, OpenAPI export, and regression safety for previous
migration phases.

## Prerequisites

- Python 3.12.
- Project dependencies installed in the local `uv` environment.
- Branch: `migration/backend-frontend-split`.
- Runtime directory available through `EVONITH_RUNTIME_DIR`.
- Backend command: `cd furnace-data-service && uvicorn app.main:app --host 0.0.0.0 --port 8080`.
- Frontend command: `streamlit run src/app.py`.
- `EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me` for auth checks.
- No production provider keys are required for the default or mock tests.

## Environment Setup

Direct mode setup:

```bash
export EVONITH_RUNTIME_DIR=./runtime
export BACKEND_API_BASE_URL=http://localhost:8080/api/v1
export EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me
export USE_BACKEND_API=false
export USE_BACKEND_API_COPILOT=false
```

API mode setup:

```bash
export USE_BACKEND_API=true
export USE_BACKEND_API_AUTH=true
export USE_BACKEND_API_COPILOT=true
export EVONITH_COPILOT_REQUIRE_AUTH=true
```

Safe local non-LLM setup:

```bash
export EVONITH_COPILOT_LLM_ENABLED=false
export EVONITH_COPILOT_ENABLE_PROVIDER_CALLS=false
export EVONITH_COPILOT_ALLOW_RAW_DATA_TO_LLM=false
export EVONITH_COPILOT_ENABLE_CODE_EXECUTION=false
```

Mock provider setup for tests:

```bash
export EVONITH_COPILOT_LLM_ENABLED=true
export EVONITH_COPILOT_ENABLE_PROVIDER_CALLS=true
export EVONITH_COPILOT_PROVIDER=mock
```

Real provider setup for controlled local testing:

```bash
export EVONITH_COPILOT_LLM_ENABLED=true
export EVONITH_COPILOT_ENABLE_PROVIDER_CALLS=true
export EVONITH_COPILOT_PROVIDER=openai
export EVONITH_COPILOT_MODEL=gpt-4o-mini
export EVONITH_COPILOT_API_KEY_ENV=OPENAI_API_KEY
export OPENAI_API_KEY=<secure local secret>
```

Do not commit provider keys or runtime output files.

## Automated Test Commands

Focused Phase 8 checks:

```bash
pytest furnace-data-service/tests/test_copilot_safety_service.py \
  furnace-data-service/tests/test_copilot_data_service.py \
  furnace-data-service/tests/test_copilot_anomaly_service.py \
  furnace-data-service/tests/test_copilot_context_service.py \
  furnace-data-service/tests/test_copilot_llm_service.py \
  furnace-data-service/tests/test_copilot_service.py \
  furnace-data-service/tests/test_api_v1_copilot.py -q

pytest tests/frontend/test_copilot_api.py \
  tests/frontend/test_phase4_feature_flags.py \
  tests/frontend/test_import_boundaries.py -q

pytest tests/integration/test_phase8_copilot_flow.py -q
```

Regression checks:

```bash
pytest furnace-data-service/tests -q
pytest tests/frontend -q
pytest tests/integration -q
pytest tests -q
python scripts/export_backend_openapi.py
```

Import checks:

```bash
python -c "import sys; sys.path.insert(0, 'furnace-data-service'); from app.main import app; print(app.title)"
python -c "from src.services.copilot_api import analyze_copilot; print('copilot api import ok')"
```

Boundary checks:

```bash
rg -n "import streamlit|from streamlit" furnace-data-service/app
rg -n "furnace-data-service|from app|import app" src/services/copilot_api.py
rg -n -g "copilot*.py" "qdrant|FurnaceMind|furnacemind" furnace-data-service/app/services furnace-data-service/app/api/v1/routes
```

The boundary `rg` commands should return no matches.

## Manual Backend Verification

1. Start the backend.
2. Call `/api/v1/health`.
3. Login using `/api/v1/auth/login` if Copilot auth is required.
4. Call `/api/v1/copilot/config`.
5. Call `/api/v1/copilot/recent-data` with small input rows.
6. Call `/api/v1/copilot/anomaly` with the same rows.
7. Call `/api/v1/copilot/analyze` with `allow_llm=false`.
8. Enable the mock provider and call `/api/v1/copilot/analyze` with
   `allow_llm=true`.
9. Call `/api/v1/copilot/jobs` and then
   `/api/v1/copilot/jobs/{job_id}`.
10. Request export output and verify
   `/api/v1/copilot/artifacts/{artifact_id}/download` returns the file.
11. Verify request IDs appear in wrapped responses.
12. Verify internal filesystem paths, bearer tokens, and raw provider prompts
   are not exposed in responses.

Example:

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8080/api/v1/copilot/config

curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer <token>" \
  http://localhost:8080/api/v1/copilot/analyze \
  -d '{"question":"Summarise furnace stability","input_data":[{"timestamp":"2026-07-04T06:00:00Z","top_temperature":825,"pressure_drop":1.8}],"allow_llm":false}'
```

## Manual Frontend Verification

1. Run Streamlit with `USE_BACKEND_API_COPILOT=false`.
2. Open AI Copilot and verify the existing direct page still loads.
3. Start the backend.
4. Run Streamlit with `USE_BACKEND_API_AUTH=true` and
   `USE_BACKEND_API_COPILOT=true`.
5. Login through backend auth mode if required.
6. Open AI Copilot and verify the page caption says backend API mode.
7. Fetch recent data from input rows.
8. Run anomaly analysis.
9. Run deterministic analysis with LLM disabled.
10. Enable mock provider and run LLM analysis in a controlled test environment.
11. Verify warnings render clearly.
12. Verify artifact download buttons use backend URLs.
13. Stop the backend and verify API mode shows a clear backend error.
14. Disable the flag and verify direct mode is restored.

## Security And Safety Verification

- `EVONITH_COPILOT_LLM_ENABLED=false` returns deterministic analysis.
- `EVONITH_COPILOT_ENABLE_PROVIDER_CALLS=false` blocks provider calls.
- `EVONITH_COPILOT_ALLOW_RAW_DATA_TO_LLM=false` excludes sample rows from
  provider context.
- `EVONITH_COPILOT_ENABLE_CODE_EXECUTION=false` rejects code execution requests.
- Prompts and row payloads are capped.
- Sensitive keys and token-like strings are redacted.
- Copilot artifact download checks artifact workflow.
- Backend Copilot files do not import Streamlit.
- Frontend adapter does not import backend app, database, Influx, Qdrant,
  FurnaceMind, or provider SDK modules.

## Regression Verification

- Phase 1 runtime behavior remains available.
- Phase 2 backend API foundation tests pass.
- Phase 3 frontend API client tests pass.
- Phase 4 data/dataset API tests pass in backend suite.
- Phase 5 auth/admin tests pass.
- Phase 6 feedback tests pass.
- Phase 7 domain compute tests pass.
- Direct-mode AI Copilot remains available when the flag is false.
- Full test suite status is documented in the Phase 8 execution report.

## Expected Outcomes

- Focused Phase 8 backend, frontend, and integration tests pass.
- OpenAPI includes all `/api/v1/copilot/*` endpoints.
- Backend starts independently and does not import Streamlit.
- API mode is selected only with `USE_BACKEND_API_COPILOT=true`.
- Direct mode remains the default.
- LLM calls are disabled by default and mockable in tests.
- No FurnaceMind, Qdrant, RAG, persistent conversation, or tool execution
  migration occurred.

## Troubleshooting

- Missing auth token: login through backend auth or set
  `EVONITH_COPILOT_REQUIRE_AUTH=false` for local unauthenticated testing.
- Backend unavailable: start the backend or disable `USE_BACKEND_API_COPILOT`.
- LLM disabled response: set both `EVONITH_COPILOT_LLM_ENABLED=true` and
  `EVONITH_COPILOT_ENABLE_PROVIDER_CALLS=true`, then configure provider mode.
- Raw data warning: expected when raw rows are excluded from provider context.
- Prompt/context too large: reduce row count or increase local caps deliberately.
- Artifact not found: verify the artifact ID came from a Copilot response and
  has not expired.
- 401 means no or invalid authentication; 403 means authenticated but not
  permitted or blocked by safety settings.
