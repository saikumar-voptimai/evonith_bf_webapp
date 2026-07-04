# Phase 9 Testing Guide

## Purpose

This guide verifies backend-owned FurnaceMind API mode, direct-mode backward
compatibility, auth integration, conversation persistence, document upload and
indexing, run/event polling, prompt/context safety, redaction, optional/mock LLM
behavior, safe tools, artifacts, OpenAPI export, and regression safety for
previous migration phases.

## Prerequisites

- Python 3.12.
- Project dependencies installed in the local `uv` environment.
- Branch: `migration/backend-frontend-split`.
- Runtime directory available through `EVONITH_RUNTIME_DIR`.
- Backend command: `cd furnace-data-service && uvicorn app.main:app --host 0.0.0.0 --port 8080`.
- Frontend command: `streamlit run src/app.py`.
- `EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me` for auth checks.
- No production provider keys, Qdrant instance, embeddings, or tool services are
  required for default or mock tests.

## Environment Setup

Direct mode setup:

```bash
export EVONITH_RUNTIME_DIR=./runtime
export BACKEND_API_BASE_URL=http://localhost:8080/api/v1
export EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me
export USE_BACKEND_API=false
export USE_BACKEND_API_FURNACEMIND=false
```

API mode setup:

```bash
export USE_BACKEND_API=true
export USE_BACKEND_API_AUTH=true
export USE_BACKEND_API_FURNACEMIND=true
export EVONITH_FURNACEMIND_REQUIRE_AUTH=true
```

Safe local non-LLM setup:

```bash
export EVONITH_FURNACEMIND_LLM_ENABLED=false
export EVONITH_FURNACEMIND_ENABLE_PROVIDER_CALLS=false
export EVONITH_FURNACEMIND_ALLOW_RAW_DOCS_TO_LLM=false
export EVONITH_FURNACEMIND_MEMORY_ENABLED=false
export EVONITH_FURNACEMIND_EMBEDDINGS_ENABLED=false
export EVONITH_FURNACEMIND_TOOLS_ENABLED=false
export EVONITH_FURNACEMIND_ENABLE_CODE_EXECUTION=false
export EVONITH_FURNACEMIND_ENABLE_SHELL_EXECUTION=false
```

Mock provider setup for tests:

```bash
export EVONITH_FURNACEMIND_LLM_ENABLED=true
export EVONITH_FURNACEMIND_ENABLE_PROVIDER_CALLS=true
export EVONITH_FURNACEMIND_PROVIDER=mock
```

Fake memory setup for tests:

```bash
export EVONITH_FURNACEMIND_MEMORY_ENABLED=true
export EVONITH_FURNACEMIND_VECTOR_BACKEND=fake
```

Real provider setup for controlled local testing:

```bash
export EVONITH_FURNACEMIND_LLM_ENABLED=true
export EVONITH_FURNACEMIND_ENABLE_PROVIDER_CALLS=true
export EVONITH_FURNACEMIND_PROVIDER=openai
export EVONITH_FURNACEMIND_MODEL=gpt-4o-mini
export EVONITH_FURNACEMIND_API_KEY_ENV=OPENAI_API_KEY
export OPENAI_API_KEY=<secure local secret>
```

Do not commit provider keys, Qdrant keys, uploaded documents, SQLite databases,
or runtime output files.

## Automated Test Commands

Focused Phase 9 checks:

```bash
pytest furnace-data-service/tests/test_furnacemind_repositories.py \
  furnace-data-service/tests/test_furnacemind_services.py \
  furnace-data-service/tests/test_api_v1_furnacemind.py -q

pytest tests/frontend/test_furnacemind_api.py \
  tests/frontend/test_phase4_feature_flags.py \
  tests/frontend/test_import_boundaries.py -q

pytest tests/integration/test_phase9_furnacemind_flow.py -q
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
python -c "from src.services.furnacemind_api import create_conversation; print('furnacemind api import ok')"
python -c "import sys; sys.path.insert(0, 'furnace-data-service'); from app.core.config import BackendSettings; s=BackendSettings(); print(s.furnacemind_require_auth, s.furnacemind_memory_enabled)"
```

Boundary checks:

```bash
rg -n "import streamlit|from streamlit" furnace-data-service/app
rg -n "furnace-data-service|from app|import app" src/services/furnacemind_api.py
rg -n -g "furnacemind*.py" "subprocess|os\.system|exec\(|eval\(" furnace-data-service/app/services furnace-data-service/app/api/v1/routes
```

The boundary `rg` commands should return no matches.

## Manual Backend Verification

1. Start the backend.
2. Call `/api/v1/health`.
3. Login using `/api/v1/auth/login` if FurnaceMind auth is required.
4. Call `/api/v1/furnacemind/config`.
5. Create a conversation.
6. Send a user message.
7. Start a run with `allow_llm=false`.
8. Poll `/api/v1/furnacemind/runs/{run_id}` and
   `/api/v1/furnacemind/runs/{run_id}/events`.
9. Upload a small `.txt`, `.md`, `.csv`, or `.json` document.
10. Index the document with fake memory enabled or confirm disabled-memory
    warnings with memory disabled.
11. List tools and verify they are disabled by default.
12. Enable mock provider in a controlled environment and start a run with
    `allow_llm=true`.
13. Submit feedback for an assistant message.
14. Verify request IDs appear in wrapped responses.
15. Verify internal filesystem paths, bearer tokens, provider prompts, and
    secrets are not exposed in responses.

Example:

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8080/api/v1/furnacemind/config

curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer <token>" \
  http://localhost:8080/api/v1/furnacemind/conversations \
  -d '{"title":"Shift handover"}'

curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer <token>" \
  http://localhost:8080/api/v1/furnacemind/conversations/<conversation_id>/messages \
  -d '{"role":"user","content":"Summarise furnace stability risks for the last shift."}'

curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer <token>" \
  http://localhost:8080/api/v1/furnacemind/conversations/<conversation_id>/runs \
  -d '{"allow_llm":false}'
```

## Manual Frontend Verification

1. Run Streamlit with `USE_BACKEND_API_FURNACEMIND=false`.
2. Open FurnaceMind and verify the existing direct page still loads.
3. Start the backend.
4. Run Streamlit with `USE_BACKEND_API_AUTH=true` and
   `USE_BACKEND_API_FURNACEMIND=true`.
5. Login through backend auth mode if required.
6. Open FurnaceMind and verify the page caption says backend API mode.
7. Create or select a conversation.
8. Send a prompt and verify a deterministic answer when LLM is disabled.
9. Verify run status and event messages render clearly.
10. Upload a small allowed document and verify it appears in the document list.
11. Verify tools render as disabled by default.
12. Submit feedback for an assistant response.
13. Stop the backend and verify API mode shows a clear backend error.
14. Disable the flag and verify direct mode is restored.

## Security And Safety Verification

- `EVONITH_FURNACEMIND_LLM_ENABLED=false` returns deterministic analysis.
- `EVONITH_FURNACEMIND_ENABLE_PROVIDER_CALLS=false` blocks provider calls.
- `EVONITH_FURNACEMIND_ALLOW_RAW_DOCS_TO_LLM=false` excludes raw document text
  from provider prompts.
- `EVONITH_FURNACEMIND_MEMORY_ENABLED=false` avoids Qdrant/vector calls.
- `EVONITH_FURNACEMIND_TOOLS_ENABLED=false` disables tools.
- `EVONITH_FURNACEMIND_ENABLE_CODE_EXECUTION=false` blocks code execution.
- `EVONITH_FURNACEMIND_ENABLE_SHELL_EXECUTION=false` blocks shell execution.
- Prompts, messages, history, context, extracted documents, outputs, and events
  are capped.
- Sensitive keys, token-like strings, and runtime filesystem paths are redacted.
- Artifact download checks artifact workflow.
- Backend FurnaceMind files do not import Streamlit.
- Frontend adapter does not import backend app, database, Influx, Qdrant,
  provider SDK, or legacy direct-mode agent modules.

## Regression Verification

- Phase 1 runtime behavior remains available.
- Phase 2 backend API foundation tests pass.
- Phase 3 frontend API client tests pass.
- Phase 4 data/dataset tests pass.
- Phase 5 auth/admin tests pass.
- Phase 6 feedback tests pass.
- Phase 7 domain compute tests pass.
- Phase 8 AI Copilot tests pass.
- Direct-mode FurnaceMind remains available when the flag is false.
- Full test suite status is documented in the Phase 9 execution report.

## Expected Outcomes

- Focused Phase 9 backend, frontend, and integration tests pass.
- OpenAPI includes all `/api/v1/furnacemind/*` endpoints.
- Backend starts independently and does not import Streamlit.
- API mode is selected only with `USE_BACKEND_API_FURNACEMIND=true`.
- Direct mode remains the default.
- LLM calls, Qdrant/vector memory, embeddings, tools, code execution, and shell
  execution are disabled by default and mockable or fakeable in tests.
- No frontend/backend repository split occurred.

## Troubleshooting

- Missing auth token: login through backend auth or set
  `EVONITH_FURNACEMIND_REQUIRE_AUTH=false` for local unauthenticated testing.
- Backend unavailable: start the backend or disable
  `USE_BACKEND_API_FURNACEMIND`.
- LLM disabled response: set both `EVONITH_FURNACEMIND_LLM_ENABLED=true` and
  `EVONITH_FURNACEMIND_ENABLE_PROVIDER_CALLS=true`, then configure provider
  mode.
- Memory disabled warning: expected unless memory is explicitly enabled.
- Qdrant unavailable: keep memory disabled or use the fake backend for tests.
- Raw document warning: expected when raw document text is excluded from provider
  context.
- Prompt/context too large: reduce message, document, or history size, or
  increase local caps deliberately.
- Artifact not found: verify the artifact ID came from a FurnaceMind response
  and has not expired.
- 401 means no or invalid authentication; 403 means authenticated but not
  permitted or blocked by safety settings.
