# Phase 3 Frontend API Client

Phase 3 adds a Streamlit/frontend API client foundation for the versioned
backend. Existing Streamlit pages still use their current direct-mode data
paths unless a future phase enables a guarded page-specific API path.

## Audit

| Area | Current state | Phase 3 action |
| --- | --- | --- |
| Streamlit entrypoint | `src/app.py` sets page config, initializes runtime/logging, gates login, and registers pages | Added a small backend status badge after login |
| Pages | Welcome, Data Explorer, V-Board, V-Sense, CoPilot, Material Balance, FurnaceMind, Blend Optimizer, Feedback | No page rewrite or full page migration |
| Direct imports | Pages still import `furnace_data`, `data.*`, `utils.*`, `agents.*`, and geometry/domain modules directly | Documented as expected until later phases |
| Backend base URL | Phase 2 exposes `/api/v1` from the FastAPI sidecar | Added `BACKEND_API_BASE_URL`, defaulting to `http://localhost:8080/api/v1` |
| Available API endpoints | `/api/v1/health`, `/api/v1/readiness`, `/api/v1/status/runtime`, `/api/v1/data/...`, `/api/v1/datasets/...` | Client and status service use health/readiness only |
| Safe first integration | Health/readiness are stable, lightweight, and require no external services | Status badge checks backend availability/readiness |
| Data Explorer candidate | Data Explorer already maps conceptually to `/api/v1/data/...`, but the page has rich direct-mode state and large data flows | Deferred to Phase 4; no API mode wiring in this phase |

## What Changed

- Added environment-driven frontend settings and API migration feature flags.
- Added frontend-safe API error types.
- Added a reusable synchronous `ApiClient` using `httpx`.
- Added backend health/readiness status helpers.
- Added a small sidebar status badge controlled by `SHOW_BACKEND_STATUS_BADGE`.
- Added tests for settings, request IDs, retries, structured errors, status handling, and import boundaries.

## What Did Not Change

- No Streamlit page was fully migrated to API calls.
- Direct-mode data loading remains in place.
- No auth/login migration was attempted.
- Feedback, Material Balance, Recommendations, Blend Optimizer, AI Copilot, and FurnaceMind were not migrated.
- Backend API contracts were not changed.
- No database schemas were changed.
- No React or other frontend framework was added.

## New Frontend Files

- `src/config/frontend_settings.py`
- `src/services/api_errors.py`
- `src/services/api_client.py`
- `src/services/backend_status.py`
- `src/ui/backend_status_badge.py`

## Environment Variables

```bash
BACKEND_API_BASE_URL=http://localhost:8080/api/v1
USE_BACKEND_API=false
BACKEND_API_TIMEOUT_SECONDS=30
BACKEND_API_CONNECT_TIMEOUT_SECONDS=5
BACKEND_API_MAX_RETRIES=1
BACKEND_API_VERIFY_SSL=true
SHOW_BACKEND_STATUS_BADGE=true
BACKEND_API_HEALTH_PATH=/health
BACKEND_API_READINESS_PATH=/readiness
```

Page-specific flags default to false:

```bash
USE_BACKEND_API_DATA_EXPLORER=false
USE_BACKEND_API_DATASETS=false
USE_BACKEND_API_FEEDBACK=false
USE_BACKEND_API_MATERIAL_BALANCE=false
USE_BACKEND_API_RECOMMENDATIONS=false
USE_BACKEND_API_BLEND_OPTIMIZER=false
USE_BACKEND_API_COPILOT=false
USE_BACKEND_API_FURNACEMIND=false
```

`USE_BACKEND_API=true` only affects features implemented in current and later
phases. Most pages still use direct mode after Phase 3.

## Local Development

Terminal 1:

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Terminal 2:

```bash
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 \
USE_BACKEND_API=false \
SHOW_BACKEND_STATUS_BADGE=true \
streamlit run src/app.py
```

Optional API mode flag:

```bash
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 \
USE_BACKEND_API=true \
streamlit run src/app.py
```

## Status Badge

The badge appears in the Streamlit sidebar after login when
`SHOW_BACKEND_STATUS_BADGE=true`. It checks `/health` and `/readiness`, caches
the result briefly, and shows one of:

- Backend API available
- Backend API unavailable
- Backend API not ready

Backend downtime is reported as a clean unavailable state and does not stop page
loading.

## Error Handling

The client parses Phase 2 structured errors:

```json
{
  "request_id": "req-id",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {}
  }
}
```

It raises frontend-safe exceptions that preserve the status code, backend error
code, request id, and details without exposing stack traces.

## Request IDs

`ApiClient` generates an `X-Request-ID` for every request unless one is provided.
It captures the backend `X-Request-ID` response header and stores it on the
client as `last_response_request_id`.

## Tests

```bash
pytest tests/frontend -q
pytest furnace-data-service/tests -q
```

Smoke checks:

```bash
python -c "from src.services.api_client import ApiClient; print(ApiClient().base_url)"
python -c "from src.services.backend_status import check_backend_health; print(check_backend_health())"
```

## Known Limitations

- Data Explorer API-mode wiring is deferred. Phase 4 should decide how to avoid
  large JSON transfers and preserve the existing UI state behavior.
- The status badge uses health/readiness only; it does not validate data,
  database, InfluxDB, Qdrant, or LLM connectivity.
- Page-specific flags exist but are intentionally false by default.

## Phase 4 Follow-Up

- Add a guarded Data Explorer adapter for `/api/v1/data/...` if endpoint
  behavior and payload sizes are acceptable.
- Start replacing direct page imports with `src/services/api_client.py` calls one
  page at a time.
- Add endpoint-specific frontend services only where they reduce page coupling.
- Keep direct-mode fallback until each migrated page is stable.
