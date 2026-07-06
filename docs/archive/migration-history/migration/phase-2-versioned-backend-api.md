# Phase 2 Versioned Backend API

Phase 2 promotes the existing FastAPI sidecar into the official backend API
foundation under `/api/v1`. The Streamlit app still runs through the old flow
and has not been migrated to API calls.

## Audit

| Area | Current state | Phase 2 action |
| --- | --- | --- |
| Backend entrypoint | `furnace-data-service/app/main.py` creates the FastAPI app | Refactored into `create_app()` and a global `app` |
| Legacy routes | `/health`, `/data/...`, `/dataset/...` | Retained by default behind `EVONITH_ENABLE_LEGACY_ROUTES=true` |
| Config behavior | Dataset settings live in `app/config.py`; runtime paths use Phase 1 helpers | Added backend API settings in `app/core/config.py` |
| CORS behavior | Previously wildcard origins and headers | Now controlled by `BACKEND_CORS_ORIGINS`, defaults to Streamlit local dev origins |
| Health behavior | Legacy `/health` returns status/version | Added `/api/v1/health`, `/api/v1/readiness`, and `/api/v1/status/runtime` |
| Fragile imports | Legacy route imports load fetcher modules and config, but do not connect to external services at import time | Kept behavior, no optional service startup required |
| Streamlit imports | No `streamlit` imports found under `furnace-data-service/app` | Added tests/checks to preserve this boundary |

## What Changed

- Added environment-backed backend API settings.
- Added request ID middleware using `X-Request-ID`.
- Added structured JSON error responses with stable error codes.
- Added `/api/v1` router and health/readiness/runtime status endpoints.
- Exposed existing data routes under `/api/v1/data/...`.
- Exposed existing dataset behavior under `/api/v1/datasets/...` using thin wrappers.
- Added safe CORS defaults for Streamlit local development.
- Added OpenAPI export support at `docs/api/openapi-v1.json`.

## What Did Not Change

- No Streamlit pages were migrated to API calls.
- No auth/login migration was attempted.
- Feedback, Material Balance, Recommendations, Blend Optimizer, AI Copilot, and FurnaceMind were not migrated.
- Existing legacy FastAPI route files were not removed or renamed.
- No database schema changes were made.
- Optional services such as PostgreSQL, InfluxDB, Qdrant, and LLM providers are still endpoint-specific.

## New API Paths

- `GET /api/v1/health`
- `GET /api/v1/readiness`
- `GET /api/v1/status/runtime`
- Existing data endpoints under `/api/v1/data/...`
- Existing dataset endpoints under `/api/v1/datasets/...`

## Legacy Routes

Legacy routes remain enabled by default:

- `GET /health`
- `/data/...`
- `/dataset/...`

Set `EVONITH_ENABLE_LEGACY_ROUTES=false` to expose only `/api/v1` routes.

## Environment Variables

```bash
EVONITH_RUNTIME_DIR=./runtime
EVONITH_API_PREFIX=/api/v1
EVONITH_BACKEND_ENV=local
EVONITH_BACKEND_LOG_LEVEL=INFO
BACKEND_CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
EVONITH_ENABLE_LEGACY_ROUTES=true
EVONITH_OPENAPI_TITLE=Evonith BF Backend API
EVONITH_OPENAPI_VERSION=0.1.0
EVONITH_OPENAPI_DESCRIPTION=Versioned backend API for Evonith BF web application
```

For edge production, use a persistent runtime directory such as:

```bash
EVONITH_RUNTIME_DIR=/var/lib/evonith-bf
```

Restrict `BACKEND_CORS_ORIGINS` to trusted frontend origins in production.

## Local Startup

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Basic checks:

```bash
curl http://localhost:8080/api/v1/health
curl http://localhost:8080/api/v1/readiness
```

Streamlit still starts separately using the existing flow.

## OpenAPI Export

From the repo root:

```bash
python scripts/export_backend_openapi.py
```

The schema is written to:

```text
docs/api/openapi-v1.json
```

## Testing

```bash
pytest furnace-data-service/tests -q
pytest furnace-data-service/test -q
```

## Known Risks

- Legacy data/dataset routers are still imported into the backend app, so import
  reliability depends on their current lightweight import behavior.
- Legacy route error bodies now include the structured API error envelope, while
  preserving the old `detail` field for compatibility.
- OpenAPI currently includes both versioned and legacy routes when legacy routes
  are enabled.

## Phase 3 Follow-Up

- Add the frontend API client.
- Begin migrating Streamlit pages to `/api/v1` endpoint calls.
- Decide whether legacy routes should be hidden from OpenAPI before removal.
- Introduce backend auth only when the frontend migration plan is ready.
