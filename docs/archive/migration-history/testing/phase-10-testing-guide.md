# Phase 10 Testing Guide

## Purpose

Use this guide to verify Phase 10 operational hardening: status, dependency
checks, metrics, unified jobs, cleanup, audit events, error-code registry,
structured logging/redaction, frontend adapters, and Phase 9 FurnaceMind
regression safety.

## Prerequisites

- Dependencies installed through the project `uv` environment.
- No InfluxDB, Qdrant, LLM provider, or external audit database is required.
- `EVONITH_RUNTIME_DIR` can point to a temporary or local runtime directory.
- Use only development or test credentials.

## Environment Setup

PowerShell example:

```powershell
$env:EVONITH_RUNTIME_DIR = ".\runtime"
$env:EVONITH_AUTH_SECRET_KEY = "dev-only-secret-change-me"
$env:EVONITH_LOG_FORMAT = "json"
$env:EVONITH_ACCESS_LOG_ENABLED = "true"
$env:EVONITH_METRICS_ENABLED = "true"
$env:EVONITH_AUDIT_LOG_ENABLED = "true"
```

POSIX shell example:

```bash
export EVONITH_RUNTIME_DIR=./runtime
export EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me
export EVONITH_LOG_FORMAT=json
export EVONITH_ACCESS_LOG_ENABLED=true
export EVONITH_METRICS_ENABLED=true
export EVONITH_AUDIT_LOG_ENABLED=true
```

## Automated Test Commands

From the repository root:

```bash
uv run pytest furnace-data-service/tests -q
uv run pytest tests/frontend -q
uv run pytest tests/integration -q
uv run pytest tests -q
uv run python scripts/export_backend_openapi.py
```

Focused Phase 10 checks:

```bash
uv run pytest furnace-data-service/tests/test_phase10_operational_services.py furnace-data-service/tests/test_api_v1_ops.py -q
uv run pytest tests/frontend/test_ops_status_api.py tests/frontend/test_frontend_settings.py tests/frontend/test_import_boundaries.py -q
uv run pytest tests/integration/test_phase10_ops_flow.py -q
```

## Manual Backend Verification

Import/startup check from the repository root:

```bash
uv run python -c "import sys; sys.path.insert(0, 'furnace-data-service'); from app.main import app; print(app.title)"
```

Sidecar startup example:

```bash
cd furnace-data-service
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Verify public endpoints:

```bash
curl http://localhost:8080/api/v1/health
curl http://localhost:8080/api/v1/readiness
curl http://localhost:8080/api/v1/status
```

Verify admin endpoints with an admin bearer token:

```bash
curl -H "Authorization: Bearer <admin-token>" http://localhost:8080/api/v1/status/runtime/details
curl -H "Authorization: Bearer <admin-token>" http://localhost:8080/api/v1/status/dependencies
curl -H "Authorization: Bearer <admin-token>" http://localhost:8080/api/v1/metrics
curl -H "Authorization: Bearer <admin-token>" http://localhost:8080/api/v1/jobs
curl -H "Authorization: Bearer <admin-token>" http://localhost:8080/api/v1/ops/audit/events
curl -H "Authorization: Bearer <admin-token>" http://localhost:8080/api/v1/ops/error-codes
```

Cleanup dry-run should be preferred first:

```bash
curl -X POST -H "Authorization: Bearer <admin-token>" -H "Content-Type: application/json" \
  -d "{\"targets\":[\"temp\",\"jobs\"],\"dry_run\":true}" \
  http://localhost:8080/api/v1/ops/cleanup/dry-run
```

## Manual Frontend Verification

Adapter import checks:

```bash
uv run python -c "from src.services.status_api import get_status; print('status api import ok')"
uv run python -c "from src.services.ops_api import list_jobs; print('ops api import ok')"
```

Optional UI status details:

```bash
SHOW_ADVANCED_BACKEND_STATUS=true USE_BACKEND_API_OPS=true streamlit run src/app.py
```

Expected behavior:

- Existing pages still render.
- Direct-mode fallback behavior is unchanged.
- The backend status badge still works.
- Advanced status details appear only when both frontend flags are enabled.

## Phase 9 FurnaceMind Regression Verification

Run the integration suite:

```bash
uv run pytest tests/integration/test_phase10_ops_flow.py -q
```

Expected behavior:

- FurnaceMind conversation, document, run, and tool endpoints still use the
  Phase 9 API contracts.
- Access logs, metrics, and audit events are best-effort and do not change
  FurnaceMind response bodies.
- Raw prompts, documents, memory content, and provider payloads are not exposed
  through operational APIs.

## Regression Verification

Boundary checks:

```bash
rg -n "import streamlit|from streamlit" furnace-data-service/app
rg -n "furnace-data-service|from app|import app" src/services/status_api.py src/services/ops_api.py
rg -n "Bearer |OPENAI_API_KEY=|QDRANT_API_KEY=|EVONITH_AUTH_SECRET_KEY=.*[A-Za-z0-9]" docs/migration/phase-10-test-execution-report.md docs/testing/phase-10-testing-guide.md
```

The last command may find only the documented development placeholder
`EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me`; it must not find production
secrets.

Diff review:

```bash
git diff --stat
git diff --check
```

## Expected Outcomes

- Backend operational endpoints are additive and admin-protected where needed.
- Runtime status uses safe labels and avoids absolute internal paths.
- Dependency status reports configuration/readiness without requiring optional
  services to be reachable.
- Metrics are in-process and do not require a new service.
- Audit events are redacted and stored under runtime by default.
- Cleanup is dry-run by default and validates runtime containment.
- Frontend status/ops adapters call backend APIs through `ApiClient`.
- No frontend/backend split, business-page migration, or direct-mode removal is
  attempted.

## Troubleshooting

- If admin endpoints return `AUTH_REQUIRED`, login and retry with a bearer
  token.
- If admin endpoints return `FORBIDDEN`, use a user with the admin role.
- If runtime status reports warnings, check `EVONITH_RUNTIME_DIR` permissions
  and free disk space.
- If audit events are unavailable, check runtime write permissions and
  `EVONITH_AUDIT_LOG_ENABLED`.
- If cleanup reports no files, lower the TTL in a test environment or create an
  old file under a supported runtime subdirectory.
- If optional dependencies report degraded status, verify the related feature
  flags and provider configuration.

