# Phase 10 Operational Hardening

## Phase 10 Goal

Harden the separated backend/frontend system for production operations and
edge-device deployment without migrating another business page or removing any
direct-mode fallback.

## Audit

1. Existing logging behavior:
   Phase 2 had standard text logging configured from
   `EVONITH_BACKEND_LOG_LEVEL`. It did not provide JSON formatting, access logs,
   file rotation, or central redaction.

2. Existing request ID behavior:
   `RequestIdMiddleware` sets `request.state.request_id` and `X-Request-ID`.
   Structured error responses include `request_id`.

3. Existing error response behavior:
   `ApiError` and exception handlers return the Phase 2 envelope:
   `request_id` plus `error.code`, `error.message`, and `error.details`.
   Stack traces are not exposed.

4. Existing health/readiness behavior:
   `/api/v1/health`, `/api/v1/readiness`, and the compatibility
   `/api/v1/status/runtime` endpoint existed. Health is public and minimal.

5. Existing dependency checks:
   Runtime readiness was checked. Optional service configuration was mostly
   implicit in feature services.

6. Existing job/artifact systems:
   Phase 4 dataset jobs use `job_service`. Phase 7/8/9 compute-style workflows
   use `compute_job_service` and `ComputeArtifactService`. Data/dataset exports
   use runtime-backed dataset artifacts.

7. Existing runtime cleanup behavior:
   Dataset artifact expiry helpers existed, but no unified runtime cleanup API
   was available.

8. Existing audit logging:
   No central audit repository or audit API existed.

9. Existing metrics/status behavior:
   No central request metrics endpoint existed. Frontend had a simple health and
   readiness status helper and sidebar badge.

10. Current frontend backend-status behavior:
    `src/services/backend_status.py` checks health/readiness. The sidebar badge
    shows availability and request ID. No advanced ops adapter existed.

11. Security risks in logs/status endpoints:
    Logs could include unredacted paths or secret-like strings if callers logged
    them. Runtime status exposed absolute paths in local/test. There was no
    unified secret scrubber for errors, logs, audit, and metadata.

12. Edge-device operational risks:
    Runtime growth from artifacts, temp files, logs, uploads, and jobs was not
    centrally visible. Optional services could be difficult to diagnose without
    checking individual feature settings.

13. What Phase 10 implements:
    Redaction service, structured logging, access logging middleware, audit
    repository/service/API, runtime and dependency status services, metrics
    service/API, unified jobs API, safe cleanup API, error registry, frontend
    status/ops adapters, optional advanced status badge details, OpenAPI update,
    and focused regression tests.

14. What Phase 10 intentionally defers:
    Prometheus/OpenTelemetry exporters, external queues, ELK/Grafana, final
    dependency grouping, repository split, business-page migrations, direct-mode
    removal, and production PostgreSQL audit storage.

## Backend Endpoints Added

- `GET /api/v1/status`
- `GET /api/v1/status/runtime/details`
- `GET /api/v1/status/dependencies`
- `GET /api/v1/metrics`
- `POST /api/v1/metrics/reset`
- `GET /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `POST /api/v1/ops/cleanup/dry-run`
- `POST /api/v1/ops/cleanup/run`
- `GET /api/v1/ops/audit/events`
- `POST /api/v1/ops/audit/retention`
- `GET /api/v1/ops/error-codes`
- `GET /api/v1/ops/error-codes/{code}`

The existing public `/api/v1/health`, `/api/v1/readiness`, and compatibility
`/api/v1/status/runtime` endpoints remain available.

## Security Model

- Public health remains minimal.
- Detailed status, dependency checks, metrics, unified jobs, cleanup, audit, and
  error-code APIs are admin-protected by default.
- Redaction is applied to logs, audit metadata, error details, and operational
  payloads.
- Request bodies, raw documents, prompts, datasets, Authorization headers,
  provider keys, database URLs, and runtime absolute paths are not logged or
  exposed through Phase 10 APIs.

## Runtime Behavior

Operational state uses `EVONITH_RUNTIME_DIR`:

- Audit SQLite default: `runtime/audit/audit.db`
- Optional file logs: `runtime/logs/backend.log`
- Cleanup targets: `runtime/temp`, `runtime/jobs`,
  `runtime/compute/artifacts`, `runtime/datasets/results/artifacts`, and
  optionally `runtime/logs` / `runtime/uploads` when explicitly enabled.

Cleanup uses dry-run by default and validates every delete target remains under
the runtime directory.

## Frontend Changes

- Added `src/services/status_api.py`.
- Added `src/services/ops_api.py`.
- Added `USE_BACKEND_API_OPS=false`.
- Added `SHOW_ADVANCED_BACKEND_STATUS=false`.
- Existing backend status badge still works. Advanced status details are shown
  only when both Phase 10 frontend flags are enabled.

## Compatibility

- No new business page was migrated.
- Direct-mode fallbacks remain available.
- Legacy backend routes remain available.
- Phase 1-9 API contracts are additive and unchanged except for safer logging,
  metrics, audit, and redaction hooks.
- Phase 9 FurnaceMind was regression-tested with Phase 10 middleware enabled.

## Deferred

- External metrics export.
- Full audit PostgreSQL backend.
- Background cleanup scheduler.
- Per-feature job persistence across process restarts.
- Rich operations dashboard.
- Final production dependency grouping and repository split.

