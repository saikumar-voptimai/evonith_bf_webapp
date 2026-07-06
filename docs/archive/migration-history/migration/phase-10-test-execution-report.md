# Phase 10 Test Execution Report

## Execution Summary

Phase 10 operational hardening was verified with focused backend service tests,
backend API tests, frontend adapter tests, integration regression tests, import
checks, OpenAPI export, and boundary checks.

Final status: ready for review.

## Test Environment

- Repository branch: `migration/backend-frontend-split`
- Runtime mode: local development defaults
- Optional services: not required for these tests
- Auth secret used in tests: deterministic test-only values
- Audit storage: runtime-backed SQLite in temporary test directories

## Automated Commands

| Command | Result |
|---|---|
| `uv run pytest furnace-data-service/tests/test_phase10_operational_services.py furnace-data-service/tests/test_api_v1_ops.py -q` | Passed: 7 tests |
| `uv run pytest tests/frontend/test_ops_status_api.py tests/frontend/test_frontend_settings.py tests/frontend/test_import_boundaries.py -q` | Passed: 7 tests |
| `uv run pytest tests/integration/test_phase10_ops_flow.py -q` | Passed: 1 test |
| `uv run pytest furnace-data-service/tests -q` | Passed: 95 tests |
| `uv run pytest tests/frontend -q` | Passed: 75 tests |
| `uv run pytest tests/integration -q` | Passed: 5 tests |
| `uv run pytest tests -q` | Passed: 304 tests, 5 warnings |
| `uv run python scripts/export_backend_openapi.py` | Passed: exported `docs/api/openapi-v1.json` |

## Backend API Test Cases

| ID | Area | Coverage | Result |
|---|---|---|---|
| OPS-001 | Status | Public status returns safe health/runtime summary | Passed |
| OPS-002 | Runtime status | Detailed runtime status requires admin access | Passed |
| OPS-003 | Dependencies | Dependency status is admin-protected and does not probe optional services deeply | Passed |
| OPS-004 | Metrics | Metrics snapshot records requests and hides behind admin auth | Passed |
| OPS-005 | Metrics reset | Reset endpoint is disabled unless explicitly enabled | Passed |
| OPS-006 | Unified jobs | Jobs API returns safe job summaries without result bodies or filesystem paths | Passed |
| OPS-007 | Cleanup dry-run | Dry-run reports candidates without deleting files | Passed |
| OPS-008 | Cleanup run | Cleanup can delete eligible runtime files when requested by admin | Passed |
| OPS-009 | Audit events | Audit event listing is admin-protected and paginated | Passed |
| OPS-010 | Error registry | Error-code registry exposes safe code metadata | Passed |

## Backend Service Test Cases

| ID | Area | Coverage | Result |
|---|---|---|---|
| SVC-001 | Redaction | Tokens, API keys, database URLs, and runtime paths are scrubbed | Passed |
| SVC-002 | Metrics | Request counters, status counters, durations, and error-code counters work in memory | Passed |
| SVC-003 | Audit | Events persist to runtime-backed SQLite with redacted metadata | Passed |
| SVC-004 | Runtime status | Runtime directory status returns relative labels and disk information | Passed |
| SVC-005 | Cleanup | Cleanup validates runtime containment and skips unsafe targets | Passed |
| SVC-006 | Error codes | Registry lookup and list behavior cover operational families | Passed |

## Frontend Test Cases

| ID | Area | Coverage | Result |
|---|---|---|---|
| FE-001 | Settings | `USE_BACKEND_API_OPS` and `SHOW_ADVANCED_BACKEND_STATUS` parse from environment | Passed |
| FE-002 | Status adapter | `src/services/status_api.py` calls backend status endpoints through `ApiClient` | Passed |
| FE-003 | Ops adapter | `src/services/ops_api.py` calls jobs, cleanup, audit, and error-code endpoints through `ApiClient` | Passed |
| FE-004 | Boundaries | Frontend adapters do not import backend implementation modules | Passed |

## Integration Test Cases

| ID | Area | Coverage | Result |
|---|---|---|---|
| INT-001 | Backend flow | Health, readiness, login, status, metrics, runtime, cleanup, audit, and error-code APIs work together | Passed |
| INT-002 | Phase 9 regression | FurnaceMind conversation/run/document endpoints remain usable with Phase 10 middleware enabled | Passed |
| INT-003 | Prior phases | Copilot, feedback, and data API compatibility checks remain covered by the integration suite | Passed |

## Regression Results

- No business page was migrated.
- No direct-mode fallback was removed.
- Legacy routes remain available.
- Phase 9 FurnaceMind regression coverage passed with the new access-log,
  audit, metrics, and redaction hooks enabled.
- Existing backend, frontend, integration, and full root suites passed.

## Security Verification

- Admin-only operations are protected by auth dependencies.
- Runtime status and logs do not expose absolute runtime paths through new APIs.
- Request bodies, prompts, raw documents, uploaded content, provider keys, and
  database URLs are not emitted by the new access logs or operational APIs.
- Audit metadata is redacted before persistence.
- Cleanup validates all delete candidates under `EVONITH_RUNTIME_DIR`.

## Failed Tests And Fixes

One focused API test initially depended on event ordering in the audit list.
The test was corrected to assert on the intended event presence instead of
ordering, and the focused Phase 10 backend tests passed on rerun.

## Skipped Tests

No Phase 10 tests were intentionally skipped.

## Follow-Up Notes

- External metrics export, OpenTelemetry, and dashboard work remain Phase 2+
  operational follow-ups.
- PostgreSQL audit storage is documented as future work; Phase 10 uses safe
  runtime-backed SQLite.
- A scheduled cleanup worker is deferred. Phase 10 exposes explicit admin
  cleanup APIs only.
