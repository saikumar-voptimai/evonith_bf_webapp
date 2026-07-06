# Phase 9 Test Execution Report

## Execution Summary

Phase 9 FurnaceMind verification passed:

- Backend FurnaceMind focused tests: `8 passed`
- Frontend FurnaceMind flag/adapter/boundary tests: `18 passed`
- Phase 9 integration flow: `1 passed`
- Backend regression suite: `88 passed`
- Frontend suite: `73 passed`
- Integration suite: `4 passed`
- Full root suite: `301 passed`
- OpenAPI export: passed
- Import and boundary checks: passed

Previously documented broad-suite failures from Phase 8 were fixed during Phase
9 verification with focused compatibility changes outside the FurnaceMind API
surface.

## Test Environment

| Item | Value |
|---|---|
| Branch | migration/backend-frontend-split |
| Phase | Phase 9 - FurnaceMind API Migration |
| Python version | 3.12.12 |
| OS/Environment | Windows / local Codex workspace |
| EVONITH_RUNTIME_DIR | `./runtime` unless test overrides |
| BACKEND_API_BASE_URL | `http://localhost:8080/api/v1` |
| Auth mode | direct/api/test |
| FurnaceMind auth required | true by default; mocked in focused tests where needed |
| FurnaceMind provider mode | disabled by default; mock in tests |
| FurnaceMind memory mode | disabled by default; fake vector backend in tests |
| Test date | 2026-07-04 |

## Automated Commands

| Command | Result | Notes |
|---|---|---|
| `uv run python -m py_compile furnace-data-service/app/api/v1/routes/furnacemind.py furnace-data-service/app/api/v1/schemas/furnacemind.py furnace-data-service/app/repositories/furnacemind_conversation_repository.py furnace-data-service/app/repositories/furnacemind_document_repository.py furnace-data-service/app/repositories/furnacemind_run_repository.py furnace-data-service/app/services/furnacemind_document_service.py furnace-data-service/app/services/furnacemind_event_service.py furnace-data-service/app/services/furnacemind_llm_service.py furnace-data-service/app/services/furnacemind_memory_service.py furnace-data-service/app/services/furnacemind_prompt_service.py furnace-data-service/app/services/furnacemind_retrieval_service.py furnace-data-service/app/services/furnacemind_safety_service.py furnace-data-service/app/services/furnacemind_service.py furnace-data-service/app/services/furnacemind_tool_executor.py furnace-data-service/app/services/furnacemind_tool_registry.py src/services/furnacemind_api.py` | Passed | Phase 9 files compile |
| `uv run pytest furnace-data-service/tests/test_furnacemind_repositories.py furnace-data-service/tests/test_furnacemind_services.py furnace-data-service/tests/test_api_v1_furnacemind.py -q` | Passed | 8 passed, 1 warning |
| `uv run pytest tests/frontend/test_furnacemind_api.py tests/frontend/test_phase4_feature_flags.py tests/frontend/test_import_boundaries.py -q` | Passed | 18 passed |
| `uv run pytest tests/integration/test_phase9_furnacemind_flow.py -q` | Passed | 1 passed, 2 warnings |
| `uv run pytest furnace-data-service/tests -q` | Passed | 88 passed, 2 warnings |
| `uv run pytest tests/frontend -q` | Passed | 73 passed |
| `uv run pytest tests/integration -q` | Passed | 4 passed, 4 warnings |
| `uv run pytest tests -q` | Passed | 301 passed, 5 warnings |
| `uv run python scripts/export_backend_openapi.py` | Passed | Exported `docs/api/openapi-v1.json` |
| Backend app import check | Passed | Printed `Evonith BF Backend API` |
| Frontend FurnaceMind adapter import check | Passed | Printed `furnacemind api import ok` |
| Backend settings import check | Passed | Printed `True False` for auth required and memory disabled |
| Boundary `rg` checks | Passed | No Streamlit imports in backend app; no backend imports in FurnaceMind adapter; no unsafe code/shell execution references in backend FurnaceMind service/route files |

## Backend API Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| API-FM-001 | FurnaceMind config | Wrapped config response | Passed | Passed | Safe defaults visible |
| API-FM-002 | Auth required | Unauthenticated config blocked | Passed | Passed | |
| API-FM-003 | Conversation create/list/get/update/archive | Owner-scoped state | Passed | Passed | |
| API-FM-004 | Message create/list | Sanitized message persistence | Passed | Passed | |
| API-FM-005 | Run without LLM | Deterministic answer and events | Passed | Passed | |
| API-FM-006 | Run with mock LLM | Mock provider answer | Passed | Passed | No real provider SDK required |
| API-FM-007 | Document upload/list/get/delete | Runtime storage and metadata | Passed | Passed | No internal paths exposed |
| API-FM-008 | Document index | Disabled/fake memory behavior | Passed | Passed | |
| API-FM-009 | Tools list/run warnings | Disabled by default, allowlisted when enabled | Passed | Passed | |
| API-FM-010 | Artifact download | Workflow-checked download | Passed | Passed | |
| API-FM-011 | Message feedback | Feedback stored by message | Passed | Passed | |
| API-FM-012 | OpenAPI includes FurnaceMind | Routes in schema | Passed | Passed | |

## Service Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| S-FM-REPO-001 | SQLite schema creation | Lazy runtime DB setup | Passed | Passed | |
| S-FM-REPO-002 | Conversation ownership | Cross-user access blocked | Passed | Passed | |
| S-FM-SAFE-001 | Redaction | Secrets and runtime paths removed | Passed | Passed | |
| S-FM-SAFE-002 | Message/prompt caps | Oversized text truncated or rejected | Passed | Passed | |
| S-FM-DOC-001 | Filename sanitization | Traversal names rejected/sanitized | Passed | Passed | |
| S-FM-DOC-002 | Type and size validation | Unsafe uploads rejected | Passed | Passed | |
| S-FM-MEM-001 | Disabled memory | Structured disabled result | Passed | Passed | |
| S-FM-MEM-002 | Fake vector backend | Test-only index/search works | Passed | Passed | |
| S-FM-LLM-001 | Disabled provider | Structured disabled result | Passed | Passed | |
| S-FM-LLM-002 | Mock provider | Mock response returned | Passed | Passed | |
| S-FM-TOOL-001 | Tool allowlist | Unknown and unsafe tools rejected | Passed | Passed | |
| S-FM-RUN-001 | Events | Polling events are ordered and redacted | Passed | Passed | |

## Frontend Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| F-FM-001 | Config adapter | Calls `/furnacemind/config` | Passed | Passed | Auth header supported |
| F-FM-002 | Conversation adapter | Calls conversation endpoints | Passed | Passed | |
| F-FM-003 | Message/run adapter | Calls message and run endpoints | Passed | Passed | |
| F-FM-004 | Document adapter | Calls upload/list/delete/index endpoints | Passed | Passed | |
| F-FM-005 | Tool adapter | Calls tools endpoint | Passed | Passed | |
| F-FM-006 | Artifact URL | Uses backend base URL | Passed | Passed | |
| F-FM-007 | Feedback adapter | Calls message feedback endpoint | Passed | Passed | |
| F-FLAG-001 | FurnaceMind flag false | Direct mode selected | Passed | Passed | |
| F-FLAG-002 | FurnaceMind flag true | API mode selected | Passed | Passed | |
| F-BOUNDARY-001 | Frontend adapter imports | No backend/DB/provider imports | Passed | Passed | |

## Integration Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| INT-FM-001 | Unauthenticated FurnaceMind config | Rejected | Passed | Passed | |
| INT-FM-002 | Login and token reuse | Authenticated requests work | Passed | Passed | |
| INT-FM-003 | Conversation/message/run/events | Complete chat flow works | Passed | Passed | |
| INT-FM-004 | Mock LLM and artifact download | Artifact returned and downloadable | Passed | Passed | |
| INT-FM-005 | Document upload/index | Runtime document flow works | Passed | Passed | |
| INT-FM-006 | Feedback | Message feedback persists | Passed | Passed | |
| INT-FM-007 | Prior API regressions | Auth/admin/data/datasets/feedback/domain/copilot endpoints still respond | Passed | Passed | |

## Regression Test Results

| Phase | Regression Area | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| Phase 1 | Runtime paths | Runtime behavior remains available | Covered by suites | Passed | FurnaceMind writes under runtime |
| Phase 2 | API foundation | Health/errors/CORS/imports pass | Backend suite passed | Passed | |
| Phase 3 | ApiClient/status | Frontend client tests pass | Frontend suite passed | Passed | |
| Phase 4 | Data/datasets | Data and dataset tests pass | Full root suite passed | Passed | Includes compatibility fixes |
| Phase 5 | Auth/admin | Auth and admin tests pass | Backend and integration suites passed | Passed | |
| Phase 6 | Feedback | Feedback tests pass | Backend and integration suites passed | Passed | |
| Phase 7 | Domain compute | Domain compute APIs pass | Backend and integration suites passed | Passed | |
| Phase 8 | AI Copilot | Copilot tests pass | Backend/frontend/integration suites passed | Passed | |
| Phase 9 | FurnaceMind | FurnaceMind focused tests pass | Focused suites passed | Passed | |
| Full suite | `uv run pytest tests -q` | All pass | 301 passed, 5 warnings | Passed | |

## Compatibility Fixes

| Area | Failure Summary | Fix Applied | Rerun Result |
|---|---|---|---|
| BMO fuel rates | Regression expected total coke rate `438.0` | Aligned inverse estimate defaults and rounding with contract | Passed |
| Burden distribution | UTC midnight snapshot forward-filled from prior day | Normalized distribution snapshots to daily anchors | Passed |
| Static dataset cache | Stale cutoff/synthetic future fixture dates produced empty cache output | Added loader bridge, stale-cutoff safeguard, dataclass-safe config replacement, and robust clipping | Passed |

## Security Verification

| Check | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|
| Backend no Streamlit imports | No `streamlit` imports in backend app | No matches | Passed | |
| Frontend adapter boundary | No backend app imports in `src/services/furnacemind_api.py` | No matches | Passed | |
| Provider imports lazy | No provider import required for default startup/tests | Covered by import checks | Passed | |
| Qdrant imports lazy | Qdrant only imported when explicitly enabled | Covered by tests/import checks | Passed | |
| Raw docs to provider disabled | Raw document text excluded by default | Covered by tests | Passed | |
| Secrets redacted | Sensitive keys and token-like strings redacted | Covered by tests | Passed | |
| Code execution blocked | Unsafe option rejected | Covered by service behavior | Passed | |
| Shell execution blocked | Unsafe option rejected | Covered by service behavior | Passed | |
| Artifact path safety | Download validates artifact ID and workflow | Covered by route/integration tests | Passed | |

## Final Readiness Status

Overall Phase 9 status: Implemented and verified.

All tests passing: Yes. `uv run pytest tests -q` reports 301 passed and 5
warnings.

Ready for Phase 10: Yes for the Phase 9 FurnaceMind API surface.

Summary:
Phase 9 adds a feature-flagged backend FurnaceMind API while preserving direct
mode. LLM calls, Qdrant/vector memory, embeddings, tools, code execution, and
shell execution remain disabled by default.
