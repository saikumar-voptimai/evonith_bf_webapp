# Phase 8 Test Execution Report

## Execution Summary

Phase 8 scoped verification passed:

- Backend Copilot tests: `20 passed`
- Frontend Copilot and boundary tests: `16 passed`
- Phase 8 integration flow: `1 passed`
- Backend regression suite: `80 passed`
- Frontend suite: `68 passed`
- Integration suite: `3 passed`
- OpenAPI export: passed
- Import and boundary checks: passed

The broad command `uv run pytest tests -q` still fails outside the Phase 8
Copilot surface with `285 passed, 10 failed, 4 warnings`. The failures are the
same existing BMO fuel-rate, burden-distribution, and static-dataset cache
failures previously tracked in Phase 7. They were not fixed here to avoid
changing unrelated business logic during the Copilot migration.

## Test Environment

| Item | Value |
|---|---|
| Branch | migration/backend-frontend-split |
| Phase | Phase 8 - AI Copilot API Migration |
| Python version | 3.12.12 |
| OS/Environment | Windows / local Codex workspace |
| EVONITH_RUNTIME_DIR | `./runtime` unless test overrides |
| BACKEND_API_BASE_URL | `http://localhost:8080/api/v1` |
| Auth mode | direct/api/test |
| Copilot auth required | true by default; false or mocked in focused tests where needed |
| Copilot provider mode | disabled by default; mock in tests |
| Test date | 2026-07-04 |

## Automated Commands

| Command | Result | Notes |
|---|---|---|
| `uv run pytest furnace-data-service/tests/test_copilot_safety_service.py furnace-data-service/tests/test_copilot_data_service.py furnace-data-service/tests/test_copilot_anomaly_service.py furnace-data-service/tests/test_copilot_context_service.py furnace-data-service/tests/test_copilot_llm_service.py furnace-data-service/tests/test_copilot_service.py furnace-data-service/tests/test_api_v1_copilot.py -q` | Passed | 20 passed, 1 warning |
| `uv run pytest tests/frontend/test_copilot_api.py tests/frontend/test_phase4_feature_flags.py tests/frontend/test_import_boundaries.py -q` | Passed | 16 passed |
| `uv run pytest tests/integration/test_phase8_copilot_flow.py -q` | Passed | 1 passed, 2 warnings |
| `uv run pytest furnace-data-service/tests -q` | Passed | 80 passed, 2 warnings |
| `uv run pytest tests/frontend -q` | Passed | 68 passed |
| `uv run pytest tests/integration -q` | Passed | 3 passed, 3 warnings |
| `uv run pytest tests -q` | Failed | 285 passed, 10 failed, 4 warnings; failures listed below |
| `uv run python scripts/export_backend_openapi.py` | Passed | Exported `docs/api/openapi-v1.json` |
| Backend app import check | Passed | Printed `Evonith BF Backend API` |
| Frontend Copilot adapter import check | Passed | Printed `copilot api import ok` |
| Boundary `rg` checks | Passed | No Streamlit imports in backend app; no backend imports in Copilot adapter; no FurnaceMind/Qdrant references in Copilot backend files |

## Backend API Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| API-COP-001 | Copilot config | Wrapped config response | Passed | Passed | Includes LLM disabled warning |
| API-COP-002 | Recent data from input rows | JSON-safe rows and summary | Passed | Passed | |
| API-COP-003 | Anomaly analysis | Stable anomaly signals | Passed | Passed | |
| API-COP-004 | Analyze without LLM | Deterministic non-LLM answer | Passed | Passed | |
| API-COP-005 | Analyze with mock LLM | Mock provider answer | Passed | Passed | No real provider SDK required |
| API-COP-006 | Analyze export | JSON artifact is created | Passed | Passed | |
| API-COP-007 | Artifact download | Workflow-checked file response | Passed | Passed | |
| API-COP-008 | Job endpoint | Inline job status returned | Passed | Passed | |
| API-COP-009 | Auth required | Unauthenticated request blocked | Passed | Passed | |
| API-COP-010 | OpenAPI includes Copilot | Copilot routes in schema | Passed | Passed | |

## Service Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| S-COP-SAFE-001 | Redaction | Secrets/tokens removed | Passed | Passed | |
| S-COP-SAFE-002 | Row cap | Context rows capped | Passed | Passed | |
| S-COP-SAFE-003 | Prompt cap | Prompt truncated with warning | Passed | Passed | |
| S-COP-DATA-001 | Input rows | DataFrame preview returned | Passed | Passed | |
| S-COP-DATA-002 | Empty rows | Empty warning returned | Passed | Passed | |
| S-COP-DATA-003 | Invalid shape | Structured request error | Passed | Passed | |
| S-COP-ANOM-001 | Numeric anomaly | Severity/signals returned | Passed | Passed | |
| S-COP-ANOM-002 | Invalid input | `COPILOT_ANOMALY_INPUT_INVALID` | Passed | Passed | |
| S-COP-CTX-001 | Raw data excluded | Warning returned | Passed | Passed | |
| S-COP-LLM-001 | Disabled provider | `COPILOT_LLM_DISABLED` | Passed | Passed | |
| S-COP-LLM-002 | Mock provider | Mock response returned | Passed | Passed | |
| S-COP-LLM-003 | Timeout simulation | `COPILOT_LLM_TIMEOUT` | Passed | Passed | |
| S-COP-LLM-004 | Mock isolation | No `openai` import required | Passed | Passed | |

## Frontend Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| F-COP-001 | Config adapter | Calls `/copilot/config` | Passed | Passed | Auth header supported |
| F-COP-002 | Recent data adapter | Calls `/copilot/recent-data` | Passed | Passed | |
| F-COP-003 | Anomaly adapter | Calls `/copilot/anomaly` | Passed | Passed | |
| F-COP-004 | Analyze adapter | Calls `/copilot/analyze` | Passed | Passed | |
| F-COP-005 | Job adapter | Calls `/copilot/jobs` and job status | Passed | Passed | |
| F-COP-006 | Artifact URL | Uses backend base URL | Passed | Passed | |
| F-FLAG-001 | Copilot flag false | Direct mode selected | Passed | Passed | |
| F-FLAG-002 | Copilot flag true | API mode selected | Passed | Passed | |
| F-BOUNDARY-001 | Frontend adapter imports | No backend/DB/provider imports | Passed | Passed | |

## Integration Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| INT-COP-001 | Unauthenticated Copilot config | Rejected | Passed | Passed | |
| INT-COP-002 | Login and token reuse | Authenticated requests work | Passed | Passed | |
| INT-COP-003 | Config/recent/anomaly/analyze | Copilot flow works | Passed | Passed | |
| INT-COP-004 | Mock LLM and artifact download | Artifact returned and downloadable | Passed | Passed | |
| INT-COP-005 | Job status | Job completes inline | Passed | Passed | |
| INT-COP-006 | Prior API regressions | Auth/admin/data/datasets/feedback/domain config endpoints still respond | Passed | Passed | |

## Regression Test Results

| Phase | Regression Area | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| Phase 1 | Runtime paths | Runtime behavior remains available | Covered by broader tests | Passed in scoped suites | |
| Phase 2 | API foundation | Health/errors/CORS/imports pass | Backend suite passed | Passed | |
| Phase 3 | ApiClient/status | Frontend client tests pass | Frontend suite passed | Passed | |
| Phase 4 | Data/datasets | Data and dataset APIs pass | Backend suite passed; some root static dataset tests fail | Partial | Existing root-suite failures |
| Phase 5 | Auth/admin | Auth and admin tests pass | Backend and integration suites passed | Passed | |
| Phase 6 | Feedback | Feedback tests pass | Backend and integration suites passed | Passed | |
| Phase 7 | Domain compute | Domain compute APIs pass | Backend and integration suites passed | Passed | |
| Phase 8 | AI Copilot | Copilot focused tests pass | Focused suites passed | Passed | |
| Full suite | `uv run pytest tests -q` | All pass | 285 passed, 10 failed, 4 warnings | Failed | Failures listed below |

## Failed Tests and Fixes

| Test | Failure Summary | Root Cause | Fix Applied | Rerun Result |
|---|---|---|---|---|
| `tests/test_bmo_v4_integration.py::test_blend_fuel_prediction_helper_attaches_model_cost` | Expected fuel-rate estimate differs from current output | Existing BMO fuel-rate formula/defaults mismatch | Not changed in Phase 8 | Still failing in broad suite |
| `tests/test_bmo_v4_integration.py::test_estimated_fuel_rates_use_predicted_cost_and_recent_rates` | Expected total coke rate `438.0`, current output `380.357...` | Existing BMO fuel-rate formula/defaults mismatch | Not changed in Phase 8 | Still failing in broad suite |
| `tests/test_bmo_v4_integration.py::test_estimated_fuel_rates_use_requested_nut_coke_fallback_only` | Expected coke rate `368.0`, current output `310.357...` | Existing BMO fuel-rate formula/defaults mismatch | Not changed in Phase 8 | Still failing in broad suite |
| `tests/test_dataset_service_distribution_sqlalchemy.py::test_fetch_distribution_data_returns_expected_windowed_rows` | Expected burden angle `35.0`, current output `30.0` | Existing burden-distribution day/window behavior | Not changed in Phase 8 | Still failing in broad suite |
| `tests/test_static_dataset_cache.py::test_load_static_dataset_fetches_canonical_table_when_no_copy` | Save rejects empty static dataset | Existing static dataset cutoff/cache behavior | Not changed in Phase 8 | Still failing in broad suite |
| `tests/test_static_dataset_cache.py::test_static_dataset_manager_cleans_before_save` | Save rejects empty static dataset | Existing static dataset cutoff/cache behavior | Not changed in Phase 8 | Still failing in broad suite |
| `tests/test_static_dataset_cache.py::test_static_dataset_delta_fills_missing_pci_quantity_rowwise` | Monkeypatched config is not dataclass for `replace` | Existing static dataset test fixture/config behavior | Not changed in Phase 8 | Still failing in broad suite |
| `tests/test_static_dataset_cache.py::test_static_dataset_manager_does_not_backfill_new_quantity_columns` | Missing `ORE_1_CALC_MT` in combined output | Existing static dataset merge behavior | Not changed in Phase 8 | Still failing in broad suite |
| `tests/test_static_dataset_cache.py::test_static_dataset_manager_raises_when_required_delta_is_empty` | Expected runtime error not raised | Existing static dataset delta behavior | Not changed in Phase 8 | Still failing in broad suite |
| `tests/test_static_dataset_cache.py::test_static_dataset_manager_does_not_median_fill_base_only_columns` | Combined result is empty | Existing static dataset cutoff/cache behavior | Not changed in Phase 8 | Still failing in broad suite |

## Security Verification

| Check | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|
| Backend no Streamlit imports | No `streamlit` imports in backend app | No matches | Passed | |
| Frontend adapter boundary | No backend app imports in `src/services/copilot_api.py` | No matches | Passed | |
| FurnaceMind/Qdrant boundary | No references in Copilot backend files | No matches | Passed | |
| Raw data to provider disabled | Raw rows excluded from provider context by default | Covered by tests | Passed | |
| Secrets redacted | Sensitive keys and token-looking strings redacted | Covered by tests | Passed | |
| Code execution blocked | Unsafe option rejected | Covered by service behavior | Passed | |
| Provider calls disabled by default | LLM calls require explicit configuration | Covered by tests | Passed | |
| Artifact path safety | Download validates artifact ID and workflow | Covered by route test | Passed | |

## Final Readiness Status

Overall Phase 8 status: Implemented; scoped backend/frontend/integration tests
pass.

All tests passing: No. `uv run pytest tests -q` reports 285 passed and 10
failed in existing BMO/static dataset areas outside the Phase 8 Copilot surface.

Ready for Phase 9: Yes for the Phase 8 Copilot API surface, with the root-suite
failures tracked separately.

Blocking issues:
No Phase 8 Copilot API blocker. Full-suite blockers remain in BMO fuel-rate,
burden-distribution, and static-dataset tests.

Summary:
Phase 8 adds a feature-flagged backend AI Copilot API while preserving direct
mode. FurnaceMind, Qdrant memory, document RAG, persistent conversations, and
tool/code execution were not migrated.
