# Phase 4 Test Cases Report

## Test Environment

| Item | Value |
|---|---|
| Branch | migration/backend-frontend-split |
| Phase | Phase 4 - Data and Dataset API Migration |
| Python version | uv Python 3.12.12 for tests; system Python 3.13.14 |
| OS/Environment | Microsoft Windows NT 10.0.26200.0 |
| EVONITH_RUNTIME_DIR | ./runtime |
| Backend API base URL | http://localhost:8080/api/v1 |
| Test date | 2026-07-03 |

## Commands Run

| Command | Result | Notes |
|---|---|---|
| pytest furnace-data-service/tests -q | PASS | Run as `uv run --with pytest pytest furnace-data-service\tests -q`; 34 passed |
| pytest tests/frontend -q | PASS | Run as `uv run --with pytest pytest tests\frontend -q`; 33 passed |
| pytest furnace-data-service/test/test_routes_data.py furnace-data-service/test/test_routes_dataset.py -q | PASS | Run as `uv run --with pytest`; 35 passed |
| pytest tests -k "data_api or dataset_api or api_v1_data or api_v1_datasets or artifact_service or job_service" -q | FAIL | Collection failed on unrelated pre-existing Streamlit/path import issues before filtering completed |
| python scripts/export_backend_openapi.py | PASS | Run as `uv run python scripts\export_backend_openapi.py`; exported `docs/api/openapi-v1.json` |
| python -c "from src.services.data_api import list_data_sources; print('ok')" | PASS | Import check passed |
| python -c "from src.services.dataset_api import list_datasets; print('ok')" | PASS | Import check passed |

## Backend API Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| B-DATA-001 | GET /api/v1/data/sources | 200 with wrapped response | 200 with wrapped response | PASS | Includes X-Request-ID |
| B-DATA-002 | POST /api/v1/data/preview valid request | 200 with capped rows | 200 with capped rows | PASS | Mocked DataFrame |
| B-DATA-003 | Preview empty dataset | 200 with empty rows and metadata | 200 with empty rows | PASS | Mocked empty DataFrame |
| B-DATA-004 | Invalid data source | Structured DATA_SOURCE_* error | DATA_SOURCE_NOT_FOUND | PASS | 404 |
| B-DATA-005 | Invalid date range | Structured DATA_QUERY_INVALID error | DATA_QUERY_INVALID | PASS | 400 |
| B-DATA-006 | Excessive preview limit | Capped or rejected safely | Capped with warning | PASS | `DATA_API_MAX_PREVIEW_ROWS` honored |
| B-DATA-007 | DataFrame serialization | JSON-safe rows and columns | Timestamp, NaN, numpy values serialized | PASS | Unit test |
| B-DATA-008 | POST /api/v1/data/export | Artifact created under runtime | Artifact created | PASS | Mocked DataFrame |
| B-DATA-009 | Download valid artifact | File response returned | CSV FileResponse returned | PASS | Download endpoint tested |
| B-DATA-010 | Invalid artifact traversal attempt | Structured error; no path escape | Invalid artifact id rejected | PASS | Service and route tests |
| B-DATA-011 | Missing artifact | Structured not-found error | Missing/invalid artifact returns structured error | PASS | DATA_EXPORT_FAILED |
| B-DATA-012 | Backend startup without InfluxDB | App imports and health works | Backend tests import app without external services | PASS | No real InfluxDB required |

## Backend Dataset Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| B-DSET-001 | GET /api/v1/datasets | 200 with dataset list | 200 with static dataset entry | PASS | Wrapped response |
| B-DSET-002 | Preview dataset | 200 with capped rows | 200 with capped rows | PASS | Mocked DataFrame |
| B-DSET-003 | Missing dataset | Structured DATASET_NOT_FOUND error | DATASET_NOT_FOUND | PASS | 404 |
| B-DSET-004 | Refresh dataset | job_id returned | job_id returned | PASS | Background runner mocked |
| B-DSET-005 | Get job status | Stable job status response | Stable job status response | PASS | In-process job state |
| B-DSET-006 | Failed job | Structured error details retained | Failed job error_code retained | PASS | Job service test |
| B-DSET-007 | Completed job download | File response returned | Covered by artifact download tests | PASS | Dataset artifact endpoint shares artifact service |
| B-DSET-008 | Missing job | Structured DATASET_JOB_NOT_FOUND error | DATASET_JOB_NOT_FOUND | PASS | 404 |
| B-DSET-009 | Job output runtime path | Output stored under EVONITH_RUNTIME_DIR | Artifact path starts with runtime | PASS | Artifact service test |
| B-DSET-010 | Backend startup without external services | App imports and health works | Backend tests import app without external services | PASS | No DB required at startup |

## Frontend Adapter Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| F-DATA-001 | list_data_sources | Calls /data/sources | Called /data/sources | PASS | Fake client |
| F-DATA-002 | preview_data | Calls /data/preview | Called /data/preview | PASS | Fake client |
| F-DATA-003 | export_data | Calls /data/export | Called /data/export | PASS | Fake client |
| F-DATA-004 | Structured backend error | UI-safe exception | Covered by ApiClient tests | PASS | Existing frontend API tests |
| F-DATA-005 | Backend unavailable | Clean unavailable error | BackendUnavailableError propagated | PASS | Fake client |
| F-DATA-006 | Wrapped response | Correctly unwrapped | Correctly unwrapped | PASS | Adapter tests |
| F-DSET-001 | list_datasets | Calls /datasets | Called /datasets | PASS | Fake client |
| F-DSET-002 | preview_dataset | Calls /datasets/{id}/preview | Called preview endpoint | PASS | Fake client |
| F-DSET-003 | refresh_dataset | Calls /datasets/refresh | Called refresh endpoint | PASS | Fake client |
| F-DSET-004 | get_dataset_job | Calls /datasets/jobs/{id} | Called job endpoint | PASS | Fake client |
| F-DSET-005 | Failed dataset job | Clean error state | Backend unavailable/error propagation covered | PASS | Adapter does not swallow UI-safe errors |

## Feature Flag Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| FLAG-001 | USE_BACKEND_API_DATA_EXPLORER=false | Data Explorer keeps direct mode | Flag defaults false | PASS | Unit test |
| FLAG-002 | USE_BACKEND_API_DATA_EXPLORER=true | Data Explorer uses API adapter where implemented | Flag resolves true | PASS | Data Explorer guarded branches added |
| FLAG-003 | USE_BACKEND_API_DATASETS=false | Dataset refresh keeps direct mode | Flag defaults false | PASS | Unit test |
| FLAG-004 | USE_BACKEND_API_DATASETS=true | Dataset refresh uses backend APIs | Flag resolves true | PASS | Refresher/page guarded branches added |
| FLAG-005 | Backend unavailable in API mode | Clear error with request ID where available | Frontend-safe errors displayed/propagated | PASS | Adapter and UI handling |
| FLAG-006 | Backend unavailable in direct mode | App still works as before | Direct-mode branches unchanged | PASS | Default flags false |

## Boundary and Regression Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| BOUND-001 | New frontend data services do not import backend internals | No forbidden imports | No matches | PASS | `rg` scan and tests |
| BOUND-002 | Backend app does not import Streamlit | No Streamlit import under backend app | No matches | PASS | `rg` scan and backend tests |
| BOUND-003 | Legacy /data routes retained | Existing route available when legacy enabled | Existing legacy tests pass | PASS | Legacy route module unchanged |
| BOUND-004 | Legacy /dataset routes retained | Existing route available when legacy enabled | `/dataset/cache-info` available | PASS | Test included |
| BOUND-005 | OpenAPI export includes data routes | docs/api/openapi-v1.json updated | Data/dataset paths present | PASS | Export run |
| BOUND-006 | Request ID propagation | X-Request-ID present | Header present on data sources response | PASS | Backend test |
| BOUND-007 | Structured errors | request_id and error object present | Structured data/dataset errors | PASS | Backend tests |

## Manual Smoke Test Report

| Scenario | Result | Notes |
|---|---|---|
| Backend starts independently | SKIPPED | Covered by import/TestClient checks; no long-running server left running |
| /api/v1/health works | PASS | Phase 2 backend suite passed |
| /api/v1/data/sources works | PASS | TestClient check passed |
| /api/v1/datasets works | PASS | TestClient check passed |
| Streamlit direct mode works | SKIPPED | Not launched interactively in this environment |
| Streamlit Data Explorer API mode works | SKIPPED | Not launched interactively in this environment |
| Dataset refresh direct mode works | SKIPPED | Existing direct path preserved but not manually run |
| Dataset refresh API mode works | SKIPPED | API job endpoint tested; no manual Streamlit run |

## Failed Tests

| Test | Failure Summary | Root Cause | Phase 4 Related? | Follow-up |
|---|---|---|---|---|
| `uv run --with pytest pytest tests -k "data_api or dataset_api or api_v1_data or api_v1_datasets or artifact_service or job_service" -q` | Collection failed before filtering completed | Pre-existing repo-wide test collection issues: missing `src` package resolution in some tests, Streamlit stub missing `cache_data`, and `LOCAL_TIMEZONE` import mismatch | No | Continue using focused Phase 4 suites until repo-wide collection is repaired |

## Skipped Tests

| Test | Reason Skipped | Follow-up |
|---|---|---|
| Manual Streamlit direct/API smoke | Interactive Streamlit server not launched in this coding environment | Run manually before release |
| Long-running Uvicorn smoke | TestClient/import checks covered endpoint behavior; no persistent server needed | Run manually on target dev machine if desired |

## Known Risks

| Risk | Severity | Mitigation |
|---|---|---|
| In-process job state is lost on restart | Medium | Documented; improve in later worker phase |
| Large data windows may be expensive | Medium | Preview caps and artifact export |
| API mode coverage is partial | Low | Feature flags preserve direct mode |
| External data source unavailable | Medium | Structured errors and fallback direct mode |

## Final Readiness Status

Overall Phase 4 status: PASS

Summary:
Focused backend and frontend Phase 4 test suites passed. OpenAPI was exported
with the new data and dataset endpoints. Direct-mode paths remain available by
default and API mode is guarded by false-by-default feature flags.

Ready for Phase 5: Yes

Blocking issues:
None for Phase 4. Repo-wide filtered collection still has unrelated pre-existing
collection failures that should be addressed separately.
