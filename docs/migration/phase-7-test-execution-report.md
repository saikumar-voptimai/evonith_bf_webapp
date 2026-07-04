# Phase 7 Test Execution Report

## Execution Summary

Phase 7 scoped verification passed:

- Backend Phase 7 API tests: `7 passed`
- Backend regression suite: `60 passed`
- Frontend suite: `63 passed`
- Integration suite: `2 passed`
- OpenAPI export: passed
- Boundary checks: passed

The broad command `uv run pytest tests -q` still fails outside the Phase 7 API
surface with `279 passed, 10 failed, 3 warnings`. The remaining failures are in
existing BMO fuel-rate, burden-distribution, and static-dataset cache tests, and
were not fixed here to avoid changing business logic during this migration phase.

## Test Environment

| Item | Value |
|---|---|
| Branch | migration/backend-frontend-split |
| Phase | Phase 7 - Domain Compute API Migration |
| Python version | 3.12.12 |
| OS/Environment | Windows / local Codex workspace |
| EVONITH_RUNTIME_DIR | `./runtime` unless test overrides |
| BACKEND_API_BASE_URL | `http://localhost:8080/api/v1` |
| Auth mode | direct/api/test |
| Compute auth required | true by default; false in focused no-auth tests |
| Model mode | optional local assets / mocked test model |
| Test date | 2026-07-04 |

## Automated Commands

| Command | Result | Notes |
|---|---|---|
| `uv run pytest furnace-data-service/tests/test_api_v1_domain_compute.py -q` | Passed | 7 passed, 1 warning |
| `uv run pytest furnace-data-service/tests -q` | Passed | 60 passed, 2 warnings |
| `uv run pytest tests/frontend -q` | Passed | 63 passed |
| `uv run pytest tests/integration -q` | Passed | 2 passed, 2 warnings |
| `uv run pytest tests -q` | Failed | 279 passed, 10 failed, 3 warnings; failures listed below |
| `uv run python scripts/export_backend_openapi.py` | Passed | Exported `docs/api/openapi-v1.json` |
| `uv run python -m py_compile ...` | Passed | New Phase 7 routes/services/schemas/adapters/pages compile |
| Backend app import check | Passed | Printed `Evonith BF Backend API` |
| Frontend adapter import check | Passed | Phase 7 adapters import |
| Boundary `rg` checks | Passed | No Streamlit imports in backend app; no backend app imports in frontend adapters |

## Backend API Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| API-MB-001 | Material Balance config | Wrapped config response | Passed | Passed | |
| API-MB-002 | Material Balance validate | Valid response | Passed | Passed | |
| API-MB-003 | Material Balance run input data | JSON-safe result | Passed | Passed | |
| API-MB-004 | Material Balance artifact | CSV download | Passed | Passed | |
| API-REC-001 | Recommendations config | Wrapped config response | Passed | Passed | |
| API-REC-002 | Recommendations run | Capped items returned | Passed | Passed | |
| API-BMO-001 | Blend context | Context returned | Passed | Passed | |
| API-BMO-002 | Blend optimize | Candidates returned | Passed | Passed | |
| API-BMO-003 | Model registry | Lazy list/load behavior | Passed | Passed | |
| API-SEC-001 | Compute auth required | AUTH_REQUIRED | Passed | Passed | |
| API-SEC-002 | Invalid model name | MODEL_PATH_INVALID | Passed | Passed | |
| API-SEC-003 | No model/runtime paths | Paths hidden | Passed | Passed | |

## Frontend Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| F-MB-001 | get_material_balance_config | Calls `/material-balance/config` | Passed | Passed | |
| F-MB-002 | validate_material_balance | Calls `/material-balance/validate` | Passed | Passed | |
| F-MB-003 | run_material_balance | Calls `/material-balance/run` | Passed | Passed | |
| F-REC-001 | get_recommendations_config | Calls `/recommendations/config` | Passed | Passed | |
| F-REC-002 | run_recommendations | Calls `/recommendations/run` | Passed | Passed | |
| F-BMO-001 | get_blend_optimizer_context | Calls `/blend-optimizer/context` | Passed | Passed | |
| F-BMO-002 | list_blend_optimizer_models | Calls `/blend-optimizer/models` | Passed | Passed | |
| F-BMO-003 | predict_blend_outputs | Calls `/blend-optimizer/predict` | Passed | Passed | |
| F-BMO-004 | optimize_blend | Calls `/blend-optimizer/optimize` | Passed | Passed | |
| F-FLAG-001 | Phase 7 flags false | Direct mode selected | Passed | Passed | |
| F-FLAG-002 | Phase 7 flags true | API mode selected | Passed | Passed | |

## Integration Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| INT-COMP-001 | Material Balance run | JSON-safe result | Passed | Passed | |
| INT-COMP-002 | Recommendations run | Items returned | Passed | Passed | |
| INT-COMP-003 | Blend Optimizer context | Context returned | Passed | Passed | |
| INT-COMP-004 | Blend Optimizer optimize | Candidates returned | Passed | Passed | |
| INT-COMP-005 | Feedback regression | `/api/v1/feedback/config` works | Passed | Passed | |

## Regression Test Results

| Phase | Regression Area | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| Phase 1 | Runtime paths | Runtime tests pass; compute files under runtime | Covered by broad suite where collected | Partial | Full suite has unrelated failures |
| Phase 2 | API foundation | Health/readiness/errors/CORS pass | Backend suite passed | Passed | |
| Phase 3 | ApiClient/status | Client and backend status tests pass | Frontend suite passed | Passed | |
| Phase 4 | Data/datasets | Data and dataset tests pass | Some static dataset tests fail in broad suite | Failed | Existing static dataset follow-up |
| Phase 5 | Auth/admin | Auth and admin tests pass | Backend suite passed | Passed | |
| Phase 6 | Feedback | Feedback tests pass | Backend/frontend/integration suites passed | Passed | |
| Phase 7 | Domain compute | Domain compute tests pass | Focused and backend suites passed | Passed | |
| Full suite | `uv run pytest tests -q` | All pass | 279 passed, 10 failed, 3 warnings | Failed | Failures listed below |

## Performance and Edge Observations

| Area | Observation | Result | Notes |
|---|---|---|---|
| Backend app import | No eager model loading | Passed | App import check passed |
| Optional ML deps | Missing optional deps do not break startup | Passed | Backend starts and tests pass without loading all optional models |
| Model registry | Lists models without loading | Passed | Unit test verifies unloaded status before prediction |
| Model cache | Cache max size enforced | Covered | Config and registry path tested with cache max set |
| Material Balance | Test run duration | Passed | Focused tests complete normally |
| Recommendations | Test run duration | Passed | Focused tests complete normally |
| Blend Optimizer | Test run duration and timeout behavior | Passed | Bounded candidate API tests pass |
| Large outputs | Capped/exported behavior | Passed | CSV artifact tests pass |
| Pi/Jetson readiness | Memory/CPU safeguards documented | Passed | See Phase 7 domain compute API doc |

## Failed Tests and Fixes

| Test | Failure Summary | Root Cause | Fix Applied | Rerun Result |
|---|---|---|---|---|
| `tests/test_bmo_v4_integration.py::test_blend_fuel_prediction_helper_attaches_model_cost` | Expected fuel-rate estimate differs from current output | Existing BMO fuel-rate formula/defaults mismatch | Not changed in Phase 7 | Still failing in broad suite |
| `tests/test_bmo_v4_integration.py::test_estimated_fuel_rates_use_predicted_cost_and_recent_rates` | Expected total coke rate `438.0`, current output `380.357...` | Existing BMO fuel-rate formula/defaults mismatch | Not changed in Phase 7 | Still failing in broad suite |
| `tests/test_bmo_v4_integration.py::test_estimated_fuel_rates_use_requested_nut_coke_fallback_only` | Expected coke rate `368.0`, current output `310.357...` | Existing BMO fuel-rate formula/defaults mismatch | Not changed in Phase 7 | Still failing in broad suite |
| `tests/test_dataset_service_distribution_sqlalchemy.py::test_fetch_distribution_data_returns_expected_windowed_rows` | Expected burden day row differs after timezone normalization | Existing distribution day-anchor behavior | Not changed in Phase 7 | Still failing in broad suite |
| `tests/test_static_dataset_cache.py` six tests | Static dataset update/cache tests return empty or reject monkeypatched config | Existing cutoff/cache/test fixture behavior | Not changed in Phase 7 | Still failing in broad suite |

## Skipped Tests

| Test | Reason Skipped | Follow-up |
|---|---|---|
| None planned | Not applicable | Not applicable |

## Security Verification

| Check | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|
| No internal model paths | API responses hide paths | No paths exposed in model status test | Passed | |
| No internal runtime paths | API responses hide paths | Artifact responses expose IDs/URLs, not filesystem paths | Passed | |
| Model path traversal blocked | Invalid model ID rejected | `MODEL_PATH_INVALID` | Passed | |
| Artifact path traversal blocked | Invalid artifact ID rejected | Structured route errors | Passed | |
| Tokens not logged | No bearer tokens in logs | No token logging added | Passed | |
| Raw full datasets not logged | Logs contain only metadata | No raw dataset logging added | Passed | |
| Compute auth enforced | 401/403 where expected | `AUTH_REQUIRED` test passed | Passed | |

## Final Readiness Status

Overall Phase 7 status: Implemented; scoped backend/frontend/integration tests pass.

All tests passing: No. `uv run pytest tests -q` reports 279 passed and 10 failed in existing BMO/static dataset areas.

Ready for Phase 8: Yes for the Phase 7 compute API surface, with the full-suite failures tracked separately.

Blocking issues:
No Phase 7 compute API blocker. Full-suite blockers remain in BMO fuel-rate, burden-distribution, and static-dataset tests.

Summary:
Phase 7 implementation has been added and verified through focused backend,
frontend, integration, boundary, import, and OpenAPI checks. AI Copilot,
FurnaceMind, LLM chat, and Qdrant were not migrated.
