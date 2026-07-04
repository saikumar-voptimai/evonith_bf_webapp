# Phase 11 Test Execution Report

## Test Environment

| Item | Value |
|---|---|
| Branch | migration/backend-frontend-split |
| Phase | Phase 11 - Dependency and Edge Runtime Hardening |
| Python version | 3.12.12 |
| OS/Environment | Windows-11-10.0.26200-SP0 |
| Package manager | uv |
| EVONITH_RUNTIME_DIR | ./runtime and pytest tmp runtime directories |
| Runtime profile | edge for Phase 11 checks, local for default regression checks |
| Edge mode | true for Phase 11 checks |
| Backend profile checked | backend-base |
| Frontend profile checked | frontend |
| Test date | 2026-07-04 |

## Implementation Summary Under Test

| Area | Summary |
|---|---|
| Dependency groups | Added backend-base, backend-data, backend-ml, backend-ai, backend-vector, backend-documents, frontend, dev, edge |
| Backend-base profile | FastAPI startup profile excluding Streamlit, provider SDKs, Qdrant, torch, sentence-transformers, and heavy ML extras |
| Frontend profile | Streamlit/UI/API-adapter profile excluding backend internals and heavy backend integrations |
| Optional AI/ML/vector/document profiles | Separate opt-in profiles for provider SDKs, model/compute, vector/memory, and document extraction |
| Edge profile | Conservative backend profile with one worker and AI/vector/local LLM disabled by default |
| Lazy-import hardening | Existing lazy imports wrapped by optional dependency guard |
| Optional dependency guard | `optional_dependency_service` checks availability with `find_spec` and raises structured errors |
| Import-boundary checks | `scripts/check_import_boundaries.py` and frontend import script |
| Backend minimal startup check | `scripts/check_backend_minimal_startup.py` |
| Frontend API import check | `scripts/check_frontend_api_imports.py` |
| Edge startup scripts | `scripts/edge_start_backend.sh` and `scripts/edge_start_frontend.sh` |
| Status/dependency endpoint updates | Added `/status/config` and enriched `/status/dependencies` |
| Phase 10 regression scope | status, dependencies, metrics, jobs, cleanup/audit through existing and Phase 11 tests |
| Full regression scope | backend, frontend, integration, dependency, and root test suites |

## Commands Executed

| Command | Result | Duration | Notes |
|---|---|---:|---|
| pytest furnace-data-service/tests -q | PASS | 66.61s | 102 passed, 2 warnings |
| pytest tests/frontend -q | PASS | 0.47s | 78 passed |
| pytest tests/integration -q | PASS | 23.13s | 7 passed, 4 warnings |
| pytest tests/dependency -q | PASS | 9.18s | 8 passed |
| pytest tests -q | PASS | 99.93s | 317 passed, 5 warnings |
| python scripts/export_backend_openapi.py | PASS | 5.30s | Exported `docs/api/openapi-v1.json` |
| python scripts/check_import_boundaries.py | PASS | 0.62s | Boundary script passed |
| python scripts/check_dependency_profiles.py | PASS | 0.42s | Dependency profile script passed |
| python scripts/check_backend_minimal_startup.py | PASS | 4.66s | Backend import/startup check passed |
| python scripts/check_frontend_api_imports.py | PASS | 0.64s | Frontend API import check passed |
| python -c "from app.main import app; print(app.title)" from furnace-data-service | PASS | 6.63s | Printed `Evonith BF Backend API` |

## Dependency Profile Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| DEP-001 | backend-base group exists | Group present | Present | PASS | Root pyproject |
| DEP-002 | frontend group exists | Group present | Present | PASS | Root pyproject |
| DEP-003 | backend-ai group exists | Group present | Present | PASS | Root pyproject |
| DEP-004 | backend-ml group exists | Group present | Present | PASS | Root pyproject |
| DEP-005 | backend-vector group exists | Group present | Present | PASS | Root pyproject |
| DEP-006 | backend-documents group exists | Group present | Present | PASS | Root pyproject |
| DEP-007 | dev group exists | Group present | Present | PASS | Root pyproject |
| DEP-008 | edge group exists | Group present | Present | PASS | Root pyproject |
| DEP-009 | Streamlit excluded from backend-base | Not present | Not present | PASS | Script checked |
| DEP-010 | FastAPI/Uvicorn included in backend-base | Present | Present | PASS | Script checked |
| DEP-011 | Qdrant excluded from backend-base | Not present unless documented | Not present | PASS | Script checked |
| DEP-012 | LLM provider SDK excluded from backend-base | Not present unless documented | Not present | PASS | Script checked |
| DEP-013 | Heavy ML excluded from backend-base | Not present unless documented | Not present | PASS | Script checked |
| DEP-014 | Streamlit included in frontend | Present | Present | PASS | Script checked |
| DEP-015 | pytest included in dev/test | Present | Present | PASS | Script checked |

## Backend Lazy Import Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| LZY-001 | Backend import excludes Streamlit | streamlit not imported | Not imported | PASS | startup script |
| LZY-002 | Backend import excludes Qdrant when disabled | qdrant_client not imported | Not imported | PASS | startup script/unit test |
| LZY-003 | Backend import excludes provider SDK when disabled | provider SDK not imported | Not imported | PASS | startup script/unit test |
| LZY-004 | Backend import excludes torch/sentence-transformers when disabled | heavy modules not imported | Not imported | PASS | startup script |
| LZY-005 | Backend import does not load model files | No model load at startup | No model load observed | PASS | model registry remains lazy |
| LZY-006 | Missing optional package | Structured feature-unavailable error | `DEPENDENCY_OPTIONAL_NOT_INSTALLED` | PASS | unit test |
| LZY-007 | Optional dependency guard caches checks | Cache used | Cache used | PASS | unit test |

## Script Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| SCR-001 | check_import_boundaries.py | Exits 0 | Exits 0 | PASS | Current tree |
| SCR-002 | check_dependency_profiles.py | Exits 0 | Exits 0 | PASS | Current metadata |
| SCR-003 | check_backend_minimal_startup.py | Exits 0 | Exits 0 | PASS | Health and OpenAPI generated |
| SCR-004 | check_frontend_api_imports.py | Exits 0 | Exits 0 | PASS | API adapters import |
| SCR-005 | edge_start_backend.sh | Exists and safe | Exists and safe | PASS | Script test |
| SCR-006 | edge_start_frontend.sh | Exists and safe | Exists and safe | PASS | Script test |
| SCR-007 | Edge scripts contain thread limits | Limits present | Present | PASS | Script test |
| SCR-008 | Edge scripts contain no secrets | No secrets | No secrets | PASS | Script test |

## API Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| API-DEP-001 | /status/config includes runtime_profile | Present | Present | PASS | API test |
| API-DEP-002 | /status/config includes edge_mode | Present | Present | PASS | API test |
| API-DEP-003 | /status/config hides secrets | No secrets | No secrets | PASS | API test |
| API-DEP-004 | /status/dependencies includes optional dependency summary | Present | Present | PASS | API test |
| API-DEP-005 | Optional dependency missing | unavailable/unconfigured | Reported in optional summary | PASS | API/unit tests |
| API-DEP-006 | Optional missing does not break health | Health 200 | 200 | PASS | API test |
| API-DEP-007 | Optional missing does not break readiness | Readiness ok/degraded appropriately | 200 | PASS | API test |
| API-DEP-008 | Detailed dependency status protected | 401/403 when required | Protected | PASS | Existing Phase 10 and Phase 11 tests |
| API-DEP-009 | Dependency status hides package paths | No absolute paths | No package paths | PASS | API/unit tests |

## Frontend Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| F-DEP-001 | status_api parses runtime profile | Parsed correctly | Parsed | PASS | Frontend test |
| F-DEP-002 | status_api parses optional dependency status | Parsed correctly | Parsed | PASS | Frontend test |
| F-DEP-003 | Advanced dependency details hidden by default | Hidden | Hidden | PASS | Frontend test |
| F-DEP-004 | Advanced status displays profile summary | Safe summary | Profile caption code added | PASS | Source/unit coverage |
| F-DEP-005 | Backend unavailable handling | Clean error | Existing tests cover | PASS | Regression suite |
| F-BOUND-001 | Frontend API adapters import without backend | Imports pass | Pass | PASS | Script |
| F-BOUND-002 | Frontend API adapters avoid backend internals | No forbidden imports | None found | PASS | Script/frontend test |
| F-BOUND-003 | Frontend API adapters avoid DB/vector/LLM/model packages | No forbidden imports | None found | PASS | Script/frontend test |

## Integration Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| INT-DEP-001 | Backend imports with optional AI/vector disabled | Import succeeds | Succeeds | PASS | Integration test |
| INT-DEP-002 | Health under edge-like env | 200 | 200 | PASS | Integration test |
| INT-DEP-003 | Readiness under edge-like env | Ready/degraded appropriately | 200 | PASS | Integration test |
| INT-DEP-004 | Dependency status shows optional disabled | Summary correct | Correct | PASS | Integration test |
| INT-DEP-005 | OpenAPI export | Success | Export succeeded | PASS | `docs/api/openapi-v1.json` regenerated |
| INT-DEP-006 | Backend minimal startup script | PASS | PASS | PASS | Integration/script test |
| INT-DEP-007 | Frontend API import script | PASS | PASS | PASS | Integration/script test |
| INT-DEP-008 | Import boundary script | PASS | PASS | PASS | Integration/script test |
| INT-DEP-009 | Dependency profile script | PASS | PASS | PASS | Integration/script test |
| INT-REG-001 | Phase 10 status regression | /status works | 200 | PASS | Integration test |
| INT-REG-002 | Phase 10 metrics regression | /metrics works for admin | 200 | PASS | Integration test |
| INT-REG-003 | Phase 9 FurnaceMind regression | /furnacemind/config works | 200 | PASS | Integration test |
| INT-REG-004 | Phase 8 Copilot regression | /copilot/config works | 200 | PASS | Integration test |
| INT-REG-005 | Phase 7 compute regression | Compute config endpoints work | 200 | PASS | Integration/API tests |
| INT-REG-006 | Phase 5 auth regression | Auth works | 200 | PASS | Integration/API tests |

## Regression Test Results

| Phase | Regression Area | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| Phase 1 | Runtime paths | Tests pass | Full suite passed | PASS | Root suite |
| Phase 2 | API foundation | Tests pass | Full suite passed | PASS | Backend/root suites |
| Phase 3 | API client/status | Tests pass | Full suite passed | PASS | Frontend/root suites |
| Phase 4 | Data/datasets | Tests pass | Full suite passed | PASS | Backend/root suites |
| Phase 5 | Auth/admin | Tests pass | Full suite passed | PASS | Backend/root suites |
| Phase 6 | Feedback | Tests pass | Full suite passed | PASS | Backend/root suites |
| Phase 7 | Domain compute | Tests pass | Full suite passed | PASS | Backend/integration/root suites |
| Phase 8 | AI Copilot | Tests pass | Full suite passed | PASS | Backend/integration/root suites |
| Phase 9 | FurnaceMind | Tests pass | Full suite passed | PASS | Backend/integration/root suites |
| Phase 10 | Ops hardening | Tests pass | Full suite passed | PASS | Backend/integration/root suites |
| Phase 11 | Dependency/runtime hardening | Tests pass | Full suite passed | PASS | Dependency/backend/frontend/integration/root suites |
| Full suite | pytest tests -q | All pass | 317 passed, 5 warnings | PASS | Final verification |

## Performance and Edge Observations

| Area | Observation | Result | Notes |
|---|---|---|---|
| Backend app import | Import duration observation | Startup script completed in 20.69s including OpenAPI and health | Includes process startup and logs |
| Backend minimal startup script | Duration observation | 20.69s | No optional services required |
| OpenAPI export | Duration observation | 5.30s | Export succeeded |
| Dependency status | Duration/caching observation | Cached service and `find_spec` checks | No network probes |
| Frontend API import | Duration observation | 22.56s script wall time | Includes process startup |
| Optional heavy imports | Not imported at startup | PASS | Startup script checks |
| Streamlit backend boundary | Streamlit not imported by backend | PASS | Boundary/startup scripts |
| Backend-base size | Dependency count vs full/dev | 14 deps vs legacy full/dev root list | Root dependency list retained for compatibility |
| Edge scripts | Thread limits and runtime defaults present | PASS | Script tests |

## Failed Tests and Fixes

| Test | Failure Summary | Root Cause | Fix Applied | Rerun Result |
|---|---|---|---|---|
| `test_phase11_status_config_and_dependency_status` | Expected unauthenticated compute/Copilot config calls to return 200 | Existing endpoint auth protections were correct | Added admin headers in test | PASS |
| `test_advanced_dependency_details_are_hidden_by_default` | Used `FrontendSettings()` directly and missed env-derived page flags | Existing settings loader populates page flags | Switched test to `load_frontend_settings()` | PASS |

## Skipped Tests

| Test | Reason Skipped | Follow-up |
|---|---|---|
| None | Not applicable | None |

## Security Verification

| Check | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|
| No secrets in scripts | Scripts contain no secrets | No secrets found in script tests | PASS | Placeholder-free scripts |
| No secrets in docs/report | No real secrets | Only documented dev placeholder matched | PASS | `dev-only-secret-change-me` is not a production secret |
| Backend excludes Streamlit | No backend Streamlit import | None found | PASS | Script |
| Frontend API avoids backend internals | No forbidden imports | None found | PASS | Script |
| Optional providers lazy | Provider SDK not imported at startup | Not imported | PASS | Startup script |
| Qdrant lazy | Qdrant not imported at startup when disabled | Not imported | PASS | Startup script |
| Runtime profile status safe | No secret/path leakage | No secrets/package paths in tests | PASS | API tests |
| Production auth secret behavior preserved | Safe handling | Existing auth behavior unchanged | PASS | Auth regression |

## Final Readiness Status

Overall Phase 11 status: PASS

All tests passing: Yes

Ready for Phase 12: Yes

Blocking issues:
None currently known.

Summary:
Phase 11 dependency/runtime hardening is implemented. Backend, frontend,
integration, dependency, root, OpenAPI, startup, import-boundary, and profile
checks all passed.
