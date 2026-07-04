# Phase 13 Test Execution Report

## Test Environment

| Item | Value |
|---|---|
| Branch | `migration/backend-frontend-split` |
| Runtime directory | `./runtime` unless overridden by tests |
| Auth secret | `dev-only-secret-change-me` for tests only |
| Package runner | `uv run` for dependency-backed checks; `python` for pure deployment scripts |
| Host | Windows, PowerShell |
| Date | 2026-07-04 |

## Implementation Summary Under Test

Phase 13 deployment/cutover readiness assets are under test: environment
templates, dry-run startup scripts, runtime bootstrap, deployment validation,
smoke testing, API cutover validation, backup/restore, release readiness,
systemd/reverse proxy examples, and Phase 12 structure regressions.

## Commands Executed

| Command | Result | Notes |
|---|---|---|
| `uv run pytest tests/deployment -q` | Pass: 9 passed, 1 skipped | Symlink escape test skipped when Windows symlink creation is unavailable |
| `uv run pytest furnace-data-service/tests -q` | Pass: 102 passed, 2 warnings | Backend/API regression suite |
| `uv run pytest tests/frontend -q` | Pass: 78 passed | Frontend API/client tests |
| `uv run pytest tests/integration -q` | Pass: 7 passed, 4 warnings | Cross-phase API-mode flows |
| `uv run pytest tests/dependency -q` | Pass: 8 passed | Dependency profile checks |
| `uv run pytest tests/structure -q` | Pass: 13 passed, 1 warning | Phase 12 structure checks |
| `uv run pytest tests -q` | Pass: 339 passed, 1 skipped, 5 warnings | Full regression run on current tree |
| `uv run python scripts/export_backend_openapi.py` | Pass | Re-exported `docs/api/openapi-v1.json` |
| `python scripts/check_repository_structure.py` | Pass with warning | Warned that legacy runtime-like files still exist under ignored old paths |
| `python scripts/check_import_boundaries.py` | Pass | No frontend/backend boundary violations |
| `python scripts/check_dependency_profiles.py` | Pass | Dependency groups present |
| `uv run python scripts/check_backend_minimal_startup.py` | Pass | Backend app imports and `/api/v1/health` works under test client |
| `python scripts/check_frontend_api_imports.py` | Pass | Frontend adapters import |
| `python scripts/bootstrap_runtime.py --dry-run` | Pass with warning | No writes; warns runtime subdirs would be created |
| `python scripts/validate_deployment.py --profile local --offline` | Pass with warnings | Local/offline warnings for missing runtime, missing API URL, and plain-python backend deps |
| `python scripts/validate_api_cutover.py --allow-partial --json` | Pass with warning status | Cutover flags intentionally not all enabled yet; adapters and OpenAPI paths pass |
| `python scripts/backup_runtime.py --dry-run` | Pass | No archive written |
| `python scripts/verify_release_readiness.py --allow-dirty --skip-tests` | Pass with dirty-tree warning | Expected during implementation |
| `uv run python -c "from apps.backend_api.app.main import app; print(app.title)"` | Pass | Canonical backend import |
| `uv run python -c "import sys; sys.path.insert(0, 'furnace-data-service'); from app.main import app; print(app.title)"` | Pass | Legacy backend shim import via root project env |
| `python -c "from apps.frontend_streamlit.services.status_api import get_status; print('new frontend import ok')"` | Pass | Canonical frontend adapter import |
| `python -c "from src.services.status_api import get_status; print('old frontend shim import ok')"` | Pass | Legacy frontend shim import |
| `bash -lc "DRY_RUN=1 scripts/edge_start_backend.sh"` | Host blocked | Windows WSL/bash instance creation denied; script contract covered by tests |
| `bash -lc "DRY_RUN=1 scripts/edge_start_frontend.sh"` | Host blocked | Windows WSL/bash instance creation denied; script contract covered by tests |

## Deployment Asset Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| D-001 | Env templates | Placeholders only | `infra/env/*.example` use placeholder secrets and documented roles | Pass | Covered by deployment tests and secret scan |
| D-002 | Edge scripts dry-run | No services started | Scripts include `DRY_RUN` command printing; shell execution blocked by host WSL policy | Pass with host limitation | Static contract covered in tests |
| D-003 | Systemd examples | Canonical scripts | Backend/frontend unit examples call edge scripts and include `PrivateTmp=true` | Pass | infra/systemd |
| D-004 | Reverse proxy examples | No secrets | Nginx/Caddy examples use placeholders and local upstreams | Pass | nginx/caddy |
| D-005 | Deployment docs | Exist | Required guides and checklist exist | Pass | docs/deployment |

## Script Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| S-001 | Runtime bootstrap | Safe dry-run/create/check | Dry-run exits 0 without writes; create builds expected dirs in tmp runtime | Pass | bootstrap_runtime.py |
| S-002 | Deployment validation | Local offline pass | Exits 0 with expected local/offline warnings | Pass | validate_deployment.py |
| S-003 | Production placeholder detection | Fails | Production profile rejects placeholder `EVONITH_AUTH_SECRET_KEY` | Pass | Deployment test |
| S-004 | Smoke test | HTTP checks work | Fake backend returns pass for public endpoints, request ID, structured error | Pass | smoke_test_deployment.py |
| S-005 | API cutover validation | Allow partial pass | Exits 0 with warning because flags are intentionally off | Pass | validate_api_cutover.py |
| S-006 | Backup runtime | Dry-run and archive | Dry-run passes; test archive is written from isolated runtime | Pass | backup_runtime.py |
| S-007 | Restore runtime | Dry-run and isolated apply | Dry-run writes nothing; apply restores isolated test files | Pass | restore_runtime.py |
| S-008 | Release readiness | Gate passes with allow-dirty/skip-tests | Exits 0 with dirty-tree warning | Pass | verify_release_readiness.py |

## Phase 12 Repository-Structure Regression Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| P12-001 | Canonical backend path | Imports | `apps.backend_api.app.main` imports under `uv run` | Pass | App title printed |
| P12-002 | Canonical frontend path | Exists | `apps/frontend_streamlit/app.py` remains canonical | Pass | Structure check |
| P12-003 | Old backend shim | Imports | `furnace-data-service/app/main.py` imports via root project env | Pass | Service-local `uv` was host-cache blocked |
| P12-004 | Old frontend shim | Exists/imports | `src.services.status_api` shim imports | Pass | Direct-mode compatibility preserved |
| P12-005 | Structure check | Passes | Pass with warning for old ignored runtime-like files | Pass | Non-destructive migration policy |
| P12-006 | Import boundaries | Passes | No forbidden imports | Pass | check_import_boundaries.py |
| P12-007 | Dependency profiles | Passes | Groups present and validated | Pass | check_dependency_profiles.py |
| P12-008 | Backend minimal startup | Passes | Backend app and health route pass | Pass | check_backend_minimal_startup.py |
| P12-009 | Frontend API imports | Passes | All adapters import | Pass | check_frontend_api_imports.py |

## API Regression Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| API-001 | `/api/v1/health` | 200 | Covered by backend and minimal-startup tests | Pass | Backend tests |
| API-002 | `/api/v1/readiness` | Expected readiness | Covered by backend tests | Pass | Backend tests |
| API-003 | `/api/v1/status` | Works | Covered by backend tests and OpenAPI cutover validation | Pass | Backend tests |
| API-004 | `/api/v1/auth/me` | Works with token | Covered by backend auth tests | Pass | Backend tests |
| API-005 | `/api/v1/data/sources` | Works | Covered by backend data tests | Pass | Backend tests |
| API-006 | `/api/v1/datasets` | Works | Covered by backend dataset tests | Pass | Backend tests |
| API-007 | `/api/v1/feedback/config` | Works | Covered by backend feedback tests | Pass | Backend tests |
| API-008 | `/api/v1/material-balance/config` | Works | Covered by backend compute tests | Pass | Backend tests |
| API-009 | `/api/v1/recommendations/config` | Works | Covered by backend compute tests | Pass | Backend tests |
| API-010 | `/api/v1/blend-optimizer/context` | Works | Covered by backend compute tests | Pass | Backend tests |
| API-011 | `/api/v1/copilot/config` | Works | Covered by backend copilot tests | Pass | Backend tests |
| API-012 | `/api/v1/furnacemind/config` | Works | Covered by backend FurnaceMind tests | Pass | Backend tests |
| API-013 | `/api/v1/metrics` | Works for admin | Covered by ops tests | Pass | Backend tests |
| API-014 | Request ID | Header present | Covered by request ID and smoke tests | Pass | Middleware tests |
| API-015 | Structured errors | Envelope unchanged | Covered by error and smoke tests | Pass | Error tests |

## Frontend Regression Test Cases

| ID | Test Case | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| F-001 | ApiClient | Tests pass | `tests/frontend` passed | Pass | 78 tests |
| F-002 | All frontend API adapters import | Imports pass | `check_frontend_api_imports.py` passed | Pass | Import gate |
| F-003 | Backend status badge | Works | Covered by frontend tests | Pass | tests/frontend |
| F-004 | Advanced status panel | Works if enabled | Covered by frontend tests | Pass | tests/frontend |
| F-005 | Full API-mode flags parse | Parsed correctly | Cutover validator parses and reports flags | Pass | Warning expected while flags are off |
| F-006 | Direct-mode fallback flags parse | Parsed correctly | Direct fallback flag remains available | Pass | validate_api_cutover.py |
| F-007 | Backend unavailable behavior | Clean error | Covered by frontend tests | Pass | tests/frontend |
| F-008 | Frontend avoids backend internals | No forbidden imports | Boundary checks pass | Pass | check_import_boundaries.py |

## Regression Test Results

| Phase | Regression Area | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|---|
| Phase 1 | Runtime paths | Tests pass | Full suite passed | Pass | Runtime path tests included |
| Phase 2 | API foundation | Tests pass | Backend tests passed | Pass | 102 backend tests |
| Phase 3 | API client/status | Tests pass | Frontend tests passed | Pass | 78 frontend tests |
| Phase 4 | Data/datasets | Tests pass | Backend/integration tests passed | Pass | Dataset route tests included |
| Phase 5 | Auth/admin | Tests pass | Backend/frontend tests passed | Pass | Auth/admin route tests included |
| Phase 6 | Feedback | Tests pass | Backend/integration tests passed | Pass | Feedback route tests included |
| Phase 7 | Domain compute | Tests pass | Backend/integration tests passed | Pass | Domain compute flow included |
| Phase 8 | AI Copilot | Tests pass | Backend/integration tests passed | Pass | Copilot flow included |
| Phase 9 | FurnaceMind | Tests pass | Backend/integration tests passed | Pass | FurnaceMind flow included |
| Phase 10 | Ops hardening | Tests pass | Backend/integration tests passed | Pass | Ops services and metrics included |
| Phase 11 | Dependency/runtime hardening | Tests pass | Dependency tests passed | Pass | 8 dependency tests |
| Phase 12 | Repository restructure | Tests pass | Structure tests passed | Pass | 13 structure tests |
| Phase 13 | Deployment/cutover readiness | Tests pass | Deployment tests passed | Pass | 9 passed, 1 platform skip |
| Full suite | `pytest tests -q` | All pass | 339 passed, 1 skipped, 5 warnings | Pass | Current tree |

## Performance and Edge Observations

| Area | Observation | Result | Notes |
|---|---|---|---|
| Deployment validation | Local/offline run is fast and non-blocking | Pass | Expected warnings do not fail local mode |
| Smoke test | Fake backend smoke completes in deployment pytest | Pass | No live deployment required |
| Runtime bootstrap | Dry-run does not write; create works in temp runtime | Pass | Default runtime remains configurable |
| Backup archive | Isolated archive writes and restores | Pass | No source files touched |
| Restore dry-run | Default restore mode writes nothing | Pass | `--apply` required |
| Backend import | Canonical backend imports under `uv run` | Pass | Plain `python` lacks backend deps on this host |
| Edge dry-run scripts | Static contract validated; shell run blocked by Windows WSL policy | Pass with host limitation | Run on Linux/edge before cutover |
| Disk validation | Free-space checks pass on host | Pass | 179 GB free reported during validation |
| Thread defaults | Present in scripts/env | Pass | Edge scripts set BLAS/tokenizer thread defaults |
| Full test suite | 339 passed, 1 skipped in 115.96s | Pass | Warnings only |

## Failed Tests and Fixes

| Test | Failure Summary | Root Cause | Fix Applied | Rerun Result |
|---|---|---|---|---|
| `python scripts/bootstrap_runtime.py --dry-run` | Exited non-zero before final run | Dry-run reused the writable probe/check path and treated missing dirs as fatal | Made dry-run advisory and non-mutating | Pass with warning |
| `python scripts/validate_deployment.py --profile local --offline` | Secret scan failed before final run | Scanner flagged Python variables and non-secret env names such as `TOKENIZERS_PARALLELISM` | Narrowed scanner to credential-like env/config assignments | Pass |
| `python scripts/validate_api_cutover.py --allow-partial --json` | Adapter imports failed before final run | Script execution path did not include repo root | Added repo-root insertion in shared deployment helpers | Pass |
| `tests/deployment::test_infra_templates_use_placeholders_and_safe_defaults` | Initial assertion expected `evonith.example.com` | Test was stricter than placeholder-based template contract | Updated assertion to validate `<your-domain-or-ip>` placeholder | Pass |

## Skipped Tests

| Test | Reason Skipped | Follow-up |
|---|---|---|
| Deployment symlink escape test | Skipped if Windows symlink creation is unavailable | Runs on platforms/users that can create symlinks |
| Edge shell script live dry-run | Windows WSL `bash` instance creation denied with `E_ACCESSDENIED` | Run `DRY_RUN=1 scripts/edge_start_backend.sh` and frontend equivalent on Linux/edge host |
| Service-local legacy backend import using `cd furnace-data-service; uv run --project .. ...` | `uv` cache initialization denied by Windows cache permissions | Root project env import of the same shim passed |
| Live deployed backend/frontend smoke | No live services started in Phase 13 validation | Run before real cutover with `scripts/smoke_test_deployment.py` |

## Security Verification

| Check | Expected Result | Actual Result | Status | Notes |
|---|---|---|---|---|
| No secrets in env examples | Placeholders only | Templates use placeholders | Pass | infra/env |
| No secrets in scripts | No real secrets | Secret scanner returned no findings | Pass | scripts |
| No secrets in infra templates | No real secrets | Secret scanner returned no findings | Pass | infra |
| No tokens printed by smoke tests | Tokens hidden | Smoke script does not print provided password/token | Pass | Smoke script test path |
| Backup prevents symlink escape | Escape blocked or skipped | Script warns/skips escaping symlink when symlink test can run | Pass / platform skip | Backup logic implemented |
| Restore prevents path traversal | Traversal blocked | Malicious tar entry rejected | Pass | Deployment test |
| Production placeholder secret detected | Validation fails | Production profile rejects placeholder auth secret | Pass | Deployment test |
| Runtime path safety | Unsafe paths rejected | Repository root runtime rejected | Pass | Deployment test |
| Admin ops protections preserved | Protected endpoints require admin | Backend ops/auth tests passed | Pass | Backend suite |

## Final Readiness Status

Overall Phase 13 status: Pass with documented host limitations.

All automated tests run for Phase 13 passed. One deployment test is a platform
skip when Windows symlink creation is unavailable. Two Linux-oriented shell
dry-runs were host-blocked by Windows WSL access denial and should be run on the
target Linux/edge environment before real cutover.

Ready for Phase 14: Yes, from repository/test readiness perspective.

Blocking issues:
None found in repository validation.

Summary:
Phase 13 deployment readiness assets are present, tested, documented, and
non-destructive. Direct-mode fallback, compatibility shims, and legacy routes
remain intact.
