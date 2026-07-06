# Phase 5 Test Execution Report

## Test Environment

| Item | Value |
| --- | --- |
| Branch | migration/backend-frontend-split |
| Phase | Phase 5 - Auth and Admin API Migration |
| Test date | 2026-07-03 |
| OS/Environment | Windows local workspace |
| Runtime dir | `./runtime` |
| Backend API base URL | `http://localhost:8080/api/v1` |

## Commands Run

| Command | Result | Notes |
| --- | --- | --- |
| `uv run --with pytest pytest furnace-data-service\tests\test_auth_password_token.py furnace-data-service\tests\test_api_v1_auth_admin.py -q` | PASS | 12 passed, 1 warning |
| `uv run --with pytest pytest tests\frontend\test_auth_admin_api.py tests\frontend\test_api_client.py tests\frontend\test_frontend_settings.py tests\frontend\test_import_boundaries.py tests\test_session_auth.py -q` | PASS | 23 passed |
| `uv run --with pytest pytest furnace-data-service\tests -q` | PASS | 46 passed, 2 warnings |
| `uv run --with pytest pytest tests\frontend -q` | PASS | 38 passed |
| `uv run --with pytest pytest tests\integration -q` | NOT RUN | `tests\integration` does not exist in this checkout |
| `uv run --with pytest pytest tests -q` | FAIL | Collection failed on pre-existing Streamlit/path issues; see failure summary |
| `uv run python scripts\export_backend_openapi.py` | PASS | Exported `docs/api/openapi-v1.json` |
| `uv run python -c "from furnace_data.runtime_paths import get_runtime_dir, ensure_runtime_dirs; print(get_runtime_dir()); ensure_runtime_dirs()"` | PASS | Printed resolved runtime path |
| `uv run python -c "from src.services.auth_api import login; print('auth api import ok')"` | PASS | Frontend auth adapter import OK |
| `uv run python -c "from src.services.admin_api import list_users; print('admin api import ok')"` | PASS | Frontend admin adapter import OK |
| `uv run python -c "import sys; sys.path.insert(0, 'furnace-data-service'); from app.main import app; print(app.title)"` | PASS | Backend app imports and prints title |
| `rg -n "import streamlit|from streamlit" furnace-data-service\app` | PASS | No matches |
| `rg -n "from app|import app|furnace-data-service|from furnace-data-service|influxdb|psycopg" src\services\auth_api.py src\services\admin_api.py` | PASS | No matches |
| `uv run python scripts\bootstrap_admin.py` | EXPECTED REFUSAL | Exit code 1; refused to run because `EVONITH_AUTH_BOOTSTRAP_ADMIN_ENABLED` is false |

## Backend Auth Test Cases

| ID | Test Case | Expected Result | Actual Result | Status |
| --- | --- | --- | --- | --- |
| AUTH-001 | Bcrypt password hash | Hash is not plaintext and verifies | Passed | PASS |
| AUTH-002 | Legacy SHA-256 verification | Valid legacy hash accepted and marked for rehash | Passed | PASS |
| AUTH-003 | Password policy | Too-short new password rejected | Passed | PASS |
| AUTH-004 | Access token roundtrip | HS256 token verifies and returns claims | Passed | PASS |
| AUTH-005 | Expired token | `TOKEN_EXPIRED` raised | Passed | PASS |
| AUTH-006 | Production missing secret | Token creation fails safely | Passed | PASS |
| AUTH-007 | Login endpoint | Returns token and profile | Passed | PASS |
| AUTH-008 | Invalid login | Structured `INVALID_CREDENTIALS` | Passed | PASS |
| AUTH-009 | Current user endpoint | Token returns profile | Passed | PASS |
| AUTH-010 | Change password wrong current password | Structured `INVALID_CREDENTIALS` | Passed | PASS |
| AUTH-011 | Legacy login upgrade | SHA-256 hash replaced with bcrypt after successful backend login | Passed | PASS |

## Backend Admin Test Cases

| ID | Test Case | Expected Result | Actual Result | Status |
| --- | --- | --- | --- | --- |
| ADMIN-001 | Missing admin token | Structured `AUTH_REQUIRED` | Passed | PASS |
| ADMIN-002 | Non-admin token | Structured `FORBIDDEN` | Passed | PASS |
| ADMIN-003 | Admin list users | 200 with paginated list | Passed | PASS |
| ADMIN-004 | Admin create user | 200 with created user profile | Passed | PASS |
| ADMIN-005 | Admin deactivate user | 200 with `is_active=false` | Passed | PASS |
| ADMIN-006 | Roles and permissions | OpenAPI exposes routes; service returns catalog | Covered by route and OpenAPI checks | PASS |

## Frontend Test Cases

| ID | Test Case | Expected Result | Actual Result | Status |
| --- | --- | --- | --- | --- |
| FRONT-AUTH-001 | `auth_api.login` | Calls `/auth/login` with credentials | Passed | PASS |
| FRONT-AUTH-002 | `auth_api.get_me` | Sends bearer token | Passed | PASS |
| FRONT-ADMIN-001 | `admin_api.list_users` | Calls `/admin/users` with token and params | Passed | PASS |
| FRONT-ADMIN-002 | `admin_api.create_user` | Posts user payload with bearer token | Passed | PASS |
| FRONT-SETTINGS-001 | Auth/admin flags | Resolve from `USE_BACKEND_API_AUTH` and `USE_BACKEND_API_ADMIN` | Passed | PASS |
| FRONT-SESSION-001 | Token session state | Stores backend token and expiry when provided | Passed | PASS |
| FRONT-CLIENT-001 | API client headers | Convenience methods forward custom headers | Passed | PASS |

## Boundary And Regression Checks

| Check | Result | Notes |
| --- | --- | --- |
| Backend app imports Streamlit | PASS | No matches under `furnace-data-service/app` |
| Frontend auth/admin adapters import backend internals | PASS | No matches for backend/internal patterns |
| OpenAPI export | PASS | Export completed after adding auth/admin routes |
| Direct-mode auth/admin removal | PASS | Direct-mode login and registration code paths remain |
| Legacy backend routes | PASS | Existing backend suite still passes |

## Full Repo Test Failure Summary

`uv run --with pytest pytest tests -q` failed during collection before running
tests. The failures match pre-existing repository collection issues:

- Streamlit stub in several BMO/static dataset tests lacks `cache_data`.
- Several tests import `src.*`, but `src` is not importable as a package in that
  collection mode.
- Shift report tests cannot import `LOCAL_TIMEZONE` from the active
  `utils.shift_windows` module.

These failures are not caused by the Phase 5 auth/admin changes. Focused
backend and frontend Phase 5 suites passed.

## Manual Smoke Coverage

| Scenario | Result | Notes |
| --- | --- | --- |
| Long-running Uvicorn server | SKIPPED | Import/TestClient checks covered startup and routes |
| Interactive Streamlit direct mode | SKIPPED | Direct branches preserved but not launched interactively |
| Interactive Streamlit API mode | SKIPPED | Frontend adapters and guarded UI code covered by tests/imports |
| Bootstrap script with real credentials | SKIPPED | Disabled-mode safety check run; no credentials were provided |

## Final Readiness Status

Overall Phase 5 status: PASS with documented repo-wide collection caveat.

Blocking issues for Phase 5:

None found in the focused Phase 5 backend/frontend suites.

Follow-up:

- Repair repo-wide test collection separately.
- Resolve Alembic/ORM identity schema mismatch before adding auth migrations.
- Consider richer admin UI controls after Phase 5.
