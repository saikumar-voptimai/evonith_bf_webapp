# Phase 5 Testing Guide

## Scope

Use this guide to verify Phase 5 auth/admin API behavior. This phase covers
backend login/current-user/password-change, backend admin user management,
frontend API adapters, and Streamlit feature-flag integration only.

## Environment

Local development:

```bash
EVONITH_RUNTIME_DIR=./runtime
EVONITH_AUTH_SECRET_KEY=dev-secret
BACKEND_API_BASE_URL=http://localhost:8080/api/v1
```

Direct mode:

```bash
USE_BACKEND_API_AUTH=false
USE_BACKEND_API_ADMIN=false
```

API mode:

```bash
USE_BACKEND_API_AUTH=true
USE_BACKEND_API_ADMIN=true
```

Edge production must set:

```bash
EVONITH_RUNTIME_DIR=/var/lib/evonith-bf
EVONITH_BACKEND_ENV=edge
EVONITH_AUTH_SECRET_KEY=<strong-random-secret>
```

## Backend Startup

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime \
EVONITH_AUTH_SECRET_KEY=dev-secret \
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Expected:

- Backend starts without requiring Streamlit.
- Startup logs include API prefix, runtime directory, and auth enabled state.
- `/api/v1/health` returns 200.

## Bootstrap Admin

Run only when a backend admin user does not already exist:

```bash
EVONITH_AUTH_BOOTSTRAP_ADMIN_ENABLED=true \
EVONITH_BOOTSTRAP_ADMIN_USERNAME=admin_user \
EVONITH_BOOTSTRAP_ADMIN_PASSWORD='set-a-strong-password' \
python scripts/bootstrap_admin.py
```

Expected:

- Script refuses to run when the enable flag is false.
- Script refuses to run without username/password.
- Script skips an existing user without deleting or changing it.

## Auth API Smoke

Login:

```bash
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin_user","password":"set-a-strong-password"}'
```

Expected:

- 200 response with `access_token`, `token_type=bearer`, expiry, and user profile.
- Invalid credentials return structured `INVALID_CREDENTIALS`.
- Inactive users return structured `USER_INACTIVE`.

Current user:

```bash
curl http://localhost:8080/api/v1/auth/me \
  -H "Authorization: Bearer <access_token>"
```

Expected:

- 200 response with user profile.
- Missing token returns `AUTH_REQUIRED`.
- Invalid token returns `INVALID_TOKEN`.
- Expired token returns `TOKEN_EXPIRED`.

Change password:

```bash
curl -X POST http://localhost:8080/api/v1/auth/change-password \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"current_password":"old-password","new_password":"new-password-123"}'
```

Expected:

- 200 response with `changed=true`.
- Wrong current password returns `INVALID_CREDENTIALS`.
- Too-short new password returns `PASSWORD_POLICY_FAILED`.

## Admin API Smoke

List users:

```bash
curl http://localhost:8080/api/v1/admin/users \
  -H "Authorization: Bearer <admin_access_token>"
```

Create user:

```bash
curl -X POST http://localhost:8080/api/v1/admin/users \
  -H "Authorization: Bearer <admin_access_token>" \
  -H "Content-Type: application/json" \
  -d '{"username":"operator_1","password":"operator-pass-123","role":"user"}'
```

Deactivate user:

```bash
curl -X POST http://localhost:8080/api/v1/admin/users/<user_id>/deactivate \
  -H "Authorization: Bearer <admin_access_token>"
```

Expected:

- Admin token can list/create/update users.
- Non-admin token returns `FORBIDDEN`.
- Missing token returns `AUTH_REQUIRED`.
- Duplicate username returns `ADMIN_USER_EXISTS`.

## Streamlit Direct Mode

```bash
USE_BACKEND_API_AUTH=false \
USE_BACKEND_API_ADMIN=false \
streamlit run src/app.py
```

Expected:

- Existing login behavior still works.
- Existing user registration path still uses direct DB mode.
- No backend token is required.

## Streamlit API Mode

```bash
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 \
USE_BACKEND_API_AUTH=true \
USE_BACKEND_API_ADMIN=true \
streamlit run src/app.py
```

Expected:

- Login calls `/api/v1/auth/login`.
- Session stores `auth_access_token`.
- Admin user-management page lists backend users and creates users through
  `/api/v1/admin/users`.
- If backend is unavailable, the page shows a frontend-safe error.

## Automated Checks

```bash
pytest furnace-data-service/tests -q
pytest tests/frontend -q
pytest tests/integration -q
pytest tests -q
python scripts/export_backend_openapi.py
```

Import checks:

```bash
cd furnace-data-service
python -c "from app.main import app; print(app.title)"
cd ..
python -c "from src.services.auth_api import login; print('auth api import ok')"
python -c "from src.services.admin_api import list_users; print('admin api import ok')"
```

Boundary checks:

```bash
rg -n "import streamlit|from streamlit" furnace-data-service/app
rg -n "from app|import app|furnace-data-service|influxdb|psycopg" src/services/auth_api.py src/services/admin_api.py
```

Expected:

- Backend and frontend focused suites pass.
- OpenAPI export includes auth/admin paths.
- Boundary greps return no matches.
