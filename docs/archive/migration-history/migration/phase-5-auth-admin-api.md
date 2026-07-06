# Phase 5 Auth and Admin API

Phase 5 moves authentication, current-user identity, RBAC checks, and user
management behind versioned backend APIs while preserving the existing
Streamlit direct-mode login and admin flows.

## Audit

| Area | Current flow | Phase 5 action |
| --- | --- | --- |
| Login UI | `src/ui/login_page.py` validates credentials through `src/domain/auth_service.py` and `src/data/db.py` | Added backend API login behind `USE_BACKEND_API_AUTH=true`; direct mode remains default |
| Session identity | `src/utils/session.py` stores `auth_user`, `auth_user_id`, `role`, and derived permissions in Streamlit session state | Added optional backend token storage while keeping existing session keys |
| Password storage | `src/data/db.py` writes SHA-256 password hashes for direct mode | Backend verifies legacy SHA-256 hashes and upgrades them to bcrypt on successful backend login |
| Default admin seed | `src/data/db.py` still seeds `admin/admin123` for direct mode | Not changed in Phase 5; backend bootstrap is explicit and has no defaults |
| User management UI | `src/ui/user_management.py` creates users through direct DB auth service | Added backend admin API creation/listing behind `USE_BACKEND_API_ADMIN=true`; direct mode remains default |
| Backend API | Existing FastAPI sidecar had health/data/dataset routes only | Added `/api/v1/auth/*` and `/api/v1/admin/*` routes |
| Backend authorization | No shared backend auth dependencies existed | Added bearer-token dependencies and admin-role guard under `app.core.auth_dependencies` |
| Frontend adapters | Data/dataset adapters existed | Added frontend-safe `auth_api.py` and `admin_api.py` adapters |

## Database Safety Audit

| File | Finding | Phase 5 decision |
| --- | --- | --- |
| `furnace_data/relational/models.py` | ORM user model expects schema-qualified `identity.users` with UUID `id`, `password_hash`, `role`, `is_active`, timestamps, and `identity.user_roles` | Reused existing ORM model; no schema changes |
| `furnace_data/relational/repositories.py` | Existing direct-mode repository seeds `admin/admin123`, validates SHA-256 hashes, and has minimal user methods | Left direct-mode repository behavior unchanged |
| `src/data/db.py` | Streamlit direct mode owns SHA-256 hashing and default admin seeding | Preserved for compatibility; backend code does not import this module |
| Legacy migration script | Created older unqualified tables and did not match current ORM schemas | Documented mismatch; no destructive migration added in Phase 5 |
| Legacy migration environment | Used `furnace_data.relational.models.Base` metadata | No change in Phase 5 |

No destructive migration was added. Optional `email`, `full_name`, and
`last_login` fields are represented as nullable API fields but are only written
when the deployed ORM/schema supports them.

## Backend Endpoints

- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `POST /api/v1/auth/logout`
- `POST /api/v1/auth/change-password`
- `GET /api/v1/admin/users`
- `POST /api/v1/admin/users`
- `GET /api/v1/admin/users/{user_id}`
- `PATCH /api/v1/admin/users/{user_id}`
- `POST /api/v1/admin/users/{user_id}/reset-password`
- `POST /api/v1/admin/users/{user_id}/deactivate`
- `POST /api/v1/admin/users/{user_id}/activate`
- `GET /api/v1/admin/roles`
- `GET /api/v1/admin/permissions`

Master-data admin APIs are deferred; hopper, burden, feedback, FurnaceMind,
material balance, recommendations, blend optimizer, and AI Copilot were not
migrated in this phase.

## Auth Behavior

- Modern backend password hashes use bcrypt.
- Legacy SHA-256 hashes remain accepted when
  `EVONITH_AUTH_ALLOW_LEGACY_PASSWORD_HASHES=true`.
- Successful backend login upgrades a legacy hash when
  `EVONITH_AUTH_UPGRADE_LEGACY_HASH_ON_LOGIN=true`.
- New passwords must satisfy `EVONITH_AUTH_MIN_PASSWORD_LENGTH`.
- The backend issues signed HS256 bearer access tokens.
- `EVONITH_AUTH_SECRET_KEY` is required when `EVONITH_BACKEND_ENV` is
  `production`, `prod`, or `edge` and
  `EVONITH_AUTH_REQUIRE_SECRET_IN_PRODUCTION=true`.
- Local/test environments without a configured secret use an in-process
  ephemeral secret, so tokens are invalid after restart.

## Feature Flags

```bash
USE_BACKEND_API_AUTH=false
USE_BACKEND_API_ADMIN=false
```

When flags are false, Streamlit continues to use direct-mode auth/admin logic.
When enabled, login and user-management creation/listing use the backend API
adapters.

## Admin Bootstrap

The backend bootstrap script never runs automatically:

```bash
EVONITH_AUTH_BOOTSTRAP_ADMIN_ENABLED=true \
EVONITH_BOOTSTRAP_ADMIN_USERNAME=admin_user \
EVONITH_BOOTSTRAP_ADMIN_PASSWORD='set-a-strong-password' \
python scripts/bootstrap_admin.py
```

The script refuses to run without the explicit enable flag and username/password.
It never deletes or rewrites an existing user.

## Local Usage

Backend:

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime \
EVONITH_AUTH_SECRET_KEY=dev-secret \
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Streamlit direct mode:

```bash
USE_BACKEND_API_AUTH=false \
USE_BACKEND_API_ADMIN=false \
streamlit run src/app.py
```

Streamlit auth/admin API mode:

```bash
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 \
USE_BACKEND_API_AUTH=true \
USE_BACKEND_API_ADMIN=true \
streamlit run src/app.py
```

Edge production example:

```bash
EVONITH_RUNTIME_DIR=/var/lib/evonith-bf
EVONITH_BACKEND_ENV=edge
EVONITH_AUTH_SECRET_KEY=<strong-random-secret>
```

## OpenAPI

Run:

```bash
python scripts/export_backend_openapi.py
```

The exported `docs/api/openapi-v1.json` includes the new auth and admin paths.

## Phase 5 Boundaries

- No backend/frontend split was attempted.
- No Streamlit pages were rewritten.
- No direct-mode auth/admin path was removed.
- No legacy backend route was removed.
- No non-auth feature migration was attempted.
- Backend app modules do not import Streamlit.
- Frontend auth/admin adapters do not import backend internals.

## Follow-Up For Later Phases

- Resolve the Alembic/ORM schema mismatch before database migrations expand.
- Decide whether to add `email`, `full_name`, and `last_login_at` columns.
- Consider refresh tokens or server-side token revocation if operationally needed.
- Replace the direct-mode default admin seed before auth is fully backend-owned.
- Add API-backed edit/reset/deactivate UI controls beyond the minimal Phase 5
  user creation/listing integration.
