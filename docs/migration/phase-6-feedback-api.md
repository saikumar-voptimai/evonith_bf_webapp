# Phase 6 Feedback API

Phase 6 moves Feedback/Tickets behind backend-owned APIs while keeping the
existing Streamlit direct-mode ticket service available behind
`USE_BACKEND_API_FEEDBACK=false`.

## Audit

| Area | Current Behavior | Phase 6 Action |
| --- | --- | --- |
| Feedback page | `src/custom_pages/8_Feedback.py` renders a Streamlit form, ticket KPIs, management panel, and ticket board | Preserve direct mode; add API mode behind `USE_BACKEND_API_FEEDBACK=true` |
| Direct UI helpers | `src/utils/feedback_page.py` converts uploads to bytes, renders filters, events, status changes, deletes, and local image previews | Keep for direct mode; add a smaller API-mode renderer without importing backend internals |
| Current storage location | `src/data/tickets/engine.py` defaults ticket SQLite to `runtime/feedback/tickets.db` through Phase 1 runtime paths, with safe copy fallback from old `src/storage/feedback/tickets.db` and `storage/feedback/tickets.db` | Backend repository also defaults to runtime SQLite; no source-controlled writes |
| Current attachments | Direct mode stores screenshot files under `runtime/uploads/feedback` and metadata in `ticket_images` | Backend attachment service stores files under `runtime/uploads/feedback` with generated attachment IDs and sanitized filenames |
| Current ticket fields | `id`, `ticket_code`, `page_name`, `reported_by`, `criticality`, `description`, `ideal_closure_text`, `status`, timestamps, `created_by`, `updated_by` | Backend exposes `title`, `description`, `category`, `priority`, `status`, `page`, reporter/owner metadata, counts, tags, and metadata while preserving migrated legacy fields |
| Current statuses | `open`, `in_progress`, `resolved`, `dependency_conflict`, `closed` | Backend statuses are configured by `EVONITH_FEEDBACK_ALLOWED_STATUSES`; default Phase 6 values are `open,in_progress,resolved,closed,rejected` |
| Current priorities | Direct mode calls them criticalities: `low`, `medium`, `high`, `critical` | Backend uses `priority` with the same default values |
| Current direct-mode code paths | `TicketService`, `TicketRepository`, `TicketImageUpload`, and Feedback UI imports remain under `src/` | Not removed or rewritten |
| Runtime migration behavior | Phase 1 safely copies old ticket DB into runtime if missing; `scripts/migrate_runtime_files.py` is non-destructive | Phase 6 adds feedback-specific dry-run/copy migration into backend-owned tables and attachment files |
| Auth/session behavior | Phase 5 provides backend bearer tokens and `auth_user`, `auth_user_id`, `role`, permissions in Streamlit session state | Feedback APIs use Phase 5 current-user dependencies when `EVONITH_FEEDBACK_REQUIRE_AUTH=true` |
| Backward-compatible data | Old direct SQLite tables and screenshot files must remain readable by direct mode and migratable to backend API mode | Migration is copy-based and duplicate-safe |
| Old paths needing fallback | `src/storage/feedback/tickets.db`, `src/storage/feedback/images`, `storage/feedback/tickets.db`, and runtime direct tables | Direct mode retains its existing fallback; migration script reads old/runtime sources without deleting |

## Phase 6 Changes

- Add backend `/api/v1/feedback` routes for config, tickets, comments, and
  attachments.
- Add backend Pydantic schemas, repository, ticket service, attachment service,
  and migration service.
- Add frontend `src/services/feedback_api.py`.
- Add feature-flagged API mode to the Feedback page.
- Add a non-destructive `scripts/migrate_feedback_tickets.py`.
- Update `.env.example`, OpenAPI, tests, and execution docs.

## Runtime Storage

Backend feedback metadata defaults to the Phase 1 runtime SQLite file at
`runtime/feedback/tickets.db`. Attachment bytes are stored under
`runtime/uploads/feedback/`.

`EVONITH_FEEDBACK_DATABASE_URL` can point the metadata repository at another
SQLite file. `EVONITH_RUNTIME_DIR` continues to control uploaded file storage
and the default metadata path.

## Backend Endpoints

All endpoints are under `/api/v1/feedback`.

| Endpoint | Purpose |
| --- | --- |
| `GET /config` | Return statuses, priorities, categories, and upload policy |
| `GET /tickets` | List visible tickets with filters |
| `POST /tickets` | Create a ticket |
| `GET /tickets/{ticket_id}` | Read a ticket by backend ID or ticket number |
| `PATCH /tickets/{ticket_id}` | Update allowed fields |
| `POST /tickets/{ticket_id}/close` | Close a ticket |
| `POST /tickets/{ticket_id}/reopen` | Reopen a ticket |
| `GET /tickets/{ticket_id}/comments` | List ticket comments |
| `POST /tickets/{ticket_id}/comments` | Add a comment |
| `GET /tickets/{ticket_id}/attachments` | List active attachments |
| `POST /tickets/{ticket_id}/attachments` | Upload one attachment |
| `GET /attachments/{attachment_id}/download` | Download attachment bytes |
| `DELETE /attachments/{attachment_id}` | Soft-delete attachment metadata and remove file |

## Auth and Ownership

`EVONITH_FEEDBACK_REQUIRE_AUTH=true` is the default. In authenticated mode,
normal users can create and view their own tickets. Admin and supervisor users
can list, update, close, reopen, and delete attachments across all tickets.

When `EVONITH_FEEDBACK_REQUIRE_AUTH=false`, local API smoke tests and dev
workflows can exercise the API without bearer tokens.

## Attachment Policy

Attachment filenames are sanitized before storage. Backend responses expose
logical filenames and download URLs only; runtime filesystem paths are never
returned. Size, content type, extension, and per-ticket count are controlled by
the `EVONITH_FEEDBACK_*ATTACHMENT*` environment variables.

## Frontend Feature Flag

Direct Streamlit mode remains the default:

```bash
USE_BACKEND_API_FEEDBACK=false streamlit run src/app.py
```

API mode is enabled per page:

```bash
USE_BACKEND_API_AUTH=true USE_BACKEND_API_FEEDBACK=true streamlit run src/app.py
```

Backend sidecar example:

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Migration

The Phase 6 migration is copy-only and non-destructive:

```bash
python scripts/migrate_feedback_tickets.py --dry-run
python scripts/migrate_feedback_tickets.py
python scripts/migrate_feedback_tickets.py --overwrite
```

The script reads legacy direct-mode ticket tables from runtime and known old
fallback paths. It never deletes old ticket rows or old attachment files.

## Deferred

- Material Balance, Recommendations, Blend Optimizer, AI Copilot, and
  FurnaceMind migration.
- Removal of direct-mode Feedback storage and UI.
- Destructive DB migrations.
- Object storage integration.
- Full PostgreSQL migrations for feedback tables; repository shape remains
  compatible with later PostgreSQL work.
