# Logging And Audit

## Structured Logging

Phase 10 supports:

- `EVONITH_LOG_LEVEL`
- `EVONITH_LOG_FORMAT=json|text`
- `EVONITH_ACCESS_LOG_ENABLED`
- `EVONITH_ACCESS_LOG_INCLUDE_QUERY_PARAMS`
- `EVONITH_LOG_REDACTION_ENABLED`
- `EVONITH_LOG_FILE_ENABLED`
- `EVONITH_LOG_MAX_FILE_MB`
- `EVONITH_LOG_BACKUP_COUNT`

Console logging is always available. Optional file logging writes only under
`EVONITH_RUNTIME_DIR/logs` with rotation.

Access logs include safe fields:

- `request_id`
- `user_id` when authentication has already resolved
- `method`
- route path
- `status_code`
- `duration_ms`
- `error_code`

Request bodies, file bodies, prompts, documents, raw datasets, Authorization
headers, provider keys, and database connection strings are not logged.

## Redaction

The central redaction service scrubs sensitive keys and values from nested
objects, headers, errors, audit metadata, and log extras. It redacts secret-like
keys including password, token, authorization, bearer, api key, secret,
credential, connection string, database URL, Qdrant key, OpenAI key, access
token, refresh token, and password hash.

It also redacts bearer-token-like values, JWT-like values, database URLs, known
secret assignments, and runtime absolute paths.

## Audit

Audit logging is best-effort and must not break business endpoints. Default
storage is SQLite under `EVONITH_RUNTIME_DIR/audit/audit.db`.

Audit records include:

- id and timestamp
- request id
- actor user id and username when available
- event type
- resource type/id
- action and result
- status code and error code
- hashed client marker
- redacted metadata

Audit events are admin-readable through `/api/v1/ops/audit/events` when
`EVONITH_AUDIT_ADMIN_READ_ENABLED=true`.

Retention cleanup is available through `/api/v1/ops/audit/retention`.

