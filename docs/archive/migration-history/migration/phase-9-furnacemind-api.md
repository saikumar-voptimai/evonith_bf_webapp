# Phase 9 FurnaceMind API

## Phase 9 Goal

Move FurnaceMind conversations, messages, runs, events, documents, memory
lookups, optional LLM calls, safe tools, artifacts, and feedback behind
backend-owned APIs while keeping the current direct Streamlit FurnaceMind page
available behind a false-by-default feature flag.

## Audit

### Current Direct FurnaceMind Flow

`src/custom_pages/7_FurnaceMind.py` remains the direct-mode FurnaceMind page.
It imports the Streamlit UI stack and the legacy `src/agents/furnacemind`
implementation. Direct mode still owns the existing agent loop, prompt assets,
skills, Qdrant integrations, reports, and UI state when
`USE_BACKEND_API_FURNACEMIND=false`.

### Runtime Data And State

Phase 9 backend API mode stores FurnaceMind runtime state under
`EVONITH_RUNTIME_DIR`:

- `runtime/furnacemind/furnacemind.db` for SQLite conversations, messages,
  runs, events, document metadata, chunks, and feedback.
- `runtime/uploads/furnacemind/documents/` for uploaded document bytes.
- Existing compute artifact runtime storage for FurnaceMind JSON artifacts.

No legacy source data is deleted or moved.

### Provider, Memory, And Tool Paths

Direct-mode provider, Qdrant, and tool code remains in `src/agents`. Backend API
mode adds safe backend-owned abstractions with lazy imports:

- LLM provider calls are disabled by default.
- Qdrant/vector memory and embeddings are disabled by default.
- Tools are disabled by default and limited to an allowlist.
- Code execution and shell execution are blocked.

### Deferred Direct-Mode Areas

These areas were intentionally not removed or rewritten in Phase 9:

- Legacy direct Streamlit FurnaceMind agent mode.
- Legacy direct Qdrant/vector memory behavior.
- Legacy direct prompt files and source assets.
- SSE streaming transport.
- Full external job queue or worker process.
- Full frontend/backend repository split.

## What Changed

- Added backend route module
  `furnace-data-service/app/api/v1/routes/furnacemind.py`.
- Added schemas in
  `furnace-data-service/app/api/v1/schemas/furnacemind.py`.
- Added SQLite repositories for conversations/messages/feedback, runs/events,
  and documents/chunks.
- Added backend services for safety, documents, memory, retrieval, prompts,
  optional LLM calls, safe tools, events, and orchestration.
- Registered FurnaceMind service startup in `furnace-data-service/app/main.py`.
- Added frontend adapter `src/services/furnacemind_api.py`.
- Added API-mode branch to `src/custom_pages/7_FurnaceMind.py` behind
  `USE_BACKEND_API_FURNACEMIND=true`.
- Added FurnaceMind backend settings to `.env.example`.
- Regenerated `docs/api/openapi-v1.json`.
- Added focused backend, frontend, and integration tests.

## What Did Not Change

- Direct-mode FurnaceMind remains the default.
- Existing direct-mode page behavior remains available when
  `USE_BACKEND_API_FURNACEMIND=false`.
- The FastAPI sidecar remains in place.
- No page was migrated into a separate frontend app.
- No direct-mode prompt assets, source assets, or legacy data files were deleted.
- No arbitrary code execution, shell execution, or provider call is enabled by
  default.

## Backend Endpoints

- `GET /api/v1/furnacemind/config`
- `GET /api/v1/furnacemind/conversations`
- `POST /api/v1/furnacemind/conversations`
- `GET /api/v1/furnacemind/conversations/{conversation_id}`
- `PATCH /api/v1/furnacemind/conversations/{conversation_id}`
- `POST /api/v1/furnacemind/conversations/{conversation_id}/archive`
- `GET /api/v1/furnacemind/conversations/{conversation_id}/messages`
- `POST /api/v1/furnacemind/conversations/{conversation_id}/messages`
- `POST /api/v1/furnacemind/conversations/{conversation_id}/runs`
- `GET /api/v1/furnacemind/runs/{run_id}`
- `GET /api/v1/furnacemind/runs/{run_id}/events`
- `GET /api/v1/furnacemind/documents`
- `POST /api/v1/furnacemind/documents`
- `GET /api/v1/furnacemind/documents/{document_id}`
- `DELETE /api/v1/furnacemind/documents/{document_id}`
- `POST /api/v1/furnacemind/documents/{document_id}/index`
- `GET /api/v1/furnacemind/tools`
- `GET /api/v1/furnacemind/artifacts/{artifact_id}/download`
- `POST /api/v1/furnacemind/messages/{message_id}/feedback`

## Feature Flag

API mode is disabled by default:

```bash
USE_BACKEND_API_FURNACEMIND=false
```

Enable only for Phase 9 testing or controlled rollout:

```bash
USE_BACKEND_API=true
USE_BACKEND_API_AUTH=true
USE_BACKEND_API_FURNACEMIND=true
```

When the flag is true, the Streamlit page calls
`src/services/furnacemind_api.py`. If the backend is unavailable or rejects the
request, the page shows a backend error instead of silently falling back to
direct mode.

## Backend Settings

The backend FurnaceMind settings added in Phase 9 are documented in
`.env.example`. Important safe defaults:

- `EVONITH_FURNACEMIND_REQUIRE_AUTH=true`
- `EVONITH_FURNACEMIND_LLM_ENABLED=false`
- `EVONITH_FURNACEMIND_ENABLE_PROVIDER_CALLS=false`
- `EVONITH_FURNACEMIND_ALLOW_RAW_DOCS_TO_LLM=false`
- `EVONITH_FURNACEMIND_MEMORY_ENABLED=false`
- `EVONITH_FURNACEMIND_EMBEDDINGS_ENABLED=false`
- `EVONITH_FURNACEMIND_TOOLS_ENABLED=false`
- `EVONITH_FURNACEMIND_ENABLE_CODE_EXECUTION=false`
- `EVONITH_FURNACEMIND_ENABLE_SHELL_EXECUTION=false`

Provider, vector, and embedding API keys must live in secure environment
variables. The `*_API_KEY_ENV` settings only name which environment variable to
read.

## Backend-Owned Responsibilities

Phase 9 backend API mode owns:

- Conversation and message persistence.
- User ownership and auth gating.
- Run lifecycle and polling events.
- Document upload validation, safe storage, metadata, text extraction, and
  optional indexing.
- Bounded retrieval context from conversation history, documents, and optional
  memory.
- Prompt construction and prompt length capping.
- Optional/mockable LLM provider abstraction.
- Safe allowlisted tool summaries.
- JSON artifacts through the existing runtime artifact service.
- Message feedback persistence.

## Safety Controls

- Sensitive keys, bearer/API-key-like text, runtime paths, and provider secrets
  are redacted from API-visible text.
- Message, history, context, prompt, output, document, event, and tool-call sizes
  are capped.
- Raw document text is excluded from provider prompts unless explicitly allowed.
- Provider calls require both LLM enablement and provider-call enablement.
- Qdrant and provider SDKs are imported lazily only when enabled and needed.
- Tool execution is allowlisted and disabled by default.
- Code execution and shell execution are rejected.
- Uploaded filenames are sanitized and stored under runtime directories.
- API responses expose IDs and URLs, not internal filesystem paths.
- Backend services do not import Streamlit.

## LLM, Memory, And Tools

Default local and edge behavior is deterministic non-LLM analysis. Mock provider
mode is available for tests without real provider SDKs or secrets.

Memory defaults to disabled. Tests may use the fake vector backend. Qdrant
requires explicit enablement and configuration and is not imported during
ordinary startup.

Tools default to disabled. When enabled, only the configured allowlisted summary
tools are available; arbitrary code and shell tools are not registered.

## OpenAPI

`docs/api/openapi-v1.json` was regenerated after adding the FurnaceMind router.
The schema includes all `/api/v1/furnacemind/*` endpoints listed above.

## Compatibility Fixes During Phase 9 Verification

The full root suite exposed previously documented failures outside FurnaceMind.
Small compatibility fixes were applied so the broad suite is useful again:

- BMO fuel-rate inverse estimates now match the regression contract.
- Burden-distribution snapshots normalize to daily anchors before forward-fill.
- Static dataset cache handling tolerates synthetic future fixture dates and
  stale cutoffs while preserving the explicit future-hour clipping behavior.

## Deferred

- SSE streaming transport for run events.
- PostgreSQL storage backend for FurnaceMind state.
- Production Qdrant and embedding provider integration hardening.
- Richer backend tool implementations beyond safe summaries.
- Full direct-mode FurnaceMind retirement.
- Phase 2-style frontend/backend repository split.
