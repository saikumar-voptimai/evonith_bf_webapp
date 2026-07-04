# Phase 8 AI Copilot API

## Phase 8 Goal

Move AI Copilot data retrieval, context construction, anomaly summaries, safety
controls, optional LLM calls, jobs, and artifacts behind backend-owned APIs while
keeping the current direct Streamlit Copilot available behind a false-by-default
feature flag.

## Audit

### Current Direct Copilot Flow

`src/custom_pages/5_AI_Copilot.py` is still the direct-mode Copilot page. It
imports Streamlit UI dependencies, fetches recent online data, computes
Channeling propensity, loads static markdown findings, builds prompts, and calls
the direct LLM helper when the operator enables LLM analysis.

### Current Data Sources Used

- Live/recent online data through `src/utils/copilot/data.py` and
  `furnace_data.influx.online.fetch_online_df`.
- Static Copilot analysis text in `src/assets/data/copilot_analysis`.
- Existing Channeling anomaly logic through `src/utils/anomaly_propensity.py`.
- Optional direct-mode OpenAI/OpenRouter calls through
  `src/agents/llm/llm_client.py`.

### Current Prompt And LLM Paths

Direct mode still uses `src/utils/copilot/prompts.py` and
`src/agents/llm/llm_client.py`. Phase 8 adds backend prompt construction and an
optional provider abstraction, but does not remove direct-mode prompt or LLM
code.

### Current Runtime And Output Behavior

The direct page renders analysis in Streamlit only. Phase 8 backend API mode can
create JSON analysis artifacts under the existing runtime compute artifact
storage when exports are requested or results are large.

### Current Imports Unsafe In Backend

Direct page modules import Streamlit and must not be imported by backend routes.
Backend Copilot services avoid Streamlit imports and use lazy imports for
selected existing source-domain helpers only when a request needs them.

### Deferred Copilot Areas

These areas were intentionally not migrated in Phase 8:

- FurnaceMind.
- Qdrant memory.
- Document ingestion or RAG.
- Persistent conversations.
- Autonomous agents, tool execution, and code execution.
- Page rewrite or UI redesign.

## What Changed

- Added backend route module `furnace-data-service/app/api/v1/routes/copilot.py`.
- Added backend schemas in `furnace-data-service/app/api/v1/schemas/copilot.py`.
- Added backend services for Copilot orchestration, data retrieval, anomaly
  scoring, context, prompt safety, redaction, optional LLM calls, and artifacts.
- Added frontend adapter `src/services/copilot_api.py`.
- Added API-mode branch to `src/custom_pages/5_AI_Copilot.py` behind
  `USE_BACKEND_API_COPILOT=true`.
- Added Copilot backend settings to `.env.example`.
- Exported OpenAPI schema with Copilot endpoints.
- Added focused backend, frontend, and integration tests.

## What Did Not Change

- Direct-mode AI Copilot remains the default.
- Existing Streamlit page behavior remains available when
  `USE_BACKEND_API_COPILOT=false`.
- FurnaceMind, Qdrant, document RAG, persistent conversations, autonomous agent
  execution, and code execution were not migrated.
- No frontend/backend split was attempted.
- No existing Copilot prompt assets were moved or deleted.

## Backend Endpoints

- `GET /api/v1/copilot/config`
- `POST /api/v1/copilot/recent-data`
- `POST /api/v1/copilot/anomaly`
- `POST /api/v1/copilot/analyze`
- `POST /api/v1/copilot/jobs`
- `GET /api/v1/copilot/jobs/{job_id}`
- `GET /api/v1/copilot/artifacts/{artifact_id}/download`

## Feature Flag

API mode is disabled by default:

```bash
USE_BACKEND_API_COPILOT=false
```

Enable only for Phase 8 testing or controlled rollout:

```bash
USE_BACKEND_API_COPILOT=true
```

When the flag is true, the Streamlit page calls `src/services/copilot_api.py`.
If the backend is unavailable or rejects the request, the page shows the backend
error and request ID when available. It does not silently fall back to direct
mode.

## Backend Settings

The backend Copilot settings added in Phase 8 are documented in `.env.example`.
Important defaults:

- `EVONITH_COPILOT_REQUIRE_AUTH=true`
- `EVONITH_COPILOT_LLM_ENABLED=false`
- `EVONITH_COPILOT_ENABLE_PROVIDER_CALLS=false`
- `EVONITH_COPILOT_ALLOW_RAW_DATA_TO_LLM=false`
- `EVONITH_COPILOT_ENABLE_CODE_EXECUTION=false`
- `EVONITH_COPILOT_ENABLE_DATA_REDACTION=true`

Provider calls require both LLM enablement and provider-call enablement. The
mock provider is available for tests without importing real provider SDKs.

## Backend-Owned Responsibilities

Phase 8 backend API mode owns:

- Recent data retrieval and JSON-safe preview shaping.
- Numeric anomaly summaries with optional lazy Channeling detector use.
- Compact redacted context construction.
- Prompt construction and prompt length capping.
- Optional/mockable LLM provider abstraction.
- Safety warnings for raw data exclusion, prompt truncation, and empty data.
- In-process Copilot jobs using the existing compute job service.
- JSON artifacts using the existing runtime artifact service.

## Safety Controls

- Sensitive keys and bearer/API-key-looking text are redacted.
- Raw sample rows are excluded from provider context unless explicitly allowed.
- Prompt length, output length, context rows, and JSON rows are capped.
- Code execution requests are rejected with `COPILOT_UNSAFE_INPUT`.
- Provider calls are disabled unless explicitly configured.
- API responses expose artifact IDs/download URLs, not filesystem paths.
- Backend services do not import Streamlit.

## LLM Behavior

If `allow_llm=true`, the backend attempts the configured provider call and
returns structured errors for disabled, unconfigured, unavailable, timeout, or
invalid provider responses.

If `allow_llm=false` or omitted, the backend returns deterministic non-LLM
analysis using recent data and anomaly summaries. This keeps API mode useful in
local and edge deployments without requiring secrets.

## Jobs And Artifacts

Copilot jobs use the existing in-process compute job service. The Phase 8 job
endpoint runs inline for now, matching the Phase 7 migration style. Large or
exported analysis results create JSON artifacts under the runtime compute
artifact directory.

## OpenAPI

`docs/api/openapi-v1.json` was regenerated after adding the Copilot router. The
schema includes all seven `/api/v1/copilot/*` endpoints.

## Deferred

- Persistent Copilot conversation storage.
- External job queue or worker process.
- Full provider matrix beyond mock and OpenAI-compatible implementation.
- FurnaceMind, Qdrant memory, document ingestion, and RAG.
- Rich frontend redesign for Copilot API mode.
- Phase 2 frontend/backend split work.
