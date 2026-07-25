# FurnaceMind Phase 0 Baseline

- Status: complete
- Baseline date: 2026-07-18
- DEV branch: `dev`
- DEV commit: `1fc2ba73b01f6aa890f3c307c95945669d9f05d1`
- UAT reference: `evonith_bf_webapp-UAT.zip`
- UAT SHA-256: `D57323E481CD4926A00442F72E787D23EACB3C1F1A8E4033054304C1994A1067`
- Roadmap SHA-256: `9FC9B0B3A9CCD1D254E4524BC1D1F22BB7E5D7F1EE51FA6652D2C0045679C178`

## Outcome

The current DEV branch has a stable, safety-first FurnaceMind API scaffold, but it does not yet provide the roadmap's complete API-based assistant. The first implementation branch can start with the LangGraph backend runtime.

The agreed delivery strategy supersedes the roadmap's suggestion of one long-lived integration branch: create one short-lived feature branch at a time from the latest merged `dev`, complete and validate that feature, merge it, and then start the next branch.

The core API chat gate is FM-01 through FM-04. Do not start skills, memory, documents, web search, or UI enhancements until those four features work together through FastAPI.

## Sources reviewed

- Current DEV backend, Streamlit frontend, shared data package, configuration, tests, and archived Phase 9 notes.
- The supplied UAT archive, including its LangGraph workflow, reasoning-profile implementation, and FurnaceMind tests.
- `FurnaceMind_Implementation_Roadmap.docx`.

UAT is a reference implementation, not a merge source. Its LangGraph code is coupled to Streamlit status UI and direct frontend tools, so the state transitions should be adapted to backend services rather than copied unchanged. UAT contains reasoning-profile work, but no approved web-search or URL-ingestion implementation was found; web access is net-new work.

## Current foundation

### API and safety

- FastAPI routes already cover configuration, conversations, messages, runs, polling events, documents, tool metadata, artifact downloads, and message feedback.
- Authentication is required by default, and conversation/document access is owner-scoped with an admin override.
- Provider calls, FurnaceMind LLM execution, tools, memory, code execution, and shell execution use safe defaults and are disabled unless explicitly configured.
- A Streamlit API adapter and API-mode page exist, but `USE_BACKEND_API_FURNACEMIND` defaults to `false`. API-mode chat currently sends `allow_llm: false`, and direct Streamlit execution remains available.

### Persistence

- The active FurnaceMind API repositories store conversations, messages, runs, events, documents, chunks, and feedback in SQLite and create their tables directly.
- The shared `furnace_data` package already defines PostgreSQL-oriented SQLAlchemy models and repositories for conversations, messages, documents, summaries, facts, skills, and feedback.
- These two persistence paths are not wired together. There is no active migration path that creates and evolves the shared FurnaceMind PostgreSQL schema for the backend API.
- The obsolete UAT migration must not be copied: its schema does not match the current separated architecture.

### Runtime capabilities

- A synchronous run service builds prompts, stores run state, and can call the configured LLM service. It is not a LangGraph runtime.
- API mode disables LLM use in its request. Tool execution is driven by client-supplied `options.tool_calls`, not by model decisions.
- The backend tool registry exposes five allowlisted names. Only `data_summary` and `anomaly_summary` perform calculations; the remaining three return availability notes.
- Run events support polling. There is no SSE endpoint, background run worker, or cancellation endpoint.
- Backend document upload supports TXT, Markdown, CSV, and JSON extraction. PDF can be accepted by configuration but extraction is explicitly unavailable. Office documents, slides, images, table-aware extraction, and citation-rich retrieval are not implemented.
- The fake memory mode works for tests. Real Qdrant indexing and search are explicitly unavailable in the backend runtime.
- The feedback API supports rating, helpful state, comment, and tags. The API-mode UI exposes only a `Helpful` button.
- Downloadable JSON artifacts exist. Rich inline plots, tables, and approved images are still direct-mode behavior.

## Missing-feature matrix

| ID | Feature | DEV status | UAT/reference finding | Required result |
| --- | --- | --- | --- | --- |
| FM-01 | Backend LangGraph runtime | Missing | UAT has a direct-mode `StateGraph` with model, tool, and finalize nodes | Backend-owned, bounded graph with testable state and no Streamlit imports |
| FM-02 | PostgreSQL chat persistence and migrations | Partial | Shared ORM models/repositories exist; active API uses SQLite | Versioned schema and backend repositories for conversations, messages, runs, events, and errors |
| FM-03 | Core chat API, real LLM, and progress | Partial | API contracts exist; processing is synchronous and API UI disables LLM | Authenticated backend graph run returns an answer, persists failure/success, and reports progress |
| FM-04 | Streamlit API-only basic chat | Partial | API adapter exists; direct fallback remains | Multi-turn chat uses FastAPI only and never executes the agent in Streamlit |
| FM-05 | Backend skill runtime | Direct-mode only | Rich direct-mode skill registry/engine exists | Backend lists, selects, validates, and executes enabled skills |
| FM-06 | Furnace data, plot, and report tools | Partial/direct-mode only | Direct frontend has real furnace tools; backend mostly exposes placeholders | Model-driven, allowlisted backend tools with permissions, enforced timeout, audit, and safe errors |
| FM-07 | Low/Medium/High reasoning profiles | UAT only | UAT has OpenRouter profiles and tests | API request, backend model selection, response metadata, and UI selection agree |
| FM-08 | Short-term memory and summarization | Partial/direct-mode only | API includes recent history; direct mode has summarization | Backend context window plus persisted summaries of older messages |
| FM-09 | Long-term memory | Scaffold/direct-mode only | Shared models and direct Qdrant code exist; backend real mode is unavailable | PostgreSQL/Qdrant save, deduplicate, retrieve, isolate, and delete memories |
| FM-10 | Knowledge documents and citations | Partial | API CRUD/chunk scaffold exists | Approved formats are extracted/indexed and answers include verifiable citations |
| FM-11 | Web search and approved URL ingestion | Missing | No implementation found in UAT or DEV | Opt-in safe search/ingestion with SSRF controls, limits, citations, and audit logging |
| FM-12 | Inline artifacts | Partial/direct-mode only | API offers JSON downloads; direct UI has richer artifacts | Text, plots, tables, files, and approved images render from typed API artifacts |
| FM-13 | Complete feedback UI and storage | Partial | Backend contract is richer than API-mode UI | Thumbs up/down and agreed comment behavior persist with user ownership |
| FM-14 | Run lifecycle and dependency states | Partial | Polling events/status fields exist | Queued, running, completed, failed, cancelled, and dependency-unavailable states are actionable |
| FM-15 | Full UAT and API-only release | Pending | Direct fallback is still the default | Full regression/UAT passes with direct fallback disabled and approvals recorded |

## Implementation order and branches

Each branch starts from the latest `dev` after the previous accepted branch is merged.

| Order | Branch | Scope | Depends on |
| --- | --- | --- | --- |
| 1 | `feature/furnacemind-langgraph-runtime` | FM-01 | Phase 0 |
| 2 | `feature/furnacemind-postgres-persistence` | FM-02 | FM-01 contracts |
| 3 | `feature/furnacemind-core-chat-api` | FM-03 | FM-01, FM-02 |
| 4 | `feature/furnacemind-streamlit-api-chat` | FM-04 | FM-03 |
| 5 | `feature/furnacemind-skill-runtime` | FM-05 | Core API chat gate |
| 6 | `feature/furnacemind-furnace-tools` | FM-06 | FM-05 |
| 7 | `feature/furnacemind-reasoning-profiles` | FM-07 | FM-03 |
| 8 | `feature/furnacemind-short-term-memory` | FM-08 | FM-02, FM-03 |
| 9 | `feature/furnacemind-long-term-memory` | FM-09 | FM-08 |
| 10 | `feature/furnacemind-knowledge-documents` | FM-10 | FM-02, FM-09 |
| 11 | `feature/furnacemind-web-search` | FM-11 | FM-03, FM-10 citation contract |
| 12 | `feature/furnacemind-inline-artifacts` | FM-12 | FM-04, FM-06 |
| 13 | `feature/furnacemind-feedback` | FM-13 | FM-02, FM-04 |
| 14 | `feature/furnacemind-run-controls` | FM-14 | FM-03, FM-04 |
| 15 | `release/furnacemind-api-uat` | FM-15 only: integration fixes, UAT, and release evidence | FM-01 through FM-14 |

FM-07 may be implemented after FM-04 if UI selection is included in that branch. It must not delay the core API chat gate. FM-15 must not introduce a new feature; any failed acceptance item is fixed in its owning feature area before release approval.

## Acceptance checklist

These scenarios are the minimum automated acceptance tests. Each feature branch may add lower-level tests, failure cases, and operational checks.

### AT-FM-01 — LangGraph runtime

Given a model stub that first requests an allowlisted tool and then returns an answer, when the backend graph runs, then it traverses model, tool, and finalize nodes, stops within the configured loop limit, and returns a serializable final state without importing Streamlit.

### AT-FM-02 — PostgreSQL persistence

Given an authenticated user and a migrated empty database, when a conversation run succeeds and the backend restarts, then its conversation, ordered messages, run, and events remain available; a second user receives `403` or `404` for those records.

### AT-FM-03 — Core chat API

Given a configured provider stub and LLM execution enabled, when an authenticated client posts a run, then the backend invokes LangGraph, persists user and assistant messages, exposes progress, and returns the provider answer. A provider failure produces a persisted failed run with a stable error code.

### AT-FM-04 — Streamlit API-only chat

Given API mode enabled, when a user sends two related messages, then Streamlit calls only FurnaceMind API endpoints and displays both answers from persisted history. If the API is unavailable, the page shows an error and does not invoke the direct-mode agent.

### AT-FM-05 — Skill runtime

Given one enabled skill with valid input, when the model selects it, then the backend validates and executes the registered skill and records the result. A disabled or unknown skill is rejected with a stable safe error.

### AT-FM-06 — Furnace tools

Given one allowlisted furnace-data tool, when the graph supplies valid input, then the backend executes the real service and returns a redacted result. Invalid input, missing permission, and an enforced timeout return distinct stable errors without executing arbitrary code.

### AT-FM-07 — Reasoning profiles

Given Low, Medium, and High requests, when each run is created, then the configured model/effort pair is selected and recorded in run metadata. An unavailable profile returns a clear dependency/configuration error.

### AT-FM-08 — Short-term memory

Given a conversation longer than the message-window limit, when the next turn runs, then recent messages and a backend-generated summary of older messages are included once, and the summary remains available after restart.

### AT-FM-09 — Long-term memory

Given a user-approved durable fact, when it is saved twice and later queried, then only one PostgreSQL/Qdrant memory is retrieved for that user. Another user cannot retrieve it, and deletion removes both relational metadata and vector points.

### AT-FM-10 — Knowledge documents

Given one representative file for each approved format, when files are uploaded and indexed, then relevant questions return source identifiers and page/section/chunk citations. Listing shows only the owner's documents, and deletion removes stored files, metadata, and vector points.

### AT-FM-11 — Web search and URL ingestion

Given web access disabled, when a prompt requests current web data, then no network tool runs. Given web access enabled, a safe public HTTPS source can be retrieved, limited, cited, and audited, while loopback, private-network, non-HTTP, redirect-to-private, and oversized targets are blocked.

### AT-FM-12 — Inline artifacts

Given a tool result containing a typed plot, table, file, or approved image artifact, when the API-mode UI renders the assistant message, then the artifact appears inline with a working authorized download where applicable; malformed or unauthorized artifacts render a safe error.

### AT-FM-13 — Feedback

Given an assistant message, when its owner submits thumbs up or thumbs down with the agreed optional/required comment, then the UI reflects the saved state and the backend persists rating, comment, tags, message, conversation, and user ownership. Another user cannot alter it.

### AT-FM-14 — Run lifecycle

Given a deliberately slow run, when it starts and is then cancelled, then ordered events show queued, running, and cancelled states, no later completed answer is persisted, and repeated cancellation is idempotent. Missing LLM, database, Qdrant, and web dependencies produce distinguishable unavailable states.

### AT-FM-15 — API-only UAT and release

Given direct fallback disabled, when the full automated suite and representative operator UAT are run, then every FM-01 through FM-14 acceptance item passes, authentication and user-data isolation pass, failure drills pass, and Sasikumar's technical approval plus Sai's final approval are recorded.

## Definition of done for every feature branch

- The branch contains only its named feature plus necessary tests and documentation.
- Public API/schema/config changes are documented and keep safe defaults.
- Automated acceptance for the feature passes alongside the existing FurnaceMind regression suite.
- Authentication, authorization, redaction, timeouts, and audit behavior are tested where relevant.
- No new direct Streamlit execution dependency is introduced into backend code.
- The feature can be disabled or rolled back without corrupting persisted data.
- The branch is reviewed and merged before the next dependent branch is created.

## Baseline verification

The following existing tests pass on the recorded DEV commit:

```text
tests/backend/service_api/test_api_v1_furnacemind.py
tests/backend/service_api/test_furnacemind_services.py
tests/backend/service_api/test_furnacemind_repositories.py
tests/frontend/test_furnacemind_api.py
tests/integration/test_phase9_furnacemind_flow.py

12 passed, 2 deprecation warnings
```

The warnings are existing Python datetime deprecations in `reactivex` and `furnace_data.bmo.data.context_provider`; they do not block Phase 0.

## Phase 0 exit decision

- DEV source and exact reference artifacts are recorded.
- Missing and partial capabilities are identified.
- Database and API foundations are confirmed, including the persistence split that must be resolved.
- One acceptance scenario exists for every planned feature.
- Branch order and merge discipline are fixed.
- Phase 1 may start on `feature/furnacemind-langgraph-runtime` from the latest `dev`.
