# Phase 7 Domain Compute API

## Phase 7 Goal

Move Material Balance, Recommendations, and Blend Optimizer compute workloads
behind backend-owned APIs while keeping direct Streamlit mode available behind
false-by-default feature flags.

## Audit

### Current Material Balance Flow

`src/custom_pages/6_Material_Balance.py` gates on Streamlit login, reads static
ML dataset availability, accepts a date, lag values, and dust catcher tonnes,
then calls `utils.material_balance.compute.run_full_balance`. The page renders
KPIs, Plotly figures, closure tables, ash-analysis editors, DPR mapping editors,
and assumptions.

### Current Recommendations Flow

`src/custom_pages/4_Recommendations.py` loads `setting_vsense.yml`, loads model
artifacts with `joblib`, creates a `DataframesProcessor`, fetches static/live
data, lets operators adjust control bounds, runs `utils.recommendations.optimiser`,
and optionally calls an LLM for prose review.

### Current Blend Optimizer Flow

`src/custom_pages/9_Blend_Optimizer.py` reads BMO settings/mapping, builds ore,
flux, fuel, dust, hot-metal, and slag context, saves operator preferences under
runtime cache, runs LP baseline and nonlinear optimization, and optionally runs
fuel/Si prediction models.

### Current Model Assets Used

Model assets live under `src/assets/models/**`, including `*.joblib`, `*.pkl`,
and `*.json` files for V-Sense and BMO fuel/Si predictions. Direct pages load
some models eagerly during page execution.

### Current Data Sources Used

- Static ML dataset: `src/assets/data/furnace_dataset.csv` and runtime dataset
  cache from earlier phases.
- Material Balance: static dataset plus optional DPR/Influx data and
  `src/config/material_balance.yml`.
- Recommendations: static ML dataset, optional recent live Influx values,
  `setting_vsense.yml`, persisted control bounds.
- Blend Optimizer: BMO YAML mapping/settings, offline feed tables, static ML
  history, current stock/chemistry snapshots, runtime operator preferences.

### Current Runtime Files Created Or Modified

- Recommendations bounds: `runtime/cache/control_bounds.json`.
- BMO operator preferences: `runtime/cache/bmo_operator_inputs.yml`.
- Static dataset versions and metadata through Phase 4 dataset services.
- Material Balance direct page can save ash analyses and DPR mapping into the
  source config file; Phase 7 backend does not mutate those files.

### Current Config Files Read And Modified

- Read: `setting_vsense.yml`, `setting_bmo.yml`, `bmo_ore_mapping.yml`,
  `material_balance.yml`, `setting_ds_dv.yml`, `shift_report.yml`.
- Direct-mode mutable paths remain in direct pages.
- Backend Phase 7 treats source configs as read-only unless a future runtime
  config feature is explicitly enabled.

### Current Direct-Mode Code Paths

Direct-mode pages remain:

- `src/custom_pages/6_Material_Balance.py`
- `src/custom_pages/4_Recommendations.py`
- `src/custom_pages/9_Blend_Optimizer.py`

Existing domain modules remain under `src/utils/**`, `src/domain/**`,
`src/data/**`, and `src/ui/**`.

### Current Output Formats, Tables, And Charts

Material Balance renders closure tables and Plotly charts. Recommendations
renders metrics and optional LLM prose. Blend Optimizer renders material tables,
metrics, diagnostics, comparisons, and charted share/cost views.

Phase 7 APIs return JSON-safe summaries, table rows, chart-series-friendly data,
and CSV artifacts for large/exported results. They do not return Streamlit,
Plotly, or filesystem objects.

### Current Error-Prone Or Heavy Operations

- Eager model loading in Recommendations direct mode.
- Optional live database/Influx reads.
- Nonlinear optimizer iterations.
- YAML mutation from direct editor flows.
- Large table serialization.
- Optional model dependencies.

### Current Imports Unsafe In Backend

Direct page modules import Streamlit and should not be imported by backend
routes. Backend services use lazy imports for `src/utils` and `src/data` code
only when a request needs those workflows.

### Safe Migration Points For Phase 7

- Wrap `run_full_balance` for Material Balance static dataset requests.
- Provide deterministic non-LLM recommendation results for API mode.
- Use bounded Blend Optimizer candidate generation and a lazy model registry.
- Reuse Phase 1 runtime paths and Phase 4 job/artifact patterns.
- Keep direct-mode pages intact and add feature-flag API branches.

## What Changed

- Added `/api/v1/material-balance`, `/api/v1/recommendations`, and
  `/api/v1/blend-optimizer` route modules.
- Added compute common schemas and domain schemas.
- Added `MaterialBalanceService`, `RecommendationService`,
  `BlendOptimizerService`, `ModelRegistryService`,
  `ComputeArtifactService`, and `ComputeJobService`.
- Added frontend API adapters:
  `src/services/material_balance_api.py`,
  `src/services/recommendations_api.py`,
  `src/services/blend_optimizer_api.py`.
- Added feature-flagged API mode branches to the three Streamlit pages.

## What Did Not Change

- AI Copilot, FurnaceMind, LLM chat, and Qdrant memory were not migrated.
- Direct-mode Material Balance, Recommendations, and Blend Optimizer remain.
- Existing source config files are not migrated or destructively rewritten.
- Legacy backend routes remain.

## Backend Endpoints

Material Balance:

- `GET /api/v1/material-balance/config`
- `POST /api/v1/material-balance/validate`
- `POST /api/v1/material-balance/run`
- `POST /api/v1/material-balance/jobs`
- `GET /api/v1/material-balance/jobs/{job_id}`
- `GET /api/v1/material-balance/artifacts/{artifact_id}/download`

Recommendations:

- `GET /api/v1/recommendations/config`
- `POST /api/v1/recommendations/run`
- `POST /api/v1/recommendations/jobs`
- `GET /api/v1/recommendations/jobs/{job_id}`
- `GET /api/v1/recommendations/artifacts/{artifact_id}/download`

Blend Optimizer:

- `GET /api/v1/blend-optimizer/context`
- `GET /api/v1/blend-optimizer/models`
- `POST /api/v1/blend-optimizer/predict`
- `POST /api/v1/blend-optimizer/optimize`
- `POST /api/v1/blend-optimizer/jobs`
- `GET /api/v1/blend-optimizer/jobs/{job_id}`
- `GET /api/v1/blend-optimizer/artifacts/{artifact_id}/download`

## Feature Flags

API mode is disabled by default:

- `USE_BACKEND_API_MATERIAL_BALANCE=false`
- `USE_BACKEND_API_RECOMMENDATIONS=false`
- `USE_BACKEND_API_BLEND_OPTIMIZER=false`

When enabled, the corresponding page calls its frontend API adapter. If backend
calls fail, the page shows a clean error with request ID when available.

## Model Registry And Lazy Loading

`ModelRegistryService` discovers model files from `EVONITH_MODEL_DIR` or the
read-only fallback `src/assets/models`. It lists model metadata without loading
files, validates model names, blocks path traversal, and loads only the selected
model on prediction. Cache size is bounded by `EVONITH_MODEL_CACHE_MAX_ITEMS`.

## Compute Jobs And Artifacts

`ComputeJobService` is an in-process registry suitable for Phase 7 edge-device
deployments and tests. `ComputeArtifactService` writes CSV/JSON artifacts under
`EVONITH_RUNTIME_DIR/compute/artifacts` and returns download URLs without
exposing internal paths.

## Auth And Authorization

When `EVONITH_COMPUTE_REQUIRE_AUTH=true`, compute endpoints require a valid
Phase 5 bearer token. Tests can disable compute auth with
`EVONITH_COMPUTE_REQUIRE_AUTH=false`.

## Edge-Device Safety Controls

- Models are lazy-loaded.
- Model cache size defaults to two items.
- JSON row counts and input sizes are capped.
- Large/exported results use artifacts.
- Compute worker count defaults to one.
- Optional model dependency failures are structured request errors, not startup
  failures.

## Deferred

- Full PostgreSQL-backed compute persistence.
- External worker queues such as Celery/Redis.
- AI Copilot, FurnaceMind, LLM recommendations, Qdrant memory.
- Full redesign of domain algorithms or pages.
