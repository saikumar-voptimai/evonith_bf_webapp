# Phase 4 Data and Dataset API

Phase 4 moves data and dataset access behind versioned backend APIs while
keeping the Streamlit direct-mode paths available.

## Audit

| Area | Current flow | Phase 4 action |
| --- | --- | --- |
| Online data | Data Explorer imports `furnace_data.influx.online.fetch_online_df` directly | Added `/api/v1/data/preview` support for `source=online`; Data Explorer uses it only when `USE_BACKEND_API_DATA_EXPLORER=true` |
| Offline data | Data Explorer imports `furnace_data.offline` report/table fetchers directly | Added `/api/v1/data/preview` support for `source=offline`; direct imports remain for fallback |
| Static dataset | Streamlit reads the runtime/static CSV through `data.ml.static_csv` | Added `/api/v1/datasets` and `/api/v1/datasets/{dataset_id}/preview` |
| Dataset refresh/job | Streamlit refreshes the static dataset locally through `utils.dataset_refresher` and page actions | Added `/api/v1/datasets/refresh` and job status/download endpoints |
| Runtime writes | Phase 1 routes generated outputs to `EVONITH_RUNTIME_DIR` | Artifacts are written under `runtime/datasets/results/artifacts/` |
| Legacy backend routes | `/data/...` and `/dataset/...` still exist | Retained by default through Phase 2 legacy route wiring |
| Existing v1 routes | Phase 2 exposed legacy data routes under `/api/v1/data` and thin dataset wrappers under `/api/v1/datasets` | Added stable service-backed data/dataset endpoints and preserved v1 dataset compatibility wrappers |
| Data Explorer direct imports | The page still imports direct data/domain helpers | Expected during migration; only selected fetch actions are guarded by API flags |

## What Changed

- Added stable Pydantic schemas for data and dataset APIs.
- Added backend serialization, artifact, job, data, and dataset services.
- Added capped data preview and artifact export endpoints.
- Added dataset list, preview, refresh job, job status, and artifact download endpoints.
- Added frontend `data_api.py` and `dataset_api.py` adapters.
- Added guarded Data Explorer API mode for online/offline previews.
- Added guarded backend dataset refresh mode.

## What Did Not Change

- No authentication migration.
- No Feedback, Material Balance, Recommendations, Blend Optimizer, AI Copilot, or FurnaceMind migration.
- No database schema change.
- No full Streamlit rewrite.
- Direct-mode Data Explorer and dataset refresh paths remain available.
- Legacy backend routes remain available by default.

## New Backend Endpoints

- `GET /api/v1/data/sources`
- `GET /api/v1/data/offline/report-types`
- `GET /api/v1/data/offline/tables`
- `POST /api/v1/data/preview`
- `POST /api/v1/data/export`
- `GET /api/v1/data/artifacts/{artifact_id}/download`
- `GET /api/v1/datasets`
- `GET /api/v1/datasets/{dataset_id}/preview`
- `POST /api/v1/datasets/refresh`
- `GET /api/v1/datasets/jobs/{job_id}`
- `GET /api/v1/datasets/jobs/{job_id}/download`
- `GET /api/v1/datasets/artifacts/{artifact_id}/download`

## Frontend Adapters

- `src/services/data_api.py`
- `src/services/dataset_api.py`

These adapters use `src/services/api_client.py` and do not import backend
internals, database clients, or InfluxDB clients.

## Feature Flags

```bash
USE_BACKEND_API=false
USE_BACKEND_API_DATA_EXPLORER=false
USE_BACKEND_API_DATASETS=false
```

When flags are false, direct mode is used. When enabled, supported Data Explorer
fetches and dataset refresh actions use backend API adapters.

## Data Size Limits

```bash
DATA_API_MAX_PREVIEW_ROWS=500
DATA_API_MAX_JSON_ROWS=5000
DATA_API_EXPORT_FORMAT=csv
DATA_API_JOB_TTL_HOURS=24
DATA_API_ARTIFACT_TTL_HOURS=24
```

Preview endpoints cap returned rows. Export endpoints write CSV artifacts
instead of returning large JSON payloads.

## Artifact And Job Behavior

CSV artifacts are stored under:

```text
runtime/datasets/results/artifacts/
```

Artifact IDs are UUID-style hex strings and filenames are sanitized. Download
routes resolve artifacts by ID and do not expose internal filesystem paths.

Dataset jobs are currently in-process. State is lost on backend process restart;
this is documented and should be improved in a later worker/persistence phase.

## Legacy Routes

Legacy `/data/...` and `/dataset/...` routes are retained when
`EVONITH_ENABLE_LEGACY_ROUTES=true`. Old response shapes are preserved there.

## Local Development

Backend:

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Frontend direct mode:

```bash
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 \
USE_BACKEND_API=false \
USE_BACKEND_API_DATA_EXPLORER=false \
USE_BACKEND_API_DATASETS=false \
streamlit run src/app.py
```

Frontend API mode:

```bash
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 \
USE_BACKEND_API=true \
USE_BACKEND_API_DATA_EXPLORER=true \
USE_BACKEND_API_DATASETS=true \
streamlit run src/app.py
```

Smoke checks:

```bash
curl http://localhost:8080/api/v1/health
curl http://localhost:8080/api/v1/data/sources
curl http://localhost:8080/api/v1/datasets
```

## Tests

```bash
pytest furnace-data-service/tests -q
pytest tests/frontend -q
pytest tests -k "data_api or dataset_api or api_v1_data or api_v1_datasets or artifact_service or job_service" -q
python scripts/export_backend_openapi.py
```

Detailed results are in:

```text
docs/migration/phase-4-test-report.md
```

## Known Limitations

- Data Explorer API mode currently covers online/offline preview fetch actions,
  not every direct-mode operation on the page.
- Dataset refresh jobs are in-process and not persisted.
- External source availability is still endpoint-specific.
- Large export UX is available as artifact metadata/download URL; richer browser
  download integration can be improved in Phase 5.

## Phase 5 Follow-Up

- Persist job state or introduce a lightweight worker model.
- Expand Data Explorer API mode to more operations after payload-size review.
- Add richer frontend polling for dataset refresh jobs.
- Consider hiding legacy routes from OpenAPI before eventual deprecation.
