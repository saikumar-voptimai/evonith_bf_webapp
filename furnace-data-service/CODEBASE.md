# Furnace Data API — Codebase Guide

A FastAPI service running on Raspberry Pi. It exposes three groups of endpoints:

1. **`/data/online`** — synchronous InfluxDB fetch for live furnace telemetry
2. **`/data/offline`** — synchronous InfluxDB fetch for manual/shift report data
3. **`/dataset`** — async ML dataset pipeline with lag-aware caching and CSV rotation

Port: **8080**

---

## Quick Start

```bash
# Install (Pi or local)
uv sync

# Run
python run.py

# Run tests
uv run --group dev pytest test/ --cov=app --cov-fail-under=80

# Check formatting
uv run --group dev black --check .

# Lint
uv run --group dev pylint app --fail-under=5.0
```

---

## Directory Structure

```
ml-dataset-api/
│
├── app/
│   ├── main.py               Entry point: creates FastAPI app, registers routers
│   ├── config.py             Pydantic BaseSettings — all env vars in one place
│   │
│   ├── core/                 Business logic — no FastAPI dependencies
│   │   ├── config_loader.py  Loads setting_ds_dv.yml
│   │   ├── influx_client.py  BaseDataFetcher + fetch_offline_data (ported from Streamlit app)
│   │   ├── online_fetcher.py fetch_online() — wraps BaseDataFetcher for 1–6 measurements
│   │   ├── offline_fetcher.py fetch_offline() — wraps fetch_offline_data for report types
│   │   ├── dataset_service.py MlDatasetService — 4-step multi-source fetch
│   │   ├── dataset_fetcher.py MlDatasetFetcher — range-aware in-memory cache
│   │   ├── data_cleaning.py  DataCleaner — 16-stage configurable cleaning pipeline
│   │   └── static_manager.py StaticDatasetManager — lag-aware CSV cache + rotation
│   │
│   ├── models/
│   │   └── schemas.py        All Pydantic request/response models
│   │
│   ├── routes/
│   │   ├── health.py         GET /health
│   │   ├── data.py           /data/online/* and /data/offline/* (sync)
│   │   └── dataset.py        /dataset/* (async background tasks)
│   │
│   └── tasks/
│       └── task_manager.py   In-memory task registry; runs jobs in daemon threads
│
├── config/
│   └── setting_ds_dv.yml     InfluxDB data mappings, measurement→field lists, rename dict
│
├── data/
│   ├── results/              Temp CSVs for ad-hoc /dataset/fetch downloads (auto-rotated)
│   └── static/
│       ├── cache_meta.json   Cache state: confirmed_end, raw_end, lag, rows
│       └── ml_dataset_*.csv  Versioned static ML dataset (max 3 kept)
│
├── test/                     Pytest test suite (no real InfluxDB/Postgres required)
├── run.py                    Uvicorn launcher
├── pyproject.toml
├── CODEBASE.md               This file
└── SERVICE_DESIGN.md         Architecture decisions and caching design rationale
```

---

## Module Reference

### `app/config.py`
Pydantic `BaseSettings` singleton. Reads from `.env` or environment variables.
All other modules import `from app.config import settings`.

Key settings:
| Variable | Default | Purpose |
|---|---|---|
| `INFLUX_ONLINE_TOKEN` | — | InfluxDB online bucket token |
| `INFLUX_OFFLINE_TOKEN` | — | InfluxDB offline bucket token |
| `DATABASE_URL` | — | PostgreSQL (Neon) connection string |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | Uvicorn bind |
| `OFFLINE_LAG_DAYS` | `3` | Days to keep "unconfirmed" in cache |
| `STATIC_MAX_VERSIONS` | `3` | Max versioned CSVs to keep |
| `LEGACY_CSV_PATH` | `""` | Path to a pre-existing CSV to bootstrap from |

---

### `app/core/influx_client.py`
Ported from `src/data_fetchers/base_data_fetcher.py` in the Streamlit app.

- `BaseDataFetcher` — low-level InfluxDB client. Builds InfluxQL queries and returns raw DataFrames.
- `fetch_offline_data(measurement, time_range, database)` — convenience wrapper for offline bucket queries.
- `query_builder()` — constructs InfluxQL for `ts`, `windowed-average`, `average`, `avg-min-max` query types.

---

### `app/core/online_fetcher.py`
- `fetch_online(measurements, query_type, window, start_time, end_time, preset)` — fetches one or more measurements, drops `iox::measurement` metadata columns, outer-joins on time index, normalises to IST tz-naive.
- `list_measurements()` — reads `data_mapping` from YAML, returns field lists per measurement.

---

### `app/core/offline_fetcher.py`
- `fetch_offline(report_type, start_time, end_time, preset)` — maps user-facing `report_type` (`HM_SLAG`, `CHARGE`, `RM_COMPOSITION`, `DPR`) to InfluxDB measurement names and delegates to `fetch_offline_data`.

---

### `app/core/dataset_service.py`
Ported from `src/ml_pipeline/ml_dataset_service.py`. Four fetch steps:

| Step | Source | When used |
|---|---|---|
| `fetch_step1` | InfluxDB legacy `rm_charge_dis_hm_slag` | Before cutoff date (2025-08-24) |
| `fetch_step2` | InfluxDB `rm_charge_dataset` or `rm_dpr_data` | After cutoff |
| `fetch_hotmetal_hourly` | InfluxDB `hotmetal_slag_updated_data` | After cutoff; time-interpolated |
| `fetch_distribution_data` | PostgreSQL `burden_distribution_history` (SCD Type-2) | After cutoff; daily granularity |

---

### `app/core/dataset_fetcher.py`
`MlDatasetFetcher` with in-memory `RangeCache`:
- On cache hit (same rm_mode, requested range ⊆ cached range): slices from memory.
- On partial overlap: fetches only the missing slice, concatenates.
- Handles the pre/post cutoff boundary by fetching both halves and concatenating.

---

### `app/core/data_cleaning.py`
`DataCleaner` with a 16-stage configurable pipeline:
1. Time index normalisation (floor to hour, dedup by mean)
2. Column uppercasing
3. Schema enforcement (keep only configured columns)
4. `_MT` → `_CALC_MT` rename
5. Drop unnamed columns
6. Fill default zeros (STEAM, FLUX, PELLET, SINTER sources)
7. Combine SINTER_SP_01 + SP_02 → SINTER_CALC_MT
8. Drop sparse rows (< 50% non-NaN)
9. Drop unnecessary columns
10. Add `UNITCOST LAKHS/THM` feature
11. Drop high-NaN columns (> 30%)
12. Cruising filters (production, PCI, ETA CO, etc.)
13. Outlier rules (NaN values outside bounds)
14. Selective MICE imputation (RAFT, DP, O₂, production)
15. Tonnage sanity caps
16. Final median/most-frequent imputation

`build_default_config()` returns the default `CleaningConfig` matching the production notebook.

---

### `app/core/static_manager.py`
The core caching logic. See `SERVICE_DESIGN.md` for full rationale.

**Key concept — confirmed vs raw end:**
```
confirmed_end = raw_end - offline_lag_days
```
Rows up to `confirmed_end` are frozen (offline data is stable by then). Rows from `confirmed_end` to `raw_end` are re-fetched on every run to pick up delayed offline data (RM composition, HM/slag typically arrive 2–3 days late).

**On each `update_static()` call:**
1. Load `cache_meta.json` → find active CSV
2. Compute `confirmed_end`; freeze rows up to there
3. Re-fetch `confirmed_end → today`
4. Merge (new wins on overlap), clean, save new versioned CSV
5. Rotate old CSVs (keep `STATIC_MAX_VERSIONS`)
6. Write updated `cache_meta.json`

**`CacheMeta`** (dataclass, JSON-serialisable):
```
data_start, confirmed_end, raw_end — ISO date strings
offline_lag_days — int
rows, columns, csv_file, last_updated, rm_choice
```

---

### `app/tasks/task_manager.py`
In-memory task registry backed by a `dict` + `threading.Lock`.
- Each task runs in a daemon thread (`threading.Thread`).
- States: `pending → running → completed | failed`.
- On completion, result DataFrame is saved as a timestamped CSV in `data/results/`.
- Result files are auto-rotated (keep `max_result_files`, default 3).
- Optional `callback_url`: on completion, a `POST` with the task status JSON is sent.

---

## Data Flow

```
                        ┌─────────────────────────────────────────┐
 POST /data/online/fetch│  online_fetcher.fetch_online()          │
  (sync, returns JSON   │  → BaseDataFetcher per measurement      │
   or CSV stream)       │  → outer-join on time index → IST norm  │
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
POST /data/offline/fetch│  offline_fetcher.fetch_offline()        │
  (sync)                │  → fetch_offline_data()                 │
                        │  → BaseDataFetcher(offline bucket)      │
                        └─────────────────────────────────────────┘

POST /dataset/fetch      ┌──────────────────────────────────────────────────┐
  (async)                │  MlDatasetFetcher.get_ml_dataset()               │
                         │  ├─ before cutoff: fetch_step1()                 │
                         │  └─ after cutoff:                                │
                         │       fetch_step2() + fetch_hotmetal_hourly()    │
                         │       + fetch_distribution_data() (PostgreSQL)   │
                         │  → rename columns (RENAME_DICT)                  │
                         │  → [optional] DataCleaner.clean()                │
                         │  → save timestamped CSV → TaskManager            │
                         └──────────────────────────────────────────────────┘

POST /dataset/update-static
  (async)                ┌──────────────────────────────────────────────────┐
                         │  StaticDatasetManager.update_static()            │
                         │  ├─ load cache_meta.json + active CSV            │
                         │  ├─ compute confirmed_end = raw_end - lag_days   │
                         │  ├─ freeze rows ≤ confirmed_end                  │
                         │  ├─ re-fetch confirmed_end → today               │
                         │  ├─ merge + clean + save versioned CSV           │
                         │  └─ rotate (keep max 3), update cache_meta.json  │
                         └──────────────────────────────────────────────────┘
```

---

## Endpoint Reference

| Method | Path | Sync/Async | Description |
|---|---|---|---|
| GET | `/health` | sync | Liveness check |
| GET | `/data/online/measurements` | sync | List measurements + fields |
| POST | `/data/online/fetch` | sync | Fetch telemetry (JSON or CSV) |
| POST | `/data/offline/fetch` | sync | Fetch shift reports (JSON or CSV) |
| GET | `/data/offline/report-types` | sync | List report type → measurement mapping |
| POST | `/dataset/fetch` | async | Ad-hoc ML dataset for a date range |
| POST | `/dataset/update-static` | async | Smart incremental static CSV update |
| GET | `/dataset/status/{id}` | sync | Poll task progress |
| GET | `/dataset/download/{id}` | sync | Download completed task result CSV |
| GET | `/dataset/static` | sync | Download current static ML CSV |
| GET | `/dataset/cache-info` | sync | Inspect cache_meta.json state |

Interactive docs: `http://<pi-ip>:8080/docs`

---

## Adding a New Data Fetcher

1. Add measurement config to `config/setting_ds_dv.yml` under `data_mapping`.
2. If online bucket: add to `ONLINE_MEASUREMENTS` in `online_fetcher.py`.
3. If offline bucket: add to `OFFLINE_REPORT_MAP` in `offline_fetcher.py`.
4. Add enum value to `OfflineReportType` or extend `OnlineFetchRequest.measurements` validation in `schemas.py`.
5. No route changes needed — existing validation is dynamic.

---

## Running Tests

```bash
# All tests with coverage
uv run --group dev pytest test/ -v --cov=app --cov-report=term-missing

# Single file
uv run --group dev pytest test/test_static_manager.py -v

# Coverage threshold (CI gate)
uv run --group dev pytest test/ --cov=app --cov-fail-under=80
```

Tests mock all external I/O — no InfluxDB or PostgreSQL credentials needed.

---

## Deployment on Pi

See `SERVICE_DESIGN.md` for the full systemd unit template.

```bash
# One-time setup
sudo systemctl enable furnace-api
sudo systemctl start furnace-api

# After a new deployment (CI does this automatically)
sudo systemctl restart furnace-api

# Logs
journalctl -u furnace-api -f
```

The CI/CD pipeline (`deploy.yml`) deploys on every merge to `release`:
1. Runs tests + lint + formatting check
2. Archives the project (excluding venv, cache, data)
3. SCPs the archive to the Pi
4. Extracts, creates venv, updates symlink
5. Restarts the systemd service
