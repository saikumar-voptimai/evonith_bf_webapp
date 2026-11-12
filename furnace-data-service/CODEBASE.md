# Furnace Data Service

A small FastAPI sidecar that publishes BF2 data over REST: online telemetry from InfluxDB, offline reports from
Neon PostgreSQL, and the hourly ML dataset with an incremental, lag-aware cache. It is a thin HTTP layer over the
shared `furnace_data` package (`../furnace_data/`), which holds the fetchers, the dataset pipeline and the cache
logic; this service adds routes, background tasks and the cache directories.

Updated 10 Nov 2026. The webapp's own reference is `../CLAUDE.md` section 10; deployment is `../deploy/jetson/README.md`.

## Layout

```
furnace-data-service/
├── run.py                    Uvicorn launcher for development (PORT, default 8080; reload)
├── app/
│   ├── main.py               FastAPI app: loads ./.env, points FURNACE_CONFIG_DIR at ../src/config, mounts the routers
│   ├── config.py             pydantic-settings `Settings` (tokens, DATABASE_URL, data dirs, cache limits)
│   ├── core/
│   │   ├── online_fetcher.py InfluxDB online fetch (furnace_data.influx)
│   │   └── offline_fetcher.py Neon offline fetch (furnace_data.offline)
│   ├── models/schemas.py     Request and response models
│   ├── routes/
│   │   ├── health.py         GET /health
│   │   ├── data.py           /data/* (online, offline, raw-material live)
│   │   └── dataset.py        /dataset/* (background tasks, static cache)
│   └── tasks/task_manager.py Background task registry; result CSVs rotated (keep `max_result_files`)
├── data/
│   ├── results/              Per-task result CSVs (runtime, git-ignored)
│   └── static/               Static dataset cache: versioned CSVs + cache_meta.json (runtime, git-ignored)
├── test/                     pytest suite (uses temporary directories, no live services)
├── pyproject.toml, uv.lock   Own environment: `uv sync` here; depends on `furnace_data @ file:../furnace_data`
└── .env.example              Variables the service reads
```

## Run

```bash
cd furnace-data-service
uv sync
python run.py                          # http://127.0.0.1:8080, reload on
uv run pytest test/
```

On the Jetson the service is the systemd unit `bf-furnace-data` (`../deploy/jetson/systemd/`), run as
`python -m uvicorn app.main:app --host 127.0.0.1 --port 8000` from this directory with `/etc/bf-webapp/env`.

Settings (`app/config.py`, environment or `./.env`): `INFLUX_ONLINE_TOKEN`, `INFLUX_OFFLINE_TOKEN`, `DATABASE_URL`,
`HOST`, `PORT`, `RESULTS_DIR`, `STATIC_DIR`, `LEGACY_CSV_PATH` (bootstrap of the cache from an existing CSV),
`MAX_RESULT_FILES` (3), `OFFLINE_LAG_DAYS` (3), `STATIC_MAX_VERSIONS` (3). The YAML configuration is the webapp's
`src/config/setting_ds_dv.yml` through `FURNACE_CONFIG_DIR`.

## Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/data/online/measurements` | InfluxDB measurements and their fields |
| `POST` | `/data/online/fetch` | Online telemetry for a preset or explicit range, optional windowed average |
| `POST` | `/data/offline/fetch` | Offline data by report type (`HM_SLAG`, `CHARGE`, `DPR`, `RM_COMPOSITION`, `RAW_MATERIAL_STRENGTH`, `BURDEN_DISTRIBUTION`, `HOPPER_MANAGEMENT`) or explicit table; `ts`, `windowed-average` or `average`; JSON or CSV |
| `GET` | `/data/offline/report-types` | Report type → tables |
| `GET` | `/data/offline/tables` | The table whitelist with columns and aliases (`furnace_data.offline.list_offline_tables`) |
| `POST` | `/data/rm/live` | Latest raw-material composition (`RM_COMPOSITION`, last N days) |
| `POST` | `/dataset/fetch` | Build the ML dataset for a date range in the background → `task_id` |
| `POST` | `/dataset/update-static` | Incremental update of the static dataset cache in the background → `task_id` |
| `GET` | `/dataset/status/{task_id}` | Task state (optional `callback_url` on creation) |
| `GET` | `/dataset/download/{task_id}` | The task's CSV |
| `GET` | `/dataset/static` | The current static dataset CSV |
| `GET` | `/dataset/cache-info` | `cache_meta.json` of the static cache |

## Static dataset cache (lag-aware)

Offline reports are entered late: the hourly dataset of the last few days keeps changing after it is first built.
`furnace_data.dataset.static.StaticDatasetManager` therefore keeps two markers in `data/static/cache_meta.json`:

| Field | Meaning |
|---|---|
| `raw_end` | Last date fetched; rows up to here exist but the newest may still be incomplete |
| `confirmed_end` | `raw_end − offline_lag_days`; rows up to here are frozen and never fetched again |
| `offline_lag_days` | Days kept unconfirmed (default 3) |

Each `update-static` run keeps the CSV up to `confirmed_end`, re-fetches from `confirmed_end` to today (new data wins
on overlap), saves a new `furnace_dataset_YYYYMMDD_HHMMSS.csv`, updates the markers and keeps the three newest files.
Without a `cache_meta.json` the first run bootstraps from `LEGACY_CSV_PATH` when it is set (its last three days are
re-verified), else from `data_start`. The same cache logic serves the webapp's local copy
(`src/data/ml/static_dataset_manager.py`) and the Data Explorer's range cache (`furnace_data.dataset.fetcher`).
