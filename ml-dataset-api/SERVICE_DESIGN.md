# Furnace Data API — Service Design

## Overview

FastAPI service running on Raspberry Pi. Two concerns:

1. **`/data/online` and `/data/offline`** — lightweight synchronous InfluxDB fetch endpoints for external consumers (scripts, scheduled jobs, etc.). Not intended to proxy the Streamlit app.
2. **`/dataset`** — ML dataset pipeline: multi-source fetch, optional cleaning, smart cache, CSV rotation.

Port: **8080**

---

## Route Map

```
GET  /health

POST /data/online/fetch              sync — returns JSON or CSV
GET  /data/online/measurements       static — lists available measurements + fields

POST /data/offline/fetch             sync — returns JSON or CSV

POST /dataset/fetch                  async — ad-hoc ML dataset for a date range
GET  /dataset/status/{task_id}       poll task progress
GET  /dataset/download/{task_id}     download result CSV
POST /dataset/update-static          async — smart incremental update of static CSV
GET  /dataset/static                 download current static CSV
GET  /dataset/cache-info             inspect cache metadata (dates, lag, rows, file)
```

---

## ML Dataset Caching Design

### The Problem

Offline data (RM composition, HM/slag, charge reports) is **manually entered and delayed**.
Data from day X typically arrives 2–3 days later in InfluxDB (sometimes up to 5 days for RM).

This means: if the last static CSV row is `2026-03-28`, the data for Mar 26–28 may still be
incomplete — offline fields are NaN or were not yet entered when the last fetch ran.

The current naive design uses `last_date_in_csv` as the fetch start, which means:
- It fetches from Mar 28 onward → fine for new data
- But it never re-checks Mar 26–28 to fill in the offline data that arrived later

### Solution: Confirmed End vs Raw End

Two date markers are tracked in `cache_meta.json`:

| Field | Meaning |
|-------|---------|
| `raw_end` | Last date we fetched data for. Data exists but may be partially incomplete (offline still arriving). |
| `confirmed_end` | `raw_end - offline_lag_days`. Data up to here is considered stable — we won't re-fetch it. |
| `offline_lag_days` | How many days to keep "unconfirmed". Default: **3**. Configurable. |

### Fetch Strategy on Each `update-static` Call

```
1. Read existing CSV + cache_meta.json
2. confirmed_end = raw_end - offline_lag_days
3. Keep CSV rows up to confirmed_end (frozen — won't change)
4. Re-fetch from confirmed_end to today  ← the "uncertain window"
   (new offline data may have arrived for these days since last run)
5. Merge: new data wins on overlap
6. Update meta:
     confirmed_end = today - offline_lag_days
     raw_end = today
7. Save new CSV, rotate old files (keep 3 latest)
```

### Example Timeline

```
Today = Apr 1

Previous run (Mar 31):
  raw_end = Mar 31
  confirmed_end = Mar 28   (lag = 3 days)
  CSV contains rows through Mar 31

New run (Apr 1):
  Re-fetch: Mar 28 → Apr 1    ← picks up any offline data that arrived for Mar 28–31
  Keep frozen: up to Mar 28
  New confirmed_end = Mar 29   (Apr 1 - 3)
  New raw_end = Apr 1
```

### Legacy CSV Bootstrap

On first run, if no `cache_meta.json` exists:
1. Check `settings.legacy_csv_path` (points to `V13_df_filtered.csv` or similar)
2. If found, read it as the initial cache
3. Extract its `max_date` → treat as `raw_end` of the legacy file
4. Set `confirmed_end = raw_end - lag_days`
5. Fetch from `confirmed_end` onward to fill the uncertain window and all new data
6. Merge and save as the first versioned CSV

This means if you have data from 2023–Feb 2, only the last 3 days of that file are
re-verified, and then incremental fetches continue from there.

### Cache Metadata File: `data/static/cache_meta.json`

```json
{
  "version": 1,
  "rm_choice": "charge",
  "data_start": "2023-01-01",
  "confirmed_end": "2026-03-28",
  "raw_end": "2026-03-31",
  "last_updated": "2026-03-31T09:00:00",
  "offline_lag_days": 3,
  "rows": 25000,
  "columns": 120,
  "csv_file": "ml_dataset_20260331_090000.csv"
}
```

### CSV Rotation

- Files saved as: `ml_dataset_YYYYMMDD_HHMMSS.csv`
- After each save, keep only the **3 most recent** files; delete older ones
- `cache_meta.json` always points to the current active file via `csv_file` field

---

## Online/Offline Data Endpoints

### `POST /data/online/fetch`

Synchronous. Wraps `BaseDataFetcher` for 1–6 measurements, outer-joins results.

```json
{
  "measurements": ["process_params", "heatload_delta_t"],
  "start_time": "2025-10-01T00:00:00",   // provide start+end OR preset
  "end_time":   "2025-10-02T00:00:00",
  "preset":     "last 8 hours",           // overrides start/end if provided
  "query_type": "windowed-average",       // "ts" | "windowed-average" | "average" | "avg-min-max"
  "window":     "1h",                     // "15m" | "30m" | "1h" | "6h" | "1d"
  "format":     "json"                    // "json" | "csv"
}
```

Response (`json`):
```json
{
  "meta": {
    "measurements": ["process_params"],
    "query_type": "windowed-average",
    "window": "1h",
    "start": "...", "end": "...",
    "rows": 24, "columns": 30
  },
  "columns": ["hot_blast_vol_nm3h", ...],
  "data": [{"time": "2025-10-01T00:00:00", "hot_blast_vol_nm3h": 95000, ...}, ...]
}
```

Response (`csv`): streamed directly, no file written to disk.

### `POST /data/offline/fetch`

Same pattern, maps `report_type` → InfluxDB measurement.

```json
{
  "report_type": "HM_SLAG",              // "HM_SLAG" | "CHARGE" | "RM_COMPOSITION" | "DPR"
  "start_time":  "2025-10-01T00:00:00",
  "end_time":    "2025-10-10T00:00:00",
  "preset":      "last 1 month",
  "format":      "json"
}
```

### `GET /data/online/measurements`

Returns YAML config's `data_mapping` keys and field lists. No InfluxDB call.

---

## Why NOT route Streamlit through this API

- Pi CPU/RAM becomes the bottleneck for large DataFrames (e.g. 110-col temperature_profile,
  30-day raw ts). InfluxDB can stream directly to the Streamlit host faster.
- If the Pi goes down, Streamlit would break too.
- Token management: both machines already have .env. Centralising on Pi is a phase-2 decision.

The `/data/online` endpoint exists for: external scripts, scheduled jobs, future mobile
dashboards, and the ML pipeline needing a fresh online snapshot. Not for proxying Streamlit.

---

## Deployment (Pi)

```bash
cd ~/furnace-api
uv sync
python run.py          # starts on 0.0.0.0:8080

# Or as a systemd service:
# ExecStart=/home/pi/.local/bin/uv run python run.py
```

### Systemd unit: `/etc/systemd/system/furnace-api.service`

```ini
[Unit]
Description=Furnace Data API
After=network.target

[Service]
WorkingDirectory=/home/pi/furnace-api
ExecStart=/home/pi/.local/bin/uv run python run.py
Restart=on-failure
EnvironmentFile=/home/pi/furnace-api/.env

[Install]
WantedBy=multi-user.target
```

---

## Environment Variables

```
# InfluxDB
INFLUX_ONLINE_TOKEN=...
INFLUX_OFFLINE_TOKEN=...

# PostgreSQL (Neon)
DATABASE_URL=postgresql://...

# API settings
API_PORT=8080
OFFLINE_LAG_DAYS=3                # days to keep "unconfirmed" in cache
STATIC_MAX_VERSIONS=3             # number of CSV versions to keep
LEGACY_CSV_PATH=/path/to/V13_df_filtered.csv   # optional bootstrap file
```

---

## File Layout

```
ml-dataset-api/
├── app/
│   ├── core/
│   │   ├── config_loader.py
│   │   ├── influx_client.py       # BaseDataFetcher + fetch_offline_data
│   │   ├── online_fetcher.py      # NEW: wraps BaseDataFetcher for /data/online
│   │   ├── offline_fetcher.py     # NEW: wraps fetch_offline_data for /data/offline
│   │   ├── dataset_service.py     # MlDatasetService (4-step fetch)
│   │   ├── dataset_fetcher.py     # MlDatasetFetcher (RangeCache)
│   │   ├── data_cleaning.py       # DataCleaner
│   │   └── static_manager.py      # StaticDatasetManager (UPDATED with lag logic)
│   ├── models/
│   │   └── schemas.py             # All Pydantic request/response models
│   ├── routes/
│   │   ├── health.py
│   │   ├── data.py                # NEW: /data/online, /data/offline
│   │   └── dataset.py             # /dataset/* (UPDATED with cache-info endpoint)
│   ├── tasks/
│   │   └── task_manager.py
│   ├── config.py                  # Pydantic BaseSettings
│   └── main.py
├── config/
│   └── setting_ds_dv.yml
├── data/
│   ├── results/                   # Task result CSVs (temp, auto-cleaned)
│   └── static/
│       ├── cache_meta.json        # Cache metadata
│       └── ml_dataset_*.csv       # Versioned static CSVs (max 3)
├── SERVICE_DESIGN.md              # This file
├── pyproject.toml
├── run.py
└── .env
```
