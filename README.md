# Evonith BF2 — Blast Furnace Intelligence Platform

Monitoring, modelling and optimisation of **Blast Furnace 2 (BF2)** at Evonith Steel: live telemetry from InfluxDB,
manual plant reports from Neon PostgreSQL, a two-zone energy model of the furnace, blend and fuel optimisers, a
conversational assistant (FurnaceMind) and a live digital twin evaluated every hour.

The platform has three deployable parts that share one Python service layer, plus the shared data library:

| Part | Path | What it is |
|---|---|---|
| **Streamlit app** (legacy) | `src/app.py`, `src/custom_pages/` | The original 11-page dashboard; still the home of the model pages. Served under `/legacy` in production |
| **twin-api** | `src/api/` | FastAPI behind the SPA: sign-in, jobs, event stream, twin, optimiser, reports, FurnaceMind, admin |
| **twin-worker** | `src/worker/` | arq worker on Redis: scheduled and on-demand model runs (twin evaluation, optimiser, calibration, forecasts, reports) |
| **SPA** | `frontend/` | SvelteKit 3 / Svelte 5 / TypeScript single-page app that is replacing the Streamlit pages one by one |
| **Service layer** | `src/services/` | Streamlit-free data assembly and model orchestration shared by the API, the worker and the Streamlit pages |
| `furnace_data/` | editable package | Shared data access: InfluxDB fetchers, Neon offline tables, relational models, dataset pipeline |
| `furnace-data-service/` | FastAPI sidecar | Publishes the hourly ML dataset and offline reports over REST (port 8000 on the plant host) |

`CLAUDE.md` is the detailed reference (furnace basics, data schema, every model and page, the API and worker, the
conventions); this README is the orientation.

---

## Contents

1. [Architecture](#1-architecture)
2. [Quick start](#2-quick-start)
3. [Repository layout](#3-repository-layout)
4. [Pages](#4-pages)
5. [Models](#5-models)
6. [Data sources](#6-data-sources)
7. [Configuration](#7-configuration)
8. [Environment variables](#8-environment-variables)
9. [Tests and checks](#9-tests-and-checks)
10. [Deployment](#10-deployment)
11. [Documentation](#11-documentation)
12. [Domain notes](#12-domain-notes)

---

## 1. Architecture

```
 Browser ──HTTPS──▶ Caddy (one origin)
                     ├── /            SPA build (frontend/build, 200.html fallback)
                     ├── /api/*       twin-api  (uvicorn, :8100)  ──┐
                     └── /legacy/*    Streamlit (:8501)            │   src/services/  (shared, no Streamlit)
                                                                   │        │
                     twin-worker (python -m worker) ◀── Redis ─────┘        │
                       scheduled jobs, SSE events (stream twin:events)      │
                                                                            ▼
        InfluxDB (AWS)          Neon PostgreSQL                 Qdrant            Open-Meteo
        bf2_evonith_raw         offline_feed, plant_master,     FurnaceMind       blast moisture
        online telemetry        ops_config, identity, twin      memory            (Wardha RH)
```

- Every Python process runs with `src/` on the path and `src/` as the working directory: imports are
  `from services.x`, `from utils.x`, never `from src.x`.
- Routers in `src/api/` and job handlers in `src/worker/kinds/` call `services.*` only. Streamlit pages wrap the
  same service building blocks in `st.cache_data` through `src/ui/*/loaders.py`
  (`tests/test_services_architecture.py` fails on a Streamlit import under `services/`).
- The worker publishes job progress and model events to the Redis stream `twin:events`; the API forwards them to the
  browser as server-sent events on `GET /api/stream`.
- Persistent results of the API and worker live in the Neon schema `twin` (`src/data/twin/`, migrations
  `alembic/twin/`).

## 2. Quick start

Prerequisites: Python 3.12 (uv installs it), [uv](https://docs.astral.sh/uv/) 0.7.x, Node.js ≥ 20 with pnpm, a
`.env` at the repo root (section 8). Redis is optional in development: without it the worker is not started and the
job routes answer 503.

```bash
uv sync                          # Python environment, furnace_data/ installed editable
pnpm --dir frontend install      # SPA

python scripts/dev.py            # SPA :5173 + twin-api :8100 + worker (if Redis answers) + Streamlit /legacy :8501
python scripts/dev.py api web    # a subset: api | web | worker | legacy

python run_streamlit.py          # Streamlit alone (options pass through, e.g. --server.baseUrlPath legacy)
```

Open http://localhost:5173 (the dev server proxies `/api` and `/legacy`, so the browser sees one origin as in
production). Sign in with an account from Neon `identity.users`; roles are `user` < `supervisor` < `admin`.

**Never run `streamlit run` directly.** `run_streamlit.py` imports torch first on Windows (DLL load order), runs
`src/app.py` from the repository root whatever the current directory, and passes its arguments to Streamlit.

Other everyday commands (`make help` lists them; on Windows run the recipes by hand):

```bash
make lint          # Prettier + ESLint; Ruff 0.17.2 on the programme's Python (make lint-py)
make check         # svelte-check
make test          # Vitest + pytest with BURDEN_ENGINE_FAST=1
make client        # regenerate the typed API client after changing a route (commit openapi.json + schema.d.ts)
make fixtures      # chart and legacy-page fixtures from the Python models (commit the JSON)
make db-upgrade    # alembic -n twin upgrade head  (BF_TWIN_DB_URL, else DATABASE_URL)
make lock-check    # uv lock --check + resolution for the Jetson (aarch64, CPython 3.12)
```

Lock changes: `uvx --from "uv==0.7.13" uv lock`, then `uv sync --frozen` (uv 0.11 rewrites the lock format).

## 3. Repository layout

```
evonith_webapp/
├── run_streamlit.py         Streamlit entry point (use this, not `streamlit run`)
├── pyproject.toml, uv.lock  Python project (uv); furnace_data/ as an editable path dependency
├── Makefile                 dev, lint, test, client, fixtures, db-*, audit-ui, lock-check
├── CLAUDE.md                Detailed reference of the system
├── alembic/                 twin/ = the `twin` schema migrations (-n twin); versions/ = the relational tables
├── deploy/jetson/           Provisioning, systemd units, Caddyfile, release and rollback scripts
├── docs/                    Model notes, decision records, benchmarks (section 11)
├── frontend/                SvelteKit SPA (DESIGN.md: design system and chart rules)
├── furnace_data/            Shared data-access package
├── furnace-data-service/    Dataset REST sidecar (own pyproject and tests)
├── scripts/                 dev.py, CI step wrapper, exporters, audits, model studies
├── tests/                   pytest suite (174 files) and fixtures
└── src/
    ├── app.py               Streamlit navigation and login gate
    ├── api/                 twin-api: main, auth, settings, routers/ (24 routers), security/
    ├── worker/              twin-worker: runtime, settings, kinds/ (job handlers)
    ├── services/            Service layer (45 modules): twin, optimiser, bmo, reports, status, facts, ...
    ├── custom_pages/        The 11 Streamlit pages
    ├── ui/                  Streamlit UI: per-page packages (bmo, burden_optimiser, coke_energy, furnace_status, furnacemind), admin pages, login
    ├── utils/               Models and helpers: burden_engine, coke_energy, bmo, energy_balance, strata_flow (vendored), material_balance, recommendations, settings, memo
    ├── data/                Adapters: fetchers (InfluxDB), ml (static dataset), bmo, burden_engine, coke_energy, furnace_status, tickets, twin
    ├── domain/              Pure domain logic: auth, plant config, bmo, furnace_status, optimization_runtime
    ├── agents/              FurnaceMind: agent loop, tools, skills, memory, embeddings, LLM clients, multimodal ingestion
    ├── reports/             Shift report builder (furnace_report/)
    ├── plotters/            Plotly contour and material-balance figures
    ├── geometries/          Furnace profile and sensor masks
    ├── config/              YAML configuration, page_registry.py, facts_register.json, schemas/
    ├── assets/              Models, calibrations, CSS, static data and notes
    └── storage/             Runtime data (mostly git-ignored): ledgers, caches, snapshots, tickets
```

## 4. Pages

### Streamlit (`src/custom_pages/`, registry `src/config/page_registry.py`)

| Title | File | What it does |
|---|---|---|
| Welcome | `1_Welcome.py` | Landing tiles to every page; admin tools (users, hoppers, burden) |
| Data Explorer | `2_Data_Explorer.py` | Browse InfluxDB measurements and Neon offline tables; build and download the ML dataset |
| V-Board | `3_Data_Visualisation.py` | Circumferential and longitudinal temperature contours, stave heat loads |
| V-Sense | `4_Recommendations.py` | Blast-parameter optimiser: differential evolution on pre-trained models for ηCO, production and unit cost |
| CoPilot | `5_AI_Copilot.py` | Unit-cost drivers, burden-distribution findings, channeling propensity and anomaly review with an LLM |
| Material Balance | `6_Material_Balance.py` | Per-element daily mass balance (Sankey, bars, closure table) |
| FurnaceMind | `7_FurnaceMind.py` | Tool-calling assistant over live, offline and twin data; skills; shift reports tab |
| Blend Mix Optimiser | `9_Blend_Optimizer.py` | Ore + flux LP blend at a fixed production with the energy-balance coke rate, Si and charged-coke forecasts |
| Coke Energy Model | `10_Coke_Energy.py` | Two-zone (Rist) energy balance per shift: coke burnt vs charged vs modelled; calibration |
| Burden & Fuel Optimiser | `11_Burden_Fuel_Optimiser.py` | Ore + flux + fuel blend and production level on the energy model, Fixed-ω or STRATA-ω |
| Feedback | `8_Feedback.py` | Ticket board (SQLite by default) |

### SPA (`frontend/src/routes/`, navigation `frontend/src/lib/nav.ts`)

| Route | Role | Page |
|---|---|---|
| `/status` (landing) | user | Furnace status: V-Board snapshot every 10 s from the worker, zone status, trends |
| `/reports` | user | Shift reports with KPI checks, handover record, print layout |
| `/furnacemind` | user | FurnaceMind chat over the API (SSE, Plotly artifacts, skills, feedback) |
| `/bmo` | user | Blend Mix Optimiser: plan board, constraints, Step 2 Fixed-ω runs, accept plan |
| `/optimiser` | user (runs: supervisor) | Blend Simulator — the Burden & Fuel Optimiser through `POST /api/optimiser/runs` |
| `/models` | user | Digital twin models (placeholder until SV-7.6) |
| `/admin/*` | admin | Fact register, what-if, burnt vs charged, validation (STRATA-Learn), calibration, users & hoppers, jobs & health, audit log; model views Live Twin, TRACE, FORGE, STRATA |
| `/dev/components` | — | Component catalogue |

Streamlit pages are linked from the SPA by title (`legacyHref`, generated from the page registry). Retirement of a
Streamlit page follows `docs/retirement.md`.

## 5. Models

| Model | Where | One line |
|---|---|---|
| Coke Energy Model | `utils/coke_energy/`, page 10, `services/coke_energy.py` | Two-zone (Rist) energy and oxygen balance per 8-h shift; burnt coke from the blast, charged coke from the charge record; calibration on the last 14 days of stable shifts; blast moisture from Open-Meteo |
| Burden engine | `utils/burden_engine/`, page 11, `services/optimiser.py` | Sequential LP (HiGHS) on the two-zone balance: cheapest production level, fixed production, maximum production; prices and slag credit; fact register `config/facts_register.json` |
| STRATA-ω | `utils/burden_engine/omega_model.py`, `assets/burden_engine/omega_model.json` | Blend-dependent shaft efficiency from the STRATA-Flow network (`utils/strata_flow/`, vendored); solver processes; STRATA-Learn shadow mode in `learn/` |
| Live twin | `services/twin.py`, `utils/coke_energy/twin.py` | Window [t − 8 h, t) every hour: K_TRACE, K_FORGE, K_STRATA, ω_imp, ω_STRATA; drift alarm; burden transit; gated HMT forecast |
| Blend Mix Optimiser | `utils/bmo/`, `services/bmo*.py`, page 9 | Ore + flux LP at fixed production; energy-balance coke correction; charged-coke and HM Si forecasts (ledgers in `src/storage/bmo_*`) |
| V-Sense | `utils/recommendations/`, page 4 | Differential evolution over 7 blast controls on XGBoost/sklearn models (`assets/models/`) |
| Material balance | `utils/material_balance/`, page 6 | 12-element daily balance from raw materials, blast, steam, hot metal, slag and top gas |
| FurnaceMind | `agents/`, `services/furnacemind.py`, page 7 | OpenRouter tool-calling loop with 13 tools (data fetch, merge, plot, Qdrant search, twin reads) and pre-computed skills |
| Shift reports | `reports/furnace_report/`, `services/reports.py` | KPI checks against `config/shift_report.yml`, LLM narrative, stored in `twin.shift_report` |

Details, formulas and decisions: `CLAUDE.md` sections 4–7 and 11–12, and the documents in section 11 below.

## 6. Data sources

| Source | Content | Access |
|---|---|---|
| InfluxDB `bf2_evonith_raw` (AWS eu-central-1) | Online telemetry: `process_params`, `temperature_profile` (110 wall sensors), `heatload_delta_t`, `delta_t`, `cooling_water`, `miscellaneous` | `furnace_data.influx`, field map `src/config/setting_ds_dv.yml → data_mapping` |
| Neon PostgreSQL `offline_feed` | Charge, DPR, hot metal and slag analysis, raw-material chemistry and strength, the historical ML dataset | `furnace_data.offline` (`fetch_offline_data`, `fetch_offline_report`; whitelist `offline_tables.yml`) |
| Neon `plant_master`, `ops_config`, `identity` | Materials, hoppers, units; burden and hopper history (SCD-2); users and roles | `furnace_data.relational` |
| Neon `twin` | Jobs, snapshots, optimiser runs, decisions, shift records and reports, alarms, fact versions, audit log, operator inputs, FurnaceMind conversations | `src/data/twin/` (`TwinStore`), `alembic/twin/` |
| Hourly ML dataset | Merged online + offline hourly rows (IST) | Published by `furnace-data-service`; local rotating cache `src/data/ml/static_dataset_manager.py` |
| Qdrant | FurnaceMind shift summaries (384-dim local embeddings), knowledge documents and skills (1024-dim Voyage) | `agents/memory/`, `agents/embeddings/` |
| Redis | Job queue, event stream `twin:events`, worker heartbeat, live status copies, session revocations | `BF_REDIS_URL` |
| Open-Meteo | Daily relative humidity at Wardha → blast moisture | `utils/coke_energy/weather.py`; shipped record in `assets/coke_energy/` |
| SQLite | Feedback tickets (`src/storage/feedback/tickets.db`, git-ignored); twin-store outbox | `TICKETS_DB_URL`, `BF_TWIN_OUTBOX` |

## 7. Configuration

All in `src/config/`:

| File | Purpose |
|---|---|
| `setting_ds_dv.yml` | InfluxDB measurement and field mappings, furnace geometry, sensor layout, dataset URL and cleaning |
| `setting_vsense.yml` | V-Sense models, control and input parameters, regularisation |
| `setting_bmo.yml`, `bmo_operator_inputs.yml`, `bmo_ore_mapping.yml` | Blend Mix Optimiser: models, limits, prices, ore mapping; operator inputs are versioned in `twin.operator_inputs` |
| `coke_energy.yml`, `energy_balance.yml` | Coke Energy Model, twin settings (freshness, alarms, drift, transit), blast moisture source |
| `burden_optimiser.yml`, `facts_register.json` | Burden & Fuel Optimiser limits and prices; the fact register (every coefficient outside the balance, plausibility bands) |
| `furnace_status.yml`, `shift_report.yml`, `live_inputs.yml` | Furnace status display rules, report KPI thresholds, live-input groups |
| `material_balance.yml`, `materials.yml`, `materials_map.yml` | Material balance constants and DPR mapping; hoppers and materials |
| `furnacemind.yml`, `anomaly.yml`, `logger_setting.yml` | FurnaceMind, channeling propensity, logging |
| `page_registry.py` | The Streamlit pages, their titles, sections and retirement state |

## 8. Environment variables

Read from the environment or `.env` at the repo root (`scripts/dev.py` passes it to every process).

| Group | Variables |
|---|---|
| Databases | `DATABASE_URL` (Neon; `NEON_DATABASE_URL` / `NEON_STR` fallbacks), `BF_TWIN_DB_URL` (default `DATABASE_URL`; `off` disables the twin store), `BF_TWIN_OUTBOX`, `TICKETS_DB_URL` |
| InfluxDB | `INFLUX_ONLINE_TOKEN`, `INFLUX_OFFLINE_TOKEN` (sidecar) |
| Redis and worker | `BF_REDIS_URL`, `BF_WORKER_POOL`, `BF_WORKER_MAX_JOBS`, `BF_WORKER_JOB_TIMEOUT`, `BF_WORKER_SKIP_SCHEDULES`, `BF_WORKER_CATCH_UP`, `BF_WORKER_LIVE_STATUS`, `BF_STRATA_SOLVERS` |
| twin-api | `BF_API_JWT_SECRET` (≥ 32 characters, required in prod), `BF_API_COOKIE_SECURE`, `BF_API_COOKIE_NAME`, `BF_API_TRUSTED_PROXIES`, `BF_API_SESSION_HOURS`, `BF_API_LOGIN_MAX_FAILURES`, `BF_API_LOGIN_WINDOW_S`, `BF_API_STREAM_HEARTBEAT_S`, `BF_API_ENVIRONMENT`, `BF_API_STATIC_DIR`, `BF_API_FM_WARM_UP`, `BF_API_IDENTITY_DB_URL` |
| LLM | `LLM_PROVIDER`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (+ `_LOW_`/`_MEDIUM_`/`_HIGH_MODEL` and reasoning efforts), `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_API_MODE`, `LLM_MAX_TOKENS`, `MEMORY_*`, `SEMANTIC_MEMORY_*` |
| Embeddings | `LOCAL_EMBEDDING_{PROVIDER,MODEL,DEVICE,DIM}`, `CLOUD_EMBEDDING_{PROVIDER,MODEL,API_KEY,DIM}` (`VOYAGE_API_KEY` fallback) |
| Qdrant | `QDRANT_ENDPOINT` + `QDRANT_API_KEY` (cloud) or `QDRANT_URL` (local), `SHIFT_QDRANT_COLLECTION`, `KNOWLEDGE_QDRANT_COLLECTION`, `SKILL_QDRANT_COLLECTION`, `SEMANTIC_MEMORY_QDRANT_COLLECTION`, `FEEDBACK_QDRANT_COLLECTION` |
| Models | `COKE_ENERGY_WEATHER_LIVE` (`0`: no Open-Meteo request), `COKE_ENERGY_TODAY` (fix the plan day), `BF_LIVE_PCI` (`0`: no live PCI default) |
| Caddy (production) | `BF_SITE`, `BF_TLS`, `BF_HSTS`, `BF_SPA_ROOT`, `BF_API_UPSTREAM`, `BF_LEGACY_UPSTREAM`, `BF_HTTP_PORT`, `BF_HTTPS_PORT` |
| Tests | `BURDEN_ENGINE_FAST`, `BF_TEST_REQUIRE_SERVICES`, `BF_TEST_DATABASE_URL` |

Defaults and the file that reads each variable: `CLAUDE.md` section 12 and the tail of that file.

## 9. Tests and checks

```bash
pytest tests/                              # full suite
BURDEN_ENGINE_FAST=1 pytest tests/         # skips the 10–50 s production cases of the burden engine (make test-py)
pytest tests/test_burden_engine*.py        # one area
cd furnace-data-service && uv run pytest test/
pnpm --dir frontend test                   # Vitest
make audit-ui                              # axe-core, overflow, keyboard checks on the built SPA (Playwright)
```

- `tests/conftest.py` puts `src/` and `furnace_data/` on the path, turns off the Open-Meteo request and the twin
  store, fixes the plan day (`COKE_ENERGY_TODAY=2026-10-15`), leaves every plant-data job unscheduled and sets
  placeholder `QDRANT_URL` and `OPENROUTER_API_KEY` values (no test calls Qdrant or an LLM).
- Tests that need PostgreSQL, Redis or Caddy carry `requires_service` (`needs_postgres`, `needs_redis`): skipped
  without the service, failed with `BF_TEST_REQUIRE_SERVICES=1` (CI).
- Never write stubs into `sys.modules` while a file is collected; use `tests.module_stubs.import_with_stubs`.
- Parity fixtures: `make fixtures` (charts, legacy pages), `make client` (OpenAPI client); tests fail on drift.
- Golden page-9 runs: `tests/test_bmo_golden.py` (5 plant days, about 4 min); rebuild with
  `python -m tests.bmo_golden --write`.

**CI** (`.github/workflows/`): `ci.yml` (static, python-tests with PostgreSQL + Redis + Caddy, frontend),
`nightly.yml` (full suite), `release.yml` (artifacts of a tag). All three run on `workflow_dispatch` only
(`gh workflow run ci.yml --ref <branch>`); nothing runs on push, pull request or schedule.

## 10. Deployment

Production is one NVIDIA Jetson (JetPack 6, Ubuntu 22.04, aarch64) behind Caddy. Everything is in `deploy/jetson/`:

| Step | Command |
|---|---|
| Base system (root, idempotent) | `provision.sh` then `verify_base.sh` |
| Python on the device | `install_python.sh` (uv 0.7.13, CPython 3.12) |
| Services | systemd units `bf-redis`, `bf-furnace-data` (:8000), `bf-twin-api` (:8100), `bf-twin-worker`, `bf-streamlit` (:8501), `bf-caddy` (:443), `bf-healthcheck.timer`, grouped by `bf-webapp.target` |
| Release | `sudo deploy/jetson/release.sh <tag>` (CI release artifacts, `uv sync --frozen`, migrations, switch, smoke test, automatic rollback); `rollback.sh [<tag>]` |
| Benchmark | `benchmark.sh --open-issue` → `docs/jetson_benchmark.md` |

Every Python dependency needs a linux-aarch64 wheel (`make lock-check`, `docs/jetson_wheel_audit.md`). Settings for
the units come from `/etc/bf-webapp/env`. See `deploy/jetson/README.md` and `docs/deployment_handover.md`.

## 11. Documentation

| Document | Content |
|---|---|
| `CLAUDE.md` | The system reference: furnace basics, data schema, every page and model, API, worker, conventions |
| `docs/burden_fuel_optimiser.md` | Burden & Fuel Optimiser guide: modes, engine, STRATA-ω, fact register, validation |
| `docs/burnt_vs_charged_coke.md` | Burnt vs charged coke per averaging window (CCC, Bland–Altman) |
| `docs/coke_energy_model_plan.md` | Plan and status of the Coke Energy Model (page 10) |
| `docs/energy_balance_calculation_procedure.md`, `docs/energy_balance_findings_and_open_decisions.md` | Energy-balance procedure and the decision record behind it |
| `docs/bmo_fuel_slag_si_findings.md`, `docs/bmo_si_live_forecast.md`, `docs/bmo_charged_coke_phase1_acceptance.md` | Blend Mix Optimiser studies: fuel model, slag, Si forecast, charged-coke forecast acceptance |
| `docs/coke_rate_model_outcomes.md`, `docs/coke_rate_model_v5_review.md` | Coke-rate model backtests and the V5 review |
| `docs/operator_action_ledger.md`, `docs/plant_data_request.md`, `docs/slag_replay_vs_lab.md` | Operator-action study, plant data request, slag replay against the laboratory |
| `docs/retirement.md` | Retirement of the Streamlit pages: replacement, parity tests, gates, sign-off |
| `docs/deployment_handover.md`, `docs/jetson_benchmark.md`, `docs/jetson_wheel_audit.md` | Deployment handover, device benchmark, wheel audit |
| `frontend/DESIGN.md` | SPA design system: tokens, components, charts, rules |
| `deploy/jetson/README.md` | Device setup, units, Caddy, release flow |
| `furnace-data-service/CODEBASE.md` | The dataset sidecar: layout, routes, lag-aware cache |
| `src/assets/burden_engine/assumptions.md`, `optimiser_notes.md` | Assumptions shown on page 11 / `/optimiser`; notes sent with the optimiser commentary |

## 12. Domain notes

- **Shifts:** fixed 8-hour windows in IST — A 06:00–14:00, B 14:00–22:00, C 22:00–06:00. Every shift analysis,
  report and twin window uses them.
- **KPIs:** fuel rate (`fuel_rate`, kg/tHM, lower is better), coke rate (`coke_rate`), PCI rate
  (`coal_rate_actual_value`), ηCO (`body_etaco`, higher is better), production (`production_per_hour`), RAFT
  (`body_raft`), permeability (`body_perm`), total ΔP (`body_dp_total`).
- **Zones** (elevation from the bottom): hearth 0–5.5 m, tuyere 5.5–10.5 m, bosh 10.5–12.9 m, belly 12.9–15.0 m,
  stack 15.0–20.0 m. The furnace profile and the 110 wall sensors are in `setting_ds_dv.yml`.
- **Coke-to-PCI substitution:** 0.53 kg coke per kg PCI; **unit cost** = 0.25 × (coke rate + 0.53 × PCI rate)
  Lakhs/tHM (V-Sense).
- **Naming:** the same parameter has an InfluxDB field, an ML-dataset column and a plant tag
  (`fuel_rate` / `Act. Fuel RateKg/Thm.` / `BF2 Fuel rate (Kg/THM)`); `setting_ds_dv.yml → data_mapping` is the
  translation table.
