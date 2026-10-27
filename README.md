# Evonith BF2 — Blast Furnace Intelligence Platform

A production-grade Streamlit web application for real-time monitoring, AI-powered analysis, and optimisation of **Blast Furnace 2 (BF2)** at Evonith Steel. The platform integrates live telemetry from InfluxDB, operational memory via Qdrant, relational data in PostgreSQL, and multiple LLM backends to provide operators and engineers with actionable furnace intelligence.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Pages](#4-pages)
5. [Data Layer](#5-data-layer)
6. [Core Analysis Engine](#6-core-analysis-engine)
7. [AI Systems](#7-ai-systems)
8. [Supporting Packages](#8-supporting-packages)
9. [Configuration](#9-configuration)
10. [Environment Variables](#10-environment-variables)
11. [Setup and Running](#11-setup-and-running)
12. [ML Dataset API Sidecar](#12-ml-dataset-api-sidecar)
13. [Testing](#13-testing)
14. [Key Domain Concepts](#14-key-domain-concepts)
15. [Svelte Frontend and twin-api](#15-svelte-frontend-and-twin-api)

---

## 1. System Overview

The platform has two deployable components:

| Component | Purpose | Tech |
|---|---|---|
| **Streamlit App** (`evonith_webapp/`) | Multi-page dashboard for operators and engineers | Python, Streamlit, LangChain |
| **ML Dataset API** (`ml-dataset-api/`) | Raspberry Pi sidecar for ML dataset caching and delivery | Python, FastAPI, Uvicorn |

The Streamlit app serves 8 pages, each tackling a distinct operational concern:

- **Live monitoring** — temperature profiles, heat loads, cooling water
- **Data exploration** — raw InfluxDB browsing and ML dataset management
- **AI optimisation** — blast parameter recommendations against 3 objectives
- **Channeling detection** — propensity scoring for gas-flow asymmetry
- **Material balance** — per-element daily mass balance with Sankey, bars, and closure table
- **FurnaceMind** — conversational AI co-operator with structured shift memory and reports
- **Feedback** — ticket board for bug reports, feature requests, and issue tracking

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit App (src/)                        │
│                                                                  │
│  custom_pages/         ← 8 page entry points                    │
│  │                                                               │
│  ├── agents/           ← FurnaceMind subsystem (all AI code)      │
│  │   ├── furnacemind/  ← Agent loop, skills, context, prompts    │
│  │   ├── llm/          ← OpenRouter + OpenAI client wrappers     │
│  │   ├── memory/       ← Qdrant vector stores, structured JSON   │
│  │   ├── embeddings/   ← Sentence-transformer + cloud embeddings │
│  │   └── multimodal/   ← File ingestion (PDF, DOCX, XLSX…)       │
│  │                                                               │
│  ├── data/             ← InfluxDB fetchers, ML pipeline,         │
│  │   ├── fetchers/     ← specialised time-series fetchers        │
│  │   └── ml/           ← ML dataset service, cleaning, caching   │
│  │                                                               │
│  ├── utils/            ← Logger, session, settings, prompts,     │
│  │   ├── copilot/      ← AI Copilot page helpers                 │
│  │   └── recommendations/ ← V-OptimAIse helpers                  │
│  │                                                               │
│  ├── ui/               ← Shared Streamlit components             │
│  ├── plotters/         ← Contour chart builders                  │
│  ├── domain/           ← Auth service (PostgreSQL)               │
│  └── geometries/       ← Furnace shape geometry                  │
└─────────────────────────────────────────────────────────────────┘
          │                         │                    │
          ▼                         ▼                    ▼
   InfluxDB (AWS)            Qdrant (cloud)       PostgreSQL (Neon)
   bf2_evonith_raw         shift_summaries,       users, hoppers,
   bf2_evonith_offline_utc knowledge_docs         materials, history

                 ▲
                 │  REST
   ┌─────────────────────────────┐
   │  ML Dataset API (Pi/local)  │
   │  FastAPI on :8080           │
   │  lag-aware CSV caching      │
   └─────────────────────────────┘
```

---

## 3. Repository Structure

```
evonith_webapp/
├── run_streamlit.py            Entry point — any directory; imports torch first on Windows (DLL fix)
├── pyproject.toml
├── CLAUDE.md                   AI-assistant domain knowledge
├── README.md                   This file
│
├── src/                        All application code; CWD at runtime
│   ├── app.py                  Streamlit navigation + auth gate
│   ├── __init__.py
│   │
│   ├── custom_pages/           Page modules (named for Streamlit ordering)
│   │   ├── 1_Welcome.py
│   │   ├── 2_Data_Explorer.py
│   │   ├── 3_Data_Visualisation.py
│   │   ├── 4_Recommendations.py
│   │   ├── 5_AI_Copilot.py
│   │   ├── 6_Material_Balance.py
│   │   ├── 7_FurnaceMind.py
│   │   └── 8_Feedback.py
│   │
│   ├── agents/                 LLM agent machinery
│   │   ├── furnace_tools.py    9 tool functions (fetch, merge, plot, search)
│   │   ├── tool_errors.md      Log of recent tool call failures (fed to LLM context)
│   │   └── furnacemind/         Agent orchestration sub-package
│   │       ├── agent.py        Tool-calling loop + reasoning-model cleanup
│   │       ├── artifacts.py    Plot/Data artifact panel renderer
│   │       ├── context.py      SystemPromptContext: assembles system prompt
│   │       ├── page.py         render_ai_cooperate() — top-level tab renderer
│   │       ├── prompts.py      Static prompt strings (TOOL_POLICY, heatload, etc.)
│   │       └── skills.py       SkillEngine: pre-computes analysis, builds skill prompts
│   │
│   ├── data/                   Data access layer
│   │   ├── retrieval.py        Low-level InfluxDB fetch helpers (online + offline)
│   │   ├── db.py               PostgreSQL SQLAlchemy session
│   │   ├── tickets/            Feedback ticket persistence (SQLAlchemy 2.0)
│   │   │   ├── models.py       ORM models: Ticket, TicketEvent, TicketImage
│   │   │   ├── engine.py       Engine/session factory (SQLite default)
│   │   │   ├── repository.py   Low-level CRUD
│   │   │   └── service.py      Business logic + Pydantic view models
│   │   ├── fetchers/           Specialised InfluxDB fetcher classes
│   │   │   ├── base_data_fetcher.py
│   │   │   ├── ts_data_fetcher.py
│   │   │   ├── ts_heatload_data_fetcher.py
│   │   │   ├── temp_data_fetcher.py
│   │   │   ├── average_heatload_data_fetcher.py
│   │   │   ├── circular_temperature_contour_data_fetcher.py
│   │   │   └── longitudinal_temperature_contour_data_fetcher.py
│   │   └── ml/                 ML dataset pipeline
│   │       ├── main.py         MlDatasetFetcher — multi-source range fetch
│   │       ├── ml_dataset_service.py  MlDatasetService — orchestrates full pipeline
│   │       ├── static_dataset_manager.py  StaticDatasetManager — lag-aware CSV caching
│   │       └── data_cleaning.py  DataCleaner — 16-stage configurable pipeline
│   │
│   ├── agents/                 FurnaceMind agent subsystem
│   │   ├── furnacemind/         Agent orchestration (page, agent loop, context, skills)
│   │   ├── llm/                LLM clients (OpenRouter, OpenAI)
│   │   ├── memory/             Operational memory (schemas, stores, fm_memory, aggregation)
│   │   ├── embeddings/         Embedding clients (local sentence-transformers, cloud)
│   │   ├── multimodal/         Document ingestion + parsers
│   │   └── furnace_tools.py    9 tool functions (fetch, merge, search, plot)
│   │
│   ├── ui/                     Streamlit UI components
│   │   ├── layout.py           Page header, sidebar layout helpers
│   │   ├── styles.py           CSS injection
│   │   ├── components.py       Reusable widgets
│   │   ├── furnacemind_sections.py  Constants + nav tab selector (shared across FM tabs)
│   │   ├── live_ops.py         render_live_operations() + render_furnace_intelligence()
│   │   ├── reports.py          render_reports() — shift/day/week/biweek report viewer
│   │   ├── login_page.py       Cookie-based login form
│   │   ├── user_management.py  Admin: create/edit users
│   │   ├── burden_admin_page.py  Admin: burden distribution config
│   │   └── hopper_admin_page.py  Admin: hopper/material config
│   │
│   ├── utils/                  Cross-cutting utilities
│   │   ├── feedback_page.py    Feedback page UI helpers (board, KPIs, management)
│   │   ├── logger.py           setup_logger() (YAML config) + get_logger(name)
│   │   ├── session.py          is_logged_in(), is_admin(), is_supervisor()
│   │   ├── settings.py         Pydantic Settings singleton (LLM, Qdrant, embeddings)
│   │   ├── prompts.py          PromptTemplates: shift/daily/weekly/biweekly report prompts
│   │   ├── anomaly_propensity.py  Channeling propensity z-score suite
│   │   ├── payload_helpers.py  Payload construction helpers
│   │   ├── rag_methods.py      RAG helper utilities
│   │   ├── validators.py       Input validation helpers
│   │   ├── window_helpers.py   Time-window construction
│   │   ├── copilot/            AI Copilot page helpers
│   │   │   ├── data.py         fetch_recent_online() + df_packet
│   │   │   ├── llm.py          call_llm(), OPENAI_MODEL
│   │   │   └── prompts.py      AI Copilot tab-specific prompt builders
│   │   ├── material_balance/   Material Balance Visualiser package
│   │   │   ├── __init__.py
│   │   │   ├── constants.py    Atomic weights, oxide→element, MaterialSpec registry
│   │   │   ├── data_sources.py Day-window fetchers (RM, HM/Slag, DPR, online)
│   │   │   ├── dpr_mapping.py  DPR field mapping persistence + discovery
│   │   │   └── compute.py      Element-balance math + run_full_balance(day)
│   │   └── recommendations/    V-OptimAIse helpers
│   │       ├── data.py         DataframesProcessor — merges CSV + live InfluxDB
│   │       ├── optimiser.py    run_optimiser() — differential_evolution wrapper
│   │       ├── optimiser_subprc.py  Subprocess-safe optimiser variant
│   │       ├── bounds.py       Control parameter bounds (JSON-backed)
│   │       ├── features.py     Feature engineering for model input
│   │       ├── dependencies.py build_bf_dependency_graph()
│   │       ├── llm.py          LLM review after optimisation
│   │       └── prompts.py      Recommendations LLM prompt templates
│   │
│   ├── plotters/               Plotly chart builders
│   │   ├── base_contour.py     Base class + furnace outline
│   │   ├── circumferential_contour.py  CircumferentialPlotter
│   │   ├── longitudinal_temp_contour.py  LongitudinalTemperaturePlotter
│   │   ├── heatload_contour.py HeatloadContourPlotter
│   │   └── material_balance_plots.py   Sankey, per-element bars, closure styler, furnace diagram
│   │
│   ├── llm/                    LLM client wrappers
│   │   └── llm_client.py       OpenRouterClient + OpenAIClient + get_llm_client()
│   │
│   ├── embeddings/             Embedding clients
│   │   ├── sentence_embedding.py  SentenceEmbedding (local, 384-dim)
│   │   └── cloud_embedding.py  CloudEmbeddingClient (Voyage multimodal, 1024-dim)
│   │
│   ├── multimodal/             Document ingestion
│   │   ├── ingestion.py        process_file() — builds DocumentPart records + indexes MRAG
│   │   └── parsers.py          PDF, DOCX, PPTX, XLS/XLSX, TXT/image extractors
│   │
│   ├── domain/                 Business domain services
│   │   └── auth_service.py     AuthService — PostgreSQL user auth + session cookies
│   │
│   ├── geometries/             Furnace physical geometry
│   │   └── furnace_gen.py      Furnace class — profile coordinates, zone lookup
│   │
│   ├── config/                 Static configuration
│   │   ├── config_loader.py    load_config(filename) → dict
│   │   ├── page_registry.py    Central page navigation registry (PAGE_REGISTRY)
│   │   ├── logger_setting.yml  Logging configuration (YAML-based)
│   │   ├── setting_ds_dv.yml   InfluxDB mappings, data_mapping, furnace geometry
│   │   ├── setting_vsense.yml  V-OptimAIse: models, control/input/output params
│   │   ├── materials.yml       Hopper + raw material definitions
│   │   ├── material_balance.yml Element list, ash assumptions, DPR mapping, closure thresholds
│   │   └── schemas/            JSON payload schemas for shift/day/week/biweek reports
│   │
│   ├── assets/                 Static non-code assets
│   │   ├── data/               CSVs, JSON, copilot analysis Markdown
│   │   │   ├── ml_dataset_filtered.csv   Latest ML dataset (used by FurnaceMind)
│   │   │   ├── control_bounds.json       V-OptimAIse parameter bounds
│   │   │   └── copilot_analysis/         Static LLM context docs
│   │   ├── models/             Pre-trained joblib models + scalers (3 objectives)
│   │   ├── css/                Custom CSS for pages (incl. feedback_style.css)
│   │   └── templates/          Jinja/HTML templates
│   │
│   ├── storage/                Runtime-generated storage
│   │   ├── shift_summaries.json
│   │   ├── daily_summaries.json
│   │   ├── weekly_summaries.json
│   │   ├── biweekly_summaries.json
│   │   └── furnacemind/         FurnaceMind storage
│   │       ├── ai_cooperate_memory.json  Persistent conversation memory
│   │       ├── skill_params.yml          SkillEngine calibration (recalibrate without code changes)
│   │       ├── TOOLS1.md / TOOLS2.md     Tool routing rules (injected into system prompt)
│   │       ├── SKILLS_BESTSHIFT.md       Best-shift parameter bands
│   │       ├── SKILLS_HEATLOAD.md        Heatload skill reference
│   │       ├── SKILLS_OPTIMISE.md        Optimise skill reference
│   │       └── SKILLS_SHIFTREPORT.md    Shift report skill reference
│   │
│
└── ml-dataset-api/             FastAPI sidecar (see §12)
    ├── app/
    ├── config/
    ├── data/
    └── CODEBASE.md
```

---

## 4. Pages

### Page 1 — Welcome (`1_🏭_Welcome.py`)

Landing dashboard showing plant branding and navigation overview. Loads logos from `assets/data/`.

---

### Page 2 — Data Explorer (`2_📓_Data_Explorer.py`)

Interactive browser for all InfluxDB data and the ML dataset pipeline.

**Key features:**
- Select any of the 6 online measurements or 4 offline report types
- Configurable time range, aggregation interval, resampling frequency
- ML Dataset tab: trigger a full pipeline rebuild, preview the cleaned dataset, download CSV
- Raw data download as CSV

**Dependencies:** `data/fetchers/`, `data/ml/`, `data/retrieval.py`, `config/setting_ds_dv.yml`

---

### Page 3 — Data Visualisation (`3_📈_Data_Visualisation.py`)

Interactive contour visualisations of the physical furnace.

**Chart types:**
- **Circumferential temperature** — ring-by-ring sensor heatmap at each elevation level
- **Longitudinal temperature** — axial temperature profile (elevation vs time)
- **Heatload contours** — stave-level heat loads across R6–R10

**Dependencies:** `plotters/`, `data/fetchers/`, `geometries/furnace_gen.py`

---

### Page 4 — V-OptimAIse Recommendations (`4_💡_Recommendations.py`)

ML-driven blast parameter optimiser. Answers: *"Given current raw material quality and burden, what blast settings best achieve my target?"*

**Three optimisation objectives:**

| Objective | Model file | Direction |
|---|---|---|
| Maximise ETA CO | `assets/models/etaco_opt_dec.pkl` | Maximise |
| Maximise Production Rate | `assets/models/prodrate_opt_dec.pkl` | Maximise |
| Minimise Unit Cost | `assets/models/unitcost_opt_dec.pkl` | Minimise |

**Algorithm:** `scipy.optimize.differential_evolution` (strategy `best1bin`, polish=True, λ_reg regularisation penalty against large jumps from current operating point).

**7 control parameters** (user can fix/lock any):
Hot Blast Pressure, Top Pressure, Hot Blast Temperature, Steam, Hot Blast Volume, O₂ Enrichment, PCI Rate.

**9 read-only input groups:** Coke, Nut Coke, PCI, Ore, Sinter, Pellet, Flux, Burden distribution, Other.

After optimisation, results are sent to OpenAI for a concise numeric review.

**Dependencies:** `utils/recommendations/`, `assets/models/`, `assets/data/control_bounds.json`, `config/setting_vsense.yml`

---

### Page 5 — AI Copilot (`5_🤖_AI_Copilot.py`)

**Three independent analysis tabs using OpenAI Responses API + code_interpreter.**

#### Tab 1: Unit Cost Drivers
Static LLM analysis of fuel cost drivers vs the best historical period (Apr–Jun 2024). Sections: high-confidence drivers, counter-moving levers, action checklist.

#### Tab 2: Burden Distribution Impact
Empirical findings from an OLS model (~5,900 rows, R²≈0.43):
- More NON-COKE portions → lower cost
- More COKE portions → higher cost  
- NON-COKE weighted angle outward → lower cost
- Best pattern: ~10–11 coke portions, ~8 non-coke portions, NON-COKE angle ~28° with ≥25% outer share

#### Tab 3: Channeling Analysis
Channeling = asymmetric gas flow through the burden, detected via temperature spread at the 18660mm (Stack) sensors.

`compute_propensity_suite()` fetches the last 8 hours of live data (15-min average) and returns 4 propensity scores (0–100):

| Metric | Signal |
|---|---|
| Channeling | Std-dev of `temp_18660_a/b/c/d` |
| ΔP instability | `body_dp_total` anomaly |
| Permeability instability | `body_perm` anomaly |
| Top pressure instability | `top_press_avg` variability |

Score formula: `100 × (0.6 × magnitude + 0.4 × variability)` — alarm when z_abs_max ≥ 2.5 AND z_std ≥ 1.0.

**Dependencies:** `utils/copilot/`, `utils/anomaly_propensity.py`, `data/retrieval.py`

---

### Page 6a — Material Balance Visualiser (`6_⚖️_Material_Balance.py`)

Single-date element balance: for each of 12 elements (Fe, C, Si, Ca, Mg, Al, Mn, S, P, O, N, H), how many tonnes entered the furnace via raw materials + blast + steam, and how many left via hot metal + slag + top gas.

**Key features:**
- Date picker (default yesterday IST, max yesterday); Refresh button
- Overall mass closure KPI tile (colour-coded green/yellow/red)
- Three tabs: Sankey diagram (total mass or element-focused), per-element stacked bars (4×3 grid), closure table (traffic-light row colours)
- Lightweight furnace cross-section diagram with labelled inflow/outflow arrows
- DPR field mapping expander (one-time configuration; persisted to `material_balance.yml`)
- Assumptions & limitations expander

**Element conversion pipeline:**
1. Fetch RM (3-shift avg), HM/Slag (day avg), DPR, online `process_params` for the picked day
2. Resolve material masses from DPR (with RM-sum fallback)
3. For each material, apply `direct/oxide/H2O/ASH/LOI` rules via `MaterialSpec` to compute element tonnes
4. Add gas-phase inputs: blast O+N, O₂ enrichment O, steam H+O
5. Compute outputs: HM elements (direct wt%), slag elements (oxide split), top-gas C+O+H+N
6. Build closure table: In_t / Out_t / Closure% per element

**Math layer (`utils/material_balance/`)** is fully decoupled from Streamlit — `compute.py` and `constants.py` have zero Streamlit imports and are unit-testable.

**Dependencies:** `utils/material_balance/`, `plotters/material_balance_plots.py`, `data/retrieval.py`, `utils/recommendations/dependencies.py` (bosh-vol formula), `config/material_balance.yml`

---

### Page 7 — FurnaceMind (`7_🧠_FurnaceMind.py`)

**Two-tab AI assistant for operators.**

#### Tab: AI Co-Operate

Conversational AI backed by OpenRouter with a full tool-calling loop. Supports reasoning models (DeepSeek-R1, MiniMax M2.5) — `<think>` blocks are stripped before display.

**System prompt assembly** (`agents/furnacemind/context.py`):
1. `AI_COOPERATE_SYSTEM` base prompt
2. `CLAUDE.md` domain knowledge (up to 24,000 chars)
3. `TOOLS*.md` — tool routing rules
4. `SKILLS*.md` — skill benchmark data
5. Persistent conversation memory (summary + do-not-repeat rules + preferences)
6. Recent tool errors (last 2,500 chars of `tool_errors.md`)

**Skills** (pre-computed before LLM is called — model receives numbers, not code):

| Skill | Trigger phrase | What happens |
|---|---|---|
| Optimise Unit Cost | "optimise" / "unit cost" | 30-day ML data → Tier1/Tier2/Tier3 gap analysis → prompt with pre-computed numbers |
| Shift to Best | "shift to best" + date + label | Static shift data → Tier1 gap vs best-shift bands → bar chart |
| Check Heatloads | "heatload" / "check heat" | LLM fetches live data + executes plot code → report template |

**Tools available to the LLM:**

| Tool | Function |
|---|---|
| `fetch_online_data` | InfluxDB live telemetry (up to 90 days, auto-windowed) |
| `fetch_offline_data` | Manual/shift report data (HM_SLAG, CHARGE, RAW_MATERIAL, DPR) |
| `merge_furnace_data` | Align + merge online + offline datasets |
| `fetch_ml_data` | Load static ML dataset into session |
| `load_static_shift_data` | Load a specific shift's data slice |
| `search_shift_history` | Semantic search on Qdrant shift summaries (384-dim) |
| `search_knowledge_docs` | Semantic search on uploaded operator docs (1024-dim) |
| `execute_python_plot` | Execute sandboxed Plotly code; result in `st.session_state["fm_fig"]` |
| `concat_datasets` | Concatenate two datasets |

**Persistent memory** (`agents/memory/fm_memory.py`):
- `conversation_summary` — compressed summary of past sessions
- `do_not_repeat` — rules learned from operator corrections (max 12)
- `preferences` — operator preferences
- `recent_turns` — last 8 turn pairs

#### Tab: Reports

Renders `ui/reports.py` — shift/daily/weekly/bi-weekly report viewer embedded in the FurnaceMind layout.

---

### Page 8 — Feedback (`8_Feedback.py`)

A shared ticket board for operators and engineers to report bugs, suggest improvements, and track resolution.

**Key features:**
- New Feedback form: page selector, criticality, description, ideal closure, screenshot uploads
- Board view: filterable/sortable ticket cards with status badges and criticality indicators
- Overview KPIs: open/in-progress/resolved counts
- Management panel (admin/supervisor only): status updates, comments, deletion

**Data layer:** `data/tickets/` — SQLAlchemy 2.0 ORM with service/repository pattern. Three tables: `tickets`, `ticket_events` (audit trail), `ticket_images` (screenshot metadata). SQLite by default (`src/storage/feedback/tickets.db`); override via `TICKETS_DB_URL` env var.

**Dependencies:** `data/tickets/`, `utils/feedback_page.py`, `config/page_registry.py`, `assets/css/feedback_style.css`

---

## 5. Data Layer

### InfluxDB Buckets

**Online:** `bf2_evonith_raw` (30-second resolution, AWS eu-central-1)

| Measurement | Content |
|---|---|
| `process_params` | ~30 primary operational fields (blast, gas, KPIs) |
| `temperature_profile` | 110 circumferential wall sensors across 11 elevations |
| `heatload_delta_t` | Cooling stave heat loads R6–R10 + quadrant aggregates |
| `delta_t` | Cooling water ΔT per stave and row average |
| `cooling_water` | Flow, pressure, temperature per cooling circuit zone |
| `miscellaneous` | Stock levels, flare stack, PCI switches, charging rate |

**Offline:** `bf2_evonith_offline_utc` (manual entry)

| Report | Measurement | Cadence |
|---|---|---|
| Hot Metal & Slag | `hotmetal_slag_updated_data` | Hourly |
| Charge | `latest_charge_data` | Hourly |
| Raw Material Composition | `rm_updated_data` | 8-hourly |
| Daily Production Report | `dpr_data` | Daily |

### Fetcher Class Hierarchy

```
BaseDataFetcher (data/fetchers/base_data_fetcher.py)
├── TimeSeriesDataFetcher     — generic time-series
├── TimeSeriesHeatLoadDataFetcher — heatload rows + quadrants
├── TempDataFetcher           — temperature_profile fields
├── AverageHeatLoadDataFetcher — per-row averages
├── CircumferentialTemperatureDataFetcher — elevation ring data
└── LongitudinalTemperatureDataFetcher    — axial profile
```

All fetchers read their field lists from `config/setting_ds_dv.yml → data_mapping`.

### ML Dataset Pipeline (`data/ml/`)

`MlDatasetService` orchestrates a 4-step multi-source fetch:
1. Fetch all 6 online InfluxDB measurements over the requested range
2. Fetch offline data (charge, raw material) and align on timestamp
3. Apply `DataCleaner` (16-stage configurable pipeline: outlier removal, interpolation, lag features, derived columns)
4. Write versioned CSVs to `storage/` (max 3 kept by `StaticDatasetManager`)

The static dataset at `assets/data/ml_dataset_filtered.csv` is the latest committed snapshot used by FurnaceMind's ML skills.

### PostgreSQL

Used for authentication and admin data (users, hoppers, materials, burden config). Accessed via SQLAlchemy through `data/db.py`. Schema uses **SCD Type-2** pattern (`valid_upto IS NULL` = current row).

### Tickets Database

SQLite-first, PostgreSQL-ready persistence for the Feedback page. Default location: `src/storage/feedback/tickets.db`. Override via `TICKETS_DB_URL` env var.

Three tables: `tickets`, `ticket_events`, `ticket_images`. Schema auto-created via `Base.metadata.create_all()`. Follows service → repository → models pattern in `data/tickets/`.

### Qdrant Vector Store

Two collections:

| Collection | Embeddings | Dim | Content |
|---|---|---|---|
| `furnace_shift_summaries` (env: `SHIFT_QDRANT_COLLECTION`) | Local sentence-transformers | 384 | Shift/day/week/biweek report summaries |
| `furnacemind_knowledge` (env: `KNOWLEDGE_QDRANT_COLLECTION`) | Voyage multimodal | 1024 | Uploaded operator documents |

---

## 6. Core Analysis Engine

All core logic in `core/` has zero Streamlit dependencies — fully testable in isolation.

### Shift Building (`core/shift_builder.py`)

`ShiftBuilder.build_shifts(df)` partitions a time-indexed DataFrame into fixed 8-hour windows:

```
Shift A: 06:00–14:00 IST
Shift B: 14:00–22:00 IST
Shift C: 22:00–06:00 IST
```

Returns `Dict[shift_id, ShiftData]` where `shift_id` = `YYYY-MM-DD_Shift_A/B/C`.

### Shift Analysis (`core/shift_analyzer.py`)

`ShiftAnalyzer.analyze()` performs z-score analysis on each numeric column:
- z_warn threshold ≈ 2.5, z_critical ≈ 3.5
- 0 anomalies → stable | 1–3 → warning | >3 → unstable
- Calls LLM with structured prompts from `utils/prompts.py` (7-section report format)
- Returns `(llm_summary, structured_summary)` which feeds `memory/structured_store.py`

### Stability Index (`core/stability_index.py`)

`FurnaceStabilityIndex.compute()` — three penalty components (each capped at sub-100):
- **Variability penalty** (max 40) — avg coefficient of variation across critical parameters
- **Anomaly penalty** (max 40) — 5 points per anomalous parameter
- **Trend penalty** (max 20) — slope of ETA CO linear fit

Final score: `100 - sum(penalties)`, clamped to [0, 100].

### Contextual Analysis (`core/contextual_analyzer.py`)

Compares the current shift against the N most recent shifts from `StructuredStore`, identifying parameter deltas, trend direction, and persistent anomalies. Uses `PromptTemplates.CONTEXTUAL_ANALYSIS_TASK` for the LLM report.

### Aggregation (`memory/aggregation.py`)

`aggregate_daily()`, `aggregate_weekly()`, `aggregate_biweekly()` — roll up shift summaries using LLM synthesis. Results saved to `StructuredStore`.

---

## 7. AI Systems

### FurnaceMind AI Co-Operate

**LLM:** OpenRouter (any model — tested with GPT-4o, DeepSeek-R1, MiniMax M2.5)  
**Architecture:** Custom tool-calling loop (`agents/furnacemind/agent.py`) — NOT LangChain AgentExecutor  
**Max iterations:** 8 tool calls per turn  
**Thinking models:** `<think>…</think>` blocks stripped from output

The `SkillEngine` (`agents/furnacemind/skills.py`) pre-computes all numerical analysis before calling the LLM. The model receives only numbers and a report template — never generates or runs code for analysis.

Calibration lives entirely in `storage/furnacemind/skill_params.yml` — recalibrate regression coefficients, benchmark percentiles, and parameter bands without touching Python.

### V-OptimAIse

**LLM:** OpenAI Responses API (for post-optimisation review)  
**Optimiser:** `scipy.optimize.differential_evolution`

Objective function:
```
f(x) = maxmin × ŷ_scaled(x) + λ_reg × ‖x_scaled − x_prev_scaled‖²
```
where `maxmin = -1` for maximise objectives. The regularisation term prevents impractical step-changes from the current operating point.

### AI Copilot

**LLM:** OpenAI Responses API with `code_interpreter`  
**Static analysis:** Pre-built context docs in `assets/data/copilot_analysis/`

---

## 8. Supporting Packages

### `llm/llm_client.py`

Two clients behind `get_llm_client(prefer)`:

| Client | Use case | API |
|---|---|---|
| `OpenRouterClient` | FurnaceMind agent | OpenRouter (OpenAI-compatible) |
| `OpenAIClient` | AI Copilot, Recommendations review | OpenAI Responses API |

`OpenRouterClient.chat_completions()` sends `HTTP-Referer` and `X-Title` headers for OpenRouter tracking.

### `embeddings/`

| Client | Model | Dim | Use |
|---|---|---|---|
| `SentenceEmbedding` | `all-MiniLM-L6-v2` (local) | 384 | Shift summary search |
| `CloudEmbeddingClient` | Voyage `voyage-multimodal-3.5` | 1024 | Knowledge doc and image search |

### `multimodal/`

`process_file(file, knowledge_store, embedding_client)` — normalizes uploads into `DocumentPart` records, embeds text/table/image/page parts with `CloudEmbeddingClient`, upserts deterministic Qdrant points, and records document metadata in PostgreSQL when a user-scoped repository is available.

Supported formats: PDF (text + rendered page images), DOCX, PPTX (text + embedded images), XLS/XLSX, CSV, TXT/Markdown, and image files.

### `plotters/`

Contour plotters extend `base_contour.py` which loads `Furnace` geometry from `geometries/furnace_gen.py`. The material balance plotter is standalone.

| Plotter | Chart |
|---|---|
| `CircumferentialPlotter` | Polar heatmap of ring sensors |
| `LongitudinalTemperaturePlotter` | Contour plot (elevation × time) |
| `HeatloadContourPlotter` | Stave heat load grid |
| `material_balance_plots` | Sankey, per-element bars, closure table styler, furnace diagram |

### `memory/structured_store.py`

Atomic JSON writes (write to `.tmp`, then rename) for shift/daily/weekly/biweekly summaries. Provides idempotent save, range queries, and a unified `get_report(level, window_id)` API used by the Reports UI.

### `domain/auth_service.py`

Cookie-based session management via `streamlit-cookies-manager` (prefix `bf_dashboard_`). Three roles: `admin`, `supervisor`, `user`. Checked via `utils/session.py`: `is_logged_in()`, `is_admin()`, `is_supervisor()`.

---

## 9. Configuration

### `config/setting_ds_dv.yml`

Master data mapping config:
- `influx_online` / `influx_offline` — connection details
- `data_mapping` — `{measurement: {human_label: influx_field}}` — used everywhere for label↔field translation
- `furnace_geometry` — sensor elevations, counts, zone boundaries
- `DATA` — path to the historical ML dataset CSV (`src/assets/data/V13_df_filtered.csv`)

### `config/setting_vsense.yml`

V-OptimAIse configuration:
- `Optimisation.models` — 3 model entries with `{file, scaler, target, direction}`
- `Optimisation.control_parameters` — 7 control params with display names
- `Optimisation.input_parameters` — 9 groups of read-only features
- `LAMBDA_REG`, `OPTIM_STEPS`, `TIMESTEPS`

### `config/material_balance.yml`

Material Balance Visualiser configuration:
- `elements` — 12 tracked elements (Fe, C, Si, Ca, Mg, Al, Mn, S, P, O, N, H)
- `constants` — gas-phase constants (air density, O₂/N₂ fractions, molar volume)
- `coke_ash_assumption_pct` / `pci_ash_assumption_pct` — constant oxide split for coke and PCI ash (used when shift-level ash chemistry is unavailable)
- `dpr_field_mapping` — user-configured mapping from canonical mass fields to raw DPR column names (persisted via the in-page expander)
- `future_streams` — placeholder fields for dust catcher, sludge, granulation loss (all null in v1)
- `closure_thresholds` — `good: [95, 105]`, `warning: [85, 115]` for traffic-light row colours

### `storage/furnacemind/skill_params.yml`

SkillEngine calibration — recalibrate after a new regression run without touching Python:
- `tier1` — `{col: {best_mid, abs_coeff, lag_hours, unit, adverse: {type, threshold}}}`
- `tier2` — `{col: {band_lo, band_hi, adverse_threshold}}`
- `tier3_guardrail_cols`
- `unit_cost_col`, `benchmark_percentile`

---

## 10. Environment Variables

```bash
# LLM
LLM_PROVIDER=openrouter          # or openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_MODE=responses        # or chat
OPENROUTER_API_KEY=sk-or-...

# Embeddings
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
CLOUD_EMBEDDING_PROVIDER=voyage
CLOUD_EMBEDDING_MODEL=voyage-multimodal-3.5
CLOUD_EMBEDDING_API_KEY=<voyage-api-key>   # or VOYAGE_API_KEY
CLOUD_EMBEDDING_DIM=1024

# Qdrant
QDRANT_ENDPOINT=https://...      # or QDRANT_URL
QDRANT_API_KEY=...
SHIFT_QDRANT_COLLECTION=furnace_shift_summaries
KNOWLEDGE_QDRANT_COLLECTION=furnacemind_knowledge

# PostgreSQL
DATABASE_URL=postgresql+psycopg2://user:pass@host/db

# Tickets (optional — SQLite default if unset)
TICKETS_DB_URL=sqlite:///src/storage/feedback/tickets.db

# InfluxDB
INFLUX_ONLINE_TOKEN=...
INFLUX_OFFLINE_TOKEN=...
```

---

## 11. Setup and Running

### Prerequisites

- Python 3.12 (`requires-python >= 3.12`; `uv` installs it when absent)
- `uv` package manager (lock changes with uv 0.7.13, see CLAUDE.md)
- Access to InfluxDB, Qdrant, PostgreSQL instances
- `.env` file at repo root with variables from §10

### Install and Run

```bash
# Install dependencies
uv sync

# Run the app (always use this — imports torch first on Windows for the DLL fix; works from any directory)
python run_streamlit.py
python run_streamlit.py --server.baseUrlPath legacy    # Streamlit options pass through (as behind Caddy)

# Run tests
pytest tests/

# Add a new dependency
uv add <package>
```

**Do not invoke `streamlit run` directly.** `run_streamlit.py` pre-imports `torch` on Windows before Streamlit starts (DLL load-order errors), runs `src/app.py` from the repository root whatever the current directory, and passes its arguments to `streamlit run`.

All pages execute with `src/` as the Python working directory. Use:
```python
from agents.furnacemind.agent import run_agent_loop   # correct
from src.agents.furnacemind.agent import run_agent_loop  # wrong
```

---

## 12. ML Dataset API Sidecar

A FastAPI service intended to run on a Raspberry Pi (or any local server). See `ml-dataset-api/CODEBASE.md` for full details.

**Port:** 8080

**Endpoint groups:**

| Group | Endpoints | Description |
|---|---|---|
| `/data/online` | `POST /data/online/fetch` | Synchronous InfluxDB live telemetry fetch |
| `/data/offline` | `POST /data/offline/fetch` | Synchronous offline report fetch |
| `/dataset` | `POST /dataset/trigger`, `GET /dataset/status/{id}`, `GET /dataset/download` | Async ML dataset pipeline with background task tracking |

**Caching strategy:** `StaticDatasetManager` maintains a versioned CSV with lag-aware end timestamps. New fetches only pull the uncached tail; the static head is reused. Maximum 3 CSVs kept; auto-rotation on rebuild.

**Run:**
```bash
cd ml-dataset-api
uv sync
python run.py
```

---

## 13. Testing

```bash
# Run all tests (BURDEN_ENGINE_FAST=1 skips the 10-50 s production cases of the burden engine, as CI does on PRs)
pytest tests/
BURDEN_ENGINE_FAST=1 pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# ML Dataset API tests (no real InfluxDB required)
cd ml-dataset-api
uv run --group dev pytest test/ --cov=app --cov-fail-under=80
```

**Test environment.** `tests/conftest.py` disables the Open-Meteo request and the twin store, fixes the plan day,
and sets placeholder `QDRANT_URL` and `OPENROUTER_API_KEY` values (`utils.settings` validates both at import; no test
calls Qdrant or an LLM). Tests that need PostgreSQL, Redis or Caddy carry the marker `requires_service`
(`needs_postgres`, `needs_redis`): skipped without the service, failed with `BF_TEST_REQUIRE_SERVICES=1` (CI).

**Module stubs.** A test that replaces dependencies of the module under test imports it with
`tests.module_stubs.import_with_stubs(name, stubs)`, which restores `sys.modules` afterwards. Stubs written into
`sys.modules` while a file is collected stay for the rest of the session: a `streamlit` stub without `cache_data` made
50 test files fail at collection, and the failures depended on the order of collection.

**Key test files:**
- `tests/test_tickets_service.py` — Ticket CRUD, status transitions, audit events (uses in-memory SQLite)
- `tests/test_page_registry.py` — Page registry consistency checks
- `tests/test_database_facade_sqlalchemy.py` — PostgreSQL database facade

---

## 14. Key Domain Concepts

### Blast Furnace Basics

BF2 converts iron ore + coke + sinter/pellets into hot metal (liquid iron). Hot blast (preheated air + O₂ + steam) is injected at the bottom through tuyeres. PCI (Pulverised Coal Injection) partially replaces coke.

**Coke-to-PCI substitution ratio: 0.53** — 1 kg PCI replaces 0.53 kg coke.

**Unit Cost formula:** `0.25 × (Coke Rate + 0.53 × PCI Rate)` [Lakhs/tHM]

### Furnace Zones (bottom to top)

| Zone | Elevation (m) | Description |
|---|---|---|
| Hearth | 0–5.5 | Molten iron + slag; tapped via taphole |
| Tuyere | 5.5–10.5 | Hot blast injection; PCI injected; raceway combustion |
| Bosh | 10.5–12.9 | Widest zone; burden softening and melting begins |
| Belly | 12.9–15.0 | Transition zone |
| Stack | 15.0–20.0 | Solid burden descends; CO reduction of ore |

### Key KPIs

| KPI | InfluxDB field | Good direction |
|---|---|---|
| Fuel Rate | `fuel_rate` | Lower |
| ETA CO | `body_etaco` | Higher (gas utilisation) |
| Production Rate | `production_per_hour` | Higher |
| Coke Rate | `coke_rate` | Lower |
| PCI Rate | `coal_rate_actual_value` | Within RAFT constraints |
| RAFT | `body_raft` | ~2100–2200°C |
| Permeability | `body_perm` | Higher (less resistance) |
| Total ΔP | `body_dp_total` | Lower |

### Shifts

Fixed 8-hour windows in IST:
- **Shift A:** 06:00–14:00
- **Shift B:** 14:00–22:00
- **Shift C:** 22:00–06:00

### Three Naming Systems for the Same Parameters

The codebase bridges three naming conventions:

| Parameter | InfluxDB field | ML dataset column | MCartech tag |
|---|---|---|---|
| Fuel Rate | `fuel_rate` | `Act. Fuel RateKg/Thm.` | `BF2 Fuel rate (Kg/THM)` |
| ETA CO | `body_etaco` | `FurnaceTopGasAnalysisCO2ETACO` | `BF2_BODY_ETACO` |
| Production | `production_per_hour` | `ProductionTonnesPerHr` | `BF2 Production per hr` |

The `data_mapping` section in `config/setting_ds_dv.yml` is the authoritative translation table used throughout the app.

---

## 15. Svelte Frontend and twin-api

Programme epic #262 (milestone #M5) moves the operator pages to a SvelteKit single-page application served behind
Caddy with a FastAPI service (`twin-api`). Streamlit keeps running under `/legacy` until each page passes parity.

| Part | Path | Run |
|---|---|---|
| SPA (SvelteKit 3, Svelte 5, TypeScript) | `frontend/` | `pnpm --dir frontend dev` (port 5173) |
| twin-api (FastAPI) | `src/api/` | `python -m uvicorn api.main:app --app-dir src --port 8100` |
| Service layer (no Streamlit) | `src/services/` | imported by the API, the worker and the Streamlit pages |
| twin-worker (arq on Redis) | `src/worker/` | `PYTHONPATH=src python -m worker` |
| Streamlit (legacy) | `src/app.py` | `python run_streamlit.py --server.baseUrlPath legacy` (port 8501), served under `/legacy` |

**Development.** Node.js >= 20 with pnpm, plus the Python environment from section 11:

```bash
pnpm --dir frontend install
python scripts/dev.py            # SPA + API + worker + Streamlit; open http://localhost:5173
python scripts/dev.py api web    # a subset
make dev / make lint / make test # Linux, macOS, Jetson
```

The SPA dev server proxies `/api` and `/legacy`, so the browser sees one origin, as in production behind Caddy.

**Production origin.** On the Jetson, Caddy (`deploy/jetson/Caddyfile`, Caddy 2.10) serves one HTTPS origin: the
SPA build with the `200.html` fallback (deep links survive a reload; hashed assets cached for a year, the rest
revalidated), `/api/*` to the twin-api with `flush_interval -1` (server-sent events unbuffered), `/legacy` and
`/legacy/*` to Streamlit with its WebSocket, and a 308 redirect from HTTP to HTTPS. Host name, certificate (Caddy's
internal CA or the plant CA), build directory and upstreams come from `BF_*` variables; see
`deploy/jetson/README.md`. The SPA links to the Streamlit pages by name (`#lib/legacy.ts`, generated from
`config/page_registry.py` by `make fixtures`). `tests/test_caddyfile.py` checks the routes Caddy derives from the file
(needs `caddy` on PATH or `BF_CADDY_BIN`).

**Checks and CI.** Locally: `make lint` (Prettier and ESLint; Ruff on the programme's Python, `make lint-py`),
`make check` (svelte-check), `make test` (Vitest; pytest with `BURDEN_ENGINE_FAST=1`), `make audit-ui`,
`make lock-check`. GitHub Actions (`.github/workflows/`, SV-0.11):

| Workflow | Trigger | Content |
|---|---|---|
| `ci.yml` | pull request; push to `main` and `feature/**`; manual | `static`: Ruff, `uv lock --check`, resolution for the Jetson (`make lock-check`). `python-tests`: `uv sync --frozen`, pytest (`BURDEN_ENGINE_FAST=1`) with PostgreSQL 16 (service container), Redis and Caddy 2.10.2; a missing service fails the run (`BF_TEST_REQUIRE_SERVICES=1`). `frontend`: install from the frozen lock, Prettier and ESLint, svelte-check, Vitest, `schema.d.ts` regenerated from `openapi.json` and compared, build, UI audit (report and screenshots kept 14 days) |
| `nightly.yml` | daily 21:17 UTC (02:47 IST), manual | `ci.yml` with the full suite (the slow production cases of the burden engine) |
| `release.yml` | tag `v*` | `ci.yml` with the full suite, then the release of the tag: `frontend-build.tar.gz` (`build/`), `uv.lock`, `pnpm-lock.yaml`, `SHA256SUMS` |

CI uses uv 0.7.13 (lock rule), CPython 3.12 installed by uv (as on the Jetson), Node 22, pnpm 10.28.0, Ruff 0.17.2,
Playwright 1.63.0 (devDependency of the frontend) and Caddy 2.10.2 (SHA-512 checked); actions are pinned to
commits. Scheduled runs start from the default branch only. Every command step runs through `scripts/ci/step.sh`:
when it fails, its last 50 lines appear as an error annotation on the pull request and in the Checks API.

**Jetson packages.** `make lock-check` exports `uv.lock` and resolves it for `aarch64-manylinux_2_35` (JetPack 6:
Ubuntu 22.04, glibc 2.35) and CPython 3.12 from wheels only, except `pypika` (pure Python) and `quickjs` (C extension,
built with the Jetson's compiler), which publish no aarch64 wheel. xgboost 3.3 requires NCCL on every Linux machine,
and NCCL 2.19.3 (pinned by torch 2.2.2) has x86_64 wheels only; `[tool.uv] override-dependencies` in `pyproject.toml`
limits that requirement to x86_64 Linux, where torch needs it.

**Imports.** Frontend modules under `frontend/src/lib` are imported as `#lib/...` with the file extension
(`#lib/api/client.ts`; SvelteKit 3 subpath imports).

**Sign-in and roles.** The SPA and the API use the accounts and roles of the Streamlit app (Neon
`identity.users`; roles user < supervisor < admin). `POST /api/auth/login` sets a 12 h HS256 session token in an
httpOnly, Secure, SameSite=Strict cookie (path `/api`); `GET /api/auth/me` returns the user; `POST /api/auth/logout`
revokes the token. After 5 failed sign-ins from one address within 60 s further attempts get 429 until the oldest
failure leaves the window. Every API route outside `api.auth.PUBLIC_PATHS` declares a minimum role with
`require_role`, and `tests/test_api_auth.py` fails on a route without one. Settings (environment or `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `BF_API_JWT_SECRET` | random per process (dev) | Signing key, >= 32 characters; **required in prod** (`openssl rand -hex 32`). `scripts/dev.py` keeps one in `src/storage/twin_api/` |
| `BF_API_COOKIE_SECURE` | `true` | Set `false` only for plain-HTTP access from another machine in development |
| `BF_API_TRUSTED_PROXIES` | `127.0.0.1,::1` | Peers whose `X-Forwarded-For` is used for the client address (Caddy) |
| `BF_API_SESSION_HOURS` | `12` | Session lifetime |
| `BF_API_LOGIN_MAX_FAILURES`, `BF_API_LOGIN_WINDOW_S` | `5`, `60` | Failed sign-in limit per address |

The failure limit and the revoked-token list are held in the API process, so the API runs as one process (its
async event loop serves the event streams). Running several API processes needs both shared through Redis; that
decision belongs to the deployment issue (#279).

**Live events and background jobs.** Model runs execute in the twin-worker (`python -m worker`, arq 0.28 on Redis);
the API and the scheduler queue them, and the worker publishes progress to the Redis stream `twin:events`. Each API
process reads the stream once and forwards it to the browsers as server-sent events.

| Route | Role | Purpose |
|---|---|---|
| `GET /api/stream` | user | Event stream: `live.status`, `twin.snapshot`, `forecast.si`, `forecast.charged_coke`, `job.progress`, `job.done`, `alarm.raised`, `alarm.cleared`, and a heartbeat every 15 s. Missed events (at most the last 100) are replayed after a reconnect with `Last-Event-ID` (or `?last_event_id=`). The session is re-checked every 15 s; the stream ends when it is revoked or expired |
| `GET /api/stream/recent` | user | Last events, oldest first |
| `GET /api/jobs/kinds` | user | Job kinds with role, time limit, retries, schedule (IST), next slot and parameter schema |
| `GET /api/jobs`, `GET /api/jobs/{id}` | user | Job records of the last 7 days |
| `GET /api/jobs/stats` | user | Per kind for the jobs of `days` (default 14): runs, done, failed, cancelled, skipped, execution time p50 / p95 / max (over the tries), start delay p95; from `twin.jobs` (Redis, 7 days, without a twin store) |
| `GET /api/jobs/schedule` | user | Audit of the schedule over `days` (default 14): per scheduled kind the slots done, failed, cancelled, running, skipped, missed, the latest 50 not done with their reason, the next slot |
| `POST /api/jobs` | per kind | Queue a job (202); a request equal to a queued or running job (same kind and parameters) returns that job with `duplicate: true` |
| `POST /api/jobs/{id}/cancel` | supervisor | A queued job is not started; a running job is stopped; both end as `cancelled` |

Job semantics: parameters are validated against the kind's model before queuing; the time limit of each kind is
enforced in the worker; transient failures are retried with the kind's retry delay up to `max_tries`; CPU-bound
model solves run in a process pool; a job interrupted by a worker stop runs again after the arq time limit plus
10 s; scheduled kinds start at their IST times (see Scheduled jobs). Kinds: `system.selftest` (admin; progress steps
and one pool task, also in the component catalogue), `system.housekeeping`, the live forecasts `forecast.si` (every
5 min) and `forecast.charged_coke` (HH:07; supervisor, see Live forecasts), `optimiser.run` (supervisor, see Twin and
optimiser API) and the scheduled twin, optimiser, calibration and STRATA-Learn kinds (see Scheduled jobs); the model
kinds follow with their pages. In the SPA, `#lib/live/stream.svelte.ts` holds one EventSource per tab (backoff,
watchdog, session check), `#lib/live/connection.svelte.ts` drives the header state (offline after 10 s without the
stream), and `#lib/live/jobs.svelte.ts` (`JobTracker`) starts and follows a job.

| Variable | Default | Meaning |
|---|---|---|
| `BF_REDIS_URL` | unset (API), `redis://127.0.0.1:6379/0` (worker) | Redis shared by the API and the worker. Without it the API keeps events in-process and the job routes answer 503; `scripts/dev.py` sets it when a local Redis answers |
| `BF_API_STREAM_HEARTBEAT_S` | `15` | Heartbeat and session-check interval of `/api/stream` |
| `BF_WORKER_POOL` | `8` | Processes of the model-solve pool |
| `BF_WORKER_MAX_JOBS` | `4` | Jobs run at the same time by one worker |
| `BF_WORKER_JOB_TIMEOUT` | longest kind limit + 60 s | arq time limit of one delivery |
| `BF_WORKER_SKIP_SCHEDULES` | unset | Kinds, or prefixes ending in `.`, that are not scheduled (still submittable); the test suite sets `forecast.,twin.,optimiser.,calibration.,learn.` |
| `BF_WORKER_CATCH_UP` | `1` | `0` turns off the catch-up of missed slots at worker start (the test suite sets `0`) |

Redis: `apt install redis-server` (Linux, Jetson; Redis 6.0 or later), Memurai or WSL on Windows. Tests that need it
(`tests/test_jobs.py`, `tests/test_api_live.py`) start a throw-away `redis-server` and are skipped without the binary.

**Twin store (Neon `twin` schema).** Persistent results of the twin-api and the worker, in the schema `twin` of the
Neon database: SQLAlchemy 2.0 models and repositories in `src/data/twin/` (`TwinStore`), migrations in
`alembic/twin/`. Times are UTC (`timestamptz`); every time column queried by range leads an index.

| Table | Content | Indexes |
|---|---|---|
| `jobs` | Job records beyond the 7 days kept in Redis: written by the worker at every change of state (ordered, non-blocking), copied again by the nightly housekeeping job; `scheduled_for` (the slot a scheduler job serves) and `run_s` (execution time over the tries) since twin_0002 | `submitted_at`, `(kind, submitted_at)`, `finished_at`, `scheduled_for` |
| `snapshot` | Hourly twin evaluation of the window [t − 8 h, t): coke rates (TRACE, FORGE, STRATA ± 1σ, charged), ω (implied, STRATA, calibration), other quantities as JSON; one row per window end, a re-evaluation replaces it | `window_end` (unique), `evaluated_at` |
| `optimiser_run` | Optimiser runs: mode, ω mode, status, production, cost and saving, coke ± σ, certainty; inputs and plan as JSON | `started_at`, `(mode, started_at)` |
| `plan_decision` | Accept or reject of a run by a supervisor; the latest decision applies; append-only | `decided_at`, `(run_id, decided_at)` |
| `shift_record` | Shift handover record (twin values, plan and decision, report), written once at the shift close | `shift_start` |
| `alarm_event` | Alarms from raise to clear; at most one open alarm per kind and subject | `raised_at`; open alarms unique |
| `facts_version` | Versions of the fact-register overlay; a version that changes a fact's status needs a second administrator | `created_at` |
| `audit_log` | Admin writes and plan decisions (actor, role, address, object, details); append-only | `at`, `(actor, at)`, `(object_type, object_id, at)` |

`shift_record`, `plan_decision` and `audit_log` refuse UPDATE and DELETE (trigger `twin.reject_change`). A user action
(plan decision, fact-register proposal or review, alarm acknowledgement) writes its audit row in the same transaction.
Migrations: `make db-upgrade`, `make db-downgrade` (one revision), `make db-current`, i.e.
`alembic -n twin upgrade head` etc.; the URL comes from `-x url=...`, `BF_TWIN_DB_URL` or `DATABASE_URL`, and Alembic
keeps its version in `twin.alembic_version`. The default Alembic environment (`alembic upgrade head` without
`-n twin`) is the earlier relational migration of `furnace_data` and not part of the twin store.

| Variable | Default | Meaning |
|---|---|---|
| `BF_TWIN_DB_URL` | `DATABASE_URL` | Database of the twin store; `off` disables it (the test suite sets `off`). The worker archives job records when a URL is set |
| `BF_TEST_DATABASE_URL` | unset | PostgreSQL server for `tests/test_twin_store.py` (a role that may create databases: CI service, Neon branch); unset, the tests start a throw-away local cluster with `initdb` / `pg_ctl` (PostgreSQL 16: `apt install postgresql`) and are skipped without one |

**Twin evaluation (SV-1.1).** `services.twin.evaluate(t)` evaluates the window W = [t − 8 h, t), t floored to the
full hour (IST), on the Coke Energy sources and pipeline run of `services.coke_energy`; `services.twin.snapshot_values`
gives its `twin.snapshot` row. The window is aggregated and balanced by the pipeline's own statements
(`build_shift_inputs(key_fn=window_key(start, end))`, `shift_balance`); every 14-day quantity comes from the stable
shifts dated in [d − 14 d, d), d the date of the window's start, as on page 10 (`src/utils/coke_energy/twin.py`).

| Quantity | Definition |
|---|---|
| K_TRACE | Oxygen-balance coke of the metered blast at the analyser's ηCO + 14-day median of (charged − burnt) |
| K_FORGE | Two-zone balance of W with ω, Q_LZ, Q_UZ, r_H at their 14-day medians |
| K_STRATA ± 1σ | The K_FORGE balance with ω = ω_STRATA; σ = \|dK/dω\| · σ(ω_STRATA), dK/dω at fixed ω |
| ω_imp | Two-zone calibration of W on the reconciled basis (charged coke, O-balance air, analyser ηCO) |
| ω_STRATA ± 1σ | STRATA 1.5 for the window's blend and operation against the same 14 days (`src/utils/burden_engine/strata_window.py`) |
| K_charged, production, HMT, Si, blast moisture | From the window's balance; data flags: hourly rows, 10-min samples, taps, provisional DPR dust |

ω_STRATA follows the protocol on which the STRATA 1.5 posterior was fitted: anchor A = blend and operation of the 14
days (ω_A = their median ω, the ω of K_FORGE; κ refitted so that w(A) = ω_A; network context and sinter RDI of the
anchor), network solves at the window's blend (B), blend + production (P) and full operation at the charged coke (F).
The ores enter by type (`ORE_k_CALC_MT` summed per window, mapped by the asset's `plant_map`). The network runs in
exact mode (no Jacobian): one solve per new anchor and three per window, about 1 s each in the development container
(about 3.5 s per window, 6 s with a new anchor). Status: `complete` (TRACE, FORGE, ω_imp and STRATA evaluated),
`partial` (a 14-day quantity or STRATA missing; the notes say which), `failed` (too little data, a known data error, or
a balance that does not solve). Tests: `tests/test_twin_window.py` (every window on a completed shift equals the
pipeline: largest difference 0.0 kg/tHM and 0.0 in ω on 54 synthetic shifts), `tests/test_twin_strata.py`,
`tests/test_services_twin.py`. On plant data (InfluxDB and Neon reachable): `python scripts/twin_parity.py --days 14`
evaluates the window of every balanced shift of the last 14 days against the pipeline table and exits with 1 above
0.1 kg/tHM or 0.0005 in ω (`--strata` adds K_STRATA and the evaluation time per window, `--out` a JSON report).

**Drift alarm (SV-1.2).** `services.drift.run(store, bus, config)` reads the stored windows of the last 56 days and
tests r = ω_imp − ω_STRATA (`src/utils/coke_energy/drift.py`): AR(1) innovations between non-overlapping windows (lag
8 h, i.e. eight interleaved chains; the hourly windows overlap by 7 h and an AR(1) on the hourly series leaves
innovations correlated 0.86 at one hour), ρ and σ_e re-estimated at every evaluation (σ_e from the median absolute
deviation), two-sided CUSUM k 0.5, h 5. Alarm kind `omega_drift`, subject `high` (ω_imp above STRATA) or `low`, raised
at the first signal, cleared 24 h after the last; raises and clears are published as `alarm.raised` /
`alarm.cleared` (code `OMEGA_DRIFT`) and reach `/api/stream`. Settings: `config/coke_energy.yml → twin.drift`.
Replay: `python scripts/drift_replay.py` (daily series of the STRATA evidence, 1 Jan – 7 Oct 2026; `--source
snapshots` for the stored hourly windows).

**Burden transit (SV-1.3).** `src/utils/coke_energy/burden_column.py`, `services.transit`. Reference of the layered
gas-flow model (drawn profile, 23.5 m³ per charge): at 5.9 charges/h the cohesive-zone top 2.2 h after charging, the
raceways 6.2 h (zones 2.2 / 0.9 / 3.1 h), coke 61 / 14 / 59 t = 133 t in transit (`coke_energy.yml → twin.transit`).
Descent times: τ = τ_ref · 5.9 / n_ch at the current rate, and from the record the age of the charge at each zone
boundary (the zones hold fixed numbers of charges, 12.98 / 5.31 / 18.29). The ledger counts the charges back from a
time through the hourly record and gives per band the charges, the tonnes and wet blend of the ferrous charge, coke per
charge, per tonne of ferrous and per tonne of hot metal; coke above and in the zone scales with the coke share of the
charge volume (reference 0.387), the coke bed below it is the reference 59 t. Every window evaluation carries the
column at its end (`tau_cz_h`, `tau_raceway_h`, `coke_*_t`). Implied lag: coke per tonne of ferrous charged against
K_TRACE of the stored windows over 14 days, 95 % moving-block bootstrap interval. Report:
`python scripts/transit_report.py [--at ...] [--lag]`.

**HMT forecast (SV-1.4, gated).** `src/utils/coke_energy/hmt_forecast.py`: HMT_f(t + τ_RW) = HMT(t) + [K_arr(t) −
K_FORGE(HMT(t), x(t))] / (∂K_FORGE/∂HMT), HMT(t) the latest tap (≤ 3 h old), K_arr the dry coke per tHM of the last
hour's charges (their coke-to-ferrous ratio times the window's ferrous rate), K_FORGE the window's FORGE balance at
HMT(t), the slope by a 10 °C difference, τ_RW from the burden column. Every window carries these quantities
(`hmt_now_c`, `k_arr_kg_thm`, `dk_dhmt_kg_thm_per_c`, `hmt_forecast_c`, `hmt_forecast_horizon_h`); the Live Twin
shows the forecast only when `twin.hmt_forecast.gate.status` is `passed`. Gate run on plant data:
`python scripts/hmt_gate.py --from 2026-01-01 --to 2026-10-08 --write` (RMSE at least 10 % below persistence and
|bias| ≤ 5 °C; recorded as COKE-18, the 2023–24 run as COKE-19).

**Live forecasts (SV-1.5).** The worker issues the HM Si path (`forecast.si`, every 5 min IST) and the charged-coke
block (`forecast.charged_coke`, HH:07, retried every 5 min up to HH:27 while the hour's build of the published dataset
is missing; the local copy is fetched again first when it predates the hour) through `services.forecasts`, with the
functions page 9 used, into the same ledgers (`src/storage/bmo_si_forecast/`, `src/storage/bmo_charged_coke/`, format
unchanged; Si outcomes matched on every call, charged-coke outcomes on each new row). `setting_bmo.yml →
bmo.forecast_issuer`: `auto` (page 9 reads the ledgers while the worker's last attempt is younger than 15 min (Si) or
75 min (charged coke), else issues itself as before), `worker` (page 9 only reads; an out-of-date record is not passed
to the optimiser) or `page`. Every attempt is logged in `attempts/YYYY-MM.jsonl` of the storage folder (the latest with
its record in `latest_attempt.json`, which page 9 shows for a Si attempt with unusable inputs); events
`forecast.si` and `forecast.charged_coke`. The ledgers are append-only JSON lines shared by processes
(`utils.jsonl_log`: incremental reads, appends under a cross-process lock). Gap check:
`python scripts/forecast_gaps.py --days 7` (exit 1 on a Si gap above 10 min or a charged-coke gap above 60 min).
`BF_WORKER_SKIP_SCHEDULES=forecast.` leaves the forecast jobs unscheduled (the test suite leaves out every kind that
reads plant data).

**Twin and optimiser API (SV-1.6).** Read from the twin store (`twin` schema; `services.store`, 503 without
`BF_TWIN_DB_URL` / `DATABASE_URL`, 503 while the database is unreachable). Every numeric field states its unit in the
schema (`tests/test_api_twin.py` checks it). Coke kg/tHM dry, ω dimensionless, costs Rs/tHM, production t/d, times UTC.

| Route | Role | Purpose |
|---|---|---|
| `GET /api/twin/now` | user | Latest evaluation (`TwinSnapshotData`), its age, the lag behind the window that should exist (run at HH:10, due at HH:15; stale from one missed hour) and the open alarms |
| `GET /api/twin/series` | user | Stored windows in [start, end] (default 7 days, at most 92), `resolution=hour` or `shift` (the windows ending 06:00, 14:00, 22:00 IST: shifts C, A, B), selected `quantities`; 30 days hourly in 40–180 ms (SQLite and PostgreSQL, test fixture) |
| `GET /api/twin/transit` | user | Burden column at `at` from the hourly dataset: descent times, coke by zone and in transit, the blend by zone; `lag=true` adds the implied lag against the stored K_TRACE |
| `GET /api/twin/alarms` | user | Open alarms and the history of `days` (default 7), optionally one `kind` |
| `POST /api/optimiser/runs` | supervisor | Queue `optimiser.run` (202): `mode` step1, step2 or max_production, `omega_mode` strata or fixed, Step 2 `production_t_d` (default: the latest twin window's production, at most 3 h old, else page 11's regime median), certainty filter, risk aversion, `calibration` day (default: the calibration of 02:30) or live (page 11's); page 11's defaults otherwise (`services.optimiser.default_case`) |
| `GET /api/optimiser/runs`, `GET /api/optimiser/runs/{run or job id}` | user | Recent runs with their decision; one run with its inputs, result and decisions, or the job while it runs |
| `GET /api/optimiser/latest` | user | Latest plan (status ok), optionally per `mode` |
| `POST /api/optimiser/runs/{id}/decision` | supervisor | Accept or reject a plan with a comment; appended to `twin.plan_decision` with its audit row (`plan.decide`, actor, role, client address) in one transaction; the latest decision applies |

The worker solves `optimiser.run` in the model-solve pool and stores the plan in `twin.optimiser_run` (status ok,
infeasible with the LP's or the band's reason, or failed), with the job record it refers to.

**Scheduled jobs (SV-1.7).** The worker's schedule (IST); every slot is submitted with `scheduled_for` = the slot.

| Kind | Time | Work | Tries |
|---|---|---|---|
| `forecast.si` | every 5 min | HM Si path into the Si ledger (Live forecasts) | 1 |
| `forecast.charged_coke` | HH:07 | Charged-coke block of the latest complete row | 5, every 5 min |
| `twin.evaluate` | HH:10 | Window [HH − 8, HH) into `twin.snapshot`, once the local hourly dataset holds the row HH − 1 (published in the build of about HH:06; at HH:05 the window would have had 7 of its 8 hours); the windows of the last 24 h without a snapshot, or evaluated without their last row, first; then the implied-ω drift alarm and the `twin.snapshot` event of W | 4, every 5 min; the last try evaluates on the rows there are and flags the window |
| `optimiser.step2` | 06:15, 14:15, 22:15 | Step 2 with STRATA-ω and Fixed-ω in parallel in the pool, at the current production (latest twin window, at most 3 h old, else the regime median), live yard and prices, the day's calibration; two `twin.optimiser_run` rows | 3, every 5 min while the yard or calibration falls back to the reference; the last try runs on it, labelled |
| `optimiser.step1` | 06:30, 14:30, 22:30 | Step 1 (cheapest production level) with STRATA-ω, the day's calibration | 2 |
| `calibration.nightly` | 02:30 | Two-zone calibration on the stable shifts dated in the 14 days before today (`src/storage/burden_engine/calibration/<date>.json`, 30 days kept) and the STRATA-Flow Jacobian at it (17 network solves, disk cache) | 3, every 10 min |
| `learn.shadow` | 03:00 | STRATA-Learn shadow run on the stored evidence into `src/storage/burden_engine/learn/`; changes nothing in the asset, the register or the pages | 1 |
| `system.housekeeping` | 04:00 | Archive sync, index trim, runtime statistics of the last 14 days into its result | 2 |

The day's calibration (`services.day_calibration`) serves every optimiser run of the worker: the scheduled plans of a
day compare on one calibration, and none spends the 17 Jacobian solves (21 s here) that a new live calibration costs
after each balanced shift. Page 11 keeps its live calibration (`calibration=live` reproduces it). Catch-up at worker
start (`worker.runtime.catch_up`): the latest slot of each kind runs when it is at most 30 min old; the other slots
since the last heartbeat (Redis `twin:worker:alive`, every 10 s) are recorded as cancelled jobs `scheduler
(skipped)` with the reason, as is a slot whose previous job of the kind is still queued or running. Audit:
`python scripts/schedule_audit.py --days 14` (from `twin.jobs`; exit 1 on a missed or skipped slot, `--strict` also on
a failed run; with the runtime statistics per kind) or `GET /api/jobs/schedule`. Runtime of the scheduled optimiser
runs on a host, as the job runs them: `python scripts/optimiser_bench.py --mode step2` (STRATA-ω and Fixed-ω in a
2-process pool; exit 1 above `--limit-s`, default 90 s; `--fresh-jacobian` times the 17 Jacobian solves first) or
`--mode step1`.

**Design system.** Tokens, type, rules and components are described in `frontend/DESIGN.md`; every component is
shown in its states at `/dev/components`. Colours come only from `src/lib/theme/tokens.css` (V-OptimAIse product-family
look as in Optimus: dark sidebar, light canvas, white panels, yellow accent, Inter; text contrast >= 4.5:1 on every
surface, checked by a unit test). `make audit-ui` builds the SPA and runs axe-core,
overflow, keyboard and dialog checks at 1920x1080, 1366x768 and 768x1024, and on the catalogue at 390x844
(Playwright with Chromium required).

**Charts and furnace schematic.** `frontend/src/lib/charts/` holds the chart kit on Apache ECharts 6 (SVG renderer,
loaded on first use): time series on an IST axis with shift starts, uncertainty and plant bands, annotations and
direct end labels; scatter with identity and limit lines; bars; donut; the Rist operating diagram of a shift. Every
chart has a legend that toggles series and a table view with every value. Series colours are the six validated
categorical slots of `tokens.css` (TRACE, FORGE, STRATA and HMT keep slots 1-4 on every page). The BF-02 schematic
(`frontend/src/lib/furnace/`) is drawn from the configured furnace profile. `make fixtures` writes the parity fixtures
from the Python models (`utils.coke_energy.rist_diagram`, `ui.furnace_status.schematic`); the Vitest suite checks the
Svelte drawings against them and `tests/test_chart_fixtures.py` fails when a committed fixture is out of date. Rules:
`frontend/DESIGN.md`, section 6.

**Typed API client.** `frontend/src/lib/api/client.ts` wraps `openapi-fetch` with the types generated from the API's
OpenAPI schema. After changing a route: `make client` (or `python scripts/export_openapi.py` and
`pnpm --dir frontend api:types`), then commit `openapi.json` and `schema.d.ts`; `tests/test_api_foundation.py` fails
on drift.
Python routers import `services.*` only, never `ui.*` or Streamlit.

