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
├── run_streamlit.py            Entry point — imports torch first (Windows DLL fix)
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
│   │   └── cloud_embedding.py  CloudEmbeddingClient (OpenAI/Voyage, 1024-dim)
│   │
│   ├── multimodal/             Document ingestion
│   │   ├── ingestion.py        process_file() — routes to parser, chunks, embeds
│   │   └── parsers.py          PDF, DOCX, PPTX, XLS/XLSX, TXT parsers
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

Used for authentication and admin data (users, hoppers, materials, burden config). Accessed via SQLAlchemy through `data/db.py`. Burden config uses **SCD Type-2** rows (`valid_upto IS NULL` = current row); hopper raw materials use timestamped all-hopper snapshots in `hopper_raw_material_history`.

### Tickets Database

SQLite-first, PostgreSQL-ready persistence for the Feedback page. Default location: `src/storage/feedback/tickets.db`. Override via `TICKETS_DB_URL` env var.

Three tables: `tickets`, `ticket_events`, `ticket_images`. Schema auto-created via `Base.metadata.create_all()`. Follows service → repository → models pattern in `data/tickets/`.

### Qdrant Vector Store

Two collections:

| Collection | Embeddings | Dim | Content |
|---|---|---|---|
| `furnace_shift_summaries` (env: `SHIFT_QDRANT_COLLECTION`) | Local sentence-transformers | 384 | Shift/day/week/biweek report summaries |
| `knowledge_docs_voyage_1024` (env: `KNOWLEDGE_QDRANT_COLLECTION`) | Cloud (OpenAI/Voyage) | 1024 | Uploaded operator documents |

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
| `CloudEmbeddingClient` | OpenAI `text-embedding-3-large` or Voyage | 1024 | Knowledge doc search |

### `multimodal/`

`process_file(file, knowledge_store, embedding_client)` — routes uploaded files to parsers, chunks text, embeds with `CloudEmbeddingClient`, and upserts into Qdrant.

Supported formats: PDF (PyMuPDF), DOCX, PPTX, XLS/XLSX, TXT.

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
CLOUD_EMBEDDING_MODEL=text-embedding-3-large
CLOUD_EMBEDDING_API_KEY=sk-...   # OpenAI or Voyage key

# Qdrant
QDRANT_ENDPOINT=https://...      # or QDRANT_URL
QDRANT_API_KEY=...
SHIFT_QDRANT_COLLECTION=furnace_shift_summaries
KNOWLEDGE_QDRANT_COLLECTION=knowledge_docs_voyage_1024

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

- Python 3.11+
- `uv` package manager
- Access to InfluxDB, Qdrant, PostgreSQL instances
- `.env` file at repo root with variables from §10

### Install and Run

```bash
# Install dependencies
uv sync

# Run the app (always use this — imports torch first for Windows DLL fix)
python run_streamlit.py

# Run tests
pytest tests/

# Add a new dependency
uv add <package>
```

**Do not invoke `streamlit run` directly.** `run_streamlit.py` pre-imports `torch` before Streamlit starts to prevent Windows DLL load-order errors.

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
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# ML Dataset API tests (no real InfluxDB required)
cd ml-dataset-api
uv run --group dev pytest test/ --cov=app --cov-fail-under=80
```

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
