# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Blast Furnace Web Application — an industrial manufacturing dashboard for real-time blast furnace monitoring, AI-powered analysis, and optimization recommendations. Built with Streamlit (multi-page app) backed by PostgreSQL, InfluxDB (time-series), and Qdrant (vector search).

## Commands

```bash
uv sync                    # Install dependencies
python run_streamlit.py    # Run the app (always use this, not streamlit run directly; Streamlit options pass through)
pytest tests/              # Run tests
uv add <package>           # Add dependency
```

**Do not invoke `streamlit run` directly** — `run_streamlit.py` imports torch first on Windows (DLL errors), runs `src/app.py` from the repository root from any directory, and passes its arguments to Streamlit (`--server.baseUrlPath legacy` behind Caddy).

**All pages run with `src/` as the working directory** — use `from agents.xxx`, `from utils.xxx`, etc., not `from src.agents.xxx`. The old `FurnaceMind/` package was flattened into top-level dirs under `src/`.

---

## 1. Blast Furnace Basics

### What this monitors

**BF2** at Evonith Steel. A blast furnace converts iron ore + coke + sinter/pellets into hot metal (liquid iron). Key inputs from the top (burden — ore, coke, sinter, pellet, flux) descend while hot blast (preheated air + O₂ + steam) is injected at the bottom through tuyeres. PCI (Pulverised Coal Injection) is co-injected with the blast as a partial coke substitute.

### Key Performance Indicators (KPIs)

| KPI | InfluxDB field | Description |
|---|---|---|
| Fuel Rate | `fuel_rate` | Total fuel consumed per tonne hot metal (kg/tHM) |
| Coke Rate | `coke_rate` | Coke consumed per tonne hot metal (kg/tHM) |
| PCI Rate | `coal_rate_actual_value` | PCI coal injected (kg/tHM) |
| ETA CO | `body_etaco` | Gas utilization efficiency — CO→CO₂ conversion; higher is better |
| Production Rate | `production_per_hour` | Hot metal produced (t/hr) |
| RAFT | `body_raft` | Raceway Adiabatic Flame Temperature — thermal driving force |
| Permeability | `body_perm` | Burden gas permeability index |
| Total ΔP | `body_dp_total` | Top-to-bottom pressure differential across burden |

**Coke-to-PCI substitution ratio: 0.53** — 1 kg PCI replaces 0.53 kg coke.

**Unit Cost formula** = 0.25 × (Coke Rate + 0.53 × PCI Rate) [Lakhs/tHM]

### Furnace Coordinate System

- **Y-axis** = elevation from furnace bottom (metres), range 0–20 m
- **X-axis** = radial distance from centreline (negative = outward), range -4–0 m
- Furnace profile outline (x, y): (-2.8, 4.374) → (-2.8, 6.795) → (-3.15, 8.335) → (-3.15, 11.290) → (-3.65, 14.390) → (-3.65, 15.89) → (-2.898, 20.0)

### Furnace Zones

| Zone | Elevation (m) | Description |
|---|---|---|
| Hearth | 0 – 5.5 | Molten iron and slag accumulate; tapped via taphole |
| Tuyere | 5.5 – 10.5 | Hot blast injection zone; PCI injected here; raceway combustion |
| Bosh | 10.5 – 12.9 | Widest zone; burden softening and melting begins |
| Belly | 12.9 – 15.0 | Transition zone |
| Stack | 15.0 – 20.0 | Solid burden descends; ore reduction with CO gas |

### Shifts

Operations run in fixed **8-hour shifts** (A: 06:00–14:00, B: 14:00–22:00, C: 22:00–06:00 IST). All shift analysis uses these windows.

---

## 2. InfluxDB Data Schema

**Bucket:** `bf2_evonith_raw` | **Org:** Blast Furnace, Evonith | **Host:** AWS eu-central-1

There are **6 measurements** (InfluxDB tables). Config in `src/config/setting_ds_dv.yml` under `data_mapping`.

### Measurement: `temperature_profile`

**110 circumferential temperature sensors** embedded in the furnace wall at 11 elevations.

Field naming: `temp_{elevation_mm}_{letter}` — e.g., `temp_4373_a`, `temp_8335_n`

| Elevation (mm) | Elevation (m) | Sensor Count | Letters | Zone |
|---|---|---|---|---|
| 4373 | 4.373 | 7 | A–G | Hearth |
| 5411 | 5.411 | 13 | A–M | Tuyere |
| 5757 | 5.757 | 13 | A–M | Tuyere |
| 6103 | 6.103 | 13 | A–M | Tuyere |
| 6795 | 6.795 | 12 | A–L | Tuyere |
| 7565 | 7.565 | 14 | A–N | Tuyere |
| 8335 | 8.335 | 14 | A–N | Tuyere/Bosh |
| 9105 | 9.105 | 12 | A–L | Bosh |
| 12975 | 12.975 | 4 | A–D | Belly |
| 15162 | 15.162 | 4 | A–D | Stack |
| 18660 | 18.660 | 4 | A–D | Stack |

The letters (A, B, C, …) represent circumferential positions around the furnace at that elevation. The 18660mm level (4 sensors) is the key indicator for channeling detection — abnormal spread across A–D signals asymmetric gas flow.

### Measurement: `process_params`

~30 primary operational fields:

| Field | Description |
|---|---|
| `hot_blast_vol_nm3h` | Hot blast volume (Nm³/hr) |
| `hot_blast_press` | Hot blast pressure (bar) |
| `hot_blast_temp` | Hot blast temperature (°C) |
| `top_press_avg` | Top pressure average (bar); also `top_press_1/2/3/4` individually |
| `top_temp_avg` | Top gas temperature average (°C); also `top_temp_1/2/3/4` |
| `steam_injection` | Steam injection (kg/hr) |
| `oxygen_enrichment_pct` | O₂ enrichment (%) |
| `tuyere_velocity` | Tuyere blast velocity |
| `co_pct` / `co2_pct` / `h2_pct` | Top gas composition (%) |
| `body_etaco` | ETA CO (gas utilization) |
| `body_raft` | RAFT (°C) |
| `body_perm` | Permeability index |
| `body_dp_bottom` / `body_dp_top` / `body_dp_total` | Differential pressure (bar) |
| `fuel_rate` | Total fuel rate (kg/tHM) |
| `coke_rate` | Coke rate (kg/tHM) |
| `coal_rate_actual_value` | PCI coal rate (kg/tHM) |
| `production_per_hour` | Hot metal production (t/hr) |
| `runner_temp_pci_taphole` / `runner_temp_cr_taphole` | Runner temps near taphole (PCI side / CR side) |
| `runner_temp_pci_skimmer` / `runner_temp_cr_skimmer` | Runner temps near skimmer |

### Measurement: `heatload_delta_t`

Cooling stave heat load arranged in **Rows R6–R10** (5 rows), each row with **32 staves** divided into 4 quadrants:

| Quadrant | Staves |
|---|---|
| Q1 | 1–8 |
| Q2 | 9–16 |
| Q3 | 17–24 |
| Q4 | 25–32 |

Fields: `heat_load_row_{6-10}` (per-row), `heat_load_r{6-10}_q{1-4}` (per quadrant), `heat_load_row6_10_q{1-4}` (cross-row quadrant averages).

### Measurement: `delta_t`

Cooling water temperature differential (ΔT = T_out - T_in) per stave:
- Row averages: `delta_t_avg_row{6-10}`
- Quadrant averages: `delta_t_avg_row6_10_q{1-4}`
- Individual stave pairs: `delta_t_r{row}_stave_{n}` (e.g., `delta_t_r6_stave_1` through `delta_t_r6_stave_32`)

### Measurement: `cooling_water`

Flow (m³/hr), pressure (bar), and temperature (°C) per cooling circuit zone:

| Field | Zone |
|---|---|
| `cw_hearth_flow_m3h` / `cw_hearth_press_bar` | Hearth |
| `cw_bottom_flow_m3h` / `cw_bottom_press_bar` | Bottom |
| `cw_bosh_belly_flow_m3h` / `cw_bosh_belly_press_bar` | Bosh & Belly |
| `cw_lower_stack_flow_m3h` / `cw_lower_stack_press_bar` | Lower Stack |
| `cw_upper_shaft_flow_m3h` / `cw_upper_shaft_press_bar` | Upper Shaft |
| `cw_tuyere_nose_flow_m3h` / `cw_tuyere_nose_press_bar` | Tuyere Nose |
| `cw_mains_temp_c` / `cw_hp_mains_temp_c` | Mains temperatures |

### Measurement: `miscellaneous`

Stock levels, flare stack, coal injection switches, and charging:
- `stock_rod_radar_level`, `stock_rod1_pos`, `stock_rod2_pos` — burden stock level
- `flare_press`, `flare_bypass_press`
- `coal_sw_01` through `coal_sw_18` — PCI coal flow switches (18 lances)
- `skip_car_trips_hour` — charging rate

### Offline Bucket: `bf2_evonith_offline_utc`

Manual-entry data at lower temporal resolution:

| Report | Measurement | Default cadence |
|---|---|---|
| Hot Metal & Slag | `hotmetal_slag_updated_data` | Hourly |
| Charge data | `latest_charge_data` | Hourly |
| Bunker / Raw Material | `rm_updated_data` | 8-hourly (shift-wise) |
| Daily Production Report | `dpr_data` | Daily |

### Parameter Synonyms (3 naming systems)

| Parameter | InfluxDB field | ML dataset column | MCartech tag |
|---|---|---|---|
| Fuel Rate | `fuel_rate` | `Act. Fuel RateKg/Thm.` | `BF2 Fuel rate (Kg/THM)` |
| ETA CO | `body_etaco` | `FurnaceTopGasAnalysisCO2ETACO` | `BF2_BODY_ETACO` |
| Production | `production_per_hour` | `ProductionTonnesPerHr` | `BF2 Production per hr` |
| Coke Rate | `coke_rate` | `Coke Rate Kg/Thm` | `BF2 Coke rate (Kg/THM)` |
| Hot Blast Volume | `hot_blast_vol_nm3h` | `Hot Blast VolumeNm3/Hr.` | `BF2_PROC Hot Blast Volume` |

---

## 3. Qdrant Shift Summaries (Work In Progress)

**Status: WIP.** The shift summary pipeline exists but aggregation and regular scheduling are incomplete.

### Two Qdrant Collections

| Collection | Embeddings | Dim | Purpose |
|---|---|---|---|
| `furnace_shift_summaries` (env: `SHIFT_QDRANT_COLLECTION`) | Local sentence-transformers | 384 | Shift/day/week reports |
| `furnacemind_knowledge` (env: `KNOWLEDGE_QDRANT_COLLECTION`) | Voyage multimodal | 1024 | Uploaded operator docs |

### Shift Summary Schema (`src/agents/memory/schemas.py`)

Each `ShiftSummary` stored in Qdrant has:

```
shift_id:              YYYY-MM-DD_Shift_A|B|C
shift_start/end:       Timestamps
stability_index:       0–100 numeric score
stability_status:      UNKNOWN | NORMAL | STABLE | WARNING | UNSTABLE
anomalous_parameters:  List of parameter names flagged
anomaly_details:       {param → {severity, trend, z_score, delta_vs_prev, reasons}}
stability_penalties:   {param → penalty_contribution}
early_drift:           {param → early warning signals}
operator_context:      {notes, feedback: {rating 1–5, comment}}
```

Optional future fields: `control_actions`, `fuel_efficiency_indicator`, `thermal_balance_indicator`, `material_condition_notes`

### How Shifts Are Built and Analyzed

**`ShiftBuilder`** (deleted — was `core/shift_builder.py`): Partitions a DataFrame into 8-hour windows (A/B/C), handles IST timezone.

**`ShiftAnalyzer`** (deleted — was `core/shift_analyzer.py`): Z-score analysis on each shift:
- z_warn threshold ≈ 2.5, z_critical ≈ 3.5
- 0 anomalies → stable; 1–3 → warning; >3 → unstable
- Calls LLM with `SHIFT_ANALYZER_SYSTEM` + `SHIFT_ANALYSIS_TASK` prompts (7-section structured report)
- Returns `(llm_summary, structured_summary)` where structured_summary feeds the Qdrant payload

### LLM Reporting Prompts (`src/utils/prompts.py`)

All prompts enforce **ACTION / REASON / MAGNITUDE** discipline. KPI priority order is hard-coded:
- **High**: Fuel Rate, ETA CO, Production Rate
- **Medium**: Quadrant heat loads (R6–R10, Q1–Q4)

Report types: `SHIFT_ANALYSIS_TASK`, `CONTEXTUAL_ANALYSIS_TASK` (vs prior shifts), `DAILY_REPORT_TASK`, `WEEKLY_REPORT_TASK`, `BIWEEKLY_REPORT_TASK`

### Persistence (`src/agents/memory/structured_store.py`)

File-based JSON (atomic writes via `.tmp` files):
- `shift_summaries.json`
- `daily_summaries.json`
- `weekly_summaries.json`
- `biweekly_summaries.json`

---

## 4. V-OptimAIse — Recommendations Page

**File:** `src/custom_pages/4_💡_Recommendations.py`

The optimizer answers: *"Given the current raw material quality and burden, what blast parameters should I set to best achieve my target?"*

### Three Optimization Objectives

| Model | Output | Direction | Model file |
|---|---|---|---|
| Eta CO | `FURNACETOPGASANALYSISCO2ETACO` | Maximize | `src/models/etaco_opt_dec.pkl` |
| Production Rate | `PRODUCTIONTONNESPERHR` | Maximize | `src/models/prodrate_opt_dec.pkl` |
| Unit Cost | `UNITCOST LAKHS/THM` | Minimize | `src/models/unitcost_opt_dec.pkl` |

Models are pre-trained XGBoost/sklearn joblib models with paired scalers (`*_scaler_dec.pkl`).

### Control Parameters (7 — what the optimizer can vary)

1. `HOT BLAST PRESSUREBAR`
2. `TOPPRESSUREBAR`
3. `HOT BLAST TEMP.OC`
4. `STEAMKGS/HR.`
5. `HOT BLAST VOLUMENM3/HR.`
6. `O2 ENRICHMENT %`
7. `PCI_KG/THM` (PCI injection rate)

The user can **fix** any control parameter (checkbox override), locking it at a specified value. Only free parameters are passed to the optimizer. Bounds are persisted in `src/data/control_bounds.json` and loaded on page startup.

### Input Parameters (features, read-only — 9 groups)

These describe current raw material quality and burden, fetched from live InfluxDB ML dataset and historical CSV:

| Group | Key fields |
|---|---|
| **Coke** | VM%, Ash%, IM%, FC%, Moisture%, Calc MT |
| **Nut Coke** | VM%, IM%, FC%, Ash%, Moisture%, Calc MT |
| **PCI** | FC%, Ash%, VM%, IM%, Calc MT |
| **Ore** | Fe(T)%, LOI%, TM%, MgO%, SiO₂%, Al₂O₃%, CaO%, K₂O%, Na₂O%, P%, TiO₂%, MnO%, Calc MT |
| **Sinter** | RI, RDI, AI (strength indices), P%, SiO₂%, MgO%, FeO%, Al₂O₃%, CaO%, Fe(T)%, Basicity, Calc MT |
| **Pellet** | SiO₂%, Al₂O₃%, Fe₂O₃%, MgO%, TiO₂%, CaO%, MnO%, K₂O%, Na₂O%, TM%, LOI%, P%, Calc MT |
| **Flux** | TM%, SiO₂%, Fe₂O₃%, Al₂O₃%, LOI%, MgO%, CaO%, Calc MT |
| **Burden distribution** | Coke discharge time, weighted coke angle, total coke portions, non-coke discharge time, weighted non-coke angle, total non-coke portions |
| **Other** | Charges/hr, Stock rod level, H₂% in top gas |

### Optimization Algorithm (`src/utils/recommendations/optimiser.py`)

**Algorithm:** `scipy.optimize.differential_evolution` (strategy: `best1bin`, polish: True, population: 15, max_iter: 30, tol: 0.01)

**Objective function** (minimized by scipy):
```
f(x) = maxmin * y_pred_scaled(x) + lambda_reg * ||x_scaled - x_prev_scaled||²
```
- `maxmin = -1.0` for maximize objectives, `+1.0` for minimize
- `lambda_reg` = regularization parameter (default 0.05, user-adjustable 0.0–0.5 via slider)
- The penalty term penalizes large jumps from current operating point — prevents impractical recommendations
- Prediction and penalty both computed in scaled space (scaler loaded per model)

**Returns:** Dict with optimal control param values, `{param}_previous` vs `{param}_current` for all three model outputs, and optionally dependent variables from `build_bf_dependency_graph()`.

### Data Flow (`src/utils/recommendations/data.py`)

1. `df_hist` — Historical ML dataset CSV (`config['DATA']`, currently `V13_df_filtered.csv`)
2. `df_live_row` — Latest values from InfluxDB ML bucket (last 1 month, averaged to single row)
3. `df_full` — Concat of both; used for scaler fitting and feature engineering
4. Lag features created if the model expects them (configured by `TIMESTEPS: 3` in vsense config)
5. `UNITCOST LAKHS/THM` computed as derived column: `0.25 × (COKE_RATE + 0.53 × PCI_RATE)`

### LLM Review

After optimization, results are sent to OpenAI Responses API with `code_interpreter` tool for a concise, numeric, actionable recommendation. System prompt: "senior blast furnace advisor — be concise, numeric, and actionable."

---

## 5. AI Copilot Page

**File:** `src/custom_pages/5_🤖_AI_Copilot.py`

Uses **OpenAI Responses API** (not LangChain) with `code_interpreter` tool enabled. `OPENAI_MODEL` env var (default: `gpt-5-mini`).

### Tab 1 & 2: Unit Cost & Burden Distribution Analysis

Static LLM analysis against best historical period (Apr–Jun 2024). Tabs differ in scope:
- **Tab 1**: Fuel cost drivers — wind parameters, flux, ash%, alkalis, PCI sensitivity. Sections: high-confidence drivers, counter-moving levers, action checklist.
- **Tab 2**: Burden distribution impact on unit cost. Key empirical findings (R²≈0.43 OLS on ~5,900 rows):
  - More NON-COKE portions → lower cost (strongest effect)
  - More COKE portions → higher cost
  - NON-COKE angle outward → lower cost
  - Common best pattern: ~10–11 coke portions, ~8 non-coke portions, NON-COKE weighted angle ~28° with ≥25% outer share

### Tab 3: Channeling Analysis (Demo Model — UAT model pending merge)

**Channeling** = asymmetric/preferential gas flow through part of the burden, causing uneven thermal/chemical profiles. Detected from temperature spread at the 18660mm (Stack) sensors.

**`compute_propensity_suite()`** (`src/utils/anomaly_propensity.py`) — fetches last 8 hours of online data (15-min average) and computes 4 propensity scores:

| Metric | Signal used | Description |
|---|---|---|
| **Channeling** | 18660mm sensor spread | Std dev across `temp_18660_a/b/c/d`; high spread = asymmetric gas flow |
| **ΔP instability** | `body_dp_total` | Pressure differential anomaly |
| **Permeability instability** | `body_perm` | Permeability index anomaly |
| **Top pressure instability** | `top_press_avg` | Top pressure variability |

**Score formula** (each metric, 0–100):
```
score = 100 × (0.6 × mag + 0.4 × var)

mag = min(1.0, |z_max| / z_threshold)     # magnitude component
var = min(1.0, z_std / zstd_threshold)     # variability component
```
- z-scores computed on rolling window vs historical baseline
- **Alarm condition:** z_abs_max ≥ 2.5 AND z_std ≥ 1.0
- Each metric returns: `score_0_100`, `alarm` (bool), `alarm_reason`, `z_abs_max_last_hour`, `z_std_last_hour`

The full channeling model (UAT) will replace or extend this demo when merged. The demo is intentionally simple — production model will incorporate more features and higher predictive accuracy.

**Anomaly LLM analysis** (`build_anomaly_prompt()`): sends the 8-hour sensor snapshot to LLM which checks for temperature spikes, heatload excursions, gas/pressure instability, blowdowns/startups, and maps findings to actionable controllables (HB temp/vol/pressure, O₂, steam, PCI, burden quality).

### Operator Feedback Section

Thumbs up/down voting with optional text feedback on AI responses. Currently stored in session state; persistence to vector DB is a future TODO.

---

## 6. FurnaceMind — AI Co-Operate Chatbot

**Files:**
- Page: `src/custom_pages/7_🧠_FurnaceMind.py`
- Agent loop: `src/agents/furnacemind/agent.py`
- Tools: `src/agents/furnace_tools.py`
- Skills: `src/agents/furnacemind/skills.py`
- System prompt: `src/agents/furnacemind/context.py`
- UI components: `src/ui/furnacemind_sections.py`
- LLM client: `src/agents/llm/llm_client.py`
- Prompts: `src/utils/prompts.py`

### Agent Architecture

**Custom tool-calling loop** (`agents/furnacemind/agent.py`) with **OpenRouter** (OpenAI-compatible API), max 8 iterations per turn. Supports reasoning models (DeepSeek-R1, MiniMax M2.5) — `<think>` blocks are stripped before display. **Not** LangChain AgentExecutor.

**Session state keys:**
- `fm_df` — Current DataFrame (set by fetch tools)
- `fm_df_meta` — Metadata about the fetched dataset
- `fm_fig` — Current Plotly figure (set by `execute_python_plot`)
- `fm_datasets` — Dict of all fetched datasets keyed by dataset_id
- `chat_history` — List of `{"role", "content", "type"}` dicts; type is "text" or "plotly"
- `shift_store` — QdrantVectorStore (shift summaries, 384-dim local embeddings)
- `knowledge_store` — KnowledgeVectorStore (uploaded docs, 1024-dim cloud embeddings)

### UI Layout

- **Left 60%**: Current dataframe (from CSV) + download button + Plotly chart
- **Right 40%**: Full chat history (text + embedded Plotly charts)
- **Bottom**: `st.chat_input()` full-width

### Tools (9 functions dispatched by `execute_openai_tool_call`)

| # | Tool | Description |
|---|------|-------------|
| 1 | `fetch_online_data` | InfluxDB live telemetry (up to 90 days, auto-windowed) |
| 2 | `fetch_offline_data` | Manual/shift report data (HM_SLAG, CHARGE, RAW_MATERIAL, DPR) |
| 3 | `merge_furnace_data` | Align + merge online + offline datasets on timestamps |
| 4 | `fetch_ml_data` | Load date-range slice from pre-merged ML dataset (hourly, IST) |
| 5 | `concat_datasets` | Concatenate datasets vertically (temporal union) |
| 6 | `load_static_shift_data` | Load 8-hour shift data from static ML dataset |
| 7 | `search_shift_history` | Semantic search on Qdrant shift summaries (384-dim) |
| 8 | `search_knowledge_docs` | Semantic search on uploaded operator docs (1024-dim) |
| 9 | `execute_python_plot` | Execute sandboxed Plotly code; result in `fm_fig` |

### LLM Client (`src/agents/llm/llm_client.py`)

Two clients:
- **`OpenRouterClient`** — used by FurnaceMind agent. Wraps OpenRouter API (OpenAI-compatible). Methods: `generate(system, user)`, `chat_completions(messages, tools, tool_choice)`. Sends HTTP-Referer and X-Title headers for OpenRouter tracking.
- **`OpenAIClient`** — used by AI Copilot and Recommendations. Wraps OpenAI Responses API + Chat Completions fallback. Methods: `generate(system, user)`, `generate_with_tools(messages, tools, tool_choice)`.

`get_llm_client(prefer)` returns `OpenRouterClient` by default.

---

## 7. Material Balance Visualiser

**File:** `src/custom_pages/6_⚖️_Material_Balance.py`

Single-date element balance showing how many tonnes of each element (Fe, C, Si, Ca, Mg, Al, Mn, S, P, O, N, H) entered the furnace via raw materials + blast + steam and how many tonnes left via hot metal + slag + top gas. Surfaces accuracy gaps in raw-material reporting and provides an empirical baseline before finer streams (dust catcher, sludge, granulation losses) are added.

### Files

| Path | Role |
|---|---|
| `src/custom_pages/6_⚖️_Material_Balance.py` | Streamlit page — UI orchestration only |
| `src/utils/material_balance/__init__.py` | Package init |
| `src/utils/material_balance/constants.py` | Atomic weights, oxide→element table, `MaterialSpec` registry |
| `src/utils/material_balance/data_sources.py` | Day-window fetchers for RM, HM/Slag, DPR, online `process_params` |
| `src/utils/material_balance/compute.py` | Pure element-balance math + `run_full_balance(day)` entry point |
| `src/utils/material_balance/dpr_mapping.py` | DPR field discovery + yml load/save |
| `src/plotters/material_balance_plots.py` | Sankey, per-element bars, furnace diagram, closure styler |
| `src/config/material_balance.yml` | Constants, ash assumptions, DPR field mapping, future-stream hooks |

### Data Flow

```
date_picker (default = yesterday IST, max = yesterday)
        │
get_day_window_utc(date) → (start_utc, end_utc)
        │
        ├── fetch_rm_for_day          → clean_rm_data → 1-row composition DF (3-shift avg)
        ├── fetch_hm_slag_for_day     → 1-row chemistry DF (24-hour avg)
        ├── fetch_dpr_for_day         → raw DPR row(s); apply_dpr_mapping → mass dict
        └── fetch_online_aggregates   → dict {hot_blast_vol, o2_enr, steam, co_pct, …}
                │
                ▼
        compute.run_full_balance(day)
                │
        ┌───────┼─────────────────┐
        ▼       ▼                 ▼
    inputs   outputs        closure_table
                │
        ┌───────┼────────────────┬──────────────────┐
        ▼       ▼                ▼                  ▼
   build_sankey  build_per_     style_closure_     build_furnace_
                 element_bars   table              diagram
```

### Element Conversion Algorithm

For each raw material, `material_to_elements(mass_t, row, spec, ash_assumptions)` walks the `MaterialSpec.composition` dict. Each column maps to a `(token, kind)`:

- **`"direct"`**: column reports the element wt% directly → `out[token] += mass_t × pct/100`
- **`"oxide"`**: column reports an oxide wt% → split into element + O via `OXIDE_TO_ELEMENT_MASS_FRAC`
- **`"H2O"`**: moisture → split into H + O by molecular weight
- **`"ASH"`**: ash% → distributed among oxides using a constant assumption from `material_balance.yml` (`coke_ash_assumption_pct` or `pci_ash_assumption_pct`)
- **`"LOI"`**: loss-on-ignition → dropped in v1 (TODO: split into CO₂ + H₂O)

### Gas-Phase Math

**Blast (O + N from air + O₂ enrichment):**
```
o2_flow = (wind × (20.8 + enr)/100 − 0.208 × wind) / 0.792
air_only = wind − o2_flow
O_air_t  = air_only × 24 × 1.293 × 0.232 / 1000
N_air_t  = air_only × 24 × 1.293 × 0.755 / 1000
O_enrich = o2_flow × 24 × (32 / 22.414) / 1000
```

**Steam (H + O):**
```
H_steam_t = steam_kgh × 24 × (2.016 / 18.015) / 1000
O_steam_t = steam_kgh × 24 × (16.0 / 18.015) / 1000
```

**Top gas (C + O + H + N):** uses `bosh_vol_from_formula` from `utils/recommendations/dependencies.py` to estimate total top-gas volume, then applies CO/CO₂/H₂/N₂ percentages from online `process_params`. Runtime sanity check caps volume at 10× daily wind.

### MaterialSpec Registry (`constants.py`)

7 input materials registered with their InfluxDB field names:

| Material | Mass field | Key composition columns | Ash assumption |
|---|---|---|---|
| Coke | `coke_mt` | `coke_fc_pct` (C), `coke_ash_pct`, `coke_moist_pct` | `coke` |
| Nut Coke | `nutcoke_prime_mt` | `nutcoke_fc_pct` (C), `nutcoke_ash_pct`, `nutcoke_moist_pct` | `coke` |
| PCI | `pci2_mt` | `pci2_fc_pct` (C), `pci2_ash_pct` | `pci` |
| Ore | `ore_mt` | `ore_fe_total_pct` (Fe), `ore_sio2_pct`, `ore_cao_pct`, … | — |
| Sinter | `sinter_mt` | `sinter_fe_total_pct`, `sinter_sio2_pct`, `sinter_feo_pct`, … | — |
| Pellet | `lloyds_pellet_mt` | `lloyds_pellet_pct_fe2o3`, `lloyds_pellet_pct_sio2`, … | — |
| Flux | `flux_mt` | `flux_sio2_pct`, `flux_fe2o3_pct`, `flux_cao_pct`, … | — |

### Output Streams

- **Hot Metal**: `chem_pct_fe`, `chem_pct_c`, `chem_pct_si`, `chem_pct_mn`, `chem_pct_p`, `chem_pct_s`, `chem_pct_ti` — all direct element wt%
- **Slag**: `slag_pct_sio2`, `slag_pct_cao`, `slag_pct_mgo`, `slag_pct_al2o3`, `slag_pct_feo`, `slag_pct_mno`, `slag_pct_k2o`, `slag_pct_na2o`, `slag_pct_tio2` (oxides) + `slag_pct_s` (direct)
- **Top Gas**: C, O, H, N from CO/CO₂/H₂/N₂ gas composition
- **Unaccounted**: placeholder at 0 t in v1 (future: dust catcher, sludge, granulation)

### DPR Field Mapping

DPR column names are not documented. The page exposes a one-time mapping UI that lists every column found on a sample DPR row and persists choices to `material_balance.yml`. Nine canonical fields: `hm_mass_t`, `slag_mass_t`, `coke_mass_t`, `nut_coke_mass_t`, `pci_mass_t`, `ore_mass_t`, `sinter_mass_t`, `pellet_mass_t`, `flux_mass_t`.

When the mapping is incomplete, masses fall back to RM `*_mt` columns; HM falls back to `production_per_hour × 24`; slag falls back to `0.30 × HM`.

### UI Layout

- **Top row**: Date picker (default yesterday IST) | Refresh button | Overall closure KPI tile
- **Left 70%**: 3 tabs — Sankey | Per-element bars | Closure table
- **Right 30%**: Lightweight furnace cross-section diagram with labelled inflow/outflow arrows
- **Bottom**: Sankey-mode radio (Total mass / Element-focused), DPR mapping expander, Assumptions expander

### Closure Table

Per-element In_t / Out_t / Closure% / Delta_t with traffic-light row colours: green (95–105 %), yellow (85–115 %), red (outside). Thresholds configurable in `material_balance.yml`.

### Caching

| Function | TTL | Notes |
|---|---|---|
| `fetch_rm_for_day` | 1 h | RM is 8-hourly |
| `fetch_dpr_for_day` | 1 h | DPR is daily |
| `fetch_hm_slag_for_day` | 10 min | Hourly source |
| `fetch_online_aggregates_for_day` | 10 min | 1 h-windowed `process_params` |

Refresh button calls `clear_day_caches(day)` to invalidate all four.

### Future-Extension Hooks

Each is a no-op function or null yml field today:

1. **Dust catcher / top-gas solid losses** — `compute_unaccounted_solids()` returns `{}` in v1; Sankey "Unaccounted" node already wired at 0 t
2. **Sludge** — `material_balance.yml → future_streams.sludge_t` placeholder
3. **Slag granulation loss** — separate output stream slot reserved
4. **LOI breakdown** — per-material `loi_split_pct = {CO2: x, H2O: y}` in yml
5. **Per-element historical trend** — future tab: loop `run_full_balance` over 30 days
6. **Hot blast humidity** — additional H + O input if a humidity field appears
7. **Lab ash chemistry** — replace constant ash assumption with monthly lab updates

---

## 8. Architecture Summary

### Entry Points
- `run_streamlit.py` → `src/app.py` → authentication gate → `st.navigation()` using `page_registry.get_navigation_pages()`

### Pages

| Page | File | Purpose |
|---|---|---|
| Welcome | `1_Welcome.py` | Dashboard landing |
| Data Explorer | `2_Data_Explorer.py` | Browse InfluxDB data; build/manage ML dataset |
| Data Visualisation | `3_Data_Visualisation.py` | Temperature + heatload contour plots |
| V-OptimAIse | `4_Recommendations.py` | ML optimizer for blast parameters |
| AI Copilot | `5_AI_Copilot.py` | Channeling analysis, unit cost review, anomaly LLM |
| Material Balance | `6_Material_Balance.py` | Per-element daily mass balance (12 elements, Sankey + bars + closure table) |
| FurnaceMind | `7_FurnaceMind.py` | Custom tool-calling agent with 9 data/plot tools |
| Feedback | `8_Feedback.py` | Feedback ticket board (SQLAlchemy, SQLite/PostgreSQL) |
| Blend Mix Optimiser | `9_Blend_Optimizer.py` | Ore + flux LP blend at a fixed production (BMO) |
| Coke Energy Model | `10_Coke_Energy.py` | Two-zone (Rist) energy balance per shift; coke burnt vs charged vs modelled |
| Burden & Fuel Optimiser | `11_Burden_Fuel_Optimiser.py` | Ore + flux + fuel blend and production level on the energy model (section 11) |

### Key Supporting Modules

| Module | Path | Role |
|---|---|---|
| `DataframesProcessor` | `src/utils/recommendations/data.py` | Merges historical CSV + live InfluxDB for optimizer |
| `run_optimiser` | `src/utils/recommendations/optimiser.py` | Differential evolution optimizer |
| `compute_propensity_suite` | `src/utils/anomaly_propensity.py` | 4 channeling/instability z-score metrics |
| `furnace_tools` | `src/agents/furnace_tools.py` | 9 tool functions (fetch, merge, search, plot) |
| `agent.py` | `src/agents/furnacemind/agent.py` | Custom tool-calling loop + reasoning-model cleanup |
| `QdrantVectorStore` | `src/agents/memory/vector_store.py` | Shift summary semantic search (384-dim, cosine) |
| `StructuredStore` | `src/agents/memory/structured_store.py` | JSON persistence for shift/daily/weekly summaries |
| `Settings` | `src/utils/settings.py` | Pydantic singleton for all FurnaceMind config |
| `OpenRouterClient` | `src/agents/llm/llm_client.py` | LLM wrapper (FurnaceMind agent) |
| `TicketService` | `src/data/tickets/service.py` | Ticket CRUD + audit events (Feedback page) |
| `TicketRepository` | `src/data/tickets/repository.py` | SQLAlchemy ticket persistence |
| `page_registry` | `src/config/page_registry.py` | Central page navigation registry |
| `run_full_balance` | `src/utils/material_balance/compute.py` | Element-balance math: material→element, gas-phase, closure table |
| `MaterialSpec` | `src/utils/material_balance/constants.py` | Declarative spec per raw material (composition → element mapping) |
| `material_balance_plots` | `src/plotters/material_balance_plots.py` | Sankey, per-element bars, closure styler, furnace diagram |

### Authentication
- Cookie-based sessions (`streamlit-cookies-manager`, prefix `bf_dashboard_`)
- `src/utils/session.py` — `is_logged_in()`, `is_admin()`, `is_supervisor()`
- Three roles: `admin`, `supervisor`, `user`

### Configuration Files
- `src/config/setting_ds_dv.yml` — InfluxDB mappings, furnace geometry, sensor layout
- `src/config/setting_vsense.yml` — V-OptimAIse: 3 models, control/input/output params, `LAMBDA_REG`, `OPTIM_STEPS`, `TIMESTEPS`
- `src/config/materials.yml` — Hoppers, materials, burden fields (for PostgreSQL `Database`)
- `src/config/material_balance.yml` — Element list, ash assumptions, DPR field mapping, closure thresholds, future-stream hooks
- `src/config/page_registry.py` — Central page navigation registry (`PAGE_REGISTRY` tuple)
- `src/config/logger_setting.yml` — Logging configuration (YAML-based)
- `.env` — All secrets

### Key Patterns
- **SCD Type-2** for PostgreSQL history tables — `valid_upto IS NULL` = current row
- **Data fetcher adapter** — all InfluxDB fetchers extend `BaseDataFetcher`; mappings from `data_mapping` in yml
- **Settings singleton** — `from utils.settings import settings` for all FurnaceMind config
- **Page registry** — `src/config/page_registry.py` defines all pages; `app.py` uses `get_navigation_pages()`
- **Tickets layered pattern** — `data/tickets/` follows service → repository → models (SQLAlchemy 2.0 ORM)
- **Shift windows** — always 8-hour A/B/C; `ShiftBuilder` creates them, `ShiftAnalyzer` processes
- **KPI priority** in all LLM prompts — Fuel Rate → ETA CO → Production Rate → Quadrant heat loads; ACTION/REASON/MAGNITUDE discipline enforced

## 9. Feedback & Support Desk (Page 8)

**File:** `src/custom_pages/8_Feedback.py`

A shared ticket board for operators and engineers to report bugs, suggest improvements, and track resolution.

### Data Layer (`src/data/tickets/`)

| File | Role |
|---|---|
| `models.py` | SQLAlchemy 2.0 ORM: `Ticket`, `TicketEvent`, `TicketImage` + enums |
| `engine.py` | Engine/session factory; SQLite default at `src/storage/feedback/tickets.db` |
| `repository.py` | Low-level CRUD (session-scoped) |
| `service.py` | Business logic + Pydantic view models (`TicketService`, `TicketView`, etc.) |
| `__init__.py` | Public API re-exports |

### Enums
- `TicketCriticality`: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- `TicketStatus`: `OPEN`, `IN_PROGRESS`, `RESOLVED`, `DEPENDENCY_CONFLICT`, `CLOSED`

### Tables (3)
- `tickets` — main ticket record (page_name, criticality, description, status, audit fields)
- `ticket_events` — audit trail (status transitions + comments)
- `ticket_images` — screenshot metadata (file path + original filename)

### Authorization
- All logged-in users can create tickets and view the board
- Only `admin` / `supervisor` roles can update status, delete tickets, or access the management panel

### Storage
- SQLite by default: `src/storage/feedback/tickets.db`
- Override via `TICKETS_DB_URL` env var for PostgreSQL
- Schema auto-created via `Base.metadata.create_all()` (no Alembic)

### UI Helpers
- `src/utils/feedback_page.py` — `render_board()`, `render_overview_kpis()`, `render_management_panel()`
- `src/assets/css/feedback_style.css` — ticket card styling

---

## 10. Neon DB Offline Data Layer (branch: `109_data_feed_neon`)

All "offline" manual-entry data has migrated from InfluxDB (`bf2_evonith_offline_utc`) to **Neon PostgreSQL**. The shared library lives in `furnace_data/furnace_data/neon_db/` and is consumed by the webapp, the FastAPI sidecar, and the `DatasetService` pipeline.

### PostgreSQL Schema Structure (4 schemas)

| Schema | Purpose |
|---|---|
| `offline_feed` | Operational offline data — charge, DPR, HM/Slag, raw material chemistry, static ML dataset |
| `ops_config` | Operator configuration snapshots — burden distribution, hopper material assignments |
| `plant_master` | Reference data — materials, hoppers, units, material categories |
| `identity` | User auth — users, user_roles |

### Key Tables (`furnace_data/furnace_data/neon_db/neon_tables.yml`)

**`offline_feed` schema:**
| Table | Time col | Aggregatable | Key columns |
|---|---|---|---|
| `charge_data` | `date_time` | yes | sinter/pellet/ore/flux/coke/nut_coke/pci _mt (1..N variants) |
| `dpr_data` | `date_time` | yes | same + dust/slag/hm mass columns |
| `hot_metal_slag_analysis` | `date_time` | yes | chem_pct_c/mn/si/s/p/ti/fe, slag_pct_sio2/cao/mgo/al2o3/feo/s/na2o/k2o/tio2 |
| `ore_chemistry` | `date_time` | yes | material_code, fe_t, sio2, al2o3, cao, mgo, loi, tm, p, tio2, na2o, k2o, mno |
| `sinter_chemistry` | `date_time` | yes | material_code, fe_t, feo, sio2, al2o3, cao, mgo, basicity |
| `fuel_chemistry` | `date_time` | yes | material_code, tm, moisture, ash, vm, fc |
| `flux_chemistry` | `date_time` | yes | material_code, cao, mgo, sio2, al2o3, fe2o3, loi, tm |
| `raw_material_strength_analysis` | `date_time` | yes | ai, ti, rdi, ri |
| `raw_material_stock` | `date_time` | yes | material_code, stock_mt |
| `v_charge_material_quantities` | `date_time` | no | charge_data_id, material_code, quantity, unit_code |
| `v_dpr_material_quantities` | `date_time` | no | dpr_data_id, material_code, quantity |
| `feed_material_columns` | — | no | feed_name, source_column_name, material_code |
| `historical_static_ml_dataset` | `date_time` | no | allow_all_columns: all ML feature columns |

**`ops_config` schema:**
| Table | Description |
|---|---|
| `burden_history` | Wide snapshot: coke/noncoke _p01..p11 _rings/_angles, discharge_time, charge_pattern, purpose |
| `hopper_raw_material_history` | Wide snapshot: hopper_01..hopper_19 material codes |

**`plant_master` schema:** `materials`, `hoppers`, `units`, `material_categories`

### Logical Report Types (8 — maps multiple tables)

| Report Key | Tables Joined |
|---|---|
| `HM_SLAG` | `offline_feed.hot_metal_slag_analysis` |
| `CHARGE` | `offline_feed.charge_data` |
| `DPR` | `offline_feed.dpr_data` |
| `RM_COMPOSITION` | ore_chemistry + sinter_chemistry + fuel_chemistry + flux_chemistry + raw_material_strength_analysis + plant_master.materials |
| `BURDEN_DISTRIBUTION` | `ops_config.burden_history` |
| `HOPPER_MANAGEMENT` | `ops_config.hopper_raw_material_history` |

### Query Types

| `query_type` | Description |
|---|---|
| `ts` / `raw` | Raw time-series rows, ordered by time |
| `average` | Single average row over entire window |
| `windowed-average` | Time-binned averages via PostgreSQL `date_bin()` |
| `hourly-average` | Alias for windowed-average with 1-hour bin |

### Core Library (`furnace_data/furnace_data/neon_db/offline.py`)

```python
fetch_offline_data(table_name, time_range, query_type, window, columns, database_url) -> pd.DataFrame
fetch_offline_report(report_type, time_range, query_type, window, database_url) -> pd.DataFrame
get_offline_table_bounds(table_name) -> (start, end, count)
get_offline_report_bounds(report_type) -> (start, end, count)
resolve_neon_table_name(alias_or_full_name) -> canonical_name
list_neon_offline_tables() -> dict  # JSON-serialisable whitelist snapshot
```

`time_range` accepts: preset string (e.g. `"last 1 week"`), `"full"` (2023-01-01 → now), or `(start, end)` tuple.

Table whitelist enforced: unknown columns raise `ValueError`. `non_averaged_columns` (id, import_batch_id, date_time, etc.) are excluded from AVG aggregates.

### 4-Step Interactive ML Dataset Pipeline (`furnace_data/furnace_data/dataset/service.py`)

`DatasetService` feeds the interactive date-range selector in the Data Explorer:

- **Step 1 (`fetch` / `fetch_step1`)** — historical static: queries `offline_feed.historical_static_ml_dataset` for the pre-cutoff range
- **Step 2 (`fetch_rm_data` / `fetch_step2`)** — post-cutoff: fetches `charge_data` or `dpr_data` + `raw_material_strength_analysis` + weighted chemistry (ore/sinter/fuel/flux via `v_charge/dpr_material_quantities` and `merge_asof` to match latest lab sample before each charge)
- **Step 3 (`fetch_hotmetal_hourly`)** — `hot_metal_slag_analysis` interpolated onto a regular hourly grid
- **Step 4 (`fetch_distribution_data`)** — `ops_config.burden_history` via ORM, pivoted to daily rows with derived `total_coke_portions`, `weighted_coke_angle`, etc.

`DatasetFetcher` (`fetcher.py`) wraps `DatasetService` with a range-aware in-memory cache (`RangeCache`) and incremental fetch logic (only fetches missing date slices).

### Static ML Dataset (from Neon)

`src/data/ml/static_csv.py::fetch_static_dataset_from_database()`:
- Queries `offline_feed.historical_static_ml_dataset` with `query_type="raw"`
- Selects only columns present both in the DB table and in `setting_ds_dv.yml` cleaning config
- Normalises index: UTC → IST (Asia/Kolkata), tz-naive
- Applies `rename_dict` from config

`StaticDatasetManager` (`src/data/ml/static_dataset_manager.py`):
- Maintains a rotating local CSV cache (max 3 versioned files, e.g. `furnace_dataset_20260512_...csv`)
- `update_static()` → fetch from DB → `DataCleaner.clean()` → `save()` (atomic copy + metadata)
- Background refresh via `dataset_refresher.py::maybe_refresh()` — triggers if cache > 6 hours old

### FastAPI Sidecar (`furnace-data-service/`)

New REST endpoints (replaces old `offline_fetcher.py`):

| Method | Path | Description |
|---|---|---|
| `POST` | `/data/offline/fetch` | Fetch offline data by report type or explicit table name |
| `GET` | `/data/offline/report-types` | List report types → table mappings |
| `GET` | `/data/offline/neon-tables` | Full whitelist: tables, columns, aliases |
| `POST` | `/data/rm/live` | Latest RM composition (`RM_COMPOSITION`, last N days) |
| `POST` | `/data/online/fetch` | Online InfluxDB fetch (unchanged) |
| `GET` | `/data/online/measurements` | List InfluxDB measurements (unchanged) |

`OfflineFetchRequest` schema:
```python
report_type: OfflineReportType  # HM_SLAG, CHARGE, DPR, RM_COMPOSITION, BURDEN_DISTRIBUTION, HOPPER_MANAGEMENT
table_name: Optional[str]       # explicit table override (alias or schema-qualified)
preset: Optional[str]           # e.g. "last 1 month"
start_time / end_time: Optional[datetime]
query_type: QueryType           # ts, windowed-average, average
window: Optional[str]           # PostgreSQL interval, e.g. "1 hour"
format: ResponseFormat          # json (default) or csv
```

### Data Explorer Integration (`src/custom_pages/2_Data_Explorer.py`)

- Imports `NEON_OFFLINE_REPORT_MAP as OFFLINE_DATABASE_REPORT_MAP` and `NEON_OFFLINE_TABLES as OFFLINE_DATABASE_TABLES` from `furnace_data.neon_db.offline`
- Two browse modes: **"Logical report"** (uses report map) and **"Table"** (explicit table from whitelist)
- `OFFLINE_REPORT_LABEL_MAP` in `src/data/fetch_presets.py` provides UI labels for all 8 report types + legacy aliases

### FurnaceMind Tool Integration

`fetch_offline_data` tool in `src/agents/furnace_tools.py` routes through `furnace_data.neon_db.offline` (`_fetch_neon_table_df`, `_fetch_neon_report_df`). The `RAW_MATERIAL_COMPOSITION` alias is mapped to `RM_COMPOSITION` internally.

---

## 11. Burden & Fuel Optimiser (Page 11)

**File:** `src/custom_pages/11_Burden_Fuel_Optimiser.py` · full guide `docs/burden_fuel_optimiser.md` · epic #260

Optimises the ferrous blend and flux for **ore + flux + fuel** cost per tHM and the **production level** under the
day's limits. Coke for every candidate blend comes from the two-zone (Rist) balance in `utils/coke_energy`
(page 10's model, unchanged), calibrated on the last 14 days of stable shifts. PCI and nut coke are fixed inputs per
run; their page defaults are the regime window's medians (`scenarios.fuel_defaults`; ASM-08 values 190 / 70 only as
fallback).

- **Modes:** Step 1 cheapest production level (`Engine.min_total_cost` + frontier), Step 2 fixed production
  (`min_cost(fuel=True)` next to `fuel=False`, the page-9 objective), maximum production (`max_production`).
- **Model check** (top of the page): latest shift's burnt coke next to the energy- and oxygen-balance predictions at
  its HMT / Si, the window record and a trend (`utils/burden_engine/model_check.py`, from the Coke Energy pipeline's
  `shifts` table; thresholds in `burden_optimiser.yml → model_check`). Step 2 also prices today's charged blend
  (`run_today`, last 24 h of `charge_data`) and shows the saving against it. A second tab compares burnt with charged
  coke per averaging window (`utils/coke_energy/transit.py`, CCC and Bland–Altman; `docs/burnt_vs_charged_coke.md`).
- **Layout and commentary:** results follow page 9 (overview strip, Cost / Fuel / Slag / Charging panels with the plant
  bands from `utils/slag_bands.py`, blend table beside a donut). *Generate commentary* sends the optimiser notes
  (`src/assets/burden_engine/optimiser_notes.md`), the fact register as a table and the run to the LLM; context is
  built in `utils/burden_engine/commentary.py` (pure, tested), the panel is `ui/burden_optimiser/commentary.py`.
- **Engine:** `src/utils/burden_engine/` — pure numerical (no Streamlit). Sequential LP (HiGHS) re-linearising the
  balance; blast limit (Pb² − Pt²) = K·V^1.8 with burden-resistance rules; RAFT from `utils.energy_balance.raft`.
- **Shaft efficiency:** Fixed-ω (Rist baseline, held at the calibration; ASM-01, `Settings.omega_mode="fixed"`, the
  engine default) or STRATA-ω (ASM-18, page default): `utils/burden_engine/omega_model.py` +
  `assets/burden_engine/omega_model.json` (**STRATA 1.5** — Stratified Transport & Reduction Analytics, formerly OMEGA:
  STRATA-Flow ring counter-current model of the thermal reserve zone; RI and RDI from the Evonith laboratory tests
  (set T26, GAS-21; Gomti 65 % without test), rate law [ln(1−RI)/ln(1−RI_sinter)]^β (11.25/d)^n with β 0, n 1.0
  calibrated on the plant data; plant-data scales for pellet (1.42 ± 0.45) and lump ore (0.92 ± 0.46), an
  other-operation channel (coke layers, fuels, O₂, blast; 0.80 ± 0.28) and a nut-coke term (0.0010 ± 0.0003 per
  kg/tHM) from an out-of-sample-tested AR(1) posterior, GAS-13..26; **no production channel** since 1.4
  (`operating_scales.production_channel: false`, scale 0: it failed the strict 2023–24 test, GAS-16 retired, GAS-25);
  **no moisture term** since 1.5 (`moisture.coef` 0: the day's Open-Meteo moisture acts through the balance, GAS-19
  retired); gas-capacity channel retired), solved inside the balance via `rist.solve_2z(omega_fn=)`. Validation (Open-
  Meteo daily moisture): 2026 expanding test RMSE 5.99 vs 6.39 kg/tHM for Fixed-ω (coke skill +0.120; STRATA 1.4 on the
  RH-scaled record 6.44 vs 6.94); strict 2023–24 test with frozen parameters, 411 days, RMSE 8.00 vs 8.23
  (`validation.strict_2023_24` in the asset; on the smoothed day-of-year moisture 1.4 gave 7.40 vs 7.95).
  **No surrogate:** the ring quantities
  come from the STRATA-Flow network itself (`src/utils/strata_flow/`, vendored bf2_momentum v0.4.1, ~1.1 s per solve)
  through `utils/burden_engine/strata_network.py`: a solve at the calibration plus its Jacobian (16 solves, cached in
  `src/storage/strata_flow/` per calibration and catalogue), one solve per SLP iterate (plan blend), the plan's three
  states (blend, blend + production, full operation) before a final LP step and again for the reported plan; the
  Jacobian is used only between solves and each result reports its solves and distance from the nearest solve. The
  production state is still solved: the other-operation channel is defined against it, and `d_prod_phys` is reported.
  Production searches reuse solves within `Engine.SEARCH_TOL` (1 scaled unit) and `Engine.refine` re-solves the plan
  they return exactly. When the first STRATA-ω linearisation has no feasible LP, `Engine.slp` re-starts from the
  Fixed-ω optimum of the same case; inside the searches a point that stops with a small limit breach on coarse solves
  (total < `Engine.SEARCH_RETRY_VIOL`, 0.1), and any failure at the low end of a search, is re-solved exactly before it
  counts as infeasible (`Engine._try_search`). Parallel solves (SV-1.8): independent states go through
  `NetworkParts.solve_many` / `anchor_many` (the Jacobian, a plan's three states, the twin window's three states), and
  the searches' independent items through `Engine.run_items` (Step 1: oxygen grid and bracket rounds of both oversize
  options together, `bracket_rounds`, `o2_points` 4 per round, replacing the golden section over oxygen, then the
  frontier levels of both options; maximum-production grid; value-of-limits relaxations): every item starts from the anchors present before the items, the anchors are
  merged in item order, so serial and parallel give the same plans, anchors and solve counts (a serial run solves more
  than SV-1.7: Step 2 73 vs 52); executor `strata_network.start_solvers(n)` / `solver_processes(n)` (`BF_STRATA_SOLVERS`,
  else min(8, cpus); `__main__` empty while they start; the forkserver builds the STRATA-Flow context once via
  `strata_preload` when `src` is on `PYTHONPATH`; a context variable per thread / asyncio task; a pool that fails
  leaves the solves serial with a warning), used by `services.optimiser.solve_case` (the worker's and page 11's
  STRATA-ω runs), the nightly Jacobian and twin backfills of ≥ 3 windows; the anchors of a calibration are locked
  (`NetworkParts.lock`, held by `Engine.run_items`); benchmark `scripts/optimiser_bench.py`.
  `OmegaModel.load(backend="surrogate")` keeps the STRATA 1.2
  surrogate for diagnostics only. Two oversize types: ore_6 Lloyds fines oversize (reduction rate at the Lloyds CLO
  size, no size credit) and ore_11 Donimalai oversize return (= Donimalai). Blend mode adds σ(coke), certainty tiers High / Medium / Low / Investigate (pellet > 25 %
  or σ > 8 kg/tHM; GAS-18), a certainty filter (`Settings.certainty`, page default "low" = all but Investigate), risk
  aversion (`Settings.risk_k`), critical assumptions per plan (`Engine.critical_assumptions`; no production row in 1.4)
  and auto mode (`scenarios.run_candidates` + `auto_review.py`: fixed rule picks, never an Investigate plan; the LLM
  only reviews). Fact-register groups MOD (Fixed-ω vs STRATA-ω responses) and VAL (per-blend out-of-sample) are
  generated by `python -m utils.burden_engine.model_facts --write`.
- **STRATA-Learn (shadow mode):** `utils/burden_engine/learn/` — evidence store (seed
  `assets/burden_engine/strata_evidence_seed.csv`), random-walk Kalman filter on the plant-data scales, champion /
  challenger gate (60 days, Diebold–Mariano), CUSUM drift alarm, register proposals (automatic within 2 se; status
  changes need approval), Thompson-sampling exploration under a regret budget. `python -m utils.burden_engine.learn.run`.
  Never changes the asset, the register or the page. λ_production (STRATA 1.4) and a_moisture (STRATA 1.5) are held
  at 0 (prior sd 0, no random walk; `channel_scales.fixed`); the q_prod and d_humidity_g columns stay in the evidence
  for a later re-test. Seed rebuilt on the STRATA 1.5 channels (191 days).
- **Prices and slag credit:** ferrous and flux prices from the plant (23 Oct 2026), fuels from the BF-2 cost sheet
  (FTD Costing 22 Oct 2026) — DATA-14 — in `bmo_operator_inputs.yml` (live) and `burden_optimiser.yml → prices`
  (reference yard). The objective credits the slag sold (`fuel["slag_price"]`, granulated slag 457.85 Rs/t, ASM-20):
  cost `total` = ore + flux + fuel − slag credit (`gross` before the credit).
- **Sinter RI / RDI:** from the latest raw-material strength data (calibration `sinter_strength`, pipeline columns
  SINTER_HOT_STRENGTH_RI / _RDI), not from the laboratory report.
- **Blast moisture:** Open-Meteo daily mean RH at Wardha (`utils/coke_energy/weather.py`, DATA-11, STRATA 1.5):
  default `open_meteo` — daily RH mapped linearly to 7–21 g/Nm³ (RH 26.1 → 7, 86.2 → 21), no smoothing; validated on
  the top-gas H₂ (+0.041 pt H₂ per g/Nm³ in 2026, +0.038 in 2023–24, the physical slope; DATA-16). Shipped file
  `assets/coke_energy/open_meteo_rh_wardha_daily.csv` (1 Jan 2023 – 15 Oct 2026); later days from the Open-Meteo
  forecast endpoint (`blast_moisture.open_meteo`: refresh 6 h, failure retry 60 min, cache
  `src/storage/coke_energy/`; `COKE_ENERGY_WEATHER_LIVE=0` disables the fetch, set in `tests/conftest.py`); missing
  days use the Open-Meteo day-of-year mean, then the monthly table. `rh_scaled` (31-day centred RH of the 2025–26
  weather record, the STRATA 1.4 basis), `psychrometric` and `table` remain selectable in
  `config/coke_energy.yml → blast_moisture.source`. Used everywhere blast moisture enters (user decision 25 Oct 2026):
  the Coke Energy pipeline (`run_shifts.case_from_shift`, each shift's day); a calibration (its window's Open-Meteo
  mean; the shipped `cal_2z_jul_oct_2026.json` is re-solved at 18.45 g/Nm³ instead of 15); every plan (the plan day's
  value, fetched live: `weather.plan_blast_moisture`, the engine `Settings()` default, page 11 default; today's value is
  fetched again every 6 h); the Blend Mix Optimiser energy balance and RAFT (`energy_balance.blast_moisture_g_per_nm3:
  null` → the day's value unless the operator enters one). `Engine.cal_settings()` = the calibration point (engine
  calls without settings). `COKE_ENERGY_TODAY=YYYY-MM-DD` fixes the plan day (tests: 2026-10-15). The 16 Oct case report
  (15 g/Nm³) is reproduced by the tests on `tests/fixtures/cal_2z_jul_oct_2026_case_report_15g.json`. The asset's
  `moisture.source` must match (tested).
- **Fact register:** `src/config/facts_register.json` v1.8, 116 facts (schema in `config/schemas/`). Every coefficient outside the
  balance, the plausibility bands (fuel rate 520–600 kg/tHM, HMT 1,440–1,520 °C, Si 0.20–0.80 %) and default limits.
  Change values there, not in code; retired facts are refused by `FactRegister.active()`.
- **Adapters:** `src/data/burden_engine/inputs.py` (yard from `EvonithBmoContextProvider` + `bmo_operator_inputs.yml`),
  `src/data/burden_engine/calibration.py` (live calibration from the Coke Energy pipeline result, fallback to
  `src/assets/burden_engine/cal_2z_jul_oct_2026.json`).
- **UI:** `src/ui/burden_optimiser/` (sections, cached loaders, frontier figure). Config `src/config/burden_optimiser.yml`.
- **Offline:** without Neon / InfluxDB the page uses the reference yard (16 Oct 2026) and the shipped calibration, labelled.
- **Tests:** `pytest tests/test_burden_engine*.py` reproduce the 16 Oct 2026 case report; `BURDEN_ENGINE_FAST=1` skips
  the slow production cases.

---

## 12. Svelte Frontend, twin-api and Service Layer (programme #262, milestone #M5)

Branch `feature/262_svelte_twin`; plan in the epic #262 and its phase epics #263-#269 (child issues SV-x.y = #270-#321).

| Part | Path | Rule |
|---|---|---|
| SPA | `frontend/` | SvelteKit 3 (config inside `sveltekit()` in `vite.config.ts`, no `svelte.config.js`), Svelte 5 runes, TypeScript 6; imports from `src/lib` as `#lib/...` **with the file extension** (`#lib/api/client.ts`; `$lib` is removed in Kit 3, and TypeScript does not resolve extensionless `#lib` paths); static build with `200.html` fallback |
| twin-api | `src/api/` | FastAPI, run with `src/` on the path (`uvicorn api.main:app --app-dir src`); routers (`auth`, `system`, `stream`, `jobs`, `twin`, `optimiser`, `reports`, `admin`, `status`, `operator_inputs`, `trace`, `furnacemind`, `bmo`; SV-1.6: twin now / series / transit / alarms, optimiser runs, latest plan, audited decisions; SV-2.3: `GET /api/reports`, `/api/reports/{id}`, `POST /api/reports/live`; SV-2.4: `/api/reports/thresholds`, report `kpis`; SV-2.1: `GET /api/status/snapshot`, `/api/status/trend`; SV-2.2: `/api/status/settings`; SV-4.1: `GET /api/admin/audit`, admin only; SV-4.2: `/api/admin/facts` register, versions, review, diff, export, admin only; SV-3.1: `GET` / `PUT /api/operator-inputs` (user / supervisor); SV-5.1: `POST` / `GET /api/fm/conversations[/{id}]`, `POST /api/fm/conversations/{id}/messages` as `text/event-stream` (user); SV-5.2: `GET /api/fm/skills`, ask with `skill`, `PUT /api/fm/conversations/{id}/messages/{seq}/feedback` (owner, audited `fm.feedback`), `GET .../datasets/{artifact_id}/csv`; store dependency `api.store`, views `services.twin_views`) call `services.*` only; errors are RFC 7807 problems (`api.errors.ApiError`); after changing a route run `make client` (or `python scripts/export_openapi.py && pnpm --dir frontend api:types`) and commit `frontend/src/lib/api/openapi.json` + `schema.d.ts` (a test fails on drift) |
| Service layer | `src/services/` | All data assembly and model orchestration shared by the API, the worker and Streamlit. **No Streamlit or `ui.*` import** (tests/test_services_architecture.py, also transitive) |
| Worker | `src/worker/` | `python -m worker`: arq 0.28 on Redis (`BF_REDIS_URL`); job kinds in `services.jobs.CATALOG` (role, time limit, retries, IST schedule, parameter model), handlers in `worker/kinds/` (`@handles("kind")`, `JobContext.progress`, CPU work through `ctx.run_cpu` in the process pool); events to the Redis stream `twin:events` (`services.events`, typed `StreamEvent`), forwarded by the API as SSE (`/api/stream`). Live forecasts (SV-1.5): `forecast.si` every 5 min, `forecast.charged_coke` at HH:07 (`services/forecasts.py`, `worker/kinds/forecasts.py`), same ledgers as page 9 (format unchanged; `utils/jsonl_log.py`: incremental reads, cross-process append lock); page 9 reads them per `setting_bmo.yml → bmo.forecast_issuer` (auto / worker / page); attempts in `attempts/YYYY-MM.jsonl` of each storage folder; gap check `scripts/forecast_gaps.py`. Schedule (SV-1.7, IST): `twin.evaluate` HH:10 (`worker/kinds/twin.py`: the window's last hourly row is in the build of about HH:06; retried to HH:25; backfill of 24 h; drift alarm; `twin.snapshot` event), `optimiser.step2` 06:15/14:15/22:15 (STRATA-ω and Fixed-ω in parallel, current production = latest twin window), `optimiser.step1` 06:30/14:30/22:30, `calibration.nightly` 02:30 (`services/day_calibration.py`: the day's calibration of every worker optimiser run + its network Jacobian; page 11 stays live), `learn.shadow` 03:00 (`services/learn.py`), `report.shift` 06:20/14:20/22:20 (SV-2.3, the closed shift; `report.live` on demand), `shift.close` 06:10/14:10/22:10 (SV-2.5, handover record; retried to HH:30), housekeeping 04:00 (runtime statistics in its result). Every scheduler job carries `scheduled_for` (its slot); `services/schedule.py`: slots, catch-up at start (latest slot within 30 min runs, other slots since the last heartbeat `twin:worker:alive` recorded as `scheduler (skipped)`), audit; `services/jobstats.py` (p50/p95 of `run_s`, the execution time over the tries); routes `GET /api/jobs/stats`, `/api/jobs/schedule`; check `scripts/schedule_audit.py --days 14` |
| Cache | `src/utils/memo.py` | `@memo(ttl_seconds=, max_entries=, copy=)` replaces `st.cache_data` outside Streamlit; key = argument values |
| Twin evaluation | `src/services/twin.py`, `src/utils/coke_energy/twin.py`, `src/utils/burden_engine/strata_window.py` | Window W = [t − 8 h, t) aggregated and balanced by the pipeline's statements (`build_shift_inputs(key_fn=window_key(...))`); 14-day quantities from the stable shifts dated in [d − 14 d, d) of the pipeline result; a window on a completed shift must equal page 10 (`tests/test_twin_window.py`). K_STRATA = the K_FORGE balance at ω_STRATA (anchor A = the same 14 days, states B / P / F of the window, network in exact mode: `OmegaModel.calibrate(..., jacobian=False)`). Drift alarm `services/drift.py` + `utils/coke_energy/drift.py`: CUSUM on AR(1) innovations of ω_imp − ω_STRATA between non-overlapping windows (lag 8 h), alarm `omega_drift` high / low, settings `coke_energy.yml → twin.drift`. Burden column `utils/coke_energy/burden_column.py` + `services/transit.py`: descent times, ledger of the charges by zone, coke in transit, implied lag (`twin.transit`). HMT forecast `utils/coke_energy/hmt_forecast.py` (gated: shown only with `twin.hmt_forecast.gate.status: passed`; gate run `scripts/hmt_gate.py`) |
| Blend Mix Optimiser | `src/services/bmo.py` | Page 9's orchestration (SV-3.4): `BlendCase`, `run_balanced` / `run_intensive` / `fall_back_to_lp`, `comparison_view`, `charged_prediction`, `si_forecast_for_page`, `take_snapshot`. Provider, model service, dataset, clock and live tags are arguments, so the Sandbox still swaps page 9's loaders. Golden page-9 runs: `tests/test_bmo_golden.py` (5 plant days, AppTest, about 4 min); rebuild with `python -m tests.bmo_golden --write` and review the JSON diff |
| FurnaceMind service | `src/services/furnacemind.py`, `src/agents/furnacemind/turn.py`, `tool_state.py` | SV-5.1: page 7's turn (`turn.run_turn`) shared by page 7 and the API; the tools' state is a bound mapping (`tool_state`, st.session_state inside Streamlit), so `agents/furnace_tools.py` has no Streamlit import. Conversations in the twin store; turns in a 4-thread pool (never on the event loop), one per conversation (409); SSE events message.user, token, tool.call, tool.result, artifact (Plotly JSON, dataset preview), message.assistant | error. Parity: `tests/test_services_furnacemind.py` (20 reference questions, page 7 under AppTest vs the API, scripted fake LLM `tests/furnacemind_fakes.py`). SV-5.3 twin tools (`agents/twin_tools.py`, in the schema list and dispatcher of `furnace_tools`): `get_twin_snapshot`, `get_twin_series` (also a dataset for the plot tool), `get_transit`, `get_optimiser_result` read the process's twin store through the routes' compositions (`services.twin_reads` now / gaps / series / run_summary, `services.transit.view`, `optimiser_results.result_view`); JSON with IST (+05:30) times, units, warnings and a `cite` line the answer quotes; routing in `prompts.TOOL_POLICY` and `storage/furnacemind/TOOLS1.md`; `tests/test_services_furnacemind_twin.py` SV-5.2 (page `/furnacemind`, `frontend/src/lib/furnacemind/`): page 7's quick skills (`SkillRegistry` / `SkillEngine`, now on `tool_state`, no Streamlit) run in the turn thread, their prompt stored as the question (`meta.display`); full dataset CSV while the process holds it (`artifact_id`, 12 per conversation); feedback up / down + comment in `twin.fm_feedback`; agent stack imported at API start (`BF_API_FM_WARM_UP`, 4.4 s otherwise on the first turn). SPA: SSE over `fetch` (`sse.ts`), safe Markdown (`markdown.ts`), Plotly.js (`plotly.js-dist-min` 3.7.0, plotly.js 3 schema as Python plotly 6) only through `import()` in `plotly.ts` (own 4.6 MB chunk, `plotly.test.ts`). Tests `tests/test_api_furnacemind_page.py` |
| TRACE page (SV-1.10) | `src/services/trace.py`, `src/api/routers/trace.py`, `frontend/src/routes/trace/`, `frontend/src/lib/trace/` | `/api/trace` (summary), `/series` (hour / shift / day), `/agreement`. Job `twin.trace` (HH:20) builds `src/storage/coke_energy/trace_view.json` from the Coke Energy page data (shift table = page 10, day = `accuracy.window_means` w 3, agreement = `transit.summary` + `transit.window_pairs`); per hour from the twin snapshots; a request never runs the pipeline. Freshness is assessed at the build (when the sources were read), the view's age separately (`age_s`, stale after 2 h) |
| STRATA page (SV-1.12) | `src/services/strata.py`, `src/api/routers/strata.py`, `src/worker/kinds/strata.py`, `frontend/src/routes/strata/`, `frontend/src/lib/strata/` | `/api/strata/omega` (snapshots; CUSUM = `services.drift.evaluate` on the alarm's `history_days` only, band ω_STRATA + μ ± 2σₑ), `/summary` (asset `validation`, register VAL / MOD, `OmegaModel` tiers, tier of the latest window's `flags.strata`), `/live-point` (file `src/storage/burden_engine/strata/live_point.json`). Job `strata.live_point` (03:15 IST, supervisor on demand): `model_facts.model_facts(model_facts.reference_engine(cal))` on the day's calibration + ring view (`OmegaModel.rings` on the network parts at the calibration state, no surrogate); a request never solves the network |
| BMO API | `src/services/bmo_case.py`, `src/services/bmo_views.py`, `src/api/routers/bmo.py`, `src/worker/kinds/bmo.py`, `src/utils/bmo/editor_tables.py` | SV-3.5: page 9's run inputs as functions shared by the page and the API (`compose_case(CaseSources, CaseInputs)`, `plan_lp`, `run_lp`; editor-table builders moved out of `ui/bmo/` and re-exported there). `POST /api/bmo/runs` queues `bmo.run` (role user; params `BmoRunParams`: unset = page 9's default; ore edits by id), result `BmoRunResult` at `GET /api/bmo/runs/{job}`; equals page 9 on the 5 golden days (`tests/test_services_bmo_case.py`, about 1 min). `GET /api/bmo/forecasts/si`, `/charged-coke` read the SV-1.5 ledgers (`si_for_page` / `charged_for_page`), never issue; `GET /api/bmo/snapshots[/{id}]` without the frozen plant data. Model modules imported lazily in `bmo_views` (API start) |
| What-if (SV-4.3) | `src/services/whatif.py`, `src/api/routers/whatif.py`, `src/worker/kinds/whatif.py`, `frontend/src/routes/admin/what-if/`, `frontend/src/lib/admin/whatif.ts` | Admin only. `/api/admin/whatif/basis` (stored what-if days, calibration point, register version, materials), `/cases` (per-user CRUD in `twin.whatif_case`, twin_0009, audited `whatif.save` / `whatif.delete`), `/runs` (1-4 saved cases). `whatif.forge`: levers via `coke_energy.whatif.run_scenario` (= ηCO-held column), then `rist.solve_2z` with ω, Q_LZ, Q_UZ (GJ/h), r_H held and CaO/SiO₂ held by `flux_needed` (the deck's two-zone column; reproduces COKE-01..08/14 on the 15 g/Nm³ fixture). `whatif.strata`: `optimiser.default_case(plan_day=, calibration_info=)` + forced blends `scenarios.run_today(tol_pct=0)` or `Engine.min_cost`. Calibration = the day's stored calibration (never a fallback) or the shipped one; each result stores basis + `FactRegister.label` with its case (owner and unchanged inputs only) |
| Twin store | `src/data/twin/`, `alembic/twin/` | SQLAlchemy 2.0 models and repositories (`TwinStore`) of the Neon schema `twin`: jobs (twin_0002: `scheduled_for`, `run_s`), snapshot, optimiser_run, plan_decision, shift_record, alarm_event, facts_version, audit_log (twin_0003: `before`, `after`, JSONB), shift_report (twin_0004), operator_inputs (twin_0005, SCD-2: trigger `operator_inputs_scd2`, one current row by `uq_operator_inputs_current`), fm_conversation / fm_message (twin_0006, `seq` per conversation under a row lock), fm_feedback (twin_0008, one row per answer, audited), whatif_case (twin_0009, per owner, audited). Migrate with `alembic -n twin upgrade head` (`make db-upgrade`; URL `BF_TWIN_DB_URL`, else `DATABASE_URL`). A model change needs a migration (`tests/test_twin_store.py` compares both). shift_record, plan_decision, audit_log are append-only (trigger); user actions write their audit row in the same transaction. The worker archives job records into `twin.jobs` (`services.job_archive`) |

- Pattern: a service exposes uncached building blocks (`load_*`, `read_*`, `build_*`, `compute_*`) and memoised
  entry points for the API/worker. Streamlit loaders (`ui/*/loaders.py`) wrap the uncached blocks in `st.cache_data`
  (spinners, copy-on-read) and pass them to the same composition, e.g. `services.coke_energy.build_page_data(keys,
  settings, loaders)`. Never compose a quantity twice.
- Auth: every router declares its minimum role, `APIRouter(..., dependencies=[Depends(require_role("user"))])` or per
  route (`api.auth`; admin > supervisor > user); only `api.auth.PUBLIC_PATHS` answer without a session, and
  `tests/test_api_auth.py` fails on a route without a role. Accounts come from Neon `identity.users` through
  `services.auth.NeonIdentityStore` (never seeds users). `BF_API_JWT_SECRET` (>= 32 characters) is required in prod.
  The SPA guards pages in `frontend/src/routes/+layout.ts` (session) and per section (`routes/admin/+layout.ts`).
- Single sign-on with `/legacy` (SV-6.6): the session cookie has path `/`; `services/session_tokens.py` (no FastAPI,
  no Streamlit) holds the token rules, `SessionSettings` (`BF_API_JWT_SECRET`, `BF_API_COOKIE_NAME`, `BF_REDIS_URL`)
  and `RedisRevocations` (`twin:auth:revoked:<jti>`, shared by the API deny list and Streamlit when Redis is
  configured). `utils.session.sync_api_session()` (called in `app.py` before the login gate) signs the Streamlit
  session in from a valid cookie with the token's user and role, ends it when the token expires or is revoked, and
  the Streamlit login form stays the fallback; a Streamlit sign-out revokes the API session. Tests
  `tests/test_session_sso.py`, `tests/test_services_session_tokens.py`.
- Admin space (SV-4.1): the views are `ADMIN_VIEWS` in `nav.ts`, listed by `shell/SectionNav`, one folder each under
  `routes/admin/<view>/`. Admin writes go through `services.audit.admin_write(store, actor, action, object_type,
  object_id)`: write through `write.session`, set `write.before` / `write.after` (`row_values`); one `twin.audit_log`
  row with before and after in the same transaction. Read the log through `GET /api/admin/audit` (admin only).
- Fact register editor (SV-4.2, `/admin/facts`): `services/facts.py` + `api/routers/facts.py` (`GET /api/admin/facts[?version]`,
  `/versions`, `POST /versions` propose, `POST /versions/{n}/review`, `/diff`, `/export`; admin only). A version of
  `twin.facts_version` is a cumulative overlay `{fact id: {value?, band?, status?}}` on the repository file, made from the
  version in force (409 otherwise), schema- and engine-checked on proposal and approval; a status change needs a second
  administrator; writes audited (`facts.propose`, `facts.review`). `utils.burden_engine.facts.load_register()` = the file
  with the latest approved overlay made against its version (`services.facts.approved_overlay`, memo 60 s), else the
  file; `load_register(path)` / `overlay=False` read the file only; retired facts refused (`RetiredFactError`). The
  export is the next file version (v1.8 -> v1.9, changed facts stamped with the approval day IST, revision-log entry).
- Burnt vs charged (SV-4.4, `/admin/burnt-vs-charged`): `services/burnt_vs_charged.py` (windows 1 shift - 1 month via
  `transit.window_pairs` + `concordance` = `window_agreement`, `decompose_window_sd`, `lag_scan`, `closure` + per month;
  leak-water check `utils/coke_energy/leak_check.py`: recent 3 d vs 14 d step of charged - raw burnt, top-gas H2, RAFT,
  model `strata_flow.plant_effects.leak_for_gap`, thresholds `coke_energy.yml -> twin.burnt_vs_charged.leak_check`),
  bases all / stable; 8 s for both on 1,325 shifts, so `twin.trace` builds it after the TRACE view (`trace.refresh`) into
  `burnt_vs_charged.json` beside it; `GET /api/admin/burnt-vs-charged?basis=` (`api/routers/burnt_vs_charged.py`, admin).
- Calibration console (SV-4.6, `/admin/calibration`): `services/calibration_console.py` reads what `calibration.nightly`
  stored (`day_calibration`, now with the window's `mask` = `utils.burden_engine.calibration.shift_mask` over
  `whatif.STABLE_BOUNDS` and the calibration `settings`; values unrounded = `live_calibration`), the Jacobian cache
  (`src/storage/strata_flow/`, `NetworkParts.stats()['cache_file']`) and the job's runs; `GET /api/admin/calibration[?day]`,
  `POST /api/admin/calibration/run` (admin). Kinds with `JobKind.audited` (`calibration.nightly`) are started through
  `services.job_audit.submit` (also via `POST /api/jobs`): audit `job.start`, 503 without the twin store, the job
  cancelled when the entry cannot be written.
- User, hopper and burden admin (SV-4.7, `/admin/users`, tabs `?tab=hoppers|burden`): the Welcome page's tools
  (`ui/user_management`, `hopper_admin_page`, `burden_admin_page`) and `services/plant_admin.py` share the form logic
  (`domain/plant_config.py`) and the session-level writes of `furnace_data.relational` (`UserRepository.create_user`,
  `HopperHistoryRepository` / `BurdenHistoryRepository.add_snapshot` and `delete_rows`); the service writes them on the
  twin store's database (Neon `identity` / `plant_master` / `ops_config`) inside `audit.admin_write` (`user.create`,
  `hopper.assign`, `hopper.delete`, `burden.update`, `burden.delete`). A save adds a wide snapshot effective from an
  aware time and keeps the earlier rows (SCD-2). `GET/POST /api/admin/users|hoppers|burden`, `DELETE
  .../hoppers|burden/history?ids=` (admin); 503 when the plant tables are missing.
- Shift reports (SV-2.3): `services/reports.py` composes the FurnaceMind Reports tab's `ShiftReportService.build` /
  `build_previous` / `document`; `report.shift` (closed shift) and `report.live` store into `twin.shift_report`. The
  narrative uses page 7's `OpenRouterClient` (120 s per call, 1 retry); when the LLM fails the report is stored
  without it (`narrative_status` failed).
- Shift handover record (SV-2.5): `services/shift_records.py` + `worker/kinds/shift.py`; `shift.close` freezes the closed
  shift once into `twin.shift_record` (hourly snapshots ending in the shift, the shift window's K / ω as columns, latest
  plan decision in the shift with its run, `report_ref` = `/reports?date=…&shift=…` since the report comes at HH:20);
  waits for the shift window to HH:30, then freezes what there is; idempotent per shift, backfills 3 shifts without a
  record. `GET /api/shift-records`, `/{shift_key}` (user, read-only: 405 on writes, trigger on the table); a report's
  `record` links it, the /reports page shows it (`#handover-record`, not printed).
- Furnace status (SV-2.1): `services/status.py` gives the V-Board snapshot and trend views; limits come from the fact
  register via `limits: {fact, low, high}` in `furnace_status.yml`. `services/live_status.py`: API processes write their
  SSE client counts to `twin:live:subscribers:*` (TTL 20 s); while the total is above 0 the worker heartbeat takes one
  snapshot per 10 s, writes `twin:live:status` and fills `LiveStatusData` of `live.status`. `BF_WORKER_LIVE_STATUS=0`
  turns the producer off (tests).
- Operator inputs (SV-3.1): `services/operator_inputs.py` reads and writes the `bmo_operator_inputs.yml` document as
  SCD-2 versions in `twin.operator_inputs` (audit `operator_inputs.update`; the first save seeds the YAML as v1); the
  YAML is the fallback when the store is off, empty or unreachable. Page 9 (`ui/bmo/operator_inputs.py`), page 11 and
  the worker (`load_yard` → `EvonithBmoContextProvider.get_operator_inputs`) and `GET` / `PUT /api/operator-inputs`
  (`?at=` date = end of the IST day) all read this one source.
- Burden & Fuel Optimiser page (SV-3.3): `routes/optimiser` with `#lib/optimiser/` (builders `optimiser.ts`, state
  `optimiser.svelte.ts`); runs through `POST /api/optimiser/runs` (the Live Twin card's endpoint and store), result
  `GET /api/optimiser/runs/{id}/result` (also the case in API names, the yard, page 11's lead line, plan limits, labels
  of binding limits), `GET /api/optimiser/reference` (plant slag bands of `utils/slag_bands.py`, BAND-01..03, the
  engine's facts, MOD / VAL, `assets/burden_engine/assumptions.md` shared with page 11); yard edits save operator-input
  versions. Parity: `lib/optimiser/fixtures/result.json` (`make fixtures`, page 11's figures, relative 1e-6).
- FORGE page (SV-1.11): the `twin.forge` job (HH:20 IST, `services/forge.py` `refresh`) stores the Coke Energy
  pipeline's balanced shifts and Rist frames in `src/storage/forge/`; `GET /api/forge/prediction | rist | calibration`
  (role user) only read that view. Statistics are page 10's (`accuracy.period_rows` / `accuracy`; the rolling window
  is the `twin.forge.stats_days` before each day and the day, 31 days), Rist lines are `rist_diagram`'s points
  (`tests/test_services_forge.py`). Bands in `TimeSeriesChart` stack with `stackStrategy: 'all'` (negative edges).
- UI: look of the V-OptimAIse product family as in Optimus (user decision 27 Oct 2026): dark green-charcoal sidebar,
  light canvas, white panels, yellow accent, Inter, tracked upper-case section labels. Colours, sizes and spacing only
  from `frontend/src/lib/theme/tokens.css` (no hex values in components); build pages from
  `frontend/src/lib/components/` (catalogue at `/dev/components`, rules in `frontend/DESIGN.md`): status always with
  icon and word, `updatedAt` on live panels, units on every readout joined by a no-break space (`formatQuantity`),
  chemical labels with Unicode subscripts (SiO₂). New nav pages go into `frontend/src/lib/nav.ts`. Check with
  `make audit-ui` (axe-core, three viewports).
- Charts in the SPA: `#lib/charts/` (`TimeSeriesChart`, `ScatterChart`, `BarChart`, `DonutChart`, `RistDiagram`;
  ECharts 6 SVG, loaded lazily; pure option builders with unit tests) and `#lib/furnace/FurnaceSchematic.svelte`.
  Series colours only by token: categorical slots `cat-1..6` in fixed order, entity aliases `series-trace` / `-forge` /
  `-strata` / `-hmt` (slots 1-4), `series-context` for references; never a second value axis. After changing
  `utils/coke_energy/rist_diagram.py` or `ui/furnace_status/schematic.py` run `make fixtures` and commit the JSON
  (`tests/test_chart_fixtures.py` fails on drift). Rules in `frontend/DESIGN.md` section 6.
- One origin in production: `deploy/jetson/Caddyfile` (SPA with `200.html` fallback, `/api` with `flush_interval -1`, `/legacy` to Streamlit with WebSocket, HTTP to HTTPS; settings from `BF_*` variables, see `deploy/jetson/README.md`). Link to a Streamlit page with `legacyHref('<title>')` from `#lib/legacy.ts`; the titles are generated from `config/page_registry.py` (`make fixtures`, checked by `tests/test_legacy_pages.py`), so a renamed page fails the type check.
- Release (SV-6.4): `sudo deploy/jetson/release.sh <tag>` (CI release + source archive, `SHA256SUMS`, `uv sync --frozen`, `alembic -n twin upgrade head`, switch `current`/`previous`, restart, smoke test through Caddy, automatic rollback, exit 3), `rollback.sh [<tag>]`; tested on a fake host (`tests/test_jetson_release.py`). It writes `<release>/RELEASE.json`, read by `services/release.py` for `GET /api/version` (`release`, `installed_at`, `frontend_build`) and the footer (`shell/VersionInfo`). SPA build version: `BF_BUILD_VERSION` (release.yml: the tag), else `git describe`; read it from `$app/env` (Kit 3), `updated` of `$app/state` offers the reload.
- Jetson base system (SV-6.1): `deploy/jetson/provision.sh` (root, idempotent: JetPack 6 check, NVMe by UUID at
  `/var/lib/bf-webapp`, MAXN + jetson_clocks unit, chrony to the plant NTP only, ufw 443/80 + 22 from the admin subnet,
  user `bfapp`, `/opt/bf-webapp/releases` + `current`, `/etc/bf-webapp/env` root:bfapp 0640, 14-day logs) and
  `verify_base.sh` (offset < 100 ms, units back after a reboot, open-port list); name service units `bf-*` with
  `RequiresMountsFor=/var/lib/bf-webapp`; tests `tests/test_jetson_scripts.py` (bash via Git Bash on Windows).
- Jetson services (SV-6.3): `deploy/jetson/systemd/` (`bf-redis` 127.0.0.1:6379, `bf-furnace-data` :8000, `bf-twin-api`
  uvicorn 2 workers :8100, `bf-twin-worker` pool 8, `bf-streamlit` :8501 `/legacy`, `bf-caddy` :443, all `WantedBy=` /
  `PartOf=bf-webapp.target`; `bf-healthcheck.timer` = `deploy/jetson/healthcheck.sh`, restarts a unit after 2 failed
  probes), installed and enabled by `provision.sh` step `services` (Caddy 2.10.2 from the release, package units
  masked). User `bfapp`, `EnvironmentFile=/etc/bf-webapp/env`, `Restart=always` 2 s, `MemoryMax`; `Wants=` (never
  `Requires=`) between services; Python units run `./.venv/bin/python` relative to
  `WorkingDirectory=/opt/bf-webapp/current` (release pinned at start). Ports and keys checked by
  `tests/test_jetson_units.py`; change a port in the unit, the Caddyfile and the health check together.
- Outages (SV-6.5): `services/source_health.py` (registry `HEALTH` of the five sources InfluxDB / hourly / HM-slag /
  DPR / Open-Meteo: `record_ok` / `record_failure` from `services.twin.load_frames`, every status snapshot view, the
  hourly-dataset refresh (`record_dataset_refresh`; the refresh returns its error) and the Si source's reads
  (`forecasts.live_si_source`); a source that failed is `down` for `RETRY_AFTER_S` (5 min): `twin.evaluate`,
  `forecast.si` and `forecast.charged_coke` skip (`worker.runtime.SkipJob(reason, source=)`: job `cancelled` with the
  reason, the source's `source_stale` alarm instead of `job_failed`, a `skipped` forecast attempt) and do not retry;
  the worker writes its registry to Redis every beat (`twin:sources:health`), `/api/twin/now` merges it and its
  sources carry `down` / `error`, so the SPA shows "Down" at once). Twin-store writes of the worker (snapshots, job
  upserts, the job archive) go through `services/outbox.py` (`BufferedTwinStore` over `data/twin/outbox.py`, SQLite WAL
  + `synchronous=FULL` at `BF_TWIN_OUTBOX`, default `src/storage/twin/outbox.sqlite`): writes Neon cannot take are
  queued and replayed in order by the heartbeat every 30 s, `twin.evaluate` and housekeeping; idempotent (upserts by
  key), the entry is deleted after the Neon commit. Tests `tests/test_outbox.py`, `tests/test_source_health.py`.
- Live Twin page (SV-1.9, `routes/+page.svelte`): state in `#lib/twin/live-twin.svelte.ts` (`LiveTwin.start()` returns
  the dispose; bounded buffers `MAX_ROWS` / `MAX_SAMPLES`), pure builders in `#lib/twin/live-twin.ts` tested on the
  recorded payloads of `#lib/twin/fixtures/payloads.ts`; API additions `/api/twin/now` sources and gaps,
  `/api/twin/window`, `/api/twin/hot-metal`, `/api/twin/transit` blend change; alarms `services/twin_alarms.py`
  (STRATA − TRACE gap, stale source, implied lag, job failed; `coke_energy.yml -> twin.alarms`, `twin.freshness`);
  optimiser runs store `result.critical_assumptions`. `FurnaceSchematic` takes an `overlay` snippet.
- Furnace Status page (SV-2.2, `routes/status/+page.svelte`): state in `#lib/status/furnace-status.svelte.ts`
  (`FurnaceStatus.start()` returns the dispose; 10 s = every `live.status` beat with a snapshot reads
  `/api/status/snapshot` from the worker's Redis copy, polls after 2.5 silent beats; 30 s / 1 min / 5 min poll; an open
  trend follows the cadence unless custom), pure builders in `#lib/status/status.ts` (V-Board layout, row and callout
  texts, zone status, strip spreads, heat-load shares, trend series and stats, IST custom range) tested on the recorded
  snapshot `#lib/status/fixtures/snapshot.json` (written by `scripts/export_chart_fixtures.py`, checked by
  `tests/test_chart_fixtures.py`). `GET /api/status/settings` serves the display rules of `furnace_status.yml`
  (stale-after, live-age, refresh choices, trend intervals, longest custom range, operating-state rule, zones); the SPA
  holds no copy. `FurnaceSchematic` takes `zoneStatus` (icon and word per zone button) and `onselect`.
- Shift Reports page (SV-2.4, `routes/reports/+page.svelte`, URL `?date=&shift=&id=`): state in
  `#lib/reports/shift-reports.svelte.ts` (list, report, thresholds, twin row of the shift via `/api/twin/series`
  `resolution=shift`, or the latest hourly window of a running shift; regenerate = `report.shift` with `shift_key`,
  supervisor, or `report.live`, user; the page calls `jobFinished()` when the tracker reads done), builders in
  `#lib/reports/reports.ts` with the tab's decimals (parity with the document of `#lib/reports/fixtures/report.json`,
  written by `scripts/export_chart_fixtures.py`). KPI levels: `reports/furnace_report/status.py`, one rule for
  `builder._status` and `services.reports.kpi_checks`; thresholds read from `shift_report.yml` per request. Print
  theme: `--vo-print-*` tokens (paper #FFFBE6), `theme/print.css` (`@page` A4, `data-print="hide"`), print rules in
  Panel (`breakable`), DataTable, AppShell, PageHeader; `#lib/reports/print.ts` plans the pages (limit 3).
- Blend Mix Optimiser page (SV-3.6, `routes/bmo/+page.svelte`): `#lib/bmo/runs.svelte.ts` opens on the newest
  `bmo.run` (`GET /api/bmo/runs`, `JobStore.list_kind`) and queues runs through `POST /api/bmo/runs`; plan in
  `#lib/bmo/PlanTabs.svelte` (Blend, Fuel, Slag, Charging). The plan's slag ratios, uncorrected coke and `charging`
  are page 9's Comparison rows (`bmo_views.page_outcomes` = `services.bmo.comparison_view`; parity with the recorded
  cells on the 5 golden days in `tests/test_services_bmo_case.py`), plant bands `slag_bands` from `utils.slag_bands`.
  Sections of page 9 in Streamlit: `PageSection` in `config/page_registry.py` → `LEGACY_SECTIONS` (`make fixtures`)
  → `legacyHref(page, section, params)`; page 9 opens `?section=sandbox|replay&snapshot=` in
  `ui.bmo.snapshot_panel.open_linked_section` and `#de-search` is the run form's heading anchor.
- Live data in the SPA: subscribe through `#lib/live/stream.svelte.ts` (`live.on(type, handler)`, one EventSource
  per tab); follow jobs with `JobTracker` (`#lib/live/jobs.svelte.ts`). A new event type is added to `EventType` and
  the payload models in `services/events.py`, then the OpenAPI client is regenerated (`make client`).
- Dev: `python scripts/dev.py` (SPA 5173 + API 8100 + worker + Streamlit under `/legacy` on 8501; the worker only
  when Redis answers). Lint the new Python with `uvx ruff@0.17.2` (paths in the Makefile). Lock changes with
  `uvx --from "uv==0.7.13" uv lock` (keeps `uv.lock` at revision 2 with pure additions; uv 0.11 rewrites it to
  revision 3 and later 0.7.x releases add marker churn), then `uv sync --frozen`.
- CI (`.github/workflows/`, SV-0.11): **manual runs only** (`workflow_dispatch`; user decision 29 Oct 2026, to save
  GitHub Actions minutes: nothing runs on push, pull request, schedule or tag; run the checks locally and start a
  workflow by hand when needed, `gh workflow run ci.yml --ref <branch>`). `ci.yml` (`static`:
  `make lint-py`, `make lock-check`; `python-tests`: pytest with PostgreSQL, Redis and Caddy,
  `BF_TEST_REQUIRE_SERVICES=1`, `BURDEN_ENGINE_FAST=1`; `frontend`: lint, check, Vitest, `schema.d.ts` against
  `openapi.json`, build, UI audit), `nightly.yml` (full suite), `release.yml` (dispatched on a tag `v*`:
  `frontend-build.tar.gz`, `uv.lock`, `pnpm-lock.yaml`, `SHA256SUMS`). Keep every CI step runnable as a Make target or pnpm script, and run
  it through `bash scripts/ci/step.sh "<title>" <command>` (a failure's last 50 lines become an error annotation,
  readable through the Checks API: `gh api repos/:owner/:repo/check-runs/<id>/annotations`). Servers started by a
  check bind to `127.0.0.1`, not `localhost` (`::1` first on the runners).
- A new Python package needs a linux-aarch64 wheel for the Jetson (`make lock-check`: glibc 2.35, CPython 3.12), or
  an entry in `JETSON_SOURCE_BUILDS` (Makefile) with the reason. Never write `-o /dev/null` for a uv command: uv
  replaces the output file by a rename, which turns `/dev/null` into a regular file when run as root.
  After a lock change re-run the wheel audit (`python scripts/jetson_wheel_audit.py --write` → `docs/jetson_wheel_audit.md`;
  `tests/test_jetson_wheel_audit.py` fails on drift). On the device (SV-6.2): `deploy/jetson/install_python.sh` (uv
  0.7.13, uv-managed CPython 3.12, `/etc/bf-webapp/uv.env`; `--release DIR`: `uv sync --frozen` as DIR's owner, import
  check), then `deploy/jetson/benchmark.sh --open-issue` (→ `docs/jetson_benchmark.md`; targets solve 2.8 s, Jacobian
  6 s, Step 2 90 s, maximum production 300 s, Step 1 600 s).
- Tests: never write stubs into `sys.modules` while a file is collected (they stay for the session and break later
  files); import the module under test with `tests/module_stubs.import_with_stubs`. Tests that need a service use
  the `requires_service` marker (`needs_postgres`, `needs_redis`). `tests/conftest.py` sets placeholder
  `QDRANT_URL` and `OPENROUTER_API_KEY` for the import-time validation of `utils.settings`.
- Optimiser API (SV-3.2): `POST /api/optimiser/runs` (`OptimiserRunParams`: mode, ω mode, `calibration`
  day / live / reference, `yard` live / reference, `settings` / `limits` overrides in API names →
  `services.optimiser.SETTINGS_FIELDS` / `LIMITS_FIELDS`, applied last; BAND-01 stays the register's);
  `GET /runs/{id}/result` = `services.optimiser_results.result_view` (typed plan from `scenarios.plan_tiles`, ore +
  flux plan, today's blend, frontier, value of limits, critical assumptions, network solves);
  `POST|GET /runs/{id}/commentary` = job `optimiser.commentary` (`services.optimiser_commentary`: page 11's
  `build_context` on the stored case, one LLM retry) into `twin.optimiser_commentary` (twin_0007). The case report is
  reproduced through the API in `tests/test_api_optimiser.py`.

## Environment Variables

```
LLM:        LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_MODE, OPENROUTER_API_KEY
Embeddings: LOCAL_EMBEDDING_*, CLOUD_EMBEDDING_*
Qdrant:     QDRANT_ENDPOINT or QDRANT_URL, QDRANT_API_KEY
            SHIFT_QDRANT_COLLECTION, KNOWLEDGE_QDRANT_COLLECTION
Database:   DATABASE_URL (PostgreSQL/Neon)
Tickets:    TICKETS_DB_URL (SQLite default; set for PostgreSQL override)
InfluxDB:   INFLUX_ONLINE_TOKEN, INFLUX_OFFLINE_TOKEN
twin-api:   BF_API_JWT_SECRET (required in prod), BF_API_COOKIE_SECURE, BF_API_TRUSTED_PROXIES, BF_API_STATIC_DIR,
            BF_API_ENVIRONMENT, BF_API_SESSION_HOURS, BF_API_LOGIN_MAX_FAILURES, BF_API_LOGIN_WINDOW_S,
            BF_API_STREAM_HEARTBEAT_S, BF_API_FM_WARM_UP (0: no FurnaceMind agent-stack import at start; tests: 0)
Worker:     BF_REDIS_URL (API and worker), BF_WORKER_POOL, BF_WORKER_MAX_JOBS, BF_WORKER_JOB_TIMEOUT,
            BF_WORKER_SKIP_SCHEDULES (kinds or prefixes left unscheduled; tests: forecast.,twin.,optimiser.,calibration.,learn.,report.),
            BF_WORKER_CATCH_UP (0 = no catch-up of missed slots at start; tests: 0),
            BF_WORKER_LIVE_STATUS (0 = no live furnace-status snapshots from the heartbeat; tests: 0),
            BF_STRATA_SOLVERS (network-solver processes per STRATA-ω run; default min(8, cpus); 0 or 1 = serial; tests: 0)
Twin store: BF_TWIN_DB_URL (default DATABASE_URL; `off` disables), BF_TEST_DATABASE_URL (tests; else a local cluster)
```
