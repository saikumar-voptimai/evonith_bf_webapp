# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Blast Furnace Web Application — an industrial manufacturing dashboard for real-time blast furnace monitoring, AI-powered analysis, and optimization recommendations. Built with Streamlit (multi-page app) backed by PostgreSQL, InfluxDB (time-series), and Qdrant (vector search).

## Commands

```bash
uv sync                    # Install dependencies
python run_streamlit.py    # Run the app (always use this, not streamlit run directly)
pytest tests/              # Run tests
uv add <package>           # Add dependency
```

**Do not invoke `streamlit run` directly** — `run_streamlit.py` imports torch first to prevent Windows DLL errors.

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

## Environment Variables

```
LLM:        LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_MODE, OPENROUTER_API_KEY
Embeddings: LOCAL_EMBEDDING_*, CLOUD_EMBEDDING_*
Qdrant:     QDRANT_ENDPOINT or QDRANT_URL, QDRANT_API_KEY
            SHIFT_QDRANT_COLLECTION, KNOWLEDGE_QDRANT_COLLECTION
Database:   DATABASE_URL (PostgreSQL/Neon)
Tickets:    TICKETS_DB_URL (SQLite default; set for PostgreSQL override)
InfluxDB:   INFLUX_ONLINE_TOKEN, INFLUX_OFFLINE_TOKEN
```
