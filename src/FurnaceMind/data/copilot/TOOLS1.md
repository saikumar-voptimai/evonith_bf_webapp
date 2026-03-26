# FurnaceMind AI Co-Operate — Tools Reference

This file is injected into the AI Co-Operate system prompt on every interaction.

## Available Tools (function-calling)

### 1) `fetch_online_data`
- Purpose: Fetch online telemetry from InfluxDB (`bf2_evonith_raw`).
- Limits: max lookback 90 days.
- Defaults:
  - If lookback > 1 day and `window` omitted → 1 hour avg
  - Else → 15 minutes avg
- Output:
  - Saves to `current_furnace_data.csv`
  - Updates `st.session_state.copilot_df`
  - Returns `dataset_id` + column preview
- Args:
  - `lookback_days` (1–90) OR `lookback_hours` OR `lookback_minutes`
  - `window` (optional, e.g. "15 minutes", "1 hour")
  - `measurement_groups` (optional subset):
    - `process_params`, `cooling_water`, `heatload_delta_t`, `delta_t`, `temperature_profile`, `miscellaneous`

### 2) `fetch_offline_data`
- Purpose: Fetch offline/manual reports from InfluxDB (`bf2_evonith_offline_utc`).
- Default cadences:
  - `HM_SLAG` → 1h
  - `CHARGE` → 1h
  - `RAW_MATERIAL_COMPOSITION` → 8h
  - `DPR` → 1d
- Output: same as online (writes `current_furnace_data.csv`, sets `copilot_df`, returns `dataset_id`).
- Args:
  - `report_type`: `HM_SLAG` | `CHARGE` | `RAW_MATERIAL_COMPOSITION` | `DPR`
  - Optional: `start_time_utc`, `end_time_utc`, or `lookback_days` (1–365)
  - Optional: `cadence`: `1h` | `8h` | `1d`

### 3) `merge_furnace_data`
- Purpose: Merge offline datasets onto an online dataset by timestamp alignment.
- Policy:
  - Offline data can be forward-filled onto online timestamps when `fill_method="ffill"`.
- Output: merged dataset saved to `current_furnace_data.csv`, `copilot_df` updated, returns merged `dataset_id`.
- Args:
  - `online_dataset_id`
  - `offline_dataset_ids` (list)
  - `fill_method`: `ffill` | `none`

### 4) `search_shift_history`
- Purpose: Semantic search over stored shift summaries (Qdrant).
- Args: `query` (natural language)

### 5) `search_knowledge_docs`
- Purpose: Semantic search over uploaded operator documents (SOPs/specs).
- Args: `query` (natural language)

### 6) `execute_python_plot`
- Purpose: Run restricted Python to create a Plotly figure named `fig` using the current `df`.
- Requirements:
  - Code MUST create a variable called `fig`.
  - Use provided `df`, `pd`, `px`, `go` (no imports).
- Output:
  - Stores `fig` into `st.session_state.copilot_fig`.

## Output Discipline
- If you need numbers or plots, call tools; do not guess.
- Keep responses short:
  - 1-line conclusion
  - up to 3 action bullets
  - up to 2 evidence bullets
