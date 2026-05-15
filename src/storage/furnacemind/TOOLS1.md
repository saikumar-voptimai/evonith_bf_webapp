# FurnaceMind AI Co-Operate - Tools Playbook (v2)

This file is injected into the AI Co-Operate system prompt on every turn.  
It is the source of truth for tool routing and plotting behavior.

## Quick Start Flow

1. Choose data source using the routing table below.
2. Fetch only the minimum required datasets.
3. Validate `df.columns` before writing plot code.
4. Call `execute_python_plot` once with one final chart.

## Non-Negotiable Runtime Rules

- There is no runtime dependency on `current_furnace_data.csv`.
- The active dataset always lives in `st.session_state.fm_df`.
- `execute_python_plot` runs against preloaded variables:
  `df`, `pd`, `px`, `go`, `np`, `make_subplots`.
- Never use `import`, `__import__`, `open`, `os`, `subprocess`, `sys`, `eval`, `exec`, or `fig.show()`.
- Plot code must assign a Plotly figure to `fig`.
- If uncertain about columns, run diagnostics first:
  `print(df.columns)` and `print(df.index.min(), df.index.max())`.

## Data Source Routing

| Scenario | Primary Tool |
|---|---|
| Historical analysis (> 2 days) | `fetch_ml_data` |
| Historical request extends beyond ML static end | `fetch_ml_data` then follow GAP NOTE (`fetch_online_data` + `concat_datasets`) |
| Recent window (<= 2 days) or sub-hourly trend | `fetch_online_data` |
| HM/Slag chemistry, Charge, Bunker, DPR report data | `fetch_offline_data` |
| Stitch static + online row-wise | `concat_datasets` |
| Join offline columns onto online/static timeline | `merge_furnace_data` |
| Single 8-hour shift from static dataset | `load_static_shift_data` |

## Pre-Flight Checklist for `execute_python_plot`

- [ ] `df` exists and is not empty.
- [ ] Required columns exist in `df.columns`.
- [ ] Null-heavy columns are cleaned (`dropna` or filtering).
- [ ] One final figure is selected for this response.
- [ ] X-axis logic is explicit (`df.index` unless user asked otherwise).

## Tool Contracts

### 1. `fetch_ml_data` (primary for history)
- Reads pre-merged static ML dataset (hourly, IST-naive index).
- Fast path, no live Influx call.
- Inputs: `start_time`, optional `end_time`, optional `resample`, optional `columns`.
- If out of near-present range, returns GAP NOTE with exact follow-up calls.

### 2. `fetch_online_data`
- Fetches live telemetry from Influx (`bf2_evonith_raw`).
- Max lookback is 90 days.
- **Parameters — use EITHER `lookback` OR `start_time_utc`/`end_time_utc`, never both:**
  - `lookback` — compact string like `"8h"`, `"2d"`, `"30m"`, `"1 week"`.
  - `start_time_utc` / `end_time_utc` — ISO-8601 UTC strings for exact windows.
  - Omit any param you don't need. Do NOT set unused fields to `""` or `0`.
- Default averaging when `window` is omitted:
  - `> 1 day` lookback -> `1 hour`
  - otherwise -> `15 minutes`
- Output columns are mapped display labels, not raw Influx field IDs.

**Example — last 8h ETA CO:**
```json
{"lookback": "8h", "measurement_groups": ["process_params"]}
```
**Example — exact shift window:**
```json
{"start_time_utc": "2026-05-01T00:30:00Z", "end_time_utc": "2026-05-01T08:30:00Z", "measurement_groups": ["process_params"]}
```

### 3. `fetch_offline_data`
- Fetches offline/manual reports from offline bucket.
- Supported report types: `HM_SLAG`, `CHARGE`, `RAW_MATERIAL_COMPOSITION`, `DPR`.
- Default cadence:
  - `HM_SLAG` and `CHARGE` -> `1h`
  - `RAW_MATERIAL_COMPOSITION` -> `8h`
  - `DPR` -> `1d`

### 4. `concat_datasets`
- Vertical union by timestamp.
- Duplicate timestamps keep the last dataset row.
- Primary use: stitch static + recent online windows.

### 5. `merge_furnace_data`
- Column-wise join of offline datasets onto online/static base timeline.
- Use `fill_method='ffill'` for sparse report datasets.

### 6. `execute_python_plot`
- Executes restricted Python with preloaded context.
- Must create `fig`.
- Diagnostic print-only calls are allowed and return text output.

### 7. `search_shift_history` and `search_knowledge_docs`
- Use for narrative context, SOP grounding, and evidence text.
- Do not use these to derive live numeric values.

## Column Naming Guardrails

The same metric may appear under different naming families.

### Static ML naming (examples)
- `ACT. FUEL RATEKG/THM.`
- `COKE RATE KG/THM`
- `FURNACETOPGASANALYSISCO2ETACO`
- `CHEM_PCT_SI`

### Online mapped naming (examples)
- `Process Params - fuel_rate`
- `Heatload Delta T - Heat load Row 6`
- `Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp A`

### After `concat_datasets`
- Both naming families may co-exist.
- Always inspect `df.columns`.
- Plot whichever column is present and non-null in the requested interval.

## Anti-Patterns

1. Do not reference or read `current_furnace_data.csv`.
2. Do not call `execute_python_plot` multiple times unless user requested iterative debugging.
3. Do not assume column names from memory.
4. Do not provide numeric claims without a data fetch or explicit evidence.

## Response Discipline

1. Fetch only the data required for the question.
2. Produce one relevant final figure.
3. Return concise conclusion, actions, and evidence.
