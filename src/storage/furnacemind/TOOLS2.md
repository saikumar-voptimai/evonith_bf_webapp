# Tool Reference: `load_static_shift_data` (v2)

Load one complete 8-hour shift from the static ML dataset and set it as active `df` for analysis/plotting.

## Best Use Cases

1. Shift-to-best comparisons.
2. Historical shift diagnostics where hourly granularity is enough.
3. Fast setup before KPI, burden, or heatload review for one shift.

## Do Not Use When

1. You need sub-hourly trends (use `fetch_online_data`).
2. Requested shift is outside static coverage (use live/offline fetch tools).

## Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `shift_date` | string | yes | ISO date `YYYY-MM-DD` |
| `shift_label` | enum | yes | `A`, `B`, or `C` |

### Shift Windows (IST)

| Shift | Start | End |
|---|---|---|
| A | 00:00 | 08:00 |
| B | 08:00 | 16:00 |
| C | 16:00 | 00:00 (next day) |

## Runtime Behavior

- Validates shift against static dataset coverage.
- Returns `dataset_id`, shape, columns, and preview.
- Stores active data in:
  - `st.session_state.fm_df`
  - `st.session_state.fm_df_meta`
- Does not require or write `current_furnace_data.csv`.

## Mandatory Checks Before Plotting

1. Ensure response is not an out-of-range error.
2. Confirm target columns are present in `df.columns`.
3. Run diagnostics when uncertain:

```python
print(df.columns)
print(df.index.min(), df.index.max())
```

4. Then call `execute_python_plot` with code that sets `fig`.

## Example Tool Call

```json
{
  "name": "load_static_shift_data",
  "arguments": {
    "shift_date": "2026-02-21",
    "shift_label": "C"
  }
}
```

## Common Failure Modes

### 1) Out-of-range shift request

Use `fetch_online_data` plus `fetch_offline_data`, then:
- `merge_furnace_data` for column-wise alignment, or
- `concat_datasets` for row-wise stitching.

### 2) Static vs online column-name mismatch

- Static uses ML-style columns (for example `ACT. FUEL RATEKG/THM.`, `CHEM_PCT_SI`).
- Online uses mapped display names (for example `Process Params - fuel_rate`).
- Always inspect `df.columns` before plotting.
