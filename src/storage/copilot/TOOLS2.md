# Tool: `load_static_shift_data`

Load a complete 8-hour shift from the **static ML dataset** (`data/ml_dataset_filtered.csv`).
This dataset contains hourly rows with online *and* offline data already merged and aligned.

## When to use
- **Shift-to-Best comparison** — load a historical shift to compare against SKILLS_BESTSHIFT.md benchmarks.
- **Any historical shift analysis** where the date falls within Jan 2024 – Mar 2026.
- Faster than fetching from InfluxDB because the data is local and pre-merged.

## When NOT to use
- If the requested date is **outside** the dataset range (Jan 2024 – Mar 2026), the tool will return an error. Fall back to `fetch_online_data` + `fetch_offline_data` + `merge_furnace_data`.
- If you need **sub-hourly resolution** (the static dataset is hourly).

## Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `shift_date` | string | yes | ISO date `YYYY-MM-DD` |
| `shift_label` | string (enum) | yes | `"A"`, `"B"`, or `"C"` |

### Shift windows (IST, timezone-naive in the CSV)

| Label | Start | End |
|---|---|---|
| A | 00:00 | 08:00 |
| B | 08:00 | 16:00 |
| C | 16:00 | 00:00 (next day) |

## Output
- Saves the shift slice (up to 8 hourly rows) to `current_furnace_data.csv` and `st.session_state.copilot_df`.
- Returns `dataset_id`, shape, column list, and a 2-row preview.

## Example call
```json
{
  "name": "load_static_shift_data",
  "arguments": {
    "shift_date": "2026-02-21",
    "shift_label": "C"
  }
}
```

## Column highlights (150+ columns available)
The dataset includes all parameters referenced in SKILLS_BESTSHIFT.md:

**Tier 1 controls:** `O2 ENRICHMENT %`, `HOT BLAST TEMP.OC`, `TOPPRESSUREBAR`, `WEIGHTED_COKE_ANGLE`, `WEIGHTED_NON_COKE_ANGLE`, `CHARGES/HRS.`, `STEAMKGS/HR.`

**Tier 2 raw materials:** `PCI_2_ASH%`, `PCI_2_IM%`, `COKE_MOIST%`, `COKE_ASH%`, `NUTCOKE_ASH%`, `NUTCOKE_MOIST%`, `SINTER_SP_02_FEO%`, `SINTER_SP_02_AL2O3%`, `SINTER_SP_02_SIO2%`, `SINTER_SP_02_BASICITY`, `ORE_TM%`, `ORE_LOI%`

**Tier 3 guardrails:** `TOPBAR`, `DIFFERENTIAL PRESSURETOTALBAR`, `PERMEABILITYKGS/HR.`, `CHEM_PCT_SI`, `CHEM_PCT_S`, `SLAG_BASICITY`, `SLAG_PCT_FEO`, `TOTAL HEAT LOAD`, `RAFTOC`, `bosh_quad_spread`, `uptake_quad_spread`

**KPIs:** `ACT. FUEL RATEKG/THM.`, `COKE RATE KG/THM`, `FURNACETOPGASANALYSISCO2ETACO`, `PRODUCTIONTONNESPERHR`, `UNITCOST LAKHS/THM`
