# Skill: Check Heatloads & Skin Temperatures

## Purpose
Detect elevated heatloads and temperature spikes in the furnace body by comparing the last 8 hours of operating data against a 2-month rolling baseline. Flag any stave row, quadrant, or temperature zone showing abnormal values.

## Why it matters
Elevated heatloads indicate refractory wear, channeling gas flow, or raceway imbalance. Early detection prevents brick damage and unplanned shutdowns. Skin temperatures at 18660 mm (Lower Stack) are the primary channeling indicator.

---

## Data sources required

### Column naming — IMPORTANT
Online data columns use the format **`"{Measurement Label} - {Field Label}"`** — NOT raw InfluxDB field names.
Always rely on the column list returned by `fetch_online_data` rather than guessing names.
Use `df.filter(like='...', axis=1)` or `[c for c in df.columns if '...' in c]` for robust selection.

| What you need | DataFrame column pattern | Example exact name |
|---|---|---|
| Row heat loads (R6–R10) | `"Heatload Delta T - Heat load Row {N}"` | `"Heatload Delta T - Heat load Row 6"` |
| Cross-row quadrant averages | `"Heatload Delta T - Heat load Row6-10 Q{N}(Stave …)"` | `"Heatload Delta T - Heat load Row6-10 Q1(Stave 1-8)"` |
| Per-row quadrant | `"Heatload Delta T - Heat load R{row} Q{N}(Stave No …)"` | `"Heatload Delta T - Heat load R6 Q1(Stave No 1-8)"` |
| Delta-T row averages | `"Delta T - DELTA T avg Row{N}"` | `"Delta T - DELTA T avg Row6"` |
| 18660 mm temps | `"Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp {A/B/C/D}"` | `"Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp A"` |
| 15162 mm temps | `"Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp {A/B/C/D}"` | `"Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp A"` |
| 12975 mm temps | `"Temperature Profile - BF2_BFBD Furnace Body 12975mm Temp {A/B/C/D}"` | `"Temperature Profile - BF2_BFBD Furnace Body 12975mm Temp A"` |
| Production rate | `"Process Params - production_per_hour"` | `"Process Params - production_per_hour"` |

### Current state (last 8h)
- **Tool**: `fetch_online_data`
- **Groups**: `heatload_delta_t`, `temperature_profile`, `process_params`
- **Window**: `15 minutes` or `1 hour`
- **What to get**: Row heat loads (R6–R10), cross-row quadrant averages (Q1–Q4), per-row quadrant loads, upper/lower stack temperatures (18660 mm, 15162 mm, 12975 mm), production rate

### Baseline (2-month rolling)
- **Tool**: `fetch_online_data`
- **lookback_days**: 60
- **Groups**: `heatload_delta_t`, `temperature_profile`
- **Window**: `1 hour` (keep data volume manageable)
- **Note**: Same column naming format as current-state fetch — row loads, quadrant loads, and stack temps are all available at row/quadrant resolution, unlike ML static which only has `TOTAL HEAT LOAD`.

---

## Analysis methodology

### Step 1 — Fetch both datasets
1. Fetch online current (last 8h, 15-min window) → dataset A
2. Fetch online baseline (last 60 days, 1h window) → dataset B

### Step 2 — Compute baseline statistics
For each column in dataset B (60-day baseline):
- `baseline_mean = col.mean()`
- `baseline_std = col.std()`
- `baseline_p95 = col.quantile(0.95)`

### Step 3 — Compute current averages
For the online last-8h data:
- Average each heatload/temperature column

### Step 4 — Compute z-scores and flag
```
z_score = (current_value - baseline_mean) / baseline_std
```
Flags:
- `z > 2.0` → ELEVATED (yellow) — monitor
- `z > 3.0` → HIGH (orange) — investigate
- `z > 4.0` → CRITICAL (red) — urgent action

### Step 5 — Temperature spread (channeling indicator)
For the 18660 mm level (A, B, C, D sensors):
- `spread = max(temp_18660_*) - min(temp_18660_*)`
- Normal spread: < 40°C
- Elevated: 40–80°C (may indicate incipient channeling)
- Critical: > 80°C (likely channeling)

For the 15162 mm and 12975 mm levels:
- Also compute max-min spread
- Flag if spread > 2× baseline spread

### Step 6 — Quadrant asymmetry
- Compare Q1/Q2/Q3/Q4 heat loads to the row average
- Asymmetry = `max(Qn) / mean(Q1..Q4)`
- Normal: < 1.3
- Elevated: 1.3–1.6
- Critical: > 1.6

---

## Report format

```
HEATLOAD & SKIN TEMPERATURE CHECK — [date] [shift]

OVERALL STATUS: [NORMAL / ELEVATED / HIGH / CRITICAL]

HEAT LOAD BY ROW (current vs 2-month baseline)
  Row 6: X GJ  (baseline mean Y GJ, z=Z) [status]
  Row 7: ...
  ...
  Row 10: ...

QUADRANT ASYMMETRY (R6-R10 average)
  Q1: X GJ | Q2: Y GJ | Q3: Z GJ | Q4: W GJ
  Max/Avg ratio: [value] [status]

TEMPERATURE PROFILE CHECKS
  18660 mm (Upper Stack) — spread: X°C  [status, channeling indicator]
  15162 mm (Lower Stack) — spread: X°C  [status]
  12975 mm (Belly)       — spread: X°C  [status]
  8335 mm  (Tuyere/Bosh) — spread: X°C  [status]

FLAGGED SENSORS (z > 2):
  - [sensor name]: current X°C, baseline mean Y°C (z=Z) — [ELEVATED/HIGH/CRITICAL]

ACTIONS:
  1. [Most urgent action]
  2. [Second action if needed]
```

---

## Furnace zone context for interpretation

| Zone | Elevation | What elevated heatload means |
|---|---|---|
| Hearth | 0–5.5 m | Refractory wear, possible elephant foot |
| Tuyere/Bosh | 5.5–12.9 m | Raceway imbalance, PCI distribution issue |
| Belly/Stack | 12.9–20 m | Channeling gas flow, burden hang or slip |

---

## Known normal operating ranges (best-shift context)
- **TOTAL HEAT LOAD**: best-shift envelope — keep away from persistent elevation vs baseline
- **RAFTOC**: persistent elevation (> best-shift band) = early warning of over-heating
- **Delta-T by row**: higher delta-T in a single row vs others = localized hot zone
- **Upper stack (18660 mm) spread > 60°C**: combine with body_dp_total instability to confirm channeling

---

## Recommended plot
Generate a subplot figure:
1. **Top**: Bar chart — current heat load per row (R6–R10) vs baseline mean ± 1 std (error bars)
2. **Middle**: Grouped bar chart — current Q1/Q2/Q3/Q4 heat loads for each row (quadrant asymmetry)
3. **Bottom**: Scatter/line — temperature sensor readings at 18660 mm, 15162 mm, 12975 mm with baseline mean band shown as shaded region
