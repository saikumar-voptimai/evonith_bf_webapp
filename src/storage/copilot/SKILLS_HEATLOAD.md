# Skill: Check Heatloads & Skin Temperatures

## Purpose
Detect elevated heatloads and temperature spikes in the furnace body by comparing the last 8 hours of operating data against a 2-month rolling baseline. Flag any stave row, quadrant, or temperature zone showing abnormal values.

## Why it matters
Elevated heatloads indicate refractory wear, channeling gas flow, or raceway imbalance. Early detection prevents brick damage and unplanned shutdowns. Skin temperatures at 18660 mm (Lower Stack) are the primary channeling indicator.

---

## Data sources required

### Current state (last 8h)
- **Tool**: `fetch_online_data`
- **Groups**: `heatload_delta_t`, `temperature_profile`, `process_params`
- **Window**: `15 minutes` or `1 hour`
- **What to get**:
  - Heat load by row: `heat_load_row_6` through `heat_load_row_10`
  - Heat load by quadrant: `heat_load_r6_q1` through `heat_load_r10_q4`
  - Cross-row quadrant averages: `heat_load_row6_10_q1` through `heat_load_row6_10_q4`
  - Delta-T by row: `delta_t_avg_row6` through `delta_t_avg_row10`
  - Quadrant delta-T: `delta_t_avg_row6_10_q1` through `delta_t_avg_row6_10_q4`
  - Temperature profile (all 11 elevations):
    - Hearth (4373 mm): `temp_4373_a` through `temp_4373_g`
    - Tuyere (5411–8335 mm): `temp_5411_*`, `temp_5757_*`, `temp_6103_*`, `temp_6795_*`, `temp_7565_*`, `temp_8335_*`
    - Bosh (9105 mm): `temp_9105_*`
    - Belly (12975 mm): `temp_12975_a` through `temp_12975_d`
    - Lower Stack (15162 mm): `temp_15162_a` through `temp_15162_d`
    - Upper Stack (18660 mm): `temp_18660_a`, `temp_18660_b`, `temp_18660_c`, `temp_18660_d`
  - Total heat load: `body_dp_total` (as context), production rate: `production_per_hour`

### Baseline (2-month rolling)
- **Tool**: `fetch_ml_data`
- **start_time**: 60 days ago from today
- **end_time**: 8 hours ago (exclude current window)
- **columns filter**: `['total heat load', 'lower stack', 'belly', 'bosh', 'hearth', 'uptake temp']`
- **resample**: `'1h'` (native)

---

## Analysis methodology

### Step 1 — Fetch both datasets
1. Fetch online (last 8h)
2. Fetch ML static (60-day baseline, up to 8h ago)

### Step 2 — Compute baseline statistics
For each ML static column:
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
  Row 6: X MW  (baseline mean Y MW, z=Z) [status]
  Row 7: ...
  ...
  Row 10: ...

QUADRANT ASYMMETRY (R6-R10 average)
  Q1: X MW | Q2: Y MW | Q3: Z MW | Q4: W MW
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
