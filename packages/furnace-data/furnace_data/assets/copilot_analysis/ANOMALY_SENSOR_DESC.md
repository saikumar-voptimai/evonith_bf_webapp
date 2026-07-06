# Furnace Sensor & Zone Description

**Last updated:** 2026-04-04
**Used by:** Anomaly analysis prompt — update when sensors are added/removed or zones change.

---

## Temperature Profile Sensors

Field naming convention: `"Temperature Profile - BF2_BFBF Furnace Body [elevation_mm]mm Temp [circumferential_position]"`

| Elevation (mm) | Proxy name | Number of sensors | Circumferential positions |
|---|---|---|---|
| 4373 | — | 7 | A–G |
| 5411 | — | 13 | A–M |
| 5757 | Hearth | 13 | A–M |
| 6103 | — | 13 | A–M |
| 6795 | — | 12 | A–L |
| 7565 | — | 14 | A–N |
| 8335 | — | 14 | A–N |
| 9105 | — | 12 | A–L |
| 12975 | Bosh | 4 | A–D |
| 15162 | Belly | 4 | A–D |
| 18660 | Stack | 4 | A–D |

Tuyeres are located at **10500 mm**.

---

## Heatload Sensors

Field naming convention: `"Heatload Delta T - Heat load [row/zone] [quadrant]"`

- `Heat load R8 Q3 (Stave No 17-24)` — Average heatload for Row 8, Quadrant 3 (staves 17–24)
- `Heat load Row6-10 Q1 (Stave No 1-8)` — Average heatload across Rows 6–10, Quadrant 1 only
- `Heat load Row6` — Average heatload in Row 6 across all quadrants

Quadrant layout (32 staves per row):
- Q1: staves 1–8
- Q2: staves 9–16
- Q3: staves 17–24
- Q4: staves 25–32

---

## Process Parameters

Key fields available under `"Process Params - ..."`:
- Top pressure (bar), hot blast volume (Nm³/hr), hot blast temperature (°C)
- O₂ enrichment (%), steam injection (kg/hr)
- PCI coal rate (ActualKg/Thm.)
- ETA CO (gas utilisation efficiency)
- Permeability index, differential pressure (total, top, bottom)
- Fuel rate, coke rate

---

## Interpretation Notes

- **Skin temperature spread at 18660 mm (Stack):** asymmetry across A–D sensors is the primary channeling indicator. Normal spread < 40°C; critical > 80°C.
- **Heatload by row:** elevated Row 6 suggests hearth/bosh issues; elevated Row 8–10 suggests stack/burden issues.
- **Quadrant asymmetry:** max(Q1..Q4) / mean(Q1..Q4) > 1.6 is critical; indicates localised gas flow.
- **Permeability falling + ΔP rising:** burden hang or slip risk.
