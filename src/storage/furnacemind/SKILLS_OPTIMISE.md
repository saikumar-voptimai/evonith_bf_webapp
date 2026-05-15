# Skill: Optimise Unit Cost

## Purpose
Analyse the current furnace state against the best-shift operating envelope and identify the highest-impact lever adjustments to reduce unit cost (Lakhs/tHM).

**Target variable**: `UNITCOST LAKHS/THM` (formula: `0.25 × (coke_rate + 0.53 × pci_rate)`)

---

## Key principle
This skill does NOT re-run regression. The regression has already been performed on plant history and the results are encoded in `SKILLS_BESTSHIFT.md`. This skill:
1. Fetches recent operating data (latest ML static rows = last 8h average)
2. Fetches baseline history (30 days) for context and variability
3. Compares current values against the best-shift bands from SKILLS_BESTSHIFT.md
4. Scores each parameter by: `gap_score = gap_from_band × |regression_coefficient|`
5. Ranks levers by impact score and provides targeted setpoint guidance

---

## Data sources required

### Recent data (for current state)
- **Tool**: `fetch_ml_data`
- **start_time**: 30 days ago
- **end_time**: now (default)
- **resample**: `'1h'` (native — need 8h tail for current shift average)
- **columns**: omit (return all — 142 columns, all needed)

---

## Tier 1 — Control parameters to check (with best-shift bands)

These are operator-adjustable NOW. Prioritise these in the report.

| Column in ML dataset | Best-shift band | Adverse region | Lag | Coeff |
|---|---|---|---|---|
| `O2 ENRICHMENT %` | 4.4 – 4.8 % | < 3.8 % | 0h | -1.204 |
| `HOT BLAST TEMP.OC` | 1199 – 1225 °C | < 1190 °C | 0h | +0.390 |
| `TOPPRESSUREBAR` | 1.347 – 1.348 bar | N/A | 3h | weak |
| `WEIGHTED_COKE_ANGLE` | ~27.3 | 25.1 – 26.8 | 4h | -3.732 |
| `WEIGHTED_NON_COKE_ANGLE` | ~27.3 | 25.8 – 27.3 | 4h | +3.092 |
| `CHARGES/HRS.` | >= 6.5 | 6.35 – 6.54 | 0h | — |
| `STEAMKGS/HR.` | diagnostic only | — | 0h | — |

**Note on burden angles**: because of the 4h lag, compute the 4h-lagged average (i.e. values from t-4h to now) for comparison against the band.

---

## Tier 2 — Raw material / quality flags (for next procurement cycle)

Report these as advisory — not operator-adjustable in the current shift.

| Column in ML dataset | Best-shift band | Adverse threshold | Lag | Coeff |
|---|---|---|---|---|
| `PCI_ASH%` | 8.4 – 10.4 % | > 11.1 % | 4h | — |
| `PCI_IM%` | keep low | elevated = warning | 0h | — |
| `COKE_MOIST%` | 2.3 – 3.0 % | > 4.35 % | 1h | — |
| `COKE_ASH%` | 11.39 – 11.60 % | > 11.85 % | 0h | +0.263 |
| `NUTCOKE_ASH%` | keep near best-shift | elevated = warning | 2h | +0.657 |
| `NUTCOKE_MOIST%` | keep near best-shift | elevated = warning | 4h | — |
| `SINTER_FEO%` | monitor | — | 0h | -0.405 |
| `SINTER_AL2O3%` | monitor | elevated = warning | 2h | +0.488 |
| `SINTER_BASICITY` | keep in proven range | — | 2h | -1.045 |
| `ORE_TM%` | 3.05 – 3.30 % | > 3.63 % | 1h | +0.410 |
| `ORE_LOI%` | keep low | elevated = warning | 1h | — |

---

## Tier 3 — Guardrails (check before any setpoint recommendation)

Do NOT recommend increasing enrichment or changing distribution if any of these are adverse.

| Column | Adverse condition | Action |
|---|---|---|
| `TOPBAR` | weakening trend | Do not push Tier 1 harder. Investigate stability first. |
| `DIFFERENTIAL PRESSURETOTALBAR` | rising | Assess permeability; check distribution |
| `PERMEABILITYKGS/HR.` | falling | Burden/gas flow issue; check distribution |
| `CHEM_PCT_SI` | > plant high-Si threshold | Furnace hotter than optimal; check fuel rate |
| `TOTAL HEAT LOAD` | elevated vs baseline | Thermal stress; do not increase enrichment aggressively |
| `RAFTOC` | elevated (>4h trend) | Lower-furnace warning; check tuyere condition |

---

## Analysis procedure (pre-computed — no tool calls required)

When this skill is invoked via the "💰 Optimise Unit Cost" button, the SkillEngine
pre-computes all values (8h averages, gap scores, benchmark) and builds the Tier 1
gap chart **before** the LLM is called. The chart is already stored in the Artifacts
panel. The user message will contain all computed numbers inline.

**Do NOT call `execute_python_plot` or any other tool for this skill.**
Simply read the pre-computed data in the user message and write the operator report.

---

## Report format

```
UNIT COST OPTIMISATION — [date] [shift]
Current unit cost (8h avg): X.XX Lakhs/tHM
Best-shift benchmark: Y.YY Lakhs/tHM (Feb–Mar 2026 best 20%)

GUARDRAIL CHECK: [CLEAR / WARNING — specify which Tier 3 param is adverse]

TOP TIER 1 ACTIONS (ranked by impact):
  1. [Parameter]: current X.X → target Y.Y | ACTION: [specific setpoint change]
     REASON: [regression signal from SKILLS_BESTSHIFT.md] | MAGNITUDE: [expected cost impact]
  2. ...
  3. ...

TIER 2 QUALITY FLAGS (advisory — next cycle):
  - [Parameter]: current X.X vs best-shift band Y.Y–Z.Z [WITHIN / ADVERSE]
  - ...

EVIDENCE: [which ML data range was used, how many rows]
```

---

## Important caveats to include in every response
1. These recommendations use **lagged regression signals** — burden angle changes take 4h to show cost impact.
2. Do not push O2 enrichment higher if guardrails (TOPBAR, permeability) are adverse.
3. Raw material quality (Tier 2) reflects procurement decisions — flag for the next shift/procurement meeting, not for immediate action.
4. Unit cost formula: `0.25 × (coke_rate + 0.53 × pci_rate)` — improvements require either lower coke rate, lower PCI rate, or both.
