# Unit Cost & Burden Distribution — Analysis Findings

**Last updated:** 2026-04-04
**Source:** Manual analysis — update after re-running `notebooks/burden_unitcost_analysis.ipynb`
**Covers:** Burden distribution impact on Unit Cost, with process parameter controls

---

## How to update this file

After running a new regression or analysis:
1. Update the findings sections below with new coefficients, envelopes, and best-window examples.
2. Update the `Last updated` date above.
3. No Python changes needed — the page reads this file at render time.

---

## Unit Cost Definition

```
Unit_Cost = Coke Rate (Kg/Thm) + 0.53 × PCI Rate (ActualKg/Thm.)
```

---

## Modelling Approach

**Model:** Transparent linear model (OLS) to reduce confounding.

**Controls held constant:**
- Hot Blast Temp
- TopPressureBar
- O₂ Enrichment %
- ETA CO *(FurnaceTopGasAnalysisCO2ETACO)*
- PCI ActualKg/Thm

**Burden distribution features:**
- `portions_total_COKE`, `portions_total_NON COKE`
- `angle_wmean_COKE`, `angle_wmean_NON COKE`
- `outer_share_COKE`, `outer_share_NON COKE`
- `lmg_angle`

**Model quality:**
- **R² ≈ 0.43** on ~5,900 valid rows *(burden + controls explain ~43% of Unit Cost variance)*

---

## Most Influential Burden Effects *(holding controls constant)*

- **More NON-COKE portions → lower Unit Cost** *(strong, significant)*
  Intuition: better ore/sinter coverage enabling efficient gas flow.

- **More COKE portions → higher Unit Cost** *(strong, significant)*
  Intuition: coke rings consume more and drive cost up.

- **Pushing NON-COKE outward (higher `angle_wmean_NON COKE`) → lower Unit Cost** *(significant)*
  Intuition: spreading ore toward the periphery improves permeability/ETA and saves fuel.

- **Higher LMG angle → slight reduction in Unit Cost** *(small but significant negative coefficient)*

- **Outer share of COKE:** small negative coefficient *(cost ↓)*, not statistically strong once other features enter.

---

## Process Controls (behaved as expected)

- **Higher PCI rate** and **higher ETA CO** both **reduce Unit Cost**.
- **Higher Hot Blast Temp** **increases** cost *(likely proxying for periods requiring more heat input)*.

---

## Best Burden Windows Found in History

Each distribution change interval was evaluated by the **realised mean Unit Cost** until the next change.

**Top performing windows (≥300 data rows; long enough to trust):**

| # | Date | Mean Unit Cost | COKE portions | NON-COKE portions | angle_wmean_COKE | angle_wmean_NON-COKE | outer_share_NON-COKE | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 2024-03-28 17:00 | 487.0 | 11 | 8 | 26.0° | 28.0° | 0.25 | "TO IMPROVE THE CENTER GAS FLOW"; LMG angle 42.5°, pattern P→C |
| 2 | 2024-11-08 12:47 | 489.5 | 37 | 24 | 27.4° | 28.5° | 0.25 | Multiple ring sets at same timestamp (summed); "TO INCREASE THE UTILISATION" |
| 3 | 2025-08-02 10:00 | 492.0 | 11 | 8 | 26.7° | 28.0° | 0.25 | — |
| 4 | 2024-08-20 20:00 | 493.9 | 11 | 8 | 26.8° | 28.3° | 0.33 | "TO CONTROL PRE" |

---

## Common Pattern Across Best Windows

- **Moderate COKE portions (~10–11)** and **adequate NON-COKE portions (~8)**.
- **NON-COKE weighted angle near ~28°** and **≥25% in the outer ring (≥32°)**.
- **LMG angle ~40–43** with **P→C charging pattern** frequently noted.
- These windows also show **good ETA CO** and **healthy PCI**.

> **Rule of thumb (from your data):**
> Keep **coke portions lean**, keep **non-coke portions ample**, and **bias the non-coke outward**
> *(centre-of-mass ~28° with ≥25% outer share).*

**Key driver summary:**
- `portions_total_NON-COKE` is the strongest cost **reducer**.
- `portions_total_COKE` is the strongest cost **increaser**.
- `angle_wmean_NON-COKE` reduces cost *(more outer non-coke)*.

---

## Why This Is Physically Faithful

- Respects the **6-row block design**; carries **date/time forward** so every material record is stamped correctly.
- Pairs each **"RINGS"** row with its **following "Angle"** row **only** to obtain degrees *(avoids double-counting portions)*.
- Separates **`metric=portions`** vs **`metric=percent`** so Extra-Coke "IN %" entries don't pollute portion counts.
- Models **change windows** *(pattern holds until next change)*, not isolated points.

---

## Actionable Recommendations

1. **Lock a candidate best pattern from history:**
   - ~10–11 COKE portions, ~8 NON-COKE portions
   - Target `angle_wmean_NON-COKE` ≈ 28°, ≥25% outer share
   - LMG angle ~40–43, charging pattern P→C

2. **What to avoid:**
   - Increasing COKE portions without compensating NON-COKE.
   - Inward-biased NON-COKE distribution (low `angle_wmean_NON-COKE`).

3. **Combined with process levers:**
   - Maintain high PCI rate and O₂ enrichment alongside a good burden distribution for maximum cost reduction.
   - ETA CO is the key feedback signal — burden changes that improve ETA CO will reduce cost.
