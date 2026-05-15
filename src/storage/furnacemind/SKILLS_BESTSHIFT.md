# This is your best shift analysis skill with which you compare any shift data
# with the best shift characteristics that minimises UNIT COST

# Best Shift Characteristics — UNIT COST Minimisation Benchmark

This file is intended to be loaded as a static benchmark for **8-hour shift-to-best-shift comparison**.

## Scope and method
- Target used: **`UNITCOST LAKHS/THM`**
- Shift window: **8 hours**
- Best-shift envelope: **bottom 20% of 8-hour shifts by UNIT COST**
- Lag logic: for each parameter, hourly lags from **0 to 4 h** were tested; the lag with the strongest absolute hourly relationship to UNIT COST was then applied before 8-hour aggregation.
- Tier logic:
  - **Tier 1** = true furnace controls
  - **Tier 2** = semi-controllable raw-material / quality levers
  - **Tier 3** = state / stability / HM / slag / heat-load / asymmetry checks
- Important caution: regression coefficients below are **plant-specific directional sensitivities**, not proof of causality. Tier assignment follows controllability and furnace physics.

## Executive summary
- Best observed low-cost shifts clustered in **mid/late February 2026**, especially:
  - `2026-02-21 16:00`
  - `2026-02-18 00:00`
  - `2026-02-28 08:00`
  - `2026-02-15 16:00`
  - `2026-03-01 08:00`
  - `2026-02-19 16:00`
  - `2026-02-24 16:00`
  - `2026-02-13 16:00`
  - `2026-02-27 16:00`
  - `2026-02-20 16:00`
- The strongest operator-relevant patterns were:
  - **higher O2 enrichment**
  - **better burden-distribution angles** (lagged ~4 h)
  - **cleaner raw materials**, especially lower **PCI ash**, lower **coke ash**, and lower **coke moisture**
  - **lower ore TM**
  - stable **TOPBAR**, acceptable **HM Si**, controlled **heat load**, and low **quadrant asymmetry**

## Top lag-aware regression coefficients for Tier 1 + Tier 2
These are **standardized coefficients** from a lag-adjusted ridge regression on 8-hour shifts.

- **WEIGHTED_COKE_ANGLE**: `-3.732`
- **WEIGHTED_NON_COKE_ANGLE**: `+3.092`
- **O2 ENRICHMENT %**: `-1.204`
- **SINTER_BASICITY**: `-1.045`
- **NUTCOKE_ASH%**: `+0.657`
- **SINTER_AL2O3%**: `+0.488`
- **ORE_LOI%**: `+0.410`
- **SINTER_FEO%**: `-0.405`
- **HOT BLAST TEMP.OC**: `+0.390`
- **COKE_ASH%**: `+0.263`

Interpretation:
- Negative coefficient: higher parameter values tended to align with **lower UNIT COST** after controlling for the other Tier 1 / Tier 2 variables.
- Positive coefficient: higher parameter values tended to align with **higher UNIT COST**.
- Because burden chemistry and control variables co-move, **use the coefficients only as directional weightings**. For operating guidance, the **bands and sensitivity plateaus below are more reliable**.

---

## Tier 1 — True furnace control parameters

### 1) O2 ENRICHMENT %
- **Best lag:** `0 h`
- **Hourly relationship to UNIT COST:** strong and favorable
- **Best-shift operating band:** target around **4.4 to 4.8 %**
- **Adverse region:** below roughly **3.8 %**
- **Interpretation:** in this dataset, higher enrichment consistently aligned with lower UNIT COST.
- **Sensitivity:** lowest UNIT COST was seen around **4.8 %**, with a practical low-cost plateau from about **4.45 to 4.80 %**.
- **ETA-CO note:** ETA-CO did not improve monotonically across the whole enrichment range, so use enrichment mainly as a **cost/stability lever**, not as an ETA-CO-only lever.
- **Example best-shift dates:** `2026-02-21 16:00`, `2026-02-18 00:00`, `2026-02-28 08:00`, `2026-02-15 16:00`, `2026-03-01 08:00`

### 2) HOT BLAST TEMP.OC
- **Best lag:** `0 h`
- **Hourly relationship to UNIT COST:** favorable but weaker than O2 enrichment
- **Best-shift operating band:** keep near the upper operating range, roughly **1199 to 1225 °C**
- **Adverse region:** below roughly **1190–1195 °C**
- **Sensitivity:** cost improved as blast temperature moved upward, with the practical low-cost / ETA-CO plateau around **1205 to 1225 °C**
- **Saturation:** above about **1215–1225 °C**, the incremental benefit looked small in this 3-month data
- **Interpretation:** keep blast temperature high and stable, but do not expect it alone to explain most shift-to-shift cost variation
- **Example best-shift dates:** `2026-02-21 16:00`, `2026-02-18 00:00`, `2026-02-28 08:00`

### 3) TOPPRESSUREBAR
- **Best lag:** `3 h`
- **Signal strength:** weak but directionally favorable
- **Observed best-shift range:** very narrow in this dataset, around **1.347 to 1.348 bar**
- **Interpretation:** use as a **supporting control / constraint**, not the main discriminator
- **Practical rule:** avoid pressure weakening ahead of cost deterioration; assess in combination with TOPBAR and differential pressure

### 4) WEIGHTED_COKE_ANGLE
- **Best lag:** `4 h`
- **Signal strength:** strong
- **Best-shift operating band:** centered around **~27.3**
- **Adverse region:** centered around **~25.1 to 26.8**
- **Interpretation:** burden-distribution angle is one of the strongest lagged control signatures in the data
- **Rule:** evaluate distribution changes on a **t+2 to t+4 h** basis, not instantly

### 5) WEIGHTED_NON_COKE_ANGLE
- **Best lag:** `4 h`
- **Signal strength:** strong
- **Best-shift operating band:** centered around **~27.3**
- **Adverse region:** **~25.8 to 27.3**, with lower values generally worse
- **Interpretation:** non-coke distribution matters, but judge it with lag

### 6) CHARGES/HRS.
- **Best lag:** `0 h`
- **Best-shift operating band:** roughly **6.5 and above**
- **Adverse region:** around **6.35 to 6.54**
- **Interpretation:** lower charging frequency tended to align with higher UNIT COST, but this should be checked alongside distribution and permeability

### 7) STEAMKGS/HR.
- **Best lag:** `0 h`
- **Usefulness in this dataset:** low
- **Reason:** values were near zero for large parts of the window
- **Rule:** keep as a constraint / diagnostic only for this dataset

---

## Tier 1 sensitivity notes
- **O2 enrichment** is the clearest actionable lever in this history.
- **Hot blast temperature** helps, but much of its benefit appears to saturate once the furnace is already in the upper operating range.
- **Burden-distribution angles** must be treated as **lagged causal candidates** rather than same-hour controls.

---

## Tier 2 — Semi-controllable raw-material / quality levers

### 1) PCI_ASH%
- **Best lag:** `4 h`
- **Direction:** higher values hurt
- **Best-shift operating band:** roughly **8.4 to 10.4 %**
- **Adverse region:** above about **11.1 %**, especially **>11.3 %**
- **Sensitivity:** lowest cost occurred near **~9.7 %**; practical low-cost plateau was below roughly **10.4–10.9 %**
- **Rule:** treat rising PCI ash as a next-shift fuel-cost penalty signal

### 2) PCI_IM%
- **Best lag:** `0 h`
- **Direction:** higher values hurt
- **Best-shift operating band:** keep near the lower end of recent history
- **Rule:** minimize moisture/inert dilution in PCI as part of the fuel-cost envelope

### 3) COKE_MOIST%
- **Best lag:** `1 h`
- **Direction:** higher values hurt
- **Best-shift operating band:** roughly **2.3 to 3.0 %**
- **Strongest low-cost zone:** around **2.45 to 2.65 %**
- **Adverse region:** especially **>4.35 %**
- **Sensitivity:** cost rose sharply once coke moisture moved into the **4.3–5.0 %** band

### 4) COKE_ASH%
- **Best lag:** `0 h`
- **Direction:** higher values hurt
- **Best-shift operating band:** roughly **11.39 to 11.60 %**
- **Adverse region:** especially **>11.85 %**, and clearly worse above **12.0 %**
- **Sensitivity:** lowest-cost bins sat near **11.4–11.5 %**

### 5) NUTCOKE_ASH%
- **Best lag:** `2 h`
- **Direction:** higher values generally hurt
- **Interpretation:** weaker than PCI/coke ash, but still worth keeping close to recent low-cost values

### 6) NUTCOKE_MOIST%
- **Best lag:** `4 h`
- **Direction:** higher values generally hurt
- **Interpretation:** supportive quality lever, but weaker than coke moisture and coke ash in this window

### 7) SINTER_FEO%
- **Best lag:** `0 h`
- **Direction in regression:** mixed / confounded
- **Use:** monitor as part of the burden-reducibility package rather than as a single independent knob

### 8) SINTER_AL2O3%
- **Best lag:** `2 h`
- **Direction:** higher values hurt
- **Interpretation:** more alumina generally increases slag burden / viscosity burden and aligned with higher cost here

### 9) SINTER_SIO2%
- **Best lag:** `4 h`
- **Direction:** higher values generally hurt
- **Interpretation:** use as a burden-penalty feature together with Al2O3 and FeO

### 10) SINTER_BASICITY
- **Best lag:** `2 h`
- **Regression direction:** favorable when higher, but this is not monotonic in the raw bins
- **Practical interpretation:** best to keep within the plant’s proven best-shift envelope rather than push aggressively in either direction
- **ETA-CO note:** higher ETA-CO appeared more often at the upper end of the recent basicity range, but cost minimum was broad

### 11) ORE_TM%
- **Best lag:** `1 h`
- **Direction:** higher values hurt
- **Best-shift operating band:** roughly **3.05 to 3.30 %**
- **Adverse region:** especially **>3.63 %**, and clearly worse above **3.78 %**
- **Sensitivity:** cost was lowest around **3.1–3.2 %**

### 12) ORE_LOI%
- **Best lag:** `1 h`
- **Direction:** higher values hurt
- **Interpretation:** treat higher LOI as a fuel-cost penalty indicator

---

## Tier 2 sensitivity notes
- **PCI ash**, **coke ash**, **coke moisture**, and **ore TM** were the cleanest raw-material quality signals.
- Practical raw-material guidance from this history:
  - **PCI ash:** keep **below ~10.5–10.9 %**
  - **Coke ash:** keep near **11.4–11.6 %**, avoid **>11.85 %**
  - **Coke moisture:** hold near **2.4–2.8 %**, avoid **>4.35 %**
  - **Ore TM:** aim near **3.1–3.3 %**, avoid **>3.6 %**

---

## Tier 3 — Guardrails / stability / quality checks

### 1) TOPBAR
- **Best lag:** `4 h`
- **Meaning:** one of the strongest warning indicators
- **Rule:** weakening TOPBAR was a clear precursor of higher UNIT COST
- **Use:** guardrail, not optimization target

### 2) DIFFERENTIAL PRESSURETOTALBAR
- **Best lag:** `2 h`
- **Use:** read together with permeability and distribution changes
- **Rule:** if differential pressure and TOPBAR weaken together, do not push Tier 1 controls harder

### 3) PERMEABILITYKGS/HR.
- **Best lag:** `2 h`
- **Use:** health check only
- **Rule:** it is a state response, not a direct lever

### 4) CHEM_PCT_SI
- **Best lag:** `0 h`
- **Use:** main thermal-balance proxy
- **Rule:** if HM Si rises above the plant’s low-cost envelope while UNIT COST rises, treat the furnace as hotter / less efficient

### 5) CHEM_PCT_S
- **Best lag:** `0 h`
- **Use:** maintain within normal plant limits
- **Rule:** check jointly with slag basicity and FeO

### 6) SLAG_BASICITY / SLAG_PCT_FEO
- **Use:** molten-zone / reduction-loss guardrails
- **Rule:** do not optimize directly, but review when HM Si / heat load / UNIT COST deteriorate

### 7) TOTAL HEAT LOAD
- **Best lag:** `0 h`
- **Use:** cost/stability warning
- **Rule:** persistent high values relative to the best-shift envelope indicate a more expensive thermal state

### 8) RAFTOC
- **Best lag:** `4 h`
- **Use:** lower-furnace thermal guardrail
- **Rule:** persistent elevation is a warning flag before copying a best-shift pattern

### 9) bosh_quad_spread
- **Best lag:** `4 h`
- **Use:** main asymmetry proxy
- **Rule:** if one quadrant diverges and spread rises materially, inspect burden distribution, tuyere/raceway condition, local cooling, and gas flow before forcing a “best-shift” recipe

### 10) uptake_quad_spread
- **Best lag:** `2 h`
- **Use:** top-end gas-flow imbalance check
- **Rule:** supporting asymmetry indicator only

---

## Stability, molten-metal, heat-load, and asymmetry checks
- **Stability**
  - Keep **TOPBAR** close to the historical low-cost zone.
  - Read **PERMEABILITYKGS/HR.**, **DIFFERENTIAL PRESSURETOTALBAR**, and **TOPBAR** together before pushing enrichment or distribution harder.
- **Molten metal**
  - **HM Si** is the key thermal-balance check.
  - If HM Si moves high while cost is worsening, do not interpret a lower coke rate as sustainable efficiency.
- **Heat load**
  - **TOTAL HEAT LOAD** and **RAFTOC** should stay away from sustained adverse movement.
- **Asymmetry**
  - **bosh_quad_spread** is the most useful asymmetry metric in this file.
  - If one quadrant behaves differently from the others, inspect physical furnace condition first, then compare against best shift.

---

## Best actual historical shifts by UNIT COST
- `2026-02-21 16:00`
- `2026-02-18 00:00`
- `2026-02-28 08:00`
- `2026-02-15 16:00`
- `2026-03-01 08:00`
- `2026-02-19 16:00`
- `2026-02-24 16:00`
- `2026-02-13 16:00`
- `2026-02-27 16:00`
- `2026-02-20 16:00`

## How to use this file for shift-to-best-shift comparison
1. Compare the current 8-hour shift average of each **Tier 1** and **Tier 2** parameter against the **ideal band**.
2. Respect the **lag** shown for each parameter.
   - Example: for burden-distribution angles, compare using the **4-hour-lag-adjusted signal**, not only the same-hour value.
3. Use **Tier 3** only as guardrails.
4. Escalate when:
   - a Tier 1 or Tier 2 parameter sits in the **adverse band** for two consecutive shifts, or
   - a Tier 3 asymmetry / HM Si / heat-load check deteriorates while UNIT COST is rising.

## LLM-ready feature keys
- `shift_start`
- `unit_cost`
- `actual_fuel_rate`
- `eta_co`
- `tier1.o2_enrichment`
- `tier1.hot_blast_temp`
- `tier1.top_pressure`
- `tier1.weighted_coke_angle`
- `tier1.weighted_non_coke_angle`
- `tier1.charges_per_hr`
- `tier2.pci_ash`
- `tier2.pci_im`
- `tier2.coke_moist`
- `tier2.coke_ash`
- `tier2.nutcoke_ash`
- `tier2.nutcoke_moist`
- `tier2.sinter_feo`
- `tier2.sinter_al2o3`
- `tier2.sinter_sio2`
- `tier2.sinter_basicity`
- `tier2.ore_tm`
- `tier2.ore_loi`
- `tier3.topbar`
- `tier3.bottom_bar`
- `tier3.permeability`
- `tier3.diff_pressure`
- `tier3.hm_si`
- `tier3.hm_s`
- `tier3.slag_basicity`
- `tier3.slag_feo`
- `tier3.total_heat_load`
- `tier3.raft`
- `tier3.bosh_quad_spread`
- `tier3.uptake_quad_spread`