# Burden & Fuel Optimiser (page 11)

Epic #260 · milestone #M4 · issues #250–#259

The Blend Mix Optimiser picks the cheapest ore + flux blend for a fixed production; fuel (about half the cost of a
tonne) stays out of the choice. This page optimises **ore + flux + fuel** per tonne of hot metal, and the **production
level** the day's limits allow. Coke is not a hand coefficient: for every candidate blend it is the coke the two-zone
(Rist) heat and mass balance needs for the target HMT and Si, calibrated on recent stable shifts (the Coke Energy
model, page 10, unchanged). PCI and nut coke are fixed inputs.

## What the operator gets

| Mode | Question | Engine call | Typical time |
|---|---|---|---|
| Step 1 · Cheapest production level | Under today's limits (blast, O₂, pressures, HBT, charging, yard, slag), what output is cheapest per tHM, how far can output go, and what does each extra tonne cost? | `min_total_cost` (+ frontier) | ~50 s |
| Step 2 · Fixed production | For a target t/d, which blend minimises ore + flux + fuel, and how does it differ from the ore + flux answer? What is relaxing each limit worth? | `min_cost(fuel=True)` and `min_cost(fuel=False)` | ~1 s |
| Maximum production | The ceiling for the day, the blend and controls that reach it, and what binds | `max_production` | ~10 s |

Every plan shows: blend (wet % of ferrous, kg/tHM, t/d), fuel & cost split, slag (rate, CaO/SiO₂, Al₂O₃, MgO,
MgO/Al₂O₃ against the limits), process (fuel rate, ηCO, top gas, blast, O₂, enrichment tag, RAFT, pressure needed,
charging), binding limits with their shadow prices, value in use per tonne of Fe, and the facts it rests on.

## How it works

```
 yard (BMO provider + saved inputs) ──► data.burden_engine.inputs ──► engine inputs
 Coke Energy pipeline (page 10 cache) ─► data.burden_engine.calibration ─► calibration + operating regime
 facts_register.json ──────────────────► utils.burden_engine.facts ──► rules, bands, default limits
                                                  │
                         utils.burden_engine.engine.Engine  (sequential LP, HiGHS)
                         coke, blast, O₂, slag, top gas = two-zone balance (utils.coke_energy.rist)
                                                  │
                         utils.burden_engine.scenarios ──► ui.burden_optimiser ──► page 11
```

* **Sequential LP.** Coke, blast and O₂ per tonne, slag oxides, top-gas temperature and the Fe balance are
  linearised around the current blend; HiGHS solves the LP with the page-9 rows (slag rate, basicity, Al₂O₃, MgO,
  MgO/Al₂O₃, shares, charging) plus the process rows; the balance is re-solved at the new blend; 2–5 LPs.
  Physics rows are elastic, so an infeasible day reports which limit blocks it.
* **Blast limit.** (Pb² − Pt²) = K·V^1.8 (bar absolute). Burden resistance K moves with pellet share, Gomti share,
  Lloyds oversize, O₂ and coke rate (register FLOW-02..07), relative to the calibration window's own operating point.
* **Calibration** (`live_calibration`): the stable shifts of the 14 days before today, from the same pipeline result
  as the Coke Energy page — shaft efficiency, lower / upper-zone losses (GJ/h), r_H, fuel and hot-metal analyses,
  dust, packed-bed anchor. Fewer than 10 stable shifts → the shipped Jul–Oct 2026 calibration, and the page says so.
* **Operating regime**: 14-day medians of HBT, blast, O₂, pressures, ΔP, HMT, Si, production, ηCO, top gas.
  These pre-fill the controls (Step 2) and appear as "today" next to each limit.
* **Fact register** (`src/config/facts_register.json`, v1.2): every coefficient outside the balance, the plausibility
  bands and the default limits. Retired facts (COKE-17 webapp hand coefficients, DATA-10 regressed RAFT) are
  refused in code.
* **Bands.** Fuel rate (coke wet + nut + PCI) above 600 kg/tHM is a limit row; below 520 the result is marked
  infeasible and not reported (a row there would make the optimiser burn coke). HMT outside 1,440–1,520 °C and Si
  outside 0.20–0.80 % are refused.
* **RAFT** is the plant workbook (`utils.energy_balance.raft`) at the input blast moisture (not measured at BF-02;
  default = the month's typical value from the Coke Energy table).

## Files

| Path | Role |
|---|---|
| `src/custom_pages/11_Burden_Fuel_Optimiser.py` | Page (UI orchestration only) |
| `src/ui/burden_optimiser/` | `sections.py` (renderers), `loaders.py` (cached yard / calibration / runs), `figures.py` (frontier) |
| `src/utils/burden_engine/engine.py` | `Engine`, `Settings`, `Limits`, `value_in_use`, `summary` |
| `src/utils/burden_engine/hydraulics.py` | Packed-bed limit, enrichment tag, per-tonne RAFT |
| `src/utils/burden_engine/facts.py` | Fact register loader (guarded access) |
| `src/utils/burden_engine/calibration.py` | Two-zone calibration and regime from pipeline frames |
| `src/utils/burden_engine/scenarios.py` | Mode runs and result tables |
| `src/data/burden_engine/inputs.py` | Yard / prices / limits from the BMO provider and `bmo_operator_inputs.yml` |
| `src/data/burden_engine/calibration.py` | Live calibration over the window, with fallback |
| `src/config/burden_optimiser.yml` | Page config (yard window, charging, thermal basis, frontier grid, calibration window) |
| `src/config/facts_register.json` + `schemas/facts_register.schema.json` | Fact register v1.2 |
| `src/assets/burden_engine/` | Reference yard (16 Oct 2026 UAT) and Jul–Oct 2026 calibration |
| `scripts/compare_bmo_vs_burden_engine.py` | Page-9 LP against the engine (ore + flux objective) |
| `tests/test_burden_engine*.py` | Engine, inputs, calibration and scenario tests |

Also changed for this page: `OreChemistry.loi_pct` (default 0; read from the same lab row, so page 9 is unaffected),
Titani ferrous CLO mapped to `ore_13` (was `ore_11`, the Donimalai oversize return).

## Acceptance checklist

Reference: *BF-02 blend optimiser with fuel cost and production: case report* (16 Oct 2026), UAT yard and prices,
PCI 190, nut coke 70, HMT 1,470 °C, Jul–Oct 2026 calibration. `pytest tests/test_burden_engine*.py`
(`BURDEN_ENGINE_FAST=1` skips the three slow production cases).

- [x] Case 2 (fixed 2,270 t/d, with fuel): coke 305.7 kg/tHM dry, slag 340, Rs 30,139/tHM, sinter 58 / Lloyds CLO 18.6 /
      Geomin 13.4 / pellet 10.0 %, limestone 13.7 kg/tHM, RAFT 2,214 °C
- [x] Case 2b (extended yard): Rs 29,306/tHM, Gomti 22 % flagged as extrapolated
- [x] Case 3 (max production): 2,412 t/d, charging binds, pellet 19.8 % flagged
- [x] Case 4 (cheapest level): 2,290 t/d at Rs 29,935/tHM, HBT 1,230 °C
- [x] Case 5b (blast ≤ 105,000 Nm³/h): 2,257 t/d, O₂ 6,490 Nm³/h, blast volume binds
- [x] Value of limits at 2,270 t/d: slag cap +10 → Rs 53, pellet floor −2 pt → Rs 44, basicity floor −0.01 → Rs 27
- [x] Band guards: ideal shaft (fuel 518) and +60 GJ/h lower-zone loss (fuel 611) not reported; HMT / Si out of band refused
- [x] Retired facts refused; engine coefficients equal the register's
- [x] Live calibration reproduces the window's actual coke at its own operating point; falls back below 10 shifts
- [x] Page opens without Neon / InfluxDB (reference yard + shipped calibration, labelled)
- [ ] On UAT with live data: yard chemistry dates and stock shown; calibration window = last 14 days; Step 2 at today's
      production within a few kg/tHM of the Coke Energy page's energy-model coke for the current blend
- [ ] Operator walk-through (below) signed off

## Comparison with the Blend Mix Optimiser LP

`python scripts/compare_bmo_vs_burden_engine.py [--live] [--target 2270]` runs both on the same yard, prices, bounds,
slag limits and page-9 dust rows, with fuel left out of the objective (`docs/burden_fuel_optimiser_vs_bmo.md`).

On the reference yard at 2,270 t/d the page-9 LP binds on Al₂O₃ ≤ 20 % and MgO/Al₂O₃ ≥ 0.345, the engine does not; the
page-9 blend moves about 4 points from Lloyds CLO to sinter, Geomin and NMDC ROM and costs Rs 81/tHM more in ore.

**Why, with one blend through both slag calculations** (engine's ore + flux blend, kg/tHM):

| | Engine | Page 9 (with its dust) |
|---|---:|---:|
| Al₂O₃ | 67.1 | 67.0 |
| SiO₂ / CaO / MgO | 116.9 / 122.7 / 23.4 | 117.8 / 121.4 / 22.1 |
| TiO₂ | 4.2 | 0 (left out "per spec") |
| K₂O + Na₂O + S + P₂O₅ allowance | 4.1 (1.2 % of slag) | 0 |
| **Slag total** | **340.0** | **330.0** |
| **Al₂O₃ %** | **19.74** | **20.31** |
| CaO/SiO₂ | 1.050 | 1.030 |

* Not the 20 % limit (both use the operator's 20 %) and not averaging (both divide total Al₂O₃ by total slag).
  The Al₂O₃ mass agrees; the **slag total** differs. Page 9's total excludes TiO₂ and has no allowance for alkalis,
  sulphur and P₂O₅, so it is about 10 kg/tHM lighter and every percentage in it reads higher: the 20 % cap bites
  about 0.6 pt earlier.
* Page 9's dust row (530 kg per charge) also removes about 2.0 kg CaO and 1.3 kg MgO per tHM; the engine's dust removes
  only CaO 0.65 kg and no MgO. So page 9's basicity reads about 0.02 lower and it needs more flux.
* The lab slag analyses include TiO₂ (1.1–1.3 %) and K₂O + Na₂O + S (about 1 %) in their total, as the engine does
  (its TiO₂ share, 1.23 %, matches). On that basis page 9 overstates slag Al₂O₃ % and understates basicity.
* To settle it on plant data: run both slag calculations on each day's actual charge and compare with the day's slag
  analysis (needs the Neon tables; #258).

## Operator walk-through

1. **Status line**: yard source (live / reference), calibration window and shaft efficiency, register version.
2. **Step 1**: open *Day limits* and set today's caps (e.g. blast 105,000 Nm³/h on a blower-limited day). Optimise.
   Read the cheapest output, the highest output, the cost of each extra tonne between them, and what binds.
   Switch *Plan shown below* to see either plan's blend and controls.
3. **Step 2**: enter the target t/d. Optimise. Compare the fuel-aware blend with the ore + flux answer (the KPI delta),
   then *Value of limits* for the cheapest limit to relax.
4. **Warnings** mean a rule is extrapolated (pellet > 16.6 %, Gomti > 11.6 %) or RAFT is outside its band; an error
   names the limit that blocks the plan.

## What the engine does not see

* Shaft efficiency is held at the calibration value for every blend (GAS-10): no measurable blend effect in the plant
  data; its day-to-day spread (±0.028, about ±7 kg/tHM coke) is the largest single uncertainty. Uncertainty and the
  certainty menu are #259 (phase 2).
* Process limits default to the Jun–Oct 2026 operating envelope, not equipment ratings. Oxygen and stove gas are
  unpriced.
* Not limited: phosphorus, alkalis, zinc, sulphur; the sinter plant's cost of a basicity change.
* Steady day average: no transition between blends; dust and heat losses (GJ/h) stay at the calibration.
