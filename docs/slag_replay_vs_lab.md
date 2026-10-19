# Slag replay against the lab, 2025-09-01 to 2026-10-14

284 days with a full charge record (hot metal >= 1,800 t, >= 120 charges, >= 3 slag analyses, lab chemistry for >= 98 % of the ferrous charge) out of 325. Model minus lab, daily means.

| | Engine (two-zone) | Page 9 now (TiO2 in, DPR dust) | Page 9, workbook dust row | Page 9 before (workbook dust, no TiO2) |
|---|---:|---:|---:|---:|
| Al2O3, % pt | +0.02 (MAE 0.53) | +0.01 (MAE 0.54) | +0.04 (MAE 0.55) | +0.24 (MAE 0.59) |
| MgO, % pt | +0.00 (MAE 0.35) | -0.06 (MAE 0.35) | -0.36 (MAE 0.45) | -0.28 (MAE 0.41) |
| SiO2, % pt | +0.30 (MAE 0.82) | +0.32 (MAE 0.83) | +0.72 (MAE 1.01) | +1.08 (MAE 1.27) |
| CaO, % pt | -0.24 (MAE 0.62) | -0.29 (MAE 0.63) | -0.54 (MAE 0.76) | -0.16 (MAE 0.61) |
| TiO2, % pt | -0.10 (MAE 0.12) | -0.13 (MAE 0.14) | -0.09 (MAE 0.12) | -1.13 (MAE 1.13) |
| CaO/SiO2 | -0.015 (MAE 0.041) | -0.017 (MAE 0.041) | -0.037 (MAE 0.051) | -0.037 (MAE 0.051) |
| Slag, kg/tHM (mean) | 338 | 338 | 336 | 333 |
| Al2O3-tracer slag, kg/tHM (mean) | 338 | 338 | 337 | 337 |

Lab means: Al2O3 18.84 %, MgO 7.68 %, CaO/SiO2 1.081, TiO2 1.13 %, oxide sum 99.8 %. DPR slag generation 333 kg/tHM (not measured), DPR dust 32 kg/tHM.

Basis: charged masses by material from `charge_data`, each material's lab chemistry averaged over the 7 days to the day, PCI / dust / hot metal from the DPR, the day's mean hot-metal analysis for both models (page 9 normally holds its configured HM chemistry). Fuel ash oxides are not in Neon: both use the calibration's coke / nut / PCI ash analysis. Page 9's dust now: the day's DPR dust-catcher and GCP masses with the analyses in setting_bmo.yml (CaO ~2.1 %, MgO 0.5-0.8 %); before: the workbook row (530 kg per charge, CaO 5.63 %, MgO 3.57 %). The engine's: the DPR dust mass with SiO2 7.6, Al2O3 5.1, CaO 2.0, MgO 0 %. The Al2O3-tracer slag is each model's Al2O3 mass over the lab's Al2O3 %.

`python scripts/replay_slag_vs_lab.py --start 2025-09-01 --out docs/slag_replay_vs_lab.md`

