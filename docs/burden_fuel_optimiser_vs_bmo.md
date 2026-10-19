# Blend Mix Optimiser LP vs burden engine (ore + flux objective), 2,270 t/d

Yard: reference, 16 Oct 2026; prices and bounds from bmo_operator_inputs.yml; slag <= 340 kg/tHM, CaO/SiO2 1.05-1.15.

| Material | Engine, wet % | Page-9 LP, wet % |
|---|---:|---:|
| SINTER (SP-02) | 58.0 | 58.0 |
| GEOMIN CLO | 13.4 | 13.0 |
| LLOYDS CLO | 18.6 | 19.0 |
| NMDC ROM | 0.0 | 0.0 |
| LLOYDS PELLET | 10.0 | 10.0 |

| Flux | Engine, kg/tHM | Page-9 LP, kg/tHM |
|---|---:|---:|
| limestone | 13.7 | 4.5 |
| dolomite | 0.0 | 21.5 |
| quartz | 0.0 | 0.0 |

| | Engine | Page-9 LP |
|---|---:|---:|
| Slag, kg/tHM | 340 | 340 |
| CaO/SiO2 | 1.050 | 1.050 |
| Al2O3, % | 19.74 | 20.00 |
| MgO, % | 6.87 | 8.14 |
| MgO/Al2O3 | 0.348 | 0.407 |
| Ore cost, Rs/tHM | 14,269 | 14,187 |

Binding in the engine: basicity min, slag cap. Slag-quality limits at their bound in the page-9 LP: Al2O3 max.

The engine's slag is the two-zone balance's own (fuel ash at the solved coke rate, dust, Si/Mn/Ti to hot metal); the LP's is utils.bmo.slag_balance with fixed fuel rates. Where the two slag models put a different Al2O3 / MgO on the same blend, the LP and the engine buy different flux (dolomite dilutes Al2O3; limestone only lifts basicity). Check both against the measured slag (register DATA-09: Jul-Oct Al2O3 18.9 %, MgO 7.18 %, MgO/Al2O3 0.381).
