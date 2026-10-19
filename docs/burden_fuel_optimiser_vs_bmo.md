# Blend Mix Optimiser LP vs burden engine (ore + flux objective), 2,270 t/d

Yard: reference, 16 Oct 2026; prices and bounds from bmo_operator_inputs.yml; slag <= 340 kg/tHM, CaO/SiO2 1.05-1.15.

| Material | Engine, wet % | Page-9 LP, wet % |
|---|---:|---:|
| SINTER (SP-02) | 58.0 | 58.0 |
| GEOMIN CLO | 13.4 | 12.7 |
| LLOYDS CLO | 18.6 | 19.3 |
| NMDC ROM | 0.0 | 0.0 |
| LLOYDS PELLET | 10.0 | 10.0 |

| Flux | Engine, kg/tHM | Page-9 LP, kg/tHM |
|---|---:|---:|
| limestone | 13.7 | 14.4 |
| dolomite | 0.0 | 3.7 |
| quartz | 0.0 | 0.0 |

| | Engine | Page-9 LP |
|---|---:|---:|
| Slag, kg/tHM | 340 | 340 |
| CaO/SiO2 | 1.050 | 1.050 |
| Al2O3, % | 19.74 | 19.59 |
| MgO, % | 6.87 | 6.76 |
| MgO/Al2O3 | 0.348 | 0.345 |
| Ferrous charged, kg/tHM wet | 1,718 | 1,725 |
| Ore cost, Rs/tHM | 14,269 | 14,380 |

Binding in the engine: basicity min, slag cap. Slag-quality limits at their bound in the page-9 LP: MgO/Al2O3 min.

Same blend through both slag calculations: see docs/burden_fuel_optimiser.md. Both count TiO2, alkalis and sulphur in the slag total (page 9 from its analyses, the engine as a 1.2 % allowance). What still differs is the dust row and the Fe sizing: page 9's dust removes about 2.0 kg CaO and 1.3 kg MgO per tHM (the engine's 0.65 kg CaO, no MgO), so page 9 needs dolomite and a little Geomin swapped for Lloyds CLO to hold MgO/Al2O3; and it charges a few kg/tHM more ferrous.
