# Blend Mix Optimiser LP vs burden engine (ore + flux objective), 2,270 t/d

Yard: reference, 16 Oct 2026; prices and bounds from bmo_operator_inputs.yml; slag <= 340 kg/tHM, CaO/SiO2 1.05-1.15.

| Material | Engine, wet % | Page-9 LP, wet % |
|---|---:|---:|
| SINTER (SP-02) | 58.0 | 58.0 |
| GEOMIN CLO | 13.4 | 13.3 |
| LLOYDS CLO | 18.6 | 18.7 |
| NMDC ROM | 0.0 | 0.0 |
| LLOYDS PELLET | 10.0 | 10.0 |

| Flux | Engine, kg/tHM | Page-9 LP, kg/tHM |
|---|---:|---:|
| limestone | 13.7 | 13.5 |
| dolomite | 0.0 | 0.0 |
| quartz | 0.0 | 0.0 |

| | Engine | Page-9 LP |
|---|---:|---:|
| Slag, kg/tHM | 340 | 340 |
| CaO/SiO2 | 1.050 | 1.050 |
| Al2O3, % | 19.74 | 19.78 |
| MgO, % | 6.87 | 6.83 |
| MgO/Al2O3 | 0.348 | 0.345 |
| Ferrous charged, kg/tHM wet | 1,718 | 1,722 |
| Ore cost, Rs/tHM | 14,269 | 14,315 |

Binding in the engine: basicity min, slag cap. Slag-quality limits at their bound in the page-9 LP: MgO/Al2O3 min.

Same blend through both slag calculations: see docs/burden_fuel_optimiser.md. Both count TiO2, alkalis and sulphur in the slag total (page 9 from its analyses, the engine as a 1.2 % allowance), and both take the top-gas dust from the DPR with the plant's dust analyses (page 9 per stream, the engine pooled and without MgO). What remains is the Fe sizing and the dust MgO (page 9 removes about 0.2 kg MgO/tHM more, so its MgO/Al2O3 limit binds a little earlier).
