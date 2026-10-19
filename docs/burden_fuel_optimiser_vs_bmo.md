# Blend Mix Optimiser LP vs burden engine (ore + flux objective), 2,270 t/d

Yard: reference, 16 Oct 2026; prices and bounds from bmo_operator_inputs.yml; slag <= 340 kg/tHM, CaO/SiO2 1.05-1.15.

| Material | Engine, wet % | Page-9 LP, wet % |
|---|---:|---:|
| SINTER (SP-02) | 58.0 | 59.8 |
| GEOMIN CLO | 13.4 | 14.3 |
| LLOYDS CLO | 18.6 | 14.6 |
| NMDC ROM | 0.0 | 1.4 |
| LLOYDS PELLET | 10.0 | 10.0 |

| Flux | Engine, kg/tHM | Page-9 LP, kg/tHM |
|---|---:|---:|
| limestone | 13.7 | 13.4 |
| dolomite | 0.0 | 2.5 |
| quartz | 0.0 | 0.0 |

| | Engine | Page-9 LP |
|---|---:|---:|
| Slag, kg/tHM | 340 | 340 |
| CaO/SiO2 | 1.050 | 1.050 |
| Al2O3, % | 19.74 | 20.00 |
| MgO, % | 6.87 | 6.90 |
| MgO/Al2O3 | 0.348 | 0.345 |
| Ore cost, Rs/tHM | 14,269 | 14,350 |

Binding in the engine: basicity min, slag cap. Slag-quality limits at their bound in the page-9 LP: Al2O3 max, MgO/Al2O3 min.

Why the slag numbers differ for the same blend (see docs/burden_fuel_optimiser.md): the Al2O3 mass is the same in both, but the page-9 slag total leaves out TiO2 (about 4 kg/tHM; 'not added to slag per spec') and has no allowance for K2O + Na2O + S + P2O5, so it is about 10 kg/tHM lighter and its Al2O3 % reads about 0.6 pt higher. Its dust row also removes about 2 kg CaO and 1.3 kg MgO per tHM, so its basicity reads about 0.02 lower. The lab slag analyses include TiO2 (1.1-1.3 %) and alkalis + S (about 1 %), as the engine does.
