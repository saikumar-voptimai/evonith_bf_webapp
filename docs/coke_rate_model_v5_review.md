# Coke-Rate Model V5 — What Was Done, and a Leakage Audit

**For independent review.** Script: `scripts/train_bmo_fuel_v5.py`
**Headline: the reported R² of +0.43 does not mean what it appears to.** See §4.

---

## 1. What was done

Ported the V4 notebook (`scripts/BMO_V4_Final_corrected (2).ipynb`) to a runnable
script, with four changes:

| # | Change | Reason |
|---|---|---|
| 1 | **Target changed** from unit cost to `COKE RATE KG/THM` | Unit cost expands to `4·coke + 24·TotalFuel − 6·PCI`, dominated by the quantity the plant holds flat |
| 2 | **Split fixed** to chronological | V4 cell 35 reads `shuffle=True` under a comment claiming `shuffle=False` |
| 3 | **Coke strength added** — M-40, M-10, CRI, CSR | Not in `furnace_dataset.csv`; pulled from `offline_feed.raw_material_strength_analysis` (325 daily rows), merged as-of |
| 4 | **14 energy features added** | Physics prior that responds to a blend never charged |

Data: `src/assets/data/furnace_dataset.csv`, 6,559 rows after the V4 cruising
filters, 2025-04-01 → 2026-08-21. Model and hyperparameters unchanged from V4
(XGBoost, 600 trees, depth 4, lr 0.02, α=3, λ=7).

### The V4 split defect

Same features, same model, two splits:

| Split | MAE (kg/THM) | R² |
|---|---|---|
| Random (as shipped in V4) | 3.18 | **+0.9668** |
| Chronological | 10.29 | **+0.3589** |

On hourly autocorrelated data a random split puts adjacent hours either side of
the fence. **The shipped model bundle was trained this way**, so its stated
accuracy is not what it delivers forward. This is independent of everything else
here and should be confirmed first.

---

## 2. The 14 energy features

All are per tonne hot metal. `prod` = `PRODUCTIONTONNESPERHR`; `ore`, `sinter`,
`pellet`, `flux` are the corresponding `*_CALC_MT` divided by `prod` × 1000.

| # | Feature | Formula |
|---|---|---|
| 1 | `SLAG_RATE_CALC_KG_THM` | Σ over ore, sinter, pellet, flux of `qty × (SiO₂+Al₂O₃+CaO+MgO)% / 100` |
| 2 | `BURDEN_TOTAL_KG_THM` | `ore + sinter + pellet + flux` |
| 3 | `FE_INPUT_KG_THM` | `ore×ORE_FE(T)% + sinter×SINTER_FE(T)%` + `pellet×PELLET_PCT_FE2O3×0.6994`, all /100 |
| 4 | `BURDEN_MOISTURE_KG_THM` | `(ore×ORE_TM% + pellet×PELLET_PCT_TM + flux×FLUX_TM%) / 100` |
| 5 | `FLUX_LOI_KG_THM` | `flux × FLUX_LOI% / 100` |
| 6 | `ENERGY_DEMAND_MJ_THM` | `Fe×7.38 + gangue×1.80 + 1378 + moisture×2.70 + LOI×4.05` |
| 7 | `BLAST_SENSIBLE_MJ_THM` | `blast_per_thm × 1.40 × (T_blast − 25) / 1000` |
| 8 | `BLAST_NM3_THM` | `HOT BLAST VOLUMENM3/HR. / prod` |
| 9 | `OXYGEN_NM3_THM` | `OXYGENFLOWNM3/HR. / prod` |
| 10 | `STEAM_KG_THM` | `STEAMKGS/HR. / prod` |
| 11 | `ENERGY_DEFICIT_MJ_THM` | `(6) − (7)` — energy carbon must supply |
| 12 | `IMPLIED_CARBON_KG_THM` | `(11) / 32.8` |
| 13 | **`IMPLIED_COKE_KG_THM`** | `((12) − PCI×0.75) / 0.87` — **the only one using PCI** |
| 14 | `DEMAND_PER_NM3_BLAST` | `(6) / blast_per_thm` |

Constants match `src/utils/energy_balance/`: Fe₂O₃→Fe 7.38 MJ/kg Fe, slag 1.80
MJ/kg, hot metal 1378 MJ/t, moisture 2.70 MJ/kg, calcination 4.05 MJ/kg CO₂,
blast cp 1.40 kJ/Nm³K, C→CO₂ 32.8 MJ/kg, C fractions coke 0.87 / PCI 0.75.

---

## 3. Reported results (chronological, before the audit)

| Variant | Features | MAE | R² |
|---|---|---|---|
| baseline | 125 | 10.29 | +0.3589 |
| + coke strength | 129 | 10.59 | +0.2879 |
| + energy terms | 139 | 9.41 | **+0.4302** |
| energy features only | 14 | 9.25 | +0.4076 |

This is what I reported initially, with the claim that "14 physics features beat
125 plant features". **That claim does not survive the audit below.**

---

## 4. LEAKAGE AUDIT — the finding that matters

### 4.1 PCI nearly determines coke rate, by construction

The V4 cruising filter keeps only rows where `ACT. FUEL RATEKG/THM.` ∈ [500,600].
Nut coke is held near 70. Therefore within the filtered set:

```
coke ≈ TotalFuel − 70 − PCI
```

Measured on the test period:

| Quantity | Value |
|---|---|
| corr(coke, PCI) | **−0.629** |
| corr(coke, TotalFuel − 70 − PCI) | **+0.716** |
| sd TotalFuel | 8.1 kg/THM |
| sd PCI | 17.2 kg/THM |
| sd coke | 16.1 kg/THM |

Total fuel barely moves while PCI moves twice as much as coke, so knowing PCI
gets you most of the way to coke without any physics at all.

**Decisive test — ablating PCI:**

| Feature set | Features | MAE | R² |
|---|---|---|---|
| **PCI alone** | **1** | **9.29** | **+0.3950** |
| PCI + blast volume + blast sensible | 3 | **8.11** | **+0.5764** |
| energy 14, all | 14 | 9.25 | +0.4076 |
| energy minus `IMPLIED_COKE_KG_THM` | 13 | 9.80 | +0.2479 |
| energy minus every PCI-bearing term | 11 | 9.75 | +0.2533 |

**A single feature — PCI — scores R² 0.395 against the full 14-feature set's
0.408.** Three features beat all fourteen. Removing the one feature that carries
PCI drops the set from 0.408 to 0.248.

So the 14 energy features add roughly **nothing** over PCI alone. The model is
substantially learning the arithmetic identity that the cruising filter created,
not the physics I intended to inject.

**Is this leakage?** PCI is a genuine control known at prediction time, so it is
not leakage in the strict sense. But the *filter* turns it into a near-accounting
identity for the target, and reporting R² 0.43 as evidence of predictive skill
overstates what has been demonstrated. I consider it target leakage induced by
the row filter, and I should have tested it before reporting §3.

### 4.2 Shared denominator — checked, largely clear

Every feature and the target are divided by `PRODUCTIONTONNESPERHR`, which can
manufacture correlation. Measured:

| Quantity | Value |
|---|---|
| production sd | 3.66 t/hr on a mean of 91.4 (4%) |
| corr(coke, production) | −0.358 |
| corr(`FE_INPUT_KG_THM`, production) | **−0.047** |
| corr(coke, `FE_INPUT_KG_THM`) | −0.017 |

Production varies little, and the energy features show almost no correlation
with it. This concern appears **not** to be driving the result — but it is worth
an independent check, since it is the failure mode I would expect next.

### 4.3 Contemporaneous state variables — NOT resolved

Inherited from V4 and left in the baseline feature set:

- **Furnace temperatures** — `HEARTH_TEMP_AVG` (rank 4 by gain), `BOSH_TEMP_*`,
  `BELLY_TEMP_*`. Hearth temperature at time *t* is a **consequence** of the coke
  charged, not a cause. Using it to predict the same hour's coke rate is reverse
  causality.
- **Pressures** — `BOTTOMBAR` (rank 6), `TOPBAR`.
- **`FURNACE TOP GAS ANALYSISH2%`** (rank 8) — top-gas hydrogen partly reflects
  fuel hydrogen, so it is downstream of the fuel rate.

V4 excluded `CO%`, `CO₂%` and `ETACO` as outcomes but kept temperatures,
pressures and H₂%. I kept that boundary rather than moving it, so this is
**unchanged from V4, not introduced by me** — but it is unjustified in both, and
it inflates the 125-feature baseline.

### 4.4 Coke strength forward-fill

M-40/M-10/CRI/CSR arrive on a roughly monthly lab cadence and are
forward-filled across hourly rows with a 21-day tolerance. This is
directionally safe (`merge_asof` with `direction="backward"` cannot see the
future) but produces long flat runs. It likely explains why coke strength
*hurt* this target (−0.071): little within-window variation, more width for the
tree to overfit.

### 4.5 What I believe is clean

- Target-bearing columns are excluded: `COKE RATE KG/THM`, `ACT. FUEL RATE`,
  `UNITCOST`, `COKE_CALC_MT/THM`, `NUTCOKE_CALC_MT/THM`, `COKE_OFF/ON_THM`.
  With coke as the target these are not precautions — total fuel *contains* coke.
- Scaler fitted on train only.
- `merge_asof` is backward-only.
- No target lags.

---

## 5. Honest bottom line

**What survives:** the target change from unit cost to coke rate is sound and
substantial. Unit cost shifted +150 Rs/THM between train and test halves (a tree
cannot extrapolate to that); coke rate shifts −0.5 kg/THM. That is a real fix,
independent of the PCI issue.

**What does not survive:** my claim that 14 physics features beat 125 plant
features. They score 0.408 against PCI-alone's 0.395. The physics is not
carrying the result.

**The number that actually matters for a blend optimiser** is the one with PCI
removed: **R² ≈ 0.25**. That is how much of coke-rate variance the blend and
burden explain at fixed PCI, and it is the honest ceiling for blend
discrimination on this record.

---

## 6. Suggested tests for the reviewer

Ranked by what would change the conclusion:

1. **Reproduce the PCI ablation** (§4.1). If PCI-alone really matches the full
   set, no feature engineering on this filtered dataset will help.
2. **Relax or remove the cruising filter** on total fuel and re-run. If R²
   collapses, the score was the filter's identity; if it holds, there is real
   signal underneath.
3. **Drop the contemporaneous state variables** (§4.3) — temperatures,
   pressures, top-gas H₂ — and re-measure the 125-feature baseline. I expect it
   to fall below the energy-only set.
4. **Predict absolute `COKE_CALC_MT`** from absolute quantities, no
   per-tHM normalisation, to close out §4.2 independently.
5. **Confirm the V4 split defect** (§1) and re-measure the shipped bundle.
6. **Check my energy formulas** against `src/utils/energy_balance/compute.py` —
   the feature versions are simplified (no top-gas term, no shell loss) and I
   have not verified they agree with the full balance.

---

## 7. Files

| Path | What |
|---|---|
| `scripts/train_bmo_fuel_v5.py` | The pipeline, all variants, and §5 reframings |
| `scripts/coke_rate_backtest.py` | Energy balance vs actual coke, 239 days |
| `scripts/coke_model_shootout.py` | ML vs energy balance vs persistence |
| `docs/coke_model_shootout.png` | Predicted vs actual, with residual panel |
| `src/utils/energy_balance/` | The full balance the features are simplified from |
