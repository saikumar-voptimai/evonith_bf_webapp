# BMO — Empirical findings on fuel, slag and silicon

Analysis record for the physics coke-rate correction in `src/utils/bmo/coke_correction.py`.
Everything below was measured on this plant's own data between 2026-07-30 and 2026-07-31.
Figures are reproducible from the sources named in each section.

**Read this before changing any coefficient in `bmo.coke_rate_correction`.**

---

## 0. Data sources and their reliability

| Source | What it is | Verdict |
|---|---|---|
| `offline_feed.charge_data` (Neon) | One row per charge, ~6.3 charges/hr | **Trustworthy.** Use for all mass work. |
| `src/assets/data/furnace_dataset.csv` | Hourly static ML dataset | Mixed — see below |
| `offline_feed.dpr_data` | Daily production report | **Coke and nut-coke masses unreliable** |

### 0.1 `COKE_CALC_MT` in the static CSV is unreliable

Correlation with the coke actually dumped per the charge reports, daily, n=223:

| Column | corr with charge reports |
|---|---|
| `COKE_CALC_MT` | **+0.16** |
| `NUTCOKE_CALC_MT` | +0.95 |

Only the coke column is affected. Any fuel-mass analysis must take coke from
`charge_data` (`coke_1_mt + coke_2_mt`), not from the CSV. An earlier round of
regressions built on `COKE_CALC_MT` produced slag coefficients ranging from +6 to
+36 with confidence intervals spanning zero; rebuilding on charge data collapsed
the noise and every specification became significant at p<0.001.

### 0.2 DPR masses under-report

Same-day comparison against the plant tags, n=271:

| | DPR | tag | ratio |
|---|---|---|---|
| coke | 274.9 | 309.3 | 0.871 |
| nut coke | 44.3 | 71.6 | **0.531** |
| PCI | 173.4 | 166.9 | 1.044 |
| total coke | 319.2 | 380.9 | 0.841 |

The gap is proportional rather than additive (ratio CV 0.12 vs 0.63) and day-to-day
correlation is only +0.53. DPR nut coke swings 15–75 kg/THM where the tag sits
steadily at 70–73. **DPR slag mass is still the only measured slag figure and is
usable; DPR coke and nut-coke masses are not.**

### 0.3 The coke setpoint is delivered with a ~4% excess

`COKE RATE KG/THM` is an operator setpoint, not a measurement. Against actual dumps:

- setpoint mean 305.0, delivered 323.6, **ratio 1.038**, corr +0.778
- within 10 kg/THM on only 42% of days

Either a scale/definition difference or a genuine overshoot. Unresolved.

### 0.4 Two slag measures, agreeing on level but not day to day

| | mean MT/day |
|---|---|
| Al₂O₃ tracer (from charge masses + slag chemistry) | 727 |
| DPR `slag_generation_mt` | 739 |

Ratio 0.980, but correlation only **+0.59**. The tracer is preferred for regression
because it derives from the same charge reports as the fuel masses, so measurement
error is not independent across the two sides of the equation.

Tracer method: all Al₂O₃ reports to slag, so
`slag_MT = (Σ Al₂O₃ in burden + flux + fuel ash) / (SLAG_PCT_AL2O3/100)`.

### 0.5 Reconciling calculated slag against the plant

`scripts/slag_calculation_report.py` traces every stage of
`calculate_full_slag_balance` for a named blend and compares the resulting slag
chemistry against the plant's own analysis.

**Diagnostic method.** CaO and Al₂O₃ are both inert — neither is reduced or
volatilised, so all of each reports to slag. Their *ratio in the slag must equal
their ratio in the total input*. Any mismatch is an input-side error, not a
process one. Dividing each calculated component mass by the plant's percentage
for that component gives an "implied total slag"; if the model were right, every
component would imply the same number.

**Result on the real charged mix, last 3 months:**

| Source | Al₂O₃ MT | CaO MT |
|---|---|---|
| burden (301,550 MT at measured weighted 2.71% Al₂O₃ / 7.27% CaO) | 8,172 | 21,923 |
| fuel ash | 3,154 | 464 |
| flux | 5 | 362 |
| **total** | **11,331** | **22,749** |

Input CaO/Al₂O₃ = **2.008** against slag CaO/Al₂O₃ = 36.63/18.71 = **1.958**, a
**+2.6% gap**. The component chemistry closes.

**The dominant discrepancy is flux quantity, not chemistry.** The plant charges
`FLUX_CALC_MT` ≈ 0.415 MT/hr = **10 MT/day**, which is 0.27% of the burden — the
sinter (basicity 2.01) carries essentially all the CaO. Running the same blend at
an assumed 160 MT/day of flux inflated the calculated slag from 789 to 874 MT and
pushed B2 from 1.060 to 1.286.

| Blend | flux MT/day | model slag MT | Al₂O₃ tracer implies | gap |
|---|---|---|---|---|
| current operation | 10 | 789 | 733 | +7.6% |
| lean + heavy LP flux | 260 | 961 | 729 | **+31.9%** |

So when the LP buys lean acidic ore and fluxes the basicity back up, the slag
estimate rises far faster than the plant's own experience — correctly in physics
(flux really does make slag) but well outside the operating envelope in §8.
**A BMO recommendation carrying large optimizer-added flux should be treated as
extrapolation.**

### 0.6 Two components the slag balance omits

Found while tracing the calculation:

1. **TiO₂ never enters the slag.** `calculate_full_slag_balance` builds
   `raw_slag_components` with keys for alkali, sio2, al2o3, cao, mgo, feo, mno, s
   and caf2 — there is **no `tio2` key**. Ti that does not go to hot metal is
   computed as `tio2_unaccounted_mt` (5.71 MT in the reference blend) and then
   discarded. Plant slag runs 0.52% TiO₂, i.e. ~4.1 MT.
2. **MnO comes out at zero.** All Mn is assigned to pig iron whenever HM Mn% ×
   PI mass exceeds the Mn available (0.25% × 2,343 = 5.86 MT capacity against
   4.89 MT available). Plant slag runs 0.30% MnO, i.e. ~2.4 MT.

Combined the model understates slag by ~6.5 MT, about **0.8%** — small beside the
flux effect, but systematic and one-directional.

---

## 1. The deployed fuel model cannot see the burden

Direct sweep of `src/assets/models/unitcost_fuel_model.json` (256 features):

| Perturbation | Effect |
|---|---|
| `ORE_CALC_THM` 0.515 → 1.03 (double ore per THM) | +1.6 Rs/THM out of 13,364 (**0.012%**) |
| `FLUX_CALC_THM` ×8 | −5 Rs/THM |
| `TOTAL_CLO_THM` ×1.5 | +1.2 Rs/THM |

That is an implied coke change of 0.06 kg/THM across a doubling of the ore burden.
`hm_si_model.json` is similarly flat (0.371 → 0.384 %Si at 2× sinter).

**Consequence:** the model output is effectively a constant ~13,364 Rs/THM. Back-solving
coke from it pins the reported total fuel rate to

```
total fuel ≈ (13,364 − 70×24)/28 + 70 + PCI×(1 − 18/28)
           ≈ 487 + 0.357 × PCI
```

i.e. 546–552 kg/THM for any realistic PCI, regardless of what the furnace is doing.
`bmo.fuel_rate_anchor_basis` is nevertheless set to **`model_cost`**: the reported
coke rate has to remain a *prediction* of what the blend will need. Anchoring on the
live coke tag (`observed`, still available) would turn an input into the answer —
every recommendation would open at today's coke rate and differ from it only by the
physics correction. The consequence of `model_cost` is that the anchor carries almost
no blend information, so **all** blend sensitivity comes from §2's correction.

Note the decomposition conserves **cost, not mass**: a PCI reading 15 kg/THM low does
not cut total fuel by 15, it cuts it by 5 and inflates coke by 9.

---

## 2. Slag → fuel: the central coefficient

Mass balance, `fuel MT/day ~ hot metal t/day + slag MT/day + controls`, 222 complete days.
Coke and burden from charge reports, slag from the Al₂O₃ tracer.

| Specification | kg fuel / 100 kg slag | 95% CI | t | R² |
|---|---|---|---|---|
| 1. hot metal only | +20.6 | [13.3, 27.9] | 5.54 | 0.765 |
| 2. + blast vol, temp, O₂, steam | +18.4 | [11.7, 25.1] | 5.38 | 0.812 |
| 3. + RM strength (RDI, RI, AI, TI) | +18.5 | [11.5, 25.6] | 5.15 | 0.813 |
| 4. + burden distribution | +26.9 | [19.5, 34.4] | 7.09 | 0.840 |
| 5. + Si into HM | +24.8 | [17.3, 32.2] | 6.49 | 0.846 |
| 6. + eta CO | +21.8 | [14.4, 29.1] | 5.78 | 0.857 |

Bootstrap (2,000 resamples, full specification): median **+20.9**, 90% CI [+11.4, +30.0],
P(coefficient > 0) = 1.00.

**Shipped value: 22 kg coke per 100 kg slag** (`bmo.coke_rate_correction.terms.slag_heat`,
changed from 30 on 2026-07-31). In physics terms that implies a marginal coke heat of
1.8 MJ/kg ÷ 0.22 = 8.2 MJ/kg. Pinned by
`tests/test_bmo_coke_correction.py::test_shipped_slag_coefficient_is_the_empirically_anchored_value`.

Sensitivity to the one uncertain physical constant:

| marginal coke heat MJ/kg | implied k per 100 kg |
|---|---|
| 5.0 | 36.0 |
| 6.0 | 30.0 |
| 7.0 | 25.7 |
| **8.2** | **22.0** |

### Shared-error check (2SLS)

The tracer includes coke ash, which is 19.5% of the Al₂O₃ and comes from the same charge
masses as the fuel figure. A day where coke mass is over-recorded raises *both* sides,
which would inflate the coefficient.

Instrumenting total slag with **burden-only** slag (sinter + ore + pellet + flux Al₂O₃,
sharing no input with the fuel masses):

| | value |
|---|---|
| stage 1, d(total slag)/d(burden-only slag) | 0.9923 (t = +51.5) |
| reduced form, d(fuel)/d(burden-only slag) | +19.9 per 100 (t = +5.03) |
| **IV estimate** | **+20.0 per 100 kg total slag, ±7.8** |
| OLS for comparison | +21.8 |

Shared error inflates the estimate by less than 2 units. **The coefficient is robust at
~20, and 22 sits within one standard error while erring on the cautious side** (a higher
value penalises lean burden more, which is the safe direction for a tool that would
otherwise over-buy cheap high-gangue ore).

**Caveat:** substituting DPR measured slag for the tracer gives **+9.0 ± 8.3**, not
significant. The +9 to +22 spread is genuine uncertainty about the slag measurement,
not about the fuel side.

### Estimates that should NOT be used

| Estimate | Value | Why it is wrong |
|---|---|---|
| DPR coke ~ slag, raw | +34.8 | Built on under-reported DPR coke (§0.2) |
| tag coke ~ slag, raw | +54 to +62 | Contaminated by coke↔PCI substitution |
| coke ~ slag + PCI controlled | −4 to −7 | Over-adjusted; PCI is the operator's *response* to slag, not a confounder. R² leaps 0.03 → 0.74 when PCI enters |
| any fuel-*rate* ~ slag-*rate* regression | ~0 | Both share HM in the denominator |

---

## 3. Raw-material strength: no measurable effect

Adding RDI, RI, AI and TI to the mass balance moves R² by **0.001** (0.812 → 0.813)
and the slag coefficient by 0.1.

Standardised effect on fuel rate, all controls in:

| Parameter | kg/THM per SD | t | p5–p95 range |
|---|---|---|---|
| sinter RDI | +1.38 | +0.96 | 28.1 – 38.9 |
| sinter RI | −0.00 | −0.00 | 64.3 – 72.2 |
| sinter AI | −0.47 | −0.53 | 4.91 – 5.31 |
| sinter TI | −1.62 | −1.68 | 78.5 – 80.2 |

**Read as underpowered, not refuted.** AI spans 0.4 units across the entire dataset;
there is almost nothing to regress against.

On RDI reportedly showing high impact in the fuel cost model: that is *feature
importance*, which measures split frequency and rewards high-cardinality noisy
variables. It is not a causal effect, and it comes from the model established in §1 as
blind to the burden.

---

## 4. Burden distribution: the largest lever in the data

Standardised effects on fuel rate, all controls in:

| Driver | kg/THM per SD | t |
|---|---|---|
| **non-coke angle** | **−13.90** | −4.98 *** |
| **coke angle** | **+10.93** | +4.04 *** |
| slag rate | +7.52 | +6.90 *** |
| coke portions | +4.07 | +2.23 * |
| eta CO | −3.77 | −3.33 *** |
| HM Si % | −2.45 | −2.63 ** |

Burden distribution outweighs slag rate. Signs match the AI Copilot page's documented
OLS finding: more coke portions costs fuel, non-coke angle outward saves it.

**Not in the correction model.** BMO does not optimise burden distribution, so it would
enter as a constant — but it is a bigger lever than raw-material quality and worth
knowing about when interpreting a recommendation.

---

## 5. Silicon

Physics: SiO₂ + 2C → Si + 2CO at 690 kJ/mol Si = 24.6 MJ/kg Si, giving ~4.1 kg coke of
heat plus 0.98 kg coke of carbon = **~5 kg coke per 0.1% Si**.

Observed: **−2.45 kg/THM per SD (t = −2.63)** — significant but **wrong-signed**. Higher
Si associates with *less* fuel. Same control-law contamination as everywhere: a hot
furnace makes high Si, and the operator responds by cutting fuel.

The Si term is shipped enabled but contributes ≈0 because its input model is blend-flat
(§1). It becomes live the moment a blend-sensitive Si model is deployed.
`tests/test_bmo_coke_correction.py` asserts its contribution stays under 1 kg/THM, so
nobody can "fix" its apparent uselessness by inflating the coefficient.

---

## 6. High slag costs production, not just fuel

Clean test with no HM in the regressor:

```
HM t/day ~ burden MT/day + slag-per-MT-of-burden        R² = 0.584
    burden           +0.6122 t HM per MT burden    t = +17.28
    slag/burden      −1280.8                       t = −1.82
```

**+1 percentage point more gangue per MT of burden costs ~13 t HM/day.**

The charging ceiling is real: charges/hr median 6.38, p95 6.75, p99 7.02, **max 7.25**.
At the ceiling the furnace cannot push more burden, so a leaner burden means less hot
metal rather than more fuel. BMO fixes target HM, so it represents the counterfactual
where that penalty appears as fuel instead — a regime the plant rarely operates in.

**Side finding:** actual total charge mass is **30.1 MT** (p95 31.2) against
`burden_capacity.charge_mass_mt: 26.4` in config. If 30.1 is right the capacity
constraint understates throughput by ~14%. Needs plant confirmation.

---

## 7. Why operational data keeps giving wrong signs

Contemporaneous correlations of fuel rate against thermal state are around **−0.45**:
operators cut fuel when the furnace runs hot. Cross-correlation at lag shows the sign
**flips positive at 4–9 h, peaking at 6–7 h** — exactly burden descent time. So the
causal direction is recoverable, but the recovered gains are 7–20× too weak (36 kg fuel
per 0.1% Si against a physics value of 5) because the controller cancels the excursion
you would need to measure. This is closed-loop identification bias and no amount of
retraining escapes it.

**Three artifacts to avoid in this dataset:**

1. **Ratio artifacts.** `slag_rate = slag/HM` and `fuel_rate = fuel/HM` share a
   denominator. Regressing one on the other, or splitting a sample on one while
   regressing the other, manufactures relationships. Work in masses.
2. **Over-adjustment.** Controlling for PCI removes the slag effect because PCI is the
   operator's response to slag, not an independent confounder.
3. **Multiple testing on an underpowered sample.** Regime splits produced two of three
   "significant" interactions with implied values as absurd as +104 kg fuel per 100 kg
   slag. Regime-specific coefficients are not supportable.

---

## 8. Operating envelope — where extrapolation begins

| Quantity | observed range |
|---|---|
| slag, DPR daily | p5 630 – p95 804 MT/day; **8 of 256 days > 800, zero > 850** |
| slag rate | ~324 – 386 kg/THM |
| burden Fe grade | 52.5 – 57.4 % |
| fuel rate (tags) | ~530 – 563 kg/THM, std 12.9 |
| charges/hr | up to 7.25 |

A 900 MT/day slag recommendation sits beyond anything the furnace has demonstrated.
The correction extrapolates linearly there, which is the optimistic direction — real
furnaces pick up non-linear penalties at high slag volume (viscosity, especially with
high-Al₂O₃ CLO; hearth drainage; gas flow) that this model does not represent.

---

## 9. Open questions

1. Why does `COKE_CALC_MT` disagree with the charge reports? (§0.1)
2. Why do DPR coke and nut-coke masses under-report by ~16% and ~47%? (§0.2)
3. Why is delivered coke ~4% above setpoint? (§0.3)
4. Is `charge_mass_mt: 26.4` correct when actual is 30.1 MT? (§6)
5. Would a handful of deliberate high-slag days at held production and held PCI
   identify the coefficient directly? That is the one experiment that would settle §2.
6. Should TiO₂ and MnO be added to `raw_slag_components`? Together ~0.8% of slag,
   currently dropped. (§0.6)
7. Is the LP allowed to add more flux than the plant would ever charge? At 10 MT/day
   actual against optimizer solutions carrying 100–260 MT/day, the slag estimate
   leaves the observed envelope quickly. (§0.5)
