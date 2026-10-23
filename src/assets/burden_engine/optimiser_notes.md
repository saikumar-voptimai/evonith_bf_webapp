# Burden & Fuel Optimiser (page 11): notes for the furnace commentary

These notes are the static context the furnace commentary is given with every request, ahead of the fact register and
the live results. They say what the page's numbers are, how they were made and validated, and where they are weak.
They are written for a metallurgist and for the commentary model alike. Everything here holds for BF-02 at Evonith
Steel as of 24 October 2026 (STRATA 1.4, fact register v1.7); the fact register version sent with the request governs
any coefficient.

## 1. What the page answers

* **Question.** Which ferrous blend (sinter, ores, CLO, pellet) and flux minimise **ore + flux + fuel** cost per tonne
  of hot metal, and what production the day's limits allow. The Blend Mix Optimiser (page 9) minimises ore + flux at a
  fixed production; fuel, about half the cost of a tonne, stays outside its choice. Page 11 puts fuel inside.
* **Modes.**
  * Step 1, cheapest production level: cost per tHM against output under the day's limits (the frontier), the
    cheapest point, the highest output, and the cost of each extra tonne between them.
  * Step 2, fixed production: the cheapest blend at a target t/d, next to the ore + flux-only answer (page 9's
    objective) and, with the live yard, next to **today's charged blend** (last 24 h of `charge_data`) priced at the
    same output, HMT / Si target and controls.
  * Maximum production: the ceiling, the blend and controls that reach it, and what binds.
* **Fixed inputs.** PCI and nut coke rates are inputs, not choices. HMT and Si are targets the coke is solved for.

## 2. How coke is computed

* **Two-zone (Rist) heat and mass balance** (`utils.coke_energy`, the Coke Energy Model of page 10, unchanged). For
  each candidate blend the coke rate is what the balance needs to make hot metal at the target HMT and Si with that
  burden, blast and PCI. It is not a hand coefficient; the webapp's old hand coefficients (COKE-17) are retired and
  refused in code.
* **Calibration.** Shaft efficiency (omega), lower- and upper-zone heat losses (GJ/h), r_H and the packed-bed anchor
  are the medians of the stable shifts of the 14 days before today, from the same pipeline result as page 10. With
  fewer than 10 stable shifts the shipped Jul–Oct 2026 calibration is used and the page says so.
* **Shaft efficiency: two modes.** *Fixed-ω* (Rist baseline, ASM-01): omega at the calibration value for every blend;
  a blend moves coke through its chemistry only. *STRATA-ω* (ASM-18, page default, **STRATA 1.4** — the BF-02 gas-flow
  model, formerly OMEGA): omega from STRATA-Flow network solves made for every plan (GAS-13; no surrogate). The thermal
  reserve zone is six rings, each a counter-current exchanger between the gas and the ore's wüstite oxygen; transfer
  units grow with the blend's reduction rate and the residence time in the zone. The reduction rate follows particle
  size, (11.25/d)^1.0: the laboratory RI differences (GAS-21) show no effect on the plant's omega, so the calibrated
  RI exponent is 0. Omega responds to the blend (pellet reduces fastest by size; NMDC ROM and Donimalai lumps slowest),
  the coke layers, fuels, oxygen and blast (other-operation scale 0.83 ± 0.29), nut coke (GAS-23: +0.00093 ± 0.00035
  per kg/tHM) and blast moisture (GAS-19: +0.0016 ± 0.0024 per g/Nm³, not resolved). Blend scales (GAS-14): pellet
  1.39 ± 0.46, lump ore 0.97 ± 0.46. **There is no production channel in STRATA 1.4** (GAS-16 retired): the
  residence-time penalty of the network was not replicated on 2023–24 data and lowered out-of-sample skill in both
  periods; production lowers coke through the balance (fixed heat loss per hour spread over more tonnes; GAS-22).
  Coke responses of both models side by side: MOD-01..11 (10 pt pellet for sinter: Fixed-ω -1.7, STRATA-ω
  -5.8 ± 1.1 kg/tHM; nut coke +10 kg/tHM: -9.9 and -13.0; production +1 t/h: -0.38 and
  -0.41). 0.01 of omega is about 2.5 kg/tHM of coke.
* **Reducibility and RDI are measured (GAS-21, DATA-13).** Evonith laboratory tests as given: RI pellet 82, Lloyds CLO
  84, Geomin 79, NMDC ROM 61, Donimalai 78 %; RDI 7-28 %; Gomti not tested (65 % assumed). The ore RDIs act in the
  network as shaft fines. The RI-to-rate exponent (0) and the size exponent (1.0) are calibrated on the plant data and
  are the critical assumptions to state first for a plan that is Low, Investigate or outside what BF-02 ran.
* **Two oversize materials (GAS-24).** Lloyds fines oversize (ore_6, 8-16 mm, Lloyds chemistry) reduces at the Lloyds
  CLO size (no size credit; the plant data reject the credit); Donimalai oversize return (ore_11) is Donimalai.
* **Investigate tier (GAS-18).** Pellet above 25 % of ferrous or σ(coke) above 8 kg/tHM: beyond common practice. Such a
  plan is a trial proposal, never an operating plan; the default filter and auto mode exclude it.
* **Uncertainty and certainty (blend mode).** Each plan carries σ(coke) relative to the calibration (posterior of the
  blend scales, the operating scales, the moisture and nut-coke coefficients) and a class (GAS-18): High = σ ≤ 2
  kg/tHM and blend and production inside what BF-02 ran (1st–99th percentile of 191 days: pellet 0–15.4 %, sinter
  58.9–69.5 %, production 86.9–96.2 t/h); Medium = σ ≤ 5 and within 5 pt / 3 t/h of that range; Low otherwise. Not
  in σ: the calibration's own error (about ±2 kg/tHM, common to every plan) and the day-to-day spread of omega (about
  ±8 kg/tHM), which averages out over days. The saving against today's blend is shown with σ of the difference
  (parameters shared by both plans cancel). Options: a certainty filter (Medium or better / High only) restricts the
  optimiser; risk aversion k minimises cost + k·σ(cost); auto mode prices three candidates (lowest cost, cost + 1.28σ,
  High only) beside today's blend, picks by a fixed rule (lowest cost + 1σ; hold today's blend when the saving is
  inside 1σ; ASM-19) and offers a language-model review of that pick.
* **Out-of-sample test (GAS-20).** Each day's omega and burnt coke predicted from the previous 14 days with every
  parameter estimated on earlier days only (175 days, Jan–Oct 2026). Skill of STRATA 1.4 against holding omega: omega
  +0.155, burnt coke +0.139 (day; RMSE 6.44 against 6.94 kg/tHM) and +0.221 (7-day); time splits
  +0.147 and +0.114. By blend family (VAL-01..12) seven groups are Supported, one Neutral, four Modelled only
  (fewer than 10 days), none Disputed. Per ore (GAS-26), no ore is over- or under-credited in 2026 (joint p 0.70).
* **Strict test on unseen 2023–24 data (GAS-25).** 413 stable days, Mar 2023 – Jul 2024, every parameter frozen at
  the 2026 fit: day RMSE of burnt coke Fixed-ω 7.95, STRATA 1.2 7.48, 1.3 7.57, **1.4 7.40**
  (skill 0.133; paired t -2.7 against Fixed-ω). The regime differed (coke 409 against 309 kg/tHM, PCI
  107 against 190, steam injection); the ω level was the same (0.80). The gain is mainly the moisture term (steam).
  In 2023–24 every model, Fixed-ω included, over-credits CLO and NMDC replacing sinter (+9 and +12 kg/tHM per 10 pt):
  a sinter-quality effect outside the model; sinter strength is not in the 2023–24 files (GAS-26, Measured gap).
* **What the plant data could not settle.** Within periods the blend moves by 1–3 pt; the pellet effect is resolved to
  about ±4 kg/tHM per 10 pt (GAS-15: +0.0123 ± 0.0041 ω per 10 pt replacing sinter). A switchback trial (pellet
  10 ↔ 20 % against Lloyds CLO, 3-day blocks, 30 days, matrix and fuel practice constant) measures the effect to
  ±2.8 kg/tHM; a single step needs about ±8 kg/tHM.
* **Gas flow and wall flow by material (FLOW-15, FLOW-16).** Within days and within months the burden resistance does
  not fall when lump ore or pellet replaces sinter (+0.00 to +0.06 in ln K per 10 pt; the network predicts −0.04 to
  −0.07); Gomti raises it in both. Lump ore raises the row 6–10 stave heat load (+0.05 to +0.41 in ln(heat load) per
  10 pt); pellet does not. The blast-limit rules (FLOW-02..07) are unchanged.
* **Solver.** Sequential LP (HiGHS): coke, blast and O₂ per tonne, slag oxides, top-gas temperature and the Fe
  balance are linearised around the current blend, the LP is solved with page 9's rows (slag rate, CaO/SiO₂, Al₂O₃,
  MgO, MgO/Al₂O₃, shares, charging) plus the process rows, and the balance is re-solved at the new blend; 2–5 LPs.
  Physics rows are elastic, so an infeasible day names the limit that blocks it.
* **Blast limit.** (Pb² − Pt²) = K·V^1.8 (bar absolute). Burden resistance K moves with pellet share, Gomti share,
  Lloyds oversize, O₂ and coke rate (FLOW-02..07), relative to the calibration window's operating point. The rules
  were fitted on pellet up to 16.6 % and Gomti up to 11.6 % of ferrous; beyond that the page warns that it is
  extrapolating (ASM-09).
* **RAFT** is the plant workbook formula at the input blast moisture. Blast humidity is not measured at BF-02
  (DATA-04); the default is the RH-scaled weather-record estimate (DATA-11: 31-day mean RH mapped to 7–21 g/Nm³).
* **Plausibility bands** (fact register): fuel rate 520–600 kg/tHM (coke wet + nut coke + PCI; the upper end is a
  limit row, below 520 a result is marked infeasible and not reported), HMT 1,440–1,520 °C, Si 0.20–0.80 %.

## 3. Model check: is the model right on today's furnace?

The page opens with the model against the furnace, before any plan.

* **Burnt coke** (the measurement) = the dry coke rate that closes the oxygen balance at the metered blast air with
  the analyser's ηCO, plus the 14-day median of (charged − burnt). That offset carries the blast meter's scale error.
* **Energy balance** = the optimiser's model with omega and losses at the 14-day medians before the shift: out of
  sample. **Oxygen balance** = ηCO fixed at the analyser value.
* **Verdict.** On track when, over the chosen period's stable shifts, the energy model's bias is within ±8 kg/tHM and
  its SD within ±12 (about 1.5 × the validated shift SD). The validated record (DATA-06, RH-scaled moisture, 545
  stable shifts Dec 2025 – Oct 2026): by shift bias -1.0, SD 7.8, MAE 6.1, r 0.91; one day SD 7.0; one week SD
  5.8. Oxygen balance with the analyser's ηCO: SD 4.8, r 0.97. Last 14 days to 7 Oct 2026: bias -1.8, SD 4.0 over
  38 stable shifts (97 % within ±10 kg/tHM).
* A warning appears when the latest scored shift is more than 2 days old; then the check is for that older period.
* **What to make of it.** On track means the coke figures of the plans can be read at face value within about
  ±8 kg/tHM per shift and ±5 kg/tHM per day. "Check before trusting the plan" means the calibration window no longer
  describes the furnace (a gas-distribution change, a heat-load excursion, an analyser or meter fault): blend
  rankings still hold better than the absolute coke figure.

## 4. Burnt coke against charged coke: the same coke, a descent time apart

* Coke tipped at the stockline reaches the tuyeres one burden-descent time later: **6–7 h** on BF-02 (the lag at
  which the thermal state answers a fuel change: fuel against thermal state flips positive at 4–9 h and peaks at
  6–7 h). In any window, burnt and charged coke differ by the change in coke in transit plus measurement error.
* **How it is shown** (`utils.coke_energy.transit`): Lin's concordance correlation coefficient (CCC = r × C_b, 1 only
  when the series are identical), Bland–Altman bias and 95 % limits, per averaging window; a fit of the window spread
  SD(W)² = A/W² (transit) + B/W (random) + C (floor); and the cross-correlation of shift-to-shift movements by lag.
* **Result on 568 stable shifts, 2025-09-13 – 2026-10-07** (pipeline result with RH-scaled moisture):

  | Window | Windows | Bias, kg/tHM | SD | 95 % band | CCC |
  |---|---:|---:|---:|---:|---:|
  | Shift (8 h) | 568 | +0.5 | 11.2 | ±22 | 0.86 |
  | 1 day | 200 | +0.5 | 7.9 | ±16 | 0.92 |
  | 2 days | 87 | +0.7 | 6.4 | ±12 | 0.93 |
  | 3 days | 60 | +0.8 | 5.8 | ±11 | 0.94 |
  | 5 days | 34 | +0.6 | 5.2 | ±10 | 0.92 |
  | 1 week | 23 | +0.3 | 5.2 | ±10 | 0.92 |

  C_b stays at 0.99–1.00: no level or scale difference, the CCC is limited by scatter, and the scatter averages
  away to a floor of about ±5 kg/tHM SD over a week, which is instrument drift inside the 14-day offset window.
  Shift-to-shift movements correlate 0.35 with the charged coke one shift earlier and 0.20 with the charged
  coke one shift later: burnt follows charged, not the reverse. The uncalibrated oxygen balance closes on the charged
  coke within 1.2 % over three months (1.2 % over six); coke in transit (≈ 133 t) is at most 0.3 % of that.
  Hourly data (charge dumps): gain 1.04 ± 0.31, no reverse link; 90 coke steps: delay 4 h (2–12 h), complete by 6–7 h.
* **Limits of this evidence.** The level agreement is partly by construction (the 14-day offset); the spread, its
  decay and the direction of the lag are not. 8-hour shifts with one shared production figure cannot measure a 6–7 h
  delay (the peak sits inside the shift); that needs hourly charge-dump times. Tonnes per hour cannot show the delay
  either: charging follows the stockline, so charged t/h trails burnt t/h by about an hour; it is the coke content of
  the burden that travels. Compare charged and burnt coke over a day or more, never shift by shift.

## 5. Slag, dust and flux

* Slag oxides follow page 9's basis: TiO₂ counts in the slag total; alkalis 80 % to slag; sulphur from ore, flux and
  fuel; MnO and FeO from the analyses; phosphorus to the hot metal.
* **Against the lab** (284 days, Sep 2025 – Oct 2026, each day's actual charge, model minus lab): engine Al₂O₃ +0.02
  pt, MgO +0.00, TiO₂ −0.10, CaO/SiO₂ −0.015; page 9 +0.01 / −0.06 / −0.13 / −0.017. MAE Al₂O₃ 0.54 pt, MgO 0.35 pt,
  CaO/SiO₂ 0.041. Slag rate 338 kg/tHM, equal to the Al₂O₃-tracer estimate.
* **Top-gas dust** (page 9): quantities from the DPR (dust catcher `flue_dust_mt`, GCP `gcp_dust_mt`) over the charges
  counted on the same days, about 219 and 237 kg per charge; chemistry configured from the plant's dust analyses
  (CaO about 2 %, MgO 0.5–0.8 %). The old workbook row (530 kg/charge, CaO 5.63 %, MgO 3.57 %) read slag MgO 0.36 pt
  and CaO/SiO₂ 0.037 low against the lab.
* **Flux** is page 9's catalogue with its saved prices (quartz, limestone, dolomite), chosen by the LP within its
  limits. Limestone brings CO₂ that the coke must calcine; the balance includes it.

## 6. Reading the results

* **Plan line** (Step 2): what to charge, ore + flux + fuel cost, coke, and the saving against today's blend in Rs/tHM
  and Rs lakh/day. The saving is a model difference at fixed controls and targets; it is real only if larger than the
  model's own error on the coke difference (a few kg/tHM of coke is Rs 100–150/tHM at Rs 32,000/t coke).
* **What limits the plan**: the LP's binding rows and their shadow prices (Rs/tHM per unit of extra room) at the
  final linearisation.
* **Value of limits** (Step 2): each limit relaxed by one step at the same production and re-optimised; the saving is
  what that limit costs today. Large values point at the limit worth questioning (for example a slag-rate cap or a
  share floor set by habit rather than by the furnace).
* **Value in use**: each material's cost per tonne of Fe including the flux to bring its own gangue to CaO/SiO₂ 1.10
  and the coke response at this plan. Negative flux or coke = a credit (sinter carries lime).
* **Frontier** (Step 1): below the cheapest output fixed heat losses spread over fewer tonnes; above it a limit forces
  dearer ore or controls. The marginal Rs per extra tonne is from the cost per day.

## 7. Validation against the case report (16 Oct 2026, UAT yard, PCI 190, nut coke 70, HMT 1,470 °C)

* Fixed-ω, UAT prices, no slag credit (the report's basis): fixed 2,270 t/d with fuel: coke 305.7 kg/tHM dry, slag
  340, Rs 30,139/tHM; sinter 58 / Lloyds CLO 18.6 / Geomin 13.4 / pellet 10.0 %, limestone 13.7 kg/tHM; RAFT 2,214 °C.
  Maximum production 2,412 t/d (charging binds); cheapest level 2,290 t/d at Rs 29,935/tHM; blast ≤ 105,000 Nm³/h →
  2,257 t/d. Page-9 LP and the engine on the same yard and objective agree within 0.3 pt of share.
* **At the plant prices of 23 Oct 2026 with the slag credit** (DATA-14, ASM-20; slag 340 kg/tHM × 457.85 Rs/t =
  Rs 156/tHM): Fixed-ω Rs 29,913/tHM, coke 305.7, same blend; STRATA 1.4 Rs 29,949 ± 11/tHM, coke
  304.6 ± 0.4 kg/tHM, sinter 61.5 / Lloyds CLO 16.1 / Geomin 12.3 / pellet 10.0 %, no flux (High certainty).
  STRATA carries more sinter: sinter reduces faster per unit Fe at its size, and its lime replaces the limestone.
* Pellet share at 2,270 t/d (rest re-optimised, plant prices): STRATA-ω cost is flat within Rs 30/tHM between 0 and
  12.5 % pellet; coke falls about 1.2 kg/tHM per 2.5 pt; tiers High to 15 %, Medium to 20 %, Low to 25 %, Investigate
  above. Fixed-ω coke is flat and its cost rises with the pellet price.

## 8. What the engine does not see

* With Fixed-ω, shaft efficiency does not respond to the blend. With STRATA-ω it does, but the charging matrix is held
  at the one in force over the calibration window (a matrix change is not an optimiser variable); coke quality (CSR,
  M40) and sinter strength act on omega only through the calibration; production acts through the balance only (no
  residence-time channel in STRATA 1.4); PCI and hot-blast temperature act through the balance and the network's
  other-operation channel; nut coke also through its data term (GAS-23).
* Process limits default to the Jun–Oct 2026 operating envelope, not equipment ratings. Oxygen and stove gas have no
  price.
* Not limited: phosphorus, alkalis, zinc, sulphur; the sinter plant's cost of a basicity change.
* Steady day average: no transition between blends (a burden change takes a descent time to reach the tuyeres and
  longer to settle the hearth); dust and heat losses (GJ/h) stay at the calibration.
* No maintenance, tuyere, stove or casthouse events; nothing the previous shift saw that is not in the data.

## 9. Operator practice the commentary should respect

* Lead with feasibility and the model check. A saving on a plan that breaks a limit, or from a model that is off
  track, is not a saving.
* Move the burden in steps (page 9's transition ladder), one descent time apart, and watch HMT, Si, ΔP and the stave
  heat loads before the next step.
* A saving inside the model's error band is not a reason to change the burden. Say so plainly.
