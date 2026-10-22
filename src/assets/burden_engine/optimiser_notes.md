# Burden & Fuel Optimiser (page 11): notes for the furnace commentary

These notes are the static context the furnace commentary is given with every request, ahead of the fact register and
the live results. They say what the page's numbers are, how they were made and validated, and where they are weak.
They are written for a metallurgist and for the commentary model alike. Everything here holds for BF-02 at Evonith
Steel as of October 2026; the fact register version sent with the request governs any coefficient.

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
  a blend moves coke through its chemistry only. *STRATA-ω* (ASM-18, page default, STRATA 1.2 — the BF-02 gas-flow
  model, formerly OMEGA): omega from STRATA-Flow (GAS-13). The thermal reserve zone is six rings, each a
  counter-current exchanger between the gas and the ore's wüstite oxygen; transfer units grow with the blend's
  reduction rate (ISO 7215 RI and size of each ferrous, Fe-weighted) and the residence time in the zone. Omega responds
  to the blend (pellet and sinter reduce faster than lump ore; ROM and Donimalai slowest), production (shorter residence,
  GAS-16: data scale 0.68 ± 0.29), the coke layers, fuels and oxygen (scale 1.37 ± 0.28), and the blast moisture of the
  Wardha weather record (GAS-19: +0.0044 ± 0.0004 per g/Nm³). Blend scales (GAS-14): pellet 1.27 ± 0.43, lump ore
  0.47 ± 0.35. Coke responses of both models side by side: MOD-01..10 (e.g. 10 pt pellet for sinter: Fixed-ω −1.7,
  STRATA-ω −6.4 ± 1.7 kg/tHM). 0.01 of omega is about 2.9 kg/tHM of coke.
* **Reducibility is an assumption (GAS-21).** Sinter RI is the plant mean; pellet and lump-ore RI are literature and
  mineralogy values (pellet 65, Lloyds CLO 65, NMDC ROM 65, Donimalai 60, Geomin / Gomti 75 %). The plant data cannot
  separate RI sets. When a plan is Low, Investigate or outside what BF-02 ran, state its critical assumptions first.
* **Investigate tier (GAS-18).** Pellet above 25 % of ferrous or σ(coke) above 8 kg/tHM: beyond common practice. Such a
  plan is a trial proposal, never an operating plan; the default filter and auto mode exclude it.
* **Uncertainty and certainty (blend mode).** Each plan carries σ(coke) relative to the calibration (posterior of the
  blend scales, the operating scale, the moisture coefficient, surrogate error) and a class (GAS-18): High = σ ≤ 2
  kg/tHM and blend and production inside what BF-02 ran (1st–99th percentile of 200 days: pellet 0–15.4 %, sinter
  58.9–69.5 %, production 86.9–96.2 t/h); Medium = σ ≤ 5 and within 5 pt / 3 t/h of that range; Low otherwise. Not
  in σ: the calibration's own error (about ±2 kg/tHM, common to every plan) and the day-to-day spread of omega (about
  ±8 kg/tHM), which averages out over days. The saving against today's blend is shown with σ of the difference
  (parameters shared by both plans cancel). Options: a certainty filter (Medium or better / High only) restricts the
  optimiser; risk aversion k minimises cost + k·σ(cost); auto mode prices three candidates (lowest cost, cost + 1.28σ,
  High only) beside today's blend, picks by a fixed rule (lowest cost + 1σ; hold today's blend when the saving is
  inside 1σ; ASM-19) and offers a language-model review of that pick.
* **Out-of-sample test (GAS-20).** Each day's omega and burnt coke predicted from the previous 14 days with every
  parameter estimated on earlier days only (175 days, Jan–Oct 2026). Skill against holding omega: v1.1 +0.25 (omega),
  +0.18 (burnt coke, day and 7-day); in time splits +0.22 and +0.16 (burnt coke). v1.0 was worse than holding omega
  in every test (−0.14 to −0.33). The skill comes from the moisture and operating terms; the blend term is neither
  confirmed nor rejected by the record (a model without it scores +0.21, difference not significant, t 1.0).
* **What the plant data could not settle.** The pellet start of 1 June 2026 coincides with the weather record's
  monsoon step (+13 g/Nm³ between the weeks either side); with the moisture coefficient estimated from within-month weather variation,
  the June rise of omega leaves +0.021 ± 0.022 for the blend against +0.029 expected. Within periods the blend moves
  by 1–3 pt, too little to measure. Pellet and lump-ore reducibility are literature values (no plant RI). A switchback
  trial (pellet 5 ↔ 20 % against Lloyds CLO, 3-day blocks, 30–36 days, matrix and fuel practice constant) measures
  the effect to ±2.8 kg/tHM; a single step needs about ±8 kg/tHM.
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
  (DATA-04); the default is the month's typical ambient value.
* **Plausibility bands** (fact register): fuel rate 520–600 kg/tHM (coke wet + nut coke + PCI; the upper end is a
  limit row, below 520 a result is marked infeasible and not reported), HMT 1,440–1,520 °C, Si 0.20–0.80 %.

## 3. Model check: is the model right on today's furnace?

The page opens with the model against the furnace, before any plan.

* **Burnt coke** (the measurement) = the dry coke rate that closes the oxygen balance at the metered blast air with
  the analyser's ηCO, plus the 14-day median of (charged − burnt). That offset carries the blast meter's scale error.
* **Energy balance** = the optimiser's model with omega and losses at the 14-day medians before the shift: out of
  sample. **Oxygen balance** = ηCO fixed at the analyser value.
* **Verdict.** On track when, over the chosen period's stable shifts, the energy model's bias is within ±8 kg/tHM and
  its SD within ±12 (about 1.5 × the validated shift SD). The validated record (DATA-06): by shift SD 7.7, MAE 6.0,
  r 0.91; daily SD 7.0 (5.5 Jul–Oct). On the validation workbook's last 14 days to 8 Oct 2026: bias −0.4 kg/tHM,
  ±3.9 over 40 stable shifts; oxygen balance +2.8 / ±4.2.
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
* **Result on 610 shifts, Sep 2025 – Oct 2026** (validation export):

  | Window | Windows | Bias, kg/tHM | SD | 95 % band | CCC |
  |---|---:|---:|---:|---:|---:|
  | Shift (8 h) | 610 | +0.4 | 11.2 | ±22 | 0.86 |
  | 1 day | 155 | +1.2 | 7.6 | ±15 | 0.88 |
  | 2 days | 83 | +1.2 | 6.1 | ±12 | 0.92 |
  | 1 week | 25 | −0.1 | 4.8 | ±9 | 0.93 |

  C_b stays at 0.99–1.00: no level or scale difference, the CCC is limited by scatter, and the scatter averages
  away (about ±10 kg/tHM per shift, falling as 1/√shifts) to a floor of about ±4.4 kg/tHM SD (±9 at 95 %), which is
  instrument drift inside the 14-day offset window. Shift-to-shift movements correlate 0.33 with the charged coke one
  shift earlier and 0.17 with the charged coke one shift later: burnt follows charged, not the reverse.
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

* Fixed 2,270 t/d with fuel: coke 305.7 kg/tHM dry, slag 340, Rs 30,139/tHM; sinter 58 / Lloyds CLO 18.6 / Geomin
  13.4 / pellet 10.0 %, limestone 13.7 kg/tHM; RAFT 2,214 °C.
* Maximum production 2,412 t/d (charging binds, pellet 19.8 % flagged as extrapolated); cheapest level 2,290 t/d at
  Rs 29,935/tHM; blast ≤ 105,000 Nm³/h → 2,257 t/d, O₂ 6,490 Nm³/h.
* Value of limits at 2,270 t/d: slag cap +10 kg/tHM → Rs 53/tHM, pellet floor −2 pt → Rs 44, basicity floor −0.01 →
  Rs 27.
* Page-9 LP and the engine on the same yard and objective agree within 0.3 pt of share; page 9 costs Rs 46–54/tHM more
  from charging about 4 kg/tHM more ferrous.

## 8. What the engine does not see

* With Fixed-ω, shaft efficiency does not respond to the blend. With STRATA-ω it does,
  but the charging matrix is held at the one in force over the calibration window (a matrix change is not an
  optimiser variable), and coke quality (CSR, M40), nut-coke rate, PCI and hot-blast temperature act on omega only
  through the calibration, not through the model.
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
