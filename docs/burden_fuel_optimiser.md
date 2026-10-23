# Burden & Fuel Optimiser (page 11)

Epic #260 · milestone #M4 · issues #250–#259

The Blend Mix Optimiser picks the cheapest ore + flux blend for a fixed production; fuel (about half the cost of a
tonne) stays out of the choice. This page optimises **ore + flux + fuel** per tonne of hot metal, and the **production
level** the day's limits allow. Coke is not a hand coefficient: for every candidate blend it is the coke the two-zone
(Rist) heat and mass balance needs for the target HMT and Si, calibrated on recent stable shifts (the Coke Energy
model, page 10, unchanged). PCI and nut coke are fixed inputs.

## What the operator gets

**First, the model check** (top of the page, before any plan): is the energy model right on today's furnace? For the
latest scored shift it shows the coke the blast actually burnt next to what the energy balance (the optimiser's model)
and the oxygen balance predict for that shift's HMT, Si, blast and burden, both out of sample (calibrated only on the
14 days before the shift). Below that, the model's record over the last 7 / 14 / 30 days on stable shifts and a trend
of burnt coke against both predictions per shift. *On track* when the energy model's bias is within ±8 kg/tHM and its
typical error within ±12 (about 1.5 × the validated ±7.7, register DATA-06); otherwise *Check before trusting the
plan*, with the reason. A warning appears when the latest scored shift is more than 2 days old. Thresholds and the
default period are in `burden_optimiser.yml → model_check`. The numbers are the Coke Energy pipeline's shift table
(`PipelineResult.shifts`: `K_burnt`, `K_energy`, `K_obal`), the same result the optimiser calibrates on; the page
does no physics of its own here (`utils/burden_engine/model_check.py`).

The model check has a second tab, **Burnt against charged coke** (`utils/coke_energy/transit.py`): Lin's concordance
(CCC) and Bland–Altman per averaging window, the part of the spread that does not average out, and the lag
correlation. On 610 shifts: CCC 0.86 per shift, 0.88 per day, 0.93 per week, no level or scale gap, burnt follows
charged. Method, results and limits: `docs/burnt_vs_charged_coke.md`.

**Then the plan:**

| Mode | Question | Engine call | Typical time |
|---|---|---|---|
| Step 1 · Cheapest production level | Under today's limits (blast, O₂, pressures, HBT, charging, yard, slag), what output is cheapest per tHM, how far can output go, and what does each extra tonne cost? | `min_total_cost` (+ frontier) | ~50 s |
| Step 2 · Fixed production | For a target t/d, which blend minimises ore + flux + fuel, and how does it differ from the ore + flux answer? What is relaxing each limit worth? | `min_cost(fuel=True)` and `min_cost(fuel=False)` | ~1 s |
| Maximum production | The ceiling for the day, the blend and controls that reach it, and what binds | `max_production` | ~10 s |

**Step 2 against today's blend.** With the live yard, Step 2 also prices the ferrous blend actually charged in the
last 24 h (`charge_data`, `load_charged_blend`) at the same production, HMT / Si target and controls
(`run_today`: shares pinned, flux chosen to the same basicity band, slag-quality, charging and stock limits lifted
because that blend did run). The plan opens with one line an operator can act on: what to charge, the cost, and how
much less (or more) it costs than today's blend in Rs/tHM and Rs lakh/day. The KPIs, blend, flux, cost and slag
tables show *Today* next to *Plan*. On the Neon export of 19 Oct 2026 (shipped calibration): today's blend (sinter
60.8 / Lloyds CLO 14.1 / Geomin 9.7 / NMDC ROM 5.3 / pellet 10.1 %) costs Rs 30,357/tHM at 2,270 t/d; the plan
(sinter 58 / Lloyds CLO 19 / Geomin 13 / pellet 10 %, limestone 8 kg) Rs 30,144, Rs 214/tHM (Rs 4.9 lakh/day) less
for 1.6 kg/tHM more coke. Charged materials without lab chemistry in the window are listed and left out.

**Layout (as page 9).** Each plan opens with its title and a feasibility badge, an overview strip (ore + flux + fuel,
production, burden Fe, saving against today's blend in Rs lakh/day or the change of production against today), then
four bordered panels in process order: 💰 Cost (ore, flux, fuel, total), 🔥 Fuel rates (coke dry, nut coke, PCI, total),
🌋 Slag (rate, t/d, Al₂O₃, MgO; CaO/SiO₂, T-basicity and IB4 against the plant's own p5–p95 bands, MgO/Al₂O₃) and
🚚 Charging (coke, nut coke and PCI t/d, charges/h, hot metal per charge, flux rate). With today's blend the panels
show the change from it. The 🧱 Blend tab is the comparison table beside a donut of the ferrous charge (Plan / Today,
one colour per material). The plant bands are shared with page 9 (`utils/slag_bands.py`).

**Furnace commentary.** Below the plan, *Generate commentary* sends the model the optimiser notes
(`src/assets/burden_engine/optimiser_notes.md`: how every number is made, validated and where it is weak), the fact
register as a table, and the run (set-up, model check, burnt against charged coke, the plan in the panel grouping,
today's blend and the ore + flux answer side by side, binding limits and shadow prices, value of limits, frontier,
yard). Context assembly is pure (`utils/burden_engine/commentary.py`, no LLM inside, unit-tested); the call uses the
app's LLM client (`ui/burden_optimiser/commentary.py`). The full context is shown and can be downloaded.

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
* **Fact register** (`src/config/facts_register.json`, v1.3): every coefficient outside the balance, the plausibility
  bands and the default limits. Retired facts (COKE-17 webapp hand coefficients, DATA-10 regressed RAFT) are
  refused in code.
* **Bands.** Fuel rate (coke wet + nut + PCI) above 600 kg/tHM is a limit row; below 520 the result is marked
  infeasible and not reported (a row there would make the optimiser burn coke). HMT outside 1,440–1,520 °C and Si
  outside 0.20–0.80 % are refused.
* **RAFT** is the plant workbook (`utils.energy_balance.raft`) at the input blast moisture (not measured at BF-02;
  default = the month's typical value from the Coke Energy table).

## STRATA-ω shaft efficiency (#262, closes the certainty menu of #259) — STRATA 1.3

**STRATA** (Stratified Transport & Reduction Analytics) is the name of the BF-02 gas-flow / shaft-efficiency model,
formerly OMEGA. Modules: **STRATA-Flow** (layered packed-bed gas-flow model of BF-02, ring description of the thermal
reserve zone; vendored in `src/utils/strata_flow/`), **STRATA-ω** (blend- and operation-dependent shaft efficiency,
this section) and **STRATA-Learn** (continuous learning, shadow mode). The page switch offers **Fixed-ω (Rist
baseline)** — omega held at the calibration, ASM-01 — and **STRATA-ω** (ASM-18, page default; `Settings.omega_mode=
"blend"` in code).

```
 omega(plan) = omega_cal                                   14-day calibration (the anchor, ASM-01)
             + [w(blend, op_cal) - w(cal)]                 network blend term (GAS-13)
             + sum_c (lam_g - 1) g_c (s_c - s_c,cal)       plant-data scale per group: pellet 1.38, lump ore 0.96 (GAS-14)
             + lam_prod [w(blend, op_cal, P) - w(blend, op_cal)]      production / TRZ residence, 0.59 +/- 0.30 (GAS-16)
             + lam_rest [w(blend, op) - w(blend, op_cal, P)]          coke layers, fuels, O2, 0.98 +/- 0.30
             + 0.0016 (moisture - moisture_cal)            blast moisture, RH estimate, +/- 0.0024 (GAS-19, DATA-11)
             + 0.00094 (nut - nut_cal)                     nut coke in the ore layer, +/- 0.00035 (GAS-23)
```

### No surrogate: the network is solved in the optimiser

* **w** is STRATA-Flow reduced to its thermal reserve zone: six rings, each a counter-current exchanger between the
  gas (capacity R·f_i) and the ore's wüstite oxygen (o_i); NTU = κ·k·t_TRZ·C_s/C_min with k the Fe-weighted reduction
  rate and t_TRZ the residence time. f, o, t come from **network solves** (`utils/burden_engine/strata_network.py`,
  about 1.1 s each): one at the calibration plus 16 for the network Jacobian (cached in `src/storage/strata_flow/`
  per calibration and catalogue; about 20 s the first time), **one per SLP iteration** at the iterate's blend, and
  **three per reported plan** (blend; blend + production; full operation at the plan's coke rate). Between solves —
  the finite differences of one SLP iteration and the coke fixed point of the balance — the network Jacobian moves
  the ring quantities from the nearest solve. At convergence the SLP solves the plan's three states and takes one more
  LP step on them, so the reported plan meets its limits on the network itself. Every result reports the solves made
  and its distance from the nearest solve (0 = solved at the plan). A fixed-production plan takes 10–15 s and about 8
  solves. The production searches (Step 1, maximum production, frontier) reuse a solve within 1 scaled unit (1 pt of a
  share, 5 kg/tHM coke, 1 t/h) and re-solve the plan they return exactly (`Engine.network_tolerance`,
  `Engine.refine`): Step 1 with the 9-level oxygen search takes about 8 min (≈ 170 solves), maximum production
  about 2 min (≈ 40 solves), Step 2 (with and without fuel in the objective) about 30 s.
* STRATA 1.2 used a quadratic surrogate of the network (520-state design); it stays in the asset for diagnostics only
  (`OmegaModel.load(backend="surrogate")`). The network extrapolates by its physics where the surrogate's design
  space ended.
* κ is re-fitted at start-up so that w(calibration state) equals the calibrated omega. The network context (charging
  matrix, top-gas η_CO, H₂, temperature, HBT) is the calibration window's: the asset's for the shipped calibration;
  a live calibration passes its own top gas and HBT (`cal["strata_flow"]`).

### Reducibility and RDI from the laboratory (GAS-21, DATA-13)

* RI (IS 8167:1989) and RDI (ISO 4696-1) from the Evonith laboratory report (48 tests, 2015–2026), **values as given**:
  the test of the same source nearest the model period; the tested sample's Fe, SiO₂ and Al₂O₃ are within normal
  supply variation of the period ore, and the composition adjustment by the report's own slopes is below 0.4 pt RI
  and 0.9 pt RDI. Sinter is not from the report: its RI and RDI are the latest raw-material strength test (calibration
  `sinter_strength`; shipped calibration 4 Oct 2026: RI 66.0, RDI 30.06). RI: pellet 82.4, Lloyds CLO 84.5, Lloyds fines oversize 83.9, Geomin
  79.2, NMDC ROM 60.6, Donimalai 78.4, Gomti 65 (no test). RDI: pellet 14.5, Lloyds CLO 22.1, oversize 22.6, Geomin 6.8,
  NMDC ROM 27.8, Donimalai 18.8 %.
* The ore and pellet RDIs act in STRATA-Flow as sinter RDI already did: a share of the RDI appears as fines in the ore
  layer of the shaft (bed resistance, gas distribution). Out of sample the change is neutral (coke skill 0.128 with,
  0.129 without).
* **Link from RI to rate, calibrated on the plant data:** k_m = [ln(1−RI_m)/ln(1−RI_sinter)]^β · (11.25/d_m)^n. With the
  first-order link (β = 1, n = 1.5) the laboratory spread (NMDC ROM 61 to Lloyds CLO 84) gives out-of-sample coke
  skill 0.037; the skill rises as β falls and is highest at **β = 0, n = 1.0** (0.128): the measured RI differences
  show no effect on the plant's omega, the particle size does (`s13/ri_link_scan.py`; user decision 23 Oct 2026).
  Likely causes: the 180-min test at 900 °C on 10–12.5 mm pieces saturates for these ores while 15–25 mm lumps in
  the shaft are limited by gas diffusion; supply variation (Lloyds 51–84 % since 2022); the sinter routine RI may be on
  another basis (a sinter test in the same laboratory settles it). The critical assumptions of every plan show
  β 0 → 0.25 and n 1.0 → 1.5.

### Two oversize materials (GAS-24)

* **ore_6 LLOYDS FINES OVER SIZE** — Lloyds CLO chemistry, 8–16 mm (−6 mm 3.9 % against 1.0 % for Lloyds CLO). Its
  reduction rate is taken at the Lloyds CLO size (no credit for its smaller size); with the credit the model rated it
  better than sinter, against the plant data (out-of-sample coke skill 0.116 with the credit, 0.128 without; split
  Dec–May → Jun–Oct 0.024 / 0.061). STRATA-Flow packing keeps its sieve size.
* **ore_11 DONIMALAI OVER SIZE RTRN** — Donimalai chemistry (Fe 59.3, SiO₂ 10.5 %); now modelled as Donimalai (STRATA
  1.2 counted it as Lloyds oversize).

### Data scales, moisture, PCI and nut coke

* **What the scales are.** λ multiplies the network's own Δω for one channel: Δω_channel = λ × Δω_physics. λ = 1 takes
  the physics as it is; λ > 1 means the plant responds more than the physics, λ < 1 less. Channels: pellet (the blend
  term of pellet replacing sinter), lump ore (Lloyds CLO, Geomin, NMDC ROM, Donimalai, Gomti, oversize), production
  (TRZ residence at the plan's production) and other operation (coke layers, fuels, oxygen, blast). AR(1) Bayesian
  posterior of the daily change of omega against its 14-day anchor on exact network channels (191 days, 15 Dec 2025 –
  7 Oct 2026), priors N(1, 0.5²): pellet 1.38 ± 0.45, lump 0.96 ± 0.42, production 0.59 ± 0.30, other 0.98 ± 0.30.
  Stored as (x_blend = 1, b_c = (λ_g − 1) g_c) so `PlanOmega` reads them as in v1.0.
* **Blast moisture** (DATA-11): the smoothed weather-record RH mapped to 7–21 g/Nm³ — no step changes. The data term
  is small and not resolved (0.0016 ± 0.0024 per g/Nm³; STRATA 1.2 had 0.0044 on the stepped series, whose 1 June
  step coincided with the pellet start). Moisture acts on coke through the balance (+0.8 kg/tHM per g/Nm³).
* **PCI and nut coke** vary shift to shift. Both enter (1) the two-zone balance of every shift and plan (carbon,
  energy, H₂), (2) the network's other-operation channel (nut coke −0.0002 ω per kg as coke-layer mass, PCI about 0)
  and, for nut coke, (3) a data term +0.00094 ± 0.00035 ω per kg (GAS-23: nut coke mixed in the ore layer raises the
  shaft efficiency; out-of-sample coke skill 0.128 with it, 0.115 without). A PCI data term lowers the skill (0.104)
  and is not used. The page defaults PCI and nut coke to the regime window's medians (ASM-08).
* **Uncertainty and certainty** (`omega_model.OmegaModel`): σ(omega) from the posterior of the blend scales, the two
  operating scales (correlated), the moisture and nut-coke coefficients, converted with the loop-inclusive dK/dω.
  Tiers (GAS-18): **High** σ ≤ 2 kg/tHM and inside the 1st–99th percentile of what BF-02 ran; **Medium** σ ≤ 5 and
  within 5 pt / 3 t/h; **Low** otherwise; **Investigate** pellet > 25 % of ferrous or σ > 8 kg/tHM.
* **Optimiser options.** Certainty filter (`Settings.certainty` = low (all but Investigate, page default) / medium /
  high); risk aversion (`Settings.risk_k`); auto mode (candidates A, B with `certainty="low"`, C High only, T today;
  the rule never picks an Investigate plan; a language model reviews the pick).

### Prices and slag credit (DATA-14, ASM-20)

* Yard prices (plant, 23 Oct 2026), Rs/t: NMDC ROM 10,323, Lloyds CLO 8,877, NMDC Donimalai 6,141, Geomin 4,204,
  sinter 8,784, Lloyds pellet 10,515; limestone 2,369, dolomite 1,196, quartzite 1,792. Materials without a given
  price keep the operator's saved value. Fuels from the BF-2 October-26 cost sheet (FTD Costing 22 Oct 2026): coke
  31,960, nut coke 26,292, PCI 21,168. Live yard: `bmo_operator_inputs.yml`; reference yard: `burden_optimiser.yml →
  prices`.
* Slag credit: objective = ore + flux + coke − slag sold × slag price; the reported total is net of the credit
  (`cost["gross"]` before it). Granulated slag sale price 457.85 Rs/t (plant, 23 Oct 2026; config
  `prices.slag_price_rs_per_t`; the operator's saved value takes precedence). Between two blends of nearly equal cost
  the one with more saleable slag is preferred; with the slag cap binding (340 kg/tHM in the reference case) the credit
  does not move the plan.

### Out-of-sample test (GAS-20, VAL-01..12)

Day minus its 14-day anchor; channels from network runs of each day; parameters from earlier days only (monthly
refit); 175 days, 1 Jan – 7 Oct 2026; skill = 1 − MSE/MSE(Fixed-ω). On the step-free moisture the Fixed-ω baseline
itself improved (RMSE 8.11 → 6.94 kg/tHM), so the skills are not comparable with the STRATA 1.2 figures.

| Test (burnt coke unless stated) | STRATA 1.2 physics, same days | STRATA 1.3 |
|---|---:|---:|
| Expanding, omega, day | +0.155 | +0.160 |
| Expanding, day (RMSE Fixed-ω 6.94 kg/tHM) | +0.110 | +0.128 (6.48) |
| Expanding, 7-day | +0.196 | +0.219 |
| Train Dec–May, test Jun–Oct, day | +0.015 | +0.061 |
| Train Jun–Oct, test Jan–May, day | +0.113 | +0.134 |

By blend family (VAL facts): Supported — no pellet with oversize; no pellet with ROM; pellet with CLO; pellet with
ROM; all days; ROM ≥ 10 %; pellet ≥ 12 %. Neutral — no pellet, sinter + CLO (bias +0.1 kg/tHM). Modelled only
(< 10 days) — Gomti (bias +2.8 ± 3.6), ROM + oversize, pellet + oversize (+4.7 ± 2.0, June 2026, the first month of
pellet), pellet + ROM + oversize (+1.9 ± 2.7). No family is Disputed.

* **Fact register groups MOD and VAL** (`utils/burden_engine/model_facts.py`): MOD-01..11 hold the coke response of
  Fixed-ω and STRATA-ω side by side at the reference point (substitutions of 10 wet pt, production +1 t/h, moisture
  +1 g/Nm³, nut coke +10 kg/tHM); VAL-01..12 the per-family out-of-sample result. `python -m
  utils.burden_engine.model_facts --write` regenerates them after a recalibration.
* **Reference results at 2,270 t/d** (16 Oct yard, shipped calibration, blend re-optimised at each pinned pellet):

REFTABLE

## STRATA-Learn — continuous learning, shadow mode (v1)

`src/utils/burden_engine/learn/` (pure, no Streamlit). The learner never changes the asset, the register or the page;
it writes proposals. The 14-day calibration anchor stays: the learner works on day-minus-anchor changes only.

1. **Evidence store** (`evidence.py`): one row per day — the six channels (pellet, lump ore, production, other
   operation from STRATA-Flow day runs; blast moisture and nut coke against the window), the observed change of omega,
   burnt coke, the Fixed-ω prediction, dK/dω, class shares, blend family and a trial flag. Seed:
   `assets/burden_engine/strata_evidence_seed.csv` (191 days, STRATA 1.3 channels).
2. **Random-walk Kalman filter** (`kalman.py`) on (λ_pellet, λ_lump, λ_production, λ_other, a_moisture, a_nut) with
   AR(1) day errors (Prais–Winsten form, ρ 0.71, σ_e 0.0179); random-walk sd 0.01 per day on the scales. Each day is
   predicted with the previous state before it updates the state, so the prediction history is out of sample.
   Replay of the record from the priors (STRATA 1.3 evidence): burnt-coke skill +0.12, omega +0.16 since 1 Jan 2026
   (expanding-window test +0.13 / +0.16); final state within 0.02 of the deployed posterior.
3. **Gate** (`gate.py`): champion = deployed asset, challenger = Kalman state; on the last 60 days after deployment,
   promotion is proposed when the challenger's MSE is ≥ 5 % lower and the Diebold–Mariano t ≥ 2 and no drift alarm
   is open. **Drift**: two-sided CUSUM (k 0.5, h 5) on the AR(1) innovations of the champion's errors.
4. **Register proposals** (`autofill.py`): automatic — GAS-14 / GAS-19 / GAS-23 values and se within 2 registered se, VAL
   counts and means; approval — any status change (e.g. Modelled only → Supported), a value outside 2 se, growth of
   the GAS-18 support envelope (> 0.5 pt), promotion, drift. `apply()` writes approved items with a revision entry.
5. **Constrained exploration** (`explore.py`): Thompson-sampling probability that each candidate plan is optimal and
   its expected regret; a plan is an exploration candidate only outside the Investigate tier and within the regret
   budget (default Rs 15/tHM). Reported, not applied.

Nightly run: `python -m utils.burden_engine.learn.run --evidence <store.csv> --add <new_days.csv> --out <dir>`
→ `strata_learn_report.md`, `strata_learn_state.json`, `strata_learn_history.csv`, `register_proposal.json`.

## Files

| Path | Role |
|---|---|
| `src/custom_pages/11_Burden_Fuel_Optimiser.py` | Page (UI orchestration only) |
| `src/ui/burden_optimiser/` | `sections.py` (renderers), `loaders.py` (cached yard / calibration / today's blend / runs), `figures.py` (frontier, model check) |
| `src/utils/burden_engine/model_check.py` | Model check: scored shifts, window, accuracy and verdict |
| `src/utils/coke_energy/transit.py` | Burnt against charged coke: concordance, window agreement, spread split, lag scan, closure |
| `src/utils/burden_engine/commentary.py`, `src/ui/burden_optimiser/commentary.py` | Commentary context (pure) and panel |
| `src/assets/burden_engine/optimiser_notes.md` | Optimiser notes: the commentary's static context |
| `src/utils/slag_bands.py` | Plant p5–p95 slag bands, shared with page 9 |
| `src/utils/burden_engine/engine.py` | `Engine`, `Settings`, `Limits`, `value_in_use`, `summary` |
| `src/utils/burden_engine/omega_model.py` + `src/assets/burden_engine/omega_model.json` | STRATA-ω shaft efficiency: ring model, posterior, σ, certainty tiers, critical assumptions |
| `src/utils/burden_engine/strata_network.py` | Network solves inside the optimiser: anchors, Jacobian (disk cache), distance reporting |
| `src/utils/strata_flow/` | STRATA-Flow network (vendored bf2_momentum v0.4.1, laboratory RI / RDI catalogue) |
| `src/utils/burden_engine/learn/` | STRATA-Learn shadow mode: evidence store, Kalman, gate, drift, register proposals, exploration |
| `src/utils/burden_engine/model_facts.py` | Fact-register groups MOD (Fixed-ω vs STRATA-ω responses) and VAL (per-blend out-of-sample) |
| `src/utils/burden_engine/auto_review.py`, `src/ui/burden_optimiser/omega_panel.py` | Auto mode rule and review checks (pure); omega settings, panel and review UI |
| `src/utils/burden_engine/hydraulics.py` | Packed-bed limit, enrichment tag, per-tonne RAFT |
| `src/utils/burden_engine/facts.py` | Fact register loader (guarded access) |
| `src/utils/burden_engine/calibration.py` | Two-zone calibration and regime from pipeline frames |
| `src/utils/burden_engine/scenarios.py` | Mode runs (incl. `run_today`), result and comparison tables, operator labels of the limit rows |
| `src/data/burden_engine/inputs.py` | Yard / prices / limits from the BMO provider and `bmo_operator_inputs.yml`; today's charged blend |
| `src/data/burden_engine/calibration.py` | Live calibration over the window, with fallback |
| `src/config/burden_optimiser.yml` | Page config (yard window, charging, thermal basis, frontier grid, calibration window, model check) |
| `src/config/facts_register.json` + `schemas/facts_register.schema.json` | Fact register v1.6 |
| `src/assets/burden_engine/` | Reference yard (16 Oct 2026 UAT) and Jul–Oct 2026 calibration |
| `scripts/compare_bmo_vs_burden_engine.py` | Page-9 LP against the engine (ore + flux objective) |
| `scripts/replay_slag_vs_lab.py` | Both slag calculations on each day's actual charge against the lab's slag analysis |
| `scripts/burnt_vs_charged_coke.py` | Burnt against charged coke from a shift table |
| `tests/test_burden_engine*.py`, `tests/test_burden_optimiser_*.py` | Engine, inputs, calibration, scenario, model-check and today's-blend tests |

Also changed on page 9 for this work (all checked against the lab, below):
* `OreChemistry.loi_pct` (default 0; read from the same lab row, so page 9 picks the same rows).
* Titani ferrous CLO mapped to `ore_13` (was `ore_11`, the Donimalai oversize return). `charge_data` has no
  `ore_13_mt` column, so the provider now skips charge columns the table does not have instead of failing the query.
* The slag total includes TiO₂ (`SlagBalanceSettings.tio2_to_slag`, `setting_bmo.yml`).
* **Top-gas dust from the DPR.** The single "BF Gas Dust" row from the plant BURDEN workbook (530 kg per charge,
  CaO 5.63 %, MgO 3.57 %) is replaced by the two streams the DPR records, dust catcher and GCP dust
  (`data/bmo/dust_inputs.py`). Quantity: the DPR's dust over the charges `charge_data` counts on the same days, last
  30 days, as kg per charge (DPR `flue_dust_mt` = dust catcher, `gcp_dust_mt` = GCP). Chemistry is configured in
  `setting_bmo.yml → dust_inputs` (default `data_sources.dust_chemistry_source: configured`): the means of the
  plant's dust analyses in the hourly dataset `furnace_dataset.csv` (`DUST_1_*` dust catcher, `DUST_2_*` GCP,
  180 days to Oct 2026). `DUST_1` has only a few samples since mid-June (last 23 Jun), so the page does not read
  the dataset by default; set `dust_chemistry_source: dataset` to use its last 60 days. Quantities and chemistry are
  editable under *Advanced Slag Balance Inputs*; Save pins the operator's values, *Use DPR values* drops them.

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
- [x] Live yard (Neon export of 19 Oct 2026, read through the app's own loaders): chemistry date and stock shown per
      material; fluxes are page 9's catalogue with saved prices (fixed: they came from the last charge, keyed
      flux_1/2, unpriced, limestone missing). Step 2 at 2,270 t/d: Rs 30,138/tHM, coke 304.7 kg/tHM dry. Step 1:
      cheapest 2,303 t/d at Rs 29,930/tHM, highest 2,407 t/d (shipped calibration; InfluxDB not reachable)
- [x] Slag against 284 days of lab analyses (`scripts/replay_slag_vs_lab.py`, below)
- [x] Model check (shift results of the Coke Energy validation workbook to 8 Oct; InfluxDB not reachable here):
      energy model bias −0.4 kg/tHM, typical error ±3.9 over 40 stable shifts of the last 14 days → *On track*;
      oxygen balance +2.8 / ±4.2
- [x] Step 2 against today's charged blend (Neon export, 19 Oct): Rs 214/tHM less, coke +1.6 kg/tHM
- [ ] On UAT with live data: calibration window = last 14 days; the model check reads the live pipeline (latest shift
      within a day); Step 2 at today's production within a few kg/tHM of the Coke Energy page's energy-model coke
      for the current blend
- [ ] Operator walk-through (below) signed off

## Comparison with the Blend Mix Optimiser LP

`python scripts/compare_bmo_vs_burden_engine.py [--live] [--target 2270]` runs both on the same yard, prices, bounds,
slag limits, page-9 dust rows and page-9 fuel S / alkali, with fuel left out of the objective
(`docs/burden_fuel_optimiser_vs_bmo.md`).

Page 9's slag total counts TiO₂, alkalis (80 % to slag), sulphur (ore, flux and fuel), MnO, FeO and CaF₂ from the
analyses; phosphorus goes to the hot metal. Its dust is the DPR's two streams with the measured dust chemistry.

At 2,270 t/d the two blends agree within 0.3 pt:

| | Reference yard (16 Oct) | Live yard (Neon, 19 Oct) |
|---|---|---|
| Engine | sinter 58.0 / Geomin 13.4 / Lloyds CLO 18.6 / pellet 10.0 %, limestone 13.7 kg | 58.0 / 13.0 / 19.0 / 10.0 %, limestone 7.6 kg |
| Page-9 LP | 58.0 / 13.3 / 18.7 / 10.0 %, limestone 13.5 kg | 58.0 / 12.7 / 19.3 / 10.0 %, limestone 7.0 kg |
| Ore cost, page 9 − engine | +Rs 46/tHM | +Rs 54/tHM |

The cost gap is page 9 charging 4 kg/tHM more ferrous (its Fe balance sizes the dust Fe and pig iron differently). On
the reference yard page 9 still sits on MgO/Al₂O₃ ≥ 0.345 (engine 0.348): its dust takes out about 0.2 kg/tHM MgO that
the engine's MgO-free dust leaves in. Before the dust and TiO₂ changes, page 9 moved about 4 pt from Lloyds CLO to
sinter, Geomin and NMDC ROM and bought 3.7 kg/tHM dolomite.

**One blend through both slag calculations** (engine's ore + flux blend on the reference yard, kg/tHM):

| | Engine | Page 9 now | Page 9 before (workbook dust, no TiO₂) |
|---|---:|---:|---:|
| SiO₂ / Al₂O₃ / CaO / MgO | 116.9 / 67.1 / 122.7 / 23.4 | 117.0 / 67.2 / 122.7 / 23.2 | 117.8 / 67.0 / 121.4 / 22.1 |
| TiO₂ | 4.2 | 4.1 | 0 |
| MnO + FeO | 1.7 | 1.7 | 1.8 |
| S + K₂O + Na₂O | 4.1 (flat 1.2 % of slag) | 3.6 (from analyses) | 3.8 |
| **Slag total** | **340.0** | **339.5** | **333.7** |
| **Al₂O₃ %** | **19.74** | **19.79** | **20.08** |
| MgO % | 6.87 | 6.82 | 6.61 |
| CaO/SiO₂ | 1.050 | 1.049 | 1.030 |

## Slag against the lab: daily replay

`python scripts/replay_slag_vs_lab.py` runs both slag calculations on each day's actual charge (Neon `charge_data`,
lab chemistry per material, DPR hot metal / PCI / dust, the day's hot-metal analysis) and compares them with the day's
mean slag analysis (`docs/slag_replay_vs_lab.md`). 284 days, Sep 2025 – Oct 2026, model minus lab:

| | Engine | Page 9 now | Page 9, workbook dust row | Page 9 before (workbook dust, no TiO₂) |
|---|---:|---:|---:|---:|
| Al₂O₃, % pt | +0.02 | +0.01 | +0.04 | +0.24 |
| MgO, % pt | +0.00 | −0.06 | −0.36 | −0.28 |
| TiO₂, % pt | −0.10 | −0.13 | −0.09 | **−1.13** |
| CaO/SiO₂ | −0.015 | −0.017 | −0.037 | −0.037 |
| Slag, kg/tHM | 338 | 338 | 336 | 333 |

* **TiO₂ belongs in the slag total.** The lab's slag carries 1.13 % TiO₂ on average; leaving it out under-reads it every
  month (−0.8 to −1.4 pt) and puts Al₂O₃ 0.24 pt high.
* **The workbook dust row removed too much CaO and MgO** (MgO −0.36 pt, CaO/SiO₂ −0.037, the same offset every month).
  The DPR quantities with the measured dust chemistry (CaO ~2 %, MgO 0.5–0.8 %) remove it.
* **Page 9 and the engine now match the lab and each other**, with no material bias (MAE: Al₂O₃ 0.54 pt, MgO 0.35 pt,
  CaO/SiO₂ 0.041), and their slag rate equals the Al₂O₃-tracer estimate (338 kg/tHM).

## Operator walk-through

1. **Status line**: yard source (live / reference), calibration window and shaft efficiency, register version.
2. **Model check**: compare the latest shift's burnt coke with the energy-balance figure (same HMT and Si). *On track*
   over the period means the plan's coke figures can be trusted; *Check before trusting the plan* names the bias or
   scatter that is out. Switch the period to 7 or 30 days; click *Charged coke* in the legend to add it. The second
   tab shows burnt and charged coke converging as the window grows: compare them over a day or more.
3. **Step 1**: open *Day limits* and set today's caps (e.g. blast 105,000 Nm³/h on a blower-limited day). Optimise.
   Read the cheapest output, the highest output, the cost of each extra tonne between them, and what binds.
   Switch *Plan shown below* to see either plan's blend and controls.
4. **Step 2**: enter the target t/d (default: today's production). Optimise. Read the plan line (charge, cost, saving
   against today's blend), the *Today* columns, then *Value of limits* for the cheapest limit to relax.
5. **Commentary**: press *Generate commentary* for the model's reading of the run; open *What the model is given*
   to audit it, or download the context.
6. **Warnings** mean a rule is extrapolated (pellet > 16.6 %, Gomti > 11.6 %) or RAFT is outside its band; an error
   names the limit that blocks the plan.

## What the engine does not see

* STRATA-ω holds the charging matrix at the one in force over the calibration window, and
  coke quality, nut coke, PCI and hot-blast temperature act on omega only through the calibration. The day-to-day
  spread of omega (out-of-sample RMSE 0.028, about ±8 kg/tHM) is not in σ; it averages out over days.
* Blast moisture is the Wardha weather record's daily value (DATA-11, `config/coke_energy.yml → blast_moisture`);
  top-gas H2 does not follow the record's day-to-day moisture (GAS-19), so the moisture term is a data term.
* Process limits default to the Jun–Oct 2026 operating envelope, not equipment ratings. Oxygen and stove gas are
  unpriced.
* Not limited: phosphorus, alkalis, zinc, sulphur; the sinter plant's cost of a basicity change.
* Steady day average: no transition between blends; dust and heat losses (GJ/h) stay at the calibration.
