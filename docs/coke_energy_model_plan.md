# Coke Energy Model — plan and status

BF-02 mass and energy balance, brought into the app as its own page: **Coke Energy Model**
(`src/custom_pages/10_Coke_Energy.py`). The model itself was built in the Claude web app (14–15 Oct 2026) and is kept
almost unchanged in `src/utils/coke_energy/`. Delivery is step by step; each step is approved before the next starts.

## Decisions

**15 Oct 2026, first round**

1. **Keep the web model almost unchanged.** Integrate more smoothly later. `src/utils/coke_energy/__init__.py` lists
   every change (import paths, `__main__` blocks moved into `pipeline.py`, a global warnings filter removed). A golden
   test pins the app copy to the original code.
2. **Separate page for now.** BMO is not changed until the optimiser step, and then only behind a new option.
3. **The Intensive Optimizer may take up to about 15 minutes** once the energy model prices every candidate. Measured:
   a two-zone solve takes 2.5–5 ms, so a DE run of a few thousand candidates adds 10–30 s; there is ample margin.
4. **Two models, two roles.**
   - Oxygen-balance model (ηCO fixed at the analyser's value): for checking shifts.
   - Energy-balance model (two-zone Rist, shaft efficiency ω fixed): for what-ifs, because there ηCO and the top-gas
     temperature are outputs.

**15 Oct 2026, second round (plant answers)**

5. **Stave heat-load tags are GJ/h.** The web model's reading stands. Each row tag is the exact sum of its four
   quadrant tags, so those are GJ/h too; see open item 4 for BMO.
6. **There is no coke CSI property.** Coke strength is M-40, M-10, CRI and CSR.
7. **Oversize CLO (`ore_6`, Lloyds fines oversize) is 8–16 mm**, as the size log shows.
8. **Pellet size is fixed** (5 mm screen). It is not a variable for the ω model and is not checked for freshness.
9. **Slag temperature rule.** Slag runs about 30 °C above hot metal for a low-Al₂O₃ slag (~16 %), and hotter as Al₂O₃
   rises, up to about 60 °C.
   - Implemented as linear from 16 % → 30 °C to 21 % → 60 °C, flat outside. The 21 % point is this page's choice: the
     95th percentile of tap analyses since Aug 2025 is 20.6 %.
   - Each shift's balance (calibration and predictions) uses its measured slag Al₂O₃; the blend what-if will use the
     blend's slag Al₂O₃.
   - Effect on shift accuracy is negligible (oxygen-balance model vs burnt SD 4.80 → 4.80; energy model 7.81 → 7.82),
     because the 14-day calibration absorbs the average. It matters for blends, where Al₂O₃ moves.
   - Config: `slag_superheat` in `coke_energy.yml` (`null` restores the web's constant 40 °C).
10. **Freshness notices.** When a value rests on a data point older than its limit, the page says so and names the last
    valid data point. Limits:
    - plant: HM/slag analysis 3 h; raw-material chemistry, size analysis, strength and DPR 1 day; fuel ash 15 days;
      burden distribution 1 month;
    - this page's choice: 10-min data 30 min, hourly dataset 3 h, charge reports 2 h, dust Fe/carbon 1 day, dust oxides
      15 days.

    Ages are measured to the end of the shift shown, or to now for the status line.
11. **Presentation (after the first review).**
    - Charged coke is hidden by default, behind one switch.
    - Page order: tiles, then prediction against burnt coke, then the Rist diagram.
    - The Rist diagram is full width, with the coke strip below it.

## Where the data comes from

- **Offline data comes from the plant database, not Neon.** `DATABASE_URL` (plant PostgreSQL at 125.23.100.62) is
  what `furnace_data` resolves first, so the app reads the plant database.
- **Neon "Evonith DB" project.** It has three branches: `production` (default), `developer` (the `NEON_STR` endpoint)
  and `development`.
- **Comparison on 15 Oct.** The `developer` branch's `offline_feed` has the same lab data: every source's latest valid
  sample matches. It holds slightly fewer charge, tap and DPR rows, `burden_history` ends in Mar 2026 against May on the
  plant database, and `historical_static_ml_dataset` ends in Dec 2025 against Jun 2026. The page stays on the plant
  database.
- **Hourly charge and composition** come from the published `furnace_dataset.csv`, the same file and URL as BMO.
- **10-min data** comes from InfluxDB, cached per month under `src/storage/coke_energy/`.

## The validated basis, reproduced in the app

The app pipeline runs on live sources and reproduces the web model's validation on the same period (stable shifts,
Dec 2025 – Oct 2026, out of sample).

| Check (per shift unless stated) | Web write-up | App, 15 Oct |
|---|---|---|
| ω on stable shifts | 0.80 ± 0.04 | 0.799 ± 0.035 |
| Oxygen-balance model vs burnt coke, SD | 4.8 | 4.8 |
| Oxygen-balance model vs charged coke, SD | 10.8 | 10.9 |
| Charged vs burnt coke, SD (timing) | 10.9 | 11.1 |
| Energy-balance model vs burnt coke, SD | 7.7 | 7.8 |
| Energy-balance model vs charged coke, SD | 11.5 | 11.5 |
| Top-gas temperature, model vs measured (r, SD) | 0.90, 11.7 °C | 0.90, 11.8 °C |
| ηCO, model vs measured (r) | 0.52 | 0.53 |
| Oxygen-balance model vs charged, 2-day windows, SD | 6.3 | 6.0 |
| Oxygen-balance model vs charged, 1-week windows, SD | 4.2 | 4.3 |

These figures use the web's constant slag temperature; with the plant rule they are the same to 0.02 kg/tHM. The raw
web input files are no longer needed for parity.

## Step 1 — delivered (awaiting approval)

**On the page, top to bottom**

1. **Status line.**
   - How far each source reaches.
   - The **Show charged coke** switch (off by default) and Refresh.
   - A warning that lists every model input carried forward beyond its limit as of now, with the last valid data point.
   - An expander with the full freshness table (every source and material in use, last valid data point, age, limit,
     status, which part of the page uses it).
2. **Coke rate being used.** For the latest balanced shift (or a chosen one):
   - three tiles: oxygen-balance model, energy-balance model (with its predicted ηCO and top-gas temperature), and
     burnt coke. Charged coke appears only with the switch on.
   - Each model tile shows its difference from burnt coke.
   - Below the tiles: the shift's own freshness check (what the balance of that shift used beyond its limit, or a tick).
   - An expander lists the shift's inputs and calibration, including slag Al₂O₃ and the slag temperature used.
3. **Prediction against burnt coke.**
   - **Panels:** one per model against burnt coke, sharing the time axis.
   - **Lines:** "24 h" (default) or "48 h" draws a centred production-weighted mean, with the shifts as faint points;
     "Per shift" joins the shifts.
   - **Band:** each panel's band is that model's typical error at the chosen smoothing.
   - **Below the panels:** predicted vs burnt coke per shift (1:1 line, ±10 kg/tHM band) and the model error per shift
     over time.
   - **Table:** bias, r, the shares within ±5 and ±10, and the SD per shift, per day, per 2 days and per week.
4. **Rist diagram.** Full width, animated over the stable shifts of a chosen period (default 14 days), with ▶ Play and a
   slider; the strip below marks the same shift.
   - **Shift line:** drawn from burnt coke (bias-corrected), with the blast that closes the oxygen balance at the
     analyser's ηCO.
   - **Model line:** the energy model's line, with ω fixed.
   - **Labels:** E, W and P only; the other points explain themselves on hover. The slider label carries the shift's ω
     and the three coke rates.
   - **Zoom:** "Whole furnace" and "Reserve zone" buttons; the zoomed view shows the E–W gap and the two lines apart.

**Code**

| Path | Role |
|---|---|
| `src/utils/coke_energy/` | The web model (bf_thermo, bf_balance, rist, run_shifts, whatif, burnt, twozone_shifts, build_shifts, rolling_predict), plus `pipeline.py` (the scripts' `__main__` blocks in memory, per-shift memo, slag-temperature rule), `rist_diagram.py` (diagram geometry), `accuracy.py` (windows, smoothing, statistics), `freshness.py` (limits and the check) |
| `src/data/coke_energy/` | `sources.py` (hourly dataset, 10-min Influx cache, HM/slag taps, DPR, provisional DPR dust); `freshness.py` (one read-only query for every lab table's valid sample times, and hourly charged tonnes per material) |
| `src/ui/coke_energy/` | `loaders.py` (cached loading), `figures.py` (Plotly), `sections.py` (renderers) |
| `src/config/coke_energy.yml` | Data sources, Influx field contract, rolling windows, slag-temperature rule, freshness limits |
| `tests/test_coke_energy_pipeline.py`, `tests/test_coke_energy_page_parts.py` | Golden parity with the original web code, pipeline vs web scripts, Rist geometry, windows, smoothing, freshness, slag rule, figures |

**Behaviour to know**

- **Predictions for every balanced shift.** The web scripts predicted stable shifts only. The page predicts every
  balanced shift from the same calibration pool (stable shifts on earlier dates). Results on stable shifts equal the
  web's exactly; a test holds this.
- **Why the Rist line is drawn on burnt coke.** Charged coke reaches the tuyeres 4–5 h after weighing. On the
  charged-coke basis, the per-shift lower-zone heat loss jumped by 18.3 GJ/h on average from one shift to the next, and
  ω by 0.018. On burnt coke those jumps are 6.3 GJ/h and 0.012, with the same mean ω (0.802 against 0.801). The energy
  model itself still calibrates on charged coke, as the web did (open item 6).
- **DPR dust carried forward.** Up to 3 missing DPR days take the median dust per tonne of the previous 7 DPR days. The
  freshness check names the last real DPR once that is more than a day old.
- **Lag of the hourly dataset.** The published dataset trails the plant by about 20 h (on 15 Oct it ended at
  14 Oct 21:00), so the latest balanced shift is usually yesterday's.
- **Speed.**
  - First 10-min fetch: about 1 minute, once.
  - Full balance of ~830 shifts: about 10 s cold and about 1 s warm (per-shift memo).
  - Page data cache: 30 minutes.

## Next steps (proposed; each one waits for approval)

### Step 2 — Blend what-if (use case 1)

The optimal coke rate for a BMO blend, with PCI and nut coke fixed at the BMO override values.

**Inputs**

- The blend comes from the current BMO session result (Balanced, Intensive or Manual), a saved BMO snapshot
  (`src/storage/bmo_snapshots/`), or a manual table.
- Only BMO's calculated values are read: per-ore dry kg/tHM, chemistry, flux additions, the fuel-ash table, dust and
  slag rate.
- The adapter adds what BMO does not carry:
  - ore LOI and sinter FeO, from `ore_chemistry` / `sinter_chemistry`;
  - dust carbon, from `dust_basic_analysis`;
  - HM Mn, P and Ti, from recent casts.
- Slag uses BMO's rate, adjusted inside the solve for coke ash as coke moves. The model's own oxide-balance slag is
  shown as a check.

**Solve.** Two-zone model, with ω held at the recent median. Targets are HM Si and HMT. Slag temperature comes from
the plant rule at the blend's slag Al₂O₃.

**Outputs.** Each output is shown against the current blend run through the same engine:

- coke dry, as charged, and set-point equivalent;
- ηCO and top-gas temperature;
- blast in Nm³/tHM and Nm³/h, and O₂;
- slag;
- fuel cost;
- heat-balance waterfalls for the lower and upper zones.

**ω band.** Show P10, median and P90 of ω next to the answer. Blend chemistry alone moves coke by under 2 kg/tHM, while
0.01 of ω is worth 2.5 kg/tHM.

**Done when** the current blend reproduces current coke within calibration error, and the Jul–Oct vs Mar–May blend
difference matches the web (+1.6 kg/tHM, limestone mode).

### Step 3 — Intensive Optimizer on energy-based fuel cost (use case 4)

- **New fuel basis.** `energy_two_zone` in `utils/bmo/fuel_prediction.evaluate_blend_with_fuel_prediction`. Every DE
  candidate gets a two-zone solve, and fuel cost = coke (as charged) × price + nut coke + PCI at their fixed rates.
- **Correction off.** The physics coke correction is switched off in this basis, because keeping both would double
  count.
- **Calibration** is resolved once per run and frozen.
- **Runs from the new page** on BMO's inputs. BMO's default path is unchanged, and the 270–370 kg/tHM limits stay.
- **Later:** energy-model-derived linear coefficients per ore for the Balanced (LP) optimiser.
- **Done when** a DE run completes end to end within the agreed time (≤ 15 min) and BMO page output is unchanged by
  default.

### Step 4 — Operating envelope (constraints)

The question it answers: for example, the blast cannot exceed 110,000 Nm³/h, and O₂ and PCI have caps today — what is
the best production at the optimal coke rate?

- **Variables:** production, O₂ flow and PCI, optionally HBT. Blend, nut coke, HM targets and ω are fixed. Heat losses
  are held in GJ/h, so losses per tonne fall as production rises.
- **Constraints:**
  - hot blast (air + O₂) ≤ cap;
  - O₂ ≤ cap;
  - PCI ≤ cap;
  - RAFT within a band (reuse `utils/energy_balance/raft.py`);
  - top-gas temperature ≥ a minimum;
  - charging capacity.
- **Objective:** maximum production, or minimum fuel cost at a production target.
- **Output:** the result, its limiting constraint, and a frontier chart.
- **Caveat:** the balance knows nothing about ΔP, permeability or hanging. Until the ΔP model of step 6 exists, an
  editable bosh-gas ceiling (best sustained history) stands in as the gas-flow limit.

### Step 5 — Thermal state: expected Si / HMT (use case 3)

- **Physics input:** coke reaching the tuyeres now (charged 4–5 h earlier) minus coke needed now at the Si and HMT
  targets.
- **Fitted mapping:** history shows +0.06 % Si and +4 °C HMT per 10 kg/tHM surplus over 1–3 shifts. That is about a
  sixth of what all-to-hot-metal physics says (the coke stock absorbs the rest), so the surplus → Si / HMT mapping is
  fitted with time-ordered folds.
- **Gate:** shown as a forecast only if it beats "same as last cast" and the existing Si forecast on unseen days 3–8 h
  ahead. Otherwise it stays an advisory thermal-surplus index, offered as a feature to the Si model.

### Step 6 — ω and ΔP / permeability data models

- **ω** per shift is already computed (two-zone calibration).
- **Model:** relate ω to burden quality, sizing and distribution (data below). ω varies by about 0.03 shift to shift,
  and 0.01 of ω ≈ 2.5 kg/tHM coke, so a useful model must explain a real part of that out of sample.
- **Interface:** the solver takes ω from a provider. Today that is the 14-day median; later it is the data model, and
  the page states which one supplied the value. The ΔP / permeability predictor for the envelope uses the same pattern.

## Data for the ω model

Described by the plant team and checked against the database on 15 Oct 2026.

| What | Where | Coverage | Notes |
|---|---|---|---|
| Size distribution per material | `offline_feed.material_fines_analysis` (`material_code`, `plus_100mm` … `plus_5mm`, `minus_10mm`, `minus_6mm`, `minus_5mm`) | 1 Apr 2026 →; sinter_3 310, ore_2 (Lloyds CLO) 180, ore_3 161, pellet_1 128, coke_1 99, nut_coke_1 97, ore_1 58, ore_6 31, flux 96, others few | **`plus_X` is the share retained between X and the next larger sieve reported, not cumulative.** The plus columns and the lowest minus column add to ~100 %. `minus_Y` is cumulative passing Y. Each material uses its own subset of sieves. |
| Minimum size charged | plant practice | — | Sinter +6, CLO +8, pellet +5, coke +30, nut coke +14 mm. The undersize is screened out before charging (charged weights exclude fines), so the charged distribution is the part above the cut. Pellet size is fixed. |
| Coke strength | `offline_feed.raw_material_strength_analysis`, mapped by `plant_master.material_property_mapping` | coke_1: 452 rows, May 2025 → | property_1 M-40 (~79.5), property_2 M-10 (~6.9), property_3 CRI (~26.6), property_4 CSR (~63.9). M-40/M-10 are drum-test indices, separate from the as-received size distribution (coke +40 mm ≈ 65 % there). No CSI. |
| Sinter strength | same table | sinter_3: 462 rows, May 2025 →; sinter_1: 49 | property_1 AI (~5.1), property_2 TI (~79.7), property_3 RDI (~33.6), property_4 RI (~68.5) |
| Oversize CLO | `plant_master.materials` and the hopper mapping (`ops_config.hopper_raw_material_history`, 19 hoppers, 49 snapshots Jul 2025 → Sep 2026) | — | ore_6 = Lloyds fines oversize (8–16 mm, confirmed), ore_11 = Donimalai oversize; each has its own chemistry in `ore_chemistry`. |
| Coke, nut coke, PCI ash chemistry | `offline_feed.ash_chemical_analysis` | 104 rows, fortnightly, 15 Apr 2025 → 1 Oct 2026 (Aug–Sep loaded on 15 Oct) | Latest: coke 29 Sep, nut coke 1 Oct, PCI 28 Sep |
| Fuel proximate | `offline_feed.fuel_chemistry` | coke_1/2, nut_coke_1, pci_1; daily | TM, moisture, ash, VM, FC |
| Dust, basic | `offline_feed.dust_basic_analysis` | 995 rows, daily (dust_1 dust catcher, dust_2 GCP) | Fe, LOI, ash, VM, FC |
| Dust, chemical | `offline_feed.dust_chemical_analysis` | 77 rows (75 GCP, 2 dust catcher) | Fe₂O₃, SiO₂, Al₂O₃, TiO₂, MnO, CaO, MgO, Na₂O, K₂O, P₂O₅, SO₃, Cr₂O₃, ZnO, PbO, LOI |
| Flux chemistry | `offline_feed.flux_chemistry` | 555 rows (dolomite 130, quartz 383, limestone 42) | CaO, MgO, SiO₂, Al₂O₃, Fe₂O₃, LOI, TM |
| Ore / sinter chemistry | `offline_feed.ore_chemistry`, `offline_feed.sinter_chemistry` | daily | Ore LOI and sinter FeO are needed by the balance (BMO's ore table lacks both) |
| Burden distribution | `ops_config.burden_history` | 27 changes, 28 Jun 2025 → 15 May 2026 | Coke and non-coke rings and angles per position p01–p11, discharge times, charge pattern (PTOC), purpose. Also in the hourly dataset as weighted coke and non-coke angles. |

**Candidate ω features**

- **Charged shares:** sinter, CLO, oversize CLO and pellet shares (charge data and hopper mapping).
- **Size, per material above its cut (not pellets):** mean size (harmonic / Sauter), share in the band just above the
  cut, and undersize % (the screening load).
- **Strength:** coke M-40, M-10, CRI and CSR; sinter AI, TI, RDI and RI; nut coke rate.
- **Burden distribution:** weighted angles, ring counts and charge pattern.
- **Operation:** production, PCI, blast per tonne, permeability and ΔP.
- **Limit:** the size history starts on 1 Apr 2026, so size features restrict the training window to Apr 2026 onwards.

## Open items

1. **Slag-temperature rule, upper point.** Confirm 60 °C at 21 % Al₂O₃ (implemented), or give the Al₂O₃ at which slag
   reaches 60 °C.
2. **Hourly dataset lag (~20 h).** Decide whether to fill the latest day from Neon `charge_data` and the chemistry
   tables so that the page reaches the running shift.
3. **Data-entry lags.** As of 15 Oct evening, the freshness check shows these carried forward beyond their limits:
   - HM/slag analysis (last 06:09, ~11 h);
   - ore, pellet, fuel and flux chemistry (last 14 Oct 07:00, ~1.5 days; quartz and dolomite 12 Oct);
   - **offline coke (`coke_2`) chemistry from 25 Sep while still being charged**;
   - coke and PCI ash (29 and 28 Sep, past 15 days);
   - coke strength (5 Oct);
   - charge reports (~14 h);
   - DPR (to 13 Oct);
   - burden distribution (last change 15 May).
4. **BMO stave-loss conversion — fixed 15 Oct 2026 (approved).**
   - **Fix:** the ×3.6 factor was removed from `utils/bmo/coke_history._shell_loss_daily` and from the BMO page's live
     snapshot; the tags are GJ/h.
   - **Effect:** the stave loss, rows 6–10, now reads a median 4.9 GJ/h, against 17.6 before.
   - **Calibration refit:** BMO's coke-rate offset was refitted, 16.4 → 7.3 kg/tHM (daily residual SD 10.2 → 9.2).
   - **Record:** `docs/energy_balance_findings_and_open_decisions.md` §5.
5. **Blast moisture** is still the monthly climate assumption. The steam tag reads 0 for Aug 2025 – Oct 2026; 5 g/Nm³
   is worth about 4 kg/tHM in the two-zone model.
6. **Energy-model calibration basis.** The web model calibrates ω and the zone heat losses on charged coke. On burnt
   coke the per-shift heat-loss noise falls by two-thirds, and the 14-day medians could be steadier. This is worth
   testing; it changes the web model, so it needs approval.
7. **Step-2 inputs to agree:**
   - plant aims for HM Si and HMT;
   - which ω means "maximum ηCO possible" (median, or P90 of the last 30 days as best demonstrated);
   - basicity for BMO blends (recommended: BMO's flux additions as designed).
