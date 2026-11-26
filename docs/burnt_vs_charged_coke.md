# Burnt coke against charged coke: the same coke, a descent time apart

Question (19 Oct 2026): how do we show that burnt coke and charged coke are the same quantity, only shifted by the
time the coke takes from the stockline to the tuyeres?

Code: `src/utils/coke_energy/transit.py` · page 11, *Model check → Burnt against charged coke* ·
`python scripts/burnt_vs_charged_coke.py SHIFTS.csv` · tests `tests/test_burden_optimiser_transit_and_commentary.py`.
Admin view `/admin/burnt-vs-charged` (SV-4.4): windows from a shift to a month, the variance split, the lag scan, the
raw mass closure per month and a leak-water check (`services/burnt_vs_charged.py`, `utils/coke_energy/leak_check.py`).

## What "the same, delayed" implies

Coke tipped at the stockline at time t reaches the raceway at t + τ, τ being the burden descent time (6–7 h on BF-02:
the lag at which the thermal state answers a fuel change, `docs/bmo_fuel_slag_si_findings.md`). Write
burnt(t) = charged(t − τ) + e(t), e = measurement error. Over any window of length W the two totals then differ by
the change in coke in transit (bounded: at most a descent time's worth of coke) plus the error. Three things follow,
and each gives a number:

| If they are the same coke… | Test | Number |
|---|---|---|
| they agree on the 45° line, with no level or scale gap | Lin's concordance correlation coefficient CCC = 2s_xy / (s_x² + s_y² + (m_x − m_y)²) = r × C_b, with Bland–Altman bias and 95 % limits | CCC → 1, C_b ≈ 1, bias ≈ 0 |
| the gap averages out as the window grows | SD of the window difference against W; fit SD² = A/W² (transit) + B/W (random) + C (floor) | C small; A and B fall away |
| burnt follows charged by the descent time | cross-correlation of the shift-to-shift movements (deviations from a 7-day mean) by lag | peak at τ; burnt after charged, not before |

Pearson's r alone is not enough: two series can correlate perfectly and differ by 10 kg/tHM. CCC is 1 only when the
series are identical; C_b (≤ 1) is the level-and-scale part, r the scatter part.

## Update 25 Nov 2026: dry fuel basis

**Finding.** From 7 Dec 2025 the hourly dataset carried the inherent moisture (IM) of the laboratory sample
(0.30–0.53 %) in `COKE_MOIST%` and `NUTCOKE_MOIST%`, the same values as `COKE_IM%` / `NUTCOKE_IM%`, instead of the total
moisture (TM). The dataset service mapped `fuel_chemistry.moisture` to both columns. The charge record (`charge_data`)
is wet, as charged. The pipeline therefore entered the wet coke and nut coke as dry. The plant's tags `coke_rate`,
`nut_coke_rate` and `coal_rate_actual_value` are dry, and `fuel_rate` is their sum.

**Correction (user decision 25 Nov 2026).**

- Coke and nut coke, dry = charge record × (1 − TM/100). TM comes from `fuel_chemistry.tm`: the latest sample before
  each charge, weighted by the charge over coke_1 / coke_2 and nut_coke_1 / nut_coke_2.
- The pipeline takes a reading as TM only when it is ≥ 1.0 % and differs from the same hour's IM. Otherwise it uses the
  default TM (coke 2.5 %, nut coke 6.5 %).
- PCI is taken as recorded, without the earlier deduction of its IM.
- Fix in `furnace_data/dataset/service.py`. The published dataset needs a rebuild before the corrected TM reaches the
  app.

**Data.** TM samples from the 19 Oct and 12 Nov exports cover 1 Aug 2025 – 12 Nov 2026. Hourly charge-weighted TM was
1.40–9.19 % for coke (monthly 1.87 % in May 2026 to 5.90 % in Sep 2025) and 3.50–13.80 % for nut coke. Two pipeline runs
on identical inputs (earlier basis, dry basis) cover 1 Aug 2025 – 25 Nov 2026 (780 shifts).

| Quantity (stable shifts) | Earlier basis | Dry basis |
|---|---:|---:|
| Charge record over the plant's fuel-rate tag, median per shift (901 shifts) | +2.83 % (robust SD 2.49) | +0.44 % (2.35) |
| Daily medians 1 Sep – 11 Nov: coke charged / tag | 305.3 wet / 297.6 | 297.7 dry / 297.6 |
| Daily medians 1 Sep – 11 Nov: nut coke charged / tag | 75.3 / 70.0 | 71.1 / 70.0 |
| Charged coke, kg/tHM | 310.0 | 304.7 |
| Raw burnt coke (GBC), kg/tHM | 314.0 | 314.3 |
| TRACE offset (charged − raw), kg/tHM | −5.0 | −10.6 |
| Raw mass closure (raw burnt / charged) | 1.013 | 1.031 |
| Blast meter over the reconciled blast, median | 1.7 % | 4.3 % |
| ω (FORGE per shift) | 0.802 | 0.797 |
| Lower / upper-zone loss, GJ/h | 43.2 / 20.7 | 32.9 / 15.3 |
| FORGE − TRACE SD per shift | 5.94 | 6.20 |
| Charged − TRACE SD per shift | 10.71 | 10.60 |

The dry charge record agrees with the plant's dry tags to 0.4 %. The raw gas-side estimate is unchanged by the fuel
basis, since it uses no coke moisture. Raw burnt coke now exceeds the dry charge by 3.1 %, and the blast meter reads 4.3 %
above the blast that closes the carbon and oxygen balance at the dry charge (DATA-01). The TRACE offset absorbs this
gas-side bias. The calibrated zone losses fall because less coke carbon has to be dissipated, and the coke moisture now
evaporates explicitly in the upper zone (about 27 MJ/tHM).

Concordance of charged and burnt coke (CCC; stable shifts / all shifts):

| Window | Stable, earlier | Stable, dry | All, earlier | All, dry |
|---|---:|---:|---:|---:|
| Shift | 0.871 | 0.883 | 0.818 | 0.817 |
| 1 day | 0.908 | 0.913 | 0.908 | 0.894 |
| 1 week | 0.956 | 0.945 | 0.958 | 0.941 |

**Two checks on the slight widening of FORGE − TRACE (+0.26 kg/tHM):**

- The TM sampling is not the cause: replacing the per-sample TM with its 30-day mean gives the same day error.
- The PCI basis is not the cause: with PCI less its IM, the day error is 5.28 against 5.29.

The smaller calibrated loss weakens the model's production response. The slope of the error on production rises from
0.58 to 0.61 kg/tHM per t/h.

**The stable band of the blast meter (−9 to +15 %) is kept.** On the dry basis 61 shifts fall outside it, against 48 on
the earlier basis. A band shifted by the +2.8-point change of the median does not restore the earlier exclusions,
because the change has a per-shift SD of 2.0 points.

Fact register v1.9: DATA-17 (this evidence), BAND-01 (dry fuel rate), DATA-01, DATA-06, DATA-12, ASM-01, ASM-03 and
ASM-08 restated. Charge-record check band −9 to +10 % (median ± 4 robust SD; it was −6 to +12 %).

## Update 21 Nov 2026: the charged − burnt step of 18–20 Nov

From about 12 Nov charged coke ran above TRACE and FORGE, by 35–45 kg/tHM raw (charged − raw burnt) on 18–20 Nov
against 5–10 on 8–11 Nov. TRACE (oxygen balance) and FORGE (energy balance) agreed with each other at ~290–300
throughout. Nearly every shift from 17 Nov C failed the stable filter's blast-meter test (the same gap seen from the
other side), so the TRACE offset stayed at ~14.7 kg/tHM and did not follow (by design; decision 21 Nov 2026: diagnose
before changing the calibration).

Attribution (one input group at a time set to its 8–12 Nov stable-shift mean, raw burnt coke re-solved with
`burnt.k_burnt`; 13 baseline and 9 recent shifts):

| | 8–12 Nov (stable) | 18–20 Nov | Step |
|---|---:|---:|---:|
| Charged coke, dry | 302.0 | 312.9 | +10.9 |
| Raw burnt coke | 295.8 | 278.7 | −17.1 |
| Gap | 6.2 | 34.2 | +28.0 |

Raw burnt coke, contribution of each input to the −17.1 (sum −16.8): metered blast air per tHM −8.9, nut coke rate
−3.5, PCI rate −1.9, DPR dust −1.7, coke analysis −1.5, hot metal −0.6, burden −0.4, O₂/air +0.4, ηH₂ +0.6, ηCO +0.8.
Charged coke: the coke-rate set point went 296.5 → 303.9 (+7.4) and the charge record ran a further +3.5 above it.

Two observations point at the measurements rather than the furnace:

* **The hot furnace that surplus coke makes did not happen.** On 14–16 Nov the record showed ~327 kg/tHM and Si
  went to 0.7–0.8 % (TRACE followed charged then). On 18–20 Nov hot metal stayed at 1,481–1,489 °C, Si 0.41–0.46 %.
* **Hot-blast volume per tHM fell by about the oxygen flow on 18–19 Nov** (1,188 → 1,127–1,134 Nm³/tHM while O₂ rose
  to 69–71 Nm³/tHM). The balance takes blast air = hot-blast volume − O₂ (plant confirmation of 14 Oct 2026 that the
  tag includes the enrichment oxygen). If the oxygen joined after the meter on those days, blast air was read ~6 %
  low, about the −8.9 kg/tHM above.

Checks for the plant: where the enrichment O₂ joins relative to the hot-blast flow meter since 18 Nov; the coke
batch weights against the 304 kg/tHM set point; the coke moisture (a flat ~0.45 % for a month reads like a default).
No leak: the leak check reads `gap_only` (H₂ step 0.0 against the ~1.9 pt a leak of that size would give).

## Update 20 Nov 2026: charge-record errors spoiled the all-shift comparison

The admin page showed CCC 0.32 / 0.29 / 0.25 / 0.04 (shift / day / week / month) over every balanced shift, a floor
of ±67 kg/tHM (95 %) that did not average out and a raw closure of 104 %, while the stable shifts gave 0.87 / 0.91 /
0.96. The cause is not the calibration and not the descent time. It is the charge record (the hourly dataset's
`*_CALC_MT`), which was wrong in three periods:

| Period | What the record says | What the plant's own tags say | Effect |
|---|---|---|---|
| 1 Nov - 5 Dec 2025 | coke about 165 kg/tHM | coke-rate tag about 315, fuel-rate tag 520-545 | charged coke half; raw closure −9,100 t |
| 6-16 Dec 2025 | PCI 200-250 kg/tHM | PCI tag 150-180 | burnt coke low by the extra PCI carbon |
| 16-21 Oct 2026 | PCI about 420 kg/tHM | DPR PCI 190-195, fuel-rate tag 555-563 | burnt coke 30-60 kg/tHM (the PCI takes the oxygen); raw closure +3,500 t |

None of these shifts was stable (they fail the carbon closure or the blast-meter test), so the calibrated offset
of TRACE, FORGE's ω and the optimiser's calibration never used them. They entered only what is computed over every
balanced shift: the agreement, the variance split, the raw closure and the twin windows of those days.

**Check.** The plant's fuel-rate tag (`ACT. FUEL RATE`, kg/tHM) is an independent record of the same fuel. On
normal shifts the charge record's coke + nut coke + PCI runs 3.0 % above the tag (robust SD 2.2 %, the same each
month from Jan to Nov 2026). The pipeline now treats a shift whose record lies outside −6 % to +12 % of the tag
(±4 SD) as a data error, as it already did for an impossible ore moisture: it is not balanced, the twin window
fails with the reason, and the admin page lists it under *Shifts left out*
(`pipeline.exclusion_reasons`, `coke_energy.yml → charge_record_check.fuel_rate_band_pct`). On the record to 14 Nov
2026, 133 of 930 valid shifts are left out (Nov 2025 77, Dec 2025 29, Oct 2026 18, nine others); no stable shift is
among them.

Every balanced shift, 6 Sep 2025 - 14 Nov 2026, before → after the check:

| Window | CCC before | CCC after | SD before, kg/tHM | SD after |
|---|---:|---:|---:|---:|
| Shift (8 h) | 0.315 | 0.811 | 42.4 | 13.9 |
| 1 day | 0.293 | 0.896 | 39.7 | 8.4 |
| 3 days | 0.383 | 0.944 | 33.0 | 6.2 |
| 1 week | 0.247 | 0.958 | 33.4 | 4.5 |
| 2 weeks | 0.516 | 0.973 | 21.6 | 3.9 |

Floor √C 34.3 → 4.0 kg/tHM, raw closure 104.2 % → 100.7 %, C_b 0.67-0.79 → 0.995-0.999. The stable basis does not
change (shift 0.870, day 0.907, week 0.957; floor 1.7 kg/tHM).

**Tried and not adopted.**

* *Robust calibration.* A weighted median, a 3-MAD trimmed mean and Huber means (c 1.0, 1.5) in place of the
  tracker's weighted mean: no better on either basis (stable week SD 3.1 → 3.2-3.3; FORGE − TRACE SD 5.89 → 5.90-6.37).
  The stable filter already keeps outliers out of the pool.
* *Charged coke moved earlier by the descent time.* Charged coke over the 8 h window τ hours earlier
  (τ = 0-12 h): the day SD is flat for τ = 0-3 h (8.4-8.7 kg/tHM) and grows beyond. With the coke content of the
  burden (coke / ferrous) instead, the stable shifts peak at τ = 2-3 h, but the gain is below 0.2 kg/tHM. Over a day
  or more the change of coke in transit is small against the measurement error. Shift-resolution data cannot
  show a 4-7 h delay; hourly charge times against an hourly burnt coke are still needed for that.

What remains on the all-shift basis is the blast meter against the carbon balance on unsteady shifts (blast-meter
test failed, burnt − charged −20 to −65 kg/tHM on single shifts), which averages out as it should (day SD 8.4,
week 4.5). In Nov 2026 the raw burnt coke runs 4 % below charged (closure 95.9 %, blast meter −4 % against the carbon
balance); the tracker's offset (+10 to +12 kg/tHM) follows it. That is worth checking against the leak-water check
and the meter's calibration.

## Update 24 Oct 2026: recomputed on the pipeline of that day (RH-scaled blast moisture)

The pipeline of 24 Oct took blast moisture from the RH-scaled weather record (DATA-11), which enters the oxygen
balance; since 25 Oct the default source is the Open-Meteo daily RH (STRATA 1.5), and the live figures are the ones
on the admin page `/admin/burnt-vs-charged` (`services/burnt_vs_charged.py`, rebuilt by the `twin.trace` job). The
table below is kept as the record of that recomputation.
Recomputed on 568 stable shifts, 2025-09-13 – 2026-10-07 (production-weighted, non-overlapping windows; a
window needs two thirds of its shifts):

| Window | Windows | Bias, kg/tHM | SD | 95 % band | r | CCC | C_b |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shift (8 h) | 568 | +0.5 | 11.2 | -21.5 to +22.5 | 0.862 | 0.861 | 0.998 |
| 1 day | 200 | +0.5 | 7.9 | -15.0 to +16.0 | 0.922 | 0.921 | 0.999 |
| 2 days | 87 | +0.7 | 6.4 | -11.8 to +13.2 | 0.933 | 0.929 | 0.996 |
| 3 days | 60 | +0.8 | 5.8 | -10.6 to +12.2 | 0.944 | 0.943 | 0.999 |
| 5 days | 34 | +0.6 | 5.2 | -9.6 to +10.8 | 0.927 | 0.920 | 0.993 |
| 1 week | 23 | +0.3 | 5.2 | -10.0 to +10.5 | 0.934 | 0.923 | 0.988 |

Shift-to-shift movements (deviations from a 7-day centred mean): burnt coke correlates 0.35 with the charged coke
one shift earlier, 0.59 in the same shift and 0.20 one shift later. Raw (uncalibrated) oxygen balance against
the pipeline's charged coke: 1.012 [1.007, 1.018] over the last 3 months, 1.012 over 6 months,
1.017 over all months (week-block bootstrap 95 %). The conclusions below are unchanged.

## Result: 610 shifts, Sep 2025 – Oct 2026 (validation export)

| Window | Windows | Bias, kg/tHM | SD | 95 % band | r | CCC | C_b |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shift (8 h) | 610 | +0.4 | 11.2 | −21.6 to +22.4 | 0.857 | 0.855 | 0.998 |
| 1 day | 155 | +1.2 | 7.6 | −13.6 to +16.0 | 0.887 | 0.881 | 0.993 |
| 2 days | 83 | +1.2 | 6.1 | −10.8 to +13.2 | 0.925 | 0.920 | 0.995 |
| 3 days | 55 | +0.9 | 5.4 | −9.5 to +11.4 | 0.936 | 0.926 | 0.990 |
| 5 days | 34 | +0.8 | 5.1 | −9.2 to +10.8 | 0.922 | 0.917 | 0.994 |
| 1 week | 25 | −0.1 | 4.8 | −9.5 to +9.3 | 0.941 | 0.933 | 0.991 |

* **No level or scale gap.** C_b is 0.99–1.00 in every window: the CCC is limited by scatter alone.
* **The scatter averages out.** The fit gives a random part of ±10.3 kg/tHM per shift (falls as 1/√shifts) and a
  floor of ±4.4 kg/tHM SD (±9 at 95 %) that does not average out over a week: instrument drift inside the 14-day
  offset window (blast meter, top-gas analyser), about 1.4 % of the coke rate. The transit part (1/W²) cannot be
  separated from the random part at shift resolution.
* **Burnt follows charged.** Shift-to-shift movements correlate 0.57 in the same shift, 0.33 with the charged coke one
  shift earlier and 0.17 with the charged coke one shift later. The asymmetry is the direction of the lag.

So the two figures are the same coke within about ±9 kg/tHM (95 %) over a week, ±15 over a day and ±22 over a shift,
and the difference behaves like coke in transit plus measurement error, not like a second quantity.

## What this evidence cannot show, and why

* **The level agreement is partly by construction.** The page's burnt coke carries the 14-day median of
  (charged − burnt), which carries the blast meter's scale error. The spread, its decay and the direction of the lag
  do not depend on the offset. The independent level test is the mass closure of the RAW burnt coke (`K_burnt_raw` in
  the pipeline's shift table; `closure()`): total burnt / total charged over months, with the gap read as the meter's
  scale error. It needs the live pipeline (UAT).
* **8-hour shifts cannot measure a 6–7 h delay.** The correlation peak sits inside the same shift (about 1 h by
  interpolation), partly because both per-tHM figures of a shift share one production figure, which adds a lag-0
  correlation of its own. Measuring τ needs hourly charge-dump times (Neon `charge_data`, one row per charge) against
  an hourly burnt coke from the oxygen balance.
* **Tonnes per hour cannot show the delay at all.** Charging follows the stockline: the furnace takes charges as fast
  as the blast burns. On the hourly dataset, charged t/h trails burnt t/h by about an hour (r 0.22 at −1 h). It is the
  coke CONTENT of the burden (kg per tHM, coke per charge) that travels down the shaft.
* **Operators react.** Coke is added when the furnace is consuming more (cold, slips); that puts correlation at lag 0
  and smears the delay. An event study on clean coke-rate steps would isolate it, but the hourly dataset holds only
  13 steps with 12 h quiet before and 24 h after, too few on the current `COKE_CALC_MT` (which correlates only +0.16
  with the actual charge dumps).

## Practical reading

Compare burnt and charged coke over a day or more, never shift by shift. A shift where they differ by 15 kg/tHM is
normal; a week where they differ by more than ±10 kg/tHM is a meter, analyser or data problem worth chasing.
