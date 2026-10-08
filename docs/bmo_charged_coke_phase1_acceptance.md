# Charged-coke engine: acceptance record

Branch `feature/235_condition_aware_coke_model`, 7 October 2026. Data-Driven
now means the charged-coke engine only. The robust 24-h engine is no longer
selectable or executed by the BMO page; its old artifacts remain only as an
audit/reproducibility record.

## 1. What is forecast

There are five values per completed hour. For h = 1 … 5 rows after the latest
complete row t, each value is the 4-h charged coke rate as it will read at row
t+h:

    1000 × Σ COKE_CALC_MT(rows t+h−3…t+h) / Σ PRODUCTIONTONNESPERHR(same rows)

The 4-h basis is used because a single hour's ratio swings with whole-charge
counting:

| Measure | Hour-to-hour change (std) | Error from holding the current value, +1 … +5 h |
|---|---|---|
| Single-hour ratio | 43.6 kg/THM | 35 / 27 / 29 / 29 / 29 |
| 4-h rate | 9.1 kg/THM | 7.2 / 7.6 / 8.5 / 10.3 / 9.3 |

The +5 h value is the block the research study validated (rows t+2…t+5). It
feeds the BMO optimiser's coke level.

**Timestamp contract (measured on the live sources, 5–7 Oct):**

| Row labelled T (IST) | Covers | Evidence |
|---|---|---|
| `COKE_CALC_MT`, `NUTCOKE_CALC_MT` | charges in [T−1 h, T) | per-charge records +60 min reproduce the column; median difference 0 t |
| Online tags (blast, PCI, production …) | [T−30 min, T+30 min) | raw Influx points: wind within 8 Nm³/h, PCI within 0.07; other alignments off by 200–800 Nm³/h |
| Clock | IST | UTC alignment triples the error |

The file is rebuilt hourly at about :06. Its newest row is still filling, so a
row is complete only when T + 30 min ≤ build time.

This rule was checked on two consecutive builds (04:06:53 → 05:06:54):
- Rows the rule called complete did not change.
- The in-progress row did change: wind by 751 Nm³/h, PCI by 1.5 kg/THM.

For each forecast, data time, issue time (about t+1 h 06 min) and the value's
own clock window are kept separate. The +h value is plotted at t+h. It covers
coke charged in [t+h−4 h, t+h) against production over
[t+h−3.5 h, t+h+0.5 h).

## 2. The pipeline reproduces the research study exactly

On the study's own export (SHA-256 `bd626c1c…`), the ported feature pipeline,
NumPy kernel and display policy reproduce the study's single-horizon model:

| Check | Result |
|---|---|
| Causal frame, slag + parts, 23 channels, labels, gates (6,648 h) | identical (1e-9); our slag code = the pinned snapshot |
| Gate inputs, severity, reason text (6,648 h) | 0 mismatches |
| 463 computable frozen outputs | 463 of 463, max 1.7e-13 kg/THM |
| 480 hourly display states, ranges, reasons, previous-forecast fields | 480 of 480 |
| 12 research what-if scenarios | exact |
| Truncation; reported rates set to junk | outputs unchanged |

The study's replay was seeded with its own pre-freeze recovery count. That only
matters for matching its hour-by-hour states. A live model starts its own
hourly chain when it is deployed.

## 3. The live model: retrained, reviewed, accepted

**Retrain** (page → Data-Driven panel → Model & retrain, about 6–9 minutes in
the background):
- Same structure as the study, with five outputs. It fits per-horizon signed PCI/nut/slag coefficients, a ridge, and three attention seeds on the remaining residual.
- Training data: January 2026 onward (this avoids the November 2025 coke undercount), core-gate hours with all five labels matured.
- Four 7-day folds, each predicted by a model fitted only on earlier data. They set the error range for every condition and horizon, and the novelty and branch-disagreement limits.
- The model deployed is the last fold's model, so the most recent 7 days were never used to fit it.
- The review decides every clock hour exactly as the live service does.

**Accept / roll back:**
- Supervisor or admin only.
- Each accept or roll-back records the version, who did it, when, and the previous version.
- Nothing changes until a candidate is accepted.

**Shipped model** (`20261007_0400`): data to 7 Oct 04:00, fitted to 30 Sep,
scored on 30 Sep–7 Oct, which it never saw.

| Horizon | +1 h | +2 h | +3 h | +4 h | +5 h |
|---|---|---|---|---|---|
| Forecast MAE (kg/THM) | 6.6 | 7.1 | 8.0 | 8.5 | 8.6 |
| Holding current rate | 7.0 | 7.7 | 8.8 | 10.9 | 10.5 |
| Skill | 5% | 9% | 9% | 23% | 18% |
| Inside 90% range | 90% | 87% | 86% | 83% | 87% |

- **By condition (+5 h):** cruising 4.5 (holding current rate 7.3), mild instability 7.9 (9.9), outside cruise 10.6 (12.2).
- **Availability:** 148 of 168 clock hours shown.
- **Coefficients:** PCI −0.40 to −0.44 at every horizon; nut coke −0.09 to −0.22; slag 0 to +0.06.
- **Kernel check:** the exported model reproduces the training code to 2.3e-13 kg/THM. Live states match the review on 168 of 168 hours.

### Why the robust 24-h model was retired

Its later-time MAE was 6.8 kg/THM, but that is error against a trailing,
heavily smoothed 24-hour target. It is not a forecast of the charged-coke rate
five hours ahead, so that number cannot be compared directly with the charged
model's +5 h MAE. On its own later-time test it also had R2 -0.28 and bias
-6.3 kg/THM.

The charged model predicts the operating quantity and horizon the optimiser
needs. At +5 h its MAE is 8.6 kg/THM versus 10.5 for holding the current rate,
and it has positive skill at every horizon. The old engine therefore has no
demonstrated operational advantage for this use case.

**Operator view** (three tabs under the status line):
- **Next 5 hours:** the five values with their ranges, over the last 24 h measured.
- **Last 7 days:** forecast vs actual, one horizon at a time, with markers coloured by condition (cruising / mild instability / outside cruise). Paused hours are left out. Only hours after the model's training cutoff are shown, and errors are tabulated against holding the current rate.
- **Model & retrain:** the active model's report, retrain, candidate review, and accept or roll back.

**Force-predict:** off by default and available only with Data-Driven selected.
If the model computed a value but the display policy withheld it, the operator
may expose and use that value. The page keeps every exact policy violation
visible, labels the result as forced, and does not claim a validated error
range. The issued forecast ledger remains unchanged. Missing model inputs still
cannot be bypassed.

The result area also retains **Model accuracy**: it shows the active five-hour
model's unseen-data report, plus the Physics-Driven energy-balance calibration
and recent accuracy and the hot-metal silicon model.

## 4. Remaining items

1. **The live source is filtered and imputed upstream.**
   - The cleaner drops hours outside its cruising filters, including a reported fuel rate of 100–670.
   - It refills gaps with an imputer fitted over all rows, so history is revised between builds.
   - Consequence: off-cruise hours show only as missing.
   - Mitigation: issued forecasts are recorded once and never recomputed. An unfiltered, un-imputed hourly source is needed for an honest non-cruise status.
2. **Build time:** the completeness rule needs the publisher's build time. The app now stores `Last-Modified` on download; older caches fall back to "newest row − 1 h".
3. **Optimiser blend response:** fixed at +0.10 kg coke per kg slag for plant
   testing. It is stored in config, locked for normal users and editable by an
   admin. The forecast already includes raw-material composition; this is the
   single extra response applied when comparing candidate blends with the
   current burden.
4. **Nut-coke basis:** without an override, the BMO uses its 70 kg/THM set point while the forecast reflects the live 4-h nut ratio.
5. **Page run:** the panel was checked through a harness on real data. A logged-in run of the full page with the charged engine selected is still to do.
