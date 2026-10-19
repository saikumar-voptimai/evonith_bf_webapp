# Burnt coke against charged coke: the same coke, a descent time apart

Question (19 Oct 2026): how do we show that burnt coke and charged coke are the same quantity, only shifted by the
time the coke takes from the stockline to the tuyeres?

Code: `src/utils/coke_energy/transit.py` · page 11, *Model check → Burnt against charged coke* ·
`python scripts/burnt_vs_charged_coke.py SHIFTS.csv` · tests `tests/test_burden_optimiser_transit_and_commentary.py`.

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
