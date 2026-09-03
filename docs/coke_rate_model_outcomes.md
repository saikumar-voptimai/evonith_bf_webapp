# Coke Rate Prediction — What Was Built, and What It Actually Tells Us

*Last updated: 2 September 2026. Written in plain language; the arithmetic and
the code references are at the end of each section for anyone who wants to
check them.*

---

## 1. The short version

You asked for one thing: **a good coke rate prediction model, using both energy
balance and data, and a proper understanding of when the operator changes the
coke rate and why.**

Here is where that landed.

| Question | Answer |
|---|---|
| Can we predict the coke rate? | **Yes.** Typical error about 3.4%, roughly 11 kg/THM on a 330 kg/THM base. |
| What actually does the work? | The **energy balance** — physics, not a fitted model. |
| Then why is a data step needed at all? | The balance gets the *movement* right but the *level* wrong. One number fixed from recent plant history corrects the level. |
| Can we learn the coke-change rule from history? | **Only partly.** About 80% of changes are the operator reacting to something we can see. The remaining fifth we cannot explain from the tags we have. |
| Is the balance now driving the BMO fuel cost? | **Yes** — as of this change. Previously the fuel cost sat on the ML model's near-constant output. |

The honest headline: **the physics works, the plant record does not add much on
top of it, and what the plant record is genuinely good for is telling us how
much the physics is currently off by.**

---

## 2. Why the energy balance, and not a trained model

We tried both, side by side, on the same days.

An energy balance is not a model in the machine-learning sense. It is an
accounting statement: heat in must equal heat out. Blast sensible heat, carbon
combustion, hydrogen from moisture and volatiles on one side; ore reduction,
melting, slag, silicon and manganese reduction, shell losses, top gas leaving
the furnace on the other. Coke rate is whatever number makes the two sides
agree. Nothing is fitted.

Machine-learning models on this plant's record kept failing in the same way, and
it is worth being precise about the reason, because it is not a modelling
mistake — it is a property of the data:

> **The operator has already cancelled the signal we are trying to learn.**

If the furnace starts running cold, the operator raises the coke rate. By the
time the thermal indicators are recorded, the correction is already in. So in
the historical record, *low temperature appears alongside high fuel* — and a
model trained on it learns exactly the wrong direction. We measured this
directly: the fuel-versus-thermal correlation over the same hour is **−0.45**,
which is backwards. It only turns the right way around at a lag of 6–7 hours,
which is how long the burden takes to descend.

This is a well-known trap in control engineering (a plant under closed-loop
control hides its own response). It cannot be fixed with more data or a better
algorithm, because the information has been removed by the operator's own
competence.

The energy balance has no such problem. It does not learn from history at all.

**Reference:** `docs/coke_rate_model_v5_review.md`,
`docs/bmo_fuel_slag_si_findings.md` §7.

---

## 3. What the balance gets right, and what it gets wrong

Backtested over **239 days**:

| | Bias | Typical error | MAPE | R² |
|---|---|---|---|---|
| Energy balance, as-is | **+19.7 kg/THM** | — | 7.24% | +0.07 |
| Energy balance + one offset | +0.2 | ~11 kg/THM | **3.37%** | **+0.74** |

And re-measured independently over the most recent **89 days** (the window the
in-app chart draws), which is a more disturbed period:

| | Bias | Typical error | MAPE | R² |
|---|---|---|---|---|
| Energy balance, as-is | +24.2 kg/THM | 28.7 | 9.54% | −0.41 |
| Energy balance + offset | −0.3 | 15.5 | **4.95%** | **+0.24** |

**Expect the in-app figures to be worse than the 239-day ones.** They are
measuring a shorter, noisier stretch. What matters is that the correction moves
things the same way in both: bias to near zero, and the error roughly halved.

Read the two rows carefully, because they say different things.

- **R² = 0.07** on the raw balance means it is barely better than just guessing
  the average coke rate every day. That sounds like failure.
- But the bias is **+19.7 and consistent**. The balance is not confused — it is
  *systematically* high by about the same amount every day.

That combination is the good case. A model that is wrong by a varying amount is
broken. A model that is wrong by a *steady* amount is correct in shape and
merely mis-levelled, and one subtraction fixes it. After that subtraction the R²
goes from 0.07 to **0.74** — meaning it now tracks three-quarters of the
day-to-day movement.

### Why is it 20 kg/THM high?

We know the two main suspects, and neither is a modelling problem:

1. **The top-gas analyser appears to under-read.** CO + CO₂ comes to about 3
   percentage points less than a carbon balance says it should. Gas leaving the
   furnace carries heat out; if we under-count the gas, we under-count the heat
   leaving, and the balance demands extra coke to make up the difference.
2. **The shell heat-loss basis is unresolved.** Depending on which basis is
   correct, the answer moves by about **11% of the coke rate**. This one is
   still open and is worth settling — it is the single largest known uncertainty
   in the whole calculation.

This matters for how we should read the offset. **The offset is a measurement of
how much the balance is still missing.** It is deliberately shown on screen
rather than folded in silently. When the analyser and shell-loss questions are
settled, the offset should shrink on its own — and if it does not, something
else is wrong and we would want to know.

**Reference:** `docs/energy_balance_findings_and_open_decisions.md`,
`docs/plant_data_request.md` (what we have asked the plant for and why).

---

## 4. Why one number, and not a correction model

We did try fitting a proper correction — a small model on nine features,
predicting the residual. It scored slightly better (MAPE 3.16% against 3.37%).

We did not ship it, for this reason:

> It wanted **−18.8 kg of coke per 1% of silicon**, against an energy balance
> that already carries silicon reduction at 24.6 MJ/kg — a *positive* coke cost.
> The correction was arguing with the model it was supposed to be correcting.

That is patching, not calibrating. A correction that fights the physics will
work on the days it was fitted on and mislead on any day that is genuinely
different — which is precisely the day an operator most needs it. A single
offset has one parameter, can be inspected on screen, and shrinks toward zero as
the underlying defects get fixed.

**Reference:** `src/utils/bmo/coke_calibration.py` (the reasoning is recorded in
the module docstring), `docs/coke_model_shootout.png`.

---

## 5. How often the offset must be refitted — this was measured, not guessed

The offset drifts, because the things causing it drift: **+16.0, +16.9, +22.4,
+23.8 kg/THM** across four successive quarters.

You asked whether refitting daily on the last 90 days improves things or at
least holds them. We tested four refresh policies over 281 days, every one of
them strictly causal (no day is allowed to see its own correction):

| Policy | Refits | Bias (kg/THM) | MAE | MAPE | R² |
|---|---|---|---|---|---|
| Fit once, never again | 1 | +37.1 | 38.3 | 12.82% | −1.334 |
| Every quarter | 3 | +18.8 | 22.6 | 7.62% | −0.166 |
| Every month | 9 | +9.0 | 15.7 | 5.21% | +0.328 |
| **Every day, trailing 90 days** | 281 | **+6.5** | **13.9** | **4.59%** | **+0.428** |

**Answer: daily refitting improves it, and comfortably.** It is not a wash —
daily against monthly is **1.8 kg/THM less error and +0.10 on R²**, and both of
the slower policies land at a *negative* R², meaning they are worse than no
correction at all.

Note these R² figures are lower than the +0.74 quoted in §3. That is not a
contradiction — this test scores every policy on the same held-out days after a
90-day warm-up, including the disturbed periods, whereas §3 is the forward test
on the shipped configuration. The ranking is what matters here, and it is
unambiguous.

The reason is in the staleness table, which is the more useful result:

| Offset held without refitting | MAE | MAPE | R² |
|---|---|---|---|
| 0 days (fresh) | 13.9 | 4.59% | **+0.428** |
| 30 days | 18.9 | 6.34% | +0.054 |
| 60 days | 27.7 | 9.29% | −0.820 |
| 90 days | 33.0 | 11.12% | **−1.232** |
| 180 days | 56.6 | 18.82% | −3.002 |

Read the 90-day row plainly: **a calibration left for three months makes the
prediction worse than applying no correction at all.** A negative R² means you
would have done better guessing the average.

The drift works out to a median **1.14 kg/THM over 7 days and 3.26 kg/THM over
30 days** — call it 3.3 kg/THM per month. An earlier version of
this code assumed 2 kg/THM per *quarter* and warned after 45 days. That was
wrong by roughly five times, and has been corrected — the warning now fires at
**14 days**.

**Reference:** `scripts/coke_calibration_cadence.py`,
`src/utils/bmo/coke_calibration.py::STALE_AFTER_DAYS`.

---

## 6. When does the operator change the coke rate, and why

This was the second half of what you asked for. Over 188 days we found **244
coke setpoint changes** — about one every 18 hours — and tried to explain each
one.

### What we found

**There are two quite different populations of change, not one:**

| Type | Count | What it is |
|---|---|---|
| **Trims** | 211 | Small adjustments, a few kg/THM, in normal running |
| **Blanks / large moves** | 33 | Big steps, usually tied to banking, stoppages or PCI being cut |

Averaging these together produces a meaningless middle figure. They are separate
behaviours and have to be counted separately.

**Cuts outnumber raises by about 1.5 to 1.** The operator trims the coke rate
down more often than up — consistent with steadily pushing for efficiency and
correcting back when the furnace protests.

**About 80% of changes are reactive.** Something had already moved — a thermal
indicator, gas utilisation, silicon — and the operator corrected it. Roughly 2%
we could not explain at all from any tag we have. The rest sit in between.

### Three things worth knowing about this result

1. **The 20% we cannot explain is a real finding, not a failure.** It bounds how
   much of operator behaviour is recoverable from the instrumentation we
   currently have. If we want more, we need more tags — not more analysis.

2. **The response lag we first reported was wrong and has been retracted.** An
   early run found a clean 5.7-hour lag. On checking, it tracked the analysis
   window rather than the furnace — it came out at 0.6× the lookback every
   single time, whatever the lookback was. That is an artefact. It is recorded
   here because a plausible-looking number that turns out to be a measurement of
   your own method is exactly the kind of thing that should be written down.

3. **The setpoint is not the coke rate.** `coke_rate` in the online data is the
   operator's *instruction*. It sits flat at exactly 300.00 for days and then
   steps. What is actually charged runs about **3.8% higher**, and the two agree
   within 10 kg/THM on only 42% of days. Any analysis that treats the setpoint
   as a measurement will be quietly wrong.

**Reference:** `docs/operator_action_ledger.md` (the full per-event ledger),
`scripts/operator_action_*.py`, `tests/test_operator_action.py`.

---

## 7. What changed in the Blend Mix Optimizer

### The fuel cost now sits on the energy balance

Previously the reported fuel cost was back-solved from the ML fuel-cost model.
That model's output is very nearly a constant — around 13,364 Rs/THM regardless
of blend — so the reported fuel rate was effectively pinned at
`487 + 0.357 × PCI` no matter what the furnace was doing.

Now:

```
coke rate  =  energy balance at the CURRENT controls and burden
              −  the rolling bias offset
              +  the physics correction for THIS blend
```

The three parts have clean, separate jobs:

- **The balance** sets the level, and responds to what the furnace is actually
  being asked to do.
- **The offset** removes the known bias. One number, shown on screen, currently
  **+24.5 kg/THM** fitted on 88 days.
- **The physics correction** supplies the blend-to-blend difference. It is zero
  at current conditions by construction, so it can never shift the level — only
  the comparison between blends.

Keeping them separate is what stops them double-counting each other.

**If any part is unavailable** — live blast tags missing, no calibration, or a
balance that fails to solve — the page falls back to the observed coke rate
**and says so on screen**. It never presents a fallback as though it were the
balance's answer.

### A retrain button

On the new **Model accuracy** tab. It rebuilds the daily history from the plant
record and refits the offset over the trailing 90 days. It warns when the
current offset is over 14 days old, for the reasons in §5.

### Two accuracy charts

Because neither prediction can be checked by eye at the moment it is shown:

- **Coke rate:** predicted against what was actually charged, day by day, at the
  PCI and nut coke actually run — with the raw uncorrected series available
  underneath, so the offset's size is visible rather than merely stated.
- **Silicon:** predicted against cast analysis. **Read this one with care.**
  Among the Si model's inputs are *earlier silicon readings*, so part of what
  looks like skill is yesterday's cast carried forward. It is a fair reflection
  of what the model does in service — an operator does know the last cast — but
  it is not evidence that the burden chemistry terms are doing the work. This
  caveat is printed under the chart, not just here.

### The results screen was rebuilt

The LP and DE panels each used to be a single column of about ten stacked
sections. They are now grouped into sub-tabs — **Blend, Fuel & coke, Slag,
Controls, Path there** — with the headline metrics staying above the tabs, since
cost and feasibility are what gets checked first.

The blend table and its share donut now sit side by side rather than one under
the other. The Manual-versus-optimizer comparison leads with the decision (which
option is cheapest, and by how much) and puts the 23-row detail table behind
grouped tabs, instead of the other way round.

---

## 8. What is still open

| Item | Why it matters | Status |
|---|---|---|
| **Shell heat-loss basis** | Worth about **11% of the coke rate** — the largest single uncertainty | **Deferred by you.** Still open. |
| **Top-gas analyser calibration** | ~3 points of CO+CO₂ missing; drives most of the +20 kg/THM bias | Requested from plant |
| **GCP and flue dust daily analysis** | Currently using average values | Requested from plant |
| **A PCI setpoint tag** | `BF2 Coal rate actual value` implies a *set value* sibling exists but is not ingested. It would give the PCI action signal directly — and PCI is the lever operators trade against coke | Requested from plant |
| **Residual scatter differs between the two pipelines** | The in-app rebuild and the original analysis script now agree on the offset to within 0.3 kg/THM, but the in-app one shows more day-to-day scatter (sd 16.9 against 10.4 after outlier removal). The *level* is right; a handful of individual days are noisier. Suspects are the shell-loss basis and the dust figures | **Open.** Does not affect the offset the button writes. |

### One bug worth recording

While building the in-app chart, the rebuilt pipeline was found to be defaulting
the **top-gas temperature to a constant 140 °C** when the real value in the
dataset averages **191 °C** with a spread of 30. The balance moves 2.4 kg/THM
per 20 °C of top-gas temperature, so this alone was under-predicting the coke
rate by about 6 kg/THM, and it made the in-app pipeline disagree with the
shipped calibration by roughly that much.

It was found only because the predicted-vs-measured chart was built and the two
numbers were compared. That is the argument for having the chart at all: a
constant standing in for a measured variable is invisible in the code and
obvious in the plot.

None of these block using the tool. All of them would make the offset smaller,
which is the point.

---

## 9. If you read only one thing

The energy balance predicts the coke rate well **once its level is corrected**,
and the correction has to be refreshed roughly monthly or it becomes worse than
useless. The size of that correction is not an embarrassment to be hidden — it
is a live measurement of how much the furnace instrumentation is currently
lying to us, and it is displayed for exactly that reason.

The operator's coke changes are mostly explicable and mostly reactive, but one
change in five cannot be accounted for with the tags we have today. That is a
statement about the instrumentation, not about the operator.
