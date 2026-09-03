# Operator Action Signal — Findings

**What this is:** an attempt to explain each coke-rate decision an operator made,
by treating setpoint changes as timestamped interventions rather than as
observations.

**Status:** Phases 1–2 complete on a **180-day window (240 routine events)**.
Both methods now agree. A 30-day pilot was underpowered and is retained below
only for the methodological errors it exposed.

Scripts: `scripts/operator_action_events.py`, `scripts/operator_action_attribution.py`

---

## 1. Why this line of work exists

Every fuel model in this project has hit the same wall. The plant holds total
fuel nearly flat and the operator closes the loop, so passive correlation cannot
separate what coke *does* from what the operator does *about* it. The documented
contemporaneous fuel/thermal correlation is **−0.45** — the wrong sign, because
it measures the controller and not the furnace.

The unlock: **`coke_rate` in InfluxDB is not a measurement. It is the operator
setpoint.** Raw, it is piecewise constant — 28 distinct levels over 30 days,
sitting at exactly 305.00 for hours at a time. Every change is a deliberate,
timestamped decision.

That converts an observational problem into a quasi-experimental one.

---

## 2. The action signal

Over 188 days, **244 routine control events** on a normally running furnace
(plus restart- and PCI-outage-related changes, classified out).

### The ratchet

147 cuts against 97 raises — **1.52 cuts per raise** — at an *identical* median
step of 5.0 kg/THM, but with the raises carrying the fatter tail.

**Operators trim down often in small steps and add back in fewer, larger moves.**
That asymmetry is a policy, and it is the first thing any attribution has to
explain.

### Two populations, not one

| | n | median \|Δ\| | median held |
|---|---|---|---|
| trim cut | 124 | 5.0 kg/THM | 10.4 h |
| trim raise | 87 | 5.0 kg/THM | 9.2 h |
| large cut | 23 | 30 kg/THM | 3.0 h |
| large raise | 10 | 50 kg/THM | 5.9 h |

A big raise that is stepped back down within hours is a **coke blank charged
against a chill** — a different decision with a different cause. Pooling the two
would confound both, so they are attributed separately.

### Timing and shift

Median **8.4 h** between actions; only 3% come within an hour of the previous
one, so these are genuinely separate decisions rather than one entry keyed in
stages. Shift split A 90 / B 85 / C 69 — no strong handover skew, which rules
out the most obvious non-physical explanation.

---

## 3. What prompted each decision

**Both methods now agree**, which is the strongest thing here — they are
independent of one another and converge on the same short list.

### Method A — how often each observation leads

The null is that any of the 21 tracked observations could top the ranking by
chance, 4.8% each. **The placebo lands at exactly 6% (14/240)**, which is the
check working as designed.

| Observation | Leads | Share | p (binomial) |
|---|---|---|---|
| **hm_per_charge** | 33/240 | 14% | **5.5 × 10⁻⁸** |
| **runner_temp_cr_taphole** | 31/240 | 13% | **5.5 × 10⁻⁷** |
| **top_press_avg** | 27/240 | 11% | **3.6 × 10⁻⁵** |
| body_raft | 19/240 | 8% | 0.022 |
| h2_pct | 16/240 | 7% | 0.11 |
| *PLACEBO* | *14/240* | *6%* | — |

### Method B — case-control on excursion magnitude

Did observations deviate *further* before an action than during a quiet spell?
240 cases against 960 controls, matched on shift.

| Observation | case median \|z\| | control | gap |
|---|---|---|---|
| **runner_temp_cr_skimmer** | 1.119 | 0.320 | **+0.799** |
| **hm_per_charge** | 1.529 | 1.156 | **+0.373** |
| h2_pct | 2.209 | 1.967 | +0.241 |
| runner_temp_pci_skimmer | 1.437 | 1.225 | +0.212 |
| top_press_avg | 0.685 | 0.496 | +0.188 |
| runner_temp_cr_taphole | 1.232 | 1.054 | +0.178 |
| *PLACEBO* | *2.428* | *2.366* | *+0.062* |

Six observations clear the placebo bar; the placebo ranks 7th of 21, exactly
where noise belongs.

**Permutation test: p = 0.000.** Observed top-5 mean gap 0.363 against a null
median of 0.138 and 95th percentile 0.257, over 200 relabellings of the pooled
set. At 30 days this same test gave p = 0.425 — the effect was always there, the
sample was not.

### What the two methods agree on

| Signal | Method A | Method B |
|---|---|---|
| **hm_per_charge** | 1st | 2nd |
| **Runner temperature** (thermal) | 2nd | 1st and 4th |
| **top_press_avg** | 3rd | 5th |
| RAFT / H₂ | 4th / 5th | — / 3rd |

Three things drive coke decisions on this furnace:

1. **Hot metal per charge** — the burden yielding less iron per charge is a
   direct reason to add coke.
2. **Runner temperature** — the most direct thermal reading an operator has.
3. **Top pressure** — the aerodynamic state of the stack.

**The 30-day pilot got the thermal question wrong.** It reported that thermal
indicators do *not* lead, and flagged that as surprising. With 240 events they
lead strongly by both methods. That is a straightforward power artefact, and a
caution against reading a null from a small sample.

---

## 4. Four errors the checks caught

Each would have produced a confident, plausible, wrong answer. Recorded because
the same traps will recur.

| Error | What it did | Fix |
|---|---|---|
| **Value-band filter** | Rejected setpoints outside `[280,420]` (a *BMO recommendation* guardrail, not a limit on operators). Clipping a real 297→560 escalation mid-flight **invented** a "+113" event and a "−40" | Filter on operating state — blast > 40,000 Nm³/h, production > 20 t/h |
| **Nominal z threshold** | Called 41/41 events "explained" — the peak is over 21 observations × 32 time points, so noise clears \|z\| ≥ 2 almost always | Use the placebo's own peak distribution as the null (95th pct = 3.02) |
| **Collapsed scale** | `top_press_avg` scored **z = −133** because its rolling IQR fell toward zero during a quiet week | Floor the rolling scale at 20% of the tag's long-run IQR |
| **Controls inside a stoppage** | The placebo column is synthetic and never NaN, so adding it before `dropna(how="all")` made that filter a no-op — 33 controls came from inside a 38 h outage | Judge usability on real observations only |

Two further corrections to method:

- **The trigger window included the decision instant**, letting a mechanical
  consequence pose as a trigger. It now stops 30 min short. `hm_per_charge`
  dropped only 13 → 12, confirming it leads genuinely rather than mechanically.
- **The first shuffle test compared real events against random *times***. Not a
  fair null: controls are drawn from quiet spells, so random times differ from
  them for reasons unrelated to actions. Replaced with a proper permutation over
  the pooled set.

---

## 5. Operating context, and one finding for the plant

**51.7 h of stoppage in 693 h — 7.5% downtime** across four outages, the longest
38.4 h.

During those outages the coke setpoint sits parked at 480–560 kg/THM while
production, blast and PCI are all zero and RAFT reads **−3000** (the calculation
dividing by a blast that is not there). Those are held values, not decisions.

**PCI loss is the marker no time buffer catches.** On 2026-08-24 PCI fell
210 → 112 → 2 → 0 while blast held at ~105,000 and the coke setpoint jumped
297 → 515 — banking ahead of a blow-down that did not begin for another **nine
hours**. Before and after are tested separately, deliberately: a window spanning
the event gives a median of 33.5 kg/THM, above any sane threshold, while the
post-event median is 0.0. The two cases are different decisions — *PCI already
off then coke raised* is replacing lost fuel; *coke raised then PCI cut* is
banking.

---

## 6. Reactive or anticipatory (Phase 3)

All 240 routine decisions, classified on what was true before and after each:

| Class | n | % |
|---|---|---|
| **reactive** - the state had already deviated | 193 | 80% |
| coordinated - another control moved with it | 21 | 9% |
| anticipatory? - quiet before, moved after | 20 | 8% |
| unexplained | 6 | 2% |

**Only 2% unexplained.** The observable tags span most of what operators act
on, which is a better result than the plan expected.

**Every large move is reactive - all 33.** A coke blank is never a coordinated
package and never anticipatory; it is always a response to a state that has
already moved. That vindicates separating trims from blanks.

**The ratchet is NOT explained by class**, contrary to expectation. Cuts are
77% reactive and raises 85% - if anything raises are *more* reactive. The
asymmetry lives inside the reactive population, so it is about *how* operators
respond rather than about responding to different things.

**Controls moving with coke:** hot blast temperature 38% of events, PCI 32%,
oxygen 26%, blast volume 20%. Blast temperature co-moves more often than PCI -
the coke setpoint is part of a thermal package more often than a fuel
substitution.

### A number retracted inside the same run

The script first reported a median operator response lag of **5.7 h**. It is an
artefact - widening the lookback window moves it in lockstep:

| Window | 4 h | 8 h | 12 h | 24 h | 48 h |
|---|---|---|---|---|---|
| Median lag | 2.8 | 5.3 | 6.8 | 13.8 | 27.9 |

Every one is about 0.6 x the window - what you get when the peak falls at a
random point inside it. **Do not quote a reaction time from this work.**
Measuring one needs excursion *onset* rather than peak, plus a control
comparison.

---

## 7. How long inputs take to show up (Phase 4)

Six inputs against eleven responses, lags 0-24 h, on three filter bases with a
placebo input as the bar.

| Basis | Gas | Thermal | Aero |
|---|---|---|---|
| levels | 0 h | 5.5 h | 0 h |
| detrended | 0 h | **12.5 h** | 0 h |
| differenced | 0 h | 0 h | 0 h |

**Gas and aero respond at lag 0**, robustly across all three. The shipped
model's `lag1` for GasImpact is about right.

**Thermal is not identified.** Three answers from the same data, ranges
spanning the whole sweep. Neither the docs' 6-7 h nor the model's `lag4` is
supported - and neither is refuted. That matters because **127 of the shipped
model's 253 features are lagged on that assumption.**

Using only two bases would have produced a confident wrong answer either way.
Levels alone inflates everything through shared drift; first differencing is a
high-pass filter that removes the very band a multi-hour lag lives in, so its
flat 0 h cannot refute a lag.

**A tautology worth recording:** `body_raft` is *calculated* from blast
temperature, oxygen and PCI - the inputs being swept against it. Its lag-0
correlation is definitional. Drop it from any future thermal sweep, leaving the
runner temperatures, which are measured only at tapping - a likely reason the
thermal lag resists identification at all.

---

## 8. Phase 5 could not be done, and the plan was wrong to expect it

The plan proposed validating the attribution against `burden_changing_purpose`,
the operator's own written reason, as labelled ground truth. **That was my
error, on two counts.**

**Wrong table for the question.** The field records why a *burden distribution*
change was made - rings, angles, charge pattern - not why the coke rate moved.
They are different decisions.

**And there is almost no data.** `ops_config.burden_history` and its source
`public.burden_distribution` are the same 21 rows, spanning 2025-06-28 to
**2026-03-15**. Against 240 coke events (2026-03-06 to 2026-08-28) the overlap
is **9 days, 4 coke events and 2 burden changes**, with exactly one coke event
within 6 h of a burden change.

**The table appears to have stopped being written in March 2026** - worth the
plant knowing independently of this analysis.

No other source of operator intent exists. Every schema in the replica was
searched for columns named like remark, reason, comment, purpose, note,
observation or action; the only hits are feedback tickets and this field.

### What the reasons do still tell us

Qualitatively the vocabulary corroborates the quantitative findings - wall
temperature fluctuation, temperature spike in the bosh region, top temperature
deviation, centre working, gas flow in the periphery. Thermal and aerodynamic
language throughout, the same two families Phases 2-3 found dominant. That is
corroboration of a weak kind, not validation, and should not be presented as
more.

---

## 9. What is settled, and what is not

**Settled:**

- Coke setpoint changes are recoverable as clean, timestamped interventions.
- Three drivers, agreed by two independent methods: hot metal per charge,
  runner temperature, top pressure.
- 80% of decisions are reactive; only 2% unexplained.
- Every large move (coke blank) is reactive.
- Gas and aero respond immediately; the thermal lag is not identifiable.

**Not settled, and now known to be unreachable with current data:**

- **Anticipation.** The 8% class is an upper bound. Separating anticipation
  from consequence needs the disturbance, and the disturbance log has 21
  records ending in March.
- **Operator reaction time.** Needs onset-based measurement, not peak.
- **The thermal lag.** Needs a thermal response measured continuously rather
  than at tapping.

**Worth doing next, in order of value:**

1. Ask the plant to resume logging burden changes, and to log *coke* changes
   with a reason. That one change would make anticipation answerable.
2. Re-run Phase 4 without `body_raft`, on runner temperatures alone.
3. Onset-based reaction time, with a control comparison.
