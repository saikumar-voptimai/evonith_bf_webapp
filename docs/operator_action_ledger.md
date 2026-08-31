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

## 6. What is settled, and what is next

**Settled by the 180-day window:**

- The case-control comparison reaches significance (p = 0.000 against p = 0.425
  at 30 days). The effect was always present; the pilot lacked the sample.
- Thermal indicators *do* lead. The 30-day pilot's contrary finding was a power
  artefact.
- Both independent methods converge on the same three drivers.

**Still open, and the subject of Phases 3–5:**

1. Are trims and coke blanks triggered by *different* observations? 211 trims
   against 33 large moves is now enough to split them.
2. Is the 1.52 cuts-per-raise asymmetry explained by different triggers, or the
   same trigger treated asymmetrically?
3. **Reactive or anticipatory?** Whether the operator responds to a state that
   has already moved, or to an input change before the state moves. This is
   Phase 3 and the reason the work was framed this way.
4. Agreement with `burden_changing_purpose` — the operator's own stated reason,
   free text in `ops_config.burden_history`, as labelled ground truth.

## 7. Ask for the plant

`BF2 Coal rate actual value` implies a plant-side **set value** sibling tag that
is not ingested anywhere. Requesting it — and any coke-rate equivalent — would
give the PCI action signal directly. PCI is the co-control operators trade
against coke, so its action signal would roughly double what can be explained.
