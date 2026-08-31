# Operator Action Signal — Findings

**What this is:** an attempt to explain each coke-rate decision an operator made,
by treating setpoint changes as timestamped interventions rather than as
observations.

**Status:** Phases 1–2 complete on a 30-day window. One firm result, one
underpowered result, and a clear reason to widen the window.

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

Over 30 days, **55 setpoint changes**, of which:

| Context | n | What it is |
|---|---|---|
| **normal** | **40** | Routine control on a running furnace — the population of interest |
| restart | 14 | Inside a stoppage window or its buffers; a blow-in, not control |
| pci_off | 1 | PCI lost or cut while still blowing; coke replacing lost fuel |

### The ratchet

26 cuts against 14 raises — **1.86 cuts per raise** — at an *identical* median
step of 5.0 kg/THM. But the raises carry the fatter tail (p90 38 against 15).

**Operators trim down often in small steps and add back in fewer, larger moves.**
That asymmetry is a policy, and it is the first thing any attribution has to
explain.

### Two populations, not one

| | n | median \|Δ\| | median held |
|---|---|---|---|
| trim | 35 | 5.0 kg/THM | ~8 h |
| large | 5 | 30–50 kg/THM | 6.4 h |

A big raise that is stepped back down within hours is a **coke blank charged
against a chill** — a different decision with a different cause. Pooling the two
would confound both, so they are attributed separately.

### Timing and shift

Median **7.6 h** between actions; only 3% come within an hour of the previous
one, so these are genuinely separate decisions rather than one entry keyed in
stages. Shift split A 17 / B 13 / C 10 — no strong handover skew, which rules
out the most obvious non-physical explanation.

---

## 3. What prompted each decision

### The firm result: two observations lead actions

The null is that any of the 21 tracked observations could top the ranking by
chance — 4.8% each. **The placebo hits exactly that, 2/41**, which is the check
working as designed.

| Observation | Leads | Share | p (binomial) |
|---|---|---|---|
| **hm_per_charge** | 12/41 | 29% | **2.9 × 10⁻⁷** |
| **top_press_avg** | 7/41 | 17% | **3.0 × 10⁻³** |
| tuyere_velocity | 4/41 | 10% | 0.13 |
| runner_temp_pci_taphole | 4/41 | 10% | 0.13 |
| body_dp_bottom | 3/41 | 7% | 0.31 |
| *PLACEBO* | *2/41* | *5%* | — |

**Hot metal per charge and top pressure are real leaders.** Nothing below them
separates from noise at this sample size.

Both are physically sensible. HM per charge falling means the burden is yielding
less iron per charge — a direct reason to add coke. Top pressure is the
aerodynamic state of the stack, which an operator watches continuously.

Notably, the *thermal* indicators (RAFT, runner temperatures) do **not** lead
strongly, which is mildly surprising and worth testing on a longer window before
drawing anything from it.

### The underpowered result: case-control

A permutation test — pooling cases and controls, relabelling at random 200 times
— gives **p = 0.425**. Action hours are **not** distinguishable from no-action
hours by these effect sizes.

**This is a power problem, not evidence of absence.** Actions occur every ~7 h,
so a 12 h exclusion leaves controls only in a few quiet spells: 164 control
timestamps collapse to **16 contiguous stretches**, and timestamps hours apart
in the same spell are not independent observations.

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

## 6. What is needed next

**30 days is too few.** The method works and the ranking already yields signal,
but the case-control arm has no power, and Phase 3 (reactive vs anticipatory)
splits the 41 events further still.

Re-running Phases 1–2 over **180 days (~330 expected events)** changes nothing in
the method — only the power. That is in progress.

Open questions the longer window should settle:

1. Does the case-control comparison reach significance, or is the effect
   genuinely absent?
2. Do thermal indicators lead once there are enough events to see them?
3. Are trims and blanks triggered by *different* observations?
4. Is the up/down asymmetry explained by different triggers, or the same trigger
   treated asymmetrically?

## 7. Ask for the plant

`BF2 Coal rate actual value` implies a plant-side **set value** sibling tag that
is not ingested anywhere. Requesting it — and any coke-rate equivalent — would
give the PCI action signal directly. PCI is the co-control operators trade
against coke, so its action signal would roughly double what can be explained.
