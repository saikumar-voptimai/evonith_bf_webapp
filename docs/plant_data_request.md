# BF2 — Data Request to Plant Team

**Purpose:** six items needed to complete and validate the BF2 energy balance
model. Each is listed with why it is needed, exactly what is wanted, and the
ideal format.

**Prepared:** August 2026
**Reference:** `docs/energy_balance_findings_and_open_decisions.md`

---

## Please read first

Two of these requests are not really "please send data". They are **possible
instrument problems we have found**, which affect the plant's own reported
numbers, not only our model:

- **The top gas analyser may be under-reading CO+CO₂ by about 3 percentage
  points.** If so, the plant's own ETA CO and gas utilisation figures are also
  affected, since both are computed from the same analysis.
- **Shell heat loss cannot presently be established within a factor of three**
  (298 vs 886 MJ/tHM), because per-circuit cooling return temperatures are not
  logged.

We would suggest framing the email around these two, and treating the remaining
four as routine.

---

## Summary — for triage

| # | What we need | Priority | Effort at plant end |
|---|---|---|---|
| 1 | Top gas analyser calibration & service record | **High** | Low — pull existing log |
| 2 | Per-circuit cooling water inlet/outlet temperature | **High** | Medium — may need instrumentation |
| 3 | Carbon content of flue dust and GCP dust | **High** | Low — 2 lab samples |
| 4 | Clarification: are flue and GCP dust separate streams? | Medium | Very low — one answer |
| 5 | Cold blast flow meter calibration record | Medium | Low — pull existing record |
| 6 | Operating limits: k-value, RAFT, blast temperature, O₂ | Medium | Low — from operating standard |

---

# Request 1 — Top gas analyser calibration and service record

### Why we need it

We can calculate the top gas volume in two completely independent ways, and they
must agree because they describe the same gas:

- **Nitrogen route** — nitrogen is inert, so whatever enters with the blast
  leaves at the top.
- **Carbon route** — every kilogram of carbon burnt leaves as CO or CO₂.

Neither uses any fitted or assumed constant. Over 221 days they do **not** agree:

| Route | Median top gas volume |
|---|---|
| Nitrogen | 1,632 Nm³/tHM |
| Carbon | 1,867 Nm³/tHM |

A 14% disagreement. Both calculations use the top gas analysis, and an
under-reading of CO+CO₂ would push the two apart in exactly this way — it
inflates the nitrogen figure (lowering one estimate) while shrinking the carbon
divisor (raising the other). One fault explains both.

To reconcile them, CO+CO₂ would have to read about **3 percentage points
higher** than it does (45.6% against the measured 42.4%).

**What makes us fairly confident it is the instrument:**

| Quarter | CO+CO₂ measured | Shortfall | CO₂/(CO+CO₂) |
|---|---|---|---|
| 2025 Q4 | 41.80% | 4.00 pts | 42.87% |
| 2026 Q1 | 42.62% | 3.29 pts | 42.69% |
| 2026 Q2 | 42.25% | 3.10 pts | 41.97% |
| 2026 Q3 | 43.78% | 0.96 pts | 42.22% |

The **ratio CO₂/(CO+CO₂) stays flat at 42–43%** while the **sum drifts**. A real
change in furnace gas utilisation would move both. A calibration or span error
moves only the sum. And the shortfall is steadily reducing, which reads like an
analyser gradually coming back into calibration — possibly after a service.

> **This matters to the plant independently of our model.** ETA CO is computed
> from the same analysis. If CO+CO₂ is reading low, reported gas utilisation is
> affected too.

### What we need

1. Calibration and service log for the BF2 top gas analyser, **November 2025 to
   date**. Specifically the dates of:
   - zero and span calibration
   - calibration gas cylinder changes
   - cell / detector replacement
   - sample line cleaning, filter or dryer changes
   - any repair or re-commissioning
2. Calibration gas certificates — certified composition and date.
3. Confirmation of two things about how the analysis is reported:
   - Is it on a **dry basis** or wet basis?
   - Is **CH₄ or any other hydrocarbon** measured, or is the balance simply
     taken as N₂?
4. Analyser make, model and measurement principle (NDIR, TCD, etc.).

### Ideal format

- Service log as a simple table: **date | action taken | done by**. An export
  from the maintenance system is perfectly fine — it need not be tidied.
- Calibration certificates as PDF or scan.
- Items 3 and 4 can simply be answered in the reply email.

---

# Request 2 — Per-circuit cooling water inlet and outlet temperature

### Why we need it

Shell heat loss is one of the larger output terms in the balance, and at present
we cannot pin it down within a factor of three.

We have good data for stave rows 6 to 10, which gives **298 MJ/tHM**. But rows 6
to 10 cover only bosh, belly and lower stack. Hearth, bottom, tuyere nose and
upper shaft are not included. To cover them we currently scale up by cooling
water **flow share**, which gives **886 MJ/tHM**.

That scale-up assumes **every cooling circuit runs the same temperature rise**,
which is almost certainly not true — a hearth circuit and an upper shaft circuit
should behave quite differently.

The 590 MJ/tHM difference is not academic. It decides which of two validations
of the model is correct:

| Shell loss used | Energy balance closure | Model's coke rate vs actual |
|---|---|---|
| 298 MJ/tHM | 0.967 | 339.7 vs 337.2 kg/tHM — **0.7% error** |
| 886 MJ/tHM | 1.004 | 375.2 vs 337.2 kg/tHM — 11.3% error |

Both cannot be right. A blast furnace would typically lose 200–400 MJ/tHM
through the shell, which favours the lower figure — but that leaves the energy
balance 3% short of closing.

**What we already have from the existing tags:** cooling water **flow** for each
circuit (hearth, bottom, bosh & belly, lower stack, upper shaft, tuyere nose),
and a **common mains temperature**.

**What is missing:** the **return (outlet) temperature for each circuit
individually.** Without it we cannot compute heat pick-up per circuit.

### What we need

For each of the six circuits — **hearth, bottom, bosh & belly, lower stack,
upper shaft, tuyere nose**:

| Parameter | Unit |
|---|---|
| Cooling water flow | m³/hr |
| Inlet temperature | °C |
| **Outlet / return temperature** | °C |

### Ideal format

**Best case:** if return temperatures are already instrumented but simply not
tagged into the historian, please let us know the tag names — we can pull them
ourselves. Hourly data for any representative month would be plenty.

**If not instrumented:** a **manual reading set** would still settle it. Please
take inlet and outlet temperature plus flow for all six circuits **at the same
time**, on a normal operating day, and repeat perhaps three or four times across
different shifts. A simple table is fine:

```
Date | Time | Circuit | Flow m³/hr | Inlet °C | Outlet °C
```

Even a single good simultaneous set across all six circuits would let us replace
the flow-share assumption with a measured split.

---

# Request 3 — Carbon content of flue dust and GCP dust

### Why we need it

Carbon that leaves the furnace in dust is charged but never burns. It must be
deducted from the fuel input, exactly like the carbon that dissolves into the
hot metal.

Currently we **assume** the carbon content, because it is not measured anywhere:

| Stream | Assumed carbon | Basis |
|---|---|---|
| Flue dust (dust catcher) | 30% | Literature only |
| GCP dust | 20% | Literature only |

At the recorded dust quantities this is about **7.6 kg C/tHM**, or 1.6% of
carbon charged. These two numbers are the **weakest assumptions in the entire
carbon balance**, and a single lab analysis of each would remove them
permanently.

### What we need

One representative sample each of **dust catcher dust** and **GCP dust**,
analysed for:

| Parameter | Priority |
|---|---|
| **Total carbon %** (or fixed carbon + volatile matter) | Essential |
| Total iron % | Useful |
| Moisture % | Useful |
| Ash % or LOI | Useful |

### Ideal format

- Standard lab report, whatever format the lab normally issues.
- **Please state the basis clearly** — as-received or dry. This matters, and is
  the most common source of confusion in this kind of number.
- If it is convenient, **two or three samples of each taken on different weeks**
  would let us see how much it varies. One of each is still a large improvement
  on what we have.

---

# Request 4 — Are flue dust and GCP dust separate streams?

### Why we need it

The daily production report records four dust streams:

| Stream | Median |
|---|---|
| Flue dust | 33.7 t/day |
| GCP dust | 33.5 t/day |
| Cast house dust | 8.2 t/day |
| Stock house dust | 8.3 t/day |

Flue and GCP together come to about **29 kg/tHM**, which is somewhat above the
usual 10–25 kg/tHM for a blast furnace. Before we treat both as furnace
carry-over, we would like to be sure they are not partly the same material
counted twice.

We are currently treating them as follows, and would like this confirmed or
corrected:

| Stream | Our treatment |
|---|---|
| Flue dust | Leaves through the top — **counted** in the carbon balance |
| GCP dust | Leaves through the top — **counted** |
| Cast house dust | Leaves at the cast house — **not counted** |
| Stock house dust | Handling loss before charging — **not counted** |

### What we need

1. Are flue dust and GCP dust collected at **physically separate points**, or
   does the GCP figure include material already weighed at the dust catcher?
2. At what point is each stream weighed?
3. Is stock house dust a loss occurring **before** the material is weighed for
   charging, or after? If after, our charged tonnages may be slightly overstated.

### Ideal format

A few sentences in reply is sufficient. If a simple gas-cleaning flow sketch
exists, that would answer all three at once.

---

# Request 5 — Cold blast flow meter calibration record

### Why we need it

This is the **alternative explanation** for the discrepancy in Request 1, and we
would like to eliminate it properly rather than assume.

A cold blast flow meter reading about **9% low** would produce the same symptom —
it would depress the nitrogen-route top gas volume while leaving the
carbon-route figure untouched.

We think the analyser is the more likely cause, because the flat CO₂/(CO+CO₂)
ratio is something a flow meter cannot produce. But if Request 1 comes back
showing the analyser is in good order, then this becomes the leading candidate.

### What we need

1. Last calibration date of the BF2 cold blast flow meter, and the method used.
2. Meter type — orifice, venturi, annubar, ultrasonic, etc.
3. Confirmation that the reading is **compensated for temperature and pressure**,
   and whether the reported value is at normal conditions (Nm³/hr) or actual.
4. Whether oxygen enrichment is injected **upstream or downstream** of this
   meter — that is, does the reported blast volume already include the enrichment
   oxygen, or not?

> Item 4 matters on its own account. Our calculation currently assumes the
> enrichment oxygen **is** included in the measured cold blast volume. If it is
> injected downstream of the meter, that assumption is wrong and we will correct
> it.

### Ideal format

Answers in the reply email, plus the meter datasheet if readily available.

---

# Request 6 — Operating limits for the recommendation system

### Why we need it

The optimiser recommends blast parameters that minimise fuel cost. In a pure
energy balance, **higher blast temperature and higher oxygen enrichment always
help** — so with nothing to push back, the optimiser simply recommends
increasing both until it hits whatever step limit we impose.

That step limit is currently the only safety discipline in the system, and it is
a software constraint, not a furnace one.

We attempted to derive real limits from the historical data and **could not**.
The reason is a good one: the plant controls these parameters tightly. The
k-value sits between 69 and 88 for 90% of the time, and RAFT between 2,197 and
2,347 °C. The record therefore contains almost no examples of the bad operating
regime that a limit is meant to prevent, so there is nothing for the data to
learn from.

The limits have to come from operating experience instead.

### What we need

From the operating standard, or from an experienced operator's judgement:

| Parameter | What we need | Also useful |
|---|---|---|
| **k-value (permeability)** | Maximum acceptable value | What happens above it — hanging, slipping, channeling? |
| **RAFT** | Maximum acceptable, and minimum | Which failure appears first — tuyere burnout, SiO volatilisation, coke degradation? |
| **Hot blast temperature** | Maximum | Stove limitation or furnace limitation? |
| **Oxygen enrichment** | Maximum % | What limits it |
| **Blast volume** | Maximum and minimum | |
| **Top pressure** | Normal operating range | |

### Ideal format

A simple table of parameter, maximum, minimum, and a one-line note on what
happens if exceeded:

```
Parameter | Max | Min | What happens if exceeded
```

If different limits apply under different conditions — say a lower RAFT ceiling
when coke quality is poor — please do mention it. That kind of conditional rule
is exactly what the model cannot infer on its own, and it is straightforward for
us to encode once stated.

---

## What we are NOT asking for

To avoid anyone spending effort unnecessarily:

- **PCI ultimate analysis (C/H/O/N/S) is no longer required.** We understand it
  is not done at the plant and not available from the vendor. We have established
  the hydrogen content from the coal's rank instead, using the proximate analysis
  already reported, and have since confirmed the hydrogen term is not needed. No
  action required on this.
- No additional routine reporting is being requested. Items 1, 3, 4, 5 and 6 are
  one-time. Only Item 2 might justify a permanent tag, and only if the return
  temperatures turn out not to be instrumented today.

---

## What we will send back

So the request is not one-directional:

1. If Request 1 confirms an analyser drift, we will supply the **corrected
   historical ETA CO and gas utilisation** figures for the affected period.
2. Once Request 2 is answered, we will provide a **measured per-circuit heat
   loss breakdown** rather than a flow-share estimate — useful for cooling
   system monitoring in its own right.
3. Once Request 6 is answered, the recommendation system will carry **real
   furnace limits** instead of arbitrary step caps, which makes its output
   directly actionable by operators.
