# BF2 Energy Balance — Findings and Open Decisions

**Period covered:** work through 2026-08-22
**Branch:** `featre/232_energybal_recommendations`
**Status:** energy balance built and validated; three plant-side questions open

---

## Why this document exists

The energy balance work turned up several things that are **not** visible in the
code, and a few that contradict what the code appeared to say earlier. Some are
plant findings that need action outside the software. One is an open decision
that has been deliberately left open.

Anyone picking this up later should read §2 and §6 first. Everything else is
supporting detail.

---

## 1. What was built

| Layer | Module | What it answers |
|---|---|---|
| 1 | `utils/bmo/lp_solver.py` | Cheapest ore blend meeting six slag limits |
| 1b | `utils/bmo/transition.py` | How to get there from today's blend, in steps |
| 2 | `utils/bmo/process_recommendation.py` | Control settings that supply that blend's energy |
| — | `utils/energy_balance/` | The closed balance underneath Layer 2 |
| — | `utils/energy_balance/assumptions.py` | Operator input for every unmeasured constant |

### The energy balance

Measured over **221 days** (2025-11-07 → 2026-08-11):

- closure median **1.002**
- across-quarter spread **3.2%**
- within-quarter CV **2.1–2.8%**

For comparison, a fitted response surface on the same data ranged from
R² −1.43 to +0.68 across the same quarters. That contrast is the argument for a
physics-anchored approach over a data-driven one on this plant's record, and it
is measured rather than asserted.

**Caveat, added later:** the 1.002 figure depends on which shell loss is fed in.
See §5.

### Layer 2 — what it will and will not touch

Only three of the seven controls appear in an energy balance:

- **Optimised:** blast temperature, oxygen enrichment, blast volume (plus PCI when released)
- **Pass-through:** hot blast pressure, top pressure, steam — these act through
  permeability and gas utilisation, not through heat. Optimising them here would
  be fabricating a recommendation.

Derived blast-temperature coefficient: **−9.93 kg coke per 100 °C**, against
−10.0 already in `setting_bmo.yml`. Derived independently, agreeing to 0.7%.

PCI substitution from the energy balance is 0.86 (pure carbon equivalence); the
plant uses **0.53** and should continue to — coke also holds the burden column
open, which no energy balance can see.

**RAFT is advisory only.** Forward R² is 0.11 with an unattributed seasonal bias
up to +46 °C. It is displayed with that uncertainty and never blocks a
recommendation. An earlier expectation that it would fit above R² 0.95 was
wrong.

### Transition ladder

Re-solves the LP under a per-ore move cap, anchoring each rung to the previous
one. Every rung is a genuine LP solve, so each independently satisfies all six
slag limits — no step on the path is one the furnace could not run.

It also reports the **binding limit per rung** (including ore share caps, not
just slag chemistry) and detects when the *current* blend is already out of
bounds, which is a different problem from "the optimum is far away".

---

## 2. Headline findings

### 2.1 The top gas analyser appears to under-read CO+CO₂ by ~3 percentage points

**This is the most actionable finding in the whole exercise.**

Top-gas volume is not measured — it is inferred. It can be inferred two
independent ways, neither using a fitted constant, and they must agree:

```
NITROGEN   N₂ is inert       V = V_blast × N₂%_blast / N₂%_top
CARBON     all C leaves as
           CO or CO₂         V = (C_burnt/12.011) × 22.414 / ((CO%+CO₂%)/100)
```

Over 221 days:

| Basis | Median | p5 | p95 |
|---|---|---|---|
| Nitrogen (what the model uses) | 1,632 Nm³/tHM | 1,550 | 1,787 |
| Carbon | 1,867 Nm³/tHM | 1,790 | 1,986 |

Ratio median **1.140**. Gap **+228 Nm³/tHM ≈ +762 MJ/tHM**.

This is not a bias to shrug at:

- it tracks the balance's residual day by day, **r = +0.774**
- removing it collapses the across-quarter drift in back-calculated shell loss
  from **712 → 165 MJ/tHM**
- residual standard deviation falls from 445 → 285 MJ/tHM

**Why the analyser and not the model.** Both volumes are computed *from* the gas
analysis, and an under-read of CO+CO₂ moves them in **opposite** directions:
N₂_top = 100 − CO − CO₂ − H₂ is inflated so V_nitrogen falls, while V_carbon's
divisor shrinks so it rises. One fault, two opposite signs — exactly the pattern
observed.

Reconciling needs CO+CO₂ about **3 points higher** than measured (45.6 vs 42.4,
+7.3% relative).

**The confirming detail:**

| Quarter | CO+CO₂ measured | Reconciling | Shortfall | CO₂/(CO+CO₂) |
|---|---|---|---|---|
| 2025Q4 | 41.80% | 45.84% | 4.00 pts | 42.87% |
| 2026Q1 | 42.62% | 45.88% | 3.29 pts | 42.69% |
| 2026Q2 | 42.25% | 45.41% | 3.10 pts | 41.97% |
| 2026Q3 | 43.78% | 44.81% | 0.96 pts | 42.22% |

**The ratio holds flat while the sum drifts.** A genuine change in gas
utilisation would move both. A span error moves only the sum. And the shortfall
shrinks steadily — an instrument coming back into calibration, not anything the
furnace did.

**Alternatives checked and failed on magnitude:**

- *Dust:* would need 52 kg C/tHM leaving unburnt, 12% of carbon charged, against
  a measured 7.6 (see §3)
- *Blast composition:* would need 85.3% N₂ in the blast; air is 79.2%
- *Carbon fractions:* PCI is set at 0.75 where rank implies 0.79 — raising it
  **widens** the gap

**One alternative that is NOT ruled out:** a blast flow meter reading ~9% low
would depress V_nitrogen with no effect on V_carbon — the same signature. What
still favours the analyser is the flat CO₂/(CO+CO₂) ratio, which a flow meter
cannot produce. But it is a second candidate.

**Reported, not corrected.** `compute()` returns the carbon-basis volume, the
ratio, and a `gas_analysis_suspect` flag that trips beyond 5%. Rescaling plant
measurements in code to flatter our own closure would bury an instrument fault
that someone can actually go and fix — silently, on every future day.

> **Plant action:** check the top gas analyser service record against these
> dates. Do not trust pre-2026Q3 closure figures until this is settled.

### 2.2 Fuel hydrogen — H% settled, term ruled out

The hydrogen term had been gated for months on "no ultimate analysis exists".
The vendor does not supply one either, so that block would never have cleared.

**It did not need to.** Proximate analysis is enough to fix the coal's *rank*,
and rank determines hydrogen closely enough for a term of this size.

Plant PCI, median of 966 samples: **VM 20.03, ash 9.18, moisture 1.45,
FC 69.68**. On a dry ash-free basis VM = 22.4%, which is squarely
**medium-volatile bituminous** (22–31% VM dmmf), running H = 4.5–4.9% daf →
**4.2% as charged**.

> Note: this is **imported-grade PCI, not indigenous Indian coal.** At 9.2% ash
> it is nowhere near the 35–50% of domestic coal, so Indian indigenous coal
> analyses are the wrong reference class and were deliberately not used.

Coke: VM 0.94, FC 87.4 → **0.35% H**.

**The old fallback was wrong and was doing damage.** With `hydrogen_pct` null,
H% fell back to VM × 0.25, returning 5.0% for PCI at VM 20 — above any real
medium-volatile coal. The docstring called this "the standard correlation"; it
is not. **That overstatement produced the often-quoted 0.910 closure** and made
the hydrogen term look less trustworthy than it deserved.

**With H% settled, the real question became answerable — and the answer is no.**

The falsifying test: if fuel hydrogen were the missing input, back-calculated
shell loss must **fall** as PCI rate rises, with slope −120 × H_fraction. That
would also have handed us H% from the plant's own record for free.

Over 221 days and PCI 152–205 kg/tHM: **correlation −0.05.** Regression returns
a *positive* +16.8 MJ per kg PCI, implying negative hydrogen content. Refuted,
not merely unsupported.

| H_pci | Closure | Implied shell | Quarter drift |
|---|---|---|---|
| **off** | **1.002** | **597 MJ/tHM** | 712 MJ/tHM |
| 3.0% | 0.953 | 1,420 | 668 |
| 4.2% | 0.937 | 1,680 | 652 |
| 5.0% | 0.927 | 1,847 | 631 |

Measured shell loss is 195–550 MJ/tHM. Every H% drives the residual further from
it, and none meaningfully reduces the drift hydrogen was supposed to explain.

**What it did find:** the residual scales with **total fuel, not hydrogen**.
Coke, nut coke and PCI all carry positive coefficients of +17 to +29 MJ/kg
against a carbon credit of 28.5 MJ/kg coke — only about a third of marginal fuel
energy reaches a modelled output. That pointed directly at §2.1. The two fuels
are collinear (−0.85), so those coefficients are read jointly, not individually.

**Decision: `include_fuel_hydrogen` stays `false`.** Not blocked on data —
tested and ruled out. Re-run `scripts/pci_hydrogen_from_closure.py` after the
analyser question is settled.

---

## 3. Dust — real, measurable, and was entirely missing

`dpr_data` records four dust streams and the balance used **none** of them.

| Stream | Median | Leaves via top? |
|---|---|---|
| `flue_dust_mt` (dust catcher, coarse) | 33.67 t/d | **yes** |
| `gcp_dust_mt` (gas cleaning plant, fine) | 33.49 t/d | **yes** |
| `cast_house_dust_mt` (tapping fume) | 8.22 t/d | no |
| `stock_house_dust_mt` (handling loss, pre-charging) | 8.30 t/d | no |

Carbon in top-leaving dust is charged but **never burnt** — the same category of
error as crediting carbon dissolved in hot metal, which once put closure at
0.47. It now comes off the input side.

On the reference day: 29.1 kg dust/tHM, giving **7.6 kg C/tHM (1.6% of carbon
charged)** at assumed carbon contents.

**Dust was dismissed too quickly at first.** The earlier argument was that the
top-gas gap needed 52 kg C/tHM against "a realistic 3–10", so dust could not
explain it. That ruling-out was sound — the real figure is 7.6 — but the term is
genuine and was simply absent. It closes **19%** of the top-gas volume gap
(1,832 → 1,799 Nm³/tHM against nitrogen's 1,658).

**And it makes the cancellation visible.** Removing dust carbon *raises* closure
from 1.002 to 1.018 — pushing the error the **opposite** way from the top-gas
gap. Today's near-perfect closure is two errors cancelling, not a balance that is
right. Anyone tuning against closure alone would be tuning against a coincidence.

> **Plant action:** dust carbon content is assumed (flue 30%, GCP 20%), not
> measured. **One lab analysis of each dust stream would remove the weakest
> assumption in the carbon balance.** Sensitivity across 20–40% is shown in the
> day audit; the conclusion holds across the range.

Also worth confirming: 29.1 kg/tHM is above the usual 10–25. Are flue dust and
GCP dust genuinely separate streams, or the same material weighed twice?

---

## 4. Reference day worked numbers — 2025-12-31

Chosen because its closure sits closest to the population median. Reproduce with
`python scripts/energy_balance_day_audit.py 2025-12-31`.

### As recorded

| | | | |
|---|---|---|---|
| Hot metal | 2,275.3 t | Blast volume | 111,392 Nm³/hr |
| Slag | 756.3 t | Blast temperature | 1,200 °C |
| Coke | 767.2 t | O₂ enrichment | 3.81% |
| Nut coke | 171.5 t | Top gas CO | 24.95% |
| PCI | 342.6 t | Top gas CO₂ | 18.75% |
| Ore / Sinter / Pellet | 1,337 / 2,532 / 3.1 t | Top gas H₂ | 2.87% |
| Flux | 13.1 t | Top gas temperature | 137.3 °C |
| Flue dust | 41.4 t | HM carbon | 4.28% |
| GCP dust | 24.8 t | HM iron | 94.96% |

### Rates

Coke 337.2 · Nut coke 75.4 · PCI 150.6 · Slag 332.4 · Flux 5.8 kg/tHM ·
Blast 1,175 Nm³/tHM · **Total fuel 563.1 kg/tHM** (plant target ~560–580) ·
Top-gas dust 29.1 kg/tHM

### Carbon balance

```
Coke      337.2 × 0.87 = 293.4 kg C/tHM
Nut coke   75.4 × 0.87 =  65.6
PCI       150.6 × 0.75 = 112.9      ← rank implies ~0.79
CARBON CHARGED         = 471.9

less into hot metal    =  -42.8     (4.28% of 1000 kg)
less leaving in dust   =   -7.6     (flue @30%C, GCP @20%C)
CARBON BURNT           = 421.4
```

### Energy

| Input | MJ/tHM | Output | MJ/tHM |
|---|---|---|---|
| Carbon | 14,072 | Iron oxide reduction | 7,008 |
| Blast sensible | 1,933 | Top gas chemical (CO + H₂) | 5,739 |
| | | Hot metal @ 1500 °C | 1,378 |
| | | Shell loss (flow-scaled) | 886 |
| | | Slag | 598 |
| | | Top gas sensible | 257 |
| | | Si / burden moisture / Mn / FeO | 171 |
| **Total** | **16,005** | **Total** | **16,038** |

```
closure, as modelled today       16,038 / 16,005 = 1.002
closure, dust carbon removed     16,038 / 15,754 = 1.018
```

### Top gas, two ways

```
NITROGEN  1,175 × 75.39 / 53.42            = 1,658 Nm³/tHM
CARBON    421.4/12.011 × 22.414 / 0.437    = 1,799 Nm³/tHM   ratio 1.085
```

---

## 5. OPEN DECISION — which shell loss?

**Deferred by the user, 2026-08-22. Recorded here so it is not lost.**

The reference-day fixture feeds `shell_loss_gj_per_hr` the **stave rows 6–10**
figure (298 MJ/tHM). The documented worked example uses the **flow-scaled
all-circuits** figure (886 MJ/tHM). They have never agreed, and a ±0.05
tolerance on the closure test was wide enough to hide the 590 MJ/tHM difference.

It is not resolved because **the two strongest validations want opposite
answers:**

| Shell loss | Closure | Solved coke rate | vs measured 337.2 |
|---|---|---|---|
| 298 MJ/tHM (stave rows 6–10) | 0.967 | 339.7 | **+0.7%** |
| 886 MJ/tHM (flow-scaled) | 1.004 | 375.2 | +11.3% |

Population medians: stave 195, flow-scaled 550 MJ/tHM. Physical expectation for
a BF is **200–400 MJ/tHM**, which favours the stave figure — but 298 leaves
closure 3% short.

The flow-scaled figure assumes **every cooling circuit runs the same temperature
rise**. That is its weakest link, and it is measurable.

**Currently pinned at the stave value** so the solver validation stands, with the
conflict written into `test_balance_closes_on_the_reference_day` rather than
tidied away.

> **What would settle it:** per-circuit ΔT for hearth, bottom, tuyere nose and
> upper shaft. This is plant data that already exists in principle.

**This blocks:** trusting any single closure figure, and therefore the accuracy
of energy-balance-driven fuel estimation in BMO. It does not block the structure
of that integration.

---

## 6. Open items and plant actions

| # | Item | Blocks | Owner |
|---|---|---|---|
| 1 | **Top gas analyser service record** — check against 2025Q4–2026Q3 | Trusting pre-2026Q3 closure | Plant |
| 2 | **Dust carbon analysis** — one sample each of flue and GCP dust | Weakest assumption in carbon balance | Plant lab |
| 3 | **Per-circuit cooling ΔT** — settles §5 | Which shell loss is correct | Plant |
| 4 | Confirm flue vs GCP dust are not double-counted | Dust tonnage credibility | Plant |
| 5 | Blast flow meter check — *only if* #1 comes back clean | Alternative cause for §2.1 | Plant |
| 6 | k-value / RAFT hard limits from operating standard | Layer 2 has no permeability guard | Plant |

### Not derivable from data — do not retry without new information

- **Permeability ceiling.** Correlating k against the channeling score over 120
  days gives Spearman **+0.04**. k sits in 69–88 for 90% of windows because the
  plant already controls it tightly, so the record contains almost no examples
  of the bad regime a ceiling is meant to prevent. RAFT behaves identically
  (2,197–2,347, corr −0.04). Kept as
  `scripts/permeability_limit_from_channeling.py` so nobody repeats it.
- Until a plant operating standard supplies these, **move limits are the
  discipline** in Layer 2.

### Separate defects found in the channeling detector (not fixed)

Found while investigating the above, in `utils/anomaly_propensity.py`:

1. The composite is documented as clipping each signal to [0,1] before
   averaging, but `_compute_heatload` returns an unbounded `diag_sum` that
   reached **296**. The composite hits **42.45** on a nominal 0–1 scale and is
   effectively just heat load.
2. Its default 10-minute window is shorter than the 15-minute data interval, so
   most windows hold at most one sample: **34% NaN, p95 only 0.243** — it never
   reaches the operator's stated 0.6–0.7 tolerance band at all.

This is a live detector on the AI Copilot page and deserves its own change.

---

## 7. Corrections made during this work

Recorded because each one changed a conclusion, and because two of them are the
kind of error that recurs.

| Error | Impact | Status |
|---|---|---|
| Omitted iron oxide reduction | Closure sat at 0.47 — it is ~45% of output | Fixed, pinned by test |
| Credited carbon dissolved in HM as burnt | Over-stated input ~1,400 MJ/tHM (4.3% C = 43 kg C/tHM). **Worth grepping for elsewhere — biases any coke figure by ~4%** | Fixed, pinned by test |
| Heat load unit: "GW·hr → GJ" ×3600 | **1000× too large.** Tags read MW; ×3.6 gives GJ/hr (median 16.8, matching operator expectation of 15–20) | Fixed |
| Predicted RAFT would fit at R² > 0.95 | Forward R² is 0.11 with +46 °C bias | RAFT demoted to advisory |
| Claimed hbp +0.206 / eta_co +0.211 showed signal | Those were **unclipped outlier artefacts**; clipped to [0,1] the correlation collapses to +0.04 | Retracted |
| Dismissed dust without checking DPR | Dust data existed all along; term was genuinely missing | Fixed, §3 |
| Attributed §2.1 to the gas analyser with too much certainty | A blast flow meter ~9% low gives the same signature | Caveat added |
| Factor-of-100 slip in reconciliation algebra | Produced a nonsense 0.86% CO+CO₂ | Caught and fixed before use |

---

## 8. Operator input table

`utils/energy_balance/assumptions.py` collects **every number the plant has not
measured** into one registry, surfaced in the Blend Optimizer page under
*"Plant assumptions — operator input"*. Ten parameters, ordered by impact:

| Parameter | Default | Unit | Confidence |
|---|---|---|---|
| Carbon in flue dust | 30.0 | % | assumed |
| Carbon in GCP dust | 20.0 | % | assumed |
| Carbon fraction of PCI | 0.75 | fraction | assumed |
| Carbon fraction of coke | 0.87 | fraction | literature |
| Hydrogen in PCI | 4.2 | % | literature |
| Hydrogen in coke | 0.35 | % | literature |
| Moisture in the blast | 15.0 | g/Nm³ | assumed |
| Hot metal enthalpy at tap | 1,378 | MJ/t | measured |
| Slag enthalpy | 1.80 | MJ/kg | literature |
| Burden moisture, heat to evaporate | 2.70 | MJ/kg H₂O | literature |

Each carries its basis, what changes if it is wrong, and physical bounds.
Values persist between sessions and are applied at the app boundary, so the math
layer stays pure and tests stay hermetic.

**Physics is deliberately excluded** — reduction enthalpy, calorific values,
molar volume. A test asserts none can ever appear there.

---

## 9. Scripts

| Script | Purpose |
|---|---|
| `energy_balance_phase0.py` | Builds the 221-day dataset; the gate that validated closure |
| `energy_balance_worked_example.py` | **Generates** `energy_balance_calculation_procedure.md` — the doc cannot drift from code |
| `energy_balance_day_audit.py` | Full single-day audit, every step shown, dust included |
| `pci_hydrogen_from_closure.py` | The hydrogen falsification test (§2.2) |
| `topgas_carbon_vs_nitrogen.py` | The two-volume test (§2.1) |
| `permeability_limit_from_channeling.py` | Negative result — kept so it is not repeated |

---

## 10. Commit index

| Commit | Subject |
|---|---|
| `c7ac1e2` | feat(energy): closed blast-furnace energy balance |
| `4776299` | feat(bmo): recommend control parameters for a chosen blend |
| `737a29a` | chore(analysis): k-value ceiling from channeling — negative result |
| `4ca502c` | feat(bmo): step-by-step path from the current blend to the optimum |
| `2d4410d` | feat(bmo): surface the process recommendation and transition path |
| `cbf64e3` | fix(energy): pin fuel H% from coal rank, and rule the term out on evidence |
| `d8e24f1` | fix(energy): trace the balance drift to the top gas analyser |
| `ff4702c` | feat(energy): single-day audit script, and bring dust into the carbon balance |
| `7b5b465` | feat(energy): operator input table for every unmeasured constant |

---

## 11. Next phase — BMO integration

Planned by the user, not yet started:

1. Integrate the energy balance into BMO
2. Support fuel requirement estimation from it
3. Cost prediction
4. Recommendations

**Two things to carry into that work:**

- The headline closure of 1.002 depends on which shell loss is fed in (§5). The
  structure of the integration does not depend on resolving it; the accuracy
  does.
- Closure near 1.00 is currently **two errors cancelling** — an understated
  top-gas output against an over-credited carbon input. Do not tune against
  closure alone.
