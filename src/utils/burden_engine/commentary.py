"""Context for the furnace commentary on the Burden & Fuel Optimiser page (page 11).

The commentary model is given, in this order:

1. the optimiser notes (``src/assets/burden_engine/optimiser_notes.md``): what the page's numbers are, how they were
   made and validated, and where they are weak;
2. the fact register as a table (every coefficient outside the balance, bands, default limits, retired facts);
3. the live run: data sources and calibration, the targets and limits, the model check (energy model against burnt
   coke, burnt against charged coke), the plan in the page-9 grouping (cost, fuel, slag, charging, process), what
   binds, today's charged blend, the ore + flux-only answer, the value of limits and the frontier, and the yard.

Same rule as page 9 (``utils/bmo/commentary.py``): NO LLM IN HERE. Assembly is pure - dictionaries and frames in, one
string out - so "was the model given the right facts?" is a unit test, not a reading of generated prose. The model
call lives in ``ui/burden_optimiser/commentary.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from utils.burden_engine.facts import FactRegister
from utils.burden_engine.scenarios import binding_table, limit_label, limit_labels, plan_tiles
from utils.slag_bands import PLANT_SLAG_BANDS

NOTES_PATH = Path(__file__).resolve().parents[2] / "assets" / "burden_engine" / "optimiser_notes.md"

SYSTEM_PROMPT = """\
You are a senior blast furnace advisor writing a commentary for the operating team at Evonith Steel BF-2 on the \
Burden & Fuel Optimiser's result. They are practising metallurgists. Do not explain what a blast furnace is, what \
coke does or what basicity means.

WRITE IN INDIAN PROFESSIONAL ENGLISH - the register used in Indian plant engineering. Point-wise where it helps. \
Precise, not florid.

YOU ARE GIVEN: the optimiser notes (how every number is made, validated and where it is weak), the fact register as \
a table, and the live run. Treat the notes and the register as the authority on method; treat the live run as the \
facts of today.

RULES YOU MUST FOLLOW:

1. Use ONLY the figures supplied. Do not invent a number, a trend or a material. If something needed for a \
   judgement is missing, say plainly that it is not available rather than estimating it.
2. Every recommendation gets ACTION, REASON and MAGNITUDE ("move 4 pt from sinter to Lloyds CLO to save Rs 210/tHM; \
   coke rises 1.6 kg/tHM because ...").
3. Start from the model check. If the energy model is not on track, or the latest scored shift is stale, say what \
   that means for the plan before anything else. Quote the fact register ID when you lean on a coefficient.
4. If the plan breaks a limit, lead with it. A saving on an infeasible plan is not a saving.
5. Respect the stated uncertainties: shaft efficiency either held at the calibration or blend-dependent with the \
   plan's sigma(coke) and certainty class (High / Medium / Low) as stated; the day-to-day spread (about +/-7 kg/tHM \
   coke); burden-resistance rules extrapolated beyond their fitted range; process limits that are an operating \
   envelope and not equipment ratings. A saving inside the model's error band (smaller than one sigma of the saving) \
   is not a reason to change the burden - say so. A Low-certainty plan is outside what BF-2 has run: say so first.
6. Burnt and charged coke are the same coke a descent time apart; compare them over a day or more, never shift by \
   shift.

STRUCTURE - use these five headings exactly:

**Furnace and model today** - how the furnace has been running (latest shift, HMT, Si, production) and whether the \
energy model matches the burnt coke (verdict, bias, scatter, staleness). Two to four lines.

**What the optimiser proposes** - the blend and flux, the cost and the fuel effect, as changes against today's \
charged blend where it is supplied (else against the ore + flux-only answer), in Rs/tHM and Rs lakh/day.

**Why it makes sense** - the metallurgical argument: Fe units, gangue and slag (rate, CaO/SiO2, Al2O3, MgO against \
limits and the plant band), coke response, what binds and what that limit costs. If the argument is weak, say so.

**How to get there** - steps, size of each step, one descent time apart, and what to check between steps.

**What to watch** - the limit closest to its bound, extrapolated rules, data gaps, and the first thing to check if \
the furnace does not respond as expected.

Keep the whole thing under 550 words."""


@dataclass
class CommentaryContext:
    """Assembled context, and what could not be gathered."""

    text: str
    missing: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """A commentary with no plan is not worth asking for."""

        return bool(self.text.strip()) and "OPTIMISED PLAN" in self.text


def _f(value: Any, spec: str = ",.1f") -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(x) else format(x, spec)


def _section(title: str, lines: Sequence[str]) -> str:
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    return f"\n## {title}\n" + "\n".join(f"- {ln}" for ln in lines) + "\n"


def _cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return str(v).replace("|", "/").replace("\n", " ").strip()


def load_notes(path: str | Path | None = None) -> str:
    """The optimiser notes (static context). Empty string when the file is missing."""

    p = Path(path) if path else NOTES_PATH
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def facts_table(reg: FactRegister) -> str:
    """The fact register as a markdown table: every fact, retired ones included (marked), with its basis."""

    head = (f"Fact register: {reg.name} v{reg.version} ({reg.date}). Statuses: Model = follows from the balance; "
            "Supported = BF-02 data agree; Measured = from plant data; Assumed = literature or judgement; Retired = "
            "never used by the engine.\n\n"
            "| ID | Group | Statement | Value | Unit | Uncertainty | Status | Grade | Basis | Note |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for f in reg.facts():
        unc = f"± {f.se:g}" if f.se is not None else (f"{f.range[0]:g} to {f.range[1]:g}" if f.range else "")
        rows.append("| " + " | ".join(_cell(x) for x in (f.id, f.group, f.statement, "" if f.value is None else f"{f.value:g}",
                                                           f.unit, unc, f.status, f.grade, f.basis, f.note)) + " |")
    return head + "\n" + "\n".join(rows)


def describe_inputs(s: Mapping[str, Any], L: Mapping[str, Any], *, yard_source: str, calibration: Mapping[str, Any],
                    regime: Mapping[str, Any] | None, mode: str, target_t_day: float | None) -> str:
    cal = dict(calibration)
    reg = dict(regime or {})
    lines = [
        f"Mode: {mode}" + (f", target {_f(target_t_day, ',.0f')} t/d" if target_t_day else ""),
        "Yard: " + ("live chemistry (Neon lab tables, window means), stock and the saved prices" if yard_source == "live"
                    else "reference yard of 16 Oct 2026 (live yard not available)"),
        f"Calibration: {cal.get('source', 'n/a')}, {cal.get('window', '')}; shaft efficiency {_f(cal.get('omega'), '.3f')}, "
        f"losses {_f(cal.get('Q_LZ_GJh'), '.0f')} / {_f(cal.get('Q_UZ_GJh'), '.0f')} GJ/h (lower / upper)"
        + (f"; {cal['message']}" if cal.get("message") else ""),
        f"Thermal target: HMT {_f(s.get('HMT'), ',.0f')} °C, Si {_f(s.get('hm_Si'), '.2f')} %; PCI "
        f"{_f(s.get('pci_plant'), '.0f')} and nut coke {_f(s.get('nut_plant'), '.0f')} kg/tHM (fixed inputs); blast "
        f"moisture {_f(s.get('humidity_g'), '.0f')} g/Nm3",
        f"Controls: HBT {_f(s.get('HBT'), ',.0f')} °C, hot-blast pressure {_f(s.get('Pb'), '.2f')} bar g, top pressure "
        f"{_f(s.get('Pt'), '.2f')} bar g",
        f"Slag limits: rate <= {_f(L.get('slag_max'), '.0f')} kg/tHM, CaO/SiO2 {_f(L.get('B2_min'), '.2f')}-"
        f"{_f(L.get('B2_max'), '.2f')}, Al2O3 <= {_f(L.get('Al2O3_max'), '.1f')} %, MgO >= {_f(L.get('MgO_min'), '.1f')} %, "
        f"MgO/Al2O3 >= {_f(L.get('MgO_Al2O3_min'), '.3f')}",
        f"Charging: <= {_f(L.get('charges_per_h'), '.2f')} charges/h x {_f(L.get('charge_t'), '.2f')} t non-coke",
        (f"Process limits: blast <= {_f(L.get('V_max'), ',.0f')} Nm3/h, O2 <= {_f(L.get('O2_flow_max'), ',.0f')} Nm3/h, "
         f"enrichment tag <= {_f(L.get('enrich_max'), '.1f')} %, HBT <= {_f(L.get('HBT_max'), ',.0f')} °C, hot-blast "
         f"pressure <= {_f(L.get('Pb_max'), '.2f')} bar g, dP <= {_f(L.get('dP_max'), '.2f')} bar, RAFT "
         f"{_f(L.get('RAFT_min'), ',.0f')}-{_f(L.get('RAFT_max'), ',.0f')} °C") if L.get("process") else "",
        ("Today's operating point (14-day medians): " + ", ".join(
            f"{lbl} {_f(reg.get(k), fmt)}" for k, lbl, fmt in (
                ("prod_th", "production t/h", ".1f"), ("HBT", "HBT °C", ",.0f"), ("hot_blast_nm3h", "blast Nm3/h", ",.0f"),
                ("O2_nm3h", "O2 Nm3/h", ",.0f"), ("Pb", "hot-blast pressure bar g", ".2f"), ("Pt", "top pressure bar g", ".2f"),
                ("dP", "dP bar", ".2f"), ("HMT", "HMT °C", ",.0f"), ("hm_si", "Si %", ".2f"))
            if reg.get(k) is not None)) if reg else "",
    ]
    return _section("RUN SET-UP", lines)


def describe_model_check(shown: Mapping[str, Any] | None) -> str:
    """The model check as the page showed it (``render_model_check``'s return)."""

    if not shown or shown.get("latest") is None:
        return _section("MODEL CHECK", ["Not available: the Coke Energy pipeline's shift results were not reachable."])
    row = shown["latest"]
    e, o = shown.get("energy") or {}, shown.get("oxygen") or {}
    lines = [
        f"Latest scored shift {shown.get('latest_name', row.name)}"
        + (f", {shown['age_days']} days old" if shown.get("age_days") is not None else "")
        + f": burnt coke {_f(row.get('K_burnt'))}, energy balance {_f(row.get('K_energy'))}, oxygen balance "
          f"{_f(row.get('K_obal'))}, charged {_f(row.get('K_charged'))} kg/tHM dry; HMT {_f(row.get('HMT'), ',.0f')} °C, "
          f"Si {_f(row.get('hm_si'), '.2f')} %, {_f(row.get('prod_th'))} t/h, PCI {_f(row.get('pci_kgthm_dry'), '.0f')}, "
          f"nut coke {_f(row.get('nut_kgthm_dry'), '.0f')} kg/tHM dry",
        f"Energy model over the last {shown.get('period', '?')}: verdict {shown.get('state', '?').upper()} "
        f"({shown.get('reason', '')})",
        f"Oxygen balance over the same period: bias {_f(o.get('bias'), '+.1f')}, typical error +/-{_f(o.get('sd'))} kg/tHM, "
        f"{o.get('n', 0)} stable shifts" if o.get("n") else "",
        f"Energy model n={e.get('n', 0)}, MAE {_f(e.get('mae'))}, r {_f(e.get('r'), '.2f')}" if e.get("n") else "",
    ]
    return _section("MODEL CHECK: ENERGY MODEL AGAINST BURNT COKE", lines)


def describe_transit(S: Mapping[str, Any] | None) -> str:
    """Burnt against charged coke (``utils.coke_energy.transit.summary``)."""

    if not S:
        return ""
    t = S["table"]
    lines = [f"{r.window}: {int(r.n)} windows, bias {_f(r.bias, '+.1f')}, SD {_f(r.sd)}, 95 % band "
             f"{_f(r.loa_lo, '+.1f')} to {_f(r.loa_hi, '+.1f')} kg/tHM, r {_f(r.r, '.3f')}, CCC {_f(r.ccc, '.3f')}"
             for r in t.itertuples() if r.n and r.n >= 3]
    sp = S.get("split") or {}
    transit = sp.get("transit_sd")
    lines.append(f"Spread split: random part +/-{_f(sp.get('random_sd'))} per shift (averages out), "
                 + (f"transit part +/-{_f(transit)} per shift, " if transit and transit >= 0.5
                    else "transit part not separable from the random part at shift resolution, ")
                 + f"floor +/-{_f(sp.get('floor_sd'))} kg/tHM SD (does not average out)")
    scan = S["scan"].set_index("lag_shifts")["r"]
    lines.append("Correlation of shift-to-shift movements, burnt against charged coke L shifts earlier: "
                 + ", ".join(f"L={int(k):+d}: {_f(v, '.2f')}" for k, v in scan.items())
                 + f"; peak by interpolation {_f(S.get('peak_lag_h'), '.1f')} h (8-hour shifts cannot resolve the 6-7 h "
                   "descent)")
    clo = S.get("closure")
    if clo and clo.get("n"):
        lines.append(f"Mass closure, raw burnt coke over charged coke, {clo['n']} shifts: {_f(100 * clo['ratio'], '.1f')} %")
    return _section("BURNT COKE AGAINST CHARGED COKE (same coke, a descent time apart)", lines)


def describe_plan(r: Mapping[str, Any], label: str, names: Mapping[str, str], flux_ids: Sequence[str],
                  *, process: bool = True) -> str:
    """One plan in the page-9 grouping, plus process, binding limits and warnings."""

    if r is None:
        return ""
    P = plan_tiles(r, flux_ids)
    fer = sorted(((names.get(i, i), sh, r["wet_kg"].get(i, 0.0), r["t_day"].get(i, 0.0))
                  for i, sh in r["share_wet"].items() if sh > 0.05), key=lambda x: -x[1])
    flux = [(names.get(i, i), r["wet_kg"].get(i, 0.0)) for i in flux_ids if r["wet_kg"].get(i, 0.0) >= 0.5]
    c, f, s, g, o = P["cost"], P["fuel"], P["slag"], P["charging"], r["out"]
    bad = {k: v for k, v in r.get("viol", {}).items() if v > 1e-3}
    lines = [
        "Feasibility: " + ("meets every limit" if r.get("feasible", True) and not bad
                           else "BREAKS " + ", ".join(f"{limit_label(k, names)} (by {v:,.2f})" for k, v in bad.items())),
        f"Production {_f(r['P_tday'], ',.0f')} t/d ({_f(r['P_th'])} t/h)",
        "Ferrous blend (wet % of ferrous, kg/tHM wet, t/d): "
        + "; ".join(f"{n} {sh:.1f} %, {kg:,.0f} kg, {td:,.0f} t/d" for n, sh, kg, td in fer),
        "Flux: " + (", ".join(f"{n} {kg:.1f} kg/tHM" for n, kg in flux) if flux else "none"),
        f"Burden Fe {_f(P['overview']['fe_wet_pct'])} % wet; ferrous {_f(P['overview']['ferrous_kg'], ',.0f')} kg/tHM",
        f"Cost Rs/tHM: ore {_f(c['ore'], ',.0f')}, flux {_f(c['flux'], ',.0f')}, fuel {_f(c['fuel'], ',.0f')} (coke "
        f"{_f(c['coke'], ',.0f')}, nut coke {_f(c['nut'], ',.0f')}, PCI {_f(c['pci'], ',.0f')}), TOTAL {_f(c['total'], ',.0f')}",
        f"Fuel kg/tHM: coke {_f(f['coke_dry'])} dry ({_f(f['coke_wet'])} wet), nut coke {_f(f['nut'], '.0f')}, PCI "
        f"{_f(f['pci'], '.0f')}, total fuel {_f(f['total'], '.0f')} (band 520-600)",
        f"Slag: {_f(s['rate'], '.0f')} kg/tHM, {_f(s['t_day'], ',.0f')} t/d; Al2O3 {_f(s['Al2O3_pct'], '.2f')} %, MgO "
        f"{_f(s['MgO_pct'], '.2f')} %; CaO/SiO2 {_f(s['b2'], '.3f')} (plant band {PLANT_SLAG_BANDS['b2'][0]}-"
        f"{PLANT_SLAG_BANDS['b2'][1]}), T-basicity {_f(s['t_basicity'], '.3f')} (plant {PLANT_SLAG_BANDS['t_basicity'][0]}-"
        f"{PLANT_SLAG_BANDS['t_basicity'][1]}), IB4 {_f(s['ib4'], '.3f')} (plant {PLANT_SLAG_BANDS['ib4'][0]}-"
        f"{PLANT_SLAG_BANDS['ib4'][1]}), MgO/Al2O3 {_f(s['mgo_al2o3'], '.3f')}",
        f"Charging: coke {_f(g['coke_t'], ',.0f')} t/d, nut coke {_f(g['nut_t'], ',.0f')} t/d, PCI {_f(g['pci_t'], ',.0f')} "
        f"t/d, ferrous {_f(g['ferrous_t'], ',.0f')} t/d, {_f(g['charges_per_h'], '.2f')} charges/h, hot metal per charge "
        f"{_f(g['hm_per_charge'], '.3f')} t",
    ]
    if process:
        lines.append(
            f"Process: HBT {_f(r['settings'].get('HBT'), ',.0f')} °C, blast {_f(r.get('blast_nm3h'), ',.0f')} Nm3/h, O2 "
            f"{_f(r.get('O2_nm3h'), ',.0f')} Nm3/h (enrichment tag {_f(r.get('enrich_tag'), '.2f')} %), RAFT "
            f"{_f(r.get('raft'), ',.0f')} °C, eta CO {_f(o.get('eta'))} %, top gas {_f(o.get('T_top'), '.0f')} °C, "
            f"hot-blast pressure needed {_f(r.get('Pb_needed'), '.2f')} bar g, dP {_f(r.get('dP_needed'), '.2f')} bar, "
            f"shaft efficiency {_f(o.get('omega'), '.3f')} "
            + ("(blend-dependent)" if (r.get("omega") or {}).get("mode") == "blend" else "(held at the calibration)"))
    om = r.get("omega") or {}
    if om.get("mode") == "blend":
        lines.append(
            f"Shaft efficiency (blend-dependent, GAS-13..18): {_f(om.get('value'), '.3f')} against the calibration "
            f"{_f(om.get('omega_cal'), '.3f')}; parts: blend {_f(om.get('d_blend'), '+.4f')}, class correction "
            f"{_f(om.get('d_disc'), '+.4f')}, operation {_f(om.get('d_oper'), '+.4f')}, gas quantity {_f(om.get('d_R'), '+.4f')}; "
            f"sigma(coke) +/-{_f(om.get('sigma_K'), '.1f')} kg/tHM, sigma(cost) +/-{_f(r['cost'].get('sigma'), ',.0f')} "
            f"Rs/tHM; certainty {om.get('certainty')}"
            + (f"; outside what BF-2 ran: {'; '.join(om.get('outside') or [])}" if om.get("outside") else "")
            + f"; calibration error +/-{_f(om.get('cal_se_K'), '.1f')} and day-to-day spread +/-{_f(om.get('day_sd_K'), '.1f')} "
              "kg/tHM are not in sigma")
    bt = binding_table(r)
    if not bt.empty:
        lines.append("Binding limits (shadow price, Rs/tHM per unit): " + "; ".join(
            f"{limit_label(x.limit, names)} {x.rs_thm_per_unit:.2f} {x.unit}" for x in bt.itertuples()))
    for w in r.get("warnings", []) or []:
        lines.append(f"WARNING: {w}")
    return _section(label, lines)


def describe_against(r: Mapping[str, Any], other: Mapping[str, Any] | None, label: str, names: Mapping[str, str],
                     flux_ids: Sequence[str]) -> str:
    """The plan against another plan at the same output (today's blend, or the ore + flux-only answer)."""

    if other is None:
        return ""
    P, O = plan_tiles(r, flux_ids), plan_tiles(other, flux_ids)
    d = O["cost"]["total"] - P["cost"]["total"]
    ids = sorted({i for res in (r, other) for i, sh in res["share_wet"].items() if sh > 0.05},
                 key=lambda i: -r["share_wet"].get(i, 0.0))
    lines = [
        f"Plan costs Rs {abs(d):,.0f}/tHM {'less' if d >= 0 else 'more'} than {label} "
        f"(Rs {abs(d) * r['P_tday'] / 1e5:,.1f} lakh/day at {r['P_tday']:,.0f} t/d)",
        "Shares plan vs " + label + ": " + "; ".join(
            f"{names.get(i, i)} {r['share_wet'].get(i, 0.0):.1f} vs {other['share_wet'].get(i, 0.0):.1f} %" for i in ids),
        "Flux plan vs " + label + ": " + ", ".join(
            f"{names.get(i, i)} {r['wet_kg'].get(i, 0.0):.1f} vs {other['wet_kg'].get(i, 0.0):.1f} kg/tHM" for i in flux_ids
            if max(r["wet_kg"].get(i, 0.0), other["wet_kg"].get(i, 0.0)) >= 0.5) or "none",
        f"Cost split change (plan minus {label}), Rs/tHM: ore {P['cost']['ore'] - O['cost']['ore']:+,.0f}, flux "
        f"{P['cost']['flux'] - O['cost']['flux']:+,.0f}, fuel {P['cost']['fuel'] - O['cost']['fuel']:+,.0f}",
        f"Coke {P['fuel']['coke_dry'] - O['fuel']['coke_dry']:+.1f} kg/tHM dry, fuel rate "
        f"{P['fuel']['total'] - O['fuel']['total']:+.1f}, slag {P['slag']['rate'] - O['slag']['rate']:+.0f} kg/tHM, "
        f"CaO/SiO2 {P['slag']['b2'] - O['slag']['b2']:+.3f}, Al2O3 {P['slag']['Al2O3_pct'] - O['slag']['Al2O3_pct']:+.2f} pt, "
        f"MgO {P['slag']['MgO_pct'] - O['slag']['MgO_pct']:+.2f} pt, burden Fe "
        f"{P['overview']['fe_wet_pct'] - O['overview']['fe_wet_pct']:+.2f} pt",
        f"{label.capitalize()} itself: slag {O['slag']['rate']:.0f} kg/tHM, CaO/SiO2 {O['slag']['b2']:.3f}, Al2O3 "
        f"{O['slag']['Al2O3_pct']:.2f} %, MgO {O['slag']['MgO_pct']:.2f} %, coke {O['fuel']['coke_dry']:.1f} kg/tHM dry",
    ]
    return _section(f"PLAN AGAINST {label.upper()}", lines)


def describe_value_of_limits(VL: pd.DataFrame | None, names: Mapping[str, str]) -> str:
    if VL is None or VL.empty:
        return ""
    lines, none = [], []
    for x in VL.itertuples():
        if x.note != "infeasible" and abs(float(x.saving_rs_thm or 0.0)) < 0.5:
            none.append(x.change)
            continue
        note = (limit_labels([y.strip() for y in str(x.note).split(",") if y.strip()], names)
                if x.note != "infeasible" else "infeasible")
        lines.append(f"{x.change}: saving {_f(x.saving_rs_thm, '+.1f')} Rs/tHM; binds after: {note or 'nothing'}")
    if none:
        lines.append("No saving (< Rs 0.5/tHM) from: " + ", ".join(none))
    return _section("VALUE OF LIMITS (each relaxed one step at the same production, re-optimised)", lines)


def describe_frontier(res: Mapping[str, Any] | None, names: Mapping[str, str]) -> str:
    if not res or res.get("mode") != "cheapest" or res.get("best") is None:
        return ""
    F, best, top = res["frontier"], res["best"], res["top"]
    lines = [f"Cheapest output {best['P_tday']:,.0f} t/d at Rs {best['cost']['total']:,.0f}/tHM; binds: "
             + (limit_labels(best.get("binding", []), names) or "-")]
    if top is not None:
        lines.append(f"Highest output {top['P_tday']:,.0f} t/d at Rs {top['cost']['total']:,.0f}/tHM; binds: "
                     + (limit_labels(top.get("binding", []), names) or "-"))
    if F is not None and not F.empty:
        step = max(1, len(F) // 8)
        lines += [f"{x.prod_t_day:,.0f} t/d: Rs {x.cost_rs_thm:,.0f}/tHM, coke {x.coke_dry:.1f}, fuel {x.fuel_rate:.0f}, "
                  f"extra tonne Rs {_f(x.marginal_rs_per_t, ',.0f')}"
                  for x in list(F.itertuples())[::step]]
    return _section("FRONTIER (Step 1)", lines)


def describe_yard(materials: Sequence[Mapping[str, Any]], fluxes: Sequence[Mapping[str, Any]],
                  fuel: Mapping[str, Any]) -> str:
    lines = []
    for m in materials:
        if not m.get("carried", True):
            continue
        chem = m.get("chem", {})
        lines.append(
            f"{m.get('name', m['id'])} ({m.get('kind', '')}){'' if m.get('selected', True) else ' NOT OFFERED'}: price "
            f"Rs {_f(m.get('price'), ',.0f')}/t, Fe {_f(chem.get('Fe'))} %, SiO2 {_f(chem.get('SiO2'), '.2f')}, Al2O3 "
            f"{_f(chem.get('Al2O3'), '.2f')}, CaO {_f(chem.get('CaO'), '.2f')}, MgO {_f(chem.get('MgO'), '.2f')}, moisture "
            f"{_f(m.get('moist'))} %; share {_f(m.get('lo'), '.0f')}-{_f(m.get('hi'), '.0f')} %; stock "
            f"{_f(m.get('stock_mt'), ',.0f')} t" + (f"; lab sample {m['chem_time']}" if m.get("chem_time") else ""))
    for x in fluxes:
        lines.append(f"Flux {x.get('name', x['id'])}: Rs {_f(x.get('price'), ',.0f')}/t, max {_f(x.get('max_qty_mt'), ',.0f')} t/d")
    lines.append(f"Fuel prices Rs/t: coke {_f(fuel.get('coke_price'), ',.0f')}, nut coke {_f(fuel.get('nut_price'), ',.0f')}, "
                 f"PCI {_f(fuel.get('pci_price'), ',.0f')}")
    left = [m.get("name", m["id"]) for m in materials if not m.get("carried", True)]
    if left:
        lines.append("In the yard list but not offered (no lab chemistry in the window): " + ", ".join(left))
    return _section("YARD", lines)


def build_context(*, notes: str, reg: FactRegister, mode: str, plan: Mapping[str, Any] | None, plan_label: str,
                  names: Mapping[str, str], flux_ids: Sequence[str], settings: Mapping[str, Any],
                  limits: Mapping[str, Any], yard_source: str, calibration: Mapping[str, Any],
                  regime: Mapping[str, Any] | None = None, target_t_day: float | None = None,
                  model_check: Mapping[str, Any] | None = None, today: Mapping[str, Any] | None = None,
                  today_window: str = "", ore_only: Mapping[str, Any] | None = None,
                  value_of_limits: pd.DataFrame | None = None, frontier: Mapping[str, Any] | None = None,
                  materials: Sequence[Mapping[str, Any]] = (), fluxes: Sequence[Mapping[str, Any]] = (),
                  fuel: Mapping[str, Any] | None = None) -> CommentaryContext:
    """Everything the commentary model is given, and what was not available."""

    missing = []
    if not notes:
        missing.append("the optimiser notes")
    if not model_check or model_check.get("latest") is None:
        missing.append("the model check (Coke Energy shift results)")
    if plan is None:
        missing.append("an optimised plan")
    if "fixed" in mode.lower() and today is None:
        missing.append("today's charged blend")
    if yard_source != "live":
        missing.append("the live yard (reference yard of 16 Oct 2026 used)")
    live = [
        describe_inputs(settings, limits, yard_source=yard_source, calibration=calibration, regime=regime, mode=mode,
                        target_t_day=target_t_day),
        describe_model_check(model_check),
        describe_transit((model_check or {}).get("transit")),
        describe_plan(plan, f"OPTIMISED PLAN - {plan_label}", names, flux_ids) if plan is not None else "",
        describe_against(plan, today, "today's charged blend", names, flux_ids) if plan is not None else "",
        (f"\nToday's charged blend = the ferrous blend charged {today_window}, modelled at the plan's output, targets and "
         "controls with flux chosen to the same basicity band; slag-quality, charging and stock limits lifted because "
         "that blend did run.\n") if today is not None else "",
        describe_plan(today, "TODAY'S CHARGED BLEND, MODELLED", names, flux_ids, process=False) if today is not None else "",
        describe_against(plan, ore_only, "the ore + flux-only answer", names, flux_ids)
        if plan is not None and ore_only is not None and abs(ore_only["cost"]["total"] - plan["cost"]["total"]) >= 0.5
        else ("\nThe ore + flux-only answer (page 9's objective) is the same blend.\n" if ore_only is not None else ""),
        describe_value_of_limits(value_of_limits, names),
        describe_frontier(frontier, names),
        describe_yard(materials, fluxes, fuel or {}) if materials else "",
    ]
    if missing:
        live.append(_section("NOT AVAILABLE FOR THIS RUN", ["Do not speculate about these; say they were unavailable "
                                                             "if they matter: " + ", ".join(missing)]))
    text = "\n".join([
        "# PART 1 - OPTIMISER NOTES (static context)\n", notes or "(notes not available)",
        "\n# PART 2 - FACT REGISTER\n", facts_table(reg),
        "\n# PART 3 - LIVE RUN\n", "\n".join(x for x in live if x),
    ])
    return CommentaryContext(text=text.strip(), missing=missing)
