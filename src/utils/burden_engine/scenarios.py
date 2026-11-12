"""Page-facing scenario runs and result tables for the Burden & Fuel Optimiser (#255-#257).

Pure functions on top of :class:`utils.burden_engine.Engine`: build the starting controls and limits from the operating
regime and the register, run the three modes, and turn results into the tables the page shows. No Streamlit.

Modes
  * Step 1 - cheapest production level (``run_cheapest_level``): ore + flux + fuel cost per tHM against production
    under the day's limits; the cheapest point, the highest production and the frontier between them.
  * Step 2 - fixed production (``run_fixed``): the blend that minimises ore + flux + fuel cost at the target, next to
    the ore + flux (page-9) answer from the same rows.
  * Maximum production (``run_max``): blend and controls at the production ceiling.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from utils.burden_engine.engine import Engine, Limits, Settings, value_in_use
from utils.burden_engine.facts import FactRegister
from utils.coke_energy.slag_basis import SLAG_KEYS

MODES = ("cheapest", "fixed", "max")
#: Largest PCI a plan takes, kg/tHM as injected (the page's and the API's input bound).
PCI_MAX_KG_THM = 260.0


# ---------------------------------------------------------------------------------------------- starting point
def fuel_defaults(regime: Mapping[str, Any] | None, thermal: Mapping[str, Any], cal: Mapping[str, Any],
                  live_pci: float | None = None) -> dict[str, Any]:
    """PCI and nut coke defaults, kg/tHM as injected / as charged (user decision 7 Nov 2026). PCI: the live tag's
    1-h mean (``live_pci``, the Blend Mix Optimiser's reading; ignored above ``PCI_MAX_KG_THM``: the tag spikes when
    production stops and stuck at about 430 on 16-21 Oct 2026), else the regime window's median, else the
    calibration's, else the page config. Nut coke: the page config (70 kg/tHM). Returns pci_kg_thm, nut_kg_thm and the
    source of each."""
    case = cal.get("case", {}) if isinstance(cal, Mapping) else {}
    reg = dict(regime or {})
    out: dict[str, Any] = {}
    moist = float((case.get("pci") or {}).get("moist_pct", 0.0)) / 100.0
    v = reg.get("pci_kgthm_dry")
    if live_pci is not None and np.isfinite(live_pci) and 0 < live_pci <= PCI_MAX_KG_THM:
        out["pci_kg_thm"], out["pci_kg_thm_source"] = float(live_pci), "live tag, 1-h mean"
    elif v is not None and np.isfinite(v) and v > 0:
        out["pci_kg_thm"], out["pci_kg_thm_source"] = float(v) / (1 - moist), "regime median"
    elif case.get("pci_kg"):
        out["pci_kg_thm"], out["pci_kg_thm_source"] = float(case["pci_kg"]) / (1 - moist), "calibration"
    else:
        out["pci_kg_thm"], out["pci_kg_thm_source"] = float(thermal.get("pci_kg_thm", Settings.pci_plant)), "config"
    out["nut_kg_thm"], out["nut_kg_thm_source"] = float(thermal.get("nut_kg_thm", Settings.nut_plant)), "config"
    return out


def settings_from_regime(regime: Mapping[str, Any] | None, thermal: Mapping[str, Any], cal: Mapping[str, Any],
                         **overrides: Any) -> Settings:
    """Today's controls: the regime's medians where known, else the calibration's; thermal targets from the page config
    (client basis, ASM-08); PCI and nut coke from ``fuel_defaults`` (regime median PCI, nut coke from the config)."""
    case = cal.get("case", {})
    reg = dict(regime or {})
    pick = lambda key, fallback: float(reg[key]) if reg.get(key) is not None and np.isfinite(reg[key]) else float(fallback)
    s = Settings(
        HBT=pick("HBT", case.get("HBT", Settings.HBT)),
        O2_per_air=pick("O2_per_air", case.get("O2_per_air", Settings.O2_per_air)),
        humidity_g=None,                 # today's Open-Meteo blast moisture, fetched live (DATA-11)
        HMT=float(thermal.get("HMT", Settings.HMT)), hm_Si=float(thermal.get("hm_Si", Settings.hm_Si)),
        pci_plant=float(fuel_defaults(regime, thermal, cal)["pci_kg_thm"]),
        nut_plant=float(fuel_defaults(regime, thermal, cal)["nut_kg_thm"]),
        Pb=pick("Pb", Settings.Pb), Pt=pick("Pt", Settings.Pt),
    )
    return replace(s, **overrides)


def limits_from(reg: FactRegister, slag: Mapping[str, float] | None = None, charging: Mapping[str, float] | None = None,
                **overrides: Any) -> Limits:
    """Register defaults (ASM-11/12/13, BAND-01), then the operator's saved slag limits and charging, then overrides."""
    d = dict(slag or {})
    d.update({k: v for k, v in (charging or {}).items() if k in ("charge_t", "charges_per_h")})
    d.update(overrides)
    return Limits.from_register(reg, **d)


def o2_levels(eng: Engine, s: Settings, L: Limits, n: int = 9) -> list[float]:
    """O2/air grid from today's up to the enrichment limit (the case report's grid)."""
    r_max = eng.hy.o2_per_air_from_tag(L.enrich_max)
    if r_max <= s.O2_per_air:
        return [s.O2_per_air]
    return list(np.linspace(s.O2_per_air, r_max, max(2, int(n))))


# ---------------------------------------------------------------------------------------------- modes
def run_fixed(eng: Engine, prod_t_day: float, s: Settings, L: Limits) -> dict[str, Any]:
    """Step 2: cheapest blend at a fixed production with fuel in the objective, and the ore + flux answer."""
    with_fuel = eng.min_cost(prod_t_day=prod_t_day, s=s, L=L, fuel=True)
    ore_only = eng.min_cost(prod_t_day=prod_t_day, s=s, L=L, fuel=False, z0=with_fuel["z"])
    return dict(mode="fixed", with_fuel=with_fuel, ore_only=ore_only)


def run_today(inputs: Mapping[str, Any], cal: Mapping[str, Any], shares_wet: Mapping[str, float], prod_t_day: float,
              s: Settings, L: Limits, facts: FactRegister | None = None, tol_pct: float = 0.05) -> dict[str, Any] | None:
    """Today's charged blend, priced and modelled at the plan's production, targets and controls.

    The ferrous shares are pinned to today's (renormalised over the materials the engine carries); flux is chosen to
    the same basicity band; slag-quality, charging and stock limits are lifted, because today's blend did run - the
    result reports its slag rather than refusing it. None when the shares are empty or the LP fails."""
    ids = [m["id"] for m in inputs["materials"] if shares_wet.get(m["id"], 0.0) > 0]
    total = sum(shares_wet[i] for i in ids)
    if not ids or total <= 0:
        return None
    bounds = {i: (max(0.0, 100 * shares_wet[i] / total - tol_pct), min(100.0, 100 * shares_wet[i] / total + tol_pct))
              for i in ids}
    inp = dict(inputs, materials=[dict(m, max_t_day=None) for m in inputs["materials"]],
               fluxes=[dict(f, max_t_day=None) for f in inputs["fluxes"]])
    eng = Engine(inp, cal, facts=facts, materials=ids, bounds=bounds)
    Lt = replace(L, slag_max=1000.0, Al2O3_max=100.0, MgO_min=0.0, MgO_Al2O3_min=0.0, charges_per_h=99.0,
                 process=False)
    return eng._try(s, prod_t_day / 24, Lt, True)


def run_cheapest_level(eng: Engine, s: Settings, L: Limits, *, search_o2: bool = True, n_o2: int = 9,
                       P_lo: float = 84.0, step: float = 1.0) -> dict[str, Any]:
    """Step 1: the production level with the lowest ore + flux + fuel cost per tHM, the frontier, and the ceiling."""
    L = replace(L, process=True)
    levels = o2_levels(eng, s, L, n_o2) if search_o2 else None
    best = eng.min_total_cost(L, base=s, P_lo=P_lo, step=step, o2_levels=levels)
    if best is None:
        return dict(mode="cheapest", best=None, frontier=pd.DataFrame(), top=None)
    curves = best.get("curves", {})
    rows = [x for c in curves.values() for x in c if x.get("feasible")]
    frontier = frontier_table(rows)
    top = max(rows, key=lambda x: x["P_th"]) if rows else None
    return dict(mode="cheapest", best=best, frontier=frontier, top=top)


def run_max(eng: Engine, s: Settings, L: Limits) -> dict[str, Any]:
    r = eng.max_production(replace(L, process=True), base=s)
    return dict(mode="max", result=r)


def run_candidates(eng: Engine, prod_t_day: float, s: Settings, L: Limits,
                   today: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Auto mode (``utils.burden_engine.auto_review``): the plan at one production under the three decision settings
    (lowest expected cost, risk-averse, High certainty only) with blend-dependent shaft efficiency, today's charged
    blend alongside, the rows of the comparison and the fixed rule's pick."""
    from utils.burden_engine.auto_review import CANDIDATES, candidate_row, pick

    base = replace(s, omega_mode="blend")
    results, rows = {}, []
    z0 = None
    for cid, label, ov in CANDIDATES:
        r = eng._try(replace(base, **ov), prod_t_day / 24, L, True, z0=z0)
        if r is not None and z0 is None:
            z0 = r["z"]
        results[cid] = r
        sig = eng.saving_sigma(r, today) if (r is not None and today is not None) else None
        rows.append(candidate_row(cid, label, r, today, sig))
    if today is not None:
        results["T"] = today
        rows.append(candidate_row("T", "Today's charged blend", today))
    cid, reason = pick(rows)
    return dict(rows=rows, results=results, pick=cid, reason=reason, prod_t_day=prod_t_day)


# ---------------------------------------------------------------------------------------------- tables
def frontier_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Cheapest feasible plan at each production level, with the marginal ore + flux + fuel cost of each extra tonne
    (Rs per extra t of hot metal, from the cost per day)."""
    if not rows:
        return pd.DataFrame()
    best: dict[float, Mapping[str, Any]] = {}
    for r in rows:
        key = round(r["P_tday"], 1)
        if key not in best or r["cost"]["total"] < best[key]["cost"]["total"]:
            best[key] = r
    recs = []
    for key in sorted(best):
        r = best[key]
        recs.append(dict(prod_t_day=r["P_tday"], cost_rs_thm=r["cost"]["total"], ore_flux_rs_thm=r["cost"]["ore"] + r["cost"]["flux"],
                         fuel_rs_thm=r["cost"]["fuel"], coke_dry=r["out"]["K"],
                         coke_wet=r["out"].get("K_wet", r["out"]["K"]), fuel_rate=r["out"]["fuel_rate"],
                         slag=r["out"]["slag"], HBT=r["settings"]["HBT"], enrich_tag=r["enrich_tag"], blast_nm3h=r["blast_nm3h"],
                         O2_nm3h=r["O2_nm3h"], Pb_needed=r["Pb_needed"], charges_per_h=r["charges_per_h"],
                         binding=", ".join(b for b in r.get("binding", []) if not b.startswith(("min share", "max share")))))
    T = pd.DataFrame(recs)
    day_cost = T.prod_t_day * T.cost_rs_thm
    T["marginal_rs_per_t"] = day_cost.diff() / T.prod_t_day.diff()
    return T


def blend_table(eng: Engine, r: Mapping[str, Any], other: Mapping[str, Any] | None = None,
                other_label: str = "ore + flux only") -> pd.DataFrame:
    """Materials in the plan: wet % of ferrous, kg/tHM, t/day, price; optionally the other plan's share alongside."""
    rows = []
    for v in eng.vars:
        kg = r["wet_kg"][v.id]
        if kg < 0.05 and (other is None or other["wet_kg"][v.id] < 0.05):
            continue
        row = dict(material=v.name, kind=v.kind, share_wet_pct=(r["share_wet"].get(v.id) if v.kind != "flux" else None),
                   kg_per_thm_wet=kg, t_per_day=r["t_day"][v.id], price_rs_t=v.price)
        if other is not None:
            row[f"share_{other_label}"] = other["share_wet"].get(v.id) if v.kind != "flux" else None
            row[f"kg_{other_label}"] = other["wet_kg"][v.id]
        rows.append(row)
    return pd.DataFrame(rows)


# Operator names for the LP's limit rows (the engine's row names stay as they are).
LIMIT_LABELS = {
    "slag cap": "slag rate cap", "basicity min": "CaO/SiO₂ floor", "basicity max": "CaO/SiO₂ ceiling",
    "Al2O3 max": "Al₂O₃ cap", "MgO min": "MgO floor", "MgO/Al2O3 min": "MgO/Al₂O₃ floor",
    "top gas temp min": "top-gas temperature floor", "fuel rate max": "fuel-rate ceiling (600)",
    "charging": "charging rate", "blast pressure": "blast pressure", "blast volume": "hot-blast volume",
    "oxygen flow": "oxygen flow", "enrichment": "enrichment tag",
}


def limit_label(name: str, names: Mapping[str, str] | None = None) -> str:
    """'min share lloyds_pellet' -> 'LLOYDS PELLET share floor'; 'basicity min' -> 'CaO/SiO₂ floor'."""
    for prefix, word in (("min share ", "share floor"), ("max share ", "share cap")):
        if name.startswith(prefix):
            mid = name[len(prefix):]
            return f"{(names or {}).get(mid, mid)} {word}"
    return LIMIT_LABELS.get(name, name)


def limit_labels(names_: Sequence[str], names: Mapping[str, str] | None = None, *, shares: bool = False) -> str:
    """Comma-joined operator names of binding rows (share bounds left out unless ``shares``)."""
    keep = [n for n in names_ if shares or not n.startswith(("min share", "max share"))]
    return ", ".join(limit_label(n, names) for n in keep)


TRACE_KG = 0.5          # flux below this (kg/tHM) is left out of the comparison tables


def blend_compare_table(r: Mapping[str, Any], others: Sequence[tuple[str, Mapping[str, Any] | None]] = (),
                        names: Mapping[str, str] | None = None, prices: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Ferrous materials: wet % of ferrous and kg/tHM for the plan and each other plan (today's blend, ore + flux
    only), the plan's t/day and price. Rows: materials used by any of them, plan's order first."""
    results = [("Plan", r), *[(lbl, o) for lbl, o in others if o is not None]]
    ids: list[str] = []
    for _, res in results:
        ids += [i for i, sh in res["share_wet"].items() if sh > 0.05 and i not in ids]
    rows = []
    for i in ids:
        row = {"Material": (names or {}).get(i, i)}
        for lbl, res in results:
            row[f"{lbl}, wet %"] = float(res["share_wet"].get(i, 0.0))
        for lbl, res in results:
            row[f"{lbl}, kg/tHM"] = float(res["wet_kg"].get(i, 0.0))
        row["Plan, t/d"] = float(r["t_day"].get(i, 0.0))
        row["Price, Rs/t"] = float((prices or {}).get(i, float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


def flux_compare_table(r: Mapping[str, Any], others: Sequence[tuple[str, Mapping[str, Any] | None]] = (),
                       names: Mapping[str, str] | None = None, flux_ids: Sequence[str] = ()) -> pd.DataFrame:
    """Fluxes used (>= TRACE_KG kg/tHM in any plan), kg/tHM wet per plan."""
    results = [("Plan", r), *[(lbl, o) for lbl, o in others if o is not None]]
    rows = []
    for i in flux_ids:
        kg = {lbl: float(res["wet_kg"].get(i, 0.0)) for lbl, res in results}
        if max(kg.values()) >= TRACE_KG:
            rows.append({"Flux": (names or {}).get(i, i), **{f"{lbl}, kg/tHM": v for lbl, v in kg.items()}})
    return pd.DataFrame(rows)


def plan_tiles(r: Mapping[str, Any], flux_ids: Sequence[str] = ()) -> dict[str, dict[str, float]]:
    """The figures of the page-9-style result panels, grouped as the furnace runs: cost (Rs/tHM), fuel (kg/tHM),
    slag (rate, t/d, chemistry and the three basicity ratios) and charging (t/d, charges/h, hot metal per charge,
    flux kg/tHM). Coke, nut coke and PCI tonnages are as charged (wet coke, the operator's nut coke and PCI)."""

    o, c, s = r["out"], r["cost"], r["settings"]
    P_t = float(r["P_tday"])
    ox = {k: float(o.get(k, 0.0) or 0.0) for k in ("CaO", "SiO2", "MgO", "Al2O3")}
    ratio = lambda num, den: num / den if den else math.nan  # noqa: E731
    flux_kg = sum(float(r["wet_kg"].get(i, 0.0)) for i in flux_ids)
    cph = float(r.get("charges_per_h") or 0.0)
    return dict(
        overview=dict(total=float(c["total"]), P_tday=P_t, P_th=float(r["P_th"]), fe_wet_pct=float(r["fe_wet_pct"]),
                      ferrous_kg=float(r["ferrous_wet"])),
        cost=dict(ore=float(c["ore"]), flux=float(c["flux"]), fuel=float(c["fuel"]), coke=float(c["coke"]),
                  nut=float(c["nut"]), pci=float(c["pci"]), slag_credit=float(c.get("slag_credit", 0.0)),
                  gross=float(c.get("gross", c["total"])), total=float(c["total"])),
        fuel=dict(coke_dry=float(o["K"]), coke_wet=float(o.get("K_wet", o["K"])), nut=float(s["nut_plant"]),
                  pci=float(s["pci_plant"]), total=float(o["fuel_rate"])),
        slag=dict(rate=float(o["slag"]), t_day=float(o["slag"]) * P_t / 1000.0, Al2O3_pct=float(o["Al2O3_pct"]),
                  MgO_pct=float(o["MgO_pct"]), b2=ratio(ox["CaO"], ox["SiO2"]),
                  t_basicity=ratio(ox["CaO"] + ox["MgO"], ox["SiO2"]),
                  ib4=ratio(ox["CaO"] + ox["MgO"], ox["SiO2"] + ox["Al2O3"]), mgo_al2o3=ratio(ox["MgO"], ox["Al2O3"])),
        charging=dict(coke_t=float(o.get("K_wet", o["K"])) * P_t / 1000.0, nut_t=float(s["nut_plant"]) * P_t / 1000.0,
                      pci_t=float(s["pci_plant"]) * P_t / 1000.0, ferrous_t=float(r["ferrous_wet"]) * P_t / 1000.0,
                      charges_per_h=cph, hm_per_charge=float(r["P_th"]) / cph if cph else math.nan,
                      flux_kg=flux_kg),
    )


def plan_sentence(r: Mapping[str, Any], names: Mapping[str, str], flux_ids: list[str],
                  today: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """(kind, markdown): one line an operator can act on - what to charge, what it costs, what it saves against
    today's blend. kind is "saving", "dearer" or "plain" (no comparison)."""
    fer = sorted(((names.get(i, i), sh) for i, sh in r["share_wet"].items() if sh > 0.05), key=lambda t: -t[1])
    flux = [(names.get(i, i), r["wet_kg"].get(i, 0.0)) for i in flux_ids if r["wet_kg"].get(i, 0.0) >= 0.5]
    # two decimals on the shares, flux and coke (user decision 11 Nov 2026); coke as charged (wet), the client's basis
    text = "**Charge** " + ", ".join(f"{n} {sh:.2f} %" for n, sh in fer)
    text += (" with " + ", ".join(f"{n.lower()} {kg:.2f} kg/tHM" for n, kg in flux)) if flux else ", no flux"
    text += f": Rs {r['cost']['total']:,.0f}/tHM, coke {r['out'].get('K_wet', r['out']['K']):.2f} kg/tHM as charged."
    if today is None:
        return "plain", text
    d = today["cost"]["total"] - r["cost"]["total"]
    if abs(d) < 0.5:
        return "plain", text + " Same cost as today's blend."
    lakh = abs(d) * r["P_tday"] / 1e5
    dk = r["out"].get("K_wet", r["out"]["K"]) - today["out"].get("K_wet", today["out"]["K"])
    text += (f" **Rs {abs(d):,.0f}/tHM {'less' if d > 0 else 'more'} than today's blend** (Rs {lakh:,.1f} lakh/day at "
             f"this output; coke {dk:+.2f} kg/tHM).")
    return ("saving" if d > 0 else "dearer"), text


def cost_table(r: Mapping[str, Any]) -> pd.DataFrame:
    c, o, s = r["cost"], r["out"], r["settings"]
    rows = [("Ore", c["ore"], ""), ("Flux", c["flux"], ""),
            ("Coke", c["coke"], f"{o['K_wet']:.1f} kg/tHM wet ({o['K']:.1f} dry)"),
            ("Nut coke", c["nut"], f"{s['nut_plant']:.0f} kg/tHM"), ("PCI", c["pci"], f"{s['pci_plant']:.0f} kg/tHM")]
    if c.get("slag_credit", 0.0):
        rows.append(("Slag credit", -c["slag_credit"], f"{o['slag']:.0f} kg/tHM at Rs {c.get('slag_price', 0.0):,.0f}/t"))
    rows.append(("Total", c["total"], f"{c['total'] * r['P_tday'] / 1e5:,.1f} lakh Rs/day"))
    return pd.DataFrame(rows, columns=["item", "rs_per_thm", "basis"])


def slag_table(r: Mapping[str, Any]) -> pd.DataFrame:
    o, L = r["out"], r["limits"]
    rows = [("Slag rate, kg/tHM", o["slag"], f"≤ {L['slag_max']:.0f}"),
            ("CaO/SiO₂", o["B2"], f"{L['B2_min']:.2f} – {L['B2_max']:.2f}"),
            ("Al₂O₃, %", o["Al2O3_pct"], f"≤ {L['Al2O3_max']:.1f}"),
            ("MgO, %", o["MgO_pct"], f"≥ {L['MgO_min']:.1f}"),
            ("MgO/Al₂O₃", o["MgO"] / o["Al2O3"] if o["Al2O3"] else math.nan, f"≥ {L['MgO_Al2O3_min']:.3f}")]
    return pd.DataFrame(rows, columns=["quantity", "value", "limit"])


def slag_detail_of(r: Mapping[str, Any]) -> dict[str, Any] | None:
    """The slag breakup of a plan (``utils.coke_energy.slag_basis.slag_detail``), None on the calibration's own
    slag rule (a plan without the Blend Mix Optimiser's basis)."""
    d = (r.get("out") or {}).get("slag_detail")
    return dict(d) if d else None


def material_rows(r: Mapping[str, Any], names: Mapping[str, str] | None = None,
                  prices: Mapping[str, float] | None = None, kinds: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Each material of the plan (ferrous and flux, >= 0.05 kg/tHM) with page 9's blend-table columns per tHM: wet
    share, wet and dry kg/tHM, Fe and slag-forming oxides kg/tHM (its ``Fe (MT)`` and ``Slag (MT)``), kg per charge
    (wet kg/tHM x hot metal per charge), t/d, price and cost Rs/tHM. Fe and slag come from the plan's slag breakup and
    are None without it."""
    d = slag_detail_of(r) or {}
    by = d.get("by_material") or {}
    cph = float(r.get("charges_per_h") or 0.0)
    hm_per_charge = float(r["P_th"]) / cph if cph else None
    rows = []
    for i, wet in r["wet_kg"].items():
        wet = float(wet)
        if wet < 0.05:
            continue
        m = by.get(i)
        price = float((prices or {}).get(i, float("nan")))
        rows.append(dict(id=i, material=(names or {}).get(i, i), kind=(kinds or {}).get(i) or (m or {}).get("kind"),
                         share_wet_pct=(float(r["share_wet"][i]) if i in r["share_wet"] else None), wet_kg=wet,
                         dry_kg=float(r["dry_kg"].get(i, 0.0)), fe_kg=(float(m["fe_kg"]) if m else None),
                         slag_kg=(float(m["slag_kg"]) if m else None),
                         kg_per_charge=(wet * hm_per_charge if hm_per_charge else None), t_day=float(r["t_day"].get(i, 0.0)),
                         price_rs_t=price, cost_rs_thm=(wet * price / 1000.0 if math.isfinite(price) else None)))
    return rows


def slag_components_table(r: Mapping[str, Any]) -> pd.DataFrame:
    """The slag's components (kg/tHM and % of the slag) on the Blend Mix Optimiser's basis, page 9's order, with a
    total row; empty without the breakup."""
    d = slag_detail_of(r)
    if not d:
        return pd.DataFrame(columns=["component", "kg_per_thm", "pct"])
    slag = d["slag"]
    total = float(d["rollup"]["slag"]) or float(sum(slag.values()))
    rows = [(label, float(slag[k]), 100 * float(slag[k]) / total if total else math.nan) for k, label in SLAG_KEYS]
    rows.append(("Slag", total, 100.0 if total else math.nan))
    return pd.DataFrame(rows, columns=["component", "kg_per_thm", "pct"])


#: The rows of ``slag_balance_table``: component label, the key of its inputs (elements for Fe, Mn, Ti), the key of its
#: slag form, the removals that take it to the hot metal or the gas.
_BALANCE_ROWS = (("SiO₂", "sio2", "SiO2", ("SiO2_to_Si",)), ("Al₂O₃", "al2o3", "Al2O3", ()),
                 ("CaO", "cao", "CaO", ()), ("MgO", "mgo", "MgO", ()),
                 ("Na₂O + K₂O", "alkali", "alk", ("alkali_to_gas",)), ("S", "s", "S", ("S_to_hot_metal", "S_to_gas")),
                 ("CaF₂", "caf2", "CaF2", ()), ("Fe → FeO", "fe", "FeO", ("Fe_to_hot_metal",)),
                 ("Mn → MnO", "mn", "MnO", ("MnO_to_hot_metal",)), ("Ti → TiO₂", "ti", "TiO2", ("TiO2_to_hot_metal", "TiO2_left_out")))


def slag_balance_table(r: Mapping[str, Any]) -> pd.DataFrame:
    """Where each component goes (page 9's per-oxide balance, kg/tHM): charged by the ore, the flux and the fuel ash,
    carried away by the dust, taken by the hot metal or the gas, left in the slag. Fe, Mn and Ti are charged as the
    element and leave as FeO, MnO, TiO₂; ``hot_metal_gas`` of those rows is in the slag's form (FeO's Fe as the
    element). Empty without the breakup."""
    d = slag_detail_of(r)
    cols = ["component", "ore", "flux", "fuel_ash", "dust", "hot_metal_gas", "slag"]
    if not d:
        return pd.DataFrame(columns=cols)
    src, rem, slag = d["sources"], d["removals"], d["slag"]
    rows = []
    for label, key, out_key, removals in _BALANCE_ROWS:
        charged = {k: float(src[k][key]) for k in ("ore", "flux", "fuel_ash")}
        dust = float(src["dust"][key])
        net = float(d["net"][key])
        dust_taken = sum(charged.values()) - net                      # the dust deduction is clamped at zero per component
        out = float(slag[out_key])
        rows.append(dict(component=label, **charged, dust=-dust_taken,
                         hot_metal_gas=-sum(float(rem[k]) for k in removals) if removals else 0.0, slag=out))
    return pd.DataFrame(rows, columns=cols)


def slag_rollup_table(r: Mapping[str, Any]) -> pd.DataFrame:
    """Page 9's per-source rollup (kg/tHM): the slag-forming oxides charged by the ore, the flux and the fuel ash,
    less the dust, less what the hot metal and the gas take, plus the FeO, S and CaF₂ of the slag, equal to the slag
    rate. Empty without the breakup."""
    d = slag_detail_of(r)
    if not d:
        return pd.DataFrame(columns=["source", "kg_per_thm"])
    ru = d["rollup"]
    rows = [("Ore (slag-forming oxides)", ru["ore"]), ("Flux", ru["flux"]), ("Fuel ash", ru["fuel_ash"]),
            ("Top-gas dust", ru["dust"]), ("To hot metal and gas", ru["hot_metal_gas"]),
            ("FeO, S and CaF₂ of the slag", ru["other"]), ("Slag", ru["slag"])]
    return pd.DataFrame(rows, columns=["source", "kg_per_thm"])


def slag_removals_table(r: Mapping[str, Any]) -> pd.DataFrame:
    """Page 9's worked hot-metal removals (kg/tHM) with the rule each follows."""
    d = slag_detail_of(r)
    cols = ["removal", "rule", "kg_per_thm"]
    if not d:
        return pd.DataFrame(columns=cols)
    pi, rem, ru = d["pig_iron"], d["removals"], d["rules"]
    rows = [("SiO₂ reduced to hot-metal Si", f"theoretical pig iron {pi['theoretical_kg']:,.0f} kg x Si {ru['Si_pct']:.2f} % x 2.14",
             rem["SiO2_to_Si"]),
            ("Fe to the hot metal", f"{100 * ru['fe_to_pig_iron_fraction']:.1f} % of the Fe charged less the dust's; the rest FeO",
             rem["Fe_to_hot_metal"]),
            ("MnO reduced to hot-metal Mn", "the hot metal's Mn, the rest MnO", rem["MnO_to_hot_metal"]),
            ("TiO₂ reduced to hot-metal Ti", "the hot metal's Ti" + (", the rest TiO₂ in the slag" if ru["tio2_to_slag"]
                                                                      else "; the rest left out (workbook basis)"),
             rem["TiO2_to_hot_metal"]),
            ("S to the hot metal", f"theoretical pig iron x S {ru['S_pct']:.3f} %", rem["S_to_hot_metal"]),
            ("S to the gas", f"{ru['sulphur_gas_loss_pct']:.0f} % of the S charged less the dust's", rem["S_to_gas"]),
            ("Alkalis to the gas", f"{100 * (1 - ru['alkali_to_slag_fraction']):.0f} % of Na₂O + K₂O charged less the dust's",
             rem["alkali_to_gas"]),
            ("P and Zn to the hot metal", "all", rem["P_to_hot_metal"] + rem["Zn_to_hot_metal"])]
    if not ru["tio2_to_slag"]:
        rows.append(("TiO₂ left out of the slag", "workbook basis", rem["TiO2_left_out"]))
    return pd.DataFrame(rows, columns=cols)


FUEL_LABELS = (("coke", "Coke"), ("nut", "Nut coke"), ("pci", "PCI"))


def fuel_ash_table(r: Mapping[str, Any]) -> pd.DataFrame:
    """Page 9's fuel-ash inputs on the plan (coke, nut coke, PCI): the fuel as charged and dry (kg/tHM), its moisture
    and ash %, the ash kg/tHM, the ash analysis (% of the ash: SiO₂, Al₂O₃, CaO, MgO, Fe₂O₃, MnO, TiO₂, alkalis), the
    S and P of the dry fuel and the slag-forming oxides it brings (kg/tHM). Empty without the breakup."""
    d = slag_detail_of(r)
    cols = ["fuel", "wet_kg", "dry_kg", "moist_pct", "ash_pct", "ash_kg", "sio2_pct", "al2o3_pct", "cao_pct", "mgo_pct",
            "fe2o3_pct", "mno_pct", "tio2_pct", "alk_pct", "s_pct", "p_pct", "slag_kg"]
    if not d:
        return pd.DataFrame(columns=cols)
    rows = []
    for key, label in FUEL_LABELS:
        f = (d.get("by_fuel") or {}).get(key)
        if not f:
            continue
        a = f.get("ash_comp") or {}
        rows.append(dict(fuel=label, wet_kg=float(f.get("wet_kg", f["dry_kg"])), dry_kg=float(f["dry_kg"]),
                         moist_pct=float(f.get("moist_pct", 0.0)), ash_pct=float(f.get("ash_pct", 0.0)), ash_kg=float(f["ash_kg"]),
                         sio2_pct=float(a.get("SiO2", 0.0)), al2o3_pct=float(a.get("Al2O3", 0.0)), cao_pct=float(a.get("CaO", 0.0)),
                         mgo_pct=float(a.get("MgO", 0.0)), fe2o3_pct=float(a.get("Fe2O3", 0.0)), mno_pct=float(a.get("MnO", 0.0)),
                         tio2_pct=float(a.get("TiO2", 0.0)), alk_pct=float(a.get("alk", 0.0)), s_pct=float(f.get("S_pct", 0.0)),
                         p_pct=float(f.get("P_pct", 0.0)), slag_kg=float(f["slag_kg"])))
    return pd.DataFrame(rows, columns=cols)


def dust_table(r: Mapping[str, Any]) -> pd.DataFrame:
    """Page 9's top-gas dust rows on the plan, per enabled stream: kg per charge (the configured or saved figure),
    kg/tHM wet at the plan's charges per tHM, dry kg/tHM, the analysis (% of the dry dust: SiO₂, Al₂O₃, CaO, MgO, Fe),
    the Fe and the slag-forming oxides it carries away (kg/tHM) and where the figure came from. Empty without the
    breakup."""
    d = slag_detail_of(r)
    cols = ["id", "stream", "kg_per_charge", "kg_per_thm", "dry_kg", "moist_pct", "sio2_pct", "al2o3_pct", "cao_pct", "mgo_pct",
            "fe_pct", "fe_kg", "slag_kg", "source"]
    if not d:
        return pd.DataFrame(columns=cols)
    rows = []
    for i, s in (d.get("dust") or {}).items():
        p = s.get("pct") or {}
        rows.append(dict(id=i, stream=s["name"], kg_per_charge=s.get("kg_per_charge"), kg_per_thm=float(s["kg"]),
                         dry_kg=float(s["dry_kg"]), moist_pct=float(s.get("moist_pct", 0.0)), sio2_pct=float(p.get("SiO2", 0.0)),
                         al2o3_pct=float(p.get("Al2O3", 0.0)), cao_pct=float(p.get("CaO", 0.0)), mgo_pct=float(p.get("MgO", 0.0)),
                         fe_pct=float(p.get("Fe", 0.0)), fe_kg=float(s["fe_kg"]), slag_kg=float(s["slag_kg"]),
                         source=str(s.get("source", "") or "")))
    return pd.DataFrame(rows, columns=cols)


def slag_basis_note(r: Mapping[str, Any]) -> str | None:
    """One line on the slag basis of a plan (its held hot-metal chemistry, dust and minor-oxide rules), None on the
    calibration's rule."""
    d = slag_detail_of(r)
    if not d:
        return None
    ru = d["rules"]
    dust = ", ".join(f"{v['name']} {v['kg']:.1f}" for v in d["dust"].values())
    return (f"Slag on the Blend Mix Optimiser's basis: hot-metal Si {ru['Si_pct']:.2f} %, S {ru['S_pct']:.3f} %, "
            f"C {ru['C_pct']:.1f} % held for the slag side (the coke balance keeps Si {r['settings']['hm_Si']:.2f} %); "
            f"top-gas dust {dust} kg/tHM deducted; TiO₂ {'in' if ru['tio2_to_slag'] else 'left out of'} the slag; "
            f"alkalis {100 * ru['alkali_to_slag_fraction']:.0f} % to slag; FeO from {100 * (1 - ru['fe_to_pig_iron_fraction']):.1f} % "
            "of the Fe charged.")


def process_table(r: Mapping[str, Any]) -> pd.DataFrame:
    o, s, L = r["out"], r["settings"], r["limits"]
    rows = [
        ("Production", f"{r['P_th']:.1f} t/h · {r['P_tday']:,.0f} t/d", ""),
        ("Fuel rate (coke wet + nut + PCI)", f"{o['fuel_rate']:.1f} kg/tHM", f"{L['fuel_min']:.0f} – {L['fuel_max']:.0f} (BAND-01)"),
        ("ηCO / top gas", f"{o['eta']:.1f} % · {o['T_top']:.0f} °C", f"top gas ≥ {L['T_top_min']:.0f} °C"),
        ("Shaft efficiency ω", f"{o['omega']:.3f}" + (f" ± {r['omega']['sigma_omega']:.3f}" if (r.get("omega") or {}).get("mode") == "blend" else ""),
         "STRATA-ω, " + (r.get("omega") or {}).get("certainty", "") + " certainty"
         if (r.get("omega") or {}).get("mode") == "blend" else "Fixed-ω (held at the calibration)"),
        ("Hot blast temperature", f"{s['HBT']:,.0f} °C", f"≤ {L['HBT_max']:,.0f}"),
        ("Hot blast (meter basis)", f"{r['blast_nm3h']:,.0f} Nm³/h", f"≤ {L['V_max']:,.0f}" if L["process"] else ""),
        ("Oxygen / enrichment tag", f"{r['O2_nm3h']:,.0f} Nm³/h · {r['enrich_tag']:.2f} %",
         f"≤ {L['O2_flow_max']:,.0f} · ≤ {L['enrich_max']:.1f}" if L["process"] else ""),
        ("RAFT (plant workbook)", f"{r['raft']:,.0f} °C at {s['humidity_g']:.0f} g/Nm³",
         f"{L['RAFT_min']:,.0f} – {L['RAFT_max']:,.0f}"),
        ("Hot-blast pressure needed", f"{r['Pb_needed']:.2f} bar g at top {s['Pt']:.2f}", f"≤ {L['Pb_max']:.2f}" if L["process"] else ""),
        ("Pressure drop", f"{r['dP_needed']:.2f} bar", f"≤ {L['dP_max']:.2f}" if L["process"] else ""),
        ("Blast at the set hot-blast pressure", f"{r['V_ceiling']:,.0f} Nm³/h at {s['Pb']:.2f} bar g", ""),
        ("Charging", f"{r['charges_per_h']:.2f} charges/h · {r['charging_th']:.0f} t/h",
         f"≤ {L['charges_per_h']:.2f} × {L['charge_t']:.2f} t"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value", "limit"])


_DUAL_UNITS = {
    "slag cap": "per kg/tHM of slag cap", "basicity min": "per kg/tHM CaO shortfall", "basicity max": "per kg/tHM CaO excess",
    "Al2O3 max": "per kg/tHM Al2O3", "MgO min": "per kg/tHM MgO", "MgO/Al2O3 min": "per kg/tHM MgO",
    "top gas temp min": "per °C", "fuel rate max": "per kg/tHM fuel", "charging": "per t/h of charge",
    "blast pressure": "per Nm³/h of blast", "blast volume": "per Nm³/h of blast", "oxygen flow": "per Nm³/h O2",
}


def binding_table(r: Mapping[str, Any]) -> pd.DataFrame:
    """Binding rows and what relaxing each by one unit is worth (LP shadow price, Rs/tHM; the LP's objective)."""
    rows = []
    for name, mu in sorted(r.get("duals", {}).items(), key=lambda kv: -abs(kv[1])):
        if name == "Fe balance":
            continue
        unit = _DUAL_UNITS.get(name, "per %-pt of ferrous" if name.startswith(("min share", "max share")) else "per unit")
        rows.append(dict(limit=name, rs_thm_per_unit=-mu, unit=unit))
    return pd.DataFrame(rows)


def viu_table(eng: Engine, r: Mapping[str, Any]) -> pd.DataFrame:
    """Value in use per tonne of Fe at this plan (purchase + flux + coke for its slag)."""
    s = Settings(**r["settings"])
    rows, _, _ = value_in_use(eng, r["z"], s, r["P_th"])
    T = pd.DataFrame(rows)
    return T[["name", "price", "fe_wet_pct", "buy_per_tFe", "flux_per_tFe", "coke_kg_per_tFe", "fuel_per_tFe",
              "slag_kg_per_tFe", "viu_per_tFe"]].sort_values("viu_per_tFe")


# ---------------------------------------------------------------------------------------------- value of limits
def relax_steps(eng: Engine, L: Limits, s: Settings) -> list[tuple[str, dict[str, Any]]]:
    """One-step relaxations of the limits the case report priced (Limits / Settings / bounds overrides)."""
    steps: list[tuple[str, dict[str, Any]]] = [
        ("Slag cap +10 kg/tHM", dict(L=dict(slag_max=L.slag_max + 10))),
        ("CaO/SiO₂ floor −0.01", dict(L=dict(B2_min=L.B2_min - 0.01))),
        ("Charges per hour +0.1", dict(L=dict(charges_per_h=L.charges_per_h + 0.1))),
        ("Charge mass +0.5 t", dict(L=dict(charge_t=L.charge_t + 0.5))),
        ("Hot blast +40 °C", dict(S=dict(HBT=s.HBT + 40.0))),
    ]
    for v in eng.fer:
        if v.lo > 0:
            steps.append((f"{v.name} floor −2 pt", dict(bounds={v.id: (max(0.0, v.lo - 2), v.hi)})))
        if v.hi < 100:
            steps.append((f"{v.name} cap +5 pt", dict(bounds={v.id: (v.lo, min(100.0, v.hi + 5))})))
    return steps


def _relaxed(eng: Engine, s: Settings, prod_t_day: float, L: Limits, ch: Mapping[str, Any], z0) -> Any:
    """The plan at a fixed production with one limit relaxed (an item of ``Engine.run_items``)."""
    e = eng
    if "bounds" in ch:
        e = Engine(eng.inp, eng.cal, facts=eng.facts, materials=[v.id for v in eng.fer],
                   prices={v.id: v.price for v in eng.vars},
                   bounds={**{v.id: (v.lo, v.hi) for v in eng.fer}, **ch["bounds"]})
    return e._try(replace(s, **ch.get("S", {})), prod_t_day / 24, replace(L, **ch.get("L", {})), True, z0=z0)


def value_of_limits(eng: Engine, prod_t_day: float, s: Settings, L: Limits,
                    base: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Saving in Rs/tHM at a fixed production from relaxing each limit one step (Step 2 rows; ~0.1 s each with
    Fixed-ω). The relaxations do not depend on each other: with STRATA-ω each starts from the same network anchors,
    in parallel with a solver executor (``Engine.run_items``, SV-1.8)."""
    base = base or eng.min_cost(prod_t_day=prod_t_day, s=s, L=L, fuel=True)
    rows = []
    steps = relax_steps(eng, L, s)
    plans = eng.run_items(_relaxed, [(s, prod_t_day, L, ch, base["z"]) for _, ch in steps],
                          network=s.omega_mode == "blend")
    for (label, ch), r in zip(steps, plans):
        if r is None or not r["feasible"]:
            rows.append(dict(change=label, saving_rs_thm=None, note="infeasible"))
            continue
        rows.append(dict(change=label, saving_rs_thm=base["cost"]["total"] - r["cost"]["total"],
                         note=", ".join(b for b in r["binding"] if not b.startswith(("min share", "max share")))))
    return pd.DataFrame(rows).sort_values("saving_rs_thm", ascending=False, na_position="last")
