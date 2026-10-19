"""Section renderers for the Burden & Fuel Optimiser page."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import streamlit as st

from data.burden_engine.calibration import CalibrationInfo
from ui.burden_optimiser.figures import frontier_figure
from utils.burden_engine.engine import Engine, Limits
from utils.burden_engine.facts import ENGINE_FACTS, FactRegister
from utils.burden_engine.scenarios import (binding_table, blend_table, cost_table, process_table, slag_table,
                                           viu_table)

RS = "Rs/tHM"


def _num(v: Any, fmt: str = ",.0f", suffix: str = "") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(f) else f"{f:{fmt}}{suffix}"


# ---------------------------------------------------------------------------------------------- status
def render_status(yard_source: str, yard_msg: str, info: CalibrationInfo, reg: FactRegister,
                  on_refresh: Callable[[], None]) -> None:
    c1, c2, c3, c4 = st.columns([3, 4, 2, 1])
    if yard_source == "live":
        c1.markdown("**Yard** · live chemistry, stock and saved prices")
    else:
        c1.markdown("**Yard** · reference yard of 16 Oct 2026")
    cal = info.cal
    if info.source == "live":
        c2.markdown(f"**Calibration** · {info.n_shifts} stable shifts, {info.window[0]} to {info.window[1]} · "
                    f"ω {cal['omega']:.3f} · losses {cal['Q_LZ_GJh']:.0f} / {cal['Q_UZ_GJh']:.0f} GJ/h (lower / upper)")
    else:
        c2.markdown(f"**Calibration** · shipped, {cal.get('period', ['', ''])[0]} to {cal.get('period', ['', ''])[1]} · "
                    f"ω {cal['omega']:.3f}")
    c3.markdown(f"**Fact register** · v{reg.version} ({reg.date})")
    if c4.button("Refresh", help="Re-read the yard and the plant data"):
        on_refresh()
    if yard_source != "live":
        st.info(f"{yard_msg} The page is using the reference yard of the case report (16 Oct 2026 UAT prices and "
                "Jul-Oct lab chemistry). Prices, shares and stock can still be edited below.")
    if info.source != "live" and info.message:
        st.info(info.message)


# ---------------------------------------------------------------------------------------------- inputs
_YARD_COLS = {
    "selected": st.column_config.CheckboxColumn("Use", help="Offer this material to the optimiser"),
    "name": st.column_config.TextColumn("Material", disabled=True),
    "kind": st.column_config.TextColumn("Kind", disabled=True),
    "Fe": st.column_config.NumberColumn("Fe %", format="%.1f", disabled=True),
    "SiO2": st.column_config.NumberColumn("SiO₂ %", format="%.2f", disabled=True),
    "Al2O3": st.column_config.NumberColumn("Al₂O₃ %", format="%.2f", disabled=True),
    "CaO": st.column_config.NumberColumn("CaO %", format="%.2f", disabled=True),
    "LOI": st.column_config.NumberColumn("LOI %", format="%.2f", disabled=True),
    "moist": st.column_config.NumberColumn("Moisture %", format="%.1f", disabled=True),
    "price": st.column_config.NumberColumn("Price Rs/t", format="%.0f", min_value=0.0),
    "lo": st.column_config.NumberColumn("Min share %", format="%.1f", min_value=0.0, max_value=100.0),
    "hi": st.column_config.NumberColumn("Max share %", format="%.1f", min_value=0.0, max_value=100.0),
    "stock_mt": st.column_config.NumberColumn("Stock t", format="%.0f", min_value=0.0,
                                              help="Caps t/day (stock over the plan horizon). Blank = no cap."),
    "chem_source": st.column_config.TextColumn("Chemistry", disabled=True),
}


def render_yard_editor(yard: Mapping[str, Any], key: str) -> tuple[list[dict], list[dict], dict]:
    """Editable yard (selection, price, shares, stock), fluxes and fuel prices. Returns edited copies."""
    mats = [m for m in yard["materials"] if m.get("carried", True)]
    left = [m for m in yard["materials"] if not m.get("carried", True)]
    df = pd.DataFrame([dict(id=m["id"], selected=m.get("selected", True), name=m.get("name", m["id"]), kind=m["kind"],
                            **{k: m["chem"].get(k, 0.0) for k in ("Fe", "SiO2", "Al2O3", "CaO", "LOI")},
                            moist=m["moist"], price=m["price"], lo=m["lo"], hi=m["hi"], stock_mt=m.get("stock_mt"),
                            chem_source=f"{m.get('chem_source', '')} {m.get('chem_time', '')}".strip()) for m in mats])
    st.markdown("**Iron-bearing materials** · shares are wet % of the ferrous charge")
    ed = st.data_editor(df.set_index("id"), column_config=_YARD_COLS, width="stretch", key=f"{key}_yard",
                        num_rows="fixed")
    if left:
        st.caption("Left out: " + "; ".join(f"{m.get('name', m['id'])} ({m.get('reason', '')})" for m in left))
    edited = []
    for m in mats:
        row = ed.loc[m["id"]]
        stock = row["stock_mt"]
        edited.append(dict(m, selected=bool(row["selected"]), price=float(row["price"]), lo=float(row["lo"]),
                           hi=float(row["hi"]), stock_mt=(None if stock is None or pd.isna(stock) else float(stock))))
    st.markdown("**Fluxes** · chosen by the optimiser within these limits")
    fl = yard["fluxes"]
    fdf = pd.DataFrame([dict(id=f["id"], name=f.get("name", f["id"]), price=f["price"], max_qty_mt=f.get("max_qty_mt"),
                             stock_mt=f.get("stock_mt"), CaO=f["chem"]["CaO"], MgO=f["chem"]["MgO"],
                             SiO2=f["chem"]["SiO2"], LOI=f["chem"]["LOI"]) for f in fl])
    fed = st.data_editor(fdf.set_index("id"), width="stretch", key=f"{key}_flux", num_rows="fixed",
                         column_config={"name": st.column_config.TextColumn("Flux", disabled=True),
                                        "price": st.column_config.NumberColumn("Price Rs/t", format="%.0f"),
                                        "max_qty_mt": st.column_config.NumberColumn("Max t/day", format="%.0f"),
                                        "stock_mt": st.column_config.NumberColumn("Stock t", format="%.0f"),
                                        **{c: st.column_config.NumberColumn(f"{c} %", format="%.1f", disabled=True)
                                           for c in ("CaO", "MgO", "SiO2", "LOI")}})
    fluxes = []
    for f in fl:
        row = fed.loc[f["id"]]
        mq, sk = row["max_qty_mt"], row["stock_mt"]
        fluxes.append(dict(f, price=float(row["price"]), max_qty_mt=None if pd.isna(mq) else float(mq),
                           stock_mt=None if sk is None or pd.isna(sk) else float(sk)))
    st.markdown("**Fuel prices**, Rs/t")
    fu = dict(yard["fuel"])
    c1, c2, c3 = st.columns(3)
    fuel = dict(coke_price=c1.number_input("Coke", value=float(fu["coke_price"]), step=100.0, key=f"{key}_pc"),
                nut_price=c2.number_input("Nut coke", value=float(fu["nut_price"]), step=100.0, key=f"{key}_pn"),
                pci_price=c3.number_input("PCI", value=float(fu["pci_price"]), step=100.0, key=f"{key}_pp"))
    return edited, fluxes, fuel


def render_thermal(thermal: Mapping[str, Any], reg: FactRegister, humidity_default: float, key: str) -> dict:
    hmt_lo, hmt_hi = reg.band("BAND-02")
    si_lo, si_hi = reg.band("BAND-03")
    c1, c2, c3, c4, c5 = st.columns(5)
    out = dict(
        HMT=c1.number_input("Hot metal temperature, °C", min_value=hmt_lo, max_value=hmt_hi,
                            value=float(thermal.get("HMT", 1470.0)), step=5.0, key=f"{key}_hmt",
                            help="Thermal target the coke rate is solved for (BAND-02)."),
        hm_Si=c2.number_input("Hot metal Si, %", min_value=si_lo, max_value=si_hi, value=float(thermal.get("hm_Si", 0.38)),
                              step=0.01, format="%.2f", key=f"{key}_si", help="BAND-03"),
        pci_plant=c3.number_input("PCI, kg/tHM", min_value=0.0, max_value=260.0, value=float(thermal.get("pci_kg_thm", 190.0)),
                                  step=5.0, key=f"{key}_pci", help="Fixed input, as injected"),
        nut_plant=c4.number_input("Nut coke, kg/tHM", min_value=0.0, max_value=120.0,
                                  value=float(thermal.get("nut_kg_thm", 70.0)), step=5.0, key=f"{key}_nut",
                                  help="Fixed input, as charged"),
        humidity_g=c5.number_input("Blast moisture, g/Nm³", min_value=0.0, max_value=40.0, value=float(humidity_default),
                                   step=1.0, key=f"{key}_hum",
                                   help="Not measured at BF-02 (DATA-04): defaults to the month's typical ambient value."),
    )
    return out


def render_limits(L0: Limits, regime: Mapping[str, Any] | None, key: str, show_process: bool) -> Limits:
    """Day limits with today's 14-day medians alongside. Returns the edited Limits."""
    reg = dict(regime or {})
    d: dict[str, Any] = {}
    if show_process:
        st.markdown("**Blast and gas** · today's value is the median of the regime window")
        rows = [("V_max", "Hot blast max, Nm³/h", "hot_blast_nm3h", 500.0, ",.0f", "%.0f"),
                ("O2_flow_max", "Oxygen max, Nm³/h", "O2_nm3h", 50.0, ",.0f", "%.0f"),
                ("enrich_max", "Enrichment tag max, %", "enrich_tag", 0.1, ".2f", "%.2f"),
                ("HBT_max", "Hot blast temperature max, °C", "HBT", 5.0, ",.0f", "%.0f"),
                ("Pb_max", "Hot-blast pressure max, bar g", "Pb", 0.01, ".2f", "%.2f"),
                ("dP_max", "Pressure drop max, bar", "dP", 0.01, ".2f", "%.2f"),
                ("Pt_min", "Top pressure min, bar g", "Pt", 0.01, ".2f", "%.2f"),
                ("Pt_max", "Top pressure max, bar g", "Pt", 0.01, ".2f", "%.2f"),
                ("RAFT_min", "RAFT min, °C", "raft_dcs", 10.0, ",.0f", "%.0f"),
                ("RAFT_max", "RAFT max, °C", "raft_dcs", 10.0, ",.0f", "%.0f")]
        cols = st.columns(4)
        for i, (field, label, reg_key, step, fmt, nfmt) in enumerate(rows):
            today = reg.get(reg_key)
            d[field] = cols[i % 4].number_input(label, value=float(getattr(L0, field)), step=step, format=nfmt,
                                                key=f"{key}_{field}",
                                                help=f"Today: {_num(today, fmt)}" if today is not None else None)
    st.markdown("**Charging and slag**")
    c = st.columns(4)
    d["charge_t"] = c[0].number_input("Non-coke t per charge", value=float(L0.charge_t), step=0.1, format="%.2f",
                                      key=f"{key}_ct")
    d["charges_per_h"] = c[1].number_input("Charges per hour, max", value=float(L0.charges_per_h), step=0.05,
                                           format="%.2f", key=f"{key}_ch")
    d["slag_max"] = c[2].number_input("Slag rate max, kg/tHM", value=float(L0.slag_max), step=5.0, format="%.0f",
                                      key=f"{key}_slag")
    d["T_top_min"] = c[3].number_input("Top gas min, °C", value=float(L0.T_top_min), step=5.0, format="%.0f",
                                       key=f"{key}_ttop")
    c = st.columns(4)
    d["B2_min"] = c[0].number_input("CaO/SiO₂ min", value=float(L0.B2_min), step=0.01, format="%.2f", key=f"{key}_b2lo")
    d["B2_max"] = c[1].number_input("CaO/SiO₂ max", value=float(L0.B2_max), step=0.01, format="%.2f", key=f"{key}_b2hi")
    d["Al2O3_max"] = c[2].number_input("Al₂O₃ max, %", value=float(L0.Al2O3_max), step=0.5, format="%.1f", key=f"{key}_al")
    d["MgO_min"] = c[3].number_input("MgO min, %", value=float(L0.MgO_min), step=0.1, format="%.1f", key=f"{key}_mg")
    c = st.columns(4)
    d["MgO_Al2O3_min"] = c[0].number_input("MgO/Al₂O₃ min", value=float(L0.MgO_Al2O3_min), step=0.005, format="%.3f",
                                           key=f"{key}_mgal")
    out = Limits(**{**L0.__dict__, **{k: float(v) for k, v in d.items()}})
    return out


def render_controls(s0: Mapping[str, Any], hy_tag: Callable[[float], float], tag_to_r: Callable[[float], float],
                    regime: Mapping[str, Any] | None, key: str) -> dict:
    """Controls held for a fixed-production plan (start: today's medians)."""
    reg = dict(regime or {})
    c = st.columns(4)
    hbt = c[0].number_input("Hot blast temperature, °C", value=float(s0["HBT"]), step=5.0, key=f"{key}_hbt",
                            help=f"Today: {_num(reg.get('HBT'))}")
    tag = c[1].number_input("Enrichment tag, %", value=float(hy_tag(s0["O2_per_air"])), step=0.05, format="%.2f",
                            key=f"{key}_tag", help=f"Today: {_num(reg.get('enrich_tag'), '.2f')}")
    pb = c[2].number_input("Hot-blast pressure, bar g", value=float(s0["Pb"]), step=0.01, format="%.2f", key=f"{key}_pb",
                           help="Only reported: the blast the pressures allow, against the blast the plan needs.")
    pt = c[3].number_input("Top pressure, bar g", value=float(s0["Pt"]), step=0.01, format="%.2f", key=f"{key}_pt")
    return dict(HBT=hbt, O2_per_air=tag_to_r(tag), Pb=pb, Pt=pt)


# ---------------------------------------------------------------------------------------------- results
def render_problems(r: Mapping[str, Any]) -> None:
    if not r["feasible"]:
        bad = {k: v for k, v in r["viol"].items() if v > 1e-3}
        st.error("No plan meets every limit. Closest plan shown; it breaks: "
                 + ", ".join(f"{k} (by {v:,.2f})" for k, v in bad.items()) + ". Relax a limit or the share bounds.")
    for w in r.get("warnings", []):
        st.warning(w)


def render_kpis(r: Mapping[str, Any], other: Mapping[str, Any] | None = None, other_label: str = "") -> None:
    o, c = r["out"], r["cost"]
    k = st.columns(6)
    diff = None if other is None else c["total"] - other["cost"]["total"]
    delta = None if diff is None or abs(diff) < 0.5 else f"{diff:+,.0f} vs {other_label}"
    k[0].metric(f"Ore + flux + fuel, {RS}", _num(c["total"]), delta, delta_color="inverse")
    if diff is not None and delta is None:
        k[0].caption(f"Same blend as the {other_label} answer")
    k[1].metric("Hot metal", f"{r['P_tday']:,.0f} t/d")
    k[2].metric("Coke, dry", f"{o['K']:.1f} kg/tHM")
    k[3].metric("Fuel rate", f"{o['fuel_rate']:.0f} kg/tHM", help="Coke wet + nut coke + PCI (BAND-01: 520-600)")
    k[4].metric("Slag", f"{o['slag']:.0f} kg/tHM")
    k[5].metric("ηCO", f"{o['eta']:.1f} %", help="Output of the two-zone model at the calibrated shaft efficiency")
    k[5].caption(f"Top gas {o['T_top']:.0f} °C")


def render_plan_tabs(eng: Engine, r: Mapping[str, Any], key: str, other: Mapping[str, Any] | None = None,
                     other_label: str = "ore + flux only", value_of_limits: pd.DataFrame | None = None) -> None:
    names = ["Blend", "Fuel & cost", "Slag", "Process", "Limits & shadow prices", "Value in use"]
    if value_of_limits is not None:
        names.append("Value of limits")
    tabs = st.tabs(names)
    with tabs[0]:
        T = blend_table(eng, r, other, other_label)
        st.dataframe(T, width="stretch", hide_index=True, column_config={
            "share_wet_pct": st.column_config.NumberColumn("Share, wet %", format="%.1f"),
            "kg_per_thm_wet": st.column_config.NumberColumn("kg/tHM wet", format="%.1f"),
            "t_per_day": st.column_config.NumberColumn("t/day", format="%.0f"),
            "price_rs_t": st.column_config.NumberColumn("Price Rs/t", format="%.0f"),
            f"share_{other_label}": st.column_config.NumberColumn(f"Share, {other_label}", format="%.1f"),
            f"kg_{other_label}": st.column_config.NumberColumn(f"kg/tHM, {other_label}", format="%.1f")})
        st.caption(f"Ferrous {r['ferrous_wet']:,.0f} kg/tHM wet at {r['fe_wet_pct']:.1f} % Fe; sized so the Fe balance "
                   "closes (hot metal, slag, dust).")
    with tabs[1]:
        st.dataframe(cost_table(r), width="stretch", hide_index=True, column_config={
            "rs_per_thm": st.column_config.NumberColumn(RS, format="%.0f")})
        if other is not None:
            st.caption(f"{other_label}: Rs {other['cost']['total']:,.0f}/tHM, coke {other['out']['K']:.1f} kg/tHM dry. "
                       "Same rows, fuel left out of the objective.")
    with tabs[2]:
        st.dataframe(slag_table(r), width="stretch", hide_index=True,
                     column_config={"value": st.column_config.NumberColumn(format="%.3f")})
        flux = {v.name: r["wet_kg"][v.id] for v in eng.flux if r["wet_kg"][v.id] > 0.05}
        st.caption("Flux in the furnace: " + (", ".join(f"{k} {v:.1f} kg/tHM" for k, v in flux.items()) or "none"))
    with tabs[3]:
        st.dataframe(process_table(r), width="stretch", hide_index=True)
    with tabs[4]:
        B = binding_table(r)
        if B.empty:
            st.write("No limit binds.")
        else:
            st.dataframe(B, width="stretch", hide_index=True, column_config={
                "rs_thm_per_unit": st.column_config.NumberColumn(f"{RS} per unit relaxed", format="%.3f")})
            st.caption("LP shadow prices at the final linearisation: what one unit more room on each limit is worth in "
                       "the objective.")
    with tabs[5]:
        st.dataframe(viu_table(eng, r), width="stretch", hide_index=True, column_config={
            "price": st.column_config.NumberColumn("Price Rs/t", format="%.0f"),
            "fe_wet_pct": st.column_config.NumberColumn("Fe % wet", format="%.1f"),
            "buy_per_tFe": st.column_config.NumberColumn("Purchase Rs/t Fe", format="%.0f"),
            "flux_per_tFe": st.column_config.NumberColumn("Flux Rs/t Fe", format="%.0f"),
            "coke_kg_per_tFe": st.column_config.NumberColumn("Coke kg/t Fe", format="%.0f"),
            "fuel_per_tFe": st.column_config.NumberColumn("Coke Rs/t Fe", format="%.0f"),
            "slag_kg_per_tFe": st.column_config.NumberColumn("Slag kg/t Fe", format="%.0f"),
            "viu_per_tFe": st.column_config.NumberColumn("All-in Rs/t Fe", format="%.0f")})
        st.caption("Each material's own gangue brought to CaO/SiO₂ 1.10 with limestone; coke is the two-zone response "
                   "at this plan. Cheapest all-in first.")
    if value_of_limits is not None:
        with tabs[6]:
            st.dataframe(value_of_limits, width="stretch", hide_index=True, column_config={
                "saving_rs_thm": st.column_config.NumberColumn(f"Saving, {RS}", format="%.1f"),
                "note": st.column_config.TextColumn("Binds after the change")})
            st.caption("Each limit relaxed by one step at the same production, re-optimised.")


def render_frontier(res: Mapping[str, Any]) -> None:
    best, top, F = res["best"], res["top"], res["frontier"]
    k = st.columns(4)
    k[0].metric("Cheapest output", f"{best['P_tday']:,.0f} t/d", help="Lowest ore + flux + fuel cost per tHM")
    k[0].caption(f"Rs {best['cost']['total']:,.0f}/tHM")
    if top is not None:
        k[1].metric("Highest output", f"{top['P_tday']:,.0f} t/d")
        k[1].caption(f"Rs {top['cost']['total']:,.0f}/tHM")
        dP = top["P_tday"] - best["P_tday"]
        if dP > 1:
            extra = (top["cost"]["total"] * top["P_tday"] - best["cost"]["total"] * best["P_tday"]) / dP
            k[2].metric("Each extra tonne above the cheapest", f"Rs {extra:,.0f}/t",
                        help="Ore + flux + fuel per extra tonne of hot metal between the two points")
    binds = [b for b in best.get("binding", []) if not b.startswith(("min share", "max share"))]
    k[3].markdown("**What binds at the cheapest**")
    k[3].markdown(", ".join(binds) or "–")
    if top is not None:
        tb = [b for b in top.get("binding", []) if not b.startswith(("min share", "max share"))]
        k[3].caption("At the highest output: " + (", ".join(tb) or "–"))
    st.plotly_chart(frontier_figure(F, best, top), width="stretch")
    with st.expander("Frontier table"):
        st.dataframe(F, width="stretch", hide_index=True, column_config={
            "prod_t_day": st.column_config.NumberColumn("t/d", format="%.0f"),
            "cost_rs_thm": st.column_config.NumberColumn(RS, format="%.0f"),
            "ore_flux_rs_thm": st.column_config.NumberColumn("Ore + flux", format="%.0f"),
            "fuel_rs_thm": st.column_config.NumberColumn("Fuel", format="%.0f"),
            "coke_dry": st.column_config.NumberColumn("Coke dry", format="%.1f"),
            "fuel_rate": st.column_config.NumberColumn("Fuel rate", format="%.0f"),
            "slag": st.column_config.NumberColumn("Slag", format="%.0f"),
            "HBT": st.column_config.NumberColumn("HBT", format="%.0f"),
            "enrich_tag": st.column_config.NumberColumn("Enrich %", format="%.2f"),
            "blast_nm3h": st.column_config.NumberColumn("Blast", format="%.0f"),
            "O2_nm3h": st.column_config.NumberColumn("O₂", format="%.0f"),
            "Pb_needed": st.column_config.NumberColumn("Pb needed", format="%.2f"),
            "charges_per_h": st.column_config.NumberColumn("Charges/h", format="%.2f"),
            "marginal_rs_per_t": st.column_config.NumberColumn("Rs per extra t", format="%.0f"),
            "binding": st.column_config.TextColumn("Binds")})
    st.caption("Below the cheapest point, fixed heat losses spread over fewer tonnes; above it, a limit forces dearer "
               "ore. Controls at their limits (HBT, pressures); oxygen only where it buys output (it has no price here).")


def render_facts(reg: FactRegister, r: Mapping[str, Any] | None) -> None:
    ids = list(ENGINE_FACTS)
    rows = pd.DataFrame(reg.table(ids))
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(f"Fact register v{reg.version} ({reg.date}). Statuses: Model = follows from the balance; Supported = "
               "BF-02 data agree; Assumed = literature or judgement. Retired facts (COKE-17 webapp hand coefficients, "
               "DATA-10 regressed RAFT) are never used.")
