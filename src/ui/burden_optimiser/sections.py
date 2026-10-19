"""Section renderers for the Burden & Fuel Optimiser page."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import streamlit as st

from data.burden_engine.calibration import CalibrationInfo
from ui.burden_optimiser.figures import frontier_figure, model_check_figure
from utils.burden_engine.engine import Engine, Limits
from utils.burden_engine import model_check as mc
from utils.burden_engine.facts import ENGINE_FACTS, FactRegister
from utils.burden_engine.scenarios import (binding_table, blend_compare_table, cost_table, flux_compare_table,
                                           limit_label, limit_labels, process_table, slag_table, viu_table)

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


# ---------------------------------------------------------------------------------------------- model check
def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _against(model: Any, burnt: Any) -> str:
    try:
        d = float(model) - float(burnt)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(d):
        return ""
    return "same as burnt" if abs(d) < 0.05 else f"{abs(d):.1f} kg/tHM {'above' if d > 0 else 'below'} burnt"


def _shift_name(skey: str) -> str:
    day, shift = str(skey).split("_") if "_" in str(skey) else (str(skey), "")
    try:
        return f"{pd.Timestamp(day):%d %b} · shift {shift}"
    except (ValueError, TypeError):
        return str(skey)


def render_model_check(T: pd.DataFrame | None, cfg: Mapping[str, Any], reg: FactRegister) -> None:
    """Is the energy model right on today's furnace? Latest shift's burnt coke against both balances at that shift's
    HMT and Si, the model's record over recent days, and the trend. Shown before the optimisation."""
    th = mc.Thresholds.from_config(cfg)
    scored = mc.scored_shifts(T)
    st.subheader("Model check · does the energy model match the furnace?")
    if scored.empty:
        basis = reg["DATA-06"].basis if "DATA-06" in reg else ""
        st.info("The model check needs the shift results of the Coke Energy Model page (InfluxDB and Neon), which are "
                f"not available here. The model's validated record against burnt coke (register DATA-06): {basis}.")
        return
    row = mc.latest(scored)
    age = mc.age_days(row, today=pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize())
    st.caption(
        f"Latest shift with a prediction: **{_shift_name(row.name)}**. Burnt coke is what the blast actually burnt; the "
        "two balances predict it for that shift's HMT, Si, blast and burden, calibrated only on the 14 days before "
        "(out of sample). The energy balance is the model the optimiser uses.")
    if age is not None and age > int(cfg.get("stale_after_days", 2)):
        st.warning(f"The latest scored shift is {age} days old: the Coke Energy pipeline has no newer shift with burnt "
                   "coke and a prediction (data gap, or no stable shift since). The check below is for that period.",
                   icon="🕒")
    burnt = row.get("K_burnt")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Burnt coke, kg/tHM", _num(burnt, ".1f"),
              help="Realised: coke the metered blast gasified in the shift (oxygen balance at the metered blast with "
                   "the analyser's ηCO), less the 14-day charged − burnt offset. Dry, per tonne of hot metal.")
    c1.caption(f"what the furnace used · charged {_num(row.get('K_charged'), '.1f')}")
    c2.metric("Energy balance, kg/tHM", _num(row.get("K_energy"), ".1f"),
              help="Two-zone (Rist) balance with shaft efficiency and heat losses held at the 14-day medians: the "
                   "optimiser's model.")
    c2.caption(_against(row.get("K_energy"), burnt))
    c3.metric("Oxygen balance, kg/tHM", _num(row.get("K_obal"), ".1f"),
              help="Oxygen and heat balance at the analyser's ηCO with the 14-day heat-loss allowance.")
    c3.caption(_against(row.get("K_obal"), burnt))
    c4.metric("Hot metal temperature", f"{_num(row.get('HMT'), ',.0f')} °C",
              help="The shift's mean HMT: the thermal state both predictions are made for.")
    c4.caption(f"{_num(row.get('prod_th'), '.1f')} t/h")
    c5.metric("Hot metal Si", f"{_num(row.get('hm_si'), '.2f')} %", help="The shift's mean silicon.")
    c5.caption(f"PCI {_num(row.get('pci_kgthm_dry'), '.0f')} · nut {_num(row.get('nut_kgthm_dry'), '.0f')} kg/tHM dry")

    days_opts = {"7 days": 7, "14 days": 14, "30 days": 30}
    default = next((k for k, v in days_opts.items() if v == int(cfg.get("default_days", 14))), "14 days")
    left, right = st.columns([0.7, 0.3], vertical_alignment="bottom")
    with right:
        span = st.segmented_control("Period", list(days_opts), default=default, required=True, key="bfo_mc_days") or default
    frame = mc.window(scored, days_opts[span])
    e = mc.summary(frame, "K_energy")
    o = mc.summary(frame, "K_obal")
    state, reason = mc.verdict(e, th)
    with left:
        text = f"Energy-balance model over the last {span}: {reason}."
        if o["n"] >= 2:
            text += f" Oxygen balance: bias {o['bias']:+.1f}, typical error ±{o['sd']:.1f} kg/tHM."
        if state == "ok":
            st.success("**On track.** " + text, icon="✅")
        elif state == "watch":
            st.warning("**Check before trusting the plan.** " + text, icon="⚠️")
        else:
            st.info(text)
    st.plotly_chart(model_check_figure(frame), width="stretch", key="bfo_mc_chart", config={"displaylogo": False})
    st.caption(f"Stable shifts are filled points; hollow points failed the stability filter and are not scored. "
               f"On track = bias within ±{th.bias_kgthm:.0f} and typical error within ±{th.sd_kgthm:.0f} kg/tHM "
               "(validated record: ±7.7 per shift, register DATA-06). Full analysis on the Coke Energy Model page.")


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
    "price": st.column_config.NumberColumn("Price Rs/t", format="%,.0f", min_value=0.0),
    "lo": st.column_config.NumberColumn("Min share %", format="%.1f", min_value=0.0, max_value=100.0),
    "hi": st.column_config.NumberColumn("Max share %", format="%.1f", min_value=0.0, max_value=100.0),
    "stock_mt": st.column_config.NumberColumn("Stock t", format="%,.0f", min_value=0.0,
                                              help="Caps t/day (stock over the plan horizon). Blank = no cap."),
    "chem_source": st.column_config.TextColumn("Lab chemistry", disabled=True),
}


def _chem_label(m: Mapping[str, Any]) -> str:
    """'lab, 30-day mean · last sample 14 Oct' from the provider's source tags."""
    src, t = str(m.get("chem_source", "") or ""), str(m.get("chem_time", "") or "")
    when = ""
    if t:
        try:
            when = f" · last sample {pd.Timestamp(t).tz_convert('Asia/Kolkata'):%d %b}" if pd.Timestamp(t).tzinfo \
                else f" · last sample {pd.Timestamp(t):%d %b}"
        except (ValueError, TypeError):
            when = f" · {t}"
    if src.startswith("offline_db_avg"):
        return f"lab, window mean{when}"
    if src.startswith("offline_db_latest"):
        return f"lab, latest{when}"
    return (src or "lab") + when


def _reasons(left: list[Mapping[str, Any]]) -> str:
    reasons = sorted({str(m.get("reason", "")) for m in left if m.get("reason")})
    return ("Reason: " + "; ".join(reasons) + ".") if reasons else ""


def render_yard_editor(yard: Mapping[str, Any], key: str) -> tuple[list[dict], list[dict], dict]:
    """Editable yard (selection, price, shares, stock), fluxes and fuel prices. Returns edited copies."""
    mats = [m for m in yard["materials"] if m.get("carried", True)]
    left = [m for m in yard["materials"] if not m.get("carried", True)]
    df = pd.DataFrame([dict(id=m["id"], selected=m.get("selected", True), name=m.get("name", m["id"]), kind=m["kind"],
                            **{k: m["chem"].get(k, 0.0) for k in ("Fe", "SiO2", "Al2O3", "CaO", "LOI")},
                            moist=m["moist"], price=m["price"], lo=m["lo"], hi=m["hi"], stock_mt=m.get("stock_mt"),
                            chem_source=_chem_label(m)) for m in mats])
    st.markdown("**Iron-bearing materials** · shares are wet % of the ferrous charge")
    ed = st.data_editor(df.set_index("id"), column_config=_YARD_COLS, width="stretch", key=f"{key}_yard",
                        num_rows="fixed", hide_index=True)
    if left:
        st.caption(f"{len(left)} more in the yard list are not offered: "
                   + ", ".join(m.get("name", m["id"]) for m in left)
                   + ". " + _reasons(left))
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
    fed = st.data_editor(fdf.set_index("id"), width="stretch", key=f"{key}_flux", num_rows="fixed", hide_index=True,
                         column_config={"name": st.column_config.TextColumn("Flux", disabled=True),
                                        "price": st.column_config.NumberColumn("Price Rs/t", format="%,.0f"),
                                        "max_qty_mt": st.column_config.NumberColumn("Max t/day", format="%,.0f"),
                                        "stock_mt": st.column_config.NumberColumn("Stock t", format="%,.0f"),
                                        **{c: st.column_config.NumberColumn(f"{lbl} %", format="%.1f", disabled=True)
                                           for c, lbl in (("CaO", "CaO"), ("MgO", "MgO"), ("SiO2", "SiO₂"),
                                                          ("LOI", "LOI"))}})
    fluxes = []
    for f in fl:
        row = fed.loc[f["id"]]
        mq, sk = row["max_qty_mt"], row["stock_mt"]
        fluxes.append(dict(f, price=float(row["price"]), max_qty_mt=None if pd.isna(mq) else float(mq),
                           stock_mt=None if sk is None or pd.isna(sk) else float(sk)))
    st.markdown("**Fuel prices**, Rs/t")
    fu = dict(yard["fuel"])
    c1, c2, c3 = st.columns(3)
    fuel = dict(coke_price=c1.number_input("Coke", value=float(fu["coke_price"]), step=100.0, format="%.0f",
                                           key=f"{key}_pc"),
                nut_price=c2.number_input("Nut coke", value=float(fu["nut_price"]), step=100.0, format="%.0f",
                                          key=f"{key}_pn"),
                pci_price=c3.number_input("PCI", value=float(fu["pci_price"]), step=100.0, format="%.0f",
                                          key=f"{key}_pp"))
    return edited, fluxes, fuel


def render_thermal(thermal: Mapping[str, Any], reg: FactRegister, humidity_default: float, key: str) -> dict:
    hmt_lo, hmt_hi = reg.band("BAND-02")
    si_lo, si_hi = reg.band("BAND-03")
    c1, c2, c3, c4, c5 = st.columns(5)
    out = dict(
        HMT=c1.number_input("Hot metal temperature, °C", min_value=hmt_lo, max_value=hmt_hi,
                            value=float(thermal.get("HMT", 1470.0)), step=5.0, format="%.0f", key=f"{key}_hmt",
                            help="Thermal target the coke rate is solved for (BAND-02)."),
        hm_Si=c2.number_input("Hot metal Si, %", min_value=si_lo, max_value=si_hi, value=float(thermal.get("hm_Si", 0.38)),
                              step=0.01, format="%.2f", key=f"{key}_si", help="BAND-03"),
        pci_plant=c3.number_input("PCI, kg/tHM", min_value=0.0, max_value=260.0, value=float(thermal.get("pci_kg_thm", 190.0)),
                                  step=5.0, format="%.0f", key=f"{key}_pci", help="Fixed input, as injected"),
        nut_plant=c4.number_input("Nut coke, kg/tHM", min_value=0.0, max_value=120.0,
                                  value=float(thermal.get("nut_kg_thm", 70.0)), step=5.0, format="%.0f", key=f"{key}_nut",
                                  help="Fixed input, as charged"),
        humidity_g=c5.number_input("Blast moisture, g/Nm³", min_value=0.0, max_value=40.0, value=float(humidity_default),
                                   step=1.0, format="%.0f", key=f"{key}_hum",
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
    hbt = c[0].number_input("Hot blast temperature, °C", value=float(round(s0["HBT"])), step=5.0, format="%.0f",
                            key=f"{key}_hbt",
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


def render_kpis(r: Mapping[str, Any], other: Mapping[str, Any] | None = None, other_label: str = "",
                today: Mapping[str, Any] | None = None) -> None:
    """Six KPIs. With ``today`` (today's blend at the same production and targets) the cost, coke and slag show the
    change from it; otherwise the cost shows the change from ``other``."""
    o, c = r["out"], r["cost"]
    k = st.columns(6)
    ref, ref_label = (today, "today's blend") if today is not None else (other, other_label)
    diff = None if ref is None else c["total"] - ref["cost"]["total"]
    delta = None if diff is None or abs(diff) < 0.5 else f"{diff:+,.0f} vs {ref_label}"
    k[0].metric(f"Ore + flux + fuel, {RS}", _num(c["total"]), delta, delta_color="inverse")
    if other is not None:
        d0 = c["total"] - other["cost"]["total"]
        k[0].caption(f"Same blend as the {other_label} answer" if abs(d0) < 0.5
                     else f"{other_label}: Rs {other['cost']['total']:,.0f}/tHM")
    k[1].metric("Hot metal, t/d", f"{r['P_tday']:,.0f}")
    dk = None if today is None else o["K"] - today["out"]["K"]
    k[2].metric("Coke dry, kg/tHM", f"{o['K']:.1f}", None if dk is None or abs(dk) < 0.05 else f"{dk:+.1f} vs today",
                delta_color="inverse")
    df_ = None if today is None else o["fuel_rate"] - today["out"]["fuel_rate"]
    k[3].metric("Fuel rate, kg/tHM", f"{o['fuel_rate']:.0f}",
                None if df_ is None or abs(df_) < 0.5 else f"{df_:+.0f} vs today", delta_color="inverse",
                help="Coke wet + nut coke + PCI (BAND-01: 520-600)")
    ds = None if today is None else o["slag"] - today["out"]["slag"]
    k[4].metric("Slag, kg/tHM", f"{o['slag']:.0f}", None if ds is None or abs(ds) < 0.5 else f"{ds:+.0f} vs today",
                delta_color="off")
    k[5].metric("ηCO, %", f"{o['eta']:.1f}", help="Output of the two-zone model at the calibrated shaft efficiency")
    k[5].caption(f"Top gas {o['T_top']:.0f} °C")


def plan_sentence(r: Mapping[str, Any], names: Mapping[str, str], flux_ids: list[str],
                  today: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """(kind, markdown): one line an operator can act on - what to charge, what it costs, what it saves against
    today's blend. kind is "saving", "dearer" or "plain" (no comparison)."""
    fer = sorted(((names.get(i, i), sh) for i, sh in r["share_wet"].items() if sh > 0.05), key=lambda t: -t[1])
    flux = [(names.get(i, i), r["wet_kg"].get(i, 0.0)) for i in flux_ids if r["wet_kg"].get(i, 0.0) >= 0.5]
    text = "**Charge** " + ", ".join(f"{n} {sh:.1f} %" for n, sh in fer)
    text += (" with " + ", ".join(f"{n.lower()} {kg:.0f} kg/tHM" for n, kg in flux)) if flux else ", no flux"
    text += f": Rs {r['cost']['total']:,.0f}/tHM, coke {r['out']['K']:.1f} kg/tHM dry."
    if today is None:
        return "plain", text
    d = today["cost"]["total"] - r["cost"]["total"]
    if abs(d) < 0.5:
        return "plain", text + " Same cost as today's blend."
    lakh = abs(d) * r["P_tday"] / 1e5
    dk = r["out"]["K"] - today["out"]["K"]
    text += (f" **Rs {abs(d):,.0f}/tHM {'less' if d > 0 else 'more'} than today's blend** (Rs {lakh:,.1f} lakh/day at "
             f"this output; coke {dk:+.1f} kg/tHM).")
    return ("saving" if d > 0 else "dearer"), text


def render_plan_tabs(eng: Engine, r: Mapping[str, Any], key: str, other: Mapping[str, Any] | None = None,
                     other_label: str = "ore + flux only", value_of_limits: pd.DataFrame | None = None,
                     today: Mapping[str, Any] | None = None, names: Mapping[str, str] | None = None,
                     prices: Mapping[str, float] | None = None, today_window: str = "over the last 24 h") -> None:
    names = {**{v.id: v.name for v in eng.vars}, **(names or {})}
    prices = {**{v.id: v.price for v in eng.vars}, **(prices or {})}
    tabs_ = ["Blend", "Fuel & cost", "Slag", "Process", "What limits the plan", "Value in use"]
    if value_of_limits is not None:
        tabs_.append("Value of limits")
    tabs = st.tabs(tabs_)
    others = []
    if today is not None:
        others.append(("Today", today))
    if other is not None and abs(other["cost"]["total"] - r["cost"]["total"]) >= 0.5:
        others.append((other_label.capitalize(), other))
    num = st.column_config.NumberColumn
    with tabs[0]:
        B = blend_compare_table(r, others, names, prices)
        cfg = {c: num(c, format="%.1f") for c in B.columns if c.endswith("wet %")}
        cfg.update({c: num(c, format="%,.1f") for c in B.columns if c.endswith("kg/tHM")})
        cfg.update({"Plan, t/d": num("Plan, t/d", format="%,.0f"), "Price, Rs/t": num("Price, Rs/t", format="%,.0f")})
        st.dataframe(B, width="stretch", hide_index=True, column_config=cfg)
        F = flux_compare_table(r, others, names, [v.id for v in eng.flux])
        if F.empty:
            st.caption("No flux needed: the ferrous blend meets the basicity band on its own.")
        else:
            st.dataframe(F, width="stretch", hide_index=True,
                         column_config={c: num(c, format="%.1f") for c in F.columns if c.endswith("kg/tHM")})
        st.caption(f"Ferrous {r['ferrous_wet']:,.0f} kg/tHM wet at {r['fe_wet_pct']:.1f} % Fe; sized so the Fe balance "
                   "closes (hot metal, slag, dust)."
                   + (f" Today = the ferrous blend charged {today_window}, modelled at this plan's output, "
                      "targets and controls with flux to the same basicity band." if today is not None else ""))
    with tabs[1]:
        C = cost_table(r).rename(columns={"item": "Item", "rs_per_thm": RS, "basis": "Basis"})
        if today is not None:
            C.insert(2, f"Today, {RS}", cost_table(today)["rs_per_thm"].values)
        st.dataframe(C, width="stretch", hide_index=True,
                     column_config={c: num(c, format="%,.0f") for c in C.columns if RS in c})
        if other is not None:
            st.caption(f"{other_label.capitalize()}: Rs {other['cost']['total']:,.0f}/tHM, coke {other['out']['K']:.1f} "
                       "kg/tHM dry (same rows, fuel left out of the objective).")
    with tabs[2]:
        fmts = [".0f", ".3f", ".2f", ".2f", ".3f"]          # slag rate, CaO/SiO2, Al2O3 %, MgO %, MgO/Al2O3
        S = slag_table(r).rename(columns={"quantity": "Quantity", "value": "Plan", "limit": "Limit"})
        S["Plan"] = [_num(v, f) for v, f in zip(S["Plan"], fmts)]
        if today is not None:
            S.insert(2, "Today", [_num(v, f) for v, f in zip(slag_table(today)["value"], fmts)])
        st.dataframe(S, width="stretch", hide_index=True)
    with tabs[3]:
        P = process_table(r).rename(columns={"quantity": "Quantity", "value": "Value", "limit": "Limit"})
        st.dataframe(P, width="stretch", hide_index=True, height=35 * (len(P) + 1) + 3)
    with tabs[4]:
        Bt = binding_table(r)
        if Bt.empty:
            st.write("No limit binds.")
        else:
            Bt = Bt.assign(limit=[_cap(limit_label(x, names)) for x in Bt["limit"]]).rename(
                columns={"limit": "Limit", "rs_thm_per_unit": f"Worth, {RS} per unit", "unit": "Unit"})
            st.dataframe(Bt, width="stretch", hide_index=True,
                         column_config={f"Worth, {RS} per unit": num(f"Worth, {RS} per unit", format="%.2f")})
            st.caption("The limits holding this plan back, and what one unit more room on each is worth (LP shadow "
                       "price at the final linearisation).")
    with tabs[5]:
        V = viu_table(eng, r).rename(columns={"name": "Material"})
        st.dataframe(V, width="stretch", hide_index=True, column_config={
            "price": num("Price Rs/t", format="%,.0f"),
            "fe_wet_pct": num("Fe % wet", format="%.1f"),
            "buy_per_tFe": num("Purchase Rs/t Fe", format="%,.0f"),
            "flux_per_tFe": num("Flux Rs/t Fe", format="%,.0f"),
            "coke_kg_per_tFe": num("Coke kg/t Fe", format="%,.0f"),
            "fuel_per_tFe": num("Coke Rs/t Fe", format="%,.0f"),
            "slag_kg_per_tFe": num("Slag kg/t Fe", format="%,.0f"),
            "viu_per_tFe": num("All-in Rs/t Fe", format="%,.0f")})
        st.caption("Each material's own gangue brought to CaO/SiO₂ 1.10 with limestone; coke is the two-zone response "
                   "at this plan. A negative flux or coke figure is a credit: a basic material (sinter) carries lime "
                   "that the rest of the blend would otherwise buy as limestone, and the coke to calcine it. Cheapest "
                   "all-in first.")
    if value_of_limits is not None:
        with tabs[6]:
            VL = value_of_limits.copy()
            VL["note"] = [limit_labels([x.strip() for x in str(n).split(",") if x.strip()], names) if n != "infeasible"
                          else n for n in VL["note"]]
            useful = VL[VL["saving_rs_thm"].fillna(0).abs() >= 0.5]
            st.dataframe(useful.rename(columns={"change": "Change"}), width="stretch", hide_index=True, column_config={
                "saving_rs_thm": num(f"Saving, {RS}", format="%.1f"),
                "note": st.column_config.TextColumn("Binds after the change")})
            rest = VL[VL["saving_rs_thm"].fillna(0).abs() < 0.5]["change"].tolist()
            st.caption("Each limit relaxed by one step at the same production, re-optimised."
                       + (f" No saving from: {', '.join(rest)}." if rest else ""))


def render_frontier(res: Mapping[str, Any]) -> None:
    best, top, F = res["best"], res["top"], res["frontier"]
    k = st.columns(4)
    k[0].metric("Cheapest output, t/d", f"{best['P_tday']:,.0f}", help="Lowest ore + flux + fuel cost per tHM")
    k[0].caption(f"Rs {best['cost']['total']:,.0f}/tHM")
    if top is not None:
        k[1].metric("Highest output, t/d", f"{top['P_tday']:,.0f}")
        k[1].caption(f"Rs {top['cost']['total']:,.0f}/tHM")
        dP = top["P_tday"] - best["P_tday"]
        if dP > 1:
            extra = (top["cost"]["total"] * top["P_tday"] - best["cost"]["total"] * best["P_tday"]) / dP
            k[2].metric("Each extra tonne above it, Rs/t", f"{extra:,.0f}",
                        help="Ore + flux + fuel per extra tonne of hot metal between the two points")
    k[3].markdown("**What limits the cheapest**")
    k[3].markdown(limit_labels(best.get("binding", [])) or "–")
    if top is not None:
        k[3].caption("At the highest output: " + (limit_labels(top.get("binding", [])) or "–"))
    st.plotly_chart(frontier_figure(F, best, top), width="stretch")
    with st.expander("Frontier table"):
        F = F.assign(binding=[limit_labels([x.strip() for x in str(b).split(",") if x.strip()]) for b in F["binding"]])
        st.dataframe(F, width="stretch", hide_index=True, column_config={
            "prod_t_day": st.column_config.NumberColumn("t/d", format="%,.0f"),
            "cost_rs_thm": st.column_config.NumberColumn(RS, format="%,.0f"),
            "ore_flux_rs_thm": st.column_config.NumberColumn("Ore + flux", format="%,.0f"),
            "fuel_rs_thm": st.column_config.NumberColumn("Fuel", format="%,.0f"),
            "coke_dry": st.column_config.NumberColumn("Coke dry", format="%.1f"),
            "fuel_rate": st.column_config.NumberColumn("Fuel rate", format="%.0f"),
            "slag": st.column_config.NumberColumn("Slag", format="%.0f"),
            "HBT": st.column_config.NumberColumn("HBT", format="%.0f"),
            "enrich_tag": st.column_config.NumberColumn("Enrich %", format="%.2f"),
            "blast_nm3h": st.column_config.NumberColumn("Blast", format="%,.0f"),
            "O2_nm3h": st.column_config.NumberColumn("O₂", format="%,.0f"),
            "Pb_needed": st.column_config.NumberColumn("Pb needed", format="%.2f"),
            "charges_per_h": st.column_config.NumberColumn("Charges/h", format="%.2f"),
            "marginal_rs_per_t": st.column_config.NumberColumn("Rs per extra t", format="%,.0f"),
            "binding": st.column_config.TextColumn("Limits")})
    st.caption("Below the cheapest point, fixed heat losses spread over fewer tonnes; above it, a limit forces dearer "
               "ore. Controls at their limits (HBT, pressures); oxygen only where it buys output (it has no price here).")


def render_facts(reg: FactRegister, r: Mapping[str, Any] | None) -> None:
    ids = list(ENGINE_FACTS)
    rows = pd.DataFrame(reg.table(ids))
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(f"Fact register v{reg.version} ({reg.date}). Statuses: Model = follows from the balance; Supported = "
               "BF-02 data agree; Assumed = literature or judgement. Retired facts (COKE-17 webapp hand coefficients, "
               "DATA-10 regressed RAFT) are never used.")
