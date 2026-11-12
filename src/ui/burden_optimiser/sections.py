"""Section renderers for the Burden & Fuel Optimiser page."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import streamlit as st

from data.burden_engine.calibration import CalibrationInfo
from ui.burden_optimiser.figures import (blend_donut, frontier_figure, material_colours, model_check_figure,
                                         transit_lag_figure, transit_window_figure)
from utils.burden_engine.engine import Engine, Limits
from utils.burden_engine import model_check as mc
from utils.burden_engine.facts import ENGINE_FACTS, FactRegister
from utils.coke_energy import transit as tr
from utils.burden_engine.scenarios import (FUEL_LABELS, binding_table, blend_compare_table, cost_table,  # noqa: F401
                                           dust_table, flux_compare_table, fuel_ash_table, limit_label, limit_labels,
                                           material_rows, plan_sentence, plan_tiles, process_table, slag_balance_table,
                                           slag_basis_note, slag_components_table, slag_removals_table,
                                           slag_rollup_table, slag_table, viu_table)
from utils.slag_bands import PLANT_SLAG_BANDS, band_note

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


def render_model_check(T: pd.DataFrame | None, cfg: Mapping[str, Any], reg: FactRegister) -> dict[str, Any]:
    """Is the energy model right on today's furnace? Two tabs: the latest shift's burnt coke against both balances
    at that shift's HMT and Si, with the model's record and trend; and burnt against charged coke (the same coke,
    a descent time apart). Shown before the optimisation. Returns what was shown, for the commentary."""

    th = mc.Thresholds.from_config(cfg)
    scored = mc.scored_shifts(T)
    st.subheader("Model check")
    if scored.empty:
        basis = reg["DATA-06"].basis if "DATA-06" in reg else ""
        st.info("The model check needs the shift results of the Coke Energy Model page (InfluxDB and Neon), which are "
                f"not available here. The model's validated record against burnt coke (register DATA-06): {basis}.")
        return {}
    row = mc.latest(scored)
    age = mc.age_days(row, today=pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize())
    if age is not None and age > int(cfg.get("stale_after_days", 2)):
        st.warning(f"The latest scored shift is {age} days old: the Coke Energy pipeline has no newer shift with burnt "
                   "coke and a prediction (data gap, or no stable shift since). The check below is for that period.",
                   icon="🕒")
    tab_model, tab_transit = st.tabs(["Energy model against burnt coke", "Burnt against charged coke"])
    with tab_model:
        shown = _render_model_vs_burnt(scored, row, cfg, th)
    with tab_transit:
        shown["transit"] = render_transit(T, cfg)
    shown.update(age_days=age)
    return shown


def _render_model_vs_burnt(scored: pd.DataFrame, row: pd.Series, cfg: Mapping[str, Any],
                           th: mc.Thresholds) -> dict[str, Any]:
    st.caption(f"Shift: **{_shift_name(row.name)}**")
    burnt = row.get("K_burnt")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Actual coke rate (burnt), kg/tHM", _num(burnt, ".1f"),
              help="Realised: coke the metered blast gasified in the shift (oxygen balance at the metered blast with "
                   "the analyser's ηCO), less the 14-day charged − burnt offset. Dry, per tonne of hot metal.")
    c1.caption(f"charged {_num(row.get('K_charged'), '.1f')}")
    c2.metric("Modelled coke rate, kg/tHM", _num(row.get("K_energy"), ".1f"),
              help="Two-zone (Rist) energy balance with shaft efficiency and heat losses held at the 14-day medians, "
                   "calibrated only on the 14 days before the shift: the optimiser's model.")
    c2.caption(_against(row.get("K_energy"), burnt))
    c3.metric("Hot metal temperature", f"{_num(row.get('HMT'), ',.0f')} °C",
              help="The shift's mean HMT: the thermal state the model is solved for.")
    c3.caption(f"Si {_num(row.get('hm_si'), '.2f')} % · {_num(row.get('prod_th'), '.1f')} t/h")
    c4.metric("PCI used, kg/tHM", _num(row.get("pci_kgthm_dry"), ".0f"), help="The shift's PCI rate, dry.")
    c5.metric("Nut coke used, kg/tHM", _num(row.get("nut_kgthm_dry"), ".0f"), help="The shift's nut-coke rate, dry.")

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
    return dict(latest=row, latest_name=_shift_name(row.name), period=span, energy=e, oxygen=o, state=state,
                reason=reason)


def render_transit(T: pd.DataFrame | None, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Burnt against charged coke: agreement per averaging window, what does not average out, and the direction of
    the lag. Returns the transit summary (empty when the table has no charged coke)."""

    descent = float(cfg.get("descent_time_h", 6.5))
    S = tr.summary(T) if T is not None and "K_charged" in T else None
    tab = None if S is None else S["table"].set_index("window")
    if tab is None or tab["n"].fillna(0).max() < 3:
        st.info("Charged coke per shift is not in the Coke Energy results here, so burnt and charged coke cannot be "
                "compared.")
        return {}
    st.caption(
        f"Coke tipped at the stockline reaches the tuyeres one burden-descent time later (about {descent:.1f} h on "
        "BF-02). In any window the coke the blast burnt and the coke charged differ by the change in coke in transit "
        "plus measurement error. If they are the same coke the difference must average out as the window grows, with "
        "no bias, and the burnt coke must follow the charged coke, not lead it.")
    shift, day, week = (tab.loc[k] if k in tab.index else None for k in ("Shift (8 h)", "1 day", "1 week"))
    split = S["split"]
    scan = S["scan"].set_index("lag_shifts")
    m = st.columns(4)
    for col, label, r in ((m[0], "Agreement per shift", shift), (m[1], "Per day", day), (m[2], "Per week", week)):
        if r is None or not np.isfinite(r["ccc"]):
            col.metric(label, "–")
            continue
        col.metric(f"{label} (CCC)", f"{r['ccc']:.2f}",
                   help="Lin's concordance correlation coefficient: 1 only when the two series are identical "
                        "(scatter and level). It is Pearson's r times a level-and-scale factor C_b.")
        col.caption(f"bias {r['bias']:+.1f} · 95 % band ±{1.96 * r['sd']:.0f} kg/tHM · {int(r['n'])} windows")
    floor = split.get("floor_sd")
    m[3].metric("Left after averaging", f"±{1.96 * floor:.0f} kg/tHM" if np.isfinite(floor) else "–",
                help="Fit of the window spread: SD² = A/W² (coke in transit) + B/W (random error) + C (floor). The "
                     "floor is what would be a real difference between the two figures or a drifting instrument.")
    if np.isfinite(floor):
        m[3].caption(f"95 % band of the part that does not average out; random part ±{split['random_sd']:.0f} per "
                     "shift")
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.plotly_chart(transit_window_figure(S["table"], split), width="stretch", key="bfo_tr_win",
                        config={"displaylogo": False})
    with right:
        st.plotly_chart(transit_lag_figure(S["scan"]), width="stretch", key="bfo_tr_lag",
                        config={"displaylogo": False})
    r_after = scan["r"].get(1, np.nan)
    r_before = scan["r"].get(-1, np.nan)
    if np.isfinite(r_after) and np.isfinite(r_before):
        st.caption(
            f"Shift-to-shift movements (deviations from a 7-day mean): burnt coke correlates {r_after:.2f} with the "
            f"charged coke one shift earlier and {r_before:.2f} with the charged coke one shift later, so burnt follows "
            f"charged. The peak sits inside the same shift (about {S['peak_lag_h']:.0f} h by interpolation): 8-hour "
            f"shifts that share one production figure cannot resolve a {descent:.1f} h descent; that needs the "
            "charge-dump times hour by hour.")
    clo = S.get("closure")
    if clo and clo.get("n"):
        st.caption(f"Mass closure over {clo['n']} shifts: raw burnt coke (before the 14-day offset) totals "
                   f"{100 * clo['ratio']:.1f} % of the coke charged ({clo['burnt_t']:,.0f} t against "
                   f"{clo['charged_t']:,.0f} t). The offset carries the blast meter's scale error.")
    with st.expander("Window table"):
        show = S["table"][["window", "n", "bias", "sd", "loa_lo", "loa_hi", "r", "ccc", "slope"]].rename(columns={
            "window": "Window", "n": "Windows", "bias": "Bias, kg/tHM", "sd": "SD, kg/tHM", "loa_lo": "95 % low",
            "loa_hi": "95 % high", "r": "r", "ccc": "CCC", "slope": "Slope (Deming)"})
        num = st.column_config.NumberColumn
        st.dataframe(show, width="stretch", hide_index=True, column_config={
            **{c: num(c, format="%+.1f") for c in ("Bias, kg/tHM", "95 % low", "95 % high")},
            "SD, kg/tHM": num("SD, kg/tHM", format="%.1f"), "r": num("r", format="%.3f"),
            "CCC": num("CCC", format="%.3f"), "Slope (Deming)": num("Slope (Deming)", format="%.2f")})
        st.caption("Production-weighted means over non-overlapping windows of the full shift grid (a window needs "
                   "two thirds of its shifts). Burnt coke here carries the 14-day charged − burnt offset, so the bias "
                   "is near zero by construction; the spread, its decay and the lag are independent of the offset.")
    return S


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
    st.markdown("**Fuel prices and slag sale**, Rs/t")
    fu = dict(yard["fuel"])
    c1, c2, c3, c4 = st.columns(4)
    fuel = dict(coke_price=c1.number_input("Coke", value=float(fu["coke_price"]), step=100.0, format="%.0f",
                                           key=f"{key}_pc"),
                nut_price=c2.number_input("Nut coke", value=float(fu["nut_price"]), step=100.0, format="%.0f",
                                          key=f"{key}_pn"),
                pci_price=c3.number_input("PCI", value=float(fu["pci_price"]), step=100.0, format="%.0f",
                                          key=f"{key}_pp"),
                slag_price=c4.number_input("Slag (sale)", value=float(fu.get("slag_price", 0.0) or 0.0), min_value=0.0,
                                           step=50.0, format="%.0f", key=f"{key}_ps",
                                           help="Price the granulated slag sells at (ASM-20). The optimiser credits "
                                                "the slag sold, so of two blends at nearly the same cost the one with "
                                                "more saleable slag is preferred; the slag cap and quality limits stay."))
    return edited, fluxes, fuel


#: Dust-editor columns: the slag-basis key, the column label, the number format and the saved (operator-inputs) key.
_DUST_COLS = (("kg_per_charge", "Dust, kg/charge", "%.0f", "quantity_kg_per_charge"),
              ("CaO_pct", "CaO %", "%.2f", "cao_pct"), ("MgO_pct", "MgO %", "%.2f", "mgo_pct"),
              ("SiO2_pct", "SiO₂ %", "%.2f", "sio2_pct"), ("Al2O3_pct", "Al₂O₃ %", "%.2f", "al2o3_pct"),
              ("Fe_pct", "Fe %", "%.2f", "fe_pct"), ("moist_pct", "Moisture %", "%.1f", "moisture_pct"))


def render_dust_editor(basis: Mapping[str, Any] | None, key: str) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    """The top-gas dust streams of the slag basis (page 9's dust editor, the columns the engine uses): which stream
    is deducted, its kg per charge and analysis. Returns the basis with the edits (applied to this run) and the rows
    in the form the operator inputs persist (``dust_id``, ``enabled``, ``quantity_kg_per_charge``, ``cao_pct`` …), for
    *Save as operator inputs*."""
    if not basis or not basis.get("dust"):
        st.caption("No dust stream configured: nothing is deducted from the slag side.")
        return (dict(basis) if basis else None), pd.DataFrame()
    rows = [d for d in basis["dust"] if d.get("kg_per_charge") is not None]
    df = pd.DataFrame([dict(id=d["id"], enabled=bool(d.get("enabled", True)), name=d["name"],
                            **{k: float(d.get(k) or 0.0) for k, *_ in _DUST_COLS}, source=d.get("source", ""))
                       for d in rows])
    st.markdown("**Top-gas dust** · deducted from the burden before the slag forms, per charge of the plan's blend")
    ed = st.data_editor(df.set_index("id"), width="stretch", key=f"{key}_dust", num_rows="fixed", hide_index=True,
                        column_config={"enabled": st.column_config.CheckboxColumn("Deduct"),
                                       "name": st.column_config.TextColumn("Stream", disabled=True),
                                       "source": st.column_config.TextColumn("Source", disabled=True),
                                       **{k: st.column_config.NumberColumn(lbl, format=fmt, min_value=0.0,
                                                                           max_value=None if k == "kg_per_charge" else 100.0)
                                          for k, lbl, fmt, _ in _DUST_COLS}})
    out, saved = dict(basis), []
    edited = []
    for d in basis["dust"]:
        if d.get("kg_per_charge") is None:
            edited.append(dict(d))
            continue
        row = ed.loc[d["id"]]
        e = dict(d, enabled=bool(row["enabled"]), **{k: float(row[k]) for k, *_ in _DUST_COLS})
        edited.append(e)
        saved.append(dict(dust_id=d["id"], enabled=e["enabled"], rate_basis="kg_per_charge",
                          **{sk: e[k] for k, _, _, sk in _DUST_COLS}))
    out["dust"] = edited
    return out, pd.DataFrame(saved)


def render_fuel_ash_inputs(cal_case: Mapping[str, Any] | None, basis: Mapping[str, Any] | None) -> None:
    """Page 9's fuel-ash inputs as the engine holds them: the calibration's analyses of coke, nut coke and PCI (ash,
    moisture and the ash's oxides from the plant's fuel chemistry) and the minor elements the Blend Mix Optimiser
    configures (alkalis, S, P, MnO). Read-only here; the quantities are the plan's."""
    if not cal_case:
        st.caption("Fuel analyses come with the calibration; none loaded.")
        return
    minor = (basis or {}).get("fuel") or {}
    rows = []
    for key, label in FUEL_LABELS:
        fu = cal_case.get(key) or {}
        a = fu.get("ash_comp") or {}
        mn = minor.get({"coke": "coke", "nut": "nut_coke", "pci": "pci"}[key]) or {}
        rows.append(dict(fuel=label, moist=float(fu.get("moist_pct") or 0.0), ash=float(fu.get("ash") or 0.0),
                         **{k: float(a.get(k) or 0.0) for k in ("SiO2", "Al2O3", "CaO", "MgO", "Fe2O3", "TiO2")},
                         alk=float(mn.get("alk_pct") or 0.0), S=float(mn.get("S_pct") or 0.0), P=float(mn.get("P_pct") or 0.0)))
    st.markdown("**Fuel ash** · the calibration's analyses; alkalis, S and P from the Blend Mix Optimiser's fuel-ash inputs")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
        "fuel": st.column_config.TextColumn("Fuel"), "moist": st.column_config.NumberColumn("Moisture %", format="%.2f"),
        "ash": st.column_config.NumberColumn("Ash %", format="%.2f"),
        **{k: st.column_config.NumberColumn(f"{lbl} % of ash", format="%.2f") for k, lbl in
           (("SiO2", "SiO₂"), ("Al2O3", "Al₂O₃"), ("CaO", "CaO"), ("MgO", "MgO"), ("Fe2O3", "Fe₂O₃"), ("TiO2", "TiO₂"),
            ("alk", "Na₂O + K₂O"))},
        "S": st.column_config.NumberColumn("S % of fuel", format="%.3f"), "P": st.column_config.NumberColumn("P % of fuel", format="%.3f")})


def _pci_input(col: Any, thermal: Mapping[str, Any], key: str) -> float:
    """PCI, kg/tHM as injected: the default (live tag, else the regime median) unless the override is on."""
    default = float(thermal.get("pci_kg_thm", 190.0))
    source = str(thermal.get("pci_kg_thm_source", "config"))
    slot = col.container()               # the input above its toggle, in line with the other inputs
    on = col.toggle("Override PCI", key=f"{key}_pci_on",
                    help=f"Off: the default PCI ({source}), {default:.0f} kg/tHM. On: the entered rate.")
    k = f"{key}_pci"
    if not on or k not in st.session_state:
        st.session_state[k] = default
    return slot.number_input("PCI, kg/tHM", min_value=0.0, max_value=260.0, step=1.0, format="%.0f", key=k,
                            disabled=not on,
                            help=f"Fixed input, as injected. Default: {source}. PCI acts on coke through the balance "
                                 "(carbon, energy, H2).")


def render_thermal(thermal: Mapping[str, Any], reg: FactRegister, humidity_default: Mapping[str, Any] | float,
                   key: str) -> dict:
    hmt_lo, hmt_hi = reg.band("BAND-02")
    si_lo, si_hi = reg.band("BAND-03")
    hd = dict(humidity_default) if isinstance(humidity_default, Mapping) else dict(value=float(humidity_default))
    hum_help = ("Not measured at BF-02 (DATA-04). Default: the Open-Meteo daily relative humidity at Wardha for "
                f"{hd.get('day', 'today')} ({hd.get('source', 'Open-Meteo')}), mapped to 7-21 g/Nm³ (DATA-11), fetched "
                "live and refreshed every 6 h.")
    if hd.get("window") is not None:
        hum_help += f" The calibration was made at its window's Open-Meteo mean, {float(hd['window']):.1f} g/Nm³."
    c1, c2, c3, c4, c5 = st.columns(5)
    out = dict(
        HMT=c1.number_input("Hot metal temperature, °C", min_value=hmt_lo, max_value=hmt_hi,
                            value=float(thermal.get("HMT", 1470.0)), step=5.0, format="%.0f", key=f"{key}_hmt",
                            help="Thermal target the coke rate is solved for (BAND-02)."),
        hm_Si=c2.number_input("Hot metal Si, %", min_value=si_lo, max_value=si_hi, value=float(thermal.get("hm_Si", 0.38)),
                              step=0.01, format="%.2f", key=f"{key}_si",
                              help="Thermal Si the coke rate is solved for (BAND-03). The slag balance holds the "
                                   "Blend Mix Optimiser's Si instead (setting_bmo.yml slag_balance), so both pages "
                                   "report one slag for one blend."),
        pci_plant=_pci_input(c3, thermal, key),
        nut_plant=c4.number_input("Nut coke, kg/tHM", min_value=0.0, max_value=120.0,
                                  value=float(thermal.get("nut_kg_thm", 70.0)), step=1.0, format="%.0f", key=f"{key}_nut",
                                  help="Fixed input, as charged. Default 70 kg/tHM. Acts through the balance and, in "
                                       "STRATA-omega, through the nut-coke term (GAS-23)."),
        humidity_g=c5.number_input("Blast moisture, g/Nm³", min_value=0.0, max_value=40.0, value=round(float(hd["value"]), 1),
                                   step=0.1, format="%.1f", key=f"{key}_hum", help=hum_help),
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


def _badge(r: Mapping[str, Any]) -> str:
    bad = [k for k, v in r.get("viol", {}).items() if v > 1e-3]
    if r.get("feasible", True) and not bad:
        return ":green-badge[:material/check_circle: Meets every limit]"
    return f":red-badge[:material/error: {len(bad)} limit{'s' if len(bad) != 1 else ''} broken]"


def _delta(v: float | None, fmt: str, suffix: str = " vs today", tol: float = 0.05) -> str | None:
    return None if v is None or not np.isfinite(v) or abs(v) < tol else f"{v:{fmt}}{suffix}"


def render_plan_summary(r: Mapping[str, Any], title: str, *, flux_ids: list[str], today: Mapping[str, Any] | None = None,
                        lead: tuple[str, str] | None = None, today_t_day: float | None = None,
                        other: Mapping[str, Any] | None = None, other_label: str = "") -> None:
    """A plan laid out the way page 9 lays out a blend: title with the feasibility badge, an overview strip, then
    the material through the process in four panels - cost, fuel, slag, charging. With ``today`` (today's charged
    blend at the same output and targets) the panels show the change from it."""

    P, T0 = plan_tiles(r, flux_ids), (plan_tiles(today, flux_ids) if today is not None else None)
    d = (lambda grp, k: P[grp][k] - T0[grp][k]) if T0 is not None else (lambda grp, k: None)  # noqa: E731
    st.markdown(f"#### {title} {_badge(r)}")
    if lead is not None:
        (st.success if lead[0] == "saving" else st.info)(lead[1])

    ov = P["overview"]
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Ore + flux + fuel − slag credit (Rs/tHM)" if P["cost"].get("slag_credit") else "Ore + flux + fuel (Rs/tHM)",
              _num(ov["total"]), _delta(d("cost", "total"), "+,.0f", tol=0.5), delta_color="inverse",
              help="Ore, flux and fuel at the prices in the yard editor, less the slag sold at the slag price (ASM-20).")
    if other is not None:
        d0 = ov["total"] - other["cost"]["total"]
        o1.caption(f"Same blend as the {other_label} answer" if abs(d0) < 0.5
                   else f"{other_label}: Rs {other['cost']['total']:,.0f}/tHM")
    o2.metric("Production (t/d)", _num(ov["P_tday"]), help=f"{ov['P_th']:.1f} t/h of hot metal.")
    o3.metric("Burden Fe (%)", _num(ov["fe_wet_pct"], ".1f"),
              help=f"Fe of the ferrous charge, wet; {ov['ferrous_kg']:,.0f} kg ferrous per tHM.")
    if T0 is not None:
        saving = (T0["cost"]["total"] - ov["total"]) * ov["P_tday"] / 1e5
        o4.metric("Against today's blend (Rs lakh/day)", f"{saving:+,.1f}",
                  help="Today's charged blend priced at this output, HMT / Si target and controls; positive = the "
                       "plan saves.")
        o4.caption(f"{T0['cost']['total'] - ov['total']:+,.0f} Rs/tHM at {ov['P_tday']:,.0f} t/d")
    elif today_t_day:
        o4.metric("Δ Production vs today", f"{100 * (ov['P_tday'] / today_t_day - 1):+.1f}%",
                  help="Against today's production (median of the regime window).")
        o4.caption(f"{ov['P_tday'] - today_t_day:+,.0f} t/d ({today_t_day:,.0f} to {ov['P_tday']:,.0f})")
    else:
        o4.metric("Coke, as charged (kg/tHM)", _num(P["fuel"]["coke_wet"], ".2f"),
                  help=f"{P['fuel']['coke_dry']:.2f} kg/tHM dry.")

    c = P["cost"]
    with st.container(border=True):
        st.markdown("###### 💰 &nbsp;Cost &nbsp;<small>Rs / tHM</small>", unsafe_allow_html=True)
        k = st.columns(4)
        for col, label, key in ((k[0], "Ore", "ore"), (k[1], "Flux", "flux"), (k[2], "Fuel", "fuel"),
                                (k[3], "**Total**", "total")):
            col.metric(label, _num(c[key]), _delta(d("cost", key), "+,.0f", tol=0.5), delta_color="inverse")
        st.caption(f"Fuel = coke Rs {c['coke']:,.0f} + nut coke Rs {c['nut']:,.0f} + PCI Rs {c['pci']:,.0f}."
                   + (f" Total = ore + flux + fuel Rs {c['gross']:,.0f} − slag credit Rs {c['slag_credit']:,.0f} "
                      f"({P['slag']['rate']:.0f} kg/tHM sold)." if c.get("slag_credit") else ""))

    f = P["fuel"]
    with st.container(border=True):
        st.markdown("###### 🔥 &nbsp;Fuel rates &nbsp;<small>kg / tHM</small>", unsafe_allow_html=True)
        k = st.columns(4)
        # coke as charged (wet, the client's basis) and two decimals throughout, so the fuels add up as shown
        k[0].metric("Coke (as charged)", _num(f["coke_wet"], ".2f"), _delta(d("fuel", "coke_wet"), "+.2f", tol=0.005),
                    delta_color="inverse", help=f"From the two-zone energy balance; {f['coke_dry']:.2f} kg/tHM dry.")
        k[1].metric("Nut coke", _num(f["nut"], ".2f"), help="Fixed input, as charged.")
        k[2].metric("PCI", _num(f["pci"], ".2f"), help="Fixed input, as injected.")
        k[3].metric("**Total fuel**", _num(f["total"], ".2f"), _delta(d("fuel", "total"), "+.2f", tol=0.005),
                    delta_color="inverse", help="Coke as charged + nut coke + PCI (BAND-01: 520–600).")

    s = P["slag"]
    L = r["limits"]
    with st.container(border=True):
        st.markdown("###### 🌋 &nbsp;Slag", unsafe_allow_html=True)
        k = st.columns(4)
        k[0].metric("Rate (kg/tHM)", _num(s["rate"]), _delta(d("slag", "rate"), "+.0f", tol=0.5), delta_color="off",
                    help=f"Limit ≤ {L['slag_max']:.0f}.")
        k[1].metric("Quantity (t/d)", _num(s["t_day"]))
        k[2].metric("Al₂O₃ (%)", _num(s["Al2O3_pct"], ".2f"), help=f"Limit ≤ {L['Al2O3_max']:.1f}.")
        k[3].metric("MgO (%)", _num(s["MgO_pct"], ".2f"), help=f"Limit ≥ {L['MgO_min']:.1f}.")
        st.markdown("<div style='font-size:0.85em;opacity:0.75;margin-top:2px'>Basicity ratios — against this plant's "
                    "own p5–p95 band</div>", unsafe_allow_html=True)
        k = st.columns(4)
        for col, key, label, helptext in (
                (k[0], "b2", "B2  CaO/SiO₂", f"The ratio this page constrains: {L['B2_min']:.2f}–{L['B2_max']:.2f}."),
                (k[1], "t_basicity", "T-Basicity  (CaO+MgO)/SiO₂", "Display only here (page 9 constrains it)."),
                (k[2], "ib4", "IB4  (CaO+MgO)/(SiO₂+Al₂O₃)",
                 "Display only. Reads lower than T-Basicity because Al₂O₃ is in the denominator.")):
            low, high = PLANT_SLAG_BANDS[key]
            note, inside = band_note(s[key] if np.isfinite(s[key]) else None, low, high)
            col.metric(label, _num(s[key], ".3f"), help=helptext)
            col.markdown(f"<div style='margin-top:-10px;font-size:11px;"
                         f"{'opacity:0.7' if inside else 'color:#b26a00;font-weight:600'}'>{note}</div>",
                         unsafe_allow_html=True)
        k[3].metric("MgO/Al₂O₃", _num(s["mgo_al2o3"], ".3f"), help=f"Mass ratio; limit ≥ {L['MgO_Al2O3_min']:.3f}.")

    g = P["charging"]
    with st.container(border=True):
        st.markdown("###### 🚚 &nbsp;Charging", unsafe_allow_html=True)
        k = st.columns(5)
        k[0].metric("Coke (t/d)", _num(g["coke_t"], ",.1f"), help="Coke rate wet × production.")
        k[1].metric("Nut coke (t/d)", _num(g["nut_t"], ",.1f"), help="Nut coke (as charged) × production.")
        k[2].metric("PCI (t/d)", _num(g["pci_t"], ",.1f"), help="PCI × production.")
        k[3].metric("Charges (/h)", _num(g["charges_per_h"], ".2f"),
                    help=f"Non-coke burden over {L['charge_t']:.2f} t per charge; limit ≤ {L['charges_per_h']:.2f}/h.")
        k[4].metric("Hot metal per charge (t)", _num(g["hm_per_charge"], ".3f"))
        st.caption(f"Ferrous **{g['ferrous_t']:,.0f} t/d** wet · flux rate **{g['flux_kg']:,.1f} kg/tHM**.")
    render_problems(r)


def render_plan_tabs(eng: Engine, r: Mapping[str, Any], key: str, other: Mapping[str, Any] | None = None,
                     other_label: str = "ore + flux only", value_of_limits: pd.DataFrame | None = None,
                     today: Mapping[str, Any] | None = None, names: Mapping[str, str] | None = None,
                     prices: Mapping[str, float] | None = None, today_window: str = "over the last 24 h") -> None:
    names = {**{v.id: v.name for v in eng.vars}, **(names or {})}
    prices = {**{v.id: v.price for v in eng.vars}, **(prices or {})}
    tabs_ = ["🧱 Blend", "🔥 Fuel & cost", "🌋 Slag", "🎛️ Process", "🔒 What limits the plan", "💎 Value in use"]
    if value_of_limits is not None:
        tabs_.append("📈 Value of limits")
    tabs = st.tabs(tabs_)
    others = []
    if today is not None:
        others.append(("Today", today))
    if other is not None and abs(other["cost"]["total"] - r["cost"]["total"]) >= 0.5:
        others.append((other_label.capitalize(), other))
    num = st.column_config.NumberColumn
    with tabs[0]:
        table_col, donut_col = st.columns([3, 2], gap="medium")
        with table_col:
            B = blend_compare_table(r, others, names, prices)
            M = {m["material"]: m for m in material_rows(r, names, prices)}
            detail = any(m["fe_kg"] is not None for m in M.values())
            if detail:   # page 9's blend-table columns: Fe, slag-forming oxides, kg per charge and cost per material
                at = list(B.columns).index("Plan, t/d") + 1
                for j, (col, key) in enumerate((("Fe, kg/tHM", "fe_kg"), ("Slag, kg/tHM", "slag_kg"),
                                                ("Plan, kg/charge", "kg_per_charge"), ("Cost, Rs/tHM", "cost_rs_thm"))):
                    B.insert(at + j, col, [(M.get(n) or {}).get(key) for n in B["Material"]])
            cfg = {c: num(c, format="%.2f") for c in B.columns if c.endswith("wet %")}
            cfg.update({c: num(c, format="%,.2f") for c in B.columns if c.endswith("kg/tHM")})
            cfg.update({"Plan, t/d": num("Plan, t/d", format="%,.0f"), "Plan, kg/charge": num("Plan, kg/charge", format="%,.0f"),
                        "Price, Rs/t": num("Price, Rs/t", format="%,.0f"), "Cost, Rs/tHM": num("Cost, Rs/tHM", format="%,.0f")})
            st.dataframe(B, width="stretch", hide_index=True, column_config=cfg)
            F = flux_compare_table(r, others, names, [v.id for v in eng.flux])
            if F.empty:
                st.caption("No flux needed: the ferrous blend meets the basicity band on its own.")
            else:
                if detail:
                    at = len(F.columns)
                    for j, (col, key) in enumerate((("Slag, kg/tHM", "slag_kg"), ("Plan, kg/charge", "kg_per_charge"),
                                                    ("Cost, Rs/tHM", "cost_rs_thm"))):
                        F.insert(at + j, col, [(M.get(n) or {}).get(key) for n in F["Flux"]])
                st.dataframe(F, width="stretch", hide_index=True,
                             column_config={**{c: num(c, format="%.2f") for c in F.columns if c.endswith("kg/tHM")},
                                            "Plan, kg/charge": num("Plan, kg/charge", format="%,.0f"),
                                            "Cost, Rs/tHM": num("Cost, Rs/tHM", format="%,.0f")})
            if detail:
                st.caption("Fe and slag per material as page 9's blend table: Fe = dry kg × total Fe %, slag = dry kg × "
                           "slag-forming oxides (SiO₂ + Al₂O₃ + CaO + MgO + TiO₂ + MnO + Na₂O + K₂O); kg per charge "
                           "at the plan's hot metal per charge.")
        with donut_col:
            colours = material_colours([v.id for v in eng.vars if v.kind != "flux"])
            which = "Plan"
            if today is not None:
                which = st.segmented_control("Blend shown", ["Plan", "Today"], default="Plan", required=True,
                                             key=f"bfo_donut_{key}", label_visibility="collapsed") or "Plan"
            res = today if which == "Today" and today is not None else r
            ferrous_t = res["ferrous_wet"] * res["P_tday"] / 1000.0
            st.plotly_chart(blend_donut(res["share_wet"], res["t_day"], names, colours,
                                        "Today's charged blend" if res is today else "Plan: share of the ferrous charge",
                                        f"{ferrous_t:,.0f}", "t/d ferrous"),
                            width="stretch", key=f"bfo_donut_fig_{key}", config={"displaylogo": False})
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
        note = slag_basis_note(r)
        if note:
            st.caption(note)
        Cm = slag_components_table(r)
        if not Cm.empty:   # page 9's Slag Balance Details on the plan
            c1, c2 = st.columns([2, 3], gap="medium")
            with c1:
                st.markdown("###### Composition")
                Cm = Cm.rename(columns={"component": "Component", "kg_per_thm": "Plan, kg/tHM", "pct": "Plan, %"})
                if today is not None:
                    T = slag_components_table(today)
                    if not T.empty:
                        Cm["Today, kg/tHM"] = T["kg_per_thm"].values
                        Cm["Today, %"] = T["pct"].values
                st.dataframe(Cm, width="stretch", hide_index=True, height=35 * (len(Cm) + 1) + 3, column_config={
                    **{c: num(c, format="%.2f") for c in Cm.columns if c.endswith("kg/tHM")},
                    **{c: num(c, format="%.2f") for c in Cm.columns if c.endswith("%")}})
                st.markdown("###### Per source")
                Ru = slag_rollup_table(r).rename(columns={"source": "Source", "kg_per_thm": "kg/tHM"})
                st.dataframe(Ru, width="stretch", hide_index=True, height=35 * (len(Ru) + 1) + 3,
                             column_config={"kg/tHM": num("kg/tHM", format="%+.2f")})
            with c2:
                st.markdown("###### Where each component goes, kg/tHM")
                Bl = slag_balance_table(r).rename(columns={
                    "component": "Component", "ore": "Ore", "flux": "Flux", "fuel_ash": "Fuel ash", "dust": "Dust",
                    "hot_metal_gas": "Hot metal, gas", "slag": "Slag"})
                st.dataframe(Bl, width="stretch", hide_index=True, height=35 * (len(Bl) + 1) + 3,
                             column_config={c: num(c, format="%.2f") for c in Bl.columns if c != "Component"})
                st.caption("Fe, Mn and Ti are charged as the element and leave as FeO, MnO and TiO₂ in the slag; the "
                           "dust deduction is per component, never below zero.")
                st.markdown("###### Hot-metal and gas removals")
                Rm = slag_removals_table(r).rename(columns={"removal": "Removal", "rule": "Rule", "kg_per_thm": "kg/tHM"})
                st.dataframe(Rm, width="stretch", hide_index=True, height=35 * (len(Rm) + 1) + 3,
                             column_config={"kg/tHM": num("kg/tHM", format="%.2f")})
            st.markdown("###### Fuels and ash")
            Fa = fuel_ash_table(r).rename(columns={
                "fuel": "Fuel", "wet_kg": "As charged, kg/tHM", "dry_kg": "Dry, kg/tHM", "moist_pct": "Moisture %",
                "ash_pct": "Ash %", "ash_kg": "Ash, kg/tHM", "sio2_pct": "SiO₂ % of ash", "al2o3_pct": "Al₂O₃ % of ash",
                "cao_pct": "CaO % of ash", "mgo_pct": "MgO % of ash", "fe2o3_pct": "Fe₂O₃ % of ash", "mno_pct": "MnO % of ash",
                "tio2_pct": "TiO₂ % of ash", "alk_pct": "Na₂O + K₂O % of ash", "s_pct": "S % of fuel", "p_pct": "P % of fuel",
                "slag_kg": "Slag-forming oxides, kg/tHM"})
            st.dataframe(Fa, width="stretch", hide_index=True, height=35 * (len(Fa) + 1) + 3, column_config={
                **{c: num(c, format="%.2f") for c in Fa.columns if c != "Fuel"},
                "S % of fuel": num("S % of fuel", format="%.3f"), "P % of fuel": num("P % of fuel", format="%.3f")})
            st.caption("Page 9's fuel-ash inputs on the plan: ash and its oxides from the calibration's fuel analyses, "
                       "alkalis, S and P from the Blend Mix Optimiser's configuration; PCI and nut coke are the fixed "
                       "inputs, coke the plan's.")
            st.markdown("###### Top-gas dust")
            Du = dust_table(r).drop(columns=["id"]).rename(columns={
                "stream": "Stream", "kg_per_charge": "kg/charge", "kg_per_thm": "Wet, kg/tHM", "dry_kg": "Dry, kg/tHM",
                "moist_pct": "Moisture %", "sio2_pct": "SiO₂ %", "al2o3_pct": "Al₂O₃ %", "cao_pct": "CaO %", "mgo_pct": "MgO %",
                "fe_pct": "Fe %", "fe_kg": "Fe carried away, kg/tHM", "slag_kg": "Slag-forming oxides carried away, kg/tHM",
                "source": "Source"})
            st.dataframe(Du, width="stretch", hide_index=True, height=35 * (len(Du) + 1) + 3, column_config={
                **{c: num(c, format="%.2f") for c in Du.columns if c not in ("Stream", "Source")},
                "kg/charge": num("kg/charge", format="%.0f")})
            st.caption(f"Dust per tHM = kg per charge × the plan's charges per tHM ({r.get('charges_per_h', 0):.2f} charges/h "
                       f"at {r['P_th']:.1f} t/h); change the kg per charge and the analysis under *Top-gas dust and "
                       "fuel ash* above.")
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
            "coke_wet": st.column_config.NumberColumn("Coke as charged", format="%.2f"),
            "coke_dry": st.column_config.NumberColumn("Coke dry", format="%.2f"),
            "fuel_rate": st.column_config.NumberColumn("Fuel rate", format="%.2f"),
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
               "BF-02 data agree; Validated = tested out of sample; Assumed = literature or judgement; Modelled only = "
               "too few days to test. Retired facts (COKE-17 webapp hand coefficients, DATA-10 regressed RAFT, GAS-17 "
               "gas-capacity channel) are never used.")
    mt = pd.DataFrame(reg.model_table())
    if not mt.empty:
        st.markdown("**Fixed-ω and STRATA-ω side by side** — MOD: coke response at the reference point; VAL: "
                    "out-of-sample burnt coke by blend family (realised against both models).")
        num = st.column_config.NumberColumn
        st.dataframe(mt, width="stretch", hide_index=True, column_config={
            "fixed_omega": num("Fixed-ω", format="%.2f"), "strata_omega": num("STRATA-ω", format="%.2f"),
            "sigma_strata": num("± (STRATA)", format="%.2f"), "realised": num("Realised", format="%.1f"),
            "rmse_fixed": num("RMSE Fixed", format="%.2f"), "rmse_strata": num("RMSE STRATA", format="%.2f"),
            "skill": num("Skill", format="%+.2f")})
