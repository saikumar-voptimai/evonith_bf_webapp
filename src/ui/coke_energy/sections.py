"""Section renderers for the Coke Energy Model page."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

from data.coke_energy.sources import SourceFreshness
from ui.coke_energy.figures import (
    MODELS,
    parity_figure,
    prediction_figure,
    residual_figure,
    rist_animation_figure,
    shift_label,
)
from utils.coke_energy.accuracy import WINDOWS, accuracy, rolling_mean, window_means
from utils.coke_energy.freshness import format_age, format_limit, item_label as _item
from utils.coke_energy.rist_diagram import RistShift

KG = "kg/tHM"
COKE_COLS = ["K_charged", "K_burnt", "K_obal", "K_energy"]
SMOOTHING = {"Per shift": 1, "24 h": 3, "48 h": 6}
# shift end, hours after the shift date
_SHIFT_END_HOURS = {"A": 14, "B": 22, "C": 30}


def _fmt_time(ts: pd.Timestamp | None, fmt: str = "%d %b %H:%M") -> str:
    return "–" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime(fmt)


def _num(value: float | None, fmt: str = ".1f", suffix: str = "") -> str:
    return "–" if value is None or not np.isfinite(value) else f"{value:{fmt}}{suffix}"


def shift_window(row: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """IST start and end of a shift from its table row."""

    end = row.sdate + pd.Timedelta(hours=_SHIFT_END_HOURS[row["shift"]])
    return end - pd.Timedelta(hours=8), end


# --------------------------------------------------------------------------------------------- freshness
def stale_lines(assessed: pd.DataFrame, names: Mapping[str, str], feeds_only: bool = True) -> list[str]:
    """'Lloyds CLO chemistry: last 14 Oct 07:00 (35.9 h, limit 1 day)' for each source past its limit."""

    rows = assessed[assessed.stale & (assessed.feeds_models if feeds_only else True)]
    out = []
    for r in rows.to_dict("records"):
        last = "no data" if r["last"] is None or pd.isna(r["last"]) else f"last {_fmt_time(r['last'])}"
        out.append(f"{_item(r, names)}: {last} ({format_age(r['age'])}, limit {format_limit(r['limit'])})")
    return out


def freshness_table(assessed: pd.DataFrame, names: Mapping[str, str]) -> pd.DataFrame:
    rows = []
    for r in assessed.to_dict("records"):
        rows.append({"Status": "⚠ carried forward" if r["stale"] else "✓ current", "Data": _item(r, names),
                     "Last valid data point": _fmt_time(r["last"], "%d %b %Y %H:%M") if r["last"] is not None else "no data",
                     "Age": format_age(r["age"]), "Limit": format_limit(r["limit"]), "Used by": r["used_by"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- status
def render_status(fresh: SourceFreshness, assessed_now: pd.DataFrame, names: Mapping[str, str],
                  on_refresh: Callable[[], None]) -> bool:
    """Data freshness (as of now), the charged-coke switch and the Refresh button. Returns the switch."""

    left, mid, right = st.columns([0.64, 0.18, 0.18], vertical_alignment="center")
    with left:
        st.caption(
            f"Hourly charges & composition to **{_fmt_time(fresh.hourly_end)}** · 10-min process data to "
            f"**{_fmt_time(fresh.online_end)}** · last HM/slag analysis **{_fmt_time(fresh.hm_slag_end)}** · "
            f"DPR to **{_fmt_time(fresh.dpr_end, '%d %b')}**. A shift is balanced once all four cover it.")
    with mid:
        show_charged = st.toggle("Show charged coke", value=False, key="ce_show_charged",
                                 help="Charged coke is weighed into the top and reaches the tuyeres 4-5 h later, so shift "
                                      "by shift it is a timing signal rather than what the furnace used.")
    with right:
        if st.button("Refresh data", key="ce_refresh", width="stretch",
                     help="Fetch the latest hourly dataset, 10-min data, HM/slag analyses, DPR and lab results."):
            on_refresh()
    lines = stale_lines(assessed_now, names)
    if lines:
        shown = lines[:6]
        more = f" … and {len(lines) - 6} more" if len(lines) > 6 else ""
        st.warning("**Carried forward beyond the limit, as of now:** " + "; ".join(shown) + more + ".")
    with st.expander("Data freshness: every source, as of now"):
        st.dataframe(freshness_table(assessed_now, names), hide_index=True, width="stretch")
        st.caption("Limits: HM/slag 3 h; raw-material chemistry, size analysis, strength and DPR 1 day; fuel ash "
                   "15 days; burden distribution 1 month (plant). Materials are those charged in the last 24 h of "
                   "charge reports; pellet size is fixed, so it is not checked. Daily reports count as current until "
                   "the day after their date ends.")
    return show_charged


# --------------------------------------------------------------------------------------------- latest shift
def render_latest(T: pd.DataFrame, assess_shift: Callable[[str], pd.DataFrame], names: Mapping[str, str],
                  show_charged: bool) -> str | None:
    """The coke rate being used: both models and burnt coke for one shift (the latest by default). Returns its key."""

    cand = T[T.K_obal.notna() | T.K_energy.notna()]
    if cand.empty:
        st.info("No shift has a prediction yet: the models need 14 days of stable shifts before the shift.")
        return None
    options = list(cand.index[::-1])

    def label(k: str) -> str:
        row = T.loc[k]
        tags = [] if row.stable else ["not a stable shift"]
        return shift_label(k) + (f"  ({', '.join(tags)})" if tags else "")

    pick, note = st.columns([0.42, 0.58], vertical_alignment="bottom")
    with pick:
        k = st.selectbox("Shift", options, index=0, format_func=label, key="ce_shift",
                         help="The latest balanced shift is selected. Shifts: A 06-14, B 14-22, C 22-06 IST.")
    row = T.loc[k]
    with note:
        st.caption("Dry coke per tonne of hot metal. Both models are out of sample: their calibration comes only from "
                   "stable shifts in the 14 days before this shift's date. Burnt coke is what the furnace used.")

    burnt = row.K_burnt
    cols = st.columns(4 if show_charged else 3)
    with cols[0]:
        st.metric("Oxygen-balance model", _num(row.K_obal, ".1f", f" {KG}"),
                  help="Oxygen balance and heat balance solved together for coke and blast at the analyser's ηCO, "
                       "with the heat-loss allowance of the previous 14 days.")
        st.caption(f"{_num(row.K_obal - burnt, '+.1f', f' {KG}')} against burnt · ηCO fixed at the analyser's "
                   f"{_num(row.eta_meas, '.2f', ' %')}")
    with cols[1]:
        st.metric("Energy-balance model", _num(row.K_energy, ".1f", f" {KG}"),
                  help="Two-zone (Rist) model: lower-zone heat balance and the reserve-zone limit with shaft efficiency "
                       "ω fixed. ηCO and the top-gas temperature come out of it; the analyser is not used.")
        st.caption(f"{_num(row.K_energy - burnt, '+.1f', f' {KG}')} against burnt · ω {_num(row.omega_fixed, '.3f')} · "
                   f"predicts ηCO {_num(row.eta_energy, '.2f', ' %')}, top gas {_num(row.T_top_energy, '.0f', ' °C')} "
                   f"(measured {_num(row.T_top_meas, '.0f', ' °C')})")
    with cols[2]:
        st.metric("Burnt coke", _num(burnt, ".1f", f" {KG}"),
                  help="The coke the metered blast gasified in the shift: oxygen balance at the metered blast air with "
                       "the analyser's ηCO, less the slow meter/analyser offset (median charged − burnt over the "
                       "previous 14 days).")
        st.caption(f"what the furnace used · meter offset {_num(row.burnt_offset, '+.1f', f' {KG}')}")
    if show_charged:
        with cols[3]:
            st.metric("Charged coke", _num(row.K_charged, ".1f", f" {KG}"),
                      help="Weighed coke charged in the shift, dry, per tonne of hot metal counted from the charges.")
            st.caption("weighed at the top; reaches the tuyeres 4-5 h later")

    lines = stale_lines(assess_shift(k), names)
    if lines:
        st.warning("**For this shift the balance used values carried forward beyond their limit:** " + "; ".join(lines) + ".")
    else:
        st.caption("✓ Every input of this shift was within its freshness limit.")

    with st.expander("Shift inputs and calibration"):
        items = [
            ("Production", row.prod_th, "t/h", ".1f"),
            ("PCI (dry)", row.pci_kgthm_dry, KG, ".1f"),
            ("Nut coke (dry)", row.nut_kgthm_dry, KG, ".1f"),
            ("Hot metal Si", row.hm_si, "%", ".2f"),
            ("Hot metal temperature", row.HMT, "°C", ".0f"),
            ("Slag Al₂O₃", row.get("slag_al2o3_pct", np.nan), "%", ".1f"),
            ("Slag temperature above hot metal (plant rule)", row.get("slag_superheat_c", np.nan), "°C", ".0f"),
            ("Hot blast temperature", row.HBT, "°C", ".0f"),
            ("Blast air, metered", row.blast_air_nm3thm, "Nm³/tHM", ",.0f"),
            ("Blast air, energy model", row.V_energy, "Nm³/tHM", ",.0f"),
            ("ηCO, analyser", row.eta_meas, "%", ".2f"),
            ("ηCO, energy model", row.eta_energy, "%", ".2f"),
            ("Top gas, measured", row.T_top_meas, "°C", ".0f"),
            ("Top gas, energy model", row.T_top_energy, "°C", ".0f"),
            ("ω fixed (14-day median)", row.omega_fixed, "–", ".3f"),
            ("Lower-zone heat loss, fixed", row.Q_LZ_GJh_fixed, "GJ/h", ".1f"),
            ("Upper-zone heat loss, fixed", row.Q_UZ_GJh_fixed, "GJ/h", ".1f"),
            ("Blast moisture (Open-Meteo daily RH, DATA-11)", row.humidity_g, "g/Nm³", ".1f"),
        ]
        if show_charged:
            items.insert(0, ("Charged coke", row.K_charged, KG, ".1f"))
        st.dataframe(pd.DataFrame([(n, _num(v, f), u) for n, v, u, f in items], columns=["Quantity", "Value", "Unit"]),
                     hide_index=True, width="stretch")
    return k


# --------------------------------------------------------------------------------------------- prediction vs burnt
def render_prediction(T: pd.DataFrame, default_days: int, show_charged: bool, highlight: str | None) -> None:
    """Each model against burnt coke over time, the predicted-vs-burnt scatter, the error over time and the table."""

    scored = T[T.K_obal.notna() & T.K_energy.notna() & T.K_burnt.notna()]
    if scored.empty:
        st.info("No shifts with both predictions yet.")
        return
    last_day, first_day = scored.sdate.max().date(), scored.sdate.min().date()
    c1, c2, c3 = st.columns([0.36, 0.40, 0.24], vertical_alignment="bottom")
    with c1:
        period = st.date_input("Period", value=(max(first_day, last_day - timedelta(days=default_days)), last_day),
                               min_value=first_day, max_value=last_day, key="ce_pred_period")
    with c2:
        smoothing = st.segmented_control("Lines", list(SMOOTHING), default="24 h", required=True, key="ce_pred_smooth",
                                         help="24 h and 48 h draw a centred production-weighted mean through the shifts "
                                              "(the shifts stay as faint points); 'Per shift' joins the shifts.") or "24 h"
    with c3:
        with_unstable = st.toggle("Non-stable shifts", value=False, key="ce_pred_unstable",
                                  help="Include shifts that failed the stability filter (upsets, slips, stoppage "
                                       "recovery). The models were validated on stable shifts only.")
    start, end = (period if isinstance(period, (tuple, list)) and len(period) == 2 else (period, period))
    rows = scored[(scored.sdate.dt.date >= start) & (scored.sdate.dt.date <= end)]
    if not with_unstable:
        rows = rows[rows.stable]
    if len(rows) < 2:
        st.info("Not enough shifts in that period.")
        return
    n = SMOOTHING[smoothing]
    raw = rolling_mean(rows, 1, COKE_COLS)
    smooth = rolling_mean(rows, n, COKE_COLS) if n > 1 else None
    hl = T.loc[highlight, "t_mid"] if highlight in rows.index else None
    st.plotly_chart(prediction_figure(raw, smooth, show_charged=show_charged, smoothing_label=f"{smoothing} mean" if n > 1 else "",
                                      highlight=hl), width="stretch", key="ce_pred_chart", config={"displaylogo": False})

    left, right = st.columns(2)
    with left:
        st.markdown("**Predicted against burnt coke, per shift**")
        st.plotly_chart(parity_figure(rows), width="stretch", key="ce_parity", config={"displaylogo": False})
    with right:
        st.markdown("**Model error per shift (model − burnt)**")
        st.plotly_chart(residual_figure(rows), width="stretch", key="ce_residual", config={"displaylogo": False})

    pairs = [(f"{MODELS['K_obal'][1]} vs burnt", "K_obal", "K_burnt"), (f"{MODELS['K_energy'][1]} vs burnt", "K_energy", "K_burnt")]
    if show_charged:
        pairs += [(f"{MODELS['K_obal'][1]} vs charged", "K_obal", "K_charged"),
                  (f"{MODELS['K_energy'][1]} vs charged", "K_energy", "K_charged"),
                  ("Charged vs burnt (timing, not model error)", "K_charged", "K_burnt")]
    by_window = {label: window_means(rows, w, COKE_COLS) if w > 1 else rows for label, w in WINDOWS.items()}
    table = []
    for name, m, a in pairs:
        per_shift = accuracy(rows[m], rows[a])
        entry = {"Comparison": name, "Shifts": per_shift["n"], f"Bias, {KG}": per_shift["bias"], "r": per_shift["r"],
                 "Within ±5": 100 * per_shift["within_5"], "Within ±10": 100 * per_shift["within_10"]}
        for label, frame in by_window.items():
            entry[f"SD {label.lower()}"] = accuracy(frame[m], frame[a])["sd"]
        table.append(entry)
    fmt = {f"Bias, {KG}": "%+.1f", "r": "%.2f", "Within ±5": "%.0f%%", "Within ±10": "%.0f%%"}
    fmt.update({f"SD {label.lower()}": "%.1f" for label in WINDOWS})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format=f) for c, f in fmt.items()})
    st.caption("SD = typical difference in kg/tHM. Windows of 1 day, 2 days and 1 week are production-weighted and "
               "count only when ceil(0.67 × shifts) of their shifts are present (the web model's rule). Judge the models "
               "against burnt coke shift by shift; charged coke only catches up over 2 days or more.")


# --------------------------------------------------------------------------------------------- Rist diagram
def render_rist(T: pd.DataFrame, frames_for: Callable[[tuple[str, ...]], list[RistShift]], default_days: int,
                show_charged: bool) -> None:
    """Animated Rist diagram over the stable shifts of a chosen period."""

    stable = T[T.stable & T.K_energy.notna() & T.K_burnt.notna()]
    if stable.empty:
        st.info("No stable shift with a fixed-ω prediction yet.")
        return
    last_day, first_day = stable.sdate.max().date(), stable.sdate.min().date()
    c1, c2 = st.columns([0.4, 0.6], vertical_alignment="bottom")
    with c1:
        period = st.date_input("Stable shifts from – to",
                               value=(max(first_day, last_day - timedelta(days=default_days)), last_day),
                               min_value=first_day, max_value=last_day, key="ce_rist_period")
    start, end = (period if isinstance(period, (tuple, list)) and len(period) == 2 else (period, period))
    keys = stable[(stable.sdate.dt.date >= start) & (stable.sdate.dt.date <= end)].index
    with c2:
        st.caption(f"{len(keys)} stable shifts. Press ▶ Play, or drag the slider under the chart; the strip below the "
                   "diagram marks the same shift.")
    if len(keys) == 0:
        st.info("No stable shifts in that period.")
        return
    if len(keys) > 360:
        st.warning(f"{len(keys)} shifts selected; showing the latest 360 to keep the animation responsive.")
        keys = keys[-360:]
    frames = frames_for(tuple(keys))
    if not frames:
        st.error("The Rist geometry could not be solved for these shifts.")
        return
    strip = rolling_mean(T.loc[[f.skey for f in frames]], 1, COKE_COLS)
    st.plotly_chart(rist_animation_figure(frames, T, strip, show_charged=show_charged), width="stretch",
                    key="ce_rist_chart", config={"displaylogo": False})
    with st.expander("How to read the Rist diagram"):
        st.markdown(
            "- Both axes count **oxygen atoms per iron atom**. Across is how oxidised the gas is: 0 carbon not yet "
            "gasified, 1 all CO. Up is oxygen per iron atom: from the ore above zero, brought by the blast (with PCI, "
            "the Si/Mn/P/Ti oxides and carbonate CO₂) below zero.\n"
            "- **The navy line is the shift's oxygen balance**, drawn from the coke the blast burnt in the shift. Its "
            "slope is carbon gasified per iron atom: **a steeper line burns more coke**. Where it crosses the vertical "
            "at 1 is the ore oxygen removed directly by coke; its top end is the top gas.\n"
            "- **W** (red star) is the wüstite–iron equilibrium at the 950 °C reserve zone: the most oxygen the gas can "
            "take there. **E** is where the gas actually gets to; shaft efficiency **ω = how far E gets towards W** "
            "(1 = equilibrium). Hydrogen's share is taken out, so W sits a little below wüstite's 1.056.\n"
            "- **P** (diamond) comes from the lower-zone heat balance: every line that balances the lower zone passes "
            "through it. Turning the line about P until it just touches W gives the **ideal line** (grey dotted, "
            "ω = 1), the least coke that heat demand allows.\n"
            "- **Amber dashed: the energy model's line**, with ω and the heat losses fixed at their 14-day medians (its "
            "pivot and E are the hollow amber marks). The gap between the navy and amber lines is the model's error "
            "for that shift. The slider label gives the shift's ω and the three coke rates.")
