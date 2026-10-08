"""Operator view of the charged-coke engine: next 5 hours, last 7 days, retrain.

Three tabs:

* Next 5 hours: status, the forecast path (+1..+5 h, each with its range) over
  the last day of measured 4-h charged coke rate.
* Last 7 days: what the active model forecast against what was charged, one
  horizon at a time, markers coloured by furnace condition. Paused hours
  (large deviations, unfamiliar conditions, missing inputs) are left out, as
  operators never saw a forecast for them.
* Model & retrain: the active model's report, a background retrain, review of
  the candidate, and accept / roll back with who and when.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.bmo.charged_coke.contract import ForecastWindow
from utils.bmo.charged_coke import deployments

TEAL, NAVY = "#07827f", "#25344b"
STATE_COLORS = {"cruise": "#2f9e44", "unsettled": "#f08c00", "off_cruise": "#d9480f"}
STATE_LABELS = {
    "cruise": "Steady",
    "unsettled": "Variable",
    "off_cruise": "Outside normal",
}
PAUSE_WORDS = {
    "large_hold": "large deviation", "novel_hold": "unfamiliar conditions",
    "uncertainty_hold": "range too uncertain", "planned_change_hold": "planned change",
    "recovering": "recovering", "data_hold": "inputs incomplete", "data_missing": "hour missing",
}


def _when(value: Any) -> str:
    return f"{pd.Timestamp(value):%d %b %H:%M}" if value else "unknown"


# ---- status ------------------------------------------------------------------


def _render_technical_status(prediction: Any, detail: dict[str, Any]) -> None:
    """State, the +5 h level the optimiser uses, reasons and provenance."""

    record = detail.get("record") or {}
    state = str(record.get("state", ""))
    badge = {
        "cruise": ":green-badge[Cruising]",
        "unsettled": ":orange-badge[Wider range: recent instability]",
        "off_cruise": ":orange-badge[Wider range: outside cruise]",
    }.get(state, f":red-badge[Paused: {PAUSE_WORDS.get(state, state.replace('_', ' '))}]")
    reasons = [r for r in (record.get("reasons") or []) if r]
    if prediction.usable:
        window = ForecastWindow.for_row(pd.Timestamp(record["data_row"]), pd.Timestamp(record["issued_at"]))
        text = window.describe()
        st.success(
            f"Charged-coke forecast {badge}: **{prediction.value_kg_per_thm:,.1f} kg/THM** in 5 h "
            f"({text[:1].lower() + text[1:]}). The optimiser uses this +5 h value. Range "
            f"{record.get('upper_kg_thm', 0) - record.get('prediction_kg_thm', 0):,.0f} kg/THM either side: 90% of "
            "past forecast errors in this condition."
        )
        if detail.get("override"):
            effects = detail["override"].get("effects", {})
            st.caption(
                f"Includes the operator's PCI/nut-coke change, applied once with the "
                f"{detail['override'].get('mode')} response: PCI {effects.get('pci', 0):+.1f}, nut coke "
                f"{effects.get('nut', 0):+.1f}, slag {effects.get('slag', 0):+.2f} kg/THM. The range is for "
                "the live forecast, not for this change."
            )
    else:
        st.error(f"Charged-coke forecast {badge}. No current Data-Driven level.")
        last = record.get("last_valid")
        if last:
            st.caption(
                f"Last valid forecast: {last.get('prediction_kg_thm', 0):,.1f} kg/THM for coke charged "
                f"{pd.Timestamp(last['coke_window'][0]):%H:%M}-{pd.Timestamp(last['coke_window'][1]):%H:%M}, from "
                f"the {_when(last['data_row'])} row. Reference only; it is not moved to a new window."
            )
        st.info("Physics-Driven can be selected as a separate calculation if its own inputs pass. "
                "It does not inherit this forecast's range or validation.")
    for reason in reasons[:4]:
        st.caption(f"• {reason}")
    st.caption(
        f"Data to the {_when(record.get('data_row'))} row (complete {_when(record.get('data_complete_at'))}), "
        f"issued {_when(record.get('issued_at'))} IST; model {prediction.deployment_id}; "
        "conditional on current operation continuing."
    )


# ---- next 5 hours ------------------------------------------------------------


def _realised(recent: pd.DataFrame) -> pd.Series:
    """Measured 4-h charged coke rate, at the end of each charge window."""

    if recent is None or recent.empty:
        return pd.Series(dtype=float)
    one = recent[recent["horizon"] == recent["horizon"].min()]
    series = one.set_index("at")["actual"].dropna().sort_index()
    series = series[~series.index.duplicated()]
    if series.empty:
        return series
    # On the hourly clock, so a missing hour breaks the line instead of
    # being drawn over.
    return series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="h"))


def path_figure(record: dict[str, Any], recent: pd.DataFrame) -> go.Figure:
    """Last 24 h measured, then the five forecast points with their ranges."""

    fig = go.Figure()
    realised = _realised(recent)
    issued = pd.Timestamp(record["issued_at"])
    realised = realised[realised.index >= issued - pd.Timedelta(hours=24)]
    if len(realised):
        fig.add_trace(go.Scatter(x=realised.index, y=realised, mode="lines+markers", name="Measured (4-h charged)",
                                 line=dict(color=NAVY, width=1.5), marker=dict(size=5, color=NAVY),
                                 hovertemplate="%{x|%d %b %H:%M}: %{y:.1f} kg/THM<extra>measured</extra>"))
    path = [p for p in record.get("path") or [] if p.get("prediction_kg_thm") is not None]
    if path:
        at = [pd.Timestamp(p["at"]) for p in path]
        value = [p["prediction_kg_thm"] for p in path]
        low = [p["lower_kg_thm"] for p in path]
        high = [p["upper_kg_thm"] for p in path]
        has_range = all(
            bound is not None and np.isfinite(float(bound))
            for bound in [*low, *high]
        )
        if has_range:
            fig.add_trace(go.Scatter(x=at + at[::-1], y=high + low[::-1], fill="toself", fillcolor="rgba(7,130,127,0.14)",
                                     mode="lines", line=dict(width=0), hoverinfo="skip", name="Range (90% of past errors)"))
        last = realised.dropna()
        if len(last):
            fig.add_trace(go.Scatter(x=[last.index[-1], at[0]], y=[last.iloc[-1], value[0]], mode="lines",
                                     line=dict(color=TEAL, dash="dot", width=1), showlegend=False, hoverinfo="skip"))
        forecast_trace = dict(
            x=at,
            y=value,
            mode="lines+markers+text",
            name="Forecast",
            line=dict(color=TEAL, dash="dash", width=2),
            marker=dict(size=9, color=TEAL),
            text=[f"+{p['horizon']} h<br>{v:.0f}" for p, v in zip(path, value)],
            textposition="top center",
        )
        if has_range:
            forecast_trace.update(
                customdata=np.column_stack([low, high]),
                hovertemplate="%{x|%d %b %H:%M}: %{y:.1f} kg/THM (%{customdata[0]:.0f}-%{customdata[1]:.0f})<extra>forecast</extra>",
            )
        else:
            forecast_trace["hovertemplate"] = (
                "%{x|%d %b %H:%M}: %{y:.1f} kg/THM"
                "<extra>forced forecast; range not validated</extra>"
            )
        fig.add_trace(go.Scatter(**forecast_trace))
    fig.add_vline(x=issued, line=dict(color=NAVY, width=1, dash="dot"))
    fig.add_annotation(x=issued, y=1, yref="paper", text=f"issued {issued:%H:%M}", showarrow=False, yshift=10,
                       font=dict(size=11, color=NAVY))
    _style(fig, height=360)
    return fig


# ---- last 7 days -------------------------------------------------------------


def trust_figure(recent: pd.DataFrame, horizon: int) -> go.Figure:
    """Forecast (coloured by condition) against measured, one horizon."""

    fig = go.Figure()
    realised = _realised(recent)
    fig.add_trace(go.Scatter(x=realised.index, y=realised, mode="lines", name="Measured (4-h charged)",
                             line=dict(color=NAVY, width=1.4),
                             hovertemplate="%{x|%d %b %H:%M}: %{y:.1f} kg/THM<extra>measured</extra>"))
    rows = recent[(recent["horizon"] == horizon) & recent["shown"]]
    for state, colour in STATE_COLORS.items():
        part = rows[rows["state"] == state]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["at"], y=part["prediction"], mode="markers", name=f"Forecast · {STATE_LABELS[state]}",
            marker=dict(size=7, color=colour, line=dict(width=0.5, color="white")),
            customdata=np.column_stack([part["lower"], part["upper"], part["actual"], part["issue_row"].dt.strftime("%d %b %H:%M")]),
            hovertemplate=("%{x|%d %b %H:%M}: %{y:.1f} kg/THM (%{customdata[0]:.0f}-%{customdata[1]:.0f})"
                           "<br>measured %{customdata[2]:.1f}; from the %{customdata[3]} row<extra>" + STATE_LABELS[state] + "</extra>")))
    _style(fig, height=380)
    return fig


def trust_summary(recent: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Error by condition for one horizon, against carrying the current rate."""

    rows = recent[(recent["horizon"] == horizon) & recent["shown"]].dropna(subset=["actual", "prediction"])
    out = []
    for state in [*STATE_COLORS, "all"]:
        part = rows if state == "all" else rows[rows["state"] == state]
        if part.empty:
            continue
        err = (part["prediction"] - part["actual"]).abs()
        naive = (part["anchor"] - part["actual"]).abs()
        inside = part["actual"].between(part["lower"], part["upper"])
        out.append({
            "Condition": "All shown" if state == "all" else STATE_LABELS[state], "Hours": len(part),
            "Forecast error (kg/THM)": f"{err.mean():.1f}",
            "Holding current rate (kg/THM)": f"{naive.mean():.1f}",
            "Inside range": f"{inside.mean():.0%}",
        })
    return pd.DataFrame(out)


# ---- retrain -----------------------------------------------------------------


def report_tables(report: dict[str, Any]) -> None:
    """Per-horizon errors, conditions, coefficients and the acceptance checks."""

    per = pd.DataFrame(report.get("per_horizon", []))
    if not per.empty:
        per = per.assign(
            Horizon=[f"+{h} h" for h in per["horizon"]],
            Hours=per["n"],
            **{"Forecast error": per["mae"].map("{:.1f}".format),
               "Holding current rate": per["persistence_mae"].map("{:.1f}".format),
               "Skill": per["skill_vs_persistence"].map(lambda v: f"{v:.0%}" if v is not None else "n/a"),
               "Inside range": per["range_coverage"].map(lambda v: f"{v:.0%}" if v is not None else "n/a")})
        st.dataframe(per[["Horizon", "Hours", "Forecast error", "Holding current rate", "Skill", "Inside range"]],
                     hide_index=True, width="stretch")
    checks = report.get("checks", {})
    if checks:
        st.markdown(" ".join(
            f":{'green' if ok else 'red'}-badge[:material/{'check' if ok else 'close'}: {name}]"
            for name, ok in checks.items()))
    beta = pd.DataFrame(report.get("beta", []))
    if not beta.empty:
        st.caption("Coke response per kg/THM change (PCI, nut coke, expected slag), by horizon: " + "; ".join(
            f"+{r.horizon} h {r.pci:+.2f} / {r.nut:+.2f} / {r.slag:+.3f}" for r in beta.itertuples()))
    st.caption(
        f"Trained on {report.get('train_hours', '?')} hours from {_when(report.get('train_start'))} to "
        f"{_when(report.get('trained_until'))}; scored on the next seven days, which it never saw "
        f"({_when(report.get('holdout', [None])[0])} to {_when(report.get('data_end'))})."
    )


def render_model_report(*, storage_dir: Path, base_bundle: Path) -> None:
    """Accuracy report for the active five-hour charged-coke model."""

    import json

    active = deployments.active_bundle(storage_dir, base_bundle)
    info = deployments.active_info(storage_dir)
    accepted = (
        f"; accepted by {info.get('accepted_by')} on {_when(info.get('accepted_at'))}"
        if info
        else "; shipped with the app"
    )
    st.caption(f"Active version: {active.name}{accepted}.")
    report_path = active / "report.json"
    if not report_path.is_file():
        st.warning("The active charged-coke model has no saved accuracy report.")
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.warning(f"The charged-coke accuracy report could not be read: {exc}")
        return
    report_tables(report)


def _holdout_recent(folder: Path) -> pd.DataFrame:
    from utils.bmo.charged_coke.contract import horizon_window

    hold = pd.read_csv(folder / "holdout.csv.gz", parse_dates=["time"]).set_index("time")
    frames = []
    for h in range(1, 6):
        if f"prediction_h{h}" not in hold:
            continue
        frames.append(pd.DataFrame({
            "issue_row": hold.index, "horizon": h, "at": [horizon_window(t, h)["coke_end"] for t in hold.index],
            "prediction": hold[f"prediction_h{h}"].to_numpy(), "lower": hold[f"lower_h{h}"].to_numpy(),
            "upper": hold[f"upper_h{h}"].to_numpy(), "actual": hold[f"actual_h{h}"].to_numpy(),
            "anchor": hold["charge4"].to_numpy(), "state": hold["state"].to_numpy(),
        }))
    out = pd.concat(frames, ignore_index=True)
    out["shown"] = out["state"].isin(STATE_COLORS)
    return out


def render_retrain(*, storage_dir: Path, dataset_path: Path, base_bundle: Path, data_end: pd.Timestamp | None,
                   can_manage: bool, user: str, on_change) -> None:
    """Active model, background retrain, candidate review, accept / roll back."""

    info = deployments.active_info(storage_dir)
    active = deployments.active_bundle(storage_dir, base_bundle)
    st.markdown(f"**Active model:** `{active.name}`" + (
        f" · accepted by {info.get('accepted_by')} on {_when(info.get('accepted_at'))}" if info else " · shipped with the app"))
    report_path = active / "report.json"
    if report_path.is_file():
        import json

        with st.expander("Active model's holdout report", expanded=False):
            report_tables(json.loads(report_path.read_text(encoding="utf-8")))

    status = deployments.retrain_status(storage_dir)
    running = status.get("stage") == "running"
    cols = st.columns([1, 2])
    if cols[0].button("Retrain on latest data", disabled=running or not can_manage or data_end is None,
                      help="Fits on everything up to seven days ago and scores the last seven days, about ten "
                           "minutes in the background. Nothing changes until the result is accepted.",
                      key="bmo_charged_retrain"):
        deployments.start_retrain(storage_dir=storage_dir, dataset_path=dataset_path, base_bundle=base_bundle,
                                  end=data_end, requested_by=user)
        st.rerun()
    if not can_manage:
        cols[1].caption("Retraining and accepting models needs a supervisor or admin login.")
    if running:
        cols[1].info(f"Retraining ({status.get('message')}), started {status.get('started_at', '')[11:16]} by "
                     f"{status.get('requested_by', '')}.")
        return
    if status.get("stage") == "failed":
        cols[1].error(f"Last retrain failed: {status.get('message')}")

    candidates = [v for v in deployments.versions(storage_dir) if v["version"] != active.name]
    newest = candidates[0] if candidates else None
    if newest and status.get("version") == newest["version"]:
        st.markdown(f"##### Candidate `{newest['version']}`, ready for review")
        report_tables(newest["report"])
        recent = _holdout_recent(newest["path"])
        horizon = st.segmented_control("Horizon", [1, 2, 3, 4, 5], default=5, format_func=lambda h: f"+{h} h",
                                       key="bmo_charged_candidate_horizon") or 5
        st.plotly_chart(trust_figure(recent, int(horizon)), width="stretch")
        st.dataframe(trust_summary(recent, int(horizon)), hide_index=True, width="stretch")
        if st.button(f"Accept and deploy {newest['version']}", type="primary", disabled=not can_manage,
                     key="bmo_charged_accept"):
            deployments.activate(storage_dir, newest["version"], accepted_by=user, now=datetime.now(),
                                 note="accepted after review")
            on_change()
            st.rerun()
    others = deployments.versions(storage_dir)
    if others:
        with st.expander("Roll back to another version", expanded=False):
            choice = st.selectbox("Version", [v["version"] for v in others], key="bmo_charged_rollback_version")
            if st.button("Make this version active", disabled=not can_manage, key="bmo_charged_rollback"):
                deployments.activate(storage_dir, choice, accepted_by=user, now=datetime.now(), note="roll back")
                on_change()
                st.rerun()


# ---- panel -------------------------------------------------------------------


def render_status(prediction: Any, detail: dict[str, Any]) -> None:
    """Plain operator summary of the +5 h value used by the optimiser."""

    record = detail.get("record") or {}
    state = str(record.get("state", ""))
    reasons = [reason for reason in (record.get("reasons") or []) if reason]
    forced = bool(detail.get("forced") or record.get("forced"))
    st.markdown("**Charged coke forecast**")
    st.caption(
        "Charged coke is the coke physically charged into the furnace per tonne "
        "of hot metal, measured over four hours. The four-hour rate removes the "
        "normal swings caused by whole charges."
    )
    if prediction.usable:
        if forced:
            warning = (
                "**Force predict is ON.** This value is outside the model's "
                "normal display rules and has no validated error range."
            )
            if reasons:
                warning += "\n\n" + "\n".join(f"- {reason}" for reason in reasons)
            st.warning(warning)
        value_col, range_col, state_col = st.columns(3)
        value_col.caption("Expected in 5 hours")
        value_col.markdown(f"**{prediction.value_kg_per_thm:,.1f} kg/THM**")
        range_col.caption("Expected range")
        if forced:
            range_col.markdown("**Not validated**")
        else:
            lower = float(record.get("lower_kg_thm", prediction.value_kg_per_thm))
            upper = float(record.get("upper_kg_thm", prediction.value_kg_per_thm))
            range_col.markdown(f"**{lower:,.0f}-{upper:,.0f} kg/THM**")
        state_col.caption("Furnace condition")
        state_col.markdown(
            f"**{'Forced' if forced else STATE_LABELS.get(state, 'Check')}**"
        )
        if forced:
            st.caption(
                "The Data-Driven optimiser uses this forced 5-hour value. Treat "
                "it as an extrapolation, not a validated operating forecast."
            )
        else:
            st.caption(
                "The Data-Driven optimiser uses the 5-hour value. The shaded range "
                "on the trend shows normal forecast uncertainty."
            )
        if detail.get("override"):
            st.caption(
                "This value includes the operator's PCI or nut-coke override. "
                "Its displayed range comes from the live forecast."
            )
    else:
        why = PAUSE_WORDS.get(state, state.replace("_", " ")) or "not available"
        st.warning(
            f"No charged-coke forecast is available now ({why}). "
            "Use Physics-Driven or wait for the next valid forecast."
        )

    with st.popover("Forecast details"):
        st.markdown(
            "The forecast is paused when data are incomplete or the furnace is "
            "outside conditions the model has learned. The expected range covers "
            "90% of comparable past forecast errors."
        )
        if prediction.usable and record.get("data_row"):
            window = ForecastWindow.for_row(
                pd.Timestamp(record["data_row"]), pd.Timestamp(record["issued_at"])
            )
            st.caption(window.describe())
        last = record.get("last_valid")
        if not prediction.usable and last:
            st.caption(
                f"Last valid forecast: {last.get('prediction_kg_thm', 0):,.1f} "
                f"kg/THM from the {_when(last['data_row'])} row."
            )
        for reason in reasons[:4]:
            st.caption(f"- {reason}")
        st.caption(
            f"Data row {_when(record.get('data_row'))}; issued "
            f"{_when(record.get('issued_at'))} IST; model {prediction.deployment_id}."
        )


def render_panel(
    prediction: Any,
    detail: dict[str, Any],
    *,
    recent: pd.DataFrame | None,
    retrain: dict[str, Any],
    show_status: bool = True,
) -> None:
    """Charged-coke trend, recent check and manager-only model controls."""

    st.markdown("### Charged coke trend")
    if show_status:
        render_status(prediction, detail)
    record = detail.get("record") or {}
    tab_names = ["Next 5 hours", "Last 7 days"]
    if retrain.get("can_manage"):
        tab_names.append("Model")
    tabs = st.tabs(tab_names)
    next_tab, week_tab = tabs[:2]
    with next_tab:
        if record.get("path") and prediction.usable:
            st.plotly_chart(path_figure(record, recent), width="stretch")
            st.caption(
                "Each point is the expected four-hour charged coke rate at that "
                "time. The shaded area is the expected range."
            )
        else:
            st.caption("The trend will return when a valid forecast is available.")
    with week_tab:
        if recent is None or recent.empty:
            st.caption("No completed forecast checks are available yet.")
        else:
            horizon = st.segmented_control("Horizon", [1, 2, 3, 4, 5], default=5, format_func=lambda h: f"+{h} h",
                                           key="bmo_charged_trust_horizon") or 5
            st.plotly_chart(trust_figure(recent, int(horizon)), width="stretch")
            st.dataframe(trust_summary(recent, int(horizon)), hide_index=True, width="stretch")
            hidden = recent[(recent["horizon"] == int(horizon)) & ~recent["shown"]]
            st.caption(
                f"Dots are the forecasts and the line is the measured charged "
                f"coke rate. Green is steady, amber is variable and red is outside "
                f"normal. The model paused for {len(hidden)} hours, so no dot is "
                "shown for those hours."
            )
    if len(tabs) == 3:
        with tabs[2]:
            render_retrain(**retrain)


def _style(fig: go.Figure, *, height: int) -> None:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Charged coke (kg/THM)", gridcolor="rgba(120,130,145,0.18)"),
        xaxis=dict(showgrid=False, tickformat="%d %b<br>%H:%M"),
    )


__all__ = [
    "path_figure",
    "render_model_report",
    "render_panel",
    "render_status",
    "trust_figure",
    "trust_summary",
]
