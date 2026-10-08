"""Live HM Si path, seven-day accuracy and governed model deployment."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.bmo.si_forecast import deployments
from utils.bmo.si_forecast.ledger import SiliconForecastLedger
from utils.bmo.si_forecast.service import forecast_horizons

PLANT_TZ = "Asia/Kolkata"
TEAL = "#07827f"
NAVY = "#25344b"
AMBER = "#f08c00"
REPLAY_BLUE = "#4c91c7"


def _ist(value: Any, format_string: str = "%d %b %Y, %H:%M IST") -> str:
    if value in (None, ""):
        return "not available"
    try:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return stamp.tz_convert(PLANT_TZ).strftime(format_string)
    except (TypeError, ValueError):
        return str(value)


def _label(minutes: int) -> str:
    return "Now" if int(minutes) == 0 else f"+{int(minutes) // 60} h"


def _ist_axis(values: Any) -> pd.DatetimeIndex:
    """Plotly gets a timezone-free IST clock so browsers cannot redraw UTC."""

    parsed = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="coerce"))
    return parsed.tz_convert(PLANT_TZ).tz_localize(None)


def _ist_axis_one(value: Any) -> pd.Timestamp:
    return _ist_axis([value])[0]


def _style(fig: go.Figure, *, height: int = 350) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="HM Si (%)", gridcolor="rgba(120,130,145,0.18)"),
        xaxis=dict(showgrid=False, tickformat="%d %b<br>%H:%M"),
    )


def path_figure(
    record: Mapping[str, Any],
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    replay_forecasts: pd.DataFrame | None = None,
    replay_horizon_minutes: int | None = None,
) -> go.Figure:
    """Last 24 h of irregular samples/nowcasts plus the latest future path."""

    fig = go.Figure()
    issued = pd.Timestamp(record.get("issued_at") or record["origin_at"])
    cutoff = issued - pd.Timedelta(hours=24)
    if actuals is not None and not actuals.empty:
        measured = actuals[actuals["sample_at"].ge(cutoff)].sort_values("sample_at")
        fig.add_trace(
            go.Scatter(
                x=_ist_axis(measured["sample_at"]),
                y=measured["actual"],
                mode="lines+markers",
                name="Measured Si",
                line=dict(color=NAVY, width=1.4),
                marker=dict(size=7, color=NAVY),
                hovertemplate=(
                    "%{x|%d %b %H:%M} IST: %{y:.3f}%<extra>measured</extra>"
                ),
            )
        )
    selected_horizon = replay_horizon_minutes
    if replay_forecasts is not None and not replay_forecasts.empty:
        available = sorted(
            int(value) for value in replay_forecasts["horizon_minutes"].unique()
        )
        if selected_horizon not in available:
            selected_horizon = 0 if 0 in available else available[0]
        replay = replay_forecasts[
            replay_forecasts["horizon_minutes"].eq(int(selected_horizon))
            & replay_forecasts["target_start"].ge(cutoff)
            & replay_forecasts["target_start"].le(issued)
            & replay_forecasts["prediction"].notna()
        ].sort_values("target_start")
        if not replay.empty:
            fig.add_trace(
                go.Scatter(
                    x=_ist_axis(replay["target_start"]),
                    y=replay["prediction"],
                    mode="lines",
                    name=f"Model replay ({_label(int(selected_horizon))})",
                    line=dict(color=REPLAY_BLUE, width=1.6),
                    hovertemplate=(
                        "%{x|%d %b %H:%M} IST: %{y:.3f}%"
                        "<extra>model replay</extra>"
                    ),
                )
            )
    if forecasts is not None and not forecasts.empty:
        issued_horizons = sorted(
            int(value) for value in forecasts["horizon_minutes"].unique()
        )
        issued_horizon = 0 if 0 in issued_horizons else (
            int(selected_horizon)
            if selected_horizon in issued_horizons
            else issued_horizons[0]
        )
        issued_rows = forecasts[
            forecasts["horizon_minutes"].eq(issued_horizon)
            & forecasts["origin_at"].ge(cutoff)
            & forecasts["prediction"].notna()
        ].sort_values("target_start")
        if not issued_rows.empty:
            fig.add_trace(
                go.Scatter(
                    x=_ist_axis(issued_rows["target_start"]),
                    y=issued_rows["prediction"],
                    mode="lines+markers",
                    name=f"Live forecast ({_label(issued_horizon)})",
                    line=dict(color=TEAL, width=1.3),
                    marker=dict(color=TEAL, size=5),
                    hovertemplate=(
                        "%{x|%d %b %H:%M} IST: %{y:.3f}%"
                        "<extra>live forecast</extra>"
                    ),
                )
            )

    path = [row for row in forecast_horizons(record) if row.get("si_pct") is not None]
    if path:
        at = list(_ist_axis([row["target_start"] for row in path]))
        values = [float(row["si_pct"]) for row in path]
        low = [row.get("lower_pct") for row in path]
        high = [row.get("upper_pct") for row in path]
        has_range = all(
            value is not None and np.isfinite(float(value))
            for value in [*low, *high]
        )
        if has_range:
            fig.add_trace(
                go.Scatter(
                    x=at + at[::-1],
                    y=[float(value) for value in high]
                    + [float(value) for value in low[::-1]],
                    fill="toself",
                    fillcolor="rgba(7,130,127,0.14)",
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    name="Expected range (90%)",
                )
            )
        trace: dict[str, Any] = {
            "x": at,
            "y": values,
            "mode": "lines+markers+text",
            "name": "Latest forecast",
            "line": dict(color=TEAL, dash="dash", width=2),
            "marker": dict(size=9, color=TEAL),
            "text": [
                f"{_label(int(row['horizon_minutes']))}<br>{value:.2f}"
                for row, value in zip(path, values)
            ],
            "textposition": "top center",
        }
        if has_range:
            trace["customdata"] = np.column_stack([low, high])
            trace["hovertemplate"] = (
                "%{x|%d %b %H:%M} IST: %{y:.3f}% "
                "(%{customdata[0]:.3f}-%{customdata[1]:.3f})"
                "<extra>forecast</extra>"
            )
        fig.add_trace(go.Scatter(**trace))
    fig.add_vline(
        x=_ist_axis_one(issued), line=dict(color=NAVY, width=1, dash="dot")
    )
    _style(fig)
    return fig


def trust_figure(
    scored: pd.DataFrame,
    horizon_minutes: int,
    *,
    forecast_name: str = "Model",
    forecasts: pd.DataFrame | None = None,
    actuals: pd.DataFrame | None = None,
) -> go.Figure:
    rows = (
        scored[scored["horizon_minutes"].eq(int(horizon_minutes))].sort_values(
            "sample_at"
        )
        if {"horizon_minutes", "sample_at"}.issubset(scored.columns)
        else pd.DataFrame()
    )
    measured = (
        actuals.sort_values("sample_at")
        if actuals is not None and not actuals.empty
        else rows.rename(columns={"actual": "actual"})
    )
    inferred = (
        forecasts[
            forecasts["horizon_minutes"].eq(int(horizon_minutes))
            & forecasts["prediction"].notna()
        ].sort_values("target_start")
        if forecasts is not None and not forecasts.empty
        else rows
    )
    fig = go.Figure()
    if not measured.empty:
        fig.add_trace(
            go.Scatter(
                x=_ist_axis(measured["sample_at"]),
                y=measured["actual"],
                mode="markers",
                name="Measured Si",
                marker=dict(size=6, color=NAVY),
                hovertemplate=(
                    "%{x|%d %b %H:%M} IST: %{y:.3f}%<extra>measured</extra>"
                ),
            )
        )
    if not inferred.empty:
        statuses = inferred.get("status", pd.Series("ok", index=inferred.index))
        colours = [AMBER if status != "ok" else TEAL for status in statuses]
        target_column = "target_start" if "target_start" in inferred else "sample_at"
        fig.add_trace(
            go.Scatter(
                x=_ist_axis(inferred[target_column]),
                y=inferred["prediction"],
                mode="lines+markers",
                name=forecast_name,
                line=dict(color=TEAL, width=1.5),
                marker=dict(size=4, color=colours),
                hovertemplate=(
                    "%{x|%d %b %H:%M} IST: %{y:.3f}%<extra>model</extra>"
                ),
            )
        )
    _style(fig, height=370)
    return fig


def trust_summary(scored: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    rows = scored[scored["horizon_minutes"].eq(int(horizon_minutes))].dropna(
        subset=["actual", "prediction"]
    )
    if rows.empty:
        return pd.DataFrame()
    error = rows["prediction"] - rows["actual"]
    persistence = (rows["persistence"] - rows["actual"]).abs().dropna()
    inside = rows["inside_range"].dropna()
    return pd.DataFrame(
        [
            {
                "Matched samples": len(rows),
                "Forecast MAE": f"{error.abs().mean():.3f}",
                "Holding latest Si": (
                    f"{persistence.mean():.3f}" if len(persistence) else "n/a"
                ),
                "Bias": f"{error.mean():+.3f}",
                "Inside range": f"{inside.mean():.0%}" if len(inside) else "n/a",
            }
        ]
    )


def _report_table(report: Mapping[str, Any]) -> pd.DataFrame:
    per_horizon = report.get("per_horizon") or []
    if per_horizon:
        rows = []
        for horizon in per_horizon:
            model = horizon.get("extra_trees") or {}
            persistence = horizon.get("latest_si") or {}
            rows.append(
                {
                    "Horizon": horizon.get("label") or _label(horizon["horizon_minutes"]),
                    "Raw samples": model.get("n"),
                    "MAE": model.get("mae"),
                    "Hold latest Si": persistence.get("mae"),
                    "Bias": model.get("bias"),
                    "Inside 90% range": horizon.get("range_coverage"),
                }
            )
        return pd.DataFrame(rows)
    labels = {
        "extra_trees": "ExtraTrees",
        "latest_si": "Hold latest measured Si",
        "mean_last_3_si": "Mean of last 3 Si samples",
    }
    return pd.DataFrame(
        [
            {
                "Method": labels.get(name, str(name)),
                "Raw samples": metrics.get("n"),
                "MAE": metrics.get("mae"),
                "Bias": metrics.get("bias"),
                "Inside +/-0.10": metrics.get("within_010"),
            }
            for name, metrics in (report.get("metrics") or {}).items()
        ]
    )


def render_report(report: Mapping[str, Any]) -> None:
    table = _report_table(report)
    if not table.empty:
        st.dataframe(table, hide_index=True, width="stretch")
    holdout = report.get("holdout") or [None, None]
    st.caption(
        f"Trained through {_ist(report.get('trained_until'))}; scored on later raw "
        f"samples from {_ist(holdout[0], '%d %b')} to {_ist(holdout[-1], '%d %b')} "
        "that were not used for fitting. Raw Si targets are never interpolated."
    )
    checks = list(report.get("checks") or [])
    failed = [row for row in checks if not row.get("passed")]
    if checks:
        st.markdown(
            f":{'green' if not failed else 'red'}-badge["
            f"{len(checks) - len(failed)}/{len(checks)} deployment checks passed]"
        )
    for check in failed:
        st.caption(
            "FAIL - "
            + str(check.get("name", "check")).replace("_", " ").title()
            + ": "
            + str(check.get("detail", ""))
        )


def _live_metrics(storage_dir: Path) -> pd.DataFrame:
    frame = SiliconForecastLedger(storage_dir).scored_frame()
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for horizon, part in frame.groupby("horizon_minutes"):
        error = part["prediction"] - part["actual"]
        persistence = (part["persistence"] - part["actual"]).abs().dropna()
        inside = part["inside_range"].dropna()
        rows.append(
            {
                "Horizon": _label(int(horizon)),
                "Samples": len(part),
                "MAE": error.abs().mean(),
                "Hold latest Si": persistence.mean() if len(persistence) else None,
                "Bias": error.mean(),
                "Inside range": inside.mean() if len(inside) else None,
            }
        )
    return pd.DataFrame(rows)


def render_accuracy(*, storage_dir: Path, default_bundle: Path) -> None:
    active = deployments.active_bundle(storage_dir, default_bundle)
    info = deployments.active_info(storage_dir)
    accepted = (
        f"; accepted by {info.get('accepted_by')} on {_ist(info.get('accepted_at'))}"
        if info
        else "; initial production version shipped with the app"
    )
    st.caption(f"Active production version: `{active.name}`{accepted}.")
    report_path = active / "report.json"
    if report_path.is_file():
        render_report(json.loads(report_path.read_text(encoding="utf-8")))
    else:
        st.warning("The active silicon model has no saved review report.")
    live = _live_metrics(storage_dir)
    if not live.empty:
        st.markdown("**Completed live forecast checks**")
        st.dataframe(live, hide_index=True, width="stretch")
    else:
        st.caption("No live forecasts have matured into matched lab samples yet.")


def render_model_management(
    *,
    storage_dir: Path,
    default_bundle: Path,
    static_dataset_path: Path,
    can_manage: bool,
    user: str,
    on_change: Callable[[], None] | None = None,
) -> None:
    active = deployments.active_bundle(storage_dir, default_bundle)
    info = deployments.active_info(storage_dir)
    st.markdown(
        f"**Active model:** `{active.name}`"
        + (
            f" - accepted by {info.get('accepted_by')} on {_ist(info.get('accepted_at'))}"
            if info
            else " - initial production version"
        )
    )
    status = deployments.retrain_status(storage_dir)
    running = status.get("stage") == "running"
    data_end = pd.Timestamp.now(tz=PLANT_TZ).floor("5min") - pd.Timedelta(
        hours=3, minutes=5
    )
    controls = st.columns([1, 2])
    if controls[0].button(
        "Train new candidate",
        disabled=running or not can_manage,
        help=(
            "Builds a new Now, +1 h, +2 h and +3 h candidate from the latest "
            "60 days, then tests it on a locked seven-day holdout. It does not "
            "replace the active model until you approve it."
        ),
        key="bmo_si_retrain",
    ):
        deployments.start_retrain(
            storage_dir=storage_dir,
            base_bundle=active,
            static_dataset_path=static_dataset_path,
            data_end_origin=data_end,
            requested_by=user,
        )
        st.rerun()
    if not can_manage:
        controls[1].caption("Retraining and deployment need a supervisor or admin login.")
    elif running:
        controls[1].info(
            f"Retraining: {status.get('message', 'running')} - requested by "
            f"{status.get('requested_by', '')}."
        )
    if status.get("stage") == "failed":
        controls[1].error(f"Last retrain failed: {status.get('message')}")

    candidates = [
        item for item in deployments.versions(storage_dir) if item["version"] != active.name
    ]
    newest = candidates[0] if candidates else None
    if newest and status.get("version") == newest["version"]:
        st.markdown(f"##### Candidate `{newest['version']}` - ready for review")
        report = newest["report"]
        render_report(report)
        holdout_path = newest["path"] / "holdout.csv.gz"
        if holdout_path.is_file():
            holdout = pd.read_csv(holdout_path, parse_dates=["sample_time"])
            choices = sorted(holdout["horizon_minutes"].unique())
            choice = st.segmented_control(
                "Candidate horizon",
                choices,
                default=choices[0],
                format_func=_label,
                key="bmo_si_candidate_horizon",
            )
            choice = choices[0] if choice is None else choice
            selected = holdout[holdout["horizon_minutes"].eq(int(choice))]
            selected = selected.copy()
            selected["sample_time"] = _ist_axis(selected["sample_time"])
            st.line_chart(
                selected.set_index("sample_time")[["actual", "prediction", "persistence"]],
                height=250,
            )
        if st.button(
            f"Approve and deploy {newest['version']}",
            type="primary",
            disabled=not can_manage or not bool(report.get("all_checks_passed")),
            help="Every causal, sample-count, bias and persistence check must pass.",
            key="bmo_si_accept",
        ):
            deployments.activate(
                storage_dir,
                newest["version"],
                default_bundle=default_bundle,
                accepted_by=user,
                now=datetime.now(),
                note="accepted after multi-horizon holdout review",
            )
            if on_change:
                on_change()
            st.rerun()

    all_versions = deployments.versions(storage_dir, default_bundle)
    if all_versions:
        with st.expander("Roll back to another Si model version", expanded=False):
            choices = [item["version"] for item in all_versions]
            choice = st.selectbox("Version", choices, key="bmo_si_rollback_version")
            if st.button(
                "Make this version active",
                disabled=not can_manage or choice == active.name,
                key="bmo_si_rollback",
            ):
                deployments.activate(
                    storage_dir,
                    choice,
                    default_bundle=default_bundle,
                    accepted_by=user,
                    now=datetime.now(),
                    note="roll back",
                )
                if on_change:
                    on_change()
                st.rerun()


def render_forecast(
    record: Mapping[str, Any] | None,
    *,
    storage_dir: Path | None = None,
    default_bundle: Path | None = None,
    static_dataset_path: Path | None = None,
    can_manage: bool = False,
    user: str = "unknown",
    on_change: Callable[[], None] | None = None,
    replay_forecasts: pd.DataFrame | None = None,
    raw_actuals: pd.DataFrame | None = None,
    replay_scored: pd.DataFrame | None = None,
    history_error: str | None = None,
) -> None:
    """Render the charged-coke-style operator trend and governed model tabs."""

    st.markdown("### HM Si trend")
    if not record:
        st.warning("HM Si forecast is not available.")
        return
    status = str(record.get("status") or "insufficient_data")
    usable = [
        row for row in forecast_horizons(record) if row.get("si_pct") is not None
    ]
    reasons = [str(reason) for reason in record.get("reasons") or []]
    if not usable:
        st.warning(
            "HM Si forecast is not available"
            + (f": {reasons[0]}" if reasons else ".")
        )
    elif status != "ok" and reasons:
        st.warning("HM Si forecast shown with warning: " + reasons[0])
    forecasts = pd.DataFrame()
    actuals = pd.DataFrame()
    scored = pd.DataFrame()
    if storage_dir is not None:
        ledger = SiliconForecastLedger(storage_dir)
        forecasts = ledger.forecast_frame()
        actuals = ledger.actual_frame()
        scored = ledger.scored_frame()
    if raw_actuals is not None and not raw_actuals.empty:
        actuals = pd.concat([actuals, raw_actuals], ignore_index=True)
        actuals = actuals.drop_duplicates(["sample_id", "sample_at"], keep="last")
    replay_forecasts = (
        replay_forecasts if replay_forecasts is not None else pd.DataFrame()
    )
    replay_scored = replay_scored if replay_scored is not None else pd.DataFrame()
    tab_names = ["Next 3 hours", "Last 7 days"]
    if default_bundle is not None and static_dataset_path is not None:
        tab_names.append("Model")
    tabs = st.tabs(tab_names)
    with tabs[0]:
        replay_horizon: int | None = None
        if not replay_forecasts.empty:
            choices = sorted(
                int(value) for value in replay_forecasts["horizon_minutes"].unique()
            )
            if len(choices) > 1:
                replay_horizon = st.segmented_control(
                    "Past model line",
                    choices,
                    default=0 if 0 in choices else choices[0],
                    format_func=_label,
                    key="bmo_si_replay_horizon",
                )
            replay_horizon = (
                (0 if 0 in choices else choices[0])
                if replay_horizon is None
                else int(replay_horizon)
            )
        st.plotly_chart(
            path_figure(
                record,
                forecasts,
                actuals,
                replay_forecasts=replay_forecasts,
                replay_horizon_minutes=replay_horizon,
            ),
            width="stretch",
        )
        if history_error:
            st.warning("Recent model replay could not be refreshed.")
    with tabs[1]:
        display_forecasts = (
            replay_forecasts if not replay_forecasts.empty else forecasts
        )
        display_scores = replay_scored if not replay_scored.empty else scored
        display_actuals = actuals
        if display_forecasts.empty and display_actuals.empty:
            st.caption("No HM Si history is available yet.")
        else:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
            recent_forecasts = display_forecasts[
                display_forecasts["target_start"].ge(cutoff)
            ]
            recent_actuals = (
                display_actuals[display_actuals["sample_at"].ge(cutoff)]
                if "sample_at" in display_actuals
                else pd.DataFrame(columns=["sample_at", "actual"])
            )
            recent_scores = (
                display_scores[display_scores["sample_at"].ge(cutoff)]
                if "sample_at" in display_scores
                else pd.DataFrame(
                    columns=[
                        "horizon_minutes",
                        "sample_at",
                        "actual",
                        "prediction",
                        "persistence",
                        "inside_range",
                    ]
                )
            )
            choices = sorted(recent_forecasts["horizon_minutes"].unique())
            if not choices:
                st.caption("No model history is available in the last seven days.")
            else:
                horizon = st.segmented_control(
                    "Horizon",
                    choices,
                    default=choices[0],
                    format_func=_label,
                    key="bmo_si_trust_horizon",
                )
                horizon = choices[0] if horizon is None else horizon
                st.plotly_chart(
                    trust_figure(
                        recent_scores,
                        int(horizon),
                        forecast_name="Model",
                        forecasts=recent_forecasts,
                        actuals=recent_actuals,
                    ),
                    width="stretch",
                )
                summary = trust_summary(recent_scores, int(horizon))
                if not summary.empty:
                    st.dataframe(summary, hide_index=True, width="stretch")
    if len(tabs) == 3:
        with tabs[2]:
            render_model_management(
                storage_dir=Path(storage_dir or default_bundle.parent),
                default_bundle=default_bundle,
                static_dataset_path=static_dataset_path,
                can_manage=can_manage,
                user=user,
                on_change=on_change,
            )


__all__ = [
    "path_figure",
    "render_accuracy",
    "render_forecast",
    "render_model_management",
    "render_report",
    "trust_figure",
    "trust_summary",
]
