"""Does the coke-rate model actually work? Shown, not asserted.

Two predictions on this page cannot be checked by eye — the coke rate the fuel
cost is built on, and the silicon that drives the correction's thermal term. An
operator has no way to tell a good one from a bad one at the moment it is shown.

So this panel puts both against what the plant actually measured, day by day,
over the recent record. Not a summary statistic: the whole series, so a run of
days where the model drifted is visible as a run rather than averaged away.

The retrain control lives here for the same reason. Refitting the offset is
only sensible when you can see what it is being fitted to, and the effect of a
refit shows up immediately in the chart beneath it.

The measured target is built from paired hourly COKE_CALC_MT and hot-metal
production. Only complete days enter the offset; the current partial day is
still shown by the live BMO rate but cannot distort calibration.
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

log = logging.getLogger(__name__)

HISTORY_DAYS = 120
CALIBRATION_WINDOW_DAYS = 90
TRACKING_DISPLAY_DAYS = 3
# One palette for every accuracy chart on this tab: model in teal, plant
# measurement in navy, the model's typical error as a pale teal band.
_TRACKING_TEAL = "#07827f"
_TRACKING_NAVY = "#25344b"
_BAND_FILL = "rgba(7,130,127,0.14)"
_CONTEXT_AMBER = "#c77d2a"
# Plant users read time in IST; the model and dataset work in UTC.
_PLANT_TZ = "Asia/Kolkata"


@st.cache_data(ttl=1800, show_spinner=False)
def _coke_history(
    days: int, cache_bust: int
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Daily predicted-vs-realised coke and the days that could not be scored.

    ``cache_bust`` forces a refetch.
    """

    from utils.bmo.coke_history import build_daily_history

    result = build_daily_history(days)
    return result.frame, list(result.warnings), result.excluded


@st.cache_data(ttl=1800, show_spinner=False)
def _si_history(days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Daily predicted-vs-realised silicon, with the rebuild's own report."""

    from utils.bmo.si_history import build_si_history

    result = build_si_history(days)
    return result.frame, {
        "derived": result.derived,
        "filled": result.filled,
        "filled_names": list(result.filled_names),
        "trustworthy": result.is_trustworthy,
        "notes": list(result.notes),
    }


def _readable_failure(exc: Exception) -> str:
    """A message safe and useful to put on screen.

    Database errors arrive carrying the DSN — host, user, database name — and
    a SQLAlchemy traceback. None of that belongs in front of an operator, and
    the host and username in particular should not be on a shared screen. The
    full exception goes to the log; this returns what to display.
    """

    text = str(exc)
    if (
        "pg_hba.conf" in text
        or "could not connect" in text
        or ("connection to server" in text)
    ):
        return (
            "The offline database is not reachable from this machine. If you are "
            "off the plant network, or this host's address has not been "
            "whitelisted, that is the usual cause. Nothing else on the page is "
            "affected."
        )
    if "timeout" in text.lower():
        return "The offline database did not respond in time. Try again shortly."
    # Anything unrecognised: first line only, no traceback, length-capped.
    return text.splitlines()[0][:200] if text.strip() else exc.__class__.__name__


def _scores(predicted: pd.Series, actual: pd.Series) -> dict[str, float]:
    """Bias, MAE, MAPE and R2 for one aligned pair."""

    both = (
        pd.concat([predicted, actual], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if both.empty:
        return {}
    err = both.iloc[:, 0] - both.iloc[:, 1]
    truth = both.iloc[:, 1]
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return {
        "n": float(len(both)),
        "bias": float(err.mean()),
        "MAE": float(err.abs().mean()),
        "MAPE": float((err.abs() / truth.abs().replace(0, np.nan)).mean() * 100.0),
        "R2": float(1.0 - float((err**2).sum()) / ss_tot) if ss_tot else float("nan"),
    }


def _paired_chart(
    frame: pd.DataFrame,
    *,
    predicted_col: str,
    actual_col: str,
    unit: str,
    predicted_name: str = "Predicted",
    actual_name: str = "Measured",
    band: float | None = None,
    decimals: int = 1,
) -> go.Figure:
    """Daily predicted over measured, with the band the model is good to."""

    return _prediction_figure(
        frame,
        predicted_col=predicted_col,
        actual_col=actual_col,
        unit=unit,
        predicted_name=predicted_name,
        actual_name=actual_name,
        band=band,
        band_label=(
            f"±{band:,.{decimals}f} {unit} typical error" if band else None
        ),
        decimals=decimals,
    )


def _score_row(scores: dict[str, float], unit: str, decimals: int = 1) -> None:
    if not scores:
        st.caption("Not enough paired days to score.")
        return
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric(
            "Days scored",
            f"{scores['n']:,.0f}",
            help="Days with both a prediction and a plant measurement in this window.",
        )
        cols[1].metric(
            "Average offset",
            f"{scores['bias']:+,.{decimals}f} {unit}",
            help="Predicted minus measured, averaged. Near zero means no "
            "systematic over- or under-prediction.",
        )
        cols[2].metric(
            "Typical error",
            f"{scores['MAE']:,.{decimals}f} {unit}",
            help="Mean absolute error: what to expect on any one day.",
        )
        cols[3].metric(
            "R²",
            f"{scores['R2']:+.2f}",
            help="Share of the day-to-day movement the model tracks. "
            "Zero means it does no better than predicting the average.",
        )


def render_retrain_control() -> None:
    """The offset, its age, and a button to refit it on the last 90 days."""

    from utils.bmo.coke_calibration import NO_CALIBRATION, load_calibration
    from utils.bmo.coke_history import refit_calibration

    calib = load_calibration()
    age = calib.age_days()

    needs_refit = calib is NO_CALIBRATION or not calib.is_usable or calib.is_stale()
    with st.container(border=True):
        left, right = st.columns([4, 1], vertical_alignment="center")
        with left:
            st.markdown("**Energy-balance offset**")
            if calib is NO_CALIBRATION or not calib.is_usable:
                detail = " ".join(calib.notes or [])
                st.badge(
                    "No usable calibration", icon=":material/warning:", color="orange"
                )
                st.caption(
                    "Fuel cost is falling back to the observed coke rate. Refit to "
                    "switch the physics anchor on." + (f" {detail}" if detail else "")
                )
            elif calib.is_stale():
                st.badge(
                    f"Offset {calib.offset_kg_per_thm:+,.1f} kg/THM · {age} days old",
                    icon=":material/warning:",
                    color="orange",
                )
                st.caption(
                    "Past the conservative refresh window. Refit it against the "
                    "latest measured plant coke rate before trusting the correction."
                )
            else:
                st.badge(
                    f"Offset {calib.offset_kg_per_thm:+,.1f} kg/THM",
                    icon=":material/check_circle:",
                    color="green",
                )
                st.caption(
                    f"Fitted {'today' if age == 0 else f'{age} days ago'} on "
                    f"{calib.sample_days} days ({calib.first_day} to "
                    f"{calib.last_day}); day-to-day scatter "
                    f"±{calib.residual_sd_kg_per_thm:,.0f} kg/THM."
                )
        with right:
            clicked = st.button(
                "Refit on last 90 days",
                icon=":material/refresh:",
                width="stretch",
                type="primary" if needs_refit else "secondary",
                help="Rebuilds the daily history from the plant record and refits "
                "the bias offset over the trailing 90 days. Takes a minute or "
                "two: it queries the offline tables day by day.",
            )

    if clicked:
        with st.status("Refitting the coke-rate offset…", expanded=True) as status:
            st.write("Assembling daily charge, DPR and process history…")
            try:
                new_calib, history = refit_calibration(
                    days=HISTORY_DAYS, window=CALIBRATION_WINDOW_DAYS
                )
            except Exception as exc:  # noqa: BLE001 - a failed refit must not
                # take the page down; the previous calibration stays in force.
                log.exception("Coke calibration refit failed")
                status.update(label="Refit failed", state="error")
                st.error(
                    f"**Could not refit.** {_readable_failure(exc)}\n\n"
                    "The previous calibration is still in force."
                )
                return

            if not new_calib.is_usable:
                status.update(label="Refit produced nothing usable", state="error")
                st.error(
                    "The refit did not find enough paired days. "
                    + " ".join(new_calib.notes or [])
                )
                return

            st.write(
                f"Fitted on {new_calib.sample_days} days "
                f"({new_calib.first_day} → {new_calib.last_day})."
            )
            moved = new_calib.offset_kg_per_thm - calib.offset_kg_per_thm
            status.update(
                label=(
                    f"Offset now {new_calib.offset_kg_per_thm:+,.1f} kg/THM "
                    f"({moved:+,.1f} from before)"
                ),
                state="complete",
            )
            for warning in history.warnings:
                st.caption(f"⚠️ {warning}")
        # Both the chart cache and the anchor read the stored calibration, so
        # everything downstream has to be rebuilt from the new one.
        _coke_history.clear()
        st.session_state["bmo_calibration_bust"] = (
            int(st.session_state.get("bmo_calibration_bust", 0)) + 1
        )
        st.rerun()


def render_coke_accuracy(days: int = HISTORY_DAYS) -> None:
    """Predicted vs realised coke rate, at the PCI and nut coke actually run."""

    from utils.bmo.coke_calibration import load_calibration

    bust = int(st.session_state.get("bmo_calibration_bust", 0))
    try:
        frame, warnings, excluded = _coke_history(days, bust)
    except Exception as exc:  # noqa: BLE001
        log.exception("Coke history failed")
        st.error(f"Could not build the coke history: {exc}")
        return

    if frame.empty or "predicted_coke" not in frame:
        st.info("No paired days available yet.")
        return

    calib = load_calibration()
    work = frame.copy()
    # The chart shows what the page actually reports, which is the corrected
    # figure. The raw series is kept alongside so the offset's size is visible
    # rather than merely stated.
    work["corrected"] = (
        work["predicted_coke"] - calib.offset_kg_per_thm
        if calib.is_usable
        else work["predicted_coke"]
    )
    paired = (
        work[["corrected", "actual_coke"]].replace([np.inf, -np.inf], np.nan).dropna()
    )

    corrected_scores = _scores(paired["corrected"], paired["actual_coke"])
    st.markdown(
        "#### Energy balance vs measured coke rate"
        if calib.is_usable
        else "#### Uncorrected energy balance vs measured coke rate"
    )
    st.caption(
        "Daily. Predicted is the closed energy balance solved at each day's own "
        "PCI, nut coke, blast and burden, "
        + (
            "less the bias offset. "
            if calib.is_usable
            else "with no offset, because no compatible calibration is active. "
        )
        + "Measured is 1,000 x COKE_CALC_MT / hot-metal tonnes, mass-weighted "
        "over each complete day (not the operator's set point). Both are the same "
        "day, so a gap is a real disagreement and not a lag. The band is this "
        "window's typical error; scores are for this window only. A prediction "
        "without a measured point is a day the balance solved but the plant "
        "record could not score."
    )
    _score_row(corrected_scores, "kg/THM")
    st.plotly_chart(
        _paired_chart(
            # Every solved day, measured or not: dropping unmeasured days here
            # used to leave holes that looked like the model had stopped.
            work.dropna(subset=["corrected"]),
            predicted_col="corrected",
            actual_col="actual_coke",
            unit="kg/THM",
            predicted_name=(
                "Energy balance + offset" if calib.is_usable else "Energy balance"
            ),
            actual_name="Measured plant rate",
            # The band is the error MEASURED ON THIS CHART, not the sd recorded
            # when the calibration was fitted. Those two can differ (the fit
            # drops outlier days and this does not), and a band that disagrees
            # with the points drawn inside it is worse than no band.
            band=round(corrected_scores["MAE"]) if corrected_scores else None,
        ),
        width="stretch",
    )
    _render_unscored_days(excluded)

    with st.expander("The raw balance, before the offset", expanded=False):
        raw_scores = _scores(work["predicted_coke"], work["actual_coke"])
        _score_row(raw_scores, "kg/THM")
        st.plotly_chart(
            _paired_chart(
                work.dropna(subset=["predicted_coke"]),
                predicted_col="predicted_coke",
                actual_col="actual_coke",
                unit="kg/THM",
                predicted_name="Uncorrected energy balance",
                actual_name="Measured plant rate",
            ),
            width="stretch",
        )
        st.caption(
            "The shape is right and the level is not, which is exactly what one "
            "offset can fix and a fitted residual model cannot improve on without "
            "arguing with the physics it is correcting."
        )

    _render_control_context(work)
    for warning in warnings:
        st.caption(f"⚠️ {warning}")


def _render_unscored_days(excluded: pd.DataFrame) -> None:
    """Say why days are missing, so the chart's gaps read as record gaps."""

    if excluded is None or excluded.empty:
        return
    counts = excluded["reason"].value_counts()
    st.caption(
        f"**{len(excluded)} days in this window could not be scored:** "
        + "; ".join(
            f"{reason[:1].lower()}{reason[1:]} ({count})"
            for reason, count in counts.items()
        )
        + ". These are gaps in the plant record, not in the model."
    )
    with st.expander("Days not scored and why", expanded=False):
        table = excluded.reset_index().rename(
            columns={"date": "Day", "reason": "Reason", "detail": "Detail"}
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Day": st.column_config.DateColumn("Day", format="DD MMM YYYY"),
            },
        )


def _render_control_context(frame: pd.DataFrame) -> None:
    """PCI, nut coke and the operator's setpoint over the same window.

    The coke rate is not a free variable — it is what is left after PCI and nut
    coke, both of which the operator sets. Showing them under the accuracy chart
    is what makes a divergence readable: a jump in the residual that coincides
    with PCI being cut is the balance responding correctly to a real change, not
    a model failure.
    """

    columns = [
        c
        for c in ("pci_kg_thm", "nut_coke_kg_thm", "coke_setpoint_kg_thm")
        if c in frame.columns
    ]
    if not columns:
        return

    with st.expander("What PCI and nut coke were doing", expanded=False):
        styles = {
            "pci_kg_thm": ("PCI", dict(color=_TRACKING_TEAL, width=2.2)),
            "nut_coke_kg_thm": ("Nut coke", dict(color=_CONTEXT_AMBER, width=2.2)),
            "coke_setpoint_kg_thm": (
                "Coke set point",
                dict(color=_TRACKING_NAVY, width=1.8, dash="dot"),
            ),
        }
        plotted = _break_gaps(frame[columns])
        fig = go.Figure()
        for column in columns:
            name, line = styles[column]
            fig.add_trace(
                go.Scatter(
                    x=plotted.index,
                    y=plotted[column],
                    mode="lines",
                    name=name,
                    line=line,
                    hovertemplate=f"%{{y:,.1f}} kg/THM<extra>{name}</extra>",
                )
            )
        _style_figure(fig, unit="kg/THM", hourly=False, height=300)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "The set point is the operator's instruction, not a measurement: it "
            "sits flat for days and then steps. The measured plant coke rate is "
            "the navy points in the chart above."
        )


def render_si_accuracy(days: int = 180) -> None:
    """Predicted vs measured hot-metal silicon, from the shipped Si model."""

    try:
        frame, report = _si_history(days)
    except Exception as exc:  # noqa: BLE001
        log.exception("Si history failed")
        st.error(f"Could not build the silicon history: {exc}")
        return

    if frame.empty:
        st.info(
            "Silicon history unavailable. " + " ".join(report.get("notes", []) or [])
        )
        return

    if not report.get("trustworthy", False):
        # The Si model wants 194 features and the static dataset carries 112.
        # The rest are rebuilt here. If too many had to be filled with medians,
        # the chart would be drawing the fill rather than the model — and it
        # would look perfectly reasonable while doing so.
        st.warning(
            f"**Not showing this chart.** The Si model needs "
            f"{report['derived'] + report['filled']} inputs and "
            f"{report['filled']} of them could not be rebuilt from the stored "
            "dataset, so they were filled with typical values. A chart drawn "
            "from that would be measuring the fill, not the model."
        )
        with st.expander("Which inputs are missing", expanded=False):
            st.write(report.get("filled_names") or [])
        return

    st.markdown("#### Hot-metal silicon: model vs cast analysis")
    si_scores = _scores(frame["predicted_si"], frame["actual_si"])
    _score_row(si_scores, "%", decimals=3)
    st.plotly_chart(
        _paired_chart(
            frame,
            predicted_col="predicted_si",
            actual_col="actual_si",
            unit="Si %",
            actual_name="Cast analysis",
            # Same rule as the coke chart: the band is the typical error measured
            # on this window, so the points drawn inside it agree with it.
            band=round(si_scores["MAE"], 3) if si_scores else None,
            decimals=3,
        ),
        width="stretch",
    )
    st.caption(
        "Measured is the silicon in the cast analysis, averaged over the day. "
        "**Read the level with care:** among the model's inputs are earlier "
        "silicon readings, so part of what looks like skill here is simply "
        "yesterday's cast carried forward. It is a fair reflection of what the "
        "model does in service — an operator does know the last cast — but it "
        "is not evidence that the burden chemistry terms are doing the work."
    )
    for note in report.get("notes", []) or []:
        st.caption(note)


def _direct_coke_settings() -> tuple[dict[str, Any], Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    settings = yaml.safe_load(
        (repo_root / "src/config/setting_bmo.yml").read_text(encoding="utf-8")
    )["bmo"]
    cfg = dict(settings.get("data_driven_coke", {}) or {})

    def resolve(value: str, fallback: str) -> Path:
        path = Path(str(value or fallback))
        return path if path.is_absolute() else repo_root / path

    return (
        cfg,
        resolve(cfg.get("bundled_model_dir", ""), "src/assets/models/bmo_coke_robust"),
        resolve(cfg.get("deployment_dir", ""), "src/storage/bmo_coke_model"),
        resolve(cfg.get("dataset_path", ""), "src/assets/data/furnace_dataset.csv"),
    )


@st.cache_data(ttl=300, show_spinner=False)
def _data_driven_coke_history(
    *,
    dataset_path: str,
    dataset_mtime_ns: int,
    bundled_dir: str,
    deployment_dir: str,
    deployment_mtime_ns: int,
    bias_window_hours: float,
    bias_min_periods: int,
    history_days: int,
) -> pd.DataFrame:
    from utils.bmo.direct_coke_model import DirectCokeModelService

    del dataset_mtime_ns, deployment_mtime_ns
    service = DirectCokeModelService(
        bundled_dir=bundled_dir,
        deployment_dir=deployment_dir,
    )
    return service.prediction_history(
        dataset_path,
        bias_window_hours=bias_window_hours,
        bias_min_periods=bias_min_periods,
        history_days=history_days,
    )


def _prediction_error(
    frame: pd.DataFrame, prediction_column: str
) -> tuple[float | None, float | None, int]:
    paired = frame[[prediction_column, "actual_coke_kg_per_thm"]].dropna()
    if paired.empty:
        return None, None, 0
    error = paired[prediction_column] - paired["actual_coke_kg_per_thm"]
    return float(error.abs().mean()), float(error.mean()), int(len(error))


def _recent_tracking_window(
    history: pd.DataFrame, days: int = TRACKING_DISPLAY_DAYS
) -> pd.DataFrame:
    """Return only the requested trailing time span for tracking and scores."""

    if history.empty:
        return history.copy()
    latest = history.index.max()
    cutoff = latest - pd.Timedelta(days=max(1, int(days)))
    return history.loc[history.index >= cutoff].copy()


def _plant_time(index: pd.Index) -> pd.Index:
    """Show UTC hours in IST. Plain calendar days are left as they are."""

    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        return index
    return index.tz_convert(_PLANT_TZ).tz_localize(None)


def _break_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Put missing periods back as empty rows so lines break instead of bridging.

    Without this a fortnight with no paired days is drawn as one confident
    straight line, which reads as data.
    """

    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame) < 3:
        return frame
    step = frame.index.to_series().diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return frame
    grid = pd.date_range(frame.index.min(), frame.index.max(), freq=step)
    # Only an index that genuinely sits on a regular grid is filled; the cap
    # guards the browser against a mis-inferred, very fine step.
    if len(grid) > 20_000 or not frame.index.isin(grid).all():
        return frame
    return frame.reindex(grid)


def _style_figure(fig: go.Figure, *, unit: str, hourly: bool, height: int = 360) -> None:
    """Shared layout: legend on its own row (no Plotly title to collide with),
    transparent background for both themes, units on the axis and in hovers."""

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=62, b=10),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            tickformat="%d %b<br>%H:%M" if hourly else "%d %b<br>%Y",
            hoverformat="%d %b %Y, %H:%M IST" if hourly else "%d %b %Y",
            nticks=8,
        ),
        yaxis=dict(
            title=unit,
            gridcolor="rgba(120,130,145,0.18)",
            zeroline=False,
            ticksuffix=" ",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )


def _prediction_figure(
    frame: pd.DataFrame,
    *,
    predicted_col: str,
    actual_col: str,
    unit: str,
    predicted_name: str = "Prediction",
    actual_name: str = "Actual",
    band: float | None = None,
    band_label: str | None = None,
    decimals: int = 1,
    hourly: bool = False,
) -> go.Figure:
    """Model line over measured points, with an optional typical-error band.

    The band is drawn around the PREDICTION, so a measured point inside it is
    one the model called within its usual error.
    """

    plotted = _break_gaps(frame[[predicted_col, actual_col]])
    x = _plant_time(plotted.index)
    hover = f"%{{y:,.{decimals}f}} {unit}"
    fig = go.Figure()
    if band:
        band_x, band_y = _band_outline(x, plotted[predicted_col], band)
        # One closed shape per unbroken run of predictions. A between-traces fill
        # would bridge every gap in the record with a wedge.
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor=_BAND_FILL,
                hoverinfo="skip",
                legendrank=3,
                name=band_label or f"±{band:,.{decimals}f} {unit} typical error",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=plotted[predicted_col],
            name=predicted_name,
            # Dashed line through the prediction points: the eye reads the gap
            # between each teal point and its navy measurement directly, and a
            # day with no neighbours still shows as a point.
            mode="lines+markers",
            line=dict(color=_TRACKING_TEAL, width=1.8, dash="dash"),
            marker=dict(color=_TRACKING_TEAL, size=5 if hourly else 6),
            legendrank=1,
            hovertemplate=f"{hover}<extra>{predicted_name}</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=plotted[actual_col],
            name=actual_name,
            mode="markers",
            marker=dict(
                color=_TRACKING_NAVY,
                size=6,
                opacity=0.85,
                line=dict(color="rgba(255,255,255,0.9)", width=1),
            ),
            legendrank=2,
            hovertemplate=f"{hover}<extra>{actual_name}</extra>",
        )
    )
    _style_figure(fig, unit=unit, hourly=hourly)
    fig.update_layout(legend_traceorder="normal")
    return fig


def _band_outline(
    x: pd.Index, centre: pd.Series, band: float
) -> tuple[list[Any], list[Any]]:
    """Closed outline of centre ± band for each run without gaps, None-separated."""

    xs: list[Any] = []
    ys: list[Any] = []
    values = centre.to_numpy(dtype=float)
    positions = list(x)
    run: list[int] = []
    for i, value in enumerate([*values, np.nan]):
        if np.isfinite(value):
            run.append(i)
            continue
        if run:
            forward = [positions[j] for j in run]
            xs.extend([*forward, *forward[::-1], forward[0], None])
            ys.extend(
                [
                    *(values[j] + band for j in run),
                    *(values[j] - band for j in run[::-1]),
                    values[run[0]] + band,
                    None,
                ]
            )
            run = []
    return xs, ys


def _coke_tracking_figure(
    plotted: pd.DataFrame, band: float | None = None
) -> go.Figure:
    """Hourly Data-Driven prediction against the robust target, in IST."""

    return _prediction_figure(
        plotted,
        predicted_col="raw_predicted_coke_kg_per_thm",
        actual_col="actual_coke_kg_per_thm",
        unit="kg/THM",
        band=band,
        band_label=(
            f"±{band:,.1f} kg/THM typical error on unseen days" if band else None
        ),
        hourly=True,
    )


def _ist_text(value: Any, fmt: str = "%d %b %Y, %H:%M IST") -> str | None:
    """Plant-time text from an ISO timestamp or a ``YYYYmmddTHHMMSSffffffZ`` id."""

    if value in (None, ""):
        return None
    text = str(value)
    try:
        stamp = pd.Timestamp(datetime.strptime(text, "%Y%m%dT%H%M%S%fZ"), tz="UTC")
    except ValueError:
        try:
            stamp = pd.Timestamp(text)
        except (TypeError, ValueError):
            return None
        if pd.isna(stamp):
            return None
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
    return stamp.tz_convert(_PLANT_TZ).strftime(fmt)


def _finite(metrics: dict[str, Any], name: str) -> float | None:
    try:
        number = float(metrics.get(name))
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _kg_text(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+,.1f} kg/THM" if signed else f"{value:,.1f} kg/THM"


def _deployment_checks(
    validation: dict[str, Any], gates: dict[str, Any]
) -> list[tuple[bool | None, str, str]]:
    """The four deployment gates as (passed, badge label, explanation).

    ``passed`` is None when the active model predates the gate and never
    recorded that score.
    """

    random_day = dict(validation.get("random_day", {}) or {})
    later = dict(validation.get("later_time", {}) or {})
    fuel = dict(validation.get("fuel_response", {}) or {})
    r2 = _finite(random_day, "r2")
    mae = _finite(later, "mae")
    skill = _finite(later, "skill_vs_last_week_level")
    replacement = _finite(fuel, "pci_replacement_kg_per_kg")
    low, high = gates["pci_replacement_range"]
    held_days = int(random_day.get("held_out_days", 0) or 0)

    if skill is None:
        skill_label = "Compared with last week's level"
    elif skill >= 0:
        skill_label = f"{skill:.0%} better than last week's level"
    else:
        skill_label = f"{-skill:.0%} worse than last week's level"
    return [
        (
            None if r2 is None else r2 >= gates["min_random_r2"],
            f"R² {r2:.2f} on unseen days" if r2 is not None else "R² on unseen days",
            f"Share of the day-to-day movement explained on {held_days} whole days "
            f"held out at random. Needs at least {gates['min_random_r2']:.2f}.",
        ),
        (
            None if mae is None else mae <= gates["max_later_time_mae"],
            f"{mae:.1f} kg/THM typical error, latest 14 days"
            if mae is not None
            else "Latest 14 days",
            "Typical error on the most recent 14 days, which the model was not "
            f"trained on. Needs at most {gates['max_later_time_mae']:.1f} kg/THM.",
        ),
        (
            None if skill is None else skill >= gates["min_later_time_skill"],
            skill_label,
            "Error on the latest 14 days compared with simply assuming last week's "
            "average coke rate carries on. The model must do at least as well.",
        ),
        (
            None if replacement is None else low <= replacement <= high,
            f"{replacement:.2f} kg coke per kg PCI"
            if replacement is not None
            else "PCI response",
            "Coke rise when PCI is cut, averaged over recent operating states. It "
            f"must lie between {low:.2f} and {high:.2f}; within-month plant data "
            "gives 0.65-0.95.",
        ),
    ]


def _render_check_badges(checks: list[tuple[bool | None, str, str]]) -> None:
    with st.container(horizontal=True, gap="small"):
        for passed, label, explanation in checks:
            if passed is None:
                st.badge(
                    f"{label}: not recorded",
                    icon=":material/help:",
                    color="gray",
                    help=explanation,
                )
            elif passed:
                st.badge(
                    label, icon=":material/check_circle:", color="green", help=explanation
                )
            else:
                st.badge(label, icon=":material/cancel:", color="red", help=explanation)


def _auto_retrain_text(state: dict[str, Any]) -> str:
    when = _ist_text(state.get("last_attempt_utc")) or "unknown time"
    if state.get("error"):
        return f"Last automatic run {when}: failed ({state['error']}); the live version was kept."
    if state.get("deployed"):
        return f"Last automatic run {when}: new version passed every check and went live."
    reasons = " ".join(map(str, state.get("reasons") or []))
    return f"Last automatic run {when}: the live version was kept. {reasons}".rstrip()


def render_data_driven_coke_accuracy() -> None:
    """Live model status, accuracy on unseen days, tracking, and retraining."""

    from utils.bmo.direct_coke_model import (
        DirectCokeModelService,
        auto_retrain_state,
        retrain_and_maybe_deploy,
        retrain_kwargs_from_config,
    )

    cfg, bundled_dir, deployment_dir, dataset_path = _direct_coke_settings()
    retrain_cfg = dict(cfg.get("retraining", {}) or {})
    gates = retrain_kwargs_from_config(retrain_cfg)
    try:
        service = DirectCokeModelService(
            bundled_dir=bundled_dir,
            deployment_dir=deployment_dir,
            max_stale_hours=float(cfg.get("max_input_stale_hours", 6.0)),
            max_source_age_hours=float(cfg.get("max_source_age_hours", 6.0)),
        )
        status = service.status()
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not load direct coke model")
        st.error(f"Could not load the direct coke-rate model: {_readable_failure(exc)}")
        return

    validation = dict(status.get("validation", {}) or {})
    later_metrics = dict(validation.get("later_time", {}) or validation.get("test", {}) or {})
    random_metrics = dict(validation.get("random_day", {}) or {})
    fuel_response = dict(validation.get("fuel_response", {}) or {})
    window_hours = status.get("target_window_hours")
    checks = _deployment_checks(validation, gates)
    deployment_id = str(status.get("deployment_id", "bundled"))
    current = st.session_state.get("bmo_data_driven_coke_prediction", {}) or {}
    current_value = (
        _finite(current, "value_kg_per_thm") if current.get("usable") else None
    )

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            if not window_hours:
                st.badge(
                    "Earlier hourly-target model",
                    icon=":material/warning:",
                    color="orange",
                )
            elif all(passed for passed, _label, _help in checks):
                st.badge(
                    "Live · passed every check", icon=":material/verified:", color="green"
                )
            else:
                st.badge(
                    "Live · a check is not met", icon=":material/warning:", color="orange"
                )
            trained = _ist_text(deployment_id) if deployment_id != "bundled" else None
            data_end = _ist_text(status.get("fit_cutoff"), "%d %b %Y")
            months = int(service.config.get("months", 12) or 12)
            st.caption(
                (f"Version of {trained}" if trained else "Shipped version")
                + f" · trained on {months} months of plant data"
                + (f" up to {data_end}" if data_end else "")
                + (f" · {int(window_hours)}-hour coke rate" if window_hours else ""),
                help=f"Deployment id: {deployment_id}",
            )

        cols = st.columns(4)
        cols[0].metric(
            "Current prediction",
            _kg_text(current_value),
            help=(
                "The coke rate the optimiser is using now, for the operator's "
                "current PCI and nut coke rates"
                + (
                    f", windows ending {_ist_text(current.get('window_end_utc'))}."
                    if current_value is not None
                    else ". Shown once the optimiser has evaluated the current state."
                )
            ),
        )
        cols[1].metric(
            "Typical error, unseen days",
            _kg_text(_finite(random_metrics, "mae")),
            help=(
                "Mean absolute error on whole days held out at random from training"
                + (
                    f" (R² {_finite(random_metrics, 'r2'):.2f})."
                    if _finite(random_metrics, "r2") is not None
                    else "."
                )
            ),
        )
        cols[2].metric(
            "Typical error, latest 14 days",
            _kg_text(_finite(later_metrics, "mae")),
            help=(
                "Mean absolute error on the most recent 14 days, predicted by a model "
                "trained only on earlier data"
                + (
                    f"; average offset {_finite(later_metrics, 'bias'):+.1f} kg/THM."
                    if _finite(later_metrics, "bias") is not None
                    else "."
                )
            ),
        )
        cols[3].metric(
            "Coke rise if PCI falls 40 kg/THM",
            _kg_text(_finite(fuel_response, "coke_change_for_pci_minus_40"), signed=True),
            help=(
                "Predicted coke-rate change for 40 kg/THM less PCI, averaged over "
                "recent operating states. Nut coke 10 kg/THM lower: "
                + _kg_text(
                    _finite(fuel_response, "coke_change_for_nut_coke_minus_10"),
                    signed=True,
                )
                + "."
            ),
        )
        _render_check_badges(checks)

    if current and not current.get("usable"):
        reasons = " ".join(map(str, current.get("reasons", []) or []))
        st.warning("The current prediction is on hold: " + reasons)
        latest_inputs = current.get("latest_input_diagnostics", {}) or {}
        if latest_inputs:
            st.caption(
                "Latest-row inputs: "
                f"burden {float(latest_inputs.get('burden_mt') or 0):,.1f} MT, "
                f"production {float(latest_inputs.get('production_mt_per_hr') or 0):,.1f} MT/h."
            )

    bias_window_hours = float(cfg.get("bias_correction_window_hours", 24.0))
    bias_min_periods = int(cfg.get("bias_correction_min_periods", 6))
    history_days = int(cfg.get("accuracy_history_days", 14))
    active_pointer = deployment_dir / "active.json"
    try:
        history = _data_driven_coke_history(
            dataset_path=str(dataset_path),
            dataset_mtime_ns=(
                int(dataset_path.stat().st_mtime_ns) if dataset_path.is_file() else 0
            ),
            bundled_dir=str(bundled_dir),
            deployment_dir=str(deployment_dir),
            deployment_mtime_ns=(
                int(active_pointer.stat().st_mtime_ns)
                if active_pointer.is_file()
                else 0
            ),
            bias_window_hours=bias_window_hours,
            bias_min_periods=bias_min_periods,
            history_days=history_days,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not build direct coke prediction history")
        st.warning(
            "Could not build the recent predicted-vs-actual coke chart: "
            f"{_readable_failure(exc)}"
        )
    else:
        if history.empty:
            st.info("No eligible paired model prediction and coke-mass rows are available.")
        else:
            plotted = _recent_tracking_window(history)
            mae, _bias, _paired_n = _prediction_error(
                plotted, "raw_predicted_coke_kg_per_thm"
            )
            st.markdown("#### Predicted vs actual coke rate")
            basis = (
                f"{int(window_hours)}-hour coke rate"
                if window_hours
                else "Hourly coke rate"
            )
            unseen_mae = _finite(random_metrics, "mae")
            st.caption(
                f"{basis}, last {TRACKING_DISPLAY_DAYS * 24} hours, IST. These hours "
                "are part of the live model's training data, so a close fit is "
                "expected"
                + (f" (average gap {mae:.1f} kg/THM)" if mae is not None else "")
                + (
                    f". The band is ±{unseen_mae:.1f} kg/THM, the model's typical "
                    "error on days it had not seen: expect new days to land inside it."
                    if unseen_mae is not None
                    else "; the error figures above are measured on days it had not seen."
                )
            )
            st.plotly_chart(
                _coke_tracking_figure(plotted, band=unseen_mae), width="stretch"
            )

    with st.container(border=True):
        left, right = st.columns([4, 1], vertical_alignment="center")
        with left:
            st.markdown("**Model updates**")
            auto = auto_retrain_state(deployment_dir)
            st.caption(
                "Retrains automatically once a day when new plant data arrives. A new "
                "version goes live only if it passes all four checks above; the "
                "last 30 versions are kept for rollback."
                + (f" {_auto_retrain_text(auto)}" if auto else "")
            )
        clicked = right.button(
            "Retrain now",
            key="bmo_retrain_direct_coke",
            icon=":material/refresh:",
            width="stretch",
            help=(
                "Rebuilds the 24-hour coke target and features from the furnace "
                "dataset, re-runs all four checks, and puts the new version live "
                "only if it passes every one."
            ),
        )
    if clicked:
        with st.status("Training and checking a new version...", expanded=True) as progress:
            try:
                report = retrain_and_maybe_deploy(
                    dataset_path,
                    bundled_dir=bundled_dir,
                    deployment_dir=deployment_dir,
                    **gates,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("Direct coke model retraining failed")
                progress.update(label="Retraining failed", state="error")
                st.error(_readable_failure(exc))
                return

            st.session_state["bmo_direct_coke_retrain_report"] = report.to_dict()
            summary = (
                f"error on unseen days {_kg_text(_finite(report.random_metrics, 'mae'))}, "
                f"latest 14 days {_kg_text(_finite(report.later_time_metrics, 'mae'))}, "
                "PCI -40 "
                + _kg_text(
                    _finite(report.fuel_response, "coke_change_for_pci_minus_40"),
                    signed=True,
                )
            )
            if report.deployed:
                progress.update(
                    label=f"New version is live: {summary}", state="complete"
                )
                st.success(
                    "The new version passed every check and is now live. Earlier "
                    "versions are kept for rollback."
                )
            else:
                progress.update(
                    label="New version not deployed; the live version is unchanged",
                    state="error",
                )
                st.error(f"{summary[0].upper()}{summary[1:]}. " + " ".join(report.reasons))

    with st.expander("How the model is built and checked", expanded=False):
        later_r2 = _finite(later_metrics, "r2")
        st.markdown(
            "- **What it predicts:** the coke rate over the last 24 hours, as coke "
            "charged per tonne of Fe charged × Fe per tonne of hot metal (over a "
            "week). Coke and ore are counted on the same charges, so charge-to-"
            "charge counting cancels; hourly `COKE_CALC_MT / production` does not.\n"
            "- **Data it ignores:** days whose coke mass disagrees with the set "
            "point by more than -20% / +25% (e.g. the Nov 2025 undercount) and "
            "single-hour weighing spikes.\n"
            "- **Inputs:** 24-hour PCI, nut coke, flux, burden shares and grade, "
            "slag, blast, top gas, burden distribution and key raw-material "
            "analyses. No coke mass, reported coke rate, fuel rate or unit cost.\n"
            "- **Physics guard:** more PCI or nut coke can only lower the predicted "
            "coke; the plant data sets by how much.\n"
            "- **Checks:** whole days held out at random, and the latest 14 days "
            "held out entirely. R² is not used on the latest 14 days"
            + (f" (it is {later_r2:.2f})" if later_r2 is not None else "")
            + ": a quiet fortnight moves only a few kg/THM, so R² there measures "
            "the fortnight rather than the model. The error in kg/THM and the "
            "comparison with last week's level are used instead.\n"
            "- The model estimates the coke rate for given conditions. It does not "
            "prove a lower coke rate is achievable."
        )


def render_model_accuracy_tab() -> None:
    """Data-Driven coke validation first; supporting models remain inspectable."""

    st.markdown("##### Data-Driven coke-rate model")
    render_data_driven_coke_accuracy()
    with st.expander("Energy-balance calibration and recent accuracy", expanded=False):
        render_retrain_control()
        render_coke_accuracy()
    with st.expander("Hot-metal silicon model", expanded=False):
        render_si_accuracy()
