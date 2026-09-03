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

WHY REFITTING IS NOT OPTIONAL. Measured over 281 days, letting a calibration
sit without refitting:

    held for     MAE   MAPE%      R2
        0 d     13.9    4.59   +0.428
       30 d     18.9    6.34   +0.054
       90 d     33.0   11.12   -1.232

At ninety days the correction is worse than applying none at all. The bias drifts
about 3.3 kg/THM per month, which is why the button warns after fourteen days
rather than after a quarter.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger(__name__)

HISTORY_DAYS = 120
CALIBRATION_WINDOW_DAYS = 90
# Plotly template that reads on both themes without hard-coding a background.
_LAYOUT = dict(
    margin=dict(l=10, r=10, t=34, b=10),
    height=340,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
_PREDICTED = "#f2a03d"
_ACTUAL = "#2f6fd0"


@st.cache_data(ttl=1800, show_spinner=False)
def _coke_history(days: int, cache_bust: int) -> tuple[pd.DataFrame, list[str]]:
    """Daily predicted-vs-realised coke. ``cache_bust`` forces a refetch."""

    from utils.bmo.coke_history import build_daily_history

    result = build_daily_history(days)
    return result.frame, list(result.warnings)


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
    if "pg_hba.conf" in text or "could not connect" in text or (
        "connection to server" in text
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

    both = pd.concat([predicted, actual], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
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
        "R2": float(1.0 - float((err ** 2).sum()) / ss_tot) if ss_tot else float("nan"),
    }


def _paired_chart(
    frame: pd.DataFrame,
    *,
    predicted_col: str,
    actual_col: str,
    title: str,
    unit: str,
    band: float | None = None,
) -> go.Figure:
    """Predicted over measured, with the residual band the model is good to."""

    fig = go.Figure()
    if band:
        # The scatter the model is expected to leave. Drawn from the PREDICTION,
        # so a measured point inside the band is one the model called correctly.
        upper = frame[predicted_col] + band
        lower = frame[predicted_col] - band
        fig.add_trace(go.Scatter(
            x=frame.index, y=upper, mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False, name="",
        ))
        fig.add_trace(go.Scatter(
            x=frame.index, y=lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(242,160,61,0.16)",
            hoverinfo="skip", name=f"±{band:g} {unit} expected scatter",
        ))
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame[actual_col], mode="lines+markers",
        name="Measured", line=dict(color=_ACTUAL, width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame[predicted_col], mode="lines",
        name="Predicted", line=dict(color=_PREDICTED, width=2, dash="solid"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=unit, xaxis_title=None, **_LAYOUT,
    )
    return fig


def _score_row(scores: dict[str, float], unit: str, decimals: int = 1) -> None:
    if not scores:
        st.caption("Not enough paired days to score.")
        return
    cols = st.columns(4)
    cols[0].metric("Days scored", f"{scores['n']:,.0f}")
    cols[1].metric(f"Bias ({unit})", f"{scores['bias']:+,.{decimals}f}",
                   help="Average over-prediction. Near zero is the whole point "
                        "of the offset.")
    cols[2].metric(f"Typical error ({unit})", f"{scores['MAE']:,.{decimals}f}",
                   help="Mean absolute error — what to expect on any one day.")
    cols[3].metric("R²", f"{scores['R2']:+.2f}",
                   help="Share of the day-to-day movement the model tracks. "
                        "Zero means it does no better than predicting the average.")


def render_retrain_control() -> None:
    """The offset, its age, and a button to refit it on the last 90 days."""

    from utils.bmo.coke_calibration import NO_CALIBRATION, load_calibration
    from utils.bmo.coke_history import refit_calibration

    calib = load_calibration()
    age = calib.age_days()

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        if calib is NO_CALIBRATION or not calib.is_usable:
            st.warning(
                "**No usable calibration on file.** The energy balance runs about "
                "20 kg/THM high uncorrected, so the fuel cost is falling back to "
                "the observed coke rate. Refit to switch the physics anchor on."
            )
        elif calib.is_stale():
            st.warning(
                f"**Offset is {calib.offset_kg_per_thm:+,.1f} kg/THM, fitted "
                f"{age} days ago.** The bias drifts about 3.3 kg/THM a month, so "
                "this one is past its useful life — refit before trusting the "
                "coke rate."
            )
        else:
            st.success(
                f"**Offset {calib.offset_kg_per_thm:+,.1f} kg/THM**, fitted "
                f"{'today' if age == 0 else f'{age} days ago'} on "
                f"{calib.sample_days} days "
                f"({calib.first_day} → {calib.last_day}). Day-to-day scatter "
                f"±{calib.residual_sd_kg_per_thm:,.0f} kg/THM."
            )
    with right:
        clicked = st.button(
            "🔄 Retrain on last 90 days",
            width="stretch",
            type="primary" if (calib is NO_CALIBRATION or calib.is_stale()) else
            "secondary",
            help="Rebuilds the daily history from the plant record and refits "
                 "the bias offset over the trailing 90 days. Takes a minute or "
                 "two — it queries the offline tables day by day.",
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
        frame, warnings = _coke_history(days, bust)
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
    work["corrected"] = work["predicted_coke"] - calib.offset_kg_per_thm
    paired = work[["corrected", "actual_coke"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    corrected_scores = _scores(paired["corrected"], paired["actual_coke"])
    _score_row(corrected_scores, "kg/THM")

    fig = _paired_chart(
        work.dropna(subset=["corrected", "actual_coke"]),
        predicted_col="corrected", actual_col="actual_coke",
        title="Coke rate — energy balance + offset vs charge reports",
        unit="kg/THM",
        # The band is the error MEASURED ON THIS CHART, not the sd recorded when
        # the calibration was fitted. Those two can differ — the fit drops
        # outlier days and this does not — and a band that disagrees with the
        # points drawn inside it is worse than no band.
        band=round(corrected_scores["MAE"]) if corrected_scores else None,
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Measured is the coke actually charged, from the daily charge reports — "
        "not the operator's setpoint, which runs about 4% below it. Predicted is "
        "the closed energy balance solved at each day's own PCI, nut coke, blast "
        "and burden, less the bias offset. Both are on the same day, so a gap is "
        "a real disagreement and not a lag. **The scores above are for this "
        "window only** and will not match the figures quoted from the 239-day "
        "backtest; a quiet quarter scores better than a disturbed one."
    )

    with st.expander("The raw balance, before the offset", expanded=False):
        raw_scores = _scores(work["predicted_coke"], work["actual_coke"])
        _score_row(raw_scores, "kg/THM")
        st.plotly_chart(
            _paired_chart(
                work.dropna(subset=["predicted_coke", "actual_coke"]),
                predicted_col="predicted_coke", actual_col="actual_coke",
                title="Uncorrected energy balance",
                unit="kg/THM",
            ),
            width="stretch",
        )
        st.markdown(
            "The shape is right and the level is not — which is exactly what "
            "one offset can fix and a fitted residual model cannot improve on "
            "without arguing with the physics it is correcting."
        )

    _render_control_context(work)
    for warning in warnings:
        st.caption(f"⚠️ {warning}")


def _render_control_context(frame: pd.DataFrame) -> None:
    """PCI, nut coke and the operator's setpoint over the same window.

    The coke rate is not a free variable — it is what is left after PCI and nut
    coke, both of which the operator sets. Showing them under the accuracy chart
    is what makes a divergence readable: a jump in the residual that coincides
    with PCI being cut is the balance responding correctly to a real change, not
    a model failure.
    """

    columns = [c for c in ("pci_kg_thm", "nut_coke_kg_thm", "coke_setpoint_kg_thm")
               if c in frame.columns]
    if not columns:
        return

    with st.expander("What PCI and nut coke were doing", expanded=False):
        labels = {
            "pci_kg_thm": "PCI (kg/THM)",
            "nut_coke_kg_thm": "Nut coke (kg/THM)",
            "coke_setpoint_kg_thm": "Coke setpoint (kg/THM)",
        }
        fig = go.Figure()
        for column in columns:
            fig.add_trace(go.Scatter(
                x=frame.index, y=frame[column], mode="lines", name=labels[column]
            ))
        fig.update_layout(yaxis_title="kg/THM", xaxis_title=None, **_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "The setpoint is the operator's instruction, not a measurement — it "
            "sits flat for days and then steps. What was actually charged "
            "(the blue line above) runs about 4% above it."
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
            "Silicon history unavailable. "
            + " ".join(report.get("notes", []) or [])
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

    _score_row(_scores(frame["predicted_si"], frame["actual_si"]), "%", decimals=3)
    st.plotly_chart(
        _paired_chart(
            frame, predicted_col="predicted_si", actual_col="actual_si",
            title="Hot metal silicon — model vs cast analysis",
            unit="Si %",
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


def render_model_accuracy_tab() -> None:
    """The whole panel: retrain control, then coke, then silicon."""

    st.markdown("##### Is the coke-rate model working?")
    render_retrain_control()
    st.divider()
    render_coke_accuracy()
    st.divider()
    st.markdown("##### Hot metal silicon")
    render_si_accuracy()
