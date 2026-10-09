"""Review scorecard for the charged-coke engine, from forecasts as issued.

Promotion of a retrained model, or of this one to the optimiser's default,
should weigh more than overall MAE:

* errors by display state (cruise / wider range), not pooled;
* skill against persistence (carrying the current 4-h charged ratio);
* availability on the full hourly clock, with data downtime kept apart from
  process-related pauses;
* coverage of the displayed ranges;
* the PCI / nut-coke / slag response (signs and size of the coefficients).

Four-weekly review is a starting schedule, not an established optimum: the
research had two weekly refits over 20 days to go on. Rising novelty is a sign
that support needs widening, not only that a refit is due.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.bmo.charged_coke.policy import DISPLAY_STATES

DATA_STATES = frozenset({"data_missing", "data_hold"})


def scorecard(issued: pd.DataFrame, *, beta: np.ndarray | None = None) -> dict[str, Any]:
    """Accuracy, skill, coverage and availability of issued forecasts.

    Args:
        issued: ``ForecastLedger.frame()``: one row per issued data row with
            ``state``, ``prediction_kg_thm``, ``lower_kg_thm``/``upper_kg_thm``,
            ``anchor_charge4_kg_thm``, ``withheld_value_kg_thm`` and the matured
            ``actual_kg_thm``.
        beta: The model's PCI, nut and slag coefficients, reported with signs.

    Returns:
        ``by_state`` metrics, ``availability`` and ``response`` sections.
    """

    out: dict[str, Any] = {"hours": int(len(issued))}
    if issued.empty:
        return out
    states = issued["state"].astype(str)
    shown = states.isin(DISPLAY_STATES)
    out["availability"] = {
        "shown_share": float(shown.mean()),
        "data_pause_share": float(states.isin(DATA_STATES).mean()),
        "process_pause_share": float((~shown & ~states.isin(DATA_STATES)).mean()),
        "by_state": states.value_counts().to_dict(),
    }

    def metrics(frame: pd.DataFrame, value_column: str) -> dict[str, Any]:
        frame = frame.dropna(subset=[value_column, "actual_kg_thm", "anchor_charge4_kg_thm"])
        if frame.empty:
            return {"n": 0}
        error = frame[value_column] - frame["actual_kg_thm"]
        naive = frame["anchor_charge4_kg_thm"] - frame["actual_kg_thm"]
        result = {
            "n": int(len(frame)),
            "days": int(pd.DatetimeIndex(frame.index).floor("D").nunique()),
            "mae": float(error.abs().mean()),
            "rmse": float(np.sqrt((error**2).mean())),
            "bias": float(error.mean()),
            "persistence_mae": float(naive.abs().mean()),
            "skill_vs_persistence": float(1 - error.abs().mean() / naive.abs().mean()) if naive.abs().mean() > 0 else None,
            "p90_abs_error": float(error.abs().quantile(0.9)),
        }
        if {"lower_kg_thm", "upper_kg_thm"} <= set(frame.columns):
            bounded = frame.dropna(subset=["lower_kg_thm", "upper_kg_thm"])
            if len(bounded):
                inside = bounded["actual_kg_thm"].between(bounded["lower_kg_thm"], bounded["upper_kg_thm"])
                result["range_coverage"] = float(inside.mean())
        return result

    by_state: dict[str, Any] = {}
    shown_rows = issued[shown]
    for state, frame in shown_rows.groupby(shown_rows["state"].astype(str)):
        by_state[state] = metrics(frame, "prediction_kg_thm")
    by_state["all_shown"] = metrics(shown_rows, "prediction_kg_thm")
    # Withheld values are scored separately so pauses can be reviewed, never
    # mixed into the shown forecasts' accuracy.
    if "withheld_value_kg_thm" in issued:
        by_state["withheld"] = metrics(issued[~shown], "withheld_value_kg_thm")
    out["by_state"] = by_state
    if beta is not None:
        b = np.asarray(beta, dtype=float)
        out["response"] = {
            "pci_kg_per_kg": float(b[0]), "nut_kg_per_kg": float(b[1]), "slag_kg_per_kg": float(b[2]),
            "signs_as_expected": bool(b[0] < 0 and b[1] < 0 and b[2] >= 0),
        }
    return out


__all__ = ["DATA_STATES", "scorecard"]
