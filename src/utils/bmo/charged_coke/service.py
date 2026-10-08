"""End-to-end charged-coke forecasts and display states for a run of hours.

    source rows -> build_features -> gate inputs + model -> novelty -> policy

``forecast_hours`` is a port of the research replay (``all_conditions/
replay.py`` audit and prediction, ``policy.py`` decision loop) on top of the
ported feature pipeline. Each issue row t gets:

* ``prediction``: charged coke over rows t+2..t+5, i.e. the 4-hour block
  starting 1 h and ending 5 h after row t completes (rows are interval-start
  labels, complete at t+1h). Computed wherever the model has its inputs,
  whether or not it may be shown.
* ``state`` and its reasons, range and the last valid forecast.
* ``actual``: the realised block, once it exists, for scoring only.

Nothing in a row depends on source data after that row (see the truncation
test), and the reported coke rate, reported fuel rate and unit cost are dropped
before any of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.bmo.charged_coke.features import (
    DEFAULT_WARMUP,
    ETA,
    PCI,
    PROD,
    TEMP,
    WIND,
    CO,
    CO2,
    FeatureSet,
    build_features,
)
from utils.bmo.charged_coke.model import DEFAULT_BUNDLE_DIR, ChargedCokeModel
from utils.bmo.charged_coke.physics import load_physics_config
from utils.bmo.charged_coke.policy import DISPLAY_STATES, PolicyConfig, apply_policy, audit_hour, novelty

TARGET_ROWS = (2, 5)
MAX_HORIZON = 5
# Rows computed before the first decided row so its rolling inputs are whole:
# 24-h control reference and count, 12-h sequence and history count.
DECISION_CONTEXT = pd.Timedelta(hours=48)
_LEVELS = {"wind": WIND, "blast": TEMP, "prod": PROD, "pci": PCI, "eta": ETA, "raft": "RAFTOC",
           "dp": "DIFFERENTIAL PRESSURETOTALBAR"}
_OPERATING = [WIND, TEMP, PROD, PCI, ETA, "RAFTOC", "TOPPRESSUREBAR", "HOT BLAST PRESSUREBAR"]


@dataclass
class ChargedCokeEngine:
    """The frozen model, its policy and its physics, loaded once."""

    model: ChargedCokeModel
    policy: PolicyConfig
    physics_config: dict[str, Any]

    @classmethod
    def load(cls, bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> "ChargedCokeEngine":
        return cls(
            model=ChargedCokeModel(bundle_dir),
            policy=PolicyConfig.load(bundle_dir),
            physics_config=load_physics_config(bundle_dir),
        )


def gate_inputs(fs: FeatureSet) -> pd.DataFrame:
    """Current-hour inputs of the display policy (research ``replay.audit``).

    Levels and four-hour variability come from the screened source, the
    charged reference and controls from the model channels, and integrity
    flags from the feature gates. ``actual`` is the realised 4-h block five
    rows later, with the replay's accounting-validity rule; it is for scoring
    only and never reaches a decision.
    """

    d, ch, g, raw = fs.cleaned, fs.channels, fs.gates, fs.raw
    a = pd.DataFrame(index=d.index)
    for name, column in _LEVELS.items():
        a[name] = d[column]
    a["gas_sum"] = d[CO] + d[CO2]
    a["charge4"] = fs.labels.current4
    a["slag"] = ch.slag
    a["nut"] = ch.nut
    for name in ("wind", "prod", "pci"):
        a[name + "_cv4"] = a[name].rolling(4).std() / a[name].rolling(4).mean()
    for name in ("blast", "eta"):
        a[name + "_range4"] = a[name].rolling(4).max() - a[name].rolling(4).min()
    a["charge_cv4"] = d.COKE_CALC_MT.rolling(4).std() / d.COKE_CALC_MT.rolling(4).mean()
    a["history_count"] = g.valid.rolling(12).sum()
    a["mass_count"] = g.mass_ok.rolling(4).sum()
    a["reference_count"] = ch[["pci", "nut", "slag"]].rolling(24).count().min(axis=1)
    a["stable_original"] = g.stable
    a["source_present"] = raw[WIND].notna()
    a["sensor_frozen"] = g.frozen
    a["identity_bad"] = g.eta_bad | g.dp_bad
    a["shadow_possible"] = ch[["charge4", "pci", "nut", "slag"]].notna().all(axis=1)
    a["operating_readings_valid"] = d[_OPERATING].notna().all(axis=1)

    mass_valid = d.COKE_CALC_MT.between(0, 100) & d[PROD].between(20, 160) & raw[WIND].notna()
    both_frozen = g.frozen & raw[["COKE_CALC_MT", PROD]].diff().abs().lt(1e-9).all(axis=1)
    mass_valid &= ~both_frozen
    coke, prod = d.COKE_CALC_MT.where(mass_valid), d[PROD].where(mass_valid)
    broad = 1000 * coke.rolling(4).sum() / prod.rolling(4).sum()
    a["observed_charge4"] = broad
    a["actual"] = broad.shift(-TARGET_ROWS[1])
    # The realised 4-h rate at each horizon a model may forecast.
    for h in range(1, MAX_HORIZON + 1):
        a[f"actual_h{h}"] = broad.shift(-h)
    return a


def sequences(channels: pd.DataFrame, features: list[str], hours: int) -> np.ndarray:
    """``(n, hours, features)`` inputs, oldest first, built as in training.

    The research build stacked shifted channels as float32; the cast is kept
    because it moves predictions by up to ~1e-5 kg/THM.
    """

    stack = np.stack(
        [channels.shift(h).to_numpy(dtype=np.float32) for h in range(hours - 1, -1, -1)], axis=1
    )
    ids = [channels.columns.get_loc(name) for name in features]
    return stack[:, :, ids].astype(float)


def forecast_hours(
    source: pd.DataFrame,
    engine: ChargedCokeEngine,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup: pd.Timedelta = DEFAULT_WARMUP,
    context: pd.Timedelta = DECISION_CONTEXT,
    planned_change: pd.Series | None = None,
    initial_streak: int = 0,
    initial_last: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Forecast and display state for every issue row in ``start..end``.

    Args:
        source: Hourly furnace dataset, interval-start labels, plant clock.
        engine: Loaded model, policy and physics.
        start: First issue row decided.
        end: Last issue row (the latest complete hour, live).
        warmup: Feature history read before the decision context: covers the
            24-h assay delay with its 12-h forward fill and the rolling
            windows of the features themselves.
        context: Rows computed before ``start`` so its rolling gate inputs,
            24-h control reference and 12-h sequence are whole.
        planned_change: Optional boolean per issue row.
        initial_streak: Eligible updates already counted before ``start``.
        initial_last: Last shown forecast before ``start``
            (``{"issue": Timestamp, "value": float}``), so its window can
            still be shown.

    Returns:
        One row per issue hour: gate inputs, model outputs, novelty, state,
        range, reasons and the last valid forecast.
    """

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # Rolling gate inputs, the 12-h sequence and the 24-h control reference
    # need rows before ``start`` on the same clock, beyond the feature warm-up.
    fs = build_features(source, engine.physics_config, start=start - context, end=end, warmup=warmup)
    hours = gate_inputs(fs)

    model = engine.model
    seq = sequences(fs.channels, model.features, model.sequence_hours)
    deviation = model.control_deviation(fs.channels).to_numpy()
    keep = np.asarray(hours.index >= start)
    hours, seq, deviation = hours.loc[keep].copy(), seq[keep], deviation[keep]
    anchor = fs.labels.current4.to_numpy()[keep]
    computable = hours["shadow_possible"].to_numpy()
    out = model.predict(seq[computable], anchor[computable], deviation[computable])
    columns: dict[str, np.ndarray] = {}
    for name in ("prediction", "spread", "linear", "attention", "control_adjustment"):
        values = getattr(out, name)
        for i, h in enumerate(model.horizons):
            column = np.full(len(hours), np.nan)
            column[computable] = values[:, i]
            columns[f"{name}_h{h}"] = column
    hours = hours.assign(**columns)
    # The longest horizon is the headline: it feeds the BMO level. The
    # disagreement gate reads the widest branch spread over all horizons.
    headline = model.horizons[-1]
    for name in ("prediction", "linear", "attention", "control_adjustment"):
        hours[name] = hours[f"{name}_h{headline}"]
    hours["spread"] = hours[[f"spread_h{h}" for h in model.horizons]].max(axis=1, skipna=False)
    hours[["pci_deviation", "nut_deviation", "slag_deviation"]] = deviation

    audits = [audit_hour(row, engine.policy) for _, row in hours.iterrows()]
    for key in ("severity", "data_ready", "outside_core_count", "reasons", "codes"):
        hours[key] = [a[key] for a in audits]
    hours["novelty"] = novelty(hours, engine.policy)
    p = engine.policy.parameters
    hours["unusual"] = hours["novelty"].gt(p["novelty_warn"]) | hours["spread"].gt(p["spread_warn"])
    decided = apply_policy(
        hours, engine.policy, planned_change=planned_change,
        initial_streak=initial_streak, initial_last=initial_last,
    )
    decided["target_start_row"] = decided.index + pd.Timedelta(hours=TARGET_ROWS[0])
    decided["target_end_row"] = decided.index + pd.Timedelta(hours=TARGET_ROWS[1])
    shown = decided["state"].isin(DISPLAY_STATES)
    for h in model.horizons:
        radius = [
            horizon_radius(p, severity, unusual, h) if is_shown else np.nan
            for severity, unusual, is_shown in zip(decided["severity"], decided["unusual"], shown)
        ]
        decided[f"radius90_h{h}"] = radius
        decided[f"lower_h{h}"] = decided[f"prediction_h{h}"].where(shown) - decided[f"radius90_h{h}"]
        decided[f"upper_h{h}"] = decided[f"prediction_h{h}"].where(shown) + decided[f"radius90_h{h}"]
    return decided


def horizon_radius(parameters: dict[str, Any], severity: str, unusual: bool, horizon: int) -> float:
    """Calibrated 90% error half-width for one horizon in one condition.

    A model trained here stores one radius per horizon per condition; the
    research bundle has one radius per condition for its single horizon.
    """

    bins = parameters["bins"].get(severity)
    if bins is None:
        return np.nan
    radius = float(bins.get("radius_by_horizon", {}).get(str(horizon), bins["radius"]))
    if unusual:
        unusual_radius = parameters.get("unusual_radius_by_horizon", {}).get(str(horizon), parameters["unusual_radius"])
        radius = max(radius, float(unusual_radius))
    return radius


MISSING_HOUR_NOTE = (
    "The publisher drops hours outside its cruising filter (blast < 90,000 Nm³/h, PCI < 70 kg/THM, "
    "ETA CO outside 38-47 %, production < 60 t/h or reported fuel rate outside 100-670 kg/THM), "
    "so a missing hour can mean the furnace was off cruise, not only that data failed."
)


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _seed_from_ledger(ledger, row: pd.Timestamp) -> tuple[pd.Timestamp, int, dict[str, Any] | None, str]:
    """Where the hourly decision chain resumes, and its state there.

    The chain continues from the last issued row so recovery counts one step
    per completed hourly update. A gap longer than the decision context, or no
    record at all, starts a fresh chain over the context.
    """

    earlier = {t: r for t, r in ledger.issues().items() if t < row}
    fresh_start = row - DECISION_CONTEXT + pd.Timedelta(hours=1)
    if not earlier:
        return fresh_start, 0, None, "no earlier issue: fresh 48-h chain"
    last_row = max(earlier)
    record = earlier[last_row]
    if row - last_row > DECISION_CONTEXT:
        return fresh_start, 0, None, f"last issue {last_row} is over 48 h old: fresh chain"
    last_valid = None
    if record.get("shown"):
        last_valid = {"issue": last_row, "value": float(record["prediction_kg_thm"])}
    elif record.get("last_valid"):
        last_valid = {"issue": pd.Timestamp(record["last_valid"]["data_row"]),
                      "value": float(record["last_valid"]["prediction_kg_thm"])}
    return last_row + pd.Timedelta(hours=1), int(record["eligible_streak"]), last_valid, f"continues from issue {last_row}"


OUTCOME_LOOKBACK = pd.Timedelta(days=7)


def _record_matured_outcomes(source, engine, ledger, *, row: pd.Timestamp, recorded_at: pd.Timestamp) -> None:
    """Append the realised block of every issued row whose block has arrived.

    A block (rows t+2..t+5) is complete once row t+5 is, so issued rows up to
    ``row - 5 h`` can be scored. Each outcome is appended once, with the same
    accounting-validity rule as the research replay; the issued record itself
    is never touched. Rows older than a week are left unscored.
    """

    matured_until = row - pd.Timedelta(hours=TARGET_ROWS[1])
    known = ledger.outcomes()
    pending = [t for t in ledger.issues() if t not in known and row - OUTCOME_LOOKBACK <= t <= matured_until]
    if not pending:
        return
    fs = build_features(source, engine.physics_config, start=min(pending), end=row)
    actual = gate_inputs(fs)["actual"]
    for issued_row in pending:
        value = _number(actual.get(issued_row))
        if value is not None:
            ledger.record_outcome(issued_row, value, recorded_at=recorded_at)


def issue_latest(
    source: pd.DataFrame,
    engine: ChargedCokeEngine,
    *,
    ledger,
    plans,
    now: pd.Timestamp,
    built_at: pd.Timestamp | None = None,
    model_version: str = "",
) -> dict[str, Any]:
    """Issue (once) the forecast for the latest complete data row.

    Args:
        source: The published hourly dataset.
        engine: Loaded model, policy and physics.
        ledger: :class:`~utils.bmo.charged_coke.ledger.ForecastLedger`.
        plans: :class:`~utils.bmo.charged_coke.ledger.PlannedChangeRegister`.
        now: Wall-clock issue time (IST).
        built_at: When the publisher built the file. Without it the newest row
            is taken to be in progress and the one before it is used.
        model_version: Bundle version recorded with the forecast.

    Returns:
        The issued record for that row: newly written, or the existing one.
    """

    from utils.bmo.charged_coke.contract import ForecastWindow, latest_complete_row

    last_row = pd.Timestamp(source.index.max())
    row = (
        latest_complete_row(built_at, last_row)
        if built_at is not None
        else last_row - pd.Timedelta(hours=1)
    )
    if row is None:
        return {"data_row": None, "state": "data_missing", "shown": False,
                "reasons": ["No complete hour in the published dataset yet"]}
    existing = ledger.get(row)
    if existing is not None:
        return existing

    chain_start, streak, last_valid, chain_note = _seed_from_ledger(ledger, row)
    chain = pd.date_range(chain_start, row, freq="h")
    # A declaration counts if it was active when that hour's forecast was due.
    due = {t: (pd.Timestamp(now) if t == row else t + pd.Timedelta(hours=1, minutes=6)) for t in chain}
    planned = pd.Series({t: bool(plans.active(moment)) for t, moment in due.items()}, dtype=bool)
    decided = forecast_hours(source, engine, start=chain_start, end=row, planned_change=planned,
                             initial_streak=streak, initial_last=last_valid)
    r = decided.loc[row]
    window = ForecastWindow.for_row(row, now)
    shown = bool(r["state"] in DISPLAY_STATES)
    reasons = list(r["reasons"])
    if r["state"] == "data_missing":
        reasons.append(MISSING_HOUR_NOTE)
    record: dict[str, Any] = {
        "data_row": row,
        "issued_at": window.issued_at,
        "dataset_built_at": built_at,
        "data_complete_at": window.data_complete_at,
        "coke_window": [window.coke_start, window.coke_end],
        "production_window": [window.production_start, window.production_end],
        "model_version": model_version,
        "state": r["state"],
        "shown": shown,
        "prediction_kg_thm": _number(r["prediction"]) if shown else None,
        "lower_kg_thm": _number(r["lower"]),
        "upper_kg_thm": _number(r["upper"]),
        "radius90_kg_thm": _number(r["radius90"]),
        # Computed but withheld values are kept for review, never displayed.
        "withheld_value_kg_thm": None if shown else _number(r["prediction"]),
        "reasons": reasons,
        "codes": list(r["codes"]),
        "severity": r["severity"],
        "eligible_streak": int(r["eligible_streak"]),
        "anchor_charge4_kg_thm": _number(r["charge4"]),
        "controls": {k: _number(r[k]) for k in ("pci", "nut", "slag")},
        "control_deviation": {k: _number(r[f"{k}_deviation"]) for k in ("pci", "nut", "slag")},
        "control_adjustment_kg_thm": _number(r["control_adjustment"]),
        "branches": {k: _number(r[k]) for k in ("linear", "attention", "spread")},
        "novelty": _number(r["novelty"]),
        "unusual": bool(r["unusual"]),
        "planned_change_ids": [c.id for c in plans.active(now)],
        "chain": chain_note,
    }
    # The hourly path: one value per horizon, each with its own clock window.
    from utils.bmo.charged_coke.contract import horizon_window

    path = []
    for h in engine.model.horizons:
        span = horizon_window(row, h)
        value = _number(r[f"prediction_h{h}"])
        path.append({
            "horizon": int(h), "at": span["coke_end"], "coke_window": [span["coke_start"], span["coke_end"]],
            "production_window": [span["production_start"], span["production_end"]],
            "prediction_kg_thm": value if shown else None,
            "withheld_value_kg_thm": None if shown else value,
            "lower_kg_thm": _number(r[f"lower_h{h}"]), "upper_kg_thm": _number(r[f"upper_h{h}"]),
        })
    record["path"] = path
    _record_matured_outcomes(source, engine, ledger, row=row, recorded_at=window.issued_at)
    if not shown and pd.notna(r["last_forecast"]):
        previous = pd.Timestamp(r["last_issue"])
        prev_window = ForecastWindow.for_row(previous, previous + pd.Timedelta(hours=1))
        record["last_valid"] = {
            "data_row": previous,
            "prediction_kg_thm": _number(r["last_forecast"]),
            "coke_window": [prev_window.coke_start, prev_window.coke_end],
            "expires_after_row": pd.Timestamp(r["last_window_end"]),
        }
    return ledger.record_issue(record)


def recent_forecasts(
    source: pd.DataFrame,
    engine: ChargedCokeEngine,
    *,
    end: pd.Timestamp,
    days: int = 7,
) -> pd.DataFrame:
    """Forecast vs actual over recent days, one row per issue hour and horizon.

    The active model is replayed causally over the window, so every hour shows
    what it would have issued then. Only hours after the model's training
    cutoff are kept: earlier hours were part of its fit and would flatter it.

    Returns:
        Columns ``issue_row``, ``horizon``, ``at`` (clock end of the 4-h charge
        window), ``prediction``, ``lower``, ``upper``, ``actual``, ``state``,
        ``severity`` and ``shown``.
    """

    from utils.bmo.charged_coke.contract import horizon_window

    end = pd.Timestamp(end)
    start = end - pd.Timedelta(days=days) + pd.Timedelta(hours=1)
    trained_until = engine.model.manifest.get("trained_until") or engine.model.manifest.get("frozen_at")
    if trained_until:
        start = max(start, pd.Timestamp(trained_until))
    if start > end:
        return pd.DataFrame()
    decided = forecast_hours(source, engine, start=start, end=end)
    frames = []
    for h in engine.model.horizons:
        frame = pd.DataFrame({
            "issue_row": decided.index, "horizon": h,
            "at": [horizon_window(t, h)["coke_end"] for t in decided.index],
            "prediction": decided[f"prediction_h{h}"].to_numpy(),
            "lower": decided[f"lower_h{h}"].to_numpy(), "upper": decided[f"upper_h{h}"].to_numpy(),
            "actual": decided[f"actual_h{h}"].to_numpy(), "state": decided["state"].to_numpy(),
            "severity": decided["severity"].to_numpy(),
            # The rate operators had at issue: what "holding the current rate" means.
            "anchor": decided["charge4"].to_numpy(),
        })
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["shown"] = out["state"].isin(DISPLAY_STATES)
    return out


__all__ = [
    "recent_forecasts",
    "ChargedCokeEngine",
    "horizon_radius",
    "MISSING_HOUR_NOTE",
    "TARGET_ROWS",
    "forecast_hours",
    "gate_inputs",
    "issue_latest",
    "sequences",
]
