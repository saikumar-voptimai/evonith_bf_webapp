"""Condition-aware display policy for the charged-coke forecast.

Every hourly update gets a status. A forecast is shown, with an empirical error
range that widens with how far the furnace is from cruising, or it is paused
with the reason:

    cruise       forecast, about +/-16 kg/THM
    unsettled    forecast, wider range (recent instability, charge variability)
    off_cruise   forecast, wider still (a level outside the cruise envelope)
    recovering   paused: first eligible hour after a hold; resumes on the second
    *_hold / data_missing   paused: large deviation, unsupported combination,
                 branch disagreement, too-wide range, planned change or inputs

This is a port of the research package's ``replay.audit`` gates,
``policy.py`` novelty check and ``decision.decide``. The thresholds are the
package's pre-freeze values in ``policy.json``; they are proposed display
limits, not furnace protection limits. Nothing here looks at a future outcome.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.bmo.charged_coke.model import DEFAULT_BUNDLE_DIR

DISPLAY_STATES = frozenset({"cruise", "unsettled", "off_cruise"})

_NAMES = {
    "wind": "Blast volume", "blast": "Blast temperature", "prod": "Production",
    "pci": "PCI", "eta": "ETA CO", "raft": "RAFT", "gas_sum": "CO + CO2",
}
_UNITS = {
    "wind": "Nm³/h", "blast": "°C", "prod": "t/h", "pci": "kg/thm", "eta": "%",
    "raft": "°C", "gas_sum": "%",
}


@dataclass(frozen=True)
class PolicyConfig:
    """Thresholds and calibration from ``policy.json`` and ``support.npz``."""

    parameters: dict[str, Any]
    core: dict[str, tuple[float, float]]
    extended: dict[str, tuple[float, float]]
    variability: dict[str, tuple[float, float]]
    minimum_history: int
    minimum_reference: int
    large_charge_cv: float
    mild_charge_cv: float
    support_center: np.ndarray
    support_scale: np.ndarray
    support_rows: np.ndarray = field(repr=False)

    @classmethod
    def load(cls, bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> "PolicyConfig":
        bundle_dir = Path(bundle_dir)
        doc = json.loads((bundle_dir / "policy.json").read_text(encoding="utf-8"))
        support = np.load(bundle_dir / "support.npz")
        limits = doc["thresholds"]

        def pairs(block: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
            return {key: (float(lo), float(hi)) for key, (lo, hi) in block.items()}

        return cls(
            parameters=doc["parameters"],
            core=pairs(limits["core"]),
            extended=pairs(limits["extended"]),
            variability=pairs(limits["variability"]),
            minimum_history=int(limits["minimum_history"]),
            minimum_reference=int(limits["minimum_reference"]),
            large_charge_cv=float(limits["large_charge_cv"]),
            mild_charge_cv=float(limits["mild_charge_cv"]),
            support_center=support["center"],
            support_scale=support["scale"],
            support_rows=support["rows"],
        )

    @property
    def novelty_features(self) -> list[str]:
        return list(self.parameters["features"])


def _between(value: Any, lo: float, hi: float) -> bool:
    """``pd.Series.between``: a missing value is never inside."""

    return bool(pd.notna(value) and lo <= value <= hi)


def audit_hour(row: pd.Series | dict[str, Any], config: PolicyConfig) -> dict[str, Any]:
    """Severity, readiness and plain-language reasons for one issue hour.

    Args:
        row: Current-hour gate inputs: the levels (``wind``, ``blast``,
            ``prod``, ``pci``, ``eta``, ``raft``, ``gas_sum``), four-hour
            variability (``wind_cv4`` ... ``eta_range4``, ``charge_cv4``),
            ``charge4``, ``slag``, history and reference counts, and the
            integrity flags ``source_present``, ``sensor_frozen``,
            ``identity_bad``, ``operating_readings_valid``,
            ``shadow_possible`` and ``stable_original``.
        config: Loaded policy configuration.

    Returns:
        ``severity``, ``data_ready``, ``outside_core_count``, ``reasons`` and
        ``codes``.
    """

    r = row
    outside_core = sum(not _between(r[k], lo, hi) for k, (lo, hi) in config.core.items())
    major_level = any(
        pd.notna(r[k]) and not lo <= r[k] <= hi for k, (lo, hi) in config.extended.items()
    )
    major_dynamic = any(
        pd.notna(r[k]) and r[k] > hard for k, (_soft, hard) in config.variability.items()
    ) or (pd.notna(r["charge_cv4"]) and r["charge_cv4"] > config.large_charge_cv)
    unstable = not bool(r["stable_original"]) or (
        pd.notna(r["charge_cv4"]) and r["charge_cv4"] > config.mild_charge_cv
    )
    if major_level or major_dynamic:
        severity = "large"
    elif outside_core > 0:
        severity = "moderate"
    elif unstable:
        severity = "mild"
    else:
        severity = "cruise"

    history = r["history_count"]
    reference = r["reference_count"]
    data_ready = bool(
        r["source_present"]
        and not r["sensor_frozen"]
        and not r["identity_bad"]
        and r["operating_readings_valid"]
        and r["shadow_possible"]
        and pd.notna(history) and history >= config.minimum_history
        and pd.notna(reference) and reference >= config.minimum_reference
    )

    why: list[str] = []
    codes: list[str] = []
    present = bool(r["source_present"])
    if not present:
        why.append("Expected hourly input is missing")
        codes.append("missing_hour")
    elif r["sensor_frozen"]:
        why.append("Eight independent process readings are unchanged: suspected frozen feed")
        codes.append("frozen_feed")
    if r["identity_bad"]:
        why.append("Gas utilisation or pressure readings fail their consistency check")
        codes.append("inconsistent_tags")
    if present and not r["operating_readings_valid"]:
        why.append("A critical operating reading is missing or physically implausible")
        codes.append("invalid_reading")
    if present and pd.isna(r["charge4"]):
        mass = r["mass_count"]
        available = int(mass) if pd.notna(mass) and np.isfinite(mass) else 0
        why.append(f"Charged-rate reference needs four valid hours ({available}/4 available)")
        codes.append("charge_reference")
    if present and pd.isna(r["slag"]):
        why.append("Expected slag unavailable: burden or delayed material assays incomplete")
        codes.append("slag_unavailable")
    if present and pd.notna(history) and history < config.minimum_history:
        why.append(f"Valid process history {int(history)}/12 hours; at least 9 needed")
        codes.append("short_history")
    if present and pd.notna(reference) and reference < config.minimum_reference:
        why.append(f"PCI/nut/slag reference has {int(reference)}/24 hours; at least 12 needed")
        codes.append("short_control_reference")
    if present:
        for k, (lo, hi) in config.core.items():
            el, eh = config.extended[k]
            value = r[k]
            if pd.notna(value) and (not lo <= value <= hi or not el <= value <= eh):
                major = not el <= value <= eh
                threshold = f"{el:g}–{eh:g}" if major else f"{lo:g}–{hi:g}"
                why.append(
                    f"{_NAMES[k]} {value:,.1f} {_UNITS[k]}; "
                    + ("outside display range " if major else "outside cruise range ")
                    + threshold
                )
                codes.append(("large_" if major else "offcore_") + k)
        for k, (soft, hard) in config.variability.items():
            value = r[k]
            if pd.notna(value) and value > soft:
                label = k.split("_")[0].capitalize()
                if "cv" in k:
                    why.append(
                        f"{label} varies {value * 100:.1f}% over four hours (cruise ≤{soft * 100:g}%)"
                    )
                else:
                    why.append(f"{label} four-hour range {value:.1f} (cruise ≤{soft:g})")
                codes.append(("large_" if value > hard else "varying_") + k)
        if pd.notna(r["charge_cv4"]) and r["charge_cv4"] > config.mild_charge_cv:
            why.append(
                f"Hourly charged-coke mass varies {r['charge_cv4'] * 100:.1f}% over four hours"
            )
            codes.append("charge_volatility")
        if not bool(r["stable_original"]) and not any(
            c.startswith(("offcore", "large", "varying")) for c in codes
        ):
            why.append("Four consecutive cruise hours not yet established")
            codes.append("recent_transition")
    return {
        "severity": severity,
        "data_ready": data_ready,
        "outside_core_count": outside_core,
        "reasons": why,
        "codes": codes,
    }


def novelty(features: pd.DataFrame, config: PolicyConfig) -> np.ndarray:
    """Distance to the nearest supported historical hour (no physical unit).

    Robust-scaled current features against the pre-freeze support set; above
    ``novelty_warn`` the range widens, above ``novelty_stop`` the forecast is
    paused. An hour with a missing feature has no distance (NaN), which the
    decision treats as unsupported.
    """

    x = features[config.novelty_features].astype(float).to_numpy()
    out = np.full(len(x), np.nan)
    complete = np.isfinite(x).all(axis=1)
    if complete.any():
        scaled = (x[complete] - config.support_center) / config.support_scale
        rows = config.support_rows
        # Chunked so a long history never builds one huge distance matrix.
        distances = []
        for start in range(0, len(scaled), 256):
            block = scaled[start : start + 256]
            d2 = (
                (block**2).sum(axis=1)[:, None]
                - 2.0 * block @ rows.T
                + (rows**2).sum(axis=1)[None, :]
            )
            distances.append(np.sqrt(np.maximum(d2.min(axis=1), 0.0)))
        out[complete] = np.concatenate(distances)
    return out


def decide(
    row: pd.Series | dict[str, Any],
    parameters: dict[str, Any],
    eligible_streak: int = 0,
    *,
    planned_change: bool = False,
) -> dict[str, Any]:
    """The research package's causal display decision for one issue hour.

    Args:
        row: The hour's audit (``severity``, ``data_ready``, ``reasons``,
            ``codes``), ``source_present``, model ``prediction`` and
            ``spread``, ``novelty`` and ``unusual``.
        parameters: ``policy.json`` parameters.
        eligible_streak: Consecutive eligible updates before this one.
        planned_change: A major operating change is planned within the
            forecast window.

    Returns:
        ``state``, the updated ``eligible_streak``, ``radius90`` (NaN unless
        shown), ``reasons``, ``reason_codes`` and ``show_prediction``.
    """

    r, p = row, parameters
    why = list(r["reasons"])
    codes = list(r["codes"])
    radius = np.nan
    if not r["source_present"]:
        status, eligible_streak = "data_missing", 0
    elif not r["data_ready"] or pd.isna(r.get("prediction")):
        status, eligible_streak = "data_hold", 0
    elif planned_change:
        status, eligible_streak = "planned_change_hold", 0
        why.insert(0, "A major operating change is planned within the forecast window; current-condition forecast is not applicable")
        codes.insert(0, "planned_change")
    elif r["severity"] == "large":
        status, eligible_streak = "large_hold", 0
    elif pd.isna(r["novelty"]) or r["novelty"] > p["novelty_stop"]:
        status, eligible_streak = "novel_hold", 0
        why.insert(0, "Current combination of operating conditions is outside validated historical support")
        codes.insert(0, "novel_combination")
    elif r["spread"] > p["spread_stop"]:
        status, eligible_streak = "uncertainty_hold", 0
        why.insert(0, f"Model branches disagree strongly ({r['spread']:.1f} kg/thm spread; limit {p['spread_stop']:.1f})")
        codes.insert(0, "expert_disagreement")
    else:
        evidence = p["bins"][r["severity"]]
        radius = evidence["radius"]
        if r["unusual"]:
            radius = max(radius, p["unusual_radius"])
            why.insert(0, "Few close historical matches or elevated model disagreement; wider error range")
            codes.insert(0, "unusual_supported")
        if (
            evidence["n"] < p["minimum_calibration_hours"]
            or evidence["days"] < p["minimum_calibration_days"]
        ):
            status, eligible_streak = "uncertainty_hold", 0
            why.insert(0, "Too few independently observed comparable days to calibrate an error range")
            codes.insert(0, "limited_evidence")
        elif radius > p["display_width_cap"]:
            status, eligible_streak = "uncertainty_hold", 0
            why.insert(0, f"Calibrated error range ±{radius:.1f} kg/thm exceeds the proposed ±{p['display_width_cap']:g} display limit")
            codes.insert(0, "range_too_wide")
        else:
            eligible_streak += 1
            if eligible_streak < p["recovery_updates"]:
                status = "recovering"
                why.insert(0, "Inputs have recovered; waiting for a second consecutive eligible update")
                codes.insert(0, "recovery_debounce")
            elif r["severity"] == "cruise" and not r["unusual"]:
                status = "cruise"
            elif r["severity"] == "moderate":
                status = "off_cruise"
            else:
                status = "unsettled"
    shown = status in DISPLAY_STATES
    return {
        "state": status,
        "eligible_streak": eligible_streak,
        "radius90": float(radius) if shown else np.nan,
        "reasons": why,
        "reason_codes": codes,
        "show_prediction": shown,
    }


def apply_policy(
    hours: pd.DataFrame,
    config: PolicyConfig,
    *,
    planned_change: pd.Series | None = None,
    forecast_hours: int = 5,
    initial_streak: int = 0,
    initial_last: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run the decision over consecutive issue hours, oldest first.

    The eligible streak carries from hour to hour and resets on every hold. A
    previous forecast stays visible only with its original issue hour and
    target window, and disappears once that window has ended.

    Args:
        hours: One row per hourly update on a complete clock, indexed by the
            issue row, with the audit outputs (``severity``, ``data_ready``,
            ``reasons``, ``codes``), ``source_present``, ``prediction``,
            ``spread``, ``novelty`` and ``unusual``.
        config: Loaded policy configuration.
        planned_change: Optional boolean per hour.
        forecast_hours: Rows from issue to the end of the target block.
        initial_streak: Eligible updates already counted before the first row
            (from the issued-forecast record, live).
        initial_last: Last shown forecast before the first row,
            ``{"issue": Timestamp, "value": float}``.

    Returns:
        ``hours`` plus ``state``, ``eligible_streak``, ``radius90``, ``lower``/``upper``,
        ``display_forecast``, ``reasons``, ``codes`` and the last valid
        forecast (``last_forecast``, ``last_issue``, ``last_window_end``).
    """

    states, radii, reasons, codes, streaks = [], [], [], [], []
    last_values, last_issues, last_ends = [], [], []
    streak, last = int(initial_streak), (dict(initial_last) if initial_last else None)
    for t, row in hours.iterrows():
        planned = bool(planned_change.get(t, False)) if planned_change is not None else False
        outcome = decide(row, config.parameters, streak, planned_change=planned)
        streak = outcome["eligible_streak"]
        streaks.append(streak)
        if outcome["show_prediction"]:
            last = {"issue": t, "value": float(row["prediction"])}
        live = last is not None and (t - last["issue"]) <= pd.Timedelta(hours=forecast_hours)
        last_values.append(last["value"] if live else np.nan)
        last_issues.append(last["issue"] if last else pd.NaT)
        last_ends.append(last["issue"] + pd.Timedelta(hours=forecast_hours) if last else pd.NaT)
        states.append(outcome["state"])
        radii.append(outcome["radius90"])
        reasons.append(outcome["reasons"])
        codes.append(outcome["reason_codes"])
    out = hours.copy()
    out["state"] = states
    out["eligible_streak"] = streaks
    out["radius90"] = radii
    out["display_forecast"] = out["prediction"].where(out["state"].isin(DISPLAY_STATES))
    out["lower"] = out["display_forecast"] - out["radius90"]
    out["upper"] = out["display_forecast"] + out["radius90"]
    out["reasons"] = reasons
    out["codes"] = codes
    out["last_forecast"] = last_values
    out["last_issue"] = last_issues
    out["last_window_end"] = last_ends
    return out


__all__ = [
    "DISPLAY_STATES",
    "PolicyConfig",
    "apply_policy",
    "audit_hour",
    "decide",
    "novelty",
]
