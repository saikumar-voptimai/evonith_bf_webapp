"""Robust coke-rate target and matching window features for the direct coke model.

Why the hourly mass ratio is not a usable target
------------------------------------------------
``1,000 x COKE_CALC_MT / PRODUCTIONTONNESPERHR`` divides a charge-counted mass
by a production figure that runs on a different clock. Hourly production
correlates only 0.40 with the Fe charged in the same hour, yet the two agree to
0.99 +/- 0.03 over a day. Coke and ore, on the other hand, share one charge
count: hour-to-hour changes in the coke rate correlate 0.98 with those in the
same-hour burden rate (levels 0.76 over a year of trusted hours, 0.93 over the
fortnight the earlier model was validated on). A model given that burden rate
scores a high R2 by predicting the charge count,
and has nothing left to learn from PCI or nut coke: the hourly-target model
moved +7 kg/THM for a 40 kg/THM PCI cut, and the wrong way for nut coke.

The reported ``COKE RATE KG/THM`` is the other extreme: an operator set point
that changes in 15% of hours.

The target used instead keeps numerator and denominator on the same charges::

    coke_rate(t) = 1,000 x [sum_W coke / sum_W Fe charged]
                         x [sum_week Fe charged / sum_week production]

The first factor is the coke-to-iron ratio of the charges actually made in the
trailing window, so charge counting cancels (hour-to-hour change falls from
~31 to ~3 kg/THM before any windowing). The second converts Fe charged to hot
metal at the plant's own production level over a week, so the rate stays on the
reported-production basis that fuel prices and the set point use.

Features are aggregated over the same trailing window, so the model maps the
window's conditions to the window's coke rate. PCI and nut coke enter as window
mass ratios, and the trainer constrains both to be monotone decreasing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import re
from typing import Any

import numpy as np
import pandas as pd

TASK_NAME = "coke_robust_window"
TARGET_COLUMN = "robust_coke_kg_per_thm"

# Pellet Fe is reported as Fe2O3; 2 x 55.845 / 159.69.
PELLET_FE_PER_FE2O3 = 111.69 / 159.69

PCI_FEATURE = "PCI_KG_THM__win"
NUT_COKE_FEATURE = "NUTCOKE_KG_THM__win"
SLAG_FEATURE = "slag_kg_thm_repo__win"
# Physical direction of the two fuel substitutes: more of either needs less coke.
MONOTONE_CONSTRAINTS: dict[str, int] = {PCI_FEATURE: -1, NUT_COKE_FEATURE: -1}
DRIVER_FEATURES: tuple[str, ...] = (PCI_FEATURE, NUT_COKE_FEATURE, SLAG_FEATURE)

_FUEL_RATE_FEATURES = {
    PCI_FEATURE: "PCI_CALC_MT",
    NUT_COKE_FEATURE: "NUTCOKE_CALC_MT",
    "FLUX_KG_THM__win": "FLUX_CALC_MT",
}
_PROCESS_COLUMNS = (
    "HOT BLAST VOLUMENM3/HR.",
    "HOT BLAST TEMP.OC",
    "HOT BLAST PRESSUREBAR",
    "OXYGENFLOWNM3/HR.",
    "O2 ENRICHMENT %",
    "STEAMKGS/HR.",
    "TOPPRESSUREBAR",
    "DIFFERENTIAL PRESSURETOTALBAR",
    "PRODUCTIONTONNESPERHR",
    "CHARGES/HRS.",
    "STOCKRODLEVEL",
    "COKE_DISCHARGE_TIME",
    "WEIGHTED_COKE_ANGLE",
    "TOTAL_COKE_PORTIONS",
    "NON_COKE_DISCHARGE_TIME",
    "WEIGHTED_NON_COKE_ANGLE",
    "TOTAL_NON_COKE_PORTIONS",
    "FURNACE TOP GAS ANALYSISCO2%",
    "FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%",
    "FURNACE TOP GAS ANALYSISH2%",
    "FURNACETOPGASANALYSISCO2ETACO",
)
_SLAG_COLUMNS = (
    "slag_kg_thm_repo",
    "slag_basicity_calc",
    "slag_al2o3_pct_calc",
    "slag_mgo_pct_calc",
)
# Raw-material quality that plausibly moves coke demand. The full assay list
# was tested and rejected: slow, rarely-updated analyses (PCI ash sulphur, coke
# ash alumina) became calendar markers that let the model memorise each
# period's level, which showed up as a later-time bias.
DEFAULT_ASSAY_COLUMNS: tuple[str, ...] = (
    "COKE_ASH%",
    "COKE_MOIST%",
    "COKE_VM%",
    "NUTCOKE_ASH%",
    "PCI_ASH%",
    "PCI_VM%",
    "PCI_IM%",
    "ORE_FE(T)%",
    "SINTER_FE(T)%",
    "SINTER_FEO%",
    "ORE_AL2O3%",
    "SINTER_AL2O3%",
)


@dataclass(frozen=True)
class RobustCokeSettings:
    """Target window, data-quality limits, and assay handling.

    Attributes:
        window_hours: Trailing window for both the target and the features.
        fe_level_hours: Centred window that levels Fe charged to production.
        min_window_coverage: Share of window hours that must be usable.
        outlier_mad_multiple: Hourly coke-to-Fe outlier limit, in robust sigmas.
        setpoint_column: Reported coke rate used only as a quality reference.
        setpoint_ratio_min: Lowest 24-h coke-mass / set-point agreement kept.
        setpoint_ratio_max: Highest 24-h coke-mass / set-point agreement kept.
        assay_delay_hours: Publication delay applied to assay features.
        assay_columns: Raw-material analyses offered to the model.
    """

    window_hours: int = 24
    fe_level_hours: int = 168
    min_window_coverage: float = 0.75
    outlier_mad_multiple: float = 6.0
    setpoint_column: str = "COKE RATE KG/THM"
    setpoint_ratio_min: float = 0.80
    setpoint_ratio_max: float = 1.25
    assay_delay_hours: int = 24
    assay_columns: tuple[str, ...] = DEFAULT_ASSAY_COLUMNS

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> RobustCokeSettings:
        raw = dict((config or {}).get("robust_target", {}) or {})
        known = {item.name for item in fields(cls)}
        values = {key: value for key, value in raw.items() if key in known}
        if "assay_columns" in values:
            values["assay_columns"] = tuple(str(item) for item in values["assay_columns"])
        return cls(**values)


def is_robust_config(config: Mapping[str, Any] | None) -> bool:
    """Whether a bundle config asks for the robust target and window features."""

    return "robust_target" in (config or {})


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _fe_part(
    mass: pd.Series, fe_pct: pd.Series, moisture_pct: pd.Series | None = None
) -> pd.Series:
    dry = mass if moisture_pct is None else mass * (1.0 - moisture_pct.fillna(0.0) / 100.0)
    # A material that was not charged contributes no Fe even when its assay is
    # missing; an uncharged slot must not blank the whole hour.
    return (dry * fe_pct / 100.0).where(~mass.eq(0.0), 0.0)


def fe_charged_mt(frame: pd.DataFrame) -> pd.Series:
    """Fe in the iron-bearing burden charged each hour (t).

    Ore and pellet masses are wet, so moisture comes off before the dry-basis Fe
    assay applies; sinter is charged hot and dry.
    """

    ore = _fe_part(
        _numeric(frame, "ORE_CALC_MT"),
        _numeric(frame, "ORE_FE(T)%"),
        _numeric(frame, "ORE_TM%"),
    )
    sinter = _fe_part(_numeric(frame, "SINTER_CALC_MT"), _numeric(frame, "SINTER_FE(T)%"))
    pellet = _fe_part(
        _numeric(frame, "TOTAL_PELLET_CALC_MT").fillna(0.0),
        _numeric(frame, "PELLET_PCT_FE2O3") * PELLET_FE_PER_FE2O3,
        _numeric(frame, "PELLET_PCT_TM"),
    )
    return ore + sinter + pellet


def running_hours(frame: pd.DataFrame, audit: pd.DataFrame) -> pd.Series:
    """Normal-operation hours with a known Fe charge: the feature basis."""

    eligible = audit["normal_eligible"].reindex(frame.index).fillna(False).astype(bool)
    return eligible & fe_charged_mt(frame).gt(0.0)


def window_coverage(
    frame: pd.DataFrame, audit: pd.DataFrame, settings: RobustCokeSettings
) -> pd.Series:
    """Share of the trailing window's hours that are running hours."""

    hours = int(settings.window_hours)
    running = running_hours(frame, audit).astype(float)
    return running.rolling(hours, min_periods=1).sum() / hours


def trusted_coke_hours(
    frame: pd.DataFrame, audit: pd.DataFrame, settings: RobustCokeSettings
) -> pd.Series:
    """Running hours whose coke mass can be believed.

    Two failure modes are removed. Single-hour weighing spikes: an hour whose
    coke-to-Fe ratio sits far outside its week. Multi-day undercount spells:
    from 1 Nov to 5 Dec 2025 COKE_CALC_MT read about half the real coke
    (155-175 kg/THM against a 300-317 set point and an unchanged fuel rate), so
    days whose coke-mass rate disagrees with the set point are dropped. The set
    point is a quality reference here, never a target or a feature.
    """

    fe = fe_charged_mt(frame)
    coke = _numeric(frame, "COKE_CALC_MT")
    production = _numeric(frame, "PRODUCTIONTONNESPERHR")
    trusted = running_hours(frame, audit) & coke.gt(0.0) & production.gt(0.0)

    week = int(settings.fe_level_hours)
    ratio = (coke / fe).where(trusted)
    centre = ratio.rolling(week, min_periods=24, center=True).median()
    spread = (ratio - centre).abs().rolling(week, min_periods=24, center=True).median()
    limit = float(settings.outlier_mad_multiple) * 1.4826 * spread.clip(lower=0.005)
    trusted &= (ratio - centre).abs().le(limit) | centre.isna()

    if settings.setpoint_column in frame:
        setpoint = _numeric(frame, settings.setpoint_column).where(trusted)
        mass_rate = (
            1000.0
            * coke.where(trusted).rolling(24, min_periods=12, center=True).sum()
            / production.where(trusted).rolling(24, min_periods=12, center=True).sum()
        )
        agreement = mass_rate / setpoint.rolling(24, min_periods=12, center=True).mean()
        in_band = agreement.between(
            float(settings.setpoint_ratio_min), float(settings.setpoint_ratio_max)
        )
        trusted &= in_band | agreement.isna()
    return trusted


def _window_sum(values: pd.Series, mask: pd.Series, hours: int) -> pd.Series:
    return values.where(mask).rolling(int(hours), min_periods=1).sum()


def robust_coke_target(
    frame: pd.DataFrame, audit: pd.DataFrame, settings: RobustCokeSettings
) -> pd.DataFrame:
    """Window coke rate on the charged-Fe basis, levelled to production.

    Args:
        frame: Cleaned hourly furnace frame on a regular hourly grid.
        audit: Hourly quality flags with ``normal_eligible``.
        settings: Target window and data-quality limits.

    Returns:
        Frame with the target (``robust_coke_kg_per_thm``, NaN where the window
        lacks trusted coverage) and its two factors for inspection.
    """

    hours = int(settings.window_hours)
    week = int(settings.fe_level_hours)
    fe = fe_charged_mt(frame)
    coke = _numeric(frame, "COKE_CALC_MT")
    production = _numeric(frame, "PRODUCTIONTONNESPERHR")
    trusted = trusted_coke_hours(frame, audit, settings)

    coke_per_fe = _window_sum(coke, trusted, hours) / _window_sum(fe, trusted, hours)
    level_min = max(24, week // 4)
    fe_per_thm = (
        fe.where(trusted).rolling(week, min_periods=level_min, center=True).sum()
        / production.where(trusted).rolling(week, min_periods=level_min, center=True).sum()
    )
    coverage = trusted.astype(float).rolling(hours, min_periods=1).sum() / hours
    rate = (1000.0 * coke_per_fe * fe_per_thm).replace([np.inf, -np.inf], np.nan)
    valid = trusted & coverage.ge(float(settings.min_window_coverage)) & rate.notna()
    return pd.DataFrame(
        {
            TARGET_COLUMN: rate.where(valid),
            "coke_per_fe_t": coke_per_fe,
            "fe_per_thm_t": fe_per_thm,
            "trusted_coke_hour": trusted,
            "trusted_window_coverage": coverage,
        },
        index=frame.index,
    )


def _iron_columns(frame: pd.DataFrame) -> list[str]:
    ore_slots = sorted(
        (column for column in frame if re.fullmatch(r"ORE_\d+_CALC_MT", str(column))),
        key=lambda column: int(str(column).split("_")[1]),
    )
    return (ore_slots or ["ORE_CALC_MT"]) + ["SINTER_CALC_MT", "TOTAL_PELLET_CALC_MT"]


def window_features(
    frame: pd.DataFrame,
    slag: pd.DataFrame,
    audit: pd.DataFrame,
    settings: RobustCokeSettings,
) -> pd.DataFrame:
    """Trailing-window conditions matching the robust target's window.

    No coke mass or coke rate enters. Burden composition is expressed as window
    mass shares and as burden per tonne Fe (grade), never per tonne of hot
    metal, because a per-THM burden rate carries the same charge count as coke.

    Args:
        frame: Cleaned hourly furnace frame on a regular hourly grid.
        slag: Hourly slag features computed with the fixed reference coke.
        audit: Hourly quality flags with ``normal_eligible``.
        settings: Window and assay handling.

    Returns:
        Hourly feature frame; each row describes its own trailing window.
    """

    hours = int(settings.window_hours)
    fe = fe_charged_mt(frame)
    running = running_hours(frame, audit)
    production = _window_sum(_numeric(frame, "PRODUCTIONTONNESPERHR"), running, hours)
    production = production.where(production.gt(0.0))

    features: dict[str, pd.Series] = {}
    for name, column in _FUEL_RATE_FEATURES.items():
        mass = _window_sum(_numeric(frame, column).fillna(0.0), running, hours)
        features[name] = 1000.0 * mass / production

    iron = {
        column: _window_sum(_numeric(frame, column).fillna(0.0), running, hours)
        for column in _iron_columns(frame)
        if column in frame
    }
    iron_total = sum(iron.values(), start=pd.Series(0.0, index=frame.index))
    iron_total = iron_total.where(iron_total.gt(0.0))
    for column, mass in iron.items():
        features[column.replace("_CALC_MT", "_SHARE__win")] = mass / iron_total
    features["BURDEN_PER_FE__win"] = iron_total / _window_sum(fe, running, hours)

    for column in _PROCESS_COLUMNS:
        if column in frame:
            features[column + "__win"] = (
                _numeric(frame, column).where(running).rolling(hours, min_periods=1).mean()
            )
    for column in _SLAG_COLUMNS:
        if column in slag:
            values = pd.to_numeric(slag[column], errors="coerce").reindex(frame.index)
            features[column + "__win"] = (
                values.where(running).rolling(hours, min_periods=1).mean()
            )
    for column in settings.assay_columns:
        if column in frame:
            features[column + "__assay"] = _numeric(frame, column).shift(
                int(settings.assay_delay_hours)
            )
    return pd.DataFrame(features, index=frame.index).replace([np.inf, -np.inf], np.nan)


def training_columns(features: pd.DataFrame) -> list[str]:
    """Usable features for a training slice, drivers first.

    Every sufficiently populated, non-constant window feature is kept; the set is
    small (~50) and hand-built, so no correlation ranking is applied.
    """

    usable = features.columns[
        (features.notna().mean() >= 0.5) & (features.nunique() > 1)
    ].tolist()
    drivers = [name for name in DRIVER_FEATURES if name in usable]
    return drivers + [name for name in usable if name not in drivers]


def monotone_constraint_string(columns: list[str]) -> str:
    """XGBoost ``monotone_constraints`` for an ordered feature list."""

    return "(" + ",".join(str(MONOTONE_CONSTRAINTS.get(name, 0)) for name in columns) + ")"


__all__ = [
    "DEFAULT_ASSAY_COLUMNS",
    "DRIVER_FEATURES",
    "MONOTONE_CONSTRAINTS",
    "NUT_COKE_FEATURE",
    "PCI_FEATURE",
    "SLAG_FEATURE",
    "TARGET_COLUMN",
    "TASK_NAME",
    "RobustCokeSettings",
    "fe_charged_mt",
    "is_robust_config",
    "monotone_constraint_string",
    "robust_coke_target",
    "running_hours",
    "training_columns",
    "trusted_coke_hours",
    "window_coverage",
    "window_features",
]
