"""Direct, price-independent coke-rate model inference and retraining.

The deployed model predicts the robust window coke rate in kg/THM (see
``utils.bmo.robust_coke_target``) for the current burden/process state, with PCI
and nut coke constrained to lower it. In Data-Driven mode BMO prices fuel on
that prediction; its explicit correction layer supplies blend deltas.

A candidate is deployed only when it clears a random whole-day holdout, a strict
later-time holdout (MAE and skill over last week's level), and a PCI-response
check. Deployment is an atomic pointer swap to a versioned bundle, so an
interrupted write cannot corrupt the active model and every previous deployment
remains available for rollback. ``maybe_retrain_in_background`` repeats this
automatically once the dataset has moved on.

Bundles without a ``robust_target`` config are the earlier hourly-target model
and still load, on its lag features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from utils.bmo.coke_model_pipeline import pipeline
from utils.bmo.coke_model_pipeline.context_model import context_features
from utils.bmo.robust_coke_target import (
    MONOTONE_CONSTRAINTS,
    NUT_COKE_FEATURE,
    PCI_FEATURE,
    TARGET_COLUMN,
    TASK_NAME,
    RobustCokeSettings,
    is_robust_config,
    monotone_constraint_string,
    robust_coke_target,
    training_columns,
    window_coverage,
    window_features,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_DIR = REPO_ROOT / "src/assets/models/bmo_coke_robust"
DEFAULT_DEPLOYMENT_DIR = REPO_ROOT / "src/storage/bmo_coke_model"
DEFAULT_DATASET_PATH = REPO_ROOT / "src/assets/data/furnace_dataset.csv"
AUTO_RETRAIN_STATE = "auto_retrain.json"


@dataclass(frozen=True)
class DirectCokePrediction:
    """One guarded direct-coke inference result."""

    value_kg_per_thm: float | None
    usable: bool
    origin_utc: str = ""
    window_start_utc: str = ""
    window_end_utc: str = ""
    lookback_hours: float = 1.0
    hourly_prediction_count: int = 0
    aggregation: str = "single_hour"
    latest_source_origin_utc: str = ""
    stale_hours: float | None = None
    source_age_hours: float | None = None
    missing_fraction: float | None = None
    outside_training_p01_p99: tuple[str, ...] = ()
    outside_training_details: tuple[dict[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()
    deployment_id: str = "bundled"
    model_path: str = ""
    later_date_r2: float | None = None
    target_window_hours: int | None = None
    latest_input_diagnostics: dict[str, float | str | None] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrainReport:
    """Validation and deployment outcome for one retraining request."""

    passed: bool
    deployed: bool
    deployment_id: str = ""
    random_metrics: dict[str, float | int | None] = field(default_factory=dict)
    later_time_metrics: dict[str, float | int | None] = field(default_factory=dict)
    fuel_response: dict[str, float | None] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    training_rows: int = 0
    feature_count: int = 0
    first_origin_utc: str = ""
    last_origin_utc: str = ""
    dataset_sha256: str = ""
    reasons: tuple[str, ...] = ()
    deployed_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_empty_labs(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Supply non-target dummy rows required by the shared feature function.

    The conditional coke schema contains no hot-metal Si/HMT features.  The
    shared joint-model feature builder nevertheless validates that those two lab
    streams exist.  A deliberately ancient row satisfies that structural check;
    its values are outside the recency tolerance and cannot enter coke features.
    """

    at = index.min() - pd.Timedelta(days=30)
    return pd.DataFrame(
        {
            "sample_id": ["not-used-by-coke-context"],
            "observation_time": [at],
            "available_time": [at],
            "si": [0.3],
            "hmt": [1480.0],
            "eligible": [True],
        }
    )


def _load_hourly(hourly_frame: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(hourly_frame, pd.DataFrame):
        return hourly_frame.copy()
    return pd.read_csv(Path(hourly_frame))


def _apply_current_fuel_overrides(
    raw: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    pci_kg_per_thm: float | None,
    nut_coke_kg_per_thm: float | None,
    lookback_hours: float = 1.0,
) -> pd.DataFrame:
    """Overlay operator fuel rates across the newest usable lookback window.

    Trailing source rows can have production but zero burden while the current
    hour is still being assembled. Those rows are not valid model origins, so
    applying an override to the raw maximum timestamp silently missed the row
    that inference actually selected. Restricting the overlay to positive-
    burden rows keeps operator settings aligned with eligible inference rows.
    Rebuilding features afterward updates lag0, rolling means, and slopes.
    """

    if pci_kg_per_thm is None and nut_coke_kg_per_thm is None:
        return raw
    time_col = str(cfg.get("time_col", "time"))
    if raw.empty or time_col not in raw or "PRODUCTIONTONNESPERHR" not in raw:
        return raw
    parsed = pd.to_datetime(
        raw[time_col], errors="coerce", format="mixed", dayfirst=True
    )
    if parsed.notna().sum() == 0:
        return raw
    production = pd.to_numeric(raw["PRODUCTIONTONNESPERHR"], errors="coerce")
    eligible = parsed.notna() & production.gt(0.0)
    burden_columns = [
        column
        for column in ("ORE_CALC_MT", "SINTER_CALC_MT", "TOTAL_PELLET_CALC_MT")
        if column in raw
    ]
    if burden_columns:
        burden = sum(
            (
                pd.to_numeric(raw[column], errors="coerce").fillna(0.0)
                for column in burden_columns
            ),
            start=pd.Series(0.0, index=raw.index),
        )
        eligible &= burden.gt(0.0)
    if not eligible.any():
        return raw

    newest = parsed.loc[eligible].max()
    hours = max(1.0, float(lookback_hours))
    selected = eligible & parsed.gt(newest - pd.Timedelta(hours=hours))
    selected &= parsed.le(newest)
    out = raw.copy()
    if pci_kg_per_thm is not None:
        out.loc[selected, "PCI_CALC_MT"] = (
            max(0.0, float(pci_kg_per_thm)) * production.loc[selected] / 1000.0
        )
    if nut_coke_kg_per_thm is not None:
        out.loc[selected, "NUTCOKE_CALC_MT"] = (
            max(0.0, float(nut_coke_kg_per_thm)) * production.loc[selected] / 1000.0
        )
    return out


def _trim_to_recent(raw: pd.DataFrame, cfg: dict[str, Any], hours: float) -> pd.DataFrame:
    """Keep only the newest ``hours`` of source rows (inference needs no more)."""

    time_col = str(cfg.get("time_col", "time"))
    if raw.empty or time_col not in raw:
        return raw
    parsed = pd.to_datetime(raw[time_col], errors="coerce", format="mixed", dayfirst=True)
    latest = parsed.max()
    if pd.isna(latest):
        return raw
    return raw.loc[parsed.ge(latest - pd.Timedelta(hours=float(hours)))].copy()


def _prepare_context(
    hourly_frame: pd.DataFrame | str | Path,
    config: dict[str, Any],
    *,
    pci_kg_per_thm: float | None = None,
    nut_coke_kg_per_thm: float | None = None,
    lookback_hours: float = 1.0,
    recent_hours: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    cfg = {**pipeline.DEFAULT, **config}
    raw = _load_hourly(hourly_frame)
    if recent_hours is not None:
        raw = _trim_to_recent(raw, cfg, recent_hours)
    override_hours = float(lookback_hours)
    if is_robust_config(cfg):
        # Every origin in the lookback describes its own trailing window, so an
        # operator rate must cover all of those windows to be seen in full.
        override_hours += RobustCokeSettings.from_config(cfg).window_hours - 1
    raw = _apply_current_fuel_overrides(
        raw,
        cfg,
        pci_kg_per_thm=pci_kg_per_thm,
        nut_coke_kg_per_thm=nut_coke_kg_per_thm,
        lookback_hours=override_hours,
    )
    cleaned, audit, _ = pipeline.clean_furnace(raw, cfg)
    if is_robust_config(cfg):
        # Same fixed 300 kg/THM reference coke as the earlier model, so the
        # target cannot reach the slag features through coke ash.
        slag, slag_errors = pipeline.slag_features(
            cleaned, {**cfg, "slag_reference_coke_rate": 300.0}
        )
        features = window_features(
            cleaned, slag, audit, RobustCokeSettings.from_config(cfg)
        )
        return cleaned, audit, features, cfg, slag_errors
    labs = _canonical_empty_labs(cleaned.index)
    features, _slag, slag_errors = context_features(cleaned, labs, cfg)
    return cleaned, audit, features, cfg, slag_errors


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


class DirectCokeModelService:
    """Load the active direct-coke bundle and perform guarded inference."""

    def __init__(
        self,
        *,
        bundled_dir: str | Path = DEFAULT_BUNDLE_DIR,
        deployment_dir: str | Path = DEFAULT_DEPLOYMENT_DIR,
        max_stale_hours: float = 6.0,
        max_source_age_hours: float = 6.0,
    ) -> None:
        self.bundled_dir = Path(bundled_dir)
        self.deployment_dir = Path(deployment_dir)
        self.max_stale_hours = max(0.0, float(max_stale_hours))
        self.max_source_age_hours = max(0.0, float(max_source_age_hours))
        self.bundle_dir, self.deployment_id = self._resolve_active_bundle()
        self.config = _read_json(self.bundle_dir / "config.json")
        self.schema = _read_json(self.bundle_dir / "coke_context_schema.json")
        self.model_path = self.bundle_dir / "coke_context.json"
        self.model = xgb.Booster(model_file=str(self.model_path))
        self.target_settings = (
            RobustCokeSettings.from_config(self.config)
            if is_robust_config(self.config)
            else None
        )

    def _inference_history_hours(self, lookback_hours: float) -> float | None:
        """Source history a current-state prediction needs, or None for all.

        The window features need the lookback plus one window; the 24-h assay
        delay and the stop-recovery flags need a little more. A week covers it
        and keeps the per-row slag balance off the full year on every page load.
        """

        if self.target_settings is None:
            return None
        settings = self.target_settings
        needed = (
            float(lookback_hours)
            + 2 * settings.window_hours
            + settings.assay_delay_hours
            + 48
        )
        return max(168.0, needed)

    def _resolve_active_bundle(self) -> tuple[Path, str]:
        pointer = self.deployment_dir / "active.json"
        try:
            active = _read_json(pointer)
            deployment_id = str(active["deployment_id"])
            version = self.deployment_dir / "versions" / deployment_id
            required = (
                version / "coke_context.json",
                version / "coke_context_schema.json",
                version / "config.json",
            )
            if all(path.is_file() for path in required):
                return version, deployment_id
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return self.bundled_dir, "bundled"

    def status(self) -> dict[str, Any]:
        validation = self.schema.get("validation", {}) or {}
        test = validation.get("test", {}) or validation.get("later_time", {}) or {}
        metadata_path = self.bundle_dir / "deployment_metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        return {
            "loaded": True,
            "deployment_id": self.deployment_id,
            "bundle_dir": str(self.bundle_dir),
            "task": str(self.schema.get("task", "")),
            "target_window_hours": (
                self.target_settings.window_hours if self.target_settings else None
            ),
            "feature_count": len(self.schema.get("features", [])),
            "later_date_r2": _float_or_none(test.get("r2")),
            "training_rows": int(self.schema.get("training_rows", 0) or 0),
            "fit_cutoff": str(self.schema.get("fit_cutoff", "")),
            "validation": validation,
            "metadata": metadata,
        }

    def prediction_history(
        self,
        hourly_frame: pd.DataFrame | str | Path,
        *,
        bias_window_hours: float = 24.0,
        bias_min_periods: int = 6,
        history_days: int | None = None,
    ) -> pd.DataFrame:
        """Return leakage-safe hourly raw, corrected, and actual coke rates.

        For a robust-target model "actual" is the robust window coke rate the
        model was trained on; for the earlier model it is the hourly mass ratio.
        The correction at hour t is the median raw-model residual from strictly
        earlier observations in the configured trailing window. The actual
        value at t therefore cannot correct its own prediction.
        """

        source = _load_hourly(hourly_frame)
        if history_days is not None and not source.empty:
            time_col = str(self.config.get("time_col", "time"))
            if time_col in source:
                timestamps = pd.to_datetime(
                    source[time_col],
                    errors="coerce",
                    format="mixed",
                    dayfirst=True,
                )
                latest = timestamps.max()
                if pd.notna(latest):
                    warmup_hours = max(72.0, float(bias_window_hours) + 24.0)
                    if self.target_settings is not None:
                        # The target's week-long Fe levelling and outlier
                        # screen are centred, so they need half a week either side.
                        warmup_hours = max(
                            warmup_hours,
                            float(
                                self.target_settings.window_hours
                                + self.target_settings.fe_level_hours
                            ),
                        )
                    start = latest - pd.Timedelta(
                        days=max(1, int(history_days)),
                        hours=warmup_hours,
                    )
                    source = source.loc[timestamps.ge(start)].copy()

        cleaned, audit, features, _cfg, _errors = _prepare_context(
            source,
            self.config,
        )
        columns = [str(item) for item in self.schema.get("features", [])]
        rows = features.reindex(columns=columns).replace([np.inf, -np.inf], np.nan)
        feature_usable = (
            rows.isna().mean(axis=1).le(0.5)
            if columns
            else pd.Series(False, index=rows.index)
        )
        coke_mt = _numeric_column(cleaned, "COKE_CALC_MT")
        hot_metal_mt = _numeric_column(cleaned, "PRODUCTIONTONNESPERHR")
        if self.target_settings is not None:
            actual = robust_coke_target(cleaned, audit, self.target_settings)[
                TARGET_COLUMN
            ]
        else:
            actual = coke_mt.mul(1000.0).div(hot_metal_mt.where(hot_metal_mt.gt(0.0)))
        eligible = audit["normal_eligible"].fillna(False)
        usable = eligible & feature_usable & actual.notna() & np.isfinite(actual)
        origins = rows.index[usable]
        if origins.empty:
            return pd.DataFrame(
                columns=[
                    "raw_predicted_coke_kg_per_thm",
                    "corrected_predicted_coke_kg_per_thm",
                    "actual_coke_kg_per_thm",
                    "prior_bias_correction_kg_per_thm",
                    "coke_calc_mt",
                    "hot_metal_mt",
                ]
            )

        matrix = xgb.DMatrix(rows.loc[origins].to_numpy(dtype=np.float32), nthread=2)
        raw = pd.Series(self.model.predict(matrix), index=origins, dtype=float)
        result = pd.DataFrame(
            {
                "raw_predicted_coke_kg_per_thm": raw,
                "actual_coke_kg_per_thm": actual.loc[origins].astype(float),
                "coke_calc_mt": coke_mt.loc[origins].astype(float),
                "hot_metal_mt": hot_metal_mt.loc[origins].astype(float),
            }
        ).sort_index()
        residual = (
            result["actual_coke_kg_per_thm"]
            - result["raw_predicted_coke_kg_per_thm"]
        )
        prior_residual = residual.shift(1)
        correction = prior_residual.rolling(
            f"{max(1.0, float(bias_window_hours)):g}h",
            min_periods=max(1, int(bias_min_periods)),
        ).median()
        result["prior_bias_correction_kg_per_thm"] = correction
        result["corrected_predicted_coke_kg_per_thm"] = (
            result["raw_predicted_coke_kg_per_thm"] + correction
        )
        if history_days is not None:
            cutoff = result.index.max() - pd.Timedelta(
                days=max(1, int(history_days))
            )
            result = result.loc[result.index >= cutoff]
        return result

    def predict_from_history(
        self,
        hourly_frame: pd.DataFrame | str | Path,
        *,
        pci_kg_per_thm: float | None = None,
        nut_coke_kg_per_thm: float | None = None,
        lookback_hours: float = 1.0,
        now: datetime | pd.Timestamp | None = None,
    ) -> DirectCokePrediction:
        hours = max(1.0, float(lookback_hours))
        cleaned, audit, features, _cfg, _errors = _prepare_context(
            hourly_frame,
            self.config,
            pci_kg_per_thm=pci_kg_per_thm,
            nut_coke_kg_per_thm=nut_coke_kg_per_thm,
            lookback_hours=hours,
            recent_hours=self._inference_history_hours(hours),
        )
        latest = cleaned.index.max()
        eligible = audit["normal_eligible"].fillna(False)
        window_hours = None
        if self.target_settings is not None:
            window_hours = int(self.target_settings.window_hours)
            eligible &= window_coverage(cleaned, audit, self.target_settings).ge(
                float(self.target_settings.min_window_coverage)
            )
        if not eligible.any():
            return DirectCokePrediction(
                value_kg_per_thm=None,
                usable=False,
                latest_source_origin_utc=str(latest),
                reasons=(
                    "No eligible normal-operation row with positive burden"
                    + (
                        f" and {window_hours}-hour window coverage."
                        if window_hours
                        else "."
                    ),
                ),
                deployment_id=self.deployment_id,
                model_path=str(self.model_path),
                target_window_hours=window_hours,
            )

        at = cleaned.index[eligible][-1]
        window_start = at - pd.Timedelta(hours=hours)
        origins = cleaned.index[eligible & (cleaned.index > window_start)]
        origins = origins[origins <= at]
        stale_hours = float((latest - at) / pd.Timedelta(hours=1))
        current_time = pd.Timestamp(now or datetime.now(timezone.utc))
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize("UTC")
        else:
            current_time = current_time.tz_convert("UTC")
        source_age_hours = max(
            0.0, float((current_time - latest) / pd.Timedelta(hours=1))
        )

        columns = [str(item) for item in self.schema.get("features", [])]
        rows = features.loc[origins].reindex(columns=columns)
        rows = rows.replace([np.inf, -np.inf], np.nan)
        missing_by_row = (
            rows.isna().mean(axis=1) if columns else pd.Series(1.0, index=origins)
        )
        missing_fraction = float(missing_by_row.loc[at]) if len(origins) else 1.0
        usable_origins = missing_by_row.index[missing_by_row.le(0.5)]
        outside: list[str] = []
        outside_details: list[dict[str, Any]] = []
        limits = self.schema.get("feature_limits", {}) or {}
        for name in columns:
            bounds = limits.get(name)
            if not bounds or usable_origins.empty:
                continue
            values = rows.loc[usable_origins, name].dropna()
            lower, upper = float(bounds[0]), float(bounds[1])
            if ((values < lower) | (values > upper)).any():
                outside.append(str(name))
                received = (
                    _float_or_none(rows.loc[at, name]) if at in rows.index else None
                )
                outside_details.append(
                    {
                        "feature": str(name),
                        "expected_p01": lower,
                        "expected_p99": upper,
                        "received": received,
                        "lookback_min": _float_or_none(values.min()),
                        "lookback_max": _float_or_none(values.max()),
                    }
                )

        value: float | None = None
        hourly_prediction_count = 0
        if columns and not usable_origins.empty:
            matrix = xgb.DMatrix(
                rows.loc[usable_origins].to_numpy(dtype=np.float32), nthread=2
            )
            hourly_predictions = self.model.predict(matrix)
            finite_predictions = hourly_predictions[np.isfinite(hourly_predictions)]
            hourly_prediction_count = int(len(finite_predictions))
            if hourly_prediction_count:
                value = float(np.median(finite_predictions))

        reasons: list[str] = []
        if stale_hours > self.max_stale_hours:
            reasons.append(
                f"Newest eligible burden/process row is {stale_hours:.1f} hours "
                f"behind the dataset ({at})."
            )
        if source_age_hours > self.max_source_age_hours:
            reasons.append(
                f"Dataset source is {source_age_hours:.1f} hours old ({latest})."
            )
        if usable_origins.empty:
            reasons.append(
                "No row in the selected lookback has enough model features available."
            )
        if value is None or not np.isfinite(value):
            reasons.append("The model did not produce a finite coke-rate prediction.")

        selected_row = cleaned.loc[at]
        burden = sum(
            float(selected_row.get(column, 0.0) or 0.0)
            for column in (
                "ORE_CALC_MT",
                "SINTER_CALC_MT",
                "TOTAL_PELLET_CALC_MT",
            )
            if pd.notna(selected_row.get(column))
        )
        validation = self.schema.get("validation", {}) or {}
        later_validation = (
            validation.get("test", {}) or validation.get("later_time", {}) or {}
        )
        return DirectCokePrediction(
            value_kg_per_thm=value,
            usable=not reasons,
            origin_utc=str(at),
            window_start_utc=str(origins.min()) if len(origins) else "",
            window_end_utc=str(at),
            lookback_hours=hours,
            hourly_prediction_count=hourly_prediction_count,
            aggregation="median" if hourly_prediction_count > 1 else "single_hour",
            latest_source_origin_utc=str(latest),
            stale_hours=stale_hours,
            source_age_hours=source_age_hours,
            missing_fraction=missing_fraction,
            outside_training_p01_p99=tuple(outside),
            outside_training_details=tuple(outside_details),
            reasons=tuple(reasons),
            deployment_id=self.deployment_id,
            model_path=str(self.model_path),
            later_date_r2=_float_or_none(later_validation.get("r2")),
            target_window_hours=window_hours,
            latest_input_diagnostics={
                "burden_mt": burden,
                "production_mt_per_hr": _float_or_none(
                    selected_row.get("PRODUCTIONTONNESPERHR")
                ),
                "ore_mt": _float_or_none(selected_row.get("ORE_CALC_MT")),
                "sinter_mt": _float_or_none(selected_row.get("SINTER_CALC_MT")),
                "pellet_mt": _float_or_none(selected_row.get("TOTAL_PELLET_CALC_MT")),
            },
        )


def _select_features(features: pd.DataFrame, target: pd.Series) -> list[str]:
    del target  # every usable window feature is kept; see training_columns
    return training_columns(features)


def _train_model(
    features: pd.DataFrame,
    target: pd.Series,
    columns: list[str],
    *,
    depth: int,
    rounds: int,
    seed: int,
) -> xgb.Booster:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": int(depth),
        "min_child_weight": 20,
        "eta": 0.04,
        "reg_lambda": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "seed": int(seed),
        "nthread": 2,
        # PCI and nut coke may only lower coke; the model sets the size.
        "monotone_constraints": monotone_constraint_string(columns),
    }
    matrix = xgb.DMatrix(
        features.loc[:, columns].to_numpy(dtype=np.float32),
        label=target.to_numpy(dtype=float),
        nthread=2,
    )
    return xgb.train(params, matrix, num_boost_round=int(rounds))


def _predict_model(
    model: xgb.Booster, features: pd.DataFrame, columns: list[str]
) -> np.ndarray:
    matrix = xgb.DMatrix(features.loc[:, columns].to_numpy(dtype=np.float32), nthread=2)
    return model.predict(matrix)


def _metrics(target: pd.Series, prediction: np.ndarray) -> dict[str, Any]:
    result = pipeline.metrics(target.to_numpy(dtype=float), prediction)
    error = np.asarray(prediction, dtype=float) - target.to_numpy(dtype=float)
    result["bias"] = float(np.nanmean(error)) if len(error) else None
    return result


def _training_target(
    cleaned: pd.DataFrame, audit: pd.DataFrame, cfg: dict[str, Any]
) -> pd.Series:
    settings = RobustCokeSettings.from_config(cfg)
    return robust_coke_target(cleaned, audit, settings)[TARGET_COLUMN]


def _random_day_split(
    origins: pd.DatetimeIndex, *, fraction: float, seed: int, purge_hours: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole random days; drop training rows whose window touches one.

    Neighbouring hours share most of a trailing window, so a random-row split
    would score a row against its own near-duplicate.
    """

    days = origins.floor("D")
    unique = pd.DatetimeIndex(days.unique())
    rng = np.random.default_rng(int(seed))
    count = min(len(unique), max(1, int(round(len(unique) * float(fraction)))))
    held = unique[rng.choice(len(unique), size=count, replace=False)]
    is_test = np.asarray(days.isin(held))
    flags = pd.Series(is_test.astype(float), index=origins)
    # Origins keep the IST half-hour offset, so the purge runs on their own
    # hourly lattice rather than on clock-hour bins.
    lattice = pd.date_range(origins.min(), origins.max(), freq="h")
    grid = flags.reindex(lattice).fillna(0.0)
    span = int(purge_hours) + 1
    near = (
        grid.rolling(span, min_periods=1).max()
        + grid[::-1].rolling(span, min_periods=1).max()[::-1]
    )
    touching = near.reindex(origins).fillna(0.0).to_numpy() > 0.0
    return np.flatnonzero(~touching), np.flatnonzero(is_test)


def _fuel_response(
    model: Any, features: pd.DataFrame, columns: list[str], name: str, delta: float
) -> float | None:
    """Mean change in predicted coke when one fuel feature moves by ``delta``."""

    if name not in columns or features.empty:
        return None
    shifted = features.copy()
    shifted[name] = shifted[name] + float(delta)
    change = _predict_model(model, shifted, columns) - _predict_model(
        model, features, columns
    )
    return float(np.mean(change))


def _prune_versions(deployment_dir: Path, *, keep: int, active_id: str) -> None:
    """Keep the newest ``keep`` versions and the active one; retraining is daily."""

    versions_dir = deployment_dir / "versions"
    if keep <= 0 or not versions_dir.is_dir():
        return
    versions = sorted(
        (path for path in versions_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for path in versions[keep:]:
        if path.name == active_id:
            continue
        for item in path.iterdir():
            item.unlink()
        path.rmdir()


def _rejected(reason: str, **values: Any) -> RetrainReport:
    return RetrainReport(passed=False, deployed=False, reasons=(reason,), **values)


def retrain_kwargs_from_config(retraining: dict[str, Any] | None) -> dict[str, Any]:
    """Gate and split settings for ``retrain_and_maybe_deploy`` from YAML."""

    cfg = dict(retraining or {})
    low, high = cfg.get("pci_replacement_range", (0.2, 1.5))
    return {
        "min_random_r2": float(cfg.get("min_random_r2", 0.70)),
        "max_later_time_mae": float(cfg.get("max_later_time_mae", 10.0)),
        "min_later_time_skill": float(cfg.get("min_later_time_skill", 0.0)),
        "pci_replacement_range": (float(low), float(high)),
        "random_test_fraction": float(cfg.get("random_test_fraction", 0.20)),
        "later_test_days": int(cfg.get("later_test_days", 14)),
        "seed": int(cfg.get("seed", 20260922)),
        "keep_versions": int(cfg.get("keep_versions", 30)),
    }


def retrain_and_maybe_deploy(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    bundled_dir: str | Path = DEFAULT_BUNDLE_DIR,
    deployment_dir: str | Path = DEFAULT_DEPLOYMENT_DIR,
    min_random_r2: float = 0.70,
    max_later_time_mae: float = 10.0,
    min_later_time_skill: float = 0.0,
    pci_replacement_range: tuple[float, float] = (0.2, 1.5),
    random_test_fraction: float = 0.20,
    later_test_days: int = 14,
    seed: int = 20260922,
    rounds_override: int | None = None,
    keep_versions: int = 30,
) -> RetrainReport:
    """Retrain on the robust coke target and activate only a validated candidate.

    Gates, all required:

    - Random whole-day holdout R2 >= ``min_random_r2``.
    - Strict later-time holdout MAE <= ``max_later_time_mae`` kg/THM. Later-time
      R2 is reported but not gated: a quiet fortnight has a standard deviation
      near 8 kg/THM, so R2 there measures the fortnight, not the model.
    - Later-time skill >= ``min_later_time_skill``: 1 - MAE / MAE of simply
      carrying the last week's average level forward.
    - PCI replacement within ``pci_replacement_range`` kg coke per kg PCI on the
      later-time rows, so a candidate that has stopped responding to PCI is
      never deployed.

    Args:
        dataset_path: Hourly furnace dataset CSV.
        bundled_dir: Bundle whose ``config.json`` defines cleaning and target.
        deployment_dir: Versioned deployments and the ``active.json`` pointer.
        min_random_r2: Random whole-day holdout gate.
        max_later_time_mae: Later-time MAE gate, kg/THM.
        min_later_time_skill: Later-time skill gate over the last-week level.
        pci_replacement_range: Allowed kg coke per kg PCI (inclusive).
        random_test_fraction: Share of development days held out at random.
        later_test_days: Length of the strict later-time holdout.
        seed: Random seed for the day split and XGBoost.
        rounds_override: Boosting rounds instead of the configured value.
        keep_versions: Deployed versions retained for rollback (0 keeps all).

    Returns:
        The validation and deployment outcome.
    """

    dataset_path = Path(dataset_path)
    bundled_dir = Path(bundled_dir)
    deployment_dir = Path(deployment_dir)
    config = _read_json(bundled_dir / "config.json")
    config.setdefault("robust_target", {})
    cleaned, audit, all_features, cfg, _errors = _prepare_context(dataset_path, config)
    settings = RobustCokeSettings.from_config(cfg)
    target = _training_target(cleaned, audit, cfg)
    thresholds: dict[str, Any] = {
        "random_r2": float(min_random_r2),
        "later_time_mae": float(max_later_time_mae),
        "later_time_skill": float(min_later_time_skill),
        "pci_replacement_range": [float(v) for v in pci_replacement_range],
    }
    dataset_sha = _dataset_hash(dataset_path)

    eligible = (
        audit["normal_eligible"].fillna(False)
        & audit["in_requested_window"].fillna(False)
        & target.notna()
    )
    origins = cleaned.index[eligible]
    min_train_rows = int(cfg.get("min_train_rows", 100))
    if len(origins) < max(min_train_rows + 20, 150):
        return _rejected(
            "Not enough eligible rows with a trusted coke window to retrain.",
            thresholds=thresholds,
            training_rows=int(len(origins)),
            dataset_sha256=dataset_sha,
        )

    xgb_cfg = dict(cfg.get("xgboost", {}) or {})
    depth = int(xgb_cfg.get("depth", 3))
    rounds = int(rounds_override or xgb_cfg.get("rounds", 300))
    # Labels are trailing windows: a gap shorter than one window would let a
    # training label share hours with a test label.
    purge_hours = max(int(cfg.get("purge_hours", 12)), int(settings.window_hours))

    X = all_features.loc[eligible].copy()
    y = target.loc[eligible].astype(float)
    last_origin = origins.max()
    later_start = last_origin.floor("D") - pd.Timedelta(days=max(1, later_test_days))
    later_train_mask = origins < later_start - pd.Timedelta(hours=purge_hours)
    later_test_mask = origins >= later_start
    if later_train_mask.sum() < min_train_rows:
        return _rejected(
            "Later-time split leaves too few training rows.",
            thresholds=thresholds,
            training_rows=int(len(origins)),
            dataset_sha256=dataset_sha,
        )
    if later_test_mask.sum() < int(cfg.get("min_test_rows", 20)):
        return _rejected(
            "Later-time holdout has too few eligible rows.",
            thresholds=thresholds,
            training_rows=int(len(origins)),
            dataset_sha256=dataset_sha,
        )

    development_positions = np.flatnonzero(later_train_mask)
    development = origins[development_positions]
    random_train_local, random_test_local = _random_day_split(
        development,
        fraction=random_test_fraction,
        seed=seed,
        purge_hours=purge_hours,
    )
    random_train_positions = development_positions[random_train_local]
    random_test_positions = development_positions[random_test_local]
    if len(random_train_positions) < min_train_rows or len(random_test_positions) < 2:
        return _rejected(
            "Random-day split leaves too few rows.",
            thresholds=thresholds,
            training_rows=int(len(origins)),
            dataset_sha256=dataset_sha,
        )

    random_columns = _select_features(
        X.iloc[random_train_positions], y.iloc[random_train_positions]
    )
    random_model = _train_model(
        X.iloc[random_train_positions],
        y.iloc[random_train_positions],
        random_columns,
        depth=depth,
        rounds=rounds,
        seed=seed,
    )
    random_metrics = _metrics(
        y.iloc[random_test_positions],
        _predict_model(random_model, X.iloc[random_test_positions], random_columns),
    )
    random_metrics["held_out_days"] = int(
        pd.DatetimeIndex(origins[random_test_positions]).floor("D").nunique()
    )

    later_train_positions = np.flatnonzero(later_train_mask)
    later_test_positions = np.flatnonzero(later_test_mask)
    later_columns = _select_features(
        X.iloc[later_train_positions], y.iloc[later_train_positions]
    )
    later_model = _train_model(
        X.iloc[later_train_positions],
        y.iloc[later_train_positions],
        later_columns,
        depth=depth,
        rounds=rounds,
        seed=seed,
    )
    later_test_frame = X.iloc[later_test_positions]
    later_test_target = y.iloc[later_test_positions]
    later_metrics = _metrics(
        later_test_target,
        _predict_model(later_model, later_test_frame, later_columns),
    )
    cutoff = later_start - pd.Timedelta(hours=purge_hours)
    last_week = y.loc[(y.index < cutoff) & (y.index >= cutoff - pd.Timedelta(days=7))]
    baseline_mae = (
        float(np.mean(np.abs(later_test_target.to_numpy() - float(last_week.mean()))))
        if not last_week.empty
        else None
    )
    later_mae = _float_or_none(later_metrics.get("mae"))
    skill = (
        1.0 - later_mae / baseline_mae
        if later_mae is not None and baseline_mae
        else None
    )
    later_metrics.update(
        {
            "last_week_level_mae": baseline_mae,
            "skill_vs_last_week_level": skill,
            "test_start": str(later_start),
            "test_end": str(last_origin),
        }
    )

    pci_change = _fuel_response(
        later_model, later_test_frame, later_columns, PCI_FEATURE, -40.0
    )
    nut_change = _fuel_response(
        later_model, later_test_frame, later_columns, NUT_COKE_FEATURE, -10.0
    )
    fuel_response = {
        "coke_change_for_pci_minus_40": pci_change,
        "pci_replacement_kg_per_kg": None if pci_change is None else pci_change / 40.0,
        "coke_change_for_nut_coke_minus_10": nut_change,
    }

    random_r2 = _float_or_none(random_metrics.get("r2"))
    replacement = fuel_response["pci_replacement_kg_per_kg"]
    reasons: list[str] = []
    if random_r2 is None or random_r2 < float(min_random_r2):
        reasons.append(f"Random-day R2 {random_r2!s} is below {min_random_r2:.2f}.")
    if later_mae is None or later_mae > float(max_later_time_mae):
        reasons.append(
            f"Later-time MAE {later_mae!s} kg/THM is above {max_later_time_mae:.1f}."
        )
    if skill is None or skill < float(min_later_time_skill):
        reasons.append(
            f"Later-time skill {skill!s} over the last-week level is below "
            f"{min_later_time_skill:.2f}."
        )
    low, high = (float(v) for v in pci_replacement_range)
    if replacement is None or not low <= replacement <= high:
        reasons.append(
            f"PCI replacement {replacement!s} kg/kg is outside {low:.2f}-{high:.2f}."
        )
    if reasons:
        return RetrainReport(
            passed=False,
            deployed=False,
            random_metrics=random_metrics,
            later_time_metrics=later_metrics,
            fuel_response=fuel_response,
            thresholds=thresholds,
            training_rows=int(len(origins)),
            feature_count=len(later_columns),
            first_origin_utc=str(origins.min()),
            last_origin_utc=str(origins.max()),
            dataset_sha256=dataset_sha,
            reasons=tuple(reasons),
        )

    # The deployed model learns from every trusted window, newest included.
    final_columns = _select_features(X, y)
    final_model = _train_model(
        X, y, final_columns, depth=depth, rounds=rounds, seed=seed
    )

    trained_at = datetime.now(timezone.utc)
    deployment_id = trained_at.strftime("%Y%m%dT%H%M%S%fZ")
    version_dir = deployment_dir / "versions" / deployment_id
    version_dir.mkdir(parents=True, exist_ok=False)
    final_model.save_model(str(version_dir / "coke_context.json"))
    feature_limits: dict[str, list[float]] = {}
    for column in final_columns:
        quantiles = X[column].quantile([0.01, 0.99])
        if quantiles.notna().all():
            feature_limits[column] = [
                float(quantiles.iloc[0]),
                float(quantiles.iloc[1]),
            ]

    schema = {
        "task": TASK_NAME,
        "units": "kg/tHM",
        "target": (
            "1,000 x trailing-window coke / Fe charged x week Fe charged / "
            "production; see utils.bmo.robust_coke_target"
        ),
        "target_settings": asdict(settings),
        "features": final_columns,
        "monotone_constraints": {
            name: sign
            for name, sign in MONOTONE_CONSTRAINTS.items()
            if name in final_columns
        },
        "feature_limits": feature_limits,
        "fit_cutoff": str(cleaned.index.max()),
        "latest_label_available": str(last_origin),
        "training_rows": int(len(origins)),
        "parameters": {"depth": depth, "rounds": rounds, "seed": int(seed)},
        "slag_coke_basis_kg_thm": 300,
        "validation": {
            "random_day": random_metrics,
            "later_time": later_metrics,
            "fuel_response": fuel_response,
        },
        "deployment_gate_passed": True,
        # Predictive validation is not causal optimisation validation. BMO uses
        # the model only for the current state and the operator's fuel rates.
        "optimisation_validated": False,
    }
    _write_json(version_dir / "coke_context_schema.json", schema)
    _write_json(version_dir / "config.json", config)
    metadata = {
        "deployment_id": deployment_id,
        "trained_at_utc": trained_at.isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "random_metrics": random_metrics,
        "later_time_metrics": later_metrics,
        "fuel_response": fuel_response,
        "thresholds": thresholds,
        "training_rows": int(len(origins)),
        "features": len(final_columns),
        "xgboost_version": xgb.__version__,
    }
    _write_json(version_dir / "deployment_metadata.json", metadata)

    deployment_dir.mkdir(parents=True, exist_ok=True)
    pointer_tmp = deployment_dir / "active.tmp.json"
    _write_json(pointer_tmp, {"deployment_id": deployment_id})
    os.replace(pointer_tmp, deployment_dir / "active.json")
    try:
        _prune_versions(deployment_dir, keep=int(keep_versions), active_id=deployment_id)
    except OSError:  # a locked old file must not undo a good deployment
        log.warning("Could not prune old coke-model versions", exc_info=True)

    return RetrainReport(
        passed=True,
        deployed=True,
        deployment_id=deployment_id,
        random_metrics=random_metrics,
        later_time_metrics=later_metrics,
        fuel_response=fuel_response,
        thresholds=thresholds,
        training_rows=int(len(origins)),
        feature_count=len(final_columns),
        first_origin_utc=str(origins.min()),
        last_origin_utc=str(origins.max()),
        dataset_sha256=dataset_sha,
        deployed_path=str(version_dir),
    )


_auto_retrain_lock = threading.Lock()


def auto_retrain_state(deployment_dir: str | Path = DEFAULT_DEPLOYMENT_DIR) -> dict[str, Any]:
    """Outcome of the last automatic retrain attempt, or an empty dict."""

    path = Path(deployment_dir) / AUTO_RETRAIN_STATE
    try:
        return _read_json(path)
    except (OSError, ValueError):
        return {}


def auto_retrain_due(
    *,
    dataset_path: str | Path,
    deployment_dir: str | Path,
    every_hours: float,
    now: datetime | None = None,
) -> bool:
    """Whether the dataset has changed and the last attempt is old enough.

    A failed attempt counts as an attempt, so a candidate that keeps missing a
    gate is retried once per interval rather than on every page load.
    """

    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        return False
    state = auto_retrain_state(deployment_dir)
    if not state:
        return True
    if int(state.get("dataset_mtime_ns", -1)) == int(dataset_path.stat().st_mtime_ns):
        return False
    try:
        last = datetime.fromisoformat(str(state.get("last_attempt_utc")))
    except ValueError:
        return True
    current = now or datetime.now(timezone.utc)
    return (current - last).total_seconds() >= float(every_hours) * 3600.0


def _auto_retrain_worker(
    dataset_path: Path,
    bundled_dir: Path,
    deployment_dir: Path,
    retrain_kwargs: dict[str, Any],
) -> None:
    started = datetime.now(timezone.utc)
    state: dict[str, Any] = {
        "last_attempt_utc": started.isoformat(),
        "dataset_mtime_ns": int(dataset_path.stat().st_mtime_ns),
    }
    try:
        report = retrain_and_maybe_deploy(
            dataset_path,
            bundled_dir=bundled_dir,
            deployment_dir=deployment_dir,
            **retrain_kwargs,
        )
        state.update(
            {
                "deployed": bool(report.deployed),
                "deployment_id": report.deployment_id,
                "reasons": list(report.reasons),
                "later_time_mae": report.later_time_metrics.get("mae"),
                "random_day_r2": report.random_metrics.get("r2"),
                "pci_replacement_kg_per_kg": report.fuel_response.get(
                    "pci_replacement_kg_per_kg"
                ),
            }
        )
        log.info(
            "Automatic coke-model retrain finished: deployed=%s %s",
            report.deployed,
            "; ".join(report.reasons),
        )
    except Exception as exc:  # noqa: BLE001 - recorded, never raised into the app
        log.exception("Automatic coke-model retrain failed")
        state.update({"deployed": False, "error": str(exc)})
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            deployment_dir.mkdir(parents=True, exist_ok=True)
            tmp = deployment_dir / (AUTO_RETRAIN_STATE + ".tmp")
            _write_json(tmp, state)
            os.replace(tmp, deployment_dir / AUTO_RETRAIN_STATE)
        finally:
            _auto_retrain_lock.release()


def maybe_retrain_in_background(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    bundled_dir: str | Path = DEFAULT_BUNDLE_DIR,
    deployment_dir: str | Path = DEFAULT_DEPLOYMENT_DIR,
    every_hours: float = 24.0,
    retrain_kwargs: dict[str, Any] | None = None,
) -> bool:
    """Start one background retrain when it is due; return whether one is running.

    Called on page load. The retrain is the same gated routine as the manual
    button, so an automatic run can only replace the active model with one that
    passes every gate; the page picks it up through the ``active.json`` token.
    """

    if not auto_retrain_due(
        dataset_path=dataset_path,
        deployment_dir=deployment_dir,
        every_hours=every_hours,
    ):
        return _auto_retrain_lock.locked()
    if not _auto_retrain_lock.acquire(blocking=False):
        return True
    threading.Thread(
        target=_auto_retrain_worker,
        args=(
            Path(dataset_path),
            Path(bundled_dir),
            Path(deployment_dir),
            dict(retrain_kwargs or {}),
        ),
        name="bmo-coke-retrain",
        daemon=True,
    ).start()
    return True


if __name__ == "__main__":
    # Manual / scheduled use: python -m utils.bmo.direct_coke_model (from src/)
    print(json.dumps(retrain_and_maybe_deploy().to_dict(), indent=2, default=str))
