"""Governed retraining for the live multi-horizon BF2 Si forecast."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd

from data.bmo.si_forecast_context import SiliconForecastSources
from utils.bmo.si_forecast.features import (
    HORIZON_MINUTES,
    PLANT_TZ,
    TARGET_WINDOW_MINUTES,
    build_features,
    prepare_labs,
    supervised_examples,
)


@dataclass(frozen=True)
class TrainingSettings:
    history_days: int = 60
    holdout_days: int = 7
    n_estimators: int = 240
    min_samples_leaf: int = 10
    max_features: float = 0.8
    random_state: int = 42
    min_train_samples: int = 200
    min_holdout_samples: int = 20
    horizons: tuple[int, ...] = HORIZON_MINUTES
    target_window_minutes: int = TARGET_WINDOW_MINUTES
    interval_coverage: float = 0.90


@dataclass
class HorizonTrainingResult:
    horizon_minutes: int
    estimator: Any
    median: np.ndarray
    clip_low: np.ndarray
    clip_high: np.ndarray
    scale_mean: np.ndarray
    scale_scale: np.ndarray
    interval_half_width: float
    train_count: int
    holdout_count: int


@dataclass
class SiliconTrainingResult:
    version: str
    models: dict[int, HorizonTrainingResult]
    feature_order: list[str]
    report: dict[str, Any]
    holdout: pd.DataFrame
    data_end: pd.Timestamp
    trained_until: pd.Timestamp

    # Compatibility for callers which inspected the former one-horizon result.
    @property
    def estimator(self) -> Any:
        return self.models[sorted(self.models)[0]].estimator


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ok = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[ok], predicted[ok]
    if not len(actual):
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "bias": None}
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else None,
        "bias": float(np.mean(error)),
        "within_005": float(np.mean(np.abs(error) <= 0.05)),
        "within_010": float(np.mean(np.abs(error) <= 0.10)),
    }


def _utc_series(values: pd.Series) -> pd.Series:
    clock = pd.to_datetime(values, errors="coerce", format="mixed")
    if getattr(clock.dt, "tz", None) is None:
        clock = clock.dt.tz_localize(PLANT_TZ)
    return clock.dt.tz_convert("UTC")


def _fit_preprocessing(
    train: pd.DataFrame, features: list[str]
) -> tuple[Any, np.ndarray, np.ndarray, Any, np.ndarray]:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    if train[features].notna().sum().eq(0).any():
        missing = train[features].notna().sum()
        raise ValueError(
            "Selected feature(s) have no training observations: "
            + ", ".join(missing[missing.eq(0)].index)
        )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    transformed = imputer.fit_transform(train[features])
    low = np.quantile(transformed, 0.005, axis=0)
    high = np.quantile(transformed, 0.995, axis=0)
    scaler = StandardScaler().fit(np.clip(transformed, low, high))
    scaled = scaler.transform(np.clip(transformed, low, high))
    return imputer, low, high, scaler, scaled


def _transform(
    frame: pd.DataFrame,
    features: list[str],
    imputer: Any,
    low: np.ndarray,
    high: np.ndarray,
    scaler: Any,
) -> np.ndarray:
    values = imputer.transform(frame[features])
    return scaler.transform(np.clip(values, low, high))


def _extra_trees(settings: TrainingSettings, *, seed_offset: int = 0) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=settings.n_estimators,
        min_samples_leaf=settings.min_samples_leaf,
        max_features=settings.max_features,
        n_jobs=1,
        random_state=settings.random_state + seed_offset,
    )


def _interval_from_training(
    train_rows: pd.DataFrame,
    features: list[str],
    settings: TrainingSettings,
    *,
    seed_offset: int,
) -> float:
    """Calibrate the interval on a chronological tail inside training only."""

    ordered = train_rows.sort_values("sample_time").reset_index(drop=True)
    calibration_count = max(1, int(np.ceil(len(ordered) * 0.20)))
    fit_rows = ordered.iloc[:-calibration_count]
    calibration = ordered.iloc[-calibration_count:]
    if fit_rows.empty:
        fit_rows = ordered
    imputer, low, high, scaler, x_fit = _fit_preprocessing(fit_rows, features)
    model = _extra_trees(settings, seed_offset=seed_offset).fit(
        x_fit, fit_rows["actual"].to_numpy(dtype=float)
    )
    prediction = model.predict(
        _transform(calibration, features, imputer, low, high, scaler)
    )
    error = np.abs(prediction - calibration["actual"].to_numpy(dtype=float))
    return float(np.quantile(error, settings.interval_coverage, method="higher"))


def train(
    sources: SiliconForecastSources,
    *,
    base_bundle: str | Path,
    data_end_origin: Any,
    settings: TrainingSettings | None = None,
    progress: Callable[[str], None] | None = None,
) -> SiliconTrainingResult:
    """Fit four direct candidates through the holdout boundary; never activate."""

    cfg = settings or TrainingSettings()
    say = progress or (lambda _message: None)
    bundle = Path(base_bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    features = [str(name) for name in manifest["feature_order"]]
    data_end = pd.Timestamp(data_end_origin)
    if data_end.tzinfo is None:
        data_end = data_end.tz_localize(PLANT_TZ)
    data_end = (
        data_end.tz_convert(PLANT_TZ)
        .floor(f"{cfg.target_window_minutes}min")
        .tz_convert("UTC")
    )
    holdout_start = data_end - pd.Timedelta(days=cfg.holdout_days)
    history_start = holdout_start - pd.Timedelta(days=cfg.history_days)
    maturity_end = data_end + pd.Timedelta(
        minutes=max(cfg.horizons) + cfg.target_window_minutes
    )

    say("building five-minute time-safe features")
    lab_rows = prepare_labs(sources.labs)
    origin_sets = []
    for horizon in cfg.horizons:
        origin_sets.append(
            lab_rows["sample"].dt.floor(f"{cfg.target_window_minutes}min")
            - pd.Timedelta(minutes=int(horizon))
        )
    target_origins = pd.concat(origin_sets).drop_duplicates().sort_values()
    lower = history_start.tz_convert(PLANT_TZ).tz_localize(None)
    upper = data_end.tz_convert(PLANT_TZ).tz_localize(None)
    target_origins = target_origins[target_origins.between(lower, upper)]
    if target_origins.empty:
        raise ValueError("No raw-cast targets are available for retraining.")
    origins = pd.DatetimeIndex(target_origins).tz_localize(PLANT_TZ)
    built = build_features(
        sources.online,
        sources.labs,
        sources.charge,
        sources.hourly_gate,
        origins,
        bundle,
        clock_version=2,
    )

    examples_by_horizon: dict[int, pd.DataFrame] = {}
    for horizon in cfg.horizons:
        examples = supervised_examples(
            built,
            sources.labs,
            available_by=maturity_end,
            horizon_minutes=int(horizon),
            target_window_minutes=cfg.target_window_minutes,
        )
        if examples.empty:
            raise ValueError(f"No eligible raw casts remain at +{horizon} minutes.")
        for column in ("origin", "sample_time", "available", "target_start", "target_end"):
            examples[column] = _utc_series(examples[column])
        examples_by_horizon[int(horizon)] = examples

    models: dict[int, HorizonTrainingResult] = {}
    holdout_frames: list[pd.DataFrame] = []
    per_horizon: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for ordinal, horizon in enumerate(cfg.horizons):
        horizon = int(horizon)
        examples = examples_by_horizon[horizon]
        train_mask = (
            examples["origin"].ge(history_start)
            & examples["origin"].lt(holdout_start)
            & examples["sample_time"].lt(holdout_start)
            & examples["available"].lt(holdout_start)
        )
        holdout_mask = (
            examples["origin"].ge(holdout_start)
            & examples["origin"].le(data_end)
            & examples["available"].le(maturity_end)
        )
        train_rows = examples.loc[train_mask].copy()
        holdout = examples.loc[holdout_mask].copy()
        if len(train_rows) < cfg.min_train_samples:
            raise ValueError(
                f"Only {len(train_rows)} eligible training casts at +{horizon} minutes; "
                f"need {cfg.min_train_samples}."
            )
        if len(holdout) < cfg.min_holdout_samples:
            raise ValueError(
                f"Only {len(holdout)} eligible holdout casts at +{horizon} minutes; "
                f"need {cfg.min_holdout_samples}."
            )

        say(f"fitting ExtraTrees for +{horizon} minutes")
        interval = _interval_from_training(
            train_rows, features, cfg, seed_offset=100 + ordinal
        )
        imputer, low, high, scaler, x_train = _fit_preprocessing(
            train_rows, features
        )
        model = _extra_trees(cfg, seed_offset=ordinal).fit(
            x_train, train_rows["actual"].to_numpy(dtype=float)
        )
        holdout["prediction"] = model.predict(
            _transform(holdout, features, imputer, low, high, scaler)
        )
        holdout["persistence"] = holdout["si_last"].to_numpy(dtype=float)
        holdout["mean3"] = holdout["si_mean3"].to_numpy(dtype=float)
        holdout["lower"] = np.maximum(0.0, holdout["prediction"] - interval)
        holdout["upper"] = holdout["prediction"] + interval
        holdout["inside_range"] = holdout["actual"].between(
            holdout["lower"], holdout["upper"]
        )

        model_metrics = _metrics(holdout["actual"], holdout["prediction"])
        persistence_metrics = _metrics(holdout["actual"], holdout["persistence"])
        mean3_metrics = _metrics(holdout["actual"], holdout["mean3"])
        range_coverage = float(holdout["inside_range"].mean())
        label = "Now" if horizon == 0 else f"+{horizon // 60} h"
        per_horizon.append(
            {
                "horizon_minutes": horizon,
                "label": label,
                "train_casts": int(len(train_rows)),
                "holdout_casts": int(len(holdout)),
                "extra_trees": model_metrics,
                "latest_si": persistence_metrics,
                "mean_last_3_si": mean3_metrics,
                "interval_half_width": interval,
                "range_coverage": range_coverage,
            }
        )
        checks.extend(
            [
                {
                    "name": f"{label}_training_sample_count",
                    "passed": len(train_rows) >= cfg.min_train_samples,
                    "detail": f"{len(train_rows)} casts (minimum {cfg.min_train_samples})",
                },
                {
                    "name": f"{label}_honest_holdout_sample_count",
                    "passed": len(holdout) >= cfg.min_holdout_samples,
                    "detail": f"{len(holdout)} casts (minimum {cfg.min_holdout_samples})",
                },
                {
                    "name": f"{label}_beats_persistence_mae",
                    "passed": bool(model_metrics["mae"] <= persistence_metrics["mae"]),
                    "detail": (
                        f"model {model_metrics['mae']:.4f} vs latest-Si "
                        f"{persistence_metrics['mae']:.4f}"
                    ),
                },
                {
                    "name": f"{label}_bias_within_0p10",
                    "passed": bool(abs(model_metrics["bias"]) <= 0.10),
                    "detail": f"bias {model_metrics['bias']:+.4f} Si percentage points",
                },
            ]
        )
        models[horizon] = HorizonTrainingResult(
            horizon_minutes=horizon,
            estimator=model,
            median=np.asarray(imputer.statistics_, dtype=float),
            clip_low=np.asarray(low, dtype=float),
            clip_high=np.asarray(high, dtype=float),
            scale_mean=np.asarray(scaler.mean_, dtype=float),
            scale_scale=np.asarray(scaler.scale_, dtype=float),
            interval_half_width=interval,
            train_count=len(train_rows),
            holdout_count=len(holdout),
        )
        holdout_frames.append(holdout)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    checks.append(
        {
            "name": "all_selected_features_observed",
            "passed": True,
            "detail": "Every selected feature has at least one training observation.",
        }
    )
    report = {
        "version": version,
        "model_family": "ExtraTreesRegressor",
        "target": "Raw HM Si in five-minute bins at Now, +1 h, +2 h and +3 h; no interpolation",
        "train_start": history_start.isoformat(),
        "trained_until": holdout_start.isoformat(),
        "holdout": [holdout_start.isoformat(), data_end.isoformat()],
        "data_end": data_end.isoformat(),
        "feature_count": len(features),
        "horizons": list(map(int, cfg.horizons)),
        "per_horizon": per_horizon,
        "checks": checks,
        "all_checks_passed": bool(all(row["passed"] for row in checks)),
        "parameters": {
            "history_days": cfg.history_days,
            "holdout_days": cfg.holdout_days,
            "n_estimators": cfg.n_estimators,
            "min_samples_leaf": cfg.min_samples_leaf,
            "max_features": cfg.max_features,
            "random_state": cfg.random_state,
            "clip_quantiles": [0.005, 0.995],
            "target_window_minutes": cfg.target_window_minutes,
            "interval_coverage": cfg.interval_coverage,
        },
    }
    say("preparing multi-horizon candidate bundle")
    return SiliconTrainingResult(
        version=version,
        models=models,
        feature_order=features,
        report=report,
        holdout=pd.concat(holdout_frames, ignore_index=True),
        data_end=data_end,
        trained_until=holdout_start,
    )


def _forest_arrays(result: HorizonTrainingResult) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in ("left", "right", "feature", "threshold", "value")
    }
    roots: list[int] = []
    offset = 0
    for estimator in result.estimator.estimators_:
        tree = estimator.tree_
        roots.append(offset)
        arrays["left"].append(
            np.where(tree.children_left >= 0, tree.children_left + offset, -1)
        )
        arrays["right"].append(
            np.where(tree.children_right >= 0, tree.children_right + offset, -1)
        )
        arrays["feature"].append(tree.feature)
        arrays["threshold"].append(tree.threshold)
        arrays["value"].append(tree.value[:, 0, 0])
        offset += tree.node_count
    packed = {name: np.concatenate(parts) for name, parts in arrays.items()}
    packed.update(
        roots=np.asarray(roots, dtype=np.int64),
        median=result.median,
        clip_low=result.clip_low,
        clip_high=result.clip_high,
        scale_mean=result.scale_mean,
        scale_scale=result.scale_scale,
    )
    return packed


def write_bundle(
    result: SiliconTrainingResult,
    directory: str | Path,
    *,
    base_bundle: str | Path,
) -> Path:
    """Export a candidate to the portable NumPy-only format-v2 bundle."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    horizons: list[dict[str, Any]] = []
    tree_count = 0
    node_count = 0
    for minutes, trained in sorted(result.models.items()):
        artifact = f"forest_h{minutes:03d}.npz"
        np.savez_compressed(target / artifact, **_forest_arrays(trained))
        horizons.append(
            {
                "minutes": minutes,
                "target_window_minutes": TARGET_WINDOW_MINUTES,
                "artifact": artifact,
                "sha256": hashlib.sha256((target / artifact).read_bytes()).hexdigest(),
                "interval_half_width": trained.interval_half_width,
                "interval_coverage": 0.90,
            }
        )
        tree_count += len(trained.estimator.estimators_)
        node_count += sum(tree.tree_.node_count for tree in trained.estimator.estimators_)
    manifest = {
        "format_version": 2,
        "version": result.version,
        "model_id": f"bf2_si_extra_5min60_{result.version}",
        "model_family": "ExtraTreesRegressor",
        "feature_order": result.feature_order,
        "horizons": horizons,
        "n_trees_total": int(tree_count),
        "n_nodes_total": int(node_count),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "trained_until": result.trained_until.isoformat(),
        "training_history_cap_days": 60,
        "target": "Raw cast Si in five-minute bins at 0, 60, 120 and 180 minutes",
        "origin": "Five-minute tick in Asia/Kolkata; floor before UTC conversion",
        "online": "Ten-minute completed bins with one-bin latency",
        "output_units": "Si mass percent; 0.40 means 0.40%",
        "uncertainty": "90% absolute-error interval calibrated inside training per horizon",
        "deployment_lifecycle": "candidate_until_approved",
        "training_parameters": result.report["parameters"],
        "holdout_metrics": result.report["per_horizon"],
        "rm_composition_features_in_this_selected_model": False,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    base = Path(base_bundle)
    for name in ("selected_feature_manifest.json", "online_channel_map.json"):
        shutil.copy2(base / name, target / name)
    (target / "report.json").write_text(
        json.dumps(result.report, indent=2), encoding="utf-8"
    )
    keep = [
        "id",
        "lab_sample_id",
        "origin",
        "horizon_minutes",
        "target_start",
        "target_end",
        "sample_time",
        "available",
        "actual",
        "prediction",
        "persistence",
        "mean3",
        "lower",
        "upper",
        "inside_range",
    ]
    result.holdout.loc[:, keep].to_csv(
        target / "holdout.csv.gz", index=False, compression="gzip"
    )
    return target


__all__ = [
    "HorizonTrainingResult",
    "SiliconTrainingResult",
    "TrainingSettings",
    "train",
    "write_bundle",
]
