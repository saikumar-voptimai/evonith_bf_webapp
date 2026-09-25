"""Direct, price-independent coke-rate XGBoost inference and retraining.

The deployed model predicts the observed coke rate in kg/THM for the current
burden/process state.  It is an anchor, not a candidate-by-candidate causal
response model; BMO's explicit coke-correction layer supplies blend deltas.

Retraining uses the same frozen preprocessing contract as the audited artifact.
A candidate is deployed only when both a random development split and a strict
later-time holdout clear configured R2 gates.  Deployment is an atomic pointer
swap to a versioned bundle, so an interrupted write cannot corrupt the active
model and every previous deployment remains available for rollback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from utils.bmo.coke_model_pipeline import pipeline
from utils.bmo.coke_model_pipeline.context_model import context_features

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_DIR = REPO_ROOT / "src/assets/models/bmo_coke_xgb"
DEFAULT_DEPLOYMENT_DIR = REPO_ROOT / "src/storage/bmo_coke_model"
DEFAULT_DATASET_PATH = REPO_ROOT / "src/assets/data/furnace_dataset.csv"


@dataclass(frozen=True)
class DirectCokePrediction:
    """One guarded direct-coke inference result."""

    value_kg_per_thm: float | None
    usable: bool
    origin_utc: str = ""
    latest_source_origin_utc: str = ""
    stale_hours: float | None = None
    source_age_hours: float | None = None
    missing_fraction: float | None = None
    outside_training_p01_p99: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    deployment_id: str = "bundled"
    model_path: str = ""
    later_date_r2: float | None = None
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
    thresholds: dict[str, float] = field(default_factory=dict)
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
) -> pd.DataFrame:
    """Overlay only the newest hour's operator fuel rates before engineering.

    Historical lags remain observed.  Rebuilding features after this overlay
    updates lag0, the four-hour mean, and the four-hour slope coherently.
    """

    if pci_kg_per_thm is None and nut_coke_kg_per_thm is None:
        return raw
    time_col = str(cfg.get("time_col", "time"))
    if raw.empty or time_col not in raw or "PRODUCTIONTONNESPERHR" not in raw:
        return raw
    parsed = pd.to_datetime(raw[time_col], errors="coerce")
    if parsed.notna().sum() == 0:
        return raw
    row_id = parsed.idxmax()
    production = pd.to_numeric(
        pd.Series([raw.at[row_id, "PRODUCTIONTONNESPERHR"]]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(production) or production <= 0:
        return raw
    out = raw.copy()
    if pci_kg_per_thm is not None:
        out.at[row_id, "PCI_CALC_MT"] = (
            max(0.0, float(pci_kg_per_thm)) * float(production) / 1000.0
        )
    if nut_coke_kg_per_thm is not None:
        out.at[row_id, "NUTCOKE_CALC_MT"] = (
            max(0.0, float(nut_coke_kg_per_thm)) * float(production) / 1000.0
        )
    return out


def _prepare_context(
    hourly_frame: pd.DataFrame | str | Path,
    config: dict[str, Any],
    *,
    pci_kg_per_thm: float | None = None,
    nut_coke_kg_per_thm: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    cfg = {**pipeline.DEFAULT, **config}
    raw = _apply_current_fuel_overrides(
        _load_hourly(hourly_frame),
        cfg,
        pci_kg_per_thm=pci_kg_per_thm,
        nut_coke_kg_per_thm=nut_coke_kg_per_thm,
    )
    cleaned, audit, _ = pipeline.clean_furnace(raw, cfg)
    labs = _canonical_empty_labs(cleaned.index)
    features, _slag, slag_errors = context_features(cleaned, labs, cfg)
    return cleaned, audit, features, cfg, slag_errors


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


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
            "feature_count": len(self.schema.get("features", [])),
            "later_date_r2": _float_or_none(test.get("r2")),
            "training_rows": int(self.schema.get("training_rows", 0) or 0),
            "fit_cutoff": str(self.schema.get("fit_cutoff", "")),
            "metadata": metadata,
        }

    def predict_from_history(
        self,
        hourly_frame: pd.DataFrame | str | Path,
        *,
        pci_kg_per_thm: float | None = None,
        nut_coke_kg_per_thm: float | None = None,
        now: datetime | pd.Timestamp | None = None,
    ) -> DirectCokePrediction:
        cleaned, audit, features, _cfg, _errors = _prepare_context(
            hourly_frame,
            self.config,
            pci_kg_per_thm=pci_kg_per_thm,
            nut_coke_kg_per_thm=nut_coke_kg_per_thm,
        )
        latest = cleaned.index.max()
        eligible = audit["normal_eligible"].fillna(False)
        if not eligible.any():
            return DirectCokePrediction(
                value_kg_per_thm=None,
                usable=False,
                latest_source_origin_utc=str(latest),
                reasons=("No eligible normal-operation row with positive burden.",),
                deployment_id=self.deployment_id,
                model_path=str(self.model_path),
            )

        at = cleaned.index[eligible][-1]
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
        row = features.loc[at].reindex(columns).replace([np.inf, -np.inf], np.nan)
        missing_fraction = float(row.isna().mean()) if columns else 1.0
        outside: list[str] = []
        limits = self.schema.get("feature_limits", {}) or {}
        for name, value in row.items():
            bounds = limits.get(name)
            if (
                bounds
                and pd.notna(value)
                and (float(value) < float(bounds[0]) or float(value) > float(bounds[1]))
            ):
                outside.append(str(name))

        value: float | None = None
        if columns and missing_fraction <= 0.5:
            matrix = xgb.DMatrix(row.to_numpy(dtype=np.float32)[None, :], nthread=2)
            value = float(self.model.predict(matrix)[0])

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
        if missing_fraction > 0.5:
            reasons.append(
                f"{missing_fraction:.0%} of selected model features are unavailable."
            )
        if value is None or not np.isfinite(value):
            reasons.append("The model did not produce a finite coke-rate prediction.")

        latest_row = cleaned.loc[latest]
        burden = sum(
            float(latest_row.get(column, 0.0) or 0.0)
            for column in (
                "ORE_CALC_MT",
                "SINTER_CALC_MT",
                "TOTAL_PELLET_CALC_MT",
            )
            if pd.notna(latest_row.get(column))
        )
        validation = self.schema.get("validation", {}) or {}
        later_validation = (
            validation.get("test", {}) or validation.get("later_time", {}) or {}
        )
        return DirectCokePrediction(
            value_kg_per_thm=value,
            usable=not reasons,
            origin_utc=str(at),
            latest_source_origin_utc=str(latest),
            stale_hours=stale_hours,
            source_age_hours=source_age_hours,
            missing_fraction=missing_fraction,
            outside_training_p01_p99=tuple(outside),
            reasons=tuple(reasons),
            deployment_id=self.deployment_id,
            model_path=str(self.model_path),
            later_date_r2=_float_or_none(later_validation.get("r2")),
            latest_input_diagnostics={
                "burden_mt": burden,
                "production_mt_per_hr": _float_or_none(
                    latest_row.get("PRODUCTIONTONNESPERHR")
                ),
                "ore_mt": _float_or_none(latest_row.get("ORE_CALC_MT")),
                "sinter_mt": _float_or_none(latest_row.get("SINTER_CALC_MT")),
                "pellet_mt": _float_or_none(latest_row.get("TOTAL_PELLET_CALC_MT")),
            },
        )


def _select_features(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    family: str,
    top_k: int,
) -> list[str]:
    local = features.reset_index(drop=True)
    mask = pd.Series(True, index=local.index)
    return pipeline.chosen_columns(
        local,
        mask,
        family,
        int(top_k),
        target.reset_index(drop=True),
    )


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
    return pipeline.metrics(target.to_numpy(dtype=float), prediction)


def retrain_and_maybe_deploy(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    bundled_dir: str | Path = DEFAULT_BUNDLE_DIR,
    deployment_dir: str | Path = DEFAULT_DEPLOYMENT_DIR,
    min_random_r2: float = 0.70,
    min_later_time_r2: float = 0.65,
    random_test_fraction: float = 0.20,
    later_test_days: int = 14,
    seed: int = 20260922,
    rounds_override: int | None = None,
) -> RetrainReport:
    """Retrain and atomically activate only a twice-validated candidate."""

    dataset_path = Path(dataset_path)
    bundled_dir = Path(bundled_dir)
    deployment_dir = Path(deployment_dir)
    config = _read_json(bundled_dir / "config.json")
    base_schema = _read_json(bundled_dir / "coke_context_schema.json")
    cleaned, audit, all_features, cfg, _errors = _prepare_context(dataset_path, config)
    eligible = (
        audit["normal_eligible"].fillna(False)
        & audit["in_requested_window"].fillna(False)
        & cleaned["COKE_CALC_KG_THM"].notna()
    )
    origins = cleaned.index[eligible]
    if len(origins) < max(int(cfg.get("min_train_rows", 100)) + 20, 150):
        return RetrainReport(
            passed=False,
            deployed=False,
            thresholds={
                "random_r2": float(min_random_r2),
                "later_time_r2": float(min_later_time_r2),
            },
            training_rows=int(len(origins)),
            dataset_sha256=_dataset_hash(dataset_path),
            reasons=("Not enough eligible normal-operation rows to retrain.",),
        )

    recipe = dict(base_schema.get("parameters", {}) or {})
    window_days = int(recipe.get("window") or 60)
    family = str(recipe.get("family", "process"))
    top_k = int(recipe.get("top_k", 80))
    depth = int(recipe.get("depth", 3))
    rounds = int(rounds_override or recipe.get("rounds", 300))
    purge_hours = int(cfg.get("purge_hours", 12))

    X = all_features.loc[eligible].copy()
    y = cleaned.loc[eligible, "COKE_CALC_KG_THM"].astype(float)
    last_origin = origins.max()
    later_start = last_origin.floor("D") - pd.Timedelta(days=max(1, later_test_days))
    later_train_mask = (origins < later_start - pd.Timedelta(hours=purge_hours)) & (
        origins >= later_start - pd.Timedelta(days=window_days)
    )
    later_test_mask = origins >= later_start
    if later_train_mask.sum() < int(cfg.get("min_train_rows", 100)):
        return RetrainReport(
            passed=False,
            deployed=False,
            thresholds={
                "random_r2": float(min_random_r2),
                "later_time_r2": float(min_later_time_r2),
            },
            training_rows=int(len(origins)),
            dataset_sha256=_dataset_hash(dataset_path),
            reasons=("Later-time split leaves too few training rows.",),
        )
    if later_test_mask.sum() < int(cfg.get("min_test_rows", 20)):
        return RetrainReport(
            passed=False,
            deployed=False,
            thresholds={
                "random_r2": float(min_random_r2),
                "later_time_r2": float(min_later_time_r2),
            },
            training_rows=int(len(origins)),
            dataset_sha256=_dataset_hash(dataset_path),
            reasons=("Later-time holdout has too few eligible rows.",),
        )

    development_positions = np.flatnonzero(later_train_mask)
    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(development_positions)
    random_test_count = max(20, int(round(len(shuffled) * random_test_fraction)))
    random_test_positions = shuffled[:random_test_count]
    random_train_positions = shuffled[random_test_count:]
    if len(random_train_positions) < int(cfg.get("min_train_rows", 100)):
        return RetrainReport(
            passed=False,
            deployed=False,
            training_rows=int(len(origins)),
            dataset_sha256=_dataset_hash(dataset_path),
            reasons=("Random split leaves too few training rows.",),
        )

    random_columns = _select_features(
        X.iloc[random_train_positions],
        y.iloc[random_train_positions],
        family=family,
        top_k=top_k,
    )
    random_model = _train_model(
        X.iloc[random_train_positions],
        y.iloc[random_train_positions],
        random_columns,
        depth=depth,
        rounds=rounds,
        seed=seed,
    )
    random_prediction = _predict_model(
        random_model, X.iloc[random_test_positions], random_columns
    )
    random_metrics = _metrics(y.iloc[random_test_positions], random_prediction)

    later_train_positions = np.flatnonzero(later_train_mask)
    later_test_positions = np.flatnonzero(later_test_mask)
    later_columns = _select_features(
        X.iloc[later_train_positions],
        y.iloc[later_train_positions],
        family=family,
        top_k=top_k,
    )
    later_model = _train_model(
        X.iloc[later_train_positions],
        y.iloc[later_train_positions],
        later_columns,
        depth=depth,
        rounds=rounds,
        seed=seed,
    )
    later_prediction = _predict_model(
        later_model, X.iloc[later_test_positions], later_columns
    )
    later_metrics = _metrics(y.iloc[later_test_positions], later_prediction)

    random_r2 = _float_or_none(random_metrics.get("r2"))
    later_r2 = _float_or_none(later_metrics.get("r2"))
    reasons: list[str] = []
    if random_r2 is None or random_r2 < float(min_random_r2):
        reasons.append(f"Random-split R2 {random_r2!s} is below {min_random_r2:.2f}.")
    if later_r2 is None or later_r2 < float(min_later_time_r2):
        reasons.append(f"Later-time R2 {later_r2!s} is below {min_later_time_r2:.2f}.")
    thresholds = {
        "random_r2": float(min_random_r2),
        "later_time_r2": float(min_later_time_r2),
    }
    dataset_sha = _dataset_hash(dataset_path)
    if reasons:
        return RetrainReport(
            passed=False,
            deployed=False,
            random_metrics=random_metrics,
            later_time_metrics=later_metrics,
            thresholds=thresholds,
            training_rows=int(len(origins)),
            feature_count=len(later_columns),
            first_origin_utc=str(origins.min()),
            last_origin_utc=str(origins.max()),
            dataset_sha256=dataset_sha,
            reasons=tuple(reasons),
        )

    final_cutoff = cleaned.index.max() + pd.Timedelta(seconds=1)
    final_mask = (origins < final_cutoff - pd.Timedelta(hours=purge_hours)) & (
        origins >= final_cutoff - pd.Timedelta(days=window_days)
    )
    final_positions = np.flatnonzero(final_mask)
    final_columns = _select_features(
        X.iloc[final_positions],
        y.iloc[final_positions],
        family=family,
        top_k=top_k,
    )
    final_model = _train_model(
        X.iloc[final_positions],
        y.iloc[final_positions],
        final_columns,
        depth=depth,
        rounds=rounds,
        seed=seed,
    )

    trained_at = datetime.now(timezone.utc)
    deployment_id = trained_at.strftime("%Y%m%dT%H%M%S%fZ")
    version_dir = deployment_dir / "versions" / deployment_id
    version_dir.mkdir(parents=True, exist_ok=False)
    final_model.save_model(str(version_dir / "coke_context.json"))
    feature_limits: dict[str, list[float]] = {}
    final_training = X.iloc[final_positions]
    for column in final_columns:
        quantiles = final_training[column].quantile([0.01, 0.99])
        if quantiles.notna().all():
            feature_limits[column] = [
                float(quantiles.iloc[0]),
                float(quantiles.iloc[1]),
            ]

    schema = {
        **base_schema,
        "features": final_columns,
        "feature_limits": feature_limits,
        "fit_cutoff": str(final_cutoff),
        "latest_label_available": str(origins[final_mask].max()),
        "training_rows": int(final_mask.sum()),
        "parameters": {**recipe, "rounds": rounds, "rolling": False},
        "validation": {
            "random_split": random_metrics,
            "later_time": {
                **later_metrics,
                "test_start": str(later_start),
                "test_end": str(last_origin),
            },
        },
        "deployment_gate_passed": True,
        # This remains false: predictive validation is not causal optimisation
        # validation.  BMO uses the model only as a frozen current-state anchor.
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
        "thresholds": thresholds,
        "training_rows": int(final_mask.sum()),
        "features": len(final_columns),
        "xgboost_version": xgb.__version__,
    }
    _write_json(version_dir / "deployment_metadata.json", metadata)

    deployment_dir.mkdir(parents=True, exist_ok=True)
    pointer_tmp = deployment_dir / "active.tmp.json"
    _write_json(pointer_tmp, {"deployment_id": deployment_id})
    os.replace(pointer_tmp, deployment_dir / "active.json")

    return RetrainReport(
        passed=True,
        deployed=True,
        deployment_id=deployment_id,
        random_metrics=random_metrics,
        later_time_metrics=later_metrics,
        thresholds=thresholds,
        training_rows=int(final_mask.sum()),
        feature_count=len(final_columns),
        first_origin_utc=str(origins[final_mask].min()),
        last_origin_utc=str(origins[final_mask].max()),
        dataset_sha256=dataset_sha,
        deployed_path=str(version_dir),
    )
