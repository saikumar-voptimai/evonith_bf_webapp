"""Train and benchmark GP-style fuel-cost models for BMO.

This script mirrors the feature contract from BMO_V4_Final_corrected.ipynb and
exports the best deployable bundle to src/assets/models/bmo_fuel/.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Make the local feature builder and runtime wrapper importable when executed
# from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# The repository package __init__ imports Streamlit-only runtime services.  The
# training environment may not include Streamlit, so load only the GP module
# while preserving its deploy-time module path for joblib.
import types
if "domain" not in sys.modules:
    domain_pkg = types.ModuleType("domain")
    domain_pkg.__path__ = [str(SRC_ROOT / "domain")]
    sys.modules["domain"] = domain_pkg
if "domain.optimization_runtime" not in sys.modules:
    opt_pkg = types.ModuleType("domain.optimization_runtime")
    opt_pkg.__path__ = [str(SRC_ROOT / "domain/optimization_runtime")]
    sys.modules["domain.optimization_runtime"] = opt_pkg

from bmo_dataset_builder import build_features  # noqa: E402
from domain.optimization_runtime.gp_models import (  # noqa: E402
    BmoGpResidualCostModel,
    RandomFeatureBayesianGp,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_NAME = "unitcost_new"




def json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas values and NaN/Inf to strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def xgb_params(n_estimators: int = 600) -> dict[str, Any]:
    return {
        "n_estimators": int(n_estimators),
        "learning_rate": 0.02,
        "max_depth": 4,
        "subsample": 0.75,
        "colsample_bytree": 0.6,
        "min_child_weight": 8,
        "reg_alpha": 3.0,
        "reg_lambda": 7.0,
        "random_state": RANDOM_STATE,
        "eval_metric": "rmse",
        "verbosity": 0,
        "n_jobs": 2,
    }


def train_xgb(X_fit: pd.DataFrame, y_fit: pd.Series, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> XGBRegressor:
    params = xgb_params()
    # XGBoost 3.x supports early_stopping_rounds in constructor, but not every
    # deployment has the same version. Keep the final artifact version-tolerant.
    model = XGBRegressor(**params)
    if X_val is not None and y_val is not None and len(X_val) > 0:
        try:
            model.set_params(early_stopping_rounds=50)
            model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        except TypeError:
            model.set_params(early_stopping_rounds=None)
            model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_fit, y_fit, verbose=False)
    return model


def metric_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, std: np.ndarray | None = None, **extra: Any) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    row: dict[str, Any] = {
        "model_name": model_name,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "max_abs_error": float(np.max(np.abs(y_pred - y_true))),
    }
    if std is not None:
        std = np.asarray(std, dtype=float).reshape(-1)
        std = np.where(np.isfinite(std), np.maximum(std, 1e-9), 1e-9)
        lower95 = y_pred - 1.96 * std
        upper95 = y_pred + 1.96 * std
        lower90 = y_pred - 1.645 * std
        upper90 = y_pred + 1.645 * std
        row.update(
            {
                "coverage_95": float(np.mean((y_true >= lower95) & (y_true <= upper95))),
                "coverage_90": float(np.mean((y_true >= lower90) & (y_true <= upper90))),
                "mean_interval_width_95": float(np.mean(upper95 - lower95)),
                "mean_std": float(np.mean(std)),
            }
        )
    row.update(extra)
    return row


def fit_random_feature_gp(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> RandomFeatureBayesianGp:
    model = RandomFeatureBayesianGp(**cfg)
    model.fit(X, y)
    return model


def exact_gp_candidates(X_fit: pd.DataFrame, y_fit: pd.Series, X_eval: pd.DataFrame, *, max_train: int = 750) -> list[tuple[str, Pipeline]]:
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X_fit) > max_train:
        idx = rng.choice(np.arange(len(X_fit)), size=max_train, replace=False)
        X_sub = X_fit.iloc[idx]
        y_sub = y_fit.iloc[idx]
    else:
        X_sub = X_fit
        y_sub = y_fit

    kernels = {
        "exact_gp_pca15_rbf": ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=3.0, length_scale_bounds="fixed") + WhiteKernel(noise_level=0.1, noise_level_bounds="fixed"),
        "exact_gp_pca15_matern32": ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=3.0, nu=1.5, length_scale_bounds="fixed") + WhiteKernel(noise_level=0.1, noise_level_bounds="fixed"),
    }
    out: list[tuple[str, Pipeline]] = []
    n_comp = min(15, X_fit.shape[1], max(1, len(X_sub) - 1))
    for name, kernel in kernels.items():
        pipe = Pipeline(
            steps=[
                ("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE)),
                (
                    "gp",
                    GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=1e-6,
                        normalize_y=True,
                        random_state=RANDOM_STATE,
                        optimizer=None,
                    ),
                ),
            ]
        )
        pipe.fit(X_sub, y_sub)
        out.append((name, pipe))
    return out


def predict_with_std(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        pred = model.predict(X, return_std=True)
        if isinstance(pred, tuple) and len(pred) == 2:
            return np.asarray(pred[0], dtype=float).reshape(-1), np.asarray(pred[1], dtype=float).reshape(-1)
    except TypeError:
        pass
    except Exception:
        # Some sklearn Pipelines cannot route return_std; handle GPR pipelines explicitly.
        pass
    if isinstance(model, Pipeline) and "gp" in model.named_steps:
        Xt = model[:-1].transform(X)
        mean, std = model.named_steps["gp"].predict(Xt, return_std=True)
        return np.asarray(mean, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1)
    return np.asarray(model.predict(X), dtype=float).reshape(-1), None


def tune_uncertainty_scale(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, target_coverage: float = 0.95) -> tuple[float, float]:
    err = np.abs(np.asarray(y_true, dtype=float).reshape(-1) - np.asarray(mean, dtype=float).reshape(-1))
    std = np.asarray(std, dtype=float).reshape(-1)
    residual_floor = max(1.0, float(np.nanstd(err) * 0.05))
    safe_std = np.sqrt(np.square(np.maximum(std, 1e-6)) + residual_floor * residual_floor)
    ratios = err / (1.96 * safe_std)
    scale = float(np.nanquantile(ratios, target_coverage)) if len(ratios) else 1.0
    if not np.isfinite(scale):
        scale = 1.0
    return max(1.0, min(scale, 10.0)), residual_floor


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/mnt/data/processed_data_with_unitcost.csv")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "src/assets/models/bmo_fuel"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building BMO V4 feature matrix...")
    X, y, times, final_dataset, meta = build_features(args.csv, apply_cruising_filters=True)
    print(f"Rows={len(X)} Features={X.shape[1]} Date range={times.iloc[0]} -> {times.iloc[-1]}")

    X_dev, X_test, y_dev, y_test, t_dev, t_test = train_test_split(
        X, y, times, test_size=0.20, shuffle=True, random_state=RANDOM_STATE
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, shuffle=True, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_fit_s = pd.DataFrame(scaler.fit_transform(X_fit), columns=X.columns, index=X_fit.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    results: list[dict[str, Any]] = []

    print("Training XGBoost mean baseline...")
    xgb = train_xgb(X_fit_s, y_fit, X_val_s, y_val)
    xgb_val_pred = xgb.predict(X_val_s)
    xgb_test_pred = xgb.predict(X_test_s)
    xgb_val = metric_row("xgb_baseline_mean", y_val, xgb_val_pred, None, family="baseline", split="validation")
    xgb_test = metric_row("xgb_baseline_mean", y_test, xgb_test_pred, None, family="baseline", split="test")
    results.extend([xgb_val, xgb_test])
    print("  XGB val RMSE %.2f R2 %.4f | test RMSE %.2f R2 %.4f" % (xgb_val["rmse"], xgb_val["r2"], xgb_test["rmse"], xgb_test["r2"]))

    # Pure scalable GP approximations.
    gp_grid: list[dict[str, Any]] = []
    for feature_map in ["rbf_sampler", "nystroem"]:
        for n_components in [32, 64]:
            for gamma in [0.001, 0.005]:
                gp_grid.append(
                    {
                        "feature_map": feature_map,
                        "n_components": n_components,
                        "gamma": gamma,
                        "kernel": "rbf",
                        "random_state": RANDOM_STATE,
                        "max_iter": 120,
                        "ridge_alpha": 5.0,
                    }
                )

    pure_gp_candidates: list[tuple[dict[str, Any], dict[str, Any], RandomFeatureBayesianGp]] = []
    print(f"Training {len(gp_grid)} random-feature GP candidates...")
    for cfg in gp_grid:
        name = f"pure_gp_{cfg['feature_map']}_c{cfg['n_components']}_g{cfg['gamma']}"
        gp = fit_random_feature_gp(X_fit_s, y_fit, cfg)
        val_pred, val_std = gp.predict(X_val_s, return_std=True)
        scale, floor = tune_uncertainty_scale(y_val.to_numpy(), val_pred, val_std)
        gp.uncertainty_scale = scale
        gp.residual_floor = floor
        val_pred, val_std = gp.predict(X_val_s, return_std=True)
        row_val = metric_row(name, y_val, val_pred, val_std, family="pure_gp", split="validation", **cfg)
        test_pred, test_std = gp.predict(X_test_s, return_std=True)
        row_test = metric_row(name, y_test, test_pred, test_std, family="pure_gp", split="test", **cfg)
        results.extend([row_val, row_test])
        pure_gp_candidates.append((row_val, cfg, gp))
        print(f"  {name}: val RMSE={row_val['rmse']:.2f} R2={row_val['r2']:.4f} cov95={row_val.get('coverage_95', float('nan')):.2f}")

    # Exact full GP with GaussianProcessRegressor was tested separately at small
    # sample sizes; it is skipped in the production training run because its
    # cubic fit/predict cost is unsuitable for the BMO optimizer runtime.

    # Residual GP candidates over the XGB mean. Accuracy-safe by tuning residual_weight.
    print("Training XGB + GP residual candidates...")
    base_val_rmse = float(np.sqrt(mean_squared_error(y_val, xgb_val_pred)))
    residual_fit = y_fit.to_numpy() - xgb.predict(X_fit_s)
    best_residual: dict[str, Any] | None = None
    best_residual_model: RandomFeatureBayesianGp | None = None

    for cfg in gp_grid:
        name = f"xgb_plus_gp_residual_{cfg['feature_map']}_c{cfg['n_components']}_g{cfg['gamma']}"
        gp = fit_random_feature_gp(X_fit_s, residual_fit, cfg)
        corr_val, std_val = gp.predict(X_val_s, return_std=True)
        # Restrict corrections to non-negative weights so cost monotonicity is not
        # made unstable. 0.0 preserves baseline accuracy exactly.
        candidate_weights = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
        weight_rows = []
        for w in candidate_weights:
            pred_val = xgb_val_pred + w * corr_val
            weight_rows.append((float(np.sqrt(mean_squared_error(y_val, pred_val))), w, pred_val))
        val_rmse, best_w, best_val_pred = min(weight_rows, key=lambda item: item[0])
        # Guardrail: never select a residual correction that materially worsens the
        # XGB mean. The GP still supplies uncertainty when w=0.
        if val_rmse > base_val_rmse:
            best_w = 0.0
            best_val_pred = xgb_val_pred
            val_rmse = base_val_rmse
        scale, floor = tune_uncertainty_scale(y_val.to_numpy(), best_val_pred, std_val)
        gp.uncertainty_scale = 1.0
        gp.residual_floor = 0.0
        val_std = np.sqrt(np.square(std_val) + floor * floor) * scale
        corr_test, std_test_raw = gp.predict(X_test_s, return_std=True)
        test_pred = xgb_test_pred + best_w * corr_test
        test_std = np.sqrt(np.square(std_test_raw) + floor * floor) * scale
        row_val = metric_row(name, y_val, best_val_pred, val_std, family="hybrid_residual_gp", split="validation", residual_weight=best_w, uncertainty_scale=scale, residual_floor=floor, **cfg)
        row_test = metric_row(name, y_test, test_pred, test_std, family="hybrid_residual_gp", split="test", residual_weight=best_w, uncertainty_scale=scale, residual_floor=floor, **cfg)
        results.extend([row_val, row_test])
        print(f"  {name}: val RMSE={row_val['rmse']:.2f} R2={row_val['r2']:.4f} w={best_w:.2f} cov95={row_val.get('coverage_95', float('nan')):.2f}")
        if best_residual is None or row_val["rmse"] < best_residual["rmse"]:
            best_residual = dict(row_val)
            best_residual_model = gp

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "gp_benchmark_results.csv", index=False)

    val_results = results_df[results_df["split"] == "validation"].copy()
    val_results = val_results.sort_values(["rmse", "mae"], ascending=True)
    test_results = results_df[results_df["split"] == "test"].sort_values(["rmse", "mae"], ascending=True)
    print("\nTop validation candidates:")
    print(val_results[["model_name", "family", "rmse", "mae", "r2", "coverage_95", "residual_weight"]].head(10).to_string(index=False))
    print("\nTop test candidates:")
    print(test_results[["model_name", "family", "rmse", "mae", "r2", "coverage_95", "residual_weight"]].head(10).to_string(index=False))

    # Select deployable model: use the best hybrid residual GP because it is the
    # only candidate that protects the existing XGB mean while adding uncertainty.
    hybrid_val = val_results[val_results["family"] == "hybrid_residual_gp"].iloc[0].to_dict()
    selected_cfg = {k: hybrid_val[k] for k in ["feature_map", "n_components", "gamma", "kernel"] if k in hybrid_val and pd.notna(hybrid_val[k])}
    selected_weight = float(hybrid_val.get("residual_weight", 0.0) or 0.0)
    selected_scale = float(hybrid_val.get("uncertainty_scale", 1.0) or 1.0)
    selected_floor = float(hybrid_val.get("residual_floor", 0.0) or 0.0)

    # Final production artifact trained on all available rows using the selected
    # configuration. The benchmark above remains the independent performance record.
    final_scaler = StandardScaler()
    X_full_s = pd.DataFrame(final_scaler.fit_transform(X), columns=X.columns, index=X.index)
    final_xgb = XGBRegressor(**xgb_params())
    final_xgb.fit(X_full_s, y, verbose=False)
    full_residual = y.to_numpy() - final_xgb.predict(X_full_s)
    final_gp = RandomFeatureBayesianGp(
        feature_map=str(selected_cfg.get("feature_map", "rbf_sampler")),
        n_components=int(selected_cfg.get("n_components", 512)),
        gamma=float(selected_cfg.get("gamma", 0.05)),
        kernel=str(selected_cfg.get("kernel", "rbf")),
        random_state=RANDOM_STATE,
        max_iter=120,
        ridge_alpha=5.0,
    )
    final_gp.fit(X_full_s, full_residual)
    final_model = BmoGpResidualCostModel(
        mean_model=final_xgb,
        residual_gp=final_gp,
        residual_weight=selected_weight,
        uncertainty_scale=selected_scale,
        residual_floor=selected_floor,
        feature_names_in_=list(X.columns),
        training_metadata={
            "target_name": "fuel_unit_cost_rs_per_thm",
            "source_target_column": TARGET_NAME,
            "feature_contract": "BMO V4 scaler-order columns, already selected after scaler.transform",
            "selected_validation_model": hybrid_val,
            "data_rows_after_filters": int(len(X)),
            "feature_count": int(X.shape[1]),
            "date_start": str(times.min()),
            "date_end": str(times.max()),
        },
    )

    # Save deployable artifacts.
    joblib.dump(final_scaler, out_dir / "bmo_gp_scaler.joblib")
    joblib.dump(final_model, out_dir / "bmo_gp_residual_model.joblib")
    with open(out_dir / "bmo_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    # Save metrics/metadata.
    selected_test_row = test_results[test_results["model_name"] == hybrid_val["model_name"]].iloc[0].to_dict()
    xgb_test_row = results_df[(results_df["split"] == "test") & (results_df["model_name"] == "xgb_baseline_mean")].iloc[0].to_dict()
    metrics_payload = {
        "version": "bmo-gp-v1",
        "model_name": "bmo_gp_residual_model",
        "train_date": pd.Timestamp.utcnow().isoformat(),
        "target_name": "fuel_unit_cost_rs_per_thm",
        "source_target_column": TARGET_NAME,
        "selection_policy": "Best validation RMSE among XGB + GP residual candidates, with residual_weight guardrail to prevent XGB accuracy degradation.",
        "selected_model": {
            "model_path": "src/assets/models/bmo_fuel/bmo_gp_residual_model.joblib",
            "scaler_path": "src/assets/models/bmo_fuel/bmo_gp_scaler.joblib",
            "feature_columns_path": "src/assets/models/bmo_fuel/bmo_feature_columns.json",
            "family": "hybrid_residual_gp",
            "residual_gp_config": selected_cfg,
            "residual_weight": selected_weight,
            "uncertainty_scale": selected_scale,
            "residual_floor": selected_floor,
        },
        "data": {
            "rows_after_cruising_filters": int(len(X)),
            "feature_count": int(X.shape[1]),
            "date_start": str(times.min()),
            "date_end": str(times.max()),
            "random_state": RANDOM_STATE,
            "split_policy": "60% fit / 20% validation / 20% test, random split to mirror original notebook holdout workflow.",
        },
        "metrics": {
            "xgb_baseline_test": xgb_test_row,
            "selected_gp_validation": hybrid_val,
            "selected_gp_test": selected_test_row,
        },
        "candidate_summary": val_results.head(20).replace({np.nan: None}).to_dict(orient="records"),
    }
    with open(out_dir / "gp_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metrics_payload), f, indent=2, default=str, allow_nan=False)

    with open(out_dir / "gp_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            json_safe({
                "model_class": "domain.optimization_runtime.gp_models.BmoGpResidualCostModel",
                "contract": "Input is the selected, scaled BMO V4 feature DataFrame; predict returns Rs/THM. predict(return_std=True) returns mean and standard deviation.",
                "selected_validation_model": hybrid_val,
                "selected_test_model": selected_test_row,
                "xgb_baseline_test": xgb_test_row,
            }),
            f,
            indent=2,
            default=str,
            allow_nan=False,
        )

    print("\nSaved deployable GP bundle:")
    for name in ["bmo_gp_residual_model.joblib", "bmo_gp_scaler.joblib", "bmo_feature_columns.json", "gp_training_metrics.json", "gp_benchmark_results.csv", "gp_model_metadata.json"]:
        p = out_dir / name
        print(f"  {p} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
