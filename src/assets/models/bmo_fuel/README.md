# BMO Fuel Model Bundle

## Active deployment bundle

The active BMO fuel/unit-cost bundle is the GP residual model:

- `bmo_gp_residual_model.joblib` — XGBoost mean predictor plus calibrated GP residual/uncertainty layer.
- `bmo_gp_scaler.joblib` — `StandardScaler` fitted on the BMO V4 training feature matrix.
- `bmo_feature_columns.json` — 252 selected, scaled BMO V4 model features in inference order.
- `gp_training_metrics.json` — benchmark metrics, selected model metadata, and candidate summary.
- `gp_benchmark_results.csv` — validation/test metrics for all tested GP-style candidates.
- `gp_model_metadata.json` — compact runtime metadata for the selected GP model.

The inference contract remains compatible with the existing BMO page:

1. Build the full raw BMO V4 feature row.
2. Scale the row with `bmo_gp_scaler.joblib`.
3. Select the 252 columns listed in `bmo_feature_columns.json`.
4. Call `predict(X)` on `bmo_gp_residual_model.joblib` for Rs/THM unit fuel cost.
5. Where supported, call `predict(X, return_std=True)` or `predict_with_uncertainty(X)` to obtain prediction standard deviation and a 95% interval.

## Legacy deterministic-mean bundle

The previous XGBoost-only assets are retained for rollback/reference:

- `bmo_xgb_model.json`
- `bmo_scaler.joblib`
- `training_metrics.json`

To roll back, update `src/config/setting_bmo.yml` model-bundle paths back to the legacy model/scaler/training metrics files.

## Fallback behavior

The webapp continues to use the deterministic fallback fuel formula when the model, scaler, or selected feature artifacts are unavailable or inference fails.
