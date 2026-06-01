# BMO Fuel Model Bundle

The BMO V4 fuel-cost model uses a two-stage inference contract:

- build the full raw feature row expected by `bmo_scaler.joblib`
- scale that full row
- select the 184 columns listed in `bmo_feature_columns.json`
- pass those scaled columns to `bmo_xgb_model.json`

Required V4 artifacts:

- `bmo_xgb_model.json`
- `bmo_scaler.joblib`
- `bmo_feature_columns.json`

Optional metadata artifacts:

- `feature_manifest.json`
- `lag_map.json`
- `training_metrics.json`

The webapp runs in fallback mode when model/scaler/feature-selection artifacts are
not present or cannot be used.
