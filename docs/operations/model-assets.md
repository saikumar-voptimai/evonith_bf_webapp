# Model Assets

## Canonical Locations

Active bundled model assets live in:

```text
packages/furnace-data/furnace_data/assets/models/
```

The backend model registry and direct-mode model bundle loaders use this
directory by default. Deployments that keep models outside the repo can set:

```bash
EVONITH_MODEL_DIR=/path/to/active/models
```

When `EVONITH_MODEL_DIR` is set, old-style compatibility paths such as
`src/assets/models/unitcost_fuel_model.json` resolve against that configured
model directory first.

## Active Bundles

- Root-level model/scaler/metadata files are active bundled assets.
- `bmo_fuel/` is the only nested active model bundle discovered by the backend
  model registry.
- Missing optional model files should degrade to structured model status or API
  errors. They must not fail backend startup.
- Model loading must remain lazy. Importing the backend or listing models should
  not load model artifacts into memory.

## Archives

Old archive folders are not production source:

- `src/assets/models/old_26_14/`
- `src/assets/models/old_bmo_12062026/`

Keep historical archives in an external artifact store, release archive, or
backup location with date/version metadata. Restore only the specific required
files into the active canonical model directory or `EVONITH_MODEL_DIR`.

## Runtime Data

Generated datasets and mutable operator overrides are runtime files, not source
assets:

- `runtime/datasets/static/`
- `runtime/cache/control_bounds.json`

Only `runtime/.gitkeep` should be tracked.
