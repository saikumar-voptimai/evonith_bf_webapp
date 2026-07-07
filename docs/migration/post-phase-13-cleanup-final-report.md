# Post-Phase 13 Cleanup Final Report

## Production Readiness Status

The cleaned target structure is production-ready pending the final validation command results recorded in the current cleanup task.

## Final Structure

```text
apps/backend_api/app
apps/frontend_streamlit
packages/furnace-data/furnace_data
infra
scripts
docs
tests
runtime
```

## Cleanup Summary

- Canonical backend implementation lives only under `apps/backend_api/app`.
- The old backend sidecar has been removed from active source.
- Canonical frontend app, pages, services, config, UI helpers, assets, agents, data, domain, reports, plotters, geometries, and utilities live under `apps/frontend_streamlit`.
- The shared package lives under `packages/furnace-data/furnace_data` and is imported as `furnace_data`.
- Runtime/generated files belong under `runtime` or the configured `EVONITH_RUNTIME_DIR`.
- Alembic leftovers and obsolete generated artifacts were removed earlier in the cleanup series.
- Duplicate model archives were removed or excluded from active model discovery earlier in the cleanup series.
- Dependency profiles keep backend-base free of frontend and heavy optional packages.

## Compatibility Shims Retained`r`n`r`n- No legacy frontend source-folder shims are retained.`r`n
## Known Remaining Technical Debt

- Continue monitoring direct-mode fallback through canonical frontend/package modules.

## Recommended Future Cleanup

1. Keep removed legacy frontend paths out of active docs, tests, and scripts.
2. Keep editable/package installs standard in every deployment profile so `import furnace_data` continues to resolve from the canonical package.
3. Keep structure, import-boundary, dependency, and release-readiness checks in CI.

## Validation

Run the final validation set from the cleanup task before release. Any failure should be fixed if small and within cleanup scope, otherwise documented as a blocking follow-up.

