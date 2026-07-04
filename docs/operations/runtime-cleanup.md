# Runtime Cleanup

Phase 10 runtime cleanup prevents unbounded growth of generated runtime files on
edge devices.

Cleanup is controlled by:

- `EVONITH_CLEANUP_ENABLED`
- `EVONITH_CLEANUP_DRY_RUN_DEFAULT`
- `EVONITH_CLEANUP_REQUIRE_ADMIN`
- `EVONITH_CLEANUP_MAX_DELETE_PER_RUN`
- `EVONITH_CLEANUP_INCLUDE_LOGS`
- `EVONITH_CLEANUP_INCLUDE_UPLOADS`
- `EVONITH_CLEANUP_JOB_TTL_HOURS`
- `EVONITH_CLEANUP_ARTIFACT_TTL_HOURS`
- `EVONITH_CLEANUP_TEMP_TTL_HOURS`

## Endpoints

- `POST /api/v1/ops/cleanup/dry-run`
- `POST /api/v1/ops/cleanup/run`

Both endpoints are admin-protected. Dry-run is non-destructive and reports which
runtime-relative files would be removed.

## Safety Rules

- Every candidate path is resolved and checked under `EVONITH_RUNTIME_DIR`.
- Symlinks are skipped.
- `.gitkeep` files are skipped.
- Deletes are capped by `EVONITH_CLEANUP_MAX_DELETE_PER_RUN`.
- Logs and uploads are excluded unless explicitly enabled.
- Responses expose runtime-relative labels, not absolute paths.

Default cleanup targets:

- `runtime/temp`
- `runtime/jobs`
- `runtime/compute/artifacts`
- `runtime/datasets/results/artifacts`

