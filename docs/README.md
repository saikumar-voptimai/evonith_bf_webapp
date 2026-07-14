# Evonith BF Documentation

This directory is split into active operating docs and archived migration history. Use the active guides below for current commands and canonical paths.

## Start Here

- [Repository cleanup plan](migration/post-phase-13-structure-cleanup-plan.md) - current restructure map, compatibility surfaces, risks, and validation history.
- [Testing guide](testing/phase-13-testing-guide.md) - current regression, deployment, release-readiness, and manual verification commands.
- [OpenAPI v1 export](api/openapi-v1.json) - generated backend API contract.

## Deployment

- [Production deployment guide](deployment/production-deployment-guide.md)
- [Local install guide](deployment/local-install-guide.md)
- [Local and staging deployment guide](deployment/local-staging-deployment-guide.md)
- [Edge device deployment guide](deployment/edge-device-deployment-guide.md)
- [Jetson backend with hosted Streamlit](deployment/jetson-streamlit-deployment-guide.md)
- [Edge runtime guide](deployment/edge-runtime-guide.md)
- [Dependency profiles](deployment/dependency-profiles.md)
- [Environment variables](deployment/environment-variables-production.md)
- [Release checklist](deployment/release-checklist.md)
- [Cutover guide](deployment/cutover-guide.md)
- [Rollback guide](deployment/rollback-guide.md)
- [Backup and restore guide](deployment/backup-restore-guide.md)

## Operations

- [Jetson backend operations runbook](operations/jetson-backend-runbook.md)
- [Runtime cleanup](operations/runtime-cleanup.md)
- [Model assets](operations/model-assets.md)
- [Logging and audit](operations/logging-and-audit.md)
- [Error codes](operations/error-codes.md)

## Architecture And Structure

Canonical paths:

- Backend API: `apps/backend_api/app`
- Streamlit frontend: `apps/frontend_streamlit`
- Shared package: `packages/furnace-data/furnace_data`
- Runtime data: `runtime` or the directory selected by `EVONITH_RUNTIME_DIR`

The active restructure reference is [post-phase-13 structure cleanup plan](migration/post-phase-13-structure-cleanup-plan.md).

## Migration Archive

Detailed phase-by-phase implementation notes and old test reports live under [archive/migration-history](archive/migration-history/). They are retained for history, but active deployment and testing commands should come from this index and the current guides above.
