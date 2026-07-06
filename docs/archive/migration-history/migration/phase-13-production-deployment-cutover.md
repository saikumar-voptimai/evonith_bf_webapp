# Phase 13 Production Deployment Cutover

## Purpose

Phase 13 prepares safe deployment and staged API-mode cutover for the separated
FastAPI backend and Streamlit frontend created in Phase 12. This phase adds
deployment assets, validation scripts, smoke tests, backup/restore tooling,
rollback documentation, and release gates. It does not remove direct-mode
fallbacks, compatibility shims, or legacy backend routes.

## Audit

| Area | Current State | Phase 13 Action |
|---|---|---|
| Canonical backend command | `uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080` | Kept and used by edge scripts/systemd docs |
| Canonical frontend command | `streamlit run apps/frontend_streamlit/app.py` | Kept and used by edge scripts/systemd docs |
| Compatibility backend command | `cd furnace-data-service && uvicorn app.main:app --host 0.0.0.0 --port 8080` | Kept for rollback/compatibility |
| Compatibility frontend command | `streamlit run src/app.py` | Kept for rollback/compatibility |
| Runtime directory behavior | `EVONITH_RUNTIME_DIR`, default `./runtime` | Bootstrap/backup/restore validate and operate only on runtime |
| Edge startup scripts | Existing scripts used canonical paths | Added `DRY_RUN=1`, deployment profile, safer summaries |
| Systemd examples | Existing backend/frontend service examples | Retained scripts and added safe hardening |
| Docker/Compose examples | None | Deferred; non-Docker deployment remains supported |
| Environment examples | `.env.example` plus Phase 11 comments | Added `infra/env/*.example` and Phase 13 env variables |
| Backup/restore behavior | No deployment backup scripts | Added non-destructive backup/restore scripts |
| Deployment validation behavior | Phase 11 dependency/import checks | Added deployment validation and release readiness scripts |
| Smoke test coverage | Existing pytest/API tests | Added running-deployment smoke script |
| Cutover flag behavior | Page-level API flags already exist | Added cutover validator and docs |
| Rollback path | Turn flags off and use old commands | Documented direct-mode rollback and compatibility commands |
| Deployment security risks | Secrets could be accidentally placed in env files | Templates use placeholders; validation flags production placeholders |
| Pi/Jetson risks | Low disk, thread oversubscription, optional deps | Edge env/script defaults constrain threads and optional features |

## Implemented

- `infra/env/backend.env.example`, `frontend.env.example`, and `edge.env.example`
- `infra/nginx/evonith.conf.example` and `infra/caddy/Caddyfile.example`
- `scripts/bootstrap_runtime.py`
- `scripts/validate_deployment.py`
- `scripts/smoke_test_deployment.py`
- `scripts/validate_api_cutover.py`
- `scripts/backup_runtime.py`
- `scripts/restore_runtime.py`
- `scripts/verify_release_readiness.py`
- Deployment guides under `docs/deployment/`
- Deployment tests under `tests/deployment/`

## Intentionally Deferred

- Docker image and Compose production templates; direct/systemd deployment is
  the supported low-risk path for this phase.
- Removal of direct-mode fallbacks, compatibility shims, or legacy routes.
- Production secret-manager integration.
- Real systemd start/stop automation in tests.

## Safety Notes

- Backup and restore operate on runtime data, not source code.
- Restore is dry-run unless `--apply` is used.
- Validation scripts redact or avoid secrets.
- Production-like profiles fail validation when `EVONITH_AUTH_SECRET_KEY` is
  blank or a placeholder.

