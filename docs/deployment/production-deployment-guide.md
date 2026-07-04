# Production Deployment Guide

## Topology

Run two independent services:

- Backend: `apps.backend_api.app.main:app` on port `8080`
- Frontend: `apps/frontend_streamlit/app.py` on port `8501`

Runtime data must live outside source code, for example
`EVONITH_RUNTIME_DIR=/var/lib/evonith-bf`.

## Environment

Use `infra/env/backend.env.example` and `infra/env/frontend.env.example` as
templates. Populate real secrets only in protected deployment env files.

Required production changes:

- Set `EVONITH_AUTH_SECRET_KEY` to a strong secret.
- Set `EVONITH_RUNTIME_DIR` to persistent storage.
- Set `BACKEND_API_BASE_URL` for the frontend.
- Keep optional AI/vector keys blank unless those features are enabled.

## Bootstrap

```bash
python scripts/bootstrap_runtime.py --create
python scripts/validate_deployment.py --profile production --offline --strict
```

## Start Commands

Backend:

```bash
EVONITH_RUNTIME_DIR=/var/lib/evonith-bf \
EVONITH_AUTH_SECRET_KEY=<set-a-strong-random-secret> \
uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

Frontend:

```bash
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 \
streamlit run apps/frontend_streamlit/app.py --server.address 0.0.0.0 --server.port 8501
```

Systemd examples live in `infra/systemd/`. Reverse proxy examples are optional
and live in `infra/nginx/` and `infra/caddy/`.

## Validation

```bash
python scripts/validate_deployment.py --profile production --offline --strict
python scripts/validate_api_cutover.py --strict
python scripts/smoke_test_deployment.py --backend-url http://localhost:8080/api/v1 --skip-auth
```

## Operations

- Use journald or configured runtime logs.
- Use `/api/v1/health` and `/api/v1/readiness` for monitoring.
- Run cleanup with dry-run defaults.
- Back up runtime data before upgrades.

