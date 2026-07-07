# Local And Staging Deployment Guide

## Local

```bash
export EVONITH_RUNTIME_DIR=./runtime
export EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me
python scripts/bootstrap_runtime.py --create
uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080
BACKEND_API_BASE_URL=http://localhost:8080/api/v1 streamlit run apps/frontend_streamlit/app.py
```

The canonical Streamlit entrypoint is `apps/frontend_streamlit/app.py`, and the
canonical page files live under `apps/frontend_streamlit/custom_pages`.

## Staging

Staging should mirror production paths and API-mode flags while using staging
secrets and isolated runtime storage.

```bash
python scripts/validate_deployment.py --profile staging --offline --strict
python scripts/validate_api_cutover.py --strict
python scripts/smoke_test_deployment.py --backend-url http://localhost:8080/api/v1 --skip-auth
```



