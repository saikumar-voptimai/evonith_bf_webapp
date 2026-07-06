# Rollback Guide

## Fast Rollback

1. Set `USE_BACKEND_API=false`.
2. Keep `EVONITH_ALLOW_DIRECT_MODE_FALLBACK=true`.
3. Restart the frontend service.
4. Verify direct-mode pages still load.

## Compatibility Startup

Backend compatibility:

```bash
cd furnace-data-service
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Frontend compatibility:

```bash
uv run streamlit run src/app.py
```

## Data Safety

Rollback does not delete runtime data. Do not run restore or cleanup without a
dry-run review.

## Validation

```bash
python scripts/validate_deployment.py --profile local --offline
python scripts/check_repository_structure.py
```
