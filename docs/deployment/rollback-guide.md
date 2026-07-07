# Rollback Guide

## Fast Rollback

1. Set `USE_BACKEND_API=false`.
2. Keep `EVONITH_ALLOW_DIRECT_MODE_FALLBACK=true`.
3. Restart the frontend service.
4. Verify direct-mode pages still load.

## Canonical Startup

Use uv run streamlit run apps/frontend_streamlit/app.py for frontend rollback validation.

## Data Safety

Rollback does not delete runtime data. Do not run restore or cleanup without a
dry-run review.

## Validation

```bash
uv run python scripts/validate_deployment.py --profile local --offline
uv run python scripts/check_repository_structure.py
```





