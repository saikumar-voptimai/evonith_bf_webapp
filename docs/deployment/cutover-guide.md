# API Cutover Guide

## Goal

Move the frontend to backend API mode while keeping direct-mode rollback
available.

## Full API Mode

```bash
export USE_BACKEND_API=true
export USE_BACKEND_API_AUTH=true
export USE_BACKEND_API_ADMIN=true
export USE_BACKEND_API_DATA_EXPLORER=true
export USE_BACKEND_API_DATASETS=true
export USE_BACKEND_API_FEEDBACK=true
export USE_BACKEND_API_MATERIAL_BALANCE=true
export USE_BACKEND_API_RECOMMENDATIONS=true
export USE_BACKEND_API_BLEND_OPTIMIZER=true
export USE_BACKEND_API_COPILOT=true
export USE_BACKEND_API_FURNACEMIND=true
export USE_BACKEND_API_OPS=true
```

Validate:

```bash
python scripts/validate_api_cutover.py --strict --backend-url http://localhost:8080/api/v1
```

## Staged Cutover

Enable one page flag at a time, run frontend regression tests, and keep
`EVONITH_ALLOW_DIRECT_MODE_FALLBACK=true`.

## Rollback

Set `USE_BACKEND_API=false` or disable a page-specific flag, then restart the
frontend. Backend can remain running.

