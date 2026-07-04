# Error Codes

Phase 10 adds a safe error-code registry exposed through
`/api/v1/ops/error-codes` for admin users.

The structured error envelope remains:

```json
{
  "request_id": "...",
  "error": {
    "code": "AUTH_REQUIRED",
    "message": "Authentication is required.",
    "details": {}
  }
}
```

Stack traces, secrets, raw tokens, provider keys, prompts, documents, runtime
absolute paths, and database URLs must not be exposed in error responses.

## Families

| Family | Typical Status | Meaning | Remediation |
|---|---:|---|---|
| `AUTH_*` | 401 | Authentication or token failure | Login again, verify auth secret and token expiry |
| `ADMIN_*` | 403/404 | Admin user-management issue | Use admin role and verify user id |
| `DATA_*` | 400/422 | Data API source/query/export issue | Check source id and request parameters |
| `DATASET_*` | 400/404 | Dataset preview/refresh/artifact issue | Check dataset id, job id, artifact id |
| `FEEDBACK_*` | 400/403/404 | Ticket/comment/attachment issue | Check ownership, status, file type, and id |
| `MATERIAL_BALANCE_*` | 422 | Material Balance validation/run issue | Validate payload and required fields |
| `RECOMMENDATION_*` | 422 | Recommendations validation/run issue | Validate input rows and configuration |
| `BLEND_OPTIMIZER_*` | 422 | Blend optimizer validation/run issue | Validate optimizer payload and model settings |
| `MODEL_*` | 404/503 | Model registry/load/predict issue | Check model path, lazy-load settings, and optional artifacts |
| `COPILOT_*` | 400/409/503 | Copilot safety/provider/artifact issue | Check Copilot flags, provider config, and input caps |
| `FURNACEMIND_*` | 400/409/503 | FurnaceMind conversation/run/document/memory/tool issue | Check auth, runtime storage, provider/memory/tool flags |
| `OPS_*` | 403 | Operational endpoint access issue | Use an admin token |
| `RUNTIME_*` | 503 | Runtime directory/disk issue | Check `EVONITH_RUNTIME_DIR`, permissions, and free space |
| `DEPENDENCY_*` | 503 | Dependency check issue | Check optional dependency configuration |
| `JOB_*` | 404 | Unified job lookup issue | Verify job id and workflow |
| `AUDIT_*` | 500/503 | Audit storage/read issue | Check runtime audit DB and retention settings |
| `CLEANUP_*` | 409/403 | Cleanup disabled or not permitted | Enable cleanup and use admin token |
| `METRICS_*` | 403/404 | Metrics disabled or not permitted | Enable metrics and use admin token |

## Common Codes

- `AUTH_REQUIRED`: Login and retry with a bearer token.
- `FORBIDDEN`: Current user lacks the required role.
- `VALIDATION_ERROR`: Fix request body, path, or query parameters.
- `INTERNAL_SERVER_ERROR`: Use request ID to inspect redacted backend logs.
- `RUNTIME_NOT_READY`: Runtime directory is missing, unwritable, or low on disk.
- `JOB_NOT_FOUND`: The unified job registry has no matching job id.
- `AUDIT_UNAVAILABLE`: Audit storage could not be read or written.
- `CLEANUP_DISABLED`: Runtime cleanup is disabled.
- `METRICS_DISABLED`: Metrics endpoint is disabled.

