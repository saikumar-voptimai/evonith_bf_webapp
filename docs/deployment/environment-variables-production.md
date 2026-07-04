# Production Environment Variables

## Deployment

| Variable | Purpose | Example |
|---|---|---|
| `EVONITH_DEPLOYMENT_PROFILE` | Deployment profile | `production` |
| `EVONITH_DEPLOYMENT_ENV` | Environment name | `production` |
| `EVONITH_DEPLOYMENT_ROLE` | Service role | `backend`, `frontend`, `single-node` |
| `EVONITH_RELEASE_VERSION` | Release identifier | Empty in templates |
| `EVONITH_RELEASE_CHANNEL` | Release channel | `stable` |

## Runtime

| Variable | Purpose | Example |
|---|---|---|
| `EVONITH_RUNTIME_DIR` | Mutable runtime data root | `/var/lib/evonith-bf` |
| `EVONITH_BACKUP_DIR` | Backup archive directory | `/var/lib/evonith-bf/backups` |
| `EVONITH_BACKUP_MAX_ARCHIVES` | Retention count | `10` |

## Security

`EVONITH_AUTH_SECRET_KEY` must be a strong secret in production. Optional
provider keys such as `OPENAI_API_KEY` and `QDRANT_API_KEY` must be provided
only through protected environment files or a secret manager.

## Cutover

Set `USE_BACKEND_API=true` and page-specific `USE_BACKEND_API_*` flags for full
API mode. Keep `EVONITH_ALLOW_DIRECT_MODE_FALLBACK=true` until a later cleanup
phase removes direct mode.

