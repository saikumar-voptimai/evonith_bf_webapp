# Release Checklist

## Preflight

- [ ] Runtime directory is configured outside source.
- [ ] `EVONITH_AUTH_SECRET_KEY` is set outside git.
- [ ] Backend and frontend env files use placeholders in repo templates only.
- [ ] `python scripts/bootstrap_runtime.py --dry-run` passes.
- [ ] `python scripts/validate_deployment.py --profile production --offline --strict` passes.
- [ ] `python scripts/validate_api_cutover.py --strict` passes for cutover.
- [ ] `python scripts/backup_runtime.py --dry-run` passes.
- [ ] OpenAPI export passes.
- [ ] Canonical repository structure check passes.
- [ ] Full tests pass.

## Release Gate

```bash
python scripts/verify_release_readiness.py --allow-dirty --skip-tests
```

Run without `--skip-tests` in CI or a full validation environment.

## Post-Deploy

- [ ] `/api/v1/health` passes.
- [ ] `/api/v1/readiness` passes.
- [ ] Frontend status badge shows backend status.
- [ ] Smoke test passes.
- [ ] Backup archive is created and stored safely.
