# Backup And Restore Guide

## Backup

Default backup includes uploads, feedback metadata, and audit data. It excludes
logs, temp files, cache, and generated artifacts unless requested.

```bash
python scripts/backup_runtime.py --dry-run
python scripts/backup_runtime.py --output ./runtime/backups/evonith-runtime-test.tar.gz
```

## Restore

Restore is dry-run unless `--apply` is provided.

```bash
python scripts/restore_runtime.py --backup ./runtime/backups/evonith-runtime-test.tar.gz --target-runtime-dir ./runtime-restore-test
python scripts/restore_runtime.py --apply --backup ./runtime/backups/evonith-runtime-test.tar.gz --target-runtime-dir ./runtime-restore-test
```

## Safety

- Restore refuses unsafe runtime paths.
- Archive path traversal is blocked.
- Existing files are not overwritten unless `--overwrite` is provided.
- Database migrations are not run by restore.
- Secrets are not intentionally included.

