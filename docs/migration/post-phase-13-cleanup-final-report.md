# Post-Phase 13 Cleanup Final Report

Date: 2026-07-06  
Branch: `migration/backend-frontend-split`  
Rollback checkpoint before Prompt 11: `7c6dda2 chore: checkpoint before legacy shim reduction`

## Production-Readiness Status

Status: **Ready for production packaging and deployment validation from the canonical structure**, with documented compatibility surfaces retained for rollback and direct-mode fallback.

The managed repository environment passes the final structure, dependency, import-boundary, deployment, OpenAPI, backend startup, frontend import, and full regression checks. The only validation warnings are expected local/offline conditions: partial API cutover flags under `--allow-partial`, missing `BACKEND_API_BASE_URL` for offline local validation, and an allowed dirty working tree during this cleanup report.

Bare `python` backend imports fail in this shell because the unmanaged interpreter does not have FastAPI installed. Bare `pytest` is not available on this PowerShell PATH. The repo-supported `uv run ...` commands pass.

## Final Repository Structure

Canonical source layout:

```text
apps/
  backend_api/
    app/
      api/
      core/
      services/
      repositories/
      tasks/
      main.py
  frontend_streamlit/
    app.py
    custom_pages/
    services/
    config/
    ui/
    assets/
packages/
  furnace-data/
    furnace_data/
infra/
scripts/
docs/
tests/
runtime/
```

Confirmed canonical paths:

| Area | Path | Status |
|---|---|---|
| Backend API | `apps/backend_api/app` | Canonical FastAPI app, `/api/v1` routes, legacy routes, services, repositories, tasks. |
| Frontend Streamlit | `apps/frontend_streamlit` | Canonical app, pages, frontend API adapters, config helpers, UI helpers, assets. |
| Shared package | `packages/furnace-data/furnace_data` | Canonical package imported as `furnace_data`. |
| Runtime | `runtime` | Ignored generated/local data root except `.gitkeep`. |
| Tests | `tests` | Canonical backend, frontend, integration, dependency, deployment, structure, and fixtures layout. |

## Old Folders Removed Or Retained

| Old Path | Decision | Reason |
|---|---|---|
| `furnace-data-service/app` | Retained as shim-only | Old backend entrypoint/import compatibility remains required. Only `__init__.py` and `main.py` remain. |
| `furnace-data-service/test` and `furnace-data-service/tests` | Retained as deprecation README placeholders | Tests were migrated to `tests/backend`; placeholders document the old roots. |
| Root `furnace_data` | Retained as import shims only | Keeps `import furnace_data` and `import furnace_data.runtime_paths` working while canonical code lives in `packages/furnace-data`. |
| `src/app.py` | Retained as frontend compatibility shim | Keeps `streamlit run src/app.py` available for rollback. |
| `src/services` | Retained as re-export wrappers | Keeps old frontend service imports working. |
| `src/custom_pages` | Retained as page delegation wrappers | Keeps old page imports and compatibility checks working. |
| `src/config` and `src/ui` | Retained as compatibility wrappers plus shared config surface | Some direct-mode and frontend code still imports these paths. |
| `src/data`, `src/domain`, `src/utils`, `src/agents`, `src/reports`, `src/plotters`, `src/geometries` | Retained intentionally | Active direct-mode fallback and domain logic, not proven duplicate code. |

## Compatibility Shims Retained

- `furnace-data-service/app/__init__.py`
- `furnace-data-service/app/main.py`
- `src/app.py`
- `src/services/*.py`
- `src/custom_pages/*.py`
- `src/config/config_loader.py`
- `src/config/frontend_settings.py`
- `src/config/page_registry.py`
- `src/ui/**/*.py` wrappers
- `furnace_data/__init__.py`
- `furnace_data/runtime_paths.py`
- `apps/backend_api/app/__init__.py` temporary `app` alias for remaining backend `app.*` imports

These shims are intentionally retained until a separate deprecation/removal phase retires old commands and direct-mode fallback with coverage.

## Generated Files Removed

Source-tree generated/runtime artifacts were removed from active source folders. After the final cleanup pass, generated artifact count under `apps/`, `packages/`, `src/`, `furnace-data-service/`, and `scripts/` is `0`.

Runtime/generated data remains under `runtime/`, which is ignored except for `.gitkeep`. Current local runtime inventory contains `114` files and is not part of production source packaging.

## Alembic Leftovers Removed

Alembic active files and dependencies are absent:

- No active `alembic/` directory.
- No active `alembic.ini`.
- Dependency profile checks confirm Alembic is absent from active dependency files.
- Remaining `rg "alembic" .` matches are only negative guards in `scripts/check_dependency_profiles.py` and `tests/dependency/test_dependency_cleanup.py`.

## Model And Asset Cleanup

Active model assets live under:

```text
packages/furnace-data/furnace_data/assets/models/
```

Duplicate/outdated old model archive folders have been removed from production source. The model registry does not recursively expose old archive folders. Optional model loading remains lazy; backend import/startup does not load model files.

Frontend assets live under:

```text
apps/frontend_streamlit/assets/
```

Frontend assets do not contain backend model files.

## Dependency Cleanup Summary

Dependency groups remain split by runtime concern:

- `backend-base`
- `backend-data`
- `backend-ml`
- `backend-ai`
- `backend-vector`
- `backend-documents`
- `frontend`
- `dev`
- `edge`

`backend-base` excludes Streamlit, vector clients, provider SDKs, document/OCR stacks, and other heavy optional packages. The frontend group owns Streamlit and frontend UI/API-client needs. The shared package is editable from `./packages/furnace-data`.

## Documentation Cleanup Summary

Active docs now describe canonical paths as primary:

- `README.md`
- `docs/README.md`
- deployment guides under `docs/deployment/`
- `docs/testing/phase-13-testing-guide.md`
- operations docs under `docs/operations/`
- API docs under `docs/api/`
- cleanup plan under `docs/migration/post-phase-13-structure-cleanup-plan.md`

Historical phase docs are archived under `docs/archive/migration-history/`. Old commands remain documented only as compatibility or rollback commands.

## Final Validation Results

| Command | Result | Notes |
|---|---|---|
| `python scripts/export_backend_openapi.py` | Failed in bare shell | Bare `python` lacks FastAPI. Managed env passed. |
| `uv run python scripts/export_backend_openapi.py` | Passed | Exported `docs/api/openapi-v1.json`. |
| `python scripts/check_repository_structure.py` | Passed | Canonical structure and generated-artifact checks passed. |
| `python scripts/check_import_boundaries.py` | Passed | Backend/frontend import boundaries clean. |
| `python scripts/check_dependency_profiles.py` | Passed | Required groups and exclusions valid. |
| `python scripts/check_backend_minimal_startup.py` | Failed in bare shell | Bare `python` lacks FastAPI. Managed env passed. |
| `uv run python scripts/check_backend_minimal_startup.py` | Passed | `/api/v1/health` returned 200; app title `Evonith BF Backend API`. |
| `python scripts/check_frontend_api_imports.py` | Passed | Canonical frontend adapters and old wrappers import cleanly. |
| `python scripts/validate_deployment.py --profile local --offline` | Passed with warnings | Offline local warnings: runtime missing/not writable in bare shell, backend API URL not configured, backend import unavailable without FastAPI. |
| `uv run python scripts/validate_deployment.py --profile local --offline` | Passed with warning | Runtime writable and backend import pass; backend API URL not configured warning is expected for offline local validation. |
| `python scripts/validate_api_cutover.py --allow-partial --json` | Passed with warning | Partial cutover flags missing; allowed by `--allow-partial`. Required OpenAPI paths present. |
| `uv run python scripts/validate_api_cutover.py --allow-partial --json` | Passed with warning | Same expected partial cutover warning. |
| `python scripts/bootstrap_runtime.py --dry-run` | Passed with warning | No filesystem changes; directories would be created in bare dry-run context. |
| `python scripts/backup_runtime.py --dry-run` | Passed | No archive written; backup plan created. |
| `python scripts/verify_release_readiness.py --allow-dirty --skip-tests` | Passed with warning | Dirty tree allowed for cleanup report. |
| `uv run python scripts/verify_release_readiness.py --allow-dirty --skip-tests` | Passed with warning | Dirty tree allowed. |
| `python -c "from apps.backend_api.app.main import app; print(app.title)"` | Failed in bare shell | Bare `python` lacks FastAPI. Managed env passed. |
| `uv run python -c "from apps.backend_api.app.main import app; print(app.title)"` | Passed | Printed `Evonith BF Backend API`. |
| `python -c "from apps.frontend_streamlit.services.status_api import get_status; print('frontend import ok')"` | Passed | Printed `frontend import ok`. |
| `python -c "import furnace_data; print('furnace_data import ok')"` | Passed | Printed `furnace_data import ok`. |
| `pytest tests/backend -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/backend -q` | Passed | 387 passed, 1 skipped, 8 warnings. |
| `pytest tests/frontend -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/frontend -q` | Passed | 78 passed. |
| `pytest tests/integration -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/integration -q` | Passed | 7 passed, 4 warnings. |
| `pytest tests/dependency -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/dependency -q` | Passed | 18 passed. |
| `pytest tests/structure -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/structure -q` | Passed | 89 passed, 1 warning. |
| `pytest tests/deployment -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/deployment -q` | Passed | 9 passed, 1 skipped. |
| `pytest tests -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests -q` | Passed | 588 passed, 2 skipped, 11 warnings. |

## Stale-Reference Scan Results

| Scan | Result | Classification |
|---|---|---|
| `rg "alembic" .` | Matches only dependency-profile checker and dependency test guard. | Intentional negative guards. |
| `rg "src/storage" .` | Matches cleanup plan, archive docs, runtime migration script, structure checker, docs-current-path test, and one direct-mode fallback check in `src/agents/memory/structured_store.py`. | Intentional migration/compatibility references; no active source runtime artifacts. |
| Removed Neon DB package scan | Matches only `tests/backend/test_offline_static_guards.py`. | Intentional negative guard; the removed package path is not referenced by active code or docs. |
| `rg "bad-test.exe|phase6-test.txt|phase9_test_doc.txt|gw_config.json|run_time.txt" .` | Matches cleanup plan, structure checker, and no-generated-artifacts test. | Intentional removed-file guards/documentation. |
| Secret-pattern scan | Matches infra placeholders, local dev-only examples, archive docs, and current local testing commands. | Intentional placeholders/dev-only examples; no production secrets found by release-readiness scan. |

## Size And Inventory Summary

| Item | Value |
|---|---:|
| Working tree excluding `.git`, `.venv`, and tool caches | 249.25 MB |
| `apps/` | 9.23 MB |
| `packages/` | 92.74 MB |
| `runtime/` | 141.70 MB |
| Model assets | 92.43 MB |
| Runtime file count | 114 |
| Generated source artifact count | 0 |

Largest 20 files in the working tree, excluding `.git`, `.venv`, and tool caches:

| Size | Path |
|---:|---|
| 37.20 MB | `packages/furnace-data/furnace_data/assets/models/eta_co_opt.pkl` |
| 21.62 MB | `runtime/datasets/results/5bd9ceae810a.csv` |
| 21.62 MB | `runtime/datasets/static/ml_dataset_20260401_151134.csv` |
| 17.69 MB | `packages/furnace-data/furnace_data/assets/models/fuel_rate_opt.json` |
| 13.25 MB | `runtime/datasets/results/ee3425d7d20a.csv` |
| 13.25 MB | `runtime/datasets/static/ml_dataset.csv` |
| 11.36 MB | `packages/furnace-data/furnace_data/assets/models/coke_rate_opt.json` |
| 11.13 MB | `packages/furnace-data/furnace_data/assets/models/fuel_rate_opt.pkl` |
| 10.06 MB | `runtime/datasets/static/furnace_dataset_20260706_140637_892522.csv` |
| 10.06 MB | `runtime/datasets/static/furnace_dataset_20260706_140118_204793.csv` |
| 9.99 MB | `runtime/datasets/static/furnace_dataset_20260704_225647_662722.csv` |
| 9.99 MB | `runtime/datasets/static/furnace_dataset_20260703_173029_681966.csv` |
| 9.99 MB | `runtime/datasets/static/furnace_dataset.csv` |
| 9.99 MB | `runtime/datasets/static/furnace_dataset_20260703_165216_109695.csv` |
| 9.71 MB | `runtime/datasets/static/furnace_dataset_20260625_181235_159053.csv` |
| 7.58 MB | `apps/frontend_streamlit/assets/data/bf_hero.png` |
| 7.25 MB | `packages/furnace-data/furnace_data/assets/models/coke_rate_opt.pkl` |
| 2.06 MB | `packages/furnace-data/furnace_data/assets/models/etaco_model.pkl` |
| 1.52 MB | `packages/furnace-data/furnace_data/assets/models/productionrate_model.pkl` |
| 983.64 KB | `packages/furnace-data/furnace_data/assets/models/unitcost_fuel_model.json` |

## Final Source-Tree Confirmation

- Canonical backend path exists and imports in the managed environment.
- Canonical frontend path exists and imports frontend adapters.
- Canonical shared package path exists and `import furnace_data` still works.
- No active Alembic files remain.
- No generated source artifacts remain under canonical or legacy source roots.
- Old duplicate model archive folders are absent from production source.
- Source runtime DB/uploads are absent; runtime artifacts live under `runtime/`.
- Backend import-boundary checks confirm no Streamlit imports in backend code.
- Frontend API import checks confirm adapters do not import backend internals, DB clients, vector clients, LLM SDKs, or model loaders.
- Full managed regression suite passes.

## Known Remaining Technical Debt

- Direct-mode fallback remains active under `src/data`, `src/domain`, `src/utils`, `src/agents`, `src/reports`, `src/plotters`, and `src/geometries`.
- Compatibility shims remain under `src`, `furnace-data-service/app`, and root `furnace_data`.
- Backend legacy route modules remain enabled by compatibility requirements.
- Shared YAML under `src/config` still has mixed frontend/direct-mode ownership.
- Runtime contains local generated datasets/uploads/log-like artifacts that should be managed by runtime retention policies outside source control.
- Some tests report deprecation warnings for `datetime.utcnow()` and a pandas fill/downcast future warning.
- Managed commands should be preferred in docs/automation because bare `python` in this shell does not include backend dependencies and bare `pytest` is unavailable.

## Recommended Future Cleanup

1. Plan a separate direct-mode deprecation phase only after product owners approve retiring fallback behavior.
2. Remove `src` wrappers, `furnace-data-service/app` shims, and root `furnace_data` shims only after compatibility commands are formally deprecated and tests are updated.
3. Review shared YAML ownership under `src/config` and move canonical config only with direct-mode impact analysis.
4. Decide whether large bundled model assets should remain packaged or move to externally provisioned `EVONITH_MODEL_DIR` for production artifacts.
5. Add runtime retention/backup policy for local datasets, uploads, cache files, and logs under `runtime/`.
6. Address `datetime.utcnow()` deprecation warnings and pandas downcast warning in a normal maintenance pass.
7. Ensure CI uses `uv run ...` or an equivalent environment bootstrap so final validation matches the managed local environment.