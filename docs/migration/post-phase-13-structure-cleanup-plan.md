# Post-Phase 13 Structure Cleanup Plan

## Purpose

This document is the baseline and migration map for the safe repository
restructuring cleanup after Phase 13. The initial planning pass used these
constraints:

- Do not move files yet.
- Do not delete files yet.
- Do not change imports yet.
- Do not change API contracts.
- Do not remove compatibility shims.
- Do not remove direct-mode fallback.
- Do not remove legacy routes.
- Do not change dependency groups yet.

Target structure:

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

## Skeleton Status

The canonical directory skeleton was created before relocating real backend,
frontend, or shared package business code. The backend child folders under
`apps/backend_api/app` now contain the relocated backend implementation.
The shared package implementation now lives under
`packages/furnace-data/furnace_data`.

No backend code, frontend code, shared package implementation, imports, API
contracts, compatibility shims, direct-mode fallback, legacy route modules, or
dependency groups were moved or removed in this skeleton step.

The backend relocation into `apps/backend_api/app` has now been completed while
preserving `furnace-data-service/app` compatibility imports and legacy route
behavior. Frontend support modules, the Streamlit entrypoint, page files, and
the shared `furnace_data` package have also been relocated. The next planned
implementation step is auditing remaining model assets, shared YAML config,
and runtime-like legacy folders before any deletion.

## Asset And Model Cleanup Status

The asset cleanup pass has now separated frontend UI assets, packaged source
assets, active model artifacts, and generated runtime data.

Current classification and location:

- Frontend UI assets live under `apps/frontend_streamlit/assets`. This includes
  CSS, logo/hero images, and the Excel template.
- Active backend/model assets live under
  `packages/furnace-data/furnace_data/assets/models`. `EVONITH_MODEL_DIR`
  remains supported as an override for deployments that keep models outside the
  repo checkout.
- AI Copilot source analysis markdown lives under
  `packages/furnace-data/furnace_data/assets/copilot_analysis`.
- FurnaceMind built-in prompt/skill source assets live under
  `packages/furnace-data/furnace_data/assets/furnacemind`.
- Generated/static dataset cache files and control-bound overrides live under
  `runtime/datasets/static` and `runtime/cache`.
- `src/assets/models/old_26_14` and
  `src/assets/models/old_bmo_12062026` were removed from the source tree after
  confirming no production references to the archive folders remained. The two
  V-Sense Unit Cost files still referenced by `setting_vsense.yml` were moved
  into the active canonical model directory before removing the archive folder.

Model registry behavior after cleanup:

- `ModelRegistryService` defaults to packaged model assets when
  `EVONITH_MODEL_DIR` is not set.
- Model discovery lists only root-level model files plus the explicitly allowed
  active `bmo_fuel` bundle directory.
- Discovery no longer recursively exposes arbitrary nested folders or `old_*`
  archive directories as model IDs.
- Model loading remains lazy; missing optional models return structured
  model-status/error responses instead of failing backend startup.

No API contracts, direct-mode fallback, legacy routes, compatibility shims, or
dependency groups were intentionally changed in the asset cleanup step.

Asset cleanup validation results:

- `python scripts/check_repository_structure.py` passed during the asset move.
  The runtime-like source artifacts noted in that earlier run were removed in
  the generated-artifact cleanup pass below.
- `python scripts/check_import_boundaries.py` passed.
- `python scripts/check_backend_minimal_startup.py` failed with the PATH
  interpreter because `fastapi` is not installed there.
- `uv run python scripts/check_backend_minimal_startup.py` passed.
- Bare `pytest ...` commands are not available on the current PowerShell PATH.
  The meaningful test runs used `uv run pytest ...`.
- `uv run pytest tests/structure/test_assets_and_models_structure.py -q`
  passed with 9 tests and 1 warning.
- `uv run pytest tests/structure -q` passed with 59 tests and 1 warning.
- `uv run pytest furnace-data-service/tests -q` passed with 102 tests and
  2 warnings.
- `uv run pytest tests/integration -q` passed with 7 tests and 4 warnings.
- `uv run pytest tests -q` passed with 385 tests, 1 skipped, and 5 warnings.

## Generated Artifact Cleanup Status

Generated/runtime artifacts and manual clutter were removed from the source
tree after running `uv run python scripts/migrate_runtime_files.py` to copy or
confirm runtime equivalents.

Removed source-runtime artifacts:

- `furnace-data-service/data/results/*`
- `furnace-data-service/data/static/*`
- `src/storage/feedback/tickets.db`
- `src/storage/feedback/images/*`
- `src/storage/feedback/write_probe.txt`
- `src/storage/*_summaries.json`
- source-tree `__pycache__` directories and `*.pyc` files

Removed obsolete/manual clutter:

- `bad-test.exe`
- `phase6-test.txt`
- `phase9_test_doc.txt`
- `gw_config.json`
- root `main.py` scaffold
- `run_time.txt`
- root `static/` placeholder assets
- `scripts/diagnose_fetch_pipeline.py`
- `scripts/diagnose_fetch_pipeline.report.md`
- `scripts/validate_slag_balance.py`
- `scripts/slag_validation_results.csv`

Removed obsolete database-migration leftovers:

- root legacy migration folder
- root migration `.ini`
- legacy migration dependency entries from `pyproject.toml`,
  `requirements.txt`, and `uv.lock`

Intentionally retained:

- Deployment/runtime scripts such as `scripts/bootstrap_runtime.py`,
  `scripts/backup_runtime.py`, `scripts/restore_runtime.py`,
  `scripts/validate_deployment.py`, and edge start scripts.
- Compatibility shims, direct-mode fallback code, legacy routes, active config
  files, and active model assets.
- `scripts/migrate_runtime_files.py`, which still knows old source locations so
  it can migrate older checkouts into `EVONITH_RUNTIME_DIR`.
- Historical migration docs that describe old runtime source paths.

Skeleton validation results:

- `python scripts/check_repository_structure.py` passed. Runtime-like source
  artifacts from that earlier phase were removed in the later cleanup pass.

Generated-artifact cleanup validation results:

- `python scripts/check_repository_structure.py` passed.
- `python scripts/check_dependency_profiles.py` passed.
- The legacy migration-tool text search returned no stale references.
- Bare `pytest` is not available on the current PowerShell PATH; the meaningful
  test runs used `uv run`.
- `uv run pytest tests/structure -q` passed with 66 tests and 1 warning.
- `uv run pytest tests/deployment -q` passed with 9 tests and 1 skipped.
- `uv run pytest tests -q` passed with 392 tests, 1 skipped, and 5 warnings.

## Backend Relocation Status

The backend implementation has now been moved from `furnace-data-service/app`
to `apps/backend_api/app`. The canonical backend implementation tree contains
`main.py`, `api`, `core`, `services`, `repositories`, `tasks`, `models`,
`routes`, and `config.py`.

The old `furnace-data-service/app` package is now a temporary
Phase 12/cleanup compatibility shim. It contains only `__init__.py` and
`main.py` as Python source files; old `app.*` imports are routed to the
canonical implementation, and `furnace-data-service/app/main.py` re-exports
`app` and `create_app` from `apps.backend_api.app.main`.

No frontend code, `furnace_data` package code, API contracts, legacy route
behavior, direct-mode fallback, or dependency groups were moved or removed in
this backend relocation step.

Canonical backend command:

```bash
EVONITH_RUNTIME_DIR=./runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uvicorn apps.backend_api.app.main:app --host 0.0.0.0 --port 8080
```

Temporary compatibility command:

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime \
EVONITH_AUTH_SECRET_KEY=dev-only-secret-change-me \
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Frontend Support Relocation Status

Frontend support modules were moved into the canonical Streamlit app package
before the frontend entrypoint/page move. At that support-module step,
`src/app.py`, `src/custom_pages`, direct-mode backend-heavy `src/utils`, and
the `furnace_data` package were intentionally left in place.

- API adapters now live under `apps/frontend_streamlit/services`.
- Frontend Python config helpers now live under `apps/frontend_streamlit/config`.
  Shared/domain YAML files remain under `src/config` for direct-mode and model
  consumers; the canonical `config_loader` searches that legacy config data
  location.
- UI helper modules now live under `apps/frontend_streamlit/ui`.
- Frontend-owned CSS, logo/hero images, and the Excel template now live under
  `apps/frontend_streamlit/assets`.
- `src/services`, `src/config`, and `src/ui` contain temporary compatibility
  wrappers that re-export the canonical modules.
- Model, prompt, and generated runtime assets were moved in a later asset
  cleanup pass documented above.

No page behavior, API contracts, direct-mode fallback, dependency groups,
custom page implementations, backend code, model assets, or shared package code
were intentionally changed in this frontend support relocation step.

Frontend support relocation validation results:

- `python scripts/check_frontend_api_imports.py` passed.
- `python scripts/check_import_boundaries.py` passed.
- `python -c "from apps.frontend_streamlit.services.status_api import get_status; print('new status import ok')"` passed.
- `python -c "from src.services.status_api import get_status; print('old status shim import ok')"` passed.
- Bare `pytest` is not available on the current PowerShell PATH; the meaningful
  test runs used `uv run`.
- `uv run pytest tests/frontend -q` passed with 78 tests.
- `uv run pytest tests/structure -q` passed with 35 tests and 1 warning.
- `uv run pytest tests -q` reproduced the known BMO baseline failure:
  `tests/test_bmo_context_provider.py::test_history_frame_layers_recent_online_context_for_model_lags`
  expected `ORE_CALC_MT == 30.0` but got `53.704`. The run completed with
  1 failed, 360 passed, 1 skipped, and 5 warnings.

## Frontend App And Page Relocation Status

The canonical Streamlit app and page files now live under
`apps/frontend_streamlit`.

- `apps/frontend_streamlit/app.py` is the canonical Streamlit entrypoint.
- `apps/frontend_streamlit/custom_pages/*.py` contains the real page
  implementations.
- `src/app.py` remains a compatibility shim for `streamlit run src/app.py`.
- `src/custom_pages/*.py` are temporary compatibility wrappers that delegate to
  the canonical page files through `run_canonical_page`.
- Page order and names remain defined by
  `apps/frontend_streamlit/config/page_registry.py`.

No page logic, API contracts, feature flags, backend status badge behavior,
direct-mode fallback, backend code, `furnace_data`, or backend-heavy
`src/utils` modules were intentionally changed in this frontend app/page
relocation step.

Frontend app/page relocation validation results:

- `python scripts/check_repository_structure.py` passed. Runtime-like source
  artifacts from that earlier phase were removed in the later cleanup pass.
- `python scripts/check_frontend_api_imports.py` passed.
- `python scripts/check_import_boundaries.py` passed.
- `python -m py_compile apps/frontend_streamlit/app.py` and
  `python -m py_compile src/app.py` initially failed inside the sandbox with
  Windows access-denied errors while replacing `__pycache__` files; the same
  exact commands passed when rerun outside the sandbox.
- Bare `pytest` is not available on the current PowerShell PATH; the meaningful
  test runs used `uv run`.
- `uv run pytest tests/frontend -q` passed with 78 tests.
- `uv run pytest tests/structure -q` passed with 42 tests and 1 warning.
- `uv run pytest tests -q` reproduced the known BMO baseline failure:
  `tests/test_bmo_context_provider.py::test_history_frame_layers_recent_online_context_for_model_lags`
  expected `ORE_CALC_MT == 30.0` but got `53.704`. The run completed with
  1 failed, 367 passed, 1 skipped, and 5 warnings.

## Shared Package Relocation Status

The shared `furnace_data` implementation has now been moved from
`furnace_data/furnace_data` to `packages/furnace-data/furnace_data`.

- `packages/furnace-data/pyproject.toml` is the canonical shared package
  metadata.
- Root `pyproject.toml` and `uv.lock` now point the editable `furnace_data`
  source at `./packages/furnace-data`.
- `furnace_data/__init__.py` is a temporary compatibility shim that extends
  the package path to the canonical implementation.
- `furnace_data/runtime_paths.py` remains a temporary compatibility shim for
  direct file-path imports; normal imports resolve
  `furnace_data.runtime_paths` from the canonical package first.
- `import furnace_data`, `import furnace_data.runtime_paths`, and
  `import furnace_data.relational` remain supported.

No business logic, API contracts, backend startup behavior, frontend
direct-mode fallback, model assets, shared YAML config, or dependency groups
were intentionally changed in this shared package relocation step.

Shared package relocation validation results:

- `python -c "import furnace_data; print(furnace_data)"` passed and imported
  the root compatibility package.
- `python -c "import furnace_data.runtime_paths; print('runtime paths ok')"`
  passed.
- `python -c "from apps.backend_api.app.main import app; print(app.title)"`
  failed under bare Python with `ModuleNotFoundError: No module named
  'fastapi'`. The same command passed with `uv run` and printed
  `Evonith BF Backend API`.
- `python scripts/check_backend_minimal_startup.py` failed under bare Python
  with `No module named 'fastapi'`. The same command passed with `uv run`.
- `python scripts/check_dependency_profiles.py` passed.
- `python scripts/export_backend_openapi.py` failed under bare Python with
  `ModuleNotFoundError: No module named 'fastapi'`. The same command passed
  with `uv run`.
- Bare `pytest` is not available on the current PowerShell PATH; the meaningful
  test runs used `uv run`.
- `uv run pytest tests/structure/test_shared_package_structure.py -q` passed
  with 8 tests and 1 warning.
- `uv run pytest tests/structure -q` passed with 50 tests and 1 warning.
- `uv run pytest furnace-data-service/tests -q` passed with 102 tests and
  2 warnings.
- `uv run pytest tests/integration -q` passed with 7 tests and 4 warnings.
- `uv run pytest tests -q` reproduced the known BMO baseline failure:
  `tests/test_bmo_context_provider.py::test_history_frame_layers_recent_online_context_for_model_lags`
  expected `ORE_CALC_MT == 30.0` but got `53.704`. The run completed with
  1 failed, 375 passed, 1 skipped, and 5 warnings.

## Current Layout Summary

- `apps/backend_api/app/main.py` is the current canonical FastAPI entrypoint.
  Backend implementation modules now live under `apps/backend_api/app`.
- `apps/backend_api/app/__init__.py` keeps existing absolute `app.*` imports
  working when the canonical package is imported first.
- `apps/frontend_streamlit/app.py` is the current canonical Streamlit
  entrypoint.
- `apps/frontend_streamlit/custom_pages` contains the canonical Streamlit page
  implementations.
- `apps/frontend_streamlit/services` contains the canonical frontend API
  adapter implementations.
- `apps/frontend_streamlit/config` contains canonical frontend Python config
  helpers. Shared/domain YAML config files remain under `src/config`.
- `apps/frontend_streamlit/ui` contains canonical frontend UI helper modules.
- `apps/frontend_streamlit/assets` contains frontend-owned CSS, logo/hero
  images, and templates. Model assets and generated/static dataset/cache files
  have been moved out of `src/assets`.
- `furnace-data-service/app` is a temporary compatibility package for old
  `app.*` imports and old service-local startup commands.
- `src` is still the direct-mode domain and compatibility tree. It contains
  Streamlit entrypoint/page wrappers, frontend support compatibility wrappers,
  shared YAML config, data access, domain logic,
  agents, reports, plotters, geometry helpers, and direct-mode fallbacks.
- `packages/furnace-data/furnace_data` is the current shared package
  implementation.
- `furnace_data` is a temporary compatibility package for repo-root imports.
- `runtime` is the canonical generated runtime directory and should remain
  ignored except for `runtime/.gitkeep`.
- `scripts`, `infra`, `docs`, and `tests` already exist at the repository root.
  They should remain root-level targets, with path references updated only
  after actual moves.
- Obsolete legacy database-migration leftovers were removed from the root.

## Current Canonical Paths

| Area | Current Canonical Path | Notes |
|---|---|---|
| Backend entrypoint | `apps/backend_api/app/main.py` | Exposes `app` and `create_app`. Includes versioned API router and legacy routes when enabled. |
| Frontend entrypoint | `apps/frontend_streamlit/app.py` | Sets up Streamlit navigation and legacy `src` import paths. |
| Backend OpenAPI export | `scripts/export_backend_openapi.py` | Imports `apps.backend_api.app.main`. |
| Frontend launcher | `run_streamlit.py` | Starts `apps/frontend_streamlit/app.py`. |

## Current Backend Compatibility Paths

- `furnace-data-service/app/main.py` re-exports `app` and `create_app` from
  `apps.backend_api.app.main`.
- `furnace-data-service/run.py` starts `apps.backend_api.app.main:app` while
  preserving service-local startup behavior.
- `apps/backend_api/app/__init__.py` registers the canonical package as the
  temporary top-level `app` module so remaining backend `app.*` imports resolve
  without copying code back into `furnace-data-service/app`.
- `apps/backend_api/app/routes/{health,data,dataset}.py` are legacy
  unversioned route modules and remain enabled by default through
  `EVONITH_ENABLE_LEGACY_ROUTES=true`.
- `apps/backend_api/app/api/v1/routes/datasets.py` still delegates some
  compatibility endpoints to `app.routes.dataset`; those imports resolve
  through the canonical `app` alias.
- Service-local test roots now contain deprecated README placeholders only; migrated
  tests live under `tests/backend/service_api` and
  `tests/backend/legacy_service`.

## Current Frontend Compatibility Paths

- `src/app.py` is the legacy Streamlit entrypoint shim. It runs
  `apps/frontend_streamlit/app.py`.
- `apps/frontend_streamlit/_legacy.py` adds `src` and the repo root to
  `sys.path` and provides helper functions for legacy and canonical page
  delegation.
- `apps/frontend_streamlit/__init__.py` calls
  `ensure_frontend_legacy_paths()`.
- `src/custom_pages/*.py` are wrappers that execute matching canonical
  `apps/frontend_streamlit/custom_pages/*.py` files.
- `src/services/*.py` re-export matching
  `apps/frontend_streamlit/services/*.py` adapters.
- `src/config/{config_loader,frontend_settings,page_registry}.py` re-export
  matching `apps/frontend_streamlit/config` modules.
- `src/ui/**/*.py` re-export matching `apps/frontend_streamlit/ui` modules.
- Frontend-owned CSS/images/templates moved to
  `apps/frontend_streamlit/assets`, and hardcoded page/helper asset paths were
  updated where needed.
- Old-style `src/assets/models/...` paths remain supported by the model asset
  resolver for compatibility, but active bundled models now live under
  `packages/furnace-data/furnace_data/assets/models`.
- `apps/frontend_streamlit/{assets,config,ui,utils}/__init__.py` extend package
  paths to the corresponding `src` directories.
- Direct-mode modules under `src/data`, `src/domain`, `src/utils`,
  `src/agents`, `src/reports`, `src/plotters`, and `src/geometries` remain in
  active use and must not be deleted.

## Current Shared Package Locations

- `packages/furnace-data/furnace_data` is the current package implementation.
- `packages/furnace-data/pyproject.toml` is the current package metadata.
- `packages/furnace-data/furnace_data/assets/models` contains active bundled
  model assets.
- `packages/furnace-data/furnace_data/assets/copilot_analysis` contains AI
  Copilot source analysis markdown.
- `packages/furnace-data/furnace_data/assets/furnacemind` contains built-in
  FurnaceMind source prompt and skill assets.
- Root `pyproject.toml` uses:

```toml
[tool.uv.sources]
furnace_data = { path = "./packages/furnace-data", editable = true }
```

- `furnace_data/__init__.py` is a compatibility package shim that exposes the
  canonical package path to repo-root imports.
- `furnace_data/runtime_paths.py` is a compatibility wrapper for direct
  file-path imports and delegates to
  `packages/furnace-data/furnace_data/runtime_paths.py`.
- The root compatibility surface should remain until direct imports, editable
  metadata, and tests are deliberately cut over in a later compatibility-removal
  phase.

## Current Runtime And Generated Locations

Canonical generated runtime location:

- `runtime/.gitkeep`
- `runtime/cache`
- `runtime/jobs`
- `runtime/uploads`
- `runtime/uploads/feedback`
- `runtime/uploads/furnacemind`
- `runtime/feedback`
- `runtime/datasets`
- `runtime/datasets/results`
- `runtime/datasets/static`
- `runtime/logs`
- `runtime/qdrant`
- `runtime/temp`
- Additional runtime subdirectories currently observed: `runtime/audit`,
  `runtime/compute`, `runtime/copilot`, and `runtime/furnacemind`.

Legacy or compatibility runtime-like locations still referenced:

- `storage/feedback`
- `logs`
- `source_files`
- old-style `src/assets/data/furnace_dataset.csv` config/test paths, where
  still deliberately covered by compatibility tests
- `src/config/bmo_operator_inputs.yml`
- `src/geometries/mask_*.pkl` as geometry mask read fallbacks

The structure check now fails if generated DB, CSV, log, upload, or obsolete
manual artifacts return under source roots.

## Scripts, Infra, Docs, And Tests

- `scripts` contains migration, validation, deployment, OpenAPI, and runtime
  utility scripts. Keep the directory root-level.
- `infra` contains `caddy`, `env`, `nginx`, and `systemd` examples. Keep the
  directory root-level and update paths only after canonical moves.
- `docs` contains `api`, `deployment`, `migration`, `operations`, and
  `testing`. Keep historical phase docs intact.
- `tests` contains the canonical `backend`, `frontend`, `integration`,
  `dependency`, `structure`, `deployment`, and `fixtures` suites.
- `furnace-data-service/test` and `furnace-data-service/tests` are deprecated
  README placeholders only; they must not contain active tests.
- `requirements` contains profile files generated from or aligned with
  dependency groups. Do not change dependency groups in this cleanup step.

## Known Duplicate Directories

- `apps/backend_api/app` and `furnace-data-service/app` remain paired only by
  compatibility shims; implementation code is canonical under
  `apps/backend_api/app`.
- `apps/frontend_streamlit/custom_pages` and `src/custom_pages`
- `apps/frontend_streamlit/services` and `src/services`
- `apps/frontend_streamlit/ui` and `src/ui`
- `apps/frontend_streamlit/config` and `src/config`
- `apps/frontend_streamlit/assets` and `src/assets`
- `apps/frontend_streamlit/utils` and `src/utils`
- `furnace_data` compatibility shims and
  `packages/furnace-data/furnace_data` canonical implementation
- `tests/backend/service_api` and `tests/backend/legacy_service` preserve formerly service-local test coverage; old service test roots are README placeholders
- `runtime`, `storage`, `src/storage`, and `furnace-data-service/data`
- `static`, `src/assets`, and `apps/frontend_streamlit/assets`
- `src/domain/bmo` and `src/utils/bmo` have overlapping BMO domain modules and
  need a separate ownership audit before any deletion.

## Known Compatibility Shims

- `furnace-data-service/app/main.py`
- `apps/backend_api/app/__init__.py`
- `src/app.py`
- `apps/frontend_streamlit/_legacy.py`
- `apps/frontend_streamlit/__init__.py`
- `src/custom_pages/*.py`
- `src/services/*.py`
- `src/config/{config_loader,frontend_settings,page_registry}.py`
- `src/ui/**/*.py`
- `apps/frontend_streamlit/{assets,config,ui,utils}/__init__.py`
- `furnace_data/__init__.py`
- `furnace_data/runtime_paths.py`
- `furnace-data-service/run.py`
- v1 compatibility wrappers that delegate to legacy route modules, especially
  `apps/backend_api/app/api/v1/routes/datasets.py`

## Migration Map

| Current Path | Target Path | Action | Compatibility Strategy | Risk Level | Tests Required | Notes |
|---|---|---|---|---|---|---|
| `furnace-data-service/app` | `apps/backend_api/app` | Moved in backend relocation step. | `furnace-data-service/app` is now a temporary compatibility package that resolves old `app.*` imports to canonical `apps.backend_api.app`. `main.py` re-exports `app` and `create_app`. | High | OpenAPI export, backend minimal startup, import boundaries, backend route tests, structure tests, full pytest. | Preserve `EVONITH_ENABLE_LEGACY_ROUTES`, response schemas, service state names, runtime paths, and optional dependency laziness. |
| `furnace-data-service/app/api` | `apps/backend_api/app/api` | Moved. | Old `app.api.*` imports resolve through the package shim. | High | OpenAPI old/new equivalence, API v1 tests, import boundaries. | Versioned API remains under `/api/v1`. |
| `furnace-data-service/app/core` | `apps/backend_api/app/core` | Moved. | Old `app.core.*` imports resolve through the package shim. | High | Startup, auth/security tests, dependency profile checks. | Contains settings, security, middleware, errors, and logging. |
| `furnace-data-service/app/services` | `apps/backend_api/app/services` | Moved. | Old `app.services.*` imports resolve through the package shim. | High | Backend service tests, frontend API flow tests, minimal startup forbidden-module check. | Optional dependencies remain lazy. |
| `furnace-data-service/app/repositories` | `apps/backend_api/app/repositories` | Moved. | Old `app.repositories.*` imports resolve through the package shim. | High | Repository tests, feedback/auth/furnacemind tests, runtime path tests. | Storage semantics remain under `EVONITH_RUNTIME_DIR`. |
| `furnace-data-service/app/tasks` | `apps/backend_api/app/tasks` | Moved. | Old `app.tasks.*` imports resolve through the package shim. | Medium | Dataset/job route tests and structure checks. | Generated job files stay in runtime, not source. |
| `furnace-data-service/app/models` | `apps/backend_api/app/models` | Moved. | Old `app.models.*` imports resolve through the package shim. | Medium | OpenAPI export, schema tests, route tests. | Schema ownership can still be reviewed later, but no contract change was made. |
| `furnace-data-service/app/routes` | `apps/backend_api/app/routes` | Moved while legacy routes remain supported. | Old `app.routes.*` imports resolve through the package shim and `EVONITH_ENABLE_LEGACY_ROUTES=true` remains the default. | High | Legacy route tests, OpenAPI compatibility checks, frontend direct/API fallback tests. | Legacy route behavior is preserved. |
| `src/app.py` | `apps/frontend_streamlit/app.py` | Canonicalized; legacy `src/app.py` shim preserved. | Keep `streamlit run src/app.py` working through `runpy` delegation. | Medium | Frontend entrypoint tests, compatibility shim tests, Streamlit smoke/manual check. | Do not remove until direct-mode fallback is retired. |
| `src/custom_pages` | `apps/frontend_streamlit/custom_pages` | Moved page implementations. | `src/custom_pages` files are temporary wrappers that execute canonical page files. | High | Frontend entrypoint tests, page registry tests, page-specific API/direct fallback tests. | Page filenames and registry order are preserved for Streamlit navigation. |
| `src/services` | `apps/frontend_streamlit/services` | Moved API adapters. | `src/services` files are temporary wrappers that re-export canonical adapters. | Medium | `check_frontend_api_imports.py`, frontend API adapter tests, import-boundary tests. | Frontend adapters must not import backend internals, DB clients, Qdrant clients, LLM SDKs, or model loaders. |
| `src/config` | `apps/frontend_streamlit/config` for frontend-owned Python config; shared config target TBD | Moved `config_loader.py`, `frontend_settings.py`, and `page_registry.py`. Shared/domain YAML files remain in `src/config`. | `src/config` Python files are temporary wrappers. Canonical `config_loader` searches canonical config first, then legacy `src/config` YAML data. | High | Config loader tests, material balance tests, BMO tests, backend startup, frontend smoke. | `FURNACE_CONFIG_DIR` currently defaults to `src/config`; YAML ownership still needs a later split. |
| `src/ui` | `apps/frontend_streamlit/ui` | Moved UI helper modules. | `src/ui` files are temporary wrappers that re-export canonical UI modules while legacy page imports remain supported. | Medium | Frontend page tests, UI unit tests, import-boundary tests. | Direct-mode page imports remain supported through compatibility wrappers. |
| `src/assets` frontend-only assets | `apps/frontend_streamlit/assets` | Moved CSS, logo/hero images, and the frontend Excel template. | Hardcoded page/helper paths for moved assets now point at canonical assets. | High | Frontend visual/page smoke, asset structure tests, full pytest. | Frontend assets must not contain backend model files or `models` archive folders. |
| `src/assets/models` | `packages/furnace-data/furnace_data/assets/models` with optional external `EVONITH_MODEL_DIR` override | Moved active model files and the active `bmo_fuel` bundle. Removed old archive folders from source after extracting the two referenced V-Sense Unit Cost files. | Keep `EVONITH_MODEL_DIR` support and compatibility resolution for old-style `src/assets/models/...` paths used by tests or older configs. | High | BMO tests, optimization runtime tests, model registry tests, API compute tests, full pytest. | `ModelRegistryService` now defaults to packaged models and only lists root files plus the allowed `bmo_fuel` bundle, not recursive `old_*` folders. |
| `src/assets/data/copilot_analysis` | `packages/furnace-data/furnace_data/assets/copilot_analysis` | Moved source analysis markdown. | Copilot prompt loader reads package assets first, then legacy path if present. | Medium | Copilot prompt/page tests, asset structure tests. | These are source prompt assets, not generated runtime files. |
| `furnace_data/furnace_data` | `packages/furnace-data/furnace_data` | Moved shared package implementation. | Root `furnace_data/__init__.py` extends the package path to the canonical implementation, and root `pyproject.toml` points editable installs at `./packages/furnace-data`. | High | Runtime path tests, package import tests, dependency profile checks, backend/frontend startup, full pytest. | Dependency group names were not changed; only editable source metadata changed. |
| `furnace_data/runtime_paths.py` compatibility wrapper | Temporary root shim plus canonical `packages/furnace-data/furnace_data/runtime_paths.py` | Retained root wrapper. | Continue supporting `from furnace_data.runtime_paths import ...` from repo-root and installed-package contexts. | High | `tests/test_runtime_paths.py`, backend startup, frontend startup, runtime migration scripts. | Shim decision: retain root wrapper during migration; remove only in a later compatibility-removal phase. |
| `scripts` | `scripts` | Keep root-level. Update path constants after moves. | Scripts should prefer canonical app/package paths while accepting legacy paths during transition. | Medium | Script checks, deployment tests, release readiness validation. | Do not update scripts before the actual move they depend on. |
| `infra` | `infra` | Keep root-level. Update service examples after moves. | Preserve old commands in rollback docs while promoting canonical commands. | Medium | Deployment tests, validation scripts, manual dry-run of edge scripts. | Includes `infra/env`, `infra/nginx`, `infra/caddy`, and `infra/systemd`. |
| `tests` | `tests/{backend,frontend,integration,dependency,structure,deployment,fixtures}` | Reorganized. | Keep compatibility shim tests in canonical suites while old service test roots remain deprecated placeholders. | Low | Full pytest, targeted structure/backend/frontend suites. | Completed in Prompt 8. |
| `furnace-data-service/test` and `furnace-data-service/tests` | `tests/backend/service_api` and `tests/backend/legacy_service` | Moved. | Old service test roots contain deprecation READMEs only. | Low | `uv run pytest tests/backend -q`; `uv run pytest tests -q`. | Completed in Prompt 8; compatibility shim tests remain active. |
| `runtime` | `runtime` with only `.gitkeep` tracked | Keep generated files ignored. Do not move into app/package directories. | Runtime helpers continue to resolve `EVONITH_RUNTIME_DIR`, defaulting to repo `runtime`. | High | Runtime path tests, bootstrap/backup/restore tests, health endpoint tests. | Delete generated runtime files only with explicit cleanup approval, not as source restructure. |
| `furnace-data-service/data/results` | `runtime/datasets/results` | Removed source copies after copying to runtime with `scripts/migrate_runtime_files.py`. | Runtime helpers remain canonical. | Medium | Structure check, dataset tests, runtime migration dry-run. | Generated sidecar CSVs are runtime artifacts. |
| `furnace-data-service/data/static` | `runtime/datasets/static` | Removed source copies after copying to runtime with `scripts/migrate_runtime_files.py`. | Runtime helpers remain canonical. | Medium | Static dataset tests, dataset service tests. | Generated static cache files are runtime artifacts. |
| `src/storage/furnacemind` built-in prompt assets | `packages/furnace-data/furnace_data/assets/furnacemind` | Moved built-in FurnaceMind markdown/YAML source assets. | Runtime-uploaded skill markdown still loads from `runtime/uploads/furnacemind/skills` first; source loaders fall back to the legacy path if needed. | High | FurnaceMind prompt/skill tests, asset structure tests. | Runtime-editable FurnaceMind files stay under `EVONITH_RUNTIME_DIR`; source assets are packaged. |
| `src/storage` and `storage` other runtime-like content | `runtime/cache`, `runtime/feedback`, `runtime/uploads/furnacemind` | Removed tracked source runtime artifacts after copying to runtime. | Runtime helpers remain canonical; old read fallbacks can remain in code while source artifacts are absent. | High | Runtime migration scripts, feedback tests, FurnaceMind tests. | `src/storage/furnacemind` source assets moved earlier; feedback DB/images and summaries are runtime artifacts. |
| `static` | Removed | Removed unreferenced root placeholder/static leftovers. | Frontend assets live under `apps/frontend_streamlit/assets`. | Low | Structure tests and frontend smoke if UI changes. | No active code references were found. |
| legacy database migration leftovers | Removed | Removed obsolete root migration folder, root `.ini`, and dependency entries. | No reconciliation was added. | Medium | Structure tests, dependency profile check, deployment tests. | Active ORM/table creation paths remain unchanged. |
| `requirements` | `requirements` | Keep as-is. | Continue aligning profile files with existing dependency groups. | Low | `check_dependency_profiles.py`. | Dependency group changes are explicitly out of scope. |
| `docs/api/openapi-v1.json` | `docs/api/openapi-v1.json` | Keep generated contract baseline. | Regenerate before and after backend moves and diff path/operation compatibility. | Medium | OpenAPI export and old/new equivalence tests. | Baseline export produced no worktree diff. |

## Baseline Commands Before Moving Files

Run these commands before any move/delete/import-change step:

```bash
python scripts/export_backend_openapi.py
python scripts/check_repository_structure.py
python scripts/check_import_boundaries.py
python scripts/check_dependency_profiles.py
python scripts/check_backend_minimal_startup.py
python scripts/check_frontend_api_imports.py
pytest tests -q
```

Local execution note: the bare `python` interpreter on PATH does not have the
project dependencies installed. The first exact command failed with
`ModuleNotFoundError: No module named 'fastapi'`. The meaningful baseline was
therefore run through the repo-managed environment as `uv run python ...` and
`uv run pytest ...`.

| Command | Result | Failure Summary | Pre-existing Or Blocking |
|---|---|---|---|
| `python scripts/export_backend_openapi.py` | Failed in local shell | `ModuleNotFoundError: No module named 'fastapi'` from the PATH Python interpreter. | Environment invocation issue, not a code failure. Use `uv run` or an activated project environment for baseline. |
| `uv run python scripts/export_backend_openapi.py` | Passed | Exported `docs/api/openapi-v1.json`; no worktree diff observed. | Not blocking. |
| `uv run python scripts/check_repository_structure.py` | Passed with warning | Earlier run warned about source runtime artifacts that were removed in the generated-artifact cleanup pass. | Historical baseline note. |
| `uv run python scripts/check_import_boundaries.py` | Passed | None. | Not blocking. |
| `uv run python scripts/check_dependency_profiles.py` | Passed | None. Dependency groups found: `backend-ai`, `backend-base`, `backend-data`, `backend-documents`, `backend-ml`, `backend-vector`, `dev`, `edge`, `frontend`. | Not blocking. |
| `uv run python scripts/check_backend_minimal_startup.py` | Passed | Backend app imported, OpenAPI generated, `/api/v1/health` returned 200, and forbidden heavy startup modules were not loaded. | Not blocking. |
| `uv run python scripts/check_frontend_api_imports.py` | Passed | None. | Not blocking. |
| `uv run pytest tests -q` | Failed | `tests/test_bmo_context_provider.py::test_history_frame_layers_recent_online_context_for_model_lags` expected `history_df.iloc[-1]["ORE_CALC_MT"] == 30.0`, but got `52.869`. Summary: 1 failed, 338 passed, 1 skipped, 5 warnings in 185.31s. | Pre-existing baseline failure because no source changes had been made. Blocking for a fully green move gate unless accepted as a known failure; not blocking this planning-only document. |

## Do Not Delete Yet

Direct-mode source modules:

- `src/custom_pages`
- `src/data`
- `src/domain`
- `src/utils`
- `src/ui`
- `src/services`
- `src/agents`
- `src/reports`
- `src/plotters`
- `src/geometries`

Compatibility shims:

- `furnace-data-service/app/main.py`
- `apps/backend_api/app/__init__.py`
- `src/app.py`
- `apps/frontend_streamlit/_legacy.py`
- `apps/frontend_streamlit/__init__.py`
- `src/custom_pages/*.py`
- `src/services/*.py`
- `src/config/{config_loader,frontend_settings,page_registry}.py`
- `src/ui/**/*.py`
- `apps/frontend_streamlit/{assets,config,ui,utils}/__init__.py`
- `furnace_data/__init__.py`
- `furnace_data/runtime_paths.py`
- `furnace-data-service/run.py`

Legacy route modules:

- `apps/backend_api/app/routes`
- v1 wrappers that call legacy route code, especially
  `apps/backend_api/app/api/v1/routes/datasets.py`
- `EVONITH_ENABLE_LEGACY_ROUTES` settings and tests

Model files still used by tests/runtime:

- `packages/furnace-data/furnace_data/assets/models/**`
- `packages/furnace-data/furnace_data/assets/models/bmo_fuel/**`
- any external model directory selected by `EVONITH_MODEL_DIR`
- compatibility resolver support for old-style `src/assets/models/...` paths
- test fixtures that reference `src/assets/models/...`
- model config keys in `src/config/setting_bmo.yml`
- model config keys in `src/config/setting_vsense.yml`

Runtime migration and runtime operation scripts:

- `scripts/migrate_runtime_files.py`
- `scripts/migrate_feedback_tickets.py`
- `scripts/bootstrap_runtime.py`
- `scripts/backup_runtime.py`
- `scripts/restore_runtime.py`
- `scripts/validate_deployment.py`
- `scripts/smoke_test_deployment.py`
- `scripts/validate_api_cutover.py`
- `scripts/verify_release_readiness.py`

Deployment scripts and examples:

- `scripts/edge_start_backend.sh`
- `scripts/edge_start_frontend.sh`
- `infra/env/*.example`
- `infra/systemd/*.service.example`
- `infra/nginx/evonith.conf.example`
- `infra/caddy/Caddyfile.example`
- deployment docs under `docs/deployment`

Files and folders whose usage is unclear or mixed:

- `storage`
- empty legacy `src/assets` subdirectories, if present in a local checkout
- `furnace-data-service/data`

## Risk List

- Backend package alias inversion is risky. Today canonical backend imports
  resolve legacy modules by extending `apps.backend_api.app.__path__`; after
  moving files, legacy `app.*` imports must continue to work.
- `app` is a top-level package name for the backend, while `src/app.py` is the
  legacy frontend entrypoint. `apps/backend_api/app/__init__.py` currently
  handles a partial-module cache edge case. Preserve this behavior during
  migration.
- Legacy routes are still part of supported rollback and compatibility. Do not
  remove `apps/backend_api/app/routes` or disable
  `EVONITH_ENABLE_LEGACY_ROUTES` during structure cleanup.
- Direct-mode fallback remains supported. Do not delete `src` modules just
  because canonical wrappers exist.
- `FURNACE_CONFIG_DIR` currently defaults to `src/config`, and config files are
  shared across frontend, backend, and direct-mode code. Split ownership before
  moving config.
- Model path compatibility is still required for tests and older config values.
  `ModelRegistryService` defaults to packaged models when `EVONITH_MODEL_DIR`
  is not set, while model bundle loaders still understand old-style
  `src/assets/models/...` paths.
- Source-tree runtime artifacts have been removed and ignored. Keep
  `scripts/migrate_runtime_files.py` until older checkouts no longer need a
  migration bridge into `EVONITH_RUNTIME_DIR`.
- FurnaceMind source prompt assets now live under the shared package; runtime
  uploaded skills and generated MRAG assets must remain under
  `EVONITH_RUNTIME_DIR`.
- Dependency groups and profile requirement files are stable Phase 11/13
  surfaces. Do not change them during file movement.
- The full `uv run pytest tests -q` suite is green as of this cleanup pass.
- Service-local tests have been migrated into the canonical `tests/backend`
  tree. The old `furnace-data-service/test` and `furnace-data-service/tests`
  directories are deprecated README placeholders only.
- Deployment docs and scripts include both canonical and rollback commands.
  Update them only in the same phase as the path they reference.

## Prompt 8 Test Consolidation

- Root-level backend/domain tests were moved under `tests/backend`.
- `furnace-data-service/tests` was moved to `tests/backend/service_api`.
- `furnace-data-service/test` was moved to `tests/backend/legacy_service`.
- `tests/fixtures` was created for shared lightweight fixtures.
- Root pytest discovery now points to `tests`; the service pyproject points to
  `../tests/backend` for compatibility when invoked from the service folder.
- The old service test roots intentionally contain only deprecation READMEs.
- Bare `pytest tests/structure -q` is unavailable on this PowerShell PATH;
  validation used `uv run pytest ...`.
- `uv run pytest tests/structure -q` passed with 71 tests and 1 warning.
- `uv run pytest tests/backend -q` passed with 387 tests, 1 skipped, and 8 warnings.
- `uv run pytest tests/frontend -q` passed with 78 tests.
- `uv run pytest tests/dependency -q` passed with 8 tests.
- `uv run pytest tests/deployment -q` passed with 9 tests and 1 skipped.
- `uv run pytest tests/integration -q` passed with 7 tests and 4 warnings.
- `uv run pytest tests -q` passed with 560 tests, 2 skipped, and 11 warnings.

## Prompt 9 Dependency Cleanup

- Root `[project.dependencies]` was reduced to the editable shared package only.
- Named dependency groups remain split by runtime concern: `backend-base`,
  `backend-data`, `backend-ml`, `backend-ai`, `backend-vector`,
  `backend-documents`, `frontend`, `dev`, and `edge`.
- `dev` now includes the app/test dependency groups through dependency-group
  includes, while backend/frontend/optional groups remain separate.
- `requirements.txt` is retained as a documented full-dev convenience pointer
  to `requirements/dev.txt`; targeted installs should use explicit profile
  files under `requirements/`.
- `requirements/backend-base.txt` now installs the canonical shared package from
  `./packages/furnace-data`.
- `uv.lock` was refreshed after removing the old default full-stack dependency
  surface. The lock no longer carries removed legacy default-only stacks such as
  LangChain, Chroma, ONNX Runtime, and Alembic.
- `scripts/check_dependency_profiles.py` now validates the canonical app/package
  paths, dependency-group includes, isolated optional groups, root requirements
  delegation, backend-base exclusions, frontend Streamlit ownership, and Alembic
  absence across active dependency files.
- Added `tests/dependency/test_dependency_cleanup.py` to pin backend-base,
  frontend, dev, edge, optional-group, Alembic, and checker behavior.

Prompt 9 validation results:

| Command | Result | Notes |
|---|---|---|
| `python scripts/check_dependency_profiles.py` | Passed | Found all required groups and slim default dependency metadata. |
| `python scripts/check_backend_minimal_startup.py` | Failed in bare shell | Bare `python` environment does not have FastAPI installed (`No module named 'fastapi'`). Not a dependency cleanup blocker; managed env passed. |
| `uv run python scripts/check_backend_minimal_startup.py` | Passed | `/api/v1/health` returned 200; backend title `Evonith BF Backend API`. |
| `python scripts/check_frontend_api_imports.py` | Passed | Frontend API import boundary check remains green. |
| `pytest tests/dependency -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Not a cleanup blocker; managed env passed. |
| `uv run pytest tests/dependency -q` | Passed | 18 passed. |
| `uv run pytest tests -q` | Passed | 570 passed, 2 skipped, 11 warnings. |

## Prompt 10 Active Documentation Cleanup

- Rewrote `README.md` around the canonical backend, frontend, shared package,
  runtime, dependency profiles, local startup commands, tests, deployment links,
  and temporary compatibility surfaces.
- Rewrote `CLAUDE.md` as current repository guidance for coding agents, with
  canonical commands and explicit compatibility-shim boundaries.
- Created `docs/README.md` as the active documentation index for deployment,
  testing, operations, API, architecture/restructure, and migration archive
  references.
- Updated `.devcontainer/devcontainer.json` to open canonical app files, install
  through `uv`, bootstrap `runtime`, and start canonical backend/frontend
  commands.
- Updated active deployment and testing docs so canonical commands are primary
  and old frontend/backend commands appear only as temporary compatibility or
  rollback paths.
- Archived detailed phase implementation docs and old phase test guides under
  `docs/archive/migration-history/`; active `docs/migration` now retains only
  the post-phase-13 cleanup plan, and active `docs/testing` retains the current
  Phase 13 testing guide.
- Updated release-readiness required-doc checks to use the active docs index,
  current guides, cleanup plan, and archived phase-13 history copies.
- Added `tests/structure/test_docs_current_paths.py` to guard canonical README
  paths, active-doc stale path references, compatibility-command context,
  archived migration history, and obvious committed secret patterns.

Prompt 10 validation results:

| Command | Result | Notes |
|---|---|---|
| `uv run pytest tests/structure/test_docs_current_paths.py -q` | Passed | 8 passed. |
| `pytest tests/structure -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/structure -q` | Passed | 79 passed, 1 warning. |
| `python scripts/verify_release_readiness.py --allow-dirty --skip-tests` | Passed | Working tree dirty was allowed; required docs, secret scan, structure/import/dependency/deployment checks passed. |
| `pytest tests -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests -q` | Passed | 578 passed, 2 skipped, 11 warnings. |

## Prompt 11 Legacy Shim Reduction

Rollback checkpoint before this cleanup: `7c6dda2 chore: checkpoint before legacy shim reduction`.

Prompt 11 reduced only legacy paths that were proven duplicated by canonical locations:

| Legacy Path | Decision | Reason | Status |
|---|---|---|---|
| `furnace-data-service/app` | Reduced to shim-only package | Backend implementation lives in `apps/backend_api/app`; old service entrypoint must still import. | Kept `__init__.py` and `main.py`; removed duplicate empty legacy subdirectories and generated caches. |
| Root `furnace_data` | Reduced to import shim only | Canonical shared package lives in `packages/furnace-data/furnace_data`; root imports remain compatibility surface. | Kept `__init__.py` and `runtime_paths.py`; removed stale duplicate root package metadata. |
| `src/app.py` | Kept as Streamlit compatibility shim | `streamlit run src/app.py` remains a documented temporary rollback command. | No page business logic should be added here. |
| `src/services` | Kept as re-export wrappers | Old frontend adapter imports are still tested and supported temporarily. | Wrappers re-export `apps.frontend_streamlit.services.*`. |
| `src/custom_pages` | Kept as page delegation wrappers | Old page imports and rollback checks still need the paths. | Wrappers call canonical page files under `apps/frontend_streamlit/custom_pages`. |
| `src/config` and `src/ui` | Kept as compatibility wrappers plus shared config surface | Some direct-mode/frontend code still imports these paths. | Review shared YAML ownership before any later reduction. |
| `src/data`, `src/domain`, `src/utils`, `src/agents`, `src/reports`, `src/plotters`, `src/geometries` | Kept temporarily | These contain active direct-mode fallback and domain logic, not proven duplicate code. | Do not delete without a dedicated direct-mode retirement plan. |

Structure enforcement was strengthened so legacy backend and root shared-package paths may contain only small compatibility shims, and old frontend wrapper locations may contain only wrapper/delegation files. Generated `__pycache__` and `.pyc` files remain ignored and should not be committed.

Uncertain leftovers for manual review:

- Shared YAML under `src/config` still has mixed frontend/direct-mode ownership.
- Direct-mode source modules under `src/data`, `src/domain`, `src/utils`, `src/agents`, `src/reports`, `src/plotters`, and `src/geometries` are still active and intentionally retained.
- Legacy backend route modules under `apps/backend_api/app/routes` remain supported by compatibility requirements.
- `run_streamlit.py` remains a frontend launcher compatibility helper.

Prompt 11 validation results:

| Command | Result | Notes |
|---|---|---|
| `python scripts/check_repository_structure.py` | Passed | Canonical backend, frontend, shared package, runtime, generated-artifact, and shim-reduction checks passed. |
| `python scripts/check_import_boundaries.py` | Passed | Backend/frontend import boundaries remain clean. |
| `python scripts/check_dependency_profiles.py` | Passed | Required dependency groups and backend-base exclusions remain valid. |
| `python scripts/check_backend_minimal_startup.py` | Failed in bare shell | Bare `python` environment does not have FastAPI installed (`No module named 'fastapi'`). Managed env passed. |
| `uv run python scripts/check_backend_minimal_startup.py` | Passed | `/api/v1/health` returned 200; backend title `Evonith BF Backend API`. |
| `python scripts/check_frontend_api_imports.py` | Passed | Canonical frontend adapters and old wrappers import cleanly. |
| `python scripts/export_backend_openapi.py` | Failed in bare shell | Bare `python` environment does not have FastAPI installed (`No module named 'fastapi'`). Managed env passed. |
| `uv run python scripts/export_backend_openapi.py` | Passed | Exported `docs/api/openapi-v1.json`. |
| `pytest tests/structure -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/structure -q` | Passed | 89 passed, 1 warning. |
| `pytest tests/frontend -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/frontend -q` | Passed | 78 passed. |
| `pytest tests/backend -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/backend -q` | Passed | 387 passed, 1 skipped, 8 warnings. |
| `pytest tests/integration -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests/integration -q` | Passed | 7 passed, 4 warnings. |
| `pytest tests -q` | Failed in bare shell | `pytest` is not on this PowerShell PATH. Managed env passed. |
| `uv run pytest tests -q` | Passed | 588 passed, 2 skipped, 11 warnings. |

## Suggested Move Gates

1. Re-run the baseline before the next source move.
2. Audit remaining shared YAML config ownership before any config relocation.
3. Update scripts, infra, docs, and tests to canonical paths as each remaining
   model/config/runtime path moves.
4. Only after compatibility telemetry/tests are stable, plan a separate
   deprecation phase for deleting shims, direct-mode fallback, and legacy
   routes.
