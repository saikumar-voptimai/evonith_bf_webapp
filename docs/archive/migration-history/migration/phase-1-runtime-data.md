# Phase 1 Runtime Data Audit

Phase 1 separates generated runtime data from source-controlled code without
splitting the Streamlit app and FastAPI sidecar.

## Runtime Directory

`EVONITH_RUNTIME_DIR` controls generated runtime files. If it is not set, the
default is `./runtime` at the repository root.

Local Streamlit:

```bash
EVONITH_RUNTIME_DIR=./runtime streamlit run src/app.py
```

Backend sidecar:

```bash
cd furnace-data-service
EVONITH_RUNTIME_DIR=../runtime uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Edge production:

```bash
EVONITH_RUNTIME_DIR=/var/lib/evonith-bf
```

Runtime data must not be committed. Do not store secrets in runtime files unless
that is explicitly required and the files are secured.

## Implemented Mappings

| Old path | Purpose | New runtime path | Files/modules affected | Implemented |
| --- | --- | --- | --- | --- |
| `src/storage/feedback/tickets.db` | Feedback ticket SQLite DB | `runtime/feedback/tickets.db` | `src/data/tickets/engine.py` | Yes |
| `src/storage/feedback/images/` | Feedback screenshot uploads | `runtime/uploads/feedback/` | `src/data/tickets/service.py`, `src/utils/feedback_page.py` | Yes |
| `src/storage/*_summaries.json` | FurnaceMind generated summaries | `runtime/cache/` | `src/agents/memory/structured_store.py` | Yes |
| `furnace-data-service/data/results/` | Sidecar task result CSVs | `runtime/datasets/results/` | `furnace-data-service/app/config.py`, `app/tasks/task_manager.py` | Yes |
| `furnace-data-service/data/static/` | Sidecar generated static dataset cache (`cache_meta.json`, `ml_dataset*.csv`) | `runtime/datasets/static/` | `furnace-data-service/app/config.py`, `furnace_data.dataset.static` | Yes |
| `src/assets/data/furnace_dataset.csv` writes | Streamlit-side generated static dataset cache | `runtime/datasets/static/furnace_dataset.csv` | `src/data/ml/static_csv.py`, `src/data/ml/static_dataset_manager.py` | Yes |
| `logs/app.log` | Streamlit rotating log file | `runtime/logs/app.log` | `src/utils/logger.py` | Yes |
| `src/agents/tool_errors.md` | FurnaceMind tool error log | `runtime/logs/tool_errors.md` | `src/agents/furnace_tools.py` | Yes |
| `src/storage/furnacemind/mrag_images/` | Generated MRAG visual chunks | `runtime/uploads/furnacemind/mrag_images/` | `src/agents/multimodal/ingestion.py` | Yes |
| `src/storage/furnacemind/<uploaded skill>.md` | Uploaded custom skill context files | `runtime/uploads/furnacemind/skills/` | `src/utils/furnacemind/skill_ui.py`, skill registry/vector store | Yes |
| `src/geometries/mask_*.pkl` new writes | Generated contour mask cache | `runtime/cache/geometries/` | `src/geometries/furnace_gen.py` | Yes |
| `source_files/` | Legacy RAG temporary upload staging | `runtime/temp/source_files/` | `src/utils/rag_methods.py` | Yes |
| OS temp PPTX render folders | MRAG slide-render temporary files | `runtime/temp/` | `src/agents/multimodal/parsers.py` | Yes |
| `src/assets/data/control_bounds.json` writes | Operator control-bound overrides | `runtime/cache/control_bounds.json` | `src/utils/recommendations/bounds.py`, `src/custom_pages/4_Recommendations.py` | Yes |
| `src/config/bmo_operator_inputs.yml` writes | BMO operator preferences | `runtime/cache/bmo_operator_inputs.yml` | `src/data/bmo/ore_editor_preferences.py`, `src/custom_pages/9_Blend_Optimizer.py` | Yes |

## Fallbacks Kept

- Feedback DB is copied once from `src/storage/feedback/tickets.db` or
  `storage/feedback/tickets.db` when the new default runtime DB is absent.
- Existing feedback image paths stored in the DB remain readable.
- Summary JSON files are copied once from `src/storage/` into runtime cache when
  the default summary store is initialized and runtime files are absent.
- Sidecar static dataset reads can fall back to
  `furnace-data-service/data/static/ml_dataset.csv`.
- Streamlit static dataset reads can fall back to
  `src/assets/data/furnace_dataset.csv`.
- Built-in FurnaceMind skill markdown remains read from
  `src/storage/furnacemind`; uploaded skill markdown is read from runtime first.
- Existing contour mask pickle files in `src/geometries` remain read fallbacks.
- `src/assets/data/control_bounds.json` and
  `src/config/bmo_operator_inputs.yml` remain read fallbacks.

## Review Later

- `src/custom_pages/4_Recommendations.py` still updates
  `src/config/setting_vsense.yml` for `OPTIM_STEPS`. This is mutable app
  configuration, not a generated cache, and should be reviewed separately.
- `src/data/ml/static_csv.py::update_cutoff_date()` still edits
  `src/config/setting_ds_dv.yml` after dataset extension. This is a source
  configuration mutation and should be redesigned deliberately in a later phase.
- Database-backed Qdrant/PostgreSQL data is not moved in this phase. The runtime
  helper reserves `runtime/qdrant/` for future local Qdrant storage if needed.

## Migration Script

Use the copy-only script to seed runtime from existing legacy artifacts:

```bash
python scripts/migrate_runtime_files.py --dry-run
python scripts/migrate_runtime_files.py
```

Targets are skipped when they already exist. Use `--overwrite` only when
explicitly replacing runtime copies is intended.
