# Dependency Profiles

Phase 11 defined install profiles. After the canonical backend, frontend,
shared-package, and test moves, those profiles are now the source of truth for
installing backend, frontend, optional, development, and edge dependencies.

## Package Management

The repository uses `uv` with `pyproject.toml` and `uv.lock`. The root
`[project.dependencies]` list is intentionally slim and contains only the
editable shared package dependency. Heavy optional stacks are isolated in named
dependency groups and are not installed by backend-base or default project
metadata.

The shared `furnace_data` package is editable from `./packages/furnace-data`.
The root `furnace_data` directory remains a temporary compatibility shim for
repo-root imports.

`requirements.txt` is retained as a full local development convenience pointer
to `requirements/dev.txt`. Production, edge, backend-only, and frontend-only
installs should use the explicit profile files in `requirements/` or the
matching `uv` groups.

Preferred local metadata checks:

```bash
uv run python scripts/check_dependency_profiles.py
uv run python scripts/check_import_boundaries.py
```

## Profiles

| Profile | Purpose | Includes | Excludes by default |
|---|---|---|---|
| `backend-base` | FastAPI backend import/start profile | FastAPI, Uvicorn, Pydantic, settings, auth hashing, SQLAlchemy, pandas/numpy, shared package | Streamlit, OpenAI/Anthropic, Qdrant, torch, sentence-transformers, xgboost |
| `backend-data` | Backend data connectors | InfluxDB client, PostgreSQL driver, pyarrow/openpyxl | AI/vector/local LLM |
| `backend-ml` | Optional model/compute support | scikit-learn, scipy, joblib, xgboost | Provider SDKs and vector stores |
| `backend-ai` | Optional provider LLM calls and orchestration | OpenAI, Anthropic, LangGraph, tiktoken | Local LLM stack |
| `backend-vector` | Optional vector/memory support | qdrant-client, sentence-transformers, torch, voyageai | Enabled features remain off unless configured |
| `backend-documents` | Optional document extraction | PyMuPDF, python-docx, python-pptx, pypdf | OCR stack |
| `frontend` | Streamlit UI and API adapters | Streamlit, plotting/UI libraries, HTTP clients | Backend server internals, DB/vector/LLM/model packages |
| `dev` | Full local development/testing | Test/formatting tools plus backend, frontend, and optional feature groups through group includes | Production secrets; not intended for production images |
| `edge` | Shared Jetson/Pi backend profile | backend-base plus selected data/ML packages | AI/vector/local LLM and Streamlit; Pi selects CPU in its environment |

## Install Examples

Use `uv` dependency groups for metadata-aware installs:

```bash
uv sync --no-dev --group backend-base
uv sync --no-dev --group frontend
uv sync --group dev
uv sync --no-dev --group edge
```

For a smaller profile install without using the full-dev root aggregate, use
requirements profiles:

```bash
uv pip install -r requirements/backend-base.txt
uv pip install -r requirements/frontend.txt
uv pip install -r requirements/edge.txt
```

Backend with data and compute:

```bash
uv pip install -r requirements/backend-data.txt
uv pip install -r requirements/backend-ml.txt
```

Backend with optional AI/vector features:

```bash
uv pip install -r requirements/backend-ai.txt
uv pip install -r requirements/backend-vector.txt
```

Optional AI/vector packages remain lazy. Installing a profile does not enable a
feature by itself; the corresponding `EVONITH_*` feature flags and provider
configuration must still be set.

## Known Limitations

- `requirements.txt` is a full-dev convenience aggregate. It intentionally pulls
  in optional AI/vector/document/frontend packages through `requirements/dev.txt`.
- The default root project dependencies are deliberately slim and should stay
  free of Streamlit, provider SDKs, vector stores, document parsers, and model
  runtimes.
- `furnace_data` still carries shared data/domain dependencies, but its
  canonical source tree is `packages/furnace-data/furnace_data`.
- A production image should install only the selected requirements profile or
  `uv` group, for example `backend-base`, `frontend`, or `edge`.
