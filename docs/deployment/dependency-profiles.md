# Dependency Profiles

Phase 11 defines install profiles without removing the legacy full development
environment. Phase 12 keeps those profiles while moving canonical startup to
`apps.backend_api.app.main:app` and `apps/frontend_streamlit/app.py`.

## Package Management

The repository uses `uv` with `pyproject.toml` and `uv.lock`. The root
`pyproject.toml` still contains the legacy full/dev dependency set for backward
compatibility, and Phase 11 adds named dependency groups plus requirements
profile files for smaller installs.

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
| `backend-ai` | Optional provider LLM calls | OpenAI, Anthropic, tiktoken | Local LLM stack |
| `backend-vector` | Optional vector/memory support | qdrant-client, sentence-transformers, torch, voyageai | Enabled features remain off unless configured |
| `backend-documents` | Optional document extraction | PyMuPDF, python-docx, python-pptx, pypdf | OCR stack |
| `frontend` | Streamlit UI and API adapters | Streamlit, plotting/UI libraries, HTTP clients | Backend server internals, DB/vector/LLM/model packages |
| `dev` | Full local development/testing | Test and formatting tools plus profile files | Production secrets |
| `edge` | Conservative edge backend profile | backend-base plus selected data/ML packages | AI/vector/local LLM and Streamlit |

## Install Examples

Use `uv` dependency groups for metadata-aware installs:

```bash
uv sync --group backend-base
uv sync --group frontend
uv sync --group dev
```

For a smaller profile install without relying on the legacy root dependency
list, use requirements profiles:

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

- The root `dependencies` list remains a legacy full/dev install surface after
  Phase 12. Later packaging phases can slim wheel metadata further.
- `furnace_data` still carries shared data/domain dependencies for compatibility.
- A true production image should install only the selected requirements profile
  or use a later packaging phase to slim wheel metadata further.
