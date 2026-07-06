# Test Layout

The canonical test suite lives under `tests/`.

- `tests/backend`: backend API, backend services, runtime path, data/domain, security/redaction, and legacy backend compatibility coverage.
- `tests/frontend`: Streamlit app, page registry, UI helpers, and frontend API adapter coverage.
- `tests/integration`: cross-layer Phase 6-13 workflow and runtime-profile coverage.
- `tests/dependency`: dependency profile and script validation coverage.
- `tests/structure`: repository layout, compatibility shim, import boundary, generated artifact, and migration guard coverage.
- `tests/deployment`: deployment script and edge/local staging validation coverage.
- `tests/fixtures`: shared lightweight test fixtures only.

Common commands:

```bash
uv run pytest tests -q
uv run pytest tests/backend -q
uv run pytest tests/frontend -q
uv run pytest tests/structure -q
uv run pytest tests/dependency -q
uv run pytest tests/deployment -q
uv run pytest tests/integration -q
```

Compatibility shim tests remain active. Backend shim coverage is grouped with
`tests/backend` and `tests/structure` so old `app.*` and
`furnace-data-service/app/main.py` imports stay protected while canonical code
lives under `apps/backend_api/app`.
