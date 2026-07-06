# Phase 6 Test Execution Report

This report records the commands run for the Feedback/Tickets backend API
migration. Update the status entries whenever the suite is re-run.

| Command | Status | Notes |
| --- | --- | --- |
| `pytest furnace-data-service/tests/test_api_v1_feedback.py -q` | Passed | 7 passed, 1 dependency deprecation warning |
| `pytest tests/frontend/test_feedback_api.py tests/frontend/test_api_client.py tests/frontend/test_phase4_feature_flags.py tests/frontend/test_import_boundaries.py -q` | Passed | 28 passed |
| `pytest tests/integration/test_phase6_feedback_flow.py -q` | Passed | 1 passed, 1 dependency deprecation warning |
| `pytest furnace-data-service/tests -q` | Passed | 53 passed, 2 dependency deprecation warnings |
| `pytest tests -q` | Failed before Phase 6 tests ran | Existing collection issues: Streamlit test stub lacks `cache_data`; `utils.shift_windows` import lacks `LOCAL_TIMEZONE` in report tests |
| `python scripts/migrate_feedback_tickets.py --dry-run` | Passed | Found 2 legacy tickets, 6 comments, and 1 attachment to copy; no writes performed |
| `python scripts/export_backend_openapi.py` | Passed | Regenerated `docs/api/openapi-v1.json` |
| `python -m py_compile ...` | Passed | Feedback backend, frontend adapter/page, and migration script compiled |
| `python -c "from furnace_data.runtime_paths import ..."` | Passed | Runtime resolved to local `runtime` and directories ensured |

## Boundary Checks

`rg /api/v1/feedback docs/api/openapi-v1.json -n` confirms OpenAPI contains
the feedback config, tickets, comments, close/reopen, and attachment routes.

`rg "src/storage|furnace-data-service/data/results|src/storage/feedback|storage/feedback" furnace-data-service/app src scripts tests docs -n`
found only documented fallback/migration references plus pre-existing Phase 1
runtime migration helpers. No new active backend feedback write path targets the
old source storage directories.

## Known Phase 6 Scope Limits

- Direct Streamlit ticket mode remains available and is still the default.
- Ticket delete is not exposed through the backend API in this phase; direct
  mode retains existing delete behavior.
- Feedback metadata is SQLite-backed in Phase 6. PostgreSQL support is deferred.
- No Material Balance, Recommendations, Blend Optimizer, AI Copilot, or
  FurnaceMind migration is included.
