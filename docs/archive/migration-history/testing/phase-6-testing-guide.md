# Phase 6 Feedback API Testing Guide

Run focused backend tests:

```bash
pytest furnace-data-service/tests/test_api_v1_feedback.py -q
```

Run focused frontend adapter tests:

```bash
pytest tests/frontend/test_feedback_api.py tests/frontend/test_api_client.py tests/frontend/test_phase4_feature_flags.py tests/frontend/test_import_boundaries.py -q
```

Run the integration smoke:

```bash
pytest tests/integration/test_phase6_feedback_flow.py -q
```

Run the copy-only migration preview:

```bash
python scripts/migrate_feedback_tickets.py --dry-run
```

Export OpenAPI:

```bash
python scripts/export_backend_openapi.py
```

Full repository check requested for the phase:

```bash
pytest tests -q
```

The full `tests` suite does not include `furnace-data-service/tests`; run both
when validating backend API behavior.
