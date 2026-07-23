"""Regression tests for retired standalone data routes.

The old ``/data/*`` route family accepted physical table names and bypassed the
typed, authenticated v1 Data Explorer boundary. It is intentionally no longer
mounted, even while legacy health/dataset compatibility routes are enabled.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "path",
    [
        "/data/online/measurements",
        "/data/offline/report-types",
        "/data/offline/tables",
    ],
)
def test_unsafe_legacy_data_read_routes_are_not_mounted(client, path: str) -> None:
    assert client.get(path).status_code == 404


@pytest.mark.parametrize(
    "path, payload",
    [
        ("/data/online/fetch", {"measurements": ["process_params"]}),
        ("/data/offline/fetch", {"report_type": "HM_SLAG"}),
        ("/data/rm/live", {"lookback_days": 1}),
    ],
)
def test_unsafe_legacy_data_mutation_or_fetch_routes_are_not_mounted(
    client, path: str, payload: dict[str, object]
) -> None:
    assert client.post(path, json=payload).status_code == 404