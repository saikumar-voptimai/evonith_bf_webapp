"""Tests for JSON-safe DataFrame preview serialization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from apps.backend_api.app.services.serialization import dataframe_to_preview


def test_dataframe_serialization_handles_timestamp_nan_and_numpy_values():
    df = pd.DataFrame(
        {
            "ts": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "nan": [np.nan],
            "count": [np.int64(7)],
            "ratio": [np.float64(1.5)],
            "flag": [np.bool_(True)],
        }
    )

    columns, rows, row_count, truncated = dataframe_to_preview(df, limit=5)

    assert row_count == 1
    assert truncated is False
    assert [column.name for column in columns] == ["ts", "nan", "count", "ratio", "flag"]
    assert rows[0]["ts"].startswith("2026-01-01T00:00:00")
    assert rows[0]["nan"] is None
    assert rows[0]["count"] == 7
    assert rows[0]["ratio"] == 1.5
    assert rows[0]["flag"] is True
