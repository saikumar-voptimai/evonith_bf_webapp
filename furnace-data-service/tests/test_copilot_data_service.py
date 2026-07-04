"""Tests for Copilot data service."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.config import BackendSettings
from app.services.copilot_data_service import CopilotDataService


def test_copilot_recent_data_from_mock_rows_is_json_safe():
    service = CopilotDataService(settings=BackendSettings(backend_env="test"))

    result = service.fetch_recent_data(
        {
            "source": "input_data",
            "filters": {
                "rows": [
                    {"time": datetime(2026, 7, 4), "value": np.float64(1.25), "bad": math.nan},
                ]
            },
            "limit": 10,
        }
    )

    assert result["row_count"] == 1
    assert result["rows"][0]["value"] == 1.25
    assert result["rows"][0]["bad"] is None
    assert result["summary"]["numeric_stats"]["value"]["latest"] == 1.25


def test_copilot_recent_data_caps_rows_and_empty_warns():
    service = CopilotDataService(
        settings=BackendSettings(backend_env="test", copilot_max_context_rows=1)
    )

    capped = service.fetch_recent_data(
        {
            "source": "input_data",
            "filters": {"rows": [{"x": 1}, {"x": 2}]},
            "limit": 2,
        }
    )
    empty = service.fetch_recent_data({"source": "input_data", "filters": {"rows": []}})

    assert capped["returned_rows"] == 1
    assert capped["truncated"] is True
    assert empty["warnings"][0]["code"] == "COPILOT_DATA_EMPTY"


def test_copilot_data_service_import_does_not_require_influx():
    service = CopilotDataService(settings=BackendSettings(backend_env="test"))
    df = service.dataframe_from_input(pd.DataFrame([{"x": 1}]).to_dict("records"))

    assert list(df.columns) == ["x"]
