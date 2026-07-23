from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from furnace_data.influx.query import query_builder
from furnace_data.vboard.catalog import (
    heatload_fields_for_row,
    heatload_fields_for_rows,
    load_vboard_catalog,
)
from furnace_data.vboard.transforms import (
    transform_heatload_contour,
    transform_heatload_timeseries,
    transform_temperature_contour,
)


def test_catalog_levels_groups_and_public_shape():
    catalog = load_vboard_catalog()

    level = next(item for item in catalog["temperature_levels"] if item["id"] == "6795")
    assert level["elevation_m"] == 6.795
    assert level["label"] == "6.795 m"
    known_levels = {item["id"] for item in catalog["temperature_levels"]}
    assert all(
        level_id in known_levels
        for group in catalog["circumferential_temperature_groups"]
        for level_id in group["level_ids"]
    )
    serialized = json.dumps(catalog).lower()
    assert ".pkl" not in serialized
    assert "runtime" not in serialized
    assert "bucket" not in serialized
    assert "token" not in serialized
    assert catalog["display"]["heatload_unit"] is None
    assert catalog["display"]["heatload_label"] == "Heat-load index"


def test_selected_heatload_fields_are_allowlisted_and_unknown_fields_fail():
    assert heatload_fields_for_row("R6") == [
        "heat_load_r6_q1",
        "heat_load_r6_q2",
        "heat_load_r6_q3",
        "heat_load_r6_q4",
    ]
    assert len(heatload_fields_for_rows()) == 20

    start = datetime(2026, 7, 23, tzinfo=timezone.utc)
    stop = datetime(2026, 7, 23, 1, tzinfo=timezone.utc)
    query = query_builder(
        "heatload_delta_t",
        start,
        stop,
        type="windowed-average",
        window_by="1m",
        selected_fields=heatload_fields_for_row("R6"),
    )

    assert 'MEAN("heat_load_r6_q1")' in query
    assert "heat_load_r7_q1" not in query
    with pytest.raises(ValueError):
        query_builder(
            "heatload_delta_t",
            start,
            stop,
            type="ts",
            selected_fields=["heat_load_r6_q1; DROP"],
        )


def test_temperature_transform_converts_non_finite_values_to_null():
    frame = pd.DataFrame(
        {
            "temp_4373_a_mean": [np.nan],
            "temp_4373_a_min": [np.inf],
            "temp_4373_a_max": [400.0],
        }
    )

    result = transform_temperature_contour(frame)
    level = next(item for item in result["levels"] if item["level_id"] == "4373")

    assert any(quadrant["mean"] is None for quadrant in level["quadrants"])
    assert result["status"] in {"partial", "empty"}


def test_heatload_contour_clips_index_and_does_not_zero_missing_values():
    frame = pd.DataFrame(
        {
            "heat_load_r6_q1_mean": [0.2],
            "heat_load_r6_q1_min": [0.1],
            "heat_load_r6_q1_max": [1.5],
            "heat_load_r6_q2_mean": [np.nan],
            "heat_load_r6_q2_min": [np.nan],
            "heat_load_r6_q2_max": [np.nan],
            "heat_load_r6_q3_mean": [0.6],
            "heat_load_r6_q3_min": [0.3],
            "heat_load_r6_q3_max": [0.9],
            "heat_load_r6_q4_mean": [-1.0],
            "heat_load_r6_q4_min": [-1.0],
            "heat_load_r6_q4_max": [-1.0],
        }
    )

    result = transform_heatload_contour(frame)
    row = next(item for item in result["rows"] if item["row_id"] == "R6")
    q1 = row["quadrants"][0]
    q4 = row["quadrants"][3]

    assert q1["maximum"] == 1.0
    assert q4["mean"] is None


def test_heatload_timeseries_sorts_deduplicates_bounds_and_returns_nulls():
    frame = pd.DataFrame(
        {
            "time": [
                "2026-07-23T00:02:00Z",
                "2026-07-23T00:00:00Z",
                "2026-07-23T00:00:00Z",
            ],
            "heat_load_r6_q1": [0.2, 0.1, 0.4],
            "heat_load_r6_q2": [np.inf, 0.2, 0.3],
            "heat_load_r6_q3": [2.0, 0.0, 0.2],
            "heat_load_r6_q4": [np.nan, np.nan, np.nan],
        }
    )

    result = transform_heatload_timeseries(
        frame,
        row_id="R6",
        resolved_window_seconds=60,
        max_points_per_quadrant=2,
    )

    q1 = result["series"][0]
    q4 = result["series"][3]
    assert q1["returned_points"] == 2
    assert q1["points"][0]["timestamp"] == "2026-07-23T00:00:00Z"
    assert q4["points"][0]["value"] is None
    assert result["downsampled"] is False
    assert result["processing"]["normalization"] == "legacy_compatible"
