from __future__ import annotations

from datetime import date, time

import pytest

from apps.frontend_streamlit.services.vboard_adapters import (
    circumferential_temperature_groups,
    longitudinal_temperature_arrays,
)
from apps.frontend_streamlit.services.vboard_page_helpers import (
    absolute_time_range_from_inputs,
    build_heatload_timeseries_figure,
    request_fingerprint,
)


def test_absolute_time_range_payload_is_ist_aware():
    payload = absolute_time_range_from_inputs(
        date(2026, 7, 23),
        time(8, 0),
        date(2026, 7, 23),
        time(10, 0),
    )

    assert payload == {
        "kind": "absolute",
        "start": "2026-07-23T08:00:00+05:30",
        "end": "2026-07-23T10:00:00+05:30",
    }


def test_absolute_time_range_rejects_start_after_end():
    with pytest.raises(ValueError):
        absolute_time_range_from_inputs(
            date(2026, 7, 23),
            time(10, 0),
            date(2026, 7, 23),
            time(8, 0),
        )


def test_request_fingerprint_is_stable():
    left = {"b": 2, "a": {"c": 1}}
    right = {"a": {"c": 1}, "b": 2}

    assert request_fingerprint(left) == request_fingerprint(right)


def test_adapters_preserve_6795_label_and_title_data_lengths():
    catalog = {
        "temperature_levels": [
            {"id": "4373", "label": "4.373 m"},
            {"id": "6795", "label": "6.795 m"},
        ],
        "circumferential_temperature_groups": [
            {"id": "mid", "title": "Mid", "level_ids": ["6795"]},
        ],
    }
    contours = {
        "temperature": {
            "levels": [
                {
                    "level_id": "4373",
                    "quadrants": [
                        {"quadrant_id": f"Q{idx}", "mean": 1.0, "minimum": None, "maximum": 2.0}
                        for idx in range(1, 5)
                    ],
                },
                {
                    "level_id": "6795",
                    "quadrants": [
                        {"quadrant_id": f"Q{idx}", "mean": None, "minimum": None, "maximum": None}
                        for idx in range(1, 5)
                    ],
                },
            ]
        }
    }

    means, maxima, minima = longitudinal_temperature_arrays(contours)
    groups = circumferential_temperature_groups(catalog, contours)

    assert means[0] == [1.0, None]
    assert maxima[0] == [2.0, None]
    assert minima[0] == [None, None]
    assert groups[0]["titles"] == ["At 6.795 m"]
    assert len(groups[0]["titles"]) == len(groups[0]["field_values"])


def test_heatload_timeseries_figure_uses_one_shared_y_axis():
    result = {
        "row": {"id": "R6"},
        "display_label": "Heat-load index",
        "unit": None,
        "series": [
            {
                "quadrant_id": f"Q{idx}",
                "points": [{"timestamp": "2026-07-23T00:00:00Z", "value": 0.1 * idx}],
            }
            for idx in range(1, 5)
        ],
    }

    fig = build_heatload_timeseries_figure(result)
    layout = fig.to_dict()["layout"]

    assert len(fig.data) == 4
    assert "yaxis" in layout
    assert "yaxis2" not in layout
    assert all(getattr(trace, "yaxis", None) is None for trace in fig.data)
    assert layout["yaxis"]["title"]["text"] == "Heat-load index"
