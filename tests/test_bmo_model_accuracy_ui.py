from __future__ import annotations

from pathlib import Path

import pandas as pd

from ui.bmo.model_accuracy import (
    TRACKING_DISPLAY_DAYS,
    _coke_tracking_figure,
    _recent_tracking_window,
)


def test_coke_tracking_window_is_limited_to_latest_three_days() -> None:
    index = pd.date_range("2026-09-20 00:00", periods=121, freq="h", tz="UTC")
    history = pd.DataFrame({"actual_coke_kg_per_thm": range(len(index))}, index=index)

    plotted = _recent_tracking_window(history)

    assert TRACKING_DISPLAY_DAYS == 3
    assert plotted.index.max() == history.index.max()
    assert plotted.index.min() == history.index.max() - pd.Timedelta(days=3)
    assert plotted.index.max() - plotted.index.min() == pd.Timedelta(days=3)


def test_tracking_palette_and_removed_capacity_copy_are_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    accuracy_ui = (root / "src/ui/bmo/model_accuracy.py").read_text(encoding="utf-8")
    page = (root / "src/custom_pages/9_Blend_Optimizer.py").read_text(
        encoding="utf-8"
    )

    assert 'mode="markers"' in accuracy_ui
    assert "_TRACKING_TEAL" in accuracy_ui
    assert "_TRACKING_NAVY" in accuracy_ui
    assert "Charges per hour and tonnes per charge are the only two numbers" not in page


def test_tracking_title_and_legend_have_separate_layout_space() -> None:
    index = pd.date_range("2026-09-24", periods=4, freq="h", tz="UTC")
    plotted = pd.DataFrame(
        {
            "raw_predicted_coke_kg_per_thm": [320.0, 321.0, 319.0, 322.0],
            "corrected_predicted_coke_kg_per_thm": [310.0, 311.0, 309.0, 312.0],
            "actual_coke_kg_per_thm": [308.0, 314.0, 307.0, 313.0],
        },
        index=index,
    )

    figure = _coke_tracking_figure(plotted)

    assert figure.layout.title.text is None
    assert "title" not in figure.to_plotly_json()["layout"]
    assert figure.layout.margin.t >= 60
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.y > 1.0
    assert [trace.name for trace in figure.data] == [
        "Raw prediction",
        "Corrected prediction",
        "Actual",
    ]
    assert figure.data[-1].mode == "markers"
