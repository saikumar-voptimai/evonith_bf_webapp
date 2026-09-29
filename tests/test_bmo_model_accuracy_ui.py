from __future__ import annotations

from pathlib import Path

import pandas as pd

from ui.bmo.model_accuracy import (
    TRACKING_DISPLAY_DAYS,
    _band_outline,
    _break_gaps,
    _coke_tracking_figure,
    _paired_chart,
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
    assert [trace.name for trace in figure.data] == ["Prediction", "Actual"]
    assert figure.data[-1].mode == "markers"


def test_uncertainty_band_is_one_closed_shape_per_unbroken_run() -> None:
    index = pd.date_range("2026-09-24", periods=5, freq="h", tz="UTC")
    centre = pd.Series([300.0, 302.0, float("nan"), 304.0, 306.0], index=index)

    xs, ys = _band_outline(index, centre, band=5.0)

    # Two runs either side of the gap, each closed and None-terminated, so the
    # fill can never bridge the missing hour with a wedge.
    assert xs.count(None) == ys.count(None) == 2
    first = ys[: ys.index(None)]
    assert first == [305.0, 307.0, 297.0, 295.0, 305.0]


def test_missing_days_become_gaps_not_straight_lines() -> None:
    frame = pd.DataFrame(
        {"p": [1.0, 2.0, 3.0, 4.0]},
        index=pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-22", "2026-09-06"]),
    )

    filled = _break_gaps(frame)

    assert len(filled) == 18
    assert filled["p"].isna().sum() == 14


def test_tracking_band_uses_unseen_day_error_and_keeps_legend_order() -> None:
    index = pd.date_range("2026-09-24", periods=4, freq="h", tz="UTC")
    plotted = pd.DataFrame(
        {
            "raw_predicted_coke_kg_per_thm": [320.0, 321.0, 319.0, 322.0],
            "actual_coke_kg_per_thm": [318.0, 323.0, 317.0, 324.0],
        },
        index=index,
    )

    figure = _coke_tracking_figure(plotted, band=5.6)

    band = figure.data[0]
    assert band.fill == "toself"
    assert band.name == "±5.6 kg/THM typical error on unseen days"
    assert figure.layout.legend.traceorder == "normal"
    ranks = {trace.name: trace.legendrank for trace in figure.data}
    assert ranks["Prediction"] < ranks["Actual"] < ranks[band.name]
    assert figure.data[-1].mode == "markers"


def test_energy_balance_chart_shares_the_tracking_style() -> None:
    days = pd.date_range("2026-06-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {"corrected": [300.0, 305.0, 298.0, 310.0, 303.0, 301.0],
         "actual_coke": [302.0, 303.0, 300.0, 307.0, 306.0, 299.0]},
        index=days,
    )

    figure = _paired_chart(
        frame,
        predicted_col="corrected",
        actual_col="actual_coke",
        unit="kg/THM",
        band=7.0,
    )

    assert "title" not in figure.to_plotly_json()["layout"]
    names = [trace.name for trace in figure.data]
    assert names == ["±7.0 kg/THM typical error", "Predicted", "Measured"]
    assert figure.data[1].line.color == "#07827f"
    # Prediction is a dashed line through its own points.
    assert figure.data[1].mode == "lines+markers"
    assert figure.data[1].line.dash == "dash"
    assert figure.data[2].marker.color == "#25344b"
