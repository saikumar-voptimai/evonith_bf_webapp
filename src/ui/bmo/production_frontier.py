"""Streamlit product surface for the Stochastic Burden Frontier Engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.bmo.production_frontier import (
    MAX_VARIATION_SCENARIOS,
    FrontierSettings,
    run_frontier_simulation,
    validate_frontier_inputs,
)
from utils.bmo.types import (
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceSettings,
)


_RESULT_KEY = "_bmo_sbfe_result"
_SELECTED_PRODUCTION_KEY = "_bmo_sbfe_selected_production_mt"
_FRONTIER_CHART_KEY = "bmo_sbfe_frontier_curve"
_POINT_SLIDER_KEY = "bmo_sbfe_point_slider"
_POINT_PICKER_KEY = "bmo_sbfe_point_picker"
# Fewer varied scenarios than this cannot describe day-to-day spread: with one,
# every target is 0% or 100% feasible and the knee "distribution" is one value.
MIN_VARIATION_SCENARIOS = 5
_KNEE_BLUE = "#5c7cfa"
_FEASIBLE_AMBER = "#f59f00"


def _estimated_remaining_seconds(
    *, elapsed_seconds: float, completed: int, total: int
) -> float | None:
    """Estimate remaining runtime from completed solver iterations."""

    done = max(0, int(completed))
    count = max(0, int(total))
    if done <= 0 or count <= done:
        return 0.0 if count > 0 and done >= count else None
    seconds_per_iteration = max(0.0, float(elapsed_seconds)) / done
    return seconds_per_iteration * (count - done)


def _duration_text(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _input_fingerprint(ores: list[OreInput]) -> str:
    payload = [
        {
            "id": ore.ore_id,
            "stock": ore.stock_mt,
            "price": ore.price_rs_per_mt,
            "min": ore.min_share_pct,
            "max": ore.max_share_pct,
            "fe": ore.chemistry.fe_t_pct,
            "sio2": ore.chemistry.sio2_pct,
            "al2o3": ore.chemistry.al2o3_pct,
        }
        for ore in ores
    ]
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:10]


def _material_editor_frame(ores: list[OreInput]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Use": float(ore.max_share_pct) > 0.0,
                "ore_id": ore.ore_id,
                "Material": ore.display_name,
                "Stock MT": float(ore.stock_mt),
                "Price Rs/MT": float(ore.price_rs_per_mt),
                "Min share %": float(ore.min_share_pct),
                "Max share %": float(ore.max_share_pct),
                "Fe %": float(ore.chemistry.fe_t_pct),
                "SiO2 %": float(ore.chemistry.sio2_pct),
                "Al2O3 %": float(ore.chemistry.al2o3_pct),
            }
            for ore in ores
        ]
    )


def _ores_from_editor(editor: pd.DataFrame, ores: list[OreInput]) -> list[OreInput]:
    by_id = {ore.ore_id: ore for ore in ores}
    selected: list[OreInput] = []
    for _, row in editor.iterrows():
        ore_id = str(row.get("ore_id", ""))
        base = by_id.get(ore_id)
        if base is None or not bool(row.get("Use", False)):
            continue
        chemistry = replace(
            base.chemistry,
            fe_t_pct=float(row.get("Fe %", base.chemistry.fe_t_pct) or 0.0),
            sio2_pct=float(row.get("SiO2 %", base.chemistry.sio2_pct) or 0.0),
            al2o3_pct=float(row.get("Al2O3 %", base.chemistry.al2o3_pct) or 0.0),
        )
        selected.append(
            replace(
                base,
                stock_mt=float(row.get("Stock MT", base.stock_mt) or 0.0),
                price_rs_per_mt=float(
                    row.get("Price Rs/MT", base.price_rs_per_mt) or 0.0
                ),
                min_share_pct=float(
                    row.get("Min share %", base.min_share_pct) or 0.0
                ),
                max_share_pct=float(
                    row.get("Max share %", base.max_share_pct) or 0.0
                ),
                chemistry=chemistry,
            )
        )
    return selected


def _flux_editor_frame(fluxes: list[FluxInput]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "flux_id": flux.flux_id,
                "Flux": flux.display_name,
                "Enabled": bool(flux.enabled),
                "Min MT": float(flux.min_qty_mt),
                "Max MT": float(
                    flux.stock_mt if flux.max_qty_mt is None else flux.max_qty_mt
                ),
                "Stock MT": float(flux.stock_mt),
                "Price Rs/MT": float(flux.price_rs_per_mt),
            }
            for flux in fluxes
            if flux.optimizable
        ]
    )


def _fluxes_from_editor(
    editor: pd.DataFrame, fluxes: list[FluxInput]
) -> list[FluxInput]:
    by_id = {flux.flux_id: flux for flux in fluxes}
    overrides: dict[str, FluxInput] = {}
    for _, row in editor.iterrows():
        flux_id = str(row.get("flux_id", ""))
        base = by_id.get(flux_id)
        if base is None:
            continue
        overrides[flux_id] = replace(
            base,
            enabled=bool(row.get("Enabled", base.enabled)),
            min_qty_mt=float(row.get("Min MT", base.min_qty_mt) or 0.0),
            max_qty_mt=float(row.get("Max MT", base.max_qty_mt) or 0.0),
            stock_mt=float(row.get("Stock MT", base.stock_mt) or 0.0),
            price_rs_per_mt=float(
                row.get("Price Rs/MT", base.price_rs_per_mt) or 0.0
            ),
        )
    return [overrides.get(flux.flux_id, flux) for flux in fluxes]


def _optional_limit(value: float) -> float | None:
    return float(value) if float(value) > 0.0 else None


def _base_knees(result: dict[str, Any]) -> list[dict[str, Any]]:
    knees = result.get("base_knees", []) or []
    if knees:
        return [dict(item) for item in knees]
    scenarios = result.get("scenarios", pd.DataFrame())
    if isinstance(scenarios, pd.DataFrame) and not scenarios.empty:
        base = scenarios.loc[scenarios["is_base"]]
        if not base.empty:
            fallback = base.iloc[0].get("knee_candidates", []) or []
            if isinstance(fallback, list):
                return [dict(item) for item in fallback]
            knee = base.iloc[0].get("knee_production_mt")
            cost = base.iloc[0].get("knee_unit_cost_rs_per_thm")
            if pd.notna(knee):
                return [
                    {
                        "production_mt": float(knee),
                        "unit_cost_rs_per_thm": float(cost),
                        "reason": "detected knee",
                    }
                ]
    return []


def _default_selected_production(result: dict[str, Any]) -> float | None:
    knees = _base_knees(result)
    if knees:
        selected = min(
            knees,
            key=lambda item: (
                float(item.get("unit_cost_rs_per_thm", np.inf)),
                float(item.get("production_mt", np.inf)),
            ),
        )
        return float(selected["production_mt"])
    base = result["frontier"].loc[
        result["frontier"]["is_base"] & result["frontier"]["feasible"]
    ]
    if base.empty:
        return None
    return float(
        base.sort_values(
            ["unit_cost_rs_per_thm", "production_mt"]
        ).iloc[0]["production_mt"]
    )


def _selected_production_from_event(event: Any) -> float | None:
    """Extract production from a Streamlit Plotly selection payload."""

    if event is None:
        return None
    selection = (
        event.get("selection", {})
        if isinstance(event, Mapping)
        else getattr(event, "selection", {})
    )
    points = (
        selection.get("points", [])
        if isinstance(selection, Mapping)
        else getattr(selection, "points", [])
    )
    for point in reversed(list(points or [])):
        custom = (
            point.get("customdata")
            if isinstance(point, Mapping)
            else getattr(point, "customdata", None)
        )
        try:
            custom_values = list(custom) if custom is not None else []
        except TypeError:
            custom_values = []
        if len(custom_values) >= 2 and str(custom_values[1]) in {
            "base",
            "knee",
            "selected",
        }:
            try:
                return float(custom_values[0])
            except (TypeError, ValueError):
                pass

        # Some Streamlit/Plotly combinations omit customdata from point-click
        # events even though it remains available to hover templates. The x
        # coordinate is the production target and is therefore an equivalent,
        # version-independent fallback. Feasibility is checked by the caller.
        x_value = (
            point.get("x")
            if isinstance(point, Mapping)
            else getattr(point, "x", None)
        )
        try:
            return float(x_value)
        except (TypeError, ValueError):
            continue
    return None


def _sync_frontier_selection() -> None:
    """Persist a chart click before Streamlit renders the rerun."""

    clicked = _selected_production_from_event(
        st.session_state.get(_FRONTIER_CHART_KEY)
    )
    if clicked is not None:
        st.session_state[_SELECTED_PRODUCTION_KEY] = clicked


def _sync_point_control(widget_key: str) -> None:
    """Persist a slider or dropdown choice as the shared SBFE selection."""

    value = st.session_state.get(widget_key)
    if value is not None:
        st.session_state[_SELECTED_PRODUCTION_KEY] = float(value)


def _feasible_points(result: dict[str, Any]) -> pd.DataFrame:
    """Feasible current-state frontier points, one per production target."""

    frontier = result["frontier"]
    return (
        frontier.loc[frontier["is_base"] & frontier["feasible"]]
        .sort_values("production_mt")
        .drop_duplicates("production_mt")
    )


def _snap_to_point(value: Any, productions: list[float]) -> float | None:
    """Return the feasible production equal to ``value``, if there is one.

    Chart clicks arrive as JSON floats, so they are matched with ``np.isclose``
    and replaced by the exact option the slider and dropdown were built from.
    """

    if value is None or not productions:
        return None
    try:
        target = float(value)
    except (TypeError, ValueError):
        return None
    nearest = min(productions, key=lambda item: abs(item - target))
    return nearest if np.isclose(nearest, target) else None


def _render_point_controls(
    points: pd.DataFrame,
    knees: list[dict[str, Any]],
    selected_production_mt: float,
) -> None:
    """Slider and dropdown over the feasible points, kept in step with clicks."""

    productions = [float(value) for value in points["production_mt"]]
    cost_by_production = dict(
        zip(productions, points["unit_cost_rs_per_thm"].astype(float))
    )
    knee_productions = [float(item["production_mt"]) for item in knees]

    def point_label(production: float) -> str:
        label = (
            f"{production:,.0f} MT · Rs {cost_by_production[production]:,.0f}/THM"
        )
        if any(np.isclose(production, knee) for knee in knee_productions):
            label += " · knee"
        return label

    # Both widgets are re-seeded from the shared selection on every run, so a
    # chart click moves the slider and the dropdown, and either control moves
    # the other.
    st.session_state[_POINT_SLIDER_KEY] = selected_production_mt
    st.session_state[_POINT_PICKER_KEY] = selected_production_mt
    slider_col, picker_col = st.columns([2.2, 1.0])
    if len(productions) > 1:
        slider_col.select_slider(
            "Slide along the feasible frontier (MT/day)",
            options=productions,
            format_func=lambda production: f"{production:,.0f}",
            key=_POINT_SLIDER_KEY,
            on_change=_sync_point_control,
            args=(_POINT_SLIDER_KEY,),
        )
    picker_col.selectbox(
        "Or pick a point",
        options=productions,
        format_func=point_label,
        key=_POINT_PICKER_KEY,
        on_change=_sync_point_control,
        args=(_POINT_PICKER_KEY,),
    )


def _frontier_figure(
    result: dict[str, Any],
    selected_production_mt: float | None = None,
) -> go.Figure:
    frontier = result["frontier"]
    aggregate = result["aggregate"].sort_values("production_mt")
    base = frontier.loc[frontier["is_base"] & frontier["feasible"]].sort_values(
        "production_mt"
    )
    scenarios = result["scenarios"]
    varied_knees = scenarios.loc[
        (~scenarios["is_base"]) & scenarios["knee_production_mt"].notna(),
        "knee_production_mt",
    ]
    # A band and a "median" drawn from one or two scenarios would present a
    # single draw as a distribution.
    show_variation = _varied_scenario_count(result) >= MIN_VARIATION_SCENARIOS

    figure = go.Figure()
    if show_variation:
        figure.add_trace(
            go.Scatter(
                x=aggregate["production_mt"],
                y=aggregate["p90_unit_cost_rs_per_thm"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=aggregate["production_mt"],
                y=aggregate["p10_unit_cost_rs_per_thm"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(92, 124, 250, 0.18)",
                name="Daily variation P10-P90",
                hovertemplate="Production %{x:,.0f} MT<extra>P10-P90 band</extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=aggregate["production_mt"],
                y=aggregate["median_unit_cost_rs_per_thm"],
                mode="lines",
                line={"color": _KNEE_BLUE, "width": 2},
                name="Scenario median",
                hovertemplate="%{x:,.0f} MT<br>Rs %{y:,.0f}/THM<extra>Median</extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=base["production_mt"],
            y=base["unit_cost_rs_per_thm"],
            mode="lines+markers",
            line={"color": "#12b886", "width": 3},
            marker={"size": 7, "color": "#12b886"},
            customdata=np.column_stack(
                [
                    base["production_mt"].to_numpy(dtype=float),
                    np.repeat("base", len(base)),
                ]
            ),
            name="Feasible current-state points",
            hovertemplate=(
                "%{x:,.0f} MT/day<br>Rs %{y:,.0f}/THM"
                "<br>Click for solution details<extra>Current state</extra>"
            ),
        )
    )
    knees = _base_knees(result)
    if knees:
        knee_frame = pd.DataFrame(knees)
        figure.add_trace(
            go.Scatter(
                x=knee_frame["production_mt"],
                y=knee_frame["unit_cost_rs_per_thm"],
                mode="markers",
                marker={
                    "size": 12,
                    "symbol": "diamond",
                    "color": "#f59f00",
                    "line": {"color": "#7f4f00", "width": 1},
                },
                customdata=np.column_stack(
                    [
                        knee_frame["production_mt"].to_numpy(dtype=float),
                        np.repeat("knee", len(knee_frame)),
                        knee_frame["reason"].astype(str).to_numpy(),
                    ]
                ),
                name="Detected knees",
                hovertemplate=(
                    "%{x:,.0f} MT/day<br>Rs %{y:,.0f}/THM"
                    "<br>%{customdata[2]}<br>Click for solution details"
                    "<extra>Detected knee</extra>"
                ),
            )
        )
    if selected_production_mt is not None and not base.empty:
        selected_rows = base.loc[
            np.isclose(base["production_mt"], float(selected_production_mt))
        ]
        if not selected_rows.empty:
            selected = selected_rows.iloc[0]
            figure.add_trace(
                go.Scatter(
                    x=[float(selected["production_mt"])],
                    y=[float(selected["unit_cost_rs_per_thm"])],
                    mode="markers",
                    marker={
                        "size": 18,
                        "symbol": "circle-open",
                        "color": "#172b4d",
                        "line": {"color": "#172b4d", "width": 3},
                    },
                    customdata=[
                        [float(selected["production_mt"]), "selected"]
                    ],
                    name="Selected solution",
                    hovertemplate=(
                        "%{x:,.0f} MT/day<br>Rs %{y:,.0f}/THM"
                        "<extra>Selected solution</extra>"
                    ),
                )
            )
    if show_variation and not varied_knees.empty:
        median_knee = float(_knee_quantiles(varied_knees)["median"])
        figure.add_vline(
            x=median_knee,
            line_dash="dot",
            line_color="#364fc7",
            annotation_text=f"Scenario median {median_knee:,.0f} MT",
            annotation_position="top right",
        )
    targets = frontier["production_mt"].astype(float)
    if not base.empty and not targets.empty:
        ceiling = float(base["production_mt"].max())
        highest = float(targets.max())
        if highest > ceiling:
            half_step = _production_step(result) / 2.0
            figure.add_vrect(
                x0=ceiling + half_step,
                x1=highest + half_step,
                fillcolor="rgba(134, 142, 150, 0.12)",
                line_width=0,
                layer="below",
                annotation_text="Not feasible at current state",
                annotation_position="top left",
                annotation_font={"color": "#868e96", "size": 11},
            )
    figure.update_layout(
        height=480,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified",
        clickmode="event+select",
        dragmode=False,
        legend={"orientation": "h", "y": 1.13, "x": 0.0},
        xaxis_title="Hot-metal production target (MT/day)",
        yaxis_title="Minimum ore + flux cost (Rs/THM)",
    )
    return figure


def _varied_scenario_count(result: dict[str, Any]) -> int:
    """Daily-variation scenarios run, excluding the current-state base."""

    scenarios = result.get("scenarios")
    if not isinstance(scenarios, pd.DataFrame) or scenarios.empty:
        return 0
    return int((~scenarios["is_base"].astype(bool)).sum())


def _production_step(result: dict[str, Any]) -> float:
    """Spacing of the production grid, from the settings or the targets."""

    settings = result.get("settings")
    step = getattr(settings, "production_step_mt", None)
    if step:
        return float(step)
    targets = np.unique(result["frontier"]["production_mt"].astype(float))
    return float(np.min(np.diff(targets))) if len(targets) > 1 else 0.0


def _knee_quantiles(knees: pd.Series) -> dict[str, float | None]:
    """P10 / median / P90 of scenario knees, kept on the production grid.

    Knees can only fall on production targets, so interpolated quantiles such as
    2,230 between the 2,220 and 2,240 targets describe no possible outcome. P10
    rounds down and P90 up, so the range always holds at least 80% of scenarios.
    """

    if knees.empty:
        return {"p10": None, "median": None, "p90": None}
    values = knees.astype(float)
    return {
        "p10": float(values.quantile(0.10, interpolation="lower")),
        "median": float(values.quantile(0.50, interpolation="nearest")),
        "p90": float(values.quantile(0.90, interpolation="higher")),
    }


def _variation_summary(result: dict[str, Any]) -> dict[str, float | None]:
    """Knee spread and how far production stays feasible across scenarios."""

    scenarios = result["scenarios"]
    knees = scenarios.loc[
        (~scenarios["is_base"]) & scenarios["knee_production_mt"].notna(),
        "knee_production_mt",
    ].astype(float)
    aggregate = result["aggregate"].sort_values("production_mt")

    def feasible_up_to(share_pct: float) -> float | None:
        # Contiguous from the lowest target: a target that recovers feasibility
        # after an infeasible one does not extend the range.
        reached = None
        for production, pct in zip(aggregate["production_mt"], aggregate["feasibility_pct"]):
            if float(pct) + 1e-9 < share_pct:
                break
            reached = float(production)
        return reached

    return {
        **_knee_quantiles(knees),
        "all_feasible_up_to": feasible_up_to(100.0),
        "half_feasible_up_to": feasible_up_to(50.0),
    }


def _feasibility_figure(result: dict[str, Any]) -> go.Figure:
    """Share of daily-variation scenarios in which each target is feasible."""

    aggregate = result["aggregate"].sort_values("production_mt")
    figure = go.Figure(
        go.Scatter(
            x=aggregate["production_mt"],
            y=aggregate["feasibility_pct"],
            mode="lines",
            line={"color": _FEASIBLE_AMBER, "width": 2.5, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(245, 159, 0, 0.18)",
            name="Feasible scenarios",
            hovertemplate=(
                "%{x:,.0f} MT/day: %{y:.0f}% of scenarios feasible<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=50, line_dash="dot", line_color="#adb5bd", line_width=1)
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        showlegend=False,
        xaxis={
            "title": "Hot-metal production target (MT/day)",
            "range": _production_axis_range(result),
        },
        yaxis={"title": "Scenarios feasible", "range": [0, 105], "ticksuffix": "%"},
    )
    return figure


def _production_axis_range(result: dict[str, Any]) -> list[float]:
    """One x-span for both Daily variation charts, half a step past each end."""

    targets = result["aggregate"]["production_mt"].astype(float)
    half_step = _production_step(result) / 2.0
    return [float(targets.min()) - half_step, float(targets.max()) + half_step]


def _knee_distribution_figure(result: dict[str, Any]) -> go.Figure:
    """Where the knee lands across scenarios, on the production grid itself.

    Knees can only fall on production targets, so they are counted per target;
    an automatic histogram would invent bins between targets (one knee drew as
    a single bar spanning 2,199.6-2,200.4).
    """

    scenarios = result["scenarios"]
    knees = scenarios.loc[
        (~scenarios["is_base"]) & scenarios["knee_production_mt"].notna(),
        "knee_production_mt",
    ].astype(float)
    counts = knees.value_counts().sort_index()
    share = counts / max(1, int(counts.sum())) * 100.0
    step = _production_step(result)
    figure = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.to_numpy(),
            width=step * 0.8 if step else None,
            marker_color=_KNEE_BLUE,
            customdata=share.to_numpy(),
            hovertemplate=(
                "Knee at %{x:,.0f} MT/day<br>%{y} scenarios (%{customdata:.0f}%)"
                "<extra></extra>"
            ),
        )
    )
    # One label per distinct value, so equal quantiles merge into one label.
    # Knees sit one grid step apart, so labels are anchored away from each
    # other: the lowest reads to the left, the highest to the right, and a
    # middle one is raised above both.
    marks: dict[float, list[str]] = {}
    quantiles = _knee_quantiles(knees)
    for label, key in (("P10", "p10"), ("Median", "median"), ("P90", "p90")):
        if quantiles[key] is not None:
            marks.setdefault(round(float(quantiles[key]), 6), []).append(label)
    ordered = sorted(marks.items())
    for position, (value, labels) in enumerate(ordered):
        figure.add_vline(
            x=value,
            line_dash="solid" if "Median" in labels else "dot",
            line_color="#364fc7",
            line_width=1.5,
        )
        if len(ordered) == 1:
            anchor, height = "center", 1.02
        elif position == 0:
            anchor, height = "right", 1.02
        elif position == len(ordered) - 1:
            anchor, height = "left", 1.02
        else:
            anchor, height = "center", 1.13
        figure.add_annotation(
            x=value,
            y=height,
            yref="paper",
            text=f"{' / '.join(labels)} {value:,.0f}",
            showarrow=False,
            xanchor=anchor,
            yanchor="bottom",
            font={"size": 11, "color": "#364fc7"},
        )
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        showlegend=False,
        bargap=0.1,
        xaxis={
            "title": "Hot-metal production at the knee (MT/day)",
            # Same span as the feasibility chart above, so the two read together.
            "range": _production_axis_range(result),
        },
        yaxis={
            "title": "Scenarios",
            "dtick": 1 if int(counts.max() if len(counts) else 0) <= 10 else None,
        },
    )
    return figure


def _render_selected_solution(
    result: dict[str, Any],
    production_mt: float | None,
) -> None:
    frontier = result["frontier"]
    names = result["material_names"]
    if production_mt is None:
        st.info("No feasible current-state solution is available.")
        return
    rows = frontier.loc[
        frontier["is_base"]
        & frontier["feasible"]
        & np.isclose(frontier["production_mt"], float(production_mt))
    ]
    if rows.empty:
        st.warning("The selected point is not a feasible current-state solution.")
        return
    row = rows.iloc[0]
    knees = _base_knees(result)
    matching_knee = next(
        (
            item
            for item in knees
            if np.isclose(float(item["production_mt"]), float(production_mt))
        ),
        None,
    )
    st.markdown(f"#### Selected solution · {float(production_mt):,.0f} MT/day")
    if matching_knee:
        st.caption(f"Detected knee: {matching_knee.get('reason', 'cost curvature')}.")
    else:
        st.caption("Feasible frontier point.")

    def number(key: str) -> float | None:
        try:
            value = float(row.get(key, np.nan))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    total_unit_cost = number("unit_cost_rs_per_thm")
    total_daily_cost = number("total_cost_rs")
    slag_rate = number("slag_rate_kg_per_thm")
    summary_cols = st.columns(2)
    summary_cols[0].metric(
        "Ore + flux cost",
        f"Rs {total_unit_cost:,.0f}/THM"
        if total_unit_cost is not None
        else "Not available",
    )
    summary_cols[1].metric(
        "Slag rate",
        f"{slag_rate:,.1f} kg/THM" if slag_rate is not None else "Not available",
    )

    blend_rows = []
    for ore_id, name in names.items():
        share = float(row.get(f"share_pct__{ore_id}", 0.0) or 0.0)
        quantity = float(row.get(f"quantity_mt__{ore_id}", 0.0) or 0.0)
        if share > 0.005 or quantity > 0.005:
            blend_rows.append(
                {
                    "Material": name,
                    "Wet quantity (MT)": quantity,
                    "Share (%)": share,
                }
            )
    flux_rows = []
    for flux_id, name in (result.get("flux_names", {}) or {}).items():
        quantity = float(row.get(f"flux_mt__{flux_id}", 0.0) or 0.0)
        if quantity > 0.005:
            flux_rows.append({"Flux": name, "Quantity (MT)": quantity})

    blend_tab, calculation_tab = st.tabs(["Blend and pricing", "Calculation details"])
    with blend_tab:
        left, right = st.columns([1.35, 1.0])
        with left:
            st.dataframe(
                pd.DataFrame(blend_rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Wet quantity (MT)": st.column_config.NumberColumn(
                        format="%.1f"
                    ),
                    "Share (%)": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            if flux_rows:
                st.dataframe(
                    pd.DataFrame(flux_rows),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Quantity (MT)": st.column_config.NumberColumn(format="%.1f")
                    },
                )
        with right:
            pricing = pd.DataFrame(
                [
                    {
                        "Component": "Ore",
                        "Rs/THM": number("ore_cost_per_thm_rs"),
                        "Rs/day": number("ore_cost_total_rs"),
                    },
                    {
                        "Component": "Flux",
                        "Rs/THM": number("flux_cost_per_thm_rs"),
                        "Rs/day": number("flux_cost_total_rs"),
                    },
                    {
                        "Component": "Ore + flux",
                        "Rs/THM": total_unit_cost,
                        "Rs/day": total_daily_cost,
                    },
                ]
            )
            st.dataframe(
                pricing,
                hide_index=True,
                width="stretch",
                column_config={
                    "Rs/THM": st.column_config.NumberColumn(format="localized"),
                    "Rs/day": st.column_config.NumberColumn(format="localized"),
                },
            )
    with calculation_tab:
        calculations = pd.DataFrame(
            [
                ("Production target", number("production_mt"), "MT/day"),
                ("Total ore burden", number("total_ore_qty_mt"), "MT/day"),
                ("IBRM + flux", number("total_burden_qty_mt"), "MT/day"),
                ("Charging capacity", number("max_burden_qty_mt"), "MT/day"),
                ("Slag quantity", number("slag_mt"), "MT/day"),
                ("Slag rate", slag_rate, "kg/THM"),
                ("Basicity CaO/SiO2", number("slag_basicity"), "ratio"),
                ("Total basicity", number("slag_t_basicity"), "ratio"),
                ("Slag Al2O3", number("slag_al2o3_pct"), "%"),
                ("Slag MgO", number("slag_mgo_pct"), "%"),
                ("MgO/Al2O3", number("slag_mgo_al2o3_ratio"), "ratio"),
            ],
            columns=["Calculation", "Value", "Unit"],
        )
        st.dataframe(
            calculations,
            hide_index=True,
            width="stretch",
            column_config={
                "Value": st.column_config.NumberColumn(format="%.2f")
            },
        )


def _render_result(result: dict[str, Any]) -> None:
    scenarios = result["scenarios"]
    base = scenarios.loc[scenarios["is_base"]].iloc[0]
    varied = scenarios.loc[
        (~scenarios["is_base"]) & scenarios["knee_production_mt"].notna()
    ]
    base_knee = base.get("knee_production_mt")
    quantiles = _knee_quantiles(varied["knee_production_mt"])
    median_knee = quantiles["median"] if quantiles["median"] is not None else np.nan
    p10_knee = quantiles["p10"] if quantiles["p10"] is not None else np.nan
    p90_knee = quantiles["p90"] if quantiles["p90"] is not None else np.nan
    max_feasible = base.get("max_feasible_production_mt")
    capacity_onset = base.get("capacity_onset_production_mt")
    points = _feasible_points(result)
    productions = [float(value) for value in points["production_mt"]]
    # The chart, slider, and dropdown all write this one key through their
    # callbacks, which run before the script, so every control and the details
    # below agree on the selection within a single rerun.
    selected_production = _snap_to_point(
        st.session_state.get(_SELECTED_PRODUCTION_KEY), productions
    )
    if selected_production is None:
        selected_production = _snap_to_point(
            _default_selected_production(result), productions
        )
    if selected_production is None and productions:
        selected_production = productions[0]
    if selected_production is not None:
        st.session_state[_SELECTED_PRODUCTION_KEY] = selected_production

    varied_count = _varied_scenario_count(result)
    enough_variation = varied_count >= MIN_VARIATION_SCENARIOS
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Default low-cost knee",
        f"{float(base_knee):,.0f} MT" if pd.notna(base_knee) else "No feasible knee",
    )
    if enough_variation:
        metric_cols[1].metric(
            "Daily-variation knee",
            f"{float(median_knee):,.0f} MT"
            if pd.notna(median_knee)
            else "Not available",
            help=(
                f"Median over {varied_count} scenarios. P10-P90: "
                f"{float(p10_knee):,.0f}-{float(p90_knee):,.0f} MT"
                if pd.notna(p10_knee) and pd.notna(p90_knee)
                else f"Median over {varied_count} scenarios."
            ),
        )
    else:
        metric_cols[1].metric(
            "Daily-variation knee",
            "Too few runs",
            help=(
                f"{varied_count} variation scenario(s) were run; at least "
                f"{MIN_VARIATION_SCENARIOS} are needed (20 recommended)."
            ),
        )
    metric_cols[2].metric(
        "Current feasible ceiling",
        f"{float(max_feasible):,.0f} MT" if pd.notna(max_feasible) else "None",
    )
    metric_cols[3].metric(
        "Charging-capacity onset",
        f"{float(capacity_onset):,.0f} MT"
        if pd.notna(capacity_onset)
        else "Not binding",
    )

    frontier_tab, variation_tab, drivers_tab = st.tabs(
        ["Production frontier", "Daily variation", "Knee drivers"]
    )
    with frontier_tab:
        knees = _base_knees(result)
        if knees:
            knee_text = ", ".join(
                f"{float(item['production_mt']):,.0f} MT"
                for item in knees
            )
            st.caption(
                f"Detected current-state knees: {knee_text}. Amber diamonds mark "
                "knees. Slide, pick from the list, or click any green point to "
                "see that solution."
            )
        if selected_production is not None:
            _render_point_controls(points, knees, selected_production)
        st.plotly_chart(
            _frontier_figure(result, selected_production),
            width="stretch",
            key=_FRONTIER_CHART_KEY,
            on_select=_sync_frontier_selection,
            selection_mode="points",
        )
        _render_selected_solution(result, selected_production)
    with variation_tab:
        if not enough_variation:
            st.info(
                f"**Only {varied_count} variation scenario"
                f"{'' if varied_count == 1 else 's'} {'was' if varied_count == 1 else 'were'} "
                f"run.** Day-to-day variation needs at least "
                f"{MIN_VARIATION_SCENARIOS}; 20 is recommended. With one scenario "
                "every target is either 0% or 100% feasible and the knee "
                "'distribution' is a single value, so there is nothing to read "
                "here. Raise **Variation scenarios** under Simulation constraints "
                "and run again."
            )
        else:
            summary = _variation_summary(result)
            with st.container(border=True):
                tiles = st.columns(4)
                tiles[0].metric("Scenarios run", f"{varied_count:,}")
                tiles[1].metric(
                    "Knee range (P10-P90)",
                    f"{summary['p10']:,.0f}-{summary['p90']:,.0f} MT"
                    if summary["p10"] is not None and summary["p90"] is not None
                    else "No knee",
                    help="80% of scenarios put the knee inside this range.",
                )
                tiles[2].metric(
                    "Feasible in every scenario",
                    f"up to {summary['all_feasible_up_to']:,.0f} MT"
                    if summary["all_feasible_up_to"] is not None
                    else "None",
                    help="Highest target that stays feasible whatever the day's "
                    "stock, price and chemistry within the variation set above.",
                )
                tiles[3].metric(
                    "Feasible in half the scenarios",
                    f"up to {summary['half_feasible_up_to']:,.0f} MT"
                    if summary["half_feasible_up_to"] is not None
                    else "None",
                    help="Beyond this, most days' raw materials cannot make the target.",
                )
            st.markdown("**How often each production target is feasible**")
            st.plotly_chart(_feasibility_figure(result), width="stretch")
            if not varied.empty:
                st.markdown("**Where the knee lands**")
                st.plotly_chart(_knee_distribution_figure(result), width="stretch")
            st.caption(
                "Each scenario redraws stock, price, Fe and gangue within the "
                "variation set above and re-solves every production target. The "
                "dotted line on the feasibility chart marks 50%."
            )
    with drivers_tab:
        drivers = result["drivers"].copy()
        if drivers.empty:
            st.info("Run at least four varied scenarios to estimate knee drivers.")
        else:
            figure = go.Figure(
                go.Bar(
                    x=drivers["knee_correlation"],
                    y=drivers["driver"],
                    orientation="h",
                    marker_color=[
                        "#12b886" if value >= 0.0 else "#fa5252"
                        for value in drivers["knee_correlation"]
                    ],
                    hovertemplate=(
                        "%{y}<br>Rank correlation %{x:.2f}<extra></extra>"
                    ),
                )
            )
            figure.update_layout(
                height=max(320, 48 * len(drivers)),
                margin={"l": 10, "r": 10, "t": 20, "b": 10},
                xaxis_title="Rank correlation with knee production",
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(figure, width="stretch")
            st.dataframe(
                drivers[["driver", "knee_correlation", "direction"]].rename(
                    columns={
                        "driver": "Driver",
                        "knee_correlation": "Correlation",
                        "direction": "Observed direction",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Correlation": st.column_config.NumberColumn(format="%.2f")
                },
            )
        st.caption(
            "These are scenario associations, not causal coefficients. The physical "
            "breakpoints remain the charging limit, usable Fe per charge, slag and "
            "basicity limits, material bounds, and the relative delivered prices."
        )


def render_production_frontier(
    *,
    selected_ores: list[OreInput],
    fuel_ash_inputs: list[FuelAshInput],
    flux_inputs: list[FluxInput],
    dust_inputs: list[DustInput],
    slag_balance_settings: SlagBalanceSettings,
    hm_fe_pct: float,
    feo_in_slag_pct: float,
    model_to_plant_slag_factor: float,
    default_max_charges_per_hour: float,
    default_charge_mass_mt: float,
    default_nut_coke_rate_kg_per_thm: float,
    default_nut_coke_moisture_pct: float,
) -> None:
    """Render the isolated SBFE product at the bottom of the BMO page."""

    st.divider()
    with st.container(border=True):
        st.markdown("## Stochastic Burden Frontier Engine (SBFE)")
        st.caption(
            "Map the minimum ore-and-flux cost across production targets, then "
            "stress the knee against daily stock, price, and chemistry variation. "
            "This simulation does not change the main BMO inputs or results."
        )

        fingerprint = _input_fingerprint(selected_ores)
        with st.expander("Simulation constraints and daily variation", expanded=False):
            with st.form("bmo_sbfe_form", clear_on_submit=False):
                st.markdown("#### Production and furnace envelope")
                p1, p2, p3, p4 = st.columns(4)
                production_min = p1.number_input(
                    "Minimum production (MT)",
                    min_value=100.0,
                    value=1700.0,
                    step=10.0,
                    key="bmo_frontier_production_min_mt",
                )
                production_max = p2.number_input(
                    "Maximum production (MT)",
                    min_value=100.0,
                    value=2500.0,
                    step=10.0,
                    key="bmo_frontier_production_max_mt",
                )
                production_step = p3.number_input(
                    "Production step (MT)",
                    min_value=1.0,
                    value=5.0,
                    step=5.0,
                    key="bmo_frontier_production_step_mt",
                )
                max_slag_rate = p4.number_input(
                    "Maximum slag rate (kg/THM)",
                    min_value=0.0,
                    value=375.0,
                    step=5.0,
                    key="bmo_frontier_max_slag_rate",
                )

                q1, q2, q3, q4 = st.columns(4)
                basicity_min = q1.number_input(
                    "Minimum basicity",
                    min_value=0.0,
                    value=1.0,
                    step=0.01,
                    format="%.3f",
                    key="bmo_frontier_basicity_min",
                )
                basicity_max = q2.number_input(
                    "Maximum basicity",
                    min_value=0.0,
                    value=1.2,
                    step=0.01,
                    format="%.3f",
                    key="bmo_frontier_basicity_max",
                )
                max_charges = q3.number_input(
                    "Maximum charges per hour",
                    min_value=0.1,
                    value=float(default_max_charges_per_hour),
                    step=0.05,
                    format="%.2f",
                    key="bmo_frontier_max_charges_per_hour",
                )
                charge_mass = q4.number_input(
                    "Maximum quantity per charge (MT)",
                    min_value=0.1,
                    value=float(default_charge_mass_mt),
                    step=0.1,
                    format="%.2f",
                    key="bmo_frontier_charge_mass_mt",
                )

                with st.expander("Additional slag-quality limits", expanded=False):
                    st.caption("Set a limit to 0 to switch it off.")
                    s1, s2, s3, s4, s5 = st.columns(5)
                    t_basicity_min = s1.number_input(
                        "Min total basicity",
                        min_value=0.0,
                        value=0.0,
                        step=0.01,
                        key="bmo_frontier_t_basicity_min",
                    )
                    t_basicity_max = s2.number_input(
                        "Max total basicity",
                        min_value=0.0,
                        value=0.0,
                        step=0.01,
                        key="bmo_frontier_t_basicity_max",
                    )
                    al2o3_max = s3.number_input(
                        "Max Al2O3 (%)",
                        min_value=0.0,
                        value=0.0,
                        step=0.25,
                        key="bmo_frontier_al2o3_max",
                    )
                    mgo_min = s4.number_input(
                        "Min MgO (%)",
                        min_value=0.0,
                        value=0.0,
                        step=0.25,
                        key="bmo_frontier_mgo_min",
                    )
                    mgo_al2o3_min = s5.number_input(
                        "Min MgO/Al2O3",
                        min_value=0.0,
                        value=0.0,
                        step=0.01,
                        key="bmo_frontier_mgo_al2o3_min",
                    )

                c1, c2, c3 = st.columns(3)
                enforce_capacity = c1.checkbox(
                    "Enforce charging capacity",
                    value=True,
                    key="bmo_frontier_enforce_capacity",
                )
                nut_coke_rate = c2.number_input(
                    "Nut coke (kg/THM)",
                    min_value=0.0,
                    value=float(default_nut_coke_rate_kg_per_thm),
                    step=1.0,
                    key="bmo_frontier_nut_coke_rate",
                )
                nut_coke_moisture = c3.number_input(
                    "Nut coke moisture added (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_nut_coke_moisture_pct),
                    step=0.1,
                    key="bmo_frontier_nut_coke_moisture",
                )

                st.markdown("#### Product-only material bounds and basis")
                material_editor = st.data_editor(
                    _material_editor_frame(selected_ores),
                    hide_index=True,
                    width="stretch",
                    key=f"bmo_frontier_materials_{fingerprint}",
                    column_order=(
                        "Use",
                        "Material",
                        "Stock MT",
                        "Price Rs/MT",
                        "Min share %",
                        "Max share %",
                        "Fe %",
                        "SiO2 %",
                        "Al2O3 %",
                    ),
                    column_config={
                        "ore_id": None,
                        "Material": st.column_config.TextColumn(disabled=True),
                        "Stock MT": st.column_config.NumberColumn(
                            min_value=0.0, format="localized"
                        ),
                        "Price Rs/MT": st.column_config.NumberColumn(
                            min_value=0.0, format="localized"
                        ),
                        "Min share %": st.column_config.NumberColumn(
                            min_value=0.0, max_value=100.0
                        ),
                        "Max share %": st.column_config.NumberColumn(
                            min_value=0.0, max_value=100.0
                        ),
                        "Fe %": st.column_config.NumberColumn(
                            min_value=0.0, max_value=100.0, format="%.2f"
                        ),
                        "SiO2 %": st.column_config.NumberColumn(
                            min_value=0.0, max_value=100.0, format="%.2f"
                        ),
                        "Al2O3 %": st.column_config.NumberColumn(
                            min_value=0.0, max_value=100.0, format="%.2f"
                        ),
                    },
                )
                editable_fluxes = _flux_editor_frame(flux_inputs)
                if not editable_fluxes.empty:
                    st.markdown("#### Product-only flux bounds")
                    flux_editor = st.data_editor(
                        editable_fluxes,
                        hide_index=True,
                        width="stretch",
                        key=f"bmo_frontier_fluxes_{fingerprint}",
                        column_order=(
                            "Enabled",
                            "Flux",
                            "Min MT",
                            "Max MT",
                            "Stock MT",
                            "Price Rs/MT",
                        ),
                        column_config={
                            "flux_id": None,
                            "Flux": st.column_config.TextColumn(disabled=True),
                            "Min MT": st.column_config.NumberColumn(
                                min_value=0.0, format="localized"
                            ),
                            "Max MT": st.column_config.NumberColumn(
                                min_value=0.0, format="localized"
                            ),
                            "Stock MT": st.column_config.NumberColumn(
                                min_value=0.0, format="localized"
                            ),
                            "Price Rs/MT": st.column_config.NumberColumn(
                                min_value=0.0, format="localized"
                            ),
                        },
                    )
                else:
                    flux_editor = editable_fluxes

                st.markdown("#### Daily raw-material variation")
                v1, v2, v3, v4 = st.columns(4)
                variation_scenarios = v1.number_input(
                    "Variation scenarios",
                    min_value=0,
                    max_value=MAX_VARIATION_SCENARIOS,
                    value=20,
                    step=1,
                    key="bmo_frontier_scenarios",
                    help=(
                        f"Up to {MAX_VARIATION_SCENARIOS:,}. Each scenario re-solves "
                        "every production target, so 1,000 scenarios over "
                        "1,700-2,500 MT at a 5 MT step is about 161,000 LP solves; "
                        "the progress panel shows the expected finish time."
                    ),
                )
                stock_variation = v2.number_input(
                    "Stock variation (+/- %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=15.0,
                    step=2.5,
                    key="bmo_frontier_stock_variation",
                )
                price_variation = v3.number_input(
                    "Price variation (+/- %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=7.5,
                    step=2.5,
                    key="bmo_frontier_price_variation",
                )
                fe_variation = v4.number_input(
                    "Fe variation (+/- points)",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.75,
                    step=0.25,
                    key="bmo_frontier_fe_variation",
                )
                w1, w2, w3 = st.columns(3)
                gangue_variation = w1.number_input(
                    "Gangue variation (+/- %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=7.5,
                    step=2.5,
                    key="bmo_frontier_gangue_variation",
                )
                cost_tolerance = w2.number_input(
                    "Near-minimum band (Rs/THM)",
                    min_value=0.0,
                    value=25.0,
                    step=5.0,
                    key="bmo_frontier_cost_tolerance",
                )
                random_seed = w3.number_input(
                    "Reproducibility seed",
                    min_value=0,
                    value=234,
                    step=1,
                    key="bmo_frontier_seed",
                )

                submitted = st.form_submit_button(
                    "Run SBFE analysis", type="primary", width="stretch"
                )

        if submitted:
            simulation_ores = _ores_from_editor(material_editor, selected_ores)
            simulation_fluxes = _fluxes_from_editor(flux_editor, flux_inputs)
            settings = FrontierSettings(
                production_min_mt=float(production_min),
                production_max_mt=float(production_max),
                production_step_mt=float(production_step),
                max_slag_rate_kg_per_thm=float(max_slag_rate),
                basicity_min=float(basicity_min),
                basicity_max=float(basicity_max),
                t_basicity_min=_optional_limit(t_basicity_min),
                t_basicity_max=_optional_limit(t_basicity_max),
                al2o3_max_pct=_optional_limit(al2o3_max),
                mgo_min_pct=_optional_limit(mgo_min),
                mgo_al2o3_min=_optional_limit(mgo_al2o3_min),
                max_charges_per_hour=float(max_charges),
                charge_mass_mt=float(charge_mass),
                enforce_charging_capacity=bool(enforce_capacity),
                nut_coke_rate_kg_per_thm=float(nut_coke_rate),
                nut_coke_moisture_pct=float(nut_coke_moisture),
                variation_scenarios=int(variation_scenarios),
                stock_variation_pct=float(stock_variation),
                price_variation_pct=float(price_variation),
                fe_variation_pp=float(fe_variation),
                gangue_variation_pct=float(gangue_variation),
                near_minimum_tolerance_rs_per_thm=float(cost_tolerance),
                random_seed=int(random_seed),
            )
            errors = validate_frontier_inputs(simulation_ores, settings)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                started = time.perf_counter()
                with st.status(
                    "SBFE is mapping the stochastic production frontier...",
                    expanded=True,
                ) as status:
                    phase = st.empty()
                    progress = st.progress(0.0, text="Preparing material scenarios")
                    telemetry = st.empty()
                    last_update = {"completed": -1}

                    def report(event: dict[str, Any]) -> None:
                        completed = int(event.get("completed", 0))
                        total = max(1, int(event.get("total", 1)))
                        update_interval = max(1, total // 100)
                        if (
                            completed != total
                            and completed - last_update["completed"] < update_interval
                        ):
                            return
                        last_update["completed"] = completed
                        elapsed = time.perf_counter() - started
                        remaining = _estimated_remaining_seconds(
                            elapsed_seconds=elapsed,
                            completed=completed,
                            total=total,
                        )
                        if remaining is None:
                            timing = f"Elapsed {_duration_text(elapsed)}"
                        else:
                            expected_end = datetime.now().astimezone() + timedelta(
                                seconds=remaining
                            )
                            timing = (
                                f"Elapsed {_duration_text(elapsed)}  |  "
                                f"Est. remaining {_duration_text(remaining)}  |  "
                                f"Expected end {expected_end:%H:%M:%S %Z}"
                            )
                        if event.get("stage") == "complete":
                            phase.markdown(
                                "**Ranking curvature and binding constraints**"
                            )
                        else:
                            phase.markdown(
                                "**Solving feasible blends across the daily "
                                "material envelope**"
                            )
                        progress.progress(
                            min(1.0, completed / total),
                            text=f"LP solve {completed:,} of {total:,}",
                        )
                        telemetry.code(
                            "Scenario "
                            f"{int(event.get('scenario', 0)) + 1:,}/"
                            f"{int(event.get('scenario_count', 1)):,}  |  "
                            f"Target {float(event.get('production_mt', 0.0)):,.0f} MT  |  "
                            f"{timing}",
                            language=None,
                        )

                    try:
                        result = run_frontier_simulation(
                            simulation_ores,
                            settings=settings,
                            hm_fe_pct=float(hm_fe_pct),
                            feo_in_slag_pct=float(feo_in_slag_pct),
                            model_to_plant_slag_factor=float(
                                model_to_plant_slag_factor
                            ),
                            fuel_ash_inputs=fuel_ash_inputs,
                            flux_inputs=simulation_fluxes,
                            dust_inputs=dust_inputs,
                            slag_balance_settings=slag_balance_settings,
                            progress_callback=report,
                        )
                    except Exception as exc:  # noqa: BLE001
                        status.update(
                            label="SBFE analysis failed",
                            state="error",
                            expanded=True,
                        )
                        st.exception(exc)
                    else:
                        elapsed = time.perf_counter() - started
                        result["elapsed_seconds"] = elapsed
                        result["source_fingerprint"] = fingerprint
                        st.session_state[_RESULT_KEY] = result
                        default_production = _default_selected_production(result)
                        if default_production is not None:
                            st.session_state[_SELECTED_PRODUCTION_KEY] = (
                                default_production
                            )
                        progress.progress(1.0, text="Frontier analysis complete")
                        telemetry.code(
                            f"Solved {result['solve_count']:,} LPs  |  "
                            f"Analysed {len(result['scenarios']):,} material states  |  "
                            f"Elapsed {elapsed:,.1f}s",
                            language=None,
                        )
                        status.update(
                            label=(
                                f"SBFE completed {result['solve_count']:,} "
                                f"LP solves in {elapsed:,.1f}s"
                            ),
                            state="complete",
                            expanded=False,
                        )

        result = st.session_state.get(_RESULT_KEY)
        if (
            isinstance(result, dict)
            and not result.get("frontier", pd.DataFrame()).empty
        ):
            if result.get("source_fingerprint") != fingerprint:
                st.warning(
                    "The main BMO material inputs have changed since this simulation. "
                    "Run SBFE again to refresh the frontier."
                )
            st.caption(
                f"Last run: {int(result.get('solve_count', 0)):,} LP solves in "
                f"{float(result.get('elapsed_seconds', 0.0)):,.1f} seconds."
            )
            _render_result(result)

        with st.expander("What determines the knee?", expanded=False):
            st.markdown(
                "The knee moves when one or more physical or economic constraints "
                "start binding: charging throughput and usable Fe per charge; stock "
                "of the cheapest high-Fe materials; relative prices and price spread; "
                "ore moisture and Fe variation; gangue generation and the slag cap; "
                "basicity-driven flux demand; and material or flux minimum/maximum "
                "bounds. The scenario charts separate the current-state knee from "
                "the distribution caused by daily raw-material variation."
            )


__all__ = ["render_production_frontier"]
