"""Streamlit product surface for the Stochastic Burden Frontier Engine."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.bmo.production_frontier import (
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


def _frontier_figure(result: dict[str, Any]) -> go.Figure:
    frontier = result["frontier"]
    aggregate = result["aggregate"].sort_values("production_mt")
    base = frontier.loc[frontier["is_base"] & frontier["feasible"]].sort_values(
        "production_mt"
    )
    scenarios = result["scenarios"]
    base_summary = scenarios.loc[scenarios["is_base"]].iloc[0]
    varied_knees = scenarios.loc[
        (~scenarios["is_base"]) & scenarios["knee_production_mt"].notna(),
        "knee_production_mt",
    ]

    figure = go.Figure()
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
            line={"color": "#5c7cfa", "width": 2},
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
            marker={"size": 5},
            name="Current material state",
            hovertemplate="%{x:,.0f} MT<br>Rs %{y:,.0f}/THM<extra>Current</extra>",
        )
    )
    base_knee = base_summary.get("knee_production_mt")
    if pd.notna(base_knee):
        figure.add_vline(
            x=float(base_knee),
            line_dash="dash",
            line_color="#087f5b",
            annotation_text=f"Current knee {float(base_knee):,.0f} MT",
            annotation_position="top left",
        )
    if not varied_knees.empty:
        median_knee = float(varied_knees.median())
        figure.add_vline(
            x=median_knee,
            line_dash="dot",
            line_color="#364fc7",
            annotation_text=f"Scenario median {median_knee:,.0f} MT",
            annotation_position="top right",
        )
    figure.update_layout(
        height=480,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.13, "x": 0.0},
        xaxis_title="Hot-metal production target (MT/day)",
        yaxis_title="Minimum ore + flux cost (Rs/THM)",
    )
    return figure


def _feasibility_figure(result: dict[str, Any]) -> go.Figure:
    aggregate = result["aggregate"].sort_values("production_mt")
    figure = go.Figure(
        go.Bar(
            x=aggregate["production_mt"],
            y=aggregate["feasibility_pct"],
            marker_color="#f59f00",
            hovertemplate="%{x:,.0f} MT<br>%{y:.0f}% feasible<extra></extra>",
        )
    )
    figure.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title="Hot-metal production target (MT/day)",
        yaxis_title="Scenario feasibility (%)",
        yaxis={"range": [0, 105]},
    )
    return figure


def _render_knee_blend(result: dict[str, Any]) -> None:
    scenarios = result["scenarios"]
    frontier = result["frontier"]
    names = result["material_names"]
    base_summary = scenarios.loc[scenarios["is_base"]].iloc[0]
    knee = base_summary.get("knee_production_mt")
    if pd.isna(knee):
        st.info("No feasible current-state knee was available for a blend breakdown.")
        return
    rows = frontier.loc[
        frontier["is_base"]
        & frontier["feasible"]
        & np.isclose(frontier["production_mt"], float(knee))
    ]
    if rows.empty:
        return
    row = rows.iloc[0]
    blend_rows = []
    for ore_id, name in names.items():
        share = float(row.get(f"share_pct__{ore_id}", 0.0) or 0.0)
        if share > 0.005:
            blend_rows.append({"Material": name, "Share (%)": share})
    st.dataframe(
        pd.DataFrame(blend_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Share (%)": st.column_config.NumberColumn("Share (%)", format="%.2f")
        },
    )
    st.caption(
        f"At {float(knee):,.0f} MT: Rs {float(row['unit_cost_rs_per_thm']):,.0f}/THM, "
        f"{float(row['slag_rate_kg_per_thm']):,.1f} kg/THM slag, "
        f"basicity {float(row['slag_basicity']):.3f}, and "
        f"{float(row['charging_utilization_pct']):.1f}% charging utilisation."
    )


def _render_result(result: dict[str, Any]) -> None:
    scenarios = result["scenarios"]
    base = scenarios.loc[scenarios["is_base"]].iloc[0]
    varied = scenarios.loc[
        (~scenarios["is_base"]) & scenarios["knee_production_mt"].notna()
    ]
    base_knee = base.get("knee_production_mt")
    median_knee = varied["knee_production_mt"].median() if not varied.empty else np.nan
    p10_knee = (
        varied["knee_production_mt"].quantile(0.10) if not varied.empty else np.nan
    )
    p90_knee = (
        varied["knee_production_mt"].quantile(0.90) if not varied.empty else np.nan
    )
    max_feasible = base.get("max_feasible_production_mt")
    capacity_onset = base.get("capacity_onset_production_mt")

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Current-state knee",
        f"{float(base_knee):,.0f} MT" if pd.notna(base_knee) else "No feasible knee",
    )
    metric_cols[1].metric(
        "Daily-variation knee",
        f"{float(median_knee):,.0f} MT" if pd.notna(median_knee) else "Not available",
        help=(
            f"P10-P90: {float(p10_knee):,.0f}-{float(p90_knee):,.0f} MT"
            if pd.notna(p10_knee) and pd.notna(p90_knee)
            else None
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
        st.plotly_chart(_frontier_figure(result), width="stretch")
        st.markdown("#### Current-state blend at the knee")
        _render_knee_blend(result)
    with variation_tab:
        st.plotly_chart(_feasibility_figure(result), width="stretch")
        if not varied.empty:
            distribution = go.Figure(
                go.Histogram(
                    x=varied["knee_production_mt"],
                    marker_color="#5c7cfa",
                    nbinsx=min(15, max(5, len(varied) // 2)),
                    hovertemplate=(
                        "Knee %{x:,.0f} MT<br>%{y} scenarios<extra></extra>"
                    ),
                )
            )
            distribution.update_layout(
                height=330,
                margin={"l": 10, "r": 10, "t": 20, "b": 10},
                xaxis_title="Detected knee (MT/day)",
                yaxis_title="Scenarios",
            )
            st.plotly_chart(distribution, width="stretch")
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
                    value=10.0,
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
                        "Stock MT": st.column_config.NumberColumn(min_value=0.0),
                        "Price Rs/MT": st.column_config.NumberColumn(min_value=0.0),
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
                            "Min MT": st.column_config.NumberColumn(min_value=0.0),
                            "Max MT": st.column_config.NumberColumn(min_value=0.0),
                            "Stock MT": st.column_config.NumberColumn(min_value=0.0),
                            "Price Rs/MT": st.column_config.NumberColumn(
                                min_value=0.0
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
                    max_value=50,
                    value=20,
                    step=1,
                    key="bmo_frontier_scenarios",
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
                            f"Elapsed {elapsed:,.1f}s",
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
