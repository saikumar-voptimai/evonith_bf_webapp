from __future__ import annotations

from dataclasses import replace
import inspect
from typing import Any

import pandas as pd
import streamlit as st

from config.config_loader import load_config
from data.bmo import EvonithBmoContextProvider
from domain.bmo import (
    FuelUnitCostModelService,
    OreInput,
    run_lp_baseline,
    run_nonlinear_optimizer,
)
from domain.optimization_runtime import build_runtime_config
from ui.bmo import (
    apply_bmo_styles,
    build_ore_editor_df,
    render_blend_metrics,
    render_blend_table,
    render_diagnostics,
    render_header,
    render_ore_editor,
)
from utils.session import is_logged_in


if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()


if hasattr(st, "cache_resource"):
    _resource_cache = st.cache_resource
elif hasattr(st, "experimental_singleton"):
    _resource_cache = st.experimental_singleton
else:
    _resource_cache = st.cache


def _get_bmo_config() -> dict[str, Any]:
    cfg = load_config("setting_bmo.yml")
    return cfg.get("bmo", {})


@_resource_cache(show_spinner=False)
def _get_context_provider() -> EvonithBmoContextProvider:
    return EvonithBmoContextProvider(
        setting_path="src/config/setting_bmo.yml",
        mapping_path="src/config/bmo_ore_mapping.yml",
    )


@_resource_cache(show_spinner=False)
def _get_model_service() -> FuelUnitCostModelService:
    bmo_cfg = _get_bmo_config()
    runtime_cfg = build_runtime_config(bmo_cfg)
    bundle_cfg = dict(runtime_cfg.get("model_bundle", {}))
    bundle_cfg.setdefault(
        "missing_feature_policy",
        runtime_cfg.get("feature_policy", {}).get("missing_feature_policy", "default_warn"),
    )
    return FuelUnitCostModelService(
        bundle_cfg=bundle_cfg,
        fallback_cfg=bmo_cfg.get("fallback_fuel_model", {}),
    )


def _selected_ores_from_editor(editor_df: pd.DataFrame, base_ores: list[OreInput]) -> list[OreInput]:
    by_id = {ore.ore_id: ore for ore in base_ores}
    selected_ores: list[OreInput] = []

    for _, row in editor_df.iterrows():
        if not bool(row.get("selected", False)):
            continue
        ore_id = str(row["ore_id"])
        if ore_id not in by_id:
            continue
        base = by_id[ore_id]
        selected_ores.append(
            replace(
                base,
                stock_mt=float(row["stock_mt"]),
                price_rs_per_mt=float(row["price_rs_per_mt"]),
                min_share_pct=float(row["min_share_pct"]),
                max_share_pct=float(row["max_share_pct"]),
            )
        )
    return selected_ores


apply_bmo_styles()
bmo_cfg = _get_bmo_config()
provider = _get_context_provider()
model_service = _get_model_service()
bundle_status = model_service.get_bundle_status()
render_header(bundle_status)

st.caption(
    "This page is dedicated to BMO integration (ticket 104). "
    "It does not reuse recommendation business logic."
)

ui_cfg = bmo_cfg.get("ui", {})
target_cfg = bmo_cfg.get("target", {})
runtime_cfg = build_runtime_config(bmo_cfg, default_optimizer=bmo_cfg.get("optimization", {}))
opt_cfg = runtime_cfg.get("optimizer", {})

layout_col1, layout_col2 = st.columns(2)

with layout_col1:
    chemistry_mode = st.selectbox(
        "Chemistry mode",
        options=["latest", "avg"],
        index=0 if str(bmo_cfg.get("chemistry_mode", "latest")) == "latest" else 1,
    )

with layout_col2:
    chemistry_window_days = st.slider(
        "Chemistry window (days)",
        min_value=1,
        max_value=90,
        value=int(bmo_cfg.get("chemistry_window_days", 30)),
    )

target_col1, target_col2, target_col3, target_col4 = st.columns(4)
target_total_qty_mt = target_col1.number_input(
    "Target total burden (MT)",
    min_value=1.0,
    value=float(target_cfg.get("target_total_qty_mt", 3800.0)),
    step=10.0,
)
min_fe_production_mt = target_col2.number_input(
    "Min Fe (MT)",
    min_value=0.0,
    value=float(target_cfg.get("min_fe_production_mt", 2350.0)),
    step=5.0,
)
max_fe_production_mt = target_col3.number_input(
    "Max Fe (MT)",
    min_value=0.0,
    value=float(target_cfg.get("max_fe_production_mt", 2500.0)),
    step=5.0,
)
target_slag_qty_mt = target_col4.number_input(
    "Max Slag (MT)",
    min_value=0.0,
    value=float(target_cfg.get("target_slag_qty_mt", 750.0)),
    step=5.0,
)


ores, ore_diagnostics = provider.build_ore_inputs(
    mode=chemistry_mode, window_days=chemistry_window_days
)
process_context, process_warnings = provider.get_process_context()
history_df, history_warnings = provider.get_history_frame()
all_warnings = [*ore_diagnostics.get("warnings", []), *process_warnings, *history_warnings]
ore_diagnostics["warnings"] = all_warnings

default_selected_names = set(ui_cfg.get("default_selected_ores", []))
default_selected_ids = [
    ore.ore_id for ore in ores if ore.display_name in default_selected_names
]
editor_df = build_ore_editor_df(ores, default_selected_ids=default_selected_ids)

st.markdown("### Ore Selection, Stock, Pricing, and Share Bounds")
edited_df = render_ore_editor(editor_df)

run_lp_clicked = False
run_total_clicked = False
with st.form("bmo_run_form", clear_on_submit=False):
    run_col1, run_col2 = st.columns(2)
    run_lp_clicked = run_col1.form_submit_button("Run LP Baseline")
    run_total_clicked = run_col2.form_submit_button("Run Total Cost Optimizer (DE)")

selected_ores = _selected_ores_from_editor(edited_df, ores)

if run_lp_clicked or run_total_clicked:
    if len(selected_ores) < 2:
        st.error("Select at least two ores before running optimization.")
    else:
        feo_in_slag_pct = float(bmo_cfg.get("chemistry", {}).get("feo_in_slag_pct", 0.4))
        si_in_slag_pct = float(bmo_cfg.get("chemistry", {}).get("si_in_slag_pct", 0.5))

        lp_result, lp_errors = run_lp_baseline(
            selected_ores,
            target_total_qty_mt=target_total_qty_mt,
            min_fe_production_mt=min_fe_production_mt,
            max_fe_production_mt=max_fe_production_mt,
            target_slag_qty_mt=target_slag_qty_mt,
            feo_in_slag_pct=feo_in_slag_pct,
            si_in_slag_pct=si_in_slag_pct,
        )
        st.session_state["bmo_lp_result"] = lp_result
        st.session_state["bmo_lp_errors"] = lp_errors

        if run_total_clicked:
            de_result, de_errors = run_nonlinear_optimizer(
                selected_ores,
                target_total_qty_mt=target_total_qty_mt,
                min_fe_production_mt=min_fe_production_mt,
                max_fe_production_mt=max_fe_production_mt,
                target_slag_qty_mt=target_slag_qty_mt,
                feo_in_slag_pct=feo_in_slag_pct,
                si_in_slag_pct=si_in_slag_pct,
                model_service=model_service,
                process_context=process_context,
                history_df=history_df,
                de_cfg=opt_cfg,
            )
            st.session_state["bmo_de_result"] = de_result
            st.session_state["bmo_de_errors"] = de_errors
        else:
            st.session_state.pop("bmo_de_result", None)
            st.session_state.pop("bmo_de_errors", None)


lp_result = st.session_state.get("bmo_lp_result")
lp_errors = st.session_state.get("bmo_lp_errors", [])
de_result = st.session_state.get("bmo_de_result")
de_errors = st.session_state.get("bmo_de_errors", [])

if lp_errors:
    st.error("LP baseline errors:\n- " + "\n- ".join(lp_errors))
if de_errors:
    st.error("Total cost optimizer errors:\n- " + "\n- ".join(de_errors))

if lp_result is not None or de_result is not None:
    tab_lp, tab_de, tab_cmp = st.tabs(["LP Baseline", "Total Cost (DE)", "Comparison"])

    with tab_lp:
        if lp_result is not None:
            render_blend_metrics("LP Baseline Result", lp_result)
            render_blend_table(lp_result, selected_ores)
        else:
            st.info("Run LP baseline to see deterministic cost-minimized blend.")

    with tab_de:
        if de_result is not None:
            render_blend_metrics("DE Total-Cost Result", de_result)
            render_blend_table(de_result, selected_ores)
            if not bundle_status.get("model_loaded"):
                st.info(
                    "Model artifact is missing. Fuel term is currently coming from fallback "
                    "formula until model/scaler bundle is provided."
                )
        else:
            st.info("Run total-cost optimizer to see ore + fuel optimized blend.")

    with tab_cmp:
        if lp_result is not None and de_result is not None:
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric(
                "Objective Delta (DE - LP) Rs/THM",
                f"{de_result.objective_rs_per_thm - lp_result.objective_rs_per_thm:+,.2f}",
            )
            cc2.metric(
                "Ore Cost Delta (DE - LP) Rs/THM",
                f"{de_result.ore_cost_per_thm_rs - lp_result.ore_cost_per_thm_rs:+,.2f}",
            )
            cc3.metric(
                "Fuel Cost Delta (DE - LP) Rs/THM",
                f"{de_result.fuel_cost_per_thm_rs - lp_result.fuel_cost_per_thm_rs:+,.2f}",
            )

            comparison_df = pd.DataFrame(
                [
                    {
                        "method": "LP baseline",
                        "objective_rs_per_thm": lp_result.objective_rs_per_thm,
                        "ore_cost_rs_per_thm": lp_result.ore_cost_per_thm_rs,
                        "fuel_cost_rs_per_thm": lp_result.fuel_cost_per_thm_rs,
                        "fe_production_mt": lp_result.fe_production_mt,
                        "slag_mt": lp_result.slag_mt,
                        "feasible": lp_result.feasible,
                    },
                    {
                        "method": "DE total cost",
                        "objective_rs_per_thm": de_result.objective_rs_per_thm,
                        "ore_cost_rs_per_thm": de_result.ore_cost_per_thm_rs,
                        "fuel_cost_rs_per_thm": de_result.fuel_cost_per_thm_rs,
                        "fe_production_mt": de_result.fe_production_mt,
                        "slag_mt": de_result.slag_mt,
                        "feasible": de_result.feasible,
                    },
                ]
            )
            df_kwargs: dict[str, Any] = {}
            dataframe_sig = inspect.signature(st.dataframe)
            if "hide_index" in dataframe_sig.parameters:
                df_kwargs["hide_index"] = True
            if "use_container_width" in dataframe_sig.parameters:
                df_kwargs["use_container_width"] = True
            st.dataframe(comparison_df, **df_kwargs)
        else:
            st.info("Run both LP and DE to compare deterministic and nonlinear solutions.")

render_diagnostics(de_result or lp_result, ore_diagnostics)
