"""Streamlit page flow for the Blend Mix Optimizer.

This page wires BMO configuration, data context, fuel-cost model inference,
LP baseline optimization, nonlinear DE optimization, and result rendering into
one Streamlit workflow for ore blend planning.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import replace
from typing import Any

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)

from config.config_loader import load_config
from data.bmo import EvonithBmoContextProvider
from domain.optimization_runtime import build_runtime_config
from ui.bmo import (
    apply_bmo_styles,
    build_dust_editor_df,
    build_flux_editor_df,
    build_fuel_ash_editor_df,
    build_ore_editor_df,
    render_blend_metrics,
    render_blend_table,
    render_diagnostics,
    render_dust_editor,
    render_flux_editor,
    render_fuel_ash_editor,
    render_header,
    render_hot_metal_chemistry,
    render_ore_editor,
    render_slag_balance_details,
    render_slag_balance_settings,
)
from utils.bmo import (
    DustInput,
    FluxInput,
    FuelAshInput,
    FuelUnitCostModelService,
    OreInput,
    SlagBalanceSettings,
    run_lp_baseline,
    run_nonlinear_optimizer,
)
from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
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
    """
    Load the BMO settings block from Streamlit app configuration.

    The page keeps all optimizer, model, data-source, and UI defaults in the
    dedicated BMO settings file. Returning only the ``bmo`` block avoids leaking
    unrelated application settings into the optimizer flow.

    Args:
         - None

    Returns:
         - return dict[str, Any] - BMO configuration dictionary.
    """

    cfg = load_config("setting_bmo.yml")
    return cfg.get("bmo", {})


@_resource_cache(show_spinner=False)
def _get_context_provider() -> EvonithBmoContextProvider:
    """
    Create or return the cached Evonith BMO context provider.

    The provider can perform network and file reads, so Streamlit caches the
    instance across reruns. This keeps the page responsive while preserving the
    same mapping and settings for one app session.

    Args:
         - None

    Returns:
         - return EvonithBmoContextProvider - Cached stock, chemistry, and history provider.
    """

    return EvonithBmoContextProvider(
        setting_path="src/config/setting_bmo.yml",
        mapping_path="src/config/bmo_ore_mapping.yml",
    )


@_resource_cache(show_spinner=False)
def _get_model_service() -> FuelUnitCostModelService:
    """
    Create or return the cached BMO fuel-cost model service.

    Loading the XGBoost model, scaler, and selected feature list is relatively
    expensive. Caching the service keeps model artifacts warm while allowing
    each candidate blend to request fresh predictions.

    Args:
         - None

    Returns:
         - return FuelUnitCostModelService - Cached fuel-cost prediction service.
    """

    bmo_cfg = _get_bmo_config()
    runtime_cfg = build_runtime_config(bmo_cfg)
    bundle_cfg = dict(runtime_cfg.get("model_bundle", {}))
    bundle_cfg.setdefault(
        "missing_feature_policy",
        runtime_cfg.get("feature_policy", {}).get(
            "missing_feature_policy", "default_warn"
        ),
    )
    return FuelUnitCostModelService(
        bundle_cfg=bundle_cfg,
        fallback_cfg=bmo_cfg.get("fallback_fuel_model", {}),
    )


def _load_fuel_prediction_context(
    provider: EvonithBmoContextProvider,
) -> tuple[
    FuelUnitCostModelService,
    dict[str, float],
    pd.DataFrame,
    dict[str, Any],
    list[str],
]:
    """
    Load process, history, and model inputs required for fuel inference.

    Fuel prediction needs the latest process context, recent history for lagged
    features, and the cached model service. Keeping this as a small helper lets
    LP post-solve inference and DE candidate optimization share the same loaded
    inputs during one Streamlit run.

    Args:
         - provider: EvonithBmoContextProvider - Source for process context and history.

    Returns:
         - return tuple[FuelUnitCostModelService, dict[str, float], pd.DataFrame, dict[str, Any], list[str]] - Model service, process context, history, bundle status, and warnings.
    """

    process_context, process_warnings = provider.get_process_context()
    history_df, history_warnings = provider.get_history_frame()
    model_service = _get_model_service()
    bundle_status = model_service.get_bundle_status()
    warnings = [*process_warnings, *history_warnings]
    return model_service, process_context, history_df, bundle_status, warnings


def _selected_ores_from_editor(
    editor_df: pd.DataFrame, base_ores: list[OreInput]
) -> list[OreInput]:
    """
    Convert edited Streamlit ore rows into typed selected ore inputs.

    The editor lets users change wet stock, price, share limits, and moisture.
    Moisture is written back into the nested chemistry object so the optimizer
    uses the same dry-weight Fe calculation that the result table displays.

    Args:
         - editor_df: pd.DataFrame - User-edited ore selection table.
         - base_ores: list[OreInput] - Original ore inputs from the context provider.

    Returns:
         - return list[OreInput] - Selected ores with edited stock, price, and bounds.
    """

    by_id = {ore.ore_id: ore for ore in base_ores}
    selected_ores: list[OreInput] = []

    for _, row in editor_df.iterrows():
        if not bool(row.get("selected", False)):
            continue
        ore_id = str(row["ore_id"])
        if ore_id not in by_id:
            continue
        base = by_id[ore_id]
        moisture_pct = row.get("moisture_pct", base.chemistry.moisture_pct)
        if pd.isna(moisture_pct):
            moisture_pct = base.chemistry.moisture_pct
        selected_ores.append(
            replace(
                base,
                stock_mt=float(row["stock_mt"]),
                price_rs_per_mt=float(row["price_rs_per_mt"]),
                min_share_pct=float(row["min_share_pct"]),
                max_share_pct=float(row["max_share_pct"]),
                chemistry=replace(base.chemistry, moisture_pct=float(moisture_pct)),
            )
        )
    return selected_ores


def _float_from_row(row: pd.Series, key: str, default: float = 0.0) -> float:
    """
    Read one numeric value from a Streamlit editor row.

    Data editor cells can return blank, NaN, or typed numeric values depending
    on Streamlit version and user edits. This helper normalizes those cases so
    fuel ash inputs always receive stable floats.

    Args:
         - row: pd.Series - Edited dataframe row.
         - key: str - Column name to read.
         - default: float - Value to use when the cell is blank or invalid.

    Returns:
         - return float - Parsed numeric value.
    """

    value = row.get(key, default)
    if pd.isna(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _fuel_ash_inputs_from_editor(editor_df: pd.DataFrame) -> list[FuelAshInput]:
    """
    Convert edited fuel ash rows into typed fuel ash inputs.

    The UI table stores fuel rates and ash chemistry as dataframe columns. This
    helper converts those rows into dataclass records consumed by LP, DE, and
    final blend evaluation.

    Args:
         - editor_df: pd.DataFrame - Edited fuel ash table.

    Returns:
         - return list[FuelAshInput] - Fuel ash inputs for slag calculations.
    """

    fuel_ash_inputs: list[FuelAshInput] = []
    if editor_df.empty:
        return fuel_ash_inputs

    for _, row in editor_df.iterrows():
        fuel_id = str(row.get("fuel_id", "")).strip()
        if not fuel_id:
            continue
        fuel_ash_inputs.append(
            FuelAshInput(
                fuel_id=fuel_id,
                display_name=str(row.get("fuel_name", fuel_id)),
                enabled=bool(row.get("enabled", True)),
                rate_kg_per_thm=_float_from_row(row, "rate_kg_per_thm"),
                moisture_pct=_float_from_row(row, "moisture_pct"),
                ash_pct=_float_from_row(row, "ash_pct"),
                sio2_pct=_float_from_row(row, "sio2_pct"),
                al2o3_pct=_float_from_row(row, "al2o3_pct"),
                cao_pct=_float_from_row(row, "cao_pct"),
                mgo_pct=_float_from_row(row, "mgo_pct"),
                fe2o3_pct=_float_from_row(row, "fe2o3_pct"),
                tio2_pct=_float_from_row(row, "tio2_pct"),
                na2o_pct=_float_from_row(row, "na2o_pct"),
                k2o_pct=_float_from_row(row, "k2o_pct"),
                s_pct=_float_from_row(row, "s_pct"),
                p_pct=_float_from_row(row, "p_pct"),
            )
        )
    return fuel_ash_inputs


def _flux_inputs_from_editor(editor_df: pd.DataFrame) -> list[FluxInput]:
    """
    Convert edited fixed-flux rows into typed flux inputs.

    The Streamlit table stores flux wet quantity and chemistry as dataframe
    columns. This helper converts those rows into dataclass records used by
    LP, DE, and final blend evaluation.

    Args:
         - editor_df: pd.DataFrame - Edited fixed-flux table.

    Returns:
         - return list[FluxInput] - Flux inputs for slag calculations.
    """

    flux_inputs: list[FluxInput] = []
    if editor_df.empty:
        return flux_inputs

    for _, row in editor_df.iterrows():
        flux_id = str(row.get("flux_id", "")).strip()
        if not flux_id:
            continue
        flux_inputs.append(
            FluxInput(
                flux_id=flux_id,
                display_name=str(row.get("flux_name", flux_id)),
                enabled=bool(row.get("enabled", True)),
                wet_qty_mt=_float_from_row(row, "wet_qty_mt"),
                moisture_pct=_float_from_row(row, "moisture_pct"),
                sio2_pct=_float_from_row(row, "sio2_pct"),
                al2o3_pct=_float_from_row(row, "al2o3_pct"),
                cao_pct=_float_from_row(row, "cao_pct"),
                mgo_pct=_float_from_row(row, "mgo_pct"),
                fe2o3_pct=_float_from_row(row, "fe2o3_pct"),
                mno_pct=_float_from_row(row, "mno_pct"),
                tio2_pct=_float_from_row(row, "tio2_pct"),
                na2o_pct=_float_from_row(row, "na2o_pct"),
                k2o_pct=_float_from_row(row, "k2o_pct"),
                caf2_pct=_float_from_row(row, "caf2_pct"),
                p_pct=_float_from_row(row, "p_pct"),
                s_pct=_float_from_row(row, "s_pct"),
                zn_pct=_float_from_row(row, "zn_pct"),
                loi_pct=_float_from_row(row, "loi_pct"),
            )
        )
    return flux_inputs


def _dust_inputs_from_editor(editor_df: pd.DataFrame) -> list[DustInput]:
    """
    Convert edited BF gas dust rows into typed dust inputs.

    Dust rows are used only by the full slag-balance calculation. This helper
    converts the editable dust table into dataclass records so component losses
    can be deducted from total BF input.

    Args:
         - editor_df: pd.DataFrame - Edited BF gas dust table.

    Returns:
         - return list[DustInput] - Dust inputs for full slag balance.
    """

    dust_inputs: list[DustInput] = []
    if editor_df.empty:
        return dust_inputs

    for _, row in editor_df.iterrows():
        dust_id = str(row.get("dust_id", "")).strip()
        if not dust_id:
            continue
        dust_inputs.append(
            DustInput(
                dust_id=dust_id,
                display_name=str(row.get("dust_name", dust_id)),
                enabled=bool(row.get("enabled", True)),
                wet_qty_mt=_float_from_row(row, "wet_qty_mt"),
                moisture_pct=_float_from_row(row, "moisture_pct"),
                sio2_pct=_float_from_row(row, "sio2_pct"),
                al2o3_pct=_float_from_row(row, "al2o3_pct"),
                cao_pct=_float_from_row(row, "cao_pct"),
                mgo_pct=_float_from_row(row, "mgo_pct"),
                fe_pct=_float_from_row(row, "fe_pct"),
                mn_pct=_float_from_row(row, "mn_pct"),
                p_pct=_float_from_row(row, "p_pct"),
                s_pct=_float_from_row(row, "s_pct"),
                ti_pct=_float_from_row(row, "ti_pct"),
                zn_pct=_float_from_row(row, "zn_pct"),
                na2o_pct=_float_from_row(row, "na2o_pct"),
                k2o_pct=_float_from_row(row, "k2o_pct"),
                caf2_pct=_float_from_row(row, "caf2_pct"),
            )
        )
    return dust_inputs


def _slag_balance_settings_from_editor(
    settings_values: dict[str, Any],
    hm_chem_values: dict[str, float],
    hm_snapshot: dict[str, Any] | None = None,
) -> SlagBalanceSettings:
    """
    Convert edited slag-balance setting values into a typed settings object.

    PI chemistry (C/Si/S/Others) is sourced from the live HM analysis snapshot
    so the full slag balance subtracts the actual SiO2 consumed by Si reduction
    and the actual S reporting to pig iron. HM Mn% and Ti% are pulled directly
    from the HM snapshot so Mn/Ti partitioning between hot metal and slag
    reflects observed plant chemistry instead of a fixed 60% recovery factor.
    Recovery, gas loss, alkali split, and conversion factors remain editable
    via the advanced expander.

    Args:
         - settings_values: dict[str, Any] - Edited slag-balance setting values.
         - hm_chem_values: dict[str, float] - Edited HM chemistry values from the page.
         - hm_snapshot: dict[str, Any] | None - Raw HM snapshot for per-element fields.

    Returns:
         - return SlagBalanceSettings - Full slag-balance settings.
    """

    snapshot = hm_snapshot or {}
    return SlagBalanceSettings(
        enabled=bool(settings_values.get("enabled", True)),
        carbon_pct=float(hm_chem_values.get("carbon_pct", 0.0)),
        silicon_pct=float(hm_chem_values.get("silicon_pct", 0.0)),
        sulphur_pct=float(hm_chem_values.get("sulphur_pct", 0.0)),
        other_pct=float(hm_chem_values.get("other_pct", 0.0)),
        mn_pct=float(snapshot.get("chem_pct_mn", 0.0) or 0.0),
        ti_pct=float(snapshot.get("chem_pct_ti", 0.0) or 0.0),
        pi_loss_pct=float(settings_values.get("pi_loss_pct", 0.2)),
        fe_to_pig_iron_fraction=float(
            settings_values.get("fe_to_pig_iron_fraction", 0.999)
        ),
        mn_recovery_pct=float(settings_values.get("mn_recovery_pct", 60.0)),
        sulphur_gas_loss_pct=float(settings_values.get("sulphur_gas_loss_pct", 10.0)),
        alkali_to_slag_fraction=float(
            settings_values.get("alkali_to_slag_fraction", 0.8)
        ),
        si_to_sio2_factor=float(settings_values.get("si_to_sio2_factor", 2.14)),
        fe_to_feo_factor=float(settings_values.get("fe_to_feo_factor", 72.0 / 56.0)),
        mn_to_mno_factor=float(settings_values.get("mn_to_mno_factor", 1.291)),
        slag_correction_factor=float(
            settings_values.get("slag_correction_factor", 1.0)
        ),
    )


apply_bmo_styles()
bmo_cfg = _get_bmo_config()
provider = _get_context_provider()
model_service = _get_model_service()
bundle_status = model_service.get_bundle_status()
st.session_state["bmo_bundle_status"] = bundle_status
render_header(bundle_status)

ui_cfg = bmo_cfg.get("ui", {})
target_cfg = bmo_cfg.get("target", {})
runtime_cfg = build_runtime_config(
    bmo_cfg, default_optimizer=bmo_cfg.get("optimization", {})
)
opt_cfg = runtime_cfg.get("optimizer", {})

layout_col1, layout_col2, layout_col3, layout_col4 = st.columns(4)

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

target_production_mt = layout_col3.number_input(
    "Target Hot Metal (MT)",
    min_value=0.0,
    value=float(target_cfg.get("target_production_mt", 2350.0)),
    step=5.0,
)
target_slag_qty_mt = layout_col4.number_input(
    "Max Slag (MT)",
    min_value=0.0,
    value=float(target_cfg.get("target_slag_qty_mt", 750.0)),
    step=5.0,
)


ores, ore_diagnostics = provider.build_ore_inputs(
    mode=chemistry_mode, window_days=chemistry_window_days
)
ore_diagnostics["warnings"] = list(ore_diagnostics.get("warnings", []))

hm_snapshot = provider.get_hm_slag_snapshot(
    mode=chemistry_mode, window_days=chemistry_window_days
)
ore_diagnostics["warnings"].extend(hm_snapshot.get("warnings", []))
observed_slag_rate = float(hm_snapshot.get("observed_slag_rate_kg_per_thm", 0.0) or 0.0)

default_selected_names = set(ui_cfg.get("default_selected_ores", []))
default_selected_ids = [
    ore.ore_id for ore in ores if ore.display_name in default_selected_names
]
editor_df = build_ore_editor_df(ores, default_selected_ids=default_selected_ids)

st.markdown("### Ore Selection, Stock, Pricing, and Share Bounds")
edited_df = render_ore_editor(editor_df)

fuel_ash_df = build_fuel_ash_editor_df(bmo_cfg.get("fuel_ash_inputs", []))
if not fuel_ash_df.empty:
    st.markdown("### Fuel Ash Inputs")
    edited_fuel_ash_df = render_fuel_ash_editor(fuel_ash_df)
else:
    edited_fuel_ash_df = fuel_ash_df
fuel_ash_inputs = _fuel_ash_inputs_from_editor(edited_fuel_ash_df)

flux_df = build_flux_editor_df(bmo_cfg.get("flux_inputs", []))
if not flux_df.empty:
    st.markdown("### Flux Inputs")
    edited_flux_df = render_flux_editor(flux_df)
else:
    edited_flux_df = flux_df
flux_inputs = _flux_inputs_from_editor(edited_flux_df)

hm_chem_values = render_hot_metal_chemistry(
    hm_snapshot, bmo_cfg.get("slag_balance", {})
)

with st.expander("Advanced Slag Balance Inputs", expanded=False):
    slag_settings_values = render_slag_balance_settings(bmo_cfg.get("slag_balance", {}))
    dust_df = build_dust_editor_df(bmo_cfg.get("dust_inputs", []))
    if not dust_df.empty:
        st.markdown("##### BF Gas Dust")
        edited_dust_df = render_dust_editor(dust_df)
    else:
        edited_dust_df = dust_df
dust_inputs = _dust_inputs_from_editor(edited_dust_df)
slag_balance_settings = _slag_balance_settings_from_editor(
    slag_settings_values, hm_chem_values, hm_snapshot
)

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
        feo_in_slag_pct = float(
            bmo_cfg.get("chemistry", {}).get("feo_in_slag_pct", 0.4)
        )
        fuel_context = None

        with st.spinner("Running LP baseline..."):
            lp_result, lp_errors = run_lp_baseline(
                selected_ores,
                target_production_mt=target_production_mt,
                target_slag_qty_mt=target_slag_qty_mt,
                feo_in_slag_pct=feo_in_slag_pct,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
            )
            if lp_result is not None:
                lp_physical_result = lp_result
                fuel_context = _load_fuel_prediction_context(provider)
                (
                    model_service,
                    process_context,
                    history_df,
                    bundle_status,
                    fuel_warnings,
                ) = fuel_context
                ore_diagnostics["warnings"] = [
                    *ore_diagnostics.get("warnings", []),
                    *fuel_warnings,
                ]
                st.session_state["bmo_bundle_status"] = bundle_status
                lp_result = evaluate_blend_with_fuel_prediction(
                    ores=selected_ores,
                    quantities_mt=lp_physical_result.quantities_mt,
                    feo_in_slag_pct=feo_in_slag_pct,
                    model_service=model_service,
                    process_context=process_context,
                    history_df=history_df,
                    fuel_ash_inputs=fuel_ash_inputs,
                    flux_inputs=flux_inputs,
                    dust_inputs=dust_inputs,
                    slag_balance_settings=slag_balance_settings,
                )
                lp_result.feasible = lp_physical_result.feasible
                lp_result.violations = lp_physical_result.violations
        st.session_state["bmo_lp_result"] = lp_result
        st.session_state["bmo_lp_errors"] = lp_errors

        if run_total_clicked:
            de_status = st.status(
                "Total Cost Optimizer (DE) running...", expanded=True
            )
            iteration_lines: list[str] = []

            def _de_progress(
                iteration: int,
                best_obj: float,
                best_feas: float | None,
                nfev: int,
                elapsed_s: float,
            ) -> bool:
                """
                Stream DE iteration progress to the Streamlit status panel + log.

                Args:
                     - iteration: int - 1-based DE generation index.
                     - best_obj: float - Best (penalized) objective value seen so far in Rs/THM.
                     - best_feas: float | None - Best feasible objective seen so far, if any.
                     - nfev: int - Cumulative function-evaluation count.
                     - elapsed_s: float - Seconds elapsed since DE started.

                Returns:
                     - return bool - False to keep running (no user-cancel wired yet).
                """

                feas_txt = (
                    f", best feasible {best_feas:,.1f}"
                    if best_feas is not None
                    else ""
                )
                line = (
                    f"Iter {iteration:>2}  best {best_obj:,.1f} Rs/THM{feas_txt}"
                    f"  (nfev={nfev}, {elapsed_s:.1f}s)"
                )
                iteration_lines.append(line)
                de_status.write(line)
                log.info("BMO DE %s", line)
                return False

            if fuel_context is None:
                fuel_context = _load_fuel_prediction_context(provider)
                (
                    model_service,
                    process_context,
                    history_df,
                    bundle_status,
                    fuel_warnings,
                ) = fuel_context
                ore_diagnostics["warnings"] = [
                    *ore_diagnostics.get("warnings", []),
                    *fuel_warnings,
                ]
                st.session_state["bmo_bundle_status"] = bundle_status
            else:
                (
                    model_service,
                    process_context,
                    history_df,
                    bundle_status,
                    _fuel_warnings,
                ) = fuel_context
            de_result, de_errors = run_nonlinear_optimizer(
                selected_ores,
                target_production_mt=target_production_mt,
                target_slag_qty_mt=target_slag_qty_mt,
                feo_in_slag_pct=feo_in_slag_pct,
                model_service=model_service,
                process_context=process_context,
                history_df=history_df,
                de_cfg=opt_cfg,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                progress_callback=_de_progress,
            )
            de_status.update(
                label=(
                    f"DE finished - {len(iteration_lines)} iterations"
                    if iteration_lines
                    else "DE finished"
                ),
                state="complete",
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
            render_blend_metrics(
                "LP Baseline Result",
                lp_result,
                observed_slag_rate_kg_per_thm=observed_slag_rate,
            )
            render_blend_table(lp_result, selected_ores)
            render_slag_balance_details(
                lp_result, selected_ores, fuel_ash_inputs, flux_inputs
            )
        else:
            st.info("Run LP baseline to see deterministic cost-minimized blend.")

    with tab_de:
        if de_result is not None:
            render_blend_metrics(
                "DE Total-Cost Result",
                de_result,
                observed_slag_rate_kg_per_thm=observed_slag_rate,
            )
            render_blend_table(de_result, selected_ores)
            render_slag_balance_details(
                de_result, selected_ores, fuel_ash_inputs, flux_inputs
            )
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
                        "slag_rate_kg_per_thm": lp_result.slag_rate_kg_per_thm,
                        "feasible": lp_result.feasible,
                    },
                    {
                        "method": "DE total cost",
                        "objective_rs_per_thm": de_result.objective_rs_per_thm,
                        "ore_cost_rs_per_thm": de_result.ore_cost_per_thm_rs,
                        "fuel_cost_rs_per_thm": de_result.fuel_cost_per_thm_rs,
                        "fe_production_mt": de_result.fe_production_mt,
                        "slag_mt": de_result.slag_mt,
                        "slag_rate_kg_per_thm": de_result.slag_rate_kg_per_thm,
                        "feasible": de_result.feasible,
                    },
                ]
            )
            df_kwargs: dict[str, Any] = {}
            dataframe_sig = inspect.signature(st.dataframe)
            if "hide_index" in dataframe_sig.parameters:
                df_kwargs["hide_index"] = True
            if "width" in dataframe_sig.parameters:
                df_kwargs["width"] = "stretch"
            elif "use_container_width" in dataframe_sig.parameters:
                df_kwargs["use_container_width"] = True
            st.dataframe(comparison_df, **df_kwargs)
        else:
            st.info(
                "Run both LP and DE to compare deterministic and nonlinear solutions."
            )

render_diagnostics(de_result or lp_result, ore_diagnostics)
