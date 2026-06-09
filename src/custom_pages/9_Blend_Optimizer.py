"""Streamlit page flow for the Blend Mix Optimizer.

This page wires BMO configuration, data context, fuel-cost model inference,
LP baseline optimization, nonlinear DE optimization, and result rendering into
one Streamlit workflow for ore blend planning.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

log = logging.getLogger(__name__)

from config.config_loader import load_config
from data.bmo import EvonithBmoContextProvider
from data.ml.static_dataset_manager import StaticDatasetManager
from domain.optimization_runtime import build_runtime_config
from ui.streamlit_fragments import fragment, rerun_fragment
from ui.bmo import (
    apply_bmo_styles,
    build_dust_editor_df,
    build_fuel_ash_editor_df,
    build_ore_editor_df,
    render_blend_metrics,
    render_blend_table,
    render_diagnostics,
    render_dust_editor,
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
    validate_selected_pellet_inputs,
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


def _static_dataset_manager(bmo_cfg: dict[str, Any]) -> StaticDatasetManager:
    data_sources = bmo_cfg.get("data_sources", {}) or {}
    static_path = data_sources.get(
        "static_dataset_path", "src/assets/data/furnace_dataset.csv"
    )
    return StaticDatasetManager(static_path)


def _parse_meta_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _static_dataset_status(bmo_cfg: dict[str, Any]) -> dict[str, Any]:
    manager = _static_dataset_manager(bmo_cfg)
    meta = manager.get_meta()
    csv_path = manager.current_csv_path()
    last_updated = _parse_meta_datetime(meta.last_updated if meta else None)
    max_age_minutes = int(
        (bmo_cfg.get("data_sources", {}) or {}).get(
            "static_refresh_max_age_minutes", 60
        )
        or 60
    )
    now = (
        datetime.now(last_updated.tzinfo)
        if last_updated and last_updated.tzinfo
        else datetime.now()
    )
    stale = (
        last_updated is None
        or now - last_updated >= timedelta(minutes=max_age_minutes)
    )
    return {
        "manager": manager,
        "meta": meta,
        "csv_path": csv_path,
        "exists": csv_path.exists(),
        "last_updated": last_updated,
        "latest_data_end": meta.raw_end if meta else "",
        "state": "stale" if stale else "fresh",
        "max_age_minutes": max_age_minutes,
    }


def _refresh_static_dataset_if_needed(
    bmo_cfg: dict[str, Any], *, force: bool = False
) -> dict[str, Any]:
    status = _static_dataset_status(bmo_cfg)
    if not force and status["exists"] and status["state"] == "fresh":
        return {"refreshed": False, "usable": True, "status": status}

    manager: StaticDatasetManager = status["manager"]
    try:
        df = manager.update_static("Full")
        saved_path = manager.save(df)
        return {
            "refreshed": True,
            "usable": True,
            "saved_path": saved_path,
            "status": _static_dataset_status(bmo_cfg),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "refreshed": False,
            "usable": bool(status["exists"]),
            "error": str(exc),
            "status": status,
        }


@fragment
def _render_static_dataset_bar(bmo_cfg: dict[str, Any]) -> bool:
    status = _static_dataset_status(bmo_cfg)
    state = status["state"] if status["exists"] else "missing"
    force = False
    with st.expander("Static dataset refresh", expanded=False):
        cols = st.columns([1.1, 1.2, 1.0, 1.0, 1.0])
        cols[0].metric("Static Dataset", state.title())
        cols[1].metric("Latest Data", status["latest_data_end"] or "Unknown")
        last_updated = status["last_updated"]
        cols[2].metric(
            "Last Refresh",
            last_updated.strftime("%Y-%m-%d %H:%M") if last_updated else "Never",
        )
        force = cols[3].checkbox(
            "Force refresh on run",
            value=False,
            key="bmo_force_static_refresh",
        )
        notice = st.session_state.pop("bmo_static_refresh_notice", None)
        if notice:
            st.success(str(notice))
        if cols[4].button("Refresh now"):
            with st.spinner("Refreshing static ML dataset..."):
                result = _refresh_static_dataset_if_needed(bmo_cfg, force=True)
            if result.get("error"):
                st.error(f"Static dataset refresh failed: {result['error']}")
            else:
                st.session_state["bmo_static_refresh_notice"] = (
                    "Static ML dataset refreshed."
                )
                rerun_fragment()
        if state == "stale":
            st.caption("Static dataset is older than the BMO one-hour refresh gate.")
        elif state == "missing":
            st.warning("Static ML dataset is missing; optimizer run will refresh first.")
    return force


def _context_group(field: str) -> str:
    name = field.upper()
    if any(token in name for token in ("WEIGHTED_COKE", "WEIGHTED_NON_COKE", "PORTIONS", "DISCHARGE_TIME")):
        return "burden_distribution"
    if name.startswith("CHEM_") or name.startswith("SLAG_") or name.startswith("HMT"):
        return "hm_slag"
    if "CALC_MT" in name or "CALC_THM" in name or name.endswith("_PCT"):
        return "charge_quantities"
    if any(token in name for token in ("SIO2", "AL2O3", "FE_TOTAL", "FE(T)", "MGO", "CAO", "TIO2", "MNO", "BASICITY")):
        return "rm_composition"
    return "process_params"


def _show_df(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("No diagnostic rows captured.")
    else:
        st.dataframe(df, hide_index=True, width="stretch")


def _form_submit_button(container: Any, label: str, **kwargs: Any) -> bool:
    sig = inspect.signature(container.form_submit_button)
    supported = {
        key: value for key, value in kwargs.items() if key in sig.parameters
    }
    return bool(container.form_submit_button(label, **supported))


def _render_share_pie(blend: Any, selected_ores: list[OreInput], title: str) -> None:
    rows = [
        {
            "ore_name": ore.display_name,
            "share_pct": float(blend.shares_pct.get(ore.ore_id, 0.0)),
        }
        for ore in selected_ores
        if float(blend.shares_pct.get(ore.ore_id, 0.0)) > 0
    ]
    if not rows:
        return
    fig = px.pie(
        pd.DataFrame(rows),
        names="ore_name",
        values="share_pct",
        title=title,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, width="stretch")


def _manual_quantities_for_target(
    snapshot: dict[str, Any],
    selected_ores: list[OreInput],
    target_fe_mt: float,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    rows_by_ore = {str(row.get("ore_id")): row for row in snapshot.get("rows", [])}
    raw_quantities = {
        ore.ore_id: max(
            0.0,
            float(rows_by_ore.get(ore.ore_id, {}).get("quantity_mt", 0.0) or 0.0),
        )
        for ore in selected_ores
    }
    total_raw_qty = sum(raw_quantities.values())
    if total_raw_qty <= 0:
        return (
            {},
            raw_quantities,
            ["Last completed shift has no selected-material charge."],
        )

    shares = {ore_id: qty / total_raw_qty for ore_id, qty in raw_quantities.items()}
    fe_per_blend_mt = 0.0
    for ore in selected_ores:
        dry_fraction = max(0.0, 1.0 - float(ore.chemistry.moisture_pct) / 100.0)
        fe_fraction = max(0.0, float(ore.chemistry.fe_t_pct) / 100.0)
        fe_per_blend_mt += shares.get(ore.ore_id, 0.0) * dry_fraction * fe_fraction
    if fe_per_blend_mt <= 0:
        return {}, raw_quantities, ["Manual blend Fe% is unavailable; cannot scale to target HM."]

    total_target_qty = float(target_fe_mt) / fe_per_blend_mt
    target_quantities = {
        ore_id: share * total_target_qty for ore_id, share in shares.items()
    }
    return target_quantities, raw_quantities, []


def _render_manual_blend_comparison(
    provider: EvonithBmoContextProvider,
    blend: Any,
    selected_ores: list[OreInput],
    title: str,
    *,
    target_fe_mt: float,
    target_production_mt: float,
    feo_in_slag_pct: float,
    fuel_ash_inputs: list[FuelAshInput],
    flux_inputs: list[FluxInput],
    dust_inputs: list[DustInput],
    slag_balance_settings: SlagBalanceSettings,
) -> None:
    snapshot = provider.get_recent_manual_blend_snapshot(selected_ores)
    rows_by_ore = {
        str(row.get("ore_id")): row for row in snapshot.get("rows", [])
    }
    if not rows_by_ore:
        st.info("Recent manual blend is unavailable for the last completed shift.")
        return

    rows = []
    for ore in selected_ores:
        manual = rows_by_ore.get(ore.ore_id, {})
        suggested_share = float(blend.shares_pct.get(ore.ore_id, 0.0))
        manual_share = float(manual.get("share_pct", 0.0) or 0.0)
        rows.append(
            {
                "ore_name": ore.display_name,
                "manual_share_pct": manual_share,
                "suggested_share_pct": suggested_share,
                "delta_share_pct": suggested_share - manual_share,
                "manual_shift_qty_mt": float(manual.get("quantity_mt", 0.0) or 0.0),
                "suggested_qty_mt": float(blend.quantities_mt.get(ore.ore_id, 0.0)),
            }
        )
    comparison_df = pd.DataFrame(rows).sort_values(
        "suggested_share_pct", ascending=False
    )
    st.markdown(f"##### Last Shift Manual Blend vs {title}")
    start_time = snapshot.get("start_time")
    end_time = snapshot.get("end_time")
    if start_time and end_time:
        st.caption(f"Manual blend window: {start_time} to {end_time}")
    for warning in snapshot.get("warnings", []):
        st.warning(str(warning))

    manual_quantities, _manual_shift_quantities, scale_warnings = (
        _manual_quantities_for_target(snapshot, selected_ores, target_fe_mt)
    )
    for warning in scale_warnings:
        st.warning(warning)

    manual_blend = None
    if manual_quantities:
        try:
            (
                model_service,
                process_context,
                history_df,
                _bundle_status,
                fuel_warnings,
            ) = _load_fuel_prediction_context(provider)
            manual_blend = evaluate_blend_with_fuel_prediction(
                ores=selected_ores,
                quantities_mt=manual_quantities,
                feo_in_slag_pct=feo_in_slag_pct,
                model_service=model_service,
                process_context=process_context,
                history_df=history_df,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                hot_metal_target_mt=target_production_mt,
            )
            for warning in fuel_warnings:
                st.warning(str(warning))
        except Exception as exc:
            st.warning(f"Could not evaluate manual blend cost/slag: {exc}")

    if manual_blend is not None:
        savings_rs_per_thm = (
            manual_blend.objective_rs_per_thm - blend.objective_rs_per_thm
        )
        run_savings_rs = savings_rs_per_thm * float(target_production_mt)
        ore_savings_rs_per_thm = (
            manual_blend.ore_cost_per_thm_rs - blend.ore_cost_per_thm_rs
        )
        fuel_savings_rs_per_thm = (
            manual_blend.fuel_cost_per_thm_rs - blend.fuel_cost_per_thm_rs
        )
        slag_reduction_mt = manual_blend.slag_mt - blend.slag_mt

        st.markdown("##### Cost and Slag Impact")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Savings (Rs/THM)",
            f"{savings_rs_per_thm:,.2f}",
            delta=f"{savings_rs_per_thm:+,.2f}",
        )
        k2.metric(
            "Savings for Target HM",
            f"Rs {run_savings_rs:,.0f}",
            delta=f"{run_savings_rs:+,.0f}",
        )
        k3.metric(
            "Ore Cost Saving (Rs/THM)",
            f"{ore_savings_rs_per_thm:,.2f}",
            delta=f"{ore_savings_rs_per_thm:+,.2f}",
        )
        k4.metric(
            "Fuel Cost Saving (Rs/THM)",
            f"{fuel_savings_rs_per_thm:,.2f}",
            delta=f"{fuel_savings_rs_per_thm:+,.2f}",
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Manual Total Cost", f"{manual_blend.objective_rs_per_thm:,.2f}")
        s2.metric(f"{title} Total Cost", f"{blend.objective_rs_per_thm:,.2f}")
        s3.metric(
            "Slag Reduction (MT)",
            f"{slag_reduction_mt:+,.2f}",
            delta=f"{slag_reduction_mt:+,.2f}",
        )
        s4.metric(
            "Slag Rate Reduction (kg/THM)",
            f"{manual_blend.slag_rate_kg_per_thm - blend.slag_rate_kg_per_thm:+,.2f}",
        )

        report_df = pd.DataFrame(
            [
                {
                    "blend": "Last shift manual, scaled to target",
                    "total_cost_rs_per_thm": manual_blend.objective_rs_per_thm,
                    "ore_cost_rs_per_thm": manual_blend.ore_cost_per_thm_rs,
                    "fuel_cost_rs_per_thm": manual_blend.fuel_cost_per_thm_rs,
                    "slag_mt": manual_blend.slag_mt,
                    "slag_rate_kg_per_thm": manual_blend.slag_rate_kg_per_thm,
                    "fe_production_mt": manual_blend.fe_production_mt,
                    "wet_qty_mt": manual_blend.total_qty_mt,
                    "final_fe_pct": manual_blend.fe_t_pct,
                },
                {
                    "blend": title,
                    "total_cost_rs_per_thm": blend.objective_rs_per_thm,
                    "ore_cost_rs_per_thm": blend.ore_cost_per_thm_rs,
                    "fuel_cost_rs_per_thm": blend.fuel_cost_per_thm_rs,
                    "slag_mt": blend.slag_mt,
                    "slag_rate_kg_per_thm": blend.slag_rate_kg_per_thm,
                    "fe_production_mt": blend.fe_production_mt,
                    "wet_qty_mt": blend.total_qty_mt,
                    "final_fe_pct": blend.fe_t_pct,
                },
            ]
        )
        st.dataframe(report_df, hide_index=True, width="stretch")

        pie_left, pie_right = st.columns(2)
        with pie_left:
            _render_share_pie(manual_blend, selected_ores, "Last Shift Manual Mix")
        with pie_right:
            _render_share_pie(blend, selected_ores, f"{title} Mix")

    if manual_blend is not None:
        comparison_df["manual_target_qty_mt"] = comparison_df["ore_name"].map(
            {
                ore.display_name: float(manual_blend.quantities_mt.get(ore.ore_id, 0.0))
                for ore in selected_ores
            }
        )
        comparison_df["suggested_ore_cost_rs"] = comparison_df.apply(
            lambda row: float(row["suggested_qty_mt"])
            * next(
                float(ore.price_rs_per_mt)
                for ore in selected_ores
                if ore.display_name == row["ore_name"]
            ),
            axis=1,
        )
        comparison_df["manual_ore_cost_rs"] = comparison_df.apply(
            lambda row: float(row["manual_target_qty_mt"])
            * next(
                float(ore.price_rs_per_mt)
                for ore in selected_ores
                if ore.display_name == row["ore_name"]
            ),
            axis=1,
        )

    st.markdown("##### Blend Details")
    st.dataframe(comparison_df, hide_index=True, width="stretch")


@fragment
def _render_data_diagnostics(
    provider: EvonithBmoContextProvider,
    *,
    ore_diagnostics: dict[str, Any],
    hm_snapshot: dict[str, Any],
    edited_ore_df: pd.DataFrame,
    expanded: bool,
) -> None:
    model_service = _get_model_service()
    history_df, history_warnings = provider.get_history_frame(
        online_lag_hours=model_service.get_max_lag_steps()
    )
    process_context, process_warnings = provider.get_process_context(
        history_df=history_df
    )
    provider.get_charge_mix_snapshot()
    diagnostics = provider.get_data_diagnostics()
    warnings = [
        *ore_diagnostics.get("warnings", []),
        *hm_snapshot.get("warnings", []),
        *history_warnings,
        *process_warnings,
    ]

    with st.expander("Data Diagnostics", expanded=expanded):
        if st.button("Refresh diagnostics", key="bmo_refresh_diagnostics"):
            rerun_fragment()

        source_rows = []
        stock = diagnostics.get("stock", {})
        chemistry = diagnostics.get("chemistry", {})
        hm = diagnostics.get("hm_slag", {})
        dpr = diagnostics.get("dpr", {})
        history = diagnostics.get("history", {})
        process = diagnostics.get("process", {})
        flux = diagnostics.get("flux", {})
        charge_mix = diagnostics.get("charge_mix", {})

        source_rows.extend(
            [
                {
                    "area": "stock",
                    "source": stock.get("table", "offline_feed.raw_material_stock"),
                    "timestamp/window": stock.get("time_range", ""),
                    "rows": stock.get("returned_rows", 0),
                    "note": f"{stock.get('fallback_count', 0)} fallback material(s)",
                },
                {
                    "area": "rm_chemistry",
                    "source": "offline chemistry tables",
                    "timestamp/window": f"{chemistry.get('start_time', '')} -> {chemistry.get('end_time', '')}",
                    "rows": sum(int(row.get("returned_rows", 0) or 0) for row in chemistry.get("tables", [])),
                    "note": f"{chemistry.get('fallback_count', 0)} fallback material(s), mode={chemistry.get('mode', '')}",
                },
                {
                    "area": "hm_slag",
                    "source": hm.get("source", ""),
                    "timestamp/window": hm.get("sample_timestamp") or f"{hm.get('start_time', '')} -> {hm.get('end_time', '')}",
                    "rows": hm.get("n_rows_used", 0),
                    "note": f"HM Fe basis={hm.get('hm_fe_pct_for_target', 0.0):.2f}%",
                },
                {
                    "area": "dpr_observed_slag",
                    "source": dpr.get("source", "offline_db"),
                    "timestamp/window": f"{dpr.get('start_time', '')} -> {dpr.get('end_time', '')}",
                    "rows": dpr.get("row_count", 0),
                    "note": f"slag={dpr.get('slag_generation_mt', 0.0):.1f} MT, HM={dpr.get('total_hot_metal_mt', 0.0):.1f} MT",
                },
                {
                    "area": "static_ml_dataset",
                    "source": history.get("path", ""),
                    "timestamp/window": history.get("latest_timestamp", ""),
                    "rows": history.get("rows", 0),
                    "note": f"{history.get('columns', 0)} columns",
                },
                {
                    "area": "process_charge_burden_context",
                    "source": process.get("source", "static_csv"),
                    "timestamp/window": process.get("latest_timestamp", ""),
                    "rows": process.get("field_count", 0),
                    "note": "latest cleaned static dataset row",
                },
                {
                    "area": "flux_inputs",
                    "source": flux.get("source", "offline_feed.charge_data+offline_feed.flux_chemistry"),
                    "timestamp/window": flux.get("latest_timestamp", ""),
                    "rows": len(flux.get("rows", [])),
                    "note": f"mode={flux.get('mode', '')}",
                },
                {
                    "area": "charge_mix",
                    "source": charge_mix.get("source", "offline_feed.charge_data"),
                    "timestamp/window": f"{charge_mix.get('start_time', '')} -> {charge_mix.get('end_time', '')}",
                    "rows": len(charge_mix.get("rows", [])),
                    "note": "sinter/ore/pellet/coke/nut coke/flux",
                },
            ]
        )

        tab_summary, tab_materials, tab_charge, tab_static = st.tabs(
            ["Source Summary", "Material Inputs", "Charge Mix", "Initial Feature Row"]
        )
        with tab_summary:
            _show_df(pd.DataFrame(source_rows))
            if warnings:
                st.warning("- " + "\n- ".join(str(w) for w in warnings[:12]))

        with tab_materials:
            st.markdown("##### Stock")
            _show_df(pd.DataFrame(stock.get("material_rows", [])))
            st.markdown("##### RM Chemistry")
            _show_df(pd.DataFrame(chemistry.get("material_rows", [])))
            st.markdown("##### Flux Inputs")
            _show_df(pd.DataFrame(flux.get("rows", [])))
            st.markdown("##### Current Ore Editor Inputs")
            _show_df(edited_ore_df.copy())

        with tab_charge:
            _show_df(pd.DataFrame(charge_mix.get("rows", [])))

        with tab_static:
            st.caption(
                "This is the latest numeric model context used before candidate "
                "blend overrides. Recent charge quantities come directly from "
                "offline_feed.charge_data when lagged model context is required."
            )
            rows = [
                {
                    "group": _context_group(str(field)),
                    "field": str(field),
                    "value": value,
                    "row_timestamp": process.get("latest_timestamp", ""),
                }
                for field, value in sorted(process_context.items())
            ]
            _show_df(pd.DataFrame(rows))


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

    model_service = _get_model_service()
    max_lag_hours = model_service.get_max_lag_steps()
    history_df, history_warnings = provider.get_history_frame(
        online_lag_hours=max_lag_hours
    )
    process_context, process_warnings = provider.get_process_context(
        history_df=history_df
    )
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
        edited_chemistry = replace(
            base.chemistry,
            moisture_pct=_float_from_row(
                row, "moisture_pct", base.chemistry.moisture_pct
            ),
            fe_t_pct=_float_from_row(row, "fe_t_pct", base.chemistry.fe_t_pct),
            sio2_pct=_float_from_row(row, "sio2_pct", base.chemistry.sio2_pct),
            al2o3_pct=_float_from_row(row, "al2o3_pct", base.chemistry.al2o3_pct),
            cao_pct=_float_from_row(row, "cao_pct", base.chemistry.cao_pct),
            mgo_pct=_float_from_row(row, "mgo_pct", base.chemistry.mgo_pct),
            mno_pct=_float_from_row(row, "mno_pct", base.chemistry.mno_pct),
            tio2_pct=_float_from_row(row, "tio2_pct", base.chemistry.tio2_pct),
        )
        selected_ores.append(
            replace(
                base,
                stock_mt=float(row["stock_mt"]),
                price_rs_per_mt=float(row["price_rs_per_mt"]),
                min_share_pct=float(row["min_share_pct"]),
                max_share_pct=float(row["max_share_pct"]),
                chemistry=edited_chemistry,
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
force_static_refresh = _render_static_dataset_bar(bmo_cfg)
pending_run_after_refresh = st.session_state.pop("bmo_pending_run_after_refresh", None)

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
    if chemistry_mode == "avg":
        chemistry_window_days = st.slider(
            "Chemistry window (days)",
            min_value=1,
            max_value=180,
            value=int(bmo_cfg.get("chemistry_window_days", 30)),
        )
    else:
        chemistry_window_days = int(bmo_cfg.get("chemistry_window_days", 30))
        st.caption("Latest uses the last charged instance for each material.")

target_production_mt = layout_col3.number_input(
    "Target HM / Pig Iron (MT)",
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
hm_fe_pct_for_target = float(
    hm_snapshot.get("hm_fe_pct_for_target")
    or target_cfg.get("hm_fe_pct_fallback", 94.5)
    or 94.5
)
if hm_fe_pct_for_target <= 0:
    hm_fe_pct_for_target = 94.5
    ore_diagnostics["warnings"].append(
        "HM Fe% unavailable; using 94.5% fallback to convert target HM into required Fe."
    )
target_fe_mt = float(target_production_mt) * hm_fe_pct_for_target / 100.0
st.caption(
    f"Optimiser Fe requirement: {target_fe_mt:,.1f} MT "
    f"from {target_production_mt:,.1f} MT HM at {hm_fe_pct_for_target:.2f}% Fe."
)

default_selected_names = set(ui_cfg.get("default_selected_ores", []))
default_selected_ids = [
    ore.ore_id for ore in ores if ore.display_name in default_selected_names
]
if bool(ui_cfg.get("auto_select_active_pellet", True)):
    active_pellet_ids, pellet_usage_warnings = provider.get_recent_active_pellet_ids()
    ore_diagnostics["warnings"].extend(pellet_usage_warnings)
    default_selected_ids = sorted(set(default_selected_ids).union(active_pellet_ids))
editor_df = build_ore_editor_df(ores, default_selected_ids=default_selected_ids)

st.markdown("### Ore Selection, Stock, Pricing, Chemistry, and Share Bounds")
edited_df = render_ore_editor(editor_df)

with st.expander("Slag, Fuel, Flux, and HM Assumptions", expanded=False):
    fuel_ash_df = build_fuel_ash_editor_df(bmo_cfg.get("fuel_ash_inputs", []))
    if not fuel_ash_df.empty:
        st.markdown("##### Fuel Ash Inputs")
        edited_fuel_ash_df = render_fuel_ash_editor(fuel_ash_df)
    else:
        edited_fuel_ash_df = fuel_ash_df
    fuel_ash_inputs = _fuel_ash_inputs_from_editor(edited_fuel_ash_df)

    flux_inputs, flux_warnings = provider.get_flux_inputs(
        mode=chemistry_mode, window_days=chemistry_window_days
    )
    ore_diagnostics["warnings"].extend(flux_warnings)
    st.caption(
        "Flux quantities and chemistry are loaded from charge data and flux chemistry records."
    )

    hm_chem_values = render_hot_metal_chemistry(
        hm_snapshot, bmo_cfg.get("slag_balance", {})
    )

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

# Operator-visible warning: if dust is entered but the full slag balance
# is disabled, the dust rows are silently ignored downstream. Surface
# this so the operator knows their dust entry isn't being applied.
_dust_entered_mt = sum(
    float(d.wet_qty_mt or 0.0) for d in dust_inputs if d.enabled
)
if _dust_entered_mt > 0 and not slag_balance_settings.enabled:
    st.warning(
        f"BF gas dust ({_dust_entered_mt:,.1f} MT) is entered but "
        "'Use full slag balance' is unchecked - dust will NOT be deducted "
        "from the slag balance."
    )

selected_ores = _selected_ores_from_editor(edited_df, ores)
pellet_input_issues = validate_selected_pellet_inputs(
    selected_ores,
    max_chemistry_age_days=int(
        bmo_cfg.get("data_sources", {}).get("pellet_chemistry_max_age_days", 30)
    ),
)
pellet_input_confirmed = True
if pellet_input_issues:
    st.warning(
        "Selected pellet inputs need operator review before optimisation:\n- "
        + "\n- ".join(pellet_input_issues)
    )
    pellet_input_confirmed = st.checkbox(
        "I have reviewed/edited the selected pellet stock and chemistry values.",
        value=False,
        key="bmo_confirm_pellet_inputs",
    )
visible_data_warnings = [
    str(w)
    for w in ore_diagnostics.get("warnings", [])
    if any(token in str(w).lower() for token in ("fallback", "failed", "unavailable"))
]
if visible_data_warnings:
    st.warning(
        "BMO is using fallback or incomplete source data:\n- "
        + "\n- ".join(visible_data_warnings[:5])
    )

_render_data_diagnostics(
    provider,
    ore_diagnostics=ore_diagnostics,
    hm_snapshot=hm_snapshot,
    edited_ore_df=edited_df,
    expanded=bool(visible_data_warnings),
)

run_lp_clicked = False
run_total_clicked = False
with st.form("bmo_run_form", clear_on_submit=False):
    run_col1, run_col2 = st.columns(2)
    run_lp_clicked = _form_submit_button(
        run_col1,
        "Run LP Baseline",
        type="secondary",
        width="stretch",
    )
    run_total_clicked = _form_submit_button(
        run_col2,
        "Run Total Cost Optimizer",
        type="primary",
        width="stretch",
    )

requested_lp = bool(run_lp_clicked or pending_run_after_refresh in {"lp", "both"})
requested_total = bool(
    run_total_clicked or pending_run_after_refresh in {"total", "both"}
)

if run_lp_clicked or run_total_clicked:
    with st.spinner("Checking static ML dataset freshness..."):
        refresh_result = _refresh_static_dataset_if_needed(
            bmo_cfg, force=force_static_refresh
        )
    if refresh_result.get("error"):
        if refresh_result.get("usable"):
            st.warning(
                "Static dataset refresh failed; continuing with the last usable local dataset. "
                f"Error: {refresh_result['error']}"
            )
        else:
            st.error(
                "Static dataset refresh failed and no usable local dataset exists. "
                f"Error: {refresh_result['error']}"
            )
            st.stop()
    elif refresh_result.get("refreshed"):
        st.session_state["bmo_pending_run_after_refresh"] = (
            "both"
            if run_lp_clicked and run_total_clicked
            else "lp"
            if run_lp_clicked
            else "total"
        )
        st.success("Static ML dataset refreshed. Re-running optimizer with the updated dataset.")
        st.rerun()

if requested_lp or requested_total:
    if pellet_input_issues and not pellet_input_confirmed:
        st.error(
            "Confirm the selected pellet stock and chemistry values before running BMO."
        )
    elif len(selected_ores) < 2:
        st.error("Select at least two ores before running optimization.")
    else:
        feo_in_slag_pct = float(
            bmo_cfg.get("chemistry", {}).get("feo_in_slag_pct", 0.4)
        )
        fuel_context = None

        with st.spinner("Running LP baseline..."):
            lp_result, lp_errors = run_lp_baseline(
                selected_ores,
                target_production_mt=target_fe_mt,
                target_slag_qty_mt=target_slag_qty_mt,
                feo_in_slag_pct=feo_in_slag_pct,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                hot_metal_target_mt=target_production_mt,
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
                    hot_metal_target_mt=target_production_mt,
                )
                lp_result.feasible = lp_physical_result.feasible
                lp_result.violations = lp_physical_result.violations
        if requested_lp:
            st.session_state["bmo_lp_result"] = lp_result
            st.session_state["bmo_lp_errors"] = lp_errors

        if requested_total:
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
                target_production_mt=target_fe_mt,
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
                hot_metal_target_mt=target_production_mt,
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
                is_lp_mode=True,
            )
            render_blend_table(lp_result, selected_ores)
            _render_share_pie(lp_result, selected_ores, "LP Share (%)")
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
            _render_share_pie(de_result, selected_ores, "DE Share (%)")
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
        comparison_result = de_result or lp_result
        if comparison_result is not None:
            comparison_title = (
                "DE Total-Cost Blend" if de_result is not None else "LP Baseline Blend"
            )
            _render_manual_blend_comparison(
                provider,
                comparison_result,
                selected_ores,
                comparison_title,
                target_fe_mt=target_fe_mt,
                target_production_mt=target_production_mt,
                feo_in_slag_pct=feo_in_slag_pct,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
            )

        if lp_result is not None and de_result is not None:
            # LP and DE solve DIFFERENT problems: LP minimises ore cost only,
            # DE jointly minimises ore + fuel. The headline summary makes
            # that asymmetry explicit so an operator can read what each
            # method optimised + see DE's fuel-cost value-add over LP.
            st.markdown("##### LP vs DE comparison")
            lp_ore = lp_result.ore_cost_per_thm_rs
            lp_fuel = lp_result.fuel_cost_per_thm_rs
            lp_total = lp_result.objective_rs_per_thm
            de_ore = de_result.ore_cost_per_thm_rs
            de_fuel = de_result.fuel_cost_per_thm_rs
            de_total = de_result.objective_rs_per_thm

            fuel_savings = lp_fuel - de_fuel   # +ve = DE saved fuel
            ore_premium = de_ore - lp_ore      # +ve = DE paid an ore premium
            total_delta = de_total - lp_total  # should be <= 0 if DE found improvement

            ca, cb, cc = st.columns(3)
            ca.metric(
                "LP ore-cost optimum (Rs/THM)",
                f"{lp_ore:,.2f}",
                help=(
                    "Lowest ore cost found by the linear programme. Fuel "
                    "cost shown on the LP tab is a post-hoc estimate, not "
                    "part of LP's objective."
                ),
            )
            cb.metric(
                "DE total-cost optimum (Rs/THM)",
                f"{de_total:,.2f}",
                delta=f"ore {de_ore:,.0f} + fuel {de_fuel:,.0f}",
                delta_color="off",
                help=(
                    "DE jointly minimises ore + fuel cost; the total above "
                    "is the actual optimisation objective."
                ),
            )
            cc.metric(
                "DE fuel savings vs LP (Rs/THM)",
                f"{fuel_savings:+,.2f}",
                delta=f"ore premium {ore_premium:+,.2f}",
                delta_color="off",
                help=(
                    "Positive = DE traded a higher ore cost for lower fuel "
                    "cost. Negative = DE could not improve on LP's fuel."
                ),
            )

            if total_delta > 0.5:
                st.warning(
                    f"DE total cost ({de_total:,.2f}) is higher than LP total "
                    f"({lp_total:,.2f}) by {total_delta:,.2f} Rs/THM. This "
                    "usually means DE hit the iteration/time budget before "
                    "improving on the LP seed -- increase maxiter or popsize "
                    "in setting_bmo.yml."
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
            st.dataframe(comparison_df, **df_kwargs)
        elif comparison_result is None:
            st.info(
                "Run LP or DE to compare the suggested blend with the last manual shift."
            )

render_diagnostics(de_result or lp_result, ore_diagnostics)
