"""Reusable Streamlit components for the BMO page.

This module keeps BMO page rendering helpers separate from workflow orchestration.
It renders status, ore editing, result metrics, result tables, and diagnostics
while handling Streamlit API differences across installed versions.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from utils.bmo.types import BlendEvaluation, OreInput


def apply_bmo_styles() -> None:
    """
    Inject BMO-specific CSS into the Streamlit page.

    The BMO result area uses compact KPI cards and wide tables. Keeping the CSS
    in this component module lets the page renderer stay focused on data flow
    while the visual treatment remains reusable for LP and DE tabs.

    Args:
         - None

    Returns:
         - return None - Writes CSS to the current Streamlit page when available.
    """

    css_path = Path(__file__).resolve().parents[2] / "assets" / "css" / "bmo_style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def _safe_dataframe(
    df: pd.DataFrame, *, hide_index: bool = True, use_container_width: bool = True
) -> None:
    """
    Render a Streamlit dataframe with version-compatible width parameters.

    Streamlit renamed ``use_container_width`` to ``width`` in newer versions.
    This wrapper inspects the available API so BMO tables render cleanly across
    deployed and local Streamlit versions.

    Args:
         - df: pd.DataFrame - Data frame to render.
         - hide_index: bool - Whether to hide the dataframe index.
         - use_container_width: bool - Whether the table should stretch to container width.

    Returns:
         - return None - Renders the dataframe in Streamlit.
    """

    sig = inspect.signature(st.dataframe)
    kwargs: dict[str, Any] = {}
    if "hide_index" in sig.parameters:
        kwargs["hide_index"] = hide_index
    if "width" in sig.parameters:
        kwargs["width"] = "stretch" if use_container_width else "content"
    elif "use_container_width" in sig.parameters:
        kwargs["use_container_width"] = use_container_width
    st.dataframe(df, **kwargs)


def render_header(bundle_status: dict[str, Any]) -> None:
    """
    Render the BMO page header and model bundle status.

    The header gives operators a quick signal about the fuel model state before
    they run LP or DE. It also surfaces bundle warnings when artifact versions
    or paths prevent safe model inference.

    Args:
         - bundle_status: dict[str, Any] - Model/scaler status from model service.

    Returns:
         - return None - Writes header and warnings to the Streamlit page.
    """

    if bundle_status:
        model_status = "Loaded" if bundle_status.get("model_loaded") else "Fallback"
        scaler_status = "Loaded" if bundle_status.get("scaler_loaded") else "Missing"
        feature_count = bundle_status.get("feature_count", 0)
    else:
        model_status = "Not loaded"
        scaler_status = "Not loaded"
        feature_count = 0
    status_class = (
        "bmo-status-ok" if bundle_status.get("model_loaded") else "bmo-status-warn"
    )

    st.markdown(
        f"""
        <div class="bmo-header">
          <h2>Blend Mix Optimizer (BMO)</h2>
          <p>LP baseline + nonlinear total-cost optimization for ore blend planning.</p>
        </div>
        <div class="bmo-subtle">
          Model: <span class="{status_class}">{model_status}</span>
          &nbsp;|&nbsp; Scaler: {scaler_status}
          &nbsp;|&nbsp; Expected features: {feature_count}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bundle_status.get("bundle_error"):
        st.warning(f"Model bundle warning: {bundle_status['bundle_error']}")


def build_ore_editor_df(
    ores: list[OreInput], default_selected_ids: list[str]
) -> pd.DataFrame:
    """
    Build the editable ore selection table from typed ore inputs.

    The table includes moisture because BMO final Fe% is dry-weight weighted.
    Users can override moisture for a run without changing the underlying YAML
    mapping or live chemistry source.

    Args:
         - ores: list[OreInput] - Available BMO ores.
         - default_selected_ids: list[str] - Ore ids selected by default.

    Returns:
         - return pd.DataFrame - Editable table data for Streamlit.
    """

    default_set = set(default_selected_ids or [])
    rows = []
    for ore in ores:
        rows.append(
            {
                "selected": ore.ore_id in default_set,
                "ore_id": ore.ore_id,
                "ore_name": ore.display_name,
                "stock_mt": float(ore.stock_mt),
                "price_rs_per_mt": float(ore.price_rs_per_mt),
                "moisture_pct": float(ore.chemistry.moisture_pct),
                "min_share_pct": float(ore.min_share_pct),
                "max_share_pct": float(ore.max_share_pct),
            }
        )
    return pd.DataFrame(rows)


def render_ore_editor(editor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Render the editable ore table and return the user's edited rows.

    The editor is the operator's last chance to adjust stock, prices, share
    limits, and moisture before optimization. Returning a dataframe keeps the
    Streamlit page logic separate from the rendering details.

    Args:
         - editor_df: pd.DataFrame - Ore selection and bound table.

    Returns:
         - return pd.DataFrame - Edited ore table, or original table as fallback.
    """

    editor_fn = getattr(st, "data_editor", None) or getattr(
        st, "experimental_data_editor", None
    )
    if editor_fn is None:
        st.info(
            "Editable table is not available in this Streamlit version. "
            "Using default ore mapping values for this run."
        )
        _safe_dataframe(
            editor_df.drop(columns=["ore_id"], errors="ignore"),
            hide_index=True,
            use_container_width=True,
        )
        return editor_df

    sig = inspect.signature(editor_fn)
    visible_columns = (
        "selected",
        "ore_name",
        "stock_mt",
        "price_rs_per_mt",
        "moisture_pct",
        "min_share_pct",
        "max_share_pct",
    )
    editor_kwargs: dict[str, Any] = {
        "hide_index": True,
        "column_config": {
            "selected": st.column_config.CheckboxColumn("Use", default=True),
            "ore_id": st.column_config.TextColumn("Ore ID", disabled=True),
            "ore_name": st.column_config.TextColumn("Ore / Source", disabled=True),
            "stock_mt": st.column_config.NumberColumn(
                "Stock (MT)", min_value=0.0, step=10.0
            ),
            "price_rs_per_mt": st.column_config.NumberColumn(
                "Price (Rs/MT)", min_value=0.0, step=10.0
            ),
            "moisture_pct": st.column_config.NumberColumn(
                "Moisture (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "min_share_pct": st.column_config.NumberColumn(
                "Min Share (%)", min_value=0.0, max_value=100.0, step=0.5
            ),
            "max_share_pct": st.column_config.NumberColumn(
                "Max Share (%)", min_value=0.0, max_value=100.0, step=0.5
            ),
        },
    }
    if "column_order" in sig.parameters:
        editor_kwargs["column_order"] = visible_columns
    if "width" in sig.parameters:
        editor_kwargs["width"] = "stretch"
    elif "use_container_width" in sig.parameters:
        editor_kwargs["use_container_width"] = True
    return editor_fn(editor_df, **editor_kwargs)


def render_blend_metrics(title: str, blend: BlendEvaluation) -> None:
    """
    Render summary metrics and constraint warnings for a blend result.

    The metric rows separate wet quantity, dry quantity, slag MT, and slag rate
    so users can see how moisture and slag-forming oxides affect the result.

    Args:
         - title: str - Section title shown above the metrics.
         - blend: BlendEvaluation - Evaluated blend result to display.

    Returns:
         - return None - Writes metrics and warnings to the Streamlit page.
    """

    st.markdown(f"#### {title}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objective (Rs/THM)", f"{blend.objective_rs_per_thm:,.2f}")
    c2.metric("Ore Cost (Rs/THM)", f"{blend.ore_cost_per_thm_rs:,.2f}")
    c3.metric("Fuel Cost (Rs/THM)", f"{blend.fuel_cost_per_thm_rs:,.2f}")
    c4.metric("Fe Production (MT)", f"{blend.fe_production_mt:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    dry_qty = float(blend.diagnostics.get("total_dry_qty_mt", 0.0) or 0.0)
    c5.metric("Wet Qty (MT)", f"{blend.total_qty_mt:,.2f}")
    c6.metric("Dry Qty (MT)", f"{dry_qty:,.2f}")
    c7.metric("Slag (MT)", f"{blend.slag_mt:,.2f}")
    c8.metric("Final Fe (%)", f"{blend.fe_t_pct:,.3f}")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Slag Rate (kg/THM)", f"{blend.slag_rate_kg_per_thm:,.2f}")
    c10.empty()
    c11.empty()
    c12.empty()

    if blend.violations:
        st.warning("Constraint violations:\n- " + "\n- ".join(blend.violations))


def render_blend_table(blend: BlendEvaluation, selected_ores: list[OreInput]) -> None:
    """
    Render the ore quantity, share, stock, price, and cost table for a blend.

    Per-ore dry quantity, Fe contribution, and slag contribution are read from
    blend diagnostics. This keeps the table traceable to the dry-weight Fe and
    oxide-sum slag formulas used by the optimizer and top-level metrics.

    Args:
         - blend: BlendEvaluation - Evaluated blend result to display.
         - selected_ores: list[OreInput] - Ores included in the current run.

    Returns:
         - return None - Writes the blend table to the Streamlit page.
    """

    rows = []
    dry_weight_by_ore = blend.diagnostics.get("dry_weight_mt_by_ore", {}) or {}
    fe_contribution_by_ore = (
        blend.diagnostics.get("fe_contribution_mt_by_ore", {}) or {}
    )
    slag_contribution_by_ore = (
        blend.diagnostics.get("slag_contribution_mt_by_ore", {}) or {}
    )
    for ore in selected_ores:
        qty = float(blend.quantities_mt.get(ore.ore_id, 0.0))
        share = float(blend.shares_pct.get(ore.ore_id, 0.0))
        rows.append(
            {
                "ore_name": ore.display_name,
                "quantity_mt": qty,
                "dry_quantity_mt": float(dry_weight_by_ore.get(ore.ore_id, 0.0)),
                "moisture_pct": float(ore.chemistry.moisture_pct),
                "fe_contribution_mt": float(
                    fe_contribution_by_ore.get(ore.ore_id, 0.0)
                ),
                "slag_contribution_mt": float(
                    slag_contribution_by_ore.get(ore.ore_id, 0.0)
                ),
                "share_pct": share,
                "stock_mt": ore.stock_mt,
                "price_rs_per_mt": ore.price_rs_per_mt,
                "ore_cost_rs": qty * ore.price_rs_per_mt,
            }
        )
    df = pd.DataFrame(rows).sort_values("quantity_mt", ascending=False)
    if hasattr(st, "column_config"):
        sig = inspect.signature(st.dataframe)
        df_kwargs: dict[str, Any] = {
            "column_config": {
                "ore_name": st.column_config.TextColumn("Ore"),
                "quantity_mt": st.column_config.NumberColumn(
                    "Wet Qty (MT)", format="%.1f"
                ),
                "dry_quantity_mt": st.column_config.NumberColumn(
                    "Dry Qty (MT)", format="%.1f"
                ),
                "moisture_pct": st.column_config.NumberColumn(
                    "Moisture (%)", format="%.2f"
                ),
                "fe_contribution_mt": st.column_config.NumberColumn(
                    "Fe (MT)", format="%.2f"
                ),
                "slag_contribution_mt": st.column_config.NumberColumn(
                    "Slag (MT)", format="%.2f"
                ),
                "share_pct": st.column_config.NumberColumn("Share (%)", format="%.2f"),
                "stock_mt": st.column_config.NumberColumn("Stock (MT)", format="%.1f"),
                "price_rs_per_mt": st.column_config.NumberColumn(
                    "Price (Rs/MT)", format="%.1f"
                ),
                "ore_cost_rs": st.column_config.NumberColumn(
                    "Ore Cost (Rs)", format="%.0f"
                ),
            },
        }
        if "hide_index" in sig.parameters:
            df_kwargs["hide_index"] = True
        if "width" in sig.parameters:
            df_kwargs["width"] = "stretch"
        elif "use_container_width" in sig.parameters:
            df_kwargs["use_container_width"] = True
        st.dataframe(df, **df_kwargs)
    else:
        _safe_dataframe(df, hide_index=True, use_container_width=True)


def render_diagnostics(
    blend: BlendEvaluation | None, diagnostics: dict[str, Any]
) -> None:
    """
    Render optional BMO data-source and solver diagnostics.

    Diagnostics are intentionally tucked behind an expander so the main page can
    stay operationally clean while still exposing data-source warnings, solver
    metadata, and model feature details during troubleshooting.

    Args:
         - blend: BlendEvaluation | None - Latest blend result, if one exists.
         - diagnostics: dict[str, Any] - Data-source diagnostics from context provider.

    Returns:
         - return None - Writes diagnostics inside a Streamlit expander.
    """

    with st.expander("Diagnostics", expanded=False):
        if diagnostics.get("warnings"):
            st.write("Data warnings:")
            for warning in diagnostics["warnings"]:
                st.write(f"- {warning}")
        if blend is not None and blend.diagnostics:
            st.write("Solver diagnostics:")
            st.json(blend.diagnostics)
