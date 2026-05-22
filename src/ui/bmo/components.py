from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from domain.bmo.types import BlendEvaluation, OreInput


def apply_bmo_styles() -> None:
    css_path = Path(__file__).resolve().parents[2] / "assets" / "css" / "bmo_style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def _safe_dataframe(
    df: pd.DataFrame, *, hide_index: bool = True, use_container_width: bool = True
) -> None:
    sig = inspect.signature(st.dataframe)
    kwargs: dict[str, Any] = {}
    if "hide_index" in sig.parameters:
        kwargs["hide_index"] = hide_index
    if "use_container_width" in sig.parameters:
        kwargs["use_container_width"] = use_container_width
    st.dataframe(df, **kwargs)


def render_header(bundle_status: dict[str, Any]) -> None:
    model_status = "Loaded" if bundle_status.get("model_loaded") else "Fallback"
    scaler_status = "Loaded" if bundle_status.get("scaler_loaded") else "Missing"
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
          &nbsp;|&nbsp; Expected features: {bundle_status.get("feature_count", 0)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bundle_status.get("bundle_error"):
        st.warning(f"Model bundle warning: {bundle_status['bundle_error']}")


def build_ore_editor_df(
    ores: list[OreInput], default_selected_ids: list[str]
) -> pd.DataFrame:
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
                "min_share_pct": float(ore.min_share_pct),
                "max_share_pct": float(ore.max_share_pct),
            }
        )
    return pd.DataFrame(rows)


def render_ore_editor(editor_df: pd.DataFrame) -> pd.DataFrame:
    editor_fn = getattr(st, "data_editor", None) or getattr(
        st, "experimental_data_editor", None
    )
    if editor_fn is None:
        st.info(
            "Editable table is not available in this Streamlit version. "
            "Using default ore mapping values for this run."
        )
        _safe_dataframe(editor_df, hide_index=True, use_container_width=True)
        return editor_df

    return editor_fn(
        editor_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "selected": st.column_config.CheckboxColumn("Use", default=True),
            "ore_id": st.column_config.TextColumn("Ore ID", disabled=True),
            "ore_name": st.column_config.TextColumn("Ore / Source", disabled=True),
            "stock_mt": st.column_config.NumberColumn(
                "Stock (MT)", min_value=0.0, step=10.0
            ),
            "price_rs_per_mt": st.column_config.NumberColumn(
                "Price (Rs/MT)", min_value=0.0, step=10.0
            ),
            "min_share_pct": st.column_config.NumberColumn(
                "Min Share (%)", min_value=0.0, max_value=100.0, step=0.5
            ),
            "max_share_pct": st.column_config.NumberColumn(
                "Max Share (%)", min_value=0.0, max_value=100.0, step=0.5
            ),
        },
    )


def render_blend_metrics(title: str, blend: BlendEvaluation) -> None:
    st.markdown(f"#### {title}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objective (Rs/THM)", f"{blend.objective_rs_per_thm:,.2f}")
    c2.metric("Ore Cost (Rs/THM)", f"{blend.ore_cost_per_thm_rs:,.2f}")
    c3.metric("Fuel Cost (Rs/THM)", f"{blend.fuel_cost_per_thm_rs:,.2f}")
    c4.metric("Fe Production (MT)", f"{blend.fe_production_mt:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Qty (MT)", f"{blend.total_qty_mt:,.2f}")
    c6.metric("Slag (MT)", f"{blend.slag_mt:,.2f}")
    c7.metric("Effective Fe (%)", f"{blend.effective_fe_pct:,.3f}")
    c8.metric("Fe(T) (%)", f"{blend.fe_t_pct:,.3f}")

    if blend.violations:
        st.warning("Constraint violations:\n- " + "\n- ".join(blend.violations))


def render_blend_table(blend: BlendEvaluation, selected_ores: list[OreInput]) -> None:
    rows = []
    for ore in selected_ores:
        qty = float(blend.quantities_mt.get(ore.ore_id, 0.0))
        share = float(blend.shares_pct.get(ore.ore_id, 0.0))
        rows.append(
            {
                "ore_name": ore.display_name,
                "quantity_mt": qty,
                "share_pct": share,
                "stock_mt": ore.stock_mt,
                "price_rs_per_mt": ore.price_rs_per_mt,
                "ore_cost_rs": qty * ore.price_rs_per_mt,
            }
        )
    df = pd.DataFrame(rows).sort_values("quantity_mt", ascending=False)
    if hasattr(st, "column_config"):
        st.dataframe(
            df,
            hide_index=(
                True
                if "hide_index" in inspect.signature(st.dataframe).parameters
                else False
            ),
            use_container_width=(
                True
                if "use_container_width" in inspect.signature(st.dataframe).parameters
                else False
            ),
            column_config={
                "ore_name": st.column_config.TextColumn("Ore"),
                "quantity_mt": st.column_config.NumberColumn("Qty (MT)", format="%.1f"),
                "share_pct": st.column_config.NumberColumn("Share (%)", format="%.2f"),
                "stock_mt": st.column_config.NumberColumn("Stock (MT)", format="%.1f"),
                "price_rs_per_mt": st.column_config.NumberColumn(
                    "Price (Rs/MT)", format="%.1f"
                ),
                "ore_cost_rs": st.column_config.NumberColumn(
                    "Ore Cost (Rs)", format="%.0f"
                ),
            },
        )
    else:
        _safe_dataframe(df, hide_index=True, use_container_width=True)


def render_diagnostics(
    blend: BlendEvaluation | None, diagnostics: dict[str, Any]
) -> None:
    with st.expander("Diagnostics", expanded=False):
        if diagnostics.get("warnings"):
            st.write("Data warnings:")
            for warning in diagnostics["warnings"]:
                st.write(f"- {warning}")
        if blend is not None and blend.diagnostics:
            st.write("Solver diagnostics:")
            st.json(blend.diagnostics)
