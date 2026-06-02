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


def build_fuel_ash_editor_df(fuel_ash_cfg: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Build the editable fuel ash chemistry table from BMO configuration.

    Fuel ash defaults come from the laboratory ash-analysis workbook and can be
    overridden by operators for each run. The table keeps rate, moisture, ash,
    ash oxide chemistry, and dry-fuel-basis S/P together so the slag calculation
    can apply the full fuel ash sequence.

    Args:
         - fuel_ash_cfg: list[dict[str, Any]] - Configured fuel ash defaults.

    Returns:
         - return pd.DataFrame - Editable fuel ash data for Streamlit.
    """

    rows = []
    for item in fuel_ash_cfg or []:
        rows.append(
            {
                "enabled": bool(item.get("enabled", True)),
                "fuel_id": str(item.get("fuel_id", "")),
                "fuel_name": str(item.get("display_name", item.get("fuel_id", ""))),
                "rate_kg_per_thm": float(item.get("rate_kg_per_thm", 0.0) or 0.0),
                "moisture_pct": float(item.get("moisture_pct", 0.0) or 0.0),
                "ash_pct": float(item.get("ash_pct", 0.0) or 0.0),
                "sio2_pct": float(item.get("sio2_pct", 0.0) or 0.0),
                "al2o3_pct": float(item.get("al2o3_pct", 0.0) or 0.0),
                "cao_pct": float(item.get("cao_pct", 0.0) or 0.0),
                "mgo_pct": float(item.get("mgo_pct", 0.0) or 0.0),
                "fe2o3_pct": float(item.get("fe2o3_pct", 0.0) or 0.0),
                "tio2_pct": float(item.get("tio2_pct", 0.0) or 0.0),
                "na2o_pct": float(item.get("na2o_pct", 0.0) or 0.0),
                "k2o_pct": float(item.get("k2o_pct", 0.0) or 0.0),
                "s_pct": float(item.get("s_pct", 0.0) or 0.0),
                "p_pct": float(item.get("p_pct", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def render_fuel_ash_editor(editor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Render the editable fuel ash table and return the user's edited values.

    The fuel table is intentionally separate from ore selection because fuel
    ash contributes to slag but does not participate in ore share constraints.
    Operators can disable a fuel row or adjust rate/ash chemistry before
    running LP or DE.

    Args:
         - editor_df: pd.DataFrame - Fuel ash defaults and editable values.

    Returns:
         - return pd.DataFrame - Edited fuel ash rows, or original table as fallback.
    """

    if editor_df.empty:
        return editor_df

    editor_fn = getattr(st, "data_editor", None) or getattr(
        st, "experimental_data_editor", None
    )
    if editor_fn is None:
        st.info(
            "Editable fuel ash table is not available in this Streamlit version. "
            "Using configured fuel ash defaults for this run."
        )
        _safe_dataframe(
            editor_df.drop(columns=["fuel_id"], errors="ignore"),
            hide_index=True,
            use_container_width=True,
        )
        return editor_df

    sig = inspect.signature(editor_fn)
    visible_columns = (
        "enabled",
        "fuel_name",
        "rate_kg_per_thm",
        "moisture_pct",
        "ash_pct",
        "sio2_pct",
        "al2o3_pct",
        "cao_pct",
        "mgo_pct",
        "fe2o3_pct",
        "tio2_pct",
        "na2o_pct",
        "k2o_pct",
        "s_pct",
        "p_pct",
    )
    editor_kwargs: dict[str, Any] = {
        "hide_index": True,
        "column_config": {
            "enabled": st.column_config.CheckboxColumn("Use", default=True),
            "fuel_id": st.column_config.TextColumn("Fuel ID", disabled=True),
            "fuel_name": st.column_config.TextColumn("Fuel", disabled=True),
            "rate_kg_per_thm": st.column_config.NumberColumn(
                "Rate (kg/THM)", min_value=0.0, step=1.0
            ),
            "moisture_pct": st.column_config.NumberColumn(
                "Moisture (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "ash_pct": st.column_config.NumberColumn(
                "Ash (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "sio2_pct": st.column_config.NumberColumn(
                "Ash SiO2 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "al2o3_pct": st.column_config.NumberColumn(
                "Ash Al2O3 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "cao_pct": st.column_config.NumberColumn(
                "Ash CaO (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "mgo_pct": st.column_config.NumberColumn(
                "Ash MgO (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "fe2o3_pct": st.column_config.NumberColumn(
                "Ash Fe2O3 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "tio2_pct": st.column_config.NumberColumn(
                "Ash TiO2 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "na2o_pct": st.column_config.NumberColumn(
                "Ash Na2O (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "k2o_pct": st.column_config.NumberColumn(
                "Ash K2O (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "s_pct": st.column_config.NumberColumn(
                "S in Fuel (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "p_pct": st.column_config.NumberColumn(
                "P in Fuel (%)", min_value=0.0, max_value=100.0, step=0.001
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


def build_flux_editor_df(flux_cfg: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Build the editable fixed-flux chemistry table from BMO configuration.

    Flux defaults come from plant flux chemistry references and can be adjusted
    by operators for each run. Flux rows are fixed burden additions, so they
    contribute to total slag but do not participate in ore share optimization.

    Args:
         - flux_cfg: list[dict[str, Any]] - Configured flux defaults.

    Returns:
         - return pd.DataFrame - Editable flux data for Streamlit.
    """

    rows = []
    for item in flux_cfg or []:
        rows.append(
            {
                "enabled": bool(item.get("enabled", True)),
                "flux_id": str(item.get("flux_id", "")),
                "flux_name": str(item.get("display_name", item.get("flux_id", ""))),
                "wet_qty_mt": float(item.get("wet_qty_mt", 0.0) or 0.0),
                "moisture_pct": float(item.get("moisture_pct", 0.0) or 0.0),
                "sio2_pct": float(item.get("sio2_pct", 0.0) or 0.0),
                "al2o3_pct": float(item.get("al2o3_pct", 0.0) or 0.0),
                "cao_pct": float(item.get("cao_pct", 0.0) or 0.0),
                "mgo_pct": float(item.get("mgo_pct", 0.0) or 0.0),
                "fe2o3_pct": float(item.get("fe2o3_pct", 0.0) or 0.0),
                "mno_pct": float(item.get("mno_pct", 0.0) or 0.0),
                "tio2_pct": float(item.get("tio2_pct", 0.0) or 0.0),
                "na2o_pct": float(item.get("na2o_pct", 0.0) or 0.0),
                "k2o_pct": float(item.get("k2o_pct", 0.0) or 0.0),
                "caf2_pct": float(item.get("caf2_pct", 0.0) or 0.0),
                "p_pct": float(item.get("p_pct", 0.0) or 0.0),
                "s_pct": float(item.get("s_pct", 0.0) or 0.0),
                "zn_pct": float(item.get("zn_pct", 0.0) or 0.0),
                "loi_pct": float(item.get("loi_pct", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def render_flux_editor(editor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Render the editable fixed-flux table and return the user's edited values.

    The flux table collects wet quantity, moisture, and core oxide chemistry.
    BMO uses those values to reserve slag capacity for fixed fluxes before the
    ore optimizer searches for a feasible blend.

    Args:
         - editor_df: pd.DataFrame - Flux defaults and editable values.

    Returns:
         - return pd.DataFrame - Edited flux rows, or original table as fallback.
    """

    if editor_df.empty:
        return editor_df

    editor_fn = getattr(st, "data_editor", None) or getattr(
        st, "experimental_data_editor", None
    )
    if editor_fn is None:
        st.info(
            "Editable flux table is not available in this Streamlit version. "
            "Using configured flux defaults for this run."
        )
        _safe_dataframe(
            editor_df.drop(columns=["flux_id"], errors="ignore"),
            hide_index=True,
            use_container_width=True,
        )
        return editor_df

    sig = inspect.signature(editor_fn)
    visible_columns = (
        "enabled",
        "flux_name",
        "wet_qty_mt",
        "moisture_pct",
        "sio2_pct",
        "al2o3_pct",
        "cao_pct",
        "mgo_pct",
        "fe2o3_pct",
        "mno_pct",
        "tio2_pct",
        "na2o_pct",
        "k2o_pct",
        "caf2_pct",
        "p_pct",
        "s_pct",
        "zn_pct",
        "loi_pct",
    )
    editor_kwargs: dict[str, Any] = {
        "hide_index": True,
        "column_config": {
            "enabled": st.column_config.CheckboxColumn("Use", default=True),
            "flux_id": st.column_config.TextColumn("Flux ID", disabled=True),
            "flux_name": st.column_config.TextColumn("Flux", disabled=True),
            "wet_qty_mt": st.column_config.NumberColumn(
                "Wet Qty (MT)", min_value=0.0, step=1.0
            ),
            "moisture_pct": st.column_config.NumberColumn(
                "Moisture/TM (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "sio2_pct": st.column_config.NumberColumn(
                "SiO2 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "al2o3_pct": st.column_config.NumberColumn(
                "Al2O3 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "cao_pct": st.column_config.NumberColumn(
                "CaO (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "mgo_pct": st.column_config.NumberColumn(
                "MgO (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "fe2o3_pct": st.column_config.NumberColumn(
                "Fe2O3 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "mno_pct": st.column_config.NumberColumn(
                "MnO (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "tio2_pct": st.column_config.NumberColumn(
                "TiO2 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "na2o_pct": st.column_config.NumberColumn(
                "Na2O (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "k2o_pct": st.column_config.NumberColumn(
                "K2O (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "caf2_pct": st.column_config.NumberColumn(
                "CaF2 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "p_pct": st.column_config.NumberColumn(
                "P (%)", min_value=0.0, max_value=100.0, step=0.001
            ),
            "s_pct": st.column_config.NumberColumn(
                "S (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "zn_pct": st.column_config.NumberColumn(
                "Zn (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "loi_pct": st.column_config.NumberColumn(
                "LOI (%)", min_value=0.0, max_value=100.0, step=0.1
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


def build_dust_editor_df(dust_cfg: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Build the editable BF gas dust deduction table from BMO configuration.

    Dust rows are optional full slag-balance inputs. The table keeps wet
    quantity, moisture, and component chemistry together so the calculator can
    deduct dry dust component masses after ore, flux, and fuel ash are added.

    Args:
         - dust_cfg: list[dict[str, Any]] - Configured dust defaults.

    Returns:
         - return pd.DataFrame - Editable dust data for Streamlit.
    """

    rows = []
    for item in dust_cfg or []:
        rows.append(
            {
                "enabled": bool(item.get("enabled", True)),
                "dust_id": str(item.get("dust_id", "")),
                "dust_name": str(item.get("display_name", item.get("dust_id", ""))),
                "wet_qty_mt": float(item.get("wet_qty_mt", 0.0) or 0.0),
                "moisture_pct": float(item.get("moisture_pct", 0.0) or 0.0),
                "sio2_pct": float(item.get("sio2_pct", 0.0) or 0.0),
                "al2o3_pct": float(item.get("al2o3_pct", 0.0) or 0.0),
                "cao_pct": float(item.get("cao_pct", 0.0) or 0.0),
                "mgo_pct": float(item.get("mgo_pct", 0.0) or 0.0),
                "fe_pct": float(item.get("fe_pct", 0.0) or 0.0),
                "mn_pct": float(item.get("mn_pct", 0.0) or 0.0),
                "p_pct": float(item.get("p_pct", 0.0) or 0.0),
                "s_pct": float(item.get("s_pct", 0.0) or 0.0),
                "ti_pct": float(item.get("ti_pct", 0.0) or 0.0),
                "zn_pct": float(item.get("zn_pct", 0.0) or 0.0),
                "na2o_pct": float(item.get("na2o_pct", 0.0) or 0.0),
                "k2o_pct": float(item.get("k2o_pct", 0.0) or 0.0),
                "caf2_pct": float(item.get("caf2_pct", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def render_dust_editor(editor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Render the editable BF gas dust table and return edited values.

    Dust is deducted only by the full slag-balance calculator. Keeping this as
    an editable table lets users use zero-dust defaults or enter plant dust
    chemistry when available.

    Args:
         - editor_df: pd.DataFrame - Dust defaults and editable values.

    Returns:
         - return pd.DataFrame - Edited dust rows, or original table as fallback.
    """

    if editor_df.empty:
        return editor_df

    editor_fn = getattr(st, "data_editor", None) or getattr(
        st, "experimental_data_editor", None
    )
    if editor_fn is None:
        _safe_dataframe(
            editor_df.drop(columns=["dust_id"], errors="ignore"),
            hide_index=True,
            use_container_width=True,
        )
        return editor_df

    sig = inspect.signature(editor_fn)
    visible_columns = (
        "enabled",
        "dust_name",
        "wet_qty_mt",
        "moisture_pct",
        "sio2_pct",
        "al2o3_pct",
        "cao_pct",
        "mgo_pct",
        "fe_pct",
        "mn_pct",
        "p_pct",
        "s_pct",
        "ti_pct",
        "zn_pct",
        "na2o_pct",
        "k2o_pct",
        "caf2_pct",
    )
    editor_kwargs: dict[str, Any] = {
        "hide_index": True,
        "column_config": {
            "enabled": st.column_config.CheckboxColumn("Use", default=True),
            "dust_id": st.column_config.TextColumn("Dust ID", disabled=True),
            "dust_name": st.column_config.TextColumn("Dust", disabled=True),
            "wet_qty_mt": st.column_config.NumberColumn(
                "Wet Qty (MT)", min_value=0.0, step=1.0
            ),
            "moisture_pct": st.column_config.NumberColumn(
                "Moisture (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "sio2_pct": st.column_config.NumberColumn(
                "SiO2 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "al2o3_pct": st.column_config.NumberColumn(
                "Al2O3 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "cao_pct": st.column_config.NumberColumn(
                "CaO (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "mgo_pct": st.column_config.NumberColumn(
                "MgO (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "fe_pct": st.column_config.NumberColumn(
                "Fe (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "mn_pct": st.column_config.NumberColumn(
                "Mn (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "p_pct": st.column_config.NumberColumn(
                "P (%)", min_value=0.0, max_value=100.0, step=0.001
            ),
            "s_pct": st.column_config.NumberColumn(
                "S (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "ti_pct": st.column_config.NumberColumn(
                "Ti (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "zn_pct": st.column_config.NumberColumn(
                "Zn (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "na2o_pct": st.column_config.NumberColumn(
                "Na2O (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "k2o_pct": st.column_config.NumberColumn(
                "K2O (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "caf2_pct": st.column_config.NumberColumn(
                "CaF2 (%)", min_value=0.0, max_value=100.0, step=0.1
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


def render_slag_balance_settings(
    settings_cfg: dict[str, Any],
) -> dict[str, float | bool]:
    """
    Render full slag-balance correction and plant assumptions.

    The UI uses one slag correction factor instead of exposing pig-iron
    chemistry assumptions. Operators can still tune recovery, gas loss, alkali
    reporting, and component conversion factors used after the BF component
    balance is calculated.

    Args:
         - settings_cfg: dict[str, Any] - Configured slag-balance defaults.

    Returns:
         - return dict[str, float | bool] - Edited slag-balance settings.
    """

    values: dict[str, float | bool] = {}
    values["enabled"] = st.checkbox(
        "Use full slag balance",
        value=bool(settings_cfg.get("enabled", True)),
    )

    c1, c2, c3, c4 = st.columns(4)
    values["slag_correction_factor"] = c1.number_input(
        "Slag Correction Factor",
        min_value=0.0,
        max_value=2.0,
        value=float(settings_cfg.get("slag_correction_factor", 0.95)),
        step=0.001,
    )
    values["pi_loss_pct"] = c2.number_input(
        "PI Loss (%)",
        min_value=0.0,
        max_value=99.0,
        value=float(settings_cfg.get("pi_loss_pct", 0.2)),
        step=0.01,
    )
    values["fe_to_pig_iron_fraction"] = c3.number_input(
        "Fe to PI Fraction",
        min_value=0.0,
        max_value=1.0,
        value=float(settings_cfg.get("fe_to_pig_iron_fraction", 0.999)),
        step=0.001,
    )
    values["mn_recovery_pct"] = c4.number_input(
        "Mn/Ti Recovery (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(settings_cfg.get("mn_recovery_pct", 60.0)),
        step=0.1,
    )

    c5, c6, c7, c8 = st.columns(4)
    values["sulphur_gas_loss_pct"] = c5.number_input(
        "S Gas Loss (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(settings_cfg.get("sulphur_gas_loss_pct", 10.0)),
        step=0.1,
    )
    values["alkali_to_slag_fraction"] = c6.number_input(
        "Alkali to Slag",
        min_value=0.0,
        max_value=1.0,
        value=float(settings_cfg.get("alkali_to_slag_fraction", 0.8)),
        step=0.01,
    )
    values["fe_to_feo_factor"] = c7.number_input(
        "Fe to FeO",
        min_value=0.0,
        value=float(settings_cfg.get("fe_to_feo_factor", 72.0 / 56.0)),
        step=0.001,
    )
    values["mn_to_mno_factor"] = c8.number_input(
        "Mn to MnO",
        min_value=0.0,
        value=float(settings_cfg.get("mn_to_mno_factor", 1.291)),
        step=0.001,
    )
    return values


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
    c7.metric("Final Fe (%)", f"{blend.fe_t_pct:,.3f}")
    c8.metric("Slag Rate (kg/THM)", f"{blend.slag_rate_kg_per_thm:,.2f}")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Slag (MT)", f"{blend.slag_mt:,.2f}")
    ore_slag_mt = float(blend.diagnostics.get("ore_slag_mt", 0.0) or 0.0)
    c10.metric("Ore Slag (MT)", f"{ore_slag_mt:,.2f}")
    fuel_ash_slag_mt = float(blend.diagnostics.get("fuel_ash_slag_mt", 0.0) or 0.0)
    c11.metric("Fuel Ash Slag (MT)", f"{fuel_ash_slag_mt:,.2f}")
    flux_slag_mt = float(blend.diagnostics.get("flux_slag_mt", 0.0) or 0.0)
    c12.metric("Flux Slag (MT)", f"{flux_slag_mt:,.2f}")

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

    fuel_ash_by_fuel = (
        blend.diagnostics.get("fuel_ash_contribution_mt_by_fuel", {}) or {}
    )
    flux_by_flux = blend.diagnostics.get("flux_contribution_mt_by_flux", {}) or {}
    full_balance = blend.diagnostics.get("full_slag_balance", {}) or {}
    slag_components = full_balance.get("slag_components_mt", {}) or {}
    has_fuel_details = any(
        float(value or 0.0) > 0.0 for value in fuel_ash_by_fuel.values()
    )
    has_flux_details = any(float(value or 0.0) > 0.0 for value in flux_by_flux.values())

    if has_fuel_details or has_flux_details or slag_components:
        with st.expander("Slag Balance Details", expanded=False):
            if has_fuel_details:
                fuel_rows = [
                    {
                        "fuel": str(fuel_id).replace("_", " ").title(),
                        "slag_contribution_mt": float(value or 0.0),
                    }
                    for fuel_id, value in fuel_ash_by_fuel.items()
                ]
                st.markdown("##### Fuel Ash Slag Contribution")
                _safe_dataframe(
                    pd.DataFrame(fuel_rows),
                    hide_index=True,
                    use_container_width=True,
                )

            if has_flux_details:
                flux_dry_weights = (
                    blend.diagnostics.get("flux_dry_weight_mt_by_flux", {}) or {}
                )
                flux_rows = [
                    {
                        "flux": str(flux_id).replace("_", " ").title(),
                        "dry_quantity_mt": float(
                            flux_dry_weights.get(flux_id, 0.0) or 0.0
                        ),
                        "slag_contribution_mt": float(value or 0.0),
                    }
                    for flux_id, value in flux_by_flux.items()
                ]
                st.markdown("##### Flux Slag Contribution")
                _safe_dataframe(
                    pd.DataFrame(flux_rows),
                    hide_index=True,
                    use_container_width=True,
                )

            if slag_components:
                st.markdown("##### Full Slag Balance Components")
                component_rows = [
                    {
                        "component": str(component).upper(),
                        "quantity_mt": float(value or 0.0),
                    }
                    for component, value in slag_components.items()
                ]
                _safe_dataframe(
                    pd.DataFrame(component_rows),
                    hide_index=True,
                    use_container_width=True,
                )


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
