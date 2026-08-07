"""Reusable Streamlit components for the BMO page.

This module keeps BMO page rendering helpers separate from workflow orchestration.
It renders status, ore editing, result metrics, result tables, and diagnostics
while handling Streamlit API differences across installed versions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from utils.bmo.calculations import compute_charging_requirements
from utils.bmo.fuel_rates import ASSUMED_FUEL_PRICES_RS_PER_KG
from utils.bmo.types import BlendEvaluation, FluxInput, FuelAshInput, OreInput


@st.cache_data(show_spinner=False)
def _read_bmo_css(css_path: str, mtime_ns: int) -> str:
    """Read and cache the BMO stylesheet text keyed on path + modification time.
    Caching on ``(path, mtime_ns)`` re-reads only
    when the file actually changes.

    Args:
         - css_path: Absolute path to the BMO stylesheet.
         - mtime_ns: File modification time in nanoseconds (cache key).

    Returns:
         - return str - Stylesheet contents, or empty string when missing.
    """

    path = Path(css_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


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
    try:
        mtime_ns = css_path.stat().st_mtime_ns
    except OSError:
        return
    css_text = _read_bmo_css(str(css_path), int(mtime_ns))
    if css_text:
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


def _safe_dataframe(
    df: pd.DataFrame, *, hide_index: bool = True, width: str = "stretch"
) -> None:
    """
    Render a Streamlit dataframe.

    Args:
         - df: pd.DataFrame - Data frame to render.
         - hide_index: bool - Whether to hide the dataframe index.
         - width: str - Streamlit width value.

    Returns:
         - return None - Renders the dataframe in Streamlit.
    """

    st.dataframe(df, hide_index=hide_index, width=width)


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
                "min_share_pct": float(ore.min_share_pct),
                "max_share_pct": float(ore.max_share_pct),
                "moisture_pct": float(ore.chemistry.moisture_pct),
                "fe_t_pct": float(ore.chemistry.fe_t_pct),
                "sio2_pct": float(ore.chemistry.sio2_pct),
                "al2o3_pct": float(ore.chemistry.al2o3_pct),
                "cao_pct": float(ore.chemistry.cao_pct),
                "mgo_pct": float(ore.chemistry.mgo_pct),
                "mno_pct": float(ore.chemistry.mno_pct),
                "tio2_pct": float(ore.chemistry.tio2_pct),
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

    visible_columns = (
        "selected",
        "ore_name",
        "stock_mt",
        "price_rs_per_mt",
        "min_share_pct",
        "max_share_pct",
        "moisture_pct",
        "fe_t_pct",
        "sio2_pct",
        "al2o3_pct",
        "cao_pct",
        "mgo_pct",
        "mno_pct",
        "tio2_pct",
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
            "fe_t_pct": st.column_config.NumberColumn(
                "Fe(T) (%)", min_value=0.0, max_value=100.0, step=0.1
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
            "mno_pct": st.column_config.NumberColumn(
                "MnO (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "tio2_pct": st.column_config.NumberColumn(
                "TiO2 (%)", min_value=0.0, max_value=100.0, step=0.1
            ),
            "min_share_pct": st.column_config.NumberColumn(
                "Min Share (%)", min_value=0.0, max_value=100.0, step=0.5
            ),
            "max_share_pct": st.column_config.NumberColumn(
                "Max Share (%)", min_value=0.0, max_value=100.0, step=0.5
            ),
        },
    }
    editor_kwargs["column_order"] = visible_columns
    editor_kwargs["width"] = "stretch"
    return st.data_editor(editor_df, **editor_kwargs)


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
                "price_rs_per_mt": float(item.get("price_rs_per_mt", 0.0) or 0.0),
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

    visible_columns = (
        "enabled",
        "fuel_name",
        "rate_kg_per_thm",
        "price_rs_per_mt",
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
            "price_rs_per_mt": st.column_config.NumberColumn(
                "Price (Rs/MT)", min_value=0.0, step=100.0,
                help="Current fuel price; the displayed unit fuel cost is re-priced to this.",
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
    editor_kwargs["column_order"] = visible_columns
    editor_kwargs["width"] = "stretch"
    return st.data_editor(editor_df, **editor_kwargs)


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
        if bool(item.get("hidden_by_default", False)):
            continue
        rows.append(
            {
                "enabled": bool(item.get("enabled", True)),
                "flux_id": str(item.get("flux_id", "")),
                "flux_name": str(item.get("display_name", item.get("flux_id", ""))),
                "optimizable": bool(item.get("optimizable", False)),
                "wet_qty_mt": float(item.get("wet_qty_mt", 0.0) or 0.0),
                "price_rs_per_mt": float(item.get("price_rs_per_mt", 0.0) or 0.0),
                "stock_mt": float(item.get("stock_mt", 0.0) or 0.0),
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

    visible_columns = (
        "enabled",
        "flux_name",
        "optimizable",
        "price_rs_per_mt",
        "stock_mt",
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
            "optimizable": st.column_config.CheckboxColumn(
                "LP auto",
                disabled=True,
                help=(
                    "When ticked, the optimizer decides this flux's quantity "
                    "(0..stock) to hold slag basicity within bounds."
                ),
            ),
            "price_rs_per_mt": st.column_config.NumberColumn(
                "Price (Rs/MT)", min_value=0.0, step=50.0
            ),
            "stock_mt": st.column_config.NumberColumn(
                "Stock (MT)", min_value=0.0, step=10.0
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
    editor_kwargs["column_order"] = visible_columns
    editor_kwargs["width"] = "stretch"
    return st.data_editor(editor_df, **editor_kwargs)


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
    editor_kwargs["column_order"] = visible_columns
    editor_kwargs["width"] = "stretch"
    return st.data_editor(editor_df, **editor_kwargs)


def render_hot_metal_chemistry(
    hm_snapshot: dict[str, Any],
    cfg_defaults: dict[str, Any],
) -> dict[str, float]:
    """
    Render Hot Metal PI chemistry inputs prefilled from live HM analysis.

    These four values feed the full slag-balance pig-iron partitioning: HM C/Si/S
    determine the metallic-mass denominator and the SiO2 consumed by Si reduction;
    HM Others (Mn+P+Ti+Cr) closes the PI chemistry. Operators can override the
    prefilled values when the latest HM sample looks abnormal.

    Args:
         - hm_snapshot: dict[str, Any] - Snapshot from get_hm_slag_snapshot; may be empty.
         - cfg_defaults: dict[str, Any] - yml slag_balance defaults used as fallback.

    Returns:
         - return dict[str, float] - Edited carbon_pct, silicon_pct, sulphur_pct, other_pct.
    """

    st.markdown("### Hot Metal Chemistry")
    source = hm_snapshot.get("source") if hm_snapshot else None
    n_rows = int(hm_snapshot.get("n_rows_used", 0) or 0) if hm_snapshot else 0
    if source:
        plural = "s" if n_rows != 1 else ""
        st.caption(f"Source: {source} ({n_rows} HM_SLAG row{plural})")
    else:
        st.caption("HM_SLAG data unavailable — falling back to yml defaults")

    def _prefill(live_key: str, fallback_key: str) -> float:
        live = hm_snapshot.get(live_key) if hm_snapshot else None
        try:
            live_value = float(live) if live is not None else 0.0
        except (TypeError, ValueError):
            live_value = 0.0
        if live_value > 0:
            return live_value
        try:
            return float(cfg_defaults.get(fallback_key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    c1, c2, c3, c4 = st.columns(4)
    values: dict[str, float] = {
        "carbon_pct": float(
            c1.number_input(
                "HM C (%)",
                min_value=0.0,
                max_value=10.0,
                value=_prefill("chem_pct_c", "carbon_pct"),
                step=0.01,
            )
        ),
        "silicon_pct": float(
            c2.number_input(
                "HM Si (%)",
                min_value=0.0,
                max_value=5.0,
                value=_prefill("chem_pct_si", "silicon_pct"),
                step=0.01,
            )
        ),
        "sulphur_pct": float(
            c3.number_input(
                "HM S (%)",
                min_value=0.0,
                max_value=1.0,
                value=_prefill("chem_pct_s", "sulphur_pct"),
                step=0.001,
                format="%.3f",
            )
        ),
        "other_pct": float(
            c4.number_input(
                "HM Others (%) (Mn+P+Ti+Cr)",
                min_value=0.0,
                max_value=5.0,
                value=_prefill("others_pct", "other_pct"),
                step=0.01,
            )
        ),
    }
    return values


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
        value=float(settings_cfg.get("slag_correction_factor", 1.0)),
        step=0.001,
        help=(
            "Leave at 1.0. The HM-driven slag balance already subtracts SiO2/Fe/Mn/S "
            "into pig iron; this empirical factor exists only for legacy calibration."
        ),
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


def render_blend_metrics(
    title: str,
    blend: BlendEvaluation,
    observed_slag_rate_kg_per_thm: float | None = None,
    is_lp_mode: bool = False,
    charge_mass_mt: float = 26.4,
    charging_hours_per_day: float = 24.0,
) -> None:
    """
    Render summary metrics and constraint warnings for a blend result.

    The metric rows separate wet quantity, dry quantity, slag MT, and slag rate
    so users can see how moisture and slag-forming oxides affect the result. When
    an observed slag rate is supplied, it is shown beside the computed value with
    the gap so operators can manually tune the slag correction factor.

    Args:
         - title: str - Section title shown above the metrics.
         - blend: BlendEvaluation - Evaluated blend result to display.
         - observed_slag_rate_kg_per_thm: float | None - Plant slag rate from DPR for comparison.
         - charge_mass_mt: float - Tonnes carried by one furnace charge.
         - charging_hours_per_day: float - Hours over which the daily blend is charged.

    Returns:
         - return None - Writes metrics and warnings to the Streamlit page.
    """

    st.markdown(f"#### {title}")

    # Fuel-cost source: model vs deterministic fallback. The label and help text
    # change so an operator can tell at a glance which one fed the objective.
    model_prediction = blend.diagnostics.get("model_prediction")
    fuel_used_fallback = bool(
        getattr(model_prediction, "used_fallback", False)
        if model_prediction is not None
        else False
    )
    fuel_label = (
        "Fuel Cost (Rs/THM, fallback)" if fuel_used_fallback else "Fuel Cost (Rs/THM)"
    )
    fuel_help = (
        "Fallback formula in use - the XGBoost model was unavailable or "
        "rejected the prediction. Treat the cost as a placeholder."
        if fuel_used_fallback
        else None
    )

    # Row 1: costs + Fe produced. Ore cost first in both modes; only the
    # emphasis/help differs (LP minimises ore cost; DE minimises ore + fuel).
    c1, c2, c_flux, c3, c4 = st.columns(5)
    correction_priced_into_lp = bool(
        blend.diagnostics.get("lp_coke_correction_linear_terms")
    )
    if is_lp_mode:
        c1.metric(
            "Ore Cost (Rs/THM, LP-optimised)",
            f"{blend.ore_cost_per_thm_rs:,.2f}",
            help=(
                (
                    "LP minimises ore cost plus flux cost plus the physics "
                    "coke-rate correction, subject to Fe target, slag cap, "
                    "share bounds, and stock bounds."
                )
                if correction_priced_into_lp
                else (
                    "LP minimises ore cost only, subject to Fe target, slag cap, "
                    "share bounds, and stock bounds."
                )
            ),
        )
    else:
        c1.metric(
            "Ore Cost (Rs/THM)",
            f"{blend.ore_cost_per_thm_rs:,.2f}",
            help="DE jointly minimises ore + fuel cost.",
        )
    # Fuel + total cost are shown at the operator's current fuel prices from the
    # Fuel Ash editor. The optimizer still uses the model's baseline-price cost;
    # display re-pricing falls back to a model-cost residual only when the editor
    # rates are unavailable.
    adjusted_fuel = blend.diagnostics.get("adjusted_fuel_cost_per_thm_rs")
    adjusted_total = blend.diagnostics.get("adjusted_objective_rs_per_thm")
    fuel_cost_display = (
        float(adjusted_fuel) if adjusted_fuel is not None
        else blend.fuel_cost_per_thm_rs
    )
    # Flux the optimizer bought (dolomite/quartz/limestone) is a real spend, so it
    # belongs in the total cost alongside ore + fuel.
    flux_cost_display = float(blend.diagnostics.get("flux_cost_per_thm_rs", 0.0) or 0.0)
    base_total = (
        float(adjusted_total) if adjusted_total is not None
        else blend.objective_rs_per_thm
    )
    total_display = base_total + flux_cost_display
    reprice_help = (
        " The model-predicted fuel cost for this blend, converted to the "
        "current fuel prices; the model's baseline-price value is "
        f"Rs {blend.fuel_cost_per_thm_rs:,.0f}/THM."
        if adjusted_fuel is not None
        else ""
    )
    fuel_label_est = (
        "Fuel Cost (Rs/THM, est., current price, fallback)"
        if fuel_used_fallback
        else "Fuel Cost (Rs/THM, est., current price)"
    )
    fuel_help_base = (
        "Post-hoc XGBoost estimate on the selected blend." if is_lp_mode else fuel_help
    )
    # What the physics correction costs, in the same units as the tile it sits
    # under. Without this the operator can see the coke-rate delta but has no way
    # to tell how much of the fuel bill it accounts for.
    correction_delta_kg = blend.diagnostics.get("coke_correction_delta_kg_thm")
    correction_cost = None
    if correction_delta_kg:
        correction_cost = float(correction_delta_kg) * float(
            (blend.diagnostics.get("current_fuel_prices_rs_per_kg") or {}).get(
                "coke", ASSUMED_FUEL_PRICES_RS_PER_KG["coke"]
            )
        )
    c2.metric(
        fuel_label_est,
        f"{fuel_cost_display:,.2f}",
        delta=(
            f"{correction_cost:+,.0f} physics" if correction_cost else None
        ),
        delta_color="off",
        help=((fuel_help_base or "") + reprice_help) or None,
    )
    c_flux.metric(
        "Flux Cost (Rs/THM)",
        f"{flux_cost_display:,.2f}",
        help=(
            "Cost of the flux (dolomite/quartz/limestone) the optimizer added to "
            "hold slag basicity within bounds. Included in Total Cost."
        ),
    )
    total_label = (
        "Total Cost (ore + fuel + flux, Rs/THM)"
        if is_lp_mode
        else "Total Cost (Rs/THM, current price)"
    )
    c3.metric(
        total_label,
        f"{total_display:,.2f}",
        help=(
            "Ore cost plus the current-price fuel-cost estimate plus optimizer-"
            "added flux cost."
            if is_lp_mode
            else "DE minimises ore + fuel at baseline prices; display adds "
            "optimizer flux cost and re-prices fuel at current prices."
        ),
    )
    c4.metric("Fe Produced (MT)", f"{blend.fe_production_mt:,.2f}")

    charging = compute_charging_requirements(
        blend,
        charge_mass_mt=charge_mass_mt,
        hours_per_day=charging_hours_per_day,
    )

    # Row 2: dry ore quantity, charging burden/flux rate, Fe%, and slag.
    c5, c_burden, c_flux_rate, c6, c7, c8 = st.columns(6)
    dry_qty = float(blend.diagnostics.get("total_dry_qty_mt", 0.0) or 0.0)
    burden_qty = float(
        blend.diagnostics.get("total_burden_qty_mt", blend.total_qty_mt) or 0.0
    )
    hm_basis_mt = float(blend.diagnostics.get("hot_metal_target_mt", 0.0) or 0.0)
    c5.metric("Dry Qty (MT)", f"{dry_qty:,.2f}")
    c_burden.metric(
        "IBRM + Flux (MT)",
        f"{burden_qty:,.2f}",
        help=(
            "Total wet ore/sinter/pellet plus enabled wet flux sent through the "
            "charging system. This is the quantity checked against the furnace "
            "charging-capacity limit."
        ),
    )
    flux_rate = charging["flux_rate_kg_per_thm"]
    c_flux_rate.metric(
        "Flux Rate (kg/THM)",
        f"{float(flux_rate):,.1f}" if flux_rate is not None else "n/a",
        help="Total enabled wet flux divided by the target hot metal tonnage.",
    )
    c6.metric("Final Fe (%)", f"{blend.fe_t_pct:,.2f}")
    # Slag rate is undefined when Fe production is zero; show "n/a" rather
    # than 0.00 kg/THM which would mislead an operator into thinking the
    # blend produced clean iron. The observed value is the plant's realised
    # slag rate averaged over the chosen chemistry window (recent days).
    if hm_basis_mt <= 0:
        c7.metric("Slag Rate (kg/THM)", "n/a")
    elif observed_slag_rate_kg_per_thm and observed_slag_rate_kg_per_thm > 0:
        delta = blend.slag_rate_kg_per_thm - float(observed_slag_rate_kg_per_thm)
        c7.metric(
            "Slag Rate (kg/THM)",
            f"{blend.slag_rate_kg_per_thm:,.2f}",
            delta=(
                f"{delta:+.1f} vs observed {observed_slag_rate_kg_per_thm:,.1f} "
                "(recent avg)"
            ),
            delta_color="off",
            help="Observed = plant DPR slag/HM averaged over the chemistry window.",
        )
    else:
        c7.metric("Slag Rate (kg/THM)", f"{blend.slag_rate_kg_per_thm:,.2f}")
    c8.metric("Slag (MT)", f"{blend.slag_mt:,.2f}")

    if fuel_used_fallback:
        reason = (getattr(model_prediction, "details", {}) or {}).get("reason")
        reason_text = f" Reason: {reason}." if reason else ""
        st.caption(
            "WARNING - Fuel cost above came from the deterministic fallback "
            "formula, not the BMO XGBoost model." + reason_text
        )

    # Row 3: basicity pair.
    b1, b2 = st.columns(2)
    basicity_denominator_mt = float(
        blend.diagnostics.get("slag_basicity_denominator_mt", 0.0) or 0.0
    )
    if basicity_denominator_mt > 0:
        b1.metric("Slag Basicity CaO/SiO2", f"{blend.slag_basicity:,.3f}")
        b2.metric("Slag T Basicity", f"{blend.slag_t_basicity:,.3f}")
    else:
        b1.metric("Slag Basicity CaO/SiO2", "n/a")
        b2.metric("Slag T Basicity", "n/a")

    fuel_rate_estimate = blend.diagnostics.get("fuel_rate_estimate")
    anchor_estimate = blend.diagnostics.get("fuel_rate_estimate_anchor")
    correction_delta = blend.diagnostics.get("coke_correction_delta_kg_thm")
    st.markdown("##### Estimated Fuel Rates")
    if isinstance(fuel_rate_estimate, dict):
        r1, r2, r3, r4 = st.columns(4)
        coke_help = None
        if isinstance(anchor_estimate, dict) and correction_delta:
            # Name the uncorrected value the correction moved away from, so the
            # operator never has to take the corrected number on faith.
            coke_help = (
                f"Model-derived (uncorrected): "
                f"{float(anchor_estimate.get('coke_rate_kg_thm', 0.0)):,.1f} kg/THM. "
                "Physics correction shown as the delta."
            )
        r1.metric(
            "Coke Rate (kg/THM)",
            f"{float(fuel_rate_estimate.get('coke_rate_kg_thm', 0.0)):,.1f}",
            delta=(
                f"{float(correction_delta):+,.1f} physics"
                if correction_delta
                else None
            ),
            delta_color="off",
            help=coke_help,
        )
        r2.metric(
            "Nut Coke Rate (kg/THM)",
            f"{float(fuel_rate_estimate.get('nut_coke_rate_kg_thm', 0.0)):,.1f}",
            help=f"Source: {fuel_rate_estimate.get('nut_coke_source', 'unknown')}",
        )
        total_fuel_help = None
        if isinstance(anchor_estimate, dict) and correction_delta:
            # The whole point of the correction is that it moves the total fuel
            # rate. Naming the value it moved from is what makes that legible -
            # without it the operator has no baseline to compare against and the
            # delta looks like it went missing.
            total_fuel_help = (
                "Uncorrected: "
                f"{float(anchor_estimate.get('total_fuel_rate_kg_thm', 0.0)):,.1f}"
                " kg/THM. Only coke moves; nut coke and PCI are run inputs."
            )
        r3.metric(
            "Total Fuel Rate (kg/THM)",
            f"{float(fuel_rate_estimate.get('total_fuel_rate_kg_thm', 0.0)):,.1f}",
            delta=(
                f"{float(correction_delta):+,.1f} physics"
                if correction_delta
                else None
            ),
            delta_color="off",
            help=total_fuel_help,
        )
        r4.metric(
            "PCI (kg/THM)",
            f"{float(fuel_rate_estimate.get('pci_rate_kg_thm', 0.0)):,.1f}",
            help=f"Source: {fuel_rate_estimate.get('pci_source', 'unknown')}",
        )
    else:
        st.caption("Fuel-rate estimate unavailable because latest PCI rate is missing.")

    st.markdown("##### Charging Requirement")
    charge_cols = st.columns(4)
    nut_coke_total_mt = charging["nut_coke_total_mt"]
    total_charge_mix_mt = charging["total_charge_mix_mt"]
    charge_mix_mt_per_hour = charging["charge_mix_mt_per_hour"]
    required_charges_per_hour = charging["required_charges_per_hour"]
    charge_cols[0].metric(
        "Nut Coke in Charges (MT)",
        (
            f"{float(nut_coke_total_mt):,.1f}"
            if nut_coke_total_mt is not None
            else "n/a"
        ),
    )
    charge_cols[1].metric(
        "Total Charge Mix (MT)",
        (
            f"{float(total_charge_mix_mt):,.1f}"
            if total_charge_mix_mt is not None
            else "n/a"
        ),
        help="Wet IBRM + wet flux + nut coke tonnes.",
    )
    charge_cols[2].metric(
        "Charge Mix (MT/hr)",
        (
            f"{float(charge_mix_mt_per_hour):,.1f}"
            if charge_mix_mt_per_hour is not None
            else "n/a"
        ),
        help=f"Total charge mix divided by {charging_hours_per_day:g} hours.",
    )
    charge_cols[3].metric(
        "Required Charges (/hr)",
        (
            f"{float(required_charges_per_hour):,.2f}"
            if required_charges_per_hour is not None
            else "n/a"
        ),
        help=(
            f"Charge mix MT/hr divided by {charge_mass_mt:g} MT per charge. "
            "For 4,200 MT/day: 4,200 / 24 / 26.4 = 6.63 charges/hr."
        ),
    )

    if blend.violations:
        st.warning("Constraint violations:\n- " + "\n- ".join(blend.violations))


def render_coke_correction_breakdown(blend: BlendEvaluation) -> None:
    """
    Render the per-term physics coke-rate correction for one blend.

    The correction supplies the entire blend-to-fuel sensitivity in BMO, because
    the trained fuel model barely responds to the burden at all. That makes it
    the one number an operator most needs to be able to interrogate, so every
    term shows its coefficient, both driver values, and where its reference came
    from — and both the uncorrected and corrected coke rates stay on screen.

    Args:
         - blend: BlendEvaluation - Blend carrying ``coke_correction`` diagnostics.

    Returns:
         - return None - Renders Streamlit widgets.
    """

    correction = blend.diagnostics.get("coke_correction")
    if not isinstance(correction, dict) or not correction.get("enabled"):
        return

    anchor = float(correction.get("anchor_coke_rate_kg_thm", 0.0) or 0.0)
    corrected = float(correction.get("corrected_coke_rate_kg_thm", 0.0) or 0.0)
    delta = float(correction.get("applied_delta_kg_thm", 0.0) or 0.0)
    applied = bool(blend.diagnostics.get("coke_correction_applied", False))
    warnings = list(correction.get("warnings", []) or [])

    with st.expander(
        f"Coke-rate physics correction: {anchor:,.1f} → {corrected:,.1f} kg/THM "
        f"({delta:+,.1f})",
        # A warning about an implausible anchor is an operator action, not an
        # audit detail. Open the breakdown so it cannot hide behind a collapsed
        # expander; clean runs remain compact.
        expanded=bool(warnings),
    ):
        st.caption(
            "Priced into the optimizer's objective."
            if applied
            else "Shown for reference only; not added to this blend's cost."
        )

        rows = [
            {
                "Term": term.get("label", term.get("term_id", "")),
                "Coefficient": term.get("k_display", ""),
                "Blend": term.get("x_blend"),
                "Reference": term.get("x_reference"),
                "Units": term.get("x_units", ""),
                "Reference source": (
                    term.get("reference_source")
                    or term.get("disabled_reason")
                    or ""
                ),
                "Δ Coke (kg/THM)": term.get("delta_kg_thm"),
                "Capped": bool(term.get("term_clamp_binding"))
                or bool(term.get("envelope_exceeded")),
            }
            for term in correction.get("terms", [])
            # Disabled-by-configuration terms would only add rows that always
            # read zero, which erodes trust in the rows that do matter.
            if term.get("enabled")
            or term.get("disabled_reason") != "term disabled in configuration"
        ]
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Blend": st.column_config.NumberColumn("Blend", format="%.2f"),
                    "Reference": st.column_config.NumberColumn(
                        "Reference", format="%.2f"
                    ),
                    "Δ Coke (kg/THM)": st.column_config.NumberColumn(
                        "Δ Coke (kg/THM)", format="%+.2f"
                    ),
                },
            )

        for warning in warnings:
            st.warning(str(warning))

        lp_terms = blend.diagnostics.get("lp_coke_correction_linear_terms")
        if isinstance(lp_terms, dict):
            st.caption(
                "The LP priced this correction linearly, so its signal is not "
                "clamped; the value above is. Large gaps between the two are "
                "reported as warnings."
            )


def build_blend_table_df(
    blend: BlendEvaluation, selected_ores: list[OreInput]
) -> pd.DataFrame:
    """
    Build the result table data for a blend.

    Args:
         - blend: BlendEvaluation - Evaluated blend result to display.
         - selected_ores: list[OreInput] - Ores included in the current run.

    Returns:
         - return pd.DataFrame - Per-ore result table sorted by share.
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
        fe_mt = float(fe_contribution_by_ore.get(ore.ore_id, 0.0))
        slag_mt = float(slag_contribution_by_ore.get(ore.ore_id, 0.0))
        rows.append(
            {
                "ore_name": ore.display_name,
                "quantity_mt": qty,
                "dry_quantity_mt": float(dry_weight_by_ore.get(ore.ore_id, 0.0)),
                "moisture_pct": float(ore.chemistry.moisture_pct),
                "fe_contribution_mt": fe_mt,
                "slag_contribution_mt": slag_mt,
                "slag_per_fe": slag_mt / fe_mt if fe_mt > 0 else 0.0,
                "share_pct": share,
                "stock_mt": ore.stock_mt,
                "price_rs_per_mt": ore.price_rs_per_mt,
                "ore_cost_rs": qty * ore.price_rs_per_mt,
                "ore_cost_lakhs": (qty * ore.price_rs_per_mt) / 1.0e5,
            }
        )
    return pd.DataFrame(rows).sort_values("share_pct", ascending=False)


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

    df = build_blend_table_df(blend, selected_ores)
    # Share is shown as an inline progress bar so the burden split is
    # readable at a glance; ore cost is in lakhs to keep the numbers short.
    share_max = float(max(100.0, df["share_pct"].max())) if not df.empty else 100.0
    share_col = st.column_config.ProgressColumn(
        "Share (%)", format="%.1f%%", min_value=0.0, max_value=share_max
    )
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_order=(
            "ore_name",
            "share_pct",
            "quantity_mt",
            "dry_quantity_mt",
            "fe_contribution_mt",
            "slag_contribution_mt",
            "price_rs_per_mt",
            "ore_cost_lakhs",
        ),
        column_config={
            "ore_name": st.column_config.TextColumn("Ore"),
            "share_pct": share_col,
            "quantity_mt": st.column_config.NumberColumn(
                "Wet Qty (MT)", format="%.1f"
            ),
            "dry_quantity_mt": st.column_config.NumberColumn(
                "Dry Qty (MT)", format="%.1f"
            ),
            "fe_contribution_mt": st.column_config.NumberColumn(
                "Fe (MT)", format="%.1f"
            ),
            "slag_contribution_mt": st.column_config.NumberColumn(
                "Slag (MT)", format="%.1f"
            ),
            "price_rs_per_mt": st.column_config.NumberColumn(
                "Price (Rs/MT)", format="%.0f"
            ),
            "ore_cost_lakhs": st.column_config.NumberColumn(
                "Ore Cost (₹ Lakhs)", format="%.2f"
            ),
        },
    )


def render_slag_balance_details(
    blend: BlendEvaluation,
    selected_ores: list[OreInput],
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
) -> None:
    """
    Render the full slag balance breakdown: per-source rollup, per-oxide balance,
    HM-side removals with worked formulas, and per-material per-oxide contribution.

    The user often wants to verify where each MT of slag (or each MT of TiO2)
    comes from. This panel exposes (a) source rollup ore vs fuel ash vs flux,
    (b) per-oxide balance showing what entered from each source and what stayed
    in slag vs went to HM, (c) explicit HM-removal arithmetic, and (d) per-
    material tables for each ore, fuel, and flux row so the operator can
    cross-check input chemistry against the resulting slag.

    Args:
         - blend: BlendEvaluation - Evaluated blend result.
         - selected_ores: list[OreInput] - Ores included in the run.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash rows used.
         - flux_inputs: list[FluxInput] | None - Flux rows used.

    Returns:
         - return None - Writes details inside a Streamlit expander.
    """

    diag = blend.diagnostics or {}
    full_balance = diag.get("full_slag_balance", {}) or {}
    if not full_balance:
        return

    fb_diag = full_balance.get("diagnostics", {}) or {}
    net_into_bf = full_balance.get("net_into_bf_mt", {}) or {}
    slag_components = full_balance.get("slag_components_mt", {}) or {}
    ore_components = full_balance.get("ore_components_mt", {}) or {}
    flux_components = full_balance.get("flux_components_mt", {}) or {}
    fuel_components = full_balance.get("fuel_ash_components_mt", {}) or {}
    dry_by_ore = diag.get("dry_weight_mt_by_ore", {}) or {}
    flux_dry = diag.get("flux_dry_weight_mt_by_flux", {}) or {}

    ti_factor = 79.866 / 47.867
    mn_factor = 70.937 / 54.938
    fe_to_feo_factor = 72.0 / 56.0

    with st.expander("Slag Balance Details", expanded=False):
        # --- Section 1: Per-source rollup ---
        st.markdown("##### Per-source contribution to final slag")
        raw_ore = float(diag.get("raw_ore_slag_mt", 0.0) or 0.0)
        raw_fuel = float(diag.get("raw_fuel_ash_slag_mt", 0.0) or 0.0)
        raw_flux = float(diag.get("raw_flux_slag_mt", 0.0) or 0.0)
        post_ore = float(diag.get("ore_slag_mt", 0.0) or 0.0)
        post_fuel = float(diag.get("fuel_ash_slag_mt", 0.0) or 0.0)
        post_flux = float(diag.get("flux_slag_mt", 0.0) or 0.0)
        rollup_rows = [
            {
                "source": "Ore",
                "raw_oxide_input_mt": raw_ore,
                "hm_removal_mt": max(0.0, raw_ore - post_ore),
                "final_slag_mt": post_ore,
            },
            {
                "source": "Fuel ash",
                "raw_oxide_input_mt": raw_fuel,
                "hm_removal_mt": max(0.0, raw_fuel - post_fuel),
                "final_slag_mt": post_fuel,
            },
            {
                "source": "Flux",
                "raw_oxide_input_mt": raw_flux,
                "hm_removal_mt": max(0.0, raw_flux - post_flux),
                "final_slag_mt": post_flux,
            },
        ]
        rollup_total = {
            "source": "TOTAL",
            "raw_oxide_input_mt": raw_ore + raw_fuel + raw_flux,
            "hm_removal_mt": sum(r["hm_removal_mt"] for r in rollup_rows),
            "final_slag_mt": post_ore + post_fuel + post_flux,
        }
        rollup_rows.append(rollup_total)
        _safe_dataframe(pd.DataFrame(rollup_rows), hide_index=True, width="stretch")

        # --- Section 2: Per-oxide balance ---
        st.markdown("##### Per-oxide balance (where each oxide goes)")
        oxide_rows = []
        for comp in ("sio2", "al2o3", "cao", "mgo", "alkali", "caf2"):
            input_mt = float(net_into_bf.get(comp, 0.0))
            slag_mt = float(slag_components.get(comp, 0.0))
            oxide_rows.append(
                {
                    "oxide": comp.upper(),
                    "from_ore_mt": float(ore_components.get(comp, 0.0)),
                    "from_fuel_ash_mt": float(fuel_components.get(comp, 0.0)),
                    "from_flux_mt": float(flux_components.get(comp, 0.0)),
                    "input_after_dust_mt": input_mt,
                    "removed_by_hm_mt": max(0.0, input_mt - slag_mt),
                    "in_final_slag_mt": slag_mt,
                }
            )
        # Fe (element) -> FeO in slag
        fe_input = float(net_into_bf.get("fe", 0.0))
        feo_slag = float(slag_components.get("feo", 0.0))
        oxide_rows.append(
            {
                "oxide": "Fe -> FeO",
                "from_ore_mt": float(ore_components.get("fe", 0.0)),
                "from_fuel_ash_mt": float(fuel_components.get("fe", 0.0)),
                "from_flux_mt": float(flux_components.get("fe", 0.0)),
                "input_after_dust_mt": fe_input,
                "removed_by_hm_mt": max(0.0, fe_input - feo_slag / fe_to_feo_factor),
                "in_final_slag_mt": feo_slag,
            }
        )
        # Mn (element) -> MnO in slag
        mn_input = float(net_into_bf.get("mn", 0.0))
        mno_slag = float(slag_components.get("mno", 0.0))
        oxide_rows.append(
            {
                "oxide": "Mn -> MnO",
                "from_ore_mt": float(ore_components.get("mn", 0.0)) * mn_factor,
                "from_fuel_ash_mt": float(fuel_components.get("mn", 0.0)) * mn_factor,
                "from_flux_mt": float(flux_components.get("mn", 0.0)) * mn_factor,
                "input_after_dust_mt": mn_input * mn_factor,
                "removed_by_hm_mt": max(
                    0.0, (mn_input - mno_slag / mn_factor) * mn_factor
                ),
                "in_final_slag_mt": mno_slag,
            }
        )
        # Ti (element) -> TiO2 (per spec NOT in slag; all "removed")
        ti_input = float(net_into_bf.get("ti", 0.0))
        oxide_rows.append(
            {
                "oxide": "Ti -> TiO2",
                "from_ore_mt": float(ore_components.get("ti", 0.0)) * ti_factor,
                "from_fuel_ash_mt": float(fuel_components.get("ti", 0.0)) * ti_factor,
                "from_flux_mt": float(flux_components.get("ti", 0.0)) * ti_factor,
                "input_after_dust_mt": ti_input * ti_factor,
                "removed_by_hm_mt": ti_input * ti_factor,
                "in_final_slag_mt": 0.0,
            }
        )
        # S (element) split
        s_input = float(net_into_bf.get("s", 0.0))
        s_slag = float(slag_components.get("s", 0.0))
        oxide_rows.append(
            {
                "oxide": "S",
                "from_ore_mt": float(ore_components.get("s", 0.0)),
                "from_fuel_ash_mt": float(fuel_components.get("s", 0.0)),
                "from_flux_mt": float(flux_components.get("s", 0.0)),
                "input_after_dust_mt": s_input,
                "removed_by_hm_mt": max(0.0, s_input - s_slag),
                "in_final_slag_mt": s_slag,
            }
        )
        _safe_dataframe(pd.DataFrame(oxide_rows), hide_index=True, width="stretch")

        # --- Section 3: HM-removal worked breakdown ---
        st.markdown("##### Hot-Metal removals — worked breakdown")
        actual_pi = float(full_balance.get("actual_pig_iron_mt", 0.0) or 0.0)
        hm_si = float(fb_diag.get("hm_mn_pct_used", 0.0)) * 0  # placeholder
        hm_si_pct = (
            float(blend.diagnostics.get("hm_reduction_sio2_mt", 0.0))
            / (actual_pi * 2.14)
            * 100
            if actual_pi > 0
            else 0.0
        )
        sulphur_pi = float(fb_diag.get("sulphur_to_pig_iron_mt", 0.0))
        sulphur_gas = float(fb_diag.get("sulphur_to_gas_mt", 0.0))
        hm_s_pct = (sulphur_pi / actual_pi * 100) if actual_pi > 0 else 0.0
        alkali_fraction = float(fb_diag.get("alkali_to_slag_fraction", 0.8))
        alkali_input = float(net_into_bf.get("alkali", 0.0))
        sio2_consumed = float(fb_diag.get("sio2_consumed_by_si_mt", 0.0))
        worked_rows = [
            {
                "removal": "SiO2 to HM Si",
                "formula": f"PI x HM_Si% x 2.14 = {actual_pi:.0f} x {hm_si_pct:.3f}% x 2.14",
                "value_mt": round(sio2_consumed, 2),
            },
            {
                "removal": "MnO to HM Mn",
                "formula": f"Mn_to_HM x 1.291 (Mn_to_HM = {fb_diag.get('mn_to_pig_iron_mt', 0):.2f} MT element)",
                "value_mt": round(
                    float(blend.diagnostics.get("hm_reduction_mno_mt", 0.0)), 2
                ),
            },
            {
                "removal": "TiO2 to HM Ti",
                "formula": f"Ti_to_HM x 1.668 (Ti_to_HM = {fb_diag.get('ti_to_pig_iron_mt', 0):.2f} MT element)",
                "value_mt": round(
                    float(blend.diagnostics.get("hm_reduction_tio2_mt", 0.0)), 2
                ),
            },
            {
                "removal": "S to PI",
                "formula": f"PI x HM_S% = {actual_pi:.0f} x {hm_s_pct:.4f}%",
                "value_mt": round(sulphur_pi, 3),
            },
            {
                "removal": "S to Gas",
                "formula": f"input_S x sulphur_gas_loss% = {s_input:.2f} x {float(fb_diag.get('sulphur_to_gas_mt', 0))/max(s_input,1e-9)*100:.1f}%",
                "value_mt": round(sulphur_gas, 3),
            },
            {
                "removal": "S to HM + Gas (tile value)",
                "formula": "S_to_PI + S_to_Gas",
                "value_mt": round(sulphur_pi + sulphur_gas, 3),
            },
            {
                "removal": "Alkali to Gas",
                "formula": f"input_alkali x (1 - alkali_to_slag) = {alkali_input:.2f} x (1 - {alkali_fraction:.2f})",
                "value_mt": round(alkali_input * (1 - alkali_fraction), 3),
            },
            {
                "removal": "TiO2 lost (not in slag per spec)",
                "formula": f"Ti_remaining x 1.668 (Ti_remaining = input - Ti_to_HM)",
                "value_mt": round(
                    float(blend.diagnostics.get("tio2_unaccounted_mt", 0.0)), 2
                ),
            },
        ]
        _safe_dataframe(pd.DataFrame(worked_rows), hide_index=True, width="stretch")

        # --- Section 4: Per-material slag contribution ---
        st.markdown("##### Per-material slag contribution (simplified oxide sum)")
        slag_by_ore = diag.get("slag_contribution_mt_by_ore", {}) or {}
        fuel_by_id = diag.get("fuel_ash_contribution_mt_by_fuel", {}) or {}
        flux_by_id = diag.get("flux_contribution_mt_by_flux", {}) or {}
        per_mat_rows: list[dict[str, Any]] = []
        for ore in selected_ores:
            per_mat_rows.append(
                {
                    "material": ore.display_name,
                    "type": "ore",
                    "dry_mt": round(float(dry_by_ore.get(ore.ore_id, 0.0)), 1),
                    "slag_contribution_mt": round(
                        float(slag_by_ore.get(ore.ore_id, 0.0)), 2
                    ),
                }
            )
        for fuel in fuel_ash_inputs or []:
            if not fuel.enabled:
                continue
            per_mat_rows.append(
                {
                    "material": fuel.display_name,
                    "type": "fuel ash",
                    "dry_mt": None,
                    "slag_contribution_mt": round(
                        float(fuel_by_id.get(fuel.fuel_id, 0.0)), 2
                    ),
                }
            )
        for flux in flux_inputs or []:
            if not flux.enabled:
                continue
            per_mat_rows.append(
                {
                    "material": flux.display_name,
                    "type": "flux",
                    "dry_mt": round(float(flux_dry.get(flux.flux_id, 0.0)), 1),
                    "slag_contribution_mt": round(
                        float(flux_by_id.get(flux.flux_id, 0.0)), 2
                    ),
                }
            )
        _safe_dataframe(pd.DataFrame(per_mat_rows), hide_index=True, width="stretch")

        # --- Section 5: Per-material TiO2 contribution (for verification) ---
        st.markdown(
            "##### Per-material TiO2 input (so you can verify the 'TiO2 lost' line)"
        )
        ti_rows = []
        total_tio2 = 0.0
        for ore in selected_ores:
            dry = float(dry_by_ore.get(ore.ore_id, 0.0))
            tio2_pct = float(ore.chemistry.tio2_pct or 0.0)
            tio2_mt = dry * tio2_pct / 100.0
            total_tio2 += tio2_mt
            ti_rows.append(
                {
                    "material": ore.display_name,
                    "type": "ore",
                    "dry_mt": round(dry, 1),
                    "tio2_pct": round(tio2_pct, 3),
                    "tio2_input_mt": round(tio2_mt, 3),
                }
            )
        hot_metal_for_fuel = float(blend.fe_production_mt or 0.0)
        for fuel in fuel_ash_inputs or []:
            if not fuel.enabled:
                continue
            wet_mt = float(fuel.rate_kg_per_thm) * hot_metal_for_fuel / 1000.0
            dry_mt = wet_mt * (1.0 - float(fuel.moisture_pct) / 100.0)
            ash_mt = dry_mt * (float(fuel.ash_pct) / 100.0)
            tio2_mt = ash_mt * (float(fuel.tio2_pct) / 100.0)
            total_tio2 += tio2_mt
            ti_rows.append(
                {
                    "material": f"{fuel.display_name} (ash)",
                    "type": "fuel ash",
                    "dry_mt": round(ash_mt, 1),
                    "tio2_pct": round(float(fuel.tio2_pct), 3),
                    "tio2_input_mt": round(tio2_mt, 3),
                }
            )
        for flux in flux_inputs or []:
            if not flux.enabled:
                continue
            dry = float(flux_dry.get(flux.flux_id, 0.0))
            tio2_pct = float(flux.tio2_pct or 0.0)
            tio2_mt = dry * tio2_pct / 100.0
            if tio2_mt > 0:
                total_tio2 += tio2_mt
                ti_rows.append(
                    {
                        "material": flux.display_name,
                        "type": "flux",
                        "dry_mt": round(dry, 1),
                        "tio2_pct": round(tio2_pct, 3),
                        "tio2_input_mt": round(tio2_mt, 3),
                    }
                )
        ti_rows.append(
            {
                "material": "TOTAL",
                "type": "",
                "dry_mt": None,
                "tio2_pct": None,
                "tio2_input_mt": round(total_tio2, 2),
            }
        )
        _safe_dataframe(pd.DataFrame(ti_rows), hide_index=True, width="stretch")
        st.caption(
            "Compare TOTAL above with the 'TiO2 to HM Ti' tile plus 'TiO2 lost' caption "
            "on the metric panel. If the total looks wrong, check the source materials' "
            "TiO2 % in the Ore Editor / Fuel Ash Inputs / Flux Inputs tables above."
        )

        # --- Section 5: Per-source contribution summary (merged here so the
        # page shows a single "Slag Balance Details" expander) ---
        fuel_ash_by_fuel = (
            blend.diagnostics.get("fuel_ash_contribution_mt_by_fuel", {}) or {}
        )
        flux_by_flux = (
            blend.diagnostics.get("flux_contribution_mt_by_flux", {}) or {}
        )
        if any(float(value or 0.0) > 0.0 for value in fuel_ash_by_fuel.values()):
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
                width='stretch',
            )

        if any(float(value or 0.0) > 0.0 for value in flux_by_flux.values()):
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
                width='stretch',
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
                width='stretch',
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
