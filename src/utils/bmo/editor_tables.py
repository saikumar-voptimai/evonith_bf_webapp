"""Page 9's input tables without Streamlit: the editor-table builders and their typed records (SV-3.5, #302).

The Blend Mix Optimiser page edits its inputs in tables: ``build_*_editor_df`` turn configuration and plant data into
those tables, and ``*_inputs_from_editor`` turn an edited table back into the dataclass records the LP, DE and the slag
balance read (``float_from_row`` reads one cell). ``slag_balance_defaults`` is what the Advanced Slag Balance inputs
hold before an operator edits them. They are pure (pandas and dataclasses), so the service layer
(``services/bmo_case.py``) composes a run's inputs with exactly the functions the page uses. Moved unchanged from
``ui/bmo/components.py`` and ``ui/bmo/editor_inputs.py``, which re-export them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from utils.bmo.fuel_rates import ASSUMED_FUEL_PRICES_RS_PER_KG
from utils.bmo.types import (
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceSettings,
    oxide_pct_from_basis,
)


def build_ore_editor_df(ores: list[OreInput], default_selected_ids: list[str]) -> pd.DataFrame:
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
                "mn_basis": "mno",
                "mno_pct": float(ore.chemistry.mno_pct),
                "ti_basis": "tio2",
                "tio2_pct": float(ore.chemistry.tio2_pct),
            }
        )
    return pd.DataFrame(rows)


def build_fuel_ash_editor_df(fuel_ash_cfg: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Build the editable fuel ash chemistry table from BMO configuration.

    Fuel ash defaults come from the laboratory ash-analysis workbook and can be
    overridden by operators for each run. The table keeps rate, fuel moisture
    (TM for coke/nut coke; IM for PCI), ash-analysis volatile matter (VM), ash
    chemistry, and dry-fuel-basis S/P together so the slag calculation can
    apply the full fuel ash sequence.

    Args:
         - fuel_ash_cfg: list[dict[str, Any]] - Configured fuel ash defaults.

    Returns:
         - return pd.DataFrame - Editable fuel ash data for Streamlit.
    """

    rows = []
    for item in fuel_ash_cfg or []:
        mn_basis = str(item.get("mn_basis", "mno") or "mno").lower()
        ti_basis = str(item.get("ti_basis", "tio2") or "tio2").lower()
        rows.append(
            {
                "enabled": bool(item.get("enabled", True)),
                "fuel_id": str(item.get("fuel_id", "")),
                "fuel_name": str(item.get("display_name", item.get("fuel_id", ""))),
                "rate_kg_per_thm": float(item.get("rate_kg_per_thm", 0.0) or 0.0),
                "rate_basis": str(item.get("rate_basis", "wet") or "wet").lower(),
                "add_moisture_to_rate": bool(item.get("add_moisture_to_rate", False)),
                "price_rs_per_mt": float(item.get("price_rs_per_mt", 0.0) or 0.0),
                "moisture_pct": float(item.get("moisture_pct", 0.0) or 0.0),
                "vm_pct": float(item.get("vm_pct", 0.0) or 0.0),
                "ash_pct": float(item.get("ash_pct", 0.0) or 0.0),
                "sio2_pct": float(item.get("sio2_pct", 0.0) or 0.0),
                "al2o3_pct": float(item.get("al2o3_pct", 0.0) or 0.0),
                "cao_pct": float(item.get("cao_pct", 0.0) or 0.0),
                "mgo_pct": float(item.get("mgo_pct", 0.0) or 0.0),
                "fe2o3_pct": float(item.get("fe2o3_pct", 0.0) or 0.0),
                "mn_basis": mn_basis,
                "mno_pct": float(item.get("mn_pct" if mn_basis == "mn" else "mno_pct", 0.0) or 0.0),
                "ti_basis": ti_basis,
                "tio2_pct": float(item.get("ti_pct" if ti_basis == "ti" else "tio2_pct", 0.0) or 0.0),
                "alkali_pct": float(item.get("alkali_pct", 0.0) or 0.0),
                "na2o_pct": float(item.get("na2o_pct", 0.0) or 0.0),
                "k2o_pct": float(item.get("k2o_pct", 0.0) or 0.0),
                "s_pct": float(item.get("s_pct", 0.0) or 0.0),
                "p_pct": float(item.get("p_pct", 0.0) or 0.0),
                "chemistry_source": str(item.get("chemistry_source", "default") or "default"),
            }
        )
    return pd.DataFrame(rows)


FUEL_PRICE_LABELS = {
    "coke": "Coke",
    "nut_coke": "Nut coke",
    "pci": "PCI",
}


def apply_fuel_prices(editor_df: pd.DataFrame, prices_rs_per_mt: Mapping[str, float]) -> pd.DataFrame:
    """Return fuel inputs with only the configured prices updated."""

    out = editor_df.copy()
    if "fuel_id" not in out.columns or "price_rs_per_mt" not in out.columns:
        return out

    ids = out["fuel_id"].astype(str).str.strip().str.lower()
    for fuel_id, raw_price in prices_rs_per_mt.items():
        normalized_id = str(fuel_id).strip().lower()
        if normalized_id not in FUEL_PRICE_LABELS:
            continue
        try:
            price = max(0.0, float(raw_price))
        except (TypeError, ValueError):
            continue
        out.loc[ids == normalized_id, "price_rs_per_mt"] = price
    return out


def fuel_prices_from_editor(editor_df: pd.DataFrame) -> dict[str, float]:
    """Return current per-MT prices for the three operator-facing fuels."""

    prices = {fuel_id: float(ASSUMED_FUEL_PRICES_RS_PER_KG.get(fuel_id, 0.0)) * 1000.0 for fuel_id in FUEL_PRICE_LABELS}
    if editor_df.empty or "fuel_id" not in editor_df.columns or "price_rs_per_mt" not in editor_df.columns:
        return prices

    ids = editor_df["fuel_id"].astype(str).str.strip().str.lower()
    for fuel_id in FUEL_PRICE_LABELS:
        values = pd.to_numeric(editor_df.loc[ids == fuel_id, "price_rs_per_mt"], errors="coerce").dropna()
        if not values.empty:
            prices[fuel_id] = max(0.0, float(values.iloc[0]))
    return prices


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
        mn_basis = str(item.get("mn_basis", "mno") or "mno").lower()
        ti_basis = str(item.get("ti_basis", "tio2") or "tio2").lower()
        rows.append(
            {
                "enabled": bool(item.get("enabled", True)),
                "flux_id": str(item.get("flux_id", "")),
                "flux_name": str(item.get("display_name", item.get("flux_id", ""))),
                "optimizable": bool(item.get("optimizable", False)),
                "wet_qty_mt": float(item.get("wet_qty_mt", 0.0) or 0.0),
                "price_rs_per_mt": float(item.get("price_rs_per_mt", 0.0) or 0.0),
                "stock_mt": float(item.get("stock_mt", 0.0) or 0.0),
                "min_qty_mt": float(item.get("min_qty_mt", 0.0) or 0.0),
                "max_qty_mt": float(item.get("max_qty_mt", item.get("stock_mt", 0.0)) or 0.0),
                "moisture_pct": float(item.get("moisture_pct", 0.0) or 0.0),
                "sio2_pct": float(item.get("sio2_pct", 0.0) or 0.0),
                "al2o3_pct": float(item.get("al2o3_pct", 0.0) or 0.0),
                "cao_pct": float(item.get("cao_pct", 0.0) or 0.0),
                "mgo_pct": float(item.get("mgo_pct", 0.0) or 0.0),
                "fe2o3_pct": float(item.get("fe2o3_pct", 0.0) or 0.0),
                "mn_basis": mn_basis,
                "mno_pct": float(item.get("mn_pct" if mn_basis == "mn" else "mno_pct", 0.0) or 0.0),
                "ti_basis": ti_basis,
                "tio2_pct": float(item.get("ti_pct" if ti_basis == "ti" else "tio2_pct", 0.0) or 0.0),
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
                "rate_basis": str(item.get("rate_basis", "mt_per_day") or "mt_per_day").lower(),
                "wet_qty_mt": float(item.get("wet_qty_mt", 0.0) or 0.0),
                "quantity_kg_per_charge": float(item.get("quantity_kg_per_charge", 0.0) or 0.0),
                "source": str(item.get("source", "default") or "default"),
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


def float_from_row(row: pd.Series, key: str, default: float = 0.0) -> float:
    """
    Read one numeric value from a Streamlit editor row.

    Data-editor cells can return blank, NaN, or typed numeric values depending on
    the Streamlit version and user edits. This helper normalizes those cases so
    inputs always receive stable floats.

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


def fuel_ash_inputs_from_editor(editor_df: pd.DataFrame) -> list[FuelAshInput]:
    """
    Convert edited fuel-ash rows into typed fuel-ash inputs.

    Args:
         - editor_df: pd.DataFrame - Edited fuel-ash table.

    Returns:
         - return list[FuelAshInput] - Fuel-ash inputs for slag calculations.
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
                rate_kg_per_thm=float_from_row(row, "rate_kg_per_thm"),
                rate_basis=str(row.get("rate_basis", "wet") or "wet").lower(),
                add_moisture_to_rate=bool(row.get("add_moisture_to_rate", False)),
                price_rs_per_mt=float_from_row(row, "price_rs_per_mt"),
                moisture_pct=float_from_row(row, "moisture_pct"),
                vm_pct=float_from_row(row, "vm_pct"),
                ash_pct=float_from_row(row, "ash_pct"),
                sio2_pct=float_from_row(row, "sio2_pct"),
                al2o3_pct=float_from_row(row, "al2o3_pct"),
                cao_pct=float_from_row(row, "cao_pct"),
                mgo_pct=float_from_row(row, "mgo_pct"),
                fe2o3_pct=float_from_row(row, "fe2o3_pct"),
                mno_pct=oxide_pct_from_basis(
                    float_from_row(row, "mno_pct"),
                    str(row.get("mn_basis", "mno")),
                    element="mn",
                ),
                tio2_pct=oxide_pct_from_basis(
                    float_from_row(row, "tio2_pct"),
                    str(row.get("ti_basis", "tio2")),
                    element="ti",
                ),
                alkali_pct=float_from_row(row, "alkali_pct"),
                na2o_pct=float_from_row(row, "na2o_pct"),
                k2o_pct=float_from_row(row, "k2o_pct"),
                s_pct=float_from_row(row, "s_pct"),
                p_pct=float_from_row(row, "p_pct"),
                chemistry_source=str(row.get("chemistry_source", "default") or "default"),
            )
        )
    return fuel_ash_inputs


def dust_inputs_from_editor(editor_df: pd.DataFrame) -> list[DustInput]:
    """
    Convert edited BF gas dust rows into typed dust inputs.

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
                wet_qty_mt=float_from_row(row, "wet_qty_mt"),
                quantity_kg_per_charge=float_from_row(row, "quantity_kg_per_charge"),
                rate_basis=str(row.get("rate_basis", "mt_per_day") or "mt_per_day").lower(),
                source=str(row.get("source", "manual") or "manual"),
                moisture_pct=float_from_row(row, "moisture_pct"),
                sio2_pct=float_from_row(row, "sio2_pct"),
                al2o3_pct=float_from_row(row, "al2o3_pct"),
                cao_pct=float_from_row(row, "cao_pct"),
                mgo_pct=float_from_row(row, "mgo_pct"),
                fe_pct=float_from_row(row, "fe_pct"),
                mn_pct=float_from_row(row, "mn_pct"),
                p_pct=float_from_row(row, "p_pct"),
                s_pct=float_from_row(row, "s_pct"),
                ti_pct=float_from_row(row, "ti_pct"),
                zn_pct=float_from_row(row, "zn_pct"),
                na2o_pct=float_from_row(row, "na2o_pct"),
                k2o_pct=float_from_row(row, "k2o_pct"),
                caf2_pct=float_from_row(row, "caf2_pct"),
            )
        )
    return dust_inputs


def flux_inputs_from_editor(editor_df: pd.DataFrame) -> list[FluxInput]:
    """
    Convert edited fixed-flux rows into typed flux inputs.

    Flux rows are fixed burden additions: they reserve slag capacity before the
    ore optimizer searches for a feasible blend, and feed the full slag balance.

    Args:
         - editor_df: pd.DataFrame - Edited fixed-flux table.

    Returns:
         - return list[FluxInput] - Flux inputs for the slag balance / optimizer.
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
                wet_qty_mt=float_from_row(row, "wet_qty_mt"),
                moisture_pct=float_from_row(row, "moisture_pct"),
                sio2_pct=float_from_row(row, "sio2_pct"),
                al2o3_pct=float_from_row(row, "al2o3_pct"),
                cao_pct=float_from_row(row, "cao_pct"),
                mgo_pct=float_from_row(row, "mgo_pct"),
                fe2o3_pct=float_from_row(row, "fe2o3_pct"),
                mno_pct=oxide_pct_from_basis(
                    float_from_row(row, "mno_pct"),
                    str(row.get("mn_basis", "mno")),
                    element="mn",
                ),
                tio2_pct=oxide_pct_from_basis(
                    float_from_row(row, "tio2_pct"),
                    str(row.get("ti_basis", "tio2")),
                    element="ti",
                ),
                na2o_pct=float_from_row(row, "na2o_pct"),
                k2o_pct=float_from_row(row, "k2o_pct"),
                caf2_pct=float_from_row(row, "caf2_pct"),
                p_pct=float_from_row(row, "p_pct"),
                s_pct=float_from_row(row, "s_pct"),
                zn_pct=float_from_row(row, "zn_pct"),
                loi_pct=float_from_row(row, "loi_pct"),
                price_rs_per_mt=float_from_row(row, "price_rs_per_mt"),
                stock_mt=float_from_row(row, "stock_mt"),
                min_qty_mt=float_from_row(row, "min_qty_mt"),
                max_qty_mt=(
                    float_from_row(row, "max_qty_mt")
                    if "max_qty_mt" in row and not pd.isna(row.get("max_qty_mt"))
                    else None
                ),
                optimizable=bool(row.get("optimizable", False)),
            )
        )
    return flux_inputs


def slag_balance_settings_from_editor(
    settings_values: dict[str, Any],
    hm_chem_values: dict[str, float],
    hm_snapshot: dict[str, Any] | None = None,
) -> SlagBalanceSettings:
    """
    Convert edited slag-balance setting values into a typed settings object.

    PI chemistry (C/Si/S/Others) is sourced from the live HM analysis snapshot so
    the full slag balance subtracts the actual SiO2 consumed by Si reduction and
    the actual S reporting to pig iron. HM Mn% and Ti% come directly from the HM
    snapshot so Mn/Ti partitioning reflects observed chemistry. Recovery, gas
    loss, alkali split, and conversion factors remain editable.

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
        fe_to_pig_iron_fraction=float(settings_values.get("fe_to_pig_iron_fraction", 0.999)),
        mn_recovery_pct=float(settings_values.get("mn_recovery_pct", 60.0)),
        sulphur_gas_loss_pct=float(settings_values.get("sulphur_gas_loss_pct", 10.0)),
        alkali_to_slag_fraction=float(settings_values.get("alkali_to_slag_fraction", 0.8)),
        si_to_sio2_factor=float(settings_values.get("si_to_sio2_factor", 2.14)),
        fe_to_feo_factor=float(settings_values.get("fe_to_feo_factor", 72.0 / 56.0)),
        mn_to_mno_factor=float(settings_values.get("mn_to_mno_factor", 1.291)),
        slag_correction_factor=float(settings_values.get("slag_correction_factor", 1.0)),
        tio2_to_slag=bool(settings_values.get("tio2_to_slag", True)),
    )


def slag_balance_defaults(settings_cfg: Mapping[str, Any]) -> dict[str, float | bool]:
    """The Advanced Slag Balance inputs before an operator edits them (``render_slag_balance_settings``' defaults).

    Args:
         - settings_cfg: Mapping[str, Any] - Configured slag-balance defaults (``setting_bmo.yml → slag_balance``).

    Returns:
         - return dict[str, float | bool] - The values the page passes to ``slag_balance_settings_from_editor``.
    """

    return {
        "enabled": bool(settings_cfg.get("enabled", True)),
        "slag_correction_factor": 1.0,
        "pi_loss_pct": float(settings_cfg.get("pi_loss_pct", 0.2)),
        "fe_to_pig_iron_fraction": float(settings_cfg.get("fe_to_pig_iron_fraction", 0.999)),
        "mn_recovery_pct": float(settings_cfg.get("mn_recovery_pct", 60.0)),
        "sulphur_gas_loss_pct": float(settings_cfg.get("sulphur_gas_loss_pct", 10.0)),
        "alkali_to_slag_fraction": float(settings_cfg.get("alkali_to_slag_fraction", 0.8)),
        "fe_to_feo_factor": float(settings_cfg.get("fe_to_feo_factor", 72.0 / 56.0)),
        "mn_to_mno_factor": float(settings_cfg.get("mn_to_mno_factor", 1.291)),
    }
