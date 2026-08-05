"""Convert Streamlit editor dataframes into typed BMO input records.

These are the inverse of the ``build_*_editor_df`` builders in
:mod:`ui.bmo.components`: the builders turn config into editable tables, and
these helpers turn the edited tables back into the dataclass records consumed by
LP, DE, and the slag balance. They are pure (pandas + dataclasses, no Streamlit),
so the page stays thin and the conversion logic is unit-testable in isolation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.bmo.types import (
    DustInput,
    FluxInput,
    FuelAshInput,
    SlagBalanceSettings,
)


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
                price_rs_per_mt=float_from_row(row, "price_rs_per_mt"),
                # Fuel IM was historically stored under ``moisture_pct``.
                # Prefer the correctly named editor column while retaining a
                # migration fallback for old/session-restored dataframes.
                moisture_pct=float_from_row(
                    row,
                    "im_pct",
                    float_from_row(row, "moisture_pct"),
                ),
                vm_pct=float_from_row(row, "vm_pct"),
                ash_pct=float_from_row(row, "ash_pct"),
                sio2_pct=float_from_row(row, "sio2_pct"),
                al2o3_pct=float_from_row(row, "al2o3_pct"),
                cao_pct=float_from_row(row, "cao_pct"),
                mgo_pct=float_from_row(row, "mgo_pct"),
                fe2o3_pct=float_from_row(row, "fe2o3_pct"),
                tio2_pct=float_from_row(row, "tio2_pct"),
                na2o_pct=float_from_row(row, "na2o_pct"),
                k2o_pct=float_from_row(row, "k2o_pct"),
                s_pct=float_from_row(row, "s_pct"),
                p_pct=float_from_row(row, "p_pct"),
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
                mno_pct=float_from_row(row, "mno_pct"),
                tio2_pct=float_from_row(row, "tio2_pct"),
                na2o_pct=float_from_row(row, "na2o_pct"),
                k2o_pct=float_from_row(row, "k2o_pct"),
                caf2_pct=float_from_row(row, "caf2_pct"),
                p_pct=float_from_row(row, "p_pct"),
                s_pct=float_from_row(row, "s_pct"),
                zn_pct=float_from_row(row, "zn_pct"),
                loi_pct=float_from_row(row, "loi_pct"),
                price_rs_per_mt=float_from_row(row, "price_rs_per_mt"),
                stock_mt=float_from_row(row, "stock_mt"),
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
