"""Persistence helpers for BMO operator defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PERSISTED_NUMERIC_COLUMNS = (
    "price_rs_per_mt",
    "min_share_pct",
    "max_share_pct",
)

PERSISTED_MODEL_INPUT_COLUMNS = (
    "target_slag_basicity_min",
    "target_slag_basicity_max",
    "target_slag_t_basicity_min",
    "target_slag_t_basicity_max",
    "target_slag_rate_kg_per_thm",
    "target_slag_al2o3_max_pct",
    "target_slag_mgo_min_pct",
    "target_slag_mgo_al2o3_ratio_min",
    # Charging plant. These move with skip-car condition and burden bulk density,
    # so the operator sets them rather than editing yml. Charging hours are always
    # 24 and nut-coke tonnage is derived from its rate, so neither is stored.
    "max_charges_per_hour",
    "charge_mass_mt",
)

# Flux price/stock are operator inputs (like ore price/bounds), so they persist
# across sessions. Chemistry and the optimizable flag stay driven by config.
PERSISTED_FLUX_COLUMNS = (
    "price_rs_per_mt",
    "stock_mt",
)

# Every editable Fuel Ash value is an operator input. Persist the complete row
# by stable fuel id so saved values remain aligned when config or recent-rate
# defaults are refreshed.
PERSISTED_FUEL_ASH_NUMERIC_COLUMNS = (
    "rate_kg_per_thm",
    "price_rs_per_mt",
    "moisture_pct",
    "vm_pct",
    "ash_pct",
    "sio2_pct",
    "al2o3_pct",
    "cao_pct",
    "mgo_pct",
    "fe2o3_pct",
    "mno_pct",
    "tio2_pct",
    "alkali_pct",
    "na2o_pct",
    "k2o_pct",
    "s_pct",
    "p_pct",
)

PERSISTED_FUEL_ASH_TEXT_COLUMNS = ("rate_basis", "mn_basis", "ti_basis")

PERSISTED_DUST_NUMERIC_COLUMNS = (
    "wet_qty_mt",
    "quantity_kg_per_charge",
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

PERSISTED_DUST_TEXT_COLUMNS = ("rate_basis",)


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_ore_editor_preferences(editor_df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact, disk-safe preference payload from the ore editor."""

    if editor_df.empty or "ore_id" not in editor_df.columns:
        return {"ore_editor": {"selected_ore_ids": [], "rows": {}}}

    selected_ids: list[str] = []
    rows: dict[str, dict[str, float]] = {}
    for _, row in editor_df.iterrows():
        ore_id = str(row.get("ore_id", "")).strip()
        if not ore_id:
            continue
        if bool(row.get("selected", False)):
            selected_ids.append(ore_id)
        saved_row: dict[str, float] = {}
        for column in PERSISTED_NUMERIC_COLUMNS:
            if column not in row:
                continue
            value = _float_or_none(row.get(column))
            if value is not None:
                saved_row[column] = value
        rows[ore_id] = saved_row

    return {"ore_editor": {"selected_ore_ids": selected_ids, "rows": rows}}


# Slag-side pig-iron chemistry. Held at the operating point rather than tracking
# the latest cast, so it is an operator setting that must survive a session.
PERSISTED_HM_CHEMISTRY_COLUMNS = (
    "carbon_pct",
    "silicon_pct",
    "sulphur_pct",
    "other_pct",
)


def build_hm_chemistry_preferences(values: dict[str, Any]) -> dict[str, Any]:
    """Build a preference payload for the held slag-side HM chemistry."""

    saved: dict[str, float] = {}
    for key in PERSISTED_HM_CHEMISTRY_COLUMNS:
        value = _float_or_none(values.get(key))
        if value is not None:
            saved[key] = value
    return {"hot_metal_chemistry": saved}


def apply_hm_chemistry_preferences(
    defaults: dict[str, Any], preferences: dict[str, Any]
) -> dict[str, Any]:
    """Overlay saved slag-side HM chemistry onto the yml ``slag_balance`` block.

    Only the four PI-chemistry keys are overlaid; every other slag_balance
    setting (recovery, gas loss, conversion factors) stays config-driven.
    """

    out = dict(defaults)
    saved = preferences.get("hot_metal_chemistry", {}) if preferences else {}
    for key in PERSISTED_HM_CHEMISTRY_COLUMNS:
        if key in saved:
            value = _float_or_none(saved.get(key))
            if value is not None:
                out[key] = value
    return out


def save_hm_chemistry_preferences(path: str | Path, values: dict[str, Any]) -> Path:
    """Persist the held slag-side HM chemistry and return the written path."""

    pref_path = Path(path)
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    payload = load_ore_editor_preferences(pref_path)
    payload.update(build_hm_chemistry_preferences(values))
    with open(pref_path, "w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return pref_path


def build_model_input_preferences(values: dict[str, Any]) -> dict[str, Any]:
    """Build a compact preference payload for BMO model inputs."""

    saved: dict[str, float] = {}
    for key in PERSISTED_MODEL_INPUT_COLUMNS:
        value = _float_or_none(values.get(key))
        if value is not None:
            saved[key] = value
    return {"model_inputs": saved}


def apply_model_input_preferences(
    defaults: dict[str, float], preferences: dict[str, Any]
) -> dict[str, float]:
    """Apply saved model input preferences over computed/default values."""

    out = dict(defaults)
    saved = preferences.get("model_inputs", {}) if preferences else {}
    for key in PERSISTED_MODEL_INPUT_COLUMNS:
        if key in saved:
            value = _float_or_none(saved.get(key))
            if value is not None:
                out[key] = value
    return out


def apply_ore_editor_preferences(
    editor_df: pd.DataFrame, preferences: dict[str, Any]
) -> pd.DataFrame:
    """Apply saved ore editor preferences to a freshly built editor frame."""

    if editor_df.empty or "ore_id" not in editor_df.columns:
        return editor_df

    ore_editor = preferences.get("ore_editor", {}) if preferences else {}
    saved_rows = ore_editor.get("rows", {}) or {}
    selected_ids = {str(ore_id) for ore_id in ore_editor.get("selected_ore_ids", [])}
    out = editor_df.copy()

    if selected_ids and "selected" in out.columns:
        out["selected"] = out["ore_id"].astype(str).isin(selected_ids)

    for index, row in out.iterrows():
        ore_id = str(row.get("ore_id", "")).strip()
        saved = saved_rows.get(ore_id, {}) or {}
        for column in PERSISTED_NUMERIC_COLUMNS:
            if column in out.columns and column in saved:
                value = _float_or_none(saved[column])
                if value is not None:
                    out.at[index, column] = value
    return out


def build_flux_preferences(flux_df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact preference payload of flux price/stock keyed by flux id."""

    if flux_df.empty or "flux_id" not in flux_df.columns:
        return {"flux_editor": {"rows": {}}}

    rows: dict[str, dict[str, float]] = {}
    for _, row in flux_df.iterrows():
        flux_id = str(row.get("flux_id", "")).strip()
        if not flux_id:
            continue
        saved_row: dict[str, float] = {}
        for column in PERSISTED_FLUX_COLUMNS:
            if column not in row:
                continue
            value = _float_or_none(row.get(column))
            if value is not None:
                saved_row[column] = value
        rows[flux_id] = saved_row
    return {"flux_editor": {"rows": rows}}


def apply_flux_preferences(
    flux_df: pd.DataFrame, preferences: dict[str, Any]
) -> pd.DataFrame:
    """Apply saved flux price/stock over a freshly built flux editor frame."""

    if flux_df.empty or "flux_id" not in flux_df.columns:
        return flux_df

    saved_rows = (preferences.get("flux_editor", {}) if preferences else {}).get(
        "rows", {}
    ) or {}
    out = flux_df.copy()
    for index, row in out.iterrows():
        flux_id = str(row.get("flux_id", "")).strip()
        saved = saved_rows.get(flux_id, {}) or {}
        for column in PERSISTED_FLUX_COLUMNS:
            if column in out.columns and column in saved:
                value = _float_or_none(saved[column])
                if value is not None:
                    out.at[index, column] = value
    return out


def build_fuel_ash_preferences(editor_df: pd.DataFrame) -> dict[str, Any]:
    """Build persisted Fuel Ash rows keyed by stable fuel id."""

    if editor_df.empty or "fuel_id" not in editor_df.columns:
        return {"fuel_ash_editor": {"rows": {}}}

    rows: dict[str, dict[str, Any]] = {}
    for _, row in editor_df.iterrows():
        fuel_id = str(row.get("fuel_id", "")).strip()
        if not fuel_id:
            continue
        saved_row: dict[str, Any] = {"enabled": bool(row.get("enabled", True))}
        for column in PERSISTED_FUEL_ASH_NUMERIC_COLUMNS:
            if column not in row:
                continue
            value = _float_or_none(row.get(column))
            if value is not None:
                saved_row[column] = value
        for column in PERSISTED_FUEL_ASH_TEXT_COLUMNS:
            if column in row and str(row.get(column, "")).strip():
                saved_row[column] = str(row[column]).strip().lower()
        saved_row["chemistry_source"] = "manual"
        rows[fuel_id] = saved_row
    return {"fuel_ash_editor": {"rows": rows}}


def apply_fuel_ash_preferences(
    fuel_ash_df: pd.DataFrame, preferences: dict[str, Any]
) -> pd.DataFrame:
    """Apply saved Fuel Ash inputs over freshly built configuration rows."""

    if fuel_ash_df.empty or "fuel_id" not in fuel_ash_df.columns:
        return fuel_ash_df

    saved_rows = (preferences.get("fuel_ash_editor", {}) if preferences else {}).get(
        "rows", {}
    ) or {}
    out = fuel_ash_df.copy()
    for index, row in out.iterrows():
        fuel_id = str(row.get("fuel_id", "")).strip()
        saved = saved_rows.get(fuel_id, {}) or {}
        if "enabled" in saved and "enabled" in out.columns:
            out.at[index, "enabled"] = bool(saved["enabled"])
        for column in PERSISTED_FUEL_ASH_NUMERIC_COLUMNS:
            if column in out.columns and column in saved:
                value = _float_or_none(saved[column])
                if value is not None:
                    out.at[index, column] = value
        for column in PERSISTED_FUEL_ASH_TEXT_COLUMNS:
            if column in out.columns and column in saved:
                out.at[index, column] = str(saved[column])
        if "chemistry_source" in out.columns and "chemistry_source" in saved:
            out.at[index, "chemistry_source"] = str(saved["chemistry_source"])
    return out


def build_dust_preferences(editor_df: pd.DataFrame) -> dict[str, Any]:
    """Build persisted BF Gas Dust rows keyed by stable dust id."""

    if editor_df.empty or "dust_id" not in editor_df.columns:
        return {"dust_editor": {"rows": {}}}

    rows: dict[str, dict[str, Any]] = {}
    for _, row in editor_df.iterrows():
        dust_id = str(row.get("dust_id", "")).strip()
        if not dust_id:
            continue
        saved_row: dict[str, Any] = {"enabled": bool(row.get("enabled", True))}
        for column in PERSISTED_DUST_NUMERIC_COLUMNS:
            if column not in row:
                continue
            value = _float_or_none(row.get(column))
            if value is not None:
                saved_row[column] = value
        for column in PERSISTED_DUST_TEXT_COLUMNS:
            if column in row and str(row.get(column, "")).strip():
                saved_row[column] = str(row[column]).strip().lower()
        saved_row["source"] = "manual"
        rows[dust_id] = saved_row
    return {"dust_editor": {"rows": rows}}


def apply_dust_preferences(
    dust_df: pd.DataFrame, preferences: dict[str, Any]
) -> pd.DataFrame:
    """Apply saved BF Gas Dust inputs over fresh configuration rows."""

    if dust_df.empty or "dust_id" not in dust_df.columns:
        return dust_df

    saved_rows = (preferences.get("dust_editor", {}) if preferences else {}).get(
        "rows", {}
    ) or {}
    out = dust_df.copy()
    for index, row in out.iterrows():
        dust_id = str(row.get("dust_id", "")).strip()
        saved = saved_rows.get(dust_id, {}) or {}
        if "enabled" in saved and "enabled" in out.columns:
            out.at[index, "enabled"] = bool(saved["enabled"])
        for column in PERSISTED_DUST_NUMERIC_COLUMNS:
            if column in out.columns and column in saved:
                value = _float_or_none(saved[column])
                if value is not None:
                    out.at[index, column] = value
        for column in PERSISTED_DUST_TEXT_COLUMNS:
            if column in out.columns and column in saved:
                out.at[index, column] = str(saved[column])
        if "source" in out.columns and "source" in saved:
            out.at[index, "source"] = str(saved["source"])
    return out


def load_ore_editor_preferences(path: str | Path) -> dict[str, Any]:
    """Load saved BMO ore editor preferences, returning empty prefs if absent."""

    pref_path = Path(path)
    if not pref_path.exists():
        return {}
    with open(pref_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def save_ore_editor_preferences(path: str | Path, editor_df: pd.DataFrame) -> Path:
    """Persist BMO ore editor preferences and return the written path."""

    pref_path = Path(path)
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    payload = load_ore_editor_preferences(pref_path)
    payload.update(build_ore_editor_preferences(editor_df))
    with open(pref_path, "w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return pref_path


def save_flux_preferences(path: str | Path, flux_df: pd.DataFrame) -> Path:
    """Persist flux price/stock preferences and return the written path."""

    pref_path = Path(path)
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    payload = load_ore_editor_preferences(pref_path)
    payload.update(build_flux_preferences(flux_df))
    with open(pref_path, "w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return pref_path


def save_fuel_ash_preferences(path: str | Path, fuel_ash_df: pd.DataFrame) -> Path:
    """Persist all editable Fuel Ash inputs and return the written path."""

    pref_path = Path(path)
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    payload = load_ore_editor_preferences(pref_path)
    payload.update(build_fuel_ash_preferences(fuel_ash_df))
    with open(pref_path, "w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return pref_path


def save_dust_preferences(path: str | Path, dust_df: pd.DataFrame) -> Path:
    """Persist all editable BF Gas Dust inputs and return the written path."""

    pref_path = Path(path)
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    payload = load_ore_editor_preferences(pref_path)
    payload.update(build_dust_preferences(dust_df))
    with open(pref_path, "w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return pref_path


def save_model_input_preferences(path: str | Path, values: dict[str, Any]) -> Path:
    """Persist BMO model input preferences and return the written path."""

    pref_path = Path(path)
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    payload = load_ore_editor_preferences(pref_path)
    payload.update(build_model_input_preferences(values))
    with open(pref_path, "w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return pref_path
