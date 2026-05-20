"""Compute all shift metrics from raw DataFrames — pure Python, no I/O."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

import pandas as pd

from config.config_loader import load_config
from data.material_mapping import MaterialNameMapper
from reports.base import ReportBuilder
from reports.shift_report.data import (
    ParamStats,
    ShiftRawData,
    ShiftReportData,
    TempRow,
)

# ── Column name registry ────────────────────────────────────────────────────
# Keys are short internal names; values are the exact DataFrame column names
# produced by fetch_online_df() and fetch_offline_data().

_ONLINE: dict[str, str] = {
    # process_params
    "prod_rate": "Process Params - BF2_PRODUCTION TONNES PER HR",
    "charges_hr": "Process Params - BF2_CHARGES PER HR",
    "hb_vol": "Process Params - BF2_PROC Hot Blast Volume",
    "hb_temp": "Process Params - BF2_PROC Hot Blast Temp",
    "hb_press": "Process Params - BF2_PROC Hot Blast Pressure",
    "perm": "Process Params - BF2_BODY_PERMEABILITY",
    "etaco": "Process Params - BF2_BODY_ETACO",
    "raft": "Process Params - BF2_BODY_RAFT",
    "o2_flow": "Process Params - BF2_OXYGEN FLOW",
    "o2_enr": "Process Params - BF2_OXYGEN ENRICHMENT PCT",
    "fuel_rate": "Process Params - BF2_FUEL RATE PER THM",
    "coke_rate": "Process Params - BF2_COKE RATE PER THM",
    "nutcoke_rate": "Process Params - BF2_NUT COKE RATE PER THM",
    "pci_rate": "Process Params - BF2_COAL RATE PER THM",
    "runner_temp": "Process Params - TE_40532A Runner Temp PCI side near to Taphole",
    # Differential pressure
    "furnace_top_dp": "Process Params - BF2_BODY_TOP DP",
    "furnace_bottom_dp": "Process Params - BF2_BODY_BOTTOM DP",
    "furnace_total_dp": "Process Params - BF2_BODY_TOTAL DP",
    # temperature_profile
    "hearth_4_3_a": "Temperature Profile - BF2_BFBD Furnace Body 4373mm Temp A",
    "hearth_5_4_c": "Temperature Profile - BF2_BFBD Furnace Body 5411mm Temp C",
    "hearth_5_7_c": "Temperature Profile - BF2_BFBD Furnace Body 5757mm Temp C",
    "hearth_6_1_b": "Temperature Profile - BF2_BFBD Furnace Body 6103mm Temp B",
    "ls_q1": "Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp A",
    "ls_q2": "Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp B",
    "ls_q3": "Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp C",
    "ls_q4": "Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp D",
    "belly_q1": "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp A",
    "belly_q2": "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp B",
    "belly_q3": "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp C",
    "belly_q4": "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp D",
    "uptake_q1": "Process Params - BF2_PROC Top Temp 1",
    "uptake_q2": "Process Params - BF2_PROC Top Temp 2",
    "uptake_q3": "Process Params - BF2_PROC Top Temp 3",
    "uptake_q4": "Process Params - BF2_PROC Top Temp 4",
    "bosh_q1": "Temperature Profile - BF2_BFBD Furnace Body 12975mm Temp A",
    "bosh_q2": "Temperature Profile - BF2_BFBD Furnace Body 12975mm Temp B",
    "bosh_q3": "Temperature Profile - BF2_BFBD Furnace Body 12975mm Temp C",
    "bosh_q4": "Temperature Profile - BF2_BFBD Furnace Body 12975mm Temp D",
}

_HM: dict[str, str] = {
    "si": "chem_pct_si",
    "s": "chem_pct_s",
    "hm_temp": "hm_temp",
    "slag_cao": "slag_pct_cao",
    "slag_sio2": "slag_pct_sio2",
}

_SHIFT_REPORT_CONFIG: dict[str, Any] = load_config("shift_report.yml") or {}


def _list_map(key: str) -> dict[str, list[str]]:
    raw = _SHIFT_REPORT_CONFIG.get(key) or {}
    return {
        str(name): [str(value) for value in values or []]
        for name, values in raw.items()
    }


def _source_map(key: str) -> dict[str, dict[str, Any]]:
    raw = _SHIFT_REPORT_CONFIG.get(key) or {}
    return {str(name): dict(spec or {}) for name, spec in raw.items()}


_CHARGE_COLS = _list_map("charge_columns")
_CHARGE_FALLBACK_COLS = _list_map("charge_fallback_columns")
_MATERIAL_CODE_CANDIDATES = _list_map("material_code_candidates")
_MOISTURE_SOURCES = _source_map("moisture_sources")
_FINES_SOURCES = _source_map("fines_sources")


def _charge_columns(spec: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for group in spec.get("charge_groups", []):
        columns.extend(_CHARGE_COLS[str(group)])
    columns.extend(str(col) for col in spec.get("charge_columns", []))
    return list(dict.fromkeys(columns))


_FINES_CHARGE_COLS = tuple(
    dict.fromkeys(
        col for spec in _FINES_SOURCES.values() for col in _charge_columns(spec)
    )
)
_FINES_PATTERNS = {
    source: re.compile(str(spec.get("column_pattern", r"$^")), re.IGNORECASE)
    for source, spec in _FINES_SOURCES.items()
}
_FINES_SOURCE_BY_CHARGE_COL = {
    str(col): source
    for source, spec in _FINES_SOURCES.items()
    for col in _charge_columns(spec)
}
try:
    _MATERIAL_NAME_MAPPER = MaterialNameMapper.from_file()
except Exception:
    _MATERIAL_NAME_MAPPER = None

# ── Status thresholds ────────────────────────────────────────────────────────
_THRESH = dict(_SHIFT_REPORT_CONFIG.get("status_thresholds") or {})


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mean(df: pd.DataFrame, key: str) -> Optional[float]:
    col = _ONLINE.get(key, key)
    if df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return round(float(s.mean()), 2) if len(s) else None


def _std(df: pd.DataFrame, key: str) -> Optional[float]:
    col = _ONLINE.get(key, key)
    if df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return round(float(s.std()), 2) if len(s) >= 2 else None


def _ps(df: pd.DataFrame, key: str) -> ParamStats:
    return ParamStats(mean=_mean(df, key), std=_std(df, key))


def _hm_mean(df: pd.DataFrame, key: str) -> Optional[float]:
    col = _HM.get(key, key)
    if df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return round(float(s.mean()), 2) if len(s) else None


def _sum_charge_columns(df: pd.DataFrame, cols: list[str]) -> Optional[float]:
    if df.empty:
        return None
    present = [col for col in cols if col in df.columns]
    if not present:
        return None

    values = df[present].apply(pd.to_numeric, errors="coerce")
    if not values.notna().to_numpy().any():
        return None

    total = float(values.sum(skipna=True).sum())
    return round(total, 2)


def _charge_column_total(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    values = pd.to_numeric(df[col], errors="coerce")
    return 0.0 if values.notna().sum() == 0 else float(values.sum(skipna=True))


def _charge_sum(df: pd.DataFrame, key: str) -> Optional[float]:
    value = _sum_charge_columns(df, _CHARGE_COLS[key])
    if value is not None:
        return value

    for fallback_col in _CHARGE_FALLBACK_COLS.get(key, []):
        value = _sum_charge_columns(df, [fallback_col])
        if value is not None:
            return value
    return None


def _material_name_lookup(materials_df: pd.DataFrame) -> dict[str, str]:
    if materials_df.empty or not {"material_code", "material_name"}.issubset(
        materials_df.columns
    ):
        return {}

    df = materials_df.copy()
    if "is_active" in df.columns:
        df = df[df["is_active"].fillna(True).astype(bool)]

    lookup: dict[str, str] = {}
    for _, row in df.dropna(subset=["material_code", "material_name"]).iterrows():
        code = str(row["material_code"])
        name = str(row["material_name"])
        code_without_underscores = code.replace("_", "")
        for key in dict.fromkeys(
            [
                code,
                code.casefold(),
                code_without_underscores,
                code_without_underscores.casefold(),
            ]
        ):
            lookup.setdefault(key, name)
    return lookup


def _display_material_name(material_name: str) -> str:
    if _MATERIAL_NAME_MAPPER is None:
        return material_name
    return _MATERIAL_NAME_MAPPER.primary_client_name_for_material(material_name)


def _display_material_code(material_code: str) -> str:
    return material_code.replace("_", "")


def _used_charge_materials(
    charge_df: pd.DataFrame,
    materials_df: pd.DataFrame,
) -> dict[str, str]:
    material_names = _material_name_lookup(materials_df)
    used: dict[str, str] = {}

    for label, key in (("Flux", "flux"), ("Ore", "ore")):
        items: list[str] = []
        seen: set[str] = set()
        for charge_col in _CHARGE_COLS.get(key, []):
            if _charge_column_total(charge_df, charge_col) <= 0:
                continue
            material_codes = _material_candidates(charge_col)
            material_name = next(
                (
                    material_names[material_code]
                    for material_code in material_codes
                    if material_code in material_names
                ),
                None,
            )
            if material_name:
                item = _display_material_name(material_name)
            else:
                item = _display_material_code(material_codes[0])
            if item not in seen:
                items.append(item)
                seen.add(item)
        if items:
            used[label] = ", ".join(items)

    return used


def _analysis_with_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df.copy()
    if "time" not in out.columns:
        for col in ("date_time", "index"):
            if col in out.columns:
                out = out.rename(columns={col: "time"})
                break
    if "time" not in out.columns:
        return pd.DataFrame()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    return out.dropna(subset=["time"]).sort_values("time")


def _material_candidates(charge_col: str) -> list[str]:
    base = charge_col.removesuffix("_mt")
    return _MATERIAL_CODE_CANDIDATES.get(
        charge_col,
        list(dict.fromkeys([base, base.replace("_", "")])),
    )


def _analysis_rows(
    df: pd.DataFrame,
    charge_col: str,
    *,
    end_time,
) -> pd.DataFrame:
    if df.empty or "material_code" not in df.columns:
        return pd.DataFrame()
    end_ts = pd.Timestamp(end_time)
    end_ts = (
        end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    )
    codes = _material_candidates(charge_col)
    rows = df[df["material_code"].astype(str).isin(codes) & (df["time"] <= end_ts)]
    return rows.sort_values("time", ascending=False)


def _latest_value(
    df: pd.DataFrame,
    charge_col: str,
    columns: list[str],
    *,
    end_time,
) -> Optional[float]:
    rows = _analysis_rows(df, charge_col, end_time=end_time)
    present = [col for col in columns if col in rows.columns]
    for _, row in rows.iterrows():
        for value in pd.to_numeric(row[present], errors="coerce"):
            if pd.notna(value):
                return float(value)
    return None


def _fines_source(charge_col: str) -> Optional[str]:
    return _FINES_SOURCE_BY_CHARGE_COL.get(charge_col)


def _fines_columns(df: pd.DataFrame, charge_col: str) -> list[str]:
    pattern = _FINES_PATTERNS.get(_fines_source(charge_col) or "")
    if pattern is None:
        return []
    cols = [str(col) for col in df.columns]
    return [col for col in cols if pattern.fullmatch(col)]


def _latest_sum(
    df: pd.DataFrame,
    charge_col: str,
    columns: list[str],
    *,
    end_time,
) -> Optional[float]:
    rows = _analysis_rows(df, charge_col, end_time=end_time)
    if rows.empty or not columns:
        return None
    for _, row in rows.iterrows():
        values = pd.to_numeric(row[columns], errors="coerce")
        if values.notna().sum():
            return float(values.sum(skipna=True))
    return None


def _kg_per_thm(
    percent_mt_sum: float, production_t: Optional[float]
) -> Optional[float]:
    if not production_t:
        return None
    return round((percent_mt_sum * 10.0) / production_t, 2)


def _burden_moisture_input(
    charge_df: pd.DataFrame,
    *,
    production_t: Optional[float],
    end_time,
    fuel_df: pd.DataFrame,
    ore_df: pd.DataFrame,
    flux_df: pd.DataFrame,
) -> Optional[float]:
    analyses = {
        "fuel": _analysis_with_time(fuel_df),
        "ore": _analysis_with_time(ore_df),
        "flux": _analysis_with_time(flux_df),
    }
    total = 0.0
    used = False
    for source, spec in _MOISTURE_SOURCES.items():
        analysis = analyses.get(source, pd.DataFrame())
        pct_cols = [str(col) for col in spec.get("analysis_columns", [])]
        for charge_col in _charge_columns(spec):
            mt = _charge_column_total(charge_df, charge_col)
            if mt <= 0:
                continue
            pct = _latest_value(analysis, charge_col, pct_cols, end_time=end_time)
            if pct is None:
                continue
            total += pct * mt
            used = True
    return _kg_per_thm(total, production_t) if used else None


def _fines_input(
    charge_df: pd.DataFrame,
    fines_df: pd.DataFrame,
    *,
    production_t: Optional[float],
    end_time,
) -> Optional[float]:
    fines = _analysis_with_time(fines_df)
    total = 0.0
    used = False
    for charge_col in _FINES_CHARGE_COLS:
        mt = _charge_column_total(charge_df, charge_col)
        if mt <= 0:
            continue
        pct = _latest_sum(
            fines,
            charge_col,
            _fines_columns(fines, charge_col),
            end_time=end_time,
        )
        if pct is None:
            continue
        total += pct * mt
        used = True
    return _kg_per_thm(total, production_t) if used else None


def _temp_row(df: pd.DataFrame, q1: str, q2: str, q3: str, q4: str) -> TempRow:
    return TempRow(
        q1=_mean(df, q1),
        q2=_mean(df, q2),
        q3=_mean(df, q3),
        q4=_mean(df, q4),
    )


def _status(
    etaco: Optional[float],
    fuel_rate: Optional[float],
    raft: Optional[float],
    perm: Optional[float],
) -> tuple[Literal["STABLE", "ATTENTION REQUIRED", "UNSTABLE"], list[str]]:
    flags: list[str] = []

    if etaco is not None:
        if etaco < _THRESH["etaco"]["crit"]:
            flags.append(f"ETA CO {etaco:.2f}% (critical < 40%)")
        elif etaco < _THRESH["etaco"]["warn"]:
            flags.append(f"ETA CO {etaco:.2f}% (< 42% normal)")

    if fuel_rate is not None:
        if fuel_rate > _THRESH["fuel_rate"]["crit"]:
            flags.append(f"Fuel Rate {fuel_rate:.2f} kg/tHM (critical > 570)")
        elif fuel_rate > _THRESH["fuel_rate"]["warn"]:
            flags.append(f"Fuel Rate {fuel_rate:.2f} kg/tHM (> 530 attention)")

    if raft is not None and (raft < _THRESH["raft_lo"] or raft > _THRESH["raft_hi"]):
        flags.append(f"RAFT {raft:.2f}°C (normal 2100–2350)")

    if perm is not None and (perm < _THRESH["perm_lo"] or perm > _THRESH["perm_hi"]):
        flags.append(f"Permeability {perm:.2f} (normal 1000–1600)")

    n = len(flags)
    if n == 0:
        return "STABLE", flags
    if n <= 2:
        return "ATTENTION REQUIRED", flags
    return "UNSTABLE", flags


# ── Builder ──────────────────────────────────────────────────────────────────


class ShiftBuilder(ReportBuilder[ShiftRawData, ShiftReportData]):
    def build(self, raw: ShiftRawData) -> ShiftReportData:  # type: ignore[override]
        df = raw.online_df
        #
        hm = raw.hm_slag_df
        ch = raw.charge_df

        # One row in offline_feed.charge_data represents one charge in the shift.
        total_charges = int(ch.shape[0]) if not ch.empty else 0
        prod_rate = _mean(df, "prod_rate")
        theoretical_production = round(prod_rate * 8, 2) if prod_rate else None

        # Slag basicity
        cao = _hm_mean(hm, "slag_cao")
        sio2 = _hm_mean(hm, "slag_sio2")
        slag_basicity = round(cao / sio2, 2) if cao and sio2 else None

        # HM temp: use from hot_metal_slag_data reports
        hm_temp = _hm_mean(hm, "hmt_gt_1480c")

        fuel_rate_val = _mean(df, "fuel_rate")
        raft_val = _mean(df, "raft")
        perm_val = _mean(df, "perm")
        etaco_val = _mean(df, "etaco")
        burden_moisture_input = _burden_moisture_input(
            ch,
            production_t=theoretical_production,
            end_time=raw.shift_end_ist,
            fuel_df=raw.fuel_chemistry_df,
            ore_df=raw.ore_chemistry_df,
            flux_df=raw.flux_chemistry_df,
        )
        fines_input = _fines_input(
            ch,
            raw.material_fines_df,
            production_t=theoretical_production,
            end_time=raw.shift_end_ist,
        )

        status, flags = _status(etaco_val, fuel_rate_val, raft_val, perm_val)

        return ShiftReportData(
            shift_date=raw.shift_date,
            shift_label=raw.shift_label,
            shift_start_ist=raw.shift_start_ist,
            shift_end_ist=raw.shift_end_ist,
            status=status,
            status_flags=flags,
            production_rate=prod_rate,
            theoretical_production=theoretical_production,
            total_charges=total_charges,
            coke_t=_charge_sum(ch, "coke"),
            nut_coke_t=_charge_sum(ch, "nut_coke"),
            sinter_t=_charge_sum(ch, "sinter"),
            ore_t=_charge_sum(ch, "ore"),
            pellet_t=_charge_sum(ch, "pellet"),
            flux_t=_charge_sum(ch, "flux"),
            fuel_rate=fuel_rate_val,
            coke_rate=_mean(df, "coke_rate"),
            nut_coke_rate=_mean(df, "nutcoke_rate"),
            pci_rate=_mean(df, "pci_rate"),
            hm_si=_hm_mean(hm, "si"),
            hm_s=_hm_mean(hm, "s"),
            hm_temp=hm_temp,
            slag_basicity=slag_basicity,
            total_taps=int(hm.dropna(how="all").shape[0]) if not hm.empty else None,
            blast_volume=_ps(df, "hb_vol"),
            blast_temp=_ps(df, "hb_temp"),
            blast_pressure=_ps(df, "hb_press"),
            furnace_top_dp=_ps(df, "furnace_top_dp"),
            furnace_bottom_dp=_ps(df, "furnace_bottom_dp"),
            furnace_total_dp=_ps(df, "furnace_total_dp"),
            o2_flow=_ps(df, "o2_flow"),
            o2_enrichment=_ps(df, "o2_enr"),
            permeability=_ps(df, "perm"),
            etaco=_ps(df, "etaco"),
            raft=_ps(df, "raft"),
            uptake=_temp_row(df, "uptake_q1", "uptake_q2", "uptake_q3", "uptake_q4"),
            lower_stack=_temp_row(df, "ls_q1", "ls_q2", "ls_q3", "ls_q4"),
            belly=_temp_row(df, "belly_q1", "belly_q2", "belly_q3", "belly_q4"),
            bosh=_temp_row(df, "bosh_q1", "bosh_q2", "bosh_q3", "bosh_q4"),
            hearth_4_3_a=_mean(df, "hearth_4_3_a"),
            hearth_5_4_c=_mean(df, "hearth_5_4_c"),
            hearth_5_7_c=_mean(df, "hearth_5_7_c"),
            hearth_6_1_b=_mean(df, "hearth_6_1_b"),
            burden_moisture_input=burden_moisture_input,
            fines_input=fines_input,
            used_materials=_used_charge_materials(ch, raw.materials_df),
        )
