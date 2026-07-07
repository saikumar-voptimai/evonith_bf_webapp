"""Compute all shift metrics from raw DataFrames; pure Python, no I/O."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

import pandas as pd

from apps.frontend_streamlit.config.config_loader import load_config
from furnace_data.influx.query import online_column_aliases
from apps.frontend_streamlit.reports.base import ReportBuilder
from apps.frontend_streamlit.reports.furnace_report.data import (
    ParamStats,
    ShiftRawData,
    ShiftReportData,
    TempRow,
)
from apps.frontend_streamlit.reports.furnace_report.materials import (
    MaterialAliasResolver,
    material_code_candidates,
)

# Online report metrics use canonical InfluxDB field ids from setting_ds_dv.yml.
# _series() also accepts legacy display-prefixed columns during migration.

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
# Status thresholds.
_THRESH = dict(_SHIFT_REPORT_CONFIG.get("status_thresholds") or {})


# Helpers.


def _series(df: pd.DataFrame, key: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")
    for col in online_column_aliases(key):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").dropna()
    return pd.Series(dtype="float64")


def _mean(df: pd.DataFrame, key: str) -> Optional[float]:
    s = _series(df, key)
    return round(float(s.mean()), 2) if len(s) else None


def _std(df: pd.DataFrame, key: str) -> Optional[float]:
    s = _series(df, key)
    return round(float(s.std()), 2) if len(s) >= 2 else None


def _ps(df: pd.DataFrame, key: str) -> ParamStats:
    return ParamStats(mean=_mean(df, key), std=_std(df, key))


def _dry_material_sum(
    charge_df: pd.DataFrame,
    analysis: pd.DataFrame,
    key: str,
    *,
    end_time,
) -> Optional[float]:
    total = 0.0
    for charge_col in _CHARGE_COLS[key]:
        mt = _charge_column_total(charge_df, charge_col)
        if mt <= 0:
            continue
        tm = _latest_value(analysis, charge_col, ["tm"], end_time=end_time)
        if tm is None:
            return None
        total += mt * max(0.0, 100.0 - tm) / 100.0
    return total


def _sum_required(*values: Optional[float]) -> Optional[float]:
    if any(value is None for value in values):
        return None
    return round(sum(value for value in values if value is not None), 2)


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


def _format_ratio_part(value: float) -> str:
    return f"{value:.1f}"


def calculate_burden_ratio(
    ore_t: Optional[float],
    sinter_t: Optional[float],
    pellet_t: Optional[float],
) -> Optional[str]:
    if all(value is None for value in (ore_t, sinter_t, pellet_t)):
        return None

    numeric_values = [float(value or 0.0) for value in (sinter_t, ore_t, pellet_t)]
    total = sum(numeric_values)
    if total <= 0:
        return None

    return ": ".join(
        _format_ratio_part((value / total) * 100)
        for value in numeric_values
    )


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


def _used_charge_materials(
    charge_df: pd.DataFrame,
    resolver: MaterialAliasResolver,
) -> dict[str, str]:
    used: dict[str, str] = {}

    for label, key in (("Flux", "flux"), ("Ore", "ore")):
        totals: dict[str, float] = {}
        for charge_col in _CHARGE_COLS.get(key, []):
            mt = _charge_column_total(charge_df, charge_col)
            if mt <= 0:
                continue
            item = resolver.display_name(_material_candidates(charge_col))
            totals[item] = totals.get(item, 0.0) + mt
        if totals:
            if key == "ore":
                burden_total_mt = sum(
                    float(value or 0.0)
                    for value in (
                        _charge_sum(charge_df, "sinter"),
                        _charge_sum(charge_df, "ore"),
                        _charge_sum(charge_df, "pellet"),
                    )
                )
                used[label] = "\n".join(
                    (
                        f"- {item} - {mt:.2f} mt - "
                        f"{mt / burden_total_mt * 100:.2f}% in burden"
                    )
                    for item, mt in totals.items()
                )
            else:
                used[label] = "\n".join(
                    f"- {item} - {mt:.2f} mt" for item, mt in totals.items()
                )

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
    return material_code_candidates(charge_col, _MATERIAL_CODE_CANDIDATES)


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


def _source_percent_mt_sum(
    charge_df: pd.DataFrame,
    analysis: pd.DataFrame,
    spec: dict[str, Any],
    *,
    end_time,
) -> tuple[float, bool]:
    total = 0.0
    used = False
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
    return total, used


def calculate_ibrm(
    charge_df: pd.DataFrame,
    *,
    production_t: Optional[float],
    end_time,
    ore_df: pd.DataFrame,
    sinter_t: Optional[float],
) -> Optional[float]:
    if not production_t or sinter_t is None:
        return None

    analysis = _analysis_with_time(ore_df)
    dry_ore_t = _dry_material_sum(charge_df, analysis, "ore", end_time=end_time)
    dry_pellet_t = _dry_material_sum(charge_df, analysis, "pellet", end_time=end_time)
    if dry_ore_t is None or dry_pellet_t is None:
        return None
    return round((dry_ore_t + dry_pellet_t + sinter_t) / production_t, 2)


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
        source_total, source_used = _source_percent_mt_sum(
            charge_df,
            analysis,
            spec,
            end_time=end_time,
        )
        total += source_total
        used = used or source_used
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


# Builder.


class ShiftBuilder(ReportBuilder[ShiftRawData, ShiftReportData]):
    def build(self, raw: ShiftRawData) -> ShiftReportData:  # type: ignore[override]
        df = raw.online_df
        hm = raw.hm_slag_df
        ch = raw.charge_df
        material_resolver = MaterialAliasResolver.from_dataframe(raw.materials_df)

        # One row in offline_feed.charge_data represents one charge in the shift.
        total_charges = int(ch.shape[0]) if not ch.empty else 0
        prod_rate = _mean(df, "production_per_hour")
        theoretical_production = (
            round(prod_rate * raw.duration_hours, 2) if prod_rate else None
        )
        coke_t = _charge_sum(ch, "coke")
        nut_coke_t = _charge_sum(ch, "nut_coke")
        sinter_t = _charge_sum(ch, "sinter")
        ore_t = _charge_sum(ch, "ore")
        pellet_t = _charge_sum(ch, "pellet")
        flux_t = _charge_sum(ch, "flux")
        burden_ratio = calculate_burden_ratio(ore_t, sinter_t, pellet_t)
        ibrm = calculate_ibrm(
            ch,
            production_t=theoretical_production,
            end_time=raw.shift_end_ist,
            ore_df=raw.ore_chemistry_df,
            sinter_t=sinter_t,
        )

        # Slag basicity
        cao = _hm_mean(hm, "slag_cao")
        sio2 = _hm_mean(hm, "slag_sio2")
        slag_basicity = round(cao / sio2, 2) if cao and sio2 else None

        # HM temp: use from hot_metal_slag_data reports
        hm_temp = _hm_mean(hm, "hmt_gt_1480c")

        coke_rate_val = _mean(df, "coke_rate")
        nut_coke_rate_val = _mean(df, "nut_coke_rate")
        pci_rate_val = _mean(df, "coal_rate_actual_value")
        fuel_rate_val = _sum_required(coke_rate_val, nut_coke_rate_val, pci_rate_val)
        raft_val = _mean(df, "body_raft")
        perm_val = _mean(df, "body_perm")
        etaco_val = _mean(df, "body_etaco")
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
            coke_t=coke_t,
            nut_coke_t=nut_coke_t,
            sinter_t=sinter_t,
            ore_t=ore_t,
            pellet_t=pellet_t,
            flux_t=flux_t,
            fuel_rate=fuel_rate_val,
            coke_rate=coke_rate_val,
            nut_coke_rate=nut_coke_rate_val,
            pci_rate=pci_rate_val,
            hm_si=_hm_mean(hm, "si"),
            hm_s=_hm_mean(hm, "s"),
            hm_temp=hm_temp,
            slag_basicity=slag_basicity,
            total_taps=int(hm.dropna(how="all").shape[0]) if not hm.empty else None,
            blast_volume=_ps(df, "hot_blast_vol_nm3h"),
            blast_temp=_ps(df, "hot_blast_temp"),
            blast_pressure=_ps(df, "hot_blast_press"),
            furnace_top_dp=_ps(df, "body_dp_top"),
            furnace_bottom_dp=_ps(df, "body_dp_bottom"),
            furnace_total_dp=_ps(df, "body_dp_total"),
            o2_flow=_ps(df, "oxygen_flow"),
            total_o2_flow=raw.total_o2_flow_max if raw.report_type == "Day" else None,
            o2_enrichment=_ps(df, "oxygen_enrichment_pct"),
            permeability=_ps(df, "body_perm"),
            etaco=_ps(df, "body_etaco"),
            raft=_ps(df, "body_raft"),
            uptake=_temp_row(
                df, "top_temp_1", "top_temp_2", "top_temp_3", "top_temp_4"
            ),
            skin_flow=_temp_row(
                df, "skin_flow_temp_a", "skin_flow_temp_b", "skin_flow_temp_c", "skin_flow_temp_d"
            ),
            lower_stack=_temp_row(
                df, "temp_18660_a", "temp_18660_b", "temp_18660_c", "temp_18660_d"
            ),
            belly=_temp_row(
                df, "temp_15162_a", "temp_15162_b", "temp_15162_c", "temp_15162_d"
            ),
            bosh=_temp_row(
                df, "temp_12975_a", "temp_12975_b", "temp_12975_c", "temp_12975_d"
            ),
            hearth_4_3_a=_mean(df, "temp_4373_a"),
            hearth_5_4_c=_mean(df, "temp_5411_c"),
            hearth_5_7_c=_mean(df, "temp_5757_c"),
            hearth_6_1_b=_mean(df, "temp_6103_b"),
            burden_moisture_input=burden_moisture_input,
            fines_input=fines_input,
            burden_ratio=burden_ratio,
            ibrm=ibrm,
            used_materials=_used_charge_materials(ch, material_resolver),
            report_type=raw.report_type,
        )
