"""Data context provider for the Evonith BMO page.

This module reads BMO configuration, fetches latest stock and chemistry
snapshots, loads historical model context, and builds the ore inputs consumed
by LP baseline and nonlinear total-cost optimization.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from domain.optimization_runtime import DatasetContextService, build_runtime_config
from utils.bmo.types import FluxInput, OreChemistry, OreInput
from furnace_data.dataset.service import DatasetService as OnlineDatasetService

from furnace_data.offline import (
    fetch_offline_data as _fetch_offline_data,
    fetch_offline_report as _fetch_offline_report,
)


HM_FE_COLUMN = "chem_pct_fe"
HM_PI_CHEM_COLUMNS = ("chem_pct_c", "chem_pct_si", "chem_pct_s")
HM_OTHER_CHEM_COLUMNS = ("chem_pct_mn", "chem_pct_p", "chem_pct_ti", "chem_pct_cr")
HM_ALL_CHEM_COLUMNS = (HM_FE_COLUMN,) + HM_PI_CHEM_COLUMNS + HM_OTHER_CHEM_COLUMNS


def _resolve_repo_path(path_str: str) -> Path:
    """
    Resolve a BMO config path against common repository locations.

    Streamlit can run from different working directories depending on how the
    app is launched. This helper keeps YAML paths stable by trying repository
    root, ``src``-relative, and current-working-directory locations.

    Args:
         - path_str: str - Absolute or repository-relative config path.

    Returns:
         - return Path - Existing resolved path when found, otherwise first candidate.
    """

    p = Path(path_str)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [repo_root / p, repo_root / "src" / p, Path.cwd() / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML configuration file into a dictionary.

    Missing config files are treated as empty mappings so tests can construct
    minimal providers. Production callers still receive parsed YAML when the
    configured files exist.

    Args:
         - path: Path - YAML file path to load.

    Returns:
         - return dict[str, Any] - Parsed YAML content or an empty dictionary.
    """

    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        return data or {}


class EvonithBmoContextProvider:
    """
    Provide BMO configuration, live snapshots, and model history context.

    The provider is the data boundary for the Blend Mix Optimizer page. It
    gathers online stock, online chemistry, fallback mapping values, and static
    ML history into typed ``OreInput`` records consumed by LP and DE.

    Args:
         - setting_path: str - Path to BMO settings YAML.
         - mapping_path: str - Path to ore mapping YAML.

    Returns:
         - return EvonithBmoContextProvider - Context provider for BMO workflows.
    """

    def __init__(
        self,
        setting_path: str = "src/config/setting_bmo.yml",
        mapping_path: str = "src/config/bmo_ore_mapping.yml",
    ) -> None:
        """
        Initialize BMO settings, mapping, and dataset history services.

        The provider keeps raw YAML and runtime config together so every BMO
        data read uses the same ore mapping, data-source settings, and history
        dataset configuration.

        Args:
             - setting_path: str - Path to BMO settings YAML.
             - mapping_path: str - Path to ore mapping YAML.

        Returns:
             - return None - Initializes provider configuration and services.
        """

        self.setting_path = _resolve_repo_path(setting_path)
        self.mapping_path = _resolve_repo_path(mapping_path)

        self.settings = _load_yaml(self.setting_path).get("bmo", {})
        self.mapping = _load_yaml(self.mapping_path)

        self._ores_cfg = self.mapping.get("ores", [])
        self._chem_field_map = self.mapping.get("chemistry_field_map", {})
        self.runtime_cfg = build_runtime_config(self.settings)
        self._dataset_service = DatasetContextService(
            static_dataset_path=self.runtime_cfg["dataset"].get("static_dataset_path"),
            refresh_enabled=False,
            refresh_rm_choice=str(
                self.runtime_cfg["dataset"].get("refresh_rm_choice", "Full")
            ),
        )
        self._last_stock_diagnostics: dict[str, Any] = {}
        self._last_chemistry_diagnostics: dict[str, Any] = {}
        self._last_hm_slag_diagnostics: dict[str, Any] = {}
        self._last_dpr_diagnostics: dict[str, Any] = {}
        self._last_history_diagnostics: dict[str, Any] = {}
        self._last_process_diagnostics: dict[str, Any] = {}
        self._last_flux_diagnostics: dict[str, Any] = {}
        self._last_online_context_diagnostics: dict[str, Any] = {}
        self._last_charge_context_diagnostics: dict[str, Any] = {}
        self._last_pellet_usage_diagnostics: dict[str, Any] = {}
        self._last_charge_mix_diagnostics: dict[str, Any] = {}
        self._last_manual_blend_diagnostics: dict[str, Any] = {}

    def _stock_fallback_mt(self, ore_cfg: dict[str, Any]) -> float:
        """
        Return a planning stock fallback when live stock data is unavailable.

        The optimizer should not treat a data-source outage as zero physical
        stock, because that makes normal min-share constraints infeasible. If a
        mapping-specific ``fallback_stock_mt`` is configured it is used first;
        otherwise the fallback uses a stock-only reference quantity from data
        source settings. That reference is not an optimizer target.

        Args:
             - ore_cfg: dict[str, Any] - Ore mapping configuration.

        Returns:
             - return float - Non-negative planning stock fallback in MT.
        """

        explicit_stock = ore_cfg.get("fallback_stock_mt")
        if explicit_stock is not None:
            try:
                return max(0.0, float(explicit_stock))
            except (TypeError, ValueError):
                pass

        data_sources = self.settings.get("data_sources", {}) or {}
        reference_qty = float(
            data_sources.get("stock_fallback_reference_qty_mt", 0.0) or 0.0
        )
        max_share_pct = float(ore_cfg.get("max_share_pct", 0.0) or 0.0)
        min_share_pct = float(ore_cfg.get("min_share_pct", 0.0) or 0.0)
        planning_share_pct = max(max_share_pct, min_share_pct)
        if reference_qty <= 0.0 or planning_share_pct <= 0.0:
            return 0.0
        return reference_qty * planning_share_pct / 100.0

    @staticmethod
    def _frame_with_time_column(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.reset_index()
        if "index" in out.columns and "time" not in out.columns:
            out = out.rename(columns={"index": "time"})
        if "date_time" in out.columns and "time" not in out.columns:
            out = out.rename(columns={"date_time": "time"})
        if "time" in out.columns:
            out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True)
            out = out.dropna(subset=["time"]).sort_values("time")
        return out

    @staticmethod
    def _chemistry_table_for(material_code: str) -> str:
        if material_code.startswith("sinter"):
            return "offline_feed.sinter_chemistry"
        return "offline_feed.ore_chemistry"

    @staticmethod
    def _latest_row_snapshot(
        row_group: pd.DataFrame, required_columns: list[str] | None = None
    ) -> pd.Series:
        numeric = row_group.drop(
            columns=["time", "material_code", "source_table"], errors="ignore"
        ).apply(pd.to_numeric, errors="coerce")
        if numeric.empty:
            return pd.Series(dtype=float)
        required = [col for col in (required_columns or []) if col in numeric.columns]
        if required:
            valid = numeric[required].notna().all(axis=1)
        else:
            valid = numeric.notna().any(axis=1)
        if not valid.any():
            return pd.Series(dtype=float)
        return numeric.loc[valid].iloc[-1].dropna()

    @staticmethod
    def _average_non_null(row_group: pd.DataFrame) -> pd.Series:
        numeric = row_group.drop(
            columns=["time", "material_code", "source_table"], errors="ignore"
        ).apply(pd.to_numeric, errors="coerce")
        return numeric.mean(skipna=True)

    @staticmethod
    def _average_non_zero(row_group: pd.DataFrame) -> pd.Series:
        numeric = row_group.drop(
            columns=["time", "material_code", "source_table"], errors="ignore"
        ).apply(pd.to_numeric, errors="coerce")
        return numeric.mask(numeric == 0).mean(skipna=True)

    @staticmethod
    def _is_sinter_material(material_code: str) -> bool:
        return material_code.startswith("sinter")

    @staticmethod
    def _material_charge_column(material_code: str) -> str:
        return f"{material_code}_mt"

    @staticmethod
    def _material_code_from_quantity_column(column_name: str) -> str:
        column = str(column_name).strip()
        return column[:-3] if column.endswith("_mt") else column

    @staticmethod
    def _repair_material_quantity_totals(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        aggregate_specs = {
            "ORE_CALC_MT": [f"ORE_{i}_CALC_MT" for i in range(1, 13)],
            "FLUX_CALC_MT": [f"FLUX_{i}_CALC_MT" for i in range(1, 4)],
        }
        for aggregate, columns in aggregate_specs.items():
            present = [column for column in columns if column in out.columns]
            if present and aggregate in out.columns:
                values = out[present].apply(pd.to_numeric, errors="coerce")
                has_slot_data = values.notna().any(axis=1)
                out.loc[has_slot_data, aggregate] = (
                    values.loc[has_slot_data].fillna(0.0).sum(axis=1)
                )
        return out

    @staticmethod
    def _iso(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        try:
            timestamp = pd.to_datetime(value, utc=True)
            return timestamp.isoformat()
        except Exception:
            return str(value)

    def _dataset_rename_dict(self) -> dict[str, str]:
        config = _load_yaml(_resolve_repo_path("src/config/setting_ds_dv.yml"))
        rename_dict = config.get("rename_dict", {}) or {}
        if not isinstance(rename_dict, dict):
            return {}
        return {str(key): str(value) for key, value in rename_dict.items()}

    def _charge_columns_by_group(self) -> dict[str, list[str]]:
        config = _load_yaml(_resolve_repo_path("src/config/shift_report.yml"))
        groups = config.get("charge_columns", {}) or {}
        wanted = ("coke", "nut_coke", "sinter", "ore", "pellet", "flux")
        return {
            group: [str(column) for column in groups.get(group, [])]
            for group in wanted
        }

    def _recent_context_window(
        self, max_lag_hours: int
    ) -> tuple[ZoneInfo, pd.Timestamp, pd.Timestamp]:
        tz = ZoneInfo(str(self.settings.get("timezone", "Asia/Kolkata")))
        end_local = pd.Timestamp.now(tz=tz).floor("h")
        start_local = end_local - pd.Timedelta(hours=int(max_lag_hours) + 1)
        return tz, start_local, end_local

    def _fetch_recent_online_context(
        self, max_lag_hours: int
    ) -> tuple[pd.DataFrame, list[str]]:
        if max_lag_hours <= 0:
            self._last_online_context_diagnostics = {
                "source": "online_influx",
                "enabled": False,
                "reason": "model_has_no_lag_features",
                "rows": 0,
                "columns": 0,
            }
            return pd.DataFrame(), []

        warnings: list[str] = []
        tz, start_local, end_local = self._recent_context_window(max_lag_hours)

        service = OnlineDatasetService(local_tz=str(tz))
        frames: list[pd.DataFrame] = []
        issues: list[str] = []
        fetches = [
            ("process_params", service.fetch_online_process_params),
            ("temperature_profile", service.fetch_online_temperature_params),
            ("heatload_delta_t", service.fetch_online_heatload_params),
        ]
        for label, fetch in fetches:
            try:
                frame = fetch(
                    start_local.date(),
                    end_local.date(),
                    raise_on_error=True,
                )
            except Exception as exc:
                issues.append(f"{label} fetch failed: {exc}")
                continue
            if frame is None or frame.empty:
                issues.append(f"{label} unavailable: no mapped rows")
                continue
            frames.append(frame)

        if issues:
            warnings.extend([f"Online model context issue: {issue}" for issue in issues])

        if not frames:
            self._last_online_context_diagnostics = {
                "source": "online_influx",
                "enabled": True,
                "max_lag_hours": int(max_lag_hours),
                "start_time": self._iso(start_local),
                "end_time": self._iso(end_local),
                "rows": 0,
                "columns": 0,
                "issues": issues,
            }
            return pd.DataFrame(), warnings

        online_df = pd.concat(frames, axis=1).sort_index()
        online_df = online_df.loc[:, ~online_df.columns.duplicated()]
        online_df = online_df.rename(columns=self._dataset_rename_dict())
        online_df.index = pd.to_datetime(online_df.index, errors="coerce")
        online_df = online_df[~online_df.index.isna()]
        if online_df.index.tz is None:
            online_df.index = online_df.index.tz_localize(tz)
        else:
            online_df.index = online_df.index.tz_convert(tz)
        online_df = online_df.loc[start_local:end_local]
        online_df.index = online_df.index.tz_convert(timezone.utc)

        self._last_online_context_diagnostics = {
            "source": "online_influx",
            "enabled": True,
            "max_lag_hours": int(max_lag_hours),
            "start_time": self._iso(online_df.index.min())
            if not online_df.empty
            else self._iso(start_local),
            "end_time": self._iso(online_df.index.max())
            if not online_df.empty
            else self._iso(end_local),
            "rows": int(len(online_df)),
            "columns": int(len(online_df.columns)),
            "column_names": list(map(str, online_df.columns)),
            "issues": issues,
        }
        return online_df, warnings

    def _fetch_recent_charge_context(
        self, max_lag_hours: int
    ) -> tuple[pd.DataFrame, list[str]]:
        if max_lag_hours <= 0:
            self._last_charge_context_diagnostics = {
                "source": "offline_feed.charge_data",
                "enabled": False,
                "rows": 0,
            }
            return pd.DataFrame(), []

        warnings: list[str] = []
        tz, start_local, end_local = self._recent_context_window(max_lag_hours)
        start_utc = start_local.tz_convert(timezone.utc).floor("h")
        end_utc = end_local.tz_convert(timezone.utc).floor("h")
        fetch_end_utc = end_utc + pd.Timedelta(hours=1)
        columns_by_group = self._charge_columns_by_group()
        all_columns = sorted(
            {
                column
                for columns in columns_by_group.values()
                for column in columns
            }
        )
        if not all_columns:
            self._last_charge_context_diagnostics = {
                "source": "offline_feed.charge_data",
                "enabled": True,
                "rows": 0,
                "start_time": self._iso(start_local),
                "end_time": self._iso(end_local),
                "warning": "No charge columns configured.",
            }
            return pd.DataFrame(), warnings
        try:
            df = _fetch_offline_data(
                table_name="offline_feed.charge_data",
                time_range=(
                    start_utc.to_pydatetime(),
                    fetch_end_utc.to_pydatetime(),
                ),
                query_type="raw",
                columns=["date_time", *all_columns],
            )
        except Exception as exc:
            warning = f"Could not fetch recent charge context: {exc}"
            warnings.append(warning)
            self._last_charge_context_diagnostics = {
                "source": "offline_feed.charge_data",
                "enabled": True,
                "rows": 0,
                "error": str(exc),
            }
            return pd.DataFrame(), warnings

        charge_df = self._frame_with_time_column(df)
        present_columns = [column for column in all_columns if column in charge_df.columns]
        if charge_df.empty or not present_columns:
            self._last_charge_context_diagnostics = {
                "source": "offline_feed.charge_data",
                "enabled": True,
                "rows": 0,
                "start_time": self._iso(start_local),
                "end_time": self._iso(end_local),
            }
            return pd.DataFrame(), warnings

        charge_df = charge_df.set_index("time").sort_index()
        if charge_df.index.tz is None:
            charge_df.index = charge_df.index.tz_localize(timezone.utc)
        else:
            charge_df.index = charge_df.index.tz_convert(timezone.utc)

        numeric = charge_df[present_columns].apply(pd.to_numeric, errors="coerce")
        hourly_index = pd.date_range(
            start=start_utc,
            end=end_utc,
            freq="1h",
        )
        hourly = (
            numeric.fillna(0.0)
            .resample("1h")
            .sum()
            .reindex(hourly_index, fill_value=0.0)
        )

        rename_dict = self._dataset_rename_dict()
        out = hourly.rename(columns=rename_dict)

        def group_sum(group: str) -> pd.Series | None:
            cols = [col for col in columns_by_group.get(group, []) if col in hourly.columns]
            if not cols:
                return None
            return hourly[cols].sum(axis=1)

        aggregate_map = {
            "coke": "COKE_CALC_MT",
            "nut_coke": "NUTCOKE_CALC_MT",
            "sinter": "SINTER_CALC_MT",
            "ore": "ORE_CALC_MT",
            "pellet": "TOTAL_PELLET_CALC_MT",
            "flux": "FLUX_CALC_MT",
        }
        for group, feature_name in aggregate_map.items():
            summed = group_sum(group)
            if summed is not None:
                out[feature_name] = summed
        if "coke_1_mt" in hourly.columns:
            out["COKE_ON_MT"] = hourly["coke_1_mt"]
        if "coke_2_mt" in hourly.columns:
            out["COKE_OFF_MT"] = hourly["coke_2_mt"]
        out = out.loc[:, ~out.columns.duplicated()]
        out.index.name = "time"

        self._last_charge_context_diagnostics = {
            "source": "offline_feed.charge_data",
            "enabled": True,
            "rows": int(len(charge_df)),
            "hourly_rows": int(len(out)),
            "start_time": self._iso(out.index.min()) if not out.empty else "",
            "end_time": self._iso(out.index.max()) if not out.empty else "",
            "source_columns": present_columns,
            "feature_columns": list(map(str, out.columns)),
        }
        return out, warnings

    def get_recent_active_pellet_ids(
        self, lookback_days: int | None = None
    ) -> tuple[list[str], list[str]]:
        cfg = self.settings.get("data_sources", {})
        days = int(lookback_days or cfg.get("pellet_auto_select_lookback_days", 30))
        pellet_cfgs = [
            item
            for item in self._ores_cfg
            if "pellet" in str(item.get("material_key", "")).lower()
        ]
        columns_by_id = {
            str(item.get("id")): f"{str(item.get('material_key')).strip()}_mt"
            for item in pellet_cfgs
            if str(item.get("id", "")).strip()
            and str(item.get("material_key", "")).strip()
        }
        if not columns_by_id:
            self._last_pellet_usage_diagnostics = {"rows": [], "warnings": []}
            return [], []

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        warnings: list[str] = []
        usage_rows: list[dict[str, Any]] = []
        active_ids: list[str] = []
        try:
            df = _fetch_offline_data(
                table_name="offline_feed.charge_data",
                time_range=(start_time, end_time),
                query_type="raw",
                columns=["date_time", *columns_by_id.values()],
            )
            charge_df = self._frame_with_time_column(df)
        except Exception as exc:
            warnings.append(f"Could not check recent pellet charge usage: {exc}")
            charge_df = pd.DataFrame()

        for ore_id, column in columns_by_id.items():
            total_mt = 0.0
            nonzero_rows = 0
            if column in charge_df.columns:
                values = pd.to_numeric(charge_df[column], errors="coerce").fillna(0.0)
                total_mt = float(values.sum())
                nonzero_rows = int((values > 0.0).sum())
            if total_mt > 0.0:
                active_ids.append(ore_id)
            usage_rows.append(
                {
                    "ore_id": ore_id,
                    "charge_column": column,
                    "lookback_days": days,
                    "nonzero_rows": nonzero_rows,
                    "total_mt": total_mt,
                    "selected_by_default": total_mt > 0.0,
                }
            )

        self._last_pellet_usage_diagnostics = {
            "source": "offline_feed.charge_data",
            "start_time": self._iso(start_time),
            "end_time": self._iso(end_time),
            "rows": usage_rows,
            "warnings": list(warnings),
        }
        return active_ids, warnings

    def get_charge_mix_snapshot(self) -> dict[str, Any]:
        columns_by_group = self._charge_columns_by_group()
        all_columns = sorted(
            {
                column
                for columns in columns_by_group.values()
                for column in columns
            }
        )
        if not all_columns:
            self._last_charge_mix_diagnostics = {"rows": [], "warnings": []}
            return self._last_charge_mix_diagnostics

        warnings: list[str] = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=2)
        try:
            df = _fetch_offline_data(
                table_name="offline_feed.charge_data",
                time_range=(start_time, end_time),
                query_type="raw",
                columns=["date_time", *all_columns],
            )
            charge_df = self._frame_with_time_column(df)
        except Exception as exc:
            warnings.append(f"Could not fetch charge mix diagnostics: {exc}")
            charge_df = pd.DataFrame()

        rows: list[dict[str, Any]] = []
        if not charge_df.empty:
            charge_df = charge_df.set_index("time").sort_index()
            if charge_df.index.tz is None:
                charge_df.index = charge_df.index.tz_localize(timezone.utc)
            else:
                charge_df.index = charge_df.index.tz_convert(timezone.utc)
            latest_utc = charge_df.index.max()
            latest_local = latest_utc.tz_convert(
                ZoneInfo(str(self.settings.get("timezone", "Asia/Kolkata")))
            )
            day_start_local = latest_local.normalize()
            hour = int(latest_local.hour)
            if 6 <= hour < 14:
                shift_start_local = day_start_local + pd.Timedelta(hours=6)
            elif 14 <= hour < 22:
                shift_start_local = day_start_local + pd.Timedelta(hours=14)
            else:
                shift_start_local = day_start_local + pd.Timedelta(hours=22)
                if hour < 6:
                    shift_start_local -= pd.Timedelta(days=1)

            windows = {
                "latest_row": charge_df.loc[[latest_utc]],
                "current_shift": charge_df[
                    charge_df.index >= shift_start_local.tz_convert(timezone.utc)
                ],
                "plant_day": charge_df[
                    charge_df.index >= day_start_local.tz_convert(timezone.utc)
                ],
                "last_24h": charge_df[
                    charge_df.index >= latest_utc - pd.Timedelta(hours=24)
                ],
            }
            for window_name, frame in windows.items():
                group_totals: dict[str, float] = {}
                for group, columns in columns_by_group.items():
                    present = [column for column in columns if column in frame.columns]
                    if present:
                        group_totals[group] = float(
                            frame[present].apply(pd.to_numeric, errors="coerce")
                            .fillna(0.0)
                            .sum()
                            .sum()
                        )
                    else:
                        group_totals[group] = 0.0
                total_mt = sum(group_totals.values())
                for group, total in group_totals.items():
                    rows.append(
                        {
                            "window": window_name,
                            "group": group,
                            "quantity_mt": total,
                            "share_pct": (total / total_mt * 100.0)
                            if total_mt > 0.0
                            else 0.0,
                            "rows": int(len(frame)),
                            "anchor_time": self._iso(latest_utc),
                        }
                    )

        self._last_charge_mix_diagnostics = {
            "source": "offline_feed.charge_data",
            "start_time": self._iso(start_time),
            "end_time": self._iso(end_time),
            "rows": rows,
            "warnings": warnings,
        }
        return self._last_charge_mix_diagnostics

    def get_recent_manual_blend_snapshot(
        self, selected_ores: list[OreInput]
    ) -> dict[str, Any]:
        columns_by_ore = {
            ore.ore_id: self._material_charge_column(
                str(ore.metadata.get("material_key", "")).strip()
            )
            for ore in selected_ores
            if str(ore.metadata.get("material_key", "")).strip()
        }
        if not columns_by_ore:
            self._last_manual_blend_diagnostics = {"rows": [], "warnings": []}
            return self._last_manual_blend_diagnostics

        warnings: list[str] = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=3)
        try:
            df = _fetch_offline_data(
                table_name="offline_feed.charge_data",
                time_range=(start_time, end_time),
                query_type="raw",
                columns=["date_time", *sorted(set(columns_by_ore.values()))],
            )
            charge_df = self._frame_with_time_column(df)
        except Exception as exc:
            warnings.append(f"Could not fetch recent manual blend: {exc}")
            charge_df = pd.DataFrame()

        rows: list[dict[str, Any]] = []
        window_start_utc = None
        window_end_utc = None
        if not charge_df.empty:
            charge_df = charge_df.set_index("time").sort_index()
            if charge_df.index.tz is None:
                charge_df.index = charge_df.index.tz_localize(timezone.utc)
            else:
                charge_df.index = charge_df.index.tz_convert(timezone.utc)
            latest_utc = charge_df.index.max()
            tz = ZoneInfo(str(self.settings.get("timezone", "Asia/Kolkata")))
            latest_local = latest_utc.tz_convert(tz)
            day_start_local = latest_local.normalize()
            hour = int(latest_local.hour)
            if 6 <= hour < 14:
                shift_start_local = day_start_local + pd.Timedelta(hours=6)
            elif 14 <= hour < 22:
                shift_start_local = day_start_local + pd.Timedelta(hours=14)
            else:
                shift_start_local = day_start_local + pd.Timedelta(hours=22)
                if hour < 6:
                    shift_start_local -= pd.Timedelta(days=1)

            window_start_utc = (shift_start_local - pd.Timedelta(hours=8)).tz_convert(
                timezone.utc
            )
            window_end_utc = shift_start_local.tz_convert(timezone.utc)
            frame = charge_df[
                (charge_df.index >= window_start_utc)
                & (charge_df.index < window_end_utc)
            ]
            totals_by_ore: dict[str, float] = {}
            for ore in selected_ores:
                column = columns_by_ore.get(ore.ore_id, "")
                if column and column in frame.columns:
                    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
                    totals_by_ore[ore.ore_id] = float(values.sum())
                else:
                    totals_by_ore[ore.ore_id] = 0.0
            total_mt = sum(totals_by_ore.values())
            for ore in selected_ores:
                material_code = str(ore.metadata.get("material_key", "")).strip()
                qty = float(totals_by_ore.get(ore.ore_id, 0.0))
                rows.append(
                    {
                        "ore_id": ore.ore_id,
                        "ore_name": ore.display_name,
                        "material_code": material_code,
                        "charge_column": columns_by_ore.get(ore.ore_id, ""),
                        "quantity_mt": qty,
                        "share_pct": qty / total_mt * 100.0 if total_mt > 0 else 0.0,
                    }
                )
            if total_mt <= 0:
                warnings.append("Last completed shift has zero selected-material charge.")

        self._last_manual_blend_diagnostics = {
            "source": "offline_feed.charge_data",
            "start_time": self._iso(window_start_utc),
            "end_time": self._iso(window_end_utc),
            "rows": rows,
            "warnings": warnings,
        }
        return self._last_manual_blend_diagnostics

    def _append_online_context(
        self,
        history_df: pd.DataFrame,
        online_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if online_df.empty:
            return history_df
        online_df = online_df.dropna(axis=1, how="all")
        if online_df.empty:
            return history_df
        if history_df.empty:
            return online_df.sort_index()
        combined = pd.concat([history_df, online_df], axis=0).sort_index()
        return combined.groupby(level=0).last().sort_index()

    def get_stock_snapshot(self) -> tuple[dict[str, float], list[str]]:
        """
        Build the latest stock snapshot for all configured BMO ores.

        Live stock is preferred because LP and DE enforce stock as hard bounds.
        Missing ore stock means zero available stock. Sinter is the exception:
        if the plant stock table does not carry sinter, a large planning stock
        is used so normal sinter share constraints remain feasible.

        Args:
             - None

        Returns:
             - return tuple[dict[str, float], list[str]] - Stock map and warnings.
        """

        cfg = self.settings.get("data_sources", {})
        table_name = str(cfg.get("stock_table", "offline_feed.raw_material_stock"))
        time_range = str(cfg.get("stock_time_range", "last 1 week"))

        warnings: list[str] = []
        latest_by_material: dict[str, dict[str, Any]] = {}
        returned_rows = 0
        try:
            df = _fetch_offline_data(
                table_name=table_name,
                time_range=time_range,
                query_type="raw",
                columns=["date_time", "material_code", "stock_mt"],
            )
            stock_df = self._frame_with_time_column(df)
            if stock_df.empty:
                warnings.append(f"No rows returned from {table_name}.")
            else:
                returned_rows = int(len(stock_df))
                stock_df["stock_mt"] = pd.to_numeric(
                    stock_df["stock_mt"], errors="coerce"
                )
                stock_df = stock_df.dropna(subset=["material_code"])
                latest = stock_df.groupby("material_code", sort=False).tail(1)
                latest_by_material = {
                    str(row["material_code"]): {
                        "stock_mt": row["stock_mt"],
                        "timestamp": self._iso(row.get("time")),
                    }
                    for _, row in latest.iterrows()
                }
        except Exception as exc:
            warnings.append(f"Stock query failed for {table_name}: {exc}")

        stock_map: dict[str, float] = {}
        material_rows: list[dict[str, Any]] = []
        fallback_count = 0
        zero_count = 0
        for ore_cfg in self._ores_cfg:
            ore_id = str(ore_cfg.get("id"))
            material_code = str(ore_cfg.get("material_key", "")).strip()
            source_row = latest_by_material.get(material_code)
            if source_row is None:
                timestamp = ""
                if self._is_sinter_material(material_code):
                    value = self._stock_fallback_mt(ore_cfg)
                    fallback_count += 1
                    source = "sinter_fallback"
                else:
                    value = 0.0
                    zero_count += 1
                    source = "offline_db_missing_as_zero"
            else:
                timestamp = str(source_row.get("timestamp", ""))
                if pd.isna(source_row["stock_mt"]):
                    value = 0.0
                    zero_count += 1
                    source = "offline_db_null_as_zero"
                else:
                    value = float(source_row["stock_mt"])
                    source = "offline_db"
            stock_map[ore_id] = max(0.0, value)
            material_rows.append(
                {
                    "ore_id": ore_id,
                    "display_name": str(ore_cfg.get("display_name", ore_id)),
                    "material_code": material_code,
                    "stock_mt": max(0.0, value),
                    "source": source,
                    "timestamp": timestamp,
                }
            )

        if fallback_count:
            warnings.append(
                (
                    f"Sinter stock unavailable for {fallback_count} material(s); "
                    "using planning stock fallback."
                )
            )
        self._last_stock_diagnostics = {
            "table": table_name,
            "time_range": time_range,
            "returned_rows": returned_rows,
            "fallback_count": fallback_count,
            "zero_count": zero_count,
            "material_rows": material_rows,
            "warnings": list(warnings),
        }
        return stock_map, warnings

    def _latest_usage_by_material(
        self, material_codes: set[str], lookback_days: int
    ) -> tuple[dict[str, pd.Timestamp], list[dict[str, Any]], list[str]]:
        columns_by_material = {
            material_code: self._material_charge_column(material_code)
            for material_code in material_codes
            if material_code
        }
        columns = ["date_time", *sorted(set(columns_by_material.values()))]
        end_time = datetime.now(timezone.utc)
        time_range = (end_time - timedelta(days=max(1, lookback_days)), end_time)
        warnings: list[str] = []
        rows: list[dict[str, Any]] = []
        latest_usage: dict[str, pd.Timestamp] = {}

        try:
            df = _fetch_offline_data(
                table_name="offline_feed.charge_data",
                time_range=time_range,
                query_type="raw",
                columns=columns,
            )
        except Exception as exc:
            return {}, rows, [f"Charge usage query failed for chemistry latest mode: {exc}"]

        charge_df = self._frame_with_time_column(df)
        if charge_df.empty:
            return {}, rows, ["No charge rows returned for chemistry latest usage lookup."]

        for material_code, column in columns_by_material.items():
            if column not in charge_df.columns:
                rows.append(
                    {
                        "material_code": material_code,
                        "charge_column": column,
                        "latest_used_time": "",
                        "rows_used": 0,
                    }
                )
                continue
            qty = pd.to_numeric(charge_df[column], errors="coerce").fillna(0.0)
            used = charge_df.loc[qty > 0, ["time", column]]
            if used.empty:
                rows.append(
                    {
                        "material_code": material_code,
                        "charge_column": column,
                        "latest_used_time": "",
                        "rows_used": 0,
                    }
                )
                continue
            latest_time = pd.to_datetime(used["time"].iloc[-1], utc=True)
            latest_usage[material_code] = latest_time
            rows.append(
                {
                    "material_code": material_code,
                    "charge_column": column,
                    "latest_used_time": self._iso(latest_time),
                    "rows_used": int(len(used)),
                }
            )
        return latest_usage, rows, warnings

    def get_chemistry_snapshot(
        self, mode: str = "latest", window_days: int | None = None
    ) -> tuple[dict[str, OreChemistry], list[str]]:
        """
        Build the chemistry snapshot for all configured BMO ores.

        The chemistry mapping now includes moisture/TM when available. That
        value is loaded into ``OreChemistry`` so final Fe% can be calculated on
        dry weight instead of directly averaging wet quantities.

        Args:
             - mode: str - Snapshot mode, either latest non-null or average.
             - window_days: int | None - Number of days of chemistry data to query.

        Returns:
             - return tuple[dict[str, OreChemistry], list[str]] - Chemistry map and warnings.
        """

        cfg = self.settings.get("data_sources", {})
        days = int(window_days or cfg.get("chemistry_time_range_days", 30))
        mode = "avg" if mode == "avg" else "latest"
        latest_usage_lookback_days = int(cfg.get("latest_usage_lookback_days", 365))
        query_days = max(days, latest_usage_lookback_days) if mode == "latest" else days
        end_time = datetime.now(timezone.utc)
        time_range = (end_time - timedelta(days=query_days), end_time)

        warnings: list[str] = []
        chemistry_by_ore: dict[str, OreChemistry] = {}

        material_tables: dict[str, set[str]] = {}
        for ore_cfg in self._ores_cfg:
            material_code = str(ore_cfg.get("material_key", "")).strip()
            if not material_code:
                continue
            material_tables.setdefault(
                self._chemistry_table_for(material_code), set()
            ).add(material_code)

        all_material_codes = {
            material_code
            for material_codes in material_tables.values()
            for material_code in material_codes
        }
        usage_times: dict[str, pd.Timestamp] = {}
        usage_rows: list[dict[str, Any]] = []
        if mode == "latest":
            usage_times, usage_rows, usage_warnings = self._latest_usage_by_material(
                all_material_codes, latest_usage_lookback_days
            )
            warnings.extend(usage_warnings)

        snapshots: dict[str, pd.Series] = {}
        source_meta: dict[str, dict[str, Any]] = {}
        table_rows: list[dict[str, Any]] = []
        for table_name, material_codes in material_tables.items():
            try:
                df = _fetch_offline_data(
                    table_name=table_name,
                    time_range=time_range,
                    query_type="raw",
                )
            except Exception as exc:
                warnings.append(f"Chemistry query failed for {table_name}: {exc}")
                continue
            chem_df = self._frame_with_time_column(df)
            if chem_df.empty or "material_code" not in chem_df.columns:
                table_rows.append(
                    {
                        "table": table_name,
                        "returned_rows": 0,
                        "materials_requested": ", ".join(sorted(material_codes)),
                    }
                )
                continue
            chem_df = chem_df[chem_df["material_code"].isin(material_codes)]
            table_rows.append(
                {
                    "table": table_name,
                    "returned_rows": int(len(chem_df)),
                    "materials_requested": ", ".join(sorted(material_codes)),
                    "start_time": self._iso(chem_df["time"].min())
                    if "time" in chem_df.columns and not chem_df.empty
                    else "",
                    "end_time": self._iso(chem_df["time"].max())
                    if "time" in chem_df.columns and not chem_df.empty
                    else "",
                }
            )
            if chem_df.empty:
                continue
            for material_code, group in chem_df.groupby("material_code", sort=False):
                group = group.sort_values("time") if "time" in group.columns else group
                material_code = str(material_code)
                source_group = group
                source = f"offline_db_{mode}"
                latest_used_time = usage_times.get(material_code)
                if mode == "avg":
                    snapshots[material_code] = self._average_non_zero(source_group)
                    source = "offline_db_avg_non_zero"
                else:
                    if latest_used_time is not None and "time" in group.columns:
                        used_group = group[group["time"] <= latest_used_time]
                        if not used_group.empty:
                            source_group = used_group
                            source = "offline_db_latest_used"
                        else:
                            source = "offline_db_latest_no_prior_sample"
                    elif latest_used_time is None:
                        source = "offline_db_latest_no_usage"
                    snapshot = self._latest_row_snapshot(
                        source_group, list(self._chem_field_map.values())
                    )
                    if (
                        not snapshot.empty
                        and snapshot.name is not None
                        and snapshot.name in source_group.index
                    ):
                        source_group = source_group.loc[[snapshot.name]]
                    snapshots[material_code] = snapshot
                source_meta[str(material_code)] = {
                    "source": source,
                    "table": table_name,
                    "rows_used": int(len(source_group)),
                    "start_time": self._iso(source_group["time"].min())
                    if "time" in source_group.columns and not source_group.empty
                    else "",
                    "end_time": self._iso(source_group["time"].max())
                    if "time" in source_group.columns and not source_group.empty
                    else "",
                    "sample_timestamp": self._iso(source_group["time"].max())
                    if "time" in source_group.columns and not source_group.empty
                    else "",
                    "latest_used_time": self._iso(latest_used_time),
                }

        fallback_count = 0
        material_rows: list[dict[str, Any]] = []
        for ore_cfg in self._ores_cfg:
            ore_id = str(ore_cfg.get("id"))
            material_key = str(ore_cfg.get("material_key", ""))
            fallback = ore_cfg.get("fallback_chemistry", {}) or {}
            chem_values: dict[str, float] = {}
            snapshot = snapshots.get(material_key, pd.Series(dtype=float))
            if snapshot.empty:
                fallback_count += 1
                meta = {
                    "source": "fallback",
                    "table": self._chemistry_table_for(material_key),
                    "rows_used": 0,
                    "start_time": "",
                    "end_time": "",
                    "sample_timestamp": "",
                    "latest_used_time": self._iso(usage_times.get(material_key)),
                }
            else:
                meta = source_meta.get(material_key, {})

            for chem_attr, source_column in self._chem_field_map.items():
                val = snapshot.get(source_column, None)
                if val is None or pd.isna(val):
                    val = fallback.get(chem_attr, 0.0)
                try:
                    chem_values[chem_attr] = float(val)
                except (TypeError, ValueError):
                    chem_values[chem_attr] = float(fallback.get(chem_attr, 0.0))

            chemistry_by_ore[ore_id] = OreChemistry(
                fe_t_pct=float(chem_values.get("fe_t_pct", 0.0)),
                moisture_pct=float(chem_values.get("moisture_pct", 0.0)),
                feo_pct=float(chem_values.get("feo_pct", 0.0)),
                sio2_pct=float(chem_values.get("sio2_pct", 0.0)),
                al2o3_pct=float(chem_values.get("al2o3_pct", 0.0)),
                cao_pct=float(chem_values.get("cao_pct", 0.0)),
                mgo_pct=float(chem_values.get("mgo_pct", 0.0)),
                mno_pct=float(chem_values.get("mno_pct", 0.0)),
                tio2_pct=float(chem_values.get("tio2_pct", 0.0)),
                p_pct=float(chem_values.get("p_pct", 0.0)),
                na2o_pct=float(chem_values.get("na2o_pct", 0.0)),
                k2o_pct=float(chem_values.get("k2o_pct", 0.0)),
            )
            material_rows.append(
                {
                    "ore_id": ore_id,
                    "display_name": str(ore_cfg.get("display_name", ore_id)),
                    "material_code": material_key,
                    **meta,
                    **{
                        key: float(value)
                        for key, value in chem_values.items()
                    },
                }
            )

        if fallback_count:
            warnings.append(
                f"Chemistry unavailable for {fallback_count} material(s); using configured fallback chemistry."
            )

        self._last_chemistry_diagnostics = {
            "mode": mode,
            "window_days": days,
            "start_time": self._iso(time_range[0]),
            "end_time": self._iso(time_range[1]),
            "fallback_count": fallback_count,
            "latest_usage_lookback_days": latest_usage_lookback_days
            if mode == "latest"
            else None,
            "tables": table_rows,
            "usage_rows": usage_rows,
            "material_rows": material_rows,
            "warnings": list(warnings),
        }
        return chemistry_by_ore, warnings

    def get_hm_slag_snapshot(
        self, *, mode: str = "latest", window_days: int = 30
    ) -> dict[str, Any]:
        """
        Build a hot-metal & slag analysis snapshot for the slag balance settings.

        The full BF slag balance needs HM C/Si/S and the sum of minor HM elements
        to subtract pig-iron inclusions (SiO2 reduction, S to PI, residual Fe/Mn
        to slag). This snapshot mirrors ``get_chemistry_snapshot`` so the page can
        offer the same latest/avg controls.

        Args:
             - mode: str - Snapshot mode, either "latest" non-zero or "avg" over window.
             - window_days: int - Number of days of HM_SLAG data to query.

        Returns:
             - return dict[str, Any] - chem_pct_*, others_pct, observed_slag_rate_kg_per_thm,
               n_rows_used, source, warnings.
        """

        warnings: list[str] = []
        empty: dict[str, Any] = {"warnings": warnings, "source": None, "n_rows_used": 0}

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(1, int(window_days)))

        try:
            df = _fetch_offline_report("HM_SLAG", (start, end))
        except Exception as exc:
            warnings.append(f"HM_SLAG query failed: {exc}")
            return self._hm_slag_snapshot_from_history(mode, window_days, warnings)

        if df is None or df.empty:
            warnings.append("HM_SLAG returned no rows for the window.")
            return self._hm_slag_snapshot_from_history(mode, window_days, warnings)

        df = df.sort_index()
        snapshot: dict[str, Any] = {"warnings": warnings, "n_rows_used": 0}

        if mode == "latest":
            mask = pd.Series(True, index=df.index)
            for col in ("chem_pct_si", "chem_pct_c"):
                if col in df.columns:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    mask &= numeric.notna() & (numeric > 0)
            valid = df[mask]
            if valid.empty:
                warnings.append("No HM_SLAG rows with non-zero Si/C in window.")
                return empty
            row = valid.iloc[-1]
            for col in HM_ALL_CHEM_COLUMNS:
                value = pd.to_numeric(row.get(col, 0.0), errors="coerce")
                snapshot[col] = float(value) if pd.notna(value) else 0.0
            snapshot["n_rows_used"] = 1
            snapshot["source"] = "latest"
            snapshot["sample_timestamp"] = self._iso(row.name)
        else:
            for col in HM_ALL_CHEM_COLUMNS:
                if col not in df.columns:
                    snapshot[col] = 0.0
                    continue
                series = pd.to_numeric(df[col], errors="coerce")
                series = series[series > 0]
                snapshot[col] = float(series.mean()) if len(series) > 0 else 0.0
            snapshot["n_rows_used"] = int(len(df))
            snapshot["source"] = "avg"
            snapshot["start_time"] = self._iso(df.index.min())
            snapshot["end_time"] = self._iso(df.index.max())

        if HM_FE_COLUMN in df.columns:
            fe_series = pd.to_numeric(df[HM_FE_COLUMN], errors="coerce")
            fe_series = fe_series[fe_series > 0]
            if len(fe_series) > 0:
                snapshot["hm_fe_pct_for_target"] = float(fe_series.mean())

        snapshot["others_pct"] = float(
            sum(snapshot.get(col, 0.0) for col in HM_OTHER_CHEM_COLUMNS)
        )
        snapshot["observed_slag_rate_kg_per_thm"] = self._fetch_observed_slag_rate(
            start, end, warnings
        )
        self._last_hm_slag_diagnostics = {
            "source": snapshot.get("source"),
            "sample_timestamp": snapshot.get("sample_timestamp", ""),
            "start_time": snapshot.get("start_time", self._iso(start)),
            "end_time": snapshot.get("end_time", self._iso(end)),
            "n_rows_used": snapshot.get("n_rows_used", 0),
            "hm_fe_pct_for_target": snapshot.get("hm_fe_pct_for_target", 0.0),
            "observed_slag_rate_kg_per_thm": snapshot.get(
                "observed_slag_rate_kg_per_thm", 0.0
            ),
            "values": {
                col: snapshot.get(col, 0.0)
                for col in HM_ALL_CHEM_COLUMNS
                if col in snapshot
            },
            "warnings": list(warnings),
        }
        return snapshot

    def _hm_slag_snapshot_from_history(
        self, mode: str, window_days: int, warnings: list[str]
    ) -> dict[str, Any]:
        history_df, history_warnings = self.get_history_frame()
        warnings.extend(history_warnings)
        if history_df.empty:
            return {"warnings": warnings, "source": None, "n_rows_used": 0}

        df = history_df.copy()
        if "time" in df.columns:
            parsed_time = pd.to_datetime(df["time"], errors="coerce")
            df = df.assign(_bmo_time=parsed_time).dropna(subset=["_bmo_time"])
            cutoff = pd.Timestamp.now(tz=parsed_time.dt.tz) - pd.Timedelta(
                days=max(1, int(window_days))
            )
            df = df[df["_bmo_time"] >= cutoff]
        if df.empty:
            return {"warnings": warnings, "source": None, "n_rows_used": 0}

        snapshot: dict[str, Any] = {
            "warnings": [
                *warnings,
                "HM chemistry is using cleaned static dataset fallback.",
            ],
            "source": f"static_dataset_{mode}",
            "n_rows_used": int(len(df)),
            "observed_slag_rate_kg_per_thm": 0.0,
            "sample_timestamp": self._iso(df.index.max())
            if isinstance(df.index, pd.DatetimeIndex)
            else "",
        }
        for col in HM_ALL_CHEM_COLUMNS:
            static_col = col.upper()
            if static_col not in df.columns:
                snapshot[col] = 0.0
                continue
            series = pd.to_numeric(df[static_col], errors="coerce")
            series = series[series > 0]
            if mode == "latest" and len(series) > 0:
                snapshot[col] = float(series.iloc[-1])
            else:
                snapshot[col] = float(series.mean()) if len(series) > 0 else 0.0

        if HM_FE_COLUMN.upper() in df.columns:
            fe_series = pd.to_numeric(df[HM_FE_COLUMN.upper()], errors="coerce")
            fe_series = fe_series[fe_series > 0]
            snapshot["hm_fe_pct_for_target"] = (
                float(fe_series.mean()) if len(fe_series) > 0 else 0.0
            )
        else:
            snapshot["hm_fe_pct_for_target"] = 0.0
        snapshot["others_pct"] = float(
            sum(snapshot.get(col, 0.0) for col in HM_OTHER_CHEM_COLUMNS)
        )
        self._last_hm_slag_diagnostics = {
            "source": snapshot.get("source"),
            "sample_timestamp": snapshot.get("sample_timestamp", ""),
            "start_time": self._iso(df.index.min())
            if isinstance(df.index, pd.DatetimeIndex)
            else "",
            "end_time": self._iso(df.index.max())
            if isinstance(df.index, pd.DatetimeIndex)
            else "",
            "n_rows_used": snapshot.get("n_rows_used", 0),
            "hm_fe_pct_for_target": snapshot.get("hm_fe_pct_for_target", 0.0),
            "observed_slag_rate_kg_per_thm": 0.0,
            "values": {
                col: snapshot.get(col, 0.0)
                for col in HM_ALL_CHEM_COLUMNS
                if col in snapshot
            },
            "warnings": list(snapshot.get("warnings", [])),
        }
        return snapshot

    def _fetch_observed_slag_rate(
        self,
        start: datetime,
        end: datetime,
        warnings: list[str],
    ) -> float:
        """
        Read daily DPR to compute observed slag rate over the window.

        DPR reports daily ``slag_generation_mt`` and ``total_hot_metal_mt``. The
        plant slag rate is their sum ratio over the window times 1000 to convert
        to kg/THM. Failures are non-blocking so the page still renders.

        Args:
             - start: datetime - Window start in UTC.
             - end: datetime - Window end in UTC.
             - warnings: list[str] - Warnings buffer to append diagnostic messages.

        Returns:
             - return float - Observed slag rate in kg/THM, or 0.0 when unavailable.
        """

        try:
            dpr_df = _fetch_offline_report("DPR", (start, end))
        except Exception as exc:
            warnings.append(f"DPR query for observed slag rate failed: {exc}")
            self._last_dpr_diagnostics = {
                "source": "offline_db",
                "start_time": self._iso(start),
                "end_time": self._iso(end),
                "row_count": 0,
                "error": str(exc),
            }
            return 0.0
        if dpr_df is None or dpr_df.empty:
            self._last_dpr_diagnostics = {
                "source": "offline_db",
                "start_time": self._iso(start),
                "end_time": self._iso(end),
                "row_count": 0,
            }
            return 0.0
        slag_col = "slag_generation_mt"
        hm_col = "total_hot_metal_mt"
        if slag_col not in dpr_df.columns or hm_col not in dpr_df.columns:
            self._last_dpr_diagnostics = {
                "source": "offline_db",
                "start_time": self._iso(start),
                "end_time": self._iso(end),
                "row_count": int(len(dpr_df)),
                "missing_columns": [slag_col, hm_col],
            }
            return 0.0
        slag_total = float(
            pd.to_numeric(dpr_df[slag_col], errors="coerce").sum(skipna=True)
        )
        hm_total = float(
            pd.to_numeric(dpr_df[hm_col], errors="coerce").sum(skipna=True)
        )
        if hm_total <= 0:
            self._last_dpr_diagnostics = {
                "source": "offline_db",
                "start_time": self._iso(dpr_df.index.min())
                if isinstance(dpr_df.index, pd.DatetimeIndex)
                else self._iso(start),
                "end_time": self._iso(dpr_df.index.max())
                if isinstance(dpr_df.index, pd.DatetimeIndex)
                else self._iso(end),
                "row_count": int(len(dpr_df)),
                "slag_generation_mt": slag_total,
                "total_hot_metal_mt": hm_total,
                "observed_slag_rate_kg_per_thm": 0.0,
            }
            return 0.0
        rate = slag_total / hm_total * 1000.0
        self._last_dpr_diagnostics = {
            "source": "offline_db",
            "start_time": self._iso(dpr_df.index.min())
            if isinstance(dpr_df.index, pd.DatetimeIndex)
            else self._iso(start),
            "end_time": self._iso(dpr_df.index.max())
            if isinstance(dpr_df.index, pd.DatetimeIndex)
            else self._iso(end),
            "row_count": int(len(dpr_df)),
            "slag_generation_mt": slag_total,
            "total_hot_metal_mt": hm_total,
            "observed_slag_rate_kg_per_thm": rate,
        }
        return rate

    def get_history_frame(
        self, online_lag_hours: int = 0
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Load historical process data used for model lag features.

        The fuel-cost model uses lagged operating and burden features. This
        method loads the configured static history frame and converts data-load
        failures into warnings that can be shown in diagnostics.

        Args:
             - None

        Returns:
             - return tuple[pd.DataFrame, list[str]] - History frame and warnings.
        """

        warnings: list[str] = []
        try:
            static_path = self._dataset_service.resolve_static_path()
            if not static_path.exists():
                self._last_history_diagnostics = {
                    "source": "static_csv",
                    "path": str(static_path),
                    "exists": False,
                    "rows": 0,
                    "columns": 0,
                }
                return pd.DataFrame(), [
                    f"Static dataset file not found at {static_path}."
                ]
            df = pd.read_csv(static_path, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
            df = df[~df.index.isna()].sort_index()
            online_df, online_warnings = self._fetch_recent_online_context(
                int(online_lag_hours)
            )
            charge_df, charge_warnings = self._fetch_recent_charge_context(
                int(online_lag_hours)
            )
            warnings.extend(online_warnings)
            warnings.extend(charge_warnings)
            df = self._append_online_context(df, online_df)
            df = self._append_online_context(df, charge_df)
            df = self._repair_material_quantity_totals(df)
            live_context_end = online_df.index.max() if not online_df.empty else None
            if live_context_end is not None:
                df = df.loc[df.index <= live_context_end]
            if df.empty:
                warnings.append("Static dataset loaded but empty.")
            self._last_history_diagnostics = {
                "source": "static_csv",
                "path": str(static_path),
                "exists": True,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "start_time": self._iso(df.index.min()) if not df.empty else "",
                "end_time": self._iso(df.index.max()) if not df.empty else "",
                "latest_timestamp": self._iso(df.index.max()) if not df.empty else "",
                "online_lag_hours": int(online_lag_hours),
                "online_rows": int(len(online_df)),
                "charge_context_rows": int(len(charge_df)),
                "live_context_cap": self._iso(live_context_end)
                if live_context_end is not None
                else "",
                "warnings": list(warnings),
            }
            return df, warnings
        except Exception as exc:
            warnings.append(f"Could not load static dataset: {exc}")
            self._last_history_diagnostics = {
                "source": "static_csv",
                "exists": False,
                "rows": 0,
                "columns": 0,
                "error": str(exc),
            }
            return pd.DataFrame(), warnings

    def get_process_context(
        self,
        history_df: pd.DataFrame | None = None,
    ) -> tuple[dict[str, float], list[str]]:
        """
        Extract latest numeric process context from the history frame.

        Current operating values help fill non-burden model features before lag
        resolution. Non-numeric fields are ignored because the model service only
        accepts numeric feature payload values.

        Args:
             - None

        Returns:
             - return tuple[dict[str, float], list[str]] - Latest process values and warnings.
        """

        warnings: list[str] = []
        if history_df is None:
            history_df, warnings = self.get_history_frame()
        if history_df.empty:
            self._last_process_diagnostics = {
                "source": "static_csv",
                "latest_timestamp": "",
                "field_count": 0,
                "values": {},
                "warnings": list(warnings),
            }
            return {}, warnings
        history_df = self._repair_material_quantity_totals(history_df)
        latest = history_df.iloc[-1]
        process_context: dict[str, float] = {}
        for key, value in latest.items():
            if pd.isna(value):
                continue
            try:
                process_context[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        self._last_process_diagnostics = {
            "source": "static_csv+online_influx"
            if self._last_online_context_diagnostics.get("rows", 0)
            else "static_csv",
            "latest_timestamp": self._iso(history_df.index.max()),
            "field_count": int(len(process_context)),
            "values": dict(process_context),
            "warnings": list(warnings),
        }
        return process_context, warnings

    def get_flux_inputs(
        self, mode: str = "latest", window_days: int | None = None
    ) -> tuple[list[FluxInput], list[str]]:
        """Build flux inputs from charge quantities and flux chemistry tables."""

        cfg = self.settings.get("data_sources", {})
        mode = "avg" if mode == "avg" else "latest"
        days = int(window_days or cfg.get("chemistry_time_range_days", 30))
        query_days = (
            max(days, int(cfg.get("latest_usage_lookback_days", 365)))
            if mode == "latest"
            else days
        )
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=query_days)
        warnings: list[str] = []

        flux_columns = self._charge_columns_by_group().get("flux", [])
        if not flux_columns:
            warnings.append("No flux charge columns are configured.")
            self._last_flux_diagnostics = {
                "source": "offline_feed.charge_data+offline_feed.flux_chemistry",
                "mode": mode,
                "rows": [],
                "warnings": list(warnings),
            }
            return [], warnings

        try:
            charge_df = self._frame_with_time_column(
                _fetch_offline_data(
                    table_name="offline_feed.charge_data",
                    time_range=(start_time, end_time),
                    query_type="raw",
                    columns=["date_time", *flux_columns],
                )
            )
        except Exception as exc:
            warnings.append(f"Flux quantity query failed: {exc}")
            charge_df = pd.DataFrame()

        quantity_by_code: dict[str, float] = {}
        charge_timestamp = ""
        if not charge_df.empty:
            present = [column for column in flux_columns if column in charge_df.columns]
            if present:
                numeric = charge_df[present].apply(pd.to_numeric, errors="coerce")
                valid = numeric.fillna(0.0).ne(0.0).any(axis=1)
                if valid.any():
                    row = charge_df.loc[valid].iloc[-1]
                    charge_timestamp = self._iso(row.get("time"))
                    for column in present:
                        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
                        if pd.notna(value) and float(value) != 0.0:
                            quantity_by_code[
                                self._material_code_from_quantity_column(column)
                            ] = float(value)
        if not quantity_by_code:
            warnings.append("No non-zero flux quantities found in recent charge data.")

        material_codes = set(quantity_by_code)
        chemistry_rows: dict[str, pd.Series] = {}
        chemistry_meta: dict[str, dict[str, Any]] = {}
        chem_field_map = {
            "moisture_pct": "tm",
            "sio2_pct": "sio2",
            "al2o3_pct": "al2o3",
            "cao_pct": "cao",
            "mgo_pct": "mgo",
            "fe2o3_pct": "fe2o3",
            "mno_pct": "mno",
            "tio2_pct": "tio2",
            "na2o_pct": "na2o",
            "k2o_pct": "k2o",
            "caf2_pct": "caf2",
            "p_pct": "p",
            "s_pct": "s",
            "zn_pct": "zn",
            "loi_pct": "loi",
        }
        required_flux_columns = ["tm", "sio2", "al2o3", "cao", "mgo", "fe2o3", "loi"]
        if material_codes:
            try:
                chem_df = self._frame_with_time_column(
                    _fetch_offline_data(
                        table_name="offline_feed.flux_chemistry",
                        time_range=(start_time, end_time),
                        query_type="raw",
                    )
                )
            except Exception as exc:
                warnings.append(f"Flux chemistry query failed: {exc}")
                chem_df = pd.DataFrame()

            if not chem_df.empty and "material_code" in chem_df.columns:
                chem_df = chem_df[chem_df["material_code"].isin(material_codes)]
                for material_code, group in chem_df.groupby("material_code", sort=False):
                    group = group.sort_values("time") if "time" in group.columns else group
                    source_group = group
                    if mode == "avg":
                        snapshot = self._average_non_zero(source_group)
                        source = "offline_db_avg_non_zero"
                    else:
                        if charge_timestamp and "time" in group.columns:
                            charge_time = pd.to_datetime(charge_timestamp, utc=True)
                            prior = group[group["time"] <= charge_time]
                            if not prior.empty:
                                source_group = prior
                        snapshot = self._latest_row_snapshot(
                            source_group, required_flux_columns
                        )
                        if (
                            not snapshot.empty
                            and snapshot.name is not None
                            and snapshot.name in source_group.index
                        ):
                            source_group = source_group.loc[[snapshot.name]]
                        source = "offline_db_latest"
                    material_code = str(material_code)
                    chemistry_rows[material_code] = snapshot
                    chemistry_meta[material_code] = {
                        "source": source,
                        "table": "offline_feed.flux_chemistry",
                        "rows_used": int(len(source_group)),
                        "sample_timestamp": self._iso(source_group["time"].max())
                        if "time" in source_group.columns and not source_group.empty
                        else "",
                    }

        inputs: list[FluxInput] = []
        diagnostic_rows: list[dict[str, Any]] = []
        for material_code, quantity_mt in quantity_by_code.items():
            snapshot = chemistry_rows.get(material_code, pd.Series(dtype=float))
            if snapshot.empty:
                warnings.append(f"Flux chemistry unavailable for {material_code}.")
                continue

            values = {}
            for attr, column in chem_field_map.items():
                value = snapshot.get(column, 0.0)
                values[attr] = 0.0 if pd.isna(value) else float(value)
            flux = FluxInput(
                flux_id=material_code,
                display_name=material_code,
                enabled=True,
                wet_qty_mt=float(quantity_mt),
                **values,
            )
            inputs.append(flux)
            diagnostic_rows.append(
                {
                    "material_code": material_code,
                    "quantity_mt": float(quantity_mt),
                    "charge_timestamp": charge_timestamp,
                    **chemistry_meta.get(material_code, {}),
                    **values,
                }
            )

        self._last_flux_diagnostics = {
            "source": "offline_feed.charge_data+offline_feed.flux_chemistry",
            "mode": mode,
            "window_days": days,
            "latest_timestamp": charge_timestamp,
            "rows": diagnostic_rows,
            "warnings": list(warnings),
        }
        return inputs, warnings

    def get_data_diagnostics(self) -> dict[str, Any]:
        """Return source diagnostics captured by the latest BMO data reads."""

        return {
            "stock": dict(self._last_stock_diagnostics),
            "chemistry": dict(self._last_chemistry_diagnostics),
            "hm_slag": dict(self._last_hm_slag_diagnostics),
            "dpr": dict(self._last_dpr_diagnostics),
            "history": dict(self._last_history_diagnostics),
            "process": dict(self._last_process_diagnostics),
            "flux": dict(self._last_flux_diagnostics),
            "online_context": dict(self._last_online_context_diagnostics),
            "charge_context": dict(self._last_charge_context_diagnostics),
            "pellet_usage": dict(self._last_pellet_usage_diagnostics),
            "charge_mix": dict(self._last_charge_mix_diagnostics),
            "manual_blend": dict(self._last_manual_blend_diagnostics),
        }

    def build_ore_inputs(
        self, mode: str = "latest", window_days: int | None = None
    ) -> tuple[list[OreInput], dict[str, Any]]:
        """
        Build typed BMO ore inputs and data-source diagnostics.

        This joins stock, price/share configuration, and chemistry snapshots
        into the single typed structure expected by LP, DE, model feature
        construction, and the Streamlit editor.

        Args:
             - mode: str - Chemistry snapshot mode, either latest or average.
             - window_days: int | None - Number of days of chemistry data to query.

        Returns:
             - return tuple[list[OreInput], dict[str, Any]] - Ore inputs and diagnostics.
        """

        stock_map, stock_warnings = self.get_stock_snapshot()
        chemistry_map, chem_warnings = self.get_chemistry_snapshot(
            mode=mode, window_days=window_days
        )

        stock_rows = {
            str(row.get("ore_id")): row
            for row in self._last_stock_diagnostics.get("material_rows", [])
        }
        chemistry_rows = {
            str(row.get("ore_id")): row
            for row in self._last_chemistry_diagnostics.get("material_rows", [])
        }

        ores: list[OreInput] = []
        for ore_cfg in self._ores_cfg:
            ore_id = str(ore_cfg.get("id"))
            stock_meta = stock_rows.get(ore_id, {})
            chemistry_meta = chemistry_rows.get(ore_id, {})
            ore = OreInput(
                ore_id=ore_id,
                display_name=str(ore_cfg.get("display_name", ore_id)),
                stock_mt=float(stock_map.get(ore_id, 0.0)),
                price_rs_per_mt=float(ore_cfg.get("price_rs_per_mt", 0.0)),
                min_share_pct=float(ore_cfg.get("min_share_pct", 0.0)),
                max_share_pct=float(ore_cfg.get("max_share_pct", 100.0)),
                chemistry=chemistry_map.get(
                    ore_id,
                    OreChemistry(fe_t_pct=0.0),
                ),
                metadata={
                    "material_key": ore_cfg.get("material_key"),
                    "fallback_chemistry": ore_cfg.get("fallback_chemistry", {}),
                    "stock_source": stock_meta.get("source", ""),
                    "stock_timestamp": stock_meta.get("timestamp", ""),
                    "chemistry_source": chemistry_meta.get("source", ""),
                    "chemistry_sample_timestamp": chemistry_meta.get(
                        "sample_timestamp", ""
                    ),
                    "chemistry_latest_used_time": chemistry_meta.get(
                        "latest_used_time", ""
                    ),
                    "chemistry_table": chemistry_meta.get("table", ""),
                    "chemistry_rows_used": chemistry_meta.get("rows_used", 0),
                },
            )
            ores.append(ore)

        diagnostics = {
            "warnings": [*stock_warnings, *chem_warnings],
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "ore_count": len(ores),
            "chem_mode": mode,
            "chem_window_days": int(
                window_days or self.settings.get("chemistry_window_days", 30)
            ),
            "mapping_file": str(self.mapping_path),
            "setting_file": str(self.setting_path),
            "ore_preview": [asdict(ore) for ore in ores[:3]],
        }
        return ores, diagnostics
