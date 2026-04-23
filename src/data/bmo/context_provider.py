from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from domain.bmo.types import OreChemistry, OreInput

try:
    from data.retrieval import fetch_offline_data as _fetch_offline_data
except Exception:  # pragma: no cover - environment dependent import fallback
    _fetch_offline_data = None


def _fetch_offline_data_safe(*, measurement: str, time_range: Any, database: str) -> pd.DataFrame:
    if _fetch_offline_data is None:
        raise RuntimeError(
            "fetch_offline_data is unavailable. Ensure furnace_data dependency is installed."
        )
    return _fetch_offline_data(
        measurement=measurement,
        time_range=time_range,
        database=database,
    )


def _resolve_repo_path(path_str: str) -> Path:
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
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        return data or {}


class EvonithBmoContextProvider:
    def __init__(
        self,
        setting_path: str = "src/config/setting_bmo.yml",
        mapping_path: str = "src/config/bmo_ore_mapping.yml",
    ) -> None:
        self.setting_path = _resolve_repo_path(setting_path)
        self.mapping_path = _resolve_repo_path(mapping_path)

        self.settings = _load_yaml(self.setting_path).get("bmo", {})
        self.mapping = _load_yaml(self.mapping_path)

        self._ores_cfg = self.mapping.get("ores", [])
        self._chem_field_map = self.mapping.get("chemistry_field_map", {})

    def get_raw_mapping(self) -> dict[str, Any]:
        return {"settings": self.settings, "mapping": self.mapping}

    def _query_latest_row(
        self, measurement: str, bucket: str, time_range: str
    ) -> tuple[pd.Series | None, str | None]:
        try:
            df = _fetch_offline_data_safe(
                measurement=measurement,
                time_range=time_range,
                database=bucket,
            )
        except Exception as exc:
            return None, f"Query failed for {measurement}: {exc}"

        if df is None or df.empty:
            return None, f"No data returned for {measurement}."

        df = df.sort_index()
        row = df.iloc[-1].copy()
        return row, None

    def get_stock_snapshot(self) -> tuple[dict[str, float], list[str]]:
        cfg = self.settings.get("data_sources", {})
        measurement = str(cfg.get("stock_measurement", "rm_stock"))
        bucket = str(cfg.get("stock_bucket", "Test"))
        time_range = str(cfg.get("stock_time_range", "last 7 days"))

        row, err = self._query_latest_row(measurement, bucket, time_range)
        warnings: list[str] = []
        if err:
            warnings.append(err)
            row = None

        stock_map: dict[str, float] = {}
        for ore_cfg in self._ores_cfg:
            ore_id = str(ore_cfg.get("id"))
            stock_field = str(ore_cfg.get("stock_field", ""))
            value = 0.0
            if row is not None and stock_field in row:
                try:
                    value = float(row.get(stock_field) or 0.0)
                except (TypeError, ValueError):
                    value = 0.0
            stock_map[ore_id] = max(0.0, value)
        return stock_map, warnings

    def get_chemistry_snapshot(
        self, mode: str = "latest", window_days: int | None = None
    ) -> tuple[dict[str, OreChemistry], list[str]]:
        cfg = self.settings.get("data_sources", {})
        measurement = str(cfg.get("chemistry_measurement", "rm_updated_data"))
        bucket = str(cfg.get("chemistry_bucket", "bf2_evonith_offline_utc"))
        days = int(window_days or cfg.get("chemistry_time_range_days", 30))
        time_range = f"last {days} days"

        warnings: list[str] = []
        chemistry_by_ore: dict[str, OreChemistry] = {}

        df = None
        try:
            df = _fetch_offline_data_safe(
                measurement=measurement,
                time_range=time_range,
                database=bucket,
            )
        except Exception as exc:
            warnings.append(f"Chemistry query failed for {measurement}: {exc}")

        if df is None or df.empty:
            warnings.append("Chemistry data unavailable, using fallback chemistry values.")
            df = pd.DataFrame()

        df = df.sort_index()
        if mode == "avg" and not df.empty:
            snapshot = df.mean(numeric_only=True)
        elif not df.empty:
            # latest non-null by column
            latest_values = {}
            for column in df.columns:
                series = pd.to_numeric(df[column], errors="coerce").dropna()
                if len(series) > 0:
                    latest_values[column] = float(series.iloc[-1])
            snapshot = pd.Series(latest_values)
        else:
            snapshot = pd.Series(dtype=float)

        for ore_cfg in self._ores_cfg:
            ore_id = str(ore_cfg.get("id"))
            material_key = str(ore_cfg.get("material_key", ""))
            fallback = ore_cfg.get("fallback_chemistry", {}) or {}
            chem_values: dict[str, float] = {}

            for chem_attr, chem_suffix in self._chem_field_map.items():
                field = f"{material_key}_pct_{chem_suffix}"
                val = snapshot.get(field, None)
                if val is None or pd.isna(val):
                    val = fallback.get(chem_attr, 0.0)
                try:
                    chem_values[chem_attr] = float(val)
                except (TypeError, ValueError):
                    chem_values[chem_attr] = float(fallback.get(chem_attr, 0.0))

            chemistry_by_ore[ore_id] = OreChemistry(
                fe_t_pct=float(chem_values.get("fe_t_pct", 0.0)),
                feo_pct=float(chem_values.get("feo_pct", 0.0)),
                sio2_pct=float(chem_values.get("sio2_pct", 0.0)),
                al2o3_pct=float(chem_values.get("al2o3_pct", 0.0)),
                cao_pct=float(chem_values.get("cao_pct", 0.0)),
                mgo_pct=float(chem_values.get("mgo_pct", 0.0)),
                mno_pct=float(chem_values.get("mno_pct", 0.0)),
                tio2_pct=float(chem_values.get("tio2_pct", 0.0)),
                p_pct=float(chem_values.get("p_pct", 0.0)),
            )

        return chemistry_by_ore, warnings

    def get_history_frame(self) -> tuple[pd.DataFrame, list[str]]:
        static_path = str(self.settings.get("data_sources", {}).get("static_dataset_path", ""))
        csv_path = _resolve_repo_path(static_path)

        warnings: list[str] = []
        if not csv_path.exists():
            warnings.append(f"Static dataset not found at {csv_path}.")
            return pd.DataFrame(), warnings

        try:
            df = pd.read_csv(csv_path, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
            df = df.sort_index()
            return df, warnings
        except Exception as exc:
            warnings.append(f"Could not load static dataset: {exc}")
            return pd.DataFrame(), warnings

    def get_process_context(self) -> tuple[dict[str, float], list[str]]:
        history_df, warnings = self.get_history_frame()
        if history_df.empty:
            return {}, warnings
        latest = history_df.iloc[-1]
        process_context: dict[str, float] = {}
        for key, value in latest.items():
            try:
                process_context[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return process_context, warnings

    def build_ore_inputs(
        self, mode: str = "latest", window_days: int | None = None
    ) -> tuple[list[OreInput], dict[str, Any]]:
        stock_map, stock_warnings = self.get_stock_snapshot()
        chemistry_map, chem_warnings = self.get_chemistry_snapshot(
            mode=mode, window_days=window_days
        )

        ores: list[OreInput] = []
        for ore_cfg in self._ores_cfg:
            ore_id = str(ore_cfg.get("id"))
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
                    "stock_field": ore_cfg.get("stock_field"),
                    "fallback_chemistry": ore_cfg.get("fallback_chemistry", {}),
                },
            )
            ores.append(ore)

        diagnostics = {
            "warnings": [*stock_warnings, *chem_warnings],
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "ore_count": len(ores),
            "chem_mode": mode,
            "chem_window_days": int(window_days or self.settings.get("chemistry_window_days", 30)),
            "mapping_file": str(self.mapping_path),
            "setting_file": str(self.setting_path),
            "ore_preview": [asdict(ore) for ore in ores[:3]],
        }
        return ores, diagnostics
