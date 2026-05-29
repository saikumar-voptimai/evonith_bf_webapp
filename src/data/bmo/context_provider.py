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

import pandas as pd
import yaml

from domain.optimization_runtime import DatasetContextService, build_runtime_config
from utils.bmo.types import OreChemistry, OreInput

try:
    from furnace_data.influx.offline import fetch_offline_data as _fetch_offline_data
except Exception:  # pragma: no cover - environment dependent import fallback
    _fetch_offline_data = None


def _fetch_offline_data_safe(
    *, measurement: str, time_range: Any, database: str
) -> pd.DataFrame:
    """
    Fetch offline furnace data through the optional furnace_data dependency.

    The import can be unavailable in lightweight test environments, so this
    wrapper gives the caller one predictable failure path while keeping the
    provider code independent of the concrete Influx client implementation.

    Args:
         - measurement: str - Influx measurement name to query.
         - time_range: Any - Query time range accepted by fetch_offline_data.
         - database: str - Offline database or bucket name.

    Returns:
         - return pd.DataFrame - Data frame returned by the offline data query.
    """

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
            refresh_enabled=bool(
                self.runtime_cfg["dataset"].get("refresh_enabled", False)
            ),
            refresh_rm_choice=str(
                self.runtime_cfg["dataset"].get("refresh_rm_choice", "RM Charge")
            ),
        )

    def _query_latest_row(
        self, measurement: str, bucket: str, time_range: str
    ) -> tuple[pd.Series | None, str | None]:
        """
        Query a measurement and return its latest row.

        Stock snapshots need only the newest row from the configured source.
        Query errors are returned as warning text instead of raising so the page
        can continue with fallback planning values.

        Args:
             - measurement: str - Influx measurement name to query.
             - bucket: str - Offline database or bucket name.
             - time_range: str - Relative query time range for stock snapshots.

        Returns:
             - return tuple[pd.Series | None, str | None] - Latest row and error text.
        """

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

    def get_stock_snapshot(self) -> tuple[dict[str, float], list[str]]:
        """
        Build the latest stock snapshot for all configured BMO ores.

        Live stock is preferred because LP and DE enforce stock as hard bounds.
        When the source is unavailable, planning fallback stock is generated
        from configured share caps so the optimizer can still run.

        Args:
             - None

        Returns:
             - return tuple[dict[str, float], list[str]] - Stock map and warnings.
        """

        cfg = self.settings.get("data_sources", {})
        measurement = str(cfg.get("stock_measurement", "rm_stock"))
        bucket = str(cfg.get("stock_bucket", "bf2_evonith_offline_utc"))
        time_range = str(cfg.get("stock_time_range", "last 1 week"))

        row, err = self._query_latest_row(measurement, bucket, time_range)
        warnings: list[str] = []
        if err:
            warnings.append(err)
            row = None

        stock_map: dict[str, float] = {}
        fallback_count = 0
        for ore_cfg in self._ores_cfg:
            ore_id = str(ore_cfg.get("id"))
            stock_field = str(ore_cfg.get("stock_field", ""))
            value = None
            if row is not None and stock_field in row:
                try:
                    value = float(row.get(stock_field))
                except (TypeError, ValueError):
                    value = None
                if value is not None and pd.isna(value):
                    value = None
            if value is None:
                value = self._stock_fallback_mt(ore_cfg)
                fallback_count += 1
            stock_map[ore_id] = max(0.0, value)

        if fallback_count:
            warnings.append(
                (
                    f"Stock unavailable for {fallback_count} ore(s); using planning "
                    "stock fallback from configured share caps."
                )
            )
        return stock_map, warnings

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
        measurement = str(cfg.get("chemistry_measurement", "rm_updated_data"))
        bucket = str(cfg.get("chemistry_bucket", "bf2_evonith_offline_utc"))
        days = int(window_days or cfg.get("chemistry_time_range_days", 30))
        end_time = datetime.now(timezone.utc)
        time_range = (end_time - timedelta(days=days), end_time)

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
            warnings.append(
                "Chemistry data unavailable, using fallback chemistry values."
            )
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

        return chemistry_by_ore, warnings

    def get_history_frame(self) -> tuple[pd.DataFrame, list[str]]:
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
            df = self._dataset_service.load_history()
            if df.empty:
                warnings.append("Static dataset loaded but empty.")
            return df, warnings
        except Exception as exc:
            warnings.append(f"Could not load static dataset: {exc}")
            return pd.DataFrame(), warnings

    def get_process_context(self) -> tuple[dict[str, float], list[str]]:
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
            "chem_window_days": int(
                window_days or self.settings.get("chemistry_window_days", 30)
            ),
            "mapping_file": str(self.mapping_path),
            "setting_file": str(self.setting_path),
            "ore_preview": [asdict(ore) for ore in ores[:3]],
        }
        return ores, diagnostics
