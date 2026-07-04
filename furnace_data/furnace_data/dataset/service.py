"""4-step fetch pipeline for the furnace ML/operations dataset.

Provides
--------
DatasetService   Dataclass with four fetch methods:
                   fetch / fetch_step1  — historical static ML dataset
                   fetch_rm_data / fetch_step2 — RM Charge / RM DPR dataset
                   fetch_hotmetal_hourly       — HM & Slag, interpolated hourly
                   fetch_distribution_data     — burden distribution from PostgreSQL
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from furnace_data.relational import (
    BurdenHistoryRepository,
    build_relational_engine,
    build_relational_session_factory,
)

from furnace_data.config import load_config
from furnace_data.offline import fetch_offline_data as fetch_database_offline_data

config = load_config("setting_ds_dv.yml")

_OFFLINE_RM_TABLES = {
    "charge": "offline_feed.charge_data",
    "dpr": "offline_feed.dpr_data",
}
_OFFLINE_RM_QUANTITY_VIEWS = {
    "charge": "offline_feed.v_charge_material_quantities",
    "dpr": "offline_feed.v_dpr_material_quantities",
}
_OFFLINE_STATIC_ML_TABLE = "offline_feed.historical_static_ml_dataset"
_OFFLINE_STRENGTH_TABLE = "offline_feed.raw_material_strength_analysis"
_OFFLINE_HM_SLAG_TABLE = "offline_feed.hot_metal_slag_analysis"
_STRENGTH_COMPAT_COLUMNS = ("ai", "ti", "ri", "rdi")


@dataclass
class DatasetService:
    """4-step fetch pipeline for the furnace dataset.

    Step 1: Fetch the historical static ML dataset from PostgreSQL.
    Step 2: Fetch RM Charge or RM DPR dataset (post-cutoff).
    Step 3: Fetch Hot Metal & Slag and interpolate to a regular hourly grid.
    Step 4: Fetch burden distribution records from PostgreSQL and pivot to daily rows.

    Attributes:
        local_tz:             Timezone string for index localisation.
        cutoff_date:          Date separating historical static and generated data.
        db_url:               PostgreSQL connection URL for step 4.
    """

    local_tz: str = config["ml_dataset"].get("local_tz", "Asia/Kolkata")

    cutoff_date: date = date.fromisoformat(config["ml_dataset"]["cutoff_date"])

    # Intentionally NOT frozen at class-definition time: field(default_factory=str)
    # means the env var is not captured on import.  _get_engine() re-reads at
    # call time via resolve_database_url's fallback so a late load_dotenv() works.
    db_url: str = ""

    def _get_engine(self):
        # Always prefer the live env var; self.db_url is only used when explicitly
        # set to a non-empty string (e.g. in tests or the API sidecar).
        url = self.db_url or os.getenv("DATABASE_URL", "")
        return build_relational_engine(url)

    # --------------- TIMEZONE CONVERSIONS ---------------

    def _to_utc_window(self, start_date: date, end_date: date):
        tz = ZoneInfo(self.local_tz)
        start_dt = datetime.combine(start_date, time.min).replace(tzinfo=tz).astimezone(timezone.utc)
        end_dt = datetime.combine(end_date, time.max).replace(tzinfo=tz).astimezone(timezone.utc)
        return start_dt, end_dt

    def _normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        tz = ZoneInfo(self.local_tz)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(tz).tz_localize(None)
        df.index.name = "time"
        return df.sort_index()

    @staticmethod
    def _sum_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series | None:
        existing = [col for col in columns if col in df.columns]
        if not existing:
            return None
        values = df[existing].apply(pd.to_numeric, errors="coerce")
        return values.sum(axis=1, min_count=1)

    @staticmethod
    def _collapse_strength_frame(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        out = pd.DataFrame(index=df.index)
        for idx in range(1, 5):
            name_col = f"property_{idx}_name"
            value_col = f"property_{idx}"
            if name_col not in df.columns or value_col not in df.columns:
                continue

            names = df[name_col].astype("string").str.strip().str.lower()
            values = pd.to_numeric(df[value_col], errors="coerce")
            for target_col in _STRENGTH_COMPAT_COLUMNS:
                if target_col not in out.columns:
                    out[target_col] = pd.NA
                matched = names == target_col
                out.loc[matched, target_col] = values[matched].to_numpy()

        for target_col in _STRENGTH_COMPAT_COLUMNS:
            if target_col in df.columns:
                values = pd.to_numeric(df[target_col], errors="coerce")
                if target_col not in out.columns:
                    out[target_col] = values
                else:
                    out[target_col] = out[target_col].combine_first(values)

        if out.empty:
            return pd.DataFrame()
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.groupby(level=0).first()
        return out.dropna(how="all")

    def _add_offline_ml_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ML-compatible aggregate aliases for offline report table columns."""
        if df.empty:
            return df

        df = df.copy()
        aliases = {
            "sinter_mt": [f"sinter_{i}_mt" for i in range(1, 5)],
            "pellet_mt": [f"pellet_{i}_mt" for i in range(1, 3)],
            "ore_mt": [f"ore_{i}_mt" for i in range(1, 13)],
            "nutcoke_prime_mt": [f"nut_coke_{i}_mt" for i in range(1, 3)],
            "coke_mt": [f"coke_{i}_mt" for i in range(1, 3)],
            "coke_on_mt": ["coke_1_mt"],
            "coke_off_mt": ["coke_2_mt"],
            "flux_mt": [f"flux_{i}_mt" for i in range(1, 4)],
            "pci2_mt": ["pci_mt"],
        }
        for alias, columns in aliases.items():
            summed = self._sum_existing_columns(df, columns)
            if summed is not None:
                df[alias] = summed

        rm_hm_aliases = {
            "ai": "sinter_cold_strength_ai",
            "ti": "sinter_cold_strength_ti",
            "ri": "sinter_hot_strength_ri",
            "rdi": "sinter_hot_strength_rdi",
        }
        for source_col, alias in rm_hm_aliases.items():
            if source_col in df.columns and alias not in df.columns:
                df[alias] = df[source_col]

        return df

    @staticmethod
    def _weighted_latest_before(
        quantities: pd.DataFrame,
        chemistry: pd.DataFrame,
        *,
        material_codes: set[str],
        column_map: dict[str, str | list[str]],
    ) -> pd.DataFrame:
        """Build weighted chemistry columns using latest sample before timestamp."""
        if (
            quantities.empty
            or chemistry.empty
            or "material_code" not in quantities.columns
            or "material_code" not in chemistry.columns
        ):
            return pd.DataFrame()

        q = quantities[quantities["material_code"].isin(material_codes)].copy()
        c = chemistry[chemistry["material_code"].isin(material_codes)].copy()
        if q.empty or c.empty:
            return pd.DataFrame()

        q["quantity"] = pd.to_numeric(q["quantity"], errors="coerce")
        q = q.dropna(subset=["quantity"])
        if q.empty:
            return pd.DataFrame()

        merged_parts = []
        for material_code, q_group in q.groupby("material_code", sort=False):
            c_group = c[c["material_code"] == material_code]
            if c_group.empty:
                continue
            q_group = q_group.sort_values("time")
            c_group = c_group.sort_values("time")
            merged_parts.append(
                pd.merge_asof(
                    q_group,
                    c_group,
                    on="time",
                    direction="backward",
                    suffixes=("", "_chem"),
                )
            )

        if not merged_parts:
            return pd.DataFrame()

        merged = pd.concat(merged_parts, ignore_index=True)
        out = pd.DataFrame(index=pd.DatetimeIndex(sorted(q["time"].unique()), name="time"))

        for source_col, target_cols in column_map.items():
            if source_col not in merged.columns:
                continue
            values = pd.to_numeric(merged[source_col], errors="coerce")
            valid = values.notna() & merged["quantity"].notna()
            if not valid.any():
                continue
            weighted_sum = (values[valid] * merged.loc[valid, "quantity"]).groupby(
                merged.loc[valid, "time"]
            ).sum()
            weight = merged.loc[valid].groupby("time")["quantity"].sum()
            result = weighted_sum / weight.replace(0, pd.NA)
            if isinstance(target_cols, str):
                target_cols = [target_cols]
            for target_col in target_cols:
                out[target_col] = result

        return out.dropna(how="all")

    @staticmethod
    def _latest_before_context(
        times: pd.Series,
        chemistry: pd.DataFrame,
        *,
        material_codes: set[str],
        column_map: dict[str, str | list[str]],
    ) -> pd.DataFrame:
        """Build latest-before chemistry columns for non-charged context materials."""
        if (
            chemistry.empty
            or not material_codes
            or "material_code" not in chemistry.columns
        ):
            return pd.DataFrame()

        t = pd.DataFrame({"time": pd.Series(times).dropna().drop_duplicates()})
        c = chemistry[chemistry["material_code"].isin(material_codes)].copy()
        if t.empty or c.empty:
            return pd.DataFrame()

        t = t.sort_values("time")
        c = c.sort_values("time").drop_duplicates("time", keep="last")
        merged = pd.merge_asof(t, c, on="time", direction="backward")
        out = pd.DataFrame(index=pd.DatetimeIndex(t["time"], name="time"))

        for source_col, target_cols in column_map.items():
            if source_col not in merged.columns:
                continue
            values = pd.to_numeric(merged[source_col], errors="coerce")
            if isinstance(target_cols, str):
                target_cols = [target_cols]
            for target_col in target_cols:
                out[target_col] = values.to_numpy()

        return out.dropna(how="all")

    @staticmethod
    def _table_with_time_column(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.reset_index()
        if "index" in out.columns and "time" not in out.columns:
            out = out.rename(columns={"index": "time"})
        out["time"] = pd.to_datetime(out["time"])
        if out["time"].dt.tz is not None:
            out["time"] = out["time"].dt.tz_convert(None)
        return out.sort_values("time")

    def _fetch_offline_weighted_chemistry(
        self,
        *,
        mode: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        quantity_table = _OFFLINE_RM_QUANTITY_VIEWS[mode]
        quantities = self._fetch_offline_table(quantity_table, start_dt, end_dt)
        if quantities.empty:
            return pd.DataFrame()
        quantities = self._table_with_time_column(quantities)

        chem_start = start_dt - timedelta(days=30)
        outputs: list[pd.DataFrame] = []

        ore = self._table_with_time_column(
            self._fetch_offline_table("offline_feed.ore_chemistry", chem_start, end_dt)
        )
        outputs.append(
            self._weighted_latest_before(
                quantities,
                ore,
                material_codes={code for code in quantities["material_code"].dropna().unique() if str(code).startswith("ore_")},
                column_map={
                    "fe_t": "ore_fe_total_pct",
                    "loi": "ore_loi_pct",
                    "tm": "ore_tm_pct",
                    "mgo": "ore_mgo_pct",
                    "na2o": "ore_na2o_pct",
                    "p": "ore_p_pct",
                    "tio2": "ore_tio2_pct",
                    "al2o3": "ore_al2o3_pct",
                    "sio2": "ore_sio2_pct",
                    "mno": "ore_mno_pct",
                    "cao": "ore_cao_pct",
                    "k2o": "ore_k2o_pct",
                },
            )
        )
        outputs.append(
            self._weighted_latest_before(
                quantities,
                ore,
                material_codes={code for code in quantities["material_code"].dropna().unique() if str(code).startswith("pellet_")},
                column_map={
                    "fe_t": "pellet_fe2o3_pct",
                    "sio2": "pellet_sio2_pct",
                    "al2o3": "pellet_al2o3_pct",
                    "mgo": "pellet_mgo_pct",
                    "tio2": "pellet_tio2_pct",
                    "na2o": "pellet_na2o_pct",
                    "k2o": "pellet_k2o_pct",
                    "tm": "pellet_tm_pct",
                    "loi": "pellet_loi_pct",
                    "mno": "pellet_mno_pct",
                    "cao": "pellet_cao_pct",
                    "p": "pellet_p_pct",
                },
            )
        )

        sinter = self._table_with_time_column(
            self._fetch_offline_table("offline_feed.sinter_chemistry", chem_start, end_dt)
        )
        outputs.append(
            self._weighted_latest_before(
                quantities,
                sinter,
                material_codes={code for code in quantities["material_code"].dropna().unique() if str(code).startswith("sinter_")},
                column_map={
                    "ai": "sinter_cold_strength_ai",
                    "ti": "sinter_cold_strength_ti",
                    "ri": "sinter_hot_strength_ri",
                    "rdi": "sinter_hot_strength_rdi",
                    "p": "sinter_p_pct",
                    "sio2": "sinter_sio2_pct",
                    "mgo": "sinter_mgo_pct",
                    "feo": "sinter_feo_pct",
                    "al2o3": "sinter_al2o3_pct",
                    "na2o": "sinter_na2o_pct",
                    "tio2": "sinter_tio2_pct",
                    "fe_t": "sinter_fe_total_pct",
                    "cao": "sinter_cao_pct",
                    "k2o": "sinter_k2o_pct",
                    "basicity": "sinter_basicity",
                    "mno": "sinter_mno_pct",
                },
            )
        )

        fuel = self._table_with_time_column(
            self._fetch_offline_table("offline_feed.fuel_chemistry", chem_start, end_dt)
        )
        outputs.extend(
            [
                self._weighted_latest_before(
                    quantities,
                    fuel,
                    material_codes={code for code in quantities["material_code"].dropna().unique() if str(code).startswith("coke_")},
                    column_map={
                        "moisture": ["coke_moist_pct", "coke_im_pct"],
                        "ash": "coke_ash_pct",
                        "vm": "coke_vm_pct",
                        "fc": "coke_fc_pct",
                    },
                ),
                self._weighted_latest_before(
                    quantities,
                    fuel,
                    material_codes={code for code in quantities["material_code"].dropna().unique() if str(code).startswith("nut_coke_")},
                    column_map={
                        "moisture": ["nutcoke_moist_pct", "nutcoke_im_pct"],
                        "ash": "nutcoke_ash_pct",
                        "vm": "nutcoke_vm_pct",
                        "fc": "nutcoke_fc_pct",
                    },
                ),
                self._latest_before_context(
                    quantities["time"],
                    fuel,
                    material_codes={
                        code
                        for code in (
                            fuel["material_code"].dropna().unique()
                            if "material_code" in fuel.columns
                            else []
                        )
                        if str(code).startswith("pci_")
                    },
                    column_map={
                        "tm": "pci2_tm_pct",
                        "moisture": "pci2_im_pct",
                        "ash": "pci2_ash_pct",
                        "vm": "pci2_vm_pct",
                        "fc": "pci2_fc_pct",
                    },
                ),
            ]
        )

        flux = self._table_with_time_column(
            self._fetch_offline_table("offline_feed.flux_chemistry", chem_start, end_dt)
        )
        outputs.append(
            self._weighted_latest_before(
                quantities,
                flux,
                material_codes={code for code in quantities["material_code"].dropna().unique() if str(code).startswith("flux_")},
                column_map={
                    "tm": "flux_tm_pct",
                    "sio2": "flux_sio2_pct",
                    "fe2o3": "flux_fe2o3_pct",
                    "al2o3": "flux_al2o3_pct",
                    "loi": "flux_loi_pct",
                    "mgo": "flux_mgo_pct",
                    "cao": "flux_cao_pct",
                },
            )
        )

        frames = [frame for frame in outputs if frame is not None and not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

    def _fetch_offline_table(
        self,
        table_name: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        df = fetch_database_offline_data(
            table_name=table_name,
            time_range=(start_dt, end_dt),
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return self._normalize_timezone(df)

    # ------------------- STEP 1: HISTORICAL STATIC DATASET -------------------

    def fetch(
        self,
        start_date: date,
        end_date: date,
        allowed_columns: dict | None = None,
    ) -> pd.DataFrame:
        """Fetch the historical static ML dataset from PostgreSQL.

        Args:
            start_date:       Inclusive start date (local timezone).
            end_date:         Inclusive end date (local timezone).
            allowed_columns:  If given, restrict to these column names (keys of the dict).

        Returns:
            Time-indexed :class:`pandas.DataFrame` with the historical dataset columns.
        """
        start_dt, end_dt = self._to_utc_window(start_date, end_date)
        df = self._fetch_offline_table(_OFFLINE_STATIC_ML_TABLE, start_dt, end_dt)
        if df is None or df.empty:
            return pd.DataFrame()
        if allowed_columns:
            df = df[df.columns.intersection(allowed_columns.keys())]
        return df

    # ------------------- STEP 2: RM DATASET -------------------

    def fetch_rm_data(
        self,
        start_date: date,
        end_date: date,
        mode: str = "charge",
        allowed_columns: dict | None = None,
    ) -> pd.DataFrame:
        """Fetch RM Charge or RM DPR dataset.

        Args:
            start_date:       Inclusive start date (local timezone).
            end_date:         Inclusive end date (local timezone).
            mode:             ``"charge"`` or ``"dpr"``.
            allowed_columns:  If given, restrict to these column names.

        Returns:
            Time-indexed :class:`pandas.DataFrame` with RM composition columns.
        """
        start_dt, end_dt = self._to_utc_window(start_date, end_date)

        table_name = _OFFLINE_RM_TABLES.get(mode)
        if table_name is None:
            raise ValueError("mode must be 'charge' or 'dpr'")

        df_rm = self._fetch_offline_table(table_name, start_dt, end_dt)
        df_rm_hm = self._collapse_strength_frame(
            self._fetch_offline_table(_OFFLINE_STRENGTH_TABLE, start_dt, end_dt)
        )
        df_chem = self._fetch_offline_weighted_chemistry(
            mode=mode,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        frames = [
            df
            for df in [df_rm, df_rm_hm, df_chem]
            if df is not None and not df.empty
        ]
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1).sort_index()
        df = df.loc[:, ~df.columns.duplicated()]
        df = self._add_offline_ml_aliases(df)
        if allowed_columns:
            df = df[df.columns.intersection(allowed_columns.keys())]
        return df

    # ------------------- STEP 3: HM & SLAG -------------------

    def fetch_hotmetal_hourly(
        self,
        start_date: date,
        end_date: date,
        keep_columns: list[str] | None = None,
        interval_minutes: int = 60,
    ) -> pd.DataFrame:
        """Fetch Hot Metal & Slag and interpolate to a regular hourly grid.

        Fetches one extra day before *start_date* so that forward-fill
        interpolation covers any gap at the start of the window.

        Args:
            start_date:        Inclusive start date.
            end_date:          Inclusive end date.
            keep_columns:      If given, subset to these column names after fetch.
            interval_minutes:  Target grid resolution in minutes (default 60).

        Returns:
            Time-indexed :class:`pandas.DataFrame` on an hourly grid (no timezone).
        """
        tz = ZoneInfo(self.local_tz)

        start_local = pd.Timestamp(start_date).tz_localize(tz)
        end_local = pd.Timestamp(end_date).tz_localize(tz) + pd.Timedelta(days=1)

        fetch_start = start_local - pd.Timedelta(days=1)
        fetch_end = end_local

        fetch_start_utc = fetch_start.tz_convert("UTC")
        fetch_end_utc = fetch_end.tz_convert("UTC")

        df = fetch_database_offline_data(
            table_name=_OFFLINE_HM_SLAG_TABLE,
            time_range=(fetch_start_utc, fetch_end_utc),
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(tz)
        df = df.sort_index().loc[~df.index.duplicated(keep="last")]

        if keep_columns:
            df = df[[c for c in keep_columns if c in df.columns]]

        if df.empty:
            return pd.DataFrame()

        # Partition columns into numerically-imputable vs metadata-text.  Neon
        # returns the lab/cast/batch identifiers as object-typed strings/UUIDs;
        # passing them through pd.to_numeric blindly (the old behaviour) wiped
        # them to all-NaN.  Detect "mostly numeric" columns across the whole
        # slice so sparse numeric lab fields are not misclassified as metadata
        # just because their first few rows are empty.
        coerced = df.apply(pd.to_numeric, errors="coerce")
        non_null_counts = df.notna().sum()
        numeric_cols = [
            c for c in df.columns
            if non_null_counts[c] > 0
            and coerced[c].notna().sum() >= 0.5 * non_null_counts[c]
        ]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        target_index = pd.date_range(
            start=start_local,
            end=end_local,
            freq=f"{interval_minutes}min",
            tz=tz,
        )

        combined_index = df.index.union(target_index)
        df2 = df.reindex(combined_index)

        if numeric_cols:
            df2[numeric_cols] = df2[numeric_cols].infer_objects(copy=False)
            df2[numeric_cols] = df2[numeric_cols].apply(pd.to_numeric, errors="coerce")
            df2[numeric_cols] = df2[numeric_cols].interpolate(method="time")

        if non_numeric_cols:
            # Forward-fill metadata so each hourly slot inherits the most recent
            # lab sample identifier; back-fill the first rows that have no prior
            # value so they aren't NaN.
            df2[non_numeric_cols] = df2[non_numeric_cols].ffill().bfill()

        df_final = df2.loc[target_index]

        today = pd.Timestamp.now(tz).date()
        if end_date == today:
            now = pd.Timestamp.now(tz)
            cutoff = now.floor(f"{interval_minutes}min")
            df_final = df_final.loc[start_local:cutoff]

        df_final.index = df_final.index.tz_localize(None)
        df_final.index.name = "time"

        return df_final

    # ------------------- STEP 4: BURDEN DISTRIBUTION -------------------

    def fetch_distribution_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch wide burden snapshots from PostgreSQL and expand to daily rows."""
        engine = self._get_engine()
        try:
            repository = BurdenHistoryRepository(
                build_relational_session_factory(engine)
            )
            snapshots = repository.fetch_distribution_frame(
                start_date=start_date,
                end_date=end_date,
            )
        finally:
            engine.dispose()

        if snapshots.empty:
            return pd.DataFrame()

        snapshots.index = pd.DatetimeIndex(pd.to_datetime(snapshots.index))
        if snapshots.index.tz is not None:
            # Convert to local IST first, then strip tz so the timestamps are
            # IST-naive daily anchors that align with the daily_index below.
            # tz_convert(None) alone would silently shift to UTC-naive (off by
            # 5h30m), causing burden rows to forward-fill from the prior day.
            snapshots.index = snapshots.index.tz_convert(ZoneInfo(self.local_tz)).tz_localize(None)
        snapshots.index = snapshots.index.normalize()

        daily_index = pd.date_range(start=start_date, end=end_date, freq="D")
        snapshots = snapshots[~snapshots.index.duplicated(keep="last")]
        df_pivot = snapshots.reindex(snapshots.index.union(daily_index)).sort_index().ffill()
        df_pivot = df_pivot.reindex(daily_index)
        df_pivot.index.name = "time"

        coke_rings = [f"coke_p{i:02d}_rings" for i in range(1, 12)]
        coke_angles = [f"coke_p{i:02d}_angles" for i in range(1, 12)]
        nc_rings = [f"noncoke_p{i:02d}_rings" for i in range(1, 12)]
        nc_angles = [f"noncoke_p{i:02d}_angles" for i in range(1, 12)]

        for columns in [coke_rings, coke_angles, nc_rings, nc_angles]:
            present = [col for col in columns if col in df_pivot.columns]
            df_pivot[present] = df_pivot[present].apply(pd.to_numeric, errors="coerce")

        present_coke_rings = [col for col in coke_rings if col in df_pivot.columns]
        present_coke_angles = [col for col in coke_angles if col in df_pivot.columns]
        present_nc_rings = [col for col in nc_rings if col in df_pivot.columns]
        present_nc_angles = [col for col in nc_angles if col in df_pivot.columns]

        df_pivot["total_coke_portions"] = df_pivot[present_coke_rings].sum(axis=1)
        df_pivot["total_non_coke_portions"] = df_pivot[present_nc_rings].sum(axis=1)

        def _weighted_angle(ring_cols: list[str], angle_cols: list[str]) -> pd.Series:
            rings = df_pivot[ring_cols].apply(pd.to_numeric, errors="coerce")
            angles = df_pivot[angle_cols].apply(pd.to_numeric, errors="coerce")
            angles.columns = ring_cols
            valid = rings.notna() & angles.notna()
            numerator = (rings.where(valid, 0) * angles.where(valid, 0)).sum(axis=1)
            denominator = rings.where(valid, 0).sum(axis=1)
            return numerator / denominator.replace(0, pd.NA)

        if present_coke_rings and present_coke_angles:
            df_pivot["weighted_coke_angle"] = _weighted_angle(
                present_coke_rings, present_coke_angles
            )
        if present_nc_rings and present_nc_angles:
            df_pivot["weighted_non_coke_angle"] = _weighted_angle(
                present_nc_rings, present_nc_angles
            )

        return df_pivot

    # ------------------- STEP 5: ONLINE PROCESS PARAMS (InfluxDB) -------------------

    def fetch_online_process_params(
        self,
        start_date: date,
        end_date: date,
        *,
        raise_on_error: bool = False,
    ) -> pd.DataFrame:
        """Step 5: hourly-averaged InfluxDB process params for post-cutoff ML dataset.

        Reads the ``ml_dataset.online_params`` mapping from the shared config
        (InfluxDB field name → ML dataset column name) and fetches the
        ``process_params`` measurement over the date window.

        Args:
            start_date: Inclusive start date (local timezone).
            end_date:   Inclusive end date (local timezone).

        Returns:
            IST tz-naive, hourly-indexed :class:`pandas.DataFrame` with ML
            column names, or an empty DataFrame if InfluxDB is unreachable or
            the mapping is absent.
        """
        from furnace_data.influx.base import BaseDataFetcher

        influx_to_ml: dict = config.get("ml_dataset", {}).get("online_params", {})
        if not influx_to_ml:
            return pd.DataFrame()

        start_dt, end_dt = self._to_utc_window(start_date, end_date)
        try:
            df = BaseDataFetcher("process_params").fetch_averaged_data(
                recent_data_of="over selected range",
                start_time=start_dt,
                end_time=end_dt,
                request_type="windowed-average",
                window_by="1 hour",
            )
        except Exception:
            if raise_on_error:
                raise
            return pd.DataFrame()

        if df is None or df.empty or "time" not in df.columns:
            return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"]).set_index("time")

        mapped = {k: v for k, v in influx_to_ml.items() if k in df.columns}
        if not mapped:
            return pd.DataFrame()

        return self._normalize_timezone(df[list(mapped.keys())].rename(columns=mapped))

    def fetch_online_heatload_params(
        self,
        start_date: date,
        end_date: date,
        *,
        raise_on_error: bool = False,
    ) -> pd.DataFrame:
        """Fetch hourly total heat load from the live heatload measurement."""
        from furnace_data.influx.base import BaseDataFetcher

        components = config.get("ml_dataset", {}).get("heatload_total_components", [])
        components = [str(component) for component in components]
        if not components:
            return pd.DataFrame()

        start_dt, end_dt = self._to_utc_window(start_date, end_date)
        try:
            df = BaseDataFetcher("heatload_delta_t").fetch_averaged_data(
                recent_data_of="over selected range",
                start_time=start_dt,
                end_time=end_dt,
                request_type="windowed-average",
                window_by="1 hour",
            )
        except Exception:
            if raise_on_error:
                raise
            return pd.DataFrame()

        if df is None or df.empty or "time" not in df.columns:
            return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"]).set_index("time")
        present = [column for column in components if column in df.columns]
        if not present:
            return pd.DataFrame()

        numeric = df[present].apply(pd.to_numeric, errors="coerce")
        out = pd.DataFrame(index=df.index)
        out["total_heat_load"] = numeric.sum(axis=1, min_count=1)
        return self._normalize_timezone(out)

    def fetch_online_misc_params(
        self,
        start_date: date,
        end_date: date,
        *,
        raise_on_error: bool = False,
    ) -> pd.DataFrame:
        """Fetch hourly miscellaneous online params used by the ML dataset."""
        from furnace_data.influx.base import BaseDataFetcher

        influx_to_ml: dict = config.get("ml_dataset", {}).get("misc_params", {})
        if not influx_to_ml:
            return pd.DataFrame()

        start_dt, end_dt = self._to_utc_window(start_date, end_date)
        try:
            df = BaseDataFetcher("miscellaneous").fetch_averaged_data(
                recent_data_of="over selected range",
                start_time=start_dt,
                end_time=end_dt,
                request_type="windowed-average",
                window_by="1 hour",
            )
        except Exception:
            if raise_on_error:
                raise
            return pd.DataFrame()

        if df is None or df.empty or "time" not in df.columns:
            return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"]).set_index("time")
        mapped = {k: v for k, v in influx_to_ml.items() if k in df.columns}
        if not mapped:
            return pd.DataFrame()

        return self._normalize_timezone(df[list(mapped.keys())].rename(columns=mapped))

    def fetch_online_temperature_params(
        self,
        start_date: date,
        end_date: date,
        *,
        raise_on_error: bool = False,
    ) -> pd.DataFrame:
        """Step 5b: hourly-averaged InfluxDB temperature profile for the delta.

        Reads ``ml_dataset.temperature_params`` (InfluxDB field → rename_dict
        intermediate key) and fetches the ``temperature_profile`` measurement.
        Returns an empty DataFrame if the config key is absent or InfluxDB is
        unreachable.
        """
        from furnace_data.influx.base import BaseDataFetcher

        influx_to_intermediate: dict = config.get("ml_dataset", {}).get("temperature_params", {})
        if not influx_to_intermediate:
            return pd.DataFrame()

        start_dt, end_dt = self._to_utc_window(start_date, end_date)
        try:
            df = BaseDataFetcher("temperature_profile").fetch_averaged_data(
                recent_data_of="over selected range",
                start_time=start_dt,
                end_time=end_dt,
                request_type="windowed-average",
                window_by="1 hour",
            )
        except Exception:
            if raise_on_error:
                raise
            return pd.DataFrame()

        if df is None or df.empty or "time" not in df.columns:
            return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"]).set_index("time")

        mapped = {k: v for k, v in influx_to_intermediate.items() if k in df.columns}
        if not mapped:
            return pd.DataFrame()

        out = df[list(mapped.keys())].rename(columns=mapped)
        hearth_cols = [
            "hearth_pad_a_c",
            "hearth_pad_b_c",
            "hearth_pad_c_c",
            "hearth_pad_d_c",
        ]
        if all(col in out.columns for col in hearth_cols):
            out["hearth_pad_avg_c"] = (
                out[hearth_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            )

        return self._normalize_timezone(out)

    # ------------------- ALIASES -------------------

    def fetch_step1(self, start_date: date, end_date: date, allowed_columns=None):
        """Alias for :meth:`fetch` (historical static dataset)."""
        return self.fetch(start_date, end_date, allowed_columns)

    def fetch_step2(
        self,
        start_date: date,
        end_date: date,
        mode: str,
        allowed_columns=None,
    ):
        """Alias for :meth:`fetch_rm_data` (RM Charge / DPR dataset)."""
        return self.fetch_rm_data(start_date, end_date, mode, allowed_columns)
