"""
MlDatasetService — ported from src/ml_pipeline/ml_dataset_service.py.

4-step fetch pipeline:
  Step 1: Legacy ML dataset (pre-cutoff)
  Step 2: RM Charge / RM DPR dataset (post-cutoff)
  Step 3: Hot Metal & Slag (interpolated to hourly)
  Step 4: Burden distribution from PostgreSQL
"""

import logging
import os
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine

from app.core.config_loader import load_config
from app.core.influx_client import fetch_offline_data

log = logging.getLogger(__name__)

config = load_config("setting_ds_dv.yml")


@dataclass
class MlDatasetService:
    """
    Step-1: Fetch ML Dataset (main operational dataset)
    Step-2: Fetch RM Charge or RM DPR datasets
    Step-3: Fetch Hot Metal & Slag and interpolate to given interval
    Step-4: Fetch Burden Distribution from PostgreSQL
    """

    bucket: str = config["ml_dataset"]["bucket"]
    measurement_step1: str = config["ml_dataset"]["measurement_step1"]
    measurement_rm_charge: str = config["ml_dataset"]["rm_charge_measurement"]
    measurement_rm_dpr: str = config["ml_dataset"]["rm_dpr_measurement"]

    hotmetal_bucket: str = config["influx_offline"]["database"]
    hotmetal_measurement: str = config["offline_measurements"]["HM & Slag"]

    local_tz: str = config["ml_dataset"].get("local_tz", "Asia/Kolkata")

    cutoff_date: date = date.fromisoformat(config["ml_dataset"]["cutoff_date"])

    db_url: str = os.getenv("DATABASE_URL", "")

    def _get_engine(self):
        return create_engine(self.db_url)

    # --------------- INFLUX FETCH WITH RETRIES ---------------
    def _safe_influx_call(self, measurement, start_dt, end_dt, bucket=None, retries=3, wait=2):
        db = bucket or self.bucket

        for attempt in range(retries):
            try:
                return fetch_offline_data(
                    measurement=measurement,
                    time_range=(start_dt, end_dt),
                    database=db,
                )
            except Exception as e:
                msg = str(e).lower()
                transient_errors = [
                    "resourceexhausted",
                    "simultaneous query limit exceeded",
                    "flightunavailable",
                    "timed out",
                    "deadline exceeded",
                    "504",
                    "unavailable",
                    "server never sent a data message",
                ]
                if any(err in msg for err in transient_errors) and attempt < retries - 1:
                    log.warning("Transient InfluxDB error (attempt %d/%d): %s", attempt + 1, retries, e)
                    time_module.sleep(wait)
                    continue
                raise

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

    # ------------------- STEP 1: OLD DATASET -------------------
    def fetch(self, start_date: date, end_date: date, allowed_columns=None) -> pd.DataFrame:
        start_dt, end_dt = self._to_utc_window(start_date, end_date)
        df = self._safe_influx_call(self.measurement_step1, start_dt, end_dt)
        if df is None or df.empty:
            return pd.DataFrame()
        df = self._normalize_timezone(df)
        if allowed_columns:
            df = df[df.columns.intersection(allowed_columns.keys())]
        return df

    # ------------------- STEP 2: NEW RM DATASET -------------------
    def fetch_rm_data(
        self,
        start_date: date,
        end_date: date,
        mode: str = "charge",
        allowed_columns=None,
    ) -> pd.DataFrame:
        measurement = (
            self.measurement_rm_charge if mode == "charge" else self.measurement_rm_dpr
        )
        start_dt, end_dt = self._to_utc_window(start_date, end_date)
        df = self._safe_influx_call(measurement, start_dt, end_dt)
        if df is None or df.empty:
            return pd.DataFrame()
        df = self._normalize_timezone(df)
        if allowed_columns:
            df = df[df.columns.intersection(allowed_columns.keys())]
        return df

    # ------------------- STEP 3: HM & SLAG -------------------
    def fetch_hotmetal_hourly(
        self,
        start_date: date,
        end_date: date,
        keep_columns=None,
        interval_minutes: int = 60,
    ) -> pd.DataFrame:
        tz = ZoneInfo(self.local_tz)

        start_local = pd.Timestamp(start_date).tz_localize(tz)
        end_local = pd.Timestamp(end_date).tz_localize(tz) + pd.Timedelta(days=1)

        fetch_start = start_local - pd.Timedelta(days=1)
        fetch_end = end_local

        fetch_start_utc = fetch_start.tz_convert("UTC")
        fetch_end_utc = fetch_end.tz_convert("UTC")

        df = self._safe_influx_call(
            measurement=self.hotmetal_measurement,
            start_dt=fetch_start_utc,
            end_dt=fetch_end_utc,
            bucket=self.hotmetal_bucket,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df.index = df.index.tz_convert(tz)
        df = df.sort_index().loc[~df.index.duplicated(keep="last")]

        if keep_columns:
            df = df[[c for c in keep_columns if c in df.columns]]

        if df.empty:
            return pd.DataFrame()

        numeric_cols = df.columns

        target_index = pd.date_range(
            start=start_local,
            end=end_local,
            freq=f"{interval_minutes}min",
            tz=tz,
        )

        combined_index = df.index.union(target_index)
        df2 = df.reindex(combined_index)

        df2[numeric_cols] = df2[numeric_cols].infer_objects(copy=False)
        df2[numeric_cols] = df2[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df2[numeric_cols] = df2[numeric_cols].interpolate(method="time")

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
        engine = self._get_engine()

        def _to_date(v):
            return v.date() if hasattr(v, "date") else v

        query = """
            SELECT
                field_name,
                COALESCE(
                    CAST(field_value_float AS TEXT),
                    field_value_text
                ) AS field_value,
                valid_from,
                valid_upto
            FROM burden_distribution_history
            WHERE valid_from <= %(end_date)s::date
            AND (valid_upto IS NULL OR valid_upto >= %(start_date)s::date)
            ORDER BY valid_from;
        """

        df = pd.read_sql_query(
            query,
            engine,
            params={"start_date": start_date, "end_date": end_date},
            parse_dates=["valid_from", "valid_upto"],
        )

        if df.empty:
            return pd.DataFrame()

        df["valid_upto"] = df["valid_upto"].fillna(end_date)

        expanded_rows = []
        for _, row in df.iterrows():
            valid_from = max(_to_date(row["valid_from"]), start_date)
            valid_upto = min(_to_date(row["valid_upto"]), end_date)
            rng = pd.date_range(start=valid_from, end=valid_upto, freq="D")
            for t in rng:
                expanded_rows.append([t, row["field_name"], row["field_value"]])

        df2 = pd.DataFrame(expanded_rows, columns=["time", "field_name", "field_value"])

        df_pivot = df2.pivot_table(
            index="time",
            columns="field_name",
            values="field_value",
            aggfunc="last",
        ).sort_index()

        coke_rings = [c for c in df_pivot.columns if "COKE" in c and "RINGS" in c]
        coke_angles = [c for c in df_pivot.columns if "COKE" in c and "ANGLES" in c]
        nc_rings = [c for c in df_pivot.columns if "NONCOKE" in c and "RINGS" in c]
        nc_angles = [c for c in df_pivot.columns if "NONCOKE" in c and "ANGLES" in c]

        df_pivot[coke_rings] = df_pivot[coke_rings].apply(pd.to_numeric, errors="coerce")
        df_pivot[coke_angles] = df_pivot[coke_angles].apply(pd.to_numeric, errors="coerce")
        df_pivot[nc_rings] = df_pivot[nc_rings].apply(pd.to_numeric, errors="coerce")
        df_pivot[nc_angles] = df_pivot[nc_angles].apply(pd.to_numeric, errors="coerce")

        df_pivot["TOTAL_COKE_PORTIONS"] = df_pivot[coke_rings].sum(axis=1)
        df_pivot["TOTAL_NON_COKE_PORTIONS"] = df_pivot[nc_rings].sum(axis=1)

        df_pivot["WEIGHTED_COKE_ANGLE"] = (
            (df_pivot[coke_rings].values * df_pivot[coke_angles].values).sum(axis=1)
            / df_pivot["TOTAL_COKE_PORTIONS"].replace(0, pd.NA)
        )
        df_pivot["WEIGHTED_NON_COKE_ANGLE"] = (
            (df_pivot[nc_rings].values * df_pivot[nc_angles].values).sum(axis=1)
            / df_pivot["TOTAL_NON_COKE_PORTIONS"].replace(0, pd.NA)
        )

        for col in df_pivot.columns:
            if df_pivot[col].dtype == "object":
                try:
                    df_pivot[col] = pd.to_numeric(df_pivot[col])
                except (ValueError, TypeError):
                    pass

        return df_pivot

    # ------------------- ALIASES -------------------
    def fetch_step1(self, start_date, end_date, allowed_columns=None):
        return self.fetch(start_date, end_date, allowed_columns)

    def fetch_step2(self, start_date, end_date, mode, allowed_columns=None):
        return self.fetch_rm_data(start_date, end_date, mode, allowed_columns)
