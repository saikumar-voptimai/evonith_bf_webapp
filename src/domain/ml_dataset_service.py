# from dataclasses import dataclass
# from datetime import datetime, date, time, timezone, timedelta
# from zoneinfo import ZoneInfo
# import pandas as pd
# import time as time_module   # for sleep(), avoids conflict with datetime.time

# from utils.helper_functions_explorer.data_retrieval import fetch_offline_data


# @dataclass
# class MlDatasetService:
#     """
#     Step-1: Fetch ML Dataset (main operational dataset)
#     Step-2: Fetch RM Charge or RM DPR datasets
#     Step-3: Fetch Hot Metal & Slag and interpolate to hourly data
#     """

#     bucket: str = "ML DATASET"
#     measurement_step1: str = "rm_charge_dis_hm_slag"
#     measurement_rm_charge: str = "rm_charge_data"
#     measurement_rm_dpr: str = "rm_dpr_data"

#     # STEP 3 config
#     hotmetal_bucket: str = "bf2_evonith_offline_utc"
#     hotmetal_measurement: str = "hotmetal_slag_updated_data"

#     local_tz: str = "Asia/Kolkata"
#     cutoff_date: date = date(2025, 12, 6)

#     # ----------------------------------------------------------
#     # Safe Influx Query Wrapper
#     # ----------------------------------------------------------
#     def _safe_influx_call(self, measurement, start_dt, end_dt, bucket=None, retries=3, wait=2):
#         """
#         Wraps fetch_offline_data with simple retry on transient errors
#         (simultaneous query limit, temporary unavailability, etc.).
#         """
#         db = bucket or self.bucket

#         for attempt in range(retries):
#             try:
#                 return fetch_offline_data(
#                     measurement=measurement,
#                     time_range=(start_dt, end_dt),
#                     database=db,
#                 )
#             except Exception as e:
#                 msg = str(e).lower()
#                 transient_errors = [
#                     "resourceexhausted",
#                     "simultaneous query limit exceeded",
#                     "flightunavailable",
#                     "timed out",
#                     "deadline exceeded",
#                     "504",
#                     "unavailable",
#                     "server never sent a data message",
#                 ]
#                 if any(err in msg for err in transient_errors) and attempt < retries - 1:
#                     time_module.sleep(wait)
#                     continue
#                 raise e

#     # ----------------------------------------------------------
#     # Convert date range into UTC timestamps
#     # ----------------------------------------------------------
#     def _to_utc_window(self, start_date: date, end_date: date):
#         tz = ZoneInfo(self.local_tz)
#         start_dt = datetime.combine(start_date, time.min).replace(tzinfo=tz).astimezone(timezone.utc)
#         end_dt   = datetime.combine(end_date, time.max).replace(tzinfo=tz).astimezone(timezone.utc)
#         return start_dt, end_dt

#     # ----------------------------------------------------------
#     # Normalize timezone: UTC → IST → strip tz
#     # ----------------------------------------------------------
#     def _normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
#         if df.empty:
#             return df

#         tz = ZoneInfo(self.local_tz)

#         if df.index.tz is None:
#             df.index = df.index.tz_localize("UTC")

#         df.index = df.index.tz_convert(tz).tz_localize(None)
#         df.index.name = "time"

#         return df.sort_index()

#     # ----------------------------------------------------------
#     # STEP 1: Fetch main ML dataset
#     # ----------------------------------------------------------
#     def fetch(self, start_date: date, end_date: date, allowed_columns=None) -> pd.DataFrame:

#         start_dt, end_dt = self._to_utc_window(start_date, end_date)

#         df = self._safe_influx_call(self.measurement_step1, start_dt, end_dt)

#         if df is None or df.empty:
#             return pd.DataFrame()

#         df = self._normalize_timezone(df)

#         if allowed_columns:
#             df = df[df.columns.intersection(allowed_columns.keys())]

#         return df

#     # ----------------------------------------------------------
#     # STEP 2: Fetch RM Charge or RM DPR data
#     # ----------------------------------------------------------
#     def fetch_rm_data(
#         self,
#         start_date: date,
#         end_date: date,
#         mode: str = "charge",       # "charge" or "dpr"
#         allowed_columns=None
#     ) -> pd.DataFrame:

#         measurement = (
#             self.measurement_rm_charge if mode == "charge" else self.measurement_rm_dpr
#         )

#         start_dt, end_dt = self._to_utc_window(start_date, end_date)

#         df = self._safe_influx_call(measurement, start_dt, end_dt)

#         if df is None or df.empty:
#             return pd.DataFrame()

#         df = self._normalize_timezone(df)

#         if allowed_columns:
#             df = df[df.columns.intersection(allowed_columns.keys())]

#         return df

#     # ----------------------------------------------------------
#     # STEP 3: Fetch Hot Metal / Slag and interpolate to hourly
#     # ----------------------------------------------------------
#     def fetch_hotmetal_hourly(
#         self,
#         start_date,
#         end_date,
#         keep_columns=None,
#         interval_minutes=60,
#         rename_dict=None
#     ):
#         """
#         Fetch hot metal + slag and interpolate at custom intervals.
#         This is the corrected version with safe defaults and full functionality.
#         """

#         tz = ZoneInfo(self.local_tz)

#         # Convert to timezone-aware timestamps
#         start_local = pd.Timestamp(start_date).tz_localize(tz)
#         end_local = pd.Timestamp(end_date).tz_localize(tz) + pd.Timedelta(days=1)

#         # Fetch 1 extra day before for smoother interpolation
#         fetch_start = start_local - pd.Timedelta(days=1)
#         fetch_end = end_local

#         # Convert to UTC for Influx query
#         fetch_start_utc = fetch_start.tz_convert("UTC")
#         fetch_end_utc = fetch_end.tz_convert("UTC")

#         # ---------------- FETCH RAW DATA ----------------
#         df = self._safe_influx_call(
#             measurement=self.hotmetal_measurement,
#             start_dt=fetch_start_utc,
#             end_dt=fetch_end_utc,
#             bucket=self.hotmetal_bucket
#         )

#         if df is None or df.empty:
#             return pd.DataFrame()

#         # Convert index to local timezone (IST)
#         df.index = df.index.tz_convert(tz)
#         df = df.sort_index().loc[~df.index.duplicated(keep="last")]

#         # Keep only selected columns
#         if keep_columns:
#             df = df[[c for c in keep_columns if c in df.columns]]

#         if df.empty:
#             return pd.DataFrame()

#         numeric_cols = df.columns

#         # ---------------- TARGET TIME RANGE ----------------
#         target_index = pd.date_range(
#             start=start_local,
#             end=end_local,
#             freq=f"{interval_minutes}min",
#             tz=tz
#         )

#         # Merge raw + target for interpolation
#         combined_index = df.index.union(target_index)
#         df2 = df.reindex(combined_index)

#         # Safe numeric conversion
#         df2[numeric_cols] = df2[numeric_cols].infer_objects(copy=False)
#         df2[numeric_cols] = df2[numeric_cols].apply(pd.to_numeric, errors="coerce")

#         # Interpolation
#         df2[numeric_cols] = df2[numeric_cols].interpolate(method="time")

#         # Extract only target timestamps
#         df_final = df2.loc[target_index]

#         # ---------- HANDLE CASE: end_date = today ----------
#         today = pd.Timestamp.now(tz).date()
#         if end_date == today:
#             now = pd.Timestamp.now(tz)
#             cutoff = now.floor(f"{interval_minutes}min")
#             df_final = df_final.loc[start_local:cutoff]

#         # Remove timezone for UI compatibility
#         df_final.index = df_final.index.tz_localize(None)
#         df_final.index.name = "time"

#         # Optional renaming
#         if rename_dict:
#             rename_map = {c: rename_dict[c] for c in df_final.columns if c in rename_dict}
#             df_final = df_final.rename(columns=rename_map)

#             # Keep only renamed fields
#             df_final = df_final[[v for v in rename_dict.values() if v in df_final.columns]]

#         return df_final





#     # ----------------------------------------------------------
#     # Helper for UI: Should we use Step-2 only?
#     # ----------------------------------------------------------
#     # def is_step2_only(self, start_date: date) -> bool:
#     #     return start_date > self.cutoff_date

#     # # ----------------------------------------------------------
#     # # Helper for UI: Should we combine Step-1 + Step-2?
#     # # ----------------------------------------------------------
#     # def is_mixed_range(self, start_date: date, end_date: date) -> bool:
#     #     return start_date <= self.cutoff_date < end_date

#     # ----------------------------------------------------------
#     # Wrappers to match UI function names (Step-1 / Step-2 / Step-3)
#     # ----------------------------------------------------------
#     def fetch_step1(self, start_date, end_date, allowed_columns=None):
#         return self.fetch(start_date, end_date, allowed_columns)

#     def fetch_step2(self, start_date, end_date, mode, allowed_columns=None):
#         return self.fetch_rm_data(start_date, end_date, mode, allowed_columns)

#     def fetch_hotmetal_hourly(self, start_date, end_date, keep_columns=None):
#         return self.fetch_hot_metal_hourly(start_date, end_date, keep_columns)






from dataclasses import dataclass
from datetime import datetime, date, time, timezone, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import time as time_module   # for sleep(), avoids conflict with datetime.time

from utils.helper_functions_explorer.data_retrieval import fetch_offline_data


@dataclass
class MlDatasetService:
    """
    Step-1: Fetch ML Dataset (main operational dataset)
    Step-2: Fetch RM Charge or RM DPR datasets
    Step-3: Fetch Hot Metal & Slag and interpolate to given interval
    """

    bucket: str = "ML DATASET"
    measurement_step1: str = "rm_charge_dis_hm_slag"
    measurement_rm_charge: str = "rm_charge_data"
    measurement_rm_dpr: str = "rm_dpr_data"

    # STEP 3 config
    hotmetal_bucket: str = "bf2_evonith_offline_utc"
    hotmetal_measurement: str = "hotmetal_slag_updated_data"

    local_tz: str = "Asia/Kolkata"
    cutoff_date: date = date(2025, 12, 5)

    # ----------------------------------------------------------
    # Safe Influx Query Wrapper
    # ----------------------------------------------------------
    def _safe_influx_call(self, measurement, start_dt, end_dt, bucket=None, retries=3, wait=2):
        """
        Wraps fetch_offline_data with simple retry on transient errors
        (simultaneous query limit, temporary unavailability, etc.).
        """
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
                    time_module.sleep(wait)
                    continue
                raise e

    # ----------------------------------------------------------
    # Convert date range into UTC timestamps
    # ----------------------------------------------------------
    def _to_utc_window(self, start_date: date, end_date: date):
        tz = ZoneInfo(self.local_tz)
        start_dt = datetime.combine(start_date, time.min).replace(tzinfo=tz).astimezone(timezone.utc)
        end_dt   = datetime.combine(end_date, time.max).replace(tzinfo=tz).astimezone(timezone.utc)
        return start_dt, end_dt

    # ----------------------------------------------------------
    # Normalize timezone: UTC → IST → strip tz
    # ----------------------------------------------------------
    def _normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        tz = ZoneInfo(self.local_tz)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df.index = df.index.tz_convert(tz).tz_localize(None)
        df.index.name = "time"

        return df.sort_index()

    # ----------------------------------------------------------
    # STEP 1: Fetch main ML dataset
    # ----------------------------------------------------------
    def fetch(self, start_date: date, end_date: date, allowed_columns=None) -> pd.DataFrame:

        start_dt, end_dt = self._to_utc_window(start_date, end_date)

        df = self._safe_influx_call(self.measurement_step1, start_dt, end_dt)

        if df is None or df.empty:
            return pd.DataFrame()

        df = self._normalize_timezone(df)

        if allowed_columns:
            df = df[df.columns.intersection(allowed_columns.keys())]

        return df

    # ----------------------------------------------------------
    # STEP 2: Fetch RM Charge or RM DPR data
    # ----------------------------------------------------------
    def fetch_rm_data(
        self,
        start_date: date,
        end_date: date,
        mode: str = "charge",       # "charge" or "dpr"
        allowed_columns=None
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

    # ----------------------------------------------------------
    # STEP 3: Fetch Hot Metal / Slag and interpolate to interval
    # ----------------------------------------------------------
    def fetch_hotmetal_hourly(
        self,
        start_date: date,
        end_date: date,
        keep_columns=None,
        interval_minutes: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch hot metal + slag and interpolate at custom intervals.
        Logic is aligned with the standalone HOT METAL UI feature.
        """

        tz = ZoneInfo(self.local_tz)

        # Convert to timezone-aware timestamps
        start_local = pd.Timestamp(start_date).tz_localize(tz)
        end_local   = pd.Timestamp(end_date).tz_localize(tz) + pd.Timedelta(days=1)

        # Fetch 1 extra day before for better interpolation
        fetch_start = start_local - pd.Timedelta(days=1)
        fetch_end   = end_local

        # Convert to UTC for Influx query
        fetch_start_utc = fetch_start.tz_convert("UTC")
        fetch_end_utc   = fetch_end.tz_convert("UTC")

        # ---------------- FETCH RAW DATA ----------------
        df = self._safe_influx_call(
            measurement=self.hotmetal_measurement,
            start_dt=fetch_start_utc,
            end_dt=fetch_end_utc,
            bucket=self.hotmetal_bucket
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Convert index to local timezone (IST)
        df.index = df.index.tz_convert(tz)
        df = df.sort_index().loc[~df.index.duplicated(keep="last")]

        # Keep only selected columns
        if keep_columns:
            df = df[[c for c in keep_columns if c in df.columns]]

        if df.empty:
            return pd.DataFrame()

        numeric_cols = df.columns

        # ---------------- TARGET TIME RANGE ----------------
        target_index = pd.date_range(
            start=start_local,
            end=end_local,
            freq=f"{interval_minutes}min",
            tz=tz
        )

        # Merge raw + target for interpolation
        combined_index = df.index.union(target_index)
        df2 = df.reindex(combined_index)

        # Safe numeric conversion
        df2[numeric_cols] = df2[numeric_cols].infer_objects(copy=False)
        df2[numeric_cols] = df2[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Interpolation
        df2[numeric_cols] = df2[numeric_cols].interpolate(method="time")

        # Extract only target timestamps
        df_final = df2.loc[target_index]

        # ---------- HANDLE CASE: end_date = today's date ----------
        today = pd.Timestamp.now(tz).date()
        if end_date == today:
            now = pd.Timestamp.now(tz)
            cutoff = now.floor(f"{interval_minutes}min")
            df_final = df_final.loc[start_local:cutoff]

        # Remove timezone for UI compatibility
        df_final.index = df_final.index.tz_localize(None)
        df_final.index.name = "time"

        return df_final

    # ----------------------------------------------------------
    # Wrappers to match UI function names (Step-1 / Step-2)
    # ----------------------------------------------------------
    def fetch_step1(self, start_date, end_date, allowed_columns=None):
        return self.fetch(start_date, end_date, allowed_columns)

    def fetch_step2(self, start_date, end_date, mode, allowed_columns=None):
        return self.fetch_rm_data(start_date, end_date, mode, allowed_columns)
