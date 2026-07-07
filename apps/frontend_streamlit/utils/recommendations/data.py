import warnings
from typing import Any, Dict, List

import joblib
import pandas as pd

from furnace_data.influx.base import BaseDataFetcher
from furnace_data.dataset.static_csv import load_static_dataset


class DataframesProcessor:
    """
    DataframesProcessor remembers the full dataframe which has the historic dataset
    and the live (latest) data, and then serve the data for ML models for inference.
    It works for any target output required and uses the scaler (and feature names) provided
    to generate the needed frame.
    """

    def __init__(self, config, config_vsense, debug_on: bool = False):
        """
        Initialize with the full dataframe.
        Args:
            df_full (pd.DataFrame): The complete dataframe to process.
        """
        self.config = config
        self.config_vsense = config_vsense
        self.debug_on = debug_on

        self.df_hist = self.fetch_historical_data()
        cp_op_ml_dict, meas_set = self.fetch_live_params()
        df_live = self.fetch_live_data(cp_op_ml_dict, meas_set)
        LiveMissing = False
        if len(df_live) != 0:
            self.df_live_row = self.compute_dependant_features(df_live)
        else:
            self.df_live_row = df_live
            LiveMissing = True
        self.df_latest_row = self.create_latest_row()
        self.df_full = (
            pd.concat([self.df_hist, self.df_latest_row])
            if not LiveMissing
            else self.df_hist
        )

    def fetch_historical_data(self) -> pd.DataFrame:
        """
        Fetch historical data from the static ML dataset.
        Returns:
            pd.DataFrame: DataFrame containing historical data.
        """
        data_rel_path = self.config["DATA"]
        df_hist = load_static_dataset(data_rel_path)
        df_hist.index = pd.to_datetime(df_hist.index, errors="coerce")
        return df_hist

    def fetch_live_rm_data(self) -> pd.Series:
        """
        Return the most recent raw material row from the static ML dataset.

        The static ML dataset is already a wide flat frame with ML column names,
        so no offline database query or rename is needed here.
        Returns:
            pd.Series: Latest row from the static ML dataset.
        """
        return self.df_hist.iloc[-1]

    def create_latest_row(self) -> pd.DataFrame:
        """
        Create last row of full df using live data. To maintain consistency,
        we take the last row of historical data and update columns for which
        live data is available.
        Args:
            debug_mode (bool): If True, do not update with live data.
        Returns:
            pd.DataFrame: Created last historical data row with live data.
        """
        # Live Data:
        if self.df_live_row.empty:
            print("Live data is empty. Cannot merge with historical data.")
            return self.df_hist.iloc[[-1]]  # Return last historical row as is.

        # Historical Data:
        if self.df_hist.empty:
            raise ValueError("Historical data is empty. Cannot merge with live data.")

        # Attach live:
        row_hist_last = self.df_hist.iloc[-1].copy()  # Start with last historical row
        if not self.debug_on:
            update_cols = self.df_live_row.columns.intersection(
                self.df_hist.columns
            ).tolist()
            if update_cols:
                row_live_latest = self.df_live_row.iloc[-1][update_cols]
                row_hist_last.loc[update_cols] = row_live_latest

        df_latest_row = pd.DataFrame(
            [row_hist_last.values],
            columns=self.df_hist.columns,
            index=[pd.to_datetime(self.df_live_row.index[-1])],
        )
        return df_latest_row

    def compute_dependant_features(self, live_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute dependant features needed for the ML model.
        """
        if "PCI_KG/THM" not in live_data:
            return live_data
        # Example: Unit cost feature
        live_data["UNITCOST LAKHS/THM"] = 0.25 * (
            live_data["COKE RATE KG/THM"]
            + self.config["Coke to PCI"] * live_data["PCI_KG/THM"]
        )
        return live_data

    def process_dataframe(self, scaler_path: str = None) -> pd.DataFrame:
        """
        Process the DataFrame and return the features needed for ML model inference
        as per the scaler's feature names.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: Processed DataFrame with 'time' index.
        """
        scaler = joblib.load(scaler_path)
        df = self.df_full.copy()
        df.index = pd.to_datetime(df.index, errors="coerce", format="%d/%m/%Y %H:%M")

        if scaler is None or not hasattr(scaler, "feature_names_in_"):
            raise NotImplementedError("Scaler must be provided for feature processing.")

        feature_names = scaler.feature_names_in_.tolist()

        if "hour" in feature_names and "hour" not in df.columns:
            df["hour"] = df.index.to_series().dt.hour
        if "day_of_week" in feature_names and "day_of_week" not in df.columns:
            df["day_of_week"] = df.index.to_series().dt.dayofweek
        if "month" in feature_names and "month" not in df.columns:
            df["month"] = df.index.to_series().dt.month

        idx_to_col: Dict[int, str] = {}
        missing_features: list[str] = []

        for idx, feat in enumerate(feature_names):
            if "_lag" in feat and feat.split("_lag")[0] in df.columns:
                base_name, _, lag_str = feat.partition("_lag")
                try:
                    lag = int(lag_str)
                except ValueError:
                    missing_features.append(feat)
                    continue
                lag_col = f"{base_name}_lag{lag}"
                df[lag_col] = df[base_name].shift(lag)
                idx_to_col[idx] = lag_col
            elif feat in df.columns:
                idx_to_col[idx] = feat
            elif feat in ["hour", "day_of_week", "month"]:
                idx_to_col[idx] = feat
            else:
                missing_features.append(feat)

        ordered_cols = [col for _, col in sorted(idx_to_col.items())]
        df = df[ordered_cols]
        if missing_features:
            warnings.warn(
                f"process_dataframe: Features '{missing_features}' not found in DataFrame columns."
            )

        df.dropna(inplace=True)  # NaNs peek in due to lag features
        return df

    def fetch_live_data(
        self, cp_op_ml_dict: Dict[str, Any], influx_paths: List[str]
    ) -> pd.DataFrame:
        """
        Fetch latest (present) value for control and input parameters from InfluxDB.
        Returns one-row DataFrame with most recent values.
        """

        now = pd.Timestamp.utcnow()
        lookback = now - pd.Timedelta(minutes=15)  # safety window

        # Map Influx field → ML column name
        values_needed = {
            cp_op_ml_dict[param]["InfluxName"]: cp_op_ml_dict[param]["NameInMLData"]
            for param in cp_op_ml_dict
        }

        meas_dict: Dict[str, list] = {}
        for param in cp_op_ml_dict:
            path = (
                cp_op_ml_dict[param]["InfluxBucket"]
                + "/"
                + cp_op_ml_dict[param]["InfluxMeasurement"]
            )
            meas_dict.setdefault(path, []).append(cp_op_ml_dict[param]["InfluxName"])

        df_combined = pd.DataFrame()

        for influx_path, required_vars in meas_dict.items():
            bucket, meas = influx_path.split("/")

            datafetcher = BaseDataFetcher(meas, database=bucket)

            #  Fetch recent raw data (not hourly)
            df_meas = datafetcher.fetch_averaged_data(
                recent_data_of="over selected range",
                start_time=lookback,
                end_time=now,
                request_type="ts",
                window_by=None,
            )

            if df_meas.empty:
                continue

            # Special logic
            if meas == "process_params":
                if {"coke_rate", "nut_coke_rate"}.issubset(df_meas.columns):
                    df_meas["coke_rate"] = (
                        df_meas["coke_rate"] + df_meas["nut_coke_rate"]
                    )

            df_meas = df_meas[required_vars + ["time"]]
            df_meas["time"] = pd.to_datetime(df_meas["time"], utc=True, errors="coerce")
            df_meas.set_index("time", inplace=True)
            df_meas = df_meas.sort_index()
            df_meas.index = df_meas.index.tz_convert("Asia/Kolkata")

            #  TAKE LATEST VALUE
            latest_row = df_meas.iloc[-1:]

            df_combined = (
                latest_row
                if df_combined.empty
                else df_combined.join(latest_row, how="outer")
            )

        if df_combined.empty:
            return df_combined

        # Rename to ML column names
        df_combined.rename(columns=values_needed, inplace=True)

        # Uppercase consistency
        df_combined.columns = [col.upper() for col in df_combined.columns]

        return df_combined

    def fetch_live_params(self) -> tuple[Dict[str, Any], List[str]]:
        """
        Fetch metadata for control and input parameters from configuration.
        Returns:
            tuple: (cp_op_ml_dict, meas_list)
        """
        cp_op_list = list(self.config["Parameter Synonyms"].keys())
        cp_op_ml_dict, meas_list = {}, []
        for i, param in enumerate(cp_op_list):
            cp_op_ml_dict[param] = {
                "InfluxBucket": self.config["Parameter Synonyms"][param][
                    "InfluxBucket"
                ],
                "InfluxMeasurement": self.config["Parameter Synonyms"][param][
                    "InfluxMeasurement"
                ],
                "InfluxName": self.config["Parameter Synonyms"][param]["InfluxName"],
                "NameInMLData": self.config["Parameter Synonyms"][param][
                    "NameInMLData"
                ],
            }
            meas_list.append(
                self.config["Parameter Synonyms"][param]["InfluxBucket"]
                + "/"
                + self.config["Parameter Synonyms"][param]["InfluxMeasurement"]
            )
        return cp_op_ml_dict, meas_list
