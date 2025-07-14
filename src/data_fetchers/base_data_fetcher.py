import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
import os
import pytz
import numpy as np
import pandas as pd
import requests
import dotenv
import json
import xml.etree.ElementTree as ET
from influxdb_client_3 import InfluxDBClient3, flight_client_options
import certifi
from src.config.config_loader import load_config

fh = open(certifi.where(), "r")
cert = fh.read()
fh.close()

TIMEDELTAS = {
        'last 1 minute': timedelta(minutes=1),
        'last 5 minutes': timedelta(minutes=5),
        'last 15 minutes': timedelta(minutes=15),
        'last 30 minutes': timedelta(minutes=30),
        'last 1 hour': timedelta(hours=1),
        'last 6 hours': timedelta(hours=6),
        'last 12 hours': timedelta(hours=12),
        'last 1 day': timedelta(days=1),
        'last 3 days': timedelta(days=3),
        'last 1 week': timedelta(weeks=1),
        'last 2 weeks': timedelta(weeks=2),
        'last 1 month': timedelta(days=30)
}

# Load configuration and environment variables
config = load_config()

# Initialize logging
log = logging.getLogger("root")

dotenv.load_dotenv()

def query_builder(measurement: str, start: str, stop: str, type: str='average') -> str:
    """
    Build a SQL query to get average per variable for a measurement.
    Args:
        measurement: Measurement name (e.g., 'temperature_params')
        start: Start datetime (ISO format)
        stop: End datetime (ISO format)
    Returns:
        SQL query string
    """
    start_iso = start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    end_iso = stop.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    var_map = config["data_mapping"].get(measurement, {})
    var_map_inv = {v: k for k, v in var_map.items()}
    if type == 'average':
        avg_str = [f"AVG({col}) AS {col}" for col in var_map_inv.keys()]  
        return f"SELECT {', '.join(avg_str)} FROM {measurement} WHERE time >= timestamp '{start_iso}' AND time <= timestamp '{end_iso}'"
    elif type == 'ts':
        return f"SELECT * FROM {measurement} WHERE time >= timestamp '{start_iso}' AND time <= timestamp '{end_iso}'"
    else:
        raise ValueError(f"Invalid query type: {type}. Use 'average' or 'ts'.")

class BaseDataFetcher:
    """
    Abstract base class for fetching raw data from InfluxDB.
    """

    def __init__(self, variable_tag: str, debug: bool = False, source: str = "live"):
        """
        Initialize the BaseDataFetcher with InfluxDB credentials and configuration.

        Parameters:
            debug (bool): Flag to enable debug mode.
            source (str): Source of data ('Live' or 'Historical').
            variable_tag (str): The tag in the configuration file that contains the variable names.
        """
        self.debug = debug
        self.source = source.strip().lower()
        self.timezone = pytz.timezone('Asia/Kolkata')  # GMT+5:30
        self.variable_tag = variable_tag
        self.variables = config["data_tags"].get(variable_tag, [])
        self.api_url = config["website_vars"].get("api_url", "")
        self.api_user = os.getenv("USERNAME_REALTIMEDATA")
        self.api_password = os.getenv("PASSWORD_REALTIMEDATA")
        self.measurement_type = self.variable_tag
        self.var_map = config["data_mapping"].get(self.measurement_type, {})


    def _check_and_handle_missing_vars(self, row: dict, context_method: str = "fetch_live_data"):
        """
        Checks for missing variables in the API response and handles error reporting and notification.
        Args:
            row (dict): The dictionary of variables returned from the API.
            context_method (str): The method name for context (default: 'fetch_live_data').
        Raises:
            Exception: If a significant number of expected variables are missing.
        """
        missing_vars = [var for var in self.variables if var not in row]
        total_vars = len(self.variables)
        missing_ratio = len(missing_vars) / total_vars if total_vars > 0 else 0
        if missing_vars:
            import inspect
            frame = inspect.currentframe()
            context_info = (
                f"Datetime: {datetime.now(self.timezone)}\n"
                f"Class: {self.__class__.__name__}\n"
                f"Method: {context_method}\n"
                f"Line: {frame.f_back.f_lineno}\n"
                f"Debug mode: {self.debug}\n"
                f"Source type: {self.source}\n"
                f"API URL: {self.api_url}\n"
                f"Variables expected: {self.variables}\n"
                f"Variables received: {list(row.keys())}\n"
            )
            if missing_ratio < 0.3:
                log.warning(f"API response missing <{missing_ratio*100:.2f}% of expected variables: {missing_vars}\nContext: {context_info}")
                for var in missing_vars:
                    row[var] = 0
            else:
                log.error(f"API response missing >=20% of expected variables: {missing_vars}\nContext: {context_info}")
                raise Exception(f"API response missing expected variables <{missing_ratio*100:.2f}%: {missing_vars}")

    def fetch_live_data(
        self,
    ) -> Dict[str, float]:
        """
        Fetch and process live data.

        Returns:
            dict: A dictionary of live values for each variable in self.variables.
        """
        if self.debug:
            return {var: np.mean(data["values"]) for var, data in self._get_dummy_data().items()}

        params = {
            "user": self.api_user,
            "password": self.api_password     
        }
        # Handle missing API credentials
        if not self.api_user or not self.api_password:
            log.error("API credentials are missing. Please set USERNAME_REALTIMEDATA and PASSWORD_REALTIMEDATA.")
            raise Exception("API credentials are missing.")
        response = requests.get(self.api_url, params=params)
        if response.status_code != 200:
            log.error(f"API call failed with status code: {response.status_code}")
            raise Exception(f"API call failed: {response.text}")
        
        try:
            root = ET.fromstring(response.text)
            json_data = root.text
            json_like_string = json_data.replace("'", '"')
            data = json.loads(json_like_string)
            df = pd.DataFrame(data)
            df.replace('', pd.NA, inplace=True)
            log.info("Successfully fetched and parsed API data.")
            if df.empty:
                log.error("API returned empty data.")
                raise Exception("API returned empty data.")
            # API always returns data for a single timestamp, so take the first row
            row = df.iloc[0].to_dict()
            # Change dtype of all values to float
            for key in row:
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    row[key] = np.nan
            self._check_and_handle_missing_vars(row, context_method="fetch_live_data")
            filtered_row = {k: (v if pd.notna(v) else np.nan) for k, v in row.items() if k in self.variables}
            return filtered_row
        except Exception as e:
            log.error(f"Error parsing API response: {e}")
            raise

    def fetch_averaged_data(
        self,
        average_by: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch and process averaged data over a specified time range.

        Parameters:
            average_by (str): Averaging option ('Live', 'Last 15 minutes', etc.).
            start_time (datetime, optional): Start of the time range (required for 'Over Selected Range').
            end_time (datetime, optional): End of the time range (required for 'Over Selected Range').

        Returns:
            dict: A dictionary of averaged values for each variable.
        """
        if self.debug:
            return {var: np.mean(data["values"]) for var, data in self._get_dummy_data().items()}

        now = datetime.now(timezone.utc) # DB only sotres in UTC
        average_by_norm = average_by.strip().lower()
        if average_by_norm == 'live':
            return self.fetch_live_data()
        elif average_by_norm == 'over selected range':
            if not start_time or not end_time:
                raise ValueError("For 'Over Selected Range', both start_time and end_time must be provided.")
        else:
            if average_by_norm not in TIMEDELTAS:
                raise ValueError(f"Invalid average_by value: {average_by}")
            start_time = now - TIMEDELTAS[average_by_norm]
            end_time = now
        
        database = config["influxdb"].get("database", "bf2_evonith_raw")
        host = config["influxdb"].get("host", "https://eu-central-1-1.aws.cloud2.influxdata.com")
        org = config["influxdb"].get("org", "Blast Furnace, Evonith")

        token = os.environ.get("INFLUX_TOKEN", "")
        query = query_builder(self.measurement_type, start_time, end_time, type="ts")
        client = InfluxDBClient3(host=host,
                                database=database,
                                org=org,
                                token=token,
                                flight_client_options=flight_client_options(
                                    tls_root_certs=cert))
        df = client.query(query=query, mode="pandas")
        client.close()
        return df

    def _get_variable_names(self) -> List[str]:
        """
        Retrieve variable names. Can be overridden in subclasses.

        Returns:
            List[str]: List of variable names.
        """
        return self.variables

    def _get_dummy_data(self) -> Dict[str, Dict[str, List]]:
        """
        Return a default set of dummy data for debugging purposes.

        Returns:
            dict: Dummy timestamps and values for all variables.
        """
        # Document: Returns 100 points for each variable for testing purposes.
        now = datetime.now(self.timezone)
        dummy_data = {}
        for variable in self.variables:
            dummy_data[variable] = {
                "timestamps": [(now - timedelta(minutes=i)).isoformat() for i in range(10)],
                "values": [np.random.random() * 100 for _ in range(100)]
            }
        return dummy_data
    
    def dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Convert a DataFrame with single row to a dictionary with variable names as keys and their values as floats.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            variables (List[str]): List of variable names to extract.

        Returns:
            Dict[str, float]: Dictionary with variable names as keys and their values.
        """
        var_map = config["data_mapping"].get(self.measurement_type, {})
        var_map_inv = {v: k for k, v in var_map.items()}
        # df.rename(columns={old: new for old, new in var_map_inv.items()}, inplace=True)
        dict_df = df.to_dict(orient='list')
        return dict_df