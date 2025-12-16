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
from config.config_loader import load_config

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
        'last 8 hours': timedelta(hours=8),
        'last 12 hours': timedelta(hours=12),
        'last 1 day': timedelta(days=1),
        'last 3 days': timedelta(days=3),
        'last 1 week': timedelta(weeks=1),
        'last 2 weeks': timedelta(weeks=2),
        'last 1 month': timedelta(days=30),
        'last 2 months': timedelta(days=60),
        'last 3 months': timedelta(days=90)
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
        type: 'average' for averaged values, 'ts' for time-series data (for resampling we use the 'ts')
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

    def __init__(self, 
                 variable_tag: str, 
                 debug: bool = False, 
                 source: str = "live",
                 database: str = "bf2_evonith_raw",
                 host: str = "https://eu-central-1-1.aws.cloud2.influxdata.com",
                 org: str = "Blast Furnace, Evonith",
                 token: str = "INFLUX_ONLINE_TOKEN"):
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
        self.average_window = None
        self.database = database
        self.host = host
        self.org = org

        self.token = token

    def fetch_averaged_data(
        self,
        recent_data_of: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch and process averaged data over a specified time range.

        Parameters:
            recent_data_of (str): Recent data with options option ('Live', 'Last 15 minutes', etc.).
            start_time (datetime, optional): Start of the time range (required for 'Over Selected Range').
            end_time (datetime, optional): End of the time range (required for 'Over Selected Range').

        Returns:
            dict: A dictionary of averaged values for each variable.
        """
        if self.debug:
            return {var: np.mean(data["values"]) for var, data in self._get_dummy_data().items()}

        now = datetime.now(timezone.utc) # DB only sotres in UTC
        recent_data_of_norm = recent_data_of.strip().lower()
        if recent_data_of_norm == 'live':
            return self.fetch_live_data()
        elif recent_data_of_norm == 'over selected range':
            if not start_time or not end_time:
                raise ValueError("For 'Over Selected Range', both start_time and end_time must be provided.")
        else:
            if recent_data_of_norm not in TIMEDELTAS:
                raise ValueError(f"Invalid recent_data option: {recent_data_of}")
            start_time = now - TIMEDELTAS[recent_data_of_norm]
            end_time = now
        
        if self.database != config["influxdb"].get("database", self.database):
            database = self.database
        else:
            log.warning(f"Database {self.database} not found in config. Using default database.")
            database = config["influxdb"].get("database", self.database)
        
        if self.host != config["influxdb"].get("host", self.host):
            host = self.host
        else:
            log.warning(f"Host {self.host} not found in config.")
            host = config["influxdb"].get("host", self.host)
        
        if self.org != config["influxdb"].get("org", self.org):
            org = self.org
        else:
            log.warning(f"Org {self.org} not found in config. Using default database.")
            org = config["influxdb"].get("org", self.org)
        
        token = os.environ.get(self.token, "")
        query = query_builder(
                self.measurement_type,
                start_time,
                end_time,
                type='ts'
        )
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