import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import os
import pytz
import numpy as np
import pandas as pd
import requests
import json
import xml.etree.ElementTree as ET
from config.loader import load_config
from .database import get_influx_client

# Load configuration and environment variables
config = load_config()

# Initialize logging
log = logging.getLogger("root")

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
        self.client = get_influx_client()
        self.variables = config["data_tags"].get(variable_tag, [])
        self.api_url = config["website_vars"].get("api_url", "")
        self.api_user = os.getenv("USERNAME_REALTIMEDATA")
        self.api_password = os.getenv("PASSWORD_REALTIMEDATA")


    def fetch_raw_data(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Dict[str, List]]:
        """
        Fetch raw data for all variables within a given time range.

        Parameters:
            start_time (datetime): Start of the time range.
            end_time (datetime): End of the time range.

        Returns:
            dict: A dictionary containing timestamps and values for each variable.
        """
        start_time_utc = start_time.astimezone(pytz.utc).isoformat()
        end_time_utc = end_time.astimezone(pytz.utc).isoformat()

        raw_data = {}
        for variable in self.variables:
            # Sanitize variable name to prevent injection
            safe_variable = str(variable).replace('"', '').replace("'", "")
            query = f"""
            SELECT \"value\"
            FROM \"{safe_variable}\"
            WHERE time >= '{start_time_utc}' AND time <= '{end_time_utc}'
            """
            try:
                log.debug(f"Executing query: {query}")
                result = self.client.query(query=query, language='sql')
                df = result.to_pandas()
            except Exception as e:
                log.error(f"Database query failed for {variable}: {e}")
                raw_data[variable] = {"timestamps": [], "values": []}
                continue
            if not df.empty:
                raw_data[variable] = {
                    "timestamps": df["time"].tolist(),
                    "values": df["value"].tolist()
                }
            else:
                log.warning(f"No data found for sensor {variable} in the specified time range.")
                raw_data[variable] = {"timestamps": [], "values": []}

        return raw_data
    
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
            # Only return variables in self.variables
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
    ) -> Dict[str, float]:
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

        now = datetime.now(self.timezone)

        # Normalize average_by for comparison
        average_by_norm = average_by.strip().lower()
        if average_by_norm == 'live':
            return self.fetch_live_data()
        elif average_by_norm == 'over selected range':
            if not start_time or not end_time:
                raise ValueError("For 'Over Selected Range', both start_time and end_time must be provided.")
        else:
            time_deltas = {
                'last 1 minute': timedelta(minutes=1),
                'last 15 minutes': timedelta(minutes=15),
                'last 1 hour': timedelta(hours=1),
                'last 6 hours': timedelta(hours=6),
                'last 12 hours': timedelta(hours=12),
                'last 1 day': timedelta(days=1),
                'last 1 week': timedelta(weeks=1),
                'last 1 month': timedelta(days=30)
            }
            if average_by_norm not in time_deltas:
                raise ValueError(f"Invalid average_by value: {average_by}")
            start_time = now - time_deltas[average_by_norm]
            end_time = now

        raw_data = self.fetch_raw_data(start_time, end_time)
        averaged_data = {
            variable: np.mean(data["values"]) if data["values"] else np.nan
            for variable, data in raw_data.items()
        }
        return averaged_data

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