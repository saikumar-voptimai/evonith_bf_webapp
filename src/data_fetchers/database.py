from influxdb_client_3 import InfluxDBClient3
from config.loader import load_config
import os
from dotenv import load_dotenv
import os

config = load_config()

load_dotenv()

def get_influx_client():
    """
    Initialize and return an InfluxDB client using the configuration.
    """
    config = load_config()
    return InfluxDBClient3(
        host=config["influxdb"]["host"],
        token=os.getenv("TOKEN"),
        org=config["influxdb"]["org"],
        database=config["influxdb"]["database"]
    )
