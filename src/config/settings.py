import os
from dotenv import load_dotenv

load_dotenv()

# Cloud database settings
DB_URL = os.getenv("DB_URL")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# General settings
APP_NAME = "Streamlit Manufacturing Dashboard"
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")