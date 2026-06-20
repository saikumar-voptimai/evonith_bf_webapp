"""Application settings loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # InfluxDB
    influx_online_token: str = ""
    influx_offline_token: str = ""

    # PostgreSQL
    database_url: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Paths
    results_dir: Path = Path(__file__).resolve().parents[1] / "data" / "results"
    static_dir: Path = Path(__file__).resolve().parents[1] / "data" / "static"

    # Optional: path to a pre-existing legacy CSV to bootstrap the static cache
    legacy_csv_path: str = ""

    # Task cleanup: keep only this many completed result CSVs
    max_result_files: int = 3

    # Static dataset cache settings
    offline_lag_days: int = 3           # days to keep "unconfirmed" (re-fetched each run)
    static_max_versions: int = 3        # how many versioned CSVs to keep

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
