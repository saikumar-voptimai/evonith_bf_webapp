"""Application settings loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # InfluxDB
    influx_online_token: str = ""
    influx_offline_token: str = ""

    # PostgreSQL (Neon)
    database_url: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    results_dir: Path = Path(__file__).resolve().parents[1] / "data" / "results"
    static_dataset_path: Path = Path(__file__).resolve().parents[1] / "data" / "static" / "ml_dataset.csv"

    # Task cleanup: keep only this many completed result CSVs
    max_result_files: int = 3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
