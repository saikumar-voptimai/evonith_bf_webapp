"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings

from furnace_data.runtime_paths import get_dataset_results_dir, get_dataset_static_dir

_BACKEND_APP_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_STATIC_DIR = _BACKEND_APP_ROOT / "data" / "static"
_LEGACY_STATIC_CSV = _LEGACY_STATIC_DIR / "ml_dataset.csv"


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
    results_dir: Path = get_dataset_results_dir()
    static_dir: Path = get_dataset_static_dir()

    # Optional: path to a pre-existing legacy CSV to bootstrap the static cache
    legacy_csv_path: str = str(_LEGACY_STATIC_CSV) if _LEGACY_STATIC_CSV.exists() else ""
    legacy_static_dir: Path = _LEGACY_STATIC_DIR

    # Task cleanup: keep only this many completed result CSVs
    max_result_files: int = 3

    # Static dataset cache settings
    offline_lag_days: int = 3           # days to keep "unconfirmed" (re-fetched each run)
    static_max_versions: int = 3        # how many versioned CSVs to keep
    dataset_job_ttl_hours: int = 24
    dataset_job_workers: int = 1
    dataset_max_build_range_days: int = 366

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
