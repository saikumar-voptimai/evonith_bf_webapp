from pathlib import Path

import yaml


def load_config(config_file: str = "setting_ds_dv.yml"):
    """
    Load configuration from a YAML file.
    """
    fullpath = Path(__file__).resolve().parents[1] / "config" / config_file
    config_file_path = Path(fullpath).resolve()
    if not config_file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    with open(config_file_path, "r") as file:
        return yaml.safe_load(file)


def get_furnace_dataset_url(
    config: dict | None = None, *, override_url: str | None = None
) -> str | None:
    """Return the dataset URL only when URL fetching is enabled."""
    fetch_url = (load_config("furnacemind.yml") or {}).get("fetch_url", False)
    enabled = fetch_url is True or str(fetch_url).strip().lower() == "true"
    if not enabled:
        return None

    settings = config if config is not None else load_config("setting_ds_dv.yml")
    override = str(override_url or "").strip()
    url = override or settings.get("DATA_URL", "")
    return str(url or "").strip() or None
