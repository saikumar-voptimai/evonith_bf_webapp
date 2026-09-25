#!/usr/bin/env python3

"""Generate the hourly furnace dataset with BMO's normal code pipeline."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "src" / "assets" / "data" / "furnace_dataset.csv"
IST = ZoneInfo("Asia/Kolkata")

load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "furnace_data"))


def _ist_log_time(timestamp: float):
    return datetime.fromtimestamp(timestamp, IST).timetuple()


logging.Formatter.converter = staticmethod(_ist_log_time)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s IST [%(levelname)s] %(message)s",
)
log = logging.getLogger("furnace-dataset-generator")


def generate_dataset() -> Path:
    """Run the same database/online-source pipeline BMO uses without a URL."""
    from data.ml.static_dataset_manager import StaticDatasetManager

    update_time_ist = datetime.now(IST)
    log.info("Update time (IST): %s", update_time_ist.strftime("%Y-%m-%d %H:%M:%S"))

    manager = StaticDatasetManager(DATASET_PATH)
    dataset = manager.update_static("Full")
    saved_path = manager.save(dataset)

    log.info("Updated %s rows x %s columns", len(dataset), len(dataset.columns))
    log.info(
        "Dataset time frame (IST): %s -> %s",
        dataset.index.min(),
        dataset.index.max(),
    )
    log.info("Stable dataset: %s", DATASET_PATH)
    log.info("Snapshot: %s", saved_path)
    return DATASET_PATH


if __name__ == "__main__":
    try:
        generate_dataset()
    except KeyboardInterrupt:
        log.warning("Furnace dataset update interrupted by user.")
        raise SystemExit(130)
    except Exception:
        log.exception("Furnace dataset update failed.")
        raise SystemExit(1)
