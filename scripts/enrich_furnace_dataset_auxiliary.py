"""Backfill hourly ash/dust context into a furnace-dataset CSV snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "furnace_data"))

from furnace_data.dataset.fetcher import DatasetFetcher  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "src/assets/data/furnace_dataset.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report coverage without changing the CSV.",
    )
    return parser.parse_args()


def _is_auxiliary_column(column: object) -> bool:
    return isinstance(column, str) and (
        column.startswith("ASH_CHEM_") or column.startswith("DUST_")
    )


def _update_cache_metadata(csv_path: Path, frame: pd.DataFrame) -> None:
    meta_path = csv_path.parent / "cache_meta.json"
    if not meta_path.exists():
        return
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["rows"] = len(frame)
    metadata["columns"] = len(frame.columns)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    csv_path = args.csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is required to fetch ash and dust data.")

    dataset = pd.read_csv(csv_path, index_col=0, parse_dates=[0]).sort_index()
    dataset.index = pd.DatetimeIndex(dataset.index, name="time")
    if dataset.empty:
        raise ValueError(f"Cannot enrich empty dataset: {csv_path}")

    enriched = DatasetFetcher().enrich_auxiliary_hourly(dataset)
    auxiliary_columns = [
        column for column in enriched.columns if _is_auxiliary_column(column)
    ]
    populated = int(enriched[auxiliary_columns].notna().sum().sum())
    print(
        f"{csv_path}: {len(enriched)} rows, {len(enriched.columns)} columns, "
        f"{populated} populated auxiliary values"
    )
    if args.dry_run:
        return 0

    temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    try:
        enriched.to_csv(temporary_path, index=True)
        os.replace(temporary_path, csv_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    _update_cache_metadata(csv_path, enriched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
