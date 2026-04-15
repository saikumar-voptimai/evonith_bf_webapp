"""Compatibility adapter over the shared static dataset manager.

The webapp historically worked with a single CSV file path. The shared core now
maintains versioned CSVs plus cache metadata, so this adapter preserves the old
webapp call shape while delegating the implementation to the shared core.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from furnace_data.dataset.static import CacheMeta, StaticDatasetManager as CoreStaticDatasetManager


class StaticDatasetManager:
    """Backward-compatible webapp wrapper around the shared static manager."""

    def __init__(self, static_path: str | Path) -> None:
        self.static_path = Path(static_path)
        self.static_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_rm_choice = "RM Charge"
        self._core = CoreStaticDatasetManager(
            static_dir=self.static_path.parent,
            legacy_csv_path=self.static_path,
        )

    def update_static(
        self,
        rm_choice: str,
        start_date: date | None = None,
    ) -> pd.DataFrame:
        self._last_rm_choice = rm_choice
        return self._core.update_static(
            rm_choice=rm_choice,
            reprocess_from=start_date,
        )

    def save(self, df: pd.DataFrame) -> None:
        saved_path = self._core.save(df, rm_choice=self._last_rm_choice)
        if saved_path.resolve() != self.static_path.resolve():
            shutil.copyfile(saved_path, self.static_path)

    def get_meta(self) -> CacheMeta | None:
        return self._core.get_meta()

    def current_csv_path(self) -> Path:
        current = self._core.current_csv_path()
        return current if current is not None else self.static_path


__all__ = ["CacheMeta", "StaticDatasetManager"]
