from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from furnace_data.dataset.static_csv import get_static_dataset_path, load_static_dataset
from furnace_data.optimization_runtime.types import OptimizationContext


class DatasetContextService:
    """Shared dataset bootstrap service for optimisation pages."""

    def __init__(
        self,
        *,
        static_dataset_path: str | Path | None = None,
        refresh_enabled: bool = False,
        refresh_rm_choice: str = "RM Charge",
    ) -> None:
        self.static_dataset_path = static_dataset_path
        self.refresh_enabled = bool(refresh_enabled)
        self.refresh_rm_choice = str(refresh_rm_choice or "RM Charge")

    def maybe_refresh(self, config: dict[str, Any] | None = None) -> bool:
        """Runtime refresh is handled by app-level services, not shared package code."""
        return False

    def resolve_static_path(self) -> Path:
        return get_static_dataset_path(
            str(self.static_dataset_path) if self.static_dataset_path else None
        )

    def load_history(self) -> pd.DataFrame:
        df = load_static_dataset(path=self.resolve_static_path())
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        df = df[~df.index.isna()]
        return df.sort_index()

    def build_context(
        self,
        *,
        process_context: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizationContext:
        history_df = self.load_history()
        latest_row = (
            history_df.iloc[-1].copy()
            if not history_df.empty
            else pd.Series(dtype=float)
        )
        context = {
            str(key): float(value)
            for key, value in latest_row.to_dict().items()
            if value is not None and pd.notna(value)
        }
        if process_context:
            context.update(
                {
                    str(key): float(value)
                    for key, value in process_context.items()
                    if value is not None
                }
            )

        return OptimizationContext(
            history_df=history_df,
            latest_row=latest_row,
            process_context=context,
            metadata=metadata or {},
        )