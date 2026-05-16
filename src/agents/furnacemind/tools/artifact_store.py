"""Artifact storage interfaces for FurnaceMind tools.

This module is backend-safe. It provides a small protocol used by agent tool
adapters and an in-memory default for CLI/API tests. Streamlit implements the
same protocol in ``ui.furnacemind.artifact_adapter``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

import pandas as pd


class ArtifactStore(Protocol):
    """Storage boundary used by FurnaceMind tool adapters."""

    def new_dataset_id(self, prefix: str) -> str:
        """Return a unique dataset id for a newly created artifact."""

    def save_dataset(self, *, dataset_id: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        """Persist a dataframe artifact and mark it active."""

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Return one saved dataframe entry."""

    def get_all_datasets(self) -> Dict[str, Any]:
        """Return all saved dataframe entries."""

    def get_active_df(self) -> Optional[pd.DataFrame]:
        """Return the active dataframe."""

    def get_ml_cache(self) -> Optional[pd.DataFrame]:
        """Return the cached static ML dataset."""

    def set_ml_cache(self, df: pd.DataFrame) -> None:
        """Cache the static ML dataset."""

    def save_figure(self, fig: Any, code: str) -> None:
        """Persist a figure artifact."""

    def append_plot_error(self, error: str) -> None:
        """Record the latest plotting error."""


class InMemoryArtifactStore:
    """Simple process-local artifact store for non-Streamlit callers."""

    def __init__(self) -> None:
        self._datasets: Dict[str, Any] = {}
        self._counter = 0
        self._active_df: Optional[pd.DataFrame] = None
        self._active_meta: Dict[str, Any] = {}
        self._ml_cache: Optional[pd.DataFrame] = None
        self.figure: Any = None
        self.last_plot_code: str | None = None
        self.last_plot_error: str | None = None

    def new_dataset_id(self, prefix: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._counter += 1
        return f"{prefix}_{ts}_{self._counter}"

    def save_dataset(self, *, dataset_id: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        self._datasets[dataset_id] = {"df": df, "meta": meta}
        self._active_df = df
        self._active_meta = meta

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self._datasets.get(dataset_id)

    def get_all_datasets(self) -> Dict[str, Any]:
        return self._datasets

    def get_active_df(self) -> Optional[pd.DataFrame]:
        return self._active_df

    def get_ml_cache(self) -> Optional[pd.DataFrame]:
        return self._ml_cache

    def set_ml_cache(self, df: pd.DataFrame) -> None:
        self._ml_cache = df

    def save_figure(self, fig: Any, code: str) -> None:
        self.figure = fig
        self.last_plot_code = code

    def append_plot_error(self, error: str) -> None:
        self.last_plot_error = error


_artifact_store: ArtifactStore = InMemoryArtifactStore()


def set_artifact_store(store: ArtifactStore) -> None:
    """Install the artifact store used by tool adapters."""
    global _artifact_store
    _artifact_store = store


def get_artifact_store() -> ArtifactStore:
    """Return the configured artifact store."""
    return _artifact_store
