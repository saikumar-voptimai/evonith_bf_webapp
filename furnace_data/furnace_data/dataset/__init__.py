"""ML/furnace dataset management.

Provides
--------
DatasetFetcher        Range-aware cached dataset fetcher (renamed from MlDatasetFetcher).
DatasetService        4-step fetch pipeline (renamed from MlDatasetService).
StaticDatasetManager  Lag-aware incremental CSV update manager.
DataCleaner           Configurable cleaning pipeline.
build_default_config  Build the default CleaningConfig for BF2 data.
"""

from furnace_data.dataset.cleaning import DataCleaner, build_default_config
from furnace_data.dataset.fetcher import DatasetFetcher
from furnace_data.dataset.service import DatasetService
from furnace_data.dataset.static import StaticDatasetManager

__all__ = [
    "DataCleaner",
    "DatasetFetcher",
    "DatasetService",
    "StaticDatasetManager",
    "build_default_config",
]
