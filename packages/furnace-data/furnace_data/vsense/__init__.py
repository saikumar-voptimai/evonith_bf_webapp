"""Shared V-Sense domain catalog, context, bounds, and optimizer helpers."""

from furnace_data.vsense.catalog import (
    ALGORITHM_VERSION,
    CATALOG_VERSION,
    DISPLAY_TIMEZONE,
    load_vsense_catalog,
    optimization_by_id,
    parameter_by_id,
)
from furnace_data.vsense.context import VSenseContextError, build_context_snapshot
from furnace_data.vsense.optimizer import run_legacy_optimization

__all__ = [
    "ALGORITHM_VERSION",
    "CATALOG_VERSION",
    "DISPLAY_TIMEZONE",
    "VSenseContextError",
    "build_context_snapshot",
    "load_vsense_catalog",
    "optimization_by_id",
    "parameter_by_id",
    "run_legacy_optimization",
]
