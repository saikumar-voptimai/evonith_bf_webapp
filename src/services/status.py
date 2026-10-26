"""Furnace status for the twin-api and the worker (#271).

The furnace-status feature is already layered (pure ``domain.furnace_status``, data access in
``data.furnace_status``); since #271 its cache is ``utils.memo``, so this module exposes the same operations to
callers outside Streamlit. The Streamlit page keeps importing ``data.furnace_status`` directly.
"""

from __future__ import annotations

from data.furnace_status import PARAMETERS, PARAMETERS_BY_KEY, SETTINGS
from data.furnace_status.service import clear_cache, load_status_snapshot, load_trend_view

__all__ = ["PARAMETERS", "PARAMETERS_BY_KEY", "SETTINGS", "clear_cache", "load_status_snapshot", "load_trend_view"]
