"""Domain-specific InfluxDB fetchers.

These fetchers extend :class:`~furnace_data.influx.base.BaseDataFetcher` with
blast-furnace-specific aggregation logic (quadrant weighting, elevation
grouping) that is independent of any UI framework.

Visualisation-specific reshaping (contour matrix building, colour maps) remains
in ``src/plotters/`` in the webapp.

Public API
----------
TemperatureDataFetcher      110-sensor wall temperature profile fetcher.
AverageHeatLoadDataFetcher  Per-row, per-quadrant heat-load min/mean/max fetcher.
TsHeatloadDataFetcher       Time-series heat-load fetcher.
"""

from furnace_data.domain.heatload import AverageHeatLoadDataFetcher, TsHeatloadDataFetcher
from furnace_data.domain.temperature import TemperatureDataFetcher

__all__ = [
    "AverageHeatLoadDataFetcher",
    "TemperatureDataFetcher",
    "TsHeatloadDataFetcher",
]
