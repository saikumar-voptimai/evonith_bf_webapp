"""Furnace status: latest snapshot of every catalogue parameter and trends (SV-2.1, #293).

The snapshot is the Streamlit V-Board's (``services.status``): value, unit, time and limit state of every parameter of
``config/furnace_status.yml``, zones with their related readings, operating state and freshness. While event-stream
clients are connected the worker takes it every 10 s and keeps it in Redis (``services.live_status``); the route
answers from there while it is at most 25 s old, else it takes the snapshot itself (once per 10-s slot and process).
Trends are the V-Board's raw series over 1/4/8/16/24 h or a custom range of up to 7 days, min/max-decimated.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request

from api.auth import require_role
from api.errors import ApiError
from services import live_status
from services import status as status_service
from services.status import StatusSettings, StatusSnapshotView, StatusTrend

log = logging.getLogger("api.status")

router = APIRouter(prefix="/api/status", tags=["status"], dependencies=[Depends(require_role("user"))])

TrendInterval = Literal[status_service.TREND_INTERVALS]  # type: ignore[valid-type]  # 1h ... 24h from the catalogue


@router.get("/settings", response_model=StatusSettings, summary="Furnace status display rules")
def settings() -> StatusSettings:
    """The display rules of ``config/furnace_status.yml`` (SV-2.2): stale-after and live-age minutes, the refresh
    choices, the trend intervals and the longest custom range, the operating-state rule, sections and zones. The
    Furnace Status page reads them here instead of holding copies."""
    return status_service.settings_view()


@router.get("/snapshot", response_model=StatusSnapshotView, summary="Latest furnace status snapshot")
async def snapshot(request: Request) -> StatusSnapshotView:
    """Every catalogue parameter with its latest reading of the last 15 min (value, unit, time, state against its
    limits), the zones with their related readings, the operating state and the freshness of the data. From the
    worker's live snapshot while it is fresh (``freshness.source`` producer), else taken for the request."""
    live = getattr(request.app.state, "live", None)
    redis = getattr(live, "redis", None)
    if redis is not None:
        try:
            latest = await live_status.read_latest(redis, now=status_service.clock())
        except Exception as exc:  # noqa: BLE001 - Redis briefly unavailable: take the snapshot here
            log.warning("live snapshot not read from Redis (%s)", type(exc).__name__)
            latest = None
        if latest is not None:
            return latest
    return await asyncio.to_thread(status_service.current_snapshot_view)


@router.get("/trend", response_model=StatusTrend, summary="Furnace status trends")
def trend(
    parameters: Annotated[
        str,
        Query(
            description=(
                f"Comma-separated catalogue keys, at most {status_service.MAX_TREND_PARAMETERS} "
                "(e.g. hot_blast_volume,raft)"
            )
        ),
    ],
    interval: Annotated[TrendInterval, Query(description="Range ending now, or custom (start and end)")] = "8h",
    start: Annotated[
        datetime | None, Query(description="Start of a custom range; a time without offset is IST")
    ] = None,
    end: Annotated[
        datetime | None, Query(description="End of a custom range (at most 7 days after start; clamped to now)")
    ] = None,
    max_points: Annotated[
        int, Query(ge=100, le=10_000, description="Most points per series, count (min/max decimation above it)")
    ] = status_service.DEFAULT_TREND_POINTS,
) -> StatusTrend:
    """Raw readings of each parameter over the range (the V-Board's trend: setpoint series where the parameter has
    one, a null where the data has a gap), the minimum, maximum and mean of the raw range and the current reading."""
    keys = list(dict.fromkeys(k.strip() for k in parameters.split(",") if k.strip()))
    if not keys:
        raise ApiError(422, "Name at least one parameter.")
    if len(keys) > status_service.MAX_TREND_PARAMETERS:
        raise ApiError(422, f"At most {status_service.MAX_TREND_PARAMETERS} parameters per request.")
    unknown = [k for k in keys if k not in status_service.PARAMETERS_BY_KEY]
    if unknown:
        raise ApiError(422, f"Unknown parameter: {', '.join(unknown)}.")
    try:
        return status_service.build_trend(keys, interval, start, end, max_points=max_points)
    except status_service.RangeError as exc:
        raise ApiError(422, str(exc)) from exc
