"""
/data/online and /data/offline routes - synchronous fetch endpoints.

These are lightweight, synchronous endpoints. No background tasks.
For large time ranges the connection stays open until the fetch completes.
"""

import io
import logging
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from datetime import datetime, timedelta, timezone

from app.core.offline_fetcher import fetch_database_offline
from app.core.online_fetcher import ONLINE_MEASUREMENTS, fetch_online, list_measurements
from app.models.schemas import (
    DataFetchResponse,
    DataMeta,
    OfflineFetchRequest,
    OnlineFetchRequest,
    ResponseFormat,
    RmLiveFetchRequest,
)
from furnace_data.offline import (
    OFFLINE_REPORT_MAP,
    list_offline_tables,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_response(
    df: pd.DataFrame,
    fmt: ResponseFormat,
    filename: str,
    meta: DataMeta,
) -> Any:
    """Return JSON response or streaming CSV depending on requested format."""
    if fmt == ResponseFormat.csv:
        buf = io.StringIO()
        df.to_csv(buf, index=True)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # JSON: reset index so time appears as a column
    # rename_axis ensures the index column is always called "time" regardless of
    # whether the fetcher named the index (production InfluxDB data) or not (tests)
    df_out = df.rename_axis("time").reset_index()
    df_out["time"] = df_out["time"].astype(str)
    return DataFetchResponse(
        meta=meta,
        columns=list(df_out.columns),
        data=df_out.to_dict(orient="records"),
    )


# ---------------------------------------------------------------------------
# Online endpoints
# ---------------------------------------------------------------------------

@router.get("/online/measurements")
def get_measurements() -> Dict[str, Any]:
    """List all available online measurements and their field names."""
    return {"measurements": list_measurements()}


@router.post("/online/fetch")
def fetch_online_data(req: OnlineFetchRequest):
    """
    Fetch one or more online measurements from InfluxDB.

    Provide either `preset` (e.g. "last 8 hours") or `start_time` + `end_time`.
    Multiple measurements are outer-joined on the time index.
    """
    # Validate measurements
    unknown = [m for m in req.measurements if m not in ONLINE_MEASUREMENTS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown measurement(s): {unknown}. Valid: {ONLINE_MEASUREMENTS}",
        )

    if not req.preset and (req.start_time is None or req.end_time is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'preset' or both 'start_time' and 'end_time'.",
        )

    try:
        df = fetch_online(
            measurements=req.measurements,
            query_type=req.query_type.value,
            window=req.window,
            start_time=req.start_time,
            end_time=req.end_time,
            preset=req.preset,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Online fetch failed")
        raise HTTPException(status_code=502, detail=f"InfluxDB error: {e}")

    if df.empty:
        raise HTTPException(status_code=204, detail="No data returned for the requested parameters.")

    meta = DataMeta(
        measurements=req.measurements,
        query_type=req.query_type.value,
        window=req.window,
        start=str(df.index.min()),
        end=str(df.index.max()),
        rows=len(df),
        columns=len(df.columns),
    )

    filename = f"online_{'_'.join(req.measurements)}_{meta.start[:10]}.csv"
    return _df_to_response(df, req.format, filename, meta)


# ---------------------------------------------------------------------------
# Offline endpoints
# ---------------------------------------------------------------------------

@router.post("/offline/fetch")
def fetch_offline_endpoint(req: OfflineFetchRequest):
    """
    Fetch offline report data from PostgreSQL.

    Provide either `preset` or `start_time` + `end_time`.
    """
    if not req.preset and (req.start_time is None or req.end_time is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'preset' or both 'start_time' and 'end_time'.",
        )

    try:
        table_name = req.table_name if req.table_name else None
        query_type = req.query_type.value
        window = req.window if query_type == "windowed-average" else None
        df = fetch_database_offline(
            report_type=req.report_type.value,
            start_time=req.start_time,
            end_time=req.end_time,
            preset=req.preset,
            table_name=table_name,
            query_type=query_type,
            window=window,
        )
        if table_name is None:
            mapped_tables = OFFLINE_REPORT_MAP.get(req.report_type.value, [])
            table_name = ",".join(mapped_tables) if mapped_tables else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Offline fetch failed")
        raise HTTPException(status_code=502, detail=f"Offline database error: {e}")

    if df.empty:
        raise HTTPException(status_code=204, detail="No data returned for the requested parameters.")

    meta = DataMeta(
        report_type=req.report_type.value,
        source="offline_db",
        table_name=table_name,
        query_type=query_type,
        window=window,
        start=str(df.index.min()),
        end=str(df.index.max()),
        rows=len(df),
        columns=len(df.columns),
    )

    filename = f"offline_{req.report_type.value.lower()}_{meta.start[:10]}.csv"
    return _df_to_response(df, req.format, filename, meta)


@router.get("/offline/report-types")
def get_report_types() -> Dict[str, str]:
    """List available offline database report types and their mapped tables."""
    return {k: ",".join(v) for k, v in OFFLINE_REPORT_MAP.items()}


@router.get("/offline/tables")
def get_offline_tables() -> Dict[str, Any]:
    """List available offline database reports, tables, and whitelisted columns."""
    return list_offline_tables()


# ---------------------------------------------------------------------------
# Live RM data endpoint
# ---------------------------------------------------------------------------

@router.post("/rm/live")
def fetch_rm_live(req: RmLiveFetchRequest):
    """
    Fetch the latest Raw Material composition data from the offline database.

    Provide ``lookback_days`` (1-365) and ``cadence`` (``"8h"`` | ``"1h"`` | ``"1d"``).
    """
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=req.lookback_days)

    try:
        df = fetch_database_offline(
            report_type="RM_COMPOSITION",
            start_time=start_time,
            end_time=now,
            preset=None,
            query_type="ts",
        )
    except Exception as e:
        log.exception("RM live fetch failed")
        raise HTTPException(status_code=502, detail=f"Offline database error: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=204, detail="No RM data returned for the requested window.")

    meta = DataMeta(
        report_type="RM_LIVE",
        source="offline_db",
        start=str(df.index.min()),
        end=str(df.index.max()),
        rows=len(df),
        columns=len(df.columns),
    )

    filename = f"rm_live_{meta.start[:10]}.csv"
    return _df_to_response(df, req.format, filename, meta)
