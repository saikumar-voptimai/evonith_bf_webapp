"""
/data/online and /data/offline routes — synchronous InfluxDB fetch endpoints.

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

from app.core.offline_fetcher import OFFLINE_REPORT_MAP, fetch_offline
from app.core.neon_offline_fetcher import fetch_neon_offline
from furnace_data.neon_db.offline import NEON_OFFLINE_REPORT_MAP, NEON_OFFLINE_TABLES
from app.core.online_fetcher import ONLINE_MEASUREMENTS, fetch_online, list_measurements
from app.models.schemas import (
    DataFetchResponse,
    DataMeta,
    OfflineFetchRequest,
    OnlineFetchRequest,
    ResponseFormat,
    RmLiveFetchRequest,
)
from furnace_data.influx.offline import clean_rm_data, fetch_offline_data
from furnace_data.config import load_config as _load_config

_cfg = _load_config("setting_ds_dv.yml")
_RM_MEASUREMENT: str = _cfg["offline_measurements"]["Bunker Report"]
_OFFLINE_BUCKET: str = _cfg["influx_offline"]["database"]

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
    """Serialise a DataFrame to either a JSON response or a streaming CSV download.

    Args:
         - df: pd.DataFrame - Data to serialise; index is reset and renamed to "time".
         - fmt: ResponseFormat - Desired output format (json or csv).
         - filename: str - Attachment filename used in the Content-Disposition header for CSV.
         - meta: DataMeta - Metadata block embedded in the JSON envelope.

    Returns:
         - DataFetchResponse for JSON, StreamingResponse for CSV.
    """
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
    """List all available online InfluxDB measurements and their field names.

    Returns:
         - dict with key "measurements" mapping measurement names to their field lists.
    """
    return {"measurements": list_measurements()}


@router.post("/online/fetch")
def fetch_online_data(req: OnlineFetchRequest):
    """Fetch one or more online measurements from InfluxDB.

    Provide either `preset` (e.g. "last 8 hours") or `start_time` + `end_time`.
    Multiple measurements are outer-joined on the time index.

    Args:
         - req: OnlineFetchRequest - Validated request body containing measurements, time range, query type, and format options.

    Returns:
         - DataFetchResponse (JSON) or StreamingResponse (CSV).
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
def fetch_offline_data(req: OfflineFetchRequest):
    """Fetch offline report data (HM/slag, charge, RM composition, DPR).

    Provide either `preset` or `start_time` + `end_time`.
    Defaults to existing InfluxDB behavior; pass `source="neon_db"` for Neon.

    Args:
         - req: OfflineFetchRequest - Validated request body with report type, backend source, and time range.

    Returns:
         - DataFetchResponse (JSON) or StreamingResponse (CSV).
    """
    if not req.preset and (req.start_time is None or req.end_time is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'preset' or both 'start_time' and 'end_time'.",
        )

    try:
        table_name = None
        query_type = None
        window = None
        if req.source.value == "neon_db":
            table_name = req.table_name.value if req.table_name else None
            table_name = table_name or NEON_OFFLINE_REPORT_MAP.get(req.report_type.value)
            query_type = req.query_type.value
            window = req.window if req.query_type.value == "windowed-average" else None
            df = fetch_neon_offline(
                report_type=req.report_type.value,
                start_time=req.start_time,
                end_time=req.end_time,
                preset=req.preset,
                table_name=table_name,
                query_type=query_type,
                window=window,
            )
        else:
            df = fetch_offline(
                report_type=req.report_type.value,
                start_time=req.start_time,
                end_time=req.end_time,
                preset=req.preset,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Offline fetch failed")
        backend = "Neon DB" if req.source.value == "neon_db" else "InfluxDB"
        raise HTTPException(status_code=502, detail=f"{backend} error: {e}")

    if df.empty:
        raise HTTPException(status_code=204, detail="No data returned for the requested parameters.")

    meta = DataMeta(
        report_type=req.report_type.value,
        source=req.source.value,
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
    """List available offline report types and their InfluxDB measurement names.

    Returns:
         - dict mapping report type keys to InfluxDB measurement strings.
    """
    return {k: v for k, v in OFFLINE_REPORT_MAP.items()}


@router.get("/offline/neon-tables")
def get_neon_tables() -> Dict[str, Any]:
    """List available Neon offline tables and the report-type-to-table defaults.

    Returns:
         - dict with "report_map" (report type → table name) and "tables" (table name → sorted column list).
    """
    return {
        "report_map": dict(NEON_OFFLINE_REPORT_MAP),
        "tables": {table: sorted(columns) for table, columns in NEON_OFFLINE_TABLES.items()},
    }


# ---------------------------------------------------------------------------
# Live RM data endpoint
# ---------------------------------------------------------------------------

@router.post("/rm/live")
def fetch_rm_live(req: RmLiveFetchRequest):
    """Fetch and clean the latest Raw Material composition data from InfluxDB.

    Fetches ``rm_updated_data`` from the offline bucket for the requested
    lookback window and applies :func:`furnace_data.influx.offline.clean_rm_data`
    to drop all-NaN ore groups and rename ore variants.

    Args:
         - req: RmLiveFetchRequest - Validated request body with lookback_days, cadence, and format.

    Returns:
         - DataFetchResponse (JSON) or StreamingResponse (CSV).
    """
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=req.lookback_days)

    try:
        df = fetch_offline_data(
            measurement=_RM_MEASUREMENT,
            time_range=(start_time, now),
            database=_OFFLINE_BUCKET,
        )
    except Exception as e:
        log.exception("RM live fetch failed")
        raise HTTPException(status_code=502, detail=f"InfluxDB error: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=204, detail="No RM data returned for the requested window.")

    try:
        df = clean_rm_data(df)
    except Exception as e:
        log.warning("clean_rm_data failed (returning raw): %s", e)

    meta = DataMeta(
        report_type="RM_LIVE",
        start=str(df.index.min()),
        end=str(df.index.max()),
        rows=len(df),
        columns=len(df.columns),
    )

    filename = f"rm_live_{meta.start[:10]}.csv"
    return _df_to_response(df, req.format, filename, meta)
