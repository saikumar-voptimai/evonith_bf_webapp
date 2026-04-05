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

from app.core.offline_fetcher import OFFLINE_REPORT_MAP, fetch_offline
from app.core.online_fetcher import ONLINE_MEASUREMENTS, fetch_online, list_measurements
from app.models.schemas import (
    DataFetchResponse,
    DataMeta,
    OfflineFetchRequest,
    OnlineFetchRequest,
    ResponseFormat,
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
def fetch_offline_data(req: OfflineFetchRequest):
    """
    Fetch offline report data (HM/slag, charge, RM composition, DPR) from InfluxDB.

    Provide either `preset` or `start_time` + `end_time`.
    """
    if not req.preset and (req.start_time is None or req.end_time is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'preset' or both 'start_time' and 'end_time'.",
        )

    try:
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
        raise HTTPException(status_code=502, detail=f"InfluxDB error: {e}")

    if df.empty:
        raise HTTPException(status_code=204, detail="No data returned for the requested parameters.")

    meta = DataMeta(
        report_type=req.report_type.value,
        start=str(df.index.min()),
        end=str(df.index.max()),
        rows=len(df),
        columns=len(df.columns),
    )

    filename = f"offline_{req.report_type.value.lower()}_{meta.start[:10]}.csv"
    return _df_to_response(df, req.format, filename, meta)


@router.get("/offline/report-types")
def get_report_types() -> Dict[str, str]:
    """List available offline report types and their InfluxDB measurement names."""
    return {k: v for k, v in OFFLINE_REPORT_MAP.items()}
