"""Small pure helpers for the Streamlit V-Board page."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import plotly.graph_objs as go


IST = ZoneInfo("Asia/Kolkata")


def absolute_time_range_from_inputs(
    start_date: date,
    start_time: time,
    end_date: date,
    end_time: time,
) -> dict[str, str]:
    start = datetime.combine(start_date, start_time, tzinfo=IST)
    end = datetime.combine(end_date, end_time, tzinfo=IST)
    if start >= end:
        raise ValueError("Start must be before end.")
    return {"kind": "absolute", "start": start.isoformat(), "end": end.isoformat()}


def request_fingerprint(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def utc_range_caption(resolved_range: dict[str, Any]) -> str:
    start = _parse_dt(resolved_range.get("start")).astimezone(timezone.utc)
    end = _parse_dt(resolved_range.get("end")).astimezone(timezone.utc)
    return f"UTC: {start.isoformat().replace('+00:00', 'Z')} to {end.isoformat().replace('+00:00', 'Z')}"


def ist_range_caption(resolved_range: dict[str, Any]) -> str:
    start = _parse_dt(resolved_range.get("start")).astimezone(IST)
    end = _parse_dt(resolved_range.get("end")).astimezone(IST)
    return f"IST: {start.isoformat()} to {end.isoformat()}"




def build_heatload_timeseries_figure(result: dict[str, Any]) -> go.Figure:
    """Build the Streamlit heat-load time-series figure with one shared y-axis."""

    traces = []
    row_id = result.get("row", {}).get("id", "")
    for series in result.get("series", []):
        points = series.get("points", [])
        traces.append(
            go.Scatter(
                x=[point.get("timestamp") for point in points],
                y=[point.get("value") for point in points],
                name=f"{row_id} {series.get('quadrant_id')}",
                mode="lines",
            )
        )
    y_label = result.get("display_label") or result.get("unit") or "Value"
    return go.Figure(
        data=traces,
        layout=go.Layout(
            title="Heat Load Over Time",
            xaxis=dict(title="Time"),
            yaxis=dict(title=y_label),
            height=420,
            margin=dict(t=50, b=40, l=50, r=20),
        ),
    )

def _parse_dt(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed



