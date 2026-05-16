"""Pydantic argument models for FurnaceMind tool calls."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OnlineFetchArgs(_ToolArgs):
    lookback: Optional[str] = Field(
        default=None,
        description="Relative window such as '8h', '2d', '30m', or '1 week'.",
    )
    window: Optional[str] = Field(
        default=None,
        description="Averaging window. If omitted: >1 day uses 1 hour, otherwise 15 minutes.",
    )
    start_time_utc: Optional[str] = Field(
        default=None,
        description="ISO-8601 UTC start, e.g. '2026-05-01T00:30:00Z'.",
    )
    end_time_utc: Optional[str] = Field(
        default=None,
        description="ISO-8601 UTC end. Defaults to now.",
    )
    measurement_groups: Optional[
        List[
            Literal[
                "process_params",
                "cooling_water",
                "heatload_delta_t",
                "delta_t",
                "temperature_profile",
                "miscellaneous",
            ]
        ]
    ] = Field(default=None, description="Online measurement groups to include.")


class OfflineFetchArgs(_ToolArgs):
    report_type: Literal[
        "HM_SLAG",
        "CHARGE",
        "DPR",
        "RAW_MATERIAL_COMPOSITION",
        "RM_COMPOSITION",
        "BURDEN_DISTRIBUTION",
        "HOPPER_MANAGEMENT",
    ] = Field(description="Offline dataset to fetch.")
    table_name: Optional[str] = Field(
        default=None,
        description="Optional explicit table override, e.g. ore_chemistry or charge_data.",
    )
    start_time_utc: Optional[str] = Field(default=None, description="ISO-8601 UTC start time.")
    end_time_utc: Optional[str] = Field(default=None, description="ISO-8601 UTC end time.")
    lookback_days: Optional[int] = Field(
        default=10,
        ge=1,
        le=365,
        description="If start_time_utc is omitted, fetch the last N days.",
    )
    cadence: Optional[Literal["1h", "8h", "1d"]] = Field(
        default=None,
        description="Optional resampling cadence override.",
    )


class MergeArgs(_ToolArgs):
    online_dataset_id: str = Field(description="Dataset id returned by fetch_online_data.")
    offline_dataset_ids: List[str] = Field(description="Dataset ids returned by fetch_offline_data.")
    fill_method: Literal["ffill", "none"] = Field(
        default="ffill",
        description="How to align offline rows onto online timestamps.",
    )


class StaticShiftArgs(_ToolArgs):
    shift_date: str = Field(description="ISO date string YYYY-MM-DD.")
    shift_label: Literal["A", "B", "C"] = Field(
        description="Shift: A (06:00-14:00), B (14:00-22:00), C (22:00-06:00 next day) IST."
    )


class MLDataArgs(_ToolArgs):
    start_time: str = Field(description="Start of range. ISO-8601 or YYYY-MM-DD, treated as IST.")
    end_time: Optional[str] = Field(
        default=None,
        description="End of range. ISO-8601 or YYYY-MM-DD. Defaults to current IST time.",
    )
    resample: Optional[Literal["1h", "4h", "8h", "1d"]] = Field(
        default=None,
        description="Downsampling cadence. Native is 1h.",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Optional keyword substrings to filter columns case-insensitively.",
    )


class ConcatArgs(_ToolArgs):
    dataset_ids: List[str] = Field(
        description="Dataset ids to concatenate vertically in chronological order."
    )
