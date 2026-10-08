"""Live source adapter for the production BF2 silicon forecast."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from furnace_data.influx.online import fetch_online_df
from furnace_data.offline import fetch_offline_data
from utils.bmo.si_forecast.features import (
    CHARGE_COLUMNS,
    GATE_COLUMNS,
    LAB_COLUMNS,
    PLANT_TZ,
    TARGET_WINDOW_MINUTES,
)


@dataclass(frozen=True)
class SiliconForecastSources:
    online: pd.DataFrame
    labs: pd.DataFrame
    charge: pd.DataFrame
    hourly_gate: pd.DataFrame
    fetched_at: pd.Timestamp


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


def _aware_plant(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(PLANT_TZ)
    return stamp.tz_convert(PLANT_TZ)


def _offline_plant_clock(values: pd.Series) -> pd.Series:
    """Normalise offline clocks to IST without shifting naive plant timestamps."""

    converted: list[pd.Timestamp] = []
    for value in values:
        if value is None or pd.isna(value):
            converted.append(pd.NaT)
            continue
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(PLANT_TZ)
        else:
            stamp = stamp.tz_convert(PLANT_TZ)
        converted.append(stamp)
    return pd.Series(converted, index=values.index)


def _offline_frame(
    frame: pd.DataFrame, *, time_name: str = "time (IST)"
) -> pd.DataFrame:
    out = frame.reset_index().rename(columns={frame.index.name or "index": time_name})
    if time_name not in out and "time" in out:
        out = out.rename(columns={"time": time_name})
    if time_name in out:
        out[time_name] = _offline_plant_clock(out[time_name])
    if "created_at" in out:
        out["created_at"] = _offline_plant_clock(out["created_at"])
    return out


class LiveSiliconForecastSource:
    """Read the exact raw feeds required by one model version."""

    def __init__(
        self,
        *,
        bundle_dir: str | Path,
        static_dataset_path: str | Path,
        online_fetch: Callable[..., pd.DataFrame] = fetch_online_df,
        offline_fetch: Callable[..., pd.DataFrame] = fetch_offline_data,
        plant_timezone: str = PLANT_TZ,
        online_history_hours: int = 14,
        lab_history_days: int = 7,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.static_dataset_path = _repo_path(static_dataset_path)
        self.online_fetch = online_fetch
        self.offline_fetch = offline_fetch
        self.plant_timezone = str(plant_timezone)
        self.online_history_hours = max(13, int(online_history_hours))
        self.lab_history_days = max(2, int(lab_history_days))
        self.channel_map = json.loads(
            (self.bundle_dir / "online_channel_map.json").read_text(encoding="utf-8")
        )

    def _online(
        self, origin_start: pd.Timestamp, origin_end: pd.Timestamp
    ) -> pd.DataFrame:
        mappings = self.channel_map
        measurements = list(dict.fromkeys(row["measurement"] for row in mappings))
        fields: dict[str, list[str]] = {}
        for row in mappings:
            fields.setdefault(str(row["measurement"]), []).append(str(row["field"]))
        desired_start = origin_start - pd.Timedelta(hours=self.online_history_hours)
        query_start = desired_start - pd.Timedelta(minutes=10)
        latest_end = origin_end.floor("10min") - pd.Timedelta(minutes=10)
        frame = self.online_fetch(
            selected_measurements=measurements,
            time_range="last 1 day",
            request_type="windowed-average",
            window_by="10 minutes",
            start_time_override=query_start.tz_convert("UTC").to_pydatetime(),
            end_time_override=latest_end.tz_convert("UTC").to_pydatetime(),
            column_naming="field",
            fields_by_measurement=fields,
        )
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["time (IST)"])
        online = frame.copy()
        index = pd.DatetimeIndex(pd.to_datetime(online.index, utc=True, errors="coerce"))
        index = index.tz_convert(self.plant_timezone)
        valid = (
            index.notna()
            & (index == index.floor("10min"))
            & (index >= desired_start)
            & (index <= latest_end)
        )
        online = online.loc[valid].copy()
        online.index = index[valid]
        rename = {str(row["field"]): str(row["source_display_column"]) for row in mappings}
        online = online.rename(columns=rename)
        wanted = [str(row["source_display_column"]) for row in mappings]
        present = [column for column in wanted if column in online]
        online = online.loc[:, present]
        online.insert(0, "time (IST)", online.index)
        return online.reset_index(drop=True)

    def _labs(
        self, origin_start: pd.Timestamp, lab_end: pd.Timestamp
    ) -> pd.DataFrame:
        start = origin_start - pd.Timedelta(days=self.lab_history_days)
        frame = self.offline_fetch(
            table_name="offline_feed.hot_metal_slag_analysis",
            time_range=(start.tz_convert("UTC"), lab_end.tz_convert("UTC")),
            query_type="raw",
            columns=list(LAB_COLUMNS),
        )
        return _offline_frame(frame)

    def _charge(
        self, origin_start: pd.Timestamp, origin_end: pd.Timestamp
    ) -> pd.DataFrame:
        frame = self.offline_fetch(
            table_name="offline_feed.charge_data",
            time_range=(
                (origin_start - pd.Timedelta(hours=4)).tz_convert("UTC"),
                origin_end.tz_convert("UTC"),
            ),
            query_type="raw",
            columns=list(CHARGE_COLUMNS),
        )
        return _offline_frame(frame)

    def _hourly_gate(
        self, origin_start: pd.Timestamp, origin_end: pd.Timestamp
    ) -> pd.DataFrame:
        columns = ["time", *GATE_COLUMNS]
        gate = pd.read_csv(self.static_dataset_path, usecols=columns)
        clock = pd.to_datetime(gate["time"], errors="coerce", format="mixed")
        if getattr(clock.dt, "tz", None) is None:
            clock = clock.dt.tz_localize(self.plant_timezone)
        else:
            clock = clock.dt.tz_convert(self.plant_timezone)
        gate["time"] = clock
        start = origin_start.floor("h") - pd.Timedelta(hours=1)
        end = origin_end.floor("h") - pd.Timedelta(hours=1)
        return gate[gate["time"].between(start, end)].copy()

    def fetch_range(
        self,
        *,
        origin_start: Any,
        origin_end: Any,
        lab_end: Any | None = None,
    ) -> SiliconForecastSources:
        cadence = f"{TARGET_WINDOW_MINUTES}min"
        start = _aware_plant(origin_start).floor(cadence)
        end = _aware_plant(origin_end).floor(cadence)
        if end < start:
            raise ValueError("Silicon source range ends before it starts.")
        labs_until = _aware_plant(lab_end if lab_end is not None else end)
        # These feeds are independent.  PostgreSQL HM/charge reads account for
        # most of the cold-page delay, so overlap them with the Influx query.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="bf2-si-source") as pool:
            online = pool.submit(self._online, start, end)
            labs = pool.submit(self._labs, start, labs_until)
            charge = pool.submit(self._charge, start, end)
            hourly_gate = self._hourly_gate(start, end)
            return SiliconForecastSources(
                online=online.result(),
                labs=labs.result(),
                charge=charge.result(),
                hourly_gate=hourly_gate,
                fetched_at=pd.Timestamp.now(tz="UTC"),
            )

    def fetch(self, origin: Any) -> SiliconForecastSources:
        stamp = _aware_plant(origin).floor(f"{TARGET_WINDOW_MINUTES}min")
        return self.fetch_range(origin_start=stamp, origin_end=stamp, lab_end=stamp)

    def fetch_training_sources(
        self, *, data_end_origin: Any, history_days: int = 67
    ) -> SiliconForecastSources:
        end = _aware_plant(data_end_origin).floor(f"{TARGET_WINDOW_MINUTES}min")
        start = end - pd.Timedelta(days=int(history_days))
        return self.fetch_range(
            origin_start=start,
            origin_end=end,
            lab_end=end + pd.Timedelta(
                hours=3, minutes=TARGET_WINDOW_MINUTES
            ),
        )


__all__ = ["LiveSiliconForecastSource", "SiliconForecastSources"]
