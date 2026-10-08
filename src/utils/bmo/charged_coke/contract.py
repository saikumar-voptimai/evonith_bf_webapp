"""Timestamp contract of the live furnace dataset, as measured.

A row of ``furnace_dataset.csv`` labelled T (plant clock, IST, naive) is not one
interval. Measured on 5-6 October 2026 against the raw sources:

* Charged masses (``COKE_CALC_MT``, ``NUTCOKE_CALC_MT``) sum the individual
  charge records with event times in ``[T-60 min, T)``: shifting the charge
  records by +60 min reproduces the column exactly (median difference 0 t).
* Online tags (blast volume, PCI rate, blast temperature, production ...)
  average InfluxDB's 15-minute points over ``[T-30 min, T+30 min)``: that
  alignment matches to within 8 Nm3/h and 0.07 kg/THM, every other alignment
  is out by 200-800 Nm3/h. (The hourly query buckets on UTC hours, which run
  from :30 to :30 in IST, and the cleaner floors the label to the hour.)
* The plant clock is IST: matching in UTC triples the error.
* The file is rebuilt hourly, about six minutes past the hour, and its newest
  row is still filling: its online half runs 30 minutes past its label.
  Values of that row change in the next build.

The research model was trained on rows with exactly this construction, so it
is internally consistent. What changes is how its forecast is described: for
an issue row t, the target block (rows t+2..t+5) is coke charged in
``[t+1 h, t+5 h)`` against production over ``[t+1.5 h, t+5.5 h)``. Row t is
complete at t+30 min and first appears complete in the build after that, so
the forecast is issued at about t+1 h 06 min: its block starts at about the
issue time and spans roughly the next four hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

PLANT_TZ = "Asia/Kolkata"
CHARGE_COVERAGE = (pd.Timedelta(minutes=-60), pd.Timedelta(0))
ONLINE_COVERAGE = (pd.Timedelta(minutes=-30), pd.Timedelta(minutes=30))
TARGET_ROWS = (2, 5)


def _naive_ist(value: datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(PLANT_TZ).tz_localize(None)
    return stamp


def row_complete_at(row: pd.Timestamp) -> pd.Timestamp:
    """When every input of row ``row`` has been measured (its online half)."""

    return pd.Timestamp(row) + ONLINE_COVERAGE[1]


def latest_complete_row(built_at: datetime | pd.Timestamp, last_row: pd.Timestamp) -> pd.Timestamp | None:
    """Newest row whose inputs had all been measured when the file was built.

    Args:
        built_at: When the dataset file was built (its Last-Modified time);
            naive values are read as IST.
        last_row: The newest row label in the file.

    Returns:
        The latest complete row, or None if even the oldest candidate is not.
    """

    built = _naive_ist(built_at)
    candidate = (built - ONLINE_COVERAGE[1]).floor("h")
    candidate = min(candidate, pd.Timestamp(last_row))
    return candidate if row_complete_at(candidate) <= built else None


@dataclass(frozen=True)
class ForecastWindow:
    """The three times every forecast carries, kept apart (all IST, naive).

    Attributes:
        data_row: Label of the latest row the forecast used.
        data_complete_at: When that row's inputs were all measured.
        issued_at: Wall-clock time the forecast was produced.
        coke_start / coke_end: Charges the forecast block sums.
        production_start / production_end: Production it divides by.
    """

    data_row: pd.Timestamp
    data_complete_at: pd.Timestamp
    issued_at: pd.Timestamp
    coke_start: pd.Timestamp
    coke_end: pd.Timestamp
    production_start: pd.Timestamp
    production_end: pd.Timestamp

    @classmethod
    def for_row(cls, data_row: pd.Timestamp, issued_at: datetime | pd.Timestamp) -> "ForecastWindow":
        row = pd.Timestamp(data_row)
        first, last = (row + pd.Timedelta(hours=k) for k in TARGET_ROWS)
        return cls(
            data_row=row,
            data_complete_at=row_complete_at(row),
            issued_at=_naive_ist(issued_at),
            coke_start=first + CHARGE_COVERAGE[0],
            coke_end=last + CHARGE_COVERAGE[1],
            production_start=first + ONLINE_COVERAGE[0],
            production_end=last + ONLINE_COVERAGE[1],
        )

    def describe(self) -> str:
        """'Coke charged 04:00-08:00 (production 04:30-08:30) IST'."""

        def hm(t: pd.Timestamp) -> str:
            return f"{t:%H:%M}"

        return (
            f"Coke charged {hm(self.coke_start)}–{hm(self.coke_end)} "
            f"(production {hm(self.production_start)}–{hm(self.production_end)}) IST"
        )


def horizon_window(data_row: pd.Timestamp, horizon: int) -> dict[str, pd.Timestamp]:
    """Clock span of the 4-h rate as it reads ``horizon`` rows after ``data_row``.

    The rate at row t+h sums rows t+h-3..t+h: charges in [t+h-4 h, t+h) against
    production over [t+h-3.5 h, t+h+0.5 h). It is plotted at ``coke_end``.
    """

    last = pd.Timestamp(data_row) + pd.Timedelta(hours=int(horizon))
    first = last - pd.Timedelta(hours=3)
    return {
        "row": last,
        "coke_start": first + CHARGE_COVERAGE[0],
        "coke_end": last + CHARGE_COVERAGE[1],
        "production_start": first + ONLINE_COVERAGE[0],
        "production_end": last + ONLINE_COVERAGE[1],
    }


__all__ = [
    "horizon_window",
    "CHARGE_COVERAGE",
    "ONLINE_COVERAGE",
    "PLANT_TZ",
    "ForecastWindow",
    "latest_complete_row",
    "row_complete_at",
]
