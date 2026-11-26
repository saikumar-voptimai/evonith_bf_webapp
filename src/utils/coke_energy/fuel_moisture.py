"""Total moisture of the coke and nut coke charged, per hour: the dry basis of the charge record.

The charge record weighs coke and nut coke wet; the plant's coke-rate, nut-coke-rate and fuel-rate tags are dry. The
laboratory reports two moistures per fuel sample (``offline_feed.fuel_chemistry``):

    tm        total moisture of the coke as charged          coke 2-6 %, nut coke 5-10 %
    moisture  inherent moisture of the air-dried sample      about 0.45 %

Up to 6 Dec 2025 the hourly dataset's ``COKE_MOIST%`` / ``NUTCOKE_MOIST%`` held the total moisture; since the offline
feed moved to the new tables they hold the inherent moisture, so ``dry = wet x (1 - COKE_MOIST%)`` left the coke
1.5-3.5 % and the nut coke 5-8 % too heavy. The columns are features of the data-driven models and keep their
values; the balance takes the total moisture from the laboratory instead:

    hourly_total_moisture   each hour's tonnes per coke type x the latest sample of that type taken before the hour
                            ends, weighted by the tonnes (a type without a sample takes the others' value)
    with_total_moisture     adds the result to the hourly dataset as COKE_TM% and NUTCOKE_TM%
    on_total_moisture       the pipeline's statement: COKE_MOIST% / NUTCOKE_MOIST% become the total moisture where
                            it is known (``build_shift_inputs``), so every later step reads one moisture

The plant's coal rate (PCI) is already dry: nothing is deducted from it (``build_shift_inputs``). Of the charge, coke,
nut coke, the ores and pellet are wet (total moisture deducted); sinter is dry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

# dataset column of the total moisture -> the charge record's material codes of that fuel
FAMILIES: Mapping[str, Sequence[str]] = {
    "COKE_TM%": ("coke_1", "coke_2"),
    "NUTCOKE_TM%": ("nut_coke_1", "nut_coke_2"),
}
# total-moisture column -> the column the balance reads
BALANCE_COLUMN: Mapping[str, str] = {"COKE_TM%": "COKE_MOIST%", "NUTCOKE_TM%": "NUTCOKE_MOIST%"}
VALID_PCT = (0.0, 20.0)   # a total moisture outside is not a moisture result
MAX_GAP_HOURS = 24        # an hour without a charge of the fuel takes the value of the last hour that had one


def hourly_total_moisture(charged: pd.DataFrame, samples: pd.DataFrame,
                          valid: tuple[float, float] = VALID_PCT) -> pd.DataFrame:
    """Total moisture (%) of the coke and of the nut coke charged in each hour.

    ``charged``: tonnes per IST hour (index = hour start), one column per material code. ``samples``: one row per
    laboratory sample with ``material``, ``t`` (IST sampling time) and ``tm`` (%). Returns ``COKE_TM%`` and
    ``NUTCOKE_TM%`` on ``charged``'s hours; NaN where the fuel was not charged or no sample precedes the hour's end.
    """

    hours = pd.DatetimeIndex(charged.index).sort_values()
    out = pd.DataFrame(index=hours)
    ends = pd.DataFrame({"end": hours + pd.Timedelta(hours=1)})
    lo, hi = valid
    ok = samples[samples["tm"].between(lo, hi)] if len(samples) else samples
    for column, codes in FAMILIES.items():
        water = np.zeros(len(hours))
        mass = np.zeros(len(hours))
        for code in codes:
            if code not in charged.columns or not len(ok):
                continue
            s = ok.loc[ok["material"] == code, ["t", "tm"]].sort_values("t")
            if s.empty:
                continue
            tm = pd.merge_asof(ends, s, left_on="end", right_on="t", direction="backward",
                               allow_exact_matches=False)["tm"].to_numpy(dtype=float)
            q = pd.to_numeric(charged[code], errors="coerce").reindex(hours).fillna(0.0).clip(lower=0.0).to_numpy()
            known = np.isfinite(tm)
            water += np.where(known, q * np.nan_to_num(tm), 0.0)
            mass += np.where(known, q, 0.0)
        out[column] = np.where(mass > 0, water / np.where(mass > 0, mass, 1.0), np.nan)
    return out


def with_total_moisture(fd: pd.DataFrame, moisture: pd.DataFrame | None,
                        max_gap_hours: int = MAX_GAP_HOURS) -> pd.DataFrame:
    """The hourly dataset (``time`` column, IST) with ``COKE_TM%`` / ``NUTCOKE_TM%`` from ``moisture`` (indexed by
    hour). Without a moisture frame the dataset is returned as it is: the balance then reads the dataset's own
    columns."""

    if moisture is None or moisture.empty or fd.empty:
        return fd
    columns = [c for c in FAMILIES if c in moisture.columns]
    if not columns:
        return fd
    times = pd.DatetimeIndex(fd["time"])
    grid = pd.date_range(min(moisture.index.min(), times.min()), max(moisture.index.max(), times.max()), freq="h")
    filled = moisture[columns].reindex(grid).ffill(limit=max_gap_hours).reindex(times)
    out = fd.copy()
    for c in columns:
        out[c] = filled[c].to_numpy()
    return out


def on_total_moisture(f: pd.DataFrame) -> pd.DataFrame:
    """``COKE_MOIST%`` / ``NUTCOKE_MOIST%`` of the working frame set to the total moisture where it is known (in
    place; returns the frame)."""

    for tm, column in BALANCE_COLUMN.items():
        if tm in f.columns:
            known = f[tm].notna()
            f[column] = f[tm].where(known, f[column]) if column in f.columns else f[tm]
    return f
