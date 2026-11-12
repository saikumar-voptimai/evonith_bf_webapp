"""Charged coke per tonne of hot metal from the per-charge record (user decision 12 Nov 2026).

``offline_feed.charge_data`` holds one row per charge (one about every 9.4 min) and is published once an hour
(HH:59 to HH:01; median age of a charge at publication 33 min). The coke-to-ferrous ratio of a few charges needs
neither the hourly charge counter nor the hourly production figure, which alternate from hour to hour, so it follows
the charging within the hour of publication:

    K_charged,now = f x 1000 x sum(coke) / sum(ferrous)      over the last n charges (wet tonnes)

``f`` puts the ratio on the pipeline's basis (dry coke per tonne of hot metal): the median, over the last stored 8-hour
twin windows, of the window's K_charged over the same ratio of the window's own charges. It carries the coke moisture
and the hot metal per tonne of ferrous burden, and follows a blend change within three days. 16 Oct-12 Nov 2026:
f 1.697 (weekly medians 1.689-1.716); K_charged,now of 6 charges less the 8-hour K_charged: mean +0.3, SD 8.1 kg/tHM;
hour-to-hour change 3.6 kg/tHM (robust SD) against 4.4 for the 8-hour figure.

Pure numerics on a charge table indexed by charge time with ``coke_t`` and ``ferrous_t``: no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

GROUPS = {
    "coke_t": ("coke_",),
    "nut_coke_t": ("nut_coke_",),
    "ferrous_t": ("sinter_", "ore_", "pellet_"),
    "flux_t": ("flux_",),
}


@dataclass(frozen=True)
class Settings:
    charges: int = 6  # charges of the ratio (about one hour)
    window_h: float = 8.0  # the twin window
    offset_h: float = 1.0  # the pipeline counts the charges of [end - 9 h, end - 1 h) in the window ending at ``end``
    min_window_charges: int = 40  # charges a window needs to enter the factor (an 8-hour window holds about 51)
    max_windows: int = 72  # newest windows of the factor's median
    min_windows: int = 12  # fewer: the default factor
    default_factor: float = 1.70  # 16 Oct-12 Nov 2026 median 1.697

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> Settings:
        c, d = dict(cfg or {}), cls()
        return cls(
            charges=int(c.get("charges", d.charges)),
            window_h=float(c.get("window_h", d.window_h)),
            offset_h=float(c.get("offset_h", d.offset_h)),
            min_window_charges=int(c.get("min_window_charges", d.min_window_charges)),
            max_windows=int(c.get("max_windows", d.max_windows)),
            min_windows=int(c.get("min_windows", d.min_windows)),
            default_factor=float(c.get("default_factor", d.default_factor)),
        )


def charge_table(raw: pd.DataFrame, times: Any, received: Any = None) -> pd.DataFrame:
    """Tonnes per charge by group (coke, nut coke, ferrous, flux) from the ``*_mt`` columns of ``charge_data``,
    indexed by ``times`` (charge time), with ``received`` (publication time) when given. One row per charge time."""
    out = pd.DataFrame(index=pd.DatetimeIndex(times))
    tonnes = [c for c in raw.columns if c.endswith("_mt")]
    num = raw[tonnes].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for name, prefixes in GROUPS.items():
        cols = [c for c in tonnes if c.startswith(prefixes)]
        out[name] = num[cols].sum(axis=1).to_numpy() if cols else 0.0
    out["received"] = pd.DatetimeIndex(received) if received is not None else pd.NaT
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[out["ferrous_t"] > 0]


def ratio(charges: pd.DataFrame, start: Any, end: Any) -> tuple[float | None, int]:
    """Wet coke per tonne of ferrous burden of the charges in [start, end), kg/t, and their number."""
    part = charges[(charges.index >= pd.Timestamp(start)) & (charges.index < pd.Timestamp(end))]
    ferrous = float(part["ferrous_t"].sum()) if len(part) else 0.0
    return (1000.0 * float(part["coke_t"].sum()) / ferrous if ferrous > 0 else None, len(part))


def factor(charges: pd.DataFrame, windows: Iterable[tuple[Any, float | None]], s: Settings) -> tuple[float, int]:
    """K_charged of a window over the ratio of its charges: the median over the newest windows, and their number.
    ``windows``: (window end, K_charged) on the time base of the charge table. The default without enough windows."""
    values = []
    for end, k in sorted(windows, key=lambda w: pd.Timestamp(w[0]), reverse=True):
        if k is None or not np.isfinite(k) or k <= 0:
            continue
        end = pd.Timestamp(end)
        r, n = ratio(charges, end - pd.Timedelta(hours=s.window_h + s.offset_h), end - pd.Timedelta(hours=s.offset_h))
        if r is None or n < s.min_window_charges:
            continue
        values.append(float(k) / r)
        if len(values) >= s.max_windows:
            break
    if len(values) < s.min_windows:
        return s.default_factor, len(values)
    return float(np.median(values)), len(values)


def charged_now(
    charges: pd.DataFrame, windows: Sequence[tuple[Any, float | None]], s: Settings, *, at: Any = None
) -> dict[str, Any]:
    """Charged coke of the last ``s.charges`` charges before ``at`` on the pipeline's basis, dry kg/tHM, with the
    charges' span, the publication time of the newest and the factor used."""
    part = charges if at is None else charges[charges.index <= pd.Timestamp(at)]
    last = part.tail(s.charges)
    f, n_windows = factor(charges, windows, s)
    ferrous = float(last["ferrous_t"].sum()) if len(last) else 0.0
    if len(last) < s.charges or ferrous <= 0:
        value = None
    else:
        value = f * 1000.0 * float(last["coke_t"].sum()) / ferrous
    received = last["received"].max() if len(last) and "received" in last else pd.NaT
    return {
        "value": value,
        "charges": len(last),
        "first": last.index[0] if len(last) else None,
        "last": last.index[-1] if len(last) else None,
        "received": None if pd.isna(received) else pd.Timestamp(received),
        "factor": f,
        "factor_windows": n_windows,
        "basis": "windows" if n_windows >= s.min_windows else "default",
    }
