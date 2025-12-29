from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class PropensityResult:
    score_0_100: float
    alarm: bool
    alarm_reason: str
    last_timestamp: pd.Timestamp | None
    series_5min: pd.Series
    z_5min: pd.Series
    z_abs_max_last_hour: float | None
    z_std_last_hour: float | None


@dataclass(frozen=True)
class PropensitySeriesSpec:
    """Specification for a propensity time series extracted from a dataframe."""

    name: str
    kind: str  # "temperature_spread" | "column"
    columns: Sequence[str] | None = None
    column_candidates: Sequence[str] | None = None
    z_threshold: float = 2.5
    zstd_threshold: float = 1.0


def _ensure_datetime_index_utc(obj: pd.Series) -> pd.Series:
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    s = obj.copy()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    return s


def infer_temperature_columns(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []

    # Heuristic: prefer columns that look like temperatures/thermocouples.
    candidates: list[str] = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ("temp", "temperature", "thermo", "tc")):
            if pd.api.types.is_numeric_dtype(df[c]):
                candidates.append(str(c))

    return candidates


def compute_temperature_spread(df: pd.DataFrame, temp_cols: Iterable[str] | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)

    cols = list(temp_cols) if temp_cols is not None else infer_temperature_columns(df)
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 3:
        return pd.Series(dtype=float)

    block = df[cols].apply(pd.to_numeric, errors="coerce")
    spread = block.max(axis=1) - block.min(axis=1)
    spread.name = "temperature_spread"
    return spread


def compute_first_available_numeric_series(
    df: pd.DataFrame,
    column_candidates: Sequence[str],
    *,
    name: str | None = None,
) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for col in column_candidates:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            s = pd.to_numeric(df[col], errors="coerce")
            s.name = name or str(col)
            return s
    return pd.Series(dtype=float)


def compute_propensity_from_series(
    series: pd.Series,
    *,
    sample_freq: str = "5min",
    baseline_window: str = "6h",
    alarm_window: str = "1h",
    z_threshold: float = 2.5,
    zstd_threshold: float = 1.0,
    min_points_last_hour: int = 6,
) -> PropensityResult:
    if series is None or series.empty:
        return PropensityResult(
            score_0_100=0.0,
            alarm=False,
            alarm_reason="No data",
            last_timestamp=None,
            series_5min=pd.Series(dtype=float),
            z_5min=pd.Series(dtype=float),
            z_abs_max_last_hour=None,
            z_std_last_hour=None,
        )

    s = _ensure_datetime_index_utc(series)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return PropensityResult(
            score_0_100=0.0,
            alarm=False,
            alarm_reason="No numeric data",
            last_timestamp=None,
            series_5min=pd.Series(dtype=float),
            z_5min=pd.Series(dtype=float),
            z_abs_max_last_hour=None,
            z_std_last_hour=None,
        )

    s5 = s.resample(sample_freq).mean().dropna()
    if s5.empty:
        return PropensityResult(
            score_0_100=0.0,
            alarm=False,
            alarm_reason="No data after resample",
            last_timestamp=None,
            series_5min=pd.Series(dtype=float),
            z_5min=pd.Series(dtype=float),
            z_abs_max_last_hour=None,
            z_std_last_hour=None,
        )

    sample_td = pd.Timedelta(sample_freq)
    baseline_td = pd.Timedelta(baseline_window)

    window_pts = int(baseline_td / sample_td)
    window_pts = max(window_pts, 12)  # at least 1 hour of 5-min points

    mu = s5.rolling(window_pts, min_periods=max(6, window_pts // 4)).mean()
    sig = s5.rolling(window_pts, min_periods=max(6, window_pts // 4)).std(ddof=0)
    sig = sig.replace(0.0, pd.NA)

    z = (s5 - mu) / sig
    z = z.dropna()

    last_ts = s5.index.max()
    if pd.isna(last_ts):
        return PropensityResult(
            score_0_100=0.0,
            alarm=False,
            alarm_reason="No timestamps",
            last_timestamp=None,
            series_5min=s5,
            z_5min=z,
            z_abs_max_last_hour=None,
            z_std_last_hour=None,
        )

    alarm_td = pd.Timedelta(alarm_window)
    last_hour_start = last_ts - alarm_td
    z_last = z.loc[z.index >= last_hour_start]

    if len(z_last) < min_points_last_hour:
        return PropensityResult(
            score_0_100=0.0,
            alarm=False,
            alarm_reason=f"Insufficient last-hour points ({len(z_last)}/{min_points_last_hour})",
            last_timestamp=last_ts,
            series_5min=s5,
            z_5min=z,
            z_abs_max_last_hour=float(z_last.abs().max()) if len(z_last) else None,
            z_std_last_hour=float(z_last.std(ddof=0)) if len(z_last) else None,
        )

    z_abs_max = float(z_last.abs().max())
    z_std = float(z_last.std(ddof=0))

    # Score combines magnitude + variability, clamped to [0, 100].
    mag = min(1.0, z_abs_max / max(z_threshold, 1e-9))
    var = min(1.0, z_std / max(zstd_threshold, 1e-9))
    score = 100.0 * (0.6 * mag + 0.4 * var)

    alarm = (z_abs_max >= z_threshold) and (z_std >= zstd_threshold)
    reason = (
        f"z|max={z_abs_max:.2f} (>= {z_threshold}), z_std={z_std:.2f} (>= {zstd_threshold})"
        if alarm
        else f"z|max={z_abs_max:.2f}, z_std={z_std:.2f}"
    )

    return PropensityResult(
        score_0_100=float(max(0.0, min(100.0, score))),
        alarm=bool(alarm),
        alarm_reason=reason,
        last_timestamp=last_ts,
        series_5min=s5,
        z_5min=z,
        z_abs_max_last_hour=z_abs_max,
        z_std_last_hour=z_std,
    )


def compute_channeling_propensity(df_recent: pd.DataFrame) -> PropensityResult:
    """
    Demo channeling propensity based on circumferential/axial temperature spread.
    """
    stack_temp_list: list[str] = []
    for col in df_recent.columns:
        if "18660mm" in str(col):
            stack_temp_list.append(str(col))

    spread = compute_temperature_spread(df_recent, temp_cols=stack_temp_list)
    if spread.empty:
        spread = compute_first_available_numeric_series(
            df_recent,
            ("DIFFERENTIAL PRESSURETOTALBAR", "PERMEABILITYKGS/HR.", "TopPressureBar"),
            name="channeling_fallback",
        )

    return compute_propensity_from_series(spread)


def compute_propensity_suite(
    df_recent: pd.DataFrame,
    *,
    stack_temp_cols: Sequence[str] | None = None,
    specs: Sequence[PropensitySeriesSpec] | None = None,
) -> dict[str, PropensityResult]:
    """Compute multiple demo propensities from a recent dataframe.

    Returns a dict keyed by spec name. Any spec that cannot be computed (missing
    columns / insufficient data) is omitted.
    """

    if df_recent is None or df_recent.empty:
        return {}

    if stack_temp_cols is None:
        stack_temp_cols = [str(c) for c in df_recent.columns if "18660mm" in str(c)]

    if specs is None:
        specs = (
            PropensitySeriesSpec(
                name="Channeling (stack temp spread)",
                kind="temperature_spread",
                columns=tuple(stack_temp_cols),
                z_threshold=2.5,
                zstd_threshold=1.0,
            ),
            PropensitySeriesSpec(
                name="ΔP instability",
                kind="column",
                column_candidates=(
                    "DIFFERENTIAL PRESSURETOTALBAR",
                    "DIFFERENTIAL PRESSURE",
                ),
                z_threshold=2.5,
                zstd_threshold=1.0,
            ),
            PropensitySeriesSpec(
                name="Permeability instability",
                kind="column",
                column_candidates=("PERMEABILITYKGS/HR.", "PERMEABILITY"),
                z_threshold=2.5,
                zstd_threshold=1.0,
            ),
            PropensitySeriesSpec(
                name="Top pressure instability",
                kind="column",
                column_candidates=("TopPressureBar", "TOP PRESSUREBAR", "TOP PRESSURE"),
                z_threshold=2.5,
                zstd_threshold=1.0,
            ),
        )

    out: dict[str, PropensityResult] = {}
    for spec in specs:
        if spec.kind == "temperature_spread":
            series = compute_temperature_spread(df_recent, temp_cols=spec.columns)
        elif spec.kind == "column":
            series = compute_first_available_numeric_series(
                df_recent,
                tuple(spec.column_candidates or ()),
                name=spec.name,
            )
        else:
            continue

        if series is None or series.dropna().empty:
            continue

        res = compute_propensity_from_series(
            series,
            z_threshold=spec.z_threshold,
            zstd_threshold=spec.zstd_threshold,
        )
        out[spec.name] = res

    return out
