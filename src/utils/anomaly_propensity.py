from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ChannelingConfig:
    """
    Configuration for Channeling detector.

    Notes:
    - window/step: set to "10min" for your requirement (1 hour => 6 values).
    - spread_mode:
        "max-min"            => spread = max(avg_quadrants) - min(avg_quadrants)
        "second_max-min"     => spread = 2nd-highest(avg_quadrants) - min(avg_quadrants)
          (useful if you want to mimic your example where the highest value looked ignored)
    """
    window: str = "10min"
    step: str = "10min"
    spread_mode: str = "max-min"  # "max-min" or "second_max-min"

    # Threshold / normalization params (edit to match plant tuning)
    max_spread_uptake_temp: float = 30.0      # degC
    max_spread_uptake_pressure: float = 0.2   # bar
    max_spread_skin: float = 50.0             # degC

    max_drop_hbp: float = -0.5                # bar over window (negative)
    max_drop_eta_co: float = -1.0             # %-points over window (negative)

    max_rise_heatload: float = 5            # GJ/hr rise over window
    max_rise_stave_dt: float = 5              # <-- TODO: set to your real parameter

    # --- Column mapping (exact names after stripping whitespace) ---
    uptake_pressure_cols: List[str] = field(default_factory=lambda: [
        "Process Params - BF2_PROC Top Pressure 1",
        "Process Params - BF2_PROC Top Pressure 2",
        "Process Params - BF2_PROC Top Pressure 3",
        "Process Params - BF2_PROC Top Pressure 4",
    ])
    uptake_temp_cols: List[str] = field(default_factory=lambda: [
        "Process Params - BF2_PROC Top Temp 1",
        "Process Params - BF2_PROC Top Temp 2",
        "Process Params - BF2_PROC Top Temp 3",
        "Process Params - BF2_PROC Top Temp 4",
    ])

    # 4 quadrants A-D at two levels
    skin_15162_cols: List[str] = field(default_factory=lambda: [
        "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp A",
        "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp B",
        "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp C",
        "Temperature Profile - BF2_BFBD Furnace Body 15162mm Temp D",
    ])
    skin_18660_cols: List[str] = field(default_factory=lambda: [
        "Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp A",
        "Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp B",
        "Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp C",
        "Temperature Profile - BF2_BFBD Furnace Body 18660mm Temp D",
    ])

    hbp_col: str = "Process Params - BF2_PROC Hot Blast Pressure"
    eta_co_col: str = "Process Params - BF2_BODY_ETACO"

    heatload_cols: List[str] = field(default_factory=lambda: [
        "Heatload Delta T - Heat load Row6-10 Q1(Stave 1-8)",
        "Heatload Delta T - Heat load Row6-10 Q2(Stave 9-16)",
        "Heatload Delta T - Heat load Row6-10 Q3(Stave 17-24)",
        "Heatload Delta T - Heat load Row6-10 Q4(Stave 25-32)",
    ])

    # Q2 missing in your data (kept out intentionally)
    stave_dt_cols: List[str] = field(default_factory=lambda: [
        "Delta T - DELTA T avg Row6-10 Q1(Stave 1-8)",
        "Delta T - DELTA T avg Row6-10 Q3(Stave 17-24)",
        "Delta T - DELTA T avg Row6-10 Q4(Stave 25-32)",
    ])


class Channeling:
    """
    Channeling detector:
      - scores each 10-min window
      - returns time-series (window_end indexed)
    """

    def __init__(self, cfg: Optional[ChannelingConfig] = None):
        self.cfg = cfg or ChannelingConfig()
        self._normalize_config_strings()

    def _normalize_config_strings(self) -> None:
        # strip config column names so tabs/spaces in source don’t break matching
        self.cfg.uptake_pressure_cols = [c.strip() for c in self.cfg.uptake_pressure_cols]
        self.cfg.uptake_temp_cols = [c.strip() for c in self.cfg.uptake_temp_cols]
        self.cfg.skin_15162_cols = [c.strip() for c in self.cfg.skin_15162_cols]
        self.cfg.skin_18660_cols = [c.strip() for c in self.cfg.skin_18660_cols]
        self.cfg.heatload_cols = [c.strip() for c in self.cfg.heatload_cols]
        self.cfg.stave_dt_cols = [c.strip() for c in self.cfg.stave_dt_cols]
        self.cfg.hbp_col = self.cfg.hbp_col.strip()
        self.cfg.eta_co_col = self.cfg.eta_co_col.strip()

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
        if time_col is not None:
            out = df.copy()
            out[time_col] = pd.to_datetime(out[time_col])
            out = out.set_index(time_col)
            return out
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex or provide time_col='your_time_column'")
        return df

    @staticmethod
    def _strip_column_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def _spread(self, vals: np.ndarray) -> float:
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size < 2:
            return np.nan

        if self.cfg.spread_mode == "max-min":
            return float(np.max(vals) - np.min(vals))

        if self.cfg.spread_mode == "second_max-min":
            # 2nd-highest minus min (robust to one extreme high)
            v = np.sort(vals)
            return float(v[-2] - v[0]) if v.size >= 2 else np.nan

        raise ValueError(f"Unknown spread_mode: {self.cfg.spread_mode}")

    @staticmethod
    def _first_last(series: pd.Series) -> Tuple[float, float]:
        s = series.dropna()
        if s.empty:
            return np.nan, np.nan
        return float(s.iloc[0]), float(s.iloc[-1])

    # ---------- (a) Uptake temp/pressure ----------
    def _diagnosis_uptake(self, g: pd.DataFrame) -> Dict[str, Any]:
        temp_cols = [c for c in self.cfg.uptake_temp_cols if c in g.columns]
        pres_cols = [c for c in self.cfg.uptake_pressure_cols if c in g.columns]

        temp_means = g[temp_cols].mean(axis=0, skipna=True).to_numpy() if temp_cols else np.array([])
        pres_means = g[pres_cols].mean(axis=0, skipna=True).to_numpy() if pres_cols else np.array([])

        spread_temp = self._spread(temp_means) if temp_cols else np.nan
        spread_pres = self._spread(pres_means) if pres_cols else np.nan

        diag_temp = spread_temp / self.cfg.max_spread_uptake_temp if np.isfinite(spread_temp) else np.nan
        diag_pres = spread_pres / self.cfg.max_spread_uptake_pressure if np.isfinite(spread_pres) else np.nan

        # Treat uptake as one “block” score (mean of temp/pressure if both exist)
        uptake_score = float(np.nanmean([diag_temp, diag_pres])) if np.isfinite(np.nanmean([diag_temp, diag_pres])) else np.nan

        return {
            "uptake_score": uptake_score,
            "uptake_temp_spread": spread_temp,
            "uptake_pressure_spread": spread_pres,
            "uptake_temp_diag": diag_temp,
            "uptake_pressure_diag": diag_pres,
        }

    # ---------- (b) Skin temp spread (combine 15162 + 18660) ----------
    def _diagnosis_skin(self, g: pd.DataFrame) -> Dict[str, Any]:
        cols_15162 = [c for c in self.cfg.skin_15162_cols if c in g.columns]
        cols_18660 = [c for c in self.cfg.skin_18660_cols if c in g.columns]

        means_15162 = g[cols_15162].mean(axis=0, skipna=True) if cols_15162 else pd.Series(dtype=float)
        means_18660 = g[cols_18660].mean(axis=0, skipna=True) if cols_18660 else pd.Series(dtype=float)

        # Quadrants A-D are positional in the lists; combine levels by averaging per quadrant
        quadrant_means: List[float] = []
        for i in range(4):
            vals = []
            if len(means_15162) == 4:
                vals.append(float(means_15162.iloc[i]))
            if len(means_18660) == 4:
                vals.append(float(means_18660.iloc[i]))

            quadrant_means.append(float(np.nanmean(vals)) if len(vals) else np.nan)

        spread_skin = self._spread(np.array(quadrant_means))
        skin_score = spread_skin / self.cfg.max_spread_skin if np.isfinite(spread_skin) else np.nan

        return {
            "skin_score": skin_score,
            "skin_spread": spread_skin,
        }

    # ---------- (c) / (d) Drop-based diagnoses (HBP, ETA CO) ----------
    def _diagnosis_drop(self, g: pd.DataFrame, col: str, max_drop_param: float) -> Dict[str, Any]:
        if col not in g.columns:
            return {"diag": np.nan, "delta": np.nan}

        first, last = self._first_last(g[col])
        if not np.isfinite(first) or not np.isfinite(last):
            return {"diag": np.nan, "delta": np.nan}

        delta = last - first  # negative means “drop”
        if delta >= 0:
            diag = 0.0
        else:
            diag = (delta / max_drop_param) if max_drop_param != 0 else np.nan  # (-)/(-) => positive

        return {"diag": float(diag), "delta": float(delta)}

    # ---------- (e) / (f) Sum of normalized positive rises (heatload, staves) ----------
    def _diagnosis_rise_sum(self, g: pd.DataFrame, cols: List[str], max_rise_param: float) -> Dict[str, Any]:
        present_cols = [c for c in cols if c in g.columns]
        if not present_cols:
            return {"diag_sum": np.nan}

        diag_sum = 0.0
        any_valid = False

        for c in present_cols:
            first, last = self._first_last(g[c])
            if not np.isfinite(first) or not np.isfinite(last):
                continue

            delta = last - first
            # only take rises; drops are ignored (as per your example)
            if delta > 0 and max_rise_param != 0:
                diag_sum += float(delta / max_rise_param)
            any_valid = True

        return {"diag_sum": float(diag_sum) if any_valid else np.nan}

    def _score_window(self, g: pd.DataFrame, window_end: pd.Timestamp) -> Dict[str, Any]:
        out: Dict[str, Any] = {"window_end": window_end}

        # (a)
        out.update(self._diagnosis_uptake(g))

        # (b)
        out.update(self._diagnosis_skin(g))

        # (c)
        hbp = self._diagnosis_drop(g, self.cfg.hbp_col, self.cfg.max_drop_hbp)
        out["hbp_score"] = hbp["diag"]
        out["hbp_delta"] = hbp["delta"]

        # (d)
        eta = self._diagnosis_drop(g, self.cfg.eta_co_col, self.cfg.max_drop_eta_co)
        out["eta_co_score"] = eta["diag"]
        out["eta_co_delta"] = eta["delta"]

        # (e)
        heat = self._diagnosis_rise_sum(g, self.cfg.heatload_cols, self.cfg.max_rise_heatload)
        out["heatload_score"] = heat["diag_sum"]

        # (f)
        stave = self._diagnosis_rise_sum(g, self.cfg.stave_dt_cols, self.cfg.max_rise_stave_dt)
        out["stave_score"] = stave["diag_sum"]

        # Final: equal weight across the 6 measurement blocks (a–f), adjusted for availability
        components = [
            out.get("uptake_score", np.nan),
            out.get("skin_score", np.nan),
            out.get("hbp_score", np.nan),
            out.get("eta_co_score", np.nan),
            out.get("heatload_score", np.nan),
            out.get("stave_score", np.nan),
        ]
        n_used = int(np.sum([np.isfinite(x) for x in components]))
        out["n_components_used"] = n_used
        out["channeling_score"] = float(np.nanmean(components)) if n_used > 0 else np.nan

        return out

    def score_timeseries(
        self,
        df: pd.DataFrame,
        *,
        time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute channeling score every `step`, using a lookback of `window`.

        Important behavior (matches your expectation):
        - If you give ~1 hour data, you’ll get 6 rows (10-min steps).
        - The last window can be “partial” if the data ends before the theoretical window end.
        """
        df = self._ensure_datetime_index(df, time_col=time_col)
        df = self._strip_column_whitespace(df)
        df = df.sort_index()

        win = pd.Timedelta(self.cfg.window)
        step = pd.Timedelta(self.cfg.step)

        start = df.index.min()
        end = df.index.max()

        rows: List[Dict[str, Any]] = []

        t = start
        while t <= end:
            g = df[(df.index >= t) & (df.index < t + win)]
            rows.append(self._score_window(g, window_end=t + win))
            t = t + step

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        return out.set_index("window_end")
