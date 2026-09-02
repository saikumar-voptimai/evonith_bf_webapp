"""Historical predicted-vs-realised hot-metal silicon, from the shipped model.

WHY THIS IS HARDER THAN IT LOOKS.

The Si bundle expects 194 features and the static ML dataset carries only 112 of
them directly. The other 82 are derived - ore shares, per-tonne conversions,
ratios, calendar terms and lags - and were built inside the training notebook,
not saved alongside the model.

So they have to be reconstructed here, and a reconstruction that is subtly wrong
produces a chart that looks entirely reasonable and is not. That is the whole
risk of this module.

HOW IT GUARDS AGAINST ITSELF.

``reconstruct_si_features`` reports exactly how many features it derived, how
many it had to fill with a column median, and which. If the fill count is high
the reconstruction is not faithful and the caller is expected to say so rather
than draw the chart.

The strongest check is the output: scoring a correctly rebuilt feature set
against the silicon actually measured should land near the model's training
performance. If it does not, the rebuild is wrong - and that is a statement
about this file, not about the model.

ONE THING WORTH KNOWING ABOUT THE MODEL. Among its inputs are LAGGED VALUES OF
SILICON ITSELF. Predicting today's silicon partly from yesterday's is legitimate
- an operator does know the last cast - but it means a good score here is not
evidence that the burden terms are doing the work.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "assets" / "models"
SI_TARGET = "CHEM_PCT_SI"
# The Si bundle names lags "<base>__lag2h" - double underscore, hours suffix -
# where the BMO fuel bundle uses "<base>_lag2_(GasImpact)". Two models, two
# conventions; matching only the fuel one silently left 39 features unbuilt and
# filled with medians.
_LAG_PATTERN = re.compile(r"^(?P<base>.+?)_+lag(?P<steps>\d+)h?(?:_\(.*\))?$")


@dataclass
class SiHistory:
    """Predicted and realised silicon, with an account of what was rebuilt."""

    frame: pd.DataFrame
    derived: int = 0
    filled: int = 0
    filled_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_trustworthy(self) -> bool:
        """Too many medians and the chart is drawing the fill, not the model."""

        total = self.derived + self.filled
        return bool(total) and self.filled / total < 0.15


def _add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the notebook's engineered columns from the raw ones."""

    out = frame.copy()
    prod = pd.to_numeric(out.get("PRODUCTIONTONNESPERHR"), errors="coerce")
    prod = prod.replace(0, np.nan)
    prod = prod.fillna(prod.median())

    # Every *_CALC_MT becomes a per-tonne-hot-metal figure.
    for col in [c for c in out.columns if c.endswith("_CALC_MT")]:
        out[col.replace("_CALC_MT", "_CALC_THM")] = (
            pd.to_numeric(out[col], errors="coerce") / prod
        )

    # Individual ore slots as a share of the total ore charged.
    ore_cols = sorted(
        [c for c in out.columns if re.match(r"^ORE_\d+_CALC_MT$", c)],
        key=lambda x: int(x.split("_")[1]),
    )
    if ore_cols:
        ore = out[ore_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        total_ore = ore.sum(axis=1)
        out["TOTAL_ORE_MT"] = total_ore
        out["TOTAL_ORE_THM"] = total_ore / prod
        safe = total_ore.replace(0, total_ore.median())
        for col in ore_cols:
            slot = col.split("_")[1]
            out[f"ORE_{slot}_PCT"] = ore[col] / safe * 100.0

    def num(name: str) -> pd.Series:
        # Always a Series. pd.to_numeric(None) returns a bare float, which then
        # fails on any Series method downstream.
        if name not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype=float)
        return pd.to_numeric(out[name], errors="coerce")

    clo = num("TOTAL_CLO_THM")
    if clo.isna().all():
        clo = num("TOTAL_ORE_THM")
        out["TOTAL_CLO_THM"] = clo
    sinter = num("SINTER_CALC_THM")
    pellet = num("TOTAL_PELLET_CALC_THM")
    ore_thm = num("TOTAL_ORE_THM")

    safe_clo = clo.replace(0, np.nan)
    out["SINTER_CLO_RATIO"] = sinter / safe_clo
    out["PELLET_CLO_RATIO"] = pellet / safe_clo
    out["ORE_SINTER_RATIO"] = ore_thm / sinter.replace(0, np.nan)

    # Calendar terms, exactly as the notebook built them.
    idx = pd.DatetimeIndex(out.index)
    out["month"] = idx.month
    out["week_of_year"] = idx.isocalendar().week.astype(int).to_numpy()
    out["day_of_year"] = idx.dayofyear
    out["trend_index"] = np.arange(len(out))
    return out


def _add_lags(frame: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    """Build every requested lag feature from its base column."""

    out = frame.copy()
    for name in wanted:
        match = _LAG_PATTERN.match(name)
        if not match:
            continue
        base = match.group("base").rstrip("_")
        steps = int(match.group("steps"))
        if base in out.columns:
            out[name] = pd.to_numeric(out[base], errors="coerce").shift(steps)
    return out


def reconstruct_si_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, SiHistory]:
    """Build the model's expected feature matrix from a static-dataset slice."""

    columns_path = MODEL_DIR / "hm_si_feature_columns.json"
    if not columns_path.exists():
        return pd.DataFrame(), SiHistory(pd.DataFrame(),
                                         notes=["Si feature list not found"])
    expected = json.loads(columns_path.read_text(encoding="utf-8"))

    work = _add_derived(frame)
    work = _add_lags(work, [c for c in expected if "_lag" in c])

    filled: list[str] = []
    columns: dict[str, pd.Series] = {}
    for name in expected:
        if name in work.columns:
            columns[name] = pd.to_numeric(work[name], errors="coerce")
        else:
            columns[name] = pd.Series(np.nan, index=work.index, dtype=float)
            filled.append(name)
    derived = len(expected) - len(filled)
    matrix = pd.DataFrame(columns, index=work.index)[expected]
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)

    report = SiHistory(pd.DataFrame(), derived=derived, filled=len(filled),
                       filled_names=filled[:20])
    if filled:
        report.notes.append(
            f"{len(filled)} of {len(expected)} features could not be rebuilt "
            "and were filled with column medians"
        )
    return matrix, report


def build_si_history(days: int = 120) -> SiHistory:
    """Score the shipped Si model over recent history against measured silicon.

    Args:
         - days: int - Lookback in days.

    Returns:
         - return SiHistory - Frame with predicted_si and actual_si by day, plus
           a record of how much of the feature set had to be filled.
    """

    try:
        import joblib
        import xgboost as xgb
    except ImportError as exc:  # noqa: BLE001
        return SiHistory(pd.DataFrame(), notes=[f"Si model unavailable: {exc}"])

    from data.ml.static_csv import load_static_dataset

    raw = load_static_dataset()
    if raw is None or raw.empty:
        return SiHistory(pd.DataFrame(), notes=["static ML dataset unavailable"])
    if not isinstance(raw.index, pd.DatetimeIndex):
        return SiHistory(pd.DataFrame(), notes=["static dataset has no time index"])

    cutoff = raw.index.max() - pd.Timedelta(days=days)
    window = raw[raw.index >= cutoff].copy()
    if window.empty or SI_TARGET not in window.columns:
        return SiHistory(pd.DataFrame(), notes=["no silicon measurements in window"])

    matrix, report = reconstruct_si_features(window)
    if matrix.empty:
        return report

    try:
        scaler = joblib.load(MODEL_DIR / "hm_si_scaler.joblib")
        booster = xgb.Booster()
        booster.load_model(str(MODEL_DIR / "hm_si_model.json"))
        scaled = scaler.transform(matrix)
        predicted = booster.predict(
            xgb.DMatrix(scaled, feature_names=list(matrix.columns))
        )
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"scoring failed: {str(exc)[:120]}")
        return report

    hourly = pd.DataFrame({
        "predicted_si": predicted,
        "actual_si": pd.to_numeric(window[SI_TARGET], errors="coerce").to_numpy(),
    }, index=window.index)
    daily = hourly.resample("1D").mean(numeric_only=True).dropna()
    daily.index = pd.to_datetime(daily.index.date)
    daily["residual"] = daily["predicted_si"] - daily["actual_si"]

    report.frame = daily
    if len(daily) > 10:
        err = daily["residual"]
        ss_tot = ((daily["actual_si"] - daily["actual_si"].mean()) ** 2).sum()
        r2 = 1.0 - (err ** 2).sum() / ss_tot if ss_tot else float("nan")
        report.notes.append(
            f"scored {len(daily)} days: bias {err.mean():+.3f}%, "
            f"MAE {err.abs().mean():.3f}%, R2 {r2:+.2f}"
        )
    return report
