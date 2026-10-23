"""Evidence store of STRATA-Learn: one row per day, append-only, de-duplicated by date.

Row fields (the same definitions as the out-of-sample test, GAS-20):
  date            calendar day (IST)
  q_pel, q_lump   blend channels: physics change of omega per class (g_c x change of the class share against the
                  14-day anchor), pellet and lump-ore groups
  q_prod          production channel: physics change of omega for the production change (TRZ residence);
                  STRATA 1.4 gives it coefficient 0 and keeps the column for offline re-tests (GAS-16, GAS-25)
  q_oper_rest     other operating channel (coke layers, fuels, oxygen), physics
  d_humidity_g    blast moisture minus its 14-day anchor, g/Nm3 (weather record, smoothed RH estimate, DATA-11)
  d_nut           nut coke minus its 14-day anchor, kg/tHM (STRATA 1.3, GAS-23)
  STRATA 1.3: the q channels come from STRATA-Flow network runs of each day (no surrogate), RI / RDI from the plant
  laboratory tests; q_lump covers CLO, ROM (with the Donimalai oversize return), Gomti and the Lloyds fines oversize.
  dobs            calibrated omega of the day minus its 14-day anchor (the target)
  K_burnt, K_held burnt coke (oxygen balance) and the Fixed-omega prediction, kg/tHM dry
  dKdw            dK/domega of the day, kg/tHM per unit omega
  sh_*            class shares of ferrous, wet %; family from blend_family()
  trial           trial code or empty
The seed (``assets/burden_engine/strata_evidence_seed.csv``) holds 191 days, 15 Dec 2025 - 7 Oct 2026 (STRATA 1.3
channels, s13/final.py)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[3]
SEED_PATH = SRC / "assets" / "burden_engine" / "strata_evidence_seed.csv"
REGRESSORS = ("q_pel", "q_lump", "q_prod", "q_oper_rest", "d_humidity_g", "d_nut")
PARAMS = ("lam_pellet", "lam_lump", "lam_production", "lam_other", "a_moisture", "a_nut")
REQUIRED = REGRESSORS + ("dobs",)
CLASSES = ("sh_sinter", "sh_pellet", "sh_clo", "sh_rom", "sh_gomti", "sh_oversize")


def blend_family(sh: Mapping[str, float]) -> str:
    """Exclusive blend family of a day (thresholds of oos/by_blend.py, wet % of ferrous)."""
    p, r = sh.get("sh_pellet", 0.0) >= 5, sh.get("sh_rom", 0.0) >= 3
    o, g = sh.get("sh_oversize", 0.0) >= 3, sh.get("sh_gomti", 0.0) >= 1
    if not p:
        if g:
            return "N2 No pellet, with Gomti"
        if r and o:
            return "N5 No pellet, ROM + oversize"
        if r:
            return "N4 No pellet, with ROM"
        if o:
            return "N3 No pellet, with oversize"
        return "N1 No pellet, sinter + CLO"
    if r and o:
        return "P4 Pellet, ROM + oversize"
    if r:
        return "P3 Pellet, with ROM"
    if o:
        return "P2 Pellet, with oversize"
    return "P1 Pellet, sinter + CLO"


class EvidenceStore:
    """CSV-backed store. ``path=None`` keeps the rows in memory (tests, replays)."""

    def __init__(self, path: str | Path | None = None, seed: bool = True):
        self.path = Path(path) if path else None
        if self.path is not None and self.path.exists():
            df = pd.read_csv(self.path, parse_dates=["date"])
        elif seed and SEED_PATH.exists():
            df = pd.read_csv(SEED_PATH, parse_dates=["date"])
        else:
            df = pd.DataFrame(columns=["date", *REQUIRED])
        self.df = self._clean(df)

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if "trial" not in df:
            df["trial"] = ""
        df["trial"] = df["trial"].fillna("").astype(str)
        if all(c in df for c in CLASSES):
            df["family"] = [blend_family(r) for r in df[list(CLASSES)].to_dict("records")]
        return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

    def append(self, rows: pd.DataFrame | list[dict[str, Any]]) -> int:
        """Add or replace days; returns the number of new dates. Rows missing a required field are refused."""
        new = pd.DataFrame(rows)
        miss = [c for c in REQUIRED if c not in new]
        if miss:
            raise ValueError(f"evidence rows lack {miss}")
        new = new.dropna(subset=list(REQUIRED))
        before = set(self.df["date"]) if len(self.df) else set()
        self.df = self._clean(pd.concat([self.df, new], ignore_index=True))
        return len(set(self.df["date"]) - before)

    def save(self) -> None:
        if self.path is None:
            raise ValueError("in-memory store")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        self.df.to_csv(tmp, index=False, float_format="%.6g")
        tmp.replace(self.path)

    def matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(dates, X regressors, y target) in date order."""
        d = self.df.dropna(subset=list(REQUIRED))
        return d["date"].values, d[list(REGRESSORS)].to_numpy(float), d["dobs"].to_numpy(float)

    def support(self, lo_q: float = 0.01, hi_q: float = 0.99) -> dict[str, tuple[float, float]]:
        """1st-99th percentile of each class share over the evidence (the GAS-18 support envelope)."""
        out = {}
        for c in CLASSES:
            if c in self.df:
                v = self.df[c].dropna()
                out[c[3:]] = (round(float(v.quantile(lo_q)), 1), round(float(v.quantile(hi_q)), 1))
        if "prod_th" in self.df:
            v = self.df["prod_th"].dropna()
            out["production_t_h"] = (round(float(v.quantile(lo_q)), 1), round(float(v.quantile(hi_q)), 1))
        return out
