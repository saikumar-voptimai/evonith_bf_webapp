"""Burnt against charged coke per averaging window, from a shift table (docs/burnt_vs_charged_coke.md).

    python scripts/burnt_vs_charged_coke.py SHIFTS.csv            # web export: shift, sdate, prod_th,
                                                                   # burnt_coke_kgthm, charged_coke_kgthm
    python scripts/burnt_vs_charged_coke.py SHIFTS.pkl            # the Coke Energy pipeline's shift table

Prints the window table (bias, SD, 95 % band, r, CCC), the split of the spread into what averages out and what does
not, the lag scan, and - when the table carries ``K_burnt_raw`` - the mass closure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.coke_energy import transit as tr  # noqa: E402

MID = {"A": 10, "B": 18, "C": 26}


def load(path: str) -> pd.DataFrame:
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    c = pd.read_csv(path).set_index("shift")
    T = pd.DataFrame(index=c.index)
    T["sdate"] = pd.to_datetime(c["sdate"])
    T["t_mid"] = T["sdate"] + pd.to_timedelta([MID[k[-1]] for k in T.index], unit="h")
    T["K_burnt"], T["K_charged"] = c["burnt_coke_kgthm"], c["charged_coke_kgthm"]
    T["hm_t"] = c["prod_th"] * 8
    return T


def main(path: str) -> None:
    S = tr.summary(load(path))
    cols = ["window", "n", "bias", "sd", "loa_lo", "loa_hi", "r", "ccc", "cb", "slope"]
    print(S["table"][cols].round(3).to_string(index=False))
    sp = S["split"]
    print(f"\nSpread: random +/-{sp['random_sd']:.1f} per shift, transit +/-{sp['transit_sd']:.1f} per shift, "
          f"floor +/-{sp['floor_sd']:.1f} kg/tHM SD")
    print("\nLag scan (burnt against charged coke L shifts earlier, deviations from a 7-day mean):")
    print(S["scan"].round(3).to_string(index=False))
    print(f"Peak by interpolation: {S['peak_lag_h']:.1f} h")
    if S["closure"]:
        c = S["closure"]
        print(f"\nClosure over {c['n']} shifts: raw burnt / charged = {c['ratio']:.4f} ({c['burnt_t']:,.0f} / "
              f"{c['charged_t']:,.0f} t)")


if __name__ == "__main__":
    main(sys.argv[1])
