"""Synthetic plant data for the FurnaceMind plant skills (``agents.furnacemind.plant_skills``).

``FakePlantData`` answers every read of ``PlantData`` from seeded noise around BF-02's usual values, with the last 8 h
moved on purpose: fuel and coke up, heat load of R9 Q3 doubled, the lower-stack sector B 40 °C hotter, stave R9 23/24
+6 °C, the hot-blast pressure swinging; one stuck PCI reading of 428 kg/tHM per frame (dropped by the ``valid`` range).
``FailingPlantData`` answers every read with an error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agents.furnacemind.plant_skills import PlantData

NOW = pd.Timestamp("2026-11-13 21:00")

BASE: dict[str, tuple[float, float]] = {
    "fuel_rate": (560.0, 3.0),
    "coke_rate": (297.0, 2.0),
    "nut_coke_rate": (70.0, 1.0),
    "coal_rate_actual_value": (195.0, 3.0),
    "production_per_hour": (92.5, 1.5),
    "body_etaco": (42.1, 0.4),
    "body_raft": (2210.0, 10.0),
    "body_perm": (1290.0, 12.0),
    "body_dp_total": (1.41, 0.01),
    "body_dp_bottom": (1.10, 0.01),
    "body_dp_top": (0.31, 0.01),
    "hot_blast_vol_nm3h": (109000.0, 800.0),
    "hot_blast_temp": (1193.0, 3.0),
    "hot_blast_press": (2.81, 0.01),
    "oxygen_enrichment_pct": (4.2, 0.1),
    "steam_injection": (0.0, 0.0),
    "top_press_avg": (1.398, 0.002),
    "top_temp_avg": (208.0, 10.0),
    "co_pct": (24.8, 0.3),
    "co2_pct": (18.0, 0.3),
    "h2_pct": (4.0, 0.15),
    "charges_per_hour": (6.25, 0.05),
    "runner_temp_pci_taphole": (405.0, 3.0),
    "stock_rod1_pos": (1.05, 0.05),
    "stock_rod2_pos": (0.92, 0.05),
    "stock_rod_radar_level": (1.0, 0.1),
    **{f"temp_18660_{s}": (v, 4.0) for s, v in zip("abcd", (55, 47, 40, 34), strict=False)},
    **{f"temp_15162_{s}": (v, 4.0) for s, v in zip("abcd", (45, 60, 65, 80), strict=False)},
    **{f"temp_12975_{s}": (v, 4.0) for s, v in zip("abcd", (60, 80, 82, 85), strict=False)},
    **{f"top_temp_{i}": (v, 8.0) for i, v in zip(range(1, 5), (204, 206, 190, 225), strict=False)},
    **{f"heat_load_r{r}_q{q}": (0.35, 0.04) for r in range(6, 11) for q in range(1, 5)},
    **{f"heat_load_row_{r}": (1.4, 0.1) for r in range(6, 11)},
    **{f"heat_load_row6_10_q{q}": (1.5, 0.1) for q in range(1, 5)},
    **{f"delta_t_r{r}_stave_{n}": (1.8, 0.2) for r in (6, 7, 8) for n in (1, 2, 3)},
    "delta_t_r9_stave_23_24": (1.6, 0.2),
    "delta_t_r9_stave_13_14": (1.8, 0.2),
    "cw_bottom_flow_m3h": (189.0, 0.8),
    "cw_hearth_flow_m3h": (1074.0, 2.0),
    "cw_bottom_press_bar": (1.55, 0.005),
    "cw_hearth_press_bar": (2.70, 0.01),
    "cw_mains_temp_c": (32.5, 0.8),
}

#: Moves of the last 8 h (absolute, or ``x`` a factor).
MOVES: dict[str, float | tuple[str, float]] = {
    "fuel_rate": 9.0,
    "coke_rate": 8.0,
    "heat_load_r9_q3": ("x", 2.0),
    "temp_15162_b": 40.0,
    "delta_t_r9_stave_23_24": 6.0,
    "cw_bottom_flow_m3h": -1.5,  # a steady flow 0.8 % lower: under the materiality floor
}


def _frame(start: pd.Timestamp, end: pd.Timestamp, freq: str, seed: int) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq=freq, inclusive="left")
    rng = np.random.default_rng(seed)
    data = {k: rng.normal(mu, sd, len(idx)) if sd else np.full(len(idx), mu) for k, (mu, sd) in BASE.items()}
    df = pd.DataFrame(data, index=idx)
    recent = df.index >= NOW - pd.Timedelta(hours=8)
    for key, move in MOVES.items():
        if isinstance(move, tuple):
            df.loc[recent, key] *= move[1]
        else:
            df.loc[recent, key] += move
    # The hot-blast pressure swings in the last 8 h.
    df.loc[recent, "hot_blast_press"] += 0.04 * np.sin(np.arange(int(recent.sum())))
    if len(df):
        df.iloc[len(df) // 2, df.columns.get_loc("coal_rate_actual_value")] = 428.0  # a stuck PCI reading
    return df


class FakePlantData(PlantData):
    """Every read of the skills from synthetic data; ``calls`` records them."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def now(self) -> pd.Timestamp:
        return NOW

    def online(self, start, end, groups, window):
        self.calls.append(("online", tuple(groups), window))
        freq = "15min" if str(window).startswith("15") else "1h"
        return (
            _frame(pd.Timestamp(start), pd.Timestamp(end), freq, len(self.calls)),
            f"online_fake_{len(self.calls)}",
            None,
        )

    def hot_metal(self, start, end):
        self.calls.append(("hot_metal",))
        idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="2h", inclusive="left")
        rng = np.random.default_rng(7)
        hm = pd.DataFrame(
            {
                "hmt": rng.normal(1480, 12, len(idx)),
                "chem_pct_si": np.clip(rng.normal(0.45, 0.08, len(idx)), 0.1, None),
                "chem_pct_s": np.clip(rng.normal(0.03, 0.005, len(idx)), 0.005, None),
                "slag_basicity": rng.normal(1.08, 0.03, len(idx)),
                "slag_t_basicity": rng.normal(1.30, 0.03, len(idx)),
                "slag_pct_al2o3": rng.normal(20.1, 0.4, len(idx)),
                "slag_pct_mgo": rng.normal(6.9, 0.15, len(idx)),
                "slag_pct_feo": np.clip(rng.normal(0.35, 0.06, len(idx)), 0.05, None),
                "slag_pct_s": rng.normal(0.37, 0.02, len(idx)),
            },
            index=idx,
        )
        late = hm.index >= NOW - pd.Timedelta(hours=10)
        hm.loc[late, "hmt"] -= 45.0  # the last taps run cold
        hm.loc[late, "chem_pct_si"] += 0.3
        return hm, None

    def hmt_taps(self, hm):
        return hm["hmt"] if "hmt" in hm else pd.Series(dtype=float)

    def table(self, name, start, end):
        self.calls.append(("table", name))
        if name.endswith("charge_data"):
            idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="10min", inclusive="left")
            rng = np.random.default_rng(11)
            df = pd.DataFrame(
                {
                    "sinter_3_mt": rng.normal(16.0, 0.3, len(idx)),
                    "ore_2_mt": rng.normal(6.0, 0.3, len(idx)),
                    "pellet_1_mt": rng.normal(4.0, 0.2, len(idx)),
                    "coke_1_mt": rng.normal(4.7, 0.1, len(idx)),
                    "nut_coke_1_mt": rng.normal(0.88, 0.02, len(idx)),
                    "flux_1_mt": rng.normal(0.1, 0.01, len(idx)),
                },
                index=idx,
            )
            recent = df.index >= NOW - pd.Timedelta(hours=24)
            df.loc[recent, "ore_2_mt"] += 1.5  # more lump ore in the last day
            return df, None
        if name.endswith("sinter_chemistry"):
            idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="4h", inclusive="left")
            rng = np.random.default_rng(13)
            return pd.DataFrame(
                {
                    "fe_t": rng.normal(53.9, 0.3, len(idx)),
                    "feo": rng.normal(8.9, 0.2, len(idx)),
                    "sio2": rng.normal(5.5, 0.15, len(idx)),
                    "al2o3": rng.normal(2.5, 0.08, len(idx)),
                    "cao": rng.normal(11.8, 0.3, len(idx)),
                    "mgo": rng.normal(2.25, 0.06, len(idx)),
                    "basicity": rng.normal(2.12, 0.04, len(idx)),
                },
                index=idx,
            ), None
        return pd.DataFrame(), "no such table"


class FailingPlantData(PlantData):
    """Every read fails: the skills must say what is missing and never raise."""

    def now(self) -> pd.Timestamp:
        return NOW

    def online(self, start, end, groups, window):
        return pd.DataFrame(), None, "Fetch Error: InfluxDB unreachable"

    def hot_metal(self, start, end):
        return pd.DataFrame(), "OperationalError: Neon unreachable"

    def hmt_taps(self, hm):
        return pd.Series(dtype=float)

    def table(self, name, start, end):
        return pd.DataFrame(), "OperationalError: Neon unreachable"
