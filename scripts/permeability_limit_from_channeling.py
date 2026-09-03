"""Find the k-value ceiling by correlating it with the channeling score.

Run:  python scripts/permeability_limit_from_channeling.py

WHY THIS EXISTS.

Layer 2 currently has nothing pushing back on blast temperature and oxygen.
Both coefficients are negative and monotone, so the optimiser always runs to the
move limit. A permeability ceiling would give it a real trade-off - and a real
safety guard - but only if we know what k-value the furnace actually cannot
tolerate. That number is not in any config.

THE METHOD.

The plant already has a channeling detector (utils.anomaly_propensity) that
scores 0-1 from seven independent signals: uptake temperature and pressure
spread, stack skin spread, hot-blast pressure drop, top and bottom differential
pressure rise, ETA CO drop, and heat-load rise. Notably it does NOT use k-value,
so the two are independent measurements of related physics.

k-value is the inverse-permeability proxy already in the codebase:

    k = ((HB*1000 + 1033)^2 - (Top*1000 + 1033)^2) / BoshVol^1.7

Higher k means the burden is resisting gas flow more. So:

    take every 10-minute window over several months
    compute both k and the channeling score
    find the k at which the channeling score crosses the operator's
    tolerance of 0.6-0.7

That crossing is the ceiling, derived from the plant's own behaviour rather
than assumed.

WHAT WOULD INVALIDATE IT.

If k and the channeling score turn out to be uncorrelated, there is no ceiling
to find this way and the script says so rather than reporting a spurious
threshold from noise.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from utils.anomaly_propensity import Channeling, ChannelingConfig  # noqa: E402
from furnace_data.influx.online import fetch_online_df  # noqa: E402

DAYS = 120
WINDOW = "15 minutes"
# Cache the fetch: it takes minutes and the analysis below is re-run often.
CACHE = Path(
    "C:/Users/sairi/AppData/Local/Temp/claude/e--Personal-MarketResearch-EvonithSteel-BlastFurnaceProject-PythonBlastFurnace-evonith-webapp/ef13d38c-6ac2-4dd2-964a-111b3c164734/scratchpad/channeling_raw.pkl"
)
TOLERANCE_BAND = (0.6, 0.7)  # operator-stated acceptable channeling score
# The detector defaults to a 10-minute window, but the online data arrives every
# 15 minutes, so most 10-minute windows hold at most one sample. Measured on 120
# days: the score then has std 0.049 with 34% NaN and a p95 of 0.243 - it never
# even reaches the operator's 0.6-0.7 band, and correlates with nothing because
# it barely varies. A 1-hour window gives four samples per window, 1% NaN and a
# p95 of 1.17, which is the regime the operator's tolerance refers to.
SCORING_WINDOW = "1h"


def fetch(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and CACHE.exists():
        print(f"  using cached fetch: {CACHE.name}")
        return pd.read_pickle(CACHE)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)
    frames = []
    # Fetched separately: a combined request across all three has intermittently
    # returned an auth error, and one measurement failing should not lose the rest.
    for measurement in ("process_params", "temperature_profile", "heatload_delta_t"):
        try:
            frames.append(
                fetch_online_df(
                    selected_measurements=[measurement],
                    time_range="last 1 week",
                    request_type="windowed-average",
                    window_by=WINDOW,
                    start_time_override=start,
                    end_time_override=end,
                    column_naming="field",
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN: {measurement} unavailable: {str(exc)[:90]}")
    if not frames:
        sys.exit("no online data available")
    out = pd.concat(frames, axis=1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(CACHE)
    return out


def add_k_value(df: pd.DataFrame) -> pd.DataFrame:
    """k-value and bosh volume, using the formulas already in the codebase."""

    out = df.copy()
    num = lambda c: pd.to_numeric(out.get(c), errors="coerce")  # noqa: E731

    wind = num("hot_blast_vol_nm3h")
    oxygen = num("oxygen_flow")
    steam = num("steam_injection").fillna(0.0)
    hb_p = num("hot_blast_press")
    top_p = num("top_press_avg")
    if top_p is None or top_p.isna().all():
        cols = [c for c in ("top_press_1", "top_press_2", "top_press_3",
                            "top_press_4") if c in out.columns]
        top_p = out[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # bosh_vol_from_formula, inlined so the whole series is vectorised.
    cbv = wind
    effective_o2 = ((cbv - oxygen) * 0.208 + oxygen) / wind * 100.0
    co_h2_n2 = (
        2.0 * (cbv * effective_o2 / 100.0)
        + (20.0 * (wind - oxygen)) / 18000.0
        + (steam * 1000.0) / 18000.0
    )
    out["bosh_vol"] = co_h2_n2 / 60.0
    out["k_value"] = (
        (hb_p * 1000.0 + 1033.0) ** 2 - (top_p * 1000.0 + 1033.0) ** 2
    ) / out["bosh_vol"] ** 1.7
    return out


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> None:
    print(f"fetching {DAYS} days at {WINDOW} resolution...")
    raw = fetch()
    raw = raw.loc[:, ~raw.columns.duplicated()]
    print(f"  rows {len(raw)}, columns {len(raw.columns)}")

    print("scoring channeling...")
    scores = Channeling(
        ChannelingConfig(window=SCORING_WINDOW, step=SCORING_WINDOW)
    ).score_timeseries(raw)
    scores = scores.set_index("window_end") if "window_end" in scores else scores
    print(f"  windows scored: {len(scores)}")

    keyed = add_k_value(raw)
    # The detector steps on its own grid, which does not line up with the
    # sampling grid, so an exact index join yields nothing. Pair each scored
    # window with the nearest sample within half a step.
    left = scores[["channeling_score"]].sort_index()
    right = keyed[["k_value", "bosh_vol"]].sort_index()
    for frame in (left, right):
        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    joined = pd.merge_asof(
        left, right, left_index=True, right_index=True,
        direction="nearest", tolerance=pd.Timedelta("30min"),
    ).replace([np.inf, -np.inf], np.nan).dropna(subset=["channeling_score", "k_value"])
    # k is a ratio of large squares; drop the arithmetic blow-ups.
    lo, hi = joined["k_value"].quantile([0.005, 0.995])
    joined = joined[joined["k_value"].between(lo, hi) & (joined["k_value"] > 0)]
    print(f"  usable paired windows: {len(joined)}")
    if len(joined) < 500:
        sys.exit("too few paired windows to draw a conclusion")

    banner("1. DO k AND CHANNELING MOVE TOGETHER?")
    pearson = joined["k_value"].corr(joined["channeling_score"])
    spearman = joined["k_value"].corr(joined["channeling_score"], method="spearman")
    print(f"  pearson  r = {pearson:+.3f}")
    print(f"  spearman r = {spearman:+.3f}   (rank, robust to the k tail)")
    if abs(spearman) < 0.15:
        print("\n  TOO WEAK. k and the channeling score are effectively unrelated")
        print("  in this record, so no ceiling can be derived this way. Reporting")
        print("  a threshold from this would be reading noise.")
        return

    banner("2. CHANNELING SCORE BY k DECILE")
    d = joined.copy()
    d["k_decile"] = pd.qcut(d["k_value"], 10, labels=False, duplicates="drop")
    table = d.groupby("k_decile").agg(
        k_mean=("k_value", "mean"),
        k_lo=("k_value", "min"),
        k_hi=("k_value", "max"),
        channeling_mean=("channeling_score", "mean"),
        channeling_p90=("channeling_score", lambda s: s.quantile(0.90)),
        n=("channeling_score", "size"),
    )
    print(table.to_string(float_format=lambda v: f"{v:10.3f}"))

    banner("3. THE CEILING")
    print(f"  Operator tolerance: channeling score {TOLERANCE_BAND[0]}-{TOLERANCE_BAND[1]}")
    for target in TOLERANCE_BAND:
        # Highest k whose decile mean still sits under the tolerance.
        under = table[table["channeling_mean"] <= target]
        if under.empty:
            print(f"  score {target}: every k decile already exceeds it")
            continue
        k_ceiling = float(under["k_hi"].max())
        share = float((joined["k_value"] > k_ceiling).mean())
        print(
            f"  score {target}: k ceiling ~ {k_ceiling:,.0f}   "
            f"({share:.1%} of observed windows sit above this)"
        )

    banner("4. WHAT DRIVES k, FOR THE OPTIMISER")
    print("  Layer 2 moves blast volume and oxygen, so it needs to know how those")
    print("  translate into k before a ceiling can constrain it.")
    for col, label in (
        ("hot_blast_vol_nm3h", "blast volume"),
        ("oxygen_flow", "oxygen flow"),
        ("hot_blast_press", "hot blast pressure"),
        ("top_press_avg", "top pressure"),
        ("bosh_vol", "bosh volume"),
    ):
        if col in keyed.columns:
            series = pd.to_numeric(keyed[col], errors="coerce")
            aligned = joined.join(series.rename("driver"), how="inner").dropna(
                subset=["driver"]
            )
            if len(aligned) > 100:
                print(f"  corr(k, {label:20s}) = "
                      f"{aligned['k_value'].corr(aligned['driver']):+.3f}")


if __name__ == "__main__":
    main()
